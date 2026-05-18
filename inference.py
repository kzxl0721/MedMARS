# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
import Config
from tqdm import tqdm
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from nets.MedMARS import MedMARS
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile
import pandas as pd
import torch.multiprocessing as mp
from nets.MedMARS import MultiTaskLoss
from surface_metrics import compute_surface_dice, compute_hd95  
mp.set_start_method('spawn', force=True)


def logger_config(log_path=None):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def calculate_batch_metrics(pred_batch, mask_batch):
    # 二值化处理
    pred_batch[pred_batch >= 0.5] = 1
    pred_batch[pred_batch < 0.5] = 0
    mask_batch[mask_batch > 0] = 1
    mask_batch[mask_batch <= 0] = 0
    
    pred_splits = torch.unbind(pred_batch, dim=0)
    mask_splits = torch.unbind(mask_batch, dim=0)
    
    # 计算每个样本的dice和iou
    dice_scores = []
    iou_scores = []
    
    for p, m in zip(pred_splits, mask_splits):
        # 计算交集和并集
        intersection = torch.sum(p * m)
        sum_pred = torch.sum(p)
        sum_mask = torch.sum(m)
        
        # 计算 Dice 系数
        dice = (2. * intersection + 1e-6) / (sum_pred + sum_mask + 1e-6)
        dice_scores.append(dice.item())
        
        # 计算 IoU
        union = sum_pred + sum_mask - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_scores.append(iou.item())
    
    pred_flat = pred_batch.view(-1)
    mask_flat = mask_batch.view(-1)
    global_intersection = torch.sum(pred_flat * mask_flat)
    global_dice = (2. * global_intersection + 1e-6) / (pred_flat.sum() + mask_flat.sum() + 1e-6)
    global_union = pred_flat.sum() + mask_flat.sum() - global_intersection
    global_iou = (global_intersection + 1e-6) / (global_union + 1e-6)
    
    mean_dice = sum(dice_scores) / len(dice_scores)
    mean_iou = sum(iou_scores) / len(iou_scores)
    
    return dice_scores, iou_scores, mean_dice, mean_iou, global_dice.item(), global_iou.item()


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=1, model_type='', tensorboard=True):

    test_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, test_tf, image_size=config.img_size)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=1, 
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)
    
    config_vit = config.get_CTranS_config()
    model = MedMARS(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=config.img_size)
    pretrained_UNet_model_path = "./Test_session_05.18_16h04/models/best_model-MedMARS.pth.tar"


    pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
    pretrained_UNet = pretrained_UNet['state_dict']
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    logger.info('Load successful!')
    
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    criterion = MultiTaskLoss(seg_weight=1.0, contrastive_weight=0.1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) + list(criterion.parameters())), lr=lr)
    
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    
    a_preds = []
    a_masks = []
    total_dice_sum = 0
    total_iou_sum = 0
    total_nsd_sum = 0    
    total_hd95_sum = 0   
    hd95_count = 0       
    batch_count = 0
    
    for epoch in range(1):  # loop over the dataset multiple times
        logger.info('\n============== Testing =============='.format(epoch + 1, config.epochs + 1))
        with torch.no_grad():
            model.eval()
            end = time.time()
            
            for i, (sampled_batch, names) in enumerate(tqdm(test_loader, desc=f"Testing", total=len(test_loader)), 1):
                images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
                images, masks, text = images.cuda(), masks.cuda(), text.cuda()
                keyword_positions = sampled_batch['keyword_positions']
                keyword_positions = keyword_positions.cuda()
                
                outputs = model(images, text, masks, keyword_positions)
                preds = outputs.squeeze(1)  # [1, H, W]
                
                a_preds.append(preds)  # [1, H, W]
                a_masks.append(masks.float())  # [1, H, W]
                

                if len(a_preds) == 24 or i == len(test_loader):

                    batch_preds = torch.cat(a_preds, dim=0)  # [N, H, W]
                    batch_masks = torch.cat(a_masks, dim=0)  # [N, H, W]
                    
                    dice_scores, iou_scores, mean_dice, mean_iou, global_dice, global_iou = calculate_batch_metrics(
                        batch_preds.clone(), batch_masks.clone())
                    
                    batch_nsd = compute_surface_dice(
                        batch_preds.clone(), batch_masks.clone(),
                        threshold=0.5, tolerance=2.0,
                        spacing_mm=(1.0, 1.0)
                    )
                    batch_hd95 = compute_hd95(
                        batch_preds.clone(), batch_masks.clone(),
                        threshold=0.5
                    )

                    total_dice_sum += global_dice
                    total_iou_sum += global_iou
                    total_nsd_sum += batch_nsd                        
                    if not np.isnan(batch_hd95):                      
                        total_hd95_sum += batch_hd95                  
                        hd95_count += 1                               
                    batch_count += 1
                    
                    a_preds = []
                    a_masks = []
                
            if batch_count > 0:
                avg_dice = total_dice_sum / batch_count
                avg_iou = total_iou_sum / batch_count
                avg_nsd = total_nsd_sum / batch_count                              
                avg_hd95 = total_hd95_sum / hd95_count if hd95_count > 0 else float('nan')  
                print(f"\n=== Final Results ===")
                print(f"Average Dice:         {avg_dice:.4f}")
                print(f"Average IoU:          {avg_iou:.4f}")
                print(f"Average Surface Dice: {avg_nsd:.4f}")         
                print(f"Average HD95:         {avg_hd95:.4f}")     
            else:
                print("No batches were processed!")

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(batch_size=1, model_type=config.model_name, tensorboard=True)