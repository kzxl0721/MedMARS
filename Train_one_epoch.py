# -*- coding: utf-8 -*-
import torch.optim
import os
import time
import torch.distributed as dist
from utils import *
import Config as config
import warnings
import numpy as np

from surface_metrics import compute_surface_dice, compute_hd95  

warnings.filterwarnings("ignore")


##################################################################################
#=================================================================================
#          打印函数
#=================================================================================
##################################################################################

def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger,
                  hd95=None, average_hd95=None,
                  surf_dice=None, average_surf_dice=None):
    """mode = 'Train' or 'Val'，只在主进程中打印。"""
    summary_str = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    if hd95 is not None and average_hd95 is not None:
        hd95_str     = '{:.2f}'.format(hd95)         if not np.isnan(hd95)         else 'nan'
        avg_hd95_str = '{:.2f}'.format(average_hd95) if not np.isnan(average_hd95) else 'nan'
        string += 'HD95:{} '.format(hd95_str)
        string += '(Avg {}) '.format(avg_hd95_str)
    if surf_dice is not None and average_surf_dice is not None:
        sd_str     = '{:.4f}'.format(surf_dice)         if not np.isnan(surf_dice)         else 'nan'
        avg_sd_str = '{:.4f}'.format(average_surf_dice) if not np.isnan(average_surf_dice) else 'nan'
        string += 'SurfDice:{} '.format(sd_str)
        string += '(Avg {}) '.format(avg_sd_str)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary_str += string
    logger.info(summary_str)


##################################################################################
#=================================================================================
#          DDP reduce 工具
#=================================================================================
##################################################################################

def reduce_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


##################################################################################
#=================================================================================
#          Train One Epoch for DDP
#=================================================================================
##################################################################################

def train_one_epoch(loader, model, criterion, optimizer, writer, epoch,
                    lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum  = 0.0, 0.0
    hd95_sum, surf_dice_sum = 0.0, 0.0
    hd95_valid_count, surf_dice_valid_count = 0, 0
    dices = []

    average_loss      = 0.0
    train_dice_avg    = 0.0
    average_hd95      = float('nan')
    average_surf_dice = float('nan')
    cur_hd95          = None
    cur_surf_dice     = None

    for i, (sampled_batch, names) in enumerate(loader, 1):
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images, masks, text = (sampled_batch['image'],
                                sampled_batch['label'],
                                sampled_batch['text'])
        images, masks, text = images.cuda(), masks.cuda(), text.cuda()
        keyword_positions = sampled_batch['keyword_positions'].cuda()

        # 前向传播
        outputs = model(images, text, masks, keyword_positions)
        if len(outputs) == 2:
            preds, contrastive_loss = outputs
            contrastive_loss = contrastive_loss.mean()
            preds = preds.squeeze(1)
            total_loss, loss_dict = criterion(preds, masks.float(), contrastive_loss)
        else:
            preds = outputs
            preds = preds.squeeze(1)
            total_loss, _ = criterion(preds, masks.float(), None)

        # 反向传播
        if model.training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 基础指标
        train_dice = criterion._show_dice(preds, masks.float())
        preds      = preds.unsqueeze(1)
        train_iou  = criterion._show_iou(preds, masks.float())

        if logging_mode == 'Val':
            batch_hd95      = compute_hd95(preds, masks.unsqueeze(1).float())
            batch_surf_dice = compute_surface_dice(
                pred=preds,
                target=masks.unsqueeze(1).float(),
                threshold=0.5,
                tolerance=2.0,
                spacing_mm=(1.0, 1.0)
            )

            cur_hd95      = batch_hd95
            cur_surf_dice = batch_surf_dice

            batch_size = len(images)
            if not np.isnan(batch_hd95):
                hd95_sum         += batch_size * batch_hd95
                hd95_valid_count += batch_size
            if not np.isnan(batch_surf_dice):
                surf_dice_sum         += batch_size * batch_surf_dice
                surf_dice_valid_count += batch_size
        else:
            cur_hd95      = None
            cur_surf_dice = None

        batch_time = time.time() - end

        # 可视化
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            os.makedirs(vis_path, exist_ok=True)
            criterion._save_on_batch(preds, masks.float(), names, vis_path)

        dices.append(train_dice)

        # 累计统计
        batch_size = len(images)
        time_sum += batch_size * batch_time
        loss_sum += batch_size * total_loss
        iou_sum  += batch_size * train_iou
        dice_sum += batch_size * train_dice

        total_samples = (config.batch_size * (i - 1) + batch_size
                         if i == len(loader) else i * config.batch_size)

        average_loss      = loss_sum / total_samples
        average_time      = time_sum / total_samples
        train_iou_average = iou_sum  / total_samples
        train_dice_avg    = dice_sum / total_samples

        average_hd95      = (hd95_sum      / hd95_valid_count
                             if hd95_valid_count      > 0 else float('nan'))
        average_surf_dice = (surf_dice_sum / surf_dice_valid_count
                             if surf_dice_valid_count > 0 else float('nan'))

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(
                epoch + 1, i, len(loader), total_loss, loss_name, batch_time,
                average_loss, average_time, train_iou, train_iou_average,
                train_dice, train_dice_avg, 0, 0, logging_mode,
                lr=min(g["lr"] for g in optimizer.param_groups),
                logger=logger,
                hd95=cur_hd95,
                average_hd95=average_hd95      if logging_mode == 'Val' else None,
                surf_dice=cur_surf_dice,
                average_surf_dice=average_surf_dice if logging_mode == 'Val' else None,
            )

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg