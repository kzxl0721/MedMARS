# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(num_features=out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class Mask_guided_Image_Enhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visual_features, gt_masks):
        """
        返回: [B, C, H, W] - 经过attention加权的特征
        """
        B, C, H, W = visual_features.shape
        
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)
        
        mask_resized = F.interpolate(
            gt_masks.float(), 
            size=(H, W), 
            mode='bilinear',
            align_corners=False
        )
        
        # 生成attention权重
        concat_feat = torch.cat([visual_features, mask_resized], dim=1)
        attention_weights = self.attention(concat_feat)  # [B, C, H, W]
        
        # 应用attention
        enhanced = visual_features * attention_weights
        
        return enhanced

class Enhance_Text_Fusion(nn.Module):
    
    def __init__(self, text_dim=512, num_heads=8):
        super().__init__()
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        
        # 关键词特征提取
        self.keyword_encoder = nn.Sequential(
            nn.Linear(1, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim)
        )
        
        # Cross-Attention: keyword as query
        self.q_proj = nn.Linear(text_dim, text_dim)
        self.k_proj = nn.Linear(text_dim, text_dim)
        self.v_proj = nn.Linear(text_dim, text_dim)
        self.out_proj = nn.Linear(text_dim, text_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(text_dim)
        self.norm2 = nn.LayerNorm(text_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim * 4, text_dim)
        )
        
    def forward(self, text_features, keyword_positions):
        """
        Args:
            text_features: [B, 77, 512]
            keyword_positions: [B, 77]
        """
        B, N, C = text_features.shape
        
        # 编码关键词位置信息
        keyword_pos = keyword_positions.unsqueeze(-1).float()  # [B, 77, 1]
        keyword_features = self.keyword_encoder(keyword_pos)  # [B, 77, 512]
        
        # Cross-Attention: keyword queries attend to text
        q = self.q_proj(keyword_features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(text_features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(text_features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # 残差连接
        out = self.norm1(text_features + out)
        
        # FFN
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out


class ContrastiveLearningModule(nn.Module):
    """对比学习模块：文本关键词与图像区域对比学习"""
    
    def __init__(self, text_dim=512, visual_dim=512, proj_dim=256, temperature=0.07,
                focal_gamma=2.0, triplet_margin=0.2, alpha=0.6, beta=0.4):
        """
        Args:
            focal_gamma: Focal loss的gamma参数
            triplet_margin: Triplet loss的margin
            alpha, beta: combined模式下的损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.focal_gamma = focal_gamma
        self.triplet_margin = triplet_margin
        self.alpha = alpha
        self.beta = beta
        
        # 文本投影层
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # 视觉投影层
        self.visual_projector = nn.Sequential(
            nn.Conv2d(visual_dim, proj_dim, 1),
            nn.ReLU(),
            nn.Conv2d(proj_dim, proj_dim, 1),
            nn.BatchNorm2d(proj_dim)
        )
        
        self.mask_enhancement = Mask_guided_Image_Enhancement(proj_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def compute_focal_loss(self, text_proj, roi_features):
        """Focal对比损失"""
        batch_size = text_proj.shape[0]
        sim_matrix = torch.matmul(text_proj, roi_features.T) / self.temperature
        labels = torch.arange(batch_size, device=text_proj.device)
        
        # 计算概率和focal weight
        probs = F.softmax(sim_matrix, dim=1)
        correct_probs = probs[torch.arange(batch_size), labels]
        focal_weight = (1 - correct_probs) ** self.focal_gamma
        
        # 加权CE
        ce_loss = F.cross_entropy(sim_matrix, labels, reduction='none')
        loss = (focal_weight * ce_loss).mean()
        
        return loss, sim_matrix
    
    
    def forward(self, text_features, visual_features, gt_masks):
        """
        Args:
            text_features: [B, seq_len, text_dim] 或 [B, text_dim] 文本特征
            visual_features: [B, visual_dim, H, W] 视觉特征
            gt_masks: [B, 1, H, W] or [B, H, W] GT分割掩码
        
        Returns:
            contrastive_loss: 对比学习损失
            sim_matrix: 相似度矩阵 [B, B]
        """
        batch_size = text_features.shape[0]
        
        # 处理文本特征
        if text_features.dim() == 3:  # [B, seq_len, text_dim]
            text_features = text_features.mean(dim=1)  # [B, text_dim]
        
        text_proj = self.text_projector(text_features)  # [B, proj_dim]
        text_proj = F.normalize(text_proj, dim=-1)
        
        # 投影视觉特征
        visual_proj = self.visual_projector(visual_features)  # [B, proj_dim, H, W]
        
        # Mask增强
        enhanced_visual = self.mask_enhancement(visual_proj, gt_masks)  # [B, proj_dim, H, W]
        
        # ROI特征提取
        roi_features = self.global_pool(enhanced_visual).squeeze(-1).squeeze(-1)  # [B, proj_dim]
        roi_features = F.normalize(roi_features, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(text_proj, roi_features.T) / self.temperature  # [B, B]
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size, device=text_proj.device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels)

        return contrastive_loss, sim_matrix

class MedMARS(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False, 
                 enable_contrastive=True, contrastive_weight=0.1):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.enable_contrastive = enable_contrastive
        self.contrastive_weight = contrastive_weight
        
        in_channels = config.base_channel
        
        self.inc = ConvBatchNorm(n_channels, in_channels)
        
        # Vision Transformers
        self.downVit = VisionTransformer(config, vis, img_size=self.img_size, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=self.img_size//2, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=self.img_size//4, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=self.img_size//8, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=self.img_size, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=self.img_size//2, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=self.img_size//4, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=self.img_size//8, channel_num=512, patch_size=2, embed_dim=512)

        # U-Net components
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()
        
        # Reconstruction modules
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        
        # Text processing modules
        self.text_linear4 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.text_linear3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.text_linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.text_linear1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        if self.enable_contrastive:
            self.contrastive_module = ContrastiveLearningModule(
                text_dim=512, visual_dim=512, proj_dim=256
            )
            
        self.Enhance_Text_Fusion = Enhance_Text_Fusion(text_dim=512)
        
    def forward(self, x, text, gt_masks=None, keyword_positions=None):
        """
        Args:
            x: [B, 3, H, W] 输入图像
            text: [B, 512] 文本特征
            gt_masks: [B, 1, H, W] GT分割掩码（训练时使用）
            keyword_positions: [B, 77] 关键词位置掩码
        """
        x = x.float()
        batch_size = x.shape[0]
        
        # 文本处理流程
        text4 = self.text_linear4(text)
        text3 = self.text_linear3(text4)
        text2 = self.text_linear2(text3)
        text1 = self.text_linear1(text2)
        
        # U-Net前向传播
        x1 = self.inc(x)
        y1 = self.downVit(x1, x1, text1)
        
        x2 = self.down1(x1)
        y2 = self.downVit1(x2, y1, text2)
        
        x3 = self.down2(x2)
        y3 = self.downVit2(x3, y2, text3)
        
        x4 = self.down3(x3)
        y4 = self.downVit3(x4, y3, text4)
        x5 = self.down4(x4)
        
        # 上采样路径
        y4 = self.upVit3(y4, y4, text4, True)
        y3 = self.upVit2(y3, y4, text3, True)
        y2 = self.upVit1(y2, y3, text2, True)
        y1 = self.upVit(y1, y2, text1, True)
        
        # 特征重建
        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4
        
        # 解码器
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        # 输出预测
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)

        
        
        # 对比学习（仅在训练时且提供GT mask时） 
        contrastive_loss = None
        enhanced_text = self.Enhance_Text_Fusion(text, keyword_positions)
        if self.enable_contrastive and self.training and gt_masks is not None:
            # 使用最深层特征进行对比学习
            contrastive_loss, _ = self.contrastive_module(
                enhanced_text, x4, gt_masks
            )

        if self.training and contrastive_loss is not None:
            return logits, contrastive_loss
        else:
            return logits
    
    
# 训练损失函数
class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, seg_weight=1.0, contrastive_weight=0.1):
        super().__init__()
        self.seg_weight = seg_weight
        self.contrastive_weight = contrastive_weight
        self.log_contrastive_weight1 = nn.Parameter(torch.log(torch.tensor(contrastive_weight, dtype=torch.float32)))
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = self._dice_loss    


    def _dice_loss(self, pred, target, smooth=1.0):
        """Dice损失"""
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def _show_iou(self, inputs, targets, smooth=1.0):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        pred = inputs.view(-1)
        target = targets.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def _save_on_batch(self, inputs, targets, names, vis_path):
        for i in range(inputs.shape[0]):
            pred_tmp = inputs[i][0].cpu().detach().numpy()
            mask_tmp = targets[i].cpu().detach().numpy()
            pred_tmp[pred_tmp >= 0.5] = 255
            pred_tmp[pred_tmp < 0.5] = 0
            mask_tmp[mask_tmp > 0] = 255
            mask_tmp[mask_tmp <= 0] = 0
            
            cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
            cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)

    @property
    def contrastive_weight1(self):
        """获取实际的对比学习权重"""
        return torch.exp(self.log_contrastive_weight1)

    def forward(self, pred, target, contrastive_loss=None):
        # 分割损失
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        seg_loss = bce + dice
        total_loss = self.seg_weight * seg_loss
        
        # 对比学习损失
        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss
            
        return total_loss, {
            'total_loss': total_loss.item(),
            'seg_loss': seg_loss.item(),
            'contrastive_loss': contrastive_loss.item() if contrastive_loss is not None else 0.0,
        }