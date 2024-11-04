# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class IRTPairSCELoss(nn.Module):
    def __init__(self):
        super(IRTPairSCELoss, self).__init__()

    def forward(self, pred_theta, pred_theta_pair, id, id_pair, n, *args):
        if id/n != id_pair/n:
            return torch.zeros_like(pred_theta), 0
        
        if id > id_pair:
            pos = 1.0
        else:
            pos = -1.0

        if pred_theta.dim() == 1:
            pred_theta = pred_theta.unsqueeze(-1)
        if pred_theta_pair.dim() == 1:
            pred_theta_pair = pred_theta_pair.unsqueeze(-1)
            
        pred_theta = pred_theta.mean(dim=1)
        pred_theta_pair = pred_theta_pair.mean(dim=1)
        
        if pred_theta.dim() == 1:
            pred_theta = pred_theta.unsqueeze(-1)
        if pred_theta_pair.dim() == 1:
            pred_theta_pair = pred_theta_pair.unsqueeze(-1)

        loss = torch.where(
            ((pos == 1.0) & (pred_theta > pred_theta_pair)) | ((pos == -1.0) & (pred_theta < pred_theta_pair)),
            torch.zeros_like(pred_theta),
            (pred_theta - pred_theta_pair) ** 2
        )
        
        count = torch.sum(torch.where(
            ((pos == 1.0) & (pred_theta > pred_theta_pair)) | ((pos == -1.0) & (pred_theta < pred_theta_pair)),
            torch.zeros_like(pred_theta),
            torch.ones_like(pred_theta)
        )).item()
        
        return loss.sum(), count

class IRTLoss(object):
    def __init__(self, zeta=0.5):
        self.zeta = zeta
        self.pair_loss = IRTPairSCELoss()
        
    def __call__(self, pred_scores, true_scores, pred_theta, pred_theta_pair, id, id_pair, n):
        # 预测损失（使用二元交叉熵）
        score_loss = F.binary_cross_entropy_with_logits(pred_scores, true_scores)
        
        # 配对损失
        theta_loss, count = self.pair_loss(pred_theta, pred_theta_pair, id, id_pair, n)
        
        # 组合损失
        total_loss = score_loss + self.zeta * theta_loss
        
        return total_loss, score_loss, theta_loss, count
