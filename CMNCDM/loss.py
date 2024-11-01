# coding: utf-8

import torch
import torch.nn as nn

class PairSCELoss(nn.Module):
    def __init__(self):
        super(PairSCELoss, self).__init__()

    def forward(self, pred_theta, pred_theta_pair, id , id_pair , n , *args):
        """
        still use theta and theta_pair to calculate the loss
        add two parameters: id , id_pair to judge the ability of two people
        if id > id_pair,  that means the ability of id is less than id_pair
        so the loss should be larger
        but we should make sure that they are in the same group
        that means id/n == id_pair/n
        """
        # print("pred_theta.shape")
        # print(pred_theta.shape)
        if id/n != id_pair/n:
            return torch.zeros_like(pred_theta), 0
        
        # if id > id_pair, pos = 1, otherwise pos = -1 
        if id > id_pair:
            pos = -1.0
        else:
            pos = 1.0

        if pred_theta.dim() == 1:
            pred_theta = pred_theta.unsqueeze(-1)
        if pred_theta_pair.dim() == 1:
            pred_theta_pair = pred_theta_pair.unsqueeze(-1)
        # print(pred_theta.shape)
            
        pred_theta = pred_theta.mean(dim=1)
        pred_theta_pair = pred_theta_pair.mean(dim=1)
        
        if pred_theta.dim() == 1:
            pred_theta = pred_theta.unsqueeze(-1)
        if pred_theta_pair.dim() == 1:
            pred_theta_pair = pred_theta_pair.unsqueeze(-1)

        loss = torch.where( # if the prediction is correct, the loss is 0, otherwise the loss is the square of the difference
            ((pos == 1.0) & (pred_theta > pred_theta_pair)) | ((pos == -1.0) & (pred_theta < pred_theta_pair)),
            torch.zeros_like(pred_theta),
            (pred_theta - pred_theta_pair) ** 2
        )
        
        
        count = torch.sum(torch.where( # count is the number of correct predictions
            ((pos == 1.0) & (pred_theta > pred_theta_pair)) | ((pos == -1.0) & (pred_theta < pred_theta_pair)),
            torch.zeros_like(pred_theta),
            torch.ones_like(pred_theta)
        )).item()
        # print(pos.shape)
        # print(pos.shape)
        # print(pred_theta.shape)
        # print(pred_theta_pair.shape)
        #print(loss.shape)
        return loss.sum(), count  # you can change this to .sum() if you want a total loss instead of average



class HarmonicLoss(object):
    def __init__(self, zeta=0.5):
        self.zeta = zeta

    def __call__(self, score_loss, theta_loss, *args, **kwargs):
        # return ((1 - self.zeta) * score_loss + self.zeta * theta_loss)
        return (score_loss + self.zeta * theta_loss)