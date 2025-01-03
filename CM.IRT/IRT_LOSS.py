# coding: utf-8
import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM
import pandas as pd

# 将损失函数类移到同一文件中
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

class IRTNet(nn.Module):
    def __init__(self, student_n, exer_n):
        super(IRTNet, self).__init__()
        # IRT模型参数
        self.theta = nn.Embedding(student_n, 1)  # 学生能力参数
        self.a = nn.Embedding(exer_n, 1)  # 题目区分度
        self.b = nn.Embedding(exer_n, 1)  # 题目难度
        
        # 参数初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, user_id_pair=None):
        theta = self.theta(stu_id)
        if user_id_pair is not None:
            theta_pair = self.theta(user_id_pair)
        
        a = torch.sigmoid(self.a(exer_id)) * 2  # 将区分度限制在0-2范围
        b = self.b(exer_id)
        
        pred = torch.sigmoid(1.7 * a * (theta - b))
        
        if user_id_pair is not None:
            return pred.view(-1), theta, theta_pair
        return pred.view(-1)

class IRT(CDM):
    def __init__(self, student_n, exer_n, zeta=0.5, groupsize=11):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(student_n, exer_n)
        self.zeta = zeta
        self.groupsize = groupsize
        self.loss_function = IRTLoss(self.zeta)  # 初始化损失函数

    def train(self, train_data, test_data=None, epoch=10, device="cuda", lr=0.002, silence=False):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.train()
        
        optimizer = optim.Adam(self.irt_net.parameters(), lr=lr)
        
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            count = 0
            
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                origin_id, user_id, item_id, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                y = y.to(device)

                # 构建配对ID
                pair_id = []
                for i in range(user_id.size(0)):
                    group_pair = []
                    group_index = user_id[i].item() // self.groupsize
                    for j in range(self.groupsize):
                        group_pair.append(j + group_index * self.groupsize)
                    pair_id.append(group_pair)
                
                pair_id_tensor = torch.tensor(pair_id, device=device)
                loss_theta = 0
                loss_score = 0

                for i in range(len(pair_id)):
                    sample_indices = torch.randperm(len(pair_id[i]))[:len(pair_id[i]) // 3]
                    for j in sample_indices:
                        if i != pair_id[i][j]:
                            predicted_response, predicted_theta, predicted_theta_pair = self.irt_net(
                                user_id, item_id, pair_id_tensor[i][j]
                            )
                            
                            loss, loss_s, loss_t, count_t = self.loss_function(
                                predicted_response, y,
                                predicted_theta[i], predicted_theta_pair,
                                user_id[i].item(), pair_id[i][j], self.groupsize
                            )
                            
                            loss = loss.mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            epoch_losses.append(loss.item())
                            count += count_t

            print("[Epoch %d] average loss: %.6f, Count: %d" % (epoch_i, float(np.mean(epoch_losses)), count))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_true, y_pred = [], []
        
        for batch_data in tqdm(test_data, "Evaluating"):
            origin_id, user_id, item_id, _, y = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            
            pred = self.irt_net(user_id, item_id)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
            
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath, weights_only=True))
        logging.info("load parameters from %s" % filepath)

    def extract_user_abilities(self, test_data, device="cuda", filepath="v_ability_parameters.csv"):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        
        user_theta_map = {}
        
        for batch_data in test_data:
            origin_id, user_id, item_id, _, y = batch_data
            user_id = user_id.to(device)
            
            theta = self.irt_net.theta(user_id).detach().cpu().numpy()
            
            for uid, ability in zip(user_id.cpu().numpy(), theta):
                if uid in user_theta_map:
                    user_theta_map[uid] = (user_theta_map[uid] + ability[0]) / 2
                else:
                    user_theta_map[uid] = ability[0]
        
        df = pd.DataFrame(user_theta_map.items(), columns=['user_id', 'theta'])
        df.sort_values(by="user_id", inplace=True)
        df.to_csv(filepath, index=False)
        print(f"Student abilities (theta) saved to '{filepath}'")
