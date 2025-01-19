# coding: utf-8
# 2021/3/23 @ tongshiwei

import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from loss import FairnessLoss
import random

class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

    def forward(self, user_id, item_id, fairness=False):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)
        if fairness:
            return user
        return torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)


class MCD(CDM):
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MCD, self).__init__()
        self.mf_net = MFNet(user_num, item_num, latent_dim)
        self.fairness_lambda = 0.3

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.mf_net = self.mf_net.to(device)
        bce_loss = nn.BCELoss()
        self.fairness_loss = FairnessLoss()

        trainer = torch.optim.Adam(self.mf_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            fairness_losses = []
            bce_losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response, fairness_id, group_id, groupindex, group_size = batch_data
                item_id = item_id.to(device)
                response = response.to(device)
                fairness_id = fairness_id.to(device)

                predicted_response: torch.Tensor = self.mf_net(fairness_id, item_id, fairness=False)
                response: torch.Tensor = response.to(device)
                
                score_loss = bce_loss(predicted_response, response)
                #print(score_loss)

                #计算公平性损失
                unique_groups = torch.unique(group_id)
                
                group_fairness_losses = []
                #print("groups: ", unique_groups)

                for gid in unique_groups:
                    group_mask = (group_id == gid)

                    if gid == 0:
                        # group_id = 0 的组跳过公平性损失
                        continue
                    
                    # 获取当前组的起始索引和组大小
                    group_start = groupindex[group_mask][0].item()  # 当前组的起始索引
                    group_sz = group_size[group_mask][0].item()     # 当前组的大小

                    if group_sz == 1 or group_sz == 2:
                        # 单个用户的组跳过公平性损失
                        continue

                    # 提取当前组的 fairness_id
                    group_users = [i for i in range(group_start, group_start + group_sz)]  # 组内用户的 fairness_id
                    #print("group_users: ", group_users)
                    
                    group_users = torch.tensor(group_users, dtype=torch.int64).to(device)
                    latent_dim = self.mf_net.latent_dim

                    # 获取当前组的 theta 值
                    theta_group_full = self.mf_net(group_users, item_id, fairness=True)  # shape: [group_sz, latent_dim]
                    theta_group_selected = theta_group_full[0:group_sz, 0:latent_dim]

                    # 对选定的维度进行平均，得到每个用户的 theta 值
                    theta_group = theta_group_selected.mean(dim=1)  # shape: [group_sz]

                    # 计算公平性损失
                    predictions_reshaped = theta_group.view(1, -1)  # 模型预测 theta
                    targets_reshaped = group_users.view(1, -1)  # 目标 fairness_id
                    fairness_loss_val = self.fairness_loss(predictions_reshaped, targets_reshaped)
                    group_fairness_losses.append(fairness_loss_val)

                # 合并 BCE 损失与公平性损失
                if group_fairness_losses:
                    #print("group_fairness_losses: ", group_fairness_losses)
                    total_fairness_loss =  torch.mean(torch.stack(group_fairness_losses))
                    #print("total_fairness_loss: ", total_fairness_loss)
                    loss = (1 - self.fairness_lambda) * score_loss + self.fairness_lambda * total_fairness_loss
                    #print("loss: ", loss)
                    fairness_losses.append(total_fairness_loss.item())
                    #print("fairness_loss: ", fairness_losses)
                else:
                    loss = score_loss

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                bce_losses.append(score_loss.mean().item())
                
            #print("fairnessloss: ", (fairness_losses))
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(bce_losses))))
            print("fairnessloss: %.6f" % (float(np.mean(fairness_losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d]  auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.mf_net = self.mf_net.to(device)
        self.mf_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response, fairness_id, group_id, groupindex, group_size = batch_data
            
            item_id: torch.Tensor = item_id.to(device)
            fairness_id: torch.Tensor = fairness_id.to(device)
            pred: torch.Tensor = self.mf_net(fairness_id, item_id, fairness=False)
            response: torch.Tensor = response.to(device)
            loss = loss_function(pred, response)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))

        self.mf_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.mf_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.mf_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

    def extract_ability_parameters(self, test_data, filepath, device="cpu"):
        self.mf_net = self.mf_net.to(device)
        self.mf_net.eval()  # Switch to evaluation mode

        abilities = []
        processed_user_ids = set()  # To track processed (group_id, user_id)

        for batch_data in tqdm(test_data, "Extracting abilities"):
            group_id,  user_id, item_id, response, fairness_id = batch_data
            user_id = user_id.to(device)
            fairness_id = fairness_id.to(device)
            """
            torch.tensor(groupid, dtype=torch.int64),
            torch.tensor(x, dtype=torch.int64),
            torch.tensor(y, dtype=torch.int64),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(fairnessid, dtype=torch.int64)
            """
            # Retrieve the ability (θ) parameter for the user
            theta = self.mf_net(fairness_id, item_id, fairness=True)
            theta_mean: torch.Tensor = theta.mean(dim=-1)

            # Add group_id, fairness_id, user_id, and corresponding θ value to the list
            for i, user in enumerate(fairness_id.cpu().numpy()):
                if (group_id[i].item(), user) not in processed_user_ids:
                    abilities.append([
                        int(group_id[i]),
                        int(fairness_id[i]),
                        int(user),
                        float(theta_mean[i].item())
                    ])
                    processed_user_ids.add((group_id[i].item(), user))  # Mark as processed

        # Save abilities to a CSV file with group_id and fairness_id
        df_abilities = pd.DataFrame(abilities, columns=["group_id", "fairness_id", "user_id", "theta"])
        df_abilities.sort_values(by=["group_id", "fairness_id"], inplace=True)  # Sort by group_id and fairness_id
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")

        self.mf_net.train()  # Switch back to training mode
