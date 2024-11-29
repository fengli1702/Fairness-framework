import logging
import numpy as np
import torch
from EduCDM import CDM
from loss import FairnessLoss
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))

irt3pl = irf

class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range, a_range, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)

        self.value_range = value_range
        self.a_range = a_range

    def forward(self, user, item, fairness):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)

        c = torch.sigmoid(c)

        if fairness:
            return torch.sigmoid(theta)

        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)

        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)

        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan!')

        return self.irf(theta, a, b, c, **self.irf_kwargs) 

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs) 

class IRT(CDM):
    def __init__(self, user_num, item_num, value_range=None, a_range=None, use_fairness=True, fairness_lambda=0.6 , group_size = 9):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range)
        self.use_fairness = use_fairness
        self.groupsize = group_size
        self.fairness_lambda = fairness_lambda
        if use_fairness:
            self.fairness_loss = FairnessLoss()

    def train(self, train_data, test_data=None, *, epoch: int, device="cuda", lr=0.001):
        self.irt_net = self.irt_net.to(device)
        bce_loss = nn.BCELoss()
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            bce_losses = []
            fairness_losses = []

            for batch_data in tqdm(train_data, "Epoch %s" % e):
                # 解包数据
                user_id, item_id, response, fairness_id, group_id, groupindex, group_size = batch_data

                # 移动数据到设备
                item_id = item_id.to(device)
                response = response.to(device)
                fairness_id = fairness_id.to(device)
                group_id = group_id.to(device)
                groupindex = groupindex.to(device)
                group_size = group_size.to(device)

                # 模型预测，使用 fairness_id 作为输入
                predicted_response = self.irt_net(fairness_id, item_id, fairness=False)

                # 计算 BCE 损失
                bce_loss_val = bce_loss(predicted_response, response)

                if self.use_fairness:
                    group_fairness_losses = []

                    # 遍历每个 group，计算公平性损失
                    unique_groups = torch.unique(group_id)
                    for gid in unique_groups:
                        group_mask = (group_id == gid)

                        if gid == 0:
                            # group_id = 0 的组跳过公平性损失
                            continue
                        
                        # 获取当前组的起始索引和组大小
                        group_start = groupindex[group_mask][0].item()  # 当前组的起始索引
                        group_sz = group_size[group_mask][0].item()     # 当前组的大小

                        # 提取当前组的 fairness_id
                        group_users = [i for i in range(group_start,group_start+group_sz)]  # 组内用户的 fairness_id
                        
                        group_users = torch.tensor(group_users, dtype=torch.int64).to(device)
                        # 获取当前组的 theta 值
                        theta_group = self.irt_net(group_users, item_id[group_mask], fairness=True)
                        
                        # 计算公平性损失
                        predictions_reshaped = theta_group.view(1, -1)  # 模型预测 theta
                        targets_reshaped = group_users.view(1, -1)      # 目标 fairness_id

                        # 公平性损失计算：fairness_id 越低 theta 越高
                        fairness_loss_val = self.fairness_loss(predictions_reshaped, targets_reshaped)
                        group_fairness_losses.append(fairness_loss_val)

                    # 合并 BCE 损失与公平性损失
                    if group_fairness_losses:
                        total_fairness_loss = torch.mean(torch.stack(group_fairness_losses))
                        loss = (1 - self.fairness_lambda) * bce_loss_val + self.fairness_lambda * total_fairness_loss
                        fairness_losses.append(total_fairness_loss.item())
                    else:
                        loss = bce_loss_val
                else:
                    loss = bce_loss_val

                # 优化器更新
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                # 记录损失
                losses.append(loss.item())
                bce_losses.append(bce_loss_val.item())

            # 打印日志
            log_str = "[Epoch %d] Total Loss: %.6f, BCE Loss: %.6f" % (
                e, float(np.mean(losses)), float(np.mean(bce_losses)))
            if self.use_fairness:
                log_str += ", Fairness Loss: %.6f" % float(np.mean(fairness_losses))
            print(log_str)

            # 测试集评估
            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response, fairness_id, group_id, groupindex, group_size = batch_data
            fairness_id: torch.Tensor = fairness_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(fairness_id, item_id, fairness=False)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.irt_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

    def extract_ability_parameters(self, test_data, filepath, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()  # Switch to evaluation mode

        abilities = []
        processed_fairness_ids = set()  # To track processed (group_id, fairness_id)

        for batch_data in tqdm(test_data, "Extracting abilities"):
            group_id, user_id, item_id, response, fairness_id = batch_data
            user_id = user_id.to(device)
            fairness_id = fairness_id.to(device)
            # Retrieve the ability (θ) parameter for the user based on fairness_id
            theta = self.irt_net.theta(fairness_id).squeeze()  # Assuming theta is directly linked to fairness_id

            # Add group_id, fairness_id, user_id, and corresponding θ value to the list
            for i, fairness in enumerate(fairness_id.cpu().numpy()):
                if (group_id[i].item(), fairness) not in processed_fairness_ids:
                    abilities.append([
                        int(group_id[i]),         # group_id
                        int(fairness_id[i]),      # fairness_id
                        int(user_id[i]),          # user_id
                        float(theta[i].item())    # theta
                    ])
                    processed_fairness_ids.add((group_id[i].item(), fairness))  # Mark as processed

        # Save abilities to a CSV file with group_id, fairness_id, user_id, and theta
        df_abilities = pd.DataFrame(abilities, columns=["group_id", "fairness_id", "user_id", "theta"])
        df_abilities.sort_values(by=["group_id", "fairness_id"], inplace=True)  # Sort by group_id and fairness_id
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")

        self.irt_net.train()  # Switch back to training mode
