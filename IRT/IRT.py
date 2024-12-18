# coding: utf-8
# 2021/4/23 @ tongshiwei

import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

#from ..irt import irt3pl
from sklearn.metrics import roc_auc_score, accuracy_score

###  3PL
def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


irt3pl = irf

class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range, a_range=1, irf_kwargs=None):
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

    def forward(self, user, item):  #自动调用
        #参数自动注册为模型的可训练参数
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        #print("forward first:", c[0])
        c = torch.sigmoid(c)
        #print("forward second:", c[0])
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        return self.irf(theta, a, b, c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)


class IRT(CDM):
    def __init__(self, user_num, item_num, value_range=None, a_range=None):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range) #实例化IRTNet类

    def train(self, train_data, test_data=None, *, epoch: int, device="cuda", lr=0.005) -> ...:
        self.irt_net = self.irt_net.to(device)
        loss_function = nn.BCELoss()  #二分类交叉熵作为损失函数

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)  ##注解语法表示 user_id 和 item_id 都是 torch.Tensor 类型
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.irt_net(user_id, item_id)  #隐式调用forward函数
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
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
        processed_user_ids = set()  # To track processed (group_id, user_id)

        for batch_data in tqdm(test_data, "Extracting abilities"):
            group_id,  user_id, item_id, response, fairness_id = batch_data
            user_id = user_id.to(device)
            """
            torch.tensor(groupid, dtype=torch.int64),
            torch.tensor(x, dtype=torch.int64),
            torch.tensor(y, dtype=torch.int64),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(fairnessid, dtype=torch.int64)"""
            # Retrieve the ability (θ) parameter for the user
            theta = self.irt_net.theta(user_id).squeeze()

            # Add group_id, fairness_id, user_id, and corresponding θ value to the list
            for i, user in enumerate(user_id.cpu().numpy()):
                if (group_id[i].item(), user) not in processed_user_ids:
                    abilities.append([
                        int(group_id[i]),
                        int(fairness_id[i]),
                        int(user),
                        float(theta[i].item())
                    ])
                    processed_user_ids.add((group_id[i].item(), user))  # Mark as processed

        # Save abilities to a CSV file with group_id and fairness_id
        df_abilities = pd.DataFrame(abilities, columns=["group_id", "fairness_id", "user_id", "theta"])
        df_abilities.sort_values(by=["group_id", "fairness_id"], inplace=True)  # Sort by group_id and fairness_id
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")

        self.irt_net.train()  # Switch back to training mode
