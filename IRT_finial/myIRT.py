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
    def __init__(self, user_num, item_num, value_range=None, a_range=None, use_fairness=True, fairness_lambda=0.7 , group_size = 11):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range)
        self.use_fairness = use_fairness
        self.groupsize = group_size
        self.fairness_lambda = fairness_lambda
        if use_fairness:
            self.fairness_loss = FairnessLoss()

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001):
        self.irt_net = self.irt_net.to(device)
        bce_loss = nn.BCELoss()
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            bce_losses = []
            fairness_losses = []

            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response = batch_data
                predicted_response= self.irt_net(user_id, item_id , fairness = False)
                #print("predicted_response: ",predicted_response.shape)  # torch.Size([256]) 是一个batch
                # 一个batch的response，一维向量
                # 计算BCE损失
                bce_loss_val = bce_loss(predicted_response, response) #无错，调用batch里的数据
                #print(user_id.size(0))  # 256
                #print(user_id)
                #获得user的group
                pair_id=[] #there are batch_size id in total
                #  each have a group ,so len(pair_id)=batch_size
                # len(pair_id[i]) = self.group_size 
                for i in range(user_id.size(0)):
                    group_pair = []
                    group_index = user_id[i].item() // self.groupsize
                    for j in range(self.groupsize):
                        group_pair.append(j + group_index * self.groupsize)
                    
                    pair_id.append(group_pair)
                
                pair_id_tensor = torch.tensor(pair_id , device = device)
                #print(pair_id_tensor) 没算错，里面是一个batch的group id

                # 如果启用fairness loss
                if self.use_fairness:
                    group_fairness_losses = []

                    # Iterate over the batch size (one group at a time)
                    for i in range(len(batch_data)):
                        group_user_ids = pair_id_tensor[i]  # Get user IDs for the current group

                        # Call `self.irt_net` to get predicted_response and theta for this group
                        theta_group = self.irt_net(group_user_ids, item_id , fairness = True)
                        #print("theta_group: ",theta_group.shape)  # torch.Size([1, 11]) 是一个group
                        #print("theta_group: ",theta_group)

                        # Reshape predictions and targets to match FairnessLoss input requirements
                        predictions_reshaped = theta_group.view(1, -1)  # Reshaping for a single group

                        # Reshape targets_rank as needed for fairness loss calculation
                        targets_reshaped = group_user_ids.view(1, -1)
                        

                        #print("predictions_reshaped: ",predictions_reshaped.shape)
                        #print("targets_reshaped: ",targets_reshaped.shape)

                        # Calculate the fairness loss for the current group
                        fairness_loss_val = self.fairness_loss(predictions_reshaped, targets_reshaped)

                        # Append the fairness loss for this group to the list
                        group_fairness_losses.append(fairness_loss_val)

                    # Optionally, average or sum the losses across the batch
                    total_fairness_loss = torch.mean(torch.stack(group_fairness_losses))  # Or use sum() if you prefer summing
                    
                    # 组合两种loss
                    loss = (1 - self.fairness_lambda) * bce_loss_val + self.fairness_lambda * total_fairness_loss
                    fairness_losses.append(fairness_loss_val.item())
                else:
                    loss = bce_loss_val

                trainer.zero_grad()
                loss.backward()
                #for param in  self.irt_net.parameters():
                #    print(param.grad)
                trainer.step()

                losses.append(loss.item())
                bce_losses.append(bce_loss_val.item())

            # 打印训练信息
            log_str = "[Epoch %d] Total Loss: %.6f, BCE Loss: %.6f" % (
                e, float(np.mean(losses)), float(np.mean(bce_losses)))
            if self.use_fairness:
                log_str += ", Fairness Loss: %.6f" % float(np.mean(fairness_losses))
            print(log_str)

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch_data in tqdm(test_data, "evaluating"):
                user_id, item_id, response = batch_data
                pred = self.irt_net(user_id, item_id , fairness = False)
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(response.cpu().numpy())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

    def extract_ability_parameters(self, test_data, filepath, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()

        abilities = []
        processed_user_ids = set()

        with torch.no_grad():
            for batch_data in tqdm(test_data, "Extracting abilities"):
                origin_id, user_id, item_id, response = batch_data
                theta = self.irt_net.theta(user_id).squeeze()

                for i, user in enumerate(user_id.cpu().numpy()):
                    if user not in processed_user_ids:
                        abilities.append([int(origin_id[i]), int(user), float(theta[i].item())])
                        processed_user_ids.add(user)

        df_abilities = pd.DataFrame(abilities, columns=["origin_id", "user_id", "theta"])
        df_abilities.sort_values(by="user_id", inplace=True)
        df_abilities.to_csv(filepath, index=False)