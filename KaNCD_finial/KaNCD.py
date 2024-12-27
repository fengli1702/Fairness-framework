# coding: utf-8
# 2023/7/3 @ WangFei

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
from loss import FairnessLoss


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point, fairness=False):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        if fairness:
            return torch.sigmoid(stu_emb)
        exer_emb = self.exercise_emb(input_exercise)
        
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        
        if fairness:
            return stat_emb
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # prednet
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class KaNCD(CDM):
    def __init__(self, **kwargs):
        super(KaNCD, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], mf_type, kwargs['dim'])

    # ...existing code...
class KaNCD(CDM):
    def __init__(self, **kwargs):
        super(KaNCD, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], mf_type, kwargs['dim'])

    def train(self, train_set, valid_set, lr=0.002, device='cpu', epoch_n=15):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        fairness_loss_fn = FairnessLoss()
        fairness_lambda = 0.5

        for epoch_i in range(epoch_n):
            epoch_losses = []
            epoch_score_losses = []
            epoch_fairness_losses = []

            for batch_data in tqdm(train_set, "Epoch %s" % epoch_i):
                user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size, comm_konw = batch_data
                
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device)
                fairness_id = fairness_id.to(device)
                group_id = group_id.to(device)
                groupindex = groupindex.to(device)
                group_size = group_size.to(device)
                comm_konw = comm_konw.to(device)

                # 计算预测的响应
                predicted_response = self.net(fairness_id, item_id, knowledge_emb, fairness=False)
                loss_score = bce_loss(predicted_response, y)

                # 计算公平性损失
                unique_groups = torch.unique(group_id)
                group_fairness_losses = []
                for gid in unique_groups:
                    group_mask = (group_id == gid)
                    
                    group_start = groupindex[group_mask][0].item()
                    group_sz = group_size[group_mask][0].item()
                    if group_sz <= 2:
                        continue

                    group_users = [i for i in range(group_start, group_start + group_sz)]
                    group_users = torch.tensor(group_users, dtype=torch.int64).to(device)
                    
                    selected_dims = comm_konw[group_mask][0].tolist()
                    theta_group_full = self.net(group_users, None, None, fairness=True)
                    theta_group_selected = theta_group_full[:, selected_dims]

                    theta_mean = theta_group_selected.mean(dim=1)
                    predictions_reshaped = theta_mean.view(1, -1)  # 模型预测 theta
                    targets_reshaped = group_users.view(1, -1)  # 目标 fairness_id

                    # 公平性损失计算：fairness_id 越低 theta 越高
                    fairness_loss_val = fairness_loss_fn(predictions_reshaped, targets_reshaped)

                    group_fairness_losses.append(fairness_loss_val)

                # 合并损失
                if group_fairness_losses:
                    fairness_loss = torch.mean(torch.stack(group_fairness_losses))
                    loss = (1 - fairness_lambda) * loss_score + fairness_lambda * fairness_loss
                else:
                    loss = loss_score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_score_losses.append(loss_score.item())
                if group_fairness_losses:
                    epoch_fairness_losses.append(fairness_loss.item())
                else:
                    epoch_fairness_losses.append(0.0)

            print("[Epoch %d] average loss: %.6f, score loss: %.6f, fairness loss: %.6f" % (
                epoch_i, float(np.mean(epoch_losses)), float(np.mean(epoch_score_losses)),
                float(np.mean(epoch_fairness_losses))
            ))
            auc, accuracy = self.eval(valid_set, device=device)
            print("[Epoch %d]  auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        loss_function = nn.BCELoss()
        losses = []

        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size,comm = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            fairness_id = fairness_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)

            pred = self.net(fairness_id, item_id, knowledge_emb, fairness=False)
            loss = loss_function(pred, y)
            losses.append(loss.item())

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
    
    def extract_ability_parameters(self, test_data, filepath, device="cpu",eval_data=None):
          # Ensure the entire model is moved to the device
        self.net = self.net.to(device)
        self.eval(eval_data,device)

        abilities = []
        processed_user_ids = set()  # To track processed (group_id, user_id)

        for batch_data in tqdm(test_data, desc="Extracting abilities"):
            user_id, item_id, response, group_id, fairness_id, knowledge_emb, comm = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            response = response.to(device)
            fairness_id = fairness_id.to(device)
            comm = comm.to(device)

            # Retrieve the ability (theta) parameter for the user
            student_embeddings = self.net.student_emb(fairness_id).detach().cpu().numpy()
            stat_emb = torch.sigmoid(torch.tensor(student_embeddings)).numpy()  # hs

            # Add group_id, fairness_id, user_id, and corresponding theta values to the list
            for i, user in enumerate(user_id.cpu().numpy()):
                if (group_id[i].item(), user) not in processed_user_ids:
                    selected_dims = comm[i].cpu().numpy().astype(int)  # Ensure comm is an integer array
                    theta_values = stat_emb[i, selected_dims]
                    ability_entry = [
                        int(group_id[i]),
                        int(fairness_id[i] + 1),  # fairness_id is incremented by 1
                        int(user_id[i]+1)
                    ] + theta_values.tolist()
                    abilities.append(ability_entry)
                    processed_user_ids.add((group_id[i].item(), user))

        # Create column names for the output file
        max_comm_length = max(len(comm[i]) for i in range(len(comm)))
        columns = ["group_id", "fairness_id", "user_id"] + [f"theta_{j}" for j in range(max_comm_length)]

        # Save abilities to a CSV file with group_id and fairness_id
        df_abilities = pd.DataFrame(abilities, columns=columns)
        df_abilities.sort_values(by=["group_id", "fairness_id"], inplace=True)
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")

        
