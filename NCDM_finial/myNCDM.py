# coding: utf-8

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from EduCDM import CDM
from loss import FairnessLoss
import pandas as pd
import random


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point, fairness):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        if fairness:  # return the student ability directly 
            return stat_emb
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        
        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, zeta=0.5, groupsize=11):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
        self.fairness_lambda = zeta
        self.fairness_loss = FairnessLoss()

    def train(self, train_data, test_data=None, epoch=10, device="cuda", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)

        for epoch_i in range(epoch):
            epoch_losses = []
            epoch_score_losses = []
            epoch_fairness_losses = []

            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size,comm_konw = batch_data
                
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device)
                fairness_id = fairness_id.to(device)
                group_id = group_id.to(device)
                groupindex = groupindex.to(device)
                group_size = group_size.to(device)
                comm_konw = comm_konw.to(device)

                # 计算预测的响应
                predicted_response = self.ncdm_net(fairness_id, item_id, knowledge_emb, fairness=False)
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
                    #print("group_users:",group_users)
                    #latent_dim = self.ncdm_net.student_emb.embedding_dim
                    #k = max(1, latent_dim // 24 + 20)
                    #random.seed(int(gid.item()))
                    #selected_dims = random.sample(range(latent_dim), k)
                    selected_dims = comm_konw[group_mask][0].tolist()
                    #print("selected_dims:",selected_dims)
                    theta_group_full = self.ncdm_net(group_users, None, None, fairness=True)
                    theta_group_selected = theta_group_full[:, selected_dims]

                    theta_mean = theta_group_selected.mean(dim=1)
                    predictions_reshaped = theta_mean.view(1, -1)  # 模型预测 theta
                    targets_reshaped = group_users.view(1, -1)  # 目标 fairness_id

                    # 公平性损失计算：fairness_id 越低 theta 越高
                    #print("predictions_reshaped:",predictions_reshaped)
                    #print("targets_reshaped:",targets_reshaped)
                    fairness_loss_val = self.fairness_loss(predictions_reshaped, targets_reshaped)

                    group_fairness_losses.append(fairness_loss_val)

                # 合并损失
                if group_fairness_losses:
                    fairness_loss = torch.mean(torch.stack(group_fairness_losses))
                    loss = (1 - self.fairness_lambda) * loss_score + self.fairness_lambda * fairness_loss
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

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d]  auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))
    
    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
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

            pred = self.ncdm_net(fairness_id, item_id, knowledge_emb, fairness=False)
            loss = loss_function(pred, y)
            losses.append(loss.item())

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
    # 加载时显式指定 weights_only=True
        self.ncdm_net.load_state_dict(torch.load(filepath, weights_only=True))
        logging.info("load parameters from %s" % filepath)

    def extract_ability_parameters(self, test_data, filepath, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()  # Switch to evaluation mode
    
        abilities = []
        processed_user_ids = set()  # To track processed (group_id, user_id)
    
        for batch_data in tqdm(test_data, "Extracting abilities"):
            user_id, item_id, response, group_id, fairness_id, knowledge_emb, comm = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            response = response.to(device)
            fairness_id = fairness_id.to(device)
            comm = comm.to(device)
    
            # Retrieve the ability (θ) parameter for the user
            student_embeddings = self.ncdm_net.student_emb(fairness_id).detach().cpu().numpy()
            stat_emb = torch.sigmoid(torch.tensor(student_embeddings)).numpy()  # hs
    
            # Add group_id, fairness_id, user_id, and corresponding θ values to the list
            for i, user in enumerate(fairness_id.cpu().numpy()):
                if (group_id[i].item(), user) not in processed_user_ids:
                    selected_dims = comm[i].cpu().numpy().astype(int)  # Ensure comm is an integer array
                    theta_values = stat_emb[i, selected_dims]
                    ability_entry = [
                        int(group_id[i]),
                        int(fairness_id[i] + 1),
                        int(user_id[i])
                    ] + theta_values.tolist()
                    abilities.append(ability_entry)
                    processed_user_ids.add((group_id[i].item(), user))  # Mark as processed
    
        # Create column names for the output file
        max_comm_length = max(len(comm[i]) for i in range(len(comm)))
        columns = ["group_id", "fairness_id", "user_id"] + [f"theta_{j}" for j in range(max_comm_length)]
    
        # Save abilities to a CSV file with group_id and fairness_id
        df_abilities = pd.DataFrame(abilities, columns=columns)
        df_abilities.sort_values(by=["group_id", "fairness_id"], inplace=True)  # Sort by group_id and fairness_id
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")
    
        self.ncdm_net.train()  # Switch back to training mode