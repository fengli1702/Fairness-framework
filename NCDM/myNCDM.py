# coding: utf-8
# 2021/4/1 @ WangFei

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

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        # neg计算权重的相反数, relu将负数置0, 乘2将权重放大, 加self.weight将负数权重变为正数
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

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
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

    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                #print(batch_data)
                batch_count += 1
                
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                #print(user_id)
                #print(user_id[0],user_id[1],user_id[2])
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
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