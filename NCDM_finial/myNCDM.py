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
        self.zeta = zeta
        self.groupsize = groupsize

    def train(self, train_data, test_data=None, epoch=10, device="cuda", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)#cpu
        self.ncdm_net.train()
        score_loss_function = nn.BCELoss()
        theta_loss_function = FairnessLoss()  # now has id, id_pair, n as additional inputs
        
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        
        for epoch_i in range(epoch):
            epoch_losses = []
            epoch_score_losses = []
            epoch_fairness_losses = []
            batch_count = 0

            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                origin_id, user_id, item_id, knowledge_emb, y = batch_data
                #originid index the data
                #origin_id = origin_id.to(device)
                user_id :torch.Tensor = user_id.to(device) #use this to calculate
                item_id :torch.Tensor = item_id.to(device)
                knowledge_emb :torch.Tensor = knowledge_emb.to(device)
                y :torch.Tensor = y.to(device) #score

                #分数误差
                predicted_response = self.ncdm_net(user_id, item_id, knowledge_emb, False)
                loss_score = score_loss_function(predicted_response, y)
                
                #接下来算小组误差

                pair_id=[] #there are batch_size id in total
                # each have a group ,so len(pair_id)=batch_size
                # len(pair_id[i]) = self.group_size
                for i in range(user_id.size(0)):
                    group_pair = []
                    group_index = user_id[i].item() // self.groupsize
                    for j in range(self.groupsize):
                        group_pair.append(j + group_index * self.groupsize)
                    
                    pair_id.append(group_pair)
                
                pair_id_tensor = torch.tensor(pair_id , device = device)
                loss_theta = 0
                num=0
                #下面是计算theta误差，大循环是对每一个group，小循环是对group的每个人的每一维度能力的pairloss，最后去平均
                for i in range(len(pair_id)):
                    if random.random() < 4/5:
                        continue
                    num=num+1
                    group_user_ids = pair_id_tensor[i]  # Get user IDs for the current group
                    # Call self.ncdm_net to get predicted_response and theta for this group
                    theta_group = self.ncdm_net(group_user_ids, item_id, knowledge_emb, True)

                    group_size, num_dimensions = theta_group.shape  # (11, 123)

                    # Reshape group_user_ids to match the required target format for fairness_loss
                    targets_reshaped = group_user_ids.view(1, -1)  # Shape: [1, 11]

                    # Randomly sample 1/3 of the users for ranking
                    sampled_user_indices = torch.randperm(group_size)[:group_size // 2]
                    sampled_user_ids = targets_reshaped[:, sampled_user_indices]  # Sampled targets for fairness loss

                    # Randomly sample 1/3 of the dimensions for fairness calculation
                    sampled_dimensions = torch.randperm(num_dimensions)[:num_dimensions // 8]

                    # Initialize a list to store fairness loss for each dimension
                    losses_per_dimension = []

                    for dim in sampled_dimensions:
                        # Select the sampled users' scores for the current dimension
                        predictions_reshaped = theta_group[sampled_user_indices, dim].view(1, -1)

                        # Compute fairness loss for the current dimension
                        fairness_loss_value = theta_loss_function(predictions_reshaped, sampled_user_ids)

                        # Append the result
                        losses_per_dimension.append(fairness_loss_value)

                    # Compute the mean fairness loss across all sampled dimensions
                    average_fairness_loss = torch.mean(torch.stack(losses_per_dimension))

                    # Append the fairness loss for this group to the total loss
                    loss_theta += average_fairness_loss

                # Compute the overall loss across all groups
                if num != 0:
                    loss_theta = loss_theta / num
                else:
                    loss_theta = torch.tensor(0.0, device=device)

                loss_score = loss_score.mean()

                # Final combined loss calculation
                loss = (1 - self.zeta) * loss_score + self.zeta * loss_theta
                loss = loss.mean()
                epoch_score_losses.append(loss_score.item())
                #print(type(loss_theta))
                epoch_fairness_losses.append(loss_theta.item())
                #print(type(epoch_fairness_losses[0]))# <class 'list'>
                #print(type(loss_theta))# <class 'torch.Tensor'>
                #print(type(loss_score))# <class 'torch.Tensor'>
                #print(type(epoch_score_losses[0]))# <class 'list'>
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f, score loss: %.6f ,fairness loss: %.6f" 
                  % (epoch_i, float(np.mean(epoch_losses)), float(np.mean(epoch_score_losses)),
                   float(np.mean(epoch_fairness_losses)    
                  )))

        if test_data is not None:
            auc, accuracy = self.eval(test_data, device=device)
            epoch_i=0
            print("[Epoch %d]  auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    
    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            origin_id, user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb, False)
            y: torch.Tensor = y.to(device)
            loss = loss_function(pred, y)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
    # 加载时显式指定 weights_only=True
        self.ncdm_net.load_state_dict(torch.load(filepath, weights_only=True))
        logging.info("load parameters from %s" % filepath)

    def extract_user_abilities(self, test_data, device="cpu", weighted=False, filepath="v_ability_parameters.csv"):
        """
        Extract and save student ability parameters (hs) after training, with an option to compute
        weighted abilities based on item difficulty.
    
        :param test_data: DataLoader containing the test data
        :param device: Device to use for computation ('cuda' or 'cpu')
        :param weighted: Whether to use weighted abilities based on item difficulty
        :return: DataFrame with origin_id, user_id, and their ability score (theta)
        """
        self.ncdm_net = self.ncdm_net.to(device)  # Ensure the model is moved to the correct device
        self.ncdm_net.eval()  # Set the model to evaluation mode
    
        # Prepare to store the results in a dictionary to avoid duplicates
        user_theta_map = {}
    
        for batch_data in test_data:
            origin_id, user_id, item_id, knowledge_emb, y = batch_data
            origin_id = origin_id.cpu().numpy()
            user_id = user_id.to(device)
            item_id = item_id.to(device)
    
            # Extract student embeddings and move to CPU for calculation
            student_embeddings = self.ncdm_net.student_emb(user_id).detach().cpu().numpy()
            stat_emb = torch.sigmoid(torch.tensor(student_embeddings)).numpy()  # hs
    
            # Compute the scalar ability for each student
            if weighted:
                # Use item difficulty to compute weighted ability
                k_difficulty = torch.sigmoid(self.ncdm_net.k_difficulty(item_id)).detach().cpu().numpy()
                e_difficulty = torch.sigmoid(self.ncdm_net.e_difficulty(item_id)).detach().cpu().numpy()
                weighted_ability = np.mean(stat_emb * k_difficulty * e_difficulty, axis=1)
                theta = weighted_ability
            else:
                # Simply use the average of the ability vector
                theta = np.mean(stat_emb, axis=1)
    
            # Update user_theta_map to ensure unique user_id and theta
            for oid, uid, ability in zip(origin_id, user_id.cpu().numpy(), theta):
                adjusted_uid = uid + 1  # Adjust user_id to start from 1
                adjusted_oid = oid + 1  # Adjust origin_id to match correct indexing
                if adjusted_uid in user_theta_map:
                    # If user_id already exists, you can choose to average, max, or replace the theta
                    user_theta_map[adjusted_uid] = (user_theta_map[adjusted_uid][0], 
                                                    (user_theta_map[adjusted_uid][1] + ability) / 2)  # Example: averaging
                else:
                    user_theta_map[adjusted_uid] = (adjusted_oid, ability)

        # Create a DataFrame with origin_id, user_id, and theta (ability score)
        df = pd.DataFrame([(uid, oid, theta) for uid, (oid, theta) in user_theta_map.items()], 
                          columns=['user_id', 'origin_id', 'theta'])
    
        # Save the DataFrame to a CSV file
        df.sort_values(by="user_id", inplace=True)
        df.to_csv(filepath, index=False)
        print(f"Student abilities (theta) saved to '{filepath}'")
