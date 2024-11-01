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
from loss import PairSCELoss, HarmonicLoss


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

    def forward(self, stu_id, input_exercise, input_knowledge_point, user_id_pair=None):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        if user_id_pair is not None:
            stu_emb_pair = self.student_emb(user_id_pair)
            stat_emb_pair = torch.sigmoid(stu_emb_pair)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        if user_id_pair is not None:
            return output_1.view(-1), stat_emb, stat_emb_pair
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
        theta_loss_function = PairSCELoss()  # now has id, id_pair, n as additional inputs
        loss_function = HarmonicLoss(self.zeta)
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                origin_id, user_id, item_id, knowledge_emb, y = batch_data
                #originid index the data
                origin_id = origin_id.to(device)
                user_id :torch.Tensor = user_id.to(device) #use this to calculate
                item_id :torch.Tensor = item_id.to(device)
                knowledge_emb :torch.Tensor = knowledge_emb.to(device)
                y :torch.Tensor = y.to(device) #score

                
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
                loss_score = 0

                for i in range(len(pair_id)):
                    sample_indices = torch.randperm(len(pair_id[i]))[:len(pair_id[i]) // 3]  # Randomly select 1/3
                    for j in sample_indices:
                        if i != pair_id[i][j] :  # Ensure we're not calculating loss with the user itself
                            # Call the model for the user and each group member

                            predicted_response, predicted_theta, predicted_theta_pair = self.ncdm_net(
                                user_id, item_id,
                                knowledge_emb, pair_id_tensor[i][j]
                            )
                            #print(predicted_response.shape)
                            # Calculate theta loss between the user and each group member
                            theta_i = predicted_theta[i]          # Shape [123]
                            predicted_response = torch.Tensor(predicted_response)
                            predicted_theta = torch.Tensor(theta_i)
                            predicted_theta_pair = torch.Tensor(predicted_theta_pair)
                            #print(predicted_theta.shape)
                            #print(predicted_theta_pair.shape)

                            loss1, count1 = theta_loss_function(
                                predicted_theta, predicted_theta_pair,
                                user_id[i].item(), pair_id[i][j], self.groupsize
                            )
                            loss1 = loss1.mean().item()
                            #print(loss_theta.shape)  int
                            loss_theta += loss1
                            count += count1
                
                #print(predicted_response.shape)
                #print(y.shape)
                loss_score = score_loss_function(predicted_response, y)
                loss_theta = loss_theta / count
                loss_score = loss_score.mean()

                loss = loss_function(loss_score, loss_theta)
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f, Count: %d" % (epoch_i, float(np.mean(epoch_losses)), count))

        if test_data is not None:
            rmse, mae, auc, accuracy = self.eval(test_data, device=device)
            print("[Epoch %d] rmse: %.6f, mae: %.6f, auc: %.6f, accuracy: %.6f" % (epoch_i, rmse, mae, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y: torch.Tensor = y.to(device)
            loss = loss_function(pred, y)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

    def extract_user_abilities(self, test_data, device="cuda", weighted=False, filepath="v_ability_parameters.csv"):
        """
        Extract and save student ability parameters (hs) after training, with an option to compute
        weighted abilities based on item difficulty.

        :param test_data: DataLoader containing the test data
        :param device: Device to use for computation ('cuda' or 'cpu')
        :param weighted: Whether to use weighted abilities based on item difficulty
        :return: DataFrame with user_id and their ability score (theta)
        """
        self.ncdm_net = self.ncdm_net.to(device)  # Ensure the model is moved to the correct device
        self.ncdm_net.eval()  # Set the model to evaluation mode

        # Prepare to store the results in a dictionary to avoid duplicates
        user_theta_map = {}

        for batch_data in test_data:
            user_id, item_id, knowledge_emb, y = batch_data
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
            for uid, ability in zip(user_id.cpu().numpy(), theta):
                if uid in user_theta_map:
                    # If user_id already exists, you can choose to average, max, or replace the theta
                    user_theta_map[uid] = (user_theta_map[uid] + ability) / 2  # Example: averaging the abilities
                else:
                    user_theta_map[uid] = ability

        # Create a DataFrame with user_id and theta (ability score)
        df = pd.DataFrame(user_theta_map.items(), columns=['user_id', 'theta'])

        # Save the DataFrame to a CSV file
        df.sort_values(by="user_id", inplace=True)
        df.to_csv(filepath, index=False)
        print(f"Student abilities (theta) saved to '{filepath}'")