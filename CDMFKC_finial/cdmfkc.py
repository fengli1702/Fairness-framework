import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
from loss import FairnessLoss

class CDMFKCNet(nn.Module):
    def __init__(self, g_impact_a, g_impact_b,stu_n,exer_n, knowledge_num, hidden_dims, dropout, latent_dim, device, dtype):
        super(CDMFKCNet, self).__init__()
        self.knowledge_num = knowledge_num
        self.g_impact_a = g_impact_a
        self.g_impact_b = g_impact_b
        self.device = device
        self.latent_dim = latent_dim

        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_embeddings=stu_n, embedding_dim=latent_dim).to(device)  # 10000 is an example, adjust as necessary
        self.item_embedding = nn.Embedding(num_embeddings=exer_n, embedding_dim=latent_dim).to(device)  # Similarly, adjust item num_embeddings
        
        # Impact transformation if latent_dim is provided
        if latent_dim is not None:
            self.transform_impact = nn.Linear(latent_dim, knowledge_num, dtype=dtype).to(device)

        # MLP layers for final prediction
        layers = nn.Sequential()
        input_dim = knowledge_num
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.add_module(f'linear{idx}', nn.Linear(input_dim, hidden_dim, dtype=dtype))
            layers.add_module(f'activation{idx}', nn.Tanh())
            layers.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.add_module('output', nn.Linear(hidden_dims[-1], 1, dtype=dtype))
        layers.add_module('sigmoid', nn.Sigmoid())

        self.mlp = layers.to(device)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, user_id, item_id, knowledge_emb, q_mask,fairness=False, response=None):
        
        # Convert user_id and item_id to embeddings
        user_embedding = self.user_embedding(user_id)  # Shape: [batch_size, latent_dim]
        if fairness:
            return user_embedding
        
        item_embedding = self.item_embedding(item_id)  # Shape: [batch_size, latent_dim]

        # Knowledge impact (use knowledge_emb for dynamic computation)
        h_impact = torch.sigmoid(knowledge_emb)  # Shape: [batch_size, knowledge_num]

        # Compute g_impact using provided parameters
        g_impact = torch.sigmoid(self.g_impact_a * h_impact + self.g_impact_b * item_embedding)

        # Calculate input for MLP
        input_x = user_embedding + g_impact - item_embedding
        input_x = input_x * q_mask  # Apply mask to focus on relevant knowledge points

        # Pass through MLP
        output = self.mlp(input_x)
        return output.view(-1)

    def transform(self, user_id):
        """
        For extracting user embeddings or ability parameters.
        :param user_id: Tensor of user IDs.
        :return: Transformed user embeddings.
        """
        return torch.sigmoid(self.user_embedding(user_id))


class CDMFKC(nn.Module):
    def __init__(self, student_num, item_num, knowledge_num, latent_dim, hidden_dims=None, dropout=0.5, 
                 g_impact_a=0.5, g_impact_b=0.5, device="cpu", dtype=torch.float32):
        super(CDMFKC, self).__init__()
        self.fairness_lambda = 0.4
        self.fairness_loss = FairnessLoss()

        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.cdmfkc_net = CDMFKCNet(
            g_impact_a=g_impact_a,
            g_impact_b=g_impact_b,
            stu_n = student_num,
            exer_n = item_num,
            knowledge_num=knowledge_num,
            hidden_dims=hidden_dims,
            dropout=dropout,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype
        )
    def generate_q_matrix(item_num, knowledge_num):
        """
        Generate a q_matrix with all entries as 1.
        :param item_num: Number of items (exercises).
        :param knowledge_num: Number of knowledge points.
        :return: A tensor with shape [item_num, knowledge_num] filled with 1s.
        """
        return torch.ones((item_num, knowledge_num), dtype=torch.float32)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, batch_size=256):
        self.cdmfkc_net = self.cdmfkc_net.to(device)
        self.cdmfkc_net.train()
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(self.cdmfkc_net.parameters(), lr=lr)

        for epoch_i in range(epoch):
            epoch_losses = []
            epoch_score_losses = []
            epoch_fairness_losses = []
            for batch_data in tqdm(train_data, f"Epoch {epoch_i}"):
                # Unpack batch data
                user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size,comm_konw = batch_data
                
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device)
                fairness_id = fairness_id.to(device)
                group_id = group_id.to(device)
                groupindex = groupindex.to(device)
                group_size = group_size.to(device)
                comm_konw = comm_konw.to(device)

                # Generate q_mask (if not provided in dataset)
                q_mask = (knowledge_emb > 0).float().to(device)
                
                # Forward pass
                pred = self.cdmfkc_net(fairness_id, item_id, knowledge_emb, q_mask)

                # Compute loss
                loss_score = bce_loss(pred, y)
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
                   
                    theta_group_full = self.cdmfkc_net(group_users, item_id, knowledge_emb, q_mask, fairness=True)
                    theta_group_selected = theta_group_full[:, selected_dims]

                    theta_mean = theta_group_selected.mean(dim=1)
                    predictions_reshaped = theta_mean.view(1, -1)  # 模型预测 theta
                    targets_reshaped = group_users.view(1, -1)  # 目标 fairness_id

                    # 公平性损失计算：fairness_id 越低 theta 越高
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

            print(f"[Epoch {epoch_i}] Loss: {np.mean(epoch_losses)}")
            
            if test_data is not None:
                auc, acc = self.eval(test_data, device)
                print(f"[Epoch {epoch_i}] AUC: {auc:.6f}, Accuracy: {acc:.6f}")

    def eval(self, test_data, device="cpu") -> tuple:
        """
        Evaluate the CDMFKCNet model.
        :param test_data: DataLoader for the test dataset.
        :param device: Device to run the evaluation on ("cpu" or "cuda").
        :return: Tuple containing AUC and accuracy scores.
        """
        self.cdmfkc_net = self.cdmfkc_net.to(device)
        self.cdmfkc_net.eval()
        loss_function = nn.BCELoss()
        losses = []

        y_pred = []
        y_true = []

        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size,comm = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            fairness_id = fairness_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)
            q_mask = (knowledge_emb > 0).float().to(device)
            pred: torch.Tensor = self.cdmfkc_net(fairness_id, item_id, knowledge_emb, q_mask)
            loss = loss_function(pred, y)
            losses.append(loss.item())

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)


    def save(self, filepath):
        torch.save(self.cdmfkc_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.cdmfkc_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)
        
    def extract_ability_parameters(self, test_data, filepath, device="cpu",eval_data=None):
        """
        Extract user ability parameters from the model and save to a CSV file.
        """
        self.cdmfkc_net = self.cdmfkc_net.to(device)
        self.cdmfkc_net.eval()

        abilities = []  # Store extracted ability parameters
        processed_user_ids = set()  # Track processed user_ids to debug coverage

        for batch_data in tqdm(test_data, "Extracting abilities"):
            user_id, item_id, response, group_id, fairness_id, knowledge_emb, comm = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            response = response.to(device)
            fairness_id = fairness_id.to(device)
            comm = comm.to(device)

            # Retrieve the ability (theta) parameter for the user
            student_embeddings = self.cdmfkc_net.user_embedding(fairness_id).detach().cpu().numpy()
            stat_emb = torch.sigmoid(torch.tensor(student_embeddings)).numpy()  # Convert to probabilities

            # Add ability entries for each user
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
    
        self.cdmfkc_net.train()  # Switch back to training mode
    