import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import pandas as pd
import logging
from loss import FairnessLoss

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(-self.weight) + self.weight
        return F.linear(input, weight, self.bias)

def generate_q_mask(knowledge_emb):
    # `knowledge_emb` 直接作为 q_mask 返回
    return (knowledge_emb > 0).float()


class KSCDNet(nn.Module):
    def __init__(self, knowledge_num, latent_dim, stu_num, exer_num, dropout, device, dtype):
        super(KSCDNet, self).__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device

        # 嵌入层
        self.user_embedding = nn.Embedding(num_embeddings=stu_num, embedding_dim=latent_dim)
        self.item_embedding = nn.Embedding(num_embeddings=exer_num, embedding_dim=latent_dim)
        self.knowledge_embedding = nn.Embedding(num_embeddings=self.knowledge_num, embedding_dim=self.latent_dim)

        # Prediction layers
        self.prednet_full1 = PosLinear(self.knowledge_num + self.latent_dim, self.knowledge_num, dtype=dtype).to(self.device)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.knowledge_num + self.latent_dim, self.knowledge_num, dtype=dtype).to(self.device)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(self.knowledge_num, 1, dtype=dtype).to(self.device)

        # Weight initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ts, diff_ts, q_mask, knowledge_emb, mode='train', fairness=None):
        # 嵌入计算
        student_ts = self.user_embedding(student_ts)  # Shape: [batch_size, latent_dim]
        if fairness:
            return student_ts

        knowledge_ts = self.knowledge_embedding.weight  # Shape: [knowledge_num, latent_dim]
        diff_ts = self.item_embedding(diff_ts)          # Shape: [batch_size, latent_dim]

        # Compute student ability and exercise difficulty
        stu_ability = torch.mm(student_ts, knowledge_ts.T).sigmoid()  # Shape: [batch_size, knowledge_num]
        exer_diff = torch.mm(diff_ts, knowledge_ts.T).sigmoid()       # Shape: [batch_size, knowledge_num]

        # Broadcast for batch processing
        batch_stu_vector = stu_ability.unsqueeze(1).expand(-1, self.knowledge_num, -1)
        batch_exer_vector = exer_diff.unsqueeze(1).expand(-1, self.knowledge_num, -1)
        kn_vector = knowledge_ts.repeat(stu_ability.shape[0], 1).reshape(stu_ability.shape[0], self.knowledge_num, self.latent_dim)

        # Calculate preference and difficulty
        preference = torch.tanh(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.tanh(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))

        # Compute output
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        if mode == 'train':
            # Training mode: calculate averaged prediction
            sum_out = torch.sum(o * q_mask.unsqueeze(2), dim=1)
            count_of_concept = torch.sum(q_mask, dim=1).unsqueeze(1)
            y_pd = sum_out / count_of_concept
            return y_pd.view(-1)
        elif mode == 'transform':
            # Transform mode: return raw preference output
            return o.squeeze(-1)
        else:
            raise ValueError("Unsupported mode. Use 'train' or 'transform'.")

class KSCD_IF(nn.Module):
    def __init__(self, dropout, knowledge_num, latent_dim, stu_num, exer_num, device, dtype):
        super(KSCD_IF, self).__init__()
        self.kscd_net = KSCDNet(knowledge_num, latent_dim, stu_num, exer_num, dropout, device, dtype)
        self.fairness_lambda = 0.5  # 默认公平性损失的权重
        self.fairness_loss = FairnessLoss()
    
    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.kscd_net = self.kscd_net.to(device)  # 使用 KSCD 模型
        self.kscd_net.train()
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(self.kscd_net.parameters(), lr=lr)

        for epoch_i in range(epoch):
            epoch_losses = []
            epoch_score_losses = []
            epoch_fairness_losses = []

            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):

                # 解包数据
                user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size, comm_konw = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device)
                fairness_id = fairness_id.to(device)
                group_id = group_id.to(device)
                groupindex = groupindex.to(device)
                group_size = group_size.to(device)
                comm_konw = comm_konw.to(device)

                # 使用 knowledge_emb 动态生成 q_mask
                q_mask = generate_q_mask(knowledge_emb).to(device)

                # 调用 KSCD 的 forward 方法
                pred: torch.Tensor = self.kscd_net(fairness_id, item_id, q_mask, knowledge_emb)

                # 计算响应损失
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

                    theta_group_full = self.kscd_net(group_users, diff_ts=None, q_mask=None, knowledge_emb=None, mode='transform', fairness=True)
                    theta_group_selected = theta_group_full[:, selected_dims]

                    theta_mean = theta_group_selected.mean(dim=1)
                    predictions_reshaped = theta_mean.view(1, -1)
                    targets_reshaped = group_users.view(1, -1)
                    #print("predictions_reshaped: ", predictions_reshaped)
                    #print("targets_reshaped: ", targets_reshaped)

                    fairness_loss_val = self.fairness_loss(predictions_reshaped, targets_reshaped)
                    group_fairness_losses.append(fairness_loss_val)
                    #print("fairnessloss:",fairness_loss_val)
                    #print("group_fairness_losses: ", group_fairness_losses)
                # 合并损失
                if group_fairness_losses:
                    fairness_loss = torch.mean(torch.stack(group_fairness_losses))
                    loss = (1 - self.fairness_lambda) * loss_score + self.fairness_lambda * fairness_loss
                else:
                    fairness_loss = 0.0
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
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.kscd_net = self.kscd_net.to(device)
        self.kscd_net.eval()

        y_pred = []
        y_true = []

        for batch_data in tqdm(test_data, "evaluating"):
            # 解包数据
            user_id, item_id, knowledge_emb, y, fairness_id, group_id, groupindex, group_size,comm = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            fairness_id = fairness_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            y: torch.Tensor = y.to(device)

            # 使用 knowledge_emb 动态生成 q_mask
            q_mask = generate_q_mask(knowledge_emb).to(device)

            # 调用 KSCD 的 forward 方法
            with torch.no_grad():
                pred: torch.Tensor = self.kscd_net(fairness_id, item_id, q_mask, knowledge_emb)

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        # 返回 AUC 和准确率
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    
    def save(self, path):
        torch.save(self.kscd_net.state_dict(), path)
        logging.info("save parameters to %s" % path)
    def load(self, path):
        self.kscd_net.load_state_dict(torch.load(path))
        logging.info("load parameters from %s" % path)

    def extract_ability_parameters(self, test_data, filepath, device="cpu"):
        self.kscd_net = self.kscd_net.to(device)
        self.kscd_net.eval()  # Switch to evaluation mode
    
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
            student_embeddings = self.kscd_net.user_embedding(fairness_id).detach().cpu().numpy()
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
    
        self.kscd_net.train()  # Switch back to training mode

        