import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import logging
import pandas as pd

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(-self.weight) + self.weight
        return F.linear(input, weight, self.bias)

def generate_q_mask(knowledge_emb):
    # `knowledge_emb` 直接作为 q_mask 返回
    return (knowledge_emb > 0).float()

class KSCDNet(nn.Module):
    def __init__(self, knowledge_num, latent_dim,stu_num,exer_num ,dropout, device, dtype):
        super(KSCDNet, self).__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device

        # 嵌入层
        self.user_embedding = nn.Embedding(num_embeddings=stu_num, embedding_dim=latent_dim)  # 假设用户数为 1000
        self.item_embedding = nn.Embedding(num_embeddings=exer_num, embedding_dim=latent_dim)  # 假设题目数为 1000
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

    def forward(self, student_ts, diff_ts, q_mask, knowledge_emb, mode='train'):
        # 嵌入计算
        student_ts = self.user_embedding(student_ts)  # Shape: [batch_size, latent_dim]
        diff_ts = self.item_embedding(diff_ts)        # Shape: [batch_size, latent_dim]
        knowledge_ts = self.knowledge_embedding.weight  # Shape: [knowledge_num, latent_dim]

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
    def __init__(self, dropout, knowledge_num, latent_dim,stu_num,exer_num, device, dtype):
        super(KSCD_IF, self).__init__()
        self.kscd_net = KSCDNet(knowledge_num, latent_dim,stu_num,exer_num, dropout, device, dtype)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.kscd_net = self.kscd_net.to(device)  # 使用 KSCD 模型
        self.kscd_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.kscd_net.parameters(), lr=lr)

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0

            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1

                # 解包数据
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)

                # 使用 knowledge_emb 动态生成 q_mask
                q_mask = generate_q_mask(knowledge_emb).to(device)

                # 调用 KSCD 的 forward 方法
                pred: torch.Tensor = self.kscd_net(user_id, item_id, q_mask, knowledge_emb)

                # 计算损失
                loss = loss_function(pred, y)

                # 优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            # 打印每个 epoch 的平均损失
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            # 如果提供了测试数据，进行评估
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
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            y: torch.Tensor = y.to(device)

            # 使用 knowledge_emb 动态生成 q_mask
            q_mask = generate_q_mask(knowledge_emb).to(device)

            # 调用 KSCD 的 forward 方法
            with torch.no_grad():
                pred: torch.Tensor = self.kscd_net(user_id, item_id, q_mask, knowledge_emb)

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        # 返回 AUC 和准确率
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.kscd_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.kscd_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
    
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

        