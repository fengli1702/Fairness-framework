import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
import pandas as pd
import random

from loss import FairnessLoss
def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, a_range, theta_range, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        self.a_range = a_range
        self.theta_range = theta_range

    def forward(self, user, item,fairness=False):
        theta = torch.squeeze(self.theta(user), dim=-1)
        if fairness:
            #挑选item_id对应的theta
            theta = torch.squeeze(self.theta(item), dim=-1)
            #theta按行求和
            theta = torch.sum(theta,dim=1)
            return theta
        
        a = torch.squeeze(self.a(item), dim=-1)
        if self.theta_range is not None:
            theta = self.theta_range * torch.sigmoid(theta)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        
        return self.irf(theta, a, b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


class MIRT(CDM):
    def __init__(self, user_num, item_num, latent_dim, a_range=None, theta_range=None,fairness_lambda=0.5):
        super(MIRT, self).__init__()
        self.irt_net = MIRTNet(user_num, item_num, latent_dim, a_range, theta_range)
        self.fairness_lambda = fairness_lambda  

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.004) -> ...:
        self.irt_net = self.irt_net.to(device)
        bce_loss = nn.BCELoss()
        self.fairness_loss = FairnessLoss()

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            fairness_losses = []
            bce_losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response, fairness_id, group_id, groupindex, group_size = batch_data
                item_id = item_id.to(device)
                response = response.to(device)
                fairness_id = fairness_id.to(device)

                predicted_response: torch.Tensor = self.irt_net(fairness_id, item_id, fairness=False)
                response: torch.Tensor = response.to(device)
                
                score_loss = bce_loss(predicted_response, response)
                #print(score_loss)

                #计算公平性损失
                unique_groups = torch.unique(group_id)
                # 随机选择部分组进行公平性损失计算
                #selected_groups = random.sample(list(unique_groups.numpy()), min(len(unique_groups), int(len(unique_groups)/16)))  # num_groups_to_sample 是要选择的组数
                group_fairness_losses = []
                #print("groups: ", unique_groups)

                #print("selected_groups: ", selected_groups)
                for gid in unique_groups:
                    group_mask = (group_id == gid)

                    if gid == 0:
                        # group_id = 0 的组跳过公平性损失
                        continue
                    
                    # 获取当前组的起始索引和组大小
                    group_start = groupindex[group_mask][0].item()  # 当前组的起始索引
                    group_sz = group_size[group_mask][0].item()     # 当前组的大小

                    if group_sz == 1 or group_sz == 2:
                        # 单个用户的组跳过公平性损失
                        continue

                    # 提取当前组的 fairness_id
                    group_users = [i for i in range(group_start, group_start + group_sz)]  # 组内用户的 fairness_id
                    #print("group_users: ", group_users)
                    
                    group_users = torch.tensor(group_users, dtype=torch.int64).to(device)
                    #print("group_users2: ", group_users)
                    # 随机抽取部分 theta 维度，保证同一组内一致
                    latent_dim = self.irt_net.theta.embedding_dim  # 获取 theta 的维度大小
                    k = max(1, latent_dim // 24 +3)  # 设置要抽取的维度数量，可以根据需要调整

                    # 设置随机种子，确保同一组内的随机选择一致
                    random.seed(int(gid.item()))
                    selected_dims = random.sample(range(latent_dim), k)

                    # 获取当前组的 theta 值
                    theta_group_full = self.irt_net.theta(group_users)  # shape: [group_sz, latent_dim]
                    theta_group_selected = theta_group_full[:, selected_dims]  # shape: [group_sz, k]

                    # 对选定的维度进行平均，得到每个用户的 theta 值
                    theta_group = theta_group_selected.mean(dim=1)  # shape: [group_sz]

                    # 计算公平性损失
                    predictions_reshaped = theta_group.view(1, -1)  # 模型预测 theta
                    targets_reshaped = group_users.view(1, -1)  # 目标 fairness_id

                    # 公平性损失计算：fairness_id 越低 theta 越高
                    #print("predictions_reshaped:",predictions_reshaped)
                    #print("targets_reshaped:",targets_reshaped)
                    fairness_loss_val = self.fairness_loss(predictions_reshaped, targets_reshaped)

                    if not torch.isnan(fairness_loss_val) and not torch.isinf(fairness_loss_val):
                        group_fairness_losses.append(fairness_loss_val)
                    
                # 合并 BCE 损失与公平性损失
                if group_fairness_losses:
                    #print("group_fairness_losses: ", group_fairness_losses)
                    total_fairness_loss =  torch.mean(torch.stack(group_fairness_losses))
                    #print("total_fairness_loss: ", total_fairness_loss)
                    #print("score_loss: ", score_loss)
                    loss = (1 - self.fairness_lambda) * score_loss + self.fairness_lambda * total_fairness_loss
                    fairness_losses.append(total_fairness_loss.item())
                    #print("fairness_loss: ", fairness_losses)
                else:
                    loss = score_loss

                # back propagation
                trainer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.irt_net.parameters(), max_norm=1.0)
                trainer.step()

                bce_losses.append(score_loss.mean().item())
                
            #print("fairnessloss: ", (fairness_losses))
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(bce_losses))))
            print("fairnessloss: %.6f" % (float(np.mean(fairness_losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d]  auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response, fairness_id, group_id, groupindex, group_size = batch_data
            
            item_id: torch.Tensor = item_id.to(device)
            fairness_id: torch.Tensor = fairness_id.to(device)
            pred: torch.Tensor = self.irt_net(fairness_id, item_id, fairness=False)
            response: torch.Tensor = response.to(device)
            loss = loss_function(pred, response)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        self.irt_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def generate(self, user_id, item_id, device="cpu"):
        self.irt_net = self.irt_net.to(device)
        user_id: torch.Tensor = user_id.to(device)
        item_id: torch.Tensor = item_id.to(device)
        pred: torch.Tensor = self.irt_net(user_id, item_id)
        return pred.tolist()

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
            fairness_id = fairness_id.to(device)
            """
            torch.tensor(groupid, dtype=torch.int64),
            torch.tensor(x, dtype=torch.int64),
            torch.tensor(y, dtype=torch.int64),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(fairnessid, dtype=torch.int64)
            """
            # Retrieve the ability (θ) parameter for the user
            theta = self.irt_net.theta(fairness_id).squeeze()
            theta_mean: torch.Tensor = theta.mean(dim=-1)

            # Add group_id, fairness_id, user_id, and corresponding θ value to the list
            for i, user in enumerate(fairness_id.cpu().numpy()):
                if (group_id[i].item(), user) not in processed_user_ids:
                    abilities.append([
                        int(group_id[i]),
                        int(fairness_id[i]),
                        int(user),
                        float(theta_mean[i].item())
                    ])
                    processed_user_ids.add((group_id[i].item(), user))  # Mark as processed

        # Save abilities to a CSV file with group_id and fairness_id
        df_abilities = pd.DataFrame(abilities, columns=["group_id", "fairness_id", "user_id", "theta"])
        df_abilities.sort_values(by=["group_id", "fairness_id"], inplace=True)  # Sort by group_id and fairness_id
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")

        self.irt_net.train()  # Switch back to training mode
