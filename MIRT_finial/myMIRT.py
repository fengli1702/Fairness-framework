# coding: utf-8
# 2021/7/1 @ tongshiwei


import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from loss import FairnessLoss

if torch.cuda.is_available():

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")


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
    def __init__(self, user_num, item_num, latent_dim, a_range, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        self.a_range = a_range

    def forward(self, user, item , fairness):
        theta = torch.squeeze(self.theta(user), dim=-1)
        theta = torch.sigmoid(theta)
        #求平均返回theta
        if fairness:
            theta = torch.mean(theta, dim=-1)
            return theta
        
        a = torch.squeeze(self.a(item), dim=-1)

        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(item), dim=-1)
        b = torch.sigmoid(b)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(theta, a, b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


class MIRT(CDM):
    def __init__(self, user_num, item_num, latent_dim, a_range=None , zeta=0.9, group_size = 11 , use_fairness = True):
        super(MIRT, self).__init__()
        self.irt_net = MIRTNet(user_num, item_num, latent_dim, a_range)
        self.zeta = zeta
        self.groupsize = group_size
        self.use_fairness = use_fairness
    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.irt_net = self.irt_net.to(device)
        score_loss_function = nn.BCELoss()
        fairness_loss = FairnessLoss()

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []

            epoch_score_losses = []
            epoch_fairness_losses = []

            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.irt_net(user_id, item_id, fairness = False)
                response: torch.Tensor = response.to(device)
                score_loss = score_loss_function(predicted_response, response)

                
                #下面算pair
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

                if self.use_fairness:
                    group_fairness_losses = []

                    for i in range(len(batch_data)):
                        group_user_ids = pair_id_tensor[i]  # Get user IDs for the current group

                        # Call `self.irt_net` to get predicted_response and theta for this group
                        theta_group = self.irt_net(group_user_ids, item_id, fairness = True)
                        #print("theta_group: ",theta_group.shape)  # torch.Size([1, 11]) 是一个group
                        #print("theta_group: ",theta_group)

                        # Reshape predictions and targets to match FairnessLoss input requirements
                        predictions_reshaped = theta_group.view(1, -1)  # Reshaping for a single group

                        # Reshape targets_rank as needed for fairness loss calculation
                        targets_reshaped = group_user_ids.view(1, -1)

                        # Calculate the fairness loss for the current group
                        fairness_loss_val = fairness_loss(predictions_reshaped, targets_reshaped)

                        # Append the fairness loss for this group to the list
                        group_fairness_losses.append(fairness_loss_val)

                    # Optionally, average or sum the losses across the batch
                    total_fairness_loss = torch.mean(torch.stack(group_fairness_losses))  # Or use sum() if you prefer summing
                    
                    # 组合两种loss
                    #print("total_fairness_loss: ",total_fairness_loss)
                    #print("score_loss: ",score_loss)
                    loss = (1 - self.zeta) * score_loss + self.zeta * total_fairness_loss
                    epoch_fairness_losses.append(total_fairness_loss.item())
                    epoch_score_losses.append(score_loss.item())

                else:
                    loss = score_loss

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())

            print("[Epoch %d] LogisticLoss: %.6f , score loss: %f ,fairness loss: %f"
                   % (e, float(np.mean(losses)), np.mean(epoch_score_losses), np.mean(epoch_fairness_losses)))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id, fairness = False)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.irt_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

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
        processed_user_ids = set()  # To track processed user_ids

        for batch_data in tqdm(test_data, "Extracting abilities"):
            # Unpack the batch_data, including origin_id
            origin_id, user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)

            # Retrieve the multidimensional ability (θ) parameter for the user
            theta: torch.Tensor = self.irt_net.theta(user_id).squeeze()

            # Calculate the mean of the multidimensional theta
            theta_mean: torch.Tensor = theta.mean(dim=-1)

            # Retrieve the discrimination (a) parameter for the items (if needed)
            a: torch.Tensor = self.irt_net.a(item_id).squeeze()

            # Add user_id, corresponding averaged θ value, and origin_id to the list
            for i, user in enumerate(user_id.cpu().numpy()):
                if user not in processed_user_ids:
                    abilities.append([int(origin_id[i]), int(user), float(theta_mean[i].item())])
                    processed_user_ids.add(user)  # Mark user_id as processed

        # Save abilities to a CSV file with origin_id
        df_abilities = pd.DataFrame(abilities, columns=["origin_id", "user_id", "theta_avg"])
        df_abilities.sort_values(by="user_id", inplace=True)  # Sort by user_id
        df_abilities.to_csv(filepath, index=False)
        print(f"Ability parameters saved to {filepath}")

        self.irt_net.train()  # Switch back to training mode
