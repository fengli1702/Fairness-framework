# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from myMIRT import MIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

train_data = pd.read_csv("../data/a0910/all_virtual_user_data.csv")
valid_data = pd.read_csv("../data/a0910/all_virtual_user_data.csv")
test_data = pd.read_csv("../data/a0910/all_virtual_user_data.csv")

batch_size = 256


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = MIRT(27424, 17747, 123 , a_range=3)

cdm.train(train, valid, epoch=10)
cdm.save("mirt.params")

cdm.load("mirt.params")
auc, accuracy = cdm.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


all_virtual_user_data = pd.read_csv('../data/a0910/all_virtual_user_data.csv')

# Transform function to include origin_id
def transform2(x, y, z, origin_ids, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(origin_ids, dtype=torch.int64),
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

# Prepare test_fairness dataset from all_virtual_user_data
test_fairness = transform2(
    all_virtual_user_data["user_id"], 
    all_virtual_user_data["item_id"], 
    all_virtual_user_data["score"], 
    all_virtual_user_data["origin_id"],  # Include original_id
    batch_size=256
)

# Assuming `model` is your IRT model instance
cdm.extract_ability_parameters(test_data=test_fairness, filepath="v_ability_parameters.csv")
