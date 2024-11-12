# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from MIRT import MIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

train_data = pd.read_csv("../data/a0910/all_virtual_user_data.csv")
valid_data = pd.read_csv("../data/a0910/virtual_user_valid_data.csv")
test_data = pd.read_csv("../data/a0910/test.csv")

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

cdm = MIRT(27424, 17747, 123)

cdm.load("mirt.params")
auc, accuracy = cdm.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

#存入文件，acc和accuracy
with open("test_acc.txt", "w") as f:
    f.write("MIRT: test auc: %.6f, accuracy: %.6f" % (auc, accuracy))
