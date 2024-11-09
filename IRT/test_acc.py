# test_irt.py
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from IRT import IRT
from sklearn.model_selection import train_test_split


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

# 初始化IRT模型

model = IRT(27424, 17747)



# 加载模型
model.load("irt_model.pth")

# 评估模型
auc, accuracy = model.eval(test)
print("Test AUC: {}, Test Accuracy: {}".format(auc, accuracy))



#for name, param in model.irt_net.named_parameters():
 #   print(f"Name: {name}, Shape: {param.shape}, Values: {param.data}")

