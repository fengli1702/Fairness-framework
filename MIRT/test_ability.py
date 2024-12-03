# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from myMIRT import MIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
#../data/a0910/origin_with_group_updated.csv
path = "../data/a0910/origin_with_group_shuffled.csv"
train_data = pd.read_csv(path)
valid_data = pd.read_csv("../data/a0910/valid.csv")
test_data = pd.read_csv("../data/a0910/test.csv")

batch_size = 256


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params,shuffle=True)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = MIRT(4129, 17747, 123,  a_range= 1)

cdm.train(train, valid, epoch=15, device="cuda")
cdm.save("mirt.params")
#
cdm.load("mirt.params")
auc, accuracy = cdm.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

with open("test_acc.txt", "a") as f:
    f.write("\n%s\n" % path)
    f.write("IRT: test auc: %.6f, accuracy: %.6f" % (auc, accuracy))




all_virtual_user_data = pd.read_csv('../data/a0910/origin_with_group_updated.csv')

# Transform function to include origin_id
def transform2(x, y, z, groupid,fairnessid, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(groupid, dtype=torch.int64),
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(fairnessid, dtype=torch.int64)
    )
    return DataLoader(dataset, batch_size=batch_size, **params, shuffle=False)

# Prepare test_fairness dataset from all_virtual_user_data
"""user_id,item_id,score,group_id,fairness_id,get_group
127,15222,1,1,1,[ 127  144  593  718 1018 1057 1117 1130 1615 2099]
127,9000,1,1,1,[ 127  144  593  718 1018 1057 1117 1130 1615 2099]"""

test_fairness = transform2(all_virtual_user_data["user_id"], all_virtual_user_data["item_id"], all_virtual_user_data["score"]
                          ,all_virtual_user_data["group_id"], all_virtual_user_data["fairness_id"], batch_size)
# Assuming `model` is your IRT model instance
cdm.extract_ability_parameters(test_data=test_fairness, filepath="v_ability_parameters.csv")

