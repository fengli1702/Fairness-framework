# coding: utf-8
# 2021/4/1 @ WangFei
import logging
#from EduCDM import NCDM
from myNCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

import os
import subprocess

path = "../data/a0910/origin_with_group_shuffled.csv"
train_data = pd.read_csv(path)
valid_data = pd.read_csv("../data/a0910/valid.csv")
test_data = pd.read_csv("../data/a0910/test.csv")

df_item = pd.read_csv("../data/a0910/item.csv")



item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

batch_size = 32

user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))


def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) -1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) -1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


train_set, valid_set= [
    transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
    for data in [train_data, valid_data]
]

test_set = transform(test_data["user_id"], test_data["item_id"], item2knowledge, test_data["score"], batch_size)

logging.getLogger().setLevel(logging.INFO)
cdm = NCDM(knowledge_n, item_n, user_n)
cdm.train(train_set, valid_set, epoch=10, device="cuda")
print("train finished")
cdm.save("ncdm.snapshot")
print("save finished")
cdm.load("ncdm.snapshot")
print("load finished")
auc, accuracy = cdm.eval(test_set)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

with open("test_acc.txt", "a") as f:
    f.write("\n%s\n" % path)
    f.write("IRT: test auc: %.6f, accuracy: %.6f" % (auc, accuracy))



all_virtual_user_data = pd.read_csv('../data/a0910/origin_finial.csv')
all_virtual_user_data['common_knowledge'] = all_virtual_user_data['common_knowledge'].apply(ast.literal_eval)

def transform2(user_id, item_id, score, group_id, fairness_id, common_knowledge_list, item2knowledge, batch_size, **params):
    knowledge_emb = torch.zeros((len(item_id), knowledge_n))
    for idx in range(len(item_id)):
        knowledge_codes = item2knowledge.get(item_id.iloc[idx] + 1, [])
        knowledge_emb[idx][np.array(knowledge_codes) - 1] = 1.0

    # 将'common_knowledge_list'转换为张量
    common_knowledge_emb = torch.zeros((len(item_id), knowledge_n))
    for idx in range(len(item_id)):
        common_knowledge_codes = common_knowledge_list.iloc[idx]
        if common_knowledge_codes:
            common_knowledge_emb[idx][np.array(common_knowledge_codes) - 1] = 1.0

    dataset = TensorDataset(
        torch.tensor(user_id.values, dtype=torch.int64),
        torch.tensor(item_id.values, dtype=torch.int64),
        torch.tensor(score.values, dtype=torch.float32),
        torch.tensor(group_id.values, dtype=torch.int64),
        torch.tensor(fairness_id.values, dtype=torch.int64),
        knowledge_emb,
        common_knowledge_emb
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **params)

# 准备用于提取能力参数的测试数据集
test_fairness = transform2(
    all_virtual_user_data["user_id"],
    all_virtual_user_data["item_id"] - 1,
    all_virtual_user_data["score"],
    all_virtual_user_data["group_id"],
    all_virtual_user_data["fairness_id"] - 1,
    all_virtual_user_data["common_knowledge"],
    item2knowledge,
    batch_size
)

# 提取用户能力参数
cdm.extract_ability_parameters(test_fairness, filepath="v_ability_parameters.csv")