# coding: utf-8
import logging
from myNCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import ast  # 用于解析字符串列表

path = "../data/a0910/origin_finial.csv"
train_data = pd.read_csv(path)

# 将'common_knowledge'列的字符串转换为列表
train_data['common_knowledge'] = train_data['common_knowledge'].apply(ast.literal_eval)

# 对于valid_data和test_data，使用空列表代替'common_knowledge'列
valid_data = pd.read_csv("../data/a0910/valid_with_fairness_id_origin.csv")
valid_data['common_knowledge'] = [[] for _ in range(len(valid_data))]

test_data = pd.read_csv("../data/a0910/test_with_fairness_id_origin.csv")
test_data['common_knowledge'] = [[] for _ in range(len(test_data))]

df_item = pd.read_csv("../data/a0910/item.csv")

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id = s['item_id']
    knowledge_codes = list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

batch_size = 32

user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))


def transform(x, y, z, n, k, j, i, common_knowledge_list, item2knowledge, batch_size, **params):
    knowledge_emb = torch.zeros((len(y), knowledge_n))
    for idx in range(len(y)):
        knowledge_codes = item2knowledge.get(y.iloc[idx], [])
        knowledge_emb[idx][np.array(knowledge_codes) - 1] = 1.0

    # 将'common_knowledge_list'转换为张量
    common_knowledge_emb = torch.zeros((len(y), knowledge_n))
    for idx in range(len(y)):
        common_knowledge_codes = common_knowledge_list.iloc[idx]
        if common_knowledge_codes:
            common_knowledge_emb[idx][np.array(common_knowledge_codes) - 1] = 1.0

    dataset = TensorDataset(
        torch.tensor(x.values, dtype=torch.int64) - 1,
        torch.tensor(y.values, dtype=torch.int64) - 1,
        knowledge_emb,
        torch.tensor(z.values, dtype=torch.float32),
        torch.tensor(n.values, dtype=torch.int64),
        torch.tensor(k.values, dtype=torch.int64),
        torch.tensor(j.values, dtype=torch.int64),
        torch.tensor(i.values, dtype=torch.int64),
        common_knowledge_emb
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **params)


# 准备训练和验证数据集
train_set = transform(
    train_data["user_id"],
    train_data["item_id"],
    train_data["score"],
    train_data["fairness_id"],
    train_data["group_id"],
    train_data["groupindex"],
    train_data["group_size"],
    train_data["common_knowledge"],
    item2knowledge,
    batch_size
)

valid_set = transform(
    valid_data["user_id"],
    valid_data["item_id"],
    valid_data["score"],
    valid_data["fairness_id"],
    pd.Series([0] * len(valid_data)),
    pd.Series([0] * len(valid_data)),
    pd.Series([0] * len(valid_data)),
    valid_data["common_knowledge"],  # 空列表
    item2knowledge,
    batch_size
)

test_set = transform(
    test_data["user_id"],
    test_data["item_id"],
    test_data["score"],
    test_data["fairness_id"],
    pd.Series([0] * len(test_data)),
    pd.Series([0] * len(test_data)),
    pd.Series([0] * len(test_data)),
    test_data["common_knowledge"],  # 空列表
    item2knowledge,
    batch_size
)

logging.getLogger().setLevel(logging.INFO)
cdm = NCDM(knowledge_n, item_n, user_n)
cdm.train(train_set, valid_set, epoch=10, device="cuda")
cdm.save("ncdm.snapshot")
auc, accuracy = cdm.eval(test_set)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

# 存入文件，auc和accuracy
with open("test_acc.txt", "a") as f:
    f.write("\n path: %s\n" % path)
    f.write("NCDM_final : test auc: %.6f, accuracy: %.6f\n" % (auc, accuracy))

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