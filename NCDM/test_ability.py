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

def get_available_gpus(limit=2):
    """
    Query available GPUs and return a list of GPU indices that are mostly free.
    :param limit: Max number of GPUs to use.
    :return: A comma-separated string of GPU indices.
    """
    try:
        # Use nvidia-smi to get GPU utilization
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                          '--format=csv,nounits,noheader'], encoding='utf-8')
        
        # Parse the result and find the most available GPUs
        gpu_info = result.strip().split('\n')
        available_gpus = []
        for gpu in gpu_info:
            index, memory_used, memory_total, utilization = gpu.split(', ')
            memory_used = int(memory_used)
            memory_total = int(memory_total)
            utilization = int(utilization)
            
            # Check if the GPU is considered "free" (low memory usage and low utilization)
            if memory_used < 2000 and utilization < 20:  # You can adjust these thresholds
                available_gpus.append(index)
        
        # Limit the number of GPUs to use
        if len(available_gpus) > limit:
            available_gpus = available_gpus[:limit]

        if available_gpus:
            # Set the environment variable to limit visible GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(available_gpus)
            print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print("No available GPUs found, using default device.")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to query GPUs: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Default to CPU if GPU query fails

# Call the function at the start of your script
get_available_gpus(limit=2)

# Your NCDM model and training code goes here


train_data = pd.read_csv("../data/a0910/all_virtual_user_data.csv")
valid_data = pd.read_csv("../data/a0910/all_virtual_user_data.csv")
test_data = pd.read_csv("../data/a0910/test.csv")
df_item = pd.read_csv("../data/a0910/item.csv")

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

batch_size = 64

user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))


def transform(origin_id, user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(origin_id, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


train_set= [
    transform(data["origin_id"],data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
    for data in [train_data]
]

def transform2(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

valid_set,test_set= [
    transform2(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
    for data in [valid_data,test_data]
]
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

cdm.extract_user_abilities(train_set, weighted=False, filepath="v_ability_parameters.csv")


