import pandas as pd
import numpy as np
import os
import random

# 读取train.csv数据
train_data = pd.read_csv("./data/a0910/train.csv")

def flip_up(scores, num_flips):
    flipped_scores = scores.copy()
    zero_indices = np.where(scores == 0)[0]
    if len(zero_indices) == 0:
        return flipped_scores
    flip_indices = np.random.choice(zero_indices, size=min(num_flips, len(zero_indices)), replace=False)
    flipped_scores[flip_indices] = 1
    return flipped_scores

def flip_down(scores, num_flips):
    flipped_scores = scores.copy()
    one_indices = np.where(scores == 1)[0]
    if len(one_indices) == 0:
        return flipped_scores
    flip_indices = np.random.choice(one_indices, size=min(num_flips, len(one_indices)), replace=False)
    flipped_scores[flip_indices] = 0
    return flipped_scores

def generate_strictly_monotonic_scores(original_scores, flip_func, num_flips, direction):
    scores_list = []
    current_scores = original_scores.copy()
    for _ in range(num_flips):
        flipped_scores = flip_func(current_scores, random.randint(1, 3))
        # 确保单调性
        if direction == "up" and np.all(flipped_scores >= current_scores):
            scores_list.append(flipped_scores)
            current_scores = flipped_scores  # 更新当前分数
        elif direction == "down" and np.all(flipped_scores <= current_scores):
            scores_list.append(flipped_scores)
            current_scores = flipped_scores  # 更新当前分数
    return scores_list

def save_virtual_scores(original_user_id, item_ids, original_scores, flipped_up_list, flipped_down_list):
    virtual_scores = []
    
    # 添加虚拟用户
    for scores in flipped_up_list[::-1]:  # 从多到少排列
        for score in scores:
            virtual_scores.append([None, item_ids, score])  # None 暂时占位
    
    # 添加原始数据
    for score in original_scores:
        virtual_scores.append([original_user_id, item_ids, score])
    
    # 添加向下反转的数据
    for scores in flipped_down_list:
        for score in scores:
            virtual_scores.append([None, item_ids, score])  # None 暂时占位
    
    return virtual_scores

# 生成虚拟数据并确保每组有10个条目
virtual_user_data = []
virtual_user_id_counter = 1  # 从1开始的用户ID

for original_user_id in train_data["user_id"].unique():
    user_data = train_data[train_data["user_id"] == original_user_id]
    user_items = user_data["item_id"].values
    user_scores = user_data["score"].values

    n1_up_list = generate_strictly_monotonic_scores(user_scores, flip_up, 5, "up")  # 确保有5个向上反转
    n2_down_list = generate_strictly_monotonic_scores(user_scores, flip_down, 4, "down")  # 确保有4个向下反转
    
    # 保存虚拟分数
    virtual_scores = save_virtual_scores(original_user_id, user_items, user_scores, n1_up_list, n2_down_list)
    
    # 记录数据，并替换 user_id
    for virtual_user in virtual_scores:
        user_id = virtual_user_id_counter if virtual_user[0] is None else virtual_user[0]
        original_user_id_marker = original_user_id if user_id == original_user_id else f"fake_{original_user_id}"
        virtual_user_data.append([user_id, virtual_user[1], virtual_user[2], original_user_id_marker])
        
        if virtual_user[0] is None:
            virtual_user_id_counter += 1  # 仅对虚拟用户递增ID

# 生成DataFrame并保存结果
final_virtual_user_data = pd.DataFrame(virtual_user_data, columns=["user_id", "item_id", "score", "original_user_id"])
final_virtual_user_data.to_csv("./data/a0910/virtual_user_data.csv", index=False)

print("Virtual user data saved.")
