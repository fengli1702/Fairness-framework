import pandas as pd
import numpy as np
import os
import random

# 读取train.csv数据
train_data = pd.read_csv("./data/a0910/train.csv")

# 获取第一个人的 user_id，假设是 1615
original_user_id = 1615
user_data = train_data[train_data["user_id"] == original_user_id]

# 获取该人的 item_id 和 score 列
user_items = user_data["item_id"].values
user_scores = user_data["score"].values

def flip_up(scores, num_flips):#从0向上反转
    flipped_scores = scores.copy()
    zero_indices = np.where(scores == 0)[0]  # 找到所有0的索引
    if len(zero_indices) == 0:
        return flipped_scores  # 没有0可反转，返回原始向量
    flip_indices = np.random.choice(zero_indices, size=min(num_flips, len(zero_indices)), replace=False)
    flipped_scores[flip_indices] = 1
    return flipped_scores

def flip_down(scores, num_flips):
    flipped_scores = scores.copy()
    one_indices = np.where(scores == 1)[0]  # 找到所有1的索引
    if len(one_indices) == 0:
        return flipped_scores  # 没有1可反转，返回原始向量
    flip_indices = np.random.choice(one_indices, size=min(num_flips, len(one_indices)), replace=False)
    flipped_scores[flip_indices] = 0
    return flipped_scores

# 保存虚拟数据记录的函数
def save_virtual_scores(original_user_id, item_ids, original_scores, flipped_up_list, flipped_down_list, filename_data, filename_order):
    virtual_scores = []
    ordering = []
    
    # 用于生成新的 user_id 的计数器
    virtual_user_id_counter = original_user_id + 1
    
    # 生成虚拟数据，前半部分为n1(反转0到1)，中间为真实数据，后半部分为n2(反转1到0)
    flip_count = 0
    for scores in flipped_up_list[::-1]:  # 按反转次数从多到少排列
        current_virtual_user_id = virtual_user_id_counter
        for i, score in enumerate(scores):
            virtual_scores.append([current_virtual_user_id, item_ids[i], score])
        ordering.append(current_virtual_user_id)  # 记录反转后的 user_id
        virtual_user_id_counter += 1
        flip_count += 1

    # 添加原始数据，放在中间
    for i, score in enumerate(original_scores):
        virtual_scores.append([original_user_id, item_ids[i], score])
    ordering.append(original_user_id)  # 原始数据 user_id
    
    # 向下反转的数据
    flip_count = 0
    for scores in flipped_down_list:
        current_virtual_user_id = virtual_user_id_counter
        for i, score in enumerate(scores):
            virtual_scores.append([current_virtual_user_id, item_ids[i], score])
        ordering.append(current_virtual_user_id)  # 记录反转后的 user_id
        virtual_user_id_counter += 1
        flip_count += 1

    # 将虚拟数据写入 CSV 文件
    df_data = pd.DataFrame(virtual_scores, columns=["user_id", "item_id", "score"])
    df_data.to_csv(filename_data, index=False)

    # 将排序写入 CSV 文件
    df_order = pd.DataFrame(ordering, columns=["user_id"])
    df_order.to_csv(filename_order, index=False)

# 生成虚拟向量并控制总数不超过10个
max_flips = 10 // 2  # 最多生成 10 个数据，n1 和 n2 各不超过5次


# 向上反转，形成 n1
n1_up_list = []
current_scores_up = user_scores.copy()
for i in range(max_flips):
    num_flips_up= random.randint(1, 3)  # 随机选择反转的数量
    current_scores_up = flip_up(current_scores_up, num_flips_up)
    n1_up_list.append(current_scores_up.copy())
    if len(n1_up_list) >= max_flips:
        break

# 向下反转，形成 n2
n2_down_list = []
current_scores_down = user_scores.copy()
for i in range(max_flips):
    num_flips_down= random.randint(1, 3)  # 随机选择反转的数量
    current_scores_down = flip_down(current_scores_down, num_flips_down)
    n2_down_list.append(current_scores_down.copy())
    if len(n2_down_list) >= max_flips:
        break

# 保证总虚拟数据数量不超过10个
total_virtual_count = len(n1_up_list) + len(n2_down_list) + 1  # 包括原始数据
if total_virtual_count > 10:
    extra_count = total_virtual_count - 10
    if len(n1_up_list) > len(n2_down_list):
        n1_up_list = n1_up_list[:-extra_count]
    else:
        n2_down_list = n2_down_list[:-extra_count]

# 保存结果到CSV文件
output_data_filename = os.path.join("./data/a0910/", "virtual_user_data.csv")
output_order_filename = os.path.join("./data/a0910/", "virtual_user_order.csv")
save_virtual_scores(original_user_id, user_items, user_scores, n1_up_list, n2_down_list, output_data_filename, output_order_filename)

print(f"Virtual scores saved to {output_data_filename}")
print(f"Virtual data order saved to {output_order_filename}")
