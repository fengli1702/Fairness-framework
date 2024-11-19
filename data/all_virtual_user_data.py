import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# 读取train.csv数据
train_data = pd.read_csv('./a0910/train.csv')

# 获取所有真实用户的 user_id
real_user_ids = train_data['user_id'].unique()

# 定义翻转函数
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

# 初始化所有虚拟数据
all_virtual_scores = []
virtual_user_counter = 0+12465  # 用于生成新的虚拟 user_id

# 遍历所有真实用户
for real_user_id in real_user_ids:
    user_data = train_data[train_data['user_id'] == real_user_id]
    user_items = user_data['item_id'].values
    user_scores = user_data['score'].values

    # 保留一个原始的用户分数向量用于后续操作
    current_upward_scores = user_scores.copy()
    half_len = len(user_items) // 2
    user_items = user_items[:half_len]  # 只保留一半的 item
    user_scores = user_scores[:half_len]  # 只保留一半的 score

    virtual_user_counter+=2

    for i in range(2, 0, -1):
        # Flip only the second half of the scores
        current_upward_scores = flip_up(current_upward_scores, random.randint(1, 2))
        all_virtual_scores.extend([
            [real_user_id, virtual_user_counter, item, score]
            for item, score in zip(user_items, current_upward_scores)
        ])
        virtual_user_counter -= 1

    virtual_user_counter += 3

    # 添加真实用户
    all_virtual_scores.extend([
        [real_user_id, virtual_user_counter, item, score]
        for item, score in zip(user_items, user_scores)
    ])
    virtual_user_counter += 1

    # 保留一个原始的用户分数向量用于后续操作
    current_downward_scores = user_scores.copy()

    # 生成5个单调向下的虚拟用户 (flip only the second half)
    for i in range(2):
        current_downward_scores = flip_down(current_downward_scores, random.randint(1, 2))
        all_virtual_scores.extend([
            [real_user_id, virtual_user_counter, item, score]
            for item, score in zip(user_items, current_downward_scores)
        ])
        virtual_user_counter += 1

    virtual_user_counter -= 1

# After generating all virtual user data

df_all_virtual_scores = pd.DataFrame(all_virtual_scores, columns=["origin_id", "user_id", "item_id", "score"])
df_all_virtual_scores['is_train'] = 0

# Sort by virtual_user_id before saving
df_all_virtual_scores = df_all_virtual_scores.sort_values(by=['user_id'])

# Save the sorted data to the CSV file
df_all_virtual_scores.to_csv('./a0910/test_virtual.csv', index=False)


print('All virtual scores saved to all_virtual_user_data.csv in sorted order by virtual_user_id')

