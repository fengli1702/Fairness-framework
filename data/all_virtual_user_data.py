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
virtual_user_counter = 0  # 用于生成新的虚拟 user_id

# 遍历所有真实用户
for real_user_id in real_user_ids:
    user_data = train_data[train_data['user_id'] == real_user_id]
    user_items = user_data['item_id'].values
    user_scores = user_data['score'].values

    # 保留一个原始的用户分数向量用于后续操作
    current_upward_scores = user_scores.copy()

    virtual_user_counter+=5

    # Reverse the order of storing for upward flips
    for i in range(5, 0, -1):
        current_upward_scores = flip_up(current_upward_scores, random.randint(1, 3))
        all_virtual_scores.extend([
            [real_user_id, virtual_user_counter, item, score]
            for item, score in zip(user_items, current_upward_scores)
        ])
        virtual_user_counter -= 1

    virtual_user_counter+=6

    # 添加真实用户
    all_virtual_scores.extend([
        [real_user_id, virtual_user_counter, item, score]
        for item, score in zip(user_items, user_scores)
    ])
    virtual_user_counter += 1

    # 保留一个原始的用户分数向量用于后续操作
    current_downward_scores = user_scores.copy()

    # 生成5个单调向下的虚拟用户
    for i in range(5):
        current_downward_scores = flip_down(current_downward_scores, random.randint(1, 3))
        all_virtual_scores.extend([
            [real_user_id, virtual_user_counter, item, score]
            for item, score in zip(user_items, current_downward_scores)
        ])
        virtual_user_counter += 1

    virtual_user_counter -= 1

# After generating all virtual user data
df_all_virtual_scores = pd.DataFrame(all_virtual_scores, columns=["origin_id", "user_id", "item_id", "score"])

# Sort by virtual_user_id before saving
df_all_virtual_scores = df_all_virtual_scores.sort_values(by=['user_id'])

# Save the sorted data to the CSV file
df_all_virtual_scores.to_csv('./a0910/all_virtual_user_data.csv', index=False)

print('All virtual scores saved to all_virtual_user_data.csv in sorted order by virtual_user_id')

def split_data(file_path, train_size=0.7, valid_size=0.15, test_size=0.15):
    # Ensure that the sizes add up to 1
    assert train_size + valid_size + test_size == 1.0, "Train, valid, and test sizes must sum to 1."
    
    # Read the data from CSV
    df = pd.read_csv(file_path)
    
    # Group by user_id to ensure all items for a user stay together
    grouped = df.groupby('user_id')

    # Get a list of user_ids
    user_ids = grouped.groups.keys()

    # Split the user_ids into train and temp (valid + test)
    train_user_ids, temp_user_ids = train_test_split(list(user_ids), test_size=(valid_size + test_size), random_state=42)
    
    # Further split temp user_ids into valid and test sets
    valid_user_ids, test_user_ids = train_test_split(temp_user_ids, test_size=test_size/(test_size + valid_size), random_state=42)
    
    # Create dataframes for each split and make explicit copies
    train_data = df[df['user_id'].isin(train_user_ids)].copy()
    valid_data = df[df['user_id'].isin(valid_user_ids)].copy()
    test_data = df[df['user_id'].isin(test_user_ids)].copy()

    # Save the splits into separate files
    train_data.sort_values(by="user_id", inplace=True)
    valid_data.sort_values(by="user_id", inplace=True)
    test_data.sort_values(by="user_id", inplace=True)

    train_data.to_csv('./a0910/virtual_user_train_data.csv', index=False)
    valid_data.to_csv('./a0910/virtual_user_valid_data.csv', index=False)
    test_data.to_csv('./a0910/virtual_user_test_data.csv', index=False)
    
    print("Data has been split and saved as virtual_user_train_data.csv, virtual_user_valid_data.csv, and virtual_user_test_data.csv.")

# Call the function to split the data
split_data('./a0910/all_virtual_user_data.csv')