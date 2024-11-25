import pandas as pd

# 创建数据集
data = pd.read_csv("./a0910/train_with_groups_and_fairness_strict.csv")

# 按 group_id 统计 user_id 的唯一个数
group_user_count = data.groupby('group_id')['user_id'].nunique()

# 找出 user_id 个数为 0 的 group_id
empty_groups = group_user_count[group_user_count == 1].index

# 将空组的 group_id 标记为 0
data['group_id'] = data['group_id'].apply(lambda x: 0 if x in empty_groups else x)

# 保存结果
data.to_csv("./a0910/updated_data.csv", index=False)
