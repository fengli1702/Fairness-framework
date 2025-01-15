import pandas as pd

# 读取数据集
train_data = pd.read_csv("trainfinial.csv")
test_data = pd.read_csv("testfinial.csv")

# 提取 user_id 和 fairness_id 的映射
train_mapping = train_data[['user_id', 'fairness_id']].drop_duplicates()
test_mapping = test_data[['user_id', 'fairness_id']].drop_duplicates()

# 合并映射，查找不一致的条目
merged_mapping = pd.merge(
    test_mapping, train_mapping, on='user_id', suffixes=('_test', '_train'), how='left'
)

# 找到与 train 中不一致的条目
inconsistent_user_ids = merged_mapping[
    (merged_mapping['fairness_id_train'].isna()) |  # 不在 train 中
    (merged_mapping['fairness_id_test'] != merged_mapping['fairness_id_train'])  # 映射不同
]['user_id']

# 从 test 集合中删除不一致的条目
filtered_test_data = test_data[~test_data['user_id'].isin(inconsistent_user_ids)]

# 保存过滤后的数据集
filtered_test_data.to_csv("test_finial_filtered.csv", index=False)
print(f"Filtered test data saved to 'test_finial_filtered.csv'")
