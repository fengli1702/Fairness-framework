import pandas as pd

# 读取两个数据集
train_data = pd.read_csv("trainfinial.csv")
test_data = pd.read_csv("test_finial_filtered.csv")

# 提取 user_id 和 fairness_id 的映射
train_mapping = train_data[['user_id', 'fairness_id']].drop_duplicates()
test_mapping = test_data[['user_id', 'fairness_id']].drop_duplicates()

# 检查映射一致性
merged_mapping = pd.merge(train_mapping, test_mapping, on='user_id', suffixes=('_train', '_test'), how='outer')

# 找到不一致的映射
inconsistent_mapping = merged_mapping[merged_mapping['fairness_id_train'] != merged_mapping['fairness_id_test']]

# 输出结果
if inconsistent_mapping.empty:
    print("User ID and fairness ID mappings are consistent between train and test datasets.")
else:
    print("Inconsistent mappings found:")
    print(inconsistent_mapping)

# 计算最大 fairness_id
max_fairness_id_train = train_data['fairness_id'].max()
max_fairness_id_test = test_data['fairness_id'].max()

print(f"Max fairness_id in train dataset: {max_fairness_id_train}")
print(f"Max fairness_id in test dataset: {max_fairness_id_test}")
