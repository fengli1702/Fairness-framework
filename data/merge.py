import pandas as pd

# 读取两个 CSV 文件
test_virtual = pd.read_csv('./a0910/test_virtual.csv')
train_virtual = pd.read_csv('./a0910/train_virtual.csv')

# 合并两个 DataFrame
merged_data = pd.concat([test_virtual, train_virtual], ignore_index=True)

# 按 user_id 从低到高排序
merged_data = merged_data.sort_values(by=['user_id'])

# 保存合并后的数据到新的 CSV 文件
merged_data.to_csv('./a0910/merged_virtual.csv', index=False)

print('Merged data saved to merged_virtual.csv, sorted by user_id')
