import pandas as pd

# 加载数据集
mapping_data = pd.read_csv("./a0910/extand_with_group_updated.csv")
test_data = pd.read_csv("./a0910/valid.csv")

# 构建 user_id 到 fairness_id 的映射
user_to_fairness = dict(zip(mapping_data["user_id"], mapping_data["fairness_id"]))

# 映射 test 数据集中的 fairness_id
test_data["fairness_id"] = test_data["user_id"].map(user_to_fairness)

# 检查映射结果
#print(test_data)

# 保存到文件
test_data.to_csv("./a0910/valid_with_fairness_id.csv", index=False)
