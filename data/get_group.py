import pandas as pd

# 示例数据
data = pd.read_csv("./a0910/updated_data.csv")

# 按 group_id 聚合出每个组的 user_id 列表
group_user_mapping = data.groupby("group_id")["user_id"].unique().to_dict()

# 增加一列 get_group，内容为同一组的 user_id 列表
data["get_group"] = data["group_id"].map(group_user_mapping)

# 对 group_id=0 的行，将 get_group 列设置为空列表 []
data.loc[data["group_id"] == 0, "get_group"] = data[data["group_id"] == 0].apply(lambda _: [5000], axis=1)

# 保存到 CSV 文件
data.to_csv("./a0910/origin_with_group.csv", index=False)

print("数据已保存到 output_with_group.csv 文件中！")
