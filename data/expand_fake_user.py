import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv("./a0910/updated_data.csv")

# 找到全局最大 user_id 和 fairness_id
global_user_id = data["user_id"].max()
global_fairness_id = 1

new_rows = []  # 用于存储新增的虚拟用户数据

# 定义向下翻转的函数
def flip_down(scores, num_flips):
    flipped_scores = scores.copy()
    one_indices = np.where(scores == 1)[0]
    if len(one_indices) == 0:
        return flipped_scores
    flip_indices = np.random.choice(one_indices, size=min(num_flips, len(one_indices)), replace=False)
    flipped_scores[flip_indices] = 0
    return flipped_scores

# 遍历每个 group（包括 group_id = 0）
for group_id, group_data in data.groupby("group_id"):
    user_count = group_data["user_id"].nunique()

    if group_id == 0:
        # group_id 为 0 的数据不扩充，只需重新分配唯一的 fairness_id
        group_0_data = group_data.copy()
        continue

    # 为所有用户（包括满 10 人的组）重新分配全局唯一的 fairness_id
    for user_id, user_data in group_data.groupby("user_id"):
        global_fairness_id += 1
        new_rows.extend([
            {
                "user_id": user_id,
                "item_id": row["item_id"],
                "score": row["score"],
                "group_id": group_id,
                "fairness_id": global_fairness_id,
            }
            for _, row in user_data.iterrows()
        ])

    # 如果当前组不满 10 人，则补充虚拟用户
    if user_count < 10:
        # 获取当前组内 fairness_id 最大的用户
        base_user = group_data[group_data["fairness_id"] == group_data["fairness_id"].max()]
        base_scores = base_user["score"].values
        base_items = base_user["item_id"].values
        flipped_scores = base_scores.copy()

        # 生成虚拟用户，补足到 10 人
        for _ in range(10 - user_count):
            global_user_id += 1
            global_fairness_id += 1

            # 向下翻转分数
            flipped_scores = flip_down(flipped_scores, num_flips=3)

            # 生成新用户数据
            new_rows.extend([
                {
                    "user_id": global_user_id,
                    "item_id": item,
                    "score": score,
                    "group_id": group_id,
                    "fairness_id": global_fairness_id,
                }
                for item, score in zip(base_items, flipped_scores)
            ])

# 将扩充数据保存为 DataFrame
expanded_data = pd.DataFrame(new_rows)

# 为 group_id=0 的数据重新分配全局唯一的 fairness_id
if 'group_0_data' in locals():
    group_0_data = group_0_data.copy()

    # 为每个 user_id 分配一个唯一的 fairness_id
    unique_user_ids = group_0_data["user_id"].unique()
    user_id_to_fairness_id = {}
    for user_id in unique_user_ids:
        global_fairness_id += 1
        user_id_to_fairness_id[user_id] = global_fairness_id

    # 更新 group_0_data 的 fairness_id
    group_0_data["fairness_id"] = group_0_data["user_id"].map(user_id_to_fairness_id)

    # 添加到 expanded_data
    expanded_data = pd.concat([expanded_data, group_0_data], ignore_index=True)

# 确保按 fairness_id 排序
expanded_data.sort_values("fairness_id", inplace=True)

# 保存结果
expanded_data.to_csv("./a0910/expanded_dataset.csv", index=False)
print("数据扩充完成，已保存到 expanded_dataset.csv")
