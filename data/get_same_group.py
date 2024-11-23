import pandas as pd

# 读取数据文件
data = pd.read_csv("./a0910/train.csv")

def are_strictly_ordered(user1_scores, user2_scores):
    """
    检查 user1 和 user2 是否满足强排序关系。
    """
    # 找出共同的 item_id
    common_items = set(user1_scores["item_id"]).intersection(set(user2_scores["item_id"]))
    if not common_items:
        return False  # 如果没有共同题目，强排序关系不存在

    # 对共同题目进行比较
    user1_common_scores = user1_scores[user1_scores["item_id"].isin(common_items)].set_index("item_id")["score"]
    user2_common_scores = user2_scores[user2_scores["item_id"].isin(common_items)].set_index("item_id")["score"]

    user1_common_scores, user2_common_scores = user1_common_scores.align(user2_common_scores, join="inner")

    # 检查是否满足所有分数都 >= 或 <= 的条件
    return all(user1_common_scores >= user2_common_scores) or all(user1_common_scores <= user2_common_scores)

def assign_groups_with_strict_ordering(data, max_group_size=10):
    """
    按照强排序规则分组，并基于强排序分配组内排名 (fairness_id)。
    """
    user_ids = data["user_id"].unique()
    user_scores = {user_id: data[data["user_id"] == user_id][["item_id", "score"]] for user_id in user_ids}

    assigned_users = set()  # 已分配的用户
    groupid_mapping = {user_id: 0 for user_id in user_ids}  # 用户到组的映射
    group_id = 1  # 当前组号

    for user_id in user_ids:
        if user_id in assigned_users:
            continue

        # 当前组初始化
        current_group = [user_id]
        assigned_users.add(user_id)

        for candidate_id in user_ids:
            if candidate_id in assigned_users:
                continue

            # 检查与组内所有用户的强排序关系
            if all(are_strictly_ordered(user_scores[member_id], user_scores[candidate_id]) for member_id in current_group):
                # 添加到当前组
                current_group.append(candidate_id)
                assigned_users.add(candidate_id)

            if len(current_group) >= max_group_size:
                break

        # 分配 group_id
        for member_id in current_group:
            groupid_mapping[member_id] = group_id

        group_id += 1  # 增加组号

    return groupid_mapping


# 执行分组逻辑
groupid_mapping = assign_groups_with_strict_ordering(data)

# 添加 group_id 列
data["group_id"] = data["user_id"].map(groupid_mapping)

# 为每个 user_id 分配唯一的 global fairness_id，从 1 开始递增
unique_users = data[["user_id", "group_id"]].drop_duplicates().sort_values(by=["group_id", "user_id"]).reset_index(drop=True)
unique_users["fairness_id"] = range(1, len(unique_users) + 1)

# 将 `fairness_id` 映射回原数据
data = data.merge(unique_users[["user_id", "fairness_id"]], on="user_id", how="left")

# 最后按照 fairness_id 排序
data = data.sort_values(by="fairness_id").reset_index(drop=True)

# 保存结果
output_path = "./a0910/train_with_groups_and_fairness_strict.csv"
data.to_csv(output_path, index=False)

print(f"分组完成，结果已保存到 {output_path}")
