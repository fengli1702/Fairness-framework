import pandas as pd

# 读取数据文件
data = pd.read_csv("./a0910/expanded_dataset.csv")

def check_fairnessid_order(group_data, max_group_size=10):
    """
    检查 group 内用户是否按照 fairnessid 强排序，并检查用户数量是否超限。
    """
    # 按 fairness_id 排序
    sorted_group = group_data.sort_values("fairness_id")
    user_ids = sorted_group["user_id"].unique().tolist()  # 只保留唯一用户

    item_scores = {
        user_id: sorted_group[sorted_group["user_id"] == user_id][["item_id", "score"]]
        for user_id in user_ids
    }

    # 遍历用户对，检查强排序条件
    for i in range(len(user_ids) - 1):
        user_i = user_ids[i]
        user_j = user_ids[i + 1]

        scores_i = item_scores[user_i]
        scores_j = item_scores[user_j]

        # 获取两用户的共同题目
        common_items = set(scores_i["item_id"]).intersection(set(scores_j["item_id"]))
        if not common_items:
            continue  # 无共同题目，跳过检查

        # 筛选共同题目的分数并对齐索引
        common_scores_i = scores_i[scores_i["item_id"].isin(common_items)].set_index("item_id")["score"]
        common_scores_j = scores_j[scores_j["item_id"].isin(common_items)].set_index("item_id")["score"]
        common_scores_i = common_scores_i.reindex(sorted(common_items))
        common_scores_j = common_scores_j.reindex(sorted(common_items))

        # 比较分数，检查是否满足强排序
        if not (all(common_scores_i >= common_scores_j) or all(common_scores_i <= common_scores_j)):
            # 如果不满足强排序条件，返回错误信息
            return False, user_i, user_j, list(common_items)

    # 检查 group 是否超过最大长度限制
    if len(user_ids) > max_group_size:
        return False, None, None, "Group size exceeds maximum length"
    if len(user_ids) < max_group_size:
        return False, None, None, "Group size less than maximum length"
    if len(user_ids) < 5:
        global count_none
        count_none += 1
    return True, None, None, None


# 打印检测结果
count_none = 0
count = 0
def print_group_check_details(group_id, user_id, next_user_id, issue):
    
    if issue == "Group size exceeds maximum length":
        print(f"Group {group_id}: 超过最大长度限制（10人）")
        pass
    elif (issue == "Group size less than maximum length"):
        print(f"Group {group_id}: 未达到最大长度限制（10人）")
        global count 
        count += 1
    else:
        print(f"Group {group_id}: 用户 {user_id} 和 {next_user_id} 在共同题目 {issue} 上不满足强排序条件")

# 检测所有 group
def validate_groups(data, max_group_size=10):
    """
    检查所有 group 内用户是否按照 fairness_id 强排序，同时是否满足最大长度限制。
    """
    group_ids = data["group_id"].unique()

    for group_id in group_ids:
        # 获取该组的用户数据，跳过 fairness_id = 0 的用户
        if group_id == 0:
            continue  # 跳过未分组用户
        group_data = data[data["group_id"] == group_id]
        # 检查该组是否满足排序和长度要求
        is_ordered, user_id, next_user_id, issue = check_fairnessid_order(group_data, max_group_size)
        if is_ordered:
            pass
            #print(f"Group {group_id}: 用户分数和排序满足要求")
        else:
            print_group_check_details(group_id, user_id, next_user_id, issue)

# 执行检测
validate_groups(data)
