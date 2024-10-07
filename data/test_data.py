import pandas as pd

# 读取数据文件
data = pd.read_csv("./data/a0910/all_virtual_user_data.csv")

# 获取前100个不同的 origin_id
unique_origin_ids = data["origin_id"].unique()[:100]

# 检查每个 virtual_user_id 对应的 item_id 的 score 是否满足条件
def check_scores(group_data):
    virtual_user_ids = group_data["user_id"].unique()
    for i in range(len(virtual_user_ids) - 1):
        for j in range(i + 1, len(virtual_user_ids)):
            user_i_scores = group_data[group_data["user_id"] == virtual_user_ids[i]][["item_id", "score"]]
            user_j_scores = group_data[group_data["user_id"] == virtual_user_ids[j]][["item_id", "score"]]
            
            # 检查 user_i 的每个 item_id 的 score 是否大于等于 user_j 的相同 item_id 的 score
            for _, row in user_i_scores.iterrows():
                item_id = row["item_id"]
                score_i = row["score"]
                score_j = user_j_scores[user_j_scores["item_id"] == item_id]["score"].values
                if len(score_j) > 0 and score_i < score_j[0]:
                    return False, virtual_user_ids[i], virtual_user_ids[j], item_id
    return True, None, None, None

# 打印不满足条件的详细信息
def print_unsorted_group_details(origin_id, group_number, user_id, next_user_id, item_id):
    print(f"Origin ID: {origin_id}, Group {group_number}, 用户 {user_id} 和 {next_user_id} 在 item_id {item_id} 上不满足条件")

# 遍历前100个不同的 origin_id
group_number = 1
for origin_id in unique_origin_ids:
    # 获取当前 origin_id 对应的所有数据
    origin_data = data[data["origin_id"] == origin_id]
    
    # 获取所有 unique 的 virtual_user_id
    unique_virtual_user_ids = origin_data["user_id"].unique()
    
    # 按每11个 virtual_user_id 一组进行检查
    for i in range(0, len(unique_virtual_user_ids), 11):
        group_virtual_user_ids = unique_virtual_user_ids[i:i+11]
        group_data = origin_data[origin_data["user_id"].isin(group_virtual_user_ids)]
        
        # 检查每个 virtual_user_id 对应的 item_id 的 score 是否满足条件
        is_strictly_ordered, user_id, next_user_id, item_id = check_scores(group_data)
        if is_strictly_ordered:
            print(f"Origin ID: {origin_id}, Group {group_number}, 用户分数严格排序")
        else:
            print_unsorted_group_details(origin_id, group_number, user_id, next_user_id, item_id)
    group_number += 1   

def compare_rankings(ranking1, ranking2):
    # 将排名转换为集合
    set1 = set(ranking1)
    set2 = set(ranking2)
    
    # 计算交集
    intersection = set1.intersection(set2)
    similarity = len(intersection) / (len(set1) + len(set2) - len(intersection))  # Jaccard相似度
    return similarity


def display_flipped_scores(file_path, original_user_id):
    data = pd.read_csv(file_path)
    
    # 获取翻转后的分数（假设在虚拟数据文件中）
    flipped_data = data  
    for user_id, group in flipped_data.groupby("user_id"):
        print(f"User ID: {user_id}")
        print(group[["item_id", "score"]])



# 比较两个排名
#ranking1 = [1, 2, 3]
#ranking2 = [2, 3, 4]
#similarity_score = compare_rankings(ranking1, ranking2)
#print(f"两个排名的相似度: {similarity_score:.2f}")
#
# 展示翻转效果
#display_flipped_scores("./data/a0910/virtual_user_data.csv",1615)
