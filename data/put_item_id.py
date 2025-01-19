import pandas as pd
import ast

# 读取数据集
df = pd.read_csv('./a0910/origin_with_group_shuffled.csv')
item_df = pd.read_csv('./a0910/item.csv')  # 包含 'item_id' 和 'knowledge_code' 列

# 创建 item_id 到 knowledge_code 的映射
item_to_knowledge = item_df.set_index('item_id')['knowledge_code'].to_dict()

# 过滤掉 group_id 为 0 的行
df = df[df['group_id'] != 0]

# 初始化一个字典，用于存储每个 group_id 的知识点列表
group_knowledge = {}

# 按 group_id 分组
for group_id, group in df.groupby('group_id'):
    # 1. 找出 group 内重复的 item_id（公共题目）
    item_ids = group['item_id']
    duplicated_items = item_ids[item_ids.duplicated()].unique()
    
    # 2. 根据映射关系获取知识点
    knowledge_points = set()
    for item_id in duplicated_items:
        if item_id in item_to_knowledge:
            knowledge_codes = item_to_knowledge[item_id]
            # 如果映射的知识点是一个列表，合并所有元素到 set 中去重
            if isinstance(knowledge_codes, list):
                knowledge_points.update(knowledge_codes)
            else:
                knowledge_points.add(knowledge_codes)
    
    # 3. 将知识点列表中的字符串转为真实的列表对象，并进行去重
    knowledge_points_cleaned = set()
    for kp in knowledge_points:
        # 使用 ast.literal_eval 安全地将字符串列表转为真实的 Python 列表
        try:
            kp_list = ast.literal_eval(kp)
            if isinstance(kp_list, list):  # 确保转换为列表
                knowledge_points_cleaned.update(kp_list)
        except (ValueError, SyntaxError):
            pass  # 如果转换失败，跳过该项
    
    # 4. 对去重后的知识点列表进行排序
    group_knowledge[group_id] = sorted(list(knowledge_points_cleaned))

# 将计算好的知识点列表映射回原数据集
df['common_knowledge'] = df['group_id'].map(group_knowledge)

# 保存结果
df.to_csv('./a0910/origin_finial.csv', index=False)

print("处理完成，结果已保存到 ./a0910/origin_finial.csv")
