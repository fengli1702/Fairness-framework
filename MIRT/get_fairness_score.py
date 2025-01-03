# -*- coding: utf-8 -*-
"""
公平性度量代码.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 读取文件
file_path = "v_ability_parameters.csv"
df = pd.read_csv(file_path)

def dcg(ranks, top_k):
    gains = np.power(2, ranks[:top_k]) - 1  # 增益值
    discounts = np.log2(np.arange(2, top_k + 2))  # 折扣因子
    return np.sum(gains / discounts)

def ndcg(true_ranks, pred_ranks, top_k):
    # 将相关性映射到排名位置（从高到低的相关性分数）
    true_relevance = np.argsort(-np.array(true_ranks)) + 1  # 理想排序分数
    pred_relevance = np.argsort(-np.array(pred_ranks)) + 1  # 预测排序分数

    # 计算 DCG 和 IDCG
    dcg_pred = dcg(pred_relevance, top_k)
    idcg_true = dcg(true_relevance, top_k)
    
    # 返回归一化 NDCG
    return dcg_pred / idcg_true if idcg_true > 0 else 0

# 定义Kendall Tau Distance计算函数
def kendall_tau_distance(true_ranks, pred_ranks):
    discordant_pairs = 0
    for i in range(len(true_ranks)):
        for j in range(i + 1, len(true_ranks)):
            if (true_ranks[i] < true_ranks[j]) != (pred_ranks[i] < pred_ranks[j]):
                discordant_pairs += 1
    return discordant_pairs

# 定义Spearman's Rank Correlation Coefficient计算函数
def spearman_rho(true_ranks, pred_ranks):
    return spearmanr(true_ranks, pred_ranks).correlation

# 定义Cosine Similarity计算函数
def cosine_similarity(true_ranks, pred_ranks):
    true_vec = np.array(true_ranks) - 1
    pred_vec = np.array(pred_ranks) - 1
    return np.dot(true_vec, pred_vec) / (np.linalg.norm(true_vec) * np.linalg.norm(pred_vec))

# 定义Mean Reciprocal Rank计算函数
def mean_reciprocal_rank(true_ranks, pred_ranks):
    sorted_pred_indices = np.argsort(pred_ranks)
    reciprocal_ranks = []

    for true_rank in true_ranks:
        rank_position = np.where(sorted_pred_indices == (true_rank - 1))[0]
        if rank_position.size > 0:
            reciprocal_ranks.append(1 / (rank_position[0] + 1))
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

# 过滤掉 group_id=0 的数据
df = df[df['group_id'] != 0]
df = df[df['user_id'] <= 4128 ]

# 中心化 fairness_id 和 theta
def centerize_ranks(df):
    df['centerized_fairness_id'] = df.groupby('group_id')['fairness_id'].rank(method='min').astype(int)
    df['centerized_theta'] = df.groupby('group_id')['theta'].rank(method='max', ascending=False).astype(int)
    return df

df = centerize_ranks(df)

# 初始化公平性指标的列表
ndcgs = []
kendall_tau_distances = []
spearman_rhos = []
cosine_similarities = []
mean_reciprocal_ranks = []

# 计算每个 group_id 的公平性指标
for group_id in df['group_id'].unique():
    group_data = df[df['group_id'] == group_id]
    true_ranks = group_data['centerized_fairness_id'].values
    pred_ranks = group_data['centerized_theta'].values

    ndcg_score = ndcg(true_ranks, pred_ranks, top_k=len(true_ranks))
    ndcgs.append(ndcg_score)

    ktd = kendall_tau_distance(true_ranks, pred_ranks)
    kendall_tau_distances.append(ktd)

    srho = spearman_rho(true_ranks, pred_ranks)
    spearman_rhos.append(srho)

    cos_sim = cosine_similarity(true_ranks, pred_ranks)
    cosine_similarities.append(cos_sim)

    mrr = mean_reciprocal_rank(true_ranks, pred_ranks)
    mean_reciprocal_ranks.append(mrr)

# 计算平均公平性指标
average_ndcg = np.mean(ndcgs)
average_ktd = np.mean(kendall_tau_distances)
average_srho = np.mean(spearman_rhos)
average_cos_sim = np.mean(cosine_similarities)
average_mrr = np.mean(mean_reciprocal_ranks)

# 打印平均公平性指标
print("filename: ", file_path)
print("Average NDCG Higher is better :", average_ndcg)
print("Average Kendall Tau Distance Lower is better:", average_ktd)
print("Average Spearman's Rank Correlation Coefficient Higher is better:", average_srho)
print("Average Cosine Similarity Higher is better  :", average_cos_sim)
print("Average Mean Reciprocal Rank Higher is better:", average_mrr)

# 保存结果到文件
output_path = file_path.replace("v_ability_parameters.csv", "fairness_score.txt")
with open(output_path, "a") as f:
    f.write("\nGroup-level Fairness Metrics:\n")
    f.write("Average NDCG: %.6f\n" % average_ndcg)
    f.write("Average Kendall Tau Distance: %.6f\n" % average_ktd)
    f.write("Average Spearman's Rank Correlation Coefficient: %.6f\n" % average_srho)
    f.write("Average Cosine Similarity: %.6f\n" % average_cos_sim)
    f.write("Average Mean Reciprocal Rank: %.6f\n" % average_mrr)

print(f"Fairness metrics saved to {output_path}")
