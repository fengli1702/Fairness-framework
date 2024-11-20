# -*- coding: utf-8 -*-
"""
公平性度量代码.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys

# Check if a file path argument was provided

# Read the file path from the command-line arguments
file_path = "v_ability_parameters.csv"
df = pd.read_csv(file_path)
# 定义DCG计算函数
def dcg(ranks, top_k=9):
    gains = np.power(2, ranks)[:top_k] - 1
    discounts = np.log2(np.arange(2, top_k + 2))
    return np.sum(gains / discounts)

# 定义NDCG计算函数
def ndcg(true_ranks, pred_ranks, top_k=9):
    dcg_true = dcg(np.argsort(true_ranks), top_k)
    dcg_pred = dcg(np.argsort(pred_ranks), top_k)
    idcg_true = dcg(np.arange(1, top_k + 1), top_k)
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
# Define the Mean Reciprocal Rank (MRR) calculation function
def mean_reciprocal_rank(true_ranks, pred_ranks):
    # Sort pred_ranks and get the sorted indices to determine the predicted rank order
    sorted_pred_indices = np.argsort(pred_ranks)
    reciprocal_ranks = []

    # Iterate through each true rank and find its position in the sorted predictions
    for true_rank in true_ranks:
        # Find where each true_rank appears within the sorted prediction
        rank_position = np.where(sorted_pred_indices == (true_rank - 1))[0]
        
        # If the rank_position is found, compute the reciprocal rank
        if rank_position.size > 0:
            reciprocal_ranks.append(1 / (rank_position[0] + 1))
        else:
            reciprocal_ranks.append(0)  # If true_rank is not in sorted_pred_indices

    # Compute and return the mean of all reciprocal ranks
    return np.mean(reciprocal_ranks)



# 中心化user_id和theta
def centerize_ranks(df):
    df['centerized_user_id'] = df.groupby('origin_id')['user_id'].rank(method='min').astype(int)
    df['centerized_theta'] = df.groupby('origin_id')['theta'].rank(method='max', ascending=False).astype(int)
    return df

# 应用中心化
df = centerize_ranks(df)

# 初始化公平性指标的列表
ndcgs = []
kendall_tau_distances = []
spearman_rhos = []
cosine_similarities = []
mean_reciprocal_ranks = []

# 计算每个组的公平性指标
for origin_id in df['origin_id'].unique():
    group_data = df[df['origin_id'] == origin_id]
    true_ranks = group_data['centerized_user_id'].values
    pred_ranks = group_data['centerized_theta'].values
    #print(f"Origin ID: {origin_id}")
    #print(f"True Ranks: {true_ranks}")
    #print(f"Pred Ranks: {pred_ranks}")

    ndcg_score = ndcg(true_ranks, pred_ranks, top_k=9)
    ndcgs.append(ndcg_score)
#
    ktd = kendall_tau_distance(true_ranks, pred_ranks)
    kendall_tau_distances.append(ktd)
#
    srho = spearman_rho(true_ranks, pred_ranks)
    spearman_rhos.append(srho)
#
    cos_sim = cosine_similarity(true_ranks, pred_ranks)
    cosine_similarities.append(cos_sim)

    #print(true_ranks, pred_ranks)

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

#将结果存入文件，路径按照file_path中的v_abilities_parameters.csv的同级目录
with open(file_path.replace("v_ability_parameters.csv", "fairness_score.txt"), "w") as f:
    f.write("Average NDCG: %.6f\n" % average_ndcg)
    f.write("Average Kendall Tau Distance: %.6f\n" % average_ktd)
    f.write("Average Spearman's Rank Correlation Coefficient: %.6f\n" % average_srho)
    f.write("Average Cosine Similarity: %.6f\n" % average_cos_sim)
    f.write("Average Mean Reciprocal Rank: %.6f\n" % average_mrr)

