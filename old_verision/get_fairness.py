import pandas as pd
import torch
import numpy as np

# 从 CSV 文件加载数据
def load_data(file_path):
    return pd.read_csv(file_path)

# NDCG计算相关函数
def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float))
    final = numerator / denominator
    return torch.sum(final)

def err_computation(score_rank, top_k):
    the_maxs = torch.max(score_rank).repeat(1, score_rank.shape[0])
    c = 2 * torch.ones_like(score_rank)
    score_rank = ((c.pow(score_rank) - 1) / c.pow(the_maxs))[0]
    the_ones = torch.ones_like(score_rank)
    new_score_rank = torch.cat((the_ones, 1 - score_rank))

    for i in range(score_rank.shape[0] - 1):
        score_rank = torch.mul(score_rank, new_score_rank[-score_rank.shape[0] - 1 - i : -1 - i])
    the_range = torch.arange(0., score_rank.shape[0]) + 1

    final = (1 / the_range[0:]) * score_rank[0:]
    return torch.sum(final)

def lambdas_computation(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # 排序
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    sigma_tuned = 1.0  # 示例值，根据需要设置
    length_of_k = min(top_k, y_sorted_scores.shape[1])  # 确保长度不超过 top_k
    y_sorted_scores = y_sorted_scores[:, 1: (length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1: (length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1: (length_of_k + 1)]
    
    # 确保我们不会在不足的情况下进行计算
    if y_sorted_scores.shape[0] < length_of_k:
        raise ValueError(f"y_sorted_scores的大小不足，需{length_of_k}个，实际为{y_sorted_scores.shape[0]}个")

    pairs_delta = torch.zeros(length_of_k, length_of_k, y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        if y_sorted_scores.shape[1] < length_of_k:  # 检查当前行是否足够
            print(f'Warning: Not enough elements for row {i}, expected {length_of_k}, got {y_sorted_scores.shape[1]}, skipping...')
            continue  # 跳过此行

        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(length_of_k, 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        valid_idx = y_sorted_idxs[i, :].clamp(max=x_similarity.shape[1]-1)  # 确保索引合法
        x_corresponding[i, :] = x_similarity[i, valid_idx]

    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])

    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(fraction_1[j, :, i]) - torch.sum(fraction_1[:, j, i]) 

    return lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding

# 计算每个origin_id组的平均损失
def compute_lambda(df):
    groups = df.groupby('origin_id')
    avg_losses = []
    
    for origin_id, group in groups:
        group = group.sort_values('user_id').reset_index(drop=True)
        
        # 检查组内成员数量
        if len(group) < 11:
            print(f'Warning: origin_id {origin_id} has only {len(group)} members, skipping...')
            continue
        
        user_ids = group['user_id'].values
        theta = group['theta'].values
        
        # 生成相似性矩阵
        similarity = torch.tensor([[1 if i == j else 0 for j in range(len(user_ids))] for i in range(len(user_ids))], dtype=torch.float32)
        
        # 设置top_k为每个组的大小
        top_k = len(user_ids)

        lambdas, _, _, _ = lambdas_computation(similarity, similarity, top_k)
        
        # 计算损失
        loss = lambdas.sum().item()  # 示例: 总和作为损失
        avg_losses.append(loss)
    
    return avg_losses

# 主函数
def main():
    file_path = 'v_ability_parameters.csv'  # 请修改为实际的文件路径
    data = load_data(file_path)
    
    avg_losses = compute_lambda(data)
    
    # 计算平均损失
    overall_avg_loss = sum(avg_losses) / len(avg_losses) if avg_losses else 0
    print(f'Average Loss across all origin_ids: {overall_avg_loss}')

if __name__ == '__main__':
    main()
