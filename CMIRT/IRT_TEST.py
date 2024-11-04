# coding: utf-8
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from IRT import IRT
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def load_data(file_path):
    return pd.read_csv(file_path)

def transform_data(data, batch_size, include_origin_id=False):
    if include_origin_id:
        # 用于提取能力参数时的数据格式
        dataset = TensorDataset(
            torch.tensor(data['origin_id'].values, dtype=torch.int64),
            torch.tensor(data['user_id'].values, dtype=torch.int64),
            torch.tensor(data['item_id'].values, dtype=torch.int64),
            torch.tensor(data['score'].values, dtype=torch.float32)
        )
    else:
        # 用于训练和评估时的数据格式
        dataset = TensorDataset(
            torch.tensor(data['user_id'].values, dtype=torch.int64),
            torch.tensor(data['item_id'].values, dtype=torch.int64),
            torch.tensor(data['score'].values, dtype=torch.float32)
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    # 配置参数
    batch_size = 256
    epochs = 6
    learning_rate = 0.002

    # 加载数据
    train_data = load_data("/content/Fairness-framework/data/a0910/all_virtual_user_data.csv")
    valid_data = load_data("/content/Fairness-framework/data/a0910/virtual_user_valid_data.csv")
    test_data = load_data("/content/Fairness-framework/data/a0910/virtual_user_test_data.csv")

    # 获取用户数量和题目数量
    user_num = max(train_data['user_id'].max(), valid_data['user_id'].max(), test_data['user_id'].max()) + 1
    item_num = max(train_data['item_id'].max(), valid_data['item_id'].max(), test_data['item_id'].max()) + 1

    print(f"用户数量: {user_num}, 题目数量: {item_num}")

    # 转换数据 - 训练和评估用标准格式
    train_loader = transform_data(train_data, batch_size)
    valid_loader = transform_data(valid_data, batch_size)
    test_loader = transform_data(test_data, batch_size)
    
    # 转换数据 - 提取能力参数用包含origin_id的格式
    test_loader_with_origin = transform_data(test_data, batch_size, include_origin_id=True)

    # 初始化模型
    model = IRT(user_num=user_num, item_num=item_num)
    
    # 训练模型
    print("开始训练模型...")
    model.train(
        train_data=train_loader,
        test_data=valid_loader,
        epoch=epochs,
        device=device,
        lr=learning_rate
    )

    # 保存模型
    save_path = "irt_model.pth"
    model.save(save_path)
    print(f"模型已保存至 {save_path}")

    # 评估模型
    print("在测试集上评估模型...")
    auc, accuracy = model.eval(test_loader, device=device)
    print(f"测试集结果 - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    # 提取能力参数
    print("提取用户能力参数...")
    model.extract_ability_parameters(
        test_data=test_loader_with_origin,
        device=device,
        filepath="user_abilities.csv"
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
