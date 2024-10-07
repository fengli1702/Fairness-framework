# test_irt.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from IRT import IRT
from sklearn.model_selection import train_test_split


train_data = pd.read_csv("./data/a0910/virtual_user_train_data.csv")
valid_data = pd.read_csv("./data/a0910/virtual_user_valid_data.csv")
test_data = pd.read_csv("./data/a0910/virtual_user_test_data.csv")

batch_size = 256


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

# 初始化IRT模型

model = IRT(27424, 17747)

# 训练模型
model.train(train, valid, epoch=10)

# 保存模型
model.save("irt_model.pth")

# 加载模型
model.load("irt_model.pth")

# 评估模型
auc, accuracy = model.eval(test)
print(f"Test AUC: {auc}, Test Accuracy: {accuracy}")


#for name, param in model.irt_net.named_parameters():
 #   print(f"Name: {name}, Shape: {param.shape}, Values: {param.data}")


all_virtual_user_data = pd.read_csv('./data/a0910/all_virtual_user_data.csv')

# Transform function to include origin_id
def transform2(x, y, z, origin_ids, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(origin_ids, dtype=torch.int64),
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

# Prepare test_fairness dataset from all_virtual_user_data
test_fairness = transform2(
    all_virtual_user_data["user_id"], 
    all_virtual_user_data["item_id"], 
    all_virtual_user_data["score"], 
    all_virtual_user_data["origin_id"],  # Include original_id
    batch_size=256
)

# Assuming `model` is your IRT model instance
model.extract_ability_parameters(test_data=test_fairness, filepath="v_ability_parameters.csv")

# 查看模型参数
#theta_params = model.irt_net.theta.weight.data.cpu().numpy()
#a_params = model.irt_net.a.weight.data.cpu().numpy()
#b_params = model.irt_net.b.weight.data.cpu().numpy()
#c_params = model.irt_net.c.weight.data.cpu().numpy()
#
#print("Theta parameters (user abilities):", theta_params)
#print("a parameters (item discriminations):", a_params)
#print("b parameters (item difficulties):", b_params)
#print("c parameters (guessing parameters):", c_params)
