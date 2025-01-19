import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from IRT_LOSS import IRT

class TestDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        print(f"Dataset columns: {df.columns.tolist()}")
        self.user_ids = df['user_id'].values
        self.item_ids = df['item_id'].values
        self.scores = df['score'].values
        self.origin_ids = df['origin_id'].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        return (
            torch.tensor(self.origin_ids[index]),
            torch.tensor(self.user_ids[index]),
            torch.tensor(self.item_ids[index]),
            torch.tensor(0),
            torch.tensor(float(self.scores[index]))
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_path = '/content/Fairness-framework/data/a0910'
    train_path = os.path.join(base_path, 'virtual_user_train_data.csv')
    valid_path = os.path.join(base_path, 'virtual_user_valid_data.csv')
    test_path = os.path.join(base_path, 'virtual_user_test_data.csv')
    
    print("\nTrain data preview:")
    train_df = pd.read_csv(train_path)
    print(train_df.head())
    print("\nTrain data columns:", train_df.columns.tolist())
    
    all_data = pd.concat([
        pd.read_csv(train_path),
        pd.read_csv(valid_path),
        pd.read_csv(test_path)
    ])
    
    student_n = all_data['user_id'].max() + 1
    exer_n = all_data['item_id'].max() + 1
    
    print(f"Number of students: {student_n}")
    print(f"Number of exercises: {exer_n}")
    
    train_dataset = TestDataset(train_path)
    valid_dataset = TestDataset(valid_path)
    test_dataset = TestDataset(test_path)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = IRT(
        student_n=student_n,
        exer_n=exer_n,
        zeta=0.5,
        groupsize=11
    )
    
    print("Starting training...")
    model.train(
        train_data=train_loader,
        test_data=valid_loader,  # 使用正确的参数名 test_data
        epoch=1,
        device=device,
        lr=0.002
    )
    
    print("\nEvaluating on test set...")
    test_auc, test_accuracy = model.eval(test_loader, device=device)
    print(f"Final Test Results - AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
    
    model.save("irt_model.pt")
    print("Model saved to 'irt_model.pt'")

if __name__ == "__main__":
    main()
