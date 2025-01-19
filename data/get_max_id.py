#用来获取数据中最大userid是多少
import pandas as pd

# 读取文件
file_path = "./a0910/train.csv"
df = pd.read_csv(file_path)
print(df["user_id"].max())
