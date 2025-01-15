import pandas as pd

# 读取CSV文件，跳过第一行
df = pd.read_csv('/data/feng1702/fairness/data/test/test_finial.csv', header=None, skiprows=1)

# 去除残缺值
df.dropna(inplace=True)

# 将第四列的值转换为整数
df[3] = df[3].astype(float).astype(int)

# 保存处理后的CSV文件
df.to_csv('/data/feng1702/fairness/data/test/test_finial_cleaned.csv', header=False, index=False)