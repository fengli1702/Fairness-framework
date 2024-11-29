import csv
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件并计算每个组内人数的组数
group_sizes = set()

with open('./a0910/origin_with_group.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        group_id = row['group_id']
        group = row['get_group'].strip('[]').split()
        group_size = len(group)
        group_sizes.add((group_id, group_size))

# 统计每个组内人数的组数
group_count = Counter(group_size for _, group_size in group_sizes)

# 准备数据
x = sorted(group_count.keys())
y = [group_count[size] for size in x]
# 绘制立方图
plt.bar(x, y)
plt.xlabel('people')
plt.ylabel('group num')
plt.title('Group Size Distribution')

# 保存图表到文件
plt.savefig('group_size_distribution.png')