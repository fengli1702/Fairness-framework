import csv
import random

# 读取CSV文件并按组存储数据
grouped_data = {}

with open('./a0910/origin_with_group_updated.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        group_id = row['group_id']
        if group_id not in grouped_data:
            grouped_data[group_id] = []
        grouped_data[group_id].append(row)

# 对每个组内的数据进行打乱
for group_id in grouped_data:
    random.shuffle(grouped_data[group_id])

# 将打乱后的数据写入新的CSV文件
with open('./a0910/origin_with_group_shuffled.csv', 'w', newline='') as outfile:
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for group_id in grouped_data:
        for row in grouped_data[group_id]:
            writer.writerow(row)