import csv

# 读取CSV文件并计算每个组的最小fairness_id和组内人数
group_info = {}

with open('./a0910/origin_with_group.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        group_id = row['group_id']
        fairness_id = int(row['fairness_id'])
        group = row['get_group'].strip('[]').split()
        group_size = len(group)
        
        if group_id not in group_info:
            group_info[group_id] = {'min_fairness_id': fairness_id, 'group_size': group_size}
        else:
            group_info[group_id]['min_fairness_id'] = min(group_info[group_id]['min_fairness_id'], fairness_id)

# 重新读取CSV文件并添加新列
with open('./a0910/origin_with_group.csv', 'r') as infile, open('./a0910/origin_with_group_updated.csv', 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['groupindex', 'group_size']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in reader:
        group_id = row['group_id']
        row['groupindex'] = group_info[group_id]['min_fairness_id']
        row['group_size'] = group_info[group_id]['group_size']
        writer.writerow(row)