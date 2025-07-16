import os, json, random

# …existing code…

# 将读写都指向 data/jsonl 目录
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data', 'jsonl')
os.makedirs(data_dir, exist_ok=True)

input_path  = os.path.join(data_dir, 'poems.jsonl')
train_path  = os.path.join(data_dir, 'poems_train.jsonl')
valid_path  = os.path.join(data_dir, 'poems_valid.jsonl')

# …existing code…
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
# 打乱
random.seed(42)
random.shuffle(lines)
# 切分
n = len(lines)
n_valid = int(n * 0.1)  # 10% for validation
valid_lines = lines[:n_valid]
train_lines = lines[n_valid:]
# 写文件
with open(train_path, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)
with open(valid_path, 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)
print(f'总样本: {n}, 训练: {len(train_lines)}, 验证: {len(valid_lines)}')