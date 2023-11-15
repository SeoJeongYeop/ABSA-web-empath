import os
import random

file_dir = '../data/'
# file_path = 'blog.txt'
file_path = 'news.txt'
# file_path = 'youtube.txt'


save_dir = f'../data/{file_path.split(".")[0]}/'
os.makedirs(save_dir, exist_ok=True)

data = []

with open(file_dir+file_path, 'r', encoding='utf-8') as file:
    data = file.readlines()

random.shuffle(data)

total_samples = len(data)

# 훈련/검증/테스트 비율
train_ratio = 0.7
dev_ratio = 0.1

train_size = int(total_samples * train_ratio)
dev_size = int(total_samples * dev_ratio)

train_data = data[:train_size]
dev_data = data[train_size:train_size + dev_size]
test_data = data[train_size + dev_size:]

# 저장
with open(save_dir + 'train.txt', 'w', encoding='utf-8') as train_file:
    train_file.writelines(train_data)

with open(save_dir + 'dev.txt', 'w', encoding='utf-8') as dev_file:
    dev_file.writelines(dev_data)

with open(save_dir + 'test.txt', 'w', encoding='utf-8') as test_file:
    test_file.writelines(test_data)
print(file_path)
print(f"Total samples: {total_samples}")
print(f"Train samples: {len(train_data)}")
print(f"Dev samples: {len(dev_data)}")
print(f"Test samples: {len(test_data)}")
