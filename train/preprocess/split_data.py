import json
import random

# JSON 파일 로드
with open('../data/total_ko_o_first.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 데이터 셔플
random.shuffle(data)

# 데이터를 8:2 비율로 나눔
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
test_data = data[split_index:]
print("train_data", len(train_data))
print("test_data", len(test_data))


# train과 test 데이터를 각각 파일로 저장 (선택사항)
with open('train_data_o_first.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

with open('test_data_o_first.json', 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, ensure_ascii=False, indent=4)
