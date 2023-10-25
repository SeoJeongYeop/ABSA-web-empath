# aspect가 먼저인 데이터와 opinion이 먼저인 데이터 분리

from tqdm import tqdm
import json

if __name__ == '__main__':

    data_dir = '../data/'
    input_file_name = 'total_ko.json'

    with open(data_dir+input_file_name, 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)

    aspect_first = []
    opinion_first = []
    mixed = []
    fail_count = 0
    with tqdm(input_data, unit="sentence", desc="Tokenize") as pbar:
        for data in pbar:
            # 데이터를 aspect_first, opinion_first, mixed로 분류

            is_aspect_first = False
            is_opinion_first = False

            # aspect와 opinion의 from 값 비교
            for aspect, opinion in zip(data['aspects'], data['opinions']):
                if aspect['from'] < opinion['from']:
                    is_aspect_first = True
                elif aspect['from'] > opinion['from']:
                    is_opinion_first = True
            if is_aspect_first and is_opinion_first:
                mixed.append(data)
            elif is_aspect_first:
                aspect_first.append(data)
            elif is_opinion_first:
                opinion_first.append(data)
            else:
                fail_count += 1
    print("aspect_first len:", len(aspect_first))
    print("opinion_first len:", len(opinion_first))
    print("mixed len:", len(mixed))
    print("fail_count:", fail_count)

    output_file_name = input_file_name.replace('.', '_a_first.')
    with open(data_dir+output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(aspect_first, output_file, ensure_ascii=False, indent=2)

    output_file_name = input_file_name.replace('.', '_o_first.')
    with open(data_dir+output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(opinion_first, output_file, ensure_ascii=False, indent=2)
