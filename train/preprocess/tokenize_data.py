import json
from tqdm import tqdm
from transformers import MBartTokenizerFast


def is_sublist(A, B):
    '''A가 B의 서브리스트인지 확인하는 함수'''
    for i in range(len(B) - len(A) + 1):
        if B[i:i+len(A)] == A:
            return True
    return False


def aspect_opinion_extraction(tokenizer, input_data):
    raw_words = input_data["raw_words"]
    aspects = input_data["aspects"]
    opinions = input_data["opinions"]
    if opinions[0]['term'] == '' or raw_words == '':
        # 빈 데이터면 탈출
        return None

    words = tokenizer.tokenize(raw_words)

    token_mask = [False for _ in range(len(aspects))]

    extracted_aspects = []
    extracted_opinions = []

    try:
        for aspect_index, aspect_info in enumerate(aspects):
            aspect_term = aspect_info["term"]
            aspect_term = tokenizer.tokenize(aspect_term)
            polarity = aspect_info["polarity"]

            aspect_from = -1
            aspect_to = -1
            for i in range(len(words)):
                if aspect_term[0] in words[i]:
                    aspect_from = i
                    aspect_to = i + len(aspect_term)

                    if aspect_to-1 < len(words) and aspect_term[-1] == words[aspect_to-1]:
                        token_mask[aspect_index] = True
                        break

            extracted_aspect = {
                "index": aspect_index,
                "from": aspect_from,
                "to": aspect_to,
                "polarity": polarity,
                "term": aspect_term
            }
            extracted_aspects.append(extracted_aspect)
    except:
        return {
            "raw_words": raw_words,
            "words": words,
            "aspects": [],
            "opinions": []
        }

    try:
        for opinion_index, opinion_info in enumerate(opinions):
            if not token_mask[opinion_index]:
                # 매칭되는 aspect가 없을 경우
                continue

            opinion_term = opinion_info["term"]
            opinion_term = tokenizer.tokenize(opinion_term)

            opinion_from = -1
            opinion_to = -1
            for i in range(len(words)):
                if opinion_term[0] in words[i]:
                    opinion_from = i
                    opinion_to = i + len(opinion_term)
                    if opinion_term[-1] == words[opinion_to-1]:
                        token_mask[aspect_index] = True
                        break

            extracted_opinion = {
                "index": opinion_index,
                "from": opinion_from,
                "to": opinion_to,
                "term": opinion_term
            }
            extracted_opinions.append(extracted_opinion)
    except:
        return {
            "raw_words": raw_words,
            "words": words,
            "aspects": extracted_aspects,
            "opinions": []
        }

    output = {
        "raw_words": raw_words,
        "words": words,
        "aspects": extracted_aspects,
        "opinions": extracted_opinions
    }

    return output


if __name__ == '__main__':

    bart_name = 'facebook/mbart-large-cc25'
    tokenizer = MBartTokenizerFast.from_pretrained(bart_name)
    fail_count = 0

    data_dir = '../data/'
    input_file_name = 'total.json'

    with open(data_dir+input_file_name, 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)

    result, failures = [], []
    with tqdm(input_data, unit="sentence", desc="Tokenize") as pbar:
        for raw in pbar:

            format_data = aspect_opinion_extraction(tokenizer, raw)

            if format_data is None:
                break

            aspects = format_data['aspects']
            opinions = format_data['opinions']
            words = format_data['words']
            a_len = len(format_data['aspects'])
            o_len = len(format_data['opinions'])

            is_valid = True
            if a_len == 0 or a_len != o_len:
                is_valid = False
            else:
                for asp in aspects:
                    if not is_sublist(asp['term'], words):
                        is_valid = False
                        break
                if is_valid:
                    for opn in opinions:
                        if not is_sublist(opn['term'], words):
                            is_valid = False
                            break
            if is_valid:
                result.append(format_data)
            else:
                fail_count += 1
                failures.append(format_data)

    print("result len:", len(result))
    print("fail_count:", fail_count)

    output_file_name = input_file_name.replace('.', '_ko.')
    fail_file_name = input_file_name.replace('.', '_fail.')

    with open(data_dir+output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
    with open(data_dir+fail_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(failures, output_file, ensure_ascii=False, indent=2)
