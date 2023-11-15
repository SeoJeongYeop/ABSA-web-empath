import json
from tqdm import tqdm
from transformers import T5Tokenizer


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

            a_from, a_to = -1, -1
            for i in range(len(words)):
                if aspect_term[0] == words[i] or '▁'+aspect_term[0] == words[i]:

                    a_from = i
                    a_to = i + len(aspect_term)-1

                    if a_to < len(words) and aspect_term[-1] == words[a_to]:

                        token_mask[aspect_index] = True
                        break
            if a_to == -1:
                print("aspect_term", aspect_term, "words", words)

            extracted_aspect = {
                "index": aspect_index,
                "from": a_from,
                "to": a_to,
                "polarity": polarity,
                "term": aspect_term
            }
            extracted_aspects.append(extracted_aspect)
    except Exception as e:
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

            o_from, o_to = -1, -1
            for i in range(len(words)):
                if opinion_term[0] == words[i] or '▁'+opinion_term[0] == words[i]:
                    o_from = i
                    o_to = i + len(opinion_term) - 1
                    if opinion_term[-1] == words[o_to]:
                        token_mask[opinion_index] = True
                        break
            if o_to == -1:
                print("opinion_term", opinion_term, "words", words)

            extracted_opinion = {
                "index": opinion_index,
                "from": o_from,
                "to": o_to,
                "term": opinion_term
            }
            extracted_opinions.append(extracted_opinion)
    except Exception as e:
        print(e)

        return {
            "raw_words": input_data["raw_words"],
            "words": words,
            "aspects": extracted_aspects,
            "opinions": []
        }

    output = {
        "raw_words": input_data["raw_words"],
        "words": words,
        "aspects": extracted_aspects,
        "opinions": extracted_opinions,
        "token_len": len(words)
    }

    return output


def check_valid(words, aspects, opinions):
    is_valid = True
    a_len = len(aspects)
    o_len = len(opinions)
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
    return is_valid


def format_triplet(aspects, opinions):
    ao_tri = []
    for a, o in zip(aspects, opinions):
        p_txt = f"'{a['polarity']}'"
        if a['from'] == a['to']:
            a_txt = f"[{a['from']}]"
        else:
            a_txt = f"[{a['from']}, {a['to']}]"
        if o['from'] == o['to']:
            o_txt = f"[{o['from']}]"
        else:
            o_txt = f"[{o['from']}, {o['to']}]"
        ao_tri.append(f"({a_txt}, {o_txt}, {p_txt})")
    return ao_tri


if __name__ == '__main__':

    model_name = "KETI-AIR/ke-t5-base-ko"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    fail_count = 0
    no_aspect = 0
    no_opinion = 0

    data_dir = '../data/'
    # input_file_name = 'blog.json'
    input_file_name = 'news.json'
    # input_file_name = 'youtube.json'
    max_token = 0

    with open(data_dir+input_file_name, 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)

    result, dones, failures = [], [], []
    with tqdm(input_data, unit="sentence", desc="Tokenize") as pbar:
        for raw in pbar:
            format_data = aspect_opinion_extraction(tokenizer, raw)

            if format_data is None:
                print("iter terminated")
                break

            aspects = format_data['aspects']
            opinions = format_data['opinions']
            words = format_data['words']

            if len(aspects) == 0:
                no_aspect += 1
            elif len(opinions) == 0:
                no_opinion += 1

            is_valid = check_valid(words, aspects, opinions)

            if is_valid:
                max_token = max(max_token, format_data['token_len'])
                tokens = " ".join(words)
                # tokens = format_data["raw_words"]
                ao_tri = format_triplet(aspects, opinions)
                absa_txt = f"[{', '.join(ao_tri)}]"
                result.append(tokens + "####"+absa_txt)
                dones.append(format_data)
            else:
                fail_count += 1
                failures.append(format_data)

    print("result len:", len(result))
    print("fail_count:", fail_count)
    print("no aspect:", no_aspect)
    print("no opinion:", no_opinion)
    print("no pair:", fail_count-no_aspect-no_opinion)

    print("max_token:", max_token)

    output_file_name = input_file_name.replace('.json', '.txt')
    done_file_name = input_file_name.replace('.', '_done.')
    fail_file_name = input_file_name.replace('.', '_fail.')
    data_dir = '../data/'

    with open(data_dir+output_file_name, 'w', encoding='utf-8') as output_file:
        for item in result:
            output_file.write(item + '\n')
    with open(data_dir+done_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(dones, output_file, ensure_ascii=False, indent=2)
    with open(data_dir+fail_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(failures, output_file, ensure_ascii=False, indent=2)
