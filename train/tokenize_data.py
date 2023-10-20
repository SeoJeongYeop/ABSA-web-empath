import json
from tqdm import tqdm
from transformers import MBartTokenizerFast


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

    output = {
        "raw_words": raw_words,
        "words": words,
        "aspects": extracted_aspects,
        "opinions": extracted_opinions
    }

    return output


if __name__ == '__main__':

    bart_name = 'facebook/mbart-large-cc25'
    tokenizer = MBartTokenizerFast.from_pretrained(
        bart_name, add_prefix_space=True)
    fail_count = 0

    with open('data/exsa_ko.json', 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)

    result = []
    with tqdm(input_data, unit="sentence", desc="Tokenize") as pbar:
        for raw in pbar:
            format_data = aspect_opinion_extraction(tokenizer, raw)
            if format_data is None:
                print("iter terminated")
                break
            if len(format_data['aspects']) == len(format_data['opinions']):
                result.append(format_data)
            else:
                fail_count += 1
    print("result len:", len(result))
    print("fail_count:", fail_count)

    with open('data/exsa_ko_output.json', 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
