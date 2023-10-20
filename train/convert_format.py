import json

polarityMap = {
    'positive': 'POS',
    'neutral': 'NEU',
    'negative': 'NEG'
}

if __name__ == '__main__':
    with open('data/EXSA2122303090.json', 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)

    documents = input_data.get('document')

    result = []
    for doc in documents:
        sentences = doc['sentence']
        for st in sentences:
            opinions = st.get('opinions', [])
            if len(opinions) != 0:
                ret = {}
                ret_o = []
                ret_a = []
                for _o in opinions:
                    target = _o['opinion target']
                    polarity = polarityMap[_o['polarity']]

                    if target != 'Null':
                        ret_a.append({
                            'polarity': polarity,
                            'term': target
                        })
                        ret_o.append({'term': ''})
                if len(ret_a) > 0:
                    ret['raw_words'] = st['sentence_form']
                    ret['aspects'] = ret_a
                    ret['opinions'] = ret_o

                    result.append(ret)

    print("result len:", len(result))
    with open('data/exsa.json', 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
