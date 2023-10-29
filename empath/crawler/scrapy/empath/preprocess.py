import re


def remove_escape(text):
    return re.sub(r'[\\][n|t|r]', ' ', text)


def reduce_escape(text):
    text = re.sub(r'(\\n|\\r){2,}', '\\n', text)
    text = re.sub(r'(\n|\r){2,}', '\n', text)
    return text


def remove_emoji(text):
    emojis = []
    # emoji 특수기호 + ^^ :)와 같은 이모티콘
    emoji = re.findall(
        r'[\u2700-\u27BF]|[\u2011-\u26FF]|[\u2900-\u2b59]|[!-/:-@\[-`]{2,}', text, flags=re.UNICODE)
    if len(emoji) > 0:
        emojis.append(emoji)
    # (powerful) 한글, 영어, 숫자, 페리오드, 콤마 남기고 제거, 자모 제거하지 않음
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9.,% ]', ' ', text)
    text.replace(',', '')
    text = re.sub(' +', ' ', text).strip()

    return text, emojis


def remove_jamo(text):
    jamos = []

    # 중복 띄어쓰기 제거
    text = re.sub(' +', ' ', text).strip()
    jamo = re.findall(r'[ㄱ-ㅎㅏ-ㅣ ]{2,}', text)
    if len(jamo) > 0:
        jamos.append(jamo)

    # 한글, 영어, 숫자, 페리오드, 콤마 남기고 제거, 자모도 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9.,`% ]', ' ', text)
    # 중복 띄어쓰기 제거
    text = re.sub(' +', ' ', text).strip()

    return text, jamos


def cleaning_text(text):

    # (제거대상) &amp; &gt; 등 특수문자 형식 제거
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    # (정규화 대상) 이메일 형식 제거
    text = re.sub(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9]+', 'EMAIL', text)
    # (정규화 대상)url 제거
    text = re.sub(r'(\w+:\/\/\S+)', 'URL', text)
    # (제거 대상)멘션과 해시태그 제거
    text = re.sub(r'(@[A-Za-z가-힣0-9_]+)|(#[A-Za-z가-힣0-9_]+)', '', text)
    # (정규화 대상 추가)출처 이후 문자열 제거
    text = re.sub(r'(출처).*', '출처', text)
    # 따옴표 처리를 백틱(`)으로 통일
    text = re.sub(r"""[“”‘’"']""", '`', text)
    # 말줄임표로 사용하는 페리오드 4개 이상은 3개로 축소
    text = text.replace("…", "...")
    text = re.sub(r'[.,]{4,}', '... ', text)
    text = re.sub(r'[.]{2}', '.', text)

    text = re.sub(r'[ㅜㅠ]{2,}', 'ㅠㅠ', text)
    text = re.sub(r'[ㄱㅋㅌ⫬]{2,}', 'ㅋㅋ', text)
    # 위에 해당하지 않고 자모 3개 이상 제거
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]{3,}', '', text)
    # 연속 줄바꿈 제거
    text = reduce_escape(text)
    # 중복 띄어쓰기 제거
    text = re.sub(' +', ' ', text).strip()

    return text
