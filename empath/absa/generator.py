import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from empath.settings import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "absa/checkpoint")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
TAG_TO_SPECIAL = {"POS": ("<pos>", "</pos>"),
                  "NEG": ("<neg>", "</neg>"), "NEU": ("<neu>", "</neu>")}
OPINION_TOKEN = "<opinion>"


def prepare_constrained_tokens(tokenizer):
    special_tokens = [tokenizer.eos_token]  # add end token

    special_tokens += [i[0] for i in TAG_TO_SPECIAL.values()]
    special_tokens += [OPINION_TOKEN]

    return special_tokens


class Prefix_fn_cls():
    def __init__(self, tokenizer, special_tokens, input_enc_idxs):
        self.tokenizer = tokenizer
        self.input_enc_idxs = input_enc_idxs
        self.special_ids = [element for l in self.tokenizer(
            special_tokens, add_special_tokens=False)['input_ids'] for element in l]
        self.special_ids = list(set(self.special_ids))

    def get(self, batch_id, previous_tokens):
        # get input
        inputs = list(
            set(self.input_enc_idxs[batch_id].tolist()))+self.special_ids
        return inputs


def pred_absa(input_text):
    print(f"Tokenizing...{input_text}")

    encoded_dict = tokenizer.encode_plus(
        text=input_text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    print(f"Inferencing...")

    constrained_vocab = prepare_constrained_tokens(tokenizer)
    prefix_fn_obj = Prefix_fn_cls(
        tokenizer, constrained_vocab, input_ids.to('cpu'))

    def prefix_fn(batch_id, sent): return prefix_fn_obj.get(batch_id, sent)

    model.eval()
    with torch.no_grad():
        outs_dict = model.generate(
            input_ids=input_ids.to('cpu'),
            attention_mask=attention_mask.to('cpu'),
            max_length=128,
            prefix_allowed_tokens_fn=prefix_fn,
            output_scores=True,
            return_dict_in_generate=True,
            no_repeat_ngram_size=1,
            do_sample=True,
            top_p=0.95,
            max_time=1.5
        )
        outs = outs_dict["sequences"]
        pred = [tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs]
        print("pred", pred)

        return pred
