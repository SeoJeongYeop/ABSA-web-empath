# -*- coding:utf-8 -*-
from metric import WriteResultToFileMetric, Seq2SeqSpanMetric
import os
import json

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

# for torch.load
from utils import ABSADataset, Seq2SeqLoss, collate_fn, greedy_generate
from models import BartSeq2SeqModel, SequenceGeneratorModel, FBartEncoder, FBartDecoder, SequenceGenerator

# device 설정
device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Using {device} for inference')

# 하이퍼 파리미터 설정
batch_size = 4
# nested optimizer
lr_list = [1e-6, 1e-6, 1e-6]
wd_list = [1e-2, 1e-2, 0]
# normal optimizer
lr = 1e-4
wd = 0
# Model parameter
bart_name = 'facebook/bart-base'
length_penalty = 1.0
max_len = 10
max_len_a = 1.2
bos_token_id = 0
eos_token_id = 1
# training
n_epochs = 5
validation_mode = False

# 데이터셋 설정
BASE_DIR = './'
test_data_path = 'data/test_convert.json'

print(BASE_DIR + test_data_path)
with open(BASE_DIR + test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("Data Size:", len(data))

# Test Dataset
tokenizer = BartTokenizer.from_pretrained(bart_name, add_prefix_space=True)
test_dataset = ABSADataset(BASE_DIR + test_data_path, tokenizer, limit=50)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

tokenizer = test_dataset.get_tokenizer()
print("The number of tokens in tokenizer ", len(tokenizer.decoder))

mapping2id = test_dataset.get_mapping2id()
label_ids = list(mapping2id.values())
print("mapping2id:", mapping2id)
print("label_ids:", label_ids)


def evaluate(model, test_loader, seq2seq_metric, metric, device):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src_tokens = batch['src_tokens'].long().to(device)
            tgt_tokens = batch['tgt_tokens'].long().to(device)
            src_seq_len = batch['src_seq_len'].to(device)
            tgt_seq_len = batch['tgt_seq_len'].to(device)
            target_span = batch['target_span']

            # raw_words = batch['raw_words']
            raw_words = batch['words']
            aspects = batch['aspects']
            opinions = batch['opinions']

            outputs = model.predict(src_tokens, src_seq_len)
            pred = outputs['pred'].to(device)
            seq2seq_metric.evaluate(target_span, pred, tgt_tokens)

            metric.evaluate(target_span, raw_words, aspects, opinions, pred)
    print("metric.evaluate DONE")
    res = {}
    res['seq2seq_metric'] = seq2seq_metric.get_metric()
    res['result_metric'] = metric.get_metric()

    return res


model_path = './models/model.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
print("model", model)

lr_list = [1e-6, 1e-6, 1e-6]
wd_list = [1e-2, 1e-2, 0]

dataset_name = 'seo'
fp = os.path.split(dataset_name)[-1] + '.txt'

metric = WriteResultToFileMetric(len(mapping2id) + 2, list(mapping2id.keys(
)), fp, tokenizer, eos_token_id, num_labels=len(label_ids), opinion_first=False)

seq2seq_metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(
    label_ids), opinion_first=False)

results = evaluate(model.to(device), test_dataloader,
                   seq2seq_metric, metric, device)

for key, value in results.items():
    print(f"=== {key} ===")
    print(value)
