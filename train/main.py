# -*- coding:utf-8 -*-

import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from models import BartSeq2SeqModel, SequenceGeneratorModel
from utils import ABSADataset, Seq2SeqLoss, get_optimizer, pad_batch

# 랜덤시드 고정
SEED = 1398
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

# device 설정
device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Using {device} for inference')

bart_tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-base', add_prefix_space=True)
# 데이터셋 설정
BASE_DIR = './'
data_path = 'data/train_convert.json'
print(BASE_DIR + data_path)
with open(BASE_DIR + data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("Data Size:", len(data))
dataset = ABSADataset(BASE_DIR + data_path, bart_tokenizer, limit=10)

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=pad_batch)

tokenizer = dataset.get_tokenizer()
print("The number of tokens in tokenizer ", len(tokenizer.decoder))

mapping2id = dataset.get_mapping2id()
label_ids = list(mapping2id.values())
print("mapping2id:", mapping2id)
print("label_ids:", label_ids)

bart_name = 'facebook/bart-base'
length_penalty = 1.0
max_len = 10
max_len_a = 1.2
bos_token_id = 0
eos_token_id = 1

model = BartSeq2SeqModel.build_model(
    bart_name, tokenizer, label_ids=label_ids, use_recur_pos=False)
vocab_size = len(tokenizer)
print("vocab_size", vocab_size,
      model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(
    model,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    max_length=max_len,
    max_len_a=max_len_a,
    do_sample=False,
    repetition_penalty=1,
    length_penalty=length_penalty, pad_token_id=eos_token_id)
model.to(device)


criterion = Seq2SeqLoss()
optimizer = get_optimizer(
    model, lr_list=[1e-4, 1e-4, 1e-4], wd_list=[1e-2, 1e-2, 0])

n_epochs = 10
for epoch in range(n_epochs):
    model.train()  # 학습모드로 설정
    total_loss = 0
    with tqdm(dataloader, unit="batch") as pbar:
        for batch in pbar:
            src_tokens, tgt_tokens = [], []
            src_seq_len, tgt_seq_len = [], []

            for obj in batch:
                src_tokens.append(obj['src_tokens'])
                tgt_tokens.append(obj['tgt_tokens'])
                src_seq_len.append(obj['src_seq_len'])
                tgt_seq_len.append(obj['tgt_seq_len'])

            src_tokens = torch.tensor(src_tokens)
            tgt_tokens = torch.tensor(tgt_tokens)
            src_seq_len = torch.tensor(src_seq_len)
            tgt_seq_len = torch.tensor(tgt_seq_len)

            outputs = model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)
            loss = criterion(tgt_tokens, outputs['pred'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

model.eval()
