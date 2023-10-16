# -*- coding:utf-8 -*-

import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from models import BartSeq2SeqModel, SequenceGeneratorModel
from utils import ABSADataset, Seq2SeqLoss, get_nested_optimizer, pad_batch, device

# 랜덤시드 고정
SEED = 1398
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

bart_tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-base', add_prefix_space=True)

# 데이터셋 확인
BASE_DIR = './'
data_path = '/data/train_convert.json'
valid_data_path = '/data/dev_convert.json'
test_data_path = '/data/test_convert.json'

print(BASE_DIR + data_path)
with open(BASE_DIR + data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("Train Data Size:", len(data))
print(BASE_DIR + valid_data_path)
with open(BASE_DIR + valid_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("Valid Data Size:", len(data))
print(BASE_DIR + test_data_path)
with open(BASE_DIR + test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("Test Data Size:", len(data))

# 데이터셋 설정
dataset = ABSADataset(BASE_DIR + data_path, bart_tokenizer)
tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-base', add_prefix_space=True)
valid_dataset = ABSADataset(BASE_DIR + valid_data_path, tokenizer)
tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-base', add_prefix_space=True)
test_dataset = ABSADataset(BASE_DIR + test_data_path, tokenizer)

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=pad_batch)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

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
lr = 1e-4
wd = 0
lr_list = [1e-4, 1e-4, 1e-4]
wd_list = [1e-2, 1e-2, 0]
n_epochs = 10

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
# optimizer = get_nested_optimizer(
#     model, lr_list=lr_list, wd_list=wd_list)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(n_epochs):
    # Train
    model.train()  # 학습모드로 설정
    total_loss = 0
    with tqdm(dataloader, unit="batch", desc="Training") as pbar:
        for batch in pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            src_tokens = batch['src_tokens'].long().to(
                device)  # 바로 텐서로 변환된 src_tokens 사용
            tgt_tokens = batch['tgt_tokens'].long().to(
                device)  # 바로 텐서로 변환된 tgt_tokens 사용
            src_seq_len = batch['src_seq_len'].to(device)
            tgt_seq_len = batch['tgt_seq_len'].to(device)

            outputs = model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)
            loss = criterion.get_loss(tgt_tokens, outputs['pred'], tgt_seq_len)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

            total_loss += loss.item()
    epoch_loss = round(total_loss / len(dataloader), 5)
    # Validation Phase
    model.eval()  # Set model to evaluation mode
    total_valid_loss = 0

    with torch.no_grad():  # No gradient needed for validation
        with tqdm(valid_dataloader, unit="batch", desc="Validation") as pbar:
            for batch in pbar:
                src_tokens = batch['src_tokens'].long().to(device)
                tgt_tokens = batch['tgt_tokens'].long().to(device)
                src_seq_len = batch['src_seq_len'].to(device)
                tgt_seq_len = batch['tgt_seq_len'].to(device)

                outputs = model(src_tokens, tgt_tokens,
                                src_seq_len, tgt_seq_len)
                loss = criterion.get_loss(
                    tgt_tokens, outputs['pred'], tgt_seq_len)

                pbar.set_postfix(valid_loss=loss.item())
                total_valid_loss += loss.item()

    avg_valid_loss = round(total_valid_loss / len(valid_dataloader), 5)
    print(
        f"Epoch {epoch+1} - Train Loss: {epoch_loss} / Validation Loss: {avg_valid_loss}")
