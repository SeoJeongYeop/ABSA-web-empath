import torch
import json
from functools import cmp_to_key
from itertools import chain
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


def cmp_aspect(v1, v2):
    # cmp_to_key 를 사용할 함수
    if v1[0]['from'] == v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']


def cmp_opinion(v1, v2):
    # cmp_to_key 를 사용할 함수
    if v1[1]['from'] == v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class ABSADataset(Dataset):
    def __init__(self, path, tokenizer, opinion_first=True, limit=None):
        super(ABSADataset, self).__init__()
        self.limit = limit
        self.tokenizer = tokenizer
        self.data = self._load_data(path)
        self.opinion_first = opinion_first

        self.mapping = {
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }
        self.target_shift = len(self.mapping) + 2

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(
            list(self.mapping.values()),
            key=lambda x: len(x),
            reverse=True
        )

        unique_no_split_tokens = []
        sorted_add_tokens = sorted(
            list(tokens_to_add),
            key=lambda x: len(x),
            reverse=True
        )

        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids(
                [tok])[0] == self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def _load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data[:self.limit] if self.limit else data

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        prepared_target = self.prepare_target(ins)
        target = prepared_target['tgt_tokens']
        target_spans = prepared_target['target_span']
        src_tokens = prepared_target['src_tokens']

        return {
            'src_tokens': src_tokens,
            'tgt_tokens': target,
            'target_span': target_spans,
            'src_seq_len': len(src_tokens),
            'tgt_seq_len': len(target)
        }

    def prepare_target(self, ins):
        # Byte pair
        raw_words = ins['raw_words']
        word_bpes = [[self.tokenizer.bos_token_id]]
        for word in raw_words:
            bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)

            word_bpes.append(bpes)
        word_bpes.append([self.tokenizer.eos_token_id])

        lens = list(map(len, word_bpes))
        cum_lens = np.cumsum(list(lens)).tolist()
        target = [0]  # sos를 위해 0 추가
        target_spans = []

        aspects_opinions = [(a, o)
                            for a, o in zip(ins['aspects'], ins['opinions'])]
        if self.opinion_first:
            aspects_opinions = sorted(
                aspects_opinions, key=cmp_to_key(cmp_opinion))
        else:
            aspects_opinions = sorted(
                aspects_opinions, key=cmp_to_key(cmp_aspect))

        for aspects, opinions in aspects_opinions:
            # bpe의 start를 예측
            assert aspects['index'] == opinions['index']

            a_start_bpe = cum_lens[aspects['from']]
            a_end_bpe = cum_lens[aspects['to']-1]

            o_start_bpe = cum_lens[opinions['from']]
            o_end_bpe = cum_lens[opinions['to']-1]

            if self.opinion_first:
                target_spans.append([o_start_bpe+self.target_shift, o_end_bpe+self.target_shift,
                                     a_start_bpe+self.target_shift, a_end_bpe+self.target_shift])
            else:
                target_spans.append([a_start_bpe+self.target_shift, a_end_bpe+self.target_shift,
                                     o_start_bpe+self.target_shift, o_end_bpe+self.target_shift])

            # 앞에 sos랑 eos 포함
            target_spans[-1].append(self.mapping2targetid[aspects['polarity']]+2)
            target_spans[-1] = tuple(target_spans[-1])
        target.extend(list(chain(*target_spans)))
        target.append(1)  # eos를 위해 1을 추가

        return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes))}

    def get_tokenizer(self):
        return self.tokenizer

    def get_mapping2id(self):
        return self.mapping2id


@torch.no_grad()
def greedy_generate(decoder, tokens=None, state=None, max_length=20, max_len_a=0.0,
                    bos_token_id=None, eos_token_id=None, pad_token_id=0,
                    repetition_penalty=1, length_penalty=1.0):
    """
    Greedy search for generating sequences.
    """
    token_ids = _no_beam_search_generate(
        decoder,
        tokens=tokens,
        state=state,
        max_length=max_length,
        max_len_a=max_len_a,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        pad_token_id=pad_token_id
    )

    return token_ids


def _no_beam_search_generate(decoder, state: dict, tokens=None, max_length=20,
                             max_len_a=0.0, bos_token_id=None, eos_token_id=None,
                             repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0):
    """
    Core function for greedy search, generates the next token and adds to the result sequence.
    """

    device = tokens.device if tokens is not None else state['encoder_mask'].device

    # Initialize with start tokens if not provided
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError(
                "Either `tokens` or `bos_token_id` must be specified.")
        if 'encoder_mask' in state:
            batch_size = state['encoder_mask'].size(0)
        else:
            batch_size = tokens.size(0)
        tokens = torch.full(
            [batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)

    # Begin decoding
    # Assuming the decoder's forward method returns scores
    scores = decoder(
        tokens,
        state['encoder_outputs'],
        state['encoder_mask'],
        state['src_tokens']
    )
    next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(token_ids.size(0)).bool()
    if eos_token_id:
        dones |= next_tokens.squeeze(1).eq(eos_token_id)

    # Compute max length if scaled by source length
    if max_len_a != 0 and state in 'encoder_mask':
        max_lengths = (state['encoder_mask'].sum(dim=1).float()
                       * max_len_a).long() + max_length
    else:
        max_lengths = torch.full(
            (tokens.size(0),), fill_value=max_length, dtype=torch.long, device=device)

    # Continue decoding until max length is reached or all sequences have terminated
    while cur_len < max_lengths.max().item():
        # Assuming the decoder's forward method returns scores
        scores = decoder(
            token_ids,
            state['encoder_outputs'],
            state['encoder_mask'],
            state['src_tokens']
        )

        # Handle repetition and length penalties
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            token_scores *= repetition_penalty
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            scores.masked_fill_(eos_mask.unsqueeze(0).bool(), token_scores)

        next_tokens = scores.argmax(dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, next_tokens], dim=-1)

        if eos_token_id:
            dones |= next_tokens.squeeze(1).eq(eos_token_id)
        cur_len += 1

        if dones.all():
            break

    return token_ids


class Seq2SeqLoss(torch.nn.Module):
    # 손실함수 설정

    def __init__(self):
        super(Seq2SeqLoss, self).__init__()

    def forward(self, tgt_tokens, pred):
        # assuming 0 is pad_token_id, adjust if needed
        mask = (tgt_tokens != 0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(~mask[:, 1:], -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        return loss


def get_optimizer(model, lr_list=[1e-4, 1e-4, 1e-4],  wd_list=[1e-2, 1e-2, 0]):
    # 옵티마이저 설정
    parameters = []
    params = {'lr': lr_list[0], 'weight_decay': wd_list[0]}
    params['params'] = [param for name, param in model.named_parameters() if not (
        'bart_encoder' in name or 'bart_decoder' in name)]
    parameters.append(params)

    params = {'lr': lr_list[1], 'weight_decay': wd_list[1]}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    params = {'lr': lr_list[2], 'weight_decay': wd_list[2]}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    optimizer = torch.optim.AdamW(parameters)
    return optimizer


def pad_batch(batch):
    max_src_seq_len = max([sample['src_seq_len'] for sample in batch])
    max_tgt_seq_len = max([sample['tgt_seq_len'] for sample in batch])

    for sample in batch:
        sample['src_tokens'] += [1] * (max_src_seq_len - sample['src_seq_len'])
        sample['tgt_tokens'] += [1] * (max_tgt_seq_len - sample['tgt_seq_len'])
    return batch
