from functools import partial
from transformers.models.bart.modeling_bart import BartModel, BartEncoder, BartDecoder
from torch import nn
import torch.nn.functional as F
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import BartEncoder
import torch

from utils import greedy_generate


class FBartEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = self._seq_len_to_mask(src_seq_len)
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states

    @staticmethod
    def _seq_len_to_mask(seq_len, max_len=None):
        if max_len is None:
            max_len = seq_len.max().item()
        batch_size = seq_len.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_len_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_len_expand


class FBartDecoder(nn.Module):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1] + 1
        mapping = torch.LongTensor([0, 2] + sorted(label_ids, reverse=False))
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, tokens, encoder_outputs, encoder_mask, src_tokens, first=None, past_key_values=None):
        encoder_pad_mask = encoder_mask

        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # mapping to the BART token index
        mapping_token_mask = tokens.lt(self.src_start_index)  #
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(
            src_tokens_index.lt(0), 0)

        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask,
                             tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_attention_mask=encoder_pad_mask,
                                attention_mask=decoder_pad_mask)
        else:
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_attention_mask=encoder_pad_mask,
                                attention_mask=None)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # first get the
        # bsz x max_len x 1
        print(hidden_state)
        print(hidden_state.shape)

        print(self.decoder.embed_tokens.weight[2:3].shape)

        eos_scores = F.linear(
            hidden_state, self.decoder.embed_tokens.weight[2:3])
        # bsz x max_len x num_class
        tag_scores = F.linear(
            hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])

        # bsz x max_word_len x hidden_size
        src_outputs = encoder_outputs

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        mask = encoder_mask.eq(0)
        mask = mask.unsqueeze(1).__or__(
            src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        # bsz x max_len x max_word_len
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, encoder_output, encoder_mask, src_tokens, first=None, past_key_values=None):
        return self(tokens, encoder_output, encoder_mask, src_tokens, first, past_key_values)[:, -1]


class BartSeq2SeqModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(BartSeq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, use_recur_pos=False, tag_first=False):
        model = BartModel.from_pretrained(bart_model)
        encoder = model.get_encoder()
        decoder = model.get_decoder()
        num_tokens, _ = encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(
            len(tokenizer.unique_no_split_tokens) + num_tokens)

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(
                    _tokenizer.tokenize(token[2:-2]))
                embed = encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        decoder = FBartDecoder(
            decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(
            src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = {
            'encoder_outputs': encoder_outputs,
            'encoder_mask': encoder_mask,
            'src_tokens': src_tokens,
            'first': first,
            'src_embed_outputs': src_embed_outputs
        }
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(
            src_tokens, src_seq_len)
        decoder_output = self.decoder(
            tgt_tokens, encoder_outputs, encoder_mask, src_tokens)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(
                f"Unsupported return type from Decoder: {type(self.decoder)}")


class SequenceGeneratorModel(nn.Module):
    def __init__(self, seq2seq_model, bos_token_id, eos_token_id=None, max_length=30, max_len_a=0.0,
                 do_sample=True, repetition_penalty=1, length_penalty=1.0, pad_token_id=0):
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           do_sample=do_sample, bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id, repetition_penalty=repetition_penalty,
                                           length_penalty=length_penalty, pad_token_id=pad_token_id)

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None, first=None):
        return self.seq2seq_model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first)

    def predict(self, src_tokens, src_seq_len=None, first=None):
        state = self.seq2seq_model.prepare_state(
            src_tokens, src_seq_len, first)
        result = self.generator.generate(state)
        return {'pred': result}


class SequenceGenerator:
    def __init__(self, decoder, max_length=20, max_len_a=0.0, do_sample=False, bos_token_id=None,
                 eos_token_id=None, repetition_penalty=1, length_penalty=1.0, pad_token_id=0):
        self.generate_func = partial(greedy_generate, decoder=decoder, max_length=max_length, max_len_a=max_len_a,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                     pad_token_id=pad_token_id)
        self.do_sample = do_sample
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.max_len_a = max_len_a

    def set_new_generator(self, max_length=-1, max_len_a=-1, repetition_penalty=-1, length_penalty=-1):
        if max_length == -1:
            max_length = self.max_length
        if max_len_a == -1:
            max_len_a = self.max_len_a
        if repetition_penalty == -1:
            repetition_penalty = self.repetition_penalty
        if length_penalty == -1:
            length_penalty = self.length_penalty

        self.generate_func = partial(greedy_generate, decoder=self.decoder, max_length=max_length, max_len_a=max_len_a,
                                     bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id,
                                     repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                     pad_token_id=self.pad_token_id)

    @torch.no_grad()
    def generate(self, state, tokens=None):
        return self.generate_func(tokens=tokens, state=state)
