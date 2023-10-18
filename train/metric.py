import numpy as np
import json
from collections import Counter


class WriteResultToFileMetric:
    def __init__(self, target_shift, labels, fp, tokenizer, eos_token_id, num_labels, opinion_first=False):
        self.fp = fp
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels  # 3
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos # 5
        self.opinin_first = opinion_first
        self.tokenizer = tokenizer
        self.target_shift = target_shift
        self.labels = labels

        self.raw_words = []
        self.aspect_preds = []
        self.opinion_preds = []
        self.aspect_targets = []
        self.opinion_targets = []

    def evaluate(self, target_span, raw_words, aspects, opinions, pred):

        pred_eos_index = pred.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()
        pred = pred[:, 1:]  # delete </s>
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(
            pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        for i, (_opinions, _raw_words, ts, ps) in enumerate(zip(opinions, raw_words, target_span, pred.tolist())):
            # use ts to check whether the decoding is right or not
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in _raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.eos_token_id])
            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()

            mappingback = np.full(
                shape=(cum_lens[-1]+1), fill_value=-1, dtype=int).tolist()

            for _i, _j in enumerate(cum_lens):
                mappingback[_j] = _i

            ps = ps[:pred_seq_len[i]]
            pairs = []

            # each pair with the format (a_start, a_end, o_start, o_end, class), start/end inclusive, and considering the sos in the start of sentence
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index:
                        cur_pair.append(j)
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            pass
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            _aspects = []
            _opinions = []
            for index, pair in enumerate(pairs):
                a_s, a_e, o_s, o_e, sc = pair
                o_s = mappingback[o_s - self.target_shift]
                o_e = mappingback[o_e - self.target_shift]+1
                a_s = mappingback[a_s - self.target_shift]
                a_e = mappingback[a_e - self.target_shift]+1

                _aspects.append({
                    'index': index,
                    'from': a_s,
                    'to': a_e,
                    'polarity': self.labels[sc-2].upper(),
                    'term': ' '.join(_raw_words[a_s:a_e])
                })
                _opinions.append({
                    'index': index,
                    'from': o_s,
                    'to': o_e,
                    'term': ' '.join(_raw_words[o_s:o_e])
                })
            self.aspect_preds.append(_aspects)
            self.opinion_preds.append(_opinions)
        self.raw_words.extend(raw_words)
        self.aspect_targets.extend(aspects)
        self.opinion_targets.extend(opinions)

    def get_metric(self):
        data = []
        for raw_words, aspect_targets, opinion_targets, aspect_preds, opinion_preds in zip(
            self.raw_words, self.aspect_targets, self.opinion_targets, self.aspect_preds, self.opinion_preds
        ):
            data.append({
                'words': raw_words,
                'aspects': aspect_targets,
                'opinions': opinion_targets,
                'aspect_preds': aspect_preds,
                'opinion_preds': opinion_preds
            })

        line = json.dumps(data, indent=1)
        with open(self.fp, 'w', encoding='utf-8') as f:
            f.write(line)
        return {}


class Seq2SeqSpanMetric:
    def __init__(self, eos_token_id, num_labels, opinion_first=True):
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.ae_oe_fp = 0
        self.ae_oe_tp = 0
        self.ae_oe_fn = 0
        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.em = 0
        self.invalid = 0
        self.total = 0
        self.ae_sc_fp = 0
        self.ae_sc_tp = 0
        self.ae_sc_fn = 0
        assert opinion_first is False, "Current metric only supports aspect first"

        self.opinin_first = opinion_first

    def evaluate(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(
            self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(
            pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(
            target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index:
                        cur_pair.append(j)
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())
            self.invalid += invalid

            oe_ae_target = [tuple(t[:4]) for t in ts]
            oe_ae_pred = [p[:4] for p in pairs]

            oe_ae_target_counter = Counter(oe_ae_target)
            oe_ae_pred_counter = Counter(oe_ae_pred)
            tp, fn, fp = _compute_tp_fn_fp(set(list(oe_ae_pred_counter.keys())),
                                           set(list(oe_ae_target_counter.keys())))
            self.ae_oe_fn += fn
            self.ae_oe_fp += fp
            self.ae_oe_tp += tp

            # note aesc
            ae_sc_target = [(t[0], t[1], t[-1]) for t in ts]
            ae_sc_pred = [(p[0], p[1], p[-1]) for p in pairs]
            asts = set([tuple(t) for t in ae_sc_target])
            asps = set(ae_sc_pred)
            for p in list(asps):  # pairs is a 5-tuple
                if p in asts:
                    asts.remove(p)
                    self.ae_sc_tp += 1
                else:
                    self.ae_sc_fp += 1
            self.ae_sc_fn += len(asts)

            ts = set([tuple(t) for t in ts])
            ps = set(pairs)
            for p in list(ps):
                if p in ts:
                    ts.remove(p)
                    self.triple_tp += 1
                else:
                    self.triple_fp += 1

            self.triple_fn += len(ts)

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = self._compute_f_pre_rec(
            1, self.triple_tp, self.triple_fn, self.triple_fp)

        res['triple_f'] = round(f, 4)*100
        res['triple_rec'] = round(rec, 4)*100
        res['triple_pre'] = round(pre, 4)*100

        f, pre, rec = self._compute_f_pre_rec(
            1, self.ae_oe_tp, self.ae_oe_fn, self.ae_oe_fp)

        res['oe_ae_f'] = round(f, 4)*100
        res['oe_ae_rec'] = round(rec, 4)*100
        res['oe_ae_pre'] = round(pre, 4)*100

        f, pre, rec = self._compute_f_pre_rec(
            1, self.ae_sc_tp, self.ae_sc_fn, self.ae_sc_fp)
        res["ae_sc_f"] = round(f, 4)*100
        res["ae_sc_rec"] = round(rec, 4)*100
        res["ae_sc_pre"] = round(pre, 4)*100

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        if reset:
            self.ae_oe_fp = 0
            self.ae_oe_tp = 0
            self.ae_oe_fn = 0
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.em = 0
            self.invalid = 0
            self.total = 0
            self.ae_sc_fp = 0
            self.ae_sc_tp = 0
            self.ae_sc_fn = 0
        return res

    def _compute_f_pre_rec(self, beta, tp, fn, fp):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        if beta == 0 or precision+recall == 0:
            f = 0
        else:
            numerator = (1 + beta ** 2) * precision * recall
            denominator = (beta ** 2 * precision + recall)
            f = numerator / denominator
        return f, precision, recall


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, set):
        ts = {key: 1 for key in list(ts)}
    if isinstance(ps, set):
        ps = {key: 1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp
