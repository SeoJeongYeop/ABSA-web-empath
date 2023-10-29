from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.core.cache import cache

import re
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from empath.settings import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "absa/checkpoint")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


class IndexView(TemplateView):
    def get(self, request):

        return render(request, 'index.html')


class InferView(TemplateView):

    def get(self, request, task_id):
        print("task_id", task_id)
        context = {"pred": []}

        input_text = "액션보다는 하나하나에 좀 더 몰입하고 무게를 주기 위해서 슬로우 모션과 잔인하면서도 섬뜩한 장면들을 음향과 음악으로 채워버리는 연출은 정말 대단했다"
        pred = self.pred_absa(input_text)[0]
        valid = self.check_valid(pred)
        print('valid', valid)

        if valid:
            absa_li = re.sub(r'(<pos>|<neg>|<neu>)', '####\\1',
                             pred).split("####")[1:]
            print("absa_li", absa_li)
            triplets = []
            for absa_i in absa_li:
                absa_dict = {}
                polarity = absa_i.strip()[:5]
                absa_dict['polarity'] = polarity
                ao = absa_i.strip()[5:]
                idx = ao.find('<opinion>')
                if idx >= 0:
                    aspect = ao[:idx].strip()
                    opinion = ao[idx+9:].strip()
                    absa_dict['aspect'] = aspect
                    absa_dict['opinion'] = opinion
                    triplets.append(absa_dict)
                else:
                    valid = False

        if not valid:
            print("invalid")
            context["pred"] = []
        else:
            context["pred"] = triplets
        print('context', context)

        return render(request, 'infer.html', context=context)

    def pred_absa(self, input_text):

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
        model.eval()
        with torch.no_grad():
            outs_dict = model.generate(
                input_ids=input_ids.to('cpu'),
                attention_mask=attention_mask.to('cpu'),
                max_length=128,
                prefix_allowed_tokens_fn=None,
                output_scores=True,
                return_dict_in_generate=True
            )
            outs = outs_dict["sequences"]
            pred = [tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in outs]
            print("pred", pred)

            return pred

    def check_valid(self, pred):
        valid = True

        aps = re.findall("(<\w+>)(.*?)(?=<\w+>|$)", pred)
        for ap in aps:
            for ele in ap:
                # some element is missing
                if len(ele) == 0:
                    valid = False
                    break
            if valid is False:
                break
        return valid
