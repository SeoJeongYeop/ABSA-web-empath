import json
import logging
import os
import re

import django.db
import torch
from django.db.models.signals import post_save
from django.dispatch import receiver
from kiwipiepy import Kiwi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from absa.models import Analysis, Sentiment, Triplet
from crawler.models import (NaverNewsArticle, YoutubeSearchResult,
                            YoutubeVideoComment)
from empath.settings import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "absa/checkpoint")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


@receiver(post_save, sender=Analysis)
def sentiment_create_start(sender, instance, created, **kwargs):
    if created:
        analysis = instance
        sentiment = Sentiment.objects.create(
            task=analysis.task,
            keyword=analysis.keyword,
            status="run"
        )
        platform = analysis.task.platform
        if platform == 'news':
            try:
                get_news_triplet(sentiment, sentiment.keyword)
                print('django.db.close_old_connections()')
                django.db.close_old_connections()
                sentiment.status = "done"
                sentiment.save()
            except Exception as e:
                logging.exception(f'{platform}: {e}')
                sentiment.status = 'error'
                sentiment.save()
        elif platform == 'youtube':
            try:
                get_youtube_triplet(sentiment, sentiment.keyword)
                print('django.db.close_old_connections()')
                django.db.close_old_connections()
                sentiment.status = "done"
                sentiment.save()
            except Exception as e:
                logging.exception(f'{platform}: {e}')
                sentiment.status = 'error'
                sentiment.save()
        else:
            print("no handle sentiment")


def get_news_triplet(sentiment: Sentiment, keyword: str):
    task = sentiment.task
    articles = NaverNewsArticle.objects.filter(
        task_id=task.id, keyword=keyword)
    kiwi = Kiwi()
    for article in articles:
        sentences = kiwi.split_into_sents(article.content)
        for sentence in sentences:
            preds = pred_absa(sentence.text)
            absa_dict = {'aspect': [], 'opinion': [], 'polarity': []}
            for pred in preds:
                valid = check_valid(pred)
                if not valid:
                    continue
                absa_li = re.sub(r'(<pos>|<neg>|<neu>)',
                                 '####\\1', pred).split("####")[1:]
                print("absa_li", absa_li)
                for absa_i in absa_li:
                    ao = absa_i.strip()[5:].strip()
                    idx = ao.find('<opinion>')
                    if idx >= 0:
                        polarity = absa_i.strip()[:5]
                        aspect = ao[:idx].strip()
                        opinion = ao[idx+9:].strip()
                        if 1 < len(aspect) < 16 and 1 < len(opinion) < 16:
                            absa_dict['polarity'].append(polarity)
                            absa_dict['aspect'].append(aspect)
                            absa_dict['opinion'].append(opinion)

            if len(absa_dict['aspect']) > 0:
                triplet = Triplet.objects.create(
                    sentiment=sentiment,
                    raw_sentence=sentence.text,
                    aspects=json.dumps(
                        absa_dict['aspect'], ensure_ascii=False),
                    opinions=json.dumps(
                        absa_dict['opinion'], ensure_ascii=False),
                    polarities=json.dumps(
                        absa_dict['polarity'], ensure_ascii=False),
                    source_news=article
                )
                print("news triplet", triplet)


def get_youtube_triplet(sentiment: Sentiment, keyword: str):
    task = sentiment.task
    video_ids = YoutubeSearchResult.objects.filter(
        task_id=task.id, keyword=keyword).values_list('video_id', flat=True)
    comments = YoutubeVideoComment.objects.filter(
        task_id=task.id, video_id__in=video_ids)
    kiwi = Kiwi()
    for comment in comments:
        sentences = kiwi.split_into_sents(comment.content)
        for sentence in sentences:
            preds = pred_absa(sentence.text)
            absa_dict = {'aspect': [], 'opinion': [], 'polarity': []}
            for pred in preds:
                valid = check_valid(pred)
                if not valid:
                    continue
                absa_li = re.sub(r'(<pos>|<neg>|<neu>)',
                                 '####\\1', pred).split("####")[1:]
                print("absa_li", absa_li)
                for absa_i in absa_li:
                    ao = absa_i.strip()[5:].strip()
                    idx = ao.find('<opinion>')
                    if idx >= 0:
                        polarity = absa_i.strip()[:5]
                        aspect = ao[:idx].strip()
                        opinion = ao[idx+9:].strip()
                        if 1 < len(aspect) < 16 and 1 < len(opinion) < 16:
                            absa_dict['polarity'].append(polarity)
                            absa_dict['aspect'].append(aspect)
                            absa_dict['opinion'].append(opinion)

            if len(absa_dict['aspect']) > 0:
                triplet = Triplet.objects.create(
                    sentiment=sentiment,
                    raw_sentence=sentence.text,
                    aspects=json.dumps(
                        absa_dict['aspect'], ensure_ascii=False),
                    opinions=json.dumps(
                        absa_dict['opinion'], ensure_ascii=False),
                    polarities=json.dumps(
                        absa_dict['polarity'], ensure_ascii=False),
                    source_youtube=comment
                )
                print("youtube triplet", triplet)


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


def check_valid(pred):
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
