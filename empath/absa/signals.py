import json
import logging
import re
from collections import Counter

import django.db
from django.db.models.signals import post_save
from django.dispatch import receiver
from kiwipiepy import Kiwi

from absa.generator import pred_absa
from absa.models import Analysis, Sentiment, Triplet
from crawler.models import (NaverNewsArticle, YoutubeSearchResult,
                            YoutubeVideoComment)


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
            if check_wrong_sentence(sentence.text):
                continue
            preds = pred_absa(sentence.text)
            absa_dict = {'aspect': [], 'opinion': [], 'polarity': []}
            for pred in preds:
                valid = check_valid(pred)
                if not valid:
                    continue
                absa_li = re.sub(r'(<pos>|<neg>|<neu>)',
                                 '####\\1', pred).split("####")[1:]
                for absa_i in absa_li:
                    ao = absa_i.strip()[5:].strip()
                    idx = ao.find('<opinion>')
                    if idx >= 0:
                        polarity = absa_i.strip()[:5]
                        aspect = ao[:idx].strip()
                        opinion = ao[idx+9:].strip()
                        if check_wrong_ao(aspect, opinion, sentence.text):
                            continue
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
            if check_wrong_sentence(sentence.text):
                continue
            preds = pred_absa(sentence.text)
            absa_dict = {'aspect': [], 'opinion': [], 'polarity': []}
            for pred in preds:
                valid = check_valid(pred)
                if not valid:
                    continue
                absa_li = re.sub(r'(<pos>|<neg>|<neu>)',
                                 '####\\1', pred).split("####")[1:]
                for absa_i in absa_li:
                    ao = absa_i.strip()[5:].strip()
                    idx = ao.find('<opinion>')
                    if idx >= 0:
                        polarity = absa_i.strip()[:5]
                        aspect = ao[:idx].strip()
                        opinion = ao[idx+9:].strip()
                        if check_wrong_ao(aspect, opinion, sentence.text):
                            continue
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


def check_valid(pred):
    valid = True
    if detect_repeat(pred):
        return False

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


def check_wrong_sentence(text: str):
    '''
    문장이 너무 짧거나 긴 경우 True 반환
    '''
    if len(text.strip().split()) <= 3:
        return True
    if 10 < len(text) < 128:  # 학습 토큰수
        return False
    return True


def check_wrong_ao(aspect: str, opinion: str, sentence: str):
    a = aspect.strip()
    o = opinion.strip()
    if '<opinion>' in (a + o):
        return True
    if a == o:
        return True
    if a not in sentence:
        return True
    if 1 < len(a) < 16 and 1 < len(o) < 16:
        return False
    return True


def detect_repeat(text: str):
    '''
    1. 같은 글자가 5개 이상 반복되는지 확인
    2. 띄어쓰기로 split 후 같은 단어가 5개 이상 반복되는지 확인
    '''
    if any(count >= 5 for count in Counter(text).values()):
        return True
    words = text.split()
    if any(count >= 5 for count in Counter(words).values()):
        return True
    return False
