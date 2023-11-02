import json
from collections import Counter

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from django.views.generic.base import TemplateView
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

from absa.models import Analysis, Sentiment, Triplet
from crawler.models import (
    NaverNewsArticle, Task, YoutubeSearchResult, YoutubeVideo, YoutubeVideoComment)


class IndexView(TemplateView):
    def get(self, request):
        if request.user.is_superuser:
            tasks = Task.objects.filter(status="done")
        elif request.user.is_authenticated:
            tasks = Task.objects.filter(user_id=request.user.id, status="done")
        else:
            tasks = Task.objects.filter(user_id=0, status="done")
        sort_field = request.GET.get('sort', ('-created_at'))
        sort_field = set(sort_field.split(','))
        tasks = tasks.order_by(*sort_field)
        tasks = [task.to_json() for task in tasks]
        context = {'tasks': tasks}

        return render(request, 'index.html', context=context)


class TaskDetailView(TemplateView):
    def get(self, request, task_id):
        context = {}
        task = Task.objects.get(id=task_id)
        context['task'] = task.to_json()
        print(context['task'])
        analysis = Analysis.objects.filter(task=task)
        context['analysis'] = {}
        if len(analysis) > 0:
            for analy in analysis:
                context['analysis'][analy.keyword] = analy.to_json()
        print("context", context)
        return render(request, 'taskDetail.html', context=context)


class WordCountView(View):
    def get(self, request, task_id):
        data = {}
        task = Task.objects.get(id=task_id)

        analysis = Analysis.objects.filter(task=task)
        if len(analysis) > 0:
            for analy in analysis:
                data[analy.keyword] = analy.to_json()
            print("data", data)
            return JsonResponse(data)

        # analysis가 없으면 메소드 호출해서 analysis 데이터 만듦
        keywords = task.keywords.split(",")
        for keyword in keywords:
            if task.platform == "news":
                analysis = get_news_data(task, keyword)
            elif task.platform == "youtube":
                analysis = get_youtube_data(task, keyword)
            data[keyword] = analysis

        return JsonResponse(data)


class InferView(View):

    def get(self, request, task_id):
        ret = {'sentiments': {}, 'triplets': {}}
        print("task_id", task_id)
        keywords = request.GET.get("keywords", "")  # 콤마로 연결된 문자열
        keywords = keywords.split(",")
        print("keywords", type(keywords), keywords)
        sentiments = Sentiment.objects.filter(
            task_id=task_id, keyword__in=keywords)
        print("sentiments", sentiments)

        for sentiment in sentiments:
            if sentiment.status == 'done':
                triplets = Triplet.objects.filter(sentiment_id=sentiment.id)
                triplets_json = [triplet.to_json() for triplet in triplets]
                ret['triplets'][sentiment.keyword] = triplets_json
            ret['sentiments'][sentiment.keyword] = sentiment.to_json()
        return JsonResponse(ret)


def get_news_data(task: Task, keyword: str):
    articles = NaverNewsArticle.objects.filter(
        task_id=task.id, keyword=keyword)

    articles = [article.to_json() for article in articles]

    kiwi = Kiwi()
    stopwords = Stopwords()
    total_token_counts = Counter()
    total_n_sentence = 0
    for article in articles:
        sentences = kiwi.split_into_sents(article['content'])
        n_sentence = len(sentences)
        total_n_sentence += n_sentence
        article['n_sentence'] = n_sentence

        article_token_counts = Counter()
        for sentence in sentences:
            tokens = kiwi.tokenize(
                sentence.text,
                normalize_coda=True,
                stopwords=stopwords
            )
            tokens_filtered = []
            for token in tokens:
                if token.tag in ALLOW_TAG:
                    tokens_filtered.append(token.form)
            sentence_token_counts = Counter(tokens_filtered)
            article_token_counts.update(sentence_token_counts)
            total_token_counts.update(sentence_token_counts)
        sorted_token_counts = sorted(
            article_token_counts.items(), key=lambda x: x[1], reverse=True)
        article['token_count'] = sorted_token_counts[:10]

    sorted_token_counts = sorted(
        total_token_counts.items(), key=lambda x: x[1], reverse=True)

    analysis = Analysis.objects.create(
        task=task,
        keyword=keyword,
        status='text',
        user_id=task.user_id,
        num_sentence=total_n_sentence,
        token_count=json.dumps(
            sorted_token_counts[:10], ensure_ascii=False)
    )
    print("analysis", analysis)
    ret = analysis.to_json()

    return ret


def get_youtube_data(task: Task, keyword: str):

    video_ids = YoutubeSearchResult.objects.filter(
        task_id=task.id, keyword=keyword).values_list('video_id', flat=True)
    comments = YoutubeVideoComment.objects.filter(
        task_id=task.id, video_id__in=video_ids)

    video_dict = {}
    for comment in comments:
        video_id = comment.video_id
        if video_id in video_dict:
            video_dict[video_id]['comments'].append(comment.to_json())
            pass
        else:
            video = YoutubeVideo.objects.get(video_id=comment.video_id)
            video_dict[video_id] = video.to_json()
            video_dict[video_id]['comments'] = [comment.to_json()]

    videos = [video_info for video_info in video_dict.values()]

    kiwi = Kiwi()
    stopwords = Stopwords()
    total_token_counts = Counter()
    total_n_sentence = 0
    for video in videos:
        comments = video['comments']
        video_token_counts = Counter()
        video['n_sentence'] = 0
        for comment in comments:
            sentences = kiwi.split_into_sents(comment['content'])
            n_sentence = len(sentences)
            video['n_sentence'] += n_sentence
            total_n_sentence += n_sentence

            for sentence in sentences:
                tokens = kiwi.tokenize(
                    sentence.text,
                    normalize_coda=True,
                    stopwords=stopwords
                )
                tokens_filtered = []
                for token in tokens:
                    if token.tag in ALLOW_TAG:
                        tokens_filtered.append(token.form)
                sentence_token_counts = Counter(tokens_filtered)
                video_token_counts.update(sentence_token_counts)
                total_token_counts.update(sentence_token_counts)
        sorted_token_counts = sorted(
            video_token_counts.items(), key=lambda x: x[1], reverse=True)
        video['token_count'] = sorted_token_counts[:10]

    sorted_token_counts = sorted(
        total_token_counts.items(), key=lambda x: x[1], reverse=True)

    analysis = Analysis.objects.create(
        task=task,
        keyword=keyword,
        status='text',
        user_id=task.user_id,
        num_sentence=total_n_sentence,
        token_count=json.dumps(
            sorted_token_counts[:10], ensure_ascii=False)
    )
    print("analysis", analysis)
    ret = analysis.to_json()

    return ret


ALLOW_TAG = ['NNG', 'NNP', 'NP', 'VV', 'VA', 'MM', 'MAG']
