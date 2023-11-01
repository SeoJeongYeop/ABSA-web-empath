import json
from django.shortcuts import render
from django.views.generic.base import TemplateView
from crawler.models import Task, NaverNewsArticle, YoutubeSearchResult, YoutubeVideoComment, YoutubeVideo
from absa.models import Analysis
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from collections import Counter


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
        print("TaskDetailView", task)
        context['task'] = task.to_json()
        print(context['task'])
        analysis = Analysis.objects.filter(task=task)
        if len(analysis) > 0:
            context['analysis'] = {}
            for analy in analysis:
                context['analysis'][analy.keyword] = analy.to_json()
            print("context", context)

            return render(request, 'taskDetail.html', context=context)
        print("Analysis 없어서 분석")
        # analysis가 없으면 메소드 호출해서 analysis 데이터 만듦
        keywords = task.keywords.split(",")
        analysis = {}
        for keyword in keywords:
            if task.platform == "news":
                data = self.get_news_data(task, keyword)
            elif task.platform == "youtube":
                data = self.get_youtube_data(task, keyword)
            analysis[keyword] = data

        context['analysis'] = analysis
        return render(request, 'taskDetail.html', context=context)

    def get_news_data(self, task: Task, keyword: str):
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

        # ret = {"n_sentence": total_n_sentence,
        #        "token_count": sorted_token_counts[:10],
        #        "data": articles}
        return ret

    def get_youtube_data(self, task: Task, keyword: str):

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
        # ret = {"n_sentence": total_n_sentence,
        #        "token_count": sorted_token_counts[:10],
        #        "data": videos}
        return ret


ALLOW_TAG = ['NNG', 'NNP', 'NP', 'VV', 'VA', 'MM', 'MAG']
