import json
import logging
from datetime import datetime

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.generic.base import TemplateView

from crawler.models import Task
from handle_error import get_fail_res, get_missing_data_msg


@csrf_exempt
def task_done(request):
    if request.method == 'POST':
        try:
            print(" request.POST", request.POST)
            task_id = request.POST.get('task_id')
            status = request.POST.get('status')
            task = Task.objects.get(id=task_id)
            task.status = status
            task.save()
        except Task.DoesNotExist as e:
            logging.exception(f'TaskDoneView {e}')
            return JsonResponse(get_fail_res('object_not_found'))

        return JsonResponse({'status': 'success'})


@csrf_exempt
def task_name_change(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            task_id = data.get('id')
            name = data.get('name')

            task = Task.objects.get(id=task_id)
            task.name = name
            task.save()
        except Task.DoesNotExist as e:
            logging.exception(f'TaskNameChangeView {e}')
            return JsonResponse(get_fail_res('object_not_found'))

        return JsonResponse({'status': 'success'})


class TaskCreateView(TemplateView):
    def post(self, request):
        if request.user.is_authenticated:
            # 필수
            try:
                data = json.loads(request.body.decode('utf-8'))
                platform = data.get('platform')
                keywords = data.get('keywords')
            except KeyError as e:
                logging.exception(f'MonitorView post KeyError {e}')
                appended_msg = get_missing_data_msg('platform, keywords')
                return JsonResponse(get_fail_res('missing_required_data', appended_msg))
            except Exception as e:
                logging.exception(f"Exception {e}")
                return JsonResponse(get_fail_res('undefined_exception', appended_msg))

            # 선택
            name = data.get('name', None)
            if name == None or name == '':
                now = datetime.now().strftime('%y.%m.%d')
                short_keyword = ellipsis_text(keywords)
                name = f'[{now}]{platform}:{short_keyword}'
            ds = data.get('ds', None)
            de = data.get('de', None)
            if platform == 'youtube':
                ds, de = None, None
            if ds == '':
                ds = None
            if de == '':
                de = None
            limit = data.get('limit', 5)
            if type(limit) != 'number':
                limit = 5

            print("name", name, "limit", limit)
            print("keywords", keywords)
            print("ds", ds, "de", de)

            task = Task.objects.create(
                platform=platform,
                name=name,
                keywords=keywords,
                ds=ds,
                de=de,
                limit=limit,
                status='created',
                user_id=request.user.id
            )
            print("task json", task.to_json())
            return JsonResponse({'status': 'success', 'data': task.to_json()})
        else:
            return JsonResponse(get_fail_res('require_login'))


class MonitorView(TemplateView):
    def get(self, request):
        if request.user.is_superuser:
            tasks = Task.objects.all()
        elif request.user.is_authenticated:
            tasks = Task.objects.filter(user_id=request.user.id)
        else:
            tasks = Task.objects.filter(user_id=0)
        sort_field = request.GET.get('sort', ('-created_at'))
        sort_field = set(sort_field.split(','))
        tasks = tasks.order_by(*sort_field)
        tasks = [task.to_json() for task in tasks]
        context = {'tasks': tasks}
        print("MonitorView context", context)
        return render(request, 'monitor.html', context=context)


class TaskView(TemplateView):
    def get(self, request):
        return render(request, 'taskCreator.html')


def ellipsis_text(text: str, limit=10):
    if limit < len(text) - 3:
        return text[:limit] + "..."
    elif len(text) - 3 <= limit and limit < len(text):
        return text[:limit-3] + "..."

    return text
