import os
import logging
import subprocess
from datetime import datetime

import django.db
from django.db.models.signals import post_save
from django.dispatch import receiver

from crawler.models import Task, YoutubeSearchResult
from empath.settings import BASE_DIR, CRAWLING_LOG_PATH


@receiver(post_save, sender=Task)
def task_create_start(sender, instance, created, **kwargs):
    if created:
        task = instance
        if task.platform == 'news':
            try:
                opts = {
                    'task_id': task.id,
                    'keywords': task.keywords,
                    'ds': task.ds,
                    'de': task.de,
                    'limit': task.limit
                }
                run_spider('NaverNewsKeyword', opts)
                task.status = 'run'
                task.save()
                print('django.db.close_old_connections()')
                django.db.close_old_connections()
            except Exception as e:
                logging.exception(f'{task.platform}: {e}')
                task.status = 'error'
                task.save()

        elif task.platform == 'youtube':
            try:
                opts = {
                    'task_id': task.id,
                    'keywords': task.keywords,
                    'limit': task.limit
                }
                task.status = 'run'
                task.save()
                process = run_spider('YoutubeKeyword', opts)
                process.wait()

                if process.returncode == 0:

                    vids = YoutubeSearchResult.objects.filter(
                        task_id=task.id).values_list('video_id', flat=True)
                    vids = ",".join(vids)
                    opts = {
                        'task_id': task.id,
                        'ids': vids
                    }
                    task.status = 'run2'
                    task.save()
                    run_spider('YoutubeVideo', opts)
                else:
                    print("명령 실행 중 에러가 발생했습니다.")
                    logging.exception(f'{task.platform}: {e}')
                    task.status = 'error'
                    task.save()

                print('django.db.close_old_connections()')
                django.db.close_old_connections()
            except Exception as e:
                logging.exception(f'{task.platform}: {e}')
                task.status = 'error'
                task.save()
        else:
            print("no handle task")


def run_spider(spider_name, opts):
    print(f'[{datetime.now()}] run {spider_name}')

    shell_opts = ''
    for key, value in opts.items():
        if value is None:
            continue
        elif key == 'keywords' and ' ' in value:
            a_opt = f' -a {key}="{value}"'
        else:
            a_opt = f' -a {key}={value}'
        shell_opts += a_opt

    log_file = datetime.now().strftime(f"{spider_name}_%y%m%d_%H%M.log")
    log_path = os.path.join(CRAWLING_LOG_PATH, f'{log_file}')

    process = subprocess.Popen(
        f'scrapy crawl {spider_name} {shell_opts} --loglevel=INFO --logfile={log_path}',
        shell=True,
        cwd=os.path.join(BASE_DIR, 'crawler/scrapy/empath'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return process
