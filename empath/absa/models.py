import json
from datetime import timedelta

from django.db import models
from crawler.models import Task


class Analysis(models.Model):
    id = models.AutoField(primary_key=True)
    task = models.ForeignKey(Task, models.CASCADE)
    keyword = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20)
    user_id = models.IntegerField(null=True)
    num_sentence = models.IntegerField(null=True)
    token_count = models.TextField(null=True)

    def to_json(self):
        nine_hours = timedelta(hours=9)
        kst = self.created_at + nine_hours
        return {
            "id": self.id,
            "task_id": self.task.id,
            "keyword": self.keyword,
            "created_at": kst.strftime("%Y-%m-%d %H:%M"),
            "status": self.status,
            "user_id": self.user_id,
            "num_sentence": self.num_sentence,
            "token_count": json.loads(self.token_count)
        }
