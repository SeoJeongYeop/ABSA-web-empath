from datetime import timedelta

from django.db import models


class YoutubeSearchResult(models.Model):
    id = models.AutoField(primary_key=True)
    keyword = models.CharField(max_length=100)
    video_id = models.CharField(max_length=20)
    crawled_at = models.DateTimeField(auto_now_add=True)
    task_id = models.IntegerField(null=True)


class YoutubeVideo(models.Model):
    id = models.AutoField(primary_key=True)
    video_id = models.CharField(max_length=20, unique=True, db_index=True)
    title = models.CharField(max_length=200)
    published_at = models.DateTimeField()
    description = models.TextField()
    channel_id = models.CharField(max_length=32)
    channel_title = models.CharField(max_length=100)
    tags = models.TextField()
    category_id = models.IntegerField()
    comment_count = models.IntegerField()
    crawled_at = models.DateTimeField(auto_now_add=True)
    task_id = models.IntegerField(null=True)

    def to_json(self):
        nine_hours = timedelta(hours=9)
        kst = self.crawled_at + nine_hours
        return {
            'video_id': self.video_id,
            'title': self.title,
            'published_at': self.published_at,
            'channel_title': self.channel_title,
            'comment_count': self.comment_count,
            'crawled_at': kst.strftime("%Y-%m-%d %H:%M"),
            'task_id': self.task_id,
        }


class YoutubeVideoComment(models.Model):
    id = models.AutoField(primary_key=True)
    content = models.TextField()
    video_id = models.CharField(max_length=20)
    crawled_at = models.DateTimeField(auto_now_add=True)
    task_id = models.IntegerField(null=True)

    def to_json(self):
        nine_hours = timedelta(hours=9)
        kst = self.crawled_at + nine_hours
        return {
            'id': self.id,
            'content': self.content,
            'video_id': self.video_id,
            'crawled_at': kst.strftime("%Y-%m-%d %H:%M"),
            'task_id': self.task_id
        }


class YoutubeChannel(models.Model):
    id = models.AutoField(primary_key=True)
    channel_id = models.CharField(max_length=32, unique=True, db_index=True)
    title = models.CharField(max_length=100)
    description = models.TextField()
    published_at = models.DateTimeField()
    subscriber_count = models.IntegerField()
    video_count = models.IntegerField()
    crawled_at = models.DateTimeField(auto_now_add=True)
    task_id = models.IntegerField(null=True)


class NaverSearchResult(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=100)
    keyword = models.CharField(max_length=100)
    press_name = models.CharField(max_length=20)
    link = models.CharField(max_length=200)
    summary = models.CharField(max_length=200)
    crawled_at = models.DateTimeField(auto_now_add=True)
    task_id = models.IntegerField(null=True)


class NaverNewsArticle(models.Model):
    id = models.AutoField(primary_key=True)
    oid = models.CharField(max_length=4)
    aid = models.CharField(max_length=20)
    title = models.CharField(max_length=100)
    content = models.TextField()
    published_at = models.DateTimeField()
    press_name = models.CharField(max_length=20)
    crawled_at = models.DateTimeField(auto_now_add=True)
    task_id = models.IntegerField(null=True)
    keyword = models.CharField(max_length=100, null=True)

    def to_json(self):
        nine_hours = timedelta(hours=9)
        kst = self.crawled_at + nine_hours
        return {
            'id': self.id,
            'oid': self.oid,
            'aid': self.aid,
            'title': self.title,
            'content': self.content,
            'published_at': self.published_at,
            'crawled_at': kst.strftime("%Y-%m-%d %H:%M"),
            'press_name': self.press_name,
            'task_id': self.task_id,
            'keyword': self.keyword
        }


class Task(models.Model):
    id = models.AutoField(primary_key=True)
    # platform: news, youtube
    platform = models.CharField(max_length=100)
    name = models.CharField(max_length=100, null=True)
    keywords = models.TextField()  # 콤마로 구분된 키워드 목록
    created_at = models.DateTimeField(auto_now_add=True)
    ds = models.DateTimeField(null=True)
    de = models.DateTimeField(null=True)
    limit = models.IntegerField(null=True)
    # status 상태 구분
    status = models.CharField(max_length=20)
    user_id = models.IntegerField(null=True)

    def to_json(self):
        nine_hours = timedelta(hours=9)
        kst = self.created_at + nine_hours
        return {
            'id': self.id,
            'platform': self.platform,
            'name': self.name,
            'keywords': self.keywords.split(","),
            'created_at': kst.strftime("%Y-%m-%d %H:%M"),
            'ds': self.ds,
            'de': self.de,
            'limit': self.limit,
            'status': self.status,
            'user_id': self.user_id
        }
