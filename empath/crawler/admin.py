from django.contrib import admin
from import_export.admin import ImportExportMixin

from crawler.models import (NaverNewsArticle, NaverSearchResult, Task,
                            YoutubeSearchResult, YoutubeVideo,
                            YoutubeVideoComment)


class NaverSearchResultAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["title", "keyword", "press_name"]
    list_display = ("id", "title", "keyword",
                    "press_name", "summary")


class NaverNewsArticleAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["title", "keyword", "press_name", "task_id"]
    list_display = ("id", "title", "keyword", "press_name",
                    "content", "published_at", "task_id")


class YoutubeSearchResultAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["keyword", "video_id"]
    list_display = ("id", "keyword", "video_id", "crawled_at")


class YoutubeVideoAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["video_id", "title", "channel_title"]
    list_display = ("id", "video_id", "title",
                    "published_at", "description", "channel_id", "channel_title", "tags", "category_id", "comment_count")


class YoutubeVideoCommentAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["content", "video_id"]
    list_display = ("id", "content", "video_id", "crawled_at")


class TaskAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["id", "name", "keywords", "platform"]
    list_display = ("id", "platform", "name", "keywords",
                    "created_at", "ds", "de", "limit", "status")


admin.site.register(NaverSearchResult, NaverSearchResultAdmin)
admin.site.register(NaverNewsArticle, NaverNewsArticleAdmin)
admin.site.register(YoutubeSearchResult, YoutubeSearchResultAdmin)
admin.site.register(YoutubeVideo, YoutubeVideoAdmin)
admin.site.register(YoutubeVideoComment, YoutubeVideoCommentAdmin)
admin.site.register(Task, TaskAdmin)
