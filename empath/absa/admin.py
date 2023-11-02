from django.contrib import admin
from import_export.admin import ImportExportMixin

from absa.models import Analysis, Sentiment, Triplet


class AnalysisAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["keywords"]
    list_display = ("id", "task", "keyword", "created_at",
                    "status", "user_id", "num_sentence", "token_count")


class SentimentAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["keywords"]
    list_display = ("id", "task", "keyword", "created_at",
                    "status")


class TripletAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["keywords"]
    list_display = ("id", "sentiment", "raw_sentence", "aspects",
                    "opinions", "polarities", "source_news", "source_youtube")


admin.site.register(Analysis, AnalysisAdmin)
admin.site.register(Sentiment, SentimentAdmin)
admin.site.register(Triplet, TripletAdmin)
