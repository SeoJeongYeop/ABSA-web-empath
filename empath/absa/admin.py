from django.contrib import admin
from import_export.admin import ImportExportMixin

from absa.models import Analysis


class AnalysisAdmin(ImportExportMixin, admin.ModelAdmin):
    search_fields = ["keywords"]
    list_display = ("id", "task", "keyword", "created_at",
                    "status", "user_id", "num_sentence", "token_count")


admin.site.register(Analysis, AnalysisAdmin)
