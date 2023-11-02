from django.urls import path

from absa import views

app_name = 'absa'

urlpatterns = [
    path('', views.IndexView.as_view(), name='Index'),
    path('task/<int:task_id>/word-count/',
         views.WordCountView.as_view(), name='WordCount'),
    path('task/<int:task_id>/infer/', views.InferView.as_view(), name='Infer'),
    path('task/<int:task_id>/', views.TaskDetailView.as_view(), name='TaskDetail'),
]
