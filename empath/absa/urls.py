from django.urls import path

from absa import views, views_absa

app_name = 'absa'

urlpatterns = [
    path('', views.IndexView.as_view(), name='Index'),
    path('task/<int:task_id>/word-count/',
         views.WordCountView.as_view(), name='WordCount'),
    path('task/<int:task_id>/', views.TaskDetailView.as_view(), name='TaskDetail'),
    path('infer/<int:task_id>/', views_absa.InferView.as_view(), name='Infer')
]
