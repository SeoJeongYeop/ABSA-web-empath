from django.urls import path

from crawler import views

app_name = 'crawler'

urlpatterns = [
    path('', views.TaskView.as_view(), name='Task'),
    path('task/', views.MonitorView.as_view(), name='Monitor'),
    path('task/create/', views.TaskCreateView.as_view(), name='TaskCreate'),
    path('task/done/', views.TaskDoneView.as_view(), name='TaskDone'),
]
