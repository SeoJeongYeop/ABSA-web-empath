from django.urls import path

from absa import views

app_name = 'absa'

urlpatterns = [
    path('', views.IndexView.as_view(), name='Index'),
    path('infer/<int:task_id>', views.InferView.as_view(), name='Infer')
]
