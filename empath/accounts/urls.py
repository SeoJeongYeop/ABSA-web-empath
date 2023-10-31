from django.urls import path
from accounts import views
from django.contrib.auth import views as auth_views

app_name = 'accounts'

urlpatterns = [
    path('login/', views.LoginView.as_view(), name='Login'),
    path('signup/', views.SignUpView.as_view(), name="SignUp"),
    path('logout/', auth_views.LogoutView.as_view(), name='Logout'),
    path('check-user/', views.check_user, name='CheckUser')
]
