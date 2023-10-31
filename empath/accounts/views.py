import logging

from django.contrib.auth import views as auth_views
from django.contrib.auth.models import User
from django.db import DatabaseError, transaction
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic.base import TemplateView

from handle_error import get_fail_res


class LoginView(auth_views.LoginView):
    template_name = 'login.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        return context


class SignUpView(TemplateView):
    def get(self, request):
        template_name = 'signup.html'
        return render(request, template_name, context=None)

    def post(self, request):
        fail_reason = []

        username = request.POST.get('username')
        pw = request.POST.get('password')
        pw_check = request.POST.get('password-check')

        if len(username) < 5:
            fail_reason.append('Username은 5자 이상이여야 합니다.')
        if pw != pw_check:
            fail_reason.append('password가 일치하지 않습니다.')

        if len(fail_reason) > 0:
            return JsonResponse(get_fail_res("invalid_data_format", " ".join(fail_reason)))
        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=username,
                    password=pw
                )
                user.save()
        except DatabaseError as e:
            logging.error(f'RegisterView DatabaseError: {e}')
            return JsonResponse(get_fail_res("db_exception"))

        except Exception as e:
            logging.error(f'RegisterView Exception: {e}')
            return JsonResponse(get_fail_res("undefined_exception"))
        return JsonResponse({'status': 'success'})


def check_user(request):
    # 회원가입 페이지에서 Username Check 버튼 누를 때 유효성 확인
    if request.method == 'POST':
        username = request.POST.get('username')
        if len(username) < 5:
            return JsonResponse(get_fail_res('invalid_data_format', 'Username은 5자 이상이여야 합니다.'))
        if check_duplicate_username(username):
            return JsonResponse(get_fail_res('invalid_data_format', '중복된 Username이 있습니다.'))

        return JsonResponse({'status': 'success'})


def check_duplicate_username(username):
    user = User.objects.filter(username=username)
    return len(user) > 0
