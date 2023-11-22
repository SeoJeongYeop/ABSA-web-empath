# ABSA-web-empath

2023 성균관대학교 소프트웨어학과 졸업작품 - 웹 수집 데이터 속성기반 감성분석 웹 애플리케이션

다음과 같은 기능이 포함되어 있습니다.

- 검색 수집 작업 생성
- 수집 데이터 키워드 및 속성기반 감성분석
- 분석데이터 시각화

## 프로젝트에서 사용한 기술

- Django
- Scrapy
- Oracle Database
- ABSA
<img src="https://github.com/SeoJeongYeop/ABSA-web-empath/assets/41911523/a0e0567b-7c5f-4cea-9555-90ca89b05a29" width="400"/>

## Dev Server 실행 방법

### 1. Database

settings.py에 DATABASES 변수가 다음과 같이 설정되어있는지 확인.

- NAME, USER, PASSWORD 값은 Oracle Cloud의 DB 이름, DB User, DB Password 이다.

```python
DATABASES={
    'default': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': oracle_db_name,
        'USER': DATABASE_USER,
        'PASSWORD': DATABASE_PASSWORD,
    }
}
```

cmd에서 다음 명령어 입력

```cmd
cd empath
python manage.py migrate
```

SQLite를 사용하려면 settings.py에 DATABASES 변수가 다음과 같이 설정되어있는지 확인.

```python
DATABASES={
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### 2. Django

```cmd
python manage.py runserver
```

또는

```cmd
python manage.py runserver [IP ADDRESS]:[PORT NUMBER]
```

## Production 배포 방법

- 백엔드 서버는 Oracle Cloud의 Instance를 사용하여 운영하고 있습니다.
- Database는 Oracle Cloud의 자율운영 데이터베이스를 사용하여 운영하고 있습니다.
- 백엔드 서버에서 Django를 tmux를 이용해 백그라운드로 실행 중이기 때문에 실제 성능에 이슈가 생길 수 있습니다.

## 환경 변수 및 시크릿

empath 디렉토리에 secret_sample.json 파일이 존재합니다. 다음 명령어로 파일명을 변경합니다.

```cmd
mv secret_sample.json secret.json
```

- SECRET_KEY는 Django의 SECRET_KEY 값입니다.
- DATABASE_USER와 DATABASE_PASSWORD는 Oracle DB의 값입니다.
- YOUTUBE_API_KEY는 https://console.cloud.google.com/apis에서 발급받아야 합니다.
- SERVER_ADDRESS는 Django가 실행되는 서버의 주소와 포트 번호로 이루어집니다.

```json
{
  "SECRET_KEY": "",
  "DATABASE_USER": "",
  "DATABASE_PASSWORD": "",
  "YOUTUBE_API_KEY": "YOUTUBE_API_KEY",
  "SERVER_ADDRESS": "http://127.0.0.1:8001"
}
```

## 분석 결과 화면

<img src="https://github.com/SeoJeongYeop/ABSA-web-empath/assets/41911523/fcb02629-d5ad-44c3-820b-05bd9809dbbb" width="720"/>
