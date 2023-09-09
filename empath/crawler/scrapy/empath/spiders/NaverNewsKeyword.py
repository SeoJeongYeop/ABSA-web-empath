import scrapy
from urllib import parse
from bs4 import BeautifulSoup

import logging
from datetime import datetime

from ..items import *
from ..preprocess import cleaning_text, remove_escape

NAVER_SEARCH_LINK = 'https://search.naver.com/search.naver'


class NaverNewsKeywordSpider(scrapy.Spider):
    name = 'NaverNewsKeyword'
    allowed_domains = ['search.naver.com', 'n.news.naver.com']

    def __init__(self, keywords='', ds='', de='', limit='', **kwargs):
        super().__init__(**kwargs)
        try:
            # 키워드의 공백 제거하여 저장
            self.keywords = [keyword.strip()
                             for keyword in keywords.split(',')]

            self.date_filter = False

            # ds: date_start, de: date_end
            if ds != '' and de != '':
                # 사이 간격의 날짜로 필터링해서 검색
                self.ds, self.de = ds, de
                self.date_filter = True
            elif ds != '':
                # de를 today로 설정하여 검색
                self.ds, self.de = ds, datetime.today().strftime("%Y.%m.%d")
                self.date_filter = True

            # 스크랩할 기사 개수
            self.limit = int(limit) if limit != '' else 100
            self.count = 0
            # 쿼리스트링
            self.params = {
                'where': 'news',
                'start': 1,
            }
        except Exception as e:
            logging.exception(f"{self.name} 파라미터 설정 오류: {e}")

    def start_requests(self):
        if self.date_filter:
            self.params['ds'] = self.ds
            self.params['de'] = self.de
            from_date = self.ds.replace(".", "")
            to_date = self.de.replace(".", "")
            self.params['nso'] = f'so%3Ar%2Cp%3Afrom{from_date}to{to_date}'

        for keyword in self.keywords:
            self.params['query'] = f'\"{keyword}\"'
            query_string = parse.urlencode(self.params)
            yield scrapy.Request(
                f'{NAVER_SEARCH_LINK}?{query_string}',
                self.parse_news_list,
                meta={'keyword': keyword, 'start': 1}
            )

    def parse_news_list(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')
        news_items = soup.select('div.news_area')

        for news in news_items:
            news_anchor = news.select_one(
                'div.info_group > a.info:not(.press)')
            if news_anchor is None:
                continue
            link = news_anchor['href']
            if not 'sid' not in parse.urlparse(link):
                # 정상적으로 크롤링 가능한 기사인지 확인
                continue

            # 뉴스 제목 파싱
            title = news.select_one('a.news_tit').text
            # 언론사 파싱
            press_box = news.select_one('div.info_group > a.info.press')
            press_name = press_box.text
            press_icon = press_box.select_one('i.ico_pick')
            if press_icon:
                press_name = press_name.replace(press_icon.text, '').strip()

            # 요약정보 파싱
            summary = news.select_one('a.dsc_txt_wrap').text

            # 키워드에 따른 뉴스 데이터 산출
            newsItem = NaverNewsItem()
            newsItem['title'] = title
            newsItem['keyword'] = response.meta['keyword']
            newsItem['press_name'] = press_name
            newsItem['link'] = link
            newsItem['summary'] = summary
            newsItem['crawled_at'] = datetime.now()
            yield newsItem

            # 뉴스 게시글 파싱 Request
            yield scrapy.Request(
                url=link,
                callback=self.parse_news_page,
                meta={
                    'keyword': response.meta['keyword'],
                    'summary': summary,
                    'press_name': press_name
                },
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'}
            )

        # 다음 페이지 이동하여 재귀적으로 스크래핑
        self.count += len(news_items)
        if len(news_items) != 0 and self.count < self.limit:
            self.params['start'] = response.meta['start'] + 10
            query_string = parse.urlencode(self.params)

            yield scrapy.Request(
                f'{NAVER_SEARCH_LINK}?{query_string}',
                self.parse_news_list,
                meta={'keyword': response.meta["keyword"],
                      'start': response.meta['start'] + 10},
                dont_filter=True
            )

    def article_date_to_datetime(self, article_date: str):
        '''
        인터넷 기사의 한국어 날짜입력을 datetime으로 변환
        '''

        # 오후/오후를 대/소문자로 변환
        article_date = article_date.replace("오후", "PM").replace("오전", "AM")

        # 문자열을 datetime 객체로 파싱
        datetime_obj = datetime.strptime(article_date, "%Y.%m.%d. %p %I:%M")

        return datetime_obj

    def parse_news_page(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')

        title = soup.select_one("#title_area > span").text
        title = remove_escape(title)
        content = soup.select_one("#dic_area").text
        logging.debug(f"content before {content}")
        content = cleaning_text(content)
        logging.debug(f"content after {content}")
        published_at = soup.select_one(
            "span.media_end_head_info_datestamp_time").text
        published_at = self.article_date_to_datetime(published_at)

        press_name = soup.select_one('.media_end_head_top_logo > img')['alt']

        parsed_url = parse.urlparse(response.url)
        path_params = parsed_url.path.split("/")
        aid = path_params[-1]
        oid = path_params[-2]

        articleItem = NaverNewsArticleItem()
        articleItem['oid'] = oid
        articleItem['aid'] = aid
        articleItem['title'] = title
        articleItem['content'] = content
        articleItem['published_at'] = published_at
        articleItem['press_name'] = press_name
        articleItem['crawled_at'] = datetime.now()

        yield articleItem
