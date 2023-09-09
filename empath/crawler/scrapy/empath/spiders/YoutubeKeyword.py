import scrapy
from ..items import YoutubeSearchResultItem
from ..settings import YOUTUBE_API_KEY
import json
from datetime import datetime
from urllib.parse import urlencode


class YoutubeKeywordSpider(scrapy.Spider):
    '''
    YouTube Data API의 Search: list를 사용하여 유튜브 검색결과를 수집합니다.
    수집 개수와 관계없이 호출할 때 마다 API Cost가 100 사용됩니다.
    '''
    name = 'YoutubeKeyword'
    allowed_domains = ['youtube.com', 'googleapis.com']
    start_urls = ['https://www.googleapis.com/youtube/v3/']

    def __init__(self, keywords='', limit='', *args, **kwargs):
        self.keywords = keywords
        self.keyword_list = keywords.split(',')
        try:
            self.limit = int(limit) if limit != '' else 5
            self.limit = min(self.limit, 50)
        except:
            self.limit = 5

    def start_requests(self):
        for keyword in self.keyword_list:
            item = self.get_search(keyword)
            if item is not None:
                yield item

    def get_search(self, keyword):

        BASE_SEARCH_URL = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "key": YOUTUBE_API_KEY,  # api key
            "q": keyword,  # 검색어
            "type": "video",  # video, channel, playlist 중에 리소스 선택
            "maxResults": self.limit  # 반환하는 결과 개수
        }
        query_string = urlencode(params)
        SEARCH_URL = f"{BASE_SEARCH_URL}?{query_string}"

        return scrapy.Request(SEARCH_URL, callback=self.parse, meta={'keyword': keyword})

    def parse(self, response):

        res_json = json.loads(response.body)

        if "items" in res_json:
            results = res_json['items']
            for result in results:
                if result['id']['kind'] == 'youtube#video':
                    item = YoutubeSearchResultItem()
                    item['keyword'] = response.meta['keyword']
                    item['video_id'] = result['id']['videoId']
                    item['crawled_at'] = datetime.now()
                    yield item
