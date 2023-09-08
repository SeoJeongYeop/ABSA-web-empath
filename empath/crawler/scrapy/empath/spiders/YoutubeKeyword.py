import scrapy
from ..items import SearchResultItem
from ..settings import YOUTUBE_API_KEY
import json
from datetime import datetime
from urllib.parse import urlencode


class YoutubeKeywordSpider(scrapy.Spider):
    name = 'YoutubeKeyword'
    allowed_domains = ['youtube.com', 'googleapis.com']
    start_urls = ['https://www.googleapis.com/youtube/v3/']

    def __init__(self, keywords='', *args, **kwargs):
        self.keywords = keywords
        self.keyword_list = keywords.split(',')

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
            "maxResults": 10  # 반환하는 결과 개수
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
                    item = SearchResultItem()
                    item['keyword'] = response.meta['keyword']
                    item['video_id'] = result['id']['videoId']
                    now = datetime.now()
                    item['crawled_at'] = now.strftime("%Y-%m-%d %H:%M:%S")
                    yield item
