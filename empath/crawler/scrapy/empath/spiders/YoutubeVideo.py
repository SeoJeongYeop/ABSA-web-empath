import scrapy
from ..items import YoutubeVideoItem, YoutubeCommentItem
from ..settings import YOUTUBE_API_KEY
from ..preprocess import cleaning_text, remove_emoji, remove_jamo, reduce_escape

import json
from datetime import datetime
import logging
from urllib import parse


class YoutubeVideoSpider(scrapy.Spider):
    '''
    YouTube Data API의 Videos: list를 사용해 영상의 정보를 수집합니다.
    또한 CommentThreads: list를 사용해 유튜브 댓글의 정보를 수집합니다.
    수집 개수 당 API Cost가 1씩 사용됩니다.
    '''

    name = 'YoutubeVideo'
    allowed_domains = ['youtube.com', 'googleapis.com']
    start_urls = ['https://www.googleapis.com/youtube/v3/']

    def __init__(self, ids='', limit='', *args, **kwargs):
        self.ids = ids.split(',')
        try:
            self.limit = int(limit) if limit != '' else 5
            self.limit = min(self.limit, 50)

            self.params = {
                "part": "snippet",
                "maxResults": self.limit,
                "order": "relevance",  # 인기순
                "key": YOUTUBE_API_KEY,
            }
        except:
            self.limit = 5

    def start_requests(self):
        for vid in self.ids:
            item = self.get_video_info(vid)
            if item is not None:
                yield item

    def get_video_info(self, vid):
        BASE_URL = f"https://www.googleapis.com/youtube/v3/"
        # video 정보를 얻기 위해 Videos API 사용
        SEARCH_URL = f'{BASE_URL}videos?part=id,snippet,statistics&id={vid}&key={YOUTUBE_API_KEY}'
        return scrapy.Request(SEARCH_URL, callback=self.parse_video, meta={'video_id': vid})

    def parse_video(self, response):
        '''
        Videos: list 응답을 처리
        API Cost는 댓글 하나당 1
        '''
        res_json = json.loads(response.body)
        item = YoutubeVideoItem()
        elements = res_json['items']
        for element in elements:
            try:
                item['video_id'] = element['id']
                snippet = element['snippet']

                title = snippet['title']
                title, _ = remove_emoji(title)
                item['title'] = title

                if 'description' in snippet:
                    description = snippet['description']
                    description = cleaning_text(description)
                    description, _ = remove_jamo(description)
                    description = reduce_escape(description)
                    item['description'] = description
                else:
                    item['description'] = None

                item['channel_id'] = snippet['channelId']
                item['channel_title'] = snippet['channelTitle']
                item['tags'] = "|".join(
                    snippet['tags']) if 'tags' in snippet else None
                item['category_id'] = snippet['categoryId'] if 'categoryId' in snippet else None

                statistics = element['statistics']
                item['comment_count'] = statistics['commentCount'] if 'commentCount' in statistics else None

                # datetime 형식이 동일하지 않아 통일하는 작업 필요
                published_at = self.youtube_date_to_datetime(
                    snippet['publishedAt'])
                item['published_at'] = published_at
                item['crawled_at'] = datetime.now()

                yield item

                # commentThread
                COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
                vid = response.meta['video_id']
                self.params['videoId'] = vid
                query_string = parse.urlencode(self.params)

                yield scrapy.Request(f'{COMMENTS_URL}?{query_string}', callback=self.parse_comments, meta={'video_id': vid})
            except Exception as e:
                logging.exeption(f"ERROR parse {element['id']} res {e}")

    def youtube_date_to_datetime(self, youtube_date: str):
        '''
        유튜브 날짜 형식이 통일되지 않아 datetime으로 변환
        '''
        # case 1
        try:
            # .%f가 있는 형식을 없는 형식으로 변경
            published_at = datetime.strptime(
                youtube_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            return published_at
        except:
            pass
        # case 2
        try:
            published_at = datetime.strptime(
                youtube_date, "%Y-%m-%dT%H:%M:%SZ")
            return published_at
        except:
            pass
        return None

    def parse_comments(self, response):
        '''
        https://www.googleapis.com/youtube/v3/commentThreads
        CommentThreads: list 응답을 처리
        API Cost는 댓글 하나당 1
        '''

        res_json = json.loads(response.body)
        elements = res_json["items"]
        commentItem = YoutubeCommentItem()
        for element in elements:
            # 유튜브 댓글 받아서 정제하여 저장
            snippet = element["snippet"]["topLevelComment"]["snippet"]
            content = snippet["textOriginal"]
            content = cleaning_text(content)
            content, _ = remove_emoji(content)
            content = reduce_escape(content)
            commentItem['content'] = content
            commentItem['video_id'] = response.meta['video_id']
            commentItem['crawled_at'] = datetime.now()

            yield commentItem
