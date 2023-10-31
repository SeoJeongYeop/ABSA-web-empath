# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class YoutubeSearchResultItem(scrapy.Item):
    keyword = scrapy.Field()
    video_id = scrapy.Field()
    crawled_at = scrapy.Field()
    task_id = scrapy.Field()


class YoutubeVideoItem(scrapy.Item):
    video_id = scrapy.Field()
    # snippet
    title = scrapy.Field()
    published_at = scrapy.Field()
    description = scrapy.Field()
    channel_id = scrapy.Field()
    channel_title = scrapy.Field()
    tags = scrapy.Field()
    category_id = scrapy.Field()
    #  statistics
    comment_count = scrapy.Field()
    crawled_at = scrapy.Field()
    task_id = scrapy.Field()


class YoutubeCommentItem(scrapy.Item):
    content = scrapy.Field()
    video_id = scrapy.Field()
    crawled_at = scrapy.Field()
    task_id = scrapy.Field()


class YoutubeChannelItem(scrapy.Item):
    channel_id = scrapy.Field()
    title = scrapy.Field()
    description = scrapy.Field()
    published_at = scrapy.Field()
    #  statistics
    subscriber_count = scrapy.Field()
    video_count = scrapy.Field()
    crawled_at = scrapy.Field()
    task_id = scrapy.Field()


class NaverNewsItem(scrapy.Item):
    title = scrapy.Field()
    keyword = scrapy.Field()
    press_name = scrapy.Field()
    link = scrapy.Field()
    summary = scrapy.Field()
    crawled_at = scrapy.Field()
    task_id = scrapy.Field()


class NaverNewsArticleItem(scrapy.Item):
    ''' URL 구조
    1) article/:oid/:aid
    2) /news?oid={}&aid={}
    '''
    oid = scrapy.Field()
    aid = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    published_at = scrapy.Field()
    press_name = scrapy.Field()
    crawled_at = scrapy.Field()
    task_id = scrapy.Field()
