# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class KeywordSearchItem(scrapy.Item):
    keyword = scrapy.Field()
    video_id = scrapy.Field()
    channel_id = scrapy.Field()
    crawled_at = scrapy.Field()


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


class YoutubeCommentItem(scrapy.Item):
    comment = scrapy.Field()
    video_id = scrapy.Field()
    crawled_at = scrapy.Field()


class YoutubeChannelItem(scrapy.Item):
    channel_id = scrapy.Field()
    title = scrapy.Field()
    description = scrapy.Field()
    published_at = scrapy.Field()
    #  statistics
    subscriber_count = scrapy.Field()
    video_count = scrapy.Field()
    crawled_at = scrapy.Field()
