import requests

from scrapy import signals


class EmpathSpiderMiddleware:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        endpoint_url = crawler.settings.get('SPIDER_DONE_URL')
        middleware_instance = cls(endpoint_url)
        crawler.signals.connect(
            middleware_instance.spider_closed, signal=signals.spider_closed)

        return middleware_instance

    def spider_opened(self, spider):
        spider.logger.info(f"[Middleware] Spider opened: {spider.name}")

    def spider_closed(self, spider):
        task_id = getattr(spider, 'task_id', '0')

        # HTTP POST 요청을 보냅니다.
        try:
            response = requests.post(
                url=self.endpoint_url,
                data={'status': 'done', 'task_id': task_id}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            spider.logger.error(
                f"Failed to send request to {self.endpoint_url}: {e}")
