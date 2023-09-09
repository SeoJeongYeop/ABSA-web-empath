# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
from itemadapter import ItemAdapter
import cx_Oracle
# import pymysql
from scrapy import signals
from pydispatch import dispatcher
import sys
import logging

from .items import *
from .settings import DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME


class EmpathPipeline:

    def __init__(self) -> None:

        self.crawlDB = None
        self.cursor = None
        try:
            dispatcher.connect(self.spider_opened, signals.spider_opened)
            dispatcher.connect(self.spider_closed, signals.spider_closed)
        except:
            print('ERROR: dispatcher connection failed')
            sys.exit(1)

        self.wait = {}
        self.lost = {}

    def spider_opened(self, spider):

        logging.info("spider_opened")
        try:
            '''
            MySQL 사용 코드
            '''
            # self.crawlDB = pymysql.connect(
            #     user=SQL_USER,
            #     passwd=SQL_PW,
            #     host=SQL_HOST,
            #     port=SQL_PORT,
            #     db=SQL_DB
            # )
            '''
            Oracle Database 사용 코드
            '''
            self.crawlDB = cx_Oracle.connect(
                DATABASE_USER,
                DATABASE_PASSWORD,
                DATABASE_NAME
            )
            self.cursor = self.crawlDB.cursor()
            # self.cursor.execute(
            #     """UPDATE sys.props$ SET value$ = '[UTF8]' WHERE name = 'NLS_CHARACTERSET'""")  # 미입력시 날짜 포맷팅 오류
        except:
            print('ERROR: DB connection failed')
            sys.exit(1)

    def spider_closed(self, spider):

        logging.info("spider_closed")
        try:
            self.cursor.close()
            self.crawlDB.close()
        except:
            print('ERROR: DB close failed')
            sys.exit(1)

    def process_item(self, item, spider):

        table_name = item_to_table_mapping.get(type(item))
        columns = item.keys()
        insert_sql = f'INSERT INTO {table_name}({", ".join(columns)})'
        variables = ",".join([f":{i}" for i in range(len(columns))])
        # variables = ",".join(["%s"]*len(item)) # MySQL
        insert_sql += f'VALUES({variables})'
        values = [item[col] for col in columns]

        try:
            self.cursor.execute(insert_sql, values)
            self.crawlDB.commit()
        except Exception as e:
            logging.exception(f"Pipeline Cursor Exception: {e}")
            logging.info(f'insert_sql: {insert_sql}')
            logging.info(f'values: {values}')

        return item


item_to_table_mapping = {
    YoutubeVideoItem: "crawler_youtubevideo",
    YoutubeSearchResultItem: "crawler_youtubesearchresult",
    YoutubeCommentItem: "crawler_youtubecomment",
    NaverNewsItem: "crawler_naversearchresult",
    NaverNewsArticleItem: "crawler_navernewsarticle",
}
