import requests
import json
import uuid
from scrapy import Spider
from scrapy.http import Request
from spider.items import AroundItem

class AroundSpider(Spider):
    name = 'around'
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'around/1.4.0 (iPhone; iOS 8.3; Scale/2.00)',
    }
    allowed_domains = ['around.conbus.net']
    # https://around.conbus.net/v2/articles/2288749/
    # https://around.conbus.net/v2/articles/2031158/
    #start_urls = ["https://around.conbus.net/v2/articles/%s" % num for num in xrange(2288749, 1, -1)]
    start_urls = ["https://around.conbus.net/v2/articles/%s" % num for num in xrange(664449, 2288749)]
    # 1626701
    # 664449
    # 943812
    # 536258
    # 938213

    def __init__(self):
        self.make_new_user()

    def make_new_user(self):
        new_uid = str(uuid.uuid1()).upper()
        data = {'uid': new_uid}
        self.headers['uid'] = new_uid
        r = requests.post('https://around.conbus.net/v2/users/me/session/start', data=str(data).replace("'",'"'), headers=self.headers)
        new_data ={
            "birth": 1900,
            "gender": "NONE",
            "platform": "IOS",
            "uid": new_uid
        }
        r = requests.post('https://around.conbus.net/v2/users', data=str(new_data).replace("'",'"'), headers=self.headers)

    def make_requests_from_url(self, url):
        return Request(url, headers=self.headers, dont_filter=True)

    def parse(self, response):
        j = json.loads(response.body)
        if j.has_key('error'):
            r = requests.post('https://around.conbus.net/v2/users/me/session/start', data=str(new_data).replace("'",'"'), headers=self.headers)
            yield Request(response.request.url, headers=self.headers)
        else:
            yield AroundItem(j=j)
