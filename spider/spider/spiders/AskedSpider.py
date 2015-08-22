from scrapy import Spider
from scrapy.http import Request
from spider.items import QnaItem

# http://pc.asked.kr/ask.php?id=5346194

class LinkedinspiderSpider(Spider):
    name = 'asked'
    allowed_domains = ['asked.kr']
    #start_urls = ["http://pc.asked.kr/ask.php?id=%d" % num for num in xrange(62, 5346195)]
    start_urls = ["http://pc.asked.kr/ask.php?id=%d" % num for num in xrange(12, 80)]

    def __init__(self):
        pass

    def parse(self, response):
        for floor in response.xpath("//div[@class='floor']"):
            q = floor.xpath("div/div/b/text()").extract()[0].strip()
            a = "".join(floor.xpath("div/text()").extract()).strip()
            yield QnaItem(q=q, a=a)
