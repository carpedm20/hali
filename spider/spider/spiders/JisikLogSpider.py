from scrapy import Spider
from scrapy.http import Request
from spider.items import QnaItem

# http://www.jisiklog.com/qa/22205209

class JisiklogSpider(Spider):
    name = 'jisik'
    allowed_domains = ['jisiklog.com']
    start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(1, 22205209)]
    #start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(1, 10)]

    def __init__(self):
        pass

    def make_requests_from_url(self, url):
        cookies = {
            'browser_width':581,
            'device_pixel_ratio':2.200000047683716,
            'screen_width':'1280',
        }
        return Request(url, cookies=cookies, dont_filter=True)

    def parse(self, response):
        q = "".join(response.xpath("//h5/span/text()").extract())
        a = "".join(response.xpath("//div[@class='media qna_item']/div/div/text()").extract())[:-31]
        yield QnaItem(q=q, a=a)
