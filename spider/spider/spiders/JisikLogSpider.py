from scrapy import Spider
from scrapy.http import Request
from spider.items import QnaItem

# www.jisiklog.com/qa/22205209
# www.jisiklog.com/qa/22092351
# www.jisiklog.com/qa/22022820

class JisiklogSpider(Spider):
    name = 'jisik'
    allowed_domains = ['jisiklog.com']
    #start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(1, 22205263)]
    #start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(1, 10)]
    #start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(22205262, 22205263)]
    #start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(209875, 22205263)]
    #start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(22205263, 297422, -1)]
    start_urls = ["http://www.jisiklog.com/qa/%d" % num for num in xrange(97430, 21965548)]

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
        q = "".join(response.xpath("//h5/span/text()").extract()).strip()
        a = "".join(response.xpath("//div[@class='media qna_item']/div/div/text()").extract())[:-31].strip()
        if a != "":
            yield QnaItem(q=q, a=a)
