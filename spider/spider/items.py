# -*- coding: utf-8 -*-

from scrapy import Item, Field

class QnaItem(Item):
    q = Field()
    a = Field()
