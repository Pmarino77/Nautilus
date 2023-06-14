#Uso de Scrapy
from scrapy.item import Field
from scrapy.item import Item
from scrapy.linkextractors import LinkExtractor
from scrapy.loader import ItemLoader
from scrapy.selector import Selector
from scrapy.spiders import CrawlSpider, Rule


class Hotel(Item):
    nombre=Field()
    precio=Field()
    descripcion=Field()
    servicios=Field()
class TripAdvbisor (CrawlSpider):
    name="Hoteles"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36'
    }
    start_urls=['https://www.tripadvisor.com.mx/Hotels-g1575485-Quintana_Roo_Yucatan_Peninsula-Hotels.html']
    download_delay=2
    rules=(
        Rule(
            LinkExtractor(
                allow=r'/Hotel/Rewiews-'
            ),follow=True, callback='parse_hotel'

        ),
    )
    def parse_item(self, response):
        sel=Selector(response)
        item=ItemLoader(Hotel(),sel)
        item.add_xpath('nombre','//h1[@id="HEADING"]/text()')
        item.add_xpath('precio','//div[@class="price"]/text()')
        item.add_xpath('descripcion','//div[@class="listing_details"]/text()')
        item.add_xpath('servicios','//div[@class="amenity_wrapper"]/div/text()')
        yield item.load_item()



