import json
from datetime import datetime
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from selenium import webdriver
from scrapy.selector import Selector
from scrapy.loader import ItemLoader
import scrapy
from .base_scraper import BaseSpider, ApartmentItem


class MetrocuadradoSpider(BaseSpider):
    """
    Spider to scrape apartment data from metrocuadrado.com
    """
    name = 'metrocuadrado'
    allowed_domains = ['metrocuadrado.com']
    base_url = 'https://www.metrocuadrado.com/rest-search/search'

    def __init__(self, *args, **kwargs):
        """
        Initializes the spider with a headless Chrome browser instance
        """
        super().__init__(*args, **kwargs)

        # TODO: Make it universal for any machine
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--window-size=1920x1080')
        chrome_options.add_argument(f'user-agent={UserAgent().random}')
        chrome_options.add_argument('--disk-cache=true')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            self.logger.error(f"Failed to initialize Chrome driver: {e}")
            raise

    def start_requests(self):
        """
        Generates the initial requests to scrape apartment data
        """
        headers = {
            'X-Api-Key': 'P1MfFHfQMOtL16Zpg36NcntJYCLFm8FqFfudnavl',
            'User-Agent': UserAgent().random
        }

        for operation_type in ['venta', 'arriendo']:
            for offset in range(0, 9950, 50):
                self.logger.info(
                    f'Getting {operation_type} apartments from offset {offset}')
                url = f'{self.base_url}?realEstateTypeList=apartamento&realEstateBusinessList={operation_type}&city=bogot%C3%A1&from={offset}&size=50'

                yield scrapy.Request(
                    url,
                    headers=headers,
                    callback=self.parse,
                    meta={'operation_type': operation_type}
                )

    def parse(self, response):
        """
        Parses the response from the initial requests and generates requests to scrape apartment details
        """
        try:
            result = json.loads(response.body)['results']
            self.logger.info(f'Found {len(result)} apartments')

            for item in result:
                link = item.get('link')
                if link:
                    yield scrapy.Request(
                        url=f'https://metrocuadrado.com{link}',
                        callback=self.parse_details,
                        meta={'operation_type': response.meta.get(
                            'operation_type')}
                    )
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse search results: {e}")

    def parse_details(self, response):
        """
        Parse detailed property information using Selenium
        """
        try:
            self.driver.get(response.url)
            self.logger.info(f'Getting details from {response.url}')

            page_source = self.driver.page_source
            all_script_texts = Selector(text=page_source).xpath(
                '//script/text()').getall()

            if not all_script_texts:
                self.logger.error('No script data found, retrying...')
                self.driver.get(response.url)
                self.driver.implicitly_wait(5)
                page_source = self.driver.page_source
                all_script_texts = Selector(text=page_source).xpath(
                    '//script/text()').getall()

            item_data = self.extract_property_data(
                all_script_texts, response.url)

            if item_data:
                apartment_data = self.normalize_apartment_data(
                    item_data, response)
                if apartment_data:
                    item = self.create_apartment_item(apartment_data)
                    self.scraped_items.append(dict(item))
                    yield item
            else:
                self.logger.error(f"Could not extract data for {response.url}")

        except Exception as e:
            self.logger.error(f"Error parsing details for {response.url}: {e}")

    def extract_property_data(self, script_texts, url):
        """
        Extract property data from JavaScript content
        """
        item_data = None
        script_analyzed_count = 0

        for i, script_content in enumerate(script_texts):
            if 'self.__next_f.push' in script_content and '\\"data\\"' in script_content:
                script_analyzed_count += 1

                try:
                    # Extract JSON data from script
                    start_key = '\\"data\\":'
                    start_key_pos = script_content.find(start_key)
                    if start_key_pos == -1:
                        continue

                    start_brace_pos = script_content.find(
                        '{', start_key_pos + len(start_key))
                    if start_brace_pos == -1:
                        continue

                    end_pattern = '}]]}],'
                    end_pattern_pos = script_content.find(
                        end_pattern, start_brace_pos)
                    if end_pattern_pos == -1:
                        continue

                    target_json_str = script_content[start_brace_pos:end_pattern_pos].strip(
                    )

                    if not target_json_str.endswith('}'):
                        continue

                    # Unescape and parse JSON
                    unescaped_str = target_json_str.replace('\\"', '"')
                    item_data = json.loads(unescaped_str)

                    if 'propertyId' in item_data:
                        self.logger.info(
                            f"Successfully parsed JSON for propertyId: {item_data.get('propertyId')}")
                        break
                    else:
                        item_data = None

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"Script {i}: Failed to decode JSON for {url}: {e}")
                    item_data = None
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error extracting data from script {i} in {url}: {e}")
                    item_data = None

        if not item_data and script_analyzed_count > 0:
            self.logger.error(
                f"Analyzed {script_analyzed_count} scripts but failed to extract valid data for {url}")

        return item_data

    def normalize_apartment_data(self, raw_data, response):
        """
        Convert Metrocuadrado raw data to standardized apartment format
        """
        try:
            # Extract images
            images = []
            imagenes_list = raw_data.get('images', [])
            for img in imagenes_list:
                if isinstance(img, dict) and img.get('image'):
                    images.append(img['image'])

            # Determine operation type
            operation_type = response.meta.get(
                'operation_type', raw_data.get('businessType', ''))

            return {
                'codigo': raw_data.get('propertyId'),
                'tipo_propiedad': self.try_get(raw_data, ['propertyType', 'nombre']),
                'tipo_operacion': operation_type,
                'precio_venta': raw_data.get('salePrice') if operation_type == 'venta' else None,
                'precio_arriendo': raw_data.get('rentPrice') if operation_type == 'arriendo' else None,
                'area': raw_data.get('area'),
                'habitaciones': raw_data.get('rooms'),
                'banos': raw_data.get('bathrooms'),
                'administracion': self.try_get(raw_data, ['detail', 'adminPrice']),
                'parqueaderos': raw_data.get('garages'),
                'sector': self.try_get(raw_data, ['sector', 'nombre']),
                'estrato': raw_data.get('stratum'),
                'antiguedad': raw_data.get('builtTime'),
                'estado': raw_data.get('propertyState'),
                'latitud': self.try_get(raw_data, ['coordinates', 'lat']),
                'longitud': self.try_get(raw_data, ['coordinates', 'lon']),
                'direccion': raw_data.get('address'),
                'featured_interior': self.try_get(raw_data, ['featured', 0, 'items']),
                'featured_exterior': self.try_get(raw_data, ['featured', 1, 'items']),
                'featured_zona_comun': self.try_get(raw_data, ['featured', 2, 'items']),
                'featured_sector': self.try_get(raw_data, ['featured', 3, 'items']),
                'caracteristicas': self.extract_all_features(raw_data),
                'descripcion': raw_data.get('comment'),
                'imagenes': images,
                'compania': raw_data.get('companyName'),
                'website': 'metrocuadrado.com',
                'last_view': datetime.now(),
                'datetime': datetime.now(),
                'url': response.url
            }
        except Exception as e:
            self.logger.error(f"Error normalizing apartment data: {e}")
            return {}

    def extract_all_features(self, data):
        """
        Extract and combine all feature categories
        """
        all_features = []
        featured_sections = data.get('featured', [])

        for section in featured_sections:
            if isinstance(section, dict) and 'items' in section:
                items = section.get('items', [])
                if isinstance(items, list):
                    all_features.extend(items)

        return all_features

    def closed(self, reason):
        """
        Clean up when spider closes
        """
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
                self.logger.info("Chrome driver closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing Chrome driver: {e}")

        # Call parent method to save items
        super().closed(reason)


# Alternative fields for Metrocuadrado-specific data
class MetrocuadradoItem(ApartmentItem):
    """Extended item for Metrocuadrado-specific fields"""
    precio_arriendo = scrapy.Field()
    featured_interior = scrapy.Field()
    featured_exterior = scrapy.Field()
    featured_zona_comun = scrapy.Field()
    featured_sector = scrapy.Field()
    compañia = scrapy.Field()
    estado = scrapy.Field()
