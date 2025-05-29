import json
from datetime import datetime
from fake_useragent import UserAgent
import scrapy
from .base_scraper import BaseSpider, ApartmentItem
from scrapy.loader import ItemLoader


class HabiSpider(BaseSpider):
    """
    Scraper for habi.co apartment listings using Scrapy.
    """
    name = 'habi'
    allowed_domains = ['habi.co', 'apiv2.habi.co']
    base_url = 'https://apiv2.habi.co/listing-global-api/get_properties'

    def start_requests(self):
        """
        Generate initial requests to fetch property listings.
        """
        headers = {
            'X-Api-Key': 'VnXl0bdH2gaVltgd7hJuHPOrMZAlvLa5KGHJsrr6',
            'Referer': 'https://habi.co/',
            'Origin': 'https://habi.co',
            'User-Agent': UserAgent().random
        }

        # Total 817 results, 32 per page
        for offset in range(0, 832, 32):
            url = f'{self.base_url}?offset={offset}&limit=32&filters=%7B%22cities%22:[%22bogota%22]%7D&country=CO'
            yield scrapy.Request(url, headers=headers, callback=self.parse)

    def parse(self, response):
        """
        Parse the initial response and extract property list.
        """
        try:
            result = json.loads(response.body)['messagge']['data']
            self.logger.info(f'Found {len(result)} apartments')

            for item in result:
                property_nid = item.get('property_nid')
                slug = item.get('slug')

                if not property_nid or not slug:
                    continue

                headers = {
                    'Referer': f'https://habi.co/venta-apartamentos/{property_nid}/{slug}',
                    'User-Agent': UserAgent().random
                }

                yield scrapy.Request(
                    url=f'https://habi.co/page-data/venta-apartamentos/{property_nid}/{slug}/page-data.json',
                    headers=headers,
                    callback=self.parse_details,
                    meta={'property_nid': property_nid, 'slug': slug}
                )
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse listings response: {e}")

    def parse_details(self, response):
        """
        Parse detailed property information.
        """
        try:
            details = json.loads(response.body)['result']['pageContext']
            apartment_data = self.normalize_apartment_data(details, response)

            if apartment_data:
                item = self.create_apartment_item(apartment_data)
                # Store item for later file output
                self.scraped_items.append(dict(item))
                yield item

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse property details: {e}")

    def normalize_apartment_data(self, raw_data: dict, response) -> dict:
        """
        Convert Habi raw data to standardized apartment format.
        """
        try:
            property_detail = raw_data.get(
                'propertyDetail', {}).get('property', {})
            detalles = property_detail.get('detalles_propiedad', {})

            # Process images
            images = []
            for image in property_detail.get('images', []):
                if 'url' in image:
                    url = f'https://d3hzflklh28tts.cloudfront.net/{image["url"]}?d=400x400'
                    images.append(url)

            return {
                'codigo': raw_data.get('propertyId'),
                'tipo_propiedad': detalles.get('tipo_inmueble'),
                'tipo_operacion': 'venta',
                'precio_venta': detalles.get('precio_venta'),
                'area': detalles.get('area'),
                'habitaciones': detalles.get('num_habitaciones'),
                'banos': detalles.get('baños'),
                'administracion': detalles.get('last_admin_price'),
                'parqueaderos': detalles.get('garajes'),
                'sector': detalles.get('zona_mediana'),
                'estrato': detalles.get('estrato', ''),
                'antiguedad': detalles.get('anos_antiguedad'),
                'latitud': detalles.get('latitud'),
                'longitud': detalles.get('longitud'),
                'direccion': detalles.get('direccion'),
                'caracteristicas': property_detail.get('caracteristicas_propiedad'),
                'descripcion': property_detail.get('descripcion'),
                'imagenes': images,
                'website': 'habi.co',
                'last_view': datetime.now(),
                'datetime': datetime.now(),
                'url': response.url
            }
        except Exception as e:
            self.logger.error(f"Error normalizing apartment data: {e}")
            return {}
