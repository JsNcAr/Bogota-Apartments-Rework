import scrapy
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fake_useragent import UserAgent
from scrapy.loader import ItemLoader


class ApartmentItem(scrapy.Item):
    """Simplified apartment item without external dependencies."""
    codigo = scrapy.Field()
    tipo_propiedad = scrapy.Field()
    tipo_operacion = scrapy.Field()
    precio_venta = scrapy.Field()
    precio_arriendo = scrapy.Field()  # Metrocuadrado Only
    area = scrapy.Field()
    habitaciones = scrapy.Field()
    banos = scrapy.Field()
    administracion = scrapy.Field()
    parqueaderos = scrapy.Field()
    sector = scrapy.Field()
    estrato = scrapy.Field()
    antiguedad = scrapy.Field()
    estado = scrapy.Field()  # Metrocuadrado Only
    latitud = scrapy.Field()
    longitud = scrapy.Field()
    direccion = scrapy.Field()
    featured_interior = scrapy.Field()  # Metrocuadrado Only
    featured_exterior = scrapy.Field()  # Metrocuadrado Only
    featured_zona_comun = scrapy.Field()  # Metrocuadrado Only
    featured_sector = scrapy.Field()  # Metrocuadrado Only
    caracteristicas = scrapy.Field()
    descripcion = scrapy.Field()
    compania = scrapy.Field()  # 
    imagenes = scrapy.Field()
    website = scrapy.Field()
    last_view = scrapy.Field()
    datetime = scrapy.Field()
    url = scrapy.Field()


class BaseSpider(scrapy.Spider):
    """
    Base spider class that provides common functionality for all scrapers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scraped_items = []

    def get_random_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid detection."""
        return {
            'User-Agent': UserAgent().random,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def try_get(self, dictionary: Dict, keys: List[Any]) -> Any:
        """
        Safely extract nested values from dictionary/list structures.
        """
        try:
            value = dictionary
            for key in keys:
                if isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
                    value = value[key]
                elif isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except (KeyError, TypeError, IndexError):
            return None

    def create_apartment_item(self, data: Dict) -> ApartmentItem:
        """Create and populate an apartment item."""
        loader = ItemLoader(item=ApartmentItem())

        for field, value in data.items():
            if value is not None:
                loader.add_value(field, value)

        return loader.load_item()

    def save_items_to_file(self):
        """Save collected items to CSV and JSON files."""
        if not self.scraped_items:
            self.logger.warning("No items to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Collect all unique fieldnames from all items
        all_fieldnames = set()
        for item in self.scraped_items:
            all_fieldnames.update(item.keys())
        
        fieldnames = sorted(all_fieldnames)  # Sort for consistency

        # Save to CSV
        csv_filename = self.output_dir / f"{self.name}_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Fill missing fields with empty values
            for item in self.scraped_items:
                row = {field: item.get(field, '') for field in fieldnames}
                writer.writerow(row)

        # Save to JSON (optimized)
        json_filename = self.output_dir / f"{self.name}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.scraped_items, jsonfile,
                      ensure_ascii=False, separators=(',', ':'), default=str)

        self.logger.info(
            f"Saved {len(self.scraped_items)} items to {csv_filename} and {json_filename}")

    def closed(self, reason):
        """Called when spider closes - save all collected items."""
        self.save_items_to_file()
        self.logger.info(f"Spider closed: {reason}")
