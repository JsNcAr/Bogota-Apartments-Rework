#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))


def run_spider(spider_name):
    """Run a specific spider."""
    # Import spiders after adding to path
    from scrapers.habi_scraper import HabiSpider
    from scrapers.metrocuadrado_scraper import MetrocuadradoSpider

    spiders = {
        'habi': HabiSpider,
        'metrocuadrado': MetrocuadradoSpider,
    }

    if spider_name not in spiders:
        print(f"Available spiders: {list(spiders.keys())}")
        return

    # Load settings
    settings = get_project_settings()
    settings.setmodule('config.settings')

    # Create directories
    (project_root / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
    (project_root / 'logs').mkdir(parents=True, exist_ok=True)

    # Create and configure process
    process = CrawlerProcess(settings)
    process.crawl(spiders[spider_name])
    process.start()


def run_all_spiders():
    """Run all spiders sequentially."""
    spiders = ['habi', 'metrocuadrado']
    for spider in spiders:
        print(f"Running {spider} spider...")
        run_spider(spider)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        spider_name = sys.argv[1]
        if spider_name == 'all':
            run_all_spiders()
        else:
            run_spider(spider_name)
    else:
        print("Usage: python scripts/run_scraping.py [spider_name|all]")
        print("Available spiders: habi, metrocuadrado")
        print("Use 'all' to run all spiders")
