from src.scrapers.habi_scraper import HabiScraper
from src.scrapers.metrocuadrado_scraper import MetrocuadradoScraper
from src.utils.logger import setup_logger

def run_scraping():
    setup_logger()
    
    habi_scraper = HabiScraper()
    metrocuadrado_scraper = MetrocuadradoScraper()
    
    habi_data = habi_scraper.scrape()
    metrocuadrado_data = metrocuadrado_scraper.scrape()
    
    # Save the scraped data to the raw data directory
    habi_scraper.save_data(habi_data, 'data/raw/habi_data.json')
    metrocuadrado_scraper.save_data(metrocuadrado_data, 'data/raw/metrocuadrado_data.json')

if __name__ == "__main__":
    run_scraping()