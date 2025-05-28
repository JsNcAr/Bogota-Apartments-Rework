from src.scrapers.habi_scraper import HabiScraper
from src.scrapers.metrocuadrado_scraper import MetrocuadradoScraper
from src.processing.data_cleaner import DataCleaner
from src.processing.feature_extractor import FeatureExtractor
from src.processing.data_enricher import DataEnricher
from src.utils.logger import setup_logger

def main():
    setup_logger()
    
    # Initialize scrapers
    habi_scraper = HabiScraper()
    metrocuadrado_scraper = MetrocuadradoScraper()
    
    # Scrape data
    habi_data = habi_scraper.scrape()
    metrocuadrado_data = metrocuadrado_scraper.scrape()
    
    # Combine data
    combined_data = habi_data + metrocuadrado_data
    
    # Initialize processing classes
    data_cleaner = DataCleaner()
    feature_extractor = FeatureExtractor()
    data_enricher = DataEnricher()
    
    # Clean data
    cleaned_data = data_cleaner.clean(combined_data)
    
    # Extract features
    features = feature_extractor.extract(cleaned_data)
    
    # Enrich data
    enriched_data = data_enricher.enrich(features)
    
    # Save or process the enriched data as needed
    # For example, save to a file or database

if __name__ == "__main__":
    main()