from src.scrapers.habi_scraper import HabiScraper
from src.scrapers.metrocuadrado_scraper import MetrocuadradoScraper
from src.processing.data_cleaner import DataCleaner
from src.processing.feature_extractor import FeatureExtractor
from src.processing.data_enricher import DataEnricher
import pandas as pd

def run_full_pipeline():
    # Step 1: Scraping data
    habi_scraper = HabiScraper()
    metrocuadrado_scraper = MetrocuadradoScraper()

    habi_data = habi_scraper.scrape()
    metrocuadrado_data = metrocuadrado_scraper.scrape()

    # Combine scraped data
    combined_data = pd.concat([habi_data, metrocuadrado_data], ignore_index=True)

    # Step 2: Data Cleaning
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(combined_data)

    # Step 3: Feature Extraction
    extractor = FeatureExtractor()
    features_data = extractor.extract_features(cleaned_data)

    # Step 4: Data Enrichment
    enricher = DataEnricher()
    enriched_data = enricher.enrich(features_data)

    # Step 5: Save the final data
    enriched_data.to_csv('data/final/enriched_data.csv', index=False)
    print("Full pipeline executed successfully. Data saved to 'data/final/enriched_data.csv'.")

if __name__ == "__main__":
    run_full_pipeline()