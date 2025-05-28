from src.scrapers.habi_scraper import HabiScraper
from src.scrapers.metrocuadrado_scraper import MetrocuadradoScraper
from src.processing.data_cleaner import DataCleaner
from src.processing.feature_extractor import FeatureExtractor
from src.processing.data_enricher import DataEnricher
import pandas as pd
import os

def run_data_processing():
    # Step 1: Scrape data
    habi_scraper = HabiScraper()
    metrocuadrado_scraper = MetrocuadradoScraper()

    habi_data = habi_scraper.scrape()
    metrocuadrado_data = metrocuadrado_scraper.scrape()

    # Combine data
    combined_data = pd.concat([habi_data, metrocuadrado_data], ignore_index=True)

    # Step 2: Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(combined_data)

    # Step 3: Extract features
    feature_extractor = FeatureExtractor()
    features_data = feature_extractor.extract(cleaned_data)

    # Step 4: Enrich data
    enricher = DataEnricher()
    enriched_data = enricher.enrich(features_data)

    # Step 5: Save processed data
    processed_data_path = os.path.join('data', 'processed', 'processed_data.csv')
    enriched_data.to_csv(processed_data_path, index=False)
    print(f'Processed data saved to {processed_data_path}')

if __name__ == "__main__":
    run_data_processing()