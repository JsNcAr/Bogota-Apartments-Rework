# Bogota Apartments Data Pipeline

## Overview
This module handles the data collection and ETL (Extract, Transform, Load) pipeline for apartment listings in Bogota DC. It scrapes data from multiple real estate websites and processes it for analysis.

## Technical Architecture
- **Scrapers**: [`HabiScraper`](src/scrapers/habi_scraper.py), [`MetrocuadradoScraper`](src/scrapers/metrocuadrado_scraper.py)
- **Processing**: [`DataCleaner`](src/processing/data_cleaner.py), [`FeatureExtractor`](src/processing/feature_extractor.py), [`DataEnricher`](src/processing/data_enricher.py)
- **Storage**: Local file system (raw → processed → final)


## Project Structure
```
bogota-apartments
├── src
│   ├── scrapers          # Contains scraper classes for different websites
│   │   ├── base_scraper.py      # Base class for scrapers
│   │   ├── habi_scraper.py
│   │   └── metrocuadrado_scraper.py
│   ├── processing        # Contains classes for data processing
│   │   ├── data_cleaner.py       # Class for cleaning scraped data
│   │   ├── feature_extractor.py  # Class for extracting features from data
│   │   └── data_enricher.py      # Class for enriching data with additional information
│   └── utils             # Contains utility functions and configurations
│   │   ├── config.py          # Configuration settings
│   │   └── logger.py          # Logging setup
├── data                  # Directory for storing data
│   ├── raw               # Raw scraped data
│   ├── processed         # Processed data after cleaning and feature extraction
│   └── final             # Final enriched data
├── config                # Configuration files
│   ├── settings.py       # Main configuration settings
│   └── logging.conf      # Logging configuration
├── scripts               # Scripts to run different parts of the pipeline
│   ├── run_scraping.py   # Script to run the scraping process
│   ├── run_processing.py # Script to run the data processing
│   └── run_full_pipeline.py # Script to run the full pipeline (scraping + processing)
├── tests                 # Unit tests for the project
├── pyproject.toml        # Poetry configuration and dependencies
├── poetry.lock           # Poetry lock file with exact dependency versions
├── .env                  # Environment variables for configuration
└── README.md             # Project documentation
```

## Data Sources
- Habi.co
- Metrocuadrado.com

## Installation
### Prerequisites
- Python 3.12 or higher
- Poetry (Install from https://python-poetry.org/docs/#installation)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bogota-apartments-simple
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. Set up environment variables by copying `.env.example` to `.env` and updating the values as needed.


## Usage
Make sure you're in the Poetry virtual environment before running any commands:

- To run the scraping process:
  ```bash
  poetry run python scripts/run_scraping.py
  ```

- To run the data processing:
  ```bash
  poetry run python scripts/run_processing.py
  ```

- To run the full pipeline (scraping and processing):
  ```bash
  poetry run python scripts/run_full_pipeline.py
  ```

## Data Pipeline Stages
1. **Extraction**: Scrape apartment listings from websites
2. **Cleaning**: Remove duplicates, handle missing values, standardize formats
3. **Feature Extraction**: Extract location, price, area, and property features
4. **Enrichment**: Add neighborhood data, price per sqm, categorizations

## Output Data Schema
[Description of the final data structure and columns]

## Development
### Adding Dependencies
To add new dependencies:
```bash
# Add runtime dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```