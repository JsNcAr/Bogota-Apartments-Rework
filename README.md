# README for Bogota Apartments Simple Project

## Overview
The Bogota Apartments Simple project is designed to scrape apartment listings from various websites, process the scraped data, and store it in a structured format. This project simplifies the original architecture by removing the dependency on MongoDB and focusing on local file storage for raw, processed, and final data.

## Project Structure
```
bogota-apartments-simple
├── src
│   ├── scrapers          # Contains scraper classes for different websites
│   ├── processing        # Contains classes for data processing
│   └── utils             # Contains utility functions and configurations
├── data                  # Directory for storing data
│   ├── raw               # Raw scraped data
│   ├── processed         # Processed data after cleaning and feature extraction
│   └── final             # Final enriched data
├── config                # Configuration files
├── scripts               # Scripts to run different parts of the pipeline
├── tests                 # Unit tests for the project
├── pyproject.toml        # Poetry configuration and dependencies
├── poetry.lock           # Poetry lock file with exact dependency versions
├── .env.example          # Example environment variables
├── .gitignore            # Files and directories to ignore in Git
└── README.md             # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
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

## Development

### Adding Dependencies
To add new dependencies:
```bash
# Add runtime dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

### Testing
To run the tests:
```bash
poetry run pytest tests
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

When contributing:
1. Make sure to use Poetry for dependency management
2. Run tests before submitting: `poetry run pytest`
3. Update dependencies if needed: `poetry update`

## License
This project is licensed under the MIT License. See the LICENSE file for details.