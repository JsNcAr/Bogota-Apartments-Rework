"""
This file contains configuration settings and constants used throughout the project.
"""

import os

# Define constants for file paths
RAW_DATA_PATH = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed')
FINAL_DATA_PATH = os.path.join('data', 'final')

# Define other configuration settings
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'DEBUG')
SCRAPING_TIMEOUT = int(os.getenv('SCRAPING_TIMEOUT', 30))  # seconds
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))  # number of retries for scraping requests

# Add any other configuration settings as needed
