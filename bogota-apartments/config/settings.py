"""
This file contains application settings and configurations for the bogota-apartments-simple project.

Settings:
- DATA_DIR: Directory for storing data.
- LOGGING_CONFIG: Path to the logging configuration file.
"""

from pathlib import Path
from datetime import datetime
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
LOGGING_CONFIG = os.path.join(os.path.dirname(__file__), 'logging.conf')



# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

"""
Scrapy settings for bogota-apartments project.
"""

BOT_NAME = 'bogota-apartments'

SPIDER_MODULES = ['src.scrapers']
NEWSPIDER_MODULE = 'src.scrapers'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure pipelines
ITEM_PIPELINES = {
    'src.scrapers.pipelines.FileOutputPipeline': 300,
}

# Configure delays and autothrottling
DOWNLOAD_DELAY = 1
RANDOMIZE_DOWNLOAD_DELAY = True
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 3
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Configure concurrent requests
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 8

# User agent
USER_AGENT = 'bogota-apartments (+http://www.yourdomain.com)'

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = f'logs/scrapy_{datetime.now().strftime("%Y_%d_%a_%I:%M%p")}.log'


# Request fingerprinting
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'

# Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Configure extensions
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Configure feeds (additional output options)
# FEEDS = {
#     'data/raw/%(name)s_%(time)s.jsonl': {  # Use JSONL for better performance
#         'format': 'jsonlines',
#         'encoding': 'utf8',
#         'store_empty': False,  # Don't store items with no data
#         'item_export_kwargs': {
#             'ensure_ascii': False,
#         },
#     },
#     # Remove CSV from FEEDS (CSV is slower for large datasets)
# }

