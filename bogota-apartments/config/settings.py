"""
This file contains application settings and configurations for the bogota-apartments-simple project.

Settings:
- DATA_DIR: Directory for storing data.
- LOGGING_CONFIG: Path to the logging configuration file.
"""

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
LOGGING_CONFIG = os.path.join(os.path.dirname(__file__), 'logging.conf')