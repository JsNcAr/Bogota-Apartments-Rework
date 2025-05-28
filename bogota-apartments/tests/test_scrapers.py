import unittest
from src.scrapers.habi_scraper import HabiScraper
from src.scrapers.metrocuadrado_scraper import MetrocuadradoScraper

class TestScrapers(unittest.TestCase):

    def setUp(self):
        self.habi_scraper = HabiScraper()
        self.metrocuadrado_scraper = MetrocuadradoScraper()

    def test_habi_scraper_initialization(self):
        self.assertIsInstance(self.habi_scraper, HabiScraper)

    def test_metrocuadrado_scraper_initialization(self):
        self.assertIsInstance(self.metrocuadrado_scraper, MetrocuadradoScraper)

    def test_habi_scraper_scrape(self):
        data = self.habi_scraper.scrape()
        self.assertIsInstance(data, list)  # Assuming scrape returns a list of apartments
        self.assertGreater(len(data), 0)  # Ensure data is not empty

    def test_metrocuadrado_scraper_scrape(self):
        data = self.metrocuadrado_scraper.scrape()
        self.assertIsInstance(data, list)  # Assuming scrape returns a list of apartments
        self.assertGreater(len(data), 0)  # Ensure data is not empty

if __name__ == '__main__':
    unittest.main()