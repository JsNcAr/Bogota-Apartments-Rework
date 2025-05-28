class BaseScraper:
    def __init__(self, base_url):
        self.base_url = base_url

    def fetch(self, url):
        # Logic to fetch data from the given URL
        pass

    def parse(self, response):
        # Logic to parse the response
        pass


class HabiScraper(BaseScraper):
    def __init__(self):
        super().__init__('https://www.habi.com')

    def scrape_apartments(self):
        # Logic to scrape apartment data from Habi
        pass

    def parse_apartment_details(self, response):
        # Logic to parse apartment details from the response
        pass