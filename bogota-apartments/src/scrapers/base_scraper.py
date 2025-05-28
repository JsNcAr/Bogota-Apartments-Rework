class BaseScraper:
    """
    Base class for all scrapers. Contains common functionality for scraping.
    """

    def __init__(self, base_url):
        self.base_url = base_url

    def fetch(self, url):
        """
        Fetches the content from the given URL.
        """
        import requests
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse(self, response):
        """
        Parses the response content. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")