class MetrocuadradoScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.start_url = 'https://www.metrocuadrado.com/'
    
    def parse(self, response):
        # Implement the parsing logic for the Metrocuadrado website
        apartments = response.css('div.apartment-listing')
        for apartment in apartments:
            yield {
                'title': apartment.css('h2.title::text').get(),
                'price': apartment.css('span.price::text').get(),
                'location': apartment.css('span.location::text').get(),
                'details': apartment.css('div.details::text').get(),
            }
    
    def run(self):
        response = self.fetch(self.start_url)
        if response:
            for apartment in self.parse(response):
                self.save(apartment)