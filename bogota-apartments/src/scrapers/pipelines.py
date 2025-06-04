import json
import csv
from datetime import datetime
from pathlib import Path


class FileOutputPipeline:
    """
    Pipeline to save items to CSV and JSON files.
    """

    def __init__(self):
        self.items = []
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_item(self, item, spider):
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        if not self.items:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to CSV
        csv_filename = self.output_dir / f"{spider.name}_{timestamp}.csv"
        if self.items:
            fieldnames = self.items[0].keys()
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.items)

        # Save to JSON
        json_filename = self.output_dir / f"{spider.name}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.items, jsonfile, ensure_ascii=False,
                      indent=2, default=str)

        spider.logger.info(f"Pipeline saved {len(self.items)} items to files")
