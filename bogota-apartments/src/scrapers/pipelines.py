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
        self.all_fieldnames = set()  # Track all possible fieldnames

    def process_item(self, item, spider):
        item_dict = dict(item)
        self.items.append(item_dict)
        # Collect all fieldnames from all items
        self.all_fieldnames.update(item_dict.keys())
        return item

    def close_spider(self, spider):
        if not self.items:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to CSV using ALL fieldnames found across ALL items
        csv_filename = self.output_dir / f"{spider.name}_{timestamp}.csv"
        if self.items:
            # Use all collected fieldnames, sorted for consistency
            fieldnames = sorted(self.all_fieldnames)
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write each item, filling missing fields with empty string
                for item in self.items:
                    # Create a row with all fieldnames, defaulting missing ones to ''
                    row = {field: item.get(field, '') for field in fieldnames}
                    writer.writerow(row)

        # Save to JSON (unchanged)
        json_filename = self.output_dir / f"{spider.name}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.items, jsonfile, ensure_ascii=False,
                      indent=2, default=str)

        spider.logger.info(f"Pipeline saved {len(self.items)} items to files")
