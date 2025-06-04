import json
import csv
from datetime import datetime
from pathlib import Path


class StreamingFileOutputPipeline:
    """
    Streaming pipeline that writes items immediately for better performance.
    """

    def __init__(self):
        self.items_count = 0
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_file = None
        self.csv_file = None
        self.csv_writer = None
        self.all_fieldnames = set()
        self.items_for_csv = []  # Only store items for CSV processing

    def open_spider(self, spider):
        """Initialize files when spider starts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Open JSONL file for immediate writing
        jsonl_filename = self.output_dir / f"{spider.name}_{timestamp}.jsonl"
        self.jsonl_file = open(jsonl_filename, 'w', encoding='utf-8')
        
        # Store CSV filename for later
        self.csv_filename = self.output_dir / f"{spider.name}_{timestamp}.csv"

    def process_item(self, item, spider):
        """Process each item immediately."""
        item_dict = dict(item)
        
        # Write to JSONL immediately (fastest)
        json_line = json.dumps(item_dict, ensure_ascii=False, separators=(',', ':'), default=str)
        if self.jsonl_file is not None:
            self.jsonl_file.write(json_line + '\n')
            self.jsonl_file.flush()
        else:
            raise RuntimeError("JSONL file is not open. Did you forget to call open_spider?")
        
        # Store for CSV processing
        self.items_for_csv.append(item_dict)
        self.all_fieldnames.update(item_dict.keys())
        self.items_count += 1
        
        return item

    def close_spider(self, spider):
        """Close files and write CSV."""
        # Close JSONL file
        if self.jsonl_file:
            self.jsonl_file.close()
        
        # Write CSV with all collected fieldnames
        if self.items_for_csv:
            fieldnames = sorted(self.all_fieldnames)
            
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in self.items_for_csv:
                    row = {field: item.get(field, '') for field in fieldnames}
                    writer.writerow(row)

        spider.logger.info(f"Pipeline saved {self.items_count} items to files")
