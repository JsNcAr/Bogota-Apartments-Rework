import json
from datetime import datetime
from pathlib import Path
from scrapy.exceptions import DropItem


class StreamingFileOutputPipeline:
    """
    Streaming pipeline that writes items immediately to JSONL for best performance.
    """

    def __init__(self):
        self.items_count = 0
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_file = None
        self.max_items = None

    def open_spider(self, spider):
        """Initialize JSONL file when spider starts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the item limit from settings
        self.max_items = spider.settings.getint('CLOSESPIDER_ITEMCOUNT', None)
        if self.max_items:
            spider.logger.info(f"Pipeline will enforce item limit: {self.max_items}")
        
        # Open JSONL file for immediate writing
        jsonl_filename = self.output_dir / f"{spider.name}_{timestamp}.jsonl"
        self.jsonl_file = open(jsonl_filename, 'w', encoding='utf-8')

    def process_item(self, item, spider):
        """Write each item immediately to JSONL."""
        # Check if we've reached the limit
        if self.max_items and self.items_count >= self.max_items:
            spider.logger.info(f"Pipeline reached item limit ({self.max_items}), stopping spider")
            spider.crawler.engine.close_spider(spider, 'Pipeline item limit reached')
            raise DropItem(f"Item limit reached: {self.max_items}")
        
        item_dict = dict(item)
        
        # Write to JSONL immediately (fastest)
        json_line = json.dumps(item_dict, ensure_ascii=False, separators=(',', ':'), default=str)
        if self.jsonl_file is not None:
            self.jsonl_file.write(json_line + '\n')
            self.jsonl_file.flush()
        else:
            raise RuntimeError("JSONL file is not open. Did you forget to call open_spider?")
        
        self.items_count += 1
        
        # Log progress periodically
        if self.items_count % 10 == 0:
            spider.logger.info(f"Processed {self.items_count} items")
            if self.max_items:
                spider.logger.info(f"Progress: {self.items_count}/{self.max_items}")
        
        return item

    def close_spider(self, spider):
        """Close JSONL file."""
        if self.jsonl_file:
            self.jsonl_file.close()

        spider.logger.info(f"Pipeline saved {self.items_count} items to JSONL")
