import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def load_scraped_data():
    """Load the most recent scraped data from JSONL files."""
    raw_data_dir = project_root / 'data' / 'raw'
    
    # Find JSONL files (from your pipeline)
    jsonl_files = list(raw_data_dir.glob('*.jsonl'))
    if not jsonl_files:
        print("No JSONL files found in data/raw/")
        return None
    
    # Load and combine all JSONL files
    all_data = []
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        all_data.append(data)
            print(f"Loaded {len(all_data)} records from {jsonl_file.name}")
        except Exception as e:
            print(f"Error loading {jsonl_file.name}: {e}")
    
    if all_data:
        # Convert to DataFrame
        combined_data = pd.DataFrame(all_data)
        print(f"Combined data: {len(combined_data)} total records")
        return combined_data
    
    return None

def save_to_csv(data, filename):
    """Save DataFrame to CSV with proper field handling."""
    if data is None or data.empty:
        print("No data to save")
        return
    
    # Handle missing fields by filling with empty strings
    data = data.fillna('')
    
    # Save to CSV
    data.to_csv(filename, index=False, encoding='utf-8')
    print(f"CSV saved to: {filename}")

def run_data_processing():
    """Process data that was scraped by Scrapy spiders."""
    
    # Step 1: Load scraped data from JSONL
    print("Step 1: Loading scraped data from JSONL files...")
    combined_data = load_scraped_data()
    
    if combined_data is None:
        print("No data found. Run scraping first: python scripts/run_scraping.py all")
        return
    
    # Step 2: Create directories
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Save raw CSV (before cleaning)
    print("Step 2: Saving raw CSV...")
    raw_csv_file = processed_dir / 'raw_data.csv'
    save_to_csv(combined_data, raw_csv_file)
    
    # Step 4: Basic data cleaning
    print("Step 3: Cleaning data...")
    # Remove duplicates
    initial_count = len(combined_data)
    combined_data = combined_data.drop_duplicates(subset=['codigo'], keep='first')
    print(f"Removed {initial_count - len(combined_data)} duplicates")
    
    # Remove rows with missing essential fields
    before_cleaning = len(combined_data)
    combined_data = combined_data.dropna(subset=['precio_venta', 'area'])
    print(f"Removed {before_cleaning - len(combined_data)} rows with missing essential data")
    
    # Step 5: Save processed CSV
    print("Step 4: Saving processed CSV...")
    processed_csv_file = processed_dir / 'processed_data.csv'
    save_to_csv(combined_data, processed_csv_file)
    
    print("=" * 50)
    print("PROCESSING COMPLETED!")
    print(f"Raw CSV: {raw_csv_file}")
    print(f"Processed CSV: {processed_csv_file}")
    print(f"Final record count: {len(combined_data)}")
    print("=" * 50)

if __name__ == "__main__":
    run_data_processing()