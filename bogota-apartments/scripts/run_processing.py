import pandas as pd
import json
import os
import sys
from pathlib import Path
import numpy as np

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
    total_loaded = 0
    for jsonl_file in jsonl_files:
        try:
            file_count = 0
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        all_data.append(data)
                        file_count += 1
            print(f"Loaded {file_count} records from {jsonl_file.name}")
            total_loaded += file_count
        except Exception as e:
            print(f"Error loading {jsonl_file.name}: {e}")
    
    if all_data:
        # Convert to DataFrame
        combined_data = pd.DataFrame(all_data)
        print(f"Combined data: {total_loaded} total records")
        return combined_data
    
    return None

def save_to_csv(data, filename):
    """Save DataFrame to CSV with proper field handling."""
    if data is None or data.empty:
        print("No data to save")
        return
    
    # Make a copy to avoid modifying original data
    data_copy = data.copy()
    
    # Convert categorical columns to strings before filling
    categorical_columns = data_copy.select_dtypes(include=['category']).columns
    if len(categorical_columns) > 0:
        print(f"Converting {len(categorical_columns)} categorical columns to strings...")
        for col in categorical_columns:
            data_copy[col] = data_copy[col].astype(str)
    
    # Now safely fill NaN values
    data_copy = data_copy.fillna('')
    
    # Save to CSV
    data_copy.to_csv(filename, index=False, encoding='utf-8')
    print(f"CSV saved to: {filename}")

def process_images(df):
    """Extract and process images similar to original script."""
    print("Processing images...")
    
    # Explode images column if it exists
    if 'imagenes' in df.columns:
        images_explode = df.explode('imagenes')
        images_explode = images_explode.dropna(subset=['imagenes'])
        
        if not images_explode.empty:
            # Create images DataFrame
            images_df = images_explode[['codigo', 'imagenes']].rename(columns={'imagenes': 'url_imagen'})
            
            # Save images CSV
            processed_dir = project_root / 'data' / 'processed'
            images_file = processed_dir / 'images.csv'
            save_to_csv(images_df, images_file)
            print(f"Images extracted: {len(images_df)} records")
        else:
            print("No images found to extract")
        
        # Remove images from main DataFrame to reduce memory usage
        df = df.drop(columns=['imagenes'], axis=1)
    else:
        print("No 'imagenes' column found")
    
    return df

def advanced_feature_extraction(df):
    """Use the unified FeatureExtractor for comprehensive feature engineering."""
    try:
        from processing.feature_extractor import FeatureExtractor
        
        print("Using advanced FeatureExtractor...")
        
        # Clean numeric columns first - THIS IS CRUCIAL
        print("Cleaning numeric columns...")
        df = clean_numeric_columns(df)
        
        extractor = FeatureExtractor()
        
        # This includes BOTH original boolean features AND advanced features
        enhanced_df = extractor.extract_features(df)
        
        print(f"Feature extraction completed")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   Enhanced columns: {len(enhanced_df.columns)}")
        print(f"   New features added: {len(enhanced_df.columns) - len(df.columns)}")
        
        return enhanced_df
        
    except ImportError as e:
        print(f"❌ FeatureExtractor not available: {e}")
        print("Continuing with basic processing...")
        return df
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error trace
        print("Continuing with basic processing...")
        return df

def clean_numeric_columns(df):
    """Clean numeric columns that might contain lists or strings."""
    numeric_columns = ['precio_venta', 'precio_arriendo', 'area', 'habitaciones', 'banos', 
                      'parqueaderos', 'administracion', 'antiguedad', 'estrato']
    
    for col in numeric_columns:
        if col in df.columns:
            
            def clean_numeric_value(value):
                # Handle None/NaN
                if value is None:
                    return np.nan
                
                # Handle lists
                if isinstance(value, list):
                    if len(value) == 0:
                        return np.nan
                    # Take first number if it's a list
                    try:
                        first_val = value[0]
                        return float(first_val) if first_val is not None else np.nan
                    except (ValueError, TypeError, IndexError):
                        return np.nan
                
                # Handle pandas NaN (only for non-lists)
                if not isinstance(value, list) and pd.isna(value):
                    return np.nan
                
                # Handle strings and other types
                try:
                    if isinstance(value, list):
                        return np.nan
                    return float(value)
                except (ValueError, TypeError):
                    return np.nan
            
            original_type = df[col].dtype
            df[col] = df[col].apply(clean_numeric_value)
            print(f"  {col}: {original_type} -> {df[col].dtype}")
            print(f"  Non-null values: {df[col].notna().sum()}/{len(df)}")
    
    return df

def run_data_processing():
    """Enhanced processing with comprehensive feature extraction."""
    
    # Step 1: Load scraped data from JSONL
    print("=" * 60)
    print("🏠 BOGOTÁ APARTMENTS DATA PROCESSING")
    print("=" * 60)
    print("Step 1: Loading scraped data from JSONL files...")
    combined_data = load_scraped_data()
    
    if combined_data is None:
        print("❌ No data found. Run scraping first:")
        print("   python scripts/run_scraping.py all")
        return
    
    # Step 2: Create directories
    processed_dir = project_root / 'data' / 'processed'
    interim_dir = project_root / 'data' / 'interim'
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Save raw CSV (before any processing)
    print("\nStep 2: Saving raw CSV...")
    raw_csv_file = processed_dir / 'raw_data.csv'
    save_to_csv(combined_data, raw_csv_file)
    
    # Step 4: Process images (extract to separate file)
    print("\nStep 3: Processing images...")
    combined_data = process_images(combined_data.copy())
    
    # Step 5: Basic data cleaning
    print("\nStep 4: Basic data cleaning...")
    initial_count = len(combined_data)
    
    # Remove duplicates
    combined_data = combined_data.drop_duplicates(subset=['codigo'], keep='first')
    duplicates_removed = initial_count - len(combined_data)
    print(f"Removed {duplicates_removed} duplicates")
    
    # Remove rows with missing essential fields (flexible approach)
    before_cleaning = len(combined_data)
    
    # Check which price columns exist
    has_precio_venta = 'precio_venta' in combined_data.columns
    has_precio_arriendo = 'precio_arriendo' in combined_data.columns
    has_area = 'area' in combined_data.columns
    
    print(f"Available columns: precio_venta={has_precio_venta}, precio_arriendo={has_precio_arriendo}, area={has_area}")
    
    # Build condition based on available columns
    price_conditions = []
    
    if has_precio_venta:
        price_conditions.append(combined_data['precio_venta'].notna())
    
    if has_precio_arriendo:
        price_conditions.append(combined_data['precio_arriendo'].notna())
    
    # Require at least one price column and area
    if price_conditions and has_area:
        # Combine all price conditions with OR
        price_condition = price_conditions[0]
        for condition in price_conditions[1:]:
            price_condition = price_condition | condition
        
        essential_condition = price_condition & combined_data['area'].notna()
        combined_data = combined_data[essential_condition]
        
    elif has_area:
        # If no price columns, just require area
        print("⚠️  Warning: No price columns found, only filtering by area")
        combined_data = combined_data[combined_data['area'].notna()]
        
    else:
        print("⚠️  Warning: No essential columns found, skipping data filtering")
    
    missing_removed = before_cleaning - len(combined_data)
    print(f"Removed {missing_removed} rows with missing essential data")
    
    # Step 6: Clean numeric columns
    print("\nStep 5: Cleaning numeric columns...")
    combined_data = clean_numeric_columns(combined_data)
    
    # Step 7: ADVANCED feature extraction (includes original + new features)
    print("\nStep 6: Advanced feature extraction...")
    print("This may take a moment...")
    enhanced_data = advanced_feature_extraction(combined_data)
    
    # Step 8: Save final processed data
    print("\nStep 7: Saving enhanced processed data...")
    apartments_file = interim_dir / 'apartments_enhanced.csv'
    save_to_csv(enhanced_data, apartments_file)
    
    # Also save a simple version without advanced features for compatibility
    simple_file = interim_dir / 'apartments_simple.csv'
    save_to_csv(combined_data, simple_file)
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📁 Raw data: {raw_csv_file}")
    print(f"📁 Images: {processed_dir / 'images.csv'}")
    print(f"📁 Simple apartments: {simple_file}")
    print(f"📁 Enhanced apartments: {apartments_file}")
    print(f"📊 Final record count: {len(enhanced_data)}")
    print("=" * 60)
    
    # Show sample of extracted features
    if len(enhanced_data.columns) > len(combined_data.columns):
        new_features = [col for col in enhanced_data.columns if col not in combined_data.columns]
        print("\n🔍 Sample of new features extracted:")
        
        feature_examples = [
            'jacuzzi', 'piscina', 'gimnasio', 'precio_por_m2', 'categoria_precio',
            'score_valor', 'score_lujo', 'tipo_simplificado'
        ]
        
        available_examples = [f for f in feature_examples if f in new_features]
        
        for i, feature in enumerate(available_examples[:8]):  # Show first 8
            if len(enhanced_data) > 0:
                sample_value = enhanced_data[feature].iloc[0]
                print(f"   • {feature}: {sample_value}")
        
        if len(new_features) > 8:
            print(f"   ... and {len(new_features) - 8} more features")
        
        print(f"\n📈 Total features extracted: {len(new_features)}")

if __name__ == "__main__":
    run_data_processing()