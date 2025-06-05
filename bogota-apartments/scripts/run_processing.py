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

def clean_text_columns(df):
    """Clean text columns that might contain lists or mixed types."""
    text_columns = ['sector', 'localidad', 'barrio', 'descripcion', 'tipo_propiedad']
    
    for col in text_columns:
        if col in df.columns:
            print(f"  Cleaning {col} column...")
            
            def clean_text_value(value):
                # Handle None/NaN
                if value is None or pd.isna(value):
                    return None
                
                # Handle lists - take first non-empty string
                if isinstance(value, list):
                    if len(value) == 0:
                        return None
                    for item in value:
                        if item is not None and str(item).strip():
                            return str(item).strip()
                    return None
                
                # Handle strings
                if isinstance(value, str):
                    return value.strip() if value.strip() else None
                
                # Convert other types to string
                return str(value).strip() if str(value).strip() else None
            
            original_type = df[col].dtype
            df[col] = df[col].apply(clean_text_value)
            print(f"    {col}: {original_type} -> {df[col].dtype}")
            non_null_count = df[col].notna().sum()
            print(f"    Non-null values: {non_null_count}/{len(df)}")
    
    return df

def create_enrichment_geodata_files(external_dir):
    """
    Create/prepare geodata files for the enricher from external data.
    This function maps your external data structure to what GeoDataEnricher expects.
    """
    try:
        import shutil
        
        print("   📂 Preparing geographic data for enrichment...")
        
        # Create geo directory if it doesn't exist
        geo_dir = external_dir.parent / 'geo'
        geo_dir.mkdir(exist_ok=True)
        
        # File mappings: external_path -> geo_path
        file_mappings = {
            # Localidades: copy from barrios (has localidad info) or use direct shapefile
            external_dir / 'localidades_bogota' / 'loca.shp': geo_dir / 'localidades.shp',
            
            # Barrios: direct copy
            external_dir / 'barrios_bogota' / 'barrios.geojson': geo_dir / 'barrios.geojson',
            
            # TransMilenio: copy from estaciones_troncales_tm
            external_dir / 'estaciones_troncales_tm' / 'estaciones-de-transmilenio.geojson': geo_dir / 'transmilenio.geojson',
        }
        
        # Copy files if they exist
        files_copied = 0
        for source, dest in file_mappings.items():
            if source.exists():
                try:
                    if source.suffix == '.shp':
                        # For shapefiles, copy all associated files
                        base_name = source.stem
                        source_dir = source.parent
                        dest_dir = dest.parent
                        
                        shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.shp.xml']
                        for ext in shapefile_extensions:
                            src_file = source_dir / f"{base_name}{ext}"
                            if src_file.exists():
                                dest_file = dest_dir / f"{dest.stem}{ext}"
                                shutil.copy2(src_file, dest_file)
                    else:
                        # For single files like GeoJSON
                        shutil.copy2(source, dest)
                    
                    print(f"      ✅ Copied {source.name} -> {dest.name}")
                    files_copied += 1
                except Exception as e:
                    print(f"      ⚠️  Could not copy {source.name}: {e}")
            else:
                print(f"      ⚠️  Source file not found: {source}")
        
        # Create POI file from available data (UPDATED TO USE NEW MODULE)
        poi_file = geo_dir / 'poi.geojson'
        if not poi_file.exists():
            print(f"      📍 Creating POI file from CSV data...")
            try:
                from data_creation.create_poi import create_poi_file
                poi_created = create_poi_file(external_dir, poi_file)
                if poi_created:
                    files_copied += 1
            except ImportError:
                print(f"         ⚠️  POI creator module not available")
            except Exception as e:
                print(f"         ❌ Error creating POI file: {e}")
        else:
            print(f"      📍 POI file already exists: {poi_file}")
            files_copied += 1
        
        print(f"      📊 Geographic files prepared: {files_copied} files ready")
        return geo_dir if files_copied > 0 else None
        
    except Exception as e:
        print(f"      ❌ Error preparing geodata: {e}")
        return None

def property_feature_extraction(df):
    """Extract property-level features using FeatureExtractor."""
    try:
        from processing.feature_extractor import FeatureExtractor
        
        print("🏠 Extracting property features...")
        
        # Clean numeric columns first
        df = clean_numeric_columns(df)
        
        # Clean text columns (NEW)
        df = clean_text_columns(df)
        
        extractor = FeatureExtractor()
        
        # Extract property-level features (amenities, ratios, property-specific scores)
        enhanced_df = extractor.extract_features(df)
        
        print(f"Property feature extraction completed")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   Enhanced columns: {len(enhanced_df.columns)}")
        print(f"   New features added: {len(enhanced_df.columns) - len(df.columns)}")
        
        return enhanced_df
        
    except ImportError as e:
        print(f"   ⚠️  FeatureExtractor not available: {e}")
        print("   Continuing with basic processing...")
        return df
    except Exception as e:
        print(f"   ❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with basic processing...")
        return df

def geographic_correction_and_enrichment(df):
    """Two-stage geographic processing: correction then enrichment."""
    try:
        print("🗺️  Geographic processing...")
        
        # Check if we have coordinate data
        has_coords = 'latitud' in df.columns and 'longitud' in df.columns
        if not has_coords:
            print("   ⚠️  No coordinate columns found. Skipping geographic processing.")
            return df
        
        # Check for external data directory
        external_dir = project_root / 'data' / 'external'
        if not external_dir.exists():
            print("   ⚠️  External data directory not found. Skipping geographic processing.")
            print("   To enable: Create data/external/ and add Bogotá datasets")
            return df
        
        # Stage 1: Geographic Correction (Data validation & business rules)
        print("🔧 Stage 1: Geographic correction and validation...")
        try:
            from processing.geo_data_corrector import correct_apartment_locations
            
            corrected_df = correct_apartment_locations(df, external_dir)
                
        except ImportError:
            print("      ⚠️  GeoDataCorrector not available (missing geopandas/shapely)")
            print("      To enable: pip install geopandas shapely")
            corrected_df = df
        except Exception as e:
            print(f"      ❌ Error during geographic correction: {e}")
            print("      Continuing without correction...")
            corrected_df = df
        
        # Stage 2: Advanced Geographic Enrichment (Market analysis, POI, accessibility)
        print("🚀 Stage 2: Advanced geographic enrichment...")
        try:
            from processing.geo_data_enricher import enrich_apartment_locations
            
            # Prepare geodata for enricher (creates/copies files to data/geo/)
            geo_data_dir = create_enrichment_geodata_files(external_dir)
            
            if geo_data_dir:
                enriched_df = enrich_apartment_locations(corrected_df, str(geo_data_dir))
            else:
                print("      ⚠️  Could not prepare geodata. Skipping enrichment.")
                enriched_df = corrected_df
                
        except ImportError:
            print("      ⚠️  GeoDataEnricher not available (missing geopandas)")
            enriched_df = corrected_df
        except Exception as e:
            print(f"      ❌ Error during geographic enrichment: {e}")
            print("      Continuing without enrichment...")
            enriched_df = corrected_df
        
        print("   ✅ Geographic processing completed")
        return enriched_df
        
    except Exception as e:
        print(f"   ❌ Error during geographic processing: {e}")
        import traceback
        traceback.print_exc()
        return df

def run_data_processing():
    """Enhanced three-stage processing: features → correction → enrichment."""
    
    # Step 1: Load scraped data from JSONL
    print("=" * 70)
    print("🏠 BOGOTÁ APARTMENTS DATA PROCESSING (ENHANCED)")
    print("=" * 70)
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
    
    # Step 6: Property feature extraction (NEW ARCHITECTURE)
    print("\nStep 5: Property feature extraction...")
    enhanced_data = property_feature_extraction(combined_data)
    
    # Step 7: Geographic correction and enrichment (NEW ARCHITECTURE)
    print("\nStep 6: Geographic processing...")
    enhanced_data = geographic_correction_and_enrichment(enhanced_data)
    
    # Step 8: Save final processed data
    print("\nStep 7: Saving final processed data...")
    
    # Save enhanced version with all features
    enhanced_file = interim_dir / 'apartments_enhanced.csv'
    save_to_csv(enhanced_data, enhanced_file)
    
    # Save simple version without advanced features for compatibility
    simple_file = interim_dir / 'apartments_simple.csv'
    save_to_csv(combined_data, simple_file)
    
    # Save feature-only version (no geographic processing)
    if len(enhanced_data.columns) > len(combined_data.columns):
        # Check if we have property features but not geographic features
        property_features = [col for col in enhanced_data.columns if col not in combined_data.columns]
        
        # Identify likely geographic features (added in Step 6)
        geographic_features = [col for col in property_features if any(keyword in col.lower() for keyword in [
            'zona_', 'distancia_', 'score_', 'percentil_', 'ratio_promedio', 'facilidad_', 'coords_modified'
        ])]
        
        if geographic_features:
            # Create version with only property features
            property_only_cols = [col for col in enhanced_data.columns if col not in geographic_features]
            property_only_data = enhanced_data[property_only_cols]
            
            property_file = interim_dir / 'apartments_with_features.csv'
            save_to_csv(property_only_data, property_file)
            
            print(f"📁 Property features only: {property_file}")
    
    print("\n" + "=" * 70)
    print("🎉 ENHANCED PROCESSING COMPLETED!")
    print("=" * 70)
    print(f"📁 Raw data: {raw_csv_file}")
    print(f"📁 Images: {processed_dir / 'images.csv'}")
    print(f"📁 Simple apartments: {simple_file}")
    print(f"📁 Enhanced apartments (full): {enhanced_file}")
    print(f"📊 Final record count: {len(enhanced_data)}")
    print("=" * 70)
    
    # Show comprehensive feature summary
    if len(enhanced_data.columns) > len(combined_data.columns):
        new_features = [col for col in enhanced_data.columns if col not in combined_data.columns]
        
        # Categorize features
        property_features = []
        geographic_features = []
        market_features = []
        
        for feature in new_features:
            if any(keyword in feature.lower() for keyword in [
                'jacuzzi', 'piscina', 'gimnasio', 'ascensor', 'terraza', 'amoblado', 'vigilancia',
                'chimenea', 'mascotas', 'salon_comunal', 'conjunto_cerrado', 'closets', 'piso',
                'categoria_', 'tipo_', 'tiene_', 'numero_', 'indica_', 'score_lujo', 'score_familiar',
                'score_completitud', 'precio_por_m2', 'area_por_', 'eficiencia_', 'ratio_habitaciones'
            ]):
                property_features.append(feature)
            elif any(keyword in feature.lower() for keyword in [
                'distancia_',  'coords_modified'
            ]):
                geographic_features.append(feature)
            elif any(keyword in feature.lower() for keyword in [
                'percentil_', 'ratio_promedio', 'diferencia_promedio', 'precio_promedio',
                'score_deseabilidad', 'score_inversion', 'score_conveniencia', 'score_movilidad',
                'facilidad_', 'categoria_precio_mercado', 'precio_m2_categoria'
            ]):
                market_features.append(feature)
            else:
                property_features.append(feature)  # Default to property features
        
        print("\n🔍 FEATURE EXTRACTION SUMMARY:")
        print(f"   🏠 Property features: {len(property_features)}")
        if property_features[:5]:  # Show first 5
            for feature in property_features[:5]:
                if len(enhanced_data) > 0:
                    sample_value = enhanced_data[feature].iloc[0]
                    print(f"      • {feature}: {sample_value}")
            if len(property_features) > 5:
                print(f"      ... and {len(property_features) - 5} more")
        
        if geographic_features:
            print(f"   🗺️  Geographic features: {len(geographic_features)}")
            for feature in geographic_features[:3]:  # Show first 3
                if len(enhanced_data) > 0:
                    sample_value = enhanced_data[feature].iloc[0]
                    print(f"      • {feature}: {sample_value}")
            if len(geographic_features) > 3:
                print(f"      ... and {len(geographic_features) - 3} more")
        
        if market_features:
            print(f"   📊 Market analysis features: {len(market_features)}")
            for feature in market_features[:3]:  # Show first 3
                if len(enhanced_data) > 0:
                    sample_value = enhanced_data[feature].iloc[0]
                    print(f"      • {feature}: {sample_value}")
            if len(market_features) > 3:
                print(f"      ... and {len(market_features) - 3} more")
        
        print(f"\n📈 TOTAL FEATURES EXTRACTED: {len(new_features)}")
        
        # Show data quality summary
        print(f"\n📋 DATA QUALITY SUMMARY:")
        print(f"   Records processed: {len(combined_data)} → {len(enhanced_data)}")
        
        # Count completeness
        essential_cols = ['precio_venta', 'precio_arriendo', 'area', 'habitaciones', 'banos']
        available_essential = [col for col in essential_cols if col in enhanced_data.columns]
        if available_essential:
            completeness = enhanced_data[available_essential].notna().any(axis=1).sum()
            print(f"   Records with essential data: {completeness}/{len(enhanced_data)} ({completeness/len(enhanced_data)*100:.1f}%)")
        
        # Count geographic data
        if 'latitud' in enhanced_data.columns and 'longitud' in enhanced_data.columns:
            has_coords = (enhanced_data['latitud'].notna() & enhanced_data['longitud'].notna()).sum()
            print(f"   Records with coordinates: {has_coords}/{len(enhanced_data)} ({has_coords/len(enhanced_data)*100:.1f}%)")

if __name__ == "__main__":
    run_data_processing()