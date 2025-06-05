import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from unidecode import unidecode
import logging
from pathlib import Path
from typing import Optional, Union


class GeoDataCorrector:
    """
    Handle geographic data validation and correction.
    Includes coordinate correction, business rules validation, and data quality fixes.
    """
    
    def __init__(self, shapefile_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.shapefile_dir = shapefile_dir or Path(__file__).parent.parent.parent / 'data' / 'external'
        
        # Load shapefiles for correction
        self.localidades = None
        self.barrios = None
        self._load_shapefiles()
        
    def _load_shapefiles(self):
        """Load required shapefiles for correction operations."""
        try:
            localidades_path = self.shapefile_dir / 'localidades_bogota' / 'loca.shp'
            barrios_path = self.shapefile_dir / 'barrios_bogota' / 'barrios.geojson'
            
            if localidades_path.exists():
                self.localidades = gpd.read_file(localidades_path)
                print(f"Loaded localidades for correction: {len(self.localidades)} localities")
            else:
                print(f"⚠️  Localidades shapefile not found: {localidades_path}")
                
            if barrios_path.exists():
                self.barrios = gpd.read_file(barrios_path)
                # Clean barrio data
                self.barrios['barriocomu'] = self.barrios['barriocomu'].apply(self._normalize_text)
                self.barrios['localidad'] = self.barrios['localidad'].apply(self._normalize_text)
                
                # Fix known issues
                self.barrios.loc[self.barrios['localidad'] == 'RAFAEL URIBE', 'localidad'] = 'RAFAEL URIBE URIBE'
                self.barrios.loc[self.barrios['localidad'].isna(), 'localidad'] = 'SUBA'
                
                print(f"Loaded barrios for correction: {len(self.barrios)} neighborhoods")
            else:
                print(f"⚠️  Barrios shapefile not found: {barrios_path}")
                
        except Exception as e:
            print(f"❌ Error loading shapefiles: {e}")
            
    def correct_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to apply geographic corrections and validation.
        """
        if self.localidades is None or self.barrios is None:
            print("⚠️  Shapefiles not loaded. Skipping geographic correction.")
            return df
            
        print(f"Starting geographic correction for {len(df)} properties...")
        
        # Check for required coordinate columns
        if 'latitud' not in df.columns or 'longitud' not in df.columns:
            print("⚠️  No coordinate columns found. Cannot perform geographic correction.")
            return df
        
        # Create working copy
        corrected_df = df.copy()
        
        # Initialize coords_modified column if not present
        if 'coords_modified' not in corrected_df.columns:
            corrected_df['coords_modified'] = False
        
        try:
            # Step 1: Add missing basic geographic data
            corrected_df = self._add_missing_geographic_data(corrected_df)
            
            # Step 2: Apply coordinate corrections based on sector
            corrected_df = self._apply_coordinate_corrections(corrected_df)
            
            # Step 3: Apply business rules validation
            corrected_df = self._apply_business_rules_validation(corrected_df)
            
            # Step 4: Final data quality checks
            corrected_df = self._final_data_quality_checks(corrected_df)
            
        except Exception as e:
            print(f"❌ Error during geographic correction: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Returning original data without corrections...")
            return df
        
        # Report results
        coords_modified = corrected_df.get('coords_modified', pd.Series(False, index=corrected_df.index)).sum()
        if coords_modified > 0:
            print(f"   📍 Modified coordinates for {coords_modified} properties")
            
        original_count = len(df)
        final_count = len(corrected_df)
        if original_count != final_count:
            print(f"   🗑️  Removed {original_count - final_count} invalid properties")
            
        print(f"✅ Geographic correction completed: {final_count} valid properties")
        return corrected_df
        
    def _normalize_text(self, text):
        """Normalize text by removing accents and converting to uppercase."""
        try:
            return unidecode(str(text)).upper()
        except:
            return text
            
    def _add_missing_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing localidad and barrio data using coordinates."""
        
        # Check if we have coordinate data to work with
        if 'latitud' not in df.columns or 'longitud' not in df.columns:
            print("   ⚠️  No coordinate columns found. Cannot add geographic data.")
            return df
        
        # Count properties with valid coordinates
        valid_coords = df['latitud'].notna() & df['longitud'].notna()
        if valid_coords.sum() == 0:
            print("   ⚠️  No valid coordinates found. Cannot add geographic data.")
            return df
        
        print(f"   Found {valid_coords.sum()} properties with valid coordinates")
        
        # Ensure localidad column exists with proper dtype
        if 'localidad' not in df.columns:
            df['localidad'] = pd.Series(dtype='object')  # FIX: Use object dtype for strings
        else:
            # Convert existing column to object if it's not already
            if df['localidad'].dtype != 'object':
                df['localidad'] = df['localidad'].astype('object')
    
        # Ensure barrio column exists with proper dtype
        if 'barrio' not in df.columns:
            df['barrio'] = pd.Series(dtype='object')  # FIX: Use object dtype for strings
        else:
            # Convert existing column to object if it's not already
            if df['barrio'].dtype != 'object':
                df['barrio'] = df['barrio'].astype('object')
    
        # Add missing localidades (only for properties with valid coordinates)
        missing_localidad = df['localidad'].isna() & valid_coords
        if missing_localidad.sum() > 0:
            try:
                # Apply only to rows with missing localidad AND valid coordinates
                new_localidades = df.loc[missing_localidad].apply(self._get_localidad, axis=1)
                df.loc[missing_localidad, 'localidad'] = new_localidades
                
                # Count successful additions
                added_localidades = df.loc[missing_localidad, 'localidad'].notna().sum()
                print(f"   ✅ Successfully added localidad for {added_localidades} properties")
            except Exception as e:
                print(f"   ⚠️  Error adding localidades: {e}")
    
        # Add missing barrios (only for properties with valid coordinates)
        missing_barrio = df['barrio'].isna() & valid_coords
        if missing_barrio.sum() > 0:
            try:
                # Apply only to rows with missing barrio AND valid coordinates
                new_barrios = df.loc[missing_barrio].apply(self._get_barrio, axis=1)
                df.loc[missing_barrio, 'barrio'] = new_barrios
                
                # Count successful additions
                added_barrios = df.loc[missing_barrio, 'barrio'].notna().sum()
                print(f"   ✅ Successfully added barrio for {added_barrios} properties")
            except Exception as e:
                print(f"   ⚠️  Error adding barrios: {e}")
    
        return df
        
    def _apply_coordinate_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply coordinate corrections based on sector information."""
        print("   Applying sector-based coordinate corrections...")
        # FIX: Apply to each row and reconstruct DataFrame properly
        corrected_rows = []
        for index, row in df.iterrows():
            corrected_row = self._correction_ubication(row)
            corrected_rows.append(corrected_row)
        
        # Reconstruct DataFrame from corrected rows
        corrected_df = pd.DataFrame(corrected_rows, index=df.index)
        return corrected_df
        
    def _apply_business_rules_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Bogotá-specific business rules validation."""
        print("   Applying business rules validation...")
        
        # Remove properties with invalid estrato 7 (only if estrato column exists)
        if 'estrato' in df.columns:
            invalid_estrato = df['estrato'] == 7
            if invalid_estrato.sum() > 0:
                print(f"     Removing {invalid_estrato.sum()} properties with invalid estrato 7")
                df = df[~invalid_estrato]
        else:
            print("     No 'estrato' column found. Skipping estrato validation.")
        
        # Remove unrealistic localidad-estrato combinations (only if both columns exist)
        if 'localidad' in df.columns and 'estrato' in df.columns:
            invalid_combinations = {
                'KENNEDY': [6, 5],
                'RAFAEL URIBE URIBE': [6, 5],
                'LA PAZ CENTRAL': [6],
                'BOSA': [6, 5, 4],
                'USME': [6, 5, 4, 3],
                'SAN CRISTOBAL': [6, 5, 4],
                'CIUDAD BOLIVAR': [6, 5, 4],
                'FONTIBON': [6],
                'LOS MARTIRES': [6, 5, 1],
                'SANTA FE': [6, 5],
                'TUNJUELITO': [6, 5, 4],
                'BARRIOS UNIDOS': [1, 2, 6],
                'TEUSAQUILLO': [1, 2, 6],
                'ANTONIO NARIÑO': [1, 5, 6],
                'CANDELARIA': [6, 5, 4],
            }
            
            total_removed = 0
            for localidad, estratos in invalid_combinations.items():
                for estrato in estratos:
                    invalid_mask = (df['localidad'] == localidad) & (df['estrato'] == estrato)
                    count_removed = invalid_mask.sum()
                    if count_removed > 0:
                        df = df[~invalid_mask]
                        total_removed += count_removed
                        
            if total_removed > 0:
                print(f"     Removed {total_removed} properties with unrealistic localidad-estrato combinations")
        else:
            print("     Missing 'localidad' or 'estrato' columns. Skipping combination validation.")
        
        return df
        
    def _final_data_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data quality validation."""
        
        # Check which location columns exist
        location_cols = []
        if 'localidad' in df.columns:
            location_cols.append('localidad')
        if 'barrio' in df.columns:
            location_cols.append('barrio')
    
        if location_cols:
            # Remove properties without any location data
            missing_location = df[location_cols].isna().all(axis=1)
            if missing_location.sum() > 0:
                print(f"     Removing {missing_location.sum()} properties without location data")
                df = df[~missing_location]
        else:
            print("     No location columns found. Skipping location data validation.")
            
        return df
        
    def _get_localidad(self, row) -> Union[str, float]:
        """Get localidad from coordinates."""
        try:
            if pd.isna(row.get('latitud')) or pd.isna(row.get('longitud')):
                return np.nan  # FIX: Return np.nan (float) instead of str for missing data

            if self.localidades is None:
                return np.nan

            point = Point(row['longitud'], row['latitud'])
            for i, localidad in self.localidades.iterrows():
                if point.within(localidad['geometry']):
                    return self._normalize_text(localidad['LocNombre'])
            return np.nan  # FIX: Return np.nan when no match found
        except:
            return np.nan  # FIX: Return np.nan on error
            
    def _get_barrio(self, row) -> Union[str, float]:
        """Get barrio from coordinates and localidad."""
        try:
            if pd.isna(row.get('latitud')) or pd.isna(row.get('longitud')):
                return np.nan  # FIX: Return np.nan instead of str for missing data
                
            point = Point(row['longitud'], row['latitud'])
            loca = row.get('localidad')
            
            if pd.notna(loca) and self.barrios is not None:
                barrios_localidad = self.barrios.loc[self.barrios['localidad'] == loca]
                for i, barrio in barrios_localidad.iterrows():
                    if point.within(barrio['geometry']):
                        return barrio['barriocomu']
                        
            return np.nan  # FIX: Return np.nan when no match found
        except:
            return np.nan  # FIX: Return np.nan on error
            
    def _random_coords_in_polygon(self, polygon):
        """Generate random coordinates within a polygon."""
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            x = np.random.uniform(polygon.bounds[0], polygon.bounds[2])
            y = np.random.uniform(polygon.bounds[1], polygon.bounds[3])
            point = Point(x, y)
            if polygon.contains(point):
                return x, y
            attempts += 1
            
        # Fallback to polygon centroid if random generation fails
        centroid = polygon.centroid
        return centroid.x, centroid.y
        
    def _correction_ubication(self, row):
        """Correct location based on sector information."""
        sector_dict = {
            'CHICO': {
                'filter': ('barriocomu', 'S.C. CHICO NORTE'),
                'localidad': 'CHAPINERO',
                'barrio': 'S.C. CHICO NORTE'
            },
            'CEDRITOS': {
                'filter': ('barriocomu', 'CEDRITOS'),
                'localidad': 'USAQUEN',
                'barrio': 'CEDRITOS'
            },
            'CHAPINERO ALTO': {
                'filter': ('barriocomu', 'S.C. CHAPINERO NORTE'),
                'localidad': 'CHAPINERO',
                'barrio': 'S.C. CHAPINERO NORTE'
            },
            'LOS ROSALES': {
                'filter': ('barriocomu', 'LOS ROSALES'),
                'localidad': 'CHAPINERO',
                'barrio': 'LOS ROSALES'
            },
            'SANTA BARBARA': {
                'filter': ('barriocomu', 'SANTA BARBARA OCCIDENTAL'),
                'localidad': 'USAQUEN',
                'barrio': 'SANTA BARBARA OCCIDENTAL'
            },
            'COUNTRY': {
                'filter': ('barriocomu', 'NUEVO COUNTRY'),
                'localidad': 'USAQUEN',
                'barrio': 'NUEVO COUNTRY'
            },
            'CENTRO INTERNACIONAL': {
                'filter': ('barriocomu', 'SAMPER'),
                'localidad': 'SANTA FE',
                'barrio': 'SAMPER'
            },
            'CERROS DE SUBA': {
                'filter': ('barriocomu', 'S.C. NIZA SUBA'),
                'localidad': 'SUBA',
                'barrio': 'S.C. NIZA SUBA'
            },
            'NIZA ALHAMBRA': {
                'filter': ('barriocomu', 'NIZA SUR'),
                'localidad': 'SUBA',
                'barrio': 'NIZA SUR'
            }
        }
        
        # FIX: Handle sector field that might be a list or string
        sector = row.get('sector')
        
        # Convert list to string if needed
        if isinstance(sector, list):
            if len(sector) > 0:
                sector = str(sector[0])  # Take first element
            else:
                sector = None
        elif sector is not None:
            sector = str(sector)  # Ensure it's a string
    
        # Normalize sector name
        if sector:
            sector = sector.strip().upper()
        
        if sector in sector_dict and pd.notna(sector):
            sector_info = sector_dict[sector]
            
            if row.get('localidad') != sector_info['localidad']:
                try:
                    # Get polygon based on filter
                    filter_col, filter_val = sector_info['filter']
                    
                    # FIX: Properly handle GeoDataFrame queries
                    polygon_series = None
                    if filter_col == 'barriocomu':
                        if self.barrios is not None:
                            matching_barrios = self.barrios[self.barrios[filter_col] == filter_val]
                            if not matching_barrios.empty:
                                polygon_series = matching_barrios['geometry']
                    else:
                        if self.localidades is not None:
                            matching_localidades = self.localidades[self.localidades[filter_col] == filter_val]
                            if not matching_localidades.empty:
                                polygon_series = matching_localidades['geometry']
                    
                    # FIX: Check if we found a valid polygon
                    if polygon_series is not None and len(polygon_series) > 0:
                        polygon = polygon_series.iloc[0]  # FIX: Access iloc properly
                        x, y = self._random_coords_in_polygon(polygon)
                        
                        row['latitud'] = float(y)
                        row['longitud'] = float(x)
                        row['coords_modified'] = True
                        row['localidad'] = sector_info['localidad']
                        row['barrio'] = sector_info['barrio']
                        
                except Exception as e:
                    self.logger.warning(f"Could not correct location for sector {sector}: {e}")
                
        return row


# Helper function for easy integration
def correct_apartment_locations(df: pd.DataFrame, shapefile_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Convenience function to correct apartment geographic data.
    
    Args:
        df: DataFrame with apartment data
        shapefile_dir: Path to directory containing shapefiles
        
    Returns:
        Corrected DataFrame with validated geographic data
    """
    corrector = GeoDataCorrector(shapefile_dir)
    return corrector.correct_geographic_data(df)