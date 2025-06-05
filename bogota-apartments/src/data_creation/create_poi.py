"""
POI (Points of Interest) Data Creation Module

This module creates POI GeoJSON files from various external data sources
for use in geographic enrichment analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class POICreator:
    """Creates POI (Points of Interest) data from external CSV sources."""
    
    def __init__(self, external_dir: Path):
        """
        Initialize POI Creator.
        
        Args:
            external_dir: Path to external data directory
        """
        self.external_dir = Path(external_dir)
        self.pois = []
        
    def create_shopping_pois(self) -> int:
        """
        Create shopping POIs from centros_comerciales data.
        
        Returns:
            Number of POIs created
        """
        shopping_file = self.external_dir / 'centros_comerciales' / 'centros_comerciales_bogota.csv'
        
        if not shopping_file.exists():
            logger.warning(f"Shopping centers file not found: {shopping_file}")
            return 0
            
        try:
            logger.info("Processing shopping centers...")
            shopping_df = pd.read_csv(shopping_file)
            
            # Check required columns
            required_cols = ['NAME', 'LATITUD', 'LONGITUD', 'LOCALIDAD']
            if not all(col in shopping_df.columns for col in required_cols):
                logger.warning(f"Missing required columns in shopping data")
                logger.info(f"Available columns: {list(shopping_df.columns)}")
                return 0
            
            # Filter valid coordinates
            valid_coords = shopping_df['LATITUD'].notna() & shopping_df['LONGITUD'].notna()
            valid_shopping = shopping_df[valid_coords]
            
            count = 0
            for _, row in valid_shopping.iterrows():
                try:
                    lat, lon = float(row['LATITUD']), float(row['LONGITUD'])
                    
                    # Validate coordinate ranges (rough bounds for Bogotá)
                    if not (4.0 <= lat <= 5.0 and -75.0 <= lon <= -73.5):
                        continue
                        
                    poi = {
                        'type': 'shopping',
                        'name': str(row['NAME']).strip(),
                        'category': 'mall',
                        'address': row.get('ADDRESS', ''),
                        'localidad': str(row['LOCALIDAD']).strip(),
                        'url': row.get('URL', ''),
                        'coordinates': [lon, lat]  # [longitude, latitude] for GeoJSON
                    }
                    self.pois.append(poi)
                    count += 1
                    
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid shopping coordinates: {e}")
                    continue
            
            logger.info(f"Added {count} shopping centers")
            return count
            
        except Exception as e:
            logger.error(f"Error processing shopping centers: {e}")
            return 0
    
    def create_recreation_pois(self) -> int:
        """
        Create recreation POIs from espacios_para_deporte data.
        Simplified to use single 'park' category for better performance.
        
        Returns:
            Number of POIs created
        """
        recreation_file = self.external_dir / 'espacios_para_deporte_bogota' / \
            'directorio-parques-y-escenarios-2023.csv'

        if not recreation_file.exists():
            logger.warning(f"Recreation file not found: {recreation_file}")
            return 0

        try:
            logger.info("Processing parks and recreation areas...")
            recreation_df = pd.read_csv(recreation_file)

            # Check required columns
            required_cols = ['NOMBRE DEL PARQUE O ESCENARIO',
                            'LATITUD', 'LONGITUD', 'LOCALIDAD']
            if not all(col in recreation_df.columns for col in required_cols):
                logger.warning(f"Missing required columns in recreation data")
                logger.info(f"Available columns: {list(recreation_df.columns)}")
                return 0

            # Filter valid coordinates
            valid_coords = recreation_df['LATITUD'].notna(
            ) & recreation_df['LONGITUD'].notna()
            valid_recreation = recreation_df[valid_coords]

            count = 0
            for _, row in valid_recreation.iterrows():
                try:
                    lat, lon = float(row['LATITUD']), float(row['LONGITUD'])

                    # Validate coordinate ranges (bounds for Bogotá)
                    if not (4.0 <= lat <= 5.0 and -75.0 <= lon <= -73.5):
                        continue

                    # SIMPLIFIED: All recreation areas are categorized as 'park'
                    poi = {
                        'type': 'recreation',
                        'name': str(row['NOMBRE DEL PARQUE O ESCENARIO']).strip(),
                        'category': 'park',  # Simplified single category
                        # Keep original for reference
                        'park_classification': str(row.get('TIPO DE PARQUE', '')).strip(),
                        'address': row.get('DIRECCIÓN', ''),
                        'localidad': str(row['LOCALIDAD']).strip(),
                        'coordinates': [lon, lat]
                    }
                    self.pois.append(poi)
                    count += 1

                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid recreation coordinates: {e}")
                    continue

            logger.info(
                f"Added {count} parks and recreation areas (all as 'park' category)")
            return count

        except Exception as e:
            logger.error(f"Error processing recreation data: {e}")
            return 0
    
    def create_transportation_pois(self) -> int:
        """
        Create transportation POIs from SITP data.
        
        Returns:
            Number of POIs created
        """
        sitp_file = self.external_dir / 'paraderos_zonales_SITP' / 'Paraderos_Zonales_del_SITP.csv'
        
        if not sitp_file.exists():
            logger.info("SITP transportation file not found, skipping...")
            return 0
            
        try:
            logger.info("Processing SITP transportation stops...")
            sitp_df = pd.read_csv(sitp_file)
            
            # Try to find coordinate columns (names may vary)
            lat_cols = [col for col in sitp_df.columns if any(keyword in col.lower() for keyword in ['lat', 'y', 'norte'])]
            lon_cols = [col for col in sitp_df.columns if any(keyword in col.lower() for keyword in ['lon', 'x', 'este', 'oeste'])]
            name_cols = [col for col in sitp_df.columns if any(keyword in col.lower() for keyword in ['nombre', 'name', 'denominacion'])]
            
            if not lat_cols or not lon_cols:
                logger.warning("Could not find coordinate columns in SITP data")
                logger.info(f"Available columns: {list(sitp_df.columns)}")
                return 0
            
            lat_col, lon_col = lat_cols[0], lon_cols[0]
            name_col = name_cols[0] if name_cols else lat_cols[0]  # Fallback
            
            # Filter valid coordinates and sample to avoid overcrowding
            valid_coords = sitp_df[lat_col].notna() & sitp_df[lon_col].notna()
            sampled_sitp = sitp_df[valid_coords].iloc[::5]  # Every 5th stop
            
            count = 0
            for _, row in sampled_sitp.iterrows():
                try:
                    lat, lon = float(row[lat_col]), float(row[lon_col])
                    
                    # Validate coordinate ranges
                    if not (4.0 <= lat <= 5.0 and -75.0 <= lon <= -73.5):
                        continue
                    
                    poi = {
                        'type': 'transportation',
                        'name': str(row.get(name_col, 'Parada SITP')).strip(),
                        'category': 'bus_stop',
                        'coordinates': [lon, lat]
                    }
                    self.pois.append(poi)
                    count += 1
                    
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid transportation coordinates: {e}")
                    continue
            
            logger.info(f"Added {count} transportation stops")
            return count
            
        except Exception as e:
            logger.error(f"Error processing SITP data: {e}")
            return 0
    
    def save_as_geojson(self, output_file: Path) -> bool:
        """
        Save collected POIs as GeoJSON file.
        
        Args:
            output_file: Path where to save the GeoJSON file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pois:
            logger.warning("No POIs to save")
            return False
            
        try:
            # Convert to GeoJSON format
            features = []
            for poi in self.pois:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": poi.pop('coordinates')  # Remove and use for geometry
                    },
                    "properties": poi  # All other fields go to properties
                }
                features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Save to file
            import json
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)
            
            # Summary by type
            type_counts = {}
            for poi in self.pois:
                poi_type = poi['type']
                type_counts[poi_type] = type_counts.get(poi_type, 0) + 1
            
            logger.info(f"✅ Created POI file with {len(self.pois)} points:")
            for poi_type, count in type_counts.items():
                logger.info(f"   • {poi_type}: {count} locations")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving POI file: {e}")
            return False
    
    def create_all_pois(self, output_file: Path) -> bool:
        """
        Create all POI types and save to file.
        
        Args:
            output_file: Path where to save the GeoJSON file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating POI data from external sources...")
        
        # Clear any existing POIs
        self.pois = []
        
        # Create POIs from different sources
        shopping_count = self.create_shopping_pois()
        recreation_count = self.create_recreation_pois()
        transportation_count = self.create_transportation_pois()
        
        total_count = shopping_count + recreation_count + transportation_count
        
        if total_count == 0:
            logger.warning("No POI data could be processed")
            return False
        
        # Save to file
        return self.save_as_geojson(output_file)


# Convenience function for easy integration
def create_poi_file(external_dir: Path, output_file: Path) -> bool:
    """
    Create POI GeoJSON file from external data sources.
    
    Args:
        external_dir: Path to external data directory
        output_file: Path where to save the POI GeoJSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        creator = POICreator(external_dir)
        return creator.create_all_pois(output_file)
    except Exception as e:
        logger.error(f"Error creating POI file: {e}")
        return False


if __name__ == "__main__":
    # For testing/standalone usage
    import sys
    from pathlib import Path
    
    if len(sys.argv) != 3:
        print("Usage: python create_poi.py <external_dir> <output_file>")
        print("Example: python create_poi.py data/external data/geo/poi.geojson")
        sys.exit(1)
    
    external_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    success = create_poi_file(external_dir, output_file)
    if success:
        print(f"✅ POI file created successfully: {output_file}")
    else:
        print("❌ Failed to create POI file")
        sys.exit(1)