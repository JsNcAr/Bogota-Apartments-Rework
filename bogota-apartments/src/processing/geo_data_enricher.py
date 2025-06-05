import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from unidecode import unidecode
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class GeoDataEnricher:
    """
    Enrich apartment data with advanced geographic, market, and accessibility insights.
    Assumes data has already been corrected/validated by GeoDataCorrector.
    
    Focus: Market intelligence, transportation access, POI proximity, investment scoring.
    """

    def __init__(self, geo_data_path: str = "data/external"):
        self.logger = logging.getLogger(__name__)
        self.geo_data_path = Path(geo_data_path)

        # Initialize geodata containers
        self.localidades_gdf = None
        self.barrios_gdf = None
        self.transmilenio_gdf = None
        self.poi_gdf = None

        # Market data containers
        self.market_stats = {}

        # Load geodata if available
        self._load_geodata()

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main enrichment method that applies all enrichment steps.
        Assumes data has been validated and corrected by GeoDataCorrector.
        """
        self.logger.info(f"Starting geographic enrichment for {len(df)} records...")

        # Make a copy to avoid modifying original data
        df_enriched = df.copy()

        # Apply enrichment steps
        df_enriched = self._enrich_market_data(df_enriched)
        df_enriched = self._enrich_accessibility_data(df_enriched)
        df_enriched = self._enrich_poi_proximity(df_enriched)
        df_enriched = self._enrich_comparative_metrics(df_enriched)

        self.logger.info(
            f"Geographic enrichment completed. Added {len(df_enriched.columns) - len(df.columns)} new columns")
        
        print(f"🚀 Geographic enrichment completed")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   Enriched columns: {len(df_enriched.columns)}")
        print(f"   New features added: {len(df_enriched.columns) - len(df.columns)}")
        
        return df_enriched

    def _load_geodata(self):
        """
        Load geographic data files if they exist.
        """
        try:
            # Load localidades (districts)
            localidades_path = self.geo_data_path / "localidades.shp"
            if localidades_path.exists():
                self.localidades_gdf = gpd.read_file(localidades_path)
            else:
                print(f"⚠️  Localidades geodata not found: {localidades_path}")

            # Load barrios (neighborhoods)
            barrios_path = self.geo_data_path / "barrios.geojson"
            if barrios_path.exists():
                self.barrios_gdf = gpd.read_file(barrios_path)
            else:
                print(f"⚠️  Barrios geodata not found: {barrios_path}")

            # Load TransMilenio stations
            tm_path = self.geo_data_path / "transmilenio.geojson"
            if tm_path.exists():
                self.transmilenio_gdf = gpd.read_file(tm_path)
            else:
                print(f"⚠️  TransMilenio geodata not found: {tm_path}")

            # Load POIs (Points of Interest)
            poi_path = self.geo_data_path / "poi.geojson"
            if poi_path.exists():
                self.poi_gdf = gpd.read_file(poi_path)
            else:
                print(f"⚠️  POI geodata not found: {poi_path}")

        except Exception as e:
            print(f"⚠️  Could not load some geodata files: {e}")
            self.logger.warning(f"Could not load some geodata files: {e}")

    def _enrich_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with market statistics and comparisons."""

        # Calculate market statistics by sector
        self._calculate_market_stats(df)

        # Add market comparisons
        for groupby_col in ['sector', 'localidad', 'estrato']:
            if groupby_col in df.columns:
                df = self._add_market_comparisons(df, groupby_col)

        # Add price trends and market position
        df = self._add_price_trends(df)

        return df

    def _enrich_accessibility_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with accessibility and transportation data."""

        # Distance to TransMilenio
        if self.transmilenio_gdf is not None and not self.transmilenio_gdf.empty:
            print("     Calculating distances to TransMilenio...")
            
            # Ensure we have valid coordinate columns
            if 'latitud' in df.columns and 'longitud' in df.columns:
                # Calculate both distance and closest station name
                transmilenio_results = df.apply(
                    lambda row: self._calculate_min_distance_with_info(
                        row, self.transmilenio_gdf, 'latitud', 'longitud'
                    ), axis=1
                )
                
                # Split results into distance and name columns
                df['distancia_transmilenio'] = transmilenio_results.apply(lambda x: x[0])
                df['estacion_transmilenio_cercana'] = transmilenio_results.apply(lambda x: x[1])

                # Accessibility score based on TransMilenio proximity
                df['score_transmilenio'] = self._calculate_accessibility_score(
                    df['distancia_transmilenio'])
                
                print(f"     Added TransMilenio distances and closest station names")
            else:
                print("     No coordinate columns found for TransMilenio distance calculation")
                df['distancia_transmilenio'] = np.nan
                df['estacion_transmilenio_cercana'] = None
                df['score_transmilenio'] = 0.5
        else:
            print("     No TransMilenio data available")
            df['distancia_transmilenio'] = np.nan
            df['estacion_transmilenio_cercana'] = None
            df['score_transmilenio'] = 0.5  # Neutral score

        # Add mobility indicators
        df = self._add_mobility_indicators(df)

        return df

    def _enrich_poi_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with Points of Interest proximity data (SIMPLIFIED VERSION)."""
        print("   Calculating POI proximity...")

        if self.poi_gdf is not None and not self.poi_gdf.empty:
            # Ensure we have valid coordinate columns
            if 'latitud' in df.columns and 'longitud' in df.columns:
                
                # SIMPLIFIED: Only main POI types with single categories
                primary_poi_types = ['shopping', 'recreation', 'transportation']
                
                # Calculate distances to primary POI types only
                for poi_type in primary_poi_types:
                    poi_subset = self.poi_gdf[self.poi_gdf['type'] == poi_type]
                    if not poi_subset.empty:
                        print(f"     Calculating distances to {poi_type} POIs...")
                        
                        # Calculate both distance and closest POI name
                        poi_results = df.apply(
                            lambda row: self._calculate_min_distance_with_info(
                                row, poi_subset, 'latitud', 'longitud'
                            ), axis=1
                        )
                        
                        # Split results into distance and name columns
                        df[f'distancia_{poi_type}'] = poi_results.apply(lambda x: x[0])
                        df[f'{poi_type}_cercano'] = poi_results.apply(lambda x: x[1])
                    else:
                        print(f"     No {poi_type} POIs found")
                        df[f'distancia_{poi_type}'] = np.nan
                        df[f'{poi_type}_cercano'] = None
        
            else:
                print("     No coordinate columns found for POI distance calculation")
                # Set all POI distances to NaN
                primary_poi_types = ['shopping', 'recreation', 'transportation']
                for poi_type in primary_poi_types:
                    df[f'distancia_{poi_type}'] = np.nan
                    df[f'{poi_type}_cercano'] = None
        else:
            print("     No POI data available")
            # Set all POI distances to NaN
            primary_poi_types = ['shopping', 'recreation', 'transportation']
            for poi_type in primary_poi_types:
                df[f'distancia_{poi_type}'] = np.nan
                df[f'{poi_type}_cercano'] = None

        # Calculate simplified convenience scores

        return df

    def _enrich_comparative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comparative metrics and rankings."""
        print("   Calculating comparative metrics...")

        # Price percentiles by different groupings
        for group_col in ['sector', 'estrato', 'tipo_propiedad']:
            if group_col in df.columns:
                price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'
                if price_col in df.columns:
                    df[f'percentil_precio_{group_col}'] = df.groupby(
                        group_col)[price_col].rank(pct=True)

        # Area percentiles
        if 'area' in df.columns:
            df['percentil_area'] = df['area'].rank(pct=True)

        # Overall desirability score
        df['score_deseabilidad'] = self._calculate_desirability_score(df)

        # Investment potential score
        df['score_inversion'] = self._calculate_investment_score(df)

        return df

    def _calculate_market_stats(self, df: pd.DataFrame):
        """Calculate market statistics by different groupings."""
        price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'
        
        if price_col not in df.columns:
            print("     No price column available for market statistics")
            return

        for group_col in ['sector', 'localidad', 'estrato']:
            if group_col in df.columns:
                try:
                    stats = df.groupby(group_col)[price_col].agg([
                        'mean', 'median', 'std', 'count'
                    ]).round(0)
                    self.market_stats[group_col] = stats.to_dict('index')
                except Exception as e:
                    print(f"     Warning: Could not calculate market stats for {group_col}: {e}")

    def _add_market_comparisons(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Add market comparison metrics."""
        price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'
        
        if price_col not in df.columns:
            return df

        try:
            # Average price by group
            group_avg = df.groupby(group_col)[price_col].transform('mean')
            df[f'precio_promedio_{group_col}'] = group_avg

            # Price difference from group average
            df[f'diferencia_promedio_{group_col}'] = df[price_col] - group_avg

            # Price ratio to group average
            df[f'ratio_promedio_{group_col}'] = df[price_col] / group_avg

        except Exception as e:
            print(f"     Warning: Could not add market comparisons for {group_col}: {e}")

        return df

    def _add_price_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price trend indicators."""
        price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'
        
        if price_col not in df.columns:
            return df

        try:
            # Price category within dataset
            df['categoria_precio_mercado'] = pd.qcut(
                df[price_col],
                q=5,
                labels=['muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto'],
                duplicates='drop'
            )

            # Price per m2 category
            if 'area' in df.columns:
                precio_m2 = df[price_col] / df['area']
                df['precio_m2_categoria'] = pd.qcut(
                    precio_m2,
                    q=5,
                    labels=['economico', 'accesible', 'promedio', 'premium', 'lujo'],
                    duplicates='drop'
                )

        except Exception as e:
            print(f"     Warning: Could not add price trends: {e}")

        return df

    def _ensure_projected_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Ensure GeoDataFrame is in a projected CRS suitable for distance calculations.
        
        Args:
            gdf: GeoDataFrame to check/convert
            
        Returns:
            GeoDataFrame in projected CRS
        """
        if gdf is None or gdf.empty:
            return gdf
        
        # If no CRS is set, assume WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        # If already in projected CRS suitable for Colombia, return as-is
        if gdf.crs is not None and getattr(gdf.crs, "to_epsg", None) is not None and gdf.crs.to_epsg() == 3116:  # MAGNA-SIRGAS / Colombia Bogota zone
            return gdf
        
        # Convert to projected CRS for accurate distance calculations
        return gdf.to_crs('EPSG:3116')

    def _calculate_min_distance(self, row, gdf: Optional[gpd.GeoDataFrame], lat_col: str, lon_col: str) -> float:
        """Calculate minimum distance to features in a GeoDataFrame."""
        try:
            if gdf is None:
                return np.nan
            
            # Extract coordinates
            lat = row.get(lat_col)
            lon = row.get(lon_col)
            
            # Validate coordinates exist
            if pd.isna(lat) or pd.isna(lon):
                return np.nan
            
            # Handle coordinate extraction and clean precision
            try:
                # If coordinates are in a list/array format, extract first element
                if hasattr(lat, '__len__') and not isinstance(lat, str):
                    lat = lat[0] if len(lat) > 0 else np.nan
                if hasattr(lon, '__len__') and not isinstance(lon, str):
                    lon = lon[0] if len(lon) > 0 else np.nan
                
                # Convert to float and round to 6 decimal places (11cm precision)
                lat = round(float(lat), 6)
                lon = round(float(lon), 6)
                
                # Validate coordinate ranges for Bogotá
                if not (3.5 <= lat <= 5.5 and -75.5 <= lon <= -73.0):
                    return np.nan
                    
            except (ValueError, TypeError, IndexError) as e:
                print(f"     Warning: Invalid coordinates for row {row.name}: {e}")
                return np.nan

            # Create point and ensure it's in the same projected CRS
            point = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
            
            # Convert both to projected CRS
            gdf_projected = self._ensure_projected_crs(gdf)
            point_projected = self._ensure_projected_crs(point_gdf)
            
            # Calculate distances (now in meters)
            distances = gdf_projected.geometry.distance(point_projected.geometry.values[0])
            return float(distances.min())
        
        except Exception as e:
            # Log the error for debugging but don't crash
            print(f"     Warning: Could not calculate distance for row {row.name}: {e}")
            return np.nan

    def _calculate_min_distance_with_info(self, row, gdf: Optional[gpd.GeoDataFrame], lat_col: str, lon_col: str, name_col: str = 'name') -> tuple:
        """
        Calculate minimum distance to features in a GeoDataFrame and return info about closest feature.
        
        Args:
            row: DataFrame row with coordinates
            gdf: GeoDataFrame with features to calculate distance to
            lat_col: Name of latitude column in row
            lon_col: Name of longitude column in row
            name_col: Name of column in gdf that contains feature names
            
        Returns:
            Tuple of (distance_in_meters, closest_feature_name)
        """
        try:
            if gdf is None or gdf.empty:
                return np.nan, None
            
            # Extract coordinates
            lat = row.get(lat_col)
            lon = row.get(lon_col)
            
            # Validate coordinates exist
            if pd.isna(lat) or pd.isna(lon):
                return np.nan, None
            
            # Handle coordinate extraction and clean precision
            try:
                # If coordinates are in a list/array format, extract first element
                if hasattr(lat, '__len__') and not isinstance(lat, str):
                    lat = lat[0] if len(lat) > 0 else np.nan
                if hasattr(lon, '__len__') and not isinstance(lon, str):
                    lon = lon[0] if len(lon) > 0 else np.nan
                
                # Convert to float and round to 6 decimal places (11cm precision)
                lat = round(float(lat), 6)
                lon = round(float(lon), 6)
                
                # Validate coordinate ranges for Bogotá
                if not (3.5 <= lat <= 5.5 and -75.5 <= lon <= -73.0):
                    return np.nan, None
                    
            except (ValueError, TypeError, IndexError) as e:
                return np.nan, None

            # Create point and ensure it's in the same projected CRS
            point = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
            
            # Convert both to projected CRS
            gdf_projected = self._ensure_projected_crs(gdf)
            point_projected = self._ensure_projected_crs(point_gdf)
            
            # Calculate distances (now in meters)
            distances = gdf_projected.geometry.distance(point_projected.geometry.values[0])
            
            # Find closest feature
            min_distance_idx = distances.idxmin()
            min_distance = float(distances.min())
            
            # Get name of closest feature
            closest_name = None
            if name_col in gdf_projected.columns:
                closest_name = gdf_projected.loc[min_distance_idx, name_col]
            elif 'nombre' in gdf_projected.columns:
                closest_name = gdf_projected.loc[min_distance_idx, 'nombre']
            elif 'NOMBRE' in gdf_projected.columns:
                closest_name = gdf_projected.loc[min_distance_idx, 'NOMBRE']
            elif 'nombre_estacion' in gdf_projected.columns:
                closest_name = gdf_projected.loc[min_distance_idx,
                                                 'nombre_estacion']
            else:
                # Fallback: use index or first text column
                text_columns = gdf_projected.select_dtypes(include=['object', 'string']).columns
                if len(text_columns) > 0:
                    closest_name = gdf_projected.loc[min_distance_idx, text_columns[0]]
                else:
                    closest_name = f"Station_{min_distance_idx}"
            
            # Clean the name
            if closest_name and isinstance(closest_name, str):
                closest_name = closest_name.strip()
            
            return min_distance, closest_name
        
        except Exception as e:
            # Log the error for debugging but don't crash
            print(f"     Warning: Could not calculate distance with info for row {getattr(row, 'name', 'unknown')}: {e}")
            return np.nan, None

    def _calculate_accessibility_score(self, distances: pd.Series) -> pd.Series:
        """
        Calculate accessibility score based on distances.
        
        Args:
            distances: Series of distances in meters
            
        Returns:
            Series of accessibility scores (0-1, higher is better)
        """
        # Score decreases with distance
        # Max score (1.0) at 0m, score approaches 0 at 2000m
        scores = np.maximum(0, 1 - (distances / 2000))
        return pd.Series(scores, index=distances.index).fillna(0.5)

    def _add_mobility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mobility and transportation indicators."""

        # Parking availability indicator
        if 'parqueaderos' in df.columns:
            df['facilidad_parqueo'] = df['parqueaderos'].fillna(0) > 0
        else:
            df['facilidad_parqueo'] = False

        # Transportation score (combination of TransMilenio access and parking)
        transport_score = pd.Series(0.0, index=df.index)
        
        if 'score_transmilenio' in df.columns:
            transport_score += df['score_transmilenio'] * 0.7
            
        if 'facilidad_parqueo' in df.columns:
            transport_score += df['facilidad_parqueo'].astype(int) * 0.3

        df['score_movilidad'] = transport_score

        return df



    def _calculate_desirability_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall desirability score."""
        score = pd.Series(0.0, index=df.index)

        # Location score (estrato)
        if 'estrato' in df.columns:
            score += (df['estrato'].fillna(3) / 6.0) * 0.3

        # Accessibility score
        if 'score_movilidad' in df.columns:
            score += df['score_movilidad'] * 0.3

        # Convenience score
        if 'score_conveniencia' in df.columns:
            score += df['score_conveniencia'] * 0.2

        # Property features score
        features_score = pd.Series(0.0, index=df.index)
        
        if 'habitaciones' in df.columns:
            features_score += np.minimum(df['habitaciones'].fillna(0) / 4.0, 1.0) * 0.5
            
        if 'banos' in df.columns:
            features_score += np.minimum(df['banos'].fillna(0) / 3.0, 1.0) * 0.3
            
        if 'parqueaderos' in df.columns:
            features_score += (df['parqueaderos'].fillna(0) > 0).astype(int) * 0.2

        score += features_score * 0.2

        return score.round(3)

    def _calculate_investment_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate investment potential score."""
        score = pd.Series(0.0, index=df.index)

        # Price efficiency (lower price per m2 in good locations = better investment)
        if 'precio_por_m2' in df.columns and 'estrato' in df.columns:
            try:
                # Normalize price per m2 by estrato
                price_efficiency = 1 - (df['precio_por_m2'].rank(pct=True, na_option='bottom'))
                location_quality = df['estrato'].fillna(3) / 6.0
                score += (price_efficiency * location_quality) * 0.4
            except Exception as e:
                print(f"     Warning: Could not calculate price efficiency: {e}")

        # Growth potential (periphery areas with good access)
        if 'zona_ciudad' in df.columns and 'score_movilidad' in df.columns:
            growth_areas = df['zona_ciudad'].isin(['norte', 'periferia', 'oriente'])
            good_access = df['score_movilidad'] > 0.6
            score += (growth_areas & good_access).astype(int) * 0.3

        # Market position (properties below sector average might have growth potential)
        if 'ratio_promedio_sector' in df.columns:
            below_average = df['ratio_promedio_sector'] < 0.9
            score += below_average.astype(int) * 0.3

        return score.round(3)



# Helper function for easy integration
def enrich_apartment_locations(df: pd.DataFrame, geo_data_path: str = "data/geo") -> pd.DataFrame:
    """
    Convenience function to enrich apartment data with geographic intelligence.
    
    Args:
        df: DataFrame with apartment data (should be corrected/validated)
        geo_data_path: Path to directory containing geodata files
        
    Returns:
        Enriched DataFrame with market and accessibility insights
    """
    enricher = GeoDataEnricher(geo_data_path)
    return enricher.enrich(df)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'codigo': ['A001', 'A002', 'A003'],
        'precio_venta': [300_000_000, 500_000_000, 400_000_000],
        'area': [70, 120, 85],
        'habitaciones': [2, 3, 2],
        'banos': [2, 2, 2],
        'parqueaderos': [1, 2, 0],
        'sector': ['Chapinero', 'Kennedy', 'Chicó'],
        'estrato': [4, 2, 6],
        'latitud': [4.6486, 4.6097, 4.6728],
        'longitud': [-74.0570, -74.0817, -74.0310],
        'localidad': ['CHAPINERO', 'KENNEDY', 'CHAPINERO'],
        'barrio': ['CHAPINERO CENTRAL', 'KENNEDY CENTRAL', 'CHICO NORTE']
    })

    enricher = GeoDataEnricher()
    enriched_data = enricher.enrich(sample_data)

    print("\nOriginal columns:", len(sample_data.columns))
    print("Enriched columns:", len(enriched_data.columns))
    print("\nNew enrichment columns:")
    new_cols = [col for col in enriched_data.columns if col not in sample_data.columns]
    for col in new_cols:
        print(f"- {col}")