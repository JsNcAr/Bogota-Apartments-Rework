import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from unidecode import unidecode
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from datetime import datetime


class DataEnricher:
    """
    Enrich apartment data with additional geographic, demographic, and market information.
    """

    def __init__(self, geo_data_path: str = "data/geo"):
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
        """
        self.logger.info(f"Starting data enrichment for {len(df)} records...")

        # Make a copy to avoid modifying original data
        df_enriched = df.copy()

        # Apply enrichment steps
        df_enriched = self._enrich_geographic_data(df_enriched)
        df_enriched = self._enrich_market_data(df_enriched)
        df_enriched = self._enrich_accessibility_data(df_enriched)
        df_enriched = self._enrich_poi_proximity(df_enriched)
        df_enriched = self._enrich_comparative_metrics(df_enriched)

        self.logger.info(
            f"Data enrichment completed. Added {len(df_enriched.columns) - len(df.columns)} new columns")
        return df_enriched

    def _load_geodata(self):
        """
        Load geographic data files if they exist.
        """
        try:
            # Load localidades (districts)
            localidades_path = self.geo_data_path / "localidades.geojson"
            if localidades_path.exists():
                self.localidades_gdf = gpd.read_file(localidades_path)
                self.logger.info("Loaded localidades geodata")

            # Load barrios (neighborhoods)
            barrios_path = self.geo_data_path / "barrios.geojson"
            if barrios_path.exists():
                self.barrios_gdf = gpd.read_file(barrios_path)
                self.logger.info("Loaded barrios geodata")

            # Load TransMilenio stations
            tm_path = self.geo_data_path / "transmilenio.geojson"
            if tm_path.exists():
                self.transmilenio_gdf = gpd.read_file(tm_path)
                self.logger.info("Loaded TransMilenio geodata")

            # Load POIs (Points of Interest)
            poi_path = self.geo_data_path / "poi.geojson"
            if poi_path.exists():
                self.poi_gdf = gpd.read_file(poi_path)
                self.logger.info("Loaded POI geodata")

        except Exception as e:
            self.logger.warning(f"Could not load some geodata files: {e}")

    def _enrich_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with geographic information using coordinates."""

        # Add localidad (district) information
        if self.localidades_gdf is not None and 'latitud' in df.columns and 'longitud' in df.columns:
            self.logger.info("Enriching with localidad data...")
            df['localidad'] = df.apply(
                lambda row: self._get_localidad(row, self.localidades_gdf), axis=1 # type: ignore
            ) # type: ignore

        # Add barrio (neighborhood) information
        if self.barrios_gdf is not None and 'localidad' in df.columns:
            self.logger.info("Enriching with barrio data...")
            df['barrio'] = df.apply(
                lambda row: self._get_barrio(row, self.barrios_gdf), axis=1 # type: ignore
            )

        # Add zone classification based on location
        df['zona_ciudad'] = df.apply(self._classify_city_zone, axis=1)

        return df

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
        if self.transmilenio_gdf is not None:
            self.logger.info("Calculating distances to TransMilenio...")
            df['distancia_transmilenio'] = df.apply(
                lambda row: self._calculate_min_distance(
                    row, self.transmilenio_gdf, 'latitud', 'longitud' # type: ignore
                ), axis=1
            )

            # Accessibility score based on TransMilenio proximity
            df['score_transmilenio'] = self._calculate_accessibility_score(
                df['distancia_transmilenio'])

        # Add mobility indicators
        df = self._add_mobility_indicators(df)

        return df

    def _enrich_poi_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with Points of Interest proximity data."""

        if self.poi_gdf is not None:
            # Calculate distances to different types of POIs
            poi_types = ['shopping', 'education',
                         'health', 'recreation', 'finance']

            for poi_type in poi_types:
                poi_subset = self.poi_gdf[self.poi_gdf['type'] == poi_type]
                if not poi_subset.empty:
                    df[f'distancia_{poi_type}'] = df.apply(
                        lambda row: self._calculate_min_distance(
                            row, poi_subset, 'latitud', 'longitud'
                        ), axis=1
                    )

        # Calculate convenience score
        df['score_conveniencia'] = self._calculate_convenience_score(df)

        return df

    def _enrich_comparative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comparative metrics and rankings."""

        # Price percentiles by different groupings
        for group_col in ['sector', 'estrato', 'tipo_propiedad']:
            if group_col in df.columns:
                price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'
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

    def _get_localidad(self, row, localidades: Optional[gpd.GeoDataFrame]) -> Optional[str]:
        """Get localidad name from coordinates."""
        try:
            if localidades is None or pd.isna(row['latitud']) or pd.isna(row['longitud']):
                return None

            point = Point(row['longitud'], row['latitud'])
            for i, localidad in localidades.iterrows():
                if point.within(localidad['geometry']):
                    return unidecode(localidad['LocNombre']).upper()
            return None
        except Exception:
            return None

    def _get_barrio(self, row, barrios: gpd.GeoDataFrame) -> Optional[str]:
        """Get barrio name from coordinates and localidad."""
        try:
            if pd.isna(row['latitud']) or pd.isna(row['longitud']) or pd.isna(row.get('localidad')):
                return None

            point = Point(row['longitud'], row['latitud'])
            loca = row['localidad']
            barrios_localidad = barrios.loc[barrios['localidad'] == loca]

            for i, barrio in barrios_localidad.iterrows():
                if point.within(barrio['geometry']):
                    return barrio['barriocomu']

            return None
        except Exception:
            return None

    def _classify_city_zone(self, row) -> str:
        """Classify property into city zones."""
        if pd.isna(row.get('latitud')) or pd.isna(row.get('longitud')):
            return 'unknown'

        lat, lon = row['latitud'], row['longitud']

        # Bogotá city center coordinates (approximate)
        center_lat, center_lon = 4.6097, -74.0817

        # Calculate distance from center (rough approximation)
        lat_diff = abs(lat - center_lat)
        lon_diff = abs(lon - center_lon)
        distance = (lat_diff**2 + lon_diff**2)**0.5

        if distance < 0.05:
            return 'centro'
        elif distance < 0.1:
            return 'centro_expandido'
        elif lat > center_lat:
            return 'norte'
        elif lat < center_lat:
            return 'sur'
        else:
            return 'periferia'

    def _calculate_market_stats(self, df: pd.DataFrame):
        """Calculate market statistics by different groupings."""

        price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'

        for group_col in ['sector', 'localidad', 'estrato']:
            if group_col in df.columns:
                stats = df.groupby(group_col)[price_col].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(0)
                self.market_stats[group_col] = stats.to_dict('index')

    def _add_market_comparisons(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Add market comparison metrics."""

        price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'

        # Average price by group
        group_avg = df.groupby(group_col)[price_col].transform('mean')
        df[f'precio_promedio_{group_col}'] = group_avg

        # Price difference from group average
        df[f'diferencia_promedio_{group_col}'] = df[price_col] - group_avg

        # Price ratio to group average
        df[f'ratio_promedio_{group_col}'] = df[price_col] / group_avg

        return df

    def _add_price_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price trend indicators."""

        # Price category within dataset
        price_col = 'precio_venta' if 'precio_venta' in df.columns else 'precio_arriendo'

        df['categoria_precio_mercado'] = pd.qcut(
            df[price_col],
            q=5,
            labels=['muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto'],
            duplicates='drop'
        )

        # Price per m2 category
        if 'area' in df.columns:
            df['precio_m2_categoria'] = pd.qcut(
                df[price_col] / df['area'],
                q=5,
                labels=['economico', 'accesible',
                        'promedio', 'premium', 'lujo'],
                duplicates='drop'
            )

        return df

    def _calculate_min_distance(self, row, gdf: gpd.GeoDataFrame, lat_col: str, lon_col: str) -> float:
        """Calculate minimum distance to features in a GeoDataFrame."""
        try:
            if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
                return np.nan

            point = Point(row[lon_col], row[lat_col])
            distances = gdf.geometry.distance(point)
            return distances.min() * 111000  # Convert to meters (approximate)
        except Exception:
            return np.nan

    def _calculate_accessibility_score(self, distances: pd.Series) -> pd.Series:
        """Calculate accessibility score based on distances."""
        # Score decreases with distance, max score at 0m, min score at 2000m
        return pd.Series(np.maximum(0, 1 - (distances / 2000)), index=distances.index).fillna(0)

    def _add_mobility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mobility and transportation indicators."""

        # Parking availability indicator
        df['facilidad_parqueo'] = df.get('parqueaderos', 0) > 0

        # Transportation score (combination of TransMilenio access and parking)
        transport_score = 0
        if 'score_transmilenio' in df.columns:
            transport_score += df['score_transmilenio'] * 0.7
        if 'facilidad_parqueo' in df.columns:
            transport_score += df['facilidad_parqueo'].astype(int) * 0.3

        df['score_movilidad'] = transport_score

        return df

    def _calculate_convenience_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate convenience score based on POI proximity."""
        convenience_cols = [
            col for col in df.columns if col.startswith('distancia_')]

        if not convenience_cols:
            return pd.Series(0.5, index=df.index)  # Neutral score if no data

        # Convert distances to scores (closer = higher score)
        scores = []
        for col in convenience_cols:
            # Max distance considered: 2000m
            score = pd.Series(np.maximum(0, 1 - (df[col] / 2000)), index=df.index).fillna(0.5)
            scores.append(score)

        # Average of all convenience scores
        return pd.concat(scores, axis=1).mean(axis=1)

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
        features_score = 0
        if 'habitaciones' in df.columns:
            features_score += np.minimum(
                df['habitaciones'].fillna(0) / 4.0, 1.0) * 0.5
        if 'banos' in df.columns:
            features_score += np.minimum(df['banos'].fillna(0) /
                                         3.0, 1.0) * 0.3
        if 'parqueaderos' in df.columns:
            features_score += (df['parqueaderos'].fillna(0)
                               > 0).astype(int) * 0.2

        score += features_score * 0.2

        return score.round(3)

    def _calculate_investment_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate investment potential score."""
        score = pd.Series(0.0, index=df.index)

        # Price efficiency (lower price per m2 in good locations = better investment)
        if 'precio_por_m2' in df.columns and 'estrato' in df.columns:
            # Normalize price per m2 by estrato
            price_efficiency = 1 - (df['precio_por_m2'].rank(pct=True))
            location_quality = df['estrato'].fillna(3) / 6.0
            score += (price_efficiency * location_quality) * 0.4

        # Growth potential (periphery areas with good access)
        if 'zona_ciudad' in df.columns and 'score_movilidad' in df.columns:
            growth_areas = df['zona_ciudad'].isin(['norte', 'periferia'])
            good_access = df['score_movilidad'] > 0.6
            score += (growth_areas & good_access).astype(int) * 0.3

        # Market position
        if 'ratio_promedio_sector' in df.columns:
            # Properties below sector average might have growth potential
            below_average = df['ratio_promedio_sector'] < 0.9
            score += below_average.astype(int) * 0.3

        return score.round(3)


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
    })

    enricher = DataEnricher()
    enriched_data = enricher.enrich(sample_data)

    print("Original columns:", len(sample_data.columns))
    print("Enriched columns:", len(enriched_data.columns))
    print("\nNew enrichment columns:")
    new_cols = [
        col for col in enriched_data.columns if col not in sample_data.columns]
    for col in new_cols:
        print(f"- {col}")
