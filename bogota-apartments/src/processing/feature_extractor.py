import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
from unidecode import unidecode
import logging
from pathlib import Path


class FeatureExtractor:
    """
    Extract meaningful features from cleaned apartment data.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define feature mappings
        self.luxury_keywords = [
            'lujo', 'luxury', 'premium', 'exclusivo', 'exclusive',
            'penthouse', 'duplex', 'jacuzzi', 'terraza', 'balcon',
            'vista', 'panoramica', 'club house', 'gimnasio', 'piscina'
        ]

        self.security_keywords = [
            'porteria', 'vigilancia', 'seguridad', 'citofono', 'alarma',
            'camara', 'control acceso', 'cerrado'
        ]

        # Sector classification (you can expand this based on your knowledge of Bogotá)
        self.sector_categories = {
            'premium': ['chapinero', 'zona rosa', 'rosales', 'chicó', 'usaquén'],
            'high': ['cedritos', 'country', 'santa barbara', 'la carolina'],
            'medium': ['suba', 'engativá', 'fontibón', 'teusaquillo'],
            'popular': ['kennedy', 'bosa', 'san cristóbal', 'rafael uribe']
        }

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from the cleaned data.
        """
        self.logger.info(f"Extracting features from {len(data)} records...")

        # Make a copy to avoid modifying original data
        df = data.copy()

        # Extract different types of features
        df = self._extract_price_features(df)
        df = self._extract_area_features(df)
        df = self._extract_location_features(df)
        df = self._extract_property_features(df)
        df = self._extract_text_features(df)
        df = self._extract_amenity_features(df)
        df = self._extract_derived_features(df)

        self.logger.info(
            f"Feature extraction completed. Added {len(df.columns) - len(data.columns)} new features")
        return df

    def _extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract price-related features."""

        # Price per square meter
        df['precio_por_m2'] = np.where(
            (df['area'].notna()) & (df['area'] > 0),
            df['precio_venta'].fillna(df['precio_arriendo']) / df['area'],
            np.nan
        )

        # Price categories
        df['categoria_precio'] = pd.cut(
            df['precio_venta'].fillna(df['precio_arriendo']),
            bins=[0, 200_000_000, 400_000_000, 600_000_000, np.inf],
            labels=['economico', 'medio', 'alto', 'premium'],
            include_lowest=True
        )

        # Administration percentage of price
        df['admin_percentage'] = np.where(
            df['precio_venta'].notna() & (df['precio_venta'] > 0),
            (df['administracion'].fillna(0) / df['precio_venta']) * 100,
            np.nan
        )

        return df

    def _extract_area_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract area-related features."""

        # Area categories
        df['categoria_area'] = pd.cut(
            df['area'],
            bins=[0, 50, 80, 120, 200, np.inf],
            labels=['muy_pequeno', 'pequeno', 'medio', 'grande', 'muy_grande'],
            include_lowest=True
        )

        # Area per room
        df['area_por_habitacion'] = np.where(
            (df['habitaciones'].notna()) & (df['habitaciones'] > 0),
            df['area'] / df['habitaciones'],
            np.nan
        )

        # Area efficiency (area vs number of rooms + bathrooms)
        df['eficiencia_espacial'] = np.where(
            (df['habitaciones'].fillna(0) + df['banos'].fillna(0)) > 0,
            df['area'] / (df['habitaciones'].fillna(0) +
                          df['banos'].fillna(0)),
            np.nan
        )

        return df

    def _extract_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract location-related features."""

        # Sector category based on neighborhood
        df['categoria_sector'] = df['sector'].apply(self._categorize_sector)

        # Estrato categories
        df['categoria_estrato'] = pd.cut(
            df['estrato'].astype(float, errors='ignore'),
            bins=[0, 2, 3, 4, 6],
            labels=['bajo', 'medio_bajo', 'medio', 'alto'],
            include_lowest=True
        )

        # Has coordinates
        df['tiene_coordenadas'] = (
            df['latitud'].notna()) & (df['longitud'].notna())

        return df

    def _extract_property_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract property-specific features."""

        # Room to bathroom ratio
        df['ratio_habitaciones_banos'] = np.where(
            (df['banos'].notna()) & (df['banos'] > 0),
            df['habitaciones'].fillna(0) / df['banos'],
            np.nan
        )

        # Has parking
        df['tiene_parqueadero'] = (df['parqueaderos'].fillna(0) > 0)

        # Property age categories
        df['categoria_antiguedad'] = pd.cut(
            df['antiguedad'],
            bins=[0, 5, 15, 30, np.inf],
            labels=['nuevo', 'reciente', 'usado', 'antiguo'],
            include_lowest=True
        )

        # Property type simplified
        df['tipo_simplificado'] = df['tipo_propiedad'].apply(
            self._simplify_property_type)

        return df

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from text descriptions."""

        # Description length
        df['longitud_descripcion'] = df['descripcion'].fillna('').str.len()

        # Has description
        df['tiene_descripcion'] = df['descripcion'].notna() & (
            df['descripcion'].str.len() > 10)

        # Luxury indicators from description
        df['indica_lujo'] = df['descripcion'].fillna('').apply(
            lambda x: self._contains_keywords(x, self.luxury_keywords)
        )

        # Security indicators
        df['indica_seguridad'] = df['descripcion'].fillna('').apply(
            lambda x: self._contains_keywords(x, self.security_keywords)
        )

        # Number of images
        df['numero_imagenes'] = df['imagenes'].apply(self._count_images)

        return df

    def _extract_amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract amenity-related features."""

        # Extract features from characteristics if available
        if 'caracteristicas' in df.columns:
            df['numero_caracteristicas'] = df['caracteristicas'].apply(
                self._count_features)

            # Specific amenity features
            df['tiene_piscina'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['piscina', 'pool'])
            )
            df['tiene_gimnasio'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['gimnasio', 'gym', 'fitness'])
            )
            df['tiene_salon_social'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['salon', 'social', 'eventos'])
            )

        return df

    def _extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract derived and composite features."""

        # Value score (combination of price per m2 and location)
        df['score_valor'] = self._calculate_value_score(df)

        # Completeness score (how much information is available)
        df['score_completitud'] = self._calculate_completeness_score(df)

        # Luxury score
        df['score_lujo'] = self._calculate_luxury_score(df)

        # Family friendly score
        df['score_familiar'] = self._calculate_family_score(df)

        return df

    def _categorize_sector(self, sector: str) -> str:
        """Categorize sector based on known classifications."""
        if pd.isna(sector):
            return 'unknown'

        sector_clean = unidecode(str(sector).lower())

        for category, sectors in self.sector_categories.items():
            if any(s in sector_clean for s in sectors):
                return category

        return 'other'

    def _simplify_property_type(self, tipo: str) -> str:
        """Simplify property type to main categories."""
        if pd.isna(tipo):
            return 'unknown'

        tipo_clean = unidecode(str(tipo).lower())

        if 'apartamento' in tipo_clean or 'apto' in tipo_clean:
            return 'apartamento'
        elif 'casa' in tipo_clean:
            return 'casa'
        elif 'estudio' in tipo_clean or 'loft' in tipo_clean:
            return 'estudio'
        elif 'penthouse' in tipo_clean:
            return 'penthouse'
        else:
            return 'otro'

    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        if pd.isna(text):
            return False

        text_clean = unidecode(str(text).lower())
        return any(keyword in text_clean for keyword in keywords)

    def _count_images(self, images) -> int:
        """Count number of images."""
        if pd.isna(images):
            return 0
        if isinstance(images, str):
            try:
                # If it's a string representation of a list
                import ast
                images = ast.literal_eval(images)
            except:
                return 1 if len(images) > 0 else 0
        if isinstance(images, list):
            return len(images)
        return 0

    def _count_features(self, features) -> int:
        """Count number of features/characteristics."""
        if pd.isna(features):
            return 0
        if isinstance(features, str):
            try:
                import ast
                features = ast.literal_eval(features)
            except:
                return len(features.split(',')) if ',' in features else 1
        if isinstance(features, list):
            return len(features)
        return 0

    def _has_amenity(self, features, keywords: List[str]) -> bool:
        """Check if property has specific amenity."""
        if pd.isna(features):
            return False

        features_str = str(features).lower()
        return any(keyword in features_str for keyword in keywords)

    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a value score based on price and location."""
        score = pd.Series(0.0, index=df.index)

        # Price component (inverse - lower price per m2 is better value)
        if 'precio_por_m2' in df.columns:
            price_score = 1 - \
                (df['precio_por_m2'].rank(pct=True, na_option='bottom'))
            score += price_score * 0.4

        # Location component
        location_scores = {'premium': 1.0, 'high': 0.8,
                           'medium': 0.6, 'popular': 0.4, 'other': 0.3}
        if 'categoria_sector' in df.columns:
            score += df['categoria_sector'].map(
                location_scores).fillna(0.3) * 0.3

        # Estrato component
        if 'estrato' in df.columns:
            estrato_score = df['estrato'].fillna(2) / 6.0
            score += estrato_score * 0.3

        return score.round(3)

    def _calculate_completeness_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate completeness score based on available information."""
        important_fields = [
            'precio_venta', 'precio_arriendo', 'area', 'habitaciones', 'banos',
            'sector', 'estrato', 'descripcion', 'imagenes'
        ]

        available_fields = [
            col for col in important_fields if col in df.columns]
        score = df[available_fields].notna().sum(axis=1) / \
            len(available_fields)

        return score.round(3)

    def _calculate_luxury_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate luxury score based on various indicators."""
        score = pd.Series(0.0, index=df.index)

        # High estrato
        if 'estrato' in df.columns:
            score += (df['estrato'].fillna(0) >= 5).astype(int) * 0.3

        # Premium sector
        if 'categoria_sector' in df.columns:
            score += (df['categoria_sector'] == 'premium').astype(int) * 0.3

        # High price per m2
        if 'precio_por_m2' in df.columns:
            high_price = df['precio_por_m2'] > df['precio_por_m2'].quantile(
                0.8)
            score += high_price.astype(int) * 0.2

        # Luxury indicators in description
        if 'indica_lujo' in df.columns:
            score += df['indica_lujo'].astype(int) * 0.2

        return score.round(3)

    def _calculate_family_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate family-friendly score."""
        score = pd.Series(0.0, index=df.index)

        # Multiple bedrooms
        if 'habitaciones' in df.columns:
            score += np.minimum(df['habitaciones'].fillna(0) / 4.0, 1.0) * 0.3

        # Multiple bathrooms
        if 'banos' in df.columns:
            score += np.minimum(df['banos'].fillna(0) / 3.0, 1.0) * 0.2

        # Parking
        if 'tiene_parqueadero' in df.columns:
            score += df['tiene_parqueadero'].astype(int) * 0.2

        # Good area
        if 'area' in df.columns:
            good_area = df['area'] >= 80
            score += good_area.astype(int) * 0.3

        return score.round(3)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'precio_venta': [300_000_000, 500_000_000, None],
        'precio_arriendo': [None, None, 2_000_000],
        'area': [70, 120, 45],
        'habitaciones': [2, 3, 1],
        'banos': [2, 2, 1],
        'sector': ['Chapinero', 'Kennedy', 'Chicó'],
        'estrato': [4, 2, 6],
        'descripcion': ['Apartamento de lujo con piscina', 'Cómodo apartamento', 'Estudio moderno'],
        'tipo_propiedad': ['Apartamento', 'Apartamento', 'Estudio']
    })

    extractor = FeatureExtractor()
    result = extractor.extract_features(sample_data)
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"Final columns: {len(result.columns)}")
    print("\nNew features:")
    new_features = [
        col for col in result.columns if col not in sample_data.columns]
    for feature in new_features:
        print(f"- {feature}")
