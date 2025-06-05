import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
from unidecode import unidecode
import logging


class FeatureExtractor:
    """
    Extract property-level features from cleaned apartment data.
    Focuses on: amenities, property characteristics, basic metrics, and property-specific scoring.
    
    Note: Market analysis and geographic features are handled by GeoDataEnricher.
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

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract property-level features from the cleaned data.
        """
        self.logger.info(f"Extracting property features from {len(data)} records...")

        # Make a copy to avoid modifying original data
        df = data.copy()

        # Extract different types of features
        df = self._extract_original_boolean_features(df)  # Original boolean amenities
        df = self._extract_basic_price_features(df)       # Basic price metrics only
        df = self._extract_area_features(df)              # Area-related features
        df = self._extract_location_features(df)          # Basic location features
        df = self._extract_property_features(df)          # Property characteristics
        df = self._extract_text_features(df)              # Text analysis
        df = self._extract_amenity_features(df)           # Additional amenities
        df = self._extract_property_scores(df)            # Property-specific scores only

        self.logger.info(
            f"Property feature extraction completed. Added {len(df.columns) - len(data.columns)} new features")
        return df

    def _extract_original_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract original boolean features from caracteristicas column."""
        
        if 'caracteristicas' not in df.columns:
            print("No 'caracteristicas' column found")
            return df
            
        print("Extracting boolean features from caracteristicas...")
        
        # Apply all original extraction functions
        df['jacuzzi'] = df['caracteristicas'].apply(self._check_jacuzzi)
        df['piso'] = df['caracteristicas'].apply(self._extract_piso)
        df['closets'] = df['caracteristicas'].apply(self._extract_closets)
        df['chimenea'] = df['caracteristicas'].apply(self._check_chimeny)
        df['permite_mascotas'] = df['caracteristicas'].apply(self._check_mascotas)
        df['gimnasio'] = df['caracteristicas'].apply(self._check_gimnasio)
        df['ascensor'] = df['caracteristicas'].apply(self._check_ascensor)
        df['conjunto_cerrado'] = df['caracteristicas'].apply(self._check_conjunto_cerrado)
        df['piscina'] = df['caracteristicas'].apply(self._check_piscina)
        df['salon_comunal'] = df['caracteristicas'].apply(self._check_salon_comunal)
        df['terraza'] = df['caracteristicas'].apply(self._check_terraza)
        df['amoblado'] = df['caracteristicas'].apply(self._check_amoblado)
        df['vigilancia'] = df['caracteristicas'].apply(self._check_vigilancia)

        return df

    def _extract_basic_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic price-related features (no market analysis)."""
    
        # Check which price columns exist
        has_precio_venta = 'precio_venta' in df.columns
        has_precio_arriendo = 'precio_arriendo' in df.columns
    
        # Basic price per square meter calculation
        if has_precio_venta and has_precio_arriendo:
            df['precio_por_m2'] = np.where(
                (df['area'].notna()) & (df['area'] > 0),
                df['precio_venta'].fillna(df['precio_arriendo']) / df['area'],
                np.nan
            )
            price_for_categories = df['precio_venta'].fillna(df['precio_arriendo'])
        
        elif has_precio_venta:
            df['precio_por_m2'] = np.where(
                (df['area'].notna()) & (df['area'] > 0),
                df['precio_venta'] / df['area'],
                np.nan
            )
            price_for_categories = df['precio_venta']
        
        elif has_precio_arriendo:
            df['precio_por_m2'] = np.where(
                (df['area'].notna()) & (df['area'] > 0),
                df['precio_arriendo'] / df['area'],
                np.nan
            )
            price_for_categories = df['precio_arriendo']
        
        else:
            print("⚠️  Warning: No price columns found for price feature extraction")
            df['precio_por_m2'] = np.nan
            price_for_categories = pd.Series([np.nan] * len(df))

        # Basic price categories (property-level, not market-based)
        df['categoria_precio'] = pd.cut(
            price_for_categories,
            bins=[0, 200_000_000, 400_000_000, 600_000_000, np.inf],
            labels=['economico', 'medio', 'alto', 'premium'],
            include_lowest=True
        )

        # Administration percentage (basic property metric)
        if has_precio_venta and 'administracion' in df.columns:
            df['admin_percentage'] = np.where(
                df['precio_venta'].notna() & (df['precio_venta'] > 0),
                (df['administracion'].fillna(0) / df['precio_venta']) * 100,
                np.nan
            )
        else:
            df['admin_percentage'] = np.nan

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
            df['area'] / (df['habitaciones'].fillna(0) + df['banos'].fillna(0)),
            np.nan
        )

        return df

    def _extract_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic location features (no geographic analysis)."""

        # Estrato categories
        df['categoria_estrato'] = pd.cut(
            df['estrato'].astype(float, errors='ignore'),
            bins=[0, 2, 3, 4, 6],
            labels=['bajo', 'medio_bajo', 'medio', 'alto'],
            include_lowest=True
        )

        # Has coordinates
        df['tiene_coordenadas'] = (df['latitud'].notna()) & (df['longitud'].notna())

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
        if 'antiguedad' in df.columns:
            df['categoria_antiguedad'] = pd.cut(
                df['antiguedad'],
                bins=[0, 5, 15, 30, np.inf],
                labels=['nuevo', 'reciente', 'usado', 'antiguo'],
                include_lowest=True
            )

        # Property type simplified
        if 'tipo_propiedad' in df.columns:
            df['tipo_simplificado'] = df['tipo_propiedad'].apply(self._simplify_property_type)

        return df

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from text descriptions."""

        # Description analysis
        if 'descripcion' in df.columns:
            df['tiene_descripcion'] = df['descripcion'].notna() & (df['descripcion'].str.len() > 10)
            
            # Luxury indicators from description
            df['indica_lujo'] = df['descripcion'].fillna('').apply(
                lambda x: self._contains_keywords(x, self.luxury_keywords)
            )

            # Security indicators
            df['indica_seguridad'] = df['descripcion'].fillna('').apply(
                lambda x: self._contains_keywords(x, self.security_keywords)
            )
        else:
            df['tiene_descripcion'] = False
            df['indica_lujo'] = False
            df['indica_seguridad'] = False

        return df

    def _extract_amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional amenity-related features."""
        
        if 'caracteristicas' in df.columns:
            # Group amenities by type
            df['tiene_zona_social'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['zona social', 'salon comunal', 'bbq'])
            )
            df['tiene_zona_deportiva'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['cancha', 'tenis', 'squash', 'futbol'])
            )
            
        return df

    def _extract_property_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract property-specific scores (not market-based)."""

        # Data completeness score
        df['score_completitud'] = self._calculate_completeness_score(df)

        # Luxury score (property-based)
        df['score_lujo'] = self._calculate_luxury_score(df)

        # Family friendly score (property-based)
        df['score_familiar'] = self._calculate_family_score(df)

        # NOTE: Value score removed (market-based, handled by GeoDataEnricher)

        return df

    # ========== ORIGINAL BOOLEAN FEATURE FUNCTIONS (UNCHANGED) ==========
    
    def _check_jacuzzi(self, x):
        if isinstance(x, list):
            return 1 if 'JACUZZI' in x else 0
        else:
            return 0

    def _extract_piso(self, x):
        if isinstance(x, list):
            try:
                for item in x:
                    if item.startswith('PISO '):
                        return int(item.split(' ')[1])
            except:
                return np.nan
        else:
            return np.nan

    def _extract_closets(self, x):
        if isinstance(x, list):
            try:
                for item in x:
                    if item.startswith('CLOSETS'):
                        return int(item.split(' ')[1])
            except:
                return np.nan
        else:
            return np.nan

    def _check_chimeny(self, x):
        if isinstance(x, list):
            return 1 if 'CHIMENEA' in x else 0
        else:
            return 0

    def _check_mascotas(self, x):
        if isinstance(x, list):
            if 'PERMITE MASCOTAS' in x:
                return 1
            elif 'ADMITE MASCOTAS' in x:
                return 1
            else:
                return 0
        else:
            return 0

    def _check_gimnasio(self, x):
        if isinstance(x, list):
            return 1 if 'GIMNASIO' in x else 0
        else:
            return 0

    def _check_ascensor(self, x):
        if isinstance(x, list):
            return 1 if 'ASCENSOR' in x else 0
        else:
            return 0

    def _check_conjunto_cerrado(self, x):
        if isinstance(x, list):
            return 1 if 'CONJUNTO CERRADO' in x else 0
        else:
            return 0
        
    def _check_piscina(self, x):
        if isinstance(x, list):
            return 1 if 'PISCINA' in x else 0
        else:
            return 0

    def _check_salon_comunal(self, x):
        if isinstance(x, list):
            return 1 if 'SALÓN COMUNAL' in x else 0
        else:
            return 0
        
    def _check_terraza(self, x):
        if isinstance(x, list):
            return 1 if 'TERRAZA' in x else 0
        else:
            return 0
        
    def _check_amoblado(self, x):
        if isinstance(x, list):
            return 1 if 'AMOBLADO' in x else 0
        else:
            return 0
        
    def _check_vigilancia(self, x):
        if isinstance(x, list):
            return 1 if any(re.findall(r'VIGILANCIA', str(x))) else 0
        else:
            return 0

    # ========== HELPER METHODS (UNCHANGED) ==========

    def _has_amenity(self, features, keywords: List[str]) -> bool:
        if features is None or not features:
            return False
        
        if isinstance(features, list):
            if not features:
                return False
            features_str = ' '.join(str(item).lower() for item in features)
        else:
            if pd.isna(features):
                return False
            features_str = str(features).lower()
        
        return any(keyword.lower() in features_str for keyword in keywords)



    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        if text is None:
            return False
        
        if not isinstance(text, list) and pd.isna(text):
            return False

        text_clean = unidecode(str(text).lower())
        return any(keyword in text_clean for keyword in keywords)

    def _simplify_property_type(self, tipo: str) -> str:
        if tipo is None:
            return 'unknown'
        
        if not isinstance(tipo, list) and pd.isna(tipo):
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

    def _calculate_completeness_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate completeness score based on available information."""
        important_fields = [
            'precio_venta', 'precio_arriendo', 'area', 'habitaciones', 'banos',
            'sector', 'estrato', 'descripcion', 'imagenes'
        ]

        available_fields = [col for col in important_fields if col in df.columns]
        score = df[available_fields].notna().sum(axis=1) / len(available_fields)

        return score.round(3)

    def _calculate_luxury_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate property-based luxury score."""
        score = pd.Series(0.0, index=df.index)

        # High estrato (property characteristic)
        if 'estrato' in df.columns:
            score += (df['estrato'].fillna(0) >= 5).astype(int) * 0.4

        # High price per m2 (property metric)
        if 'precio_por_m2' in df.columns and df['precio_por_m2'].notna().any():
            try:
                price_quantile = df['precio_por_m2'].quantile(0.8)
                if pd.notna(price_quantile):
                    high_price = (df['precio_por_m2'] > price_quantile).fillna(False)
                    score += high_price.astype(int) * 0.3
            except Exception as e:
                print(f"Warning: Could not calculate price quantile: {e}")

        # Luxury indicators in description (property feature)
        if 'indica_lujo' in df.columns:
            score += df['indica_lujo'].astype(int) * 0.3

        return score.round(3)

    def _calculate_family_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate family-friendly score based on property features."""
        score = pd.Series(0.0, index=df.index)

        # Multiple bedrooms
        if 'habitaciones' in df.columns:
            score += np.minimum(df['habitaciones'].fillna(0) / 4.0, 1.0) * 0.3

        # Multiple bathrooms
        if 'banos' in df.columns:
            score += np.minimum(df['banos'].fillna(0) / 3.0, 1.0) * 0.2

        # Parking availability
        if 'parqueaderos' in df.columns:
            score += (df['parqueaderos'].fillna(0) > 0).astype(int) * 0.2
        elif 'tiene_parqueadero' in df.columns:
            score += df['tiene_parqueadero'].astype(int) * 0.2

        # Good area for families
        if 'area' in df.columns and df['area'].notna().any():
            good_area = (df['area'] >= 80).fillna(False)
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
    print("\nProperty features extracted:")
    new_features = [col for col in result.columns if col not in sample_data.columns]
    for feature in new_features:
        print(f"- {feature}")
