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
    Includes both original boolean extractions and advanced feature engineering.
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
        df = self._extract_original_boolean_features(df)  # Original extract_features functions
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

    def _extract_original_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract original boolean features from caracteristicas column."""
        
        if 'caracteristicas' not in df.columns:
            print("No 'caracteristicas' column found")
            return df
            
        print("Extracting original boolean features from caracteristicas...")
        
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
        
        # Count total characteristics
        df['numero_caracteristicas'] = self._count_features(df['caracteristicas'])
        # df['numero_caracteristicas'] = df['caracteristicas'].apply(self._count_features)
        
        print("✅ Original boolean features extracted")
        return df

    # ========== ORIGINAL EXTRACT_FEATURES FUNCTIONS (INTEGRATED) ==========
    
    def _check_jacuzzi(self, x):
        """Check if 'JACUZZI' is in the list x."""
        if isinstance(x, list):
            return 1 if 'JACUZZI' in x else 0
        else:
            return 0

    def _extract_piso(self, x):
        """Extracts the floor number from a list of strings."""
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
        """Extracts the number of closets from a list of apartment features."""
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
        """Check if a list contains the string 'CHIMENEA'."""
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
        """Checks if 'GIMNASIO' is in the input list."""
        if isinstance(x, list):
            return 1 if 'GIMNASIO' in x else 0
        else:
            return 0

    def _check_ascensor(self, x):
        """Check if 'ASCENSOR' is in the given list."""
        if isinstance(x, list):
            return 1 if 'ASCENSOR' in x else 0
        else:
            return 0

    def _check_conjunto_cerrado(self, x):
        """Check if a list contains the string 'CONJUNTO CERRADO'."""
        if isinstance(x, list):
            return 1 if 'CONJUNTO CERRADO' in x else 0
        else:
            return 0
        
    def _check_piscina(self, x):
        """Check if a list contains the string 'PISCINA'."""
        if isinstance(x, list):
            return 1 if 'PISCINA' in x else 0
        else:
            return 0

    def _check_salon_comunal(self, x):
        """Check if a list contains the string 'SALÓN COMUNAL'."""
        if isinstance(x, list):
            return 1 if 'SALÓN COMUNAL' in x else 0
        else:
            return 0
        
    def _check_terraza(self, x):
        """Check if a list contains the string 'TERRAZA'."""
        if isinstance(x, list):
            return 1 if 'TERRAZA' in x else 0
        else:
            return 0
        
    def _check_amoblado(self, x):
        """Check if a list contains the string 'AMOBLADO'."""
        if isinstance(x, list):
            return 1 if 'AMOBLADO' in x else 0
        else:
            return 0
        
    def _check_vigilancia(self, x):
        """Check if a list contains the string 'VIGILANCIA'."""
        if isinstance(x, list):
            return 1 if any(re.findall(r'VIGILANCIA', str(x))) else 0
        else:
            return 0

    # ========== ADVANCED FEATURE EXTRACTION METHODS ==========

    def _extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract price-related features."""
    
        # Check which price columns exist
        has_precio_venta = 'precio_venta' in df.columns
        has_precio_arriendo = 'precio_arriendo' in df.columns
    
        # Price per square meter (handle missing columns)
        if has_precio_venta and has_precio_arriendo:
            # Both columns exist
            df['precio_por_m2'] = np.where(
                (df['area'].notna()) & (df['area'] > 0),
                df['precio_venta'].fillna(df['precio_arriendo']) / df['area'],
                np.nan
            )
            price_for_categories = df['precio_venta'].fillna(df['precio_arriendo'])
        
        elif has_precio_venta:
            # Only venta exists
            print("Extracting price per m2 from precio_venta")
            df['precio_por_m2'] = np.where(
                (df['area'].notna()) & (df['area'] > 0),
                df['precio_venta'] / df['area'],
                np.nan
            )
            print("✅ precio_por_m2 extracted from precio_venta")
            price_for_categories = df['precio_venta']
        
        elif has_precio_arriendo:
            # Only arriendo exists
            df['precio_por_m2'] = np.where(
                (df['area'].notna()) & (df['area'] > 0),
                df['precio_arriendo'] / df['area'],
                np.nan
            )
            price_for_categories = df['precio_arriendo']
        
        else:
            # No price columns exist
            print("⚠️  Warning: No price columns found for price feature extraction")
            df['precio_por_m2'] = np.nan
            price_for_categories = pd.Series([np.nan] * len(df))

        # Price categories
        df['categoria_precio'] = pd.cut(
            price_for_categories,
            bins=[0, 200_000_000, 400_000_000, 600_000_000, np.inf],
            labels=['economico', 'medio', 'alto', 'premium'],
            include_lowest=True
        )

        # Administration percentage of price (only works with precio_venta)
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
        df['categoria_antiguedad'] = pd.cut(
            df['antiguedad'],
            bins=[0, 5, 15, 30, np.inf],
            labels=['nuevo', 'reciente', 'usado', 'antiguo'],
            include_lowest=True
        )

        # Property type simplified
        df['tipo_simplificado'] = df['tipo_propiedad'].apply(self._simplify_property_type)

        return df

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from text descriptions."""

        # Description length
        if 'descripcion' in df.columns:
            df['longitud_descripcion'] = df['descripcion'].fillna('').str.len()
            # Has description
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
            df['longitud_descripcion'] = 0
            df['tiene_descripcion'] = False
            df['indica_lujo'] = False
            df['indica_seguridad'] = False

        # Number of images - Handle missing column
        if 'imagenes' in df.columns:
            df['numero_imagenes'] = df['imagenes'].apply(self._count_images)
        else:
            df['numero_imagenes'] = 0

        return df

    def _extract_amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional amenity-related features (beyond the original boolean ones)."""
        
        # Additional amenity analysis (beyond the boolean ones already extracted)
        if 'caracteristicas' in df.columns:
            # Count different types of amenities
            df['tiene_zona_social'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['zona social', 'salon comunal', 'bbq'])
            )
            df['tiene_zona_deportiva'] = df['caracteristicas'].apply(
                lambda x: self._has_amenity(x, ['cancha', 'tenis', 'squash', 'futbol'])
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

    # ========== HELPER METHODS ==========

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
                import ast
                images = ast.literal_eval(images)
            except:
                return 1 if len(images) > 0 else 0
        if isinstance(images, list):
            return len(images)
        return 0

    def _count_features(self, features) -> int:
        """Count number of features/characteristics."""
        if isinstance(features, pd.Series):
            features = features.tolist()
        if not features:
            print("No features found or NaN")
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
        if 'precio_por_m2' in df.columns and df['precio_por_m2'].notna().any():
            try:
                price_score = 1 - (df['precio_por_m2'].rank(pct=True, na_option='bottom'))
                score += price_score.fillna(0) * 0.4
            except Exception as e:
                print(f"Warning: Could not calculate price score: {e}")

        # Location component
        location_scores = {'premium': 1.0, 'high': 0.8, 'medium': 0.6, 'popular': 0.4, 'other': 0.3}
        if 'categoria_sector' in df.columns:
            score += df['categoria_sector'].map(location_scores).fillna(0.3) * 0.3

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

        available_fields = [col for col in important_fields if col in df.columns]
        score = df[available_fields].notna().sum(axis=1) / len(available_fields)

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

        # High price per m2 - FIX: Check if column has valid data first
        if 'precio_por_m2' in df.columns and df['precio_por_m2'].notna().any():
            try:
                price_quantile = df['precio_por_m2'].quantile(0.8)
                if pd.notna(price_quantile):
                    high_price = (df['precio_por_m2'] > price_quantile).fillna(False)
                    score += high_price.astype(int) * 0.2
            except Exception as e:
                print(f"Warning: Could not calculate price quantile: {e}")

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

        # Parking - FIX: Check if column exists first
        if 'parqueaderos' in df.columns:
            df['tiene_parqueadero'] = (df['parqueaderos'].fillna(0) > 0)
            score += df['tiene_parqueadero'].astype(int) * 0.2
        elif 'tiene_parqueadero' in df.columns:
            score += df['tiene_parqueadero'].astype(int) * 0.2

        # Good area - FIX: Check if column has valid data
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
    print("\nNew features:")
    new_features = [
        col for col in result.columns if col not in sample_data.columns]
    for feature in new_features:
        print(f"- {feature}")
