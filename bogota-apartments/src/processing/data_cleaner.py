import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from unidecode import unidecode
import logging
from datetime import datetime


class DataCleaner:
    """
    Clean and standardize apartment data from multiple sources.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define cleaning rules and mappings
        self.property_type_mapping = {
            'apartamento': ['apartamento', 'apto', 'apartment'],
            'casa': ['casa', 'house', 'vivienda'],
            'estudio': ['estudio', 'studio', 'loft'],
            'penthouse': ['penthouse', 'ph', 'ático'],
            'oficina': ['oficina', 'office'],
            'local': ['local', 'comercial']
        }

        self.operation_type_mapping = {
            'venta': ['venta', 'sale', 'compra', 'buy'],
            'arriendo': ['arriendo', 'alquiler', 'rent', 'rental']
        }

        # Price thresholds for outlier detection (in COP)
        self.price_thresholds = {
            'min_sale': 50_000_000,      # 50M COP minimum
            'max_sale': 10_000_000_000,  # 10B COP maximum
            'min_rent': 500_000,         # 500K COP minimum
            'max_rent': 50_000_000       # 50M COP maximum
        }

        # Area thresholds (in m²)
        self.area_thresholds = {
            'min_area': 20,    # 20 m² minimum
            'max_area': 1000   # 1000 m² maximum
        }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning method that applies all cleaning steps.
        """
        self.logger.info(f"Starting data cleaning for {len(df)} records...")

        # Make a copy to avoid modifying original data
        df_clean = df.copy()

        # Apply cleaning steps in order
        df_clean = self._standardize_columns(df_clean)
        df_clean = self._clean_text_fields(df_clean)
        df_clean = self._clean_numeric_fields(df_clean)
        df_clean = self._clean_price_fields(df_clean)
        df_clean = self._clean_location_fields(df_clean)
        df_clean = self._clean_property_features(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._remove_outliers(df_clean)
        df_clean = self._add_data_quality_flags(df_clean)

        # Log cleaning results
        initial_count = len(df)
        final_count = len(df_clean)
        self.logger.info(
            f"Data cleaning completed: {initial_count} → {final_count} records ({final_count/initial_count*100:.1f}% retained)")

        return df_clean

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and ensure required columns exist."""

        # Standardize column names (lowercase, no spaces)
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Ensure required columns exist
        required_columns = [
            'codigo', 'tipo_propiedad', 'tipo_operacion', 'precio_venta', 'precio_arriendo',
            'area', 'habitaciones', 'banos', 'sector', 'estrato', 'website'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                self.logger.warning(f"Added missing column: {col}")

        return df

    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields."""

        text_columns = ['sector', 'direccion', 'descripcion',
                        'tipo_propiedad', 'tipo_operacion']

        for col in text_columns:
            if col in df.columns:
                # Remove extra whitespace and normalize
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], np.nan)

                # Normalize encoding
                df[col] = df[col].apply(
                    lambda x: unidecode(x) if pd.notna(x) else x)

                # Convert to lowercase for standardization
                if col in ['tipo_propiedad', 'tipo_operacion']:
                    df[col] = df[col].str.lower()

        # Standardize property and operation types
        df = self._standardize_categorical_fields(df)

        return df

    def _standardize_categorical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical fields using mappings."""

        # Standardize property types
        if 'tipo_propiedad' in df.columns:
            df['tipo_propiedad'] = df['tipo_propiedad'].apply(
                lambda x: self._map_category(x, self.property_type_mapping)
            )

        # Standardize operation types
        if 'tipo_operacion' in df.columns:
            df['tipo_operacion'] = df['tipo_operacion'].apply(
                lambda x: self._map_category(x, self.operation_type_mapping)
            )

        return df

    def _map_category(self, value: str, mapping: Dict[str, List[str]]) -> str | None:
        """Map a value to standardized category."""
        if pd.isna(value):
            return None

        value_clean = str(value).lower().strip()

        for standard_name, variations in mapping.items():
            if any(var in value_clean for var in variations):
                return standard_name

        return value_clean  # Return original if no mapping found

    def _clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert numeric fields."""

        numeric_columns = ['area', 'habitaciones', 'banos',
                           'parqueaderos', 'estrato', 'antiguedad']

        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, handling errors
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Apply field-specific cleaning
                if col == 'estrato':
                    # Estrato should be between 1 and 6
                    df[col] = df[col].where((df[col] >= 1) & (df[col] <= 6))

                elif col in ['habitaciones', 'banos']:
                    # Rooms should be positive and reasonable
                    df[col] = df[col].where((df[col] >= 0) & (df[col] <= 20))

                elif col == 'parqueaderos':
                    # Parking spots should be reasonable
                    df[col] = df[col].where((df[col] >= 0) & (df[col] <= 10))

                elif col == 'antiguedad':
                    # Age should be reasonable (0-100 years)
                    df[col] = df[col].where((df[col] >= 0) & (df[col] <= 100))

        return df

    def _clean_price_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price fields and handle currency issues."""

        price_columns = ['precio_venta', 'precio_arriendo', 'administracion']

        for col in price_columns:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Remove negative prices
                df[col] = df[col].where(df[col] >= 0)

                # Apply price-specific thresholds
                if col == 'precio_venta':
                    df[col] = df[col].where(
                        (df[col] >= self.price_thresholds['min_sale']) &
                        (df[col] <= self.price_thresholds['max_sale'])
                    )
                elif col == 'precio_arriendo':
                    df[col] = df[col].where(
                        (df[col] >= self.price_thresholds['min_rent']) &
                        (df[col] <= self.price_thresholds['max_rent'])
                    )

        return df

    def _clean_location_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean location-related fields."""

        # Clean coordinates
        for coord_col in ['latitud', 'longitud']:
            if coord_col in df.columns:
                df[coord_col] = pd.to_numeric(df[coord_col], errors='coerce')

                # Bogotá approximate bounds
                if coord_col == 'latitud':
                    df[coord_col] = df[coord_col].where(
                        (df[coord_col] >= 4.0) & (df[coord_col] <= 5.0)
                    )
                elif coord_col == 'longitud':
                    df[coord_col] = df[coord_col].where(
                        (df[coord_col] >= -75.0) & (df[coord_col] <= -73.0)
                    )

        # Clean sector names
        if 'sector' in df.columns:
            df['sector'] = df['sector'].str.title()  # Proper case
            df['sector'] = df['sector'].str.replace(
                r'\s+', ' ', regex=True)  # Single spaces

        return df

    def _clean_property_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean property feature fields."""

        # Clean area field
        if 'area' in df.columns:
            df['area'] = pd.to_numeric(df['area'], errors='coerce')
            df['area'] = df['area'].where(
                (df['area'] >= self.area_thresholds['min_area']) &
                (df['area'] <= self.area_thresholds['max_area'])
            )

        # Clean image lists
        if 'imagenes' in df.columns:
            df['imagenes'] = df['imagenes'].apply(self._clean_image_list)

        # Clean characteristics lists
        if 'caracteristicas' in df.columns:
            df['caracteristicas'] = df['caracteristicas'].apply(
                self._clean_feature_list)

        return df

    def _clean_image_list(self, images) -> List[str]:
        """Clean and validate image URLs."""
        if pd.isna(images):
            return []

        if isinstance(images, str):
            try:
                import ast
                images = ast.literal_eval(images)
            except:
                return [images] if images.startswith('http') else []

        if isinstance(images, list):
            # Filter valid URLs
            valid_images = [
                img for img in images
                if isinstance(img, str) and img.startswith(('http', 'https'))
            ]
            return valid_images

        return []

    def _clean_feature_list(self, features) -> List[str]:
        """Clean and standardize feature lists."""
        if pd.isna(features):
            return []

        if isinstance(features, str):
            try:
                import ast
                features = ast.literal_eval(features)
            except:
                # Split by common delimiters
                features = re.split(r'[,;|]', features)

        if isinstance(features, list):
            # Clean each feature
            cleaned = [
                unidecode(str(feature).strip().lower())
                for feature in features
                if str(feature).strip()
            ]
            # Remove very short features
            return [f for f in cleaned if len(f) > 2]

        return []

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""

        initial_count = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()

        # Remove duplicates based on key fields
        if 'codigo' in df.columns:
            # Remove duplicates by property code
            df = df.drop_duplicates(subset=['codigo'], keep='first')
        else:
            # Remove duplicates by address + price + area
            key_columns = ['direccion', 'precio_venta',
                           'precio_arriendo', 'area']
            available_keys = [col for col in key_columns if col in df.columns]
            if available_keys:
                df = df.drop_duplicates(subset=available_keys, keep='first')

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate records")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers."""

        initial_count = len(df)

        # Remove price outliers using IQR method
        for price_col in ['precio_venta', 'precio_arriendo']:
            if price_col in df.columns:
                df = self._remove_iqr_outliers(df, price_col)

        # Remove area outliers
        if 'area' in df.columns:
            df = self._remove_iqr_outliers(df, 'area')

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outlier records")

        return df

    def _remove_iqr_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using IQR method."""

        if column not in df.columns:
            return df

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def _add_data_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flags indicating data quality."""

        # Completeness score
        important_fields = ['precio_venta', 'precio_arriendo',
                            'area', 'habitaciones', 'banos', 'sector']
        available_fields = [
            col for col in important_fields if col in df.columns]
        df['completeness_score'] = df[available_fields].notna().sum(axis=1) / \
            len(available_fields)

        # Has price flag
        df['has_price'] = (df['precio_venta'].notna()) | (
            df['precio_arriendo'].notna())

        # Has location flag
        df['has_location'] = (df['sector'].notna()) | (
            (df['latitud'].notna()) & (df['longitud'].notna()))

        # Data quality flag
        df['high_quality'] = (
            (df['completeness_score'] >= 0.7) &
            df['has_price'] &
            df['has_location'] &
            (df['area'].notna()) &
            (df['habitaciones'].notna())
        )

        return df

    def get_cleaning_report(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict:
        """Generate a report of the cleaning process."""

        report = {
            'original_count': len(df_original),
            'final_count': len(df_cleaned),
            'retention_rate': len(df_cleaned) / len(df_original) * 100,
            'completeness_distribution': df_cleaned['completeness_score'].describe().to_dict(),
            'high_quality_records': df_cleaned['high_quality'].sum(),
            'records_with_price': df_cleaned['has_price'].sum(),
            'records_with_location': df_cleaned['has_location'].sum(),
        }

        return report


# Example usage and testing
if __name__ == "__main__":
    # Create sample dirty data for testing
    sample_data = pd.DataFrame({
        'codigo': ['A001', 'A002', 'A001', None, 'A003'],  # Duplicate
        'tipo_propiedad': ['Apartamento', 'CASA', 'apto', 'house', 'Estudio'],
        'tipo_operacion': ['venta', 'ARRIENDO', 'sale', 'rent', 'venta'],
        # Low outlier
        'precio_venta': [300_000_000, None, 300_000_000, 1_000_000, 500_000_000],
        'precio_arriendo': [None, 2_000_000, None, None, None],
        'area': [70, 120, 70, 10, 85],  # Low outlier
        'habitaciones': [2, 3, 2, 1, 2],
        'banos': [2, 2, 2, 1, 2],
        'sector': ['  chapinero  ', 'Kennedy', 'chapinero', 'zona rosa', 'Suba'],
        'estrato': [4, 2, 4, 8, 3],  # High outlier
        'website': ['habi.co', 'metrocuadrado.com', 'habi.co', 'habi.co', 'metrocuadrado.com']
    })

    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(sample_data)

    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned_data.shape)
    print("\nCleaning report:")
    report = cleaner.get_cleaning_report(sample_data, cleaned_data)
    for key, value in report.items():
        print(f"{key}: {value}")
