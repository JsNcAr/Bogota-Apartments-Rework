class FeatureExtractor:
    """
    A class that contains methods for extracting features from cleaned data.
    """

    def __init__(self, data):
        """
        Initializes the FeatureExtractor with the cleaned data.

        Args:
            data (DataFrame): The cleaned data from which features will be extracted.
        """
        self.data = data

    def extract_jacuzzi(self):
        """
        Extracts the 'jacuzzi' feature from the data.

        Returns:
            Series: A Series containing the extracted 'jacuzzi' feature.
        """
        return self.data['caracteristicas'].apply(lambda x: 'jacuzzi' in x.lower())

    def extract_piso(self):
        """
        Extracts the 'piso' feature from the data.

        Returns:
            Series: A Series containing the extracted 'piso' feature.
        """
        return self.data['caracteristicas'].apply(self._extract_piso_value)

    def _extract_piso_value(self, caracteristicas):
        """
        Helper method to extract the piso value from the caracteristicas.

        Args:
            caracteristicas (str): The characteristics string.

        Returns:
            int: The extracted piso value or None if not found.
        """
        # Logic to extract piso value from caracteristicas
        # Placeholder implementation
        return None

    def extract_closets(self):
        """
        Extracts the 'closets' feature from the data.

        Returns:
            Series: A Series containing the extracted 'closets' feature.
        """
        return self.data['caracteristicas'].apply(lambda x: self._extract_closets_value(x))

    def _extract_closets_value(self, caracteristicas):
        """
        Helper method to extract the closets value from the caracteristicas.

        Args:
            caracteristicas (str): The characteristics string.

        Returns:
            int: The extracted closets value or None if not found.
        """
        # Logic to extract closets value from caracteristicas
        # Placeholder implementation
        return None

    def extract_chimenea(self):
        """
        Extracts the 'chimenea' feature from the data.

        Returns:
            Series: A Series containing the extracted 'chimenea' feature.
        """
        return self.data['caracteristicas'].apply(lambda x: 'chimenea' in x.lower())

    def extract_permite_mascotas(self):
        """
        Extracts the 'permite_mascotas' feature from the data.

        Returns:
            Series: A Series containing the extracted 'permite_mascotas' feature.
        """
        return self.data['caracteristicas'].apply(lambda x: 'mascotas' in x.lower())

    def extract_gimnasio(self):
        """
        Extracts the 'gimnasio' feature from the data.

        Returns:
            Series: A Series containing the extracted 'gimnasio' feature.
        """
        return self.data['caracteristicas'].apply(lambda x: 'gimnasio' in x.lower())

    def extract_ascensor(self):
        """
        Extracts the 'ascensor' feature from the data.

        Returns:
            Series: A Series containing the extracted 'ascensor' feature.
        """
        return self.data['caracteristicas'].apply(lambda x: 'ascensor' in x.lower())