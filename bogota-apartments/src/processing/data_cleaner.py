class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, df):
        # Implement data cleaning logic here
        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def fill_missing_values(self, df, fill_value):
        return df.fillna(fill_value)