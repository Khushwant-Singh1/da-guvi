class DataIntegrity:
    def __init__(self, data):
        self.data = data

    def check_integrity(self):
        """
        Checks for duplicate rows, missing values, and correct data types.
        Returns a dictionary with integrity check results.
        """
        results = {}
        # Check for missing values
        results['missing_values'] = self.data.isnull().sum().to_dict()
        # Check for duplicate rows
        results['duplicate_rows'] = self.data.duplicated().sum()
        # Check for correct data types
        results['data_types'] = self.data.dtypes.apply(lambda x: str(x)).to_dict()
        return results

    def ensure_consistency(self):
        """
        Ensures data consistency by dropping duplicate rows and resetting index.
        Returns the cleaned DataFrame.
        """
        # Drop duplicate rows
        self.data = self.data.drop_duplicates().reset_index(drop=True)
        return self.data

    def handle_missing_values(self, strategy='drop', fill_value=None):
        """
        Handles missing values in the dataset.
        strategy: 'drop' to remove rows with missing values, 'fill' to fill missing values.
        fill_value: value to fill missing data with (used if strategy is 'fill').
        Returns the cleaned DataFrame.
        """
        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy == 'fill':
            self.data = self.data.fillna(fill_value)
        return self.data
