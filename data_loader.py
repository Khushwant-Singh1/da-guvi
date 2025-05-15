class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Loads the Iris dataset from a local CSV file (Iris.csv) in the current directory.
        """
        import pandas as pd
        self.data = pd.read_csv("Iris.csv")
        return self.data

    def load_data_kagglehub(self, file_path="Iris.csv"):
        """
        Loads the Iris dataset from KaggleHub using the specified file name.
        Default is 'Iris.csv'.
        """
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        self.data = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "uciml/iris",
            file_path
        )
        return self.data

    def download_iris_with_kagglehub(self):
        import kagglehub
        path = kagglehub.dataset_download("uciml/iris")
        print("Path to dataset files:", path)
        return path
