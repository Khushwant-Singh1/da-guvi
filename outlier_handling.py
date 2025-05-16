import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class OutlierHandler:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.original_data = data.copy()
        self.data = data.copy()
        self.report: Dict = {}

    def detect(self,
              method: str = 'zscore',
              columns: Optional[List[str]] = None,
              group_by: str = 'Species',
              **kwargs) -> Dict:
        """
        Detect outliers using specified method
        Methods: 'zscore', 'iqr', 'isolation_forest'
        """
        if columns is None:
            columns = self._numeric_columns()
            
        if group_by and group_by in self.data.columns:
            return self._grouped_outlier_detection(method, columns, group_by, **kwargs)
            
        return self._detect(method, columns, **kwargs)

    def _numeric_columns(self) -> List[str]:
        """Get list of numeric columns"""
        return self.data.select_dtypes(include=np.number).columns.tolist()

    def _detect(self,
               method: str,
               columns: List[str],
               **kwargs) -> Dict:
        """Main detection logic"""
        detectors = {
            'zscore': self._zscore_detection,
            'iqr': self._iqr_detection,
            'isolation_forest': self._isolation_forest_detection
        }
        return detectors[method](columns, **kwargs)

    def _grouped_outlier_detection(self,
                                  method: str,
                                  columns: List[str],
                                  group_by: str,
                                  **kwargs) -> Dict:
        """Detect outliers within each group"""
        results = {}
        for group, df_group in self.data.groupby(group_by):
            handler = OutlierHandler(df_group)
            results[group] = handler._detect(method, columns, **kwargs)
        self.report['grouped_outliers'] = results
        return results

    def _zscore_detection(self,
                         columns: List[str],
                         threshold: float = 3.0) -> Dict:
        """Z-score based outlier detection"""
        outliers = {}
        for col in columns:
            z_scores = stats.zscore(self.data[col])
            mask = np.abs(z_scores) > threshold
            outliers[col] = {
                'indices': self.data[mask].index.tolist(),
                'count': mask.sum(),
                'threshold': threshold
            }
        self.report['zscore'] = outliers
        return outliers

    def _iqr_detection(self,
                      columns: List[str],
                      factor: float = 1.5) -> Dict:
        """IQR based outlier detection"""
        outliers = {}
        for col in columns:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask = (self.data[col] < lower) | (self.data[col] > upper)
            outliers[col] = {
                'indices': self.data[mask].index.tolist(),
                'count': mask.sum(),
                'bounds': (lower, upper)
            }
        self.report['iqr'] = outliers
        return outliers

    def _isolation_forest_detection(self,
                                   columns: List[str],
                                   contamination: float = 0.05) -> Dict:
        """Multivariate outlier detection"""
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(self.data[columns])
        mask = preds == -1
        return {
            'indices': self.data[mask].index.tolist(),
            'count': mask.sum(),
            'contamination': contamination
        }

    def handle(self,
              strategy: str = 'remove',
              method: str = 'zscore',
              columns: Optional[List[str]] = None,
              **kwargs) -> pd.DataFrame:
        """
        Handle detected outliers
        Strategies: 'remove', 'cap', 'impute'
        """
        if strategy == 'remove':
            return self._remove_outliers(method, columns, **kwargs)
        elif strategy == 'cap':
            return self._cap_outliers(method, columns, **kwargs)
        elif strategy == 'impute':
            return self._impute_outliers(method, columns, **kwargs)
        return self.data

    def _remove_outliers(self,
                        method: str,
                        columns: Optional[List[str]],
                        **kwargs) -> pd.DataFrame:
        """Remove detected outliers"""
        outliers = self.detect(method, columns, **kwargs)
        mask = np.ones(len(self.data), dtype=bool)
        for col in outliers:
            mask[outliers[col]['indices']] = False
        return self.data[mask].reset_index(drop=True)

    def _cap_outliers(self,
                     method: str,
                     columns: Optional[List[str]],
                     **kwargs) -> pd.DataFrame:
        """Cap outliers to detection bounds"""
        outliers = self.detect(method, columns, **kwargs)
        df = self.data.copy()
        for col, info in outliers.items():
            if method == 'iqr' and 'bounds' in info:
                df[col] = df[col].clip(*info['bounds'])
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
        return df

    def _impute_outliers(self,
                        method: str,
                        columns: Optional[List[str]],
                        **kwargs) -> pd.DataFrame:
        """Impute outliers with median values"""
        outliers = self.detect(method, columns, **kwargs)
        df = self.data.copy()
        for col in outliers:
            median = df[col].median()
            df.loc[outliers[col]['indices'], col] = median
        return df

    def visualize(self,
                 column: str,
                 method: str = 'boxplot') -> None:
        """Visualize outliers"""
        if method == 'boxplot':
            plt.figure(figsize=(10, 4))
            self.data.boxplot(column=column)
            plt.title(f"Boxplot of {column}")
        elif method == 'scatter':
            if 'Species' in self.data.columns:
                plt.figure(figsize=(10, 6))
                for species, group in self.data.groupby('Species'):
                    plt.scatter(group[column], group['Species'], label=species)
                plt.title(f"{column} by Species")
                plt.legend()
        plt.show()

    def get_report(self) -> Dict:
        """Get complete outlier analysis report"""
        return self.report

    def restore_original(self) -> None:
        """Restore data to original state"""
        self.data = self.original_data.copy()