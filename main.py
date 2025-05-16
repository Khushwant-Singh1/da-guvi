from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from data_integrity import DataIntegrity
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandler
from visualization import Visualization

if __name__ == "__main__":
    # Step 1: Load and clean data using local CSV
    loader = DataLoader("Iris.csv")
    data = loader.load_data()
    print("Data summary after load:", loader.data_summary)

    # Step 2: Feature selection and engineering
    fe = FeatureEngineering(data)
    data = fe.select_features()
    data = fe.engineer_features()

    # Step 3: Data integrity and consistency
    integrity = DataIntegrity(data)
    integrity_report = integrity.check_integrity(verbose=True)
    data = integrity.ensure_consistency()
    print("Data summary after integrity checks:", integrity.data.shape)

    # Step 4: Outlier detection and handling
    outlier_handler = OutlierHandler(data)
    outlier_report = outlier_handler.detect(method='zscore', group_by='Species')
    print("\nOutlier Report (by Species, Z-score):", outlier_report)
    # Remove outliers (optional, can use 'cap' or 'impute' as well)
    data = outlier_handler.handle(strategy='remove', method='zscore')
    print("Data shape after outlier removal:", data.shape)

    # Step 5: Summary statistics and insights
    stats = SummaryStats(data)
    print(stats.get_summary())
    print(stats.get_insights())

    # Step 6: Pattern analysis
    patterns = PatternAnalysis(data)
    patterns.identify_patterns()
    patterns.find_trends_anomalies()

    # Step 7: Visualization
    viz = Visualization(data)
    viz.plot_key_findings()