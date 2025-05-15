from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from data_integrity import DataIntegrity
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandling
from visualization import Visualization

if __name__ == "__main__":
    # Step 1: Load and clean data using KaggleHub
    loader = DataLoader("")  # Filepath not needed for KaggleHub
    data = loader.load_data_kagglehub("Iris.csv")
    data = loader.handle_missing_values()

    # Step 2: Feature selection and engineering
    fe = FeatureEngineering(data)
    data = fe.select_features()
    data = fe.engineer_features()

    # Step 3: Data integrity and consistency
    integrity = DataIntegrity(data)
    integrity.check_integrity()
    data = integrity.ensure_consistency()

    # Step 4: Summary statistics and insights
    stats = SummaryStats(data)
    print(stats.get_summary())
    print(stats.get_insights())

    # Step 5: Pattern analysis
    patterns = PatternAnalysis(data)
    patterns.identify_patterns()
    patterns.find_trends_anomalies()

    # Step 6: Outlier handling and transformation
    outliers = OutlierHandling(data)
    data = outliers.handle_outliers()
    data = outliers.transform_data()

    # Step 7: Visualization
    viz = Visualization(data)
    viz.plot_key_findings()
