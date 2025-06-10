"""
Enhanced Main Pipeline with Data Storytelling and Interpretation

This script demonstrates the complete data analysis pipeline including
statistical analysis, pattern detection, and intelligent storytelling
that transforms technical results into actionable insights.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Import analysis modules
from data_loader import DataLoader
from data_integrity import DataIntegrity
from feature_engineering import FeatureEngineering
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandler
from visualization import Visualization
from data_storytelling import DataStoryteller, create_comprehensive_story

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_comprehensive_analysis(filename: str = "Iris.csv") -> Dict[str, Any]:
    """
    Run the complete analysis pipeline with storytelling.
    
    Args:
        filename: Path to the dataset
        
    Returns:
        Dictionary containing all analysis results and stories
    """
    results = {}
    
    print("=" * 80)
    print("COMPREHENSIVE IRIS DATASET ANALYSIS WITH DATA STORYTELLING")
    print("=" * 80)
    
    # 1. Data Loading
    print("\n1. LOADING AND VALIDATING DATA")
    print("-" * 40)
    loader = DataLoader(filename)
    data = loader.load_data()
    print(f"✓ Loaded dataset: {data.shape[0]} rows × {data.shape[1]} columns")
    results['original_data'] = data.copy()
    
    # 2. Data Integrity Assessment  
    print("\n2. DATA INTEGRITY ASSESSMENT")
    print("-" * 40)
    integrity = DataIntegrity(data)
    integrity_report = integrity.check_integrity()
    print(f"✓ Data integrity assessment completed")
    missing_summary = integrity_report.get('missing_values', {})
    duplicates_info = integrity_report.get('duplicates', {})
    bio_validity = integrity_report.get('biological_validity', {})
    print(f"  - Missing values: {sum(missing_summary.get('counts', {}).values())}")
    print(f"  - Duplicate rows: {duplicates_info.get('count', 0)}")
    print(f"  - Biological validity: {all(bio_validity.values()) if bio_validity else 'Unknown'}")
    results['integrity_report'] = integrity_report
    
    # For storytelling, we'll use the original clean data
    # instead of aggressively removing outliers
    
    # 3. Feature Engineering
    print("\n3. FEATURE ENGINEERING")
    print("-" * 40)
    engineer = FeatureEngineering(data)
    
    # Select core features for analysis
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    engineered_data = engineer.select_features(keep=feature_cols)
    
    # Add meaningful features
    engineered_data = engineer.engineer_features(inplace=False)
    
    print(f"✓ Feature engineering completed")
    print(f"  - Original features: {len(feature_cols)}")
    print(f"  - Total features after engineering: {engineered_data.shape[1]}")
    results['engineered_data'] = engineered_data
    
    # 4. Statistical Summary
    print("\n4. STATISTICAL ANALYSIS")
    print("-" * 40)
    stats = SummaryStats(engineered_data)
    summary_stats = stats.get_insights()
    print(f"✓ Statistical analysis completed")
    print(f"  - Features analyzed: {len(stats.numeric_cols)} numeric, {len(stats.categorical_cols)} categorical")
    results['summary_stats'] = summary_stats
    
    # 5. Pattern Analysis
    print("\n5. PATTERN DETECTION")
    print("-" * 40)
    pattern_analyzer = PatternAnalysis(engineered_data, verbose=False)
    patterns = pattern_analyzer.identify_patterns()
    feature_importance = pattern_analyzer.feature_importance()
    
    print(f"✓ Pattern analysis completed")
    print(f"  - Strong correlations found: {len(patterns.get('strong_correlations', []))}")
    print(f"  - Most important feature: {max(feature_importance.items(), key=lambda x: x[1])[0] if feature_importance else 'None'}")
    results['patterns'] = patterns
    results['feature_importance'] = feature_importance
    
    # 6. Outlier Analysis (Conservative approach)
    print("\n6. OUTLIER DETECTION")
    print("-" * 40)
    outlier_handler = OutlierHandler(engineered_data)
    outlier_results = outlier_handler.detect(method='iqr')
    
    # Count total outliers across all features
    total_outliers = sum(info.get('count', 0) for info in outlier_results.values() if isinstance(info, dict))
    outlier_rate = total_outliers / len(engineered_data) if len(engineered_data) > 0 else 0
    
    print(f"✓ Outlier detection completed")
    print(f"  - Total outliers detected: {total_outliers} ({outlier_rate:.1%} of data)")
    results['outlier_analysis'] = outlier_results
    
    # 7. Data Storytelling and Interpretation
    print("\n7. DATA STORYTELLING AND INTERPRETATION")
    print("-" * 40)
    storyteller = create_comprehensive_story(
        engineered_data, 
        summary_stats, 
        patterns, 
        outlier_results, 
        feature_importance
    )
    
    story = storyteller.stories[0]
    print(f"✓ Data story generated")
    print(f"  - Key findings: {len(story.key_findings)}")
    print(f"  - Supporting insights: {len(story.supporting_insights)}")
    print(f"  - Recommendations: {len(story.recommendations)}")
    results['storyteller'] = storyteller
    results['story'] = story
    
    # 8. Visualization
    print("\n8. VISUALIZATION GENERATION")
    print("-" * 40)
    viz = Visualization(engineered_data)
    viz.plot_all()
    print(f"✓ Visualizations generated successfully")
    results['visualizations'] = viz
    
    return results


def display_story_reports(storyteller: DataStoryteller):
    """Display formatted story reports for different audiences."""
    story = storyteller.stories[0]
    
    print("\n" + "=" * 80)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 80)
    print(story.executive_summary)
    
    print("\n" + "=" * 80)
    print("🔍 KEY FINDINGS")
    print("=" * 80)
    for i, finding in enumerate(story.key_findings, 1):
        print(f"\n{i}. {finding.title}")
        print(f"   Category: {finding.category} | Significance: {finding.significance.upper()}")
        print(f"   {finding.description}")
        print(f"   💼 Business Impact: {finding.business_impact}")
    
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATIONS")
    print("=" * 80)
    for i, rec in enumerate(story.recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n" + "=" * 80)
    print("⚠️  LIMITATIONS")
    print("=" * 80)
    for lim in story.limitations:
        print(f"• {lim}")


def export_reports(storyteller: DataStoryteller, base_filename: str = "iris_analysis"):
    """Export narrative reports for different audiences."""
    story = storyteller.stories[0]
    
    # Technical report
    tech_report = storyteller.generate_narrative_report(story, "technical")
    with open(f"{base_filename}_technical_report.md", 'w') as f:
        f.write(tech_report)
    
    # Business report
    business_report = storyteller.generate_narrative_report(story, "business")
    with open(f"{base_filename}_business_report.md", 'w') as f:
        f.write(business_report)
    
    # General audience report
    general_report = storyteller.generate_narrative_report(story, "general")
    with open(f"{base_filename}_general_report.md", 'w') as f:
        f.write(general_report)
    
    # Export insights as JSON
    storyteller.export_insights(f"{base_filename}_insights.json")
    
    print(f"\n✓ Reports exported:")
    print(f"  - {base_filename}_technical_report.md")
    print(f"  - {base_filename}_business_report.md") 
    print(f"  - {base_filename}_general_report.md")
    print(f"  - {base_filename}_insights.json")


if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_analysis()
    
    # Display the story and insights
    print("\n" + "🎯" * 40)
    print("DATA STORYTELLING RESULTS")
    print("🎯" * 40)
    
    display_story_reports(results['storyteller'])
    
    # Export reports for different audiences
    print("\n" + "=" * 80)
    print("📁 EXPORTING REPORTS")
    print("=" * 80)
    export_reports(results['storyteller'])
    
    print(f"\n🎉 Analysis complete! Check the generated files for detailed insights.")
    print("💡 Tip: Open the HTML visualization files in your browser for interactive exploration.")
