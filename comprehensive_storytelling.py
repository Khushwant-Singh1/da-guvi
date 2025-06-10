#!/usr/bin/env python3
"""
Comprehensive Data Storytelling Report Generator
"""

import pandas as pd
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandler
from data_storytelling import DataStoryteller

def generate_comprehensive_report():
    """Generate a complete storytelling report"""
    
    print("🔍 IRIS DATASET: COMPREHENSIVE DATA STORYTELLING ANALYSIS")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n📊 PHASE 1: DATA PREPARATION")
    print("-" * 40)
    
    loader = DataLoader("Iris.csv")
    raw_data = loader.load_data()
    print(f"✓ Loaded {raw_data.shape[0]} samples with {raw_data.shape[1]} features")
    
    # Feature engineering for richer analysis
    engineer = FeatureEngineering(raw_data)
    data = engineer.select_features(keep=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
    data = engineer.engineer_features()
    print(f"✓ Engineered features: {data.shape[1]} total features")
    
    # 2. Comprehensive Analysis
    print("\n🔬 PHASE 2: ANALYTICAL DEEP DIVE")
    print("-" * 40)
    
    # Statistical analysis
    stats = SummaryStats(data)
    summary_results = stats.get_insights()
    print(f"✓ Statistical insights generated")
    
    # Pattern detection
    patterns = PatternAnalysis(data)
    pattern_results = patterns.identify_patterns()
    feature_importance = patterns.feature_importance()
    print(f"✓ Patterns identified: {len(pattern_results.get('strong_correlations', []))} strong correlations")
    
    # Outlier analysis
    outlier_handler = OutlierHandler(data)
    outlier_results = outlier_handler.detect(method='iqr')
    total_outliers = sum(info.get('count', 0) for info in outlier_results.values() if isinstance(info, dict))
    print(f"✓ Outlier detection: {total_outliers} outliers found")
    
    # 3. Data Storytelling
    print("\n📖 PHASE 3: STORY CREATION")
    print("-" * 40)
    
    storyteller = DataStoryteller(data, context="iris_classification")
    story = storyteller.analyze_and_tell_story(
        summary_results, 
        pattern_results, 
        outlier_results, 
        feature_importance
    )
    print(f"✓ Data story created with {len(story.key_findings)} key insights")
    
    # 4. Report Generation
    print("\n📝 PHASE 4: NARRATIVE REPORTS")
    print("-" * 40)
    
    # Technical Report
    tech_report = storyteller.generate_narrative_report(story, "technical")
    with open("iris_technical_report.md", 'w') as f:
        f.write(tech_report)
    print("✓ Technical report: iris_technical_report.md")
    
    # Business Report  
    business_report = storyteller.generate_narrative_report(story, "business")
    with open("iris_business_report.md", 'w') as f:
        f.write(business_report)
    print("✓ Business report: iris_business_report.md")
    
    # General Audience Report
    general_report = storyteller.generate_narrative_report(story, "general")
    with open("iris_general_report.md", 'w') as f:
        f.write(general_report)
    print("✓ General report: iris_general_report.md")
    
    # Export insights as JSON
    storyteller.export_insights("iris_insights.json")
    print("✓ Insights data: iris_insights.json")
    
    # 5. Key Insights Summary
    print("\n🎯 EXECUTIVE DASHBOARD")
    print("=" * 70)
    
    print(f"\n📋 EXECUTIVE SUMMARY")
    print("-" * 20)
    print(story.executive_summary)
    
    print(f"\n🔍 KEY FINDINGS")
    print("-" * 20)
    for i, finding in enumerate(story.key_findings, 1):
        print(f"\n{i}. {finding.title}")
        print(f"   🎯 Significance: {finding.significance.upper()}")
        print(f"   📊 Category: {finding.category}")
        print(f"   📝 {finding.description}")
        print(f"   💼 Business Impact: {finding.business_impact}")
    
    print(f"\n💡 STRATEGIC RECOMMENDATIONS")
    print("-" * 20)
    for i, rec in enumerate(story.recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n⚠️  LIMITATIONS & CONSIDERATIONS")
    print("-" * 20)
    for i, limit in enumerate(story.limitations, 1):
        print(f"{i}. {limit}")
    
    print(f"\n📈 ANALYSIS METRICS")
    print("-" * 20)
    print(f"• Dataset Size: {len(data)} samples")
    print(f"• Features Analyzed: {len(data.columns)} total")
    print(f"• Strong Correlations: {len(pattern_results.get('strong_correlations', []))}")
    print(f"• Outliers Detected: {total_outliers}")
    print(f"• Key Insights: {len(story.key_findings)}")
    print(f"• Supporting Evidence: {len(story.supporting_insights)}")
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📁 All reports and data have been saved to files.")
    print("🔍 Review the generated markdown files for detailed insights.")
    

if __name__ == "__main__":
    generate_comprehensive_report()
