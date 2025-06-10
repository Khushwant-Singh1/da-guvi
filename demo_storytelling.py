#!/usr/bin/env python3
"""
Final Demonstration: Complete Data Storytelling Pipeline
========================================================

This script demonstrates the full capabilities of the enhanced Iris dataset
analysis pipeline with intelligent data storytelling and interpretation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import json

# Import all modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandler
from data_storytelling import DataStoryteller

def main():
    """Run the complete demonstration."""
    
    print("🌸 IRIS DATASET: COMPLETE DATA STORYTELLING DEMONSTRATION")
    print("=" * 65)
    print("This demonstration showcases how raw data transforms into")
    print("actionable business insights through intelligent analysis.")
    print("=" * 65)
    
    # Phase 1: Data Foundation
    print("\n🔧 PHASE 1: BUILDING DATA FOUNDATION")
    print("-" * 45)
    
    # Load original data
    loader = DataLoader("Iris.csv")
    raw_data = loader.load_data()
    print(f"✓ Original dataset: {raw_data.shape[0]} samples, {raw_data.shape[1]} features")
    
    # Engineer features for richer analysis
    engineer = FeatureEngineering(raw_data)
    engineered_data = engineer.select_features(
        keep=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    )
    engineered_data = engineer.engineer_features()
    
    print(f"✓ Feature engineering complete: {engineered_data.shape[1]} total features")
    print(f"  - Added biological ratios and areas")
    print(f"  - Encoded categorical variables")
    
    # Phase 2: Deep Analysis
    print("\n🔍 PHASE 2: DEEP ANALYTICAL EXPLORATION")
    print("-" * 45)
    
    # Statistical foundation
    stats_analyzer = SummaryStats(engineered_data)
    statistical_insights = stats_analyzer.get_insights()
    print("✓ Statistical analysis complete")
    
    # Pattern discovery
    pattern_analyzer = PatternAnalysis(engineered_data, verbose=False)
    patterns = pattern_analyzer.identify_patterns()
    feature_importance = pattern_analyzer.feature_importance()
    
    # Count discoveries
    strong_correlations = len(patterns.get('strong_correlations', []))
    print(f"✓ Pattern analysis complete: {strong_correlations} strong correlations found")
    
    # Quality assessment
    outlier_handler = OutlierHandler(engineered_data)
    outlier_results = outlier_handler.detect(method='iqr')
    
    total_outliers = sum(info.get('count', 0) for info in outlier_results.values() 
                        if isinstance(info, dict))
    print(f"✓ Quality assessment complete: {total_outliers} outliers detected")
    
    # Phase 3: Intelligence Layer
    print("\n🧠 PHASE 3: INTELLIGENT STORYTELLING")
    print("-" * 45)
    
    # Create the storyteller
    storyteller = DataStoryteller(engineered_data, context="iris_classification")
    
    # Generate comprehensive story
    story = storyteller.analyze_and_tell_story(
        statistical_insights,
        patterns,
        outlier_results,
        feature_importance
    )
    
    print(f"✓ Story generation complete:")
    print(f"  - {len(story.key_findings)} key insights identified")
    print(f"  - {len(story.supporting_insights)} supporting findings")
    print(f"  - {len(story.recommendations)} strategic recommendations")
    
    # Phase 4: Multi-Audience Communication
    print("\n📢 PHASE 4: MULTI-AUDIENCE COMMUNICATION")
    print("-" * 45)
    
    # Generate reports for different audiences
    audiences = {
        "technical": "Data scientists and analysts",
        "business": "Executives and stakeholders", 
        "general": "Public and educational use"
    }
    
    generated_files = []
    
    for audience_type, description in audiences.items():
        report = storyteller.generate_narrative_report(story, audience_type)
        filename = f"demo_{audience_type}_report.md"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        generated_files.append(filename)
        print(f"✓ {audience_type.capitalize()} report → {filename}")
        print(f"  Target: {description}")
    
    # Export structured insights
    storyteller.export_insights("demo_insights.json")
    generated_files.append("demo_insights.json")
    print(f"✓ Structured data → demo_insights.json")
    
    # Phase 5: Executive Summary
    print("\n📊 EXECUTIVE SUMMARY")
    print("=" * 65)
    
    print("\n🎯 KEY BUSINESS INSIGHTS:")
    for i, finding in enumerate(story.key_findings[:3], 1):  # Top 3
        print(f"\n{i}. {finding.title}")
        print(f"   Significance: {finding.significance.upper()}")
        print(f"   Impact: {finding.business_impact[:100]}...")
    
    print(f"\n💡 TOP STRATEGIC RECOMMENDATIONS:")
    for i, recommendation in enumerate(story.recommendations[:3], 1):  # Top 3
        print(f"{i}. {recommendation}")
    
    print(f"\n📈 ANALYSIS SUMMARY:")
    print(f"• Dataset: {len(engineered_data)} samples × {len(engineered_data.columns)} features")
    print(f"• Correlations: {strong_correlations} strong relationships identified")
    print(f"• Data Quality: {100 - (total_outliers/len(engineered_data)*100):.1f}% clean data")
    print(f"• Key Insights: {len(story.key_findings)} critical findings")
    
    # Phase 6: Next Steps
    print(f"\n🚀 RECOMMENDED NEXT STEPS:")
    print("-" * 45)
    
    next_steps = [
        "Review generated reports for detailed technical analysis",
        "Implement feature selection based on importance rankings",
        "Develop machine learning models using identified key features",
        "Validate findings with additional iris datasets",
        "Apply insights to real-world species classification tasks"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"{i}. {step}")
    
    print(f"\n📁 Generated Files:")
    for file in generated_files:
        print(f"  • {file}")
    
    print(f"\n✅ DEMONSTRATION COMPLETE!")
    print("Your data has been transformed into actionable intelligence.")
    print("Review the generated reports to explore insights in detail.")

if __name__ == "__main__":
    main()
