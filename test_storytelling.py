#!/usr/bin/env python3
"""
Simple test of the data storytelling functionality
"""

import pandas as pd
from data_loader import DataLoader
from summary_stats import SummaryStats
from pattern_analysis import PatternAnalysis
from outlier_handling import OutlierHandler
from data_storytelling import DataStoryteller

print("🧪 Testing Data Storytelling Module")
print("=" * 50)

# Load basic data
print("Loading data...")
loader = DataLoader("Iris.csv")
data = loader.load_data()
print(f"✓ Loaded {data.shape[0]} rows, {data.shape[1]} columns")

# Basic analysis
print("\n1. Statistical Summary...")
stats = SummaryStats(data)
summary_results = stats.get_insights()
print("✓ Complete")

print("\n2. Pattern Analysis...")
patterns = PatternAnalysis(data)
pattern_results = patterns.identify_patterns()
feature_importance = patterns.feature_importance()
print("✓ Complete")

print("\n3. Outlier Detection...")
outlier_handler = OutlierHandler(data)
outlier_results = outlier_handler.detect(method='iqr')
print("✓ Complete")

print("\n4. Creating Data Story...")
storyteller = DataStoryteller(data)
story = storyteller.analyze_and_tell_story(
    summary_results, 
    pattern_results, 
    outlier_results, 
    feature_importance
)
print("✓ Complete")

print("\n" + "🎯" * 20)
print("STORY RESULTS")
print("🎯" * 20)

print(f"\nExecutive Summary:")
print(story.executive_summary)

print(f"\nKey Findings ({len(story.key_findings)}):")
for i, finding in enumerate(story.key_findings, 1):
    print(f"{i}. {finding.title} ({finding.significance})")

print(f"\nRecommendations ({len(story.recommendations)}):")
for i, rec in enumerate(story.recommendations[:3], 1):
    print(f"{i}. {rec}")

print("\n✅ Data storytelling test completed successfully!")
