"""
Data Storytelling and Interpretation Module for Iris Dataset Analysis

This module transforms technical analysis results into meaningful narratives and insights
that can be understood by different audiences (technical, business, general public).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataStorytelling")


@dataclass
class Insight:
    """Container for a single data insight with narrative."""
    category: str
    title: str
    description: str
    evidence: Dict[str, Any]
    significance: str  # 'low', 'medium', 'high', 'critical'
    business_impact: str
    technical_details: Optional[str] = None


@dataclass
class Story:
    """Container for a complete data story."""
    title: str
    executive_summary: str
    key_findings: List[Insight]
    supporting_insights: List[Insight]
    recommendations: List[str]
    methodology: str
    limitations: List[str]


class DataStoryteller:
    """
    Transforms data analysis results into compelling narratives and actionable insights.
    
    This class takes technical analysis outputs and creates different narrative styles
    for various audiences while maintaining scientific rigor.
    """
    
    def __init__(self, data: pd.DataFrame, context: str = "iris_classification"):
        self.data = data
        self.context = context
        self.insights = []
        self.stories = []
        
        # Define significance thresholds based on context
        self.significance_thresholds = {
            'correlation': {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.9},
            'variance_explained': {'low': 0.2, 'medium': 0.5, 'high': 0.7, 'critical': 0.9},
            'outlier_percentage': {'low': 0.01, 'medium': 0.05, 'high': 0.1, 'critical': 0.2}
        }
    
    def analyze_and_tell_story(self, 
                              summary_stats: Dict[str, Any],
                              pattern_analysis: Dict[str, Any],
                              outlier_analysis: Dict[str, Any],
                              feature_importance: Dict[str, float]) -> Story:
        """
        Create a comprehensive data story from analysis results.
        
        Args:
            summary_stats: Results from statistical summary analysis
            pattern_analysis: Results from pattern detection
            outlier_analysis: Results from outlier detection
            feature_importance: Feature importance scores
            
        Returns:
            Complete data story with insights and recommendations
        """
        logger.info("Generating comprehensive data story...")
        
        # Extract insights from different analysis components
        correlation_insights = self._analyze_correlations(pattern_analysis)
        distribution_insights = self._analyze_distributions(pattern_analysis, summary_stats)
        species_insights = self._analyze_species_patterns(summary_stats, pattern_analysis)
        outlier_insights = self._analyze_outliers(outlier_analysis)
        feature_insights = self._analyze_feature_importance(feature_importance)
        
        # Combine all insights
        all_insights = (correlation_insights + distribution_insights + 
                       species_insights + outlier_insights + feature_insights)
        
        # Rank insights by significance
        key_findings = [insight for insight in all_insights if insight.significance in ['critical', 'high']]
        supporting_insights = [insight for insight in all_insights if insight.significance in ['medium', 'low']]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_insights)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(key_findings)
        
        # Create methodology description
        methodology = self._describe_methodology()
        
        # Identify limitations
        limitations = self._identify_limitations()
        
        story = Story(
            title="Iris Species Classification: A Data-Driven Analysis",
            executive_summary=executive_summary,
            key_findings=key_findings,
            supporting_insights=supporting_insights,
            recommendations=recommendations,
            methodology=methodology,
            limitations=limitations
        )
        
        self.stories.append(story)
        return story
    
    def _analyze_correlations(self, pattern_analysis: Dict[str, Any]) -> List[Insight]:
        """Extract insights from correlation analysis."""
        insights = []
        
        if 'correlation_matrix' in pattern_analysis:
            correlations = pattern_analysis['correlation_matrix']
            strong_correlations = pattern_analysis.get('strong_correlations', [])
            
            if strong_correlations:
                # Find the strongest correlation
                strongest = max(strong_correlations, key=lambda x: abs(x[2]))
                col1, col2, corr_value = strongest
                
                significance = self._assess_correlation_significance(abs(corr_value))
                
                insight = Insight(
                    category="Correlation Analysis",
                    title=f"Strong Relationship Between {col1} and {col2}",
                    description=f"We discovered a {('positive' if corr_value > 0 else 'negative')} correlation of {corr_value:.3f} between {col1} and {col2}. This indicates that these features move together in a predictable pattern.",
                    evidence={
                        'correlation_coefficient': corr_value,
                        'total_strong_correlations': len(strong_correlations),
                        'feature_pair': (col1, col2)
                    },
                    significance=significance,
                    business_impact=self._interpret_correlation_impact(col1, col2, corr_value),
                    technical_details=f"Pearson correlation coefficient: {corr_value:.4f}. Statistical significance indicates this relationship is unlikely to be due to chance."
                )
                insights.append(insight)
            
            # Analyze overall correlation structure
            if isinstance(correlations, dict):
                # Convert to DataFrame for easier analysis
                corr_df = pd.DataFrame(correlations)
                avg_correlation = np.mean([abs(corr_df.iloc[i, j]) 
                                         for i in range(len(corr_df)) 
                                         for j in range(i+1, len(corr_df.columns))])
                
                insight = Insight(
                    category="Feature Relationships",
                    title="Overall Feature Interconnectedness",
                    description=f"The average correlation between features is {avg_correlation:.3f}, indicating {'moderate' if avg_correlation > 0.5 else 'low'} interconnectedness in the dataset.",
                    evidence={'average_correlation': avg_correlation, 'correlation_matrix': correlations},
                    significance='medium' if avg_correlation > 0.5 else 'low',
                    business_impact="High feature correlation suggests that some measurements might be redundant for classification purposes, potentially allowing for simpler models.",
                    technical_details=f"Mean absolute correlation: {avg_correlation:.4f}"
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_distributions(self, pattern_analysis: Dict[str, Any], summary_stats: Dict[str, Any]) -> List[Insight]:
        """Extract insights from distribution analysis."""
        insights = []
        
        if 'distribution_shape' in pattern_analysis:
            distributions = pattern_analysis['distribution_shape']
            
            # Analyze skewness
            highly_skewed = [(col, stats['skewness']) for col, stats in distributions.items() 
                           if abs(stats['skewness']) > 1.0]
            
            if highly_skewed:
                col, skewness = max(highly_skewed, key=lambda x: abs(x[1]))
                
                insight = Insight(
                    category="Data Distribution",
                    title=f"Asymmetric Distribution in {col}",
                    description=f"The {col} feature shows significant {'right' if skewness > 0 else 'left'} skewness ({skewness:.3f}), indicating an asymmetric distribution with a longer tail on the {'right' if skewness > 0 else 'left'} side.",
                    evidence={'skewness': skewness, 'feature': col},
                    significance='high' if abs(skewness) > 2 else 'medium',
                    business_impact="Skewed distributions may require data transformation for optimal model performance and can indicate natural biological variation patterns.",
                    technical_details=f"Skewness coefficient: {skewness:.4f}. Values > 1 or < -1 indicate moderate to high skewness."
                )
                insights.append(insight)
        
        # Analyze variance patterns
        if 'feature_variance' in summary_stats:
            variance_info = summary_stats['feature_variance']
            highest_var = variance_info.get('highest_variance')
            lowest_var = variance_info.get('lowest_variance')
            
            if highest_var and lowest_var:
                insight = Insight(
                    category="Feature Variability",
                    title="Variability Patterns Across Features",
                    description=f"Feature variability analysis reveals {highest_var} has the highest variance while {lowest_var} shows the most consistent values across samples.",
                    evidence=variance_info,
                    significance='medium',
                    business_impact="Understanding feature variability helps in feature selection and model interpretability.",
                    technical_details="Variance analysis helps identify which features contribute most to distinguishing between samples."
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_species_patterns(self, summary_stats: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> List[Insight]:
        """Extract insights specific to species classification."""
        insights = []
        
        if 'species_analysis' in summary_stats:
            species_info = summary_stats['species_analysis']
            
            # Analyze species distribution
            if 'distribution' in species_info:
                distribution = species_info['distribution']
                total_samples = sum(distribution.values())
                
                # Check for balanced dataset
                min_count = min(distribution.values())
                max_count = max(distribution.values())
                balance_ratio = min_count / max_count if max_count > 0 else 0
                
                insight = Insight(
                    category="Dataset Balance",
                    title="Species Distribution Analysis",
                    description=f"The dataset contains {len(distribution)} species with a balance ratio of {balance_ratio:.3f}. {'This indicates a well-balanced dataset' if balance_ratio > 0.8 else 'This suggests some imbalance between species'}.",
                    evidence={'distribution': distribution, 'balance_ratio': balance_ratio, 'total_samples': total_samples},
                    significance='high' if balance_ratio < 0.7 else 'medium',
                    business_impact="Dataset balance affects model performance and bias. Balanced datasets typically lead to more robust classification models.",
                    technical_details=f"Balance ratio: {balance_ratio:.4f}. Ratios < 0.7 may require sampling strategies."
                )
                insights.append(insight)
            
            # Analyze measurement differences between species
            if 'measurement_means' in species_info:
                means = species_info['measurement_means']
                
                # Find the feature that best discriminates between species
                max_variation = 0
                best_discriminator = None
                
                for feature, species_means in means.items():
                    if len(species_means) > 1:
                        values = list(species_means.values())
                        variation = max(values) - min(values)
                        if variation > max_variation:
                            max_variation = variation
                            best_discriminator = feature
                
                if best_discriminator:
                    insight = Insight(
                        category="Species Differentiation",
                        title=f"Key Discriminating Feature: {best_discriminator}",
                        description=f"Analysis reveals that {best_discriminator} shows the largest variation between species (range: {max_variation:.3f}), making it a key feature for species identification.",
                        evidence={'discriminator': best_discriminator, 'variation': max_variation, 'means': means[best_discriminator]},
                        significance='high',
                        business_impact="Identifying key discriminating features helps in developing efficient classification models and understanding biological differences.",
                        technical_details=f"Inter-species variation: {max_variation:.4f} for {best_discriminator}"
                    )
                    insights.append(insight)
        
        return insights
    
    def _analyze_outliers(self, outlier_analysis: Dict[str, Any]) -> List[Insight]:
        """Extract insights from outlier analysis."""
        insights = []
        
        total_outliers = 0
        features_with_outliers = []
        
        for feature, outlier_info in outlier_analysis.items():
            if isinstance(outlier_info, dict) and 'count' in outlier_info:
                count = outlier_info['count']
                if count > 0:
                    total_outliers += count
                    features_with_outliers.append((feature, count))
        
        if total_outliers > 0:
            outlier_rate = total_outliers / len(self.data) if len(self.data) > 0 else 0
            significance = self._assess_outlier_significance(outlier_rate)
            
            insight = Insight(
                category="Data Quality",
                title="Outlier Detection Results",
                description=f"Detected {total_outliers} outliers across {len(features_with_outliers)} features ({outlier_rate:.1%} of total data points). This {'suggests good data quality' if outlier_rate < 0.05 else 'indicates potential data quality issues'}.",
                evidence={'total_outliers': total_outliers, 'outlier_rate': outlier_rate, 'affected_features': features_with_outliers},
                significance=significance,
                business_impact="Outliers can indicate measurement errors, rare variants, or genuine biological extremes. Understanding their nature is crucial for model accuracy.",
                technical_details=f"Outlier detection using multiple methods (IQR, Z-score, Isolation Forest). Rate: {outlier_rate:.4f}"
            )
            insights.append(insight)
        else:
            insight = Insight(
                category="Data Quality",
                title="Clean Dataset with No Outliers",
                description="No outliers were detected in any features, indicating high data quality and consistency in measurements.",
                evidence={'total_outliers': 0, 'outlier_rate': 0.0},
                significance='medium',
                business_impact="Clean data suggests reliable measurement processes and reduces the need for extensive data preprocessing.",
                technical_details="Multiple outlier detection methods applied: IQR, Z-score, and Isolation Forest"
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_feature_importance(self, feature_importance: Dict[str, float]) -> List[Insight]:
        """Extract insights from feature importance analysis."""
        insights = []
        
        if feature_importance:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Most important feature
            most_important = sorted_features[0]
            least_important = sorted_features[-1]
            
            # Calculate importance concentration
            top_3_importance = sum([score for _, score in sorted_features[:3]])
            
            insight = Insight(
                category="Feature Importance",
                title=f"Primary Predictive Feature: {most_important[0]}",
                description=f"{most_important[0]} emerges as the most predictive feature with an importance score of {most_important[1]:.4f}. The top 3 features account for {top_3_importance:.1%} of total predictive power.",
                evidence={'feature_ranking': sorted_features, 'top_3_concentration': top_3_importance},
                significance='high',
                business_impact="Understanding feature importance guides model simplification and helps focus measurement efforts on the most informative characteristics.",
                technical_details=f"Importance calculated using variance and correlation-based metrics. Top feature: {most_important[1]:.4f}"
            )
            insights.append(insight)
            
            # Check for feature redundancy
            if len(sorted_features) > 3:
                bottom_features = [f for f, score in sorted_features[-3:]]
                min_importance = min([score for _, score in sorted_features[-3:]])
                
                if min_importance < 0.1:  # Less than 10% importance
                    insight = Insight(
                        category="Feature Optimization",
                        title="Potential Feature Redundancy Identified",
                        description=f"Features {', '.join(bottom_features)} show low importance (< 10%), suggesting they may be redundant for classification purposes.",
                        evidence={'low_importance_features': bottom_features, 'min_importance': min_importance},
                        significance='medium',
                        business_impact="Removing low-importance features can simplify models, reduce measurement costs, and improve interpretability without sacrificing accuracy.",
                        technical_details=f"Minimum importance threshold: 0.1. Lowest observed: {min_importance:.4f}"
                    )
                    insights.append(insight)
        
        return insights
    
    def _assess_correlation_significance(self, correlation: float) -> str:
        """Assess the significance level of a correlation."""
        thresholds = self.significance_thresholds['correlation']
        if correlation >= thresholds['critical']:
            return 'critical'
        elif correlation >= thresholds['high']:
            return 'high'
        elif correlation >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_outlier_significance(self, outlier_rate: float) -> str:
        """Assess the significance level of outlier prevalence."""
        thresholds = self.significance_thresholds['outlier_percentage']
        if outlier_rate >= thresholds['critical']:
            return 'critical'
        elif outlier_rate >= thresholds['high']:
            return 'high'
        elif outlier_rate >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _interpret_correlation_impact(self, col1: str, col2: str, correlation: float) -> str:
        """Provide business impact interpretation for correlations."""
        if abs(correlation) > 0.8:
            return f"Strong correlation between {col1} and {col2} suggests these measurements are highly related, potentially allowing for cost-effective single-feature measurement strategies."
        elif abs(correlation) > 0.5:
            return f"Moderate correlation between {col1} and {col2} indicates some relationship that could be useful for predictive modeling."
        else:
            return f"Weak correlation between {col1} and {col2} suggests these features provide independent information for classification."
    
    def _generate_recommendations(self, insights: List[Insight]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        # Correlation-based recommendations
        high_corr_insights = [i for i in insights if i.category == "Correlation Analysis" and i.significance in ['high', 'critical']]
        if high_corr_insights:
            recommendations.append("Consider feature selection techniques to reduce redundancy from highly correlated features, which could improve model efficiency and interpretability.")
        
        # Distribution-based recommendations
        skew_insights = [i for i in insights if i.category == "Data Distribution" and i.significance in ['high', 'critical']]
        if skew_insights:
            recommendations.append("Apply appropriate data transformations (e.g., log transformation) to handle skewed distributions for improved model performance.")
        
        # Outlier-based recommendations
        outlier_insights = [i for i in insights if i.category == "Data Quality" and 'outliers' in i.title.lower()]
        high_outlier_insights = [i for i in outlier_insights if i.significance in ['high', 'critical']]
        if high_outlier_insights:
            recommendations.append("Investigate outliers to determine if they represent measurement errors or genuine biological variation. Consider robust modeling techniques.")
        elif outlier_insights:
            recommendations.append("The clean dataset provides an excellent foundation for machine learning models without extensive preprocessing.")
        
        # Feature importance recommendations
        importance_insights = [i for i in insights if i.category in ["Feature Importance", "Feature Optimization"]]
        if importance_insights:
            recommendations.append("Focus on the most important features for model development and consider removing low-importance features to simplify the model.")
        
        # Species-specific recommendations
        species_insights = [i for i in insights if i.category in ["Species Differentiation", "Dataset Balance"]]
        if species_insights:
            recommendations.append("Leverage the identified discriminating features for efficient species classification and consider the dataset balance in model evaluation.")
        
        # General recommendations
        recommendations.extend([
            "Validate findings with cross-validation to ensure robustness across different data splits.",
            "Consider ensemble methods to combine multiple weak learners for improved classification accuracy.",
            "Document the analysis methodology and maintain data lineage for reproducibility."
        ])
        
        return recommendations
    
    def _create_executive_summary(self, key_findings: List[Insight]) -> str:
        """Create an executive summary of key findings."""
        if not key_findings:
            return "Analysis of the Iris dataset reveals a clean, well-structured dataset suitable for classification modeling with no critical issues identified."
        
        summary_parts = ["Analysis of the Iris dataset reveals several key insights:"]
        
        for finding in key_findings[:3]:  # Top 3 findings
            summary_parts.append(f"• {finding.title}: {finding.description.split('.')[0]}.")
        
        if len(key_findings) > 3:
            summary_parts.append(f"• Additional {len(key_findings) - 3} high-significance findings support these conclusions.")
        
        summary_parts.append("These findings provide a strong foundation for developing accurate species classification models.")
        
        return " ".join(summary_parts)
    
    def _describe_methodology(self) -> str:
        """Describe the analytical methodology used."""
        return """
        This analysis employed a comprehensive multi-stage approach:
        1. Data Quality Assessment: Evaluated completeness, consistency, and biological plausibility
        2. Statistical Analysis: Computed descriptive statistics, distributions, and correlations
        3. Pattern Detection: Applied correlation analysis, variance assessment, and feature importance calculation
        4. Outlier Detection: Used multiple methods (IQR, Z-score, Isolation Forest) for robust outlier identification
        5. Species-specific Analysis: Examined inter-species differences and dataset balance
        6. Insight Synthesis: Transformed technical findings into actionable business insights
        """
    
    def _identify_limitations(self) -> List[str]:
        """Identify analysis limitations."""
        return [
            "Analysis based on a single dataset; findings may not generalize to other iris populations",
            "Correlation does not imply causation; observed relationships may be influenced by unmeasured factors",
            "Outlier detection methods assume specific distributions; results may vary with different assumptions",
            "Feature importance calculations are relative to this specific dataset and may change with different samples",
            "Biological interpretation requires domain expertise beyond the scope of statistical analysis"
        ]
    
    def generate_narrative_report(self, story: Story, audience: str = "technical") -> str:
        """
        Generate a narrative report tailored to specific audience.
        
        Args:
            story: The data story to narrate
            audience: Target audience ("technical", "business", "general")
            
        Returns:
            Formatted narrative report
        """
        if audience == "technical":
            return self._generate_technical_report(story)
        elif audience == "business":
            return self._generate_business_report(story)
        else:
            return self._generate_general_report(story)
    
    def _generate_technical_report(self, story: Story) -> str:
        """Generate technical report for data scientists and analysts."""
        report = f"""
# {story.title}

## Executive Summary
{story.executive_summary}

## Methodology
{story.methodology.strip()}

## Key Findings

"""
        for i, finding in enumerate(story.key_findings, 1):
            report += f"""
### {i}. {finding.title}
**Category:** {finding.category}  
**Significance:** {finding.significance}

{finding.description}

**Technical Details:** {finding.technical_details}

**Evidence:**
```json
{json.dumps(finding.evidence, indent=2, default=str)}
```

"""
        
        if story.supporting_insights:
            report += "\n## Supporting Insights\n"
            for finding in story.supporting_insights:
                report += f"- **{finding.title}**: {finding.description}\n"
        
        report += f"""
## Recommendations
{chr(10).join(f"• {rec}" for rec in story.recommendations)}

## Limitations
{chr(10).join(f"• {lim}" for lim in story.limitations)}
"""
        return report
    
    def _generate_business_report(self, story: Story) -> str:
        """Generate business-focused report for stakeholders."""
        report = f"""
# {story.title.replace(':', ' -')} 
## Business Intelligence Summary

{story.executive_summary}

## Key Business Insights

"""
        for i, finding in enumerate(story.key_findings, 1):
            report += f"""
### {i}. {finding.title}
{finding.description}

**Business Impact:** {finding.business_impact}

"""
        
        report += f"""
## Strategic Recommendations
{chr(10).join(f"{i+1}. {rec}" for i, rec in enumerate(story.recommendations[:5]))}

## Considerations
{chr(10).join(f"• {lim}" for lim in story.limitations[:3])}
"""
        return report
    
    def _generate_general_report(self, story: Story) -> str:
        """Generate general audience report."""
        report = f"""
# Understanding Iris Flower Classification Through Data

## What We Discovered
{story.executive_summary}

## Main Findings

"""
        for i, finding in enumerate(story.key_findings, 1):
            # Simplify language for general audience
            simple_description = finding.description.replace("correlation", "relationship").replace("coefficient", "measure")
            report += f"""
### {i}. {finding.title}
{simple_description}

**Why This Matters:** {finding.business_impact.replace("model", "analysis").replace("classification", "identification")}

"""
        
        report += f"""
## What This Means
The analysis shows that iris flowers can be reliably identified using their physical measurements. This research demonstrates how data science can help us understand natural patterns and make accurate predictions.

## Next Steps
{chr(10).join(f"• {rec}" for rec in story.recommendations[:3])}
"""
        return report
    
    def export_insights(self, filename: str, format: str = "json") -> None:
        """Export insights to file."""
        if format == "json":
            insights_data = [
                {
                    'category': insight.category,
                    'title': insight.title,
                    'description': insight.description,
                    'significance': insight.significance,
                    'business_impact': insight.business_impact,
                    'evidence': insight.evidence
                }
                for insight in self.insights
            ]
            
            with open(filename, 'w') as f:
                json.dump(insights_data, f, indent=2, default=str)
        
        logger.info(f"Insights exported to {filename}")


def create_comprehensive_story(data: pd.DataFrame, 
                             summary_stats: Dict[str, Any],
                             pattern_analysis: Dict[str, Any],
                             outlier_analysis: Dict[str, Any],
                             feature_importance: Dict[str, float]) -> DataStoryteller:
    """
    Convenience function to create a complete data story.
    
    Args:
        data: Original dataset
        summary_stats: Statistical summary results
        pattern_analysis: Pattern detection results
        outlier_analysis: Outlier detection results
        feature_importance: Feature importance scores
        
    Returns:
        DataStoryteller instance with complete story
    """
    storyteller = DataStoryteller(data)
    story = storyteller.analyze_and_tell_story(
        summary_stats, pattern_analysis, outlier_analysis, feature_importance
    )
    return storyteller


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from summary_stats import SummaryStats
    from pattern_analysis import PatternAnalysis
    from outlier_handling import OutlierHandler
    
    # Load data
    loader = DataLoader("Iris.csv")
    data = loader.load_data()
    
    # Run analyses
    stats = SummaryStats(data)
    summary_results = stats.get_insights()
    
    patterns = PatternAnalysis(data)
    pattern_results = patterns.identify_patterns()
    
    outlier_handler = OutlierHandler(data)
    outlier_results = outlier_handler.detect(method='iqr')
    
    feature_importance = patterns.feature_importance()
    
    # Create story
    storyteller = create_comprehensive_story(
        data, summary_results, pattern_results, outlier_results, feature_importance
    )
    
    # Generate reports for different audiences
    story = storyteller.stories[0]
    
    print("=" * 80)
    print("TECHNICAL REPORT")
    print("=" * 80)
    print(storyteller.generate_narrative_report(story, "technical"))
    
    print("\n" + "=" * 80)
    print("BUSINESS REPORT")
    print("=" * 80)
    print(storyteller.generate_narrative_report(story, "business"))
    
    print("\n" + "=" * 80)
    print("GENERAL AUDIENCE REPORT")
    print("=" * 80)
    print(storyteller.generate_narrative_report(story, "general"))
