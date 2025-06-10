# Iris Data Analysis Project

This project provides a comprehensive pipeline for analyzing the classic Iris dataset. It includes modules for data loading, cleaning, feature engineering, statistical summary, pattern analysis, outlier detection, and both static and interactive visualizations.

## Features
- **Data Loading**: Load data from local files, Kaggle, or URLs with automatic validation
- **Data Integrity**: Assess and clean data for missing values, duplicates, and biological plausibility
- **Feature Engineering**: Create biologically meaningful features (ratios, areas) and scale them for analysis
- **Summary Statistics**: Generate detailed statistical summaries and domain-specific insights
- **Pattern Analysis**: Identify patterns, correlations, anomalies, and feature importance using multiple algorithms
- **Outlier Handling**: Detect and handle outliers using robust statistical methods (IQR, Z-score, Isolation Forest)
- **Static Visualizations**: Create publication-ready plots using Matplotlib and Seaborn
- **Interactive Visualizations**: Generate interactive HTML visualizations using Plotly
- **Data Storytelling**: Transform technical analysis into compelling narratives for different audiences

## File Structure
### Core Pipeline
- `main.py`: Complete analysis pipeline demonstrating all modules
- `main_storytelling.py`: Enhanced pipeline with comprehensive data storytelling
- `comprehensive_storytelling.py`: Focused storytelling analysis with multiple report formats
- `data_loader.py`: Data loading utilities with validation
- `data_integrity.py`: Data quality checks and cleaning procedures
- `feature_engineering.py`: Feature creation, transformation, and scaling
- `summary_stats.py`: Statistical summaries and domain insights
- `pattern_analysis.py`: Pattern detection, correlation analysis, and anomaly detection
- `outlier_handling.py`: Multi-method outlier detection and handling
- `visualization.py`: Static visualization suite (Matplotlib/Seaborn)
- `iris_visualizations.py`: Interactive visualization suite (Plotly)
- `data_storytelling.py`: Intelligent narrative generation and insight interpretation

### Data and Output
- `Iris.csv`: Classic Iris dataset
- `*.html`: Generated interactive visualization files
- `*_report.md`: Generated narrative reports for different audiences
- `*_insights.json`: Structured insights and findings data
- `__pycache__/`: Python compiled bytecode files

## Requirements
See `requirements.txt` for dependencies. Install with:

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis Pipeline
Execute the main pipeline that demonstrates all modules:
```bash
python main.py
```

### Generate Comprehensive Data Stories
Create intelligent narratives and insights from your data:
```bash
python comprehensive_storytelling.py
```

This generates:
- Technical report for data scientists (`iris_technical_report.md`)
- Business report for stakeholders (`iris_business_report.md`) 
- General audience report (`iris_general_report.md`)
- Structured insights data (`iris_insights.json`)

### Enhanced Analysis with Storytelling
Run the enhanced pipeline with integrated storytelling:
```bash
python main_storytelling.py
```

### Generate Interactive Visualizations
Create interactive HTML visualizations:
```bash
python iris_visualizations.py
```

### Individual Module Usage
You can also use individual modules for custom analysis:

```python
from data_loader import DataLoader
from data_storytelling import DataStoryteller
from pattern_analysis import PatternAnalysis

# Load and analyze data
loader = DataLoader("Iris.csv")
data = loader.load_data()

# Generate insights and stories
patterns = PatternAnalysis(data)
pattern_results = patterns.identify_patterns()
feature_importance = patterns.feature_importance()

# Create intelligent narratives
storyteller = DataStoryteller(data)
story = storyteller.analyze_and_tell_story(
    {}, pattern_results, {}, feature_importance
)

# Generate different audience reports
tech_report = storyteller.generate_narrative_report(story, "technical")
business_report = storyteller.generate_narrative_report(story, "business")
```

## Generated Outputs

### Narrative Reports
- `*_technical_report.md`: Detailed statistical analysis with code and methodology
- `*_business_report.md`: Executive summaries with strategic recommendations  
- `*_general_report.md`: Public-friendly explanations and insights
- `*_insights.json`: Structured data for further analysis or integration

### Interactive HTML Visualizations
- `parallel_coordinates.html`: Multi-dimensional feature relationships
- `3d_scatter.html`: 3D scatter plot in feature space
- `pca.html`: Principal Component Analysis projection
- `pair_plot.html`: Feature pair relationships matrix
- `radar.html`: Species feature distribution radar chart
- `box_swarm.html`: Distribution analysis with outliers
- `ratio_comparison.html`: Sepal vs Petal ratio analysis

### Static Visualizations
The main pipeline generates static plots including:
- Feature distributions and correlations
- PCA projections with species clustering
- Outlier analysis and detection results
- Statistical summary visualizations

## Sample Data Story Output

The system automatically generates insights like:

> **Key Finding**: Strong Relationship Between PetalWidthCm and petal_area  
> **Significance**: CRITICAL  
> **Description**: We discovered a positive correlation of 0.980 between PetalWidthCm and petal_area. This indicates that these features move together in a predictable pattern.  
> **Business Impact**: Strong correlation suggests these measurements are highly related, potentially allowing for cost-effective single-feature measurement strategies.  
> **Recommendation**: Consider feature selection techniques to reduce redundancy from highly correlated features.

## Key Features Implemented

### Data Analysis
- **Automated data validation** with biological plausibility checks
- **Feature engineering** including ratios (Sepal/Petal length-to-width) and areas
- **Multi-method outlier detection** (IQR, Z-score, Isolation Forest)
- **Pattern analysis** with correlation matrices and feature importance
- **Statistical summaries** with domain-specific insights

### Visualizations
- **Static plots** using Matplotlib/Seaborn for publication-ready figures
- **Interactive plots** using Plotly for exploratory data analysis
- **3D visualizations** for multi-dimensional feature relationships
- **PCA projections** for dimensionality reduction analysis

### Data Storytelling & Interpretation
- **Intelligent insight extraction** from technical analysis results
- **Multi-audience narratives** (technical, business, general public)
- **Automated recommendation generation** based on findings
- **Significance assessment** of statistical relationships
- **Business impact interpretation** of analytical results
- **Structured insight export** in JSON format for further processing

### Report Generation
- **Technical reports** with detailed statistical analysis and code
- **Business intelligence summaries** focused on actionable insights
- **General audience reports** with simplified language and explanations
- **Executive dashboards** with key findings and recommendations

## Notes
- The project is modular and can be extended to other tabular datasets with similar structure.
- Some features (e.g., Kaggle download) require additional setup (e.g., Kaggle API credentials).

## License
MIT License
