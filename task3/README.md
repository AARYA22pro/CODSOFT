# Iris Flower Classification Project

A comprehensive machine learning pipeline for classifying Iris flowers using multiple algorithms with automated model selection, hyperparameter tuning, and detailed performance analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Pipeline Workflow](#pipeline-workflow)
- [Data Analysis](#data-analysis)
- [Results](#results)
- [Customization](#customization)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project implements a complete machine learning pipeline for the classic Iris flower classification problem. The pipeline automatically trains multiple models, compares their performance, selects the best one, and performs hyperparameter tuning for optimal results.

The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The goal is to classify flowers into 3 species:
- Setosa
- Versicolor
- Virginica

## Features

- **Automated Data Loading**: Load from sklearn.datasets or external CSV files
- **Comprehensive Data Exploration**: Statistical analysis, visualizations, and correlation analysis
- **Multiple ML Models**: Train and compare 5 different algorithms simultaneously
- **Automatic Model Selection**: Select best performing model based on accuracy metrics
- **Hyperparameter Tuning**: GridSearchCV optimization for Random Forest, SVM, and KNN
- **Detailed Evaluation**: Classification reports, confusion matrices, feature importance
- **Rich Visualizations**: Pairplots, correlation heatmaps, boxplots, confusion matrices
- **Cross-Validation**: 5-fold cross-validation for robust performance estimation
- **Feature Scaling**: StandardScaler preprocessing for optimal model performance

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Download the Project

1. Download the `irs.py` file (note: filename is `irs.py`, not `iris.py`)
2. Place it in your desired directory
3. Open terminal/command prompt in that directory

## Usage

### Basic Usage

Run the complete pipeline:

```bash
python irs.py
```

This will execute the complete pipeline including:
1. Load the Iris dataset from sklearn
2. Perform exploratory data analysis with visualizations
3. Train 5 different models with cross-validation
4. Compare model performance and select the best one
5. Perform hyperparameter tuning on the best model
6. Generate detailed evaluation reports with predictions

### Using the Pipeline Class

```python
from irs import IrisClassificationPipeline

# Initialize pipeline
pipeline = IrisClassificationPipeline()

# Load data (sklearn dataset or custom CSV)
df = pipeline.load_data()  # Uses sklearn dataset
# df = pipeline.load_data('path/to/your/iris_dataset.csv')  # Custom CSV

# Explore data with visualizations
pipeline.explore_data()

# Prepare data with train/test split and scaling
X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = pipeline.prepare_data()

# Train all models
results = pipeline.train_models(X_train, X_test, y_train, y_test)

# Find and select best model
comparison_df = pipeline.find_best_model()

# Detailed evaluation of best model
pipeline.detailed_evaluation(X_test, y_test)

# Hyperparameter tuning
tuned_model = pipeline.hyperparameter_tuning(X_train, y_train)

# Make predictions with examples
predictions = pipeline.make_predictions(X_test, y_test)
```

## Project Structure

```
iris-classification/
│
├── irs.py                  # Main pipeline implementation
├── README.md               # This documentation file
└── requirements.txt        # Python dependencies (optional)
```

## Models

The pipeline implements and compares the following 5 machine learning models:

### 1. Random Forest Classifier
- **Parameters**: n_estimators=100, random_state=42
- **Hyperparameter Tuning**: n_estimators, max_depth, min_samples_split
- **Features**: Provides feature importance, handles overfitting well

### 2. Support Vector Machine (SVC)
- **Parameters**: random_state=42 (default RBF kernel)
- **Hyperparameter Tuning**: C, gamma, kernel (rbf, linear, poly)
- **Features**: Effective for high-dimensional data

### 3. Logistic Regression
- **Parameters**: random_state=42, max_iter=1000
- **Hyperparameter Tuning**: Not implemented (uses default parameters)
- **Features**: Linear classifier, provides probability estimates

### 4. K-Nearest Neighbors (KNN)
- **Parameters**: n_neighbors=3
- **Hyperparameter Tuning**: n_neighbors, weights, metric
- **Features**: Simple, non-parametric approach

### 5. Decision Tree Classifier
- **Parameters**: random_state=42
- **Hyperparameter Tuning**: Not implemented (uses default parameters)
- **Features**: Interpretable, provides feature importance

## Pipeline Workflow

The `IrisClassificationPipeline` class follows this workflow:

1. **Data Loading** (`load_data()`):
   - Loads sklearn iris dataset by default
   - Fallback to sklearn if CSV loading fails
   - Creates DataFrame with proper column names

2. **Data Exploration** (`explore_data()`):
   - Dataset overview and statistics
   - Missing value analysis
   - Species distribution
   - Creates multiple visualizations

3. **Data Preparation** (`prepare_data()`):
   - 70/30 train/test split with stratification
   - StandardScaler feature scaling
   - Returns both scaled and original data

4. **Model Training** (`train_models()`):
   - Trains all 5 models
   - Calculates accuracy, precision, recall, F1-score
   - Performs 5-fold cross-validation
   - Stores all results

5. **Model Selection** (`find_best_model()`):
   - Compares all models by accuracy
   - Selects best performing model
   - Creates comparison DataFrame

6. **Detailed Evaluation** (`detailed_evaluation()`):
   - Classification report for best model
   - Confusion matrix visualization
   - Feature importance plot (if applicable)

7. **Hyperparameter Tuning** (`hyperparameter_tuning()`):
   - GridSearchCV for Random Forest, SVM, and KNN only
   - Uses predefined parameter grids
   - Updates best model with tuned parameters

8. **Predictions** (`make_predictions()`):
   - Shows sample predictions vs actual
   - Displays confidence scores (if available)
   - Final accuracy calculation

## Data Analysis

### Statistical Analysis
- Dataset shape: (150, 5) - 150 samples, 4 features + target
- Feature information and data types
- Descriptive statistics for all numeric features
- Class distribution (50 samples per species)
- Missing value check (typically none)

### Visualizations Generated
1. **Species Distribution**: Bar chart showing balanced classes
2. **Pairplot**: Scatter plots of all feature combinations colored by species
3. **Correlation Matrix**: Heatmap showing feature relationships
4. **Feature Boxplots**: Distribution of each feature by species (2x2 grid)
5. **Confusion Matrix**: Heatmap showing prediction accuracy
6. **Feature Importance**: Bar chart (for Random Forest and Decision Tree)

## Results

### Typical Performance
Based on the 70/30 split with random_state=42:

- **Test Set Size**: 45 samples (30% of 150)
- **Training Set Size**: 105 samples (70% of 150)
- **Cross-Validation**: 5-fold CV on training data

### Model Comparison Metrics
Each model is evaluated on:
- **Accuracy**: Overall correct predictions
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score
- **CV Mean**: Mean cross-validation score
- **CV Std**: Standard deviation of CV scores

### Expected Results
The Iris dataset is well-separated, so most models achieve very high accuracy (often 95-100% on test set).

## Customization

### Adding New Models

Add to the `models_to_train` dictionary in `train_models()`:

```python
models_to_train = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Your New Model': YourModelClass(parameters),
    # ... existing models
}
```

### Adding Hyperparameter Tuning

Add to the `hyperparameter_tuning()` method:

```python
elif self.best_model_name == 'Your New Model':
    param_grid = {
        'param1': [value1, value2],
        'param2': [value3, value4]
    }
    base_model = YourModelClass()
```

### Custom Data Loading

The pipeline supports custom CSV files with the same structure as the iris dataset:
- 4 numeric feature columns
- 1 target column named 'species'

## Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

**Note**: The code uses `plt.style.use('seaborn-v0_8')` which requires seaborn 0.8+ compatibility.

## License

This project is open source and available under the MIT License.

## Notes

- The file is named `irs.py` (not `iris.py`)
- Uses `warnings.filterwarnings('ignore')` to suppress sklearn warnings
- Color palette set to "husl" for consistent visualizations
- Random state set to 42 for reproducible results
- Stratified sampling ensures balanced train/test splits