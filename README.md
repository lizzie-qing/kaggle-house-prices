# kaggle-house-prices
### House Prices: Advanced Regression Techniques
Predicting Home Values in Ames, Iowa — Kaggle Portfolio Project
# Overview

This project is part of the Kaggle competition “House Prices: Advanced Regression Techniques.”
The objective is to build a machine learning model that predicts the final sale price of homes in Ames, Iowa.

This notebook follows a professional, end-to-end Data Science workflow, including:

Exploratory Data Analysis (EDA)

Feature engineering

Preprocessing pipelines with ColumnTransformer

Multiple regression models (Lasso / Ridge / ElasticNet / XGBoost)

Feature importance visualization (XGBoost, Lasso, SHAP)

Ensemble modeling

Kaggle submission file (submission.csv)

This project is designed as a portfolio-quality machine learning example.

# Objectives

Understand the structure of the Ames Housing dataset

Handle missing values and categorical variables

Apply scaling and one-hot encoding using pipelines

Compare multiple regression models

Build a strong ensemble model for Kaggle

Export results to submission.csv

# 1. SalePrice Distribution

The target variable is heavily right-skewed.
To stabilize variance and improve model performance, log1p(SalePrice) is used during training.

Visualizations include:

Raw SalePrice histogram

Log-transformed SalePrice histogram

# 2. Data Overview

Key steps:

Display dataset shape

Check column types

Compute summary statistics

Identify missing values

Visualize missingness using heatmap

Analyze numerical vs categorical feature counts

This provides a foundation for feature engineering and modeling decisions.

# 3. Exploratory Data Analysis (EDA)

Major visual investigations:

Strong Predictors

OverallQual (strongest predictor)

GrLivArea

GarageCars / GarageArea

TotalBsmtSF

YearBuilt

Neighborhood Analysis

Median SalePrice differs greatly across neighborhoods.

Correlation Heatmap

Shows strong relationships among structural features and SalePrice.

# 4. Feature Engineering & Preprocessing

The pipeline includes:

### Handling Missing Values

Numerical: median

Categorical: most frequent

### Encoding

One-Hot Encoding for all categorical variables

Automatic handling of unseen categories in the test set

### Scaling

StandardScaler applied to numerical features

### Tools Used

ColumnTransformer

Pipeline

This ensures the full process is clean, modular, and reproducible.

# 5. Model Training & Validation

Data split:

80% training

20% validation

Models evaluated:

### Lasso Regression

Good for feature selection; stable performance.

### Ridge Regression

Performs very well on high-dimensional one-hot encoded data.

### XGBoost Regressor

Captures nonlinear interactions and complex relationships.

### Metrics:

Validation RMSE (on log-transformed target)

Lower = better

# 6. Feature Importance Visualization

This section includes 3 types of interpretability:

### 6.1 XGBoost Feature Importance

Shows the most influential structural and neighborhood features.

### 6.2 Lasso Coefficients

Highlights linear impact of each encoded feature.

### 6.3 SHAP Summary Plot

Global explanation of feature contributions

Helps understand true model behavior

These visualizations demonstrate deeper data science understanding beyond simple model fitting.

# 7. Final Model (Ensemble)

The final prediction uses a weighted ensemble:

50% Lasso

30% Ridge

20% ElasticNet

This ensemble method is a proven Kaggle strategy and improves robustness.

The final predictions are inverse-transformed using:
```
np.expm1(pred)
```
And saved as:
```
submission.csv
```
# 8. Kaggle Score

0.13472

# Tech Stack

Python
Pandas / NumPy
Seaborn / Matplotlib
Scikit-Learn
XGBoost
SHAP
Pipeline & ColumnTransformer

# Highlights (Recommended for Recruiters)

Clean end-to-end ML pipeline following industry best practices

Strong feature engineering + advanced preprocessing

Multiple model comparison with visualization

SHAP interpretability (highly valued in ML jobs)

Ensemble model with strong Kaggle performance

Fully reproducible code
