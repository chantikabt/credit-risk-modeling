# Credit Risk Modeling & Loan Default Prediction

## Overview
This project analyzes credit risk using the Home Credit Default Risk dataset. It applies exploratory data analysis (EDA), data preprocessing, and machine learning classification to predict applicants who may experience payment difficulties.

## Objectives
- Explore borrower characteristics and target distribution.
- Develop credit risk classification models.
- Compare Logistic Regression and Random Forest.
- Evaluate model performance using ROC-AUC, PR-AUC, precision, recall, and F1-score.
- Identify features that contribute to predictive performance.

## Tools & Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Google Colab

## Methodology
1. Data loading and data quality checks.
2. Exploratory Data Analysis.
3. Feature selection and preprocessing.
4. Stratified train-validation split.
5. Logistic Regression and Random Forest modeling.
6. Model evaluation and threshold comparison.
7. Feature importance analysis.

## Model Evaluation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.7418 | 0.7459 |
| PR-AUC | 0.2220 | 0.2250 |

Random Forest achieved slightly higher ROC-AUC and PR-AUC than Logistic Regression on the validation set.

## Key Findings
- Approximately 8.07% of validation records have `TARGET = 1`.
- Random Forest achieved an F1-score of approximately 0.290 at a threshold of 0.60.
- External source scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`) were important predictors.
- Threshold selection involves a trade-off between false positives and missed high-risk cases.

## Visualizations

### 1. Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The confusion matrix shows the number of correct and incorrect predictions made by the Random Forest model at a classification threshold of 0.60.

### 2. ROC Curve

![ROC Curve](roc_curve.png)

The ROC curve compares the ability of Logistic Regression and Random Forest to distinguish between the two target classes.

### 3. Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)

The Precision-Recall curve is particularly relevant because the target variable is imbalanced.

### 4. Feature Importance

![Feature Importance](feature_importance.png)

The feature importance analysis highlights variables that contribute to the predictive performance of the Random Forest model.

## Limitations
- The dataset comes from Home Credit and does not directly represent Indonesian fintech or Pindar lending data.
- `TARGET = 1` indicates payment difficulties according to the source dataset definition.
- Feature importance indicates predictive contribution, not causation.
- The selected threshold is provisional and requires further evaluation before operational use.

## Conclusion
Random Forest slightly outperformed Logistic Regression on the evaluated ranking metrics. However, further threshold optimization, validation, and assessment of business costs would be required before practical deployment.

## Project Files
- `credit_risk_modeling.ipynb`: Notebook containing data analysis, model development, and evaluation.

## Dataset
Home Credit Default Risk — available on Kaggle.
