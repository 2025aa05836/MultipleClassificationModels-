## Problem Statement

Implement multiple classification models - 
Build an interactive Streamlit web application to demonstrate your models - Deploy 
the app on Streamlit Community Cloud (FREE) - Share clickable links for evaluation

## Dataset Description

The dataset used for this project is the 'Adult' dataset (often referred to as the 'Census Income' dataset). It contains information extracted from the 1994 Census database. The prediction task is to determine whether a person makes over 50K a year or not.

Key features include:
- `age`: age of the individual.
- `workclass`: type of employment (e.g., Private, Self-emp-not-inc, Federal-gov).
- `fnlwgt`: final weight (the number of people the census estimates the row represents).
- `education`: highest level of education achieved.
- `education.num`: numerical representation of education level.
- `marital.status`: marital status of the individual.
- `occupation`: type of occupation.
- `relationship`: relationship status (e.g., Husband, Not-in-family, Own-child).
- `race`: race of the individual.
- `sex`: gender of the individual.
- `capital.gain`: capital gains.
- `capital.loss`: capital losses.
- `hours.per.week`: number of hours worked per week.
- `native.country`: country of origin.
- `income`: target variable, indicating whether income is >50K or <=50K.

## Models Used

*   **Logistic Regression**: A linear model used for binary classification, providing probabilities for predictions.
*   **Decision Tree Classifier**: A non-linear model that partitions the data into a tree-like structure based on features.
*   **K-Nearest Neighbor Classifier**: A non-parametric, instance-based learning algorithm that classifies data points based on the majority class of their k nearest neighbors.
*   **Gaussian Naive Bayes Classifier**: A probabilistic classifier based on Bayes' theorem, assuming features are normally distributed.
*   **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees and merges their predictions for improved accuracy.
*   **XGBoost Classifier**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable, known for its excellent performance.

## Model Evaluation Comparison

Here is a comparison of the evaluation metrics for all implemented models:

|     ML Model Name        | Accuracy   | AUC         | Precision   | Recall   | F1 Score   | MCC         |
|:-------------------------|:-----------|:------------|:------------|:---------|:-----------|:------------|
| Logistic Regression      | 0.7973     | 0.5949      | 0.7214      | 0.2577   | 0.3797     | 0.3448      |
| Decision Tree Classifier | 0.8133     | 0.7525      | 0.6073      | 0.6352   | 0.6209     | 0.4974      |
| K-Nearest Neighbor       | 0.7746     | 0.6757      | 0.5569      | 0.3119   | 0.3998     | 0.2919      |
| Gaussian Naive Bayes     | 0.7918     | 0.8268      | 0.6497      | 0.2934   | 0.4042     | 0.3341      |
| Random Forest            | 0.8518     | 0.8999      | 0.7296      | 0.611    | 0.665      | 0.5746      |
| XGBoost                  | 0.8689     | 0.9236      | 0.7684      | 0.6518   | 0.7053     | 0.6252      |

## Model Observations

Here are some observations regarding the performance of each model:

| ML Model Name            | Observation about model performance                                                                                 |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Moderate performance, with relatively low recall for the positive class.                                            |
| Decision Tree Classifier | Achieved a good balance between precision and recall, but may be prone to overfitting.                              |
| K-Nearest Neighbor       | Generally lower accuracy and F1-score compared to ensemble models, sensitive to feature scaling and dimensionality. |
| Gaussian Naive Bayes     | Decent accuracy, but struggled with precision for the positive class.                                               |
| Random Forest            | Strong performance overall, with high accuracy and good F1-score, indicating robustness.                            |
| XGBoost                  | Achieved the highest overall accuracy, AUC, and F1-score, making it the top-performing model.                       |
