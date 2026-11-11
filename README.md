# Task 5: Decision Trees and Random Forests

## Project Overview
This project explores tree-based machine learning models using the Breast Cancer Wisconsin dataset from Scikit-learn.

The objective is to classify tumors as malignant or benign.

## Tools and Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Tasks Performed
1. Loaded and explored the dataset.
2. Split data into training and testing sets.
3. Trained a Decision Tree Classifier.
4. Analyzed overfitting using training and testing accuracy.
5. Controlled tree complexity using `max_depth=3`.
6. Trained a Random Forest Classifier.
7. Compared model accuracy.
8. Evaluated both models using 5-fold cross-validation.
9. Identified the top 10 important features.
10. Visualized the Decision Tree and feature importances.

## Results

| Model | Training Accuracy | Testing Accuracy |
|---|---:|---:|
| Decision Tree | 100% | 91.23% |
| Decision Tree (depth 3) | 97.58% | 93.86% |
| Random Forest | 100% | 95.61% |

## Cross-Validation

| Model | Mean CV Accuracy |
|---|---:|
| Decision Tree (depth 3) | 91.91% |
| Random Forest | 95.61% |

## Important Features
The top features identified by Random Forest include:
- Worst area
- Worst concave points
- Worst radius
- Mean concave points
- Worst perimeter

## Visualizations
- `decision_tree.png`
- `feature_importance.png`

## Conclusion
Random Forest achieved the highest test accuracy and mean cross-validation accuracy among the evaluated models. Limiting Decision Tree depth improved its test accuracy and reduced overfitting.

This project is for educational purposes and is not a clinically validated diagnostic tool.