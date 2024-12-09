# BMS-URL-ANALYSIS
# Book My Show Dataset - Ensemble Classification Model

This project demonstrates an ensemble classification model built using various machine learning techniques to predict the result of a booking event in a "Book My Show" dataset. The dataset consists of various features related to booking events, and the goal is to predict whether a booking is successful or not based on the given data.

The project uses a combination of LightGBM, Logistic Regression, and Random Forest classifiers in an ensemble model to improve performance.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [ROC Curve and AUC](#roc-curve-and-auc)
- [How to Use](#how-to-use)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## **Project Overview**

This project involves:
1. **Data Exploration**: Checking for null values and unique elements in each feature.
2. **Data Preprocessing**: Dropping irrelevant columns, splitting features and target, and normalizing the data.
3. **Modeling**: Using three different classifiers: LightGBM, Logistic Regression, and Random Forest. The models are combined in an ensemble Voting Classifier.
4. **Model Evaluation**: Evaluating the model's performance using accuracy, classification reports, confusion matrix, and ROC-AUC curve.

---

## **Dependencies**

The following libraries are required to run this project:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `joblib`
  
 ---

## **Data Preprocessing**
- The dataset is loaded into a Pandas DataFrame.
- Irrelevant features (like popUpWidnow) are dropped from the dataset.
- The dataset is then split into features (X) and the target variable (y).
- The dataset is further split into training and testing sets using a train_test_split (80% training and 20% testing).
- Features are standardized using StandardScaler to improve model performance.
  
---

## **Modeling**
### Classifiers Used:
- **LightGBM Classifier**: A gradient boosting framework that is known for its speed and efficiency.
- **Logistic Regression**: A basic linear model used for binary classification tasks.
- **Random Forest Classifier**: A popular ensemble method that uses multiple decision trees for classification.

These models are combined in a Voting Classifier using soft voting (predicting based on class probabilities). This allows the model to consider multiple classifiers and make more robust predictions.

---

## Model Evaluation
- **Cross-Validation**: Stratified K-Fold cross-validation is used to evaluate each individual classifier (LightGBM, Logistic Regression, and Random Forest) to measure their performance on different subsets of the data.
- **Final Evaluation**: The model is evaluated on the test set using the following metrics:
Accuracy
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix
ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

---

## ROC Curve and AUC
The ROC curve plots the true positive rate (sensitivity) against the false positive rate (1-specificity) for various threshold values. The AUC (Area Under the Curve) is calculated to measure the model's ability to discriminate between the positive and negative classes.
