# 💳 Credit Card Fraud Detection – Logistic Regression

This project is a part of my **Data Science Internship with CodSoft**, focusing on detecting fraudulent credit card transactions using Logistic Regression.

---

## 📌 Objective

To build a supervised machine learning model that can accurately detect fraudulent transactions from a highly imbalanced dataset.

---

## 📁 Dataset

The dataset contains transactions made by European cardholders in September 2013.  
- Features: V1 to V28 (PCA anonymized), Amount, and Time  
- Target: `Class` (0 = Normal, 1 = Fraud)

Download: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 🔧 Tech Stack

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Logistic Regression, preprocessing, model evaluation
- **Imbalanced-learn (SMOTE)** – Oversampling minority class

---

## 🔍 Workflow

1. **Data Loading & Exploration**
2. **EDA** – Class distribution, transaction patterns, feature correlation
3. **Preprocessing** – Scaling with `StandardScaler`
4. **Handling Imbalance** – Using `SMOTE` to upsample fraud class
5. **Model Training** – Logistic Regression
6. **Model Evaluation** – Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## 📈 Results

The model is evaluated using:
- **Precision** (to minimize false positives)
- **Recall** (to catch as many frauds as possible)
- **F1 Score** (harmonic mean of precision and recall)

---

## 🚀 Run the Project

```bash
python simple_fraud_detection.py
