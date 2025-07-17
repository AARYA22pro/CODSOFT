🚢 Titanic Survival Prediction

This project is part of my Machine Learning Internship at CodSoft. The goal is to build a model that predicts whether a passenger on the Titanic survived or not based on various features like age, sex, class, and fare.

It’s a classic beginner ML problem that helps explore data preprocessing, model comparison, hyperparameter tuning, and evaluation.

🎯 Objective

To analyze the Titanic passenger dataset and build multiple machine learning models that can predict passenger survival.
Key steps include:

Cleaning and preprocessing the data

Building classification models

Comparing model accuracy

Tuning hyperparameters for optimal performance

🧰 Tech Stack Used

Category	Tools / Libraries
Programming Language	Python 🐍
Data Analysis	pandas, numpy
Visualization	seaborn, matplotlib
ML Algorithms	Logistic Regression, SVM, Random Forest, XGBoost
Model Evaluation	Accuracy Score, Classification Report, Cross-Validation
Optimization	GridSearchCV (for Random Forest)
Others	scikit-learn, xgboost, joblib (optional for model saving)

🗂️ Dataset

Source: Titanic dataset (commonly from Kaggle or open ML repositories)

Target Variable: Survived (1 = survived, 0 = did not survive)

Features Used:

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize

Removed features: PassengerId, Name, Ticket, Cabin (due to high missing or irrelevant data)

🧪 Project Workflow

1. Data Cleaning & Preprocessing
Filled missing values in Age (median) and Embarked (mode)

Dropped irrelevant columns

Created new feature: FamilySize = SibSp + Parch + 1

Encoded categorical variables (Sex, Embarked) using LabelEncoder

2. Model Training & Evaluation
Built and compared four classification models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

XGBoost Classifier

Evaluated models using:

Accuracy

Classification Report

Bar plot comparing model performance

3. Model Optimization
Tuned Random Forest Classifier using GridSearchCV

Evaluated best model with 5-fold cross-validation

📊 Results

✅ Accuracy Scores:

Model	Accuracy
Logistic Regression	~0.78
SVM	~0.81
Random Forest	~0.83
XGBoost	~0.82

Final model (tuned Random Forest) achieved solid performance with cross-validation accuracy around 83% ± 2%

📈 Visuals

Bar chart comparing model accuracy

Classification reports with precision, recall, and F1-score

Cross-validation output
(Optional) Save best model using joblib for deployment

🧠 Key Learnings

Data preprocessing is crucial for model performance

Different models have varying strengths; always compare

Hyperparameter tuning with GridSearchCV improves model accuracy

Visualizing results helps communicate model performance clearly

🤝 Acknowledgment

Big thanks to CodSoft for the internship opportunity and for encouraging hands-on machine learning experience through real-world datasets.
