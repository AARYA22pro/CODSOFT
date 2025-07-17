🎬 Movie Rating Prediction
This project uses machine learning techniques to predict the rating of Indian movies based on features like genre, director, actors, and runtime. Both regression and classification models are implemented to predict numeric ratings and classify them into categories (Low, Medium, High).

📌 Features

Regression using Random Forest Regressor

Classification using:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

Data preprocessing, encoding, and visualization

Model evaluation using:

Accuracy

F1 Score

Confusion Matrix

Feature importance via:

Random Forest

SHAP (SHapley Additive exPlanations)

🧠 Objective
To analyze historical IMDb movie data and build predictive models that:

Estimate movie ratings (numerical regression)

Classify movies into Low / Medium / High rating categories

📂 Dataset
File: IMDb Movies India.csv

Columns Used:

Genre

Director

Actors

Runtime

Rating

Dataset should be placed in the /content folder or adjust file_path accordingly.

📊 Algorithms Used
Task	Algorithms
Regression	Random Forest Regressor
Classification	Logistic Regression, Random Forest, SVM

🧪 Evaluation Metrics
R² Score

RMSE

Accuracy

F1 Score

Confusion Matrix

Feature Importance

SHAP Summary Plot

📈 Visualizations
Actual vs Predicted Ratings

Confusion Matrices

Accuracy & F1 Score Comparison

Feature Importance Bar Chart

SHAP Summary Plot

🚀 How to Run
bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn shap
Place your IMDb Movies India.csv file in the specified path and run the Python script.

📎 Sample Output
Regression:

R² Score: ~0.65

RMSE: ~0.80

Classification:

Accuracy and F1 Score comparison

SHAP plot highlighting top predictive features

🧠 Insights
Genre and Director have high influence on movie ratings.

Random Forest performs consistently well across both tasks.

SHAP reveals individual feature contributions to model decisions.

📌 Future Improvements
Incorporate more features like Votes, Year, Language

Hyperparameter tuning for models

Use deep learning models for improved prediction

Deploy as a web app using Streamlit or Flask
