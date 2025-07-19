📊 Sales Prediction Analyzer
This project presents a complete end-to-end machine learning pipeline for predicting sales using features like TV, Radio, and Newspaper advertising budgets. It includes data preprocessing, EDA, model training, evaluation, hyperparameter tuning, model interpretability, and sales forecasting with confidence intervals.

🔍 Project Overview
Objective:
To analyze and predict sales based on historical advertising data using regression models, visualize important patterns, and generate actionable business insights.

📁 Dataset
Dataset: Advertising Data (CSV)

Source: ISLR (Introduction to Statistical Learning)

Features:

TV: TV advertising budget

Radio: Radio advertising budget

Newspaper: Newspaper advertising budget
Target:

Sales: Product sales in units or revenue

📌 Ensure the dataset is downloaded and saved as advertising.csv in the root directory of this project.

⚙️ Features
📈 Automated data exploration and visualization

🔍 Multiple regression models:

Linear, Ridge, Lasso, Random Forest, Gradient Boosting, Polynomial Regression

🎯 Cross-validation, RMSE, R² score, MAE evaluations

🔄 Hyperparameter tuning using GridSearchCV

📊 Model comparison and visualizations

🧠 Feature importance and insights generation

📦 Prediction interface with confidence interval estimation

🧠 Models Used
LinearRegression

Ridge

Lasso

RandomForestRegressor

GradientBoostingRegressor

Polynomial Regression (degree=2)

🖼️ Visualizations
Correlation matrix

Histogram of sales

Feature scatter plots

Residual plots

Actual vs Predicted plot

Model performance comparison (R², RMSE)

Feature importances

🔧 How to Use

Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset from here and save it as:

bash
Copy
Edit
sales-prediction-analyzer/advertising.csv
Run the main script:

bash
Copy
Edit
python sales_prediction_analyzer.py
🧪 Example Prediction
python
Copy
Edit
analyzer.predict_sales(tv=100, radio=50, newspaper=25)
🧠 Key Insights
Identifies the most impactful advertising channel.

ROI estimation based on average spend vs average sales.

Recommends optimal budget allocation strategies.

🗂️ File Structure
bash
Copy
Edit
├── sales_prediction_analyzer.py   # Main script
├── advertising.csv                # Dataset (downloaded from link)
├── README.md                      # Project overview
└── requirements.txt               # Python dependencies
📦 Requirements
txt
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
Create it using:
pip freeze > requirements.txt



📃 License
This project is licensed under the MIT License - see the LICENSE file for details.




