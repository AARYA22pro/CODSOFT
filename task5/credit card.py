import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class SimpleFraudDetection:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data = None
        self.model = LogisticRegression(random_state=self.random_state)
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        print("\n🔍 Loading dataset...")
        self.data = pd.read_csv(file_path)
        print(f"✅ Data loaded! Shape: {self.data.shape}")
        print(f"Fraudulent cases: {self.data['Class'].sum()}\n")

    def eda(self):
        print("📊 Running basic EDA...")
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        self.data['Class'].value_counts().plot.pie(autopct='%1.2f%%', labels=['Normal', 'Fraud'], colors=['skyblue', 'red'])
        plt.title('Class Distribution')

        plt.subplot(1, 3, 2)
        sns.histplot(data=self.data, x='Amount', hue='Class', bins=50, log_scale=(False, True), palette='Set1')
        plt.title('Transaction Amount Distribution')

        plt.subplot(1, 3, 3)
        corr = self.data.corr()['Class'].drop('Class').abs().sort_values(ascending=False).head(5)
        sns.barplot(x=corr.values, y=corr.index, palette='viridis')
        plt.title('Top Correlated Features')

        plt.tight_layout()
        plt.show()

    def preprocess(self):
        print("\n🔧 Preprocessing data...")
        X = self.data.drop(columns=['Class'])
        y = self.data['Class']
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        print(f"✅ Split done: Train = {len(self.X_train)}, Test = {len(self.X_test)}")

    def resample_data(self):
        print("\n⚖️  Applying SMOTE...")
        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"✅ Resampled: {np.bincount(self.y_train)}")

    def train_and_evaluate_model(self):
        print("\n🚀 Training and evaluating Logistic Regression model...")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix: Logistic Regression')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def run_pipeline(self, file_path):
        self.load_data(file_path)
        self.eda()
        self.preprocess()
        self.resample_data()
        self.train_and_evaluate_model()


# Example usage
if __name__ == "__main__":
    detector = SimpleFraudDetection()
    detector.run_pipeline("C:\Users\ADMIN\Downloads\creditcard.csv\creditcard.csv")
