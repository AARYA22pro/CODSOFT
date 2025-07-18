import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IrisClassificationPipeline:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, file_path=None):
        try:
            if file_path:
                self.df = pd.read_csv(file_path)
                print(f"Dataset loaded successfully from {file_path}")
            else:
                iris = load_iris()
                self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
                self.df['species'] = iris.target_names[iris.target]
                print("Dataset loaded from sklearn.datasets")
        except Exception as e:
            print(f"Error loading from file: {e}")
            print("Loading from sklearn.datasets instead...")
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['species'] = iris.target_names[iris.target]
        return self.df
    
    def explore_data(self):
        print("="*50)
        print("IRIS FLOWER DATASET EXPLORATION")
        print("="*50)
        print("\n1. Dataset Overview:")
        print(f"Shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        print("\n2. First 5 rows:")
        print(self.df.head())
        print("\n3. Dataset Info:")
        print(self.df.info())
        print("\n4. Statistical Summary:")
        print(self.df.describe())
        print("\n5. Species Distribution:")
        print(self.df['species'].value_counts())
        print("\n6. Missing Values:")
        print(self.df.isnull().sum())
        self.create_visualizations()
        
    def create_visualizations(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        self.df['species'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Species Distribution')
        axes[0,0].set_xlabel('Species')
        axes[0,0].set_ylabel('Count')
        plt.figure(figsize=(12, 8))
        sns.pairplot(self.df, hue='species', diag_kind='hist')
        plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 8))
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        features = [col for col in self.df.columns if col != 'species']
        for i, feature in enumerate(features):
            row, col = i//2, i%2
            sns.boxplot(data=self.df, x='species', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by Species')
            axes[row, col].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        
    def prepare_data(self):
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        feature_cols = [col for col in self.df.columns if col != 'species']
        X = self.df[feature_cols]
        y = self.df['species']
        print(f"Features: {feature_cols}")
        print(f"Target: species")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        print("\n" + "="*50)
        print("MODEL TRAINING AND EVALUATION")
        print("="*50)
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Support Vector Machine': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        results = {}
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Cross-validation: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        self.models = results
        return results
    
    def find_best_model(self):
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        comparison_data = []
        for name, metrics in self.models.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            })
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        print("Model Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]['model']
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        return comparison_df
    
    def detailed_evaluation(self, X_test, y_test):
        print("\n" + "="*50)
        print(f"DETAILED EVALUATION - {self.best_model_name}")
        print("="*50)
        y_pred = self.models[self.best_model_name]['predictions']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.best_model.classes_, yticklabels=self.best_model.classes_)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = [col for col in self.df.columns if col != 'species']
            importances = self.best_model.feature_importances_
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.show()
    
    def hyperparameter_tuning(self, X_train, y_train):
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42)
        elif self.best_model_name == 'Support Vector Machine':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
            base_model = SVC(random_state=42)
        elif self.best_model_name == 'K-Nearest Neighbors':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            base_model = KNeighborsClassifier()
        else:
            print(f"Hyperparameter tuning not implemented for {self.best_model_name}")
            return self.best_model
        print(f"Tuning {self.best_model_name}...")
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def make_predictions(self, X_test, y_test):
        print("\n" + "="*50)
        print("PREDICTION EXAMPLES")
        print("="*50)
        predictions = self.best_model.predict(X_test)
        probabilities = None
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_test)
        print("Sample Predictions:")
        print("-" * 80)
        print(f"{'Actual':<12} {'Predicted':<12} {'Correct':<8} {'Confidence':<12}")
        print("-" * 80)
        for i in range(min(10, len(predictions))):
            actual = y_test.iloc[i]
            predicted = predictions[i]
            correct = "✓" if actual == predicted else "✗"
            if probabilities is not None:
                confidence = np.max(probabilities[i])
                print(f"{actual:<12} {predicted:<12} {correct:<8} {confidence:<12.3f}")
            else:
                print(f"{actual:<12} {predicted:<12} {correct:<8} {'N/A':<12}")
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nOverall Test Accuracy: {accuracy:.4f}")
        return predictions

def main():
    print("🌸 IRIS FLOWER CLASSIFICATION PROJECT 🌸")
    print("="*60)
    pipeline = IrisClassificationPipeline()
    df = pipeline.load_data(None)
    pipeline.explore_data()
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = pipeline.prepare_data()
    results = pipeline.train_models(X_train, X_test, y_train, y_test)
    comparison_df = pipeline.find_best_model()
    pipeline.detailed_evaluation(X_test, y_test)
    tuned_model = pipeline.hyperparameter_tuning(X_train, y_train)
    final_predictions = pipeline.make_predictions(X_test, y_test)
    print("\n" + "="*60)
    print("🎉 IRIS CLASSIFICATION COMPLETE! 🎉")
    print("="*60)
    print(f"✅ Best Model: {pipeline.best_model_name}")
    print(f"✅ Final Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
    print(f"✅ Dataset Size: {df.shape[0]} samples")
    print(f"✅ Features Used: {df.shape[1]-1} features")
    print("="*60)

if __name__ == "__main__":
    main()
