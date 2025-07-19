import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class SalesPredictionAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_explore_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            self.data.columns = self.data.columns.str.strip()
            self.feature_names = self.data.columns[:-1].tolist()
            self.target_name = self.data.columns[-1]
            print(f"Dataset Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            print("\nFirst 5 rows:")
            print(self.data.head())
            print("\nData Info:")
            print(self.data.info())
            print("\nStatistical Summary:")
            print(self.data.describe())
            print("\nMissing Values:")
            print(self.data.isnull().sum())
            print(f"\nTarget Variable: {self.target_name}")
            print(f"Feature Variables: {self.feature_names}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def visualize_data(self):
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        plt.subplot(3, 3, 1)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        plt.subplot(3, 3, 2)
        sns.histplot(self.data[self.target_name], kde=True, color='skyblue')
        plt.title(f'Distribution of {self.target_name}')
        plt.xlabel(self.target_name)
        
        plt.subplot(3, 3, 3)
        self.data[self.feature_names].boxplot(ax=plt.gca())
        plt.title('Feature Distributions')
        plt.xticks(rotation=45)
        
        for i, feature in enumerate(self.feature_names[:3]):
            plt.subplot(3, 3, i+4)
            plt.scatter(self.data[feature], self.data[self.target_name], alpha=0.6, color='coral')
            plt.xlabel(feature)
            plt.ylabel(self.target_name)
            plt.title(f'{feature} vs {self.target_name}')
            z = np.polyfit(self.data[feature], self.data[self.target_name], 1)
            p = np.poly1d(z)
            plt.plot(self.data[feature], p(self.data[feature]), "r--", alpha=0.8)
        
        if len(self.feature_names) > 1:
            plt.subplot(3, 3, 7)
            plt.scatter(self.data[self.feature_names[0]], self.data[self.feature_names[1]], 
                       c=self.data[self.target_name], cmap='viridis', alpha=0.6)
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.feature_names[1])
            plt.title('Feature Relationship')
            plt.colorbar(label=self.target_name)
        
        plt.subplot(3, 3, 8)
        feature_corr = self.data[self.feature_names].corrwith(self.data[self.target_name]).abs()
        feature_corr.plot(kind='bar', color='lightgreen')
        plt.title('Feature Importance (Correlation)')
        plt.xticks(rotation=45)
        plt.ylabel('Absolute Correlation')
        
        plt.subplot(3, 3, 9)
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        lr = LinearRegression()
        lr.fit(X, y)
        predictions = lr.predict(X)
        residuals = y - predictions
        plt.scatter(predictions, residuals, alpha=0.6, color='orange')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nKey Insights:")
        print(f"Strongest correlation: {feature_corr.idxmax()} ({feature_corr.max():.3f})")
        print(f"Weakest correlation: {feature_corr.idxmin()} ({feature_corr.min():.3f})")
        print(f"Average {self.target_name}: {self.data[self.target_name].mean():.2f}")
        print(f"Standard deviation: {self.data[self.target_name].std():.2f}")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_models(self):
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = self.prepare_data()
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Polynomial Regression': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        }
        
        model_results = {}
        
        for name, model in models.items():
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            
            model_results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_predictions': test_pred
            }
            
            print(f"\n{name}:")
            print(f"Train R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.models = model_results
        self.test_data = (X_test_scaled, X_test, y_test)
        
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        self.best_model = model_results[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name} (Test R² = {self.best_model['test_r2']:.4f})")
        return model_results
    
    def hyperparameter_tuning(self):
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = self.prepare_data()
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        tuned_models = {}
        
        for model_name, param_grid in param_grids.items():
            if model_name == 'Random Forest':
                base_model = RandomForestRegressor(random_state=42)
                X_data = X_train
            elif model_name == 'Ridge Regression':
                base_model = Ridge()
                X_data = X_train_scaled
            elif model_name == 'Gradient Boosting':
                base_model = GradientBoostingRegressor(random_state=42)
                X_data = X_train
            
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_data, y_train)
            
            if model_name == 'Ridge Regression':
                test_pred = grid_search.predict(X_test_scaled)
            else:
                test_pred = grid_search.predict(X_test)
            
            test_r2 = r2_score(y_test, test_pred)
            
            tuned_models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_r2': test_r2
            }
            
            print(f"\n{model_name} Tuning:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        return tuned_models
    
    def model_comparison_plot(self):
        if not self.models:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.models.keys())
        test_r2_scores = [self.models[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.models[name]['test_rmse'] for name in model_names]
        
        axes[0, 0].bar(model_names, test_r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Performance (R² Score)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, test_rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Model Performance (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        X_test_scaled, X_test, y_test = self.test_data
        best_predictions = self.best_model['test_predictions']
        
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6, color='green')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title(f'Actual vs Predicted ({self.best_model_name})')
        
        residuals = y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.6, color='orange')
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'Residual Plot ({self.best_model_name})')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nModel Comparison:")
        print(f"{'Model':<20} {'Test R²':<12} {'Test RMSE':<12} {'CV Mean':<12} {'CV Std':<12}")
        print("-" * 80)
        for name in model_names:
            print(f"{name:<20} {self.models[name]['test_r2']:<12.4f} "
                  f"{self.models[name]['test_rmse']:<12.4f} "
                  f"{self.models[name]['cv_mean']:<12.4f} "
                  f"{self.models[name]['cv_std']:<12.4f}")
    
    def predict_sales(self, **kwargs):
        if not self.best_model:
            return None
        
        input_data = []
        for feature in self.feature_names:
            if feature.lower().replace(' ', '_') in kwargs:
                input_data.append(kwargs[feature.lower().replace(' ', '_')])
            else:
                mean_val = self.data[feature].mean()
                input_data.append(mean_val)
        
        input_array = np.array(input_data).reshape(1, -1)
        
        if self.best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            input_scaled = self.scaler.transform(input_array)
            prediction = self.best_model['model'].predict(input_scaled)[0]
        else:
            prediction = self.best_model['model'].predict(input_array)[0]
        
        test_rmse = self.best_model['test_rmse']
        confidence_lower = prediction - 1.96 * test_rmse
        confidence_upper = prediction + 1.96 * test_rmse
        
        print(f"\nPrediction Results:")
        print(f"Model: {self.best_model_name}")
        print(f"Input: {dict(zip(self.feature_names, input_data))}")
        print(f"Predicted {self.target_name}: {prediction:.2f}")
        print(f"95% Confidence Interval: [{confidence_lower:.2f}, {confidence_upper:.2f}]")
        
        return prediction
    
    def feature_importance(self):
        if not self.best_model:
            return
        
        model = self.best_model['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp, x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance ({self.best_model_name})')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
        elif hasattr(model, 'coef_'):
            coefficients = model.coef_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp, x='coefficient', y='feature', palette='coolwarm')
            plt.title(f'Feature Coefficients ({self.best_model_name})')
            plt.xlabel('Coefficient Value')
            plt.tight_layout()
            plt.show()
        
        print(feature_imp)
        return feature_imp
    
    def generate_insights(self):
        if not self.data is None:
            correlations = self.data[self.feature_names].corrwith(self.data[self.target_name])
            strongest_feature = correlations.abs().idxmax()
            strongest_corr = correlations[strongest_feature]
            
            print(f"\nKey Findings:")
            print(f"{strongest_feature} has strongest relationship with {self.target_name}")
            print(f"Correlation strength: {strongest_corr:.3f}")
            
            if strongest_corr > 0:
                print(f"Increasing {strongest_feature} increases {self.target_name}")
            else:
                print(f"Increasing {strongest_feature} decreases {self.target_name}")
            
            print(f"\nROI Analysis:")
            for feature in self.feature_names:
                if 'cost' in feature.lower() or 'spend' in feature.lower() or 'budget' in feature.lower():
                    avg_spend = self.data[feature].mean()
                    avg_sales = self.data[self.target_name].mean()
                    roi = (avg_sales / avg_spend) if avg_spend > 0 else 0
                    print(f"Average ROI for {feature}: {roi:.2f}x")
            
            print(f"\nRecommendations:")
            print(f"Focus budget on {strongest_feature} for maximum impact")
            print(f"Model accuracy: {self.best_model['test_r2']:.1%}")
            print(f"Use predictions for advertising spend optimization")

def main():
    file_path = r"C:\Users\ADMIN\Downloads\advertising.csv"
    analyzer = SalesPredictionAnalyzer(file_path)
    
    if not analyzer.load_and_explore_data():
        return
    
    analyzer.visualize_data()
    analyzer.train_models()
    analyzer.model_comparison_plot()
    analyzer.hyperparameter_tuning()
    analyzer.feature_importance()
    analyzer.generate_insights()
    
    example_prediction = analyzer.predict_sales(
        tv=100,
        radio=50,
        newspaper=25
    )
    
    print(f"\nAnalysis Complete!")
    print(f"Best model: {analyzer.best_model_name}")
    print(f"Model accuracy: {analyzer.best_model['test_r2']:.1%}")

if __name__ == "__main__":
    main()
