import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

file_path = r"C:\Users\ADMIN\Downloads\IMDb Movies India.csv\IMDb Movies India.csv"

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

df = df.dropna(subset=['Rating']).drop_duplicates()
if 'Genre' in df.columns:
    df['Genre'] = df['Genre'].fillna('')
    df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
else:
    df['Genre'] = 'Unknown'

if 'Director' in df.columns:
    df['Director'] = df['Director'].fillna('Unknown')
else:
    df['Director'] = 'Unknown'

if 'Actors' in df.columns:
    df['Actors'] = df['Actors'].fillna('Unknown')
    df['Actors'] = df['Actors'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
else:
    df['Actors'] = 'Unknown'

le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actors'] = le_actor.fit_transform(df['Actors'])

features = ['Genre', 'Director', 'Actors']
X = df[features]
y_reg = df['Rating']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

print("\nRandom Forest Regressor Performance:")
print(f"R² Score: {r2_score(y_test_r, y_pred_r):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.2f}")

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.6)
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--')
plt.title("Actual vs Predicted Ratings (Regression)")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(True)
plt.tight_layout()
plt.show()

def classify_rating(rating):
    if rating <= 5.0:
        return 'Low'
    elif rating <= 7.0:
        return 'Medium'
    else:
        return 'High'

df['Rating_Class'] = df['Rating'].apply(classify_rating)
le_class = LabelEncoder()
y_class_encoded = le_class.fit_transform(df['Rating_Class'])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train_c, y_train_c)
y_pred_log = log_model.predict(X_test_c)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict(X_test_c)

svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_c, y_train_c)
y_pred_svm = svm_clf.predict(X_test_c)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test_c, y_pred_log, target_names=le_class.classes_))
print("\nRandom Forest Classifier Classification Report:")
print(classification_report(y_test_c, y_pred_rf, target_names=le_class.classes_))
print("\nSVM Classifier Classification Report:")
print(classification_report(y_test_c, y_pred_svm, target_names=le_class.classes_))

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_class.classes_, yticklabels=le_class.classes_)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

print("Logistic Regression")
plot_confusion_matrix(confusion_matrix(y_test_c, y_pred_log), "Logistic Regression")
print("Random Forest")
plot_confusion_matrix(confusion_matrix(y_test_c, y_pred_rf), "Random Forest Classifier")
print("SVM")
plot_confusion_matrix(confusion_matrix(y_test_c, y_pred_svm), "SVM Classifier")

print("\nAccuracy & F1 Score Comparison:")
models = {
    "Logistic Regression": y_pred_log,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm
}

metrics = []
for name, preds in models.items():
    acc = accuracy_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds, average='macro')
    metrics.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1
    })
    print(f"{name} → Accuracy: {acc:.2f} | F1 Score: {f1:.2f}")

metrics_df = pd.DataFrame(metrics)

plt.figure(figsize=(10, 5))
bar_width = 0.35
index = np.arange(len(metrics_df))

plt.bar(index, metrics_df["Accuracy"], bar_width, label='Accuracy', color='skyblue')
plt.bar(index + bar_width, metrics_df["F1 Score"], bar_width, label='F1 Score', color='lightgreen')

plt.xlabel('Classifier')
plt.ylabel('Score')
plt.title('Classifier Comparison: Accuracy vs F1 Score')
plt.xticks(index + bar_width / 2, metrics_df["Model"])
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
