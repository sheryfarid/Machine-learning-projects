import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

df = pd.read_csv("diabetes.csv")

X = df.drop(columns='Outcome', axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("📊 TRAIN METRICS")
print("Accuracy:", accuracy_score(y_train, train_preds))
print("Precision:", precision_score(y_train, train_preds))
print("Recall:", recall_score(y_train, train_preds))
print("F1 Score:", f1_score(y_train, train_preds))
print("ROC AUC:", roc_auc_score(y_train, train_preds))

print("\n📊 TEST METRICS")
print("Accuracy:", accuracy_score(y_test, test_preds))
print("Precision:", precision_score(y_test, test_preds))
print("Recall:", recall_score(y_test, test_preds))
print("F1 Score:", f1_score(y_test, test_preds))
print("ROC AUC:", roc_auc_score(y_test, test_preds))

print("\n📝 Classification Report (Test Set):")
print(classification_report(y_test, test_preds))

# 9. Save model and scaler
# pickle.dump(model, open('model.pkl', 'wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

