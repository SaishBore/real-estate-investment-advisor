"""
Step 3.1: Train and compare classifiers for "Good Investment".
Run from the project root:  python src/step3_classification.py
"""
import json
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

# 1. Load the prepared data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 2. Choose the input columns (Option A: everything a user could enter in the app)
TARGETS = ["Good_Investment", "Future_Price_5Y", "Annual_Growth_Rate"]
EXCLUDE = TARGETS + ["Floor_Data_Fixed", "Floor_Position"]   # targets + weak/cleaning columns
FEATURES = [c for c in train.columns if c not in EXCLUDE]

os.makedirs("models", exist_ok=True)
with open("models/feature_columns.json", "w") as f:
    json.dump(FEATURES, f)            # the Streamlit app will need this exact list

X_train, y_train = train[FEATURES], train["Good_Investment"]
X_test, y_test = test[FEATURES], test["Good_Investment"]
print(f"Using {len(FEATURES)} input columns")
print(f"Baseline (always answer 'No'): {1 - y_test.mean():.4f}\n")

# 3. Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                            n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=42, eval_metric="logloss"),
}

# 4. Train, evaluate, keep the best
results = []
best_f1, best_name, best_model = -1, None, None
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "Seconds": time.time() - start,
    }
    results.append(row)
    if row["F1"] > best_f1:
        best_f1, best_name, best_model, best_pred = row["F1"], name, model, pred

print(pd.DataFrame(results).round(4).to_string(index=False))
print(f"\nBest model: {best_name}")
print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, best_pred))

joblib.dump(best_model, "models/best_classifier.joblib")
print("Saved models/best_classifier.joblib")