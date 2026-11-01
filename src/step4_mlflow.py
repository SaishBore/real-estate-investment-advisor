"""
Step 4: Train all 6 models again, this time logging every run to MLflow.
Run from the project root:  python src/step4_mlflow.py
Then view results:          mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

# 1. MLflow setup: runs are stored in a local file called mlflow.db
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Real_Estate_Investment_Advisor")

# 2. Data (same files and input columns as before)
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
with open("models/feature_columns.json") as f:
    FEATURES = json.load(f)
X_train, X_test = train[FEATURES], test[FEATURES]

# 3. The six models
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                            n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=42, eval_metric="logloss"),
}
regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12,
                                           n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=42),
}


def simple_params(model):
    """Keep only plain settings (numbers, text, True/False) so they log cleanly."""
    return {k: v for k, v in model.get_params().items()
            if v is not None and isinstance(v, (int, float, str, bool))}


def save_model(model):
    try:
        mlflow.sklearn.log_model(model, artifact_path="model")
    except Exception as e:                     # never lose the metrics because of this
        print(f"   (model file not logged: {type(e).__name__})")


# 4. Classification runs
y_train, y_test = train["Good_Investment"], test["Good_Investment"]
for name, model in classifiers.items():
    with mlflow.start_run(run_name=f"classifier - {name}"):
        mlflow.set_tag("task", "classification")
        mlflow.log_param("model", name)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_params(simple_params(model))

        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "train_seconds": time.time() - start,
        }
        mlflow.log_metrics(metrics)
        save_model(model)
        print(f"classifier - {name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

# 5. Regression runs
y_train, y_test = train["Future_Price_5Y"], test["Future_Price_5Y"]
for name, model in regressors.items():
    with mlflow.start_run(run_name=f"regressor - {name}"):
        mlflow.set_tag("task", "regression")
        mlflow.log_param("model", name)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_params(simple_params(model))

        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": mean_absolute_error(y_test, pred),
            "r2": r2_score(y_test, pred),
            "train_seconds": time.time() - start,
        }
        mlflow.log_metrics(metrics)
        save_model(model)
        print(f"regressor - {name}: rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}")

print("\nDone. View the runs with:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")