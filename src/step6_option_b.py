"""
Step 6: Option B experiment.
Hide every input that was used to build the two targets, and see what the
remaining columns can predict on their own.
Run from the project root:  python step6_option_b.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
with open("models/feature_columns.json") as f:
    ALL_FEATURES = json.load(f)

# Everything the two targets were built from (or could be rebuilt from)
HIDDEN = (
    ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR", "Price_vs_City_Median",
     "Size_per_BHK", "Infrastructure_Score", "Nearby_Schools", "Nearby_Hospitals",
     "Public_Transport_Accessibility"]
    + [c for c in ALL_FEATURES if c.startswith("City_")]
)
FEATURES_B = [c for c in ALL_FEATURES if c not in HIDDEN]
print(f"Option A used {len(ALL_FEATURES)} inputs. Option B uses {len(FEATURES_B)}:")
print(", ".join(c for c in FEATURES_B if not c.startswith(("Facing_", "Owner_", "Property_", "Availability_"))),
      "+ property type, facing, owner type, availability\n")

X_train, X_test = train[FEATURES_B], test[FEATURES_B]

# ------------------------------------------------------------ Classification
y_train, y_test = train["Good_Investment"], test["Good_Investment"]
baseline = 1 - y_test.mean()
rows = []
for name, model in {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=42, eval_metric="logloss"),
}.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "F1": f1_score(y_test, pred, zero_division=0),
                 "ROC_AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])})
print("GOOD INVESTMENT without the label's inputs")
print(f"Baseline (always answer 'No'): accuracy {baseline:.4f}")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("(ROC_AUC of 0.5 means no better than guessing)\n")

# ---------------------------------------------------------------- Regression
y_train, y_test = train["Future_Price_5Y"], test["Future_Price_5Y"]
base_rmse = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
rows = []
for name, model in {
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=42),
}.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({"Model": name,
                 "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                 "R2": r2_score(y_test, pred)})
print("5-YEAR PRICE without the price and infrastructure inputs")
print(f"Baseline RMSE (always predict the average): {base_rmse:.2f} lakhs")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("(R2 near 0 means the inputs explain almost none of the variation)")