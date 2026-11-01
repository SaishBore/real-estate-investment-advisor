"""
Step 3.2: Train and compare regressors for "Price after 5 years" (in lakhs).
Run from the project root:  python src/step3_regression.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# 1. Load the data and the SAME input columns used by the classifier
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
with open("models/feature_columns.json") as f:
    FEATURES = json.load(f)

X_train, y_train = train[FEATURES], train["Future_Price_5Y"]
X_test, y_test = test[FEATURES], test["Future_Price_5Y"]

# Baseline: always predict the average future price
baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
print(f"Using {len(FEATURES)} input columns")
print(f"Baseline RMSE (always predict the average): {baseline_rmse:.2f} lakhs\n")

# 2. Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12,
                                           n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=42),
}

# 3. Train, evaluate, keep the best (lowest RMSE)
results = []
best_rmse, best_name, best_model, best_pred = float("inf"), None, None, None
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    row = {
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
        "Seconds": time.time() - start,
    }
    results.append(row)
    if row["RMSE"] < best_rmse:
        best_rmse, best_name, best_model, best_pred = row["RMSE"], name, model, pred

print(pd.DataFrame(results).round(4).to_string(index=False))
print(f"\nBest model: {best_name}")

# 4. Sanity check: real vs predicted for 5 test properties
check = pd.DataFrame({"Actual": y_test.values[:5], "Predicted": best_pred[:5]}).round(2)
print("\nActual vs predicted (lakhs):")
print(check.to_string(index=False))

# 5. Which inputs mattered most (tree models only)
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop 5 most important inputs:")
    print(imp.head(5).round(4).to_string())

joblib.dump(best_model, "models/best_regressor.joblib")
print("\nSaved models/best_regressor.joblib")