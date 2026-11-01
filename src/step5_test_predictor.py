"""
Step 5.1c: Check that the app's translator gives the same answers as the real data.
Run from the project root (the folder that contains data/ and models/).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

if not os.path.exists("app/invest_predictor.py"):
    sys.exit("Cannot find app/invest_predictor.py. Create a folder called 'app' in your "
             "project folder and put invest_predictor.py inside it.")
sys.path.insert(0, "app")          # so we can import app/invest_predictor.py
from invest_predictor import AMENITIES, build_features, classifier, predict, regressor

raw = pd.read_csv("data/cleaned_housing.csv")          # readable data, original row order
proc = pd.read_csv("data/processed_housing.csv")       # same rows, with the two targets

N = 2000
idx = raw.sample(N, random_state=1).index


def row_to_inputs(r):
    """Rebuild the form answers from one row of the raw data."""
    return {
        "city": r["City"], "property_type": r["Property_Type"], "bhk": int(r["BHK"]),
        "size_sqft": float(r["Size_in_SqFt"]), "price_lakhs": float(r["Price_in_Lakhs"]),
        "year_built": int(r["Year_Built"]), "furnished": r["Furnished_Status"],
        "floor_no": int(r["Floor_No"]), "total_floors": int(r["Total_Floors"]),
        "facing": r["Facing"], "schools": int(r["Nearby_Schools"]),
        "hospitals": int(r["Nearby_Hospitals"]), "transport": r["Public_Transport_Accessibility"],
        "parking": r["Parking_Space"] in ("Yes", 1), "security": r["Security"] in ("Yes", 1),
        "amenities": [a for a in AMENITIES if r[f"Has_{a}"] == 1],
        "owner_type": r["Owner_Type"], "availability": r["Availability_Status"],
    }


# 1. Build features for 2,000 real properties and predict them all at once
X = pd.concat([build_features(row_to_inputs(raw.loc[i])) for i in idx], ignore_index=True)
pred_class = classifier.predict(X)
pred_price = regressor.predict(X)

true_class = proc.loc[idx, "Good_Investment"].values
true_price = proc.loc[idx, "Future_Price_5Y"].values

# 2. Compare with the real answers
accuracy = (pred_class == true_class).mean()
errors = np.abs(pred_price - true_price)
print(f"Checked {N} real properties")
print(f"Good Investment: {accuracy * 100:.2f}% match ({(pred_class != true_class).sum()} wrong)")
print(f"Future price: average error {errors.mean():.3f} lakhs, worst error {errors.max():.3f} lakhs")

# 3. One example the way the app will call it
example = row_to_inputs(raw.loc[idx[0]])
start = time.time()
result = predict(example)
print(f"\nExample property: {example['city']}, {example['bhk']} BHK, "
      f"{example['size_sqft']:.0f} sqft, {example['price_lakhs']:.1f} lakhs")
print({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()})
print(f"One prediction took {time.time() - start:.2f} seconds")