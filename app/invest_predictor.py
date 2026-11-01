"""
Step 5.1b: Turn a few form answers into the exact 71 columns the models were trained on.
Put this file in the app/ folder (named invest_predictor.py). The app and the test script both import from it.

It repeats the Step 1 recipe: same encodings, same new features, same scaler.
"""
import json

import joblib
import pandas as pd

MODELS_DIR = "models"

# Everything below is loaded once, when this file is first imported
with open(f"{MODELS_DIR}/feature_columns.json") as f:
    FEATURES = json.load(f)
with open(f"{MODELS_DIR}/city_medians.json") as f:
    CITY_MEDIANS = json.load(f)
with open(f"{MODELS_DIR}/city_state.json") as f:
    CITY_STATE = json.load(f)

scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
classifier = joblib.load(f"{MODELS_DIR}/best_classifier.joblib")
regressor = joblib.load(f"{MODELS_DIR}/best_regressor.joblib")

# The scaler was fitted on these columns (one of them, Floor_Position, is not a model input)
SCALE_COLS = list(scaler.feature_names_in_)

AMENITIES = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
TRANSPORT = {"Low": 1, "Medium": 2, "High": 3}
FURNISHED = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}


def infrastructure_score(schools, hospitals, transport_level):
    """Same formula as Step 1.5: schools, hospitals, transport, each scaled to 0-1."""
    return (schools / 10 + hospitals / 10 + TRANSPORT[transport_level] / 3) / 3


def build_features(inp):
    """
    inp is a dictionary with these keys:
      city, property_type, bhk, size_sqft, price_lakhs, year_built, furnished,
      floor_no, total_floors, facing, schools, hospitals, transport,
      parking (True/False), security (True/False), amenities (list of names),
      owner_type, availability
    Returns a one-row table in the exact column order the models expect.
    """
    size, price, bhk = inp["size_sqft"], inp["price_lakhs"], inp["bhk"]
    total_floors = max(inp["floor_no"], inp["total_floors"])     # same repair as Step 1.3
    ppsf = price * 100000 / size                                  # price per sqft in rupees

    values = {
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Year_Built": inp["year_built"],
        "Furnished_Status": FURNISHED[inp["furnished"].lower()],
        "Floor_No": inp["floor_no"],
        "Total_Floors": total_floors,
        "Floor_Position": inp["floor_no"] / total_floors,
        "Nearby_Schools": inp["schools"],
        "Nearby_Hospitals": inp["hospitals"],
        "Public_Transport_Accessibility": TRANSPORT[inp["transport"]],
        "Parking_Space": int(inp["parking"]),
        "Security": int(inp["security"]),
        "Price_per_SqFt_INR": ppsf,
        "Amenity_Count": len(inp["amenities"]),
        "Infrastructure_Score": infrastructure_score(
            inp["schools"], inp["hospitals"], inp["transport"]),
        "Size_per_BHK": size / bhk,
        "Price_vs_City_Median": ppsf / CITY_MEDIANS[inp["city"]],
    }
    for a in AMENITIES:
        values[f"Has_{a}"] = int(a in inp["amenities"])

    # Start with every column at 0, then fill in what we know
    row = dict.fromkeys(list(dict.fromkeys(FEATURES + SCALE_COLS)), 0)
    row.update(values)

    # One-hot columns: only the chosen option gets a 1.
    # The first option of each group (e.g. Apartment) has no column, so it stays all zeros.
    for col in [f"Property_Type_{inp['property_type']}",
                f"Facing_{inp['facing']}",
                f"Owner_Type_{inp['owner_type']}",
                f"Availability_Status_{inp['availability']}",
                f"City_{inp['city']}"]:
        if col in row:
            row[col] = 1

    frame = pd.DataFrame([row])
    frame[SCALE_COLS] = scaler.transform(frame[SCALE_COLS])      # same scaling as training
    return frame[FEATURES]


def predict(inp):
    """Run both models for one property and return the results."""
    X = build_features(inp)
    probability = float(classifier.predict_proba(X)[0, 1])
    future_price = float(regressor.predict(X)[0])
    price = inp["price_lakhs"]
    return {
        "good_investment": probability >= 0.5,
        "probability": probability,
        "future_price": future_price,
        "total_growth_pct": (future_price / price - 1) * 100,
        "annual_growth_pct": ((future_price / price) ** (1 / 5) - 1) * 100,
        "infrastructure_score": infrastructure_score(
            inp["schools"], inp["hospitals"], inp["transport"]),
    }