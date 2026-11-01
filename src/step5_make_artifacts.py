"""
Step 5.1a: Save the small lookup files the app needs.
Run from the project root:  python src/step5_make_artifacts.py
"""
import json

import pandas as pd

df = pd.read_csv("data/cleaned_housing.csv")

# 1. Median price per sqft for each city (used for the "Price_vs_City_Median" feature)
city_medians = df.groupby("City")["Price_per_SqFt_INR"].median().to_dict()
with open("models/city_medians.json", "w") as f:
    json.dump(city_medians, f, indent=2)

# 2. City -> State, so the app can show the state next to the city
city_state = df.drop_duplicates("City").set_index("City")["State"].sort_index().to_dict()
with open("models/city_state.json", "w") as f:
    json.dump(city_state, f, indent=2)

print(f"Saved medians for {len(city_medians)} cities")
print(f"Example: Mumbai median = {city_medians.get('Mumbai', 'n/a')}")