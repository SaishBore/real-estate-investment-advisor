"""
Step 1: Load the raw data, clean it, create new features, and split into train/test.
Run from the project root:  python src/step1_classification.py
"""

import pandas as pd

df = pd.read_csv("data/india_housing_prices.csv")

print(df.shape)    # how many rows and columns
print(df.head())   # first 5 rows
print(df.dtypes)   # type of each column

# Missing values: count per column
print(df.isnull().sum())

# Duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Duplicate IDs (each property should have a unique ID)
print("Duplicate IDs:", df["ID"].duplicated().sum())



#1.3: Fix problem columns


          # Problem 1: Price_per_SqFt
print(df[["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]].head())
print("Rows where Price_per_SqFt is 0:", (df["Price_per_SqFt"] == 0).sum())

  # Fix: recompute it in rupees from the exact price and size:
df["Price_per_SqFt_INR"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
df = df.drop(columns=["Price_per_SqFt"])

          # Problem 2: Floor_No larger than Total_Floors
bad = df["Floor_No"] > df["Total_Floors"]
print("Impossible rows:", bad.sum())          

df["Floor_Data_Fixed"] = bad.astype(int)
df["Total_Floors"] = df[["Floor_No", "Total_Floors"]].max(axis=1)

          # Problem 3: The messy Amenities column
print(df["Amenities"].head())
print("Unique values:", df["Amenities"].nunique())

  # Fix: turn each amenity into its own 1/0 column (1 = the property has it):
amenity_list = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]

for a in amenity_list:
    df["Has_" + a] = df["Amenities"].apply(lambda text: 1 if a in text else 0)

df["Amenity_Count"] = df[["Has_" + a for a in amenity_list]].sum(axis=1)
df = df.drop(columns=["Amenities"])

print(df[["Has_Gym", "Has_Pool", "Amenity_Count"]].head())



# 1.4: Encode the text columns


          # Part A: Save a readable copy, then encode simple columns
# Save the readable cleaned data (for EDA)
df.to_csv("data/cleaned_housing.csv", index=False)

# Make a separate copy for modeling
df_model = df.copy()

# Yes/No columns -> 1/0
df_model["Parking_Space"] = df_model["Parking_Space"].map({"Yes": 1, "No": 0})
df_model["Security"] = df_model["Security"].map({"Yes": 1, "No": 0})

# Columns with a natural order -> 1, 2, 3
df_model["Public_Transport_Accessibility"] = df_model["Public_Transport_Accessibility"].map(
    {"Low": 1, "Medium": 2, "High": 3})
df_model["Furnished_Status"] = df_model["Furnished_Status"].map(
    {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2})

# Safety check: a typo in a dictionary would create empty values
print(df_model[["Parking_Space", "Security", "Public_Transport_Accessibility", "Furnished_Status"]].isnull().sum())   

          # Part B: One-hot encode columns with no order
df_model = pd.get_dummies(
    df_model,
    columns=["Property_Type", "Facing", "Owner_Type", "Availability_Status"],
    drop_first=True,
    dtype=int,
)

print(df_model.shape)
print(df_model.columns.tolist())         



# 1.5: Create new features


# 1. Combined score for the neighborhood (each part scaled to 0-1)
df_model["Infrastructure_Score"] = (
    df_model["Nearby_Schools"] / 10
    + df_model["Nearby_Hospitals"] / 10
    + df_model["Public_Transport_Accessibility"] / 3
) / 3

# 2. Where in the building the flat is (0 = ground, 1 = top floor)
df_model["Floor_Position"] = df_model["Floor_No"] / df_model["Total_Floors"]

# 3. Average room size
df_model["Size_per_BHK"] = df_model["Size_in_SqFt"] / df_model["BHK"]

# 4. Is this property cheap or expensive for its city? (1.0 = typical)
city_median = df_model.groupby("City")["Price_per_SqFt_INR"].transform("median")
df_model["Price_vs_City_Median"] = df_model["Price_per_SqFt_INR"] / city_median

print(df_model[["Infrastructure_Score", "Floor_Position",
                "Size_per_BHK", "Price_vs_City_Median"]].describe().round(2))



# 1.6: Create the two targets


# Target 1: Estimated price after 5 years (regression)
# Assumption: growth is 5% per year, plus up to 5% more for better infrastructure
df_model["Annual_Growth_Rate"] = 0.05 + 0.05 * df_model["Infrastructure_Score"]
df_model["Future_Price_5Y"] = df_model["Price_in_Lakhs"] * (1 + df_model["Annual_Growth_Rate"]) ** 5

# Target 2: Good Investment (classification)
# Rule: priced at or below its city's median per sqft AND infrastructure at or above the median
cheap_for_city = df_model["Price_vs_City_Median"] <= 1.0
good_infra = df_model["Infrastructure_Score"] >= df_model["Infrastructure_Score"].median()
df_model["Good_Investment"] = (cheap_for_city & good_infra).astype(int)

print(df_model["Good_Investment"].value_counts())
print(df_model[["Price_in_Lakhs", "Annual_Growth_Rate", "Future_Price_5Y"]].head())



# 1.7: Finish encoding, split, and scale


          # Part A: Handle the location columns and a redundant column
import os

# ID is just a label. State is fully determined by City. Locality has 500 names
# reused identically in all 42 cities, so it carries no real information.
df_model = df_model.drop(columns=["ID", "State", "Locality"])

# City has 42 values, so one-hot encoding is reasonable
df_model = pd.get_dummies(df_model, columns=["City"], drop_first=True, dtype=int)

# Age_of_Property + Year_Built always equals 2025, so they carry the same information
df_model = df_model.drop(columns=["Age_of_Property"])

df_model.to_csv("data/processed_housing.csv", index=False)
print(df_model.shape)

          # Part B: Split into train and test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df_model, test_size=0.2, random_state=42, stratify=df_model["Good_Investment"]
)
print(train_df.shape, test_df.shape)
print(train_df["Good_Investment"].mean(), test_df["Good_Investment"].mean())

          # Part C: Scale the numeric columns
from sklearn.preprocessing import StandardScaler
import joblib

scale_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
              "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Price_per_SqFt_INR",
              "Amenity_Count", "Size_per_BHK", "Price_vs_City_Median",
              "Floor_Position", "Infrastructure_Score"]

train_df = train_df.copy()
test_df = test_df.copy()

scaler = StandardScaler()
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])   # learn from train only
test_df[scale_cols] = scaler.transform(test_df[scale_cols])         # apply the same scaling

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(train_df[scale_cols].mean().round(2).abs().max())
print(train_df[scale_cols].std().round(2).min(), train_df[scale_cols].std().round(2).max())
