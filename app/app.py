"""
Step 5.3: Real Estate Investment Advisor (Streamlit app) with analytics tabs.
Put this file in the app/ folder
Run from the project root (the folder that contains models/ and data/):
    python -m streamlit run app/app.py
"""
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

# Make sure we were started from the right folder
if not os.path.exists("models/feature_columns.json"):
    st.error("Could not find the models folder. Open a terminal in your project folder "
             "(the one that contains `models/`) and run: `python -m streamlit run app/app.py`")
    st.stop()

from invest_predictor import (AMENITIES, CITY_MEDIANS, CITY_STATE, FEATURES,
                              classifier, predict, regressor)


@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_housing.csv")


# =========================================================== TAB 1: THE ADVISOR
def advisor_tab():
    st.write("Enter the details of a property to find out whether it looks like a good "
             "investment and what it might be worth in 5 years.")

    # ------------------------------------------------------------ 1. Location
    st.subheader("1. Location and type")
    c1, c2, c3 = st.columns(3)
    cities = sorted(CITY_STATE)
    city = c1.selectbox("City", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 0,
                        format_func=lambda c: f"{c} ({CITY_STATE[c]})")
    property_type = c2.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
    availability = c3.selectbox("Availability", ["Ready_to_Move", "Under_Construction"],
                                format_func=lambda s: s.replace("_", " "))

    # --------------------------------------------------------- 2. Size and price
    st.subheader("2. Size and price")
    c1, c2, c3, c4 = st.columns(4)
    bhk = c1.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
    size_sqft = c2.number_input("Size (sqft)", min_value=500, max_value=5000, value=1500, step=50)
    price_lakhs = c3.number_input("Current price (₹ lakhs)", min_value=10.0, max_value=500.0,
                                  value=150.0, step=5.0)
    year_built = c4.number_input("Year built", min_value=1990, max_value=2023, value=2015, step=1)
    st.caption(f"Price per sqft: ₹{price_lakhs * 100000 / size_sqft:,.0f}  |  "
               f"Median for {city}: ₹{CITY_MEDIANS[city]:,.0f}")

    # -------------------------------------------------------------- 3. Building
    st.subheader("3. Building")
    c1, c2, c3, c4 = st.columns(4)
    total_floors = c1.number_input("Total floors in building", min_value=1, max_value=30,
                                   value=10, step=1)
    floor_no = c2.number_input("Floor number (0 = ground)", min_value=0,
                               max_value=int(total_floors), value=min(3, int(total_floors)), step=1)
    facing = c3.selectbox("Facing", ["East", "North", "South", "West"])
    furnished = c4.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"], index=1)

    # ---------------------------------------------------------- 4. Neighborhood
    st.subheader("4. Neighborhood")
    c1, c2, c3 = st.columns(3)
    schools = c1.slider("Nearby schools", 1, 10, 5)
    hospitals = c2.slider("Nearby hospitals", 1, 10, 5)
    transport = c3.selectbox("Public transport access", ["Low", "Medium", "High"], index=1)

    # -------------------------------------------------------------- 5. Features
    st.subheader("5. Features and ownership")
    c1, c2, c3 = st.columns(3)
    with c1:
        parking = st.checkbox("Parking space", value=True)
        security = st.checkbox("Security", value=True)
    amenities = c2.multiselect("Amenities", AMENITIES, default=["Gym"])
    owner_type = c3.selectbox("Listed by", ["Owner", "Builder", "Broker"])

    # ------------------------------------------------------------------ Results
    if not st.button("Analyze investment", type="primary"):
        return

    inputs = {
        "city": city, "property_type": property_type, "bhk": int(bhk),
        "size_sqft": float(size_sqft), "price_lakhs": float(price_lakhs),
        "year_built": int(year_built), "furnished": furnished,
        "floor_no": int(floor_no), "total_floors": int(total_floors),
        "facing": facing, "schools": int(schools), "hospitals": int(hospitals),
        "transport": transport, "parking": parking, "security": security,
        "amenities": amenities, "owner_type": owner_type, "availability": availability,
    }
    result = predict(inputs)
    price_vs_city = (price_lakhs * 100000 / size_sqft) / CITY_MEDIANS[city] - 1

    st.divider()
    st.header("Results")
    left, middle, right = st.columns(3)

    with left:
        if result["good_investment"]:
            st.success("✅ Good investment")
            confidence = result["probability"]
        else:
            st.warning("⚠️ Not a strong investment")
            confidence = 1 - result["probability"]
        st.metric("Model confidence", f"{confidence * 100:.1f}%")

    with middle:
        st.metric("Estimated price after 5 years",
                  f"₹{result['future_price']:,.1f} lakhs",
                  delta=f"{result['total_growth_pct']:+.1f}% ({result['annual_growth_pct']:.1f}% per year)")

    with right:
        st.metric("Price per sqft vs city median", f"{price_vs_city * 100:+.0f}%",
                  help="Negative means cheaper than the typical property in this city.")

    st.progress(min(max(float(result["infrastructure_score"]), 0.0), 1.0),
                text=f"Neighborhood infrastructure score: {result['infrastructure_score']:.2f} out of 1.00")

    years = list(range(6))
    growth = result["annual_growth_pct"] / 100
    projection = pd.DataFrame(
        {"Estimated value (₹ lakhs)": [price_lakhs * (1 + growth) ** y for y in years]},
        index=pd.Index(years, name="Years from now"))
    st.line_chart(projection)

    with st.expander("How is this decided?"):
        st.write(
            "A property is rated a **good investment** when it is priced at or below the "
            "median price per sqft of its city **and** its neighborhood infrastructure "
            "score (schools, hospitals and public transport) is at or above the median. "
            "The 5-year price grows the current price at 5% a year plus up to 5% more "
            "for better infrastructure."
        )


# ======================================================= TAB 2: MARKET INSIGHTS
def insights_tab():
    if not os.path.exists("data/cleaned_housing.csv"):
        st.error("Could not find data/cleaned_housing.csv")
        return
    df = load_data()

    st.write("Explore the dataset. Leave a filter empty to include everything.")
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(df["State"].unique()))
    bhks = f2.multiselect("BHK", [1, 2, 3, 4, 5])
    price_lo, price_hi = f3.slider("Price range (₹ lakhs)", 10, 500, (10, 500))

    view = df[(df["Price_in_Lakhs"] >= price_lo) & (df["Price_in_Lakhs"] <= price_hi)]
    if states:
        view = view[view["State"].isin(states)]
    if bhks:
        view = view[view["BHK"].isin(bhks)]

    if view.empty:
        st.warning("No properties match these filters.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Properties", f"{len(view):,}")
    m2.metric("Average price", f"₹{view['Price_in_Lakhs'].mean():,.1f} lakhs")
    m3.metric("Average price per sqft", f"₹{view['Price_per_SqFt_INR'].mean():,.0f}")

    # ---- Price per sqft by state
    by_state = view.groupby("State")["Price_per_SqFt_INR"].mean().sort_values(ascending=False)
    st.subheader("Average price per sqft by state")
    st.bar_chart(by_state)
    if len(by_state) > 1:
        gap = (by_state.max() - by_state.min()) / by_state.mean() * 100
        st.info(f"The highest and lowest states differ by only **{gap:.1f}%**. "
                "Location makes very little difference in this dataset.")

    # ---- Heatmap: state x BHK
    st.subheader("Heatmap: average price per sqft by state and BHK")
    heat = view.pivot_table(index="State", columns="BHK", values="Price_per_SqFt_INR", aggfunc="mean")
    st.dataframe(heat.style.background_gradient(cmap="Blues", axis=None).format("{:,.0f}"),
                 height=420)
    st.caption("Darker blue = higher price per sqft. Colors are stretched to the range of the "
               "data, so small differences look bigger than they are.")

    # ---- Two charts side by side
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 cities by average price")
        top_cities = view.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_cities)
    with c2:
        st.subheader("Price per sqft by year built")
        blocks = (view["Year_Built"] // 5 * 5).rename("Built (5-year block)")
        trend = view.groupby(blocks)["Price_per_SqFt_INR"].mean()
        st.line_chart(trend)


# ========================================================== TAB 3: MODEL DETAILS
def model_tab():
    st.write(f"**Investment classifier:** {type(classifier).__name__}   |   "
             f"**Price forecaster:** {type(regressor).__name__}")
    st.write("These charts show which inputs each model relies on most. "
             "All six trained models are compared in the MLflow dashboard.")

    c1, c2 = st.columns(2)
    for col, model, title in [(c1, classifier, "Good Investment rating"),
                              (c2, regressor, "5-year price forecast")]:
        with col:
            st.subheader(f"Top inputs: {title}")
            if not hasattr(model, "feature_importances_"):
                st.write("Feature importance is not available for this model type.")
                continue
            imp = pd.Series(model.feature_importances_,
                            index=[f.replace("_", " ") for f in FEATURES])
            st.bar_chart(imp.sort_values(ascending=False).head(10))

    st.info("The models rely on the same few inputs that were used to build the targets "
            "(price, price per sqft relative to the city, and the infrastructure score). "
            "That is why they score so highly: they have learned the project's rules, "
            "not real-market behavior.")


# ====================================================================== PAGE
st.title("🏠 Real Estate Investment Advisor")
tab1, tab2, tab3 = st.tabs(["Advisor", "Market insights", "About the model"])
with tab1:
    advisor_tab()
with tab2:
    insights_tab()
with tab3:
    model_tab()

st.caption("These results come from a model trained on a synthetic dataset using rules "
           "defined for this project. They are for learning purposes and are not financial advice.")