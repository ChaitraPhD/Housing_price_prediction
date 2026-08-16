"""
app.py
------
Streamlit web application for the Housing Price Prediction project.
Loads the tuned XGBoost model (models/xgb_final_model.pkl) and lets the
user enter raw house attributes; the same feature engineering used in
training (src/preprocessing.py) is applied before predicting.

Run with:  streamlit run app.py
"""

import sys
import os
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import BINARY_COLS, FEATURE_COLUMNS  # noqa: E402

MODEL_PATH = "models/xgb_final_model.pkl"
METRICS_PATH = "outputs/metrics_report.json"

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def engineer_single_input(area, bedrooms, bathrooms, stories, mainroad,
                           guestroom, basement, hotwaterheating, airconditioning):
    """Mirror src/preprocessing.py feature engineering for one input row."""
    row = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": 1 if mainroad else 0,
        "guestroom": 1 if guestroom else 0,
        "basement": 1 if basement else 0,
        "hotwaterheating": 1 if hotwaterheating else 0,
        "airconditioning": 1 if airconditioning else 0,
    }
    total_rooms = row["bedrooms"] + row["bathrooms"]
    row["TotalSF"] = row["area"]
    row["Bathroom_Count"] = row["bathrooms"]
    row["Total_Rooms"] = total_rooms
    row["Area_per_Room"] = row["area"] / total_rooms if total_rooms else row["area"]
    row["Story_Area_Ratio"] = row["area"] / row["stories"] if row["stories"] else row["area"]
    row["Amenity_Score"] = sum(row[c] for c in BINARY_COLS)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def main():
    st.title("🏠 Housing Price Prediction")
    st.caption(
        "End-to-end ML pipeline — data cleaning, feature engineering, "
        "Linear Regression / Random Forest / XGBoost comparison, "
        "hyperparameter-optimized XGBoost, and SHAP explainability."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model found at `models/xgb_final_model.pkl`. "
            "Run `python src/main_pipeline.py` first to train and save it."
        )
        return

    model = load_model()
    metrics = load_metrics()

    st.header("Enter house attributes")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq. ft.)", min_value=500, max_value=20000, value=5000, step=50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        stories = st.slider("Stories", 1, 4, 2)
    with col2:
        mainroad = st.checkbox("On main road", value=True)
        guestroom = st.checkbox("Has guest room", value=False)
        basement = st.checkbox("Has basement", value=False)
        hotwaterheating = st.checkbox("Has hot water heating", value=False)
        airconditioning = st.checkbox("Has air conditioning", value=True)

    if st.button("Predict Price", type="primary", use_container_width=True):
        X_input = engineer_single_input(
            area, bedrooms, bathrooms, stories, mainroad,
            guestroom, basement, hotwaterheating, airconditioning
        )
        prediction = model.predict(X_input)[0]

        st.success(f"### Estimated Price: ₹ {prediction:,.0f}")

        with st.expander("Why this price? (SHAP explanation for this prediction)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_input)

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[0], show=False, max_display=13)
            st.pyplot(fig, clear_figure=True)

        with st.expander("Engineered features used for this prediction"):
            st.dataframe(X_input.T.rename(columns={0: "value"}))

    st.divider()

    if metrics:
        st.header("Model performance")
        rows = []
        for name, m in metrics["baseline_results"].items():
            rows.append({"Model": name, "RMSE": round(m["RMSE"]), "R2": round(m["R2"], 4)})
        rows.append({
            "Model": "XGBoost (tuned, FINAL)",
            "RMSE": round(metrics["tuned_xgboost_results"]["RMSE"]),
            "R2": round(metrics["tuned_xgboost_results"]["R2"], 4),
        })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Top price-driving feature (SHAP): **{metrics['top_shap_feature']}**")


if __name__ == "__main__":
    main()
