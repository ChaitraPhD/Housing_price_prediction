"""
preprocessing.py
-----------------
Data cleaning, missing-value handling, categorical encoding, and feature
engineering for the Housing Price Prediction project.

NOTE ON FEATURE ENGINEERING ADAPTATION
---------------------------------------
The original project brief asked for TotalSF, House Age, Bathroom Count,
Garden Indicator and Total Rooms. This dataset (Housing-selected-columns.csv)
only ships 10 columns: price, area, bedrooms, bathrooms, stories, mainroad,
guestroom, basement, hotwaterheating, airconditioning. There is no year-built
or lot/garden column, so "House Age" and "Garden Indicator" cannot be derived
without fabricating data. Instead, the engineered features below keep the
same spirit (encode size, comfort and layout signals the model can learn
price-driving patterns from) using only columns that genuinely exist:

    - TotalSF            -> the plot/floor `area` column, treated as the
                             total square footage measure (no separate
                             basement/1st/2nd-floor split is available)
    - Bathroom_Count      -> `bathrooms` (kept as-is, already a count)
    - Total_Rooms         -> bedrooms + bathrooms
    - Area_per_Room       -> area / Total_Rooms (space efficiency)
    - Story_Area_Ratio    -> area / stories (footprint per story)
    - Amenity_Score       -> count of "yes" among mainroad, guestroom,
                             basement, hotwaterheating, airconditioning
                             (a stand-in "comfort/luxury" indicator that
                             plays the role Garden Indicator would have)

If you later get a richer dataset with yr_built / lot / garden columns,
drop them into engineer_features() and the rest of the pipeline (training,
tuning, SHAP, the Streamlit app) will keep working unchanged as long as the
column names in FEATURE_COLUMNS below are updated to match.
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/Housing-selected-columns.csv"

BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning"]

TARGET = "price"

# Final feature set used to train every model in this project.
FEATURE_COLUMNS = [
    "TotalSF",
    "bedrooms",
    "Bathroom_Count",
    "stories",
    "Total_Rooms",
    "Area_per_Room",
    "Story_Area_Ratio",
    "Amenity_Score",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
]


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obvious data issues."""
    df = df.copy()

    # Report + handle missing values (dataset is clean, but this makes the
    # pipeline robust to future data drops that do have gaps)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = [c for c in df.columns if c not in num_cols]
        for c in num_cols:
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].mode()[0])

    # Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Guard against non-physical values
    for col in ["area", "bedrooms", "bathrooms", "stories", "price"]:
        if col in df.columns:
            df = df[df[col] > 0]

    return df.reset_index(drop=True)


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode yes/no categorical columns as 1/0."""
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0}).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features described in the module docstring."""
    df = df.copy()

    df["TotalSF"] = df["area"]
    df["Bathroom_Count"] = df["bathrooms"]
    df["Total_Rooms"] = df["bedrooms"] + df["bathrooms"]
    df["Area_per_Room"] = df["area"] / df["Total_Rooms"].replace(0, 1)
    df["Story_Area_Ratio"] = df["area"] / df["stories"].replace(0, 1)
    df["Amenity_Score"] = df[BINARY_COLS].sum(axis=1)

    return df


def build_dataset(path: str = RAW_PATH):
    """Full pipeline: load -> clean -> encode -> engineer -> split X/y."""
    df = load_data(path)
    df = clean_data(df)
    df = encode_binary_columns(df)
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return X, y, df


if __name__ == "__main__":
    X, y, df = build_dataset()
    print("Final feature matrix shape:", X.shape)
    print(X.head())
