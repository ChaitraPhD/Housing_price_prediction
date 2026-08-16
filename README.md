# Housing Price Prediction — End-to-End ML Pipeline

An end-to-end data science project: data cleaning → feature engineering →
EDA → model comparison (Linear Regression, Random Forest, XGBoost) →
hyperparameter optimization → SHAP explainability → a Streamlit app for
real-time predictions.

## Project structure

```
housing_price_prediction/
├── data/
│   └── Housing-selected-columns.csv     # raw input data (545 rows)
├── src/
│   ├── preprocessing.py                 # cleaning, encoding, feature engineering
│   ├── eda.py                           # exploratory data analysis → outputs/eda/
│   ├── train.py                         # baseline LR / RF / XGBoost training + eval
│   ├── optimize.py                      # RandomizedSearchCV tuning for XGBoost
│   ├── explain.py                       # SHAP explainability → outputs/shap/
│   └── main_pipeline.py                 # runs the whole pipeline end-to-end
├── models/
│   └── xgb_final_model.pkl              # final tuned model, saved via joblib
├── outputs/
│   ├── eda/                             # EDA charts (PNG)
│   ├── shap/                            # SHAP charts (PNG)
│   └── metrics_report.json              # all metrics + best hyperparameters
├── app.py                               # Streamlit web app
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1. Run the full pipeline (cleaning → EDA → training → tuning → SHAP → save model)
python src/main_pipeline.py

# 2. Launch the prediction app
streamlit run app.py
```

Run `main_pipeline.py` once before `streamlit run app.py` — the app loads
`models/xgb_final_model.pkl`, which the pipeline creates.

## Methodology

### 1. Data cleaning & missing values
`src/preprocessing.py::clean_data()` checks for nulls (this dataset has
none), fills any numeric gaps with the median and categorical gaps with the
mode, drops exact duplicates, and discards non-physical rows (e.g. zero or
negative area).

### 2. Categorical encoding
The five yes/no columns (`mainroad`, `guestroom`, `basement`,
`hotwaterheating`, `airconditioning`) are label-encoded to 1/0.

### 3. Feature engineering — adapted to this dataset
The brief asked for `TotalSF`, `House Age`, `Bathroom Count`,
`Garden Indicator`, and `Total Rooms`. This particular CSV only has 10
columns (`price, area, bedrooms, bathrooms, stories` + 5 yes/no amenity
flags) — there's no year-built or lot/garden column to build `House Age` or
a genuine `Garden Indicator` from, so those two were **not** fabricated.
Instead, the same idea (turn raw columns into stronger price signals) was
applied to what's actually here:

| Engineered feature | Formula | Plays the role of |
|---|---|---|
| `TotalSF` | `area` | Total square footage |
| `Bathroom_Count` | `bathrooms` | Bathroom count (kept as-is) |
| `Total_Rooms` | `bedrooms + bathrooms` | Total rooms |
| `Area_per_Room` | `area / Total_Rooms` | Space efficiency |
| `Story_Area_Ratio` | `area / stories` | Footprint per story |
| `Amenity_Score` | sum of the 5 yes/no flags | Comfort/luxury indicator (stands in for Garden Indicator) |

If you swap in a richer dataset with `yr_built` and a lot/garden column
later, add `House_Age = current_year - yr_built` and a real
`Garden_Indicator` to `engineer_features()` — the rest of the pipeline
(training, tuning, SHAP, the app) will keep working as long as
`FEATURE_COLUMNS` is updated to match.

### 4. EDA
`src/eda.py` generates: price distribution, a full correlation heatmap,
price vs. `TotalSF` scatter, price by bedrooms/bathrooms/stories boxplots,
price by `Amenity_Score`, and a ranked bar chart of each feature's
correlation with price. All saved to `outputs/eda/`.

**Top correlations with price:** `TotalSF` (0.54), `Amenity_Score` (0.52),
`Bathroom_Count` (0.52), `Total_Rooms` (0.51), `airconditioning` (0.45).

### 5. Model training & comparison
`src/train.py` splits the data 80/20, trains Linear Regression (on scaled
features), Random Forest Regressor, and a baseline XGBoost Regressor, and
evaluates each with **RMSE** and **R²** on the held-out test set.

### 6. Hyperparameter optimization
`src/optimize.py` runs **RandomizedSearchCV** (60 iterations, 5-fold CV,
scored on negative RMSE) over XGBoost's `n_estimators`, `max_depth`,
`learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`,
`reg_alpha`, `reg_lambda`, and `gamma`. The best estimator is refit and
saved with `joblib.dump()` to `models/xgb_final_model.pkl`.

### 7. SHAP explainability
`src/explain.py` runs `shap.TreeExplainer` on the tuned model and produces:
a global beeswarm summary, a mean-|SHAP value| importance bar chart, a
dependence plot for the top feature, and a waterfall plot explaining one
individual prediction. The Streamlit app also generates a **live waterfall
explanation for every prediction the user makes**, so the app is
interpretable, not just a black-box number.

### 8. Streamlit app
`app.py` collects raw house attributes through a simple form, applies the
exact same feature engineering used in training, predicts with the saved
model, and shows a SHAP waterfall explaining that specific prediction plus
a model-comparison table.

## Results — please read before presenting this as "XGBoost wins"

On this actual dataset (545 rows, 13 engineered features, 80/20 split),
here's what the pipeline measured — reported honestly rather than assumed:

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | ~1,405,000 | ~0.609 |
| Random Forest | ~1,482,000 | ~0.565 |
| XGBoost (baseline) | ~1,494,000 | ~0.559 |
| XGBoost (tuned) | ~1,502,000 | ~0.554 |

(Exact numbers are in `outputs/metrics_report.json` after you run the
pipeline — a different random seed or CV budget will shift them slightly.)

**On this specific dataset, Linear Regression actually edges out both tree
ensembles**, and tuning didn't improve XGBoost's test score here. This is a
realistic and common outcome, not a bug: with only ~545 rows and a fairly
linear price relationship (price scales steadily with size/rooms/amenities),
simpler models often generalize better than boosted trees, which need more
data to earn their extra flexibility. I kept XGBoost as the "final model" in
the pipeline/app because that's what the requested methodology calls for and
because it comes with SHAP explainability out of the box — but if your goal
is the lowest error rather than following that specific methodology, swap
`models/xgb_final_model.pkl` for a saved `LinearRegression` (with its
`StandardScaler`) instead. If you want XGBoost to actually win, the more
data you can add (this looks like a subset of a larger public Housing
dataset with `parking`, `prefarea`, and `furnishingstatus` columns) is the
highest-leverage next step — more rows and more features is exactly where
gradient boosting starts to outperform a straight line.

## Requirements

See `requirements.txt`. Core: `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`shap`, `joblib`, `streamlit`, `matplotlib`, `seaborn`.
