"""
train.py
--------
Trains and evaluates Linear Regression, Random Forest Regressor and
XGBoost Regressor baselines. Reports RMSE and R^2 on a held-out test set
so the three algorithms can be compared before optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocessing import build_dataset

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_splits(test_size: float = 0.2):
    X, y, df = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, needs_scaling=False, scaler=None):
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    preds = model.predict(X_eval)
    return {
        "RMSE": rmse(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def train_baselines():
    X_train, X_test, y_train, y_test = get_splits()

    results = {}
    fitted = {}

    # --- Linear Regression (benefits from scaling) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    results["Linear Regression"] = evaluate(lr, X_test, y_test, needs_scaling=True, scaler=scaler)
    fitted["Linear Regression"] = (lr, scaler)

    # --- Random Forest Regressor ---
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate(rf, X_test, y_test)
    fitted["Random Forest"] = (rf, None)

    # --- XGBoost Regressor (default hyperparameters, pre-tuning) ---
    xgb = XGBRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    results["XGBoost (baseline)"] = evaluate(xgb, X_test, y_test)
    fitted["XGBoost (baseline)"] = (xgb, None)

    return results, fitted, (X_train, X_test, y_train, y_test)


def print_results(results: dict):
    print(f"{'Model':<22}{'RMSE':>15}{'R2':>10}")
    print("-" * 47)
    for name, m in results.items():
        print(f"{name:<22}{m['RMSE']:>15,.0f}{m['R2']:>10.4f}")


if __name__ == "__main__":
    results, fitted, splits = train_baselines()
    print_results(results)
