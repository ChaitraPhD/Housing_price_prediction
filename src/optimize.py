"""
optimize.py
-----------
Hyperparameter optimization for the XGBoost Regressor using
RandomizedSearchCV (a randomized, budget-limited grid search) with 5-fold
cross-validation, scored on negative RMSE. The best estimator is refit on
the full training set and saved to models/xgb_final_model.pkl via joblib.
"""

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor

from preprocessing import build_dataset
from train import get_splits, evaluate, rmse, RANDOM_STATE

MODEL_PATH = "models/xgb_final_model.pkl"

PARAM_DIST = {
    "n_estimators": randint(150, 700),
    "max_depth": randint(2, 8),
    "learning_rate": uniform(0.01, 0.29),      # 0.01 - 0.30
    "subsample": uniform(0.6, 0.4),            # 0.6 - 1.0
    "colsample_bytree": uniform(0.6, 0.4),     # 0.6 - 1.0
    "min_child_weight": randint(1, 8),
    "reg_alpha": uniform(0, 1.0),
    "reg_lambda": uniform(0.5, 2.5),
    "gamma": uniform(0, 0.5),
}


def optimize_xgboost(n_iter: int = 60, cv_folds: int = 5, verbose: int = 1):
    X_train, X_test, y_train, y_test = get_splits()

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    test_metrics = evaluate(best_model, X_test, y_test)

    print("Best hyperparameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV RMSE: {-search.best_score_:,.0f}")
    print(f"Test RMSE:    {test_metrics['RMSE']:,.0f}")
    print(f"Test R2:      {test_metrics['R2']:.4f}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nFinal optimized model saved to {MODEL_PATH}")

    return best_model, search.best_params_, test_metrics


if __name__ == "__main__":
    optimize_xgboost()
