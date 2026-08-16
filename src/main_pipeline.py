"""
main_pipeline.py
-----------------
End-to-end orchestrator: cleaning -> feature engineering -> EDA ->
baseline model comparison -> XGBoost hyperparameter optimization ->
final model save (joblib) -> SHAP explainability.

Run with:  python src/main_pipeline.py
"""

import json
import joblib

from preprocessing import build_dataset
from eda import run_eda
from train import train_baselines, print_results, evaluate
from optimize import optimize_xgboost
from explain import run_shap_analysis

METRICS_PATH = "outputs/metrics_report.json"


def main():
    print("=" * 60)
    print("STEP 1: Load, clean, encode, engineer features")
    print("=" * 60)
    X, y, df = build_dataset()
    print(f"Dataset ready: {X.shape[0]} rows, {X.shape[1]} features")

    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    run_eda()

    print("\n" + "=" * 60)
    print("STEP 3: Train & compare baseline models")
    print("=" * 60)
    baseline_results, fitted, splits = train_baselines()
    print_results(baseline_results)

    print("\n" + "=" * 60)
    print("STEP 4: Hyperparameter optimization (XGBoost, RandomizedSearchCV)")
    print("=" * 60)
    best_model, best_params, tuned_metrics = optimize_xgboost(n_iter=60, cv_folds=5, verbose=0)

    print("\n" + "=" * 60)
    print("STEP 5: SHAP explainability on final model")
    print("=" * 60)
    shap_values, top_feature = run_shap_analysis()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    all_results = dict(baseline_results)
    all_results["XGBoost (tuned, FINAL)"] = tuned_metrics
    print_results(all_results)
    print(f"\nTop SHAP feature: {top_feature}")
    print(f"Final model saved to: models/xgb_final_model.pkl")

    report = {
        "baseline_results": baseline_results,
        "tuned_xgboost_results": tuned_metrics,
        "best_hyperparameters": best_params,
        "top_shap_feature": top_feature,
        "feature_columns": list(X.columns),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full metrics report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
