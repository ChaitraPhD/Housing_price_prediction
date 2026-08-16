"""
explain.py
----------
Model explainability using SHAP (SHapley Additive exPlanations) for the
final tuned XGBoost model. Produces:
  - a global feature-importance summary (beeswarm) plot
  - a mean |SHAP value| bar chart
  - a dependence plot for the top feature
  - a waterfall plot explaining one individual prediction
All saved as PNGs into outputs/shap/.
"""

import os
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import build_dataset
from train import get_splits

OUT_DIR = "outputs/shap"
MODEL_PATH = "models/xgb_final_model.pkl"


def run_shap_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test = get_splits()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 1. Beeswarm summary plot (global feature impact + direction)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 2. Mean |SHAP value| bar chart (global importance ranking)
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_importance_bar.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 3. Dependence plot for the single most important feature
    mean_abs = shap_values.abs.mean(0).values
    top_feature_idx = mean_abs.argmax()
    top_feature = X_test.columns[top_feature_idx]

    plt.figure()
    shap.plots.scatter(shap_values[:, top_feature], show=False, color=shap_values)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_dependence_{top_feature}.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 4. Waterfall plot for one individual prediction (first test row)
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False, max_display=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_waterfall_example.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"SHAP analysis complete. Plots saved to {OUT_DIR}/")
    print(f"Top price-driving feature by mean |SHAP value|: {top_feature}")

    return shap_values, top_feature


if __name__ == "__main__":
    run_shap_analysis()
