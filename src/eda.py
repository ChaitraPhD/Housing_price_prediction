"""
eda.py
------
Exploratory Data Analysis: distributions, correlations, and price-driving
patterns. Saves all plots as PNGs into outputs/eda/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocessing import build_dataset

OUT_DIR = "outputs/eda"


def run_eda():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, df = build_dataset()
    full = X.copy()
    full["price"] = y.values

    sns.set_theme(style="whitegrid")

    # 1. Price distribution
    plt.figure(figsize=(7, 4.5))
    sns.histplot(full["price"], kde=True, color="#2b6cb0")
    plt.title("Distribution of House Price")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_distribution.png", dpi=130)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = full.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=130)
    plt.close()

    # 3. Price vs TotalSF
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=full, x="TotalSF", y="price", alpha=0.6, color="#2f855a")
    plt.title("Price vs Total Square Footage")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_vs_totalsf.png", dpi=130)
    plt.close()

    # 4. Price by bedrooms / bathrooms / stories
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sns.boxplot(data=full, x="bedrooms", y="price", ax=axes[0])
    axes[0].set_title("Price by Bedrooms")
    sns.boxplot(data=full, x="Bathroom_Count", y="price", ax=axes[1])
    axes[1].set_title("Price by Bathroom Count")
    sns.boxplot(data=full, x="stories", y="price", ax=axes[2])
    axes[2].set_title("Price by Stories")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_by_rooms.png", dpi=130)
    plt.close()

    # 5. Price by amenity score
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=full, x="Amenity_Score", y="price", palette="viridis")
    plt.title("Price by Amenity Score (mainroad, guestroom, basement,\n"
               "hotwaterheating, airconditioning combined)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_by_amenity_score.png", dpi=130)
    plt.close()

    # 6. Top correlations with price (bar chart)
    plt.figure(figsize=(7, 5))
    price_corr = corr["price"].drop("price").sort_values()
    colors = ["#e53e3e" if v < 0 else "#2b6cb0" for v in price_corr.values]
    price_corr.plot(kind="barh", color=colors)
    plt.title("Feature Correlation with Price")
    plt.xlabel("Correlation coefficient")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_correlation_ranked.png", dpi=130)
    plt.close()

    print(f"EDA complete. Plots saved to {OUT_DIR}/")
    print("\nTop correlations with price:")
    print(price_corr.sort_values(ascending=False))


if __name__ == "__main__":
    run_eda()
