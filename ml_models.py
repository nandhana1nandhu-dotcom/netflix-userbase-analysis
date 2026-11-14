"""
Machine Learning extension for the Netflix Userbase Analysis
------------------------------------------------------------
1. Supervised: can demographics + device predict Subscription Type?
   (compared against a majority-class baseline, with cross-validation)
2. Unsupervised: K-Means user segmentation.

Usage:
    python ml_models.py --data "data/Netflix Userbase.csv"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis import load_and_clean  # reuse the same cleaning step

OUT = Path("outputs")
SEED = 42

NUMERIC = ["Age", "Tenure Days"]
CATEGORICAL = ["Gender", "Country", "Device"]
# NOTE: "Monthly Revenue" is deliberately NOT a feature. It is determined by the
# subscription plan, so using it to predict the plan would be target leakage.


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


# ------------------------------------------------- 1. classification
def run_classification(df: pd.DataFrame) -> None:
    print("=" * 60, "\nCLASSIFICATION: predict Subscription Type\n" + "=" * 60)
    data = df.dropna(subset=NUMERIC + CATEGORICAL + ["Subscription Type"])
    X = data[NUMERIC + CATEGORICAL]
    y = data["Subscription Type"]

    models = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1),
    }

    # 5-fold stratified cross-validation: more reliable than one split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("5-fold cross-validated accuracy:")
    scores = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        scores[name] = s.mean()
        print(f"  {name:28s} {s.mean():.3f} (+/- {s.std():.3f})")

    best_name = max((n for n in scores if "Baseline" not in n), key=scores.get)
    lift = scores[best_name] - scores["Baseline (majority class)"]
    print(f"\nBest model: {best_name} | lift over baseline: {lift:+.3f}")
    if lift < 0.02:
        print("-> Barely better than guessing the most common plan: demographics and "
              "device carry little signal for plan type (consistent with the chi-squared tests).")

    # Hold-out evaluation for the best model
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = Pipeline([("prep", make_preprocessor()), ("model", models[best_name])])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    print(f"\nHold-out test report ({best_name}):")
    print(classification_report(y_te, pred, zero_division=0))

    OUT.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_te, pred, cmap="Reds")
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()


# ---------------------------------------------------- 2. clustering
def run_clustering(df: pd.DataFrame) -> None:
    print("=" * 60, "\nCLUSTERING: K-Means user segments\n" + "=" * 60)
    cols = ["Age", "Tenure Days", "Monthly Revenue", "Device", "Subscription Type"]
    data = df.dropna(subset=cols).copy()

    prep = ColumnTransformer([
        ("num", StandardScaler(), ["Age", "Tenure Days", "Monthly Revenue"]),
        ("cat", OneHotEncoder(), ["Device", "Subscription Type"]),
    ])
    X = prep.fit_transform(data[cols])

    # Choose k with silhouette score (higher = better separated clusters)
    ks, sil = range(2, 8), []
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        sil.append(silhouette_score(X, labels))
        print(f"  k={k}: silhouette = {sil[-1]:.3f}")
    best_k = list(ks)[int(np.argmax(sil))]
    print(f"\nChosen k = {best_k} (silhouette {max(sil):.3f}). "
          "Scores below ~0.25 mean weak cluster structure.")

    data["Cluster"] = KMeans(n_clusters=best_k, n_init=10,
                             random_state=SEED).fit_predict(X)
    print("\nCluster profiles:")
    profile = data.groupby("Cluster").agg(
        users=("Age", "size"),
        avg_age=("Age", "mean"),
        avg_tenure_days=("Tenure Days", "mean"),
        avg_revenue=("Monthly Revenue", "mean"),
        top_device=("Device", lambda s: s.mode().iloc[0]),
        top_plan=("Subscription Type", lambda s: s.mode().iloc[0]),
    ).round(1)
    print(profile)

    OUT.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(list(ks), sil, marker="o", color="#e50914")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Choosing k for K-Means")
    plt.tight_layout()
    plt.savefig(OUT / "kmeans_silhouette.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    run_classification(df)
    run_clustering(df)
    print("\nDone. Charts saved in ./outputs/")
