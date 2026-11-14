"""
Netflix Userbase Statistical Analysis
-------------------------------------
Pipeline: load -> clean -> EDA -> hypothesis tests (t-test, chi-squared) -> report

Usage:
    python analysis.py --data "data/Netflix Userbase.csv"
    python analysis.py --data data/sample_netflix_userbase.csv   # synthetic test data
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
OUT = Path("outputs")


# ---------------------------------------------------------------- loading
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print(f"Raw shape: {df.shape}")
    print("Missing values per column:\n", df.isna().sum(), "\n")

    df = df.drop_duplicates()
    df = df.dropna(subset=["Age", "Gender", "Device", "Subscription Type"])

    # Dates are stored as dd-mm-yy in the Kaggle file
    for col in ["Join Date", "Last Payment Date"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # "1 Month" -> 1
    df["Plan Duration"] = (
        df["Plan Duration"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )

    # Feature engineering: how long the user has been paying
    df["Tenure Days"] = (df["Last Payment Date"] - df["Join Date"]).dt.days
    df["Age Group"] = pd.cut(
        df["Age"], bins=[0, 29, 39, 49, 120],
        labels=["<30", "30-39", "40-49", "50+"],
    )

    print(f"Clean shape: {df.shape}\n")
    return df


# -------------------------------------------------------------------- EDA
def run_eda(df: pd.DataFrame) -> None:
    OUT.mkdir(exist_ok=True)

    print("=" * 60, "\nEXPLORATORY ANALYSIS\n" + "=" * 60)
    print(df[["Age", "Monthly Revenue", "Tenure Days"]].describe().round(2), "\n")
    for col in ["Subscription Type", "Device", "Gender", "Country"]:
        print(f"{col} distribution (%):")
        print((df[col].value_counts(normalize=True) * 100).round(1), "\n")

    # Charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    df["Age"].plot.hist(bins=20, ax=axes[0, 0], color="#e50914", edgecolor="white")
    axes[0, 0].set_title("Age distribution")
    df["Device"].value_counts().plot.bar(ax=axes[0, 1], color="#221f1f")
    axes[0, 1].set_title("Users by device")
    df["Subscription Type"].value_counts().plot.bar(ax=axes[1, 0], color="#b20710")
    axes[1, 0].set_title("Users by subscription type")
    pd.crosstab(df["Device"], df["Subscription Type"]).plot.bar(
        stacked=True, ax=axes[1, 1])
    axes[1, 1].set_title("Device vs subscription type")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "eda_overview.png", dpi=150)
    plt.close()


# ------------------------------------------------------- hypothesis tests
def verdict(p: float) -> str:
    return ("REJECT H0 (statistically significant)" if p < ALPHA
            else "FAIL TO REJECT H0 (no significant evidence)")


def cohens_d(a, b) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def t_test(df, value_col, group_col, g1, g2) -> None:
    a = df.loc[df[group_col] == g1, value_col].dropna()
    b = df.loc[df[group_col] == g2, value_col].dropna()
    # Welch's t-test: does not assume equal variances
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"\nT-TEST: {value_col} by {group_col} ({g1} vs {g2})")
    print(f"  H0: mean {value_col} is the same for {g1} and {g2}")
    print(f"  {g1}: mean={a.mean():.2f} (n={len(a)}) | {g2}: mean={b.mean():.2f} (n={len(b)})")
    print(f"  t = {t:.3f}, p = {p:.4f}, Cohen's d = {cohens_d(a, b):.3f}")
    print(f"  -> {verdict(p)}")


def chi_squared(df, col1, col2) -> None:
    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    print(f"\nCHI-SQUARED: {col1} vs {col2}")
    print(f"  H0: {col1} and {col2} are independent")
    print(f"  chi2 = {chi2:.3f}, dof = {dof}, p = {p:.4f}, Cramer's V = {cramers_v:.3f}")
    print(f"  min expected count = {expected.min():.1f} "
          f"({'assumption OK' if expected.min() >= 5 else 'WARNING: <5, interpret carefully'})")
    print(f"  -> {verdict(p)}")


def run_tests(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60, "\nHYPOTHESIS TESTING (alpha = 0.05)\n" + "=" * 60)

    # Student's / Welch's t-tests (two groups only)
    t_test(df, "Age", "Gender", "Male", "Female")
    t_test(df, "Tenure Days", "Gender", "Male", "Female")
    t_test(df, "Monthly Revenue", "Gender", "Male", "Female")

    # Pairwise device comparison on age (4 devices -> 6 pairs)
    # Bonferroni correction for multiple comparisons
    devices = sorted(df["Device"].unique())
    pairs = list(combinations(devices, 2))
    print(f"\nPAIRWISE T-TESTS: Age across devices "
          f"(Bonferroni alpha = {ALPHA / len(pairs):.4f})")
    for d1, d2 in pairs:
        a = df.loc[df["Device"] == d1, "Age"]
        b = df.loc[df["Device"] == d2, "Age"]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        flag = "significant" if p < ALPHA / len(pairs) else "n.s."
        print(f"  {d1} vs {d2}: t={t:.2f}, p={p:.4f} [{flag}]")

    # Chi-squared tests of independence (categorical vs categorical)
    chi_squared(df, "Device", "Subscription Type")
    chi_squared(df, "Gender", "Subscription Type")
    chi_squared(df, "Gender", "Device")
    chi_squared(df, "Age Group", "Device")
    chi_squared(df, "Country", "Subscription Type")


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    args = parser.parse_args()

    data = load_and_clean(args.data)
    run_eda(data)
    run_tests(data)
    print("\nDone. Charts saved in ./outputs/")
