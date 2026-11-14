"""
Streamlit demo: predict a Netflix user's subscription plan.

Run:
    streamlit run app.py

It trains a Logistic Regression model on your CSV when it starts, then lets you
pick a country, device, gender, age and tenure and see the predicted plan with
probabilities. This is a learning demo, not a production system.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis import load_and_clean

NUMERIC = ["Age", "Tenure Days"]
CATEGORICAL = ["Gender", "Country", "Device"]

st.set_page_config(page_title="Netflix Plan Predictor", page_icon="🎬")
st.title("🎬 Netflix Plan Predictor")
st.caption("Predicts a user's subscription plan from country, device, gender, age and tenure.")


def find_default_csv() -> str:
    csvs = sorted(Path("data").glob("*.csv"))
    # Prefer a real dataset over the synthetic sample if both exist
    real = [c for c in csvs if "sample" not in c.name.lower()]
    return str((real or csvs or [Path("data/Netflix Userbase.csv")])[0])


csv_path = st.sidebar.text_input("CSV path", value=find_default_csv())


@st.cache_resource(show_spinner="Training model...")
def train(path: str):
    df = load_and_clean(path).dropna(subset=NUMERIC + CATEGORICAL + ["Subscription Type"])
    X, y = df[NUMERIC + CATEGORICAL], df["Subscription Type"]
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    cv_acc = cross_val_score(pipe, X, y, cv=5).mean()
    baseline = y.value_counts(normalize=True).max()
    pipe.fit(X, y)
    return pipe, df, cv_acc, baseline


if not Path(csv_path).exists():
    st.error(f"File not found: {csv_path}. Put your CSV in the data folder and fix the path in the sidebar.")
    st.stop()

model, df, cv_acc, baseline = train(csv_path)

st.sidebar.metric("Cross-validated accuracy", f"{cv_acc:.1%}")
st.sidebar.metric("Baseline (always guess most common plan)", f"{baseline:.1%}")
st.sidebar.caption("The model only beats the baseline by a useful margin if its inputs "
                   "carry real signal. In this dataset, mainly Country does.")

col1, col2 = st.columns(2)
country = col1.selectbox("Country", sorted(df["Country"].unique()))
device = col2.selectbox("Device", sorted(df["Device"].unique()))
gender = col1.selectbox("Gender", sorted(df["Gender"].unique()))
age = col2.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
tenure = st.slider("Tenure (days paying)", 0, int(df["Tenure Days"].max()),
                   int(df["Tenure Days"].median()))

if st.button("Predict plan", type="primary"):
    row = pd.DataFrame([{"Age": age, "Tenure Days": tenure,
                         "Gender": gender, "Country": country, "Device": device}])
    probs = pd.Series(model.predict_proba(row)[0], index=model.classes_).sort_values(ascending=False)
    st.subheader(f"Predicted plan: {probs.index[0]}")
    st.bar_chart(probs)
    st.caption("Bars show the model's estimated probability for each plan. "
               "Treat this as a demo of the workflow, not a business decision tool.")

with st.expander("How plan mix varies by country (from your data)"):
    mix = pd.crosstab(df["Country"], df["Subscription Type"], normalize="index").round(3) * 100
    st.dataframe(mix.style.format("{:.1f}%"))
