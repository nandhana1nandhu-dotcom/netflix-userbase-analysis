"""
Generates a SYNTHETIC dataset with the same columns as the Kaggle
"Netflix Userbase Dataset". It exists only so the pipeline can be tested
without the real file. Do NOT report results from this data as real findings.

Usage:
    python generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 2500

countries = ["United States", "Spain", "Canada", "United Kingdom", "Australia",
             "Germany", "France", "Brazil", "Mexico", "Italy"]
devices = ["Smartphone", "Tablet", "Smart TV", "Laptop"]
subs = ["Basic", "Standard", "Premium"]
revenue_map = {"Basic": (10, 15), "Standard": (10, 15), "Premium": (10, 15)}

sub_type = rng.choice(subs, N, p=[0.37, 0.38, 0.25])
revenue = rng.integers(10, 16, N)
join = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, N), unit="D")
last_pay = join + pd.to_timedelta(rng.integers(30, 400, N), unit="D")
last_pay = last_pay.where(last_pay <= pd.Timestamp("2023-07-15"), pd.Timestamp("2023-07-15"))

df = pd.DataFrame({
    "User ID": np.arange(1, N + 1),
    "Subscription Type": sub_type,
    "Monthly Revenue": revenue,
    "Join Date": join.strftime("%d-%m-%y"),
    "Last Payment Date": last_pay.strftime("%d-%m-%y"),
    "Country": rng.choice(countries, N),
    "Age": rng.integers(26, 52, N),
    "Gender": rng.choice(["Male", "Female"], N),
    "Device": rng.choice(devices, N),
    "Plan Duration": "1 Month",
})

out = Path("data")
out.mkdir(exist_ok=True)
df.to_csv(out / "sample_netflix_userbase.csv", index=False)
print(f"Wrote {len(df)} synthetic rows to {out / 'sample_netflix_userbase.csv'}")
