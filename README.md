# Netflix Userbase Statistical Analysis

Statistical analysis of a Netflix userbase dataset to study how demographics (age, gender, country),
streaming devices and subscription plans relate to each other.

**Stack:** Python, Pandas, SciPy, NumPy, Matplotlib

## Dataset
Kaggle "Netflix User Dataset" (2,500 users). Columns: User ID, Subscription Type, Monthly Revenue,
Join Date, Last Payment Date, Country, Age, Gender, Device, Plan Duration.

**Source:** [Netflix User Dataset on Kaggle](https://www.kaggle.com/datasets/shivangi2018/netflix-user-dataset)

The CSV is not included in this repo. Download it from the link above and put it in a `data/` folder
(for example `data/Netflix Userbase.csv`), then run the scripts with `--data` pointing to that file.

> `generate_sample_data.py` creates a **synthetic** file with the same columns, only for testing the code.
> Never quote its results as real findings.
## Run
```bash
pip install -r requirements.txt
python analysis.py --data "data/Netflix Userbase.csv"
```
Charts are saved to `outputs/eda_overview.png`; test results print to the console.

## Pipeline
1. **Preprocessing** - strip column names, drop duplicates / missing key fields, parse `dd-mm-yy` dates,
   numeric plan duration, engineered `Tenure Days` and `Age Group`.
2. **EDA** - summary statistics, category distributions, overview charts.
3. **t-tests** (Welch's, two groups) - Age, Tenure and Monthly Revenue by Gender; pairwise Age comparison
   across devices with Bonferroni correction.
4. **Chi-squared tests of independence** - Device x Subscription, Gender x Subscription, Gender x Device,
   Age Group x Device, Country x Subscription. Reports Cramer's V (effect size) and checks expected counts >= 5.

5. **Machine learning extension** (`ml_models.py`)
   - *Classification:* predict Subscription Type from age, gender, country, device and tenure.
     Logistic Regression and Random Forest are compared with a majority-class baseline using
     5-fold stratified cross-validation, then a hold-out classification report and confusion matrix.
   - *Clustering:* K-Means user segmentation; k chosen by silhouette score; cluster profiles printed.
   - `Monthly Revenue` is excluded from the classifier to avoid target leakage (price is set by the plan).

Run it with `python ml_models.py --data "data/Netflix Userbase.csv"`
(outputs: `confusion_matrix.png`, `kmeans_silhouette.png`).

## Key design choices
- alpha = 0.05; H0 is always "no difference / independent".
- Welch's t-test instead of Student's, because it does not assume equal variances.
- Effect sizes (Cohen's d, Cramer's V) reported alongside p-values: significance is not importance.
- Bonferroni correction when running multiple pairwise comparisons.

## Findings
- Subscription type is strongly associated with country (chi-squared p < 0.001, Cramer's V = 0.58); device, gender and age group showed no significant association with plan.
- Age differed marginally by gender (p = 0.048), but the effect was negligible (Cohen's d = -0.08), and it could be a false positive given the number of tests run.
- Logistic Regression and Random Forest predicted subscription type with about 66% accuracy vs a 40% baseline (5-fold CV), consistent with country being the main driver.
- K-Means found weak cluster structure (silhouette about 0.17), so no clear user segments.

## Business insights and recommendations
Numbers below come from the real dataset run (2,500 users). Reword them in your own voice.

| Business question | Evidence | Recommendation |
|---|---|---|
| Does where a user lives shape which plan they choose? | Country vs plan: chi-squared p < 0.001, Cramer's V = 0.58 (large) | Plan, pricing and promotion strategy should be set per country, not globally. |
| Should we personalise by device? | Device vs plan: p = 0.84, V = 0.02 | Device-based plan targeting is not supported by this data; don't invest in it. |
| Should we target by gender or age? | Gender vs plan p = 0.73; age group vs device p = 0.96; age by gender p = 0.048 with Cohen's d = -0.08 | Demographic targeting by gender or age looks unjustified; differences are negligible. |
| Can we predict a user's plan? | Logistic Regression: about 66% accuracy vs 40% baseline (5-fold CV) | Country-driven models can support market-level planning, but they are only moderately accurate for individual users. |
| Are there distinct user segments? | K-Means silhouette about 0.17 | No clear segments in this data; avoid segment-based campaigns built on it. |


ooking Kaggle dataset, correlation is not causation.
