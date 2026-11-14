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

**Limitations:** public Kaggle dataset (likely synthetic), no viewing or churn data, cross-sectional snapshot,
association is not causation. Any business use would need real behavioural data.

## Demo app
```bash
streamlit run app.py
```
Pick country, device, gender, age and tenure; the app shows the predicted plan with probabilities,
plus the model's cross-validated accuracy against the baseline. It trains on your CSV in `data/` at startup.

## Likely interview questions
- **t-test vs chi-squared?** t-test compares means of a numeric variable between two groups; chi-squared tests association between two categorical variables.
- **What is a p-value?** Probability of seeing a result at least this extreme if H0 were true. It is not the probability H0 is true.
- **Student's vs Welch's t-test?** Student's assumes equal variances; Welch's does not and is the safer default.
- **t-test assumptions?** Independent observations, roughly normal data (fine for large n by the CLT), equal variance (Student's only).
- **Chi-squared assumptions?** Independent observations, expected count >= 5 in each cell.
- **Why Bonferroni?** Each extra test raises the chance of a false positive; 20 tests at 0.05 gives about a 64% chance of at least one. Bonferroni divides alpha by the number of tests.
- **Significant but tiny effect?** With large n, tiny differences become significant; that is why Cohen's d / Cramer's V matter.
- **Why can't you use a t-test across 4 devices?** Use ANOVA (or pairwise t-tests with correction).
- **What is target leakage?** Using a feature that already encodes the answer (Monthly Revenue reveals the plan), which gives falsely high accuracy.
- **Why a baseline?** Accuracy means nothing alone; with 3 classes, always guessing the most common one already scores about 37-40%.
- **Why cross-validation?** One train/test split can be lucky or unlucky; averaging over 5 folds gives a steadier estimate.
- **How did you pick k in K-Means?** Silhouette score; it measures how well separated the clusters are (-1 to 1).
- **Why is the model weak?** The features may simply not predict the plan, which matches non-significant chi-squared results. Say that honestly.
- **Limitations?** Cross-sectional data, synthetic-looking Kaggle dataset, correlation is not causation.
