# Real Estate Price Analysis & Predictive Valuation
### An End-to-End Data Analytics & Machine Learning Study on the Ames Housing Market

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/Housing-Price-Analysis/blob/main/notebooks/Real_Estate_Data_Analysis_and_Price_Prediction.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## Executive Summary
This portfolio project delivers an end-to-end data analytics and predictive valuation study of residential real estate in Ames, Iowa, using the 82-feature **Ames Housing Dataset** (compiled by Dean De Cock from the Ames City Assessor's Office).

Designed with a **Data Analyst portfolio focus**, the project combines:
1. **Rigorous Data Quality & Domain Auditing:** Systematic differentiation of semantic absence vs. true missingness and conditional outlier treatment.
2. **Exploratory & Quantitative Analysis:** In-depth univariate, bivariate, and multivariate analysis of real estate dynamics.
3. **Structured Business Question Answering:** 10 commercial market questions structured as **Question $\rightarrow$ Metric $\rightarrow$ Visualization $\rightarrow$ Finding $\rightarrow$ Business Implication**.
4. **Leakage-Free Predictive Modeling:** A progressive 6-model benchmark hierarchy (Baseline $\rightarrow$ Linear $\rightarrow$ Ridge $\rightarrow$ Decision Tree $\rightarrow$ Random Forest $\rightarrow$ Gradient Boosting) strictly excluding leaked attributes such as `Price_per_SF`.
5. **Stakeholder Applications & Realistic Decision Support:** Pragmatic recommendations for home sellers, real estate agents, buyers, and automated valuation cross-checks.
6. **Colab-Ready Execution:** 100% OS-agnostic pathing, multi-tier dataset loading with automated fallback, and top-to-bottom executable notebook.

---

## Running in Google Colab

The project is fully adapted for **Google Colab** and requires zero manual code modifications.

### Option 1: One-Click Notebook Execution (Recommended)
1. Upload [`Real_Estate_Data_Analysis_and_Price_Prediction.ipynb`](notebooks/Real_Estate_Data_Analysis_and_Price_Prediction.ipynb) directly into [Google Colab](https://colab.research.google.com/).
2. Select **Runtime $\rightarrow$ Run all** (`Ctrl + F9`).
3. The notebook will:
   - Autodetect the Colab environment (`/content/Housing-Price-Analysis`).
   - Automatically download the official Ames Housing raw dataset from the public repository fallback.
   - Execute the complete data cleaning, feature engineering, 10 business questions, visualizations, and model benchmarks seamlessly.

### Option 2: Uploading the Complete Project ZIP
1. Compress `Housing-Price-Analysis/` into a `.zip` archive.
2. In Colab's file explorer, upload `Housing-Price-Analysis.zip`.
3. Unzip the project inside Colab:
   ```python
   !unzip -q Housing-Price-Analysis.zip -d /content/
   %cd /content/Housing-Price-Analysis
   ```
4. Open and run `notebooks/Real_Estate_Data_Analysis_and_Price_Prediction.ipynb`.

---

## Repository Structure

```text
Housing-Price-Analysis/
│
├── data/
│   ├── raw/
│   │   └── Ames_Housing_Raw.csv         # Raw Ames dataset (2,930 rows x 82 cols)
│   └── processed/
│       └── Ames_Housing_Cleaned.csv     # Cleaned modeling dataset (2,927 rows x 89 cols)
│
├── notebooks/
│   └── Real_Estate_Data_Analysis_and_Price_Prediction.ipynb  # Complete Colab-ready 14-section study
│
├── scripts/
│   ├── data_cleaning.py                 # Cross-platform cleaning & feature pipeline
│   └── model_training.py                # Cross-platform train/test & evaluation pipeline
│
├── outputs/
│   └── charts/                          # Generated high-resolution visualization artifacts
│       ├── 01_target_distribution.png
│       ├── 02_correlation_matrix.png
│       ├── 03_neighborhood_median_prices.png
│       ├── 04_remodel_premium.png
│       ├── 05_garage_capacity_vs_price.png
│       ├── 06_model_performance_comparison.png
│       ├── 07_feature_importance.png
│       └── 08_residuals_analysis.png
│
├── requirements.txt                     # Project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 14-Section Project Workflow

| Step | Section | Description |
| :---: | :--- | :--- |
| **1** | **Project Overview** | Business context, pricing transparency, and formal regression formulation. |
| **2** | **Environment & Setup** | Runtime detection (Colab vs. local), package installation, and directory structure. |
| **3** | **Dataset Ingestion & Validation** | Multi-tier loading (local, manual upload, or automated download) and schema validation. |
| **4** | **Data Quality Audit** | Structural missingness ('None' vs NaN), target distribution, and conditional outlier inspection. |
| **5** | **Data Cleaning & Preprocessing** | Categorical standardization, neighborhood-median lot imputation, and outlier treatment. |
| **6** | **Feature Engineering** | Derived domain attributes (`Total_SF`, `Property_Age`, `Remodel_Age`, `Bath_to_Bed_Ratio`). |
| **7** | **Exploratory Data Analysis** | Univariate distributions, bivariate correlations, and multivariate interaction matrices. |
| **8** | **Business / Market Analysis** | 10 commercial market questions in standardized *Question $\rightarrow$ Metric $\rightarrow$ Finding $\rightarrow$ Implication* format. |
| **9** | **Statistical Analysis** | Pearson/Spearman correlation metrics, variance explained ($R^2$), and strictly non-causal language. |
| **10** | **Regression Modeling** | 6-model benchmark suite evaluated on an 80/20 held-out test split. |
| **11** | **Model Interpretation** | Feature importances (Gini impurity), actual vs. predicted prices, and residual diagnostics. |
| **12** | **Key Findings & Applications** | Actionable takeaways tailored for sellers, buyers, agents, and valuation analysts. |
| **13** | **Limitations** | Macroeconomic omissions, cross-sectional boundaries, and non-causal disclaimers. |
| **14** | **Future Scope & Conclusion** | Interactive Streamlit deployment, GIS geospatial integration, and executive closing summary. |

---

## Key Methodological Decisions

### 1. Data Leakage Prevention (`Price_per_SF`)
A common pitfall in real estate predictive modeling is computing:
$$\text{Price\_per\_SF} = \frac{\text{SalePrice}}{\text{Total\_SF}}$$
and including it as a feature in a model predicting `SalePrice`. This introduces **severe direct target leakage**, rendering model evaluation metrics falsely optimistic.
- **Implementation Decision:** In this project, `Price_per_SF` is **strictly isolated** to descriptive market analysis and buyer valuation benchmarks. It is completely excluded from the feature matrix $X$ and modeling pipelines.

### 2. Conditional Outlier Handling vs. Blanket Cuts
Rather than indiscriminately removing all properties exceeding 4,000 square feet, records with `GrLivArea > 4000` were evaluated conditionally:
- **Legitimate High-End Homes:** Two properties sold for \$745,000 and \$755,000 with `OverallQual = 10`. These follow typical luxury price curves and were **retained**.
- **Severe Leverage Outliers:** Three properties sold for under \$200,000 under partial/commercial sale conditions. These were removed to avoid disproportionate distortion of linear and distance-based estimators, exactly as recommended by De Cock (2011).

### 3. Preprocessing Fit Strictly on Training Fold
All encoders (One-Hot) and scalers (StandardScaler) are fitted exclusively on the 80% training fold, preventing test-set information from bleeding into training data transformations.

---

## 10 Business & Market Questions Answered

Each question is structured using the framework: **Question $\rightarrow$ Metric $\rightarrow$ Finding $\rightarrow$ Business Implication**.

1. **Neighborhood Price Disparities:**
   - *Metric:* Median sale price and IQR across neighborhoods.
   - *Finding:* A 3x price spread exists between the highest-tier enclave (Stone Brook: \$310,750 median) and entry-tier neighborhoods (Meadow Village: \$105,000 median).
   - *Implication:* Micro-location serves as the primary price baseline multiplier.

2. **Remodeling Descriptive Premium:**
   - *Metric:* Median sale price for remodeled vs. original homes across age cohorts.
   - *Finding:* In older homes (36+ years old), remodeled properties recorded a 20% to 35% higher median sale price.
   - *Implication:* Targeted capital renovations substantially recover market value on aging properties.

3. **Garage Car Capacity:**
   - *Metric:* Median price and spread across garage capacities (0 to 4 cars).
   - *Finding:* Homes without a garage sold for a median of \$105,000, while 2-car garages averaged \$185,000 (+45%).
   - *Implication:* A 2-car garage is a market standard; properties lacking garages incur liquidity penalties.

4. **Age & Location Interaction:**
   - *Metric:* Pre-1960 home valuations stratified by neighborhood demand tier.
   - *Finding:* Older homes in high-demand tiers commanded \$178,000 median vs. \$107,000 in low-demand tiers.
   - *Implication:* Land and neighborhood desirability heavily buffer physical structure depreciation.

5. **Central Air Conditioning:**
   - *Metric:* Median price comparison between homes with and without central cooling.
   - *Finding:* Properties with central AC exhibited a median price of \$165,000 vs. \$103,500 without AC (+60%).
   - *Implication:* Central cooling is non-negotiable for modern buyers; its absence restricts sales to discounted investor segments.

6. **Overall Construction Quality:**
   - *Metric:* Step-wise median price across 1–10 quality ratings.
   - *Finding:* Median price scales exponentially with quality: Grade 5 (\$133k) $\rightarrow$ Grade 7 (\$200k) $\rightarrow$ Grade 9 (\$345k).
   - *Implication:* Materials and architectural finish are the single strongest quality indicator.

7. **Outdoor Living Space:**
   - *Metric:* Median price comparison for properties with porches/decks vs. none.
   - *Finding:* Properties with outdoor living square footage showed a \$174,000 median vs. \$132,000 for homes with no porch space.
   - *Implication:* Porches and decks represent cost-effective amenity additions that correlate with higher buyer willingness-to-pay.

8. **Lot Configuration:**
   - *Metric:* Median sale price across cul-de-sac, inside, and corner lots.
   - *Finding:* Cul-de-sac lots commanded the highest median price (\$216,000) over inside lots (\$158,500).
   - *Implication:* Privacy and reduced through-traffic carry quantifiable market appeal for family homes.

9. **Seasonality in Volume vs. Price:**
   - *Metric:* Monthly transaction counts and median closing prices across 2006–2010.
   - *Finding:* Transaction volume peaks sharply in May–July (>350/mo), but median closing prices remained steady (\$155k–\$165k).
   - *Implication:* Real estate seasonality primarily dictates market liquidity rather than price swings.

10. **Sale Condition & Distress:**
    - *Metric:* Median prices across normal, partial, and abnormal transactions.
    - *Finding:* Abnormal sales (foreclosures, short sales) transacted at a \$130,000 median vs. \$160,000 for normal sales.
    - *Implication:* Non-standard transactions must be filtered out when compiling comparative market appraisals.

---

## Predictive Modeling Benchmark Results

All 6 models were evaluated on the held-out 20% test partition (586 properties) without data leakage:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Median)** | \$59,594.40 | \$92,434.60 | -0.0601 | Naive central tendency benchmark. |
| **Linear Regression (OLS)** | \$15,563.33 | \$24,875.89 | 0.9232 | Strong interpretable linear baseline. |
| **Ridge Regression** | \$14,789.73 | \$22,871.06 | 0.9351 | Regularization manages multicollinearity. |
| **Decision Tree** | \$20,725.31 | \$30,105.55 | 0.8875 | Non-linear tree benchmark. |
| **Random Forest** | \$14,611.95 | \$23,222.35 | 0.9331 | Bagged ensemble reducing variance. |
| **Gradient Boosting** | **\$13,521.70** | **\$21,133.31** | **0.9446** | **Best overall predictor**, lowest error. |

### Top Predictive Features (Gradient Boosting)
1. `OverallQual` (Construction quality rating)
2. `Total_SF` (Basement + 1st Floor + 2nd Floor square footage)
3. `GrLivArea` (Above-grade living square footage)
4. `GarageCars` / `GarageArea` (Garage capacity)
5. `Total_Bathrooms` (Total bathroom count)
6. `Property_Age` (Years since original construction)
7. `Remodel_Age` (Years since last renovation)

---

## Business Applications & Stakeholder Value

- **For Home Sellers:** Renovation ROI is concentrated in functional updates (kitchen, bathrooms, central air) rather than cosmetic expansions. Older homes benefit most significantly from comprehensive kitchen/mechanical updates.
- **For Real Estate Agents:** Lot configuration (cul-de-sac premium) and micro-neighborhood tiering provide empirical justification when setting initial listing prices and counseling sellers.
- **For Home Buyers:** Descriptive price-per-SF metrics allow buyers to identify undervalued properties within specific neighborhood peer groups rather than comparing against misleading citywide medians.
- **For Valuation-Support Use Cases:** The regularized and boosted regression models provide automated baseline cross-checks to flag anomalies or unusual pricing deviations for human appraisers.

---

## Limitations & Non-Causal Disclaimers

> [!WARNING]
> **Non-Causal Nature of Findings:** All statistical findings, percentage spreads, and regression coefficients represent **empirical associations and predictive differences** in the observational dataset. They do not represent causal pricing rules (e.g., adding a second bathroom will not automatically increase every home's price by an identical fixed amount).

Additional analytical limitations:
- **Macroeconomic Indicators:** Mortgage interest rates, unemployment, and regional inflation were not recorded in the dataset.
- **Geographic Scope:** Data is specific to Ames, Iowa (2006–2010) and should not be directly applied to distinct housing markets without local recalibration.
- **Micro-Location Data:** Micro-factors such as specific school boundary assignments, power-line proximity, and street noise are absent.

---

## Installation & Local Usage

```bash
# Clone repository
git clone https://github.com/your-username/Housing-Price-Analysis.git
cd Housing-Price-Analysis

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data cleaning
python scripts/data_cleaning.py

# Run model training & chart export
python scripts/model_training.py
```
