# Bank Credit Default Prediction

A machine learning project that predicts whether a bank loan applicant will **default** on their loan, based on financial, credit-history, and loan-application attributes. The full workflow (data cleaning, feature engineering, exploratory analysis, outlier handling, and model training/evaluation) lives in a single Jupyter notebook.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Results](#results)
- [Getting Started](#getting-started)
- [Limitations and Future Work](#limitations-and-future-work)

## Overview

Credit default is costly for lenders. Spotting high-risk applicants early helps banks make better lending decisions, price risk accurately, and reduce losses. This project frames the problem as **binary classification**:

| Target (`Default`) | Meaning |
|---|---|
| `0` | Borrower did not default |
| `1` | Borrower defaulted |

**Tech stack:** Python · pandas · NumPy · scikit-learn · seaborn · Matplotlib · Jupyter / Google Colab

## Dataset

- **File:** `loan data.xls`. The file has an `.xls` extension, but it is **comma-separated text**, so the notebook loads it with `pd.read_csv`.
- **Size:** 87,500 records × 30 columns (81,351 records after cleaning)
- **Class distribution (after cleaning):** 65,984 non-default (≈81%) vs. 15,367 default (≈19%), which is **imbalanced**

### Key Features

| Category | Columns |
|---|---|
| Borrower profile | `Experience`, `Yearly_Income`, `Home_Status`, `Designation`, `State`, `Postal_Code` |
| Credit history | `GGGrade`, `Sub_GGGrade`, `Unpaid_2_years`, `Already_Defaulted`, `Deprecatory_Records`, `Inquiries`, `Account_Open`, `Usage_Rate` |
| Loan details | `Lend_Amount`, `Interest_Charged`, `Duration`, `Reason`, `Claim_Type`, `File_Status`, `Validation` |
| Balances and payments | `Debt_to_Income`, `Present_Balance`, `Gross_Collection`, `Total_Unpaid_CL`, `Unpaid_Amount`, `Due_Fee` |
| Identifiers (dropped) | `ID`, `Asst_Reg` |

## Project Structure

```
Bank-Credit-Default-Prediction/
├── loan data.xls     # Raw dataset (CSV format)
├── program.ipynb     # End-to-end analysis and modeling notebook
└── README.md
```

## Workflow

### 1. Data Cleaning
- **Missing values:**
  - `Yearly_Income`, `Debt_to_Income`: filled with the **mean**
  - `Total_Unpaid_CL`: filled with the **mode**
  - `Designation`: filled with `"unknown"`
  - Remaining rows with nulls (`Postal_Code`, `Unpaid_Amount`) were dropped
- Checked for duplicate rows (none found).
- Dropped identifier columns `ID` and `Asst_Reg`.

### 2. Categorical Feature Engineering
- **High-cardinality columns dropped:** `Designation` (~38k unique values) and `State` (49 unique values), to avoid a very wide one-hot encoding.
- **Text-to-numeric:** stripped symbols and units from `Experience` (e.g. `>10yrs` → `10`) and `Duration` (e.g. `3 years` → `3`).
- **Ordinal (label) encoding:** `GGGrade` (`I`–`VII` → `1`–`7`).
- **One-hot encoding:** `Claim_Type`, `Reason`, `File_Status`, `Home_Status`, `Validation`.

### 3. Exploratory Data Analysis and Feature Selection
- Used **mutual information** (`mutual_info_classif`) to measure linear and non-linear dependence between each feature and the target.
- Kept only features with a mutual information score above zero.
- Most informative features include `File_Status`, `Claim_Type`, `Reason` (debt consolidation), `Duration`, `Validation`, `Home_Status`, and `Lend_Amount`.

### 4. Outlier Detection and Treatment
- Visualized numeric feature distributions with **box plots**.
- Applied **IQR-based capping** (winsorization): values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` are clipped to the nearest bound.

### 5. Modeling
- **Train/test split:** 80/20 (`random_state=42`)
- **Scaling:** `StandardScaler`, fit on the training set only
- **Model:** `LogisticRegression(max_iter=1000)`
- **Iteration:** inspected the logistic regression coefficients for feature importance, dropped the low-impact features (`Gross_Collection`, `Due_Fee`, `Deprecatory_Records`, `Unpaid_2_years`), and retrained.

## Results

Logistic Regression results on the held-out test set (16,271 samples):

| Metric | Class 0 (No Default) | Class 1 (Default) |
|---|---|---|
| Precision | 0.82 | 0.53 |
| Recall | 1.00 | 0.00 |
| F1-score | 0.90 | 0.01 |

**Overall accuracy:** 81.9%

**Confusion matrix:**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 13,315 | 8 |
| **Actual 1** | 2,939 | 9 |

### Interpretation

The 81.9% accuracy is about the same as always predicting "no default" (≈81.9% of the test set). Because of the class imbalance, the model predicts the majority class almost every time and catches **only 9 of 2,948 actual defaulters**. The reduced-feature model gave the same confusion matrix. This baseline shows that accuracy alone is a misleading metric for this problem. Improving recall on the default class is the main goal for the next iteration.

**Top coefficients (by magnitude):** `Debt_to_Income`, `Duration`, `File_Status`, `Inquiries`, `Claim_Type_I`, `Validation_Source Verified`, `Sub_GGGrade`.

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook / JupyterLab, or Google Colab

### Installation

```bash
git clone https://github.com/ashoktamang002/Bank-Credit-Default-Prediction.git
cd Bank-Credit-Default-Prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

### Run the Notebook

```bash
jupyter notebook program.ipynb
```

Run the cells from top to bottom. The notebook expects `loan data.xls` to be in the same directory.

> **Google Colab:** upload both `program.ipynb` and `loan data.xls` to your Colab session, then run all cells.

## Limitations and Future Work

- **Handle class imbalance:** use `class_weight="balanced"`, SMOTE/oversampling, or undersampling.
- **Tune the decision threshold:** optimize for recall or F1 on the default class instead of using the default 0.5 cutoff.
- **Use better evaluation metrics:** report ROC-AUC, PR-AUC, and recall/F1 for the minority class instead of accuracy alone.
- **Try stronger models:** Random Forest, Gradient Boosting, XGBoost, or LightGBM, which capture non-linear interactions.
- **Prevent data leakage:** compute imputation values and outlier bounds on the training split only, inside a scikit-learn `Pipeline`.
- **Recover dropped signal:** try target or frequency encoding for `State` and `Designation`.
- **Validate more robustly:** use stratified k-fold cross-validation and hyperparameter search.
- **Make the project reproducible:** add a `requirements.txt` and move the notebook logic into reusable scripts.

## Author

**Ashok Tamang** · [GitHub @ashoktamang002](https://github.com/ashoktamang002)

---

*If you find this project useful, consider giving it a star.*
