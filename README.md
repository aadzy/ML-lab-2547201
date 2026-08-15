# Predicting Earthquake Building Damage
### ML for Social Good — Ensemble Challenge (CIA 3)
**Mission Domain:** Crisis / Disaster Response — Post-Earthquake Damage Triage

---

## Overview

This project predicts building-level earthquake damage severity (`damage_grade`: 1 = low, 2 = medium, 3 = complete destruction) using structural and legal attributes of buildings surveyed after the 2015 Gorkha earthquake in Nepal. The goal is to support disaster-response and reconstruction agencies in triaging which buildings need priority inspection and aid, at a scale (260,601 buildings) where manual assessment does not scale.

The pipeline covers the full ML lifecycle: data wrangling and EDA, domain-informed feature engineering, a baseline Decision Tree, a bagging ensemble (Random Forest), a boosting ensemble (LightGBM), a heterogeneous stacking ensemble, and SHAP-based explainability at both global and individual-prediction levels.

**Best model:** Heterogeneous stacking ensemble (Decision Tree + Random Forest + LightGBM base learners, Logistic Regression meta-learner), achieving weighted F1 = 0.6645 on a held-out test set — a ~7-point F1 improvement over the Decision Tree baseline (0.5948).

---

## Repository Contents

| File | Description |
|---|---|
| `CIA_3_2015_Gorkha_Earthquake.ipynb` | Full notebook: EDA, preprocessing, feature engineering, model training/tuning, evaluation, SHAP explainability |
| `CIA3_ML_Social_Good_Report.docx` | Written report covering impact framing, wrangling justification, model comparison, and ethics discussion |
| `train_values.csv` | Building feature data (not included in repo — see Data Access below) |
| `train_labels.csv` | Damage grade labels (not included in repo — see Data Access below) |

---

## Reproducibility Instructions

### 1. Environment

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn
```

Tested on Python 3.12 (Google Colab runtime).

### 2. Data Access

Download the following files from the dataset source (see citation below):

- `train_values.csv` — building features
- `train_labels.csv` — damage grade labels

Place both files in the same directory as the notebook, or update the file paths in the first data-loading cell to match your local/Drive path:

```python
values = pd.read_csv('train_values.csv')
labels = pd.read_csv('train_labels.csv')
```

`test_values.csv` and `submission_format.csv` from the original competition are **not used** — they lack labels and are irrelevant to this assignment's evaluation requirements. This project instead carves its own labeled train/validation/test split from the merged `train_values.csv` + `train_labels.csv`.

### 3. Run Order

Run all notebook cells sequentially, top to bottom. Each stage depends on the previous:

1. Imports
2. Load + merge data
3. EDA audit (missing values, duplicates, dtypes, class distribution, outliers, cardinality)
4. Drop `building_id`, `geo_level_3_id`; frequency-encode `geo_level_1_id`, `geo_level_2_id`
5. Domain feature engineering (`age_area_ratio`, `height_floor_ratio`, `superstructure_count`)
6. Outlier capping (`age`, 99th percentile)
7. Categorical encoding (LabelEncoder)
8. Train/validation/test split (stratified 70/15/15) + scaling (fit on train only)
9. Baseline: Decision Tree
10. Bagging: Random Forest (`RandomizedSearchCV`, tuned on a 40k-row subsample, refit on full train)
11. Boosting: LightGBM (same tuning strategy)
12. Stacking: heterogeneous ensemble with internal cross-validation for the meta-learner
13. Final evaluation on the untouched test set (F1, precision, recall, ROC-AUC)
14. Confusion matrix for the best model
15–16. SHAP explainability (global summary + local waterfall plot)

### 4. Random Seeds

All stochastic steps (`train_test_split`, `RandomizedSearchCV`, model `random_state`) use `random_state=42` for reproducibility. Re-running the notebook end-to-end should reproduce the reported metrics exactly, modulo minor floating-point/library-version variation.

### 5. Expected Runtime

- Steps 1–9: under a minute
- Step 10 (Random Forest tuning): a few minutes
- Step 11 (LightGBM tuning): typically faster than Random Forest
- Step 12 (Stacking): the slowest step — internal CV refits each base learner across folds
- Steps 13–16: under a minute

Total: roughly 10–20 minutes on a standard Colab CPU runtime.

---

## Dataset Citation

Kathmandu Living Labs and the Central Bureau of Statistics, National Planning Commission Secretariat, Government of Nepal. (2015). *2015 Nepal Earthquake Open Data.* Distributed via DrivenData, "Richter's Predictor: Modeling Earthquake Damage" (2018).
https://www.drivendata.org/competitions/57/nepal-earthquake/

---

## Responsible Use

- This model is trained on a single earthquake event in a single country and should not be deployed for other regions or seismic events without retraining and local validation.
- Predictions are intended to support, not replace, human structural inspection.
- No personally identifiable information is used — only building-level structural and legal-ownership attributes.
- See Section 4 of the accompanying report for a full discussion of bias, fairness, uncertainty, and deployment limitations.
