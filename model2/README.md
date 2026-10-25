# Model 2: PCOS risk profile

Learns what a PCOS-risk profile looks like from the Kaggle PCOS dataset (541 patients,
10 hospitals in Kerala, India), using 11 inputs a person can self-report: age, BMI,
cycle regularity, period length (days of bleeding), weight gain, hair growth, skin
darkening, hair loss, acne, fast food and regular exercise. No lab tests or ultrasound.

| File | What it does |
|---|---|
| `data/pcos_model2_selected_features.csv` | Input: the 11 self-reportable features and the PCOS label |
| `build_pcos_model2.py` | Trains and compares Logistic Regression, Random Forest, XGBoost and Extra Trees |
| `results/metrics/` | Benchmark, threshold analysis, feature importance, SHAP values |
| `results/figures/` | Charts used in the report and blog |

Run from the repository root: `python model2/build_pcos_model2.py`

Chosen model: **Extra Trees** (test ROC-AUC 0.92, recall 0.81 at threshold 0.5).
