# WAI PCOS Model 2 — Four-model pipeline
# Models: Logistic Regression, Random Forest, XGBoost, Extra Trees
# Input: data/processed/pcos_model2_selected_features.csv
# Outputs:
#   results/metrics/*.csv
#   results/figures/*.png

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)

# -------------------------------------------------------------------
# 0. Repository paths
# -------------------------------------------------------------------

# Expected location:
#   WAI_PCOS_Project/src/build_pcos_model2.py
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = REPO_ROOT / "data" / "processed" / "pcos_model2_selected_features.csv"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Repository root:", REPO_ROOT)
print("Reading data from:", DATA_PATH)
print("Metrics output:", METRICS_DIR)
print("Figures output:", FIGURES_DIR)

# Remove stale outputs before regenerating
metric_files = [
    "four_model_benchmark.csv",
    "threshold_analysis.csv",
    "feature_importance.csv",
    "shap_mean_abs.csv",
]

figure_files = [
    "model_benchmark.png",
    "threshold_tradeoff.png",
    "feature_importance.png",
    "shap_summary.png",
    "logistic_confusion_matrix.png",
]

for filename in metric_files:
    path = METRICS_DIR / filename
    if path.exists():
        path.unlink()

for filename in figure_files:
    path = FIGURES_DIR / filename
    if path.exists():
        path.unlink()

# -------------------------------------------------------------------
# 1. Load processed data
# -------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

features = [
    "Age (yrs)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]

target = "PCOS (Y/N)"

X = df[features].copy()
y = df[target].astype(int).copy()

# -------------------------------------------------------------------
# 2. Preprocessing
# -------------------------------------------------------------------

numeric_features = [
    "Age (yrs)",
    "BMI",
    "Cycle length(days)",
]

categorical_features = [
    "Cycle(R/I)",
]

binary_features = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]

preprocess = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
        (
            "bin",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            ),
            binary_features,
        ),
    ],
    sparse_threshold=0,
)

# -------------------------------------------------------------------
# 3. Train/test split
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# -------------------------------------------------------------------
# 4. Models
# -------------------------------------------------------------------

models = {
    "Logistic Regression": Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "XGBoost": Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    ),
    "Extra Trees": Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
}

# -------------------------------------------------------------------
# 5. Cross-validation + held-out test evaluation
# -------------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

benchmark_rows = []
fitted_models = {}
probabilities = {}

for name, model in models.items():
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    model.fit(X_train, y_train)
    fitted_models[name] = model

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)
    probabilities[name] = prob

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    benchmark_rows.append(
        {
            "Model": name,
            "CV Precision": cv_scores["test_precision"].mean(),
            "CV Recall": cv_scores["test_recall"].mean(),
            "CV F1": cv_scores["test_f1"].mean(),
            "CV ROC-AUC": cv_scores["test_roc_auc"].mean(),
            "CV PR-AUC": cv_scores["test_pr_auc"].mean(),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Test Precision": precision_score(y_test, pred),
            "Test Recall": recall_score(y_test, pred),
            "Test F1": f1_score(y_test, pred),
            "Test ROC-AUC": roc_auc_score(y_test, prob),
            "Test PR-AUC": average_precision_score(y_test, prob),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    )

benchmark_df = (
    pd.DataFrame(benchmark_rows)
    .sort_values(
        ["CV F1", "CV Recall", "CV PR-AUC"],
        ascending=False,
    )
)

print("\nFOUR-MODEL BENCHMARK")
print(benchmark_df.to_string(index=False))

benchmark_path = METRICS_DIR / "four_model_benchmark.csv"
benchmark_df.to_csv(benchmark_path, index=False)

# -------------------------------------------------------------------
# 6. Threshold analysis
# -------------------------------------------------------------------

thresholds = np.arange(0.20, 0.71, 0.05)
threshold_rows = []

for name, prob in probabilities.items():
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        threshold_rows.append(
            {
                "Model": name,
                "Threshold": round(float(threshold), 2),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

threshold_df = pd.DataFrame(threshold_rows)

threshold_path = METRICS_DIR / "threshold_analysis.csv"
threshold_df.to_csv(threshold_path, index=False)

# -------------------------------------------------------------------
# 7. Select best tree model
# -------------------------------------------------------------------

tree_models = [
    "Random Forest",
    "XGBoost",
    "Extra Trees",
]

best_tree_name = (
    benchmark_df[
        benchmark_df["Model"].isin(tree_models)
    ]
    .sort_values(
        ["CV F1", "CV Recall", "CV PR-AUC"],
        ascending=False,
    )
    .iloc[0]["Model"]
)

print(f"\nBest tree candidate: {best_tree_name}")

# -------------------------------------------------------------------
# 8. Feature importance
# -------------------------------------------------------------------

best_pipe = fitted_models[best_tree_name]
pre_fitted = best_pipe.named_steps["preprocess"]
best_model = best_pipe.named_steps["model"]

feature_names = []
feature_names.extend(numeric_features)

ohe = (
    pre_fitted
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

cycle_categories = ohe.categories_[0]

feature_names.extend(
    [f"Cycle(R/I)={c}" for c in cycle_categories]
)

feature_names.extend(binary_features)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": best_model.feature_importances_,
    }
).sort_values(
    "Importance",
    ascending=False,
)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

feature_importance_csv = METRICS_DIR / "feature_importance.csv"
importance_df.to_csv(feature_importance_csv, index=False)

# -------------------------------------------------------------------
# 9. SHAP explainability
# -------------------------------------------------------------------

shap_df = None

try:
    import shap

    X_test_transformed = pre_fitted.transform(X_test)

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(
        X_test_transformed
    )

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        arr = np.asarray(shap_values)
        sv = (
            arr[:, :, 1]
            if arr.ndim == 3
            else arr
        )

    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsSHAP": np.abs(sv).mean(axis=0),
        }
    ).sort_values(
        "MeanAbsSHAP",
        ascending=False,
    )

    print("\nSHAP IMPORTANCE")
    print(shap_df.to_string(index=False))

    shap_csv = METRICS_DIR / "shap_mean_abs.csv"
    shap_df.to_csv(shap_csv, index=False)

except Exception as exc:
    print(
        "\nSHAP could not be generated:",
        repr(exc),
    )

# -------------------------------------------------------------------
# 10. Figures
# -------------------------------------------------------------------

# 10a. Model benchmark
plot_df = benchmark_df.set_index("Model")[
    [
        "CV Recall",
        "CV F1",
        "CV PR-AUC",
    ]
]

fig, ax = plt.subplots(figsize=(8, 5))

plot_df.plot(
    kind="bar",
    ax=ax,
)

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("PCOS Model 2 — 5-fold CV comparison")
ax.tick_params(axis="x", rotation=20)

fig.tight_layout()

benchmark_fig = FIGURES_DIR / "model_benchmark.png"
fig.savefig(
    benchmark_fig,
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# 10b. Threshold trade-off
best_threshold_subset = threshold_df[
    threshold_df["Model"] == best_tree_name
]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    best_threshold_subset["Threshold"],
    best_threshold_subset["Precision"],
    marker="o",
    label="Precision",
)

ax.plot(
    best_threshold_subset["Threshold"],
    best_threshold_subset["Recall"],
    marker="o",
    label="Recall",
)

ax.plot(
    best_threshold_subset["Threshold"],
    best_threshold_subset["F1"],
    marker="o",
    label="F1",
)

ax.set_ylim(0, 1)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title(f"{best_tree_name} — threshold trade-off")
ax.legend()

fig.tight_layout()

threshold_fig = FIGURES_DIR / "threshold_tradeoff.png"
fig.savefig(
    threshold_fig,
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# 10c. Feature importance
top_importance = (
    importance_df
    .head(12)
    .sort_values("Importance")
)

fig, ax = plt.subplots(figsize=(7, 5))

ax.barh(
    top_importance["Feature"],
    top_importance["Importance"],
)

ax.set_xlabel("Importance")
ax.set_title(f"{best_tree_name} — feature importance")

fig.tight_layout()

feature_importance_fig = FIGURES_DIR / "feature_importance.png"
fig.savefig(
    feature_importance_fig,
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# 10d. SHAP summary
if shap_df is not None:
    top_shap = (
        shap_df
        .head(12)
        .sort_values("MeanAbsSHAP")
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.barh(
        top_shap["Feature"],
        top_shap["MeanAbsSHAP"],
    )

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{best_tree_name} — SHAP feature importance")

    fig.tight_layout()

    shap_fig = FIGURES_DIR / "shap_summary.png"
    fig.savefig(
        shap_fig,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

# 10e. Logistic Regression confusion matrix
logistic_prob = probabilities["Logistic Regression"]
logistic_pred = (logistic_prob >= 0.50).astype(int)
cm = confusion_matrix(y_test, logistic_pred)

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm)

ax.set_xticks(
    [0, 1],
    labels=["Predicted No PCOS", "Predicted PCOS"],
)

ax.set_yticks(
    [0, 1],
    labels=["Actual No PCOS", "Actual PCOS"],
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

ax.set_title("Model 2 Baseline — Logistic Regression")

fig.tight_layout()

confusion_fig = FIGURES_DIR / "logistic_confusion_matrix.png"
fig.savefig(
    confusion_fig,
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# -------------------------------------------------------------------
# 11. Verify outputs were rewritten
# -------------------------------------------------------------------

print("\nSaved metrics:")
for filename in metric_files:
    path = METRICS_DIR / filename
    if path.exists():
        print(
            f" - {path} | "
            f"size={path.stat().st_size} bytes | "
            f"modified={time.ctime(path.stat().st_mtime)}"
        )

print("\nSaved figures:")
for filename in figure_files:
    path = FIGURES_DIR / filename
    if path.exists():
        print(
            f" - {path} | "
            f"size={path.stat().st_size} bytes | "
            f"modified={time.ctime(path.stat().st_mtime)}"
        )

print("\nDone.")
