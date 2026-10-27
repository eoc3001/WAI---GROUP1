#!/usr/bin/env python3
"""Reproduce the WAI---GROUP1 final Model 2 held-out threshold comparison.

Requirements: Python 3.9+, numpy, pandas, scikit-learn>=1.2.
Optional PNG export also requires matplotlib.
Install: python -m pip install numpy pandas 'scikit-learn>=1.2' matplotlib

Drop this file into eoc3001/WAI---GROUP1 (or its src/ directory), then:
    python model2_threshold_check.py
    python model2_threshold_check.py --save
From elsewhere, point to your local checkout:
    python model2_threshold_check.py --repo-root path/to/WAI---GROUP1 --save

All data/output paths are relative to the checkout, never a user's machine.
No outputs are written unless --save is supplied. Export overwrites only the
four clearly named outputs from this script. Metrics are computed from data;
previously reported counts are not hard-coded.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path('data/processed/pcos_model2_selected_features.csv')
NUMERIC = ['Age (yrs)', 'BMI', 'Cycle length(days)']
CATEGORICAL = ['Cycle(R/I)']
BINARY = [
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
    'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)',
]
# Preserve the feature order of the final Model 2 source.
FEATURES = ['Age (yrs)', 'BMI', 'Cycle(R/I)', 'Cycle length(days)'] + BINARY
TARGET = 'PCOS (Y/N)'
THRESHOLDS = (0.50, 0.40, 0.30)


def find_repo_root(explicit_root=None):
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if not (root / DATA_PATH).is_file():
            raise FileNotFoundError(f'Missing input: {root / DATA_PATH}')
        return root
    # Supports checkout root, src/, notebooks/, and invocation from the repo.
    for anchor in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for root in (anchor, *anchor.parents):
            if (root / DATA_PATH).is_file():
                return root
    raise FileNotFoundError(
        f'Cannot find {DATA_PATH}. Put this script in WAI---GROUP1 or use '
        '--repo-root path/to/WAI---GROUP1.'
    )


def build_pipeline():
    preprocess = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERIC),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
        ('bin', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ]), BINARY),
    ])
    return Pipeline([
        ('preprocess', preprocess),
        ('model', ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight='balanced', random_state=42,
        )),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, help='Local WAI---GROUP1 checkout root')
    parser.add_argument('--save', action='store_true', help='Save metrics CSV and three confusion matrix PNGs')
    args = parser.parse_args()
    try:
        root = find_repo_root(args.repo_root)
        df = pd.read_csv(root / DATA_PATH)
        missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
        if missing:
            raise ValueError(f'Missing required columns: {missing}')
        target = pd.to_numeric(df[TARGET], errors='raise')
        if target.isna().any() or set(target.unique()) != {0, 1}:
            raise ValueError(f'{TARGET} must contain both 0 and 1, with no missing labels.')
        X = df[FEATURES].copy()
        y = target.astype(int).copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y,
        )
        model = build_pipeline()
        model.fit(X_train, y_train)
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, positive_index]

        rows, matrices = [], []
        for threshold in THRESHOLDS:
            predicted = (probabilities >= threshold).astype(int)
            cm = confusion_matrix(y_test, predicted, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            rows.append({
                'threshold': threshold,
                'precision': precision_score(y_test, predicted, zero_division=0),
                'recall': recall_score(y_test, predicted, zero_division=0),
                'F1': f1_score(y_test, predicted, zero_division=0),
                'accuracy': accuracy_score(y_test, predicted),
                'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn),
            })
            matrices.append(cm)

        results = pd.DataFrame(rows)
        baseline = accuracy_score(y_test, np.zeros(len(y_test), dtype=int))
        print(f'Repository: {root}\nInput: {root / DATA_PATH}')
        print(f'Train: {len(y_train)} | Held-out test: {len(y_test)} '
              f'({int((y_test == 0).sum())} no PCOS, {int((y_test == 1).sum())} PCOS)')
        print('\nExtra Trees held-out threshold comparison (positive if probability >= threshold)')
        formatters = {c: '{:.1%}'.format for c in ['precision', 'recall', 'F1', 'accuracy']}
        formatters['threshold'] = '{:.2f}'.format
        print(results.to_string(index=False, formatters=formatters))
        print(f'\nNaive always-no-PCOS baseline accuracy (same test set): {baseline:.2%}')
        for threshold, cm in zip(THRESHOLDS, matrices):
            print(f'\nThreshold {threshold:.2f} | rows = actual, columns = predicted')
            print(pd.DataFrame(cm, index=['Actual no PCOS', 'Actual PCOS'],
                               columns=['Pred no PCOS', 'Pred PCOS']).to_string())
        print('\nHeld-out descriptive comparison; threshold selection should use validation data.')

        if args.save:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            metrics_dir, figures_dir = root / 'results/metrics', root / 'results/figures'
            metrics_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)
            csv_path = metrics_dir / 'model2_extra_trees_threshold_check.csv'
            exported = results.copy()
            exported['always_no_pcos_baseline_accuracy'] = baseline
            exported.to_csv(csv_path, index=False)
            print(f'\nSaved: {csv_path} (metric values are fractions from 0 to 1)')
            for threshold, cm in zip(THRESHOLDS, matrices):
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(cm, display_labels=['No PCOS', 'PCOS']).plot(
                    ax=ax, cmap='Blues', colorbar=False, values_format='d',
                )
                ax.set_title(f'Model 2 Extra Trees | Threshold {threshold:.2f}')
                fig.tight_layout()
                tag = f'{threshold:.2f}'.replace('.', 'p')
                path = figures_dir / f'model2_extra_trees_confusion_matrix_threshold_{tag}.png'
                fig.savefig(path, dpi=180, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved: {path}')
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
