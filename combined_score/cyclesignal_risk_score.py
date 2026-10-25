from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


FEATURES = (
    "Age", "BMI", "Cycle(R/I)", "Cycle length(days)", "Weight gain", "hair growth", "Skin darkening",
    "Hair loss", "Pimples", "Fast food", "Reg.Exercise",
)
BINARY_FEATURES = FEATURES[4:]
DATA_PATH = Path("model2/data/pcos_model2_selected_features.csv")
WARNING = (
    "Prototype risk-awareness score only: not a probability of disease or "
    "a diagnosis. The 40/60 weights and bands are illustrative, not learned or clinically validated. Model 2 already includes cycle inputs, so cycle information overlaps. "
    "A low score does not rule out PCOS. Discuss concerning symptoms or "
    "cycle changes with a qualified healthcare professional."
)

# Accept the project's short names and common original dataset headers.
ALIASES = {
    "Age (yrs)": "Age", "Weight gain(Y/N)": "Weight gain",
    "hair growth(Y/N)": "hair growth", "Hair growth": "hair growth",
    "Skin darkening (Y/N)": "Skin darkening",
    "Hair loss(Y/N)": "Hair loss", "Pimples(Y/N)": "Pimples",
    "Fast food (Y/N)": "Fast food", "Reg.Exercise(Y/N)": "Reg.Exercise",
    "Regular exercise": "Reg.Exercise", "PCOS (Y/N)": "PCOS",
    "PCOS(Y/N)": "PCOS",
}


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [ALIASES.get(str(c).strip(), str(c).strip()) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate columns after normalising feature names.")
    return frame


def _validate_features(frame: pd.DataFrame, *, allow_missing: bool) -> pd.DataFrame:
    frame = _normalise_columns(frame)
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    # Explicit allow-list excludes lab, ultrasound and target data.
    values = frame.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("Features must not contain infinity.")
    if not allow_missing and values.isna().any().any():
        raise ValueError("All eleven features, including cycle category and cycle length, are required for scoring.")
    if (values[["Age", "BMI"]] <= 0).any().any():
        raise ValueError("Age and BMI must be positive.")
    if not values["Cycle(R/I)"].dropna().isin([2, 4, 5]).all():
        raise ValueError("Cycle(R/I) must use the training dataset codes 2, 4 or 5; do not infer codes from regularity.")
    if (values["Cycle length(days)"].dropna() < 0).any():
        raise ValueError("Cycle length(days) must be nonnegative and use the training dataset definition.")
    for name in BINARY_FEATURES:
        if not values[name].dropna().isin([0, 1]).all():
            raise ValueError(f"{name} must use 0=no, 1=yes.")
    return values


def calculate_model1_cycle_score(
    regularity: str | None,
    axis1_pattern: bool = False,
    axis1_infrequent: bool = False,
    no_period_90: bool = False,
) -> float:
    """Score already-derived longitudinal outputs; do not infer from one cycle.

    Priority: no_period_90=100, infrequent=85, persistent pattern=75,
    irregular=45, regular=10, unknown/insufficient history=20.
    Flags must be actual booleans, never strings such as 'False'.
    """
    for flag in (axis1_pattern, axis1_infrequent, no_period_90):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("Model 1 flags must be booleans.")
    if regularity is not None and not isinstance(regularity, str):
        raise ValueError("regularity must be a string or None.")
    state = regularity.strip().lower() if regularity is not None else "unknown"
    if state not in {"regular", "irregular", "unknown", "insufficient_history"}:
        raise ValueError("Unsupported regularity; use regular, irregular, unknown, or insufficient_history.")
    if no_period_90:
        return 100.0
    if axis1_infrequent:
        return 85.0
    if axis1_pattern:
        return 75.0
    return {"regular": 10.0, "irregular": 45.0}.get(state, 20.0)


def find_repository_root(repo_root: str | Path | None = None) -> Path:
    """Resolve from an explicit root, this file's ancestors, or working directory."""
    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        if not (root / DATA_PATH).is_file():
            raise FileNotFoundError(f"Required dataset not found: {root / DATA_PATH}")
        return root
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for root in (start, *start.parents):
            if (root / DATA_PATH).is_file():
                return root
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}. Put this script in the repository or pass --repo-root."
    )


def train_and_evaluate_model2(
    repo_root: str | Path | None = None, *, target_column: str = "PCOS",
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """Train the final eleven-feature Extra Trees pipeline and evaluate it.

    Preprocessing and classifier settings match model2/build_pcos_model2.py.
    Missing targets, nonnumeric predictors and invalid binary codes fail fast.
    No tuning is performed on the test set. This evaluates Model 2 only;
    the combined longitudinal score requires separate external validation.
    """
    path = find_repository_root(repo_root) / DATA_PATH
    data = _normalise_columns(pd.read_csv(path))
    target = ALIASES.get(target_column.strip(), target_column.strip())
    if target in FEATURES:
        raise ValueError("Target must not be one of the symptom predictors.")
    if target not in data:
        raise ValueError(f"Target column {target!r} not found. Available: {list(data.columns)}")
    X = _validate_features(data, allow_missing=True)
    y = pd.to_numeric(data[target], errors="raise")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain both classes encoded as 0 and 1, without missing values.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.20, stratify=y, random_state=random_state,
    )
    if X_train.isna().all().any():
        raise ValueError("A training feature has no observed values.")
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["Age", "BMI", "Cycle length(days)"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["Cycle(R/I)"]),
        ("bin", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), list(BINARY_FEATURES)),
    ], sparse_threshold=0)
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state,
        )),
    ])
    model.fit(X_train, y_train)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]
    predictions = (probabilities >= 0.50).astype(int)
    metrics = {
        "training_rows": len(X_train), "test_rows": len(X_test),
        "features": list(FEATURES), "threshold": 0.50,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "scope": "Held-out final eleven-feature Extra Trees Model 2 only. Combined score and 40/60 weights are not validated; cycle information overlaps between components.",
    }
    return model, metrics


def calculate_model2_symptom_score(model: Pipeline, user_features: Mapping) -> float:
    """Return positive-class Extra Trees output x100, not disease probability."""
    check_is_fitted(model)
    if list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("Model must be trained on exactly the eleven final Model 2 features in order.")
    frame = _validate_features(pd.DataFrame([dict(user_features)]), allow_missing=False)
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Model must use binary classes 0 and 1.")
    probability = float(model.predict_proba(frame)[0, classes.index(1)])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid classifier output.")
    return probability * 100.0


def calculate_cyclesignal_score(
    model: Pipeline, user_features: Mapping, *, regularity: str | None,
    axis1_pattern: bool = False, axis1_infrequent: bool = False,
    no_period_90: bool = False,
) -> dict:
    """Combine 40% cycle pattern + 60% full Model 2 profile signal and return display data.

    Bands are continuous intervals [0,30), [30,50), [50,70), [70,100].
    Classify the displayed one-decimal score so the label matches the display.
    """
    cycle = calculate_model1_cycle_score(
        regularity, axis1_pattern, axis1_infrequent, no_period_90,
    )
    symptom = calculate_model2_symptom_score(model, user_features)
    score = round(0.40 * cycle + 0.60 * symptom, 1)
    band = (
        "Low pattern signal" if score < 30 else
        "Some patterns to monitor" if score < 50 else
        "Increased pattern signal" if score < 70 else
        "Elevated pattern signal"
    )
    unknown = regularity is None or regularity.strip().lower() in {"unknown", "insufficient_history"}
    return {
        "name": "CycleSignal PCOS Risk Awareness Score",
        "score": score, "out_of": 100, "band": band,
        "model1_cycle_score": cycle, "model2_symptom_score": round(symptom, 1),
        "cycle_history_unknown": unknown,
        "warning": WARNING,
        "history_note": "Unknown history uses a heuristic fallback of 20; interpret cautiously." if unknown else None,
        "cycle_follow_up": "Discuss the reported 90-day absence of a period with a healthcare professional regardless of the score." if no_period_90 else None,
    }


def load_model1_cycle_outputs(
    repo_root: str | Path, participant_id: str,
    handoff_path: str | Path | None = None,
) -> dict:
    """Read one participant's diary outputs, never population summaries.

    Model 1 exports this CSV relative to its working directory. Both common
    run locations are supported; ambiguous files require an explicit path.
    """
    root = Path(repo_root).resolve()
    if handoff_path is not None:
        path = Path(handoff_path)
        path = path if path.is_absolute() else root / path
    else:
        candidates = [root / "data/cycle_handoff_v6.csv",
                      root / "model1/data/cycle_handoff_v6.csv"]
        found = [candidate for candidate in candidates if candidate.is_file()]
        if len(found) > 1:
            raise ValueError("Multiple Model 1 handoffs found; select --model1-handoff explicitly.")
        if not found:
            raise FileNotFoundError(
                "Model 1 handoff is missing. Run model1/model1_pipeline_v6.py "
                "with the repository as its working directory to generate "
                "data/cycle_handoff_v6.csv. Summary files in model1/results "
                "cannot supply individual cycle inputs."
            )
        path = found[0]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "regularity", "axis1_pattern", "axis1_infrequent", "no_period_90"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Model 1 handoff is missing columns: {sorted(missing)}")
    ids = frame["id"].str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Model 1 handoff requires unique, nonempty participant ids.")
    selected = frame.loc[ids == str(participant_id).strip()]
    if len(selected) != 1:
        raise ValueError("Participant was not found in the Model 1 handoff.")
    row = selected.iloc[0]
    state = row["regularity"].strip().lower()
    if state == "unknown (under 2 cycles)":
        state = "insufficient_history"
    outputs = {"regularity": state}
    for name in ("axis1_pattern", "axis1_infrequent", "no_period_90"):
        text = row[name].strip().lower()
        if text not in {"true", "false", "1", "0"}:
            raise ValueError(f"Invalid or missing Model 1 boolean: {name}")
        outputs[name] = text in {"true", "1"}
    calculate_model1_cycle_score(**outputs)  # Validate before using the handoff.
    return outputs


SIMULATION_NOTICE = (
    "Simulated combined profile: cycle and symptom inputs come from different "
    "people in separate datasets. Pairing is arbitrary and reproducible; "
    "this is not a real participant assessment or validation of combined accuracy. "
    "Symptom rows may have been used to train Model 2. Source diagnosis labels "
    "are not labels for these simulated profiles. Model 2 cycle inputs come from "
    "its source row and may disagree with the independently paired Model 1 profile."
)


def calculate_simulated_combined_scores(
    model: Pipeline, repo_root: str | Path, *, handoff_path=None,
    random_state: int = 42, count: int | None = None,
) -> dict:
    """Pair each saved cycle profile with a shuffled existing symptom profile.

    No identity matching is claimed. Diagnosis labels never enter scoring or
    pairing. Model 2 uses its source row's cycle predictors; these need not
    agree with the independently paired Model 1 cycle profile.
    Results remain in memory; no participant data or files are written.
    """
    root = find_repository_root(repo_root)
    if handoff_path is not None:
        path = Path(handoff_path)
        path = path if path.is_absolute() else root / path
    else:
        candidates = [root / "data/cycle_handoff_v6.csv",
                      root / "model1/data/cycle_handoff_v6.csv"]
        found = [path for path in candidates if path.is_file()]
        if len(found) != 1:
            raise ValueError("Expected one saved Model 1 handoff; use --model1-handoff to select it.")
        path = found[0]
    cycles = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "id" not in cycles or cycles.empty:
        raise ValueError("Model 1 handoff must contain participant rows and an id column.")
    ids = cycles["id"].str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Model 1 handoff ids must be unique and nonempty.")
    symptom_data = _validate_features(pd.read_csv(root / DATA_PATH), allow_missing=True)
    usable = symptom_data.dropna()
    if usable.empty:
        raise ValueError("No complete symptom profiles are available for the simulation.")
    size = len(cycles) if count is None else count
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= len(cycles):
        raise ValueError(f"Demo count must be between 1 and {len(cycles)}.")
    if size > len(usable):
        raise ValueError("Not enough complete symptom rows to pair without replacement.")
    rng = np.random.default_rng(random_state)
    positions = rng.choice(len(usable), size=size, replace=False)
    profiles = []
    for number, (cycle_id, position) in enumerate(zip(ids.iloc[:size], positions), start=1):
        outputs = load_model1_cycle_outputs(root, cycle_id, path)
        symptom_row = usable.iloc[int(position)]
        result = calculate_cyclesignal_score(model, symptom_row.to_dict(), **outputs)
        result.update({
            "profile_id": f"SIMULATED-{number:03d}",
            "simulated": True,
            "simulation_notice": SIMULATION_NOTICE,
            "cycle_source": "Model 1 handoff; input row " + str(number),
            "symptom_source": "Model 2 CSV; input row " + str(int(symptom_row.name) + 1),
            "source_rows_are_same_person": False,
        })
        profiles.append(result)
    return {
        "mode": "simulated_combined_profiles",
        "notice": SIMULATION_NOTICE,
        "warning": WARNING,
        "pairing_seed": random_state,
        "profile_count": len(profiles),
        "available_cycle_profiles": len(cycles),
        "available_complete_symptom_profiles": len(usable),
        "profiles": profiles,
    }


def canonical_model2_evaluation(repo_root: str | Path) -> dict:
    """Read the saved final benchmark for comparison with a fresh evaluation."""
    root = Path(repo_root)
    benchmark = pd.read_csv(root / "model2/results/metrics/four_model_benchmark.csv")
    rows = benchmark.loc[benchmark["Model"] == "Extra Trees"]
    if len(rows) != 1:
        raise ValueError("Expected exactly one final Extra Trees benchmark row.")
    row = rows.iloc[0]
    columns = pd.read_csv(root / DATA_PATH, nrows=0).columns
    features = [name for name in columns if name != "PCOS (Y/N)"]
    tn, fp, fn, tp = (int(row[name]) for name in ("TN", "FP", "FN", "TP"))
    test_rows = tn + fp + fn + tp
    return {
        "training_rows": len(pd.read_csv(root / DATA_PATH)) - test_rows,
        "test_rows": test_rows,
        "features": features,
        "threshold": 0.5,
        "accuracy": float(row["Test Accuracy"]),
        "precision": float(row["Test Precision"]),
        "recall": float(row["Test Recall"]),
        "f1": float(row["Test F1"]),
        "roc_auc": float(row["Test ROC-AUC"]),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "scope": "Canonical final 11-feature Extra Trees held-out evaluation at threshold 0.5; source: model2/results/metrics/four_model_benchmark.csv. The demo now uses the same eleven-feature pipeline. Combined score and 40/60 weights are not validated; cycle information overlaps between components.",
    }


def save_simulated_scores_json(simulation: dict, evaluation: dict,
                               repo_root: str | Path, output_path=None) -> Path:
    """Save bot-readable demo data, including warnings and Model 2 evaluation.

    Relative destinations resolve against the repository, not the current
    working directory. Re-running replaces the previous demo export.
    """
    root = Path(repo_root).resolve()
    destination = Path(output_path) if output_path is not None else Path("combined_score/results/cyclesignal_demo_scores.json")
    if not destination.is_absolute():
        destination = root / destination
    document = dict(simulation)
    document.update({
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Bot demonstration only; simulated profiles are not real participant assessments or training labels.",
        "score_weights": {"model1_cycle": 0.40, "model2_symptom": 0.60},
        "model2_evaluation": evaluation,
    })
    # Serialize before touching the destination; reject nonstandard NaN/Infinity.
    serialized = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized + "\n", encoding="utf-8")
    return destination.resolve()


def prompt_for_participant_inputs(root: Path, handoff_path=None, participant_id=None):
    """Collect same-person inputs in memory; never create symptom files."""
    if participant_id is None:
        if handoff_path is not None:
            path = Path(handoff_path)
            path = path if path.is_absolute() else root / path
        else:
            paths = [root / "data/cycle_handoff_v6.csv",
                     root / "model1/data/cycle_handoff_v6.csv"]
            found = [path for path in paths if path.is_file()]
            if len(found) != 1:
                raise ValueError("Expected one Model 1 handoff. Use --model1-handoff to select an existing CSV.")
            path = found[0]
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "id" not in table or "regularity" not in table:
            raise ValueError("Model 1 handoff requires id and regularity columns.")
        print("\nAvailable Model 1 participants (local only):")
        print(table[["id", "regularity"]].to_string(index=False))
        participant_id = input("\nEnter the participant id whose symptoms you will supply: ").strip()
    cycle_outputs = load_model1_cycle_outputs(root, participant_id, handoff_path)
    print("\nSelected participant's Model 1 inputs:")
    print(json.dumps(cycle_outputs, indent=2))
    print("Enter all eleven Model 2 inputs for this SAME person. Binary answers: 0=no, 1=yes.")
    print("Cycle(R/I) requires dataset codes 2/4/5. Cycle length(days) must match the training definition; do not substitute wearable cycle intervals without checking compatibility.")
    print("If those inputs are unavailable, cancel with Ctrl+C; do not substitute training rows.")
    symptoms = {}
    for feature in FEATURES:
        while True:
            answer = input(f"{feature}: ").strip()
            try:
                value = float(answer)
                if not np.isfinite(value):
                    raise ValueError
                if feature in BINARY_FEATURES and value not in (0, 1):
                    raise ValueError
                if feature in ("Age", "BMI") and value <= 0:
                    raise ValueError
                if feature == "Cycle(R/I)" and value not in (2, 4, 5):
                    raise ValueError
                if feature == "Cycle length(days)" and value < 0:
                    raise ValueError
            except ValueError:
                print("Enter 0 or 1." if feature in BINARY_FEATURES else "Enter a positive number.")
                continue
            symptoms[feature] = value
            break
    return cycle_outputs, symptoms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--target-column", default="PCOS")
    parser.add_argument("--model1-handoff", type=Path, help="Absolute or repository-relative CSV path")
    parser.add_argument("--participant-id", help="Exact id in Model 1 handoff")
    parser.add_argument("--symptoms-json", type=Path,
                        help="Optional JSON file for the SAME participant; otherwise prompt in terminal")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate Model 2 without participant prompts")
    parser.add_argument("--interactive", action="store_true", help="Enter same-person symptom inputs instead of the default simulation")
    parser.add_argument("--demo-count", type=int, help="Number of simulated profiles; default: all Model 1 rows")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible simulation pairing seed")
    parser.add_argument("--output-json", type=Path, help="Demo export path; default: combined_score/results/cyclesignal_demo_scores.json, relative to repository")
    args = parser.parse_args()
    print(WARNING)
    try:
        root = find_repository_root(args.repo_root)
        classifier, evaluation = train_and_evaluate_model2(root, target_column=args.target_column)
        print("\nFresh held-out final eleven-feature Extra Trees evaluation:")
        print(json.dumps(evaluation, indent=2))
        print("\nCanonical final 11-feature Model 2 evaluation:")
        print(json.dumps(canonical_model2_evaluation(root), indent=2))
        if args.evaluate_only:
            parser.exit(0)
        if args.symptoms_json is None and not args.interactive and args.participant_id is None:
            simulation = calculate_simulated_combined_scores(
                classifier, root, handoff_path=args.model1_handoff,
                random_state=args.seed, count=args.demo_count,
            )
            print("\nSIMULATED COMBINED PROFILES - DIFFERENT PEOPLE, NOT REAL PARTICIPANT SCORES:")
            print(json.dumps(simulation, indent=2))
            saved_path = save_simulated_scores_json(simulation, evaluation, root, args.output_json)
            message = f"\nDemo JSON saved to: {saved_path}"
            print(message.encode(sys.stdout.encoding or "utf-8", errors="backslashreplace").decode(sys.stdout.encoding or "utf-8"))
            parser.exit(0)
        if args.demo_count is not None:
            raise ValueError("--demo-count applies only to simulation mode.")
        if args.symptoms_json is not None:
            if args.participant_id is None:
                raise ValueError("--symptoms-json requires --participant-id.")
            cycle_outputs = load_model1_cycle_outputs(root, args.participant_id, args.model1_handoff)
            symptom_path = args.symptoms_json
            if not symptom_path.is_absolute():
                symptom_path = root / symptom_path
            payload = json.loads(symptom_path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict) or set(payload) != {"participant_id", "features"}:
                raise ValueError("Symptom JSON must contain participant_id and features.")
            if str(payload["participant_id"]).strip() != args.participant_id.strip():
                raise ValueError("Symptom participant_id does not match the selected Model 1 participant.")
            if not isinstance(payload["features"], dict):
                raise ValueError("Symptom features must be a JSON object.")
            symptoms = payload["features"]
        else:
            if not sys.stdin.isatty():
                raise ValueError("Run in an interactive terminal to enter symptoms, or use --participant-id with --symptoms-json. Use --evaluate-only for training metrics.")
            cycle_outputs, symptoms = prompt_for_participant_inputs(
                root, args.model1_handoff, args.participant_id,
            )
        result = calculate_cyclesignal_score(classifier, symptoms, **cycle_outputs)
        print("\nScore using saved Model 1 outputs and supplied symptom inputs:")
        print(json.dumps(result, indent=2))
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "\nCancelled. No combined score calculated; no inputs saved.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot calculate CycleSignal score: {exc}\n")

