"""CycleSignal app core: cycle flags, the combined reading and per-person explanations.

Everything here is computed from the same code the project already uses:
- the 0-100 score comes from cyclesignal_risk_score.calculate_cyclesignal_score
- cycle flags follow the rules in model1/model1_pipeline_v6.py (FIGO regularity,
  Axis 1 "2 of last 3 outside 21-35 days", fewer than 8 periods a year, no period 90+ days)
- the escalation level comes from agent/levels.py (mirrors Model 1's agent_level)

Per-person explanations use SHAP on the fitted Extra Trees model, so the reasons
shown to a user are hers, not the population-wide ranking.

Model 2 has eleven inputs: nine check-in answers plus self-reported regularity
(Cycle(R/I): 2 regular, 4 irregular) and period length in days (how many days bleeding
lasts, the training data's definition). Both are asked, never taken from the watch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "combined_score"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from agent.levels import agent_level  # noqa: E402
from cyclesignal_risk_score import (  # noqa: E402
    BINARY_FEATURES, FEATURES, WARNING, calculate_cyclesignal_score,
    calculate_model2_symptom_score, train_and_evaluate_model2,
)

REGULAR, IRREGULAR = 2, 4                 # Cycle(R/I) codes in the training data
TYPICAL_PERIOD_DAYS = 5                   # median period length for regular cycles

# Model 1 constants (model1_pipeline_v6.py)
CYCLE_SHORT, CYCLE_LONG = 21, 35
OLIGO_WINDOW, OLIGO_MIN = 3, 2
CYCLES_PER_YEAR_MIN = 8
CYCLE_WEIGHT, SYMPTOM_WEIGHT = 0.40, 0.60

# Plain-language names for the eleven Model 2 inputs
LABELS = {
    "Age": "Age", "BMI": "BMI", "Cycle(R/I)": "Periods regular? (self-reported)",
    "Cycle length(days)": "Period length (days)", "Weight gain": "Weight gain",
    "hair growth": "Excess hair growth (face/body)",
    "Skin darkening": "Skin darkening (neck, armpits)",
    "Hair loss": "Hair thinning / loss", "Pimples": "Acne",
    "Fast food": "Frequent fast food", "Reg.Exercise": "Regular exercise",
}


def load_model(repo_root: Path = ROOT):
    """Train Model 2 on its training split only; returns (model, held-out metrics)."""
    return train_and_evaluate_model2(repo_root)


def cycle_flags(lengths: list[int], days_since_last_period: int | None = None) -> dict:
    """Derive Model 1's longitudinal cycle outputs from a period log.

    lengths are completed cycle lengths in days, oldest first.
    """
    lens = pd.Series([float(x) for x in lengths], dtype=float).dropna()
    if len(lens) < 2:
        regularity = "insufficient_history"
    elif lens.between(24, 38).all() and lens.max() - lens.min() <= 9:
        regularity = "regular"            # FIGO 2018
    else:
        regularity = "irregular"
    last = lens.iloc[-OLIGO_WINDOW:]
    outside = int(((last < CYCLE_SHORT) | (last > CYCLE_LONG)).sum())
    axis1_pattern = bool(len(last) >= 2 and outside >= OLIGO_MIN)
    axis1_infrequent = bool(len(lens) >= 3 and 365 / lens.median() < CYCLES_PER_YEAR_MIN)
    no_period_90 = bool(days_since_last_period is not None and days_since_last_period >= 90)
    return {
        "regularity": regularity, "axis1_pattern": axis1_pattern,
        "axis1_infrequent": axis1_infrequent, "no_period_90": no_period_90,
        "cycles_logged": int(len(lens)), "outside_21_35_in_last_3": outside,
        "lengths": [int(x) for x in lens],
    }


class SymptomExplainer:
    """Per-person SHAP for Model 2, reported in profile-score points (0-100 scale).

    SHAP runs on the preprocessed columns; one-hot columns are summed back to their
    original input so each of the eleven answers gets one number.
    """

    def __init__(self, model):
        import shap
        self.model = model
        self.preprocess = model.named_steps["preprocess"]
        names = self.preprocess.get_feature_names_out()
        self.source = [_source_feature(n) for n in names]
        self.explainer = shap.TreeExplainer(model.named_steps["classifier"])
        base = np.asarray(self.explainer.expected_value).ravel()
        self.positive = list(model.named_steps["classifier"].classes_).index(1)
        self.base_points = float(base[self.positive] * 100) if base.size > 1 else float(base[0] * 100)

    def explain(self, features: dict) -> pd.DataFrame:
        row = pd.DataFrame([{name: features[name] for name in FEATURES}])
        values = np.asarray(self.explainer.shap_values(self.preprocess.transform(row)))
        # shap returns (rows, columns, classes) for sklearn forests
        per_column = values[0, :, self.positive] if values.ndim == 3 else values[self.positive][0]
        contrib = pd.Series(per_column, index=self.source).groupby(level=0).sum().reindex(list(FEATURES))
        frame = pd.DataFrame({
            "feature": list(FEATURES),
            "label": [LABELS[f] for f in FEATURES],
            "value": [features[f] for f in FEATURES],
            "symptom_points": contrib.to_numpy() * 100,
        })
        frame["score_points"] = frame["symptom_points"] * SYMPTOM_WEIGHT
        frame["counterintuitive"] = [
            _counterintuitive(f, v, p) for f, v, p in
            zip(frame["feature"], frame["value"], frame["symptom_points"])
        ]
        return frame.sort_values("symptom_points", key=np.abs, ascending=False).reset_index(drop=True)


def assess(model, explainer: SymptomExplainer, features: dict, lengths: list[int],
           days_since_last_period: int | None = None, paused: bool = False,
           temperature_shift_missing: int | None = None) -> dict:
    """One full CycleSignal reading: score, components, per-person drivers and agent level.

    Model 2 is cycle-neutral (no cycle inputs), so its flag counts as the androgen domain
    without double-counting cycle evidence.
    """
    flags = cycle_flags(lengths, days_since_last_period)
    if paused:
        # cycle checks pause on hormonal contraception / pregnancy; score uses unknown history
        flags.update(regularity="insufficient_history", axis1_pattern=False,
                     axis1_infrequent=False, no_period_90=False)
    score = calculate_cyclesignal_score(
        model, features, regularity=flags["regularity"],
        axis1_pattern=flags["axis1_pattern"], axis1_infrequent=flags["axis1_infrequent"],
        no_period_90=flags["no_period_90"],
    )
    drivers = explainer.explain(features)
    model2_flag = score["model2_symptom_score"] >= 50.0      # the evaluated 0.50 threshold
    # Androgen domain: Model 2 must still flag with her cycle answers set to regular,
    # so cycle evidence is not counted twice (Model 1 agent_level design)
    neutral = features | {"Cycle(R/I)": REGULAR, "Cycle length(days)": TYPICAL_PERIOD_DAYS}
    model2_flag_cycle_neutral = calculate_model2_symptom_score(model, neutral) >= 50.0
    level, message, reasons = agent_level(
        model2_flag_cycle_neutral=model2_flag_cycle_neutral, axis1_pattern=flags["axis1_pattern"],
        axis1_infrequent=flags["axis1_infrequent"], no_period_90=flags["no_period_90"],
        hormonal_contraception=paused,
    )
    return {
        "score": score, "flags": flags, "drivers": drivers,
        "symptom_base_points": explainer.base_points,
        "cycle_points": round(CYCLE_WEIGHT * score["model1_cycle_score"], 1),
        "symptom_points": round(SYMPTOM_WEIGHT * score["model2_symptom_score"], 1),
        "model2_flag": model2_flag, "model2_flag_cycle_neutral": model2_flag_cycle_neutral, "level": level, "level_message": message,
        "level_reasons": reasons, "paused": paused,
        "temperature_shift_missing": temperature_shift_missing,
        "features": dict(features), "warning": WARNING,
    }


def facts_for_agent(reading: dict, top_n: int = 5) -> dict:
    """The only information the agent sees: numbers from the models, nothing invented."""
    d = reading["drivers"]
    top = d.head(top_n)
    return {
        "score": reading["score"]["score"], "band": reading["score"]["band"],
        "cycle_component": {
            "model1_cycle_score_0_100": reading["score"]["model1_cycle_score"],
            "points_in_total": reading["cycle_points"], "weight": "40%",
            "cycle_lengths_days_oldest_first": reading["flags"]["lengths"],
            "regularity": reading["flags"]["regularity"],
            "outside_21_35_in_last_3": reading["flags"]["outside_21_35_in_last_3"],
            "fewer_than_8_periods_a_year": reading["flags"]["axis1_infrequent"],
            "no_period_90_days": reading["flags"]["no_period_90"],
            "history_note": reading["score"]["history_note"],
            "cycle_checks_paused": reading["paused"],
        },
        "symptom_component": {
            "what_it_is": "Model 2 profile from her check-in: symptoms, lifestyle, age, BMI and her own "
                          "answers about cycle regularity and period length. Cycle information overlaps with "
                          "the 40% cycle component.",
            "model2_symptom_score_0_100": reading["score"]["model2_symptom_score"],
            "points_in_total": reading["symptom_points"], "weight": "60%",
            "model_average_symptom_score": round(reading["symptom_base_points"], 1),
            "per_person_drivers_in_symptom_points": [
                {"input": r.label, "her_answer": answer_text(r.feature, r.value),
                 "effect_points": round(float(r.symptom_points), 1),
                 "data_quirk_not_advice": bool(r.counterintuitive)}
                for r in top.itertuples()
            ],
        },
        "temperature_context": {
            "cycles_without_clear_temperature_rise": reading["temperature_shift_missing"],
            "in_score": False,
            "note": "Wrist temperature is used by Model 1 to estimate cycle phase. Its ovulation check did "
                    "not pass validation in our data, so it is context only and never raises a flag.",
        },
        "agent_level": reading["level"], "agent_level_reasons": reading["level_reasons"],
        "cannot_see": ["hormone levels (testosterone, LH, FSH, AMH)", "blood sugar / insulin",
                       "ovarian ultrasound", "thyroid or prolactin causes of irregular cycles"],
        "warning": reading["warning"],
    }


def _counterintuitive(feature: str, value, points: float) -> bool:
    """True when the model pushes the opposite way to clinical expectation,
    e.g. reporting acne lowers the score. These are training-data quirks, not advice."""
    if feature in ("Age", "BMI", "Cycle length(days)") or abs(points) < 1:
        return False
    if feature == "Cycle(R/I)":
        expected_up = int(value) == IRREGULAR
    else:
        raises_when_yes = feature != "Reg.Exercise"
        expected_up = (int(value) == 1) == raises_when_yes
    return (points > 0) != expected_up


def answer_text(feature: str, value) -> str:
    """How her answer reads in plain words."""
    if feature in ("Age", "BMI"):
        return f"{value:g}"
    if feature == "Cycle(R/I)":
        return {REGULAR: "regular", IRREGULAR: "irregular"}.get(int(value), f"code {int(value)}")
    if feature == "Cycle length(days)":
        return f"{value:g} days"
    return "yes" if int(value) == 1 else "no"


def _source_feature(column: str) -> str:
    """Map a preprocessed column name (e.g. 'cat__Cycle(R/I)_4') back to its input."""
    name = column.split("__", 1)[-1]
    return "Cycle(R/I)" if name.startswith("Cycle(R/I)") else name
