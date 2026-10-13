"""
WAI Group 001 | Model 1: turn the raw mcPHASES files into one row per participant-day.

Step 1, look at the real headers (prints column names and 3 example rows per file):
    python build_daily_table.py --raw data/mcphases --inspect

Step 2, edit FILES below so each right-hand name matches a real column, then build:
    python build_daily_table.py --raw data/mcphases --out data/daily.csv

Every column name marked CHECK is a best guess based on Fitbit's standard export.
The mcPHASES files are behind a data use agreement, so they could not be checked
when this script was written. The script stops with a clear message if a column
is missing, instead of guessing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["id", "day_in_study"]  # every mcPHASES table links on these two columns

# target name -> source column, plus how to collapse several rows on the same day
FILES = {
    "labels": {
        "file": "hormones_and_selfreport.csv",
        "cols": {"phase": "phase",                    # CHECK: phase label column
                 "flow": "flow_volume",               # CHECK: period flow column (diary)
                 "flow_color": "flow_color",          # CHECK: fills flow on days where only the colour was logged
                 "lh": "lh",                          # CHECK: LH value (answer key only, never a model input)
                 "pdg": "pdg"},                       # CHECK: PdG value (answer key only, never a model input)
        "agg": "first",
        "required": ["phase"],
    },
    "temperature": {
        "file": "computed_temperature.csv",           # nightly summary; wrist_temperature.csv only as fallback
        "cols": {"night_temp": "nightly_temperature"},  # CHECK
        "agg": "median",
        "required": ["night_temp"],
    },
    "resting_hr": {
        "file": "resting_heart_rate.csv",
        "cols": {"rhr": "value"},                     # CHECK
        "agg": "mean",
        "required": ["rhr"],
    },
    "sleep": {
        "file": "sleep.csv",
        "cols": {"minutes_asleep": "minutesasleep",   # CHECK
                 "time_in_bed": "timeinbed"},         # CHECK
        "main_sleep_col": "mainsleep",                # CHECK: True for the main sleep, False for naps
        "agg": "sum",
        "required": ["minutes_asleep", "time_in_bed"],
    },
    # optional files: add one at a time, only after the core model works
    "resp_rate": {
        "file": "respiratory_rate_summary.csv",
        "cols": {"resp_rate": "full_sleep_breathing_rate"},  # CHECK
        "agg": "median", "required": [], "optional": True,
    },
    "hrv": {
        "file": "heart_rate_variability_details.csv",
        "cols": {"rmssd": "rmssd"},                   # CHECK; many readings per night -> one median
        "agg": "median", "required": [], "optional": True,
    },
    "activity": {
        "file": "active_minutes.csv",
        "cols": {"active_minutes": "very"},           # CHECK: pick the column(s) for active minutes
        "agg": "sum", "required": [], "optional": True,
    },
}


def inspect(raw: Path) -> None:
    """Print the header and 3 rows of every csv, without loading big files fully."""
    for f in sorted(Path(raw).glob("*.csv")):
        head = pd.read_csv(f, nrows=3)
        size_mb = f.stat().st_size / 1e6
        print(f"\n=== {f.name}  ({size_mb:,.1f} MB)")
        print("columns:", list(head.columns))
        print(head.to_string(index=False))


def _load(raw: Path, spec: dict) -> pd.DataFrame | None:
    path = Path(raw) / spec["file"]
    if not path.exists():
        if spec.get("optional"):
            return None
        raise SystemExit(f"Missing file {path}. Put the mcPHASES csv files in {raw}.")
    header = pd.read_csv(path, nrows=0).columns
    by_lower = {str(h).strip().lower(): h for h in header}   # match names ignoring case and spaces
    required = spec.get("required", [])
    cols = {}
    for target, source in spec["cols"].items():
        if str(source).strip().lower() in by_lower:
            cols[target] = by_lower[str(source).strip().lower()]
        elif target in required:
            raise SystemExit(f"{spec['file']}: column '{source}' not found. Run with --inspect and update FILES.")
        else:
            loud = "  <-- needed as the ovulation answer key by model1_pipeline_v4.py" if target in ("lh", "pdg") else ""
            print(f"{spec['file']}: '{source}' not found, so '{target}' is skipped (fix FILES to include it).{loud}")
    if not cols:
        return None
    spec = {**spec, "cols": cols}
    if "main_sleep_col" in spec and spec["main_sleep_col"].lower() in by_lower:
        spec["main_sleep_col"] = by_lower[spec["main_sleep_col"].lower()]
    needed = KEYS + list(cols.values()) + ([spec["main_sleep_col"]] if "main_sleep_col" in spec else [])
    missing = [c for c in needed if c not in header]
    if missing:
        raise SystemExit(f"{spec['file']}: columns {missing} not found. Run with --inspect and update FILES.")
    df = pd.read_csv(path, usecols=needed)
    if "main_sleep_col" in spec:  # naps distort sleep totals and night-time signals
        keep = df[spec["main_sleep_col"]].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[keep].drop(columns=spec["main_sleep_col"])
    df = df.rename(columns={v: k for k, v in spec["cols"].items()})
    return df.groupby(KEYS, as_index=False).agg(spec["agg"])


def build_daily_table(raw: Path, include_optional: bool = True) -> pd.DataFrame:
    """Keep every day that appears in any file. Days without a phase label stay, with phase left blank,
    so a missed diary or hormone entry never looks like a gap in the cycle."""
    daily = _load(raw, FILES["labels"])
    for name, spec in FILES.items():
        if name == "labels" or (spec.get("optional") and not include_optional):
            continue
        part = _load(raw, spec)
        if part is not None:
            daily = daily.merge(part, on=KEYS, how="outer")
    daily = daily.sort_values(KEYS).reset_index(drop=True)
    labelled = daily["phase"].notna()
    print(f"Daily table: {daily['id'].nunique()} participants, {len(daily)} days, "
          f"{labelled.sum()} with a phase label ({labelled.mean():.0%})")
    print("Share of days missing each signal:")
    print(daily.drop(columns=KEYS + ["phase", "flow"], errors="ignore").isna().mean().round(3).to_string())
    return daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/mcphases", help="folder holding the mcPHASES csv files")
    ap.add_argument("--out", default="data/daily.csv")
    ap.add_argument("--inspect", action="store_true", help="only print headers and 3 rows per file")
    ap.add_argument("--core-only", action="store_true", help="skip the optional files")
    a = ap.parse_args()
    if a.inspect:
        inspect(Path(a.raw))
    else:
        build_daily_table(Path(a.raw), include_optional=not a.core_only).to_csv(a.out, index=False)
        print(f"Saved {a.out}. Keep it on this laptop: data/ is blocked from git.")
