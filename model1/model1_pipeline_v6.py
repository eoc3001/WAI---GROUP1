# WAI Group 001 | Model 1, cycle version v6: per-cycle ovulation evidence + agent-layer rules
#
# One question per finished cycle: "Did she ovulate, and roughly which day?"
#   - Cycle length and regular/irregular come from her period log only (no model).
#   - Nightly watch signals (temperature, resting heart rate, breathing rate if present) answer
#     the ovulation question.
#   - Hormone tests are the answer key only, never an input. 
#   - The watch is compared with a guess from period dates alone (learned luteal length, and 14).
#
#

# %% 1. Settings
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

USE_SYNTHETIC = False                 # True = simulated people (safe anywhere). False = data/daily.csv, your laptop only
DATA_FILE = "data/daily.csv"          # from: python build_daily_table.py --raw data/mcphases --out data/daily.csv
RAW_DIR = Path("data/mcphases")       # only used if daily.csv has no lh/pdg columns
SYNTH_LAYOUT = "mcphases"             # simulated data copies the real layout ("full" = every source in every round)
OUT = Path(__file__).resolve().parent / ("results/synthetic" if USE_SYNTHETIC else "results")  # summary tables in model1/results, safe to commit
HANDOFF = Path("data/cycle_handoff_v6" + ("_synthetic" if USE_SYNTHETIC else "") + ".csv")  # per person, stays local

# Validation (team agreement 30 Sep: 80/20 by participant; five seeds so one lucky split can't decide)
SEEDS = [42, 43, 44, 45, 46]
TEST_SHARE = 0.2

# Period-start rule: chosen by a pre-declared rule (cell 3), not by test scores.
MIN_CYCLE_GAP = 15        # a new period needs >= 15 days since the last kept start (v4 used 22, which made
                          # cycles under 21 days, a Rotterdam sign, impossible to see) ...
MIN_DRY_DAYS = 7          # ... AND >= 7 days with no bleeding logged since her last bleeding day (so a period
                          # with a blank or "not at all" day in the middle stays one period) ...
SEEN_DRY_BEFORE = 7       # ... AND a logged "no period" day within the 7 days before it, so we know the
                          # period really started that day and wasn't just logged late.
OPENING_RUN = 3           # Exception: the first period of a study round (nothing logged before it) counts if
                          # it lasts >= 3 days, so a round that opens on a period keeps its first cycle.
                          # If the round opened mid-period, that first cycle may be 1-2 days short.
MIN_BLEED_DAYS = 1        # a bleed must last this many days in a row to count as a period start

# Period source for rounds with too little diary (team decision, see header point 2)
PERIOD_FALLBACK = "menstrual_label"   # "menstrual_label" or "none" (= those rounds get no cycles)
LABEL_MATCH_DAYS = 2      # a label start "matches" a diary start if within 2 days ...
LABEL_MATCH_MIN = 0.80    # ... and >= 80% must match, both ways (diary->label and label->diary) ...
LABEL_MATCH_MIN_STARTS = 10   # ... over >= 10 diary starts, or the label is not trusted
MIN_DIARY_COVERAGE = 0.5  # a study round uses its diary only if she filled it on >= this share of days
DIARY_GAP_MAX = 7         # a diary gap this long inside a cycle may hide a missed period: "uncertain"
NO_FLOW = {"", "not at all", "none", "no", "0", "nan", "spotting"}  # CHECK against your flow values

# Ovulation label (answer key: LH peak + 1 dates it, a sustained PdG rise confirms it)
EARLY_DAYS = (2, 8)       # cycle days used as her early-cycle (pre-ovulation) reference level ...
LUTEAL_MAX = 16           # ... cut short in short cycles: the window ends by day (length - 16), because
                          # in a 20-day cycle ovulation can come around day 4-6
PDG_FOLD = 2.0            # CHECK: confirmed = PdG >= 2x her early level ...
PDG_RUN = 3               # ... on 3 tests in a row (tests at most 3 days apart)
PDG_ABS = 5.0             # sensitivity check only: >= 5 ug/mL on 3 tests in a row (Leiva et al., BMJ Open 2019).
                          # CHECK that mcPHASES PdG is in ug/mL before reading anything into it.
PDG_MIN_TESTS = 5         # "anovulatory" needs this many tests from cycle day 8 on ...
MIDLUTEAL = (4, 10)       # ... AND tests 4-10 days before the next period (the clinic's mid-luteal check:
MIN_MIDLUTEAL_TESTS = 2   #     day 21-23 of a 28-day cycle, Carmina 2003; day 21, Boyle & Teede 2012)
LH_FOLD = 2.5             # CHECK: LH surge = >= 2.5x her early LH
LH_ABS = 25.0             # fixed LH surge rule when there is no early-cycle LH: >= 25 mIU/mL (Leiva et al. 2019).
                          # CHECK units: both fixed rules assume Mira units (PdG ug/mL, LH mIU/mL).
PDG_TO_OV = 3             # no LH surge: ovulation day = first high PdG day minus this
LH_ONLY_KEY = True        # cycles without usable PdG get the LH-only key (team decision, header point 1)
LH_WINDOW = (6, 9)        # LH-only key: a surge can fall from cycle day 6 to (length - 9)
LH_ONLY_COVERAGE = 0.80   # "not ovulated" by LH only if she tested on >= 80% of those days with no surge
MAX_CYCLE_LEN = 90        # the ovulation model skips longer cycles; the diary answers keep them
MIN_LUTEAL = 5            # a switch to "after ovulation" needs >= this many days before the next period
OV_MIN_DAY = 8            # ignore signal shifts in the first week of a cycle
CAL_FIXED_LUTEAL = 14     # the textbook calendar guess, kept as a second comparator

# Per-cycle model
CANDIDATES_ALL = ["night_temp", "rhr", "resp_rate", "sleep_hr"]   # used only if the column exists
MIN_NIGHTS = 10           # a cycle gets an answer only with >= this many nights with a signal
MIN_EARLY_NIGHTS = 3      # nights in cycle days 2-8 (her in-cycle reference level)
MIN_PER_CLASS = 3         # fit a logistic regression only with >= 3 cycles of each answer
TARGET_FP = 0.20          # "seen" threshold: anovulatory cycles called "seen" (false reassurance) <= 20%
TARGET_MISS = 0.10        # "not seen" threshold: ovulatory cycles called "not seen" (false worry) <= 10%
PROVISIONAL_P = 0.80      # in-cycle flag threshold for "after ovulation (estimated)"

# Agent layer (pre-declared before results; sources in cell 12b)
AXIS2_RUN = 2             # Axis 2 = the last 2 ANSWERED cycles both "not seen" ("unclear" or "not enough
                          # data" cycles are skipped: they neither count nor break the run), in the same
                          # study round, with at most one skipped cycle between them
AXIS2_MAX_MISS = 0.20     # Axis 2 is allowed only if the per-cycle miss rate is <= 20% ...
AXIS2_MAX_FALSE = 0.05    # ... and, against the hormone key, <= 5% of two-ovulatory-cycle runs escalate
CYCLE_LONG = 35           # Rotterdam: cycle intervals > 35 days
CYCLE_SHORT = 21          # Rotterdam: cycles < 21 days (short cycles can be anovulatory)
OLIGO_WINDOW, OLIGO_MIN = 3, 2   # "consistently" = 2 of her last 3 counted cycles outside 21-35 days
CYCLES_PER_YEAR_MIN = 8   # < 8 periods a year; with no full year of data: median length > 365/8 = 45.6 days
AMEN_DAYS = 90            # no period for 90+ days (3 months) with the diary filled in = see a doctor
AMEN_DIARY = 0.5          # ... "filled in" = diary entries on >= 50% of those days
MENARCHE_YEARS_MIN = 3    # adult criteria only from the third year after first period (Legro 2013; Boyle 2012)
                          # FIGO 24-38 days (cell 4) stays for the "regular" description sent to Model 2.
INCYCLE_SHOW = {"max_false_flag": 0.20, "min_within_3_days": 0.60}
                          # in-cycle "after ovulation (estimated)" is shown only if its false-flag
                          # rate is <= 20% and it fires 0-3 days after ovulation in >= 60% of
                          # ovulatory cycles (pre-declared gate, cell 11b)

# Decision gates (all written before looking at results)
GATE_AUC_LO = 0.60        # AUC 95% CI lower end must clear this
GATE_ECE = 0.10           # calibration error ceiling
GATE_COVERAGE = 0.70      # must give a definite answer (seen / not seen) for >= this share of labelled cycles
GATE_DAY = 0.60           # ovulation day shown only if within 2 days in >= this share of "seen" cycles

SEED = 42
B_BOOT = 1000
rng = np.random.default_rng(SEED)
NOTES = []                # warnings collected along the way, printed with the verdict


# %% 2. Load the daily table and KEEP EVERY DAY
# Days with no phase label stay, with phase blank, so a missed diary or test day never looks like
# a gap in the cycle. Wearable readings on unlabelled days are real signal.
def make_fake_people(n_people=42, seed=7, layout=SYNTH_LAYOUT):
    """Simulated people with realistic quirks: irregular people whose long cycles often skip
    ovulation, luteal phases of 10-17 days, a temperature rise 1-3 days after ovulation with
    night-to-night drift, Mira-style labels whose Luteal starts after the fertile window, sparse
    hormone tests, missed diary days and the odd week-long diary gap, and a second study round
    near day 905 for 20 people. layout="mcphases" copies the real split: round 1 has the diary and
    LH but no PdG; round 2 has LH and PdG but no diary (a few days log only the flow colour).
    Scores on these mean nothing; they only prove the code runs."""
    r = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_people + 1):
        irregular = pid % 6 == 0
        quiet = pid % 7 == 3          # regular lengths but often no ovulation (the phenotype Axis 2 is for)
        t0, hr0, br0 = r.normal(34.6, .4), r.normal(62, 6), r.normal(15, 1.2)
        rounds = [(1, 110)] + ([(905, 90)] if pid <= 20 else [])
        for start, n in rounds:
            end = start + n
            gap_from = start + int(r.integers(10, n - 10)) if r.random() < 0.25 else -99  # week-long diary gap
            day, drift = start, 0.0
            while day < end:
                L = int(r.uniform(35, 55)) if irregular else int(np.clip(r.normal(29, 3), 23, 38))
                if pid == 11:
                    L = int(r.integers(17, 22))       # short cycles (the Rotterdam < 21-day sign)
                if pid == 5 and day == start == 1:
                    L = 100                   # one long gap with the diary filled in (amenorrhea check)
                ovulates = r.random() > (0.4 if irregular else 0.55 if quiet else 0.06)
                luteal = int(r.choice(range(10, 18), p=[.04, .06, .12, .2, .24, .18, .1, .06]))
                ov, delay = L - luteal, int(r.integers(1, 4))
                for d in range(L):
                    if day >= end:
                        break
                    drift = 0.6 * drift + r.normal(0, .06)
                    after = ovulates and d >= ov + delay
                    if d < 5:
                        ph = "Menstrual"
                    elif not ovulates or d < ov - 5:
                        ph = "Follicular"
                    elif d <= ov + 1:
                        ph = "Fertility"
                    else:
                        ph = "Luteal"
                    flow = ("Light" if d in (0, 4) else "Medium") if d < 5 else "Not at all"
                    lh = r.lognormal(np.log(5), .3)
                    if ovulates and d == ov - 1:
                        lh = r.uniform(30, 60)
                    elif ovulates and d in (ov - 2, ov):
                        lh = r.uniform(12, 25)
                    pdg = r.uniform(5, 12) if ovulates and d >= ov + 2 else r.uniform(.5, 2.5)
                    tested = r.random() < 0.8
                    round2 = start > 1
                    no_diary = layout == "mcphases" and round2
                    no_pdg = layout == "mcphases" and not round2
                    rows.append({
                        "id": pid, "day_in_study": day,
                        "phase": ph if r.random() > 0.15 else np.nan,
                        "flow": (flow if (r.random() > 0.15 and not gap_from <= day < gap_from + 9 and not no_diary)
                                 else np.nan),
                        "flow_color": "Red" if (no_diary and d < 5 and r.random() < 0.05) else np.nan,
                        "lh": lh if tested else np.nan,
                        "pdg": pdg if (tested and not no_pdg) else np.nan,
                        "night_temp": t0 + 0.3 * after + drift + r.normal(0, .1) if r.random() > 0.15 else np.nan,
                        "rhr": hr0 + 1.5 * after + r.normal(0, 1.8) if r.random() > 0.15 else np.nan,
                        "resp_rate": br0 + 0.3 * after + r.normal(0, .35) if r.random() > 0.15 else np.nan,
                    })
                    day += 1
    return pd.DataFrame(rows)


raw = make_fake_people() if USE_SYNTHETIC else pd.read_csv(DATA_FILE)
if "phase" not in raw.columns:
    raw["phase"] = pd.NA     # only used as a cross-check; the answer key is LH/PdG
raw["phase"] = raw["phase"].astype("string").str.strip().str.title()
raw.loc[~raw["phase"].isin(["Menstrual", "Follicular", "Fertility", "Luteal"]), "phase"] = pd.NA
raw = raw.sort_values(["id", "day_in_study"]).reset_index(drop=True)

# Segments: a gap longer than 30 days starts a new study round; baselines never span the gap
gap = raw.groupby("id")["day_in_study"].diff()
raw["seg"] = raw["id"].astype(str) + "_r" + (gap > 30).groupby(raw["id"]).cumsum().astype(str)
print("SIMULATED DATA" if USE_SYNTHETIC else "REAL DATA (keep every output on this laptop)")
print(raw["id"].nunique(), "people,", len(raw), "days kept,", int(raw["phase"].isna().sum()),
      "days with no phase label (kept, phase blank)")


# %% 2b. Data map: which sources exist in which study round (counts only, no values)
# The real mcPHASES splits the evidence across rounds (header). This table shows it at a glance.
raw["round"] = raw["seg"].str.split("_r").str[-1].astype(int)
MAP_COLS = [c for c in ["flow", "flow_color", "phase", "lh", "pdg", "night_temp", "rhr", "resp_rate"] if c in raw.columns]
data_map = (raw.groupby("round")[MAP_COLS].agg(lambda s: s.notna().mean()).round(2)
            .assign(days=raw.groupby("round").size(), people=raw.groupby("round")["id"].nunique()))
print("Share of days with each source, by study round (round 0 = first round each person joined):")
print(data_map.to_string())


# %% 3. Period starts, decided round by round
# Source 1, always: the period diary (flow, or flow_color when only the colour was logged). A round
#   uses its diary only if she filled it in on >= 50% of days.
# Source 2, only if PERIOD_FALLBACK = "menstrual_label": for rounds with too little diary, the first
#   day of each run of Mira's "Menstrual" label. Trusted only if, in rounds that have both, label
#   starts and diary starts agree (cell 1 rules). Same start rules for both sources (cell 1).
# A round with neither gets NO period starts (its cycles are simply not counted), never wrong ones.
flow = raw["flow"].astype("string") if "flow" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")
if "flow_color" in raw.columns:
    flow = flow.fillna(raw["flow_color"].astype("string"))   # colour logged but not volume = still a log
flow_txt = flow.str.strip().str.lower()
diary_logged = flow.notna().astype(bool)
diary_bleed = (diary_logged & ~flow_txt.isin(NO_FLOW).fillna(False)
               & ~flow_txt.str.contains("spotting", na=False)).astype(bool)
if diary_logged.any():   # category names and counts only (CHECK that the split below is right)
    cats = flow_txt[diary_logged].value_counts()
    print("diary flow values ->", {k: f"{v} days, {'period' if diary_bleed[flow_txt == k].any() else 'no period'}"
                                   for k, v in cats.items()})
label_logged = raw["phase"].notna().astype(bool)
label_bleed = raw["phase"].eq("Menstrual").fillna(False).astype(bool)


def round_share(flag):
    return flag.groupby(raw["seg"]).transform("mean")


use_diary = round_share(diary_logged) >= MIN_DIARY_COVERAGE
label_ok = round_share(label_logged) >= MIN_DIARY_COVERAGE


def bleed_starts(d, bl, min_days):
    first = bl & ~bl.groupby(d["seg"]).shift(1, fill_value=False)
    run_len = bl.groupby(first.cumsum()).transform("sum")
    return first, first & (run_len >= min_days)


def clean_starts(d, flag, bl, dry_logged):
    """Keep a candidate period start only if it passes the rules in cell 1."""
    run_len = bl.groupby([d["seg"], (bl != bl.groupby(d["seg"]).shift()).cumsum()]).transform("size")
    kept = []
    for _, g in d.groupby("seg"):
        last_start = last_bleed = last_dry = None
        for i, day in g["day_in_study"].items():
            if flag[i]:
                ok_gap = last_start is None or day - last_start >= MIN_CYCLE_GAP
                ok_dry = last_bleed is None or day - last_bleed - 1 >= MIN_DRY_DAYS
                opening = last_start is None and last_dry is None and last_bleed is None
                seen_dry = ((last_dry is not None and day - last_dry <= SEEN_DRY_BEFORE)
                            or (opening and run_len[i] >= OPENING_RUN))
                if ok_gap and ok_dry and seen_dry:
                    kept.append(i)
                    last_start = day
            if bl[i]:
                last_bleed = day
            if dry_logged[i]:
                last_dry = day
    return pd.Series(d.index.isin(kept), index=d.index)


def starts_from(logged, bled, allowed):
    bl = (bled & allowed).astype(bool)
    dry = (logged & ~bled & allowed).astype(bool)          # she logged "no period" that day
    _, candidates = bleed_starts(raw, bl, MIN_BLEED_DAYS)
    return clean_starts(raw, candidates, bl, dry), bl


diary_starts, diary_bl = starts_from(diary_logged, diary_bleed, use_diary)
label_starts, label_bl = starts_from(label_logged, label_bleed, label_ok)

# Does the label put period starts where the diary does? Checked only in rounds that have both.
matched_d = total_d = matched_l = total_l = 0
offsets = []
for seg in raw.loc[use_diary & label_ok, "seg"].unique():
    in_seg = raw["seg"].eq(seg)
    d_days = raw.loc[diary_starts & in_seg, "day_in_study"].to_numpy()
    l_days = raw.loc[label_starts & in_seg, "day_in_study"].to_numpy()
    total_d, total_l = total_d + len(d_days), total_l + len(l_days)
    if len(d_days) and len(l_days):
        for x in d_days:
            gap_ = l_days - x
            j = np.argmin(np.abs(gap_))
            if abs(gap_[j]) <= LABEL_MATCH_DAYS:
                matched_d += 1
                offsets.append(int(gap_[j]))
        matched_l += int(sum(np.min(np.abs(d_days - y)) <= LABEL_MATCH_DAYS for y in l_days))
share_d = matched_d / total_d if total_d else np.nan
share_l = matched_l / total_l if total_l else np.nan
LABEL_OK = bool(total_d >= LABEL_MATCH_MIN_STARTS and share_d >= LABEL_MATCH_MIN and share_l >= LABEL_MATCH_MIN)
label_check = {"diary starts checked": total_d, "diary starts with a label start within 2 days": matched_d,
               "label starts checked": total_l, "label starts with a diary start within 2 days": matched_l,
               "median offset (label minus diary, days)": float(np.median(offsets)) if offsets else np.nan,
               "label trusted for rounds without a diary": LABEL_OK}
print("Mira 'Menstrual' label vs diary period starts, in rounds with both:", label_check)

use_label = (PERIOD_FALLBACK == "menstrual_label") & LABEL_OK & ~use_diary & label_ok
raw["period_source"] = np.select([use_diary, use_label], ["diary", "Mira period label"], "none")
raw["period_start"] = np.where(use_diary, diary_starts, np.where(use_label, label_starts, False)).astype(bool)
raw["bleed"] = np.where(use_diary, diary_bl, np.where(use_label, label_bl, False)).astype(bool)
raw["period_logged"] = np.where(use_diary, diary_logged, np.where(use_label, label_logged, False)).astype(bool)
raw["cycle_no"] = raw["period_start"].groupby(raw["seg"]).cumsum()
src_by_round = raw.groupby(["round", "period_source"])["seg"].nunique().unstack(fill_value=0)
print("Study rounds by period source (rows = round):")
print(src_by_round.to_string())
if use_label.any():
    NOTES.append(f"rounds without a diary took period starts from Mira's Menstrual label (matched the diary "
                 f"in {share_d:.0%} / {share_l:.0%} of starts): needs team sign-off, say so in the report")
elif (PERIOD_FALLBACK == "menstrual_label") and (~use_diary & label_ok).any():
    print("Rounds without a diary were NOT given label period starts: the label did not pass the match check.")


# %% 4. Cycle table: length and regularity, from logged periods only, no model
# These are outputs in their own right (they feed Model 2 as Cycle(R/I) and cycle length),
# so they must never come from a model guess. Long cycles are never dropped or filled in:
# they matter for PCOS risk.
def longest_gap(days, lo, hi):
    inside = np.sort(days[(days >= lo) & (days < hi)])
    edges = np.concatenate([[lo - 1], inside, [hi]])
    return int(np.diff(edges).max() - 1)


rows = []
for seg, g in raw.groupby("seg"):
    starts = np.sort(g.loc[g["period_start"], "day_in_study"].to_numpy())
    logged = g.loc[g["period_logged"], "day_in_study"].to_numpy()
    for k, (a, b) in enumerate(zip(starts[:-1], starts[1:]), start=1):
        lg = longest_gap(logged, a, b)
        rows.append({"seg": seg, "id": g["id"].iloc[0], "cycle_no": k, "start_day": a, "length": int(b - a),
                     "period_source": g["period_source"].iloc[0],
                     "longest_diary_gap": lg, "uncertain": lg >= DIARY_GAP_MAX,
                     "diary_share": round(float(((logged >= a) & (logged < b)).sum()) / max(b - a, 1), 3)})
cycles = pd.DataFrame(rows, columns=["seg", "id", "cycle_no", "start_day", "length", "period_source",
                                     "longest_diary_gap", "uncertain", "diary_share"])
if cycles.empty:
    raise SystemExit("No complete cycles: no study round had a period log good enough to find two period "
                     "starts. Check the flow column in build_daily_table.py, NO_FLOW, and the cell 3 printout.")
# counted = trusted for the diary answers (length, regularity, Rotterdam). Long cycles stay in.
# usable  = also short enough for the ovulation model (<= 90 days).
cycles["counted"] = ~cycles["uncertain"] & (cycles["length"] >= 15)
cycles["usable"] = cycles["counted"] & (cycles["length"] <= MAX_CYCLE_LEN)
print("cycles by period source:", cycles.groupby("period_source").size().to_dict())
print("complete cycles:", len(cycles), "| counted for diary answers:", int(cycles["counted"].sum()),
      "| usable for the ovulation model:", int(cycles["usable"].sum()),
      "| uncertain (diary gap) left out:", int(cycles["uncertain"].sum()))


def person_group(lengths):
    """FIGO 2018: regular = every cycle 24-38 days and variation <= 9 days."""
    lengths = lengths.dropna()
    if len(lengths) < 2:
        return "unknown (under 2 cycles)"
    if lengths.between(24, 38).all() and lengths.max() - lengths.min() <= 9:
        return "regular"
    return "irregular"


groups = cycles[cycles["counted"]].groupby("id")["length"].apply(person_group)
print("People per group:", groups.value_counts().to_dict())
known = groups[groups != "unknown (under 2 cycles)"]
if len(known) and (known == "irregular").mean() > 0.5:
    print("WARNING: most known people look irregular. Suspect the data before the biology - do NOT pass")
    print("these groups to Model 2 until the diary gaps are understood.")


# %% 4b. Axis 1 (period log only): the Rotterdam ovulatory-dysfunction pattern, and long gaps with no period
# Checked after every counted cycle, in time order, so the agent layer sees what she would see then.
#   oligo/poly pattern : 2 of her last 3 counted cycles in the same study round shorter than 21 or
#                        longer than 35 days ("consistently", so one 36-day cycle never flags)
#   infrequent         : fewer than 8 periods a year. mcPHASES rounds are ~3 months, so this is
#                        estimated from her median cycle length (> 45.6 days), with >= 3 cycles
#   no period 90+ days : diary filled in on >= 50% of those days, so it is not just missing logs.
#                        This is a "see a doctor" message on its own (take a pregnancy test first).
def axis1_history(cyc_table):
    out = []
    ok = cyc_table[cyc_table["counted"]].sort_values(["id", "seg", "cycle_no"])
    for pid, g in ok.groupby("id"):
        lens, segs = g["length"].to_numpy(), g["seg"].to_numpy()
        for k in range(len(lens)):
            last = lens[max(0, k - OLIGO_WINDOW + 1):k + 1][segs[max(0, k - OLIGO_WINDOW + 1):k + 1] == segs[k]]
            outside = int(((last < CYCLE_SHORT) | (last > CYCLE_LONG)).sum())
            upto = lens[:k + 1]
            infrequent = len(upto) >= 3 and 365 / np.median(upto) < CYCLES_PER_YEAR_MIN
            out.append({"id": pid, "seg": g["seg"].iloc[k], "cycle_no": g["cycle_no"].iloc[k],
                        "axis1_pattern": len(last) >= 2 and outside >= OLIGO_MIN, "axis1_infrequent": infrequent})
    return pd.DataFrame(out, columns=["id", "seg", "cycle_no", "axis1_pattern", "axis1_infrequent"])


def no_period_stretches(d):
    """Longest run of days with NO bleeding logged, per person, inside rounds with a period log.
    Counted only where the log was filled in on >= AMEN_DIARY of those days, so missing logs never
    look like a missing period. Uses every logged bleeding day, not just kept period starts."""
    best = {}
    for _, g in d.groupby("seg"):
        if g["period_source"].iloc[0] == "none":
            continue
        days = g["day_in_study"].to_numpy()
        logged = g.loc[g["period_logged"], "day_in_study"].to_numpy()
        bled = g.loc[g["bleed"], "day_in_study"].to_numpy()
        edges = np.concatenate([[days.min() - 1], np.sort(bled), [days.max() + 1]])
        for a, b in zip(edges[:-1], edges[1:]):
            dry = b - a - 1                                  # days strictly between two bleeding days
            if dry < AMEN_DAYS:
                continue
            share = ((logged > a) & (logged < b)).sum() / dry
            if share >= AMEN_DIARY:
                pid = g["id"].iloc[0]
                best[pid] = max(best.get(pid, 0), int(dry))
    return pd.Series(best, name="longest_no_period_days", dtype=float)


axis1 = axis1_history(cycles)
amen = no_period_stretches(raw)
print("Axis 1 (period log): people with the 21/35-day pattern at any point:",
      int(axis1.groupby("id")["axis1_pattern"].any().sum()),
      "| infrequent (< 8 a year):", int(axis1.groupby("id")["axis1_infrequent"].any().sum()),
      "| no period for 90+ days with the log filled in:", len(amen))


# %% 4c. Save the diary answers now (summaries to results/, per person to data/)
# These never depend on hormones, so they are saved before cell 6. If the hormone answer key turns
# out to be impossible on this data, these are the product (the kill criterion's diary-only app).
OUT.mkdir(parents=True, exist_ok=True)
HANDOFF.parent.mkdir(exist_ok=True)
(cycles[cycles["counted"]].groupby("period_source")["length"]
 .describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_csv(OUT / "cycle_length_summary.csv"))
groups.value_counts().rename("people").to_csv(OUT / "cycle_regularity.csv")
data_map.to_csv(OUT / "data_map_by_round.csv")
pd.Series(label_check).rename("value").to_csv(OUT / "period_label_check.csv")
pd.Series({"people with the 21/35-day pattern": int(axis1.groupby("id")["axis1_pattern"].any().sum()),
           "people with < 8 periods a year": int(axis1.groupby("id")["axis1_infrequent"].any().sum()),
           "people with no period for 90+ days": len(amen)}).rename("people").to_csv(OUT / "axis1_counts.csv")
_lens = cycles[cycles["counted"]].groupby("id")["length"]
_last1 = axis1.sort_values(["id", "seg", "cycle_no"]).groupby("id").last()
diary_handoff = pd.DataFrame({"cycles_counted": _lens.size(), "median_length": _lens.median(),
                              "shortest": _lens.min(), "longest": _lens.max()})
diary_handoff["regularity"] = groups.reindex(diary_handoff.index)
diary_handoff["Cycle(R/I)"] = diary_handoff["regularity"].map({"regular": 2, "irregular": 4})  # Kaggle coding, CHECK with Model 2
diary_handoff["axis1_pattern"] = _last1["axis1_pattern"].reindex(diary_handoff.index).fillna(False).astype(bool)
diary_handoff["axis1_infrequent"] = _last1["axis1_infrequent"].reindex(diary_handoff.index).fillna(False).astype(bool)
diary_handoff["no_period_90"] = diary_handoff.index.isin(amen.index)
diary_handoff.to_csv(HANDOFF)
print("diary answers saved:", OUT, "(summaries) and", HANDOFF, "(per person, local only)")


# %% 4d. Does Mira's label give the same regular / irregular answer as the diary?
# Only in study rounds that have both. Each source gets the same rules as cell 4: a cycle is the
# gap between two kept period starts, a long gap in that source's log makes the cycle uncertain,
# then FIGO groups. If they mostly agree, PERIOD_FALLBACK is safe for regularity too, not just
# for period start days (cell 3).
def counted_lengths(starts, logged, segs):
    rows = []
    for seg in segs:
        in_seg = raw["seg"].eq(seg)
        s = np.sort(raw.loc[starts & in_seg, "day_in_study"].to_numpy())
        log_days = raw.loc[logged & in_seg, "day_in_study"].to_numpy()
        for a, b in zip(s[:-1], s[1:]):
            if b - a >= 15 and longest_gap(log_days, a, b) < DIARY_GAP_MAX:
                rows.append({"id": raw.loc[in_seg, "id"].iloc[0], "length": int(b - a)})
    return pd.DataFrame(rows, columns=["id", "length"])


both_segs = raw.loc[use_diary & label_ok, "seg"].unique()
len_diary = counted_lengths(diary_starts, diary_logged, both_segs)
len_label = counted_lengths(label_starts, label_logged, both_segs)
pair = pd.DataFrame({"diary": len_diary.groupby("id")["length"].apply(person_group),
                     "label": len_label.groupby("id")["length"].apply(person_group)})
pair = pair.fillna("unknown (under 2 cycles)")
known_both = pair[(pair != "unknown (under 2 cycles)").all(axis=1)]
agree = int((known_both["diary"] == known_both["label"]).sum())
med_gap = (len_label.groupby("id")["length"].median() - len_diary.groupby("id")["length"].median()).abs().dropna()
reg_check = {"people in rounds with both": len(pair),
             "people with a known group from both": len(known_both),
             "same group (regular / irregular)": agree,
             "share agreeing": round(agree / len(known_both), 2) if len(known_both) else np.nan,
             "median gap in her median cycle length (days)": float(med_gap.median()) if len(med_gap) else np.nan}
reg_table = pd.crosstab(pair["diary"], pair["label"]).rename_axis(index="diary", columns="label")
print("Regularity, diary vs Mira label (same people, same rounds):", reg_check)
print(reg_table.to_string())
if len(known_both) < 10:
    print("NOTE: fewer than 10 people can be compared, so the share agreeing is only a rough guide.")
pd.Series(reg_check).rename("value").to_csv(OUT / "label_vs_diary_regularity.csv")
reg_table.to_csv(OUT / "label_vs_diary_regularity_table.csv")
NOTES.append(f"regularity from the label matched the diary for {agree} of {len(known_both)} people")


# %% 5. Nightly signals: deviation from her own past level (used by the signal screen in cell 7)
# Candidate signals and the pre-declared inclusion rule: a signal enters the model only if
# |effect vs the hormone label| >= 0.15 and it is missing on at most half the nights.
# Resting heart rate showed almost no cycle effect in the first daily run; the screen decides,
# not us. If only temperature passes, the model is temperature only: an honest answer, not a failure.
def add_dev(col):
    base = raw.groupby("seg")[col].transform(lambda x: x.shift(1).rolling(28, min_periods=7).median())
    raw[f"{col}_dev"] = raw[col] - base


for c in CANDIDATES_ALL:
    if c in raw.columns and raw[c].notna().mean() > 0:
        add_dev(c)
        print(c, "present on", round(raw[c].notna().mean(), 2), "of days")


# %% 6. The ovulation label: LH peak + 1 dates it, a sustained PdG rise confirms it
# Where the hormone values come from, in order:
#   1. lh / pdg columns in daily.csv (build_daily_table.py adds them; names matched ignoring case)
#   2. the raw mcPHASES hormone file, searched for in data/mcphases, data/raw and anywhere under data/
# Only column NAMES and counts are printed, never values. Hormone columns get the h_ prefix and
# cell 7 refuses to run if any model input carries it.
LH_NAMES = ["lh", "lh_value", "luteinizing_hormone"]                 # CHECK if your file uses another name
PDG_NAMES = ["pdg", "pdg_value", "pregnanediol", "progesterone"]     # CHECK (E3G/estrogen is not used)
HORMONE_SEARCH = list(dict.fromkeys([RAW_DIR, Path("data/mcphases"), Path("data/raw"), Path("data")]))


def find_col(cols, names):
    """Exact name first (ignoring case and spaces), then a whole-word match such as 'mira_lh'."""
    low = {str(c).strip().lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    for c in cols:
        words = str(c).strip().lower().replace("-", "_").replace(" ", "_").split("_")
        if any(n in words for n in names):
            return c
    return None


def hormones_from_raw_file():
    """Look for the mcPHASES hormone file and read only id, day, LH and PdG from it."""
    tried, seen = [], set()
    for folder in HORMONE_SEARCH:
        if not folder.exists():
            tried.append(f"{folder}/ (folder not found)")
            continue
        files = sorted(folder.glob("hormones_and_selfreport*.csv")) + sorted(folder.rglob("*hormone*.csv"))
        for f in files:
            if f.resolve() in seen:
                continue
            seen.add(f.resolve())
            cols = list(pd.read_csv(f, nrows=0).columns)
            id_c, day_c = find_col(cols, ["id"]), find_col(cols, ["day_in_study"])
            lh_c, pdg_c = find_col(cols, LH_NAMES), find_col(cols, PDG_NAMES)
            tried.append(f"{f}: id={id_c}, day_in_study={day_c}, LH={lh_c}, PdG={pdg_c}")
            if id_c and day_c and lh_c and pdg_c:
                print("hormones taken from", f, f"(LH column '{lh_c}', PdG column '{pdg_c}': CHECK these picks)")
                h = pd.read_csv(f, usecols=[id_c, day_c, lh_c, pdg_c])
                return h.rename(columns={id_c: "id", day_c: "day_in_study", lh_c: "h_lh", pdg_c: "h_pdg"}), tried
        if not files:
            tried.append(f"{folder}/ (no hormone csv inside)")
    return None, tried


lh_d, pdg_d = find_col(raw.columns, LH_NAMES), find_col(raw.columns, PDG_NAMES)
if lh_d and pdg_d and raw[lh_d].notna().any() and raw[pdg_d].notna().any():
    raw = raw.rename(columns={lh_d: "h_lh", pdg_d: "h_pdg"})
    print(f"hormones taken from the daily table (columns '{lh_d}' and '{pdg_d}')")
else:
    print(f"daily.csv has no usable LH/PdG columns (found LH: {lh_d}, PdG: {pdg_d}).",
          "It was probably built before lh and pdg were added to build_daily_table.py. Trying the raw file.")
    hormones, tried = (None, ["simulated data: raw files not searched"]) if USE_SYNTHETIC else hormones_from_raw_file()
    if hormones is None:
        print("Places checked:")
        for line in tried:
            print("  ", line)
        print("Columns in the daily table:", list(raw.columns))
        raise SystemExit(
            "No LH/PdG found, so the answer key does not exist. Fix it with ONE of these:\n"
            "  (a) rebuild the daily table with the current build_daily_table.py:\n"
            "      python build_daily_table.py --raw data/mcphases --out data/daily.csv\n"
            "      (it now adds lh and pdg; if it prints \"'lh' not found\", run it with --inspect and put\n"
            "      the real LH/PdG column names into FILES['labels'])\n"
            "  (b) or set RAW_DIR in cell 1 to the folder that holds hormones_and_selfreport.csv\n"
            "  (c) or, if the LH/PdG columns have other names, add them to LH_NAMES / PDG_NAMES above.\n"
            "Do not fall back to the phase label silently; if there is truly no LH/PdG, say so in the report.")
    hormones = hormones.dropna(subset=["id", "day_in_study"])
    for col in ["h_lh", "h_pdg"]:
        hormones[col] = pd.to_numeric(hormones[col], errors="coerce")
    hkey = hormones.groupby([hormones["id"].astype(str).str.strip(),
                             pd.to_numeric(hormones["day_in_study"], errors="coerce").round().astype("Int64")]
                            )[["h_lh", "h_pdg"]].mean()
    mkey = pd.MultiIndex.from_arrays([raw["id"].astype(str).str.strip().to_numpy(),
                                      pd.to_numeric(raw["day_in_study"], errors="coerce").round().astype("Int64").to_numpy()])
    raw["h_lh"] = hkey["h_lh"].reindex(mkey).to_numpy()
    raw["h_pdg"] = hkey["h_pdg"].reindex(mkey).to_numpy()
for col in ["h_lh", "h_pdg"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
share_lh, share_pdg = raw["h_lh"].notna().mean(), raw["h_pdg"].notna().mean()
print(f"days with an LH test: {share_lh:.1%} | days with a PdG test: {share_pdg:.1%}")
if share_lh == 0 and share_pdg == 0:
    raise SystemExit("The hormone file was found but no test lined up with a day in daily.csv. The id or "
                     "day_in_study values differ between the two files (for example 1 vs 'P01', or 1.0 vs 1). "
                     "Run build_daily_table.py again so both come from the same source.")

# cycle_day = day since the last kept period start. It MUST be built before the hormone floors
# below, because the floors use early-cycle days.
start_day = raw["day_in_study"].where(raw["period_start"]).groupby(raw["seg"]).ffill()
raw["cycle_day"] = raw["day_in_study"] - start_day
raw["cyc_key"] = raw["seg"].astype(str) + "|" + raw["cycle_no"].astype(int).astype(str)
CYC = {k: g.sort_values("day_in_study") for k, g in raw[raw["cycle_no"] >= 1].groupby("cyc_key")}
raw["cyc_len"] = raw["cyc_key"].map(cycles.assign(key=cycles["seg"] + "|" + cycles["cycle_no"].astype(str))
                                    .set_index("key")["length"])
early_end = np.minimum(EARLY_DAYS[1], raw["cyc_len"] - LUTEAL_MAX)          # NaN for unfinished cycles
EARLY_MASK = (raw["cycle_day"] >= EARLY_DAYS[0]) & (raw["cycle_day"] <= early_end)

# Where do the hormone tests fall? By study round, then by cycle day (counts only, no values)
in_cycle = raw["cycle_day"].notna()
round_tests = pd.DataFrame({
    "LH tests": raw["h_lh"].notna().groupby(raw["round"]).sum(),
    "PdG tests": raw["h_pdg"].notna().groupby(raw["round"]).sum(),
    "LH tests inside a cycle": (raw["h_lh"].notna() & in_cycle).groupby(raw["round"]).sum(),
    "PdG tests inside a cycle": (raw["h_pdg"].notna() & in_cycle).groupby(raw["round"]).sum()}).astype(int)
print("Hormone tests by study round ('inside a cycle' = after a period start from cell 3):")
print(round_tests.to_string())
band = pd.cut(raw["cycle_day"], [-1, 1, EARLY_DAYS[1], 12, 21, 10**4],
              labels=["day 0-1", f"day {EARLY_DAYS[0]}-{EARLY_DAYS[1]}", f"day {EARLY_DAYS[1] + 1}-12", "day 13-21", "day 22+"])
band = band.astype("string").fillna("outside any cycle")
test_map = pd.DataFrame({"PdG tests": raw["h_pdg"].notna().groupby(band).sum(),
                         "LH tests": raw["h_lh"].notna().groupby(band).sum()}).astype(int)
test_map = test_map.reindex(["day 0-1", f"day {EARLY_DAYS[0]}-{EARLY_DAYS[1]}", f"day {EARLY_DAYS[1] + 1}-12",
                             "day 13-21", "day 22+", "outside any cycle"]).fillna(0).astype(int)
print("Hormone tests by cycle day:")
print(test_map.to_string())
pdg_in_cycles = bool((raw["h_pdg"].notna() & in_cycle).any())
lh_in_cycles = bool((raw["h_lh"].notna() & in_cycle).any())
if not pdg_in_cycles and not (LH_ONLY_KEY and lh_in_cycles):
    raise SystemExit(
        "DATA LIMIT: no hormone test falls inside a cycle with known period dates, so the ovulation answer "
        "key cannot be built. The diary answers were saved in cell 4c and are the product (kill criterion). "
        "Ways forward for the team: a period source for the rounds with PdG (PERIOD_FALLBACK, or another "
        "column via build_daily_table.py), or the LH-only key (LH_ONLY_KEY = True).")
if not pdg_in_cycles:
    NOTES.append("no PdG test falls inside a cycle with period dates: every answer key is LH only (weaker)")


def low_reference(col):
    """A low 'not yet ovulated' hormone level used as a minimum baseline. Uses the early-cycle window;
    with too few early tests, the low end of all tests; with no tests at all, NaN (no minimum)."""
    v = raw.loc[EARLY_MASK, col].dropna()
    if len(v) >= 20:
        return float(v.quantile(0.25)), f"cycle days {EARLY_DAYS[0]}-{EARLY_DAYS[1]} ({len(v)} tests)"
    v = raw.loc[in_cycle, col].dropna()
    if len(v):
        return float(v.quantile(0.10)), f"lowest 10% of {len(v)} in-cycle tests (too few early-cycle tests)"
    return np.nan, "no tests inside cycles"


PDG_FLOOR, pdg_src = low_reference("h_pdg")
LH_FLOOR, lh_src = low_reference("h_lh")
# Her own early-cycle level from ALL her cycles (median of her day 2-8 tests, >= 3 tests). Not a
# percentile of all her tests: someone with short cycles who tests from day 9 has mostly
# after-ovulation tests, which would set her baseline too high and invent "no ovulation" cycles.
_early = raw[EARLY_MASK]
PERSON_LOW = {c: _early.groupby("id")[c].median().where(_early.groupby("id")[c].count() >= 3)
              for c in ["h_pdg", "h_lh"]}
print(f"PdG floor {PDG_FLOOR:.3g} from {pdg_src} | LH floor {LH_FLOOR:.3g} from {lh_src}")
if "too few" in pdg_src or "too few" in lh_src:
    NOTES.append("few early-cycle hormone tests: many cycles use the fixed 5 ug/mL / 25 mIU/mL rules; CHECK units")


def run_start(days, flags, run, max_gap=3):
    count, first, prev = 0, None, None
    for d, f in zip(days, flags):
        if f and count and d - prev <= max_gap:
            count += 1
        elif f:
            count, first = 1, d
        else:
            count = 0
        prev = d
        if count >= run:
            return first
    return None


LEVEL_SOURCE = {}   # how each cycle's baseline was set (counted and printed below)
LAST_SOURCE = {}    # the source used for the cycle being labelled right now


def early_level(t, x, floor, person_low=np.nan, col="", n=None):
    """Her level before ovulation in this cycle: median of her tests on days 2-8 of THIS cycle (>= 2;
    the window ends by day length - 16 in short cycles). If not, her day 2-8 level across her other
    cycles (>= 3 tests). If neither exists, NaN, and the caller uses the published fixed rule.
    Rejected shortcuts (each tested on simulated people): widening the window to day 12 or a
    percentile of all her tests (both catch after-ovulation tests and invent "no ovulation"
    cycles), and a group-wide floor (too low: ordinary ups and downs then pass as ovulation)."""
    hi = EARLY_DAYS[1] if n is None else min(EARLY_DAYS[1], n - LUTEAL_MAX)
    m = (t >= EARLY_DAYS[0]) & (t <= hi) & ~np.isnan(x)
    if m.sum() >= 2:
        src, level = "days 2-8", float(np.median(x[m]))
    elif not np.isnan(person_low):
        src, level = "her other cycles", float(person_low)
    else:
        src, level = "fixed rule", np.nan
    LEVEL_SOURCE[(col, src)] = LEVEL_SOURCE.get((col, src), 0) + 1
    LAST_SOURCE[col] = src
    return level if np.isnan(level) else float(np.fmax(level, floor))


def lh_threshold(t, lh, pid, n):
    base = early_level(t, lh, LH_FLOOR, PERSON_LOW["h_lh"].get(pid, np.nan), "LH", n)
    return LH_ABS if np.isnan(base) else LH_FOLD * base


def answer_key(g, n, absolute=False):
    """Answer key for one cycle of length n. Returns (label, ovulation day, dated by, key type).
    PdG+LH key (best): ovulatory = PdG high on 3 tests in a row (from day 8), dated by the LH peak + 1
      (or the PdG rise - 3 with no LH surge). Anovulatory = no such run AND >= 5 PdG tests from day 8
      AND >= 2 tests in the mid-luteal window, where a rise would show if she had ovulated.
    LH-only key (cycles with too few PdG tests, if LH_ONLY_KEY): ovulatory = an LH surge between day 6
      and length - 9, dated by the peak + 1; anovulatory = no surge although she tested on >= 80% of
      those days. Weaker: a short surge can fall between tests, and a surge does not prove the egg
      was released, so these labels are reported separately.
    absolute=True applies the fixed 5 ug/mL PdG rule to PdG cycles only (sensitivity check)."""
    t = g["cycle_day"].to_numpy(float)
    pdg, lh = g["h_pdg"].to_numpy(float), g["h_lh"].to_numpy(float)
    pid = g["id"].iloc[0]
    late = (t >= OV_MIN_DAY) & ~np.isnan(pdg)
    if late.sum() >= PDG_RUN:                                   # enough PdG to try the PdG key
        base = np.nan if absolute else early_level(t, pdg, PDG_FLOOR, PERSON_LOW["h_pdg"].get(pid, np.nan), "PdG", n)
        thr = PDG_ABS if np.isnan(base) else PDG_FOLD * base
        rise = run_start(t[late], pdg[late] >= thr, PDG_RUN)
        if rise is not None:
            if absolute:
                return "ovulatory", np.nan, "", "PdG+LH"
            win = (t >= 5) & (t <= rise) & ~np.isnan(lh)
            if win.any() and lh[win].max() >= lh_threshold(t, lh, pid, n):
                return "ovulatory", float(t[win][np.argmax(lh[win])] + 1), "LH peak + 1", "PdG+LH"
            return "ovulatory", float(rise - PDG_TO_OV), f"PdG rise - {PDG_TO_OV}", "PdG+LH"
        mid = late & (t >= n - MIDLUTEAL[1]) & (t <= n - MIDLUTEAL[0])
        if late.sum() >= PDG_MIN_TESTS and mid.sum() >= MIN_MIDLUTEAL_TESTS:
            return "anovulatory", np.nan, "", "PdG+LH"
    if absolute or not LH_ONLY_KEY:
        return "unclear", np.nan, "", ""
    lo, hi = LH_WINDOW[0], n - LH_WINDOW[1]
    win = (t >= lo) & (t <= hi) & ~np.isnan(lh)
    if hi < lo or not win.any():
        return "unclear", np.nan, "", ""
    if lh[win].max() >= lh_threshold(t, lh, pid, n):
        return "ovulatory", float(t[win][np.argmax(lh[win])] + 1), "LH peak + 1 (no PdG)", "LH only"
    if win.sum() / (hi - lo + 1) >= LH_ONLY_COVERAGE:
        return "anovulatory", np.nan, "", "LH only"
    return "unclear", np.nan, "", ""


def mira_luteal_start(g):
    """Cross-check only: the first cycle day Mira calls Luteal (after the first week)."""
    lut = g.loc[g["phase"].eq("Luteal").fillna(False) & (g["cycle_day"] >= OV_MIN_DAY), "cycle_day"]
    return float(lut.min()) if len(lut) else np.nan


cyc_rows = []
for c in cycles.to_dict("records"):
    g = CYC.get(f"{c['seg']}|{c['cycle_no']}")
    if g is None:
        continue
    LAST_SOURCE.clear()
    lab, ov_day, how, key_type = answer_key(g, c["length"])
    pdg_src_cycle = LAST_SOURCE.get("PdG", "")
    abs_lab = answer_key(g, c["length"], absolute=True)[0] if key_type == "PdG+LH" else ""
    cyc_rows.append({"key": f"{c['seg']}|{c['cycle_no']}", "seg": c["seg"], "id": c["id"], "cycle_no": c["cycle_no"],
                     "length": c["length"], "usable": c["usable"], "period_source": c["period_source"],
                     "group": groups.get(c["id"], "unknown (under 2 cycles)"),
                     "ov_label": lab, "ov_day": ov_day, "dated_by": how, "key_type": key_type or "none",
                     "mira_luteal": mira_luteal_start(g), "abs_label": abs_lab, "pdg_base": pdg_src_cycle})
cyc = pd.DataFrame(cyc_rows)
cyc["lab"] = cyc["ov_label"].map({"ovulatory": 1.0, "anovulatory": 0.0})   # unclear = NaN, not scored
u_all = cyc[cyc["usable"]]
print("Answer key for usable cycles (rows = key type, columns = answer):")
print(pd.crosstab(u_all["key_type"], u_all["ov_label"], margins=True).to_string())
n_anov = int((u_all["ov_label"] == "anovulatory").sum())
n_anov_pdg = int(((u_all["ov_label"] == "anovulatory") & (u_all["key_type"] == "PdG+LH")).sum())
print(f"Usable cycles with no ovulation: {n_anov} ({n_anov_pdg} confirmed by PdG). A few are expected in healthy adults.")
if n_anov < 15:
    print(f"WARNING: only {n_anov} anovulatory cycles - every false-alarm number below rests on them.")
    NOTES.append(f"only {n_anov} usable anovulatory cycles ({n_anov_pdg} by PdG): specificity CIs are wide")
print("Ovulation day dated by:", cyc.loc[cyc["ov_label"] == "ovulatory", "dated_by"].value_counts().to_dict())
print("Baseline level per cycle came from:", {f"{k[0]} {k[1]}": v for k, v in sorted(LEVEL_SOURCE.items())})
u = cyc[cyc["usable"] & (cyc["key_type"] == "PdG+LH") & (cyc["pdg_base"] != "fixed rule")]
n_fixed = int((cyc["usable"] & (cyc["key_type"] == "PdG+LH") & (cyc["pdg_base"] == "fixed rule")).sum())
if len(u):
    agree = (u["ov_label"] == u["abs_label"]).mean()
    print(f"Sensitivity check, fixed {PDG_ABS} ug/mL PdG rule vs her-own-level rule: same answer in {agree:.0%} "
          f"of {len(u)} usable PdG cycles (rows = main rule, columns = fixed rule):")
    print(pd.crosstab(u["ov_label"], u["abs_label"]).to_string())
print(f"{n_fixed} usable PdG cycles had no early-cycle tests and used the fixed rule directly.")
mg = (cyc["mira_luteal"] - cyc["ov_day"]).dropna()
if len(mg):
    print(f"Cross-check: Mira's Luteal starts a median {mg.median():.1f} days after the LH-dated ovulation "
          f"(middle half {mg.quantile(.25):.0f} to {mg.quantile(.75):.0f}, {len(mg)} cycles). "
          "A positive gap is expected: Mira's Luteal starts after the fertile window.")


# %% 7. Signal screen against the label (pre-declared rule) + one record per cycle
# Each signal is centred on her early-cycle level IN THAT CYCLE and scaled by her own night-to-night
# noise, so device bias and person-to-person differences cannot drive the answer. Signs are flipped
# so "after ovulation" is always higher.
# Honest caveat: the screen looks at every person's labels. It only picks which signals enter (a
# coarse yes/no with a fixed 0.15 bar, set in advance), so the optimism is small, but say so.
def person_noise(col):
    d = raw.sort_values(["seg", "day_in_study"])
    jump = d.groupby("seg")[col].diff().where(d.groupby("seg")["day_in_study"].diff().eq(1))
    noise = jump.abs().groupby(d["id"]).median() * 1.4826 / np.sqrt(2)
    return noise.clip(lower=noise.median() * 0.25)


CANDIDATES = [c for c in CANDIDATES_ALL if f"{c}_dev" in raw.columns]
assert CANDIDATES, ("No nightly signal columns found in the daily table - expected at least "
                    "'night_temp' (from computed_temperature.csv via build_daily_table).")

key_info = cyc.set_index("key")
pre_post = (raw["cyc_key"].map(key_info["usable"]).eq(True) & raw["cycle_day"].notna()
            & raw["cyc_key"].map(key_info["lab"]).notna())          # label known: ovulatory or anovulatory
ov_map = raw["cyc_key"].map(key_info["ov_day"])
after = pre_post & (raw["cycle_day"] >= ov_map + 1)
before = pre_post & raw["cycle_day"].between(EARLY_DAYS[0], np.where(ov_map.isna(), 10**9, ov_map))
labelled_phase = raw["phase"].notna()
print("screen data check - labelled usable-cycle days:", int(pre_post.sum()),
      "| after ovulation:", int(after.sum()), "| before ovulation:", int(before.sum()),
      "| days with a phase label:", int(labelled_phase.sum()))
if pre_post.sum() < 100:
    print("WARNING: very few usable-cycle days. Usability is decided by the diary period starts")
    print("(cells 3-4) - read the grouping check output before trusting anything below.")

sig_rows = {}
for col in CANDIDATES:
    dev = raw[f"{col}_dev"]
    e = (dev[after].mean() - dev[before].mean()) / dev.std() if after.any() and before.any() else np.nan
    # Second opinion: the Mira phase label, independent of the hormone label. If this effect is
    # clear while the hormone effect is not, the problem is label sparsity, not the sensor.
    e_mira = np.nan
    m = labelled_phase & dev.notna()
    if m.sum() and {"Luteal", "Follicular"} <= set(raw.loc[m, "phase"].unique()):
        by_phase = dev[m].groupby(raw.loc[m, "phase"]).mean()
        e_mira = (by_phase.get("Luteal", np.nan) - by_phase.get("Follicular", np.nan)) / dev[m].std()
    sig_rows[col] = {"effect vs hormones": round(e, 2) if pd.notna(e) else np.nan,
                     "effect vs Mira label": round(e_mira, 2) if pd.notna(e_mira) else np.nan,
                     "share missing": round(raw[col].isna().mean(), 2)}
sig_check = pd.DataFrame(sig_rows).T
print(sig_check.to_string())


def passes(e):
    return pd.notna(e) and abs(e) >= 0.15


SIGNALS = [c for c in CANDIDATES if passes(sig_rows[c]["effect vs hormones"])
           and sig_rows[c]["share missing"] <= 0.5]
# Pre-declared temperature fallback: if night_temp has good coverage but failed ONLY the hormone
# screen while the independent Mira-label effect is >= 0.15, the failure is almost certainly
# label sparsity, not a dead sensor. Keep it and say so. Temperature only - no other signal.
if "night_temp" not in SIGNALS and "night_temp" in CANDIDATES:
    if sig_rows["night_temp"]["share missing"] <= 0.5 and passes(sig_rows["night_temp"]["effect vs Mira label"]):
        SIGNALS.insert(0, "night_temp")
        NOTES.append("night_temp failed the hormone screen but passed the Mira-label screen (kept)")
        print("WARNING: night_temp kept on the Mira-label screen; the hormone label is too sparse here.")

SIGNS = {}
for c in SIGNALS:
    e = sig_rows[c]["effect vs hormones"]
    SIGNS[c] = float(np.sign(e if passes(e) else sig_rows[c]["effect vs Mira label"]))
SIG_TEMP = "night_temp" if "night_temp" in SIGNALS else None
print("Signals entering the model (pre-declared rule):", SIGNALS, "| signs:", SIGNS)
if SIG_TEMP is None:
    print("DIAGNOSTIC DUMP - why the screen failed:")
    print("candidates found:", CANDIDATES,
          "| usable days:", int(pre_post.sum()),
          "| usable cycles:", int(cyc["usable"].sum()),
          "| ovulatory cycles:", int((cyc["ov_label"] == "ovulatory").sum()),
          "| anovulatory:", int((cyc["ov_label"] == "anovulatory").sum()))
assert SIG_TEMP is not None, ("No temperature signal passed the screen. High share missing = wrong "
    "temperature column in the daily table; effects near 0 = nights misaligned with cycle days; "
    "NaN or tiny counts = the usable-cycle or hormone-label side is empty (cells 4 and 6).")
assert not any(c.startswith("h_") or c in {"lh", "pdg", "phase"} for c in SIGNALS), "hormone or label column leaked into inputs"

NOISE = {s: person_noise(s) for s in SIGNALS}
CY = []
for c in cyc[cyc["usable"]].to_dict("records"):    # one plain dict per usable cycle
    g = CYC[c["key"]]
    t = g["cycle_day"].to_numpy(int)
    Z = np.full((len(g), len(SIGNALS)), np.nan)
    for j, s in enumerate(SIGNALS):
        x = g[s].to_numpy(float)
        early = (t >= EARLY_DAYS[0]) & (t <= EARLY_DAYS[1]) & ~np.isnan(x)
        if early.sum() >= MIN_EARLY_NIGHTS:
            Z[:, j] = SIGNS[s] * (x - np.median(x[early])) / NOISE[s].get(c["id"], NOISE[s].median())
    has = ~np.isnan(Z).all(axis=1)
    temp_ok = not np.isnan(Z[:, SIGNALS.index(SIG_TEMP)]).all()
    CY.append({"key": c["key"], "id": c["id"], "seg": c["seg"], "cycle_no": int(c["cycle_no"]), "n": int(c["length"]),
               "t": t[has], "Z": Z[has], "lab": c["lab"], "ov_day": c["ov_day"], "group": c["group"],
               "key_type": c["key_type"],
               "enough": temp_ok and has.sum() >= MIN_NIGHTS})
print("cycles with a record:", len(CY), "| with enough nights:", sum(c["enough"] for c in CY),
      "| labelled anovulatory among them:", sum(c["enough"] and c["lab"] == 0 for c in CY))


# %% 8. The two models, and why both exist
# (1) Step-fit baseline: for each possible switch day fit "flat, then a higher flat" to the nightly
#     signal and compare with "flat all cycle" (BIC). Then a 2-number logistic regression on
#     (fit gain, step size). If the fancy model cannot beat THIS, we ship this.
# (2) Two-state left-to-right HMM: state 0 = before ovulation, state 1 = after; the state switches
#     once or never (never = anovulatory). Each signal has its own level and spread per state, learned
#     from training people only. With one possible switch every switch day can be summed over exactly,
#     and a luteal-length prior comes from the training people. Night-to-night correlation r is
#     removed by down-weighting evidence by (1-r)/(1+r).
_STEP_CACHE = {}   # step features depend only on the cycle, never on training, so compute once


class StepModel:
    def feats(self, c):
        if c["key"] in _STEP_CACHE:
            return _STEP_CACHE[c["key"]]
        out = None
        y = np.nanmean(c["Z"], axis=1)
        if c["enough"] and len(y) >= MIN_NIGHTS:
            rss0 = ((y - y.mean()) ** 2).sum()
            best = None
            for k in range(OV_MIN_DAY, c["n"] - MIN_LUTEAL + 1):
                a, b = y[c["t"] < k], y[c["t"] >= k]
                if len(a) < 3 or len(b) < 3:
                    continue
                rss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
                if best is None or rss < best[0]:
                    best = (rss, k, b.mean() - a.mean())
            if best is not None:
                m = len(y)
                gain = m * np.log(max(rss0, 1e-9) / max(best[0], 1e-9)) - 2 * np.log(m)
                out = (np.array([gain * np.sign(best[2]), best[2]]), best[1])
        _STEP_CACHE[c["key"]] = out
        return out

    def fit(self, train):
        X, y = [], []
        for c in train:
            f = self.feats(c)
            if f is not None and not np.isnan(c["lab"]):
                X.append(f[0])
                y.append(c["lab"])
        y = np.array(y)
        self.lr = None
        if len(y) and min((y == 0).sum(), (y == 1).sum()) >= MIN_PER_CLASS:
            self.lr = LogisticRegression(max_iter=1000).fit(np.array(X), y)
        return self

    def score(self, c):
        f = self.feats(c)
        if f is None:
            return None
        # too few cycles of one answer to fit LR: fall back to the fit gain itself (Platt rescales it)
        s = self.lr.decision_function(f[0].reshape(1, -1))[0] if self.lr is not None else f[0][0]
        return float(s), f[1], f[1], f[1]


def _col_stat(a, fn, fallback):
    """Column mean/std that never crashes on empty or all-missing columns."""
    if not len(a):
        return np.array(fallback, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = fn(a, axis=0)
    return np.where(np.isnan(v), fallback, v)


class HMMModel:
    def fit(self, train):
        use = [c for c in train if c["enough"] and not np.isnan(c["lab"])]
        k = len(SIGNALS)
        pre, post, L, kk = [], [], [], []
        for c in use:
            st = (c["t"] >= c["ov_day"] + 1) if c["lab"] == 1 else np.zeros(len(c["t"]), bool)
            pre.append(c["Z"][~st])
            post.append(c["Z"][st])
            if c["lab"] == 1:
                L.append(c["n"] - (c["ov_day"] + 1))
                kk.append(c["ov_day"] + 1)
        pre = np.vstack(pre) if pre else np.empty((0, k))
        post = np.vstack(post) if post else np.empty((0, k))
        self.mu0 = _col_stat(pre, np.nanmean, np.zeros(k))
        self.mu1 = _col_stat(post, np.nanmean, self.mu0 + 1.0)     # no ovulatory training cycles: assume a 1-noise-unit rise
        self.sd0 = np.clip(_col_stat(pre, np.nanstd, np.ones(k)), 0.3, None)
        self.sd1 = np.clip(_col_stat(post, np.nanstd, np.ones(k)), 0.3, None)
        self.Lmu, self.Lsd = (float(np.mean(L)), max(float(np.std(L)), 1.5)) if L else (13.0, 2.0)
        # prior for WHEN the switch happens (first after-ovulation day), learned here from training
        # people only. The uploaded v4 read it from every labelled cycle, including the test person.
        self.k_mu, self.k_sd = (float(np.mean(kk)), max(float(np.std(kk)), 3.0)) if kk else (16.0, 4.0)
        pi = float(np.mean([c["lab"] for c in use])) if use else 0.9
        self.pi = float(np.clip(pi, 0.02, 0.98))   # all-ovulatory training set would give log(0)
        resid = []
        for c in use:
            st = (c["t"] >= c["ov_day"] + 1) if c["lab"] == 1 else np.zeros(len(c["t"]), bool)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = np.nanmean(c["Z"] - np.where(st[:, None], self.mu1, self.mu0), axis=1)
            nxt = np.diff(c["t"]) == 1
            resid.append(np.column_stack([r[:-1][nxt], r[1:][nxt]]))
        rr = np.vstack(resid) if resid else np.empty((0, 2))
        rr = rr[~np.isnan(rr).any(axis=1)]
        rho = float(np.clip(np.asarray(np.corrcoef(rr.T))[0, 1], 0, 0.9)) if len(rr) > 10 else 0.0
        self.tau = (1 - rho) / (1 + rho)
        return self

    def loglik(self, c):
        l0 = np.nansum(norm.logpdf(c["Z"], self.mu0, self.sd0), axis=1) * self.tau
        l1 = np.nansum(norm.logpdf(c["Z"], self.mu1, self.sd1), axis=1) * self.tau
        return l0, l1

    def score(self, c):
        if not c["enough"]:
            return None
        ks = np.arange(OV_MIN_DAY, c["n"] - MIN_LUTEAL + 1)
        if not len(ks):
            return None
        l0, l1 = self.loglik(c)
        c0, c1 = np.concatenate([[0], np.cumsum(l0)]), np.concatenate([[0], np.cumsum(l1)])
        idx = np.searchsorted(c["t"], ks)
        prior = norm.logpdf(c["n"] - ks, self.Lmu, self.Lsd)
        prior -= logsumexp(prior)
        ll_k = c0[idx] + (c1[-1] - c1[idx]) + prior
        log_odds = (logsumexp(ll_k) + np.log(self.pi)) - (c0[-1] + np.log(1 - self.pi))
        post = np.exp(ll_k - logsumexp(ll_k))
        cdf = np.cumsum(post)
        return (float(log_odds), int(ks[np.argmax(post)]), int(ks[min(np.searchsorted(cdf, 0.1), len(ks) - 1)]),
                int(ks[min(np.searchsorted(cdf, 0.9), len(ks) - 1)]))

    def p_switched_by(self, c, t_end):
        """In-cycle view: chance the switch already happened, using nights up to t_end only."""
        m = c["t"] <= t_end
        if m.sum() < MIN_NIGHTS // 2:
            return np.nan
        cc = {**c, "t": c["t"][m], "Z": c["Z"][m]}
        l0, l1 = self.loglik(cc)
        c0, c1 = np.concatenate([[0], np.cumsum(l0)]), np.concatenate([[0], np.cumsum(l1)])
        all_k = np.arange(OV_MIN_DAY, MAX_CYCLE_LEN)
        pk = norm.pdf(all_k, self.k_mu, self.k_sd)
        pk /= pk.sum()
        ks = all_k[all_k <= t_end - 1]
        if not len(ks):
            return 0.0
        idx = np.searchsorted(cc["t"], ks)
        ll_k = c0[idx] + (c1[-1] - c1[idx]) + np.log(self.pi * pk[:len(ks)] + 1e-300)
        p_not_yet = self.pi * pk[len(ks):].sum() + (1 - self.pi)
        top = logsumexp(ll_k)
        return float(np.exp(top - np.logaddexp(top, c0[-1] + np.log(p_not_yet))))


MODELS = {"Step-fit": StepModel, "HMM": HMMModel}


# %% 9. Validation: 80/20 by participant x 5 seeds (headline) and leave-one-person-out (robustness)
# Same routine for both: fit on training people only. Inside the training people, grouped CV gives
# out-of-fold scores to recalibrate the probability (Platt) and to pick the threshold with false
# alarms at or under TARGET_FP. The calendar comparator's luteal length is learned the same way.
_STEP_CACHE.clear()   # cell 7 may have been rerun with different signals since cell 8 ran
labelled = [c for c in CY if not np.isnan(c["lab"])]
classes = {float(c["lab"]) for c in labelled}
print("labelled cycles:", len(labelled), "| ovulatory:", sum(c["lab"] == 1 for c in labelled),
      "| anovulatory:", sum(c["lab"] == 0 for c in labelled))
if classes != {0.0, 1.0}:
    raise SystemExit(
        "DATA LIMIT: the answer key has only one kind of cycle (" + ", ".join(
            {0.0: "not ovulated", 1.0: "ovulated"}[k] for k in sorted(classes)) + "). A yes/no ovulation "
        "model cannot be fitted or honestly scored on one class. The diary answers saved in cell 4c are the "
        "product (kill criterion). Report the answer-key table from cell 6.")


def model_scores(model, cyc_list):
    out = np.full(len(cyc_list), np.nan)
    for i, c in enumerate(cyc_list):
        s = model.score(c)
        if s is not None:
            out[i] = s[0]
    return out


class Calibrator:
    """Platt scaling. With too few cycles of one answer it falls back to the model's own odds."""
    def fit(self, s, y):
        ok = ~np.isnan(s)
        s, y = s[ok], y[ok]
        self.lr = None
        if len(y) and min((y == 0).sum(), (y == 1).sum()) >= 2:
            self.lr = LogisticRegression(max_iter=1000).fit(s.reshape(-1, 1), y)
        return self

    def predict(self, s):
        s = np.asarray(s, float).reshape(-1, 1)
        return self.lr.predict_proba(s)[:, 1] if self.lr is not None else expit(s[:, 0])


def pick_threshold(p, y, max_fp):
    neg = p[y == 0]
    if not len(neg):
        return 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        if (neg >= t).mean() <= max_fp:
            return round(float(t), 2)
    return 0.95


def pick_low_threshold(p, y, max_miss):
    """Highest threshold that still keeps ovulatory cycles called "not seen" at or under max_miss."""
    pos = p[y == 1]
    if not len(pos):
        return 0.0
    for t in np.arange(0.95, 0.04, -0.01):
        if (pos < t).mean() <= max_miss:
            return round(float(t), 2)
    return 0.05


def make_call(prob, t_high, t_low):
    if np.isnan(prob):
        return "not enough data", np.nan
    if prob >= t_high:
        return "seen", 1.0
    if prob < t_low:
        return "not seen", 0.0
    return "unclear", np.nan


def run_fold(train, test):
    """Fit both models on `train` (labelled cycles of training people) and answer every cycle in
    `test` (all cycles of the test people, labelled or not). Returns rows, the fitted HMM, and
    the learned luteal length used by the calendar comparator."""
    y_tr = np.array([c["lab"] for c in train])
    g_tr = np.array([c["id"] for c in train])
    n_groups = len(set(g_tr))
    lut = [c["n"] - c["ov_day"] for c in train if c["lab"] == 1]
    l_train = float(np.median(lut)) if lut else float(CAL_FIXED_LUTEAL)
    rows, hmm = [], None
    for name, make in MODELS.items():
        oof = np.full(len(train), np.nan)
        if n_groups >= 2:
            for a, b in GroupKFold(n_splits=min(5, n_groups)).split(np.zeros(len(train)), groups=g_tr):
                oof[b] = model_scores(make().fit([train[i] for i in a]), [train[i] for i in b])
        cal = Calibrator().fit(oof, y_tr)
        okk = ~np.isnan(oof)
        p_oof = cal.predict(oof[okk]) if okk.any() else np.array([])
        thr = pick_threshold(p_oof, y_tr[okk], TARGET_FP) if okk.any() else 0.5            # "seen" at or above
        thr_low = min(pick_low_threshold(p_oof, y_tr[okk], TARGET_MISS), thr) if okk.any() else 0.0  # "not seen" below
        final = make().fit(train)
        lags = [s[1] - c["ov_day"] for c in train if c["lab"] == 1 for s in [final.score(c)] if s is not None]
        offset = float(np.median(lags)) if lags else 1.0       # typical delay from ovulation to the signal step
        if name == "HMM":
            hmm = final
        for c in test:
            s = final.score(c)
            row = {"model": name, "key": c["key"], "id": c["id"], "seg": c["seg"], "cycle_no": c["cycle_no"],
                   "lab": c["lab"], "ov_day": c["ov_day"], "n": c["n"], "group": c["group"], "key_type": c["key_type"],
                   "enough": c["enough"], "thr": thr, "thr_low": thr_low, "luteal_learned": l_train,
                   "ov_hat_cal": c["n"] - l_train, "ov_hat_cal14": c["n"] - CAL_FIXED_LUTEAL}
            prob = np.nan if s is None else float(cal.predict([s[0]])[0])
            call, seen = make_call(prob, thr, thr_low)
            row.update(prob=prob, call=call, seen=seen, ov_hat=np.nan if s is None else s[1] - offset)
            rows.append(row)
    return rows, hmm, l_train


people = np.array(sorted({c["id"] for c in CY}))

# Headline: 80/20 by participant, five seeds
split_rows = []
for seed in SEEDS:
    tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=seed)
                      .split(people, groups=people))
    test_ids = set(people[te_i])
    rows_s, _, l_tr = run_fold([c for c in labelled if c["id"] not in test_ids],
                               [c for c in CY if c["id"] in test_ids])
    for r_ in rows_s:
        r_["split"] = seed
    split_rows += rows_s
    print(f"split {seed}: {len(people) - len(test_ids)} training people, {len(test_ids)} test people, "
          f"learned luteal length {l_tr:.1f} days")
preds_8020 = pd.DataFrame(split_rows)

# Robustness: leave one person out (every cycle is answered by a model that never saw its person)
lopo_rows, FITTED = [], {}
for pid in people:
    rows_p, FITTED[pid], _ = run_fold([c for c in labelled if c["id"] != pid], [c for c in CY if c["id"] == pid])
    lopo_rows += rows_p
preds = pd.DataFrame(lopo_rows)
print("done:", len(SEEDS), "80/20 splits and", len(people), "leave-one-person-out fits")


# %% 10. Scores: TP/TN/FP/FN, precision, recall, F1, specificity, AUC, calibration, day error
def safe_auc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 and len(np.unique(p)) > 1 else np.nan


def ece(y, p, bins=5, min_n=5):
    if not len(p):
        return np.nan
    b = np.minimum((p * bins).astype(int), bins - 1)
    return round(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean()
                     for i in range(bins) if (b == i).sum() >= min_n), 3)


def counts(y, s):
    tp, fp = int((y & s).sum()), int((~y & s).sum())
    fn, tn = int((y & ~s).sum()), int((~y & ~s).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "false alarm": fp / (fp + tn) if fp + tn else np.nan}


def day_errors(d, col):
    e = (d[col] - d["ov_day"]).abs().dropna()
    return (e.median() if len(e) else np.nan), (e.quantile(0.9) if len(e) else np.nan), \
        ((e <= 2).mean() if len(e) else np.nan)


def summarise(d):
    """One model's answers (one split, or all LOPO answers pooled). 'Ovulated' is the positive class."""
    lab = d[d["lab"].notna()]
    scored_ = lab[lab["prob"].notna()]
    have = scored_[scored_["seen"].notna()]                # a definite answer: "seen" or "not seen"
    y, s = have["lab"].to_numpy() == 1, have["seen"].to_numpy() == 1
    ys, ps = scored_["lab"].to_numpy() == 1, scored_["prob"].to_numpy(float)
    tpos = have[y & s]                                     # day error only where she ovulated and we said so
    med, p90, within2 = day_errors(tpos, "ov_hat")
    out = {"cycles labelled": len(lab), "anovulatory": int((lab["lab"] == 0).sum()),
           "unclear": int((lab["call"] == "unclear").sum()),
           "coverage": len(have) / max(len(lab), 1), **counts(y, s),
           "miss rate": float((scored_.loc[ys, "call"] == "not seen").mean()) if ys.any() else np.nan,
           "AUC": safe_auc(ys, ps), "ECE": ece(ys, ps),
           "day err model": med, "day err worst 10%": p90, "day within 2": within2,
           "day err calendar (learned)": day_errors(tpos, "ov_hat_cal")[0],     # same cycles, fair comparison
           "day err calendar 14": day_errors(tpos, "ov_hat_cal14")[0]}
    return out, scored_


def calendar_summary(d):
    """Period dates alone: always says 'ovulated', dates it at length minus the luteal length."""
    lab = d.drop_duplicates("key")
    lab = lab[lab["lab"].notna()]
    y = lab["lab"].to_numpy() == 1
    ov = lab[y]
    med, p90, within2 = day_errors(ov, "ov_hat_cal")
    return {"cycles labelled": len(lab), "anovulatory": int((~y).sum()), "unclear": 0, "coverage": 1.0,
            **counts(y, np.ones(len(lab), bool)), "miss rate": 0.0, "AUC": np.nan, "ECE": np.nan,
            "day err model": med, "day err worst 10%": p90, "day within 2": within2,
            "day err calendar (learned)": med, "day err calendar 14": day_errors(ov, "ov_hat_cal14")[0]}


CAL_NAME = "Calendar (period dates only)"

# ---- Headline: 80/20 x 5 ----
per_split = []
for seed, ds in preds_8020.groupby("split"):
    for name, d in ds.groupby("model"):
        per_split.append({"split": seed, "model": name, **summarise(d)[0]})
    per_split.append({"split": seed, "model": CAL_NAME, **calendar_summary(ds)})
per_split = pd.DataFrame(per_split)


def mean_range(x):
    x = pd.Series(x, dtype=float).dropna()
    return f"{x.mean():.2f} [{x.min():.2f} to {x.max():.2f}]" if len(x) else "n/a"


RATE_COLS = ["precision", "recall", "F1", "specificity", "miss rate", "AUC", "coverage",
             "day err model", "day err calendar (learned)", "day err calendar 14"]
headline = per_split.groupby("model")[["TP", "TN", "FP", "FN", "unclear", "anovulatory"]].sum()
for col in RATE_COLS:
    headline[col] = per_split.groupby("model")[col].apply(mean_range)
print("HEADLINE: 80/20 by participant, 5 seeds. Counts are summed over the 5 test sets;")
print("rates are the mean [lowest to highest] across splits. Positive = 'ovulated'. TP/TN/FP/FN count")
print("definite answers only; 'unclear' cycles are listed separately and lower the coverage.")
print(headline.to_string())
print("Calendar always says 'ovulated', so its specificity is 0 by design: the watch is only useful")
print("if it catches no-ovulation cycles (specificity) AND dates ovulation at least as well.")
if per_split.groupby("split")["anovulatory"].first().min() < 3:
    print("NOTE: some test sets hold under 3 anovulatory cycles, so specificity per split is very noisy.")

# ---- Robustness: leave one person out, pooled, with bootstrap CIs over people ----
checks, haves = {}, {}
for name, d in preds.groupby("model"):
    checks[name], haves[name] = summarise(d)
checks[CAL_NAME] = calendar_summary(preds)
checks_df = pd.DataFrame(checks).T
print("\nROBUSTNESS: leave one person out (every cycle answered once)")
print(checks_df.round(3).to_string())
by_key = {f"{name} | {kt}": summarise(d[d["key_type"] == kt])[0]
          for name, d in preds.groupby("model") for kt in ["PdG+LH", "LH only"]
          if (d["key_type"] == kt).any() and d.loc[d["key_type"] == kt, "lab"].notna().any()}
if by_key:
    by_key_df = pd.DataFrame(by_key).T
    print("Same scores split by answer-key type (PdG+LH is the stronger key; if the two disagree, trust PdG+LH):")
    print(by_key_df[["cycles labelled", "anovulatory", "precision", "recall", "specificity", "miss rate", "AUC",
                     "day err model"]].round(3).to_string())
else:
    by_key_df = pd.DataFrame()

scored = preds[preds["lab"].notna() & preds["prob"].notna()]
pid_list = scored["id"].unique()
draws = [rng.choice(pid_list, len(pid_list)) for _ in range(B_BOOT)]
BY = {m: {pid: (g["lab"].to_numpy() == 1, g["seen"].to_numpy(float), g["prob"].to_numpy(float), g["key"].to_numpy())
          for pid, g in d.groupby("id")} for m, d in scored.groupby("model")}


def pooled(m, draw):
    parts = [BY[m][p] for p in draw if p in BY[m]]
    if not parts:
        return None
    return tuple(np.concatenate([x[i] for x in parts]) for i in range(4))


def stat_on(m, draw, what):
    a = pooled(m, draw)
    if a is None:
        return np.nan
    y, s, p, _ = a
    dec = ~np.isnan(s)
    if what == "recall":
        return (s[y & dec] == 1).mean() if (y & dec).any() else np.nan
    if what == "false alarm":
        return (s[~y & dec] == 1).mean() if (~y & dec).any() else np.nan
    if what == "miss":
        return (s[y] == 0).mean() if y.any() else np.nan
    return safe_auc(y, p)


def boot(m, what):
    vals = np.array([stat_on(m, dr, what) for dr in draws], float)
    vals = vals[~np.isnan(vals)]
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)) if len(vals) else (np.nan, np.nan)


ci_rows = {m: {"recall CI": boot(m, "recall"), "false alarm CI": boot(m, "false alarm"),
               "miss rate CI": boot(m, "miss"), "AUC CI": boot(m, "auc")} for m in BY}
ci = pd.DataFrame(ci_rows).T
print(ci.to_string())


# %% 11. In-cycle provisional flag (the daily view)
# The app cannot wait for the next period. Same HMM on nights seen so far (each person scored by
# the HMM fitted without her); flag the first day the chance the switch already happened passes
# the threshold. Shown only as "after ovulation (estimated)", and only if cell 11b allows it.
curves = {}
for c in labelled:
    if c["enough"] and c["id"] in FITTED and FITTED[c["id"]] is not None:
        days = range(OV_MIN_DAY + 2, c["n"])
        curves[c["key"]] = (c, [(t, FITTED[c["id"]].p_switched_by(c, t)) for t in days])

prov = {}
for p_thr in [0.7, PROVISIONAL_P, 0.9]:
    lagv, f_ov, n_ov, f_an, n_an = [], 0, 0, 0, 0
    for c, curve in curves.values():
        first = next((t for t, p in curve if pd.notna(p) and p >= p_thr), None)
        if c["lab"] == 1:
            n_ov += 1
            if first is not None:
                f_ov += 1
                lagv.append(first - c["ov_day"])
        else:
            n_an += 1
            f_an += first is not None
    lagv = np.array(lagv)
    prov[f"P >= {p_thr}"] = {
        "ov cycles": n_ov, "flagged": round(f_ov / max(n_ov, 1), 3),
        "flagged within 3 days": round(float(((lagv >= 0) & (lagv <= 3)).sum()) / max(n_ov, 1), 3),
        "flagged too early": round(float((lagv < 0).sum()) / max(n_ov, 1), 3),
        "med days late": float(np.median(lagv)) if len(lagv) else np.nan,
        "anov cycles": n_an, "false flag": round(f_an / max(n_an, 1), 3) if n_an else np.nan}
prov_df = pd.DataFrame(prov).T
print(prov_df.to_string())
print("'within 3 days' = flagged 0 to 3 days after the LH-dated ovulation, as a share of ovulatory cycles.")


# %% 11b. Gate for the in-cycle message (pre-declared, so the README cannot promise an ungated output)
inc = prov[f"P >= {PROVISIONAL_P}"]
inc_pass = (pd.notna(inc["false flag"]) and inc["false flag"] <= INCYCLE_SHOW["max_false_flag"]
            and pd.notna(inc["flagged within 3 days"])
            and inc["flagged within 3 days"] >= INCYCLE_SHOW["min_within_3_days"])
print("In-cycle message gate:",
      "SHOW 'after ovulation (estimated)'" if inc_pass
      else "HIDE the in-cycle message (end-of-cycle answers only, or no ovulation wording at all)")


# %% 12. Decision, output by output, from rules written before the results (LOPO answers)
# Family: keep the HMM only if its AUC beats the step-fit with the paired-difference CI above 0.
# Equal = the simpler step-fit. Then each output gets its own pre-declared gate:
#   K1 end-of-cycle "ovulation seen / not seen / unclear":
#        a. false alarm (anovulatory called "seen") <= 20% and AUC CI lower end > 0.60
#        c. ECE <= 0.10      d. a definite answer for >= 70% of labelled cycles
#   K2 ovulation DAY from the watch: K1 passes AND b. median day error <= the calendar guess
#        (learned luteal length) on the same cycles AND within 2 days in >= 60% of "seen" cycles.
#        If K2 fails but K1 passes, the day shown is the calendar day, labelled as such.
#   K3 in-cycle "after ovulation (estimated)": K1 passes AND the cell 11b gate passes.
#   K4 Axis 2 in the agent layer: decided in cell 12b (needs K1 plus a measured false-escalation rate).
# Why output by output: the calendar can never say "no ovulation", so a working yes/no answer is
# worth keeping even if the calendar dates ovulation as well as the watch does.
# The 80/20 headline must point the same way; if it does not, report both and say so.
common = set(haves["HMM"]["key"]) & set(haves["Step-fit"]["key"])


def paired_diff(draw):
    out = []
    for m in ["HMM", "Step-fit"]:
        a = pooled(m, draw)
        if a is None:
            return np.nan
        y, _, p, keys = a
        keep = np.isin(keys, list(common))
        out.append(safe_auc(y[keep], p[keep]))
    return out[0] - out[1]


pt = paired_diff(pid_list)
vals = np.array([paired_diff(dr) for dr in draws], float)
vals = vals[~np.isnan(vals)]
fam_ci = (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)) if len(vals) else (np.nan, np.nan)
CHOSEN = "HMM" if pd.notna(fam_ci[0]) and fam_ci[0] > 0 else "Step-fit"
r = checks[CHOSEN]                     # plain dict of the LOPO scores (same numbers as checks_df)
auc_lo = ci_rows[CHOSEN]["AUC CI"][0]
gates = {
    "a. false alarm <= 20% and AUC CI low > 0.60": bool(pd.notna(r["false alarm"]) and r["false alarm"] <= TARGET_FP
                                                       and pd.notna(auc_lo) and auc_lo > GATE_AUC_LO),
    "b. beats the calendar day guess (learned luteal)": bool(pd.notna(r["day err model"]) and pd.notna(r["day err calendar (learned)"])
                                                            and r["day err model"] <= r["day err calendar (learned)"]),
    "c. ECE <= 0.10": bool(pd.notna(r["ECE"]) and r["ECE"] <= GATE_ECE),
    "d. definite answer for >= 70%": bool(r["coverage"] >= GATE_COVERAGE)}
show_day = bool(pd.notna(r["day within 2"]) and r["day within 2"] >= GATE_DAY)
K1 = gates["a. false alarm <= 20% and AUC CI low > 0.60"] and gates["c. ECE <= 0.10"] and gates["d. definite answer for >= 70%"]
K2 = K1 and gates["b. beats the calendar day guess (learned luteal)"] and show_day
K3 = K1 and inc_pass
print("Family:", CHOSEN, f"(paired AUC diff {round(pt, 3) if pd.notna(pt) else 'n/a'}, 95% CI {fam_ci})")
for g_, ok in gates.items():
    print(f"  {g_}: {'PASS' if ok else 'FAIL'}")
print(f"  day within 2 days in >= 60% of seen cycles: {'PASS' if show_day else 'FAIL'} "
      f"({round(r['day within 2'], 3) if pd.notna(r['day within 2']) else 'n/a'})")
print()
print("K1 end-of-cycle ovulation answer:", "SHIP" if K1 else "KILL")
print("K2 ovulation day:", "from the watch" if K2 else "from the calendar rule (cycle length minus learned luteal length)")
print("K3 in-cycle message:", "SHOW" if K3 else "HIDE")
h = headline.loc[CHOSEN]
print(f"80/20 headline for {CHOSEN}: F1 {h['F1']}, specificity {h['specificity']}, miss rate {h['miss rate']}, AUC {h['AUC']}")
for n_ in NOTES:
    print("NOTE:", n_)
if not K1:
    print()
    print("KILL CRITERION (pre-declared) is met. The app ships only:")
    print("  - cycle day, cycle history, cycle length and FIGO regularity from her diary")
    print("  - Axis 1 (Rotterdam pattern from the diary) and the 90-day no-period message")
    print("  - the calendar ovulation estimate, labelled 'estimated from your cycle lengths'")
    print("  - NO wearable ovulation output, in-cycle or end-of-cycle, and NO Axis 2")
    print("  The report states: in this sample, wrist data added no reliable ovulation signal beyond")
    print("  the period diary and the calendar rule.")
print("Check run: rerun every cell with (MIN_CYCLE_GAP, MIN_DRY_DAYS, MIN_BLEED_DAYS) = (18, 10, 2). If any K answer")
print("changes, report both runs and say the result depends on period-start cleaning.")


# %% 12b. Agent layer: Axis 2, measured against the hormone key, and the escalation rules
# Sources:
#   - Carmina, Contemp Ob/Gyn 2003: in hyperandrogenic women with regular periods, progesterone on
#     day 22-23; under 3 ng/mL means one more cycle is tested; "a single anovulatory cycle isn't
#     enough". 15-21% of hyperandrogenic women with normal menses are anovulatory (Azziz: up to 40%
#     of hirsute women who menstruate regularly).
#   - Boyle & Teede, Aust Fam Physician 2012 (2011 guideline): Rotterdam 2 of 3; oligo/anovulation
#     usually cycles > 35 or < 21 days; "even women with regular cycles may be anovulatory";
#     exclude thyroid, prolactin and other causes first; irregular cycles into the third year after
#     the first period should be investigated.
#   - Legro et al., Endocrine Society guideline 2013: bleeding every 25-35 days can still be
#     anovulatory; adult criteria do not apply to adolescents (85% of cycles anovulatory in year 1
#     after the first period, 59% in year 3, 25% by year 6).
# What the app can and cannot do: it can see the ovulation domain (diary, and the watch if K4
# passes) and Model 2 can see androgen signs. It cannot test thyroid, prolactin or adrenal causes,
# so it never says "PCOS"; the strongest wording is "worth discussing with a doctor soon".


# --- Axis 2 per cycle, in time order ("unclear" / "not enough data" skipped: no evidence either way)
def axis2_history(d):
    out = []
    answered = d[d["seen"].notna()].sort_values(["id", "seg", "cycle_no"])
    for pid, g in answered.groupby("id"):
        s, lab = g["seen"].to_numpy(), g["lab"].to_numpy()
        seg, cno = g["seg"].to_numpy(), g["cycle_no"].to_numpy()
        for k in range(len(s)):
            j = max(0, k - AXIS2_RUN + 1)
            run, labs = s[j:k + 1], lab[j:k + 1]
            together = (seg[j:k + 1] == seg[k]).all() and (np.diff(cno[j:k + 1]) <= 2).all()
            out.append({"id": pid, "key": g["key"].iloc[k], "seg": g["seg"].iloc[k], "cycle_no": g["cycle_no"].iloc[k],
                        "axis2": len(run) == AXIS2_RUN and together and (run == 0).all(),
                        "run_truth": ("both ovulatory" if len(labs) == AXIS2_RUN and (labs == 1).all() else
                                      "both anovulatory" if len(labs) == AXIS2_RUN and (labs == 0).all() else
                                      "mixed or unlabelled")})
    return pd.DataFrame(out, columns=["id", "key", "seg", "cycle_no", "axis2", "run_truth"])


ax2 = {m: axis2_history(preds[preds["model"] == m]) for m in MODELS}
audit_rows = {}
for m, a2 in ax2.items():
    ovov = a2[a2["run_truth"] == "both ovulatory"]
    anan = a2[a2["run_truth"] == "both anovulatory"]
    miss = checks[m]["miss rate"]
    audit_rows[m] = {"runs of 2 ovulatory cycles": len(ovov),
                     "false escalations": int(ovov["axis2"].sum()),
                     "false escalation rate": round(ovov["axis2"].mean(), 3) if len(ovov) else np.nan,
                     "expected if independent (miss rate squared)": round(miss ** 2, 3) if pd.notna(miss) else np.nan,
                     "runs of 2 anovulatory cycles": len(anan),
                     "caught": int(anan["axis2"].sum()),
                     "people with Axis 2 at any point": int(a2.groupby("id")["axis2"].any().sum())}
axis2_audit = pd.DataFrame(audit_rows).T
print("Axis 2 against the hormone key (leave-one-person-out answers):")
print(axis2_audit.to_string())
print("If the real false-escalation rate is well above 'miss rate squared', misses cluster in the same")
print("people (for example loose watch wear), and two 'not seen' cycles are weaker evidence than they look.")

a_row = audit_rows[CHOSEN]
testable = a_row["runs of 2 ovulatory cycles"] >= 10
K4 = bool(K1 and pd.notna(r["miss rate"]) and r["miss rate"] <= AXIS2_MAX_MISS and testable
          and a_row["false escalation rate"] <= AXIS2_MAX_FALSE)
print("K4 Axis 2 in the agent layer:", "ON" if K4 else "OFF",
      f"(K1 {'pass' if K1 else 'fail'}; miss rate {round(r['miss rate'], 3) if pd.notna(r['miss rate']) else 'n/a'}"
      f" vs <= {AXIS2_MAX_MISS}; false escalation {a_row['false escalation rate']} vs <= {AXIS2_MAX_FALSE};"
      f" {int(a_row['runs of 2 ovulatory cycles'])} testable runs, need >= 10)")
if a_row["runs of 2 anovulatory cycles"] < 5:
    print(f"NOTE: only {int(a_row['runs of 2 anovulatory cycles'])} runs of two anovulatory cycles, so how often Axis 2")
    print("catches real repeated anovulation cannot be measured here. Report it as untested, not as a strength.")


# --- The escalation rule: count Rotterdam DOMAINS, each once
def agent_level(model2_flag=False, model2_flag_cycle_neutral=None, scan_pcom=None,
                axis1_pattern=False, axis1_infrequent=False, axis2=False, no_period_90=False,
                years_since_menarche=None, hormonal_contraception=False, pregnant_or_breastfeeding=False,
                axis2_allowed=K4):
    """Returns (level, message, reasons). Levels: S see a doctor (no period 90+ days), L2 worth
    discussing with a doctor soon, L1 worth watching, INFO neutral ovulation note, L0 nothing.

    Domains (Rotterdam 2 of 3), each counted once:
      O  ovulation : Axis 1 (diary pattern or < 8 a year) OR Axis 2 (two "not seen" in a row, if allowed)
      A  androgen  : Model 2 still flags with her cycle inputs set to 'regular' (so cycle evidence
                     is not counted twice). Until Model 2 can give that cycle-neutral flag, its
                     normal flag counts as A, and the reasons say it may include her cycle inputs.
      M  ovary scan: only if she enters an ultrasound result (most users won't have one)
    Ovulation evidence from the watch never raises a flag alone: Axis 2 on its own gives INFO only."""
    reasons = []
    paused = hormonal_contraception or pregnant_or_breastfeeding
    if paused:
        reasons.append("cycle checks paused (hormonal contraception, pregnancy or breastfeeding)")
    if no_period_90 and not paused:
        return ("S", "You haven't logged a period for 3 months or more. If there's any chance you could be "
                     "pregnant, take a test. Either way, please see a doctor about it.", ["no period 90+ days"])
    early = years_since_menarche is not None
    diary_ok = not paused and not (early and years_since_menarche < 2)
    watch_ok = not paused and axis2_allowed and not (early and years_since_menarche < MENARCHE_YEARS_MIN)
    o_diary = diary_ok and (axis1_pattern or axis1_infrequent)
    o_watch = watch_ok and axis2
    a = bool(model2_flag_cycle_neutral) if model2_flag_cycle_neutral is not None else bool(model2_flag)
    m = bool(scan_pcom)
    if o_diary:
        reasons.append("cycle lengths outside 21-35 days consistently, or fewer than 8 a year")
    if o_watch:
        reasons.append("no sign of ovulation in her last 2 answered cycles")
    if a:
        reasons.append("Model 2 flag from androgen-related signs" if model2_flag_cycle_neutral is not None or not o_diary
                       else "Model 2 flag (may partly come from her cycle inputs: ask Model 2 for a cycle-neutral flag)")
    if m:
        reasons.append("polycystic ovaries on a scan she entered")
    domains = int(o_diary or o_watch) + int(a) + int(m)
    if domains >= 2:
        return ("L2", "What you've logged, taken together, is worth discussing with a doctor soon. It can have "
                      "several causes, and only a doctor can tell which. They may check ovulation with a "
                      "progesterone blood test in the second half of your cycle, often over two cycles, and "
                      "rule out thyroid and other hormone causes.", reasons)
    if domains == 1 and o_watch and not (o_diary or a or m):
        return ("INFO", "In your last two cycles the watch didn't pick up the usual signs that ovulation "
                        "happened. This happens now and then in healthy cycles, and the watch can miss it. "
                        "Keep logging; there's nothing you need to do now.", reasons)
    if domains == 1:
        return ("L1", "Something in your logs is worth keeping an eye on. Keep logging your periods and "
                      "symptoms and we'll look again after your next cycles. If it continues, mention it to your GP.",
                reasons)
    return ("L0", "", reasons)


# Decision table: every combination that matters, so the team can read the rule without code
scen = []
for m2 in [False, True]:
    for regular in [True, False]:
        for a2 in [False, True]:
            lv, msg, why = agent_level(model2_flag=m2, axis1_pattern=not regular, axis2=a2, axis2_allowed=True)
            lv_off = agent_level(model2_flag=m2, axis1_pattern=not regular, axis2=a2, axis2_allowed=False)[0]
            scen.append({"Model 2 flag": m2, "cycle lengths": "regular 21-35" if regular else "outside 21-35 (2 of 3)",
                         "Axis 2: 2 'not seen' in a row": a2, "level (Axis 2 ON)": lv, "level (Axis 2 OFF)": lv_off,
                         "why": "; ".join(why)})
for label, kw in [("no period 90+ days", dict(no_period_90=True)),
                  ("Model 2 flag + Axis 2, on the pill", dict(model2_flag=True, axis2=True, hormonal_contraception=True)),
                  ("Model 2 flag + Axis 2, 2 years after first period", dict(model2_flag=True, axis2=True, years_since_menarche=2)),
                  ("Model 2 flag only from cycle inputs, lengths outside 21-35",
                   dict(model2_flag=True, model2_flag_cycle_neutral=False, axis1_pattern=True)),
                  ("Model 2 flag + scan result entered", dict(model2_flag=True, scan_pcom=True))]:
    lv, msg, why = agent_level(axis2_allowed=True, **kw)
    scen.append({"Model 2 flag": label, "cycle lengths": "", "Axis 2: 2 'not seen' in a row": "",
                 "level (Axis 2 ON)": lv, "level (Axis 2 OFF)": agent_level(axis2_allowed=False, **kw)[0],
                 "why": "; ".join(why)})
decision_table = pd.DataFrame(scen)
print()
print("Agent-layer decision table (S see a doctor, L2 discuss with a doctor soon, L1 worth watching, INFO note):")
print(decision_table.drop(columns="why").to_string(index=False))
lv, msg, why = agent_level(model2_flag=True, axis2=True, axis2_allowed=True)
print("\nThe case 'Model 2 flag + 2 anovulatory cycles + regular lengths' ->", lv)
print("  because:", "; ".join(why))
print("  message:", msg)

# Applied to the mcPHASES people as they stand at their last cycle. Model 2 outputs are not in this
# dataset (different people), so this shows only what the diary and watch would contribute.
last_a1 = axis1.sort_values(["id", "seg", "cycle_no"]).groupby("id").last()
last_a2 = ax2[CHOSEN].sort_values(["id", "seg", "cycle_no"]).groupby("id").last()
lvl_rows = []
for pid in sorted(set(cycles["id"])):
    a1p = bool(last_a1["axis1_pattern"].get(pid, False))
    a1i = bool(last_a1["axis1_infrequent"].get(pid, False))
    a2v = bool(last_a2["axis2"].get(pid, False)) if len(last_a2) else False
    np90 = pid in amen.index
    lvl_rows.append({"id": pid, "axis1_pattern": a1p, "axis1_infrequent": a1i, "axis2": a2v, "no_period_90": np90,
                     "level, no Model 2 flag": agent_level(axis1_pattern=a1p, axis1_infrequent=a1i, axis2=a2v,
                                                           no_period_90=np90)[0],
                     "level, if Model 2 flagged": agent_level(model2_flag=True, axis1_pattern=a1p, axis1_infrequent=a1i,
                                                              axis2=a2v, no_period_90=np90)[0]})
person_levels = pd.DataFrame(lvl_rows).set_index("id")
print("\nmcPHASES people by level (counts only):")
print(pd.DataFrame({c: person_levels[c].value_counts() for c in ["level, no Model 2 flag", "level, if Model 2 flagged"]})
      .fillna(0).astype(int).to_string())
print("Sanity: mcPHASES is mostly healthy young adults, so few people should reach L2 even if Model 2")
print("flagged everyone. If many do, Axis 1 or Axis 2 is too loose; fix that before any wording ships.")


# %% 13. Export: summaries to results/ (safe to commit), per-person tables to data/ (stay local)
# The diary summaries were already saved in cell 4c; this adds the ovulation and agent-layer results.
OUT.mkdir(parents=True, exist_ok=True)
pd.crosstab(cyc.loc[cyc["usable"], "key_type"], cyc.loc[cyc["usable"], "ov_label"]).to_csv(OUT / "answer_key_counts.csv")
round_tests.to_csv(OUT / "hormone_tests_by_round.csv")
sig_check.to_csv(OUT / "signal_screen.csv")
headline.to_csv(OUT / "headline_80_20_x5.csv")
per_split.round(3).to_csv(OUT / "per_split_scores.csv", index=False)
checks_df.round(3).to_csv(OUT / "lopo_checks.csv")
if len(by_key_df):
    by_key_df.round(3).to_csv(OUT / "lopo_checks_by_key_type.csv")
ci.to_csv(OUT / "lopo_ci.csv")
prov_df.to_csv(OUT / "in_cycle_flag.csv")
axis2_audit.to_csv(OUT / "axis2_audit.csv")
decision_table.to_csv(OUT / "agent_decision_table.csv", index=False)
person_levels.drop(columns=[c for c in person_levels.columns if not c.startswith("level")]).apply(
    pd.Series.value_counts).fillna(0).astype(int).to_csv(OUT / "agent_level_counts.csv")
pd.Series({**gates, "family": CHOSEN, "K1 end-of-cycle answer": K1, "K2 watch day": K2,
           "K3 in-cycle message": K3, "K4 Axis 2": K4, "period fallback used": bool(use_label.any()),
           "notes": " | ".join(NOTES)}).rename("result").to_csv(OUT / "decision.csv")
print("summaries saved to", OUT.resolve())

# Per-person handoff for the agent layer and demo: the diary answers from cell 4c plus the LOPO answers
# of the chosen model (each person answered by a model that never saw her). Contains participant ids,
# so it stays in data/ and is never committed or uploaded.
ans = preds[(preds["model"] == CHOSEN) & preds["seen"].notna()].groupby("id")["seen"]
handoff = diary_handoff.copy()
handoff["cycles_answered"] = ans.size().reindex(handoff.index).fillna(0).astype(int)
handoff["ovulation_seen_share"] = ans.mean().reindex(handoff.index).round(2)
handoff["axis2"] = person_levels["axis2"].reindex(handoff.index).fillna(False).astype(bool)
handoff["axis2_allowed"] = K4
handoff.to_csv(HANDOFF)
print("per-person handoff saved to", HANDOFF, "(local only)")

# %%# How to use: open your own model1_pipeline_v6.py, run cells 1 to 13 as usual, then run the cells
# below in the same session (they reuse its variables). They print counts, medians and effect sizes
# only, never rows. Nothing here changes the pre-declared v6 result.
#   X1  is the temperature dated to the right day?             (data check)
#   X2  why does a model score BELOW chance?                   (diagnostic)
#   X3  in-cycle flag vs a calendar that only knows past cycles (fair comparator)
#   X4  OPTIONAL post-hoc variant: the switch must sit 9-17 days before the next period. Run X4, then
#       rerun v6 cells 9 to 13. Results go to a separate folder. Report it as an analysis added
#       AFTER seeing the first results, next to the pre-declared run, never instead of it.

# %% X1. Is each nightly signal dated to the right day?
# Shift each signal by one day each way and recompute the after-vs-before-ovulation effect (same
# masks as cell 7). If -1 or +1 beats 0 by 0.1 or more, the night is attached to the wrong date in
# build_daily_table.py (for example, sleep that starts on Monday evening stored under Tuesday).
align_rows = {}
for col in CANDIDATES:
    out = {}
    for shift in [-1, 0, 1]:
        dev = raw.groupby("seg")[f"{col}_dev"].shift(shift)
        out[f"shift {shift:+d}"] = round(float((dev[after].mean() - dev[before].mean()) / dev.std()), 2) \
            if after.any() and before.any() else np.nan
    align_rows[col] = out
align_check = pd.DataFrame(align_rows).T
print("Effect after vs before ovulation (in her own noise units) with the signal moved by a day:")
print(align_check.to_string())
best = align_check.abs().idxmax(axis=1)
for col, b in best.items():
    if b != "shift +0" and align_check.loc[col, b] - abs(align_check.loc[col, "shift +0"]) >= 0.1:
        print(f"CHECK: {col} separates best at {b}. Fix the date of this signal in build_daily_table.py, rebuild, rerun.")
print("For reference: a nightly wrist-temperature rise usually gives an effect well above 0.5. "
      "Much lower points to the wrong column, wrong date or a lot of missing nights.")


# %% X2. Why can a model score below chance (AUC under 0.5)?
# Compares ovulatory and anovulatory cycles: length, where the model put the switch, and the
# probability it gave. If anovulatory cycles are much LONGER and still get HIGH probabilities, slow
# temperature drift over a long cycle is being read as a post-ovulation rise (X4 tests a fix).
diag_rows = []
for m, d in preds[preds["lab"].notna()].groupby("model"):
    for lab_value, g in d.groupby("lab"):
        diag_rows.append({"model": m, "answer key": "ovulated" if lab_value == 1 else "not ovulated",
                          "cycles": len(g), "median length": g["n"].median(),
                          "median probability": round(g["prob"].median(), 3),
                          "share unclear": round(g["call"].eq("unclear").mean(), 2),
                          "share from LH-only key": round(g["key_type"].eq("LH only").mean(), 2)})
why_below = pd.DataFrame(diag_rows)
print(why_below.to_string(index=False))
print("Read it like this: a useful model gives a clearly higher median probability to 'ovulated' cycles.")


# %% X3. In-cycle flag vs a calendar that only knows her PAST cycles (the fair in-cycle comparator)
# Mid-cycle the app cannot know this cycle's length, so the fair calendar predicts ovulation from her
# previous cycle: previous length minus the typical luteal length learned from the OTHER people.
# It then shows "after ovulation (estimated)" from the day after that prediction.
lut_by_pid = {}
ov_lab = [c for c in labelled if c["lab"] == 1]
for pid in people:
    others = [c["n"] - c["ov_day"] for c in ov_lab if c["id"] != pid]
    lut_by_pid[pid] = float(np.median(others)) if others else float(CAL_FIXED_LUTEAL)
prev_len = cycles.sort_values(["seg", "cycle_no"]).groupby("seg")["length"].shift(1)
prev_map = dict(zip(cycles["seg"] + "|" + cycles["cycle_no"].astype(str), prev_len))
lags, n_ov, f_an, n_an = [], 0, 0, 0
for c in labelled:
    p = prev_map.get(c["key"], np.nan)
    if np.isnan(p):
        continue                                     # first cycle of a round: no past cycle to use
    flag_day = p - lut_by_pid[c["id"]] + 1
    flagged = flag_day < c["n"]                      # the flag only shows if the period hasn't come yet
    if c["lab"] == 1:
        n_ov += 1
        if flagged:
            lags.append(flag_day - c["ov_day"])
    else:
        n_an += 1
        f_an += int(flagged)
lags = np.array(lags)
cal_row = {"ov cycles": n_ov, "flagged": round(len(lags) / max(n_ov, 1), 3),
           "flagged within 3 days": round(float(((lags >= 0) & (lags <= 3)).sum()) / max(n_ov, 1), 3),
           "flagged too early": round(float((lags < 0).sum()) / max(n_ov, 1), 3),
           "med days late": float(np.median(lags)) if len(lags) else np.nan,
           "anov cycles": n_an, "false flag": round(f_an / max(n_an, 1), 3) if n_an else np.nan}
incycle_compare = pd.concat([prov_df.loc[[f"P >= {PROVISIONAL_P}"]].rename(index=lambda s: f"watch (HMM, {s})"),
                             pd.DataFrame([cal_row], index=["calendar (previous cycle length)"])])
print(incycle_compare.to_string())
print("The watch's in-cycle message is only worth having if it beats this calendar row on 'within 3 days'")
print("without a higher false flag. The calendar can never spot a cycle without ovulation.")
OUT.mkdir(parents=True, exist_ok=True)
align_check.to_csv(OUT / "x1_alignment_check.csv")
why_below.to_csv(OUT / "x2_below_chance_check.csv", index=False)
incycle_compare.to_csv(OUT / "x3_incycle_vs_calendar.csv")
print("saved x1-x3 summaries to", OUT)


# %% X4. OPTIONAL post-hoc variant: the switch must fall 9-17 days before the next period
# Why: after ovulation the luteal phase lasts about 10-16 days before the next period. A real rise
# therefore starts in that window and holds until the period; slow drift in a long cycle does not
# have to. Run this cell, then rerun v6 cells 9 to 13. Results go to a separate folder.
LUTEAL_RANGE = (9, 17)
_STEP_CACHE_LW = {}


def switch_days(n):
    return np.arange(max(OV_MIN_DAY, n - LUTEAL_RANGE[1]), n - LUTEAL_RANGE[0] + 1)


class StepModelLW(StepModel):
    def feats(self, c):
        if c["key"] in _STEP_CACHE_LW:
            return _STEP_CACHE_LW[c["key"]]
        out = None
        y = np.nanmean(c["Z"], axis=1)
        if c["enough"] and len(y) >= MIN_NIGHTS:
            rss0 = ((y - y.mean()) ** 2).sum()
            best_ = None
            for k in switch_days(c["n"]):
                a, b = y[c["t"] < k], y[c["t"] >= k]
                if len(a) < 3 or len(b) < 3:
                    continue
                rss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
                if best_ is None or rss < best_[0]:
                    best_ = (rss, int(k), b.mean() - a.mean())
            if best_ is not None:
                m_ = len(y)
                gain = m_ * np.log(max(rss0, 1e-9) / max(best_[0], 1e-9)) - 2 * np.log(m_)
                out = (np.array([gain * np.sign(best_[2]), best_[2]]), best_[1])
        _STEP_CACHE_LW[c["key"]] = out
        return out


class HMMModelLW(HMMModel):
    def score(self, c):
        if not c["enough"]:
            return None
        ks = switch_days(c["n"])
        if not len(ks):
            return None
        l0, l1 = self.loglik(c)
        c0, c1 = np.concatenate([[0], np.cumsum(l0)]), np.concatenate([[0], np.cumsum(l1)])
        idx = np.searchsorted(c["t"], ks)
        prior = norm.logpdf(c["n"] - ks, self.Lmu, self.Lsd)
        prior -= logsumexp(prior)
        ll_k = c0[idx] + (c1[-1] - c1[idx]) + prior
        log_odds = (logsumexp(ll_k) + np.log(self.pi)) - (c0[-1] + np.log(1 - self.pi))
        post = np.exp(ll_k - logsumexp(ll_k))
        cdf = np.cumsum(post)
        return (float(log_odds), int(ks[np.argmax(post)]), int(ks[min(np.searchsorted(cdf, 0.1), len(ks) - 1)]),
                int(ks[min(np.searchsorted(cdf, 0.9), len(ks) - 1)]))


MODELS = {"Step-fit": StepModelLW, "HMM": HMMModelLW}
OUT = OUT if OUT.name == "luteal_window" else OUT / "luteal_window"
HANDOFF = HANDOFF.with_name(HANDOFF.stem.replace("_luteal_window", "") + "_luteal_window.csv")
NOTES.append("POST-HOC variant (added after the first real run): switch limited to 9-17 days before the next period")
print("Variant set. Now rerun v6 cells 9 to 13. Results will go to", OUT)

# %%
