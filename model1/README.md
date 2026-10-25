# WAI Group 001 · Model 1 (wearable cycle-phase model)

Estimates the daily menstrual phase from Fitbit signals in mcPHASES, flags the fertile
window, and feeds cycle context to the agent layer. 

## Files
| File | What it does |
|---|---|
| `model1/model1_pipeline_v6.py` | Current cycle version: works with the real round layout, LH-only key where no PdG, diary answers saved first |
| `model1/results/` | Summary scores only (no participant rows or ids) |

## Set up (once)
    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # Mac
    pip install -r requirements.txt

Before your first commit (inside the Git repository):

    nbstripout --install            # clears notebook outputs automatically when you commit

## Run (from the repository root)
    python model1/build_daily_table.py --raw data/mcphases --inspect      # check real headers first
    python model1/build_daily_table.py --raw data/mcphases --out data/daily.csv
    python model1/model1_pipeline_v6.py                                   # settings are at the top of the file

Summary results are saved to `model1/results/`. Per-person files (`data/daily.csv`,
`data/cycle_handoff_v6.csv`) stay on your laptop; `data/` is blocked from git.


## Model
Logistic regression baseline, random forest / XGBoost later; 80/20 split by participant;
scored with TP, TN, FP, FN, precision, recall and F1.
