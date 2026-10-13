# WAI Group 001 · Model 1 (wearable cycle-phase model)

Estimates the daily menstrual phase from Fitbit signals in mcPHASES, flags the fertile
window, and feeds cycle context to the agent layer. 

## Files
| File | What it does |
|---|---|
| `model1_pipeline_v6.py` | Current cycle version: works with the real round layout, LH-only key where no PdG, diary answers saved first |
| `results/` | Summary scores only (no participant rows or ids) |

## Set up (once)
    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # Mac
    pip install -r requirements.txt

Before your first commit (inside the Git repository):

    nbstripout --install            # clears notebook outputs automatically when you commit

## Run
    python build_daily_table.py --raw data/mcphases --inspect     # check real headers first
    python build_daily_table.py --raw data/mcphases --out data/daily.csv
    python model1_pipeline.py --data data/daily.csv --repeats 5 --out results/model1_results.json


## Model
Logistic regression baseline, random forest / XGBoost later; 80/20 split by participant;
scored with TP, TN, FP, FN, precision, recall and F1.
