# CycleSignal

### AI-powered cycle tracking for earlier PCOS risk awareness

CycleSignal is a Women in AI Ireland group project exploring whether everyday cycle tracking, self-reported symptoms and machine learning can help someone notice a concerning PCOS-related pattern earlier and take better information to their GP.

> **CycleSignal is a risk-awareness prototype, not a diagnostic tool or medical device.**

[Live App](https://cyclesignal-wai.streamlit.app/) · [Project Blog](https://f-ayanae.github.io/WAI---GROUP1/) · [Source Code](https://github.com/eoc3001/WAI---GROUP1)

---

## Why CycleSignal?

PCOS can present through several signs — irregular cycles, weight changes, acne, excess hair growth or skin changes — that may be easy to dismiss individually.

CycleSignal explores whether patterns collected over time can provide a clearer signal for **earlier awareness**, while keeping the final decision with a healthcare professional.

The prototype combines:

- longitudinal cycle information from period logs
- a machine-learning PCOS risk profile from self-reportable inputs
- SHAP explanations showing which answers influenced a person's result
- a rule-based escalation layer
- an optional AI assistant that explains the result in plain language

---

## How it works

```text
Period logs
    │
    ▼
Model 1
Cycle-pattern rules
    │
    │ 40%
    ▼
                    CycleSignal
                    risk-awareness
                        score
                    0 ───── 100
                        ▲
    │ 60%               │
    │                   │
Model 2 ────────────────┘
Extra Trees PCOS profile
    ▲
    │
11 self-reportable inputs
```

### Model 1 — cycle pattern

Model 1 uses logged period history to calculate:

- cycle length
- regularity
- persistent short or long cycles
- infrequent periods
- 90+ day gaps without a period

The mcPHASES wearable dataset was also used to investigate ovulation detection from wrist temperature.

That experiment **did not pass the project's validation criteria**, so temperature is shown only as context and does not contribute to the CycleSignal score or escalation decision.

### Model 2 — PCOS risk profile

Model 2 uses **11 self-reportable inputs**:

- age
- BMI
- self-reported period regularity
- period length (days of bleeding)
- weight gain
- excess hair growth
- skin darkening
- hair loss
- acne
- frequent fast food
- regular exercise

Four classifiers were compared:

| Model | Test ROC-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| **Extra Trees** | **0.92** | **0.81** | 0.78 | **0.79** |
| Random Forest | 0.91 | 0.75 | 0.82 | 0.78 |
| Logistic Regression | 0.89 | 0.72 | **0.87** | 0.79 |
| XGBoost | 0.90 | 0.72 | 0.81 | 0.76 |

**Extra Trees** was selected for the prototype because it provided the strongest recall while maintaining competitive precision and F1.

At the default 0.50 threshold:

- ROC-AUC: **0.922**
- Recall: **80.6%**
- Precision: **78.4%**
- F1: **79.5%**
- Accuracy: **86.2%**
- TP: 29 · TN: 65 · FP: 8 · FN: 7

### Explainability with SHAP

SHAP is used at two levels:

- **Global:** identifies which features the model relies on most across the dataset.
- **Per person:** explains which of that user's answers increased or decreased their profile score.

The strongest global contributors were:

1. skin darkening
2. excess hair growth
3. weight gain

followed by fast food, acne and cycle-related inputs.

These are **model associations, not causal medical conclusions**.

---

## The CycleSignal score

The prototype combines the two components:

```text
CycleSignal score
= 40% cycle-pattern score
+ 60% Model 2 profile score
```

Scores are displayed in four awareness bands:

**Low · Some patterns to monitor · Increased · Elevated**

The **40/60 weighting and score bands are project heuristics**. They have not been clinically validated and the score is not a probability of having PCOS.

Model 2 already contains self-reported cycle information, so some cycle evidence overlaps between the two score components. The agent uses a cycle-neutralised Model 2 check when making escalation decisions to reduce double-counting there.

---

## The app

The Streamlit prototype brings the pipeline together in six steps:

**Watch → Log → Score → Assistant → What-if → Doctor summary**

The app includes:

- cycle-history tracking
- the real Model 2 Extra Trees pipeline
- CycleSignal's combined 0–100 score
- per-person SHAP explanations
- what-if analysis
- rule-based GP escalation guidance
- downloadable doctor summary
- optional Claude-powered explanations with an offline fallback

### Demo data

**Sarah** and **Aoife** are fictional demonstration users.

Wearable nights displayed in the prototype are simulated in the shape of the mcPHASES data. Model outputs, cycle rules and SHAP calculations are produced by the project code.

[Open the CycleSignal app](https://cyclesignal-wai.streamlit.app/)

---

## Data

CycleSignal uses two separate datasets because no available dataset contains both longitudinal wearable data and confirmed PCOS diagnoses for the same participants.

| Dataset | Used for | Data |
|---|---|---|
| [mcPHASES — PhysioNet](https://physionet.org/content/mcphases/1.0.0/) | Model 1 / cycle analysis | 42 participants in Canada; wearable, hormone and diary data |
| [PCOS dataset — Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data) | Model 2 | 541 clinical records from 10 hospitals in Kerala, India; 177 PCOS cases |

The two models therefore operate on **different populations**.

---

## Repository structure

```text
WAI---GROUP1/
│
├── model1/          # Cycle-pattern and wearable analysis
├── model2/          # PCOS profile modelling and evaluation
├── combined_score/  # 40/60 CycleSignal score
├── agent/           # Escalation logic and AI/offline assistant
├── app/             # Streamlit prototype
├── docs/            # Project blog / GitHub Pages
├── assets/          # Project assets
└── README.md
```

More detail is available inside the README for each component.

---

## Run locally

Clone the repository and install the requirements:

```bash
git clone https://github.com/eoc3001/WAI---GROUP1.git
cd WAI---GROUP1
pip install -r requirements.txt
```

### Run Model 2

```bash
python model2/build_pcos_model2.py
```

### Run the combined score

```bash
python combined_score/cyclesignal_risk_score.py
```

### Run the Streamlit app

```bash
streamlit run app/cyclesignal_app.py
```

Model 1 requires the mcPHASES source files locally. See [`model1/README.md`](model1/README.md) for details.

---

## Key limitations

- The wearable/cycle dataset and PCOS clinical dataset contain **different people**, so the combined CycleSignal score has not been validated end to end.
- Model 2 was trained on data from one hospital network in Kerala, India and may not generalise to other populations.
- Wrist-temperature ovulation detection did not pass the project's validation checks.
- The combined-score weights and bands are project-defined rather than clinically learned.
- Model 2 contains some cycle-related inputs, creating partial overlap with the cycle-pattern component.
- CycleSignal is a student prototype and must not be used for diagnosis, contraception or clinical decision-making.

---

## Next steps

A clinically useful version would require:

- wearable, symptom and diagnosis data from the **same participants**
- a larger and more diverse cohort
- longitudinal follow-up over 6–12 months
- validated ovulation detection
- calibration of the combined risk estimate
- prospective clinical evaluation
- appropriate privacy, regulatory and medical-device review

---

## Project links

- **Live app:** https://cyclesignal-wai.streamlit.app/
- **Project blog:** https://f-ayanae.github.io/WAI---GROUP1/
- **GitHub repository:** https://github.com/eoc3001/WAI---GROUP1
- **Model 1:** [`model1/`](model1/)
- **Model 2:** [`model2/`](model2/)
- **Combined score:** [`combined_score/`](combined_score/)
- **Agent:** [`agent/`](agent/)
- **App:** [`app/`](app/)
- **Blog source:** [`docs/`](docs/)

---

## Team

**Women in AI Ireland — Group 1**

- **Emma O'Connell** — Mentor · [LinkedIn](https://www.linkedin.com/in/emma-o-connell-735681184/) · [GitHub](https://github.com/eoc3001)
- **Anuja Thuraiyur Jayakumar** · [LinkedIn](https://www.linkedin.com/in/anuja-jayakumar-datascience001/) · [GitHub](https://github.com/anujatj0011)
- **Fiona Edocket** · [LinkedIn](https://www.linkedin.com/in/fionaedocket) · [GitHub](https://github.com/f-ayanae)
- **Ling Tsai** · [LinkedIn](https://www.linkedin.com/in/yltsai/) · [GitHub](https://github.com/ling0110)

---

## References

- [WHO — Polycystic ovary syndrome](https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome)
- [mcPHASES — PhysioNet](https://physionet.org/content/mcphases/1.0.0/)
- [PCOS dataset — Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data)
- [2023 International Evidence-based PCOS Guideline](https://doi.org/10.1210/clinem/dgad463)
- [SHAP — Lundberg & Lee](https://arxiv.org/abs/1705.07874)

---

**CycleSignal — earlier awareness, not a diagnosis.**

