---
layout: default
---

<img src="images/logo.png" alt="CycleSignal logo" width="320">

## Contributors

- **Emma O'Connell**: [LinkedIn](https://www.linkedin.com/in/emma-o-connell-735681184/) · [GitHub](https://github.com/eoc3001)
- **Anuja Thuraiyur Jayakumar**: [LinkedIn](https://www.linkedin.com/in/anuja-jayakumar-datascience001/) · [GitHub](https://github.com/anujatj0011)
- **Fiona Edocket**: [LinkedIn](https://www.linkedin.com/in/fionaedocket) · [GitHub](https://github.com/f-ayanae)
- **Ling Tsai**: [LinkedIn](https://www.linkedin.com/in/yltsai/) · [GitHub](https://github.com/ling0110)

**Try the demo:** [CycleSignal app](https://wai---group1-iwvd7wox6mmmbwhs2kxkyd.streamlit.app/) (password on request) · **Watch:** [demo video](https://YOUR-VIDEO-LINK) · **Code:** [GitHub repository](https://github.com/eoc3001/WAI---GROUP1)

# Welcome to CycleSignal: earlier awareness of PCOS risk from everyday tracking

## Introduction

Polycystic ovary syndrome (PCOS) affects an estimated 1 in 10 women of reproductive age, and the
World Health Organization reports that **up to 70% of affected women are undiagnosed**. Its signs
are scattered: irregular cycles, weight changes, acne, extra hair growth, dark patches of skin.
Each one is easy to dismiss on its own, nobody sees how a cycle behaves between GP visits, and
diagnosis often takes years.

Our question was simple: **could everyday data, from a wearable and a few self-reported answers,
help someone notice a concerning pattern earlier and take it to a GP?**

CycleSignal is our answer. It is a prototype that combines two machine learning models and an
AI assistant to give a **risk-awareness score**, explains *why* using the person's own data, and
points to a next step. It is deliberately **not a diagnosis**.

<div class="callout">A note on names: health services such as the NHS and HSE now call the condition
<b>PMOS (polyendocrine metabolic ovarian syndrome)</b>, formerly PCOS. We use PCOS here because
our datasets do, but the app explains both names.</div>

## Dataset Description

We used two public datasets, because no open dataset yet has wearable data *and* a PCOS diagnosis
for the same people.

| | mcPHASES (PhysioNet) | PCOS dataset (Kaggle) |
|---|---|---|
| What it is | Wearable, hormone and diary data for menstrual health tracking | Clinical records of patients assessed for PCOS |
| People | 42 young adults in Canada; 20 completed a second 3-month interval | 541 patients from 10 hospitals in Kerala, India (177 with PCOS) |
| What we used | Fitbit Sense overnight temperature and resting heart rate, Mira LH/PdG hormone tests, period diary | 11 inputs a person can self-report (no lab tests, no ultrasound) |
| Used for | **Model 1**: cycle tracking | **Model 2**: PCOS risk profile |

In mcPHASES we built a daily table per participant: 3,697 days from 42 people in the first interval
and 1,961 days from 20 people in the second.

## How CycleSignal works

![Logging periods and the monthly check-in](images/app-log.png)

1. **Wear and log.** The watch records overnight; she taps in period days and answers a short
   monthly check-in.
2. **Model 1** turns the period log (and watch data) into a cycle history.
3. **Model 2** scores her check-in answers against patterns learned from clinical patients.
4. **The combined score** blends the two, and **the agent layer** decides how strongly to
   recommend a GP visit and explains why.

### 1. Model 1: wearable cycle tracking

Model 1 tracks cycles from the period diary and estimates cycle phase from overnight skin
temperature, which rises slightly after ovulation.

- **Cycle length and regularity** come from logged period starts. Across 69 diary cycles the
  median length was 30 days (range 19 to 62). We use the FIGO 2018 definition of regular
  (every cycle 24 to 38 days and varying by no more than 9 days).
- **The irregular-cycle flag** follows the Rotterdam criteria: 2 of her last 3 cycles shorter
  than 21 or longer than 35 days, fewer than 8 periods a year, or no period for 90+ days
  (which always triggers "see a doctor").
- **Checking our labels:** regular/irregular from the Mira period label matched the diary for
  24 of 25 people.

**An honest result:** we also tried to detect *whether ovulation happened* from wrist temperature,
and set pass/fail criteria before testing. It **did not pass** (false-alarm rate, beating a
calendar-based guess, and answering confidently often enough all failed; only calibration
passed), partly because there were only 13 usable cycles without ovulation. So in the app,
temperature is shown as **context only** and never raises a flag on its own.

![Overnight temperature in the app](images/app-watch.png)

### 2. Model 2: PCOS risk profile from self-reported answers

We trained Model 2 on the Kaggle data using only inputs a person can answer without a clinic:
age, BMI, whether periods are regular, period length (days of bleeding), weight gain, excess hair
growth, skin darkening, hair loss, acne, fast food and regular exercise.

We compared four models with 5-fold cross-validation and a held-out 20% test set:

| Model | CV ROC-AUC | Test ROC-AUC | Test recall | Test precision | Test F1 |
|---|---|---|---|---|---|
| **Extra Trees** | **0.89** | **0.92** | **0.81** | 0.78 | **0.79** |
| Random Forest | 0.89 | 0.91 | 0.75 | 0.82 | 0.78 |
| Logistic Regression | 0.88 | 0.89 | 0.72 | 0.87 | 0.79 |
| XGBoost | 0.87 | 0.90 | 0.72 | 0.81 | 0.76 |

![Model comparison](images/model_benchmark.png)

We chose **Extra Trees** because it had the highest recall. For an awareness tool, missing
someone who should see a doctor is worse than an extra suggestion to check. At the 0.5
threshold it found about 8 in 10 PCOS cases in the test set.

![Threshold trade-off](images/threshold_tradeoff.png)

SHAP analysis shows what the model relies on most: **skin darkening, excess hair growth and
weight gain**, followed by fast food, acne and cycle regularity.

![SHAP feature importance](images/shap_summary.png)

### 3. The combined CycleSignal score

The score (0 to 100) is **40% cycle pattern** (from Model 1) plus **60% check-in profile** (from
Model 2), shown in four bands: Low, Some patterns to monitor, Increased, and Elevated pattern
signal. These weights and bands are our design choices, **not clinically validated**, and the app
says so.

### 4. The agent layer: explaining the score

A number alone doesn't help anyone. The agent layer does three things:

- **Decides the level of concern** using the Rotterdam idea that a diagnosis needs 2 of 3 areas:
  irregular ovulation (from the period log) and androgen-related signs (from Model 2). To avoid
  counting cycle evidence twice, the androgen flag is checked with her cycle answers set to
  "regular". Levels range from "nothing to act on" to "worth discussing with a doctor soon", and
  "see a doctor" if there has been no period for 90+ days.
- **Explains why, for her.** Per-person SHAP values show how much each of *her* answers moved her
  score, for example "skin darkening +12 points". When the training data pushes an answer the
  "wrong" way (for some users, acne lowers the score), the app greys it out and says it is a data
  quirk, not health advice.
- **Answers questions in plain language** using Claude (Anthropic). The assistant only sees the
  numbers our models produce; it is told never to diagnose, to always say what the app can't see
  (hormone levels, blood sugar, ultrasound), and to suggest a concrete next step. Below each
  answer, the app adds a short reading list from trusted sources (WHO, HSE, NHS, Jean Hailes),
  chosen from her own symptoms. Without an API key, a rule-based assistant gives the same facts
  offline.

![The assistant explaining a result](images/app-assistant.png)

## The app

We built the prototype as a Streamlit app with fictional demo users. Sarah has long, irregular
cycles and several symptoms; Aoife has regular cycles and few symptoms.

![Sarah's score](images/app-score.png)

| Demo user | Cycle history | Score | Band | Assistant's advice |
|---|---|---|---|---|
| Sarah, day 1 | Not enough yet | 66.0 | Increased pattern signal | Keep logging; mention it to your GP if it continues |
| Sarah, after 3 months | 38, 44 and 41 days | 88.0 | Elevated pattern signal | Worth discussing with a doctor soon |
| Aoife | 28, 30 and 29 days | 19.7 | Low pattern signal | Nothing to act on; a low score can't rule PCOS out |

Other features: a **what-if** view showing which answers move the score, a **doctor summary**
she can download and bring to an appointment, and a pause for cycle checks during hormonal
contraception, pregnancy or breastfeeding.

**Responsible design:** the app stores nothing, the API key lives only in server secrets, the
hosted demo is password-protected and rate-limited, and the assistant ignores attempts to change
its rules.

## The vision

Today our two models learn from **different people in different countries**. With the right data,
from wearable, symptoms and confirmed diagnosis for the *same* women over 6 to 12 months,
CycleSignal could show one person's full story, confirm ovulation with a validated detector,
give a calibrated likelihood with an uncertainty range, accept lab results, and share a summary
with her doctor.

![The vision](images/app-vision.png)

## Limitations

- The cycle data (Canada) and clinical data (India) come from different people, so the combined
  score has **not been validated** on anyone.
- Model 2 learned from one hospital network in Kerala, India, and may not generalise to other
  populations.
- No mcPHASES participant met the irregular-cycle criteria, so that flag could not be tested on
  real PCOS cycles.
- Wrist-temperature ovulation detection did not pass our checks.
- CycleSignal is a student prototype, not a medical device.

## Conclusion

CycleSignal showed us that useful health AI is as much about **honesty and explanation** as
about accuracy. Our best model reached a test ROC-AUC of 0.92, but the most important design work
was deciding what *not* to claim: keeping temperature as context, flagging data quirks, never
diagnosing, and always pointing to a GP. We learned to combine wearable signal processing,
tabular machine learning, explainability (SHAP) and a guarded LLM assistant into one product,
and to be clear about what would need to change before anyone relies on it.

## References

1. World Health Organization. [Polycystic ovary syndrome: fact sheet](https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome).
2. Lin B, Li JY, Kalani K, Truong K, Mariakakis A. [mcPHASES: a dataset of physiological, hormonal, and self-reported events and symptoms for menstrual health tracking with wearables](https://doi.org/10.13026/zx6a-2c81) (v1.0.0). PhysioNet, 2025.
3. Kottarathil P. [Polycystic ovary syndrome (PCOS) dataset](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data). Kaggle.
4. Teede HJ et al. [Recommendations from the 2023 International Evidence-based Guideline for the Assessment and Management of Polycystic Ovary Syndrome](https://doi.org/10.1210/clinem/dgad463). J Clin Endocrinol Metab, 2023.
5. Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group. [Revised 2003 consensus on diagnostic criteria and long-term health risks related to PCOS](https://doi.org/10.1016/j.fertnstert.2003.10.004). Fertil Steril, 2004.
6. HSE Ireland. [Polyendocrine metabolic ovarian syndrome (PMOS)](https://www2.hse.ie/conditions/polycystic-ovary-syndrome/).
7. Lundberg SM, Lee S-I. [A unified approach to interpreting model predictions (SHAP)](https://arxiv.org/abs/1705.07874). NeurIPS, 2017.
