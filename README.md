# WAI---GROUP001

 PCOS is currently detected late, often years after symptoms begin, and only once a cycle has been visibly irregular for a while or a lab test flags rising androgens or glucose. Up to 70% of women with PCOS remain undiagnosed. No widely available tool combines everyday wearable data with known clinical risk patterns to flag a possible PCOS-risk profile earlier, before someone even seeks a diagnosis.
 
Problem statement
 Many women have no ongoing way to know their cycle patterns look concerning until symptoms become disruptive enough to prompt a doctor's visit (irregular cycles, weight changes, acne, excess hair growth), at which point diagnosis can still take years, given how diffuse PCOS symptoms are. PCOS risk is currently identified reactively: at a scheduled appointment, after a lab test, or once symptoms are already affecting daily life, rather than through continuous, everyday tracking that could prompt someone to seek clinical evaluation earlier. No existing consumer tool combines passive wearable tracking (cycle phase, regularity) with known clinical PCOS-risk patterns to flag this earlier.
 
What is PCOS
 PCOS (polycystic ovary syndrome) is a common hormonal condition affecting an estimated 1 in 10 women of reproductive age, marked by some combination of irregular ovulation, elevated androgen levels, and polycystic ovaries. It's linked to insulin resistance, weight changes, fertility challenges, and increased long-term risk of type 2 diabetes and cardiovascular disease, but its symptoms are diffuse enough that diagnosis often takes years, and a majority of women with PCOS remain undiagnosed.
 
Data: 
mcPHASES (PhysioNet) for wearable and hormone-verified cycle data
https://physionet.org/content/mcphases/1.0.0/
PCOS dataset (Kaggle, 10 hospitals in Kerala, India) for clinical PCOS profiles
https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data

Current Approach::
Model 1 is trained on real wearable data (heart rate, temperature, sleep) paired with hormone-verified cycle labels, and automatically tracks which cycle phase someone is in day to day.
Model 2 is trained on real clinical PCOS patient data to learn what a PCOS-risk profile tends to look like, using features a person can realistically self-report (cycle length, weight/BMI, symptoms like hair growth or acne) rather than lab-only measurements.
The connection: Model 1's daily tracking is used to calculate an actual cycle-length pattern over time, which becomes one of the inputs to Model 2, alongside self-reported weight/symptoms, and optionally lab data if the user has it (for a more confident reading).
The output for the user: an indication of where they are in their cycle, plus a flag if their pattern suggests it's worth watching for PCOS symptoms and consulting a doctor.

## Links

- **Project blog:** https://eoc3001.github.io/WAI---GROUP1/
- **Demo app:** https://cyclesignal-wai.streamlit.app/ (password on request)

## Repository structure

| Folder | What's inside |
|---|---|
| [`model1/`](model1/) | **Model 1:** wearable cycle tracking from mcPHASES (`build_daily_table.py`, `model1_pipeline_v6.py`) and its summary results |
| [`model2/`](model2/) | **Model 2:** PCOS risk profile: input data, four-model benchmark (`build_pcos_model2.py`), metrics and figures |
| [`combined_score/`](combined_score/) | **Model 1 + Model 2:** the 0-100 CycleSignal risk-awareness score (`cyclesignal_risk_score.py`) |
| [`agent/`](agent/) | **Agent layer:** escalation level, the assistant (Claude or offline) and the trusted reading list |
| [`app/`](app/) | **Demo app:** Streamlit prototype bringing everything together ([how to run](app/README.md)) |
| [`docs/`](docs/) | **Project blog** (GitHub Pages) |
| [`assets/`](assets/) | Logo files |

### Run each part (from the repository root)

```bash
pip install -r requirements.txt
python model2/build_pcos_model2.py                      # Model 2: retrain and save metrics/figures
python combined_score/cyclesignal_risk_score.py         # Model 1 + Model 2: evaluate and print demo scores
streamlit run app/cyclesignal_app.py                    # Demo app
```
Model 1 needs the mcPHASES files on your laptop in `data/` (never committed); see [model1/README.md](model1/README.md).
