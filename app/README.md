# CycleSignal demo app

A Streamlit app that runs the real CycleSignal pipeline end to end: period log and symptoms in,
combined risk-awareness score out, with per-person explanations and an assistant that explains
the result.

## Run

From the repository root:

    pip install -r requirements.txt
    streamlit run app/cyclesignal_app.py

The first load trains Model 2 on its training split (about 20 seconds), then it is cached.

### Optional: Claude as the assistant

Without credentials the assistant is a rule-based explainer that uses the same facts.
To use Claude (`claude-opus-5-5`), set an API key before starting the app:

    set ANTHROPIC_API_KEY=sk-ant-...        # Windows (cmd)
    $env:ANTHROPIC_API_KEY="sk-ant-..."     # Windows (PowerShell)
    export ANTHROPIC_API_KEY=sk-ant-...     # Mac / Linux

Then switch on **Agent uses Claude** in the sidebar.

## Keeping the API key and the app safe

- The key is read only from `app/.streamlit/secrets.toml` (git-ignored) or an environment
  variable. Copy `app/.streamlit/secrets.example.toml` to start. Never put it in code.
- On a hosted copy, Claude stays off unless `APP_PASSWORD` is also set, so a public
  link can't spend your credit.
- 5 wrong passwords lock that browser session for 5 minutes.
- Claude limits: 15 questions per visitor, 60 per hour in total, 3 seconds between
  questions. Over the limit, the offline assistant answers instead.
- Questions are capped at 300 characters, with HTML and control characters stripped.
  Any links in Claude's answers are removed; the reading list is added by the app from
  `app/resources.py`.
- `app/.streamlit/config.toml` hides error details from visitors and keeps cross-site
  request protection on.
- Set a monthly spend limit in the Anthropic Console as the final backstop.

## Deploy on Streamlit Community Cloud

1. share.streamlit.io → Create app → repo `eoc3001/WAI---GROUP1`, branch `main`,
   main file `app/cyclesignal_app.py`, Advanced settings → Python 3.12.
2. Secrets box: paste `ANTHROPIC_API_KEY = "..."` and `APP_PASSWORD = "..."`.
3. After deploying: Settings → Sharing → only specific people (your team and assessors).

`app/requirements.txt` is used instead of the root file (which lists `artifact-tool`,
a package that doesn't exist on PyPI).

## Files

| File | What it does |
|---|---|
| `app/cyclesignal_app.py` | The app: Prototype (today) and Vision (concept) modes |
| `app/cyclesignal_core.py` | Cycle flags (Model 1 rules), per-person SHAP, agent level, facts for the assistant |
| `app/agent.py` | The assistant: Claude or offline explainer, both limited to the facts |
| `app/resources.py` | Checked reading links (WHO, HSE, NHS, Jean Hailes, 2023 guideline) |
| `app/security.py` | Secrets, password gate, rate limits, input/output cleaning |
| `demo/VIDEO_SCRIPTS.md` | Shot-by-shot scripts for the two demo videos |

## What is real and what is not

| Part | Status |
|---|---|
| Model 2 check-in score (11 inputs, incl. self-reported regularity and period length), held-out metrics | Real (`cyclesignal_risk_score.py`) |
| 40/60 combined score and bands | Real code; weights are project heuristics, not clinically validated |
| Cycle flags (FIGO regularity, 21-35 day rule, < 8 a year, 90 days) | Real rules from `model1/model1_pipeline_v6.py` |
| Assistant level (S / L2 / L1 / INFO / L0) | Mirror of `agent_level` in Model 1 (keep in sync) |
| Per-person explanations | Real SHAP on the fitted Extra Trees model |
| Wearable nights on the Watch tab | Simulated in the shape of mcPHASES (per-person rows can't ship) |
| Personas (Sarah, Aoife) | Fictional |
| Vision mode | Illustrative only, labelled on screen |

Some answers move the score the "wrong" way (for example, acne lowers it for some users).
Those are quirks of the training data. The app fades them in the chart, and the assistant
says they are not advice.
