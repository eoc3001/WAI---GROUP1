# Agent layer

Turns a CycleSignal score into a clear, safe next step for the user.

| File | What it does |
|---|---|
| `levels.py` | How strongly to recommend a GP visit (S / L2 / L1 / INFO / L0), using the Rotterdam "2 of 3" idea. Mirrors `agent_level` in `model1/model1_pipeline_v6.py`; keep the two in sync |
| `assistant.py` | Explains the result and answers questions: Claude (`claude-opus-5-5`) with strict rules, or a rule-based offline explainer. Only sees numbers the models produce |
| `resources.py` | Trusted reading links (WHO, HSE, NHS, Jean Hailes, 2023 guideline), chosen by the app from her own answers, never by the model |

Used by the demo app in [`app/`](../app/).
