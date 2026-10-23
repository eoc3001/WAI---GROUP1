"""CycleSignal demo app.

Run from the repository root:
    pip install -r requirements.txt anthropic
    streamlit run app/cyclesignal_app.py

Two modes:
- Prototype (today): the real Model 2, Model 1 cycle rules, the project's combined score,
  per-person SHAP explanations and the agent. Wearable nights are simulated in the shape
  of mcPHASES, because per-person mcPHASES rows can't be shipped with the app.
- Vision (concept): illustrative screens for what CycleSignal would do with same-person,
  clinically validated data. Every number there is made up and labelled as such.

The agent uses Claude when ANTHROPIC_API_KEY (or another Anthropic credential) is set,
otherwise a rule-based explainer built from the same facts.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import brand  # noqa: E402
import resources  # noqa: E402
import security  # noqa: E402
from agent import OfflineAgent, make_agent  # noqa: E402
from cyclesignal_core import (  # noqa: E402
    BINARY_FEATURES, FEATURES, IRREGULAR, LABELS, REGULAR, SymptomExplainer, answer_text, assess,
    facts_for_agent, load_model,
)

st.set_page_config(page_title="CycleSignal", page_icon=str(brand.ICON_PNG), layout="wide")
brand.apply()
security.require_password()

# Brand palette (matches the CycleSignal slides)
INK, RASPBERRY, PURPLE, ROSE, GREY = "#34203a", "#a8336a", "#6d5899", "#d0658f", "#9a8a9c"
AVATARS = {"user": ":material/person:", "assistant": ":material/favorite:"}
BAND_COLOURS = {"Low pattern signal": PURPLE, "Some patterns to monitor": "#a07cc5",
                "Increased pattern signal": ROSE, "Elevated pattern signal": RASPBERRY}
LEVEL_TEXT = {"S": "See a doctor", "L2": "Worth discussing with a doctor soon",
              "L1": "Worth keeping an eye on", "INFO": "Note only", "L0": "Nothing to act on"}

# Demo personas (fictional). temp_missing = cycles with no clear temperature rise.
PERSONAS = {
    "Sarah, day 1": dict(Age=26, BMI=28.4, sym={"Weight gain": 1, "hair growth": 1, "Skin darkening": 1,
                         "Hair loss": 0, "Pimples": 1, "Fast food": 1, "Reg.Exercise": 0},
                         cyc=IRREGULAR, plen=4, lengths="", days=12, ovulated=[]),
    "Sarah, after 3 months": dict(Age=26, BMI=28.4, sym={"Weight gain": 1, "hair growth": 1, "Skin darkening": 1,
                                  "Hair loss": 0, "Pimples": 1, "Fast food": 1, "Reg.Exercise": 0},
                                  cyc=IRREGULAR, plen=4, lengths="38, 44, 41", days=12, ovulated=[True, False, False]),
    "Aoife (contrast)": dict(Age=27, BMI=22.0, sym={"Weight gain": 0, "hair growth": 0, "Skin darkening": 0,
                             "Hair loss": 0, "Pimples": 1, "Fast food": 0, "Reg.Exercise": 1},
                             cyc=REGULAR, plen=5, lengths="28, 30, 29", days=10, ovulated=[True, True, True]),
}
BINARY = list(BINARY_FEATURES)
REGULARITY = {"Regular": REGULAR, "Irregular": IRREGULAR}


@st.cache_resource(show_spinner="Training Model 2 on its training split...")
def get_model():
    model, metrics = load_model()
    return model, SymptomExplainer(model), metrics


def load_persona():
    p = PERSONAS[st.session_state.persona]
    st.session_state.Age, st.session_state.BMI = p["Age"], p["BMI"]
    for f in BINARY:
        st.session_state[f"sym_{f}"] = bool(p["sym"][f])
    st.session_state.reg = "Irregular" if p["cyc"] == IRREGULAR else "Regular"
    st.session_state.plen = p["plen"]
    st.session_state.lengths = p["lengths"]
    st.session_state.days = p["days"]
    st.session_state.paused = False
    st.session_state.chat = []


def chart(c) -> None:
    st.altair_chart(c.properties(padding={"left": 4, "right": 22, "top": 6, "bottom": 4})
                    .configure(background="#ffffff"), width="stretch")


def parse_lengths(text: str) -> list[int]:
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit() and 10 <= int(part) <= 200:
            out.append(int(part))
    return out


def simulate_nights(lengths: list[int], ovulated: list[bool], seed: int = 7) -> pd.DataFrame:
    """Nightly wrist temperature and resting HR in the shape of mcPHASES (simulated)."""
    rng = np.random.default_rng(seed)
    rows, day = [], 0
    for i, L in enumerate(lengths):
        ov = ovulated[i] if i < len(ovulated) else True
        ov_day = L - 14
        for d in range(L):
            shift = 0.3 if ov and d > ov_day + 1 else 0.0
            rows.append({"night": day, "cycle": i + 1, "cycle_day": d + 1, "period": d < 5,
                         "temp_dev": round(shift + rng.normal(0, 0.08), 2),
                         "rhr": round(62 + (1.5 if shift else 0) + rng.normal(0, 1.2), 1)})
            day += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sidebar
mode = st.sidebar.radio("Mode", ["Prototype (today)", "Vision (concept)"],
                        help="Prototype runs the real models. Vision shows illustrative data only.")

if "persona" not in st.session_state:
    st.session_state.persona = "Sarah, after 3 months"
    load_persona()


def prototype():
    st.sidebar.selectbox("Demo user", list(PERSONAS), key="persona", on_change=load_persona)
    allowed, why_not = security.claude_allowed()
    use_claude = st.sidebar.toggle("Agent uses Claude", value=allowed, disabled=not allowed,
                                   help="Needs ANTHROPIC_API_KEY, and APP_PASSWORD when hosted. "
                                        "Off = rule-based explainer from the same facts.")
    st.sidebar.caption("Agent: " + ("Claude (claude-opus-5-5)" if use_claude
                                     else "offline explainer" + (f" ({why_not})" if why_not else "")))

    model, explainer, metrics = get_model()
    p = PERSONAS[st.session_state.persona]

    brand.hero("User demo", "CycleSignal", "AI powered cycle tracking for earlier PCOS risk awareness")
    brand.callout("Our idea: help someone notice the pattern and take it to a GP.")

    tabs = st.tabs(["1 · Watch", "2 · Log", "3 · Score", "4 · Assistant", "5 · What-if", "6 · Doctor summary"])

    # ----- 1. watch
    with tabs[0]:
        lengths = parse_lengths(st.session_state.lengths)
        st.subheader("What the watch records overnight")
        if not lengths:
            st.info("No complete cycles yet. The watch is recording; cycle patterns appear after your first periods are logged.")
        else:
            nights = simulate_nights(lengths, p["ovulated"])
            base = alt.Chart(nights).encode(x=alt.X("night:Q", title="Night", scale=alt.Scale(domain=[0, len(nights)], nice=False)))
            period = base.mark_bar(color=RASPBERRY, opacity=0.18).encode(
                y=alt.Y("pmax:Q", title=None, axis=None)).transform_filter("datum.period").transform_calculate(pmax="0.6")
            temp = base.mark_line(color=PURPLE, point=alt.OverlayMarkDef(size=12)).encode(
                y=alt.Y("temp_dev:Q", title="Skin temperature vs your baseline (°C)", scale=alt.Scale(domain=[-0.3, 0.6])),
                tooltip=["cycle", "cycle_day", "temp_dev", "rhr"])
            chart((period + temp).resolve_scale(y="independent").properties(height=320))
            c1, c2, c3 = st.columns(3)
            c1.metric("Nights recorded", len(nights))
            c2.metric("Cycles", len(lengths))
            missing = sum(1 for o in p["ovulated"][:len(lengths)] if not o)
            c3.metric("No clear temperature rise", f"{missing} of {len(lengths)} cycles")
            st.caption("Pink bars = logged period days. Model 1 uses overnight temperature to estimate cycle phase. "
                       "Its ovulation check did not pass validation in our data, so temperature is shown as context "
                       "and is not part of the score. Simulated nights in the shape of the mcPHASES dataset.")

    # ----- 2. log
    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.subheader("Period log")
            st.text_input("Completed cycle lengths (days, oldest first)", key="lengths",
                          help="From the period days you tap in the diary.")
            st.number_input("Days since your last period started", 0, 400, key="days")
            st.checkbox("I'm on hormonal contraception, pregnant or breastfeeding", key="paused")
            st.caption("Privacy: CycleSignal stores nothing. Your answers live only in this browser session. "
                       "If the assistant uses Claude, your score and answers (never your name) are sent to "
                       "Anthropic to write the explanation.")
        with right:
            st.subheader("Monthly check-in")
            a, b = st.columns(2)
            a.number_input("Age", 15, 55, key="Age")
            b.number_input("BMI", 14.0, 60.0, step=0.1, key="BMI")
            c, e = st.columns(2)
            c.radio("Are your periods usually regular?", list(REGULARITY), key="reg", horizontal=True)
            e.number_input("How many days does your period last?", 0, 15, key="plen",
                           help="Days of bleeding, not the whole cycle.")
            for f in BINARY:
                st.toggle(LABELS[f], key=f"sym_{f}")

    features = {"Age": float(st.session_state.Age), "BMI": float(st.session_state.BMI),
                "Cycle(R/I)": REGULARITY[st.session_state.reg], "Cycle length(days)": float(st.session_state.plen)}
    features |= {f: int(st.session_state[f"sym_{f}"]) for f in BINARY}
    lengths = parse_lengths(st.session_state.lengths)
    temp_missing = sum(1 for o in p["ovulated"][:len(lengths)] if not o) or None
    reading = assess(model, explainer, features, lengths, st.session_state.days,
                     st.session_state.paused, temp_missing)
    facts = facts_for_agent(reading)
    score = reading["score"]

    # ----- 3. score
    with tabs[2]:
        c1, c2 = st.columns([1, 2])
        with c1:
            colour = BAND_COLOURS[score["band"]]
            # HTML built only from app values (fixed band names and numbers), never user text
            st.markdown(f"<div style='font-size:64px;font-weight:700;color:{colour};line-height:1'>{score['score']}"
                        f"<span style='font-size:22px;color:{GREY}'> / 100</span></div>"
                        f"<div style='font-size:20px;color:{colour};margin-bottom:8px'>{score['band']}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"**Assistant level:** {LEVEL_TEXT[reading['level']]}")
            if score["history_note"]:
                st.warning("Low confidence: not enough cycle history yet. " + score["history_note"])
            if score["cycle_follow_up"] or reading["level"] == "S":
                st.error("No period logged for 90+ days. Please see a doctor, whatever the score.")
        with c2:
            split = pd.DataFrame({"part": ["Cycle pattern (40%)", "Check-in answers (60%)"], "order": [0, 1],
                                  "points": [reading["cycle_points"], reading["symptom_points"]],
                                  "detail": [f"Model 1 cycle score {score['model1_cycle_score']:.0f}/100",
                                             f"Model 2 profile score {score['model2_symptom_score']:.1f}/100"]})
            chart(alt.Chart(split).mark_bar(size=34).encode(
                x=alt.X("points:Q", stack="zero", scale=alt.Scale(domain=[0, 100]), title="Points in your score"),
                color=alt.Color("part:N", sort=list(split["part"]),
                                scale=alt.Scale(domain=list(split["part"]), range=[RASPBERRY, PURPLE]),
                                legend=alt.Legend(orient="bottom", title=None)),
                order=alt.Order("order:Q"),
                tooltip=["part", "points", "detail"]).properties(height=150))
        st.subheader("What moved your check-in score")
        st.caption(f"Each bar is how much one answer moved your check-in score from the model's average of "
                   f"{reading['symptom_base_points']:.0f}, for you specifically (SHAP). Faded bars = training-data quirk, not advice.")
        d = reading["drivers"].copy()
        d["direction"] = np.where(d["symptom_points"] >= 0, "raised", "lowered")
        d["answer"] = [answer_text(f, v) for f, v in zip(d.feature, d.value)]
        d["name"] = d["label"] + " (" + d["answer"] + ")"
        bars = alt.Chart(d).mark_bar().encode(
            y=alt.Y("name:N", sort=list(d["name"]), title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
            x=alt.X("symptom_points:Q", title="Effect on check-in score (points)"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raised", "lowered"], range=[ROSE, PURPLE]), legend=None),
            opacity=alt.condition("datum.counterintuitive", alt.value(0.35), alt.value(1)),
            tooltip=["label", "answer", alt.Tooltip("symptom_points:Q", format=".1f"), "counterintuitive"])
        chart(bars.properties(height=300))
        st.caption(score["warning"])

    # ----- 4. assistant
    with tabs[3]:
        key = (st.session_state.persona, str(facts), use_claude)
        if st.session_state.get("agent_key") != key:
            st.session_state.agent_key = key
            st.session_state.agent = make_agent(facts, security.get_secret("ANTHROPIC_API_KEY") if use_claude else None)
            st.session_state.offline = OfflineAgent(facts)
            st.session_state.chat = []

        def pick_agent():
            """Claude while within the rate limits, otherwise the offline explainer."""
            if not use_claude:
                return st.session_state.agent
            ok, message = security.take_claude_call()
            if not ok:
                st.info(message)
                return st.session_state.offline
            return st.session_state.agent

        def answer(chunks, reading_list):
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                text = security.clean_answer(st.write_stream(chunks))
            return text + ("\n\n" + resources.as_markdown(reading_list) if reading_list else "")

        for role, text in st.session_state.chat:
            st.chat_message(role, avatar=AVATARS[role]).markdown(text)
        if not st.session_state.chat:
            if st.button("Explain my result", type="primary"):
                text = answer(pick_agent().stream(), resources.for_reading(reading))
                st.session_state.chat.append(("assistant", text))
                st.rerun()
        else:
            st.caption("Try asking:")
            cols = st.columns(4)
            quick = ["Is it my weight gain?", "Is it the skin darkening?", "Does my temperature matter?", "Is it my cycle length?"]
            asked = None
            for col, q in zip(cols, quick):
                if col.button(q):
                    asked = q
            typed = st.chat_input("Ask about your result", max_chars=security.MAX_QUESTION_CHARS)
            question = security.clean_question(typed or asked or "")
            if question:
                st.chat_message("user", avatar=AVATARS["user"]).markdown(question)
                text = answer(pick_agent().ask(question), resources.for_question(question))
                st.session_state.chat += [("user", question), ("assistant", text)]
                st.rerun()
        with st.expander("Learn more about PCOS / PMOS"):
            st.caption("PCOS is now called PMOS (polyendocrine metabolic ovarian syndrome) by the NHS and HSE. "
                       "It is the same condition.")
            st.markdown(resources.as_markdown(list(resources.RESOURCES.values())))

    # ----- 5. what-if
    with tabs[4]:
        st.subheader("What if something changed?")
        st.caption("See which of your inputs move the result. This shows how the model behaves, not a treatment plan.")
        a, b = st.columns(2)
        changes = {}
        for i, f in enumerate(BINARY):
            col = a if i % 2 == 0 else b
            new = col.toggle(LABELS[f], value=bool(features[f]), key=f"wi_{f}")
            changes[f] = int(new)
        wi = assess(model, explainer, features | changes, lengths, st.session_state.days,
                    st.session_state.paused, temp_missing)
        c1, c2, c3 = st.columns(3)
        c1.metric("Your score now", score["score"])
        c2.metric("What-if score", wi["score"]["score"], round(wi["score"]["score"] - score["score"], 1), delta_color="inverse")
        c3.markdown(f"What-if band  \n**<span style='color:{BAND_COLOURS[wi['score']['band']]};font-size:22px'>"
                    f"{wi['score']['band']}</span>**", unsafe_allow_html=True)
        if any(wi["drivers"]["counterintuitive"] & (wi["drivers"]["symptom_points"].abs() >= 1)):
            st.info("Some inputs move the score in a direction that reflects the training data, not medical knowledge. "
                    "Don't read those as advice.")

    # ----- 6. doctor summary
    with tabs[5]:
        top = reading["drivers"][(reading["drivers"].symptom_points >= 1) & ~reading["drivers"].counterintuitive].head(4)
        summary = "\n".join([
            "# CycleSignal summary for your doctor", f"Date: {date.today():%d %B %Y}", "",
            f"**Risk-awareness score:** {score['score']}/100 ({score['band']}). Not a diagnosis or probability of disease.",
            f"**Assistant level:** {LEVEL_TEXT[reading['level']]}", "",
            "## Cycles (from period log)",
            f"- Completed cycle lengths (days): {', '.join(map(str, lengths)) or 'none yet'}",
            f"- Regularity (FIGO): {reading['flags']['regularity']}; outside 21-35 days in last 3: {reading['flags']['outside_21_35_in_last_3']}",
            f"- Days since last period started: {st.session_state.days}",
            f"- Cycles without a clear wrist-temperature rise: {temp_missing or 0} (unvalidated, context only)", "",
            "## Self-reported (monthly check-in)",
            *[f"- {LABELS[f]}: {answer_text(f, features[f])}" for f in FEATURES], "",
            "## Largest contributors for this person",
            *[f"- {r.label}: +{r.symptom_points:.1f} check-in points" for r in top.itertuples()], "",
            "## Not measured by the app",
            "- Androgen levels, LH/FSH, AMH, glucose/insulin, thyroid, prolactin, ovarian ultrasound", "",
            f"_{score['warning']}_",
        ])
        st.markdown(summary)
        st.download_button("Download summary", summary, file_name="cyclesignal_summary.md", type="primary")

    with st.sidebar.expander("Model facts"):
        st.write(f"Model 2 (Extra Trees, 9 self-reportable inputs) held-out test, n={metrics['test_rows']}:")
        st.write(f"ROC-AUC {metrics['roc_auc']:.2f} · recall {metrics['recall']:.2f} · precision {metrics['precision']:.2f}")
        st.caption("Combined score weights (40/60) and bands are project heuristics, not clinically validated.")
    brand.footer()


def vision():
    brand.hero("The vision", "CycleSignal", "What it could do with the right data, from the right people")
    st.warning("**Concept, illustrative data only.** These screens show what CycleSignal could do once it is "
               "trained and validated on wearable, symptom and diagnosis data from the same people. "
               "No number on this page comes from a real model.")
    months = pd.DataFrame({
        "month": pd.date_range("2027-01-01", periods=9, freq="MS"),
        "cycle_length": [33, 39, 44, 41, 47, 43, 52, 45, 49],
        "ovulation": ["Confirmed", "Confirmed", "Not detected", "Not detected", "Not detected",
                      "Confirmed", "Not detected", "Not detected", "Not detected"],
        "score": [31, 38, 52, 58, 63, 61, 70, 72, 74],
        "low": [22, 29, 44, 51, 56, 54, 63, 66, 68], "high": [40, 47, 60, 65, 70, 68, 77, 78, 80],
    })
    months["m"] = months["month"].dt.strftime("%b %Y")
    order = list(months["m"])
    t1, t2, t3, t4 = st.tabs(["Timeline", "Rotterdam tracker", "Labs & scan", "Clinician view"])
    with t1:
        st.subheader("Nine months of Sarah's data, one person, end to end")
        band = alt.Chart(months).mark_area(color=RASPBERRY, opacity=0.18).encode(x=alt.X("m:O", sort=order, title=None, axis=alt.Axis(labelAngle=0)), y="low:Q", y2="high:Q")
        line = alt.Chart(months).mark_line(color=RASPBERRY, point=True).encode(
            x=alt.X("m:O", sort=order), y=alt.Y("score:Q", title="Calibrated likelihood (%) with 90% interval", scale=alt.Scale(domain=[0, 100])))
        chart(band + line)
        chart(alt.Chart(months).mark_bar(size=22).encode(
            x=alt.X("m:O", sort=order, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("cycle_length:Q", title="Cycle length (days)"),
            color=alt.Color("ovulation:N", scale=alt.Scale(domain=["Confirmed", "Not detected"], range=[PURPLE, ROSE]),
                            legend=alt.Legend(orient="bottom", title="Ovulation")))
            .properties(height=220) + alt.Chart(pd.DataFrame({"y": [21, 35]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="y:Q"))
        st.caption("Dashed lines = 21 and 35 days. Ovulation from a validated watch detector plus optional LH strips. Vision: a validated ovulation detector turns temperature into real evidence, and the score becomes a "
                   "calibrated likelihood with an honest uncertainty interval.")
    with t2:
        st.subheader("Rotterdam criteria: 2 of 3 needed for a diagnosis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ovulation", "Evidence present", "6 of 9 cycles without ovulation", delta_color="off")
        c2.metric("Androgen signs", "Evidence present", "hair growth, acne, testosterone ↑", delta_color="off")
        c3.metric("Ovaries", "Unknown", "no scan yet", delta_color="off")
        st.success("Two of three areas show evidence. The assistant recommends a GP visit and has prepared a referral summary.")
    with t3:
        st.subheader("Optional: add test results for a more confident reading")
        st.dataframe(pd.DataFrame({
            "Test": ["Total testosterone", "LH : FSH ratio", "AMH", "HbA1c", "TSH (thyroid)", "Pelvic ultrasound"],
            "Result": ["2.9 nmol/L", "2.6", "8.1 ng/mL", "39 mmol/mol", "1.8 mU/L", "not done"],
            "Source": ["GP letter (photo)", "GP letter", "Clinic portal", "GP letter", "GP letter", "-"],
        }), hide_index=True, width="stretch")
        st.caption("Illustrative values. In the vision, labs are read from a photo or health record with the user's consent.")
    with t4:
        st.subheader("Shared with Dr. Murphy (with Sarah's consent)")
        st.markdown("- 9-month cycle chart and ovulation evidence\n- Symptom history with dates\n"
                    "- Score trend with uncertainty\n- Suggested workup: androgens, LH/FSH, AMH, glucose/insulin, "
                    "thyroid, prolactin, pelvic ultrasound\n- Clinician feedback (diagnosis yes/no) returns to the model "
                    "as a label, with consent, so it keeps improving")
    st.divider()
    st.subheader("What has to be true to build this")
    st.markdown("1. Wearable + symptoms + confirmed diagnosis **from the same people**, followed for 6-12 months\n"
                "2. A diverse cohort (today's check-in model learned from one hospital group in Kerala, India)\n"
                "3. An ovulation detector that passes validation against hormone tests\n"
                "4. Calibration and a clinical study before any likelihood is shown\n"
                "5. Medical-device regulatory review (EU MDR) and GDPR-grade handling of health data")
    brand.footer()


if mode == "Prototype (today)":
    prototype()
else:
    vision()
