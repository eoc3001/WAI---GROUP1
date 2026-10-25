"""Privacy notice and terms of use, shown from the links at the bottom of the app.

Plain-language text written by the team for a project prototype. It has not been
reviewed by a lawyer. Keep it in line with what the app actually does with data
(see app/security.py and agent/assistant.py).
"""

import streamlit as st

UPDATED = "25 October 2026"
CONTACT = "https://github.com/eoc3001/WAI---GROUP1"

PRIVACY = f"""
*Last updated: {UPDATED}*

CycleSignal is a prototype built for a Women in AI Ireland project. This notice explains
what happens to the information you enter.

**This is a prototype.** Please use the demo users (Sarah and Aoife) and don't enter your own
health information.

**What we store.** Nothing. CycleSignal has no accounts and no database. Your answers stay in
your browser tab and are gone when you close or refresh it.

**The assistant.** When the assistant uses Claude, your score and your answers (never your name)
are sent to Anthropic to write the explanation. Anthropic handles that data under its own
[privacy policy](https://www.anthropic.com/legal/privacy). If the assistant runs offline, nothing
leaves the app.

**Hosting.** The app runs on Streamlit Community Cloud. Like most websites, the hosting service may
keep standard technical logs, such as your IP address and the time of your visit.

**Downloads.** The doctor summary is created in your browser session and saved only where you
choose to save it.

**Questions.** Contact the team through the [project repository]({CONTACT}).
"""

TERMS = f"""
*Last updated: {UPDATED}*

By using CycleSignal you agree to the following.

**Not medical advice.** CycleSignal gives a risk-awareness score for information and education
only. It is not a diagnosis, it is not a medical device, and it does not replace a doctor. A low
score does not rule out PCOS. Talk to your GP about any symptoms or changes to your cycle.

**In an emergency,** call 112 or 999 (Ireland), or your local emergency number.

**A prototype.** The models were trained on public research datasets and have not been clinically
validated. Results can be wrong. The app is provided as it is, without any warranty, and the team
is not responsible for decisions made using it.

**Fair use.** Please don't try to misuse the app, get around its limits, or use the assistant for
anything other than understanding your result. Access to the hosted demo can be withdrawn at any
time.

**Third-party services.** The assistant uses Claude by Anthropic, and the app is hosted on
Streamlit Community Cloud. Their own terms apply to those services.

**Questions.** Contact the team through the [project repository]({CONTACT}).
"""


@st.dialog("Privacy notice", width="large")
def _privacy():
    st.markdown(PRIVACY)


@st.dialog("Terms of use", width="large")
def _terms():
    st.markdown(TERMS)


def links(key: str) -> None:
    """Small Privacy and Terms links for the bottom of a page. `key` keeps the buttons unique per page."""
    left, right, _ = st.columns([1, 1, 8])
    if left.button("Privacy", type="tertiary", key=f"privacy_{key}"):
        _privacy()
    if right.button("Terms of use", type="tertiary", key=f"terms_{key}"):
        _terms()
