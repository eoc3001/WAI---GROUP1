"""CycleSignal look and feel, matching the presentation slides.

Colours and the DM Sans font live in app/.streamlit/config.toml. This adds what the theme
can't: the blush-to-lilac gradient with a faint logo watermark, the sidebar logo, white
cards, and the slide-style header and footer.
All HTML here is fixed text written by us, never user input.
"""

from pathlib import Path
from urllib.parse import quote

import streamlit as st

LOGO_DIR = Path(__file__).resolve().parents[1] / "assets" / "logo"
ICON_PNG = LOGO_DIR / "png" / "1-cycle-pulse-icon.png"
LOGO_PNG = LOGO_DIR / "png" / "1-cycle-pulse-logo.png"


def _watermark() -> str:
    """The cycle-pulse icon as a faint CSS background (7% opacity), so it never competes with content."""
    svg = (LOGO_DIR / "1-cycle-pulse-icon.svg").read_text()
    svg = svg.replace('height="256">', 'height="256"><g opacity="0.07">').replace("</svg>", "</g></svg>")
    return "data:image/svg+xml;utf8," + quote(svg.strip())


CSS = """
<style>
.stApp {
  background: url("WATERMARK") right -90px bottom -90px / 480px 480px no-repeat fixed,
              linear-gradient(135deg, #fde6ee 0%, #f9f0f8 50%, #efe6f9 100%) fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMetric"], [data-testid="stChatMessage"], [data-testid="stExpander"],
[data-testid="stForm"] {
  background: #ffffff; border: 1px solid #f0dce7; border-radius: 16px; padding: 14px 18px;
  box-shadow: 0 2px 10px rgba(168, 51, 106, 0.05);
}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
  background: #ffffff; border: 1px solid #f0dce7; border-radius: 16px; padding: 10px; overflow: hidden;
}
[data-testid="stMetricLabel"] p { color: #a8336a; font-weight: 700; }
.stTabs [data-baseweb="tab"] p { font-weight: 500; }
.cs-hero { padding: 6px 0 18px 0; }
.cs-eyebrow { color: #a8336a; font-weight: 700; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
.cs-title { color: #34203a; font-weight: 700; font-size: 52px; line-height: 1.05; margin: 6px 0 4px 0; }
.cs-tagline { color: #5a4a5c; font-size: 19px; max-width: 560px; }
.cs-callout { color: #6d5899; font-weight: 700; font-size: 17px; margin: 6px 0 10px 0; }
.cs-footer { color: #6f5f72; font-size: 12px; margin-top: 28px; }
</style>
"""


def apply() -> None:
    st.markdown(CSS.replace("WATERMARK", _watermark()), unsafe_allow_html=True)
    st.logo(str(LOGO_PNG), icon_image=str(ICON_PNG), size="large")


def hero(eyebrow: str, title: str, tagline: str) -> None:
    st.markdown(f"<div class='cs-hero'><div class='cs-eyebrow'>{eyebrow}</div>"
                f"<div class='cs-title'>{title}</div><div class='cs-tagline'>{tagline}</div></div>",
                unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f"<div class='cs-callout'>{text}</div>", unsafe_allow_html=True)


def footer() -> None:
    st.markdown("<div class='cs-footer'>CycleSignal · user demo · Data: mcPHASES (PhysioNet) and Kaggle "
                "clinical data. Estimates for information, not a diagnosis.</div>", unsafe_allow_html=True)
