"""CycleSignal look and feel, matching the presentation slides.

Colours and the DM Sans font live in app/.streamlit/config.toml. This adds what the theme
can't: the gradient with a faint logo watermark, the sidebar logo, cards, the top bar, the
score card and the "why" list.

Every HTML string is built from app values (fixed labels, numbers, band names) and passed
through html.escape, never from free text a user typed.

Motion rules: Streamlit re-renders the page on every interaction, so nothing animates on
entry (it would replay on every click). Only press feedback and pointer-only hover move.
"""

from html import escape
from pathlib import Path
from urllib.parse import quote

import streamlit as st

LOGO_DIR = Path(__file__).resolve().parents[1] / "assets" / "logo"
ICON_PNG = LOGO_DIR / "png" / "1-cycle-pulse-icon.png"

INK, RASPBERRY, PURPLE, ROSE, MAUVE, GREY = "#34203a", "#a8336a", "#6d5899", "#d0658f", "#a07cc5", "#8a7a8c"
BANDS = [  # (upper bound, name, colour), matching calculate_cyclesignal_score
    (30, "Low pattern signal", PURPLE),
    (50, "Some patterns to monitor", MAUVE),
    (70, "Increased pattern signal", ROSE),
    (100, "Elevated pattern signal", RASPBERRY),
]
BAND_COLOURS = {name: colour for _, name, colour in BANDS}


def _watermark() -> str:
    """The cycle-pulse icon as a faint CSS background (7% opacity), so it never competes with content."""
    svg = (LOGO_DIR / "1-cycle-pulse-icon.svg").read_text()
    svg = svg.replace('height="256">', 'height="256"><g opacity="0.07">').replace("</svg>", "</g></svg>")
    return "data:image/svg+xml;utf8," + quote(svg.strip())


CSS = """
<style>
:root {
  --ink: #34203a; --muted: #6f5f72; --raspberry: #a8336a; --purple: #6d5899;
  --line: #f0dce7; --card: #ffffff; --soft: #f8eef6;
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
}
.stApp {
  background: url("WATERMARK") right -90px bottom -90px / 480px 480px no-repeat fixed,
              linear-gradient(135deg, #fde6ee 0%, #f9f0f8 50%, #efe6f9 100%) fixed;
  -webkit-tap-highlight-color: transparent;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* Cards */
[data-testid="stMetric"], [data-testid="stChatMessage"], [data-testid="stExpander"], [data-testid="stForm"],
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  box-shadow: 0 1px 2px rgba(52, 32, 58, 0.04), 0 6px 20px rgba(168, 51, 106, 0.05);
}
[data-testid="stMetric"], [data-testid="stChatMessage"], [data-testid="stForm"] { padding: 14px 18px; }
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] { padding: 10px; overflow: hidden; }
[data-testid="stMetricLabel"] p { color: var(--raspberry); font-weight: 700; }

/* Step navigation: tabs as a pill segmented control (react-aria markup) */
.stTabs [role="tablist"] {
  gap: 4px; background: rgba(255, 255, 255, 0.6); border: 1px solid var(--line) !important;
  border-radius: 999px; padding: 4px; width: fit-content; max-width: 100%; overflow-x: auto;
  box-shadow: none !important; margin-bottom: 14px; scrollbar-width: none;
}
.stTabs [data-testid="stTab"] {
  border-radius: 999px; padding: 7px 15px; height: auto; background: transparent; border: 0 !important;
  transition: background-color 160ms ease, transform 160ms var(--ease-out);
}
.stTabs [data-testid="stTab"]::before, .stTabs [data-testid="stTab"]::after { display: none !important; }
.stTabs [data-testid="stTab"] p { font-weight: 500; font-size: 14px; color: var(--muted); }
.stTabs [data-testid="stTab"][aria-selected="true"] { background: #fff; box-shadow: 0 1px 3px rgba(52, 32, 58, 0.14); }
.stTabs [data-testid="stTab"][aria-selected="true"] p { color: var(--raspberry); font-weight: 700; }
.stTabs [role="tablist"] > :not([role="tab"]) { display: none; }

/* Press feedback on everything pressable; hover only for real pointers */
.stButton button, .stDownloadButton button, .stTabs [data-testid="stTab"] {
  transition: transform 160ms var(--ease-out), background-color 160ms ease, border-color 160ms ease;
}
.stButton button:active, .stDownloadButton button:active, .stTabs [data-testid="stTab"]:active { transform: scale(0.97); }
@media (hover: hover) and (pointer: fine) {
  .stButton button:hover { border-color: var(--raspberry); }
  .stTabs [data-testid="stTab"][aria-selected="false"]:hover { background: rgba(255, 255, 255, 0.7); }
}
.stButton button:focus-visible, .stTabs [data-testid="stTab"]:focus-visible { outline: 2px solid var(--raspberry); outline-offset: 2px; }

/* Phones: 16px inputs stop iOS zooming on focus */
input, textarea, [data-baseweb="select"] div { font-size: 16px !important; }

/* Top bar */
.cs-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.cs-brand { display: flex; align-items: center; gap: 10px; }
.cs-brand img { width: 34px; height: 34px; }
.cs-brand b { font-size: 22px; color: var(--ink); letter-spacing: -0.01em; }
.cs-pill { font-size: 12.5px; font-weight: 700; color: var(--raspberry); background: #fff; border: 1px solid var(--line);
  border-radius: 999px; padding: 5px 12px; white-space: nowrap; }
.cs-pill.concept { color: #8a5a00; background: #fff7e6; border-color: #f3dfb5; }
.cs-lede { color: var(--muted); font-size: 15.5px; margin: 2px 0 18px 0; max-width: 680px; }

/* Score card */
.cs-score { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 22px 24px;
  box-shadow: 0 1px 2px rgba(52, 32, 58, 0.04), 0 10px 30px rgba(168, 51, 106, 0.07);
  display: grid; grid-template-columns: minmax(220px, 280px) 1fr; gap: 28px; align-items: center; }
.cs-gauge { position: relative; text-align: center; }
.cs-gauge svg { width: 100%; height: auto; display: block; }
.cs-num { position: absolute; left: 0; right: 0; bottom: 4px; }
.cs-num b { font-size: 52px; line-height: 1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.cs-num span { color: var(--muted); font-size: 15px; }
.cs-band { display: inline-block; font-weight: 700; font-size: 14px; color: #fff; border-radius: 999px; padding: 5px 12px; }
.cs-level { font-size: 19px; font-weight: 700; color: var(--ink); margin: 10px 0 4px 0; }
.cs-why { color: var(--muted); font-size: 14.5px; margin-bottom: 16px; }
.cs-split { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: var(--soft); }
.cs-split div { height: 100%; }
.cs-legend { display: flex; flex-direction: column; gap: 4px; font-size: 13.5px; color: var(--muted); margin-top: 8px; }
.cs-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.cs-note { font-size: 13.5px; border-radius: 12px; padding: 10px 12px; margin-top: 14px; }
.cs-note.warn { background: #fff7e6; color: #7a5200; }
.cs-note.stop { background: #fde8ec; color: #8f1d3a; }
@media (max-width: 720px) { .cs-score { grid-template-columns: 1fr; } }

/* "Why" list */
.cs-drivers { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 8px 18px; }
.cs-row { display: grid; grid-template-columns: minmax(150px, 2.2fr) 3fr 56px; gap: 14px; align-items: center;
  padding: 9px 0; border-bottom: 1px solid #f6ebf1; }
.cs-row:last-child { border-bottom: 0; }
.cs-row.quirk { opacity: 0.5; }
.cs-label { font-size: 14.5px; color: var(--ink); }
.cs-chip { display: inline-block; font-size: 12px; color: var(--muted); background: var(--soft); border-radius: 999px;
  padding: 1px 8px; margin-left: 6px; white-space: nowrap; }
.cs-tag { font-size: 11px; font-weight: 700; color: #8a5a00; background: #fff7e6; border-radius: 999px; padding: 1px 7px; margin-left: 6px; }
.cs-track { position: relative; height: 10px; }
.cs-track:before { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: #e3cfdb; }
.cs-bar { position: absolute; top: 0; height: 10px; border-radius: 999px; }
.cs-val { text-align: right; font-size: 14px; font-weight: 700; font-variant-numeric: tabular-nums; }
@media (max-width: 720px) { .cs-row { grid-template-columns: 1fr 64px; } .cs-track { grid-column: 1 / -1; order: 3; } }

/* Stat tiles */
.cs-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
.cs-tile { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
.cs-tile small { display: block; color: var(--raspberry); font-weight: 700; font-size: 12.5px; }
.cs-tile b { display: block; font-size: 26px; color: var(--ink); font-variant-numeric: tabular-nums; }
.cs-tile span { color: var(--muted); font-size: 13px; }

.cs-section { font-size: 20px; font-weight: 700; color: var(--ink); margin: 26px 0 4px 0; }
.cs-sub { color: var(--muted); font-size: 14px; margin-bottom: 12px; }
.cs-footer { color: var(--muted); font-size: 12px; margin-top: 32px; }

@media (prefers-reduced-motion: reduce) {
  .stButton button, .stDownloadButton button, .stTabs [data-testid="stTab"] { transition: none; }
  .stButton button:active, .stDownloadButton button:active { transform: none; }
}
</style>
"""


def apply() -> None:
    st.markdown(CSS.replace("WATERMARK", _watermark()), unsafe_allow_html=True)


def _icon_data_uri() -> str:
    return "data:image/svg+xml;utf8," + quote((LOGO_DIR / "1-cycle-pulse-icon.svg").read_text().strip())


def topbar(pill: str, lede: str, concept: bool = False) -> None:
    """Slim header: logo, name, a status pill that keeps the disclaimer visible, one line of context."""
    st.markdown(
        f"<div class='cs-top'><div class='cs-brand'><img src='{_icon_data_uri()}' alt=''><b>CycleSignal</b></div>"
        f"<span class='cs-pill{' concept' if concept else ''}'>{escape(pill)}</span></div>"
        f"<div class='cs-lede'>{escape(lede)}</div>", unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='cs-section'>{escape(title)}</div>"
                + (f"<div class='cs-sub'>{escape(sub)}</div>" if sub else ""), unsafe_allow_html=True)


def _gauge_svg(score: float, colour: str) -> str:
    """Half-circle gauge: four band arcs with small gaps, and a marker at the score."""
    import math
    cx, cy, r, w = 120, 120, 96, 16

    def point(value):
        a = math.radians(180 - 180 * value / 100)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    arcs, start = [], 0
    for i, (upper, _, c) in enumerate(BANDS):
        (x1, y1), (x2, y2) = point(start + (0 if i == 0 else 0.6)), point(upper - (0 if upper == 100 else 0.6))
        arcs.append(f"<path d='M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}' stroke='{c}' "
                    f"stroke-width='{w}' fill='none' stroke-linecap='butt'/>")
        start = upper
    # rounded outer ends only
    for value, c in ((0, BANDS[0][2]), (100, BANDS[-1][2])):
        ex, ey = point(value)
        arcs.append(f"<circle cx='{ex:.1f}' cy='{ey:.1f}' r='{w / 2}' fill='{c}'/>")
    mx, my = point(max(0.0, min(100.0, score)))
    marker = (f"<circle cx='{mx:.1f}' cy='{my:.1f}' r='13' fill='#fff' stroke='{colour}' stroke-width='5'/>"
              f"<circle cx='{mx:.1f}' cy='{my:.1f}' r='4' fill='{colour}'/>")
    return f"<svg viewBox='0 0 240 136' role='img' aria-label='Score gauge'>{''.join(arcs)}{marker}</svg>"


def score_card(score: float, band: str, level: str, why: str, cycle_pts: float, checkin_pts: float,
               cycle_detail: str, checkin_detail: str, notes: list[tuple[str, str]]) -> None:
    colour = BAND_COLOURS[band]
    rest = max(0.0, 100 - cycle_pts - checkin_pts)
    notes_html = "".join(f"<div class='cs-note {kind}'>{escape(text)}</div>" for kind, text in notes)
    st.markdown(
        f"<div class='cs-score'>"
        f"<div class='cs-gauge'>{_gauge_svg(score, colour)}"
        f"<div class='cs-num'><b style='color:{colour}'>{score:.1f}</b><span> / 100</span></div></div>"
        f"<div><span class='cs-band' style='background:{colour}'>{escape(band)}</span>"
        f"<div class='cs-level'>{escape(level)}</div><div class='cs-why'>{escape(why)}</div>"
        f"<div class='cs-split' aria-label='How the score adds up'>"
        f"<div style='width:{cycle_pts}%;background:{RASPBERRY}'></div>"
        f"<div style='width:{checkin_pts}%;background:{PURPLE}'></div><div style='width:{rest}%'></div></div>"
        f"<div class='cs-legend'><span><i style='background:{RASPBERRY}'></i>Cycle pattern (40%): "
        f"<b>{cycle_pts:.1f}</b> pts · {escape(cycle_detail)}</span>"
        f"<span><i style='background:{PURPLE}'></i>Check-in answers (60%): <b>{checkin_pts:.1f}</b> pts · "
        f"{escape(checkin_detail)}</span></div>{notes_html}</div></div>",
        unsafe_allow_html=True)


def drivers(rows: list[dict]) -> None:
    """Diverging bars from zero: right = raised her score, left = lowered it. Quirks are faded and tagged."""
    biggest = max([abs(r["points"]) for r in rows] + [1.0])
    html = []
    for r in rows:
        p = r["points"]
        width = 50 * abs(p) / biggest
        left = 50 if p >= 0 else 50 - width
        colour = ROSE if p >= 0 else PURPLE
        tag = "<span class='cs-tag'>data quirk</span>" if r["quirk"] else ""
        html.append(
            f"<div class='cs-row{' quirk' if r['quirk'] else ''}'>"
            f"<div class='cs-label'>{escape(r['label'])}<span class='cs-chip'>{escape(r['answer'])}</span>{tag}</div>"
            f"<div class='cs-track'><div class='cs-bar' style='left:{left:.2f}%;width:{max(width, 0.6):.2f}%;background:{colour}'></div></div>"
            f"<div class='cs-val' style='color:{colour}'>{p:+.1f}</div></div>")
    st.markdown(f"<div class='cs-drivers'>{''.join(html)}</div>", unsafe_allow_html=True)


def tiles(items: list[tuple[str, str, str]]) -> None:
    st.markdown("<div class='cs-tiles'>" + "".join(
        f"<div class='cs-tile'><small>{escape(label)}</small><b>{escape(value)}</b><span>{escape(sub)}</span></div>"
        for label, value, sub in items) + "</div>", unsafe_allow_html=True)


def footer() -> None:
    st.markdown("<div class='cs-footer'>CycleSignal · user demo · Data: mcPHASES (PhysioNet) and Kaggle "
                "clinical data. Estimates for information, not a diagnosis.</div>", unsafe_allow_html=True)
