"""Protection for the hosted demo: secrets, access, rate limits and input/output cleaning.

The app has no database, user accounts or file uploads, so those protections don't apply.
What needs protecting is the Anthropic API key (it spends money) and what users type.

Secrets (never in code or git): app/.streamlit/secrets.toml locally, or the Secrets box on
Streamlit Community Cloud:
    ANTHROPIC_API_KEY = "sk-ant-..."
    APP_PASSWORD = "a long shared password for your team and assessors"
"""

from __future__ import annotations

import hmac
import os
import re
import threading
import time

import streamlit as st

MAX_QUESTION_CHARS = 300
MAX_CLAUDE_CALLS_PER_SESSION = 15
MAX_CLAUDE_CALLS_PER_HOUR = 60          # across every visitor
MIN_SECONDS_BETWEEN_CALLS = 3
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 300


def get_secret(name: str) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:            # no secrets.toml: that's fine locally
        value = None
    value = value or os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def is_local() -> bool:
    try:
        host = (st.context.headers.get("Host") or "").split(":")[0]
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1")


def claude_allowed() -> tuple[bool, str]:
    """Claude spends the owner's credit, so it only runs locally or behind the password."""
    if not get_secret("ANTHROPIC_API_KEY"):
        return False, "no API key set"
    if is_local() or get_secret("APP_PASSWORD"):
        return True, ""
    return False, "set APP_PASSWORD before using Claude on a hosted app"


def require_password() -> None:
    """Stop the page until the visitor enters APP_PASSWORD. No-op when none is set.

    Constant-time comparison; 5 wrong tries locks this session for 5 minutes.
    """
    password = get_secret("APP_PASSWORD")
    if not password or st.session_state.get("authed"):
        return
    st.title("CycleSignal")
    st.caption("Private demo. Enter the password your team shared with you.")
    locked_until = st.session_state.get("locked_until", 0)
    if time.time() < locked_until:
        st.error(f"Too many attempts. Try again in {int(locked_until - time.time())} seconds.")
        st.stop()
    with st.form("login"):
        attempt = st.text_input("Password", type="password", max_chars=200)
        submitted = st.form_submit_button("Open demo")
    if submitted:
        if hmac.compare_digest(attempt.encode(), password.encode()):
            st.session_state.authed = True
            st.session_state.pop("failed", None)
            st.rerun()
        st.session_state.failed = st.session_state.get("failed", 0) + 1
        if st.session_state.failed >= MAX_LOGIN_ATTEMPTS:
            st.session_state.locked_until = time.time() + LOCKOUT_SECONDS
            st.session_state.failed = 0
        st.error("Wrong password.")
    st.stop()


@st.cache_resource
def _global_calls() -> dict:
    return {"lock": threading.Lock(), "times": []}


def take_claude_call() -> tuple[bool, str]:
    """Rate limit: per session, per hour across all visitors, and a short cooldown."""
    now = time.time()
    last = st.session_state.get("last_call", 0)
    if now - last < MIN_SECONDS_BETWEEN_CALLS:
        return False, "Please wait a few seconds between questions."
    used = st.session_state.get("claude_calls", 0)
    if used >= MAX_CLAUDE_CALLS_PER_SESSION:
        return False, "Question limit for this session reached, so the offline assistant will answer instead."
    shared = _global_calls()
    with shared["lock"]:
        shared["times"] = [t for t in shared["times"] if now - t < 3600]
        if len(shared["times"]) >= MAX_CLAUDE_CALLS_PER_HOUR:
            return False, "The demo is busy, so the offline assistant will answer instead."
        shared["times"].append(now)
    st.session_state.claude_calls = used + 1
    st.session_state.last_call = now
    return True, ""


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_question(text: str) -> str:
    """Strip control characters and HTML brackets, collapse spaces, cap the length."""
    text = _CONTROL.sub(" ", text or "").replace("<", " ").replace(">", " ")
    return re.sub(r"\s+", " ", text).strip()[:MAX_QUESTION_CHARS]


_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)\S+")


def clean_answer(text: str) -> str:
    """Remove any links from model output (trusted links are added by the app instead),
    and swap em and en dashes for plain punctuation."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = re.sub(r"\s*—\s*", ", ", text)
    return text.replace("–", "-")
