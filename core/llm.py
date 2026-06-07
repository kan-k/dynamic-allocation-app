"""Shared LLM helpers for the 'Economist Interpretation' blocks.

Previously inlined in pages/07_dynamic_allocation.py; lifted here so both the
interpretation component and any page can share one implementation. Keys are
read from the environment first, then Streamlit secrets (configured in the
Streamlit Cloud dashboard, never committed).
"""
from __future__ import annotations
import os
from datetime import date

import streamlit as st

try:
    from groq import Groq as GroqClient
except ImportError:               # optional dependency
    GroqClient = None

try:
    import anthropic as _ant
except ImportError:               # optional dependency
    _ant = None

# Provider labels shared by every interpretation block.
PROVIDERS = [
    "Groq — Llama 3.3 70B (Free)",
    "Anthropic — Claude Haiku (Paid)",
]


def _groq_key() -> str:
    k = os.environ.get("GROQ_API_KEY", "")
    if not k:
        try:
            k = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return k


def _ant_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    if not k:
        try:
            k = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return k


def llm(prompt: str, provider: str) -> str:
    """Run `prompt` against the chosen provider and return the text response."""
    if "Groq" in provider:
        if GroqClient is None:
            raise RuntimeError("groq package not installed.")
        client = GroqClient(api_key=_groq_key())
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",
                 "content": "You are a senior economist and portfolio strategist."},
                {"role": "user",
                 "content": f"Today: {date.today().strftime('%B %d, %Y')}.\n\n{prompt}"},
            ],
            max_tokens=1200,
        )
        return resp.choices[0].message.content

    if _ant is None:
        raise RuntimeError("anthropic package not installed.")
    client = _ant.Anthropic(api_key=_ant_key())
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1800,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))
