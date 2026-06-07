"""Reusable 'Economist Interpretation' block.

Page 07 previously repeated this provider-radio + Generate/Regenerate pattern
verbatim in two places; this is the single definition. The caller builds the
prompt string (it depends on page-specific data) and supplies a session_state
key under which the generated text is cached for the session.
"""
from __future__ import annotations
from typing import Callable

import streamlit as st

from core.llm import PROVIDERS, llm


def interpretation_block(
    prompt_builder: Callable[[], str],
    state_key: str,
    key_prefix: str,
    *,
    title: str = "Economist Interpretation",
) -> None:
    """Render the interpretation UI.

    `prompt_builder` is called only when the user clicks Generate, so the
    (potentially expensive) prompt assembly is deferred until needed.
    """
    st.markdown(f"**{title}**")
    provider = st.radio("Provider", PROVIDERS, horizontal=True, key=f"{key_prefix}prov")

    c1, c2 = st.columns([2, 2])
    if c1.button("Generate", key=f"{key_prefix}gen"):
        with st.spinner("Generating interpretation…"):
            try:
                st.session_state[state_key] = llm(prompt_builder(), provider)
            except Exception as e:
                st.error(f"LLM error: {e}")

    if state_key in st.session_state:
        if c2.button("↺ Regenerate", key=f"{key_prefix}regen"):
            del st.session_state[state_key]
            st.rerun()
        st.markdown(st.session_state[state_key])
