"""Page header with a data-freshness line and a tucked-away refresh control.

Leads every page with trust signals (how current the data is) instead of a
maintenance button in prime real estate. The refresh lives inside a popover so
the main canvas stays focused on data.
"""
from __future__ import annotations
from datetime import date

import streamlit as st

from core.data_store import get_max_price_date


def _freshness_line(latest: date | None) -> str:
    """Human-readable freshness string with a paired colour + text indicator."""
    if latest is None:
        return "⚠️ No price data yet — open the refresh menu to download."
    age = (date.today() - latest).days
    if age <= 1:
        dot = "🟢"
    elif age <= 4:
        dot = "🟡"
    else:
        dot = "🔴"
    plural = "s" if age != 1 else ""
    return f"{dot} Data as of **{latest:%d %b %Y}** · {age} day{plural} ago"


def _run_refresh() -> None:
    """Re-download benchmarks + ETFs in-process (subprocess fails on Cloud)."""
    errors: list[str] = []
    with st.spinner("Downloading global benchmarks from Yahoo Finance…"):
        try:
            from scripts.fetch_market_data import main as _fetch_market
            _fetch_market()
        except Exception as e:
            errors.append(f"Markets fetch failed: {e}")
    with st.spinner("Downloading LSE ETFs from Yahoo Finance…"):
        try:
            from scripts.fetch_global_etfs import main as _fetch_etfs
            _fetch_etfs()
        except Exception as e:
            errors.append(f"ETF fetch failed: {e}")
    st.cache_data.clear()
    if errors:
        for err in errors:
            st.error(err)
    else:
        st.success("All data refreshed.")
        st.rerun()


def render_freshness_header(
    title: str,
    caption: str = "",
    *,
    show_refresh: bool = True,
    refresh_key: str = "hdr_refresh",
) -> None:
    """Render a page title, a ↻ refresh popover, and a freshness caption."""
    left, right = st.columns([8, 1])
    with left:
        st.title(title)
    with right:
        if show_refresh:
            st.write("")
            with st.popover("↻", help="Refresh price data"):
                st.caption("Re-download all global benchmarks + LSE ETFs from "
                           "Yahoo Finance. Takes up to a minute.")
                if st.button("↻ Refresh all data", key=refresh_key):
                    _run_refresh()

    line = _freshness_line(get_max_price_date())
    if caption:
        line = f"{line}  \n{caption}"
    st.caption(line)
