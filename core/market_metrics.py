"""Shared market-return helpers.

Single source of truth for the close-price loader and the period-return maths
that page 06 (Market Overview) previously defined inline, plus higher-level
snapshot/breadth/regime helpers used by the Today landing page. Everything is
computed from real cached prices — no fabricated figures.
"""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from core.data_store import get_prices

# ---------------------------------------------------------------------------
# Low-level: single-symbol close series and period returns
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_close(ticker: str, start_str: str | None, end_str: str | None = None) -> pd.Series:
    """Cached close-price series for one ticker, optionally windowed."""
    df = get_prices(ticker, start_str)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if end_str:
        df = df[df.index <= pd.Timestamp(end_str)]
    return df["close"].dropna()


def period_return(close: pd.Series, days: int) -> float | None:
    """% return over a trailing calendar window of `days`."""
    if close.empty:
        return None
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    sub = close[close.index >= cutoff]
    if len(sub) < 2:
        return None
    return float((sub.iloc[-1] / sub.iloc[0] - 1) * 100)


def ytd_return(close: pd.Series) -> float | None:
    """% return from the first trading day of the current calendar year."""
    if close.empty:
        return None
    cutoff = pd.Timestamp(date(date.today().year, 1, 1))
    sub = close[close.index >= cutoff]
    if len(sub) < 2:
        return None
    return float((sub.iloc[-1] / sub.iloc[0] - 1) * 100)


def pct_change_1d(close: pd.Series) -> float | None:
    """% change of the most recent close vs the prior close."""
    if len(close) < 2:
        return None
    return float((close.iloc[-1] / close.iloc[-2] - 1) * 100)


# ---------------------------------------------------------------------------
# Higher-level: cross-sectional snapshot, breadth, regime
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def snapshot_frame(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Build a one-row-per-ticker frame of last price and 1D/1W/1M/YTD returns."""
    rows: list[dict] = []
    for t in tickers:
        c = load_close(t, None, None)
        rows.append({
            "Ticker": t,
            "Last":   float(c.iloc[-1]) if not c.empty else float("nan"),
            "1D":     pct_change_1d(c),
            "1W":     period_return(c, 7),
            "1M":     period_return(c, 30),
            "YTD":    ytd_return(c),
        })
    df = pd.DataFrame(rows)
    # Normalise None -> NaN for the numeric columns.
    for col in ("1D", "1W", "1M", "YTD"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def breadth(snap: pd.DataFrame, col: str = "1D") -> tuple[int, int, int]:
    """Count advancers / decliners / unchanged over the given return column."""
    vals = snap[col].dropna()
    up = int((vals > 0).sum())
    down = int((vals < 0).sum())
    flat = int((vals == 0).sum())
    return up, down, flat


def regime_snapshot(snap: pd.DataFrame) -> dict[str, str]:
    """A small, clearly-heuristic read of market regime from 1M price trends.

    Not investment advice — just a labelled summary of what prices did. Keys are
    DB tickers expected in `snap`; missing ones are skipped gracefully.
    """
    by_ticker = snap.set_index("Ticker")

    def m1(ticker: str) -> float | None:
        if ticker not in by_ticker.index:
            return None
        v = by_ticker.loc[ticker, "1M"]
        return None if pd.isna(v) else float(v)

    out: dict[str, str] = {}
    spx, gold, wti, btc = m1("SPX"), m1("GOLD"), m1("WTI"), m1("BTC")

    # Risk appetite: equities rising and outpacing gold => risk-on.
    if spx is not None and gold is not None:
        if spx > 0 and spx >= gold:
            out["Risk appetite"] = "Risk-on"
        elif spx < 0 and gold > 0:
            out["Risk appetite"] = "Risk-off"
        else:
            out["Risk appetite"] = "Mixed"

    up, down, _ = breadth(snap, "1M")
    out["Breadth (1M)"] = f"{up} up / {down} down"
    if spx is not None:
        out["S&P 500 (1M)"] = f"{spx:+.1f}%"
    if gold is not None:
        out["Gold (1M)"] = f"{gold:+.1f}%"
    if wti is not None:
        out["Crude (1M)"] = f"{wti:+.1f}%"
    if btc is not None:
        out["Bitcoin (1M)"] = f"{btc:+.1f}%"
    return out
