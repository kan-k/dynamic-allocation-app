"""@st.cache_data wrappers for expensive operations."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import date
import json

from core.config import TICKER_UNIVERSE_PATH, DEFAULT_PERIOD, YFINANCE_SUFFIX
from core.data_store import get_latest_price_date, upsert_prices, get_prices


@st.cache_data(ttl=3600)
def load_tickers() -> list[str]:
    with open(TICKER_UNIVERSE_PATH) as f:
        data = json.load(f)
    if isinstance(data, list):
        return sorted(data)
    return sorted(data.get("tickers", []))


def fetch_and_cache_prices(ticker: str) -> pd.DataFrame:
    """Fetch from yfinance only if DB is stale; return from DB otherwise."""
    latest = get_latest_price_date(ticker)
    if latest is None or latest < date.today():
        import yfinance as yf
        df = yf.download(ticker + YFINANCE_SUFFIX, period=DEFAULT_PERIOD, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns=str.lower)
        df["ticker"] = ticker
        upsert_prices(df[["date", "open", "high", "low", "close", "volume"]], ticker)
    return get_prices(ticker)


@st.cache_data(ttl=3600)
def cached_prices(ticker: str) -> pd.DataFrame:
    return fetch_and_cache_prices(ticker)
