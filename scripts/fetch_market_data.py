"""
Download 10 years of OHLCV data for 18 global market benchmarks and upsert
into the project DuckDB. Run standalone:

    python3.9 scripts/fetch_market_data.py

Also called by pages/06_market_overview.py for in-app refresh.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pandas as pd
import yfinance as yf

from core.data_store import init_db, upsert_prices

# yfinance symbol → internal DB ticker (uppercase, no suffix)
TICKERS: dict[str, str] = {
    "^GSPC":    "SPX",
    "^HSI":     "HSI",
    "^KS11":    "KOSPI",
    "^N225":    "NKY",
    "^FTSE":    "FTSE",
    "^STOXX50E":"SX5E",
    "^BVSP":    "IBOV",
    "^SET.BK":  "SET",
    "^TWII":    "TWII",
    "CL=F":     "WTI",
    "GC=F":     "GOLD",
    "SI=F":     "SILV",
    "HG=F":     "COPPER",
    "PA=F":     "PALL",
    "PL=F":     "PLAT",
    "ZS=F":     "SOYB",
    "ZC=F":     "CORN",
    "BTC-USD":  "BTC",
}

PERIOD = "10y"


def fetch_one(yf_sym: str, db_ticker: str) -> int:
    """Fetch PERIOD of history for yf_sym, upsert as db_ticker. Returns rows upserted."""
    raw = yf.download(yf_sym, period=PERIOD, auto_adjust=True, progress=False)

    if raw.empty:
        print(f"  {db_ticker:8s}  [{yf_sym}]  no data returned — skipped")
        return 0

    # yfinance ≥0.2 may return MultiIndex columns (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    if "volume" not in raw.columns:
        raw["volume"] = 0
    raw["volume"] = raw["volume"].fillna(0).astype("int64")

    # Normalise index → date column
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else "Datetime"
    raw = raw.rename(columns={date_col: "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.date

    df = raw[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

    upsert_prices(df, db_ticker)
    print(f"  {db_ticker:8s}  [{yf_sym}]  {len(df):5d} rows upserted")
    return len(df)


def main() -> None:
    init_db()
    print(f"Fetching {len(TICKERS)} global benchmarks (period={PERIOD}) …\n")
    total = 0
    errors: list[str] = []
    for yf_sym, db_ticker in TICKERS.items():
        try:
            total += fetch_one(yf_sym, db_ticker)
        except Exception as exc:
            print(f"  {db_ticker:8s}  [{yf_sym}]  ERROR: {exc}")
            errors.append(db_ticker)
        time.sleep(0.3)  # gentle rate limit

    print(f"\nDone. {total} total rows upserted across {len(TICKERS) - len(errors)} tickers.")
    if errors:
        print(f"Failed tickers: {errors}")


if __name__ == "__main__":
    main()
