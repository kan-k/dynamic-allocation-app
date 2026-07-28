"""
Fetch and store 10-year OHLCV history for all LSE ETFs defined in data/global_etfs.json.

Run from the with_claude/ directory:
    python scripts/fetch_global_etfs.py
"""
from __future__ import annotations
import sys
import os

# Ensure core/ is importable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import yfinance as yf
from core.config import GLOBAL_ETFS_PATH, GLOBAL_ETF_PERIOD
from core.data_store import init_db, upsert_prices


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from newer yfinance versions and lowercase all names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    return df


def fetch_ticker(symbol: str) -> pd.DataFrame | None:
    df = yf.download(symbol, period=GLOBAL_ETF_PERIOD, auto_adjust=True, progress=False)
    if df.empty:
        return None
    df = df.reset_index()
    df = _normalise_columns(df)
    df = df.rename(columns={"date": "date"})
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df[["date", "open", "high", "low", "close", "volume"]]


def main() -> None:
    init_db()

    with open(GLOBAL_ETFS_PATH) as f:
        etfs: dict[str, str] = json.load(f)

    summary: list[dict] = []

    for symbol, name in etfs.items():
        print(f"Fetching {symbol:12}  {name} ...", end="  ", flush=True)
        try:
            df = fetch_ticker(symbol)
            if df is None or df.empty:
                print("WARNING: no data returned")
                summary.append({"symbol": symbol, "rows": 0, "start": "-", "end": "-", "status": "no data"})
                continue
            upsert_prices(df, symbol)
            row_count = len(df)
            date_start = str(df["date"].min())[:10]
            date_end = str(df["date"].max())[:10]
            print(f"{row_count} rows  {date_start} → {date_end}")
            summary.append({"symbol": symbol, "rows": row_count, "start": date_start, "end": date_end, "status": "ok"})
        except Exception as exc:
            print(f"ERROR: {exc}")
            summary.append({"symbol": symbol, "rows": 0, "start": "-", "end": "-", "status": f"error: {exc}"})

    print("\n" + "=" * 70)
    print(f"{'Symbol':<12} {'Rows':>6}  {'Start':<12} {'End':<12} Status")
    print("-" * 70)
    for r in summary:
        print(f"{r['symbol']:<12} {r['rows']:>6}  {r['start']:<12} {r['end']:<12} {r['status']}")
    print("=" * 70)

    # Auto-correct known data anomalies so a refresh never leaves corrupt rows in
    # the DB (e.g. the 2025-10-24 batch spike, which yfinance re-introduces on each
    # fetch). See scripts/fix_price_anomalies.py for the registry of fixes.
    print("\nApplying known-anomaly corrections…")
    from scripts.fix_price_anomalies import (
        pass1_interpolate_spikes, pass2_refetch_spdm, pass3_fix_hl_fields,
    )
    pass1_interpolate_spikes()
    pass2_refetch_spdm()
    pass3_fix_hl_fields()

    ok = sum(1 for r in summary if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(etfs)} tickers stored successfully.")
    print(f"DB: {os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/trading.duckdb'))}")


if __name__ == "__main__":
    main()
