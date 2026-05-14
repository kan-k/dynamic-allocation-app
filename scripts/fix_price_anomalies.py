"""
Apply targeted corrections to known price data errors in trading.duckdb.

Three passes:
  Pass 1 — Delete single-day zero-volume spikes, replace with interpolated values
  Pass 2 — Re-fetch SPDM.L from yfinance; fix decimal-scale if still corrupt
  Pass 3 — Fix individual low-price field errors (decimal shift in H/L)

Run from with_claude/:
    python scripts/fix_price_anomalies.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from core.config import DB_PATH, GLOBAL_ETF_PERIOD

# ---------------------------------------------------------------------------
# Pass 1 — Known single-day spikes to interpolate
# ---------------------------------------------------------------------------
SPIKE_ROWS = [
    # 2025-10-24 batch corruption (same day, zero volume, ~33% spike across 7 tickers)
    ("HCHS.L", "2025-10-24"),
    ("HIES.L", "2025-10-24"),
    ("WCOB.L", "2025-10-24"),
    ("XFVT.L", "2025-10-24"),
    ("SPGP.L", "2025-10-24"),
    ("HTWN.L", "2025-10-24"),
    ("IIND.L", "2025-10-24"),
    # Other isolated zero-volume spikes
    ("HTWN.L", "2017-06-27"),
    ("COPB.L", "2023-02-21"),
    ("COPB.L", "2025-07-31"),
    ("HIES.L", "2023-01-13"),
    ("SPLT.L", "2023-03-13"),
    ("SPLT.L", "2026-01-30"),
]

# Pass 3 — Known H/L field errors (only the low column is wrong)
HL_FIXES = [
    ("IIND.L", "2024-04-08"),
    ("XFVT.L", "2021-04-16"),
]


def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def _get_neighbors(con: duckdb.DuckDBPyConnection, ticker: str, bad_date: str) -> tuple[pd.Series, pd.Series]:
    """Return the rows immediately before and after bad_date for ticker."""
    prev = con.execute(
        "SELECT * FROM prices WHERE ticker = ? AND date < ? ORDER BY date DESC LIMIT 1",
        [ticker, bad_date],
    ).fetchdf()
    nxt = con.execute(
        "SELECT * FROM prices WHERE ticker = ? AND date > ? ORDER BY date ASC LIMIT 1",
        [ticker, bad_date],
    ).fetchdf()
    if prev.empty or nxt.empty:
        raise ValueError(f"Cannot interpolate {ticker} {bad_date}: missing neighbor rows")
    return prev.iloc[0], nxt.iloc[0]


def pass1_interpolate_spikes() -> None:
    print("\n── Pass 1: Interpolating single-day spikes ──")
    con = _conn()
    for ticker, bad_date in SPIKE_ROWS:
        try:
            before, after = _get_neighbors(con, ticker, bad_date)
            interp = pd.DataFrame([{
                "ticker": ticker,
                "date":   pd.Timestamp(bad_date),
                "open":   (before["open"]  + after["open"])  / 2,
                "high":   (before["high"]  + after["high"])  / 2,
                "low":    (before["low"]   + after["low"])   / 2,
                "close":  (before["close"] + after["close"]) / 2,
                "volume": 0,
            }])
            # Delete old row then upsert interpolated row
            con.execute("DELETE FROM prices WHERE ticker = ? AND date = ?", [ticker, bad_date])
            con.execute(
                "INSERT OR REPLACE INTO prices "
                "SELECT ticker, date, open, high, low, close, volume FROM interp"
            )
            new_close = interp.iloc[0]["close"]
            print(f"  ✓ {ticker} {bad_date}: replaced with interpolated close={new_close:.4f}")
        except Exception as e:
            print(f"  ✗ {ticker} {bad_date}: {e}")
    con.close()


def pass2_refetch_spdm() -> None:
    print("\n── Pass 2: Re-fetching SPDM.L from yfinance ──")
    raw = yf.download("SPDM.L", period=GLOBAL_ETF_PERIOD, auto_adjust=True, progress=False)
    if raw.empty:
        print("  ✗ yfinance returned no data for SPDM.L")
        return

    df = raw.reset_index()
    # Flatten MultiIndex columns (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["ticker"] = "SPDM.L"

    # Check for scale corruption: days where close < 100 when median > 1000
    median_close = df["close"].median()
    bad_mask = (df["close"] < 100) & (median_close > 1000)
    if bad_mask.any():
        n_bad = bad_mask.sum()
        print(f"  ⚠ Still detecting {n_bad} corrupted rows (scale ÷100); applying ×100 correction")
        for col in ("open", "high", "low", "close"):
            df.loc[bad_mask, col] = df.loc[bad_mask, col] * 100

    con = _conn()
    con.execute(
        "INSERT OR REPLACE INTO prices "
        "SELECT ticker, date, open, high, low, close, volume FROM df"
    )
    con.close()

    # Report the fixed range
    if bad_mask.any():
        fixed_dates = df.loc[bad_mask, "date"]
        print(f"  ✓ Corrected {n_bad} rows: {fixed_dates.min().date()} → {fixed_dates.max().date()}")
    else:
        print(f"  ✓ Re-fetched {len(df)} rows — no scale corruption detected in fresh data")


def pass3_fix_hl_fields() -> None:
    print("\n── Pass 3: Fixing individual H/L field errors ──")
    con = _conn()
    for ticker, date_str in HL_FIXES:
        row = con.execute(
            "SELECT open, high, low, close FROM prices WHERE ticker = ? AND date = ?",
            [ticker, date_str],
        ).fetchdf()
        if row.empty:
            print(f"  ✗ {ticker} {date_str}: row not found")
            continue
        r = row.iloc[0]
        corrected_low = min(r["open"], r["close"]) * 0.995
        con.execute(
            "UPDATE prices SET low = ? WHERE ticker = ? AND date = ?",
            [corrected_low, ticker, date_str],
        )
        print(f"  ✓ {ticker} {date_str}: low {r['low']:.4f} → {corrected_low:.4f}")
    con.close()


def main() -> None:
    print("=== fix_price_anomalies.py ===")
    pass1_interpolate_spikes()
    pass2_refetch_spdm()
    pass3_fix_hl_fields()
    print("\n✓ All passes complete. Run audit_prices.py to verify.")


if __name__ == "__main__":
    main()
