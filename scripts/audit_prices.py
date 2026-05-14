"""
Detect price data anomalies across all tickers in trading.duckdb.

Run from with_claude/:
    python scripts/audit_prices.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import duckdb
import numpy as np
import pandas as pd
from core.config import DB_PATH

REAL_EVENTS = {
    # (ticker, date_str) pairs confirmed as real market events — suppress from report
    ("VUAG.L", "2020-03-16"),
    ("SPGP.L", "2020-03-17"),
}

def load_all(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    tickers = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchdf()["ticker"].tolist()
    out = {}
    for t in tickers:
        df = con.execute(
            "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY date", [t]
        ).fetchdf()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        out[t] = df
    return out


def check_zero_volume_moves(ticker: str, df: pd.DataFrame, threshold: float = 0.15) -> list[dict]:
    issues = []
    pct = df["close"].pct_change().abs()
    mask = (pct > threshold) & (df["volume"] == 0)
    for d, row in df[mask].iterrows():
        key = (ticker, str(d.date()))
        if key in REAL_EVENTS:
            continue
        issues.append({
            "ticker": ticker, "date": str(d.date()), "type": "zero-vol spike",
            "close": round(row["close"], 4),
            "pct_chg": f"{pct[d]*100:+.1f}%",
            "volume": int(row["volume"]),
        })
    return issues


def check_hl_ratio(ticker: str, df: pd.DataFrame, threshold: float = 2.0) -> list[dict]:
    issues = []
    ratio = df["high"] / df["low"].replace(0, np.nan)
    for d, r in ratio[ratio > threshold].items():
        key = (ticker, str(d.date()))
        if key in REAL_EVENTS:
            continue
        row = df.loc[d]
        issues.append({
            "ticker": ticker, "date": str(d.date()), "type": "H/L ratio",
            "high": round(row["high"], 4), "low": round(row["low"], 4),
            "hl_ratio": round(r, 2), "close": round(row["close"], 4),
        })
    return issues


def check_scale_corruption(ticker: str, df: pd.DataFrame, z_threshold: float = 0.1) -> list[dict]:
    """Flag runs of days where close is far below the rolling median (decimal-shift errors)."""
    issues = []
    med = df["close"].rolling(90, min_periods=20).median()
    ratio = df["close"] / med
    bad = df[ratio < z_threshold]
    if bad.empty:
        return issues
    # Group consecutive bad days into runs
    bad_dates = bad.index.tolist()
    runs, run = [bad_dates[0:1]], bad_dates[:1]
    for d in bad_dates[1:]:
        if (d - run[-1]).days <= 5:
            run.append(d)
        else:
            runs.append(run)
            run = [d]
    runs.append(run)
    for run in runs:
        start, end = run[0], run[-1]
        issues.append({
            "ticker": ticker, "date": f"{start.date()} → {end.date()}",
            "type": "scale corruption",
            "close_range": f"{df.loc[start,'close']:.2f} – {df.loc[end,'close']:.2f}",
            "expected_median": round(float(med.loc[end]), 2),
            "days": len(run),
        })
    return issues


def main() -> None:
    con = duckdb.connect(DB_PATH)
    data = load_all(con)
    con.close()

    all_issues: list[dict] = []
    for ticker, df in data.items():
        all_issues += check_zero_volume_moves(ticker, df)
        all_issues += check_hl_ratio(ticker, df)
        all_issues += check_scale_corruption(ticker, df)

    if not all_issues:
        print("✓ No anomalies detected.")
        return

    by_type: dict[str, list] = {}
    for issue in all_issues:
        by_type.setdefault(issue["type"], []).append(issue)

    for issue_type, items in by_type.items():
        print(f"\n{'='*60}")
        print(f"  {issue_type.upper()}  ({len(items)} found)")
        print(f"{'='*60}")
        for it in items:
            print("  " + "  |  ".join(f"{k}: {v}" for k, v in it.items() if k != "type"))

    print(f"\nTotal: {len(all_issues)} issue(s) across {len({i['ticker'] for i in all_issues})} ticker(s).")


if __name__ == "__main__":
    main()
