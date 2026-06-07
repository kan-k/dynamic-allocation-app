"""All DuckDB reads and writes. Pages never open duckdb directly."""
from __future__ import annotations
import os
import duckdb
import pandas as pd
from datetime import date
from core.config import DB_PATH


def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker  TEXT,
                date    DATE,
                open    DOUBLE,
                high    DOUBLE,
                low     DOUBLE,
                close   DOUBLE,
                volume  BIGINT,
                PRIMARY KEY (ticker, date)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS screening_runs (
                run_id   INTEGER PRIMARY KEY,
                run_date DATE,
                method   TEXT,
                params   JSON
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS screening_results (
                run_id       INTEGER,
                ticker       TEXT,
                alpha        DOUBLE,
                beta_market  DOUBLE,
                beta_smb     DOUBLE,
                beta_hml     DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_runs (
                run_id      INTEGER PRIMARY KEY,
                run_date    DATE,
                method      TEXT,
                total_value DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_allocations (
                run_id      INTEGER,
                ticker      TEXT,
                shares      INTEGER,
                entry_price DOUBLE,
                stop_loss   DOUBLE,
                weight      DOUBLE
            )
        """)


# --- Prices ---

def get_latest_price_date(ticker: str) -> date | None:
    with _conn() as con:
        row = con.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
        ).fetchone()
    return row[0] if row else None


def get_max_price_date() -> date | None:
    """Most recent price date across ALL tickers — powers the freshness header."""
    with _conn() as con:
        row = con.execute("SELECT MAX(date) FROM prices").fetchone()
    return row[0] if row and row[0] else None


def upsert_prices(df: pd.DataFrame, ticker: str) -> None:
    """df must have columns: date, open, high, low, close, volume."""
    df = df.copy()
    df["ticker"] = ticker
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO prices SELECT ticker, date, open, high, low, close, volume FROM df"
        )


def get_prices(ticker: str, start: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM prices WHERE ticker = ?"
    params: list = [ticker]
    if start:
        query += " AND date >= ?"
        params.append(start)
    query += " ORDER BY date"
    with _conn() as con:
        return con.execute(query, params).df()


# --- Screening ---

def save_screening_run(run_date: date, method: str, params: dict, results: pd.DataFrame) -> int:
    with _conn() as con:
        run_id = (con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM screening_runs").fetchone()[0])
        con.execute(
            "INSERT INTO screening_runs VALUES (?, ?, ?, ?)",
            [run_id, run_date, method, str(params)]
        )
        results = results.copy()
        results["run_id"] = run_id
        con.execute("INSERT INTO screening_results SELECT run_id, ticker, alpha, beta_market, beta_smb, beta_hml FROM results")
    return run_id


def get_screening_runs() -> pd.DataFrame:
    with _conn() as con:
        return con.execute("SELECT * FROM screening_runs ORDER BY run_date DESC").df()


def get_screening_results(run_id: int) -> pd.DataFrame:
    with _conn() as con:
        return con.execute(
            "SELECT * FROM screening_results WHERE run_id = ?", [run_id]
        ).df()


# --- Portfolio ---

def save_portfolio_run(run_date: date, method: str, total_value: float, allocations: pd.DataFrame) -> int:
    with _conn() as con:
        run_id = (con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM portfolio_runs").fetchone()[0])
        con.execute(
            "INSERT INTO portfolio_runs VALUES (?, ?, ?, ?)",
            [run_id, run_date, method, total_value]
        )
        allocations = allocations.copy()
        allocations["run_id"] = run_id
        con.execute(
            "INSERT INTO portfolio_allocations SELECT run_id, ticker, shares, entry_price, stop_loss, weight FROM allocations"
        )
    return run_id


def get_portfolio_runs() -> pd.DataFrame:
    with _conn() as con:
        return con.execute("SELECT * FROM portfolio_runs ORDER BY run_date DESC").df()


def get_portfolio_allocations(run_id: int) -> pd.DataFrame:
    with _conn() as con:
        return con.execute(
            "SELECT * FROM portfolio_allocations WHERE run_id = ?", [run_id]
        ).df()
