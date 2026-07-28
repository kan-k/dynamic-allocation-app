"""Shared optimisation engine used by Dynamic Allocation and Strategy Tester pages.

Owns the asset universe, EWM statistics, solver routines, and performance
metrics. Behaviour is identical to the inline versions that previously lived
in pages/07_dynamic_allocation.py.
"""
from __future__ import annotations
import itertools
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

from core.data_store import get_prices

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------

UNIVERSE: dict[str, dict[str, list[str]]] = {
    "LSE ETFs": {
        "Equities — Regional": ["VUAG.L","IKOR.L","HTWN.L","CNKY.L","HCHS.L","IIND.L","XFVT.L","HIES.L"],
        "Commodities":         ["WCOB.L","SOYO.L"],
        "Metals & Mining":     ["SPLT.L","SPDM.L","SILG.L","SPGP.L","COPB.L"],
        "Crypto (ETF)":        ["IB1T.L"],
    },
    "Global Markets": {
        "Equity Indices":  ["SPX","HSI","KOSPI","NKY","FTSE","SX5E","IBOV","SET","TWII"],
        "Energy":          ["WTI"],
        "Metals":          ["GOLD","SILV","COPPER","PALL","PLAT"],
        "Agriculture":     ["SOYB","CORN"],
        "Crypto (Spot)":   ["BTC"],
    },
}

ALL_SHORT_NAMES: dict[str, str] = {
    "VUAG.L":"S&P 500 ETF",   "IKOR.L":"Korea ETF",       "HTWN.L":"Taiwan ETF",
    "CNKY.L":"Nikkei ETF",    "HCHS.L":"China ETF",       "IIND.L":"India ETF",
    "XFVT.L":"Vietnam ETF",   "HIES.L":"EM Islamic ETF",
    "WCOB.L":"Commodity ETF", "COPB.L":"Copper ETF",      "SOYO.L":"Soybean Oil ETF",
    "SPLT.L":"Platinum ETF",  "SPDM.L":"Palladium ETF",   "SILG.L":"Silver Miners ETF",
    "SPGP.L":"Gold Prod. ETF","IB1T.L":"Bitcoin ETF",
    "SPX":"S&P 500",   "HSI":"Hang Seng",  "KOSPI":"KOSPI",    "NKY":"Nikkei 225",
    "FTSE":"FTSE 100", "SX5E":"EuroStoxx", "IBOV":"Bovespa",   "SET":"SET",
    "TWII":"TAIEX",    "WTI":"Crude Oil",  "GOLD":"Gold",      "SILV":"Silver",
    "COPPER":"Copper", "PALL":"Palladium", "PLAT":"Platinum",
    "SOYB":"Soybean",  "CORN":"Corn",      "BTC":"Bitcoin",
}

TICKER_TO_CAT: dict[str, str] = {
    t: cat
    for group in UNIVERSE.values()
    for cat, tickers in group.items()
    for t in tickers
}

ALL_TICKERS: list[str] = [
    t for group in UNIVERSE.values() for tickers in group.values() for t in tickers
]

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    # Equities — Regional (per-ETF tiers)
    "VUAG.L":(5,25),"CNKY.L":(5,25),                                            # core index
    "IKOR.L":(0,17),"IIND.L":(0,17),"HCHS.L":(0,17),"HTWN.L":(0,17),"HIES.L":(0,17),  # regional & EM
    "XFVT.L":(0,5),                                                             # frontier
    # Commodities (category 10–20%)
    "WCOB.L":(7.5,20),"SOYO.L":(2.5,10),
    # Metals & Mining (category 7.5–20%)
    "SPGP.L":(2.5,7.5),"COPB.L":(2.5,5),
    "SPLT.L":(0,5),"SPDM.L":(0,5),"SILG.L":(0,5),
    # Crypto
    "IB1T.L":(1.5,5),
}

# Default per-CATEGORY combined bounds (min%, max%). Categories not listed here
# default to (0, 100) = uncapped. Enforced by run_solver -> build_cat_cons.
DEFAULT_CAT_BOUNDS: dict[str, tuple[float, float]] = {
    "Commodities":     (10, 20),
    "Metals & Mining": (7.5, 20),
}

COLORS = [
    "#00C49F","#FF8042","#0088FE","#FFBB28","#FF6B9D","#A8DADC","#E63946","#457B9D",
    "#F4A261","#2A9D8F","#C77DFF","#FFD166","#06D6A0","#EF476F","#118AB2","#FFC8A2",
    "#B5EAD7","#FFDAC1","#9BF6FF","#BDB2FF",
]

DARK_BG = "rgba(26,29,38,1.0)"  # matches secondaryBackgroundColor #1A1D26
SOLVER_NAMES = ["Max Sharpe Ratio","Min Volatility","Max Return","Min Drawdown"]


def theme() -> dict[str, str]:
    """Chart palette for the currently-selected app theme.

    Reads st.session_state['theme'] (set by the toggle in app.py) and
    returns plot background, font, grid, and zero-line colours. Default
    is 'dark' if the key is unset.
    """
    mode = st.session_state.get("theme", "dark") if hasattr(st, "session_state") else "dark"
    if mode == "cream":
        return {
            "bg":   "rgba(245,241,232,1.0)",   # warm off-white
            "font": "#2C2A26",                  # near-black warm
            "grid": "#D9D3C0",                  # subtle warm grid
            "zero": "#A8A398",                  # muted neutral
        }
    return {
        "bg":   "rgba(26,29,38,1.0)",
        "font": "#FAFAFA",
        "grid": "#2a2d3a",
        "zero": "#4a4d5a",
    }


def themed_table_styles() -> list[dict]:
    """Pandas Styler `set_table_styles` rules matching the current theme.

    Used by pages that render tables via `st.markdown(styler.to_html(),
    unsafe_allow_html=True)` to ensure header / body colours match the
    active theme (glide-data-grid's canvas painting doesn't honour
    external CSS overrides reliably).
    """
    mode = st.session_state.get("theme", "dark") if hasattr(st, "session_state") else "dark"
    if mode == "cream":
        # Header background is left to whatever Streamlit paints (often
        # stays dark even in cream mode); we set the header TEXT to cream
        # so the column names are readable regardless of which paint wins.
        return [
            {"selector": "table",
             "props": "border-collapse: collapse; width: 100%; "
                      "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
                      "font-size: 0.92rem; margin: 0;"},
            {"selector": "thead th",
             "props": "color: #F5F1E8 !important; "
                      "font-weight: 600; padding: 8px 10px; text-align: left; "
                      "border-bottom: 1px solid #BFB89E;"},
            {"selector": "tbody td",
             "props": "background-color: #F5F1E8; color: #2C2A26; padding: 6px 10px; "
                      "border-bottom: 1px solid #E5DFCB;"},
            {"selector": "tbody tr:hover td",
             "props": "background-color: #EFE9D7;"},
        ]
    return [
        {"selector": "table",
         "props": "border-collapse: collapse; width: 100%; "
                  "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
                  "font-size: 0.92rem; margin: 0;"},
        {"selector": "thead th",
         "props": "background-color: #2a2d3a !important; color: #FAFAFA !important; "
                  "font-weight: 600; padding: 8px 10px; text-align: left; "
                  "border-bottom: 1px solid #3a3d4a;"},
        {"selector": "tbody td",
         "props": "background-color: rgba(26,29,38,1.0); color: #FAFAFA; padding: 6px 10px; "
                  "border-bottom: 1px solid #25283a;"},
        {"selector": "tbody tr:hover td",
         "props": "background-color: #25283a;"},
    ]


def render_table(styler_or_df, hide_index: bool = True) -> str:
    """Return HTML string for a DataFrame or Styler with themed styles applied.

    Callers should render via `st.markdown(html, unsafe_allow_html=True)`.
    """
    if hasattr(styler_or_df, "set_table_styles"):
        styler = styler_or_df
    else:
        styler = styler_or_df.style
    if hide_index:
        try:
            styler = styler.hide(axis="index")
        except (AttributeError, TypeError):
            pass
    styler = styler.set_table_styles(themed_table_styles(), overwrite=False)
    return styler.to_html()

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y%m%d").date()
    except ValueError:
        return None


def _resolve(s: str) -> date | None:
    return date.today() if s.strip().lower() in ("today","now") else _parse(s)


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_prices(
    tickers: tuple[str, ...],
    start: str | None,
    end: str | None = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Load an aligned close-price panel.

    `min_coverage` drops tickers whose non-NaN row fraction is below the
    threshold — useful for single-window optimisation. Set to 0.0 to keep
    partial-history tickers (e.g. for backtesters that handle per-date
    eligibility themselves).
    """
    frames: dict[str, pd.Series] = {}
    for t in tickers:
        df = get_prices(t, start)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if not df.empty:
            frames[t] = df["close"]
    if not frames:
        return pd.DataFrame()
    px = pd.DataFrame(frames).ffill(limit=5).dropna(how="all")
    if min_coverage > 0:
        ok = px.count() / len(px) >= min_coverage
        px = px.loc[:, ok]
    return px


def log_ret(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px / px.shift(1)).dropna()


# ---------------------------------------------------------------------------
# EWM statistics
# ---------------------------------------------------------------------------

def ewm_stats(rets: pd.DataFrame, halflife: int) -> tuple[pd.Series, pd.DataFrame]:
    n = len(rets)
    if n < 5:
        mu = pd.Series(0.0, index=rets.columns)
        Sigma = pd.DataFrame(np.eye(len(rets.columns)) * 0.04,
                             index=rets.columns, columns=rets.columns)
        return mu, Sigma
    alpha = 1 - np.exp(-np.log(2) / max(halflife, 1))
    w = (1 - alpha) ** np.arange(n - 1, -1, -1, dtype=float)
    w /= w.sum()
    X = rets.values
    mu_raw = w @ X
    Xd = X - mu_raw
    Sigma_raw = (Xd.T * w) @ Xd * 252
    return (
        pd.Series(mu_raw * 252, index=rets.columns),
        pd.DataFrame(Sigma_raw, index=rets.columns, columns=rets.columns),
    )


def ridge(S: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    n = len(S)
    return S + eps * pd.DataFrame(np.eye(n), index=S.index, columns=S.columns)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _w0(n: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    w = np.clip(np.ones(n) / n, lb, ub)
    s = w.sum()
    return w / s if s > 1e-10 else w


def _base_con() -> list:
    return [{"type":"eq","fun": lambda w: w.sum() - 1.0}]


def build_cat_cons(tickers: list[str], cat_bounds: dict[str, tuple[float,float]]) -> list:
    cons = _base_con()
    for cat, (lo, hi) in cat_bounds.items():
        idx = [i for i, t in enumerate(tickers) if TICKER_TO_CAT.get(t) == cat]
        if not idx:
            continue
        if lo > 0:
            cons.append({"type":"ineq","fun": lambda w, ix=idx, m=lo/100: sum(w[ix]) - m})
        if hi < 100:
            cons.append({"type":"ineq","fun": lambda w, ix=idx, m=hi/100: m - sum(w[ix])})
    return cons


def _slsqp(obj, w0, lb, ub, cons, maxiter=1500):
    res = minimize(obj, w0, method="SLSQP",
                   bounds=list(zip(lb, ub)), constraints=cons,
                   options={"maxiter": maxiter, "ftol": 1e-9})
    return res.x, res.success


def run_solver(
    mu: pd.Series, Sigma: pd.DataFrame, rets_hist: pd.DataFrame,
    lb: np.ndarray, ub: np.ndarray,
    cat_bounds: dict[str, tuple[float,float]],
    solver_name: str, solver_params: dict,
) -> tuple[pd.Series, bool]:
    tickers = mu.index.tolist()
    cons = build_cat_cons(tickers, cat_bounds)
    Sr = ridge(Sigma).values
    n = len(mu)
    rf = solver_params.get("rf", 0.0)
    w0 = _w0(n, lb, ub)

    if solver_name == "Max Sharpe Ratio":
        def obj(w):
            r = float(w @ mu.values)
            v = float(np.sqrt(w @ Sr @ w))
            return -((r - rf) / v) if v > 1e-10 else 0.0
        raw, ok = _slsqp(obj, w0, lb, ub, cons)

    elif solver_name == "Min Volatility":
        min_ret = solver_params.get("min_ret", 0.05)
        ext = [{"type":"ineq","fun": lambda w: float(w @ mu.values) - min_ret}]
        raw, ok = _slsqp(lambda w: float(np.sqrt(w @ Sr @ w)), w0, lb, ub, cons + ext)

    elif solver_name == "Max Return":
        max_vol = solver_params.get("max_vol", 0.15)
        ext = [{"type":"ineq","fun": lambda w: max_vol - float(np.sqrt(w @ Sr @ w))}]
        raw, ok = _slsqp(lambda w: float(-w @ mu.values), w0, lb, ub, cons + ext)

    else:  # Min Drawdown
        X = rets_hist.values
        def dd_obj(w):
            cum = np.exp(np.cumsum(X @ w))
            return float(-((cum / np.maximum.accumulate(cum)) - 1).min())
        best_w, best_val, ok = w0, float("inf"), False
        rng = np.random.default_rng(42)
        for _ in range(3):
            raw_i, ok_i = _slsqp(dd_obj, w0, lb, ub, cons, maxiter=800)
            val = dd_obj(raw_i)
            if val < best_val:
                best_val, best_w, ok = val, raw_i, ok_i
            w0 = np.clip(rng.dirichlet(np.ones(n)), lb, ub)
            w0 /= w0.sum()
        raw = best_w

    w = np.clip(raw, 0, None)
    s = w.sum()
    return pd.Series(w / s if s > 1e-10 else w, index=tickers), ok


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def portfolio_metrics(weights: pd.Series, px: pd.DataFrame) -> dict[str, str]:
    avail = [t for t in weights.index if t in px.columns]
    if not avail:
        return {}
    sub = px[avail].dropna()
    if len(sub) < 10:
        return {}
    w = weights[avail] / weights[avail].sum()
    rets = log_ret(sub) @ w
    ann_r = float(rets.mean() * 252)
    ann_v = float(rets.std() * np.sqrt(252))
    sharpe = ann_r / ann_v if ann_v > 1e-10 else float("nan")
    cum = np.exp(rets.cumsum())
    dd = (cum / cum.cummax()) - 1
    max_dd = float(dd.min())
    durs = [sum(1 for _ in g) for k, g in itertools.groupby((dd < 0).astype(int).tolist()) if k]
    return {
        "Daily Ret. (mean)": f"{rets.mean()*100:+.3f}%",
        "Ann. Return":       f"{ann_r*100:+.2f}%",
        "Ann. Volatility":   f"{ann_v*100:.2f}%",
        "Sharpe Ratio":      f"{sharpe:.3f}",
        "Max Drawdown":      f"{max_dd*100:.2f}%",
        "Max DD Duration":   f"{max(durs, default=0)} days",
    }


def ewm_portfolio_metrics(
    weights: pd.Series, mu: pd.Series, Sigma: pd.DataFrame, rf: float = 0.0
) -> dict[str, str]:
    """EWM-weighted 'optimiser view' metrics — same mu/Sigma the solver saw."""
    avail = [t for t in weights.index if t in mu.index]
    if not avail:
        return {}
    wv = weights[avail].values
    mv = mu[avail].values
    Sv = Sigma.loc[avail, avail].values
    r = float(wv @ mv)
    v = float(np.sqrt(max(wv @ Sv @ wv, 0.0)))
    s = (r - rf) / v if v > 1e-10 else float("nan")
    return {
        "EWM Ann. Return":     f"{r*100:+.2f}%",
        "EWM Ann. Volatility": f"{v*100:.2f}%",
        "EWM Sharpe Ratio":    f"{s:.3f}",
    }
