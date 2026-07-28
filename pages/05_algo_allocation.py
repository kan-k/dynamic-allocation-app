from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import itertools
from datetime import datetime, date

import anthropic
from groq import Groq as GroqClient

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

import yfinance as yf

from core.config import GLOBAL_ETFS_PATH
from core.data_store import get_prices, init_db

st.set_page_config(page_title="Algo Allocation", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TICKERS = (
    "VUAG.L", "IKOR.L", "HTWN.L", "CNKY.L", "HCHS.L", "IIND.L", "XFVT.L", "HIES.L",
    "WCOB.L", "SOYO.L",
    "SPLT.L", "SPDM.L", "SILG.L", "SPGP.L", "COPB.L",
    "IB1T.L",
)

SHORT_NAMES: dict[str, str] = {
    "VUAG.L": "S&P 500",    "IKOR.L": "Korea",      "HTWN.L": "Taiwan",
    "CNKY.L": "Nikkei 225", "HCHS.L": "China",      "IIND.L": "India",
    "XFVT.L": "Vietnam",    "HIES.L": "EM Islamic",
    "WCOB.L": "Commodity",  "COPB.L": "Copper",      "SOYO.L": "Soybean Oil",
    "SPLT.L": "Platinum",   "SPDM.L": "Palladium",  "SILG.L": "Silver Miners",
    "SPGP.L": "Gold Prod.", "IB1T.L": "Bitcoin",
}

# Default per-asset bounds (min%, max%) adapted from notebook
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "VUAG.L": (5.0, 25.0), "IKOR.L": (0.0, 17.0), "HTWN.L": (0.0, 17.0),
    "CNKY.L": (5.0, 25.0), "HCHS.L": (0.0, 17.0), "IIND.L": (0.0, 17.0),
    "XFVT.L": (0.0,  5.0), "HIES.L": (0.0, 17.0),
    "WCOB.L": (7.5, 20.0), "COPB.L": (2.5,  5.0), "SOYO.L": (2.5, 10.0),
    "SPLT.L": (0.0,  5.0), "SPDM.L": (0.0,  5.0), "SILG.L": (0.0,  5.0),
    "SPGP.L": (2.5,  7.5), "IB1T.L": (1.5,  5.0),
}

DARK_BG   = "rgba(20,23,30,0.9)"
BAR_GREEN = "#4ade80"
BAR_GREY  = "#555"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_yyyymmdd(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y%m%d").date()
    except ValueError:
        return None


@st.cache_data(ttl=3600)
def load_bitcoin_proxy(start_str: str, end_str: str) -> pd.Series:
    """
    Synthetic BTC price series for IB1T.L, spliced from two sources:
      - BTC-USD (yfinance) for dates before the ETF listing
      - IB1T.L (DuckDB) for dates after the ETF listing
    Splicing is done on *returns* so the level difference between the two
    instruments is irrelevant. The result is a synthetic price series
    (base=100) that has full history back to start_str.
    """
    # IB1T.L from DuckDB
    etf_df = get_prices("IB1T.L", start_str)
    if not etf_df.empty:
        etf_df["date"] = pd.to_datetime(etf_df["date"])
        etf_df = etf_df.set_index("date").sort_index()
        etf_df = etf_df[etf_df.index <= pd.Timestamp(end_str)]
        etf_px = etf_df["close"].dropna()
    else:
        etf_px = pd.Series(dtype=float)

    # BTC-USD from yfinance (add 1 day to end so yfinance includes end_str)
    end_fetch = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download("BTC-USD", start=start_str, end=end_fetch,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return etf_px  # fall back to ETF only

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    btc_px = raw["close"].sort_index()
    btc_px = btc_px[btc_px.index <= pd.Timestamp(end_str)]

    if etf_px.empty:
        return btc_px  # no ETF data at all, use BTC-USD directly

    # Splice on log returns: BTC-USD up to day before ETF listing, ETF thereafter
    etf_start = etf_px.index.min()
    btc_rets = np.log(btc_px / btc_px.shift(1)).dropna()
    etf_rets = np.log(etf_px / etf_px.shift(1)).dropna()

    rets = pd.concat([
        btc_rets[btc_rets.index < etf_start],
        etf_rets,
    ]).sort_index()
    rets = rets[~rets.index.duplicated(keep="last")]

    # Reconstruct synthetic price series (base = 100)
    proxy = 100.0 * np.exp(rets.cumsum())
    proxy.name = "IB1T.L"
    return proxy


@st.cache_data(ttl=3600)
def load_price_matrix(
    tickers: tuple, start_str: str, end_str: str, min_coverage: float = 0.50
) -> pd.DataFrame:
    """
    Returns close-price DataFrame (date × ticker). IB1T.L is loaded via the
    BTC-USD proxy splice. Tickers with < min_coverage of non-NaN rows (after
    ffill) are dropped so one recently-listed ETF can't collapse the date range.
    """
    frames: dict[str, pd.Series] = {}
    for t in tickers:
        if t == "IB1T.L":
            s = load_bitcoin_proxy(start_str, end_str)
            if not s.empty:
                frames[t] = s
            continue
        df = get_prices(t, start_str)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[df.index <= pd.Timestamp(end_str)]
        if not df.empty:
            frames[t] = df["close"]
    px = pd.DataFrame(frames).ffill()
    coverage = px.count() / len(px)
    px = px[coverage[coverage >= min_coverage].index]
    return px.dropna(how="any")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def ewm_stats(rets: pd.DataFrame, halflife: int) -> tuple[pd.Series, pd.DataFrame]:
    mu = rets.ewm(halflife=halflife, min_periods=halflife).mean().iloc[-1] * 252
    cols = rets.columns.tolist()
    Sigma = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            prod = rets[a] * rets[b]
            Sigma.loc[a, b] = prod.ewm(halflife=halflife, min_periods=halflife).mean().iloc[-1] * 252
    return mu, Sigma


def ridge(Sigma: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    n = len(Sigma)
    return Sigma + eps * pd.DataFrame(np.eye(n), index=Sigma.index, columns=Sigma.columns)


def run_max_sharpe(
    mu: pd.Series, Sigma: pd.DataFrame,
    lb: np.ndarray, ub: np.ndarray,
    rf: float = 0.0,
) -> tuple[pd.Series, bool]:
    Sigma_r = ridge(Sigma).values
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ Sigma_r @ w))
        return -((r - rf) / v) if v > 1e-10 else 0.0

    w0 = np.clip(np.ones(n) / n, lb, ub)
    w0 /= w0.sum()
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9},
    )
    w = pd.Series(np.clip(res.x, 0, None), index=mu.index)
    w /= w.sum()
    return w, res.success


def portfolio_metrics(weights: pd.Series, prices: pd.DataFrame) -> dict[str, str]:
    available = [t for t in weights.index if t in prices.columns]
    px = prices[available].dropna()
    if px.shape[0] < 10:
        return {}
    w_sub = weights[available] / weights[available].sum()
    rets = log_returns(px) @ w_sub
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-10 else float("nan")
    cum = np.exp(rets.cumsum())
    dd  = (cum / cum.cummax()) - 1
    max_dd = float(dd.min())
    in_dd = (dd < 0).astype(int).tolist()
    durations = [sum(1 for _ in g) for k, g in itertools.groupby(in_dd) if k]
    max_dur = max(durations, default=0)
    return {
        "Expected Return (ann.)": f"{ann_ret*100:+.2f}%",
        "Volatility (ann.)": f"{ann_vol*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Max Drawdown Duration": f"{max_dur} days",
    }


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return key


def _get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return key


def generate_llm_analysis(weights: pd.Series) -> str:
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "analysis_prompt_template.txt"
    )
    with open(template_path) as f:
        template = f.read()

    alloc_lines = "\n".join(
        f"  {SHORT_NAMES.get(t, t)} ({t}): {v*100:.2f}%"
        for t, v in weights.sort_values(ascending=False).items()
        if v > 0.001
    )
    prompt = template.format(allocation=alloc_lines)

    client = anthropic.Anthropic(api_key=_get_api_key())
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1800,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )


def generate_llm_analysis_groq(weights: pd.Series) -> str:
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "analysis_prompt_template.txt"
    )
    with open(template_path) as f:
        template = f.read()

    alloc_lines = "\n".join(
        f"  {SHORT_NAMES.get(t, t)} ({t}): {v*100:.2f}%"
        for t, v in weights.sort_values(ascending=False).items()
        if v > 0.001
    )
    prompt = (
        f"Today's date: {date.today().strftime('%B %d, %Y')}. "
        "Web search is not available — use your training knowledge for current context.\n\n"
        + template.format(allocation=alloc_lines)
    )

    client = GroqClient(api_key=_get_groq_key())
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
    )
    return response.choices[0].message.content


def bar_chart(weights: pd.Series) -> go.Figure:
    names  = [SHORT_NAMES.get(t, t) for t in weights.index]
    values = (weights * 100).round(2)
    colors = [BAR_GREEN if v >= 1.0 else BAR_GREY for v in values]
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Portfolio Weights",
        height=380,
        margin=dict(l=0, r=0, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=DARK_BG,
        font=dict(color="#FAFAFA", size=12),
        xaxis=dict(gridcolor="#2a2d3a", tickangle=-35),
        yaxis=dict(gridcolor="#2a2d3a", title="Weight (%)", zeroline=True, zerolinecolor="#555"),
    )
    fig.add_hline(y=0, line_color="#555", line_width=1)
    return fig


def pie_chart(weights: pd.Series) -> go.Figure:
    sig = weights[weights >= 0.005]
    if sig.empty:
        sig = weights
    labels = [SHORT_NAMES.get(t, t) for t in sig.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=(sig * 100).round(2),
        hole=0.38,
        textinfo="label+percent",
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate="<b>%{label}</b>: %{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Allocation Breakdown",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

init_db()

st.title("ETF Algo Allocation — Max-Sharpe Optimizer")
st.caption(
    "Combines a long training window and a short EWMA window via EWMA statistics, "
    "then maximises the Sharpe ratio using SLSQP. "
    "All 16 LSE ETFs are sourced from local DuckDB."
)

# --- Settings expander ---
with st.expander("⚙️ Date Periods, Half-Lives & Signal Weights", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Training window** (long baseline)")
        t_start_raw = st.text_input("Start (YYYYMMDD)", "20220101", key="t_start")
        t_end_raw   = st.text_input("End   (YYYYMMDD)", "20251231", key="t_end")
        t_hl = st.number_input("EWMA half-life (days)", min_value=1, max_value=252, value=42, key="t_hl")

    with c2:
        st.markdown("**Short EWMA window** (recent signal)")
        s_start_raw = st.text_input("Start (YYYYMMDD)", "20250401", key="s_start")
        s_end_raw   = st.text_input("End   (YYYYMMDD)", date.today().strftime("%Y%m%d"), key="s_end")
        s_hl = st.number_input("EWMA half-life (days)", min_value=1, max_value=252, value=5, key="s_hl")

    alpha = st.slider(
        "Training weight α  (short weight = 1 − α)",
        0.0, 1.0, 0.50, 0.05,
        help="Combined μ = α·μ_train + (1−α)·μ_short",
    )
    st.caption(
        f"Combined μ = **{alpha:.0%}** × μ_train + **{1-alpha:.0%}** × μ_short   |   "
        f"Same blending applied to Σ"
    )

# --- Bounds expander ---
with st.expander("📐 Asset Bounds (%) — edit inline"):
    bounds_init = pd.DataFrame(
        [
            {"Asset": SHORT_NAMES[t], "Ticker": t,
             "Min (%)": DEFAULT_BOUNDS[t][0], "Max (%)": DEFAULT_BOUNDS[t][1]}
            for t in ALL_TICKERS
        ]
    )
    bounds_edited = st.data_editor(
        bounds_init,
        column_config={
            "Asset":   st.column_config.TextColumn(disabled=True),
            "Ticker":  st.column_config.TextColumn(disabled=True),
            "Min (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5),
            "Max (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5),
        },
        hide_index=True,
        use_container_width=True,
        key="bounds_editor",
    )

run_btn = st.button("▶ Run Optimization", type="primary")

if run_btn:
    # --- Parse & validate dates ---
    t_start = parse_yyyymmdd(t_start_raw)
    t_end   = parse_yyyymmdd(t_end_raw)
    s_start = parse_yyyymmdd(s_start_raw)
    s_end   = parse_yyyymmdd(s_end_raw)

    errors = []
    if t_start is None: errors.append(f"Invalid training start '{t_start_raw}'")
    if t_end   is None: errors.append(f"Invalid training end '{t_end_raw}'")
    if s_start is None: errors.append(f"Invalid short start '{s_start_raw}'")
    if s_end   is None: errors.append(f"Invalid short end '{s_end_raw}'")
    if t_start and t_end and t_end <= t_start:
        errors.append("Training end must be after training start")
    if s_start and s_end and s_end <= s_start:
        errors.append("Short end must be after short start")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    t_start_str = t_start.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    t_end_str   = t_end.strftime("%Y-%m-%d")    # type: ignore[union-attr]
    s_start_str = s_start.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    s_end_str   = s_end.strftime("%Y-%m-%d")    # type: ignore[union-attr]

    # --- Validate bounds ---
    lb_arr = bounds_edited["Min (%)"].values / 100.0
    ub_arr = bounds_edited["Max (%)"].values / 100.0
    if lb_arr.sum() > 1.0:
        st.error(f"Sum of lower bounds ({lb_arr.sum():.1%}) exceeds 100% — reduce some minimums.")
        st.stop()
    if ub_arr.sum() < 1.0:
        st.error(f"Sum of upper bounds ({ub_arr.sum():.1%}) is less than 100% — increase some maximums.")
        st.stop()

    with st.spinner("Loading prices and computing EWMA statistics…"):
        train_px = load_price_matrix(ALL_TICKERS, t_start_str, t_end_str)
        short_px = load_price_matrix(ALL_TICKERS, s_start_str, s_end_str)

        if train_px.empty:
            st.error("No training data found — check dates and DuckDB.")
            st.stop()
        if short_px.empty:
            st.error("No short-window data found — check dates and DuckDB.")
            st.stop()

        # IB1T.L uses BTC-USD as proxy for pre-listing dates — inform user
        if "IB1T.L" in train_px.columns:
            etf_df = get_prices("IB1T.L", t_start_str)
            if not etf_df.empty:
                etf_first = pd.to_datetime(etf_df["date"]).min().date()
                st.info(
                    f"**Bitcoin (IB1T.L):** ETF listed {etf_first}. "
                    "BTC-USD (yfinance) used as return proxy for earlier dates.",
                    icon="ℹ️",
                )

        # Warn about any other tickers excluded due to sparse coverage
        excluded = [t for t in ALL_TICKERS if t not in train_px.columns or t not in short_px.columns]
        if excluded:
            names = ", ".join(SHORT_NAMES.get(t, t) for t in excluded)
            st.warning(
                f"Excluded from optimisation (< 50% data coverage in selected window): **{names}**. "
                "Shorten the training period or extend the data window to include them.",
                icon="⚠️",
            )

        # Restrict to tickers present in BOTH windows
        common = [t for t in ALL_TICKERS if t in train_px.columns and t in short_px.columns]
        if len(common) < 2:
            st.error(f"Only {len(common)} ticker(s) found in both windows — need at least 2.")
            st.stop()

        train_rets = log_returns(train_px[common])
        short_rets = log_returns(short_px[common])

        min_obs_train = int(t_hl) * 3
        min_obs_short = int(s_hl) * 3
        if len(train_rets) < min_obs_train:
            st.error(
                f"Training window only has {len(train_rets)} observations; "
                f"need at least {min_obs_train} (3 × half-life={t_hl}d)."
            )
            st.stop()
        if len(short_rets) < min_obs_short:
            st.warning(
                f"Short window has only {len(short_rets)} observations (3 × half-life={s_hl}d = {min_obs_short}). "
                "EWMA may be unreliable — consider a longer short window or smaller half-life.",
                icon="⚠️",
            )

        mu_train,  Sigma_train  = ewm_stats(train_rets, int(t_hl))
        mu_short,  Sigma_short  = ewm_stats(short_rets, int(s_hl))

        mu_combined    = alpha * mu_train    + (1 - alpha) * mu_short
        Sigma_combined = alpha * Sigma_train + (1 - alpha) * Sigma_short

        # Align bounds to common tickers
        bounds_map = dict(zip(bounds_edited["Ticker"], zip(lb_arr, ub_arr)))
        lb = np.array([bounds_map.get(t, (0.0, 1.0))[0] for t in common])
        ub = np.array([bounds_map.get(t, (0.0, 1.0))[1] for t in common])

    with st.spinner("Running max-Sharpe optimisation…"):
        weights, converged = run_max_sharpe(mu_combined, Sigma_combined, lb, ub)

    if not converged:
        st.warning(
            "Optimizer did not fully converge. Weights are a best-effort solution — "
            "try longer periods, adjusted bounds, or a different α.",
            icon="⚠️",
        )

    st.session_state.pop("llm_analysis", None)  # stale when weights change
    st.session_state["alloc_weights"] = weights
    st.session_state["alloc_train_px"] = train_px

# --- Display results ---
if "alloc_weights" in st.session_state:
    weights: pd.Series = st.session_state["alloc_weights"]
    train_px: pd.DataFrame = st.session_state["alloc_train_px"]

    st.divider()
    st.subheader("Optimised Allocation")

    col_bar, col_pie = st.columns([3, 2])
    with col_bar:
        st.plotly_chart(bar_chart(weights), use_container_width=True)
    with col_pie:
        st.plotly_chart(pie_chart(weights), use_container_width=True)

    # --- Weights table ---
    wdf = pd.DataFrame({
        "Asset":     [SHORT_NAMES.get(t, t) for t in weights.index],
        "Ticker":    weights.index.tolist(),
        "Weight (%)": (weights * 100).round(2).values,
    }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)
    st.dataframe(wdf, use_container_width=True, hide_index=True)

    # --- Statistics ---
    st.divider()
    st.subheader("Portfolio Statistics (computed on training window)")
    metrics = portfolio_metrics(weights, train_px)
    if metrics:
        m_cols = st.columns(len(metrics))
        for col, (label, value) in zip(m_cols, metrics.items()):
            col.metric(label, value)
    else:
        st.info("Insufficient price history to compute statistics.")

    # --- LLM Analysis ---
    st.divider()
    st.subheader("LLM Analysis")

    provider = st.radio(
        "Provider",
        options=["Groq — Llama 3.3 70B (Free)", "Anthropic — Claude Haiku (Paid)"],
        horizontal=True,
        key="llm_provider",
    )
    use_groq = provider.startswith("Groq")

    if use_groq:
        st.caption(
            "Llama 3.3 70B via Groq's free API — composition, diversification quality, "
            "historical risk exposure, and a bottom-line score. "
            "No live web search; uses model training knowledge with today's date as context."
        )
        api_key = _get_groq_key()
        key_name, key_example = "GROQ_API_KEY", "gsk_..."
        signup_note = "Get a free key at [console.groq.com](https://console.groq.com/keys)."
    else:
        st.caption(
            "Claude Haiku + live web search — composition, current macro context, "
            "historical risk exposure, and a bottom-line score. Requires Anthropic API credits."
        )
        api_key = _get_api_key()
        key_name, key_example = "ANTHROPIC_API_KEY", "sk-ant-..."
        signup_note = "Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys)."

    if not api_key:
        st.warning(
            f"`{key_name}` not found. Add it to `.streamlit/secrets.toml` as "
            f"`{key_name} = '{key_example}'`. {signup_note}",
            icon="🔑",
        )
    else:
        col_btn, col_reg = st.columns([2, 8])
        with col_btn:
            run_analysis = st.button("Generate Analysis", type="primary", key="llm_run")
        with col_reg:
            if "llm_analysis" in st.session_state:
                if st.button("↺ Regenerate", key="llm_regen"):
                    del st.session_state["llm_analysis"]
                    st.rerun()

        if run_analysis:
            spinner_msg = (
                "Querying Llama 3.3 70B via Groq…"
                if use_groq
                else "Searching the web and generating analysis…"
            )
            with st.spinner(spinner_msg):
                try:
                    result = (
                        generate_llm_analysis_groq(weights)
                        if use_groq
                        else generate_llm_analysis(weights)
                    )
                    st.session_state["llm_analysis"] = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "llm_analysis" in st.session_state:
            st.markdown(st.session_state["llm_analysis"])
