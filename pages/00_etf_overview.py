from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import GLOBAL_ETFS_PATH
from core.data_store import get_prices, init_db

st.set_page_config(page_title="ETF Overview", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[str]] = {
    "Equities — Regional": [
        "VUAG.L", "IKOR.L", "HTWN.L", "CNKY.L",
        "HCHS.L", "IIND.L", "XFVT.L", "HIES.L",
    ],
    "Commodities": ["WCOB.L", "SOYO.L"],
    "Metals & Mining": ["SPLT.L", "SPDM.L", "SILG.L", "SPGP.L", "COPB.L"],
    "Crypto": ["IB1T.L"],
}

SHORT_NAMES: dict[str, str] = {
    "VUAG.L": "S&P 500",
    "IKOR.L": "Korea",
    "HTWN.L": "Taiwan",
    "CNKY.L": "Nikkei 225",
    "HCHS.L": "China",
    "IIND.L": "India",
    "XFVT.L": "Vietnam",
    "HIES.L": "EM Islamic",
    "WCOB.L": "Commodity",
    "COPB.L": "Copper",
    "SOYO.L": "Soybean Oil",
    "SPLT.L": "Platinum",
    "SPDM.L": "Palladium",
    "SILG.L": "Silver Miners",
    "SPGP.L": "Gold Producers",
    "IB1T.L": "Bitcoin",
}

WINDOW_DAYS: dict[str, int | None | str] = {
    "3M": 90, "6M": 182, "1Y": 365, "3Y": 1095, "5Y": 1825, "All": None, "Custom": "custom",
}

RETURN_PERIODS: dict[str, int | str] = {
    "1M": 30, "3M": 90, "6M": 182, "YTD": "ytd", "1Y": 365, "3Y": 1095,
}

VOL_WINDOW = 21

# 16 visually distinct colours (one per ETF in the All-ETFs chart)
COLORS = [
    "#00C49F", "#FF8042", "#0088FE", "#FFBB28", "#FF6B9D",
    "#A8DADC", "#E63946", "#457B9D", "#F4A261", "#2A9D8F",
    "#C77DFF", "#FFD166", "#06D6A0", "#EF476F", "#118AB2", "#FFC8A2",
]

MODE_CUMRET = "Cumulative Return (%)"
MODE_VOL    = "Rolling Volatility (% ann.)"

DARK_BG = "rgba(20,23,30,0.9)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_etf_names() -> dict[str, str]:
    with open(GLOBAL_ETFS_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_close(ticker: str, start_str: str | None, end_str: str | None = None) -> pd.Series:
    df = get_prices(ticker, start_str)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if end_str:
        df = df[df.index <= pd.Timestamp(end_str)]
    return df["close"].dropna()


def parse_yyyymmdd(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y%m%d").date()
    except ValueError:
        return None


def to_start_str(d: date | None) -> str | None:
    return d.strftime("%Y-%m-%d") if d else None


def to_cumret(s: pd.Series) -> pd.Series:
    return (s / s.iloc[0] - 1) * 100


def to_relative_cumret(etf_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    """Excess cumulative return: (ETF normalised / BENCH normalised) − 1, in %."""
    both = pd.DataFrame({"etf": etf_close, "bench": bench_close}).dropna()
    if len(both) < 2:
        return pd.Series(dtype=float)
    return ((both["etf"] / both["etf"].iloc[0]) /
            (both["bench"] / both["bench"].iloc[0]) - 1) * 100


def rolling_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=5).std() * np.sqrt(252) * 100


def compute_series(close: pd.Series, mode: str) -> pd.Series:
    return to_cumret(close) if mode == MODE_CUMRET else rolling_vol(close)


def chart_height(n: int) -> int:
    return max(300, 200 + n * 28)


def period_return(close: pd.Series, days: int) -> float | None:
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    sub = close[close.index >= cutoff]
    if len(sub) < 2:
        return None
    return float((sub.iloc[-1] / sub.iloc[0] - 1) * 100)


def ytd_return(close: pd.Series) -> float | None:
    cutoff = pd.Timestamp(date(date.today().year, 1, 1))
    sub = close[close.index >= cutoff]
    if len(sub) < 2:
        return None
    return float((sub.iloc[-1] / sub.iloc[0] - 1) * 100)


def highlight_col_extremes(col: pd.Series) -> list[str]:
    """Per-column Styler: brightest green for max, brightest red for min, regular otherwise."""
    valid = col.dropna()
    if valid.empty:
        return ["color: #555"] * len(col)
    max_v, min_v = valid.max(), valid.min()
    styles = []
    for v in col:
        if pd.isna(v):
            styles.append("color: #555")
        elif v == max_v:
            styles.append("background-color: #14532d; color: #86efac; font-weight: bold")
        elif v == min_v:
            styles.append("background-color: #7f1d1d; color: #fca5a5; font-weight: bold")
        elif v > 0:
            styles.append("background-color: #0d3320; color: #4ade80")
        else:
            styles.append("background-color: #3d1515; color: #f87171")
    return styles


def make_category_chart(
    tickers: list[str],
    start_str: str | None,
    end_str: str | None,
    mode: str,
    full_names: dict[str, str],
    benchmark_ticker: str | None = None,
    height: int = 360,
) -> go.Figure | None:
    # Load benchmark once (only relevant in cumret mode)
    use_bench = benchmark_ticker is not None and mode == MODE_CUMRET
    bench_close: pd.Series = pd.Series(dtype=float)
    if use_bench:
        bench_close = load_close(benchmark_ticker, start_str, end_str)
        if bench_close.empty or len(bench_close) < 2:
            use_bench = False

    trace_data: list[tuple[str, pd.Series, float, int]] = []
    warned: list[str] = []

    for orig_idx, ticker in enumerate(tickers):
        if use_bench and ticker == benchmark_ticker:
            continue  # would be flat 0 — skip
        close = load_close(ticker, start_str, end_str)
        if close.empty or len(close) < 2:
            warned.append(SHORT_NAMES.get(ticker, ticker))
            continue
        if use_bench:
            series = to_relative_cumret(close, bench_close)
        else:
            series = compute_series(close, mode)
        valid = series.dropna()
        final_val = float(valid.iloc[-1]) if not valid.empty else float("-inf")
        trace_data.append((ticker, series, final_val, orig_idx))

    if warned:
        st.warning(f"No data in window: {', '.join(warned)}", icon="⚠️")

    if not trace_data:
        return None

    trace_data.sort(key=lambda x: x[2], reverse=True)

    if use_bench:
        y_title = f"Excess Return vs {SHORT_NAMES.get(benchmark_ticker, benchmark_ticker)} (%)"
    elif mode == MODE_CUMRET:
        y_title = MODE_CUMRET
    else:
        y_title = "Rolling Volatility (% ann., 21-day)"

    fig = go.Figure()

    for ticker, series, _, orig_idx in trace_data:
        short = SHORT_NAMES.get(ticker, ticker)
        color = COLORS[orig_idx % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=short,
            line=dict(color=color, width=1.8),
            hoverinfo="skip",
        ))

    df_all = pd.DataFrame({t: s for t, s, _, _ in trace_data})

    def _hover_row(row: pd.Series) -> str:
        valid = row.dropna().sort_values(ascending=False)
        if valid.empty:
            return "No data"
        parts = []
        for ticker, v in valid.items():
            name = SHORT_NAMES.get(ticker, ticker)
            fmt = f"{v:+.2f}%" if (use_bench or mode == MODE_CUMRET) else f"{v:.2f}% ann."
            parts.append(f"<b>{name}</b> ({ticker}): {fmt}")
        return "<br>".join(parts)

    hover_texts = df_all.apply(_hover_row, axis=1).tolist()

    fig.add_trace(go.Scatter(
        x=df_all.index,
        y=[0.0] * len(df_all),
        mode="markers",
        marker=dict(opacity=0, size=1),
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts,
        showlegend=False,
        name="",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=DARK_BG,
        font=dict(color="#FAFAFA", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="#2a2d3a", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#2a2d3a", showgrid=True, zeroline=True,
                   zerolinecolor="#555", title=y_title),
        hovermode="x unified",
    )
    if not use_bench and mode == MODE_CUMRET:
        fig.add_hline(y=0, line_color="#666", line_dash="dot", line_width=1)
    elif use_bench:
        fig.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1)

    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

init_db()
full_names = load_etf_names()

st.title("ETF Portfolio Overview")
st.caption("10-year price history for all 16 LSE ETFs — returns, volatility, correlations.")

# --- Controls row ---
col_win, col_mode, col_bench, col_info, col_src = st.columns([5, 4, 3, 1, 1])

with col_win:
    window = st.radio(
        "Time window", list(WINDOW_DAYS.keys()),
        index=2, horizontal=True, key="window",
    )

with col_mode:
    mode = st.radio(
        "Display mode", [MODE_CUMRET, MODE_VOL],
        horizontal=True, key="mode",
    )

with col_bench:
    bench_options: dict[str, str | None] = {"None": None}
    bench_options.update({SHORT_NAMES[t]: t for t in SHORT_NAMES})
    bench_label = st.selectbox(
        "vs Benchmark",
        list(bench_options.keys()),
        key="benchmark",
        help="Cumulative-return charts switch to excess return above the chosen benchmark. "
             "Ignored in volatility mode.",
    )
    benchmark_ticker: str | None = bench_options[bench_label]

with col_info:
    st.write("")
    with st.popover("ℹ️"):
        st.markdown("**Cumulative Return**")
        st.latex(r"R_t = \left(\frac{P_t}{P_0} - 1\right) \times 100\%")
        st.caption("P₀ = first closing price in the selected window")
        st.divider()
        st.markdown("**Rolling Volatility (annualised)**")
        st.latex(r"\sigma_t = \mathrm{std}\!\left(\ln\frac{P_t}{P_{t-1}},\,21\,\mathrm{d}\right) \times \sqrt{252} \times 100\%")
        st.caption("21-day rolling window of daily log-returns, scaled to annual %")
        st.divider()
        st.markdown("**Benchmark Excess Return**")
        st.latex(r"E_t = \left(\frac{P_t/P_0}{B_t/B_0} - 1\right) \times 100\%")
        st.caption("Relative outperformance vs the chosen benchmark (cumret mode only)")

with col_src:
    st.write("")
    with st.popover("📡"):
        st.markdown("### Data Sources — LSE ETFs")
        st.caption(
            "All prices fetched via **Yahoo Finance** (`yfinance`). "
            "Prices are **total-return adjusted** (splits + dividends, `auto_adjust=True`). "
            "Exchange: London Stock Exchange."
        )
        src_df = pd.DataFrame([
            ("S&P 500",       "VUAG.L", "Equities — Regional"),
            ("Korea",         "IKOR.L", "Equities — Regional"),
            ("Taiwan",        "HTWN.L", "Equities — Regional"),
            ("Nikkei 225",    "CNKY.L", "Equities — Regional"),
            ("China",         "HCHS.L", "Equities — Regional"),
            ("India",         "IIND.L", "Equities — Regional"),
            ("Vietnam",       "XFVT.L", "Equities — Regional"),
            ("EM Islamic",    "HIES.L", "Equities — Regional"),
            ("Commodity",     "WCOB.L", "Commodities"),
            ("Copper",        "COPB.L", "Commodities"),
            ("Soybean Oil",   "SOYO.L", "Commodities"),
            ("Platinum",      "SPLT.L", "Metals & Mining"),
            ("Palladium",     "SPDM.L", "Metals & Mining"),
            ("Silver Miners", "SILG.L", "Metals & Mining"),
            ("Gold Producers","SPGP.L", "Metals & Mining"),
            ("Bitcoin",       "IB1T.L", "Crypto"),
        ], columns=["Name", "yfinance Symbol", "Category"])
        st.dataframe(src_df, hide_index=True, use_container_width=True)

# --- Custom date inputs ---
start_str: str | None = None
end_str: str | None   = None

if window == "Custom":
    c1, c2 = st.columns(2)
    raw_start = c1.text_input("Start date (YYYYMMDD)", placeholder="e.g. 20200101", key="custom_start")
    raw_end   = c2.text_input("End date (YYYYMMDD) — leave blank for today", placeholder="e.g. 20241231", key="custom_end")

    parsed_start = parse_yyyymmdd(raw_start) if raw_start else None
    parsed_end   = parse_yyyymmdd(raw_end)   if raw_end   else date.today()

    if raw_start and parsed_start is None:
        st.error(f"Invalid start date '{raw_start}' — use YYYYMMDD format (e.g. 20200101)")
        st.stop()
    if raw_end and parse_yyyymmdd(raw_end) is None:
        st.error(f"Invalid end date '{raw_end}' — use YYYYMMDD format (e.g. 20241231)")
        st.stop()
    if parsed_start and parsed_end and parsed_end < parsed_start:
        st.error("End date must be after start date.")
        st.stop()
    if not raw_start:
        st.info("Enter a start date above to display charts.", icon="📅")
        st.stop()

    start_str = to_start_str(parsed_start)
    end_str   = to_start_str(parsed_end)
    range_label = f"{raw_start} → {raw_end or 'today'}"

else:
    days = WINDOW_DAYS[window]
    start_dt = (date.today() - timedelta(days=days)) if days else None
    start_str = to_start_str(start_dt)
    range_label = f"{start_dt or 'all available'} → today"

bench_label_display = f" · Benchmark: **{bench_label}**" if benchmark_ticker else ""
st.caption(f"Displaying: **{mode}** · Window: **{window}** · {range_label}{bench_label_display}")
st.divider()

# --- Charts (tabbed) ---
all_tickers_flat = [t for tks in CATEGORIES.values() for t in tks]

tab_cat, tab_all = st.tabs(["By Category", "All ETFs"])

with tab_cat:
    for cat_name, tickers in CATEGORIES.items():
        st.subheader(cat_name)
        fig = make_category_chart(
            tickers, start_str, end_str, mode, full_names,
            benchmark_ticker=benchmark_ticker,
            height=chart_height(len(tickers)),
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for this category in the selected window.")
        st.divider()

with tab_all:
    st.subheader("All 16 ETFs")
    fig_all = make_category_chart(
        all_tickers_flat, start_str, end_str, mode, full_names,
        benchmark_ticker=benchmark_ticker,
        height=chart_height(len(all_tickers_flat)),
    )
    if fig_all:
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.info("No data available in the selected window.")

# --- Returns summary table ---
st.subheader("Returns Summary — All Periods")
st.caption(
    "Cumulative % return per ETF across fixed time windows. "
    "Rank by YTD (1 = best). "
    "Bold = best/worst per column · Click a header to sort."
)

rows = []
for cat_name, tickers in CATEGORIES.items():
    for ticker in tickers:
        close = load_close(ticker, None, None)
        row: dict = {
            "Category": cat_name,
            "Name": SHORT_NAMES.get(ticker, ticker),
            "Ticker": ticker,
        }
        for period, days in RETURN_PERIODS.items():
            if close.empty:
                row[period] = float("nan")
            elif days == "ytd":
                v = ytd_return(close)
                row[period] = v if v is not None else float("nan")
            else:
                v = period_return(close, days)
                row[period] = v if v is not None else float("nan")
        rows.append(row)

table_df = pd.DataFrame(rows)
ret_cols = list(RETURN_PERIODS.keys())

table_df["Rank"] = (
    table_df["YTD"].rank(ascending=False, na_option="bottom").astype(int)
)
col_order = ["Rank", "Category", "Name", "Ticker"] + ret_cols
table_df = table_df[col_order]

styler = (
    table_df.style
    .format({col: lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A" for col in ret_cols})
    .apply(highlight_col_extremes, subset=ret_cols)
    .set_properties(subset=ret_cols, **{"text-align": "right", "font-family": "monospace"})
)

st.dataframe(styler, use_container_width=True, hide_index=True)

# --- Correlation heatmap ---
st.divider()
st.subheader("Return Correlations")
st.caption(
    "Pearson correlation of daily log-returns in the selected window. "
    "Green = move together · Red = move opposite · Diagonal always 1."
)

corr_frames = {t: load_close(t, start_str, end_str) for t in all_tickers_flat}
corr_px = pd.DataFrame(corr_frames).dropna(how="all").ffill().dropna(how="any")

if len(corr_px) > 5:
    corr_rets = np.log(corr_px / corr_px.shift(1)).dropna()
    corr_mat = corr_rets.corr()
    labels = [SHORT_NAMES.get(t, t) for t in corr_mat.index]
    corr_mat.index = labels
    corr_mat.columns = labels

    fig_corr = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr_mat.round(2).astype(str).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b>: %{z:.2f}<extra></extra>",
    ))
    fig_corr.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=DARK_BG,
        font=dict(color="#FAFAFA", size=11),
        xaxis=dict(tickangle=-40, side="bottom"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Select a longer window to compute correlations (need > 5 trading days).")
