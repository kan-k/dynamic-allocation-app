from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.optim_engine import theme, render_table
from core.market_metrics import (
    load_close, period_return, ytd_return, snapshot_frame, breadth,
)
from components.header import render_freshness_header

# set_page_config is owned by app.py (the st.navigation entrypoint).

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MKT_CATEGORIES: dict[str, list[str]] = {
    "Equity Indices": ["SPX", "HSI", "KOSPI", "NKY", "FTSE", "SX5E", "IBOV", "SET", "TWII"],
    "Energy":         ["WTI"],
    "Metals":         ["GOLD", "SILV", "COPPER", "PALL", "PLAT"],
    "Agriculture":    ["SOYB", "CORN"],
    "Crypto":         ["BTC"],
}

MKT_SHORT_NAMES: dict[str, str] = {
    "SPX": "S&P 500", "HSI": "Hang Seng", "KOSPI": "KOSPI", "NKY": "Nikkei 225",
    "FTSE": "FTSE 100", "SX5E": "EuroStoxx 50", "IBOV": "Bovespa", "SET": "SET",
    "TWII": "TAIEX", "WTI": "Crude Oil WTI", "GOLD": "Gold", "SILV": "Silver",
    "COPPER": "Copper", "PALL": "Palladium", "PLAT": "Platinum",
    "SOYB": "Soybean", "CORN": "Corn", "BTC": "Bitcoin",
}

ETF_CATEGORIES: dict[str, list[str]] = {
    "Equities — Regional": [
        "VUAG.L", "IKOR.L", "HTWN.L", "CNKY.L",
        "HCHS.L", "IIND.L", "XFVT.L", "HIES.L",
    ],
    "Commodities":     ["WCOB.L", "COPB.L", "SOYO.L"],
    "Metals & Mining": ["SPLT.L", "SPDM.L", "SILG.L", "SPGP.L"],
    "Crypto":          ["IB1T.L"],
}

ETF_SHORT_NAMES: dict[str, str] = {
    "VUAG.L": "S&P 500", "IKOR.L": "Korea", "HTWN.L": "Taiwan", "CNKY.L": "Nikkei 225",
    "HCHS.L": "China", "IIND.L": "India", "XFVT.L": "Vietnam", "HIES.L": "EM Islamic",
    "WCOB.L": "Commodity", "COPB.L": "Copper", "SOYO.L": "Soybean Oil",
    "SPLT.L": "Platinum", "SPDM.L": "Palladium", "SILG.L": "Silver Miners",
    "SPGP.L": "Gold Producers", "IB1T.L": "Bitcoin",
}

WINDOW_DAYS: dict[str, int | None | str] = {
    "3M": 90, "6M": 182, "1Y": 365, "3Y": 1095, "5Y": 1825, "All": None, "Custom": "custom",
}

RETURN_PERIODS: dict[str, int | str] = {
    "1M": 30, "3M": 90, "6M": 182, "YTD": "ytd", "1Y": 365, "3Y": 1095,
}

VOL_WINDOW = 21

COLORS = [
    "#00C49F", "#FF8042", "#0088FE", "#FFBB28", "#FF6B9D",
    "#A8DADC", "#E63946", "#457B9D", "#F4A261", "#2A9D8F",
    "#C77DFF", "#FFD166", "#06D6A0", "#EF476F", "#118AB2", "#FFC8A2",
    "#B5EAD7", "#FFDAC1",
]

MODE_CUMRET = "Cumulative Return (%)"
MODE_VOL    = "Rolling Volatility (% ann.)"

# ---------------------------------------------------------------------------
# Series helpers (chart maths)
# ---------------------------------------------------------------------------

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


def highlight_col_extremes(col: pd.Series) -> list[str]:
    is_cream = (
        st.session_state.get("theme", "dark") == "cream"
        if hasattr(st, "session_state") else False
    )
    if is_cream:
        nan_c = "color: #A8A398"
        max_s = "background-color: #BBE5C8; color: #1c5234; font-weight: bold"
        min_s = "background-color: #F4CDCD; color: #722828; font-weight: bold"
        pos_s = "background-color: #DCEEDF; color: #2d5e3a"
        neg_s = "background-color: #F5DEDE; color: #7e3a3a"
    else:
        nan_c = "color: #5a5d6a"
        max_s = "background-color: #0d4d40; color: #5eead4; font-weight: bold"
        min_s = "background-color: #5c1e26; color: #fca5a5; font-weight: bold"
        pos_s = "background-color: #0a2e2a; color: #5eead4"
        neg_s = "background-color: #3a1a1f; color: #f87171"

    valid = col.dropna()
    if valid.empty:
        return [nan_c] * len(col)
    max_v, min_v = valid.max(), valid.min()
    styles = []
    for v in col:
        if pd.isna(v):
            styles.append(nan_c)
        elif v == max_v:
            styles.append(max_s)
        elif v == min_v:
            styles.append(min_s)
        elif v > 0:
            styles.append(pos_s)
        else:
            styles.append(neg_s)
    return styles


def make_category_chart(
    tickers: list[str],
    start_str: str | None,
    end_str: str | None,
    mode: str,
    short_names: dict[str, str],
    benchmark_ticker: str | None = None,
    height: int = 360,
) -> go.Figure | None:
    th = theme()
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
            continue
        close = load_close(ticker, start_str, end_str)
        if close.empty or len(close) < 2:
            warned.append(short_names.get(ticker, ticker))
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
        y_title = f"Excess Return vs {short_names.get(benchmark_ticker, benchmark_ticker)} (%)"
    elif mode == MODE_CUMRET:
        y_title = MODE_CUMRET
    else:
        y_title = "Rolling Volatility (% ann., 21-day)"

    fig = go.Figure()

    for ticker, series, _, orig_idx in trace_data:
        short = short_names.get(ticker, ticker)
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
            name = short_names.get(ticker, ticker)
            fmt = f"{v:+.2f}%" if (use_bench or mode == MODE_CUMRET) else f"{v:.2f}% ann."
            parts.append(f"<b>{name}</b>: {fmt}")
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
        plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=th["grid"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=th["grid"], showgrid=True, zeroline=True,
                   zerolinecolor=th["zero"], title=y_title),
        hovermode="x unified",
    )
    if mode == MODE_CUMRET or use_bench:
        fig.add_hline(y=0, line_color=th["zero"], line_dash="dot", line_width=1)

    return fig


# ---------------------------------------------------------------------------
# Data-source reference tables (shown in the 📡 popovers)
# ---------------------------------------------------------------------------

def _mkt_source_df() -> pd.DataFrame:
    return pd.DataFrame([
        ("S&P 500",       "SPX",    "^GSPC",      "Equity Index", "NYSE / S&P"),
        ("Hang Seng",     "HSI",    "^HSI",       "Equity Index", "HKEX"),
        ("KOSPI",         "KOSPI",  "^KS11",      "Equity Index", "KRX"),
        ("Nikkei 225",    "NKY",    "^N225",      "Equity Index", "TSE"),
        ("FTSE 100",      "FTSE",   "^FTSE",      "Equity Index", "LSE"),
        ("EuroStoxx 50",  "SX5E",   "^STOXX50E",  "Equity Index", "EUREX"),
        ("Bovespa",       "IBOV",   "^BVSP",      "Equity Index", "B3"),
        ("SET",           "SET",    "^SET.BK",    "Equity Index", "SET"),
        ("TAIEX",         "TWII",   "^TWII",      "Equity Index", "TWSE"),
        ("Crude Oil WTI", "WTI",    "CL=F",       "Energy Future", "NYMEX"),
        ("Gold",          "GOLD",   "GC=F",       "Metal Future", "COMEX"),
        ("Silver",        "SILV",   "SI=F",       "Metal Future", "COMEX"),
        ("Copper",        "COPPER", "HG=F",       "Metal Future", "COMEX"),
        ("Palladium",     "PALL",   "PA=F",       "Metal Future", "NYMEX"),
        ("Platinum",      "PLAT",   "PL=F",       "Metal Future", "NYMEX"),
        ("Soybean",       "SOYB",   "ZS=F",       "Agri Future",  "CBOT"),
        ("Corn",          "CORN",   "ZC=F",       "Agri Future",  "CBOT"),
        ("Bitcoin",       "BTC",    "BTC-USD",    "Crypto",       "24/7 (USD)"),
    ], columns=["Name", "DB Ticker", "yfinance Symbol", "Asset Class", "Exchange / Market"])


def _etf_source_df() -> pd.DataFrame:
    return pd.DataFrame([
        ("S&P 500",        "VUAG.L", "Equities — Regional"),
        ("Korea",          "IKOR.L", "Equities — Regional"),
        ("Taiwan",         "HTWN.L", "Equities — Regional"),
        ("Nikkei 225",     "CNKY.L", "Equities — Regional"),
        ("China",          "HCHS.L", "Equities — Regional"),
        ("India",          "IIND.L", "Equities — Regional"),
        ("Vietnam",        "XFVT.L", "Equities — Regional"),
        ("EM Islamic",     "HIES.L", "Equities — Regional"),
        ("Commodity",      "WCOB.L", "Commodities"),
        ("Copper",         "COPB.L", "Commodities"),
        ("Soybean Oil",    "SOYO.L", "Commodities"),
        ("Platinum",       "SPLT.L", "Metals & Mining"),
        ("Palladium",      "SPDM.L", "Metals & Mining"),
        ("Silver Miners",  "SILG.L", "Metals & Mining"),
        ("Gold Producers", "SPGP.L", "Metals & Mining"),
        ("Bitcoin",        "IB1T.L", "Crypto"),
    ], columns=["Name", "yfinance Symbol", "Category"])


DATASETS: dict[str, dict] = {
    "Global Markets": {
        "categories": MKT_CATEGORIES,
        "short": MKT_SHORT_NAMES,
        "key": "mkt",
        "noun": "Markets",
        "src_title": "Global Markets",
        "src_caption": (
            "All prices fetched via **Yahoo Finance** (`yfinance`), total-return "
            "adjusted where applicable (`auto_adjust=True`). Futures reflect "
            "continuous front-month contracts."
        ),
        "src_df": _mkt_source_df,
    },
    "LSE ETFs": {
        "categories": ETF_CATEGORIES,
        "short": ETF_SHORT_NAMES,
        "key": "etf",
        "noun": "ETFs",
        "src_title": "LSE ETFs",
        "src_caption": (
            "All prices fetched via **Yahoo Finance** (`yfinance`), total-return "
            "adjusted (splits + dividends). Exchange: London Stock Exchange."
        ),
        "src_df": _etf_source_df,
    },
}

# ---------------------------------------------------------------------------
# Reusable render blocks (one code path for Markets AND ETFs)
# ---------------------------------------------------------------------------

def render_hero(tickers: list[str], short: dict[str, str]) -> None:
    """Breadth + biggest 1-day movers for the active dataset."""
    snap = snapshot_frame(tuple(tickers))
    snap = snap.copy()
    snap["Name"] = snap["Ticker"].map(lambda t: short.get(t, t))
    up, down, _ = breadth(snap, "1D")
    ranked = snap.dropna(subset=["1D"]).sort_values("1D", ascending=False)

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Advancers ▲", up)
    h2.metric("Decliners ▼", down)
    if not ranked.empty:
        top, bot = ranked.iloc[0], ranked.iloc[-1]
        h3.metric(f"Top ▲ {top['Name']}", f"{top['1D']:+.2f}%")
        h4.metric(f"Top ▼ {bot['Name']}", f"{bot['1D']:+.2f}%")


def render_returns_table(categories: dict[str, list[str]], short: dict[str, str]) -> None:
    """Render the all-periods cumulative-return table for a dataset, ranked by YTD."""
    rows = []
    for cat_name, tickers in categories.items():
        for ticker in tickers:
            close = load_close(ticker, None, None)
            row: dict = {"Category": cat_name, "Name": short.get(ticker, ticker),
                         "Ticker": ticker}
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
    table_df["Rank"] = table_df["YTD"].rank(ascending=False, na_option="bottom").astype(int)
    table_df = table_df[["Rank", "Category", "Name", "Ticker"] + ret_cols]

    styler = (
        table_df.style
        .format({col: lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A" for col in ret_cols})
        .apply(highlight_col_extremes, subset=ret_cols)
        .set_properties(subset=ret_cols, **{"text-align": "right", "font-family": "monospace"})
    )
    st.markdown(render_table(styler), unsafe_allow_html=True)


def render_correlations(tickers: list[str], short: dict[str, str],
                        start_str: str | None, end_str: str | None) -> None:
    """Render a Pearson correlation heatmap of daily log-returns over the window."""
    frames = {t: load_close(t, start_str, end_str) for t in tickers}
    corr_px = pd.DataFrame(frames).dropna(how="all").ffill().dropna(how="any")
    if len(corr_px) <= 5:
        st.info("Select a longer window to compute correlations (need > 5 trading days).")
        return

    corr_rets = np.log(corr_px / corr_px.shift(1)).dropna()
    corr_mat = corr_rets.corr()
    labels = [short.get(t, t) for t in corr_mat.index]
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
    th = theme()
    fig_corr.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=11),
        xaxis=dict(tickangle=-40, side="bottom"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def render_dataset(cfg: dict, window: str, mode: str, range_label: str,
                   start_str: str | None, end_str: str | None) -> None:
    """Render one dataset end-to-end (benchmark control, hero, charts, table, heatmap).

    `cfg` is an entry from DATASETS; this single code path serves both Global
    Markets and LSE ETFs, so the two are never duplicated.
    """
    short = cfg["short"]
    categories = cfg["categories"]
    key_prefix = cfg["key"]
    all_flat = [t for tks in categories.values() for t in tks]

    # Benchmark selector + data-source popover.
    c_bench, c_src = st.columns([5, 1])
    with c_bench:
        bench_options: dict[str, str | None] = {"None": None}
        bench_options.update({short[t]: t for t in short})
        bench_label = st.selectbox(
            "vs Benchmark", list(bench_options.keys()), key=f"{key_prefix}_benchmark",
            help="Cumulative-return charts switch to excess return vs the chosen benchmark.",
        )
        benchmark_ticker: str | None = bench_options[bench_label]
    with c_src:
        st.write("")
        with st.popover("📡"):
            st.markdown(f"### Data Sources — {cfg['src_title']}")
            st.caption(cfg["src_caption"])
            st.markdown(render_table(cfg["src_df"]()), unsafe_allow_html=True)

    bench_display = f" · Benchmark: **{bench_label}**" if benchmark_ticker else ""
    st.caption(f"Displaying: **{mode}** · Window: **{window}** · {range_label}{bench_display}")

    # Hero — breadth + movers.
    render_hero(all_flat, short)
    st.divider()

    # Charts.
    tab_cat, tab_all = st.tabs(["By Category", f"All {cfg['noun']}"])
    with tab_cat:
        for cat_name, tickers in categories.items():
            st.subheader(cat_name)
            fig = make_category_chart(
                tickers, start_str, end_str, mode, short,
                benchmark_ticker=benchmark_ticker, height=chart_height(len(tickers)),
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for this category in the selected window.")
            st.divider()
    with tab_all:
        st.subheader(f"All {len(all_flat)} {cfg['noun']}")
        fig_all = make_category_chart(
            all_flat, start_str, end_str, mode, short,
            benchmark_ticker=benchmark_ticker, height=chart_height(len(all_flat)),
        )
        if fig_all:
            st.plotly_chart(fig_all, use_container_width=True)
        else:
            st.info("No data available in the selected window.")

    # Returns summary.
    st.subheader("Returns Summary — All Periods")
    st.caption("Cumulative % return across fixed windows. Rank by YTD (1 = best). "
               "Bold = best/worst per column · Click a header to sort.")
    render_returns_table(categories, short)

    # Correlations.
    st.divider()
    st.subheader("Return Correlations")
    st.caption("Pearson correlation of daily log-returns in the selected window. "
               "Green = move together · Red = move opposite · Diagonal always 1.")
    render_correlations(all_flat, short, start_str, end_str)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_freshness_header(
    "Global Market Overview",
    "10-year price history for 18 global benchmarks and 16 LSE ETFs — "
    "equity indices, commodities, agriculture, and crypto.",
    refresh_key="mkt_refresh",
)

# --- Shared controls: window, mode, formula help ---
col_win, col_mode, col_info = st.columns([6, 5, 1])
with col_win:
    window = st.radio("Time window", list(WINDOW_DAYS.keys()),
                      index=2, horizontal=True, key="mkt_window")
with col_mode:
    mode = st.radio("Display mode", [MODE_CUMRET, MODE_VOL],
                    horizontal=True, key="mkt_mode")
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

# --- Custom date inputs ---
start_str: str | None = None
end_str: str | None = None

if window == "Custom":
    c1, c2 = st.columns(2)
    raw_start = c1.text_input("Start date (YYYYMMDD)", placeholder="e.g. 20200101", key="mkt_cstart")
    raw_end   = c2.text_input("End date (YYYYMMDD) — leave blank for today", placeholder="e.g. 20241231", key="mkt_cend")

    parsed_start = parse_yyyymmdd(raw_start) if raw_start else None
    parsed_end   = parse_yyyymmdd(raw_end)   if raw_end   else date.today()

    if raw_start and parsed_start is None:
        st.error(f"Invalid start date '{raw_start}' — use YYYYMMDD format")
        st.stop()
    if raw_end and parse_yyyymmdd(raw_end) is None:
        st.error(f"Invalid end date '{raw_end}' — use YYYYMMDD format")
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

st.divider()

# --- Dataset switch: one render path for Markets OR ETFs ---
dataset_label = st.radio(
    "Dataset", list(DATASETS.keys()), horizontal=True, key="mkt_dataset",
    help="Switch the whole board between global benchmarks and LSE-listed ETFs.",
)
render_dataset(DATASETS[dataset_label], window, mode, range_label, start_str, end_str)
