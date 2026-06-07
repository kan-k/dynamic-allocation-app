from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_store import init_db
from core.optim_engine import ALL_SHORT_NAMES, ALL_TICKERS, theme, render_table
from core.market_metrics import load_close, snapshot_frame, breadth, regime_snapshot
from components.header import render_freshness_header
from components.metrics_grid import metric_grid

# Page config is owned by app.py (the st.navigation entrypoint).

init_db()

render_freshness_header(
    "Today",
    "At-a-glance read of the global board — breadth, the day's movers, and a "
    "heuristic regime snapshot. All figures computed from cached prices.",
    refresh_key="today_refresh",
)

snap = snapshot_frame(tuple(ALL_TICKERS))
snap["Name"] = snap["Ticker"].map(lambda t: ALL_SHORT_NAMES.get(t, t))

if snap["1D"].dropna().empty and snap["YTD"].dropna().empty:
    st.info("No price data yet. Open the ↻ menu above to download benchmarks + ETFs.")
    st.stop()

# ---------------------------------------------------------------------------
# Breadth hero
# ---------------------------------------------------------------------------

up, down, flat = breadth(snap, "1D")
total = up + down + flat

st.subheader("Market Breadth — Today")
bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Advancers ▲", up)
bc2.metric("Decliners ▼", down)
bc3.metric("Unchanged ▬", flat)
bc4.metric("Universe", total)

# Thin advancers-vs-decliners proportion bar (colour paired with ▲/▼ labels).
if up + down > 0:
    th = theme()
    bar = go.Figure()
    bar.add_trace(go.Bar(x=[up], y=["Breadth"], orientation="h", name="Advancers ▲",
                         marker_color="#00C49F", hovertemplate="▲ %{x}<extra></extra>"))
    bar.add_trace(go.Bar(x=[down], y=["Breadth"], orientation="h", name="Decliners ▼",
                         marker_color="#EF476F", hovertemplate="▼ %{x}<extra></extra>"))
    bar.update_layout(
        barmode="stack", height=80, margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    st.plotly_chart(bar, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Top movers (1D and YTD)
# ---------------------------------------------------------------------------

POS = "#0f9d76"
NEG = "#d1495b"


def _best_worst(df: pd.DataFrame, col: str, n: int = 6) -> pd.DataFrame:
    """Top `n` and bottom `n` rows by `col` (deduplicated), best first."""
    ranked = df.dropna(subset=[col]).sort_values(col, ascending=False)
    return pd.concat([ranked.head(n), ranked.tail(n)]).drop_duplicates("Ticker")


def _movers_table(df: pd.DataFrame, sort_col: str) -> str:
    """Return a themed HTML table of movers, return columns coloured by sign."""
    sub = df.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)
    cols = ["Name", "Last", "1D", "YTD"]
    view = sub[cols].copy()

    def _fmt_ret(v: float) -> str:
        return f"{v:+.2f}%" if pd.notna(v) else "N/A"

    def _color(v: float) -> str:
        if pd.isna(v):
            return "color:#888"
        return f"color:{POS}" if v > 0 else (f"color:{NEG}" if v < 0 else "")

    styler = (
        view.style
        .format({"Last": "{:,.2f}", "1D": _fmt_ret, "YTD": _fmt_ret})
        .map(_color, subset=["1D", "YTD"])
        .set_properties(subset=["Last", "1D", "YTD"],
                        **{"text-align": "right", "font-family": "monospace"})
    )
    return render_table(styler)


st.subheader("Top Movers")
mc1, mc2 = st.columns(2)
with mc1:
    st.markdown("**Best & worst — 1 Day**")
    st.markdown(_movers_table(_best_worst(snap, "1D"), "1D"), unsafe_allow_html=True)
with mc2:
    st.markdown("**Best & worst — YTD**")
    st.markdown(_movers_table(_best_worst(snap, "YTD"), "YTD"), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Sparkline row — key benchmarks (90-day trend)
# ---------------------------------------------------------------------------

st.subheader("Key Benchmarks — 90-Day Trend")
SPARK_TICKERS = ["SPX", "SET", "GOLD", "WTI", "BTC"]
spark_cols = st.columns(len(SPARK_TICKERS))
start_90 = (pd.Timestamp.today() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

for col, t in zip(spark_cols, SPARK_TICKERS):
    series = load_close(t, start_90, None)
    name = ALL_SHORT_NAMES.get(t, t)
    if series.empty or len(series) < 2:
        col.caption(f"**{name}** — no data")
        continue
    chg = float((series.iloc[-1] / series.iloc[0] - 1) * 100)
    line_color = "#00C49F" if chg >= 0 else "#EF476F"
    spark = go.Figure(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=line_color, width=1.6),
        hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>",
    ))
    spark.update_layout(
        height=70, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    arrow = "▲" if chg >= 0 else "▼"
    col.caption(f"**{name}**  {arrow} {chg:+.1f}%")
    col.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# Regime snapshot (heuristic) + quick links
# ---------------------------------------------------------------------------

st.subheader("Regime Snapshot")
st.caption("Heuristic read from 1-month price trends — a summary of what prices "
           "did, not investment advice.")
metric_grid(regime_snapshot(snap), ncols=3)

st.divider()
st.subheader("Go Deeper")
ql1, ql2, ql3 = st.columns(3)
with ql1:
    st.page_link("pages/06_market_overview.py", label="📊 Market Overview", icon=None)
with ql2:
    st.page_link("pages/07_dynamic_allocation.py", label="⚖️ Dynamic Allocation", icon=None)
with ql3:
    st.page_link("pages/08_dynamic_strategy_tester.py", label="🧪 Strategy Tester", icon=None)
