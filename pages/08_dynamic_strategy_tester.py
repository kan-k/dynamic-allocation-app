from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_store import init_db
from core.optim_engine import ALL_SHORT_NAMES, log_ret, _parse, _resolve, load_prices, theme
from core.backtest_engine import (
    FREQUENCIES, run_backtest, cumret_chart, drawdown_chart, composition_chart,
)
from components.header import render_freshness_header
from components.strategy_setup import asset_selector, solver_settings, constraints_editor
from components.metrics_grid import metric_grid

# Page config is set by the app entrypoint (app.py).


def turnover_chart(weights_df: pd.DataFrame) -> go.Figure:
    """Per-rebalance one-way turnover: Σ|Δw| between consecutive rebalances."""
    th = theme()
    to = (weights_df.diff().abs().sum(axis=1) * 100).iloc[1:]
    fig = go.Figure(go.Bar(
        x=to.index, y=to.values.round(1), marker_color="#0088FE",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} pp traded<extra></extra>",
    ))
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        xaxis=dict(gridcolor=th["grid"]),
        yaxis=dict(title="Turnover (pp)", gridcolor=th["grid"]),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Page — shared setup
# ---------------------------------------------------------------------------

init_db()

render_freshness_header(
    "Dynamic Strategy Tester",
    "Backtest a periodically-rebalanced strategy: at each rebalance the solver is "
    "re-run on a trailing window, and the weights are held until the next rebalance.",
    refresh_key="dst_refresh",
)

st.subheader("1. Select Assets")
selected = asset_selector("dst_")

st.divider()
st.subheader("2. Objective & Settings")
solver_name, solver_params = solver_settings("dst_")
rf_pct = st.number_input("Risk-free rate (%, annualised)", 0.0, 20.0, 0.0, 0.25, key="dst_rf")
solver_params["rf"] = rf_pct / 100

st.divider()

if len(selected) < 2:
    st.info("Select at least 2 assets above to configure constraints and run.", icon="☝️")
    st.stop()

st.subheader("3. Constraints")
lb_map, ub_map, cat_bounds, lb_arr, ub_arr = constraints_editor(selected, "dst_")

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Backtest Setup
# ---------------------------------------------------------------------------

st.subheader("4. Backtest Setup")

today = date.today()
e1, e2, e3 = st.columns([2, 2, 3])
es_raw = e1.text_input("Evaluation start (YYYYMMDD)",
                       (today - timedelta(days=3 * 365)).strftime("%Y%m%d"), key="dst_es")
ee_raw = e2.text_input("Evaluation end (YYYYMMDD/'today')", "today", key="dst_ee")
freq = e3.radio("Rebalance frequency", FREQUENCIES, horizontal=True,
                index=FREQUENCIES.index("Monthly"), key="dst_freq")

w1, w2 = st.columns(2)
lookback = w1.number_input(
    "Lookback (trading days) — history window the solver sees at each rebalance",
    min_value=30, max_value=365, value=84, step=1, key="dst_lb",
)
halflife = w2.number_input(
    "EWM halflife (trading days)",
    min_value=5, max_value=126, value=21, step=1, key="dst_hl",
)
st.caption(
    f"Behind-the-scenes: `min_period = 2 × halflife = {2 * halflife}` observations "
    "required before a rebalance fits. Rebalances with too little history are skipped "
    "(previous weights carry over)."
)

eval_start_dt = _parse(es_raw)
eval_end_dt = _resolve(ee_raw)
if not eval_start_dt:
    st.error("Invalid evaluation start date.")
    st.stop()
if not eval_end_dt:
    st.error("Invalid evaluation end date.")
    st.stop()
if eval_end_dt < eval_start_dt:
    st.error("Evaluation end must be on or after evaluation start.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Section 5: Run
# ---------------------------------------------------------------------------

run_btn = st.button("▶ Run Backtest", type="primary", key="dst_run")

if run_btn:
    buffer_days = lookback + 30
    data_start = (eval_start_dt - timedelta(days=int(buffer_days * 1.6))).strftime("%Y-%m-%d")
    data_end = eval_end_dt.strftime("%Y-%m-%d")

    with st.spinner("Loading prices and running backtest…"):
        px = load_prices(tuple(selected), data_start, data_end, min_coverage=0.0)
        avail = [t for t in selected if t in px.columns]
        if len(avail) < 2:
            st.error("No price data found for the selected assets.")
            st.stop()
        if len(avail) < len(selected):
            missing = [ALL_SHORT_NAMES.get(t, t) for t in selected if t not in avail]
            st.warning(f"No data at all for: {missing}", icon="⚠️")
        px = px[avail]

        eval_start_ts = pd.Timestamp(eval_start_dt)
        partial = []
        for t in avail:
            first_obs = px[t].first_valid_index()
            if first_obs is not None and first_obs > eval_start_ts:
                partial.append((ALL_SHORT_NAMES.get(t, t), first_obs.date()))
        if partial:
            details = ", ".join(f"{name} (from {d})" for name, d in partial)
            st.info(f"Partial history — these assets enter the portfolio when their "
                    f"data begins: {details}", icon="ℹ️")

        lb_sub = np.array([lb_map.get(t, 0.0) / 100 for t in avail])
        ub_sub = np.array([ub_map.get(t, 30.0) / 100 for t in avail])

        run_key = hashlib.md5(json.dumps(
            [avail, solver_name, sorted(solver_params.items()),
             lb_sub.tolist(), ub_sub.tolist(), sorted(cat_bounds.items()),
             eval_start_dt.isoformat(), eval_end_dt.isoformat(),
             freq, int(lookback), int(halflife)],
            sort_keys=True, default=str,
        ).encode()).hexdigest()[:12]

        port_rets, weights_df, metrics = run_backtest(
            px=px, eval_start=eval_start_dt, eval_end=eval_end_dt, freq=freq,
            lookback=int(lookback), halflife=int(halflife),
            lb_arr=lb_sub, ub_arr=ub_sub, cat_bounds=cat_bounds,
            solver_name=solver_name, solver_params=solver_params,
        )

        # Equal-weight basket benchmark over the same evaluation window.
        eval_px = px.loc[(px.index >= eval_start_ts) & (px.index <= pd.Timestamp(eval_end_dt))]
        bench_rets = log_ret(eval_px).mean(axis=1) if not eval_px.empty else pd.Series(dtype=float)

    if port_rets.empty or weights_df.empty:
        st.error("Backtest produced no results — likely the evaluation window is "
                 "shorter than the lookback, or all rebalances were skipped for "
                 "insufficient history. Try extending the window or shortening the lookback.")
        st.stop()

    st.session_state["dst_result"] = {
        "port_rets": port_rets, "bench_rets": bench_rets,
        "weights_df": weights_df, "metrics": metrics,
        "eval_start": eval_start_dt.isoformat(), "eval_end": eval_end_dt.isoformat(),
        "freq": freq, "run_key": run_key,
    }

# ---------------------------------------------------------------------------
# Section 6: Results
# ---------------------------------------------------------------------------

if "dst_result" in st.session_state:
    r = st.session_state["dst_result"]
    metrics = r["metrics"]
    port_rets = r["port_rets"]
    weights_df = r["weights_df"]
    bench_rets = r.get("bench_rets", pd.Series(dtype=float))

    st.subheader("5. Results")
    st.caption(
        f"Strategy: **{solver_name}** · Rebalanced **{r['freq']}** · "
        f"Evaluation **{r['eval_start']} → {r['eval_end']}** · "
        f"{len(weights_df)} rebalances"
    )

    metric_grid({k: metrics.get(k, "—") for k in
                 ["Total Return", "Ann. Return", "Ann. Volatility", "Sharpe Ratio"]}, ncols=4)
    metric_grid({k: metrics.get(k, "—") for k in
                 ["Max Drawdown", "Max DD Duration", "Calmar Ratio", "# Rebalances"]}, ncols=4)

    st.markdown("**Cumulative Return** — strategy vs equal-weight basket")
    st.plotly_chart(
        cumret_chart(port_rets, benchmark=bench_rets, benchmark_name="Equal-weight"),
        use_container_width=True,
    )

    st.markdown("**Drawdown**")
    st.plotly_chart(drawdown_chart(port_rets), use_container_width=True)

    st.markdown("**Portfolio Composition Over Time**")
    st.caption("Stacked area shows weight allocated to each asset between rebalance "
               "dates. Composition changes only at rebalance points.")
    st.plotly_chart(composition_chart(weights_df), use_container_width=True)

    st.markdown("**Rebalance Turnover**")
    st.caption("Percentage points of the book traded at each rebalance (Σ|Δweight|). "
               "Higher bars = more trading cost drag in live use.")
    st.plotly_chart(turnover_chart(weights_df), use_container_width=True)
