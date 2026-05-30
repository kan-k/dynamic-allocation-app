from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from core.data_store import init_db
from core.optim_engine import (
    UNIVERSE, ALL_SHORT_NAMES, TICKER_TO_CAT, ALL_TICKERS, DEFAULT_BOUNDS,
    SOLVER_NAMES,
    _parse, _resolve, load_prices,
)
from core.backtest_engine import (
    FREQUENCIES, run_backtest, cumret_chart, composition_chart,
)

# Page config is set by the app entrypoint (app.py).

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

init_db()

st.title("Dynamic Strategy Tester")
st.caption(
    "Backtest a periodically-rebalanced dynamic strategy: at each rebalance date "
    "the solver is re-run on a trailing window of returns, and the resulting "
    "weights are held until the next rebalance."
)

# ---------------------------------------------------------------------------
# Section 1: Select Assets
# ---------------------------------------------------------------------------

st.subheader("1. Select Assets")

for group_name, categories in UNIVERSE.items():
    group_tickers = [t for tks in categories.values() for t in tks]
    with st.expander(f"**{group_name}** ({len(group_tickers)} assets)", expanded=True):
        bc1, bc2, _ = st.columns([1, 1, 8])

        def _sel(gt=group_tickers):
            for t in gt:
                st.session_state[f"dst_sel_{t}"] = True

        def _clr(gt=group_tickers):
            for t in gt:
                st.session_state[f"dst_sel_{t}"] = False

        bc1.button("Select all", key=f"dst_sa_{group_name}", on_click=_sel)
        bc2.button("Clear",      key=f"dst_ca_{group_name}", on_click=_clr)
        for cat_name, tickers in categories.items():
            st.markdown(f"**{cat_name}**")
            cols = st.columns(min(len(tickers), 4))
            for i, t in enumerate(tickers):
                cols[i % len(cols)].checkbox(
                    ALL_SHORT_NAMES.get(t, t),
                    key=f"dst_sel_{t}",
                    value=st.session_state.get(f"dst_sel_{t}", False),
                )

selected: list[str] = [t for t in ALL_TICKERS if st.session_state.get(f"dst_sel_{t}", False)]
if selected:
    st.caption(f"**{len(selected)} selected:** " +
               ", ".join(ALL_SHORT_NAMES.get(t, t) for t in selected))
else:
    st.caption("No assets selected.")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Solver & Settings
# ---------------------------------------------------------------------------

st.subheader("2. Solver & Settings")

s_col, p_col = st.columns([3, 2])
with s_col:
    solver_name = st.radio("Objective", SOLVER_NAMES, horizontal=True, key="dst_solver")

solver_params: dict = {}
with p_col:
    st.write("")
    if solver_name == "Min Volatility":
        v = st.number_input("Min annual return (%)", 0.0, 50.0, 5.0, 0.5, key="dst_minr")
        solver_params["min_ret"] = v / 100
    elif solver_name == "Max Return":
        v = st.number_input("Max annual volatility (%)", 1.0, 100.0, 15.0, 0.5, key="dst_maxv")
        solver_params["max_vol"] = v / 100
    elif solver_name == "Min Drawdown":
        dc1, dc2 = st.columns(2)
        solver_params["dd_lookback"] = dc1.number_input(
            "Lookback (days)", 60, 756, 252, 21, key="dst_ddlb")
        cap = dc2.number_input("Max DD cap (%, 0=off)", 0.0, 100.0, 0.0, 1.0, key="dst_ddcap")
        if cap > 0:
            solver_params["max_dd_cap"] = cap / 100

rf_pct = st.number_input("Risk-free rate (%, annualised)", 0.0, 20.0, 0.0, 0.25, key="dst_rf")
solver_params["rf"] = rf_pct / 100

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Constraints
# ---------------------------------------------------------------------------

if len(selected) < 2:
    st.info("Select at least 2 assets above to configure constraints.", icon="☝️")
    st.stop()

st.subheader("3. Constraints")
st.caption("▶ rows = category-level combined bounds · indented rows = individual asset bounds. All values in %.")

sel_cats: dict[str, list[str]] = {}
for t in selected:
    c = TICKER_TO_CAT.get(t, "Other")
    sel_cats.setdefault(c, []).append(t)

con_rows = []
for cat, tickers in sel_cats.items():
    con_rows.append({"Type": "category", "Name": f"▶ {cat}", "Ticker": "—",
                     "Min %": 0.0, "Max %": 100.0})
    for t in tickers:
        lo, hi = DEFAULT_BOUNDS.get(t, (0.0, 30.0))
        con_rows.append({"Type": "asset", "Name": f"  {ALL_SHORT_NAMES.get(t, t)}",
                         "Ticker": t, "Min %": float(lo), "Max %": float(hi)})

sel_hash = hashlib.md5("_".join(selected).encode()).hexdigest()[:8]
edited = st.data_editor(
    pd.DataFrame(con_rows),
    key=f"dst_con_{sel_hash}",
    column_config={
        "Type":   st.column_config.TextColumn("Type",   disabled=True, width="small"),
        "Name":   st.column_config.TextColumn("Name",   disabled=True, width="medium"),
        "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
        "Min %":  st.column_config.NumberColumn("Min %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
        "Max %":  st.column_config.NumberColumn("Max %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
    },
    hide_index=True, use_container_width=True,
)

cat_bounds: dict[str, tuple[float, float]] = {}
lb_map: dict[str, float] = {}
ub_map: dict[str, float] = {}
for _, row in edited.iterrows():
    if row["Type"] == "category":
        cat_raw = row["Name"].replace("▶ ", "", 1).strip()
        cat_bounds[cat_raw] = (float(row["Min %"]), float(row["Max %"]))
    else:
        lb_map[row["Ticker"]] = float(row["Min %"])
        ub_map[row["Ticker"]] = float(row["Max %"])

lb_arr = np.array([lb_map.get(t, 0.0) / 100 for t in selected])
ub_arr = np.array([ub_map.get(t, 30.0) / 100 for t in selected])

if lb_arr.sum() > 1.001:
    st.error(f"Sum of minimum weights ({lb_arr.sum()*100:.1f}%) exceeds 100%. Reduce some minimums.")
    st.stop()
if ub_arr.sum() < 0.999:
    st.error(f"Sum of maximum weights ({ub_arr.sum()*100:.1f}%) is below 100%. Increase some maximums.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Backtest Setup
# ---------------------------------------------------------------------------

st.subheader("4. Backtest Setup")

today = date.today()
e1, e2, e3 = st.columns([2, 2, 3])
es_raw = e1.text_input(
    "Evaluation start (YYYYMMDD)",
    (today - timedelta(days=3 * 365)).strftime("%Y%m%d"),
    key="dst_es",
)
ee_raw = e2.text_input(
    "Evaluation end (YYYYMMDD/'today')",
    "today",
    key="dst_ee",
)
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
    # Load prices with a buffer before eval_start so the first rebalance has history.
    buffer_days = lookback + 30
    data_start = (eval_start_dt - timedelta(days=int(buffer_days * 1.6))).strftime("%Y-%m-%d")
    data_end = eval_end_dt.strftime("%Y-%m-%d")

    with st.spinner("Loading prices and running backtest…"):
        # min_coverage=0.0 keeps partial-history tickers (e.g. ETFs that
        # started trading after eval_start). The backtest engine decides
        # per-rebalance which assets have enough history to include.
        px = load_prices(tuple(selected), data_start, data_end, min_coverage=0.0)
        avail = [t for t in selected if t in px.columns]
        if len(avail) < 2:
            st.error("No price data found for the selected assets.")
            st.stop()
        if len(avail) < len(selected):
            missing = [ALL_SHORT_NAMES.get(t, t) for t in selected if t not in avail]
            st.warning(f"No data at all for: {missing}", icon="⚠️")
        px = px[avail]

        # Report partial-history tickers (real data shorter than eval window)
        eval_start_ts = pd.Timestamp(eval_start_dt)
        partial = []
        for t in avail:
            first_obs = px[t].first_valid_index()
            if first_obs is not None and first_obs > eval_start_ts:
                partial.append((ALL_SHORT_NAMES.get(t, t), first_obs.date()))
        if partial:
            details = ", ".join(f"{name} (from {d})" for name, d in partial)
            st.info(
                f"Partial history — these assets enter the portfolio when their "
                f"data begins: {details}",
                icon="ℹ️",
            )

        # Subset lb/ub to available tickers
        lb_sub = np.array([lb_map.get(t, 0.0) / 100 for t in avail])
        ub_sub = np.array([ub_map.get(t, 30.0) / 100 for t in avail])

        # Hash for cache key (note: load_prices already cached upstream)
        run_key = hashlib.md5(json.dumps(
            [avail, solver_name, sorted(solver_params.items()),
             lb_sub.tolist(), ub_sub.tolist(),
             sorted(cat_bounds.items()),
             eval_start_dt.isoformat(), eval_end_dt.isoformat(),
             freq, int(lookback), int(halflife)],
            sort_keys=True, default=str,
        ).encode()).hexdigest()[:12]

        port_rets, weights_df, metrics = run_backtest(
            px=px,
            eval_start=eval_start_dt, eval_end=eval_end_dt,
            freq=freq,
            lookback=int(lookback), halflife=int(halflife),
            lb_arr=lb_sub, ub_arr=ub_sub,
            cat_bounds=cat_bounds,
            solver_name=solver_name, solver_params=solver_params,
        )

    if port_rets.empty or weights_df.empty:
        st.error(
            "Backtest produced no results — likely the evaluation window is "
            "shorter than the lookback, or all rebalances were skipped for "
            "insufficient history. Try extending the window or shortening the lookback."
        )
        st.stop()

    st.session_state["dst_result"] = {
        "port_rets": port_rets,
        "weights_df": weights_df,
        "metrics": metrics,
        "eval_start": eval_start_dt.isoformat(),
        "eval_end": eval_end_dt.isoformat(),
        "freq": freq,
        "run_key": run_key,
    }

# ---------------------------------------------------------------------------
# Section 6: Results
# ---------------------------------------------------------------------------

if "dst_result" in st.session_state:
    r = st.session_state["dst_result"]
    metrics = r["metrics"]
    port_rets = r["port_rets"]
    weights_df = r["weights_df"]

    st.subheader("5. Results")
    st.caption(
        f"Strategy: **{solver_name}** · Rebalanced **{r['freq']}** · "
        f"Evaluation **{r['eval_start']} → {r['eval_end']}** · "
        f"{len(weights_df)} rebalances"
    )

    # Performance metrics — top row
    mc = st.columns(4)
    for i, key in enumerate(["Total Return", "Ann. Return", "Ann. Volatility", "Sharpe Ratio"]):
        mc[i].metric(key, metrics.get(key, "—"))
    mc2 = st.columns(4)
    for i, key in enumerate(["Max Drawdown", "Max DD Duration", "Calmar Ratio", "# Rebalances"]):
        mc2[i].metric(key, metrics.get(key, "—"))

    st.markdown("**Cumulative Strategy Return**")
    st.plotly_chart(cumret_chart(port_rets), use_container_width=True)

    st.markdown("**Portfolio Composition Over Time**")
    st.caption(
        "Stacked area shows weight allocated to each asset between rebalance dates. "
        "Composition changes only at rebalance points."
    )
    st.plotly_chart(composition_chart(weights_df), use_container_width=True)
