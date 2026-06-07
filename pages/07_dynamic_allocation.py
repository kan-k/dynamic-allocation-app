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
from core.optim_engine import (
    ALL_SHORT_NAMES, COLORS,
    _parse, _resolve, load_prices, log_ret, ewm_stats,
    run_solver, portfolio_metrics, ewm_portfolio_metrics, theme, render_table,
)
from core.backtest_engine import cumret_chart, drawdown_chart
from components.header import render_freshness_header
from components.strategy_setup import asset_selector, solver_settings, constraints_editor
from components.metrics_grid import metric_grid
from components.llm_interpret import interpretation_block

# Page config is set by the app entrypoint (app.py).

# ---------------------------------------------------------------------------
# Rolling allocation (weekly steps, cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=7200, show_spinner=False)
def compute_rolling(
    tickers: tuple[str, ...],
    solver_name: str,
    solver_params_t: tuple,    # sorted tuple of items — hashable
    lb_t: tuple[float, ...],
    ub_t: tuple[float, ...],
    cat_bounds_t: tuple,       # sorted tuple of items
    lookback: int,
    ewm_hl: int,
    data_start: str,
    data_end: str,
) -> pd.DataFrame:
    lb = np.array(lb_t)
    ub = np.array(ub_t)
    sp = dict(solver_params_t)
    cb = dict(cat_bounds_t)

    px = load_prices(tickers, data_start, data_end)
    avail = [t for t in tickers if t in px.columns]
    if len(avail) < 2:
        return pd.DataFrame()
    px = px[avail]
    lb = lb[[i for i, t in enumerate(tickers) if t in avail]]
    ub = ub[[i for i, t in enumerate(tickers) if t in avail]]

    eligible = px.index[lookback:]
    step_dates = eligible[::5]
    hist: dict = {}
    for t_end in step_dates:
        win = px.loc[:t_end].tail(lookback)
        r = log_ret(win)
        if len(r) < max(10, ewm_hl):
            continue
        mu, Sigma = ewm_stats(r, ewm_hl)
        w, _ = run_solver(mu, Sigma, r, lb, ub, cb, solver_name, sp)
        hist[t_end] = w
    return pd.DataFrame(hist).T

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def alloc_lines(alloc_df: pd.DataFrame) -> go.Figure:
    th = theme()
    ordered = alloc_df.mean().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, t in enumerate(ordered):
        if t not in alloc_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=alloc_df.index,
            y=(alloc_df[t] * 100).round(2),
            name=ALL_SHORT_NAMES.get(t, t),
            line=dict(color=COLORS[i % len(COLORS)], width=1.8),
            hoverinfo="skip",
        ))

    def _hover(row: pd.Series) -> str:
        ranked = row.dropna().sort_values(ascending=False)
        return "<br>".join(
            f"<b>{ALL_SHORT_NAMES.get(t,t)}</b>: {v*100:.1f}%"
            for t, v in ranked.items() if v >= 0.001
        )

    fig.add_trace(go.Scatter(
        x=alloc_df.index, y=[0.0] * len(alloc_df),
        mode="markers", marker=dict(opacity=0, size=1),
        hovertemplate="%{text}<extra></extra>",
        text=alloc_df.apply(_hover, axis=1).tolist(),
        showlegend=False, name="",
    ))
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        yaxis=dict(title="Allocation (%)", gridcolor=th["grid"], ticksuffix="%"),
        xaxis=dict(gridcolor=th["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def weights_bar(weights: pd.Series) -> go.Figure:
    th = theme()
    ws = weights.sort_values(ascending=True)
    labels = [ALL_SHORT_NAMES.get(t, t) for t in ws.index]
    fig = go.Figure(go.Bar(
        x=ws.values * 100, y=labels, orientation="h",
        marker_color=["#00C49F" if v >= 0.001 else th["zero"] for v in ws],
        text=[f"{v*100:.1f}%" for v in ws], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(250, 40 + len(ws) * 28),
        margin=dict(l=0, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        xaxis=dict(title="Weight (%)", gridcolor=th["grid"], ticksuffix="%"),
        yaxis=dict(gridcolor=th["grid"]),
    )
    return fig

# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------

def _dyn_prompt(alloc_df: pd.DataFrame) -> str:
    n = len(alloc_df); thirds = max(1, n // 3)
    def avg(df): return (df.mean() * 100).sort_values(ascending=False)
    def fmt(s): return "\n".join(f"  {ALL_SHORT_NAMES.get(t,t)}: {v:.1f}%" for t, v in s.items() if v > 0.5)
    return (
        f"A dynamic portfolio optimiser was run weekly from "
        f"{alloc_df.index[0].date()} to {alloc_df.index[-1].date()}.\n\n"
        f"Early period average allocation:\n{fmt(avg(alloc_df.iloc[:thirds]))}\n\n"
        f"Mid period:\n{fmt(avg(alloc_df.iloc[thirds:2*thirds]))}\n\n"
        f"Recent period:\n{fmt(avg(alloc_df.iloc[2*thirds:]))}\n\n"
        "Interpret in 400–600 words: how did allocations shift and why? "
        "Use macro and market regime context. End with 2–3 forward-looking points."
    )


def _bt_prompt(weights: pd.Series, metrics: dict, alloc_d: str, s: str, e: str) -> str:
    alloc = "\n".join(f"  {ALL_SHORT_NAMES.get(t,t)} ({t}): {v*100:.1f}%"
                      for t, v in weights.sort_values(ascending=False).items() if v > 0.001)
    stats = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
    return (
        f"Static portfolio allocated on {alloc_d}, backtested {s}→{e}.\n\n"
        f"Allocation:\n{alloc}\n\nStatistics:\n{stats}\n\n"
        "Interpret in 400–500 words: quality of performance, likely return drivers, "
        "current risks, and whether this is an attractive allocation now."
    )

# ---------------------------------------------------------------------------
# Page — shared setup
# ---------------------------------------------------------------------------

init_db()

render_freshness_header(
    "Dynamic Allocation",
    "Pick assets, an objective, and constraints once below — then run a single "
    "optimisation, a rolling dynamic chart, or a static backtest in the tabs.",
    refresh_key="da_refresh",
)

today = date.today()

st.subheader("1. Select Assets")
selected = asset_selector("da_")

st.divider()
st.subheader("2. Objective & Settings")
solver_name, solver_params = solver_settings("da_")

with st.expander("Advanced settings"):
    ac1, ac2, ac3, ac4 = st.columns(4)
    ts_raw = ac1.text_input("Train start (YYYYMMDD)",
                            (today - timedelta(days=5 * 365)).strftime("%Y%m%d"), key="da_ts")
    te_raw = ac2.text_input("Train end (YYYYMMDD/'today')", "today", key="da_te")
    ewm_hl = ac3.slider("EWM halflife (days)", 21, 252, 63, 7, key="da_ewm")
    rf_pct = ac4.number_input("Risk-free rate (%)", 0.0, 20.0, 0.0, 0.25, key="da_rf_pct")
    solver_params["rf"] = rf_pct / 100

train_start_dt = _parse(ts_raw)
train_end_dt   = _resolve(te_raw)
if not train_start_dt:
    st.error("Invalid train start date.")
    st.stop()
if not train_end_dt:
    st.error("Invalid train end date.")
    st.stop()
ts = train_start_dt.strftime("%Y-%m-%d")
te = train_end_dt.strftime("%Y-%m-%d")

st.divider()

if len(selected) < 2:
    st.info("Select at least 2 assets above to configure constraints and run.", icon="☝️")
    st.stop()

st.subheader("3. Constraints")
lb_map, ub_map, cat_bounds, lb_arr, ub_arr = constraints_editor(selected, "da_")

st.divider()

# ---------------------------------------------------------------------------
# Workflows — tabs over the shared setup
# ---------------------------------------------------------------------------

tab_single, tab_dyn, tab_bt = st.tabs(
    ["① Single Optimisation", "② Dynamic (rolling)", "③ Static Backtest"]
)

# --- Tab 1: Single optimisation ------------------------------------------------
with tab_single:
    st.caption("Optimise once over the training window and inspect the resulting weights.")
    if st.button("▶ Run Optimisation", type="primary", key="da_run"):
        with st.spinner("Loading prices and running optimiser…"):
            px = load_prices(tuple(selected), ts, te)
            avail = [t for t in selected if t in px.columns]
            if len(avail) < 2:
                st.error("Not enough price data in training window.")
                st.stop()
            if len(avail) < len(selected):
                st.warning(f"No data for: {[ALL_SHORT_NAMES.get(t,t) for t in selected if t not in avail]}", icon="⚠️")
            px = px[avail]
            r = log_ret(px)
            if len(r) < max(10, ewm_hl):
                st.error("Too few return observations. Extend training window.")
                st.stop()
            mu, Sigma = ewm_stats(r, ewm_hl)
            lb_s = np.array([lb_map.get(t, 0.0) / 100 for t in avail])
            ub_s = np.array([ub_map.get(t, 30.0) / 100 for t in avail])
            w, ok = run_solver(mu, Sigma, r, lb_s, ub_s, cat_bounds, solver_name, solver_params)
            if not ok:
                st.warning("Optimiser did not fully converge.", icon="⚠️")
            st.session_state["da_w"]      = w
            st.session_state["da_px"]     = px
            st.session_state["da_mu"]     = mu
            st.session_state["da_Sigma"]  = Sigma
            st.session_state["da_rf"]     = float(solver_params.get("rf", 0.0))
            st.session_state["da_ewm_hl"] = ewm_hl
            for k in ("da_rolling", "da_rolling_hash", "da_bt", "da_dyn_llm", "da_bt_llm"):
                st.session_state.pop(k, None)

    if "da_w" in st.session_state:
        w   = st.session_state["da_w"]
        px_ = st.session_state["da_px"]
        m   = portfolio_metrics(w, px_)
        if m:
            st.markdown("**Realized statistics (equal-weighted over training window)**")
            metric_grid(m, ncols=3)
            em = ewm_portfolio_metrics(
                w, st.session_state.get("da_mu"), st.session_state.get("da_Sigma"),
                st.session_state.get("da_rf", 0.0),
            ) if "da_mu" in st.session_state else {}
            if em:
                st.caption(
                    "Optimiser view — EWM-weighted estimates the solver actually saw "
                    f"(halflife = {st.session_state.get('da_ewm_hl', '?')} days)."
                )
                metric_grid(em, ncols=3)

        cc1, cc2 = st.columns([3, 2])
        with cc1:
            st.plotly_chart(weights_bar(w), use_container_width=True)
        with cc2:
            st.markdown(render_table(pd.DataFrame({
                "Asset":  [ALL_SHORT_NAMES.get(t, t) for t in w.index],
                "Ticker": w.index.tolist(),
                "Weight": [f"{v*100:.2f}%" for v in w],
            })), unsafe_allow_html=True)

        # Correlation heatmap of allocated assets (weight > 0.5%)
        nz = [t for t in w.index if float(w[t]) > 0.005]
        if len(nz) < 2:
            st.info("Fewer than 2 assets with weight > 0.5% — no correlation to show.")
        else:
            corr = log_ret(px_[nz]).corr()
            labels = [ALL_SHORT_NAMES.get(t, t) for t in corr.index]
            corr.index = labels
            corr.columns = labels
            mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
            z = corr.values.copy()
            z[mask] = np.nan
            text = corr.round(2).astype(str).values
            text[mask] = ""
            st.markdown("**Asset Correlations (allocated only)**")
            st.caption("Pearson correlation of daily log-returns over the training "
                       "window; only assets with weight > 0.5% shown.")
            fig_corr = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels, colorscale="RdYlGn", zmin=-1, zmax=1,
                text=text, texttemplate="%{text}", textfont=dict(size=10),
                hoverongaps=False,
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b>: %{z:.2f}<extra></extra>",
            ))
            th_c = theme()
            fig_corr.update_layout(
                height=480, margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th_c["bg"],
                font=dict(color=th_c["font"], size=11),
                xaxis=dict(tickangle=-40, side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# --- Tab 2: Dynamic (rolling) --------------------------------------------------
with tab_dyn:
    roll_lb = st.slider("Rolling lookback (days)", 63, 756, 252, 21, key="da_rolllb")
    st.caption(
        f"Optimiser runs every 5 trading days with a {roll_lb}-day lookback. "
        "Stacked lines show weight evolution; hover for ranked allocations on any date."
    )
    if st.button("▶ Run Dynamic Chart", key="da_rundyn"):
        sp_t = tuple(sorted(solver_params.items()))
        cb_t = tuple(sorted(cat_bounds.items()))
        lb_t = tuple(lb_arr.tolist())
        ub_t = tuple(ub_arr.tolist())
        phash = hashlib.md5(json.dumps(
            [list(selected), solver_name, sp_t, lb_t, ub_t, cb_t, roll_lb, ewm_hl],
            sort_keys=True, default=str
        ).encode()).hexdigest()[:12]

        if st.session_state.get("da_rolling_hash") != phash:
            n_est = max(0, (len(load_prices(
                tuple(selected),
                (today - timedelta(days=10 * 365)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d"),
            )) - roll_lb) // 5)
            with st.spinner(f"Computing ~{n_est} optimisation steps — please wait…"):
                alloc_df = compute_rolling(
                    tuple(selected), solver_name, sp_t, lb_t, ub_t, cb_t, roll_lb, ewm_hl,
                    (today - timedelta(days=10 * 365)).strftime("%Y-%m-%d"),
                    today.strftime("%Y-%m-%d"),
                )
            st.session_state["da_rolling"]      = alloc_df
            st.session_state["da_rolling_hash"] = phash
            st.session_state.pop("da_dyn_llm", None)

    if "da_rolling" in st.session_state:
        alloc_df = st.session_state["da_rolling"]
        if not alloc_df.empty:
            st.plotly_chart(alloc_lines(alloc_df), use_container_width=True)
            st.markdown("---")
            interpretation_block(lambda: _dyn_prompt(alloc_df), "da_dyn_llm", "da_d")
        else:
            st.info("No allocation data returned — try more assets or a shorter lookback.")

# --- Tab 3: Static backtest ----------------------------------------------------
with tab_bt:
    st.caption("Compute weights at a chosen date, then track that fixed allocation "
               "over any period (benchmarked vs an equal-weight basket).")
    bc1, bc2, bc3 = st.columns(3)
    ad_raw = bc1.text_input("Allocation date (YYYYMMDD/'today')", "today", key="da_ad")
    bs_raw = bc2.text_input("Backtest start (YYYYMMDD)", ts_raw, key="da_bs")
    be_raw = bc3.text_input("Backtest end (YYYYMMDD/'today')", "today", key="da_be")

    if st.button("▶ Run Backtest", key="da_runbt"):
        alloc_dt = _resolve(ad_raw)
        bt_s_dt  = _parse(bs_raw)
        bt_e_dt  = _resolve(be_raw)
        if not alloc_dt: st.error("Invalid allocation date."); st.stop()
        if not bt_s_dt:  st.error("Invalid backtest start."); st.stop()
        if not bt_e_dt:  st.error("Invalid backtest end."); st.stop()
        if bt_e_dt < bt_s_dt: st.error("End must be after start."); st.stop()

        with st.spinner("Computing allocation and backtest…"):
            # Re-optimise at alloc_dt
            alloc_win_s = (alloc_dt - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
            alloc_win_e = alloc_dt.strftime("%Y-%m-%d")
            px_a = load_prices(tuple(selected), alloc_win_s, alloc_win_e)
            avail_a = [t for t in selected if t in px_a.columns]
            lb_a = np.array([lb_map.get(t, 0.0) / 100 for t in avail_a])
            ub_a = np.array([ub_map.get(t, 30.0) / 100 for t in avail_a])
            if len(avail_a) >= 2:
                ra = log_ret(px_a[avail_a])
                if len(ra) >= max(10, ewm_hl):
                    mu_a, Sg_a = ewm_stats(ra, ewm_hl)
                    bt_w, _ = run_solver(mu_a, Sg_a, ra, lb_a, ub_a, cat_bounds,
                                         solver_name, solver_params)
                else:
                    bt_w = pd.Series(1 / len(avail_a), index=avail_a)
            else:
                bt_w = pd.Series(1 / len(selected), index=selected)

            # Load backtest prices
            px_bt = load_prices(
                tuple(bt_w.index.tolist()),
                bt_s_dt.strftime("%Y-%m-%d"), bt_e_dt.strftime("%Y-%m-%d"),
            )
            avail_bt = [t for t in bt_w.index if t in px_bt.columns]
            if not avail_bt: st.error("No price data in backtest period."); st.stop()
            px_bt = px_bt[avail_bt].dropna()
            w_bt  = bt_w[avail_bt] / bt_w[avail_bt].sum()
            daily = log_ret(px_bt)
            rets_bt = daily @ w_bt
            bench_rets = daily.mean(axis=1)   # equal-weight basket benchmark
            met_bt = portfolio_metrics(bt_w, px_bt)

            st.session_state["da_bt"] = {
                "rets": rets_bt, "bench_rets": bench_rets,
                "weights": bt_w, "metrics": met_bt,
                "alloc_date": alloc_dt.strftime("%Y%m%d"),
                "bt_start":   bt_s_dt.strftime("%Y%m%d"),
                "bt_end":     bt_e_dt.strftime("%Y%m%d"),
            }
            st.session_state.pop("da_bt_llm", None)

    if "da_bt" in st.session_state:
        bt = st.session_state["da_bt"]
        st.caption(f"**Allocated:** {bt['alloc_date']} · "
                   f"**Period:** {bt['bt_start']} → {bt['bt_end']}")
        metric_grid(bt["metrics"], ncols=3)

        st.markdown("**Cumulative Return** — strategy vs equal-weight basket")
        st.plotly_chart(
            cumret_chart(bt["rets"], benchmark=bt["bench_rets"], benchmark_name="Equal-weight"),
            use_container_width=True,
        )
        st.markdown("**Drawdown**")
        st.plotly_chart(drawdown_chart(bt["rets"]), use_container_width=True)

        st.markdown(render_table(pd.DataFrame({
            "Asset":  [ALL_SHORT_NAMES.get(t, t) for t in bt["weights"].index],
            "Weight": [f"{v*100:.1f}%" for v in bt["weights"]],
        })), unsafe_allow_html=True)

        st.markdown("---")
        interpretation_block(
            lambda: _bt_prompt(bt["weights"], bt["metrics"],
                               bt["alloc_date"], bt["bt_start"], bt["bt_end"]),
            "da_bt_llm", "da_b",
        )
