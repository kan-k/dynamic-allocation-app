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
    UNIVERSE, ALL_SHORT_NAMES, TICKER_TO_CAT, ALL_TICKERS, DEFAULT_BOUNDS,
    COLORS, DARK_BG, SOLVER_NAMES,
    _parse, _resolve, load_prices, log_ret, ewm_stats,
    run_solver, portfolio_metrics, ewm_portfolio_metrics,
)

try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None

try:
    import anthropic as _ant
except ImportError:
    _ant = None

# Page config is set by the app entrypoint (app.py) — under st.navigation only
# the entrypoint may call st.set_page_config.

# ---------------------------------------------------------------------------
# Rolling allocation (weekly steps, cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=7200, show_spinner=False)
def compute_rolling(
    tickers: tuple[str,...],
    solver_name: str,
    solver_params_t: tuple,    # sorted tuple of items — hashable
    lb_t: tuple[float,...],
    ub_t: tuple[float,...],
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=DARK_BG,
        font=dict(color="#FAFAFA", size=12),
        yaxis=dict(title="Allocation (%)", gridcolor="#2a2d3a", ticksuffix="%"),
        xaxis=dict(gridcolor="#2a2d3a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def backtest_chart(port_rets: pd.Series) -> go.Figure:
    cum = ((np.exp(port_rets.cumsum()) - 1) * 100).round(3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values, name="Portfolio (cum. return)",
        line=dict(color="#4ade80", width=2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:+.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=DARK_BG,
        font=dict(color="#FAFAFA", size=12),
        xaxis=dict(gridcolor="#2a2d3a"),
        yaxis=dict(title="Cumulative Return (%)", gridcolor="#2a2d3a", ticksuffix="%"),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_color="#666", line_dash="dot", line_width=1)
    return fig


def weights_bar(weights: pd.Series) -> go.Figure:
    ws = weights.sort_values(ascending=True)
    labels = [ALL_SHORT_NAMES.get(t, t) for t in ws.index]
    fig = go.Figure(go.Bar(
        x=ws.values * 100, y=labels, orientation="h",
        marker_color=["#4ade80" if v >= 0.001 else "#555" for v in ws],
        text=[f"{v*100:.1f}%" for v in ws], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(250, 40 + len(ws) * 28),
        margin=dict(l=0, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=DARK_BG,
        font=dict(color="#FAFAFA", size=12),
        xaxis=dict(title="Weight (%)", gridcolor="#2a2d3a", ticksuffix="%"),
        yaxis=dict(gridcolor="#2a2d3a"),
    )
    return fig

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _groq_key() -> str:
    k = os.environ.get("GROQ_API_KEY","")
    if not k:
        try: k = st.secrets.get("GROQ_API_KEY","")
        except Exception: pass
    return k


def _ant_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY","")
    if not k:
        try: k = st.secrets.get("ANTHROPIC_API_KEY","")
        except Exception: pass
    return k


def _llm(prompt: str, provider: str) -> str:
    if "Groq" in provider:
        client = GroqClient(api_key=_groq_key())
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":"You are a senior economist and portfolio strategist."},
                {"role":"user","content": f"Today: {date.today().strftime('%B %d, %Y')}.\n\n{prompt}"},
            ],
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    client = _ant.Anthropic(api_key=_ant_key())
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1800,
        tools=[{"type":"web_search_20250305","name":"web_search"}],
        messages=[{"role":"user","content":prompt}],
    )
    return "".join(b.text for b in resp.content if hasattr(b,"text"))


def _dyn_prompt(alloc_df: pd.DataFrame) -> str:
    n = len(alloc_df); thirds = max(1, n//3)
    def avg(df): return (df.mean()*100).sort_values(ascending=False)
    def fmt(s): return "\n".join(f"  {ALL_SHORT_NAMES.get(t,t)}: {v:.1f}%" for t,v in s.items() if v>0.5)
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
                      for t,v in weights.sort_values(ascending=False).items() if v>0.001)
    stats = "\n".join(f"  {k}: {v}" for k,v in metrics.items())
    return (
        f"Static portfolio allocated on {alloc_d}, backtested {s}→{e}.\n\n"
        f"Allocation:\n{alloc}\n\nStatistics:\n{stats}\n\n"
        "Interpret in 400–500 words: quality of performance, likely return drivers, "
        "current risks, and whether this is an attractive allocation now."
    )

# ---------------------------------------------------------------------------
# Page — Section 1: Asset selector
# ---------------------------------------------------------------------------

init_db()

st.title("Dynamic Allocation")
st.caption(
    "Select assets from the full universe, choose a solver and constraints, "
    "then run a single optimisation, rolling dynamic chart, or historical backtest."
)

with st.expander("⚙ Data — refresh price history"):
    st.caption(
        "Price history is fetched from Yahoo Finance and cached in the app's "
        "database. Use this if the charts look out of date — it re-downloads "
        "every asset and may take a minute."
    )
    if st.button("↻ Refresh price data", key="da_refresh"):
        try:
            with st.spinner("Fetching latest prices from Yahoo Finance…"):
                from scripts.fetch_market_data import main as _fetch_global
                from scripts.fetch_global_etfs import main as _fetch_etfs
                _fetch_global()
                _fetch_etfs()
            st.cache_data.clear()
            st.success("Price data refreshed.")
            st.rerun()
        except Exception as e:
            st.warning(f"Refresh failed: {e}")

st.subheader("1. Select Assets")

for group_name, categories in UNIVERSE.items():
    group_tickers = [t for tks in categories.values() for t in tks]
    with st.expander(f"**{group_name}** ({len(group_tickers)} assets)", expanded=True):
        bc1, bc2, _ = st.columns([1,1,8])
        def _sel(gt=group_tickers):
            for t in gt: st.session_state[f"sel_{t}"] = True
        def _clr(gt=group_tickers):
            for t in gt: st.session_state[f"sel_{t}"] = False
        bc1.button("Select all", key=f"sa_{group_name}", on_click=_sel)
        bc2.button("Clear",      key=f"ca_{group_name}", on_click=_clr)
        for cat_name, tickers in categories.items():
            st.markdown(f"**{cat_name}**")
            cols = st.columns(min(len(tickers), 4))
            for i, t in enumerate(tickers):
                cols[i % len(cols)].checkbox(
                    ALL_SHORT_NAMES.get(t, t), key=f"sel_{t}",
                    value=st.session_state.get(f"sel_{t}", False),
                )

selected: list[str] = [t for t in ALL_TICKERS if st.session_state.get(f"sel_{t}", False)]
if selected:
    st.caption(f"**{len(selected)} selected:** " +
               ", ".join(ALL_SHORT_NAMES.get(t,t) for t in selected))
else:
    st.caption("No assets selected.")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Solver & settings
# ---------------------------------------------------------------------------

st.subheader("2. Solver & Settings")

s_col, p_col = st.columns([3, 2])
with s_col:
    solver_name = st.radio("Objective", SOLVER_NAMES, horizontal=True, key="da_solver")

solver_params: dict = {}
with p_col:
    st.write("")
    if solver_name == "Min Volatility":
        v = st.number_input("Min annual return (%)", 0.0, 50.0, 5.0, 0.5, key="da_minr")
        solver_params["min_ret"] = v / 100
    elif solver_name == "Max Return":
        v = st.number_input("Max annual volatility (%)", 1.0, 100.0, 15.0, 0.5, key="da_maxv")
        solver_params["max_vol"] = v / 100
    elif solver_name == "Min Drawdown":
        dc1, dc2 = st.columns(2)
        solver_params["dd_lookback"] = dc1.number_input(
            "Lookback (days)", 60, 756, 252, 21, key="da_ddlb")
        cap = dc2.number_input("Max DD cap (%, 0=off)", 0.0, 100.0, 0.0, 1.0, key="da_ddcap")
        if cap > 0:
            solver_params["max_dd_cap"] = cap / 100

with st.expander("Advanced settings"):
    ac1, ac2, ac3, ac4 = st.columns(4)
    today = date.today()
    ts_raw = ac1.text_input("Train start (YYYYMMDD)",
                             (today - timedelta(days=5*365)).strftime("%Y%m%d"), key="da_ts")
    te_raw = ac2.text_input("Train end (YYYYMMDD/'today')", "today", key="da_te")
    ewm_hl  = ac3.slider("EWM halflife (days)", 21, 252, 63, 7, key="da_ewm")
    rf_pct  = ac4.number_input("Risk-free rate (%)", 0.0, 20.0, 0.0, 0.25, key="da_rf")
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

# ---------------------------------------------------------------------------
# Section 3: Constraints table
# ---------------------------------------------------------------------------

if len(selected) < 2:
    st.info("Select at least 2 assets above to configure constraints.", icon="☝️")
    st.stop()

st.subheader("3. Constraints")
st.caption("▶ rows = category-level combined bounds · indented rows = individual asset bounds. All values in %.")

# Build grouped category map for selected tickers
sel_cats: dict[str, list[str]] = {}
for t in selected:
    c = TICKER_TO_CAT.get(t, "Other")
    sel_cats.setdefault(c, []).append(t)

rows = []
for cat, tickers in sel_cats.items():
    rows.append({"Type":"category","Name":f"▶ {cat}","Ticker":"—","Min %":0.0,"Max %":100.0})
    for t in tickers:
        lo, hi = DEFAULT_BOUNDS.get(t, (0.0, 30.0))
        rows.append({"Type":"asset","Name":f"  {ALL_SHORT_NAMES.get(t,t)}","Ticker":t,
                     "Min %":float(lo),"Max %":float(hi)})

sel_hash = hashlib.md5("_".join(selected).encode()).hexdigest()[:8]
edited = st.data_editor(
    pd.DataFrame(rows),
    key=f"da_con_{sel_hash}",
    column_config={
        "Type":   st.column_config.TextColumn("Type",   disabled=True, width="small"),
        "Name":   st.column_config.TextColumn("Name",   disabled=True, width="medium"),
        "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
        "Min %":  st.column_config.NumberColumn("Min %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
        "Max %":  st.column_config.NumberColumn("Max %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
    },
    hide_index=True, use_container_width=True,
)

cat_bounds: dict[str, tuple[float,float]] = {}
lb_map: dict[str, float] = {}
ub_map: dict[str, float] = {}
for _, row in edited.iterrows():
    if row["Type"] == "category":
        cat_raw = row["Name"].replace("▶ ","",1).strip()
        cat_bounds[cat_raw] = (float(row["Min %"]), float(row["Max %"]))
    else:
        lb_map[row["Ticker"]] = float(row["Min %"])
        ub_map[row["Ticker"]] = float(row["Max %"])

lb_arr = np.array([lb_map.get(t, 0.0)/100 for t in selected])
ub_arr = np.array([ub_map.get(t, 30.0)/100 for t in selected])

if lb_arr.sum() > 1.001:
    st.error(f"Sum of minimum weights ({lb_arr.sum()*100:.1f}%) exceeds 100%. Reduce some minimums.")
    st.stop()
if ub_arr.sum() < 0.999:
    st.error(f"Sum of maximum weights ({ub_arr.sum()*100:.1f}%) is below 100%. Increase some maximums.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Single optimisation
# ---------------------------------------------------------------------------

st.subheader("4. Single Optimisation")

run_btn = st.button("▶ Run Optimisation", type="primary", key="da_run")

if run_btn:
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
        lb_s = np.array([lb_map.get(t, 0.0)/100 for t in avail])
        ub_s = np.array([ub_map.get(t, 30.0)/100 for t in avail])
        w, ok = run_solver(mu, Sigma, r, lb_s, ub_s, cat_bounds, solver_name, solver_params)
        if not ok:
            st.warning("Optimiser did not fully converge.", icon="⚠️")
        st.session_state["da_w"]      = w
        st.session_state["da_px"]     = px
        st.session_state["da_mu"]     = mu
        st.session_state["da_Sigma"]  = Sigma
        st.session_state["da_rf"]     = float(solver_params.get("rf", 0.0))
        st.session_state["da_ewm_hl"] = ewm_hl
        for k in ("da_rolling","da_rolling_hash","da_bt","da_dyn_llm","da_bt_llm"):
            st.session_state.pop(k, None)

if "da_w" in st.session_state:
    w   = st.session_state["da_w"]
    px_ = st.session_state["da_px"]
    m   = portfolio_metrics(w, px_)
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.plotly_chart(weights_bar(w), use_container_width=True)
    with cc2:
        if m:
            st.markdown("**Statistics (training window)**")
            for k, v in m.items():
                st.metric(k, v)
            st.caption(
                "Equal-weighted realized stats over the full training window. "
                f"The solver optimised against EWM-weighted estimates (halflife = "
                f"{st.session_state.get('da_ewm_hl', '?')} days) — see below."
            )
            em = ewm_portfolio_metrics(
                w,
                st.session_state.get("da_mu"),
                st.session_state.get("da_Sigma"),
                st.session_state.get("da_rf", 0.0),
            ) if "da_mu" in st.session_state else {}
            if em:
                st.markdown("**Optimiser view (EWM-weighted)**")
                for k, v in em.items():
                    st.metric(k, v)
        st.dataframe(
            pd.DataFrame({"Asset":[ALL_SHORT_NAMES.get(t,t) for t in w.index],
                          "Ticker":w.index.tolist(),
                          "Weight":[f"{v*100:.2f}%" for v in w]}),
            hide_index=True, use_container_width=True,
        )

    # ---- Correlation heatmap of allocated assets (weight > 0.5%) ----
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
        st.caption(
            "Pearson correlation of daily log-returns over the training window; "
            "only assets with weight > 0.5% shown."
        )
        fig_corr = go.Figure(go.Heatmap(
            z=z,
            x=labels, y=labels,
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b>: %{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=DARK_BG,
            font=dict(color="#FAFAFA", size=11),
            xaxis=dict(tickangle=-40, side="bottom"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 5: Dynamic allocation (rolling weekly)
# ---------------------------------------------------------------------------

st.subheader("5. Dynamic Allocation")
roll_lb = st.slider("Rolling lookback (days) — for dynamic chart",
                    63, 756, 252, 21, key="da_rolllb")
st.caption(
    f"Optimiser runs every 5 trading days with a user-specified lookback window "
    f"(currently **{roll_lb} days**). Stacked area shows weight evolution over time. "
    "Hover to see ranked allocations on any date."
)

dyn_btn = st.button("▶ Run Dynamic Chart", key="da_rundyn")

if dyn_btn:
    if len(selected) < 2:
        st.warning("Select assets first.", icon="⚠️")
    else:
        sp_t  = tuple(sorted(solver_params.items()))
        cb_t  = tuple(sorted(cat_bounds.items()))
        lb_t  = tuple(lb_arr.tolist())
        ub_t  = tuple(ub_arr.tolist())
        phash = hashlib.md5(json.dumps(
            [list(selected), solver_name, sp_t, lb_t, ub_t, cb_t, roll_lb, ewm_hl],
            sort_keys=True, default=str
        ).encode()).hexdigest()[:12]

        if st.session_state.get("da_rolling_hash") != phash:
            n_est = max(0, (len(load_prices(
                tuple(selected),
                (today - timedelta(days=10*365)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            )) - roll_lb) // 5)
            with st.spinner(f"Computing ~{n_est} optimisation steps — please wait…"):
                alloc_df = compute_rolling(
                    tuple(selected), solver_name, sp_t,
                    lb_t, ub_t, cb_t, roll_lb, ewm_hl,
                    (today - timedelta(days=10*365)).strftime("%Y-%m-%d"),
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
        st.markdown("**Economist Interpretation**")
        prov_d = st.radio("Provider", ["Groq — Llama 3.3 70B (Free)","Anthropic — Claude Haiku (Paid)"],
                          horizontal=True, key="da_dprov")
        gc1, gc2 = st.columns([2,2])
        if gc1.button("Generate", key="da_dgen"):
            with st.spinner("Generating interpretation…"):
                try:
                    st.session_state["da_dyn_llm"] = _llm(_dyn_prompt(alloc_df), prov_d)
                except Exception as e:
                    st.error(f"LLM error: {e}")
        if "da_dyn_llm" in st.session_state:
            if gc2.button("↺ Regenerate", key="da_dregen"):
                del st.session_state["da_dyn_llm"]
                st.rerun()
            st.markdown(st.session_state["da_dyn_llm"])
    else:
        st.info("No allocation data returned — try more assets or a shorter lookback.")

st.divider()

# ---------------------------------------------------------------------------
# Section 6: Static backtest
# ---------------------------------------------------------------------------

st.subheader("6. Backtest Static Allocation")
st.caption(
    "Compute weights at a chosen date, then track that fixed allocation over any period."
)

bc1, bc2, bc3 = st.columns(3)
ad_raw = bc1.text_input("Allocation date (YYYYMMDD/'today')", "today", key="da_ad")
bs_raw = bc2.text_input("Backtest start (YYYYMMDD)", ts_raw, key="da_bs")
be_raw = bc3.text_input("Backtest end (YYYYMMDD/'today')", "today", key="da_be")

bt_btn = st.button("▶ Run Backtest", key="da_runbt")

if bt_btn:
    alloc_dt = _resolve(ad_raw)
    bt_s_dt  = _parse(bs_raw)
    bt_e_dt  = _resolve(be_raw)

    if not alloc_dt: st.error("Invalid allocation date."); st.stop()
    if not bt_s_dt:  st.error("Invalid backtest start."); st.stop()
    if not bt_e_dt:  st.error("Invalid backtest end."); st.stop()
    if bt_e_dt < bt_s_dt: st.error("End must be after start."); st.stop()

    with st.spinner("Computing allocation and backtest…"):
        # Re-optimise at alloc_dt
        alloc_win_s = (alloc_dt - timedelta(days=2*365)).strftime("%Y-%m-%d")
        alloc_win_e = alloc_dt.strftime("%Y-%m-%d")
        px_a = load_prices(tuple(selected), alloc_win_s, alloc_win_e)
        avail_a = [t for t in selected if t in px_a.columns]
        lb_a = np.array([lb_map.get(t, 0.0)/100 for t in avail_a])
        ub_a = np.array([ub_map.get(t, 30.0)/100 for t in avail_a])
        if len(avail_a) >= 2:
            ra = log_ret(px_a[avail_a])
            if len(ra) >= max(10, ewm_hl):
                mu_a, Sg_a = ewm_stats(ra, ewm_hl)
                bt_w, _ = run_solver(mu_a, Sg_a, ra, lb_a, ub_a, cat_bounds,
                                     solver_name, solver_params)
            else:
                bt_w = pd.Series(1/len(avail_a), index=avail_a)
        else:
            bt_w = pd.Series(1/len(selected), index=selected)

        # Load backtest prices
        px_bt = load_prices(
            tuple(bt_w.index.tolist()),
            bt_s_dt.strftime("%Y-%m-%d"),
            bt_e_dt.strftime("%Y-%m-%d"),
        )
        avail_bt = [t for t in bt_w.index if t in px_bt.columns]
        if not avail_bt: st.error("No price data in backtest period."); st.stop()
        px_bt = px_bt[avail_bt].dropna()
        w_bt  = bt_w[avail_bt] / bt_w[avail_bt].sum()
        rets_bt = log_ret(px_bt) @ w_bt
        met_bt  = portfolio_metrics(bt_w, px_bt)

        st.session_state["da_bt"] = {
            "rets": rets_bt, "weights": bt_w, "metrics": met_bt,
            "alloc_date": alloc_dt.strftime("%Y%m%d"),
            "bt_start":   bt_s_dt.strftime("%Y%m%d"),
            "bt_end":     bt_e_dt.strftime("%Y%m%d"),
        }
        st.session_state.pop("da_bt_llm", None)

if "da_bt" in st.session_state:
    bt = st.session_state["da_bt"]
    bcc1, bcc2 = st.columns([2, 1])
    with bcc1:
        st.plotly_chart(backtest_chart(bt["rets"]), use_container_width=True)
    with bcc2:
        st.markdown(f"**Allocated:** {bt['alloc_date']}  \n"
                    f"**Period:** {bt['bt_start']} → {bt['bt_end']}")
        for k, v in bt["metrics"].items():
            st.metric(k, v)
        st.dataframe(
            pd.DataFrame({"Asset":[ALL_SHORT_NAMES.get(t,t) for t in bt["weights"].index],
                          "Weight":[f"{v*100:.1f}%" for v in bt["weights"]]}),
            hide_index=True, use_container_width=True,
        )

    st.markdown("---")
    st.markdown("**Economist Interpretation**")
    prov_b = st.radio("Provider", ["Groq — Llama 3.3 70B (Free)","Anthropic — Claude Haiku (Paid)"],
                      horizontal=True, key="da_bprov")
    bc1b, bc2b = st.columns([2,2])
    if bc1b.button("Generate", key="da_bgen"):
        with st.spinner("Generating interpretation…"):
            try:
                st.session_state["da_bt_llm"] = _llm(
                    _bt_prompt(bt["weights"], bt["metrics"],
                               bt["alloc_date"], bt["bt_start"], bt["bt_end"]),
                    prov_b,
                )
            except Exception as e:
                st.error(f"LLM error: {e}")
    if "da_bt_llm" in st.session_state:
        if bc2b.button("↺ Regenerate", key="da_bregen"):
            del st.session_state["da_bt_llm"]
            st.rerun()
        st.markdown(st.session_state["da_bt_llm"])
