"""Shared 'strategy setup' widgets used by Dynamic Allocation (07) and the
Dynamic Strategy Tester (08).

Both pages need the same three building blocks — an asset selector, an objective
+ its parameters, and a category/asset bounds editor. They previously duplicated
all of this. Each function is parameterised by `key_prefix` so the two pages keep
independent widget state (e.g. "da_" vs "dst_").
"""
from __future__ import annotations
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from core.optim_engine import (
    UNIVERSE, ALL_SHORT_NAMES, ALL_TICKERS, TICKER_TO_CAT,
    DEFAULT_BOUNDS, DEFAULT_CAT_BOUNDS, SOLVER_NAMES,
)


def asset_selector(key_prefix: str, *, expanded: bool = True) -> list[str]:
    """Grouped checkbox universe with Select-all / Clear per group.

    Returns the selected tickers in canonical ALL_TICKERS order.
    """
    for group_name, categories in UNIVERSE.items():
        group_tickers = [t for tks in categories.values() for t in tks]
        with st.expander(f"**{group_name}** ({len(group_tickers)} assets)", expanded=expanded):
            bc1, bc2, _ = st.columns([1, 1, 8])

            def _sel(gt=group_tickers) -> None:
                for t in gt:
                    st.session_state[f"{key_prefix}sel_{t}"] = True

            def _clr(gt=group_tickers) -> None:
                for t in gt:
                    st.session_state[f"{key_prefix}sel_{t}"] = False

            bc1.button("Select all", key=f"{key_prefix}sa_{group_name}", on_click=_sel)
            bc2.button("Clear",      key=f"{key_prefix}ca_{group_name}", on_click=_clr)
            for cat_name, tickers in categories.items():
                st.markdown(f"**{cat_name}**")
                cols = st.columns(min(len(tickers), 4))
                for i, t in enumerate(tickers):
                    cols[i % len(cols)].checkbox(
                        ALL_SHORT_NAMES.get(t, t), key=f"{key_prefix}sel_{t}",
                        value=st.session_state.get(f"{key_prefix}sel_{t}", False),
                    )

    selected = [t for t in ALL_TICKERS if st.session_state.get(f"{key_prefix}sel_{t}", False)]
    if selected:
        st.caption(f"**{len(selected)} selected:** " +
                   ", ".join(ALL_SHORT_NAMES.get(t, t) for t in selected))
    else:
        st.caption("No assets selected.")
    return selected


def solver_settings(key_prefix: str) -> tuple[str, dict]:
    """Objective radio + objective-specific parameters.

    Returns (solver_name, solver_params). The caller adds `rf` to solver_params
    (pages place the risk-free input differently).
    """
    s_col, p_col = st.columns([3, 2])
    with s_col:
        solver_name = st.radio("Objective", SOLVER_NAMES, horizontal=True,
                               key=f"{key_prefix}solver")

    solver_params: dict = {}
    with p_col:
        st.write("")
        if solver_name == "Min Volatility":
            v = st.number_input("Min annual return (%)", 0.0, 50.0, 5.0, 0.5,
                                key=f"{key_prefix}minr")
            solver_params["min_ret"] = v / 100
        elif solver_name == "Max Return":
            v = st.number_input("Max annual volatility (%)", 1.0, 100.0, 15.0, 0.5,
                                key=f"{key_prefix}maxv")
            solver_params["max_vol"] = v / 100
        elif solver_name == "Min Drawdown":
            dc1, dc2 = st.columns(2)
            solver_params["dd_lookback"] = dc1.number_input(
                "Lookback (days)", 60, 756, 252, 21, key=f"{key_prefix}ddlb")
            cap = dc2.number_input("Max DD cap (%, 0=off)", 0.0, 100.0, 0.0, 1.0,
                                   key=f"{key_prefix}ddcap")
            if cap > 0:
                solver_params["max_dd_cap"] = cap / 100
    return solver_name, solver_params


def constraints_editor(
    selected: list[str], key_prefix: str,
) -> tuple[dict[str, float], dict[str, float],
           dict[str, tuple[float, float]], np.ndarray, np.ndarray]:
    """Category + per-asset bounds editor. Validates feasibility (st.stop on fail).

    Returns (lb_map, ub_map, cat_bounds, lb_arr, ub_arr) where the maps are in %
    and the arrays are fractions aligned to `selected`.
    """
    st.caption("▶ rows = category-level combined bounds · indented rows = "
               "individual asset bounds. All values in %.")

    sel_cats: dict[str, list[str]] = {}
    for t in selected:
        sel_cats.setdefault(TICKER_TO_CAT.get(t, "Other"), []).append(t)

    rows: list[dict] = []
    for cat, tickers in sel_cats.items():
        lo_c, hi_c = DEFAULT_CAT_BOUNDS.get(cat, (0.0, 100.0))
        rows.append({"Type": "category", "Name": f"▶ {cat}", "Ticker": "—",
                     "Min %": float(lo_c), "Max %": float(hi_c)})
        for t in tickers:
            lo, hi = DEFAULT_BOUNDS.get(t, (0.0, 30.0))
            rows.append({"Type": "asset", "Name": f"  {ALL_SHORT_NAMES.get(t, t)}",
                         "Ticker": t, "Min %": float(lo), "Max %": float(hi)})

    sel_hash = hashlib.md5("_".join(selected).encode()).hexdigest()[:8]
    edited = st.data_editor(
        pd.DataFrame(rows),
        key=f"{key_prefix}con_{sel_hash}",
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
        st.error(f"Sum of minimum weights ({lb_arr.sum()*100:.1f}%) exceeds 100%. "
                 "Reduce some minimums.")
        st.stop()
    if ub_arr.sum() < 0.999:
        st.error(f"Sum of maximum weights ({ub_arr.sum()*100:.1f}%) is below 100%. "
                 "Increase some maximums.")
        st.stop()

    return lb_map, ub_map, cat_bounds, lb_arr, ub_arr
