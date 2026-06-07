"""Backtesting engine for the Dynamic Strategy Tester page.

Schedules periodic rebalances, re-runs the solver from `core.optim_engine`
at each rebalance date on a trailing window of returns, and stitches the
resulting weights into a continuous daily portfolio return series.
"""
from __future__ import annotations
import itertools
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.optim_engine import (
    ALL_SHORT_NAMES, COLORS,
    ewm_stats, log_ret, run_solver, theme,
)

FREQUENCIES = ["Weekly", "Biweekly", "Monthly", "Bimonthly"]


# ---------------------------------------------------------------------------
# Rebalance scheduling
# ---------------------------------------------------------------------------

def _calendar_anchors(eval_start: date, eval_end: date, freq: str) -> list[pd.Timestamp]:
    """Generate the calendar anchor dates (pre-snap to trading days)."""
    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    if freq == "Weekly":
        # Every Monday on/after eval_start
        first_mon = s + pd.offsets.Week(weekday=0) if s.weekday() != 0 else s
        return list(pd.date_range(first_mon, e, freq="W-MON"))
    if freq == "Biweekly":
        first_mon = s + pd.offsets.Week(weekday=0) if s.weekday() != 0 else s
        return list(pd.date_range(first_mon, e, freq="2W-MON"))
    if freq == "Monthly":
        # 1st of each month on/after eval_start
        first = s.replace(day=1) if s.day == 1 else (s + pd.offsets.MonthBegin(1))
        return list(pd.date_range(first, e, freq="MS"))
    if freq == "Bimonthly":
        first = s.replace(day=1) if s.day == 1 else (s + pd.offsets.MonthBegin(1))
        return list(pd.date_range(first, e, freq="2MS"))
    raise ValueError(f"Unknown frequency: {freq!r}")


def rebalance_dates(
    eval_start: date, eval_end: date, freq: str, trading_index: pd.DatetimeIndex,
) -> list[pd.Timestamp]:
    """Calendar anchors snapped to the first trading day on/after each anchor."""
    anchors = _calendar_anchors(eval_start, eval_end, freq)
    snapped: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for anchor in anchors:
        idx = trading_index.searchsorted(anchor, side="left")
        if idx >= len(trading_index):
            break
        d = trading_index[idx]
        if d > pd.Timestamp(eval_end):
            break
        if d not in seen:
            snapped.append(d)
            seen.add(d)
    return snapped


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------

def _metrics_from_returns(
    port_rets: pd.Series, rf: float, n_rebalances: int,
) -> dict[str, str]:
    if port_rets.empty:
        return {}
    ann_r = float(port_rets.mean() * 252)
    ann_v = float(port_rets.std() * np.sqrt(252))
    sharpe = (ann_r - rf) / ann_v if ann_v > 1e-10 else float("nan")
    cum = np.exp(port_rets.cumsum())
    total_ret = float(cum.iloc[-1] - 1.0)
    dd = (cum / cum.cummax()) - 1
    max_dd = float(dd.min())
    durs = [sum(1 for _ in g) for k, g in itertools.groupby((dd < 0).astype(int).tolist()) if k]
    calmar = ann_r / abs(max_dd) if max_dd < -1e-10 else float("nan")
    return {
        "Total Return":    f"{total_ret*100:+.2f}%",
        "Ann. Return":     f"{ann_r*100:+.2f}%",
        "Ann. Volatility": f"{ann_v*100:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{max_dd*100:.2f}%",
        "Max DD Duration": f"{max(durs, default=0)} days",
        "Calmar Ratio":    f"{calmar:.3f}",
        "# Rebalances":    f"{n_rebalances}",
    }


def run_backtest(
    px: pd.DataFrame,
    eval_start: date, eval_end: date,
    freq: str,
    lookback: int,
    halflife: int,
    lb_arr: np.ndarray, ub_arr: np.ndarray,
    cat_bounds: dict[str, tuple[float, float]],
    solver_name: str, solver_params: dict,
) -> tuple[pd.Series, pd.DataFrame, dict[str, str]]:
    """Backtest a periodically-rebalanced strategy.

    Handles partial-history assets: at each rebalance, an asset is included
    only if it has at least `min_period` non-NaN observations in the
    trailing lookback window. Ineligible assets carry weight 0.0 at that
    rebalance (e.g. an ETF that hadn't been listed yet sits out until it
    accumulates enough data).

    Returns
    -------
    port_rets : pd.Series
        Daily log-returns of the strategy over the realised holding period.
    weights_hist : pd.DataFrame
        Weights at each rebalance (index = date, columns = ALL selected
        tickers; 0.0 for any asset ineligible at that date).
    metrics : dict[str, str]
        Formatted performance metrics.
    """
    rf = float(solver_params.get("rf", 0.0))
    min_period = max(2 * halflife, 5)
    all_tickers = px.columns.tolist()
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}

    full_index = px.index
    eval_mask = (full_index >= pd.Timestamp(eval_start)) & (full_index <= pd.Timestamp(eval_end))
    if not eval_mask.any():
        return pd.Series(dtype=float), pd.DataFrame(), {}

    sched = rebalance_dates(eval_start, eval_end, freq, full_index[eval_mask])
    if not sched:
        return pd.Series(dtype=float), pd.DataFrame(), {}

    end_sentinel = full_index[eval_mask][-1] + pd.Timedelta(days=1)

    weights_hist: dict[pd.Timestamp, pd.Series] = {}
    last_weights: pd.Series | None = None

    for t_i in sched:
        win = px.loc[:t_i].tail(lookback)
        # Per-asset availability inside the lookback window
        eligible = [t for t in all_tickers if win[t].count() >= min_period]
        if len(eligible) < 2:
            if last_weights is not None:
                weights_hist[t_i] = last_weights
            continue
        r_win = log_ret(win[eligible])
        if len(r_win) < min_period:
            if last_weights is not None:
                weights_hist[t_i] = last_weights
            continue
        # Subset bounds to eligible assets
        lb_e = np.array([lb_arr[ticker_to_idx[t]] for t in eligible])
        ub_e = np.array([ub_arr[ticker_to_idx[t]] for t in eligible])
        if lb_e.sum() > 1.001 or ub_e.sum() < 0.999:
            # Bounds infeasible with reduced universe — carry over or skip
            if last_weights is not None:
                weights_hist[t_i] = last_weights
            continue
        mu, Sigma = ewm_stats(r_win, halflife)
        w_eligible, _ = run_solver(mu, Sigma, r_win, lb_e, ub_e,
                                   cat_bounds, solver_name, solver_params)
        # Embed eligible weights back into full ticker vector (0 elsewhere)
        full_w = pd.Series(0.0, index=all_tickers)
        full_w.loc[eligible] = w_eligible.values
        weights_hist[t_i] = full_w
        last_weights = full_w

    if not weights_hist:
        return pd.Series(dtype=float), pd.DataFrame(), {}

    # Per-segment portfolio returns over the active (non-zero-weight) subset.
    boundaries = list(weights_hist.keys()) + [end_sentinel]
    port_segments: list[pd.Series] = []
    for i in range(len(boundaries) - 1):
        t_a, t_b = boundaries[i], boundaries[i + 1]
        w_i = weights_hist[t_a]
        active = w_i[w_i > 1e-10].index.tolist()
        if not active:
            continue
        seg_mask = (px.index > t_a) & (px.index <= t_b)
        seg_px = px.loc[seg_mask, active]
        if seg_px.shape[0] < 2:
            continue
        seg_rets = log_ret(seg_px)
        if seg_rets.empty:
            continue
        w_active = w_i[seg_rets.columns]
        if w_active.sum() > 1e-10:
            w_active = w_active / w_active.sum()
        port_segments.append(seg_rets @ w_active)

    if not port_segments:
        return pd.Series(dtype=float), pd.DataFrame(weights_hist).T.sort_index(), {}

    port_rets = pd.concat(port_segments).sort_index()
    weights_df = pd.DataFrame(weights_hist).T.sort_index()
    metrics = _metrics_from_returns(port_rets, rf, len(weights_hist))
    return port_rets, weights_df, metrics


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def cumret_chart(
    port_rets: pd.Series,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """Cumulative-return line for the strategy, with an optional benchmark overlay.

    `benchmark` is a daily log-return series; it is cumulated and aligned to the
    strategy's date range (so a like-for-like comparison from the same start).
    """
    th = theme()
    cum = ((np.exp(port_rets.cumsum()) - 1) * 100).round(3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values, name="Strategy (cum. return)",
        line=dict(color="#00C49F", width=2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:+.3f}%<extra></extra>",
    ))
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(port_rets.index).dropna()
        if len(b) >= 2:
            bcum = ((np.exp(b.cumsum()) - 1) * 100).round(3)
            fig.add_trace(go.Scatter(
                x=bcum.index, y=bcum.values, name=benchmark_name,
                line=dict(color=th["zero"], width=1.6, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.3f}%<extra></extra>",
            ))
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        xaxis=dict(gridcolor=th["grid"]),
        yaxis=dict(title="Cumulative Return (%)", gridcolor=th["grid"], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_color=th["zero"], line_dash="dot", line_width=1)
    return fig


def drawdown_chart(port_rets: pd.Series) -> go.Figure:
    """Underwater (drawdown-from-peak) chart for a daily log-return series."""
    th = theme()
    fig = go.Figure()
    if not port_rets.empty:
        cum = np.exp(port_rets.cumsum())
        dd = ((cum / cum.cummax()) - 1) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values.round(3), name="Drawdown",
            fill="tozeroy", line=dict(color="#EF476F", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        xaxis=dict(gridcolor=th["grid"]),
        yaxis=dict(title="Drawdown (%)", gridcolor=th["grid"], ticksuffix="%",
                   rangemode="tozero"),
        hovermode="x unified",
    )
    return fig


def composition_chart(weights_hist: pd.DataFrame) -> go.Figure:
    """Stacked-area chart of portfolio composition over time."""
    if weights_hist.empty:
        return go.Figure()
    th = theme()
    # Order traces by mean weight (largest at bottom of stack)
    ordered = weights_hist.mean().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, t in enumerate(ordered):
        fig.add_trace(go.Scatter(
            x=weights_hist.index,
            y=(weights_hist[t] * 100).round(2),
            name=ALL_SHORT_NAMES.get(t, t),
            mode="lines",
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
            stackgroup="one",
            hoverinfo="skip",
        ))

    def _hover(row: pd.Series) -> str:
        ranked = row.dropna().sort_values(ascending=False)
        return "<br>".join(
            f"<b>{ALL_SHORT_NAMES.get(t, t)}</b>: {v*100:.1f}%"
            for t, v in ranked.items() if v >= 0.001
        )

    fig.add_trace(go.Scatter(
        x=weights_hist.index, y=[0.0] * len(weights_hist),
        mode="markers", marker=dict(opacity=0, size=1),
        hovertemplate="%{text}<extra></extra>",
        text=weights_hist.apply(_hover, axis=1).tolist(),
        showlegend=False, name="",
    ))
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["font"], size=12),
        yaxis=dict(title="Allocation (%)", gridcolor=th["grid"], ticksuffix="%",
                   range=[0, 100]),
        xaxis=dict(gridcolor=th["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig
