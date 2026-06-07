"""A compact, scannable metric grid — replaces vertical st.metric stacks."""
from __future__ import annotations

import streamlit as st


def metric_grid(
    metrics: dict[str, str],
    ncols: int = 4,
    deltas: dict[str, str] | None = None,
) -> None:
    """Lay out `metrics` as a grid of `st.metric` cards, `ncols` per row.

    `deltas` optionally maps a metric label to a delta string (Streamlit colours
    it and adds an arrow — colour is always paired with the +/- sign and arrow).
    """
    if not metrics:
        return
    deltas = deltas or {}
    items = list(metrics.items())
    for i in range(0, len(items), ncols):
        chunk = items[i:i + ncols]
        cols = st.columns(ncols)
        for col, (label, value) in zip(cols, chunk):
            col.metric(label, value, delta=deltas.get(label))
