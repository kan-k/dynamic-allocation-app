import streamlit as st
from core.data_store import init_db

st.set_page_config(
    page_title="Dynamic Allocation",
    page_icon="📈",
    layout="wide",
)

# Global stylesheet: tabular numerals for all metrics + dataframe cells so
# digits line up column-to-row. Tightens H3 spacing slightly.
st.markdown(
    """
    <style>
      div[data-testid="stMetricValue"],
      div[data-testid="stDataFrame"] * {
        font-variant-numeric: tabular-nums !important;
      }
      h3 { margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

# Multi-page app. Using st.navigation disables automatic pages/ discovery,
# so any other page files in the repo remain dormant.
pg = st.navigation([
    st.Page(
        "pages/06_market_overview.py",
        title="Market Overview",
        default=True,
    ),
    st.Page(
        "pages/07_dynamic_allocation.py",
        title="Dynamic Allocation",
    ),
    st.Page(
        "pages/08_dynamic_strategy_tester.py",
        title="Dynamic Strategy Tester",
    ),
])
pg.run()
