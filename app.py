import streamlit as st
from core.data_store import init_db

st.set_page_config(
    page_title="Dynamic Allocation",
    page_icon="📈",
    layout="wide",
)

init_db()

# Multi-page app: Market Overview (default) + Dynamic Allocation.
# Using st.navigation disables automatic pages/ discovery, so any other
# page files in the repo remain dormant.
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
])
pg.run()
