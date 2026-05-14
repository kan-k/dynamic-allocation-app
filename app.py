import streamlit as st
from core.data_store import init_db

st.set_page_config(
    page_title="Dynamic Allocation",
    page_icon="📈",
    layout="wide",
)

init_db()

# Single-page app: only the Dynamic Allocation tool is exposed.
# Using st.navigation disables automatic pages/ discovery, so the other
# page files remain in the repo but never appear in the sidebar.
pg = st.navigation([
    st.Page(
        "pages/07_dynamic_allocation.py",
        title="Dynamic Allocation",
        default=True,
    )
])
pg.run()
