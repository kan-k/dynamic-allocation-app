import streamlit as st
from core.data_store import init_db

st.set_page_config(
    page_title="Dynamic Allocation",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme toggle (dark default, cream alternative)
# ---------------------------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Base stylesheet — applies in both themes.
st.markdown(
    """
    <style>
      div[data-testid="stMetricValue"],
      div[data-testid="stDataFrame"] * {
        font-variant-numeric: tabular-nums !important;
      }
      h3 { margin-top: 0.5rem; }
      /* Compact toggle button at the top-right */
      div[data-testid="column"]:has(button[kind="secondary"][aria-label]) button {
        padding: 0.15rem 0.6rem;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cream-theme overrides — only injected when active.
_CREAM_CSS = """
<style>
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"],
  [data-testid="stMain"],
  .main, .block-container {
    background-color: #F5F1E8 !important;
    color: #2C2A26 !important;
  }
  [data-testid="stSidebar"],
  [data-testid="stSidebarContent"],
  section[data-testid="stSidebar"] > div {
    background-color: #EFE9D7 !important;
  }
  h1, h2, h3, h4, h5, h6,
  p, label, span:not([data-baseweb]), li, td, th,
  [data-testid="stMarkdownContainer"],
  [data-testid="stCaptionContainer"] {
    color: #2C2A26 !important;
  }
  /* Metric labels and values */
  [data-testid="stMetricLabel"] { color: #6b6862 !important; }
  [data-testid="stMetricValue"] { color: #2C2A26 !important; }
  /* Streamlit primary buttons keep teal accent */
  .stButton button[kind="primary"] {
    background-color: #00C49F !important;
    color: white !important;
    border: 1px solid #00C49F !important;
  }
  /* Secondary buttons (incl. theme toggle) */
  .stButton button[kind="secondary"] {
    background-color: #EFE9D7 !important;
    color: #2C2A26 !important;
    border: 1px solid #D9D3C0 !important;
  }
  /* Inputs */
  input, textarea,
  [data-baseweb="input"] > div,
  [data-baseweb="select"] > div,
  [data-baseweb="textarea"] {
    background-color: #FAF6E9 !important;
    color: #2C2A26 !important;
  }
  /* Radios + sliders */
  [data-baseweb="radio"] div { color: #2C2A26 !important; }
  /* Code blocks + dividers */
  code, pre { background-color: #EFE9D7 !important; color: #2C2A26 !important; }
  hr { border-color: #D9D3C0 !important; }
  /* Expander headers */
  [data-testid="stExpander"] summary { color: #2C2A26 !important; }
  /* Dataframe headers — dark cream so they stand out from the table body.
     First block targets Streamlit's glide-data-grid widget (unstyled
     DataFrames render via canvas; these CSS vars are honoured by the
     wrapper). Second block targets HTML tables (pandas Styler output). */
  [data-testid="stDataFrame"], [data-testid="stTable"] {
    --gdg-bg-header: #C9C2A8 !important;
    --gdg-bg-header-has-focus: #BFB89E !important;
    --gdg-bg-header-hovered: #BFB89E !important;
    --gdg-text-header: #2C2A26 !important;
    --gdg-text-header-selected: #2C2A26 !important;
    --gdg-bg-cell: #F5F1E8 !important;
    --gdg-bg-cell-medium: #EFE9D7 !important;
    --gdg-text-dark: #2C2A26 !important;
    --gdg-text-medium: #4a4742 !important;
    --gdg-horizontal-border-color: #D9D3C0 !important;
  }
  [data-testid="stDataFrame"] thead,
  [data-testid="stDataFrame"] thead th,
  [data-testid="stTable"] thead,
  [data-testid="stTable"] thead th,
  .stDataFrame thead, .stDataFrame thead th,
  .stTable thead, .stTable thead th {
    background-color: #C9C2A8 !important;
    color: #2C2A26 !important;
  }
</style>
"""

if st.session_state["theme"] == "cream":
    st.markdown(_CREAM_CSS, unsafe_allow_html=True)

# Top-right toggle. The narrow right column keeps the button compact.
_, col_toggle = st.columns([11, 1])
with col_toggle:
    is_dark = st.session_state["theme"] == "dark"
    label = "☀ Cream" if is_dark else "☾ Dark"
    if st.button(label, key="theme_toggle",
                 help="Switch to cream theme" if is_dark else "Switch back to dark theme"):
        st.session_state["theme"] = "cream" if is_dark else "dark"
        st.rerun()

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
