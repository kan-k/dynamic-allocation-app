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
  /* Text inputs / text areas / number inputs */
  input, textarea,
  [data-baseweb="input"] > div,
  [data-baseweb="textarea"] {
    background-color: #FAF6E9 !important;
    color: #2C2A26 !important;
    border-color: #D9D3C0 !important;
  }
  /* Selectbox (closed state) — the dropdown "header" you see when collapsed */
  [data-baseweb="select"] > div,
  div[data-baseweb="select"] [role="combobox"] {
    background-color: #C9C2A8 !important;
    color: #2C2A26 !important;
    border-color: #BFB89E !important;
  }
  [data-baseweb="select"] svg { fill: #2C2A26 !important; }
  /* Selectbox open popover menu */
  [data-baseweb="popover"] [role="listbox"],
  [data-baseweb="popover"] [role="option"],
  [data-baseweb="menu"] li {
    background-color: #FAF6E9 !important;
    color: #2C2A26 !important;
  }
  [data-baseweb="popover"] [role="option"]:hover,
  [data-baseweb="menu"] li:hover {
    background-color: #EFE9D7 !important;
  }
  /* Checkbox — outer square */
  [data-baseweb="checkbox"] > span:first-child,
  [data-baseweb="checkbox"] > div:first-child > div {
    background-color: #FAF6E9 !important;
    border-color: #BFB89E !important;
  }
  /* Checkbox — checked state (teal fill) */
  [data-baseweb="checkbox"][aria-checked="true"] > span:first-child,
  [data-baseweb="checkbox"][aria-checked="true"] > div:first-child > div {
    background-color: #00C49F !important;
    border-color: #00C49F !important;
  }
  [data-baseweb="checkbox"] label, [data-baseweb="checkbox"] span {
    color: #2C2A26 !important;
  }
  /* Radio buttons */
  [data-baseweb="radio"] > div:first-child > div {
    background-color: #FAF6E9 !important;
    border-color: #BFB89E !important;
  }
  [data-baseweb="radio"][aria-checked="true"] > div:first-child > div {
    background-color: #00C49F !important;
    border-color: #00C49F !important;
  }
  [data-baseweb="radio"] label, [data-baseweb="radio"] span,
  [data-baseweb="radio"] div {
    color: #2C2A26 !important;
  }
  /* Slider */
  [data-baseweb="slider"] [role="slider"] { background-color: #00C49F !important; }
  /* Tabs (the in-page st.tabs widget) */
  [data-baseweb="tab-list"] {
    background-color: #EFE9D7 !important;
  }
  [data-baseweb="tab"] {
    color: #2C2A26 !important;
  }
  /* Code blocks + dividers */
  code, pre { background-color: #EFE9D7 !important; color: #2C2A26 !important; }
  hr { border-color: #D9D3C0 !important; }
  /* Expander summary bar — Streamlit paints this with a dark surface
     even in cream mode, so make the summary text + chevron light cream
     so they're readable on the dark strip. (Body content of the
     expander still inherits the page's cream/dark text rules above.)  */
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span,
  [data-testid="stExpander"] summary strong,
  [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"],
  [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
  [data-testid="stExpander"] details > summary,
  [data-testid="stExpander"] details > summary * {
    color: #F5F1E8 !important;
  }
  [data-testid="stExpander"] summary svg,
  [data-testid="stExpander"] details > summary svg {
    fill: #F5F1E8 !important;
  }
  /* Popover trigger buttons (ℹ, 📡) — same dark-trigger family */
  [data-testid="stPopover"] button,
  [data-testid="stPopover"] button *,
  [data-testid="stPopoverButton"],
  [data-testid="stPopoverButton"] * {
    color: #F5F1E8 !important;
  }
  /* Table headers only — change just the header TEXT to a light cream
     so it's readable regardless of which background paint Streamlit
     gives the header (cream-bg override is unreliable on the canvas-
     rendered glide-data-grid widget, so we don't fight it).
     Body cells continue to take the page's cream background via the
     other rules above. */
  [data-testid="stDataFrame"],
  [data-testid="stTable"],
  [data-testid="stDataEditor"],
  [data-testid="stDataFrameResizable"] {
    --gdg-text-header: #F5F1E8 !important;
    --gdg-text-header-selected: #F5F1E8 !important;
    --gdg-text-dark: #2C2A26 !important;
    --gdg-bg-cell: #F5F1E8 !important;
  }
  [data-testid="stDataFrame"] thead th,
  [data-testid="stTable"] thead th,
  [data-testid="stDataEditor"] thead th,
  .stDataFrame thead th,
  .stTable thead th {
    color: #F5F1E8 !important;
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
