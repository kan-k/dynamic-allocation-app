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
# Deliberately MINIMAL: we only restyle the page chrome + a few specific
# components (expander/popover triggers, primary buttons, table headers).
# All BaseWeb widgets (checkboxes, radios, selectboxes, sliders, tabs,
# text inputs) keep Streamlit's native dark styling — they appear as dark
# islands on the cream page, but they FUNCTION CORRECTLY (the checked
# state of checkboxes renders properly) and don't trigger the expensive
# style recalculation that broad widget overrides caused.
_CREAM_CSS = """
<style>
  /* Page chrome — light cream */
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"],
  [data-testid="stMain"],
  .main, .block-container {
    background-color: #F5F1E8 !important;
  }
  [data-testid="stSidebar"] > div {
    background-color: #EFE9D7 !important;
  }
  /* Body text — narrowly targeted, no wildcards or :not() selectors */
  [data-testid="stMain"] h1,
  [data-testid="stMain"] h2,
  [data-testid="stMain"] h3,
  [data-testid="stMain"] h4,
  [data-testid="stMain"] h5,
  [data-testid="stMain"] h6,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] strong,
  [data-testid="stCaptionContainer"] {
    color: #2C2A26 !important;
  }
  [data-testid="stMetricLabel"] { color: #6b6862 !important; }
  [data-testid="stMetricValue"] { color: #2C2A26 !important; }

  /* Primary teal button (▶ Run …) */
  .stButton button[kind="primary"] {
    background-color: #00C49F !important;
    color: white !important;
    border: 1px solid #00C49F !important;
  }
  .stButton button[kind="primary"] p,
  .stButton button[kind="primary"] [data-testid="stMarkdownContainer"] p {
    color: white !important;
  }

  /* Secondary buttons (theme toggle, Select all, Clear, ↺ Refresh, etc.)
     — Streamlit's native dark background fights our cream page; restyle
     them to match. */
  .stButton button[kind="secondary"],
  .stButton button:not([kind]) {
    background-color: #EFE9D7 !important;
    color: #2C2A26 !important;
    border: 1px solid #D9D3C0 !important;
  }
  .stButton button[kind="secondary"] p,
  .stButton button:not([kind]) p,
  .stButton button[kind="secondary"] [data-testid="stMarkdownContainer"] p,
  .stButton button:not([kind]) [data-testid="stMarkdownContainer"] p {
    color: #2C2A26 !important;
  }

  /* Sidebar navigation links (Market Overview, Dynamic Allocation, ...)
     — without this rule they keep Streamlit's native off-white text and
     disappear into the cream sidebar background. */
  [data-testid="stSidebar"] a,
  [data-testid="stSidebar"] a p,
  [data-testid="stSidebar"] a span,
  [data-testid="stSidebarNav"] a,
  [data-testid="stSidebarNav"] li,
  [data-testid="stSidebarNav"] span {
    color: #2C2A26 !important;
  }

  /* Code blocks + dividers */
  code, pre { background-color: #EFE9D7 !important; color: #2C2A26 !important; }
  hr { border-color: #D9D3C0 !important; }

  /* Expander summary — force dark strip in BOTH collapsed and expanded
     states (Streamlit only paints it dark when expanded). Light-cream
     text + chevron stay on it for readability. */
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] details > summary {
    background-color: #1A1D26 !important;
    border-radius: 0.4rem !important;
  }
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span,
  [data-testid="stExpander"] summary strong,
  [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
    color: #F5F1E8 !important;
  }
  [data-testid="stExpander"] summary svg { fill: #F5F1E8 !important; }

  /* Popover trigger (ℹ, 📡) — same dark-trigger family as expanders */
  [data-testid="stPopover"] button {
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
