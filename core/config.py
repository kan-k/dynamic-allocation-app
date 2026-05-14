import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PARENT = os.path.abspath(os.path.join(_ROOT, ".."))

DB_PATH = os.path.join(_ROOT, "data", "trading.duckdb")
TICKER_UNIVERSE_PATH = os.path.join(_PARENT, "data", "set_tickers.json")
FAMA_CSV_DIR = os.path.join(_PARENT, "data", "financial_statement")

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
YFINANCE_SUFFIX = ".BK"

GLOBAL_ETF_PERIOD = "10y"
GLOBAL_ETFS_PATH = os.path.join(_ROOT, "data", "global_etfs.json")
