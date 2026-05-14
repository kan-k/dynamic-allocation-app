"""
Single import point for all parent analytics/ functions.
Pages must import from here — never manipulate sys.path directly.
"""
import sys
import os

_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from analytics.stock_and_prediction_funcs import (  # noqa: E402
    get_stock_price_data,
    get_stock_price_data_SET,
    get_stock_technical_indicators,
    get_stock_technical_indicators_simple_fast,
    get_intermarket_data,
    get_rolling_alpha_beta,
)
from analytics.screening_funcs import (  # noqa: E402
    get_ff_Screened_SET,
    get_ff_Screened_SET100,
    get_Screened_SET,
)
from analytics.port_optim_funcs import portfolio_semivar2  # noqa: E402
from analytics.random_forest_pipeline import random_forest_investment_pipeline  # noqa: E402
from analytics.utils import load_set_tickers, mute_yfinance  # noqa: E402
