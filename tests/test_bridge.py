def test_bridge_imports():
    from core import bridge  # noqa: F401
    assert hasattr(bridge, "get_stock_price_data")
    assert hasattr(bridge, "load_set_tickers")
