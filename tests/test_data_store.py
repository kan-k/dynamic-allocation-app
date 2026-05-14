import os
import pytest
import pandas as pd
from datetime import date


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.DB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setattr("core.data_store.DB_PATH", str(tmp_path / "test.duckdb"))


def test_init_and_prices_roundtrip():
    from core.data_store import init_db, upsert_prices, get_prices
    init_db()
    df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "open": [100.0], "high": [105.0], "low": [99.0],
        "close": [103.0], "volume": [1_000_000],
    })
    upsert_prices(df, "DELTA")
    result = get_prices("DELTA")
    assert len(result) == 1
    assert result.iloc[0]["close"] == 103.0
