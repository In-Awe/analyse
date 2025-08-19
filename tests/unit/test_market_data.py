import pytest
from src.system.market_data import MarketDataHandler

def test_market_data_init():
    handler = MarketDataHandler(symbol="btcusdt")
    assert handler.symbol == "btcusdt"
    assert handler.buffer_size == 120
