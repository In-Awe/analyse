from backend.binance_client import BinanceClient

def test_binance_client_init():
    c = BinanceClient(api_key=None, api_secret=None)
    assert c is not None

def test_get_historical_klines_public_call():
    """
    This test does not call the network. It verifies the method can be accessed.
    For an integration test, supply real API keys and run manually.
    """
    c = BinanceClient(api_key=None, api_secret=None)
    assert hasattr(c, "get_historical_klines")
