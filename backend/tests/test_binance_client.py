import pytest
from unittest.mock import MagicMock
from backend.binance_client import BinanceClient

def test_binance_client_init(mocker):
    """
    Test that the BinanceClient can be initialized without making a network call.
    """
    # Patch the Client class within the binance_client module
    mocker.patch('backend.binance_client.Client', return_value=MagicMock())
    c = BinanceClient(api_key=None, api_secret=None)
    assert c is not None

def test_get_historical_klines_public_call(mocker):
    """
    This test verifies the method can be accessed without calling the network.
    """
    # Patch the Client class
    mock_client_instance = MagicMock()
    mocker.patch('backend.binance_client.Client', return_value=mock_client_instance)

    c = BinanceClient(api_key=None, api_secret=None)
    assert hasattr(c, "get_historical_klines")

    # Example of how you could test a call if you wanted to
    # c.get_historical_klines("BTCUSDT", "1m", "1 day ago UTC")
    # mock_client_instance.get_historical_klines.assert_called_once_with("BTCUSDT", "1m", "1 day ago UTC")
