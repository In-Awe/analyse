import pytest
from src.system.risk_manager import RiskManager

def test_risk_manager_init():
    manager = RiskManager()
    assert manager.portfolio['balance'] == 10000.0
    assert manager.max_position_size == 0.1
