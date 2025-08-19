"""Risk Manager skeleton

Consumes signals and current portfolio/account state and outputs target positions.
"""
from typing import Dict
import yaml

class RiskManager:
    def __init__(self, cfg_path='configs/risk_limits.yaml'):
       with open(cfg_path, 'r') as f:
           self.cfg = yaml.safe_load(f)
       self.position = 0.0

    def apply(self, signal: Dict, account_state: Dict) -> Dict:
       # Basic sizing: cap position to max_position_size
       side = signal.get('signal', 'FLAT')
       confidence = signal.get('confidence', 0.0)
       target = 0.0
       if side == 'BUY' and confidence > 0.5:
           target = min(self.cfg.get('max_position_size', 1.0), confidence * self.cfg.get('max_position_size', 1.0))
       elif side == 'SELL' and confidence > 0.5:
           target = -min(self.cfg.get('max_position_size', 1.0), confidence * self.cfg.get('max_position_size', 1.0))
       # simple risk checks
       if account_state.get('unrealized_loss', 0) > self.cfg.get('max_unrealized_loss', 5000):
           target = 0.0
       return {'type': 'target_position', 'symbol': signal['symbol'], 'ts': signal['ts'], 'target': target}

if __name__ == '__main__':
    rm = RiskManager()
    print(rm.apply({'signal':'BUY','confidence':0.8,'symbol':'BTCUSDT','ts':0},{'unrealized_loss':0}))
