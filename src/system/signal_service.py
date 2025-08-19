"""Signal Generation service skeleton

Consumes features and model outputs (or a placeholder rule-based signal) and emits signals.
"""
from typing import Callable
import time

class SignalService:
    def __init__(self, model_infer: Callable[[dict], dict], emitter: Callable[[dict], None]):
       self.model_infer = model_infer
       self.emitter = emitter

    def on_candle(self, candle: dict, features: dict = None):
       # placeholder: call model_infer with features and produce a signal
       # model_infer should return {'signal': 'BUY'|'SELL'|'FLAT', 'confidence': 0.0}
       result = self.model_infer(features or candle)
       signal = {
           'type': 'signal',
           'symbol': candle['symbol'],
           'ts': candle['ts'],
           'signal': result.get('signal', 'FLAT'),
           'confidence': float(result.get('confidence', 0.0))
       }
       self.emitter(signal)

def dummy_infer(features):
    # trivial rule: flat
    return {'signal': 'FLAT', 'confidence': 0.0}

if __name__ == '__main__':
    def print_ev(e):
       print('EMIT', e)
    s = SignalService(dummy_infer, print_ev)
    s.on_candle({'symbol': 'BTCUSDT', 'ts': int(time.time()*1000)})
