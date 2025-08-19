"""Market Data Handler (WebSocket) - skeleton

Provides a lightweight WebSocket client to subscribe to Binance kline 1m stream and
emit normalized candle events to an internal queue.
"""
import asyncio
import json
from typing import Callable

class MarketDataHandler:
    def __init__(self, symbol: str, event_emitter: Callable[[dict], None]):
       self.symbol = symbol.upper()
       self.event_emitter = event_emitter
       self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_1m"

    async def connect(self):
       import websockets
       async with websockets.connect(self.ws_url) as ws:
           async for msg in ws:
               data = json.loads(msg)
               candle = self._parse_kline(data)
               self.event_emitter(candle)

    def _parse_kline(self, raw: dict) -> dict:
       k = raw.get('k', {})
       return {
           'type': 'candle',
           'symbol': self.symbol,
           'ts': int(k.get('t', 0)),
           'open': float(k.get('o', 0)),
           'high': float(k.get('h', 0)),
           'low': float(k.get('l', 0)),
           'close': float(k.get('c', 0)),
           'volume': float(k.get('v', 0)),
           'is_closed': k.get('x', False)
       }

    def run(self):
       asyncio.get_event_loop().run_until_complete(self.connect())

if __name__ == '__main__':
    def print_ev(e):
       print(e)
    m = MarketDataHandler('BTCUSDT', print_ev)
    m.run()
