"""
Market Data Handler Module
Real-time market data ingestion and preprocessing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Callable
import websocket
import threading
from queue import Queue
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Handles real-time market data from Binance WebSocket"""
    
    def __init__(self, symbol: str = "btcusdt", message_bus: Optional[Queue] = None):
        self.symbol = symbol.lower()
        self.message_bus = message_bus or Queue()
        self.ws = None
        self.running = False
        
        # Feature calculation windows
        self.price_buffer = []
        self.volume_buffer = []
        self.buffer_size = 120  # 2 hours of 1-minute data
        
        # WebSocket URLs
        self.ws_base_url = "wss://stream.binance.com:9443/ws"
        self.streams = [
            f"{self.symbol}@kline_1m",
            f"{self.symbol}@bookTicker",
            f"{self.symbol}@aggTrade"
        ]
        
    def connect(self):
        """Establish WebSocket connection"""
        url = f"{self.ws_base_url}/{'/'.join(self.streams)}"
        logger.info(f"Connecting to Binance WebSocket: {url}")
        
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Run in separate thread
        self.running = True
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        
    def _on_message(self, ws, message):
        """Process incoming market data"""
        try:
            data = json.loads(message)
            stream = data.get('stream', '')
            
            if 'kline' in stream:
                self._process_kline(data['data'])
            elif 'bookTicker' in stream:
                self._process_book_ticker(data['data'])
            elif 'aggTrade' in stream:
                self._process_trade(data['data'])
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _process_kline(self, data):
        """Process candlestick data"""
        kline = data['k']
        
        # Extract OHLCV data
        market_data = {
            'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
            'symbol': kline['s'],
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': kline['x']
        }
        
        # Update buffers
        self.price_buffer.append(market_data['close'])
        self.volume_buffer.append(market_data['volume'])
        
        # Maintain buffer size
        if len(self.price_buffer) > self.buffer_size:
            self.price_buffer.pop(0)
            self.volume_buffer.pop(0)
            
        # Calculate features if we have enough data
        if len(self.price_buffer) >= 30:
            market_data['features'] = self._calculate_features()
            
        # Emit event
        self._emit_market_event(market_data)
        
    def _process_book_ticker(self, data):
        """Process best bid/ask data"""
        book_data = {
            'timestamp': datetime.now(),
            'symbol': data['s'],
            'bid_price': float(data['b']),
            'bid_qty': float(data['B']),
            'ask_price': float(data['a']),
            'ask_qty': float(data['A']),
            'spread': float(data['a']) - float(data['b'])
        }
        
    def _process_trade(self, data):
        """Process aggregated trade data"""
        trade_data = {
            'timestamp': datetime.fromtimestamp(data['T'] / 1000),
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'is_buyer_maker': data['m']
        }
        
    def _calculate_features(self) -> Dict:
        """Calculate real-time features from buffers"""
        features = {}
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        
        # Price-based features
        features['sma_10'] = np.mean(prices[-10:])
        features['sma_30'] = np.mean(prices[-30:])
        features['ema_10'] = self._calculate_ema(prices, 10)
        
        # Momentum indicators
        features['rsi_14'] = self._calculate_rsi(prices, 14)
        
        # Volatility
        returns = np.diff(np.log(prices[-30:]))
        features['volatility'] = np.std(returns) * np.sqrt(1440)  # Annualized
        
        # Volume features
        features['volume_ma'] = np.mean(volumes[-30:])
        features['vwap'] = np.sum(prices[-30:] * volumes[-30:]) / np.sum(volumes[-30:])
        
        return features
        
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.nan
            
        alpha = 2 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period+1:]:
            ema = price * alpha + ema * (1 - alpha)
            
        return ema
        
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.nan
            
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _emit_market_event(self, data: Dict):
        """Emit market data event to message bus"""
        event = {
            'type': 'MARKET_DATA',
            'timestamp': data['timestamp'].isoformat(),
            'data': data
        }
        
        self.message_bus.put(event)
        logger.debug(f"Emitted market event: {data['symbol']} @ {data.get('close', 'N/A')}")
        
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        
    def _on_close(self, ws):
        """Handle WebSocket close"""
        logger.info("WebSocket connection closed")
        self.running = False
        
        # Attempt reconnection
        if self.running:
            logger.info("Attempting to reconnect...")
            threading.Timer(5.0, self.connect).start()
            
    def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
            
    def get_latest_price(self) -> Optional[float]:
        """Get the most recent price"""
        if self.price_buffer:
            return self.price_buffer[-1]
        return None


if __name__ == "__main__":
    # Test market data handler
    logging.basicConfig(level=logging.INFO)
    
    handler = MarketDataHandler(symbol="btcusdt")
    handler.connect()
    
    try:
        # Run for 60 seconds
        import time
        time.sleep(60)
    finally:
        handler.disconnect()
