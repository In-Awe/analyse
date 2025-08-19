"""
Risk Management Module
Manages position sizing, stop-loss, and portfolio risk
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from queue import Queue
import numpy as np
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class RiskManager:
    """Manage trading risk and position sizing"""
    
    def __init__(self, config_path: str = "configs/risk_limits.yaml", 
                 message_bus: Optional[Queue] = None):
        self.message_bus = message_bus or Queue()
        self.config = self._load_config(config_path)
        
        # Portfolio state
        self.portfolio = {
            'balance': 10000.0,  # USDT
            'position': 0.0,     # BTC
            'avg_entry_price': 0.0,
            'total_pnl': 0.0,
            'open_pnl': 0.0
        }
        
        # Risk parameters
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_daily_loss = self.config.get('max_daily_loss', 500.0)  # $500
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)  # 4%
        
        # Kelly Criterion parameters
        self.use_kelly = self.config.get('use_kelly_criterion', True)
        self.kelly_fraction = 0.25  # Use 25% of Kelly for safety
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load risk configuration"""
        config_file = Path(config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                
        logger.warning("Using default risk configuration")
        return {
            'max_position_size': 0.1,
            'max_daily_loss': 500.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'use_kelly_criterion': True
        }
            
    def process_signal(self, signal: Dict) -> Optional[Dict]:
        """Process trading signal and determine position size"""
        # Check daily reset
        self._check_daily_reset()
        
        # Check if we can trade
        if not self._can_trade():
            logger.warning("Risk limits exceeded, skipping signal")
            return None
            
        # Determine position size
        position_size = self._calculate_position_size(signal)
        
        if position_size == 0:
            logger.debug("Position size is zero, no trade")
            return None
            
        # Calculate risk parameters
        current_price = signal.get('current_price', 0)
        if current_price == 0:
            current_price = 45000  # Default for testing
            
        stop_loss, take_profit = self._calculate_stops(
            signal['signal'], 
            current_price
        )
        
        # Create order request
        order_request = {
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'side': 'BUY' if signal['signal'] == 'UP' else 'SELL',
            'quantity': abs(position_size),
            'order_type': 'LIMIT',
            'limit_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_params': {
                'max_loss': position_size * current_price * self.stop_loss_pct,
                'position_pct': abs(position_size * current_price / self.portfolio['balance']),
                'confidence': signal.get('confidence', 0)
            }
        }
        
        # Emit order request
        self._emit_order_request(order_request)
        
        return order_request
        
    def _can_trade(self) -> bool:
        """Check if we can place new trades"""
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
            
        # Check if we have sufficient balance
        if self.portfolio['balance'] < 100:  # Minimum $100
            logger.warning(f"Insufficient balance: {self.portfolio['balance']}")
            return False
            
        return True
        
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size using Kelly Criterion or fixed sizing"""
        confidence = signal.get('confidence', 0.5)
        
        if self.use_kelly:
            # Kelly Criterion: f = (p*b - q) / b
            # p = probability of win (confidence)
            # q = probability of loss (1 - confidence)
            # b = win/loss ratio
            
            p = confidence
            q = 1 - confidence
            b = self.take_profit_pct / self.stop_loss_pct  # Win/loss ratio
            
            kelly_pct = (p * b - q) / b
            
            # Apply Kelly fraction for safety
            position_pct = max(0, min(kelly_pct * self.kelly_fraction, self.max_position_size))
            
        else:
            # Fixed position sizing based on confidence
            position_pct = self.max_position_size * confidence
            
        # Calculate actual position size in BTC
        current_price = signal.get('current_price', 45000)
        position_value = self.portfolio['balance'] * position_pct
        position_size = position_value / current_price
        
        # Apply minimum trade size (Binance minimum)
        min_trade_size = 0.00001  # BTC
        if abs(position_size) < min_trade_size:
            return 0
            
        # Determine direction
        if signal['signal'] == 'DOWN':
            position_size = -position_size
            
        return position_size
        
    def _calculate_stops(self, signal_type: str, current_price: float) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels"""
        if signal_type == 'UP':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif signal_type == 'DOWN':
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        else:  # SIDEWAYS
            stop_loss = 0
            take_profit = 0
            
        return stop_loss, take_profit
        
    def update_portfolio(self, order_status: Dict):
        """Update portfolio based on order execution"""
        if order_status['status'] == 'FILLED':
            filled_qty = order_status['filled_quantity']
            filled_price = order_status['filled_price']
            commission = order_status.get('commission', 0)
            
            # Update position
            if order_status['side'] == 'BUY':
                self.portfolio['position'] += filled_qty
                self.portfolio['balance'] -= (filled_qty * filled_price + commission)
            else:  # SELL
                self.portfolio['position'] -= filled_qty
                self.portfolio['balance'] += (filled_qty * filled_price - commission)
                
            # Update average entry price
            if self.portfolio['position'] != 0:
                # Weighted average
                self.portfolio['avg_entry_price'] = filled_price  # Simplified
                
            # Update daily trades
            self.daily_trades += 1
            
            logger.info(f"Portfolio updated: Balance={self.portfolio['balance']:.2f}, "
                       f"Position={self.portfolio['position']:.8f}")
                       
    def calculate_pnl(self, current_price: float) -> Dict:
        """Calculate current P&L"""
        # Open P&L
        if self.portfolio['position'] != 0:
            self.portfolio['open_pnl'] = (
                self.portfolio['position'] * 
                (current_price - self.portfolio['avg_entry_price'])
            )
        else:
            self.portfolio['open_pnl'] = 0
            
        return {
            'open_pnl': self.portfolio['open_pnl'],
            'total_pnl': self.portfolio['total_pnl'],
            'daily_pnl': self.daily_pnl
        }
        
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.last_reset = current_date
            logger.info("Daily risk counters reset")
            
    def _emit_order_request(self, order_request: Dict):
        """Emit order request to message bus"""
        event = {
            'type': 'ORDER_REQUEST',
            'timestamp': order_request['timestamp'].isoformat(),
            'data': order_request
        }
        
        self.message_bus.put(event)
        logger.info(f"Order request emitted: {order_request['side']} {order_request['quantity']:.8f} BTC")
        
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'portfolio': self.portfolio.copy(),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'risk_utilization': abs(self.portfolio['position'] * 45000 / self.portfolio['balance']),
            'can_trade': self._can_trade()
        }
