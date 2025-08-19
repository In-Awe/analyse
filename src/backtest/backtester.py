"""
Vectorized backtester skeleton.

Main class: VectorBacktester

Inputs:
- df: pandas DataFrame indexed by timestamp with columns: open/high/low/close/volume and a 'signal' column (BUY/SELL/FLAT)
- config: dict loaded from YAML (fees, slippage.alpha, execution, backtest)
Outputs:
- trades: DataFrame with executed trades
- equity: DataFrame with equity curve (timestamp, equity)
- summary: dict
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
from typing import Tuple, Dict

EPS = 1e-12

@dataclass
class VectorBacktester:
    df: pd.DataFrame
    config: Dict

    def __post_init__(self):
        required = ['open', 'high', 'low', 'close', 'volume']
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"input df missing required column {c}")
        self.df = self.df.copy()
        # ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'timestamp' in self.df.columns:
                self.df = self.df.set_index(pd.to_datetime(self.df['timestamp']))
            else:
                raise ValueError("df must have DatetimeIndex or timestamp column")
        # create ATR for slippage model
        self.df['tr'] = np.maximum(self.df['high'] - self.df['low'],
                                   np.maximum((self.df['high'] - self.df['close'].shift(1)).abs(),
                                              (self.df['low'] - self.df['close'].shift(1)).abs()))
        self.df['atr'] = self.df['tr'].rolling(14, min_periods=1).mean().fillna(0)

    def map_signal_to_target(self, s: pd.Series) -> pd.Series:
        mapping = self.config.get('backtest', {}).get('signal_to_position', {'BUY':1,'SELL':-1,'FLAT':0})
        return s.map(mapping).fillna(0).astype(float)

    def compute_execution_price(self) -> pd.Series:
        # execution at next-open + alpha * ATR_next
        alpha = float(self.config.get('slippage', {}).get('alpha', 0.5))
        # use next open and next atr
        next_open = self.df['open'].shift(-1)
        next_atr = self.df['atr'].shift(-1).ffill().fillna(0.0)
        exec_price = next_open + alpha * next_atr
        # if next_open is NaN (end of series), fallback to current close
        exec_price = exec_price.fillna(self.df['close'])
        return exec_price

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        df = self.df.copy()
        if 'signal' not in df.columns:
            raise ValueError("df must contain a 'signal' column before running the backtester")

        df['target_pos'] = self.map_signal_to_target(df['signal'])
        df['target_pos_prev'] = df['target_pos'].shift(1).fillna(0.0)
        df['trade_qty'] = df['target_pos'] - df['target_pos_prev']

        df['exec_price'] = self.compute_execution_price()

        df['notional'] = df['trade_qty'] * df['exec_price']

        min_notional = float(self.config.get('execution', {}).get('min_notional', 0.0))
        is_trade = (df['notional'].abs() >= min_notional) & (df['trade_qty'] != 0)

        fees_cfg = self.config.get('fees', {})
        taker_fee = float(fees_cfg.get('taker', 0.0004))
        df['fees'] = df['notional'].abs() * taker_fee

        df['cash_flow'] = -df['notional'] - df['fees']

        initial_cash = 100000.0
        df['position'] = df['trade_qty'].where(is_trade, 0).cumsum()
        df['cash'] = initial_cash + df['cash_flow'].where(is_trade, 0).cumsum()
        df['equity'] = df['cash'] + df['position'] * df['close']

        trades_df = df.loc[is_trade].copy()
        trades_df.rename(columns={'trade_qty': 'qty', 'exec_price': 'price'}, inplace=True)
        trades_df['timestamp'] = trades_df.index
        trades_df['cash_after'] = trades_df['cash']
        trades_df['position_after'] = trades_df['position']
        trades_df = trades_df[['timestamp', 'qty', 'price', 'notional', 'cash_after', 'position_after']]

        equity_df = df[['equity', 'cash', 'position']].copy()
        equity_df['timestamp'] = equity_df.index
        equity_df = equity_df[['timestamp', 'equity', 'cash', 'position']]

        returns = equity_df['equity'].pct_change().fillna(0.0)
        avg_ret = float(returns.mean())
        vol = float(returns.std())
        sharpe = float((avg_ret / (vol + EPS)) * np.sqrt(252*24*60)) if vol > 0 else 0.0

        summary = {
            'start_equity': float(equity_df['equity'].iloc[0]) if not equity_df.empty else 0.0,
            'end_equity': float(equity_df['equity'].iloc[-1]) if not equity_df.empty else 0.0,
            'total_trades': int(len(trades_df)),
            'sharpe_minute_annualized_rough': sharpe
        }

        return trades_df, equity_df, summary
