#!/usr/bin/env python3
"""Generate test data for integration testing"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_test_data(rows=1000):
    """Generate synthetic OHLCV data"""
    print(f"Generating {rows} rows of test data...")
    
    start_time = datetime.now() - timedelta(minutes=rows)
    timestamps = [start_time + timedelta(minutes=i) for i in range(rows)]
    
    # Generate price data with realistic movement
    base_price = 45000
    prices = base_price + np.cumsum(np.random.randn(rows) * 100)
    
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLCV data
        open_price = price - np.random.rand() * 50
        high_price = price + abs(np.random.randn() * 100)
        low_price = price - abs(np.random.randn() * 100)
        close_price = price
        volume = 100 + abs(np.random.randn() * 50)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': max(high_price, open_price, close_price),
            'low': min(low_price, open_price, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Save files
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    
    # Save test data
    df.to_csv("data/cleaned/test_data.csv", index=False)
    print("  ✓ Created: data/cleaned/test_data.csv")
    
    # Save as BTCUSD
    df.to_csv("data/cleaned/BTCUSD_1min.cleaned.csv", index=False)
    print("  ✓ Created: data/cleaned/BTCUSD_1min.cleaned.csv")
    
    # Generate XRP data with different characteristics
    xrp_prices = 0.65 + np.cumsum(np.random.randn(rows) * 0.01)
    xrp_data = []
    for i, (ts, price) in enumerate(zip(timestamps, xrp_prices)):
        xrp_data.append({
            'timestamp': ts,
            'open': price - np.random.rand() * 0.005,
            'high': price + abs(np.random.randn() * 0.01),
            'low': price - abs(np.random.randn() * 0.01),
            'close': price,
            'volume': 1000 + abs(np.random.randn() * 500)
        })
    
    xrp_df = pd.DataFrame(xrp_data)
    xrp_df.to_csv("data/cleaned/XRPUSD_1min.csv", index=False)
    print("  ✓ Created: data/cleaned/XRPUSD_1min.csv")
    
    print(f"Test data generation complete! Created {rows} rows for each dataset.")

if __name__ == "__main__":
    generate_test_data(1000)
