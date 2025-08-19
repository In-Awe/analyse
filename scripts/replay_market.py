#!/usr/bin/env python3
"""Market Replay Script"""
import pandas as pd
from pathlib import Path
from queue import Queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketReplay:
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.message_queue = Queue()
        
    def replay(self):
        """Replay historical market data"""
        if not self.data_file.exists():
            logger.error(f"Data file not found: {self.data_file}")
            return
            
        df = pd.read_csv(self.data_file)
        logger.info(f"Replaying {len(df)} rows of market data")
        
        for idx, row in df.iterrows():
            event = {
                'type': 'MARKET_DATA',
                'data': row.to_dict()
            }
            self.message_queue.put(event)
            
        logger.info("Replay complete")

if __name__ == "__main__":
    replay = MarketReplay("data/cleaned/test_data.csv")
    replay.replay()
