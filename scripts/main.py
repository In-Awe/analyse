#!/usr/bin/env python3
"""Main Entry Point for HFT Trading System"""
import asyncio
import sys
import logging
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.system.market_data import MarketDataHandler
from src.system.signal_service import SignalService
from src.system.risk_manager import RiskManager
from src.system.executor import ExecutionEngine
from src.system.rate_limiter import RateLimiter
from src.system.monitoring import MonitoringService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        self.message_bus = Queue()
        self.running = False
        
    def initialize(self):
        logger.info("Initializing trading system...")
        self.market_handler = MarketDataHandler(message_bus=self.message_bus)
        self.signal_service = SignalService(message_bus=self.message_bus)
        self.risk_manager = RiskManager(message_bus=self.message_bus)
        self.executor = ExecutionEngine(message_bus=self.message_bus)
        self.rate_limiter = RateLimiter()
        self.monitor = MonitoringService()
        
    async def start(self):
        logger.info("Starting HFT Trading System")
        self.running = True
        # Main event loop would go here
        
    def stop(self):
        logger.info("Stopping trading system")
        self.running = False

if __name__ == "__main__":
    system = TradingSystem()
    system.initialize()
    # asyncio.run(system.start())
