"""Monitoring and Telemetry Module"""
import logging
import json
from datetime import datetime
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self):
        self.metrics = {
            'market_data': deque(maxlen=1000),
            'signals': deque(maxlen=1000),
            'orders': deque(maxlen=1000),
            'errors': deque(maxlen=100)
        }
        self.system_status = {'overall': 'STARTING'}
        
    def record_market_data_event(self, event):
        self.metrics['market_data'].append({
            'timestamp': datetime.now().isoformat(),
            'data': event
        })
        self.system_status['market_data'] = 'HEALTHY'
        
    def record_signal_event(self, signal):
        self.metrics['signals'].append({
            'timestamp': datetime.now().isoformat(),
            'signal': signal
        })
        
    def record_order_event(self, order):
        self.metrics['orders'].append({
            'timestamp': datetime.now().isoformat(),
            'order': order
        })
        
    def get_system_health(self):
        return {
            'status': self.system_status,
            'metrics_collected': {k: len(v) for k, v in self.metrics.items()}
        }
