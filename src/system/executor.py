"""Order Execution Engine"""
import logging
import time
from datetime import datetime
from typing import Dict, Optional
from queue import Queue

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, api_key: str = None, api_secret: str = None,
                 message_bus: Optional[Queue] = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.message_bus = message_bus or Queue()
        self.testnet = testnet
        self.base_url = "https://testnet.binance.vision/api/v3" if testnet else "https://api.binance.com/api/v3"
        self.active_orders = {}
        
    def process_order_request(self, order_request: Dict) -> Optional[Dict]:
        """Process order request from risk manager"""
        # Simulate order for testing
        order_response = {
            'symbol': order_request['symbol'],
            'orderId': int(time.time() * 1000),
            'status': 'FILLED',
            'side': order_request['side'],
            'executedQty': order_request['quantity'],
            'price': order_request['limit_price']
        }
        self._emit_order_status(order_response)
        return order_response
        
    def _emit_order_status(self, order_status: Dict):
        """Emit order status to message bus"""
        event = {
            'type': 'ORDER_STATUS',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'order_id': order_status['orderId'],
                'symbol': order_status['symbol'],
                'status': order_status['status'],
                'side': order_status.get('side'),
                'filled_quantity': float(order_status.get('executedQty', 0)),
                'filled_price': float(order_status.get('price', 0))
            }
        }
        self.message_bus.put(event)
        logger.info(f"Order status emitted: {order_status['orderId']} - {order_status['status']}")
