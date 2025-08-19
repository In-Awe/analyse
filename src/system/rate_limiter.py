"""Rate Limiter Module"""
import time
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class TokenBucket:
    def __init__(self, capacity: int, refill_period: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_period = refill_period
        self.last_refill = time.time()
        self.lock = Lock()
        
    def consume(self, tokens: int = 1) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    def _refill(self):
        current_time = time.time()
        time_passed = current_time - self.last_refill
        tokens_to_add = (time_passed / self.refill_period) * self.capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time

class RateLimiter:
    def __init__(self):
        self.request_weight = TokenBucket(1200, 60)  # 1200 per minute
        self.order_second = TokenBucket(10, 1)       # 10 per second
        self.order_day = TokenBucket(200000, 86400)  # 200k per day
        
    def can_make_request(self, endpoint: str, is_order: bool = False) -> bool:
        weight = 1  # Simplified
        if not self.request_weight.consume(weight):
            return False
        if is_order:
            if not self.order_second.consume(1):
                return False
            if not self.order_day.consume(1):
                return False
        return True
