"""Simple token-bucket rate limiter for Binance API calls
"""
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
       self.rate = rate
       self.capacity = capacity
       self._tokens = capacity
       self._last = time.time()
       self._lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
       with self._lock:
           now = time.time()
           delta = now - self._last
           self._tokens = min(self.capacity, self._tokens + delta * self.rate)
           self._last = now
           if self._tokens >= tokens:
               self._tokens -= tokens
               return True
           return False

if __name__ == '__main__':
    tb = TokenBucket(rate=5, capacity=10)
    print(tb.consume())
