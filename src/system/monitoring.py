"""Monitoring helpers — Prometheus metrics export skeleton
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server

metrics = {
    'signals_total': Counter('signals_total', 'Number of signals emitted'),
    'trades_total': Counter('trades_total', 'Number of trades executed'),
    'execution_latency_ms': Histogram('execution_latency_ms', 'Latency in ms for execution'),
    'message_lag': Gauge('message_lag', 'Queue lag in ms'),
}

def start_metrics_server(port: int = 8000):
    start_http_server(port)

if __name__ == '__main__':
    start_metrics_server(8000)
    print('Prometheus metrics server started on :8000')
