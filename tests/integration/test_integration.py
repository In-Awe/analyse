import pytest
from queue import Queue

def test_message_bus():
    bus = Queue()
    bus.put({'type': 'TEST'})
    assert not bus.empty()
