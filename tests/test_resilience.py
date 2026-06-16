import time
import pytest
import asyncio
from shared.breaker import CircuitBreaker, CircuitBreakerOpenException
from shared.monitor import SystemResourceMonitor

# --- Circuit Breaker Tests ---

# A simple dummy function that we wrap in the circuit breaker to simulate success/failure
async def dummy_succeed(value):
    return value

async def dummy_fail():
    raise ConnectionError("Mock connection failure")

@pytest.mark.asyncio
async def test_circuit_breaker_flow():
    """Verify state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    # We initialize a breaker with 2 failures threshold and 0.5s recovery timeout for fast tests
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
    
    # 1. Starts CLOSED
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0
    
    # 2. First failure (increases failure count)
    with pytest.raises(ConnectionError):
        await breaker.call(dummy_fail)
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 1
    
    # 3. Second failure (trips to OPEN because threshold is 2)
    with pytest.raises(ConnectionError):
        await breaker.call(dummy_fail)
    assert breaker.state == "OPEN"
    assert breaker.failure_count == 2
    
    # 4. Immediate call when OPEN should trigger fail-fast (CircuitBreakerOpenException)
    with pytest.raises(CircuitBreakerOpenException):
        await breaker.call(dummy_succeed, "test")
        
    # 5. Wait for recovery timeout (0.5s) to elapse
    await asyncio.sleep(0.6)
    
    # 6. Call now should transition to HALF_OPEN and test success
    result = await breaker.call(dummy_succeed, "success_value")
    assert result == "success_value"
    
    # 7. A successful call in HALF_OPEN resets state to CLOSED and failure count to 0
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


# --- Resource Monitor Tests ---

def test_resource_monitor_metrics():
    """Verify that SystemResourceMonitor reads system RSS memory and handles successfully."""
    # Initialize monitor with custom limits
    monitor = SystemResourceMonitor(memory_limit_mb=500.0, fd_limit=100)
    
    mem_usage = monitor.get_memory_usage_mb()
    fds_count = monitor.get_file_descriptor_count()
    
    # Memory and Handles should be positive numbers
    assert isinstance(mem_usage, float)
    assert mem_usage > 0.0
    
    assert isinstance(fds_count, int)
    assert fds_count > 0

def test_resource_monitor_limits_pass():
    """Test that check_limits returns correct dict schema and status when under limits."""
    # We set limits very high so they pass
    monitor = SystemResourceMonitor(memory_limit_mb=50000.0, fd_limit=5000)
    
    metrics = monitor.check_limits()
    
    assert metrics["memory_ok"] is True
    assert metrics["handles_ok"] is True
    assert metrics["action_taken"] == "none"
