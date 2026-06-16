import time
import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("circuit_breaker")

class CircuitBreakerOpenException(Exception):
    """Raised when a request is blocked by the circuit breaker because it is in the OPEN state."""
    pass

class CircuitBreaker:
    """A custom client-side Circuit Breaker pattern implementation.
    
    Acts as a wrapper around async calls (like database queries or queue pushes)
    to detect consecutive failures, fail-fast when the service is down,
    and automatically test for recovery.
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        # State can be: "CLOSED" (normal), "OPEN" (tripped), "HALF_OPEN" (testing)
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_state_change = time.time()
        
    def _transition_to(self, new_state: str):
        """Helper to cleanly transition states and log the event."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        logger.warning(f"CircuitBreaker transition: {old_state} ⇄ {new_state}")

    async def call(self, async_func: Callable[..., Any], *args, **kwargs) -> Any:
        """Executes an async function wrapped in the circuit breaker logic."""
        current_time = time.time()
        
        # 1. If the circuit is OPEN, check if it's time to try recovering (Half-Open)
        if self.state == "OPEN":
            elapsed_time = current_time - self.last_state_change
            if elapsed_time >= self.recovery_timeout:
                logger.info(f"Recovery timeout of {self.recovery_timeout}s elapsed. Probing service.")
                self._transition_to("HALF_OPEN")
            else:
                # Still in cooldown period: fail-fast!
                remaining = self.recovery_timeout - elapsed_time
                logger.debug(f"Request blocked by Circuit Breaker (OPEN state). Cooldown remaining: {remaining:.1f}s")
                raise CircuitBreakerOpenException("Circuit breaker is OPEN. Service is temporarily disabled.")

        # 2. Execute the async call
        try:
            result = await async_func(*args, **kwargs)
            
            # If the request succeeds:
            if self.state == "HALF_OPEN":
                # Service has successfully recovered! Reset the fuse.
                logger.info("Probe request succeeded! Resetting circuit breaker to CLOSED.")
                self._transition_to("CLOSED")
                self.failure_count = 0
            elif self.state == "CLOSED":
                # Keep resetting failures on normal successful runs
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            # If the request fails, record the failure!
            self.failure_count += 1
            logger.error(f"Execution failed. Failure count: {self.failure_count}/{self.failure_threshold}. Error: {e}")
            
            if self.state in ("CLOSED", "HALF_OPEN") and self.failure_count >= self.failure_threshold:
                # Consecutive failures crossed threshold: trip the fuse!
                logger.error(f"Failure threshold of {self.failure_threshold} reached. Tripping fuse to OPEN.")
                self._transition_to("OPEN")
                
            raise e
