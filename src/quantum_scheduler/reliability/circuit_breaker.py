"""Advanced circuit breaker implementation for quantum scheduler fault tolerance."""

import time
import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union
from collections import deque
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking requests due to failures
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Advanced circuit breaker with sliding window failure detection."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: str = "unnamed",
        window_size: int = 100,
        half_open_max_calls: int = 3,
        slow_call_threshold: float = 5.0,
        slow_call_rate_threshold: float = 0.5
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger open state
            recovery_timeout: Time to wait before transitioning to half-open
            expected_exception: Exception types to count as failures
            name: Circuit breaker name for logging
            window_size: Size of sliding window for failure tracking
            half_open_max_calls: Max calls allowed in half-open state
            slow_call_threshold: Time threshold for slow calls (seconds)
            slow_call_rate_threshold: Ratio of slow calls to trigger open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.window_size = window_size
        self.half_open_max_calls = half_open_max_calls
        self.slow_call_threshold = slow_call_threshold
        self.slow_call_rate_threshold = slow_call_rate_threshold
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
        
        # Sliding window tracking
        self._call_history = deque(maxlen=window_size)  # (success, duration, timestamp)
        self._metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'slow_calls': 0,
            'state_changes': 0,
            'last_failure': None,
            'last_success': None
        }
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        wrapper.__name__ = f"circuit_breaker({func.__name__})"
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            # Check if we should block the call
            if self._should_block_call():
                self._metrics['total_calls'] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is {self._state.value}. "
                    f"Last failure: {self._metrics.get('last_failure', 'N/A')}"
                )
            
            # Execute the call
            start_time = time.time()
            success = False
            duration = 0.0
            
            try:
                result = func(*args, **kwargs)
                success = True
                duration = time.time() - start_time
                self._on_success(duration)
                return result
                
            except self.expected_exception as e:
                duration = time.time() - start_time
                self._on_failure(e, duration)
                raise
                
            except Exception as e:
                # Unexpected exception - don't count as failure for circuit breaker
                duration = time.time() - start_time
                logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
                self._record_call(True, duration)  # Count as success for circuit breaker
                raise
            
            finally:
                self._metrics['total_calls'] += 1
    
    def _should_block_call(self) -> bool:
        """Determine if call should be blocked based on current state."""
        current_time = time.time()
        
        if self._state == CircuitState.CLOSED:
            return False
            
        elif self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self._last_failure_time and 
                current_time - self._last_failure_time >= self.recovery_timeout):
                self._transition_to_half_open()
                return False
            return True
            
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return False
            return True
        
        return False
    
    def _on_success(self, duration: float):
        """Handle successful call."""
        self._record_call(True, duration)
        self._metrics['successful_calls'] += 1
        self._metrics['last_success'] = datetime.now().isoformat()
        
        if duration > self.slow_call_threshold:
            self._metrics['slow_calls'] += 1
        
        if self._state == CircuitState.HALF_OPEN:
            # Reset failure count on success in half-open
            self._failure_count = 0
            
            # If we've had enough successful calls, close the circuit
            if self._half_open_calls >= self.half_open_max_calls:
                self._transition_to_closed()
    
    def _on_failure(self, exception: Exception, duration: float):
        """Handle failed call."""
        self._record_call(False, duration)
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._metrics['failed_calls'] += 1
        self._metrics['last_failure'] = datetime.now().isoformat()
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure: {exception}")
        
        # Check if we should open the circuit
        if self._should_open_circuit():
            self._transition_to_open()
    
    def _record_call(self, success: bool, duration: float):
        """Record call in sliding window."""
        self._call_history.append((success, duration, time.time()))
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure patterns."""
        # Check simple failure count threshold
        if self._failure_count >= self.failure_threshold:
            return True
        
        # Check sliding window patterns
        if len(self._call_history) >= self.window_size:
            recent_calls = list(self._call_history)
            
            # Calculate failure rate
            failures = sum(1 for success, _, _ in recent_calls if not success)
            failure_rate = failures / len(recent_calls)
            
            if failure_rate >= 0.5:  # 50% failure rate
                logger.info(f"Opening circuit '{self.name}' due to high failure rate: {failure_rate:.2%}")
                return True
            
            # Calculate slow call rate
            slow_calls = sum(1 for _, duration, _ in recent_calls if duration > self.slow_call_threshold)
            slow_call_rate = slow_calls / len(recent_calls)
            
            if slow_call_rate >= self.slow_call_rate_threshold:
                logger.info(f"Opening circuit '{self.name}' due to high slow call rate: {slow_call_rate:.2%}")
                return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        if self._state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.name}' opened due to failures")
            self._state = CircuitState.OPEN
            self._metrics['state_changes'] += 1
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._metrics['state_changes'] += 1
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._metrics['state_changes'] += 1
    
    def force_open(self):
        """Manually open the circuit breaker."""
        with self._lock:
            logger.warning(f"Circuit breaker '{self.name}' manually opened")
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            self._metrics['state_changes'] += 1
    
    def force_close(self):
        """Manually close the circuit breaker."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}' manually closed")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._metrics['state_changes'] += 1
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            self._call_history.clear()
            self._metrics = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'slow_calls': 0,
                'state_changes': 0,
                'last_failure': None,
                'last_success': None
            }
            logger.info(f"Circuit breaker '{self.name}' reset")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self._state == CircuitState.HALF_OPEN
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            metrics.update({
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
                'window_size': self.window_size,
                'calls_in_window': len(self._call_history)
            })
            
            # Calculate rates if we have call history
            if self._call_history:
                recent_calls = list(self._call_history)
                successes = sum(1 for success, _, _ in recent_calls if success)
                failures = len(recent_calls) - successes
                slow_calls = sum(1 for _, duration, _ in recent_calls 
                               if duration > self.slow_call_threshold)
                
                metrics.update({
                    'success_rate': successes / len(recent_calls) if recent_calls else 0,
                    'failure_rate': failures / len(recent_calls) if recent_calls else 0,
                    'slow_call_rate': slow_calls / len(recent_calls) if recent_calls else 0,
                    'avg_response_time': statistics.mean(duration for _, duration, _ in recent_calls) if recent_calls else 0
                })
            
            return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Get health check information."""
        metrics = self.get_metrics()
        
        # Determine health status
        if self._state == CircuitState.CLOSED:
            status = "healthy"
        elif self._state == CircuitState.HALF_OPEN:
            status = "recovering"
        else:
            status = "unhealthy"
        
        return {
            'status': status,
            'circuit_state': self._state.value,
            'uptime_ratio': metrics.get('success_rate', 0),
            'total_requests': metrics['total_calls'],
            'failed_requests': metrics['failed_calls'],
            'last_failure': metrics['last_failure'],
            'message': f"Circuit breaker '{self.name}' is {status}"
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, breaker: CircuitBreaker) -> CircuitBreaker:
        """Register a circuit breaker."""
        with self._lock:
            if name in self._breakers:
                logger.warning(f"Overriding existing circuit breaker: {name}")
            self._breakers[name] = breaker
            return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        with self._lock:
            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                name=name,
                **kwargs
            )
            self._breakers[name] = breaker
            return breaker
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered circuit breakers."""
        return {name: breaker.get_metrics() 
                for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def force_open_all(self):
        """Force open all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_open()
    
    def force_close_all(self):
        """Force close all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_close()
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Get health check for all circuit breakers."""
        return {name: breaker.health_check() 
                for name, breaker in self._breakers.items()}


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _registry


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
    **kwargs
) -> Callable:
    """Decorator to add circuit breaker to a function.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures to trigger open state
        recovery_timeout: Time to wait before transitioning to half-open
        expected_exception: Exception types to count as failures
        **kwargs: Additional circuit breaker configuration
        
    Returns:
        Decorated function with circuit breaker
    """
    def decorator(func: Callable) -> Callable:
        breaker = _registry.create(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            **kwargs
        )
        return breaker(func)
    
    return decorator