"""Advanced retry policies with backoff strategies for quantum scheduler."""

import asyncio
import logging
import random
import time
import functools
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Available backoff strategies for retries."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    FIBONACCI = "fibonacci"
    JITTERED_EXPONENTIAL = "jittered_exponential"


class RetryOutcome(Enum):
    """Possible outcomes of a retry attempt."""
    SUCCESS = "success"
    RETRY = "retry"
    GIVE_UP = "give_up"


class RetryContext:
    """Context information for retry attempts."""
    
    def __init__(self, func_name: str, args: tuple, kwargs: dict):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.attempt_number = 0
        self.total_elapsed_time = 0.0
        self.last_exception = None
        self.start_time = time.time()
        self.exceptions_history: List[Exception] = []
        self.retry_delays: List[float] = []
        
    def record_attempt(self, exception: Optional[Exception] = None, delay: float = 0.0):
        """Record a retry attempt."""
        self.attempt_number += 1
        self.total_elapsed_time = time.time() - self.start_time
        
        if exception:
            self.last_exception = exception
            self.exceptions_history.append(exception)
            
        if delay > 0:
            self.retry_delays.append(delay)


class BackoffCalculator(ABC):
    """Abstract base class for backoff calculation strategies."""
    
    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay for the given attempt number."""
        pass


class FixedBackoff(BackoffCalculator):
    """Fixed delay between retries."""
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return base_delay


class LinearBackoff(BackoffCalculator):
    """Linear increase in delay between retries."""
    
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return base_delay * attempt * self.multiplier


class ExponentialBackoff(BackoffCalculator):
    """Exponential increase in delay between retries."""
    
    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return base_delay * (self.multiplier ** attempt)


class PolynomialBackoff(BackoffCalculator):
    """Polynomial increase in delay between retries."""
    
    def __init__(self, degree: float = 2.0):
        self.degree = degree
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return base_delay * (attempt ** self.degree)


class FibonacciBackoff(BackoffCalculator):
    """Fibonacci sequence-based delay between retries."""
    
    def __init__(self):
        self._fib_cache = {0: 1, 1: 1}
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number with caching."""
        if n in self._fib_cache:
            return self._fib_cache[n]
        
        self._fib_cache[n] = self._fibonacci(n-1) + self._fibonacci(n-2)
        return self._fib_cache[n]
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return base_delay * self._fibonacci(attempt)


class JitteredExponentialBackoff(BackoffCalculator):
    """Exponential backoff with jitter to avoid thundering herd."""
    
    def __init__(self, multiplier: float = 2.0, jitter_factor: float = 0.1):
        self.multiplier = multiplier
        self.jitter_factor = jitter_factor
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        base_exponential = base_delay * (self.multiplier ** attempt)
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * base_exponential
        return max(0.1, base_exponential + jitter)  # Minimum 100ms delay


class RetryPolicy:
    """Advanced retry policy with configurable strategies."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        backoff_strategy: Union[BackoffStrategy, BackoffCalculator] = BackoffStrategy.EXPONENTIAL,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
        timeout: Optional[float] = None,
        retry_condition: Optional[Callable[[Exception], bool]] = None,
        on_retry: Optional[Callable[[RetryContext], None]] = None,
        on_give_up: Optional[Callable[[RetryContext], None]] = None,
        circuit_breaker_name: Optional[str] = None
    ):
        """Initialize retry policy.
        
        Args:
            max_attempts: Maximum number of attempts (including initial)
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            backoff_strategy: Strategy for calculating retry delays
            retryable_exceptions: List of exception types that trigger retries
            non_retryable_exceptions: List of exception types that don't trigger retries
            timeout: Maximum total time for all retry attempts
            retry_condition: Custom function to determine if exception should be retried
            on_retry: Callback called before each retry attempt
            on_give_up: Callback called when giving up
            circuit_breaker_name: Name of circuit breaker to use
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.retry_condition = retry_condition
        self.on_retry = on_retry
        self.on_give_up = on_give_up
        self.circuit_breaker_name = circuit_breaker_name
        
        # Configure backoff calculator
        if isinstance(backoff_strategy, BackoffCalculator):
            self.backoff_calculator = backoff_strategy
        else:
            self.backoff_calculator = self._create_backoff_calculator(backoff_strategy)
        
        # Configure retryable exceptions
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.non_retryable_exceptions = non_retryable_exceptions or []
        
        # Metrics
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retried_calls': 0,
            'total_attempts': 0,
            'total_retry_time': 0.0
        }
    
    def _create_backoff_calculator(self, strategy: BackoffStrategy) -> BackoffCalculator:
        """Create backoff calculator based on strategy."""
        if strategy == BackoffStrategy.FIXED:
            return FixedBackoff()
        elif strategy == BackoffStrategy.LINEAR:
            return LinearBackoff()
        elif strategy == BackoffStrategy.EXPONENTIAL:
            return ExponentialBackoff()
        elif strategy == BackoffStrategy.POLYNOMIAL:
            return PolynomialBackoff()
        elif strategy == BackoffStrategy.FIBONACCI:
            return FibonacciBackoff()
        elif strategy == BackoffStrategy.JITTERED_EXPONENTIAL:
            return JitteredExponentialBackoff()
        else:
            raise ValueError(f"Unknown backoff strategy: {strategy}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry policy to a function."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
    
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Wrapper for synchronous functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, args, kwargs)
        return wrapper
    
    def _async_wrapper(self, func: Callable) -> Callable:
        """Wrapper for asynchronous functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._execute_with_retry_async(func, args, kwargs)
        return wrapper
    
    def _execute_with_retry(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with retry logic (synchronous)."""
        context = RetryContext(func.__name__, args, kwargs)
        self.metrics['total_calls'] += 1
        
        while context.attempt_number < self.max_attempts:
            try:
                # Check timeout
                if self.timeout and context.total_elapsed_time >= self.timeout:
                    logger.warning(f"Retry timeout exceeded for {context.func_name}")
                    break
                
                context.record_attempt()
                self.metrics['total_attempts'] += 1
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Success - update metrics and return
                self.metrics['successful_calls'] += 1
                if context.attempt_number > 1:
                    self.metrics['retried_calls'] += 1
                    self.metrics['total_retry_time'] += context.total_elapsed_time
                
                return result
                
            except Exception as e:
                context.record_attempt(e)
                
                # Check if we should retry this exception
                if not self._should_retry(e, context):
                    break
                
                # Check if we have more attempts
                if context.attempt_number >= self.max_attempts:
                    break
                
                # Calculate and apply delay
                delay = self._calculate_delay(context.attempt_number)
                context.record_attempt(delay=delay)
                
                logger.info(f"Retrying {context.func_name} after {delay:.2f}s "
                           f"(attempt {context.attempt_number + 1}/{self.max_attempts}): {e}")
                
                if self.on_retry:
                    try:
                        self.on_retry(context)
                    except Exception as callback_error:
                        logger.warning(f"Retry callback failed: {callback_error}")
                
                time.sleep(delay)
        
        # Give up - call callback and raise last exception
        self.metrics['failed_calls'] += 1
        
        if self.on_give_up:
            try:
                self.on_give_up(context)
            except Exception as callback_error:
                logger.warning(f"Give up callback failed: {callback_error}")
        
        logger.error(f"Gave up retrying {context.func_name} after "
                    f"{context.attempt_number} attempts in {context.total_elapsed_time:.2f}s")
        
        if context.last_exception:
            raise context.last_exception
        else:
            raise RuntimeError(f"Function {context.func_name} failed without exception")
    
    async def _execute_with_retry_async(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with retry logic (asynchronous)."""
        context = RetryContext(func.__name__, args, kwargs)
        self.metrics['total_calls'] += 1
        
        while context.attempt_number < self.max_attempts:
            try:
                # Check timeout
                if self.timeout and context.total_elapsed_time >= self.timeout:
                    logger.warning(f"Retry timeout exceeded for {context.func_name}")
                    break
                
                context.record_attempt()
                self.metrics['total_attempts'] += 1
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Success - update metrics and return
                self.metrics['successful_calls'] += 1
                if context.attempt_number > 1:
                    self.metrics['retried_calls'] += 1
                    self.metrics['total_retry_time'] += context.total_elapsed_time
                
                return result
                
            except Exception as e:
                context.record_attempt(e)
                
                # Check if we should retry this exception
                if not self._should_retry(e, context):
                    break
                
                # Check if we have more attempts
                if context.attempt_number >= self.max_attempts:
                    break
                
                # Calculate and apply delay
                delay = self._calculate_delay(context.attempt_number)
                context.record_attempt(delay=delay)
                
                logger.info(f"Retrying {context.func_name} after {delay:.2f}s "
                           f"(attempt {context.attempt_number + 1}/{self.max_attempts}): {e}")
                
                if self.on_retry:
                    try:
                        self.on_retry(context)
                    except Exception as callback_error:
                        logger.warning(f"Retry callback failed: {callback_error}")
                
                await asyncio.sleep(delay)
        
        # Give up - call callback and raise last exception
        self.metrics['failed_calls'] += 1
        
        if self.on_give_up:
            try:
                self.on_give_up(context)
            except Exception as callback_error:
                logger.warning(f"Give up callback failed: {callback_error}")
        
        logger.error(f"Gave up retrying {context.func_name} after "
                    f"{context.attempt_number} attempts in {context.total_elapsed_time:.2f}s")
        
        if context.last_exception:
            raise context.last_exception
        else:
            raise RuntimeError(f"Function {context.func_name} failed without exception")
    
    def _should_retry(self, exception: Exception, context: RetryContext) -> bool:
        """Determine if exception should trigger a retry."""
        # Check non-retryable exceptions first
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                logger.debug(f"Not retrying {context.func_name} due to non-retryable exception: {exc_type.__name__}")
                return False
        
        # Check custom retry condition
        if self.retry_condition:
            try:
                return self.retry_condition(exception)
            except Exception as e:
                logger.warning(f"Retry condition function failed: {e}")
                return False
        
        # Check retryable exceptions
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        delay = self.backoff_calculator.calculate_delay(attempt, self.base_delay)
        return min(delay, self.max_delay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry policy metrics."""
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_calls'] > 0:
            metrics['success_rate'] = metrics['successful_calls'] / metrics['total_calls']
            metrics['retry_rate'] = metrics['retried_calls'] / metrics['total_calls']
            metrics['avg_attempts'] = metrics['total_attempts'] / metrics['total_calls']
        else:
            metrics['success_rate'] = 0.0
            metrics['retry_rate'] = 0.0
            metrics['avg_attempts'] = 0.0
        
        if metrics['retried_calls'] > 0:
            metrics['avg_retry_time'] = metrics['total_retry_time'] / metrics['retried_calls']
        else:
            metrics['avg_retry_time'] = 0.0
        
        # Add configuration info
        metrics.update({
            'max_attempts': self.max_attempts,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'backoff_strategy': self.backoff_calculator.__class__.__name__,
            'timeout': self.timeout
        })
        
        return metrics
    
    def reset_metrics(self):
        """Reset retry policy metrics."""
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retried_calls': 0,
            'total_attempts': 0,
            'total_retry_time': 0.0
        }


class RetryPolicyBuilder:
    """Builder for constructing retry policies."""
    
    def __init__(self):
        self._max_attempts = 3
        self._base_delay = 1.0
        self._max_delay = 300.0
        self._backoff_strategy = BackoffStrategy.EXPONENTIAL
        self._retryable_exceptions = None
        self._non_retryable_exceptions = None
        self._timeout = None
        self._retry_condition = None
        self._on_retry = None
        self._on_give_up = None
        self._circuit_breaker_name = None
    
    def max_attempts(self, attempts: int) -> 'RetryPolicyBuilder':
        """Set maximum attempts."""
        self._max_attempts = attempts
        return self
    
    def base_delay(self, delay: float) -> 'RetryPolicyBuilder':
        """Set base delay."""
        self._base_delay = delay
        return self
    
    def max_delay(self, delay: float) -> 'RetryPolicyBuilder':
        """Set maximum delay."""
        self._max_delay = delay
        return self
    
    def exponential_backoff(self, multiplier: float = 2.0) -> 'RetryPolicyBuilder':
        """Use exponential backoff strategy."""
        self._backoff_strategy = ExponentialBackoff(multiplier)
        return self
    
    def linear_backoff(self, multiplier: float = 1.0) -> 'RetryPolicyBuilder':
        """Use linear backoff strategy."""
        self._backoff_strategy = LinearBackoff(multiplier)
        return self
    
    def fixed_backoff(self) -> 'RetryPolicyBuilder':
        """Use fixed backoff strategy."""
        self._backoff_strategy = FixedBackoff()
        return self
    
    def jittered_backoff(self, multiplier: float = 2.0, jitter_factor: float = 0.1) -> 'RetryPolicyBuilder':
        """Use jittered exponential backoff strategy."""
        self._backoff_strategy = JitteredExponentialBackoff(multiplier, jitter_factor)
        return self
    
    def retry_on(self, *exceptions: Type[Exception]) -> 'RetryPolicyBuilder':
        """Set retryable exceptions."""
        self._retryable_exceptions = list(exceptions)
        return self
    
    def do_not_retry_on(self, *exceptions: Type[Exception]) -> 'RetryPolicyBuilder':
        """Set non-retryable exceptions."""
        self._non_retryable_exceptions = list(exceptions)
        return self
    
    def timeout(self, seconds: float) -> 'RetryPolicyBuilder':
        """Set total timeout."""
        self._timeout = seconds
        return self
    
    def retry_if(self, condition: Callable[[Exception], bool]) -> 'RetryPolicyBuilder':
        """Set custom retry condition."""
        self._retry_condition = condition
        return self
    
    def on_retry(self, callback: Callable[[RetryContext], None]) -> 'RetryPolicyBuilder':
        """Set retry callback."""
        self._on_retry = callback
        return self
    
    def on_give_up(self, callback: Callable[[RetryContext], None]) -> 'RetryPolicyBuilder':
        """Set give up callback."""
        self._on_give_up = callback
        return self
    
    def with_circuit_breaker(self, name: str) -> 'RetryPolicyBuilder':
        """Use circuit breaker."""
        self._circuit_breaker_name = name
        return self
    
    def build(self) -> RetryPolicy:
        """Build the retry policy."""
        return RetryPolicy(
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            backoff_strategy=self._backoff_strategy,
            retryable_exceptions=self._retryable_exceptions,
            non_retryable_exceptions=self._non_retryable_exceptions,
            timeout=self._timeout,
            retry_condition=self._retry_condition,
            on_retry=self._on_retry,
            on_give_up=self._on_give_up,
            circuit_breaker_name=self._circuit_breaker_name
        )


# Convenience functions
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    **kwargs
) -> Callable:
    """Simple retry decorator.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries
        backoff_strategy: Backoff strategy to use
        **kwargs: Additional retry policy arguments
        
    Returns:
        Retry decorator
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy,
        **kwargs
    )
    return policy


def retry_on_exception(
    *exceptions: Type[Exception],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs
) -> Callable:
    """Retry only on specific exceptions.
    
    Args:
        *exceptions: Exception types to retry on
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries
        **kwargs: Additional retry policy arguments
        
    Returns:
        Retry decorator
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=list(exceptions),
        **kwargs
    )
    return policy