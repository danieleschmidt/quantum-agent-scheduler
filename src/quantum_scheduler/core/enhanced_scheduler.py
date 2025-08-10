"""Enhanced quantum scheduler with advanced reliability and fault tolerance."""

import time
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from collections import defaultdict
from datetime import datetime

from .models import Agent, Task, Solution, SchedulingProblem
from .exceptions import (
    ValidationError,
    BackendError,
    SolverError,
    SolverTimeoutError,
    InfeasibleProblemError,
    SkillMismatchError,
    CapacityExceededError
)
from .validators import InputValidator
from .scheduler import QuantumScheduler
from ..backends import Backend, ClassicalBackend, HybridBackend, SimulatedQuantumBackend
from ..constraints import Constraint
from ..monitoring import get_metrics_collector
from ..security import SecuritySanitizer
from ..optimization import get_solution_cache, ProblemOptimizer
from ..reliability import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    RetryPolicy,
    RetryPolicyBuilder,
    BackoffStrategy
)

logger = logging.getLogger(__name__)


class EnhancedQuantumScheduler(QuantumScheduler):
    """Enhanced quantum scheduler with circuit breakers, retry policies, and advanced fault tolerance."""
    
    def __init__(
        self, 
        backend: Union[str, Backend] = "auto",
        fallback: str = "classical",
        optimization_target: str = "minimize_cost",
        timeout: Optional[float] = None,
        enable_validation: bool = True,
        enable_metrics: bool = True,
        enable_caching: bool = True,
        enable_optimization: bool = True,
        enable_circuit_breaker: bool = True,
        enable_retry_policy: bool = True,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        retry_policy_config: Optional[Dict[str, Any]] = None,
        health_check_interval: float = 60.0,
        auto_recovery: bool = True
    ):
        """Initialize the enhanced quantum scheduler.
        
        Args:
            backend: Backend type or instance
            fallback: Fallback solver type
            optimization_target: Optimization objective
            timeout: Maximum solve time in seconds
            enable_validation: Enable input validation
            enable_metrics: Enable metrics collection
            enable_caching: Enable solution caching
            enable_optimization: Enable problem optimization
            enable_circuit_breaker: Enable circuit breaker protection
            enable_retry_policy: Enable retry policies
            circuit_breaker_config: Circuit breaker configuration
            retry_policy_config: Retry policy configuration
            health_check_interval: Health check interval in seconds
            auto_recovery: Enable automatic recovery from failures
        """
        # Initialize parent scheduler
        super().__init__(
            backend=backend,
            fallback=fallback,
            optimization_target=optimization_target,
            timeout=timeout,
            enable_validation=enable_validation,
            enable_metrics=enable_metrics,
            enable_caching=enable_caching,
            enable_optimization=enable_optimization
        )
        
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_retry_policy = enable_retry_policy
        self.health_check_interval = health_check_interval
        self.auto_recovery = auto_recovery
        
        # Initialize reliability components
        self._circuit_breaker_registry = get_circuit_breaker_registry()
        self._initialize_circuit_breakers(circuit_breaker_config or {})
        self._initialize_retry_policies(retry_policy_config or {})
        
        # Health monitoring
        self._last_health_check = time.time()
        self._health_status = {'overall': 'healthy', 'components': {}}
        self._failure_history = []
        
        # Performance tracking
        self._performance_metrics = {
            'avg_solve_time': 0.0,
            'success_rate': 1.0,
            'circuit_breaker_trips': 0,
            'retry_attempts': 0,
            'fallback_usage': 0,
            'health_check_failures': 0
        }
        
        logger.info(f"Enhanced QuantumScheduler initialized with reliability features enabled")
    
    def _initialize_circuit_breakers(self, config: Dict[str, Any]):
        """Initialize circuit breakers for various components."""
        if not self.enable_circuit_breaker:
            return
        
        # Default circuit breaker configurations
        default_config = {
            'failure_threshold': config.get('failure_threshold', 5),
            'recovery_timeout': config.get('recovery_timeout', 60.0),
            'window_size': config.get('window_size', 100),
            'slow_call_threshold': config.get('slow_call_threshold', 30.0)
        }
        
        # Primary backend circuit breaker
        self._primary_circuit_breaker = self._circuit_breaker_registry.create(
            name=f"scheduler_{self._backend_type}",
            expected_exception=(BackendError, SolverError, SolverTimeoutError),
            **default_config
        )
        
        # Fallback backend circuit breaker
        if self._fallback:
            self._fallback_circuit_breaker = self._circuit_breaker_registry.create(
                name=f"scheduler_{self._fallback}_fallback",
                expected_exception=(BackendError, SolverError),
                failure_threshold=default_config['failure_threshold'] * 2,  # More lenient for fallback
                **{k: v for k, v in default_config.items() if k != 'failure_threshold'}
            )
        
        # Cache circuit breaker
        if self._enable_caching:
            self._cache_circuit_breaker = self._circuit_breaker_registry.create(
                name="scheduler_cache",
                expected_exception=Exception,
                failure_threshold=10,
                recovery_timeout=30.0,
                window_size=50
            )
    
    def _initialize_retry_policies(self, config: Dict[str, Any]):
        """Initialize retry policies for different operations."""
        if not self.enable_retry_policy:
            return
        
        # Primary backend retry policy
        self._primary_retry_policy = RetryPolicyBuilder() \
            .max_attempts(config.get('max_attempts', 3)) \
            .base_delay(config.get('base_delay', 1.0)) \
            .jittered_backoff(multiplier=2.0, jitter_factor=0.1) \
            .retry_on(BackendError, SolverError) \
            .do_not_retry_on(ValidationError, SkillMismatchError) \
            .timeout(config.get('retry_timeout', 300.0)) \
            .on_retry(self._on_retry_callback) \
            .on_give_up(self._on_give_up_callback) \
            .build()
        
        # Fallback retry policy (more aggressive)
        self._fallback_retry_policy = RetryPolicyBuilder() \
            .max_attempts(config.get('fallback_max_attempts', 5)) \
            .base_delay(0.5) \
            .exponential_backoff(multiplier=1.5) \
            .retry_on(BackendError, SolverError) \
            .timeout(config.get('fallback_retry_timeout', 180.0)) \
            .build()
    
    def _on_retry_callback(self, context):
        """Callback for retry attempts."""
        self._performance_metrics['retry_attempts'] += 1
        logger.info(f"Retrying {context.func_name} (attempt {context.attempt_number}): {context.last_exception}")
        
        # Trigger health check if too many retries
        if context.attempt_number >= 3:
            self._trigger_health_check()
    
    def _on_give_up_callback(self, context):
        """Callback when giving up retries."""
        logger.error(f"Giving up on {context.func_name} after {context.attempt_number} attempts: {context.last_exception}")
        self._record_failure('retry_exhausted', context.last_exception)
    
    def schedule(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """Schedule tasks with enhanced reliability features."""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            # Perform health check if needed
            self._periodic_health_check()
            
            # Use circuit breaker for primary backend
            if self.enable_circuit_breaker and self._primary_circuit_breaker:
                return self._primary_circuit_breaker.call(
                    self._schedule_with_retry,
                    agents, tasks, constraints, operation_id
                )
            else:
                return self._schedule_with_retry(agents, tasks, constraints, operation_id)
                
        except Exception as e:
            self._record_failure('schedule_error', e)
            raise
        finally:
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time)
    
    def _schedule_with_retry(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Optional[Dict[str, Any]],
        operation_id: str
    ) -> Solution:
        """Schedule with retry policy applied."""
        if self.enable_retry_policy and hasattr(self, '_primary_retry_policy'):
            # Use the retry policy as a decorator
            @self._primary_retry_policy
            def schedule_func():
                return super(EnhancedQuantumScheduler, self).schedule(agents, tasks, constraints)
            return schedule_func()
        else:
            return super().schedule(agents, tasks, constraints)
    
    def _periodic_health_check(self):
        """Perform periodic health checks."""
        current_time = time.time()
        if current_time - self._last_health_check >= self.health_check_interval:
            self._trigger_health_check()
            self._last_health_check = current_time
    
    def _trigger_health_check(self):
        """Trigger a comprehensive health check."""
        try:
            health_status = self.health_check()
            self._health_status = health_status
            
            if health_status['overall'] != 'healthy':
                logger.warning(f"Health check failed: {health_status}")
                self._performance_metrics['health_check_failures'] += 1
                
                if self.auto_recovery:
                    self._attempt_auto_recovery(health_status)
                    
        except Exception as e:
            logger.error(f"Health check failed with exception: {e}")
            self._record_failure('health_check_error', e)
    
    def _attempt_auto_recovery(self, health_status: Dict[str, Any]):
        """Attempt automatic recovery from health issues."""
        logger.info("Attempting automatic recovery...")
        
        # Reset circuit breakers if they're open
        if self.enable_circuit_breaker:
            for breaker in self._circuit_breaker_registry._breakers.values():
                if breaker.is_open:
                    logger.info(f"Resetting circuit breaker: {breaker.name}")
                    breaker.reset()
        
        # Clear caches if cache is unhealthy
        if 'cache' in health_status.get('components', {}) and self._solution_cache:
            try:
                self._solution_cache.clear()
                logger.info("Cleared solution cache for recovery")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
        
        # Re-initialize backends if needed
        if health_status['overall'] == 'critical':
            try:
                self._backend = self._initialize_backend(self._backend_type)
                logger.info(f"Re-initialized backend: {self._backend_type}")
            except Exception as e:
                logger.error(f"Failed to re-initialize backend: {e}")
    
    def _record_failure(self, failure_type: str, exception: Exception):
        """Record failure in history for analysis."""
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'type': failure_type,
            'exception': str(exception),
            'exception_type': type(exception).__name__
        }
        
        self._failure_history.append(failure_record)
        
        # Keep only last 100 failures
        if len(self._failure_history) > 100:
            self._failure_history = self._failure_history[-100:]
    
    def _update_performance_metrics(self, execution_time: float):
        """Update performance metrics."""
        # Update average solve time with exponential smoothing
        alpha = 0.1  # Smoothing factor
        self._performance_metrics['avg_solve_time'] = \
            alpha * execution_time + (1 - alpha) * self._performance_metrics['avg_solve_time']
        
        # Update success rate (assuming success if we reach this point)
        if self._metrics_collector:
            system_metrics = self._metrics_collector.get_system_metrics()
            total_calls = system_metrics.total_operations or 1
            successful_calls = system_metrics.successful_operations or 1
            self._performance_metrics['success_rate'] = successful_calls / total_calls
        else:
            self._performance_metrics['success_rate'] = 1.0
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components."""
        health_status = {'overall': 'healthy', 'components': {}, 'timestamp': datetime.now().isoformat()}
        issues = []
        
        try:
            # Check backend health
            backend_health = self._check_backend_health()
            health_status['components']['backend'] = backend_health
            if backend_health['status'] != 'healthy':
                issues.append('backend')
            
            # Check circuit breakers
            if self.enable_circuit_breaker:
                cb_health = self._check_circuit_breakers_health()
                health_status['components']['circuit_breakers'] = cb_health
                if cb_health['status'] != 'healthy':
                    issues.append('circuit_breakers')
            
            # Check cache health
            if self._solution_cache:
                cache_health = self._check_cache_health()
                health_status['components']['cache'] = cache_health
                if cache_health['status'] != 'healthy':
                    issues.append('cache')
            
            # Check metrics collector
            if self._metrics_collector:
                metrics_health = self._check_metrics_health()
                health_status['components']['metrics'] = metrics_health
                if metrics_health['status'] != 'healthy':
                    issues.append('metrics')
            
            # Determine overall health
            if not issues:
                health_status['overall'] = 'healthy'
            elif len(issues) <= 1:
                health_status['overall'] = 'degraded'
            else:
                health_status['overall'] = 'critical'
            
            health_status['issues'] = issues
            health_status['performance'] = self._performance_metrics.copy()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall'] = 'critical'
            health_status['error'] = str(e)
        
        return health_status
    
    def _check_backend_health(self) -> Dict[str, Any]:
        """Check backend health."""
        try:
            # Simple test with minimal problem
            test_agents = [Agent(id='test_agent', skills=['test'], capacity=1)]
            test_tasks = [Task(id='test_task', required_skills=['test'], duration=1, priority=1)]
            test_problem = SchedulingProblem(
                agents=test_agents,
                tasks=test_tasks,
                constraints={},
                optimization_target='minimize_cost'
            )
            
            start_time = time.time()
            solution = self._backend.solve(test_problem)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'backend_type': self._backend_type,
                'message': 'Backend responding normally'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend_type': self._backend_type,
                'message': f'Backend health check failed: {e}'
            }
    
    def _check_circuit_breakers_health(self) -> Dict[str, Any]:
        """Check circuit breakers health."""
        try:
            cb_metrics = self._circuit_breaker_registry.get_all_metrics()
            open_breakers = [name for name, metrics in cb_metrics.items() 
                           if metrics['state'] == 'open']
            
            if not open_breakers:
                status = 'healthy'
                message = 'All circuit breakers closed'
            elif len(open_breakers) == 1:
                status = 'degraded'
                message = f'Circuit breaker open: {open_breakers[0]}'
            else:
                status = 'unhealthy'
                message = f'Multiple circuit breakers open: {open_breakers}'
            
            return {
                'status': status,
                'open_breakers': open_breakers,
                'total_breakers': len(cb_metrics),
                'message': message,
                'metrics': cb_metrics
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': f'Circuit breaker health check failed: {e}'
            }
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            # Test cache operations
            test_key = f'health_check_{uuid.uuid4()}'
            test_value = {'test': 'data'}
            
            # Test put and get
            self._solution_cache.put(test_key, test_value)
            retrieved = self._solution_cache.get(test_key)
            
            if retrieved == test_value:
                return {
                    'status': 'healthy',
                    'message': 'Cache operations successful'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': 'Cache data integrity issue'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': f'Cache health check failed: {e}'
            }
    
    def _check_metrics_health(self) -> Dict[str, Any]:
        """Check metrics collector health."""
        try:
            # Test metrics collection
            test_operation_id = str(uuid.uuid4())
            self._metrics_collector.start_operation(test_operation_id, 1, 1)
            self._metrics_collector.end_operation(test_operation_id, True, 'test', 1, 1.0, 1.0)
            
            return {
                'status': 'healthy',
                'message': 'Metrics collection working'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': f'Metrics health check failed: {e}'
            }
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive reliability metrics."""
        metrics = {
            'performance': self._performance_metrics.copy(),
            'health_status': self._health_status,
            'failure_history_count': len(self._failure_history),
            'last_health_check': self._last_health_check,
            'auto_recovery_enabled': self.auto_recovery
        }
        
        # Add circuit breaker metrics
        if self.enable_circuit_breaker:
            metrics['circuit_breakers'] = self._circuit_breaker_registry.get_all_metrics()
        
        # Add retry policy metrics
        if self.enable_retry_policy:
            retry_metrics = {}
            if hasattr(self, '_primary_retry_policy'):
                retry_metrics['primary'] = self._primary_retry_policy.get_metrics()
            if hasattr(self, '_fallback_retry_policy'):
                retry_metrics['fallback'] = self._fallback_retry_policy.get_metrics()
            metrics['retry_policies'] = retry_metrics
        
        return metrics
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get failure analysis report."""
        if not self._failure_history:
            return {'message': 'No failures recorded'}
        
        # Analyze failure patterns
        failure_types = defaultdict(int)
        exception_types = defaultdict(int)
        recent_failures = []
        
        for failure in self._failure_history:
            failure_types[failure['type']] += 1
            exception_types[failure['exception_type']] += 1
            
            # Recent failures (last 24 hours)
            failure_time = datetime.fromisoformat(failure['timestamp'])
            if (datetime.now() - failure_time).total_seconds() < 24 * 3600:
                recent_failures.append(failure)
        
        return {
            'total_failures': len(self._failure_history),
            'recent_failures': len(recent_failures),
            'failure_types': dict(failure_types),
            'exception_types': dict(exception_types),
            'most_common_failure': max(failure_types.items(), key=lambda x: x[1]) if failure_types else None,
            'most_common_exception': max(exception_types.items(), key=lambda x: x[1]) if exception_types else None,
            'recent_failure_details': recent_failures[-5:]  # Last 5 recent failures
        }
    
    async def schedule_async(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """Asynchronous version of schedule method."""
        # Run synchronous schedule in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.schedule, agents, tasks, constraints)
    
    def reset_reliability_state(self):
        """Reset all reliability state (useful for testing)."""
        if self.enable_circuit_breaker:
            self._circuit_breaker_registry.reset_all()
        
        if self.enable_retry_policy:
            if hasattr(self, '_primary_retry_policy'):
                self._primary_retry_policy.reset_metrics()
            if hasattr(self, '_fallback_retry_policy'):
                self._fallback_retry_policy.reset_metrics()
        
        self._failure_history.clear()
        self._performance_metrics = {
            'avg_solve_time': 0.0,
            'success_rate': 1.0,
            'circuit_breaker_trips': 0,
            'retry_attempts': 0,
            'fallback_usage': 0,
            'health_check_failures': 0
        }
        
        logger.info("Reliability state reset")
    
    def force_failover_test(self):
        """Force failover test (opens all circuit breakers)."""
        if self.enable_circuit_breaker:
            self._circuit_breaker_registry.force_open_all()
            logger.warning("Forced failover test - all circuit breakers opened")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Perform cleanup and final health check
        final_health = self.health_check()
        logger.info(f"Scheduler context exit - final health: {final_health['overall']}")
        return False