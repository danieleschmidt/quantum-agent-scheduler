"""Adaptive load balancer for quantum scheduler with intelligent backend selection."""

import time
import logging
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Available load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE_PERFORMANCE = "adaptive_performance"
    QUANTUM_ADVANTAGE = "quantum_advantage"


@dataclass
class BackendMetrics:
    """Metrics for a backend instance."""
    name: str
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: deque = None
    last_request_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    quantum_advantage_score: float = 0.0
    capacity_utilization: float = 0.0
    health_score: float = 1.0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = deque(maxlen=100)  # Keep last 100 response times
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def current_load(self) -> float:
        """Calculate current load score (0.0 to 1.0+)."""
        # Combine active connections, utilization, and response time
        connection_load = min(self.active_connections / 10.0, 1.0)  # Normalize to 10 max connections
        utilization_load = self.capacity_utilization
        response_load = min(self.avg_response_time / 30.0, 1.0)  # Normalize to 30s max response
        
        return (connection_load + utilization_load + response_load) / 3.0


class AdaptiveLoadBalancer:
    """Intelligent load balancer with adaptive backend selection."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_PERFORMANCE,
        health_check_interval: float = 30.0,
        metric_window_size: int = 100,
        rebalance_threshold: float = 0.3,
        quantum_preference_weight: float = 0.7,
        performance_weight: float = 0.3
    ):
        """Initialize adaptive load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Interval between health checks
            metric_window_size: Size of metrics history window
            rebalance_threshold: Threshold for triggering rebalancing
            quantum_preference_weight: Weight for quantum advantage in selection
            performance_weight: Weight for performance metrics in selection
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.metric_window_size = metric_window_size
        self.rebalance_threshold = rebalance_threshold
        self.quantum_preference_weight = quantum_preference_weight
        self.performance_weight = performance_weight
        
        # Backend registry and metrics
        self.backends: Dict[str, Any] = {}  # backend_name -> backend_instance
        self.backend_metrics: Dict[str, BackendMetrics] = {}
        self.backend_weights: Dict[str, float] = {}
        
        # Load balancing state
        self._current_backend_index = 0
        self._selection_history = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        # Health monitoring
        self._last_health_check = time.time()
        self._unhealthy_backends = set()
        
        # Performance tracking
        self._performance_history = deque(maxlen=metric_window_size)
        self._quantum_advantage_tracker = QuantumAdvantageTracker()
        
        logger.info(f"Initialized AdaptiveLoadBalancer with strategy: {strategy.value}")
    
    def register_backend(
        self,
        name: str,
        backend: Any,
        weight: float = 1.0,
        is_quantum: bool = False,
        max_capacity: int = 10
    ) -> None:
        """Register a backend with the load balancer.
        
        Args:
            name: Backend name/identifier
            backend: Backend instance
            weight: Relative weight for weighted strategies
            is_quantum: Whether this is a quantum backend
            max_capacity: Maximum concurrent requests capacity
        """
        with self._lock:
            self.backends[name] = backend
            self.backend_metrics[name] = BackendMetrics(name=name)
            self.backend_weights[name] = weight
            
            # Set quantum advantage score based on backend type
            if is_quantum:
                self.backend_metrics[name].quantum_advantage_score = 1.0
            
            logger.info(f"Registered backend '{name}' (quantum: {is_quantum}, weight: {weight})")
    
    def unregister_backend(self, name: str) -> None:
        """Unregister a backend."""
        with self._lock:
            if name in self.backends:
                del self.backends[name]
                del self.backend_metrics[name]
                del self.backend_weights[name]
                self._unhealthy_backends.discard(name)
                logger.info(f"Unregistered backend '{name}'")
    
    def select_backend(self, problem_context: Optional[Dict[str, Any]] = None) -> Tuple[str, Any]:
        """Select the best backend for the given context.
        
        Args:
            problem_context: Context about the problem to solve
            
        Returns:
            Tuple of (backend_name, backend_instance)
            
        Raises:
            RuntimeError: If no healthy backends available
        """
        with self._lock:
            # Perform health check if needed
            self._periodic_health_check()
            
            # Get healthy backends
            healthy_backends = [name for name in self.backends.keys() 
                             if name not in self._unhealthy_backends]
            
            if not healthy_backends:
                raise RuntimeError("No healthy backends available")
            
            # Select backend based on strategy
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected = self._round_robin_selection(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                selected = self._least_connections_selection(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                selected = self._weighted_round_robin_selection(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                selected = self._least_response_time_selection(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE_PERFORMANCE:
                selected = self._adaptive_performance_selection(healthy_backends, problem_context)
            elif self.strategy == LoadBalancingStrategy.QUANTUM_ADVANTAGE:
                selected = self._quantum_advantage_selection(healthy_backends, problem_context)
            else:
                selected = self._round_robin_selection(healthy_backends)
            
            # Record selection
            self._selection_history.append((selected, time.time()))
            self.backend_metrics[selected].active_connections += 1
            
            logger.debug(f"Selected backend '{selected}' using {self.strategy.value} strategy")
            return selected, self.backends[selected]
    
    def record_request_start(self, backend_name: str) -> str:
        """Record the start of a request.
        
        Args:
            backend_name: Name of the backend handling the request
            
        Returns:
            Request ID for tracking
        """
        request_id = f"{backend_name}_{int(time.time() * 1000000)}"
        
        with self._lock:
            if backend_name in self.backend_metrics:
                metrics = self.backend_metrics[backend_name]
                metrics.total_requests += 1
                metrics.last_request_time = time.time()
        
        return request_id
    
    def record_request_completion(
        self,
        backend_name: str,
        request_id: str,
        success: bool,
        response_time: float,
        quantum_advantage: Optional[float] = None
    ) -> None:
        """Record the completion of a request.
        
        Args:
            backend_name: Name of the backend that handled the request
            request_id: Request ID from record_request_start
            success: Whether the request was successful
            response_time: Response time in seconds
            quantum_advantage: Measured quantum advantage (if applicable)
        """
        with self._lock:
            if backend_name not in self.backend_metrics:
                return
            
            metrics = self.backend_metrics[backend_name]
            metrics.active_connections = max(0, metrics.active_connections - 1)
            metrics.response_times.append(response_time)
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                metrics.last_failure_time = time.time()
            
            # Update quantum advantage if provided
            if quantum_advantage is not None:
                self._quantum_advantage_tracker.record_measurement(
                    backend_name, quantum_advantage, response_time
                )
                metrics.quantum_advantage_score = self._quantum_advantage_tracker.get_advantage_score(backend_name)
            
            # Update health score
            self._update_health_score(backend_name)
            
            # Record performance data
            self._performance_history.append({
                'backend': backend_name,
                'timestamp': time.time(),
                'success': success,
                'response_time': response_time,
                'quantum_advantage': quantum_advantage
            })
    
    def _round_robin_selection(self, backends: List[str]) -> str:
        """Round-robin backend selection."""
        selected = backends[self._current_backend_index % len(backends)]
        self._current_backend_index += 1
        return selected
    
    def _least_connections_selection(self, backends: List[str]) -> str:
        """Select backend with least active connections."""
        return min(backends, key=lambda b: self.backend_metrics[b].active_connections)
    
    def _weighted_round_robin_selection(self, backends: List[str]) -> str:
        """Weighted round-robin selection based on backend weights."""
        # Create weighted list
        weighted_backends = []
        for backend in backends:
            weight = int(self.backend_weights.get(backend, 1) * 10)  # Scale weights
            weighted_backends.extend([backend] * weight)
        
        if not weighted_backends:
            return backends[0]
        
        selected = weighted_backends[self._current_backend_index % len(weighted_backends)]
        self._current_backend_index += 1
        return selected
    
    def _least_response_time_selection(self, backends: List[str]) -> str:
        """Select backend with lowest average response time."""
        return min(backends, key=lambda b: self.backend_metrics[b].avg_response_time or float('inf'))
    
    def _adaptive_performance_selection(
        self, 
        backends: List[str], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Adaptive selection based on performance metrics and context."""
        scores = {}
        
        for backend in backends:
            metrics = self.backend_metrics[backend]
            
            # Performance score (lower is better, so invert)
            response_score = 1.0 / (1.0 + metrics.avg_response_time)
            success_score = metrics.success_rate
            load_score = 1.0 - metrics.current_load
            health_score = metrics.health_score
            
            # Context-aware scoring
            context_score = 1.0
            if context:
                problem_size = context.get('problem_size', 0)
                # Prefer quantum backends for larger problems
                if problem_size > 50 and metrics.quantum_advantage_score > 0:
                    context_score = 1.0 + metrics.quantum_advantage_score
            
            # Combined score
            scores[backend] = (
                self.performance_weight * (response_score + success_score + load_score) / 3.0 +
                (1 - self.performance_weight) * health_score * context_score
            )
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _quantum_advantage_selection(
        self, 
        backends: List[str], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Selection optimized for quantum advantage."""
        if not context:
            # Fallback to adaptive selection
            return self._adaptive_performance_selection(backends, context)
        
        problem_size = context.get('problem_size', 0)
        problem_complexity = context.get('complexity', 'low')
        
        # Score backends based on quantum advantage potential
        scores = {}
        for backend in backends:
            metrics = self.backend_metrics[backend]
            
            # Base performance score
            performance_score = metrics.success_rate * (1.0 - metrics.current_load)
            
            # Quantum advantage score
            quantum_score = metrics.quantum_advantage_score
            
            # Context-based quantum preference
            if problem_size > 100 or problem_complexity in ['high', 'very_high']:
                quantum_multiplier = 2.0
            elif problem_size > 50 or problem_complexity == 'medium':
                quantum_multiplier = 1.5
            else:
                quantum_multiplier = 1.0
            
            # Combined score
            scores[backend] = (
                self.quantum_preference_weight * quantum_score * quantum_multiplier +
                (1 - self.quantum_preference_weight) * performance_score
            )
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _periodic_health_check(self):
        """Perform periodic health checks on backends."""
        current_time = time.time()
        if current_time - self._last_health_check < self.health_check_interval:
            return
        
        for backend_name in self.backends.keys():
            self._check_backend_health(backend_name)
        
        self._last_health_check = current_time
    
    def _check_backend_health(self, backend_name: str):
        """Check health of a specific backend."""
        metrics = self.backend_metrics[backend_name]
        current_time = time.time()
        
        # Check if backend has been failing recently
        if (metrics.last_failure_time and 
            current_time - metrics.last_failure_time < 300 and  # 5 minutes
            metrics.success_rate < 0.5):
            self._unhealthy_backends.add(backend_name)
            logger.warning(f"Marked backend '{backend_name}' as unhealthy (success rate: {metrics.success_rate:.2%})")
        elif backend_name in self._unhealthy_backends and metrics.success_rate > 0.8:
            self._unhealthy_backends.discard(backend_name)
            logger.info(f"Backend '{backend_name}' recovered and marked as healthy")
    
    def _update_health_score(self, backend_name: str):
        """Update health score for a backend."""
        metrics = self.backend_metrics[backend_name]
        
        # Health score based on success rate, response time, and load
        success_component = metrics.success_rate
        response_component = max(0, 1.0 - metrics.avg_response_time / 60.0)  # Normalize to 60s
        load_component = max(0, 1.0 - metrics.current_load)
        
        metrics.health_score = (success_component + response_component + load_component) / 3.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer metrics."""
        with self._lock:
            backend_stats = {}
            for name, metrics in self.backend_metrics.items():
                backend_stats[name] = {
                    'active_connections': metrics.active_connections,
                    'total_requests': metrics.total_requests,
                    'success_rate': metrics.success_rate,
                    'avg_response_time': metrics.avg_response_time,
                    'current_load': metrics.current_load,
                    'health_score': metrics.health_score,
                    'quantum_advantage_score': metrics.quantum_advantage_score,
                    'is_healthy': name not in self._unhealthy_backends
                }
            
            # Selection distribution
            recent_selections = list(self._selection_history)[-100:]  # Last 100 selections
            selection_counts = defaultdict(int)
            for selection, _ in recent_selections:
                selection_counts[selection] += 1
            
            return {
                'strategy': self.strategy.value,
                'total_backends': len(self.backends),
                'healthy_backends': len(self.backends) - len(self._unhealthy_backends),
                'unhealthy_backends': list(self._unhealthy_backends),
                'backend_stats': backend_stats,
                'selection_distribution': dict(selection_counts),
                'quantum_advantage_metrics': self._quantum_advantage_tracker.get_summary()
            }
    
    def rebalance(self) -> Dict[str, Any]:
        """Trigger load rebalancing if needed."""
        with self._lock:
            metrics = self.get_metrics()
            
            # Check if rebalancing is needed
            backend_loads = [stats['current_load'] for stats in metrics['backend_stats'].values()]
            if not backend_loads:
                return {'rebalanced': False, 'reason': 'no_backends'}
            
            load_variance = statistics.variance(backend_loads) if len(backend_loads) > 1 else 0.0
            
            if load_variance > self.rebalance_threshold:
                # Adjust weights based on performance
                for backend_name, stats in metrics['backend_stats'].items():
                    if stats['is_healthy']:
                        # Increase weight for high-performing backends
                        performance_score = stats['success_rate'] * (1.0 - stats['current_load'])
                        self.backend_weights[backend_name] = max(0.1, performance_score * 2.0)
                    else:
                        self.backend_weights[backend_name] = 0.1  # Minimum weight
                
                logger.info("Load rebalancing triggered - adjusted backend weights")
                return {
                    'rebalanced': True,
                    'load_variance': load_variance,
                    'new_weights': self.backend_weights.copy()
                }
            
            return {
                'rebalanced': False,
                'reason': 'below_threshold',
                'load_variance': load_variance
            }
    
    def get_backend_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for backend configuration."""
        metrics = self.get_metrics()
        recommendations = []
        
        # Check for underutilized backends
        for backend_name, stats in metrics['backend_stats'].items():
            if stats['current_load'] < 0.1 and stats['total_requests'] > 10:
                recommendations.append({
                    'type': 'underutilized',
                    'backend': backend_name,
                    'message': f"Backend '{backend_name}' is underutilized ({stats['current_load']:.1%} load)"
                })
        
        # Check for overloaded backends
        for backend_name, stats in metrics['backend_stats'].items():
            if stats['current_load'] > 0.9:
                recommendations.append({
                    'type': 'overloaded',
                    'backend': backend_name,
                    'message': f"Backend '{backend_name}' may be overloaded ({stats['current_load']:.1%} load)"
                })
        
        # Check quantum advantage utilization
        quantum_backends = [name for name, stats in metrics['backend_stats'].items() 
                          if stats['quantum_advantage_score'] > 0.5]
        if quantum_backends:
            total_quantum_requests = sum(
                metrics['selection_distribution'].get(name, 0) for name in quantum_backends
            )
            total_requests = sum(metrics['selection_distribution'].values())
            quantum_utilization = total_quantum_requests / total_requests if total_requests > 0 else 0
            
            if quantum_utilization < 0.3:
                recommendations.append({
                    'type': 'quantum_underutilized',
                    'message': f"Quantum backends underutilized ({quantum_utilization:.1%} of requests)"
                })
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'metrics_summary': {
                'avg_load': statistics.mean([s['current_load'] for s in metrics['backend_stats'].values()]),
                'avg_success_rate': statistics.mean([s['success_rate'] for s in metrics['backend_stats'].values()]),
                'healthy_backend_ratio': metrics['healthy_backends'] / metrics['total_backends'] if metrics['total_backends'] > 0 else 0
            }
        }


class QuantumAdvantageTracker:
    """Tracks quantum advantage measurements across different backends."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
    
    def record_measurement(self, backend_name: str, advantage: float, response_time: float):
        """Record a quantum advantage measurement."""
        self.measurements[backend_name].append({
            'advantage': advantage,
            'response_time': response_time,
            'timestamp': time.time()
        })
    
    def get_advantage_score(self, backend_name: str) -> float:
        """Get quantum advantage score for a backend."""
        if backend_name not in self.measurements or not self.measurements[backend_name]:
            return 0.0
        
        recent_measurements = list(self.measurements[backend_name])[-50:]  # Last 50 measurements
        if not recent_measurements:
            return 0.0
        
        advantages = [m['advantage'] for m in recent_measurements]
        return max(0.0, statistics.mean(advantages))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of quantum advantage tracking."""
        summary = {}
        
        for backend_name, measurements in self.measurements.items():
            if not measurements:
                continue
            
            recent = list(measurements)[-100:]  # Last 100 measurements
            advantages = [m['advantage'] for m in recent]
            response_times = [m['response_time'] for m in recent]
            
            summary[backend_name] = {
                'total_measurements': len(measurements),
                'avg_advantage': statistics.mean(advantages),
                'max_advantage': max(advantages),
                'avg_response_time': statistics.mean(response_times),
                'advantage_trend': self._calculate_trend(advantages[-20:]) if len(advantages) >= 20 else 0.0
            }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return max(-1.0, min(1.0, slope))  # Normalize to [-1, 1]