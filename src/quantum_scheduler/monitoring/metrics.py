"""Metrics collection and monitoring for quantum scheduler."""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class SchedulingMetrics:
    """Metrics for a single scheduling operation."""
    
    start_time: float
    end_time: Optional[float] = None
    num_agents: int = 0
    num_tasks: int = 0
    num_assignments: int = 0
    solver_type: str = "unknown"
    success: bool = False
    error_type: Optional[str] = None
    execution_time: float = 0.0
    cost: float = 0.0
    utilization_ratio: float = 0.0
    
    @property
    def duration(self) -> float:
        """Get operation duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_execution_time: float = 0.0
    total_agents_processed: int = 0
    total_tasks_processed: int = 0
    total_assignments_made: int = 0
    backend_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.successful_operations == 0:
            return 0.0
        return self.total_execution_time / self.successful_operations
    
    @property
    def average_problem_size(self) -> float:
        """Calculate average problem size (agents + tasks)."""
        if self.total_operations == 0:
            return 0.0
        return (self.total_agents_processed + self.total_tasks_processed) / self.total_operations


class MetricsCollector:
    """Collects and aggregates scheduler metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self._lock = threading.Lock()
        self._system_metrics = SystemMetrics()
        self._operation_history: deque = deque(maxlen=max_history)
        self._active_operations: Dict[str, SchedulingMetrics] = {}
        
        logger.info(f"Initialized metrics collector with history size {max_history}")
    
    def start_operation(self, operation_id: str, num_agents: int, num_tasks: int) -> None:
        """Start tracking a new operation.
        
        Args:
            operation_id: Unique identifier for the operation
            num_agents: Number of agents in the problem
            num_tasks: Number of tasks in the problem
        """
        with self._lock:
            metrics = SchedulingMetrics(
                start_time=time.time(),
                num_agents=num_agents,
                num_tasks=num_tasks
            )
            self._active_operations[operation_id] = metrics
            
            logger.debug(f"Started tracking operation {operation_id}")
    
    def end_operation(
        self, 
        operation_id: str, 
        success: bool, 
        solver_type: str = "unknown",
        num_assignments: int = 0,
        cost: float = 0.0,
        utilization_ratio: float = 0.0,
        error_type: Optional[str] = None
    ) -> None:
        """End tracking an operation.
        
        Args:
            operation_id: Operation identifier
            success: Whether operation succeeded
            solver_type: Type of solver used
            num_assignments: Number of assignments made
            cost: Total cost of solution
            utilization_ratio: Resource utilization ratio
            error_type: Type of error if failed
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return
            
            metrics = self._active_operations.pop(operation_id)
            metrics.end_time = time.time()
            metrics.success = success
            metrics.solver_type = solver_type
            metrics.num_assignments = num_assignments
            metrics.cost = cost
            metrics.utilization_ratio = utilization_ratio
            metrics.error_type = error_type
            metrics.execution_time = metrics.duration
            
            # Add to history
            self._operation_history.append(metrics)
            
            # Update system metrics
            self._system_metrics.total_operations += 1
            self._system_metrics.total_agents_processed += metrics.num_agents
            self._system_metrics.total_tasks_processed += metrics.num_tasks
            self._system_metrics.backend_usage[solver_type] += 1
            
            if success:
                self._system_metrics.successful_operations += 1
                self._system_metrics.total_execution_time += metrics.execution_time
                self._system_metrics.total_assignments_made += num_assignments
            else:
                self._system_metrics.failed_operations += 1
                if error_type:
                    self._system_metrics.error_counts[error_type] += 1
            
            logger.debug(f"Completed tracking operation {operation_id}: success={success}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics.
        
        Returns:
            System metrics snapshot
        """
        with self._lock:
            return self._system_metrics
    
    def get_recent_operations(self, count: int = 10) -> List[SchedulingMetrics]:
        """Get recent operations.
        
        Args:
            count: Number of recent operations to return
            
        Returns:
            List of recent operation metrics
        """
        with self._lock:
            return list(self._operation_history)[-count:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Returns:
            Performance summary
        """
        with self._lock:
            if not self._operation_history:
                return {"status": "no_data"}
            
            successful_ops = [op for op in self._operation_history if op.success]
            
            if not successful_ops:
                return {
                    "status": "no_successful_operations",
                    "total_operations": len(self._operation_history),
                    "success_rate": 0.0
                }
            
            execution_times = [op.execution_time for op in successful_ops]
            costs = [op.cost for op in successful_ops]
            utilizations = [op.utilization_ratio for op in successful_ops]
            
            return {
                "status": "ok",
                "total_operations": len(self._operation_history),
                "successful_operations": len(successful_ops),
                "success_rate": len(successful_ops) / len(self._operation_history),
                "execution_time": {
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "avg": sum(execution_times) / len(execution_times),
                    "median": sorted(execution_times)[len(execution_times) // 2]
                },
                "cost": {
                    "min": min(costs),
                    "max": max(costs),
                    "avg": sum(costs) / len(costs),
                    "median": sorted(costs)[len(costs) // 2]
                },
                "utilization": {
                    "min": min(utilizations),
                    "max": max(utilizations),
                    "avg": sum(utilizations) / len(utilizations),
                    "median": sorted(utilizations)[len(utilizations) // 2]
                },
                "backend_usage": dict(self._system_metrics.backend_usage),
                "error_counts": dict(self._system_metrics.error_counts)
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._system_metrics = SystemMetrics()
            self._operation_history.clear()
            self._active_operations.clear()
            
            logger.info("Reset all metrics")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring systems.
        
        Returns:
            Complete metrics data
        """
        with self._lock:
            return {
                "system_metrics": {
                    "total_operations": self._system_metrics.total_operations,
                    "successful_operations": self._system_metrics.successful_operations,
                    "failed_operations": self._system_metrics.failed_operations,
                    "success_rate": self._system_metrics.success_rate,
                    "total_execution_time": self._system_metrics.total_execution_time,
                    "average_execution_time": self._system_metrics.average_execution_time,
                    "total_agents_processed": self._system_metrics.total_agents_processed,
                    "total_tasks_processed": self._system_metrics.total_tasks_processed,
                    "total_assignments_made": self._system_metrics.total_assignments_made,
                    "average_problem_size": self._system_metrics.average_problem_size,
                    "backend_usage": dict(self._system_metrics.backend_usage),
                    "error_counts": dict(self._system_metrics.error_counts)
                },
                "active_operations": len(self._active_operations),
                "history_size": len(self._operation_history),
                "timestamp": time.time()
            }


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def configure_metrics(max_history: int = 1000) -> None:
    """Configure global metrics collector.
    
    Args:
        max_history: Maximum operations to keep in history
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector(max_history)
    logger.info(f"Configured metrics collector with history size {max_history}")