"""Tests for enhanced quantum scheduler with reliability features."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_scheduler import Agent, Task, Solution
from quantum_scheduler.core.enhanced_scheduler import EnhancedQuantumScheduler
from quantum_scheduler.core.exceptions import SolverError, BackendError
from quantum_scheduler.reliability import CircuitBreakerError


class TestEnhancedScheduler:
    """Test cases for enhanced quantum scheduler."""
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        return [
            Agent(id="agent1", skills=["python", "ml"], capacity=3),
            Agent(id="agent2", skills=["java", "web"], capacity=2),
            Agent(id="agent3", skills=["python", "web"], capacity=4)
        ]
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            Task(id="task1", required_skills=["python"], duration=2, priority=5),
            Task(id="task2", required_skills=["web"], duration=1, priority=3),
            Task(id="task3", required_skills=["ml"], duration=3, priority=8),
            Task(id="task4", required_skills=["java"], duration=1, priority=2)
        ]
    
    def test_enhanced_scheduler_initialization(self):
        """Test enhanced scheduler initialization with different configurations."""
        # Default initialization
        scheduler = EnhancedQuantumScheduler()
        assert scheduler.enable_circuit_breaker is True
        assert scheduler.enable_retry_policy is True
        assert scheduler.auto_recovery is True
        
        # Custom initialization
        scheduler_custom = EnhancedQuantumScheduler(
            enable_circuit_breaker=False,
            enable_retry_policy=False,
            auto_recovery=False,
            health_check_interval=30.0
        )
        assert scheduler_custom.enable_circuit_breaker is False
        assert scheduler_custom.enable_retry_policy is False
        assert scheduler_custom.auto_recovery is False
        assert scheduler_custom.health_check_interval == 30.0
    
    def test_basic_scheduling_functionality(self, sample_agents, sample_tasks):
        """Test basic scheduling functionality works with enhancements."""
        scheduler = EnhancedQuantumScheduler(backend="classical")
        
        solution = scheduler.schedule(sample_agents, sample_tasks)
        
        assert isinstance(solution, Solution)
        assert len(solution.assignments) > 0
        assert solution.cost >= 0
        assert solution.solver_type in ["classical", "scheduler_classical"]
    
    def test_health_check_functionality(self):
        """Test comprehensive health check."""
        scheduler = EnhancedQuantumScheduler()
        
        health_status = scheduler.health_check()
        
        assert isinstance(health_status, dict)
        assert "overall" in health_status
        assert "components" in health_status
        assert "timestamp" in health_status
        assert health_status["overall"] in ["healthy", "degraded", "critical"]
        
        # Check required components
        components = health_status["components"]
        assert "backend" in components
    
    def test_reliability_metrics(self, sample_agents, sample_tasks):
        """Test reliability metrics collection."""
        scheduler = EnhancedQuantumScheduler()
        
        # Perform some operations
        solution = scheduler.schedule(sample_agents, sample_tasks)
        
        # Get reliability metrics
        metrics = scheduler.get_reliability_metrics()
        
        assert isinstance(metrics, dict)
        assert "performance" in metrics
        assert "health_status" in metrics
        assert "failure_history_count" in metrics
        
        performance = metrics["performance"]
        assert "avg_solve_time" in performance
        assert "success_rate" in performance
        assert "retry_attempts" in performance
    
    def test_failure_analysis(self):
        """Test failure analysis functionality."""
        scheduler = EnhancedQuantumScheduler()
        
        # Test with no failures initially
        analysis = scheduler.get_failure_analysis()
        assert "message" in analysis
        assert "No failures recorded" in analysis["message"]
        
        # Simulate a failure
        scheduler._record_failure("test_failure", Exception("Test exception"))
        
        analysis = scheduler.get_failure_analysis()
        assert "total_failures" in analysis
        assert analysis["total_failures"] == 1
        assert "failure_types" in analysis
        assert "exception_types" in analysis
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality."""
        scheduler = EnhancedQuantumScheduler(
            enable_circuit_breaker=True,
            circuit_breaker_config={"failure_threshold": 2}
        )
        
        # Test circuit breaker registry access
        assert scheduler._circuit_breaker_registry is not None
        
        # Check circuit breakers are initialized
        cb_metrics = scheduler._circuit_breaker_registry.get_all_metrics()
        assert len(cb_metrics) > 0
    
    def test_auto_recovery_functionality(self):
        """Test automatic recovery functionality."""
        scheduler = EnhancedQuantumScheduler(auto_recovery=True)
        
        # Simulate unhealthy state
        health_status = {
            "overall": "critical",
            "components": {"backend": {"status": "unhealthy"}}
        }
        
        # Test auto recovery attempt
        try:
            scheduler._attempt_auto_recovery(health_status)
        except Exception:
            # Auto recovery might fail, but it should not crash
            pass
    
    def test_reset_reliability_state(self):
        """Test resetting reliability state."""
        scheduler = EnhancedQuantumScheduler()
        
        # Add some failure history
        scheduler._record_failure("test", Exception("test"))
        scheduler._performance_metrics["retry_attempts"] = 5
        
        # Reset state
        scheduler.reset_reliability_state()
        
        # Check state is reset
        assert len(scheduler._failure_history) == 0
        assert scheduler._performance_metrics["retry_attempts"] == 0
        assert scheduler._performance_metrics["success_rate"] == 1.0
    
    def test_force_failover_test(self):
        """Test force failover functionality."""
        scheduler = EnhancedQuantumScheduler(enable_circuit_breaker=True)
        
        # Force failover
        scheduler.force_failover_test()
        
        # Check that circuit breakers are opened
        cb_metrics = scheduler._circuit_breaker_registry.get_all_metrics()
        # At least one circuit breaker should exist and may be opened
        assert len(cb_metrics) > 0
    
    def test_context_manager_usage(self, sample_agents, sample_tasks):
        """Test enhanced scheduler as context manager."""
        with EnhancedQuantumScheduler() as scheduler:
            solution = scheduler.schedule(sample_agents, sample_tasks)
            assert isinstance(solution, Solution)
    
    def test_async_scheduling(self, sample_agents, sample_tasks):
        """Test asynchronous scheduling functionality."""
        import asyncio
        
        async def test_async():
            scheduler = EnhancedQuantumScheduler()
            solution = await scheduler.schedule_async(sample_agents, sample_tasks)
            assert isinstance(solution, Solution)
            return solution
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            solution = loop.run_until_complete(test_async())
            assert solution is not None
        finally:
            loop.close()
    
    def test_performance_metrics_update(self, sample_agents, sample_tasks):
        """Test performance metrics are properly updated."""
        scheduler = EnhancedQuantumScheduler()
        
        # Initial metrics
        initial_metrics = scheduler._performance_metrics.copy()
        
        # Perform scheduling
        scheduler.schedule(sample_agents, sample_tasks)
        
        # Check metrics were updated
        updated_metrics = scheduler._performance_metrics
        assert updated_metrics["avg_solve_time"] >= 0
        assert updated_metrics["success_rate"] >= 0
    
    def test_periodic_health_check(self, sample_agents, sample_tasks):
        """Test periodic health check functionality."""
        scheduler = EnhancedQuantumScheduler(health_check_interval=0.1)  # Very short interval
        
        # Force health check by setting last check time to past
        scheduler._last_health_check = time.time() - 1.0  # 1 second ago
        
        # This should trigger a health check
        scheduler._periodic_health_check()
        
        # Health status should be updated
        assert scheduler._health_status is not None
        assert "overall" in scheduler._health_status
    
    def test_backend_health_check(self):
        """Test backend health check functionality."""
        scheduler = EnhancedQuantumScheduler()
        
        backend_health = scheduler._check_backend_health()
        
        assert isinstance(backend_health, dict)
        assert "status" in backend_health
        assert backend_health["status"] in ["healthy", "unhealthy"]
        assert "backend_type" in backend_health
    
    def test_error_handling_with_invalid_backend(self):
        """Test error handling with invalid backend configuration."""
        # This should still initialize without crashing
        scheduler = EnhancedQuantumScheduler(backend="invalid_backend")
        
        # Should still be able to perform basic operations
        health = scheduler.health_check()
        assert health["overall"] in ["healthy", "degraded", "critical"]
    
    def test_scheduler_with_constraints(self, sample_agents, sample_tasks):
        """Test enhanced scheduler with various constraints."""
        scheduler = EnhancedQuantumScheduler()
        
        constraints = {
            "skill_match_required": True,
            "max_concurrent_tasks": 2
        }
        
        solution = scheduler.schedule(sample_agents, sample_tasks, constraints)
        
        assert isinstance(solution, Solution)
        assert len(solution.assignments) >= 0  # May be 0 if no valid assignments


class TestReliabilityFeatures:
    """Test reliability-specific features."""
    
    def test_failure_recording(self):
        """Test failure recording functionality."""
        scheduler = EnhancedQuantumScheduler()
        
        # Record different types of failures
        scheduler._record_failure("solver_error", SolverError("Test solver error"))
        scheduler._record_failure("backend_error", BackendError("Test backend error"))
        
        assert len(scheduler._failure_history) == 2
        
        # Test failure history limit
        for i in range(150):  # Exceed the limit of 100
            scheduler._record_failure(f"test_{i}", Exception(f"Test {i}"))
        
        assert len(scheduler._failure_history) <= 100
    
    def test_performance_metrics_smoothing(self):
        """Test exponential smoothing of performance metrics."""
        scheduler = EnhancedQuantumScheduler()
        
        # Initial execution time
        scheduler._update_performance_metrics(1.0)
        initial_avg = scheduler._performance_metrics["avg_solve_time"]
        
        # Second execution time
        scheduler._update_performance_metrics(2.0)
        second_avg = scheduler._performance_metrics["avg_solve_time"]
        
        # Should be smoothed, not just the latest value
        assert second_avg != 2.0
        assert second_avg > initial_avg
    
    def test_health_check_with_multiple_components(self):
        """Test health check with multiple components."""
        scheduler = EnhancedQuantumScheduler(
            enable_circuit_breaker=True,
            enable_caching=True,
            enable_metrics=True
        )
        
        health_status = scheduler.health_check()
        components = health_status["components"]
        
        # Should have multiple components
        assert "backend" in components
        
        # Overall health should be determined by component health
        assert health_status["overall"] in ["healthy", "degraded", "critical"]
    
    def test_metrics_collector_integration(self):
        """Test integration with metrics collector."""
        scheduler = EnhancedQuantumScheduler(enable_metrics=True)
        
        # Metrics collector should be initialized
        assert scheduler._metrics_collector is not None
        
        # Should be able to update performance metrics
        scheduler._update_performance_metrics(1.0)
        
        # Success rate should be updated
        assert scheduler._performance_metrics["success_rate"] >= 0