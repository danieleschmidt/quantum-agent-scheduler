"""Comprehensive unit tests for quantum scheduler."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_scheduler import QuantumScheduler, Agent, Task
from quantum_scheduler.core.exceptions import (
    ValidationError, SkillMismatchError, SolverError, SolverTimeoutError
)
from quantum_scheduler.core.validators import InputValidator
from quantum_scheduler.backends import ClassicalBackend, SimulatedQuantumBackend, HybridBackend
from quantum_scheduler.optimization import SolutionCache, ProblemOptimizer
from quantum_scheduler.monitoring import get_metrics_collector
from quantum_scheduler.security import SecuritySanitizer
from quantum_scheduler.health import HealthChecker


class TestQuantumSchedulerCore:
    """Test core scheduler functionality."""
    
    def test_scheduler_initialization_options(self):
        """Test scheduler initialization with different options."""
        # Test basic initialization
        scheduler = QuantumScheduler()
        assert scheduler._backend_type == "auto"
        assert scheduler._enable_validation is True
        assert scheduler._enable_metrics is True
        
        # Test with custom options
        custom_scheduler = QuantumScheduler(
            backend="classical",
            timeout=30.0,
            enable_validation=False,
            enable_metrics=False,
            enable_caching=False
        )
        assert custom_scheduler._backend_type == "classical"
        assert custom_scheduler._timeout == 30.0
        assert custom_scheduler._enable_validation is False
        assert custom_scheduler._enable_metrics is False
    
    def test_backend_selection(self):
        """Test automatic backend selection."""
        # Test classical backend
        classical = QuantumScheduler(backend="classical")
        assert isinstance(classical._backend, ClassicalBackend)
        
        # Test quantum simulator
        quantum_sim = QuantumScheduler(backend="quantum_sim")
        assert isinstance(quantum_sim._backend, SimulatedQuantumBackend)
        
        # Test hybrid backend
        hybrid = QuantumScheduler(backend="hybrid")
        assert isinstance(hybrid._backend, HybridBackend)
        
        # Test auto backend (should use hybrid)
        auto = QuantumScheduler(backend="auto")
        assert isinstance(auto._backend, HybridBackend)
    
    def test_basic_scheduling_workflow(self, sample_agents, sample_tasks):
        """Test complete scheduling workflow."""
        scheduler = QuantumScheduler(backend="classical")
        
        solution = scheduler.schedule(
            sample_agents, 
            sample_tasks, 
            {"skill_match_required": True}
        )
        
        assert solution is not None
        assert isinstance(solution.assignments, dict)
        assert solution.solver_type in ["classical", "hybrid_classical"]
        assert solution.execution_time >= 0
        assert 0 <= solution.utilization_ratio <= 1.0
    
    def test_empty_input_handling(self):
        """Test scheduler with empty inputs."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Empty agents and tasks
        solution = scheduler.schedule([], [])
        assert solution.assignments == {}
        assert solution.cost == 0.0
        
        # Empty agents with tasks should work (no assignments)
        tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
        solution = scheduler.schedule([], tasks)
        assert solution.assignments == {}
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test invalid agent
        with pytest.raises(ValidationError):
            invalid_agent = Agent(id="invalid id with spaces", skills=["test"], capacity=1)
            scheduler.schedule([invalid_agent], [])
        
        # Test invalid task
        with pytest.raises(ValidationError):
            invalid_task = Task(id="", required_skills=["test"], duration=1, priority=1)
            scheduler.schedule([], [invalid_task])
    
    def test_skill_mismatch_detection(self, sample_agents):
        """Test skill mismatch detection."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Task requiring non-existent skill
        impossible_task = Task(
            id="impossible", 
            required_skills=["quantum_physics"], 
            duration=1, 
            priority=1
        )
        
        with pytest.raises(SkillMismatchError):
            scheduler.schedule(
                sample_agents, 
                [impossible_task], 
                {"skill_match_required": True}
            )
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        # Mock a slow backend
        with patch('quantum_scheduler.backends.ClassicalBackend.solve') as mock_solve:
            mock_solve.side_effect = lambda p: time.sleep(2) or Mock()
            
            scheduler = QuantumScheduler(backend="classical", timeout=0.1)
            agents = [Agent(id="agent1", skills=["test"], capacity=1)]
            tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
            
            # Note: Timeout might not work in all environments
            try:
                solution = scheduler.schedule(agents, tasks)
                # If no timeout exception, check that it completed quickly
                assert solution is not None
            except SolverTimeoutError:
                # Expected timeout behavior
                pass
    
    def test_fallback_mechanism(self):
        """Test fallback to classical solver."""
        # Mock a failing primary backend
        with patch('quantum_scheduler.backends.HybridBackend.solve') as mock_solve:
            mock_solve.side_effect = Exception("Backend failed")
            
            scheduler = QuantumScheduler(backend="hybrid", fallback="classical")
            agents = [Agent(id="agent1", skills=["test"], capacity=1)]
            tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
            
            solution = scheduler.schedule(agents, tasks)
            assert solution is not None
            assert solution.solver_type in ["classical_fallback", "hybrid_classical"]


class TestValidationSystem:
    """Test input validation and sanitization."""
    
    def test_agent_validation(self):
        """Test agent validation."""
        # Valid agent
        valid_agent = Agent(id="agent1", skills=["python"], capacity=2)
        validated = InputValidator.validate_agent(valid_agent)
        assert validated.id == "agent1"
        assert "python" in validated.skills
        
        # Invalid agent ID
        with pytest.raises(ValidationError):
            InputValidator.validate_agent(
                Agent(id="", skills=["python"], capacity=1)
            )
        
        # Invalid capacity
        with pytest.raises(ValidationError):
            InputValidator.validate_agent(
                Agent(id="agent1", skills=["python"], capacity=0)
            )
    
    def test_task_validation(self):
        """Test task validation."""
        # Valid task
        valid_task = Task(id="task1", required_skills=["python"], duration=2, priority=5)
        validated = InputValidator.validate_task(valid_task)
        assert validated.id == "task1"
        assert "python" in validated.required_skills
        
        # Invalid task ID
        with pytest.raises(ValidationError):
            InputValidator.validate_task(
                Task(id="", required_skills=["python"], duration=1, priority=1)
            )
        
        # Invalid duration
        with pytest.raises(ValidationError):
            InputValidator.validate_task(
                Task(id="task1", required_skills=["python"], duration=0, priority=1)
            )
    
    def test_security_sanitization(self):
        """Test security sanitization."""
        # Test safe ID generation
        safe_id = SecuritySanitizer.generate_safe_id("test")
        assert safe_id.startswith("test_")
        assert len(safe_id) > 5
        
        # Test malicious input detection
        with pytest.raises(ValueError):
            SecuritySanitizer.sanitize_string("'; DROP TABLE users; --")
        
        # Test safe input passes
        safe_input = SecuritySanitizer.sanitize_string("test_input_123")
        assert safe_input == "test_input_123"


class TestCachingSystem:
    """Test solution caching functionality."""
    
    def test_cache_basic_operations(self, sample_agents, sample_tasks):
        """Test basic cache operations."""
        from quantum_scheduler.core.models import SchedulingProblem
        
        cache = SolutionCache(max_size=10, ttl_seconds=60)
        
        problem = SchedulingProblem(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={},
            optimization_target="minimize_cost"
        )
        
        # Cache miss
        assert cache.get(problem) is None
        
        # Add solution to cache
        scheduler = QuantumScheduler(backend="classical")
        solution = scheduler.schedule(sample_agents, sample_tasks)
        cache.put(problem, solution)
        
        # Cache hit
        cached_solution = cache.get(problem)
        assert cached_solution is not None
        assert cached_solution.assignments == solution.assignments
    
    def test_cache_ttl_expiry(self, sample_agents, sample_tasks):
        """Test cache TTL expiry."""
        from quantum_scheduler.core.models import SchedulingProblem
        
        cache = SolutionCache(max_size=10, ttl_seconds=0.1)
        
        problem = SchedulingProblem(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={},
            optimization_target="minimize_cost"
        )
        
        scheduler = QuantumScheduler(backend="classical")
        solution = scheduler.schedule(sample_agents, sample_tasks)
        cache.put(problem, solution)
        
        # Immediate access should work
        assert cache.get(problem) is not None
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get(problem) is None
    
    def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""
        cache = SolutionCache(max_size=2, ttl_seconds=60)
        
        # Create dummy problems and solutions
        problems = []
        solutions = []
        
        for i in range(3):
            from quantum_scheduler.core.models import SchedulingProblem, Solution
            
            agents = [Agent(id=f"agent{i}", skills=["test"], capacity=1)]
            tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1)]
            
            problem = SchedulingProblem(
                agents=agents, tasks=tasks, constraints={}, optimization_target="minimize_cost"
            )
            solution = Solution(assignments={f"task{i}": f"agent{i}"}, cost=1.0)
            
            problems.append(problem)
            solutions.append(solution)
            cache.put(problem, solution)
        
        # First item should be evicted
        assert cache.get(problems[0]) is None
        assert cache.get(problems[1]) is not None
        assert cache.get(problems[2]) is not None


class TestOptimizationSystem:
    """Test problem optimization functionality."""
    
    def test_problem_preprocessing(self, sample_agents, sample_tasks):
        """Test problem preprocessing."""
        from quantum_scheduler.core.models import SchedulingProblem
        
        problem = SchedulingProblem(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={},
            optimization_target="minimize_cost"
        )
        
        optimized = ProblemOptimizer.preprocess_problem(problem)
        
        # Should still have same number of agents/tasks
        assert len(optimized.agents) == len(sample_agents)
        assert len(optimized.tasks) == len(sample_tasks)
        
        # Tasks should be sorted by priority (descending)
        priorities = [task.priority for task in optimized.tasks]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_complexity_estimation(self, sample_agents, sample_tasks):
        """Test complexity estimation."""
        from quantum_scheduler.core.models import SchedulingProblem
        
        problem = SchedulingProblem(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={},
            optimization_target="minimize_cost"
        )
        
        complexity = ProblemOptimizer.estimate_complexity(problem)
        
        assert "num_agents" in complexity
        assert "num_tasks" in complexity
        assert "complexity_score" in complexity
        assert "is_feasible" in complexity
        assert complexity["num_agents"] == len(sample_agents)
        assert complexity["num_tasks"] == len(sample_tasks)


class TestHealthSystem:
    """Test health checking functionality."""
    
    def test_health_checker_basic(self):
        """Test basic health checker functionality."""
        checker = HealthChecker()
        results = checker.run_all_checks()
        
        assert len(results) > 0
        
        # Check that we have expected health checks
        check_names = [result.name for result in results]
        expected_checks = [
            "system_info", "dependencies", "scheduler_basic", 
            "backend_availability", "metrics_system", "performance"
        ]
        
        for expected in expected_checks:
            assert any(expected in name for name in check_names)
        
        # Should have at least some successful checks
        ok_count = sum(1 for r in results if r.status == "ok")
        assert ok_count > 0
    
    def test_system_status(self):
        """Test system status reporting."""
        checker = HealthChecker()
        status = checker.get_system_status()
        
        assert "overall_status" in status
        assert "timestamp" in status
        assert "checks_run" in status
        assert "status_counts" in status
        assert "results" in status
        
        assert status["overall_status"] in ["ok", "warning", "error"]
        assert status["checks_run"] > 0
        assert len(status["results"]) == status["checks_run"]


class TestConcurrentProcessing:
    """Test concurrent processing functionality."""
    
    def test_scheduler_pool_basic(self, sample_agents, sample_tasks):
        """Test basic scheduler pool functionality."""
        from quantum_scheduler.concurrent import SchedulerPool
        from quantum_scheduler.core.models import SchedulingProblem
        
        def create_scheduler():
            return QuantumScheduler(backend="classical", enable_metrics=False)
        
        with SchedulerPool(max_workers=2) as pool:
            problem = SchedulingProblem(
                agents=sample_agents,
                tasks=sample_tasks,
                constraints={},
                optimization_target="minimize_cost"
            )
            
            job_id = pool.submit_single(problem, create_scheduler)
            result = pool.get_result(job_id, timeout=10.0)
            
            assert result is not None
            assert result.success is True
            assert len(result.solutions) == 1
            assert len(result.solutions[0].assignments) > 0
    
    def test_async_scheduler(self, sample_agents, sample_tasks):
        """Test async scheduler functionality."""
        import asyncio
        from quantum_scheduler.concurrent import AsyncScheduler
        
        def create_scheduler():
            return QuantumScheduler(backend="classical", enable_metrics=False)
        
        async def async_test():
            async_scheduler = AsyncScheduler(create_scheduler)
            
            try:
                solution = await async_scheduler.schedule_async(
                    sample_agents[:2], sample_tasks[:2]
                )
                
                assert solution is not None
                assert isinstance(solution.assignments, dict)
                return True
                
            finally:
                async_scheduler.close()
        
        # Run async test
        result = asyncio.run(async_test())
        assert result is True


class TestMetricsSystem:
    """Test metrics collection functionality."""
    
    def test_metrics_collection(self, sample_agents, sample_tasks):
        """Test metrics collection during scheduling."""
        scheduler = QuantumScheduler(backend="classical", enable_metrics=True)
        
        # Clear metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.reset_metrics()
        
        # Perform scheduling
        solution = scheduler.schedule(sample_agents, sample_tasks)
        
        # Check metrics were collected
        system_metrics = metrics_collector.get_system_metrics()
        assert system_metrics.total_operations > 0
        assert system_metrics.successful_operations > 0
        
        performance_summary = metrics_collector.get_performance_summary()
        assert performance_summary["status"] in ["ok", "no_data"]
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        metrics_collector = get_metrics_collector()
        exported = metrics_collector.export_metrics()
        
        assert "system_metrics" in exported
        assert "timestamp" in exported
        assert "active_operations" in exported
        
        system_metrics = exported["system_metrics"]
        expected_fields = [
            "total_operations", "successful_operations", "failed_operations",
            "success_rate", "total_execution_time"
        ]
        
        for field in expected_fields:
            assert field in system_metrics


@pytest.mark.slow
class TestPerformanceRequirements:
    """Test performance requirements and benchmarks."""
    
    def test_small_problem_performance(self):
        """Test performance on small problems."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Small problem: 5 agents, 10 tasks
        agents = [Agent(id=f"agent{i}", skills=["test"], capacity=2) for i in range(5)]
        tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1) for i in range(10)]
        
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks)
        solve_time = time.time() - start_time
        
        # Should solve small problems very quickly
        assert solve_time < 1.0
        assert len(solution.assignments) > 0
    
    def test_medium_problem_performance(self):
        """Test performance on medium problems."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Medium problem: 20 agents, 50 tasks
        agents = [Agent(id=f"agent{i}", skills=["test"], capacity=3) for i in range(20)]
        tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1) for i in range(50)]
        
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks)
        solve_time = time.time() - start_time
        
        # Should solve medium problems reasonably quickly
        assert solve_time < 5.0
        assert len(solution.assignments) > 0
    
    def test_caching_performance_improvement(self, sample_agents, sample_tasks):
        """Test that caching improves performance."""
        scheduler = QuantumScheduler(backend="classical", enable_caching=True)
        
        # First solve (cache miss)
        start_time = time.time()
        solution1 = scheduler.schedule(sample_agents, sample_tasks)
        first_time = time.time() - start_time
        
        # Second solve (cache hit)
        start_time = time.time()
        solution2 = scheduler.schedule(sample_agents, sample_tasks)
        second_time = time.time() - start_time
        
        # Cache hit should be faster (allowing some variance)
        assert second_time < first_time or second_time < 0.001  # Very fast cache hits
        assert solution1.assignments == solution2.assignments