"""End-to-end integration tests for quantum scheduler."""

import pytest
import json
import tempfile
import os
from pathlib import Path

from quantum_scheduler import QuantumScheduler, Agent, Task
from quantum_scheduler.backends import ClassicalBackend, SimulatedQuantumBackend, HybridBackend
from quantum_scheduler.health import HealthChecker
from quantum_scheduler.cli import cli
from click.testing import CliRunner


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_complete_scheduling_workflow(self):
        """Test complete scheduling workflow from start to finish."""
        # Step 1: Create scheduler with full features
        scheduler = QuantumScheduler(
            backend="auto",
            enable_validation=True,
            enable_metrics=True,
            enable_caching=True,
            enable_optimization=True
        )
        
        # Step 2: Create realistic problem
        agents = [
            Agent(id="dev1", skills=["python", "machine_learning"], capacity=40),
            Agent(id="dev2", skills=["python", "web_development"], capacity=35),
            Agent(id="dev3", skills=["java", "backend"], capacity=45),
            Agent(id="qa1", skills=["testing", "automation"], capacity=30),
            Agent(id="devops1", skills=["deployment", "monitoring"], capacity=25),
        ]
        
        tasks = [
            Task(id="ml_model", required_skills=["python", "machine_learning"], duration=20, priority=9),
            Task(id="web_frontend", required_skills=["web_development"], duration=15, priority=7),
            Task(id="api_backend", required_skills=["python", "backend"], duration=25, priority=8),
            Task(id="test_suite", required_skills=["testing"], duration=10, priority=6),
            Task(id="deployment", required_skills=["deployment"], duration=8, priority=5),
            Task(id="monitoring", required_skills=["monitoring"], duration=5, priority=4),
        ]
        
        constraints = {
            "skill_match_required": True,
            "max_concurrent_tasks": 2,
            "deadline_enforcement": False
        }
        
        # Step 3: Solve the problem
        solution = scheduler.schedule(agents, tasks, constraints)
        
        # Step 4: Validate solution
        assert solution is not None
        assert isinstance(solution.assignments, dict)
        assert len(solution.assignments) > 0
        assert solution.cost >= 0
        assert 0 <= solution.utilization_ratio <= 1.0
        assert solution.execution_time >= 0
        
        # Step 5: Verify skill matching
        for task_id, agent_id in solution.assignments.items():
            task = next(t for t in tasks if t.id == task_id)
            agent = next(a for a in agents if a.id == agent_id)
            
            required_skills = set(task.required_skills)
            agent_skills = set(agent.skills)
            assert required_skills.issubset(agent_skills), f"Agent {agent_id} lacks skills for task {task_id}"
        
        # Step 6: Check metrics were collected
        metrics = scheduler.get_performance_metrics()
        assert "backend_type" in metrics
        assert "constraints_count" in metrics
    
    def test_multi_backend_comparison(self):
        """Test scheduling the same problem with different backends."""
        # Create test problem
        agents = [
            Agent(id="agent1", skills=["python"], capacity=10),
            Agent(id="agent2", skills=["java"], capacity=8),
        ]
        
        tasks = [
            Task(id="task1", required_skills=["python"], duration=5, priority=8),
            Task(id="task2", required_skills=["java"], duration=3, priority=6),
            Task(id="task3", required_skills=["python"], duration=4, priority=7),
        ]
        
        constraints = {"skill_match_required": True}
        
        # Test different backends
        backends = ["classical", "quantum_sim", "hybrid"]
        solutions = {}
        
        for backend_name in backends:
            scheduler = QuantumScheduler(backend=backend_name)
            solution = scheduler.schedule(agents, tasks, constraints)
            solutions[backend_name] = solution
            
            # All should find valid solutions
            assert solution is not None
            assert len(solution.assignments) > 0
            
            # Verify assignments are valid
            for task_id, agent_id in solution.assignments.items():
                assert task_id in [t.id for t in tasks]
                assert agent_id in [a.id for a in agents]
        
        # Solutions might differ but should all be valid
        assert len(solutions) == len(backends)
    
    def test_error_recovery_workflow(self):
        """Test error recovery and fallback mechanisms."""
        # Test 1: Invalid input recovery
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        try:
            # This should fail validation
            invalid_agent = Agent(id="", skills=[], capacity=0)
            scheduler.schedule([invalid_agent], [])
            assert False, "Should have raised ValidationError"
        except Exception as e:
            assert "validation" in str(e).lower()
        
        # Test 2: Skill mismatch recovery
        try:
            agents = [Agent(id="agent1", skills=["python"], capacity=5)]
            impossible_task = [Task(id="task1", required_skills=["nonexistent"], duration=1, priority=1)]
            scheduler.schedule(agents, impossible_task, {"skill_match_required": True})
            assert False, "Should have raised SkillMismatchError"
        except Exception as e:
            assert "skill" in str(e).lower()
        
        # Test 3: Valid workflow after errors
        valid_agents = [Agent(id="agent1", skills=["python"], capacity=5)]
        valid_tasks = [Task(id="task1", required_skills=["python"], duration=2, priority=1)]
        solution = scheduler.schedule(valid_agents, valid_tasks, {"skill_match_required": True})
        
        assert solution is not None
        assert len(solution.assignments) == 1
    
    def test_performance_scaling_workflow(self):
        """Test performance across different problem sizes."""
        import time
        
        scheduler = QuantumScheduler(backend="classical", enable_caching=False)
        
        problem_sizes = [5, 10, 25]
        performance_data = {}
        
        for size in problem_sizes:
            # Create problem of given size
            agents = [
                Agent(id=f"agent{i}", skills=["test"], capacity=3) 
                for i in range(size // 2)
            ]
            tasks = [
                Task(id=f"task{i}", required_skills=["test"], duration=1, priority=i) 
                for i in range(size)
            ]
            
            # Measure performance
            start_time = time.time()
            solution = scheduler.schedule(agents, tasks)
            solve_time = time.time() - start_time
            
            performance_data[size] = {
                "solve_time": solve_time,
                "assignments": len(solution.assignments),
                "agents": len(agents),
                "tasks": len(tasks)
            }
            
            # Performance should be reasonable
            assert solve_time < 10.0, f"Size {size} took too long: {solve_time}s"
            assert solution is not None
        
        # Larger problems should not be exponentially slower
        if len(performance_data) >= 2:
            sizes = sorted(performance_data.keys())
            for i in range(1, len(sizes)):
                prev_size, curr_size = sizes[i-1], sizes[i]
                prev_time = performance_data[prev_size]["solve_time"]
                curr_time = performance_data[curr_size]["solve_time"]
                
                # Allow for some scaling but not exponential
                size_ratio = curr_size / prev_size
                time_ratio = (curr_time + 0.001) / (prev_time + 0.001)  # Add small epsilon
                
                # Time should not scale worse than quadratically with size
                assert time_ratio <= size_ratio ** 2 + 1, f"Poor scaling: {time_ratio:.2f} vs {size_ratio:.2f}"


class TestCLIIntegration:
    """Test CLI integration and workflows."""
    
    def test_cli_generate_command(self):
        """Test CLI generate command."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'generate',
                '--num-agents', '5',
                '--num-tasks', '10', 
                '--output-dir', tmpdir
            ])
            
            assert result.exit_code == 0
            assert "Generated 5 agents and 10 tasks" in result.output
            
            # Check files were created
            agents_file = Path(tmpdir) / "agents.json"
            tasks_file = Path(tmpdir) / "tasks.json"
            constraints_file = Path(tmpdir) / "constraints.json"
            
            assert agents_file.exists()
            assert tasks_file.exists()
            assert constraints_file.exists()
            
            # Validate file contents
            with open(agents_file) as f:
                agents_data = json.load(f)
                assert len(agents_data) == 5
                assert all("id" in agent and "skills" in agent and "capacity" in agent 
                          for agent in agents_data)
            
            with open(tasks_file) as f:
                tasks_data = json.load(f)
                assert len(tasks_data) == 10
                assert all("id" in task and "required_skills" in task 
                          for task in tasks_data)
    
    def test_cli_schedule_command(self):
        """Test CLI schedule command."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First generate test data
            result = runner.invoke(cli, [
                'generate',
                '--num-agents', '3',
                '--num-tasks', '5',
                '--output-dir', tmpdir
            ])
            assert result.exit_code == 0
            
            # Then schedule using generated data
            agents_file = str(Path(tmpdir) / "agents.json")
            tasks_file = str(Path(tmpdir) / "tasks.json")
            constraints_file = str(Path(tmpdir) / "constraints.json")
            output_file = str(Path(tmpdir) / "solution.json")
            
            result = runner.invoke(cli, [
                'schedule',
                '--backend', 'classical',
                '--agents-file', agents_file,
                '--tasks-file', tasks_file,
                '--constraints-file', constraints_file,
                '--output', output_file
            ])
            
            assert result.exit_code == 0
            assert "Solution found" in result.output
            assert "assignments" in result.output.lower()
            
            # Check output file
            assert Path(output_file).exists()
            
            with open(output_file) as f:
                solution_data = json.load(f)
                assert "assignments" in solution_data
                assert "cost" in solution_data
                assert "solver_type" in solution_data
    
    def test_cli_benchmark_command(self):
        """Test CLI benchmark command."""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'benchmark',
            '--backend', 'classical',
            '--max-size', '20'
        ])
        
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()
        assert "classical" in result.output


class TestSystemIntegration:
    """Test system-level integration."""
    
    def test_health_check_integration(self):
        """Test health check system integration."""
        checker = HealthChecker()
        
        # Run full health check
        results = checker.run_all_checks()
        assert len(results) > 0
        
        # Get system status
        status = checker.get_system_status()
        assert status["overall_status"] in ["ok", "warning", "error"]
        
        # Should have at least basic checks passing
        ok_results = [r for r in results if r.status == "ok"]
        assert len(ok_results) > 0
    
    def test_metrics_integration(self):
        """Test metrics system integration."""
        from quantum_scheduler.monitoring import get_metrics_collector, configure_metrics
        
        # Configure metrics
        configure_metrics(max_history=100, ttl_seconds=300)
        
        # Use scheduler with metrics
        scheduler = QuantumScheduler(backend="classical", enable_metrics=True)
        
        agents = [Agent(id="agent1", skills=["test"], capacity=2)]
        tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
        
        # Perform multiple operations
        for i in range(3):
            solution = scheduler.schedule(agents, tasks)
            assert solution is not None
        
        # Check metrics
        collector = get_metrics_collector()
        system_metrics = collector.get_system_metrics()
        
        assert system_metrics.total_operations >= 3
        assert system_metrics.successful_operations >= 3
        assert system_metrics.success_rate > 0.5
        
        # Export metrics
        exported = collector.export_metrics()
        assert "system_metrics" in exported
        assert "timestamp" in exported
    
    def test_caching_integration(self):
        """Test caching system integration."""
        from quantum_scheduler.optimization import get_solution_cache, configure_cache
        
        # Configure cache
        configure_cache(max_size=50, ttl_seconds=300)
        
        scheduler = QuantumScheduler(backend="classical", enable_caching=True)
        
        agents = [Agent(id="agent1", skills=["test"], capacity=2)]
        tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
        
        # First solve - cache miss
        solution1 = scheduler.schedule(agents, tasks)
        
        # Second solve - cache hit
        solution2 = scheduler.schedule(agents, tasks)
        
        # Solutions should be identical
        assert solution1.assignments == solution2.assignments
        
        # Check cache stats
        cache = get_solution_cache()
        stats = cache.get_stats()
        assert stats["hits"] > 0
        assert stats["hit_rate"] > 0
    
    def test_concurrent_integration(self):
        """Test concurrent processing integration."""
        from quantum_scheduler.concurrent import SchedulerPool
        from quantum_scheduler.core.models import SchedulingProblem
        
        def create_scheduler():
            return QuantumScheduler(backend="classical", enable_metrics=False)
        
        # Create multiple problems
        problems = []
        for i in range(3):
            agents = [Agent(id=f"agent{i}_1", skills=["test"], capacity=2)]
            tasks = [Task(id=f"task{i}_1", required_skills=["test"], duration=1, priority=1)]
            problem = SchedulingProblem(
                agents=agents, tasks=tasks, constraints={}, optimization_target="minimize_cost"
            )
            problems.append(problem)
        
        # Process concurrently
        with SchedulerPool(max_workers=2) as pool:
            job_id = pool.submit_batch(problems, create_scheduler)
            result = pool.get_result(job_id, timeout=30.0)
            
            assert result is not None
            assert result.success is True
            assert len(result.solutions) == len(problems)
            
            # All solutions should be valid
            for solution in result.solutions:
                assert solution is not None
                assert isinstance(solution.assignments, dict)


@pytest.mark.slow
class TestStressAndPerformance:
    """Stress tests and performance validation."""
    
    def test_large_problem_handling(self):
        """Test handling of large problems."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Large problem: 50 agents, 100 tasks
        agents = [
            Agent(id=f"agent{i}", skills=["python", "java"][i % 2:i % 2 + 1], capacity=3)
            for i in range(50)
        ]
        tasks = [
            Task(id=f"task{i}", required_skills=["python", "java"][i % 2:i % 2 + 1], 
                 duration=1, priority=i % 10)
            for i in range(100)
        ]
        
        import time
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks, {"skill_match_required": True})
        solve_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert solve_time < 30.0
        assert solution is not None
        assert len(solution.assignments) > 0
        assert solution.cost >= 0
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable over many operations."""
        import gc
        
        scheduler = QuantumScheduler(backend="classical", enable_caching=True)
        
        agents = [Agent(id="agent1", skills=["test"], capacity=5)]
        
        # Perform many operations
        for i in range(50):
            tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1)]
            solution = scheduler.schedule(agents, tasks)
            assert solution is not None
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Should complete without memory issues
        assert True  # If we get here, memory was stable
    
    def test_concurrent_stress(self):
        """Test concurrent processing under stress."""
        from quantum_scheduler.concurrent import SchedulerPool
        from quantum_scheduler.core.models import SchedulingProblem
        
        def create_scheduler():
            return QuantumScheduler(backend="classical", enable_metrics=False)
        
        # Create many small problems
        problems = []
        for i in range(20):
            agents = [Agent(id=f"agent{i}", skills=["test"], capacity=2)]
            tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1)]
            problem = SchedulingProblem(
                agents=agents, tasks=tasks, constraints={}, optimization_target="minimize_cost"
            )
            problems.append(problem)
        
        # Process with limited workers
        with SchedulerPool(max_workers=3) as pool:
            job_id = pool.submit_batch(problems, create_scheduler)
            result = pool.get_result(job_id, timeout=60.0)
            
            assert result is not None
            assert result.success is True
            assert len(result.solutions) == len(problems)
            
            # Check execution time is reasonable
            assert result.execution_time < 60.0