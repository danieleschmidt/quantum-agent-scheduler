"""Performance benchmarking tests for quantum scheduler."""

import json
import time
import pytest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from quantum_scheduler import QuantumScheduler, Agent, Task
from quantum_scheduler.benchmarks import SchedulerBenchmark


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking."""

    @pytest.fixture
    def benchmark_runner(self):
        """Create benchmark runner instance."""
        return SchedulerBenchmark()

    @pytest.fixture
    def performance_problems(self):
        """Load performance test problems."""
        fixtures_path = Path(__file__).parent.parent / "fixtures" / "scheduling_problems.json"
        with open(fixtures_path) as f:
            return json.load(f)

    @pytest.mark.slow
    def test_small_problem_performance(self, benchmark_runner, performance_problems):
        """Test performance on small problem instances."""
        problem = performance_problems["small_problem"]
        
        # Convert to objects
        agents = [Agent(**agent_data) for agent_data in problem["agents"]]
        tasks = [Task(**task_data) for task_data in problem["tasks"]]
        
        # Benchmark classical solver
        start_time = time.time()
        solution = benchmark_runner.solve_classical(
            agents=agents,
            tasks=tasks,
            constraints=problem["constraints"]
        )
        classical_time = time.time() - start_time
        
        # Assertions
        assert solution is not None
        assert classical_time < 1.0  # Should solve in under 1 second
        assert len(solution.assignments) == len(tasks)

    @pytest.mark.slow
    def test_medium_problem_performance(self, benchmark_runner, performance_problems):
        """Test performance on medium problem instances."""
        problem = performance_problems["medium_problem"]
        
        # Generate problem instance
        agents, tasks = benchmark_runner.generate_problem(
            num_agents=len(problem["agents"]),
            num_tasks=len(problem["tasks"]),
            skill_categories=["python", "java", "ml", "web", "data"],
            seed=42
        )
        
        # Benchmark multiple solvers
        results = benchmark_runner.compare_solvers(
            agents=agents,
            tasks=tasks,
            constraints=problem["constraints"],
            solvers=["classical", "quantum_sim"],
            metrics=["time", "quality", "memory"]
        )
        
        # Assertions
        assert "classical" in results
        assert "quantum_sim" in results
        assert results["classical"]["time"] < 10.0  # Classical should be fast
        assert results["classical"]["quality"] >= 0.8  # Good solution quality

    @pytest.mark.slow
    @pytest.mark.quantum
    def test_quantum_advantage_threshold(self, benchmark_runner, performance_problems):
        """Test quantum advantage at different problem sizes."""
        problem_sizes = [10, 25, 50, 100]
        quantum_threshold = None
        
        for size in problem_sizes:
            agents, tasks = benchmark_runner.generate_problem(
                num_agents=size,
                num_tasks=size * 2,
                skill_categories=["python", "java", "ml", "web"],
                seed=size
            )
            
            results = benchmark_runner.compare_solvers(
                agents=agents,
                tasks=tasks,
                constraints={"max_concurrent_tasks": 3},
                solvers=["classical", "quantum_hw"],
                timeout=300
            )
            
            # Check if quantum shows advantage
            if (results["quantum_hw"]["time"] < results["classical"]["time"] and
                results["quantum_hw"]["quality"] >= results["classical"]["quality"]):
                quantum_threshold = size
                break
        
        # Log findings
        if quantum_threshold:
            print(f"Quantum advantage observed at {quantum_threshold} agents")
        else:
            print("No quantum advantage observed in tested range")

    @pytest.mark.slow
    def test_memory_usage_scaling(self, benchmark_runner):
        """Test memory usage scaling with problem size."""
        import psutil
        import os
        
        problem_sizes = [10, 50, 100, 200]
        memory_usage = {}
        
        for size in problem_sizes:
            # Clear memory
            import gc
            gc.collect()
            
            # Generate problem
            agents, tasks = benchmark_runner.generate_problem(
                num_agents=size,
                num_tasks=size * 2,
                seed=size
            )
            
            # Measure memory before solving
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Solve problem
            solution = benchmark_runner.solve_classical(
                agents=agents,
                tasks=tasks,
                constraints={"max_concurrent_tasks": 5}
            )
            
            # Measure memory after solving
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage[size] = memory_after - memory_before
            
            # Clean up
            del agents, tasks, solution
            gc.collect()
        
        # Check memory scaling
        for size in problem_sizes[1:]:
            prev_size = problem_sizes[problem_sizes.index(size) - 1]
            memory_ratio = memory_usage[size] / memory_usage[prev_size]
            size_ratio = size / prev_size
            
            # Memory growth should be sub-quadratic
            assert memory_ratio <= size_ratio ** 1.5, f"Memory scaling too aggressive: {memory_ratio} vs {size_ratio}"

    @pytest.mark.slow
    def test_solver_timeout_handling(self, benchmark_runner):
        """Test solver behavior with timeouts."""
        # Generate large problem
        agents, tasks = benchmark_runner.generate_problem(
            num_agents=100,
            num_tasks=500,
            complexity_factor=2.0,
            seed=999
        )
        
        # Test with very short timeout
        start_time = time.time()
        result = benchmark_runner.solve_with_timeout(
            agents=agents,
            tasks=tasks,
            constraints={"max_concurrent_tasks": 10},
            timeout=1.0  # 1 second timeout
        )
        elapsed_time = time.time() - start_time
        
        # Should respect timeout
        assert elapsed_time <= 2.0  # Allow some overhead
        assert result.status in ["timeout", "suboptimal", "feasible"]

    def test_benchmark_result_serialization(self, benchmark_runner):
        """Test benchmark result serialization."""
        # Generate small problem for quick test
        agents, tasks = benchmark_runner.generate_problem(
            num_agents=5,
            num_tasks=10,
            seed=123
        )
        
        # Run benchmark
        results = benchmark_runner.compare_solvers(
            agents=agents,
            tasks=tasks,
            constraints={},
            solvers=["classical"],
            metrics=["time", "quality", "cost"]
        )
        
        # Test serialization
        json_results = json.dumps(results, default=str)
        deserialized = json.loads(json_results)
        
        assert "classical" in deserialized
        assert "time" in deserialized["classical"]
        assert "quality" in deserialized["classical"]
        assert "cost" in deserialized["classical"]

    @pytest.mark.parametrize("solver_type", ["classical", "quantum_sim"])
    def test_solver_consistency(self, benchmark_runner, solver_type):
        """Test solver consistency across multiple runs."""
        # Generate deterministic problem
        agents, tasks = benchmark_runner.generate_problem(
            num_agents=10,
            num_tasks=20,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Run solver multiple times
        solutions = []
        for _ in range(3):
            solution = benchmark_runner.solve_with_solver(
                agents=agents,
                tasks=tasks,
                constraints={"max_concurrent_tasks": 3},
                solver=solver_type
            )
            solutions.append(solution)
        
        # Check consistency (solutions should be similar)
        base_cost = solutions[0].cost
        for solution in solutions[1:]:
            cost_difference = abs(solution.cost - base_cost) / base_cost
            assert cost_difference < 0.1, f"Solution cost variance too high: {cost_difference}"

    def test_benchmark_reporting(self, benchmark_runner, tmp_path):
        """Test benchmark report generation."""
        # Generate small problem
        agents, tasks = benchmark_runner.generate_problem(
            num_agents=5,
            num_tasks=10,
            seed=456
        )
        
        # Run benchmark
        results = benchmark_runner.compare_solvers(
            agents=agents,
            tasks=tasks,
            constraints={},
            solvers=["classical"],
            metrics=["time", "quality"]
        )
        
        # Generate report
        report_path = tmp_path / "benchmark_report.html"
        benchmark_runner.generate_report(
            results=results,
            output_path=str(report_path),
            include_plots=True
        )
        
        # Verify report exists and has content
        assert report_path.exists()
        assert report_path.stat().st_size > 1000  # Should have substantial content
        
        # Verify HTML structure
        content = report_path.read_text()
        assert "<html>" in content
        assert "benchmark" in content.lower()
        assert "classical" in content