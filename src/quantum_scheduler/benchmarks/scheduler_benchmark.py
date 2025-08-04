"""Scheduler performance benchmarking tools."""

import time
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core import QuantumScheduler, Agent, Task, Solution
from ..backends import ClassicalBackend, SimulatedQuantumBackend, HybridBackend

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a scheduler benchmark."""
    
    solver_type: str
    problem_size: int
    execution_time: float
    solution_quality: float
    memory_usage: float
    success_rate: float
    error_count: int
    
    @property
    def quality_per_second(self) -> float:
        """Quality per second metric."""
        return self.solution_quality / max(self.execution_time, 0.001)


class SchedulerBenchmark:
    """Comprehensive scheduler performance benchmarking."""
    
    def __init__(self, max_workers: int = 4, timeout: float = 300.0):
        """Initialize benchmarking system.
        
        Args:
            max_workers: Maximum number of parallel benchmark workers
            timeout: Maximum time for each benchmark run in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.results: List[BenchmarkResult] = []
        
    def compare_solvers(
        self,
        problem_sizes: List[int],
        solvers: List[str] = None,
        metrics: List[str] = None,
        num_runs: int = 5
    ) -> Dict[str, List[BenchmarkResult]]:
        """Compare performance across different solvers and problem sizes.
        
        Args:
            problem_sizes: List of problem sizes (number of agents/tasks)
            solvers: List of solver types to compare
            metrics: List of metrics to collect
            num_runs: Number of runs per configuration
            
        Returns:
            Dictionary mapping solver names to benchmark results
        """
        if solvers is None:
            solvers = ["classical", "quantum_sim", "hybrid"]
        
        if metrics is None:
            metrics = ["time", "quality", "memory"]
        
        logger.info(f"Starting comparative benchmark with {len(solvers)} solvers, {len(problem_sizes)} sizes, {num_runs} runs each")
        
        all_results = {}
        
        for solver in solvers:
            solver_results = []
            logger.info(f"Benchmarking {solver} solver...")
            
            for size in problem_sizes:
                size_results = []
                
                for run in range(num_runs):
                    try:
                        result = self._benchmark_single_run(solver, size, metrics)
                        size_results.append(result)
                        logger.debug(f"{solver} size={size} run={run}: {result.execution_time:.3f}s, quality={result.solution_quality:.2f}")
                    except Exception as e:
                        logger.error(f"Benchmark failed for {solver} size={size} run={run}: {e}")
                        # Add failed result
                        size_results.append(BenchmarkResult(
                            solver_type=solver,
                            problem_size=size,
                            execution_time=float('inf'),
                            solution_quality=0.0,
                            memory_usage=0.0,
                            success_rate=0.0,
                            error_count=1
                        ))
                
                # Aggregate results for this size
                if size_results:
                    successful_results = [r for r in size_results if r.error_count == 0]
                    if successful_results:
                        avg_result = self._aggregate_results(successful_results, solver, size)
                        avg_result.success_rate = len(successful_results) / len(size_results)
                        solver_results.append(avg_result)
                    else:
                        # All runs failed
                        failed_result = size_results[0]
                        failed_result.success_rate = 0.0
                        solver_results.append(failed_result)
            
            all_results[solver] = solver_results
        
        self.results.extend([r for results in all_results.values() for r in results])
        logger.info(f"Benchmark completed. Total results: {len(self.results)}")
        
        return all_results
    
    def _benchmark_single_run(
        self, 
        solver_type: str, 
        problem_size: int, 
        metrics: List[str]
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        # Generate test problem
        agents, tasks = self._generate_test_problem(problem_size)
        
        # Initialize scheduler
        scheduler = self._create_scheduler(solver_type)
        
        # Measure execution
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            solution = scheduler.schedule(agents, tasks, {
                "skill_match_required": True,
                "max_concurrent_tasks": min(problem_size // 2, 10)
            })
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            # Calculate quality metrics
            quality = self._calculate_solution_quality(solution, agents, tasks)
            
            return BenchmarkResult(
                solver_type=solver_type,
                problem_size=problem_size,
                execution_time=execution_time,
                solution_quality=quality,
                memory_usage=max(memory_usage, 0),
                success_rate=1.0,
                error_count=0
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Benchmark run failed: {e}")
            
            return BenchmarkResult(
                solver_type=solver_type,
                problem_size=problem_size,
                execution_time=execution_time,
                solution_quality=0.0,
                memory_usage=0.0,
                success_rate=0.0,
                error_count=1
            )
    
    def _generate_test_problem(self, size: int) -> Tuple[List[Agent], List[Task]]:
        """Generate a test problem of specified size."""
        import random
        
        skills = ["python", "java", "ml", "web", "data", "security", "devops", "testing"]
        
        # Generate agents
        agents = []
        for i in range(size):
            agent_skills = random.sample(skills, random.randint(2, 4))
            capacity = random.randint(1, 3)
            agents.append(Agent(
                id=f"agent_{i}",
                skills=agent_skills,
                capacity=capacity
            ))
        
        # Generate tasks
        tasks = []
        for i in range(size):
            required_skills = random.sample(skills, random.randint(1, 2))
            duration = random.randint(1, 5)
            priority = random.uniform(1.0, 10.0)
            tasks.append(Task(
                id=f"task_{i}",
                required_skills=required_skills,
                duration=duration,
                priority=priority
            ))
        
        return agents, tasks
    
    def _create_scheduler(self, solver_type: str) -> QuantumScheduler:
        """Create scheduler of specified type."""
        if solver_type == "classical":
            return QuantumScheduler(backend="classical", timeout=self.timeout)
        elif solver_type == "quantum_sim":
            return QuantumScheduler(backend="quantum_sim", timeout=self.timeout)
        elif solver_type == "hybrid":
            return QuantumScheduler(backend="hybrid", timeout=self.timeout)
        else:
            return QuantumScheduler(backend="auto", timeout=self.timeout)
    
    def _calculate_solution_quality(
        self, 
        solution: Solution, 
        agents: List[Agent], 
        tasks: List[Task]
    ) -> float:
        """Calculate solution quality score (0-100)."""
        if not solution.assignments:
            return 0.0
        
        # Factors: assignment efficiency, skill matching, load balancing
        assignment_ratio = len(solution.assignments) / len(tasks)
        
        # Calculate skill match quality
        skill_matches = 0
        total_assignments = len(solution.assignments)
        
        agent_map = {a.id: a for a in agents}
        task_map = {t.id: t for t in tasks}
        
        for task_id, agent_id in solution.assignments.items():
            if agent_id in agent_map and task_id in task_map:
                agent = agent_map[agent_id]
                task = task_map[task_id]
                
                required_skills = set(task.required_skills)
                agent_skills = set(agent.skills)
                
                if required_skills.issubset(agent_skills):
                    skill_matches += 1
        
        skill_match_ratio = skill_matches / max(total_assignments, 1)
        
        # Calculate load distribution quality
        agent_loads = {}
        for agent in agents:
            agent_loads[agent.id] = 0
        
        for task_id, agent_id in solution.assignments.items():
            if agent_id in agent_loads and task_id in task_map:
                task = task_map[task_id]
                agent_loads[agent_id] += task.duration
        
        if agent_loads:
            load_values = list(agent_loads.values())
            load_balance = 1.0 - (statistics.stdev(load_values) / max(statistics.mean(load_values), 1))
            load_balance = max(0, min(1, load_balance))
        else:
            load_balance = 0.0
        
        # Combine factors (weighted average)
        quality = (
            assignment_ratio * 0.4 +
            skill_match_ratio * 0.4 +
            load_balance * 0.2
        ) * 100
        
        return min(100.0, quality)
    
    def _aggregate_results(
        self, 
        results: List[BenchmarkResult], 
        solver_type: str, 
        problem_size: int
    ) -> BenchmarkResult:
        """Aggregate multiple benchmark results."""
        avg_time = statistics.mean([r.execution_time for r in results])
        avg_quality = statistics.mean([r.solution_quality for r in results])
        avg_memory = statistics.mean([r.memory_usage for r in results])
        
        return BenchmarkResult(
            solver_type=solver_type,
            problem_size=problem_size,
            execution_time=avg_time,
            solution_quality=avg_quality,
            memory_usage=avg_memory,
            success_rate=1.0,  # Will be set by caller
            error_count=0
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # psutil not available, return 0
            return 0.0
    
    def plot_results(self, results: Dict[str, List[BenchmarkResult]], save_to: str = None) -> None:
        """Plot benchmark results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Execution time comparison
            for solver, solver_results in results.items():
                sizes = [r.problem_size for r in solver_results]
                times = [r.execution_time for r in solver_results]
                ax1.plot(sizes, times, marker='o', label=solver)
            
            ax1.set_xlabel('Problem Size')
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Execution Time vs Problem Size')
            ax1.legend()
            ax1.grid(True)
            
            # Solution quality comparison
            for solver, solver_results in results.items():
                sizes = [r.problem_size for r in solver_results]
                qualities = [r.solution_quality for r in solver_results]
                ax2.plot(sizes, qualities, marker='s', label=solver)
            
            ax2.set_xlabel('Problem Size')
            ax2.set_ylabel('Solution Quality (%)')
            ax2.set_title('Solution Quality vs Problem Size')
            ax2.legend()
            ax2.grid(True)
            
            # Quality per second
            for solver, solver_results in results.items():
                sizes = [r.problem_size for r in solver_results]
                qps = [r.quality_per_second for r in solver_results]
                ax3.plot(sizes, qps, marker='^', label=solver)
            
            ax3.set_xlabel('Problem Size')
            ax3.set_ylabel('Quality per Second')
            ax3.set_title('Efficiency vs Problem Size')
            ax3.legend()
            ax3.grid(True)
            
            # Success rate
            for solver, solver_results in results.items():
                sizes = [r.problem_size for r in solver_results]
                success_rates = [r.success_rate * 100 for r in solver_results]
                ax4.plot(sizes, success_rates, marker='d', label=solver)
            
            ax4.set_xlabel('Problem Size')
            ax4.set_ylabel('Success Rate (%)')
            ax4.set_title('Success Rate vs Problem Size')
            ax4.legend()
            ax4.grid(True)
            ax4.set_ylim(0, 105)
            
            plt.tight_layout()
            
            if save_to:
                plt.savefig(save_to, dpi=300, bbox_inches='tight')
                logger.info(f"Benchmark plot saved to {save_to}")
            else:
                plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available, cannot generate plots")
            
    def export_results(self, filename: str, format: str = "csv") -> None:
        """Export benchmark results to file."""
        if format.lower() == "csv":
            self._export_csv(filename)
        elif format.lower() == "json":
            self._export_json(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self, filename: str) -> None:
        """Export results to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'solver_type', 'problem_size', 'execution_time',
                'solution_quality', 'memory_usage', 'success_rate',
                'error_count', 'quality_per_second'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'solver_type': result.solver_type,
                    'problem_size': result.problem_size,
                    'execution_time': result.execution_time,
                    'solution_quality': result.solution_quality,
                    'memory_usage': result.memory_usage,
                    'success_rate': result.success_rate,
                    'error_count': result.error_count,
                    'quality_per_second': result.quality_per_second
                })
        
        logger.info(f"Results exported to {filename}")
    
    def _export_json(self, filename: str) -> None:
        """Export results to JSON format."""
        import json
        
        data = []
        for result in self.results:
            data.append({
                'solver_type': result.solver_type,
                'problem_size': result.problem_size,
                'execution_time': result.execution_time,
                'solution_quality': result.solution_quality,
                'memory_usage': result.memory_usage,
                'success_rate': result.success_rate,
                'error_count': result.error_count,
                'quality_per_second': result.quality_per_second
            })
        
        with open(filename, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)
        
        logger.info(f"Results exported to {filename}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all benchmark results."""
        if not self.results:
            return {}
        
        by_solver = {}
        for result in self.results:
            solver = result.solver_type
            if solver not in by_solver:
                by_solver[solver] = []
            by_solver[solver].append(result)
        
        summary = {}
        for solver, results in by_solver.items():
            times = [r.execution_time for r in results if r.error_count == 0]
            qualities = [r.solution_quality for r in results if r.error_count == 0]
            
            if times and qualities:
                summary[solver] = {
                    'avg_time': statistics.mean(times),
                    'median_time': statistics.median(times),
                    'avg_quality': statistics.mean(qualities),
                    'median_quality': statistics.median(qualities),
                    'success_rate': len(times) / len(results),
                    'total_runs': len(results)
                }
        
        return summary