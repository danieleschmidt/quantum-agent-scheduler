"""Enhanced performance tests for quantum scheduler."""

import pytest
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

from quantum_scheduler import Agent, Task, QuantumScheduler
from quantum_scheduler.core.enhanced_scheduler import EnhancedQuantumScheduler
from quantum_scheduler.optimization.distributed_scheduler import DistributedQuantumScheduler
from quantum_scheduler.optimization.adaptive_load_balancer import AdaptiveLoadBalancer, LoadBalancingStrategy


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def generate_test_data(self, num_agents: int, num_tasks: int):
        """Generate test agents and tasks."""
        skills_pool = ["python", "java", "web", "ml", "data", "mobile", "devops", "security"]
        
        agents = []
        for i in range(num_agents):
            # Each agent has 2-4 random skills
            agent_skills = [skills_pool[j % len(skills_pool)] for j in range(i, i + 3)]
            agents.append(Agent(
                id=f"agent_{i}",
                skills=agent_skills,
                capacity=5 + (i % 3)
            ))
        
        tasks = []
        for i in range(num_tasks):
            # Each task requires 1-2 skills
            required_skills = [skills_pool[(i + 2) % len(skills_pool)]]
            if i % 3 == 0:  # Some tasks require 2 skills
                required_skills.append(skills_pool[(i + 5) % len(skills_pool)])
            
            tasks.append(Task(
                id=f"task_{i}",
                required_skills=required_skills,
                duration=1 + (i % 4),
                priority=max(1, 10 - (i % 10))  # Ensure priority is never 0
            ))
        
        return agents, tasks
    
    @pytest.mark.performance
    def test_small_problem_performance(self):
        """Test performance with small problems (10 agents, 20 tasks)."""
        scheduler = QuantumScheduler(backend="classical")
        agents, tasks = self.generate_test_data(10, 20)
        
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Small problems should solve quickly (< 1 second)
        assert execution_time < 1.0
        assert len(solution.assignments) > 0
        assert solution.cost >= 0
        
        print(f"Small problem (10A/20T): {execution_time:.3f}s, {len(solution.assignments)} assignments")
    
    @pytest.mark.performance
    def test_medium_problem_performance(self):
        """Test performance with medium problems (50 agents, 100 tasks)."""
        scheduler = QuantumScheduler(backend="classical")
        agents, tasks = self.generate_test_data(50, 100)
        
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Medium problems should solve reasonably quickly (< 5 seconds)
        assert execution_time < 5.0
        assert len(solution.assignments) > 0
        assert solution.cost >= 0
        
        print(f"Medium problem (50A/100T): {execution_time:.3f}s, {len(solution.assignments)} assignments")
    
    @pytest.mark.performance
    def test_large_problem_performance(self):
        """Test performance with large problems (100 agents, 200 tasks)."""
        scheduler = QuantumScheduler(backend="classical")
        agents, tasks = self.generate_test_data(100, 200)
        
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Large problems should solve within reasonable time (< 30 seconds)
        assert execution_time < 30.0
        assert len(solution.assignments) > 0
        assert solution.cost >= 0
        
        print(f"Large problem (100A/200T): {execution_time:.3f}s, {len(solution.assignments)} assignments")
    
    @pytest.mark.performance
    def test_enhanced_scheduler_performance_overhead(self):
        """Test performance overhead of enhanced features."""
        # Basic scheduler
        basic_scheduler = QuantumScheduler(backend="classical")
        
        # Enhanced scheduler with all features
        enhanced_scheduler = EnhancedQuantumScheduler(
            backend="classical",
            enable_circuit_breaker=True,
            enable_retry_policy=True,
            enable_metrics=True
        )
        
        agents, tasks = self.generate_test_data(50, 100)
        
        # Measure basic scheduler
        start_time = time.time()
        basic_solution = basic_scheduler.schedule(agents, tasks)
        basic_time = time.time() - start_time
        
        # Measure enhanced scheduler
        start_time = time.time()
        enhanced_solution = enhanced_scheduler.schedule(agents, tasks)
        enhanced_time = time.time() - start_time
        
        # Enhanced scheduler shouldn't be more than 50% slower
        overhead_ratio = enhanced_time / basic_time if basic_time > 0 else 1.0
        assert overhead_ratio < 1.5, f"Enhanced scheduler overhead too high: {overhead_ratio:.2f}x"
        
        # Both should produce valid solutions
        assert len(basic_solution.assignments) > 0
        assert len(enhanced_solution.assignments) > 0
        
        print(f"Basic: {basic_time:.3f}s, Enhanced: {enhanced_time:.3f}s, Overhead: {overhead_ratio:.2f}x")
    
    @pytest.mark.performance
    def test_concurrent_scheduling_performance(self):
        """Test performance under concurrent load."""
        scheduler = EnhancedQuantumScheduler(backend="classical")
        agents, tasks = self.generate_test_data(30, 60)
        
        def schedule_task():
            start_time = time.time()
            solution = scheduler.schedule(agents, tasks)
            return time.time() - start_time, len(solution.assignments)
        
        # Run concurrent scheduling tasks
        num_threads = 5
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(schedule_task) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Check all tasks completed successfully
        assert len(results) == num_threads
        execution_times = [result[0] for result in results]
        assignments_counts = [result[1] for result in results]
        
        # All should have produced assignments
        assert all(count > 0 for count in assignments_counts)
        
        # Average execution time should be reasonable
        avg_time = statistics.mean(execution_times)
        assert avg_time < 5.0
        
        print(f"Concurrent ({num_threads} threads): Total {total_time:.3f}s, Avg {avg_time:.3f}s")
    
    @pytest.mark.performance
    def test_memory_usage_performance(self):
        """Test memory usage with large problems."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        scheduler = QuantumScheduler(backend="classical")
        
        # Process multiple problems
        for problem_size in [50, 100, 150]:
            agents, tasks = self.generate_test_data(problem_size, problem_size * 2)
            solution = scheduler.schedule(agents, tasks)
            assert len(solution.assignments) > 0
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for these tests)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
        
        print(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")


class TestLoadBalancerPerformance:
    """Test adaptive load balancer performance."""
    
    @pytest.mark.performance
    def test_load_balancer_backend_selection_performance(self):
        """Test performance of backend selection."""
        load_balancer = AdaptiveLoadBalancer(strategy=LoadBalancingStrategy.ADAPTIVE_PERFORMANCE)
        
        # Register multiple mock backends
        for i in range(10):
            mock_backend = Mock()
            mock_backend.solve = Mock(return_value=Mock(cost=1.0, assignments={}))
            load_balancer.register_backend(
                name=f"backend_{i}",
                backend=mock_backend,
                weight=1.0,
                is_quantum=i % 2 == 0
            )
        
        # Measure backend selection performance
        num_selections = 1000
        start_time = time.time()
        
        for i in range(num_selections):
            context = {"problem_size": 50 + (i % 100), "complexity": "medium"}
            backend_name, backend = load_balancer.select_backend(context)
            assert backend_name is not None
            assert backend is not None
        
        total_time = time.time() - start_time
        avg_time_per_selection = total_time / num_selections * 1000  # ms
        
        # Backend selection should be fast (< 1ms per selection)
        assert avg_time_per_selection < 1.0
        
        print(f"Backend selection: {avg_time_per_selection:.3f}ms per selection")
    
    @pytest.mark.performance
    def test_load_balancer_metrics_update_performance(self):
        """Test performance of metrics updates."""
        load_balancer = AdaptiveLoadBalancer()
        
        # Register a backend
        mock_backend = Mock()
        load_balancer.register_backend("test_backend", mock_backend)
        
        # Measure metrics update performance
        num_updates = 10000
        start_time = time.time()
        
        for i in range(num_updates):
            request_id = load_balancer.record_request_start("test_backend")
            load_balancer.record_request_completion(
                "test_backend", request_id, True, 0.1 + (i % 10) * 0.01
            )
        
        total_time = time.time() - start_time
        avg_time_per_update = total_time / num_updates * 1000000  # μs
        
        # Metrics updates should be very fast (< 100μs per update)
        assert avg_time_per_update < 100
        
        print(f"Metrics update: {avg_time_per_update:.1f}μs per update")


class TestDistributedSchedulerPerformance:
    """Test distributed scheduler performance."""
    
    @pytest.mark.performance
    def test_distributed_vs_single_scheduler_performance(self):
        """Compare distributed vs single scheduler performance."""
        # Create large problem
        agents, tasks = self.generate_test_data(80, 160)
        
        # Single scheduler
        single_scheduler = QuantumScheduler(backend="classical")
        start_time = time.time()
        single_solution = single_scheduler.schedule(agents, tasks)
        single_time = time.time() - start_time
        
        # Distributed scheduler
        distributed_scheduler = DistributedQuantumScheduler(
            max_workers=4,
            max_partition_size=40
        )
        
        # Register backends
        for i in range(4):
            distributed_scheduler.register_backend(
                name=f"backend_{i}",
                backend_factory=lambda: QuantumScheduler(backend="classical"),
                weight=1.0
            )
        
        start_time = time.time()
        distributed_solution = distributed_scheduler.schedule(agents, tasks)
        distributed_time = time.time() - start_time
        
        # Both should produce valid solutions
        assert len(single_solution.assignments) > 0
        assert len(distributed_solution.assignments) > 0
        
        # For large problems, distributed should be faster or comparable
        # (May not always be faster due to overhead with small test problems)
        speedup = single_time / distributed_time if distributed_time > 0 else 1.0
        
        print(f"Single: {single_time:.3f}s, Distributed: {distributed_time:.3f}s, Speedup: {speedup:.2f}x")
        
        # Distributed scheduler should not be significantly slower
        assert speedup > 0.5, f"Distributed scheduler too slow: {speedup:.2f}x speedup"
        
        distributed_scheduler.shutdown()
    
    @pytest.mark.performance
    def test_distributed_scheduler_scalability(self):
        """Test distributed scheduler scalability with different worker counts."""
        agents, tasks = self.generate_test_data(60, 120)
        
        results = {}
        
        for worker_count in [1, 2, 4]:
            distributed_scheduler = DistributedQuantumScheduler(
                max_workers=worker_count,
                max_partition_size=30
            )
            
            # Register backends
            for i in range(worker_count):
                distributed_scheduler.register_backend(
                    name=f"backend_{i}",
                    backend_factory=lambda: QuantumScheduler(backend="classical"),
                    weight=1.0
                )
            
            # Measure performance
            start_time = time.time()
            solution = distributed_scheduler.schedule(agents, tasks)
            execution_time = time.time() - start_time
            
            results[worker_count] = {
                'time': execution_time,
                'assignments': len(solution.assignments)
            }
            
            distributed_scheduler.shutdown()
        
        # Check results
        for worker_count, result in results.items():
            assert result['assignments'] > 0
            print(f"Workers: {worker_count}, Time: {result['time']:.3f}s, Assignments: {result['assignments']}")
        
        # Generally, more workers should not significantly increase time
        # (though overhead might make this not always true for small problems)
        assert results[4]['time'] < results[1]['time'] * 2.0


class TestCachingPerformance:
    """Test caching performance impact."""
    
    @pytest.mark.performance
    def test_caching_performance_benefit(self):
        """Test performance benefit of solution caching."""
        agents, tasks = self.generate_test_data(40, 80)
        
        # Scheduler with caching enabled
        cached_scheduler = QuantumScheduler(backend="classical", enable_caching=True)
        
        # First solve (cache miss)
        start_time = time.time()
        solution1 = cached_scheduler.schedule(agents, tasks)
        first_time = time.time() - start_time
        
        # Second solve with same input (cache hit)
        start_time = time.time()
        solution2 = cached_scheduler.schedule(agents, tasks)
        second_time = time.time() - start_time
        
        # Cache hit should be significantly faster
        speedup = first_time / second_time if second_time > 0 else float('inf')
        assert speedup > 5.0, f"Cache speedup too low: {speedup:.2f}x"
        
        # Solutions should be equivalent
        assert len(solution1.assignments) == len(solution2.assignments)
        assert solution1.cost == solution2.cost
        
        print(f"Cache performance: {first_time:.3f}s → {second_time:.3f}s ({speedup:.1f}x speedup)")
    
    @pytest.mark.performance
    def test_cache_overhead_performance(self):
        """Test performance overhead of caching system."""
        agents, tasks = self.generate_test_data(30, 60)
        
        # Scheduler without caching
        no_cache_scheduler = QuantumScheduler(backend="classical", enable_caching=False)
        
        # Scheduler with caching
        cached_scheduler = QuantumScheduler(backend="classical", enable_caching=True)
        
        # Measure without cache (multiple runs for average)
        no_cache_times = []
        for _ in range(5):
            start_time = time.time()
            solution = no_cache_scheduler.schedule(agents, tasks)
            no_cache_times.append(time.time() - start_time)
            assert len(solution.assignments) > 0
        
        # Measure with cache (first run, cache miss)
        cached_times = []
        for i in range(5):
            # Use slightly different task priorities to avoid cache hits
            modified_tasks = [
                Task(id=task.id, required_skills=task.required_skills, 
                     duration=task.duration, priority=task.priority + i * 0.1)
                for task in tasks
            ]
            start_time = time.time()
            solution = cached_scheduler.schedule(agents, modified_tasks)
            cached_times.append(time.time() - start_time)
            assert len(solution.assignments) > 0
        
        avg_no_cache = statistics.mean(no_cache_times)
        avg_cached = statistics.mean(cached_times)
        overhead = avg_cached / avg_no_cache if avg_no_cache > 0 else 1.0
        
        # Caching overhead should be minimal (< 20%)
        assert overhead < 1.2, f"Cache overhead too high: {overhead:.2f}x"
        
        print(f"Cache overhead: {avg_no_cache:.3f}s → {avg_cached:.3f}s ({overhead:.2f}x)")


class TestScalabilityBenchmarks:
    """Test system scalability with increasing problem sizes."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scheduler_scalability_curve(self):
        """Test how scheduler performance scales with problem size."""
        scheduler = QuantumScheduler(backend="classical")
        
        problem_sizes = [10, 25, 50, 75, 100]
        results = []
        
        for size in problem_sizes:
            agents, tasks = self.generate_test_data(size, size * 2)
            
            # Measure multiple runs for stability
            times = []
            for run in range(3):
                start_time = time.time()
                solution = scheduler.schedule(agents, tasks)
                execution_time = time.time() - start_time
                times.append(execution_time)
                assert len(solution.assignments) > 0
            
            avg_time = statistics.mean(times)
            results.append((size, avg_time))
            print(f"Size {size}: {avg_time:.3f}s avg")
        
        # Check that growth is reasonable (not exponential)
        # Time should grow roughly linearly or quadratically, not exponentially
        for i in range(1, len(results)):
            prev_size, prev_time = results[i-1]
            curr_size, curr_time = results[i]
            
            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time if prev_time > 0 else 1.0
            
            # Time growth should be reasonable relative to size growth
            # Allow up to quadratic growth (time_ratio <= size_ratio^2)
            max_acceptable_ratio = size_ratio ** 2
            assert time_ratio <= max_acceptable_ratio * 1.5, \
                f"Performance degradation too steep: {time_ratio:.2f}x for {size_ratio:.2f}x size increase"
    
    @pytest.mark.performance
    def test_memory_scalability(self):
        """Test memory usage scaling with problem size."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        scheduler = QuantumScheduler(backend="classical")
        
        memory_usage = []
        problem_sizes = [20, 40, 60, 80]
        
        for size in problem_sizes:
            # Force garbage collection before measurement
            import gc
            gc.collect()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            agents, tasks = self.generate_test_data(size, size * 2)
            solution = scheduler.schedule(agents, tasks)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            memory_usage.append((size, memory_increase))
            assert len(solution.assignments) > 0
            
            print(f"Size {size}: +{memory_increase:.1f}MB")
        
        # Memory usage should scale reasonably (not exponentially)
        for i in range(1, len(memory_usage)):
            prev_size, prev_mem = memory_usage[i-1]
            curr_size, curr_mem = memory_usage[i]
            
            if prev_mem > 0:
                size_ratio = curr_size / prev_size
                mem_ratio = curr_mem / prev_mem
                
                # Memory should not grow faster than quadratically with problem size
                max_acceptable_ratio = size_ratio ** 2
                assert mem_ratio <= max_acceptable_ratio * 2.0, \
                    f"Memory growth too steep: {mem_ratio:.2f}x for {size_ratio:.2f}x size increase"