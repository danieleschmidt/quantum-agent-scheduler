"""Distributed quantum scheduler for large-scale multi-agent systems."""

import asyncio
import logging
import time
import uuid
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock, RLock
from collections import defaultdict, deque
from datetime import datetime, timedelta

from ..core.models import Agent, Task, Solution, SchedulingProblem
from ..core.exceptions import SolverError, ValidationError
from ..core.scheduler import QuantumScheduler
from .adaptive_load_balancer import AdaptiveLoadBalancer, LoadBalancingStrategy
from ..monitoring import get_metrics_collector
from ..security import SecuritySanitizer

logger = logging.getLogger(__name__)


@dataclass
class DistributedTask:
    """Task for distributed processing."""
    task_id: str
    partition_id: str
    agents: List[Agent]
    tasks: List[Task]
    constraints: Dict[str, Any]
    priority: float = 1.0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class PartitionResult:
    """Result from processing a partition."""
    partition_id: str
    solution: Optional[Solution]
    processing_time: float
    backend_used: str
    success: bool
    error: Optional[str] = None
    quantum_advantage: Optional[float] = None


class ProblemPartitioner:
    """Intelligent problem partitioner for distributed processing."""
    
    def __init__(
        self,
        max_partition_size: int = 50,
        min_partition_size: int = 5,
        overlap_ratio: float = 0.1,
        partitioning_strategy: str = "balanced"
    ):
        """Initialize problem partitioner.
        
        Args:
            max_partition_size: Maximum number of tasks per partition
            min_partition_size: Minimum number of tasks per partition
            overlap_ratio: Overlap ratio between partitions for coordination
            partitioning_strategy: Strategy for partitioning ("balanced", "skill_based", "priority")
        """
        self.max_partition_size = max_partition_size
        self.min_partition_size = min_partition_size
        self.overlap_ratio = overlap_ratio
        self.partitioning_strategy = partitioning_strategy
        
    def partition_problem(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Dict[str, Any]
    ) -> List[DistributedTask]:
        """Partition a large problem into smaller subproblems."""
        if len(tasks) <= self.max_partition_size:
            # Single partition
            return [DistributedTask(
                task_id=str(uuid.uuid4()),
                partition_id="single",
                agents=agents,
                tasks=tasks,
                constraints=constraints
            )]
        
        # Multi-partition approach
        if self.partitioning_strategy == "balanced":
            return self._balanced_partition(agents, tasks, constraints)
        elif self.partitioning_strategy == "skill_based":
            return self._skill_based_partition(agents, tasks, constraints)
        elif self.partitioning_strategy == "priority":
            return self._priority_based_partition(agents, tasks, constraints)
        else:
            return self._balanced_partition(agents, tasks, constraints)
    
    def _balanced_partition(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Dict[str, Any]
    ) -> List[DistributedTask]:
        """Create balanced partitions by task count."""
        partitions = []
        num_partitions = max(1, len(tasks) // self.max_partition_size)
        tasks_per_partition = len(tasks) // num_partitions
        
        for i in range(num_partitions):
            start_idx = i * tasks_per_partition
            end_idx = start_idx + tasks_per_partition if i < num_partitions - 1 else len(tasks)
            
            partition_tasks = tasks[start_idx:end_idx]
            
            # Add overlap with next partition
            if i < num_partitions - 1 and self.overlap_ratio > 0:
                overlap_size = int(len(partition_tasks) * self.overlap_ratio)
                if end_idx + overlap_size <= len(tasks):
                    partition_tasks.extend(tasks[end_idx:end_idx + overlap_size])
            
            # Select relevant agents for this partition
            partition_agents = self._select_relevant_agents(agents, partition_tasks)
            
            partitions.append(DistributedTask(
                task_id=str(uuid.uuid4()),
                partition_id=f"balanced_{i}",
                agents=partition_agents,
                tasks=partition_tasks,
                constraints=constraints.copy(),
                priority=1.0
            ))
        
        return partitions
    
    def _skill_based_partition(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Dict[str, Any]
    ) -> List[DistributedTask]:
        """Create partitions based on skill requirements."""
        # Group tasks by required skills
        skill_groups = defaultdict(list)
        for task in tasks:
            skill_key = tuple(sorted(task.required_skills))
            skill_groups[skill_key].append(task)
        
        partitions = []
        partition_id = 0
        
        for skill_key, skill_tasks in skill_groups.items():
            # Further partition if too many tasks with same skills
            if len(skill_tasks) > self.max_partition_size:
                # Split by priority and duration
                skill_tasks.sort(key=lambda t: (-t.priority, t.duration))
                
                for i in range(0, len(skill_tasks), self.max_partition_size):
                    batch_tasks = skill_tasks[i:i + self.max_partition_size]
                    relevant_agents = self._select_relevant_agents(agents, batch_tasks)
                    
                    partitions.append(DistributedTask(
                        task_id=str(uuid.uuid4()),
                        partition_id=f"skill_{partition_id}",
                        agents=relevant_agents,
                        tasks=batch_tasks,
                        constraints=constraints.copy(),
                        priority=sum(t.priority for t in batch_tasks) / len(batch_tasks)
                    ))
                    partition_id += 1
            else:
                relevant_agents = self._select_relevant_agents(agents, skill_tasks)
                
                partitions.append(DistributedTask(
                    task_id=str(uuid.uuid4()),
                    partition_id=f"skill_{partition_id}",
                    agents=relevant_agents,
                    tasks=skill_tasks,
                    constraints=constraints.copy(),
                    priority=sum(t.priority for t in skill_tasks) / len(skill_tasks)
                ))
                partition_id += 1
        
        return partitions
    
    def _priority_based_partition(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Dict[str, Any]
    ) -> List[DistributedTask]:
        """Create partitions based on task priorities."""
        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)
        
        partitions = []
        for i in range(0, len(sorted_tasks), self.max_partition_size):
            partition_tasks = sorted_tasks[i:i + self.max_partition_size]
            relevant_agents = self._select_relevant_agents(agents, partition_tasks)
            
            avg_priority = sum(t.priority for t in partition_tasks) / len(partition_tasks)
            
            partitions.append(DistributedTask(
                task_id=str(uuid.uuid4()),
                partition_id=f"priority_{i // self.max_partition_size}",
                agents=relevant_agents,
                tasks=partition_tasks,
                constraints=constraints.copy(),
                priority=avg_priority
            ))
        
        return partitions
    
    def _select_relevant_agents(self, agents: List[Agent], tasks: List[Task]) -> List[Agent]:
        """Select agents relevant for the given tasks."""
        required_skills = set()
        for task in tasks:
            required_skills.update(task.required_skills)
        
        relevant_agents = []
        for agent in agents:
            agent_skills = set(agent.skills)
            if agent_skills.intersection(required_skills):
                relevant_agents.append(agent)
        
        # Ensure we have at least some agents
        if not relevant_agents and agents:
            relevant_agents = agents[:min(len(agents), 10)]  # Take first 10 as fallback
        
        return relevant_agents


class ResultAggregator:
    """Aggregates results from distributed processing."""
    
    def __init__(self, merge_strategy: str = "optimal"):
        """Initialize result aggregator.
        
        Args:
            merge_strategy: Strategy for merging results ("optimal", "greedy", "weighted")
        """
        self.merge_strategy = merge_strategy
    
    def aggregate_results(
        self, 
        partition_results: List[PartitionResult],
        original_agents: List[Agent],
        original_tasks: List[Task]
    ) -> Solution:
        """Aggregate partition results into final solution."""
        successful_results = [r for r in partition_results if r.success and r.solution]
        
        if not successful_results:
            raise SolverError("No successful partition results to aggregate")
        
        if self.merge_strategy == "optimal":
            return self._optimal_merge(successful_results, original_agents, original_tasks)
        elif self.merge_strategy == "greedy":
            return self._greedy_merge(successful_results, original_agents, original_tasks)
        elif self.merge_strategy == "weighted":
            return self._weighted_merge(successful_results, original_agents, original_tasks)
        else:
            return self._optimal_merge(successful_results, original_agents, original_tasks)
    
    def _optimal_merge(
        self, 
        results: List[PartitionResult],
        original_agents: List[Agent],
        original_tasks: List[Task]
    ) -> Solution:
        """Optimal merge considering global constraints."""
        merged_assignments = {}
        total_cost = 0.0
        used_agents = set()
        
        # Sort results by cost efficiency
        results.sort(key=lambda r: r.solution.cost / len(r.solution.assignments))
        
        for result in results:
            for task_id, agent_id in result.solution.assignments.items():
                if task_id not in merged_assignments and agent_id not in used_agents:
                    # Check if this assignment is still valid
                    if self._validate_assignment(task_id, agent_id, original_tasks, original_agents):
                        merged_assignments[task_id] = agent_id
                        used_agents.add(agent_id)
                        total_cost += result.solution.cost / len(result.solution.assignments)
        
        # Handle unassigned tasks with remaining agents
        unassigned_tasks = [t for t in original_tasks if t.id not in merged_assignments]
        available_agents = [a for a in original_agents if a.id not in used_agents]
        
        if unassigned_tasks and available_agents:
            # Simple greedy assignment for remaining tasks
            for task in unassigned_tasks:
                suitable_agent = self._find_suitable_agent(task, available_agents)
                if suitable_agent:
                    merged_assignments[task.id] = suitable_agent.id
                    available_agents.remove(suitable_agent)
                    total_cost += task.duration * 0.1  # Penalty for suboptimal assignment
        
        return Solution(
            assignments=merged_assignments,
            cost=total_cost,
            solver_type="distributed_optimal",
            total_assignments=len(merged_assignments),
            execution_time=sum(r.processing_time for r in results),
            utilization_ratio=len(merged_assignments) / len(original_tasks) if original_tasks else 0.0
        )
    
    def _greedy_merge(
        self, 
        results: List[PartitionResult],
        original_agents: List[Agent],
        original_tasks: List[Task]
    ) -> Solution:
        """Greedy merge taking first valid assignments."""
        merged_assignments = {}
        total_cost = 0.0
        used_agents = set()
        
        for result in results:
            for task_id, agent_id in result.solution.assignments.items():
                if task_id not in merged_assignments and agent_id not in used_agents:
                    merged_assignments[task_id] = agent_id
                    used_agents.add(agent_id)
                    total_cost += result.solution.cost / len(result.solution.assignments)
        
        return Solution(
            assignments=merged_assignments,
            cost=total_cost,
            solver_type="distributed_greedy",
            total_assignments=len(merged_assignments),
            execution_time=sum(r.processing_time for r in results),
            utilization_ratio=len(merged_assignments) / len(original_tasks) if original_tasks else 0.0
        )
    
    def _weighted_merge(
        self, 
        results: List[PartitionResult],
        original_agents: List[Agent],
        original_tasks: List[Task]
    ) -> Solution:
        """Weighted merge considering partition quality and quantum advantage."""
        # Calculate weights based on solution quality and quantum advantage
        weights = {}
        for result in results:
            if not result.solution:
                continue
            
            base_weight = 1.0 / (1.0 + result.solution.cost)  # Lower cost = higher weight
            quantum_bonus = (result.quantum_advantage or 0.0) * 0.5
            time_penalty = min(result.processing_time / 60.0, 1.0)  # Penalty for slow processing
            
            weights[result.partition_id] = base_weight + quantum_bonus - time_penalty
        
        merged_assignments = {}
        total_cost = 0.0
        used_agents = set()
        
        # Sort results by weight (highest first)
        sorted_results = sorted(results, key=lambda r: weights.get(r.partition_id, 0.0), reverse=True)
        
        for result in sorted_results:
            if not result.solution:
                continue
            
            weight = weights.get(result.partition_id, 1.0)
            
            for task_id, agent_id in result.solution.assignments.items():
                if task_id not in merged_assignments and agent_id not in used_agents:
                    merged_assignments[task_id] = agent_id
                    used_agents.add(agent_id)
                    total_cost += (result.solution.cost / len(result.solution.assignments)) / weight
        
        return Solution(
            assignments=merged_assignments,
            cost=total_cost,
            solver_type="distributed_weighted",
            total_assignments=len(merged_assignments),
            execution_time=sum(r.processing_time for r in results),
            utilization_ratio=len(merged_assignments) / len(original_tasks) if original_tasks else 0.0
        )
    
    def _validate_assignment(
        self, 
        task_id: str, 
        agent_id: str, 
        tasks: List[Task], 
        agents: List[Agent]
    ) -> bool:
        """Validate if an assignment is valid in global context."""
        task = next((t for t in tasks if t.id == task_id), None)
        agent = next((a for a in agents if a.id == agent_id), None)
        
        if not task or not agent:
            return False
        
        # Check skill requirements
        required_skills = set(task.required_skills)
        agent_skills = set(agent.skills)
        
        return required_skills.issubset(agent_skills)
    
    def _find_suitable_agent(self, task: Task, available_agents: List[Agent]) -> Optional[Agent]:
        """Find the most suitable agent for a task."""
        suitable_agents = []
        
        for agent in available_agents:
            agent_skills = set(agent.skills)
            required_skills = set(task.required_skills)
            
            if required_skills.issubset(agent_skills):
                suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Return agent with highest capacity and most matching skills
        return max(suitable_agents, key=lambda a: (a.capacity, len(set(a.skills).intersection(set(task.required_skills)))))


class DistributedQuantumScheduler:
    """Distributed quantum scheduler for large-scale problems."""
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_PERFORMANCE,
        partitioning_strategy: str = "skill_based",
        merge_strategy: str = "optimal",
        max_partition_size: int = 50,
        enable_coordination: bool = True,
        coordination_interval: float = 30.0
    ):
        """Initialize distributed quantum scheduler.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
            load_balancing_strategy: Strategy for backend load balancing
            partitioning_strategy: Strategy for problem partitioning
            merge_strategy: Strategy for result merging
            max_partition_size: Maximum size of each partition
            enable_coordination: Enable coordination between partitions
            coordination_interval: Interval for coordination checks
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.partitioning_strategy = partitioning_strategy
        self.merge_strategy = merge_strategy
        self.max_partition_size = max_partition_size
        self.enable_coordination = enable_coordination
        self.coordination_interval = coordination_interval
        
        # Initialize components
        self.partitioner = ProblemPartitioner(
            max_partition_size=max_partition_size,
            partitioning_strategy=partitioning_strategy
        )
        self.aggregator = ResultAggregator(merge_strategy=merge_strategy)
        self.load_balancer = AdaptiveLoadBalancer(strategy=load_balancing_strategy)
        
        # Worker management
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Scheduler instances for workers (one per worker)
        self.schedulers = {}
        self._scheduler_lock = RLock()
        
        # Coordination state
        self.active_partitions = {}
        self.coordination_data = {}
        self._coordination_lock = Lock()
        
        # Metrics
        self.metrics = {
            'total_problems': 0,
            'total_partitions': 0,
            'avg_partition_time': 0.0,
            'distribution_efficiency': 0.0,
            'coordination_overhead': 0.0
        }
        
        logger.info(f"Initialized DistributedQuantumScheduler with {max_workers} workers")
    
    def register_backend(
        self,
        name: str,
        backend_factory: Callable[[], Any],
        weight: float = 1.0,
        is_quantum: bool = False
    ):
        """Register a backend factory with the distributed scheduler."""
        # Create a scheduler instance for this backend
        scheduler = QuantumScheduler(backend=backend_factory())
        
        with self._scheduler_lock:
            self.schedulers[name] = scheduler
        
        # Register with load balancer (using a dummy backend for metrics)
        self.load_balancer.register_backend(
            name=name,
            backend=scheduler,
            weight=weight,
            is_quantum=is_quantum
        )
        
        logger.info(f"Registered distributed backend: {name} (quantum: {is_quantum})")
    
    def schedule(
        self,
        agents: List[Agent],
        tasks: List[Task],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """Schedule tasks using distributed processing."""
        start_time = time.time()
        self.metrics['total_problems'] += 1
        
        constraints = constraints or {}
        
        try:
            # Check if problem is small enough for single scheduler
            if len(tasks) <= self.max_partition_size and len(agents) <= 20:
                return self._single_scheduler_solve(agents, tasks, constraints)
            
            # Partition the problem
            partitions = self.partitioner.partition_problem(agents, tasks, constraints)
            self.metrics['total_partitions'] += len(partitions)
            
            logger.info(f"Partitioned problem into {len(partitions)} partitions")
            
            # Process partitions in parallel
            partition_results = self._process_partitions_parallel(partitions)
            
            # Aggregate results
            final_solution = self.aggregator.aggregate_results(
                partition_results, agents, tasks
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            final_solution.execution_time = execution_time
            self._update_metrics(execution_time, len(partitions), partition_results)
            
            logger.info(f"Distributed scheduling completed in {execution_time:.2f}s")
            return final_solution
            
        except Exception as e:
            logger.error(f"Distributed scheduling failed: {e}")
            # Fallback to single scheduler
            try:
                return self._single_scheduler_solve(agents, tasks, constraints)
            except Exception as fallback_error:
                raise SolverError(f"Both distributed and fallback scheduling failed: {e}, {fallback_error}")
    
    def _single_scheduler_solve(
        self,
        agents: List[Agent],
        tasks: List[Task],
        constraints: Dict[str, Any]
    ) -> Solution:
        """Solve using a single scheduler instance."""
        backend_name, scheduler = self.load_balancer.select_backend({
            'problem_size': len(tasks),
            'complexity': 'low' if len(tasks) < 20 else 'medium'
        })
        
        request_id = self.load_balancer.record_request_start(backend_name)
        start_time = time.time()
        
        try:
            solution = scheduler.schedule(agents, tasks, constraints)
            processing_time = time.time() - start_time
            
            self.load_balancer.record_request_completion(
                backend_name, request_id, True, processing_time
            )
            
            solution.solver_type = f"single_{backend_name}"
            return solution
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.load_balancer.record_request_completion(
                backend_name, request_id, False, processing_time
            )
            raise
    
    def _process_partitions_parallel(self, partitions: List[DistributedTask]) -> List[PartitionResult]:
        """Process partitions in parallel."""
        # Sort partitions by priority
        sorted_partitions = sorted(partitions, key=lambda p: p.priority, reverse=True)
        
        # Submit tasks to executor
        future_to_partition = {}
        
        for partition in sorted_partitions:
            future = self.executor.submit(self._process_single_partition, partition)
            future_to_partition[future] = partition
        
        # Collect results
        results = []
        coordination_start = time.time()
        
        for future in as_completed(future_to_partition):
            partition = future_to_partition[future]
            
            try:
                result = future.result()
                results.append(result)
                
                # Coordination between partitions
                if self.enable_coordination:
                    self._update_coordination_data(result)
                    
            except Exception as e:
                logger.error(f"Partition {partition.partition_id} failed: {e}")
                # Create failed result
                results.append(PartitionResult(
                    partition_id=partition.partition_id,
                    solution=None,
                    processing_time=0.0,
                    backend_used="unknown",
                    success=False,
                    error=str(e)
                ))
        
        coordination_time = time.time() - coordination_start
        self.metrics['coordination_overhead'] = coordination_time
        
        return results
    
    def _process_single_partition(self, partition: DistributedTask) -> PartitionResult:
        """Process a single partition."""
        start_time = time.time()
        
        try:
            # Select backend for this partition
            context = {
                'problem_size': len(partition.tasks),
                'complexity': self._estimate_complexity(partition.tasks),
                'partition_id': partition.partition_id
            }
            
            backend_name, scheduler = self.load_balancer.select_backend(context)
            request_id = self.load_balancer.record_request_start(backend_name)
            
            # Solve partition
            solution = scheduler.schedule(
                partition.agents,
                partition.tasks,
                partition.constraints
            )
            
            processing_time = time.time() - start_time
            
            # Record success
            self.load_balancer.record_request_completion(
                backend_name, request_id, True, processing_time
            )
            
            return PartitionResult(
                partition_id=partition.partition_id,
                solution=solution,
                processing_time=processing_time,
                backend_used=backend_name,
                success=True,
                quantum_advantage=getattr(solution, 'quantum_advantage', None)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Partition {partition.partition_id} processing failed: {e}")
            
            return PartitionResult(
                partition_id=partition.partition_id,
                solution=None,
                processing_time=processing_time,
                backend_used="unknown",
                success=False,
                error=str(e)
            )
    
    def _estimate_complexity(self, tasks: List[Task]) -> str:
        """Estimate problem complexity for backend selection."""
        if len(tasks) < 10:
            return "low"
        elif len(tasks) < 50:
            return "medium"
        elif len(tasks) < 100:
            return "high"
        else:
            return "very_high"
    
    def _update_coordination_data(self, result: PartitionResult):
        """Update coordination data with partition result."""
        with self._coordination_lock:
            if result.success and result.solution:
                self.coordination_data[result.partition_id] = {
                    'assignments': result.solution.assignments,
                    'cost': result.solution.cost,
                    'backend': result.backend_used,
                    'timestamp': time.time()
                }
    
    def _update_metrics(
        self,
        execution_time: float,
        num_partitions: int,
        results: List[PartitionResult]
    ):
        """Update distributed scheduler metrics."""
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            avg_partition_time = sum(r.processing_time for r in successful_results) / len(successful_results)
            self.metrics['avg_partition_time'] = avg_partition_time
        
        # Distribution efficiency (speedup vs sequential)
        if num_partitions > 1 and successful_results:
            sequential_time = sum(r.processing_time for r in successful_results)
            speedup = sequential_time / execution_time if execution_time > 0 else 1.0
            self.metrics['distribution_efficiency'] = min(speedup / num_partitions, 1.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive distributed scheduler metrics."""
        base_metrics = self.metrics.copy()
        
        # Add load balancer metrics
        base_metrics['load_balancer'] = self.load_balancer.get_metrics()
        
        # Add backend recommendations
        base_metrics['recommendations'] = self.load_balancer.get_backend_recommendations()
        
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on distributed system."""
        health_status = {
            'overall': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check executor health
        try:
            # Submit a simple test task
            test_future = self.executor.submit(lambda: time.time())
            test_result = test_future.result(timeout=5.0)
            
            health_status['components']['executor'] = {
                'status': 'healthy',
                'type': 'process' if self.use_processes else 'thread',
                'max_workers': self.max_workers
            }
        except Exception as e:
            health_status['components']['executor'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall'] = 'degraded'
        
        # Check load balancer health
        lb_metrics = self.load_balancer.get_metrics()
        if lb_metrics['healthy_backends'] == 0:
            health_status['overall'] = 'critical'
            
        health_status['components']['load_balancer'] = {
            'status': 'healthy' if lb_metrics['healthy_backends'] > 0 else 'unhealthy',
            'healthy_backends': lb_metrics['healthy_backends'],
            'total_backends': lb_metrics['total_backends']
        }
        
        # Check individual schedulers
        scheduler_health = {}
        with self._scheduler_lock:
            for name, scheduler in self.schedulers.items():
                try:
                    if hasattr(scheduler, 'health_check'):
                        scheduler_health[name] = scheduler.health_check()
                    else:
                        scheduler_health[name] = {'status': 'unknown'}
                except Exception as e:
                    scheduler_health[name] = {'status': 'unhealthy', 'error': str(e)}
        
        health_status['components']['schedulers'] = scheduler_health
        
        return health_status
    
    def shutdown(self):
        """Shutdown the distributed scheduler."""
        logger.info("Shutting down distributed scheduler...")
        
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during executor shutdown: {e}")
        
        logger.info("Distributed scheduler shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


# Async version for high-performance scenarios
class AsyncDistributedQuantumScheduler(DistributedQuantumScheduler):
    """Asynchronous distributed quantum scheduler."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = None
    
    async def schedule_async(
        self,
        agents: List[Agent],
        tasks: List[Task],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """Asynchronous version of schedule method."""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        
        # Run synchronous schedule in thread pool
        return await self.loop.run_in_executor(
            None, self.schedule, agents, tasks, constraints
        )
    
    async def batch_schedule_async(
        self,
        problem_batch: List[Tuple[List[Agent], List[Task], Optional[Dict[str, Any]]]]
    ) -> List[Solution]:
        """Schedule multiple problems asynchronously."""
        tasks = [
            self.schedule_async(agents, tasks, constraints)
            for agents, tasks, constraints in problem_batch
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)