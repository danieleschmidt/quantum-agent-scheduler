"""Caching and optimization utilities for quantum scheduler."""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import threading
from functools import lru_cache

from ..core.models import Agent, Task, Solution, SchedulingProblem

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    solution: Solution
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    problem_hash: str = ""
    
    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp


class SolutionCache:
    """Intelligent caching system for scheduling solutions."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        """Initialize solution cache.
        
        Args:
            max_size: Maximum number of cached solutions
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0
        }
        
        logger.info(f"Initialized solution cache with max_size={max_size}, ttl={ttl_seconds}s")
    
    def _hash_problem(self, problem: SchedulingProblem) -> str:
        """Generate a hash for the scheduling problem.
        
        Args:
            problem: Scheduling problem to hash
            
        Returns:
            Hash string
        """
        # Create a deterministic representation
        problem_data = {
            "agents": [
                {
                    "id": agent.id,
                    "skills": sorted(agent.skills),
                    "capacity": agent.capacity
                }
                for agent in sorted(problem.agents, key=lambda a: a.id)
            ],
            "tasks": [
                {
                    "id": task.id,
                    "required_skills": sorted(task.required_skills),
                    "duration": task.duration,
                    "priority": task.priority,
                    "dependencies": sorted(task.dependencies)
                }
                for task in sorted(problem.tasks, key=lambda t: t.id)
            ],
            "constraints": dict(sorted(problem.constraints.items())),
            "optimization_target": problem.optimization_target
        }
        
        # Generate hash
        problem_json = json.dumps(problem_data, sort_keys=True)
        return hashlib.sha256(problem_json.encode()).hexdigest()
    
    def get(self, problem: SchedulingProblem) -> Optional[Solution]:
        """Get cached solution for problem.
        
        Args:
            problem: Scheduling problem
            
        Returns:
            Cached solution if available, None otherwise
        """
        problem_hash = self._hash_problem(problem)
        
        with self._lock:
            if problem_hash not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[problem_hash]
            current_time = time.time()
            
            # Check TTL
            if current_time - entry.timestamp > self.ttl_seconds:
                del self._cache[problem_hash]
                self._stats["invalidations"] += 1
                self._stats["misses"] += 1
                logger.debug(f"Cache entry expired for problem {problem_hash[:8]}")
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = current_time
            self._stats["hits"] += 1
            
            logger.debug(f"Cache hit for problem {problem_hash[:8]} (accessed {entry.access_count} times)")
            return entry.solution
    
    def put(self, problem: SchedulingProblem, solution: Solution) -> None:
        """Cache a solution for the given problem.
        
        Args:
            problem: Scheduling problem
            solution: Solution to cache
        """
        problem_hash = self._hash_problem(problem)
        current_time = time.time()
        
        with self._lock:
            # Check if cache is full and evict if necessary
            if len(self._cache) >= self.max_size and problem_hash not in self._cache:
                self._evict_lru()
            
            # Create cache entry
            entry = CacheEntry(
                solution=solution,
                timestamp=current_time,
                problem_hash=problem_hash
            )
            
            self._cache[problem_hash] = entry
            logger.debug(f"Cached solution for problem {problem_hash[:8]}")
    
    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]
        self._stats["evictions"] += 1
        
        logger.debug(f"Evicted LRU cache entry {lru_key[:8]}")
    
    def invalidate(self, problem: SchedulingProblem = None) -> None:
        """Invalidate cache entries.
        
        Args:
            problem: Specific problem to invalidate, or None to clear all
        """
        with self._lock:
            if problem is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._stats["invalidations"] += count
                logger.info(f"Invalidated all {count} cache entries")
            else:
                # Clear specific problem
                problem_hash = self._hash_problem(problem)
                if problem_hash in self._cache:
                    del self._cache[problem_hash]
                    self._stats["invalidations"] += 1
                    logger.debug(f"Invalidated cache entry for problem {problem_hash[:8]}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "invalidations": self._stats["invalidations"],
                "utilization": len(self._cache) / self.max_size
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            self._stats["invalidations"] += len(expired_keys)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


class ProblemOptimizer:
    """Optimizes scheduling problems for better performance."""
    
    @staticmethod
    def preprocess_problem(problem: SchedulingProblem) -> SchedulingProblem:
        """Preprocess problem for optimization.
        
        Args:
            problem: Original scheduling problem
            
        Returns:
            Optimized scheduling problem
        """
        # Sort agents by capacity (descending) for better greedy performance
        sorted_agents = sorted(problem.agents, key=lambda a: a.capacity, reverse=True)
        
        # Sort tasks by priority (descending) and duration (ascending) for better assignment
        sorted_tasks = sorted(
            problem.tasks, 
            key=lambda t: (-t.priority, t.duration)
        )
        
        # Create optimized problem
        optimized_problem = SchedulingProblem(
            agents=sorted_agents,
            tasks=sorted_tasks,
            constraints=problem.constraints,
            optimization_target=problem.optimization_target
        )
        
        return optimized_problem
    
    @staticmethod
    def estimate_complexity(problem: SchedulingProblem) -> Dict[str, Any]:
        """Estimate problem complexity metrics.
        
        Args:
            problem: Scheduling problem
            
        Returns:
            Complexity metrics
        """
        num_agents = len(problem.agents)
        num_tasks = len(problem.tasks)
        
        # Calculate skill diversity
        all_skills = set()
        for agent in problem.agents:
            all_skills.update(agent.skills)
        
        required_skills = set()
        for task in problem.tasks:
            required_skills.update(task.required_skills)
        
        skill_coverage = len(required_skills & all_skills) / len(required_skills) if required_skills else 1.0
        
        # Calculate capacity utilization
        total_capacity = sum(agent.capacity for agent in problem.agents)
        total_workload = sum(task.duration for task in problem.tasks)
        capacity_utilization = total_workload / total_capacity if total_capacity > 0 else float('inf')
        
        # Estimate search space size
        search_space = num_agents ** num_tasks  # Rough approximation
        
        return {
            "num_agents": num_agents,
            "num_tasks": num_tasks,
            "skill_coverage": skill_coverage,
            "capacity_utilization": capacity_utilization,
            "search_space_log": len(str(search_space)),  # Avoid huge numbers
            "complexity_score": (num_agents * num_tasks) / max(1, skill_coverage),
            "is_feasible": capacity_utilization <= 1.0
        }
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_skill_compatibility_matrix(
        agent_skills_tuple: Tuple[Tuple[str, ...], ...], 
        task_skills_tuple: Tuple[Tuple[str, ...], ...]
    ) -> Tuple[Tuple[bool, ...], ...]:
        """Get cached skill compatibility matrix.
        
        Args:
            agent_skills_tuple: Tuple of agent skill tuples
            task_skills_tuple: Tuple of task skill tuples
            
        Returns:
            Compatibility matrix as nested tuples
        """
        matrix = []
        
        for agent_skills in agent_skills_tuple:
            agent_row = []
            agent_skill_set = set(agent_skills)
            
            for task_skills in task_skills_tuple:
                task_skill_set = set(task_skills)
                compatible = task_skill_set.issubset(agent_skill_set)
                agent_row.append(compatible)
            
            matrix.append(tuple(agent_row))
        
        return tuple(matrix)
    
    @staticmethod
    def optimize_for_backend(problem: SchedulingProblem, backend_type: str) -> SchedulingProblem:
        """Optimize problem for specific backend type.
        
        Args:
            problem: Original problem
            backend_type: Backend type ("classical", "quantum", "hybrid")
            
        Returns:
            Backend-optimized problem
        """
        if backend_type == "quantum":
            # For quantum backends, reduce problem size if too large
            max_qubits = 50  # Typical quantum hardware limit
            total_variables = len(problem.agents) * len(problem.tasks)
            
            if total_variables > max_qubits:
                # Prioritize highest priority tasks and highest capacity agents
                top_tasks = sorted(problem.tasks, key=lambda t: t.priority, reverse=True)[:max_qubits//2]
                top_agents = sorted(problem.agents, key=lambda a: a.capacity, reverse=True)[:max_qubits//2]
                
                logger.warning(f"Reduced problem size for quantum backend: {len(top_tasks)} tasks, {len(top_agents)} agents")
                
                return SchedulingProblem(
                    agents=top_agents,
                    tasks=top_tasks,
                    constraints=problem.constraints,
                    optimization_target=problem.optimization_target
                )
        
        return problem


# Global cache instance
_solution_cache = SolutionCache()


def get_solution_cache() -> SolutionCache:
    """Get the global solution cache instance."""
    return _solution_cache


def configure_cache(max_size: int = 1000, ttl_seconds: float = 3600) -> None:
    """Configure global solution cache.
    
    Args:
        max_size: Maximum number of cached solutions
        ttl_seconds: Time-to-live for cache entries
    """
    global _solution_cache
    _solution_cache = SolutionCache(max_size, ttl_seconds)
    logger.info(f"Configured solution cache with max_size={max_size}, ttl={ttl_seconds}s")