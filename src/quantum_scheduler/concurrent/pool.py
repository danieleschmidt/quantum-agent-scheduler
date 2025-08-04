"""Concurrent processing utilities for quantum scheduler."""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import threading
from queue import Queue, Empty

from ..core.models import Agent, Task, Solution, SchedulingProblem
from ..core.exceptions import SolverError, SolverTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a batch scheduling job."""
    job_id: str
    problems: List[SchedulingProblem]
    callback: Optional[Callable] = None
    priority: int = 1
    submitted_at: float = 0.0
    
    def __post_init__(self):
        if self.submitted_at == 0.0:
            self.submitted_at = time.time()


@dataclass
class JobResult:
    """Result of a batch job."""
    job_id: str
    solutions: List[Solution]
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


class SchedulerPool:
    """Thread/process pool for concurrent scheduling operations."""
    
    def __init__(
        self, 
        max_workers: int = None,
        use_processes: bool = False,
        timeout: float = 300.0
    ):
        """Initialize scheduler pool.
        
        Args:
            max_workers: Maximum number of workers (None for auto-detect)
            use_processes: Use process pool instead of thread pool
            timeout: Default timeout for operations
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.timeout = timeout
        
        # Initialize executor
        if use_processes:
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Job management
        self._active_jobs: Dict[str, asyncio.Future] = {}
        self._job_queue: Queue = Queue()
        self._results: Dict[str, JobResult] = {}
        self._lock = threading.Lock()
        
        logger.info(f"Initialized {'process' if use_processes else 'thread'} pool with {max_workers or 'auto'} workers")
    
    def submit_batch(
        self, 
        problems: List[SchedulingProblem], 
        scheduler_factory: Callable,
        job_id: str = None,
        priority: int = 1
    ) -> str:
        """Submit a batch of problems for concurrent processing.
        
        Args:
            problems: List of scheduling problems
            scheduler_factory: Function that creates scheduler instances
            job_id: Optional job ID (auto-generated if None)
            priority: Job priority (higher = more important)
            
        Returns:
            Job ID for tracking
        """
        if job_id is None:
            job_id = f"batch_{time.time()}_{len(problems)}"
        
        batch_job = BatchJob(
            job_id=job_id,
            problems=problems,
            priority=priority
        )
        
        # Submit to thread pool
        future = self._executor.submit(
            self._process_batch,
            batch_job,
            scheduler_factory
        )
        
        with self._lock:
            self._active_jobs[job_id] = future
        
        logger.info(f"Submitted batch job {job_id} with {len(problems)} problems")
        return job_id
    
    def submit_single(
        self,
        problem: SchedulingProblem,
        scheduler_factory: Callable,
        job_id: str = None
    ) -> str:
        """Submit a single problem for processing.
        
        Args:
            problem: Scheduling problem
            scheduler_factory: Function that creates scheduler instance
            job_id: Optional job ID
            
        Returns:
            Job ID for tracking
        """
        return self.submit_batch([problem], scheduler_factory, job_id)
    
    def get_result(self, job_id: str, timeout: float = None) -> Optional[JobResult]:
        """Get result for a job.
        
        Args:
            job_id: Job ID to check
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Job result if available, None if still running
        """
        with self._lock:
            # Check if result is already cached
            if job_id in self._results:
                return self._results[job_id]
            
            # Check if job is active
            if job_id not in self._active_jobs:
                return None
            
            future = self._active_jobs[job_id]
        
        # Wait for result with timeout
        try:
            result = future.result(timeout=timeout or self.timeout)
            
            with self._lock:
                self._results[job_id] = result
                if job_id in self._active_jobs:
                    del self._active_jobs[job_id]
            
            return result
            
        except TimeoutError:
            logger.warning(f"Job {job_id} timed out after {timeout or self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            error_result = JobResult(
                job_id=job_id,
                solutions=[],
                success=False,
                error=str(e)
            )
            
            with self._lock:
                self._results[job_id] = error_result
                if job_id in self._active_jobs:
                    del self._active_jobs[job_id]
            
            return error_result
    
    def wait_for_completion(self, job_ids: List[str], timeout: float = None) -> Dict[str, JobResult]:
        """Wait for multiple jobs to complete.
        
        Args:
            job_ids: List of job IDs to wait for
            timeout: Total timeout in seconds
            
        Returns:
            Dictionary mapping job IDs to results
        """
        start_time = time.time()
        results = {}
        
        for job_id in job_ids:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
            
            result = self.get_result(job_id, remaining_timeout)
            if result:
                results[job_id] = result
            else:
                logger.warning(f"Job {job_id} did not complete within timeout")
        
        return results
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled successfully
        """
        with self._lock:
            if job_id in self._active_jobs:
                future = self._active_jobs[job_id]
                cancelled = future.cancel()
                
                if cancelled:
                    del self._active_jobs[job_id]
                    logger.info(f"Cancelled job {job_id}")
                
                return cancelled
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status information.
        
        Returns:
            Status information
        """
        with self._lock:
            active_count = len(self._active_jobs)
            completed_count = len(self._results)
        
        return {
            "max_workers": self.max_workers,
            "executor_type": "process" if self.use_processes else "thread",
            "active_jobs": active_count,
            "completed_jobs": completed_count,
            "timeout": self.timeout
        }
    
    def _process_batch(self, batch_job: BatchJob, scheduler_factory: Callable) -> JobResult:
        """Process a batch of problems.
        
        Args:
            batch_job: Batch job to process
            scheduler_factory: Function to create scheduler instances
            
        Returns:
            Job result
        """
        start_time = time.time()
        solutions = []
        errors = []
        
        logger.info(f"Processing batch job {batch_job.job_id} with {len(batch_job.problems)} problems")
        
        try:
            # Process each problem
            for i, problem in enumerate(batch_job.problems):
                try:
                    # Create scheduler instance
                    scheduler = scheduler_factory()
                    
                    # Solve problem
                    solution = scheduler.schedule(
                        problem.agents,
                        problem.tasks,
                        problem.constraints
                    )
                    
                    solutions.append(solution)
                    
                except Exception as e:
                    logger.error(f"Problem {i} in batch {batch_job.job_id} failed: {e}")
                    errors.append(str(e))
                    
                    # Create empty solution for failed problem
                    solutions.append(Solution(
                        assignments={},
                        cost=float('inf'),
                        solver_type="failed"
                    ))
            
            execution_time = time.time() - start_time
            success = len(errors) == 0
            
            result = JobResult(
                job_id=batch_job.job_id,
                solutions=solutions,
                success=success,
                error="; ".join(errors) if errors else None,
                execution_time=execution_time
            )
            
            logger.info(f"Completed batch job {batch_job.job_id} in {execution_time:.3f}s ({'success' if success else 'partial'})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Batch job {batch_job.job_id} failed completely: {e}")
            
            return JobResult(
                job_id=batch_job.job_id,
                solutions=[],
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler pool.
        
        Args:
            wait: Wait for active jobs to complete
        """
        logger.info("Shutting down scheduler pool...")
        
        if wait:
            # Cancel all active jobs
            with self._lock:
                active_jobs = list(self._active_jobs.keys())
            
            for job_id in active_jobs:
                self.cancel_job(job_id)
        
        self._executor.shutdown(wait=wait)
        logger.info("Scheduler pool shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class AsyncScheduler:
    """Asynchronous wrapper for quantum scheduler."""
    
    def __init__(self, scheduler_factory: Callable):
        """Initialize async scheduler.
        
        Args:
            scheduler_factory: Function that creates scheduler instances
        """
        self.scheduler_factory = scheduler_factory
        self._pool = SchedulerPool(max_workers=4, use_processes=False)
    
    async def schedule_async(
        self,
        agents: List[Agent],
        tasks: List[Task],
        constraints: Dict[str, Any] = None
    ) -> Solution:
        """Schedule asynchronously.
        
        Args:
            agents: List of agents
            tasks: List of tasks
            constraints: Scheduling constraints
            
        Returns:
            Scheduling solution
        """
        problem = SchedulingProblem(
            agents=agents,
            tasks=tasks,
            constraints=constraints or {},
            optimization_target="minimize_cost"
        )
        
        # Submit to thread pool
        job_id = self._pool.submit_single(problem, self.scheduler_factory)
        
        # Wait for result asynchronously
        loop = asyncio.get_event_loop()
        
        while True:
            result = await loop.run_in_executor(
                None, 
                self._pool.get_result, 
                job_id, 
                0.1  # Short timeout for polling
            )
            
            if result is not None:
                if result.success and result.solutions:
                    return result.solutions[0]
                else:
                    raise SolverError(result.error or "Async scheduling failed")
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
    
    async def schedule_batch_async(
        self,
        problems: List[SchedulingProblem]
    ) -> List[Solution]:
        """Schedule multiple problems asynchronously.
        
        Args:
            problems: List of scheduling problems
            
        Returns:
            List of solutions
        """
        job_id = self._pool.submit_batch(problems, self.scheduler_factory)
        
        # Wait for result asynchronously
        loop = asyncio.get_event_loop()
        
        while True:
            result = await loop.run_in_executor(
                None,
                self._pool.get_result,
                job_id,
                0.1
            )
            
            if result is not None:
                if result.success:
                    return result.solutions
                else:
                    raise SolverError(result.error or "Async batch scheduling failed")
            
            await asyncio.sleep(0.1)
    
    def close(self):
        """Close the async scheduler."""
        self._pool.shutdown()