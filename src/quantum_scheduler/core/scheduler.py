"""Core quantum scheduler implementation."""

import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

from .models import Agent, Task, Solution, SchedulingProblem
from .exceptions import (
    ValidationError,
    BackendError,
    SolverError,
    SolverTimeoutError,
    InfeasibleProblemError,
    SkillMismatchError,
    CapacityExceededError
)
from .validators import InputValidator
from ..backends import Backend, ClassicalBackend, HybridBackend, SimulatedQuantumBackend
from ..constraints import Constraint
from ..monitoring import get_metrics_collector
from ..security import SecuritySanitizer
from ..optimization import get_solution_cache, ProblemOptimizer

logger = logging.getLogger(__name__)


class QuantumScheduler:
    """Hybrid classical-quantum scheduler for multi-agent systems."""
    
    def __init__(
        self, 
        backend: Union[str, Backend] = "auto",
        fallback: str = "classical",
        optimization_target: str = "minimize_cost",
        timeout: Optional[float] = None,
        enable_validation: bool = True,
        enable_metrics: bool = True,
        enable_caching: bool = True,
        enable_optimization: bool = True
    ):
        """Initialize the quantum scheduler.
        
        Args:
            backend: Backend type or instance ("auto", "classical", "quantum", etc.)
            fallback: Fallback solver type when primary backend fails
            optimization_target: Optimization objective ("minimize_cost", "minimize_time", etc.)
            timeout: Maximum solve time in seconds (None for no timeout)
            enable_validation: Enable input validation and sanitization
            enable_metrics: Enable metrics collection
            enable_caching: Enable solution caching
            enable_optimization: Enable problem preprocessing optimization
        """
        # Validate and sanitize inputs
        if isinstance(backend, str):
            backend = SecuritySanitizer.sanitize_id(backend)
        
        if fallback:
            fallback = SecuritySanitizer.sanitize_id(fallback)
        
        optimization_target = SecuritySanitizer.sanitize_id(optimization_target)
        
        if timeout is not None:
            timeout = SecuritySanitizer.sanitize_number(timeout, min_val=0.1, max_val=3600)
        
        # Initialize components
        self._backend_type = backend if isinstance(backend, str) else "custom"
        self._backend = self._initialize_backend(backend)
        self._fallback = fallback
        self._optimization_target = optimization_target
        self._timeout = timeout
        self._enable_validation = enable_validation
        self._enable_metrics = enable_metrics
        self._enable_caching = enable_caching
        self._enable_optimization = enable_optimization
        self._constraints: List[Constraint] = []
        self._metrics_collector = get_metrics_collector() if enable_metrics else None
        self._solution_cache = get_solution_cache() if enable_caching else None
        
        logger.info(f"Initialized QuantumScheduler with backend: {self._backend_type}, timeout: {timeout}s, caching: {enable_caching}, optimization: {enable_optimization}")
    
    def _initialize_backend(self, backend: Union[str, Backend]) -> Backend:
        """Initialize the appropriate backend."""
        if isinstance(backend, Backend):
            return backend
        
        if backend == "classical":
            return ClassicalBackend()
        elif backend == "quantum_sim":
            return SimulatedQuantumBackend(noise_level=0.1)
        elif backend == "hybrid":
            return HybridBackend(quantum_threshold=50, prefer_quantum=True)
        elif backend == "auto":
            # Auto-select best available backend
            return HybridBackend(quantum_threshold=25, prefer_quantum=True)
        else:
            logger.warning(f"Backend {backend} not implemented, falling back to hybrid")
            return HybridBackend(quantum_threshold=50, prefer_quantum=False)
    
    def schedule(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """Schedule tasks to agents optimally.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to schedule
            constraints: Additional scheduling constraints
            
        Returns:
            Optimal scheduling solution
            
        Raises:
            ValidationError: If inputs are invalid
            SolverError: If solving fails
            SolverTimeoutError: If solver exceeds timeout
        """
        operation_id = str(uuid.uuid4())
        
        try:
            # Start metrics collection
            if self._metrics_collector:
                self._metrics_collector.start_operation(operation_id, len(agents), len(tasks))
            
            # Handle empty inputs gracefully
            if not agents and not tasks:
                solution = Solution(assignments={}, cost=0.0, solver_type=self._backend_type)
                if self._metrics_collector:
                    self._metrics_collector.end_operation(
                        operation_id, True, self._backend_type, 0, 0.0, 0.0
                    )
                return solution
            
            # Validate and sanitize inputs
            if self._enable_validation:
                agents = self._validate_agents(agents)
                tasks = self._validate_tasks(tasks)
                constraints = self._validate_constraints(constraints or {})
            
            # Create and validate problem
            problem = SchedulingProblem(
                agents=agents,
                tasks=tasks,
                constraints=constraints or {},
                optimization_target=self._optimization_target
            )
            
            if self._enable_validation:
                problem = InputValidator.validate_problem(problem)
            
            # Check cache first
            if self._solution_cache:
                cached_solution = self._solution_cache.get(problem)
                if cached_solution:
                    if self._metrics_collector:
                        self._metrics_collector.end_operation(
                            operation_id, True, "cached",
                            cached_solution.total_assignments, cached_solution.cost, cached_solution.utilization_ratio
                        )
                    logger.info(f"Found cached solution with {len(cached_solution.assignments)} assignments")
                    return cached_solution
            
            # Optimize problem if enabled
            if self._enable_optimization:
                problem = ProblemOptimizer.preprocess_problem(problem)
                problem = ProblemOptimizer.optimize_for_backend(problem, self._backend_type)
            
            # Check skill compatibility early
            if constraints and constraints.get("skill_match_required", False):
                try:
                    InputValidator.check_skill_availability(agents, tasks)
                except SkillMismatchError as e:
                    logger.error(f"Skill mismatch detected: {e}")
                    if self._metrics_collector:
                        self._metrics_collector.end_operation(
                            operation_id, False, self._backend_type, error_type="SkillMismatchError"
                        )
                    raise
            
            # Check capacity feasibility
            if self._enable_validation:
                try:
                    InputValidator.check_capacity_feasibility(agents, tasks)
                except CapacityExceededError as e:
                    logger.warning(f"Capacity warning: {e}")
            
            start_time = time.time()
            
            try:
                # Attempt to solve with primary backend
                solution = self._solve_with_timeout(problem)
                solution.execution_time = time.time() - start_time
                solution.solver_type = self._backend_type
                
                # Cache the solution
                if self._solution_cache:
                    self._solution_cache.put(problem, solution)
                
                # Record success metrics
                if self._metrics_collector:
                    self._metrics_collector.end_operation(
                        operation_id, True, self._backend_type,
                        solution.total_assignments, solution.cost, solution.utilization_ratio
                    )
                
                logger.info(f"Solved scheduling problem with {len(solution.assignments)} assignments in {solution.execution_time:.3f}s")
                return solution
                
            except (SolverTimeoutError, SolverError, BackendError) as e:
                logger.error(f"Primary backend failed: {e}")
                
                # Attempt fallback if available
                if self._fallback == "classical" and self._backend_type != "classical":
                    logger.info("Falling back to classical solver")
                    
                    try:
                        fallback_backend = ClassicalBackend()
                        solution = fallback_backend.solve(problem)
                        solution.execution_time = time.time() - start_time
                        solution.solver_type = "classical_fallback"
                        
                        # Record fallback success
                        if self._metrics_collector:
                            self._metrics_collector.end_operation(
                                operation_id, True, "classical_fallback",
                                solution.total_assignments, solution.cost, solution.utilization_ratio
                            )
                        
                        logger.info(f"Fallback solution found with {len(solution.assignments)} assignments")
                        return solution
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        if self._metrics_collector:
                            self._metrics_collector.end_operation(
                                operation_id, False, "classical_fallback", 
                                error_type=type(fallback_error).__name__
                            )
                        raise SolverError(f"Both primary and fallback solvers failed: {e}, {fallback_error}")
                else:
                    if self._metrics_collector:
                        self._metrics_collector.end_operation(
                            operation_id, False, self._backend_type, error_type=type(e).__name__
                        )
                    raise
                    
        except (ValidationError, SkillMismatchError, CapacityExceededError):
            # Re-raise validation errors without wrapping
            if self._metrics_collector:
                self._metrics_collector.end_operation(
                    operation_id, False, self._backend_type, error_type="ValidationError"
                )
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in schedule(): {e}", exc_info=True)
            if self._metrics_collector:
                self._metrics_collector.end_operation(
                    operation_id, False, self._backend_type, error_type=type(e).__name__
                )
            raise SolverError(f"Scheduling failed due to unexpected error: {e}")
    
    def _solve_with_timeout(self, problem: SchedulingProblem) -> Solution:
        """Solve the scheduling problem with timeout handling."""
        if self._timeout is None:
            return self._backend.solve(problem)
        
        import signal
        
        def timeout_handler(signum, frame):
            raise SolverTimeoutError(self._timeout, self._backend_type)
        
        # Set up timeout (Unix-like systems only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self._timeout))
            
            try:
                solution = self._backend.solve(problem)
                signal.alarm(0)  # Cancel alarm
                return solution
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                raise
                
        except AttributeError:
            # Windows or system without signal support - use basic timeout
            start_time = time.time()
            solution = self._backend.solve(problem)
            
            if time.time() - start_time > self._timeout:
                logger.warning(f"Solver exceeded timeout {self._timeout}s but completed anyway")
            
            return solution
    
    def _validate_agents(self, agents: List[Agent]) -> List[Agent]:
        """Validate and sanitize agent list."""
        if not isinstance(agents, list):
            raise ValidationError(f"Agents must be a list, got {type(agents)}")
        
        validated_agents = []
        for i, agent in enumerate(agents):
            if not isinstance(agent, Agent):
                raise ValidationError(f"Agent {i} is not an Agent instance: {type(agent)}")
            
            try:
                validated_agent = InputValidator.validate_agent(agent)
                validated_agents.append(validated_agent)
            except ValidationError as e:
                raise ValidationError(f"Agent {i} validation failed: {e}")
        
        return validated_agents
    
    def _validate_tasks(self, tasks: List[Task]) -> List[Task]:
        """Validate and sanitize task list."""
        if not isinstance(tasks, list):
            raise ValidationError(f"Tasks must be a list, got {type(tasks)}")
        
        validated_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, Task):
                raise ValidationError(f"Task {i} is not a Task instance: {type(task)}")
            
            try:
                validated_task = InputValidator.validate_task(task)
                validated_tasks.append(validated_task)
            except ValidationError as e:
                raise ValidationError(f"Task {i} validation failed: {e}")
        
        return validated_tasks
    
    def _validate_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize constraints."""
        if not isinstance(constraints, dict):
            raise ValidationError(f"Constraints must be a dict, got {type(constraints)}")
        
        try:
            return InputValidator.validate_constraints(constraints)
        except ValidationError as e:
            raise ValidationError(f"Constraints validation failed: {e}")
    
    def _solve(self, problem: SchedulingProblem) -> Solution:
        """Solve the scheduling problem using the configured backend."""
        return self._backend.solve(problem)
    
    def _validate_skill_matching(self, agents: List[Agent], tasks: List[Task]):
        """Validate that all tasks can be completed by available agents."""
        available_skills = set()
        for agent in agents:
            available_skills.update(agent.skills)
        
        for task in tasks:
            required_skills = set(task.required_skills)
            if not required_skills.issubset(available_skills):
                missing_skills = required_skills - available_skills
                raise ValueError(
                    f"No agent has required skills for task {task.id}: {missing_skills}"
                )
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a custom constraint to the scheduler.
        
        Args:
            constraint: Constraint instance to add
        """
        self._constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.__class__.__name__}")
    
    def set_backend(self, backend: Union[str, Backend]) -> None:
        """Change the scheduler backend.
        
        Args:
            backend: New backend type or instance
        """
        self._backend_type = backend if isinstance(backend, str) else "custom"
        self._backend = self._initialize_backend(backend)
        logger.info(f"Switched to backend: {self._backend_type}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get scheduler performance metrics."""
        return {
            "backend_type": self._backend_type,
            "constraints_count": len(self._constraints),
            "optimization_target": self._optimization_target,
            "has_fallback": self._fallback is not None
        }
    
    def estimate_solve_time(self, num_agents: int, num_tasks: int) -> float:
        """Estimate solve time for given problem size.
        
        Args:
            num_agents: Number of agents
            num_tasks: Number of tasks
            
        Returns:
            Estimated solve time in seconds
        """
        # Simple heuristic - real implementation would be backend-specific
        complexity = num_agents * num_tasks
        
        if self._backend_type == "classical":
            return complexity * 0.001  # Linear scaling for classical
        else:
            return complexity * 0.0001  # Better scaling for quantum (theoretical)