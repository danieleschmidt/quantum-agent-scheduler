"""Custom exceptions for quantum scheduler."""


class QuantumSchedulerError(Exception):
    """Base exception for quantum scheduler errors."""
    pass


class ValidationError(QuantumSchedulerError):
    """Raised when validation fails."""
    pass


class BackendError(QuantumSchedulerError):
    """Raised when backend operations fail."""
    pass


class ConstraintError(QuantumSchedulerError):
    """Raised when constraint operations fail."""
    pass


class SolverError(QuantumSchedulerError):
    """Raised when solving fails."""
    pass


class ConfigurationError(QuantumSchedulerError):
    """Raised when configuration is invalid."""
    pass


class ResourceError(QuantumSchedulerError):
    """Raised when resource limits are exceeded."""
    pass


class SkillMismatchError(ValidationError):
    """Raised when no agent has required skills for a task."""
    
    def __init__(self, task_id: str, required_skills: list, available_skills: set):
        self.task_id = task_id
        self.required_skills = required_skills
        self.available_skills = available_skills
        missing = set(required_skills) - available_skills
        super().__init__(
            f"Task {task_id} requires skills {missing} not available in any agent"
        )


class CapacityExceededError(ValidationError):
    """Raised when agent capacity is exceeded."""
    
    def __init__(self, agent_id: str, capacity: int, workload: int):
        self.agent_id = agent_id
        self.capacity = capacity
        self.workload = workload
        super().__init__(
            f"Agent {agent_id} capacity {capacity} exceeded by workload {workload}"
        )


class BackendUnavailableError(BackendError):
    """Raised when a backend is not available."""
    
    def __init__(self, backend_name: str, reason: str = None):
        self.backend_name = backend_name
        self.reason = reason
        message = f"Backend {backend_name} is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class SolverTimeoutError(SolverError):
    """Raised when solver exceeds time limit."""
    
    def __init__(self, timeout: float, backend: str = None):
        self.timeout = timeout
        self.backend = backend
        message = f"Solver timed out after {timeout}s"
        if backend:
            message += f" using {backend} backend"
        super().__init__(message)


class InfeasibleProblemError(SolverError):
    """Raised when problem has no feasible solution."""
    
    def __init__(self, reason: str = None):
        self.reason = reason
        message = "Problem has no feasible solution"
        if reason:
            message += f": {reason}"
        super().__init__(message)