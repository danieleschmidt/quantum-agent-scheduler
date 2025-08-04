"""Core data models for quantum scheduling."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Priority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    CRITICAL = 8


@dataclass
class Agent:
    """Represents an agent with specific capabilities and capacity."""
    
    id: str
    skills: List[str]
    capacity: int
    availability: Optional[List[tuple]] = None  # List of (start, end) time tuples
    performance_history: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate agent data after initialization."""
        if not self.id:
            raise ValueError("Agent ID cannot be empty")
        if self.capacity <= 0:
            raise ValueError("Agent capacity must be positive")
        if not self.skills:
            raise ValueError("Agent must have at least one skill")


@dataclass
class Task:
    """Represents a task to be scheduled."""
    
    id: str
    required_skills: List[str]
    duration: int
    priority: float
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    estimated_value: float = 1.0
    
    def __post_init__(self):
        """Validate task data after initialization."""
        if not self.id:
            raise ValueError("Task ID cannot be empty")
        if self.duration <= 0:
            raise ValueError("Task duration must be positive")
        if not self.required_skills:
            raise ValueError("Task must require at least one skill")
        if self.priority < 0:
            raise ValueError("Task priority cannot be negative")


@dataclass
class Assignment:
    """Represents an assignment of a task to an agent."""
    
    task_id: str
    agent_id: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class Solution:
    """Represents a complete scheduling solution."""
    
    assignments: Dict[str, str]  # task_id -> agent_id
    start_times: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    solver_type: str = "unknown"
    quantum_circuit: Optional[Any] = None
    execution_time: float = 0.0
    success_probability: float = 1.0
    
    @property
    def total_assignments(self) -> int:
        """Get total number of assignments."""
        return len(self.assignments)
    
    @property
    def utilization_ratio(self) -> float:
        """Calculate resource utilization ratio (0.0 to 1.0)."""
        if not self.assignments:
            return 0.0
        unique_agents = len(set(self.assignments.values()))
        if unique_agents == 0:
            return 0.0
        # Calculate based on agent utilization, capped at 1.0
        return min(1.0, len(self.assignments) / max(1, unique_agents))


@dataclass
class SchedulingProblem:
    """Encapsulates a complete scheduling problem definition."""
    
    agents: List[Agent]
    tasks: List[Task]
    constraints: Dict[str, Any]
    optimization_target: str = "minimize_cost"
    
    def validate(self) -> bool:
        """Validate the scheduling problem is well-formed."""
        if not self.agents:
            raise ValueError("Must have at least one agent")
        if not self.tasks:
            raise ValueError("Must have at least one task")
        
        # Check that all task dependencies exist
        task_ids = {task.id for task in self.tasks}
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep}")
        
        return True
    
    @property
    def total_capacity(self) -> int:
        """Get total agent capacity."""
        return sum(agent.capacity for agent in self.agents)
    
    @property
    def total_workload(self) -> int:
        """Get total task workload."""
        return sum(task.duration for task in self.tasks)
    
    @property
    def capacity_utilization(self) -> float:
        """Calculate expected capacity utilization."""
        if self.total_capacity == 0:
            return 0.0
        return min(1.0, self.total_workload / self.total_capacity)