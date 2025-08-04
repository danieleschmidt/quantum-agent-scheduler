"""Base constraint classes and common implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import logging

from ..core.models import Agent, Task

logger = logging.getLogger(__name__)


class Constraint(ABC):
    """Abstract base class for scheduling constraints."""
    
    def __init__(self, penalty_weight: float = 1.0):
        """Initialize constraint with penalty weight.
        
        Args:
            penalty_weight: Weight for constraint violations in optimization
        """
        self.penalty_weight = penalty_weight
    
    @abstractmethod
    def to_qubo(self, tasks: List[Task], agents: List[Agent]) -> Dict[Tuple[int, int], float]:
        """Convert constraint to QUBO (Quadratic Unconstrained Binary Optimization) format.
        
        Args:
            tasks: List of tasks
            agents: List of agents
            
        Returns:
            QUBO matrix entries as dictionary
        """
        pass
    
    @abstractmethod
    def validate(self, assignments: Dict[str, str], tasks: List[Task], agents: List[Agent]) -> bool:
        """Validate if assignments satisfy this constraint.
        
        Args:
            assignments: Task-to-agent assignments
            tasks: List of tasks
            agents: List of agents
            
        Returns:
            True if constraint is satisfied
        """
        pass
    
    def get_violation_penalty(self, assignments: Dict[str, str], tasks: List[Task], agents: List[Agent]) -> float:
        """Calculate penalty for constraint violations.
        
        Args:
            assignments: Task-to-agent assignments
            tasks: List of tasks  
            agents: List of agents
            
        Returns:
            Penalty value for violations
        """
        if self.validate(assignments, tasks, agents):
            return 0.0
        return self.penalty_weight


class CapacityConstraint(Constraint):
    """Constraint ensuring agents don't exceed their capacity."""
    
    def to_qubo(self, tasks: List[Task], agents: List[Agent]) -> Dict[Tuple[int, int], float]:
        """Convert capacity constraint to QUBO format."""
        qubo = {}
        
        # For each agent, penalize assignments that exceed capacity
        for agent_idx, agent in enumerate(agents):
            total_duration = 0
            task_indices = []
            
            for task_idx, task in enumerate(tasks):
                # Binary variable x_{task_idx,agent_idx} indicates task assigned to agent
                var_idx = task_idx * len(agents) + agent_idx
                total_duration += task.duration
                task_indices.append(var_idx)
            
            # If total possible duration exceeds capacity, add quadratic penalty
            if total_duration > agent.capacity:
                excess = total_duration - agent.capacity
                penalty = self.penalty_weight * excess
                
                # Add quadratic terms for all pairs of tasks assigned to this agent
                for i in range(len(task_indices)):
                    for j in range(i, len(task_indices)):
                        if i == j:
                            qubo[(task_indices[i], task_indices[i])] = penalty
                        else:
                            qubo[(task_indices[i], task_indices[j])] = 2 * penalty
        
        return qubo
    
    def validate(self, assignments: Dict[str, str], tasks: List[Task], agents: List[Agent]) -> bool:
        """Validate capacity constraints are satisfied."""
        agent_workload = {}
        
        # Calculate workload for each agent
        for task in tasks:
            if task.id in assignments:
                agent_id = assignments[task.id]
                agent_workload[agent_id] = agent_workload.get(agent_id, 0) + task.duration
        
        # Check capacity limits
        agent_capacity = {agent.id: agent.capacity for agent in agents}
        for agent_id, workload in agent_workload.items():
            if workload > agent_capacity.get(agent_id, 0):
                return False
        
        return True


class SkillMatchConstraint(Constraint):
    """Constraint ensuring tasks are assigned to agents with required skills."""
    
    def to_qubo(self, tasks: List[Task], agents: List[Agent]) -> Dict[Tuple[int, int], float]:
        """Convert skill matching constraint to QUBO format."""
        qubo = {}
        
        for task_idx, task in enumerate(tasks):
            for agent_idx, agent in enumerate(agents):
                # Check if agent has all required skills
                required_skills = set(task.required_skills)
                agent_skills = set(agent.skills)
                
                if not required_skills.issubset(agent_skills):
                    # Heavy penalty for invalid assignments
                    var_idx = task_idx * len(agents) + agent_idx
                    qubo[(var_idx, var_idx)] = self.penalty_weight * 1000
        
        return qubo
    
    def validate(self, assignments: Dict[str, str], tasks: List[Task], agents: List[Agent]) -> bool:
        """Validate skill matching constraints."""
        agent_skills = {agent.id: set(agent.skills) for agent in agents}
        
        for task in tasks:
            if task.id in assignments:
                agent_id = assignments[task.id]
                required_skills = set(task.required_skills)
                
                if agent_id not in agent_skills:
                    return False
                
                if not required_skills.issubset(agent_skills[agent_id]):
                    return False
        
        return True


class OneTaskPerAgentConstraint(Constraint):
    """Constraint ensuring each task is assigned to exactly one agent."""
    
    def to_qubo(self, tasks: List[Task], agents: List[Agent]) -> Dict[Tuple[int, int], float]:
        """Convert one-task-per-agent constraint to QUBO format."""
        qubo = {}
        
        for task_idx in range(len(tasks)):
            # Each task must be assigned to exactly one agent
            # Penalty for not assigning: -penalty_weight
            # Penalty for multiple assignments: penalty_weight
            
            agent_vars = []
            for agent_idx in range(len(agents)):
                var_idx = task_idx * len(agents) + agent_idx
                agent_vars.append(var_idx)
                
                # Linear penalty to encourage assignment
                qubo[(var_idx, var_idx)] = qubo.get((var_idx, var_idx), 0) - self.penalty_weight
            
            # Quadratic penalty for multiple assignments
            for i in range(len(agent_vars)):
                for j in range(i + 1, len(agent_vars)):
                    qubo[(agent_vars[i], agent_vars[j])] = 2 * self.penalty_weight
        
        return qubo
    
    def validate(self, assignments: Dict[str, str], tasks: List[Task], agents: List[Agent]) -> bool:
        """Validate one-task-per-agent constraint."""
        task_ids = {task.id for task in tasks}
        
        for task_id in task_ids:
            assignment_count = sum(1 for t_id, a_id in assignments.items() if t_id == task_id)
            if assignment_count != 1:
                return False
        
        return True