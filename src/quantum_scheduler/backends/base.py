"""Base backend interface and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

from ..core.models import SchedulingProblem, Solution

logger = logging.getLogger(__name__)


class Backend(ABC):
    """Abstract base class for scheduler backends."""
    
    @abstractmethod
    def solve(self, problem: SchedulingProblem) -> Solution:
        """Solve the scheduling problem.
        
        Args:
            problem: The scheduling problem to solve
            
        Returns:
            Optimal solution
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and ready."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities and limitations."""
        pass


class ClassicalBackend(Backend):
    """Classical optimization backend using greedy assignment."""
    
    def solve(self, problem: SchedulingProblem) -> Solution:
        """Solve using classical greedy algorithm."""
        if not problem.agents or not problem.tasks:
            return Solution(assignments={}, cost=0.0, solver_type="classical")
        
        # Sort tasks by priority (descending) and agents by capacity (descending)
        sorted_tasks = sorted(problem.tasks, key=lambda t: t.priority, reverse=True)
        sorted_agents = sorted(problem.agents, key=lambda a: a.capacity, reverse=True)
        
        assignments = {}
        agent_workload = {agent.id: 0 for agent in problem.agents}
        total_cost = 0.0
        
        # Greedy assignment
        for task in sorted_tasks:
            best_agent = self._find_best_agent(task, sorted_agents, agent_workload, problem.constraints)
            
            if best_agent and agent_workload[best_agent.id] < best_agent.capacity:
                assignments[task.id] = best_agent.id
                agent_workload[best_agent.id] += task.duration
                total_cost += task.duration * (1.0 / task.priority)  # Higher priority = lower cost
        
        logger.info(f"Classical solver assigned {len(assignments)} tasks")
        
        return Solution(
            assignments=assignments,
            cost=total_cost,
            solver_type="classical"
        )
    
    def _find_best_agent(self, task, agents, workload, constraints):
        """Find the best agent for a task based on skills and capacity."""
        skill_match_required = constraints.get("skill_match_required", False)
        
        best_agent = None
        best_score = float('-inf')
        
        for agent in agents:
            # Check skill compatibility
            if skill_match_required:
                required_skills = set(task.required_skills)
                agent_skills = set(agent.skills)
                if not required_skills.issubset(agent_skills):
                    continue
            
            # Check capacity
            if workload[agent.id] + task.duration > agent.capacity:
                continue
            
            # Calculate score (prefer agents with more relevant skills and lower current workload)
            skill_overlap = len(set(task.required_skills) & set(agent.skills))
            remaining_capacity = agent.capacity - workload[agent.id]
            score = skill_overlap * 10 + remaining_capacity
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def is_available(self) -> bool:
        """Classical backend is always available."""
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get classical backend capabilities."""
        return {
            "max_agents": 1000,
            "max_tasks": 10000,
            "supports_constraints": True,
            "supports_dependencies": False,  # Not implemented yet
            "optimization_types": ["minimize_cost", "minimize_time"],
            "execution_mode": "deterministic"
        }