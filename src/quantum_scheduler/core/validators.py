"""Input validation and sanitization utilities."""

import re
from typing import List, Dict, Any, Set
import logging

from .models import Agent, Task, SchedulingProblem
from .exceptions import (
    ValidationError,
    SkillMismatchError,
    CapacityExceededError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes scheduler inputs."""
    
    # Security patterns
    SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    SAFE_SKILL_PATTERN = re.compile(r'^[a-zA-Z0-9_+-]+$')
    MAX_ID_LENGTH = 100
    MAX_SKILL_LENGTH = 50
    MAX_AGENTS = 10000
    MAX_TASKS = 100000
    
    @classmethod
    def validate_agent(cls, agent: Agent) -> Agent:
        """Validate and sanitize agent data.
        
        Args:
            agent: Agent to validate
            
        Returns:
            Validated agent
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate ID
        if not agent.id or not isinstance(agent.id, str):
            raise ValidationError("Agent ID must be a non-empty string")
        
        if len(agent.id) > cls.MAX_ID_LENGTH:
            raise ValidationError(f"Agent ID too long: {len(agent.id)} > {cls.MAX_ID_LENGTH}")
        
        if not cls.SAFE_ID_PATTERN.match(agent.id):
            raise ValidationError(f"Agent ID contains invalid characters: {agent.id}")
        
        # Validate skills
        if not agent.skills or not isinstance(agent.skills, list):
            raise ValidationError("Agent skills must be a non-empty list")
        
        for skill in agent.skills:
            if not isinstance(skill, str):
                raise ValidationError(f"Agent skill must be string: {skill}")
            
            if len(skill) > cls.MAX_SKILL_LENGTH:
                raise ValidationError(f"Skill name too long: {len(skill)} > {cls.MAX_SKILL_LENGTH}")
            
            if not cls.SAFE_SKILL_PATTERN.match(skill):
                raise ValidationError(f"Skill contains invalid characters: {skill}")
        
        # Remove duplicates and normalize
        agent.skills = list(set(skill.lower().strip() for skill in agent.skills))
        
        # Validate capacity
        if not isinstance(agent.capacity, int) or agent.capacity <= 0:
            raise ValidationError(f"Agent capacity must be positive integer: {agent.capacity}")
        
        if agent.capacity > 1000:  # Reasonable upper bound
            raise ValidationError(f"Agent capacity too high: {agent.capacity}")
        
        # Validate performance history
        if agent.performance_history:
            if not isinstance(agent.performance_history, dict):
                raise ValidationError("Performance history must be a dictionary")
            
            for skill, performance in agent.performance_history.items():
                if not isinstance(skill, str) or not isinstance(performance, (int, float)):
                    raise ValidationError("Performance history must map skills to numbers")
                
                if not 0 <= performance <= 1:
                    logger.warning(f"Performance value {performance} for skill {skill} outside [0,1]")
        
        logger.debug(f"Validated agent {agent.id} with {len(agent.skills)} skills")
        return agent
    
    @classmethod
    def validate_task(cls, task: Task) -> Task:
        """Validate and sanitize task data.
        
        Args:
            task: Task to validate
            
        Returns:
            Validated task
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate ID
        if not task.id or not isinstance(task.id, str):
            raise ValidationError("Task ID must be a non-empty string")
        
        if len(task.id) > cls.MAX_ID_LENGTH:
            raise ValidationError(f"Task ID too long: {len(task.id)} > {cls.MAX_ID_LENGTH}")
        
        if not cls.SAFE_ID_PATTERN.match(task.id):
            raise ValidationError(f"Task ID contains invalid characters: {task.id}")
        
        # Validate required skills
        if not task.required_skills or not isinstance(task.required_skills, list):
            raise ValidationError("Task required_skills must be a non-empty list")
        
        for skill in task.required_skills:
            if not isinstance(skill, str):
                raise ValidationError(f"Required skill must be string: {skill}")
            
            if len(skill) > cls.MAX_SKILL_LENGTH:
                raise ValidationError(f"Skill name too long: {len(skill)} > {cls.MAX_SKILL_LENGTH}")
            
            if not cls.SAFE_SKILL_PATTERN.match(skill):
                raise ValidationError(f"Skill contains invalid characters: {skill}")
        
        # Normalize skills
        task.required_skills = list(set(skill.lower().strip() for skill in task.required_skills))
        
        # Validate duration
        if not isinstance(task.duration, int) or task.duration <= 0:
            raise ValidationError(f"Task duration must be positive integer: {task.duration}")
        
        if task.duration > 10000:  # Reasonable upper bound
            raise ValidationError(f"Task duration too high: {task.duration}")
        
        # Validate priority
        if not isinstance(task.priority, (int, float)) or task.priority < 0:
            raise ValidationError(f"Task priority must be non-negative number: {task.priority}")
        
        # Validate dependencies
        if task.dependencies:
            if not isinstance(task.dependencies, list):
                raise ValidationError("Task dependencies must be a list")
            
            for dep in task.dependencies:
                if not isinstance(dep, str):
                    raise ValidationError(f"Dependency must be string: {dep}")
                
                if not cls.SAFE_ID_PATTERN.match(dep):
                    raise ValidationError(f"Dependency ID contains invalid characters: {dep}")
        
        # Validate deadline
        if task.deadline is not None:
            if not isinstance(task.deadline, (int, float)) or task.deadline < 0:
                raise ValidationError(f"Task deadline must be non-negative number: {task.deadline}")
        
        # Validate estimated value
        if not isinstance(task.estimated_value, (int, float)) or task.estimated_value <= 0:
            raise ValidationError(f"Task estimated_value must be positive number: {task.estimated_value}")
        
        logger.debug(f"Validated task {task.id} with {len(task.required_skills)} required skills")
        return task
    
    @classmethod
    def validate_problem(cls, problem: SchedulingProblem) -> SchedulingProblem:
        """Validate complete scheduling problem.
        
        Args:
            problem: Problem to validate
            
        Returns:
            Validated problem
            
        Raises:
            ValidationError: If validation fails
        """
        # Check problem size limits
        if len(problem.agents) > cls.MAX_AGENTS:
            raise ValidationError(f"Too many agents: {len(problem.agents)} > {cls.MAX_AGENTS}")
        
        if len(problem.tasks) > cls.MAX_TASKS:
            raise ValidationError(f"Too many tasks: {len(problem.tasks)} > {cls.MAX_TASKS}")
        
        # Validate individual agents and tasks
        validated_agents = []
        agent_ids = set()
        
        for agent in problem.agents:
            validated_agent = cls.validate_agent(agent)
            
            if validated_agent.id in agent_ids:
                raise ValidationError(f"Duplicate agent ID: {validated_agent.id}")
            
            agent_ids.add(validated_agent.id)
            validated_agents.append(validated_agent)
        
        validated_tasks = []
        task_ids = set()
        
        for task in problem.tasks:
            validated_task = cls.validate_task(task)
            
            if validated_task.id in task_ids:
                raise ValidationError(f"Duplicate task ID: {validated_task.id}")
            
            task_ids.add(validated_task.id)
            validated_tasks.append(validated_task)
        
        # Validate task dependencies exist
        for task in validated_tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValidationError(f"Task {task.id} depends on non-existent task {dep_id}")
        
        # Check for circular dependencies
        cls._check_circular_dependencies(validated_tasks)
        
        # Validate constraints
        if problem.constraints and not isinstance(problem.constraints, dict):
            raise ValidationError("Constraints must be a dictionary")
        
        # Update problem with validated data
        problem.agents = validated_agents
        problem.tasks = validated_tasks
        
        logger.info(f"Validated problem with {len(validated_agents)} agents and {len(validated_tasks)} tasks")
        return problem
    
    @classmethod
    def _check_circular_dependencies(cls, tasks: List[Task]) -> None:
        """Check for circular dependencies in task list.
        
        Args:
            tasks: List of tasks to check
            
        Raises:
            ValidationError: If circular dependency found
        """
        task_deps = {task.id: set(task.dependencies) for task in tasks}
        
        def has_cycle(task_id: str, visited: Set[str], path: Set[str]) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            path.add(task_id)
            
            for dep in task_deps.get(task_id, []):
                if has_cycle(dep, visited, path):
                    return True
            
            path.remove(task_id)
            return False
        
        visited = set()
        for task_id in task_deps:
            if has_cycle(task_id, visited, set()):
                raise ValidationError(f"Circular dependency detected involving task {task_id}")
    
    @classmethod
    def validate_constraints(cls, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize constraints.
        
        Args:
            constraints: Constraints to validate
            
        Returns:
            Validated constraints
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(constraints, dict):
            raise ValidationError("Constraints must be a dictionary")
        
        validated = {}
        
        # Known constraint types and their validation
        constraint_validators = {
            'skill_match_required': lambda x: isinstance(x, bool),
            'max_concurrent_tasks': lambda x: isinstance(x, int) and x > 0,
            'deadline_enforcement': lambda x: isinstance(x, bool),
            'allow_partial_assignment': lambda x: isinstance(x, bool),
            'optimization_timeout': lambda x: isinstance(x, (int, float)) and x > 0,
        }
        
        for key, value in constraints.items():
            if not isinstance(key, str):
                raise ValidationError(f"Constraint key must be string: {key}")
            
            if key in constraint_validators:
                if not constraint_validators[key](value):
                    raise ValidationError(f"Invalid value for constraint {key}: {value}")
            else:
                logger.warning(f"Unknown constraint type: {key}")
            
            validated[key] = value
        
        return validated
    
    @classmethod
    def check_skill_availability(cls, agents: List[Agent], tasks: List[Task]) -> None:
        """Check that all required skills are available.
        
        Args:
            agents: List of agents
            tasks: List of tasks
            
        Raises:
            SkillMismatchError: If skills are missing
        """
        available_skills = set()
        for agent in agents:
            available_skills.update(agent.skills)
        
        for task in tasks:
            required_skills = set(task.required_skills)
            missing_skills = required_skills - available_skills
            
            if missing_skills:
                raise SkillMismatchError(task.id, list(missing_skills), available_skills)
    
    @classmethod
    def check_capacity_feasibility(cls, agents: List[Agent], tasks: List[Task]) -> None:
        """Check if total capacity can handle total workload.
        
        Args:
            agents: List of agents
            tasks: List of tasks
            
        Raises:
            CapacityExceededError: If total capacity insufficient
        """
        total_capacity = sum(agent.capacity for agent in agents)
        total_workload = sum(task.duration for task in tasks)
        
        if total_workload > total_capacity:
            raise CapacityExceededError(
                "total", total_capacity, total_workload
            )