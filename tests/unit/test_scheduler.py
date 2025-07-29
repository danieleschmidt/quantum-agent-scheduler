"""Unit tests for quantum scheduler core functionality."""

import pytest
from unittest.mock import Mock, patch

from quantum_scheduler import QuantumScheduler, Agent, Task


class TestQuantumScheduler:
    """Test cases for QuantumScheduler class."""

    def test_scheduler_initialization_with_auto_backend(self):
        """Test scheduler initializes with auto backend selection."""
        scheduler = QuantumScheduler(backend="auto")
        assert scheduler is not None
        assert scheduler._backend_type == "auto"

    def test_scheduler_initialization_with_classical_backend(self):
        """Test scheduler initializes with classical backend."""
        scheduler = QuantumScheduler(backend="classical")
        assert scheduler is not None
        assert scheduler._backend_type == "classical"

    def test_schedule_basic_assignment(self, sample_agents, sample_tasks, basic_constraints):
        """Test basic task assignment functionality."""
        scheduler = QuantumScheduler(backend="classical")
        
        with patch.object(scheduler, '_solve') as mock_solve:
            mock_solve.return_value = Mock(
                assignments={"task1": "agent1", "task2": "agent2"},
                cost=15.0,
                solver_type="classical"
            )
            
            solution = scheduler.schedule(
                agents=sample_agents,
                tasks=sample_tasks,
                constraints=basic_constraints
            )
            
            assert solution is not None
            assert solution.assignments == {"task1": "agent1", "task2": "agent2"}
            assert solution.cost == 15.0
            assert solution.solver_type == "classical"

    def test_skill_matching_constraint(self, sample_agents, sample_tasks):
        """Test that skill matching constraints are enforced."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Task requiring skills not available in any agent
        invalid_task = Task(
            id="invalid_task", 
            required_skills=["nonexistent_skill"], 
            duration=1, 
            priority=1
        )
        
        with pytest.raises(ValueError, match="No agent has required skills"):
            scheduler.schedule(
                agents=sample_agents,
                tasks=[invalid_task],
                constraints={"skill_match_required": True}
            )

    def test_capacity_constraint(self, sample_agents, basic_constraints):
        """Test that agent capacity constraints are respected."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Create more tasks than total agent capacity
        many_tasks = [
            Task(id=f"task{i}", required_skills=["python"], duration=1, priority=1)
            for i in range(10)  # More tasks than total capacity
        ]
        
        solution = scheduler.schedule(
            agents=sample_agents,
            tasks=many_tasks,
            constraints=basic_constraints
        )
        
        # Should handle capacity overflow gracefully
        assert solution is not None
        assert len(solution.assignments) <= sum(agent.capacity for agent in sample_agents)

    def test_empty_input_handling(self):
        """Test scheduler handles empty inputs gracefully."""
        scheduler = QuantumScheduler(backend="classical")
        
        solution = scheduler.schedule(agents=[], tasks=[], constraints={})
        
        assert solution is not None
        assert solution.assignments == {}
        assert solution.cost == 0.0

    @pytest.mark.slow
    def test_large_problem_handling(self):
        """Test scheduler performance with larger problem sizes."""
        scheduler = QuantumScheduler(backend="classical")
        
        # Create larger problem
        agents = [
            Agent(id=f"agent{i}", skills=["python", "ml"], capacity=3)
            for i in range(50)
        ]
        tasks = [
            Task(id=f"task{i}", required_skills=["python"], duration=2, priority=i)
            for i in range(100)
        ]
        
        solution = scheduler.schedule(
            agents=agents,
            tasks=tasks,
            constraints={"skill_match_required": True}
        )
        
        assert solution is not None
        assert isinstance(solution.assignments, dict)

    def test_add_custom_constraint(self):
        """Test adding custom constraints to scheduler."""
        scheduler = QuantumScheduler(backend="classical")
        
        mock_constraint = Mock()
        mock_constraint.to_qubo.return_value = {}
        
        scheduler.add_constraint(mock_constraint)
        
        assert mock_constraint in scheduler._constraints

    def test_backend_switching(self):
        """Test dynamic backend switching functionality."""
        scheduler = QuantumScheduler(backend="classical")
        assert scheduler._backend_type == "classical"
        
        scheduler.set_backend("auto")
        assert scheduler._backend_type == "auto"