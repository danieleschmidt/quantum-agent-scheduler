"""Shared test configuration and fixtures."""

import pytest
from typing import List
from unittest.mock import Mock

from quantum_scheduler import Agent, Task, QuantumScheduler


@pytest.fixture
def sample_agents() -> List[Agent]:
    """Create sample agents for testing."""
    return [
        Agent(id="agent1", skills=["python", "ml"], capacity=2),
        Agent(id="agent2", skills=["java", "web"], capacity=3),
        Agent(id="agent3", skills=["python", "web"], capacity=2),
    ]


@pytest.fixture
def sample_tasks() -> List[Task]:
    """Create sample tasks for testing."""
    return [
        Task(id="task1", required_skills=["python"], duration=2, priority=5),
        Task(id="task2", required_skills=["web"], duration=1, priority=3),
        Task(id="task3", required_skills=["ml", "python"], duration=3, priority=8),
    ]


@pytest.fixture
def mock_quantum_backend():
    """Mock quantum backend for testing."""
    backend = Mock()
    backend.solve.return_value = {"assignments": {"task1": "agent1"}, "cost": 10.0}
    return backend


@pytest.fixture
def scheduler_with_mock_backend(mock_quantum_backend):
    """Create scheduler with mocked backend."""
    scheduler = QuantumScheduler(backend="mock")
    scheduler._backend = mock_quantum_backend
    return scheduler


@pytest.fixture
def basic_constraints():
    """Basic scheduling constraints for testing."""
    return {
        "max_concurrent_tasks": 2,
        "skill_match_required": True,
        "deadline_enforcement": False,
    }


# Quantum hardware testing markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "quantum: mark test as requiring quantum hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip quantum tests if quantum hardware not available."""
    try:
        if config.getoption("--quantum"):
            # Run quantum tests
            return
    except ValueError:
        # --quantum option not defined, skip quantum tests
        pass
    
    skip_quantum = pytest.mark.skip(reason="need --quantum option to run")
    for item in items:
        if "quantum" in item.keywords:
            item.add_marker(skip_quantum)