"""Tests for the main Scheduler interface."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from quantum_scheduler import Scheduler, Task, Agent


@pytest.fixture
def tasks_10():
    skills_pool = [["python"], ["web"], ["ml", "python"], ["java"], ["devops"],
                   ["python", "web"], ["ml"], ["java", "web"], ["python"], ["devops", "ml"]]
    return [
        Task(id=f"T{i}", required_skills=skills_pool[i], duration=i % 4 + 1, priority=float(i % 5 + 1))
        for i in range(10)
    ]


@pytest.fixture
def agents_4():
    return [
        Agent(id="Alpha",   skills=["python", "ml", "web"],         capacity=15),
        Agent(id="Beta",    skills=["java", "web", "devops"],        capacity=12),
        Agent(id="Gamma",   skills=["python", "devops"],             capacity=10),
        Agent(id="Delta",   skills=["ml", "java", "python", "web"], capacity=20),
    ]


def test_scheduler_sa_assigns_tasks(tasks_10, agents_4):
    s = Scheduler(backend="sa", num_reads=5, num_sweeps=500, seed=42)
    result = s.assign(tasks_10, agents_4)
    assert len(result.assignments) >= 5   # should assign most tasks
    assert result.solver == "qubo-sa"
    assert result.qubo_energy is not None


def test_scheduler_greedy_assigns_tasks(tasks_10, agents_4):
    s = Scheduler(backend="greedy")
    result = s.assign(tasks_10, agents_4)
    assert len(result.assignments) >= 8
    assert result.solver == "greedy"
    assert result.qubo_energy is None


def test_scheduler_empty_input():
    s = Scheduler()
    result = s.assign([], [], {})
    assert result.assignments == {}
    assert result.num_tasks == 0


def test_scheduler_capacity_respected(agents_4):
    """Tasks shouldn't exceed agent capacity."""
    s = Scheduler(backend="sa", num_reads=5, num_sweeps=300, seed=0)
    tasks = [Task(id=f"T{i}", required_skills=["python"], duration=3, priority=5.0) for i in range(8)]
    result = s.assign(tasks, agents_4)

    # Verify no agent is over capacity
    for agent in agents_4:
        workload = result.agent_workload.get(agent.id, 0)
        assert workload <= agent.capacity, f"{agent.id} overloaded: {workload} > {agent.capacity}"


def test_scheduler_skill_match_constraint(tasks_10, agents_4):
    s = Scheduler(backend="sa", num_reads=5, num_sweeps=500, seed=1)
    result = s.assign(tasks_10, agents_4, constraints={"skill_match_required": True})
    # Verify skill matching
    agent_map = {a.id: a for a in agents_4}
    task_map = {t.id: t for t in tasks_10}
    for tid, aid in result.assignments.items():
        task = task_map[tid]
        agent = agent_map[aid]
        for skill in task.required_skills:
            assert skill in agent.skills, f"{tid} needs {skill} but {aid} lacks it"


def test_scheduler_assignment_result_str(tasks_10, agents_4):
    s = Scheduler(backend="greedy")
    result = s.assign(tasks_10, agents_4)
    text = str(result)
    assert "greedy" in text
    assert "Tasks:" in text


def test_assignment_rate(tasks_10, agents_4):
    s = Scheduler(backend="greedy")
    result = s.assign(tasks_10, agents_4)
    assert 0.0 <= result.assignment_rate <= 1.0
