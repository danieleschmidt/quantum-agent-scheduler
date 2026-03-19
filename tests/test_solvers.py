"""Tests for solver backends."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from quantum_scheduler.core.models import Agent, Task
from quantum_scheduler.qubo import build_qubo
from quantum_scheduler.solvers.simulated_annealing import SimulatedAnnealingSolver
from quantum_scheduler.solvers.dwave_stub import DWaveStubSolver
from quantum_scheduler.solvers.greedy import GreedySolver


@pytest.fixture
def small_tasks():
    return [
        Task(id="T0", required_skills=["python"], duration=2, priority=5.0),
        Task(id="T1", required_skills=["web"], duration=1, priority=3.0),
        Task(id="T2", required_skills=["python", "web"], duration=1, priority=8.0),
    ]


@pytest.fixture
def small_agents():
    return [
        Agent(id="A0", skills=["python", "web"], capacity=6),
        Agent(id="A1", skills=["python"], capacity=4),
    ]


def test_sa_solver_returns_valid_bits(small_tasks, small_agents):
    qubo = build_qubo(small_tasks, small_agents)
    solver = SimulatedAnnealingSolver(num_sweeps=100, num_reads=3, seed=42)
    result = solver.solve(qubo)
    assert result.bits.shape == (qubo.num_variables,)
    assert set(result.bits.tolist()).issubset({0, 1})


def test_sa_solver_energy_finite(small_tasks, small_agents):
    qubo = build_qubo(small_tasks, small_agents)
    solver = SimulatedAnnealingSolver(num_sweeps=200, num_reads=5, seed=0)
    result = solver.solve(qubo)
    assert np.isfinite(result.energy)


def test_sa_solver_assigns_most_tasks(small_tasks, small_agents):
    qubo = build_qubo(small_tasks, small_agents)
    solver = SimulatedAnnealingSolver(num_sweeps=1000, num_reads=10, seed=7)
    result = solver.solve(qubo)
    decoded = qubo.decode(result.bits)
    # At least 2 of 3 tasks should get assigned
    assert len(decoded) >= 2


def test_dwave_stub_fallback(small_tasks, small_agents):
    """DWaveStubSolver must work even without Ocean SDK."""
    qubo = build_qubo(small_tasks, small_agents)
    solver = DWaveStubSolver(use_qpu=False, num_reads=5, seed=1)
    result = solver.solve(qubo)
    assert result.bits is not None
    assert np.isfinite(result.energy)


def test_greedy_assigns_all_feasible(small_tasks, small_agents):
    solver = GreedySolver(skill_match_required=False)
    assignments = solver.solve(small_tasks, small_agents)
    # All tasks should be assignable (total duration 4, total capacity 10)
    assert len(assignments) == len(small_tasks)


def test_greedy_respects_capacity():
    tasks = [
        Task(id="T0", required_skills=["x"], duration=5, priority=1.0),
        Task(id="T1", required_skills=["x"], duration=5, priority=1.0),
    ]
    agents = [Agent(id="A0", skills=["x"], capacity=5)]
    solver = GreedySolver()
    assignments = solver.solve(tasks, agents)
    # Only one task can fit
    assert len(assignments) == 1


def test_greedy_respects_skill_match():
    tasks = [Task(id="T0", required_skills=["quantum"], duration=1, priority=5.0)]
    agents = [Agent(id="A0", skills=["classical"], capacity=10)]
    solver = GreedySolver(skill_match_required=True)
    assignments = solver.solve(tasks, agents)
    assert len(assignments) == 0  # no compatible agent
