"""Tests for QUBO formulation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from quantum_scheduler.core.models import Agent, Task
from quantum_scheduler.qubo import build_qubo, QUBOProblem


@pytest.fixture
def simple_problem():
    tasks = [
        Task(id="T0", required_skills=["python"], duration=2, priority=5.0),
        Task(id="T1", required_skills=["web"], duration=1, priority=3.0),
    ]
    agents = [
        Agent(id="A0", skills=["python", "web"], capacity=5),
        Agent(id="A1", skills=["java"], capacity=5),
    ]
    return tasks, agents


def test_qubo_shape(simple_problem):
    tasks, agents = simple_problem
    qubo = build_qubo(tasks, agents)
    n, m = len(tasks), len(agents)
    assert qubo.Q.shape == (n * m, n * m)
    assert qubo.num_variables == n * m


def test_qubo_variable_map(simple_problem):
    tasks, agents = simple_problem
    qubo = build_qubo(tasks, agents)
    n, m = len(tasks), len(agents)
    # All variable pairs should be in the map
    for i in range(n):
        for j in range(m):
            assert (i, j) in qubo.variable_map
    # Reverse map should be consistent
    for k, (i, j) in qubo.reverse_map.items():
        assert qubo.variable_map[(i, j)] == k


def test_qubo_energy_all_zeros(simple_problem):
    tasks, agents = simple_problem
    qubo = build_qubo(tasks, agents)
    bits = np.zeros(qubo.num_variables, dtype=np.int8)
    assert qubo.energy(bits) == 0.0


def test_qubo_decode_single_assignment(simple_problem):
    tasks, agents = simple_problem
    qubo = build_qubo(tasks, agents)
    # Manually set T0→A0
    bits = np.zeros(qubo.num_variables, dtype=np.int8)
    bits[qubo.variable_map[(0, 0)]] = 1
    decoded = qubo.decode(bits)
    assert decoded.get("T0") == "A0"
    assert "T1" not in decoded


def test_skill_mismatch_increases_energy(simple_problem):
    tasks, agents = simple_problem
    qubo = build_qubo(tasks, agents, penalty_skills=20.0)

    # Assign T1 (needs 'web') to A1 (no 'web' skill) → high energy
    bits_bad = np.zeros(qubo.num_variables, dtype=np.int8)
    bits_bad[qubo.variable_map[(1, 1)]] = 1  # T1 → A1 (mismatch)

    # Assign T1 → A0 (has 'web') → lower energy
    bits_good = np.zeros(qubo.num_variables, dtype=np.int8)
    bits_good[qubo.variable_map[(1, 0)]] = 1  # T1 → A0

    assert qubo.energy(bits_bad) > qubo.energy(bits_good)


def test_double_assignment_penalised(simple_problem):
    tasks, agents = simple_problem
    qubo = build_qubo(tasks, agents, penalty_assignment=10.0)

    # Assign T0 to both agents (double-assignment, bad)
    bits_double = np.zeros(qubo.num_variables, dtype=np.int8)
    bits_double[qubo.variable_map[(0, 0)]] = 1
    bits_double[qubo.variable_map[(0, 1)]] = 1

    # Assign T0 to one agent only
    bits_single = np.zeros(qubo.num_variables, dtype=np.int8)
    bits_single[qubo.variable_map[(0, 0)]] = 1

    assert qubo.energy(bits_double) > qubo.energy(bits_single)
