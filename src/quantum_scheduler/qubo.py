"""
QUBO formulation of the agent task assignment problem.

The task assignment problem is modeled as a Quadratic Unconstrained Binary
Optimization (QUBO) problem.  Decision variables:

    x[i, j] ∈ {0, 1}  →  1 if task i is assigned to agent j

Objective (minimise total weighted cost):
    min  Σ_{i,j}  cost[i,j] * x[i,j]

Constraints encoded as penalty terms:

    1. Each task assigned at most once:
       P_1 * Σ_i  (1 - Σ_j x[i,j])^2

    2. Each agent's workload does not exceed capacity:
       P_2 * Σ_j  max(0, Σ_i duration[i] * x[i,j] - capacity[j])^2
       (linearised via slack variables or approximated directly)

    3. Skill compatibility (hard constraint – large penalty):
       P_3 * Σ_{(i,j) incompatible}  x[i,j]

The QUBO matrix Q is an (n*m) × (n*m) matrix where entry Q[a,b]
represents the coefficient of x_a * x_b in the objective.
For a = b this is a linear term (bias); for a ≠ b it is a quadratic
coupling.  The variable ordering is row-major: variable index
k = i * num_agents + j  corresponds to (task i, agent j).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.models import Agent, Task


@dataclass
class QUBOProblem:
    """Container for a QUBO problem ready to hand to a solver."""

    Q: np.ndarray                       # (N×N) symmetric QUBO matrix
    num_tasks: int
    num_agents: int
    variable_map: Dict[Tuple[int, int], int]   # (task_idx, agent_idx) -> var_idx
    reverse_map: Dict[int, Tuple[int, int]]    # var_idx -> (task_idx, agent_idx)
    task_ids: List[str]
    agent_ids: List[str]
    # For reference / debugging
    penalty_assignment: float = field(repr=False, default=0.0)
    penalty_capacity: float = field(repr=False, default=0.0)
    penalty_skills: float = field(repr=False, default=0.0)

    @property
    def num_variables(self) -> int:
        return self.num_tasks * self.num_agents

    def energy(self, bits: np.ndarray) -> float:
        """Evaluate QUBO energy for a binary solution vector."""
        return float(bits @ self.Q @ bits)

    def decode(self, bits: np.ndarray) -> Dict[str, str]:
        """Convert a binary vector to a task→agent assignment dict."""
        assignments: Dict[str, str] = {}
        for var_idx, (ti, aj) in self.reverse_map.items():
            if bits[var_idx] == 1:
                task_id = self.task_ids[ti]
                agent_id = self.agent_ids[aj]
                # If a task is assigned multiple times (infeasible) keep lowest cost
                if task_id not in assignments:
                    assignments[task_id] = agent_id
        return assignments


def build_qubo(
    tasks: List[Task],
    agents: List[Agent],
    *,
    penalty_assignment: float = 10.0,
    penalty_capacity: float = 8.0,
    penalty_skills: float = 20.0,
    cost_matrix: Optional[np.ndarray] = None,
) -> QUBOProblem:
    """
    Build the QUBO matrix for a task-assignment problem.

    Parameters
    ----------
    tasks:
        List of tasks to schedule.
    agents:
        List of agents available.
    penalty_assignment:
        Lagrange multiplier for the "assign each task exactly once" constraint.
    penalty_capacity:
        Lagrange multiplier for capacity constraints.
    penalty_skills:
        Lagrange multiplier for skill-mismatch (infeasible) assignments.
    cost_matrix:
        Optional (num_tasks × num_agents) matrix of base assignment costs.
        If None, a default cost based on task priority and duration is used.

    Returns
    -------
    QUBOProblem
        The assembled QUBO ready for a solver.
    """
    n = len(tasks)
    m = len(agents)
    N = n * m

    # Variable indexing: k = i*m + j  ↔  (task i, agent j)
    variable_map: Dict[Tuple[int, int], int] = {}
    reverse_map: Dict[int, Tuple[int, int]] = {}
    for i in range(n):
        for j in range(m):
            k = i * m + j
            variable_map[(i, j)] = k
            reverse_map[k] = (i, j)

    Q = np.zeros((N, N), dtype=float)

    # ── 1. Base cost terms (linear, diagonal) ─────────────────────────────
    if cost_matrix is not None:
        assert cost_matrix.shape == (n, m), (
            f"cost_matrix shape {cost_matrix.shape} != ({n}, {m})"
        )
        C = cost_matrix.astype(float)
    else:
        # Default cost: higher priority tasks should be assigned to agents
        # with the best skill match; penalise long durations on low-capacity agents.
        C = np.zeros((n, m), dtype=float)
        for i, task in enumerate(tasks):
            for j, agent in enumerate(agents):
                skill_overlap = len(set(task.required_skills) & set(agent.skills))
                max_skills = max(len(task.required_skills), 1)
                skill_score = skill_overlap / max_skills            # 0..1
                # Base cost: lower is better → invert skill score, weight by priority
                C[i, j] = task.duration * (1.0 - 0.8 * skill_score) / max(task.priority, 1e-3)

    for i in range(n):
        for j in range(m):
            k = variable_map[(i, j)]
            Q[k, k] += C[i, j]

    # ── 2. Assignment constraint: each task assigned exactly once ──────────
    # Penalty: P * (1 - Σ_j x[i,j])^2
    #        = P * (1 - 2 Σ_j x[i,j] + (Σ_j x[i,j])^2)
    # Linear term: -2P * x[i,j]  → add -2P to diagonal
    # Quadratic:   +2P * x[i,j1] * x[i,j2]  for j1 ≠ j2
    # Constant P is dropped (doesn't affect optimum)
    P1 = penalty_assignment
    for i in range(n):
        for j1 in range(m):
            k1 = variable_map[(i, j1)]
            Q[k1, k1] += -2.0 * P1          # linear part
            for j2 in range(j1 + 1, m):
                k2 = variable_map[(i, j2)]
                Q[k1, k2] += 2.0 * P1       # coupling (upper-triangular)

    # ── 3. Capacity constraint: Σ_i duration[i]*x[i,j] ≤ capacity[j] ─────
    # Approximation via quadratic penalty on overload:
    #   For each agent j: let W_j = Σ_i d_i * x[i,j]
    #   Penalty: P * max(0, W_j - cap_j)^2  ≈ P * (W_j - cap_j)^2 when W_j > cap_j
    # We relax this to a soft penalty P * (Σ_i d_i * x[i,j])^2
    # weighted to discourage over-capacity:
    #   P * (Σ_i d_i * x[i,j] / cap_j)^2  * I[W_j likely > cap_j]
    # Simpler tractable form: penalise pairwise co-assignments to same agent.
    P2 = penalty_capacity
    for j, agent in enumerate(agents):
        cap = agent.capacity
        for i1, task1 in enumerate(tasks):
            k1 = variable_map[(i1, j)]
            d1 = task1.duration
            # Self term: d1^2 / cap^2 * P2
            Q[k1, k1] += P2 * (d1 / max(cap, 1)) ** 2
            for i2 in range(i1 + 1, n):
                task2 = tasks[i2]
                k2 = variable_map[(i2, j)]
                d2 = task2.duration
                # Cross term: 2 * d1*d2 / cap^2 * P2
                Q[k1, k2] += 2.0 * P2 * (d1 * d2) / max(cap ** 2, 1)

    # ── 4. Skill mismatch penalty (hard) ──────────────────────────────────
    P3 = penalty_skills
    for i, task in enumerate(tasks):
        required = set(task.required_skills)
        for j, agent in enumerate(agents):
            available = set(agent.skills)
            if not required.issubset(available):
                k = variable_map[(i, j)]
                Q[k, k] += P3       # large diagonal penalty makes this unattractive

    # Symmetrise: QUBO solvers expect Q upper-triangular or symmetric
    # We keep it upper-triangular (standard QUBO form)
    return QUBOProblem(
        Q=Q,
        num_tasks=n,
        num_agents=m,
        variable_map=variable_map,
        reverse_map=reverse_map,
        task_ids=[t.id for t in tasks],
        agent_ids=[a.id for a in agents],
        penalty_assignment=P1,
        penalty_capacity=P2,
        penalty_skills=P3,
    )
