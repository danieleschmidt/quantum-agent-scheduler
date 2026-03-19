# quantum-agent-scheduler

A hybrid quantum-classical task scheduler that formulates agent task assignment as **QUBO (Quadratic Unconstrained Binary Optimization)** problems.

## What it does

Given a set of tasks and a set of agents (with skills and capacity constraints), the scheduler finds an optimal task-to-agent assignment.  The core novelty is the **QUBO formulation** — the same mathematical structure used by quantum annealers (D-Wave) and quantum approximate optimization algorithms (QAOA).

The primary backend is **Simulated Annealing** (no quantum hardware needed), with a clean integration path to D-Wave Ocean SDK when available.

---

## QUBO Formulation

### Decision variables

$$x_{ij} \in \{0, 1\}, \quad i \in \text{tasks},\ j \in \text{agents}$$

$x_{ij} = 1$ means task $i$ is assigned to agent $j$.

### Objective

Minimize total weighted assignment cost:

$$\min \sum_{i,j} C_{ij} \cdot x_{ij}$$

where $C_{ij}$ is the cost of assigning task $i$ to agent $j$ (lower for better skill matches and higher priority tasks).

### Constraints (encoded as penalty terms)

**1. Each task assigned at most once:**

$$\mathcal{P}_1 \cdot \sum_i \left(1 - \sum_j x_{ij}\right)^2$$

**2. Agent capacity:**

$$\mathcal{P}_2 \cdot \sum_j \sum_{i_1 < i_2} \frac{2 \cdot d_{i_1} d_{i_2}}{c_j^2} \cdot x_{i_1 j} \cdot x_{i_2 j}$$

where $d_i$ is task duration and $c_j$ is agent capacity.

**3. Skill mismatch (hard):**

$$\mathcal{P}_3 \cdot \sum_{(i,j)\ \text{incompatible}} x_{ij}$$

### QUBO matrix

All terms are assembled into an $(n \cdot m) \times (n \cdot m)$ symmetric matrix $Q$, with variable ordering $k = i \cdot m + j$.  The energy of a binary vector $\mathbf{x}$ is:

$$E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$$

A quantum annealer (or classical SA) finds the $\mathbf{x}$ that minimises $E$.

---

## Quick start

```python
from quantum_scheduler import Scheduler, Task, Agent

tasks = [
    Task(id="ml_training",  required_skills=["python", "ml"], duration=4, priority=9.0),
    Task(id="api_server",   required_skills=["python", "web"], duration=2, priority=7.0),
    Task(id="db_migration", required_skills=["sql"],           duration=1, priority=8.0),
]

agents = [
    Agent(id="Alice", skills=["python", "ml", "web"], capacity=10),
    Agent(id="Bob",   skills=["python", "sql"],       capacity=8),
]

scheduler = Scheduler(backend="sa")   # Simulated Annealing on QUBO
result = scheduler.assign(tasks, agents, constraints={"skill_match_required": True})
print(result)
```

Output:
```
AssignmentResult [qubo-sa]
  Tasks:      3/3 assigned (100%)
  Agents:     2
  Solve time: 450.3 ms
  QUBO energy: -18.2500
  Assignments:
    api_server   → Alice  (agent load: 6)
    db_migration → Bob    (agent load: 1)
    ml_training  → Alice  (agent load: 4)
```

---

## Backends

| Backend | Description | Requires |
|---------|-------------|----------|
| `"sa"` | Simulated Annealing on QUBO *(default)* | numpy only |
| `"greedy"` | Priority-greedy baseline (no QUBO) | nothing |
| `"dwave"` | D-Wave Ocean SDK sampler | `pip install dwave-ocean-sdk` |

The `"dwave"` backend gracefully falls back to built-in SA if the Ocean SDK is not installed.

### D-Wave / QPU

```python
from quantum_scheduler.solvers import DWaveStubSolver

solver = DWaveStubSolver(use_qpu=True, num_reads=100)
# Requires: pip install dwave-ocean-sdk + DWAVE_API_TOKEN env var
```

---

## Scheduler parameters

```python
Scheduler(
    backend="sa",             # 'sa', 'greedy', 'dwave'
    num_reads=20,             # Independent SA runs (more = better quality)
    num_sweeps=2000,          # SA sweeps per read
    penalty_assignment=10.0,  # λ for "assign each task exactly once"
    penalty_capacity=8.0,     # λ for capacity overrun
    penalty_skills=20.0,      # λ for skill mismatch
    seed=None,                # Reproducibility
)
```

---

## Low-level QUBO API

```python
from quantum_scheduler.qubo import build_qubo
from quantum_scheduler.solvers import SimulatedAnnealingSolver

qubo = build_qubo(tasks, agents, penalty_assignment=10, penalty_capacity=8, penalty_skills=20)
print(f"Variables: {qubo.num_variables} ({qubo.num_tasks} tasks × {qubo.num_agents} agents)")

solver = SimulatedAnnealingSolver(num_sweeps=3000, num_reads=20, seed=42)
result = solver.solve(qubo)
assignments = qubo.decode(result.bits)
# {'ml_training': 'Alice', 'api_server': 'Alice', 'db_migration': 'Bob'}
```

---

## Demo

```bash
python examples/demo_10tasks_4agents.py
```

Schedules 10 real-world-flavoured tasks (data pipeline + API + ML deployment) across 4 agents with skill and capacity constraints.  Compares greedy vs QUBO-SA.

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Installation

```bash
pip install -e .
# Optional D-Wave support:
pip install dwave-ocean-sdk
```

---

## Structure

```
src/quantum_scheduler/
├── __init__.py          # Public API
├── scheduler.py         # Main Scheduler + AssignmentResult
├── qubo.py              # QUBO formulation
├── solvers/
│   ├── simulated_annealing.py  # Primary classical solver
│   ├── dwave_stub.py           # D-Wave / Ocean SDK integration
│   └── greedy.py               # Greedy baseline
├── core/
│   ├── models.py        # Task, Agent, Solution dataclasses
│   └── exceptions.py    # Exception hierarchy
├── backends/            # Legacy backend compatibility layer
└── constraints/         # Constraint base classes
tests/
├── test_qubo.py         # QUBO matrix properties
├── test_solvers.py      # Solver behaviour
└── test_scheduler.py    # End-to-end scheduler tests
examples/
└── demo_10tasks_4agents.py
```

---

## Why QUBO?

QUBO is the native problem format for quantum annealers.  By formulating task scheduling as QUBO, this scheduler can run on:
- Classical hardware (Simulated Annealing, this repo's default)
- D-Wave quantum annealers (via Ocean SDK)
- Gate-model quantum computers via QAOA (future)

The formulation is exact — no heuristic pre-processing strips away the constraint structure.  The penalty parameters ($\mathcal{P}_1, \mathcal{P}_2, \mathcal{P}_3$) control the trade-off between constraint satisfaction and objective minimisation.
