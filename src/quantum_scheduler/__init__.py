"""Quantum Agent Scheduler — hybrid quantum-classical task scheduler.

Core API
--------
    from quantum_scheduler import Scheduler, Task, Agent

    scheduler = Scheduler(backend='sa')
    result = scheduler.assign(tasks, agents, constraints)

QUBO internals
--------------
    from quantum_scheduler.qubo import build_qubo
    from quantum_scheduler.solvers import SimulatedAnnealingSolver
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from .core.models import Agent, Task, Solution, SchedulingProblem
from .scheduler import Scheduler, AssignmentResult
from .qubo import build_qubo, QUBOProblem
from .solvers import SimulatedAnnealingSolver, DWaveStubSolver, GreedySolver

__all__ = [
    # High-level
    "Scheduler",
    "AssignmentResult",
    # Data models
    "Agent",
    "Task",
    "Solution",
    "SchedulingProblem",
    # QUBO
    "build_qubo",
    "QUBOProblem",
    # Solvers
    "SimulatedAnnealingSolver",
    "DWaveStubSolver",
    "GreedySolver",
]
