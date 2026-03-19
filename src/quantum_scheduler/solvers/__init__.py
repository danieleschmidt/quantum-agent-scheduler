"""Solver backends for QUBO problems."""

from .simulated_annealing import SimulatedAnnealingSolver
from .dwave_stub import DWaveStubSolver
from .greedy import GreedySolver

__all__ = ["SimulatedAnnealingSolver", "DWaveStubSolver", "GreedySolver"]
