"""Quantum Agent Scheduler - Hybrid classical-quantum scheduler for multi-agent systems."""

__version__ = "0.1.0"
__author__ = "Your Organization"
__email__ = "quantum-ai@your-org.com"

from .core import QuantumScheduler, Agent, Task, Solution
from .backends import Backend
from .constraints import Constraint

__all__ = [
    "QuantumScheduler",
    "Agent", 
    "Task",
    "Solution",
    "Backend",
    "Constraint",
]