"""
QuantumScheduler – legacy compatibility wrapper.

New code should use `quantum_scheduler.Scheduler` directly.
This module remains to avoid breaking imports from the old scaffolding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .models import Agent, Task, Solution, SchedulingProblem
from .exceptions import ValidationError, SolverError
from ..backends import Backend, ClassicalBackend
from ..constraints import Constraint

logger = logging.getLogger(__name__)


class QuantumScheduler:
    """
    Legacy high-level scheduler.  Prefer `quantum_scheduler.Scheduler` for new code.

    This class wraps the new `Scheduler` with the old interface:
        scheduler.schedule(agents, tasks, constraints) -> Solution
    """

    def __init__(
        self,
        backend: Union[str, Backend] = "auto",
        fallback: str = "classical",
        optimization_target: str = "minimize_cost",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        from ..scheduler import Scheduler

        # Map old backend names to new ones
        _backend_map = {
            "classical": "greedy",
            "auto": "sa",
            "hybrid": "sa",
            "quantum_sim": "sa",
        }
        new_backend = backend if isinstance(backend, str) else "sa"
        new_backend = _backend_map.get(new_backend, "sa")

        self._scheduler = Scheduler(backend=new_backend)
        self._fallback = fallback
        self._optimization_target = optimization_target
        self._timeout = timeout
        self._constraints: List[Constraint] = []

    def schedule(
        self,
        agents: List[Agent],
        tasks: List[Task],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Solution:
        """Schedule tasks to agents and return a Solution object."""
        result = self._scheduler.assign(tasks, agents, constraints)
        return Solution(
            assignments=result.assignments,
            cost=result.qubo_energy or 0.0,
            solver_type=result.solver,
            execution_time=result.solve_time,
        )

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "backend_type": self._scheduler.backend,
            "constraints_count": len(self._constraints),
            "optimization_target": self._optimization_target,
        }
