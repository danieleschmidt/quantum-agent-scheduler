"""
Hybrid quantum-classical task scheduler.

High-level API
--------------
    from quantum_scheduler import Scheduler, Task, Agent

    scheduler = Scheduler()
    result = scheduler.assign(tasks, agents, constraints)

The scheduler:
  1. Builds a QUBO formulation of the assignment problem
  2. Solves it with Simulated Annealing (classical) or D-Wave (if configured)
  3. Returns an AssignmentResult with the task→agent mapping and metadata
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .core.models import Agent, Task
from .qubo import QUBOProblem, build_qubo
from .solvers.simulated_annealing import SimulatedAnnealingSolver
from .solvers.dwave_stub import DWaveStubSolver
from .solvers.greedy import GreedySolver

logger = logging.getLogger(__name__)


@dataclass
class AssignmentResult:
    """Result of a scheduling run."""

    assignments: Dict[str, str]     # task_id → agent_id
    unassigned: List[str]           # task_ids that couldn't be assigned
    solver: str                     # name of the solver used
    qubo_energy: Optional[float]    # QUBO objective value (None for greedy)
    solve_time: float               # wall-clock seconds
    num_tasks: int
    num_agents: int
    # Extra info
    agent_workload: Dict[str, int] = field(default_factory=dict)  # agent_id → total duration
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def assignment_rate(self) -> float:
        """Fraction of tasks that were assigned."""
        if self.num_tasks == 0:
            return 1.0
        return len(self.assignments) / self.num_tasks

    def __str__(self) -> str:
        lines = [
            f"AssignmentResult [{self.solver}]",
            f"  Tasks:      {len(self.assignments)}/{self.num_tasks} assigned "
            f"({self.assignment_rate:.0%})",
            f"  Agents:     {self.num_agents}",
            f"  Solve time: {self.solve_time * 1000:.1f} ms",
        ]
        if self.qubo_energy is not None:
            lines.append(f"  QUBO energy: {self.qubo_energy:.4f}")
        lines.append("  Assignments:")
        for tid, aid in sorted(self.assignments.items()):
            wl = self.agent_workload.get(aid, "?")
            lines.append(f"    {tid} → {aid}  (agent load: {wl})")
        if self.unassigned:
            lines.append(f"  Unassigned: {self.unassigned}")
        return "\n".join(lines)


class Scheduler:
    """
    Hybrid quantum-classical task scheduler.

    Parameters
    ----------
    backend:
        Solver backend:
        - ``'sa'``     – Simulated Annealing (default, no dependencies)
        - ``'dwave'``  – D-Wave Ocean SDK sampler (falls back to SA if not installed)
        - ``'greedy'`` – Pure greedy baseline (no QUBO formulation)
    num_reads:
        Number of independent SA runs (more = better quality, slower).
    num_sweeps:
        SA sweeps per read.
    penalty_assignment:
        Lagrange penalty for "assign each task exactly once".
    penalty_capacity:
        Lagrange penalty for capacity overrun.
    penalty_skills:
        Lagrange penalty for skill-mismatch assignments.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        backend: str = "sa",
        num_reads: int = 20,
        num_sweeps: int = 2000,
        penalty_assignment: float = 10.0,
        penalty_capacity: float = 8.0,
        penalty_skills: float = 20.0,
        seed: Optional[int] = None,
    ):
        self.backend = backend.lower()
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.penalty_assignment = penalty_assignment
        self.penalty_capacity = penalty_capacity
        self.penalty_skills = penalty_skills
        self.seed = seed

    def assign(
        self,
        tasks: Sequence[Task],
        agents: Sequence[Agent],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> AssignmentResult:
        """
        Assign tasks to agents.

        Parameters
        ----------
        tasks:
            Tasks to schedule.
        agents:
            Available agents.
        constraints:
            Optional dict with keys:
            - ``skill_match_required`` (bool): only assign tasks to agents with
              all required skills.
            - ``cost_matrix`` (np.ndarray, shape n×m): override default costs.

        Returns
        -------
        AssignmentResult
        """
        tasks = list(tasks)
        agents = list(agents)
        constraints = constraints or {}

        if not tasks or not agents:
            return AssignmentResult(
                assignments={},
                unassigned=[t.id for t in tasks],
                solver=self.backend,
                qubo_energy=None,
                solve_time=0.0,
                num_tasks=len(tasks),
                num_agents=len(agents),
            )

        skill_match_required = constraints.get("skill_match_required", False)
        cost_matrix = constraints.get("cost_matrix", None)

        t0 = time.perf_counter()

        if self.backend == "greedy":
            raw = self._solve_greedy(tasks, agents, skill_match_required)
            qubo_energy = None
            solver_name = "greedy"
        else:
            qubo = build_qubo(
                tasks,
                agents,
                penalty_assignment=self.penalty_assignment,
                penalty_capacity=self.penalty_capacity,
                penalty_skills=self.penalty_skills,
                cost_matrix=cost_matrix,
            )
            raw_result = self._solve_qubo(qubo)
            raw = qubo.decode(raw_result.bits)
            qubo_energy = raw_result.energy
            solver_name = f"qubo-{self.backend}"

        solve_time = time.perf_counter() - t0

        # Post-process: enforce constraints, compute workload
        assignments, unassigned = self._postprocess(
            raw, tasks, agents, skill_match_required
        )
        agent_workload = self._compute_workload(assignments, tasks)

        logger.info(
            "Scheduler [%s]: %d/%d tasks assigned in %.1f ms",
            solver_name,
            len(assignments),
            len(tasks),
            solve_time * 1000,
        )

        return AssignmentResult(
            assignments=assignments,
            unassigned=unassigned,
            solver=solver_name,
            qubo_energy=qubo_energy,
            solve_time=solve_time,
            num_tasks=len(tasks),
            num_agents=len(agents),
            agent_workload=agent_workload,
        )

    def _solve_greedy(
        self, tasks: List[Task], agents: List[Agent], skill_match_required: bool
    ) -> Dict[str, str]:
        solver = GreedySolver(skill_match_required=skill_match_required)
        return solver.solve(tasks, agents)

    def _solve_qubo(self, qubo: QUBOProblem):
        if self.backend == "dwave":
            solver = DWaveStubSolver(
                use_qpu=False,
                num_reads=self.num_reads,
                seed=self.seed,
            )
        else:  # 'sa' or default
            solver = SimulatedAnnealingSolver(
                num_sweeps=self.num_sweeps,
                num_reads=self.num_reads,
                seed=self.seed,
            )
        return solver.solve(qubo)

    def _postprocess(
        self,
        raw: Dict[str, str],
        tasks: List[Task],
        agents: List[Agent],
        skill_match_required: bool,
    ):
        """Remove infeasible assignments and identify unassigned tasks."""
        agent_map = {a.id: a for a in agents}
        workload: Dict[str, int] = {a.id: 0 for a in agents}
        valid: Dict[str, str] = {}

        # Sort by priority to prioritise which tasks keep their assignment
        task_priority = {t.id: t.priority for t in tasks}
        task_duration = {t.id: t.duration for t in tasks}

        for tid, aid in sorted(raw.items(), key=lambda x: task_priority.get(x[0], 0), reverse=True):
            if aid not in agent_map:
                continue
            agent = agent_map[aid]
            duration = task_duration.get(tid, 0)

            # Check skill constraint
            if skill_match_required:
                task_obj = next((t for t in tasks if t.id == tid), None)
                if task_obj and not set(task_obj.required_skills).issubset(set(agent.skills)):
                    continue

            # Check capacity
            if workload[aid] + duration > agent.capacity:
                continue

            valid[tid] = aid
            workload[aid] += duration

        all_task_ids = {t.id for t in tasks}
        unassigned = sorted(all_task_ids - set(valid.keys()))
        return valid, unassigned

    def _compute_workload(
        self, assignments: Dict[str, str], tasks: List[Task]
    ) -> Dict[str, int]:
        duration_map = {t.id: t.duration for t in tasks}
        workload: Dict[str, int] = {}
        for tid, aid in assignments.items():
            workload[aid] = workload.get(aid, 0) + duration_map.get(tid, 0)
        return workload
