"""
Greedy task scheduler – classical baseline for comparison.

Assignment strategy:
  1. Sort tasks by priority (descending)
  2. For each task, assign to the agent with:
     - Required skills (if skill_match_required)
     - Remaining capacity ≥ task.duration
     - Highest skill overlap, then most remaining capacity
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..core.models import Agent, Task


class GreedySolver:
    """Simple priority-greedy task assignment solver."""

    def __init__(self, skill_match_required: bool = False):
        self.skill_match_required = skill_match_required

    def solve(
        self,
        tasks: List[Task],
        agents: List[Agent],
    ) -> Dict[str, str]:
        """
        Assign tasks to agents greedily.

        Returns
        -------
        dict
            Mapping task_id → agent_id for all assigned tasks.
        """
        t0 = time.perf_counter()
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        workload: Dict[str, int] = {a.id: 0 for a in agents}
        assignments: Dict[str, str] = {}

        for task in sorted_tasks:
            best_agent: Optional[Agent] = None
            best_score = float("-inf")

            for agent in agents:
                if self.skill_match_required:
                    if not set(task.required_skills).issubset(set(agent.skills)):
                        continue
                remaining = agent.capacity - workload[agent.id]
                if remaining < task.duration:
                    continue
                skill_overlap = len(set(task.required_skills) & set(agent.skills))
                score = skill_overlap * 1000 + remaining
                if score > best_score:
                    best_score = score
                    best_agent = agent

            if best_agent is not None:
                assignments[task.id] = best_agent.id
                workload[best_agent.id] += task.duration

        self.last_solve_time = time.perf_counter() - t0
        return assignments
