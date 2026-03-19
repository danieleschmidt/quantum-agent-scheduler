#!/usr/bin/env python3
"""
Demo: Schedule 10 tasks across 4 agents with dependencies and capacity constraints.
Compare greedy assignment vs QUBO Simulated Annealing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_scheduler import Scheduler, Task, Agent

# ── Problem definition ────────────────────────────────────────────────────────

TASKS = [
    Task(id="data_ingest",    required_skills=["python"],           duration=2,  priority=8.0),
    Task(id="preprocess",     required_skills=["python", "ml"],     duration=3,  priority=7.0,
         dependencies=["data_ingest"]),
    Task(id="feature_eng",    required_skills=["ml", "python"],     duration=2,  priority=6.0,
         dependencies=["preprocess"]),
    Task(id="train_model",    required_skills=["ml"],               duration=4,  priority=9.0,
         dependencies=["feature_eng"]),
    Task(id="api_server",     required_skills=["python", "web"],    duration=3,  priority=7.0),
    Task(id="frontend",       required_skills=["web", "javascript"],duration=2,  priority=5.0,
         dependencies=["api_server"]),
    Task(id="db_migration",   required_skills=["sql"],              duration=1,  priority=8.0),
    Task(id="deploy_infra",   required_skills=["devops"],           duration=2,  priority=6.0,
         dependencies=["db_migration"]),
    Task(id="model_serve",    required_skills=["python", "devops"], duration=2,  priority=9.0,
         dependencies=["train_model", "deploy_infra"]),
    Task(id="integration_qa", required_skills=["python", "web"],   duration=1,  priority=5.0,
         dependencies=["api_server", "model_serve"]),
]

AGENTS = [
    Agent(id="Alice",  skills=["python", "ml", "web"],               capacity=10),
    Agent(id="Bob",    skills=["python", "sql", "devops"],            capacity=10),
    Agent(id="Carol",  skills=["web", "javascript", "python"],        capacity=8),
    Agent(id="Dave",   skills=["ml", "python", "devops", "sql"],     capacity=12),
]

CONSTRAINTS = {
    "skill_match_required": True,
}


def print_assignments(result, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Solver    : {result.solver}")
    print(f"  Assigned  : {len(result.assignments)}/{result.num_tasks}")
    if result.qubo_energy is not None:
        print(f"  QUBO Energy: {result.qubo_energy:.4f}")
    print(f"  Solve time: {result.solve_time*1000:.1f} ms")
    print()
    print(f"  {'Task':<20} {'Agent':<10} {'Skills Needed':<30}")
    print(f"  {'-'*20} {'-'*10} {'-'*30}")
    task_map = {t.id: t for t in TASKS}
    for tid in sorted(result.assignments.keys()):
        aid = result.assignments[tid]
        skills = ", ".join(task_map[tid].required_skills)
        print(f"  {tid:<20} {aid:<10} {skills}")
    if result.unassigned:
        print(f"\n  ⚠ Unassigned: {result.unassigned}")
    print()
    print("  Agent workloads:")
    for agent in AGENTS:
        wl = result.agent_workload.get(agent.id, 0)
        bar = "█" * wl + "░" * max(0, agent.capacity - wl)
        print(f"    {agent.id:<8} [{bar}] {wl}/{agent.capacity}")


if __name__ == "__main__":
    print("\nQuantum Agent Scheduler — Demo")
    print("10 tasks, 4 agents, skill constraints, capacity constraints")
    print()

    # ── Greedy baseline ───────────────────────────────────────────────────────
    greedy = Scheduler(backend="greedy")
    r_greedy = greedy.assign(TASKS, AGENTS, CONSTRAINTS)
    print_assignments(r_greedy, "Greedy Baseline")

    # ── QUBO + Simulated Annealing ────────────────────────────────────────────
    sa = Scheduler(
        backend="sa",
        num_reads=30,
        num_sweeps=3000,
        penalty_assignment=12.0,
        penalty_capacity=10.0,
        penalty_skills=25.0,
        seed=42,
    )
    r_sa = sa.assign(TASKS, AGENTS, CONSTRAINTS)
    print_assignments(r_sa, "QUBO — Simulated Annealing")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Greedy':>10}  {'QUBO-SA':>10}")
    print(f"  {'-'*25} {'-'*10}  {'-'*10}")
    print(f"  {'Tasks assigned':<25} {len(r_greedy.assignments):>10}  {len(r_sa.assignments):>10}")
    print(f"  {'Assignment rate':<25} {r_greedy.assignment_rate:>9.0%}  {r_sa.assignment_rate:>9.0%}")
    print(f"  {'Solve time (ms)':<25} {r_greedy.solve_time*1000:>9.1f}  {r_sa.solve_time*1000:>9.1f}")
    e_sa = f"{r_sa.qubo_energy:.2f}" if r_sa.qubo_energy is not None else "N/A"
    print(f"  {'QUBO energy':<25} {'N/A':>10}  {e_sa:>10}")
    print()
