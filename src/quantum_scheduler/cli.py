"""Command-line interface for quantum scheduler."""

import click
import json
import time
from typing import List, Dict, Any
from pathlib import Path

from .core import QuantumScheduler, Agent, Task
from .core.models import Priority


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """Quantum Agent Scheduler CLI."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option('--backend', '-b', default='auto', help='Backend to use (auto, classical, quantum)')
@click.option('--agents-file', '-a', required=True, help='JSON file containing agent definitions')
@click.option('--tasks-file', '-t', required=True, help='JSON file containing task definitions')
@click.option('--constraints-file', '-c', help='JSON file containing constraints')
@click.option('--output', '-o', help='Output file for solution (JSON format)')
def schedule(backend: str, agents_file: str, tasks_file: str, constraints_file: str, output: str):
    """Schedule tasks to agents using quantum optimization."""
    try:
        # Load agents
        with open(agents_file, 'r') as f:
            agents_data = json.load(f)
        agents = [Agent(**agent_data) for agent_data in agents_data]
        
        # Load tasks
        with open(tasks_file, 'r') as f:
            tasks_data = json.load(f)
        tasks = [Task(**task_data) for task_data in tasks_data]
        
        # Load constraints if provided
        constraints = {}
        if constraints_file:
            with open(constraints_file, 'r') as f:
                constraints = json.load(f)
        
        # Initialize scheduler
        scheduler = QuantumScheduler(backend=backend)
        
        # Solve
        click.echo(f"Solving scheduling problem with {len(agents)} agents and {len(tasks)} tasks...")
        start_time = time.time()
        solution = scheduler.schedule(agents, tasks, constraints)
        solve_time = time.time() - start_time
        
        # Display results
        click.echo(f"✅ Solution found in {solve_time:.2f}s using {solution.solver_type}")
        click.echo(f"📊 Assignments: {len(solution.assignments)}")
        click.echo(f"💰 Total cost: {solution.cost:.2f}")
        click.echo(f"⚡ Utilization: {solution.utilization_ratio:.2%}")
        
        # Show assignments
        if solution.assignments:
            click.echo("\n📋 Task Assignments:")
            for task_id, agent_id in solution.assignments.items():
                click.echo(f"  {task_id} → {agent_id}")
        
        # Save output if requested
        if output:
            output_data = {
                "assignments": solution.assignments,
                "cost": solution.cost,
                "solver_type": solution.solver_type,
                "execution_time": solution.execution_time,
                "metadata": {
                    "total_agents": len(agents),
                    "total_tasks": len(tasks),
                    "total_assignments": solution.total_assignments,
                    "utilization_ratio": solution.utilization_ratio
                }
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"\n💾 Solution saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--num-agents', '-a', default=10, help='Number of agents to generate')
@click.option('--num-tasks', '-t', default=20, help='Number of tasks to generate')
@click.option('--output-dir', '-o', default='.', help='Output directory for generated files')
def generate(num_agents: int, num_tasks: int, output_dir: str):
    """Generate sample agents and tasks for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate agents
    skills_pool = ["python", "java", "javascript", "ml", "web", "mobile", "devops", "database"]
    agents = []
    
    for i in range(1, num_agents + 1):
        agent = {
            "id": f"agent{i}",
            "skills": list(set([
                skills_pool[i % len(skills_pool)],
                skills_pool[(i * 2) % len(skills_pool)]
            ])),
            "capacity": 2 + (i % 4)  # Capacity between 2-5
        }
        agents.append(agent)
    
    # Generate tasks
    tasks = []
    priorities = [p.value for p in Priority]
    
    for i in range(1, num_tasks + 1):
        task = {
            "id": f"task{i}",
            "required_skills": [skills_pool[i % len(skills_pool)]],
            "duration": 1 + (i % 3),  # Duration between 1-3
            "priority": priorities[i % len(priorities)]
        }
        tasks.append(task)
    
    # Save files
    agents_file = output_path / "agents.json"
    tasks_file = output_path / "tasks.json"
    constraints_file = output_path / "constraints.json"
    
    with open(agents_file, 'w') as f:
        json.dump(agents, f, indent=2)
    
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    # Generate sample constraints
    constraints = {
        "skill_match_required": True,
        "max_concurrent_tasks": 2,
        "deadline_enforcement": False
    }
    
    with open(constraints_file, 'w') as f:
        json.dump(constraints, f, indent=2)
    
    click.echo(f"✅ Generated {num_agents} agents and {num_tasks} tasks")
    click.echo(f"📁 Files saved to: {output_path.absolute()}")
    click.echo(f"  - {agents_file.name}")
    click.echo(f"  - {tasks_file.name}")
    click.echo(f"  - {constraints_file.name}")


@cli.command()
@click.option('--backend', '-b', help='Backend to benchmark')
@click.option('--max-size', default=100, help='Maximum problem size to test')
def benchmark(backend: str, max_size: int):
    """Benchmark scheduler performance across problem sizes."""
    backends_to_test = [backend] if backend else ['classical', 'auto']
    problem_sizes = [10, 25, 50, 100, max_size] if max_size > 100 else [10, 25, 50, max_size]
    
    click.echo("🚀 Starting scheduler benchmark...")
    click.echo(f"Backends: {backends_to_test}")
    click.echo(f"Problem sizes: {problem_sizes}")
    
    results = []
    
    for backend_name in backends_to_test:
        for size in problem_sizes:
            # Generate problem
            agents = [
                Agent(id=f"agent{i}", skills=["python", "ml"], capacity=3)
                for i in range(size // 2)
            ]
            tasks = [
                Task(id=f"task{i}", required_skills=["python"], duration=2, priority=5)
                for i in range(size)
            ]
            
            # Benchmark
            scheduler = QuantumScheduler(backend=backend_name)
            start_time = time.time()
            
            try:
                solution = scheduler.schedule(agents, tasks)
                solve_time = time.time() - start_time
                
                result = {
                    "backend": backend_name,
                    "size": size,
                    "agents": len(agents),
                    "tasks": len(tasks),
                    "solve_time": solve_time,
                    "assignments": len(solution.assignments),
                    "cost": solution.cost,
                    "success": True
                }
                
                click.echo(f"✅ {backend_name:10} | Size: {size:3} | Time: {solve_time:.3f}s | Assignments: {len(solution.assignments)}")
                
            except Exception as e:
                result = {
                    "backend": backend_name,
                    "size": size,
                    "agents": len(agents),
                    "tasks": len(tasks),
                    "solve_time": None,
                    "assignments": 0,
                    "cost": float('inf'),
                    "error": str(e),
                    "success": False
                }
                
                click.echo(f"❌ {backend_name:10} | Size: {size:3} | Error: {e}")
            
            results.append(result)
    
    # Summary
    click.echo("\n📊 Benchmark Summary:")
    for backend_name in backends_to_test:
        backend_results = [r for r in results if r["backend"] == backend_name and r["success"]]
        if backend_results:
            avg_time = sum(r["solve_time"] for r in backend_results) / len(backend_results)
            max_size_solved = max(r["size"] for r in backend_results)
            click.echo(f"  {backend_name}: Avg time: {avg_time:.3f}s, Max size: {max_size_solved}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()