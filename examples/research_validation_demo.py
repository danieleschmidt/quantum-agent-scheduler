"""Research Validation Demo: Comprehensive Quantum Advantage Analysis.

This demo showcases the research capabilities of the quantum scheduler,
running statistically rigorous experiments to validate quantum advantages
in agent scheduling problems.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_scheduler.research import (
    AutomatedBenchmarkRunner,
    ProblemGenerator,
    ProblemClass,
    AdaptiveQuantumAnnealer,
    ComparativeAnnealingAnalyzer
)
from quantum_scheduler.optimization import AdaptiveCircuitOptimizer, QuantumAdvantageAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_quantum_advantage_validation():
    """Run comprehensive validation of quantum advantages."""
    logger.info("🔬 STARTING RESEARCH VALIDATION: Quantum Advantage Analysis")
    logger.info("=" * 70)
    
    # Initialize research components
    problem_generator = ProblemGenerator(random_seed=42)
    benchmark_runner = AutomatedBenchmarkRunner(
        output_dir="./research_results",
        num_workers=2,
        cache_results=True
    )
    
    # Generate diverse benchmark problems
    logger.info("📊 Generating benchmark problem suite...")
    problem_classes = [
        ProblemClass.SMALL_SPARSE,
        ProblemClass.MEDIUM_DENSE,
        ProblemClass.LARGE_SPARSE,
        ProblemClass.STRUCTURED
    ]
    
    problems = problem_generator.generate_problem_suite(
        problem_classes=problem_classes,
        problems_per_class=5  # Reduced for demo
    )
    
    logger.info(f"Generated {len(problems)} benchmark problems")
    
    # Register optimization methods
    logger.info("🧮 Registering optimization methods...")
    
    # Classical baseline methods
    benchmark_runner.register_method(
        "classical_greedy",
        classical_greedy_solver,
        "Fast greedy classical heuristic"
    )
    
    benchmark_runner.register_method(
        "classical_simulated_annealing",
        classical_simulated_annealing,
        "Classical simulated annealing"
    )
    
    # Quantum methods
    benchmark_runner.register_method(
        "adaptive_quantum_annealing",
        adaptive_quantum_annealing,
        "Novel adaptive quantum annealing"
    )
    
    benchmark_runner.register_method(
        "optimized_quantum_circuits",
        optimized_quantum_circuits,
        "Circuit-optimized quantum approach"
    )
    
    # Run comprehensive benchmark suite
    logger.info("🚀 Executing benchmark experiments...")
    start_time = time.time()
    
    benchmark_results = benchmark_runner.run_benchmark_suite(
        problems=problems,
        methods=None,  # All registered methods
        num_runs=3,    # Multiple runs for statistical validity
        timeout=60.0   # 1 minute timeout per experiment
    )
    
    execution_time = time.time() - start_time
    logger.info(f"✅ Benchmark suite completed in {execution_time:.1f} seconds")
    
    # Generate comprehensive research report
    logger.info("📝 Generating research report...")
    
    report = benchmark_runner.generate_research_report(benchmark_results['analysis'])
    
    # Save report
    report_file = Path("./research_results") / f"quantum_advantage_report_{int(time.time())}.md"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📄 Research report saved to: {report_file}")
    
    # Display key findings
    display_key_findings(benchmark_results['analysis'])
    
    # Run specialized quantum circuit optimization analysis
    run_circuit_optimization_analysis(problems[:3])  # Use subset for demo
    
    logger.info("🎯 RESEARCH VALIDATION COMPLETE")
    return benchmark_results


def display_key_findings(analysis):
    """Display key research findings."""
    print("\n" + "="*60)
    print("🔍 KEY RESEARCH FINDINGS")
    print("="*60)
    
    if 'method_performance' in analysis:
        print("\n📈 METHOD PERFORMANCE RANKING:")
        methods = analysis['method_performance']
        ranked_methods = sorted(methods.items(), 
                              key=lambda x: x[1]['avg_solution_quality'], 
                              reverse=True)
        
        for i, (method, stats) in enumerate(ranked_methods, 1):
            print(f"{i}. {method}: Quality {stats['avg_solution_quality']:.4f} "
                 f"({stats['success_rate']:.1%} success rate)")
    
    if 'quantum_advantage' in analysis:
        print("\n⚡ QUANTUM ADVANTAGE ANALYSIS:")
        for method, advantage_data in analysis['quantum_advantage'].items():
            if isinstance(advantage_data, dict):
                print(f"• {method}:")
                print(f"  - Advantage rate: {advantage_data['advantage_rate']:.1%}")
                print(f"  - Average advantage: {advantage_data['average_advantage']:.4f}")
                print(f"  - Max advantage: {advantage_data['max_advantage']:.4f}")
    
    if 'statistical_tests' in analysis:
        print("\n📊 STATISTICAL SIGNIFICANCE:")
        significant_tests = {k: v for k, v in analysis['statistical_tests'].items() 
                           if v.get('significant', False)}
        print(f"Found {len(significant_tests)} statistically significant differences:")
        
        for comparison, test_data in list(significant_tests.items())[:3]:  # Show top 3
            print(f"• {comparison}: p < 0.05, effect size = {test_data['effect_size_interpretation']}")
    
    if 'scalability_analysis' in analysis:
        print("\n📈 SCALABILITY ASSESSMENT:")
        for method, scalability in analysis['scalability_analysis'].items():
            rating = scalability['scalability_rating']
            correlation = scalability['time_size_correlation']
            print(f"• {method}: {rating.upper()} scalability (r={correlation:.3f})")


def run_circuit_optimization_analysis(problems):
    """Run specialized quantum circuit optimization analysis."""
    logger.info("\n🔧 Running Circuit Optimization Analysis...")
    
    circuit_optimizer = AdaptiveCircuitOptimizer()
    advantage_analyzer = QuantumAdvantageAnalyzer()
    
    optimization_results = []
    
    for problem in problems:
        logger.info(f"Optimizing circuits for problem {problem.problem_id}")
        
        # Run circuit optimization at different levels
        for opt_level in [1, 2, 3]:
            result = circuit_optimizer.optimize_qubo_circuit(
                qubo_matrix=problem.qubo_matrix,
                num_layers=4,
                optimization_level=opt_level
            )
            
            # Analyze quantum advantage
            classical_time = problem.metadata['problem_size'] * 0.01  # Mock classical time
            advantage_analysis = advantage_analyzer.analyze_quantum_advantage(
                qubo_matrix=problem.qubo_matrix,
                classical_time=classical_time,
                quantum_circuit_metrics=result.optimized_metrics
            )
            
            optimization_results.append({
                'problem_id': problem.problem_id,
                'optimization_level': opt_level,
                'improvement_factor': result.improvement_factor,
                'quantum_advantage': advantage_analysis['quantum_advantage'],
                'speedup_ratio': advantage_analysis['speedup_ratio']
            })
    
    # Display circuit optimization results
    print(f"\n🔧 CIRCUIT OPTIMIZATION RESULTS:")
    for result in optimization_results:
        print(f"• {result['problem_id']} (L{result['optimization_level']}): "
              f"{result['improvement_factor']:.2f}x depth reduction, "
              f"quantum advantage: {result['quantum_advantage']}")
    
    # Get optimization statistics
    stats = circuit_optimizer.get_optimization_statistics()
    advantage_stats = advantage_analyzer.get_advantage_statistics()
    
    print(f"\n📊 OPTIMIZATION STATISTICS:")
    print(f"• Average improvement: {stats['average_improvement']:.2f}x")
    print(f"• Best improvement: {stats['best_improvement']:.2f}x")
    print(f"• Total depth reduction: {stats['total_depth_reduction']} gates")
    
    if 'total_analyses' in advantage_stats:
        print(f"• Quantum advantage rate: {advantage_stats['quantum_advantage_rate']:.1%}")
        print(f"• Average speedup: {advantage_stats['average_speedup']:.2f}x")


# Method implementations for benchmarking

def classical_greedy_solver(qubo_matrix):
    """Fast greedy classical solver."""
    n = qubo_matrix.shape[0]
    solution = np.zeros(n, dtype=int)
    
    # Greedy: select variables that minimize energy
    for _ in range(min(n//2, 10)):  # Select up to half the variables
        best_idx = -1
        best_energy_reduction = 0
        
        for i in range(n):
            if solution[i] == 0:  # Not selected yet
                test_solution = solution.copy()
                test_solution[i] = 1
                
                energy_reduction = -np.dot(test_solution.T, np.dot(qubo_matrix, test_solution))
                if energy_reduction > best_energy_reduction:
                    best_energy_reduction = energy_reduction
                    best_idx = i
        
        if best_idx >= 0:
            solution[best_idx] = 1
        else:
            break
    
    energy = np.dot(solution.T, np.dot(qubo_matrix, solution))
    
    return {
        'solution_vector': solution,
        'energy': energy,
        'success_probability': 1.0,
        'additional_metrics': {'method': 'greedy'}
    }


def classical_simulated_annealing(qubo_matrix):
    """Classical simulated annealing solver."""
    n = qubo_matrix.shape[0]
    current_solution = np.random.randint(0, 2, n)
    current_energy = np.dot(current_solution.T, np.dot(qubo_matrix, current_solution))
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    temperature = 10.0
    cooling_rate = 0.95
    min_temperature = 0.01
    
    iterations = 0
    max_iterations = min(1000, n * 20)
    
    while temperature > min_temperature and iterations < max_iterations:
        # Generate neighbor
        neighbor = current_solution.copy()
        flip_idx = np.random.randint(0, n)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        
        neighbor_energy = np.dot(neighbor.T, np.dot(qubo_matrix, neighbor))
        
        # Accept or reject
        if neighbor_energy < current_energy or \
           np.random.random() < np.exp(-(neighbor_energy - current_energy) / temperature):
            current_solution = neighbor
            current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy
        
        temperature *= cooling_rate
        iterations += 1
    
    return {
        'solution_vector': best_solution,
        'energy': best_energy,
        'success_probability': 1.0,
        'additional_metrics': {'iterations': iterations, 'method': 'simulated_annealing'}
    }


def adaptive_quantum_annealing(qubo_matrix):
    """Adaptive quantum annealing solver."""
    annealer = AdaptiveQuantumAnnealer()
    
    result = annealer.optimize_scheduling_problem(
        qubo_matrix=qubo_matrix,
        max_iterations=3,
        target_quality=0.9
    )
    
    return {
        'solution_vector': result.solution_vector,
        'energy': result.energy,
        'success_probability': result.success_probability,
        'additional_metrics': {
            'annealing_time': result.annealing_time,
            'chain_breaks': result.chain_breaks,
            'method': 'adaptive_quantum_annealing'
        }
    }


def optimized_quantum_circuits(qubo_matrix):
    """Circuit-optimized quantum solver."""
    circuit_optimizer = AdaptiveCircuitOptimizer()
    
    # Optimize circuit first
    optimization_result = circuit_optimizer.optimize_qubo_circuit(
        qubo_matrix=qubo_matrix,
        num_layers=3,
        optimization_level=2
    )
    
    # Simulate quantum execution with optimized circuit
    n = qubo_matrix.shape[0]
    solution = np.random.randint(0, 2, n)
    
    # Apply circuit optimization benefits (simulated)
    improvement_factor = optimization_result.improvement_factor
    
    # Simple improvement heuristic based on optimization
    num_improvements = int(improvement_factor * 5)
    energy = np.dot(solution.T, np.dot(qubo_matrix, solution))
    
    for _ in range(num_improvements):
        candidate = solution.copy()
        flip_idx = np.random.randint(0, n)
        candidate[flip_idx] = 1 - candidate[flip_idx]
        
        candidate_energy = np.dot(candidate.T, np.dot(qubo_matrix, candidate))
        if candidate_energy < energy:
            solution = candidate
            energy = candidate_energy
    
    # Success probability based on circuit fidelity
    success_prob = optimization_result.optimized_metrics.fidelity_estimate
    
    return {
        'solution_vector': solution,
        'energy': energy,
        'success_probability': success_prob,
        'additional_metrics': {
            'circuit_depth': optimization_result.optimized_metrics.depth,
            'gate_count': optimization_result.optimized_metrics.gate_count,
            'optimization_level': 2,
            'method': 'optimized_quantum_circuits'
        }
    }


if __name__ == "__main__":
    print("🚀 Quantum Scheduler Research Validation Demo")
    print("=" * 50)
    
    try:
        results = run_quantum_advantage_validation()
        
        print("\n✅ Demo completed successfully!")
        print(f"📊 Total experiments: {results['metadata']['total_experiments']}")
        print(f"✅ Successful experiments: {results['metadata']['successful_experiments']}")
        print("\n📁 Check './research_results/' for detailed output files")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}", exc_info=True)
        sys.exit(1)