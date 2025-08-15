#!/usr/bin/env python3
"""
Quantum Advantage Research Demonstration

This script demonstrates the advanced quantum advantage prediction and
adaptive QUBO optimization capabilities implemented for research purposes.
"""

import asyncio
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import research modules
try:
    from quantum_scheduler.research.quantum_advantage_predictor import (
        QuantumAdvantagePredictor, 
        ProblemFeatures,
        QuantumAdvantageExperiment
    )
    from quantum_scheduler.research.adaptive_qubo_optimizer import (
        AdaptiveQUBOOptimizer,
        ProblemContext,
        AlgorithmType
    )
    from quantum_scheduler.research.comparative_analysis_framework import (
        ComparativeAnalysisFramework,
        ExperimentConfig,
        ProblemCategory
    )
except ImportError as e:
    logger.error(f"Failed to import research modules: {e}")
    logger.error("Please ensure the quantum_scheduler package is properly installed")
    exit(1)


class ResearchDemo:
    """Comprehensive research demonstration framework."""
    
    def __init__(self, output_dir: str = "research_demo_results"):
        """Initialize research demo.
        
        Args:
            output_dir: Directory for saving research outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize research components
        self.advantage_predictor = QuantumAdvantagePredictor(
            model_cache_path=self.output_dir / "models",
            min_training_samples=20
        )
        
        self.adaptive_optimizer = AdaptiveQUBOOptimizer()
        
        self.analysis_framework = ComparativeAnalysisFramework(
            output_directory=self.output_dir / "comparative_analysis"
        )
        
        logger.info(f"Research demo initialized with output directory: {self.output_dir}")
    
    def generate_research_problem(self, size: int, problem_type: str = "random") -> Dict[str, Any]:
        """Generate a research problem for testing.
        
        Args:
            size: Problem size (number of variables)
            problem_type: Type of problem ("random", "structured", "sparse")
            
        Returns:
            Dictionary containing problem data
        """
        np.random.seed(int(time.time()) % 1000)  # Semi-random seed
        
        if problem_type == "structured":
            # Create structured QUBO (block diagonal-like)
            qubo = np.zeros((size, size))
            block_size = max(1, size // 4)
            
            for i in range(0, size, block_size):
                end = min(i + block_size, size)
                # Strong intra-block connections
                for j in range(i, end):
                    for k in range(i, end):
                        if j != k:
                            qubo[j, k] = np.random.uniform(1, 3)
                    qubo[j, j] = np.random.uniform(-5, -1)  # Diagonal penalty
                
                # Weak inter-block connections
                if end < size:
                    next_end = min(end + block_size, size)
                    for j in range(i, end):
                        for k in range(end, next_end):
                            if np.random.random() < 0.3:
                                value = np.random.uniform(-1, 1)
                                qubo[j, k] = value
                                qubo[k, j] = value
        
        elif problem_type == "sparse":
            # Create sparse QUBO
            density = 0.1 + 0.2 * np.random.random()  # 10-30% density
            qubo = np.zeros((size, size))
            
            # Add diagonal terms
            np.fill_diagonal(qubo, np.random.uniform(-10, 10, size))
            
            # Add sparse off-diagonal terms
            num_connections = int(size * size * density) - size
            for _ in range(num_connections):
                i, j = np.random.randint(0, size, 2)
                if i != j:
                    value = np.random.uniform(-3, 3)
                    qubo[i, j] = value
                    qubo[j, i] = value
        
        else:  # "random"
            # Create random dense QUBO
            qubo = np.random.uniform(-5, 5, (size, size))
            qubo = (qubo + qubo.T) / 2  # Make symmetric
        
        # Generate agents and tasks
        num_agents = max(1, size // 4)
        num_tasks = size - num_agents
        
        skills_pool = ['python', 'java', 'ml', 'web', 'data', 'mobile', 'devops', 'design', 'ai', 'quantum']
        
        agents = []
        for i in range(num_agents):
            num_skills = np.random.randint(1, 4)
            skills = list(np.random.choice(skills_pool, num_skills, replace=False))
            capacity = np.random.randint(1, 4)
            agents.append({
                'id': f'agent_{i}',
                'skills': skills,
                'capacity': capacity
            })
        
        tasks = []
        for i in range(num_tasks):
            num_required = np.random.randint(1, 3)
            required_skills = list(np.random.choice(skills_pool, num_required, replace=False))
            duration = np.random.randint(1, 6)
            priority = np.random.uniform(1, 10)
            tasks.append({
                'id': f'task_{i}',
                'required_skills': required_skills,
                'duration': duration,
                'priority': priority
            })
        
        return {
            'qubo_matrix': qubo,
            'agents': agents,
            'tasks': tasks,
            'problem_type': problem_type,
            'problem_size': size,
            'metadata': {
                'generation_time': time.time(),
                'density': np.count_nonzero(qubo) / (size * size)
            }
        }
    
    def simulate_quantum_execution(self, qubo_matrix: np.ndarray, backend: str = "quantum_sim") -> tuple:
        """Simulate quantum algorithm execution.
        
        Args:
            qubo_matrix: QUBO problem matrix
            backend: Quantum backend identifier
            
        Returns:
            Tuple of (solution, energy, execution_time)
        """
        size = qubo_matrix.shape[0]
        start_time = time.time()
        
        if backend == "quantum_hw":
            # Simulate quantum hardware with setup overhead but better scaling
            if size < 30:
                execution_time = np.random.uniform(10, 20)  # High setup cost
                solution_quality = np.random.uniform(0.7, 0.9)
            else:
                # Better scaling for larger problems
                execution_time = size * np.random.uniform(0.05, 0.15)
                solution_quality = np.random.uniform(0.8, 0.95)
        
        elif backend == "quantum_sim":
            # Quantum simulator - moderate overhead, decent scaling
            execution_time = size * np.random.uniform(0.1, 0.3)
            solution_quality = np.random.uniform(0.75, 0.92)
        
        else:  # Default quantum simulation
            execution_time = size * np.random.uniform(0.08, 0.25)
            solution_quality = np.random.uniform(0.8, 0.9)
        
        # Generate quantum-inspired solution
        try:
            # Use eigenvalue decomposition to simulate quantum behavior
            eigenvals, eigenvecs = np.linalg.eigh(qubo_matrix)
            dominant_eigenvec = eigenvecs[:, np.argmax(np.abs(eigenvals))]
            
            # Convert to probabilistic binary solution
            probabilities = np.abs(dominant_eigenvec) / np.sum(np.abs(dominant_eigenvec))
            solution = np.random.binomial(1, probabilities)
            
            # Add some quantum noise/improvement
            for _ in range(min(5, size // 4)):
                flip_idx = np.random.randint(0, size)
                test_solution = solution.copy()
                test_solution[flip_idx] = 1 - test_solution[flip_idx]
                
                test_energy = np.dot(test_solution, np.dot(qubo_matrix, test_solution))
                current_energy = np.dot(solution, np.dot(qubo_matrix, solution))
                
                if test_energy < current_energy:  # Improvement
                    solution = test_solution
            
        except:
            # Fallback to random solution
            solution = np.random.randint(0, 2, size)
        
        energy = np.dot(solution, np.dot(qubo_matrix, solution))
        
        # Simulate execution time
        time.sleep(min(0.01, execution_time / 1000))  # Small delay for realism
        actual_execution_time = time.time() - start_time
        
        return solution, energy, execution_time  # Return simulated time, not actual
    
    def simulate_classical_execution(self, qubo_matrix: np.ndarray) -> tuple:
        """Simulate classical algorithm execution.
        
        Args:
            qubo_matrix: QUBO problem matrix
            
        Returns:
            Tuple of (solution, energy, execution_time)
        """
        size = qubo_matrix.shape[0]
        start_time = time.time()
        
        # Simulate classical greedy algorithm
        solution = np.zeros(size, dtype=int)
        remaining_indices = list(range(size))
        
        # Classical scaling: approximately O(n^2)
        execution_time = (size ** 1.8) * np.random.uniform(0.001, 0.003)
        
        # Greedy selection
        while remaining_indices:
            best_improvement = float('-inf')
            best_index = None
            
            for i in remaining_indices:
                # Calculate energy change if we set x_i = 1
                energy_change = qubo_matrix[i, i]  # Diagonal term
                
                # Add interaction terms with already selected variables
                for j in range(size):
                    if solution[j] == 1 and j != i:
                        energy_change += qubo_matrix[i, j] + qubo_matrix[j, i]
                
                improvement = -energy_change  # We want to minimize energy
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_index = i
            
            if best_index is not None and best_improvement > 0:
                solution[best_index] = 1
                remaining_indices.remove(best_index)
            else:
                break
        
        energy = np.dot(solution, np.dot(qubo_matrix, solution))
        
        # Add small delay for realism
        time.sleep(min(0.005, execution_time / 2000))
        actual_execution_time = time.time() - start_time
        
        return solution, energy, execution_time  # Return simulated time
    
    def demonstrate_quantum_advantage_prediction(self, num_problems: int = 25) -> None:
        """Demonstrate quantum advantage prediction capabilities.
        
        Args:
            num_problems: Number of problems to generate for training
        """
        logger.info(f"Demonstrating quantum advantage prediction with {num_problems} problems")
        
        problem_sizes = [10, 15, 20, 30, 40, 50, 75, 100]
        problem_types = ["random", "structured", "sparse"]
        
        # Generate training data
        logger.info("Generating training data...")
        for i in range(num_problems):
            size = np.random.choice(problem_sizes)
            problem_type = np.random.choice(problem_types)
            
            problem = self.generate_research_problem(size, problem_type)
            
            # Extract features
            features = self.advantage_predictor.extract_features(
                problem['agents'], 
                problem['tasks'], 
                problem['qubo_matrix']
            )
            
            # Simulate both classical and quantum execution
            classical_solution, classical_energy, classical_time = self.simulate_classical_execution(
                problem['qubo_matrix']
            )
            
            quantum_solution, quantum_energy, quantum_time = self.simulate_quantum_execution(
                problem['qubo_matrix'], backend="quantum_sim"
            )
            
            # Calculate solution qualities
            classical_quality = max(0, 1.0 / (1.0 + abs(classical_energy) / 10))
            quantum_quality = max(0, 1.0 / (1.0 + abs(quantum_energy) / 10))
            
            # Record observation
            self.advantage_predictor.record_advantage_observation(
                features=features,
                classical_time=classical_time,
                quantum_time=quantum_time,
                classical_quality=classical_quality,
                quantum_quality=quantum_quality,
                backend_used="quantum_sim"
            )
            
            if (i + 1) % 5 == 0:
                logger.info(f"Generated {i + 1}/{num_problems} training samples")
        
        # Test prediction on new problems
        logger.info("Testing predictions on new problems...")
        test_problems = [
            (15, "random"),
            (35, "structured"), 
            (60, "sparse"),
            (80, "random")
        ]
        
        prediction_results = []
        for size, problem_type in test_problems:
            test_problem = self.generate_research_problem(size, problem_type)
            test_features = self.advantage_predictor.extract_features(
                test_problem['agents'],
                test_problem['tasks'],
                test_problem['qubo_matrix']
            )
            
            prediction = self.advantage_predictor.predict_quantum_advantage(test_features)
            prediction_results.append({
                'problem_size': size,
                'problem_type': problem_type,
                'prediction': prediction
            })
            
            logger.info(f"Problem {size} ({problem_type}): "
                       f"Advantage prob={prediction['advantage_probability']:.3f}, "
                       f"Expected speedup={prediction['expected_speedup']:.2f}x, "
                       f"Recommendation={prediction['recommendation']}")
        
        # Generate insights report
        insights_report = self.advantage_predictor.generate_insights_report()
        
        # Save results
        report_path = self.output_dir / "quantum_advantage_prediction_demo.md"
        with open(report_path, 'w') as f:
            f.write("# Quantum Advantage Prediction Demonstration\n\n")
            f.write(f"**Training Samples**: {len(self.advantage_predictor.training_records)}\n")
            f.write(f"**Test Problems**: {len(test_problems)}\n\n")
            
            f.write("## Test Results\n\n")
            for result in prediction_results:
                pred = result['prediction']
                f.write(f"### Problem Size {result['problem_size']} ({result['problem_type']})\n")
                f.write(f"- **Advantage Probability**: {pred['advantage_probability']:.3f}\n")
                f.write(f"- **Expected Speedup**: {pred['expected_speedup']:.2f}x\n")
                f.write(f"- **Confidence**: {pred['confidence']:.3f}\n")
                f.write(f"- **Recommendation**: {pred['recommendation']}\n")
                f.write(f"- **Reason**: {pred['reason']}\n\n")
            
            f.write("## Model Insights\n\n")
            f.write(insights_report)
        
        logger.info(f"Quantum advantage prediction demo completed. Report saved to: {report_path}")
    
    def demonstrate_adaptive_optimization(self, num_problems: int = 15) -> None:
        """Demonstrate adaptive QUBO optimization capabilities.
        
        Args:
            num_problems: Number of problems to test
        """
        logger.info(f"Demonstrating adaptive QUBO optimization with {num_problems} problems")
        
        problem_configs = [
            (10, "structured"),
            (25, "random"),
            (15, "sparse"),
            (40, "structured"),
            (30, "random"),
            (50, "sparse"),
            (20, "structured"),
            (35, "random")
        ]
        
        results = []
        
        for i, (size, problem_type) in enumerate(problem_configs[:num_problems]):
            logger.info(f"Testing problem {i+1}/{min(num_problems, len(problem_configs))}: "
                       f"size={size}, type={problem_type}")
            
            problem = self.generate_research_problem(size, problem_type)
            
            context = ProblemContext(
                problem_size=size,
                density=problem['metadata']['density'],
                problem_class=problem_type,
                constraints={'test_problem': True}
            )
            
            # Test different selection strategies
            strategies = ["adaptive", "portfolio", "best"]
            problem_results = {'problem_size': size, 'problem_type': problem_type, 'strategies': {}}
            
            for strategy in strategies:
                start_time = time.time()
                
                solution, energy, metadata = self.adaptive_optimizer.solve(
                    qubo_matrix=problem['qubo_matrix'],
                    context=context,
                    max_time=2.0,
                    algorithm_selection=strategy
                )
                
                execution_time = time.time() - start_time
                
                problem_results['strategies'][strategy] = {
                    'algorithm_used': metadata['algorithm_used'],
                    'energy': energy,
                    'execution_time': execution_time,
                    'solution_size': np.sum(solution),
                    'metadata': metadata
                }
                
                logger.info(f"  {strategy}: {metadata['algorithm_used']}, "
                           f"energy={energy:.2f}, time={execution_time:.3f}s")
            
            results.append(problem_results)
        
        # Generate statistics
        algorithm_stats = self.adaptive_optimizer.get_algorithm_statistics()
        performance_report = self.adaptive_optimizer.generate_performance_report()
        
        # Save results
        report_path = self.output_dir / "adaptive_optimization_demo.md"
        with open(report_path, 'w') as f:
            f.write("# Adaptive QUBO Optimization Demonstration\n\n")
            f.write(f"**Problems Tested**: {len(results)}\n")
            f.write(f"**Algorithms Available**: {len(self.adaptive_optimizer.algorithms)}\n\n")
            
            f.write("## Problem Results\n\n")
            for i, result in enumerate(results, 1):
                f.write(f"### Problem {i}: Size {result['problem_size']} ({result['problem_type']})\n\n")
                
                for strategy, strategy_result in result['strategies'].items():
                    f.write(f"**{strategy.title()} Strategy**:\n")
                    f.write(f"- Algorithm: {strategy_result['algorithm_used']}\n")
                    f.write(f"- Energy: {strategy_result['energy']:.3f}\n")
                    f.write(f"- Execution Time: {strategy_result['execution_time']:.3f}s\n")
                    f.write(f"- Solution Size: {strategy_result['solution_size']} variables selected\n\n")
            
            f.write("## Algorithm Statistics\n\n")
            for algo_name, stats in algorithm_stats.items():
                f.write(f"### {algo_name.replace('_', ' ').title()}\n")
                f.write(f"- Total Executions: {stats['total_executions']}\n")
                f.write(f"- Average Execution Time: {stats['average_execution_time']:.4f}s\n")
                f.write(f"- Average Solution Quality: {stats['average_solution_quality']:.3f}\n")
                f.write(f"- Success Rate: {stats['success_probability']:.1%}\n")
                f.write(f"- Selection Count: {stats['selection_count']}\n\n")
            
            f.write("## Performance Report\n\n")
            f.write(performance_report)
        
        logger.info(f"Adaptive optimization demo completed. Report saved to: {report_path}")
    
    async def demonstrate_comparative_analysis(self) -> None:
        """Demonstrate comparative analysis framework."""
        logger.info("Demonstrating comparative analysis framework")
        
        # Define algorithms for comparison
        def classical_solver(qubo_matrix, agents, tasks, max_time=60.0):
            """Classical solver for comparison."""
            solution, energy, exec_time = self.simulate_classical_execution(qubo_matrix)
            return solution, energy, {'solver_type': 'classical', 'execution_time': exec_time}
        
        def quantum_sim_solver(qubo_matrix, agents, tasks, max_time=60.0):
            """Quantum simulator solver for comparison."""
            solution, energy, exec_time = self.simulate_quantum_execution(qubo_matrix, "quantum_sim")
            return solution, energy, {'solver_type': 'quantum_sim', 'execution_time': exec_time}
        
        def quantum_hw_solver(qubo_matrix, agents, tasks, max_time=60.0):
            """Quantum hardware solver for comparison."""
            solution, energy, exec_time = self.simulate_quantum_execution(qubo_matrix, "quantum_hw")
            return solution, energy, {'solver_type': 'quantum_hw', 'execution_time': exec_time}
        
        algorithms = {
            'classical': classical_solver,
            'quantum_sim': quantum_sim_solver,
            'quantum_hw': quantum_hw_solver
        }
        
        # Configure experiment
        config = ExperimentConfig(
            problem_sizes=[10, 20, 30, 50],
            problem_categories=[ProblemCategory.SMALL_DENSE, ProblemCategory.MEDIUM_SPARSE],
            num_trials_per_config=3,
            max_execution_time=5.0,
            significance_level=0.05
        )
        
        # Run comparative study
        logger.info("Running comparative study...")
        results = await self.analysis_framework.run_comparative_study(algorithms, config)
        
        # Generate visualization report
        try:
            plot_path = self.analysis_framework.generate_visualization_report()
            logger.info(f"Visualization plots saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization plots: {e}")
        
        # Save summary
        summary_path = self.output_dir / "comparative_analysis_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Comparative Analysis Demonstration\n\n")
            f.write(f"**Study Duration**: {results['study_metadata']['duration_seconds']:.2f} seconds\n")
            f.write(f"**Total Trials**: {results['study_metadata']['total_trials']}\n")
            f.write(f"**Algorithms**: {', '.join(results['study_metadata']['algorithms_tested'])}\n\n")
            
            f.write("## Summary Statistics\n\n")
            for algo_name, stats in results['summary_statistics'].items():
                f.write(f"### {algo_name.title()}\n")
                f.write(f"- Success Rate: {stats['success_rate']:.1%}\n")
                f.write(f"- Average Execution Time: {stats['avg_execution_time']:.4f}s\n")
                f.write(f"- Average Solution Quality: {stats['avg_solution_quality']:.3f}\n\n")
            
            if results['insights']['significant_differences']:
                f.write("## Significant Performance Differences\n\n")
                for diff in results['insights']['significant_differences']:
                    f.write(f"- **{diff['faster_algorithm']}** is {diff['speedup_factor']:.2f}x faster than **{diff['slower_algorithm']}** (p < {diff['p_value']:.3f})\n")
            
            if results['insights']['recommendations']:
                f.write("\n## Recommendations\n\n")
                for rec in results['insights']['recommendations']:
                    f.write(f"- {rec}\n")
        
        logger.info(f"Comparative analysis demo completed. Summary saved to: {summary_path}")
    
    async def run_complete_research_demo(self) -> None:
        """Run the complete research demonstration."""
        logger.info("Starting complete research demonstration")
        
        print("\n" + "="*60)
        print("   QUANTUM SCHEDULER RESEARCH DEMONSTRATION")
        print("="*60)
        
        print("\n🔬 This demonstration showcases advanced research capabilities:")
        print("   • Quantum Advantage Prediction with ML models")
        print("   • Adaptive QUBO Optimization with algorithm portfolios")
        print("   • Comparative Analysis with statistical significance testing")
        
        # Run demonstrations
        try:
            print("\n📊 Phase 1: Quantum Advantage Prediction")
            print("-" * 40)
            self.demonstrate_quantum_advantage_prediction(num_problems=30)
            
            print("\n⚙️  Phase 2: Adaptive QUBO Optimization")
            print("-" * 40)
            self.demonstrate_adaptive_optimization(num_problems=12)
            
            print("\n📈 Phase 3: Comparative Analysis Framework")
            print("-" * 40)
            await self.demonstrate_comparative_analysis()
            
            print("\n✅ Research demonstration completed successfully!")
            print(f"📁 All results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Research demonstration failed: {e}")
            raise


def main():
    """Main function to run the research demonstration."""
    demo = ResearchDemo()
    
    try:
        asyncio.run(demo.run_complete_research_demo())
    except KeyboardInterrupt:
        logger.info("Research demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Research demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())