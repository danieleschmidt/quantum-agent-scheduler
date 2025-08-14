"""Adaptive QUBO Optimization with Multi-Algorithm Portfolio.

This module implements an adaptive optimization framework that dynamically
selects and combines multiple QUBO solving algorithms based on problem
characteristics and real-time performance feedback.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Types of QUBO optimization algorithms."""
    CLASSICAL_GREEDY = "classical_greedy"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    TABU_SEARCH = "tabu_search"
    QAOA = "qaoa"
    VQE = "vqe"
    QUANTUM_ANNEALING = "quantum_annealing"
    HYBRID_CLASSICAL_QUANTUM = "hybrid_cq"


@dataclass
class AlgorithmPerformance:
    """Performance metrics for a QUBO algorithm."""
    algorithm_type: AlgorithmType
    execution_time: float
    solution_quality: float
    energy_value: float
    convergence_iterations: int
    success_probability: float
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (quality/time)."""
        if self.execution_time <= 0:
            return 0.0
        return self.solution_quality / self.execution_time


@dataclass
class ProblemContext:
    """Context information about a QUBO problem."""
    problem_size: int
    density: float
    problem_class: str
    constraints: Dict[str, Any]
    deadline: Optional[float] = None
    quality_threshold: float = 0.8
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.problem_size,
            self.density,
            len(self.constraints),
            self.quality_threshold
        ])


class QUBOAlgorithm(ABC):
    """Abstract base class for QUBO optimization algorithms."""
    
    def __init__(self, algorithm_type: AlgorithmType):
        self.algorithm_type = algorithm_type
        self.performance_history: List[AlgorithmPerformance] = []
    
    @abstractmethod
    def solve(self, 
              qubo_matrix: np.ndarray, 
              context: ProblemContext,
              max_time: float = 60.0) -> Tuple[np.ndarray, float]:
        """Solve QUBO problem.
        
        Args:
            qubo_matrix: QUBO problem matrix
            context: Problem context information
            max_time: Maximum execution time
            
        Returns:
            Tuple of (solution_vector, energy_value)
        """
        pass
    
    def get_average_performance(self, 
                               problem_size_range: Tuple[int, int] = None) -> AlgorithmPerformance:
        """Get average performance metrics.
        
        Args:
            problem_size_range: Optional size range filter
            
        Returns:
            Average performance metrics
        """
        if not self.performance_history:
            return AlgorithmPerformance(
                algorithm_type=self.algorithm_type,
                execution_time=float('inf'),
                solution_quality=0.0,
                energy_value=float('inf'),
                convergence_iterations=0,
                success_probability=0.0
            )
        
        # Filter by problem size if specified
        filtered_history = self.performance_history
        if problem_size_range:
            filtered_history = [
                p for p in self.performance_history
                if problem_size_range[0] <= p.resource_usage.get('problem_size', 0) <= problem_size_range[1]
            ]
        
        if not filtered_history:
            filtered_history = self.performance_history
        
        return AlgorithmPerformance(
            algorithm_type=self.algorithm_type,
            execution_time=np.mean([p.execution_time for p in filtered_history]),
            solution_quality=np.mean([p.solution_quality for p in filtered_history]),
            energy_value=np.mean([p.energy_value for p in filtered_history]),
            convergence_iterations=int(np.mean([p.convergence_iterations for p in filtered_history])),
            success_probability=np.mean([p.success_probability for p in filtered_history])
        )


class ClassicalGreedyQUBO(QUBOAlgorithm):
    """Classical greedy algorithm for QUBO problems."""
    
    def __init__(self):
        super().__init__(AlgorithmType.CLASSICAL_GREEDY)
    
    def solve(self, 
              qubo_matrix: np.ndarray, 
              context: ProblemContext,
              max_time: float = 60.0) -> Tuple[np.ndarray, float]:
        """Solve using greedy heuristic."""
        start_time = time.time()
        n = qubo_matrix.shape[0]
        
        # Initialize solution vector
        solution = np.zeros(n, dtype=int)
        
        # Greedy selection based on diagonal dominance
        remaining_indices = list(range(n))
        
        while remaining_indices and (time.time() - start_time) < max_time:
            # Calculate potential energy reduction for each remaining variable
            best_improvement = float('-inf')
            best_index = None
            
            for i in remaining_indices:
                # Calculate energy change if we set x_i = 1
                current_energy = self._calculate_energy_change(qubo_matrix, solution, i)
                
                if current_energy > best_improvement:
                    best_improvement = current_energy
                    best_index = i
            
            if best_index is not None and best_improvement > 0:
                solution[best_index] = 1
                remaining_indices.remove(best_index)
            else:
                break
        
        energy = self._calculate_total_energy(qubo_matrix, solution)
        execution_time = time.time() - start_time
        
        # Record performance
        performance = AlgorithmPerformance(
            algorithm_type=self.algorithm_type,
            execution_time=execution_time,
            solution_quality=self._calculate_solution_quality(energy, context),
            energy_value=energy,
            convergence_iterations=n - len(remaining_indices),
            success_probability=1.0 if len(remaining_indices) == 0 else 0.5,
            resource_usage={'problem_size': n}
        )
        self.performance_history.append(performance)
        
        return solution, energy
    
    def _calculate_energy_change(self, 
                                qubo_matrix: np.ndarray, 
                                solution: np.ndarray, 
                                index: int) -> float:
        """Calculate energy change for setting variable to 1."""
        energy_change = qubo_matrix[index, index]  # Diagonal term
        
        # Add interaction terms with already selected variables
        for j in range(len(solution)):
            if solution[j] == 1 and j != index:
                energy_change += qubo_matrix[index, j] + qubo_matrix[j, index]
        
        return -energy_change  # Negative because we want to minimize energy
    
    def _calculate_total_energy(self, 
                               qubo_matrix: np.ndarray, 
                               solution: np.ndarray) -> float:
        """Calculate total energy of solution."""
        return np.dot(solution, np.dot(qubo_matrix, solution))
    
    def _calculate_solution_quality(self, 
                                   energy: float, 
                                   context: ProblemContext) -> float:
        """Calculate solution quality score."""
        # Normalize energy to [0, 1] quality score
        # This is problem-dependent and simplified here
        max_possible_energy = np.abs(np.sum(np.diag(context.problem_size * [1.0])))
        if max_possible_energy <= 0:
            return 1.0
        
        normalized_energy = abs(energy) / max_possible_energy
        return max(0.0, 1.0 - normalized_energy)


class SimulatedAnnealingQUBO(QUBOAlgorithm):
    """Simulated annealing algorithm for QUBO problems."""
    
    def __init__(self, initial_temp: float = 1000.0, cooling_rate: float = 0.995):
        super().__init__(AlgorithmType.SIMULATED_ANNEALING)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def solve(self, 
              qubo_matrix: np.ndarray, 
              context: ProblemContext,
              max_time: float = 60.0) -> Tuple[np.ndarray, float]:
        """Solve using simulated annealing."""
        start_time = time.time()
        n = qubo_matrix.shape[0]
        
        # Initialize random solution
        current_solution = np.random.randint(0, 2, n)
        current_energy = self._calculate_total_energy(qubo_matrix, current_solution)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temperature = self.initial_temp
        iterations = 0
        
        while temperature > 0.01 and (time.time() - start_time) < max_time:
            # Generate neighbor by flipping a random bit
            neighbor = current_solution.copy()
            flip_index = random.randint(0, n - 1)
            neighbor[flip_index] = 1 - neighbor[flip_index]
            
            neighbor_energy = self._calculate_total_energy(qubo_matrix, neighbor)
            
            # Accept or reject based on Metropolis criterion
            energy_diff = neighbor_energy - current_energy
            
            if energy_diff < 0 or random.random() < np.exp(-energy_diff / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            
            temperature *= self.cooling_rate
            iterations += 1
        
        execution_time = time.time() - start_time
        
        # Record performance
        performance = AlgorithmPerformance(
            algorithm_type=self.algorithm_type,
            execution_time=execution_time,
            solution_quality=self._calculate_solution_quality(best_energy, context),
            energy_value=best_energy,
            convergence_iterations=iterations,
            success_probability=0.8,  # Stochastic algorithm
            resource_usage={'problem_size': n, 'final_temperature': temperature}
        )
        self.performance_history.append(performance)
        
        return best_solution, best_energy
    
    def _calculate_total_energy(self, 
                               qubo_matrix: np.ndarray, 
                               solution: np.ndarray) -> float:
        """Calculate total energy of solution."""
        return np.dot(solution, np.dot(qubo_matrix, solution))
    
    def _calculate_solution_quality(self, 
                                   energy: float, 
                                   context: ProblemContext) -> float:
        """Calculate solution quality score."""
        # Simple quality metric - can be improved
        return max(0.0, 1.0 / (1.0 + abs(energy)))


class QAOAQuantumAlgorithm(QUBOAlgorithm):
    """Quantum Approximate Optimization Algorithm for QUBO problems."""
    
    def __init__(self, num_layers: int = 4, num_shots: int = 1000):
        super().__init__(AlgorithmType.QAOA)
        self.num_layers = num_layers
        self.num_shots = num_shots
    
    def solve(self, 
              qubo_matrix: np.ndarray, 
              context: ProblemContext,
              max_time: float = 60.0) -> Tuple[np.ndarray, float]:
        """Solve using QAOA (simulated for now)."""
        start_time = time.time()
        n = qubo_matrix.shape[0]
        
        # Simulate QAOA execution
        # In real implementation, this would construct and execute quantum circuits
        
        # For simulation, we'll use a probabilistic approach
        # that performs better on certain problem structures
        
        # Eigenvalue-based heuristic to simulate QAOA behavior
        try:
            eigenvals, eigenvecs = np.linalg.eigh(qubo_matrix)
            
            # Use dominant eigenvector as initial guess
            dominant_eigenvec = eigenvecs[:, np.argmax(np.abs(eigenvals))]
            
            # Convert to binary solution probabilistically
            probabilities = np.abs(dominant_eigenvec) / np.sum(np.abs(dominant_eigenvec))
            solution = np.random.binomial(1, probabilities)
            
        except:
            # Fallback to random solution
            solution = np.random.randint(0, 2, n)
        
        # Local optimization to improve solution
        solution = self._local_optimization(qubo_matrix, solution, max_time - (time.time() - start_time))
        
        energy = self._calculate_total_energy(qubo_matrix, solution)
        execution_time = time.time() - start_time
        
        # Record performance
        performance = AlgorithmPerformance(
            algorithm_type=self.algorithm_type,
            execution_time=execution_time,
            solution_quality=self._calculate_solution_quality(energy, context),
            energy_value=energy,
            convergence_iterations=self.num_layers,
            success_probability=0.7,  # Quantum algorithms are probabilistic
            resource_usage={'problem_size': n, 'num_layers': self.num_layers, 'num_shots': self.num_shots}
        )
        self.performance_history.append(performance)
        
        return solution, energy
    
    def _local_optimization(self, 
                           qubo_matrix: np.ndarray, 
                           initial_solution: np.ndarray,
                           max_time: float) -> np.ndarray:
        """Local optimization to improve quantum solution."""
        current_solution = initial_solution.copy()
        current_energy = self._calculate_total_energy(qubo_matrix, current_solution)
        
        start_time = time.time()
        n = len(current_solution)
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            for i in range(n):
                # Try flipping bit i
                test_solution = current_solution.copy()
                test_solution[i] = 1 - test_solution[i]
                test_energy = self._calculate_total_energy(qubo_matrix, test_solution)
                
                if test_energy < current_energy:
                    current_solution = test_solution
                    current_energy = test_energy
                    improved = True
                    break
        
        return current_solution
    
    def _calculate_total_energy(self, 
                               qubo_matrix: np.ndarray, 
                               solution: np.ndarray) -> float:
        """Calculate total energy of solution."""
        return np.dot(solution, np.dot(qubo_matrix, solution))
    
    def _calculate_solution_quality(self, 
                                   energy: float, 
                                   context: ProblemContext) -> float:
        """Calculate solution quality score."""
        return max(0.0, 1.0 / (1.0 + abs(energy)))


class AdaptiveQUBOOptimizer:
    """Adaptive optimizer that selects best algorithm based on problem characteristics."""
    
    def __init__(self, 
                 algorithms: Optional[List[QUBOAlgorithm]] = None,
                 performance_window: int = 100):
        """Initialize adaptive optimizer.
        
        Args:
            algorithms: List of available algorithms
            performance_window: Window size for performance tracking
        """
        self.algorithms = algorithms or self._default_algorithms()
        self.performance_window = performance_window
        self.performance_history = deque(maxlen=performance_window)
        self.algorithm_selection_history = defaultdict(list)
        
        logger.info(f"Initialized AdaptiveQUBOOptimizer with {len(self.algorithms)} algorithms")
    
    def _default_algorithms(self) -> List[QUBOAlgorithm]:
        """Create default algorithm portfolio."""
        return [
            ClassicalGreedyQUBO(),
            SimulatedAnnealingQUBO(),
            QAOAQuantumAlgorithm(num_layers=3),
            QAOAQuantumAlgorithm(num_layers=6)
        ]
    
    def solve(self, 
              qubo_matrix: np.ndarray,
              context: ProblemContext,
              max_time: float = 60.0,
              algorithm_selection: str = "adaptive") -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve QUBO problem using adaptive algorithm selection.
        
        Args:
            qubo_matrix: QUBO problem matrix
            context: Problem context
            max_time: Maximum execution time
            algorithm_selection: Selection strategy ("adaptive", "portfolio", "best")
            
        Returns:
            Tuple of (solution, energy, metadata)
        """
        if algorithm_selection == "adaptive":
            return self._solve_adaptive(qubo_matrix, context, max_time)
        elif algorithm_selection == "portfolio":
            return self._solve_portfolio(qubo_matrix, context, max_time)
        elif algorithm_selection == "best":
            return self._solve_best(qubo_matrix, context, max_time)
        else:
            # Fallback to first algorithm
            return self._solve_single(self.algorithms[0], qubo_matrix, context, max_time)
    
    def _solve_adaptive(self, 
                       qubo_matrix: np.ndarray,
                       context: ProblemContext,
                       max_time: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve using adaptive algorithm selection."""
        # Select algorithm based on problem characteristics and past performance
        selected_algorithm = self._select_algorithm(context)
        
        start_time = time.time()
        solution, energy = selected_algorithm.solve(qubo_matrix, context, max_time)
        execution_time = time.time() - start_time
        
        # Record selection decision
        self.algorithm_selection_history[selected_algorithm.algorithm_type].append({
            'context': context,
            'performance': selected_algorithm.performance_history[-1] if selected_algorithm.performance_history else None,
            'timestamp': time.time()
        })
        
        metadata = {
            'algorithm_used': selected_algorithm.algorithm_type.value,
            'execution_time': execution_time,
            'selection_reason': 'adaptive',
            'total_algorithms_available': len(self.algorithms)
        }
        
        return solution, energy, metadata
    
    def _solve_portfolio(self, 
                        qubo_matrix: np.ndarray,
                        context: ProblemContext,
                        max_time: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve using portfolio approach (run multiple algorithms)."""
        time_per_algorithm = max_time / len(self.algorithms)
        results = []
        
        # Run all algorithms in parallel
        with ThreadPoolExecutor(max_workers=len(self.algorithms)) as executor:
            futures = {
                executor.submit(algorithm.solve, qubo_matrix, context, time_per_algorithm): algorithm
                for algorithm in self.algorithms
            }
            
            for future in as_completed(futures):
                algorithm = futures[future]
                try:
                    solution, energy = future.result()
                    results.append((solution, energy, algorithm))
                except Exception as e:
                    logger.warning(f"Algorithm {algorithm.algorithm_type} failed: {e}")
        
        # Select best result
        if results:
            best_solution, best_energy, best_algorithm = min(results, key=lambda x: x[1])
            
            metadata = {
                'algorithm_used': best_algorithm.algorithm_type.value,
                'portfolio_size': len(results),
                'selection_reason': 'portfolio_best',
                'all_energies': [r[1] for r in results]
            }
            
            return best_solution, best_energy, metadata
        else:
            # Fallback
            return self._solve_single(self.algorithms[0], qubo_matrix, context, max_time)
    
    def _solve_best(self, 
                   qubo_matrix: np.ndarray,
                   context: ProblemContext,
                   max_time: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve using historically best algorithm."""
        # Find best performing algorithm for similar problems
        best_algorithm = self._get_best_algorithm(context)
        
        solution, energy = best_algorithm.solve(qubo_matrix, context, max_time)
        
        metadata = {
            'algorithm_used': best_algorithm.algorithm_type.value,
            'selection_reason': 'historical_best',
            'avg_performance': best_algorithm.get_average_performance().efficiency_score
        }
        
        return solution, energy, metadata
    
    def _solve_single(self, 
                     algorithm: QUBOAlgorithm,
                     qubo_matrix: np.ndarray,
                     context: ProblemContext,
                     max_time: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve using a single algorithm."""
        solution, energy = algorithm.solve(qubo_matrix, context, max_time)
        
        metadata = {
            'algorithm_used': algorithm.algorithm_type.value,
            'selection_reason': 'specified',
        }
        
        return solution, energy, metadata
    
    def _select_algorithm(self, context: ProblemContext) -> QUBOAlgorithm:
        """Select best algorithm based on problem context."""
        # Simple heuristic-based selection
        # In practice, this could use ML models
        
        problem_size = context.problem_size
        density = context.density
        
        # Size-based selection
        if problem_size < 20:
            # Small problems: try quantum algorithms
            quantum_algorithms = [alg for alg in self.algorithms 
                                if alg.algorithm_type in [AlgorithmType.QAOA, AlgorithmType.VQE]]
            if quantum_algorithms:
                return quantum_algorithms[0]
        
        elif problem_size < 100:
            # Medium problems: simulated annealing often works well
            sa_algorithms = [alg for alg in self.algorithms 
                           if alg.algorithm_type == AlgorithmType.SIMULATED_ANNEALING]
            if sa_algorithms:
                return sa_algorithms[0]
        
        else:
            # Large problems: classical greedy for speed
            greedy_algorithms = [alg for alg in self.algorithms 
                               if alg.algorithm_type == AlgorithmType.CLASSICAL_GREEDY]
            if greedy_algorithms:
                return greedy_algorithms[0]
        
        # Fallback to first available algorithm
        return self.algorithms[0]
    
    def _get_best_algorithm(self, context: ProblemContext) -> QUBOAlgorithm:
        """Get historically best performing algorithm."""
        best_algorithm = self.algorithms[0]
        best_score = 0.0
        
        problem_size_range = (context.problem_size * 0.8, context.problem_size * 1.2)
        
        for algorithm in self.algorithms:
            avg_performance = algorithm.get_average_performance(problem_size_range)
            if avg_performance.efficiency_score > best_score:
                best_score = avg_performance.efficiency_score
                best_algorithm = algorithm
        
        return best_algorithm
    
    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about algorithm performance.
        
        Returns:
            Dictionary with performance statistics for each algorithm
        """
        statistics = {}
        
        for algorithm in self.algorithms:
            algo_name = algorithm.algorithm_type.value
            avg_perf = algorithm.get_average_performance()
            
            statistics[algo_name] = {
                'total_executions': len(algorithm.performance_history),
                'average_execution_time': avg_perf.execution_time,
                'average_solution_quality': avg_perf.solution_quality,
                'average_efficiency_score': avg_perf.efficiency_score,
                'success_probability': avg_perf.success_probability,
                'selection_count': len(self.algorithm_selection_history[algorithm.algorithm_type])
            }
        
        return statistics
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report.
        
        Returns:
            Formatted performance report string
        """
        stats = self.get_algorithm_statistics()
        
        report = [
            "# Adaptive QUBO Optimizer Performance Report",
            "",
            f"**Total Algorithms**: {len(self.algorithms)}",
            f"**Performance Window**: {self.performance_window}",
            "",
            "## Algorithm Performance Summary",
            ""
        ]
        
        # Sort algorithms by efficiency score
        sorted_algorithms = sorted(stats.items(), 
                                 key=lambda x: x[1]['average_efficiency_score'], 
                                 reverse=True)
        
        for algo_name, algo_stats in sorted_algorithms:
            report.extend([
                f"### {algo_name.replace('_', ' ').title()}",
                f"- **Executions**: {algo_stats['total_executions']}",
                f"- **Avg Execution Time**: {algo_stats['average_execution_time']:.4f}s",
                f"- **Avg Solution Quality**: {algo_stats['average_solution_quality']:.3f}",
                f"- **Efficiency Score**: {algo_stats['average_efficiency_score']:.3f}",
                f"- **Success Rate**: {algo_stats['success_probability']:.1%}",
                f"- **Selection Count**: {algo_stats['selection_count']}",
                ""
            ])
        
        # Performance insights
        report.extend([
            "## Key Insights",
            ""
        ])
        
        if sorted_algorithms:
            best_algo = sorted_algorithms[0]
            worst_algo = sorted_algorithms[-1]
            
            report.append(f"- **Best Performer**: {best_algo[0]} (efficiency: {best_algo[1]['average_efficiency_score']:.3f})")
            report.append(f"- **Needs Improvement**: {worst_algo[0]} (efficiency: {worst_algo[1]['average_efficiency_score']:.3f})")
            
            # Calculate performance spread
            efficiency_scores = [stats[1]['average_efficiency_score'] for stats in sorted_algorithms]
            if len(efficiency_scores) > 1:
                performance_spread = max(efficiency_scores) - min(efficiency_scores)
                report.append(f"- **Performance Spread**: {performance_spread:.3f}")
        
        return "\n".join(report)