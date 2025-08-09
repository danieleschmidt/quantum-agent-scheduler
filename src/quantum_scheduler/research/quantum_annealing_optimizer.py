"""Novel Quantum Annealing Optimization for Agent Scheduling.

This module implements cutting-edge quantum annealing techniques with adaptive
parameter tuning and multi-objective optimization for large-scale agent scheduling problems.

Research Focus:
- Dynamic annealing schedules based on problem topology
- Hybrid embedding strategies for large QUBO problems  
- Real-time parameter adaptation using machine learning
- Comparative analysis against classical and gate-based quantum methods
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnnealingStrategy(Enum):
    """Different annealing strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"
    REVERSE_ANNEALING = "reverse_annealing"
    MULTI_STAGE = "multi_stage"


@dataclass
class AnnealingSchedule:
    """Annealing schedule parameters."""
    strategy: AnnealingStrategy
    total_time: float
    pause_duration: float = 0.0
    slope: float = 1.0
    quench_rate: float = 0.1
    stages: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class EmbeddingMetrics:
    """Metrics for QUBO embedding quality."""
    embedding_ratio: float
    chain_lengths: List[int]
    max_chain_length: int
    connectivity_utilization: float
    embedding_time: float
    success_probability: float


@dataclass
class AnnealingResult:
    """Result from quantum annealing optimization."""
    energy: float
    solution_vector: np.ndarray
    success_probability: float
    annealing_time: float
    embedding_metrics: EmbeddingMetrics
    schedule_used: AnnealingSchedule
    num_reads: int
    chain_breaks: int
    
    @property
    def solution_quality(self) -> float:
        """Calculate solution quality metric."""
        return self.success_probability * (1.0 - self.chain_breaks / max(self.num_reads, 1))


class AdaptiveQuantumAnnealer:
    """Advanced quantum annealer with adaptive parameter tuning."""
    
    def __init__(self, 
                 hardware_constraints: Optional[Dict[str, Any]] = None,
                 learning_rate: float = 0.01,
                 history_window: int = 100):
        """Initialize adaptive quantum annealer.
        
        Args:
            hardware_constraints: Hardware-specific constraints
            learning_rate: Learning rate for parameter adaptation
            history_window: Number of previous results to consider for adaptation
        """
        self.hardware_constraints = hardware_constraints or self._default_hardware_constraints()
        self.learning_rate = learning_rate
        self.history_window = history_window
        
        # Performance tracking
        self.optimization_history = []
        self.parameter_history = defaultdict(list)
        self.problem_topology_cache = {}
        
        # ML model for parameter prediction (simplified)
        self.parameter_predictor = self._initialize_parameter_predictor()
        
        logger.info("Initialized AdaptiveQuantumAnnealer with ML parameter tuning")
    
    def _default_hardware_constraints(self) -> Dict[str, Any]:
        """Default quantum annealing hardware constraints."""
        return {
            'max_qubits': 5000,
            'connectivity': 'chimera',  # or 'pegasus', 'zephyr'
            'programming_thermalization': 1000,  # microseconds
            'readout_thermalization': 1000,      # microseconds
            'max_annealing_time': 2000,         # microseconds
            'min_annealing_time': 1,            # microseconds
            'max_pause_duration': 40,           # microseconds
            'annealing_offset_ranges': [-0.5, 0.5],
            'flux_bias_ranges': [-2.0, 2.0],
            'chain_strength_range': [0.1, 10.0]
        }
    
    def _initialize_parameter_predictor(self) -> Dict[str, Any]:
        """Initialize ML model for parameter prediction."""
        return {
            'model_type': 'adaptive_regression',
            'features': ['problem_size', 'density', 'conditioning', 'topology_complexity'],
            'parameters': ['annealing_time', 'chain_strength', 'num_reads', 'pause_duration'],
            'weights': np.random.normal(0, 0.1, (4, 4)),  # 4 features x 4 parameters
            'bias': np.zeros(4)
        }
    
    def optimize_scheduling_problem(self, 
                                  qubo_matrix: np.ndarray,
                                  max_iterations: int = 10,
                                  target_quality: float = 0.95) -> AnnealingResult:
        """Optimize agent scheduling problem using adaptive quantum annealing.
        
        Args:
            qubo_matrix: QUBO formulation of scheduling problem
            max_iterations: Maximum optimization iterations
            target_quality: Target solution quality threshold
            
        Returns:
            Best annealing result found
        """
        start_time = time.time()
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem_topology(qubo_matrix)
        
        # Generate initial annealing schedule using ML predictions
        initial_schedule = self._predict_optimal_schedule(problem_analysis)
        
        # Perform iterative optimization with adaptation
        best_result = None
        iteration_results = []
        
        for iteration in range(max_iterations):
            logger.info(f"Annealing optimization iteration {iteration + 1}/{max_iterations}")
            
            # Adapt schedule based on previous results
            if iteration > 0:
                schedule = self._adapt_schedule(initial_schedule, iteration_results, problem_analysis)
            else:
                schedule = initial_schedule
            
            # Perform quantum annealing
            result = self._execute_quantum_annealing(qubo_matrix, schedule, problem_analysis)
            iteration_results.append(result)
            
            # Update best result
            if best_result is None or result.solution_quality > best_result.solution_quality:
                best_result = result
                logger.info(f"New best solution quality: {result.solution_quality:.4f}")
            
            # Check if target quality reached
            if result.solution_quality >= target_quality:
                logger.info(f"Target quality {target_quality:.4f} reached at iteration {iteration + 1}")
                break
            
            # Update ML model with new data
            self._update_parameter_predictor(problem_analysis, schedule, result)
        
        # Record optimization in history
        optimization_record = {
            'problem_size': qubo_matrix.shape[0],
            'total_time': time.time() - start_time,
            'iterations': len(iteration_results),
            'best_quality': best_result.solution_quality,
            'improvement_rate': self._calculate_improvement_rate(iteration_results)
        }
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Optimization complete: {len(iteration_results)} iterations, "
                   f"best quality: {best_result.solution_quality:.4f}")
        
        return best_result
    
    def _analyze_problem_topology(self, qubo_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze QUBO problem topology and characteristics."""
        problem_hash = hash(qubo_matrix.tobytes())
        
        if problem_hash in self.problem_topology_cache:
            return self.problem_topology_cache[problem_hash]
        
        n = qubo_matrix.shape[0]
        
        # Basic metrics
        density = np.count_nonzero(qubo_matrix) / (n * n)
        matrix_norm = np.linalg.norm(qubo_matrix)
        
        # Conditioning analysis
        eigenvals = np.linalg.eigvals(qubo_matrix + qubo_matrix.T)
        eigenvals = eigenvals[np.abs(eigenvals) > 1e-10]
        condition_number = np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals)) if len(eigenvals) > 0 else 1.0
        
        # Graph connectivity analysis
        adjacency = (np.abs(qubo_matrix) > 1e-10).astype(int)
        degree_sequence = np.sum(adjacency, axis=1)
        avg_degree = np.mean(degree_sequence)
        degree_variance = np.var(degree_sequence)
        
        # Clustering coefficient approximation
        clustering_coeff = self._estimate_clustering_coefficient(adjacency)
        
        # Community structure detection (simplified)
        community_modularity = self._estimate_community_structure(adjacency)
        
        # Topology complexity score
        topology_complexity = (
            density * np.log(n) +
            np.log(condition_number) * 0.1 +
            degree_variance / (avg_degree + 1) +
            clustering_coeff * 0.5 +
            community_modularity * 0.3
        )
        
        analysis = {
            'problem_size': n,
            'density': density,
            'matrix_norm': matrix_norm,
            'condition_number': condition_number,
            'avg_degree': avg_degree,
            'degree_variance': degree_variance,
            'clustering_coefficient': clustering_coeff,
            'community_modularity': community_modularity,
            'topology_complexity': topology_complexity,
            'sparsity_pattern': self._analyze_sparsity_pattern(qubo_matrix)
        }
        
        self.problem_topology_cache[problem_hash] = analysis
        return analysis
    
    def _estimate_clustering_coefficient(self, adjacency: np.ndarray) -> float:
        """Estimate clustering coefficient of the problem graph."""
        n = adjacency.shape[0]
        clustering_sum = 0.0
        
        for i in range(n):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adjacency[neighbors[j], neighbors[k]] > 0:
                        triangles += 1
            
            # Local clustering coefficient
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            if possible_triangles > 0:
                clustering_sum += triangles / possible_triangles
        
        return clustering_sum / n if n > 0 else 0.0
    
    def _estimate_community_structure(self, adjacency: np.ndarray) -> float:
        """Estimate community structure using modularity."""
        # Simplified modularity estimation
        n = adjacency.shape[0]
        total_edges = np.sum(adjacency) / 2
        
        if total_edges == 0:
            return 0.0
        
        # Random partition for modularity estimation
        partition_size = min(10, n // 2)
        if partition_size == 0:
            return 0.0
        
        modularity = 0.0
        for i in range(0, n, partition_size):
            community = list(range(i, min(i + partition_size, n)))
            
            # Internal edges
            internal_edges = 0
            for u in community:
                for v in community:
                    if u < v and adjacency[u, v] > 0:
                        internal_edges += 1
            
            # Expected internal edges
            community_degree = sum(np.sum(adjacency[u]) for u in community)
            expected_internal = (community_degree ** 2) / (4 * total_edges)
            
            modularity += (internal_edges - expected_internal) / total_edges
        
        return max(0.0, modularity)
    
    def _analyze_sparsity_pattern(self, qubo_matrix: np.ndarray) -> str:
        """Analyze sparsity pattern of QUBO matrix."""
        # Identify common sparsity patterns
        n = qubo_matrix.shape[0]
        nonzeros = np.count_nonzero(qubo_matrix)
        
        if nonzeros == n:
            return "diagonal"
        elif nonzeros <= 3 * n:
            return "sparse_linear"
        elif nonzeros <= n * (n - 1) // 4:
            return "moderate_sparse"
        elif nonzeros >= n * (n - 1) // 2:
            return "dense"
        else:
            return "irregular"
    
    def _predict_optimal_schedule(self, problem_analysis: Dict[str, float]) -> AnnealingSchedule:
        """Predict optimal annealing schedule using ML model."""
        # Extract features for prediction
        features = np.array([
            problem_analysis['problem_size'],
            problem_analysis['density'],
            problem_analysis['condition_number'],
            problem_analysis['topology_complexity']
        ])
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Predict parameters using linear model
        predictor = self.parameter_predictor
        predicted_params = np.dot(features, predictor['weights']) + predictor['bias']
        
        # Convert predictions to actual parameters
        annealing_time = max(1, min(2000, int(predicted_params[0] * 1000)))
        pause_duration = max(0, min(40, predicted_params[3] * 20))
        
        # Select strategy based on problem characteristics
        if problem_analysis['topology_complexity'] > 5.0:
            strategy = AnnealingStrategy.MULTI_STAGE
            stages = self._generate_multi_stage_schedule(problem_analysis)
        elif problem_analysis['condition_number'] > 100:
            strategy = AnnealingStrategy.REVERSE_ANNEALING
            stages = []
        else:
            strategy = AnnealingStrategy.ADAPTIVE
            stages = []
        
        schedule = AnnealingSchedule(
            strategy=strategy,
            total_time=annealing_time,
            pause_duration=pause_duration,
            slope=1.0,
            quench_rate=0.1,
            stages=stages
        )
        
        logger.info(f"Predicted schedule: {strategy.value}, time={annealing_time}μs, pause={pause_duration}μs")
        return schedule
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for ML model."""
        # Simple min-max normalization with reasonable bounds
        bounds = np.array([
            [1, 5000],      # problem_size
            [0, 1],         # density
            [1, 1000],      # condition_number
            [0, 20]         # topology_complexity
        ])
        
        normalized = np.zeros_like(features)
        for i, (feature, (min_val, max_val)) in enumerate(zip(features, bounds)):
            normalized[i] = (feature - min_val) / (max_val - min_val)
            normalized[i] = np.clip(normalized[i], 0, 1)
        
        return normalized
    
    def _generate_multi_stage_schedule(self, problem_analysis: Dict[str, float]) -> List[Dict[str, float]]:
        """Generate multi-stage annealing schedule for complex problems."""
        stages = []
        
        # Stage 1: Slow initial annealing
        stages.append({
            'duration_fraction': 0.3,
            'slope': 0.3,
            'pause_duration': 10.0
        })
        
        # Stage 2: Fast intermediate annealing
        stages.append({
            'duration_fraction': 0.4,
            'slope': 1.5,
            'pause_duration': 5.0
        })
        
        # Stage 3: Careful final annealing
        stages.append({
            'duration_fraction': 0.3,
            'slope': 0.8,
            'pause_duration': 2.0
        })
        
        return stages
    
    def _adapt_schedule(self, 
                       initial_schedule: AnnealingSchedule,
                       results_history: List[AnnealingResult],
                       problem_analysis: Dict[str, float]) -> AnnealingSchedule:
        """Adapt annealing schedule based on previous results."""
        if not results_history:
            return initial_schedule
        
        # Analyze recent performance
        recent_results = results_history[-min(5, len(results_history)):]
        avg_quality = np.mean([r.solution_quality for r in recent_results])
        quality_trend = self._calculate_quality_trend(recent_results)
        
        # Adapt parameters based on performance
        adapted_schedule = AnnealingSchedule(
            strategy=initial_schedule.strategy,
            total_time=initial_schedule.total_time,
            pause_duration=initial_schedule.pause_duration,
            slope=initial_schedule.slope,
            quench_rate=initial_schedule.quench_rate,
            stages=initial_schedule.stages.copy()
        )
        
        # Adjust annealing time
        if avg_quality < 0.7:  # Poor performance
            adapted_schedule.total_time = min(2000, int(adapted_schedule.total_time * 1.2))
            adapted_schedule.pause_duration = min(40, adapted_schedule.pause_duration * 1.1)
        elif avg_quality > 0.9:  # Good performance
            adapted_schedule.total_time = max(1, int(adapted_schedule.total_time * 0.95))
        
        # Adjust based on trend
        if quality_trend < -0.1:  # Declining performance
            adapted_schedule.slope *= 0.8  # Slower annealing
            adapted_schedule.strategy = AnnealingStrategy.MULTI_STAGE
        
        logger.info(f"Adapted schedule: time={adapted_schedule.total_time}μs, "
                   f"pause={adapted_schedule.pause_duration}μs, quality_trend={quality_trend:.3f}")
        
        return adapted_schedule
    
    def _calculate_quality_trend(self, results: List[AnnealingResult]) -> float:
        """Calculate trend in solution quality."""
        if len(results) < 2:
            return 0.0
        
        qualities = [r.solution_quality for r in results]
        x = np.arange(len(qualities))
        
        # Simple linear regression for trend
        if len(qualities) > 1:
            slope, _ = np.polyfit(x, qualities, 1)
            return slope
        
        return 0.0
    
    def _execute_quantum_annealing(self, 
                                  qubo_matrix: np.ndarray,
                                  schedule: AnnealingSchedule,
                                  problem_analysis: Dict[str, float]) -> AnnealingResult:
        """Execute quantum annealing with given schedule."""
        start_time = time.time()
        
        # Generate embedding (simplified simulation)
        embedding_metrics = self._generate_embedding(qubo_matrix, problem_analysis)
        
        # Simulate annealing execution
        annealing_time = schedule.total_time * 1e-6  # Convert to seconds
        time.sleep(min(0.1, annealing_time * 1000))  # Simulated execution delay
        
        # Generate solution (simplified - would use actual annealer)
        solution = self._simulate_annealing_solution(qubo_matrix, schedule, embedding_metrics)
        
        execution_time = time.time() - start_time
        
        result = AnnealingResult(
            energy=solution['energy'],
            solution_vector=solution['vector'],
            success_probability=solution['success_prob'],
            annealing_time=execution_time,
            embedding_metrics=embedding_metrics,
            schedule_used=schedule,
            num_reads=1000,  # Would be configurable
            chain_breaks=solution['chain_breaks']
        )
        
        logger.info(f"Annealing executed: energy={result.energy:.4f}, "
                   f"quality={result.solution_quality:.4f}, time={execution_time:.3f}s")
        
        return result
    
    def _generate_embedding(self, qubo_matrix: np.ndarray, problem_analysis: Dict[str, float]) -> EmbeddingMetrics:
        """Generate embedding for QUBO problem on hardware."""
        n = qubo_matrix.shape[0]
        
        # Simplified embedding metrics
        embedding_ratio = min(1.0, self.hardware_constraints['max_qubits'] / (n * 2))
        
        # Chain length estimation based on problem topology
        avg_chain_length = int(2 + problem_analysis['density'] * 3)
        chain_lengths = np.random.poisson(avg_chain_length, n).tolist()
        max_chain_length = max(chain_lengths)
        
        # Connectivity utilization
        total_qubits_used = sum(chain_lengths)
        connectivity_utilization = total_qubits_used / self.hardware_constraints['max_qubits']
        
        # Embedding success probability
        complexity_penalty = problem_analysis['topology_complexity'] / 10.0
        success_probability = max(0.5, 1.0 - complexity_penalty - (max_chain_length / 20.0))
        
        embedding_time = 0.1 * n / 100  # Simplified timing model
        
        return EmbeddingMetrics(
            embedding_ratio=embedding_ratio,
            chain_lengths=chain_lengths,
            max_chain_length=max_chain_length,
            connectivity_utilization=connectivity_utilization,
            embedding_time=embedding_time,
            success_probability=success_probability
        )
    
    def _simulate_annealing_solution(self, 
                                   qubo_matrix: np.ndarray,
                                   schedule: AnnealingSchedule,
                                   embedding_metrics: EmbeddingMetrics) -> Dict[str, Any]:
        """Simulate quantum annealing solution."""
        n = qubo_matrix.shape[0]
        
        # Generate random solution and improve it
        solution_vector = np.random.randint(0, 2, n)
        
        # Calculate energy
        energy = np.dot(solution_vector.T, np.dot(qubo_matrix, solution_vector))
        
        # Simple improvement heuristic
        for _ in range(10):
            improved_solution = solution_vector.copy()
            flip_idx = np.random.randint(0, n)
            improved_solution[flip_idx] = 1 - improved_solution[flip_idx]
            
            improved_energy = np.dot(improved_solution.T, np.dot(qubo_matrix, improved_solution))
            if improved_energy < energy:
                solution_vector = improved_solution
                energy = improved_energy
        
        # Success probability based on schedule and embedding quality
        base_success = embedding_metrics.success_probability
        schedule_bonus = 0.1 if schedule.strategy == AnnealingStrategy.MULTI_STAGE else 0.0
        time_bonus = min(0.1, schedule.total_time / 10000)  # Longer annealing helps
        
        success_prob = min(1.0, base_success + schedule_bonus + time_bonus)
        
        # Chain breaks estimation
        chain_breaks = max(0, int(np.random.poisson(embedding_metrics.max_chain_length * 0.1)))
        
        return {
            'energy': energy,
            'vector': solution_vector,
            'success_prob': success_prob,
            'chain_breaks': chain_breaks
        }
    
    def _update_parameter_predictor(self, 
                                  problem_analysis: Dict[str, float],
                                  schedule: AnnealingSchedule,
                                  result: AnnealingResult) -> None:
        """Update ML model with new optimization data."""
        # Extract features and outcomes
        features = np.array([
            problem_analysis['problem_size'],
            problem_analysis['density'],
            problem_analysis['condition_number'],
            problem_analysis['topology_complexity']
        ])
        
        features = self._normalize_features(features)
        
        # Target parameters (what actually worked)
        targets = np.array([
            schedule.total_time / 1000.0,          # Normalized annealing time
            0.5,                                    # Placeholder for chain strength
            1000 / 1000.0,                         # Normalized num_reads
            schedule.pause_duration / 20.0         # Normalized pause duration
        ])
        
        # Simple gradient update
        predictor = self.parameter_predictor
        predicted = np.dot(features, predictor['weights']) + predictor['bias']
        error = targets - predicted
        
        # Weight update with quality-weighted learning rate
        effective_lr = self.learning_rate * result.solution_quality
        predictor['weights'] += effective_lr * np.outer(features, error)
        predictor['bias'] += effective_lr * error
        
        # Store parameter history
        self.parameter_history['annealing_time'].append(schedule.total_time)
        self.parameter_history['pause_duration'].append(schedule.pause_duration)
        self.parameter_history['solution_quality'].append(result.solution_quality)
        
        # Keep history bounded
        for key in self.parameter_history:
            if len(self.parameter_history[key]) > self.history_window:
                self.parameter_history[key] = self.parameter_history[key][-self.history_window:]
    
    def _calculate_improvement_rate(self, results: List[AnnealingResult]) -> float:
        """Calculate improvement rate across iterations."""
        if len(results) < 2:
            return 0.0
        
        qualities = [r.solution_quality for r in results]
        improvements = [qualities[i] - qualities[i-1] for i in range(1, len(qualities))]
        
        return np.mean(improvements) if improvements else 0.0
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}
        
        qualities = [opt['best_quality'] for opt in self.optimization_history]
        times = [opt['total_time'] for opt in self.optimization_history]
        iterations = [opt['iterations'] for opt in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "average_quality": np.mean(qualities),
            "best_quality": np.max(qualities),
            "quality_std": np.std(qualities),
            "average_time": np.mean(times),
            "average_iterations": np.mean(iterations),
            "parameter_adaptation_stats": {
                "annealing_time_range": [
                    np.min(self.parameter_history.get('annealing_time', [0])),
                    np.max(self.parameter_history.get('annealing_time', [0]))
                ],
                "quality_improvement": self._calculate_overall_improvement()
            },
            "ml_model_performance": self._evaluate_predictor_performance()
        }
    
    def _calculate_overall_improvement(self) -> float:
        """Calculate overall improvement in solution quality over time."""
        qualities = self.parameter_history.get('solution_quality', [])
        if len(qualities) < 10:
            return 0.0
        
        early_avg = np.mean(qualities[:len(qualities)//3])
        late_avg = np.mean(qualities[-len(qualities)//3:])
        
        return late_avg - early_avg
    
    def _evaluate_predictor_performance(self) -> Dict[str, float]:
        """Evaluate performance of the ML parameter predictor."""
        if len(self.parameter_history.get('solution_quality', [])) < 5:
            return {"message": "Insufficient data for evaluation"}
        
        # Simple performance metrics
        qualities = self.parameter_history['solution_quality']
        quality_variance = np.var(qualities)
        quality_trend = np.polyfit(range(len(qualities)), qualities, 1)[0] if len(qualities) > 1 else 0
        
        return {
            "prediction_stability": 1.0 / (1.0 + quality_variance),
            "learning_trend": quality_trend,
            "convergence_indicator": max(0, min(1, quality_trend / 0.1))
        }


class ComparativeAnnealingAnalyzer:
    """Analyze and compare different quantum annealing approaches."""
    
    def __init__(self):
        self.benchmark_results = []
        self.baseline_methods = ['classical_sa', 'classical_exact', 'qaoa_gate']
    
    def run_comparative_study(self, 
                            qubo_problems: List[np.ndarray],
                            methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive comparative study of annealing methods."""
        methods = methods or ['adaptive_annealing'] + self.baseline_methods
        
        logger.info(f"Starting comparative study with {len(methods)} methods on {len(qubo_problems)} problems")
        
        all_results = {}
        
        for method in methods:
            method_results = []
            
            for i, qubo_matrix in enumerate(qubo_problems):
                logger.info(f"Running {method} on problem {i+1}/{len(qubo_problems)}")
                
                start_time = time.time()
                result = self._run_method(method, qubo_matrix)
                execution_time = time.time() - start_time
                
                method_results.append({
                    'problem_index': i,
                    'problem_size': qubo_matrix.shape[0],
                    'method': method,
                    'energy': result.get('energy', float('inf')),
                    'quality': result.get('quality', 0.0),
                    'execution_time': execution_time,
                    'success_probability': result.get('success_probability', 0.0)
                })
            
            all_results[method] = method_results
        
        # Analyze results
        analysis = self._analyze_comparative_results(all_results)
        
        # Store benchmark
        benchmark_record = {
            'timestamp': time.time(),
            'num_problems': len(qubo_problems),
            'methods_compared': methods,
            'results': all_results,
            'analysis': analysis
        }
        self.benchmark_results.append(benchmark_record)
        
        logger.info("Comparative study complete")
        return analysis
    
    def _run_method(self, method: str, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Run specific optimization method on QUBO problem."""
        if method == 'adaptive_annealing':
            annealer = AdaptiveQuantumAnnealer()
            result = annealer.optimize_scheduling_problem(qubo_matrix, max_iterations=3)
            return {
                'energy': result.energy,
                'quality': result.solution_quality,
                'success_probability': result.success_probability
            }
        
        elif method == 'classical_sa':
            # Simulated annealing baseline
            return self._run_simulated_annealing(qubo_matrix)
        
        elif method == 'classical_exact':
            # Exact classical solution (for small problems)
            return self._run_exact_classical(qubo_matrix)
        
        elif method == 'qaoa_gate':
            # QAOA gate-based quantum approach
            return self._run_qaoa_simulation(qubo_matrix)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_simulated_annealing(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Run classical simulated annealing."""
        n = qubo_matrix.shape[0]
        current_solution = np.random.randint(0, 2, n)
        current_energy = np.dot(current_solution.T, np.dot(qubo_matrix, current_solution))
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temperature = 100.0
        cooling_rate = 0.99
        min_temperature = 0.01
        
        while temperature > min_temperature:
            # Generate neighbor
            neighbor = current_solution.copy()
            flip_idx = np.random.randint(0, n)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            neighbor_energy = np.dot(neighbor.T, np.dot(qubo_matrix, neighbor))
            
            # Accept or reject
            if neighbor_energy < current_energy or np.random.random() < np.exp(-(neighbor_energy - current_energy) / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            
            temperature *= cooling_rate
        
        return {
            'energy': best_energy,
            'quality': max(0, 1.0 - abs(best_energy) / max(abs(best_energy) + 1, 1)),
            'success_probability': 1.0
        }
    
    def _run_exact_classical(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Run exact classical solution (brute force for small problems)."""
        n = qubo_matrix.shape[0]
        
        if n > 20:  # Too large for exact solution
            return {'energy': float('inf'), 'quality': 0.0, 'success_probability': 0.0}
        
        best_energy = float('inf')
        best_solution = None
        
        # Try all 2^n possible solutions
        for i in range(2**n):
            solution = np.array([(i >> j) & 1 for j in range(n)])
            energy = np.dot(solution.T, np.dot(qubo_matrix, solution))
            
            if energy < best_energy:
                best_energy = energy
                best_solution = solution
        
        return {
            'energy': best_energy,
            'quality': 1.0,  # Exact solution
            'success_probability': 1.0
        }
    
    def _run_qaoa_simulation(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Run QAOA simulation."""
        # Simplified QAOA simulation
        n = qubo_matrix.shape[0]
        
        # Generate random solution with some bias towards good solutions
        solution = np.random.randint(0, 2, n)
        energy = np.dot(solution.T, np.dot(qubo_matrix, solution))
        
        # Simple improvement (would be quantum optimization in real QAOA)
        for _ in range(min(10, n)):
            candidate = solution.copy()
            flip_idx = np.random.randint(0, n)
            candidate[flip_idx] = 1 - candidate[flip_idx]
            
            candidate_energy = np.dot(candidate.T, np.dot(qubo_matrix, candidate))
            if candidate_energy < energy:
                solution = candidate
                energy = candidate_energy
        
        # Simulate quantum noise
        success_prob = max(0.7, 1.0 - n / 100.0)
        
        return {
            'energy': energy,
            'quality': success_prob * 0.9,  # QAOA typically gets good but not perfect solutions
            'success_probability': success_prob
        }
    
    def _analyze_comparative_results(self, all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze comparative results across methods."""
        analysis = {}
        
        methods = list(all_results.keys())
        
        # Performance comparison
        for metric in ['energy', 'quality', 'execution_time']:
            analysis[f'{metric}_comparison'] = {}
            
            for method in methods:
                values = [result[metric] for result in all_results[method]]
                analysis[f'{metric}_comparison'][method] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Ranking analysis
        problem_rankings = []
        for i in range(len(all_results[methods[0]])):
            problem_results = {method: all_results[method][i] for method in methods}
            
            # Rank by quality (higher is better)
            ranked_methods = sorted(methods, key=lambda m: problem_results[m]['quality'], reverse=True)
            problem_rankings.append({
                'problem_index': i,
                'ranking': {method: ranked_methods.index(method) + 1 for method in methods}
            })
        
        analysis['method_rankings'] = self._calculate_average_rankings(problem_rankings, methods)
        
        # Statistical significance testing (simplified)
        analysis['statistical_tests'] = self._run_statistical_tests(all_results)
        
        # Quantum advantage analysis
        if 'adaptive_annealing' in methods:
            analysis['quantum_advantage'] = self._analyze_quantum_advantage(all_results)
        
        return analysis
    
    def _calculate_average_rankings(self, problem_rankings: List[Dict], methods: List[str]) -> Dict[str, float]:
        """Calculate average ranking for each method."""
        rankings = {method: [] for method in methods}
        
        for problem in problem_rankings:
            for method, rank in problem['ranking'].items():
                rankings[method].append(rank)
        
        return {method: np.mean(ranks) for method, ranks in rankings.items()}
    
    def _run_statistical_tests(self, all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Run statistical significance tests."""
        # Simplified statistical analysis
        methods = list(all_results.keys())
        
        if len(methods) < 2:
            return {"message": "Need at least 2 methods for comparison"}
        
        tests = {}
        
        # Pairwise comparisons of quality
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                values1 = [r['quality'] for r in all_results[method1]]
                values2 = [r['quality'] for r in all_results[method2]]
                
                # Simple t-test approximation
                mean_diff = np.mean(values1) - np.mean(values2)
                pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
                
                if pooled_std > 0:
                    t_stat = mean_diff / (pooled_std / np.sqrt(len(values1)))
                    p_value_approx = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(len(values1) - 2)))
                else:
                    t_stat = 0
                    p_value_approx = 1.0
                
                tests[f'{method1}_vs_{method2}'] = {
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value_approx': max(0, min(1, p_value_approx)),
                    'significant': p_value_approx < 0.05
                }
        
        return tests
    
    def _analyze_quantum_advantage(self, all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze quantum advantage of adaptive annealing."""
        annealing_results = all_results.get('adaptive_annealing', [])
        
        if not annealing_results:
            return {"message": "No quantum annealing results found"}
        
        # Compare with best classical method
        classical_methods = ['classical_sa', 'classical_exact']
        best_classical_results = []
        
        for i in range(len(annealing_results)):
            problem_classical_results = []
            for method in classical_methods:
                if method in all_results and i < len(all_results[method]):
                    problem_classical_results.append(all_results[method][i]['quality'])
            
            if problem_classical_results:
                best_classical_results.append(max(problem_classical_results))
            else:
                best_classical_results.append(0.0)
        
        # Calculate advantage metrics
        quantum_qualities = [r['quality'] for r in annealing_results]
        advantages = [q - c for q, c in zip(quantum_qualities, best_classical_results)]
        
        return {
            'average_advantage': np.mean(advantages),
            'advantage_std': np.std(advantages),
            'problems_with_advantage': sum(1 for a in advantages if a > 0),
            'advantage_rate': sum(1 for a in advantages if a > 0) / len(advantages),
            'max_advantage': np.max(advantages),
            'significant_advantage_threshold': 0.1,
            'problems_with_significant_advantage': sum(1 for a in advantages if a > 0.1)
        }
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report."""
        if not self.benchmark_results:
            return "No benchmark data available for report generation."
        
        latest_benchmark = self.benchmark_results[-1]
        analysis = latest_benchmark['analysis']
        
        report = []
        report.append("# Quantum Annealing Optimization Research Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Executive Summary")
        report.append(f"Comparative analysis of {len(latest_benchmark['methods_compared'])} optimization methods")
        report.append(f"across {latest_benchmark['num_problems']} QUBO problems.")
        report.append("")
        
        # Performance comparison
        report.append("## Performance Comparison")
        if 'quality_comparison' in analysis:
            report.append("### Solution Quality")
            for method, stats in analysis['quality_comparison'].items():
                report.append(f"- **{method}**: {stats['mean']:.4f} ± {stats['std']:.4f}")
            report.append("")
        
        # Rankings
        if 'method_rankings' in analysis:
            report.append("### Method Rankings (lower is better)")
            rankings = analysis['method_rankings']
            sorted_methods = sorted(rankings.items(), key=lambda x: x[1])
            for i, (method, avg_rank) in enumerate(sorted_methods):
                report.append(f"{i+1}. **{method}**: {avg_rank:.2f}")
            report.append("")
        
        # Quantum advantage
        if 'quantum_advantage' in analysis:
            qa = analysis['quantum_advantage']
            report.append("## Quantum Advantage Analysis")
            report.append(f"- Average quantum advantage: {qa['average_advantage']:.4f}")
            report.append(f"- Problems showing advantage: {qa['problems_with_advantage']}/{latest_benchmark['num_problems']}")
            report.append(f"- Quantum advantage rate: {qa['advantage_rate']:.1%}")
            report.append("")
        
        # Statistical significance
        if 'statistical_tests' in analysis:
            report.append("## Statistical Significance")
            tests = analysis['statistical_tests']
            significant_comparisons = [k for k, v in tests.items() if v.get('significant', False)]
            report.append(f"Significant performance differences found in {len(significant_comparisons)} comparisons:")
            for comparison in significant_comparisons:
                test_result = tests[comparison]
                report.append(f"- {comparison}: p < 0.05, difference = {test_result['mean_difference']:.4f}")
            report.append("")
        
        report.append("## Conclusions and Recommendations")
        report.append("Based on the comparative analysis:")
        
        if 'method_rankings' in analysis:
            best_method = min(analysis['method_rankings'].items(), key=lambda x: x[1])[0]
            report.append(f"- **Best overall method**: {best_method}")
        
        if 'quantum_advantage' in analysis and analysis['quantum_advantage']['advantage_rate'] > 0.5:
            report.append("- Quantum annealing shows consistent advantage over classical methods")
            report.append("- Recommend quantum annealing for production deployment")
        else:
            report.append("- Classical methods remain competitive")
            report.append("- Consider hybrid approaches for optimal performance")
        
        return "\n".join(report)