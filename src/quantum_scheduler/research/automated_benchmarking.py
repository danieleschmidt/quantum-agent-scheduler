"""Automated Benchmarking Framework for Quantum Scheduling Research.

This module provides comprehensive automated benchmarking capabilities for
comparative analysis of quantum vs classical scheduling algorithms with
statistical rigor and reproducible experimental design.
"""

import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProblemClass(Enum):
    """Different classes of scheduling problems."""
    SMALL_SPARSE = "small_sparse"
    MEDIUM_DENSE = "medium_dense" 
    LARGE_SPARSE = "large_sparse"
    ULTRA_LARGE = "ultra_large"
    STRUCTURED = "structured"
    RANDOM = "random"


class MetricType(Enum):
    """Types of performance metrics."""
    SOLUTION_QUALITY = "solution_quality"
    EXECUTION_TIME = "execution_time"
    ENERGY = "energy"
    SUCCESS_RATE = "success_rate"
    SCALABILITY = "scalability"
    CONVERGENCE_RATE = "convergence_rate"


@dataclass
class BenchmarkProblem:
    """Definition of a benchmark problem."""
    problem_id: str
    problem_class: ProblemClass
    qubo_matrix: np.ndarray
    optimal_solution: Optional[np.ndarray] = None
    optimal_energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate problem characteristics after initialization."""
        n = self.qubo_matrix.shape[0]
        density = np.count_nonzero(self.qubo_matrix) / (n * n)
        
        self.metadata.update({
            'problem_size': n,
            'density': density,
            'sparsity': 1.0 - density,
            'matrix_norm': float(np.linalg.norm(self.qubo_matrix)),
            'trace': float(np.trace(self.qubo_matrix))
        })


@dataclass
class ExperimentResult:
    """Result from a single benchmark experiment."""
    problem_id: str
    method_name: str
    solution_vector: np.ndarray
    energy: float
    execution_time: float
    success_probability: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def solution_quality(self) -> float:
        """Calculate solution quality score."""
        return self.success_probability * (1.0 / (1.0 + abs(self.energy)))


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of benchmark results."""
    metric_name: str
    method_comparisons: Dict[str, Dict[str, float]]
    significance_tests: Dict[str, Dict[str, Any]]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class ProblemGenerator:
    """Generate diverse benchmark problems for comprehensive testing."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize problem generator.
        
        Args:
            random_seed: Seed for reproducible problem generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.generation_history = []
    
    def generate_problem_suite(self, 
                              problem_classes: List[ProblemClass],
                              problems_per_class: int = 10) -> List[BenchmarkProblem]:
        """Generate comprehensive suite of benchmark problems.
        
        Args:
            problem_classes: Types of problems to generate
            problems_per_class: Number of problems per class
            
        Returns:
            List of generated benchmark problems
        """
        problems = []
        
        for problem_class in problem_classes:
            logger.info(f"Generating {problems_per_class} problems of class {problem_class.value}")
            
            for i in range(problems_per_class):
                problem = self._generate_single_problem(problem_class, i)
                problems.append(problem)
        
        logger.info(f"Generated {len(problems)} benchmark problems")
        return problems
    
    def _generate_single_problem(self, problem_class: ProblemClass, index: int) -> BenchmarkProblem:
        """Generate a single benchmark problem."""
        problem_id = f"{problem_class.value}_{index:03d}"
        
        if problem_class == ProblemClass.SMALL_SPARSE:
            return self._generate_small_sparse_problem(problem_id)
        elif problem_class == ProblemClass.MEDIUM_DENSE:
            return self._generate_medium_dense_problem(problem_id)
        elif problem_class == ProblemClass.LARGE_SPARSE:
            return self._generate_large_sparse_problem(problem_id)
        elif problem_class == ProblemClass.ULTRA_LARGE:
            return self._generate_ultra_large_problem(problem_id)
        elif problem_class == ProblemClass.STRUCTURED:
            return self._generate_structured_problem(problem_id)
        elif problem_class == ProblemClass.RANDOM:
            return self._generate_random_problem(problem_id)
        else:
            raise ValueError(f"Unknown problem class: {problem_class}")
    
    def _generate_small_sparse_problem(self, problem_id: str) -> BenchmarkProblem:
        """Generate small sparse scheduling problem."""
        n = np.random.randint(10, 25)  # 10-24 variables
        density = np.random.uniform(0.1, 0.3)  # 10-30% density
        
        qubo_matrix = self._create_sparse_matrix(n, density)
        optimal_solution, optimal_energy = self._find_optimal_solution(qubo_matrix)
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_class=ProblemClass.SMALL_SPARSE,
            qubo_matrix=qubo_matrix,
            optimal_solution=optimal_solution,
            optimal_energy=optimal_energy,
            generation_params={'target_density': density}
        )
    
    def _generate_medium_dense_problem(self, problem_id: str) -> BenchmarkProblem:
        """Generate medium-sized dense scheduling problem."""
        n = np.random.randint(25, 60)  # 25-59 variables
        density = np.random.uniform(0.4, 0.8)  # 40-80% density
        
        qubo_matrix = self._create_dense_matrix(n, density)
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_class=ProblemClass.MEDIUM_DENSE,
            qubo_matrix=qubo_matrix,
            generation_params={'target_density': density}
        )
    
    def _generate_large_sparse_problem(self, problem_id: str) -> BenchmarkProblem:
        """Generate large sparse scheduling problem."""
        n = np.random.randint(60, 150)  # 60-149 variables
        density = np.random.uniform(0.05, 0.2)  # 5-20% density
        
        qubo_matrix = self._create_sparse_matrix(n, density)
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_class=ProblemClass.LARGE_SPARSE,
            qubo_matrix=qubo_matrix,
            generation_params={'target_density': density}
        )
    
    def _generate_ultra_large_problem(self, problem_id: str) -> BenchmarkProblem:
        """Generate ultra-large scheduling problem."""
        n = np.random.randint(150, 500)  # 150-499 variables
        density = np.random.uniform(0.01, 0.1)  # 1-10% density
        
        qubo_matrix = self._create_sparse_matrix(n, density)
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_class=ProblemClass.ULTRA_LARGE,
            qubo_matrix=qubo_matrix,
            generation_params={'target_density': density}
        )
    
    def _generate_structured_problem(self, problem_id: str) -> BenchmarkProblem:
        """Generate structured scheduling problem (e.g., grid, hierarchical)."""
        structure_type = np.random.choice(['grid', 'hierarchical', 'clustered'])
        
        if structure_type == 'grid':
            n = 36  # 6x6 grid
            qubo_matrix = self._create_grid_structured_matrix(6, 6)
        elif structure_type == 'hierarchical':
            n = 63  # 3-level hierarchy
            qubo_matrix = self._create_hierarchical_matrix(3, 3)
        else:  # clustered
            n = 45  # 3 clusters of 15
            qubo_matrix = self._create_clustered_matrix([15, 15, 15])
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_class=ProblemClass.STRUCTURED,
            qubo_matrix=qubo_matrix,
            generation_params={'structure_type': structure_type}
        )
    
    def _generate_random_problem(self, problem_id: str) -> BenchmarkProblem:
        """Generate completely random scheduling problem."""
        n = np.random.randint(20, 100)  # Random size
        density = np.random.uniform(0.1, 0.9)  # Random density
        
        qubo_matrix = np.random.normal(0, 1, (n, n))
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2  # Make symmetric
        
        # Zero out elements to achieve target density
        mask = np.random.random((n, n)) < density
        qubo_matrix *= mask
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_class=ProblemClass.RANDOM,
            qubo_matrix=qubo_matrix,
            generation_params={'target_density': density}
        )
    
    def _create_sparse_matrix(self, n: int, density: float) -> np.ndarray:
        """Create sparse QUBO matrix with specific density."""
        # Generate random symmetric matrix
        matrix = np.random.normal(0, 1, (n, n))
        matrix = (matrix + matrix.T) / 2
        
        # Create sparsity mask
        mask = np.random.random((n, n)) < density
        # Ensure diagonal is preserved
        np.fill_diagonal(mask, True)
        
        matrix *= mask
        return matrix
    
    def _create_dense_matrix(self, n: int, density: float) -> np.ndarray:
        """Create dense QUBO matrix."""
        matrix = np.random.normal(0, 2, (n, n))
        matrix = (matrix + matrix.T) / 2
        
        # Add structure to make it more realistic
        for i in range(n):
            for j in range(i+1, min(i+5, n)):  # Local connections
                if np.random.random() < 0.8:
                    matrix[i, j] *= 1.5
                    matrix[j, i] = matrix[i, j]
        
        # Apply density mask
        mask = np.random.random((n, n)) < density
        np.fill_diagonal(mask, True)
        matrix *= mask
        
        return matrix
    
    def _create_grid_structured_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Create grid-structured QUBO matrix."""
        n = rows * cols
        matrix = np.zeros((n, n))
        
        # Add grid connections
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                
                # Diagonal term
                matrix[idx, idx] = np.random.normal(0, 1)
                
                # Right neighbor
                if j < cols - 1:
                    neighbor_idx = i * cols + (j + 1)
                    weight = np.random.normal(0, 0.5)
                    matrix[idx, neighbor_idx] = weight
                    matrix[neighbor_idx, idx] = weight
                
                # Bottom neighbor
                if i < rows - 1:
                    neighbor_idx = (i + 1) * cols + j
                    weight = np.random.normal(0, 0.5)
                    matrix[idx, neighbor_idx] = weight
                    matrix[neighbor_idx, idx] = weight
        
        return matrix
    
    def _create_hierarchical_matrix(self, levels: int, branching: int) -> np.ndarray:
        """Create hierarchical QUBO matrix."""
        # Simple hierarchical structure
        n = sum(branching ** i for i in range(levels))
        matrix = np.random.normal(0, 0.5, (n, n))
        matrix = (matrix + matrix.T) / 2
        
        # Add hierarchical connections
        level_start = 0
        for level in range(levels - 1):
            level_size = branching ** level
            next_level_size = branching ** (level + 1)
            
            for i in range(level_size):
                parent_idx = level_start + i
                for j in range(branching):
                    child_idx = level_start + level_size + i * branching + j
                    if child_idx < n:
                        weight = np.random.normal(1, 0.2)
                        matrix[parent_idx, child_idx] = weight
                        matrix[child_idx, parent_idx] = weight
            
            level_start += level_size
        
        return matrix
    
    def _create_clustered_matrix(self, cluster_sizes: List[int]) -> np.ndarray:
        """Create clustered QUBO matrix."""
        n = sum(cluster_sizes)
        matrix = np.zeros((n, n))
        
        start_idx = 0
        for cluster_size in cluster_sizes:
            end_idx = start_idx + cluster_size
            
            # Dense connections within cluster
            cluster_matrix = np.random.normal(0, 1, (cluster_size, cluster_size))
            cluster_matrix = (cluster_matrix + cluster_matrix.T) / 2
            matrix[start_idx:end_idx, start_idx:end_idx] = cluster_matrix
            
            start_idx = end_idx
        
        # Sparse connections between clusters
        for i in range(len(cluster_sizes)):
            for j in range(i + 1, len(cluster_sizes)):
                start_i = sum(cluster_sizes[:i])
                end_i = start_i + cluster_sizes[i]
                start_j = sum(cluster_sizes[:j])
                end_j = start_j + cluster_sizes[j]
                
                # Add few random connections
                num_connections = max(1, cluster_sizes[i] * cluster_sizes[j] // 20)
                for _ in range(num_connections):
                    idx_i = np.random.randint(start_i, end_i)
                    idx_j = np.random.randint(start_j, end_j)
                    weight = np.random.normal(0, 0.3)
                    matrix[idx_i, idx_j] = weight
                    matrix[idx_j, idx_i] = weight
        
        return matrix
    
    def _find_optimal_solution(self, qubo_matrix: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Find optimal solution for small problems using brute force."""
        n = qubo_matrix.shape[0]
        
        if n > 20:  # Too large for exact solution
            return None, None
        
        best_energy = float('inf')
        best_solution = None
        
        for i in range(2**n):
            solution = np.array([(i >> j) & 1 for j in range(n)])
            energy = np.dot(solution.T, np.dot(qubo_matrix, solution))
            
            if energy < best_energy:
                best_energy = energy
                best_solution = solution
        
        return best_solution, best_energy


class AutomatedBenchmarkRunner:
    """Automated benchmark execution and analysis framework."""
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "./benchmark_results",
                 num_workers: int = 4,
                 cache_results: bool = True):
        """Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            num_workers: Number of parallel workers
            cache_results: Whether to cache results for faster reruns
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.cache_results = cache_results
        
        # Initialize cache
        self.cache_file = self.output_dir / "benchmark_cache.pkl"
        self.result_cache = self._load_cache() if cache_results else {}
        
        # Registered methods
        self.registered_methods = {}
        
        logger.info(f"Initialized benchmark runner with {num_workers} workers")
    
    def register_method(self, 
                       name: str, 
                       method_func: Callable[[np.ndarray], Dict[str, Any]],
                       description: str = ""):
        """Register a method for benchmarking.
        
        Args:
            name: Method name
            method_func: Function that takes QUBO matrix and returns results dict
            description: Method description
        """
        self.registered_methods[name] = {
            'function': method_func,
            'description': description
        }
        logger.info(f"Registered method: {name}")
    
    def run_benchmark_suite(self, 
                          problems: List[BenchmarkProblem],
                          methods: Optional[List[str]] = None,
                          num_runs: int = 3,
                          timeout: float = 300.0) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Args:
            problems: List of benchmark problems
            methods: List of method names to benchmark (None for all registered)
            num_runs: Number of runs per problem-method combination
            timeout: Timeout per method execution in seconds
            
        Returns:
            Comprehensive benchmark results
        """
        if methods is None:
            methods = list(self.registered_methods.keys())
        
        logger.info(f"Starting benchmark suite: {len(problems)} problems × {len(methods)} methods × {num_runs} runs")
        
        # Generate all experiment configurations
        experiments = []
        for problem in problems:
            for method_name in methods:
                for run_id in range(num_runs):
                    experiment_id = f"{problem.problem_id}_{method_name}_{run_id}"
                    experiments.append({
                        'experiment_id': experiment_id,
                        'problem': problem,
                        'method_name': method_name,
                        'run_id': run_id
                    })
        
        # Execute experiments in parallel
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(self._run_single_experiment, exp, timeout): exp
                for exp in experiments
            }
            
            # Collect results
            for future in as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    if completed % 50 == 0:  # Progress update
                        logger.info(f"Completed {completed}/{len(experiments)} experiments")
                        
                except Exception as e:
                    logger.error(f"Experiment {experiment['experiment_id']} failed: {e}")
        
        logger.info(f"Benchmark suite complete: {len(results)} successful runs")
        
        # Save raw results
        self._save_results(results)
        
        # Perform analysis
        analysis = self._analyze_results(results, problems, methods)
        
        # Save analysis
        self._save_analysis(analysis)
        
        return {
            'raw_results': results,
            'analysis': analysis,
            'metadata': {
                'num_problems': len(problems),
                'num_methods': len(methods),
                'num_runs': num_runs,
                'total_experiments': len(experiments),
                'successful_experiments': len(results)
            }
        }
    
    def _run_single_experiment(self, 
                              experiment: Dict[str, Any], 
                              timeout: float) -> Optional[ExperimentResult]:
        """Run a single benchmark experiment."""
        experiment_id = experiment['experiment_id']
        
        # Check cache first
        if self.cache_results and experiment_id in self.result_cache:
            return self.result_cache[experiment_id]
        
        problem = experiment['problem']
        method_name = experiment['method_name']
        
        if method_name not in self.registered_methods:
            logger.error(f"Method {method_name} not registered")
            return None
        
        method_func = self.registered_methods[method_name]['function']
        
        try:
            # Execute method with timeout
            start_time = time.time()
            
            # Run the method
            method_result = method_func(problem.qubo_matrix)
            
            execution_time = time.time() - start_time
            
            if execution_time > timeout:
                logger.warning(f"Experiment {experiment_id} exceeded timeout")
                return None
            
            # Create result object
            result = ExperimentResult(
                problem_id=problem.problem_id,
                method_name=method_name,
                solution_vector=method_result.get('solution_vector', np.array([])),
                energy=method_result.get('energy', float('inf')),
                execution_time=execution_time,
                success_probability=method_result.get('success_probability', 0.0),
                additional_metrics=method_result.get('additional_metrics', {})
            )
            
            # Cache result
            if self.cache_results:
                self.result_cache[experiment_id] = result
                self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in experiment {experiment_id}: {e}")
            return None
    
    def _analyze_results(self, 
                        results: List[ExperimentResult], 
                        problems: List[BenchmarkProblem],
                        methods: List[str]) -> Dict[str, Any]:
        """Perform comprehensive analysis of benchmark results."""
        logger.info("Performing statistical analysis of results")
        
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame([asdict(result) for result in results])
        
        analysis = {}
        
        # Performance summary by method
        analysis['method_performance'] = self._analyze_method_performance(df)
        
        # Problem class analysis
        analysis['problem_class_analysis'] = self._analyze_by_problem_class(df, problems)
        
        # Scalability analysis
        analysis['scalability_analysis'] = self._analyze_scalability(df, problems)
        
        # Statistical significance tests
        analysis['statistical_tests'] = self._perform_statistical_tests(df, methods)
        
        # Quantum advantage analysis
        analysis['quantum_advantage'] = self._analyze_quantum_advantage(df)
        
        # Performance correlation analysis
        analysis['correlation_analysis'] = self._analyze_correlations(df, problems)
        
        return analysis
    
    def _analyze_method_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by method."""
        method_stats = {}
        
        for method in df['method_name'].unique():
            method_df = df[df['method_name'] == method]
            
            method_stats[method] = {
                'num_experiments': len(method_df),
                'avg_solution_quality': method_df['solution_quality'].mean(),
                'std_solution_quality': method_df['solution_quality'].std(),
                'avg_execution_time': method_df['execution_time'].mean(),
                'std_execution_time': method_df['execution_time'].std(),
                'avg_energy': method_df['energy'].mean(),
                'success_rate': (method_df['success_probability'] > 0.5).mean(),
                'reliability': method_df['success_probability'].mean()
            }
        
        return method_stats
    
    def _analyze_by_problem_class(self, 
                                 df: pd.DataFrame, 
                                 problems: List[BenchmarkProblem]) -> Dict[str, Any]:
        """Analyze performance by problem class."""
        # Create problem class mapping
        problem_class_map = {p.problem_id: p.problem_class.value for p in problems}
        df['problem_class'] = df['problem_id'].map(problem_class_map)
        
        class_analysis = {}
        
        for problem_class in df['problem_class'].unique():
            if pd.isna(problem_class):
                continue
                
            class_df = df[df['problem_class'] == problem_class]
            
            # Performance by method within this class
            method_performance = {}
            for method in class_df['method_name'].unique():
                method_class_df = class_df[class_df['method_name'] == method]
                method_performance[method] = {
                    'avg_quality': method_class_df['solution_quality'].mean(),
                    'avg_time': method_class_df['execution_time'].mean(),
                    'success_rate': (method_class_df['success_probability'] > 0.5).mean()
                }
            
            class_analysis[problem_class] = {
                'num_problems': len(class_df['problem_id'].unique()),
                'num_experiments': len(class_df),
                'method_performance': method_performance,
                'best_method': max(method_performance.keys(), 
                                 key=lambda m: method_performance[m]['avg_quality']) if method_performance else None
            }
        
        return class_analysis
    
    def _analyze_scalability(self, 
                           df: pd.DataFrame, 
                           problems: List[BenchmarkProblem]) -> Dict[str, Any]:
        """Analyze scalability with problem size."""
        # Add problem size information
        size_map = {p.problem_id: p.metadata['problem_size'] for p in problems}
        df['problem_size'] = df['problem_id'].map(size_map)
        
        scalability_analysis = {}
        
        for method in df['method_name'].unique():
            method_df = df[df['method_name'] == method]
            
            # Bin by problem size
            size_bins = [0, 25, 50, 100, 200, float('inf')]
            bin_labels = ['tiny', 'small', 'medium', 'large', 'huge']
            
            method_df['size_bin'] = pd.cut(method_df['problem_size'], 
                                         bins=size_bins, labels=bin_labels)
            
            bin_analysis = {}
            for bin_label in bin_labels:
                bin_df = method_df[method_df['size_bin'] == bin_label]
                if len(bin_df) > 0:
                    bin_analysis[bin_label] = {
                        'count': len(bin_df),
                        'avg_quality': bin_df['solution_quality'].mean(),
                        'avg_time': bin_df['execution_time'].mean(),
                        'time_std': bin_df['execution_time'].std()
                    }
            
            # Calculate scaling coefficients
            if len(method_df) > 5:
                # Simple linear regression for time vs size
                sizes = method_df['problem_size'].values
                times = method_df['execution_time'].values
                
                if len(sizes) > 1 and np.std(sizes) > 0:
                    scaling_coeff = np.corrcoef(sizes, times)[0, 1]
                else:
                    scaling_coeff = 0.0
            else:
                scaling_coeff = 0.0
            
            scalability_analysis[method] = {
                'size_bin_analysis': bin_analysis,
                'time_size_correlation': scaling_coeff,
                'scalability_rating': self._rate_scalability(scaling_coeff)
            }
        
        return scalability_analysis
    
    def _rate_scalability(self, correlation: float) -> str:
        """Rate scalability based on time-size correlation."""
        if correlation < 0.3:
            return "excellent"
        elif correlation < 0.6:
            return "good"
        elif correlation < 0.8:
            return "moderate"
        else:
            return "poor"
    
    def _perform_statistical_tests(self, 
                                  df: pd.DataFrame, 
                                  methods: List[str]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        if len(methods) < 2:
            return {"message": "Need at least 2 methods for statistical tests"}
        
        tests = {}
        
        # Pairwise comparisons for solution quality
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                method1_data = df[df['method_name'] == method1]['solution_quality']
                method2_data = df[df['method_name'] == method2]['solution_quality']
                
                if len(method1_data) > 0 and len(method2_data) > 0:
                    # Perform t-test (simplified)
                    mean1, mean2 = method1_data.mean(), method2_data.mean()
                    var1, var2 = method1_data.var(), method2_data.var()
                    n1, n2 = len(method1_data), len(method2_data)
                    
                    # Pooled standard error
                    pooled_se = np.sqrt(var1/n1 + var2/n2)
                    
                    if pooled_se > 0:
                        t_stat = (mean1 - mean2) / pooled_se
                        # Approximate p-value
                        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(n1 + n2 - 2)))
                        p_value = max(0, min(1, p_value))
                    else:
                        t_stat = 0
                        p_value = 1.0
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((var1 * (n1-1) + var2 * (n2-1)) / (n1 + n2 - 2))
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    tests[f"{method1}_vs_{method2}"] = {
                        'mean_difference': mean1 - mean2,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
                        'significant': p_value < 0.05,
                        'sample_sizes': [n1, n2]
                    }
        
        return tests
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_quantum_advantage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quantum advantage across methods."""
        quantum_methods = [method for method in df['method_name'].unique() 
                          if 'quantum' in method.lower() or 'annealing' in method.lower()]
        classical_methods = [method for method in df['method_name'].unique() 
                           if method not in quantum_methods]
        
        if not quantum_methods or not classical_methods:
            return {"message": "Need both quantum and classical methods for advantage analysis"}
        
        advantage_analysis = {}
        
        for quantum_method in quantum_methods:
            quantum_data = df[df['method_name'] == quantum_method]
            
            # Find best classical performance for each problem
            problem_advantages = []
            
            for problem_id in quantum_data['problem_id'].unique():
                quantum_quality = quantum_data[quantum_data['problem_id'] == problem_id]['solution_quality'].mean()
                
                classical_qualities = []
                for classical_method in classical_methods:
                    classical_data = df[(df['method_name'] == classical_method) & 
                                      (df['problem_id'] == problem_id)]
                    if len(classical_data) > 0:
                        classical_qualities.append(classical_data['solution_quality'].mean())
                
                if classical_qualities:
                    best_classical_quality = max(classical_qualities)
                    advantage = quantum_quality - best_classical_quality
                    problem_advantages.append(advantage)
            
            if problem_advantages:
                advantage_analysis[quantum_method] = {
                    'average_advantage': np.mean(problem_advantages),
                    'advantage_std': np.std(problem_advantages),
                    'problems_with_advantage': sum(1 for a in problem_advantages if a > 0),
                    'total_problems': len(problem_advantages),
                    'advantage_rate': sum(1 for a in problem_advantages if a > 0) / len(problem_advantages),
                    'significant_advantage_rate': sum(1 for a in problem_advantages if a > 0.1) / len(problem_advantages),
                    'max_advantage': max(problem_advantages) if problem_advantages else 0
                }
        
        return advantage_analysis
    
    def _analyze_correlations(self, 
                            df: pd.DataFrame, 
                            problems: List[BenchmarkProblem]) -> Dict[str, Any]:
        """Analyze correlations between problem characteristics and performance."""
        # Add problem characteristics to dataframe
        char_map = {p.problem_id: p.metadata for p in problems}
        
        for char in ['problem_size', 'density', 'matrix_norm']:
            df[char] = df['problem_id'].map(lambda pid: char_map.get(pid, {}).get(char, 0))
        
        correlations = {}
        
        for method in df['method_name'].unique():
            method_df = df[df['method_name'] == method]
            
            method_correlations = {}
            for char in ['problem_size', 'density', 'matrix_norm']:
                if len(method_df) > 3 and method_df[char].std() > 0:
                    corr_quality = np.corrcoef(method_df[char], method_df['solution_quality'])[0, 1]
                    corr_time = np.corrcoef(method_df[char], method_df['execution_time'])[0, 1]
                    
                    method_correlations[char] = {
                        'quality_correlation': corr_quality if not np.isnan(corr_quality) else 0,
                        'time_correlation': corr_time if not np.isnan(corr_time) else 0
                    }
            
            correlations[method] = method_correlations
        
        return correlations
    
    def _save_results(self, results: List[ExperimentResult]) -> None:
        """Save raw results to disk."""
        results_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        
        with open(results_file, 'w') as f:
            json.dump([asdict(result) for result in results], f, indent=2, default=str)
        
        logger.info(f"Saved results to {results_file}")
    
    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save analysis to disk."""
        analysis_file = self.output_dir / f"benchmark_analysis_{int(time.time())}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Saved analysis to {analysis_file}")
    
    def _load_cache(self) -> Dict[str, ExperimentResult]:
        """Load cached results."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def _save_cache(self) -> None:
        """Save results cache."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.result_cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def generate_research_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive research report from analysis."""
        report = []
        
        report.append("# Automated Quantum Scheduling Benchmark Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        if 'method_performance' in analysis:
            num_methods = len(analysis['method_performance'])
            best_method = max(analysis['method_performance'].items(), 
                            key=lambda x: x[1]['avg_solution_quality'])
            report.append(f"Compared {num_methods} optimization methods across multiple problem classes.")
            report.append(f"**Best performing method**: {best_method[0]} (avg quality: {best_method[1]['avg_solution_quality']:.4f})")
        report.append("")
        
        # Method Performance
        report.append("## Method Performance Summary")
        if 'method_performance' in analysis:
            report.append("| Method | Avg Quality | Avg Time (s) | Success Rate | Reliability |")
            report.append("|--------|-------------|--------------|--------------|-------------|")
            
            for method, stats in analysis['method_performance'].items():
                report.append(f"| {method} | {stats['avg_solution_quality']:.4f} | "
                            f"{stats['avg_execution_time']:.3f} | {stats['success_rate']:.2%} | "
                            f"{stats['reliability']:.3f} |")
        report.append("")
        
        # Problem Class Analysis
        if 'problem_class_analysis' in analysis:
            report.append("## Performance by Problem Class")
            for problem_class, class_data in analysis['problem_class_analysis'].items():
                report.append(f"### {problem_class}")
                report.append(f"- Problems: {class_data['num_problems']}")
                report.append(f"- Best method: {class_data.get('best_method', 'N/A')}")
                report.append("")
        
        # Scalability Analysis  
        if 'scalability_analysis' in analysis:
            report.append("## Scalability Analysis")
            report.append("| Method | Scalability Rating | Time-Size Correlation |")
            report.append("|--------|--------------------|----------------------|")
            
            for method, scalability in analysis['scalability_analysis'].items():
                report.append(f"| {method} | {scalability['scalability_rating']} | "
                            f"{scalability['time_size_correlation']:.3f} |")
        report.append("")
        
        # Statistical Significance
        if 'statistical_tests' in analysis:
            report.append("## Statistical Significance Tests")
            significant_tests = {k: v for k, v in analysis['statistical_tests'].items() if v.get('significant', False)}
            report.append(f"Found {len(significant_tests)} significant performance differences:")
            
            for comparison, test_data in significant_tests.items():
                report.append(f"- **{comparison}**: difference = {test_data['mean_difference']:.4f}, "
                            f"effect size = {test_data['effect_size_interpretation']}")
        report.append("")
        
        # Quantum Advantage
        if 'quantum_advantage' in analysis:
            report.append("## Quantum Advantage Analysis")
            for method, advantage_data in analysis['quantum_advantage'].items():
                if isinstance(advantage_data, dict):
                    report.append(f"### {method}")
                    report.append(f"- Average advantage: {advantage_data['average_advantage']:.4f}")
                    report.append(f"- Problems with advantage: {advantage_data['problems_with_advantage']}/{advantage_data['total_problems']}")
                    report.append(f"- Advantage rate: {advantage_data['advantage_rate']:.1%}")
                    report.append(f"- Max advantage: {advantage_data['max_advantage']:.4f}")
        report.append("")
        
        # Conclusions
        report.append("## Key Findings and Recommendations")
        report.append("Based on the comprehensive benchmark analysis:")
        report.append("")
        
        if 'method_performance' in analysis:
            # Find quantum vs classical performance
            methods = list(analysis['method_performance'].keys())
            quantum_methods = [m for m in methods if 'quantum' in m.lower() or 'annealing' in m.lower()]
            
            if quantum_methods and 'quantum_advantage' in analysis:
                total_advantage_rate = np.mean([
                    data['advantage_rate'] for data in analysis['quantum_advantage'].values()
                    if isinstance(data, dict)
                ])
                
                if total_advantage_rate > 0.6:
                    report.append("✅ **Quantum methods show significant advantage**")
                    report.append("- Recommend quantum annealing for production deployment")
                    report.append("- Focus resources on quantum algorithm development")
                elif total_advantage_rate > 0.3:
                    report.append("⚠️ **Mixed quantum advantage**")
                    report.append("- Quantum methods advantageous for specific problem types")
                    report.append("- Recommend hybrid classical-quantum approach")
                else:
                    report.append("❌ **Limited quantum advantage observed**")
                    report.append("- Classical methods remain competitive")
                    report.append("- Continue quantum research for future advantage")
        
        report.append("")
        report.append("## Methodology Notes")
        report.append("- All statistical tests performed with α = 0.05 significance level")
        report.append("- Effect sizes calculated using Cohen's d")
        report.append("- Multiple runs performed for statistical reliability")
        report.append("- Results cached for reproducibility")
        
        return "\n".join(report)