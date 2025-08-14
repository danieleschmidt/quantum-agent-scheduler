"""Comparative Analysis Framework for Quantum vs Classical Scheduling.

This module provides a comprehensive framework for conducting rigorous
comparative studies between quantum and classical scheduling algorithms
with statistical significance testing and visualization capabilities.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import pickle
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Types of comparison metrics."""
    EXECUTION_TIME = "execution_time"
    SOLUTION_QUALITY = "solution_quality"
    ENERGY_VALUE = "energy_value"
    SUCCESS_RATE = "success_rate"
    CONVERGENCE_SPEED = "convergence_speed"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SCALABILITY_FACTOR = "scalability_factor"


class ProblemCategory(Enum):
    """Categories of scheduling problems."""
    SMALL_DENSE = "small_dense"
    SMALL_SPARSE = "small_sparse"
    MEDIUM_DENSE = "medium_dense"
    MEDIUM_SPARSE = "medium_sparse"
    LARGE_DENSE = "large_dense"
    LARGE_SPARSE = "large_sparse"
    STRUCTURED = "structured"
    RANDOM = "random"


@dataclass
class ExperimentConfig:
    """Configuration for comparative experiments."""
    problem_sizes: List[int]
    problem_categories: List[ProblemCategory]
    num_trials_per_config: int
    max_execution_time: float
    significance_level: float = 0.05
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'problem_sizes': self.problem_sizes,
            'problem_categories': [cat.value for cat in self.problem_categories],
            'num_trials_per_config': self.num_trials_per_config,
            'max_execution_time': self.max_execution_time,
            'significance_level': self.significance_level,
            'random_seed': self.random_seed
        }


@dataclass
class TrialResult:
    """Result from a single trial."""
    trial_id: str
    algorithm_name: str
    problem_size: int
    problem_category: ProblemCategory
    execution_time: float
    solution_quality: float
    energy_value: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['problem_category'] = self.problem_category.value
        return result


@dataclass
class StatisticalComparison:
    """Statistical comparison between two algorithms."""
    algorithm_a: str
    algorithm_b: str
    metric: ComparisonMetric
    sample_size_a: int
    sample_size_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    test_statistic: float
    test_method: str
    significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['metric'] = self.metric.value
        return result


class ProblemGenerator:
    """Generate diverse scheduling problems for comparative analysis."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize problem generator.
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_problem(self, 
                        size: int, 
                        category: ProblemCategory,
                        problem_id: str = None) -> Dict[str, Any]:
        """Generate a scheduling problem.
        
        Args:
            size: Problem size (number of variables)
            category: Problem category
            problem_id: Optional problem identifier
            
        Returns:
            Dictionary containing problem data
        """
        problem_id = problem_id or f"{category.value}_{size}_{int(time.time())}"
        
        # Generate base QUBO matrix
        if category in [ProblemCategory.SMALL_SPARSE, ProblemCategory.MEDIUM_SPARSE, ProblemCategory.LARGE_SPARSE]:
            density = 0.1 + 0.2 * np.random.random()  # 10-30% density
        else:
            density = 0.4 + 0.4 * np.random.random()  # 40-80% density
        
        # Create QUBO matrix
        qubo_matrix = self._generate_qubo_matrix(size, density, category)
        
        # Generate agents and tasks
        num_agents = max(1, size // 4)
        num_tasks = size - num_agents
        
        agents = self._generate_agents(num_agents)
        tasks = self._generate_tasks(num_tasks, category)
        
        return {
            'problem_id': problem_id,
            'problem_size': size,
            'problem_category': category,
            'qubo_matrix': qubo_matrix,
            'agents': agents,
            'tasks': tasks,
            'metadata': {
                'density': density,
                'generation_time': time.time(),
                'random_seed': self.random_seed
            }
        }
    
    def _generate_qubo_matrix(self, 
                             size: int, 
                             density: float, 
                             category: ProblemCategory) -> np.ndarray:
        """Generate QUBO matrix with specific characteristics."""
        # Initialize matrix
        matrix = np.zeros((size, size))
        
        if category == ProblemCategory.STRUCTURED:
            # Generate structured problem (e.g., graph coloring-like)
            return self._generate_structured_qubo(size, density)
        else:
            # Generate random QUBO
            num_nonzero = int(size * size * density)
            
            # Add diagonal terms (variable penalties/rewards)
            diagonal_values = np.random.uniform(-10, 10, size)
            np.fill_diagonal(matrix, diagonal_values)
            
            # Add off-diagonal terms (interactions)
            for _ in range(num_nonzero - size):
                i, j = np.random.randint(0, size, 2)
                if i != j:
                    value = np.random.uniform(-5, 5)
                    matrix[i, j] = value
                    matrix[j, i] = value  # Keep symmetric
        
        return matrix
    
    def _generate_structured_qubo(self, size: int, density: float) -> np.ndarray:
        """Generate structured QUBO matrix."""
        matrix = np.zeros((size, size))
        
        # Create block structure
        block_size = max(1, size // 4)
        
        for block_start in range(0, size, block_size):
            block_end = min(block_start + block_size, size)
            block_range = range(block_start, block_end)
            
            # Strong intra-block connections
            for i in block_range:
                for j in block_range:
                    if i != j:
                        matrix[i, j] = np.random.uniform(2, 5)
                
                # Diagonal penalty
                matrix[i, i] = np.random.uniform(-3, -1)
            
            # Weaker inter-block connections
            if block_end < size:
                next_block_start = block_end
                next_block_end = min(next_block_start + block_size, size)
                
                for i in block_range:
                    for j in range(next_block_start, next_block_end):
                        if np.random.random() < density * 0.5:
                            value = np.random.uniform(-2, 2)
                            matrix[i, j] = value
                            matrix[j, i] = value
        
        return matrix
    
    def _generate_agents(self, num_agents: int) -> List[Dict[str, Any]]:
        """Generate agent data."""
        skills_pool = ['python', 'java', 'ml', 'web', 'data', 'mobile', 'devops', 'design']
        
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
        
        return agents
    
    def _generate_tasks(self, num_tasks: int, category: ProblemCategory) -> List[Dict[str, Any]]:
        """Generate task data."""
        skills_pool = ['python', 'java', 'ml', 'web', 'data', 'mobile', 'devops', 'design']
        
        tasks = []
        for i in range(num_tasks):
            if category in [ProblemCategory.STRUCTURED]:
                # Structured tasks with dependencies
                required_skills = [np.random.choice(skills_pool)]
            else:
                # Random task requirements
                num_required = np.random.randint(1, 3)
                required_skills = list(np.random.choice(skills_pool, num_required, replace=False))
            
            duration = np.random.randint(1, 5)
            priority = np.random.uniform(1, 10)
            
            tasks.append({
                'id': f'task_{i}',
                'required_skills': required_skills,
                'duration': duration,
                'priority': priority
            })
        
        return tasks


class ComparativeAnalysisFramework:
    """Framework for conducting comparative analysis studies."""
    
    def __init__(self, 
                 output_directory: Path = None,
                 parallel_execution: bool = True,
                 max_workers: int = None):
        """Initialize comparative analysis framework.
        
        Args:
            output_directory: Directory for saving results
            parallel_execution: Whether to use parallel execution
            max_workers: Maximum number of parallel workers
        """
        self.output_directory = output_directory or Path("comparative_analysis_results")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers or 4
        
        self.problem_generator = ProblemGenerator()
        self.trial_results: List[TrialResult] = []
        self.statistical_comparisons: List[StatisticalComparison] = []
        
        logger.info(f"Initialized ComparativeAnalysisFramework with output dir: {self.output_directory}")
    
    async def run_comparative_study(self,
                                  algorithms: Dict[str, Callable],
                                  config: ExperimentConfig) -> Dict[str, Any]:
        """Run comprehensive comparative study.
        
        Args:
            algorithms: Dictionary mapping algorithm names to callable functions
            config: Experiment configuration
            
        Returns:
            Dictionary containing study results and analysis
        """
        logger.info(f"Starting comparative study with {len(algorithms)} algorithms")
        logger.info(f"Problem sizes: {config.problem_sizes}")
        logger.info(f"Categories: {[cat.value for cat in config.problem_categories]}")
        
        start_time = time.time()
        
        # Generate all experiment configurations
        experiment_configs = []
        for size in config.problem_sizes:
            for category in config.problem_categories:
                for trial in range(config.num_trials_per_config):
                    experiment_configs.append((size, category, trial))
        
        logger.info(f"Total experiments: {len(experiment_configs)} × {len(algorithms)} = {len(experiment_configs) * len(algorithms)}")
        
        # Run experiments
        if self.parallel_execution:
            await self._run_experiments_parallel(algorithms, experiment_configs, config)
        else:
            await self._run_experiments_sequential(algorithms, experiment_configs, config)
        
        # Perform statistical analysis
        self._perform_statistical_analysis(list(algorithms.keys()), config)
        
        # Generate comprehensive results
        study_duration = time.time() - start_time
        results = self._compile_study_results(algorithms, config, study_duration)
        
        # Save results
        self._save_study_results(results)
        
        logger.info(f"Comparative study completed in {study_duration:.2f}s")
        
        return results
    
    async def _run_experiments_parallel(self,
                                      algorithms: Dict[str, Callable],
                                      experiment_configs: List[Tuple],
                                      config: ExperimentConfig) -> None:
        """Run experiments in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for size, category, trial in experiment_configs:
                # Generate problem
                problem = self.problem_generator.generate_problem(size, category)
                
                # Submit algorithm executions
                for algo_name, algo_func in algorithms.items():
                    future = executor.submit(
                        self._run_single_trial,
                        algo_name, algo_func, problem, trial, config
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    trial_result = future.result()
                    if trial_result:
                        self.trial_results.append(trial_result)
                except Exception as e:
                    logger.error(f"Trial failed: {e}")
    
    async def _run_experiments_sequential(self,
                                        algorithms: Dict[str, Callable],
                                        experiment_configs: List[Tuple],
                                        config: ExperimentConfig) -> None:
        """Run experiments sequentially."""
        for size, category, trial in experiment_configs:
            # Generate problem
            problem = self.problem_generator.generate_problem(size, category)
            
            # Run each algorithm
            for algo_name, algo_func in algorithms.items():
                trial_result = self._run_single_trial(
                    algo_name, algo_func, problem, trial, config
                )
                if trial_result:
                    self.trial_results.append(trial_result)
    
    def _run_single_trial(self,
                         algorithm_name: str,
                         algorithm_func: Callable,
                         problem: Dict[str, Any],
                         trial_index: int,
                         config: ExperimentConfig) -> Optional[TrialResult]:
        """Run a single trial."""
        trial_id = f"{algorithm_name}_{problem['problem_id']}_{trial_index}"
        
        try:
            start_time = time.time()
            
            # Execute algorithm
            result = algorithm_func(
                qubo_matrix=problem['qubo_matrix'],
                agents=problem['agents'],
                tasks=problem['tasks'],
                max_time=config.max_execution_time
            )
            
            execution_time = time.time() - start_time
            
            # Extract results
            if isinstance(result, tuple) and len(result) >= 2:
                solution, energy = result[:2]
                metadata = result[2] if len(result) > 2 else {}
            else:
                solution, energy, metadata = result, 0.0, {}
            
            # Calculate solution quality
            solution_quality = self._calculate_solution_quality(solution, energy, problem)
            
            return TrialResult(
                trial_id=trial_id,
                algorithm_name=algorithm_name,
                problem_size=problem['problem_size'],
                problem_category=problem['problem_category'],
                execution_time=execution_time,
                solution_quality=solution_quality,
                energy_value=energy,
                success=True,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Trial {trial_id} failed: {e}")
            return TrialResult(
                trial_id=trial_id,
                algorithm_name=algorithm_name,
                problem_size=problem['problem_size'],
                problem_category=problem['problem_category'],
                execution_time=config.max_execution_time,
                solution_quality=0.0,
                energy_value=float('inf'),
                success=False,
                metadata={'error': str(e)}
            )
    
    def _calculate_solution_quality(self,
                                   solution: Any,
                                   energy: float,
                                   problem: Dict[str, Any]) -> float:
        """Calculate solution quality score."""
        try:
            if isinstance(solution, np.ndarray):
                # Basic quality metric based on energy
                max_energy = np.abs(np.trace(problem['qubo_matrix'])) + 1
                normalized_energy = abs(energy) / max_energy
                return max(0.0, 1.0 - normalized_energy)
            else:
                # For non-array solutions, use heuristic
                return 0.5
        except:
            return 0.0
    
    def _perform_statistical_analysis(self,
                                    algorithm_names: List[str],
                                    config: ExperimentConfig) -> None:
        """Perform statistical analysis of results."""
        logger.info("Performing statistical analysis...")
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([result.to_dict() for result in self.trial_results])
        
        # Compare each pair of algorithms
        for i, algo_a in enumerate(algorithm_names):
            for j, algo_b in enumerate(algorithm_names[i+1:], i+1):
                for metric in ComparisonMetric:
                    comparison = self._compare_algorithms(df, algo_a, algo_b, metric, config)
                    if comparison:
                        self.statistical_comparisons.append(comparison)
    
    def _compare_algorithms(self,
                           df: pd.DataFrame,
                           algo_a: str,
                           algo_b: str,
                           metric: ComparisonMetric,
                           config: ExperimentConfig) -> Optional[StatisticalComparison]:
        """Compare two algorithms on a specific metric."""
        try:
            # Filter data
            data_a = df[df['algorithm_name'] == algo_a][metric.value].values
            data_b = df[df['algorithm_name'] == algo_b][metric.value].values
            
            if len(data_a) == 0 or len(data_b) == 0:
                return None
            
            # Remove infinite values
            data_a = data_a[np.isfinite(data_a)]
            data_b = data_b[np.isfinite(data_b)]
            
            if len(data_a) == 0 or len(data_b) == 0:
                return None
            
            # Calculate basic statistics
            mean_a, std_a = np.mean(data_a), np.std(data_a)
            mean_b, std_b = np.mean(data_b), np.std(data_b)
            
            # Perform statistical test
            if len(data_a) >= 30 and len(data_b) >= 30:
                # Use t-test for large samples
                test_stat, p_value = ttest_ind(data_a, data_b)
                test_method = "independent_t_test"
            else:
                # Use Mann-Whitney U test for small samples
                test_stat, p_value = mannwhitneyu(data_a, data_b, alternative='two-sided')
                test_method = "mann_whitney_u"
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data_a) - 1) * std_a**2 + (len(data_b) - 1) * std_b**2) / 
                                (len(data_a) + len(data_b) - 2))
            effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate confidence interval for difference in means
            se_diff = pooled_std * np.sqrt(1/len(data_a) + 1/len(data_b))
            t_critical = stats.t.ppf(1 - config.significance_level/2, len(data_a) + len(data_b) - 2)
            diff = mean_a - mean_b
            margin_error = t_critical * se_diff
            ci = (diff - margin_error, diff + margin_error)
            
            return StatisticalComparison(
                algorithm_a=algo_a,
                algorithm_b=algo_b,
                metric=metric,
                sample_size_a=len(data_a),
                sample_size_b=len(data_b),
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=std_a,
                std_b=std_b,
                effect_size=abs(effect_size),
                p_value=p_value,
                confidence_interval=ci,
                test_statistic=test_stat,
                test_method=test_method,
                significant=p_value < config.significance_level
            )
            
        except Exception as e:
            logger.warning(f"Statistical comparison failed for {algo_a} vs {algo_b} on {metric.value}: {e}")
            return None
    
    def _compile_study_results(self,
                              algorithms: Dict[str, Callable],
                              config: ExperimentConfig,
                              study_duration: float) -> Dict[str, Any]:
        """Compile comprehensive study results."""
        # Convert results to DataFrame
        df = pd.DataFrame([result.to_dict() for result in self.trial_results])
        
        # Calculate summary statistics
        summary_stats = {}
        for algo_name in algorithms.keys():
            algo_data = df[df['algorithm_name'] == algo_name]
            
            if len(algo_data) > 0:
                summary_stats[algo_name] = {
                    'total_trials': len(algo_data),
                    'success_rate': algo_data['success'].mean(),
                    'avg_execution_time': algo_data['execution_time'].mean(),
                    'std_execution_time': algo_data['execution_time'].std(),
                    'avg_solution_quality': algo_data['solution_quality'].mean(),
                    'std_solution_quality': algo_data['solution_quality'].std(),
                    'avg_energy': algo_data['energy_value'][np.isfinite(algo_data['energy_value'])].mean(),
                }
        
        # Compile results
        results = {
            'experiment_config': config.to_dict(),
            'study_metadata': {
                'start_time': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': study_duration,
                'total_trials': len(self.trial_results),
                'algorithms_tested': list(algorithms.keys()),
                'framework_version': '1.0.0'
            },
            'summary_statistics': summary_stats,
            'statistical_comparisons': [comp.to_dict() for comp in self.statistical_comparisons],
            'raw_trial_results': [result.to_dict() for result in self.trial_results],
            'insights': self._generate_insights()
        }
        
        return results
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate insights from the analysis."""
        if not self.statistical_comparisons:
            return {'message': 'No statistical comparisons available'}
        
        insights = {
            'significant_differences': [],
            'best_performers': {},
            'scalability_analysis': {},
            'recommendations': []
        }
        
        # Find significant differences
        for comp in self.statistical_comparisons:
            if comp.significant and comp.metric == ComparisonMetric.EXECUTION_TIME:
                faster_algo = comp.algorithm_a if comp.mean_a < comp.mean_b else comp.algorithm_b
                slower_algo = comp.algorithm_b if comp.mean_a < comp.mean_b else comp.algorithm_a
                speedup = max(comp.mean_a, comp.mean_b) / min(comp.mean_a, comp.mean_b)
                
                insights['significant_differences'].append({
                    'faster_algorithm': faster_algo,
                    'slower_algorithm': slower_algo,
                    'speedup_factor': speedup,
                    'p_value': comp.p_value,
                    'effect_size': comp.effect_size
                })
        
        # Generate recommendations
        if insights['significant_differences']:
            fastest_algo = min(insights['significant_differences'], 
                             key=lambda x: x['speedup_factor'])['faster_algorithm']
            insights['recommendations'].append(f"Consider {fastest_algo} for time-critical applications")
        
        return insights
    
    def _save_study_results(self, results: Dict[str, Any]) -> None:
        """Save study results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_directory / f"comparative_study_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save pickle for Python objects
        pickle_path = self.output_directory / f"comparative_study_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'trial_results': self.trial_results,
                'statistical_comparisons': self.statistical_comparisons
            }, f)
        
        logger.info(f"Results saved to {json_path} and {pickle_path}")
    
    def generate_visualization_report(self) -> str:
        """Generate visualization report with plots.
        
        Returns:
            Path to generated HTML report
        """
        if not self.trial_results:
            return "No trial results available for visualization"
        
        # Convert to DataFrame
        df = pd.DataFrame([result.to_dict() for result in self.trial_results])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time comparison
        sns.boxplot(data=df, x='algorithm_name', y='execution_time', ax=axes[0,0])
        axes[0,0].set_title('Execution Time Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Solution quality comparison
        sns.boxplot(data=df, x='algorithm_name', y='solution_quality', ax=axes[0,1])
        axes[0,1].set_title('Solution Quality Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Scalability analysis
        df_success = df[df['success'] == True]
        if len(df_success) > 0:
            sns.scatterplot(data=df_success, x='problem_size', y='execution_time', 
                           hue='algorithm_name', ax=axes[1,0])
            axes[1,0].set_title('Scalability Analysis')
            axes[1,0].set_yscale('log')
        
        # Success rate by problem size
        success_rates = df.groupby(['algorithm_name', 'problem_size'])['success'].mean().reset_index()
        sns.lineplot(data=success_rates, x='problem_size', y='success', 
                    hue='algorithm_name', ax=axes[1,1])
        axes[1,1].set_title('Success Rate by Problem Size')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_directory / f"comparative_analysis_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)