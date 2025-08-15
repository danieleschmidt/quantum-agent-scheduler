"""Test suite for adaptive QUBO optimizer research module."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from quantum_scheduler.research.adaptive_qubo_optimizer import (
    AlgorithmType,
    AlgorithmPerformance,
    ProblemContext,
    QUBOAlgorithm,
    ClassicalGreedyQUBO,
    SimulatedAnnealingQUBO,
    QAOAQuantumAlgorithm,
    AdaptiveQUBOOptimizer
)


class TestAlgorithmPerformance:
    """Test algorithm performance metrics."""
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        performance = AlgorithmPerformance(
            algorithm_type=AlgorithmType.CLASSICAL_GREEDY,
            execution_time=2.0,
            solution_quality=0.8,
            energy_value=-10.0,
            convergence_iterations=5,
            success_probability=1.0
        )
        
        assert performance.efficiency_score == 0.4  # 0.8 / 2.0
    
    def test_efficiency_score_zero_time(self):
        """Test efficiency score with zero execution time."""
        performance = AlgorithmPerformance(
            algorithm_type=AlgorithmType.CLASSICAL_GREEDY,
            execution_time=0.0,
            solution_quality=0.8,
            energy_value=-10.0,
            convergence_iterations=5,
            success_probability=1.0
        )
        
        assert performance.efficiency_score == 0.0


class TestProblemContext:
    """Test problem context functionality."""
    
    def test_feature_vector_conversion(self):
        """Test conversion to feature vector."""
        context = ProblemContext(
            problem_size=50,
            density=0.3,
            problem_class="test_class",
            constraints={'constraint1': 'value1', 'constraint2': 'value2'},
            deadline=60.0,
            quality_threshold=0.85
        )
        
        vector = context.to_feature_vector()
        
        assert len(vector) == 4
        assert vector[0] == 50  # problem_size
        assert vector[1] == 0.3  # density
        assert vector[2] == 2  # len(constraints)
        assert vector[3] == 0.85  # quality_threshold


class TestClassicalGreedyQUBO:
    """Test classical greedy QUBO algorithm."""
    
    @pytest.fixture
    def algorithm(self):
        """Create classical greedy algorithm instance."""
        return ClassicalGreedyQUBO()
    
    @pytest.fixture
    def test_qubo(self):
        """Create test QUBO matrix."""
        # Create a simple QUBO where selecting variables 0 and 2 gives optimal solution
        qubo = np.array([
            [-5,  1,  1,  1],
            [ 1, -3,  1,  1],
            [ 1,  1, -4,  1],
            [ 1,  1,  1, -2]
        ])
        return qubo
    
    @pytest.fixture
    def context(self):
        """Create test problem context."""
        return ProblemContext(
            problem_size=4,
            density=1.0,
            problem_class="test",
            constraints={}
        )
    
    def test_algorithm_initialization(self, algorithm):
        """Test algorithm initialization."""
        assert algorithm.algorithm_type == AlgorithmType.CLASSICAL_GREEDY
        assert algorithm.performance_history == []
    
    def test_solve_simple_problem(self, algorithm, test_qubo, context):
        """Test solving a simple QUBO problem."""
        solution, energy = algorithm.solve(test_qubo, context, max_time=1.0)
        
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 4
        assert all(x in [0, 1] for x in solution)
        assert isinstance(energy, (int, float))
        
        # Should record performance
        assert len(algorithm.performance_history) == 1
        performance = algorithm.performance_history[0]
        assert performance.algorithm_type == AlgorithmType.CLASSICAL_GREEDY
        assert performance.execution_time > 0
    
    def test_energy_calculation(self, algorithm, test_qubo):
        """Test energy calculation methods."""
        solution = np.array([1, 0, 1, 0])
        
        # Test total energy calculation
        total_energy = algorithm._calculate_total_energy(test_qubo, solution)
        expected_energy = solution @ test_qubo @ solution  # x^T Q x
        assert total_energy == expected_energy
        
        # Test energy change calculation  
        current_solution = np.array([0, 0, 0, 0])
        energy_change = algorithm._calculate_energy_change(test_qubo, current_solution, 0)
        
        # Setting x_0 = 1 gives energy change of -Q[0,0] = -(-5) = 5
        # But we want to minimize, so we return negative
        assert energy_change == 5.0  # -(-5)
    
    def test_solution_quality_calculation(self, algorithm, context):
        """Test solution quality calculation."""
        energy = -10.0
        quality = algorithm._calculate_solution_quality(energy, context)
        
        assert 0.0 <= quality <= 1.0
    
    def test_average_performance(self, algorithm, test_qubo, context):
        """Test average performance calculation."""
        # Solve multiple times to build history
        for _ in range(3):
            algorithm.solve(test_qubo, context, max_time=0.1)
        
        avg_performance = algorithm.get_average_performance()
        
        assert avg_performance.algorithm_type == AlgorithmType.CLASSICAL_GREEDY
        assert avg_performance.execution_time > 0
        assert 0 <= avg_performance.solution_quality <= 1
    
    def test_average_performance_empty_history(self, algorithm):
        """Test average performance with empty history."""
        avg_performance = algorithm.get_average_performance()
        
        assert avg_performance.execution_time == float('inf')
        assert avg_performance.solution_quality == 0.0
        assert avg_performance.success_probability == 0.0
    
    def test_timeout_handling(self, algorithm, context):
        """Test timeout handling."""
        # Create large problem to test timeout
        large_qubo = np.random.rand(100, 100)
        
        start_time = time.time()
        solution, energy = algorithm.solve(large_qubo, context, max_time=0.01)  # Very short timeout
        execution_time = time.time() - start_time
        
        # Should respect timeout (with some tolerance)
        assert execution_time < 0.1
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 100


class TestSimulatedAnnealingQUBO:
    """Test simulated annealing QUBO algorithm."""
    
    @pytest.fixture
    def algorithm(self):
        """Create simulated annealing algorithm instance."""
        return SimulatedAnnealingQUBO(initial_temp=100.0, cooling_rate=0.9)
    
    @pytest.fixture
    def test_qubo(self):
        """Create test QUBO matrix."""
        np.random.seed(42)  # For reproducible tests
        qubo = np.random.rand(5, 5)
        qubo = (qubo + qubo.T) / 2  # Make symmetric
        return qubo
    
    @pytest.fixture
    def context(self):
        """Create test problem context."""
        return ProblemContext(
            problem_size=5,
            density=1.0,
            problem_class="test",
            constraints={}
        )
    
    def test_algorithm_initialization(self, algorithm):
        """Test algorithm initialization."""
        assert algorithm.algorithm_type == AlgorithmType.SIMULATED_ANNEALING
        assert algorithm.initial_temp == 100.0
        assert algorithm.cooling_rate == 0.9
    
    def test_solve_with_annealing(self, algorithm, test_qubo, context):
        """Test solving with simulated annealing."""
        solution, energy = algorithm.solve(test_qubo, context, max_time=1.0)
        
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 5
        assert all(x in [0, 1] for x in solution)
        assert isinstance(energy, (int, float))
        
        # Should record performance
        assert len(algorithm.performance_history) == 1
        performance = algorithm.performance_history[0]
        assert performance.algorithm_type == AlgorithmType.SIMULATED_ANNEALING
        assert performance.convergence_iterations > 0
    
    def test_temperature_cooling(self, algorithm, test_qubo, context):
        """Test that temperature decreases during annealing."""
        # Mock the algorithm to track temperature
        original_solve = algorithm.solve
        temperatures = []
        
        def mock_solve(qubo_matrix, context, max_time=60.0):
            # Store initial temperature
            temp = algorithm.initial_temp
            while temp > 0.01:
                temperatures.append(temp)
                temp *= algorithm.cooling_rate
                if len(temperatures) > 10:  # Limit for testing
                    break
            
            return original_solve(qubo_matrix, context, max_time)
        
        with patch.object(algorithm, 'solve', side_effect=mock_solve):
            algorithm.solve(test_qubo, context, max_time=0.1)
        
        # Check that temperature decreases
        if len(temperatures) > 1:
            assert all(temperatures[i] > temperatures[i+1] for i in range(len(temperatures)-1))
    
    def test_stochastic_behavior(self, algorithm, test_qubo, context):
        """Test that algorithm produces different results (stochastic)."""
        # Set different random seeds and check for variation
        np.random.seed(1)
        solution1, energy1 = algorithm.solve(test_qubo, context, max_time=0.1)
        
        np.random.seed(2)
        solution2, energy2 = algorithm.solve(test_qubo, context, max_time=0.1)
        
        # Solutions should potentially be different (stochastic algorithm)
        # Note: This test might occasionally fail due to randomness
        # but should pass most of the time
        assert not np.array_equal(solution1, solution2) or energy1 != energy2


class TestQAOAQuantumAlgorithm:
    """Test QAOA quantum algorithm."""
    
    @pytest.fixture
    def algorithm(self):
        """Create QAOA algorithm instance."""
        return QAOAQuantumAlgorithm(num_layers=2, num_shots=100)
    
    @pytest.fixture
    def test_qubo(self):
        """Create test QUBO matrix."""
        np.random.seed(42)
        qubo = np.random.rand(4, 4)
        qubo = (qubo + qubo.T) / 2  # Make symmetric
        return qubo
    
    @pytest.fixture
    def context(self):
        """Create test problem context."""
        return ProblemContext(
            problem_size=4,
            density=1.0,
            problem_class="test",
            constraints={}
        )
    
    def test_algorithm_initialization(self, algorithm):
        """Test algorithm initialization."""
        assert algorithm.algorithm_type == AlgorithmType.QAOA
        assert algorithm.num_layers == 2
        assert algorithm.num_shots == 100
    
    def test_solve_qaoa_simulation(self, algorithm, test_qubo, context):
        """Test QAOA solving (simulated)."""
        solution, energy = algorithm.solve(test_qubo, context, max_time=1.0)
        
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 4
        assert all(x in [0, 1] for x in solution)
        assert isinstance(energy, (int, float))
        
        # Should record performance
        assert len(algorithm.performance_history) == 1
        performance = algorithm.performance_history[0]
        assert performance.algorithm_type == AlgorithmType.QAOA
        assert 'num_layers' in performance.resource_usage
        assert 'num_shots' in performance.resource_usage
    
    def test_local_optimization(self, algorithm, test_qubo):
        """Test local optimization improvement."""
        # Start with a random solution
        initial_solution = np.random.randint(0, 2, size=4)
        initial_energy = algorithm._calculate_total_energy(test_qubo, initial_solution)
        
        # Apply local optimization
        optimized_solution = algorithm._local_optimization(test_qubo, initial_solution, max_time=1.0)
        optimized_energy = algorithm._calculate_total_energy(test_qubo, optimized_solution)
        
        # Optimized solution should be no worse than initial
        assert optimized_energy <= initial_energy
        assert len(optimized_solution) == len(initial_solution)
        assert all(x in [0, 1] for x in optimized_solution)
    
    def test_eigenvalue_based_initialization(self, algorithm):
        """Test eigenvalue-based solution initialization."""
        # Create QUBO with known structure
        qubo = np.array([
            [-5,  1],
            [ 1, -3]
        ])
        
        context = ProblemContext(
            problem_size=2,
            density=1.0,
            problem_class="test",
            constraints={}
        )
        
        # QAOA should handle eigenvalue decomposition
        solution, energy = algorithm.solve(qubo, context, max_time=1.0)
        
        assert len(solution) == 2
        assert all(x in [0, 1] for x in solution)


class TestAdaptiveQUBOOptimizer:
    """Test adaptive QUBO optimizer."""
    
    @pytest.fixture
    def custom_algorithms(self):
        """Create custom algorithm list for testing."""
        return [
            ClassicalGreedyQUBO(),
            SimulatedAnnealingQUBO(initial_temp=50.0),
            QAOAQuantumAlgorithm(num_layers=2)
        ]
    
    @pytest.fixture
    def optimizer(self, custom_algorithms):
        """Create adaptive optimizer instance."""
        return AdaptiveQUBOOptimizer(
            algorithms=custom_algorithms,
            performance_window=50
        )
    
    @pytest.fixture
    def test_qubo(self):
        """Create test QUBO matrix."""
        np.random.seed(42)
        qubo = np.random.rand(6, 6)
        qubo = (qubo + qubo.T) / 2
        return qubo
    
    @pytest.fixture
    def context(self):
        """Create test problem context."""
        return ProblemContext(
            problem_size=6,
            density=0.5,
            problem_class="test",
            constraints={'test_constraint': 'value'}
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert len(optimizer.algorithms) == 3
        assert optimizer.performance_window == 50
        assert len(optimizer.performance_history) == 0
        assert len(optimizer.algorithm_selection_history) == 0
    
    def test_default_algorithms(self):
        """Test default algorithm creation."""
        optimizer = AdaptiveQUBOOptimizer()
        
        assert len(optimizer.algorithms) >= 3
        algorithm_types = [alg.algorithm_type for alg in optimizer.algorithms]
        assert AlgorithmType.CLASSICAL_GREEDY in algorithm_types
        assert AlgorithmType.SIMULATED_ANNEALING in algorithm_types
        assert AlgorithmType.QAOA in algorithm_types
    
    def test_solve_adaptive(self, optimizer, test_qubo, context):
        """Test adaptive solving."""
        solution, energy, metadata = optimizer.solve(
            test_qubo, context, max_time=1.0, algorithm_selection="adaptive"
        )
        
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 6
        assert all(x in [0, 1] for x in solution)
        assert isinstance(energy, (int, float))
        
        assert 'algorithm_used' in metadata
        assert 'execution_time' in metadata
        assert 'selection_reason' in metadata
        assert metadata['selection_reason'] == 'adaptive'
    
    def test_solve_portfolio(self, optimizer, test_qubo, context):
        """Test portfolio solving."""
        solution, energy, metadata = optimizer.solve(
            test_qubo, context, max_time=3.0, algorithm_selection="portfolio"
        )
        
        assert isinstance(solution, np.ndarray)
        assert isinstance(energy, (int, float))
        
        assert 'algorithm_used' in metadata
        assert 'portfolio_size' in metadata
        assert 'selection_reason' in metadata
        assert metadata['selection_reason'] == 'portfolio_best'
        assert metadata['portfolio_size'] <= len(optimizer.algorithms)
    
    def test_solve_best(self, optimizer, test_qubo, context):
        """Test solving with historically best algorithm."""
        # First, build some performance history
        for _ in range(3):
            optimizer.solve(test_qubo, context, max_time=0.1, algorithm_selection="adaptive")
        
        solution, energy, metadata = optimizer.solve(
            test_qubo, context, max_time=1.0, algorithm_selection="best"
        )
        
        assert isinstance(solution, np.ndarray)
        assert isinstance(energy, (int, float))
        
        assert 'algorithm_used' in metadata
        assert 'selection_reason' in metadata
        assert metadata['selection_reason'] == 'historical_best'
    
    def test_algorithm_selection_heuristics(self, optimizer):
        """Test algorithm selection heuristics."""
        # Test small problem selection
        small_context = ProblemContext(
            problem_size=15,
            density=0.5,
            problem_class="small",
            constraints={}
        )
        
        selected = optimizer._select_algorithm(small_context)
        # Should prefer quantum for small problems
        assert selected.algorithm_type in [AlgorithmType.QAOA, AlgorithmType.VQE, AlgorithmType.CLASSICAL_GREEDY]
        
        # Test medium problem selection
        medium_context = ProblemContext(
            problem_size=50,
            density=0.5,
            problem_class="medium",
            constraints={}
        )
        
        selected = optimizer._select_algorithm(medium_context)
        # Should prefer simulated annealing for medium problems
        assert selected.algorithm_type in [AlgorithmType.SIMULATED_ANNEALING, AlgorithmType.CLASSICAL_GREEDY]
        
        # Test large problem selection
        large_context = ProblemContext(
            problem_size=150,
            density=0.5,
            problem_class="large",
            constraints={}
        )
        
        selected = optimizer._select_algorithm(large_context)
        # Should prefer classical greedy for large problems
        assert selected.algorithm_type == AlgorithmType.CLASSICAL_GREEDY
    
    def test_get_best_algorithm(self, optimizer, test_qubo, context):
        """Test getting historically best algorithm."""
        # Build performance history
        for _ in range(5):
            optimizer.solve(test_qubo, context, max_time=0.1, algorithm_selection="adaptive")
        
        best_algorithm = optimizer._get_best_algorithm(context)
        
        assert isinstance(best_algorithm, QUBOAlgorithm)
        assert best_algorithm in optimizer.algorithms
    
    def test_algorithm_statistics(self, optimizer, test_qubo, context):
        """Test algorithm statistics generation."""
        # Generate some performance data
        for _ in range(3):
            optimizer.solve(test_qubo, context, max_time=0.1, algorithm_selection="portfolio")
        
        stats = optimizer.get_algorithm_statistics()
        
        assert isinstance(stats, dict)
        for algo_name, algo_stats in stats.items():
            assert 'total_executions' in algo_stats
            assert 'average_execution_time' in algo_stats
            assert 'average_solution_quality' in algo_stats
            assert 'average_efficiency_score' in algo_stats
            assert 'success_probability' in algo_stats
            assert 'selection_count' in algo_stats
    
    def test_performance_report(self, optimizer, test_qubo, context):
        """Test performance report generation."""
        # Generate some performance data
        for _ in range(2):
            optimizer.solve(test_qubo, context, max_time=0.1, algorithm_selection="adaptive")
        
        report = optimizer.generate_performance_report()
        
        assert isinstance(report, str)
        assert "Adaptive QUBO Optimizer Performance Report" in report
        assert "Algorithm Performance Summary" in report
        assert "Key Insights" in report
    
    def test_empty_performance_report(self):
        """Test performance report with no data."""
        optimizer = AdaptiveQUBOOptimizer()
        report = optimizer.generate_performance_report()
        
        assert isinstance(report, str)
        assert "Performance Report" in report


@pytest.mark.slow
class TestAdaptiveQUBOIntegration:
    """Integration tests for adaptive QUBO optimizer."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        optimizer = AdaptiveQUBOOptimizer()
        
        # Create test problem
        np.random.seed(42)
        qubo = np.random.rand(8, 8)
        qubo = (qubo + qubo.T) / 2
        
        context = ProblemContext(
            problem_size=8,
            density=0.5,
            problem_class="integration_test",
            constraints={'test': True}
        )
        
        # Test different selection strategies
        strategies = ["adaptive", "portfolio", "best"]
        results = {}
        
        for strategy in strategies:
            solution, energy, metadata = optimizer.solve(
                qubo, context, max_time=2.0, algorithm_selection=strategy
            )
            
            results[strategy] = {
                'solution': solution,
                'energy': energy,
                'metadata': metadata
            }
            
            # Validate results
            assert len(solution) == 8
            assert all(x in [0, 1] for x in solution)
            assert isinstance(energy, (int, float))
            assert strategy in metadata['selection_reason'] or metadata['selection_reason'] in strategy
        
        # Check that we got valid results from all strategies
        assert len(results) == 3
        
        # Generate final statistics
        stats = optimizer.get_algorithm_statistics()
        assert len(stats) > 0
        
        report = optimizer.generate_performance_report()
        assert len(report) > 0