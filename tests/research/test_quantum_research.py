"""Comprehensive tests for quantum scheduling research components.

This test suite validates the research-grade quantum scheduling implementations
including circuit optimization, annealing algorithms, and benchmarking frameworks.
"""

import pytest
import numpy as np
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from quantum_scheduler.optimization import AdaptiveCircuitOptimizer, QuantumAdvantageAnalyzer
from quantum_scheduler.research import (
    AdaptiveQuantumAnnealer,
    ComparativeAnnealingAnalyzer,
    AnnealingStrategy,
    AutomatedBenchmarkRunner,
    ProblemGenerator,
    ProblemClass,
    BenchmarkProblem,
    ExperimentResult
)
from quantum_scheduler.reliability import (
    QuantumErrorCorrector,
    NoiseMitigation,
    FaultTolerantScheduler,
    NoiseModel,
    ErrorCorrectionCode,
    NoiseParameters
)
from quantum_scheduler.cloud import (
    CloudOrchestrator,
    MultiRegionScheduler,
    LoadBalancer,
    CloudResource,
    WorkloadRequest,
    CloudProvider,
    Region,
    WorkloadType
)


class TestQuantumCircuitOptimization:
    """Test suite for quantum circuit optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = AdaptiveCircuitOptimizer()
        self.advantage_analyzer = QuantumAdvantageAnalyzer()
        
        # Create test QUBO matrix
        self.test_qubo = np.array([
            [1.0, -0.5, 0.0],
            [-0.5, 2.0, -1.0], 
            [0.0, -1.0, 1.5]
        ])
    
    def test_circuit_optimizer_initialization(self):
        """Test circuit optimizer initialization."""
        assert self.optimizer.noise_model is not None
        assert self.optimizer.hardware_constraints is not None
        assert 'max_qubits' in self.optimizer.hardware_constraints
    
    def test_qubo_circuit_optimization(self):
        """Test QUBO circuit optimization."""
        result = self.optimizer.optimize_qubo_circuit(
            qubo_matrix=self.test_qubo,
            num_layers=3,
            optimization_level=2
        )
        
        # Verify optimization result structure
        assert hasattr(result, 'original_metrics')
        assert hasattr(result, 'optimized_metrics')
        assert hasattr(result, 'improvement_factor')
        assert hasattr(result, 'optimization_time')
        
        # Verify improvement (should reduce depth)
        assert result.improvement_factor >= 1.0
        assert result.optimized_metrics.depth <= result.original_metrics.depth
        
        # Verify timing
        assert result.optimization_time > 0
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        results = []
        
        for level in [1, 2, 3]:
            result = self.optimizer.optimize_qubo_circuit(
                qubo_matrix=self.test_qubo,
                num_layers=4,
                optimization_level=level
            )
            results.append(result)
            
            # Higher levels should generally give better results
            assert result.improvement_factor >= 1.0
        
        # Check that techniques are different
        techniques = [r.technique_used for r in results]
        assert len(set(techniques)) > 1  # Should use different techniques
    
    def test_optimization_statistics(self):
        """Test optimization statistics collection."""
        # Run multiple optimizations
        for _ in range(3):
            self.optimizer.optimize_qubo_circuit(
                qubo_matrix=self.test_qubo,
                num_layers=2,
                optimization_level=1
            )
        
        stats = self.optimizer.get_optimization_statistics()
        
        assert stats['total_optimizations'] == 3
        assert 'average_improvement' in stats
        assert 'best_improvement' in stats
        assert 'total_depth_reduction' in stats
    
    def test_quantum_advantage_analysis(self):
        """Test quantum advantage analysis."""
        # Create circuit metrics
        circuit_metrics = Mock()
        circuit_metrics.estimated_execution_time = 0.1
        circuit_metrics.fidelity_estimate = 0.95
        circuit_metrics.parallelization_factor = 2.0
        
        classical_time = 1.0
        
        analysis = self.advantage_analyzer.analyze_quantum_advantage(
            qubo_matrix=self.test_qubo,
            classical_time=classical_time,
            quantum_circuit_metrics=circuit_metrics
        )
        
        # Verify analysis structure
        assert 'problem_size' in analysis
        assert 'speedup_ratio' in analysis
        assert 'quantum_advantage' in analysis
        assert 'recommendation' in analysis
        
        # Should show advantage with 10x speedup (1.0/0.1)
        assert analysis['speedup_ratio'] == 10.0
        assert analysis['quantum_advantage'] is True


class TestQuantumAnnealing:
    """Test suite for adaptive quantum annealing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.annealer = AdaptiveQuantumAnnealer()
        self.analyzer = ComparativeAnnealingAnalyzer()
        
        # Create test QUBO problems
        self.small_qubo = np.random.random((10, 10))
        self.small_qubo = (self.small_qubo + self.small_qubo.T) / 2
        
        self.medium_qubo = np.random.random((50, 50))
        self.medium_qubo = (self.medium_qubo + self.medium_qubo.T) / 2
    
    def test_annealer_initialization(self):
        """Test annealer initialization."""
        assert self.annealer.hardware_constraints is not None
        assert self.annealer.learning_rate > 0
        assert self.annealer.parameter_predictor is not None
    
    def test_problem_topology_analysis(self):
        """Test problem topology analysis."""
        analysis = self.annealer._analyze_problem_topology(self.small_qubo)
        
        # Verify analysis components
        required_keys = [
            'problem_size', 'density', 'condition_number',
            'clustering_coefficient', 'topology_complexity'
        ]
        
        for key in required_keys:
            assert key in analysis
            assert isinstance(analysis[key], (int, float))
        
        assert analysis['problem_size'] == 10
        assert 0 <= analysis['density'] <= 1
    
    def test_schedule_prediction(self):
        """Test annealing schedule prediction."""
        analysis = self.annealer._analyze_problem_topology(self.small_qubo)
        schedule = self.annealer._predict_optimal_schedule(analysis)
        
        # Verify schedule structure
        assert hasattr(schedule, 'strategy')
        assert hasattr(schedule, 'total_time')
        assert hasattr(schedule, 'pause_duration')
        
        # Verify reasonable values
        assert schedule.total_time > 0
        assert schedule.pause_duration >= 0
        assert isinstance(schedule.strategy, AnnealingStrategy)
    
    def test_optimization_execution(self):
        """Test quantum annealing optimization."""
        result = self.annealer.optimize_scheduling_problem(
            qubo_matrix=self.small_qubo,
            max_iterations=2,
            target_quality=0.8
        )
        
        # Verify result structure
        assert hasattr(result, 'energy')
        assert hasattr(result, 'solution_vector')
        assert hasattr(result, 'success_probability')
        assert hasattr(result, 'annealing_time')
        assert hasattr(result, 'solution_quality')
        
        # Verify solution validity
        assert len(result.solution_vector) == self.small_qubo.shape[0]
        assert all(bit in [0, 1] for bit in result.solution_vector)
        assert 0 <= result.success_probability <= 1
    
    def test_comparative_analysis(self):
        """Test comparative annealing analysis."""
        # Create multiple test problems
        problems = [self.small_qubo, self.medium_qubo]
        
        # Run comparative study
        analysis = self.analyzer.run_comparative_study(
            qubo_problems=problems,
            methods=['adaptive_annealing', 'classical_sa']
        )
        
        # Verify analysis structure
        assert 'total_analyses' in analysis
        assert 'method_rankings' in analysis
        
        # Verify methods were compared
        assert len(analysis['method_rankings']) >= 2
    
    def test_research_report_generation(self):
        """Test research report generation."""
        # Run a simple study first
        problems = [self.small_qubo]
        self.analyzer.run_comparative_study(problems, methods=['adaptive_annealing'])
        
        # Generate report
        report = self.analyzer.generate_research_report()
        
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert "Quantum Annealing Optimization Research Report" in report


class TestAutomatedBenchmarking:
    """Test suite for automated benchmarking framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = ProblemGenerator(random_seed=42)
        
        # Create temporary directory for benchmark results
        self.temp_dir = tempfile.mkdtemp()
        self.runner = AutomatedBenchmarkRunner(
            output_dir=self.temp_dir,
            num_workers=2,
            cache_results=False  # Disable caching for tests
        )
    
    def test_problem_generator(self):
        """Test benchmark problem generation."""
        problems = self.generator.generate_problem_suite(
            problem_classes=[ProblemClass.SMALL_SPARSE, ProblemClass.MEDIUM_DENSE],
            problems_per_class=2
        )
        
        # Verify problem generation
        assert len(problems) == 4  # 2 classes × 2 problems
        
        for problem in problems:
            assert isinstance(problem, BenchmarkProblem)
            assert problem.qubo_matrix is not None
            assert problem.metadata['problem_size'] > 0
            assert 0 <= problem.metadata['density'] <= 1
    
    def test_method_registration(self):
        """Test method registration for benchmarking."""
        def test_method(qubo_matrix):
            return {
                'solution_vector': np.zeros(qubo_matrix.shape[0]),
                'energy': 0.0,
                'success_probability': 1.0
            }
        
        self.runner.register_method("test_method", test_method, "Test method")
        
        assert "test_method" in self.runner.registered_methods
        assert self.runner.registered_methods["test_method"]['function'] == test_method
    
    def test_benchmark_execution(self):
        """Test benchmark suite execution."""
        # Generate small test problems
        problems = self.generator.generate_problem_suite(
            problem_classes=[ProblemClass.SMALL_SPARSE],
            problems_per_class=1
        )
        
        # Register simple test method
        def simple_method(qubo_matrix):
            return {
                'solution_vector': np.zeros(qubo_matrix.shape[0]),
                'energy': np.random.random(),
                'success_probability': 0.9
            }
        
        self.runner.register_method("simple", simple_method)
        
        # Run benchmark
        results = self.runner.run_benchmark_suite(
            problems=problems,
            methods=["simple"],
            num_runs=2,
            timeout=10.0
        )
        
        # Verify results
        assert 'raw_results' in results
        assert 'analysis' in results
        assert 'metadata' in results
        
        assert len(results['raw_results']) == 2  # 1 problem × 1 method × 2 runs
        assert results['metadata']['successful_experiments'] >= 0
    
    def test_statistical_analysis(self):
        """Test statistical analysis of benchmark results."""
        # Create mock results
        results = [
            ExperimentResult(
                problem_id="test_001",
                method_name="method_a",
                solution_vector=np.array([0, 1]),
                energy=-1.5,
                execution_time=0.1,
                success_probability=0.9
            ),
            ExperimentResult(
                problem_id="test_001",
                method_name="method_b", 
                solution_vector=np.array([1, 0]),
                energy=-1.2,
                execution_time=0.2,
                success_probability=0.85
            )
        ]
        
        # Mock problems for analysis
        problems = [BenchmarkProblem(
            problem_id="test_001",
            problem_class=ProblemClass.SMALL_SPARSE,
            qubo_matrix=np.eye(2)
        )]
        
        analysis = self.runner._analyze_results(results, problems, ["method_a", "method_b"])
        
        # Verify analysis structure
        assert 'method_performance' in analysis
        assert 'statistical_tests' in analysis
        
        # Verify method performance analysis
        assert 'method_a' in analysis['method_performance']
        assert 'method_b' in analysis['method_performance']


class TestErrorCorrection:
    """Test suite for quantum error correction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.corrector = QuantumErrorCorrector(
            correction_code=ErrorCorrectionCode.STEANE_CODE,
            code_distance=3
        )
        self.mitigator = NoiseMitigation()
        self.noise_params = NoiseParameters(NoiseModel.DEPOLARIZING)
    
    def test_error_corrector_initialization(self):
        """Test error corrector initialization."""
        assert self.corrector.correction_code == ErrorCorrectionCode.STEANE_CODE
        assert self.corrector.code_distance == 3
        assert self.corrector.code_parameters is not None
        
        # Check Steane code specific parameters
        assert self.corrector.code_parameters['physical_qubits'] == 7
        assert self.corrector.code_parameters['logical_qubits'] == 1
    
    def test_syndrome_measurement(self):
        """Test error syndrome measurement."""
        # Create test quantum state
        test_state = np.array([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0])
        
        syndrome = self.corrector._measure_syndrome(test_state, self.noise_params)
        
        # Verify syndrome structure
        assert hasattr(syndrome, 'error_detected')
        assert hasattr(syndrome, 'error_type')
        assert hasattr(syndrome, 'syndrome_pattern')
        assert hasattr(syndrome, 'confidence')
        
        assert isinstance(syndrome.error_detected, bool)
        assert 0 <= syndrome.confidence <= 1
    
    def test_error_correction_application(self):
        """Test error correction application."""
        test_state = np.random.random(7)
        
        corrected_state, syndromes = self.corrector.apply_error_correction(
            test_state, self.noise_params, circuit_depth=10
        )
        
        # Verify correction results
        assert corrected_state is not None
        assert len(corrected_state) == len(test_state)
        assert isinstance(syndromes, list)
    
    def test_noise_mitigation(self):
        """Test noise mitigation techniques."""
        # Test zero-noise extrapolation
        results = [
            {'energy': -1.0, 'success_probability': 0.9},
            {'energy': -0.8, 'success_probability': 0.8},
            {'energy': -0.6, 'success_probability': 0.7}
        ]
        noise_levels = [0.0, 0.01, 0.02]
        
        mitigated = self.mitigator.apply_zero_noise_extrapolation(results, noise_levels)
        
        assert 'energy' in mitigated
        assert 'success_probability' in mitigated
        assert 'mitigation_method' in mitigated
        assert mitigated['mitigation_method'] == 'zero_noise_extrapolation'
    
    def test_fault_tolerant_scheduler(self):
        """Test fault-tolerant scheduler integration."""
        # Mock base scheduler
        base_scheduler = Mock()
        base_scheduler.schedule.return_value = Mock(
            assignments={'task1': 'agent1'},
            cost=5.0,
            success_probability=0.9
        )
        
        # Create fault-tolerant scheduler
        ft_scheduler = FaultTolerantScheduler(
            base_scheduler=base_scheduler,
            error_corrector=self.corrector,
            noise_mitigator=self.mitigator,
            fault_tolerance_level="medium"
        )
        
        # Test scheduling with fault tolerance
        agents = [Mock(id='agent1', skills=['skill1'], capacity=2)]
        tasks = [Mock(id='task1', required_skills=['skill1'], duration=1, priority=5)]
        
        result = ft_scheduler.schedule_with_fault_tolerance(
            agents=agents,
            tasks=tasks,
            noise_params=self.noise_params
        )
        
        # Verify fault tolerance was applied
        assert hasattr(result, 'fault_tolerance_metadata')
        assert result.fault_tolerance_metadata['total_attempts'] >= 1


class TestCloudOrchestration:
    """Test suite for multi-region cloud orchestration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = CloudOrchestrator(max_concurrent_jobs=5)
        self.load_balancer = LoadBalancer()
        
        # Create test resources
        self.test_resources = [
            CloudResource(
                provider=CloudProvider.IBM_QUANTUM,
                region=Region.US_EAST_1,
                device_name="test_device_1",
                max_qubits=127,
                queue_length=0,
                cost_per_shot=0.001
            ),
            CloudResource(
                provider=CloudProvider.AWS_BRAKET,
                region=Region.US_WEST_2,
                device_name="test_device_2", 
                max_qubits=34,
                queue_length=5,
                cost_per_shot=0.0005
            )
        ]
    
    def test_orchestrator_initialization(self):
        """Test cloud orchestrator initialization."""
        assert self.orchestrator.max_concurrent_jobs == 5
        assert len(self.orchestrator.available_resources) > 0
        assert len(self.orchestrator.job_queue) == 0
        assert len(self.orchestrator.active_jobs) == 0
    
    def test_load_balancer(self):
        """Test load balancer resource selection."""
        request = WorkloadRequest(
            request_id="test_req",
            workload_type=WorkloadType.MEDIUM_QUBO,
            priority=5,
            max_cost=5.0
        )
        
        selected = self.load_balancer.select_optimal_resource(
            request, self.test_resources
        )
        
        assert selected is not None
        assert selected in self.test_resources
        assert selected.max_qubits >= request.estimated_qubits
    
    @pytest.mark.asyncio
    async def test_workload_submission(self):
        """Test workload submission to orchestrator."""
        request = WorkloadRequest(
            request_id="",
            workload_type=WorkloadType.SMALL_QUBO,
            priority=7
        )
        
        job_id = await self.orchestrator.submit_workload(request)
        
        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0
    
    @pytest.mark.asyncio
    async def test_job_status_tracking(self):
        """Test job status tracking."""
        request = WorkloadRequest(
            request_id="",
            workload_type=WorkloadType.SMALL_QUBO
        )
        
        job_id = await self.orchestrator.submit_workload(request)
        
        # Give some time for processing
        await asyncio.sleep(0.1)
        
        status = await self.orchestrator.get_job_status(job_id)
        
        assert status is not None
        assert 'job_id' in status
        assert 'status' in status
        assert status['job_id'] == job_id
    
    def test_orchestration_metrics(self):
        """Test orchestration metrics collection."""
        metrics = self.orchestrator.get_orchestration_metrics()
        
        required_keys = [
            'total_requests', 'successful_requests', 'success_rate',
            'active_jobs', 'queued_jobs', 'available_resources'
        ]
        
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
    
    def test_resource_status(self):
        """Test resource status reporting."""
        status = self.orchestrator.get_resource_status()
        
        assert isinstance(status, list)
        assert len(status) > 0
        
        for resource_status in status:
            required_keys = [
                'provider', 'region', 'device_name', 'max_qubits',
                'availability', 'utilization_score'
            ]
            
            for key in required_keys:
                assert key in resource_status
    
    @pytest.mark.asyncio 
    async def test_multi_region_scheduler(self):
        """Test multi-region scheduler interface."""
        scheduler = MultiRegionScheduler(
            enable_multi_region=True,
            preferred_regions=[Region.US_EAST_1, Region.EU_WEST_1]
        )
        
        # Mock agents and tasks
        agents = [Mock(id=f'agent_{i}') for i in range(10)]
        tasks = [Mock(id=f'task_{i}') for i in range(15)]
        
        job_id = await scheduler.schedule_agents_at_scale(
            agents=agents,
            tasks=tasks,
            priority=8
        )
        
        assert job_id is not None
        
        # Test metrics
        metrics = await scheduler.get_global_metrics()
        assert 'orchestration' in metrics
        assert 'total_quantum_resources' in metrics
        
        await scheduler.shutdown()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_quantum_scheduling(self):
        """Test end-to-end quantum scheduling pipeline."""
        # Initialize components
        circuit_optimizer = AdaptiveCircuitOptimizer()
        annealer = AdaptiveQuantumAnnealer()
        error_corrector = QuantumErrorCorrector()
        scheduler = MultiRegionScheduler()
        
        # Create test problem
        test_qubo = np.random.random((20, 20))
        test_qubo = (test_qubo + test_qubo.T) / 2
        
        # Step 1: Optimize quantum circuit
        optimization_result = circuit_optimizer.optimize_qubo_circuit(
            qubo_matrix=test_qubo,
            num_layers=3,
            optimization_level=2
        )
        
        assert optimization_result.improvement_factor >= 1.0
        
        # Step 2: Run adaptive annealing
        annealing_result = annealer.optimize_scheduling_problem(
            qubo_matrix=test_qubo,
            max_iterations=2,
            target_quality=0.8
        )
        
        assert annealing_result.solution_quality > 0
        
        # Step 3: Test error correction
        test_state = np.random.random(7)
        noise_params = NoiseParameters(NoiseModel.DEPOLARIZING)
        
        corrected_state, syndromes = error_corrector.apply_error_correction(
            test_state, noise_params, circuit_depth=5
        )
        
        assert corrected_state is not None
        
        # Step 4: Schedule at scale
        agents = [Mock(id=f'agent_{i}') for i in range(20)]
        tasks = [Mock(id=f'task_{i}') for i in range(25)]
        
        job_id = await scheduler.schedule_agents_at_scale(
            agents=agents,
            tasks=tasks
        )
        
        assert job_id is not None
        
        await scheduler.shutdown()
    
    def test_performance_benchmarking(self):
        """Test performance of key algorithms."""
        # Test circuit optimization performance
        circuit_optimizer = AdaptiveCircuitOptimizer()
        
        start_time = time.time()
        
        # Test multiple optimization levels
        for level in [1, 2, 3]:
            test_qubo = np.random.random((15, 15))
            test_qubo = (test_qubo + test_qubo.T) / 2
            
            result = circuit_optimizer.optimize_qubo_circuit(
                qubo_matrix=test_qubo,
                optimization_level=level
            )
            
            assert result.optimization_time < 5.0  # Should complete within 5 seconds
        
        total_time = time.time() - start_time
        assert total_time < 15.0  # All optimizations should complete within 15 seconds
    
    def test_scalability_limits(self):
        """Test scalability limits of algorithms."""
        # Test problem generator scalability
        generator = ProblemGenerator()
        
        # Generate large problems
        problems = generator.generate_problem_suite(
            problem_classes=[ProblemClass.LARGE_SPARSE],
            problems_per_class=1
        )
        
        large_problem = problems[0]
        assert large_problem.metadata['problem_size'] >= 60
        assert large_problem.metadata['problem_size'] < 150
        
        # Test that algorithms can handle larger problems
        annealer = AdaptiveQuantumAnnealer()
        
        start_time = time.time()
        result = annealer.optimize_scheduling_problem(
            qubo_matrix=large_problem.qubo_matrix,
            max_iterations=1,  # Limit iterations for performance
            target_quality=0.7
        )
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time even for large problems
        assert execution_time < 30.0
        assert result.solution_quality > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])