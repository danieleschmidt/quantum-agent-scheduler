"""Comprehensive tests for quantum error mitigation research module."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from quantum_scheduler.research.quantum_error_mitigation import (
    NoiseModel,
    ErrorMitigationResult,
    ZeroNoiseExtrapolation,
    SymmetryVerification,
    AdaptiveErrorMitigation,
    NISQSchedulingOptimizer,
    create_test_noise_model,
    ErrorMitigationType
)


class TestNoiseModel:
    """Test cases for NoiseModel class."""
    
    def test_noise_model_initialization(self):
        """Test NoiseModel initialization with default parameters."""
        noise_model = NoiseModel()
        
        assert 'single_qubit' in noise_model.gate_error_rates
        assert 'two_qubit' in noise_model.gate_error_rates
        assert noise_model.measurement_error_rate == 0.01
        assert 'T1' in noise_model.decoherence_times
        assert 'T2' in noise_model.decoherence_times
    
    def test_noise_model_custom_parameters(self):
        """Test NoiseModel with custom parameters."""
        custom_gate_errors = {'single_qubit': 0.005, 'two_qubit': 0.02}
        custom_decoherence = {'T1': 30e-6, 'T2': 40e-6}
        
        noise_model = NoiseModel(
            gate_error_rates=custom_gate_errors,
            measurement_error_rate=0.03,
            decoherence_times=custom_decoherence,
            thermal_excitation_rate=0.002
        )
        
        assert noise_model.gate_error_rates == custom_gate_errors
        assert noise_model.measurement_error_rate == 0.03
        assert noise_model.decoherence_times == custom_decoherence
        assert noise_model.thermal_excitation_rate == 0.002


class TestErrorMitigationResult:
    """Test cases for ErrorMitigationResult class."""
    
    def test_error_mitigation_result_creation(self):
        """Test ErrorMitigationResult creation."""
        result = ErrorMitigationResult(
            original_energy=-10.0,
            mitigated_energy=-12.0,
            error_reduction=0.2,
            mitigation_overhead=0.5,
            confidence_interval=(-13.0, -11.0),
            noise_amplification_factor=3.0,
            statistical_uncertainty=0.1
        )
        
        assert result.original_energy == -10.0
        assert result.mitigated_energy == -12.0
        assert result.error_reduction == 0.2
    
    def test_improvement_factor_calculation(self):
        """Test improvement factor calculation."""
        result = ErrorMitigationResult(
            original_energy=-10.0,
            mitigated_energy=-15.0,
            error_reduction=0.5,
            mitigation_overhead=0.5,
            confidence_interval=(-16.0, -14.0),
            noise_amplification_factor=2.0,
            statistical_uncertainty=0.1
        )
        
        expected_improvement = 15.0 / 10.0
        assert abs(result.improvement_factor - expected_improvement) < 1e-6
    
    def test_improvement_factor_zero_energy(self):
        """Test improvement factor with zero original energy."""
        result = ErrorMitigationResult(
            original_energy=0.0,
            mitigated_energy=-5.0,
            error_reduction=0.0,
            mitigation_overhead=0.5,
            confidence_interval=(-6.0, -4.0),
            noise_amplification_factor=2.0,
            statistical_uncertainty=0.1
        )
        
        assert result.improvement_factor == 1.0


class TestZeroNoiseExtrapolation:
    """Test cases for ZeroNoiseExtrapolation class."""
    
    def test_zne_initialization(self):
        """Test ZNE initialization."""
        zne = ZeroNoiseExtrapolation(
            noise_factors=[1.0, 2.0, 3.0],
            extrapolation_method="linear",
            max_circuits=5
        )
        
        assert zne.noise_factors == [1.0, 2.0, 3.0]
        assert zne.extrapolation_method == "linear"
        assert zne.max_circuits == 5
    
    def test_zne_default_initialization(self):
        """Test ZNE with default parameters."""
        zne = ZeroNoiseExtrapolation()
        
        assert len(zne.noise_factors) == 5
        assert zne.extrapolation_method == "exponential"
        assert zne.max_circuits == 10
    
    @patch('quantum_scheduler.research.quantum_error_mitigation.HAS_QISKIT', False)
    def test_zne_mitigate_errors_mock(self):
        """Test ZNE error mitigation without Qiskit."""
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0])
        noise_model = NoiseModel()
        
        # Mock circuit
        mock_circuit = Mock()
        mock_circuit.num_qubits = 5
        mock_circuit.data = [Mock() for _ in range(10)]
        
        result = zne.mitigate_errors(mock_circuit, shots=1024, noise_model=noise_model)
        
        assert isinstance(result, ErrorMitigationResult)
        assert result.original_energy != 0
        assert result.mitigated_energy != 0
        assert result.mitigation_overhead > 0
    
    def test_zne_extrapolation_methods(self):
        """Test different extrapolation methods."""
        noise_results = [(1.0, -10.0), (2.0, -9.5), (3.0, -9.0)]
        
        # Test linear extrapolation
        zne_linear = ZeroNoiseExtrapolation(extrapolation_method="linear")
        energy_linear, conf_linear = zne_linear._extrapolate_to_zero_noise(noise_results)
        
        # Test exponential extrapolation
        zne_exp = ZeroNoiseExtrapolation(extrapolation_method="exponential")
        energy_exp, conf_exp = zne_exp._extrapolate_to_zero_noise(noise_results)
        
        # Test polynomial extrapolation
        zne_poly = ZeroNoiseExtrapolation(extrapolation_method="polynomial")
        energy_poly, conf_poly = zne_poly._extrapolate_to_zero_noise(noise_results)
        
        assert isinstance(energy_linear, float)
        assert isinstance(energy_exp, float)
        assert isinstance(energy_poly, float)
        assert len(conf_linear) == 2
        assert len(conf_exp) == 2
        assert len(conf_poly) == 2
    
    def test_zne_estimate_overhead(self):
        """Test ZNE overhead estimation."""
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0, 4.0])
        mock_circuit = Mock()
        
        overhead = zne.estimate_overhead(mock_circuit)
        
        assert overhead == 4.0  # Number of noise factors


class TestSymmetryVerification:
    """Test cases for SymmetryVerification class."""
    
    def test_sv_initialization(self):
        """Test SymmetryVerification initialization."""
        sv = SymmetryVerification(
            symmetry_groups=["task_assignment", "agent_capacity"],
            tolerance=1e-8,
            max_correction_iterations=5
        )
        
        assert "task_assignment" in sv.symmetry_groups
        assert "agent_capacity" in sv.symmetry_groups
        assert sv.tolerance == 1e-8
        assert sv.max_correction_iterations == 5
    
    def test_sv_default_initialization(self):
        """Test SymmetryVerification with default parameters."""
        sv = SymmetryVerification()
        
        assert len(sv.symmetry_groups) == 2
        assert sv.tolerance == 1e-6
        assert sv.max_correction_iterations == 3
    
    def test_sv_mitigate_errors_no_violations(self):
        """Test SymmetryVerification with no violations."""
        sv = SymmetryVerification()
        noise_model = NoiseModel()
        
        mock_circuit = Mock()
        
        # Mock no violations
        with patch.object(sv, '_check_symmetry_violations', return_value=[]):
            result = sv.mitigate_errors(mock_circuit, shots=1024, noise_model=noise_model)
        
        assert isinstance(result, ErrorMitigationResult)
        assert result.error_reduction == 0.0
        assert result.original_energy == result.mitigated_energy
    
    def test_sv_mitigate_errors_with_violations(self):
        """Test SymmetryVerification with violations."""
        sv = SymmetryVerification()
        noise_model = NoiseModel()
        
        mock_circuit = Mock()
        
        # Mock violations detected
        with patch.object(sv, '_check_symmetry_violations', return_value=['task_assignment']):
            with patch.object(sv, '_apply_symmetry_corrections', return_value=-12.0):
                result = sv.mitigate_errors(mock_circuit, shots=1024, noise_model=noise_model)
        
        assert isinstance(result, ErrorMitigationResult)
        assert result.error_reduction > 0
        assert result.mitigated_energy == -12.0
    
    def test_sv_check_symmetry_violations(self):
        """Test symmetry violation detection."""
        sv = SymmetryVerification(tolerance=0.1)
        mock_circuit = Mock()
        
        # Test with violation (non-integer energy)
        violations = sv._check_symmetry_violations(mock_circuit, energy=10.5)
        assert "task_assignment" in violations
        
        # Test with positive energy violation
        violations = sv._check_symmetry_violations(mock_circuit, energy=5.0)
        assert "agent_capacity" in violations
        
        # Test with no violations
        violations = sv._check_symmetry_violations(mock_circuit, energy=-8.0)
        assert len(violations) == 0
    
    def test_sv_apply_symmetry_corrections(self):
        """Test symmetry correction application."""
        sv = SymmetryVerification()
        mock_circuit = Mock()
        
        # Test task assignment correction
        corrected = sv._apply_symmetry_corrections(mock_circuit, 10.7, ["task_assignment"])
        assert corrected == 11.0  # Rounded to nearest integer
        
        # Test agent capacity correction
        corrected = sv._apply_symmetry_corrections(mock_circuit, 5.0, ["agent_capacity"])
        assert corrected == -5.0  # Made negative
        
        # Test multiple corrections
        corrected = sv._apply_symmetry_corrections(mock_circuit, 10.7, ["task_assignment", "agent_capacity"])
        assert corrected == -11.0
    
    def test_sv_estimate_overhead(self):
        """Test SymmetryVerification overhead estimation."""
        sv = SymmetryVerification()
        mock_circuit = Mock()
        
        overhead = sv.estimate_overhead(mock_circuit)
        
        assert overhead == 0.1  # Minimal overhead for post-processing


class TestAdaptiveErrorMitigation:
    """Test cases for AdaptiveErrorMitigation class."""
    
    def test_adaptive_initialization(self):
        """Test AdaptiveErrorMitigation initialization."""
        adaptive = AdaptiveErrorMitigation()
        
        assert ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION in adaptive.mitigators
        assert ErrorMitigationType.SYMMETRY_VERIFICATION in adaptive.mitigators
        assert len(adaptive.performance_history) == 0
    
    def test_adaptive_select_mitigation_strategy(self):
        """Test mitigation strategy selection."""
        adaptive = AdaptiveErrorMitigation()
        mock_circuit = Mock()
        noise_model = NoiseModel()
        
        # Test small problem
        strategies = adaptive.select_mitigation_strategy(mock_circuit, noise_model, problem_size=30)
        assert ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION in strategies
        
        # Test large problem
        strategies = adaptive.select_mitigation_strategy(mock_circuit, noise_model, problem_size=100)
        assert ErrorMitigationType.SYMMETRY_VERIFICATION in strategies
    
    def test_adaptive_apply_mitigation(self):
        """Test adaptive mitigation application."""
        adaptive = AdaptiveErrorMitigation()
        mock_circuit = Mock()
        noise_model = NoiseModel()
        
        result = adaptive.apply_mitigation(mock_circuit, shots=1024, noise_model=noise_model, problem_size=50)
        
        assert isinstance(result, ErrorMitigationResult)
        assert result.mitigation_overhead > 0
    
    def test_adaptive_combine_results_single(self):
        """Test result combination with single result."""
        adaptive = AdaptiveErrorMitigation()
        
        mock_result = ErrorMitigationResult(
            original_energy=-10.0,
            mitigated_energy=-12.0,
            error_reduction=0.2,
            mitigation_overhead=0.5,
            confidence_interval=(-13.0, -11.0),
            noise_amplification_factor=2.0,
            statistical_uncertainty=0.1
        )
        
        results = [(ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION, mock_result)]
        combined = adaptive._combine_results(results)
        
        assert combined.mitigated_energy == mock_result.mitigated_energy
        assert combined.statistical_uncertainty == mock_result.statistical_uncertainty
    
    def test_adaptive_combine_results_multiple(self):
        """Test result combination with multiple results."""
        adaptive = AdaptiveErrorMitigation()
        
        result1 = ErrorMitigationResult(
            original_energy=-10.0,
            mitigated_energy=-12.0,
            error_reduction=0.2,
            mitigation_overhead=0.5,
            confidence_interval=(-13.0, -11.0),
            noise_amplification_factor=2.0,
            statistical_uncertainty=0.1
        )
        
        result2 = ErrorMitigationResult(
            original_energy=-10.0,
            mitigated_energy=-11.0,
            error_reduction=0.1,
            mitigation_overhead=0.3,
            confidence_interval=(-12.0, -10.0),
            noise_amplification_factor=1.5,
            statistical_uncertainty=0.05
        )
        
        results = [
            (ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION, result1),
            (ErrorMitigationType.SYMMETRY_VERIFICATION, result2)
        ]
        
        combined = adaptive._combine_results(results)
        
        # Should be weighted average
        assert -12.0 < combined.mitigated_energy < -11.0
        assert 0.05 < combined.statistical_uncertainty < 0.1
    
    def test_adaptive_update_performance_history(self):
        """Test performance history updates."""
        adaptive = AdaptiveErrorMitigation()
        
        strategies = [ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION]
        mock_result = Mock()
        mock_result.error_reduction = 0.3
        
        adaptive._update_performance_history(strategies, mock_result)
        
        assert len(adaptive.performance_history[ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION]) == 1
        assert adaptive.performance_history[ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION][0] == 0.3


class TestNISQSchedulingOptimizer:
    """Test cases for NISQSchedulingOptimizer class."""
    
    def test_nisq_optimizer_initialization(self):
        """Test NISQSchedulingOptimizer initialization."""
        noise_model = NoiseModel()
        optimizer = NISQSchedulingOptimizer(noise_model=noise_model, mitigation_strategy="adaptive")
        
        assert optimizer.noise_model == noise_model
        assert optimizer.mitigation_strategy == "adaptive"
        assert isinstance(optimizer.error_mitigator, AdaptiveErrorMitigation)
    
    def test_nisq_optimizer_default_initialization(self):
        """Test NISQSchedulingOptimizer with default parameters."""
        optimizer = NISQSchedulingOptimizer()
        
        assert isinstance(optimizer.noise_model, NoiseModel)
        assert optimizer.mitigation_strategy == "adaptive"
        assert isinstance(optimizer.error_mitigator, AdaptiveErrorMitigation)
    
    @patch('quantum_scheduler.research.quantum_error_mitigation.HAS_QISKIT', False)
    def test_nisq_optimize_schedule(self):
        """Test NISQ scheduling optimization."""
        optimizer = NISQSchedulingOptimizer()
        
        # Create test QUBO matrix
        qubo_matrix = np.random.random((10, 10))
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2  # Make symmetric
        
        assignment, mitigation_result = optimizer.optimize_schedule(
            qubo_matrix, problem_size=10, shots=1024
        )
        
        assert isinstance(assignment, np.ndarray)
        assert len(assignment) == 10
        assert isinstance(mitigation_result, ErrorMitigationResult)
    
    @patch('quantum_scheduler.research.quantum_error_mitigation.HAS_QISKIT', False)
    def test_nisq_create_qubo_circuit(self):
        """Test QUBO circuit creation."""
        optimizer = NISQSchedulingOptimizer()
        
        qubo_matrix = np.random.random((5, 5))
        circuit = optimizer._create_qubo_circuit(qubo_matrix)
        
        # Should create mock circuit when Qiskit not available
        assert hasattr(circuit, 'num_qubits')
        assert circuit.num_qubits == 5
    
    def test_nisq_decode_quantum_solution(self):
        """Test quantum solution decoding."""
        optimizer = NISQSchedulingOptimizer()
        
        assignment = optimizer._decode_quantum_solution(energy=-8.5, problem_size=8)
        
        assert isinstance(assignment, np.ndarray)
        assert len(assignment) == 8
        assert all(x in [0, 1] for x in assignment)
    
    def test_nisq_benchmark_mitigation_effectiveness(self):
        """Test mitigation effectiveness benchmarking."""
        optimizer = NISQSchedulingOptimizer()
        
        # Create test problems
        test_problems = [
            np.random.random((5, 5)),
            np.random.random((8, 8)),
            np.random.random((10, 10))
        ]
        
        # Make symmetric
        for i, problem in enumerate(test_problems):
            test_problems[i] = (problem + problem.T) / 2
        
        results = optimizer.benchmark_mitigation_effectiveness(
            test_problems, shots_list=[512, 1024]
        )
        
        assert 'problem_sizes' in results
        assert 'error_reductions' in results
        assert 'mitigation_overheads' in results
        assert 'avg_error_reduction' in results
        assert len(results['problem_sizes']) == 6  # 3 problems × 2 shots configs


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_test_noise_model_ibm(self):
        """Test IBM quantum noise model creation."""
        noise_model = create_test_noise_model("ibm_quantum")
        
        assert noise_model.gate_error_rates['single_qubit'] == 0.0005
        assert noise_model.gate_error_rates['two_qubit'] == 0.015
        assert noise_model.measurement_error_rate == 0.02
        assert noise_model.decoherence_times['T1'] == 45e-6
    
    def test_create_test_noise_model_google(self):
        """Test Google quantum noise model creation."""
        noise_model = create_test_noise_model("google_quantum")
        
        assert noise_model.gate_error_rates['single_qubit'] == 0.0003
        assert noise_model.gate_error_rates['two_qubit'] == 0.012
        assert noise_model.measurement_error_rate == 0.015
        assert noise_model.decoherence_times['T1'] == 60e-6
    
    def test_create_test_noise_model_generic(self):
        """Test generic noise model creation."""
        noise_model = create_test_noise_model("unknown_device")
        
        # Should return default NoiseModel
        assert isinstance(noise_model, NoiseModel)
        assert 'single_qubit' in noise_model.gate_error_rates


class TestIntegration:
    """Integration tests for quantum error mitigation."""
    
    @patch('quantum_scheduler.research.quantum_error_mitigation.HAS_QISKIT', False)
    def test_end_to_end_error_mitigation(self):
        """Test complete error mitigation workflow."""
        # Create noise model
        noise_model = create_test_noise_model("ibm_quantum")
        
        # Initialize optimizer
        optimizer = NISQSchedulingOptimizer(noise_model=noise_model)
        
        # Create test problem
        qubo_matrix = np.array([
            [1, -2, 0],
            [-2, 3, -1],
            [0, -1, 2]
        ])
        
        # Optimize with error mitigation
        assignment, result = optimizer.optimize_schedule(qubo_matrix, problem_size=3, shots=1024)
        
        # Verify results
        assert isinstance(assignment, np.ndarray)
        assert len(assignment) == 3
        assert isinstance(result, ErrorMitigationResult)
        assert result.mitigation_overhead > 0
        assert not result.error_occurred if hasattr(result, 'error_occurred') else True
    
    def test_performance_comparison_multiple_strategies(self):
        """Test performance comparison between different strategies."""
        noise_model = NoiseModel()
        mock_circuit = Mock()
        mock_circuit.num_qubits = 10
        mock_circuit.data = [Mock() for _ in range(20)]
        
        # Test ZNE
        zne = ZeroNoiseExtrapolation()
        zne_result = zne.mitigate_errors(mock_circuit, shots=1024, noise_model=noise_model)
        
        # Test Symmetry Verification
        sv = SymmetryVerification()
        sv_result = sv.mitigate_errors(mock_circuit, shots=1024, noise_model=noise_model)
        
        # Test Adaptive
        adaptive = AdaptiveErrorMitigation()
        adaptive_result = adaptive.apply_mitigation(mock_circuit, shots=1024, noise_model=noise_model, problem_size=50)
        
        # All should return valid results
        assert isinstance(zne_result, ErrorMitigationResult)
        assert isinstance(sv_result, ErrorMitigationResult)
        assert isinstance(adaptive_result, ErrorMitigationResult)
        
        # Compare performance metrics
        assert zne_result.mitigation_overhead > 0
        assert sv_result.mitigation_overhead >= 0
        assert adaptive_result.mitigation_overhead > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])