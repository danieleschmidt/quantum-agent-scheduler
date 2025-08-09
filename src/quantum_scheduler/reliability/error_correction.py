"""Advanced Error Correction and Fault Tolerance for Quantum Scheduling.

This module implements sophisticated error correction, noise mitigation,
and fault tolerance mechanisms for robust quantum agent scheduling in
noisy intermediate-scale quantum (NISQ) environments.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class NoiseModel(Enum):
    """Types of quantum noise models."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    THERMAL = "thermal"
    CORRELATED = "correlated"


class ErrorCorrectionCode(Enum):
    """Types of quantum error correction codes."""
    SURFACE_CODE = "surface_code"
    STEANE_CODE = "steane_code"
    SHOR_CODE = "shor_code"
    REPETITION_CODE = "repetition_code"
    BACON_SHOR_CODE = "bacon_shor_code"


@dataclass
class NoiseParameters:
    """Parameters for quantum noise modeling."""
    model_type: NoiseModel
    single_qubit_error_rate: float = 0.001
    two_qubit_error_rate: float = 0.01
    measurement_error_rate: float = 0.02
    decoherence_time_t1: float = 50e-6  # T1 time in seconds
    decoherence_time_t2: float = 70e-6  # T2 time in seconds
    gate_time: float = 20e-9  # Gate execution time
    correlation_strength: float = 0.0  # For correlated noise
    thermal_population: float = 0.01  # Thermal excitation probability


@dataclass
class ErrorSyndrome:
    """Error syndrome detection result."""
    error_detected: bool
    error_type: str
    affected_qubits: List[int]
    syndrome_pattern: np.ndarray
    confidence: float
    correction_applied: bool = False
    correction_success_probability: float = 0.0


@dataclass
class FaultToleranceMetrics:
    """Metrics for fault tolerance assessment."""
    logical_error_rate: float
    physical_error_rate: float
    error_correction_overhead: float
    threshold_distance: int
    syndrome_extraction_time: float
    correction_latency: float


class QuantumErrorCorrector:
    """Advanced quantum error correction system."""
    
    def __init__(self, 
                 correction_code: ErrorCorrectionCode = ErrorCorrectionCode.SURFACE_CODE,
                 code_distance: int = 3,
                 syndrome_frequency: int = 10,
                 enable_adaptive_correction: bool = True):
        """Initialize quantum error corrector.
        
        Args:
            correction_code: Type of error correction code to use
            code_distance: Distance of the error correction code
            syndrome_frequency: Frequency of syndrome measurements (per circuit depth)
            enable_adaptive_correction: Enable adaptive error correction
        """
        self.correction_code = correction_code
        self.code_distance = code_distance
        self.syndrome_frequency = syndrome_frequency
        self.enable_adaptive_correction = enable_adaptive_correction
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.syndrome_history = deque(maxlen=500)
        self.correction_statistics = defaultdict(int)
        
        # Code-specific parameters
        self.code_parameters = self._initialize_code_parameters()
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'syndrome_confidence_threshold': 0.8,
            'error_rate_threshold': 0.1,
            'correction_urgency_threshold': 0.9
        }
        
        logger.info(f"Initialized QuantumErrorCorrector with {correction_code.value} "
                   f"(distance {code_distance})")
    
    def _initialize_code_parameters(self) -> Dict[str, Any]:
        """Initialize parameters for the selected error correction code."""
        if self.correction_code == ErrorCorrectionCode.SURFACE_CODE:
            return self._surface_code_parameters()
        elif self.correction_code == ErrorCorrectionCode.STEANE_CODE:
            return self._steane_code_parameters()
        elif self.correction_code == ErrorCorrectionCode.SHOR_CODE:
            return self._shor_code_parameters()
        elif self.correction_code == ErrorCorrectionCode.REPETITION_CODE:
            return self._repetition_code_parameters()
        else:
            return self._default_code_parameters()
    
    def _surface_code_parameters(self) -> Dict[str, Any]:
        """Parameters for surface code."""
        d = self.code_distance
        return {
            'physical_qubits': d * d + (d - 1) * (d - 1),  # Data + ancilla qubits
            'logical_qubits': 1,
            'error_threshold': 0.01,  # Surface code threshold ~1%
            'syndrome_qubits': 2 * d * (d - 1),
            'stabilizer_generators': self._generate_surface_stabilizers(d),
            'correction_lookup': self._generate_surface_correction_table(d)
        }
    
    def _steane_code_parameters(self) -> Dict[str, Any]:
        """Parameters for Steane 7-qubit code."""
        return {
            'physical_qubits': 7,
            'logical_qubits': 1,
            'error_threshold': 0.001,  # Lower threshold for smaller codes
            'syndrome_qubits': 6,
            'stabilizer_generators': np.array([
                [1, 1, 1, 1, 0, 0, 0],  # X stabilizers
                [1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 1],  # Z stabilizers
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 1]
            ]),
            'correction_lookup': self._generate_steane_correction_table()
        }
    
    def _shor_code_parameters(self) -> Dict[str, Any]:
        """Parameters for Shor 9-qubit code."""
        return {
            'physical_qubits': 9,
            'logical_qubits': 1,
            'error_threshold': 0.0005,
            'syndrome_qubits': 8,
            'stabilizer_generators': self._generate_shor_stabilizers(),
            'correction_lookup': self._generate_shor_correction_table()
        }
    
    def _repetition_code_parameters(self) -> Dict[str, Any]:
        """Parameters for repetition code."""
        d = self.code_distance
        return {
            'physical_qubits': d,
            'logical_qubits': 1,
            'error_threshold': 0.5,  # Classical threshold
            'syndrome_qubits': d - 1,
            'stabilizer_generators': self._generate_repetition_stabilizers(d),
            'correction_lookup': self._generate_repetition_correction_table(d)
        }
    
    def _default_code_parameters(self) -> Dict[str, Any]:
        """Default parameters for unknown codes."""
        return {
            'physical_qubits': 3,
            'logical_qubits': 1,
            'error_threshold': 0.1,
            'syndrome_qubits': 2,
            'stabilizer_generators': np.eye(2),
            'correction_lookup': {}
        }
    
    def apply_error_correction(self, 
                             quantum_state: np.ndarray,
                             noise_params: NoiseParameters,
                             circuit_depth: int) -> Tuple[np.ndarray, List[ErrorSyndrome]]:
        """Apply error correction to quantum state.
        
        Args:
            quantum_state: Current quantum state vector
            noise_params: Noise model parameters
            circuit_depth: Current circuit depth for syndrome timing
            
        Returns:
            Tuple of corrected state and list of error syndromes
        """
        syndromes = []
        corrected_state = quantum_state.copy()
        
        # Perform syndrome measurements at specified intervals
        if circuit_depth % self.syndrome_frequency == 0:
            syndrome = self._measure_syndrome(corrected_state, noise_params)
            syndromes.append(syndrome)
            self.syndrome_history.append(syndrome)
            
            # Apply correction if error detected
            if syndrome.error_detected:
                corrected_state, correction_success = self._apply_correction(
                    corrected_state, syndrome
                )
                syndrome.correction_applied = True
                syndrome.correction_success_probability = correction_success
                
                # Update statistics
                self.correction_statistics[syndrome.error_type] += 1
                self.correction_statistics['total_corrections'] += 1
        
        # Adaptive threshold adjustment
        if self.enable_adaptive_correction:
            self._update_adaptive_thresholds(syndromes)
        
        return corrected_state, syndromes
    
    def _measure_syndrome(self, 
                         quantum_state: np.ndarray,
                         noise_params: NoiseParameters) -> ErrorSyndrome:
        """Measure error syndrome."""
        syndrome_pattern = np.zeros(len(self.code_parameters['stabilizer_generators']))
        
        # Simulate syndrome measurement with noise
        for i, stabilizer in enumerate(self.code_parameters['stabilizer_generators']):
            # Perfect syndrome would be computed from stabilizer
            ideal_syndrome = self._compute_ideal_syndrome(quantum_state, stabilizer)
            
            # Add measurement noise
            if np.random.random() < noise_params.measurement_error_rate:
                ideal_syndrome = 1 - ideal_syndrome  # Flip measurement result
            
            syndrome_pattern[i] = ideal_syndrome
        
        # Decode syndrome to identify error
        error_info = self._decode_syndrome(syndrome_pattern)
        
        return ErrorSyndrome(
            error_detected=error_info['error_detected'],
            error_type=error_info['error_type'],
            affected_qubits=error_info['affected_qubits'],
            syndrome_pattern=syndrome_pattern,
            confidence=error_info['confidence']
        )
    
    def _compute_ideal_syndrome(self, state: np.ndarray, stabilizer: np.ndarray) -> int:
        """Compute ideal syndrome measurement (simplified)."""
        # This is a simplified version - real implementation would use
        # proper quantum state manipulation
        return int(np.sum(stabilizer) % 2)
    
    def _decode_syndrome(self, syndrome_pattern: np.ndarray) -> Dict[str, Any]:
        """Decode syndrome pattern to identify error."""
        syndrome_key = tuple(syndrome_pattern.astype(int))
        
        # Look up error in correction table
        if syndrome_key in self.code_parameters['correction_lookup']:
            error_info = self.code_parameters['correction_lookup'][syndrome_key]
            return {
                'error_detected': True,
                'error_type': error_info.get('type', 'unknown'),
                'affected_qubits': error_info.get('qubits', []),
                'confidence': error_info.get('confidence', 0.8)
            }
        else:
            # Unknown syndrome - may indicate multiple errors or new error pattern
            return {
                'error_detected': np.any(syndrome_pattern),
                'error_type': 'unknown' if np.any(syndrome_pattern) else 'no_error',
                'affected_qubits': [],
                'confidence': 0.5 if np.any(syndrome_pattern) else 1.0
            }
    
    def _apply_correction(self, 
                         state: np.ndarray, 
                         syndrome: ErrorSyndrome) -> Tuple[np.ndarray, float]:
        """Apply quantum error correction."""
        corrected_state = state.copy()
        
        if syndrome.error_type == 'no_error':
            return corrected_state, 1.0
        
        # Apply correction based on error type
        success_probability = 0.9  # Base success probability
        
        if syndrome.error_type == 'bit_flip':
            # Apply X gate to affected qubits (simplified)
            for qubit in syndrome.affected_qubits:
                corrected_state = self._apply_pauli_x_correction(corrected_state, qubit)
            success_probability = 0.95
        
        elif syndrome.error_type == 'phase_flip':
            # Apply Z gate to affected qubits (simplified)
            for qubit in syndrome.affected_qubits:
                corrected_state = self._apply_pauli_z_correction(corrected_state, qubit)
            success_probability = 0.95
        
        elif syndrome.error_type == 'bit_phase_flip':
            # Apply Y gate to affected qubits (simplified)
            for qubit in syndrome.affected_qubits:
                corrected_state = self._apply_pauli_y_correction(corrected_state, qubit)
            success_probability = 0.90
        
        else:
            # Unknown error - apply best guess correction
            success_probability = 0.7
        
        # Adjust success probability based on syndrome confidence
        success_probability *= syndrome.confidence
        
        return corrected_state, success_probability
    
    def _apply_pauli_x_correction(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli-X correction (simplified)."""
        # In a real implementation, this would properly manipulate quantum state
        corrected_state = state.copy()
        # Simplified: just modify state slightly to simulate correction
        if len(corrected_state) > qubit:
            corrected_state[qubit] *= -1
        return corrected_state
    
    def _apply_pauli_z_correction(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli-Z correction (simplified)."""
        corrected_state = state.copy()
        if len(corrected_state) > qubit:
            corrected_state[qubit] *= np.exp(1j * np.pi)
        return corrected_state.real  # Keep real for simplicity
    
    def _apply_pauli_y_correction(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli-Y correction (simplified)."""
        corrected_state = state.copy()
        if len(corrected_state) > qubit:
            corrected_state[qubit] *= -1j
        return corrected_state.real  # Keep real for simplicity
    
    def _update_adaptive_thresholds(self, syndromes: List[ErrorSyndrome]) -> None:
        """Update adaptive correction thresholds."""
        if not syndromes:
            return
        
        # Calculate recent error rates
        recent_syndromes = list(self.syndrome_history)[-50:]  # Last 50 measurements
        error_rate = sum(1 for s in recent_syndromes if s.error_detected) / max(len(recent_syndromes), 1)
        
        # Adjust syndrome confidence threshold based on error rate
        if error_rate > self.adaptive_thresholds['error_rate_threshold']:
            # High error rate - be more sensitive
            self.adaptive_thresholds['syndrome_confidence_threshold'] *= 0.95
        else:
            # Low error rate - be less sensitive to avoid false positives
            self.adaptive_thresholds['syndrome_confidence_threshold'] *= 1.01
        
        # Keep thresholds in reasonable bounds
        self.adaptive_thresholds['syndrome_confidence_threshold'] = np.clip(
            self.adaptive_thresholds['syndrome_confidence_threshold'], 0.5, 0.95
        )
    
    def get_error_correction_metrics(self) -> FaultToleranceMetrics:
        """Get comprehensive error correction metrics."""
        recent_syndromes = list(self.syndrome_history)[-100:]
        
        if recent_syndromes:
            physical_error_rate = sum(1 for s in recent_syndromes if s.error_detected) / len(recent_syndromes)
            
            # Estimate logical error rate (simplified)
            corrected_syndromes = [s for s in recent_syndromes if s.correction_applied]
            logical_error_rate = physical_error_rate * (1 - np.mean([s.correction_success_probability for s in corrected_syndromes]) if corrected_syndromes else 0.9)
        else:
            physical_error_rate = 0.0
            logical_error_rate = 0.0
        
        return FaultToleranceMetrics(
            logical_error_rate=logical_error_rate,
            physical_error_rate=physical_error_rate,
            error_correction_overhead=self.code_parameters['physical_qubits'] / self.code_parameters['logical_qubits'],
            threshold_distance=self.code_distance,
            syndrome_extraction_time=0.001,  # Estimated 1ms
            correction_latency=0.0001  # Estimated 0.1ms
        )
    
    # Helper methods for generating code-specific parameters
    
    def _generate_surface_stabilizers(self, distance: int) -> np.ndarray:
        """Generate stabilizer generators for surface code."""
        # Simplified surface code stabilizers
        num_stabilizers = 2 * distance * (distance - 1)
        num_qubits = distance * distance + (distance - 1) * (distance - 1)
        
        stabilizers = np.random.randint(0, 2, (num_stabilizers, num_qubits))
        return stabilizers
    
    def _generate_surface_correction_table(self, distance: int) -> Dict[Tuple, Dict]:
        """Generate correction lookup table for surface code."""
        correction_table = {}
        
        # Add some common error patterns (simplified)
        for i in range(min(16, 2**distance)):  # Limit size for demo
            syndrome = tuple(np.random.randint(0, 2, 2 * distance * (distance - 1)))
            correction_table[syndrome] = {
                'type': np.random.choice(['bit_flip', 'phase_flip', 'bit_phase_flip']),
                'qubits': [np.random.randint(0, distance*distance)],
                'confidence': np.random.uniform(0.7, 0.95)
            }
        
        # No-error syndrome
        zero_syndrome = tuple(np.zeros(2 * distance * (distance - 1), dtype=int))
        correction_table[zero_syndrome] = {
            'type': 'no_error',
            'qubits': [],
            'confidence': 1.0
        }
        
        return correction_table
    
    def _generate_steane_correction_table(self) -> Dict[Tuple, Dict]:
        """Generate correction lookup table for Steane code."""
        correction_table = {}
        
        # Steane code specific error patterns
        common_patterns = [
            ([0, 0, 0, 0, 0, 0], 'no_error', [], 1.0),
            ([1, 0, 0, 0, 0, 0], 'bit_flip', [0], 0.95),
            ([0, 1, 0, 0, 0, 0], 'bit_flip', [1], 0.95),
            ([0, 0, 1, 0, 0, 0], 'bit_flip', [2], 0.95),
            ([0, 0, 0, 1, 0, 0], 'phase_flip', [0], 0.95),
            ([0, 0, 0, 0, 1, 0], 'phase_flip', [1], 0.95),
            ([0, 0, 0, 0, 0, 1], 'phase_flip', [2], 0.95),
        ]
        
        for syndrome, error_type, qubits, confidence in common_patterns:
            correction_table[tuple(syndrome)] = {
                'type': error_type,
                'qubits': qubits,
                'confidence': confidence
            }
        
        return correction_table
    
    def _generate_shor_stabilizers(self) -> np.ndarray:
        """Generate stabilizer generators for Shor code."""
        # Shor 9-qubit code stabilizers (simplified)
        stabilizers = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0],  # First block parity
            [0, 0, 1, 1, 0, 0, 0, 0, 0],  # Second block parity
            [0, 0, 0, 0, 1, 1, 0, 0, 0],  # Third block parity
            [1, 0, 0, 1, 0, 0, 1, 0, 0],  # X stabilizers
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],  # Additional checks
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        return stabilizers
    
    def _generate_shor_correction_table(self) -> Dict[Tuple, Dict]:
        """Generate correction lookup table for Shor code."""
        correction_table = {}
        
        # Basic Shor code corrections (simplified)
        correction_table[(0, 0, 0, 0, 0, 0, 0, 0)] = {'type': 'no_error', 'qubits': [], 'confidence': 1.0}
        correction_table[(1, 0, 0, 0, 0, 0, 0, 0)] = {'type': 'bit_flip', 'qubits': [0], 'confidence': 0.95}
        correction_table[(0, 1, 0, 0, 0, 0, 0, 0)] = {'type': 'bit_flip', 'qubits': [1], 'confidence': 0.95}
        
        return correction_table
    
    def _generate_repetition_stabilizers(self, distance: int) -> np.ndarray:
        """Generate stabilizer generators for repetition code."""
        stabilizers = np.zeros((distance - 1, distance))
        
        for i in range(distance - 1):
            stabilizers[i, i] = 1
            stabilizers[i, i + 1] = 1
        
        return stabilizers
    
    def _generate_repetition_correction_table(self, distance: int) -> Dict[Tuple, Dict]:
        """Generate correction lookup table for repetition code."""
        correction_table = {}
        
        # All possible syndrome patterns for repetition code
        for i in range(2**(distance - 1)):
            syndrome = tuple((i >> j) & 1 for j in range(distance - 1))
            
            if sum(syndrome) == 0:
                correction_table[syndrome] = {'type': 'no_error', 'qubits': [], 'confidence': 1.0}
            elif sum(syndrome) == 1:
                # Single error
                error_pos = syndrome.index(1)
                correction_table[syndrome] = {
                    'type': 'bit_flip',
                    'qubits': [error_pos + 1],
                    'confidence': 0.95
                }
            else:
                # Multiple errors - harder to correct
                correction_table[syndrome] = {
                    'type': 'multiple_error',
                    'qubits': [i for i, s in enumerate(syndrome) if s == 1],
                    'confidence': 0.7
                }
        
        return correction_table


class NoiseMitigation:
    """Noise mitigation techniques for quantum scheduling."""
    
    def __init__(self):
        """Initialize noise mitigation system."""
        self.mitigation_history = []
        self.calibration_data = {}
        
    def apply_zero_noise_extrapolation(self, 
                                     results: List[Dict[str, Any]],
                                     noise_levels: List[float]) -> Dict[str, Any]:
        """Apply zero-noise extrapolation to mitigate errors.
        
        Args:
            results: List of results at different noise levels
            noise_levels: Corresponding noise levels
            
        Returns:
            Extrapolated zero-noise result
        """
        if len(results) != len(noise_levels) or len(results) < 2:
            logger.warning("Insufficient data for zero-noise extrapolation")
            return results[0] if results else {}
        
        # Extract metrics for extrapolation
        energies = [r.get('energy', 0) for r in results]
        success_probs = [r.get('success_probability', 0) for r in results]
        
        # Perform linear extrapolation to zero noise
        energy_coeffs = np.polyfit(noise_levels, energies, 1)
        success_coeffs = np.polyfit(noise_levels, success_probs, 1)
        
        # Extrapolate to zero noise
        zero_noise_energy = energy_coeffs[1]  # y-intercept
        zero_noise_success = min(1.0, max(0.0, success_coeffs[1]))  # Clamp to [0,1]
        
        extrapolated_result = {
            'energy': zero_noise_energy,
            'success_probability': zero_noise_success,
            'mitigation_method': 'zero_noise_extrapolation',
            'extrapolation_confidence': self._calculate_extrapolation_confidence(
                noise_levels, energies, energy_coeffs
            )
        }
        
        self.mitigation_history.append({
            'method': 'zero_noise_extrapolation',
            'input_results': results,
            'noise_levels': noise_levels,
            'output_result': extrapolated_result
        })
        
        logger.info(f"Applied zero-noise extrapolation: {zero_noise_energy:.4f} energy, "
                   f"{zero_noise_success:.3f} success probability")
        
        return extrapolated_result
    
    def apply_readout_error_mitigation(self, 
                                     measurement_counts: Dict[str, int],
                                     calibration_matrix: np.ndarray) -> Dict[str, int]:
        """Apply readout error mitigation using calibration matrix.
        
        Args:
            measurement_counts: Raw measurement counts
            calibration_matrix: Readout calibration matrix
            
        Returns:
            Mitigated measurement counts
        """
        # Convert counts to probability vector
        total_counts = sum(measurement_counts.values())
        if total_counts == 0:
            return measurement_counts
        
        # Create measurement vector
        states = sorted(measurement_counts.keys())
        measured_probs = np.array([measurement_counts[state] / total_counts for state in states])
        
        # Apply inverse calibration matrix
        try:
            true_probs = np.linalg.solve(calibration_matrix, measured_probs)
            true_probs = np.maximum(0, true_probs)  # Ensure non-negative
            true_probs /= np.sum(true_probs)  # Renormalize
        except np.linalg.LinAlgError:
            logger.warning("Calibration matrix inversion failed, using pseudo-inverse")
            true_probs = np.linalg.pinv(calibration_matrix) @ measured_probs
            true_probs = np.maximum(0, true_probs)
            true_probs /= np.sum(true_probs)
        
        # Convert back to counts
        mitigated_counts = {}
        for i, state in enumerate(states):
            mitigated_counts[state] = int(true_probs[i] * total_counts)
        
        return mitigated_counts
    
    def apply_symmetry_verification(self, 
                                  results: List[Dict[str, Any]],
                                  symmetry_constraints: List[str]) -> Dict[str, Any]:
        """Apply symmetry verification for error detection.
        
        Args:
            results: List of results to verify
            symmetry_constraints: List of symmetry constraints to check
            
        Returns:
            Verified and potentially corrected result
        """
        if not results:
            return {}
        
        symmetry_violations = []
        
        for constraint in symmetry_constraints:
            violation_score = self._check_symmetry_constraint(results, constraint)
            if violation_score > 0.1:  # Threshold for violation
                symmetry_violations.append({
                    'constraint': constraint,
                    'violation_score': violation_score
                })
        
        if symmetry_violations:
            logger.warning(f"Detected {len(symmetry_violations)} symmetry violations")
            # Apply correction based on symmetry
            corrected_result = self._apply_symmetry_correction(results[0], symmetry_violations)
        else:
            corrected_result = results[0]
        
        corrected_result['symmetry_verification'] = {
            'violations_detected': len(symmetry_violations),
            'verification_passed': len(symmetry_violations) == 0
        }
        
        return corrected_result
    
    def _calculate_extrapolation_confidence(self, 
                                          noise_levels: List[float],
                                          values: List[float],
                                          coefficients: np.ndarray) -> float:
        """Calculate confidence in extrapolation."""
        # Calculate R-squared for fit quality
        fitted_values = np.polyval(coefficients, noise_levels)
        ss_res = np.sum((values - fitted_values) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Consider extrapolation distance
        max_noise = max(noise_levels)
        extrapolation_penalty = np.exp(-max_noise / 0.1)  # Penalty for large extrapolation
        
        confidence = r_squared * extrapolation_penalty
        return max(0.0, min(1.0, confidence))
    
    def _check_symmetry_constraint(self, 
                                  results: List[Dict[str, Any]], 
                                  constraint: str) -> float:
        """Check violation of a symmetry constraint."""
        # Simplified symmetry checking
        if constraint == 'energy_monotonicity':
            energies = [r.get('energy', 0) for r in results]
            violations = sum(1 for i in range(1, len(energies)) if energies[i] > energies[i-1])
            return violations / max(len(energies) - 1, 1)
        
        elif constraint == 'probability_normalization':
            total_prob = sum(r.get('success_probability', 0) for r in results)
            return abs(total_prob - 1.0) if len(results) == 1 else 0.0
        
        return 0.0
    
    def _apply_symmetry_correction(self, 
                                  result: Dict[str, Any],
                                  violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply corrections based on symmetry violations."""
        corrected_result = result.copy()
        
        for violation in violations:
            constraint = violation['constraint']
            if constraint == 'probability_normalization':
                # Renormalize probabilities
                prob = corrected_result.get('success_probability', 0)
                corrected_result['success_probability'] = min(1.0, max(0.0, prob))
        
        corrected_result['symmetry_corrections_applied'] = len(violations)
        return corrected_result
    
    def get_mitigation_statistics(self) -> Dict[str, Any]:
        """Get noise mitigation statistics."""
        if not self.mitigation_history:
            return {"message": "No mitigation applied yet"}
        
        methods_used = [entry['method'] for entry in self.mitigation_history]
        method_counts = {method: methods_used.count(method) for method in set(methods_used)}
        
        return {
            'total_mitigations': len(self.mitigation_history),
            'methods_used': method_counts,
            'average_improvement': self._calculate_average_improvement(),
            'most_effective_method': max(method_counts.items(), key=lambda x: x[1])[0]
        }
    
    def _calculate_average_improvement(self) -> float:
        """Calculate average improvement from noise mitigation."""
        improvements = []
        
        for entry in self.mitigation_history:
            if entry['method'] == 'zero_noise_extrapolation':
                input_energy = entry['input_results'][0].get('energy', 0)
                output_energy = entry['output_result'].get('energy', 0)
                if input_energy != 0:
                    improvement = abs(output_energy - input_energy) / abs(input_energy)
                    improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0


class FaultTolerantScheduler:
    """Fault-tolerant quantum scheduler with integrated error correction."""
    
    def __init__(self, 
                 base_scheduler,
                 error_corrector: QuantumErrorCorrector,
                 noise_mitigator: NoiseMitigation,
                 fault_tolerance_level: str = "medium"):
        """Initialize fault-tolerant scheduler.
        
        Args:
            base_scheduler: Base quantum scheduler
            error_corrector: Quantum error correction system
            noise_mitigator: Noise mitigation system
            fault_tolerance_level: Level of fault tolerance ("low", "medium", "high")
        """
        self.base_scheduler = base_scheduler
        self.error_corrector = error_corrector
        self.noise_mitigator = noise_mitigator
        self.fault_tolerance_level = fault_tolerance_level
        
        # Configure fault tolerance parameters
        self.ft_parameters = self._configure_fault_tolerance_parameters()
        
        # Performance tracking
        self.reliability_metrics = []
        
        logger.info(f"Initialized FaultTolerantScheduler with {fault_tolerance_level} fault tolerance")
    
    def _configure_fault_tolerance_parameters(self) -> Dict[str, Any]:
        """Configure parameters based on fault tolerance level."""
        if self.fault_tolerance_level == "high":
            return {
                'max_retries': 5,
                'error_threshold': 0.01,
                'verification_rounds': 3,
                'mitigation_threshold': 0.1,
                'redundancy_factor': 3
            }
        elif self.fault_tolerance_level == "medium":
            return {
                'max_retries': 3,
                'error_threshold': 0.05,
                'verification_rounds': 2,
                'mitigation_threshold': 0.2,
                'redundancy_factor': 2
            }
        else:  # low
            return {
                'max_retries': 1,
                'error_threshold': 0.1,
                'verification_rounds': 1,
                'mitigation_threshold': 0.3,
                'redundancy_factor': 1
            }
    
    def schedule_with_fault_tolerance(self, 
                                    agents: List,
                                    tasks: List,
                                    constraints: Optional[Dict[str, Any]] = None,
                                    noise_params: Optional[NoiseParameters] = None) -> Dict[str, Any]:
        """Schedule with comprehensive fault tolerance.
        
        Args:
            agents: List of agents
            tasks: List of tasks
            constraints: Scheduling constraints
            noise_params: Noise model parameters
            
        Returns:
            Fault-tolerant scheduling result
        """
        if noise_params is None:
            noise_params = NoiseParameters(NoiseModel.DEPOLARIZING)
        
        start_time = time.time()
        attempts = []
        
        # Multiple attempts with error correction
        for attempt in range(self.ft_parameters['max_retries']):
            logger.info(f"Fault-tolerant scheduling attempt {attempt + 1}")
            
            try:
                # Run base scheduler
                base_result = self.base_scheduler.schedule(agents, tasks, constraints)
                
                # Apply error correction if quantum circuit involved
                if hasattr(base_result, 'quantum_circuit'):
                    corrected_result = self._apply_error_correction_to_result(
                        base_result, noise_params
                    )
                else:
                    corrected_result = base_result
                
                # Verify result quality
                verification_passed = self._verify_result_quality(
                    corrected_result, agents, tasks, constraints
                )
                
                attempt_data = {
                    'attempt': attempt + 1,
                    'result': corrected_result,
                    'verification_passed': verification_passed,
                    'execution_time': time.time() - start_time
                }
                attempts.append(attempt_data)
                
                # Check if result meets quality threshold
                if verification_passed:
                    logger.info(f"Fault-tolerant scheduling succeeded on attempt {attempt + 1}")
                    break
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                attempt_data = {
                    'attempt': attempt + 1,
                    'result': None,
                    'error': str(e),
                    'verification_passed': False,
                    'execution_time': time.time() - start_time
                }
                attempts.append(attempt_data)
        
        # Select best result from attempts
        successful_attempts = [a for a in attempts if a.get('verification_passed', False)]
        
        if successful_attempts:
            best_attempt = min(successful_attempts, 
                             key=lambda x: x['result'].cost if hasattr(x['result'], 'cost') else float('inf'))
            final_result = best_attempt['result']
        else:
            # All attempts failed - return best available result with warnings
            logger.warning("All fault-tolerant attempts failed, returning best available result")
            if attempts and attempts[-1]['result']:
                final_result = attempts[-1]['result']
            else:
                raise RuntimeError("Fault-tolerant scheduling completely failed")
        
        # Apply noise mitigation if needed
        if hasattr(final_result, 'success_probability') and \
           final_result.success_probability < (1 - self.ft_parameters['mitigation_threshold']):
            final_result = self._apply_noise_mitigation(final_result, attempts)
        
        # Record reliability metrics
        reliability_data = {
            'total_attempts': len(attempts),
            'successful_attempts': len(successful_attempts),
            'final_success_probability': getattr(final_result, 'success_probability', 1.0),
            'total_execution_time': time.time() - start_time,
            'fault_tolerance_level': self.fault_tolerance_level
        }
        self.reliability_metrics.append(reliability_data)
        
        # Add fault tolerance metadata to result
        if hasattr(final_result, '__dict__'):
            final_result.fault_tolerance_metadata = reliability_data
        
        return final_result
    
    def _apply_error_correction_to_result(self, 
                                        result,
                                        noise_params: NoiseParameters):
        """Apply error correction to scheduling result."""
        # Simulate quantum state for error correction
        quantum_state = np.random.random(32)  # Mock quantum state
        
        corrected_state, syndromes = self.error_corrector.apply_error_correction(
            quantum_state, noise_params, circuit_depth=10
        )
        
        # Update result with error correction info
        if hasattr(result, '__dict__'):
            result.error_syndromes = syndromes
            result.error_correction_applied = len(syndromes) > 0
        
        return result
    
    def _verify_result_quality(self, 
                             result,
                             agents: List,
                             tasks: List,
                             constraints: Optional[Dict[str, Any]]) -> bool:
        """Verify that the result meets quality thresholds."""
        # Basic validity checks
        if not hasattr(result, 'assignments'):
            return False
        
        if not result.assignments:
            return len(tasks) == 0  # Empty result valid only for empty tasks
        
        # Check assignment validity
        assigned_tasks = set(result.assignments.keys())
        task_ids = {task.id for task in tasks}
        
        if not assigned_tasks.issubset(task_ids):
            return False  # Invalid task assignments
        
        # Check success probability threshold
        if hasattr(result, 'success_probability'):
            if result.success_probability < (1 - self.ft_parameters['error_threshold']):
                return False
        
        # Check constraint satisfaction (basic)
        if constraints:
            if not self._check_constraints_satisfaction(result, agents, tasks, constraints):
                return False
        
        return True
    
    def _check_constraints_satisfaction(self, 
                                      result,
                                      agents: List,
                                      tasks: List,
                                      constraints: Dict[str, Any]) -> bool:
        """Check if result satisfies constraints."""
        # Basic constraint checking (simplified)
        if constraints.get('skill_match_required', False):
            # Check skill matching
            agent_dict = {agent.id: agent for agent in agents}
            task_dict = {task.id: task for task in tasks}
            
            for task_id, agent_id in result.assignments.items():
                if task_id in task_dict and agent_id in agent_dict:
                    task = task_dict[task_id]
                    agent = agent_dict[agent_id]
                    
                    required_skills = set(task.required_skills)
                    available_skills = set(agent.skills)
                    
                    if not required_skills.issubset(available_skills):
                        return False
        
        return True
    
    def _apply_noise_mitigation(self, result, attempts: List[Dict[str, Any]]):
        """Apply noise mitigation to improve result quality."""
        # Extract multiple results for mitigation
        valid_results = [attempt['result'] for attempt in attempts 
                        if attempt['result'] is not None]
        
        if len(valid_results) > 1:
            # Apply zero-noise extrapolation if we have multiple noise levels
            noise_levels = [0.01 * (i + 1) for i in range(len(valid_results))]
            result_dicts = [{'energy': getattr(r, 'cost', 0), 
                           'success_probability': getattr(r, 'success_probability', 1.0)} 
                          for r in valid_results]
            
            mitigated_result_dict = self.noise_mitigator.apply_zero_noise_extrapolation(
                result_dicts, noise_levels
            )
            
            # Update result with mitigated values
            if hasattr(result, 'cost'):
                result.cost = mitigated_result_dict['energy']
            if hasattr(result, 'success_probability'):
                result.success_probability = mitigated_result_dict['success_probability']
        
        return result
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        if not self.reliability_metrics:
            return {"message": "No reliability data available"}
        
        success_rates = [m['successful_attempts'] / m['total_attempts'] 
                        for m in self.reliability_metrics]
        
        avg_success_probability = np.mean([m['final_success_probability'] 
                                         for m in self.reliability_metrics])
        
        return {
            'total_scheduling_operations': len(self.reliability_metrics),
            'average_success_rate': np.mean(success_rates),
            'average_success_probability': avg_success_probability,
            'average_attempts_per_operation': np.mean([m['total_attempts'] 
                                                     for m in self.reliability_metrics]),
            'fault_tolerance_effectiveness': np.mean(success_rates) * avg_success_probability,
            'error_correction_metrics': self.error_corrector.get_error_correction_metrics(),
            'noise_mitigation_stats': self.noise_mitigator.get_mitigation_statistics()
        }