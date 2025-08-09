"""Quantum Circuit Depth Optimization for Enhanced QUBO Solving.

This module implements novel quantum circuit optimization techniques to reduce
circuit depth while maintaining solution quality for QUBO problems in agent scheduling.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CircuitMetrics:
    """Metrics for quantum circuit analysis."""
    depth: int
    gate_count: int
    two_qubit_gates: int
    parallelization_factor: float
    estimated_execution_time: float
    fidelity_estimate: float


@dataclass
class OptimizationResult:
    """Result of circuit optimization."""
    original_metrics: CircuitMetrics
    optimized_metrics: CircuitMetrics
    improvement_factor: float
    optimization_time: float
    technique_used: str


class AdaptiveCircuitOptimizer:
    """Adaptive quantum circuit optimizer for QUBO scheduling problems."""
    
    def __init__(self, 
                 noise_model: Optional[Dict[str, float]] = None,
                 hardware_constraints: Optional[Dict[str, Any]] = None):
        """Initialize the adaptive circuit optimizer.
        
        Args:
            noise_model: Dictionary of noise parameters for different gates
            hardware_constraints: Hardware-specific constraints (connectivity, gate times)
        """
        self.noise_model = noise_model or self._default_noise_model()
        self.hardware_constraints = hardware_constraints or self._default_hardware_constraints()
        self.optimization_history = []
        
        logger.info("Initialized AdaptiveCircuitOptimizer with noise model")
    
    def _default_noise_model(self) -> Dict[str, float]:
        """Default noise model for quantum gates."""
        return {
            'single_qubit_error': 0.001,
            'two_qubit_error': 0.01,
            'readout_error': 0.02,
            'decoherence_time_t1': 50e-6,  # 50 microseconds
            'decoherence_time_t2': 70e-6   # 70 microseconds
        }
    
    def _default_hardware_constraints(self) -> Dict[str, Any]:
        """Default hardware constraints."""
        return {
            'max_qubits': 127,
            'coupling_map': 'heavy_hex',
            'single_qubit_gate_time': 35e-9,   # 35 nanoseconds
            'two_qubit_gate_time': 300e-9,     # 300 nanoseconds
            'measurement_time': 1e-6           # 1 microsecond
        }
    
    def optimize_qubo_circuit(self, 
                             qubo_matrix: np.ndarray,
                             num_layers: int = 4,
                             optimization_level: int = 3) -> OptimizationResult:
        """Optimize quantum circuit for QUBO problem.
        
        Args:
            qubo_matrix: QUBO problem matrix
            num_layers: Number of QAOA layers
            optimization_level: Optimization intensity (1-3)
            
        Returns:
            OptimizationResult with optimization metrics
        """
        start_time = time.time()
        
        # Generate original circuit metrics
        original_circuit = self._construct_qaoa_circuit(qubo_matrix, num_layers)
        original_metrics = self._analyze_circuit(original_circuit)
        
        # Apply optimization techniques based on level
        if optimization_level == 1:
            optimized_circuit = self._basic_optimization(original_circuit, qubo_matrix)
            technique = "basic_gate_commutation"
        elif optimization_level == 2:
            optimized_circuit = self._advanced_optimization(original_circuit, qubo_matrix)
            technique = "layer_fusion_and_symmetry"
        else:
            optimized_circuit = self._adaptive_optimization(original_circuit, qubo_matrix)
            technique = "adaptive_parameterized_depth_reduction"
        
        # Analyze optimized circuit
        optimized_metrics = self._analyze_circuit(optimized_circuit)
        
        # Calculate improvement
        improvement = original_metrics.depth / max(1, optimized_metrics.depth)
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_factor=improvement,
            optimization_time=optimization_time,
            technique_used=technique
        )
        
        self.optimization_history.append(result)
        
        logger.info(f"Circuit optimization complete: {improvement:.2f}x depth reduction in {optimization_time:.3f}s")
        return result
    
    def _construct_qaoa_circuit(self, qubo_matrix: np.ndarray, num_layers: int) -> Dict[str, Any]:
        """Construct QAOA circuit for QUBO problem."""
        n_qubits = qubo_matrix.shape[0]
        
        # Simplified circuit representation
        circuit = {
            'n_qubits': n_qubits,
            'layers': [],
            'parameters': []
        }
        
        # Initial superposition layer
        circuit['layers'].append({
            'type': 'hadamard_layer',
            'qubits': list(range(n_qubits)),
            'depth': 1
        })
        
        # QAOA layers
        for layer in range(num_layers):
            # Problem Hamiltonian layer
            problem_layer = self._create_problem_layer(qubo_matrix)
            circuit['layers'].append(problem_layer)
            
            # Mixer Hamiltonian layer  
            mixer_layer = self._create_mixer_layer(n_qubits)
            circuit['layers'].append(mixer_layer)
            
            circuit['parameters'].extend([f'gamma_{layer}', f'beta_{layer}'])
        
        return circuit
    
    def _create_problem_layer(self, qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Create problem Hamiltonian layer."""
        gates = []
        max_depth = 0
        
        # Add single qubit Z rotations
        for i in range(qubo_matrix.shape[0]):
            if qubo_matrix[i, i] != 0:
                gates.append({
                    'type': 'rz',
                    'qubit': i,
                    'parameter': f'2*gamma*{qubo_matrix[i,i]}',
                    'time_slot': 0
                })
        
        # Add two-qubit interactions
        time_slot = 1
        for i in range(qubo_matrix.shape[0]):
            for j in range(i + 1, qubo_matrix.shape[1]):
                if qubo_matrix[i, j] != 0:
                    gates.extend([
                        {'type': 'cnot', 'control': i, 'target': j, 'time_slot': time_slot},
                        {'type': 'rz', 'qubit': j, 'parameter': f'gamma*{qubo_matrix[i,j]}', 'time_slot': time_slot + 1},
                        {'type': 'cnot', 'control': i, 'target': j, 'time_slot': time_slot + 2}
                    ])
                    max_depth = max(max_depth, time_slot + 3)
                    time_slot += 3
        
        return {
            'type': 'problem_layer',
            'gates': gates,
            'depth': max_depth
        }
    
    def _create_mixer_layer(self, n_qubits: int) -> Dict[str, Any]:
        """Create mixer Hamiltonian layer."""
        gates = []
        for i in range(n_qubits):
            gates.append({
                'type': 'rx',
                'qubit': i,
                'parameter': f'2*beta',
                'time_slot': 0
            })
        
        return {
            'type': 'mixer_layer',
            'gates': gates,
            'depth': 1
        }
    
    def _basic_optimization(self, circuit: Dict[str, Any], qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Apply basic optimization techniques."""
        optimized_circuit = circuit.copy()
        
        # Gate commutation optimization
        for layer in optimized_circuit['layers']:
            if layer['type'] == 'problem_layer':
                layer['gates'] = self._commute_gates(layer['gates'])
                layer['depth'] = self._calculate_layer_depth(layer['gates'])
        
        return optimized_circuit
    
    def _advanced_optimization(self, circuit: Dict[str, Any], qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Apply advanced optimization techniques."""
        optimized_circuit = self._basic_optimization(circuit, qubo_matrix)
        
        # Layer fusion optimization
        optimized_circuit = self._fuse_layers(optimized_circuit)
        
        # Symmetry-based optimization
        optimized_circuit = self._exploit_symmetries(optimized_circuit, qubo_matrix)
        
        return optimized_circuit
    
    def _adaptive_optimization(self, circuit: Dict[str, Any], qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Apply adaptive optimization with machine learning insights."""
        optimized_circuit = self._advanced_optimization(circuit, qubo_matrix)
        
        # Adaptive parameterization
        optimized_circuit = self._adaptive_parameterization(optimized_circuit, qubo_matrix)
        
        # Hardware-aware optimization
        optimized_circuit = self._hardware_aware_optimization(optimized_circuit)
        
        return optimized_circuit
    
    def _commute_gates(self, gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply gate commutation rules to reduce depth."""
        # Group gates by qubits they act on
        qubit_gates = defaultdict(list)
        for gate in gates:
            if 'qubit' in gate:
                qubit_gates[gate['qubit']].append(gate)
            elif 'control' in gate and 'target' in gate:
                qubit_gates[gate['control']].append(gate)
                qubit_gates[gate['target']].append(gate)
        
        # Optimize time slots for parallel execution
        optimized_gates = []
        current_time = 0
        
        while any(qubit_gates.values()):
            # Find gates that can execute in parallel
            parallel_gates = []
            used_qubits = set()
            
            for qubit, gate_list in qubit_gates.items():
                if not gate_list:
                    continue
                    
                gate = gate_list[0]
                gate_qubits = self._get_gate_qubits(gate)
                
                if not gate_qubits.intersection(used_qubits):
                    parallel_gates.append(gate)
                    used_qubits.update(gate_qubits)
                    gate_list.pop(0)
            
            # Assign time slots
            for gate in parallel_gates:
                gate['time_slot'] = current_time
                optimized_gates.append(gate)
            
            current_time += 1
        
        return optimized_gates
    
    def _get_gate_qubits(self, gate: Dict[str, Any]) -> set:
        """Get set of qubits that a gate acts on."""
        qubits = set()
        if 'qubit' in gate:
            qubits.add(gate['qubit'])
        if 'control' in gate:
            qubits.add(gate['control'])
        if 'target' in gate:
            qubits.add(gate['target'])
        return qubits
    
    def _calculate_layer_depth(self, gates: List[Dict[str, Any]]) -> int:
        """Calculate the depth of a layer after optimization."""
        if not gates:
            return 0
        return max(gate.get('time_slot', 0) for gate in gates) + 1
    
    def _fuse_layers(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse compatible layers to reduce circuit depth."""
        fused_circuit = {
            'n_qubits': circuit['n_qubits'],
            'layers': [],
            'parameters': circuit['parameters']
        }
        
        i = 0
        while i < len(circuit['layers']):
            current_layer = circuit['layers'][i]
            
            # Try to fuse with next layer if compatible
            if i + 1 < len(circuit['layers']):
                next_layer = circuit['layers'][i + 1]
                
                if self._can_fuse_layers(current_layer, next_layer):
                    fused_layer = self._merge_layers(current_layer, next_layer)
                    fused_circuit['layers'].append(fused_layer)
                    i += 2  # Skip next layer as it's been fused
                    continue
            
            fused_circuit['layers'].append(current_layer)
            i += 1
        
        return fused_circuit
    
    def _can_fuse_layers(self, layer1: Dict[str, Any], layer2: Dict[str, Any]) -> bool:
        """Check if two layers can be fused."""
        # Simple heuristic: can fuse if they don't have overlapping two-qubit gates
        if layer1['type'] == 'mixer_layer' and layer2['type'] == 'hadamard_layer':
            return True
        return False
    
    def _merge_layers(self, layer1: Dict[str, Any], layer2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two compatible layers."""
        merged = {
            'type': f"fused_{layer1['type']}_{layer2['type']}",
            'gates': layer1.get('gates', []) + layer2.get('gates', []),
            'qubits': layer1.get('qubits', []) + layer2.get('qubits', []),
            'depth': max(layer1.get('depth', 0), layer2.get('depth', 0))
        }
        return merged
    
    def _exploit_symmetries(self, circuit: Dict[str, Any], qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Exploit problem symmetries to reduce circuit complexity."""
        # Detect symmetries in QUBO matrix
        symmetries = self._detect_qubo_symmetries(qubo_matrix)
        
        if symmetries:
            logger.info(f"Found {len(symmetries)} symmetries in QUBO matrix")
            # Apply symmetry-based reductions (simplified implementation)
            for layer in circuit['layers']:
                if layer['type'] == 'problem_layer':
                    layer['gates'] = self._apply_symmetry_reduction(layer['gates'], symmetries)
        
        return circuit
    
    def _detect_qubo_symmetries(self, qubo_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Detect symmetries in QUBO matrix."""
        symmetries = []
        n = qubo_matrix.shape[0]
        
        # Check for variable symmetries (simplified)
        for i in range(n):
            for j in range(i + 1, n):
                if np.allclose(qubo_matrix[i, :], qubo_matrix[j, :]) and \
                   np.allclose(qubo_matrix[:, i], qubo_matrix[:, j]):
                    symmetries.append((i, j))
        
        return symmetries
    
    def _apply_symmetry_reduction(self, gates: List[Dict[str, Any]], 
                                 symmetries: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Apply symmetry reduction to gates."""
        # Simplified implementation: remove redundant gates for symmetric variables
        reduced_gates = []
        processed_pairs = set()
        
        for gate in gates:
            gate_qubits = self._get_gate_qubits(gate)
            
            # Check if this gate operates on symmetric qubits
            skip_gate = False
            for qubit in gate_qubits:
                for sym_pair in symmetries:
                    if qubit in sym_pair and sym_pair not in processed_pairs:
                        processed_pairs.add(sym_pair)
                        skip_gate = False
                        break
                    elif qubit in sym_pair and sym_pair in processed_pairs:
                        skip_gate = True
                        break
            
            if not skip_gate:
                reduced_gates.append(gate)
        
        return reduced_gates
    
    def _adaptive_parameterization(self, circuit: Dict[str, Any], 
                                  qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Apply adaptive parameterization based on problem structure."""
        # Analyze problem structure
        problem_analysis = self._analyze_qubo_structure(qubo_matrix)
        
        # Adjust parameters based on analysis
        if problem_analysis['sparsity'] > 0.7:
            # Sparse problem - reduce layer count
            circuit = self._reduce_layers(circuit, factor=0.8)
        elif problem_analysis['conditioning'] > 100:
            # Ill-conditioned problem - add regularization layers
            circuit = self._add_regularization(circuit)
        
        return circuit
    
    def _analyze_qubo_structure(self, qubo_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze QUBO matrix structure."""
        n = qubo_matrix.shape[0]
        nonzero_elements = np.count_nonzero(qubo_matrix)
        total_elements = n * n
        sparsity = 1.0 - (nonzero_elements / total_elements)
        
        # Condition number estimation
        eigenvals = np.linalg.eigvals(qubo_matrix + qubo_matrix.T)
        eigenvals = eigenvals[eigenvals > 1e-10]
        condition_number = np.max(eigenvals) / np.min(eigenvals) if len(eigenvals) > 0 else 1.0
        
        return {
            'sparsity': sparsity,
            'conditioning': condition_number,
            'nonzero_elements': nonzero_elements,
            'matrix_norm': np.linalg.norm(qubo_matrix)
        }
    
    def _reduce_layers(self, circuit: Dict[str, Any], factor: float) -> Dict[str, Any]:
        """Reduce number of layers for sparse problems."""
        reduced_circuit = circuit.copy()
        original_layer_count = len([l for l in circuit['layers'] 
                                   if l['type'] in ['problem_layer', 'mixer_layer']])
        new_layer_count = max(1, int(original_layer_count * factor))
        
        # Keep only essential layers
        essential_layers = []
        layer_count = 0
        
        for layer in circuit['layers']:
            if layer['type'] in ['problem_layer', 'mixer_layer']:
                if layer_count < new_layer_count:
                    essential_layers.append(layer)
                    layer_count += 1
            else:
                essential_layers.append(layer)
        
        reduced_circuit['layers'] = essential_layers
        logger.info(f"Reduced circuit layers from {original_layer_count} to {new_layer_count}")
        
        return reduced_circuit
    
    def _add_regularization(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Add regularization for ill-conditioned problems."""
        # Add small random rotations to break ill-conditioning
        regularization_layer = {
            'type': 'regularization_layer',
            'gates': [
                {'type': 'rz', 'qubit': i, 'parameter': f'0.01*random_{i}', 'time_slot': 0}
                for i in range(circuit['n_qubits'])
            ],
            'depth': 1
        }
        
        circuit['layers'].insert(-1, regularization_layer)
        logger.info("Added regularization layer for ill-conditioned problem")
        
        return circuit
    
    def _hardware_aware_optimization(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hardware-specific optimizations."""
        # Route gates based on hardware connectivity
        optimized_circuit = self._route_for_connectivity(circuit)
        
        # Optimize for gate error rates
        optimized_circuit = self._optimize_for_errors(optimized_circuit)
        
        return optimized_circuit
    
    def _route_for_connectivity(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Route circuit for hardware connectivity."""
        # Simplified routing - in practice would use SWAP insertion
        routed_circuit = circuit.copy()
        
        for layer in routed_circuit['layers']:
            if 'gates' in layer:
                layer['gates'] = self._insert_swaps_if_needed(layer['gates'])
        
        return routed_circuit
    
    def _insert_swaps_if_needed(self, gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert SWAP gates for connectivity (simplified)."""
        # This would implement actual SWAP routing
        # For now, just return original gates
        return gates
    
    def _optimize_for_errors(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize gate sequence for minimal error."""
        # Reorder gates to minimize error accumulation
        for layer in circuit['layers']:
            if 'gates' in layer:
                layer['gates'] = self._reorder_for_minimal_error(layer['gates'])
        
        return circuit
    
    def _reorder_for_minimal_error(self, gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder gates to minimize error accumulation."""
        # Sort by error rate (two-qubit gates last)
        single_qubit_gates = [g for g in gates if g['type'] in ['rx', 'ry', 'rz']]
        two_qubit_gates = [g for g in gates if g['type'] in ['cnot', 'cz']]
        
        return single_qubit_gates + two_qubit_gates
    
    def _analyze_circuit(self, circuit: Dict[str, Any]) -> CircuitMetrics:
        """Analyze circuit and compute metrics."""
        total_depth = 0
        total_gates = 0
        two_qubit_gates = 0
        
        for layer in circuit['layers']:
            layer_depth = layer.get('depth', 0)
            total_depth += layer_depth
            
            if 'gates' in layer:
                layer_gates = len(layer['gates'])
                total_gates += layer_gates
                two_qubit_gates += len([g for g in layer['gates'] 
                                      if g['type'] in ['cnot', 'cz']])
            elif 'qubits' in layer:
                total_gates += len(layer['qubits'])
        
        # Calculate parallelization factor
        sequential_time = total_gates * self.hardware_constraints['single_qubit_gate_time']
        parallel_time = total_depth * self.hardware_constraints['single_qubit_gate_time']
        parallelization_factor = sequential_time / max(parallel_time, 1e-9)
        
        # Estimate execution time
        execution_time = (
            total_depth * self.hardware_constraints['single_qubit_gate_time'] +
            two_qubit_gates * self.hardware_constraints['two_qubit_gate_time']
        )
        
        # Estimate fidelity
        fidelity = (
            (1 - self.noise_model['single_qubit_error']) ** (total_gates - two_qubit_gates) *
            (1 - self.noise_model['two_qubit_error']) ** two_qubit_gates
        )
        
        return CircuitMetrics(
            depth=total_depth,
            gate_count=total_gates,
            two_qubit_gates=two_qubit_gates,
            parallelization_factor=parallelization_factor,
            estimated_execution_time=execution_time,
            fidelity_estimate=fidelity
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics from optimization history."""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}
        
        improvements = [result.improvement_factor for result in self.optimization_history]
        times = [result.optimization_time for result in self.optimization_history]
        techniques = [result.technique_used for result in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": np.mean(improvements),
            "best_improvement": np.max(improvements),
            "average_optimization_time": np.mean(times),
            "techniques_used": list(set(techniques)),
            "total_depth_reduction": sum(
                result.original_metrics.depth - result.optimized_metrics.depth
                for result in self.optimization_history
            )
        }


class QuantumAdvantageAnalyzer:
    """Analyze when quantum circuits provide advantage over classical methods."""
    
    def __init__(self):
        self.analysis_cache = {}
        self.benchmarks = []
    
    def analyze_quantum_advantage(self, 
                                qubo_matrix: np.ndarray,
                                classical_time: float,
                                quantum_circuit_metrics: CircuitMetrics) -> Dict[str, Any]:
        """Analyze quantum advantage for a specific problem."""
        problem_size = qubo_matrix.shape[0]
        problem_hash = hash(qubo_matrix.tobytes())
        
        if problem_hash in self.analysis_cache:
            return self.analysis_cache[problem_hash]
        
        # Calculate quantum advantage metrics
        quantum_time = quantum_circuit_metrics.estimated_execution_time
        speedup_ratio = classical_time / quantum_time if quantum_time > 0 else 0
        
        # Quality metrics
        solution_quality_factor = quantum_circuit_metrics.fidelity_estimate
        
        # Resource efficiency
        resource_efficiency = quantum_circuit_metrics.parallelization_factor
        
        # Theoretical advantage threshold
        theoretical_threshold = self._calculate_theoretical_threshold(problem_size)
        
        analysis = {
            "problem_size": problem_size,
            "speedup_ratio": speedup_ratio,
            "quantum_advantage": speedup_ratio > 1.0,
            "classical_time": classical_time,
            "quantum_time": quantum_time,
            "solution_quality_factor": solution_quality_factor,
            "resource_efficiency": resource_efficiency,
            "theoretical_threshold": theoretical_threshold,
            "advantage_margin": speedup_ratio - theoretical_threshold,
            "recommendation": self._generate_recommendation(speedup_ratio, theoretical_threshold)
        }
        
        self.analysis_cache[problem_hash] = analysis
        self.benchmarks.append(analysis)
        
        return analysis
    
    def _calculate_theoretical_threshold(self, problem_size: int) -> float:
        """Calculate theoretical threshold for quantum advantage."""
        # Based on complexity analysis - quantum advantage typically starts
        # around problems with exponential classical scaling
        if problem_size < 20:
            return 10.0  # High threshold for small problems
        elif problem_size < 50:
            return 2.0   # Medium threshold
        else:
            return 1.2   # Lower threshold for large problems
    
    def _generate_recommendation(self, speedup_ratio: float, threshold: float) -> str:
        """Generate recommendation based on analysis."""
        if speedup_ratio > threshold:
            return f"Quantum advantage achieved (speedup: {speedup_ratio:.2f}x). Recommend quantum execution."
        elif speedup_ratio > 0.8 * threshold:
            return f"Near quantum advantage (speedup: {speedup_ratio:.2f}x). Consider quantum execution."
        else:
            return f"Classical advantage (speedup: {speedup_ratio:.2f}x). Recommend classical execution."
    
    def get_advantage_statistics(self) -> Dict[str, Any]:
        """Get statistics on quantum advantage across benchmarks."""
        if not self.benchmarks:
            return {"message": "No analyses performed yet"}
        
        quantum_advantageous = [b for b in self.benchmarks if b['quantum_advantage']]
        
        return {
            "total_analyses": len(self.benchmarks),
            "quantum_advantageous": len(quantum_advantageous),
            "quantum_advantage_rate": len(quantum_advantageous) / len(self.benchmarks),
            "average_speedup": np.mean([b['speedup_ratio'] for b in self.benchmarks]),
            "best_speedup": np.max([b['speedup_ratio'] for b in self.benchmarks]),
            "problem_size_threshold": self._find_size_threshold()
        }
    
    def _find_size_threshold(self) -> int:
        """Find problem size threshold where quantum advantage typically occurs."""
        advantageous = [b for b in self.benchmarks if b['quantum_advantage']]
        if not advantageous:
            return float('inf')
        
        return min(b['problem_size'] for b in advantageous)