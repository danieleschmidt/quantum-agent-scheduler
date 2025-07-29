"""Integration tests for quantum backends."""

import pytest
from unittest.mock import Mock, patch

from quantum_scheduler import QuantumScheduler


class TestQuantumBackendIntegration:
    """Test quantum backend integrations."""

    @pytest.mark.quantum
    def test_qiskit_backend_integration(self, sample_agents, sample_tasks):
        """Test integration with Qiskit backend."""
        pytest.importorskip("qiskit")
        
        scheduler = QuantumScheduler(backend="qiskit")
        
        solution = scheduler.schedule(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={"skill_match_required": True}
        )
        
        assert solution is not None
        assert solution.solver_type in ["quantum_sim", "quantum_hw"]

    @pytest.mark.quantum
    def test_braket_backend_integration(self, sample_agents, sample_tasks):
        """Test integration with AWS Braket backend."""
        pytest.importorskip("braket")
        
        with patch('boto3.client') as mock_client:
            mock_client.return_value = Mock()
            
            scheduler = QuantumScheduler(backend="braket")
            
            solution = scheduler.schedule(
                agents=sample_agents,
                tasks=sample_tasks,
                constraints={"skill_match_required": True}
            )
            
            assert solution is not None

    @pytest.mark.quantum
    def test_dwave_backend_integration(self, sample_agents, sample_tasks):
        """Test integration with D-Wave backend."""
        pytest.importorskip("dwave")
        
        with patch('dwave.system.DWaveSampler') as mock_sampler:
            mock_sampler.return_value = Mock()
            
            scheduler = QuantumScheduler(backend="dwave")
            
            solution = scheduler.schedule(
                agents=sample_agents,
                tasks=sample_tasks,
                constraints={"skill_match_required": True}
            )
            
            assert solution is not None

    def test_automatic_backend_selection(self, sample_agents, sample_tasks):
        """Test automatic backend selection based on problem size."""
        scheduler = QuantumScheduler(backend="auto")
        
        # Small problem should use classical
        small_solution = scheduler.schedule(
            agents=sample_agents[:1],
            tasks=sample_tasks[:1],
            constraints={}
        )
        
        assert small_solution.solver_type == "classical"

    def test_fallback_to_classical(self, sample_agents, sample_tasks):
        """Test fallback to classical when quantum fails."""
        scheduler = QuantumScheduler(
            backend="quantum", 
            fallback="classical"
        )
        
        with patch.object(scheduler, '_quantum_solve') as mock_quantum:
            mock_quantum.side_effect = Exception("Quantum backend unavailable")
            
            solution = scheduler.schedule(
                agents=sample_agents,
                tasks=sample_tasks,
                constraints={}
            )
            
            assert solution is not None
            assert solution.solver_type == "classical"

    @pytest.mark.slow
    def test_performance_comparison(self, sample_agents, sample_tasks):
        """Test performance comparison between classical and quantum."""
        classical_scheduler = QuantumScheduler(backend="classical")
        quantum_scheduler = QuantumScheduler(backend="quantum_sim")
        
        import time
        
        # Classical timing
        start_time = time.time()
        classical_solution = classical_scheduler.schedule(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={}
        )
        classical_time = time.time() - start_time
        
        # Quantum simulation timing
        start_time = time.time()
        quantum_solution = quantum_scheduler.schedule(
            agents=sample_agents,
            tasks=sample_tasks,
            constraints={}
        )
        quantum_time = time.time() - start_time
        
        # Both should produce valid solutions
        assert classical_solution is not None
        assert quantum_solution is not None
        
        # Record performance metrics (would be sent to monitoring in real system)
        performance_metrics = {
            "classical_time": classical_time,
            "quantum_time": quantum_time,
            "classical_cost": classical_solution.cost,
            "quantum_cost": quantum_solution.cost,
        }
        
        assert performance_metrics["classical_time"] > 0
        assert performance_metrics["quantum_time"] > 0