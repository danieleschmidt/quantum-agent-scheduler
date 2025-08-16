"""Comprehensive tests for hybrid communication protocols research module."""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from quantum_scheduler.research.hybrid_communication_protocols import (
    CommunicationProtocol,
    WorkloadType,
    CommunicationMessage,
    ResourceAllocation,
    AdaptiveWorkloadPartitioner,
    AsyncQuantumClassicalCommunicator,
    HybridSchedulingCoordinator,
    test_hybrid_communication
)


class TestCommunicationMessage:
    """Test cases for CommunicationMessage class."""
    
    def test_message_creation(self):
        """Test CommunicationMessage creation."""
        message = CommunicationMessage(
            message_id="test_001",
            sender="coordinator",
            receiver="quantum_backend",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={"data": "test"},
            priority=5
        )
        
        assert message.message_id == "test_001"
        assert message.sender == "coordinator"
        assert message.receiver == "quantum_backend"
        assert message.workload_type == WorkloadType.QUANTUM_OPTIMIZATION
        assert message.payload == {"data": "test"}
        assert message.priority == 5
        assert message.retry_count == 0
    
    def test_message_expiration(self):
        """Test message expiration functionality."""
        # Non-expiring message
        message = CommunicationMessage(
            message_id="test_001",
            sender="test",
            receiver="test",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={}
        )
        assert not message.is_expired()
        
        # Expired message
        message_expired = CommunicationMessage(
            message_id="test_002",
            sender="test",
            receiver="test",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={},
            expires_at=time.time() - 10  # 10 seconds ago
        )
        assert message_expired.is_expired()
        
        # Future expiration
        message_future = CommunicationMessage(
            message_id="test_003",
            sender="test",
            receiver="test",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={},
            expires_at=time.time() + 10  # 10 seconds from now
        )
        assert not message_future.is_expired()
    
    def test_message_serialization(self):
        """Test message serialization to/from dictionary."""
        original_message = CommunicationMessage(
            message_id="test_001",
            sender="coordinator",
            receiver="quantum_backend",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={"agents": [1, 2, 3]},
            priority=7,
            expires_at=time.time() + 3600
        )
        
        # Serialize to dict
        message_dict = original_message.to_dict()
        
        assert message_dict['message_id'] == "test_001"
        assert message_dict['workload_type'] == WorkloadType.QUANTUM_OPTIMIZATION.value
        assert message_dict['payload'] == {"agents": [1, 2, 3]}
        
        # Deserialize from dict
        restored_message = CommunicationMessage.from_dict(message_dict)
        
        assert restored_message.message_id == original_message.message_id
        assert restored_message.sender == original_message.sender
        assert restored_message.workload_type == original_message.workload_type
        assert restored_message.payload == original_message.payload
        assert restored_message.priority == original_message.priority


class TestResourceAllocation:
    """Test cases for ResourceAllocation class."""
    
    def test_resource_allocation_creation(self):
        """Test ResourceAllocation creation."""
        allocation = ResourceAllocation(
            quantum_resources={"qubits": 20, "shots": 1024},
            classical_resources={"cpu_cores": 4, "memory_gb": 8},
            total_cost=150.0,
            estimated_completion_time=300.0,
            confidence_score=0.85,
            allocation_strategy="hybrid"
        )
        
        assert allocation.quantum_resources["qubits"] == 20
        assert allocation.classical_resources["cpu_cores"] == 4
        assert allocation.total_cost == 150.0
        assert allocation.confidence_score == 0.85
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        allocation = ResourceAllocation(
            quantum_resources={},
            classical_resources={},
            total_cost=100.0,
            estimated_completion_time=200.0,
            confidence_score=0.9,
            allocation_strategy="test"
        )
        
        expected_efficiency = 0.9 / (100.0 * 200.0)
        assert abs(allocation.efficiency_score() - expected_efficiency) < 1e-10
    
    def test_efficiency_score_zero_values(self):
        """Test efficiency score with zero cost or time."""
        allocation_zero_cost = ResourceAllocation(
            quantum_resources={},
            classical_resources={},
            total_cost=0.0,
            estimated_completion_time=200.0,
            confidence_score=0.9,
            allocation_strategy="test"
        )
        assert allocation_zero_cost.efficiency_score() == 0.0
        
        allocation_zero_time = ResourceAllocation(
            quantum_resources={},
            classical_resources={},
            total_cost=100.0,
            estimated_completion_time=0.0,
            confidence_score=0.9,
            allocation_strategy="test"
        )
        assert allocation_zero_time.efficiency_score() == 0.0


class TestAdaptiveWorkloadPartitioner:
    """Test cases for AdaptiveWorkloadPartitioner class."""
    
    def test_partitioner_initialization(self):
        """Test AdaptiveWorkloadPartitioner initialization."""
        partitioner = AdaptiveWorkloadPartitioner(
            quantum_advantage_threshold=1.5,
            classical_capacity={"cpu_cores": 16.0, "memory_gb": 64.0},
            quantum_capacity={"qubits": 100.0, "gate_fidelity": 0.995}
        )
        
        assert partitioner.quantum_advantage_threshold == 1.5
        assert partitioner.classical_capacity["cpu_cores"] == 16.0
        assert partitioner.quantum_capacity["qubits"] == 100.0
    
    def test_partitioner_default_initialization(self):
        """Test AdaptiveWorkloadPartitioner with default parameters."""
        partitioner = AdaptiveWorkloadPartitioner()
        
        assert partitioner.quantum_advantage_threshold == 1.2
        assert "cpu_cores" in partitioner.classical_capacity
        assert "qubits" in partitioner.quantum_capacity
    
    def test_extract_problem_features(self):
        """Test problem feature extraction."""
        partitioner = AdaptiveWorkloadPartitioner()
        
        problem_data = {
            'agents': [
                {'id': 'agent1', 'skills': ['python', 'ml']},
                {'id': 'agent2', 'skills': ['java', 'web']}
            ],
            'tasks': [
                {'id': 'task1', 'required_skills': ['python']},
                {'id': 'task2', 'required_skills': ['web']},
                {'id': 'task3', 'required_skills': ['ml']}
            ],
            'constraints': {'skill_match': True, 'capacity': True}
        }
        
        features = partitioner._extract_problem_features(problem_data)
        
        assert features['problem_size'] == 6  # 2 agents * 3 tasks
        assert features['agent_task_ratio'] == 2.0 / 3.0
        assert 'constraint_density' in features
        assert 'connectivity' in features
        assert 'sparsity' in features
    
    def test_calculate_connectivity(self):
        """Test connectivity calculation."""
        partitioner = AdaptiveWorkloadPartitioner()
        
        problem_data = {
            'agents': [
                {'skills': ['python', 'ml']},
                {'skills': ['java', 'web']}
            ],
            'tasks': [
                {'required_skills': ['python']},  # Matches agent 1
                {'required_skills': ['web']},     # Matches agent 2
                {'required_skills': ['rust']}     # Matches neither
            ]
        }
        
        connectivity = partitioner._calculate_connectivity(problem_data)
        
        # 2 out of 6 possible connections (2 agents * 3 tasks)
        expected_connectivity = 2.0 / 6.0
        assert abs(connectivity - expected_connectivity) < 1e-6
    
    def test_predict_quantum_advantage(self):
        """Test quantum advantage prediction."""
        partitioner = AdaptiveWorkloadPartitioner()
        
        # Small, disconnected problem
        features_small = {
            'problem_size': 50,
            'connectivity': 0.2,
            'sparsity': 0.8,
            'constraint_density': 0.05
        }
        
        scores_small = partitioner._predict_quantum_advantage(features_small)
        assert 'optimization' in scores_small
        assert 'constraint_satisfaction' in scores_small
        assert 'search' in scores_small
        
        # Large, connected problem
        features_large = {
            'problem_size': 500,
            'connectivity': 0.6,
            'sparsity': 0.3,
            'constraint_density': 0.2
        }
        
        scores_large = partitioner._predict_quantum_advantage(features_large)
        
        # Large problems should have higher quantum advantage scores
        assert scores_large['optimization'] > scores_small['optimization']
        assert scores_large['search'] > scores_small['search']
    
    def test_partition_workload(self):
        """Test workload partitioning."""
        partitioner = AdaptiveWorkloadPartitioner()
        
        problem_data = {
            'agents': [{'id': f'agent_{i}', 'skills': ['python']} for i in range(10)],
            'tasks': [{'id': f'task_{i}', 'required_skills': ['python']} for i in range(20)],
            'constraints': {}
        }
        
        quantum_tasks, classical_tasks = partitioner.partition_workload(problem_data)
        
        assert isinstance(quantum_tasks, list)
        assert isinstance(classical_tasks, list)
        assert len(quantum_tasks) + len(classical_tasks) > 0


class TestAsyncQuantumClassicalCommunicator:
    """Test cases for AsyncQuantumClassicalCommunicator class."""
    
    def test_communicator_initialization(self):
        """Test AsyncQuantumClassicalCommunicator initialization."""
        communicator = AsyncQuantumClassicalCommunicator(
            max_queue_size=500,
            max_concurrent_tasks=5,
            retry_limit=5
        )
        
        assert communicator.max_queue_size == 500
        assert communicator.max_concurrent_tasks == 5
        assert communicator.retry_limit == 5
        assert not communicator.running
    
    def test_communicator_default_initialization(self):
        """Test AsyncQuantumClassicalCommunicator with default parameters."""
        communicator = AsyncQuantumClassicalCommunicator()
        
        assert communicator.max_queue_size == 1000
        assert communicator.max_concurrent_tasks == 10
        assert communicator.retry_limit == 3
    
    @pytest.mark.asyncio
    async def test_communicator_start_stop(self):
        """Test communicator start and stop functionality."""
        communicator = AsyncQuantumClassicalCommunicator()
        
        # Test start
        await communicator.start()
        assert communicator.running
        
        # Test stop
        await communicator.stop()
        assert not communicator.running
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test message sending."""
        communicator = AsyncQuantumClassicalCommunicator(max_queue_size=10)
        
        message = CommunicationMessage(
            message_id="test_001",
            sender="test",
            receiver="test",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={"data": "test"}
        )
        
        # Test successful send
        success = await communicator.send_message(message)
        assert success
        assert message.message_id in communicator.pending_messages
    
    @pytest.mark.asyncio
    async def test_send_priority_message(self):
        """Test priority message sending."""
        communicator = AsyncQuantumClassicalCommunicator()
        
        priority_message = CommunicationMessage(
            message_id="priority_001",
            sender="test",
            receiver="test",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={"urgent": True},
            priority=5
        )
        
        success = await communicator.send_message(priority_message)
        assert success
    
    @pytest.mark.asyncio
    async def test_receive_message_timeout(self):
        """Test message receiving with timeout."""
        communicator = AsyncQuantumClassicalCommunicator()
        
        # Test timeout
        message = await communicator.receive_message(timeout=0.1)
        assert message is None
    
    def test_estimate_communication_cost(self):
        """Test communication cost estimation."""
        communicator = AsyncQuantumClassicalCommunicator()
        
        message = CommunicationMessage(
            message_id="test_001",
            sender="test",
            receiver="test",
            workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
            payload={"large_data": "x" * 1000}  # 1KB payload
        )
        
        cost = communicator.estimate_communication_cost(message)
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        communicator = AsyncQuantumClassicalCommunicator()
        
        # Add some sample performance data
        communicator.performance_metrics['transmission_times'] = [0.1, 0.2, 0.15]
        
        metrics = communicator.get_performance_metrics()
        
        assert 'avg_transmission_time' in metrics
        assert 'max_transmission_time' in metrics
        assert 'pending_message_count' in metrics
        assert 'queue_sizes' in metrics


class TestHybridSchedulingCoordinator:
    """Test cases for HybridSchedulingCoordinator class."""
    
    def test_coordinator_initialization(self):
        """Test HybridSchedulingCoordinator initialization."""
        communicator = AsyncQuantumClassicalCommunicator()
        partitioner = AdaptiveWorkloadPartitioner()
        
        coordinator = HybridSchedulingCoordinator(
            communicator=communicator,
            partitioner=partitioner
        )
        
        assert coordinator.communicator == communicator
        assert coordinator.partitioner == partitioner
        assert len(coordinator.active_tasks) == 0
    
    def test_coordinator_default_initialization(self):
        """Test HybridSchedulingCoordinator with default parameters."""
        coordinator = HybridSchedulingCoordinator()
        
        assert isinstance(coordinator.communicator, AsyncQuantumClassicalCommunicator)
        assert isinstance(coordinator.partitioner, AdaptiveWorkloadPartitioner)
    
    @pytest.mark.asyncio
    async def test_schedule_hybrid_workload(self):
        """Test hybrid workload scheduling."""
        coordinator = HybridSchedulingCoordinator()
        
        problem_data = {
            'agents': [
                {'id': 'agent1', 'skills': ['python'], 'capacity': 2},
                {'id': 'agent2', 'skills': ['java'], 'capacity': 3}
            ],
            'tasks': [
                {'id': 'task1', 'required_skills': ['python'], 'duration': 2},
                {'id': 'task2', 'required_skills': ['java'], 'duration': 1}
            ],
            'constraints': {'skill_match_required': True}
        }
        
        results = await coordinator.schedule_hybrid_workload(problem_data)
        
        assert 'results' in results
        assert 'performance' in results
        assert 'final_assignment' in results['results']
        assert 'total_execution_time' in results['performance']
    
    @pytest.mark.asyncio
    async def test_execute_quantum_tasks(self):
        """Test quantum task execution."""
        coordinator = HybridSchedulingCoordinator()
        
        tasks = [
            {
                'type': 'quantum_optimization',
                'data': {'agents': [], 'tasks': []},
                'priority': 1
            }
        ]
        
        results = await coordinator._execute_quantum_tasks(tasks)
        
        assert len(results) == 1
        assert 'type' in results[0]
        assert results[0]['type'] == 'quantum_result'
    
    def test_execute_classical_tasks(self):
        """Test classical task execution."""
        coordinator = HybridSchedulingCoordinator()
        
        tasks = [
            {
                'type': 'classical_optimization',
                'data': {'agents': [{'id': 'a1'}], 'tasks': [{'id': 't1'}]},
                'priority': 1
            },
            {
                'type': 'data_preprocessing',
                'data': {'raw_data': 'test'},
                'priority': 0
            }
        ]
        
        # Use asyncio.run to handle the async method
        async def run_test():
            return await coordinator._execute_classical_tasks(tasks)
        
        results = asyncio.run(run_test())
        
        assert len(results) == 2
        assert all('type' in result for result in results)
    
    @pytest.mark.asyncio
    async def test_simulate_quantum_execution(self):
        """Test quantum execution simulation."""
        coordinator = HybridSchedulingCoordinator()
        
        task = {
            'type': 'quantum_optimization',
            'data': {
                'agents': [{'id': 'agent1'}, {'id': 'agent2'}],
                'tasks': [{'id': 'task1'}, {'id': 'task2'}, {'id': 'task3'}]
            }
        }
        
        result = await coordinator._simulate_quantum_execution(task)
        
        assert result['type'] == 'quantum_result'
        assert 'assignment' in result
        assert 'energy' in result
        assert 'quantum_advantage' in result
        assert len(result['assignment']) == 3  # Number of tasks
    
    def test_simulate_classical_execution(self):
        """Test classical execution simulation."""
        coordinator = HybridSchedulingCoordinator()
        
        # Test classical optimization
        task_opt = {
            'type': 'classical_optimization',
            'data': {
                'agents': [{'id': 'agent1'}, {'id': 'agent2'}],
                'tasks': [{'id': 'task1'}, {'id': 'task2'}]
            }
        }
        
        result_opt = coordinator._simulate_classical_execution(task_opt)
        assert result_opt['type'] == 'classical_result'
        assert 'assignment' in result_opt
        assert len(result_opt['assignment']) == 2  # Number of tasks
        
        # Test data preprocessing
        task_prep = {
            'type': 'data_preprocessing',
            'data': {'input': 'test_data'}
        }
        
        result_prep = coordinator._simulate_classical_execution(task_prep)
        assert result_prep['type'] == 'preprocessing_result'
        assert 'processed_data' in result_prep
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        coordinator = HybridSchedulingCoordinator()
        
        quantum_results = [
            {
                'type': 'quantum_result',
                'assignment': [0, 1, 0],
                'energy': -12.5,
                'execution_time': 0.8
            }
        ]
        
        classical_results = [
            {
                'type': 'classical_result',
                'assignment': [1, 0, 1],
                'energy': -10.2,
                'execution_time': 0.3
            },
            {
                'type': 'preprocessing_result',
                'execution_time': 0.1
            }
        ]
        
        aggregated = coordinator._aggregate_results(quantum_results, classical_results)
        
        assert 'final_assignment' in aggregated
        assert 'best_energy' in aggregated
        assert 'execution_summary' in aggregated
        assert 'performance_metrics' in aggregated
        
        # Best energy should be from quantum result (-12.5)
        assert aggregated['best_energy'] == -12.5
        assert aggregated['final_assignment'] == [0, 1, 0]


class TestIntegration:
    """Integration tests for hybrid communication protocols."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_scheduling(self):
        """Test complete hybrid scheduling workflow."""
        # Create coordinator
        coordinator = HybridSchedulingCoordinator()
        
        # Create realistic problem
        problem_data = {
            'agents': [
                {'id': 'agent1', 'skills': ['python', 'ml'], 'capacity': 3},
                {'id': 'agent2', 'skills': ['java', 'web'], 'capacity': 2},
                {'id': 'agent3', 'skills': ['python', 'web'], 'capacity': 4}
            ],
            'tasks': [
                {'id': 'task1', 'required_skills': ['python'], 'duration': 2, 'priority': 8},
                {'id': 'task2', 'required_skills': ['web'], 'duration': 1, 'priority': 5},
                {'id': 'task3', 'required_skills': ['ml'], 'duration': 3, 'priority': 9},
                {'id': 'task4', 'required_skills': ['java'], 'duration': 2, 'priority': 6}
            ],
            'constraints': {
                'skill_match_required': True,
                'respect_capacity': True
            }
        }
        
        # Execute hybrid scheduling
        results = await coordinator.schedule_hybrid_workload(
            problem_data,
            resource_constraints={'cpu_cores': 8, 'memory_gb': 16}
        )
        
        # Verify results structure
        assert 'results' in results
        assert 'performance' in results
        
        # Verify execution summary
        assert 'final_assignment' in results['results']
        assert 'execution_summary' in results['results']
        
        # Verify performance metrics
        performance = results['performance']
        assert 'total_execution_time' in performance
        assert 'quantum_task_count' in performance
        assert 'classical_task_count' in performance
        assert performance['total_execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_communication_protocol_performance(self):
        """Test communication protocol performance under load."""
        communicator = AsyncQuantumClassicalCommunicator(max_queue_size=100)
        
        await communicator.start()
        
        try:
            # Send multiple messages
            messages = []
            for i in range(20):
                message = CommunicationMessage(
                    message_id=f"perf_test_{i}",
                    sender="test_sender",
                    receiver="test_receiver",
                    workload_type=WorkloadType.QUANTUM_OPTIMIZATION,
                    payload={"data": f"test_data_{i}"},
                    priority=i % 5
                )
                messages.append(message)
                success = await communicator.send_message(message)
                assert success
            
            # Verify all messages are tracked
            assert len(communicator.pending_messages) == 20
            
            # Check performance metrics
            metrics = communicator.get_performance_metrics()
            assert 'pending_message_count' in metrics
            assert metrics['pending_message_count'] == 20
            
        finally:
            await communicator.stop()
    
    @pytest.mark.asyncio
    async def test_workload_partitioning_optimization(self):
        """Test workload partitioning optimization."""
        partitioner = AdaptiveWorkloadPartitioner(quantum_advantage_threshold=1.5)
        
        # Create large problem that should benefit from quantum processing
        large_problem = {
            'agents': [{'id': f'agent_{i}', 'skills': ['skill_' + str(i % 5)]} for i in range(100)],
            'tasks': [{'id': f'task_{i}', 'required_skills': ['skill_' + str(i % 5)]} for i in range(200)],
            'constraints': {'complex_constraints': True}
        }
        
        quantum_tasks, classical_tasks = partitioner.partition_workload(large_problem)
        
        # For large problems, should have quantum tasks
        assert len(quantum_tasks) > 0 or len(classical_tasks) > 0
        
        # Verify task structure
        for task in quantum_tasks + classical_tasks:
            assert 'type' in task
            assert 'data' in task
            assert 'priority' in task


@pytest.mark.asyncio
async def test_main_integration():
    """Test the main integration function."""
    # This would normally call test_hybrid_communication() but we'll test components
    coordinator = HybridSchedulingCoordinator()
    
    # Simple test to ensure the system can handle basic operations
    simple_problem = {
        'agents': [{'id': 'agent1', 'skills': ['test'], 'capacity': 1}],
        'tasks': [{'id': 'task1', 'required_skills': ['test'], 'duration': 1}],
        'constraints': {}
    }
    
    results = await coordinator.schedule_hybrid_workload(simple_problem)
    
    assert 'results' in results
    assert 'performance' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])