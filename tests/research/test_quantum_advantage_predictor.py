"""Test suite for quantum advantage predictor research module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from quantum_scheduler.research.quantum_advantage_predictor import (
    QuantumAdvantagePredictor,
    ProblemFeatures,
    QuantumAdvantageRecord,
    QuantumAdvantageExperiment
)


class TestProblemFeatures:
    """Test problem features extraction and conversion."""
    
    def test_feature_vector_conversion(self):
        """Test conversion to feature vector."""
        features = ProblemFeatures(
            problem_size=50,
            density=0.3,
            sparsity=0.7,
            constraint_ratio=0.1,
            agent_capacity_variance=2.5,
            task_priority_spread=3.2,
            skill_overlap_ratio=0.4,
            graph_connectivity=0.6,
            matrix_condition_number=100.0,
            eigenvalue_spread=1.5
        )
        
        vector = features.to_vector()
        
        assert len(vector) == 10
        assert vector[0] == 50  # problem_size
        assert vector[1] == 0.3  # density
        assert vector[2] == 0.7  # sparsity
        assert vector[9] == 1.5  # eigenvalue_spread


class TestQuantumAdvantageRecord:
    """Test quantum advantage record functionality."""
    
    def test_normalized_advantage_calculation(self):
        """Test normalized advantage calculation."""
        features = ProblemFeatures(
            problem_size=10, density=0.5, sparsity=0.5, constraint_ratio=0.1,
            agent_capacity_variance=1.0, task_priority_spread=2.0,
            skill_overlap_ratio=0.3, graph_connectivity=0.4,
            matrix_condition_number=50.0, eigenvalue_spread=1.0
        )
        
        record = QuantumAdvantageRecord(
            features=features,
            classical_time=10.0,
            quantum_time=2.0,
            speedup_ratio=5.0,
            advantage_observed=True,
            solution_quality_classical=0.9,
            solution_quality_quantum=0.85,
            timestamp=1234567890,
            backend_used="quantum_sim"
        )
        
        # Raw speedup is 5.0, quality penalty is 0.05, so normalized = 5.0 * 0.95 = 4.75
        normalized = record.normalized_advantage
        assert normalized == pytest.approx(4.75, rel=1e-3)
    
    def test_normalized_advantage_no_quantum_time(self):
        """Test normalized advantage when quantum time is zero."""
        features = ProblemFeatures(
            problem_size=10, density=0.5, sparsity=0.5, constraint_ratio=0.1,
            agent_capacity_variance=1.0, task_priority_spread=2.0,
            skill_overlap_ratio=0.3, graph_connectivity=0.4,
            matrix_condition_number=50.0, eigenvalue_spread=1.0
        )
        
        record = QuantumAdvantageRecord(
            features=features,
            classical_time=10.0,
            quantum_time=0.0,
            speedup_ratio=0.0,
            advantage_observed=False,
            solution_quality_classical=0.9,
            solution_quality_quantum=0.8,
            timestamp=1234567890,
            backend_used="quantum_sim"
        )
        
        assert record.normalized_advantage == 0.0


class TestQuantumAdvantagePredictor:
    """Test quantum advantage predictor functionality."""
    
    @pytest.fixture
    def temp_model_path(self):
        """Create temporary directory for model storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def predictor(self, temp_model_path):
        """Create predictor instance with temporary model path."""
        return QuantumAdvantagePredictor(
            model_cache_path=temp_model_path,
            min_training_samples=5  # Low threshold for testing
        )
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        agents = []
        for i in range(3):
            agent = Mock()
            agent.capacity = i + 1
            agent.skills = ['python', 'ml'] if i == 0 else ['java', 'web']
            agents.append(agent)
        return agents
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        tasks = []
        for i in range(5):
            task = Mock()
            task.priority = float(i + 1)
            task.required_skills = ['python'] if i < 2 else ['java']
            tasks.append(task)
        return tasks
    
    def test_feature_extraction(self, predictor, sample_agents, sample_tasks):
        """Test feature extraction from problem data."""
        qubo_matrix = np.random.rand(8, 8)
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2  # Make symmetric
        
        features = predictor.extract_features(sample_agents, sample_tasks, qubo_matrix)
        
        assert isinstance(features, ProblemFeatures)
        assert features.problem_size == 8
        assert 0 <= features.density <= 1
        assert 0 <= features.sparsity <= 1
        assert features.density + features.sparsity == pytest.approx(1.0, rel=1e-6)
    
    def test_feature_extraction_empty_agents_tasks(self, predictor):
        """Test feature extraction with empty agents and tasks."""
        qubo_matrix = np.eye(5)
        
        features = predictor.extract_features([], [], qubo_matrix)
        
        assert features.problem_size == 5
        assert features.agent_capacity_variance == 0.0
        assert features.task_priority_spread == 0.0
        assert features.skill_overlap_ratio == 0.0
    
    def test_record_advantage_observation(self, predictor, sample_agents, sample_tasks):
        """Test recording advantage observations."""
        qubo_matrix = np.random.rand(5, 5)
        features = predictor.extract_features(sample_agents, sample_tasks, qubo_matrix)
        
        initial_count = len(predictor.training_records)
        
        predictor.record_advantage_observation(
            features=features,
            classical_time=5.0,
            quantum_time=1.0,
            classical_quality=0.9,
            quantum_quality=0.85,
            backend_used="test_backend"
        )
        
        assert len(predictor.training_records) == initial_count + 1
        
        record = predictor.training_records[-1]
        assert record.speedup_ratio == 5.0
        assert record.advantage_observed is True  # 5x speedup with 95% quality retention
        assert record.backend_used == "test_backend"
    
    def test_predict_quantum_advantage_no_models(self, predictor):
        """Test prediction when no models are trained."""
        features = ProblemFeatures(
            problem_size=10, density=0.5, sparsity=0.5, constraint_ratio=0.1,
            agent_capacity_variance=1.0, task_priority_spread=2.0,
            skill_overlap_ratio=0.3, graph_connectivity=0.4,
            matrix_condition_number=50.0, eigenvalue_spread=1.0
        )
        
        prediction = predictor.predict_quantum_advantage(features)
        
        assert prediction['advantage_probability'] == 0.5
        assert prediction['expected_speedup'] == 1.0
        assert prediction['confidence'] == 0.0
        assert prediction['recommendation'] == 'classical'
        assert prediction['reason'] == 'insufficient_training_data'
    
    def test_model_training(self, predictor, sample_agents, sample_tasks):
        """Test model training with sufficient data."""
        qubo_matrix = np.random.rand(5, 5)
        features = predictor.extract_features(sample_agents, sample_tasks, qubo_matrix)
        
        # Add training data
        for i in range(10):  # More than min_training_samples
            classical_time = 5.0 + np.random.normal(0, 0.5)
            quantum_time = 2.0 + np.random.normal(0, 0.3)
            
            predictor.record_advantage_observation(
                features=features,
                classical_time=classical_time,
                quantum_time=quantum_time,
                classical_quality=0.9,
                quantum_quality=0.85
            )
        
        # Models should be trained
        assert predictor.advantage_classifier is not None
        assert predictor.speedup_regressor is not None
        assert predictor.model_metrics['training_samples'] >= 10
    
    @patch('pickle.dump')
    @patch('pickle.load')
    def test_model_persistence(self, mock_load, mock_dump, predictor):
        """Test model saving and loading."""
        # Test saving
        predictor.advantage_classifier = Mock()
        predictor.speedup_regressor = Mock()
        
        predictor._save_models()
        
        # Verify save was called
        assert mock_dump.called
        
        # Test loading
        mock_load.return_value = {
            'advantage_classifier': Mock(),
            'speedup_regressor': Mock(),
            'feature_scaler': Mock(),
            'training_records': [],
            'model_metrics': {}
        }
        
        predictor._load_models()
        
        # Verify load was called
        assert mock_load.called
    
    def test_feature_importance(self, predictor):
        """Test feature importance extraction."""
        # Create mock models with feature importance
        mock_classifier = Mock()
        mock_classifier.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05])
        
        mock_regressor = Mock()
        mock_regressor.feature_importances_ = np.array([0.15, 0.1, 0.2, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05])
        
        predictor.advantage_classifier = mock_classifier
        predictor.speedup_regressor = mock_regressor
        
        importance = predictor.get_feature_importance()
        
        assert len(importance) == 10
        assert 'problem_size' in importance
        assert 'density' in importance
        assert all(0 <= score <= 1 for score in importance.values())
    
    def test_insights_report_insufficient_data(self, predictor):
        """Test insights generation with insufficient data."""
        report = predictor.generate_insights_report()
        
        assert "Insufficient data" in report
    
    def test_insights_report_with_data(self, predictor, sample_agents, sample_tasks):
        """Test insights generation with sufficient data."""
        qubo_matrix = np.random.rand(5, 5)
        features = predictor.extract_features(sample_agents, sample_tasks, qubo_matrix)
        
        # Add diverse training data
        for i in range(20):
            advantage_observed = i % 2 == 0  # 50% advantage rate
            classical_time = 5.0
            quantum_time = 2.0 if advantage_observed else 6.0
            
            predictor.record_advantage_observation(
                features=features,
                classical_time=classical_time,
                quantum_time=quantum_time,
                classical_quality=0.9,
                quantum_quality=0.85
            )
        
        # Mock models for feature importance
        mock_classifier = Mock()
        mock_classifier.feature_importances_ = np.random.rand(10)
        mock_regressor = Mock()
        mock_regressor.feature_importances_ = np.random.rand(10)
        
        predictor.advantage_classifier = mock_classifier
        predictor.speedup_regressor = mock_regressor
        
        report = predictor.generate_insights_report()
        
        assert "Quantum Advantage Analysis Report" in report
        assert "50.0%" in report  # Advantage rate
        assert "Most Predictive Features" in report
        assert "Recommendations" in report


class TestQuantumAdvantageExperiment:
    """Test quantum advantage experiment framework."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor for experiments."""
        return QuantumAdvantagePredictor(min_training_samples=5)
    
    @pytest.fixture
    def experiment(self, predictor):
        """Create experiment instance."""
        return QuantumAdvantageExperiment(predictor)
    
    def test_experiment_initialization(self, experiment, predictor):
        """Test experiment initialization."""
        assert experiment.predictor is predictor
        assert experiment.experiment_results == []
    
    def test_simulate_backend_execution(self, experiment):
        """Test backend execution simulation."""
        problem = {
            'qubo_matrix': np.random.rand(10, 10)
        }
        
        # Test classical backend
        exec_time, quality = experiment._simulate_backend_execution(problem, 'classical')
        assert exec_time > 0
        assert 0 <= quality <= 1
        
        # Test quantum simulator
        exec_time, quality = experiment._simulate_backend_execution(problem, 'quantum_sim')
        assert exec_time > 0
        assert 0 <= quality <= 1
        
        # Test quantum hardware
        exec_time, quality = experiment._simulate_backend_execution(problem, 'quantum_hw')
        assert exec_time > 0
        assert 0 <= quality <= 1
    
    @pytest.mark.asyncio
    async def test_comparative_study(self, experiment):
        """Test running comparative study."""
        def problem_generator(size):
            return {
                'qubo_matrix': np.random.rand(size, size),
                'agents': [{'id': f'agent_{i}', 'capacity': 1} for i in range(size//4 or 1)],
                'tasks': [{'id': f'task_{i}', 'priority': 1.0} for i in range(size//2 or 1)]
            }
        
        results = await experiment.run_comparative_study(
            problem_generator=problem_generator,
            problem_sizes=[5, 10],
            num_problems_per_size=2,
            backends=['classical', 'quantum_sim']
        )
        
        assert 'problem_sizes' in results
        assert 'backends' in results
        assert 'experiments' in results
        assert 'summary_statistics' in results
        
        assert results['problem_sizes'] == [5, 10]
        assert results['backends'] == ['classical', 'quantum_sim']
        assert len(results['experiments']) == 4  # 2 sizes × 2 problems each
    
    def test_compute_study_statistics(self, experiment):
        """Test computation of study statistics."""
        # Create mock results
        mock_results = {
            'problem_sizes': [10, 20],
            'backends': ['classical', 'quantum_sim'],
            'experiments': [
                {
                    'problem_size': 10,
                    'backend_results': {
                        'classical': {'execution_time': 1.0, 'solution_quality': 0.9},
                        'quantum_sim': {'execution_time': 0.5, 'solution_quality': 0.85}
                    }
                },
                {
                    'problem_size': 10,
                    'backend_results': {
                        'classical': {'execution_time': 1.2, 'solution_quality': 0.88},
                        'quantum_sim': {'execution_time': 0.6, 'solution_quality': 0.87}
                    }
                }
            ]
        }
        
        stats = experiment._compute_study_statistics(mock_results)
        
        assert 'size_10' in stats
        size_10_stats = stats['size_10']
        
        assert 'classical' in size_10_stats
        assert 'quantum_sim' in size_10_stats
        assert 'quantum_speedup' in size_10_stats
        assert 'quantum_advantage' in size_10_stats
        
        # Quantum should show speedup (1.1 / 0.55 ≈ 2x)
        assert size_10_stats['quantum_speedup'] > 1.0
        assert size_10_stats['quantum_advantage'] is True


@pytest.mark.slow
class TestQuantumAdvantageIntegration:
    """Integration tests for quantum advantage prediction."""
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete prediction workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = QuantumAdvantagePredictor(
                model_cache_path=Path(temp_dir),
                min_training_samples=5
            )
            
            # Generate training data
            for i in range(10):
                qubo_matrix = np.random.rand(10, 10)
                agents = [Mock(capacity=1, skills=['python']) for _ in range(3)]
                tasks = [Mock(priority=1.0, required_skills=['python']) for _ in range(5)]
                
                features = predictor.extract_features(agents, tasks, qubo_matrix)
                
                # Simulate quantum advantage for larger problems
                advantage = i >= 5
                classical_time = 5.0
                quantum_time = 2.0 if advantage else 6.0
                
                predictor.record_advantage_observation(
                    features=features,
                    classical_time=classical_time,
                    quantum_time=quantum_time,
                    classical_quality=0.9,
                    quantum_quality=0.85
                )
            
            # Test prediction
            test_features = ProblemFeatures(
                problem_size=15, density=0.5, sparsity=0.5, constraint_ratio=0.1,
                agent_capacity_variance=1.0, task_priority_spread=2.0,
                skill_overlap_ratio=0.3, graph_connectivity=0.4,
                matrix_condition_number=50.0, eigenvalue_spread=1.0
            )
            
            prediction = predictor.predict_quantum_advantage(test_features)
            
            # Should have trained models and made prediction
            assert predictor.advantage_classifier is not None
            assert predictor.speedup_regressor is not None
            assert 'advantage_probability' in prediction
            assert 'expected_speedup' in prediction
            assert 'recommendation' in prediction