"""Advanced Quantum Advantage Prediction Framework.

This module implements machine learning-enhanced prediction of quantum advantage
for scheduling problems, enabling intelligent quantum/classical selection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ProblemFeatures:
    """Feature vector for quantum advantage prediction."""
    problem_size: int
    density: float
    sparsity: float
    constraint_ratio: float
    agent_capacity_variance: float
    task_priority_spread: float
    skill_overlap_ratio: float
    graph_connectivity: float
    matrix_condition_number: float
    eigenvalue_spread: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.problem_size,
            self.density,
            self.sparsity,
            self.constraint_ratio,
            self.agent_capacity_variance,
            self.task_priority_spread,
            self.skill_overlap_ratio,
            self.graph_connectivity,
            self.matrix_condition_number,
            self.eigenvalue_spread
        ])


@dataclass
class QuantumAdvantageRecord:
    """Historical record of quantum advantage observation."""
    features: ProblemFeatures
    classical_time: float
    quantum_time: float
    speedup_ratio: float
    advantage_observed: bool
    solution_quality_classical: float
    solution_quality_quantum: float
    timestamp: float
    backend_used: str
    
    @property
    def normalized_advantage(self) -> float:
        """Compute quality-adjusted quantum advantage."""
        if self.quantum_time <= 0:
            return 0.0
        raw_speedup = self.classical_time / self.quantum_time
        quality_penalty = max(0, self.solution_quality_classical - self.solution_quality_quantum)
        return raw_speedup * (1.0 - quality_penalty)


class QuantumAdvantagePredictor:
    """ML-powered predictor for quantum advantage in scheduling problems."""
    
    def __init__(self, 
                 model_cache_path: Optional[Path] = None,
                 min_training_samples: int = 100):
        """Initialize the quantum advantage predictor.
        
        Args:
            model_cache_path: Path to cache trained models
            min_training_samples: Minimum samples needed for training
        """
        self.model_cache_path = model_cache_path or Path("models/quantum_advantage")
        self.model_cache_path.mkdir(parents=True, exist_ok=True)
        
        self.min_training_samples = min_training_samples
        self.training_records: List[QuantumAdvantageRecord] = []
        
        # ML Models
        self.advantage_classifier = None  # Predicts if quantum will have advantage
        self.speedup_regressor = None     # Predicts expected speedup ratio
        self.feature_scaler = StandardScaler()
        
        # Model performance tracking
        self.model_metrics = {
            'classifier_accuracy': 0.0,
            'regressor_mae': float('inf'),
            'last_training_time': 0.0,
            'training_samples': 0
        }
        
        self._load_models()
        logger.info(f"Initialized QuantumAdvantagePredictor with {len(self.training_records)} historical records")
    
    def extract_features(self, 
                        agents: List[Any], 
                        tasks: List[Any], 
                        qubo_matrix: np.ndarray) -> ProblemFeatures:
        """Extract feature vector from scheduling problem.
        
        Args:
            agents: List of agents with capabilities
            tasks: List of tasks with requirements
            qubo_matrix: QUBO problem formulation
            
        Returns:
            ProblemFeatures object with extracted features
        """
        n = qubo_matrix.shape[0]
        
        # Basic matrix properties
        density = np.count_nonzero(qubo_matrix) / (n * n)
        sparsity = 1.0 - density
        
        # Eigenvalue analysis for problem structure
        try:
            eigenvals = np.linalg.eigvals(qubo_matrix)
            eigenvalue_spread = np.std(eigenvals) / (np.mean(np.abs(eigenvals)) + 1e-8)
            condition_number = np.linalg.cond(qubo_matrix)
        except:
            eigenvalue_spread = 0.0
            condition_number = 1.0
        
        # Agent-task structure analysis
        if agents and tasks:
            agent_capacities = [getattr(agent, 'capacity', 1) for agent in agents]
            task_priorities = [getattr(task, 'priority', 1.0) for task in tasks]
            
            agent_capacity_variance = np.var(agent_capacities) if len(agent_capacities) > 1 else 0.0
            task_priority_spread = np.std(task_priorities) if len(task_priorities) > 1 else 0.0
            
            # Skill overlap analysis
            all_skills = set()
            agent_skills = []
            for agent in agents:
                skills = getattr(agent, 'skills', [])
                agent_skills.append(set(skills))
                all_skills.update(skills)
            
            if len(all_skills) > 0 and len(agent_skills) > 1:
                overlaps = []
                for i in range(len(agent_skills)):
                    for j in range(i+1, len(agent_skills)):
                        overlap = len(agent_skills[i] & agent_skills[j]) / len(agent_skills[i] | agent_skills[j])
                        overlaps.append(overlap)
                skill_overlap_ratio = np.mean(overlaps) if overlaps else 0.0
            else:
                skill_overlap_ratio = 0.0
        else:
            agent_capacity_variance = 0.0
            task_priority_spread = 0.0
            skill_overlap_ratio = 0.0
        
        # Graph connectivity (approximate from QUBO structure)
        graph_connectivity = np.mean(np.sum(qubo_matrix != 0, axis=1)) / n if n > 0 else 0.0
        
        # Constraint ratio estimation
        constraint_ratio = np.count_nonzero(np.diag(qubo_matrix)) / n if n > 0 else 0.0
        
        return ProblemFeatures(
            problem_size=n,
            density=density,
            sparsity=sparsity,
            constraint_ratio=constraint_ratio,
            agent_capacity_variance=agent_capacity_variance,
            task_priority_spread=task_priority_spread,
            skill_overlap_ratio=skill_overlap_ratio,
            graph_connectivity=graph_connectivity,
            matrix_condition_number=min(condition_number, 1e6),  # Cap for numerical stability
            eigenvalue_spread=eigenvalue_spread
        )
    
    def record_advantage_observation(self,
                                   features: ProblemFeatures,
                                   classical_time: float,
                                   quantum_time: float,
                                   classical_quality: float,
                                   quantum_quality: float,
                                   backend_used: str = "unknown") -> None:
        """Record a quantum advantage observation for training.
        
        Args:
            features: Problem features
            classical_time: Classical solver execution time
            quantum_time: Quantum solver execution time
            classical_quality: Classical solution quality
            quantum_quality: Quantum solution quality
            backend_used: Quantum backend identifier
        """
        import time
        
        speedup_ratio = classical_time / quantum_time if quantum_time > 0 else 0.0
        advantage_observed = speedup_ratio > 1.0 and quantum_quality >= 0.95 * classical_quality
        
        record = QuantumAdvantageRecord(
            features=features,
            classical_time=classical_time,
            quantum_time=quantum_time,
            speedup_ratio=speedup_ratio,
            advantage_observed=advantage_observed,
            solution_quality_classical=classical_quality,
            solution_quality_quantum=quantum_quality,
            timestamp=time.time(),
            backend_used=backend_used
        )
        
        self.training_records.append(record)
        logger.debug(f"Recorded advantage observation: speedup={speedup_ratio:.2f}x, advantage={advantage_observed}")
        
        # Retrain models if we have enough data
        if len(self.training_records) >= self.min_training_samples:
            if len(self.training_records) % 50 == 0:  # Retrain every 50 observations
                self._train_models()
    
    def predict_quantum_advantage(self, 
                                 features: ProblemFeatures,
                                 confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Predict quantum advantage for a given problem.
        
        Args:
            features: Problem feature vector
            confidence_threshold: Minimum confidence for quantum recommendation
            
        Returns:
            Prediction dictionary with advantage probability and expected speedup
        """
        if self.advantage_classifier is None or self.speedup_regressor is None:
            return {
                'advantage_probability': 0.5,  # Neutral prediction
                'expected_speedup': 1.0,
                'confidence': 0.0,
                'recommendation': 'classical',  # Safe default
                'reason': 'insufficient_training_data'
            }
        
        # Prepare features
        feature_vector = features.to_vector().reshape(1, -1)
        scaled_features = self.feature_scaler.transform(feature_vector)
        
        # Get predictions
        advantage_proba = self.advantage_classifier.predict_proba(scaled_features)[0][1]
        expected_speedup = max(0.1, self.speedup_regressor.predict(scaled_features)[0])
        
        # Calculate confidence based on model performance and prediction certainty
        model_confidence = 1.0 - self.model_metrics['regressor_mae'] / 10.0  # Normalize MAE
        prediction_confidence = abs(advantage_proba - 0.5) * 2  # Distance from uncertain (0.5)
        overall_confidence = (model_confidence + prediction_confidence) / 2
        
        # Make recommendation
        if advantage_proba > confidence_threshold and expected_speedup > 1.2:
            recommendation = 'quantum'
            reason = f'high_advantage_probability_{advantage_proba:.2f}'
        elif advantage_proba > 0.6 and expected_speedup > 1.0:
            recommendation = 'hybrid'
            reason = f'moderate_advantage_probability_{advantage_proba:.2f}'
        else:
            recommendation = 'classical'
            reason = f'low_advantage_probability_{advantage_proba:.2f}'
        
        return {
            'advantage_probability': advantage_proba,
            'expected_speedup': expected_speedup,
            'confidence': overall_confidence,
            'recommendation': recommendation,
            'reason': reason,
            'model_metrics': self.model_metrics.copy()
        }
    
    def _train_models(self) -> None:
        """Train ML models on historical advantage observations."""
        if len(self.training_records) < self.min_training_samples:
            logger.warning(f"Insufficient training data: {len(self.training_records)} < {self.min_training_samples}")
            return
        
        import time
        start_time = time.time()
        
        # Prepare training data
        X = np.array([record.features.to_vector() for record in self.training_records])
        y_classification = np.array([record.advantage_observed for record in self.training_records])
        y_regression = np.array([record.normalized_advantage for record in self.training_records])
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train advantage classifier
        self.advantage_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Hyperparameter tuning for classifier
        param_grid_clf = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [4, 6, 8]
        }
        
        grid_search_clf = GridSearchCV(
            self.advantage_classifier, 
            param_grid_clf, 
            cv=5, 
            scoring='accuracy'
        )
        grid_search_clf.fit(X_scaled, y_classification)
        self.advantage_classifier = grid_search_clf.best_estimator_
        
        # Train speedup regressor
        self.speedup_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Hyperparameter tuning for regressor
        param_grid_reg = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search_reg = GridSearchCV(
            self.speedup_regressor,
            param_grid_reg,
            cv=5,
            scoring='neg_mean_absolute_error'
        )
        grid_search_reg.fit(X_scaled, y_regression)
        self.speedup_regressor = grid_search_reg.best_estimator_
        
        # Evaluate model performance
        clf_scores = cross_val_score(self.advantage_classifier, X_scaled, y_classification, cv=5)
        reg_scores = cross_val_score(self.speedup_regressor, X_scaled, y_regression, cv=5, scoring='neg_mean_absolute_error')
        
        self.model_metrics.update({
            'classifier_accuracy': np.mean(clf_scores),
            'regressor_mae': abs(np.mean(reg_scores)),
            'last_training_time': time.time() - start_time,
            'training_samples': len(self.training_records)
        })
        
        # Save models
        self._save_models()
        
        logger.info(f"Trained quantum advantage models in {self.model_metrics['last_training_time']:.2f}s")
        logger.info(f"Classifier accuracy: {self.model_metrics['classifier_accuracy']:.3f}")
        logger.info(f"Regressor MAE: {self.model_metrics['regressor_mae']:.3f}")
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            model_data = {
                'advantage_classifier': self.advantage_classifier,
                'speedup_regressor': self.speedup_regressor,
                'feature_scaler': self.feature_scaler,
                'training_records': self.training_records,
                'model_metrics': self.model_metrics
            }
            
            model_path = self.model_cache_path / "quantum_advantage_models.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.debug(f"Saved models to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        try:
            model_path = self.model_cache_path / "quantum_advantage_models.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.advantage_classifier = model_data.get('advantage_classifier')
                self.speedup_regressor = model_data.get('speedup_regressor')
                self.feature_scaler = model_data.get('feature_scaler', StandardScaler())
                self.training_records = model_data.get('training_records', [])
                self.model_metrics = model_data.get('model_metrics', self.model_metrics)
                
                logger.info(f"Loaded models with {len(self.training_records)} training records")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings from trained models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.advantage_classifier is None:
            return {}
        
        feature_names = [
            'problem_size', 'density', 'sparsity', 'constraint_ratio',
            'agent_capacity_variance', 'task_priority_spread', 'skill_overlap_ratio',
            'graph_connectivity', 'matrix_condition_number', 'eigenvalue_spread'
        ]
        
        # Get importance from classifier
        clf_importance = self.advantage_classifier.feature_importances_
        
        # Get importance from regressor  
        reg_importance = self.speedup_regressor.feature_importances_ if self.speedup_regressor else clf_importance
        
        # Average importances
        avg_importance = (clf_importance + reg_importance) / 2
        
        return dict(zip(feature_names, avg_importance))
    
    def generate_insights_report(self) -> str:
        """Generate comprehensive insights report about quantum advantage patterns.
        
        Returns:
            Formatted report string with insights and recommendations
        """
        if len(self.training_records) < 10:
            return "Insufficient data for insights generation. Need at least 10 observations."
        
        # Analyze advantage patterns
        advantageous_records = [r for r in self.training_records if r.advantage_observed]
        advantage_rate = len(advantageous_records) / len(self.training_records)
        
        # Size thresholds
        if advantageous_records:
            min_advantage_size = min(r.features.problem_size for r in advantageous_records)
            avg_advantage_size = np.mean([r.features.problem_size for r in advantageous_records])
            max_speedup = max(r.speedup_ratio for r in advantageous_records)
        else:
            min_advantage_size = avg_advantage_size = max_speedup = 0
        
        # Feature importance
        importance = self.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        report = [
            "# Quantum Advantage Analysis Report",
            "",
            f"**Analysis Period**: {len(self.training_records)} observations",
            f"**Quantum Advantage Rate**: {advantage_rate:.1%}",
            f"**Model Performance**: {self.model_metrics['classifier_accuracy']:.3f} accuracy",
            "",
            "## Key Findings",
            "",
            f"- **Minimum problem size for advantage**: {min_advantage_size} variables",
            f"- **Average advantageous problem size**: {avg_advantage_size:.0f} variables", 
            f"- **Maximum observed speedup**: {max_speedup:.1f}x",
            "",
            "## Most Predictive Features",
            ""
        ]
        
        for i, (feature, importance_score) in enumerate(top_features, 1):
            report.append(f"{i}. **{feature.replace('_', ' ').title()}**: {importance_score:.3f}")
        
        report.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        if advantage_rate > 0.6:
            report.append("✅ **Strong quantum advantage observed** - Prioritize quantum development")
        elif advantage_rate > 0.3:
            report.append("⚠️ **Moderate quantum advantage** - Continue hybrid approach")
        else:
            report.append("❌ **Limited quantum advantage** - Focus on classical optimization")
        
        if min_advantage_size > 0:
            report.append(f"- Consider quantum backends for problems with >{min_advantage_size} variables")
        
        if max_speedup > 5:
            report.append(f"- Significant speedup potential ({max_speedup:.1f}x) justifies quantum investment")
        
        return "\n".join(report)


class QuantumAdvantageExperiment:
    """Automated experiment framework for quantum advantage research."""
    
    def __init__(self, predictor: QuantumAdvantagePredictor):
        """Initialize experiment framework.
        
        Args:
            predictor: Quantum advantage predictor instance
        """
        self.predictor = predictor
        self.experiment_results = []
    
    async def run_comparative_study(self,
                                  problem_generator: Any,
                                  problem_sizes: List[int],
                                  num_problems_per_size: int = 10,
                                  backends: List[str] = None) -> Dict[str, Any]:
        """Run systematic comparative study across problem sizes.
        
        Args:
            problem_generator: Problem generation function
            problem_sizes: List of problem sizes to test
            num_problems_per_size: Number of problems per size category
            backends: List of quantum backends to test
            
        Returns:
            Comprehensive study results dictionary
        """
        backends = backends or ['classical', 'quantum_sim', 'quantum_hw']
        
        results = {
            'problem_sizes': problem_sizes,
            'backends': backends,
            'experiments': [],
            'summary_statistics': {}
        }
        
        for size in problem_sizes:
            for problem_idx in range(num_problems_per_size):
                # Generate problem
                problem = problem_generator(size)
                features = self.predictor.extract_features(
                    problem.get('agents', []), 
                    problem.get('tasks', []), 
                    problem['qubo_matrix']
                )
                
                # Test all backends
                backend_results = {}
                for backend in backends:
                    # Simulate backend execution (replace with actual execution)
                    execution_time, solution_quality = self._simulate_backend_execution(
                        problem, backend
                    )
                    
                    backend_results[backend] = {
                        'execution_time': execution_time,
                        'solution_quality': solution_quality
                    }
                
                # Record observations
                if 'classical' in backend_results and 'quantum_sim' in backend_results:
                    self.predictor.record_advantage_observation(
                        features=features,
                        classical_time=backend_results['classical']['execution_time'],
                        quantum_time=backend_results['quantum_sim']['execution_time'],
                        classical_quality=backend_results['classical']['solution_quality'],
                        quantum_quality=backend_results['quantum_sim']['solution_quality'],
                        backend_used='quantum_sim'
                    )
                
                results['experiments'].append({
                    'problem_size': size,
                    'problem_index': problem_idx,
                    'features': features,
                    'backend_results': backend_results
                })
        
        # Generate summary statistics
        results['summary_statistics'] = self._compute_study_statistics(results)
        
        return results
    
    def _simulate_backend_execution(self, 
                                  problem: Dict[str, Any], 
                                  backend: str) -> Tuple[float, float]:
        """Simulate backend execution for testing purposes.
        
        Args:
            problem: Problem definition
            backend: Backend identifier
            
        Returns:
            Tuple of (execution_time, solution_quality)
        """
        import random
        
        size = problem['qubo_matrix'].shape[0]
        
        if backend == 'classical':
            # Classical scaling: roughly O(n^2)
            execution_time = (size ** 2) * random.uniform(0.001, 0.005)
            solution_quality = random.uniform(0.85, 0.95)
        elif backend == 'quantum_sim':
            # Quantum simulator: varies with circuit depth
            execution_time = size * random.uniform(0.1, 0.5) 
            solution_quality = random.uniform(0.80, 0.98)
        elif backend == 'quantum_hw':
            # Quantum hardware: high overhead but potentially faster for large problems
            if size < 50:
                execution_time = random.uniform(15, 30)  # High setup overhead
                solution_quality = random.uniform(0.75, 0.90)
            else:
                execution_time = size * random.uniform(0.05, 0.2)  # Better scaling
                solution_quality = random.uniform(0.85, 0.95)
        else:
            execution_time = 1.0
            solution_quality = 0.5
        
        return execution_time, solution_quality
    
    def _compute_study_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics for the comparative study.
        
        Args:
            results: Raw experimental results
            
        Returns:
            Summary statistics dictionary
        """
        statistics = {}
        
        # Group by problem size
        for size in results['problem_sizes']:
            size_experiments = [exp for exp in results['experiments'] 
                              if exp['problem_size'] == size]
            
            if not size_experiments:
                continue
            
            size_stats = {}
            for backend in results['backends']:
                times = [exp['backend_results'][backend]['execution_time'] 
                        for exp in size_experiments if backend in exp['backend_results']]
                qualities = [exp['backend_results'][backend]['solution_quality'] 
                           for exp in size_experiments if backend in exp['backend_results']]
                
                if times and qualities:
                    size_stats[backend] = {
                        'mean_time': np.mean(times),
                        'std_time': np.std(times),
                        'mean_quality': np.mean(qualities),
                        'std_quality': np.std(qualities)
                    }
            
            # Compute speedup ratios
            if 'classical' in size_stats and 'quantum_sim' in size_stats:
                speedup = (size_stats['classical']['mean_time'] / 
                          size_stats['quantum_sim']['mean_time'])
                size_stats['quantum_speedup'] = speedup
                size_stats['quantum_advantage'] = speedup > 1.0
            
            statistics[f'size_{size}'] = size_stats
        
        return statistics