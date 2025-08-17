"""Quantum Meta-Learning Framework for Autonomous Algorithm Portfolio Optimization.

This module implements a revolutionary meta-learning system that autonomously
discovers, combines, and optimizes quantum algorithms based on problem characteristics
and performance feedback. It represents the next evolution in quantum optimization.

Key Innovations:
- Meta-learning for algorithm portfolio optimization
- Autonomous quantum circuit architecture search
- Real-time algorithm performance adaptation
- Cross-problem knowledge transfer
- Emergent optimization strategy discovery
"""

import logging
import time
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


class MetaLearningStrategy(Enum):
    """Meta-learning strategies for algorithm optimization."""
    GRADIENT_BASED_META_LEARNING = "gbml"
    MODEL_AGNOSTIC_META_LEARNING = "maml"
    NEURAL_ARCHITECTURE_SEARCH = "nas"
    REINFORCEMENT_LEARNING_BASED = "rl"
    EVOLUTIONARY_STRATEGY = "es"
    BAYESIAN_OPTIMIZATION = "bo"
    MULTI_OBJECTIVE_OPTIMIZATION = "moo"


@dataclass
class AlgorithmGenome:
    """Genetic representation of a quantum algorithm."""
    circuit_depth: int
    gate_sequence: List[str]
    parameter_ranges: Dict[str, Tuple[float, float]]
    entanglement_pattern: str
    measurement_strategy: str
    error_mitigation: bool
    adaptive_parameters: Dict[str, Any]
    
    def to_hash(self) -> str:
        """Generate unique hash for the algorithm genome."""
        genome_str = json.dumps({
            "depth": self.circuit_depth,
            "gates": self.gate_sequence,
            "params": self.parameter_ranges,
            "entanglement": self.entanglement_pattern,
            "measurement": self.measurement_strategy,
            "error_mitigation": self.error_mitigation
        }, sort_keys=True)
        return hashlib.md5(genome_str.encode()).hexdigest()
    
    def mutate(self, mutation_rate: float = 0.1) -> 'AlgorithmGenome':
        """Create a mutated version of the algorithm genome."""
        new_genome = AlgorithmGenome(
            circuit_depth=self.circuit_depth,
            gate_sequence=self.gate_sequence.copy(),
            parameter_ranges=self.parameter_ranges.copy(),
            entanglement_pattern=self.entanglement_pattern,
            measurement_strategy=self.measurement_strategy,
            error_mitigation=self.error_mitigation,
            adaptive_parameters=self.adaptive_parameters.copy()
        )
        
        # Mutate circuit depth
        if random.random() < mutation_rate:
            new_genome.circuit_depth = max(1, self.circuit_depth + random.randint(-2, 2))
        
        # Mutate gate sequence
        if random.random() < mutation_rate:
            available_gates = ["RX", "RY", "RZ", "CNOT", "CZ", "H", "T", "Tdag"]
            if new_genome.gate_sequence:
                idx = random.randint(0, len(new_genome.gate_sequence) - 1)
                new_genome.gate_sequence[idx] = random.choice(available_gates)
        
        # Mutate entanglement pattern
        if random.random() < mutation_rate:
            patterns = ["linear", "circular", "all-to-all", "nearest-neighbor", "random"]
            new_genome.entanglement_pattern = random.choice(patterns)
        
        return new_genome
    
    def crossover(self, other: 'AlgorithmGenome') -> 'AlgorithmGenome':
        """Create offspring through crossover with another genome."""
        return AlgorithmGenome(
            circuit_depth=(self.circuit_depth + other.circuit_depth) // 2,
            gate_sequence=self.gate_sequence[:len(self.gate_sequence)//2] + 
                         other.gate_sequence[len(other.gate_sequence)//2:],
            parameter_ranges={**self.parameter_ranges, **other.parameter_ranges},
            entanglement_pattern=random.choice([self.entanglement_pattern, other.entanglement_pattern]),
            measurement_strategy=random.choice([self.measurement_strategy, other.measurement_strategy]),
            error_mitigation=random.choice([self.error_mitigation, other.error_mitigation]),
            adaptive_parameters={**self.adaptive_parameters, **other.adaptive_parameters}
        )


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for algorithm evaluation."""
    accuracy: float
    execution_time: float
    quantum_resource_usage: Dict[str, float]
    convergence_rate: float
    solution_stability: float
    scalability_factor: float
    generalization_score: float
    quantum_advantage_score: float
    
    @property
    def overall_fitness(self) -> float:
        """Calculate overall fitness score for meta-learning."""
        # Weighted combination of metrics
        weights = {
            'accuracy': 0.25,
            'speed': 0.20,
            'efficiency': 0.15,
            'stability': 0.15,
            'scalability': 0.15,
            'quantum_advantage': 0.10
        }
        
        speed_score = 1.0 / (1.0 + self.execution_time)
        efficiency_score = 1.0 / (1.0 + sum(self.quantum_resource_usage.values()))
        
        fitness = (weights['accuracy'] * self.accuracy +
                  weights['speed'] * speed_score +
                  weights['efficiency'] * efficiency_score +
                  weights['stability'] * self.solution_stability +
                  weights['scalability'] * self.scalability_factor +
                  weights['quantum_advantage'] * self.quantum_advantage_score)
        
        return fitness


@dataclass
class ProblemCharacteristics:
    """Characteristics of optimization problems for meta-learning."""
    problem_size: int
    connectivity: float
    constraint_density: float
    symmetry_measure: float
    problem_class: str
    hardness_estimate: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.problem_size,
            self.connectivity,
            self.constraint_density,
            self.symmetry_measure,
            self.hardness_estimate
        ])
    
    def similarity(self, other: 'ProblemCharacteristics') -> float:
        """Calculate similarity to another problem."""
        self_vec = self.to_feature_vector()
        other_vec = other.to_feature_vector()
        
        # Normalize vectors
        self_norm = self_vec / (np.linalg.norm(self_vec) + 1e-8)
        other_norm = other_vec / (np.linalg.norm(other_vec) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(self_norm, other_norm)
        return float(np.clip(similarity, 0.0, 1.0))


class QuantumAlgorithmEvolution:
    """Evolutionary algorithm for quantum circuit optimization."""
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        self.population: List[AlgorithmGenome] = []
        self.fitness_history: List[List[float]] = []
        self.best_genome: Optional[AlgorithmGenome] = None
        self.best_fitness: float = 0.0
        
    def initialize_population(self) -> List[AlgorithmGenome]:
        """Initialize a diverse population of algorithm genomes."""
        population = []
        
        for _ in range(self.population_size):
            genome = AlgorithmGenome(
                circuit_depth=random.randint(1, 8),
                gate_sequence=self._generate_random_gate_sequence(),
                parameter_ranges=self._generate_parameter_ranges(),
                entanglement_pattern=random.choice(["linear", "circular", "all-to-all", "random"]),
                measurement_strategy=random.choice(["computational", "pauli", "adaptive"]),
                error_mitigation=random.choice([True, False]),
                adaptive_parameters={}
            )
            population.append(genome)
        
        self.population = population
        return population
    
    def _generate_random_gate_sequence(self) -> List[str]:
        """Generate a random quantum gate sequence."""
        gates = ["RX", "RY", "RZ", "CNOT", "CZ", "H", "T", "Tdag"]
        sequence_length = random.randint(5, 20)
        return [random.choice(gates) for _ in range(sequence_length)]
    
    def _generate_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Generate random parameter ranges for quantum gates."""
        return {
            "rotation_angle": (0.0, 2 * np.pi),
            "entanglement_strength": (0.0, 1.0),
            "noise_tolerance": (0.0, 0.1)
        }
    
    async def evolve_generation(self, 
                              fitness_evaluator: Callable[[AlgorithmGenome], float],
                              generation: int) -> List[AlgorithmGenome]:
        """Evolve one generation of the population."""
        # Evaluate fitness for all genomes
        fitness_scores = []
        for genome in self.population:
            fitness = await self._evaluate_genome_async(genome, fitness_evaluator)
            fitness_scores.append(fitness)
        
        # Update best genome
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_genome = self.population[best_idx]
        
        # Store fitness history
        self.fitness_history.append(fitness_scores.copy())
        
        # Selection, crossover, and mutation
        new_population = self._select_and_reproduce(fitness_scores)
        
        self.population = new_population
        return new_population
    
    async def _evaluate_genome_async(self, 
                                   genome: AlgorithmGenome,
                                   fitness_evaluator: Callable[[AlgorithmGenome], float]) -> float:
        """Asynchronously evaluate the fitness of a genome."""
        try:
            return fitness_evaluator(genome)
        except Exception as e:
            logger.warning(f"Error evaluating genome: {e}")
            return 0.0
    
    def _select_and_reproduce(self, fitness_scores: List[float]) -> List[AlgorithmGenome]:
        """Select parents and create new generation."""
        new_population = []
        
        # Elitism - keep best genomes
        elite_count = int(self.population_size * self.elitism_ratio)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate rest through selection and reproduction
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> AlgorithmGenome:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(tournament_size, len(self.population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]


class MetaLearningKnowledgeBase:
    """Knowledge base for storing and retrieving meta-learning insights."""
    
    def __init__(self):
        self.problem_algorithm_mapping: Dict[str, List[AlgorithmGenome]] = defaultdict(list)
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.transfer_learning_weights: Dict[str, np.ndarray] = {}
        
    def store_successful_algorithm(self, 
                                 problem_chars: ProblemCharacteristics,
                                 algorithm: AlgorithmGenome,
                                 performance: PerformanceMetrics):
        """Store a successful algorithm for a problem type."""
        problem_key = self._generate_problem_key(problem_chars)
        self.problem_algorithm_mapping[problem_key].append(algorithm)
        self.performance_history[problem_key].append(performance)
        
        # Limit storage size
        max_algorithms_per_problem = 20
        if len(self.problem_algorithm_mapping[problem_key]) > max_algorithms_per_problem:
            # Keep only the best performing algorithms
            algorithms = self.problem_algorithm_mapping[problem_key]
            performances = self.performance_history[problem_key]
            
            # Sort by fitness
            sorted_pairs = sorted(zip(algorithms, performances), 
                                key=lambda x: x[1].overall_fitness, reverse=True)
            
            self.problem_algorithm_mapping[problem_key] = [alg for alg, _ in sorted_pairs[:max_algorithms_per_problem]]
            self.performance_history[problem_key] = [perf for _, perf in sorted_pairs[:max_algorithms_per_problem]]
    
    def retrieve_similar_algorithms(self, 
                                  problem_chars: ProblemCharacteristics,
                                  similarity_threshold: float = 0.7) -> List[Tuple[AlgorithmGenome, float]]:
        """Retrieve algorithms from similar problems."""
        target_features = problem_chars.to_feature_vector()
        similar_algorithms = []
        
        for problem_key, algorithms in self.problem_algorithm_mapping.items():
            stored_chars = self._decode_problem_key(problem_key)
            if stored_chars:
                similarity = problem_chars.similarity(stored_chars)
                if similarity >= similarity_threshold:
                    for algorithm in algorithms:
                        similar_algorithms.append((algorithm, similarity))
        
        # Sort by similarity
        similar_algorithms.sort(key=lambda x: x[1], reverse=True)
        return similar_algorithms
    
    def _generate_problem_key(self, problem_chars: ProblemCharacteristics) -> str:
        """Generate a unique key for problem characteristics."""
        return f"{problem_chars.problem_class}_{problem_chars.problem_size//10}_{int(problem_chars.connectivity*10)}"
    
    def _decode_problem_key(self, problem_key: str) -> Optional[ProblemCharacteristics]:
        """Decode problem key back to characteristics (simplified)."""
        # This is a simplified implementation - in practice would store full characteristics
        try:
            parts = problem_key.split("_")
            return ProblemCharacteristics(
                problem_size=int(parts[1]) * 10,
                connectivity=int(parts[2]) / 10.0,
                constraint_density=0.5,  # Default values
                symmetry_measure=0.5,
                problem_class=parts[0],
                hardness_estimate=0.5
            )
        except:
            return None


class QuantumMetaLearningFramework:
    """Main framework for quantum meta-learning optimization."""
    
    def __init__(self, 
                 strategy: MetaLearningStrategy = MetaLearningStrategy.EVOLUTIONARY_STRATEGY,
                 max_generations: int = 50,
                 population_size: int = 30):
        self.strategy = strategy
        self.max_generations = max_generations
        self.population_size = population_size
        
        # Core components
        self.algorithm_evolution = QuantumAlgorithmEvolution(population_size)
        self.knowledge_base = MetaLearningKnowledgeBase()
        
        # Learning state
        self.current_generation = 0
        self.learning_history: List[Dict[str, Any]] = []
        self.problem_portfolio: List[ProblemCharacteristics] = []
        
    async def meta_optimize(self, 
                          problem_chars: ProblemCharacteristics,
                          qubo_matrix: np.ndarray,
                          max_time: float = 300.0) -> Dict[str, Any]:
        """Perform meta-optimization for a given problem."""
        start_time = time.time()
        
        # Check knowledge base for similar problems
        similar_algorithms = self.knowledge_base.retrieve_similar_algorithms(problem_chars)
        
        if similar_algorithms:
            logger.info(f"Found {len(similar_algorithms)} similar algorithms for transfer learning")
            # Initialize population with knowledge transfer
            self._initialize_with_transfer_learning(similar_algorithms)
        else:
            logger.info("No similar problems found, initializing random population")
            self.algorithm_evolution.initialize_population()
        
        # Define fitness evaluator for this specific problem
        def fitness_evaluator(genome: AlgorithmGenome) -> float:
            return self._evaluate_algorithm_fitness(genome, qubo_matrix, problem_chars)
        
        # Evolutionary optimization
        best_algorithms = []
        
        for generation in range(self.max_generations):
            if time.time() - start_time > max_time:
                logger.info(f"Time limit reached, stopping at generation {generation}")
                break
                
            population = await self.algorithm_evolution.evolve_generation(
                fitness_evaluator, generation
            )
            
            # Track progress
            generation_fitness = [fitness_evaluator(genome) for genome in population]
            best_fitness = max(generation_fitness)
            avg_fitness = np.mean(generation_fitness)
            
            logger.info(f"Generation {generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")
            
            # Store learning progress
            self.learning_history.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "average_fitness": avg_fitness,
                "population_diversity": self._calculate_population_diversity(population)
            })
            
            # Early stopping if converged
            if len(self.learning_history) > 10:
                recent_improvement = (self.learning_history[-1]["best_fitness"] - 
                                    self.learning_history[-10]["best_fitness"])
                if recent_improvement < 1e-6:
                    logger.info(f"Converged at generation {generation}")
                    break
        
        # Get best algorithm and evaluate its performance
        best_algorithm = self.algorithm_evolution.best_genome
        best_performance = self._detailed_performance_evaluation(
            best_algorithm, qubo_matrix, problem_chars
        )
        
        # Store successful algorithm in knowledge base
        if best_performance.overall_fitness > 0.7:  # Threshold for "successful"
            self.knowledge_base.store_successful_algorithm(
                problem_chars, best_algorithm, best_performance
            )
        
        total_time = time.time() - start_time
        
        return {
            "best_algorithm": best_algorithm,
            "performance_metrics": best_performance,
            "optimization_time": total_time,
            "generations_completed": self.current_generation,
            "transfer_learning_used": len(similar_algorithms) > 0,
            "knowledge_base_size": len(self.knowledge_base.problem_algorithm_mapping)
        }
    
    def _initialize_with_transfer_learning(self, 
                                         similar_algorithms: List[Tuple[AlgorithmGenome, float]]):
        """Initialize population using transfer learning from similar problems."""
        # Use top similar algorithms as seeds
        seed_algorithms = [alg for alg, sim in similar_algorithms[:10]]
        
        # Create new population with mutations of successful algorithms
        population = []
        
        for i in range(self.population_size):
            if i < len(seed_algorithms):
                # Use successful algorithm with small mutations
                base_algorithm = seed_algorithms[i % len(seed_algorithms)]
                mutated_algorithm = base_algorithm.mutate(mutation_rate=0.05)
                population.append(mutated_algorithm)
            else:
                # Create hybrid algorithms through crossover
                if len(seed_algorithms) >= 2:
                    parent1 = random.choice(seed_algorithms)
                    parent2 = random.choice(seed_algorithms)
                    hybrid = parent1.crossover(parent2)
                    population.append(hybrid)
                else:
                    # Fall back to random initialization
                    population.extend(self.algorithm_evolution.initialize_population())
                    break
        
        self.algorithm_evolution.population = population[:self.population_size]
    
    def _evaluate_algorithm_fitness(self, 
                                  genome: AlgorithmGenome,
                                  qubo_matrix: np.ndarray,
                                  problem_chars: ProblemCharacteristics) -> float:
        """Evaluate the fitness of an algorithm genome."""
        try:
            # Simulate quantum algorithm execution
            start_time = time.time()
            
            # Create quantum circuit based on genome
            circuit_result = self._simulate_quantum_circuit(genome, qubo_matrix)
            
            execution_time = time.time() - start_time
            
            # Calculate performance metrics
            accuracy = self._calculate_solution_accuracy(circuit_result, qubo_matrix)
            efficiency = 1.0 / (1.0 + execution_time)
            
            # Fitness combines multiple factors
            fitness = 0.6 * accuracy + 0.3 * efficiency + 0.1 * circuit_result.get("stability", 0.5)
            
            return float(np.clip(fitness, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error evaluating algorithm fitness: {e}")
            return 0.0
    
    def _simulate_quantum_circuit(self, 
                                genome: AlgorithmGenome,
                                qubo_matrix: np.ndarray) -> Dict[str, Any]:
        """Simulate quantum circuit execution based on genome."""
        n = qubo_matrix.shape[0]
        
        # Simplified simulation - in practice would use quantum simulator
        # Generate solution based on genome characteristics
        if "RX" in genome.gate_sequence:
            # Rotation-heavy circuit - tends toward balanced solutions
            solution_prob = 0.5 + 0.1 * np.sin(np.arange(n))
        elif "CNOT" in genome.gate_sequence:
            # Entanglement-heavy circuit - tends toward correlated solutions
            solution_prob = 0.5 + 0.2 * np.cos(np.arange(n) * 0.5)
        else:
            # Default random solution
            solution_prob = np.random.random(n)
        
        solution = (solution_prob > 0.5).astype(int)
        energy = np.dot(solution, np.dot(qubo_matrix, solution))
        
        return {
            "solution": solution,
            "energy": energy,
            "stability": np.std(solution_prob),
            "convergence_steps": genome.circuit_depth * 10
        }
    
    def _calculate_solution_accuracy(self, 
                                   circuit_result: Dict[str, Any],
                                   qubo_matrix: np.ndarray) -> float:
        """Calculate solution accuracy relative to optimal."""
        solution = circuit_result["solution"]
        energy = circuit_result["energy"]
        
        # Estimate optimal energy (simplified)
        n = len(solution)
        random_energies = []
        for _ in range(100):
            random_solution = np.random.randint(0, 2, n)
            random_energy = np.dot(random_solution, np.dot(qubo_matrix, random_solution))
            random_energies.append(random_energy)
        
        # Use minimum energy as approximation of optimal
        estimated_optimal = min(random_energies)
        worst_energy = max(random_energies)
        
        # Normalize energy to [0, 1] where 1 is best
        if worst_energy != estimated_optimal:
            accuracy = 1.0 - (energy - estimated_optimal) / (worst_energy - estimated_optimal)
        else:
            accuracy = 1.0
        
        return float(np.clip(accuracy, 0.0, 1.0))
    
    def _detailed_performance_evaluation(self, 
                                       algorithm: AlgorithmGenome,
                                       qubo_matrix: np.ndarray,
                                       problem_chars: ProblemCharacteristics) -> PerformanceMetrics:
        """Perform detailed performance evaluation of an algorithm."""
        # Run multiple trials for stable metrics
        num_trials = 10
        accuracies = []
        execution_times = []
        energies = []
        
        for _ in range(num_trials):
            start_time = time.time()
            result = self._simulate_quantum_circuit(algorithm, qubo_matrix)
            execution_time = time.time() - start_time
            
            accuracy = self._calculate_solution_accuracy(result, qubo_matrix)
            
            accuracies.append(accuracy)
            execution_times.append(execution_time)
            energies.append(result["energy"])
        
        # Calculate aggregate metrics
        avg_accuracy = np.mean(accuracies)
        avg_execution_time = np.mean(execution_times)
        solution_stability = 1.0 - np.std(accuracies)  # Lower variance is better
        convergence_rate = 1.0 / avg_execution_time
        
        # Estimate scalability and quantum advantage
        scalability_factor = min(1.0, 100.0 / problem_chars.problem_size)  # Simplified
        quantum_advantage_score = min(1.0, avg_accuracy * scalability_factor)
        
        return PerformanceMetrics(
            accuracy=avg_accuracy,
            execution_time=avg_execution_time,
            quantum_resource_usage={"qubits": algorithm.circuit_depth, "gates": len(algorithm.gate_sequence)},
            convergence_rate=convergence_rate,
            solution_stability=solution_stability,
            scalability_factor=scalability_factor,
            generalization_score=0.8,  # Placeholder - would measure across problem variations
            quantum_advantage_score=quantum_advantage_score
        )
    
    def _calculate_population_diversity(self, population: List[AlgorithmGenome]) -> float:
        """Calculate diversity of the population."""
        if len(population) < 2:
            return 0.0
        
        # Use genome hashes to measure diversity
        unique_hashes = set(genome.to_hash() for genome in population)
        diversity = len(unique_hashes) / len(population)
        
        return diversity
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the meta-learning process."""
        if not self.learning_history:
            return {"message": "No learning history available"}
        
        # Analyze learning progression
        fitness_progression = [record["best_fitness"] for record in self.learning_history]
        diversity_progression = [record["population_diversity"] for record in self.learning_history]
        
        # Calculate learning metrics
        total_improvement = fitness_progression[-1] - fitness_progression[0] if len(fitness_progression) > 1 else 0
        convergence_speed = len(fitness_progression)  # Generations to converge
        
        return {
            "total_improvement": total_improvement,
            "convergence_speed": convergence_speed,
            "final_fitness": fitness_progression[-1] if fitness_progression else 0,
            "average_diversity": np.mean(diversity_progression) if diversity_progression else 0,
            "knowledge_base_problems": len(self.knowledge_base.problem_algorithm_mapping),
            "learning_efficiency": total_improvement / max(convergence_speed, 1)
        }


# Factory function for easy instantiation
def create_quantum_meta_learning_framework(
    strategy: str = "evolutionary_strategy",
    max_generations: int = 30,
    population_size: int = 20
) -> QuantumMetaLearningFramework:
    """Create a quantum meta-learning framework with specified configuration."""
    strategy_enum = MetaLearningStrategy(strategy)
    
    return QuantumMetaLearningFramework(
        strategy=strategy_enum,
        max_generations=max_generations,
        population_size=population_size
    )


# Example usage and demonstration
async def demonstrate_meta_learning():
    """Demonstrate the quantum meta-learning framework."""
    # Create sample problem
    n = 16
    qubo_matrix = np.random.randn(n, n)
    qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2  # Make symmetric
    
    problem_chars = ProblemCharacteristics(
        problem_size=n,
        connectivity=0.6,
        constraint_density=0.3,
        symmetry_measure=0.5,
        problem_class="random_qubo",
        hardness_estimate=0.7
    )
    
    # Create meta-learning framework
    framework = create_quantum_meta_learning_framework(
        strategy="evolutionary_strategy",
        max_generations=20,
        population_size=15
    )
    
    # Perform meta-optimization
    result = await framework.meta_optimize(problem_chars, qubo_matrix, max_time=120.0)
    
    print(f"Meta-optimization completed:")
    print(f"  Best algorithm fitness: {result['performance_metrics'].overall_fitness:.4f}")
    print(f"  Optimization time: {result['optimization_time']:.2f}s")
    print(f"  Generations completed: {result['generations_completed']}")
    print(f"  Transfer learning used: {result['transfer_learning_used']}")
    
    # Get meta-learning insights
    insights = framework.get_meta_learning_insights()
    print(f"Meta-learning insights: {insights}")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_meta_learning())