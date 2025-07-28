# ADR-0002: QUBO Formulation Strategy

## Status
Accepted

## Context
Multi-agent scheduling problems need to be converted into QUBO (Quadratic Unconstrained Binary Optimization) format for quantum annealing systems. This conversion must handle various constraint types (capacity, skills, deadlines) while maintaining problem tractability and solution quality.

## Decision
Implement a flexible QUBO formulation strategy using:

1. **Binary Variable Encoding**: Use x_{i,j} variables where task i is assigned to agent j
2. **Penalty Method**: Convert constraints to penalty terms in the objective function
3. **Configurable Weights**: Allow tuning of objective vs constraint penalties
4. **Sparse Matrix Optimization**: Use sparse representations for large problems
5. **Problem Decomposition**: Break large problems into smaller sub-problems when needed

### Objective Function Structure
```
Minimize: Σ(completion_time_penalty) + Σ(constraint_violations * penalty_weight)
```

### Constraint Handling
- **Hard Constraints**: High penalty weights (10-100x objective scale)
- **Soft Constraints**: Lower penalty weights (1-5x objective scale)
- **Constraint Validation**: Pre-solve validation to catch infeasible problems

## Consequences

### Positive
- **Quantum Compatible**: QUBO format works with all quantum annealing systems
- **Flexible Constraints**: Penalty method handles arbitrary constraint types
- **Tunable Trade-offs**: Weights allow balancing objectives vs constraint satisfaction
- **Scalable**: Sparse representation handles large problem instances

### Negative
- **Parameter Tuning**: Penalty weights require careful calibration
- **Approximation**: Penalty method may not enforce hard constraints perfectly
- **Matrix Size**: Binary encoding can create large QUBO matrices

### Risks
- **Constraint Violations**: Poorly tuned penalties may result in infeasible solutions
- **Local Minima**: QUBO formulation may have many local optima
- **Numerical Precision**: Large penalty values may cause numerical issues

## Alternatives Considered

### Constraint Programming
**Rejected**: Not directly compatible with quantum annealing hardware

### Integer Linear Programming
**Rejected**: Requires conversion to QUBO anyway for quantum backends

### Graph Coloring Formulation
**Partially Adopted**: Used as inspiration for conflict constraint modeling