# ADR-0003: Hybrid Solver Selection

## Status
Accepted

## Context
Not all scheduling problems benefit from quantum computing. Small problems may be solved faster classically, while quantum hardware access is expensive and limited. We need an intelligent strategy for selecting between classical and quantum solvers based on problem characteristics and available resources.

## Decision
Implement a multi-tier solver selection strategy:

### Tier 1: Classical Solvers (0-50 variables)
- **Solvers**: Gurobi, CPLEX, scipy.optimize
- **Criteria**: Problem size < 50 variables OR time limit < 10 seconds
- **Benefits**: Fast execution, no quantum resource costs

### Tier 2: Quantum Simulators (50-200 variables)
- **Solvers**: Qiskit Aer, Amazon Braket simulators
- **Criteria**: 50-200 variables AND classical solver timeout
- **Benefits**: Quantum algorithm testing without hardware costs

### Tier 3: Quantum Hardware (200+ variables)
- **Solvers**: D-Wave Advantage, IBM Quantum, IonQ
- **Criteria**: >200 variables OR quantum advantage demonstrated
- **Benefits**: True quantum speedup for large problems

### Selection Algorithm
1. **Problem Analysis**: Count variables, estimate complexity
2. **Resource Check**: Verify quantum backend availability and cost budget
3. **Performance History**: Use past results to predict optimal solver
4. **Timeout Strategy**: Fallback chain with increasing time limits

## Consequences

### Positive
- **Cost Efficiency**: Avoid expensive quantum calls for simple problems
- **Performance Optimization**: Each solver used in its optimal range
- **Reliability**: Multiple fallback options ensure solution delivery
- **Learning System**: Improves selection over time based on results

### Negative
- **Selection Overhead**: Analysis and decision logic adds latency
- **Complexity**: Multiple solver paths increase code complexity
- **Threshold Tuning**: Problem size thresholds need empirical validation

### Risks
- **Mis-classification**: Problems may be assigned to sub-optimal solvers
- **Resource Starvation**: High demand may exhaust quantum resources
- **Vendor Dependence**: Strategy relies on specific provider capabilities

## Alternatives Considered

### Single Solver Approach
**Rejected**: Misses optimization opportunities and quantum advantages

### User-Specified Solver
**Partially Adopted**: Available as override option for expert users

### Machine Learning Selection
**Future Consideration**: Too complex for initial implementation, planned for v2.0