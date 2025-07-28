# ADR-0001: Quantum Backend Abstraction

## Status
Accepted

## Context
The quantum computing landscape includes multiple providers (AWS Braket, IBM Quantum, Azure Quantum, D-Wave) with different APIs, authentication methods, and quantum computing paradigms (gate-based vs annealing). We need a unified interface that allows the scheduler to work with any quantum backend without vendor lock-in.

## Decision
Implement a backend abstraction layer with the following design:

1. **Common Interface**: All backends implement a shared `QuantumBackend` interface
2. **Provider-Specific Adapters**: Each quantum provider has a dedicated adapter class
3. **Automatic Provider Detection**: System can auto-detect available providers and capabilities
4. **Graceful Degradation**: Fallback to simulators when hardware unavailable
5. **Cost Awareness**: Backends report estimated costs before execution

```python
class QuantumBackend(ABC):
    @abstractmethod
    def solve_qubo(self, Q: np.ndarray, **kwargs) -> QuantumResult
    
    @abstractmethod
    def estimate_cost(self, problem_size: int) -> float
    
    @abstractmethod
    def is_available(self) -> bool
```

## Consequences

### Positive
- **Vendor Neutrality**: Easy switching between quantum providers
- **Future-Proof**: New providers can be added without core changes
- **Development Flexibility**: Can use simulators during development, hardware in production
- **Cost Control**: Transparent cost estimation across providers

### Negative
- **Abstraction Overhead**: Some provider-specific optimizations may be lost
- **Complexity**: Additional layer increases system complexity
- **Testing Burden**: Must test against multiple provider implementations

### Risks
- **Provider API Changes**: Backend updates may break adapters
- **Performance Variance**: Different providers may have significant performance differences
- **Feature Parity**: Not all features available on all providers

## Alternatives Considered

### Direct Provider Integration
**Rejected**: Would create tight coupling and make multi-provider support difficult

### Plugin Architecture
**Rejected**: Too complex for the initial implementation, though may be considered for future versions

### Provider-Specific Builds
**Rejected**: Would require maintaining separate distributions for each quantum provider