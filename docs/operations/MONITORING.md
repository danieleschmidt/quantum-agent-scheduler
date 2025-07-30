# Monitoring and Observability

This document describes the monitoring and observability setup for quantum-agent-scheduler.

## Overview

The quantum-agent-scheduler includes comprehensive monitoring capabilities to track:
- Performance metrics (scheduling time, quantum execution time)
- Resource utilization (CPU, memory, quantum credits)
- Error rates and failure patterns
- Business metrics (problems solved, quantum advantage achieved)

## Metrics Collection

### Application Metrics

The scheduler exposes metrics in Prometheus format on `/metrics` endpoint:

```python
from quantum_scheduler.monitoring import MetricsCollector

# Initialize metrics collection
metrics = MetricsCollector()

# Track scheduling performance
with metrics.timing('scheduling_duration'):
    solution = scheduler.schedule(agents, tasks)

# Track quantum backend usage
metrics.increment('quantum_backend_calls', tags={'backend': 'qiskit'})
metrics.gauge('quantum_credits_used', credits_consumed)
```

### Key Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `scheduling_duration_seconds` | Histogram | Time to solve scheduling problem | `backend`, `problem_size` |
| `quantum_backend_calls_total` | Counter | Quantum backend API calls | `backend`, `status` |
| `quantum_credits_used` | Gauge | Quantum credits consumed | `backend`, `provider` |
| `problems_solved_total` | Counter | Successfully solved problems | `backend`, `algorithm` |
| `quantum_advantage_ratio` | Gauge | Speedup vs classical solver | `problem_size` |
| `error_rate` | Counter | Error occurrences | `error_type`, `component` |

## Alerting Rules

### Critical Alerts

```yaml
# High Error Rate
- alert: QuantumSchedulerHighErrorRate
  expr: rate(errors_total[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate in quantum scheduler"
    description: "Error rate is {{ $value }} errors/second"

# Quantum Backend Failure
- alert: QuantumBackendDown
  expr: quantum_backend_health == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Quantum backend {{ $labels.backend }} is down"

# Performance Regression
- alert: SchedulingPerformanceRegression
  expr: rate(scheduling_duration_seconds_sum[5m]) / rate(scheduling_duration_seconds_count[5m]) > 30
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Scheduling performance degraded"
    description: "Average scheduling time is {{ $value }}s"
```

### Warning Alerts

```yaml
# High Quantum Credit Usage
- alert: QuantumCreditsHighUsage
  expr: quantum_credits_used > 1000
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High quantum credit usage detected"

# Low Quantum Advantage
- alert: LowQuantumAdvantage
  expr: quantum_advantage_ratio < 1.5
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Quantum advantage below threshold"
```

## Dashboards

### Performance Dashboard

Key visualizations:
- Scheduling time distribution (histogram)
- Quantum vs classical performance comparison
- Backend usage patterns
- Problem size scaling trends

### Operational Dashboard

Key visualizations:
- Error rate trends
- Service health status
- Resource utilization
- Quantum credit consumption

### Business Dashboard

Key visualizations:
- Problems solved per day
- Quantum advantage achievements
- Cost per solution
- Customer usage patterns

## Tracing

### Distributed Tracing Setup

```python
from quantum_scheduler.tracing import setup_tracing

# Initialize tracing
setup_tracing(
    service_name="quantum-agent-scheduler",
    jaeger_endpoint="http://jaeger:14268/api/traces"
)

# Trace scheduling operations
@trace_operation("schedule_agents")
def schedule_with_tracing(agents, tasks, constraints):
    with trace_span("qubo_formulation") as span:
        qubo = formulate_qubo(agents, tasks, constraints)
        span.set_attribute("qubo_size", qubo.shape[0])
    
    with trace_span("quantum_execution") as span:
        solution = quantum_backend.solve(qubo)
        span.set_attribute("backend", quantum_backend.name)
        span.set_attribute("execution_time", solution.execution_time)
    
    return solution
```

### Key Trace Spans

- `schedule_agents` - Full scheduling operation
- `qubo_formulation` - QUBO matrix construction
- `constraint_processing` - Constraint validation and encoding
- `quantum_execution` - Quantum backend execution
- `solution_decoding` - Converting quantum results to assignments

## Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger("quantum_scheduler")

# Log scheduling start
logger.info(
    "scheduling_started",
    num_agents=len(agents),
    num_tasks=len(tasks),
    backend=backend_name,
    problem_id=problem_id
)

# Log quantum execution
logger.info(
    "quantum_execution_completed",
    backend=backend_name,
    execution_time=execution_time,
    shots=shots,
    success=success,
    problem_id=problem_id
)
```

### Log Levels

- **ERROR**: Failed operations, quantum backend errors, constraint violations
- **WARN**: Performance degradation, quantum credit warnings, fallback usage
- **INFO**: Scheduling operations, solution summaries, backend selections
- **DEBUG**: Detailed algorithm steps, intermediate results, timing details

## Health Checks

### Application Health

```python
from quantum_scheduler.health import HealthChecker

health_checker = HealthChecker()

# Add health checks
health_checker.add_check("quantum_backends", check_quantum_backends_health)
health_checker.add_check("database", check_database_connection)
health_checker.add_check("memory_usage", check_memory_usage)

# Health endpoint returns:
{
    "status": "healthy",
    "checks": {
        "quantum_backends": {"status": "healthy", "details": "All backends responsive"},
        "database": {"status": "healthy", "details": "Connection active"},
        "memory_usage": {"status": "warning", "details": "Memory usage at 85%"}
    },
    "timestamp": "2024-07-30T07:20:00Z"
}
```

### Quantum Backend Health

```python
def check_quantum_backend_health():
    """Check quantum backend connectivity and queue status."""
    results = {}
    
    for backend_name, backend in quantum_backends.items():
        try:
            # Test connectivity
            status = backend.get_status()
            queue_length = backend.get_queue_length()
            
            results[backend_name] = {
                "status": "healthy" if status.operational else "degraded",
                "queue_length": queue_length,
                "available_qubits": status.n_qubits if hasattr(status, 'n_qubits') else None
            }
        except Exception as e:
            results[backend_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return results
```

## Performance Analysis

### Automated Performance Reports

```python
# Generate daily performance report
def generate_performance_report():
    report = {
        "date": datetime.now().isoformat(),
        "problems_solved": get_problems_solved_count(),
        "average_scheduling_time": get_average_scheduling_time(),
        "quantum_advantage_stats": get_quantum_advantage_stats(),
        "backend_usage": get_backend_usage_stats(),
        "error_summary": get_error_summary()
    }
    
    # Store report
    store_performance_report(report)
    
    # Send to monitoring system
    send_metrics_to_monitoring(report)
```

### Regression Detection

```python
def detect_performance_regression():
    """Detect performance regressions using statistical analysis."""
    current_metrics = get_current_performance_metrics()
    historical_metrics = get_historical_performance_metrics(days=7)
    
    for metric_name, current_value in current_metrics.items():
        historical_values = historical_metrics[metric_name]
        
        # Statistical significance test
        if is_significant_regression(current_value, historical_values):
            alert_performance_regression(metric_name, current_value, historical_values)
```

## Incident Response

### Runbooks

1. **High Error Rate**: Check quantum backend status, verify credentials, examine logs
2. **Performance Degradation**: Check resource usage, quantum queue lengths, problem complexity
3. **Backend Failure**: Switch to backup backend, contact provider, update status page
4. **Memory Issues**: Check for memory leaks, restart service, scale resources

### On-Call Procedures

1. **Severity 1 (Critical)**: Immediate response, quantum backends down
2. **Severity 2 (High)**: 2-hour response, performance degradation
3. **Severity 3 (Medium)**: 8-hour response, feature issues
4. **Severity 4 (Low)**: Next business day, improvements

## Cost Monitoring

### Quantum Cost Tracking

```python
def track_quantum_costs():
    """Track quantum computing costs across backends."""
    costs = {}
    
    for backend_name, backend in quantum_backends.items():
        # Get usage statistics
        shots_used = get_shots_used(backend_name)
        execution_time = get_execution_time(backend_name)
        
        # Calculate costs based on provider pricing
        cost = calculate_backend_cost(backend_name, shots_used, execution_time)
        
        costs[backend_name] = {
            "total_cost": cost,
            "shots_used": shots_used,
            "execution_time": execution_time,
            "cost_per_shot": cost / shots_used if shots_used > 0 else 0
        }
    
    return costs
```

### Budget Alerts

```yaml
# Cost threshold alerts
- alert: QuantumCostThresholdExceeded
  expr: quantum_cost_daily > 500
  labels:
    severity: warning
  annotations:
    summary: "Daily quantum costs exceeded $500"

- alert: QuantumCostProjectionHigh
  expr: predict_linear(quantum_cost_daily[7d], 86400 * 23) > 10000
  labels:
    severity: critical
  annotations:
    summary: "Monthly quantum cost projection exceeds $10,000"
```