# Testing Guide

Comprehensive testing guide for the Quantum Agent Scheduler project.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Performance Testing](#performance-testing)
- [Quantum Backend Testing](#quantum-backend-testing)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)

## 🎯 Overview

Our testing strategy ensures reliability, performance, and correctness across:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing  
- **Performance Tests**: Benchmarking and scaling tests
- **Quantum Tests**: Real quantum hardware validation
- **End-to-End Tests**: Full workflow testing

## 🏗️ Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── __init__.py
├── unit/                    # Unit tests
│   ├── test_scheduler.py    # Core scheduler logic
│   ├── test_agents.py       # Agent management
│   ├── test_tasks.py        # Task handling
│   ├── test_constraints.py  # Constraint system
│   └── test_backends/       # Backend-specific tests
├── integration/             # Integration tests
│   ├── test_quantum_backends.py  # Backend integration
│   ├── test_api_endpoints.py     # API testing
│   └── test_framework_integration.py
├── performance/             # Performance tests
│   ├── test_benchmarks.py   # Benchmark suite
│   ├── test_scaling.py      # Scaling tests
│   └── test_memory_usage.py # Memory profiling
├── quantum/                 # Quantum hardware tests
│   ├── test_braket.py       # AWS Braket tests
│   ├── test_qiskit.py       # IBM Quantum tests
│   ├── test_azure.py        # Azure Quantum tests
│   └── test_dwave.py        # D-Wave tests
├── e2e/                     # End-to-end tests
│   ├── test_workflows.py    # Complete workflows
│   └── test_cli.py          # CLI testing
└── fixtures/                # Test data
    ├── quantum_backends.json
    ├── scheduling_problems.json
    └── test_data/
```

## 🚀 Running Tests

### Basic Test Execution

```bash
# Run all tests
npm test
# or
poetry run pytest

# Run specific test categories
npm run test:unit          # Unit tests only
npm run test:integration   # Integration tests only
npm run test:quantum       # Quantum hardware tests (requires --quantum flag)
```

### Advanced Test Options

```bash
# Run with coverage
npm run test:coverage

# Run performance tests
pytest tests/performance/ -v -m slow

# Run quantum tests (requires quantum credentials)
pytest tests/quantum/ -v --quantum

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/test_scheduler.py -v

# Run tests matching pattern
pytest -k "test_qubo" -v
```

### Test Markers

```bash
# Run only fast tests (default)
pytest -m "not slow and not quantum"

# Run slow tests
pytest -m slow

# Run quantum hardware tests
pytest -m quantum --quantum

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance
```

## 🧪 Test Categories

### Unit Tests
Test individual components in isolation:

```python
def test_agent_skill_matching():
    """Test agent skill matching logic."""
    agent = Agent(id="test", skills=["python", "ml"], capacity=2)
    
    assert agent.has_skill("python")
    assert agent.has_skill("ml")
    assert not agent.has_skill("java")
    assert agent.can_handle_task(Task(required_skills=["python"]))
```

### Integration Tests
Test component interactions:

```python
@pytest.mark.integration
def test_scheduler_backend_integration():
    """Test scheduler integration with quantum backends."""
    scheduler = QuantumScheduler(backend="ibm_quantum")
    solution = scheduler.schedule(agents, tasks, constraints)
    
    assert solution.solver_type == "quantum"
    assert solution.assignments is not None
```

### Performance Tests
Benchmark performance and scaling:

```python
@pytest.mark.slow
@pytest.mark.performance
def test_large_problem_scaling():
    """Test performance on large problem instances."""
    agents, tasks = generate_large_problem(1000, 5000)
    
    start_time = time.time()
    solution = scheduler.schedule(agents, tasks)
    duration = time.time() - start_time
    
    assert duration < 300  # Should complete within 5 minutes
    assert solution.quality > 0.8  # Good solution quality
```

### Quantum Tests
Test with real quantum hardware:

```python
@pytest.mark.quantum
def test_braket_integration():
    """Test AWS Braket integration."""
    if not has_braket_credentials():
        pytest.skip("AWS Braket credentials not configured")
    
    backend = BraketBackend(device="arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    solution = backend.solve(qubo_matrix, shots=100)
    
    assert solution.status == "completed"
    assert len(solution.samples) == 100
```

## ✍️ Writing Tests

### Test Naming Convention

```python
# Good test names
def test_agent_creation_with_valid_data():
def test_scheduler_handles_infeasible_problem():
def test_qubo_formulation_with_constraints():

# Avoid generic names
def test_agent():  # Too vague
def test_error():  # What kind of error?
```

### Test Structure (AAA Pattern)

```python
def test_task_priority_sorting():
    # Arrange
    tasks = [
        Task(id="low", priority=1),
        Task(id="high", priority=10),
        Task(id="medium", priority=5)
    ]
    
    # Act
    sorted_tasks = sort_tasks_by_priority(tasks)
    
    # Assert
    assert sorted_tasks[0].id == "high"
    assert sorted_tasks[1].id == "medium" 
    assert sorted_tasks[2].id == "low"
```

### Using Fixtures

```python
@pytest.fixture
def sample_scheduling_problem():
    """Create a standard scheduling problem for testing."""
    agents = [Agent(id=f"agent{i}", skills=["python"], capacity=2) for i in range(3)]
    tasks = [Task(id=f"task{i}", required_skills=["python"], duration=1) for i in range(5)]
    return agents, tasks

def test_basic_scheduling(sample_scheduling_problem):
    agents, tasks = sample_scheduling_problem
    scheduler = QuantumScheduler()
    solution = scheduler.schedule(agents, tasks)
    assert len(solution.assignments) == len(tasks)
```

### Parameterized Tests

```python
@pytest.mark.parametrize("backend_type,expected_solver", [
    ("classical", "gurobi"),
    ("quantum_sim", "qiskit_simulator"),
    ("braket", "aws_braket"),
])
def test_backend_selection(backend_type, expected_solver):
    scheduler = QuantumScheduler(backend=backend_type)
    assert scheduler.backend.solver_type == expected_solver
```

### Mocking External Dependencies

```python
@patch('quantum_scheduler.backends.braket.AwsDevice')
def test_braket_backend_without_aws(mock_device):
    """Test Braket backend without actual AWS calls."""
    mock_device.return_value.run.return_value = Mock(result=Mock(samples=[1,0,1]))
    
    backend = BraketBackend()
    result = backend.solve(qubo_matrix)
    
    assert result is not None
    mock_device.assert_called_once()
```

## ⚡ Performance Testing

### Benchmark Framework

```python
from quantum_scheduler.benchmarks import SchedulerBenchmark

def test_solver_comparison():
    benchmark = SchedulerBenchmark()
    
    results = benchmark.compare_solvers(
        problem_size=100,
        solvers=["classical", "quantum_sim", "quantum_hw"],
        metrics=["time", "quality", "cost"],
        runs=5
    )
    
    benchmark.plot_results(results, save_to="benchmark_results.png")
    benchmark.generate_report(results, "benchmark_report.html")
```

### Memory Profiling

```python
import tracemalloc

def test_memory_usage():
    tracemalloc.start()
    
    # Run memory-intensive operation
    agents, tasks = generate_large_problem(500, 1000)
    solution = scheduler.schedule(agents, tasks)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert memory usage is reasonable
    assert peak < 1024 * 1024 * 1024  # Less than 1GB
```

### Load Testing

```python
import concurrent.futures
import time

def test_concurrent_scheduling():
    """Test scheduler under concurrent load."""
    def schedule_problem():
        agents, tasks = generate_problem(10, 20)
        return scheduler.schedule(agents, tasks)
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(schedule_problem) for _ in range(50)]
        results = [f.result() for f in futures]
    
    duration = time.time() - start_time
    
    assert len(results) == 50
    assert all(r.status == "optimal" for r in results)
    assert duration < 60  # Should complete within 1 minute
```

## 🌐 Quantum Backend Testing

### Backend Credentials

```bash
# Set up environment variables for testing
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export QISKIT_IBM_TOKEN=your_token
export DWAVE_API_TOKEN=your_token
```

### Mock Quantum Responses

```python
@pytest.fixture
def mock_quantum_response():
    return {
        "braket": {
            "measurements": [[1,0,1], [0,1,0]],
            "measurement_counts": {"101": 50, "010": 50}
        },
        "qiskit": {
            "counts": {"000": 512, "111": 512},
            "job_id": "test_job_123"
        }
    }

def test_quantum_result_parsing(mock_quantum_response):
    backend = BraketBackend()
    solution = backend.parse_result(mock_quantum_response["braket"])
    
    assert solution.samples is not None
    assert len(solution.samples) == 2
```

### Quantum Test Configuration

```python
def pytest_configure(config):
    """Configure quantum test markers."""
    config.addinivalue_line("markers", "quantum: requires quantum hardware")
    config.addinivalue_line("markers", "braket: requires AWS Braket")
    config.addinivalue_line("markers", "qiskit: requires IBM Quantum")

def pytest_collection_modifyitems(config, items):
    """Skip quantum tests if not requested."""
    if not config.getoption("--quantum"):
        skip_quantum = pytest.mark.skip(reason="use --quantum to run")
        for item in items:
            if "quantum" in item.keywords:
                item.add_marker(skip_quantum)
```

## 📊 Test Coverage

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/quantum_scheduler"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

### Coverage Commands

```bash
# Generate coverage report
pytest --cov=src/quantum_scheduler --cov-report=html --cov-report=term

# Coverage with missing lines
pytest --cov=src/quantum_scheduler --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=src/quantum_scheduler --cov-fail-under=80
```

### Coverage Targets

- **Overall Coverage**: ≥ 80%
- **Core Modules**: ≥ 90%
- **Critical Paths**: 100%
- **Documentation**: All public APIs

## 🔄 Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install --with dev
        
    - name: Run unit tests
      run: poetry run pytest tests/unit/ -v
      
    - name: Run integration tests
      run: poetry run pytest tests/integration/ -v
      
    - name: Run coverage
      run: |
        poetry run pytest --cov=src/quantum_scheduler --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Quality Gates

- All tests must pass
- Coverage ≥ 80%
- No security vulnerabilities
- Performance within acceptable bounds
- Documentation up to date

## 🐛 Debugging Tests

### Common Issues

```python
# Debug failing tests
pytest tests/unit/test_scheduler.py::test_basic_scheduling -v -s --pdb

# Capture output
pytest -v -s --capture=no

# Show local variables on failure
pytest --tb=long

# Run last failed tests only
pytest --lf
```

### Test Data Inspection

```python
def test_with_debugging():
    """Example test with debugging helpers."""
    agents, tasks = generate_problem(5, 10)
    
    # Save test data for inspection
    import json
    with open("/tmp/test_data.json", "w") as f:
        json.dump({
            "agents": [a.to_dict() for a in agents],
            "tasks": [t.to_dict() for t in tasks]
        }, f, indent=2)
    
    solution = scheduler.schedule(agents, tasks)
    
    # Print solution for manual verification
    print(f"Solution: {solution.assignments}")
    print(f"Cost: {solution.cost}")
    
    assert solution.status == "optimal"
```

## 📚 Best Practices

1. **Write tests first** (TDD approach)
2. **Keep tests independent** - no shared state
3. **Use descriptive test names** - test intent should be clear
4. **Test edge cases** - empty inputs, large inputs, invalid data
5. **Mock external dependencies** - tests should be fast and reliable
6. **Maintain test data** - keep fixtures up to date
7. **Review test coverage** - aim for meaningful coverage, not just numbers
8. **Document complex tests** - explain the test scenario and expectations

## 🔗 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Quantum Testing Best Practices](https://qiskit.org/documentation/tutorials/circuits_advanced/01_advanced_circuits.html)