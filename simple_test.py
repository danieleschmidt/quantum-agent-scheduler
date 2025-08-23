#!/usr/bin/env python3
"""Simple test script to validate core quantum scheduler functionality without external dependencies."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_models():
    """Test basic data models without external dependencies."""
    try:
        from quantum_scheduler.core.models import Agent, Task, Solution, SchedulingProblem
        
        # Test Agent creation
        agent = Agent(id="test_agent", skills=["python", "ml"], capacity=2)
        assert agent.id == "test_agent"
        assert len(agent.skills) == 2
        assert agent.capacity == 2
        print("✓ Agent model test passed")
        
        # Test Task creation
        task = Task(id="test_task", required_skills=["python"], duration=1, priority=5.0)
        assert task.id == "test_task"
        assert task.duration == 1
        assert task.priority == 5.0
        print("✓ Task model test passed")
        
        # Test Solution creation
        solution = Solution(assignments={"test_task": "test_agent"}, cost=10.0)
        assert solution.total_assignments == 1
        assert solution.cost == 10.0
        print("✓ Solution model test passed")
        
        # Test SchedulingProblem validation
        problem = SchedulingProblem(
            agents=[agent],
            tasks=[task],
            constraints={}
        )
        assert problem.validate() == True
        assert problem.total_capacity == 2
        assert problem.total_workload == 1
        print("✓ SchedulingProblem validation test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model tests failed: {e}")
        return False

def test_exceptions():
    """Test exception classes."""
    try:
        from quantum_scheduler.core.exceptions import (
            ValidationError, BackendError, SolverError
        )
        
        # Test that exceptions can be instantiated
        validation_error = ValidationError("Test validation error")
        backend_error = BackendError("Test backend error")
        solver_error = SolverError("Test solver error")
        
        assert str(validation_error) == "Test validation error"
        assert str(backend_error) == "Test backend error"
        assert str(solver_error) == "Test solver error"
        
        print("✓ Exception classes test passed")
        return True
        
    except Exception as e:
        print(f"✗ Exception tests failed: {e}")
        return False

def test_validators():
    """Test input validation logic."""
    try:
        from quantum_scheduler.core.validators import InputValidator
        from quantum_scheduler.core.models import Agent, Task, SchedulingProblem
        
        # Test valid input
        agents = [Agent(id="agent1", skills=["python"], capacity=2)]
        tasks = [Task(id="task1", required_skills=["python"], duration=1, priority=5.0)]
        problem = SchedulingProblem(agents=agents, tasks=tasks, constraints={})
        
        # Test validation method with correct signature
        validated_problem = InputValidator.validate_problem(problem)
        assert validated_problem is not None
        print("✓ Input validator test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Validator tests failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Running basic quantum scheduler tests...")
    print("=" * 50)
    
    tests = [
        ("Basic Models", test_basic_models),
        ("Exceptions", test_exceptions),
        ("Validators", test_validators),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        print("✅ Generation 1 (MAKE IT WORK) - Core functionality validated")
        return True
    else:
        print("❌ Some tests failed - core functionality needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)