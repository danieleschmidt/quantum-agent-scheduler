#!/usr/bin/env python3
"""Debug error handling logic."""

# Mock classes for testing
class MockAgent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockTask:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def debug_scheduler(agents, tasks, test_name):
    """Debug the scheduler logic."""
    print(f"\n🔍 Debugging: {test_name}")
    print(f"  Agents: {[str(vars(a)) for a in agents]}")
    print(f"  Tasks: {[str(vars(t)) for t in tasks]}")
    
    # Check validation
    if not agents:
        print("  ❌ No agents - validation error")
        return "validation"
    if not tasks:
        print("  ❌ No tasks - validation error")
        return "validation"
    
    # Check capacity
    total_capacity = sum(getattr(a, 'capacity', 0) for a in agents)
    total_work = sum(getattr(t, 'duration', 0) for t in tasks)
    print(f"  Capacity check: {total_work} work vs {total_capacity} capacity")
    if total_work > total_capacity:
        print("  ❌ Capacity exceeded")
        return "capacity"
    
    # Check skills
    agent_skills = set()
    for agent in agents:
        agent_skills.update(getattr(agent, 'skills', []))
    
    required_skills = set()
    for task in tasks:
        required_skills.update(getattr(task, 'required_skills', []))
    
    missing_skills = required_skills - agent_skills
    print(f"  Skill check: agent_skills={agent_skills}, required={required_skills}, missing={missing_skills}")
    if missing_skills:
        print("  ❌ Skills missing")
        return "skills"
    
    print("  ✅ All checks passed")
    return "success"

# Test the problematic cases
test_cases = [
    ("Capacity exceeded", [MockAgent(capacity=5)], [MockTask(duration=10)]),
    ("Skill mismatch", [MockAgent(skills=["python"])], [MockTask(required_skills=["java"])])
]

for test_name, agents, tasks in test_cases:
    result = debug_scheduler(agents, tasks, test_name)
    print(f"  Result: {result}")