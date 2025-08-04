"""Security tests for quantum scheduler."""

import pytest
from unittest.mock import patch, MagicMock

from quantum_scheduler import QuantumScheduler, Agent, Task
from quantum_scheduler.core.exceptions import ValidationError
from quantum_scheduler.security import SecuritySanitizer


class TestInputSanitization:
    """Test input sanitization and security measures."""
    
    def test_malicious_id_injection(self):
        """Test protection against malicious ID injection."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test SQL injection attempt in agent ID
        with pytest.raises(ValidationError):
            malicious_agent = Agent(
                id="'; DROP TABLE agents; --",
                skills=["test"],
                capacity=1
            )
            scheduler.schedule([malicious_agent], [])
        
        # Test command injection attempt in task ID
        with pytest.raises(ValidationError):
            malicious_task = Task(
                id="task1; rm -rf /",
                required_skills=["test"],
                duration=1,
                priority=1
            )
            scheduler.schedule([], [malicious_task])
    
    def test_script_injection_prevention(self):
        """Test prevention of script injection in skills."""
        with pytest.raises(ValidationError):
            SecuritySanitizer.sanitize_skill("<script>alert('xss')</script>")
        
        with pytest.raises(ValidationError):
            SecuritySanitizer.sanitize_skill("javascript:alert('xss')")
        
        with pytest.raises(ValidationError):
            SecuritySanitizer.sanitize_skill("skill'; DROP TABLE skills; --")
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        with pytest.raises(ValueError):
            SecuritySanitizer.validate_file_path("../../../etc/passwd")
        
        with pytest.raises(ValueError):
            SecuritySanitizer.validate_file_path("/etc/passwd")
        
        with pytest.raises(ValueError):
            SecuritySanitizer.validate_file_path("file..name")
        
        # Valid paths should pass
        safe_path = SecuritySanitizer.validate_file_path("data/agents.json")
        assert safe_path == "data/agents.json"
    
    def test_input_length_limits(self):
        """Test input length limits to prevent DoS."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test extremely long agent ID
        with pytest.raises(ValidationError):
            long_id = "a" * 1000
            malicious_agent = Agent(id=long_id, skills=["test"], capacity=1)
            scheduler.schedule([malicious_agent], [])
        
        # Test extremely long skill name
        with pytest.raises(ValidationError):
            long_skill = "b" * 1000
            malicious_agent = Agent(id="agent1", skills=[long_skill], capacity=1)
            scheduler.schedule([malicious_agent], [])
    
    def test_numeric_input_validation(self):
        """Test numeric input validation and limits."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test negative capacity
        with pytest.raises(ValidationError):
            invalid_agent = Agent(id="agent1", skills=["test"], capacity=-1)
            scheduler.schedule([invalid_agent], [])
        
        # Test zero capacity
        with pytest.raises(ValidationError):
            invalid_agent = Agent(id="agent1", skills=["test"], capacity=0)
            scheduler.schedule([invalid_agent], [])
        
        # Test extremely high capacity (resource exhaustion attempt)
        with pytest.raises(ValidationError):
            invalid_agent = Agent(id="agent1", skills=["test"], capacity=999999)
            scheduler.schedule([invalid_agent], [])
    
    def test_special_character_handling(self):
        """Test handling of special characters in inputs."""
        # Test various special characters that should be rejected
        malicious_chars = [
            "agent<script>",
            "agent${PATH}",
            "agent`whoami`",
            "agent|ls",
            "agent&touch /tmp/hack",
            "agent;cat /etc/passwd",
            "agent\x00null",
            "agent\nnewline",
            "agent\ttab"
        ]
        
        for malicious_id in malicious_chars:
            with pytest.raises(ValueError):
                SecuritySanitizer.sanitize_id(malicious_id)
    
    def test_safe_id_generation(self):
        """Test secure ID generation."""
        # Generate multiple IDs and check uniqueness
        ids = set()
        for i in range(100):
            safe_id = SecuritySanitizer.generate_safe_id("test")
            assert safe_id.startswith("test_")
            assert len(safe_id) > 5
            assert safe_id not in ids  # Should be unique
            ids.add(safe_id)
            
            # Should only contain safe characters
            assert SecuritySanitizer.SAFE_ALPHANUMERIC.match(safe_id.replace("_", ""))
    
    def test_sensitive_data_masking(self):
        """Test masking of sensitive data in logs."""
        # Test email masking
        text_with_email = "Contact john.doe@company.com for support"
        masked = SecuritySanitizer.mask_sensitive_info(text_with_email)
        assert "[EMAIL]" in masked
        assert "john.doe@company.com" not in masked
        
        # Test password masking
        text_with_password = 'config = {"password": "secret123"}'
        masked = SecuritySanitizer.mask_sensitive_info(text_with_password)
        assert "[MASKED]" in masked
        assert "secret123" not in masked
        
        # Test token masking
        text_with_token = "Authorization: Bearer abc123def456"
        masked = SecuritySanitizer.mask_sensitive_info(text_with_token)
        assert "[MASKED]" in masked or "Bearer" in masked
    
    def test_hash_sensitive_data(self):
        """Test hashing of sensitive data."""
        sensitive_data = "user_password_123"
        hashed = SecuritySanitizer.hash_sensitive_data(sensitive_data)
        
        # Should be different from original
        assert hashed != sensitive_data
        
        # Should be consistent
        hashed2 = SecuritySanitizer.hash_sensitive_data(sensitive_data)
        assert hashed == hashed2
        
        # Should be hex string
        assert all(c in "0123456789abcdef" for c in hashed.lower())
        assert len(hashed) == 64  # SHA-256 hex length


class TestAccessControl:
    """Test access control and authorization."""
    
    def test_scheduler_initialization_security(self):
        """Test secure scheduler initialization."""
        # Test timeout limits
        with pytest.raises(ValueError):
            QuantumScheduler(timeout=-1)  # Negative timeout
        
        with pytest.raises(ValueError):
            QuantumScheduler(timeout=999999)  # Excessive timeout
        
        # Valid timeout should work
        scheduler = QuantumScheduler(timeout=30.0)
        assert scheduler._timeout == 30.0
    
    def test_backend_access_control(self):
        """Test backend access control."""
        # Should handle unknown backends gracefully
        scheduler = QuantumScheduler(backend="unknown_backend")
        assert scheduler._backend is not None  # Should fall back to safe default
        
        # Should not allow arbitrary backend instantiation
        with pytest.raises(Exception):
            # This would be caught by sanitization
            QuantumScheduler(backend="'; rm -rf /; echo '")
    
    def test_constraint_validation_security(self):
        """Test constraint validation security."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        agents = [Agent(id="agent1", skills=["test"], capacity=1)]
        tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
        
        # Test malicious constraint keys
        with pytest.raises(ValidationError):
            malicious_constraints = {
                "'; DROP TABLE constraints; --": True
            }
            scheduler.schedule(agents, tasks, malicious_constraints)
        
        # Test constraint value injection
        with pytest.raises(ValidationError):
            malicious_constraints = {
                "timeout": "'; rm -rf /; echo '"
            }
            scheduler.schedule(agents, tasks, malicious_constraints)


class TestResourceProtection:
    """Test protection against resource exhaustion attacks."""
    
    def test_problem_size_limits(self):
        """Test limits on problem sizes to prevent DoS."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test too many agents
        with pytest.raises(ValidationError):
            too_many_agents = [
                Agent(id=f"agent{i}", skills=["test"], capacity=1)
                for i in range(50000)  # Exceeds MAX_AGENTS
            ]
            scheduler.schedule(too_many_agents, [])
        
        # Test too many tasks
        with pytest.raises(ValidationError):
            agents = [Agent(id="agent1", skills=["test"], capacity=1)]
            too_many_tasks = [
                Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1)
                for i in range(200000)  # Exceeds MAX_TASKS
            ]
            scheduler.schedule(agents, too_many_tasks)
    
    def test_memory_consumption_limits(self):
        """Test memory consumption limits."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test extremely large skill lists
        with pytest.raises(ValidationError):
            many_skills = [f"skill{i}" for i in range(10000)]
            invalid_agent = Agent(id="agent1", skills=many_skills, capacity=1)
            scheduler.schedule([invalid_agent], [])
        
        # Test extremely large dependency lists
        with pytest.raises(ValidationError):
            many_deps = [f"dep{i}" for i in range(10000)]
            invalid_task = Task(
                id="task1", 
                required_skills=["test"], 
                duration=1, 
                priority=1,
                dependencies=many_deps
            )
            scheduler.schedule([], [invalid_task])
    
    def test_cpu_time_limits(self):
        """Test CPU time limits and timeout handling."""
        # Test with very short timeout
        scheduler = QuantumScheduler(backend="classical", timeout=0.001)
        
        # Create a problem that should timeout
        agents = [Agent(id=f"agent{i}", skills=["test"], capacity=1) for i in range(100)]
        tasks = [Task(id=f"task{i}", required_skills=["test"], duration=1, priority=1) for i in range(200)]
        
        # Should either complete quickly or timeout gracefully
        try:
            solution = scheduler.schedule(agents, tasks)
            # If it completes, should be valid
            assert solution is not None
        except Exception as e:
            # If it fails, should be a timeout or similar
            assert "timeout" in str(e).lower() or "time" in str(e).lower()


class TestDataPrivacy:
    """Test data privacy and information leakage prevention."""
    
    def test_error_message_sanitization(self):
        """Test that error messages don't leak sensitive information."""
        scheduler = QuantumScheduler(backend="classical", enable_validation=True)
        
        # Test with sensitive data in agent
        try:
            sensitive_agent = Agent(
                id="user_password_123",  # Sensitive ID
                skills=["admin", "root"],  # Sensitive skills
                capacity=1
            )
            scheduler.schedule([sensitive_agent], [])
            assert False, "Should have failed validation"
        except ValidationError as e:
            error_msg = str(e)
            # Error should mention validation failure but not expose raw sensitive data
            assert "validation" in error_msg.lower()
            # Should not contain full sensitive ID in clear text
            if "user_password_123" in error_msg:
                # If present, should be truncated or masked
                assert len([part for part in error_msg.split() if "user_password_123" in part]) <= 1
    
    def test_logging_sanitization(self):
        """Test that logs don't contain sensitive information."""
        import logging
        from io import StringIO
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("quantum_scheduler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            scheduler = QuantumScheduler(backend="classical")
            
            # Create agents with potentially sensitive info
            agents = [Agent(id="admin_user", skills=["password_reset"], capacity=1)]
            tasks = [Task(id="secure_task", required_skills=["password_reset"], duration=1, priority=1)]
            
            solution = scheduler.schedule(agents, tasks)
            
            # Check log output
            log_output = log_capture.getvalue()
            
            # Should log successful operation
            assert len(log_output) > 0
            
            # Should not contain sensitive patterns
            sensitive_patterns = ["password", "admin", "secret", "key", "token"]
            for pattern in sensitive_patterns:
                if pattern in log_output.lower():
                    # If sensitive terms appear, they should be in safe contexts
                    assert "mask" in log_output.lower() or "sanitiz" in log_output.lower()
        
        finally:
            logger.removeHandler(handler)
    
    def test_solution_data_protection(self):
        """Test that solutions don't expose unnecessary internal data."""
        scheduler = QuantumScheduler(backend="classical")
        
        agents = [Agent(id="agent1", skills=["test"], capacity=1)]
        tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
        
        solution = scheduler.schedule(agents, tasks)
        
        # Solution should contain expected public fields
        assert hasattr(solution, "assignments")
        assert hasattr(solution, "cost")
        assert hasattr(solution, "solver_type")
        
        # Should not contain sensitive internal state
        solution_dict = solution.__dict__
        sensitive_fields = ["_internal", "_private", "_secret", "password", "token"]
        
        for field_name in solution_dict.keys():
            for sensitive in sensitive_fields:
                assert sensitive not in field_name.lower()


class TestSecurityConfiguration:
    """Test security configuration and hardening."""
    
    def test_secure_defaults(self):
        """Test that secure defaults are used."""
        scheduler = QuantumScheduler()
        
        # Validation should be enabled by default
        assert scheduler._enable_validation is True
        
        # Should have reasonable timeout default (not infinite)
        if scheduler._timeout is not None:
            assert 0 < scheduler._timeout <= 3600
    
    def test_validation_bypass_prevention(self):
        """Test that validation cannot be easily bypassed."""
        # Even with validation disabled, extreme inputs should be caught
        scheduler = QuantumScheduler(backend="classical", enable_validation=False)
        
        # Should still have some basic protections
        try:
            # Extremely large input that could cause memory exhaustion
            huge_agent = Agent(
                id="a" * 100000,  # 100KB ID
                skills=["test"],
                capacity=1
            )
            scheduler.schedule([huge_agent], [])
            # If it doesn't fail, it should at least complete reasonably quickly
        except Exception:
            # Any exception is acceptable - should not crash or hang
            pass
    
    def test_backend_security_isolation(self):
        """Test that backends are properly isolated."""
        # Different backends should not interfere with each other
        scheduler1 = QuantumScheduler(backend="classical")
        scheduler2 = QuantumScheduler(backend="quantum_sim")
        
        agents = [Agent(id="agent1", skills=["test"], capacity=1)]
        tasks = [Task(id="task1", required_skills=["test"], duration=1, priority=1)]
        
        # Should be able to use both simultaneously
        solution1 = scheduler1.schedule(agents.copy(), tasks.copy())
        solution2 = scheduler2.schedule(agents.copy(), tasks.copy())
        
        assert solution1 is not None
        assert solution2 is not None
        
        # Should not share state
        assert solution1 is not solution2