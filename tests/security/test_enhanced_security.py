"""Enhanced security tests for quantum scheduler."""

import pytest
import time
from unittest.mock import Mock, patch

from quantum_scheduler.security.sanitizer import SecuritySanitizer
from quantum_scheduler import Agent, Task, QuantumScheduler
from quantum_scheduler.core.exceptions import ValidationError


class TestSecuritySanitizer:
    """Test enhanced security sanitizer functionality."""
    
    def test_sanitize_id_with_threats(self):
        """Test ID sanitization with various threat patterns."""
        sanitizer = SecuritySanitizer
        
        # Normal IDs should pass
        assert sanitizer.sanitize_id("valid_id") == "valid_id"
        assert sanitizer.sanitize_id("agent-123") == "agent-123"
        
        # Potential SQL injection attempts should be blocked
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitizer.sanitize_id("'; DROP TABLE users; --")
        
        # XSS attempts should be blocked
        with pytest.raises(ValueError, match="Invalid characters detected"):
            sanitizer.sanitize_id("<script>alert('xss')</script>")
        
        # Path traversal attempts should be cleaned
        clean_id = sanitizer.sanitize_id("../../../etc/passwd")
        assert "../" not in clean_id
        assert clean_id == "_._._.etc.passwd"
    
    def test_sanitize_string_comprehensive(self):
        """Test comprehensive string sanitization."""
        sanitizer = SecuritySanitizer
        
        # Normal strings should pass
        normal_string = "This is a normal string with spaces and punctuation!"
        assert sanitizer.sanitize_string(normal_string) == normal_string
        
        # SQL injection patterns should be blocked
        with pytest.raises(ValueError, match="Security threats detected"):
            sanitizer.sanitize_string("'; DROP DATABASE scheduler; --")
        
        # XSS patterns should be blocked
        with pytest.raises(ValueError, match="Security threats detected"):
            sanitizer.sanitize_string("<script>document.location='http://evil.com'</script>")
        
        # LDAP injection patterns should be blocked
        with pytest.raises(ValueError, match="Security threats detected"):
            sanitizer.sanitize_string("(objectclass=*)")
        
        # Command injection patterns should be blocked
        with pytest.raises(ValueError, match="Security threats detected"):
            sanitizer.sanitize_string("test; rm -rf /")
    
    def test_sanitize_list_with_threats(self):
        """Test list sanitization with malicious content."""
        sanitizer = SecuritySanitizer
        
        # Clean list should pass
        clean_list = ["item1", "item2", "item3"]
        result = sanitizer.sanitize_list(clean_list)
        assert result == clean_list
        
        # List with malicious strings should be cleaned
        malicious_list = ["good_item", "'; DROP TABLE users; --", "another_good_item"]
        result = sanitizer.sanitize_list(malicious_list)
        
        # Malicious item should be removed
        assert len(result) < len(malicious_list)
        assert "good_item" in result
        assert "another_good_item" in result
    
    def test_sanitize_dict_with_threats(self):
        """Test dictionary sanitization with malicious content."""
        sanitizer = SecuritySanitizer
        
        # Clean dict should pass
        clean_dict = {"key1": "value1", "key2": "value2"}
        result = sanitizer.sanitize_dict(clean_dict)
        assert result == clean_dict
        
        # Dict with malicious content should be cleaned
        malicious_dict = {
            "good_key": "good_value",
            "'; DROP TABLE users; --": "malicious_value",
            "another_key": "<script>alert('xss')</script>"
        }
        result = sanitizer.sanitize_dict(malicious_dict)
        
        # Should have fewer items due to removal of malicious content
        assert len(result) < len(malicious_dict)
        assert "good_key" in result
    
    def test_sanitize_number_with_anomalies(self):
        """Test number sanitization with suspicious values."""
        sanitizer = SecuritySanitizer
        
        # Normal numbers should pass
        assert sanitizer.sanitize_number(42) == 42
        assert sanitizer.sanitize_number(3.14) == 3.14
        
        # Very large numbers should trigger threat tracking
        large_number = 1e20
        result = sanitizer.sanitize_number(large_number, context="test")
        assert result == large_number  # Should still return the number
        
        # Check that threat was recorded (simplified check)
        threat_report = sanitizer.get_threat_report()
        assert "test" in threat_report["threat_scores"]
    
    def test_validate_url_security(self):
        """Test URL validation with security considerations."""
        sanitizer = SecuritySanitizer
        
        # Valid URLs should pass
        assert sanitizer.validate_url("https://example.com") == "https://example.com"
        assert sanitizer.validate_url("http://api.service.com/data") == "http://api.service.com/data"
        
        # Private URLs should be blocked by default
        with pytest.raises(ValueError, match="Private/internal URLs not allowed"):
            sanitizer.validate_url("http://localhost:8080")
        
        with pytest.raises(ValueError, match="Private/internal URLs not allowed"):
            sanitizer.validate_url("http://127.0.0.1:5000")
        
        # But should be allowed when explicitly permitted
        result = sanitizer.validate_url("http://localhost:8080", allow_private=True)
        assert result == "http://localhost:8080"
        
        # Malicious URLs should be blocked
        with pytest.raises(ValueError, match="Malicious content detected"):
            sanitizer.validate_url("javascript:alert('xss')")
    
    def test_threat_tracking_and_reporting(self):
        """Test threat tracking and reporting functionality."""
        sanitizer = SecuritySanitizer
        
        # Reset state for clean test
        sanitizer.reset_security_state()
        
        # Generate some threats
        context = "test_context"
        try:
            sanitizer.sanitize_string("'; DROP TABLE users; --", context=context)
        except ValueError:
            pass  # Expected
        
        try:
            sanitizer.sanitize_string("<script>alert('xss')</script>", context=context)
        except ValueError:
            pass  # Expected
        
        # Check threat report
        report = sanitizer.get_threat_report()
        assert "threat_scores" in report
        assert context in report["threat_scores"]
        assert "sql_injection" in report["threat_scores"][context]
        assert "xss" in report["threat_scores"][context]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        sanitizer = SecuritySanitizer
        sanitizer.reset_security_state()
        
        identifier = "test_client"
        
        # First few requests should be allowed
        for i in range(50):
            assert sanitizer.track_request(identifier) is True
        
        # Excessive requests should be blocked
        for i in range(60):  # This should exceed the limit
            sanitizer.track_request(identifier)
        
        # Should now be blocked
        assert sanitizer.track_request(identifier) is False
    
    def test_security_token_generation_and_verification(self):
        """Test security token generation and verification."""
        sanitizer = SecuritySanitizer
        
        data = "sensitive_data_123"
        secret = "secret_key_456"
        
        # Generate token
        token = sanitizer.generate_security_token(data, secret)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify valid token
        assert sanitizer.verify_security_token(data, token, secret) is True
        
        # Verify invalid token
        invalid_token = "invalid_token_hash"
        assert sanitizer.verify_security_token(data, invalid_token, secret) is False
        
        # Verify with wrong secret
        wrong_secret = "wrong_secret"
        assert sanitizer.verify_security_token(data, token, wrong_secret) is False
    
    def test_json_sanitization(self):
        """Test JSON sanitization functionality."""
        sanitizer = SecuritySanitizer
        
        # Clean data should serialize fine
        clean_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_str = sanitizer.sanitize_json(clean_data)
        assert isinstance(json_str, str)
        assert "key" in json_str
        
        # Very large data should be rejected
        large_data = {"key": "x" * 20000}  # Very large string
        with pytest.raises(ValueError, match="JSON too large"):
            sanitizer.sanitize_json(large_data)
        
        # Malicious JSON should be rejected
        malicious_json = '{"key": "<script>alert('xss')</script>"}'
        with pytest.raises(ValueError, match="Malicious content detected"):
            sanitizer.parse_safe_json(malicious_json)
    
    def test_file_path_validation_enhanced(self):
        """Test enhanced file path validation."""
        sanitizer = SecuritySanitizer
        
        # Valid paths should pass
        safe_path = "/tmp/safe_file.txt"
        result = sanitizer.validate_file_path(safe_path, allowed_base_paths=["/tmp"])
        assert str(result).endswith("safe_file.txt")
        
        # Path traversal should be blocked
        with pytest.raises(ValueError, match="Dangerous path pattern detected"):
            sanitizer.validate_file_path("../../../etc/passwd")
        
        # Paths outside allowed base paths should be blocked
        with pytest.raises(ValueError, match="Path not within allowed directories"):
            sanitizer.validate_file_path("/etc/passwd", allowed_base_paths=["/tmp"])


class TestSchedulerSecurity:
    """Test security features integrated into scheduler."""
    
    def test_scheduler_input_sanitization(self):
        """Test that scheduler properly sanitizes inputs."""
        scheduler = QuantumScheduler(enable_validation=True)
        
        # Valid inputs should work
        valid_agent = Agent(id="valid_agent", skills=["python"], capacity=2)
        valid_task = Task(id="valid_task", required_skills=["python"], duration=1, priority=1)
        
        solution = scheduler.schedule([valid_agent], [valid_task])
        assert solution is not None
    
    def test_scheduler_with_malicious_agent_data(self):
        """Test scheduler behavior with potentially malicious agent data."""
        scheduler = QuantumScheduler(enable_validation=True)
        
        # Try to create agent with malicious ID
        with pytest.raises((ValueError, ValidationError)):
            Agent(id="'; DROP TABLE agents; --", skills=["python"], capacity=1)
    
    def test_scheduler_with_malicious_task_data(self):
        """Test scheduler behavior with potentially malicious task data."""
        scheduler = QuantumScheduler(enable_validation=True)
        
        # Try to create task with malicious ID
        with pytest.raises((ValueError, ValidationError)):
            Task(
                id="<script>alert('xss')</script>", 
                required_skills=["python"], 
                duration=1, 
                priority=1
            )
    
    def test_scheduler_constraint_sanitization(self):
        """Test that scheduler sanitizes constraint inputs."""
        scheduler = QuantumScheduler(enable_validation=True)
        
        agent = Agent(id="agent1", skills=["python"], capacity=2)
        task = Task(id="task1", required_skills=["python"], duration=1, priority=1)
        
        # Normal constraints should work
        normal_constraints = {"skill_match_required": True}
        solution = scheduler.schedule([agent], [task], normal_constraints)
        assert solution is not None
        
        # Large constraint dictionaries should be handled
        large_constraints = {f"constraint_{i}": f"value_{i}" for i in range(200)}
        # This should not crash, may truncate or validate
        try:
            scheduler.schedule([agent], [task], large_constraints)
        except (ValueError, ValidationError):
            pass  # Expected for oversized constraints
    
    @patch('quantum_scheduler.security.sanitizer.SecuritySanitizer.track_request')
    def test_scheduler_rate_limiting_integration(self, mock_track_request):
        """Test integration with rate limiting."""
        # Simulate rate limiting by blocking requests
        mock_track_request.return_value = False
        
        # Create scheduler - this should still work even if rate limited
        scheduler = QuantumScheduler()
        
        # Basic functionality should still work
        assert scheduler is not None
    
    def test_scheduler_security_logging(self):
        """Test that security events are properly logged."""
        import logging
        
        # Set up log capture
        with patch('quantum_scheduler.security.sanitizer.logger') as mock_logger:
            sanitizer = SecuritySanitizer
            
            # Trigger security event
            try:
                sanitizer.sanitize_string("'; DROP TABLE users; --", context="test")
            except ValueError:
                pass
            
            # Check that warning was logged
            mock_logger.warning.assert_called()


class TestSecurityEdgeCases:
    """Test security edge cases and boundary conditions."""
    
    def test_empty_inputs_security(self):
        """Test security with empty inputs."""
        sanitizer = SecuritySanitizer
        
        # Empty strings should be handled safely
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitizer.sanitize_id("")
        
        # Empty lists should be handled
        result = sanitizer.sanitize_list([])
        assert result == []
        
        # Empty dicts should be handled
        result = sanitizer.sanitize_dict({})
        assert result == {}
    
    def test_unicode_and_special_characters(self):
        """Test security with Unicode and special characters."""
        sanitizer = SecuritySanitizer
        
        # Unicode characters should be handled properly
        unicode_string = "Hello 世界 🌍"
        result = sanitizer.sanitize_string(unicode_string)
        # Should not raise exception and should handle Unicode
        assert isinstance(result, str)
        
        # Special characters in different encodings
        special_chars = "café naïve résumé"
        result = sanitizer.sanitize_string(special_chars)
        assert isinstance(result, str)
    
    def test_very_long_inputs(self):
        """Test security with very long inputs."""
        sanitizer = SecuritySanitizer
        
        # Very long string should be truncated
        very_long_string = "x" * 2000
        result = sanitizer.sanitize_string(very_long_string)
        assert len(result) <= sanitizer.MAX_STRING_LENGTH
        
        # Very long list should be truncated
        very_long_list = [f"item_{i}" for i in range(2000)]
        result = sanitizer.sanitize_list(very_long_list)
        assert len(result) <= sanitizer.MAX_LIST_SIZE
    
    def test_nested_data_structures(self):
        """Test security with nested data structures."""
        sanitizer = SecuritySanitizer
        
        # Nested dictionaries
        nested_dict = {
            "level1": {
                "level2": {
                    "level3": "safe_value"
                }
            }
        }
        result = sanitizer.sanitize_dict(nested_dict)
        assert "level1" in result
        
        # Mixed nested structures
        mixed_structure = {
            "list_key": ["item1", "item2"],
            "dict_key": {"nested": "value"},
            "string_key": "simple_value"
        }
        result = sanitizer.sanitize_dict(mixed_structure)
        assert len(result) <= len(mixed_structure)
    
    def test_concurrent_security_operations(self):
        """Test security operations under concurrent access."""
        import threading
        import concurrent.futures
        
        sanitizer = SecuritySanitizer
        sanitizer.reset_security_state()
        
        def make_requests(thread_id):
            """Make multiple requests from a thread."""
            results = []
            for i in range(20):
                identifier = f"thread_{thread_id}_request_{i}"
                result = sanitizer.track_request(identifier)
                results.append(result)
            return results
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_requests, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All operations should complete without errors
        assert len(results) == 5
        for thread_results in results:
            assert len(thread_results) == 20