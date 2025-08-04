"""Security utilities for input sanitization and validation."""

import re
import hashlib
import secrets
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


class SecuritySanitizer:
    """Provides security-focused input sanitization."""
    
    # Security patterns
    INJECTION_PATTERNS = [
        re.compile(r'[;&|`$(){}[\]\\<>]', re.IGNORECASE),  # Command injection
        re.compile(r'(select|insert|update|delete|drop|create|alter)\s+', re.IGNORECASE),  # SQL injection
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),  # XSS
        re.compile(r'javascript:', re.IGNORECASE),  # JavaScript injection
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'eval\s*\(', re.IGNORECASE),  # Code evaluation
        re.compile(r'exec\s*\(', re.IGNORECASE),  # Code execution
    ]
    
    # Safe character patterns
    SAFE_ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9_-]+$')
    SAFE_FILENAME = re.compile(r'^[a-zA-Z0-9_.-]+$')
    SAFE_EMAIL = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    # Limits
    MAX_STRING_LENGTH = 10000
    MAX_LIST_LENGTH = 10000
    MAX_DICT_SIZE = 1000
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = None) -> str:
        """Sanitize a string value.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If string contains malicious content
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value)}")
        
        # Check length
        max_len = max_length or cls.MAX_STRING_LENGTH
        if len(value) > max_len:
            logger.warning(f"String truncated from {len(value)} to {max_len} characters")
            value = value[:max_len]
        
        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"String contains potentially malicious content: {value[:50]}...")
        
        # Strip and normalize
        return value.strip()
    
    @classmethod
    def sanitize_id(cls, value: str) -> str:
        """Sanitize an ID string to ensure it's safe.
        
        Args:
            value: ID string to sanitize
            
        Returns:
            Sanitized ID
            
        Raises:
            ValueError: If ID is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"ID must be string, got {type(value)}")
        
        value = value.strip()
        
        if not value:
            raise ValueError("ID cannot be empty")
        
        if len(value) > 100:
            raise ValueError(f"ID too long: {len(value)} > 100")
        
        if not cls.SAFE_ALPHANUMERIC.match(value):
            raise ValueError(f"ID contains invalid characters: {value}")
        
        return value
    
    @classmethod
    def sanitize_skill(cls, value: str) -> str:
        """Sanitize a skill name.
        
        Args:
            value: Skill name to sanitize
            
        Returns:
            Sanitized skill name
            
        Raises:
            ValueError: If skill name is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"Skill must be string, got {type(value)}")
        
        value = value.strip().lower()
        
        if not value:
            raise ValueError("Skill name cannot be empty")
        
        if len(value) > 50:
            raise ValueError(f"Skill name too long: {len(value)} > 50")
        
        # Allow alphanumeric, underscore, hyphen, plus
        if not re.match(r'^[a-zA-Z0-9_+-]+$', value):
            raise ValueError(f"Skill name contains invalid characters: {value}")
        
        return value
    
    @classmethod
    def sanitize_number(cls, value: Union[int, float], min_val: float = None, max_val: float = None) -> Union[int, float]:
        """Sanitize a numeric value.
        
        Args:
            value: Number to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Sanitized number
            
        Raises:
            ValueError: If number is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected number, got {type(value)}")
        
        if not (isinstance(value, bool) or isinstance(value, (int, float))):
            raise ValueError(f"Invalid numeric type: {type(value)}")
        
        # Check for special float values
        if isinstance(value, float):
            if not (value == value):  # NaN check
                raise ValueError("NaN values not allowed")
            if value == float('inf') or value == float('-inf'):
                raise ValueError("Infinite values not allowed")
        
        # Check bounds
        if min_val is not None and value < min_val:
            raise ValueError(f"Value {value} below minimum {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"Value {value} above maximum {max_val}")
        
        return value
    
    @classmethod
    def sanitize_list(cls, value: List[Any], max_length: int = None, item_sanitizer=None) -> List[Any]:
        """Sanitize a list.
        
        Args:
            value: List to sanitize
            max_length: Maximum allowed length
            item_sanitizer: Function to sanitize individual items
            
        Returns:
            Sanitized list
            
        Raises:
            ValueError: If list is invalid
        """
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value)}")
        
        max_len = max_length or cls.MAX_LIST_LENGTH
        if len(value) > max_len:
            raise ValueError(f"List too long: {len(value)} > {max_len}")
        
        if item_sanitizer:
            try:
                return [item_sanitizer(item) for item in value]
            except Exception as e:
                raise ValueError(f"Failed to sanitize list item: {e}")
        
        return value
    
    @classmethod
    def sanitize_dict(cls, value: Dict[str, Any], max_size: int = None, key_sanitizer=None, value_sanitizer=None) -> Dict[str, Any]:
        """Sanitize a dictionary.
        
        Args:
            value: Dictionary to sanitize
            max_size: Maximum allowed size
            key_sanitizer: Function to sanitize keys
            value_sanitizer: Function to sanitize values
            
        Returns:
            Sanitized dictionary
            
        Raises:
            ValueError: If dictionary is invalid
        """
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")
        
        max_sz = max_size or cls.MAX_DICT_SIZE
        if len(value) > max_sz:
            raise ValueError(f"Dictionary too large: {len(value)} > {max_sz}")
        
        sanitized = {}
        
        for k, v in value.items():
            # Sanitize key
            if key_sanitizer:
                try:
                    k = key_sanitizer(k)
                except Exception as e:
                    raise ValueError(f"Failed to sanitize dict key {k}: {e}")
            
            # Sanitize value
            if value_sanitizer:
                try:
                    v = value_sanitizer(v)
                except Exception as e:
                    raise ValueError(f"Failed to sanitize dict value for key {k}: {e}")
            
            sanitized[k] = v
        
        return sanitized
    
    @classmethod
    def generate_safe_id(cls, prefix: str = "id", length: int = 8) -> str:
        """Generate a cryptographically secure random ID.
        
        Args:
            prefix: Prefix for the ID
            length: Length of random part
            
        Returns:
            Secure random ID
        """
        random_part = secrets.token_urlsafe(length)[:length]
        return f"{prefix}_{random_part}"
    
    @classmethod
    def hash_sensitive_data(cls, data: str) -> str:
        """Hash sensitive data for logging/storage.
        
        Args:
            data: Sensitive data to hash
            
        Returns:
            SHA-256 hash of the data
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @classmethod
    def mask_sensitive_info(cls, text: str, patterns: List[str] = None) -> str:
        """Mask sensitive information in text.
        
        Args:
            text: Text potentially containing sensitive info
            patterns: Custom patterns to mask
            
        Returns:
            Text with sensitive info masked
        """
        if not isinstance(text, str):
            return str(text)
        
        # Default patterns to mask
        default_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
            (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]'),  # Credit card
            (r'(?i)password["\s]*[:=]["\s]*[^\s"]+', 'password="[MASKED]"'),  # Password
            (r'(?i)token["\s]*[:=]["\s]*[^\s"]+', 'token="[MASKED]"'),  # Token
            (r'(?i)key["\s]*[:=]["\s]*[^\s"]+', 'key="[MASKED]"'),  # API key
        ]
        
        # Apply patterns
        masked_text = text
        for pattern, replacement in default_patterns:
            masked_text = re.sub(pattern, replacement, masked_text)
        
        # Apply custom patterns
        if patterns:
            for pattern in patterns:
                masked_text = re.sub(pattern, '[MASKED]', masked_text)
        
        return masked_text
    
    @classmethod
    def validate_file_path(cls, path: str) -> str:
        """Validate and sanitize a file path.
        
        Args:
            path: File path to validate
            
        Returns:
            Sanitized path
            
        Raises:
            ValueError: If path is unsafe
        """
        if not isinstance(path, str):
            raise ValueError(f"Path must be string, got {type(path)}")
        
        path = path.strip()
        
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            raise ValueError(f"Unsafe path detected: {path}")
        
        # Validate filename characters
        import os
        filename = os.path.basename(path)
        if not cls.SAFE_FILENAME.match(filename):
            raise ValueError(f"Invalid filename: {filename}")
        
        return path