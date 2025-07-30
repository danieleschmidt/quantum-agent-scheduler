#!/usr/bin/env python3
"""
Validate quantum backend configurations for quantum-agent-scheduler.

This script ensures that quantum backend configurations are properly formatted
and contain all required fields for each supported quantum provider.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Required fields for each quantum backend
BACKEND_SCHEMAS = {
    'qiskit': {
        'required_fields': ['backend_name', 'hub', 'group', 'project'],
        'optional_fields': ['shots', 'optimization_level', 'seed_simulator']
    },
    'braket': {
        'required_fields': ['device_arn', 's3_bucket', 'region'],
        'optional_fields': ['shots', 'device_parameters', 'tags']
    },
    'dwave': {
        'required_fields': ['solver', 'region'],
        'optional_fields': ['num_reads', 'annealing_time', 'chain_strength']
    },
    'azure': {
        'required_fields': ['resource_id', 'location', 'provider'],
        'optional_fields': ['target', 'shots', 'job_timeout']
    }
}

def validate_json_config(file_path: Path) -> List[str]:
    """Validate JSON configuration file."""
    errors = []
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"{file_path}: Invalid JSON format - {e}")
        return errors
    except Exception as e:
        errors.append(f"{file_path}: Error reading file - {e}")
        return errors
    
    # Validate structure if it's a quantum backend config
    if 'backends' in config:
        for backend_name, backend_config in config['backends'].items():
            if backend_name in BACKEND_SCHEMAS:
                schema = BACKEND_SCHEMAS[backend_name]
                
                # Check required fields
                for field in schema['required_fields']:
                    if field not in backend_config:
                        errors.append(
                            f"{file_path}: Missing required field '{field}' "
                            f"for {backend_name} backend"
                        )
                
                # Check for unknown fields
                all_valid_fields = set(schema['required_fields'] + schema['optional_fields'])
                for field in backend_config:
                    if field not in all_valid_fields:
                        errors.append(
                            f"{file_path}: Unknown field '{field}' "
                            f"for {backend_name} backend"
                        )
    
    return errors

def validate_python_imports(file_path: Path) -> List[str]:
    """Validate Python backend imports and basic structure."""
    errors = []
    
    try:
        content = file_path.read_text()
    except Exception as e:
        errors.append(f"{file_path}: Error reading file - {e}")
        return errors
    
    # Check for proper quantum imports
    quantum_imports = {
        'qiskit': ['from qiskit', 'import qiskit'],
        'braket': ['from braket', 'import braket'],
        'dwave': ['from dwave', 'import dwave'],
        'azure': ['from azure.quantum', 'import azure.quantum']
    }
    
    file_name = file_path.name.lower()
    for backend, import_patterns in quantum_imports.items():
        if backend in file_name:
            if not any(pattern in content for pattern in import_patterns):
                errors.append(
                    f"{file_path}: Missing {backend} imports in backend file"
                )
    
    # Check for basic error handling
    if 'try:' in content and 'except' not in content:
        errors.append(f"{file_path}: Found try block without except clause")
    
    # Check for proper class structure in backend files
    if 'backend' in file_name.lower() and 'class' in content:
        if 'def __init__' not in content:
            errors.append(f"{file_path}: Backend class missing __init__ method")
        if 'def submit' not in content and 'def run' not in content:
            errors.append(f"{file_path}: Backend class missing submit/run method")
    
    return errors

def main():
    """Main validation function."""
    errors = []
    
    # Find all relevant files
    repo_root = Path(__file__).parent.parent
    config_files = list(repo_root.glob('config/**/*.json'))
    config_files.extend(repo_root.glob('config/**/*.yml'))
    config_files.extend(repo_root.glob('config/**/*.yaml'))
    
    backend_files = list(repo_root.glob('src/quantum_scheduler/backends/**/*.py'))
    
    # Validate JSON/YAML configs
    for config_file in config_files:
        if config_file.suffix == '.json':
            errors.extend(validate_json_config(config_file))
    
    # Validate Python backend files
    for backend_file in backend_files:
        if backend_file.name != '__init__.py':
            errors.extend(validate_python_imports(backend_file))
    
    # Report results
    if errors:
        print("❌ Quantum configuration validation failed:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("✅ All quantum configurations are valid")
        sys.exit(0)

if __name__ == '__main__':
    main()