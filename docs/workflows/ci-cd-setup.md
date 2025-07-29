# CI/CD Workflow Setup Guide

This document provides templates and setup instructions for GitHub Actions workflows for the quantum-agent-scheduler project.

## Required Workflows

### 1. Continuous Integration (CI)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: 1.6.1
        virtualenvs-create: true
        virtualenvs-in-project: true
        
    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
        
    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --with dev
      
    - name: Run pre-commit hooks
      run: |
        poetry run pre-commit run --all-files
        
    - name: Run tests
      run: |
        poetry run pytest tests/ -v --cov=src --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install --with dev
      
    - name: Run Bandit security scan
      run: poetry run bandit -r src/
      
    - name: Run Safety check
      run: poetry run safety check

  type-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install --with dev
      
    - name: Run MyPy type checking
      run: poetry run mypy src/

  build:
    runs-on: ubuntu-latest
    needs: [test, security, type-check]
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Build package
      run: poetry build
      
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

### 2. Security Scanning

Create `.github/workflows/security.yml`:

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
        
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install --with dev
      
    - name: Run Pip-Audit
      run: |
        pip install pip-audit
        pip-audit --requirement <(poetry export --format requirements.txt)
        
    - name: Run Safety
      run: poetry run safety check --json | tee safety-report.json
      
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          safety-report.json

  sbom-generation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Generate SBOM
      uses: anchore/sbom-action@v0
      with:
        path: .
        format: spdx-json
        
    - name: Upload SBOM
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.spdx.json
```

### 3. Release Automation

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Build package
      run: poetry build
      
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
        
    - name: Publish to PyPI
      env:
        POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
      run: poetry publish

  docker-release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: your-org/quantum-agent-scheduler
        tags: |
          type=ref,event=tag
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 4. Documentation Deployment

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install --with docs
      
    - name: Build documentation
      run: poetry run mkdocs build
      
    - name: Upload docs artifact
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: site/
        
  deploy-docs:
    runs-on: ubuntu-latest
    needs: build-docs
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install --with docs
      
    - name: Deploy to GitHub Pages
      run: poetry run mkdocs gh-deploy --force
```

## Environment Variables and Secrets

Configure these secrets in GitHub repository settings:

### Required Secrets
- `PYPI_TOKEN`: PyPI API token for package publishing
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

### Optional Secrets (for quantum backends)
- `IBM_QUANTUM_TOKEN`: IBM Quantum API token
- `AWS_ACCESS_KEY_ID`: AWS access key for Braket
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for Braket
- `DWAVE_API_TOKEN`: D-Wave API token
- `AZURE_QUANTUM_CONNECTION_STRING`: Azure Quantum connection string

## Branch Protection Rules

Configure these branch protection rules for `main` branch:

1. **Require pull request reviews**
   - Required approving reviews: 1
   - Dismiss stale reviews when new commits are pushed
   - Require review from code owners

2. **Require status checks to pass**
   - Require branches to be up to date
   - Required status checks:
     - `test (3.9)`
     - `test (3.10)`
     - `test (3.11)`
     - `test (3.12)`
     - `security`
     - `type-check`
     - `build`

3. **Restrict pushes to matching branches**
   - Include administrators: false
   - Allow force pushes: false
   - Allow deletions: false

## Monitoring and Notifications

### Slack Integration
Add webhook URL as `SLACK_WEBHOOK` secret for build notifications.

### Email Notifications
Configure via GitHub notification settings for security alerts and failed builds.

## Performance Optimization

### Cache Strategy
- Poetry dependencies cached by lock file hash
- Docker layer caching enabled for multi-stage builds
- Test results cached to speed up subsequent runs

### Parallel Execution
- Matrix builds for multiple Python versions
- Parallel job execution for independent tasks
- Conditional job execution based on file changes

## Compliance and Governance

### SLSA Compliance
Generate provenance attestations for releases using SLSA GitHub generator.

### SBOM Generation
Automatically generate Software Bill of Materials for security compliance.

### Audit Logging
All CI/CD actions logged and available for compliance auditing.

## Troubleshooting

### Common Issues
1. **Poetry lock file conflicts**: Update poetry.lock and commit
2. **Test failures in matrix builds**: Check Python version compatibility
3. **Security scan failures**: Review and update dependencies
4. **Docker build failures**: Check Dockerfile syntax and dependencies

### Debug Mode
Enable debug logging by setting `ACTIONS_STEP_DEBUG: true` in workflow environment.