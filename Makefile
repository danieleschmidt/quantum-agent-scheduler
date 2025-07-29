# Quantum Agent Scheduler - Development Makefile

.PHONY: help install install-dev clean test test-unit test-integration test-quantum lint format type-check security docs docs-serve build release docker-build docker-run

# Default target
help: ## Show this help message
	@echo "Quantum Agent Scheduler - Development Commands"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install production dependencies
	poetry install --only=main

install-dev: ## Install all dependencies including dev
	poetry install --with dev,docs,quantum
	poetry run pre-commit install

install-all: ## Install with all extras
	poetry install --with dev,docs,quantum --extras all

# Cleaning
clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Testing
test: ## Run all tests
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	poetry run pytest tests/unit/ -v

test-integration: ## Run integration tests
	poetry run pytest tests/integration/ -v

test-quantum: ## Run quantum backend tests (requires quantum access)
	poetry run pytest tests/ -v -m quantum --quantum

test-fast: ## Run fast tests (skip slow and quantum)
	poetry run pytest tests/ -v -m "not slow and not quantum"

test-coverage: ## Run tests with detailed coverage report
	poetry run pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Code Quality
lint: ## Run all linting checks
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

format: ## Format code with black and isort
	poetry run black src/ tests/
	poetry run isort src/ tests/

format-check: ## Check if code is properly formatted
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

type-check: ## Run type checking with mypy
	poetry run mypy src/

security: ## Run security scans
	poetry run bandit -r src/
	poetry run safety check

pre-commit: ## Run all pre-commit hooks
	poetry run pre-commit run --all-files

# Documentation
docs: ## Build documentation
	poetry run mkdocs build

docs-serve: ## Serve documentation locally
	poetry run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	poetry run mkdocs gh-deploy

# Building and Distribution
build: ## Build package for distribution
	poetry build

build-wheel: ## Build wheel only
	poetry build --format wheel

build-sdist: ## Build source distribution only
	poetry build --format sdist

release: ## Build and release to PyPI (requires POETRY_PYPI_TOKEN_PYPI)
	poetry build
	poetry publish

release-test: ## Release to Test PyPI
	poetry build
	poetry publish --repository testpypi

# Docker
docker-build: ## Build Docker image
	docker build -t quantum-agent-scheduler:latest .

docker-build-dev: ## Build development Docker image
	docker build --target development -t quantum-agent-scheduler:dev .

docker-run: ## Run Docker container
	docker-compose up quantum-scheduler-dev

docker-run-prod: ## Run production Docker container
	docker-compose up quantum-scheduler-prod

docker-stop: ## Stop all Docker containers
	docker-compose down

docker-logs: ## View Docker container logs
	docker-compose logs -f quantum-scheduler-dev

# Development Environment
dev-setup: ## Set up complete development environment
	$(MAKE) install-dev
	poetry shell
	@echo "Development environment ready! Run 'poetry shell' to activate."

dev-reset: ## Reset development environment
	$(MAKE) clean
	poetry env remove --all
	$(MAKE) install-dev

jupyter: ## Start Jupyter Lab for quantum experimentation
	docker-compose up jupyter

# Benchmarking and Performance
benchmark: ## Run performance benchmarks
	poetry run pytest tests/ -v --benchmark-only

profile: ## Run profiling on test suite
	poetry run python -m cProfile -o profile.stats -m pytest tests/
	poetry run python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Quantum Backend Testing
test-qiskit: ## Test Qiskit backend integration
	poetry run pytest tests/integration/test_quantum_backends.py::TestQuantumBackendIntegration::test_qiskit_backend_integration -v

test-braket: ## Test AWS Braket backend integration
	poetry run pytest tests/integration/test_quantum_backends.py::TestQuantumBackendIntegration::test_braket_backend_integration -v

test-dwave: ## Test D-Wave backend integration
	poetry run pytest tests/integration/test_quantum_backends.py::TestQuantumBackendIntegration::test_dwave_backend_integration -v

# Utilities
check-poetry: ## Verify poetry configuration
	poetry check

update-deps: ## Update all dependencies
	poetry update

lock-deps: ## Update poetry.lock without installing
	poetry lock --no-update

requirements: ## Export requirements.txt for compatibility
	poetry export --format requirements.txt --output requirements.txt --without-hashes

requirements-dev: ## Export dev requirements
	poetry export --format requirements.txt --output requirements-dev.txt --with dev --without-hashes

version: ## Show current version
	@poetry version

bump-patch: ## Bump patch version
	poetry version patch

bump-minor: ## Bump minor version
	poetry version minor

bump-major: ## Bump major version
	poetry version major

# CI/CD Simulation
ci-test: ## Simulate CI testing pipeline
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) test
	$(MAKE) build

# Monitoring and Health
health-check: ## Run health checks on the system
	poetry run python -c "import quantum_scheduler; print('✓ Package imports successfully')"
	poetry run python -c "import sys; print(f'✓ Python version: {sys.version}')"
	@echo "✓ Poetry version: $(shell poetry --version)"
	@echo "✓ Health check completed"

# Database and Infrastructure (for development)
db-start: ## Start development database
	docker-compose up -d postgres redis

db-stop: ## Stop development database
	docker-compose stop postgres redis

db-reset: ## Reset development database
	docker-compose down postgres redis
	docker volume rm quantum-agent-scheduler_postgres-data quantum-agent-scheduler_redis-data
	$(MAKE) db-start

# Monitoring
monitor-start: ## Start monitoring stack (Prometheus + Grafana)
	docker-compose up -d prometheus grafana

monitor-stop: ## Stop monitoring stack
	docker-compose stop prometheus grafana

# Development Workflow Shortcuts
quick-test: format lint type-check test-fast ## Quick development test cycle

full-check: clean install-dev lint type-check security test build ## Full pre-commit check

# Environment Information
env-info: ## Show development environment information
	@echo "=== Environment Information ==="
	@echo "Python: $(shell python --version)"
	@echo "Poetry: $(shell poetry --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Git Branch: $(shell git branch --show-current 2>/dev/null || echo 'N/A')"
	@echo "Git Status: $(shell git status --porcelain | wc -l) files changed"
	@echo "Virtual Environment: $(shell poetry env info --path)"
	@echo "Installed Packages: $(shell poetry show | wc -l) packages"