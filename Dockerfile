# Multi-stage build for quantum-agent-scheduler
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Development stage
FROM base as development

# Install all dependencies including dev
RUN poetry install --with dev,docs,quantum && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Create non-root user
RUN groupadd -r scheduler && useradd -r -g scheduler scheduler
RUN chown -R scheduler:scheduler /app
USER scheduler

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "quantum_scheduler.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install only production dependencies
RUN poetry install --only=main --extras=all && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY src/ ./src/
COPY README.md LICENSE ./

# Create non-root user
RUN groupadd -r scheduler && useradd -r -g scheduler scheduler
RUN chown -R scheduler:scheduler /app
USER scheduler

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["poetry", "run", "quantum-scheduler", "serve", "--host", "0.0.0.0", "--port", "8000"]

# Quantum hardware stage (for quantum cloud deployment)
FROM production as quantum-hw

# Install quantum provider dependencies
USER root
RUN poetry install --extras="braket qiskit dwave azure" && rm -rf $POETRY_CACHE_DIR

# Add quantum-specific configurations
COPY config/quantum/ ./config/quantum/

USER scheduler

# Override with quantum-optimized settings
ENV QUANTUM_ENABLED=true
ENV FALLBACK_TO_CLASSICAL=true
ENV MAX_QUANTUM_COST=100.0

CMD ["poetry", "run", "quantum-scheduler", "serve", "--quantum-enabled", "--host", "0.0.0.0", "--port", "8000"]