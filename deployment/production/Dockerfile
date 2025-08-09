# Multi-stage Dockerfile for Quantum Agent Scheduler Production Deployment

# =============================================================================
# Stage 1: Base Dependencies
# =============================================================================
FROM python:3.11-slim AS base

LABEL maintainer="Terragon Labs <quantum-ai@terragon.com>"
LABEL description="Quantum Agent Scheduler - Production-ready quantum scheduling system"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# =============================================================================
# Stage 2: Development Dependencies
# =============================================================================
FROM base AS development

# Install all dependencies including dev dependencies
RUN poetry install --with dev,quantum && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum
RUN chown -R quantum:quantum /app

USER quantum

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m quantum_scheduler.health --quick

# Default command
CMD ["poetry", "run", "python", "-m", "quantum_scheduler.cli", "serve"]

# =============================================================================
# Stage 3: Production Build
# =============================================================================
FROM base AS builder

# Install only production dependencies
RUN poetry install --only=main,quantum --no-dev && rm -rf $POETRY_CACHE_DIR

# =============================================================================
# Stage 4: Production Runtime
# =============================================================================
FROM python:3.11-slim AS production

# Production environment variables
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    PORT=8080 \
    METRICS_PORT=9090

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r quantum && useradd -r -g quantum -s /bin/false quantum

# Create application directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Make sure we use venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=quantum:quantum src/ ./src/
COPY --chown=quantum:quantum pyproject.toml ./
COPY --chown=quantum:quantum README.md ./

# Copy production configuration
COPY --chown=quantum:quantum deployment/production/config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/tmp && \
    chown -R quantum:quantum /app/logs /app/cache /app/tmp

# Install the package
RUN pip install -e .

# Switch to non-root user
USER quantum

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m quantum_scheduler.health --check-all

# Production startup script
COPY --chown=quantum:quantum deployment/production/scripts/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# =============================================================================
# Stage 5: Testing
# =============================================================================
FROM development AS testing

# Install additional test dependencies
RUN poetry install --with dev,quantum,test

# Copy test files
COPY tests/ ./tests/

# Run tests during build (optional - can be disabled for CI)
ARG RUN_TESTS=false
RUN if [ "$RUN_TESTS" = "true" ]; then \
    poetry run python -m pytest tests/ -v --tb=short || exit 1; \
    fi

# =============================================================================
# Labels for production image
# =============================================================================
LABEL org.opencontainers.image.title="Quantum Agent Scheduler" \
      org.opencontainers.image.description="Production-ready quantum scheduling system with multi-region cloud orchestration" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/terragon-labs/quantum-agent-scheduler" \
      org.opencontainers.image.documentation="https://docs.terragon.com/quantum-scheduler"