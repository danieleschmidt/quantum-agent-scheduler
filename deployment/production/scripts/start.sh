#!/bin/bash

# Quantum Agent Scheduler Production Startup Script
set -euo pipefail

# Configuration
APP_NAME="quantum-scheduler"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
WORKERS="${WORKERS:-4}"
PORT="${PORT:-8080}"
METRICS_PORT="${METRICS_PORT:-9090}"
ENVIRONMENT="${ENVIRONMENT:-production}"

echo "🚀 Starting Quantum Agent Scheduler in ${ENVIRONMENT} mode"
echo "📊 Configuration:"
echo "  - Workers: ${WORKERS}"
echo "  - Port: ${PORT}"
echo "  - Metrics Port: ${METRICS_PORT}"
echo "  - Log Level: ${LOG_LEVEL}"

# Wait for dependencies to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo "⏳ Waiting for ${service_name} at ${host}:${port}..."
    
    for i in {1..30}; do
        if nc -z "${host}" "${port}" 2>/dev/null; then
            echo "✅ ${service_name} is ready"
            return 0
        fi
        echo "   Attempt ${i}/30: ${service_name} not ready, waiting..."
        sleep 2
    done
    
    echo "❌ ${service_name} failed to become ready after 60 seconds"
    return 1
}

# Check for required services
if [[ -n "${POSTGRES_URL:-}" ]]; then
    postgres_host=$(echo "$POSTGRES_URL" | cut -d'@' -f2 | cut -d':' -f1)
    postgres_port=$(echo "$POSTGRES_URL" | cut -d':' -f4 | cut -d'/' -f1)
    wait_for_service "$postgres_host" "$postgres_port" "PostgreSQL"
fi

if [[ -n "${REDIS_URL:-}" ]]; then
    redis_host=$(echo "$REDIS_URL" | cut -d'/' -f3 | cut -d':' -f1)
    redis_port=$(echo "$REDIS_URL" | cut -d':' -f3)
    wait_for_service "$redis_host" "$redis_port" "Redis"
fi

# Run database migrations if needed
if [[ -n "${POSTGRES_URL:-}" ]]; then
    echo "🗄️  Running database migrations..."
    python -m quantum_scheduler.db.migrations migrate --auto || {
        echo "⚠️  Database migration failed, but continuing..."
    }
fi

# Pre-flight checks
echo "🔍 Running pre-flight checks..."
python -m quantum_scheduler.health --check-all || {
    echo "❌ Pre-flight checks failed"
    exit 1
}

# Initialize quantum providers
echo "⚛️  Initializing quantum providers..."
python -c "
import sys
sys.path.insert(0, 'src')
from quantum_scheduler.cloud import CloudOrchestrator
orchestrator = CloudOrchestrator()
print(f'✅ Initialized with {len(orchestrator.available_resources)} quantum resources')
"

# Create PID directory
mkdir -p /app/tmp

# Function to handle graceful shutdown
cleanup() {
    echo "🛑 Received shutdown signal, gracefully shutting down..."
    
    # Kill all child processes
    if [[ -n "${MAIN_PID:-}" ]]; then
        echo "   Stopping main process (PID: $MAIN_PID)..."
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$MAIN_PID" 2>/dev/null; then
                echo "✅ Main process stopped gracefully"
                break
            fi
            echo "   Waiting for graceful shutdown ($i/10)..."
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            echo "⚠️  Forcing shutdown..."
            kill -KILL "$MAIN_PID" 2>/dev/null || true
        fi
    fi
    
    echo "✅ Shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the application
echo "🎯 Starting Quantum Agent Scheduler..."

if [[ "${ENVIRONMENT}" == "production" ]]; then
    # Production mode with Gunicorn
    exec gunicorn \
        --name "$APP_NAME" \
        --bind "0.0.0.0:$PORT" \
        --workers "$WORKERS" \
        --worker-class "uvicorn.workers.UvicornWorker" \
        --worker-connections 1000 \
        --max-requests 10000 \
        --max-requests-jitter 1000 \
        --timeout 300 \
        --keep-alive 30 \
        --log-level "$LOG_LEVEL" \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --pid /app/tmp/quantum-scheduler.pid \
        --preload \
        "quantum_scheduler.api.main:app" &
        
    MAIN_PID=$!
    
    # Start metrics server in background
    echo "📊 Starting metrics server on port $METRICS_PORT..."
    python -m quantum_scheduler.monitoring.metrics_server \
        --port "$METRICS_PORT" \
        --log-level "$LOG_LEVEL" &
        
    METRICS_PID=$!
    
    # Monitor processes
    echo "✅ Quantum Agent Scheduler started successfully"
    echo "   Main PID: $MAIN_PID"
    echo "   Metrics PID: $METRICS_PID"
    
    # Wait for main process
    wait $MAIN_PID
    
else
    # Development mode
    echo "🔧 Running in development mode"
    exec python -m quantum_scheduler.cli serve \
        --host "0.0.0.0" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --reload &
    
    MAIN_PID=$!
    wait $MAIN_PID
fi