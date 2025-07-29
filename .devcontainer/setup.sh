#!/bin/bash

# Development container setup script

set -e

echo "🚀 Setting up Quantum Agent Scheduler development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install --with dev,docs,quantum

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
poetry run pre-commit install

# Set up development database
echo "🗄️ Setting up development database..."
export PGPASSWORD=quantum_pass
while ! pg_isready -h postgres -p 5432 -U quantum_user; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

# Create database schema (if schema files exist)
if [ -f "scripts/schema.sql" ]; then
  psql -h postgres -U quantum_user -d quantum_scheduler_dev -f scripts/schema.sql
fi

# Set up Redis
echo "🔴 Setting up Redis..."
while ! redis-cli -h redis ping > /dev/null 2>&1; do
  echo "Waiting for Redis..."
  sleep 2
done

# Create development directories
echo "📁 Creating development directories..."
mkdir -p logs data notebooks temp

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
poetry run python -m ipykernel install --user --name quantum-scheduler --display-name "Quantum Scheduler"

# Download sample data (if applicable)
if [ -f "scripts/download_sample_data.py" ]; then
  echo "📊 Downloading sample data..."
  poetry run python scripts/download_sample_data.py
fi

# Run health check
echo "🩺 Running health check..."
poetry run python -c "import quantum_scheduler; print('✅ Package imports successfully')"

# Set up Git hooks
echo "🔗 Setting up Git hooks..."
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "🔍 Running pre-push checks..."
make quick-test
if [ $? -ne 0 ]; then
  echo "❌ Pre-push checks failed. Push aborted."
  exit 1
fi
echo "✅ Pre-push checks passed."
EOF
chmod +x .git/hooks/pre-push

# Create development configuration
echo "⚙️ Creating development configuration..."
cat > .env.development << 'EOF'
# Development environment configuration
QUANTUM_SCHEDULER_ENV=development
QUANTUM_SCHEDULER_LOG_LEVEL=DEBUG
QUANTUM_SCHEDULER_DATABASE_URL=postgresql://quantum_user:quantum_pass@postgres:5432/quantum_scheduler_dev
QUANTUM_SCHEDULER_REDIS_URL=redis://redis:6379/0
QUANTUM_SCHEDULER_DEBUG=true
QUANTUM_SCHEDULER_PROMETHEUS_PORT=9090
EOF

# Set up shell aliases
echo "🐚 Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# Quantum Scheduler development aliases
alias qs='cd /workspace'
alias qstest='poetry run pytest tests/ -v'
alias qsfast='poetry run pytest tests/ -v -m "not slow and not quantum"'
alias qslint='poetry run ruff check src/ tests/'
alias qsformat='poetry run black src/ tests/ && poetry run isort src/ tests/'
alias qsdocs='poetry run mkdocs serve'
alias qsapi='poetry run python -m quantum_scheduler.api.main'
alias qsjupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Git aliases
alias gs='git status'
alias gd='git diff'
alias gl='git log --oneline -10'
alias gp='git pull'

# Docker aliases
alias dps='docker ps'
alias dlogs='docker-compose logs -f'
alias dup='docker-compose up'
alias ddown='docker-compose down'
EOF

# Create development Makefile targets
echo "🎯 Adding development Makefile targets..."
cat >> Makefile << 'EOF'

# Development container specific targets
dev-shell: ## Open shell in development container
	docker-compose exec devcontainer bash

dev-jupyter: ## Start Jupyter Lab in development container
	docker-compose exec devcontainer jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

dev-logs: ## View development container logs
	docker-compose logs -f devcontainer

dev-reset: ## Reset development environment
	docker-compose down -v
	docker-compose up -d
	$(MAKE) dev-setup
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 You can now:"
echo "   - Run tests: make test"
echo "   - Start Jupyter: make dev-jupyter"
echo "   - Format code: make format"
echo "   - Run linting: make lint"
echo "   - Build docs: make docs-serve"
echo ""
echo "📚 Documentation: http://localhost:8000"
echo "📊 Grafana: http://localhost:3000 (admin/admin)"
echo "🔍 Prometheus: http://localhost:9090"
echo ""
echo "Happy coding! 🚀"