# Deployment Guide

Comprehensive deployment guide for the Quantum Agent Scheduler.

## 📋 Table of Contents

- [Overview](#overview)
- [Container Images](#container-images)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Configuration Management](#configuration-management)
- [Security Best Practices](#security-best-practices)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Quantum Agent Scheduler supports multiple deployment strategies:

- **Development**: Local development with hot reload
- **Production**: Optimized container for production workloads
- **Quantum Hardware**: Specialized deployment with quantum provider access
- **Microservices**: Distributed deployment with separate API and worker services

## 🐳 Container Images

### Build Stages

Our multi-stage Dockerfile provides different targets:

```bash
# Development build
docker build --target development -t quantum-scheduler:dev .

# Production build
docker build --target production -t quantum-scheduler:prod .

# Quantum hardware build
docker build --target quantum-hw -t quantum-scheduler:quantum .
```

### Image Variants

| Stage | Size | Use Case | Quantum Support |
|-------|------|----------|-----------------|
| `development` | ~2.1GB | Local dev, testing | Simulator only |
| `production` | ~800MB | Production API | Configurable |
| `quantum-hw` | ~1.2GB | Quantum cloud | Full support |

### Security Features

- ✅ Non-root user (`scheduler`)
- ✅ Multi-stage build (reduced attack surface)
- ✅ No package caches in final image
- ✅ Health checks included
- ✅ Secrets via environment variables only

## 🚀 Local Development

### Docker Compose Development

```bash
# Start development environment
docker-compose up --build

# Start with specific services
docker-compose up api postgres redis

# Run tests in container
docker-compose run --rm api poetry run pytest

# Access shell in running container
docker-compose exec api bash
```

### Development Services

```yaml
# docker-compose.override.yml (auto-loaded in dev)
version: '3.8'
services:
  api:
    target: development
    volumes:
      - .:/app
      - ~/.aws:/home/scheduler/.aws:ro
      - ~/.qiskit:/home/scheduler/.qiskit:ro
    environment:
      - QUANTUM_SCHEDULER_ENV=development
      - QUANTUM_SCHEDULER_LOG_LEVEL=DEBUG
    ports:
      - "8080:8000"
```

### Hot Reload Setup

```bash
# Development with auto-reload
docker-compose run --rm -p 8080:8000 api \
  poetry run uvicorn quantum_scheduler.api:app \
  --host 0.0.0.0 --port 8000 --reload
```

## 🏭 Production Deployment

### Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    image: quantum-scheduler:prod
    target: production
    restart: unless-stopped
    environment:
      - QUANTUM_SCHEDULER_ENV=production
      - QUANTUM_SCHEDULER_LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://user:pass@postgres:5432/scheduler
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "80:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  worker:
    image: quantum-scheduler:prod
    restart: unless-stopped
    command: ["poetry", "run", "celery", "worker", "-A", "quantum_scheduler.worker"]
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis
      - postgres
```

### Production Startup

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale worker=4

# View logs
docker-compose logs -f api
```

### Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## ☸️ Kubernetes Deployment

### Basic Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-scheduler
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-scheduler-api
  namespace: quantum-scheduler
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-scheduler-api
  template:
    metadata:
      labels:
        app: quantum-scheduler-api
    spec:
      containers:
      - name: api
        image: quantum-scheduler:prod
        ports:
        - containerPort: 8000
        env:
        - name: QUANTUM_SCHEDULER_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: quantum-scheduler-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Helm Chart

```bash
# Install with Helm
helm repo add quantum-scheduler https://charts.quantum-scheduler.io
helm install my-scheduler quantum-scheduler/quantum-scheduler \
  --namespace quantum-scheduler \
  --create-namespace \
  --values values.yaml
```

### Quantum Hardware Access

```yaml
# k8s/quantum-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: quantum-credentials
  namespace: quantum-scheduler
type: Opaque
data:
  aws-access-key-id: <base64-encoded>
  aws-secret-access-key: <base64-encoded>
  qiskit-token: <base64-encoded>
  dwave-token: <base64-encoded>
```

## ☁️ Cloud Deployments

### AWS ECS

```yaml
# ecs-task-definition.json
{
  "family": "quantum-scheduler",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/quantumSchedulerTaskRole",
  "containerDefinitions": [
    {
      "name": "quantum-scheduler-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/quantum-scheduler:prod",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "QUANTUM_SCHEDULER_ENV", "value": "production"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"}
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:quantum-scheduler/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/quantum-scheduler",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: quantum-scheduler
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containers:
      - image: gcr.io/project-id/quantum-scheduler:prod
        ports:
        - containerPort: 8000
        env:
        - name: QUANTUM_SCHEDULER_ENV
          value: "production"
        - name: PORT
          value: "8000"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          timeoutSeconds: 5
          periodSeconds: 10
          failureThreshold: 3
```

### Azure Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group quantum-scheduler-rg \
  --name quantum-scheduler \
  --image your-registry.azurecr.io/quantum-scheduler:prod \
  --cpu 2 \
  --memory 4 \
  --ports 80 \
  --dns-name-label quantum-scheduler-unique \
  --environment-variables \
    QUANTUM_SCHEDULER_ENV=production \
    AZURE_QUANTUM_LOCATION=westus \
  --secure-environment-variables \
    DATABASE_URL=postgresql://... \
    AZURE_CLIENT_SECRET=... \
  --restart-policy Always
```

## ⚙️ Configuration Management

### Environment Variables

```bash
# Production environment template
export QUANTUM_SCHEDULER_ENV=production
export QUANTUM_SCHEDULER_LOG_LEVEL=INFO
export API_WORKERS=4
export API_TIMEOUT=300

# Database
export DATABASE_URL=postgresql://user:pass@localhost:5432/scheduler
export DATABASE_POOL_SIZE=20
export DATABASE_POOL_TIMEOUT=30

# Quantum Backends
export AWS_DEFAULT_REGION=us-east-1
export QISKIT_IBM_INSTANCE=ibm-q/open/main
export AZURE_QUANTUM_LOCATION=westus

# Performance
export QUANTUM_THRESHOLD=50
export COST_BUDGET=100.0
export TIME_LIMIT=300
export CLASSICAL_THREADS=4

# Monitoring
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
export PROMETHEUS_PORT=9090
export GRAFANA_PORT=3000
```

### ConfigMaps (Kubernetes)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-scheduler-config
data:
  config.yaml: |
    scheduler:
      quantum_threshold: 50
      cost_budget: 100.0
      time_limit: 300
    
    backends:
      classical:
        solver: gurobi
        threads: 4
      
      quantum:
        fallback_enabled: true
        max_cost: 50.0
    
    api:
      workers: 4
      timeout: 300
      cors_origins:
        - "https://app.example.com"
        - "https://dashboard.example.com"
```

### Secrets Management

```bash
# Using Docker secrets
echo "postgresql://user:pass@db:5432/scheduler" | docker secret create db_url -

# Using Kubernetes secrets
kubectl create secret generic quantum-scheduler-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=aws-access-key-id="..." \
  --from-literal=aws-secret-access-key="..." \
  --namespace quantum-scheduler

# Using HashiCorp Vault
vault kv put secret/quantum-scheduler/prod \
  database_url="postgresql://..." \
  aws_access_key_id="..." \
  aws_secret_access_key="..."
```

## 🔒 Security Best Practices

### Container Security

```dockerfile
# Security best practices in Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r scheduler && useradd -r -g scheduler scheduler

# Use specific package versions
RUN apt-get update && apt-get install -y \
    curl=7.74.0-1.3+deb11u2 \
    && rm -rf /var/lib/apt/lists/*

# Don't run as root
USER scheduler

# Use HTTPS for external resources
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
```

### Network Security

```yaml
# Network policies (Kubernetes)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quantum-scheduler-netpol
spec:
  podSelector:
    matchLabels:
      app: quantum-scheduler
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
```

### Secrets Scanning

```bash
# Scan for secrets before deployment
docker run --rm -v $(pwd):/src trufflesecurity/trufflehog:latest filesystem /src

# Scan container image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image quantum-scheduler:prod
```

## 📊 Monitoring & Observability

### Health Checks

```python
# Custom health check endpoint
@app.get("/health")
async def health_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "quantum_backends": await check_quantum_backends()
    }
    
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail=checks)
```

### Monitoring Stack

```yaml
# monitoring/docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards
  
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
  
handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: /app/logs/quantum-scheduler.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  quantum_scheduler:
    level: INFO
    handlers: [console, file]
    propagate: false
  
  uvicorn:
    level: INFO
    handlers: [console]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs quantum-scheduler-api

# Common fixes
docker run -it --entrypoint /bin/bash quantum-scheduler:prod
# Check file permissions
ls -la /app
# Check environment variables
env | grep QUANTUM
```

#### Database Connection Issues

```bash
# Test database connectivity
docker run --rm --network host postgres:13 \
  psql "postgresql://user:pass@localhost:5432/scheduler" -c "SELECT 1;"

# Check DNS resolution
docker run --rm --network quantum-scheduler_default \
  busybox nslookup postgres
```

#### Quantum Backend Errors

```bash
# Test quantum provider connectivity
docker run --rm -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  quantum-scheduler:quantum \
  python -c "
import boto3
client = boto3.client('braket', region_name='us-east-1')
print(client.search_devices())
"
```

#### Performance Issues

```bash
# Monitor resource usage
docker stats quantum-scheduler-api

# Check application metrics
curl http://localhost:8080/metrics

# Profile memory usage
docker exec quantum-scheduler-api \
  python -m memory_profiler quantum_scheduler/api/main.py
```

### Debug Mode

```bash
# Run in debug mode
docker run -it --rm \
  -e QUANTUM_SCHEDULER_LOG_LEVEL=DEBUG \
  -e PYTHONPATH=/app/src \
  -p 8080:8000 \
  quantum-scheduler:dev \
  python -m pdb -c continue quantum_scheduler/api/main.py
```

### Log Analysis

```bash
# Filter logs by level
docker logs quantum-scheduler-api 2>&1 | grep ERROR

# Follow logs in real-time
docker logs -f quantum-scheduler-api

# Export logs for analysis
docker logs quantum-scheduler-api > scheduler.log 2>&1
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)
- [Quantum Computing in the Cloud](https://aws.amazon.com/braket/)
- [Container Security Guide](https://sysdig.com/blog/dockerfile-best-practices/)
- [Production Readiness Checklist](https://gruntwork.io/devops-checklist/)