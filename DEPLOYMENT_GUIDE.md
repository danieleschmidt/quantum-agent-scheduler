# Quantum Agent Scheduler - Production Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying the Quantum Agent Scheduler to production environments using Docker, Kubernetes, or cloud platforms.

---

## 🏗️ Architecture Overview

The production deployment consists of:

- **Quantum Scheduler Core**: Main scheduling service with quantum algorithm optimization
- **Circuit Optimizer**: Dedicated service for quantum circuit depth optimization  
- **Cloud Orchestrator**: Multi-region quantum cloud resource management
- **PostgreSQL**: Primary data store for scheduling state and history
- **Redis**: Caching and message broker for distributed operations
- **NGINX**: Load balancer and reverse proxy with SSL termination
- **Monitoring Stack**: Prometheus, Grafana, and logging aggregation

---

## 🚀 Quick Start (Docker Compose)

### Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- 16GB+ RAM and 8+ CPU cores recommended
- SSL certificates for HTTPS (self-signed acceptable for testing)

### 1. Clone and Configure

```bash
git clone https://github.com/terragon-labs/quantum-agent-scheduler
cd quantum-agent-scheduler

# Create environment file
cp .env.example .env.prod
vim .env.prod  # Edit with your values
```

### 2. Generate SSL Certificates

```bash
# Self-signed certificate for testing
mkdir -p deployment/production/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/production/nginx/ssl/quantum-scheduler.key \
  -out deployment/production/nginx/ssl/quantum-scheduler.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=quantum-scheduler"
```

### 3. Build and Deploy

```bash
# Build images
docker-compose -f deployment/production/docker-compose.prod.yml build

# Start services
docker-compose -f deployment/production/docker-compose.prod.yml up -d

# Check status
docker-compose -f deployment/production/docker-compose.prod.yml ps
```

### 4. Verify Deployment

```bash
# Health check
curl -k https://localhost/health

# API test
curl -k https://localhost/api/v1/status

# Metrics
curl -k https://localhost:9090/metrics
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster 1.24+ with at least 32GB memory and 16 CPU cores
- kubectl configured with cluster access
- StorageClass for persistent volumes
- Ingress controller (NGINX, Traefik, etc.)

### 1. Prepare Cluster

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Create secrets
kubectl create secret generic quantum-scheduler-secrets \
  --from-literal=db-password=your-db-password \
  --from-literal=redis-password=your-redis-password \
  --from-literal=grafana-password=your-grafana-password \
  --from-literal=aws-access-key=your-aws-key \
  --from-literal=aws-secret-key=your-aws-secret \
  --from-literal=ibm-quantum-token=your-ibm-token \
  --from-literal=dwave-token=your-dwave-token \
  -n quantum-scheduler
```

### 2. Deploy Infrastructure

```bash
# Deploy PostgreSQL and Redis
kubectl apply -f deployment/kubernetes/infrastructure.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n quantum-scheduler --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n quantum-scheduler --timeout=300s
```

### 3. Deploy Application

```bash
# Deploy quantum scheduler
kubectl apply -f deployment/kubernetes/quantum-scheduler.yaml

# Check deployment status
kubectl get pods -n quantum-scheduler
kubectl logs -f deployment/quantum-scheduler -n quantum-scheduler
```

### 4. Configure Ingress

```bash
# Deploy ingress (customize for your ingress controller)
kubectl apply -f deployment/kubernetes/ingress.yaml

# Get external IP
kubectl get ingress -n quantum-scheduler
```

### 5. Monitoring Setup

```bash
# Deploy monitoring stack
kubectl apply -f deployment/kubernetes/monitoring.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n quantum-scheduler
```

---

## ☁️ Cloud Platform Deployment

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name quantum-scheduler \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name quantum-scheduler

# Deploy using Kubernetes manifests
kubectl apply -f deployment/kubernetes/
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create quantum-scheduler \
  --num-nodes=3 \
  --machine-type=n1-standard-8 \
  --zone=us-central1-a \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10

# Get credentials
gcloud container clusters get-credentials quantum-scheduler --zone=us-central1-a

# Deploy
kubectl apply -f deployment/kubernetes/
```

### Azure AKS

```bash
# Create resource group
az group create --name quantum-scheduler-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group quantum-scheduler-rg \
  --name quantum-scheduler \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group quantum-scheduler-rg --name quantum-scheduler

# Deploy
kubectl apply -f deployment/kubernetes/
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `WORKERS` | Number of worker processes | `4` |
| `MAX_CONCURRENT_JOBS` | Maximum concurrent quantum jobs | `500` |
| `ENABLE_AUTO_SCALING` | Enable automatic scaling | `true` |
| `POSTGRES_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `QUANTUM_PROVIDERS` | Enabled quantum providers | `aws_braket,ibm_quantum,dwave` |

### Quantum Provider Configuration

#### AWS Braket
```yaml
aws_braket:
  region: us-west-2
  s3_bucket: your-braket-results-bucket
  devices:
    - arn:aws:braket:::device/quantum-simulator/amazon/sv1
    - arn:aws:braket:::device/qpu/d-wave/Advantage_system6.1
```

#### IBM Quantum
```yaml
ibm_quantum:
  hub: ibm-q
  group: open
  project: main
  backends:
    - ibmq_qasm_simulator
    - ibm_lagos
```

#### D-Wave
```yaml
dwave:
  endpoint: https://cloud.dwavesys.com/sapi
  solver: Advantage_system6.1
```

---

## 🔍 Monitoring & Observability

### Prometheus Metrics

Key metrics exposed:

- `quantum_jobs_total` - Total number of quantum jobs
- `quantum_jobs_duration_seconds` - Job execution time
- `quantum_circuit_depth_reduction_ratio` - Circuit optimization effectiveness
- `quantum_provider_availability` - Provider availability status
- `quantum_error_correction_rate` - Error correction success rate

### Grafana Dashboards

Pre-built dashboards available:

1. **Quantum Scheduling Overview**: High-level metrics and KPIs
2. **Circuit Optimization**: Detailed optimization performance
3. **Multi-Region Orchestration**: Cloud resource utilization
4. **Error Correction**: Fault tolerance metrics
5. **Performance Benchmarking**: Comparative algorithm performance

### Alerting Rules

Critical alerts configured:

- High quantum job failure rate (>5%)
- Circuit optimization taking >30s
- Quantum provider unavailable >5min
- Error correction failure rate >10%
- Memory usage >90%
- Response time >10s

### Log Aggregation

Structured logging with:

- Application logs: JSON format with correlation IDs
- Access logs: NGINX access logs with performance metrics
- Audit logs: Quantum provider API calls and results
- Error logs: Detailed error traces with context

---

## 🔐 Security Configuration

### Network Security

- **TLS 1.3**: All external communication encrypted
- **mTLS**: Inter-service communication secured
- **Network Policies**: Kubernetes network segmentation
- **Firewall Rules**: Cloud provider security groups

### Authentication & Authorization

- **API Keys**: Quantum provider credentials rotation
- **RBAC**: Kubernetes role-based access control
- **Service Accounts**: Minimal privilege principles
- **Secrets Management**: Encrypted secret storage

### Data Protection

- **Encryption at Rest**: Database and cache encryption
- **Encryption in Transit**: All network traffic encrypted
- **Data Retention**: Configurable data lifecycle policies
- **Audit Logging**: Complete audit trail

---

## 📈 Scaling & Performance

### Horizontal Scaling

```yaml
# HPA configuration
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        averageUtilization: 80
```

### Vertical Scaling

Resource recommendations:

| Component | CPU | Memory | Storage |
|-----------|-----|---------|---------|
| Quantum Scheduler | 2-4 cores | 4-8GB | 20GB |
| Circuit Optimizer | 4-6 cores | 8-12GB | 10GB |
| Cloud Orchestrator | 1-2 cores | 2-4GB | 5GB |
| PostgreSQL | 2-4 cores | 4-8GB | 50-100GB |
| Redis | 1-2 cores | 2-4GB | 20GB |

### Performance Tuning

#### Database Optimization

```sql
-- PostgreSQL tuning
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 256MB
max_connections = 200
```

#### Cache Configuration

```yaml
# Redis tuning
maxmemory: 2gb
maxmemory-policy: allkeys-lru
tcp-keepalive: 60
timeout: 300
```

---

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and Push Images
      run: |
        docker build -t quantum-scheduler:${{ github.ref_name }} .
        docker push quantum-scheduler:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/quantum-scheduler \
          quantum-scheduler=quantum-scheduler:${{ github.ref_name }}
```

### Helm Charts

```bash
# Package Helm chart
helm package deployment/helm/quantum-scheduler

# Deploy with Helm
helm install quantum-scheduler ./quantum-scheduler-1.0.0.tgz \
  --namespace quantum-scheduler \
  --create-namespace \
  --values values.production.yaml
```

---

## 🔧 Maintenance & Operations

### Health Checks

```bash
# Comprehensive health check
curl -k https://your-domain/health

# Component-specific checks
curl -k https://your-domain/api/v1/status
curl -k https://your-domain/api/v1/providers/status
curl -k https://your-domain/metrics
```

### Database Maintenance

```bash
# Backup database
kubectl exec -n quantum-scheduler postgres-0 -- \
  pg_dump -U scheduler quantum_scheduler > backup.sql

# Restore database
kubectl exec -i -n quantum-scheduler postgres-0 -- \
  psql -U scheduler quantum_scheduler < backup.sql

# Vacuum and analyze
kubectl exec -n quantum-scheduler postgres-0 -- \
  psql -U scheduler -d quantum_scheduler -c "VACUUM ANALYZE;"
```

### Log Management

```bash
# View application logs
kubectl logs -f deployment/quantum-scheduler -n quantum-scheduler

# View aggregated logs in Grafana
# Access Loki data source in Grafana dashboards

# Export logs for analysis
kubectl logs deployment/quantum-scheduler -n quantum-scheduler \
  --since=24h > quantum-scheduler-logs.txt
```

### Certificate Management

```bash
# Auto-renewal with cert-manager (Kubernetes)
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Manual certificate renewal
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout new-key.pem -out new-cert.pem

# Update certificates in deployment
kubectl create secret tls quantum-scheduler-tls \
  --cert=new-cert.pem --key=new-key.pem \
  -n quantum-scheduler --dry-run=client -o yaml | kubectl apply -f -
```

---

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n quantum-scheduler

# Investigate quantum jobs
curl -k https://your-domain/api/v1/jobs/active

# Solution: Reduce MAX_CONCURRENT_JOBS or scale up
```

#### Circuit Optimization Timeout

```bash
# Check optimization queue
curl -k https://your-domain/api/v1/optimize/status

# Increase optimization timeout
kubectl set env deployment/circuit-optimizer \
  OPTIMIZATION_TIMEOUT=300 -n quantum-scheduler
```

#### Quantum Provider Connection Issues

```bash
# Check provider status
curl -k https://your-domain/api/v1/providers/status

# Verify credentials
kubectl get secret quantum-scheduler-secrets -n quantum-scheduler -o yaml

# Test provider connectivity
kubectl exec -it deployment/quantum-scheduler -n quantum-scheduler -- \
  python -c "from quantum_scheduler.cloud import CloudOrchestrator; print(CloudOrchestrator().get_resource_status())"
```

### Performance Issues

#### Slow Database Queries

```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Analyze table statistics
ANALYZE;

-- Check indexes
\d+ table_name
```

#### High Circuit Optimization Latency

```bash
# Check optimization metrics
curl -k https://your-domain/metrics | grep circuit_optimization

# Scale circuit optimizer
kubectl scale deployment circuit-optimizer --replicas=3 -n quantum-scheduler

# Tune optimization levels
kubectl set env deployment/circuit-optimizer \
  DEFAULT_OPTIMIZATION_LEVEL=2 -n quantum-scheduler
```

---

## 📞 Support & Resources

### Documentation

- [API Reference](https://docs.terragon.com/quantum-scheduler/api)
- [Configuration Guide](https://docs.terragon.com/quantum-scheduler/config)
- [Best Practices](https://docs.terragon.com/quantum-scheduler/best-practices)

### Community

- GitHub Issues: Report bugs and request features
- Discord: Join our quantum computing community
- Slack: Quantum scheduler workspace

### Professional Support

- Enterprise Support: 24/7 support with SLA
- Consulting Services: Deployment and optimization consulting
- Training Programs: Quantum scheduling workshops

---

*This deployment guide is maintained by Terragon Labs. For updates and additional resources, visit our [documentation site](https://docs.terragon.com/quantum-scheduler).*