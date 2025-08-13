# Quantum Scheduler - Production Deployment Guide

## Overview

This guide covers deploying the Quantum Scheduler in production environments with high availability, scalability, and monitoring.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load Balancer  │    │   API Gateway   │    │   Monitoring    │
│   (nginx/HAProxy│────▶│   (Kong/Istio)  │────▶│   (Grafana/    │
│                 │    │                 │    │    Prometheus) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        ▼                        │
         │              ┌─────────────────┐                │
         │              │ Quantum Scheduler│                │
         │              │   API Servers   │                │
         │              │   (3+ replicas) │                │
         │              └─────────────────┘                │
         │                        │                        │
         │                        ▼                        │
         │              ┌─────────────────┐                │
         │              │ Worker Processes │                │
         │              │ (Classical,     │◀───────────────┘
         │              │  Quantum, Hybrid)│
         │              └─────────────────┘
         │                        │
         │                        ▼
         │              ┌─────────────────┐
         │              │   Data Layer    │
         └──────────────▶│ (Redis, Storage)│
                        └─────────────────┘
```

## Quick Start

### 1. Docker Compose (Recommended for Development)

```bash
# Clone and setup
git clone <repository>
cd quantum-scheduler

# Build and start all services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f quantum-scheduler-api
```

### 2. Kubernetes (Recommended for Production)

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/production.yaml

# Check deployment status
kubectl get pods -n quantum-scheduler

# Check services
kubectl get svc -n quantum-scheduler

# View logs
kubectl logs -f deployment/quantum-scheduler-api -n quantum-scheduler
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTUM_SCHEDULER_ENV` | `development` | Environment (development/staging/production) |
| `QUANTUM_SCHEDULER_LOG_LEVEL` | `info` | Logging level (debug/info/warning/error) |
| `QUANTUM_SCHEDULER_MAX_WORKERS` | `4` | Maximum worker processes |
| `QUANTUM_SCHEDULER_CACHE_SIZE` | `1000` | Solution cache size |
| `QUANTUM_SCHEDULER_ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `QUANTUM_SCHEDULER_BACKEND_STRATEGY` | `hybrid` | Default backend strategy |
| `QUANTUM_SCHEDULER_HEALTH_CHECK_INTERVAL` | `30` | Health check interval (seconds) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

### Production Configuration

Create a production configuration file:

```yaml
# config/production.yaml
quantum_scheduler:
  environment: production
  log_level: info
  
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 8
    timeout: 300
    max_request_size: 100MB
  
  scheduler:
    max_workers: 16
    backend_strategy: "adaptive_performance"
    cache:
      size: 10000
      ttl: 3600
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 60
    retry_policy:
      max_attempts: 3
      base_delay: 1.0
      backoff_strategy: "exponential"
  
  monitoring:
    enable_metrics: true
    metrics_port: 9090
    health_check_interval: 30
    log_sampling_rate: 0.01
  
  security:
    enable_auth: true
    jwt_secret: "${JWT_SECRET}"
    api_keys_required: true
    rate_limiting:
      requests_per_minute: 1000
      burst_size: 100
  
  backends:
    classical:
      enabled: true
      weight: 1.0
      max_problem_size: 1000
    quantum_simulator:
      enabled: true
      weight: 2.0
      max_qubits: 50
    quantum_hardware:
      enabled: false  # Enable when hardware is available
      provider: "ibm"
      max_qubits: 127
      queue_timeout: 3600
```

## Scaling

### Horizontal Scaling

The system supports horizontal scaling at multiple levels:

1. **API Servers**: Scale API pods based on CPU/memory usage
2. **Worker Processes**: Scale worker pods based on queue length and processing time
3. **Backend Services**: Scale different backend types independently

```bash
# Scale API servers
kubectl scale deployment quantum-scheduler-api --replicas=6 -n quantum-scheduler

# Scale workers
kubectl scale deployment quantum-scheduler-worker --replicas=12 -n quantum-scheduler
```

### Auto-scaling Configuration

The HPA (Horizontal Pod Autoscaler) is configured for both API and worker deployments:

```yaml
# API Auto-scaling: 3-10 replicas based on CPU/Memory
# Worker Auto-scaling: 4-20 replicas based on load
```

### Vertical Scaling

Adjust resource limits based on workload:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi" 
    cpu: "4000m"
```

## Monitoring and Observability

### Metrics Collection

Prometheus metrics are exposed on `/metrics` endpoint:

- **Request metrics**: Rate, duration, errors
- **Scheduler metrics**: Queue size, processing time, success rate
- **Resource metrics**: CPU, memory, disk usage
- **Business metrics**: Problem complexity, quantum advantage

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default credentials: admin/quantum_scheduler_2024)

Pre-configured dashboards:
1. **System Overview**: High-level system health and performance
2. **Scheduler Performance**: Detailed scheduling metrics and trends
3. **Resource Utilization**: Infrastructure usage and capacity planning
4. **Quantum Metrics**: Quantum backend performance and advantage tracking

### Alerting Rules

Critical alerts configured in Prometheus:

```yaml
groups:
- name: quantum-scheduler-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    labels:
      severity: critical
  
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    labels:
      severity: warning
  
  - alert: QuantumBackendDown
    expr: quantum_backend_health{backend_type="quantum"} == 0
    labels:
      severity: critical
```

### Distributed Tracing

Jaeger tracing is enabled for request flow analysis:

```bash
# Access Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686 -n quantum-scheduler
```

## Security

### Authentication & Authorization

1. **API Key Authentication**: Required for all API endpoints
2. **JWT Tokens**: For user session management
3. **Role-based Access Control**: Different permissions for users/admins

### Network Security

```yaml
# Network Policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quantum-scheduler-network-policy
spec:
  podSelector:
    matchLabels:
      app: quantum-scheduler-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

### Data Encryption

- **In Transit**: TLS 1.3 for all communications
- **At Rest**: AES-256 encryption for sensitive data
- **Secrets Management**: Kubernetes secrets with encryption at rest

### Security Scanning

```bash
# Container security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image quantum-scheduler:latest

# Kubernetes security scan
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml
```

## High Availability

### Multi-Region Deployment

Deploy across multiple availability zones:

```yaml
# Node affinity for zone distribution
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: topology.kubernetes.io/zone
          operator: In
          values:
          - us-west-2a
          - us-west-2b
          - us-west-2c
```

### Backup and Recovery

```bash
# Automated backup script
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

# Backup Redis data
kubectl exec -n quantum-scheduler redis-0 -- redis-cli BGSAVE
kubectl cp quantum-scheduler/redis-0:/data/dump.rdb ./backups/redis_${BACKUP_DATE}.rdb

# Backup application data
kubectl exec -n quantum-scheduler $(kubectl get pod -l app=quantum-scheduler-api -o jsonpath="{.items[0].metadata.name}") -- \
  tar -czf /tmp/app_data_${BACKUP_DATE}.tar.gz /app/data
kubectl cp quantum-scheduler/$(kubectl get pod -l app=quantum-scheduler-api -o jsonpath="{.items[0].metadata.name}"):/tmp/app_data_${BACKUP_DATE}.tar.gz \
  ./backups/app_data_${BACKUP_DATE}.tar.gz
```

### Disaster Recovery

1. **RTO (Recovery Time Objective)**: < 15 minutes
2. **RPO (Recovery Point Objective)**: < 5 minutes
3. **Automated failover** with health checks
4. **Cross-region replication** for critical data

## Performance Optimization

### Resource Tuning

```yaml
# JVM tuning for quantum simulators (if using Java-based backends)
env:
- name: JAVA_OPTS
  value: "-Xms2g -Xmx8g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Python optimization
env:
- name: PYTHONUNBUFFERED
  value: "1"
- name: PYTHONDONTWRITEBYTECODE  
  value: "1"
```

### Caching Strategy

1. **Solution Cache**: LRU cache for computed solutions
2. **Redis Cache**: Distributed caching for worker coordination
3. **CDN**: Static asset caching (if serving web UI)

### Database Optimization

```yaml
# Redis optimization for high throughput
redis:
  config:
    maxmemory: 8gb
    maxmemory-policy: allkeys-lru
    save: "900 1 300 10 60 10000"
    timeout: 300
    tcp-keepalive: 60
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory consumption
   kubectl top pods -n quantum-scheduler
   
   # Analyze memory leaks
   kubectl exec -it pod-name -n quantum-scheduler -- python -m memory_profiler app.py
   ```

2. **Slow Response Times**
   ```bash
   # Check API response times
   curl -w "@curl-format.txt" -s -o /dev/null http://api-endpoint/health
   
   # Profile application
   kubectl exec -it pod-name -n quantum-scheduler -- py-spy top --pid 1 --duration 60
   ```

3. **Quantum Backend Issues**
   ```bash
   # Check quantum backend connectivity
   kubectl logs -f deployment/quantum-scheduler-api -n quantum-scheduler | grep -i quantum
   
   # Test quantum simulator
   kubectl exec -it pod-name -n quantum-scheduler -- python -c "
   from quantum_scheduler.backends.quantum import QuantumBackend
   backend = QuantumBackend()
   print(backend.health_check())
   "
   ```

### Log Analysis

```bash
# Centralized logging with ELK stack
kubectl apply -f monitoring/elasticsearch.yaml
kubectl apply -f monitoring/logstash.yaml
kubectl apply -f monitoring/kibana.yaml

# Query logs
kubectl logs -n quantum-scheduler -l app=quantum-scheduler-api --since=1h | grep ERROR
```

### Performance Profiling

```bash
# CPU profiling
kubectl exec -it pod-name -n quantum-scheduler -- py-spy record -o profile.svg --pid 1 --duration 60

# Memory profiling  
kubectl exec -it pod-name -n quantum-scheduler -- python -m memory_profiler --precision 2 app.py
```

## Maintenance

### Rolling Updates

```bash
# Update deployment with zero downtime
kubectl set image deployment/quantum-scheduler-api quantum-scheduler-api=quantum-scheduler:1.1.0 -n quantum-scheduler
kubectl rollout status deployment/quantum-scheduler-api -n quantum-scheduler

# Rollback if needed
kubectl rollout undo deployment/quantum-scheduler-api -n quantum-scheduler
```

### Health Checks

```bash
# Manual health check
curl -f http://api-endpoint/health

# Detailed system status
curl -f http://api-endpoint/status

# Metrics endpoint
curl http://api-endpoint/metrics
```

### Capacity Planning

Monitor these metrics for scaling decisions:

1. **CPU Utilization**: Target < 70%
2. **Memory Usage**: Target < 80%
3. **Request Queue Length**: Target < 100
4. **Response Time**: Target < 200ms (95th percentile)
5. **Error Rate**: Target < 0.1%

## Support

### Contact Information

- **Technical Support**: support@terragon-labs.com
- **Documentation**: https://docs.quantum-scheduler.com
- **Issues**: https://github.com/terragon-labs/quantum-scheduler/issues

### SLA

- **Uptime**: 99.9% availability
- **Response Time**: < 200ms (95th percentile)
- **Support Response**: < 4 hours (business hours)
- **Resolution Time**: < 24 hours (critical issues)

---

**© 2024 Terragon Labs - Production Deployment Guide v1.0**