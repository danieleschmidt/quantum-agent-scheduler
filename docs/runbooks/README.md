# Operational Runbooks

Comprehensive runbooks for operating the Quantum Agent Scheduler in production.

## 📋 Table of Contents

- [Overview](#overview)
- [General Procedures](#general-procedures)
- [Service Management](#service-management)
- [Incident Response](#incident-response)
- [Performance Troubleshooting](#performance-troubleshooting)
- [Quantum Backend Management](#quantum-backend-management)
- [Database Operations](#database-operations)
- [Security Procedures](#security-procedures)
- [Backup & Recovery](#backup--recovery)

## 🎯 Overview

This directory contains operational runbooks for managing the Quantum Agent Scheduler in production. Each runbook provides step-by-step procedures for common operational tasks and incident response.

### Runbook Structure

Each runbook follows this structure:
- **Overview**: Brief description of the scenario
- **Prerequisites**: Required access and tools
- **Immediate Actions**: First steps to take (0-5 minutes)
- **Investigation**: Diagnostic procedures (5-30 minutes)
- **Resolution**: Step-by-step solution
- **Prevention**: How to prevent recurrence
- **Escalation**: When and how to escalate

## 🔧 General Procedures

### Accessing Production Systems

```bash
# Kubernetes cluster access
kubectl config use-context production-cluster
kubectl get pods -n quantum-scheduler

# SSH to production nodes (if needed)
ssh -i ~/.ssh/production-key user@production-node

# Database access
psql "postgresql://user:pass@db.internal:5432/scheduler"

# Monitoring systems
# Grafana: https://grafana.internal
# Prometheus: https://prometheus.internal:9090
# Alertmanager: https://alertmanager.internal:9093
```

### Log Analysis

```bash
# Application logs
kubectl logs -f deployment/quantum-scheduler-api -n quantum-scheduler

# Aggregate logs with stern
stern quantum-scheduler -n quantum-scheduler

# Search logs with specific patterns
kubectl logs deployment/quantum-scheduler-api -n quantum-scheduler | grep ERROR

# Export logs for analysis
kubectl logs deployment/quantum-scheduler-api -n quantum-scheduler --since=1h > /tmp/app-logs.txt
```

### Metrics and Monitoring

```bash
# Check service health
curl -f http://quantum-scheduler-api.internal/health

# Prometheus queries
curl -G 'http://prometheus.internal:9090/api/v1/query' \
  --data-urlencode 'query=up{job="quantum-scheduler"}'

# Grafana dashboard URLs
echo "Main Dashboard: https://grafana.internal/d/quantum-scheduler-main"
echo "Performance Dashboard: https://grafana.internal/d/quantum-scheduler-perf"
```

## 🚀 Service Management

### Service Restart

**When to use**: Service appears hung, memory leaks, configuration changes

```bash
# Kubernetes rolling restart
kubectl rollout restart deployment/quantum-scheduler-api -n quantum-scheduler

# Check restart status
kubectl rollout status deployment/quantum-scheduler-api -n quantum-scheduler

# Verify pods are healthy
kubectl get pods -n quantum-scheduler -l app=quantum-scheduler-api

# Check logs after restart
kubectl logs -f deployment/quantum-scheduler-api -n quantum-scheduler --tail=50
```

### Scaling Operations

**When to use**: High load, performance issues, maintenance

```bash
# Scale up API pods
kubectl scale deployment/quantum-scheduler-api --replicas=5 -n quantum-scheduler

# Scale worker pods
kubectl scale deployment/quantum-scheduler-worker --replicas=8 -n quantum-scheduler

# Check scaling status
kubectl get deployment -n quantum-scheduler

# Monitor resource usage during scaling
kubectl top pods -n quantum-scheduler
```

### Configuration Updates

**When to use**: Environment variable changes, feature flags

```bash
# Update ConfigMap
kubectl edit configmap quantum-scheduler-config -n quantum-scheduler

# Update Secret
kubectl edit secret quantum-scheduler-secrets -n quantum-scheduler

# Restart pods to pick up config changes
kubectl rollout restart deployment/quantum-scheduler-api -n quantum-scheduler

# Verify configuration is loaded
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  env | grep QUANTUM_
```

## 🚨 Incident Response

### Service Down

**Symptoms**: Service unavailable, health checks failing, alerts firing

**Immediate Actions (0-5 minutes)**:
```bash
# 1. Check pod status
kubectl get pods -n quantum-scheduler

# 2. Check recent events
kubectl get events -n quantum-scheduler --sort-by='.lastTimestamp' | tail -20

# 3. Check resource availability
kubectl describe nodes

# 4. Check external dependencies
curl -f http://postgres.internal:5432
curl -f http://redis.internal:6379
```

**Investigation (5-30 minutes)**:
```bash
# Check pod logs
kubectl logs deployment/quantum-scheduler-api -n quantum-scheduler --tail=100

# Check resource limits
kubectl describe pod -l app=quantum-scheduler-api -n quantum-scheduler

# Check service and ingress
kubectl get svc,ingress -n quantum-scheduler

# Check persistent volumes
kubectl get pv,pvc -n quantum-scheduler
```

**Resolution**:
1. **If OOMKilled**: Increase memory limits
2. **If ImagePullBackOff**: Check image availability
3. **If CrashLoopBackOff**: Fix application configuration
4. **If pending pods**: Check resource availability

### High Error Rate

**Symptoms**: 5xx errors, increased error rate alerts

**Immediate Actions**:
```bash
# Check error distribution
kubectl logs deployment/quantum-scheduler-api -n quantum-scheduler | \
  grep -E "(ERROR|500|502|503|504)" | tail -50

# Check recent deployments
kubectl rollout history deployment/quantum-scheduler-api -n quantum-scheduler

# Check quantum backend status
curl -f http://quantum-scheduler-api.internal/api/v1/backends/status
```

**Investigation**:
```bash
# Analyze error patterns
kubectl logs deployment/quantum-scheduler-api -n quantum-scheduler --since=30m | \
  grep ERROR | awk '{print $1, $2}' | sort | uniq -c | sort -nr

# Check database connectivity
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://...')
    result = await conn.fetchval('SELECT 1')
    print('DB OK:', result)
asyncio.run(test())
"

# Check quantum providers
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  python -c "
from quantum_scheduler.backends import BraketBackend
backend = BraketBackend()
print('Braket status:', backend.health_check())
"
```

**Resolution**:
1. If recent deployment caused issues: Rollback
2. If database issues: Check connections and queries
3. If quantum backend issues: Switch to fallback
4. If authentication issues: Check tokens and certificates

### Performance Degradation

**Symptoms**: High response times, timeout errors

**Immediate Actions**:
```bash
# Check current load
kubectl top pods -n quantum-scheduler

# Check active connections
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  netstat -an | grep ESTABLISHED | wc -l

# Check queue lengths
curl http://quantum-scheduler-api.internal/metrics | grep queue_length
```

**Investigation**:
```bash
# Profile application performance
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  python -m cProfile -o profile.stats quantum_scheduler/api/main.py

# Check database performance
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  psql -c "
  SELECT query, calls, total_time, mean_time 
  FROM pg_stat_statements 
  ORDER BY total_time DESC 
  LIMIT 10;"

# Check memory usage patterns
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB')
print(f'CPU: {process.cpu_percent()}%')
"
```

## 🔬 Quantum Backend Management

### Backend Health Check

```bash
# Check all backend status
curl http://quantum-scheduler-api.internal/api/v1/backends/status | jq .

# Test specific backend
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  python -c "
from quantum_scheduler.backends import BraketBackend
backend = BraketBackend()
try:
    result = backend.health_check()
    print('Backend healthy:', result)
except Exception as e:
    print('Backend error:', e)
"
```

### Cost Management

```bash
# Check current costs
curl http://quantum-scheduler-api.internal/metrics | grep quantum_cost

# Get cost breakdown by provider
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  python -c "
from quantum_scheduler.monitoring import CostTracker
tracker = CostTracker()
costs = tracker.get_daily_costs()
for provider, cost in costs.items():
    print(f'{provider}: ${cost:.2f}')
"

# Set cost limits (emergency brake)
kubectl patch configmap quantum-scheduler-config -n quantum-scheduler --patch '
data:
  COST_BUDGET: "50.0"
  QUANTUM_ENABLED: "false"
'
```

### Backend Failover

```bash
# Switch to classical fallback
kubectl patch configmap quantum-scheduler-config -n quantum-scheduler --patch '
data:
  FALLBACK_TO_CLASSICAL: "true"
  QUANTUM_THRESHOLD: "999999"
'

# Enable specific quantum backend only
kubectl patch configmap quantum-scheduler-config -n quantum-scheduler --patch '
data:
  ENABLED_BACKENDS: "qiskit"
  BRAKET_ENABLED: "false"
  DWAVE_ENABLED: "false"
'

# Restart to apply changes
kubectl rollout restart deployment/quantum-scheduler-api -n quantum-scheduler
```

## 💾 Database Operations

### Database Health Check

```bash
# Check database connectivity
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  pg_isready -h localhost -p 5432

# Check database size
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  psql -c "
  SELECT 
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
  FROM pg_database;
  "

# Check active connections
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  psql -c "SELECT count(*) FROM pg_stat_activity;"
```

### Performance Tuning

```bash
# Check slow queries
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  psql -c "
  SELECT query, calls, total_time, mean_time, rows
  FROM pg_stat_statements
  WHERE mean_time > 1000
  ORDER BY total_time DESC
  LIMIT 10;
  "

# Check index usage
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  psql -c "
  SELECT schemaname, tablename, attname, n_distinct, correlation
  FROM pg_stats
  WHERE schemaname = 'public'
  ORDER BY n_distinct DESC;
  "

# Vacuum and analyze
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  psql -c "VACUUM ANALYZE;"
```

### Backup Operations

```bash
# Create database backup
kubectl exec -it postgres-0 -n quantum-scheduler -- \
  pg_dump quantum_scheduler > backup-$(date +%Y%m%d-%H%M%S).sql

# Restore from backup
kubectl exec -i postgres-0 -n quantum-scheduler -- \
  psql quantum_scheduler < backup-20240101-120000.sql

# Check backup status
kubectl get job -n quantum-scheduler -l app=postgres-backup
```

## 🔒 Security Procedures

### Certificate Management

```bash
# Check certificate expiration
kubectl get secret -n quantum-scheduler -o yaml | \
  grep -E "(tls.crt|ca.crt)" | head -5 | \
  while read line; do
    echo $line | base64 -d | openssl x509 -noout -dates
  done

# Renew certificates (cert-manager)
kubectl delete secret tls-quantum-scheduler -n quantum-scheduler
kubectl annotate certificate quantum-scheduler-tls -n quantum-scheduler \
  cert-manager.io/issue-temp-certificate=true --overwrite
```

### Access Control

```bash
# Check RBAC permissions
kubectl auth can-i create pods --as=system:serviceaccount:quantum-scheduler:api

# Audit access logs
kubectl logs -n kube-system -l component=kube-apiserver | \
  grep quantum-scheduler | grep -E "(get|post|put|delete)"

# Review service account permissions
kubectl describe rolebinding -n quantum-scheduler
kubectl describe clusterrolebinding | grep quantum-scheduler
```

### Secrets Rotation

```bash
# Rotate API keys
kubectl create secret generic quantum-scheduler-secrets-new \
  --from-literal=aws-access-key-id="new-key" \
  --from-literal=aws-secret-access-key="new-secret" \
  -n quantum-scheduler

# Update deployment to use new secret
kubectl patch deployment quantum-scheduler-api -n quantum-scheduler \
  -p '{"spec":{"template":{"spec":{"volumes":[{"name":"secrets","secret":{"secretName":"quantum-scheduler-secrets-new"}}]}}}}'

# Verify new secrets are loaded
kubectl exec -it deploy/quantum-scheduler-api -n quantum-scheduler -- \
  env | grep AWS_ACCESS_KEY_ID
```

## 📚 Maintenance Procedures

### Planned Maintenance

```bash
# 1. Notify users (update status page)
curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
  -H "Authorization: OAuth TOKEN" \
  -d "incident[name]=Scheduled Maintenance" \
  -d "incident[status]=investigating"

# 2. Scale down to reduce load
kubectl scale deployment/quantum-scheduler-api --replicas=1 -n quantum-scheduler

# 3. Enable maintenance mode
kubectl patch configmap quantum-scheduler-config -n quantum-scheduler --patch '
data:
  MAINTENANCE_MODE: "true"
'

# 4. Perform maintenance tasks
# ... (specific maintenance operations)

# 5. Scale back up
kubectl scale deployment/quantum-scheduler-api --replicas=3 -n quantum-scheduler

# 6. Disable maintenance mode
kubectl patch configmap quantum-scheduler-config -n quantum-scheduler --patch '
data:
  MAINTENANCE_MODE: "false"
'

# 7. Update status page
curl -X PATCH https://api.statuspage.io/v1/pages/PAGE_ID/incidents/INCIDENT_ID \
  -H "Authorization: OAuth TOKEN" \
  -d "incident[status]=resolved"
```

### Update Procedures

```bash
# 1. Test in staging environment first
kubectl apply -f k8s/staging/ -n quantum-scheduler-staging

# 2. Blue-green deployment
kubectl set image deployment/quantum-scheduler-api \
  api=quantum-scheduler:v2.0.0 -n quantum-scheduler

# 3. Monitor deployment
kubectl rollout status deployment/quantum-scheduler-api -n quantum-scheduler

# 4. Verify health
curl -f http://quantum-scheduler-api.internal/health

# 5. If issues, rollback
kubectl rollout undo deployment/quantum-scheduler-api -n quantum-scheduler
```

## 📞 Escalation Contacts

- **On-Call Engineer**: Slack @oncall or PagerDuty
- **Engineering Manager**: manager@company.com
- **DevOps Team**: Slack #devops
- **Security Team**: security@company.com
- **Database Admin**: dba@company.com

## 📖 Additional Resources

- [Kubernetes Troubleshooting Guide](https://kubernetes.io/docs/tasks/debug-application-cluster/)
- [PostgreSQL Administration](https://www.postgresql.org/docs/current/admin.html)
- [Prometheus Monitoring](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [Quantum Computing Error Handling](https://qiskit.org/documentation/tutorials/noise/)