# Incident Response Playbook

## Overview

This document outlines the incident response procedures for the Quantum Agent Scheduler system.

## Incident Classification

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P0 - Critical** | Complete service outage | 15 minutes | All quantum backends down, API completely unavailable |
| **P1 - High** | Major functionality impaired | 1 hour | Single quantum backend down, high error rates |
| **P2 - Medium** | Minor functionality impaired | 4 hours | Performance degradation, non-critical features down |
| **P3 - Low** | Minimal impact | 24 hours | Documentation issues, minor bugs |

## Response Procedures

### P0 - Critical Incidents

#### Immediate Actions (0-15 minutes)
1. **Alert Acknowledgment**
   ```bash
   # Acknowledge alert in monitoring system
   curl -X POST https://alerts.example.com/api/alerts/{alert_id}/ack
   ```

2. **Initial Assessment**
   - Check system health dashboard
   - Verify if issue affects all users or subset
   - Document symptoms in incident ticket

3. **Communication**
   - Create incident channel: `#incident-YYYY-MM-DD-001`
   - Notify stakeholders via PagerDuty/Slack
   - Update status page

#### Investigation & Resolution (15-60 minutes)
1. **Diagnostics**
   ```bash
   # Check service health
   kubectl get pods -n quantum-scheduler
   kubectl logs -f deployment/quantum-scheduler --tail=100
   
   # Check quantum backends
   curl -s https://api.quantum-scheduler.com/v1/backends/health
   
   # Check database connectivity
   pg_isready -h db-host -p 5432
   ```

2. **Common Resolution Steps**
   - Restart services if needed
   - Scale up resources if resource-constrained
   - Switch to backup quantum backends
   - Enable classical-only mode if all quantum backends down

### P1 - High Incidents

#### Investigation Checklist
- [ ] Check Grafana dashboards for anomalies
- [ ] Review recent deployments/changes
- [ ] Analyze error logs for patterns
- [ ] Verify quantum backend availability
- [ ] Check network connectivity

#### Resolution Patterns
```bash
# Backend failover
kubectl patch deployment quantum-scheduler -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"app","env":[{"name":"FALLBACK_TO_CLASSICAL","value":"true"}]}]}}}}'

# Resource scaling
kubectl scale deployment quantum-scheduler --replicas=5

# Circuit breaker reset
curl -X POST https://api.quantum-scheduler.com/v1/admin/circuit-breaker/reset
```

## Runbooks

### Quantum Backend Failures

#### IBM Quantum Unavailable
```bash
# Check IBM Quantum status
curl -s https://api.quantum-computing.ibm.com/api/Network/devices/v/1 \
  -H "Authorization: Bearer $IBM_TOKEN"

# Failover to AWS Braket
export QUANTUM_BACKEND_PRIORITY="braket,dwave,simulator"
kubectl set env deployment/quantum-scheduler QUANTUM_BACKEND_PRIORITY=$QUANTUM_BACKEND_PRIORITY
```

#### AWS Braket Unavailable
```bash
# Check Braket service health
aws braket get-device --device-arn arn:aws:braket:::device/quantum-simulator/amazon/sv1

# Switch to alternative backends
export QUANTUM_BACKEND_PRIORITY="ibm,dwave,simulator"
kubectl set env deployment/quantum-scheduler QUANTUM_BACKEND_PRIORITY=$QUANTUM_BACKEND_PRIORITY
```

### Performance Issues

#### High Latency
1. **Identify Bottlenecks**
   ```bash
   # Check database performance
   SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
   
   # Check quantum job queue
   kubectl exec deployment/quantum-scheduler -- \
     python -c "from quantum_scheduler.monitoring import queue_stats; print(queue_stats())"
   ```

2. **Optimization Steps**
   ```bash
   # Enable caching
   kubectl set env deployment/quantum-scheduler ENABLE_SOLUTION_CACHE=true
   
   # Reduce quantum job timeout
   kubectl set env deployment/quantum-scheduler QUANTUM_TIMEOUT_SECONDS=300
   ```

#### Memory Issues
```bash
# Check memory usage
kubectl top pods -n quantum-scheduler

# Restart high-memory pods
kubectl delete pod -l app=quantum-scheduler --force --grace-period=0

# Increase memory limits
kubectl patch deployment quantum-scheduler -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
```

### Database Issues

#### Connection Pool Exhaustion
```bash
# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Kill long-running queries
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE state = 'active' AND query_start < now() - interval '5 minutes';

# Restart connection pool
kubectl rollout restart deployment/quantum-scheduler
```

#### Slow Queries
```bash
# Enable slow query logging
kubectl exec postgres-0 -- psql -c "ALTER SYSTEM SET log_min_duration_statement = 1000;"
kubectl exec postgres-0 -- psql -c "SELECT pg_reload_conf();"

# Analyze slow queries
kubectl exec postgres-0 -- tail -f /var/lib/postgresql/data/log/postgresql.log | grep "slow query"
```

## Recovery Procedures

### Data Recovery
```bash
# Restore from backup (if needed)
pg_restore --clean --if-exists -d quantum_scheduler backup.sql

# Verify data integrity
python scripts/verify_data_integrity.py
```

### Service Recovery
```bash
# Rolling restart
kubectl rollout restart deployment/quantum-scheduler

# Health check
curl -f https://api.quantum-scheduler.com/v1/health

# Smoke test
python scripts/smoke_test.py --endpoint https://api.quantum-scheduler.com
```

## Post-Incident Actions

### Immediate (within 1 hour)
- [ ] Verify full service restoration
- [ ] Update status page with resolution
- [ ] Send initial incident summary to stakeholders
- [ ] Close incident in monitoring system

### Short-term (within 24 hours)
- [ ] Conduct post-incident review meeting
- [ ] Document timeline and root cause
- [ ] Identify action items for prevention
- [ ] Update monitoring/alerting if needed

### Long-term (within 1 week)
- [ ] Implement preventive measures
- [ ] Update runbooks based on learnings
- [ ] Conduct tabletop exercises if needed
- [ ] Share learnings with broader team

## Escalation Matrix

| Role | Primary Contact | Backup | Expertise |
|------|----------------|---------|-----------|
| **On-Call Engineer** | @oncall-primary | @oncall-backup | First response, basic troubleshooting |
| **Senior Engineer** | @senior-eng | @tech-lead | Complex technical issues |
| **Quantum Expert** | @quantum-lead | @quantum-architect | Quantum backend issues |
| **SRE Lead** | @sre-lead | @platform-lead | Infrastructure issues |
| **Engineering Manager** | @eng-manager | @director-eng | Stakeholder communication |

## Communication Templates

### Initial Alert
```
🚨 INCIDENT: [P0/P1/P2] - Brief Description
Status: Investigating
Impact: [User impact description]
ETA: [Estimated resolution time]
Incident Channel: #incident-YYYY-MM-DD-001
```

### Resolution Update
```
✅ RESOLVED: [P0/P1/P2] - Brief Description
Root Cause: [Brief description]
Duration: [XX minutes]
Next Steps: Post-incident review scheduled for [time]
```

## Monitoring and Alerts

### Key Metrics to Monitor
- Quantum solver success rate
- API response times (p95, p99)
- Error rates by component
- Backend availability
- Cost per optimization
- Queue depth and processing time

### Alert Channels
- **Slack**: #alerts-quantum-scheduler
- **PagerDuty**: quantum-scheduler-oncall
- **Email**: quantum-scheduler-alerts@company.com

## Tools and Resources

### Monitoring
- **Grafana**: https://grafana.company.com/d/quantum-scheduler
- **Prometheus**: https://prometheus.company.com
- **Logs**: `kubectl logs -f deployment/quantum-scheduler`

### Quantum Backends
- **IBM Quantum**: https://quantum-computing.ibm.com/
- **AWS Braket**: https://console.aws.amazon.com/braket/
- **D-Wave Leap**: https://cloud.dwavesys.com/

### Documentation
- **Architecture**: `/docs/ARCHITECTURE.md`
- **API Reference**: https://docs.quantum-scheduler.com
- **Deployment Guide**: `/docs/deployment/`