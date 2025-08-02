# Alerting Guide

Comprehensive alerting setup for the Quantum Agent Scheduler.

## 📋 Table of Contents

- [Overview](#overview)
- [Alert Categories](#alert-categories)
- [Prometheus Rules](#prometheus-rules)
- [Grafana Alerts](#grafana-alerts)
- [Notification Channels](#notification-channels)
- [Alert Playbooks](#alert-playbooks)
- [Escalation Procedures](#escalation-procedures)
- [Testing Alerts](#testing-alerts)

## 🎯 Overview

Our alerting strategy covers:
- **Application Health**: API availability, response times, error rates
- **Infrastructure**: Resource utilization, container health
- **Quantum Backends**: Provider availability, cost thresholds
- **Business Metrics**: Scheduling success rates, user impact
- **Security**: Authentication failures, unusual access patterns

## 🚨 Alert Categories

### Critical (P0) - Immediate Response Required
- Service completely down
- Data corruption detected
- Security breach indicators
- Quantum hardware failures causing complete service disruption

### High (P1) - Response within 1 hour
- Elevated error rates (>5%)
- High response times (>5s p95)
- Database connection issues
- Major quantum backend unavailable

### Medium (P2) - Response within 4 hours
- Resource utilization warnings
- Performance degradation
- Non-critical backend issues
- Elevated costs

### Low (P3) - Response within 24 hours
- Informational alerts
- Capacity planning warnings
- Documentation updates needed

## 📊 Prometheus Rules

### Application Health Rules

```yaml
# observability/rules/application.yml
groups:
  - name: quantum_scheduler_app
    rules:
    # Service availability
    - alert: ServiceDown
      expr: up{job="quantum-scheduler"} == 0
      for: 1m
      labels:
        severity: critical
        component: api
      annotations:
        summary: "Quantum Scheduler service is down"
        description: "Service {{ $labels.instance }} has been down for more than 1 minute"
        runbook_url: "https://docs.quantum-scheduler.io/runbooks/service-down"
    
    # High error rate
    - alert: HighErrorRate
      expr: |
        (
          rate(http_requests_total{job="quantum-scheduler",status=~"5.."}[5m]) /
          rate(http_requests_total{job="quantum-scheduler"}[5m])
        ) * 100 > 5
      for: 5m
      labels:
        severity: high
        component: api
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }}% for the last 5 minutes"
    
    # High response time
    - alert: HighResponseTime
      expr: |
        histogram_quantile(0.95,
          rate(http_request_duration_seconds_bucket{job="quantum-scheduler"}[5m])
        ) > 5
      for: 5m
      labels:
        severity: high
        component: api
      annotations:
        summary: "High response time detected"
        description: "95th percentile response time is {{ $value }}s"
    
    # Low scheduling success rate
    - alert: LowSchedulingSuccessRate
      expr: |
        (
          rate(scheduling_requests_total{status="success"}[10m]) /
          rate(scheduling_requests_total[10m])
        ) * 100 < 90
      for: 10m
      labels:
        severity: medium
        component: scheduler
      annotations:
        summary: "Low scheduling success rate"
        description: "Scheduling success rate is {{ $value }}% over the last 10 minutes"
```

### Infrastructure Rules

```yaml
# observability/rules/infrastructure.yml
groups:
  - name: quantum_scheduler_infra
    rules:
    # High CPU usage
    - alert: HighCPUUsage
      expr: |
        (
          rate(container_cpu_usage_seconds_total{name=~".*quantum-scheduler.*"}[5m]) * 100
        ) > 80
      for: 10m
      labels:
        severity: medium
        component: infrastructure
      annotations:
        summary: "High CPU usage detected"
        description: "CPU usage is {{ $value }}% for container {{ $labels.name }}"
    
    # High memory usage
    - alert: HighMemoryUsage
      expr: |
        (
          container_memory_usage_bytes{name=~".*quantum-scheduler.*"} /
          container_spec_memory_limit_bytes{name=~".*quantum-scheduler.*"}
        ) * 100 > 85
      for: 5m
      labels:
        severity: medium
        component: infrastructure
      annotations:
        summary: "High memory usage detected"
        description: "Memory usage is {{ $value }}% for container {{ $labels.name }}"
    
    # Disk space warning
    - alert: DiskSpaceWarning
      expr: |
        (
          (node_filesystem_size_bytes - node_filesystem_free_bytes) /
          node_filesystem_size_bytes
        ) * 100 > 80
      for: 5m
      labels:
        severity: medium
        component: infrastructure
      annotations:
        summary: "Disk space running low"
        description: "Disk usage is {{ $value }}% on {{ $labels.instance }}"
```

### Quantum Backend Rules

```yaml
# observability/rules/quantum.yml
groups:
  - name: quantum_backends
    rules:
    # Quantum backend unavailable
    - alert: QuantumBackendUnavailable
      expr: quantum_backend_available{provider=~"braket|qiskit|dwave|azure"} == 0
      for: 2m
      labels:
        severity: high
        component: quantum
      annotations:
        summary: "Quantum backend unavailable"
        description: "{{ $labels.provider }} backend has been unavailable for 2 minutes"
    
    # High quantum costs
    - alert: HighQuantumCosts
      expr: |
        increase(quantum_cost_total_dollars[1h]) > 50
      for: 0m
      labels:
        severity: medium
        component: quantum
      annotations:
        summary: "High quantum computing costs"
        description: "Quantum costs increased by ${{ $value }} in the last hour"
    
    # Quantum queue length high
    - alert: QuantumQueueLengthHigh
      expr: quantum_job_queue_length > 100
      for: 5m
      labels:
        severity: medium
        component: quantum
      annotations:
        summary: "Quantum job queue length high"
        description: "{{ $labels.provider }} queue has {{ $value }} jobs pending"
    
    # Quantum success rate low
    - alert: QuantumSuccessRateLow
      expr: |
        (
          rate(quantum_jobs_total{status="success"}[10m]) /
          rate(quantum_jobs_total[10m])
        ) * 100 < 80
      for: 10m
      labels:
        severity: high
        component: quantum
      annotations:
        summary: "Low quantum job success rate"
        description: "Quantum job success rate is {{ $value }}% for {{ $labels.provider }}"
```

### Security Rules

```yaml
# observability/rules/security.yml
groups:
  - name: security
    rules:
    # High authentication failure rate
    - alert: HighAuthFailureRate
      expr: |
        rate(auth_failures_total[5m]) > 10
      for: 2m
      labels:
        severity: high
        component: security
      annotations:
        summary: "High authentication failure rate"
        description: "{{ $value }} authentication failures per second"
    
    # Unusual access patterns
    - alert: UnusualAccessPattern
      expr: |
        rate(http_requests_total{job="quantum-scheduler"}[5m]) > 
        rate(http_requests_total{job="quantum-scheduler"}[1h] offset 1h) * 5
      for: 5m
      labels:
        severity: medium
        component: security
      annotations:
        summary: "Unusual access pattern detected"
        description: "Request rate is 5x higher than usual"
```

## 📈 Grafana Alerts

### Alert Rules Configuration

```json
{
  "alert": {
    "id": 1,
    "dashboardId": 1,
    "panelId": 1,
    "name": "Quantum Scheduler API Response Time",
    "message": "API response time is too high",
    "frequency": "10s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A",
          "model": {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"quantum-scheduler\"}[5m]))",
            "interval": "",
            "refId": "A"
          }
        },
        "reducer": {
          "type": "last",
          "params": []
        },
        "evaluator": {
          "params": [5],
          "type": "gt"
        }
      }
    ],
    "executionErrorState": "alerting",
    "noDataState": "no_data",
    "for": "5m"
  }
}
```

### Dashboard Alerts

```yaml
# grafana/dashboards/alerts.json
{
  "dashboard": {
    "title": "Quantum Scheduler Alerts",
    "panels": [
      {
        "title": "Active Alerts",
        "type": "stat",
        "targets": [
          {
            "expr": "ALERTS{alertstate=\"firing\",job=\"quantum-scheduler\"}",
            "legendFormat": "{{ alertname }}"
          }
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {
                "params": [1],
                "type": "gt"
              },
              "operator": {
                "type": "and"
              },
              "query": {
                "params": ["A", "1m", "now"]
              },
              "reducer": {
                "params": [],
                "type": "last"
              },
              "type": "query"
            }
          ],
          "executionErrorState": "alerting",
          "for": "1m",
          "frequency": "10s",
          "handler": 1,
          "name": "Active Alerts Panel",
          "noDataState": "no_data"
        }
      }
    ]
  }
}
```

## 📢 Notification Channels

### Slack Integration

```yaml
# alertmanager/config.yml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: high
    receiver: 'high-alerts'

receivers:
- name: 'default-receiver'
  slack_configs:
  - channel: '#alerts'
    title: 'Quantum Scheduler Alert'
    text: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

- name: 'critical-alerts'
  slack_configs:
  - channel: '#critical-alerts'
    title: '🚨 CRITICAL: Quantum Scheduler'
    text: |
      {{ range .Alerts }}
      **CRITICAL ALERT**
      Summary: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Runbook: {{ .Annotations.runbook_url }}
      {{ end }}
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_KEY'
    description: '{{ .CommonAnnotations.summary }}'
```

### Email Notifications

```yaml
# Email configuration
- name: 'email-alerts'
  email_configs:
  - to: 'quantum-scheduler-alerts@company.com'
    from: 'alerts@company.com'
    smarthost: 'smtp.company.com:587'
    auth_username: 'alerts@company.com'
    auth_password: 'password'
    subject: 'Quantum Scheduler Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      
      Description: {{ .Annotations.description }}
      
      Labels:
      {{ range .Labels.SortedPairs }}
      - {{ .Name }}: {{ .Value }}
      {{ end }}
      
      Runbook: {{ .Annotations.runbook_url }}
      {{ end }}
```

### PagerDuty Integration

```yaml
# PagerDuty configuration
- name: 'pagerduty-critical'
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
    description: '{{ .CommonAnnotations.summary }}'
    details:
      firing: '{{ .Alerts.Firing | len }}'
      resolved: '{{ .Alerts.Resolved | len }}'
      alert_details: |
        {{ range .Alerts }}
        - Alert: {{ .Annotations.summary }}
        - Description: {{ .Annotations.description }}
        - Runbook: {{ .Annotations.runbook_url }}
        {{ end }}
```

## 📖 Alert Playbooks

### Service Down Playbook

```markdown
# Service Down Response

## Immediate Actions (0-5 minutes)
1. Check service status: `kubectl get pods -n quantum-scheduler`
2. Check recent deployments: `kubectl rollout history deployment/quantum-scheduler-api`
3. Check logs: `kubectl logs -f deployment/quantum-scheduler-api --tail=100`
4. Verify external dependencies (database, Redis, quantum providers)

## Investigation (5-15 minutes)
1. Check resource utilization: `kubectl top pods -n quantum-scheduler`
2. Check recent events: `kubectl get events -n quantum-scheduler --sort-by='.lastTimestamp'`
3. Verify configuration: `kubectl describe configmap quantum-scheduler-config`
4. Check network connectivity

## Resolution
1. If OOMKilled: Increase memory limits
2. If CrashLoopBackOff: Check application logs and fix configuration
3. If resource exhaustion: Scale up or optimize
4. If external dependency issue: Failover or wait for recovery

## Communication
- Update status page
- Notify stakeholders via Slack #incidents
- Create incident report
```

### High Error Rate Playbook

```markdown
# High Error Rate Response

## Immediate Actions
1. Identify error types: Check Grafana dashboard
2. Check recent changes: Review deployment history
3. Verify quantum backend status
4. Check database connectivity

## Investigation
1. Analyze error logs for patterns
2. Check specific endpoints with high error rates
3. Verify authentication service
4. Check rate limiting configuration

## Resolution
1. If deployment issue: Rollback recent changes
2. If backend issue: Switch to fallback providers
3. If database issue: Check connections and queries
4. If rate limiting: Adjust limits or scaling

## Prevention
- Add more comprehensive testing
- Improve monitoring for specific error types
- Review deployment procedures
```

### High Response Time Playbook

```markdown
# High Response Time Response

## Immediate Actions
1. Check current load: Grafana performance dashboard
2. Check resource utilization (CPU, memory, I/O)
3. Verify database performance
4. Check quantum provider response times

## Investigation
1. Identify slow endpoints using APM traces
2. Check database query performance
3. Analyze quantum job queues
4. Review application performance metrics

## Resolution
1. Scale up if resource constrained
2. Optimize slow database queries
3. Implement caching for frequent requests
4. Switch to faster quantum backends if available

## Long-term
- Capacity planning review
- Performance optimization backlog
- SLO/SLA review
```

## 🔄 Escalation Procedures

### Escalation Matrix

| Severity | Initial Response | Escalation Time | Escalation Target |
|----------|------------------|-----------------|-------------------|
| Critical | On-call engineer | 15 minutes | Engineering manager |
| High | On-call engineer | 1 hour | Team lead |
| Medium | Team member | 4 hours | On-call engineer |
| Low | Team member | 24 hours | Team lead |

### On-Call Rotation

```yaml
# PagerDuty schedule configuration
schedules:
  - name: "Quantum Scheduler Primary"
    time_zone: "UTC"
    layers:
      - name: "Primary On-Call"
        start: "2024-01-01T00:00:00Z"
        rotation_virtual_start: "2024-01-01T00:00:00Z"
        rotation_turn_length_seconds: 604800  # 1 week
        users:
          - user_reference: "engineer1@company.com"
          - user_reference: "engineer2@company.com"
          - user_reference: "engineer3@company.com"
```

### Escalation Automation

```yaml
# Escalation rules
escalation_policies:
  - name: "Quantum Scheduler Escalation"
    escalation_rules:
      - escalation_delay_in_minutes: 0
        targets:
          - type: "user_reference"
            id: "primary_oncall"
      - escalation_delay_in_minutes: 15
        targets:
          - type: "user_reference" 
            id: "secondary_oncall"
      - escalation_delay_in_minutes: 30
        targets:
          - type: "schedule_reference"
            id: "engineering_manager_schedule"
```

## 🧪 Testing Alerts

### Alert Testing Framework

```python
# tests/test_alerts.py
import pytest
import time
from prometheus_client import CollectorRegistry, Counter, push_to_gateway

class TestAlerts:
    """Test alert conditions and notifications."""
    
    def test_service_down_alert(self, prometheus_gateway):
        """Test service down alert triggers correctly."""
        # Simulate service down
        registry = CollectorRegistry()
        up_metric = Counter('up', 'Service up status', registry=registry)
        up_metric._value._value = 0  # Set to down
        
        push_to_gateway(prometheus_gateway, job='quantum-scheduler', registry=registry)
        
        # Wait for alert to fire
        time.sleep(70)  # Alert fires after 1 minute
        
        # Verify alert is active
        alerts = self.get_active_alerts()
        assert any(alert['alertname'] == 'ServiceDown' for alert in alerts)
    
    def test_high_error_rate_alert(self, metrics_client):
        """Test high error rate alert."""
        # Simulate high error rate
        for _ in range(100):
            metrics_client.http_requests_total.labels(status='500').inc()
        
        # Wait for alert evaluation
        time.sleep(300)  # 5 minutes
        
        alerts = self.get_active_alerts()
        assert any(alert['alertname'] == 'HighErrorRate' for alert in alerts)
    
    def get_active_alerts(self):
        """Get currently active alerts from Alertmanager."""
        import requests
        response = requests.get('http://alertmanager:9093/api/v1/alerts')
        return response.json()['data']
```

### Manual Alert Testing

```bash
# Test alert firing
curl -X POST http://alertmanager:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "service": "quantum-scheduler",
      "severity": "critical"
    },
    "annotations": {
      "summary": "Test alert for validation",
      "description": "This is a test alert to verify notification channels"
    },
    "startsAt": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'",
    "endsAt": "'$(date -u -d '+5 minutes' +%Y-%m-%dT%H:%M:%S.%3NZ)'",
    "generatorURL": "http://prometheus:9090/graph"
  }]'

# Test notification channels
# Slack webhook test
curl -X POST https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK \
  -H "Content-Type: application/json" \
  -d '{"text": "Test alert notification from Quantum Scheduler"}'

# Email test (if using SMTP)
echo "Test alert email" | mail -s "Test Alert" alerts@company.com
```

### Alert Validation Checklist

- [ ] Alert fires within expected timeframe
- [ ] Alert resolves when condition clears
- [ ] Notifications sent to correct channels
- [ ] Escalation procedures work correctly
- [ ] Runbook links are accessible
- [ ] Alert descriptions are clear and actionable
- [ ] No false positives in production
- [ ] Alert thresholds are appropriate

## 📚 Best Practices

1. **Alert Fatigue Prevention**
   - Set appropriate thresholds
   - Use alert grouping and inhibition
   - Regular alert review and cleanup

2. **Meaningful Alerts**
   - Every alert should be actionable
   - Include context and runbook links
   - Clear escalation paths

3. **Testing**
   - Regular alert testing
   - Chaos engineering exercises
   - Alert runbook validation

4. **Documentation**
   - Keep runbooks updated
   - Document escalation procedures
   - Alert review processes

5. **Continuous Improvement**
   - Post-incident alert reviews
   - Metrics on alert effectiveness
   - Regular threshold tuning