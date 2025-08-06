# API Cost Monitoring System - Troubleshooting Guide

## Table of Contents

1. [Overview](#overview)
2. [Diagnostic Tools](#diagnostic-tools)
3. [Common Issues](#common-issues)
4. [Configuration Problems](#configuration-problems)
5. [Database Issues](#database-issues)
6. [Alert System Problems](#alert-system-problems)
7. [Performance Issues](#performance-issues)
8. [Integration Problems](#integration-problems)
9. [Recovery Procedures](#recovery-procedures)
10. [Prevention and Maintenance](#prevention-and-maintenance)

---

## Overview

This troubleshooting guide provides systematic approaches to diagnosing and resolving common issues with the API Cost Monitoring System. It includes diagnostic procedures, common problem solutions, and recovery strategies.

### Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

```bash
# 1. Check system status
sudo systemctl status lightrag-monitor

# 2. Check recent logs
tail -f /opt/lightrag/logs/lightrag_integration.log

# 3. Verify configuration
python3 -c "from lightrag_integration.config import LightRAGConfig; config = LightRAGConfig.get_config(); config.validate()"

# 4. Test API connectivity
curl -I https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"

# 5. Check database
sqlite3 /opt/lightrag/data/cost_tracking.db ".tables"

# 6. Verify disk space
df -h /opt/lightrag

# 7. Check memory usage
free -m
```

### Getting Help

1. **Check Logs**: Always start with system and application logs
2. **Run Diagnostics**: Use built-in diagnostic tools
3. **Verify Configuration**: Ensure all settings are correct
4. **Test Components**: Isolate and test individual components
5. **Recovery**: Follow recovery procedures if needed

---

## Diagnostic Tools

### Built-in Diagnostics

```python
from lightrag_integration.diagnostics import BudgetSystemDiagnostics

# Initialize diagnostics
diagnostics = BudgetSystemDiagnostics(budget_system)

# Run comprehensive health check
health_report = diagnostics.run_health_check()
print(f"Overall status: {health_report['overall_status']}")

if health_report['issues']:
    print("Issues found:")
    for issue in health_report['issues']:
        print(f"  - {issue}")

# Analyze performance
performance_report = diagnostics.analyze_performance(duration_minutes=60)
print(f"Average response time: {performance_report['performance_metrics'].get('avg_response_time_ms', 'N/A')}ms")

# Export debug data
debug_file = diagnostics.export_debug_data("/tmp/lightrag_debug.json")
print(f"Debug data exported to: {debug_file}")
```

### System Health Check Script

```bash
#!/bin/bash
# lightrag_health_check.sh

echo "=== LightRAG Budget Monitor Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check service status
echo "1. Service Status:"
if systemctl is-active --quiet lightrag-monitor; then
    echo "   ✅ Service is running"
else
    echo "   ❌ Service is not running"
    echo "   Status: $(systemctl is-active lightrag-monitor)"
fi
echo ""

# Check configuration
echo "2. Configuration:"
if [ -f /opt/lightrag/config/.env ]; then
    echo "   ✅ Configuration file exists"
    if grep -q "OPENAI_API_KEY=" /opt/lightrag/config/.env; then
        echo "   ✅ API key configured"
    else
        echo "   ❌ API key not found in configuration"
    fi
else
    echo "   ❌ Configuration file missing"
fi
echo ""

# Check database
echo "3. Database:"
if [ -f /opt/lightrag/data/cost_tracking.db ]; then
    echo "   ✅ Database file exists"
    DB_SIZE=$(du -h /opt/lightrag/data/cost_tracking.db | cut -f1)
    echo "   Database size: $DB_SIZE"
    
    # Check database integrity
    if sqlite3 /opt/lightrag/data/cost_tracking.db "PRAGMA integrity_check;" | grep -q "ok"; then
        echo "   ✅ Database integrity check passed"
    else
        echo "   ❌ Database integrity check failed"
    fi
else
    echo "   ❌ Database file missing"
fi
echo ""

# Check disk space
echo "4. Disk Space:"
DISK_USAGE=$(df -h /opt/lightrag | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    echo "   ✅ Disk usage: ${DISK_USAGE}%"
else
    echo "   ⚠️  High disk usage: ${DISK_USAGE}%"
fi
echo ""

# Check memory
echo "5. Memory Usage:"
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "   Memory usage: ${MEMORY_USAGE}%"
echo ""

# Check logs for errors
echo "6. Recent Errors:"
if [ -f /opt/lightrag/logs/lightrag_integration.log ]; then
    ERROR_COUNT=$(tail -1000 /opt/lightrag/logs/lightrag_integration.log | grep -c "ERROR")
    if [ "$ERROR_COUNT" -eq 0 ]; then
        echo "   ✅ No recent errors found"
    else
        echo "   ⚠️  Found $ERROR_COUNT errors in last 1000 log lines"
        echo "   Recent errors:"
        tail -1000 /opt/lightrag/logs/lightrag_integration.log | grep "ERROR" | tail -5
    fi
else
    echo "   ❌ Log file not found"
fi
echo ""

# Network connectivity
echo "7. Network Connectivity:"
if curl -s --connect-timeout 10 https://api.openai.com/v1/models > /dev/null; then
    echo "   ✅ OpenAI API reachable"
else
    echo "   ❌ Cannot reach OpenAI API"
fi
echo ""

echo "=== Health Check Complete ==="
```

### Log Analysis Tools

```bash
# Extract cost-related log entries
grep -E "(cost|budget|alert)" /opt/lightrag/logs/lightrag_integration.log | tail -50

# Find error patterns
grep -E "(ERROR|CRITICAL)" /opt/lightrag/logs/lightrag_integration.log | \
    awk '{print $3}' | sort | uniq -c | sort -rn

# Analyze response times
grep "response_time" /opt/lightrag/logs/lightrag_integration.log | \
    grep -oE "response_time[^,]*" | \
    sed 's/response_time[^0-9]*//g' | \
    awk '{sum+=$1; count++} END {printf "Average response time: %.2f ms\n", sum/count}'

# Check for memory issues
grep -i "memory\|oom\|killed" /var/log/syslog | grep lightrag

# Monitor real-time logs
tail -f /opt/lightrag/logs/lightrag_integration.log | grep -E "(ERROR|WARN|cost|budget)"
```

---

## Common Issues

### Issue 1: Service Won't Start

**Symptoms:**
- `systemctl start lightrag-monitor` fails
- Service shows "failed" status
- Application doesn't respond to requests

**Diagnosis:**
```bash
# Check service status
sudo systemctl status lightrag-monitor -l

# Check service logs
sudo journalctl -u lightrag-monitor -f

# Check application logs
tail -50 /opt/lightrag/logs/service.log

# Test configuration manually
sudo -u lightrag /opt/lightrag/venv/bin/python -c "
from lightrag_integration.config import LightRAGConfig
config = LightRAGConfig.get_config()
config.validate()
print('Configuration valid')
"
```

**Common Causes & Solutions:**

1. **Invalid Configuration**
   ```bash
   # Check for missing API key
   grep OPENAI_API_KEY /opt/lightrag/config/.env
   
   # Validate configuration
   python3 -c "from lightrag_integration.config import LightRAGConfig; LightRAGConfig.get_config().validate()"
   ```

2. **Permission Issues**
   ```bash
   # Fix ownership
   sudo chown -R lightrag:lightrag /opt/lightrag
   
   # Fix permissions
   sudo chmod 750 /opt/lightrag/data
   sudo chmod 600 /opt/lightrag/config/.env
   ```

3. **Python Environment Issues**
   ```bash
   # Recreate virtual environment
   sudo -u lightrag bash -c "
   cd /opt/lightrag
   rm -rf venv
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   "
   ```

4. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   sudo lsof -i :8000
   
   # Kill conflicting process or change port
   sudo kill -9 $(lsof -t -i:8000)
   ```

### Issue 2: Budget Tracking Not Working

**Symptoms:**
- Cost data not being recorded
- Budget percentages show as 0%
- No budget alerts being generated

**Diagnosis:**
```python
# Test budget tracking manually
from lightrag_integration import BudgetManagementFactory
from lightrag_integration.config import LightRAGConfig

config = LightRAGConfig.get_config()
system = BudgetManagementFactory.create_complete_system(config)

# Test operation tracking
with system.track_operation("test_operation") as tracker:
    tracker.set_cost(0.01)
    tracker.set_tokens(prompt=10, completion=5)

# Check if cost was recorded
status = system.get_budget_status()
print(f"Daily cost: ${status['daily_budget']['total_cost']:.4f}")
```

**Solutions:**

1. **Database Connection Issues**
   ```bash
   # Check database file exists and is writable
   ls -la /opt/lightrag/data/cost_tracking.db
   
   # Test database connection
   sqlite3 /opt/lightrag/data/cost_tracking.db ".tables"
   
   # Check for database locks
   lsof /opt/lightrag/data/cost_tracking.db
   ```

2. **Missing Cost Persistence**
   ```python
   # Verify cost persistence is enabled
   config = LightRAGConfig.get_config()
   print(f"Cost persistence enabled: {config.cost_persistence_enabled}")
   
   # Check if cost records table exists
   import sqlite3
   conn = sqlite3.connect(config.cost_db_path)
   cursor = conn.cursor()
   cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
   tables = cursor.fetchall()
   print(f"Database tables: {tables}")
   ```

3. **Configuration Issues**
   ```bash
   # Verify budget limits are set
   echo "Daily budget: $LIGHTRAG_DAILY_BUDGET_LIMIT"
   echo "Monthly budget: $LIGHTRAG_MONTHLY_BUDGET_LIMIT"
   ```

### Issue 3: High Memory Usage

**Symptoms:**
- System running out of memory
- OOM killer terminating processes
- Slow response times

**Diagnosis:**
```bash
# Check memory usage
free -m
ps aux | grep python | grep lightrag

# Check for memory leaks
sudo -u lightrag python3 -c "
import psutil
import time
from lightrag_integration import BudgetManagementSystem

# Monitor memory usage
process = psutil.Process()
print(f'Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')

# Simulate operations
system = BudgetManagementSystem(...)
for i in range(100):
    with system.track_operation('test') as tracker:
        tracker.set_cost(0.001)
    if i % 20 == 0:
        print(f'After {i} operations: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**Solutions:**

1. **Implement Memory Limits**
   ```bash
   # Add memory limits to systemd service
   sudo systemctl edit lightrag-monitor
   
   # Add the following:
   [Service]
   MemoryLimit=1G
   MemoryHigh=800M
   ```

2. **Optimize Database Operations**
   ```python
   # Implement connection pooling and batch operations
   from lightrag_integration.optimizations import OptimizedCostDatabase
   
   # Use batch inserts instead of individual records
   cost_records = [...]
   database.batch_insert_cost_records(cost_records)
   ```

3. **Configure Garbage Collection**
   ```python
   # Add to application startup
   import gc
   gc.set_threshold(700, 10, 10)
   ```

### Issue 4: API Rate Limiting

**Symptoms:**
- Getting 429 "Too Many Requests" errors
- Intermittent API failures
- High error rates in logs

**Diagnosis:**
```bash
# Check for rate limit errors in logs
grep -E "(429|rate.limit|too.many)" /opt/lightrag/logs/lightrag_integration.log

# Monitor request rate
grep "API call" /opt/lightrag/logs/lightrag_integration.log | \
    tail -1000 | \
    awk '{print $1" "$2}' | \
    uniq -c | \
    tail -10
```

**Solutions:**

1. **Implement Request Throttling**
   ```python
   # Add rate limiting to configuration
   config = LightRAGConfig(
       max_async=8,  # Reduce concurrent requests
       # Add backoff delays
   )
   
   # Implement exponential backoff
   import time
   import random
   
   def api_call_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               if attempt < max_retries - 1:
                   delay = (2 ** attempt) + random.uniform(0, 1)
                   time.sleep(delay)
               else:
                   raise
   ```

2. **Use Circuit Breakers**
   ```python
   # Configure circuit breaker for rate limiting
   from lightrag_integration.cost_based_circuit_breaker import CostThresholdRule, CostThresholdType
   
   rate_limit_rule = CostThresholdRule(
       rule_id="rate_limit_protection",
       threshold_type=CostThresholdType.RATE_BASED,
       threshold_value=50.0,  # 50 requests per hour
       action="throttle",
       throttle_factor=0.5
   )
   ```

---

## Configuration Problems

### Missing Environment Variables

**Problem:** Application fails to start due to missing configuration.

**Diagnosis:**
```bash
# Check all required environment variables
env | grep LIGHTRAG
env | grep OPENAI
env | grep ALERT

# Verify .env file is loaded
sudo -u lightrag bash -c "source /opt/lightrag/config/.env && env | grep LIGHTRAG"
```

**Solution:**
```bash
# Create complete environment file
cat > /opt/lightrag/config/.env << 'EOF'
# Required
OPENAI_API_KEY=your-key-here

# Budget Configuration
LIGHTRAG_DAILY_BUDGET_LIMIT=100.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=2000.0

# System Configuration
LIGHTRAG_WORKING_DIR=/opt/lightrag/data
LIGHTRAG_LOG_DIR=/opt/lightrag/logs
LIGHTRAG_LOG_LEVEL=INFO

# Optional Alert Configuration
ALERT_EMAIL_SMTP_SERVER=smtp.yourdomain.com
ALERT_EMAIL_USERNAME=alerts@yourdomain.com
ALERT_EMAIL_PASSWORD=your-password
EOF

# Set secure permissions
chmod 600 /opt/lightrag/config/.env
chown lightrag:lightrag /opt/lightrag/config/.env
```

### Invalid Configuration Values

**Problem:** Configuration validation fails.

**Diagnosis:**
```python
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError

try:
    config = LightRAGConfig.get_config()
    config.validate()
    print("Configuration is valid")
except LightRAGConfigError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Common Solutions:**

1. **Numeric Values**
   ```bash
   # Ensure numeric values are properly formatted
   LIGHTRAG_DAILY_BUDGET_LIMIT=100.0  # Not "100"
   LIGHTRAG_MAX_ASYNC=16              # Not "sixteen"
   ```

2. **Path Values**
   ```bash
   # Ensure paths exist and are accessible
   mkdir -p /opt/lightrag/data
   mkdir -p /opt/lightrag/logs
   chown -R lightrag:lightrag /opt/lightrag
   ```

3. **Boolean Values**
   ```bash
   # Use proper boolean formats
   LIGHTRAG_ENABLE_COST_TRACKING=true    # Not "True" or "yes"
   LIGHTRAG_ENABLE_BUDGET_ALERTS=false   # Not "False" or "no"
   ```

---

## Database Issues

### Database Corruption

**Symptoms:**
- SQLite database errors
- Data inconsistencies
- Application crashes on database operations

**Diagnosis:**
```bash
# Check database integrity
sqlite3 /opt/lightrag/data/cost_tracking.db "PRAGMA integrity_check;"

# Check database size and growth
ls -lh /opt/lightrag/data/cost_tracking.db
du -h /opt/lightrag/data/cost_tracking.db

# Look for corruption signs in logs
grep -i "corrupt\|database.*error\|sqlite.*error" /opt/lightrag/logs/lightrag_integration.log
```

**Recovery:**
```bash
# 1. Stop the service
sudo systemctl stop lightrag-monitor

# 2. Backup corrupted database
cp /opt/lightrag/data/cost_tracking.db /opt/lightrag/backups/cost_tracking.db.corrupt

# 3. Try to recover
sqlite3 /opt/lightrag/data/cost_tracking.db ".recover" | \
sqlite3 /opt/lightrag/data/cost_tracking_recovered.db

# 4. Verify recovered database
sqlite3 /opt/lightrag/data/cost_tracking_recovered.db "PRAGMA integrity_check;"

# 5. Replace if recovery successful
if [ $? -eq 0 ]; then
    mv /opt/lightrag/data/cost_tracking.db /opt/lightrag/backups/cost_tracking.db.old
    mv /opt/lightrag/data/cost_tracking_recovered.db /opt/lightrag/data/cost_tracking.db
    chown lightrag:lightrag /opt/lightrag/data/cost_tracking.db
fi

# 6. Restart service
sudo systemctl start lightrag-monitor
```

### Database Lock Issues

**Problem:** Database operations hang or fail with "database is locked" errors.

**Diagnosis:**
```bash
# Check for processes holding database locks
lsof /opt/lightrag/data/cost_tracking.db

# Check for zombie processes
ps aux | grep lightrag | grep -v grep

# Look for lock timeouts in logs
grep -i "lock\|timeout\|busy" /opt/lightrag/logs/lightrag_integration.log
```

**Solutions:**

1. **Kill Hanging Processes**
   ```bash
   # Kill processes holding locks
   sudo pkill -f "lightrag"
   
   # Remove lock files if they exist
   rm -f /opt/lightrag/data/cost_tracking.db-shm
   rm -f /opt/lightrag/data/cost_tracking.db-wal
   ```

2. **Configure WAL Mode**
   ```bash
   # Enable WAL mode for better concurrency
   sqlite3 /opt/lightrag/data/cost_tracking.db "PRAGMA journal_mode=WAL;"
   sqlite3 /opt/lightrag/data/cost_tracking.db "PRAGMA synchronous=NORMAL;"
   ```

### Database Performance Issues

**Problem:** Slow database operations affecting system performance.

**Optimization:**
```bash
# Analyze database
sqlite3 /opt/lightrag/data/cost_tracking.db << 'EOF'
ANALYZE;
PRAGMA optimize;
.exit
EOF

# Vacuum database to reclaim space
sqlite3 /opt/lightrag/data/cost_tracking.db "VACUUM;"

# Check indexes
sqlite3 /opt/lightrag/data/cost_tracking.db ".indices"

# Add missing indexes if needed
sqlite3 /opt/lightrag/data/cost_tracking.db << 'EOF'
CREATE INDEX IF NOT EXISTS idx_cost_records_timestamp ON cost_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_cost_records_operation_type ON cost_records(operation_type);
CREATE INDEX IF NOT EXISTS idx_cost_records_research_category ON cost_records(research_category);
.exit
EOF
```

---

## Alert System Problems

### Email Alerts Not Working

**Diagnosis:**
```python
# Test email configuration
from lightrag_integration.alert_system import AlertNotificationSystem, EmailAlertConfig

email_config = EmailAlertConfig(
    smtp_server="smtp.yourdomain.com",
    smtp_port=587,
    username="alerts@yourdomain.com",
    password="your-password",
    sender_email="alerts@yourdomain.com",
    recipient_emails=["admin@yourdomain.com"]
)

alert_system = AlertNotificationSystem(email_config)
test_result = alert_system.test_email_channel()
print(f"Email test result: {test_result}")
```

**Common Solutions:**

1. **SMTP Authentication Issues**
   ```bash
   # Test SMTP connection manually
   telnet smtp.yourdomain.com 587
   # Or use swaks for testing
   swaks --to admin@yourdomain.com --from alerts@yourdomain.com --server smtp.yourdomain.com:587 --auth-user alerts@yourdomain.com --auth-password your-password
   ```

2. **Firewall/Network Issues**
   ```bash
   # Check if SMTP ports are accessible
   nc -zv smtp.yourdomain.com 587
   nc -zv smtp.yourdomain.com 465
   
   # Check DNS resolution
   nslookup smtp.yourdomain.com
   ```

3. **SSL/TLS Issues**
   ```python
   # Update email configuration for SSL/TLS
   email_config = EmailAlertConfig(
       smtp_server="smtp.yourdomain.com",
       smtp_port=587,
       use_tls=True,  # Enable TLS
       username="alerts@yourdomain.com",
       password="your-password"
   )
   ```

### Slack Alerts Not Working

**Diagnosis:**
```bash
# Test Slack webhook manually
curl -X POST -H 'Content-type: application/json' \
--data '{"text":"Test message from LightRAG"}' \
$ALERT_SLACK_WEBHOOK_URL

# Check webhook URL format
echo $ALERT_SLACK_WEBHOOK_URL | grep -E "^https://hooks\.slack\.com/services/"
```

**Solutions:**

1. **Invalid Webhook URL**
   ```bash
   # Verify webhook URL format
   # Should be: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
   
   # Update configuration with correct URL
   export ALERT_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

2. **Channel Permissions**
   ```python
   # Verify Slack app has permission to post to channel
   slack_config = SlackAlertConfig(
       webhook_url=webhook_url,
       channel="#budget-alerts",  # Use # for public channels
       # channel="@username"      # Use @ for direct messages
   )
   ```

### Alert Delivery Delays

**Problem:** Alerts are delivered significantly after events occur.

**Diagnosis:**
```python
# Check alert system performance
delivery_stats = alert_system.get_delivery_stats()
print(f"Average delivery time: {delivery_stats.get('avg_delivery_time_ms', 'N/A')}ms")

# Check for queued alerts
pending_alerts = alert_system.get_pending_alerts()
print(f"Pending alerts: {len(pending_alerts)}")
```

**Solutions:**

1. **Increase Processing Threads**
   ```python
   alert_config = AlertConfig(
       max_concurrent_deliveries=5,  # Increase from default
       delivery_timeout_seconds=10   # Reduce timeout
   )
   ```

2. **Implement Alert Prioritization**
   ```python
   # Prioritize critical alerts
   if alert.alert_level == AlertLevel.CRITICAL:
       alert_system.send_alert_immediate(alert)
   else:
       alert_system.send_alert_async(alert)
   ```

---

## Performance Issues

### Slow API Response Times

**Problem:** API calls taking longer than expected.

**Diagnosis:**
```python
# Analyze response times from logs
import re
import statistics

log_file = "/opt/lightrag/logs/lightrag_integration.log"
response_times = []

with open(log_file, 'r') as f:
    for line in f:
        match = re.search(r'response_time.*?(\d+(?:\.\d+)?)(?:ms|s)', line)
        if match:
            time_val = float(match.group(1))
            if 's' in match.group(0):
                time_val *= 1000  # Convert seconds to milliseconds
            response_times.append(time_val)

if response_times:
    print(f"Average response time: {statistics.mean(response_times[-100:]):.1f}ms")
    print(f"95th percentile: {statistics.quantiles(response_times[-100:], n=20)[18]:.1f}ms")
```

**Solutions:**

1. **Optimize Request Parameters**
   ```python
   # Reduce token limits for better performance
   config = LightRAGConfig(
       max_tokens=16384,  # Reduce from 32768
       max_async=8        # Reduce concurrent requests
   )
   ```

2. **Implement Caching**
   ```python
   # Use response caching for repeated queries
   from lightrag_integration.optimizations import ResponseCache
   
   cache = ResponseCache(cache_duration_hours=24)
   
   def cached_api_call(prompt, model):
       cached_response = cache.get_cached_response(prompt, model)
       if cached_response:
           return cached_response
       
       response = make_api_call(prompt, model)
       cache.cache_response(prompt, model, response, calculate_cost(response))
       return response
   ```

3. **Use Connection Pooling**
   ```python
   import requests
   from requests.adapters import HTTPAdapter
   from urllib3.util.retry import Retry
   
   # Configure HTTP session with connection pooling
   session = requests.Session()
   retry_strategy = Retry(
       total=3,
       backoff_factor=1,
       status_forcelist=[429, 500, 502, 503, 504]
   )
   adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retry_strategy)
   session.mount("https://", adapter)
   ```

### High CPU Usage

**Problem:** System showing high CPU utilization.

**Diagnosis:**
```bash
# Monitor CPU usage
top -p $(pgrep -f lightrag)
htop -p $(pgrep -f lightrag)

# Check for CPU-intensive operations
perf top -p $(pgrep -f lightrag)

# Profile Python application
python -m cProfile -o profile_output.prof your_script.py
```

**Solutions:**

1. **Optimize Database Operations**
   ```python
   # Use batch operations instead of individual inserts
   cost_records = []
   for operation in operations:
       cost_records.append(create_cost_record(operation))
   
   # Batch insert every 100 records
   if len(cost_records) >= 100:
       database.batch_insert_cost_records(cost_records)
       cost_records.clear()
   ```

2. **Reduce Logging Overhead**
   ```python
   # Use appropriate log levels
   config = LightRAGConfig(log_level="INFO")  # Not DEBUG in production
   
   # Implement structured logging for better performance
   import structlog
   logger = structlog.get_logger()
   ```

---

## Integration Problems

### LightRAG Integration Issues

**Problem:** Budget monitoring not properly integrated with LightRAG operations.

**Diagnosis:**
```python
# Test integration manually
from lightrag_integration import BudgetManagementSystem
from lightrag_integration.config import LightRAGConfig

config = LightRAGConfig.get_config()
system = BudgetManagementSystem(config)

# Test operation tracking
with system.track_operation("test_integration") as tracker:
    # Simulate LightRAG operation
    tracker.set_tokens(prompt=100, completion=50)
    tracker.set_cost(0.05)
    print("Integration test successful")

# Verify cost was recorded
status = system.get_budget_status()
print(f"Recorded cost: ${status['daily_budget']['total_cost']:.4f}")
```

**Solutions:**

1. **Proper Context Manager Usage**
   ```python
   # Always use context managers for operation tracking
   with budget_system.track_operation("llm_call", model="gpt-4o-mini") as tracker:
       try:
           result = lightrag_operation()
           tracker.set_tokens(prompt=tokens['input'], completion=tokens['output'])
           tracker.set_cost(calculate_cost(result))
           return result
       except Exception as e:
           tracker.set_error(type(e).__name__, str(e))
           raise
   ```

2. **Circuit Breaker Integration**
   ```python
   # Integrate with LightRAG's existing circuit breakers
   def protected_lightrag_operation():
       return circuit_breaker_manager.execute_with_protection(
           breaker_name="lightrag_operations",
           operation_callable=lightrag_operation,
           operation_type="knowledge_extraction",
           estimated_cost=0.05
       )
   ```

### External Service Integration

**Problem:** Integration with monitoring or alerting systems fails.

**Common Solutions:**

1. **Prometheus Integration**
   ```python
   # Ensure metrics are properly exposed
   from prometheus_client import start_http_server, Counter, Histogram
   
   # Start metrics server
   start_http_server(8000)
   
   # Define metrics
   api_calls_total = Counter('lightrag_api_calls_total', 'Total API calls', ['operation_type'])
   response_time_seconds = Histogram('lightrag_response_time_seconds', 'Response time')
   ```

2. **External Webhook Integration**
   ```python
   # Test webhook endpoints
   import requests
   
   def test_webhook(url, payload):
       try:
           response = requests.post(url, json=payload, timeout=10)
           response.raise_for_status()
           return True
       except Exception as e:
           print(f"Webhook test failed: {e}")
           return False
   ```

---

## Recovery Procedures

### Emergency Budget Override

When budget limits are exceeded but critical operations must continue:

```python
# Emergency budget override procedure
from lightrag_integration import BudgetManagementSystem

system = BudgetManagementSystem.get_instance()

# 1. Temporarily increase budget limits
system.budget_manager.update_budget_limits(
    daily_budget=system.budget_manager.daily_budget_limit * 1.5,
    monthly_budget=system.budget_manager.monthly_budget_limit * 1.2
)

# 2. Reset circuit breakers
if hasattr(system, 'circuit_breaker_manager'):
    system.circuit_breaker_manager.reset_all_breakers("Emergency override")

# 3. Log the override
system.audit_trail.record_event(
    event_type="emergency_budget_override",
    event_data={
        "reason": "Critical research operations required",
        "previous_daily_limit": original_daily_limit,
        "new_daily_limit": new_daily_limit,
        "authorized_by": "system_admin"
    }
)

# 4. Set up monitoring for the override period
system.alert_system.send_immediate_alert(
    "Emergency budget override activated - monitoring closely"
)
```

### Database Recovery from Backup

```bash
#!/bin/bash
# database_recovery.sh

echo "Starting database recovery procedure..."

# 1. Stop service
echo "Stopping service..."
sudo systemctl stop lightrag-monitor

# 2. Backup current database (in case of partial corruption)
echo "Backing up current database..."
cp /opt/lightrag/data/cost_tracking.db /opt/lightrag/backups/cost_tracking.db.$(date +%Y%m%d_%H%M%S)

# 3. Find latest backup
LATEST_BACKUP=$(ls -t /opt/lightrag/backups/cost_tracking.db.* | head -n1)
echo "Using backup: $LATEST_BACKUP"

# 4. Restore backup
echo "Restoring database..."
cp "$LATEST_BACKUP" /opt/lightrag/data/cost_tracking.db
chown lightrag:lightrag /opt/lightrag/data/cost_tracking.db

# 5. Verify database integrity
echo "Verifying database integrity..."
if sqlite3 /opt/lightrag/data/cost_tracking.db "PRAGMA integrity_check;" | grep -q "ok"; then
    echo "Database integrity verified"
else
    echo "Database integrity check failed - manual intervention required"
    exit 1
fi

# 6. Restart service
echo "Starting service..."
sudo systemctl start lightrag-monitor

# 7. Verify service is running
sleep 5
if systemctl is-active --quiet lightrag-monitor; then
    echo "Service restarted successfully"
else
    echo "Service failed to start - check logs"
    sudo journalctl -u lightrag-monitor -n 20
    exit 1
fi

echo "Database recovery completed successfully"
```

### Configuration Recovery

```bash
#!/bin/bash
# config_recovery.sh

echo "Starting configuration recovery..."

# 1. Backup current configuration
cp /opt/lightrag/config/.env /opt/lightrag/backups/.env.$(date +%Y%m%d_%H%M%S)

# 2. Restore from template
cat > /opt/lightrag/config/.env << 'EOF'
# Core Configuration
OPENAI_API_KEY=REPLACE_WITH_ACTUAL_KEY
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small

# Directories
LIGHTRAG_WORKING_DIR=/opt/lightrag/data
LIGHTRAG_LOG_DIR=/opt/lightrag/logs

# Budget Configuration
LIGHTRAG_DAILY_BUDGET_LIMIT=100.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=2000.0
LIGHTRAG_COST_ALERT_THRESHOLD=75.0

# System Configuration
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_ENABLE_BUDGET_ALERTS=true

# Database
LIGHTRAG_COST_DB_PATH=/opt/lightrag/data/cost_tracking.db
EOF

# 3. Set secure permissions
chmod 600 /opt/lightrag/config/.env
chown lightrag:lightrag /opt/lightrag/config/.env

echo "Configuration template created."
echo "Please update the configuration with actual values:"
echo "  - OPENAI_API_KEY"
echo "  - Email/Slack settings if needed"
echo "  - Budget limits appropriate for your use case"
```

---

## Prevention and Maintenance

### Automated Health Monitoring

```bash
# Create monitoring script
cat > /opt/lightrag/scripts/health_monitor.sh << 'EOF'
#!/bin/bash
# Automated health monitoring

LOG_FILE="/opt/lightrag/logs/health_monitor.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Function to log with timestamp
log() {
    echo "[$DATE] $1" >> $LOG_FILE
}

# Check service status
if ! systemctl is-active --quiet lightrag-monitor; then
    log "CRITICAL: Service is not running"
    systemctl start lightrag-monitor
    sleep 10
    if systemctl is-active --quiet lightrag-monitor; then
        log "INFO: Service restarted successfully"
    else
        log "CRITICAL: Failed to restart service"
        # Send emergency alert
        echo "LightRAG service failed to start" | mail -s "CRITICAL: LightRAG Service Down" admin@yourdomain.com
    fi
fi

# Check disk space
DISK_USAGE=$(df -h /opt/lightrag | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    log "WARNING: High disk usage: ${DISK_USAGE}%"
    # Clean up old logs
    find /opt/lightrag/logs -name "*.log.*" -mtime +7 -delete
fi

# Check database integrity
if ! sqlite3 /opt/lightrag/data/cost_tracking.db "PRAGMA integrity_check;" | grep -q "ok"; then
    log "CRITICAL: Database integrity check failed"
    # Trigger database recovery procedure
    /opt/lightrag/scripts/database_recovery.sh
fi

# Check recent errors
ERROR_COUNT=$(tail -1000 /opt/lightrag/logs/lightrag_integration.log 2>/dev/null | grep -c "ERROR" || echo 0)
if [ "$ERROR_COUNT" -gt 10 ]; then
    log "WARNING: High error rate: $ERROR_COUNT errors in last 1000 log lines"
fi

log "Health check completed"
EOF

chmod +x /opt/lightrag/scripts/health_monitor.sh

# Add to crontab for regular monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/lightrag/scripts/health_monitor.sh") | crontab -
```

### Preventive Maintenance Tasks

```bash
# Weekly maintenance script
cat > /opt/lightrag/scripts/weekly_maintenance.sh << 'EOF'
#!/bin/bash
# Weekly maintenance tasks

echo "Starting weekly maintenance..."

# 1. Database optimization
echo "Optimizing database..."
sqlite3 /opt/lightrag/data/cost_tracking.db << 'SQL'
PRAGMA analysis_limit=1000;
PRAGMA optimize;
ANALYZE;
SQL

# 2. Log rotation and cleanup
echo "Cleaning up logs..."
find /opt/lightrag/logs -name "*.log.*" -mtime +30 -delete
journalctl --vacuum-time=30d

# 3. Database backup
echo "Creating database backup..."
cp /opt/lightrag/data/cost_tracking.db /opt/lightrag/backups/cost_tracking.db.weekly.$(date +%Y%m%d)

# 4. Clean old backups
echo "Cleaning old backups..."
find /opt/lightrag/backups -name "cost_tracking.db.*" -mtime +30 -delete

# 5. System updates (security only)
echo "Installing security updates..."
apt list --upgradable | grep -E "(security|CVE)" | cut -d'/' -f1 | xargs apt-get install -y

# 6. Restart service for fresh start
echo "Restarting service..."
systemctl restart lightrag-monitor

# 7. Verify everything is working
sleep 10
if systemctl is-active --quiet lightrag-monitor; then
    echo "Weekly maintenance completed successfully"
else
    echo "WARNING: Service failed to start after maintenance"
fi
EOF

chmod +x /opt/lightrag/scripts/weekly_maintenance.sh

# Schedule weekly maintenance (Sunday 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * 0 /opt/lightrag/scripts/weekly_maintenance.sh") | crontab -
```

### Monitoring Best Practices

1. **Set up proper alerting thresholds**
2. **Regular backup verification**
3. **Performance baseline monitoring**
4. **Security update automation**
5. **Documentation maintenance**

### Emergency Contacts and Procedures

Create an emergency procedures document:

```markdown
# LightRAG Budget Monitor Emergency Procedures

## Emergency Contacts
- System Administrator: admin@yourdomain.com
- Research Lead: research@yourdomain.com
- IT Support: support@yourdomain.com

## Critical Issues Response

### Service Down
1. Check service status: `systemctl status lightrag-monitor`
2. Review logs: `journalctl -u lightrag-monitor -f`
3. Restart service: `systemctl restart lightrag-monitor`
4. If restart fails, escalate to System Administrator

### Database Corruption
1. Stop service immediately
2. Run database recovery script: `/opt/lightrag/scripts/database_recovery.sh`
3. If recovery fails, restore from backup
4. Document incident and notify Research Lead

### Budget Override Required
1. Contact Research Lead for authorization
2. Use emergency budget override procedure
3. Monitor usage closely during override period
4. Reset to normal limits as soon as possible

## Escalation Matrix
- Level 1: Automated monitoring and self-healing
- Level 2: System Administrator intervention
- Level 3: Vendor support engagement
- Level 4: Research operations impact assessment
```

---

This comprehensive troubleshooting guide provides systematic approaches to diagnosing and resolving issues with the API Cost Monitoring System. For additional support, refer to the other documentation guides in this suite.

---

*This troubleshooting guide is part of the Clinical Metabolomics Oracle API Cost Monitoring System documentation suite.*