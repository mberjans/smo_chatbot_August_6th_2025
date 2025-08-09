# Graceful Degradation System - Operational Guide

## Table of Contents
1. [Overview for Operators](#overview-for-operators)
2. [Daily Operations](#daily-operations)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Load Level Management](#load-level-management)
5. [Configuration Management](#configuration-management)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Emergency Procedures](#emergency-procedures)
9. [Maintenance Tasks](#maintenance-tasks)
10. [Runbooks](#runbooks)

## Overview for Operators

The Clinical Metabolomics Oracle Graceful Degradation System is designed to be self-managing, but operators play a crucial role in monitoring system health, tuning performance, and responding to exceptional conditions.

### Key Operator Responsibilities

- **Monitor system health** through dashboards and alerts
- **Respond to degradation events** and system alerts  
- **Tune configuration** based on observed performance patterns
- **Investigate performance issues** when they occur
- **Manage capacity planning** based on load trends
- **Execute emergency procedures** during critical situations

### System States Overview

The system operates in five distinct states, each with different operational characteristics:

| State | Description | Operator Actions Required |
|-------|-------------|---------------------------|
| **NORMAL** | Optimal performance, all features enabled | Monitor trends, no action needed |
| **ELEVATED** | Minor load increase, minor optimizations active | Monitor for patterns, consider capacity |
| **HIGH** | Significant load, performance degradation active | Investigate load sources, check capacity |
| **CRITICAL** | High load, aggressive degradation measures | Active monitoring, prepare for scaling |
| **EMERGENCY** | System protection mode, minimal functionality | Immediate investigation, emergency response |

## Daily Operations

### Morning Health Check Routine

**Time Required: 10-15 minutes**

1. **System Status Verification**
   ```bash
   # Check overall system health
   curl -X GET http://localhost:8000/health
   
   # Expected response:
   # {
   #   "status": "healthy",
   #   "uptime_seconds": 86400,
   #   "current_load_level": "NORMAL",
   #   "total_requests_processed": 150000,
   #   "component_status": {
   #     "load_monitoring": "active",
   #     "degradation_controller": "active", 
   #     "throttling_system": "active"
   #   }
   # }
   ```

2. **Load Level History Review**
   ```bash
   # Review load patterns from last 24 hours
   python -c "
   from graceful_degradation_integration import *
   orchestrator = get_running_orchestrator()
   history = orchestrator.get_metrics_history(hours=24)
   
   # Print load level distribution
   levels = [m['load_level'] for m in history]
   from collections import Counter
   print('Load Level Distribution (24h):')
   for level, count in Counter(levels).items():
       print(f'  {level}: {count} measurements ({count/len(levels)*100:.1f}%)')
   "
   ```

3. **Performance Metrics Review**
   ```bash
   # Check key performance indicators
   python -c "
   from graceful_degradation_integration import *
   orchestrator = get_running_orchestrator()
   status = orchestrator.get_system_status()
   
   print('System Performance Summary:')
   print(f\"Running Time: {status['uptime_seconds']/3600:.1f} hours\")
   print(f\"Total Requests: {status['total_requests_processed']:,}\")
   print(f\"Current Load: {status['current_load_level']}\")
   
   if 'throttling_system' in status:
       throttling = status['throttling_system']['throttling']
       print(f\"Success Rate: {throttling.get('success_rate', 0):.1f}%\")
       print(f\"Current Rate: {throttling.get('current_rate', 0):.1f} req/s\")
   "
   ```

4. **Alert Review**
   ```bash
   # Check for any active alerts or recent issues
   tail -100 /path/to/logs/graceful_degradation.log | grep -i "error\|warn\|critical"
   ```

### Evening Performance Review

**Time Required: 15-20 minutes**

1. **Traffic Pattern Analysis**
   ```python
   # Analyze daily traffic patterns
   import matplotlib.pyplot as plt
   from datetime import datetime, timedelta
   
   history = orchestrator.get_metrics_history(hours=24)
   
   # Plot load levels over time
   timestamps = [datetime.fromisoformat(m['timestamp']) for m in history]
   load_levels = [m['load_level'] for m in history]
   
   plt.figure(figsize=(12, 6))
   plt.scatter(timestamps, load_levels, alpha=0.6)
   plt.title('Load Levels Over Last 24 Hours')
   plt.xlabel('Time')
   plt.ylabel('Load Level')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig('/tmp/daily_load_pattern.png')
   print("Daily load pattern saved to /tmp/daily_load_pattern.png")
   ```

2. **Capacity Utilization Review**
   ```python
   # Review resource utilization trends
   history = orchestrator.get_metrics_history(hours=24)
   
   avg_cpu = sum(m['cpu_utilization'] for m in history) / len(history)
   avg_memory = sum(m['memory_pressure'] for m in history) / len(history)
   max_response_time = max(m.get('response_time_p95', 0) for m in history)
   
   print(f"24-Hour Resource Utilization Summary:")
   print(f"  Average CPU: {avg_cpu:.1f}%")
   print(f"  Average Memory: {avg_memory:.1f}%") 
   print(f"  Max Response Time (P95): {max_response_time:.0f}ms")
   
   # Alert if trending high
   if avg_cpu > 70 or avg_memory > 75:
       print("⚠️  HIGH UTILIZATION WARNING - Consider capacity planning")
   ```

## Monitoring and Alerting

### Key Metrics to Monitor

#### System Health Metrics
- **System Status**: Overall health (healthy/degraded/critical)
- **Uptime**: System availability and restart frequency
- **Component Status**: Individual component health status
- **Error Rate**: System-wide error percentage

#### Load Metrics  
- **Current Load Level**: NORMAL/ELEVATED/HIGH/CRITICAL/EMERGENCY
- **Load Transitions**: Frequency of load level changes
- **CPU Utilization**: System CPU usage percentage
- **Memory Pressure**: Available memory percentage
- **Response Times**: P95 and P99 response time metrics

#### Throttling Metrics
- **Request Rate**: Current requests per second
- **Success Rate**: Percentage of successfully processed requests
- **Queue Utilization**: Percentage of queue capacity used
- **Queue Depths**: Requests pending in each priority queue

### Dashboard Configuration

**Primary Dashboard Panels:**

1. **System Status Panel**
   ```json
   {
     "title": "Graceful Degradation System Status",
     "panels": [
       {
         "title": "Current Load Level",
         "type": "stat",
         "target": "graceful_degradation.load_level",
         "thresholds": {
           "NORMAL": "green",
           "ELEVATED": "yellow", 
           "HIGH": "orange",
           "CRITICAL": "red",
           "EMERGENCY": "red"
         }
       },
       {
         "title": "Success Rate",
         "type": "gauge",
         "target": "graceful_degradation.success_rate",
         "min": 0,
         "max": 100,
         "thresholds": [
           {"value": 95, "color": "green"},
           {"value": 90, "color": "yellow"},
           {"value": 0, "color": "red"}
         ]
       }
     ]
   }
   ```

2. **Load Metrics Panel**
   ```json
   {
     "title": "System Load Metrics",
     "panels": [
       {
         "title": "CPU & Memory Utilization",
         "type": "timeseries",
         "targets": [
           "graceful_degradation.cpu_utilization",
           "graceful_degradation.memory_pressure"
         ],
         "thresholds": [
           {"value": 80, "color": "orange"},
           {"value": 90, "color": "red"}
         ]
       },
       {
         "title": "Response Times",
         "type": "timeseries",
         "targets": [
           "graceful_degradation.response_time_p95",
           "graceful_degradation.response_time_p99"
         ],
         "unit": "ms"
       }
     ]
   }
   ```

### Alert Configuration

#### Critical Alerts (Immediate Response Required)

```yaml
alerts:
  - name: "System Emergency Mode"
    condition: "graceful_degradation.load_level == 'EMERGENCY'"
    duration: "1m"
    severity: "critical"
    notification: "page_oncall"
    message: "Graceful degradation system in EMERGENCY mode - immediate investigation required"
    
  - name: "System Health Critical"
    condition: "graceful_degradation.health_status == 'critical'"
    duration: "2m" 
    severity: "critical"
    notification: "page_oncall"
    message: "Graceful degradation system health is critical"

  - name: "Success Rate Below 80%"
    condition: "graceful_degradation.success_rate < 80"
    duration: "5m"
    severity: "critical" 
    notification: "page_oncall"
    message: "Request success rate critically low: {{ $value }}%"
```

#### Warning Alerts (Monitor Closely)

```yaml
  - name: "High Load Level Sustained"
    condition: "graceful_degradation.load_level == 'HIGH'"
    duration: "10m"
    severity: "warning"
    notification: "slack_channel"
    message: "System in HIGH load state for 10+ minutes"
    
  - name: "Success Rate Below 95%"
    condition: "graceful_degradation.success_rate < 95"
    duration: "15m"
    severity: "warning" 
    notification: "slack_channel"
    message: "Request success rate degraded: {{ $value }}%"

  - name: "Queue Utilization High"
    condition: "graceful_degradation.queue_utilization > 90"
    duration: "5m"
    severity: "warning"
    notification: "slack_channel" 
    message: "Request queue {{ $value }}% full - potential backlog building"
```

### Monitoring Commands

```bash
#!/bin/bash
# monitoring_commands.sh - Common monitoring commands for operators

# Get current system status
get_status() {
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    status = orchestrator.get_system_status()
    health = orchestrator.get_health_check()
    
    print(f'Status: {health[\"status\"]}')
    print(f'Load Level: {health[\"current_load_level\"]}')
    print(f'Uptime: {health[\"uptime_seconds\"]:.1f}s')
    print(f'Requests: {health[\"total_requests_processed\"]:,}')
    if health['issues']:
        print(f'Issues: {', '.join(health[\"issues\"])}')
    "
}

# Get detailed metrics
get_metrics() {
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    status = orchestrator.get_system_status()
    
    if 'load_monitoring' in status:
        metrics = status['load_monitoring']['current_metrics']
        print('Current System Metrics:')
        print(f'  CPU: {metrics.get(\"cpu_utilization\", 0):.1f}%')
        print(f'  Memory: {metrics.get(\"memory_pressure\", 0):.1f}%')
        print(f'  Response P95: {metrics.get(\"response_time_p95\", 0):.0f}ms')
        print(f'  Error Rate: {metrics.get(\"error_rate\", 0):.2f}%')
    "
}

# Check for issues
check_issues() {
    echo "Recent errors and warnings:"
    tail -50 /path/to/logs/graceful_degradation.log | grep -E "(ERROR|WARN|CRITICAL)" | tail -10
    
    echo "Current system issues:"
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    health = orchestrator.get_health_check()
    
    if health['issues']:
        for issue in health['issues']:
            print(f'  - {issue}')
    else:
        print('  No active issues')
    "
}
```

## Load Level Management

### Understanding Load Levels

Each load level represents a different operational state with specific characteristics:

#### NORMAL Operations
- **Characteristics**: Full functionality, optimal response times
- **Typical Conditions**: CPU < 50%, Memory < 60%, Response P95 < 1000ms
- **Operator Actions**: 
  - Monitor trends for capacity planning
  - No immediate action required
  - Good time for maintenance tasks

#### ELEVATED Load
- **Characteristics**: Minor optimizations active, slightly reduced logging
- **Typical Conditions**: CPU 50-65%, Memory 60-70%, Response P95 1000-2000ms
- **Operator Actions**:
  - Monitor load sources and patterns
  - Review upcoming scheduled tasks
  - Consider proactive scaling

#### HIGH Load  
- **Characteristics**: Significant degradation, timeouts reduced, some features disabled
- **Typical Conditions**: CPU 65-80%, Memory 70-75%, Response P95 2000-3000ms
- **Operator Actions**:
  - **Investigate load sources** - identify what's driving high load
  - **Review capacity** - determine if scaling is needed
  - **Monitor queues** - watch for backlog building
  - **Prepare for escalation** if load continues increasing

#### CRITICAL Load
- **Characteristics**: Aggressive degradation, core functionality only
- **Typical Conditions**: CPU 80-90%, Memory 75-85%, Response P95 3000-5000ms  
- **Operator Actions**:
  - **Active monitoring required** - stay engaged with system
  - **Scale resources** if possible
  - **Identify non-essential load** and consider rate limiting specific users/endpoints
  - **Prepare emergency procedures**

#### EMERGENCY Mode
- **Characteristics**: Minimal functionality, system protection active
- **Typical Conditions**: CPU > 90%, Memory > 85%, Response P95 > 5000ms
- **Operator Actions**:
  - **Immediate investigation** - critical system state
  - **Execute emergency scaling** 
  - **Consider load shedding** - reject low priority requests
  - **Coordinate with development team** if application issues suspected

### Load Level Response Procedures

#### When Load Level Increases

```bash
#!/bin/bash
# load_increase_response.sh

check_load_increase() {
    NEW_LEVEL=$1
    echo "Load level increased to: $NEW_LEVEL"
    
    case $NEW_LEVEL in
        "ELEVATED")
            echo "Monitoring increased load..."
            # Log the increase
            logger "Graceful degradation: Load level ELEVATED"
            ;;
            
        "HIGH") 
            echo "HIGH load detected - investigating..."
            # Get immediate diagnostics
            get_metrics
            check_top_consumers
            alert_team "HIGH load level reached"
            ;;
            
        "CRITICAL")
            echo "CRITICAL load - active response required"
            get_metrics
            check_top_consumers  
            check_available_capacity
            alert_team "CRITICAL load level - scaling may be needed"
            ;;
            
        "EMERGENCY")
            echo "EMERGENCY MODE - immediate action required"
            get_metrics
            execute_emergency_scaling
            alert_oncall "EMERGENCY load level - system protection active"
            ;;
    esac
}

check_top_consumers() {
    echo "Top resource consumers:"
    
    # Check top CPU processes
    echo "Top CPU processes:"
    ps aux --sort=-%cpu | head -10
    
    # Check top memory processes  
    echo "Top memory processes:"
    ps aux --sort=-%mem | head -10
    
    # Check network connections
    echo "Active connections:"
    netstat -an | grep ESTABLISHED | wc -l
}

check_available_capacity() {
    echo "Available capacity check:"
    
    # Check if auto-scaling is possible
    if command -v kubectl > /dev/null; then
        echo "Kubernetes pod status:"
        kubectl get pods -l app=clinical-metabolomics-oracle
        
        echo "Node resource usage:"
        kubectl top nodes
    fi
    
    # Check system resources
    echo "System resource availability:"
    free -h
    df -h
}
```

#### When Load Level Decreases

```bash
recovery_monitoring() {
    PREVIOUS_LEVEL=$1
    CURRENT_LEVEL=$2
    
    echo "Load level decreased: $PREVIOUS_LEVEL -> $CURRENT_LEVEL"
    
    # Monitor for stability
    echo "Monitoring recovery stability..."
    
    for i in {1..5}; do
        sleep 30
        CURRENT_STATUS=$(get_current_load_level)
        echo "Recovery check $i/5: $CURRENT_STATUS"
        
        if [ "$CURRENT_STATUS" != "$CURRENT_LEVEL" ]; then
            echo "Load level changed during recovery monitoring"
            break
        fi
    done
    
    echo "Recovery appears stable"
    logger "Graceful degradation: Load level recovered to $CURRENT_LEVEL"
}
```

## Configuration Management

### Configuration Files

The system uses hierarchical configuration with environment-specific overrides:

```
config/
├── base.yaml                    # Base configuration
├── production.yaml             # Production overrides  
├── staging.yaml               # Staging overrides
└── development.yaml           # Development overrides
```

### Base Configuration Structure

```yaml
# config/base.yaml
graceful_degradation:
  monitoring:
    interval: 5.0
    enable_trend_analysis: true
    hysteresis_enabled: true
    hysteresis_factor: 0.85
    
    thresholds:
      cpu:
        normal: 50.0
        elevated: 65.0
        high: 80.0
        critical: 90.0
        emergency: 95.0
        
      memory:
        normal: 60.0
        elevated: 70.0
        high: 75.0
        critical: 85.0
        emergency: 90.0
        
      response_time_p95:
        normal: 1000.0
        elevated: 2000.0
        high: 3000.0
        critical: 5000.0
        emergency: 8000.0
  
  throttling:
    base_rate_per_second: 10.0
    burst_capacity: 20
    max_queue_size: 1000
    max_concurrent_requests: 50
    
    priority_weights:
      critical: 1.0
      high: 0.8
      medium: 0.6
      low: 0.4
      background: 0.2
      
  degradation:
    timeout_multipliers:
      normal:
        lightrag: 1.0
        literature_search: 1.0
        openai_api: 1.0
        perplexity_api: 1.0
      elevated:
        lightrag: 0.9
        literature_search: 0.9
        openai_api: 0.95
        perplexity_api: 0.95
      high:
        lightrag: 0.75
        literature_search: 0.70
        openai_api: 0.85
        perplexity_api: 0.80
```

### Production Configuration Tuning

```yaml
# config/production.yaml
graceful_degradation:
  monitoring:
    interval: 3.0                    # More frequent monitoring in production
    
    thresholds:
      cpu:
        high: 75.0                   # More aggressive CPU threshold
        critical: 85.0
        emergency: 92.0
        
      memory:
        high: 70.0                   # More aggressive memory threshold
        critical: 80.0
        emergency: 87.0
        
  throttling:
    base_rate_per_second: 50.0       # Higher capacity for production
    burst_capacity: 100
    max_queue_size: 2000
    max_concurrent_requests: 100
    
    starvation_threshold: 300.0      # 5 minutes anti-starvation
    
  integration:
    auto_start_monitoring: true
    enable_production_integration: true
    metrics_retention_hours: 48      # Longer retention in production
```

### Configuration Update Procedures

#### Safe Configuration Updates

```bash
#!/bin/bash
# update_configuration.sh - Safe configuration update procedure

update_config() {
    CONFIG_FILE=$1
    
    echo "Updating configuration from $CONFIG_FILE"
    
    # 1. Validate configuration syntax
    echo "Validating configuration..."
    python -c "
    import yaml
    try:
        with open('$CONFIG_FILE', 'r') as f:
            config = yaml.safe_load(f)
        print('✅ Configuration syntax valid')
    except Exception as e:
        print(f'❌ Configuration syntax error: {e}')
        exit(1)
    " || exit 1
    
    # 2. Test configuration with dry-run
    echo "Testing configuration with dry-run..."
    python -c "
    from graceful_degradation_integration import GracefulDegradationConfig
    try:
        config = GracefulDegradationConfig.load_from_file('$CONFIG_FILE')
        print('✅ Configuration structure valid')
    except Exception as e:
        print(f'❌ Configuration structure error: {e}')
        exit(1)
    " || exit 1
    
    # 3. Backup current configuration
    echo "Backing up current configuration..."
    BACKUP_FILE="config/backup/config_$(date +%Y%m%d_%H%M%S).yaml"
    mkdir -p config/backup
    cp config/current.yaml "$BACKUP_FILE"
    echo "Backup saved to $BACKUP_FILE"
    
    # 4. Apply new configuration
    echo "Applying new configuration..."
    cp "$CONFIG_FILE" config/current.yaml
    
    # 5. Signal configuration reload
    echo "Reloading system configuration..."
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    orchestrator.configuration_manager.reload_configuration('config/current.yaml')
    print('✅ Configuration reloaded successfully')
    "
    
    echo "Configuration update completed successfully"
}

# Rollback configuration if needed
rollback_config() {
    BACKUP_FILE=$1
    
    echo "Rolling back configuration to $BACKUP_FILE"
    
    cp "$BACKUP_FILE" config/current.yaml
    
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()  
    orchestrator.configuration_manager.reload_configuration('config/current.yaml')
    print('✅ Configuration rollback completed')
    "
}
```

#### Runtime Configuration Adjustments

```python
# Quick runtime adjustments for common scenarios

def adjust_for_high_traffic():
    """Adjust configuration for expected high traffic period."""
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    
    # Increase rate limits temporarily
    new_config = {
        'throttling': {
            'base_rate_per_second': 100.0,  # Double normal rate
            'burst_capacity': 200,
            'max_concurrent_requests': 150
        }
    }
    
    orchestrator.configuration_manager.update_runtime_config(new_config)
    print("✅ Configuration adjusted for high traffic")

def adjust_for_maintenance_window():
    """Adjust configuration for maintenance window."""
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    
    # More conservative thresholds during maintenance
    new_config = {
        'monitoring': {
            'thresholds': {
                'cpu': {'high': 60.0, 'critical': 70.0},
                'memory': {'high': 60.0, 'critical': 70.0}
            }
        }
    }
    
    orchestrator.configuration_manager.update_runtime_config(new_config)
    print("✅ Configuration adjusted for maintenance window")
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: High Request Rejection Rate

**Symptoms:**
- Success rate below 90%
- Users reporting errors or timeouts
- High queue utilization

**Investigation Steps:**
```bash
# Check current throttling status
python -c "
from graceful_degradation_integration import get_running_orchestrator
orchestrator = get_running_orchestrator()
status = orchestrator.get_system_status()

throttling = status['throttling_system']['throttling']
print(f'Success Rate: {throttling[\"success_rate\"]:.1f}%')
print(f'Current Rate: {throttling[\"current_rate\"]:.1f} req/s')
print(f'Configured Rate: {throttling[\"configured_rate\"]:.1f} req/s')

queue = status['throttling_system']['queue']  
print(f'Queue Utilization: {queue[\"utilization\"]:.1f}%')
print(f'Queue Sizes: {queue[\"priority_breakdown\"]}')
"
```

**Solutions:**
1. **Increase Rate Limits** (if resources allow):
   ```python
   # Temporary rate limit increase
   orchestrator.configuration_manager.update_runtime_config({
       'throttling': {
           'base_rate_per_second': 75.0,  # Increase from current
           'burst_capacity': 150
       }
   })
   ```

2. **Scale Resources** (if auto-scaling available):
   ```bash
   # Scale up Kubernetes deployment
   kubectl scale deployment clinical-metabolomics-oracle --replicas=5
   ```

3. **Prioritize Critical Requests**:
   ```python
   # Adjust priority weights to favor high-priority requests
   orchestrator.configuration_manager.update_runtime_config({
       'throttling': {
           'priority_weights': {
               'critical': 1.0,
               'high': 0.9,      # Increased from 0.8
               'medium': 0.4,    # Decreased from 0.6
               'low': 0.2,       # Decreased from 0.4
               'background': 0.1 # Decreased from 0.2
           }
       }
   })
   ```

#### Issue: Stuck in High Load Level

**Symptoms:**
- Load level remains HIGH or CRITICAL despite low system utilization
- Hysteresis preventing load level decrease

**Investigation Steps:**
```python
# Check load level history and hysteresis
from graceful_degradation_integration import get_running_orchestrator
orchestrator = get_running_orchestrator()

# Check recent load levels
history = orchestrator.get_metrics_history(hours=1)
recent_levels = [m['load_level'] for m in history[-10:]]
print(f"Recent load levels: {recent_levels}")

# Check current metrics vs thresholds  
current_metrics = orchestrator.load_detector.get_system_metrics()
thresholds = orchestrator.load_detector.thresholds

print(f"Current CPU: {current_metrics.cpu_utilization:.1f}% (threshold: {thresholds.cpu_high:.1f}%)")
print(f"Current Memory: {current_metrics.memory_pressure:.1f}% (threshold: {thresholds.memory_high:.1f}%)")
```

**Solutions:**
1. **Manual Load Level Override** (temporary):
   ```python
   # Force load level to NORMAL if metrics support it
   orchestrator.degradation_controller.force_load_level(
       SystemLoadLevel.NORMAL, 
       "Manual override - metrics support normal operation"
   )
   ```

2. **Adjust Hysteresis Settings**:
   ```python
   # Reduce hysteresis factor for faster recovery
   orchestrator.configuration_manager.update_runtime_config({
       'monitoring': {
           'hysteresis_factor': 0.7  # Reduced from 0.85
       }
   })
   ```

#### Issue: Component Failures

**Symptoms:**
- Component status showing as inactive
- Error messages in logs about component failures

**Investigation Steps:**
```bash
# Check component status
python -c "
from graceful_degradation_integration import get_running_orchestrator
orchestrator = get_running_orchestrator()
health = orchestrator.get_health_check()

print('Component Status:')
for component, status in health['component_status'].items():
    print(f'  {component}: {status}')

if health['issues']:
    print('Active Issues:')
    for issue in health['issues']:
        print(f'  - {issue}')
"

# Check logs for component errors
tail -100 /path/to/logs/graceful_degradation.log | grep -i "component\|error"
```

**Solutions:**
1. **Restart Failed Components**:
   ```python
   # Restart specific component
   orchestrator = get_running_orchestrator()
   
   if not orchestrator._integration_status.load_monitoring_active:
       await orchestrator._initialize_load_monitoring()
       print("Load monitoring component restarted")
   
   if not orchestrator._integration_status.throttling_system_active:
       await orchestrator._initialize_throttling_system()
       print("Throttling system component restarted")
   ```

2. **Fallback Mode Operation**:
   ```python
   # Enable fallback mode for degraded operation
   orchestrator.enable_fallback_mode()
   print("System operating in fallback mode")
   ```

### Diagnostic Commands

```bash
#!/bin/bash
# diagnostic_commands.sh

# Full system diagnostic
full_diagnostic() {
    echo "=== Graceful Degradation System Diagnostic ==="
    echo "Timestamp: $(date)"
    echo
    
    echo "1. System Status:"
    get_status
    echo
    
    echo "2. Current Metrics:"
    get_metrics
    echo
    
    echo "3. Recent Issues:"
    check_issues
    echo
    
    echo "4. Component Details:"
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    status = orchestrator.get_system_status()
    
    print('Load Monitoring:', 'Active' if status['integration_status']['load_monitoring_active'] else 'Inactive')
    print('Degradation Controller:', 'Active' if status['integration_status']['degradation_controller_active'] else 'Inactive')
    print('Throttling System:', 'Active' if status['integration_status']['throttling_system_active'] else 'Inactive')
    print('Load Balancer Integration:', 'Yes' if status['integration_status']['integrated_load_balancer'] else 'No')
    print('RAG System Integration:', 'Yes' if status['integration_status']['integrated_rag_system'] else 'No')
    "
    echo
    
    echo "5. System Resources:"
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2 $3}'
    echo "Memory Usage:"  
    free -h | grep Mem
    echo "Disk Usage:"
    df -h | grep -v tmpfs | grep -v udev
    echo
}

# Performance diagnostic
performance_diagnostic() {
    echo "=== Performance Diagnostic ==="
    
    # Response time analysis
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    history = orchestrator.get_metrics_history(hours=1)
    
    if history:
        response_times = [m.get('response_time_p95', 0) for m in history]
        avg_response = sum(response_times) / len(response_times)
        max_response = max(response_times)
        
        print(f'Average Response Time (1h): {avg_response:.0f}ms')
        print(f'Maximum Response Time (1h): {max_response:.0f}ms')
        
        # Find slow periods
        slow_periods = [m for m in history if m.get('response_time_p95', 0) > 3000]
        if slow_periods:
            print(f'Slow periods detected: {len(slow_periods)} measurements > 3000ms')
    "
}

# Queue diagnostic  
queue_diagnostic() {
    echo "=== Queue Diagnostic ==="
    
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    orchestrator = get_running_orchestrator()
    status = orchestrator.get_system_status()
    
    if 'throttling_system' in status:
        queue = status['throttling_system']['queue']
        lifecycle = status['throttling_system']['lifecycle']
        
        print(f'Total Queue Size: {queue[\"total_size\"]}/{queue[\"max_size\"]} ({queue[\"utilization\"]:.1f}%)')
        print('Priority Breakdown:')
        for priority, size in queue['priority_breakdown'].items():
            print(f'  {priority}: {size} requests')
        
        print(f'Active Requests: {len(lifecycle.get(\"active_requests\", {}))}')
        print(f'Completed Requests: {lifecycle.get(\"completed_count\", 0)}')
        print(f'Failed Requests: {lifecycle.get(\"failed_count\", 0)}')
    "
}
```

## Performance Tuning

### Capacity Planning

#### Traffic Pattern Analysis

```python
def analyze_traffic_patterns():
    """Analyze traffic patterns for capacity planning."""
    from graceful_degradation_integration import get_running_orchestrator
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    orchestrator = get_running_orchestrator()
    
    # Get 7 days of data
    history = orchestrator.get_metrics_history(hours=168)  # 7 days
    
    # Analyze by hour of day
    hourly_patterns = {}
    for metric in history:
        hour = datetime.fromisoformat(metric['timestamp']).hour
        if hour not in hourly_patterns:
            hourly_patterns[hour] = []
        hourly_patterns[hour].append(metric['cpu_utilization'])
    
    # Calculate average utilization by hour
    hourly_averages = {}
    for hour, utilizations in hourly_patterns.items():
        hourly_averages[hour] = sum(utilizations) / len(utilizations)
    
    # Identify peak hours
    peak_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("Traffic Pattern Analysis:")
    print(f"Peak hours (highest CPU utilization):")
    for hour, avg_cpu in peak_hours:
        print(f"  {hour:02d}:00 - Average CPU: {avg_cpu:.1f}%")
    
    # Recommend capacity adjustments
    max_cpu = max(hourly_averages.values())
    if max_cpu > 70:
        print(f"\n⚠️  Recommendation: Consider scaling up during peak hours")
        print(f"   Peak utilization: {max_cpu:.1f}%")
    
    return hourly_averages

def capacity_recommendation():
    """Generate capacity recommendations based on usage patterns."""
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    
    # Analyze recent performance
    history = orchestrator.get_metrics_history(hours=72)  # 3 days
    
    if not history:
        print("Insufficient data for capacity analysis")
        return
    
    # Calculate key metrics
    cpu_values = [m['cpu_utilization'] for m in history]
    memory_values = [m['memory_pressure'] for m in history]
    response_times = [m.get('response_time_p95', 0) for m in history]
    
    avg_cpu = sum(cpu_values) / len(cpu_values)
    max_cpu = max(cpu_values)
    p95_cpu = sorted(cpu_values)[int(len(cpu_values) * 0.95)]
    
    avg_memory = sum(memory_values) / len(memory_values)
    max_memory = max(memory_values)
    
    avg_response = sum(response_times) / len(response_times)
    p95_response = sorted(response_times)[int(len(response_times) * 0.95)]
    
    print("Capacity Analysis (72 hours):")
    print(f"CPU - Average: {avg_cpu:.1f}%, P95: {p95_cpu:.1f}%, Max: {max_cpu:.1f}%")
    print(f"Memory - Average: {avg_memory:.1f}%, Max: {max_memory:.1f}%")
    print(f"Response Time - Average: {avg_response:.0f}ms, P95: {p95_response:.0f}ms")
    
    # Generate recommendations
    recommendations = []
    
    if p95_cpu > 80:
        recommendations.append("Scale up CPU resources - P95 utilization > 80%")
    elif max_cpu > 90:
        recommendations.append("Scale up CPU resources - Max utilization > 90%")
    
    if max_memory > 85:
        recommendations.append("Scale up memory resources - Max utilization > 85%")
    
    if p95_response > 3000:
        recommendations.append("Investigate performance issues - P95 response time > 3s")
    
    if avg_cpu < 30 and max_cpu < 60:
        recommendations.append("Consider scaling down resources - Low utilization")
    
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("\n✅ Current capacity appears appropriate")
```

#### Threshold Optimization

```python
def optimize_thresholds():
    """Optimize load thresholds based on historical performance."""
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    history = orchestrator.get_metrics_history(hours=168)  # 7 days
    
    if len(history) < 100:
        print("Insufficient data for threshold optimization")
        return
    
    # Analyze actual system behavior at different utilization levels
    cpu_values = [m['cpu_utilization'] for m in history]
    response_times = [m.get('response_time_p95', 0) for m in history]
    
    # Find CPU level where response times start degrading
    degradation_points = []
    for i in range(len(history)):
        if response_times[i] > 2000:  # Response time degradation threshold
            degradation_points.append(cpu_values[i])
    
    if degradation_points:
        # Calculate percentiles of degradation points
        degradation_points.sort()
        p25_degradation = degradation_points[int(len(degradation_points) * 0.25)]
        p50_degradation = degradation_points[int(len(degradation_points) * 0.50)]
        
        print("Threshold Optimization Analysis:")
        print(f"Response degradation typically starts at:")
        print(f"  25th percentile: {p25_degradation:.1f}% CPU")
        print(f"  50th percentile: {p50_degradation:.1f}% CPU")
        
        # Recommend new thresholds
        current_thresholds = orchestrator.load_detector.thresholds
        
        print(f"\nCurrent HIGH threshold: {current_thresholds.cpu_high:.1f}%")
        print(f"Recommended HIGH threshold: {max(60, p25_degradation - 5):.1f}%")
        
        recommended_high = max(60, p25_degradation - 5)
        if abs(recommended_high - current_thresholds.cpu_high) > 5:
            print("⚠️  Consider adjusting HIGH load threshold")
```

### Rate Limiting Optimization

```python
def optimize_rate_limits():
    """Optimize rate limiting based on system capacity and performance."""
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    status = orchestrator.get_system_status()
    
    if 'throttling_system' not in status:
        print("Throttling system not available for analysis")
        return
    
    throttling = status['throttling_system']['throttling']
    current_rate = throttling['current_rate']
    success_rate = throttling.get('success_rate', 0)
    
    # Get recent performance metrics
    history = orchestrator.get_metrics_history(hours=24)
    
    if history:
        # Analyze system performance at current rate
        recent_cpu = [m['cpu_utilization'] for m in history[-60:]]  # Last hour
        recent_response = [m.get('response_time_p95', 0) for m in history[-60:]]
        
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        avg_response = sum(recent_response) / len(recent_response)
        
        print("Rate Limiting Analysis:")
        print(f"Current rate: {current_rate:.1f} req/s")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average CPU (1h): {avg_cpu:.1f}%")
        print(f"Average response time (1h): {avg_response:.0f}ms")
        
        # Make recommendations
        recommendations = []
        
        if success_rate < 95 and avg_cpu < 70:
            # Low success rate but system not stressed - increase rate limit
            new_rate = current_rate * 1.2
            recommendations.append(f"Increase rate limit to {new_rate:.1f} req/s")
        
        elif avg_cpu > 80 or avg_response > 3000:
            # System stressed - decrease rate limit
            new_rate = current_rate * 0.9
            recommendations.append(f"Decrease rate limit to {new_rate:.1f} req/s")
        
        elif success_rate > 98 and avg_cpu < 50 and avg_response < 1500:
            # System underutilized - can increase rate
            new_rate = current_rate * 1.1
            recommendations.append(f"Consider increasing rate limit to {new_rate:.1f} req/s")
        
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        else:
            print("\n✅ Current rate limiting appears optimal")
```

## Emergency Procedures

### Emergency Response Runbook

#### System in EMERGENCY Mode

**Immediate Actions (0-5 minutes):**

1. **Acknowledge the Alert**
   ```bash
   # Confirm system is in emergency mode
   python -c "
   from graceful_degradation_integration import get_running_orchestrator
   orchestrator = get_running_orchestrator()
   health = orchestrator.get_health_check()
   print(f'System Status: {health[\"status\"]}')
   print(f'Load Level: {health[\"current_load_level\"]}')
   "
   ```

2. **Get Immediate Diagnostics**
   ```bash
   # Quick system overview
   echo "System Resources:"
   top -bn1 | head -20
   
   echo "Memory Usage:"
   free -h
   
   echo "Disk Usage:"
   df -h
   
   # Check for obvious issues
   echo "Recent Errors:"
   tail -50 /var/log/syslog | grep -i "error\|critical\|fail"
   ```

3. **Emergency Scaling (if auto-scaling available)**
   ```bash
   # Scale up immediately
   if command -v kubectl > /dev/null; then
       echo "Scaling up Kubernetes deployment..."
       kubectl scale deployment clinical-metabolomics-oracle --replicas=8
       kubectl get pods -l app=clinical-metabolomics-oracle
   fi
   
   # If using AWS Auto Scaling
   if command -v aws > /dev/null; then
       aws autoscaling set-desired-capacity \
           --auto-scaling-group-name clinical-metabolomics-oracle-asg \
           --desired-capacity 6
   fi
   ```

**Short-term Actions (5-15 minutes):**

4. **Load Shedding**
   ```python
   # Implement aggressive load shedding
   from graceful_degradation_integration import get_running_orchestrator
   
   orchestrator = get_running_orchestrator()
   
   # Drastically reduce rate limits
   emergency_config = {
       'throttling': {
           'base_rate_per_second': 5.0,  # Reduce to 10% of normal
           'max_concurrent_requests': 10,
           'emergency_mode': True
       }
   }
   
   orchestrator.configuration_manager.update_runtime_config(emergency_config)
   print("Emergency load shedding activated")
   ```

5. **Investigate Root Cause**
   ```bash
   # Check for resource exhaustion
   echo "Top CPU consumers:"
   ps aux --sort=-%cpu | head -15
   
   echo "Top memory consumers:"
   ps aux --sort=-%mem | head -15
   
   # Check for unusual network activity
   echo "Network connections:"
   netstat -tuln | wc -l
   
   # Check disk I/O
   iostat -x 1 3
   ```

6. **Communication**
   ```bash
   # Alert development team
   alert_team "EMERGENCY: Graceful degradation system in emergency mode. Immediate scaling and load shedding activated."
   
   # Update status page (if applicable)
   update_status_page "Investigating performance issues. Some delays may occur."
   ```

**Medium-term Actions (15-60 minutes):**

7. **Monitor Recovery**
   ```bash
   # Monitor system recovery every 5 minutes
   for i in {1..12}; do
       echo "Recovery check $i/12 - $(date)"
       
       python -c "
       from graceful_degradation_integration import get_running_orchestrator
       orchestrator = get_running_orchestrator()
       health = orchestrator.get_health_check()
       status = orchestrator.get_system_status()
       
       print(f'Load Level: {health[\"current_load_level\"]}')
       print(f'Success Rate: {status.get(\"throttling_system\", {}).get(\"throttling\", {}).get(\"success_rate\", 0):.1f}%')
       
       if 'load_monitoring' in status:
           metrics = status['load_monitoring']['current_metrics']
           print(f'CPU: {metrics.get(\"cpu_utilization\", 0):.1f}%')
           print(f'Memory: {metrics.get(\"memory_pressure\", 0):.1f}%')
       "
       
       sleep 300  # 5 minutes
   done
   ```

8. **Gradual Recovery**
   ```python
   # Gradually increase capacity as system recovers
   def gradual_recovery():
       from graceful_degradation_integration import get_running_orchestrator
       
       orchestrator = get_running_orchestrator()
       health = orchestrator.get_health_check()
       
       if health['current_load_level'] in ['NORMAL', 'ELEVATED']:
           # System recovering - increase rate limits gradually
           current_status = orchestrator.get_system_status()
           current_rate = current_status['throttling_system']['throttling']['current_rate']
           
           new_rate = min(50.0, current_rate * 1.5)  # Increase by 50%, max 50 req/s
           
           recovery_config = {
               'throttling': {
                   'base_rate_per_second': new_rate,
                   'max_concurrent_requests': min(100, int(new_rate * 2))
               }
           }
           
           orchestrator.configuration_manager.update_runtime_config(recovery_config)
           print(f"Recovery: Increased rate limit to {new_rate} req/s")
       
       return health['current_load_level']
   
   gradual_recovery()
   ```

#### Load Balancer Failure

**If production load balancer integration fails:**

1. **Switch to Fallback Mode**
   ```python
   from graceful_degradation_integration import get_running_orchestrator
   
   orchestrator = get_running_orchestrator()
   orchestrator.enable_fallback_mode('load_balancer_failure')
   
   print("Switched to fallback mode - direct request handling")
   ```

2. **Manual Load Distribution**
   ```bash
   # If using HAProxy, update backend weights
   echo "disable server clinical-rag/server1" | socat stdio /var/run/haproxy/admin.sock
   echo "set weight clinical-rag/server2 50" | socat stdio /var/run/haproxy/admin.sock
   ```

### Recovery Procedures

#### Post-Emergency Recovery

```bash
#!/bin/bash
# post_emergency_recovery.sh

post_emergency_recovery() {
    echo "Starting post-emergency recovery procedure..."
    
    # 1. Verify system stability
    echo "Verifying system stability..."
    
    for i in {1..5}; do
        LOAD_LEVEL=$(python -c "
        from graceful_degradation_integration import get_running_orchestrator
        orchestrator = get_running_orchestrator()
        health = orchestrator.get_health_check()
        print(health['current_load_level'])
        ")
        
        echo "Stability check $i/5: Load level $LOAD_LEVEL"
        
        if [[ "$LOAD_LEVEL" != "NORMAL" && "$LOAD_LEVEL" != "ELEVATED" ]]; then
            echo "System not yet stable, waiting..."
            sleep 60
        else
            echo "System appears stable"
            break
        fi
    done
    
    # 2. Gradually restore normal configuration
    echo "Restoring normal configuration..."
    
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    import time
    
    orchestrator = get_running_orchestrator()
    
    # Restore normal rate limits gradually
    target_rate = 50.0  # Normal production rate
    current_rate = 5.0  # Emergency rate
    
    steps = 5
    for step in range(1, steps + 1):
        rate = current_rate + (target_rate - current_rate) * (step / steps)
        
        config = {
            'throttling': {
                'base_rate_per_second': rate,
                'max_concurrent_requests': int(rate * 2)
            }
        }
        
        orchestrator.configuration_manager.update_runtime_config(config)
        print(f'Recovery step {step}/{steps}: Rate limit increased to {rate:.1f} req/s')
        
        time.sleep(30)  # Wait 30 seconds between steps
    
    print('Normal configuration restored')
    "
    
    # 3. Verify full functionality
    echo "Verifying full functionality..."
    
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    health = orchestrator.get_health_check()
    
    print('Final System Status:')
    print(f'  Health: {health[\"status\"]}')
    print(f'  Load Level: {health[\"current_load_level\"]}')
    print(f'  Total Requests: {health[\"total_requests_processed\"]:,}')
    
    for component, status in health['component_status'].items():
        print(f'  {component}: {status}')
    "
    
    # 4. Generate incident report
    echo "Generating incident report..."
    
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    from datetime import datetime, timedelta
    
    orchestrator = get_running_orchestrator()
    
    # Get incident timeline
    incident_start = datetime.now() - timedelta(hours=2)  # Assume 2-hour incident
    history = orchestrator.get_metrics_history(hours=3)
    
    incident_metrics = [m for m in history if datetime.fromisoformat(m['timestamp']) >= incident_start]
    
    print('Incident Summary:')
    print(f'Duration: {len(incident_metrics)} measurement periods')
    
    if incident_metrics:
        max_cpu = max(m['cpu_utilization'] for m in incident_metrics)
        max_memory = max(m['memory_pressure'] for m in incident_metrics)
        max_response = max(m.get('response_time_p95', 0) for m in incident_metrics)
        
        print(f'Peak CPU: {max_cpu:.1f}%')
        print(f'Peak Memory: {max_memory:.1f}%')
        print(f'Peak Response Time: {max_response:.0f}ms')
    
    print('Post-emergency recovery completed successfully')
    "
    
    echo "Post-emergency recovery procedure completed"
}
```

## Maintenance Tasks

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

weekly_maintenance() {
    echo "Starting weekly graceful degradation maintenance..."
    
    # 1. Log rotation and cleanup
    echo "Rotating logs..."
    find /path/to/logs -name "graceful_degradation*.log" -mtime +7 -delete
    find /path/to/logs -name "*.log" -size +100M -exec truncate -s 50M {} \;
    
    # 2. Metrics data cleanup
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    
    # Clean up old metrics (keep 7 days)
    cleaned_count = orchestrator.cleanup_old_metrics(days=7)
    print(f'Cleaned {cleaned_count} old metric entries')
    "
    
    # 3. Configuration backup
    echo "Backing up configuration..."
    BACKUP_DIR="/path/to/backups/graceful_degradation/$(date +%Y%m%d)"
    mkdir -p "$BACKUP_DIR"
    
    cp config/current.yaml "$BACKUP_DIR/config.yaml"
    cp /path/to/logs/graceful_degradation.log "$BACKUP_DIR/latest.log"
    
    # Keep only 4 weeks of backups
    find /path/to/backups/graceful_degradation -maxdepth 1 -type d -mtime +28 -exec rm -rf {} \;
    
    # 4. Health check validation
    echo "Running comprehensive health check..."
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    health = orchestrator.get_health_check()
    
    if health['status'] != 'healthy':
        print(f'⚠️  System health issue: {health[\"status\"]}')
        for issue in health['issues']:
            print(f'  - {issue}')
    else:
        print('✅ System health check passed')
    "
    
    # 5. Performance analysis
    echo "Running weekly performance analysis..."
    python -c "
    import sys
    sys.path.append('/path/to/operational/scripts')
    from performance_analysis import analyze_traffic_patterns, capacity_recommendation
    
    print('=== Weekly Performance Analysis ===')
    analyze_traffic_patterns()
    print()
    capacity_recommendation()
    "
    
    echo "Weekly maintenance completed"
}
```

### Monthly Maintenance

```bash
#!/bin/bash  
# monthly_maintenance.sh

monthly_maintenance() {
    echo "Starting monthly graceful degradation maintenance..."
    
    # 1. Threshold optimization analysis
    echo "Analyzing threshold optimization opportunities..."
    python -c "
    import sys
    sys.path.append('/path/to/operational/scripts')  
    from performance_analysis import optimize_thresholds, optimize_rate_limits
    
    print('=== Monthly Threshold Analysis ===')
    optimize_thresholds()
    print()
    optimize_rate_limits()
    "
    
    # 2. Historical performance report
    echo "Generating monthly performance report..."
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    from datetime import datetime, timedelta
    import json
    
    orchestrator = get_running_orchestrator()
    history = orchestrator.get_metrics_history(hours=720)  # 30 days
    
    if history:
        # Calculate monthly statistics
        load_levels = [m['load_level'] for m in history]
        cpu_values = [m['cpu_utilization'] for m in history]
        response_times = [m.get('response_time_p95', 0) for m in history]
        
        from collections import Counter
        level_distribution = Counter(load_levels)
        
        report = {
            'period': '30 days',
            'total_measurements': len(history),
            'load_level_distribution': dict(level_distribution),
            'cpu_stats': {
                'average': sum(cpu_values) / len(cpu_values),
                'maximum': max(cpu_values),
                'p95': sorted(cpu_values)[int(len(cpu_values) * 0.95)]
            },
            'response_time_stats': {
                'average': sum(response_times) / len(response_times),
                'maximum': max(response_times),
                'p95': sorted(response_times)[int(len(response_times) * 0.95)]
            }
        }
        
        # Save report
        with open(f'/path/to/reports/monthly_report_{datetime.now().strftime(\"%Y%m\")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print('Monthly Performance Report:')
        print(f'  Period: {report[\"period\"]}')
        print(f'  Measurements: {report[\"total_measurements\"]:,}')
        print('  Load Distribution:')
        for level, count in report['load_level_distribution'].items():
            pct = count / report['total_measurements'] * 100
            print(f'    {level}: {count:,} ({pct:.1f}%)')
        print(f'  Average CPU: {report[\"cpu_stats\"][\"average\"]:.1f}%')
        print(f'  Average Response Time: {report[\"response_time_stats\"][\"average\"]:.0f}ms')
    "
    
    # 3. Configuration audit
    echo "Auditing configuration..."
    python -c "
    from graceful_degradation_integration import get_running_orchestrator
    
    orchestrator = get_running_orchestrator()
    config = orchestrator.config
    
    print('Configuration Audit:')
    print(f'  Monitoring Interval: {config.monitoring_interval}s')
    print(f'  Base Rate Limit: {config.base_rate_per_second} req/s')
    print(f'  Max Queue Size: {config.max_queue_size}')
    print(f'  Max Concurrent Requests: {config.max_concurrent_requests}')
    
    # Check for recommended adjustments based on usage
    status = orchestrator.get_system_status()
    if 'throttling_system' in status:
        success_rate = status['throttling_system']['throttling'].get('success_rate', 100)
        if success_rate < 95:
            print('  ⚠️  Consider increasing rate limits - low success rate')
    
    print('Configuration audit completed')
    "
    
    echo "Monthly maintenance completed"
}
```

This operational guide provides comprehensive procedures and tools for effectively managing the graceful degradation system in production environments. Operators should familiarize themselves with these procedures and customize them based on their specific operational requirements and infrastructure setup.