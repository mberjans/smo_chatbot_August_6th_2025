# Unified System Health Dashboard API Documentation

**Version:** 1.0.0  
**Author:** Claude Code (Anthropic)  
**Created:** 2025-08-09  
**Task:** CMO-LIGHTRAG-014-T07 - System Health Monitoring Dashboard

## Overview

The Unified System Health Dashboard provides a comprehensive REST API and WebSocket interface for monitoring the Clinical Metabolomics Oracle system health. This documentation covers all available endpoints, data models, and integration patterns.

## Base Configuration

- **Default Port:** 8092
- **API Base Path:** `/api/v1`
- **WebSocket Endpoint:** `/ws/health`
- **Framework:** FastAPI (with Flask fallback)
- **Authentication:** Optional API key

## Quick Start

### Starting the Dashboard

```python
from lightrag_integration.unified_system_health_dashboard import create_and_start_dashboard
from lightrag_integration.dashboard_integration_helper import get_development_config

# Quick start for development
config = get_development_config()
dashboard = await create_and_start_dashboard(config=config.to_dashboard_config())
```

### Using the Integration Helper

```python
from lightrag_integration.dashboard_integration_helper import quick_start_dashboard

# One-line deployment
dashboard, report = await quick_start_dashboard(
    deployment_type="development",
    port=8092,
    enable_all_features=True
)
```

## REST API Endpoints

### Health and Status Endpoints

#### GET `/api/v1/health`
Get current system health status.

**Response:**
```json
{
  "status": "success",
  "data": {
    "timestamp": "2025-08-09T12:00:00Z",
    "snapshot_id": "abc12345",
    "system_uptime_seconds": 3600.0,
    "overall_health": "healthy",
    "health_score": 0.85,
    "load_level": "NORMAL",
    "load_score": 0.3,
    "cpu_utilization": 45.2,
    "memory_pressure": 62.1,
    "response_time_p95": 850.0,
    "error_rate": 0.1,
    "request_queue_depth": 5,
    "active_connections": 25,
    "degradation_active": false,
    "degradation_level": "NORMAL",
    "emergency_mode": false,
    "active_degradations": [],
    "disabled_features": [],
    "integrated_services": {
      "load_monitoring": true,
      "degradation_controller": true,
      "throttling_system": true,
      "load_balancer": true,
      "rag_system": true,
      "monitoring_system": true
    },
    "circuit_breakers": {
      "lightrag": "closed",
      "openai": "closed"
    },
    "throughput_rps": 15.5,
    "success_rate": 99.8,
    "total_requests_processed": 12450,
    "connection_pool_usage": 35.0,
    "thread_pool_usage": 28.0,
    "memory_usage_mb": 512.0,
    "active_alerts_count": 0,
    "critical_alerts_count": 0,
    "active_issues": [],
    "load_trend": "stable",
    "performance_trend": "improving",
    "predicted_load_change": "none"
  }
}
```

**Status Codes:**
- `200`: Success - health data returned
- `503`: Service Unavailable - no health data available

#### GET `/api/v1/health/history`
Get historical health data.

**Query Parameters:**
- `hours` (optional, int): Number of hours of history to retrieve (max 168, default 24)

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "timestamp": "2025-08-09T11:00:00Z",
      "overall_health": "healthy",
      "load_level": "NORMAL",
      "cpu_utilization": 42.1,
      "memory_pressure": 58.3,
      "response_time_p95": 920.0,
      "error_rate": 0.2
    }
  ]
}
```

#### GET `/api/v1/system/status`
Get comprehensive system status and integration information.

**Response:**
```json
{
  "status": "success",
  "data": {
    "dashboard_uptime_seconds": 7200.0,
    "framework": "fastapi",
    "config": {
      "websockets_enabled": true,
      "historical_data_enabled": true,
      "alerts_enabled": true,
      "retention_hours": 72
    },
    "current_health": { /* Current health snapshot */ },
    "connected_websocket_clients": 3,
    "total_snapshots": 1440,
    "active_alerts": 0
  }
}
```

### Alert Management Endpoints

#### GET `/api/v1/alerts`
Get active alerts.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "load_change_abc12345",
      "timestamp": "2025-08-09T12:00:00Z",
      "severity": "warning",
      "source": "load_detector",
      "title": "Load Level Changed",
      "message": "System load level changed from NORMAL to ELEVATED",
      "category": "performance",
      "resolved": false,
      "resolved_at": null,
      "resolved_by": null,
      "related_metrics": {
        "previous_level": "NORMAL",
        "current_level": "ELEVATED",
        "load_score": 0.6
      },
      "affected_services": [],
      "recommended_actions": []
    }
  ]
}
```

#### GET `/api/v1/alerts/history`
Get alert history.

**Query Parameters:**
- `hours` (optional, int): Number of hours of history to retrieve (max 168, default 24)

**Response:** Same format as active alerts but includes resolved alerts.

#### POST `/api/v1/alerts/{alert_id}/resolve`
Resolve an active alert.

**Response:**
```json
{
  "status": "success",
  "message": "Alert load_change_abc12345 resolved"
}
```

**Status Codes:**
- `200`: Success - alert resolved
- `404`: Not Found - alert not found or already resolved

## WebSocket Interface

### Connection

Connect to the WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8092/ws/health');
```

### Message Types

#### Initial Data Message
Sent immediately after WebSocket connection is established.

```json
{
  "type": "initial_data",
  "timestamp": "2025-08-09T12:00:00Z",
  "data": { /* Complete health snapshot */ }
}
```

#### Health Update Message
Sent whenever system health metrics are updated (default: every 2 seconds).

```json
{
  "type": "health_update",
  "timestamp": "2025-08-09T12:00:05Z",
  "data": { /* Complete health snapshot */ }
}
```

#### Alert Message
Sent when a new alert is generated.

```json
{
  "type": "alert",
  "timestamp": "2025-08-09T12:00:10Z",
  "data": { /* Alert object */ }
}
```

### JavaScript Client Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard Client</title>
</head>
<body>
    <div id="health-status">Connecting...</div>
    <div id="alerts"></div>

    <script>
        const ws = new WebSocket('ws://localhost:8092/ws/health');
        
        ws.onopen = function() {
            document.getElementById('health-status').textContent = 'Connected';
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'health_update') {
                const health = message.data;
                document.getElementById('health-status').innerHTML = `
                    Health: ${health.overall_health}<br>
                    Load: ${health.load_level}<br>
                    CPU: ${health.cpu_utilization.toFixed(1)}%<br>
                    Response: ${health.response_time_p95.toFixed(0)}ms
                `;
            }
            
            if (message.type === 'alert') {
                const alert = message.data;
                const alertDiv = document.createElement('div');
                alertDiv.innerHTML = `
                    <strong>${alert.severity.toUpperCase()}</strong>: ${alert.title}<br>
                    ${alert.message}
                `;
                document.getElementById('alerts').appendChild(alertDiv);
            }
        };
        
        ws.onclose = function() {
            document.getElementById('health-status').textContent = 'Disconnected';
        };
    </script>
</body>
</html>
```

## Data Models

### SystemHealthSnapshot

Main data structure representing system health at a point in time.

```typescript
interface SystemHealthSnapshot {
  timestamp: string;           // ISO datetime
  snapshot_id: string;         // Unique identifier
  system_uptime_seconds: number;
  
  // Overall health
  overall_health: "healthy" | "warning" | "degraded" | "critical" | "emergency";
  health_score: number;        // 0.0 to 1.0
  
  // Load metrics
  load_level: "NORMAL" | "ELEVATED" | "HIGH" | "CRITICAL" | "EMERGENCY";
  load_score: number;          // 0.0 to 1.0
  cpu_utilization: number;     // Percentage
  memory_pressure: number;     // Percentage
  response_time_p95: number;   // Milliseconds
  error_rate: number;          // Percentage
  request_queue_depth: number;
  active_connections: number;
  
  // Degradation status
  degradation_active: boolean;
  degradation_level: string;
  emergency_mode: boolean;
  active_degradations: string[];
  disabled_features: string[];
  
  // Service integration
  integrated_services: Record<string, boolean>;
  circuit_breakers: Record<string, string>;
  
  // Performance metrics
  throughput_rps: number;
  success_rate: number;
  total_requests_processed: number;
  
  // Resource usage
  connection_pool_usage: number;  // Percentage
  thread_pool_usage: number;      // Percentage
  memory_usage_mb: number;
  
  // Alerts
  active_alerts_count: number;
  critical_alerts_count: number;
  active_issues: string[];
  
  // Trends (for predictive analytics)
  load_trend: "improving" | "stable" | "degrading";
  performance_trend: "improving" | "stable" | "degrading";
  predicted_load_change: "increase" | "decrease" | "none";
}
```

### AlertEvent

Structure representing an alert in the system.

```typescript
interface AlertEvent {
  id: string;
  timestamp: string;           // ISO datetime
  severity: "info" | "warning" | "critical" | "emergency";
  source: string;              // Component that generated alert
  title: string;
  message: string;
  category: "performance" | "degradation" | "circuit_breaker" | "resource";
  resolved: boolean;
  resolved_at?: string;        // ISO datetime
  resolved_by?: string;
  
  // Additional context
  related_metrics: Record<string, any>;
  affected_services: string[];
  recommended_actions: string[];
}
```

## Integration Examples

### Python Client

```python
import asyncio
import aiohttp
import json

class DashboardClient:
    def __init__(self, base_url="http://localhost:8092"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    async def get_health(self):
        """Get current health status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"]
                return None
    
    async def get_alerts(self):
        """Get active alerts."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/alerts") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"]
                return []
    
    async def resolve_alert(self, alert_id):
        """Resolve an alert."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_base}/alerts/{alert_id}/resolve") as response:
                return response.status == 200

# Usage
async def main():
    client = DashboardClient()
    
    health = await client.get_health()
    print(f"System health: {health['overall_health']}")
    print(f"Load level: {health['load_level']}")
    
    alerts = await client.get_alerts()
    print(f"Active alerts: {len(alerts)}")
    
    for alert in alerts:
        print(f"  {alert['severity']}: {alert['title']}")

asyncio.run(main())
```

### curl Examples

```bash
# Get current health
curl -X GET "http://localhost:8092/api/v1/health" \
  -H "Content-Type: application/json"

# Get health history (last 6 hours)
curl -X GET "http://localhost:8092/api/v1/health/history?hours=6" \
  -H "Content-Type: application/json"

# Get active alerts
curl -X GET "http://localhost:8092/api/v1/alerts" \
  -H "Content-Type: application/json"

# Resolve an alert
curl -X POST "http://localhost:8092/api/v1/alerts/alert_id_here/resolve" \
  -H "Content-Type: application/json"

# Get system status
curl -X GET "http://localhost:8092/api/v1/system/status" \
  -H "Content-Type: application/json"
```

## Configuration

### DashboardConfig

```python
from lightrag_integration.unified_system_health_dashboard import DashboardConfig

config = DashboardConfig(
    host="0.0.0.0",
    port=8092,
    enable_cors=True,
    enable_websockets=True,
    websocket_update_interval=2.0,
    enable_historical_data=True,
    historical_retention_hours=72,
    enable_alerts=True,
    alert_cooldown_seconds=300,
    db_path="dashboard.db",
    enable_db_persistence=True
)
```

### Environment Variables

The dashboard can be configured using environment variables:

```bash
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8092
export ENABLE_WEBSOCKETS=true
export WEBSOCKET_UPDATE_INTERVAL=2.0
export ENABLE_HISTORICAL_DATA=true
export HISTORICAL_RETENTION_HOURS=72
export ENABLE_ALERTS=true
export ALERT_COOLDOWN_SECONDS=300
export DB_PATH=dashboard.db
export ENABLE_DB_PERSISTENCE=true
```

## Deployment

### Docker Deployment

Generate Docker configuration:

```python
from lightrag_integration.dashboard_integration_helper import generate_docker_compose, get_production_config

config = get_production_config()
generate_docker_compose(config, "docker-compose.yml")
```

Run with Docker Compose:

```bash
docker-compose up -d
```

### Production Deployment

```python
from lightrag_integration.dashboard_integration_helper import DashboardIntegrationHelper, get_production_config

# Create production configuration
config = get_production_config()
config.dashboard_port = 8092
config.enable_ssl = True
config.ssl_cert_path = "/etc/ssl/certs/dashboard.crt"
config.ssl_key_path = "/etc/ssl/private/dashboard.key"

# Deploy dashboard
helper = DashboardIntegrationHelper(config)
dashboard, report = await helper.deploy_dashboard()

if report["status"] == "success":
    await dashboard.start()
```

## Health Checks and Monitoring

### Kubernetes Health Check

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8092
  initialDelaySeconds: 30
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /api/v1/system/status
    port: 8092
  initialDelaySeconds: 5
  periodSeconds: 10
```

### External Monitoring Integration

The dashboard exposes metrics that can be consumed by external monitoring systems:

```python
import requests

def check_dashboard_health():
    try:
        response = requests.get("http://localhost:8092/api/v1/health", timeout=10)
        if response.status_code == 200:
            data = response.json()["data"]
            return {
                "status": "healthy" if data["overall_health"] in ["healthy", "warning"] else "unhealthy",
                "load_level": data["load_level"],
                "degradation_active": data["degradation_active"],
                "emergency_mode": data["emergency_mode"],
                "active_alerts": data["active_alerts_count"]
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

## Troubleshooting

### Common Issues

#### WebSocket Connection Fails
- Check firewall settings
- Verify WebSocket endpoint URL
- Ensure dashboard is running and accessible

#### No Monitoring Data
- Verify monitoring systems are registered
- Check that graceful degradation orchestrator is running
- Review logs for integration errors

#### High Memory Usage
- Reduce historical retention hours
- Limit WebSocket connections
- Decrease update frequency

#### SSL Certificate Issues
- Verify certificate paths
- Check certificate validity
- Ensure proper file permissions

### Debug Endpoints

```bash
# Check system integration status
curl http://localhost:8092/api/v1/system/status

# Check database connectivity (if enabled)
# Look for database-related errors in logs

# Verify WebSocket functionality
# Use browser developer tools to monitor WebSocket messages
```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Dashboard-specific logging
logging.getLogger("unified_system_health_dashboard").setLevel(logging.DEBUG)
```

## Performance Considerations

### Scalability

- **WebSocket Connections:** Default limit of 100 concurrent connections
- **Historical Data:** Default retention of 72 hours (5000 snapshots)
- **Update Frequency:** Default 2-second intervals (configurable)
- **Database:** SQLite for single-instance, consider PostgreSQL for multi-instance

### Optimization

```python
# For high-load environments
config = DashboardConfig(
    websocket_update_interval=5.0,  # Reduce update frequency
    historical_retention_hours=24,   # Reduce retention
    alert_cooldown_seconds=600      # Increase alert cooldown
)
```

## Security

### API Key Authentication

```python
config = DashboardConfig(
    enable_api_key=True,
    api_key="your-secret-api-key"
)
```

Include API key in requests:

```bash
curl -X GET "http://localhost:8092/api/v1/health" \
  -H "X-API-Key: your-secret-api-key"
```

### SSL/TLS

```python
config = DashboardConfig(
    enable_ssl=True,
    ssl_cert_path="/path/to/cert.pem",
    ssl_key_path="/path/to/key.pem"
)
```

## Support and Contributing

For issues and feature requests, please refer to the project documentation or contact the development team.

**Dashboard Version:** 1.0.0  
**API Version:** v1  
**Last Updated:** 2025-08-09