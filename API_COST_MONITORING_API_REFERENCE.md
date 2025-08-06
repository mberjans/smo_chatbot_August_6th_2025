# API Cost Monitoring System - API Reference

## Table of Contents

1. [Overview](#overview)
2. [Dashboard API Endpoints](#dashboard-api-endpoints)
3. [Budget Management API](#budget-management-api)
4. [Cost Tracking API](#cost-tracking-api)
5. [Alert Management API](#alert-management-api)
6. [Circuit Breaker API](#circuit-breaker-api)
7. [Analytics API](#analytics-api)
8. [System Management API](#system-management-api)
9. [Authentication and Security](#authentication-and-security)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [SDK and Client Libraries](#sdk-and-client-libraries)

---

## Overview

The API Cost Monitoring System provides both programmatic Python APIs and REST-like interfaces for comprehensive budget and cost management. This reference covers all available endpoints, methods, request/response formats, and integration patterns.

### API Base Classes

```python
from lightrag_integration import (
    BudgetManagementSystem,
    BudgetManager,
    APIUsageMetricsLogger,
    CostPersistence,
    AlertNotificationSystem,
    RealTimeBudgetMonitor,
    CostCircuitBreakerManager,
    BudgetDashboardAPI
)
```

### Response Format

All API responses follow a consistent format:

```json
{
  "status": "success|error",
  "data": { ... },
  "meta": {
    "timestamp": 1691337600.123,
    "request_id": "req_123456789",
    "version": "1.0.0"
  },
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": { ... }
  }
}
```

---

## Dashboard API Endpoints

### Get Dashboard Overview

Retrieve comprehensive dashboard overview with all key metrics.

**Python API:**
```python
dashboard = BudgetDashboardAPI(...)
overview = dashboard.get_dashboard_overview()
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "metrics": {
      "budget_health_score": 85.2,
      "budget_health_status": "healthy",
      "daily_cost": 23.45,
      "daily_budget": 50.0,
      "daily_percentage": 46.9,
      "monthly_cost": 346.78,
      "monthly_budget": 1000.0,
      "monthly_percentage": 34.7,
      "daily_projected_cost": 48.20,
      "monthly_projected_cost": 1012.50,
      "cost_trend_percentage": 12.5,
      "cost_trend_direction": "increasing",
      "active_alerts": 2,
      "critical_alerts": 0,
      "circuit_breakers_healthy": 3,
      "circuit_breakers_open": 0,
      "total_api_calls": 1247,
      "average_response_time": 1250.5,
      "error_rate_percentage": 2.1,
      "tokens_consumed": 125000,
      "system_uptime": 86400.5,
      "monitoring_active": true,
      "last_update_time": 1691337600.123
    },
    "system_health": {
      "overall_status": "healthy",
      "components": {
        "budget_manager": {"status": "healthy"},
        "real_time_monitor": {"status": "healthy"},
        "circuit_breakers": {"status": "healthy"},
        "alert_system": {"status": "healthy"}
      }
    },
    "recent_alerts": [
      {
        "timestamp": 1691337000.0,
        "type": "budget_warning",
        "severity": "warning",
        "message": "Daily budget usage reached 75%"
      }
    ],
    "trends": {
      "spending_trends": {
        "daily_average": 45.23,
        "weekly_average": 316.61,
        "trend_direction": "stable"
      }
    }
  }
}
```

### Get Budget Status

**Python API:**
```python
status = dashboard.get_budget_status(include_projections=True)
```

**Parameters:**
- `include_projections` (boolean, optional): Include cost projections (default: true)

**Response:**
```json
{
  "status": "success",
  "data": {
    "budget_summary": {
      "daily_budget": {
        "total_cost": 23.45,
        "budget_limit": 50.0,
        "percentage_used": 46.9,
        "over_budget": false,
        "projected_end_of_day": 48.20
      },
      "monthly_budget": {
        "total_cost": 346.78,
        "budget_limit": 1000.0,
        "percentage_used": 34.7,
        "over_budget": false,
        "projected_end_of_month": 1012.50
      },
      "budget_health": "healthy"
    },
    "projections": {
      "daily": {
        "projected_cost": 48.20,
        "confidence": 0.85,
        "projection_method": "linear_trend"
      },
      "monthly": {
        "projected_cost": 1012.50,
        "confidence": 0.78,
        "projection_method": "usage_pattern"
      }
    }
  }
}
```

### Get Cost Analytics

**Python API:**
```python
analytics = dashboard.get_cost_analytics(
    time_range="last_7_days",
    granularity="daily",
    include_categories=True
)
```

**Parameters:**
- `time_range` (string): Time range for analysis
  - Options: `"last_hour"`, `"last_6_hours"`, `"last_24_hours"`, `"last_7_days"`, `"last_30_days"`, `"current_month"`, `"current_year"`
- `granularity` (string): Data granularity (`"hourly"`, `"daily"`)
- `include_categories` (boolean): Include research category breakdown

**Response:**
```json
{
  "status": "success",
  "data": {
    "trends": {
      "time_range": "last_7_days",
      "granularity": "daily",
      "time_series": {
        "2025-08-01": 12.34,
        "2025-08-02": 15.67,
        "2025-08-03": 18.90,
        "2025-08-04": 22.15,
        "2025-08-05": 19.75,
        "2025-08-06": 23.45
      },
      "total_cost": 112.26,
      "average_cost": 18.71,
      "min_cost": 12.34,
      "max_cost": 23.45,
      "trend_analysis": {
        "trend_percentage": 15.2,
        "trend_direction": "increasing",
        "confidence": 0.78
      },
      "forecast": [
        {
          "period": 1,
          "projected_value": 25.80,
          "confidence": 0.85
        },
        {
          "period": 2,
          "projected_value": 27.20,
          "confidence": 0.80
        }
      ]
    },
    "category_analysis": {
      "category_breakdown": {
        "metabolite_identification": 50.67,
        "pathway_analysis": 28.34,
        "literature_search": 20.15,
        "data_validation": 13.10
      },
      "top_categories": [
        {
          "category": "metabolite_identification",
          "total_cost": 50.67,
          "total_calls": 234,
          "average_cost_per_call": 0.2165,
          "percentage": 45.1
        }
      ]
    },
    "efficiency_analysis": {
      "cost_per_token": 0.000025,
      "cost_per_successful_operation": 0.089,
      "overall_efficiency_score": 78.5,
      "optimization_opportunities": [
        {
          "type": "model_optimization",
          "potential_savings": 12.50,
          "recommendation": "Consider using GPT-4o-mini for simple queries"
        }
      ]
    }
  }
}
```

---

## Budget Management API

### BudgetManager Class API

#### Initialize Budget Manager

```python
from lightrag_integration.budget_manager import BudgetManager, BudgetThreshold

budget_manager = BudgetManager(
    cost_persistence=cost_persistence,
    daily_budget_limit=50.0,
    monthly_budget_limit=1000.0,
    thresholds=BudgetThreshold(
        warning_percentage=75.0,
        critical_percentage=90.0,
        exceeded_percentage=100.0
    )
)
```

#### Check Budget Status

```python
status = budget_manager.check_budget_status(
    cost_amount=5.0,
    operation_type="llm_call",
    research_category="metabolite_identification"
)
```

**Parameters:**
- `cost_amount` (float): Estimated cost of the operation
- `operation_type` (string): Type of operation
- `research_category` (string, optional): Research category

**Response:**
```python
{
    "operation_allowed": True,
    "budget_health": "healthy",
    "daily_status": {
        "total_cost": 23.45,
        "projected_cost": 28.45,
        "percentage_used": 46.9,
        "budget_limit": 50.0,
        "over_budget": False
    },
    "monthly_status": {
        "total_cost": 346.78,
        "projected_cost": 351.78,
        "percentage_used": 34.7,
        "budget_limit": 1000.0,
        "over_budget": False
    },
    "alerts_generated": [],
    "cost_breakdown": {
        "by_category": {
            "metabolite_identification": 156.34,
            "pathway_analysis": 89.23
        },
        "by_operation": {
            "llm_call": 201.45,
            "embedding_call": 45.12
        }
    }
}
```

#### Update Budget Limits

```python
budget_manager.update_budget_limits(
    daily_budget=75.0,
    monthly_budget=1500.0
)
```

#### Get Budget Summary

```python
summary = budget_manager.get_budget_summary()
```

**Response:**
```python
{
    "daily_budget": {
        "total_cost": 23.45,
        "budget_limit": 50.0,
        "percentage_used": 46.9,
        "over_budget": False,
        "remaining_budget": 26.55,
        "days_in_period": 1,
        "last_updated": "2025-08-06T14:30:00Z"
    },
    "monthly_budget": {
        "total_cost": 346.78,
        "budget_limit": 1000.0,
        "percentage_used": 34.7,
        "over_budget": False,
        "remaining_budget": 653.22,
        "days_in_period": 6,
        "last_updated": "2025-08-06T14:30:00Z"
    },
    "budget_health": "healthy",
    "total_operations": 1247,
    "average_cost_per_operation": 0.278
}
```

#### Get Spending Trends

```python
trends = budget_manager.get_spending_trends(days=30)
```

**Response:**
```python
{
    "period_days": 30,
    "total_cost": 1234.56,
    "average_daily_cost": 41.15,
    "trend_direction": "increasing",
    "trend_percentage": 8.5,
    "daily_costs": {
        "2025-07-07": 35.23,
        "2025-07-08": 42.17,
        # ... more daily data
    },
    "cost_distribution": {
        "weekdays": 89.2,
        "weekends": 10.8
    },
    "peak_usage_hours": [9, 10, 11, 14, 15, 16]
}
```

---

## Cost Tracking API

### APIUsageMetricsLogger Class API

#### Initialize Metrics Logger

```python
from lightrag_integration.api_metrics_logger import APIUsageMetricsLogger

metrics_logger = APIUsageMetricsLogger(
    config=lightrag_config,
    cost_persistence=cost_persistence,
    budget_manager=budget_manager
)
```

#### Track API Call (Context Manager)

```python
with metrics_logger.track_api_call(
    operation_name="metabolite_analysis",
    model_name="gpt-4o-mini",
    research_category="metabolite_identification"
) as tracker:
    # Make API call
    response = make_api_call()
    
    # Update tracking data
    tracker.set_tokens(prompt=100, completion=50)
    tracker.set_cost(0.023)
    tracker.set_response_details(
        response_time_ms=1500,
        request_size=2048,
        response_size=4096
    )
```

#### Log Individual Metric

```python
metrics_logger.log_api_call(
    operation_name="literature_search",
    model_name="gpt-4o-mini",
    token_usage={
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75
    },
    cost_usd=0.005,
    response_time_ms=800,
    success=True,
    research_category="literature_search"
)
```

#### Log Batch Operation

```python
metrics_logger.log_batch_operation(
    operation_name="batch_metabolite_analysis",
    batch_size=10,
    total_tokens=5000,
    total_cost=0.25,
    processing_time_ms=15000,
    success_count=9,
    error_count=1,
    research_category="metabolite_identification"
)
```

#### Get Performance Summary

```python
performance = metrics_logger.get_performance_summary()
```

**Response:**
```python
{
    "current_hour": {
        "total_calls": 45,
        "successful_calls": 43,
        "failed_calls": 2,
        "total_cost": 2.34,
        "avg_cost_per_call": 0.052,
        "total_tokens": 12500,
        "avg_tokens_per_call": 278,
        "avg_response_time_ms": 1250,
        "error_rate_percent": 4.4
    },
    "current_day": {
        "total_calls": 387,
        "successful_calls": 375,
        "failed_calls": 12,
        "total_cost": 18.92,
        "avg_cost_per_call": 0.049,
        "total_tokens": 98750,
        "avg_response_time_ms": 1180,
        "error_rate_percent": 3.1
    },
    "model_breakdown": {
        "gpt-4o-mini": {
            "total_calls": 345,
            "total_cost": 15.67,
            "success_rate": 96.8
        },
        "gpt-4o": {
            "total_calls": 42,
            "total_cost": 3.25,
            "success_rate": 95.2
        }
    },
    "category_breakdown": {
        "metabolite_identification": {
            "total_calls": 198,
            "total_cost": 9.87,
            "avg_cost_per_call": 0.050
        }
    }
}
```

---

## Alert Management API

### AlertNotificationSystem Class API

#### Initialize Alert System

```python
from lightrag_integration.alert_system import (
    AlertNotificationSystem, AlertConfig, EmailAlertConfig, SlackAlertConfig
)

alert_config = AlertConfig(
    email_config=EmailAlertConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="alerts@company.com",
        password="app-password",
        sender_email="alerts@company.com",
        recipient_emails=["admin@company.com"]
    ),
    slack_config=SlackAlertConfig(
        webhook_url="https://hooks.slack.com/services/...",
        channel="#budget-alerts"
    )
)

alert_system = AlertNotificationSystem(alert_config)
```

#### Send Alert

```python
from lightrag_integration.budget_manager import BudgetAlert, AlertLevel

alert = BudgetAlert(
    timestamp=time.time(),
    alert_level=AlertLevel.WARNING,
    period_type="daily",
    period_key="2025-08-06",
    current_cost=37.50,
    budget_limit=50.0,
    percentage_used=75.0,
    threshold_percentage=75.0,
    message="Daily budget warning: 75% of budget used"
)

delivery_result = alert_system.send_alert(alert)
```

**Response:**
```python
{
    "alert_id": "alert_123456789",
    "delivery_results": {
        "email": {
            "success": True,
            "timestamp": 1691337600.0,
            "recipients": ["admin@company.com"],
            "message_id": "msg_987654321"
        },
        "slack": {
            "success": True,
            "timestamp": 1691337601.0,
            "channel": "#budget-alerts",
            "message_ts": "1691337601.123456"
        },
        "logging": {
            "success": True,
            "timestamp": 1691337600.0,
            "log_level": "WARNING"
        }
    },
    "overall_success": True,
    "failed_channels": []
}
```

#### Test Alert Channels

```python
test_results = alert_system.test_channels()
```

**Response:**
```python
{
    "email": {
        "success": True,
        "message": "Test email sent successfully",
        "response_time_ms": 1250
    },
    "slack": {
        "success": True,
        "message": "Test Slack message sent",
        "response_time_ms": 800
    },
    "webhook": {
        "success": False,
        "error": "Connection timeout",
        "response_time_ms": 5000
    }
}
```

#### Get Delivery Statistics

```python
delivery_stats = alert_system.get_delivery_stats()
```

**Response:**
```python
{
    "total_alerts_sent": 456,
    "successful_deliveries": 442,
    "failed_deliveries": 14,
    "success_rate": 96.9,
    "channels": {
        "email": {
            "total_attempts": 456,
            "successful_deliveries": 450,
            "failed_deliveries": 6,
            "success_rate": 98.7,
            "avg_response_time_ms": 1200
        },
        "slack": {
            "total_attempts": 456,
            "successful_deliveries": 448,
            "failed_deliveries": 8,
            "success_rate": 98.2,
            "avg_response_time_ms": 850
        }
    },
    "recent_failures": [
        {
            "timestamp": 1691337000.0,
            "channel": "email",
            "error": "SMTP authentication failed"
        }
    ]
}
```

---

## Circuit Breaker API

### CostCircuitBreakerManager Class API

#### Initialize Circuit Breaker Manager

```python
from lightrag_integration.cost_based_circuit_breaker import (
    CostCircuitBreakerManager, CostThresholdRule, CostThresholdType
)

cb_manager = CostCircuitBreakerManager(
    budget_manager=budget_manager,
    cost_persistence=cost_persistence
)
```

#### Create Circuit Breaker

```python
custom_rules = [
    CostThresholdRule(
        rule_id="daily_budget_protection",
        threshold_type=CostThresholdType.PERCENTAGE_DAILY,
        threshold_value=95.0,
        action="block"
    )
]

circuit_breaker = cb_manager.create_circuit_breaker(
    name="llm_operations",
    threshold_rules=custom_rules,
    failure_threshold=5,
    recovery_timeout=60.0
)
```

#### Execute with Protection

```python
try:
    result = cb_manager.execute_with_protection(
        breaker_name="llm_operations",
        operation_callable=your_api_function,
        operation_type="llm_call",
        model_name="gpt-4o-mini",
        estimated_tokens={"input": 100, "output": 50}
    )
except CircuitBreakerError as e:
    print(f"Operation blocked: {e}")
```

#### Get System Status

```python
system_status = cb_manager.get_system_status()
```

**Response:**
```python
{
    "circuit_breakers": {
        "llm_operations": {
            "name": "llm_operations",
            "state": "closed",
            "failure_count": 0,
            "throttle_rate": 1.0,
            "statistics": {
                "total_calls": 1247,
                "allowed_calls": 1235,
                "blocked_calls": 12,
                "cost_blocked_calls": 8,
                "total_estimated_cost": 65.43,
                "total_actual_cost": 62.18,
                "cost_savings": 3.25
            },
            "last_failure_time": None,
            "last_success_time": 1691337600.0
        }
    },
    "manager_statistics": {
        "breakers_created": 3,
        "total_operations": 2456,
        "total_blocks": 23,
        "total_cost_saved": 12.45
    },
    "system_health": {
        "status": "healthy",
        "message": "All circuit breakers operational",
        "total_breakers": 3
    }
}
```

#### Update Operation Cost

```python
cb_manager.update_operation_cost(
    breaker_name="llm_operations",
    operation_id="op_123456",
    actual_cost=0.045,
    operation_type="llm_call"
)
```

---

## Analytics API

### Get Cost Report

```python
from datetime import datetime, timezone

cost_report = dashboard.get_cost_report(
    start_date="2025-08-01T00:00:00Z",
    end_date="2025-08-06T23:59:59Z",
    format="json"
)
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "cost_report": {
      "summary": {
        "total_cost": 234.56,
        "total_calls": 1247,
        "success_rate": 96.8,
        "avg_cost_per_call": 0.188,
        "date_range": {
          "start": "2025-08-01T00:00:00Z",
          "end": "2025-08-06T23:59:59Z"
        }
      },
      "daily_costs": {
        "2025-08-01": 45.23,
        "2025-08-02": 38.90,
        "2025-08-03": 52.17,
        "2025-08-04": 41.68,
        "2025-08-05": 33.14,
        "2025-08-06": 23.44
      },
      "category_breakdown": {
        "metabolite_identification": {
          "cost": 105.67,
          "calls": 487,
          "percentage": 45.1
        },
        "pathway_analysis": {
          "cost": 67.89,
          "calls": 234,
          "percentage": 28.9
        }
      },
      "model_breakdown": {
        "gpt-4o-mini": {
          "cost": 156.78,
          "calls": 1089,
          "avg_cost": 0.144
        },
        "gpt-4o": {
          "cost": 77.78,
          "calls": 158,
          "avg_cost": 0.492
        }
      },
      "hourly_distribution": {
        "00": 2.34, "01": 1.23, "02": 0.89,
        "09": 15.67, "10": 18.90, "11": 22.45,
        "14": 19.78, "15": 21.34, "16": 17.89
      }
    }
  }
}
```

### Get Research Analysis

```python
research_analysis = cost_persistence.get_research_analysis(days=30)
```

**Response:**
```python
{
    "analysis_period": {
        "days": 30,
        "start_date": "2025-07-07",
        "end_date": "2025-08-06"
    },
    "total_cost": 1234.56,
    "total_calls": 5678,
    "category_breakdown": {
        "metabolite_identification": 456.78,
        "pathway_analysis": 234.56,
        "literature_search": 345.67,
        "data_validation": 197.55
    },
    "top_categories": [
        {
            "category": "metabolite_identification",
            "total_cost": 456.78,
            "total_calls": 1234,
            "percentage": 37.0,
            "avg_cost_per_call": 0.370
        }
    ],
    "efficiency_metrics": {
        "most_efficient_category": "data_validation",
        "least_efficient_category": "pathway_analysis",
        "overall_efficiency_score": 78.5
    },
    "trends": {
        "cost_trend": "increasing",
        "usage_trend": "stable",
        "efficiency_trend": "improving"
    }
}
```

---

## System Management API

### Health Check

```python
health_check = dashboard.get_api_health_check()
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1691337600.123,
  "uptime_seconds": 86400.5,
  "request_count": 15678,
  "components": {
    "budget_manager": true,
    "api_metrics_logger": true,
    "cost_persistence": true,
    "alert_system": true,
    "real_time_monitor": true,
    "circuit_breaker_manager": true
  }
}
```

### Trigger Budget Check

```python
check_result = dashboard.trigger_budget_check()
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "results": {
      "budget_status": {
        "daily_budget": {
          "percentage_used": 46.9,
          "over_budget": false
        }
      },
      "monitoring_cycle": {
        "cycle_completed": true,
        "alerts_generated": 0,
        "anomalies_detected": 0
      }
    }
  }
}
```

### Performance Metrics

```python
performance = dashboard.get_performance_metrics()
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "performance_summary": {
      "current_hour": {
        "total_calls": 145,
        "avg_response_time_ms": 1250,
        "error_rate_percent": 2.1,
        "throughput_per_minute": 2.4
      },
      "system_resources": {
        "memory_usage_mb": 256.7,
        "cpu_usage_percent": 12.5,
        "disk_usage_mb": 1024.3
      },
      "database_performance": {
        "avg_query_time_ms": 5.2,
        "connection_pool_usage": 3,
        "total_connections": 10
      }
    }
  }
}
```

---

## Authentication and Security

### API Key Management

The system uses OpenAI API keys for authentication. Keys should be managed securely:

```python
# Environment variable (recommended)
import os
api_key = os.getenv("OPENAI_API_KEY")

# Secure key storage
from lightrag_integration.security import SecureKeyManager

key_manager = SecureKeyManager(password="secure-password")
encrypted_key = key_manager.encrypt_api_key(api_key)
```

### Request Authentication

For REST-like interfaces (if implemented), authentication follows standard patterns:

```http
GET /api/v1/dashboard/overview
Authorization: Bearer your-system-token
X-API-Key: your-api-key
Content-Type: application/json
```

### Access Control

```python
# Role-based access example
from lightrag_integration.security import AccessControl

access_control = AccessControl()
access_control.add_user_role("researcher", permissions=["read_budgets", "track_costs"])
access_control.add_user_role("admin", permissions=["read_budgets", "track_costs", "modify_budgets", "manage_alerts"])

# Check permissions
if access_control.check_permission(user_id, "modify_budgets"):
    # Allow budget modification
    pass
```

---

## Error Handling

### Standard Error Codes

| Code | Description |
|------|-------------|
| `BUDGET_EXCEEDED` | Operation would exceed budget limits |
| `CIRCUIT_BREAKER_OPEN` | Circuit breaker is blocking operations |
| `INVALID_PARAMETERS` | Invalid request parameters |
| `AUTHENTICATION_FAILED` | API key authentication failed |
| `RATE_LIMIT_EXCEEDED` | Too many requests in time window |
| `DATABASE_ERROR` | Database operation failed |
| `ALERT_DELIVERY_FAILED` | Alert notification failed |
| `CONFIGURATION_ERROR` | System configuration invalid |

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "BUDGET_EXCEEDED",
    "message": "Daily budget limit of $50.00 would be exceeded",
    "details": {
      "current_cost": 47.50,
      "requested_cost": 5.00,
      "budget_limit": 50.00,
      "projected_total": 52.50
    },
    "retry_after": null,
    "documentation_url": "https://docs.example.com/errors/budget-exceeded"
  },
  "meta": {
    "timestamp": 1691337600.123,
    "request_id": "req_123456789"
  }
}
```

### Exception Handling in Python

```python
from lightrag_integration.exceptions import (
    BudgetExceededException,
    CircuitBreakerError,
    ConfigurationError
)

try:
    with budget_system.track_operation("llm_call") as tracker:
        result = expensive_operation()
        tracker.set_cost(5.0)
        
except BudgetExceededException as e:
    # Handle budget exceeded
    logger.warning(f"Budget exceeded: {e}")
    
except CircuitBreakerError as e:
    # Handle circuit breaker
    logger.warning(f"Circuit breaker active: {e}")
    
except ConfigurationError as e:
    # Handle configuration issues
    logger.error(f"Configuration error: {e}")
    
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

---

## Rate Limiting

### Request Rate Limits

The system implements intelligent rate limiting to protect against abuse:

```python
# Rate limiting configuration
rate_limits = {
    "dashboard_requests": "100 per minute",
    "budget_checks": "1000 per minute", 
    "cost_tracking": "unlimited",
    "alert_sending": "10 per minute"
}
```

### Handling Rate Limits

```python
from lightrag_integration.exceptions import RateLimitExceeded

try:
    overview = dashboard.get_dashboard_overview()
except RateLimitExceeded as e:
    retry_after = e.retry_after_seconds
    print(f"Rate limited. Retry after {retry_after} seconds")
    time.sleep(retry_after)
```

---

## SDK and Client Libraries

### Python SDK

The primary interface is the Python SDK included with the system:

```python
# Main SDK classes
from lightrag_integration import (
    BudgetManagementFactory,
    BudgetManagementSystem,
    LightRAGConfig
)

# Initialize complete system
config = LightRAGConfig.get_config()
system = BudgetManagementFactory.create_complete_system(
    lightrag_config=config,
    daily_budget_limit=50.0
)
```

### REST Client (if implemented)

```python
import requests
from typing import Dict, Any

class BudgetMonitoringClient:
    """REST client for budget monitoring API."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get dashboard overview."""
        response = self.session.get(f"{self.base_url}/api/v1/dashboard/overview")
        response.raise_for_status()
        return response.json()
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        response = self.session.get(f"{self.base_url}/api/v1/budget/status")
        response.raise_for_status()
        return response.json()
    
    def track_cost(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track API cost."""
        response = self.session.post(
            f"{self.base_url}/api/v1/costs/track",
            json=cost_data
        )
        response.raise_for_status()
        return response.json()

# Usage
client = BudgetMonitoringClient("http://localhost:8000", "your-api-key")
overview = client.get_dashboard_overview()
```

### JavaScript/TypeScript Client

```typescript
interface BudgetStatus {
  daily_cost: number;
  daily_budget: number;
  daily_percentage: number;
  monthly_cost: number;
  monthly_budget: number;
  monthly_percentage: number;
  budget_health: string;
}

class BudgetMonitoringClient {
  private baseUrl: string;
  private apiKey: string;
  
  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
  }
  
  async getBudgetStatus(): Promise<BudgetStatus> {
    const response = await fetch(`${this.baseUrl}/api/v1/budget/status`, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result.data.budget_summary;
  }
  
  async trackCost(costData: {
    operation_type: string;
    cost: number;
    tokens?: number;
    model?: string;
  }): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/v1/costs/track`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(costData)
    });
    
    if (!response.ok) {
      throw new Error(`Cost tracking failed: ${response.statusText}`);
    }
  }
}
```

### CLI Tool

```bash
# Install CLI tool
pip install lightrag-budget-cli

# Basic usage
lightrag-budget status
lightrag-budget overview --format json
lightrag-budget report --start 2025-08-01 --end 2025-08-06
lightrag-budget set-budget --daily 75.0 --monthly 1500.0

# Configuration
lightrag-budget config set OPENAI_API_KEY "your-key"
lightrag-budget config set DAILY_BUDGET_LIMIT "50.0"
```

---

This comprehensive API reference provides complete documentation for all interfaces and methods available in the API Cost Monitoring System. For implementation examples, see the [Developer Guide](./API_COST_MONITORING_DEVELOPER_GUIDE.md), and for practical usage scenarios, refer to the [User Guide](./API_COST_MONITORING_USER_GUIDE.md).

---

*This API reference is part of the Clinical Metabolomics Oracle API Cost Monitoring System documentation suite.*