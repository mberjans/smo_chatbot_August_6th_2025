# API Cost Monitoring System - Configuration Reference

## Table of Contents

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [Configuration File Format](#configuration-file-format)
4. [Core Settings](#core-settings)
5. [Budget Configuration](#budget-configuration)
6. [Alert System Configuration](#alert-system-configuration)
7. [Logging Configuration](#logging-configuration)
8. [Database Configuration](#database-configuration)
9. [Circuit Breaker Configuration](#circuit-breaker-configuration)
10. [Advanced Settings](#advanced-settings)
11. [Validation Rules](#validation-rules)
12. [Configuration Templates](#configuration-templates)

---

## Overview

The API Cost Monitoring System uses a hierarchical configuration approach that supports:
- **Environment Variables** (highest priority)
- **Configuration Files** (JSON format)
- **Programmatic Configuration** (Python dictionaries)
- **Default Values** (fallback)

Configuration is managed through the `LightRAGConfig` class, which provides validation, type conversion, and comprehensive error handling.

---

## Environment Variables

### Core API Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | string | None | **Required.** OpenAI API key for accessing GPT and embedding models |
| `LIGHTRAG_MODEL` | string | "gpt-4o-mini" | Default LLM model for text generation |
| `LIGHTRAG_EMBEDDING_MODEL` | string | "text-embedding-3-small" | Default embedding model |

### Budget Management

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_DAILY_BUDGET_LIMIT` | float | None | Daily spending limit in USD (e.g., "50.0") |
| `LIGHTRAG_MONTHLY_BUDGET_LIMIT` | float | None | Monthly spending limit in USD (e.g., "1000.0") |
| `LIGHTRAG_COST_ALERT_THRESHOLD` | float | 80.0 | Alert threshold as percentage of budget (0-100) |
| `LIGHTRAG_ENABLE_BUDGET_ALERTS` | boolean | true | Enable/disable budget alert notifications |
| `LIGHTRAG_ENABLE_COST_TRACKING` | boolean | true | Enable/disable cost tracking functionality |

### System Directories

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_WORKING_DIR` | path | current directory | Main working directory for the system |
| `LIGHTRAG_LOG_DIR` | path | "logs" | Directory for log files |
| `LIGHTRAG_COST_DB_PATH` | path | "cost_tracking.db" | Path to SQLite cost database |

### Performance Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_MAX_ASYNC` | integer | 16 | Maximum concurrent async operations |
| `LIGHTRAG_MAX_TOKENS` | integer | 32768 | Maximum tokens per API request |

### Logging Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_LOG_LEVEL` | string | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LIGHTRAG_ENABLE_FILE_LOGGING` | boolean | true | Enable file-based logging |
| `LIGHTRAG_LOG_MAX_BYTES` | integer | 10485760 | Maximum log file size (10MB) |
| `LIGHTRAG_LOG_BACKUP_COUNT` | integer | 5 | Number of backup log files to keep |

### Data Persistence

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_COST_PERSISTENCE_ENABLED` | boolean | true | Enable cost data persistence |
| `LIGHTRAG_ENABLE_RESEARCH_CATEGORIZATION` | boolean | true | Enable research category tracking |
| `LIGHTRAG_ENABLE_AUDIT_TRAIL` | boolean | true | Enable audit trail logging |
| `LIGHTRAG_MAX_COST_RETENTION_DAYS` | integer | 365 | Maximum days to retain cost data |
| `LIGHTRAG_COST_REPORT_FREQUENCY` | string | "daily" | Report frequency (hourly, daily, weekly, monthly) |

### Alert System Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ALERT_EMAIL_SMTP_SERVER` | string | None | SMTP server hostname |
| `ALERT_EMAIL_SMTP_PORT` | integer | 587 | SMTP server port |
| `ALERT_EMAIL_USERNAME` | string | None | SMTP username |
| `ALERT_EMAIL_PASSWORD` | string | None | SMTP password or app-specific password |
| `ALERT_EMAIL_SENDER` | string | None | From email address |
| `ALERT_EMAIL_RECIPIENTS` | string | None | Comma-separated recipient email addresses |
| `ALERT_EMAIL_USE_TLS` | boolean | true | Use TLS encryption for SMTP |
| `ALERT_SLACK_WEBHOOK_URL` | string | None | Slack webhook URL |
| `ALERT_SLACK_CHANNEL` | string | "#budget-alerts" | Slack channel for alerts |
| `ALERT_SLACK_USERNAME` | string | "Budget Monitor" | Bot username for Slack alerts |
| `ALERT_WEBHOOK_URLS` | string | None | Comma-separated webhook URLs |

---

## Configuration File Format

### JSON Configuration File

```json
{
  "api_key": "your-openai-api-key",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "working_dir": "/path/to/working/directory",
  "max_async": 16,
  "max_tokens": 32768,
  
  "daily_budget_limit": 50.0,
  "monthly_budget_limit": 1000.0,
  "cost_alert_threshold_percentage": 80.0,
  "enable_budget_alerts": true,
  "enable_cost_tracking": true,
  
  "log_level": "INFO",
  "log_dir": "logs",
  "enable_file_logging": true,
  "log_max_bytes": 10485760,
  "log_backup_count": 5,
  
  "cost_db_path": "cost_tracking.db",
  "enable_research_categorization": true,
  "enable_audit_trail": true,
  "max_cost_retention_days": 365,
  "cost_report_frequency": "daily"
}
```

### Loading Configuration from File

```python
from lightrag_integration.config import LightRAGConfig

# Load from JSON file
config = LightRAGConfig.from_file("/path/to/config.json")

# Load with validation
config = LightRAGConfig.get_config(
    source="/path/to/config.json",
    validate_config=True,
    ensure_dirs=True
)
```

---

## Core Settings

### API Configuration

```python
from lightrag_integration.config import LightRAGConfig

# Basic API configuration
config = LightRAGConfig(
    api_key="your-openai-api-key",
    model="gpt-4o",  # Default LLM model
    embedding_model="text-embedding-3-large",  # Default embedding model
    max_async=32,  # Increased concurrency
    max_tokens=8192  # Token limit per request
)
```

**Model Options:**
- **LLM Models**: `gpt-4o`, `gpt-4o-mini`
- **Embedding Models**: `text-embedding-3-small`, `text-embedding-3-large`

### Directory Configuration

```python
from pathlib import Path

config = LightRAGConfig(
    working_dir=Path("/opt/metabolomics/lightrag"),
    graph_storage_dir=Path("/opt/metabolomics/lightrag/graphs"),
    log_dir=Path("/var/log/metabolomics"),
    cost_db_path=Path("/opt/metabolomics/data/costs.db")
)

# Ensure directories are created
config.ensure_directories()
```

---

## Budget Configuration

### Basic Budget Settings

```python
config = LightRAGConfig(
    daily_budget_limit=75.0,      # $75 per day
    monthly_budget_limit=1500.0,  # $1500 per month
    cost_alert_threshold_percentage=85.0,  # Alert at 85%
    enable_budget_alerts=True,
    enable_cost_tracking=True
)
```

### Advanced Budget Configuration

```python
from lightrag_integration.budget_manager import BudgetThreshold

# Custom threshold configuration
custom_thresholds = BudgetThreshold(
    warning_percentage=70.0,    # Warning at 70%
    critical_percentage=85.0,   # Critical at 85%
    exceeded_percentage=100.0   # Block at 100%
)

# Budget configuration for different environments
budget_configs = {
    "development": {
        "daily_budget_limit": 10.0,
        "monthly_budget_limit": 200.0,
        "cost_alert_threshold_percentage": 90.0
    },
    "staging": {
        "daily_budget_limit": 25.0,
        "monthly_budget_limit": 500.0,
        "cost_alert_threshold_percentage": 80.0
    },
    "production": {
        "daily_budget_limit": 100.0,
        "monthly_budget_limit": 2000.0,
        "cost_alert_threshold_percentage": 75.0
    }
}
```

### Research Category Budget Allocation

```python
# Category-specific budget allocation
category_budgets = {
    "metabolite_identification": {
        "daily_percentage": 40.0,  # 40% of daily budget
        "priority": "high"
    },
    "pathway_analysis": {
        "daily_percentage": 30.0,  # 30% of daily budget
        "priority": "high"
    },
    "literature_search": {
        "daily_percentage": 20.0,  # 20% of daily budget
        "priority": "medium"
    },
    "data_validation": {
        "daily_percentage": 10.0,  # 10% of daily budget
        "priority": "low"
    }
}
```

---

## Alert System Configuration

### Email Alert Configuration

```python
from lightrag_integration.alert_system import EmailAlertConfig

email_config = EmailAlertConfig(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="research-alerts@university.edu",
    password="app-specific-password",
    sender_email="research-alerts@university.edu",
    recipient_emails=[
        "pi@university.edu",
        "lab-admin@university.edu",
        "it-support@university.edu"
    ],
    use_tls=True,
    template_style="detailed"  # or "simple"
)
```

### Slack Alert Configuration

```python
from lightrag_integration.alert_system import SlackAlertConfig

slack_config = SlackAlertConfig(
    webhook_url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
    channel="#metabolomics-alerts",
    username="Budget Monitor Bot",
    icon_emoji=":moneybag:",
    mention_users=["@research-lead", "@admin"],
    thread_alerts=True  # Group related alerts in threads
)
```

### Webhook Alert Configuration

```python
from lightrag_integration.alert_system import WebhookAlertConfig

webhook_config = WebhookAlertConfig(
    urls=[
        "https://api.pagerduty.com/integration/v1/enqueue",
        "https://hooks.zapier.com/hooks/catch/1234567/abcdef/"
    ],
    headers={
        "Authorization": "Bearer your-webhook-token",
        "Content-Type": "application/json"
    },
    custom_payload_template={
        "service_key": "your-pagerduty-key",
        "event_type": "trigger",
        "description": "{alert_message}",
        "details": {
            "budget_usage": "{percentage_used}%",
            "cost_amount": "${current_cost}",
            "timestamp": "{timestamp}"
        }
    }
)
```

### Multi-Channel Alert Configuration

```python
from lightrag_integration.alert_system import AlertConfig, AlertChannel

alert_config = AlertConfig(
    email_config=email_config,
    slack_config=slack_config,
    webhook_config=webhook_config,
    enabled_channels={
        AlertChannel.EMAIL,
        AlertChannel.SLACK,
        AlertChannel.WEBHOOK,
        AlertChannel.LOGGING
    },
    alert_rate_limiting={
        "max_alerts_per_hour": 10,
        "max_duplicate_alerts": 3,
        "cooldown_minutes": 15
    },
    escalation_enabled=True,
    escalation_delay_minutes=30
)
```

---

## Logging Configuration

### Basic Logging Setup

```python
config = LightRAGConfig(
    log_level="INFO",
    log_dir=Path("logs"),
    enable_file_logging=True,
    log_max_bytes=10 * 1024 * 1024,  # 10MB
    log_backup_count=5,
    log_filename="lightrag_integration.log"
)

# Setup logging
logger = config.setup_lightrag_logging("metabolomics_oracle")
```

### Advanced Logging Configuration

```python
# Environment-specific logging
logging_configs = {
    "development": {
        "log_level": "DEBUG",
        "enable_file_logging": True,
        "console_logging": True
    },
    "production": {
        "log_level": "INFO",
        "enable_file_logging": True,
        "console_logging": False,
        "structured_logging": True,
        "log_rotation": "daily"
    }
}

# Structured logging for SIEM integration
structured_config = {
    "log_format": "json",
    "include_metadata": True,
    "fields": [
        "timestamp",
        "level",
        "message",
        "component",
        "operation_id",
        "cost_amount",
        "budget_usage"
    ]
}
```

### Log Analysis Configuration

```python
# Log analysis and monitoring
log_monitoring = {
    "error_threshold": {
        "errors_per_minute": 5,
        "alert_channels": ["email", "slack"]
    },
    "performance_monitoring": {
        "slow_query_threshold_ms": 5000,
        "high_cost_threshold": 1.0
    },
    "retention": {
        "debug_logs": "7 days",
        "info_logs": "30 days", 
        "error_logs": "90 days",
        "audit_logs": "365 days"
    }
}
```

---

## Database Configuration

### SQLite Configuration

```python
from pathlib import Path

config = LightRAGConfig(
    cost_db_path=Path("/opt/metabolomics/data/cost_tracking.db"),
    cost_persistence_enabled=True,
    max_cost_retention_days=365,
    enable_audit_trail=True,
    enable_research_categorization=True
)
```

### Database Optimization Settings

```python
# Database performance tuning
database_config = {
    "connection_pool_size": 10,
    "connection_timeout": 30,
    "query_timeout": 60,
    "batch_insert_size": 1000,
    "vacuum_frequency": "weekly",
    "backup_enabled": True,
    "backup_frequency": "daily",
    "backup_retention_days": 30
}

# Database indexes for performance
required_indexes = [
    "idx_cost_records_timestamp",
    "idx_cost_records_operation_type", 
    "idx_cost_records_research_category",
    "idx_budget_tracking_period_key",
    "idx_audit_trail_timestamp"
]
```

### Database Maintenance Configuration

```python
maintenance_config = {
    "auto_vacuum": "incremental",
    "page_size": 4096,
    "cache_size": 10000,
    "temp_store": "memory",
    "journal_mode": "wal",
    "synchronous": "normal",
    "foreign_keys": "on"
}
```

---

## Circuit Breaker Configuration

### Default Circuit Breaker Rules

```python
from lightrag_integration.cost_based_circuit_breaker import (
    CostThresholdRule, CostThresholdType
)

default_rules = [
    # Daily budget protection
    CostThresholdRule(
        rule_id="daily_budget_warning",
        threshold_type=CostThresholdType.PERCENTAGE_DAILY,
        threshold_value=80.0,
        action="throttle",
        throttle_factor=0.7,
        priority=10
    ),
    CostThresholdRule(
        rule_id="daily_budget_critical",
        threshold_type=CostThresholdType.PERCENTAGE_DAILY,
        threshold_value=95.0,
        action="block",
        priority=20
    ),
    
    # Monthly budget protection
    CostThresholdRule(
        rule_id="monthly_budget_critical",
        threshold_type=CostThresholdType.PERCENTAGE_MONTHLY,
        threshold_value=90.0,
        action="throttle",
        throttle_factor=0.5,
        priority=15
    ),
    
    # High-cost operation protection
    CostThresholdRule(
        rule_id="expensive_operation_alert",
        threshold_type=CostThresholdType.OPERATION_COST,
        threshold_value=5.0,  # $5.00 per operation
        action="alert_only",
        priority=5
    )
]
```

### Custom Circuit Breaker Configuration

```python
# Research-specific circuit breaker rules
research_rules = [
    # Metabolite identification protection
    CostThresholdRule(
        rule_id="metabolite_analysis_limit",
        threshold_type=CostThresholdType.OPERATION_COST,
        threshold_value=2.0,
        action="throttle",
        throttle_factor=0.8,
        applies_to_operations=["metabolite_identification"],
        applies_to_categories=["metabolite_analysis"]
    ),
    
    # Literature search cost control
    CostThresholdRule(
        rule_id="literature_search_bulk_limit",
        threshold_type=CostThresholdType.RATE_BASED,
        threshold_value=10.0,  # $10 per hour
        action="throttle",
        throttle_factor=0.6,
        time_window_minutes=60,
        applies_to_operations=["literature_search"]
    ),
    
    # Emergency override for critical research
    CostThresholdRule(
        rule_id="critical_research_override",
        threshold_type=CostThresholdType.PERCENTAGE_DAILY,
        threshold_value=120.0,  # Allow 20% over budget
        action="alert_only",
        allow_emergency_override=True,
        applies_to_categories=["critical_analysis"]
    )
]
```

### Circuit Breaker System Configuration

```python
circuit_breaker_config = {
    "default_failure_threshold": 5,
    "default_recovery_timeout": 60,
    "cost_estimation_enabled": True,
    "historical_learning_enabled": True,
    "emergency_override_enabled": True,
    "override_duration_minutes": 60,
    "monitoring_interval_seconds": 30,
    "health_check_enabled": True
}
```

---

## Advanced Settings

### Performance Optimization

```python
performance_config = {
    # Async operation limits
    "max_concurrent_operations": 50,
    "operation_timeout_seconds": 300,
    "batch_processing_enabled": True,
    "batch_size": 10,
    
    # Caching configuration
    "response_caching_enabled": True,
    "cache_ttl_seconds": 3600,
    "cache_max_size": 1000,
    "cache_compression": True,
    
    # Connection pooling
    "http_connection_pool_size": 20,
    "http_connection_timeout": 30,
    "http_read_timeout": 60,
    
    # Resource monitoring
    "memory_monitoring_enabled": True,
    "memory_threshold_mb": 1024,
    "cpu_monitoring_enabled": True,
    "cpu_threshold_percent": 80
}
```

### Security Configuration

```python
security_config = {
    # API key security
    "api_key_rotation_enabled": True,
    "api_key_rotation_days": 90,
    "api_key_encryption_enabled": True,
    
    # Data protection
    "cost_data_encryption": True,
    "audit_trail_integrity_check": True,
    "pii_anonymization": True,
    
    # Network security
    "tls_verification": True,
    "certificate_pinning": False,
    "proxy_support": True,
    
    # Access control
    "rbac_enabled": False,
    "session_timeout_minutes": 60,
    "audit_all_access": True
}
```

### Monitoring and Observability

```python
monitoring_config = {
    # Metrics collection
    "prometheus_metrics_enabled": True,
    "prometheus_port": 8000,
    "custom_metrics": [
        "budget_usage_percent",
        "api_calls_per_minute",
        "average_cost_per_operation",
        "circuit_breaker_status"
    ],
    
    # Health checks
    "health_check_interval_seconds": 60,
    "health_check_timeout_seconds": 10,
    "health_check_endpoints": [
        "/health",
        "/metrics",
        "/status"
    ],
    
    # Distributed tracing
    "tracing_enabled": False,
    "jaeger_endpoint": "http://localhost:14268/api/traces",
    "trace_sampling_rate": 0.1
}
```

---

## Validation Rules

### Configuration Validation

The system enforces these validation rules:

#### Required Fields
- `api_key`: Must be non-empty string
- `model`: Must be valid OpenAI model name
- `embedding_model`: Must be valid embedding model name

#### Numeric Constraints
```python
validation_rules = {
    "max_async": {
        "type": "integer",
        "minimum": 1,
        "maximum": 1000
    },
    "max_tokens": {
        "type": "integer",
        "minimum": 1,
        "maximum": 128000
    },
    "daily_budget_limit": {
        "type": "float",
        "minimum": 0.01,
        "maximum": 10000.0
    },
    "monthly_budget_limit": {
        "type": "float",
        "minimum": 0.01,
        "maximum": 100000.0
    },
    "cost_alert_threshold_percentage": {
        "type": "float",
        "minimum": 0.0,
        "maximum": 100.0
    }
}
```

#### Path Validation
- All path fields must be valid file system paths
- Directories must be creatable or already exist
- Database path must be in a writable location

#### String Validation
```python
string_validations = {
    "log_level": {
        "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    },
    "cost_report_frequency": {
        "enum": ["hourly", "daily", "weekly", "monthly"]
    },
    "model": {
        "pattern": "^(gpt-4o|gpt-4o-mini)$"
    },
    "embedding_model": {
        "pattern": "^text-embedding-3-(small|large)$"
    }
}
```

### Custom Validation

```python
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError

def validate_research_configuration(config: LightRAGConfig) -> None:
    """Custom validation for research-specific requirements."""
    
    # Validate budget alignment
    if config.monthly_budget_limit and config.daily_budget_limit:
        max_monthly_from_daily = config.daily_budget_limit * 31
        if config.monthly_budget_limit > max_monthly_from_daily * 1.5:
            raise LightRAGConfigError(
                f"Monthly budget ({config.monthly_budget_limit}) is too high "
                f"compared to daily budget ({config.daily_budget_limit})"
            )
    
    # Validate directory structure
    required_subdirs = ["logs", "data", "cache", "exports"]
    for subdir in required_subdirs:
        subdir_path = config.working_dir / subdir
        try:
            subdir_path.mkdir(exist_ok=True)
        except OSError as e:
            raise LightRAGConfigError(f"Cannot create required subdirectory {subdir}: {e}")
    
    # Validate alert configuration
    if config.enable_budget_alerts:
        if not any([
            os.getenv("ALERT_EMAIL_SMTP_SERVER"),
            os.getenv("ALERT_SLACK_WEBHOOK_URL"),
            os.getenv("ALERT_WEBHOOK_URLS")
        ]):
            raise LightRAGConfigError(
                "Budget alerts enabled but no alert channels configured"
            )

# Use custom validation
config = LightRAGConfig.get_config()
validate_research_configuration(config)
```

---

## Configuration Templates

### Development Environment

```json
{
  "_comment": "Development configuration for metabolomics research",
  "api_key": "${OPENAI_API_KEY}",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "working_dir": "./dev_workspace",
  "max_async": 8,
  "max_tokens": 16384,
  
  "daily_budget_limit": 10.0,
  "monthly_budget_limit": 200.0,
  "cost_alert_threshold_percentage": 90.0,
  "enable_budget_alerts": false,
  
  "log_level": "DEBUG",
  "log_dir": "dev_logs",
  "enable_file_logging": true,
  
  "cost_db_path": "./dev_costs.db",
  "enable_research_categorization": true,
  "enable_audit_trail": false,
  "max_cost_retention_days": 30
}
```

### Production Environment

```json
{
  "_comment": "Production configuration for Clinical Metabolomics Oracle",
  "api_key": "${OPENAI_API_KEY}",
  "model": "gpt-4o",
  "embedding_model": "text-embedding-3-large",
  "working_dir": "/opt/metabolomics/lightrag",
  "max_async": 32,
  "max_tokens": 32768,
  
  "daily_budget_limit": 100.0,
  "monthly_budget_limit": 2000.0,
  "cost_alert_threshold_percentage": 75.0,
  "enable_budget_alerts": true,
  
  "log_level": "INFO",
  "log_dir": "/var/log/metabolomics",
  "enable_file_logging": true,
  "log_max_bytes": 50000000,
  "log_backup_count": 10,
  
  "cost_db_path": "/opt/metabolomics/data/cost_tracking.db",
  "enable_research_categorization": true,
  "enable_audit_trail": true,
  "max_cost_retention_days": 730
}
```

### Research Team Configuration

```json
{
  "_comment": "Shared configuration for research team",
  "api_key": "${OPENAI_API_KEY}",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "working_dir": "/shared/metabolomics/lightrag",
  "max_async": 16,
  "max_tokens": 16384,
  
  "daily_budget_limit": 150.0,
  "monthly_budget_limit": 3000.0,
  "cost_alert_threshold_percentage": 80.0,
  "enable_budget_alerts": true,
  
  "log_level": "INFO",
  "log_dir": "/shared/logs/metabolomics",
  "enable_file_logging": true,
  
  "cost_db_path": "/shared/data/team_costs.db",
  "enable_research_categorization": true,
  "enable_audit_trail": true,
  "max_cost_retention_days": 365,
  
  "_team_settings": {
    "shared_budget": true,
    "user_tracking": true,
    "department": "biochemistry",
    "project_codes": ["METAB-2025-01", "METAB-2025-02"]
  }
}
```

### Docker Configuration

```json
{
  "_comment": "Docker container configuration",
  "api_key": "${OPENAI_API_KEY}",
  "model": "${LIGHTRAG_MODEL:-gpt-4o-mini}",
  "embedding_model": "${LIGHTRAG_EMBEDDING_MODEL:-text-embedding-3-small}",
  "working_dir": "/app/data",
  "max_async": "${LIGHTRAG_MAX_ASYNC:-16}",
  "max_tokens": "${LIGHTRAG_MAX_TOKENS:-32768}",
  
  "daily_budget_limit": "${LIGHTRAG_DAILY_BUDGET_LIMIT}",
  "monthly_budget_limit": "${LIGHTRAG_MONTHLY_BUDGET_LIMIT}",
  "cost_alert_threshold_percentage": "${LIGHTRAG_COST_ALERT_THRESHOLD:-80.0}",
  "enable_budget_alerts": "${LIGHTRAG_ENABLE_BUDGET_ALERTS:-true}",
  
  "log_level": "${LIGHTRAG_LOG_LEVEL:-INFO}",
  "log_dir": "/app/logs",
  "enable_file_logging": true,
  
  "cost_db_path": "/app/data/cost_tracking.db",
  "enable_research_categorization": true,
  "enable_audit_trail": true,
  "max_cost_retention_days": 365
}
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lightrag-config
  namespace: metabolomics
data:
  config.json: |
    {
      "model": "gpt-4o-mini",
      "embedding_model": "text-embedding-3-small",
      "working_dir": "/app/data",
      "max_async": 24,
      "max_tokens": 32768,
      
      "daily_budget_limit": 200.0,
      "monthly_budget_limit": 4000.0,
      "cost_alert_threshold_percentage": 80.0,
      "enable_budget_alerts": true,
      
      "log_level": "INFO",
      "log_dir": "/app/logs",
      "enable_file_logging": true,
      
      "cost_db_path": "/app/data/cost_tracking.db",
      "enable_research_categorization": true,
      "enable_audit_trail": true,
      "max_cost_retention_days": 365
    }
---
apiVersion: v1
kind: Secret
metadata:
  name: lightrag-secrets
  namespace: metabolomics
type: Opaque
stringData:
  OPENAI_API_KEY: "your-openai-api-key-here"
  ALERT_EMAIL_PASSWORD: "your-email-password-here"
  ALERT_SLACK_WEBHOOK_URL: "your-slack-webhook-url-here"
```

### Environment-Specific Overrides

```bash
#!/bin/bash
# Environment setup script

# Base configuration
export LIGHTRAG_WORKING_DIR="/opt/metabolomics"
export LIGHTRAG_LOG_LEVEL="INFO"
export LIGHTRAG_ENABLE_COST_TRACKING="true"

# Environment-specific settings
if [ "$ENVIRONMENT" = "development" ]; then
    export LIGHTRAG_DAILY_BUDGET_LIMIT="10.0"
    export LIGHTRAG_MONTHLY_BUDGET_LIMIT="200.0"
    export LIGHTRAG_LOG_LEVEL="DEBUG"
    export LIGHTRAG_ENABLE_BUDGET_ALERTS="false"
elif [ "$ENVIRONMENT" = "staging" ]; then
    export LIGHTRAG_DAILY_BUDGET_LIMIT="50.0" 
    export LIGHTRAG_MONTHLY_BUDGET_LIMIT="1000.0"
    export LIGHTRAG_ENABLE_BUDGET_ALERTS="true"
elif [ "$ENVIRONMENT" = "production" ]; then
    export LIGHTRAG_DAILY_BUDGET_LIMIT="200.0"
    export LIGHTRAG_MONTHLY_BUDGET_LIMIT="4000.0"
    export LIGHTRAG_COST_ALERT_THRESHOLD="75.0"
    export LIGHTRAG_ENABLE_BUDGET_ALERTS="true"
    export LIGHTRAG_LOG_BACKUP_COUNT="30"
fi

# Alert configuration for production/staging
if [ "$ENVIRONMENT" != "development" ]; then
    export ALERT_EMAIL_SMTP_SERVER="smtp.university.edu"
    export ALERT_EMAIL_RECIPIENTS="research-admin@university.edu,it-support@university.edu"
    export ALERT_SLACK_CHANNEL="#${ENVIRONMENT}-alerts"
fi
```

---

This comprehensive configuration reference provides all the settings and options available in the API Cost Monitoring System. For implementation examples, see the [Developer Guide](./API_COST_MONITORING_DEVELOPER_GUIDE.md), and for practical usage, refer to the [User Guide](./API_COST_MONITORING_USER_GUIDE.md).

---

*This configuration reference is part of the Clinical Metabolomics Oracle API Cost Monitoring System documentation suite.*