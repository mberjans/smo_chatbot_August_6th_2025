# LightRAG Configuration and Environment Setup Guide

## Comprehensive Configuration Reference for Clinical Metabolomics Oracle LightRAG Integration

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Environment Variables Reference](#environment-variables-reference)
4. [Configuration Validation](#configuration-validation)
5. [Advanced Configuration Options](#advanced-configuration-options)
6. [Configuration Management Best Practices](#configuration-management-best-practices)
7. [Setup Procedures](#setup-procedures)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides detailed documentation for configuring and setting up the LightRAG integration environment for the Clinical Metabolomics Oracle (CMO) system. It focuses on practical implementation details, validation procedures, and best practices for managing configuration across different environments.

### Key Configuration Components

- **LightRAGConfig dataclass** (`lightrag_integration/config.py`)
- **Environment variables** (.env files and system environment)
- **Feature flag system** for gradual rollouts and A/B testing
- **Validation and error handling** mechanisms
- **Cost tracking and monitoring** configurations

---

## Configuration File Structure

### 1. Primary Configuration File: `lightrag_integration/config.py`

The main configuration is managed through the `LightRAGConfig` dataclass, which provides:

```python
@dataclass
class LightRAGConfig:
    """Comprehensive configuration class for LightRAG integration."""
    
    # Core LightRAG settings
    api_key: Optional[str]
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    working_dir: Path
    graph_storage_dir: Optional[Path]
    max_async: int = 16
    max_tokens: int = 32768
    
    # Logging configuration
    log_level: str = "INFO"
    log_dir: Path
    enable_file_logging: bool = True
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Cost tracking
    enable_cost_tracking: bool = True
    daily_budget_limit: Optional[float]
    monthly_budget_limit: Optional[float]
    cost_alert_threshold_percentage: float = 80.0
    
    # Feature flags
    lightrag_integration_enabled: bool = False
    lightrag_rollout_percentage: float = 0.0
    lightrag_enable_ab_testing: bool = False
    lightrag_fallback_to_perplexity: bool = True
    lightrag_enable_circuit_breaker: bool = True
```

### 2. Configuration Factory Methods

The `LightRAGConfig` class provides several factory methods for creating configurations:

#### `LightRAGConfig.get_config()`
Primary method for creating configurations with validation and directory creation:

```python
# Load from environment variables (recommended)
config = LightRAGConfig.get_config()

# Load from JSON file
config = LightRAGConfig.get_config(source="/path/to/config.json")

# Load with custom overrides
config = LightRAGConfig.get_config(
    source={"api_key": "sk-test-key"},
    validate_config=True,
    ensure_dirs=True,
    max_async=32
)
```

#### `LightRAGConfig.from_environment()`
Create configuration from environment variables only:

```python
config = LightRAGConfig.from_environment(auto_create_dirs=True)
```

#### `LightRAGConfig.from_file()` and `LightRAGConfig.from_dict()`
Load from specific sources:

```python
# From JSON file
config = LightRAGConfig.from_file("production_config.json")

# From dictionary
config_dict = {"api_key": "sk-key", "model": "gpt-4o"}
config = LightRAGConfig.from_dict(config_dict)
```

### 3. Configuration Serialization

Configurations can be serialized for storage and debugging:

```python
# Convert to dictionary
config_dict = config.to_dict()

# Create deep copy
config_copy = config.copy()

# Secure string representation (API keys masked)
print(str(config))  # Shows "***masked***" for API keys
```

---

## Environment Variables Reference

### Core Integration Variables

| Variable | Default | Required | Description |
|----------|---------|-----------|-------------|
| `OPENAI_API_KEY` | None | **Yes** | OpenAI API key for LLM and embedding operations |
| `LIGHTRAG_MODEL` | `gpt-4o-mini` | No | LLM model for text generation |
| `LIGHTRAG_EMBEDDING_MODEL` | `text-embedding-3-small` | No | Model for generating embeddings |
| `LIGHTRAG_WORKING_DIR` | Current directory | No | Base directory for LightRAG data storage |
| `LIGHTRAG_MAX_ASYNC` | `16` | No | Maximum concurrent async operations |
| `LIGHTRAG_MAX_TOKENS` | `32768` | No | Maximum tokens per LLM request |

### Feature Flag Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_INTEGRATION_ENABLED` | `false` | Master switch for LightRAG functionality |
| `LIGHTRAG_ROLLOUT_PERCENTAGE` | `0.0` | Percentage of users routed to LightRAG (0-100) |
| `LIGHTRAG_ENABLE_AB_TESTING` | `false` | Enable A/B testing between LightRAG and fallback |
| `LIGHTRAG_FALLBACK_TO_PERPLEXITY` | `true` | Use Perplexity when LightRAG fails |
| `LIGHTRAG_FORCE_USER_COHORT` | None | Force users to specific cohort ('lightrag' or 'perplexity') |

### Performance and Reliability Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS` | `30.0` | Timeout for LightRAG operations |
| `LIGHTRAG_ENABLE_CIRCUIT_BREAKER` | `true` | Enable circuit breaker pattern |
| `LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Failures before circuit opens |
| `LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | `300.0` | Seconds before attempting recovery |
| `LIGHTRAG_ENABLE_QUALITY_METRICS` | `false` | Track and report quality metrics |
| `LIGHTRAG_MIN_QUALITY_THRESHOLD` | `0.7` | Minimum quality score (0.0-1.0) |

### Logging Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LIGHTRAG_LOG_DIR` | `logs` | Directory for log files |
| `LIGHTRAG_ENABLE_FILE_LOGGING` | `true` | Enable file-based logging |
| `LIGHTRAG_LOG_MAX_BYTES` | `10485760` | Maximum log file size (10MB) |
| `LIGHTRAG_LOG_BACKUP_COUNT` | `5` | Number of backup log files |

### Cost Tracking Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_COST_TRACKING` | `true` | Track API usage costs |
| `LIGHTRAG_DAILY_BUDGET_LIMIT` | None | Daily spending limit in USD |
| `LIGHTRAG_MONTHLY_BUDGET_LIMIT` | None | Monthly spending limit in USD |
| `LIGHTRAG_COST_ALERT_THRESHOLD` | `80.0` | Alert when budget usage exceeds percentage |
| `LIGHTRAG_ENABLE_BUDGET_ALERTS` | `true` | Send alerts for budget thresholds |
| `LIGHTRAG_COST_DB_PATH` | `cost_tracking.db` | SQLite database path for cost data |

### Relevance Scoring Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_RELEVANCE_SCORING` | `true` | Enable relevance scoring for responses |
| `LIGHTRAG_RELEVANCE_SCORING_MODE` | `comprehensive` | Scoring mode: basic, comprehensive, fast |
| `LIGHTRAG_RELEVANCE_CONFIDENCE_THRESHOLD` | `70.0` | Minimum confidence score (0-100) |
| `LIGHTRAG_RELEVANCE_MINIMUM_THRESHOLD` | `50.0` | Absolute minimum relevance score |
| `LIGHTRAG_ENABLE_PARALLEL_RELEVANCE_PROCESSING` | `true` | Process relevance scoring in parallel |

### Advanced Routing Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_USER_HASH_SALT` | `cmo_lightrag_2025` | Salt for consistent user routing |
| `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING` | `false` | Enable rule-based routing |
| `LIGHTRAG_ROUTING_RULES` | `{}` | JSON object with routing rules |
| `LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON` | `false` | Compare LightRAG vs fallback performance |

---

## Configuration Validation

### 1. Built-in Validation

The `LightRAGConfig.validate()` method performs comprehensive validation:

```python
config = LightRAGConfig.get_config()
try:
    config.validate()
    print("✅ Configuration is valid")
except LightRAGConfigError as e:
    print(f"❌ Configuration error: {e}")
```

### Validation Rules

#### Required Fields
- `OPENAI_API_KEY`: Must be present and non-empty

#### Numeric Validations
- `max_async`: Must be positive integer
- `max_tokens`: Must be positive integer
- `lightrag_rollout_percentage`: Must be between 0.0 and 100.0
- `cost_alert_threshold_percentage`: Must be between 0 and 100
- `relevance_confidence_threshold`: Must be between 0 and 100

#### String Validations
- `log_level`: Must be valid logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_filename`: Must end with '.log' extension
- `relevance_scoring_mode`: Must be 'basic', 'comprehensive', or 'fast'
- `cost_report_frequency`: Must be 'hourly', 'daily', 'weekly', or 'monthly'

#### Path Validations
- `working_dir`: Must exist or be creatable
- Derived directories are created automatically if `auto_create_dirs=True`

### 2. Custom Validation Script

Create a validation script to check configuration before deployment:

```python
#!/usr/bin/env python3
"""Configuration validation script for deployment."""

import sys
import os
from pathlib import Path
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError

def validate_configuration():
    """Validate current configuration."""
    try:
        # Load configuration
        config = LightRAGConfig.get_config(validate_config=True)
        
        print("✅ Configuration loaded successfully")
        print(f"   Model: {config.model}")
        print(f"   Integration enabled: {config.lightrag_integration_enabled}")
        print(f"   Rollout percentage: {config.lightrag_rollout_percentage}%")
        
        # Additional custom validations
        validate_environment_specific(config)
        validate_security_requirements(config)
        validate_performance_settings(config)
        
        return True
        
    except LightRAGConfigError as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        return False

def validate_environment_specific(config):
    """Validate environment-specific requirements."""
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'production':
        # Production-specific validations
        if config.lightrag_rollout_percentage > 0 and not config.daily_budget_limit:
            raise ValueError("Production rollout requires daily budget limit")
        
        if not config.lightrag_enable_circuit_breaker:
            raise ValueError("Circuit breaker required in production")
            
    elif environment == 'staging':
        # Staging-specific validations
        if config.lightrag_rollout_percentage > 50.0:
            print("⚠️  Warning: High rollout percentage in staging")

def validate_security_requirements(config):
    """Validate security-related configuration."""
    if config.api_key and not config.api_key.startswith('sk-'):
        raise ValueError("Invalid API key format")
    
    if config.lightrag_user_hash_salt == 'cmo_lightrag_2025':
        print("⚠️  Warning: Using default hash salt (change for production)")

def validate_performance_settings(config):
    """Validate performance-related settings."""
    if config.max_async > 32:
        print(f"⚠️  Warning: High max_async setting ({config.max_async})")
    
    if config.max_tokens > 32768:
        print(f"⚠️  Warning: High max_tokens setting ({config.max_tokens})")

if __name__ == "__main__":
    success = validate_configuration()
    sys.exit(0 if success else 1)
```

### 3. Environment-Specific Validation

Create separate validation profiles for different environments:

```python
# validation_profiles.py
VALIDATION_PROFILES = {
    'development': {
        'required_vars': ['OPENAI_API_KEY'],
        'warnings': {
            'max_async_threshold': 8,
            'rollout_percentage_max': 100.0
        }
    },
    'staging': {
        'required_vars': ['OPENAI_API_KEY'],
        'warnings': {
            'max_async_threshold': 16,
            'rollout_percentage_max': 50.0
        },
        'errors': {
            'circuit_breaker_required': True
        }
    },
    'production': {
        'required_vars': [
            'OPENAI_API_KEY',
            'LIGHTRAG_DAILY_BUDGET_LIMIT',
            'LIGHTRAG_USER_HASH_SALT'
        ],
        'warnings': {
            'max_async_threshold': 32,
            'rollout_percentage_max': 10.0  # Conservative rollout
        },
        'errors': {
            'circuit_breaker_required': True,
            'cost_tracking_required': True,
            'budget_limit_required': True
        }
    }
}
```

---

## Advanced Configuration Options

### 1. LightRAG-Specific Parameters

#### Model Configuration
```python
# Model selection for different use cases
LIGHTRAG_MODEL_CONFIGS = {
    'development': 'gpt-4o-mini',      # Cost-effective for testing
    'staging': 'gpt-4o-mini',          # Match production model
    'production_fast': 'gpt-4o-mini',   # Fast responses
    'production_quality': 'gpt-4o'     # High quality responses
}

# Embedding model optimization
EMBEDDING_MODEL_CONFIGS = {
    'small': 'text-embedding-3-small',    # 1536 dimensions, faster
    'large': 'text-embedding-3-large'     # 3072 dimensions, more accurate
}
```

#### Performance Tuning
```python
# Performance optimization based on workload
PERFORMANCE_PROFILES = {
    'low_latency': {
        'max_async': 32,
        'timeout_seconds': 15.0,
        'model': 'gpt-4o-mini'
    },
    'high_throughput': {
        'max_async': 64,
        'timeout_seconds': 45.0,
        'model': 'gpt-4o-mini'
    },
    'high_quality': {
        'max_async': 16,
        'timeout_seconds': 60.0,
        'model': 'gpt-4o'
    },
    'cost_optimized': {
        'max_async': 8,
        'timeout_seconds': 30.0,
        'model': 'gpt-4o-mini'
    }
}
```

### 2. A/B Testing Configuration

#### A/B Testing Setup
```python
# A/B testing configuration
AB_TESTING_CONFIG = {
    'enabled': True,
    'test_name': 'lightrag_vs_perplexity_v1',
    'traffic_split': {
        'lightrag': 50.0,      # 50% to LightRAG
        'perplexity': 50.0     # 50% to Perplexity fallback
    },
    'success_metrics': [
        'response_quality_score',
        'user_satisfaction_rating',
        'response_time_ms'
    ],
    'minimum_sample_size': 1000,
    'test_duration_days': 14
}

# Environment variables for A/B testing
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_AB_TEST_NAME=lightrag_vs_perplexity_v1
LIGHTRAG_AB_TEST_LIGHTRAG_PERCENTAGE=50.0
```

#### User Cohort Assignment
```python
# Consistent user assignment based on hash
import hashlib

def assign_user_cohort(user_id: str, salt: str, rollout_percentage: float) -> str:
    """Assign user to cohort based on consistent hash."""
    user_hash = hashlib.md5(f"{user_id}{salt}".encode()).hexdigest()
    hash_int = int(user_hash[:8], 16)
    percentage = (hash_int % 100) + 1
    
    if percentage <= rollout_percentage:
        return 'lightrag'
    else:
        return 'perplexity'

# Configuration for cohort assignment
LIGHTRAG_USER_HASH_SALT=your_deployment_specific_salt_here
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0
```

### 3. Circuit Breaker Configuration

#### Circuit Breaker States and Transitions
```python
CIRCUIT_BREAKER_CONFIG = {
    'failure_threshold': 3,        # Failures before opening
    'recovery_timeout': 300.0,     # Seconds before attempting recovery
    'success_threshold': 2,        # Successes needed to close circuit
    'monitoring_window': 60.0,     # Window for counting failures
    'half_open_max_calls': 5       # Max calls in half-open state
}

# Environment variables
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0
```

#### Circuit Breaker Monitoring
```python
def monitor_circuit_breaker_health(config):
    """Monitor circuit breaker health and adjust thresholds."""
    from lightrag_integration.clinical_metabolomics_rag import FeatureFlagManager
    
    manager = FeatureFlagManager(config)
    circuit_state = manager.get_circuit_breaker_state()
    
    alerts = []
    if circuit_state['is_open']:
        alerts.append({
            'level': 'critical',
            'message': 'LightRAG circuit breaker is OPEN',
            'failure_count': circuit_state['failure_count']
        })
    
    return {
        'circuit_state': circuit_state,
        'alerts': alerts,
        'recommendations': generate_circuit_breaker_recommendations(circuit_state)
    }
```

### 4. Cost Management Configuration

#### Budget Tracking and Alerting
```python
COST_MANAGEMENT_CONFIG = {
    'tracking_enabled': True,
    'daily_budget_usd': 50.0,
    'monthly_budget_usd': 1000.0,
    'alert_thresholds': {
        'warning': 70.0,    # 70% of budget
        'critical': 90.0    # 90% of budget
    },
    'cost_breakdown': {
        'track_by_model': True,
        'track_by_operation': True,
        'track_by_user_cohort': True
    }
}

# Environment variables for cost tracking
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=1000.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
LIGHTRAG_ENABLE_BUDGET_ALERTS=true
```

#### Cost Optimization Strategies
```python
COST_OPTIMIZATION_STRATEGIES = {
    'model_selection': {
        'use_gpt4o_mini_for_simple_queries': True,
        'use_gpt4o_for_complex_queries': True,
        'complexity_threshold': 100  # characters
    },
    'caching': {
        'enable_response_caching': True,
        'cache_ttl_hours': 24,
        'cache_similar_queries': True,
        'similarity_threshold': 0.95
    },
    'rate_limiting': {
        'max_requests_per_user_per_hour': 100,
        'max_concurrent_requests': 10
    }
}
```

---

## Configuration Management Best Practices

### 1. Environment Separation

#### Directory Structure
```
project_root/
├── config/
│   ├── development.json
│   ├── staging.json
│   ├── production.json
│   └── local.json.example
├── .env.development
├── .env.staging
├── .env.production
└── .env.local.example
```

#### Environment-Specific Configuration Loading
```python
import os
from pathlib import Path

def load_environment_config(environment: str = None):
    """Load configuration for specific environment."""
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    # Load base configuration
    config_file = Path(f"config/{environment}.json")
    if config_file.exists():
        config = LightRAGConfig.from_file(config_file)
    else:
        config = LightRAGConfig.from_environment()
    
    # Apply environment-specific validations
    validate_environment_config(config, environment)
    
    return config
```

### 2. Secrets Management

#### External Secrets Integration
```python
class SecretsManager:
    """Manage secrets from external providers."""
    
    def __init__(self, provider: str = 'environment'):
        self.provider = provider
    
    def get_secret(self, key: str) -> str:
        """Retrieve secret from configured provider."""
        if self.provider == 'aws_secrets_manager':
            return self._get_from_aws_secrets(key)
        elif self.provider == 'azure_key_vault':
            return self._get_from_azure_vault(key)
        elif self.provider == 'hashicorp_vault':
            return self._get_from_vault(key)
        else:
            return os.getenv(key)
    
    def _get_from_aws_secrets(self, key: str) -> str:
        """Get secret from AWS Secrets Manager."""
        import boto3
        client = boto3.client('secretsmanager')
        response = client.get_secret_value(SecretId=key)
        return response['SecretString']
```

#### API Key Rotation
```python
def rotate_api_key(old_key: str, new_key: str):
    """Rotate API key with zero downtime."""
    # Test new key validity
    test_config = LightRAGConfig.get_config()
    test_config.api_key = new_key
    
    try:
        test_config.validate()
        # Test API connectivity
        test_api_connectivity(test_config)
        
        # Update configuration atomically
        update_api_key_atomic(new_key)
        
        print("✅ API key rotation completed successfully")
        
    except Exception as e:
        print(f"❌ API key rotation failed: {e}")
        raise
```

### 3. Configuration Versioning

#### Version Control Integration
```python
CONFIG_VERSION_INFO = {
    'version': '1.2.3',
    'git_commit': 'abc123def456',
    'deployment_timestamp': '2025-08-08T12:00:00Z',
    'environment': 'production',
    'configuration_checksum': 'sha256:...'
}

def track_configuration_changes():
    """Track configuration changes in version control."""
    import git
    
    repo = git.Repo('.')
    current_commit = repo.head.commit.hexsha
    
    config_data = {
        'version': get_config_version(),
        'git_commit': current_commit,
        'timestamp': datetime.now().isoformat(),
        'changes': get_config_changes_since_last_deployment()
    }
    
    return config_data
```

### 4. Configuration Testing

#### Automated Configuration Tests
```python
import pytest
from lightrag_integration.config import LightRAGConfig

class TestConfigurationValidation:
    """Test suite for configuration validation."""
    
    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config_dict = {
            'api_key': 'sk-test-key-for-testing',
            'model': 'gpt-4o-mini'
        }
        config = LightRAGConfig.from_dict(config_dict)
        config.validate()  # Should not raise
    
    def test_invalid_api_key(self):
        """Test invalid API key validation."""
        config_dict = {'api_key': 'invalid-key'}
        config = LightRAGConfig.from_dict(config_dict)
        
        with pytest.raises(LightRAGConfigError, match="API key"):
            config.validate()
    
    def test_rollout_percentage_bounds(self):
        """Test rollout percentage validation."""
        # Test valid percentage
        config = LightRAGConfig(
            api_key='sk-test', 
            lightrag_rollout_percentage=50.0
        )
        config.validate()
        
        # Test invalid percentage
        config.lightrag_rollout_percentage = 150.0
        config.__post_init__()  # Re-run validation
        assert config.lightrag_rollout_percentage == 100.0  # Should be clamped
    
    @pytest.mark.parametrize('log_level', ['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    def test_valid_log_levels(self, log_level):
        """Test valid log level values."""
        config = LightRAGConfig(api_key='sk-test', log_level=log_level)
        config.validate()  # Should not raise
    
    def test_directory_creation(self, tmp_path):
        """Test automatic directory creation."""
        working_dir = tmp_path / "test_lightrag"
        config = LightRAGConfig(
            api_key='sk-test',
            working_dir=working_dir,
            auto_create_dirs=True
        )
        
        assert working_dir.exists()
        assert config.graph_storage_dir.exists()
```

---

## Setup Procedures

### 1. Initial Environment Setup

#### Step 1: Clone and Setup Repository
```bash
# Clone the repository
git clone <repository_url>
cd smo_chatbot_August_6th_2025

# Create Python virtual environment
python -m venv lightrag_env
source lightrag_env/bin/activate  # On Windows: lightrag_env\Scripts\activate

# Install dependencies
pip install -r requirements_lightrag.txt
```

#### Step 2: Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (replace with your values)
nano .env
```

#### Step 3: Basic Environment Variables
```bash
# Required variables
OPENAI_API_KEY=sk-your-openai-api-key-here

# Basic LightRAG settings
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_WORKING_DIR=./lightrag_storage

# Logging
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_LOG_DIR=./logs

# Feature flags (start conservative)
LIGHTRAG_ROLLOUT_PERCENTAGE=1.0
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
```

### 2. Development Environment Setup

#### Development-Specific Configuration
```bash
# Development environment
ENVIRONMENT=development
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_MAX_ASYNC=4
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=60.0

# Enable all features for testing
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true

# Lenient circuit breaker for development
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60.0

# Cost tracking (low limits for development)
LIGHTRAG_DAILY_BUDGET_LIMIT=10.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=200.0
```

#### Development Testing Script
```python
#!/usr/bin/env python3
"""Development environment test script."""

import asyncio
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG

async def test_development_setup():
    """Test development environment setup."""
    try:
        # Load and validate configuration
        config = LightRAGConfig.get_config(validate_config=True)
        print(f"✅ Configuration loaded: {config.model}")
        
        # Test LightRAG initialization
        rag = ClinicalMetabolomicsRAG(config)
        print("✅ LightRAG system initialized")
        
        # Test basic query
        test_query = "What is clinical metabolomics?"
        response = await rag.query(test_query)
        print(f"✅ Test query successful: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Development setup test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_development_setup())
    exit(0 if success else 1)
```

### 3. Production Environment Setup

#### Production Configuration Checklist
- [ ] Valid OpenAI API key configured
- [ ] Appropriate budget limits set
- [ ] Circuit breaker enabled
- [ ] Conservative rollout percentage (start with 1-5%)
- [ ] Proper logging configuration
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested

#### Production Environment Variables
```bash
# Production environment
ENVIRONMENT=production
LIGHTRAG_LOG_LEVEL=WARNING
LIGHTRAG_MAX_ASYNC=32
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=30.0

# Conservative feature flags
LIGHTRAG_ROLLOUT_PERCENTAGE=1.0  # Start very low
LIGHTRAG_ENABLE_AB_TESTING=false
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=false

# Production-hardened circuit breaker
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Production cost limits
LIGHTRAG_DAILY_BUDGET_LIMIT=100.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=2000.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
LIGHTRAG_ENABLE_BUDGET_ALERTS=true

# Security
LIGHTRAG_USER_HASH_SALT=production_secure_salt_2025
```

### 4. Configuration Deployment Script

#### Automated Deployment
```python
#!/usr/bin/env python3
"""Configuration deployment script."""

import sys
import os
import json
from pathlib import Path
from lightrag_integration.config import LightRAGConfig

def deploy_configuration(environment: str, config_file: str):
    """Deploy configuration to specified environment."""
    try:
        # Validate environment
        if environment not in ['development', 'staging', 'production']:
            raise ValueError(f"Invalid environment: {environment}")
        
        # Load configuration
        if Path(config_file).exists():
            config = LightRAGConfig.from_file(config_file)
        else:
            config = LightRAGConfig.from_environment()
        
        # Environment-specific validations
        validate_deployment_config(config, environment)
        
        # Create deployment info
        deployment_info = create_deployment_info(config, environment)
        
        # Deploy configuration
        deploy_config_to_environment(config, environment)
        
        # Save deployment record
        save_deployment_record(deployment_info)
        
        print(f"✅ Configuration deployed successfully to {environment}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration deployment failed: {e}")
        return False

def validate_deployment_config(config, environment):
    """Validate configuration for deployment."""
    # Basic validation
    config.validate()
    
    # Environment-specific checks
    if environment == 'production':
        if config.lightrag_rollout_percentage > 10.0:
            raise ValueError("Production rollout percentage too high for initial deployment")
        
        if not config.daily_budget_limit:
            raise ValueError("Daily budget limit required for production")
        
        if not config.lightrag_enable_circuit_breaker:
            raise ValueError("Circuit breaker required for production")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: deploy_config.py <environment> [config_file]")
        sys.exit(1)
    
    environment = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = deploy_configuration(environment, config_file)
    sys.exit(0 if success else 1)
```

---

## Troubleshooting

### 1. Common Configuration Issues

#### Issue: "API key is required and cannot be empty"
**Cause**: OPENAI_API_KEY environment variable not set or empty
**Solution**:
```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Set API key (replace with your actual key)
export OPENAI_API_KEY=sk-your-api-key-here

# Or add to .env file
echo "OPENAI_API_KEY=sk-your-api-key-here" >> .env
```

#### Issue: "Working directory does not exist and cannot be created"
**Cause**: Insufficient permissions or invalid path
**Solution**:
```bash
# Check current permissions
ls -la $(dirname $LIGHTRAG_WORKING_DIR)

# Create directory manually
mkdir -p $LIGHTRAG_WORKING_DIR

# Fix permissions if needed
chmod 755 $LIGHTRAG_WORKING_DIR
```

#### Issue: "lightrag_rollout_percentage must be between 0 and 100"
**Cause**: Invalid rollout percentage value
**Solution**:
```bash
# Check current value
echo $LIGHTRAG_ROLLOUT_PERCENTAGE

# Set valid value
export LIGHTRAG_ROLLOUT_PERCENTAGE=25.0
```

### 2. Configuration Debugging

#### Debug Configuration Loading
```python
#!/usr/bin/env python3
"""Configuration debugging script."""

import os
import traceback
from lightrag_integration.config import LightRAGConfig

def debug_configuration():
    """Debug configuration loading issues."""
    print("=== Configuration Debug Information ===")
    
    # Environment variables
    print("\n1. Environment Variables:")
    env_vars = [var for var in os.environ.keys() if 'LIGHTRAG' in var or 'OPENAI' in var]
    for var in sorted(env_vars):
        value = os.environ[var]
        if 'API_KEY' in var:
            value = f"{value[:8]}..." if len(value) > 8 else "***"
        print(f"   {var}={value}")
    
    # Configuration loading
    print("\n2. Configuration Loading:")
    try:
        config = LightRAGConfig.get_config(validate_config=False)
        print("   ✅ Configuration loaded successfully")
        
        print(f"   Model: {config.model}")
        print(f"   Working dir: {config.working_dir}")
        print(f"   Integration enabled: {config.lightrag_integration_enabled}")
        print(f"   Rollout percentage: {config.lightrag_rollout_percentage}")
        
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        traceback.print_exc()
    
    # Validation
    print("\n3. Configuration Validation:")
    try:
        config = LightRAGConfig.get_config(validate_config=True)
        print("   ✅ Configuration validation passed")
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
    
    # Directory accessibility
    print("\n4. Directory Accessibility:")
    try:
        config = LightRAGConfig.get_config(validate_config=False)
        working_dir = config.working_dir
        
        print(f"   Working directory: {working_dir}")
        print(f"   Exists: {working_dir.exists()}")
        print(f"   Is directory: {working_dir.is_dir() if working_dir.exists() else 'N/A'}")
        print(f"   Writable: {os.access(working_dir, os.W_OK) if working_dir.exists() else 'N/A'}")
        
    except Exception as e:
        print(f"   ❌ Directory check failed: {e}")

if __name__ == "__main__":
    debug_configuration()
```

#### Health Check Script
```python
#!/usr/bin/env python3
"""Configuration health check script."""

import asyncio
import sys
from lightrag_integration.config import LightRAGConfig

async def health_check():
    """Perform comprehensive configuration health check."""
    checks = []
    
    # Configuration loading
    try:
        config = LightRAGConfig.get_config(validate_config=True)
        checks.append(("Configuration Loading", "PASS", "Configuration loaded and validated"))
    except Exception as e:
        checks.append(("Configuration Loading", "FAIL", str(e)))
        return checks
    
    # API connectivity
    try:
        # Test OpenAI API (replace with actual test)
        # This is a placeholder - implement actual API test
        checks.append(("API Connectivity", "PASS", "API accessible"))
    except Exception as e:
        checks.append(("API Connectivity", "FAIL", str(e)))
    
    # Feature flags consistency
    try:
        if config.lightrag_integration_enabled and config.lightrag_rollout_percentage == 0:
            checks.append(("Feature Flags", "WARN", "Integration enabled but rollout is 0%"))
        else:
            checks.append(("Feature Flags", "PASS", "Feature flags consistent"))
    except Exception as e:
        checks.append(("Feature Flags", "FAIL", str(e)))
    
    # Budget configuration
    try:
        if config.lightrag_rollout_percentage > 0 and not config.daily_budget_limit:
            checks.append(("Budget Configuration", "WARN", "Rollout enabled without budget limit"))
        else:
            checks.append(("Budget Configuration", "PASS", "Budget configuration valid"))
    except Exception as e:
        checks.append(("Budget Configuration", "FAIL", str(e)))
    
    return checks

def print_health_check_results(checks):
    """Print health check results."""
    print("=== Configuration Health Check ===")
    
    passed = failed = warned = 0
    
    for check_name, status, message in checks:
        status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[status]
        print(f"{status_icon} {check_name}: {message}")
        
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            warned += 1
    
    print(f"\nSummary: {passed} passed, {warned} warnings, {failed} failed")
    
    return failed == 0

async def main():
    checks = await health_check()
    success = print_health_check_results(checks)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Emergency Configuration Recovery

#### Emergency Disable Script
```python
#!/usr/bin/env python3
"""Emergency configuration disable script."""

import os
import sys

def emergency_disable():
    """Disable LightRAG integration immediately."""
    try:
        # Set emergency environment variables
        os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'false'
        os.environ['LIGHTRAG_ROLLOUT_PERCENTAGE'] = '0.0'
        
        # Update .env file if it exists
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            with open(env_file, 'w') as f:
                for line in lines:
                    if line.startswith('LIGHTRAG_INTEGRATION_ENABLED'):
                        f.write('LIGHTRAG_INTEGRATION_ENABLED=false\n')
                    elif line.startswith('LIGHTRAG_ROLLOUT_PERCENTAGE'):
                        f.write('LIGHTRAG_ROLLOUT_PERCENTAGE=0.0\n')
                    else:
                        f.write(line)
        
        print("✅ Emergency disable completed")
        return True
        
    except Exception as e:
        print(f"❌ Emergency disable failed: {e}")
        return False

if __name__ == "__main__":
    success = emergency_disable()
    sys.exit(0 if success else 1)
```

---

This comprehensive configuration setup guide provides detailed documentation for managing the LightRAG integration configuration, including validation procedures, environment-specific setups, troubleshooting, and emergency procedures. The guide focuses on practical implementation details that complement the existing high-level configuration management documentation.