# Configuration Management Guide
## Comprehensive Configuration Management for CMO-LightRAG Integration

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Variable Management](#environment-variable-management)
3. [Configuration File Structure](#configuration-file-structure)
4. [Feature Flag Management](#feature-flag-management)
5. [Environment-Specific Configurations](#environment-specific-configurations)
6. [Security Considerations](#security-considerations)
7. [Configuration Validation](#configuration-validation)
8. [Configuration Templates](#configuration-templates)
9. [Dynamic Configuration Updates](#dynamic-configuration-updates)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Deployment Strategies](#deployment-strategies)
12. [Troubleshooting and Recovery](#troubleshooting-and-recovery)

---

## Overview

The Clinical Metabolomics Oracle (CMO) LightRAG integration requires sophisticated configuration management to handle:

- **Multiple environments** (dev, staging, production)
- **Feature flag controls** for gradual rollouts
- **Security-sensitive API keys** and secrets
- **Performance tuning parameters** based on workload
- **Dynamic configuration updates** without downtime
- **Configuration validation** and error recovery

This guide provides comprehensive documentation for managing all aspects of configuration across the entire system lifecycle.

---

## Environment Variable Management

### 1. Environment Variable Categories

#### Core Integration Variables
```bash
# Master control switches
LIGHTRAG_INTEGRATION_ENABLED=true
OPENAI_API_KEY=sk-your-key-here
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true

# Model configuration
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
LIGHTRAG_MAX_TOKENS=32768
```

#### Performance and Resource Variables
```bash
# Concurrency and performance
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=30.0
LIGHTRAG_WORKING_DIR=/opt/lightrag/data

# Memory and storage management
LIGHTRAG_LOG_MAX_BYTES=10485760
LIGHTRAG_LOG_BACKUP_COUNT=5
```

#### Feature Flag Variables
```bash
# Rollout control
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0
LIGHTRAG_USER_HASH_SALT=unique_deployment_salt

# Quality and monitoring
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.7
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true

# Circuit breaker protection
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0
```

### 2. Environment Variable Loading Strategy

#### Priority Order
1. **Explicit overrides** in configuration calls
2. **Environment variables** from shell
3. **`.env` files** in order of specificity:
   - `.env.local` (highest priority)
   - `.env.{environment}` (environment-specific)
   - `.env` (default configuration)

#### Environment File Structure
```
project_root/
├── .env                    # Default configuration
├── .env.development        # Development overrides
├── .env.staging           # Staging overrides
├── .env.production        # Production overrides
├── .env.local             # Local overrides (git-ignored)
└── .env.example           # Template for new deployments
```

### 3. Environment Variable Validation

#### Validation Rules Implementation
```python
# Example validation configuration
VALIDATION_RULES = {
    'LIGHTRAG_ROLLOUT_PERCENTAGE': {
        'type': 'float',
        'range': [0.0, 100.0],
        'default': 0.0
    },
    'LIGHTRAG_MAX_ASYNC': {
        'type': 'int',
        'range': [1, 128],
        'default': 16
    },
    'LIGHTRAG_INTEGRATION_ENABLED': {
        'type': 'bool',
        'default': False
    },
    'OPENAI_API_KEY': {
        'type': 'string',
        'required': True,
        'pattern': r'^sk-[a-zA-Z0-9]{20,}$'
    }
}
```

#### Runtime Validation Example
```python
from lightrag_integration.config import LightRAGConfig

# Validate configuration on startup
try:
    config = LightRAGConfig.get_config(validate_config=True)
    print("✅ Configuration validation passed")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    # Implement fallback or error handling
```

---

## Configuration File Structure

### 1. JSON Configuration Format

#### Base Configuration Structure
```json
{
  "version": "1.0",
  "environment": "production",
  "metadata": {
    "created_at": "2025-08-08T12:00:00Z",
    "created_by": "deployment_script",
    "description": "Production configuration for CMO-LightRAG"
  },
  "core": {
    "integration_enabled": true,
    "model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "working_dir": "/opt/lightrag/production",
    "max_async": 32,
    "max_tokens": 32768,
    "timeout_seconds": 30.0
  },
  "feature_flags": {
    "rollout_percentage": 50.0,
    "enable_ab_testing": true,
    "enable_quality_metrics": true,
    "enable_circuit_breaker": true,
    "fallback_to_perplexity": true
  },
  "quality": {
    "min_quality_threshold": 0.75,
    "enable_performance_comparison": true,
    "enable_relevance_scoring": true,
    "relevance_confidence_threshold": 70.0
  },
  "circuit_breaker": {
    "failure_threshold": 5,
    "recovery_timeout": 300.0
  },
  "logging": {
    "level": "INFO",
    "enable_file_logging": true,
    "log_dir": "/var/log/lightrag",
    "max_bytes": 52428800,
    "backup_count": 10
  },
  "cost_tracking": {
    "enabled": true,
    "daily_budget_limit": 50.0,
    "monthly_budget_limit": 1000.0,
    "alert_threshold_percentage": 80.0
  },
  "routing": {
    "enable_conditional_routing": true,
    "user_hash_salt": "production_salt_v1",
    "force_user_cohort": null,
    "routing_rules": {
      "complex_queries": {
        "type": "query_length",
        "min_length": 100,
        "max_length": 2000
      },
      "clinical_queries": {
        "type": "query_type",
        "allowed_types": ["clinical", "research", "metabolomics"]
      }
    }
  }
}
```

### 2. Configuration Loading Hierarchy

#### Configuration Merger Logic
```python
def merge_configurations(base_config, environment_config, overrides):
    """
    Merge configurations with proper precedence handling.
    
    Precedence (highest to lowest):
    1. Explicit overrides
    2. Environment-specific config
    3. Base configuration
    4. Default values
    """
    merged = copy.deepcopy(base_config)
    
    # Apply environment-specific overrides
    if environment_config:
        merged = deep_merge(merged, environment_config)
    
    # Apply explicit overrides
    if overrides:
        merged = deep_merge(merged, overrides)
    
    return merged
```

### 3. Configuration Templating

#### Jinja2 Template Example
```json
{
  "version": "1.0",
  "environment": "{{ ENVIRONMENT | default('development') }}",
  "core": {
    "integration_enabled": {{ LIGHTRAG_INTEGRATION_ENABLED | default('false') | lower }},
    "model": "{{ LIGHTRAG_MODEL | default('gpt-4o-mini') }}",
    "working_dir": "{{ LIGHTRAG_WORKING_DIR | default('./lightrag_data') }}",
    "max_async": {{ LIGHTRAG_MAX_ASYNC | default(16) | int }}
  },
  "feature_flags": {
    "rollout_percentage": {{ LIGHTRAG_ROLLOUT_PERCENTAGE | default(0.0) | float }},
    "enable_ab_testing": {{ LIGHTRAG_ENABLE_AB_TESTING | default('false') | lower }}
  }
}
```

---

## Feature Flag Management

### 1. Feature Flag Architecture

#### Flag Categories
```python
# Core integration flags
CORE_FLAGS = [
    'LIGHTRAG_INTEGRATION_ENABLED',
    'LIGHTRAG_FALLBACK_TO_PERPLEXITY',
    'LIGHTRAG_ENABLE_CIRCUIT_BREAKER'
]

# Quality and performance flags
QUALITY_FLAGS = [
    'LIGHTRAG_ENABLE_QUALITY_METRICS',
    'LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON',
    'LIGHTRAG_ENABLE_RELEVANCE_SCORING'
]

# Advanced feature flags
ADVANCED_FLAGS = [
    'LIGHTRAG_ENABLE_AB_TESTING',
    'LIGHTRAG_ENABLE_CONDITIONAL_ROUTING',
    'LIGHTRAG_ENABLE_COST_TRACKING'
]
```

#### Feature Flag States
```python
class FeatureFlagState(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    ROLLOUT = "rollout"
    AB_TEST = "ab_test"
    DEPRECATED = "deprecated"
```

### 2. Gradual Rollout Strategy

#### Rollout Phases Configuration
```python
ROLLOUT_PHASES = {
    'phase_1_canary': {
        'percentage': 1.0,
        'duration_days': 3,
        'success_criteria': {
            'error_rate_threshold': 0.01,
            'quality_threshold': 0.7
        }
    },
    'phase_2_early': {
        'percentage': 5.0,
        'duration_days': 7,
        'success_criteria': {
            'error_rate_threshold': 0.02,
            'quality_threshold': 0.75
        }
    },
    'phase_3_gradual': {
        'percentage': 25.0,
        'duration_days': 14,
        'success_criteria': {
            'error_rate_threshold': 0.03,
            'quality_threshold': 0.75
        }
    },
    'phase_4_majority': {
        'percentage': 75.0,
        'duration_days': 14,
        'success_criteria': {
            'error_rate_threshold': 0.05,
            'quality_threshold': 0.7
        }
    },
    'phase_5_complete': {
        'percentage': 100.0,
        'duration_days': -1  # Permanent
    }
}
```

### 3. Feature Flag Configuration Management

#### Dynamic Flag Updates
```python
class FeatureFlagManager:
    def update_rollout_percentage(self, percentage: float, validate: bool = True):
        """Update rollout percentage with validation."""
        if validate:
            self._validate_rollout_change(self.current_percentage, percentage)
        
        # Update configuration
        self.config.lightrag_rollout_percentage = percentage
        
        # Clear caches to ensure new percentage takes effect
        self.clear_caches()
        
        # Log the change
        self.logger.info(f"Rollout updated: {self.current_percentage}% → {percentage}%")
        
        # Emit metrics
        self._emit_rollout_change_metric(percentage)
```

#### Feature Flag Monitoring
```python
def monitor_feature_flag_health():
    """Monitor feature flag health and performance."""
    manager = FeatureFlagManager(config)
    
    summary = manager.get_performance_summary()
    
    # Check rollout health
    if summary['circuit_breaker']['is_open']:
        alert("Circuit breaker is open - LightRAG degraded")
    
    # Check performance metrics
    lightrag_success_rate = summary['performance']['lightrag']['success_rate']
    if lightrag_success_rate < 0.95:
        alert(f"LightRAG success rate low: {lightrag_success_rate}")
    
    return summary
```

---

## Environment-Specific Configurations

### 1. Development Environment

#### Development Configuration Template
```bash
# Development Environment Configuration
# High observability, low performance requirements

# Core settings
OPENAI_API_KEY=sk-dev-key-here
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_WORKING_DIR=./dev_lightrag

# Performance (conservative for dev)
LIGHTRAG_MAX_ASYNC=4
LIGHTRAG_MAX_TOKENS=16384
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=60.0

# Feature flags (enable all for testing)
LIGHTRAG_ROLLOUT_PERCENTAGE=50.0
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=true

# Circuit breaker (sensitive for early detection)
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=2
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60.0

# Quality settings
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.6
LIGHTRAG_ENABLE_RELEVANCE_SCORING=true

# Logging (verbose for debugging)
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=./dev_logs
LIGHTRAG_LOG_MAX_BYTES=10485760
LIGHTRAG_LOG_BACKUP_COUNT=3

# Cost tracking (monitoring)
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=10.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=200.0

# Routing
LIGHTRAG_USER_HASH_SALT=dev_salt_2025
LIGHTRAG_ROUTING_RULES={"dev_test": {"type": "query_length", "min_length": 10}}
```

### 2. Staging Environment

#### Staging Configuration Template
```bash
# Staging Environment Configuration
# Production-like settings with enhanced monitoring

# Core settings (production-like)
OPENAI_API_KEY=sk-staging-key-here
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_WORKING_DIR=/opt/lightrag/staging

# Performance (production-like)
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_MAX_TOKENS=32768
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=30.0

# Feature flags (gradual rollout testing)
LIGHTRAG_ROLLOUT_PERCENTAGE=25.0
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true

# Circuit breaker (production-like)
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Quality settings (strict)
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.75
LIGHTRAG_ENABLE_RELEVANCE_SCORING=true

# Logging (production-level)
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=/var/log/lightrag-staging
LIGHTRAG_LOG_MAX_BYTES=52428800
LIGHTRAG_LOG_BACKUP_COUNT=5

# Cost tracking
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=25.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=500.0

# Routing
LIGHTRAG_USER_HASH_SALT=staging_salt_2025
LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=true
```

### 3. Production Environment

#### Production Configuration Template
```bash
# Production Environment Configuration
# Optimized for performance, reliability, and cost control

# Core settings (production-optimized)
OPENAI_API_KEY=sk-prod-key-here
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_MODEL=gpt-4o
LIGHTRAG_WORKING_DIR=/opt/lightrag/production

# Performance (high-performance)
LIGHTRAG_MAX_ASYNC=32
LIGHTRAG_MAX_TOKENS=32768
LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=30.0

# Feature flags (conservative rollout)
LIGHTRAG_ROLLOUT_PERCENTAGE=100.0
LIGHTRAG_ENABLE_AB_TESTING=false
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=false
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true

# Circuit breaker (production-hardened)
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Quality settings (balanced)
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.75
LIGHTRAG_ENABLE_RELEVANCE_SCORING=true

# Logging (minimal but comprehensive)
LIGHTRAG_LOG_LEVEL=WARNING
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_DIR=/var/log/lightrag
LIGHTRAG_LOG_MAX_BYTES=104857600
LIGHTRAG_LOG_BACKUP_COUNT=20

# Cost tracking (strict monitoring)
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=100.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=2000.0
LIGHTRAG_COST_ALERT_THRESHOLD=80.0
LIGHTRAG_ENABLE_BUDGET_ALERTS=true

# Routing (production-secure)
LIGHTRAG_USER_HASH_SALT=prod_salt_secure_2025
LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=true
LIGHTRAG_ROUTING_RULES={"clinical_complex": {"type": "query_length", "min_length": 200}}
```

### 4. Environment Configuration Validation

#### Environment-Specific Validation Rules
```python
ENVIRONMENT_VALIDATION = {
    'development': {
        'required_vars': ['OPENAI_API_KEY'],
        'performance_limits': {
            'max_async': 8,
            'max_tokens': 16384
        },
        'security_requirements': {
            'api_key_pattern': r'^sk-.*'  # Relaxed for dev
        }
    },
    'staging': {
        'required_vars': ['OPENAI_API_KEY'],
        'performance_limits': {
            'max_async': 24,
            'max_tokens': 32768
        },
        'security_requirements': {
            'api_key_pattern': r'^sk-[a-zA-Z0-9]{40,}$'
        }
    },
    'production': {
        'required_vars': [
            'OPENAI_API_KEY',
            'LIGHTRAG_USER_HASH_SALT',
            'LIGHTRAG_DAILY_BUDGET_LIMIT'
        ],
        'performance_limits': {
            'max_async': 64,
            'max_tokens': 32768
        },
        'security_requirements': {
            'api_key_pattern': r'^sk-[a-zA-Z0-9]{48,}$',
            'salt_min_length': 16
        }
    }
}
```

---

## Security Considerations

### 1. Secret Management

#### API Key Security
```python
class SecureConfigManager:
    def __init__(self, environment: str):
        self.environment = environment
        self.secrets_provider = self._init_secrets_provider()
    
    def _init_secrets_provider(self):
        """Initialize appropriate secrets provider for environment."""
        if self.environment == 'production':
            return AWSSecretsManager()
        elif self.environment == 'staging':
            return HashiCorpVault()
        else:
            return EnvironmentVariables()
    
    def get_api_key(self, service: str) -> str:
        """Securely retrieve API key for service."""
        key_name = f"{self.environment}/{service}/api_key"
        return self.secrets_provider.get_secret(key_name)
    
    def rotate_api_key(self, service: str, new_key: str):
        """Rotate API key with zero-downtime."""
        # Implement key rotation logic
        pass
```

#### Secrets Provider Implementations
```python
class AWSSecretsManager:
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from AWS Secrets Manager."""
        import boto3
        client = boto3.client('secretsmanager')
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']

class HashiCorpVault:
    def get_secret(self, secret_path: str) -> str:
        """Retrieve secret from HashiCorp Vault."""
        import hvac
        client = hvac.Client(url=os.getenv('VAULT_URL'))
        client.token = os.getenv('VAULT_TOKEN')
        response = client.secrets.kv.v2.read_secret_version(path=secret_path)
        return response['data']['data']['value']

class EnvironmentVariables:
    def get_secret(self, var_name: str) -> str:
        """Retrieve secret from environment variables."""
        return os.getenv(var_name.split('/')[-1].upper())
```

### 2. Configuration Security Patterns

#### Encrypted Configuration Files
```python
class EncryptedConfigLoader:
    def __init__(self, encryption_key: str):
        self.cipher = Fernet(encryption_key.encode())
    
    def load_encrypted_config(self, file_path: str) -> dict:
        """Load and decrypt configuration file."""
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def save_encrypted_config(self, config: dict, file_path: str):
        """Encrypt and save configuration file."""
        config_json = json.dumps(config, indent=2)
        encrypted_data = self.cipher.encrypt(config_json.encode())
        
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
```

### 3. Access Control and Auditing

#### Configuration Access Logging
```python
class AuditedConfigManager:
    def __init__(self, config_manager, audit_logger):
        self.config_manager = config_manager
        self.audit_logger = audit_logger
    
    def get_config(self, user_id: str, config_name: str):
        """Get configuration with audit logging."""
        self.audit_logger.log({
            'event': 'config_access',
            'user_id': user_id,
            'config_name': config_name,
            'timestamp': datetime.now().isoformat()
        })
        
        return self.config_manager.get_config(config_name)
    
    def update_config(self, user_id: str, config_name: str, changes: dict):
        """Update configuration with audit logging."""
        self.audit_logger.log({
            'event': 'config_update',
            'user_id': user_id,
            'config_name': config_name,
            'changes': changes,
            'timestamp': datetime.now().isoformat()
        })
        
        return self.config_manager.update_config(config_name, changes)
```

---

## Configuration Validation

### 1. Validation Framework

#### Comprehensive Validation System
```python
class ConfigurationValidator:
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.validators = {
            'type': self._validate_type,
            'range': self._validate_range,
            'pattern': self._validate_pattern,
            'required': self._validate_required,
            'dependencies': self._validate_dependencies
        }
    
    def validate_configuration(self, config: dict) -> ValidationResult:
        """Perform comprehensive configuration validation."""
        errors = []
        warnings = []
        
        for key, value in config.items():
            if key in self.validation_rules:
                rule = self.validation_rules[key]
                
                # Apply all relevant validators
                for validator_name, validator_func in self.validators.items():
                    if validator_name in rule:
                        try:
                            validator_func(key, value, rule[validator_name])
                        except ValidationError as e:
                            errors.append(e)
                        except ValidationWarning as e:
                            warnings.append(e)
        
        return ValidationResult(errors=errors, warnings=warnings)
```

#### Validation Rule Definitions
```python
VALIDATION_RULES = {
    'LIGHTRAG_ROLLOUT_PERCENTAGE': {
        'type': 'float',
        'range': [0.0, 100.0],
        'required': False,
        'default': 0.0,
        'description': 'Percentage of users to route to LightRAG'
    },
    'OPENAI_API_KEY': {
        'type': 'string',
        'pattern': r'^sk-[a-zA-Z0-9]{20,}$',
        'required': True,
        'secure': True,
        'description': 'OpenAI API key for LLM operations'
    },
    'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD': {
        'type': 'int',
        'range': [1, 50],
        'default': 3,
        'dependencies': ['LIGHTRAG_ENABLE_CIRCUIT_BREAKER'],
        'description': 'Number of failures before circuit breaker opens'
    }
}
```

### 2. Runtime Validation

#### Continuous Configuration Monitoring
```python
class ConfigurationMonitor:
    def __init__(self, config_manager, validator):
        self.config_manager = config_manager
        self.validator = validator
        self.monitoring_active = False
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous configuration monitoring."""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    current_config = self.config_manager.get_current_config()
                    validation_result = self.validator.validate_configuration(current_config)
                    
                    if validation_result.has_errors():
                        self._handle_validation_errors(validation_result.errors)
                    
                    if validation_result.has_warnings():
                        self._handle_validation_warnings(validation_result.warnings)
                
                except Exception as e:
                    logger.error(f"Configuration monitoring error: {e}")
                
                time.sleep(interval_seconds)
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
```

### 3. Configuration Health Checks

#### Health Check Implementation
```python
def perform_configuration_health_check() -> HealthCheckResult:
    """Comprehensive configuration health check."""
    checks = []
    
    # Check 1: Required variables present
    required_vars = ['OPENAI_API_KEY', 'LIGHTRAG_INTEGRATION_ENABLED']
    for var in required_vars:
        if not os.getenv(var):
            checks.append(HealthCheck(
                name=f"required_var_{var}",
                status="FAIL",
                message=f"Required variable {var} not set"
            ))
        else:
            checks.append(HealthCheck(
                name=f"required_var_{var}",
                status="PASS",
                message=f"Required variable {var} is set"
            ))
    
    # Check 2: API key validity
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and not api_key.startswith('sk-'):
        checks.append(HealthCheck(
            name="api_key_format",
            status="WARN",
            message="API key format appears invalid"
        ))
    
    # Check 3: Directory accessibility
    working_dir = os.getenv('LIGHTRAG_WORKING_DIR', './lightrag_data')
    try:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        checks.append(HealthCheck(
            name="working_directory",
            status="PASS",
            message=f"Working directory {working_dir} accessible"
        ))
    except Exception as e:
        checks.append(HealthCheck(
            name="working_directory",
            status="FAIL",
            message=f"Cannot access working directory: {e}"
        ))
    
    # Check 4: Feature flag consistency
    integration_enabled = os.getenv('LIGHTRAG_INTEGRATION_ENABLED', 'false').lower() == 'true'
    rollout_percentage = float(os.getenv('LIGHTRAG_ROLLOUT_PERCENTAGE', '0'))
    
    if integration_enabled and rollout_percentage == 0:
        checks.append(HealthCheck(
            name="feature_flag_consistency",
            status="WARN",
            message="Integration enabled but rollout percentage is 0"
        ))
    
    return HealthCheckResult(checks)
```

---

## Configuration Templates

### 1. Base Configuration Templates

#### Minimal Configuration Template
```json
{
  "name": "minimal_lightrag_config",
  "description": "Minimal configuration for LightRAG integration",
  "variables": {
    "OPENAI_API_KEY": {
      "required": true,
      "type": "secret",
      "description": "OpenAI API key"
    },
    "LIGHTRAG_INTEGRATION_ENABLED": {
      "default": "false",
      "type": "boolean",
      "description": "Enable LightRAG integration"
    },
    "LIGHTRAG_MODEL": {
      "default": "gpt-4o-mini",
      "type": "string",
      "description": "LLM model to use"
    }
  }
}
```

#### Full-Featured Configuration Template
```json
{
  "name": "full_lightrag_config",
  "description": "Complete configuration with all features",
  "variables": {
    "OPENAI_API_KEY": {
      "required": true,
      "type": "secret"
    },
    "LIGHTRAG_INTEGRATION_ENABLED": {
      "default": "false",
      "type": "boolean"
    },
    "LIGHTRAG_MODEL": {
      "default": "gpt-4o-mini",
      "type": "string",
      "options": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    },
    "LIGHTRAG_ROLLOUT_PERCENTAGE": {
      "default": "0.0",
      "type": "float",
      "min": 0.0,
      "max": 100.0
    },
    "LIGHTRAG_ENABLE_QUALITY_METRICS": {
      "default": "false",
      "type": "boolean"
    },
    "LIGHTRAG_MIN_QUALITY_THRESHOLD": {
      "default": "0.7",
      "type": "float",
      "min": 0.0,
      "max": 1.0
    },
    "LIGHTRAG_ENABLE_CIRCUIT_BREAKER": {
      "default": "true",
      "type": "boolean"
    },
    "LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD": {
      "default": "3",
      "type": "integer",
      "min": 1,
      "max": 20
    }
  }
}
```

### 2. Environment-Specific Templates

#### Template Generator
```python
class ConfigurationTemplateGenerator:
    def __init__(self):
        self.base_templates = self._load_base_templates()
    
    def generate_environment_config(
        self,
        environment: str,
        template_name: str,
        custom_values: dict = None
    ) -> str:
        """Generate environment-specific configuration."""
        template = self.base_templates[template_name]
        env_overrides = self._get_environment_overrides(environment)
        
        # Merge template with environment overrides and custom values
        config_values = {}
        
        for var_name, var_config in template['variables'].items():
            # Start with template default
            if 'default' in var_config:
                config_values[var_name] = var_config['default']
            
            # Apply environment-specific overrides
            if environment in env_overrides and var_name in env_overrides[environment]:
                config_values[var_name] = env_overrides[environment][var_name]
            
            # Apply custom values
            if custom_values and var_name in custom_values:
                config_values[var_name] = custom_values[var_name]
        
        return self._format_as_env_file(config_values)
    
    def _get_environment_overrides(self, environment: str) -> dict:
        """Get environment-specific configuration overrides."""
        return {
            'development': {
                'LIGHTRAG_LOG_LEVEL': 'DEBUG',
                'LIGHTRAG_MAX_ASYNC': '4',
                'LIGHTRAG_ROLLOUT_PERCENTAGE': '50.0'
            },
            'staging': {
                'LIGHTRAG_LOG_LEVEL': 'INFO',
                'LIGHTRAG_MAX_ASYNC': '16',
                'LIGHTRAG_ROLLOUT_PERCENTAGE': '25.0'
            },
            'production': {
                'LIGHTRAG_LOG_LEVEL': 'WARNING',
                'LIGHTRAG_MAX_ASYNC': '32',
                'LIGHTRAG_ROLLOUT_PERCENTAGE': '0.0'  # Start conservative
            }
        }
```

---

## Dynamic Configuration Updates

### 1. Hot Configuration Reloading

#### Configuration Update Manager
```python
class DynamicConfigurationManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.current_config = None
        self.config_watchers = []
        self.update_lock = threading.RLock()
        
        # Start file watcher
        self._start_config_watcher()
    
    def _start_config_watcher(self):
        """Start watching configuration file for changes."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ConfigChangeHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
            
            def on_modified(self, event):
                if event.src_path == str(self.manager.config_path):
                    self.manager._reload_configuration()
        
        observer = Observer()
        observer.schedule(
            ConfigChangeHandler(self),
            path=str(self.config_path.parent),
            recursive=False
        )
        observer.start()
    
    def _reload_configuration(self):
        """Reload configuration from file."""
        with self.update_lock:
            try:
                new_config = self._load_configuration()
                
                # Validate new configuration
                validation_result = self._validate_configuration(new_config)
                if validation_result.has_errors():
                    logger.error(f"Configuration validation failed: {validation_result.errors}")
                    return
                
                old_config = self.current_config
                self.current_config = new_config
                
                # Notify watchers
                self._notify_config_change(old_config, new_config)
                
                logger.info("Configuration reloaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
    
    def register_config_watcher(self, callback: Callable[[dict, dict], None]):
        """Register callback for configuration changes."""
        self.config_watchers.append(callback)
    
    def _notify_config_change(self, old_config: dict, new_config: dict):
        """Notify all registered watchers of configuration change."""
        for watcher in self.config_watchers:
            try:
                watcher(old_config, new_config)
            except Exception as e:
                logger.error(f"Configuration watcher failed: {e}")
```

### 2. Feature Flag Runtime Updates

#### Runtime Feature Flag Manager
```python
class RuntimeFeatureFlagManager:
    def __init__(self, initial_config: dict):
        self.flags = initial_config
        self.flag_history = []
        self.update_lock = threading.RLock()
    
    def update_flag(self, flag_name: str, new_value: Any, user_id: str = None):
        """Update feature flag at runtime."""
        with self.update_lock:
            old_value = self.flags.get(flag_name)
            
            # Validate flag update
            if not self._validate_flag_update(flag_name, old_value, new_value):
                raise ValueError(f"Invalid flag update: {flag_name}")
            
            # Record change in history
            self.flag_history.append({
                'flag_name': flag_name,
                'old_value': old_value,
                'new_value': new_value,
                'user_id': user_id,
                'timestamp': datetime.now()
            })
            
            # Update flag
            self.flags[flag_name] = new_value
            
            # Trigger any dependent updates
            self._handle_flag_dependencies(flag_name, new_value)
            
            logger.info(f"Feature flag updated: {flag_name} = {new_value}")
    
    def get_flag_history(self, flag_name: str = None) -> List[dict]:
        """Get history of flag changes."""
        if flag_name:
            return [h for h in self.flag_history if h['flag_name'] == flag_name]
        return self.flag_history.copy()
    
    def rollback_flag(self, flag_name: str, steps: int = 1):
        """Rollback flag to previous value."""
        flag_history = self.get_flag_history(flag_name)
        if len(flag_history) >= steps:
            previous_value = flag_history[-(steps + 1)]['old_value']
            self.update_flag(flag_name, previous_value, user_id="system_rollback")
```

### 3. Configuration Change Impact Analysis

#### Change Impact Analyzer
```python
class ConfigurationImpactAnalyzer:
    def __init__(self):
        self.impact_rules = self._load_impact_rules()
    
    def analyze_change_impact(self, old_config: dict, new_config: dict) -> ImpactAnalysis:
        """Analyze the impact of configuration changes."""
        changes = self._detect_changes(old_config, new_config)
        impacts = []
        
        for change in changes:
            impact = self._assess_change_impact(change)
            impacts.append(impact)
        
        return ImpactAnalysis(changes=changes, impacts=impacts)
    
    def _assess_change_impact(self, change: ConfigChange) -> ChangeImpact:
        """Assess the impact of a single configuration change."""
        impact_level = ImpactLevel.LOW
        affected_components = []
        required_restarts = []
        
        # Check impact rules
        if change.key in self.impact_rules:
            rule = self.impact_rules[change.key]
            impact_level = rule['impact_level']
            affected_components = rule.get('affected_components', [])
            required_restarts = rule.get('required_restarts', [])
        
        return ChangeImpact(
            change=change,
            impact_level=impact_level,
            affected_components=affected_components,
            required_restarts=required_restarts
        )
```

---

## Monitoring and Observability

### 1. Configuration Metrics

#### Configuration Metrics Collection
```python
class ConfigurationMetricsCollector:
    def __init__(self, metrics_client):
        self.metrics_client = metrics_client
        self.config_snapshot_interval = 300  # 5 minutes
    
    def collect_configuration_metrics(self, config: dict):
        """Collect and emit configuration metrics."""
        # Feature flag states
        for flag_name, flag_value in config.items():
            if flag_name.startswith('LIGHTRAG_ENABLE_'):
                self.metrics_client.gauge(
                    'lightrag.feature_flag.enabled',
                    1 if flag_value else 0,
                    tags=[f'flag:{flag_name}']
                )
        
        # Rollout percentage
        rollout_percentage = config.get('LIGHTRAG_ROLLOUT_PERCENTAGE', 0)
        self.metrics_client.gauge(
            'lightrag.rollout.percentage',
            rollout_percentage
        )
        
        # Circuit breaker settings
        if config.get('LIGHTRAG_ENABLE_CIRCUIT_BREAKER'):
            failure_threshold = config.get('LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD', 3)
            self.metrics_client.gauge(
                'lightrag.circuit_breaker.failure_threshold',
                failure_threshold
            )
        
        # Quality thresholds
        quality_threshold = config.get('LIGHTRAG_MIN_QUALITY_THRESHOLD', 0.7)
        self.metrics_client.gauge(
            'lightrag.quality.threshold',
            quality_threshold
        )
```

### 2. Configuration Health Monitoring

#### Health Check Dashboard Data
```python
def generate_configuration_health_dashboard() -> dict:
    """Generate comprehensive configuration health data."""
    config = LightRAGConfig.get_config()
    feature_flag_manager = FeatureFlagManager(config)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'environment': os.getenv('ENVIRONMENT', 'unknown'),
        'configuration_health': {
            'overall_status': 'healthy',  # healthy, warning, critical
            'last_validation': datetime.now().isoformat(),
            'validation_errors': 0,
            'validation_warnings': 1
        },
        'feature_flags': {
            'integration_enabled': config.lightrag_integration_enabled,
            'rollout_percentage': config.lightrag_rollout_percentage,
            'ab_testing_enabled': config.lightrag_enable_ab_testing,
            'circuit_breaker_enabled': config.lightrag_enable_circuit_breaker,
            'quality_metrics_enabled': config.lightrag_enable_quality_metrics
        },
        'circuit_breaker': {
            'state': 'closed',  # open, closed, half-open
            'failure_count': 0,
            'success_rate': 99.5,
            'last_failure': None
        },
        'performance': {
            'avg_response_time_ms': 250,
            'p95_response_time_ms': 500,
            'success_rate_percentage': 99.8,
            'quality_score_avg': 0.82
        },
        'cost_tracking': {
            'daily_spend': 12.50,
            'daily_budget': 50.0,
            'monthly_spend': 350.75,
            'monthly_budget': 1000.0,
            'budget_utilization_percentage': 35.1
        },
        'recent_changes': [
            {
                'timestamp': '2025-08-08T10:30:00Z',
                'changed_by': 'admin_user',
                'change_type': 'rollout_percentage',
                'old_value': '25.0',
                'new_value': '50.0'
            }
        ]
    }
```

### 3. Alerting and Notifications

#### Configuration Alert System
```python
class ConfigurationAlertManager:
    def __init__(self, alert_channels):
        self.alert_channels = alert_channels
        self.alert_rules = self._load_alert_rules()
    
    def check_configuration_alerts(self, config_health: dict):
        """Check configuration health and send alerts if needed."""
        for rule in self.alert_rules:
            if self._evaluate_alert_rule(rule, config_health):
                self._send_alert(rule, config_health)
    
    def _evaluate_alert_rule(self, rule: dict, health_data: dict) -> bool:
        """Evaluate if alert rule conditions are met."""
        condition = rule['condition']
        
        if condition['type'] == 'circuit_breaker_open':
            return health_data['circuit_breaker']['state'] == 'open'
        
        elif condition['type'] == 'success_rate_low':
            threshold = condition['threshold']
            return health_data['performance']['success_rate_percentage'] < threshold
        
        elif condition['type'] == 'budget_exceeded':
            threshold = condition['threshold']
            return health_data['cost_tracking']['budget_utilization_percentage'] > threshold
        
        return False
    
    def _send_alert(self, rule: dict, health_data: dict):
        """Send alert through configured channels."""
        alert_message = self._format_alert_message(rule, health_data)
        
        for channel in rule.get('channels', ['email']):
            if channel in self.alert_channels:
                self.alert_channels[channel].send_alert(alert_message)
```

---

## Deployment Strategies

### 1. Blue-Green Deployment Configuration

#### Blue-Green Configuration Manager
```python
class BlueGreenConfigurationManager:
    def __init__(self, config_store):
        self.config_store = config_store
        self.current_environment = 'blue'
    
    def prepare_green_environment(self, config_changes: dict):
        """Prepare green environment with new configuration."""
        # Load current blue configuration
        blue_config = self.config_store.get_config('blue')
        
        # Create green configuration with changes
        green_config = {**blue_config, **config_changes}
        
        # Validate green configuration
        validation_result = self._validate_configuration(green_config)
        if validation_result.has_errors():
            raise ConfigurationError(f"Green config validation failed: {validation_result.errors}")
        
        # Save green configuration
        self.config_store.set_config('green', green_config)
        
        return green_config
    
    def switch_to_green(self):
        """Switch traffic to green environment."""
        # Update load balancer configuration
        self._update_load_balancer('green')
        
        # Update current environment pointer
        self.current_environment = 'green'
        
        # Log the switch
        logger.info("Switched to green environment")
    
    def rollback_to_blue(self):
        """Rollback to blue environment."""
        if self.current_environment == 'green':
            self._update_load_balancer('blue')
            self.current_environment = 'blue'
            logger.info("Rolled back to blue environment")
```

### 2. Canary Deployment Configuration

#### Canary Configuration Strategy
```python
class CanaryConfigurationManager:
    def __init__(self, config_manager, traffic_manager):
        self.config_manager = config_manager
        self.traffic_manager = traffic_manager
        self.canary_phases = self._define_canary_phases()
    
    def _define_canary_phases(self):
        """Define canary deployment phases."""
        return [
            {'name': 'initial', 'traffic_percentage': 1.0, 'duration_minutes': 15},
            {'name': 'early', 'traffic_percentage': 5.0, 'duration_minutes': 30},
            {'name': 'gradual', 'traffic_percentage': 25.0, 'duration_minutes': 60},
            {'name': 'majority', 'traffic_percentage': 75.0, 'duration_minutes': 120},
            {'name': 'complete', 'traffic_percentage': 100.0, 'duration_minutes': -1}
        ]
    
    def start_canary_deployment(self, new_config: dict):
        """Start canary deployment with new configuration."""
        # Validate new configuration
        validation_result = self._validate_configuration(new_config)
        if validation_result.has_errors():
            raise ConfigurationError("Canary config validation failed")
        
        # Start with phase 1
        current_phase = self.canary_phases[0]
        
        # Update canary configuration
        self.config_manager.set_canary_config(new_config)
        
        # Route small percentage of traffic to canary
        self.traffic_manager.set_canary_traffic_percentage(
            current_phase['traffic_percentage']
        )
        
        # Schedule phase progression
        self._schedule_phase_progression()
    
    def monitor_canary_health(self) -> dict:
        """Monitor canary deployment health."""
        canary_metrics = self._collect_canary_metrics()
        
        health_status = {
            'overall_health': 'healthy',
            'error_rate': canary_metrics['error_rate'],
            'response_time_p95': canary_metrics['response_time_p95'],
            'quality_score': canary_metrics['quality_score'],
            'traffic_percentage': self.traffic_manager.get_canary_traffic_percentage()
        }
        
        # Check for issues
        if canary_metrics['error_rate'] > 0.05:  # 5% error rate threshold
            health_status['overall_health'] = 'unhealthy'
            self._trigger_canary_rollback('high_error_rate')
        
        return health_status
```

### 3. Configuration Release Pipeline

#### Automated Configuration Pipeline
```python
class ConfigurationReleasePipeline:
    def __init__(self):
        self.stages = [
            'validation',
            'security_scan',
            'integration_test',
            'staging_deploy',
            'smoke_test',
            'production_deploy'
        ]
    
    def execute_pipeline(self, config_changes: dict, target_environment: str):
        """Execute configuration release pipeline."""
        pipeline_context = {
            'config_changes': config_changes,
            'target_environment': target_environment,
            'start_time': datetime.now(),
            'stage_results': {}
        }
        
        try:
            for stage in self.stages:
                stage_result = self._execute_stage(stage, pipeline_context)
                pipeline_context['stage_results'][stage] = stage_result
                
                if not stage_result['success']:
                    raise PipelineError(f"Stage {stage} failed: {stage_result['error']}")
            
            return PipelineResult(success=True, context=pipeline_context)
        
        except Exception as e:
            self._handle_pipeline_failure(pipeline_context, str(e))
            return PipelineResult(success=False, error=str(e))
    
    def _execute_stage(self, stage: str, context: dict) -> dict:
        """Execute individual pipeline stage."""
        stage_handlers = {
            'validation': self._validate_configuration_stage,
            'security_scan': self._security_scan_stage,
            'integration_test': self._integration_test_stage,
            'staging_deploy': self._staging_deploy_stage,
            'smoke_test': self._smoke_test_stage,
            'production_deploy': self._production_deploy_stage
        }
        
        if stage in stage_handlers:
            return stage_handlers[stage](context)
        else:
            return {'success': False, 'error': f'Unknown stage: {stage}'}
```

---

## Troubleshooting and Recovery

### 1. Configuration Issue Diagnosis

#### Configuration Diagnostic Tool
```python
class ConfigurationDiagnosticTool:
    def __init__(self):
        self.diagnostic_tests = [
            self._test_environment_variables,
            self._test_file_permissions,
            self._test_api_connectivity,
            self._test_feature_flag_consistency,
            self._test_circuit_breaker_state,
            self._test_configuration_validity
        ]
    
    def run_full_diagnostic(self) -> DiagnosticReport:
        """Run complete configuration diagnostic."""
        results = []
        
        for test in self.diagnostic_tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(DiagnosticResult(
                    test_name=test.__name__,
                    status='ERROR',
                    message=f"Diagnostic test failed: {e}"
                ))
        
        return DiagnosticReport(results=results)
    
    def _test_environment_variables(self) -> DiagnosticResult:
        """Test environment variable configuration."""
        required_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return DiagnosticResult(
                test_name='environment_variables',
                status='FAIL',
                message=f"Missing required variables: {', '.join(missing_vars)}"
            )
        
        return DiagnosticResult(
            test_name='environment_variables',
            status='PASS',
            message='All required environment variables are set'
        )
    
    def _test_api_connectivity(self) -> DiagnosticResult:
        """Test API connectivity with current configuration."""
        try:
            # Test OpenAI API
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = openai_client.models.list()
            
            return DiagnosticResult(
                test_name='api_connectivity',
                status='PASS',
                message='API connectivity verified'
            )
        
        except Exception as e:
            return DiagnosticResult(
                test_name='api_connectivity',
                status='FAIL',
                message=f'API connectivity failed: {e}'
            )
```

### 2. Configuration Recovery Procedures

#### Automatic Recovery System
```python
class ConfigurationRecoverySystem:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.recovery_strategies = {
            'invalid_config': self._recover_from_invalid_config,
            'missing_secrets': self._recover_from_missing_secrets,
            'permission_error': self._recover_from_permission_error,
            'api_failure': self._recover_from_api_failure
        }
    
    def attempt_recovery(self, error_type: str, error_context: dict) -> RecoveryResult:
        """Attempt automatic recovery from configuration error."""
        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                result = strategy(error_context)
                
                if result.success:
                    logger.info(f"Successfully recovered from {error_type}")
                else:
                    logger.warning(f"Recovery attempt failed for {error_type}: {result.message}")
                
                return result
            
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                return RecoveryResult(success=False, message=str(e))
        else:
            return RecoveryResult(
                success=False,
                message=f"No recovery strategy for error type: {error_type}"
            )
    
    def _recover_from_invalid_config(self, context: dict) -> RecoveryResult:
        """Recover from invalid configuration by rolling back."""
        try:
            # Load last known good configuration
            last_good_config = self.config_manager.get_previous_config()
            if last_good_config:
                self.config_manager.restore_config(last_good_config)
                return RecoveryResult(
                    success=True,
                    message="Restored previous working configuration"
                )
            else:
                # Fall back to default configuration
                default_config = self.config_manager.get_default_config()
                self.config_manager.restore_config(default_config)
                return RecoveryResult(
                    success=True,
                    message="Restored default configuration"
                )
        except Exception as e:
            return RecoveryResult(success=False, message=str(e))
```

### 3. Emergency Configuration Procedures

#### Emergency Configuration Override
```python
class EmergencyConfigurationManager:
    def __init__(self):
        self.emergency_configs = {
            'disable_lightrag': {
                'LIGHTRAG_INTEGRATION_ENABLED': 'false',
                'LIGHTRAG_ROLLOUT_PERCENTAGE': '0.0'
            },
            'minimal_safe': {
                'LIGHTRAG_INTEGRATION_ENABLED': 'true',
                'LIGHTRAG_ROLLOUT_PERCENTAGE': '1.0',
                'LIGHTRAG_ENABLE_CIRCUIT_BREAKER': 'true',
                'LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD': '1'
            },
            'fallback_only': {
                'LIGHTRAG_INTEGRATION_ENABLED': 'false',
                'LIGHTRAG_FALLBACK_TO_PERPLEXITY': 'true'
            }
        }
    
    def activate_emergency_config(self, config_name: str) -> bool:
        """Activate emergency configuration preset."""
        if config_name not in self.emergency_configs:
            logger.error(f"Unknown emergency config: {config_name}")
            return False
        
        emergency_config = self.emergency_configs[config_name]
        
        try:
            # Apply emergency configuration
            for key, value in emergency_config.items():
                os.environ[key] = value
            
            # Force configuration reload
            self._force_configuration_reload()
            
            logger.critical(f"Emergency configuration '{config_name}' activated")
            return True
        
        except Exception as e:
            logger.error(f"Failed to activate emergency config: {e}")
            return False
    
    def get_emergency_status(self) -> dict:
        """Get current emergency configuration status."""
        return {
            'emergency_mode': self._is_emergency_mode_active(),
            'active_config': self._get_active_emergency_config(),
            'available_configs': list(self.emergency_configs.keys())
        }
```

---

## Summary

This comprehensive Configuration Management Guide provides:

1. **Complete environment variable management** with validation and security
2. **Structured configuration file formats** with templating support
3. **Advanced feature flag management** with rollout controls
4. **Environment-specific configurations** for dev, staging, and production
5. **Security considerations** including secret management and access control
6. **Comprehensive validation** with runtime monitoring
7. **Dynamic configuration updates** with impact analysis
8. **Monitoring and observability** with health checks and alerts
9. **Deployment strategies** including blue-green and canary deployments
10. **Troubleshooting and recovery** procedures for configuration issues

The guide serves as the definitive reference for managing all aspects of configuration in the CMO-LightRAG integration system, ensuring reliable, secure, and maintainable configuration management across all environments and deployment scenarios.

---

*This document is part of the Clinical Metabolomics Oracle LightRAG Integration project.*
*Last updated: August 8, 2025*
*Version: 1.0.0*