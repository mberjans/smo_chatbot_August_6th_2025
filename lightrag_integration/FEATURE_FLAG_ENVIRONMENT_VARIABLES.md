# LightRAG Feature Flag Environment Variables

This document provides a comprehensive reference for all environment variables used by the LightRAG feature flag system. These variables control the behavior of the LightRAG integration, rollout management, and routing decisions.

## Table of Contents

1. [Core Integration Variables](#core-integration-variables)
2. [Rollout Management Variables](#rollout-management-variables)
3. [Quality and Performance Variables](#quality-and-performance-variables)
4. [Circuit Breaker Variables](#circuit-breaker-variables)
5. [Routing and Conditional Logic Variables](#routing-and-conditional-logic-variables)
6. [A/B Testing Variables](#ab-testing-variables)
7. [Monitoring and Logging Variables](#monitoring-and-logging-variables)
8. [Advanced Configuration Variables](#advanced-configuration-variables)
9. [Example Configurations](#example-configurations)
10. [Migration Guide](#migration-guide)

---

## Core Integration Variables

### `LIGHTRAG_INTEGRATION_ENABLED`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Master switch to enable/disable LightRAG integration
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_INTEGRATION_ENABLED=true`

### `LIGHTRAG_FALLBACK_TO_PERPLEXITY`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable fallback to Perplexity API when LightRAG fails
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_FALLBACK_TO_PERPLEXITY=true`

### `LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS`
- **Type**: Float
- **Default**: `30.0`
- **Description**: Timeout in seconds for LightRAG API calls
- **Range**: `1.0` to `300.0`
- **Example**: `LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS=30.0`

---

## Rollout Management Variables

### `LIGHTRAG_ROLLOUT_PERCENTAGE`
- **Type**: Float
- **Default**: `0.0`
- **Description**: Percentage of users to route to LightRAG (0-100)
- **Range**: `0.0` to `100.0`
- **Example**: `LIGHTRAG_ROLLOUT_PERCENTAGE=25.0`

### `LIGHTRAG_USER_HASH_SALT`
- **Type**: String
- **Default**: `"cmo_lightrag_2025"`
- **Description**: Salt value for consistent user hash-based routing
- **Security**: Should be unique per deployment for security
- **Example**: `LIGHTRAG_USER_HASH_SALT="your_unique_salt_2025"`

### `LIGHTRAG_FORCE_USER_COHORT`
- **Type**: String (Optional)
- **Default**: `None`
- **Description**: Force all users into specific cohort (overrides percentage)
- **Values**: `"lightrag"`, `"perplexity"`, or empty/unset
- **Example**: `LIGHTRAG_FORCE_USER_COHORT=lightrag`

---

## Quality and Performance Variables

### `LIGHTRAG_ENABLE_QUALITY_METRICS`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable quality assessment and metrics collection
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_ENABLE_QUALITY_METRICS=true`

### `LIGHTRAG_MIN_QUALITY_THRESHOLD`
- **Type**: Float
- **Default**: `0.7`
- **Description**: Minimum quality score required (0.0-1.0)
- **Range**: `0.0` to `1.0`
- **Example**: `LIGHTRAG_MIN_QUALITY_THRESHOLD=0.75`

### `LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable performance comparison between LightRAG and Perplexity
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true`

---

## Circuit Breaker Variables

### `LIGHTRAG_ENABLE_CIRCUIT_BREAKER`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable circuit breaker protection for LightRAG
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true`

### `LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD`
- **Type**: Integer
- **Default**: `3`
- **Description**: Number of consecutive failures before opening circuit
- **Range**: `1` to `20`
- **Example**: `LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5`

### `LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT`
- **Type**: Float
- **Default**: `300.0`
- **Description**: Seconds to wait before attempting recovery
- **Range**: `60.0` to `3600.0`
- **Example**: `LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=600.0`

---

## Routing and Conditional Logic Variables

### `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable conditional routing based on query characteristics
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=true`

### `LIGHTRAG_ROUTING_RULES`
- **Type**: JSON String (Optional)
- **Default**: `"{}"`
- **Description**: JSON configuration for conditional routing rules
- **Format**: Valid JSON object with routing rule definitions
- **Example**: 
```bash
LIGHTRAG_ROUTING_RULES='{"long_queries": {"type": "query_length", "min_length": 100, "max_length": 1000}}'
```

---

## A/B Testing Variables

### `LIGHTRAG_ENABLE_AB_TESTING`
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable A/B testing mode with cohort tracking
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `t`, `f`, `on`, `off`
- **Example**: `LIGHTRAG_ENABLE_AB_TESTING=true`

---

## Monitoring and Logging Variables

All standard LightRAG logging variables are supported (see [LIGHTRAG_CONFIG_REFERENCE.md](../docs/LIGHTRAG_CONFIG_REFERENCE.md) for details):

- `LIGHTRAG_LOG_LEVEL`
- `LIGHTRAG_LOG_DIR`
- `LIGHTRAG_ENABLE_FILE_LOGGING`
- `LIGHTRAG_LOG_MAX_BYTES`
- `LIGHTRAG_LOG_BACKUP_COUNT`

---

## Advanced Configuration Variables

### Existing LightRAG Variables
All existing LightRAG configuration variables remain supported:
- `OPENAI_API_KEY` (required)
- `LIGHTRAG_MODEL`
- `LIGHTRAG_EMBEDDING_MODEL`
- `LIGHTRAG_WORKING_DIR`
- `LIGHTRAG_MAX_ASYNC`
- `LIGHTRAG_MAX_TOKENS`

### Cost Tracking Variables
All existing cost tracking variables remain supported:
- `LIGHTRAG_ENABLE_COST_TRACKING`
- `LIGHTRAG_DAILY_BUDGET_LIMIT`
- `LIGHTRAG_MONTHLY_BUDGET_LIMIT`
- `LIGHTRAG_COST_ALERT_THRESHOLD`
- `LIGHTRAG_ENABLE_BUDGET_ALERTS`

---

## Example Configurations

### Development Environment
```bash
# Basic development setup with LightRAG testing
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API=your_perplexity_api_key

# Enable LightRAG with small rollout
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=5.0
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true

# Enable quality metrics for testing
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.6

# Conservative circuit breaker settings
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=2
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Enable performance comparison
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true

# Logging
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_FILE_LOGGING=true
```

### Production Canary Deployment
```bash
# Production API keys
OPENAI_API_KEY=your_production_openai_key
PERPLEXITY_API=your_production_perplexity_key

# Canary rollout (1% of users)
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=1.0
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
LIGHTRAG_USER_HASH_SALT=your_unique_production_salt

# Strict quality requirements
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.8

# Aggressive circuit breaker for production safety
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=600.0

# A/B testing enabled
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true

# Production logging
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_MAX_BYTES=52428800
LIGHTRAG_LOG_BACKUP_COUNT=10
```

### A/B Testing Configuration
```bash
# A/B testing setup
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API=your_perplexity_api_key

# 50% rollout with A/B testing
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=50.0
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_USER_HASH_SALT=ab_test_salt_v1

# Quality metrics for comparison
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.7

# Moderate circuit breaker settings
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Conditional routing for complex queries
LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=true
LIGHTRAG_ROUTING_RULES='{"complex_queries": {"type": "query_length", "min_length": 200}}'
```

### Full Production Rollout
```bash
# Full production deployment
OPENAI_API_KEY=your_production_openai_key
PERPLEXITY_API=your_production_perplexity_key

# 100% rollout
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=100.0
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true
LIGHTRAG_USER_HASH_SALT=production_salt_v1

# Quality monitoring
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.75
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=false  # Disable after rollout

# Production circuit breaker
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0

# Advanced routing
LIGHTRAG_ENABLE_CONDITIONAL_ROUTING=true

# Production logging and monitoring
LIGHTRAG_LOG_LEVEL=WARNING
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_ENABLE_COST_TRACKING=true
LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
LIGHTRAG_MONTHLY_BUDGET_LIMIT=1000.0
```

### Disabled/Fallback Configuration
```bash
# Disable LightRAG completely (fallback to Perplexity only)
PERPLEXITY_API=your_perplexity_api_key

# LightRAG disabled
LIGHTRAG_INTEGRATION_ENABLED=false

# All other LightRAG variables ignored when disabled
```

---

## Migration Guide

### From Perplexity-Only to LightRAG Integration

1. **Add Required Variables**:
   ```bash
   # Add to existing .env file
   OPENAI_API_KEY=your_openai_api_key
   LIGHTRAG_INTEGRATION_ENABLED=true
   LIGHTRAG_ROLLOUT_PERCENTAGE=5.0  # Start small
   ```

2. **Enable Gradual Rollout**:
   ```bash
   # Week 1: 5% rollout
   LIGHTRAG_ROLLOUT_PERCENTAGE=5.0
   
   # Week 2: 15% rollout (if metrics look good)
   LIGHTRAG_ROLLOUT_PERCENTAGE=15.0
   
   # Week 3: 50% rollout
   LIGHTRAG_ROLLOUT_PERCENTAGE=50.0
   
   # Week 4: Full rollout
   LIGHTRAG_ROLLOUT_PERCENTAGE=100.0
   ```

3. **Enable Monitoring**:
   ```bash
   LIGHTRAG_ENABLE_QUALITY_METRICS=true
   LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
   LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
   ```

### Rollback Procedure

To quickly rollback to Perplexity-only:
```bash
# Option 1: Disable integration
LIGHTRAG_INTEGRATION_ENABLED=false

# Option 2: Set rollout to 0%
LIGHTRAG_ROLLOUT_PERCENTAGE=0.0

# Option 3: Force Perplexity cohort
LIGHTRAG_FORCE_USER_COHORT=perplexity
```

---

## Variable Validation

The system performs automatic validation of environment variables:

- **Type Validation**: Boolean, float, and integer variables are validated
- **Range Validation**: Numeric variables are checked against valid ranges
- **Format Validation**: JSON variables are parsed and validated
- **Default Fallback**: Invalid values fall back to safe defaults
- **Startup Warnings**: Invalid configurations generate warnings in logs

---

## Security Considerations

### Sensitive Variables
These variables contain sensitive information and should be secured:
- `OPENAI_API_KEY`
- `PERPLEXITY_API`
- `LIGHTRAG_USER_HASH_SALT`

### Best Practices
1. Use environment-specific `.env` files
2. Never commit API keys to version control
3. Use unique salt values per deployment
4. Rotate API keys regularly
5. Monitor for unusual usage patterns

---

## Troubleshooting

### Common Issues

1. **LightRAG Not Activating**:
   - Check `LIGHTRAG_INTEGRATION_ENABLED=true`
   - Verify `OPENAI_API_KEY` is set
   - Check `LIGHTRAG_ROLLOUT_PERCENTAGE > 0`

2. **All Requests Going to Perplexity**:
   - Verify user hash is within rollout percentage
   - Check circuit breaker status
   - Verify quality thresholds are not too restrictive

3. **Circuit Breaker Always Open**:
   - Check LightRAG service availability
   - Review error logs for failure patterns
   - Consider adjusting failure threshold

4. **Quality Scores Too Low**:
   - Review quality assessment configuration
   - Check if content meets length requirements
   - Verify citation availability

### Debug Mode

Enable debug logging to troubleshoot issues:
```bash
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_FILE_LOGGING=true
```

---

## Performance Impact

### Memory Usage
- Feature flag evaluation: ~1MB baseline
- User cohort caching: ~100 bytes per user
- Performance metrics: ~10KB per 1000 requests

### CPU Usage
- Hash calculation: ~0.1ms per request
- Routing decision: ~0.5ms per request
- Quality assessment: ~2ms per response

### Network Impact
- No additional network calls for routing decisions
- Cached routing decisions reduce repeated calculations
- Circuit breaker prevents unnecessary failed requests

---

## Version Compatibility

| Variable | Version | Status | Notes |
|----------|---------|--------|-------|
| `LIGHTRAG_INTEGRATION_ENABLED` | v1.0+ | Stable | Core integration flag |
| `LIGHTRAG_ROLLOUT_PERCENTAGE` | v1.0+ | Stable | Hash-based routing |
| `LIGHTRAG_ENABLE_AB_TESTING` | v1.0+ | Stable | A/B test support |
| `LIGHTRAG_ROUTING_RULES` | v1.0+ | Beta | JSON-based rules |
| `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING` | v1.0+ | Beta | Advanced routing |

---

For additional support, refer to the [main documentation](../docs/) or check the [troubleshooting guide](../docs/TROUBLESHOOTING.md).