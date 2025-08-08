# Feature Flags Usage Guide

This comprehensive guide demonstrates how to use the LightRAG feature flag system for intelligent routing, A/B testing, and gradual rollouts in production environments.

## 🎯 Quick Start

### Basic Feature Flag Setup

```bash
# Enable LightRAG integration with 25% traffic
export LIGHTRAG_INTEGRATION_ENABLED="true"
export LIGHTRAG_ROLLOUT_PERCENTAGE="25.0"

# Enable A/B testing for statistical comparison
export LIGHTRAG_ENABLE_AB_TESTING="true"

# Enable circuit breaker for stability
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="5"

# Enable quality monitoring
export LIGHTRAG_ENABLE_QUALITY_METRICS="true"
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.7"

# Set user assignment salt for consistent routing
export LIGHTRAG_USER_HASH_SALT="your_secure_salt_here"
```

### Run the Main Integration

```bash
# Run the main integration example
python examples/main_integration_example.py

# Or start the Chainlit server (enhanced main.py)
chainlit run examples/main_integration_example.py
```

## 🔧 Feature Flag Configuration

### Core Settings

| Environment Variable | Description | Values | Default |
|---------------------|-------------|---------|---------|
| `LIGHTRAG_INTEGRATION_ENABLED` | Master switch for LightRAG | `true`/`false` | `true` |
| `LIGHTRAG_ROLLOUT_PERCENTAGE` | Percentage of users routed to LightRAG | `0.0-100.0` | `0.0` |
| `LIGHTRAG_ENABLE_AB_TESTING` | Enable A/B testing between systems | `true`/`false` | `false` |

### Circuit Breaker Settings

| Environment Variable | Description | Values | Default |
|---------------------|-------------|---------|---------|
| `LIGHTRAG_ENABLE_CIRCUIT_BREAKER` | Enable circuit breaker protection | `true`/`false` | `true` |
| `LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | Number of failures before opening | Integer | `5` |
| `LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | Recovery timeout (seconds) | Integer | `300` |

### Quality Monitoring

| Environment Variable | Description | Values | Default |
|---------------------|-------------|---------|---------|
| `LIGHTRAG_ENABLE_QUALITY_METRICS` | Enable quality monitoring | `true`/`false` | `true` |
| `LIGHTRAG_MIN_QUALITY_THRESHOLD` | Minimum quality score (0.0-1.0) | Float | `0.7` |

### Advanced Settings

| Environment Variable | Description | Values | Default |
|---------------------|-------------|---------|---------|
| `LIGHTRAG_USER_HASH_SALT` | Salt for consistent user assignment | String | `default_salt` |
| `LIGHTRAG_FORCE_USER_COHORT` | Force all users to specific cohort | `lightrag`/`perplexity`/`null` | `null` |
| `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING` | Enable conditional routing rules | `true`/`false` | `false` |

## 🚀 Deployment Scenarios

### 1. Development Environment

```bash
# Full rollout for development
export LIGHTRAG_INTEGRATION_ENABLED="true"
export LIGHTRAG_ROLLOUT_PERCENTAGE="100.0"
export LIGHTRAG_ENABLE_AB_TESTING="false"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="10"
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.5"
```

### 2. Staging Environment

```bash
# 50% rollout with A/B testing
export LIGHTRAG_INTEGRATION_ENABLED="true"
export LIGHTRAG_ROLLOUT_PERCENTAGE="50.0"
export LIGHTRAG_ENABLE_AB_TESTING="true"
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="5"
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.7"
```

### 3. Production Environment

```bash
# Conservative 10% rollout
export LIGHTRAG_INTEGRATION_ENABLED="true"
export LIGHTRAG_ROLLOUT_PERCENTAGE="10.0"
export LIGHTRAG_ENABLE_AB_TESTING="true"
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="3"
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.8"
```

## 📊 Monitoring and Analytics

### Performance Monitoring

The feature flag system automatically tracks:

- **Response Times**: Average, median, 95th percentile
- **Success Rates**: Percentage of successful queries
- **Quality Scores**: Relevance and accuracy metrics
- **Circuit Breaker State**: Open/closed status and failure counts
- **User Cohort Distribution**: A/B testing split

### Accessing Metrics

```python
from lightrag_integration import FeatureFlagManager, LightRAGConfig

config = LightRAGConfig()
feature_manager = FeatureFlagManager(config)

# Get comprehensive performance summary
summary = feature_manager.get_performance_summary()
print(json.dumps(summary, indent=2))
```

### Example Output

```json
{
  "circuit_breaker": {
    "is_open": false,
    "failure_count": 2,
    "success_rate": 0.98,
    "total_requests": 1000
  },
  "performance": {
    "lightrag": {
      "success_count": 245,
      "error_count": 5,
      "avg_response_time": 2.3,
      "avg_quality_score": 0.85
    },
    "perplexity": {
      "success_count": 740,
      "error_count": 10,
      "avg_response_time": 1.8,
      "avg_quality_score": 0.78
    }
  }
}
```

## 🧪 A/B Testing

### Enable A/B Testing

```bash
export LIGHTRAG_ENABLE_AB_TESTING="true"
export LIGHTRAG_ROLLOUT_PERCENTAGE="50.0"  # 50% split
```

### Run A/B Testing Analysis

```python
# Run comprehensive A/B testing
python examples/ab_testing_example.py
```

### A/B Testing Metrics

- **Statistical Significance**: p-values and confidence intervals
- **Effect Sizes**: Cohen's d for practical significance
- **Business Metrics**: Cost per query, user satisfaction
- **Performance Comparison**: Response times, quality scores

## 🎚️ Gradual Rollouts

### Linear Rollout Strategy

```python
# Run rollout scenarios demonstration
python examples/rollout_scenarios.py
```

Rollout progression:
1. **5%** → Monitor for 1 hour
2. **15%** → Monitor for 2 hours  
3. **30%** → Monitor for 4 hours
4. **50%** → Monitor for 8 hours
5. **100%** → Full rollout

### Canary Deployment

```bash
# Start with 1% canary
export LIGHTRAG_ROLLOUT_PERCENTAGE="1.0"

# Monitor metrics, then increase
export LIGHTRAG_ROLLOUT_PERCENTAGE="5.0"
export LIGHTRAG_ROLLOUT_PERCENTAGE="25.0"
export LIGHTRAG_ROLLOUT_PERCENTAGE="100.0"
```

## 🛡️ Safety Features

### Circuit Breaker Protection

The circuit breaker automatically protects against failures:

```bash
# Configure circuit breaker
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="5"
export LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT="300"
```

**Behavior**:
- Opens after 5 consecutive failures
- Routes all traffic to Perplexity when open
- Attempts recovery after 5 minutes
- Logs all state changes

### Quality Gates

Automatic quality monitoring with thresholds:

```bash
# Set quality thresholds
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.7"
export LIGHTRAG_ENABLE_QUALITY_METRICS="true"
```

**Behavior**:
- Monitors average quality scores
- Routes to Perplexity if quality drops below threshold
- Tracks quality trends over time

### Emergency Rollback

```python
# Manual emergency rollback
feature_manager.update_rollout_percentage(0.0)

# Or disable integration entirely
os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'false'
```

## 🎯 Routing Logic

### User Assignment

Users are consistently assigned to cohorts using:

```python
# Consistent hash-based assignment
user_hash = hashlib.sha256(f"{user_id}:{salt}".encode()).hexdigest()
assignment_percentage = int(user_hash[-8:], 16) / (16**8 - 1) * 100

# Route based on percentage
if assignment_percentage <= rollout_percentage:
    route_to_lightrag()
else:
    route_to_perplexity()
```

### Routing Decision Tree

1. **Integration Disabled** → Perplexity
2. **Forced Cohort** → Specified system
3. **Circuit Breaker Open** → Perplexity
4. **Quality Below Threshold** → Perplexity
5. **Conditional Rules Failed** → Perplexity
6. **User Assignment** → LightRAG or Perplexity

## 📈 Production Best Practices

### 1. Gradual Rollout

```bash
# Week 1: Canary (1%)
export LIGHTRAG_ROLLOUT_PERCENTAGE="1.0"

# Week 2: Small group (5%)
export LIGHTRAG_ROLLOUT_PERCENTAGE="5.0"

# Week 3: Larger group (15%)
export LIGHTRAG_ROLLOUT_PERCENTAGE="15.0"

# Week 4: Significant portion (30%)
export LIGHTRAG_ROLLOUT_PERCENTAGE="30.0"

# Week 5+: Full rollout (100%)
export LIGHTRAG_ROLLOUT_PERCENTAGE="100.0"
```

### 2. Monitoring Schedule

- **Real-time**: Circuit breaker status, error rates
- **Hourly**: Response times, quality scores
- **Daily**: A/B testing results, cost analysis
- **Weekly**: Performance trends, user satisfaction

### 3. Alert Thresholds

```bash
# Production alert thresholds
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="3"  # Strict
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.8"            # High quality
```

### 4. Rollback Triggers

Automatic rollback conditions:
- Error rate > 5%
- Quality score < 0.6
- Circuit breaker opens
- Response time > 10 seconds

## 🔍 Troubleshooting

### Common Issues

**1. Users Not Being Routed to LightRAG**
```bash
# Check configuration
echo $LIGHTRAG_INTEGRATION_ENABLED
echo $LIGHTRAG_ROLLOUT_PERCENTAGE

# Verify circuit breaker status
python -c "
from lightrag_integration import FeatureFlagManager, LightRAGConfig
config = LightRAGConfig()
manager = FeatureFlagManager(config)
print(manager.get_performance_summary()['circuit_breaker'])
"
```

**2. High Error Rates**
```bash
# Check circuit breaker settings
echo $LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD

# Review error logs
tail -n 100 logs/lightrag_integration.log
```

**3. Quality Issues**
```bash
# Check quality threshold
echo $LIGHTRAG_MIN_QUALITY_THRESHOLD

# Run quality assessment
python examples/ab_testing_example.py
```

### Debug Mode

Enable detailed logging:

```bash
export LIGHTRAG_LOG_LEVEL="DEBUG"
export LIGHTRAG_ENABLE_DETAILED_LOGGING="true"
```

## 📚 Example Usage Patterns

### Pattern 1: Conservative Production Rollout

```python
#!/usr/bin/env python3
"""Conservative production rollout pattern."""

import os
from lightrag_integration import LightRAGConfig, FeatureFlagManager

# Conservative production settings
config = LightRAGConfig()
config.lightrag_integration_enabled = True
config.lightrag_rollout_percentage = 5.0  # Start small
config.lightrag_enable_circuit_breaker = True
config.lightrag_circuit_breaker_failure_threshold = 2  # Very strict
config.lightrag_min_quality_threshold = 0.85  # High quality bar

feature_manager = FeatureFlagManager(config)

# Monitor and gradually increase
for percentage in [5, 10, 25, 50, 100]:
    print(f"Rolling out to {percentage}%...")
    feature_manager.update_rollout_percentage(percentage)
    # Monitor metrics between increases
```

### Pattern 2: A/B Testing with Statistical Analysis

```python
#!/usr/bin/env python3
"""A/B testing pattern with statistical analysis."""

from examples.ab_testing_example import ABTestingFramework
from lightrag_integration import FeatureFlagManager, LightRAGConfig

# Enable 50/50 A/B testing
config = LightRAGConfig()
config.lightrag_rollout_percentage = 50.0
config.lightrag_enable_ab_testing = True

feature_manager = FeatureFlagManager(config)
ab_framework = ABTestingFramework(feature_manager)

# Run test for sufficient sample size
# ... collect data ...

# Generate statistical report
report = ab_framework.generate_comprehensive_report("prod_ab_test")
print(f"Winner: {report.winner}")
print(f"Confidence: {report.confidence_level:.1%}")
```

### Pattern 3: Emergency Response

```python
#!/usr/bin/env python3
"""Emergency response pattern."""

from lightrag_integration import FeatureFlagManager

# Immediate rollback
feature_manager.update_rollout_percentage(0.0)

# Or disable entirely
import os
os.environ['LIGHTRAG_INTEGRATION_ENABLED'] = 'false'

# Check system health
summary = feature_manager.get_performance_summary()
if summary['circuit_breaker']['is_open']:
    print("🚨 Circuit breaker is open - system protected")
```

## 🎓 Advanced Configuration

### Conditional Routing Rules

```bash
# Enable conditional routing
export LIGHTRAG_ENABLE_CONDITIONAL_ROUTING="true"
```

Configure rules in code:

```python
config.lightrag_routing_rules = {
    "complex_queries": {
        "type": "query_length",
        "min_length": 100,
        "max_length": 1000
    },
    "metabolomics_queries": {
        "type": "query_type",
        "allowed_types": ["metabolomics", "clinical"]
    }
}
```

### Geographic Routing

For enterprise deployments:

```python
config.lightrag_geographic_routing = {
    "us_east": {"rollout_percentage": 25.0},
    "us_west": {"rollout_percentage": 15.0}, 
    "europe": {"rollout_percentage": 10.0}
}
```

### Multi-Environment Configuration

```yaml
# config/environments.yaml
development:
  rollout_percentage: 100.0
  enable_ab_testing: false
  circuit_breaker_threshold: 10
  
staging:
  rollout_percentage: 50.0
  enable_ab_testing: true
  circuit_breaker_threshold: 5
  
production:
  rollout_percentage: 10.0
  enable_ab_testing: true  
  circuit_breaker_threshold: 3
```

## 🔗 Integration Examples

Run the comprehensive examples:

```bash
# Main integration with feature flags
python examples/main_integration_example.py

# Rollout scenarios
python examples/rollout_scenarios.py

# A/B testing framework
python examples/ab_testing_example.py

# Production deployment configs
python examples/production_deployment_guide.py
```

## 📞 Support

For additional support or questions about feature flag implementation:

1. Review the comprehensive examples in this directory
2. Check the configuration reference in `lightrag_integration/config.py`
3. Enable debug logging for detailed troubleshooting
4. Use the built-in validation and health check functions

The feature flag system is designed to provide safe, gradual, and intelligent routing between LightRAG and Perplexity systems with comprehensive monitoring and automatic safety features.