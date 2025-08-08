# LightRAG Feature Flag System

A comprehensive feature flag system for seamless LightRAG integration with the Clinical Metabolomics Oracle, providing intelligent routing, gradual rollout capabilities, and robust fallback mechanisms.

## 🚀 Overview

This feature flag system enables production-ready integration of LightRAG alongside the existing Perplexity API, with advanced capabilities for:

- **Intelligent Routing**: Hash-based consistent user assignment with percentage rollout control
- **A/B Testing**: Statistical comparison between LightRAG and Perplexity responses
- **Circuit Breaker Protection**: Automatic fallback on service degradation
- **Quality Assessment**: Real-time response quality monitoring and routing decisions
- **Gradual Rollout**: Automated progressive rollout with safety mechanisms
- **Backward Compatibility**: Zero-impact integration with existing codebase

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application                          │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │     main.py         │    │  EnhancedCMO Oracle        │  │
│  │  (minimal changes)  │◄───┤  (drop-in replacement)    │  │
│  └─────────────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────┐
│                Integration Layer                            │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │ IntegratedQuery     │◄───┤    FeatureFlagManager     │  │
│  │     Service         │    │  (routing decisions)       │  │
│  └─────────────────────┘    └─────────────────────────────┘  │
│            │                              │                  │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │   RolloutManager    │    │    PerformanceMetrics     │  │
│  │ (gradual rollout)   │    │   (quality assessment)    │  │
│  └─────────────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │  LightRAGQuery      │    │   PerplexityQuery          │  │
│  │    Service          │    │     Service                │  │
│  │                     │    │  (existing logic)          │  │
│  └─────────────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

#### 🎯 **Intelligent Routing**
- Hash-based consistent user assignment ensures session stability
- Configurable percentage rollout (0-100%)
- User cohort forcing for testing specific scenarios
- Conditional routing based on query characteristics

#### 🔄 **Circuit Breaker Protection**
- Automatic service degradation detection
- Configurable failure thresholds and recovery timeouts
- Real-time circuit state monitoring
- Graceful fallback to Perplexity on failures

#### 📊 **A/B Testing & Performance Comparison**
- Statistical significance testing
- Response quality assessment and comparison
- Performance metrics collection (response times, success rates)
- Comprehensive reporting for data-driven decisions

#### 🚦 **Gradual Rollout Management**
- Multiple rollout strategies (linear, exponential, canary)
- Automated progression based on quality metrics
- Emergency rollback capabilities
- Real-time monitoring and alerting

## 📋 Quick Start

### 1. Environment Configuration

Create or update your `.env` file:

```bash
# Required - API Keys
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API=your_perplexity_api_key

# Core Integration Settings
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=10.0
LIGHTRAG_FALLBACK_TO_PERPLEXITY=true

# Quality & Performance
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.7
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true

# Circuit Breaker
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=300.0
```

### 2. Basic Integration

#### Option A: Drop-in Replacement (Recommended)

```python
# In your main.py, replace existing query logic with:
from lightrag_integration.main_integration import create_enhanced_oracle

# Initialize at startup
oracle = create_enhanced_oracle(PERPLEXITY_API)

@cl.on_message
async def on_message(message: cl.Message):
    # Get user session for consistent routing
    user_session = {
        'user_id': cl.user_session.get("user_id"),
        'session_id': cl.user_session.get("session_id")
    }
    
    # Process with intelligent routing
    result = await oracle.process_query(message.content, user_session)
    
    # Send response (maintains existing format)
    response_message = cl.Message(content=result['content'])
    await response_message.send()
```

#### Option B: Direct Integration

```python
from lightrag_integration.integration_wrapper import create_integrated_service
from lightrag_integration.config import LightRAGConfig

# Initialize services
config = LightRAGConfig.get_config()
service = create_integrated_service(config, PERPLEXITY_API)

# Process queries
request = QueryRequest(query_text="Your question here")
response = await service.query_async(request)
```

### 3. Monitoring & Management

```python
# Get system status
status = oracle.get_system_status()
print(f"Current rollout: {status['config_summary']['rollout_percentage']}%")
print(f"Circuit breaker: {'OPEN' if status['service_performance']['circuit_breaker']['is_open'] else 'CLOSED'}")

# Start automated rollout
rollout_id = oracle.start_rollout(strategy="linear", start_percentage=5.0, increment=10.0)
print(f"Started rollout: {rollout_id}")
```

## 🔧 Configuration Reference

### Core Integration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_INTEGRATION_ENABLED` | `false` | Master switch for LightRAG integration |
| `LIGHTRAG_ROLLOUT_PERCENTAGE` | `0.0` | Percentage of users routed to LightRAG (0-100) |
| `LIGHTRAG_FALLBACK_TO_PERPLEXITY` | `true` | Enable fallback on LightRAG failures |
| `LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS` | `30.0` | Timeout for LightRAG requests |

### Quality & Performance Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_QUALITY_METRICS` | `false` | Enable response quality assessment |
| `LIGHTRAG_MIN_QUALITY_THRESHOLD` | `0.7` | Minimum quality score for routing |
| `LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON` | `false` | Enable A/B testing metrics |

### Circuit Breaker Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_CIRCUIT_BREAKER` | `true` | Enable circuit breaker protection |
| `LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Failures before opening circuit |
| `LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | `300.0` | Recovery timeout in seconds |

### Advanced Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_ENABLE_AB_TESTING` | `false` | Enable A/B testing mode |
| `LIGHTRAG_USER_HASH_SALT` | `"cmo_lightrag_2025"` | Salt for consistent user hashing |
| `LIGHTRAG_FORCE_USER_COHORT` | `None` | Force users to specific cohort |
| `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING` | `false` | Enable query-based routing rules |

See [FEATURE_FLAG_ENVIRONMENT_VARIABLES.md](./FEATURE_FLAG_ENVIRONMENT_VARIABLES.md) for complete reference.

## 📊 Rollout Strategies

### Linear Rollout
Progressive increase in fixed increments:
```python
# 5% → 15% → 25% → 35% → ... → 100%
rollout_id = oracle.start_rollout(
    strategy="linear", 
    start_percentage=5.0, 
    increment=10.0,
    stage_duration=60  # minutes
)
```

### Exponential Rollout
Rapid scaling with doubling exposure:
```python
# 1% → 2% → 4% → 8% → 16% → 32% → 64% → 100%
rollout_id = oracle.start_rollout(
    strategy="exponential",
    start_percentage=1.0,
    stage_duration=60
)
```

### Canary Rollout
Small initial exposure with validation:
```python
# 1% (extended validation) → 100% (after approval)
rollout_id = oracle.start_rollout(
    strategy="canary",
    canary_percentage=1.0,
    canary_duration=120
)
```

## 🛡️ Safety Mechanisms

### Circuit Breaker Protection
- Automatically detects service degradation
- Prevents cascading failures
- Self-healing with configurable recovery
- Real-time monitoring and alerting

### Quality-Based Routing
- Real-time response quality assessment
- Automatic fallback on quality degradation
- Configurable quality thresholds
- Performance comparison metrics

### Emergency Rollback
- Instant rollback to 0% on critical issues
- Automated rollback on quality/performance degradation
- Manual rollback capabilities
- State persistence for recovery

## 📈 Monitoring & Observability

### Performance Metrics
```python
status = oracle.get_system_status()

# Service performance
lightrag_performance = status['service_performance']['performance']['lightrag']
perplexity_performance = status['service_performance']['performance']['perplexity']

print(f"LightRAG avg response time: {lightrag_performance['avg_response_time']:.2f}s")
print(f"LightRAG success rate: {lightrag_performance['success_count'] / (lightrag_performance['success_count'] + lightrag_performance['error_count']):.1%}")
```

### Circuit Breaker Status
```python
cb_status = status['service_performance']['circuit_breaker']
print(f"Circuit breaker: {'OPEN' if cb_status['is_open'] else 'CLOSED'}")
print(f"Failure rate: {cb_status['failure_rate']:.1%}")
print(f"Total requests: {cb_status['total_requests']}")
```

### Rollout Progress
```python
if oracle.rollout_manager:
    rollout_status = oracle.rollout_manager.get_rollout_status()
    if rollout_status:
        print(f"Rollout phase: {rollout_status['phase']}")
        print(f"Current percentage: {rollout_status['current_percentage']}%")
        print(f"Stage progress: {rollout_status.get('stage_progress', {}).get('duration_progress', 0):.1%}")
```

## 🔍 A/B Testing & Analysis

### Enable A/B Testing
```bash
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ROLLOUT_PERCENTAGE=50.0  # 50/50 split
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
```

### Collect Metrics
```python
# Automatic collection during queries
# Access via performance summary
status = oracle.get_system_status()
lightrag_metrics = status['service_performance']['performance']['lightrag']
perplexity_metrics = status['service_performance']['performance']['perplexity']

# Compare average quality scores
print(f"LightRAG quality: {lightrag_metrics['avg_quality_score']:.2f}")
print(f"Perplexity quality: {perplexity_metrics['avg_quality_score']:.2f}")
```

## 🚀 Production Deployment

### Phase 1: Canary Deployment (Week 1)
```bash
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ROLLOUT_PERCENTAGE=1.0
LIGHTRAG_ENABLE_QUALITY_METRICS=true
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.8
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=2  # Conservative
```

### Phase 2: Limited Rollout (Week 2-3)
```bash
LIGHTRAG_ROLLOUT_PERCENTAGE=15.0
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=true
```

### Phase 3: Full Rollout (Week 4+)
```bash
LIGHTRAG_ROLLOUT_PERCENTAGE=100.0
LIGHTRAG_ENABLE_PERFORMANCE_COMPARISON=false  # Reduce overhead
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.75  # Production threshold
```

## 🔧 Troubleshooting

### Common Issues

#### LightRAG Not Activating
1. Verify `LIGHTRAG_INTEGRATION_ENABLED=true`
2. Check `OPENAI_API_KEY` is set correctly
3. Ensure `LIGHTRAG_ROLLOUT_PERCENTAGE > 0`
4. Check logs for initialization errors

#### All Requests Going to Perplexity
1. Check user hash falls within rollout percentage
2. Verify circuit breaker is not open
3. Check quality thresholds are not too restrictive
4. Review conditional routing rules

#### Circuit Breaker Always Open
1. Check LightRAG service connectivity
2. Review failure logs in detail
3. Adjust failure threshold if needed
4. Verify API key permissions

### Debug Mode
```bash
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_FILE_LOGGING=true
```

## 📁 File Structure

```
lightrag_integration/
├── config.py                              # Enhanced LightRAGConfig
├── feature_flag_manager.py                # Core routing logic
├── integration_wrapper.py                 # Service abstraction
├── rollout_manager.py                     # Gradual rollout management
├── main_integration.py                    # Main.py integration helpers
├── FEATURE_FLAG_ENVIRONMENT_VARIABLES.md  # Environment variable reference
└── FEATURE_FLAG_SYSTEM_README.md         # This file
```

## 🤝 Contributing

### Adding New Routing Rules

1. Extend `RoutingReason` enum in `feature_flag_manager.py`
2. Add rule logic to `_evaluate_conditional_rules()`
3. Update environment variable schema
4. Add tests for new functionality

### Custom Quality Assessors

```python
def custom_quality_assessor(response: ServiceResponse) -> Dict[QualityMetric, float]:
    # Your custom quality assessment logic
    return {
        QualityMetric.RELEVANCE: calculate_relevance(response.content),
        QualityMetric.ACCURACY: check_accuracy(response.content),
        # ... other metrics
    }

# Register with service
oracle.query_service.set_quality_assessor(custom_quality_assessor)
```

## 📚 Additional Resources

- [Environment Variables Reference](./FEATURE_FLAG_ENVIRONMENT_VARIABLES.md)
- [LightRAG Configuration Guide](../docs/LIGHTRAG_CONFIG_REFERENCE.md)
- [Performance Benchmarking](../performance_benchmarking/README.md)
- [Cost Monitoring Documentation](../API_COST_MONITORING_MASTER_DOCUMENTATION.md)

## 🔐 Security Considerations

- **API Key Management**: Store keys securely, rotate regularly
- **User Hash Salt**: Use unique salt per environment
- **Logging**: Avoid logging sensitive information
- **Access Control**: Restrict rollout management access
- **Network Security**: Use HTTPS for all API calls

## 📊 Performance Impact

### Baseline Overhead
- Feature flag evaluation: ~0.5ms per request
- User hash calculation: ~0.1ms per request
- Quality assessment: ~2ms per response (when enabled)

### Memory Usage
- Feature flag manager: ~1MB baseline
- User cohort cache: ~100 bytes per user
- Performance metrics: ~10KB per 1000 requests

### Optimization Tips
- Use caching for repeated routing decisions
- Disable performance comparison in production after rollout
- Monitor memory usage with large user bases
- Use appropriate log levels to reduce I/O overhead

---

## 🏆 Success Metrics

Track these key metrics to measure rollout success:

- **Service Availability**: Circuit breaker open rate < 1%
- **Response Quality**: Average quality score > 0.8
- **Performance**: Response time within 10% of baseline
- **User Experience**: No increase in error rates
- **Cost Efficiency**: Cost per query reduction

Ready to get started? Check the [Quick Start](#📋-quick-start) section and begin with a small canary deployment!