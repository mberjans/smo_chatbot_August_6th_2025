# Uncertainty-Aware Fallback Cascade System Integration Guide

## Overview

The Uncertainty-Aware Fallback Cascade System enhances the existing Clinical Metabolomics Oracle with a sophisticated multi-step fallback mechanism specifically designed for handling uncertain classifications. This system provides intelligent routing between LightRAG, Perplexity, and Cache systems with proactive uncertainty detection and performance optimization.

## Key Features

- **Multi-Step Cascade**: LightRAG → Perplexity → Emergency Cache
- **Uncertainty-Aware Routing**: Different cascade paths based on uncertainty types
- **Performance Optimized**: < 200ms total cascade time with circuit breakers
- **Seamless Integration**: Full backward compatibility with existing 5-level fallback hierarchy
- **Comprehensive Monitoring**: Real-time performance tracking and alerting
- **Recovery Mechanisms**: Automatic retry logic and graceful degradation

## Architecture

### Core Components

1. **UncertaintyAwareFallbackCascade**: Main orchestrator
2. **CascadeDecisionEngine**: Intelligent routing logic
3. **CascadePerformanceMonitor**: Performance tracking
4. **CascadeCircuitBreaker**: Performance optimization
5. **Integration Layer**: Backward compatibility

### Cascade Strategies

The system implements 5 cascade strategies based on uncertainty analysis:

- **FULL_CASCADE**: LightRAG → Perplexity → Cache
- **SKIP_LIGHTRAG**: Perplexity → Cache (when LightRAG unreliable)
- **DIRECT_TO_CACHE**: Emergency cache only
- **CONFIDENCE_BOOSTED**: LightRAG with enhanced confidence scoring
- **CONSENSUS_SEEKING**: Multiple approaches with consensus analysis

## Installation and Integration

### 1. Basic Setup

```python
from lightrag_integration.uncertainty_aware_cascade_system import (
    create_uncertainty_aware_cascade_system,
    integrate_cascade_with_existing_router
)

# Create cascade system
cascade_system = create_uncertainty_aware_cascade_system(
    config={
        'max_total_cascade_time_ms': 200.0,
        'lightrag_max_time_ms': 120.0,
        'perplexity_max_time_ms': 100.0,
        'cache_max_time_ms': 20.0
    },
    logger=your_logger
)
```

### 2. Integration with Existing Components

```python
# Integrate with existing system components
cascade_system.integrate_with_existing_components(
    query_router=your_query_router,
    llm_classifier=your_llm_classifier,
    research_categorizer=your_research_categorizer,
    confidence_scorer=your_confidence_scorer
)
```

### 3. Integration with Existing Fallback System

```python
from lightrag_integration.comprehensive_fallback_system import create_comprehensive_fallback_system

# Create or use existing fallback orchestrator
fallback_orchestrator = create_comprehensive_fallback_system()

# Create cascade with fallback integration
cascade_system = create_uncertainty_aware_cascade_system(
    fallback_orchestrator=fallback_orchestrator,
    config=config,
    logger=logger
)
```

## Usage

### Basic Query Processing

```python
# Process a query with uncertainty-aware cascade
result = cascade_system.process_query_with_uncertainty_cascade(
    query_text="What are the metabolic pathways involved in glucose metabolism?",
    context={'user_id': '12345', 'priority': 'normal'},
    priority='normal'
)

# Check results
if result.success:
    routing_prediction = result.routing_prediction
    confidence = routing_prediction.confidence
    reasoning = routing_prediction.reasoning
    
    print(f"Query processed successfully in {result.total_cascade_time_ms:.1f}ms")
    print(f"Strategy used: {result.cascade_path_used.value}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Steps attempted: {result.total_steps_attempted}")
else:
    print("Query processing failed")
    print(f"Debug info: {result.debug_info}")
```

### Advanced Usage with Custom Configuration

```python
# Advanced configuration
advanced_config = {
    'max_total_cascade_time_ms': 150.0,  # Stricter time limit
    'lightrag_max_time_ms': 100.0,
    'perplexity_max_time_ms': 80.0,
    'cache_max_time_ms': 15.0,
    'max_workers': 4,  # Parallel processing
    'high_confidence_threshold': 0.8,
    'medium_confidence_threshold': 0.6,
    'low_confidence_threshold': 0.4
}

cascade_system = create_uncertainty_aware_cascade_system(
    config=advanced_config,
    logger=logger
)
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "What are metabolic biomarkers for diabetes?",
    "How does temperature affect metabolite stability?",
    "Can you explain metabolomics data analysis?"
]

results = []
for query in queries:
    result = cascade_system.process_query_with_uncertainty_cascade(query)
    results.append(result)

# Analyze batch performance
total_time = sum(r.total_cascade_time_ms for r in results)
success_rate = sum(1 for r in results if r.success) / len(results)
print(f"Batch processed: {len(queries)} queries in {total_time:.1f}ms")
print(f"Success rate: {success_rate:.1%}")
```

## Monitoring and Performance

### Performance Monitoring

```python
# Get comprehensive performance summary
performance_summary = cascade_system.get_cascade_performance_summary()

print("Overall Performance:")
overall = performance_summary['overall_performance']
print(f"  Total cascades: {overall['total_cascades']}")
print(f"  Success rate: {overall['success_rate']:.1%}")
print(f"  Average time: {overall['average_cascade_time_ms']:.1f}ms")
print(f"  Compliance rate: {overall['compliance_rate']:.1%}")

print("\nStrategy Performance:")
for strategy, stats in performance_summary['strategy_performance'].items():
    print(f"  {strategy}: {stats['success_rate']:.1%} ({stats['successes']}/{stats['attempts']})")
```

### Circuit Breaker Status

```python
# Monitor circuit breaker health
cb_status = performance_summary['circuit_breaker_status']
for step_type, status in cb_status.items():
    print(f"{step_type}:")
    print(f"  State: {status['state']}")
    print(f"  Failures: {status['failure_count']}")
    print(f"  Avg response: {status['average_response_time_ms']:.1f}ms")
    print(f"  Can execute: {status['can_execute']}")
```

## Configuration Options

### Cascade System Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_total_cascade_time_ms` | 200.0 | Maximum total cascade time |
| `lightrag_max_time_ms` | 120.0 | Maximum time for LightRAG step |
| `perplexity_max_time_ms` | 100.0 | Maximum time for Perplexity step |
| `cache_max_time_ms` | 20.0 | Maximum time for cache step |
| `max_workers` | 3 | Thread pool size |

### Confidence Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `high_confidence_threshold` | 0.7 | High confidence threshold |
| `medium_confidence_threshold` | 0.5 | Medium confidence threshold |
| `low_confidence_threshold` | 0.3 | Low confidence threshold |
| `very_low_confidence_threshold` | 0.1 | Very low confidence threshold |

### Uncertainty Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ambiguity_threshold_moderate` | 0.4 | Moderate ambiguity threshold |
| `conflict_threshold_moderate` | 0.3 | Moderate conflict threshold |
| `evidence_strength_threshold_weak` | 0.3 | Weak evidence threshold |

## Error Handling

### Circuit Breaker Handling

```python
try:
    result = cascade_system.process_query_with_uncertainty_cascade(query)
except Exception as e:
    if "circuit breaker" in str(e).lower():
        print("Service temporarily unavailable - circuit breaker activated")
        # Implement fallback logic
    else:
        print(f"Unexpected error: {e}")
```

### Timeout Handling

```python
# Configure aggressive timeout for high-load scenarios
config = {'max_total_cascade_time_ms': 100.0}
cascade_system = create_uncertainty_aware_cascade_system(config=config)

result = cascade_system.process_query_with_uncertainty_cascade(query)
if result.total_cascade_time_ms > 100.0:
    print("Warning: Cascade exceeded time limit")
    # Check performance alerts
    if result.performance_alerts:
        for alert in result.performance_alerts:
            print(f"Performance alert: {alert}")
```

## Integration with Existing Router

### Drop-in Replacement Pattern

```python
# Replace existing router calls
class EnhancedRouter:
    def __init__(self, cascade_system):
        self.cascade_system = cascade_system
    
    def route_query(self, query_text: str, context: dict = None) -> RoutingPrediction:
        # Use cascade system instead of direct routing
        result = self.cascade_system.process_query_with_uncertainty_cascade(
            query_text, context
        )
        
        if result.success:
            return result.routing_prediction
        else:
            # Fallback to emergency response
            return self._create_emergency_routing()
    
    def _create_emergency_routing(self):
        # Implement emergency routing logic
        pass
```

### Compatibility Layer

```python
from lightrag_integration.enhanced_query_router_with_fallback import (
    EnhancedBiomedicalQueryRouter,
    FallbackIntegrationConfig
)

# Use enhanced router with automatic cascade integration
config = FallbackIntegrationConfig(
    enable_fallback_system=True,
    max_response_time_ms=200.0,
    confidence_threshold=0.5
)

enhanced_router = EnhancedBiomedicalQueryRouter(
    existing_router=your_existing_router,
    fallback_config=config
)

# Works as drop-in replacement
result = enhanced_router.route_query("Your query here")
```

## Testing and Validation

### Running Integration Tests

```python
from lightrag_integration.cascade_integration_example import (
    CascadeIntegrationValidator,
    run_cascade_integration_demo
)

# Run comprehensive validation
validator = CascadeIntegrationValidator()
validation_report = validator.run_comprehensive_validation()

# Check if system is production ready
if validation_report['validation_summary']['system_ready_for_production']:
    print("✓ System validated and ready for production")
else:
    print("✗ System requires improvements before production")
    for rec in validation_report['recommendations']:
        print(f"  - {rec}")
```

### Custom Testing

```python
# Test specific scenarios
test_cases = [
    {
        'query': 'High uncertainty query...',
        'expected_strategy': 'consensus_seeking',
        'max_time_ms': 200.0
    }
]

for case in test_cases:
    result = cascade_system.process_query_with_uncertainty_cascade(case['query'])
    
    assert result.cascade_path_used.value == case['expected_strategy']
    assert result.total_cascade_time_ms <= case['max_time_ms']
    assert result.success
    
    print(f"✓ Test case passed: {case['query'][:30]}...")
```

## Best Practices

### Performance Optimization

1. **Configure appropriate timeouts** based on your performance requirements
2. **Monitor circuit breaker status** regularly to detect degradation
3. **Use batch processing** for multiple queries when possible
4. **Implement caching** at the application level for frequently asked questions

### Error Handling

1. **Always check result.success** before using routing predictions
2. **Implement fallback logic** for cascade system failures
3. **Monitor performance alerts** and take corrective action
4. **Log detailed error information** for debugging

### Integration

1. **Test thoroughly** with your existing components
2. **Start with conservative timeouts** and adjust based on performance
3. **Monitor system behavior** in production before full rollout
4. **Maintain backward compatibility** by preserving existing interfaces

## Troubleshooting

### Common Issues

**Issue**: Cascade exceeds time limits
**Solution**: Reduce individual step timeouts or overall cascade timeout

**Issue**: Circuit breakers frequently open
**Solution**: Check service health and adjust failure thresholds

**Issue**: Low confidence scores
**Solution**: Review uncertainty detection thresholds and confidence boosting

**Issue**: Integration compatibility problems
**Solution**: Verify all required interfaces are properly implemented

### Debug Information

```python
# Enable detailed debugging
import logging
logging.getLogger('lightrag_integration').setLevel(logging.DEBUG)

# Access debug information
result = cascade_system.process_query_with_uncertainty_cascade(query)
print(f"Debug info: {json.dumps(result.debug_info, indent=2)}")

# Check step-by-step execution
for i, step in enumerate(result.step_results):
    print(f"Step {step.step_number}: {step.step_type.value}")
    print(f"  Success: {step.success}")
    print(f"  Time: {step.processing_time_ms:.1f}ms")
    if not step.success:
        print(f"  Error: {step.error_message}")
```

## Support and Maintenance

### Regular Monitoring

1. Check performance metrics daily
2. Review circuit breaker status weekly
3. Analyze cascade strategy effectiveness monthly
4. Update confidence thresholds based on system learning

### Updates and Improvements

1. Monitor system performance and adjust configurations
2. Implement new cascade strategies as needed
3. Update uncertainty detection algorithms based on query patterns
4. Optimize performance based on production metrics

## Conclusion

The Uncertainty-Aware Fallback Cascade System provides robust, performant, and intelligent query processing with comprehensive fallback mechanisms. By following this integration guide, you can seamlessly enhance your Clinical Metabolomics Oracle with advanced uncertainty handling while maintaining full backward compatibility and meeting strict performance requirements.

For additional support or questions, refer to the implementation code documentation or contact the development team.