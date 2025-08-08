# System Health Monitoring Integration Tests

## Overview

This comprehensive test suite validates the integration between system health monitoring and routing logic in the Clinical Metabolomics Oracle LightRAG system. The tests ensure that the routing system properly responds to service health changes and maintains system resilience under various failure scenarios.

## Test Architecture

### Core Components Tested

1. **Circuit Breaker Patterns** - Protection against cascading failures
2. **Health-Based Routing Decisions** - Intelligent service selection based on health
3. **Failure Detection and Recovery** - Automatic detection and recovery from service failures
4. **Performance Monitoring Integration** - Response time and error rate impact on routing
5. **Load Balancing** - Distribution of requests based on service health
6. **Service Availability Management** - Handling of partial and complete service outages

### Integration Points

- **Health Status → Routing Decisions**: Healthy services are preferred over degraded ones
- **Circuit Breaker States → Path Blocking**: Open circuit breakers prevent routing to failed services
- **Performance Degradation → Fallback Mechanisms**: Poor performance triggers hybrid/fallback routing
- **Service Failures → Route Re-evaluation**: Service failures cause automatic routing adjustments

## Test Structure

### Test Files

- **`test_system_health_monitoring_integration.py`** - Main test suite with comprehensive test classes
- **`run_system_health_monitoring_tests.py`** - Test runner with detailed reporting
- **`SYSTEM_HEALTH_MONITORING_INTEGRATION_README.md`** - This documentation file

### Test Classes

#### `TestCircuitBreakerIntegration`
Tests circuit breaker patterns for external API calls:
- Circuit breaker blocks unhealthy services
- Circuit breaker recovery enables normal routing
- Multiple service circuit breaker failures
- Emergency fallback when all circuit breakers are open

#### `TestHealthBasedRoutingDecisions`
Tests system health checks that influence routing decisions:
- Healthy service preference over degraded ones
- Global health score affecting routing confidence
- Emergency fallback on critically low system health
- Health metadata integration in routing decisions

#### `TestFailureDetectionAndRecovery`
Tests failure detection and recovery mechanisms:
- Consecutive failure detection and response
- Service recovery detection and routing restoration
- Performance degradation detection and mitigation
- Failure pattern analysis and prediction

#### `TestPerformanceMonitoring`
Tests performance monitoring that affects routing:
- Response time degradation impact on routing
- Error rate threshold-based routing changes
- Performance score integration in routing decisions
- Real-time performance monitoring integration

#### `TestLoadBalancing`
Tests load balancing between multiple backends:
- Load balancing with equal service health
- Load balancing with unequal service health
- Load balancing avoidance of unhealthy services
- Dynamic load balancing based on health changes

#### `TestServiceAvailabilityImpact`
Tests service availability impact on routing:
- Routing fallback when primary service becomes unavailable
- Partial availability effects on routing confidence
- Service availability recovery improving routing quality
- Multi-service availability scenarios

#### `TestHealthMonitoringIntegration`
Tests comprehensive integration scenarios:
- End-to-end health monitoring workflow
- Concurrent health monitoring under load
- Health monitoring statistics accuracy
- System resilience validation

## Mock Infrastructure

### MockServiceHealthMonitor
Simulates service health monitoring with configurable:
- Response time patterns
- Error rate injection
- Performance degradation simulation
- Circuit breaker state management

### MockSystemHealthManager
Coordinates multiple service health monitors:
- Global health score calculation
- Service registration and monitoring
- Health status aggregation
- Failure injection for testing

### HealthAwareRouter
Router that integrates with health monitoring:
- Health-based routing decision logic
- Circuit breaker integration
- Performance threshold management
- Routing statistics and metrics

## Key Test Scenarios

### 1. Circuit Breaker Integration

**Scenario**: Service experiences multiple consecutive failures
```python
# LightRAG service fails repeatedly
health_manager.inject_service_failure('lightrag', enabled=True)
for _ in range(10):
    lightrag_monitor.simulate_request()  # Triggers circuit breaker

# Routing should avoid LightRAG
result = router.route_query_with_health_awareness(query)
assert result.routing_decision != RoutingDecision.LIGHTRAG
```

**Expected Behavior**:
- Circuit breaker opens after failure threshold
- Routing automatically redirects to alternative services
- Circuit breaker recovery allows normal routing restoration

### 2. Health-Based Routing

**Scenario**: Services have different health states
```python
# Set different health states
health_manager.services['lightrag'].error_probability = 0.01  # Healthy
health_manager.services['perplexity'].set_performance_degradation(True)  # Degraded

# Routing should prefer healthy service
result = router.route_query_with_health_awareness(query)
```

**Expected Behavior**:
- Healthy services are preferred over degraded ones
- Confidence is adjusted based on service health
- Reasoning includes health considerations

### 3. Performance Monitoring

**Scenario**: Service performance degrades significantly
```python
# Inject performance degradation
health_manager.inject_service_degradation('perplexity', enabled=True)

# Performance should affect routing decisions
result = router.route_query_with_health_awareness(query)
```

**Expected Behavior**:
- Performance degradation triggers routing changes
- Hybrid routing is used for degraded services
- Confidence is reduced for poor performance

### 4. Service Recovery

**Scenario**: Failed service recovers to healthy state
```python
# Start with service failure
health_manager.inject_service_failure('lightrag', enabled=True)

# Service recovers
health_manager.inject_service_failure('lightrag', enabled=False)

# Routing should return to normal
result = router.route_query_with_health_awareness(query)
```

**Expected Behavior**:
- Service recovery is detected automatically
- Normal routing patterns are restored
- Confidence improves with service recovery

## Running the Tests

### Basic Test Execution
```bash
# Run all health monitoring integration tests
python -m pytest test_system_health_monitoring_integration.py -v

# Run specific test class
python -m pytest test_system_health_monitoring_integration.py::TestCircuitBreakerIntegration -v

# Run with health monitoring marker
python -m pytest -m health_monitoring -v
```

### Comprehensive Test Runner
```bash
# Run comprehensive test suite with detailed reporting
python run_system_health_monitoring_tests.py
```

### Test Output
The comprehensive test runner generates:
- **JSON Results**: Detailed test results with metrics
- **Text Report**: Human-readable summary and analysis
- **Performance Metrics**: Execution times and success rates
- **Effectiveness Analysis**: Health monitoring system assessment
- **Recommendations**: Actionable improvement suggestions

## Performance Requirements

### Response Time Targets
- **Individual Routing Decision**: < 100ms
- **Health Check Integration**: < 50ms additional overhead
- **Circuit Breaker Decision**: < 10ms
- **Concurrent Request Handling**: > 100 requests/second

### Reliability Targets
- **Test Success Rate**: > 95%
- **Health Detection Accuracy**: > 90%
- **Circuit Breaker Effectiveness**: > 99% failure prevention
- **Service Recovery Time**: < 30 seconds

## Monitoring and Alerting

### Health Monitoring Metrics
- **Global Health Score**: 0.0 - 1.0 (1.0 = perfect health)
- **Service Performance Scores**: Individual service health ratings
- **Circuit Breaker States**: Open/Closed/Half-Open status
- **Routing Statistics**: Health-based routing decisions

### Key Performance Indicators (KPIs)
- **Health-Based Routing Percentage**: % of requests routed based on health
- **Fallback Activation Rate**: % of requests using fallback mechanisms
- **Circuit Breaker Block Rate**: % of requests blocked by circuit breakers
- **Service Recovery Time**: Average time for service recovery detection

## Integration with Production Systems

### Configuration Requirements
```python
# Health monitoring configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60  # seconds
PERFORMANCE_DEGRADATION_THRESHOLD = 2000  # ms
```

### Monitoring Integration
- **Prometheus Metrics**: Health scores, circuit breaker states
- **Grafana Dashboards**: Real-time health monitoring visualization
- **Alert Manager**: Health degradation and circuit breaker alerts
- **Log Aggregation**: Structured health monitoring logs

## Troubleshooting

### Common Issues

#### High Test Failure Rate
- Check service mock configuration
- Verify health threshold settings
- Review concurrent test execution

#### Performance Issues
- Monitor test execution times
- Check for resource contention
- Verify mock service behavior

#### Circuit Breaker Not Triggering
- Verify failure injection settings
- Check failure threshold configuration
- Review consecutive failure detection

### Debugging Tips

#### Enable Debug Logging
```python
logging.getLogger('test_health_monitoring').setLevel(logging.DEBUG)
```

#### Analyze Test Metrics
```python
# Check routing statistics
stats = router.get_routing_statistics()
print(f"Health-based decisions: {stats['health_based_decisions']}")
print(f"Circuit breaker blocks: {stats['circuit_breaker_blocks']}")
```

#### Monitor Service Health
```python
# Get detailed health metrics
for service_name in ['lightrag', 'perplexity']:
    health = health_manager.get_service_health(service_name)
    print(f"{service_name}: {health.status.value} ({health.performance_score:.2f})")
```

## Future Enhancements

### Planned Improvements
1. **Machine Learning Integration**: Predictive health monitoring
2. **Advanced Circuit Breaker Patterns**: Bulkhead and timeout patterns
3. **Dynamic Threshold Adjustment**: Adaptive health thresholds
4. **Multi-Region Health Monitoring**: Geo-distributed health tracking
5. **Cost-Aware Health Routing**: Budget-conscious health routing

### Extension Points
- **Custom Health Metrics**: Domain-specific health indicators
- **External Health Providers**: Integration with external monitoring systems
- **Health-Based Auto-Scaling**: Automatic capacity adjustment
- **Chaos Engineering**: Automated failure injection testing

## Best Practices

### Test Development
1. **Comprehensive Coverage**: Test all health monitoring integration points
2. **Realistic Scenarios**: Use production-like failure patterns
3. **Performance Validation**: Include performance impact testing
4. **Concurrent Testing**: Validate behavior under concurrent load

### Production Deployment
1. **Gradual Rollout**: Deploy health monitoring integration incrementally
2. **Monitoring First**: Establish monitoring before enabling features
3. **Fallback Planning**: Ensure fallback mechanisms are tested
4. **Documentation**: Keep health monitoring documentation updated

### Maintenance
1. **Regular Testing**: Run integration tests regularly
2. **Threshold Tuning**: Adjust health thresholds based on production data
3. **Performance Monitoring**: Track health monitoring overhead
4. **Recovery Testing**: Regularly test failure recovery scenarios

## Conclusion

The system health monitoring integration tests provide comprehensive validation of the routing system's ability to maintain resilience and optimal performance under various failure scenarios. The test suite ensures that:

- **Circuit breakers** effectively prevent cascading failures
- **Health-based routing** optimizes service selection
- **Failure detection** enables rapid response to issues
- **Performance monitoring** maintains quality of service
- **Load balancing** distributes requests effectively
- **Service availability** management ensures system resilience

By running these tests regularly and monitoring the results, the system can maintain high availability and performance while gracefully handling service degradation and failures.