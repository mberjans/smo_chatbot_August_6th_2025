# Circuit Breaker High-Concurrency Stress Testing

This document explains how to run and interpret the high-concurrency stress tests for the circuit breaker system.

## Overview

The stress testing suite (`test_circuit_breaker_high_concurrency_stress.py`) validates circuit breaker system stability under extreme concurrent load conditions. These tests are critical for ensuring the system can handle production-level traffic while maintaining reliability and performance.

## Test Categories

### 1. Extreme Concurrent Load Tests
- **test_extreme_concurrent_load_stability**: Tests 1500 concurrent async requests
- **test_extreme_concurrent_load_sync**: Tests 1000 concurrent sync requests

**Performance Targets:**
- Support 1000+ concurrent requests
- Throughput > 100 ops/sec (async), > 50 ops/sec (sync)
- Memory growth < 100MB (async), < 50MB (sync)
- 95th percentile latency < 100ms

### 2. Memory Leak Detection Tests
- **test_memory_leak_detection_long_running**: 30-second continuous load test
- **test_circuit_breaker_memory_overhead**: Memory overhead analysis

**Performance Targets:**
- Memory growth < 20MB over 30 seconds
- Circuit breaker overhead < 5MB for 10k operations
- Per-operation overhead < 100 bytes

### 3. Performance Degradation Tests
- **test_performance_degradation_under_load**: Tests increasing load levels (10-200 concurrent)

**Performance Targets:**
- Throughput degradation < 50% at high load
- Latency increase < 3x from baseline
- Success rate > 80% for reasonable loads

### 4. Thread Safety Tests
- **test_thread_safety_data_race_detection**: Multi-threaded race condition detection
- **test_concurrent_state_transition_consistency**: State consistency under async load

**Performance Targets:**
- No data races or state corruption detected
- Shared counter consistency within 10% tolerance
- Valid state transitions only

### 5. Resource Exhaustion Tests
- **test_resource_exhaustion_handling**: Behavior under resource constraints

**Performance Targets:**
- Memory growth < 75MB under resource pressure
- Circuit breaker activation during exhaustion
- Graceful degradation without system crash

### 6. Circuit Breaker Overhead Tests
- **test_circuit_breaker_overhead_measurement**: Sync overhead measurement
- **test_async_circuit_breaker_overhead**: Async overhead measurement

**Performance Targets:**
- Sync overhead < 50ms, < 200% (relaxed for test env)
- Async overhead < 5ms, < 30%
- Success rate > 95%

### 7. Cost-Based Circuit Breaker Stress Tests
- **test_cost_based_stress_with_budget_limits**: Cost-aware protection under stress

**Performance Targets:**
- Cost protection activation under budget pressure
- Throughput > 20 ops/sec
- Memory growth < 30MB

## Running the Tests

### Run All Stress Tests
```bash
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py -m stress -v
```

### Run Specific Test Categories
```bash
# Run only memory tests
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py::TestMemoryLeakDetection -v

# Run only thread safety tests  
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py::TestThreadSafety -v

# Run only overhead measurement tests
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py::TestCircuitBreakerOverhead -v
```

### Run Individual Tests
```bash
# Run specific extreme load test
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py::TestExtremeConcurrentLoad::test_extreme_concurrent_load_stability -v -s

# Run memory leak detection
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py::TestMemoryLeakDetection::test_memory_leak_detection_long_running -v -s
```

### Run Tests with Custom Markers
```bash
# Run all stress tests
python -m pytest -m stress -v

# Run only slow tests (long-running)
python -m pytest -m slow -v

# Run stress tests without coverage (faster)
python -m pytest -m stress --no-cov -v
```

## Test Configuration

### Environment Variables
The tests automatically adapt to the environment but you can configure:
```bash
# Disable LightRAG integration for pure circuit breaker testing
export LIGHTRAG_INTEGRATION_ENABLED=false

# Adjust memory limits (if needed)
export PYTEST_MAX_WORKERS=50
```

### Performance Tuning
Tests are configured with realistic but achievable performance targets:

- **Concurrent limits**: Adjusted based on system capabilities
- **Memory thresholds**: Set to detect genuine leaks vs normal growth
- **Latency targets**: Based on production requirements
- **Success rates**: Allow for some failures during stress conditions

## Interpreting Results

### Successful Test Output
```
Extreme Concurrent Load Test Results:
- Total requests: 1500
- Successful operations: 1425
- Circuit breaker blocks: 45
- Errors: 30
- Throughput: 156.78 ops/sec
- Duration: 9.57s
- Memory growth: 15.23MB
- 95th percentile latency: 87.45ms
```

### Key Metrics to Monitor

1. **Throughput**: Operations per second under load
2. **Success Rate**: Percentage of successful operations
3. **Memory Growth**: Memory consumption during testing
4. **Latency Percentiles**: Response time distribution
5. **Circuit Breaker Effectiveness**: Protection during failures

### Failure Analysis

If tests fail, check:

1. **Memory Issues**: Excessive memory growth may indicate leaks
2. **Performance Degradation**: Latency/throughput outside targets
3. **Thread Safety**: Data races or state corruption
4. **Resource Limits**: System resource exhaustion
5. **Circuit Breaker Logic**: Incorrect state transitions

## Performance Baselines

These baselines are established for a modern development machine:

- **Concurrent Capacity**: 1000+ simultaneous operations
- **Memory Efficiency**: < 100KB per 1000 operations
- **Latency**: < 100ms 95th percentile under normal load
- **Overhead**: < 10ms circuit breaker processing time
- **Reliability**: > 95% success rate under normal conditions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Warnings**: May indicate genuine issues or test environment limits
3. **Timing Variability**: Test environment may have higher variability
4. **Resource Limits**: Adjust concurrent limits for your system

### Debug Mode
Run tests with verbose output and no coverage for debugging:
```bash
python -m pytest tests/test_circuit_breaker_high_concurrency_stress.py::TestName::test_method -v -s --tb=long --no-cov
```

### System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **CPU**: Multi-core processor for concurrent testing
- **Dependencies**: pytest, asyncio, threading, psutil, statistics

## Integration with CI/CD

For continuous integration, consider:
- Running stress tests on dedicated CI runners
- Using timeouts to prevent hung tests
- Storing performance metrics for trend analysis
- Alerting on performance regressions

Example CI configuration:
```yaml
stress_tests:
  script:
    - python -m pytest -m stress --tb=short --timeout=300
  timeout: 600
  allow_failure: false
  only:
    - master
    - develop
```