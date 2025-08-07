# Performance Testing Utilities and Benchmarking Helpers

## Overview

This document describes the comprehensive performance testing utilities and benchmarking helpers implemented for the Clinical Metabolomics Oracle LightRAG integration. These utilities build on the existing test infrastructure to provide standardized performance testing, monitoring, and analysis capabilities.

## Architecture

The performance utilities are organized into several key components:

```
tests/
├── performance_test_utilities.py          # Main performance utilities
├── demo_performance_test_utilities.py     # Demonstration script
├── test_performance_utilities_integration.py  # Integration tests
├── test_utilities.py                      # Base test utilities (existing)
├── performance_test_fixtures.py           # Performance test fixtures (existing)
└── performance_analysis_utilities.py      # Analysis utilities (existing)
```

## Key Components

### 1. PerformanceAssertionHelper

Provides comprehensive performance assertions with timing decorators, memory validation, throughput calculation, and threshold checking.

#### Features
- **Timing Decorators**: Automatic operation timing with assertions
- **Memory Monitoring**: Baseline establishment and leak detection
- **Throughput Calculation**: Operations per second measurement
- **Response Time Validation**: Latency assertions with percentiles
- **Resource Usage Assertions**: CPU and memory usage validation
- **Composite Benchmarks**: Multi-metric performance validation

#### Usage Example

```python
from performance_test_utilities import PerformanceAssertionHelper

# Initialize
assertion_helper = PerformanceAssertionHelper()
assertion_helper.establish_memory_baseline()

# Timing decorator
@assertion_helper.time_operation("query_processing", 5000, auto_assert=True)
async def process_clinical_query(query):
    # Your query processing logic
    result = await some_processing_function(query)
    return result

# Manual assertions
assertion_helper.assert_throughput(operations, duration, 2.0, "query_throughput")
assertion_helper.assert_memory_leak_absent(50.0, "memory_leak_check")
assertion_helper.assert_error_rate(errors, total, 5.0, "error_rate_check")

# Context manager
with assertion_helper.timing_context("batch_processing", 10000):
    # Batch processing logic
    pass
```

### 2. PerformanceBenchmarkSuite

Runs standardized benchmarks across different scenarios and tracks performance metrics over time with baseline comparison and regression detection.

#### Features
- **Standard Benchmarks**: Pre-configured benchmarks for common scenarios
- **Custom Benchmarks**: Configurable benchmark definitions
- **Baseline Comparison**: Historical performance tracking
- **Regression Detection**: Automatic performance regression analysis
- **Comprehensive Reporting**: Detailed performance reports with recommendations

#### Usage Example

```python
from performance_test_utilities import PerformanceBenchmarkSuite, BenchmarkConfiguration
from test_utilities import TestEnvironmentManager

# Setup
env_manager = TestEnvironmentManager()
benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)

# Run standard benchmarks
results = await benchmark_suite.run_benchmark_suite(
    benchmark_names=['clinical_query_performance', 'pdf_processing_performance'],
    operation_func=your_operation_function,
    data_generator=your_data_generator
)

# Custom benchmark
custom_benchmark = BenchmarkConfiguration(
    benchmark_name='custom_test',
    description='Custom performance test',
    target_thresholds={
        'response_time_ms': PerformanceThreshold(
            'response_time_ms', 3000, 'lte', 'ms', 'error',
            'Should respond within 3 seconds'
        )
    },
    test_scenarios=[your_test_scenario]
)

benchmark_suite.standard_benchmarks['custom_test'] = custom_benchmark
```

### 3. AdvancedResourceMonitor

Monitors system resources during performance testing with threshold-based alerts, trend analysis, and detailed diagnostics.

#### Features
- **Real-time Monitoring**: CPU, memory, I/O, network, threads, file descriptors
- **Threshold Alerts**: Configurable resource usage alerts
- **Trend Analysis**: Resource usage trend detection
- **Detailed Reporting**: Comprehensive resource usage reports
- **Integration**: Works with existing test infrastructure

#### Usage Example

```python
from performance_test_utilities import AdvancedResourceMonitor

# Initialize with custom thresholds
custom_thresholds = {
    'cpu_percent': 80.0,
    'memory_mb': 1000.0,
    'active_threads': 50
}

monitor = AdvancedResourceMonitor(
    sampling_interval=0.5,
    alert_thresholds=custom_thresholds
)

# Monitor test execution
monitor.start_monitoring()

# Your test logic here
await run_your_tests()

# Stop and analyze
snapshots = monitor.stop_monitoring()
summary = monitor.get_resource_summary()
trends = monitor.get_resource_trends()
alerts = monitor.get_alert_summary()

# Export detailed report
monitor.export_monitoring_report(Path("resource_report.json"))
```

## Integration with Existing Infrastructure

### TestEnvironmentManager Integration

```python
from test_utilities import TestEnvironmentManager, EnvironmentSpec
from performance_test_utilities import PerformanceBenchmarkSuite

# Environment with performance monitoring
env_spec = EnvironmentSpec(
    performance_monitoring=True,
    memory_limits={'test_limit': 500}
)

env_manager = TestEnvironmentManager(env_spec)
environment_data = env_manager.setup_environment()

# Use with benchmark suite
benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
```

### MockSystemFactory Integration

```python
from test_utilities import MockSystemFactory, SystemComponent, MockBehavior
from performance_test_utilities import PerformanceAssertionHelper

# Create mock system
mock_factory = MockSystemFactory()
mock_system = mock_factory.create_comprehensive_mock_set(
    [SystemComponent.LIGHTRAG_SYSTEM, SystemComponent.PDF_PROCESSOR],
    MockBehavior.SUCCESS
)

# Test with performance monitoring
assertion_helper = PerformanceAssertionHelper()

@assertion_helper.time_operation("mock_operation", 2000)
async def test_mock_operation():
    return await mock_system['lightrag_system'].aquery("test query")
```

## Standard Benchmarks

### Clinical Query Performance
- **Purpose**: Benchmark clinical query processing performance
- **Thresholds**: Response time ≤ 5s, Throughput ≥ 2 ops/sec, Error rate ≤ 5%
- **Scenarios**: Light load, Moderate load

### PDF Processing Performance
- **Purpose**: Benchmark PDF processing and ingestion performance
- **Thresholds**: Response time ≤ 15s, Throughput ≥ 0.5 ops/sec, Memory ≤ 1.2GB
- **Scenarios**: Baseline, Light load

### Scalability Benchmark
- **Purpose**: Test system behavior under increasing load
- **Thresholds**: Throughput ≥ 10 ops/sec, P95 latency ≤ 10s, Error rate ≤ 10%
- **Scenarios**: Moderate load, Heavy load, Spike test

### Endurance Benchmark
- **Purpose**: Test system stability over extended periods
- **Thresholds**: Memory ≤ 1GB (stable), Throughput ≥ 3 ops/sec, Error rate ≤ 5%
- **Scenarios**: Endurance test (10 minutes)

## Performance Thresholds

### Response Time Thresholds
```python
PerformanceThreshold('response_time_ms', 5000, 'lte', 'ms', 'error',
                     'Response time should be under 5 seconds')
```

### Throughput Thresholds
```python
PerformanceThreshold('throughput_ops_per_sec', 2.0, 'gte', 'ops/sec', 'error',
                     'Should process at least 2 operations per second')
```

### Memory Thresholds
```python
PerformanceThreshold('memory_usage_mb', 800, 'lte', 'MB', 'warning',
                     'Memory usage should be under 800MB')
```

### Error Rate Thresholds
```python
PerformanceThreshold('error_rate_percent', 5.0, 'lte', '%', 'error',
                     'Error rate should be under 5%')
```

## Pytest Fixtures

### Available Fixtures

```python
@pytest.fixture
def performance_assertion_helper():
    """Provides PerformanceAssertionHelper for tests."""

@pytest.fixture
def performance_benchmark_suite(test_environment_manager):
    """Provides PerformanceBenchmarkSuite for tests."""

@pytest.fixture
def advanced_resource_monitor():
    """Provides AdvancedResourceMonitor for tests."""

@pytest.fixture
def performance_thresholds():
    """Provides standard performance thresholds."""

@pytest.fixture
async def performance_test_with_monitoring():
    """Provides monitored test execution context."""
```

### Usage in Tests

```python
@pytest.mark.asyncio
async def test_clinical_query_performance(performance_assertion_helper,
                                        advanced_resource_monitor):
    # Establish baseline
    performance_assertion_helper.establish_memory_baseline()
    
    # Start monitoring
    advanced_resource_monitor.start_monitoring()
    
    # Execute test
    @performance_assertion_helper.time_operation("clinical_query", 5000)
    async def execute_query():
        # Your test logic
        return await process_clinical_query("test query")
    
    result, metrics = await execute_query()
    
    # Stop monitoring
    snapshots = advanced_resource_monitor.stop_monitoring()
    
    # Assertions
    performance_assertion_helper.assert_memory_leak_absent(50.0)
    assert result is not None
    assert metrics['duration_ms'] < 5000
```

## Complete Example

```python
import asyncio
from pathlib import Path
from test_utilities import create_quick_test_environment
from performance_test_utilities import (
    PerformanceAssertionHelper, AdvancedResourceMonitor, PerformanceBenchmarkSuite
)

async def comprehensive_performance_test():
    # Setup environment
    env_manager, mock_factory = create_quick_test_environment()
    
    # Initialize utilities
    assertion_helper = PerformanceAssertionHelper()
    resource_monitor = AdvancedResourceMonitor()
    benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
    
    # Establish baselines
    assertion_helper.establish_memory_baseline()
    
    # Start monitoring
    resource_monitor.start_monitoring()
    
    # Execute tests with performance tracking
    @assertion_helper.time_operation("comprehensive_test", 10000)
    async def run_comprehensive_test():
        # Your comprehensive test logic
        mock_system = mock_factory.create_comprehensive_mock_set(
            [SystemComponent.LIGHTRAG_SYSTEM], MockBehavior.SUCCESS
        )
        
        # Execute operations
        results = []
        for i in range(10):
            result = await mock_system['lightrag_system'].aquery(f"test query {i}")
            results.append(result)
            await asyncio.sleep(0.1)
        
        return results
    
    # Execute test
    test_results, test_metrics = await run_comprehensive_test()
    
    # Stop monitoring
    resource_snapshots = resource_monitor.stop_monitoring()
    
    # Performance validation
    assertion_helper.assert_throughput(len(test_results), 
                                     test_metrics['duration_ms'] / 1000, 
                                     5.0, "test_throughput")
    assertion_helper.assert_memory_leak_absent(100.0, "memory_leak_check")
    
    # Run benchmarks
    benchmark_results = await benchmark_suite.run_benchmark_suite(
        ['clinical_query_performance']
    )
    
    # Generate reports
    assertion_summary = assertion_helper.get_assertion_summary()
    resource_summary = resource_monitor.get_resource_summary()
    
    # Export results
    results_dir = Path("performance_results")
    results_dir.mkdir(exist_ok=True)
    
    assertion_helper.export_results_to_json(results_dir / "assertions.json")
    resource_monitor.export_monitoring_report(results_dir / "resources.json")
    
    # Cleanup
    env_manager.cleanup()
    
    print("Comprehensive performance test completed!")
    print(f"Assertions: {assertion_summary['passed_assertions']}/{assertion_summary['total_assertions']} passed")
    print(f"Resource alerts: {resource_monitor.get_alert_summary()['total_alerts']}")
    print(f"Benchmark success rate: {benchmark_results['suite_execution_summary']['success_rate_percent']:.1f}%")

if __name__ == "__main__":
    asyncio.run(comprehensive_performance_test())
```

## Running the Demonstration

Execute the demonstration script to see all utilities in action:

```bash
cd /Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/tests
python demo_performance_test_utilities.py
```

## Running Integration Tests

Execute the integration tests to verify everything works correctly:

```bash
cd /Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/tests
python -m pytest test_performance_utilities_integration.py -v
```

## Best Practices

### 1. Memory Management
- Always establish memory baselines before tests
- Check for memory leaks after intensive operations
- Set appropriate memory thresholds for your environment

### 2. Performance Thresholds
- Set realistic thresholds based on your system capabilities
- Use warning level for soft limits, error level for hard limits
- Adjust thresholds based on historical performance data

### 3. Resource Monitoring
- Use appropriate sampling intervals (0.5-1.0 seconds typically)
- Set conservative alert thresholds to catch issues early
- Monitor trends over time to identify gradual degradation

### 4. Benchmark Configuration
- Start with baseline scenarios before load testing
- Gradually increase load to identify breaking points
- Use endurance tests to validate stability over time

### 5. Integration Testing
- Combine all utilities for comprehensive performance validation
- Export results for analysis and historical tracking
- Use with existing test infrastructure for seamless integration

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Check for memory leaks, reduce test data size, or increase memory thresholds
2. **Assertion Failures**: Verify thresholds are appropriate for your system, check for system load
3. **Monitoring Errors**: Ensure sufficient permissions for system monitoring, check sampling intervals
4. **Benchmark Timeouts**: Reduce test duration or increase timeout thresholds

### Performance Optimization

1. **Reduce Test Duration**: Use shorter scenarios for development testing
2. **Optimize Sampling**: Adjust monitoring intervals based on test duration
3. **Batch Operations**: Group multiple operations to improve throughput measurements
4. **Resource Cleanup**: Ensure proper cleanup to prevent resource exhaustion

## Contributing

When extending these utilities:

1. Follow the existing patterns for consistency
2. Add comprehensive tests for new functionality
3. Update documentation and examples
4. Consider backward compatibility with existing tests
5. Verify integration with all existing utilities

## Support

For issues or questions about the performance utilities:

1. Check the integration tests for usage examples
2. Review the demonstration script for comprehensive examples
3. Examine the existing test infrastructure for patterns
4. Consult the performance analysis utilities for reporting features