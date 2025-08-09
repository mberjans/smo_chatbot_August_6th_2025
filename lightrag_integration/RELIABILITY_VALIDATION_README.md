# Reliability Validation System - CMO-LIGHTRAG-014-T08

## Overview

This comprehensive reliability validation system addresses Task CMO-LIGHTRAG-014-T08 by designing and implementing detailed test scenarios to validate the reliability of the Clinical Metabolomics Oracle fallback system. The system tests 5 key reliability areas identified in the analysis of the existing fallback system (94.1% success rate).

## System Architecture

### Fallback Chain
- **Primary**: LightRAG knowledge retrieval
- **Secondary**: Perplexity API fallback  
- **Tertiary**: Cached response retrieval
- **Final**: Default structured response

### Load Management
- **Progressive Degradation**: NORMAL → ELEVATED → HIGH → CRITICAL → EMERGENCY
- **Circuit Breakers**: For all external services with configurable thresholds
- **Request Throttling**: Token bucket algorithm with priority queuing
- **Queue Management**: Anti-starvation with priority-based processing

## Test Coverage

### 1. Stress Testing & Load Limits (ST-001 to ST-004)
- **ST-001**: Progressive Load Escalation - Tests system response to gradually increasing load
- **ST-002**: Burst Load Handling - Validates resilience to sudden traffic spikes  
- **ST-003**: Memory Pressure Endurance - Tests behavior under sustained memory pressure
- **ST-004**: Maximum Concurrent Request Handling - Determines capacity limits

### 2. Network Reliability (NR-001 to NR-004)
- **NR-001**: LightRAG Service Degradation - Tests fallback when primary service degrades
- **NR-002**: Perplexity API Reliability - Validates behavior when secondary service fails
- **NR-003**: Complete External Service Outage - Tests cache-only operation
- **NR-004**: Variable Network Latency - Tests adaptation to network conditions

### 3. Data Integrity & Consistency (DI-001 to DI-003)
- **DI-001**: Cross-Source Response Consistency - Validates response quality across sources
- **DI-002**: Cache Freshness and Accuracy - Tests cache management mechanisms
- **DI-003**: Malformed Response Recovery - Tests handling of corrupted responses

### 4. Production Scenario Testing (PS-001 to PS-003)
- **PS-001**: Peak Hour Load Simulation - Simulates realistic usage patterns
- **PS-002**: Multi-User Concurrent Sessions - Tests with varied user behavior
- **PS-003**: Production System Integration - Validates integration points

### 5. Integration Reliability (IR-001 to IR-003)
- **IR-001**: Circuit Breaker Threshold Validation - Tests circuit breaker behavior
- **IR-002**: Cascading Failure Prevention - Validates isolation mechanisms
- **IR-003**: Automatic Recovery Validation - Tests recovery mechanisms

## Files Structure

```
lightrag_integration/
├── RELIABILITY_VALIDATION_TEST_SCENARIOS_CMO_LIGHTRAG_014_T08.md
├── RELIABILITY_VALIDATION_README.md
├── run_reliability_validation_tests.py
├── demo_reliability_validation.py
└── tests/
    ├── reliability_test_framework.py
    ├── test_stress_testing_scenarios.py
    └── test_network_reliability_scenarios.py
```

### Key Files

- **`RELIABILITY_VALIDATION_TEST_SCENARIOS_CMO_LIGHTRAG_014_T08.md`**: Comprehensive test design document
- **`run_reliability_validation_tests.py`**: Main test execution runner with CLI interface
- **`demo_reliability_validation.py`**: Demonstration script showing system capabilities
- **`tests/reliability_test_framework.py`**: Core testing framework and utilities
- **`tests/test_stress_testing_scenarios.py`**: Implementation of ST-001 to ST-004
- **`tests/test_network_reliability_scenarios.py`**: Implementation of NR-001 to NR-004

## Quick Start

### 1. Run the Demo
```bash
python demo_reliability_validation.py
```

### 2. Run Complete Test Suite
```bash
python run_reliability_validation_tests.py
```

### 3. Run Specific Categories
```bash
# Stress testing only
python run_reliability_validation_tests.py --categories stress_testing

# Multiple categories
python run_reliability_validation_tests.py --categories stress_testing network_reliability
```

### 4. Quick Tests (Exclude Long-Running)
```bash
python run_reliability_validation_tests.py --quick
```

## Usage Examples

### Basic Framework Usage

```python
from tests.reliability_test_framework import (
    ReliabilityValidationFramework,
    ReliabilityTestConfig
)

# Create custom configuration
config = ReliabilityTestConfig(
    base_rps=10.0,
    max_rps=500.0,
    min_success_rate=0.85,
    max_response_time_ms=3000.0
)

# Initialize framework
framework = ReliabilityValidationFramework(config)

# Setup test environment
await framework.setup_test_environment()

# Define and run test
async def my_reliability_test(orchestrator, config):
    # Your test implementation
    pass

result = await framework.execute_monitored_test(
    test_name="my_test",
    test_func=my_reliability_test,
    category="custom"
)

# Cleanup
await framework.cleanup_test_environment()
```

### Running Individual Test Scenarios

```python
from tests.test_stress_testing_scenarios import test_progressive_load_escalation
from tests.test_network_reliability_scenarios import test_lightrag_service_degradation

# Run specific tests
config = ReliabilityTestConfig()
orchestrator = await create_test_orchestrator(config)

# Stress test
results = await test_progressive_load_escalation(orchestrator, config)

# Network reliability test  
results = await test_lightrag_service_degradation(orchestrator, config)
```

### Using pytest

```bash
# Run stress testing scenarios
python -m pytest tests/test_stress_testing_scenarios.py -v

# Run network reliability scenarios
python -m pytest tests/test_network_reliability_scenarios.py -v

# Run specific test
python -m pytest tests/test_stress_testing_scenarios.py::test_st_001_progressive_load_escalation -v
```

## Configuration

### ReliabilityTestConfig Parameters

```python
@dataclass
class ReliabilityTestConfig:
    # Test execution settings
    max_test_duration_minutes: int = 30
    isolation_recovery_time_seconds: int = 10
    monitoring_interval_seconds: float = 1.0
    
    # System thresholds
    min_success_rate: float = 0.85
    max_response_time_ms: float = 5000.0
    max_memory_usage_percentage: float = 0.90
    max_cpu_usage_percentage: float = 0.90
    
    # Load testing settings
    base_rps: float = 10.0
    max_rps: float = 1000.0
    ramp_up_time_seconds: int = 60
```

### Custom Configuration File

```json
{
    "base_rps": 20.0,
    "max_rps": 2000.0,
    "min_success_rate": 0.90,
    "max_response_time_ms": 2000.0,
    "max_test_duration_minutes": 45,
    "failure_injection_enabled": true
}
```

Use with: `python run_reliability_validation_tests.py --config custom_config.json`

## Command Line Options

### Main Runner Options

```bash
python run_reliability_validation_tests.py [OPTIONS]

Options:
  --categories CATEGORIES [CATEGORIES ...]
                        Test categories to run (stress_testing, network_reliability, 
                        data_integrity, production_scenarios, integration_reliability)
  --quick               Run quick tests only (exclude long-running tests)
  --parallel            Run test categories in parallel (experimental)
  --config CONFIG       Path to custom test configuration JSON file
  --output-dir OUTPUT_DIR
                        Directory to save test reports (default: current directory)
  --verbose             Enable verbose logging
  --dry-run             Show what would be run without executing tests
  -h, --help           show this help message and exit
```

### Usage Examples

```bash
# Run all tests
python run_reliability_validation_tests.py

# Run only critical tests
python run_reliability_validation_tests.py --categories stress_testing network_reliability

# Quick validation (30-45 minutes instead of 2-3 hours)
python run_reliability_validation_tests.py --quick

# With custom configuration
python run_reliability_validation_tests.py --config production_config.json

# Verbose logging
python run_reliability_validation_tests.py --verbose

# Dry run to see what would execute
python run_reliability_validation_tests.py --dry-run
```

## Test Results and Reporting

### Report Structure

The system generates comprehensive JSON reports with:

```json
{
  "execution_summary": {
    "start_time": "2025-08-09T10:30:00",
    "end_time": "2025-08-09T12:15:00", 
    "total_duration_minutes": 105,
    "categories_executed": 5,
    "total_tests": 15,
    "passed_tests": 14,
    "overall_success_rate": 0.93,
    "overall_reliability_score": 89.5
  },
  "category_results": { /* detailed results */ },
  "recommendations": [ /* improvement suggestions */ ],
  "risk_assessment": { /* risk analysis */ },
  "compliance_status": { /* compliance checks */ }
}
```

### Success Criteria

- **Minimum Reliability Threshold**: 85% overall success rate
- **Production Readiness**: 90% reliability score  
- **High Availability Standard**: 95% reliability score
- **Response Time**: P95 < 5 seconds under normal load
- **Memory Usage**: Peak < 90% available memory
- **Recovery Time**: < 30 seconds for most failure scenarios

### Reliability Scoring

The system calculates an overall reliability score (0-100%) based on weighted categories:

- **Stress Testing**: 25% weight
- **Network Reliability**: 25% weight  
- **Data Integrity**: 20% weight
- **Production Scenarios**: 20% weight
- **Integration Reliability**: 10% weight

## Failure Injection Mechanisms

### Supported Failure Types

1. **Service Outages**: Complete service unavailability
2. **Network Latency**: Variable network delays and jitter
3. **Memory Pressure**: Controlled memory usage simulation
4. **Rate Limiting**: API rate limit simulation
5. **Intermittent Failures**: Partial service degradation
6. **Timeout Conditions**: Request timeout simulation

### Example Usage

```python
from tests.reliability_test_framework import (
    ServiceOutageInjector,
    NetworkLatencyInjector,
    MemoryPressureInjector
)

# Service outage
async with ServiceOutageInjector('lightrag', orchestrator).failure_context():
    # Run tests with LightRAG unavailable
    pass

# Network latency
async with NetworkLatencyInjector(delay_ms=1000, jitter_ms=200).failure_context():
    # Run tests with high network latency
    pass

# Memory pressure  
async with MemoryPressureInjector(target_usage_percentage=0.85).failure_context():
    # Run tests under memory pressure
    pass
```

## Performance Monitoring

### Metrics Collected

- **Response Time Percentiles**: P50, P95, P99, P99.9
- **Success Rates**: Overall and per-endpoint
- **System Resources**: CPU, Memory, Network I/O
- **Queue Depths**: Request queue and connection pool metrics
- **Circuit Breaker States**: Activation and recovery events
- **Fallback Usage**: Distribution across fallback sources

### Real-Time Monitoring

```python
from tests.reliability_test_framework import ReliabilityTestMonitor

monitor = ReliabilityTestMonitor(collection_interval=1.0)
await monitor.start_monitoring()

# Run your tests
# ...

metrics = await monitor.stop_monitoring()
print(f"Average CPU: {metrics['avg_cpu_percent']:.1f}%")
print(f"Peak Memory: {metrics['max_memory_percent']:.1f}%")
```

## Integration with Existing Systems

### Test Framework Integration

The reliability validation system integrates with the existing test framework:

- **Builds on existing test utilities** in `lightrag_integration/tests/`
- **Uses existing orchestrator interfaces** from graceful degradation system
- **Leverages existing monitoring components** from production systems
- **Follows established test patterns** and naming conventions

### Production Integration

For production deployment:

1. **Validate reliability score >90%** before deployment
2. **Run critical tests** (stress + network) in CI/CD pipeline  
3. **Monitor reliability metrics** in production
4. **Set up alerts** for reliability degradation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Test Timeouts**: Increase timeout values in configuration for slower systems  
3. **Memory Issues**: Reduce concurrent request limits or memory pressure levels
4. **Permission Errors**: Ensure sufficient system permissions for resource monitoring

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python run_reliability_validation_tests.py --verbose
```

### Mock Mode

When real system components aren't available, the framework automatically falls back to mock implementations for testing the framework itself.

## Contributing

### Adding New Test Scenarios

1. **Create test function** following the pattern:
   ```python
   async def test_new_scenario(orchestrator, config: ReliabilityTestConfig):
       # Test implementation
       pass
   ```

2. **Add pytest wrapper**:
   ```python
   @pytest.mark.asyncio
   async def test_new_scenario_wrapper():
       # Pytest integration
       pass
   ```

3. **Update test runner** to include new scenario

### Extending Failure Injection

1. **Inherit from FailureInjector**:
   ```python
   class MyFailureInjector(FailureInjector):
       async def inject_failure(self):
           # Custom failure logic
           pass
   ```

2. **Implement context manager** for easy usage

3. **Add to test scenarios** where appropriate

## License and Support

This reliability validation system is part of the Clinical Metabolomics Oracle project. For support or questions about the implementation, refer to the comprehensive design document `RELIABILITY_VALIDATION_TEST_SCENARIOS_CMO_LIGHTRAG_014_T08.md`.

## Appendix

### Exit Codes

- **0**: All tests passed, reliability score ≥90%
- **1**: Some tests passed, reliability score ≥75%  
- **2**: Significant failures, reliability score <75%
- **130**: Interrupted by user (Ctrl+C)

### Performance Benchmarks

Expected performance on standard hardware:
- **Complete Test Suite**: 2-3 hours
- **Critical Tests Only**: 30-45 minutes  
- **Quick Tests**: 15-30 minutes
- **Individual Scenario**: 5-15 minutes

### Resource Requirements

- **CPU**: 2+ cores recommended for concurrent testing
- **Memory**: 8GB+ recommended (tests include memory pressure scenarios)
- **Disk**: 1GB for test data and logs
- **Network**: Internet access for external service testing