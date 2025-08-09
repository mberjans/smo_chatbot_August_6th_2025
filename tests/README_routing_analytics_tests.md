# Routing Decision Analytics Test Suite

This directory contains comprehensive unit and integration tests for the routing decision logging and analytics functionality implemented for the Clinical Metabolomics Oracle system.

## Overview

The test suite provides complete coverage for:

- **RoutingDecisionLogger**: Multi-strategy logging system with file, memory, and hybrid storage
- **RoutingAnalytics**: Real-time analytics, metrics aggregation, and anomaly detection
- **EnhancedProductionRouter**: Integration of logging with production routing system
- **Configuration Management**: Environment-based configuration and feature flags
- **Performance Monitoring**: Overhead tracking and performance impact measurement
- **Error Handling**: Comprehensive error scenarios and recovery testing

## Test Structure

### Core Test Files

- `test_routing_decision_analytics.py` - Comprehensive unit tests for logging and analytics
- `test_enhanced_production_router_integration.py` - Integration tests with production router
- `conftest_routing_analytics.py` - Shared fixtures and test utilities
- `pytest_routing_analytics.ini` - Pytest configuration
- `test_requirements_routing_analytics.txt` - Test dependencies

### Test Categories

#### 1. Unit Tests (`TestRoutingDecisionLogger`, `TestRoutingAnalytics`)
- Storage strategy testing (memory, file, hybrid)
- Async logging and batching functionality
- Configuration management and environment variables
- Log entry creation and serialization
- Real-time metrics collection and aggregation
- Anomaly detection algorithms
- Performance monitoring and overhead tracking

#### 2. Integration Tests (`TestEnhancedRouterIntegration`)
- End-to-end routing with logging integration
- Performance impact measurement
- Concurrent access and thread safety
- Production environment simulation
- Error handling and recovery scenarios

#### 3. Production Simulation (`TestProductionEnvironmentSimulation`)
- High-volume production load testing
- Realistic query patterns and timing
- Gradual performance degradation scenarios
- Production configuration validation
- Error recovery in production environment

## Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install -r tests/test_requirements_routing_analytics.txt

# Ensure the main routing analytics modules are available
pip install -e .
```

### Running Tests

#### Run All Tests
```bash
# From project root
pytest tests/test_routing_decision_analytics.py tests/test_enhanced_production_router_integration.py -v

# With coverage report
pytest tests/test_routing_decision_analytics.py tests/test_enhanced_production_router_integration.py --cov=lightrag_integration --cov-report=html
```

#### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_routing_decision_analytics.py -m "not integration"

# Integration tests only  
pytest tests/test_enhanced_production_router_integration.py -m integration

# Performance tests
pytest tests/ -m performance

# Async tests only
pytest tests/ -m async_test
```

#### Run Tests with Different Storage Strategies
```bash
# Test all storage strategies (parametrized)
pytest tests/test_routing_decision_analytics.py::TestRoutingDecisionLogger -v

# Test specific configuration
ROUTING_STORAGE_STRATEGY=hybrid pytest tests/test_routing_decision_analytics.py
```

### Test Configuration

#### Environment Variables
```bash
# Logging configuration
export ROUTING_LOGGING_ENABLED=true
export ROUTING_LOG_LEVEL=detailed
export ROUTING_STORAGE_STRATEGY=hybrid
export ROUTING_LOG_DIR=/tmp/test_logs
export ROUTING_ASYNC_LOGGING=true
export ROUTING_REAL_TIME_ANALYTICS=true

# Run tests with custom configuration
pytest tests/ -v
```

#### pytest.ini Configuration
The test suite uses `pytest_routing_analytics.ini` for configuration:

```ini
[tool:pytest]
testpaths = tests
addopts = --strict-markers --verbose --asyncio-mode=auto
markers = 
    slow: marks tests as slow
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    async_test: marks tests that require async support
```

## Test Features and Coverage

### 1. Storage Strategy Testing

Tests all three storage strategies comprehensively:

```python
# Memory-only storage
@pytest.mark.parametrize("storage_strategy", [StorageStrategy.MEMORY_ONLY])
def test_memory_storage(storage_strategy):
    # Test in-memory logging with retention policies

# File-only storage  
@pytest.mark.parametrize("storage_strategy", [StorageStrategy.FILE_ONLY])
def test_file_storage(storage_strategy):
    # Test file-based logging with rotation and compression

# Hybrid storage
@pytest.mark.parametrize("storage_strategy", [StorageStrategy.HYBRID])
def test_hybrid_storage(storage_strategy):
    # Test combined memory and file storage
```

### 2. Async Logging and Batching

Comprehensive async testing with proper cleanup:

```python
@pytest.mark.asyncio
async def test_async_logging_with_batching():
    # Test batching behavior
    # Test timeout-based flushing
    # Test graceful shutdown
```

### 3. Performance Impact Testing

Measures logging overhead and system performance:

```python
def test_logging_performance_impact():
    # Measure routing time with/without logging
    # Verify overhead stays within acceptable limits
    # Test concurrent access performance
```

### 4. Anomaly Detection Testing

Tests various anomaly scenarios:

```python
def test_anomaly_detection():
    # Confidence degradation detection
    # Slow decision detection  
    # High error rate detection
    # Custom threshold testing
```

### 5. Production Environment Simulation

Realistic production scenarios:

```python
@pytest.mark.asyncio
async def test_production_load_simulation():
    # High-volume query processing
    # Realistic query patterns
    # Performance monitoring under load
    # Resource usage tracking
```

## Mock Objects and Fixtures

### Core Fixtures

- `sample_routing_prediction` - Realistic routing prediction objects
- `mock_system_state` - Complete system state for logging
- `temp_log_dir` - Isolated temporary directories for file testing
- `enhanced_router` - Fully configured enhanced router for integration testing

### Test Data Generation

```python
@pytest.fixture
def test_data_generator():
    """Generates realistic test data"""
    return TestDataGenerator()

# Usage in tests
def test_with_generated_data(test_data_generator):
    predictions = test_data_generator.create_routing_predictions(100)
    # Test with realistic data variety
```

### Environment Management

```python
@pytest.fixture
def mock_env():
    """Manages environment variables for testing"""
    yield MockEnvironmentManager()
    # Automatic cleanup

# Usage
def test_environment_config(mock_env):
    mock_env.set_routing_env_vars(
        logging_enabled='true',
        log_level='debug',
        storage_strategy='hybrid'
    )
    # Test with custom environment
```

## Performance Testing

### Benchmarks and Profiling

```python
@pytest.mark.performance
def test_logging_performance_benchmark():
    """Benchmark logging performance under various conditions"""
    # Measure throughput
    # Memory usage tracking
    # Latency percentiles
    # Resource utilization
```

### Load Testing

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_logging_load():
    """Test system under concurrent load"""
    # 100+ concurrent requests
    # Thread safety verification
    # Resource leak detection
```

## Error Handling and Edge Cases

### Comprehensive Error Scenarios

```python
def test_error_recovery():
    # File system errors
    # Permission errors
    # Disk space issues
    # Network failures
    # Memory exhaustion
    # Invalid data handling
```

### Configuration Edge Cases

```python
def test_configuration_edge_cases():
    # Invalid log levels
    # Missing directories
    # Conflicting settings
    # Malformed environment variables
```

## Debugging and Troubleshooting

### Running Individual Tests

```bash
# Run single test method
pytest tests/test_routing_decision_analytics.py::TestRoutingDecisionLogger::test_file_logging -v

# Run with detailed output
pytest tests/test_routing_decision_analytics.py::TestRoutingAnalytics::test_anomaly_detection -vvs

# Drop into debugger on failure
pytest tests/test_routing_decision_analytics.py --pdb
```

### Debug Configuration

```bash
# Enable debug logging
PYTHONPATH=. pytest tests/ --log-cli-level=DEBUG

# Capture stdout
pytest tests/ -s

# Show local variables on failure
pytest tests/ --tb=long --showlocals
```

### Temporary File Inspection

```python
def test_debug_file_output(temp_log_dir):
    """Inspect temporary files for debugging"""
    # Log files are in temp_log_dir
    # Set breakpoint to inspect
    import pdb; pdb.set_trace()
```

## Coverage and Quality

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=lightrag_integration --cov-report=html

# View coverage report
open htmlcov/index.html

# Fail if coverage below threshold
pytest tests/ --cov=lightrag_integration --cov-fail-under=85
```

### Code Quality Checks

```bash
# Type checking
mypy lightrag_integration/routing_decision_analytics.py

# Linting
flake8 lightrag_integration/routing_decision_analytics.py tests/

# Formatting
black lightrag_integration/routing_decision_analytics.py tests/
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Routing Analytics Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
        storage-strategy: [memory_only, file_only, hybrid]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -r tests/test_requirements_routing_analytics.txt
      - name: Run tests
        env:
          ROUTING_STORAGE_STRATEGY: ${{ matrix.storage-strategy }}
        run: pytest tests/ --cov=lightrag_integration --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Contributing to Tests

### Adding New Tests

1. **Unit Tests**: Add to `test_routing_decision_analytics.py`
2. **Integration Tests**: Add to `test_enhanced_production_router_integration.py`
3. **Fixtures**: Add shared fixtures to `conftest_routing_analytics.py`

### Test Writing Guidelines

```python
class TestNewFeature:
    """Test new feature with comprehensive coverage"""
    
    def test_happy_path(self):
        """Test normal operation"""
        pass
    
    def test_edge_cases(self):
        """Test boundary conditions"""
        pass
    
    def test_error_handling(self):
        """Test error scenarios"""
        pass
    
    @pytest.mark.asyncio
    async def test_async_behavior(self):
        """Test async functionality"""
        pass
    
    @pytest.mark.performance
    def test_performance(self):
        """Test performance characteristics"""
        pass
```

### Best Practices

1. **Descriptive test names** - Test purpose should be clear from name
2. **Isolated tests** - Each test should be independent
3. **Proper cleanup** - Use fixtures for setup/teardown
4. **Realistic data** - Use representative test data
5. **Error testing** - Test both success and failure scenarios
6. **Performance awareness** - Monitor test execution time
7. **Documentation** - Document complex test scenarios

## Troubleshooting Common Issues

### Test Failures

```bash
# Permission errors with temp directories
chmod -R 755 /tmp/routing_analytics_tests_*

# Import errors
export PYTHONPATH=/path/to/project:$PYTHONPATH

# Async test failures
pip install pytest-asyncio==0.21.0
```

### Performance Issues

```bash
# Run tests in parallel
pip install pytest-xdist
pytest tests/ -n auto

# Skip slow tests during development
pytest tests/ -m "not slow"
```

### Memory Issues

```bash
# Monitor memory usage
pytest tests/ --memory-profile

# Run individual test classes
pytest tests/test_routing_decision_analytics.py::TestRoutingDecisionLogger
```

## Future Enhancements

### Planned Test Additions

1. **Database Integration Tests** - When persistence layer is added
2. **Distributed Logging Tests** - For multi-node deployments  
3. **Security Testing** - Data anonymization and access control
4. **Compliance Testing** - Audit trail and regulatory requirements
5. **Performance Regression Tests** - Automated performance monitoring

### Test Infrastructure Improvements

1. **Docker-based Testing** - Consistent test environments
2. **Property-based Testing** - Hypothesis-driven test generation
3. **Mutation Testing** - Test quality verification
4. **Load Testing Framework** - Systematic performance testing
5. **Visual Test Reports** - Enhanced test result visualization

---

For questions or issues with the test suite, please refer to the main project documentation or create an issue in the project repository.