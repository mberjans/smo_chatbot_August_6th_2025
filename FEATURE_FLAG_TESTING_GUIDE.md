# Feature Flag System - Comprehensive Testing Guide

## Overview

This document provides a comprehensive guide to the feature flag system testing suite, covering all aspects of testing from unit tests to performance benchmarks. The test suite ensures robust, reliable, and performant behavior of the feature flag system across all scenarios.

## Table of Contents

1. [Test Structure and Organization](#test-structure-and-organization)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Test Coverage Areas](#test-coverage-areas)
5. [Performance Testing](#performance-testing)
6. [Configuration and Environment](#configuration-and-environment)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)

## Test Structure and Organization

### Test Files

| File | Purpose | Coverage Areas |
|------|---------|----------------|
| `test_feature_flag_manager.py` | Core FeatureFlagManager functionality | Routing logic, hash-based assignment, circuit breaker, A/B testing |
| `test_integration_wrapper.py` | Service integration and routing | IntegratedQueryService, fallback mechanisms, error handling |
| `test_feature_flag_configuration.py` | Configuration management | Environment variable parsing, validation, defaults |
| `test_conditional_imports.py` | Module loading system | Conditional imports, graceful degradation, export management |
| `test_feature_flag_integration.py` | End-to-end workflows | Complete integration scenarios, multi-component interactions |
| `test_feature_flag_edge_cases.py` | Edge cases and error conditions | Boundary values, error handling, resource exhaustion |
| `test_feature_flag_performance.py` | Performance and stress testing | Throughput, memory usage, scalability |

### Test Class Organization

Each test file follows a consistent structure:

```python
class TestComponent:
    """Test core component functionality."""
    
    @pytest.fixture
    def component_fixture(self):
        """Set up component for testing."""
        pass
    
    def test_basic_functionality(self):
        """Test basic component behavior."""
        pass
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        pass
```

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)

**Purpose**: Test individual components in isolation

**Coverage**:
- FeatureFlagManager routing logic
- Hash calculation algorithms
- Configuration parsing and validation
- Circuit breaker state management
- Cache operations
- Performance metrics collection

**Examples**:
```bash
# Run all unit tests
python run_feature_flag_tests.py --suite unit

# Run with coverage
python run_feature_flag_tests.py --suite unit --coverage
```

### 2. Integration Tests (`@pytest.mark.integration`)

**Purpose**: Test component interactions and workflows

**Coverage**:
- Service routing end-to-end
- Fallback mechanisms
- A/B testing workflows
- Configuration-driven behavior
- Module loading integration

**Examples**:
```bash
# Run integration tests
python run_feature_flag_tests.py --suite integration

# Run with HTML reporting
python run_feature_flag_tests.py --suite integration --html-report
```

### 3. Performance Tests (`@pytest.mark.performance`)

**Purpose**: Validate performance characteristics and detect regression

**Coverage**:
- Hash calculation performance
- Routing decision throughput
- Cache efficiency
- Memory usage patterns
- Concurrent operation performance

**Examples**:
```bash
# Run performance tests
python run_feature_flag_tests.py --suite performance

# Run with memory profiling
python run_feature_flag_tests.py --suite performance --profile
```

### 4. Edge Case Tests (`@pytest.mark.edge_cases`)

**Purpose**: Test boundary conditions and error scenarios

**Coverage**:
- Boundary value testing
- Resource exhaustion
- Invalid inputs
- Network failures
- Memory limits

### 5. Stress Tests (`@pytest.mark.stress`)

**Purpose**: Test system behavior under extreme conditions

**Coverage**:
- High load scenarios
- Resource exhaustion
- Concurrent access patterns
- Memory pressure

## Running Tests

### Quick Start

```bash
# Health check
python run_feature_flag_tests.py --health-check

# Run all tests with coverage
python run_feature_flag_tests.py --suite all --coverage --html-report

# Fast unit tests only
python run_feature_flag_tests.py --suite unit --fast --parallel
```

### Test Runner Options

| Option | Description | Usage |
|--------|-------------|-------|
| `--suite` | Select test suite | `--suite unit` |
| `--coverage` | Generate coverage report | `--coverage` |
| `--parallel` | Run tests in parallel | `--parallel` |
| `--verbose` | Increase verbosity | `--verbose` |
| `--fast` | Skip slow tests | `--fast` |
| `--html-report` | Generate HTML report | `--html-report` |
| `--performance-baseline` | Run performance baselines | `--performance-baseline` |
| `--stress` | Include stress tests | `--stress` |
| `--profile` | Memory profiling | `--profile` |

### Direct pytest Usage

```bash
# Run specific test file
pytest lightrag_integration/tests/test_feature_flag_manager.py -v

# Run tests with specific marker
pytest -m "unit and not slow" -v

# Run with coverage
pytest --cov=lightrag_integration --cov-report=html

# Run performance tests only
pytest -m performance --durations=0
```

## Test Coverage Areas

### 1. FeatureFlagManager Testing

**Core Functionality**:
- ✅ Hash-based user assignment
- ✅ Rollout percentage enforcement
- ✅ A/B testing cohort assignment
- ✅ Circuit breaker behavior
- ✅ Quality threshold validation
- ✅ Conditional routing rules
- ✅ Performance metrics tracking
- ✅ Cache management

**Test Methods**:
```python
def test_hash_calculation_consistency()
def test_rollout_percentage_distribution()
def test_ab_testing_cohort_assignment()
def test_circuit_breaker_state_transitions()
def test_quality_threshold_routing()
def test_conditional_routing_evaluation()
def test_cache_hit_rate_optimization()
def test_concurrent_routing_consistency()
```

### 2. IntegratedQueryService Testing

**Service Integration**:
- ✅ Service routing decisions
- ✅ Fallback mechanism activation
- ✅ Response caching efficiency
- ✅ Error handling and recovery
- ✅ Timeout management
- ✅ Health monitoring
- ✅ A/B testing metrics collection

**Test Scenarios**:
```python
def test_lightrag_routing_complete_workflow()
def test_perplexity_routing_complete_workflow()
def test_fallback_workflow_lightrag_to_perplexity()
def test_response_caching_across_requests()
def test_concurrent_service_throughput()
def test_service_degradation_under_load()
```

### 3. Configuration Testing

**Environment Variables**:
- ✅ Boolean value parsing
- ✅ Numeric value validation
- ✅ JSON configuration parsing
- ✅ Default value handling
- ✅ Boundary value validation
- ✅ Error handling for invalid values

**Configuration Scenarios**:
```python
def test_boolean_environment_variable_parsing()
def test_numeric_environment_variables()
def test_json_environment_variables()
def test_boundary_value_analysis()
def test_contradictory_configuration_settings()
```

### 4. Conditional Import Testing

**Module Loading**:
- ✅ Feature flag-based imports
- ✅ Graceful degradation
- ✅ Export list management
- ✅ Integration status reporting
- ✅ Module availability checking

### 5. Performance and Scalability Testing

**Performance Metrics**:
- ✅ Hash calculation: >50,000 ops/sec
- ✅ Routing decisions: >10,000 ops/sec
- ✅ Memory usage: <10MB growth per 10K operations
- ✅ Cache hit rate: >80% for repeated queries
- ✅ Concurrent throughput: >5,000 ops/sec

**Stress Testing**:
- ✅ 100,000+ user simulation
- ✅ Sustained load over 10+ seconds
- ✅ Memory pressure scenarios
- ✅ Resource exhaustion handling

## Performance Testing

### Performance Benchmarks

| Metric | Target | Test Method |
|--------|--------|-------------|
| Hash Calculation | >50,000 ops/sec | `test_hash_calculation_performance` |
| Routing Decisions | >10,000 ops/sec | `test_routing_decision_performance` |
| Concurrent Routing | >5,000 ops/sec | `test_concurrent_routing_performance` |
| Memory Growth | <10MB/10K ops | `test_memory_usage_under_load` |
| Cache Hit Rate | >80% | `test_routing_cache_performance` |

### Running Performance Tests

```bash
# Basic performance suite
python run_feature_flag_tests.py --suite performance

# Performance with baseline comparison
python run_feature_flag_tests.py --suite performance --performance-baseline

# Performance with memory profiling
python run_feature_flag_tests.py --suite performance --profile

# Stress testing
python run_feature_flag_tests.py --suite performance --stress
```

### Performance Analysis

Performance test results include:
- **Response Time Distribution**: Mean, median, P95, P99
- **Throughput Analysis**: Operations per second over time
- **Memory Usage Patterns**: Growth rates and peak usage
- **Cache Efficiency**: Hit rates and optimization opportunities
- **Scalability Metrics**: Performance vs. load characteristics

## Configuration and Environment

### Test Environment Variables

```bash
# Core feature flags
LIGHTRAG_INTEGRATION_ENABLED=true
LIGHTRAG_ENABLE_QUALITY_VALIDATION=true
LIGHTRAG_ENABLE_RELEVANCE_SCORING=true
LIGHTRAG_ENABLE_AB_TESTING=true
LIGHTRAG_ENABLE_CIRCUIT_BREAKER=true

# Test-specific settings
LIGHTRAG_ROLLOUT_PERCENTAGE=50.0
LIGHTRAG_USER_HASH_SALT=test_salt_2025
LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
LIGHTRAG_MIN_QUALITY_THRESHOLD=0.7

# Mock API keys for testing
OPENAI_API_KEY=test-openai-key-for-feature-flag-tests
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
```

### Test Configuration Files

- **pytest.ini**: Pytest configuration with markers, timeouts, and reporting
- **conftest.py**: Shared fixtures and test utilities
- **run_feature_flag_tests.py**: Comprehensive test runner script

### Dependencies

**Required**:
- pytest >= 6.0
- pytest-asyncio
- pytest-mock
- unittest.mock (built-in)

**Optional** (for enhanced features):
- pytest-cov (coverage reporting)
- pytest-html (HTML reports)
- pytest-xdist (parallel execution)
- pytest-memray (memory profiling)
- pytest-benchmark (performance benchmarking)

## Test Data and Fixtures

### Shared Fixtures

```python
@pytest.fixture
def mock_config():
    """Standard mock configuration for tests."""
    
@pytest.fixture
def feature_manager():
    """Configured FeatureFlagManager instance."""
    
@pytest.fixture
def integrated_service():
    """Full IntegratedQueryService with mocks."""
    
@pytest.fixture
def performance_metrics():
    """Performance metrics collection utility."""
```

### Test Data Builders

```python
class TestDataBuilder:
    """Builder for consistent test data."""
    
    @staticmethod
    def create_routing_context(**overrides):
        """Create routing context with defaults."""
    
    @staticmethod
    def create_query_request(**overrides):
        """Create query request with defaults."""
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: ModuleNotFoundError for lightrag_integration components
**Solution**: 
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025:$PYTHONPATH

# Or use the test runner
python run_feature_flag_tests.py --health-check
```

#### 2. Async Test Failures

**Problem**: Async tests hanging or failing
**Solution**:
- Ensure pytest-asyncio is installed
- Use `@pytest.mark.asyncio` decorator
- Check event loop configuration in conftest.py

#### 3. Performance Test Variability

**Problem**: Performance tests failing intermittently
**Solution**:
- Run tests multiple times for statistical significance
- Use performance baselines with tolerance
- Consider system load during testing

#### 4. Memory Test Failures

**Problem**: Memory usage tests failing on different systems
**Solution**:
- Adjust memory thresholds for test environment
- Run garbage collection explicitly
- Use relative memory growth measurements

### Debugging Tips

1. **Verbose Output**: Use `-vv` for detailed test output
2. **Specific Tests**: Run individual test methods for isolation
3. **Logging**: Check test logs in `logs/pytest.log`
4. **Coverage Reports**: Use coverage to identify untested code paths
5. **Performance Profiling**: Use `--profile` to identify bottlenecks

### Test Environment Validation

```bash
# Quick health check
python run_feature_flag_tests.py --health-check

# List available tests
python run_feature_flag_tests.py --list-tests

# Validate test environment
pytest lightrag_integration/tests/test_conditional_imports.py::TestFeatureFlagLoading -v
```

## Contributing

### Adding New Tests

1. **Follow naming conventions**: `test_*.py` for files, `test_*` for functions
2. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
3. **Include docstrings**: Document test purpose and coverage
4. **Add fixtures**: Create reusable test fixtures when appropriate
5. **Update documentation**: Add new tests to this guide

### Test Quality Guidelines

1. **Isolation**: Tests should not depend on each other
2. **Repeatability**: Tests should produce consistent results
3. **Clear assertions**: Use descriptive assertion messages
4. **Performance awareness**: Consider test execution time
5. **Mock appropriately**: Mock external dependencies

### Code Coverage Goals

- **Overall coverage**: >90%
- **Critical paths**: 100% (routing logic, error handling)
- **Edge cases**: >85%
- **Performance code**: >80%

### Performance Test Standards

- **Benchmark consistency**: ±10% variance acceptable
- **Memory leaks**: Zero tolerance for significant leaks
- **Scalability**: Linear or sub-linear performance degradation
- **Regression prevention**: Automated baseline comparison

## Reporting and Metrics

### Generated Reports

1. **JUnit XML**: Machine-readable test results
2. **HTML Report**: Visual test results and failure analysis
3. **Coverage Report**: Code coverage with line-by-line analysis
4. **Performance Report**: Benchmark results and trends
5. **Memory Profile**: Memory usage patterns and optimization opportunities

### Key Metrics Tracked

- **Test Coverage**: Line, branch, and function coverage
- **Performance Benchmarks**: Response times and throughput
- **Memory Usage**: Growth patterns and peak usage
- **Error Rates**: Failure rates under various conditions
- **Scalability**: Performance vs. load characteristics

## Continuous Integration

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Feature Flag Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-cov pytest-html pytest-xdist
      - name: Run tests
        run: python run_feature_flag_tests.py --suite all --coverage --parallel --html-report
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Quality Gates

- **All unit tests must pass**
- **Integration tests must pass**
- **Coverage must be >85%**
- **Performance benchmarks must meet targets**
- **No memory leaks detected**

This comprehensive testing guide ensures robust, reliable, and performant feature flag system behavior across all deployment scenarios and usage patterns.