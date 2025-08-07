# Performance Benchmarking Test Suite

This directory contains comprehensive unit tests for all performance benchmarking utilities in the Clinical Metabolomics Oracle project.

## Overview

The test suite provides thorough testing coverage for:

1. **Quality Performance Benchmarks** (`quality_performance_benchmarks.py`)
2. **Performance Correlation Engine** (`performance_correlation_engine.py`) 
3. **Quality Aware Metrics Logger** (`quality_aware_metrics_logger.py`)
4. **Quality Performance Reporter** (`reporting/quality_performance_reporter.py`)

## Test Structure

### Test Files

- `test_quality_performance_benchmarks.py` - Tests for benchmark suite functionality
- `test_performance_correlation_engine.py` - Tests for correlation analysis
- `test_quality_aware_metrics_logger.py` - Tests for metrics logging
- `test_quality_performance_reporter.py` - Tests for report generation
- `conftest.py` - Shared pytest fixtures and configuration
- `pytest.ini` - Pytest configuration
- `requirements_test.txt` - Testing dependencies
- `run_all_tests.py` - Comprehensive test runner script

### Test Categories

Tests are organized into several categories using pytest markers:

- **Unit Tests** (`@pytest.mark.unit`) - Test individual components in isolation
- **Integration Tests** (`@pytest.mark.integration`) - Test component interactions
- **Performance Tests** (`@pytest.mark.performance`) - Validate test performance
- **Benchmark Tests** (`@pytest.mark.benchmark`) - Performance benchmarking
- **Slow Tests** (`@pytest.mark.slow`) - Tests that take longer to run

## Quick Start

### 1. Install Dependencies

```bash
# Install testing dependencies
pip install -r requirements_test.txt
```

### 2. Run All Tests

```bash
# Run the comprehensive test suite
python run_all_tests.py --verbose --coverage

# Or use pytest directly
pytest -v --cov
```

### 3. Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit -v

# Run integration tests
pytest -m integration -v

# Run performance tests
pytest -m performance -v
```

## Test Runner Options

The `run_all_tests.py` script provides comprehensive testing capabilities:

```bash
# Basic usage
python run_all_tests.py

# With coverage report
python run_all_tests.py --coverage --html-report

# Run all test types
python run_all_tests.py --all --verbose

# Parallel execution
python run_all_tests.py --parallel --coverage

# Custom output directory
python run_all_tests.py --output-dir custom_reports
```

### Available Options

- `--verbose, -v` - Verbose test output
- `--coverage, -c` - Generate coverage report  
- `--html-report` - Generate HTML coverage report
- `--integration` - Include integration tests
- `--performance` - Include performance validation tests
- `--parallel, -p` - Run tests in parallel
- `--output-dir` - Directory for test reports
- `--all, -a` - Run all test types

## Test Coverage

The test suite aims for comprehensive coverage of:

### Functional Testing
- ✅ All public methods and functions
- ✅ Configuration and initialization 
- ✅ Data loading and processing
- ✅ Metrics calculation and analysis
- ✅ Report generation and export
- ✅ API interactions and logging

### Error Handling
- ✅ Invalid input handling
- ✅ Network failures and timeouts
- ✅ File I/O errors
- ✅ Memory constraints
- ✅ Malformed data processing
- ✅ Missing dependencies

### Edge Cases  
- ✅ Empty datasets
- ✅ Extreme values (infinity, negative, zero)
- ✅ Large datasets and memory management
- ✅ Concurrent operations
- ✅ Resource constraints
- ✅ Time-based operations

### Integration Scenarios
- ✅ Component interactions
- ✅ End-to-end workflows
- ✅ Data pipeline testing
- ✅ Cross-system correlations
- ✅ Report generation with real data

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

The test suite provides comprehensive fixtures for:

- **Sample Data**: Quality metrics, API metrics, correlation data
- **Mock Objects**: HTTP responses, quality validation components
- **Temporary Resources**: Directories, files, databases
- **Configuration**: Test configurations and thresholds
- **Performance Data**: Baseline metrics and benchmarks

### Test Data Patterns

```python
# Using fixtures in tests
def test_example(sample_quality_metrics, temp_dir, mock_http_responses):
    # Test implementation using provided fixtures
    pass

# Parametrized tests for multiple scenarios
@pytest.mark.parametrize("accuracy,expected", [
    (85.0, "pass"),
    (75.0, "fail"), 
    (95.0, "excellent")
])
def test_accuracy_thresholds(accuracy, expected):
    # Test with different accuracy values
    pass
```

## Test Organization

### Quality Performance Benchmarks Tests

**File**: `test_quality_performance_benchmarks.py`

- `TestQualityValidationMetrics` - Metrics calculation and validation
- `TestQualityPerformanceThreshold` - Threshold checking functionality
- `TestQualityBenchmarkConfiguration` - Configuration management
- `TestQualityValidationBenchmarkSuite` - Benchmark execution
- `TestQualityBenchmarkIntegration` - Integration scenarios
- `TestErrorHandlingAndEdgeCases` - Error handling and edge cases

Key test areas:
- Metric calculation accuracy
- Threshold validation
- Benchmark execution workflows
- Resource monitoring
- Error recovery
- Performance analysis

### Performance Correlation Engine Tests

**File**: `test_performance_correlation_engine.py`

- `TestPerformanceCorrelationMetrics` - Correlation calculations
- `TestQualityPerformanceCorrelation` - Correlation analysis
- `TestPerformancePredictionModel` - Machine learning models
- `TestCrossSystemCorrelationEngine` - Correlation engine functionality
- `TestCorrelationAnalysisReport` - Report generation
- `TestIntegrationAndConvenienceFunctions` - Integration testing

Key test areas:
- Statistical correlation analysis
- Prediction model training and validation
- Cross-system data integration
- Optimization recommendations
- Report generation

### Quality Aware Metrics Logger Tests

**File**: `test_quality_aware_metrics_logger.py`

- `TestQualityAPIMetric` - API metric data structures
- `TestQualityMetricsAggregator` - Metrics aggregation
- `TestQualityAwareAPIMetricsLogger` - Logging functionality
- `TestIntegrationAndEdgeCases` - Integration and edge cases

Key test areas:
- API call logging and monitoring
- Metrics aggregation and analysis
- Quality threshold monitoring
- Cost tracking and optimization
- Data export and persistence

### Quality Performance Reporter Tests

**File**: `test_quality_performance_reporter.py`

- `TestPerformanceReportConfiguration` - Configuration management
- `TestReportMetadata` - Report metadata handling
- `TestPerformanceInsight` - Insight generation
- `TestOptimizationRecommendation` - Recommendation engine
- `TestQualityPerformanceReporter` - Report generation
- `TestVisualizationFunctionality` - Chart and visualization testing

Key test areas:
- Comprehensive report generation
- Multi-format export (JSON, HTML, CSV, text)
- Statistical analysis and insights
- Visualization generation
- Performance optimization recommendations

## Running Specific Tests

### By Test Class
```bash
# Test specific class
pytest test_quality_performance_benchmarks.py::TestQualityValidationMetrics -v

# Test specific method
pytest test_quality_performance_benchmarks.py::TestQualityValidationMetrics::test_calculate_quality_efficiency_score -v
```

### By Pattern
```bash
# Test functions matching pattern
pytest -k "correlation" -v

# Test classes matching pattern  
pytest -k "TestQuality" -v

# Exclude slow tests
pytest -k "not slow" -v
```

### By Marker
```bash
# Run only unit tests
pytest -m unit

# Run integration tests with verbose output
pytest -m integration -v

# Run performance tests with coverage
pytest -m performance --cov
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarking Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements_test.txt
    - name: Run tests
      run: |
        python run_all_tests.py --all --coverage --html-report
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Performance Monitoring

### Benchmark Tests

Performance benchmarks validate that:
- Test execution time is reasonable
- Memory usage stays within bounds
- Concurrent operations work correctly
- Large datasets are handled efficiently

```bash
# Run benchmark tests
pytest -m benchmark --benchmark-only

# Generate benchmark report
pytest -m benchmark --benchmark-json=benchmark_report.json
```

### Resource Monitoring

Tests include monitoring for:
- Memory usage patterns
- CPU utilization
- File I/O operations
- Network requests
- Temporary file cleanup

## Debugging Tests

### Running with Debugger
```bash
# Run with pdb debugger
pytest --pdb test_file.py::test_function

# Run with ipdb (enhanced debugger)
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

### Verbose Output
```bash
# Maximum verbosity
pytest -vvv --tb=long --show-capture=all

# Print output during tests
pytest -s test_file.py
```

### Test Profiling
```bash
# Profile test execution
pytest --profile test_file.py

# Memory profiling
pytest --memprof test_file.py
```

## Test Data Management

### Creating Test Data
```python
# Example test data creation
@pytest.fixture
def sample_metrics():
    return [
        QualityValidationMetrics(
            scenario_name="test",
            operations_count=10,
            average_latency_ms=1200.0
        )
    ]
```

### Test Data Validation
- All test data is validated for consistency
- Fixtures provide realistic data patterns
- Edge cases are explicitly tested
- Data cleanup is automated

## Coverage Reports

### Generating Coverage
```bash
# Terminal coverage report
pytest --cov --cov-report=term-missing

# HTML coverage report
pytest --cov --cov-report=html

# XML coverage report (for CI)
pytest --cov --cov-report=xml
```

### Coverage Targets
- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% feature coverage
- **Overall**: >85% combined coverage

## Contributing to Tests

### Adding New Tests

1. **Create test file** following naming convention `test_*.py`
2. **Use existing fixtures** from `conftest.py` when possible
3. **Add appropriate markers** for test categorization
4. **Include docstrings** describing test purpose
5. **Test both success and failure cases**

### Test Best Practices

1. **Isolation**: Tests should not depend on each other
2. **Deterministic**: Tests should produce consistent results
3. **Fast**: Unit tests should complete quickly
4. **Readable**: Test names should describe what they test
5. **Maintainable**: Use fixtures and helpers to reduce duplication

### Example Test Structure
```python
class TestFeature:
    """Test suite for Feature functionality."""
    
    def test_feature_success(self, fixture_data):
        """Test successful feature operation."""
        # Arrange
        feature = Feature()
        
        # Act
        result = feature.process(fixture_data)
        
        # Assert
        assert result.success is True
        assert result.data is not None
    
    def test_feature_error_handling(self):
        """Test feature error handling."""
        feature = Feature()
        
        with pytest.raises(ValueError):
            feature.process(invalid_data)
    
    @pytest.mark.parametrize("input,expected", [
        ("valid", True),
        ("invalid", False)
    ])
    def test_feature_validation(self, input, expected):
        """Test feature input validation."""
        result = Feature.validate(input)
        assert result == expected
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Permissions**: Check write permissions for temp directories
3. **Resource Cleanup**: Use fixtures for proper cleanup
4. **Async Tests**: Use `pytest-asyncio` for async test support
5. **Mock Issues**: Verify mock setup and expectations

### Getting Help

- Check test logs in `test_reports/` directory
- Use `pytest --collect-only` to verify test discovery
- Run individual tests with `-v` for detailed output
- Use `pytest --tb=long` for full tracebacks

## Appendix

### Dependencies Overview

**Core Testing**:
- pytest - Testing framework
- pytest-asyncio - Async test support
- pytest-mock - Mocking utilities
- pytest-cov - Coverage reporting

**Specialized Testing**:
- pytest-benchmark - Performance benchmarking
- pytest-xdist - Parallel execution
- pytest-timeout - Test timeouts
- responses - HTTP mocking

**Development**:
- rich - Enhanced output formatting
- ipdb - Enhanced debugging
- memory-profiler - Memory analysis

### Configuration Files

- `pytest.ini` - Pytest configuration
- `conftest.py` - Shared fixtures
- `requirements_test.txt` - Testing dependencies
- `.coveragerc` - Coverage configuration (optional)

### Test Reports

Generated reports include:
- Test execution results (JSON/XML)
- Coverage reports (HTML/JSON)
- Benchmark results
- Performance profiles
- Error logs and debugging information

---

For more information about the performance benchmarking utilities themselves, see the main project documentation.