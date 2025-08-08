# Comprehensive Test Suite for LLM Classification System

## Overview

This comprehensive test suite provides thorough testing and validation for the LLM-based classification system in the Clinical Metabolomics Oracle, with >95% code coverage target, performance benchmarking, and quality assurance.

## Test Architecture

### Core Components Tested

1. **LLMQueryClassifier** - Core LLM-powered classification with caching and fallback
2. **HybridConfidenceScorer** - Advanced confidence scoring with calibration
3. **BiomedicalQueryRouter** - Existing routing infrastructure integration  
4. **Multi-tiered Fallback System** - 5 levels of degradation handling
5. **Performance Optimization** - Caching, batching, and concurrent processing
6. **Error Handling** - API failures, malformed responses, resource exhaustion

### Test Categories

| Category | Description | Critical | Target Coverage |
|----------|-------------|----------|-----------------|
| `core` | Core LLM classifier functionality | ✓ | 98% |
| `confidence` | Confidence scoring and calibration | ✓ | 96% |
| `performance` | Performance optimization and load tests | ✓ | 94% |
| `integration` | Integration compatibility tests | ✓ | 95% |
| `edge_cases` | Edge cases and error handling | - | 92% |
| `quality` | Code quality and validation tests | - | 90% |

## Quick Start

### Installation

```bash
# Install test dependencies
pip install -r requirements_test.txt

# Verify installation
python -m pytest --version
python -m coverage --version
```

### Running Tests

```bash
# Run all tests with coverage
python run_comprehensive_tests.py

# Run specific category
python run_comprehensive_tests.py --categories core confidence

# Generate all report formats
python run_comprehensive_tests.py --format all

# CI/CD mode
python run_comprehensive_tests.py --ci --format json
```

### Quick Validation

```bash
# Run core functionality tests (5-10 minutes)
python run_comprehensive_tests.py --categories core

# Performance validation (<2 second response times)
python run_comprehensive_tests.py --categories performance

# Integration compatibility check  
python run_comprehensive_tests.py --categories integration
```

## Test Structure

```
lightrag_integration/tests/
├── test_comprehensive_llm_classification_system.py  # Main test suite
├── test_fixtures_comprehensive.py                   # Test data and mocks
├── test_coverage_config.py                         # Coverage infrastructure
├── conftest.py                                     # Pytest configuration
└── test_reports/                                   # Generated reports
    ├── coverage.html                               # HTML coverage report
    ├── test_report.json                           # JSON test results
    └── comprehensive_report_YYYYMMDD_HHMMSS.html  # Full report
```

## Test Features

### 1. Core LLM Classifier Tests

**File**: `test_comprehensive_llm_classification_system.py`

- ✅ **LLM Classification**: Successful classification with various query types
- ✅ **Caching System**: Cache hits/misses, TTL expiration, size limits
- ✅ **Circuit Breakers**: API failure handling, retry mechanisms
- ✅ **Cost Management**: Daily budget limits, token usage tracking
- ✅ **Fallback Mechanisms**: Graceful degradation to keyword-based classification

```python
# Example test
@patch('lightrag_integration.llm_query_classifier.AsyncOpenAI')
async def test_successful_classification(self, mock_openai_class):
    # Setup mock response
    mock_client = AsyncMock()
    mock_response.choices[0].message.content = json.dumps({
        "category": "KNOWLEDGE_GRAPH",
        "confidence": 0.85,
        "reasoning": "Query about established metabolic pathways"
    })
    
    # Test classification
    result, used_llm = await classifier.classify_query(query)
    assert used_llm == True
    assert result.category == "KNOWLEDGE_GRAPH"
    assert result.confidence == 0.85
```

### 2. Confidence Scoring Tests

**Features Tested**:
- ✅ **LLM Confidence Analysis**: Reasoning quality, consistency scoring
- ✅ **Keyword Confidence Analysis**: Pattern matching, domain specificity  
- ✅ **Hybrid Scoring**: Adaptive weighting, uncertainty quantification
- ✅ **Calibration System**: Historical accuracy tracking, confidence intervals
- ✅ **Validation Framework**: Accuracy measurement, recommendation generation

```python
# Example confidence test
async def test_hybrid_confidence_calculation(self):
    scorer = HybridConfidenceScorer(logger=self.logger)
    result = await scorer.calculate_comprehensive_confidence(
        query_text="What is glucose metabolism?",
        llm_result=classification_result
    )
    
    assert isinstance(result, HybridConfidenceResult)
    assert 0 <= result.overall_confidence <= 1
    assert result.llm_weight + result.keyword_weight == pytest.approx(1.0)
```

### 3. Performance Tests

**Performance Requirements**:
- ⏱️ **Response Time**: <2 seconds per classification
- 🚀 **Throughput**: >100 classifications per second  
- 💾 **Memory**: <512MB memory usage
- 🔄 **Concurrency**: Handle 50+ concurrent requests

```python
# Example performance test
async def test_response_time_requirement(self):
    for query in test_queries:
        start_time = time.time()
        result, used_llm = await classifier.classify_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 2.0, f"Query took {response_time:.3f}s (>2s)"
```

### 4. Integration Tests

**Compatibility Validation**:
- 🔗 **BiomedicalQueryRouter**: Seamless integration with existing routing
- 📊 **ConfidenceMetrics**: Backward compatibility with existing confidence structure
- 🏭 **Factory Functions**: Proper initialization and configuration
- 📈 **Legacy Support**: Existing API compatibility

### 5. Edge Cases and Error Handling

**Robustness Testing**:
- 🔌 **API Failures**: Timeout, network errors, rate limiting
- 🧩 **Malformed Responses**: Invalid JSON, missing fields, wrong types
- 💥 **Resource Exhaustion**: Memory limits, cache overflow
- 🔒 **Security**: Input validation, injection prevention

### 6. Quality Assurance

**Quality Metrics**:
- 📝 **Code Coverage**: >95% line coverage target
- 🎯 **Test Coverage**: All critical functions covered
- 📊 **Performance Regression**: Automated performance monitoring
- 🔍 **Code Quality**: Linting, type checking, documentation

## Test Data and Fixtures

### Realistic Biomedical Queries

**Knowledge Graph Queries**:
```python
"What is the relationship between glucose metabolism and insulin signaling pathways?"
"How does mitochondrial dysfunction affect cellular energy production?"
"Explain the citric acid cycle and its connection to amino acid biosynthesis"
```

**Real-Time Queries**:
```python
"Latest research on metabolomics biomarkers for COVID-19 in 2025"  
"Recent breakthrough in Alzheimer's disease biomarker discovery"
"New FDA approved metabolomics-based diagnostic tools this year"
```

**Edge Cases**:
```python
""  # Empty query
"a"  # Single character
"Very long query..." * 1000  # Extremely long query
"Special !@#$%^&*() characters with metabolomics 2025"  # Special chars
```

### Mock Data Generation

```python
# Generate concurrent test queries
queries = PerformanceTestDataGenerator.generate_concurrent_queries(100)

# Generate calibration test data
calibration_data = ConfidenceCalibrationTestData.generate_calibration_history(500)

# Create mock API responses
mock_client = MockLLMResponses.create_mock_openai_client("success")
```

## Running Specific Test Scenarios

### Development Testing

```bash
# Quick smoke test (2-3 minutes)
pytest lightrag_integration/tests/test_comprehensive_llm_classification_system.py::TestLLMQueryClassifier::test_classifier_initialization -v

# Core functionality validation (5-10 minutes)
python run_comprehensive_tests.py --categories core --format html

# Performance validation (10-15 minutes)  
python run_comprehensive_tests.py --categories performance --format all
```

### CI/CD Pipeline

```bash
# Fast validation suite (5 minutes)
python run_comprehensive_tests.py --categories core integration --ci

# Full validation suite (20-30 minutes)
python run_comprehensive_tests.py --ci --format json

# Performance regression check
python run_comprehensive_tests.py --categories performance --ci
```

### Production Deployment

```bash
# Pre-deployment validation
python run_comprehensive_tests.py --categories core confidence integration

# Load testing simulation
python run_comprehensive_tests.py --categories performance quality

# Full system validation
python run_comprehensive_tests.py --format all
```

## Coverage Analysis

### Coverage Targets

| Component | Target | Critical Functions |
|-----------|--------|-------------------|
| `llm_query_classifier` | 98% | `classify_query`: 100% |
| `comprehensive_confidence_scorer` | 96% | `calculate_comprehensive_confidence`: 98% |
| `query_router` | 94% | `route_query`: 95% |
| `llm_classification_prompts` | 90% | `build_primary_prompt`: 92% |

### Coverage Reports

```bash
# Generate HTML coverage report
python -m coverage html --directory=coverage_reports/htmlcov

# Generate JSON coverage report  
python -m coverage json --output=coverage_reports/coverage.json

# View coverage summary
python -m coverage report --show-missing
```

### Coverage Analysis

The coverage infrastructure provides:

- 📊 **Line Coverage**: Percentage of lines executed
- 🌲 **Branch Coverage**: Percentage of code branches taken
- 🎯 **Function Coverage**: Percentage of functions called
- 📈 **Trend Analysis**: Coverage changes over time

## Performance Benchmarking

### Benchmarking Infrastructure

```python
# Response time benchmarking
async def test_classification_latency_benchmark(self):
    for query in test_queries:
        start_time = time.time()
        result = await classifier.classify_query(query)
        latency = time.time() - start_time
        
        assert latency < 2.0, f"Latency {latency:.3f}s exceeds 2s target"
```

### Performance Metrics

- ⏱️ **Latency**: P50, P95, P99 response times
- 🔄 **Throughput**: Requests per second capacity
- 💾 **Memory**: Peak memory usage, memory efficiency
- 🌐 **Concurrency**: Concurrent request handling
- 📈 **Scalability**: Performance under load

### Performance Reports

```bash
# Run performance benchmarks
python run_comprehensive_tests.py --categories performance

# Generate performance report
python lightrag_integration/tests/test_coverage_config.py --format html
```

## Quality Assurance

### Quality Metrics

1. **Test Quality Score**: Based on test success rate and coverage
2. **Code Quality Score**: Based on linting, documentation, complexity
3. **Performance Quality**: Response times, memory usage, throughput
4. **Reliability Score**: Error handling, fallback effectiveness

### Quality Reports

```bash
# Generate quality report
python run_comprehensive_tests.py --format html

# View quality metrics
cat test_reports/comprehensive_report_*.html
```

### Quality Thresholds

```python
QUALITY_THRESHOLDS = {
    'coverage_target': 95.0,           # >95% coverage
    'performance_target_ms': 2000,     # <2s response time
    'test_success_rate': 98.0,         # >98% test success rate
    'code_quality_min': 80.0,          # >80 code quality score
    'critical_function_coverage': 98.0  # >98% critical function coverage
}
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Comprehensive Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements_test.txt
    - name: Run comprehensive tests
      run: |
        python run_comprehensive_tests.py --ci --format json
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

### Test Automation

The test suite supports:

- 🔄 **Automated Execution**: CI/CD pipeline integration
- 📊 **Automated Reporting**: Coverage, performance, quality reports  
- 🚨 **Automated Alerts**: Test failures, performance regressions
- 📈 **Trend Analysis**: Quality metrics over time

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `requirements_test.txt`
2. **API Key Missing**: Set `OPENAI_API_KEY` environment variable for API tests
3. **Timeout Errors**: Increase timeout values in `pytest.ini` for slow systems
4. **Memory Issues**: Reduce concurrent test execution or increase memory limits

### Debug Mode

```bash
# Run with verbose output
python run_comprehensive_tests.py --verbose

# Run specific failing test
pytest lightrag_integration/tests/test_comprehensive_llm_classification_system.py::TestName::test_method -v -s

# Debug with breakpoints
pytest --pdb lightrag_integration/tests/test_comprehensive_llm_classification_system.py
```

### Performance Issues

```bash
# Profile test execution
pytest --profile lightrag_integration/tests/

# Memory profiling  
pytest --memray lightrag_integration/tests/

# Parallel execution
pytest -n auto lightrag_integration/tests/
```

## Contributing

### Adding New Tests

1. **Follow Naming Conventions**: `test_*` for files, `Test*` for classes
2. **Use Appropriate Markers**: `@pytest.mark.core`, `@pytest.mark.performance`, etc.
3. **Include Docstrings**: Describe what each test validates
4. **Add to Coverage**: Ensure new code has corresponding tests
5. **Update Documentation**: Update this README if adding new test categories

### Test Quality Guidelines

1. **Isolated Tests**: Each test should be independent
2. **Descriptive Names**: Test names should clearly indicate what is being tested
3. **Proper Assertions**: Use specific assertions with descriptive error messages
4. **Mock External Dependencies**: Don't rely on external services in unit tests
5. **Performance Considerations**: Keep test execution time reasonable

## Support

For issues, questions, or contributions:

1. **Check Test Reports**: Review generated HTML reports for detailed information
2. **Run Debug Mode**: Use verbose output and debugging features
3. **Review Coverage**: Check coverage reports for untested code paths
4. **Performance Analysis**: Use benchmarking tools for performance issues

## Summary

This comprehensive test suite provides:

- ✅ **>95% Code Coverage** with detailed reporting
- ⚡ **<2 Second Response Time** validation  
- 🔧 **Comprehensive Error Handling** testing
- 🎯 **Quality Assurance** with automated metrics
- 📊 **Performance Benchmarking** with regression detection
- 🔗 **Integration Testing** with existing infrastructure
- 📈 **Continuous Monitoring** with quality trend analysis

The test suite ensures the LLM classification system meets all performance, quality, and reliability requirements for production deployment.