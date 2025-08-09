# Unified System Health Dashboard - Comprehensive Testing Guide

This guide provides complete documentation for testing the Unified System Health Dashboard implementation, including setup, execution, and interpretation of test results.

## Table of Contents

1. [Test Suite Overview](#test-suite-overview)
2. [Quick Start](#quick-start)
3. [Test Categories](#test-categories)
4. [Installation and Setup](#installation-and-setup)
5. [Running Tests](#running-tests)
6. [Test Configuration](#test-configuration)
7. [Coverage and Reporting](#coverage-and-reporting)
8. [Performance Testing](#performance-testing)
9. [Integration Testing](#integration-testing)
10. [CI/CD Integration](#cicd-integration)
11. [Troubleshooting](#troubleshooting)

## Test Suite Overview

The comprehensive test suite covers all aspects of the dashboard implementation:

### Test Statistics
- **Total Test Classes**: 8
- **Total Test Methods**: 50+
- **Lines of Test Code**: 2,800+
- **Coverage Target**: 80%+
- **Test Types**: Unit, Integration, API, WebSocket, Performance, Security

### Key Components Tested
- `UnifiedDataAggregator` - Data collection and processing
- `WebSocketManager` - Real-time communication
- `UnifiedSystemHealthDashboard` - Main dashboard application
- `DashboardIntegrationHelper` - Deployment and configuration
- REST API endpoints and error handling
- Alert generation and management
- Database persistence and retrieval
- Configuration validation

## Quick Start

### 1. Install Dependencies
```bash
# Install test dependencies
pip install -r lightrag_integration/test_requirements.txt

# Or use the test runner to install automatically
python lightrag_integration/run_dashboard_tests.py --install-deps
```

### 2. Validate Environment
```bash
# Check that everything is set up correctly
python lightrag_integration/run_dashboard_tests.py --validate-env
```

### 3. Run Basic Tests
```bash
# Run all tests
python lightrag_integration/run_dashboard_tests.py

# Run quick subset (for development)
python lightrag_integration/run_dashboard_tests.py --quick
```

### 4. Generate Coverage Report
```bash
# Run with coverage
python lightrag_integration/run_dashboard_tests.py --coverage
```

## Test Categories

### Unit Tests (`--unit`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Data aggregator, WebSocket manager, configuration classes
- **Runtime**: ~30 seconds
- **Dependencies**: Minimal (mocked external systems)

```bash
python run_dashboard_tests.py --unit
```

### Integration Tests (`--integration`)
- **Purpose**: Test interaction between dashboard components
- **Coverage**: Orchestrator integration, real-time updates, alert propagation
- **Runtime**: ~60 seconds
- **Dependencies**: Mock monitoring systems

```bash
python run_dashboard_tests.py --integration
```

### API Tests (`--api`)
- **Purpose**: Test REST API endpoints
- **Coverage**: All HTTP endpoints, error handling, response formats
- **Runtime**: ~45 seconds
- **Dependencies**: FastAPI test client

```bash
python run_dashboard_tests.py --api
```

### WebSocket Tests (`--websocket`)
- **Purpose**: Test real-time WebSocket functionality
- **Coverage**: Connection management, broadcasting, error handling
- **Runtime**: ~30 seconds
- **Dependencies**: Mock WebSocket connections

```bash
python run_dashboard_tests.py --websocket
```

### Performance Tests (`--performance`)
- **Purpose**: Test system performance under load
- **Coverage**: Data aggregation speed, WebSocket broadcasting, memory usage
- **Runtime**: ~120 seconds
- **Dependencies**: Performance monitoring tools

```bash
python run_dashboard_tests.py --performance
```

### Security Tests (`--security`)
- **Purpose**: Test security features and input validation
- **Coverage**: Authentication, CORS, input sanitization, SSL configuration
- **Runtime**: ~20 seconds
- **Dependencies**: Security testing utilities

```bash
python run_dashboard_tests.py --security
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Create Virtual Environment** (recommended)
```bash
python -m venv dashboard_test_env
source dashboard_test_env/bin/activate  # On Windows: dashboard_test_env\Scripts\activate
```

2. **Install Core Dependencies**
```bash
# Install the main dashboard dependencies first
pip install fastapi uvicorn websockets aiosqlite pyyaml
```

3. **Install Test Dependencies**
```bash
# Install all test-specific dependencies
pip install -r lightrag_integration/test_requirements.txt
```

4. **Verify Installation**
```bash
# Run environment validation
python lightrag_integration/run_dashboard_tests.py --validate-env
```

### Docker Setup (Optional)
```dockerfile
# Create test environment with Docker
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r lightrag_integration/test_requirements.txt

CMD ["python", "lightrag_integration/run_dashboard_tests.py", "--coverage"]
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests with default configuration
python lightrag_integration/run_dashboard_tests.py

# Run tests with verbose output
python lightrag_integration/run_dashboard_tests.py -v

# Run tests and stop on first failure
python lightrag_integration/run_dashboard_tests.py -x
```

### Advanced Test Execution

```bash
# Run tests in parallel (faster)
python run_dashboard_tests.py --parallel

# Run with specific number of workers
python run_dashboard_tests.py --parallel --workers 4

# Run with custom timeout
python run_dashboard_tests.py --timeout 600

# Generate HTML report
python run_dashboard_tests.py --html-report
```

### Development Workflows

```bash
# Quick development testing (excludes slow tests)
python run_dashboard_tests.py --quick

# Continuous testing (re-run on file changes)
python run_dashboard_tests.py --continuous

# Run specific test patterns
python run_dashboard_tests.py -- -k "test_data_aggregation"

# Run tests for specific file
python run_dashboard_tests.py -- tests/test_specific_component.py
```

## Test Configuration

### pytest.ini Configuration
The test suite uses `pytest.ini` for configuration:

```ini
[tool:pytest]
# Key settings
asyncio_mode = auto
testpaths = .
addopts = -v --tb=short --cov-fail-under=80
timeout = 300
```

### Environment Variables
```bash
# Test environment configuration
export TESTING=1
export DASHBOARD_ENV=test
export PYTHONPATH=.

# Database configuration for tests
export TEST_DB_PATH=":memory:"
export TEST_PORT=8093
```

### Custom Configuration
Create `test_config.yaml` for custom test settings:

```yaml
database:
  path: ":memory:"
  enable_persistence: true

dashboard:
  host: "127.0.0.1"
  port: 8093
  update_interval: 0.5

alerts:
  cooldown_seconds: 10
  enable_email: false

performance:
  max_connections: 100
  timeout_seconds: 30
```

## Coverage and Reporting

### Coverage Reports

```bash
# Generate multiple coverage report formats
python run_dashboard_tests.py --coverage

# Coverage reports generated:
# - Terminal output (immediate)
# - HTML report: htmlcov/index.html
# - XML report: coverage.xml (for CI/CD)
```

### HTML Test Reports

```bash
# Generate detailed HTML test report
python run_dashboard_tests.py --html-report

# Open report
open dashboard_test_report.html
```

### Coverage Targets
- **Minimum Coverage**: 80%
- **Target Coverage**: 90%+
- **Critical Components**: 95%+ (data aggregator, API endpoints)

### Excluded from Coverage
- Mock objects and test fixtures
- Development-only code paths
- Error handling for external system failures
- Platform-specific code branches

## Performance Testing

### Performance Test Categories

1. **Data Aggregation Performance**
   - Snapshot creation speed
   - Memory usage during aggregation
   - Database write performance

2. **WebSocket Broadcasting Performance**
   - Concurrent connection handling
   - Message broadcast latency
   - Connection cleanup efficiency

3. **API Response Performance**
   - Endpoint response times
   - Concurrent request handling
   - Database query performance

4. **Memory Usage Testing**
   - Memory growth over time
   - Garbage collection efficiency
   - Resource leak detection

### Running Performance Tests

```bash
# Run all performance tests
python run_dashboard_tests.py --performance

# Run with detailed output
python run_dashboard_tests.py --performance -s

# Profile memory usage
python -m pytest --memray lightrag_integration/test_unified_system_health_dashboard_comprehensive.py::TestDashboardPerformance
```

### Performance Benchmarks

| Component | Metric | Target | Measured |
|-----------|---------|--------|----------|
| Data Aggregation | Snapshots/sec | >10 | ~15 |
| WebSocket Broadcast | Connections | 100+ | 150+ |
| API Response | Average latency | <100ms | ~50ms |
| Memory Usage | Growth rate | <1MB/hour | ~0.5MB/hour |

## Integration Testing

### Integration Test Scenarios

1. **Dashboard-Orchestrator Integration**
   - System status data flow
   - Health check integration
   - Load level synchronization

2. **Real-time Update Flow**
   - Data aggregator → WebSocket flow
   - Alert generation → Broadcasting
   - Historical data persistence

3. **Multi-Component Interactions**
   - Full system integration
   - Error propagation handling
   - Graceful degradation scenarios

### Mock System Configuration

```python
# Example mock orchestrator setup
@pytest.fixture
def mock_orchestrator():
    mock_orch = Mock()
    mock_orch.get_system_status.return_value = {
        'start_time': datetime.now().isoformat(),
        'current_load_level': 'NORMAL',
        'total_requests_processed': 1000,
        'integration_status': {
            'load_monitoring_active': True,
            'degradation_controller_active': True,
        }
    }
    return mock_orch
```

### Running Integration Tests

```bash
# Run integration tests only
python run_dashboard_tests.py --integration

# Run with mock system debugging
python run_dashboard_tests.py --integration -s --log-cli-level=DEBUG
```

## CI/CD Integration

### GitHub Actions Configuration

```yaml
name: Dashboard Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r lightrag_integration/test_requirements.txt
    
    - name: Run tests with coverage
      run: |
        python lightrag_integration/run_dashboard_tests.py --coverage --parallel
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'python -m pip install -r lightrag_integration/test_requirements.txt'
            }
        }
        
        stage('Test') {
            steps {
                sh 'python lightrag_integration/run_dashboard_tests.py --coverage --html-report'
            }
        }
        
        stage('Publish') {
            steps {
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '.',
                    reportFiles: 'dashboard_test_report.html',
                    reportName: 'Dashboard Test Report'
                ])
                
                publishCoverage adapters: [
                    coberturaAdapter('coverage.xml')
                ]
            }
        }
    }
}
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: dashboard-tests
        name: Dashboard Tests
        entry: python lightrag_integration/run_dashboard_tests.py --quick
        language: system
        pass_filenames: false
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'unified_system_health_dashboard'
# Solution: Set PYTHONPATH
export PYTHONPATH=.
python run_dashboard_tests.py
```

#### 2. Database Connection Issues
```bash
# Error: Database locked or permission denied
# Solution: Use in-memory database for tests
export TEST_DB_PATH=":memory:"
python run_dashboard_tests.py
```

#### 3. Port Conflicts
```bash
# Error: Port 8092 already in use
# Solution: Use different test port
export TEST_PORT=8094
python run_dashboard_tests.py
```

#### 4. Timeout Issues
```bash
# Error: Test timeout after 300 seconds
# Solution: Increase timeout for slow systems
python run_dashboard_tests.py --timeout 600
```

#### 5. Memory Issues
```bash
# Error: Out of memory during performance tests
# Solution: Run performance tests separately
python run_dashboard_tests.py --unit --integration --api
python run_dashboard_tests.py --performance
```

### Debug Mode Testing

```bash
# Run with maximum debug information
python run_dashboard_tests.py --log-cli-level=DEBUG -s -v

# Run single test with debugging
python -m pytest -s -v lightrag_integration/test_unified_system_health_dashboard_comprehensive.py::TestUnifiedDataAggregator::test_initialization

# Run with pdb on failures
python run_dashboard_tests.py --pdb
```

### Test Environment Cleanup

```bash
# Clean up test artifacts
rm -rf htmlcov/
rm -f coverage.xml
rm -f dashboard_test_report.html
rm -f tests.log
rm -f .coverage

# Reset test database
rm -f test_dashboard.db
```

## Best Practices

### Writing Tests
1. **Use descriptive test names** that explain what is being tested
2. **Keep tests independent** - each test should be able to run in isolation
3. **Use appropriate fixtures** for setup and teardown
4. **Mock external dependencies** to ensure test reliability
5. **Test both success and failure scenarios**

### Test Maintenance
1. **Run tests regularly** during development
2. **Update tests when code changes** to maintain coverage
3. **Review test failures** carefully - they often indicate real issues
4. **Keep test dependencies up to date**
5. **Monitor test performance** and optimize slow tests

### Performance Considerations
1. **Use parallel testing** for faster execution
2. **Separate slow tests** with markers
3. **Optimize test fixtures** to reduce setup time
4. **Use in-memory databases** for test isolation
5. **Clean up resources** properly in teardown

## Support and Contributing

### Getting Help
- Check this documentation first
- Review test output and error messages
- Check the troubleshooting section
- Look at existing test examples for patterns

### Contributing Tests
1. Follow the existing test structure and naming conventions
2. Add appropriate markers for test categorization
3. Include docstrings explaining test purpose
4. Ensure new tests are included in the appropriate test class
5. Update this documentation if adding new test categories

### Reporting Issues
When reporting test issues, include:
- Python version and OS
- Complete error message and stack trace
- Steps to reproduce the issue
- Test command used
- Environment variables set

---

For more information about the dashboard implementation, see the main documentation files in the `lightrag_integration/` directory.