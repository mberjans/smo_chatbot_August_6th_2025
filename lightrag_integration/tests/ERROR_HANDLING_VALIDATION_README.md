# Comprehensive Error Handling Validation Framework

This comprehensive testing framework validates the error handling implementation for the Clinical Metabolomics Oracle system. It ensures that the system correctly handles failures, recovers gracefully, and maintains production-ready reliability under various stress conditions.

## Overview

The validation framework consists of multiple components that test different aspects of error handling:

1. **End-to-End Error Handling Validation** - Tests complete failure scenarios with realistic error injection
2. **Logging System Validation** - Ensures comprehensive and structured logging with correlation ID tracking
3. **Recovery Mechanism Testing** - Validates degradation modes, checkpoints, and recovery strategies
4. **Performance Under Stress** - Tests system behavior under resource constraints and repeated failures
5. **Memory Leak Detection** - Identifies potential memory leaks during error scenarios
6. **Circuit Breaker Validation** - Tests circuit breaker behavior under sustained failures

## Quick Start

### Run Complete Validation (Recommended)

```bash
# Full comprehensive validation (recommended for production readiness assessment)
cd lightrag_integration/tests
python comprehensive_error_handling_master_validation.py

# Quick validation (faster, good for CI/CD)
python comprehensive_error_handling_master_validation.py --quick

# Verbose output with detailed logging
python comprehensive_error_handling_master_validation.py --verbose
```

### Run Individual Components

```bash
# Run only E2E error handling tests
python run_error_handling_validation.py --quick

# Run only logging validation
python test_logging_validation.py

# Run specific error scenarios
python run_error_handling_validation.py --scenarios basic_error_injection,circuit_breaker
```

## Test Scenarios

### 1. Basic Error Injection
- **Purpose**: Tests handling of various failure types (API errors, network timeouts, rate limits)
- **Duration**: 30-60 seconds
- **Validates**: Error classification, retry logic, basic recovery mechanisms

### 2. Resource Pressure Simulation
- **Purpose**: Tests behavior under memory, CPU, and disk pressure
- **Duration**: 45-90 seconds  
- **Validates**: Resource monitoring, adaptive batch sizing, degradation mode switching

### 3. Circuit Breaker Testing
- **Purpose**: Tests circuit breaker patterns under sustained failures
- **Duration**: 60-120 seconds
- **Validates**: Circuit state transitions, failure counting, recovery behavior

### 4. Checkpoint & Resume Testing
- **Purpose**: Tests checkpoint creation, corruption handling, and resume functionality
- **Duration**: 30-60 seconds
- **Validates**: State persistence, corruption recovery, resume accuracy

### 5. Memory Leak Detection
- **Purpose**: Detects memory leaks during repeated error scenarios
- **Duration**: 90-180 seconds (can be skipped with `--skip-slow`)
- **Validates**: Memory usage patterns, resource cleanup, leak detection

### 6. Logging System Validation
- **Purpose**: Validates structured logging, correlation IDs, and error context
- **Duration**: 10-20 seconds
- **Validates**: Log format compliance, correlation tracking, error context inclusion

## Command Line Options

### Master Validation Script

```bash
python comprehensive_error_handling_master_validation.py [options]

Options:
  --quick              Run abbreviated tests (faster execution)
  --output-dir DIR     Directory for test results (default: ./master_validation_results)
  --verbose           Enable verbose logging
  --skip-slow         Skip slow/long-running tests
  --parallel          Run tests in parallel where possible  
  --report-format     Report format: json, text (default: text)
```

### Individual Test Runner

```bash
python run_error_handling_validation.py [options]

Options:
  --quick                      Run abbreviated tests
  --output-dir DIR            Output directory
  --verbose                   Enable verbose logging
  --scenarios SCENARIOS       Comma-separated scenario list
  --list-scenarios           List available scenarios
  --json-output              JSON output only
  --timeout SECONDS          Maximum execution time
```

## Understanding Results

### Exit Codes

- **0**: All validations passed - system ready for production
- **1**: Some validations failed - issues need to be addressed  
- **2**: Critical failures - system not ready for production

### Production Readiness Scores

- **≥90%**: Production Ready - All systems go
- **≥80%**: Conditionally Ready - Deploy with enhanced monitoring
- **≥60%**: Needs Improvement - Address issues before production
- **<60%**: Not Ready - Significant improvements required

### Status Indicators

- **✓ PASS**: Component validation succeeded
- **⚠ CONDITIONAL**: Passed with warnings/recommendations
- **✗ FAIL**: Component validation failed
- **? ERROR**: Validation encountered an error

## Example Output

```
================================================================================
COMPREHENSIVE ERROR HANDLING VALIDATION REPORT
================================================================================

Validation completed: 2025-08-07T14:30:45
Duration: 127.3 seconds
Overall Status: PRODUCTION_READY
Production Readiness Score: 92%

COMPONENT VALIDATION RESULTS:
--------------------------------------------------
e2e_validation           PASS
logging_validation       PASS  
performance_stress_test  PASS

RECOMMENDATIONS:
------------------------------
  1. Consider implementing additional monitoring for memory usage patterns
  2. Review API timeout configurations for optimal performance

================================================================================
FINAL ASSESSMENT
================================================================================
✓ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT
  All critical validations passed. The error handling system
  demonstrates robust behavior under various failure conditions.
================================================================================
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Error Handling Validation
on: [push, pull_request]

jobs:
  error-handling-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          
      - name: Run Error Handling Validation
        run: |
          cd lightrag_integration/tests
          python comprehensive_error_handling_master_validation.py --quick --report-format json
        
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: error-handling-validation-results
          path: lightrag_integration/tests/master_validation_results/
```

### Docker Integration

```dockerfile
# Add to your Dockerfile for testing
RUN cd lightrag_integration/tests && \
    python comprehensive_error_handling_master_validation.py --quick && \
    echo "Error handling validation passed"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd lightrag_integration/tests
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
   ```

2. **Memory Issues During Testing**
   ```bash
   # Use quick mode to reduce memory usage
   python comprehensive_error_handling_master_validation.py --quick --skip-slow
   ```

3. **Timeouts on Slow Systems**
   ```bash
   # Increase timeout and use sequential execution
   python run_error_handling_validation.py --timeout 900 --quick
   ```

### Debug Mode

```bash
# Enable maximum logging for debugging
python comprehensive_error_handling_master_validation.py --verbose --output-dir ./debug_results

# Check detailed logs
tail -f debug_results/master_validation_*.log
```

## Customizing Tests

### Adding New Scenarios

1. Create scenario function in `test_error_handling_e2e_validation.py`
2. Add scenario to the validator's scenario list
3. Update documentation and CLI options

### Modifying Thresholds

Edit the validation criteria in the respective test files:
- **Success rate thresholds**: Modify in each scenario's validation logic
- **Memory leak detection**: Adjust memory increase thresholds
- **Performance limits**: Update CPU/memory usage limits

### Custom Resource Limits

```python
# Example: Custom resource thresholds
resource_thresholds = ResourceThresholds(
    memory_warning_percent=70.0,   # Default: 75%
    memory_critical_percent=85.0,  # Default: 90%
    disk_warning_percent=75.0,     # Default: 80%
    disk_critical_percent=90.0     # Default: 95%
)
```

## Best Practices

### For Development
- Run `--quick` validation during development cycles
- Use `--verbose` when debugging specific issues
- Run full validation before major releases

### For CI/CD
- Use `--quick` mode for faster feedback
- Save JSON results for automated processing
- Set appropriate timeouts for your infrastructure

### For Production Deployment
- Run full validation before each production deployment
- Achieve at least 90% production readiness score
- Address all critical failures and warnings
- Consider running validation in staging environment that mirrors production

## Support and Maintenance

### Updating Test Dependencies
```bash
pip install -r test_requirements.txt --upgrade
```

### Regular Maintenance Tasks
1. Review and update failure scenarios based on production incidents
2. Adjust thresholds based on system performance characteristics
3. Add new test scenarios for newly implemented error handling features
4. Update documentation when adding new validation components

### Monitoring Production Health
The validation framework can be adapted to run periodic health checks in production:
```bash
# Example: Daily production health check
python comprehensive_error_handling_master_validation.py --quick --scenarios basic_error_injection
```

This validation framework provides comprehensive coverage of error handling scenarios and should give you confidence that the error handling system is ready for production use.