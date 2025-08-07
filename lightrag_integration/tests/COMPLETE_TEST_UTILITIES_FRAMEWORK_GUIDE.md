# Complete Test Utilities Framework Guide - CMO-LIGHTRAG-008-T06

## Overview

The Complete Test Utilities Framework provides comprehensive testing infrastructure for the Clinical Metabolomics Oracle LightRAG integration. This framework integrates all test utilities into a cohesive, well-configured testing system with automated resource management and configuration templates.

## Framework Components

### Core Utilities Integration
- **TestEnvironmentManager**: Environment setup and validation
- **MockSystemFactory**: Standardized mock object creation  
- **AsyncTestCoordinator**: Async test execution coordination
- **PerformanceAssertionHelper**: Performance testing and validation
- **ValidationTestUtilities**: Response quality and content validation

### New Configuration Management
- **ConfigurationTestHelper**: Central configuration management
- **ResourceCleanupManager**: Automated resource cleanup
- **ConfigurationValidationSuite**: Configuration testing
- **EnvironmentIsolationManager**: Test isolation and sandboxing

## Quick Start

### 1. Basic Test Configuration

```python
from configuration_test_utilities import create_complete_test_environment, TestScenarioType

# Create complete test environment
test_env = create_complete_test_environment(TestScenarioType.INTEGRATION_TEST)

# Use components
environment_manager = test_env['environment_manager']
mock_system = test_env['mock_system']
cleanup_manager = test_env['cleanup_manager']

# Automatic cleanup when done
config_helper = test_env['config_helper']
config_helper.cleanup_configuration(test_env['context_id'])
```

### 2. Managed Test Environment (Recommended)

```python
from configuration_test_utilities import managed_test_environment, TestScenarioType

async def test_with_managed_environment():
    async with managed_test_environment(TestScenarioType.BIOMEDICAL_TEST) as test_env:
        # Use test environment
        lightrag_mock = test_env['mock_system']['lightrag_system']
        response = await lightrag_mock.aquery("What is clinical metabolomics?")
        
        # Validate response
        assert response is not None
        # Automatic cleanup when exiting context
```

### 3. Using Pytest Fixtures

```python
import pytest

class TestWithFramework:
    def test_unit_scenario(self, standard_unit_test_config):
        """Test using standard unit test configuration."""
        test_env = standard_unit_test_config
        assert test_env['environment_manager'] is not None
        
    def test_integration_scenario(self, standard_integration_test_config):
        """Test using standard integration configuration."""
        test_env = standard_integration_test_config
        mock_system = test_env['mock_system']
        assert 'lightrag_system' in mock_system
        
    def test_performance_scenario(self, standard_performance_test_config):
        """Test using performance configuration.""" 
        test_env = standard_performance_test_config
        performance_helper = test_env['performance_helper']
        assert performance_helper is not None
```

## Configuration Scenarios

### Available Test Scenario Types

1. **UNIT_TEST**: Minimal resources, fast execution
2. **INTEGRATION_TEST**: Multiple components, moderate resources
3. **PERFORMANCE_TEST**: Performance monitoring, resource limits
4. **STRESS_TEST**: High load, extensive resource monitoring
5. **E2E_TEST**: Complete system integration
6. **VALIDATION_TEST**: Response quality validation focus
7. **MOCK_TEST**: Extensive mock system usage
8. **ASYNC_TEST**: Async coordination and concurrency
9. **BIOMEDICAL_TEST**: Domain-specific biomedical validation
10. **CLEANUP_TEST**: Resource cleanup validation

### Custom Configuration Overrides

```python
# Custom environment variables
custom_overrides = {
    'environment_vars': {
        'TEST_MODE': 'custom',
        'DEBUG_LEVEL': 'verbose'
    },
    'performance_thresholds': {
        'custom_metric': PerformanceThreshold(
            metric_name='response_time',
            threshold_value=2.0,
            comparison_operator='lt',
            unit='seconds'
        )
    },
    'mock_behaviors': {
        'lightrag_system': MockBehavior.PARTIAL_SUCCESS
    }
}

test_env = create_complete_test_environment(
    TestScenarioType.INTEGRATION_TEST,
    custom_overrides=custom_overrides
)
```

## Resource Cleanup Management

### Automatic Resource Tracking

```python
from configuration_test_utilities import ResourceCleanupManager, ResourceType

cleanup_manager = ResourceCleanupManager()

# Register temporary files
file_id = cleanup_manager.register_temporary_file(temp_file_path)

# Register temporary directories  
dir_id = cleanup_manager.register_temporary_directory(temp_dir_path)

# Register processes
process_id = cleanup_manager.register_process(subprocess_obj)

# Register async tasks
task_id = cleanup_manager.register_async_task(asyncio_task)

# Automatic cleanup
cleanup_manager.cleanup_all_resources()
```

### Memory Leak Detection

The framework automatically monitors memory usage and detects potential leaks:

```python
# Memory monitoring happens automatically
cleanup_stats = cleanup_manager.get_cleanup_statistics()
print(f"Memory leaks detected: {cleanup_stats['memory_leaks_detected']}")
print(f"Current memory: {cleanup_stats['current_memory_mb']}MB")
```

## Async Test Coordination

### Using AsyncTestCoordinator

```python
from configuration_test_utilities import managed_test_environment, TestScenarioType

async def test_async_operations():
    async with managed_test_environment(TestScenarioType.ASYNC_TEST) as test_env:
        coordinator = test_env['async_coordinator']
        
        # Create session
        session_id = await coordinator.create_session("test_session")
        
        # Execute concurrent operations
        async def operation_1():
            return await some_async_function()
        
        async def operation_2():
            return await another_async_function()
        
        results = await asyncio.gather(operation_1(), operation_2())
        
        # Cleanup handled automatically
```

## Performance Testing

### Performance Thresholds

```python
from performance_test_utilities import PerformanceThreshold

# Define performance expectations
execution_threshold = PerformanceThreshold(
    metric_name="execution_time",
    threshold_value=5.0,
    comparison_operator="lt", 
    unit="seconds"
)

memory_threshold = PerformanceThreshold(
    metric_name="memory_usage",
    threshold_value=256.0,
    comparison_operator="lt",
    unit="MB"
)

# Use in configuration
custom_overrides = {
    'performance_thresholds': {
        'execution': execution_threshold,
        'memory': memory_threshold
    }
}
```

### Performance Validation

```python
async def test_with_performance_validation(standard_performance_test_config):
    test_env = standard_performance_test_config
    performance_helper = test_env['performance_helper']
    
    # Execute operation with timing
    start_time = time.time()
    result = await some_operation()
    duration = time.time() - start_time
    
    # Validate against thresholds
    # (Implementation depends on PerformanceAssertionHelper interface)
    assert duration < 5.0  # Manual validation
```

## Biomedical Content Validation

### Domain-Specific Validation

```python
async def test_biomedical_content(biomedical_test_config):
    test_env = biomedical_test_config
    lightrag_mock = test_env['mock_system']['lightrag_system']
    
    # Test clinical metabolomics query
    response = await lightrag_mock.aquery(
        "What metabolites are associated with diabetes progression?"
    )
    
    # Validate biomedical content
    response_lower = response.lower()
    biomedical_terms = ['metabolite', 'glucose', 'diabetes', 'biomarker']
    found_terms = [term for term in biomedical_terms if term in response_lower]
    
    assert len(found_terms) >= 2, f"Should contain biomedical terms: {found_terms}"
    assert len(response) > 100, "Response should be substantial"
```

## Configuration Validation

### Validate Test Environment

```python
from configuration_test_utilities import validate_test_configuration

# Validate configuration
validation_errors = validate_test_configuration(test_env)

if validation_errors:
    print("Configuration issues found:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

### Environment Health Monitoring

```python
# Check environment health
environment_manager = test_env['environment_manager']
health = environment_manager.check_system_health()

print(f"Memory usage: {health['memory_usage_mb']:.1f}MB")
print(f"CPU usage: {health['cpu_percent']:.1f}%") 
print(f"Open files: {health['open_files']}")
print(f"Active threads: {health['active_threads']}")
```

## Best Practices

### 1. Use Managed Environments

Always prefer `managed_test_environment()` for automatic cleanup:

```python
# Good
async with managed_test_environment(scenario_type) as test_env:
    # Test code here
    pass  # Automatic cleanup

# Avoid manual cleanup when possible
test_env = create_complete_test_environment(scenario_type)
try:
    # Test code
    pass
finally:
    # Manual cleanup required
    test_env['config_helper'].cleanup_configuration(test_env['context_id'])
```

### 2. Choose Appropriate Scenario Types

- **UNIT_TEST**: Simple, isolated tests
- **INTEGRATION_TEST**: Component interaction tests
- **PERFORMANCE_TEST**: When performance matters
- **BIOMEDICAL_TEST**: Domain-specific validation needed
- **ASYNC_TEST**: Heavy async operations

### 3. Register Resources for Cleanup

```python
cleanup_manager = test_env['cleanup_manager']

# Always register temporary resources
temp_file = Path("temp.txt")
temp_file.write_text("data")
cleanup_manager.register_temporary_file(temp_file)

# Register processes
process = subprocess.Popen(['command'])
cleanup_manager.register_process(process)
```

### 4. Validate Configurations

```python
# Always validate critical test configurations
validation_errors = validate_test_configuration(test_env)
assert not validation_errors, f"Configuration invalid: {validation_errors}"
```

### 5. Monitor Resource Usage

```python
# Monitor resources in long-running tests
if test_env['environment_manager']:
    health = test_env['environment_manager'].check_system_health()
    assert health['memory_usage_mb'] < 512, "Memory usage too high"
```

## Error Handling

### Configuration Errors

```python
try:
    test_env = create_complete_test_environment(TestScenarioType.INTEGRATION_TEST)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Setup error: {e}")
```

### Cleanup Errors

```python
# Force cleanup in case of errors
try:
    # Test operations
    pass
finally:
    if 'config_helper' in test_env:
        test_env['config_helper'].cleanup_all_configurations(force=True)
```

## Advanced Usage

### Custom Configuration Templates

```python
from configuration_test_utilities import ConfigurationTemplate, ConfigurationTestHelper

# Define custom template
custom_template = ConfigurationTemplate(
    name="Custom Clinical Test",
    scenario_type=TestScenarioType.BIOMEDICAL_TEST,
    description="Specialized clinical testing configuration",
    environment_spec=EnvironmentSpec(
        temp_dirs=["logs", "clinical_data", "results"],
        required_imports=["custom_clinical_module"],
        async_context=True
    ),
    mock_components=[SystemComponent.LIGHTRAG_SYSTEM],
    performance_thresholds={
        "clinical_response_time": PerformanceThreshold(
            metric_name="clinical_response_time",
            threshold_value=10.0,
            comparison_operator="lt",
            unit="seconds"
        )
    },
    async_config={"enabled": True, "max_concurrent_operations": 3},
    validation_rules=["clinical_accuracy", "domain_validation"]
)

# Register and use custom template
config_helper = ConfigurationTestHelper()
config_helper.register_custom_template(custom_template)

context_id = config_helper.create_test_configuration(TestScenarioType.BIOMEDICAL_TEST)
```

### Environment Isolation

```python
from configuration_test_utilities import EnvironmentIsolationManager

isolation_manager = EnvironmentIsolationManager()

with isolation_manager.isolated_environment(
    environment_vars={'ISOLATED_MODE': 'true'},
    working_directory=Path('/tmp/isolated_test'),
    sys_path_additions=['/custom/path']
):
    # Test code runs in isolated environment
    assert os.environ['ISOLATED_MODE'] == 'true'
    # Environment restored automatically when exiting
```

## Troubleshooting

### Common Issues

1. **Configuration Validation Failures**
   ```python
   # Check validation errors
   errors = validate_test_configuration(test_env)
   if errors:
       for error in errors:
           print(f"Validation error: {error}")
   ```

2. **Resource Cleanup Issues**
   ```python
   # Force cleanup if needed
   cleanup_manager.cleanup_all_resources(force=True)
   
   # Check cleanup statistics
   stats = cleanup_manager.get_cleanup_statistics()
   print(f"Cleanup failures: {stats['cleanup_failures']}")
   ```

3. **Memory Issues**
   ```python
   # Monitor memory usage
   health = environment_manager.check_system_health()
   if health['memory_usage_mb'] > 500:
       print("High memory usage detected")
   ```

4. **Mock System Issues**
   ```python
   # Verify mock system
   mock_system = test_env.get('mock_system', {})
   if not mock_system:
       print("Mock system not configured")
   ```

### Performance Optimization

- Use `UNIT_TEST` scenario for simple tests
- Enable performance monitoring only when needed
- Register cleanup callbacks for heavy resources
- Use async coordination for concurrent operations

## Integration Examples

See the following files for complete examples:

- `demo_configuration_test_utilities.py`: Complete framework demonstration
- `example_complete_test_framework.py`: Practical test implementations
- `test_comprehensive_*.py`: Real-world test scenarios

## Framework Statistics

```python
# Get framework usage statistics
config_helper = test_env['config_helper']
stats = config_helper.get_configuration_statistics()

cleanup_manager = test_env['cleanup_manager'] 
cleanup_stats = cleanup_manager.get_cleanup_statistics()

print(f"Active configurations: {stats['active_contexts']}")
print(f"Resources tracked: {cleanup_stats['active_resources']}")
print(f"Memory usage: {cleanup_stats['current_memory_mb']}MB")
```

This complete framework provides everything needed for comprehensive Clinical Metabolomics Oracle testing with automated configuration, resource management, and validation.