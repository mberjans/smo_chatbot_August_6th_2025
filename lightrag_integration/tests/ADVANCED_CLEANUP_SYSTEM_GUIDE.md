# Advanced Cleanup System Implementation Guide

## Overview

The Advanced Cleanup System for Clinical Metabolomics Oracle LightRAG Integration provides comprehensive cleanup mechanisms that go beyond basic fixture cleanup. This system offers robust resource management, failure recovery, performance monitoring, and seamless integration with existing test infrastructure.

## Table of Contents

- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Quick Start Guide](#quick-start-guide)
- [Integration with Existing Fixtures](#integration-with-existing-fixtures)
- [Configuration Options](#configuration-options)
- [Advanced Features](#advanced-features)
- [Performance Monitoring](#performance-monitoring)
- [Failure Recovery](#failure-recovery)
- [Reporting and Analytics](#reporting-and-analytics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## System Architecture

The Advanced Cleanup System consists of three main modules:

```
advanced_cleanup_system.py          # Core resource management and orchestration
├── AdvancedCleanupOrchestrator     # Central cleanup coordinator
├── ResourceManager classes         # Specialized resource cleanup
└── CleanupPolicy/Thresholds       # Configuration management

cleanup_validation_monitor.py       # Monitoring, validation, and reporting
├── CleanupValidator               # Cleanup effectiveness validation  
├── ResourceMonitor               # Resource usage monitoring
├── PerformanceAnalyzer          # Performance analysis and optimization
├── CleanupReporter              # Report generation
└── AlertSystem                  # Automated alerting

advanced_cleanup_integration.py     # Pytest integration and bridges
├── AdvancedCleanupIntegrator      # Integration coordinator
├── Fixture bridges               # Connect with existing fixtures
└── Pytest hooks                 # Automatic lifecycle management
```

## Core Components

### 1. Advanced Cleanup Orchestrator
Central coordinator for all cleanup operations with support for:
- Multiple cleanup strategies (immediate, deferred, scheduled, resource-based)
- Resource type management (memory, file handles, database connections, processes, temp files)
- Retry logic and failure recovery
- Performance monitoring and validation

### 2. Resource Managers
Specialized managers for different resource types:
- **MemoryManager**: Garbage collection and memory optimization
- **FileHandleManager**: File closure and handle cleanup
- **DatabaseConnectionManager**: Database connection lifecycle
- **ProcessManager**: Subprocess termination and thread pool shutdown
- **TemporaryFileManager**: Temporary file and directory cleanup

### 3. Cleanup Validation Monitor
Comprehensive monitoring and validation system:
- Real-time resource usage monitoring
- Cleanup effectiveness validation
- Performance analysis and optimization recommendations
- Automated report generation
- Alert system for cleanup issues

### 4. Integration Bridge
Seamless integration with existing pytest infrastructure:
- Automatic resource registration
- Bridge between old and new cleanup systems
- Pytest lifecycle integration
- Performance tracking during tests

## Quick Start Guide

### Basic Usage

```python
from advanced_cleanup_system import AdvancedCleanupOrchestrator, ResourceType

# Create orchestrator with default settings
orchestrator = AdvancedCleanupOrchestrator()

# Register resources
orchestrator.register_resource(ResourceType.FILE_HANDLES, file_obj)
orchestrator.register_resource(ResourceType.DATABASE_CONNECTIONS, conn)
orchestrator.register_resource(ResourceType.TEMPORARY_FILES, temp_path)

# Perform cleanup
success = orchestrator.cleanup()
```

### With Monitoring and Validation

```python
from cleanup_validation_monitor import CleanupValidationMonitor

# Create full monitoring system
monitor = CleanupValidationMonitor()

with monitor.monitoring_context():
    # Your test code here
    monitor.orchestrator.register_resource(ResourceType.MEMORY, large_object)
    
    # Perform cleanup cycle with validation
    result = monitor.perform_cleanup_cycle()
    
    # Generate comprehensive report
    report = monitor.generate_comprehensive_report()
```

### Pytest Integration

```python
# In your test file
def test_with_advanced_cleanup(advanced_cleanup_bridge):
    # Use the bridge to register resources
    advanced_cleanup_bridge.register_file(file_obj)
    advanced_cleanup_bridge.register_db_connection(conn)
    
    # Test operations...
    
    # Cleanup happens automatically
```

## Integration with Existing Fixtures

The system seamlessly integrates with existing `TestDataManager` fixtures:

### Automatic Integration

```python
@pytest.fixture
def integrated_test_manager(test_data_manager, advanced_cleanup_bridge):
    """Integrated fixture combining existing and advanced cleanup."""
    
    # Bridge automatically integrates with existing manager
    bridge = advanced_cleanup_bridge
    
    # Register resources with both systems
    def register_resource(resource_type, resource):
        if resource_type == "file":
            bridge.register_file(resource)
        elif resource_type == "db":
            bridge.register_db_connection(resource)
        elif resource_type == "temp":
            bridge.register_temp_path(resource)
    
    return register_resource
```

### Manual Integration

```python
# Create integration bridge manually
from advanced_cleanup_integration import AdvancedCleanupIntegrator

integrator = AdvancedCleanupIntegrator()
integrator.register_test_data_manager(test_data_manager, "test_id")
bridge = integrator.create_integrated_fixture_bridge(test_data_manager)

# Use bridge for dual registration
bridge.register_file(file_obj, auto_close=True)
bridge.register_db_connection(conn)
bridge.perform_cleanup()  # Cleans both systems
```

## Configuration Options

### Cleanup Policies

```python
from advanced_cleanup_system import CleanupPolicy, CleanupStrategy, ResourceType

policy = CleanupPolicy(
    strategy=CleanupStrategy.RESOURCE_BASED,  # When to cleanup
    scope=CleanupScope.FUNCTION,             # Cleanup scope
    resource_types={ResourceType.MEMORY, ResourceType.FILE_HANDLES},
    max_retry_attempts=3,                    # Retry failed cleanups
    retry_delay_seconds=1.0,                # Delay between retries
    timeout_seconds=30.0,                   # Cleanup timeout
    force_cleanup=True,                     # Force cleanup on timeout
    validate_cleanup=True,                  # Validate after cleanup
    report_cleanup=True,                    # Generate reports
    emergency_cleanup=True                  # Handle signals for emergency cleanup
)
```

### Resource Thresholds

```python
from advanced_cleanup_system import ResourceThresholds

thresholds = ResourceThresholds(
    memory_mb=512,        # Memory usage threshold
    file_handles=100,     # Open file handle threshold
    db_connections=10,    # Database connection threshold
    temp_files=50,        # Temporary file count threshold
    temp_size_mb=200,     # Temporary file size threshold
    cache_entries=1000    # Cache entry threshold
)
```

### Integration Configuration

```python
from advanced_cleanup_integration import CleanupIntegrationConfig

config = CleanupIntegrationConfig(
    enabled=True,                    # Enable integration
    auto_register_resources=True,    # Auto-register from existing managers
    monitor_performance=True,        # Track performance
    generate_reports=True,          # Generate reports
    validate_cleanup=True,          # Validate cleanup effectiveness
    enable_alerts=False,            # Alerts (usually disabled for tests)
    cleanup_on_failure=True,        # Cleanup even if tests fail
    
    # Performance settings
    max_cleanup_time_seconds=10.0,
    performance_threshold_multiplier=2.0,
    
    # Test-friendly resource thresholds
    memory_threshold_mb=256,
    file_handle_threshold=50,
    db_connection_threshold=5
)
```

## Advanced Features

### Custom Resource Managers

```python
from advanced_cleanup_system import ResourceManager

class CustomResourceManager(ResourceManager):
    def cleanup(self) -> bool:
        # Implement custom cleanup logic
        try:
            # Your cleanup code here
            self.record_cleanup_attempt(True, "Custom cleanup successful")
            return True
        except Exception as e:
            self.record_cleanup_attempt(False, str(e))
            return False
    
    def validate_cleanup(self) -> bool:
        # Implement validation logic
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        # Return current usage metrics
        return {"custom_metric": 42}
```

### Async Cleanup Support

```python
from advanced_cleanup_system import AdvancedCleanupOrchestrator

orchestrator = AdvancedCleanupOrchestrator()

# Async context manager
async with orchestrator.async_cleanup_context():
    # Your async test code
    await some_async_operation()
    # Cleanup happens automatically
```

### Scheduled Cleanup

```python
policy = CleanupPolicy(strategy=CleanupStrategy.SCHEDULED)
orchestrator = AdvancedCleanupOrchestrator(policy)

# Cleanup will run on schedule (every 5 minutes by default)
# Override _should_run_scheduled_cleanup() for custom scheduling
```

## Performance Monitoring

### Real-time Monitoring

```python
from cleanup_validation_monitor import ResourceMonitor

monitor = ResourceMonitor(sample_interval_seconds=30)
monitor.start_monitoring(orchestrator)

# Get trend analysis
trends = monitor.get_trend_analysis(ResourceType.MEMORY, hours=24)
print(f"Memory trend: {trends}")
```

### Performance Analysis

```python
from cleanup_validation_monitor import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Record cleanup operation
start_time = datetime.now()
# ... perform cleanup ...
end_time = datetime.now()

metrics = analyzer.record_cleanup_operation(
    orchestrator, resource_types, start_time, end_time
)

# Analyze trends
analysis = analyzer.analyze_performance_trends(days=7)
optimizations = analyzer.identify_optimization_opportunities()
```

## Failure Recovery

### Retry Mechanisms

The system includes built-in retry mechanisms with exponential backoff:

```python
policy = CleanupPolicy(
    max_retry_attempts=5,
    retry_delay_seconds=1.0,  # Doubles each retry
    timeout_seconds=60.0
)

# Failed cleanups are automatically retried
success = orchestrator.cleanup()
```

### Emergency Cleanup

Emergency cleanup handles process signals and unexpected shutdowns:

```python
# Emergency cleanup is triggered on SIGTERM/SIGINT
orchestrator.force_cleanup()  # Manual emergency cleanup
```

### Partial Failure Handling

```python
# System continues with partial failures
resource_results = orchestrator.cleanup(force=True)

# Check individual resource cleanup results
for resource_type, manager in orchestrator._resource_managers.items():
    if manager._failed_cleanups:
        print(f"Failures in {resource_type}: {manager._failed_cleanups}")
```

## Reporting and Analytics

### Comprehensive Reports

```python
from cleanup_validation_monitor import CleanupReporter

reporter = CleanupReporter(report_dir=Path("reports/cleanup"))
report = reporter.generate_comprehensive_report(
    orchestrator, validator, monitor, analyzer
)

# Reports include:
# - System overview and resource status
# - Cleanup statistics and performance metrics
# - Validation results and health assessment
# - Trend analysis and optimization opportunities
```

### Health Assessment

```python
# Health scores from 0-100 based on:
# - Resource usage vs thresholds
# - Cleanup success rates
# - Performance trends
# - Validation results

health = report['health_assessment']
print(f"System health: {health['status']} ({health['health_score']}/100)")
```

### Alerting System

```python
from cleanup_validation_monitor import AlertSystem, AlertConfig

alert_config = AlertConfig(
    enabled=True,
    memory_threshold_mb=1024,
    file_handle_threshold=500,
    cleanup_failure_threshold=3
)

alert_system = AlertSystem(alert_config)
alerts = alert_system.check_and_alert(orchestrator, monitor, validator)

for alert in alerts:
    print(f"Alert: {alert['message']}")
```

## Best Practices

### 1. Resource Registration

```python
# Register resources as soon as they're created
file_obj = open("test_file.txt", "w")
orchestrator.register_resource(ResourceType.FILE_HANDLES, file_obj)

# Use context managers when possible
with orchestrator.cleanup_context():
    # Resources registered here are automatically cleaned up
    pass
```

### 2. Test Configuration

```python
# Use test-friendly configurations
config = CleanupIntegrationConfig(
    memory_threshold_mb=128,    # Lower thresholds for tests
    max_cleanup_time_seconds=5.0,  # Shorter timeouts
    enable_alerts=False,        # Disable alerts in tests
    generate_reports=False      # Disable reports for unit tests
)
```

### 3. Integration Patterns

```python
# Pattern 1: Automatic integration via fixtures
def test_with_auto_cleanup(advanced_cleanup_bridge):
    bridge.register_file(file_obj)
    # Test code...
    # Cleanup automatic

# Pattern 2: Manual control
def test_with_manual_cleanup(advanced_cleanup_orchestrator):
    orchestrator.register_resource(ResourceType.MEMORY, large_obj)
    # Test code...
    orchestrator.cleanup()  # Manual cleanup

# Pattern 3: Context manager
def test_with_context():
    with advanced_cleanup_context() as integrator:
        # Test code with automatic cleanup
        pass
```

### 4. Performance Optimization

```python
# Use resource-based cleanup for efficiency
policy = CleanupPolicy(strategy=CleanupStrategy.RESOURCE_BASED)

# Limit resource types to what you actually use
policy.resource_types = {ResourceType.FILE_HANDLES, ResourceType.TEMPORARY_FILES}

# Use appropriate thresholds
thresholds = ResourceThresholds(
    memory_mb=512,      # Realistic for your tests
    file_handles=100,   # Based on your usage patterns
)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Make sure you're in the correct directory
import sys
sys.path.append('/path/to/tests')

# Check module availability
try:
    from advanced_cleanup_system import AdvancedCleanupOrchestrator
except ImportError as e:
    print(f"Import error: {e}")
```

#### 2. Cleanup Failures
```python
# Check cleanup statistics for failure details
stats = orchestrator.get_cleanup_statistics()
print(f"Failed operations: {stats['total_operations'] - stats['successful_operations']}")

# Enable debug logging
import logging
logging.getLogger('advanced_cleanup_system').setLevel(logging.DEBUG)

# Use force cleanup for stubborn resources
orchestrator.force_cleanup()
```

#### 3. Performance Issues
```python
# Check performance metrics
if hasattr(monitor, 'analyzer'):
    trends = monitor.analyzer.analyze_performance_trends()
    optimizations = monitor.analyzer.identify_optimization_opportunities()
    print(f"Optimizations: {optimizations}")

# Reduce cleanup scope
policy.resource_types = {ResourceType.FILE_HANDLES}  # Only essential types
```

#### 4. Integration Problems
```python
# Check integration status
integrator = get_cleanup_integrator()
stats = integrator.get_integration_statistics()
print(f"Integration active: {stats['session_stats']['integration_active']}")

# Verify fixture registration
print(f"Registered managers: {stats['session_stats']['registered_managers']}")
```

### Debugging Tools

```python
# Enable comprehensive debugging
logging.basicConfig(level=logging.DEBUG)

# Get detailed resource usage
usage = orchestrator.get_resource_usage()
for resource_type, metrics in usage.items():
    print(f"{resource_type}: {metrics}")

# Generate debug report
report = monitor.generate_comprehensive_report()
print(f"Debug report: {report['report_id']}")

# Check validation details
validation_results = validator.validate_cleanup(orchestrator)
for resource_type, result in validation_results.items():
    if not result.success:
        print(f"Validation failed for {resource_type}: {result.issues}")
```

## API Reference

### Core Classes

#### AdvancedCleanupOrchestrator
- `__init__(policy, thresholds)`: Initialize orchestrator
- `register_resource(resource_type, resource)`: Register resource for cleanup
- `cleanup(force=False, resource_types=None)`: Perform cleanup
- `validate_cleanup(resource_types=None)`: Validate cleanup effectiveness
- `get_resource_usage()`: Get current resource usage metrics
- `get_cleanup_statistics()`: Get cleanup statistics

#### CleanupValidationMonitor
- `__init__(cleanup_policy, thresholds, alert_config, report_dir)`: Initialize monitor
- `start_monitoring()`: Start resource monitoring
- `perform_cleanup_cycle()`: Perform cleanup with validation
- `generate_comprehensive_report()`: Generate full report

#### AdvancedCleanupIntegrator
- `__init__(config)`: Initialize integration system
- `register_test_data_manager(manager, test_id)`: Register existing manager
- `create_integrated_fixture_bridge(manager)`: Create integration bridge

### Resource Types
- `ResourceType.MEMORY`: Memory and garbage collection
- `ResourceType.FILE_HANDLES`: File handles and descriptors
- `ResourceType.DATABASE_CONNECTIONS`: Database connections
- `ResourceType.PROCESSES`: Subprocesses and thread pools
- `ResourceType.TEMPORARY_FILES`: Temporary files and directories
- `ResourceType.NETWORK_CONNECTIONS`: Network connections
- `ResourceType.THREADS`: Thread management
- `ResourceType.CACHE_ENTRIES`: Cache cleanup

### Cleanup Strategies
- `CleanupStrategy.IMMEDIATE`: Clean up immediately after use
- `CleanupStrategy.DEFERRED`: Clean up at end of scope
- `CleanupStrategy.SCHEDULED`: Clean up on schedule
- `CleanupStrategy.ON_DEMAND`: Clean up only when requested
- `CleanupStrategy.RESOURCE_BASED`: Clean up based on thresholds

## Example Configurations

### Production-Like Testing
```python
config = CleanupIntegrationConfig(
    memory_threshold_mb=1024,
    file_handle_threshold=500,
    monitor_performance=True,
    generate_reports=True,
    validate_cleanup=True
)
```

### Fast Unit Testing
```python
config = CleanupIntegrationConfig(
    memory_threshold_mb=128,
    file_handle_threshold=50,
    monitor_performance=False,
    generate_reports=False,
    max_cleanup_time_seconds=2.0
)
```

### Debug Mode
```python
config = CleanupIntegrationConfig(
    enable_alerts=True,
    validate_cleanup=True,
    generate_reports=True,
    cleanup_on_failure=True,
    emergency_cleanup=True
)
```

## Demo and Testing

Run the comprehensive demo to see all features in action:

```bash
cd /path/to/tests
python demo_advanced_cleanup_system.py
```

The demo includes:
1. Basic resource management
2. Monitoring and validation
3. Test infrastructure integration
4. Performance analysis
5. Failure handling
6. Comprehensive reporting

Check the generated reports in `test_data/reports/cleanup/` for detailed analysis.

---

For more information and updates, see the individual module documentation and the project's test suite examples.