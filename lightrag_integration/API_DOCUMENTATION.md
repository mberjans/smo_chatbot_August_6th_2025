# Graceful Degradation System - API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core APIs](#core-apis)
3. [Configuration APIs](#configuration-apis)
4. [Monitoring APIs](#monitoring-apis)
5. [Integration APIs](#integration-apis)
6. [Request Submission APIs](#request-submission-apis)
7. [Event and Callback APIs](#event-and-callback-apis)
8. [Utility APIs](#utility-apis)
9. [Error Handling](#error-handling)
10. [Code Examples](#code-examples)

## Overview

The Graceful Degradation System provides a comprehensive set of APIs for programmatic interaction with all system components. These APIs enable integration, configuration management, monitoring, and operational control of the graceful degradation features.

### API Categories

- **Core APIs**: Main system lifecycle and orchestration
- **Configuration APIs**: Runtime configuration management
- **Monitoring APIs**: System metrics and health monitoring
- **Integration APIs**: Integration with external systems
- **Request APIs**: Request submission and management
- **Event APIs**: Event handling and callbacks
- **Utility APIs**: Helper functions and utilities

## Core APIs

### GracefulDegradationOrchestrator

The main orchestrator class that coordinates all graceful degradation components.

#### Constructor

```python
class GracefulDegradationOrchestrator:
    def __init__(self,
                 config: Optional[GracefulDegradationConfig] = None,
                 load_balancer: Optional[Any] = None,
                 rag_system: Optional[Any] = None,
                 monitoring_system: Optional[Any] = None)
```

**Parameters:**
- `config`: Configuration object for the system
- `load_balancer`: Production load balancer instance
- `rag_system`: Clinical metabolomics RAG system instance  
- `monitoring_system`: Production monitoring system instance

**Example:**
```python
from graceful_degradation_integration import (
    GracefulDegradationOrchestrator, 
    GracefulDegradationConfig
)

config = GracefulDegradationConfig(
    monitoring_interval=5.0,
    base_rate_per_second=50.0,
    max_queue_size=1000
)

orchestrator = GracefulDegradationOrchestrator(
    config=config,
    load_balancer=my_load_balancer,
    rag_system=my_rag_system
)
```

#### Lifecycle Methods

##### `async start()`

Start the complete graceful degradation system.

**Returns:** `None`
**Raises:** `RuntimeError` if startup fails

```python
await orchestrator.start()
```

##### `async stop()`

Stop the complete graceful degradation system.

**Returns:** `None`

```python
await orchestrator.stop()
```

#### System Status Methods

##### `get_system_status() -> Dict[str, Any]`

Get comprehensive status of the entire graceful degradation system.

**Returns:** Dictionary containing detailed system status

```python
status = orchestrator.get_system_status()

# Example response:
{
    'running': True,
    'start_time': '2025-08-09T10:30:00',
    'uptime_seconds': 3600.0,
    'integration_status': {
        'load_monitoring_active': True,
        'degradation_controller_active': True,
        'throttling_system_active': True,
        'integrated_load_balancer': True,
        'integrated_rag_system': True,
        'integrated_monitoring': True
    },
    'current_load_level': 'NORMAL',
    'last_level_change': '2025-08-09T10:25:00',
    'total_requests_processed': 15000,
    'health_status': 'healthy',
    'active_issues': [],
    'load_monitoring': {...},
    'degradation_controller': {...},
    'throttling_system': {...},
    'production_integration': {...}
}
```

##### `get_health_check() -> Dict[str, Any]`

Get health check for the entire system.

**Returns:** Dictionary containing health status and issues

```python
health = orchestrator.get_health_check()

# Example response:
{
    'status': 'healthy',  # 'healthy', 'degraded', 'critical'
    'issues': [],
    'uptime_seconds': 3600.0,
    'current_load_level': 'NORMAL',
    'total_requests_processed': 15000,
    'component_status': {
        'load_monitoring': 'active',
        'degradation_controller': 'active',
        'throttling_system': 'active'
    },
    'production_integration': {
        'load_balancer': 'integrated',
        'rag_system': 'integrated',
        'monitoring': 'integrated'
    }
}
```

## Configuration APIs

### GracefulDegradationConfig

Configuration data class for the graceful degradation system.

```python
@dataclass
class GracefulDegradationConfig:
    # Monitoring configuration
    monitoring_interval: float = 5.0
    enable_trend_analysis: bool = True
    hysteresis_enabled: bool = True
    
    # Throttling configuration
    base_rate_per_second: float = 10.0
    max_queue_size: int = 1000
    max_concurrent_requests: int = 50
    starvation_threshold: float = 300.0
    
    # Connection pool configuration
    base_pool_size: int = 20
    max_pool_size: int = 100
    
    # Integration configuration
    auto_start_monitoring: bool = True
    enable_production_integration: bool = True
    metrics_retention_hours: int = 24
    
    # Emergency handling
    emergency_max_duration: float = 300.0
    auto_recovery_enabled: bool = True
    circuit_breaker_enabled: bool = True
```

#### Class Methods

##### `load_from_file(config_path: str) -> GracefulDegradationConfig`

Load configuration from JSON or YAML file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:** `GracefulDegradationConfig` instance

```python
config = GracefulDegradationConfig.load_from_file('config/production.yaml')
```

##### `from_dict(data: Dict[str, Any]) -> GracefulDegradationConfig`

Create configuration from dictionary.

**Parameters:**
- `data`: Configuration data dictionary

**Returns:** `GracefulDegradationConfig` instance

```python
config_data = {
    'monitoring_interval': 3.0,
    'base_rate_per_second': 75.0,
    'max_queue_size': 2000
}
config = GracefulDegradationConfig.from_dict(config_data)
```

### Runtime Configuration Updates

#### `update_configuration(config_data: Dict[str, Any])`

Update system configuration at runtime.

**Parameters:**
- `config_data`: New configuration data

```python
new_config = {
    'throttling': {
        'base_rate_per_second': 100.0,
        'max_concurrent_requests': 150
    },
    'monitoring': {
        'interval': 3.0
    }
}

await orchestrator.configuration_manager.update_configuration(new_config)
```

## Monitoring APIs

### Load Detection System

#### `get_system_metrics() -> SystemLoadMetrics`

Get current system load metrics.

**Returns:** `SystemLoadMetrics` object

```python
metrics = orchestrator.load_detector.get_system_metrics()

# Example response:
SystemLoadMetrics(
    timestamp=datetime(2025, 8, 9, 10, 30, 0),
    cpu_utilization=45.2,
    memory_pressure=62.1,
    request_queue_depth=15,
    response_time_p95=850.0,
    response_time_p99=1200.0,
    error_rate=0.12,
    active_connections=42,
    disk_io_wait=5.3,
    load_level=SystemLoadLevel.NORMAL,
    load_score=0.32,
    degradation_recommended=False
)
```

#### `get_metrics_history(hours: int = 1) -> List[Dict[str, Any]]`

Get historical metrics data.

**Parameters:**
- `hours`: Number of hours of history to retrieve

**Returns:** List of metrics dictionaries

```python
history = orchestrator.get_metrics_history(hours=24)

for metric in history[-10:]:  # Last 10 measurements
    print(f"{metric['timestamp']}: {metric['load_level']} - "
          f"CPU: {metric['cpu_utilization']:.1f}%")
```

#### `add_load_change_callback(callback: Callable)`

Add callback for load level changes.

**Parameters:**
- `callback`: Function to call when load level changes

```python
def on_load_change(load_level, metrics):
    print(f"Load level changed to: {load_level.name}")

orchestrator.load_detector.add_load_change_callback(on_load_change)
```

### Performance Metrics

#### `record_request_metrics(response_time_ms: float, error: bool = False)`

Record request performance metrics.

**Parameters:**
- `response_time_ms`: Request response time in milliseconds
- `error`: Whether the request resulted in an error

```python
# Record successful request with 850ms response time
orchestrator.record_request_metrics(850.0, error=False)

# Record failed request with 5000ms response time
orchestrator.record_request_metrics(5000.0, error=True)
```

## Integration APIs

### Production System Integration

#### `integrate_with_production_system(production_load_balancer, fallback_orchestrator, clinical_rag)`

Integrate with existing production components.

**Parameters:**
- `production_load_balancer`: Production load balancer instance
- `fallback_orchestrator`: Fallback orchestrator instance
- `clinical_rag`: Clinical RAG system instance

```python
orchestrator.integrate_with_production_system(
    production_load_balancer=load_balancer,
    fallback_orchestrator=fallback_system,
    clinical_rag=rag_system
)
```

#### `add_load_change_callback(callback: Callable)`

Add callback for system-wide load level changes.

**Parameters:**
- `callback`: Function with signature `(SystemLoadLevel, SystemLoadMetrics) -> None`

```python
def handle_load_change(load_level, metrics):
    if load_level >= SystemLoadLevel.HIGH:
        # Take action for high load
        scale_up_resources()
    
orchestrator.add_load_change_callback(handle_load_change)
```

### System Adapters

#### `register_system_adapter(name: str, system: Any, adapter_type: str)`

Register a custom system adapter.

**Parameters:**
- `name`: Unique name for the system
- `system`: System instance to integrate
- `adapter_type`: Type of adapter ('load_balancer', 'rag_system', 'monitoring', 'custom')

```python
# Register custom monitoring system
orchestrator.production_integrator.register_system(
    name='custom_metrics',
    system=my_metrics_system,
    adapter_type='monitoring'
)
```

## Request Submission APIs

### Request Submission

#### `async submit_request(request_type: str, priority: Optional[str] = None, handler: Optional[Callable] = None, **kwargs) -> Tuple[bool, str, str]`

Submit a request through the integrated throttling system.

**Parameters:**
- `request_type`: Type of request ('user_query', 'batch_processing', 'health_check', etc.)
- `priority`: Request priority ('critical', 'high', 'medium', 'low', 'background')
- `handler`: Request handler function
- `**kwargs`: Additional arguments for the handler

**Returns:** Tuple of `(success: bool, message: str, request_id: str)`

```python
# Submit high-priority user query
success, message, request_id = await orchestrator.submit_request(
    request_type='user_query',
    priority='high',
    handler=process_metabolomics_query,
    query="analyze biomarker patterns",
    user_id="user123"
)

if success:
    print(f"Request {request_id} submitted successfully")
else:
    print(f"Request failed: {message}")

# Submit background processing task
success, message, request_id = await orchestrator.submit_request(
    request_type='batch_processing',
    priority='background',
    handler=process_batch_data,
    dataset_id="metabolomics_2024",
    batch_size=1000
)
```

### Request Types and Priorities

#### Request Types

- `'health_check'`: System health verification
- `'user_query'`: Interactive user queries  
- `'batch_processing'`: Batch data processing
- `'analytics'`: Reporting and analytics
- `'maintenance'`: Background maintenance
- `'admin'`: Administrative operations

#### Request Priorities

- `'critical'`: Highest priority, always processed
- `'high'`: High priority, processed before medium/low
- `'medium'`: Normal priority
- `'low'`: Lower priority
- `'background'`: Lowest priority, processed when resources available

### Request Status and Tracking

#### Request Handler Interface

```python
async def custom_request_handler(**kwargs) -> Any:
    """
    Custom request handler function.
    
    Args:
        **kwargs: Arguments passed from submit_request
        
    Returns:
        Any: Result of the request processing
        
    Raises:
        Exception: Any processing errors
    """
    # Process the request
    result = await process_request(kwargs)
    return result

# Example metabolomics query handler
async def metabolomics_query_handler(query: str, user_id: str) -> Dict[str, Any]:
    """Handle metabolomics query requests."""
    try:
        # Validate query
        if not query or len(query) > 10000:
            raise ValueError("Invalid query length")
        
        # Process query through RAG system
        result = await rag_system.process_query(query, user_id=user_id)
        
        return {
            'status': 'success',
            'result': result,
            'processing_time': result.get('processing_time', 0)
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'query': query
        }
```

## Event and Callback APIs

### Event Types

The system generates various events that can be monitored through callbacks:

- **Load Change Events**: When system load level changes
- **Degradation Events**: When degradation strategies are applied
- **Throttling Events**: When request throttling occurs
- **Health Events**: When system health status changes
- **Integration Events**: When external system integration status changes

### Callback Registration

#### Load Change Callbacks

```python
def load_change_callback(load_level: SystemLoadLevel, metrics: SystemLoadMetrics):
    """Called when system load level changes."""
    print(f"Load level changed to: {load_level.name}")
    
    if load_level >= SystemLoadLevel.CRITICAL:
        # Send alert
        send_critical_load_alert(metrics)
    
orchestrator.add_load_change_callback(load_change_callback)
```

#### Health Check Callbacks

```python
def health_check_callback():
    """Called periodically for custom health checks."""
    health = orchestrator.get_health_check()
    
    if health['status'] == 'critical':
        # Custom alerting logic
        alert_oncall_team(health)

orchestrator.add_health_check_callback(health_check_callback)
```

#### Feature Toggle Callbacks

```python
def feature_toggle_callback(feature: str, enabled: bool):
    """Called when feature availability changes."""
    print(f"Feature '{feature}' {'enabled' if enabled else 'disabled'}")
    
    # Update application behavior
    if feature == 'complex_analytics' and not enabled:
        # Disable complex analytics in application
        app_config.disable_complex_analytics()

orchestrator.feature_controller.add_feature_callback('complex_analytics', feature_toggle_callback)
```

### Custom Event Handlers

#### Integration Event Handler

```python
class CustomIntegrationEventHandler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
    def handle_load_balancer_event(self, event_type: str, data: Dict[str, Any]):
        """Handle load balancer integration events."""
        if event_type == 'backend_failure':
            # Handle backend failure
            failed_backend = data.get('backend_id')
            print(f"Backend {failed_backend} failed - adjusting capacity")
            
            # Reduce rate limits temporarily
            self.orchestrator.configuration_manager.update_runtime_config({
                'throttling': {
                    'base_rate_per_second': self.orchestrator.config.base_rate_per_second * 0.8
                }
            })
    
    def handle_rag_system_event(self, event_type: str, data: Dict[str, Any]):
        """Handle RAG system integration events."""
        if event_type == 'performance_degradation':
            # RAG system performance degrading
            latency = data.get('average_latency_ms', 0)
            
            if latency > 5000:  # 5 second threshold
                # Force HIGH load level to trigger degradation
                self.orchestrator.degradation_controller.force_load_level(
                    SystemLoadLevel.HIGH,
                    f"RAG system performance degraded: {latency}ms average latency"
                )

# Register custom event handlers
event_handler = CustomIntegrationEventHandler(orchestrator)
orchestrator.production_integrator.add_integration_callback('load_balancer', event_handler.handle_load_balancer_event)
orchestrator.production_integrator.add_integration_callback('rag_system', event_handler.handle_rag_system_event)
```

## Utility APIs

### Factory Functions

#### `create_graceful_degradation_system(config, load_balancer, rag_system, monitoring_system, auto_start=True) -> GracefulDegradationOrchestrator`

Create a complete, production-ready graceful degradation system.

```python
from graceful_degradation_integration import create_graceful_degradation_system

orchestrator = create_graceful_degradation_system(
    config=my_config,
    load_balancer=my_load_balancer,
    rag_system=my_rag_system,
    monitoring_system=my_monitoring,
    auto_start=False  # Don't auto-start
)

# Start manually
await orchestrator.start()
```

#### `async create_and_start_graceful_degradation_system(config, load_balancer, rag_system, monitoring_system) -> GracefulDegradationOrchestrator`

Create and start a complete graceful degradation system.

```python
orchestrator = await create_and_start_graceful_degradation_system(
    config=my_config,
    load_balancer=my_load_balancer,
    rag_system=my_rag_system
)
# System is already started and ready to use
```

### Load Threshold Management

#### `LoadThresholds`

Data class for configuring load detection thresholds.

```python
from enhanced_load_monitoring_system import LoadThresholds

custom_thresholds = LoadThresholds(
    # CPU thresholds (percentage)
    cpu_normal=40.0,
    cpu_elevated=55.0,
    cpu_high=70.0,
    cpu_critical=85.0,
    cpu_emergency=92.0,
    
    # Memory thresholds (percentage)
    memory_normal=50.0,
    memory_elevated=65.0,
    memory_high=75.0,
    memory_critical=85.0,
    memory_emergency=90.0,
    
    # Response time thresholds (milliseconds)
    response_p95_normal=800.0,
    response_p95_elevated=1500.0,
    response_p95_high=2500.0,
    response_p95_critical=4000.0,
    response_p95_emergency=6000.0
)

# Apply custom thresholds
orchestrator.load_detector.update_thresholds(custom_thresholds)
```

### System State Utilities

#### `should_accept_request() -> bool`

Determine if new requests should be accepted based on current load.

```python
if orchestrator.should_accept_request():
    # Process the request
    result = await process_user_request(request)
else:
    # Reject request due to high load
    return {"error": "System temporarily unavailable due to high load"}
```

#### `get_timeout_for_service(service: str) -> float`

Get current timeout for a specific service.

```python
# Get current timeout for LightRAG service
lightrag_timeout = orchestrator.get_timeout_for_service('lightrag_query')

# Use timeout in API call
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=lightrag_timeout)) as session:
    async with session.post(lightrag_url, json=query_data) as response:
        result = await response.json()
```

#### `simplify_query_params(query_params: Dict[str, Any]) -> Dict[str, Any]`

Simplify query parameters based on current load level.

```python
# Original query parameters
original_params = {
    'query': user_query,
    'max_total_tokens': 8000,
    'top_k': 10,
    'response_type': 'Comprehensive Analysis',
    'mode': 'hybrid'
}

# Get simplified parameters based on current load
simplified_params = orchestrator.simplify_query_params(original_params)

# Use simplified parameters
result = await rag_system.query(**simplified_params)
```

#### `is_feature_enabled(feature: str) -> bool`

Check if a feature is enabled under current load conditions.

```python
# Check if complex analytics is available
if orchestrator.is_feature_enabled('complex_analytics'):
    # Perform complex analysis
    analysis_result = await perform_complex_analysis(data)
    response['detailed_analysis'] = analysis_result
else:
    # Skip complex analysis due to load constraints
    response['analysis_note'] = 'Detailed analysis temporarily unavailable'
```

## Error Handling

### Exception Types

The graceful degradation system defines custom exception types for different error conditions:

```python
class GracefulDegradationError(Exception):
    """Base exception for graceful degradation system errors."""
    pass

class ComponentFailureError(GracefulDegradationError):
    """Raised when a system component fails."""
    pass

class ConfigurationError(GracefulDegradationError):
    """Raised when configuration is invalid."""
    pass

class IntegrationError(GracefulDegradationError):
    """Raised when external system integration fails."""
    pass

class ThrottlingError(GracefulDegradationError):
    """Raised when request throttling prevents processing."""
    pass

class LoadLevelError(GracefulDegradationError):
    """Raised when load level management fails."""
    pass
```

### Error Handling Patterns

#### Try-Catch with Graceful Fallback

```python
async def robust_query_processing(query: str):
    """Process query with graceful error handling."""
    try:
        # Try to submit through graceful degradation system
        success, message, request_id = await orchestrator.submit_request(
            request_type='user_query',
            priority='high',
            handler=process_metabolomics_query,
            query=query
        )
        
        if success:
            # Wait for result (implementation specific)
            result = await wait_for_request_completion(request_id)
            return result
        else:
            raise ThrottlingError(f"Request throttled: {message}")
            
    except ThrottlingError:
        # Fallback: direct processing with reduced parameters
        simplified_query = simplify_query_for_high_load(query)
        return await direct_process_query(simplified_query)
        
    except ComponentFailureError:
        # Fallback: minimal processing
        return await minimal_query_processing(query)
        
    except Exception as e:
        # Log error and return error response
        logger.error(f"Query processing failed: {e}")
        return {
            'error': 'Query processing temporarily unavailable',
            'retry_after': 30
        }
```

#### Circuit Breaker Pattern

```python
class GracefulDegradationCircuitBreaker:
    """Circuit breaker for graceful degradation API calls."""
    
    def __init__(self, orchestrator, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.orchestrator = orchestrator
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                # Circuit is open - use fallback
                return await self._fallback_processing(*args, **kwargs)
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            # Use fallback processing
            return await self._fallback_processing(*args, **kwargs)
    
    async def _fallback_processing(self, *args, **kwargs):
        """Fallback processing when circuit is open."""
        # Implement fallback logic here
        return {'status': 'fallback', 'message': 'Using fallback processing due to system issues'}

# Usage
circuit_breaker = GracefulDegradationCircuitBreaker(orchestrator)

async def protected_query_processing(query):
    return await circuit_breaker.call_with_circuit_breaker(
        orchestrator.submit_request,
        request_type='user_query',
        handler=process_metabolomics_query,
        query=query
    )
```

## Code Examples

### Complete Integration Example

```python
import asyncio
import logging
from graceful_degradation_integration import (
    GracefulDegradationConfig,
    create_and_start_graceful_degradation_system,
    SystemLoadLevel
)

async def main():
    """Complete example of graceful degradation system integration."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = GracefulDegradationConfig(
        monitoring_interval=5.0,
        base_rate_per_second=50.0,
        max_queue_size=2000,
        max_concurrent_requests=100,
        enable_trend_analysis=True,
        auto_start_monitoring=True
    )
    
    # Initialize existing systems (examples)
    load_balancer = initialize_load_balancer()
    rag_system = initialize_rag_system()
    monitoring_system = initialize_monitoring_system()
    
    try:
        # Create and start graceful degradation system
        orchestrator = await create_and_start_graceful_degradation_system(
            config=config,
            load_balancer=load_balancer,
            rag_system=rag_system,
            monitoring_system=monitoring_system
        )
        
        logger.info("✅ Graceful degradation system started successfully")
        
        # Set up load change monitoring
        def on_load_change(load_level, metrics):
            logger.info(f"Load level changed to: {load_level.name}")
            
            if load_level >= SystemLoadLevel.HIGH:
                # Alert operations team
                send_alert(f"High load detected: {load_level.name}")
                
                # Scale up if possible
                if hasattr(load_balancer, 'scale_up'):
                    load_balancer.scale_up()
        
        orchestrator.add_load_change_callback(on_load_change)
        
        # Set up feature toggle monitoring
        def on_feature_change(feature, enabled):
            logger.info(f"Feature '{feature}' {'enabled' if enabled else 'disabled'}")
            
            # Update application configuration
            app_config.set_feature_enabled(feature, enabled)
        
        orchestrator.feature_controller.add_feature_callback('complex_analytics', on_feature_change)
        orchestrator.feature_controller.add_feature_callback('detailed_logging', on_feature_change)
        
        # Example request processing
        async def process_user_query(query: str, user_id: str) -> Dict[str, Any]:
            """Process user query through graceful degradation system."""
            
            try:
                # Submit request through throttling system
                success, message, request_id = await orchestrator.submit_request(
                    request_type='user_query',
                    priority='high',
                    handler=handle_metabolomics_query,
                    query=query,
                    user_id=user_id
                )
                
                if success:
                    logger.info(f"Query submitted: {request_id}")
                    # In real implementation, you'd wait for the result
                    return {'status': 'submitted', 'request_id': request_id}
                else:
                    logger.warning(f"Query throttled: {message}")
                    return {'status': 'throttled', 'message': message, 'retry_after': 30}
                    
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                return {'status': 'error', 'message': 'Processing temporarily unavailable'}
        
        # Example query handler
        async def handle_metabolomics_query(query: str, user_id: str) -> Dict[str, Any]:
            """Handle metabolomics query with current system constraints."""
            
            # Get current system constraints
            simplified_params = orchestrator.simplify_query_params({
                'query': query,
                'max_tokens': 8000,
                'top_k': 10,
                'response_type': 'Comprehensive'
            })
            
            # Check feature availability
            use_complex_analysis = orchestrator.is_feature_enabled('complex_analytics')
            
            try:
                # Process query with RAG system
                result = await rag_system.process_query(
                    query=simplified_params['query'],
                    max_tokens=simplified_params.get('max_tokens', 4000),
                    top_k=simplified_params.get('top_k', 5),
                    enable_complex_analysis=use_complex_analysis
                )
                
                # Record successful processing
                orchestrator.record_request_metrics(result.get('processing_time', 1000), error=False)
                
                return {
                    'status': 'success',
                    'result': result,
                    'processing_time': result.get('processing_time'),
                    'simplified': len(simplified_params) < 4,
                    'features_used': {
                        'complex_analysis': use_complex_analysis
                    }
                }
                
            except Exception as e:
                # Record failed processing
                orchestrator.record_request_metrics(5000, error=True)
                raise
        
        # Run example queries
        logger.info("Processing example queries...")
        
        queries = [
            "What are the key metabolic pathways involved in diabetes?",
            "Analyze biomarker patterns in cardiovascular disease",
            "Compare metabolite profiles between healthy and diseased samples"
        ]
        
        for i, query in enumerate(queries):
            result = await process_user_query(query, f"user_{i}")
            logger.info(f"Query {i+1} result: {result['status']}")
            
            # Wait between queries
            await asyncio.sleep(2)
        
        # Monitor system for a while
        logger.info("Monitoring system performance...")
        
        for i in range(12):  # Monitor for 1 minute
            await asyncio.sleep(5)
            
            # Get system status
            health = orchestrator.get_health_check()
            status = orchestrator.get_system_status()
            
            logger.info(f"Status check {i+1}/12: "
                       f"Health={health['status']}, "
                       f"Load={health['current_load_level']}, "
                       f"Requests={health['total_requests_processed']}")
            
            # Show throttling metrics if available
            if 'throttling_system' in status:
                throttling = status['throttling_system']['throttling']
                logger.info(f"  Throttling: {throttling.get('success_rate', 0):.1f}% success, "
                           f"{throttling.get('current_rate', 0):.1f} req/s")
        
        # Get final system statistics
        final_status = orchestrator.get_system_status()
        logger.info("Final System Statistics:")
        logger.info(f"  Uptime: {final_status['uptime_seconds']:.1f} seconds")
        logger.info(f"  Total Requests: {final_status['total_requests_processed']}")
        logger.info(f"  Final Load Level: {final_status['current_load_level']}")
        
        # Get metrics history
        metrics_history = orchestrator.get_metrics_history(hours=1)
        logger.info(f"  Metrics History: {len(metrics_history)} data points collected")
        
        if metrics_history:
            avg_cpu = sum(m['cpu_utilization'] for m in metrics_history) / len(metrics_history)
            max_response = max(m.get('response_time_p95', 0) for m in metrics_history)
            logger.info(f"  Average CPU: {avg_cpu:.1f}%")
            logger.info(f"  Max Response Time: {max_response:.0f}ms")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        raise
        
    finally:
        # Cleanup
        try:
            await orchestrator.stop()
            logger.info("✅ Graceful degradation system stopped cleanly")
        except Exception as e:
            logger.error(f"Error stopping system: {e}")

# Helper functions (implementation specific)
def initialize_load_balancer():
    """Initialize load balancer (example)."""
    # Return your load balancer instance
    return None

def initialize_rag_system():
    """Initialize RAG system (example)."""
    # Return your RAG system instance
    return None

def initialize_monitoring_system():
    """Initialize monitoring system (example)."""
    # Return your monitoring system instance
    return None

def send_alert(message: str):
    """Send alert to operations team (example)."""
    print(f"ALERT: {message}")

class AppConfig:
    """Example application configuration."""
    def __init__(self):
        self.features = {}
    
    def set_feature_enabled(self, feature: str, enabled: bool):
        self.features[feature] = enabled
        print(f"App config: {feature} = {enabled}")

app_config = AppConfig()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Request Handler Example

```python
class MetabolomicsRequestHandler:
    """Custom request handler for metabolomics operations."""
    
    def __init__(self, rag_system, database_connection):
        self.rag_system = rag_system
        self.db = database_connection
        self.logger = logging.getLogger(__name__)
    
    async def handle_pathway_analysis(self, pathway_name: str, organism: str = 'human', **kwargs) -> Dict[str, Any]:
        """Handle pathway analysis requests."""
        start_time = time.time()
        
        try:
            # Validate input
            if not pathway_name or len(pathway_name) > 200:
                raise ValueError("Invalid pathway name")
            
            # Check database for cached results
            cache_key = f"pathway:{pathway_name}:{organism}"
            cached_result = await self.db.get_cached_result(cache_key)
            
            if cached_result:
                self.logger.info(f"Returning cached pathway analysis for {pathway_name}")
                return {
                    'status': 'success',
                    'result': cached_result,
                    'processing_time': time.time() - start_time,
                    'cache_hit': True
                }
            
            # Perform pathway analysis through RAG system
            analysis_query = f"Analyze the {pathway_name} metabolic pathway in {organism}"
            
            result = await self.rag_system.process_query(
                query=analysis_query,
                query_type='pathway_analysis',
                organism=organism
            )
            
            # Cache result for future requests
            await self.db.cache_result(cache_key, result, ttl=3600)  # 1 hour TTL
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'result': result,
                'processing_time': processing_time,
                'cache_hit': False,
                'pathway_name': pathway_name,
                'organism': organism
            }
            
        except Exception as e:
            self.logger.error(f"Pathway analysis failed for {pathway_name}: {e}")
            
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'pathway_name': pathway_name
            }
    
    async def handle_biomarker_search(self, search_terms: List[str], filter_criteria: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Handle biomarker search requests."""
        start_time = time.time()
        
        try:
            # Validate input
            if not search_terms or len(search_terms) > 50:
                raise ValueError("Invalid search terms")
            
            # Build search query
            search_query = f"Find biomarkers related to: {', '.join(search_terms)}"
            
            if filter_criteria:
                if 'disease' in filter_criteria:
                    search_query += f" in {filter_criteria['disease']}"
                if 'sample_type' in filter_criteria:
                    search_query += f" from {filter_criteria['sample_type']} samples"
            
            # Perform biomarker search
            result = await self.rag_system.process_query(
                query=search_query,
                query_type='biomarker_search',
                max_results=filter_criteria.get('max_results', 20) if filter_criteria else 20
            )
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'result': result,
                'processing_time': processing_time,
                'search_terms': search_terms,
                'filters_applied': filter_criteria or {}
            }
            
        except Exception as e:
            self.logger.error(f"Biomarker search failed for {search_terms}: {e}")
            
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'search_terms': search_terms
            }

# Usage with graceful degradation system
async def setup_metabolomics_handlers(orchestrator):
    """Set up metabolomics request handlers with graceful degradation."""
    
    # Initialize handler
    handler = MetabolomicsRequestHandler(rag_system, database)
    
    # Submit pathway analysis request
    success, message, request_id = await orchestrator.submit_request(
        request_type='user_query',
        priority='high',
        handler=handler.handle_pathway_analysis,
        pathway_name='glycolysis',
        organism='human'
    )
    
    if success:
        print(f"Pathway analysis request submitted: {request_id}")
    
    # Submit biomarker search request
    success, message, request_id = await orchestrator.submit_request(
        request_type='analytics',
        priority='medium', 
        handler=handler.handle_biomarker_search,
        search_terms=['glucose', 'insulin', 'diabetes'],
        filter_criteria={'disease': 'type 2 diabetes', 'max_results': 15}
    )
    
    if success:
        print(f"Biomarker search request submitted: {request_id}")
```

This API documentation provides comprehensive coverage of all programmatic interfaces available in the graceful degradation system, enabling developers to effectively integrate and use the system in their applications.