# Integration Wrapper Guide

A comprehensive guide to using the enhanced LightRAG/Perplexity Integration Wrapper for the Clinical Metabolomics Oracle.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage Patterns](#usage-patterns)
7. [Advanced Features](#advanced-features)
8. [Performance Monitoring](#performance-monitoring)
9. [Error Handling](#error-handling)
10. [Backward Compatibility](#backward-compatibility)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## Overview

The Integration Wrapper provides a unified interface for routing queries between LightRAG and Perplexity APIs based on feature flags, performance metrics, and quality assessments. It maintains backward compatibility while adding advanced capabilities like A/B testing, circuit breaker protection, and comprehensive monitoring.

### Key Components

- **IntegratedQueryService**: Main service class that handles routing and fallback
- **FeatureFlagManager**: Controls routing decisions based on configuration
- **AdvancedCircuitBreaker**: Provides fault tolerance with automatic recovery
- **ServiceHealthMonitor**: Monitors service availability and health
- **ServiceResponse**: Unified response format for all services
- **QueryRequest**: Standardized request format

## Key Features

### 🚀 Core Functionality
- **Transparent Routing**: Seamless switching between LightRAG and Perplexity
- **Automatic Fallback**: Falls back to Perplexity if LightRAG fails
- **Feature Flags**: Configurable routing based on rollout percentages
- **Response Caching**: Intelligent caching to improve performance

### 🔧 Advanced Features
- **A/B Testing**: Compare performance between services with user cohort assignment
- **Circuit Breaker**: Automatic service protection with recovery mechanisms  
- **Health Monitoring**: Continuous service health checks and reporting
- **Quality Assessment**: Pluggable quality scoring for responses
- **Performance Tracking**: Comprehensive metrics collection and analysis

### 🛡️ Reliability Features
- **Error Recovery**: Robust error handling with automatic retry logic
- **Timeout Protection**: Configurable timeouts prevent hanging requests
- **Graceful Degradation**: Maintains service availability during failures
- **Thread Safety**: Safe for concurrent usage in async environments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IntegratedQueryService                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │ FeatureFlagMgr  │ │ CircuitBreaker   │ │ HealthMonitor   │ │
│  │ - Routing Logic │ │ - Fault Tolerance│ │ - Service Health│ │
│  │ - A/B Testing   │ │ - Auto Recovery  │ │ - Availability  │ │
│  └─────────────────┘ └──────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐           ┌─────────────────────────────┐ │
│  │  LightRAGSvc    │    or     │    PerplexityService        │ │
│  │  - Graph RAG    │           │    - Web Search RAG        │ │
│  │  - Local KB     │           │    - Real-time Data        │ │
│  └─────────────────┘           └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Setup

```python
import asyncio
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.integration_wrapper import create_integrated_service, QueryRequest

async def basic_example():
    # Configure LightRAG
    config = LightRAGConfig.get_config()
    config.lightrag_integration_enabled = True
    config.lightrag_rollout_percentage = 50.0  # 50% of users get LightRAG
    
    # Create service
    service = create_integrated_service(
        config=config,
        perplexity_api_key="your-perplexity-api-key"
    )
    
    # Create query
    query = QueryRequest(
        query_text="What are the key biomarkers for diabetes?",
        user_id="user123"
    )
    
    # Execute query
    response = await service.query_async(query)
    
    print(f"Success: {response.is_success}")
    print(f"Content: {response.content}")
    print(f"Service used: {response.response_type.value}")
    
    # Cleanup
    await service.shutdown()

# Run example
asyncio.run(basic_example())
```

### Production Setup with Context Manager

```python
from lightrag_integration.integration_wrapper import managed_query_service

async def production_example():
    # Use context manager for automatic lifecycle management
    async with managed_query_service(config, perplexity_api_key) as service:
        # Service is automatically initialized and shutdown
        
        query = QueryRequest(
            query_text="Compare metabolomics vs genomics in precision medicine",
            user_id="prod_user",
            session_id="session_456"
        )
        
        response = await service.query_async(query)
        
        # Process response
        if response.is_success:
            print("Query successful!")
            print(f"Processing time: {response.processing_time:.2f}s")
            if response.quality_scores:
                print(f"Quality score: {response.average_quality_score:.2f}")
        else:
            print(f"Query failed: {response.error_details}")
```

## Configuration

### Environment Variables

Set these environment variables for full functionality:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"
export PERPLEXITY_API_KEY="your-perplexity-api-key"

# LightRAG Feature Flags
export LIGHTRAG_INTEGRATION_ENABLED="true"
export LIGHTRAG_ROLLOUT_PERCENTAGE="50.0"
export LIGHTRAG_ENABLE_AB_TESTING="true"
export LIGHTRAG_FALLBACK_TO_PERPLEXITY="true"

# Circuit Breaker Settings
export LIGHTRAG_ENABLE_CIRCUIT_BREAKER="true"
export LIGHTRAG_CIRCUIT_BREAKER_FAILURE_THRESHOLD="3"
export LIGHTRAG_CIRCUIT_BREAKER_RECOVERY_TIMEOUT="300"

# Quality and Performance
export LIGHTRAG_ENABLE_QUALITY_METRICS="true"
export LIGHTRAG_MIN_QUALITY_THRESHOLD="0.7"
export LIGHTRAG_INTEGRATION_TIMEOUT_SECONDS="30.0"
```

### Programmatic Configuration

```python
config = LightRAGConfig.get_config()

# Basic routing
config.lightrag_integration_enabled = True
config.lightrag_rollout_percentage = 25.0  # Start with 25%
config.lightrag_fallback_to_perplexity = True

# A/B Testing
config.lightrag_enable_ab_testing = True
config.lightrag_enable_performance_comparison = True

# Circuit Breaker
config.lightrag_enable_circuit_breaker = True
config.lightrag_circuit_breaker_failure_threshold = 3
config.lightrag_circuit_breaker_recovery_timeout = 300.0

# Quality Control
config.lightrag_enable_quality_metrics = True
config.lightrag_min_quality_threshold = 0.7

# Override for testing
config.lightrag_force_user_cohort = "lightrag"  # Force LightRAG for testing
```

## Usage Patterns

### 1. Factory Functions

#### Basic Integration
```python
from lightrag_integration.integration_wrapper import create_integrated_service

service = create_integrated_service(
    config=config,
    perplexity_api_key=api_key,
    logger=your_logger
)
```

#### Production Service
```python
from lightrag_integration.integration_wrapper import create_production_service

def custom_quality_assessor(response):
    # Your quality assessment logic here
    return {QualityMetric.RELEVANCE: 0.8}

service = create_production_service(
    config=config,
    perplexity_api_key=api_key,
    quality_assessor=custom_quality_assessor,
    logger=your_logger
)
```

#### Backward Compatibility
```python
from lightrag_integration.integration_wrapper import create_service_with_fallback

service = create_service_with_fallback(
    lightrag_config=config,
    perplexity_api_key=api_key,
    enable_ab_testing=False,  # Maintain legacy behavior
    logger=your_logger
)
```

### 2. Query Execution

#### Simple Query
```python
query = QueryRequest(
    query_text="Your question here",
    user_id="user123"
)

response = await service.query_async(query)
```

#### Advanced Query with Metadata
```python
query = QueryRequest(
    query_text="Complex biomedical question",
    user_id="user456",
    session_id="session789", 
    query_type="research_query",
    timeout_seconds=45.0,
    context_metadata={
        "priority": "high",
        "domain": "metabolomics"
    },
    quality_requirements={
        QualityMetric.RELEVANCE: 0.8,
        QualityMetric.ACCURACY: 0.9
    }
)

response = await service.query_async(query)
```

### 3. Response Processing

```python
# Check response status
if response.is_success:
    print("Query successful!")
    print(f"Content: {response.content}")
    
    # Check which service was used
    if response.response_type == ResponseType.LIGHTRAG:
        print("Served by LightRAG")
    elif response.response_type == ResponseType.PERPLEXITY:
        print("Served by Perplexity")
    elif response.response_type == ResponseType.CACHED:
        print("Served from cache")
    
    # Access citations if available
    if response.citations:
        print(f"Found {len(response.citations)} citations")
    
    # Check quality scores
    if response.quality_scores:
        print(f"Average quality: {response.average_quality_score:.2f}")
        for metric, score in response.quality_scores.items():
            print(f"  {metric.value}: {score:.2f}")

else:
    print(f"Query failed: {response.error_details}")
    print(f"Error occurred with: {response.response_type.value}")
```

## Advanced Features

### A/B Testing and Performance Comparison

```python
# Enable A/B testing in configuration
config.lightrag_enable_ab_testing = True
config.lightrag_rollout_percentage = 50.0

# Create service with A/B testing
service = create_service_with_fallback(
    lightrag_config=config,
    perplexity_api_key=api_key,
    enable_ab_testing=True
)

# Execute queries with different users
users = ["user1", "user2", "user3", "user4", "user5"]
for user_id in users:
    query = QueryRequest(query_text="Test query", user_id=user_id)
    response = await service.query_async(query)
    
    # Users are consistently assigned to cohorts based on their ID
    cohort = response.metadata.get('user_cohort')
    print(f"User {user_id} assigned to: {cohort}")

# Get A/B test results
ab_metrics = service.get_ab_test_metrics()
print("A/B Test Results:")
for service_name, metrics in ab_metrics.items():
    print(f"{service_name}:")
    print(f"  Sample size: {metrics['sample_size']}")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Avg response time: {metrics['avg_response_time']:.2f}s")
    print(f"  Avg quality score: {metrics['avg_quality_score']:.2f}")
```

### Circuit Breaker Protection

```python
# The circuit breaker automatically protects against failing services
# It opens after a configured number of failures and attempts recovery

# Check circuit breaker status
summary = service.get_performance_summary()
for service_name, service_info in summary['services'].items():
    cb_state = service_info.get('circuit_breaker')
    if cb_state:
        print(f"{service_name} Circuit Breaker:")
        print(f"  Open: {cb_state['is_open']}")
        print(f"  Failures: {cb_state['failure_count']}")
        print(f"  Recovery attempts: {cb_state['recovery_attempts']}")

# Reset circuit breaker manually if needed
service.feature_manager.reset_circuit_breaker()
```

### Health Monitoring

```python
# Health monitoring runs automatically in the background
# Check service health status
health_status = service.health_monitor.get_all_health_status()

for service_name, health_info in health_status.items():
    print(f"{service_name} Health:")
    print(f"  Healthy: {health_info['is_healthy']}")
    print(f"  Last check: {health_info['last_check']}")
    print(f"  Consecutive failures: {health_info['consecutive_failures']}")
    print(f"  Success rate: {health_info['successful_checks']}/{health_info['total_checks']}")
```

### Custom Quality Assessment

```python
def advanced_quality_assessor(response: ServiceResponse) -> Dict[QualityMetric, float]:
    """Advanced quality assessment function."""
    scores = {}
    
    # Assess relevance based on domain-specific keywords
    metabolomics_keywords = ['biomarker', 'metabolite', 'pathway', 'metabolism']
    keyword_score = sum(1 for kw in metabolomics_keywords if kw in response.content.lower())
    scores[QualityMetric.RELEVANCE] = min(1.0, keyword_score / len(metabolomics_keywords))
    
    # Assess completeness based on response structure
    has_intro = any(word in response.content.lower() for word in ['overview', 'introduction', 'summary'])
    has_details = len(response.content.split('.')) > 3  # Multiple sentences
    has_conclusion = any(word in response.content.lower() for word in ['conclusion', 'summary', 'therefore'])
    
    completeness = (has_intro + has_details + has_conclusion) / 3
    scores[QualityMetric.COMPLETENESS] = completeness
    
    # Assess citation quality
    if response.citations:
        # Check for peer-reviewed sources
        peer_reviewed = sum(1 for cit in response.citations 
                          if any(domain in cit.get('url', '') for domain in ['pubmed', 'doi.org', 'nature', 'science']))
        citation_score = min(1.0, peer_reviewed / len(response.citations))
        scores[QualityMetric.CITATION_QUALITY] = citation_score
    else:
        scores[QualityMetric.CITATION_QUALITY] = 0.5  # Neutral score for no citations
    
    # Response time quality (faster is better, up to a point)
    optimal_time = 3.0  # 3 seconds is optimal
    if response.processing_time <= optimal_time:
        scores[QualityMetric.RESPONSE_TIME] = 1.0
    else:
        # Degrade score for slower responses
        score = max(0.1, 1.0 - (response.processing_time - optimal_time) / 10.0)
        scores[QualityMetric.RESPONSE_TIME] = score
    
    return scores

# Set custom quality assessor
service.set_quality_assessor(advanced_quality_assessor)
```

## Performance Monitoring

### Comprehensive Performance Summary

```python
# Get detailed performance summary
summary = service.get_performance_summary()

print("=== Performance Summary ===")

# Circuit breaker status
print("\nCircuit Breaker Status:")
cb_info = summary['circuit_breaker']
print(f"  Total requests: {cb_info['total_requests']}")
print(f"  Success rate: {cb_info['success_rate']:.2%}")
print(f"  Current failures: {cb_info['failure_count']}")

# Service performance
print("\nService Performance:")
perf_info = summary['performance']
for service, metrics in perf_info.items():
    if service == 'last_updated':
        continue
    print(f"  {service.upper()}:")
    print(f"    Success count: {metrics['success_count']}")
    print(f"    Error count: {metrics['error_count']}")
    print(f"    Avg response time: {metrics['avg_response_time']:.2f}s")
    print(f"    Avg quality: {metrics['avg_quality_score']:.2f}")

# Recent performance trends
recent_perf = summary.get('recent_performance', {})
print(f"\nRecent Performance (last 20 requests):")
print(f"  Total requests: {recent_perf.get('total_requests', 0)}")
print(f"  Success rate: {recent_perf.get('success_rate', 0):.2%}")
print(f"  Avg response time: {recent_perf.get('avg_response_time', 0):.2f}s")

# Routing distribution
routing_dist = recent_perf.get('routing_distribution', {})
print(f"  Routing distribution: {routing_dist}")

# Configuration status
config_info = summary['configuration']
print(f"\nConfiguration:")
print(f"  Integration enabled: {config_info['integration_enabled']}")
print(f"  Rollout percentage: {config_info['rollout_percentage']}%")
print(f"  A/B testing: {config_info['ab_testing_enabled']}")
print(f"  Circuit breaker: {config_info['circuit_breaker_enabled']}")
```

### Real-time Monitoring Integration

```python
import json
from datetime import datetime

async def monitoring_loop(service, interval_seconds=60):
    """Example monitoring loop for production deployment."""
    
    while True:
        try:
            summary = service.get_performance_summary()
            
            # Create monitoring record
            monitoring_data = {
                'timestamp': datetime.now().isoformat(),
                'service_health': {
                    name: info['is_healthy'] 
                    for name, info in summary.get('health_monitoring', {}).items()
                },
                'performance_metrics': {
                    'success_rate': summary['circuit_breaker']['success_rate'],
                    'total_requests': summary['circuit_breaker']['total_requests'],
                    'avg_response_time': {
                        service: metrics['avg_response_time']
                        for service, metrics in summary['performance'].items()
                        if isinstance(metrics, dict)
                    }
                },
                'circuit_breaker_status': {
                    service: info.get('circuit_breaker', {}).get('is_open', False)
                    for service, info in summary['services'].items()
                }
            }
            
            # Log to monitoring system (replace with your monitoring solution)
            print(f"MONITORING: {json.dumps(monitoring_data, indent=2)}")
            
            # Check for alerts
            if summary['circuit_breaker']['success_rate'] < 0.9:
                print("ALERT: Success rate below 90%!")
            
            for service_name, health_info in summary.get('health_monitoring', {}).items():
                if not health_info.get('is_healthy', True):
                    print(f"ALERT: {service_name} service unhealthy!")
            
            await asyncio.sleep(interval_seconds)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            await asyncio.sleep(interval_seconds)

# Start monitoring in background
asyncio.create_task(monitoring_loop(service))
```

## Error Handling

### Response Error Handling

```python
async def robust_query_handling(service, query_text, user_id):
    """Example of robust query handling with comprehensive error handling."""
    
    query = QueryRequest(
        query_text=query_text,
        user_id=user_id,
        timeout_seconds=30.0
    )
    
    try:
        response = await service.query_async(query)
        
        if response.is_success:
            return {
                'success': True,
                'content': response.content,
                'service_used': response.response_type.value,
                'processing_time': response.processing_time,
                'quality_score': response.average_quality_score
            }
        else:
            # Handle different types of failures
            if response.response_type == ResponseType.CIRCUIT_BREAKER_BLOCKED:
                return {
                    'success': False,
                    'error': 'service_unavailable',
                    'message': 'Service is temporarily unavailable due to recent failures',
                    'retry_suggested': True,
                    'retry_after_seconds': 300
                }
            else:
                return {
                    'success': False,
                    'error': 'query_failed',
                    'message': response.error_details or 'Unknown error occurred',
                    'retry_suggested': True,
                    'retry_after_seconds': 60
                }
                
    except asyncio.TimeoutError:
        return {
            'success': False,
            'error': 'timeout',
            'message': f'Query timed out after {query.timeout_seconds} seconds',
            'retry_suggested': True,
            'retry_after_seconds': 30
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': 'unexpected_error',
            'message': str(e),
            'retry_suggested': False
        }
```

### Service Health Checks

```python
async def service_health_check(service):
    """Check if services are healthy before processing requests."""
    
    try:
        # Get health status
        health_status = service.health_monitor.get_all_health_status()
        
        healthy_services = []
        unhealthy_services = []
        
        for service_name, health_info in health_status.items():
            if health_info.get('is_healthy', False):
                healthy_services.append(service_name)
            else:
                unhealthy_services.append({
                    'name': service_name,
                    'consecutive_failures': health_info.get('consecutive_failures', 0),
                    'last_check': health_info.get('last_check')
                })
        
        return {
            'overall_healthy': len(unhealthy_services) == 0,
            'healthy_services': healthy_services,
            'unhealthy_services': unhealthy_services,
            'can_process_requests': len(healthy_services) > 0
        }
        
    except Exception as e:
        return {
            'overall_healthy': False,
            'error': str(e),
            'can_process_requests': False
        }
```

## Backward Compatibility

The integration wrapper maintains backward compatibility with existing systems:

### Legacy Integration Pattern

```python
# Old pattern (still supported)
def legacy_query_function(query_text, user_context=None):
    """Legacy function that can be easily migrated."""
    
    # Create service using backward compatibility factory
    service = create_service_with_fallback(
        lightrag_config=config,
        perplexity_api_key=api_key,
        enable_ab_testing=False  # Disable advanced features
    )
    
    # Convert to new query format
    query = QueryRequest(
        query_text=query_text,
        user_id=user_context.get('user_id') if user_context else None,
        session_id=user_context.get('session_id') if user_context else None
    )
    
    # Execute and convert response
    response = await service.query_async(query)
    
    # Return in legacy format
    return {
        'content': response.content,
        'success': response.is_success,
        'error': response.error_details,
        'metadata': {
            'processing_time': response.processing_time,
            'service': response.response_type.value
        }
    }
```

### Migration Path

1. **Phase 1**: Replace existing service calls with `create_service_with_fallback()`
2. **Phase 2**: Enable A/B testing with small rollout percentage
3. **Phase 3**: Add quality assessment and monitoring
4. **Phase 4**: Enable full feature set with circuit breaker protection

## Best Practices

### 1. Configuration Management

```python
# Use environment-specific configurations
def get_production_config():
    config = LightRAGConfig.get_config()
    config.lightrag_integration_enabled = True
    config.lightrag_rollout_percentage = 10.0  # Start conservative
    config.lightrag_enable_circuit_breaker = True
    config.lightrag_fallback_to_perplexity = True
    return config

def get_development_config():
    config = LightRAGConfig.get_config()
    config.lightrag_integration_enabled = True
    config.lightrag_rollout_percentage = 100.0  # Test everything
    config.lightrag_force_user_cohort = "lightrag"  # Force for testing
    return config
```

### 2. Gradual Rollout Strategy

```python
# Week 1: 5% rollout
config.lightrag_rollout_percentage = 5.0

# Week 2: Monitor metrics, increase to 15%
if avg_success_rate > 0.95 and avg_quality_score > 0.8:
    config.lightrag_rollout_percentage = 15.0

# Week 3: Continue gradual increase
config.lightrag_rollout_percentage = 30.0

# Week 4: Full rollout if metrics are good
config.lightrag_rollout_percentage = 100.0
```

### 3. Quality Monitoring

```python
# Implement quality thresholds
config.lightrag_enable_quality_metrics = True
config.lightrag_min_quality_threshold = 0.7

# Set up quality assessor with domain knowledge
def clinical_quality_assessor(response):
    # Implement clinical metabolomics specific quality checks
    return quality_scores

service.set_quality_assessor(clinical_quality_assessor)
```

### 4. Error Recovery

```python
# Implement exponential backoff for retries
async def query_with_retry(service, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await service.query_async(query)
            if response.is_success:
                return response
            
            # Wait before retry (exponential backoff)
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    # Return error response if all retries failed
    return ServiceResponse(
        content="Service temporarily unavailable",
        error_details="Max retries exceeded",
        response_type=ResponseType.FALLBACK
    )
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Circuit Breaker Opening Frequently

**Problem**: Circuit breaker opens after few requests
**Solutions**:
- Check service health status
- Increase failure threshold temporarily
- Verify API keys and connectivity
- Check timeout settings

```python
# Debug circuit breaker
summary = service.get_performance_summary()
cb_info = summary['circuit_breaker']
print(f"Failure rate: {cb_info['failure_rate']:.2%}")
print(f"Consecutive failures: {cb_info['failure_count']}")

# Temporarily increase threshold
service.lightrag_circuit_breaker.failure_threshold = 5
```

#### 2. Low Quality Scores

**Problem**: Quality scores consistently low
**Solutions**:
- Review quality assessor logic
- Check if responses contain expected content
- Adjust quality thresholds
- Verify domain-specific keywords

```python
# Debug quality assessment
response = await service.query_async(query)
if response.quality_scores:
    for metric, score in response.quality_scores.items():
        print(f"{metric.value}: {score} (threshold: {config.lightrag_min_quality_threshold})")
```

#### 3. A/B Testing Uneven Distribution

**Problem**: Users not evenly distributed between cohorts
**Solutions**:
- Check user hash salt configuration
- Verify rollout percentage settings
- Review user ID consistency

```python
# Debug user assignment
test_users = ["user1", "user2", "user3", "user4", "user5"]
for user in test_users:
    context = RoutingContext(user_id=user)
    result = service.feature_manager.should_use_lightrag(context)
    print(f"{user}: {result.decision.value} (cohort: {result.user_cohort})")
```

#### 4. Performance Issues

**Problem**: High response times or timeouts
**Solutions**:
- Check network connectivity
- Increase timeout values
- Enable response caching
- Monitor service health

```python
# Performance analysis
summary = service.get_performance_summary()
for service_name, metrics in summary['performance'].items():
    if isinstance(metrics, dict):
        print(f"{service_name} avg response time: {metrics['avg_response_time']:.2f}s")
```

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("lightrag_integration")
logger.setLevel(logging.DEBUG)

# Create service with debug logger
service = create_integrated_service(
    config=config,
    perplexity_api_key=api_key,
    logger=logger
)
```

### Health Check Endpoints

For production deployments, implement health check endpoints:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    try:
        health_status = service.health_monitor.get_all_health_status()
        
        all_healthy = all(
            info.get('is_healthy', False) 
            for info in health_status.values()
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring systems."""
    return service.get_performance_summary()
```

## Conclusion

The Integration Wrapper provides a robust, production-ready solution for routing queries between LightRAG and Perplexity while maintaining backward compatibility and adding advanced features like A/B testing, circuit breaker protection, and comprehensive monitoring.

For additional support or questions, refer to the example code in `examples/integration_wrapper_examples.py` or review the inline documentation in the source code.