# API Usage Metrics Logging System Guide

## Overview

The API Usage Metrics Logging System provides comprehensive monitoring and analytics for API calls made by the Clinical Metabolomics Oracle LightRAG integration. This system seamlessly integrates with the existing enhanced cost tracking infrastructure to provide detailed insights into API usage patterns, performance metrics, and research-specific analytics.

## Key Features

### 1. Comprehensive API Metrics Tracking
- **Token Usage**: Detailed tracking of prompt, completion, and embedding tokens
- **Cost Analysis**: Real-time cost calculation and budget utilization tracking
- **Performance Metrics**: Response times, throughput, and error rates
- **Research Categorization**: Automatic categorization of API calls by metabolomics research area

### 2. Integration with Enhanced Cost Tracking
- **Seamless Integration**: Works with existing `CostPersistence`, `BudgetManager`, and `AuditTrail` systems
- **Real-time Budget Monitoring**: Tracks budget utilization as API calls are made
- **Historical Analysis**: Stores metrics for long-term trend analysis

### 3. Structured Logging
- **JSON Format**: Structured logging for easy parsing and analysis
- **Multiple Log Files**: Separate files for metrics, audit trails, and general logging
- **Log Rotation**: Automatic log rotation with configurable retention policies

### 4. Thread-Safe Operations
- **Concurrent Logging**: Safe for use with multiple concurrent API calls
- **Performance Optimized**: Minimal overhead on API call performance

## Quick Start

### Basic Usage

```python
from lightrag_integration import (
    ClinicalMetabolomicsRAG, 
    LightRAGConfig,
    APIUsageMetricsLogger,
    ResearchCategory
)

# Initialize with API metrics logging enabled
config = LightRAGConfig.get_config(source={
    'api_key': 'your-openai-key',
    'enable_api_metrics_logging': True,
    'enable_file_logging': True,
    'log_dir': 'logs'
})

rag = ClinicalMetabolomicsRAG(config)
await rag.initialize_rag()

# Make queries - metrics are logged automatically
result = await rag.query(
    "What metabolites are involved in glucose metabolism?",
    mode="hybrid"
)

# Get comprehensive metrics summary
metrics_summary = rag.get_api_metrics_summary()
print(f"API Metrics: {metrics_summary}")
```

### Advanced Configuration

```python
# Environment variables for configuration
import os
os.environ['LIGHTRAG_ENABLE_API_METRICS_LOGGING'] = 'true'
os.environ['LIGHTRAG_ENABLE_FILE_LOGGING'] = 'true'
os.environ['LIGHTRAG_LOG_DIR'] = 'logs'
os.environ['LIGHTRAG_LOG_MAX_BYTES'] = '10485760'  # 10MB
os.environ['LIGHTRAG_LOG_BACKUP_COUNT'] = '5'

# Create enhanced system with full metrics logging
rag = create_enhanced_rag_system(
    daily_budget_limit=50.0,
    monthly_budget_limit=1000.0,
    enable_api_metrics_logging=True
)

await rag.initialize_rag()
```

## API Metrics Data Structure

### APIMetric Class

The core data structure for individual API metrics:

```python
APIMetric(
    # Core identification
    id="uuid-string",
    timestamp=1691234567.890,
    session_id="session-uuid",
    metric_type=MetricType.LLM_CALL,
    
    # API operation details
    operation_name="llm_completion",
    model_name="gpt-4o-mini",
    api_provider="openai",
    
    # Token and cost metrics
    prompt_tokens=150,
    completion_tokens=75,
    total_tokens=225,
    cost_usd=0.012,
    
    # Performance metrics
    response_time_ms=1500.5,
    throughput_tokens_per_sec=150.0,
    
    # Quality metrics
    success=True,
    error_type=None,
    retry_count=0,
    
    # Research categorization
    research_category="metabolite_identification",
    query_type="biomarker_discovery",
    subject_area="glucose_metabolism",
    
    # System resources
    memory_usage_mb=125.6,
    concurrent_operations=3,
    
    # Budget tracking
    daily_budget_used_percent=15.5,
    monthly_budget_used_percent=8.2,
    
    # Additional metadata
    metadata={
        'temperature': 0.1,
        'max_tokens': 500,
        'biomedical_optimized': True
    }
)
```

## Log Files Structure

### 1. API Metrics Log (`api_metrics.log`)

Structured JSON logging of all API metrics:

```json
{
    "timestamp": "2025-08-06T10:30:45.123Z",
    "logger": "lightrag_integration.metrics",
    "level": "INFO",
    "message": {
        "id": "uuid-string",
        "operation_name": "llm_completion",
        "model_name": "gpt-4o-mini",
        "total_tokens": 225,
        "cost_usd": 0.012,
        "response_time_ms": 1500.5,
        "success": true,
        "research_category": "metabolite_identification",
        "session_id": "session-uuid",
        "metric_type": "llm_call"
    }
}
```

### 2. API Audit Log (`api_audit.log`)

Compliance-focused audit trail:

```
2025-08-06T10:30:45.123 - AUDIT - {"event_type": "api_usage", "timestamp": 1691234567.890, "session_id": "session-uuid", "operation": "llm_completion", "model": "gpt-4o-mini", "tokens": 225, "cost_usd": 0.012, "success": true, "research_category": "metabolite_identification", "compliance_level": "standard"}
```

### 3. General Integration Log (`lightrag_integration.log`)

High-level system events and summaries:

```
2025-08-06 10:30:45,123 - lightrag_integration.clinical_metabolomics_rag - INFO - API Call: llm_completion | Model: gpt-4o-mini | Tokens: 225 | Cost: $0.012000 | Time: 1500.5ms | Success: True | Category: metabolite_identification
```

## Performance Monitoring

### Real-time Performance Metrics

```python
# Get current performance summary
summary = rag.get_api_metrics_summary()

print("Current Hour Metrics:")
print(f"  Total Calls: {summary['performance_summary']['current_hour']['total_calls']}")
print(f"  Total Cost: ${summary['performance_summary']['current_hour']['total_cost']}")
print(f"  Average Response Time: {summary['performance_summary']['current_hour']['avg_response_time_ms']}ms")
print(f"  Error Rate: {summary['performance_summary']['current_hour']['error_rate_percent']}%")

print("Current Day Metrics:")
print(f"  Total Calls: {summary['performance_summary']['current_day']['total_calls']}")
print(f"  Total Cost: ${summary['performance_summary']['current_day']['total_cost']}")
print(f"  Total Tokens: {summary['performance_summary']['current_day']['total_tokens']}")

print("Top Research Categories:")
for category, count in summary['performance_summary']['top_research_categories'].items():
    print(f"  {category}: {count} calls")

print("Recent Errors:")
for error_type, count in summary['performance_summary']['top_error_types'].items():
    print(f"  {error_type}: {count} occurrences")
```

### Batch Operation Logging

```python
# Log metrics for batch operations like PDF processing
rag.log_batch_api_operation(
    operation_name="pdf_processing_batch",
    batch_size=50,
    total_tokens=125000,
    total_cost=12.50,
    processing_time_seconds=120.5,
    success_count=48,
    error_count=2,
    research_category=ResearchCategory.DOCUMENT_PROCESSING.value,
    metadata={
        'pdf_files_processed': 50,
        'average_pages_per_pdf': 8.5,
        'extraction_method': 'enhanced_biomedical'
    }
)
```

### System Event Logging

```python
# Log important system events
rag.log_system_event(
    event_type='rag_initialization',
    event_data={
        'initialization_time_ms': 2500,
        'components_loaded': ['llm', 'embedding', 'cost_tracking', 'metrics'],
        'memory_usage_mb': 150.5,
        'config_source': 'environment'
    },
    user_id='system'
)
```

## Research Category Integration

### Automatic Categorization

The system automatically categorizes API calls based on research content:

```python
# Research categories are automatically applied based on query content
result = await rag.query(
    "Identify biomarkers for type 2 diabetes using metabolomics data",
    mode="hybrid"
)
# Automatically categorized as: biomarker_discovery

result = await rag.query(
    "What are the metabolic pathways involved in glucose metabolism?",
    mode="local"
)
# Automatically categorized as: pathway_analysis
```

### Manual Category Override

```python
# Manually specify research category for specific operations
with rag.api_metrics_logger.track_api_call(
    operation_name="custom_analysis",
    model_name="gpt-4o",
    research_category=ResearchCategory.DRUG_DISCOVERY.value,
    metadata={'analysis_type': 'compound_screening'}
) as tracker:
    # Your custom API operations
    tracker.set_tokens(prompt=200, completion=100)
    tracker.set_cost(0.025)
```

## Budget Integration

### Real-time Budget Tracking

```python
# Set budget limits
rag.set_budget_limits(daily_limit=25.0, monthly_limit=500.0)

# Budget utilization is automatically tracked in metrics
summary = rag.get_api_metrics_summary()
system_info = summary['performance_summary']['system']
print(f"Daily Budget Used: {system_info.get('daily_budget_used_percent', 0)}%")
print(f"Monthly Budget Used: {system_info.get('monthly_budget_used_percent', 0)}%")

# Get detailed budget status
enhanced_summary = rag.get_enhanced_cost_summary()
if 'budget_status' in enhanced_summary:
    budget_info = enhanced_summary['budget_status']
    print(f"Daily: ${budget_info['daily']['current_cost']:.2f} / ${budget_info['daily']['budget_limit']:.2f}")
    print(f"Monthly: ${budget_info['monthly']['current_cost']:.2f} / ${budget_info['monthly']['budget_limit']:.2f}")
```

## Configuration Options

### Environment Variables

```bash
# API Metrics Logging Configuration
export LIGHTRAG_ENABLE_API_METRICS_LOGGING=true

# Logging Configuration
export LIGHTRAG_ENABLE_FILE_LOGGING=true
export LIGHTRAG_LOG_DIR=logs
export LIGHTRAG_LOG_MAX_BYTES=10485760  # 10MB
export LIGHTRAG_LOG_BACKUP_COUNT=5
export LIGHTRAG_LOG_LEVEL=INFO

# Enhanced Cost Tracking Integration
export LIGHTRAG_ENABLE_COST_TRACKING=true
export LIGHTRAG_COST_PERSISTENCE_ENABLED=true
export LIGHTRAG_ENABLE_RESEARCH_CATEGORIZATION=true
export LIGHTRAG_ENABLE_AUDIT_TRAIL=true

# Budget Management
export LIGHTRAG_DAILY_BUDGET_LIMIT=50.0
export LIGHTRAG_MONTHLY_BUDGET_LIMIT=1000.0
export LIGHTRAG_COST_ALERT_THRESHOLD=80.0
```

### Configuration Dictionary

```python
config_dict = {
    'api_key': 'your-openai-key',
    
    # API Metrics Logging
    'enable_api_metrics_logging': True,
    
    # Logging Configuration
    'enable_file_logging': True,
    'log_dir': 'logs',
    'log_max_bytes': 10 * 1024 * 1024,  # 10MB
    'log_backup_count': 5,
    'log_level': 'INFO',
    
    # Cost Tracking Integration
    'enable_cost_tracking': True,
    'cost_persistence_enabled': True,
    'enable_research_categorization': True,
    'enable_audit_trail': True,
    
    # Budget Management
    'daily_budget_limit': 50.0,
    'monthly_budget_limit': 1000.0,
    'cost_alert_threshold_percentage': 80.0,
    
    # Database Configuration
    'cost_db_path': 'cost_tracking.db',
    'max_cost_retention_days': 365
}

config = LightRAGConfig.get_config(source=config_dict)
```

## Analytics and Reporting

### Historical Cost Analysis

```python
# Generate comprehensive cost report with metrics integration
cost_report = rag.generate_cost_report(
    days=30,
    include_audit_data=True
)

print("Cost Report Summary:")
print(f"Total Cost: ${cost_report['summary']['total_cost']:.2f}")
print(f"Total API Calls: {cost_report['summary']['total_operations']}")
print(f"Average Cost per Call: ${cost_report['summary']['average_cost_per_operation']:.4f}")

print("\nTop Research Categories by Cost:")
for category_data in cost_report['research_categories'][:5]:
    print(f"  {category_data['category']}: ${category_data['total_cost']:.2f} ({category_data['operation_count']} calls)")

print("\nDaily Cost Trends:")
for daily_data in cost_report['daily_trends'][-7:]:  # Last 7 days
    print(f"  {daily_data['date']}: ${daily_data['total_cost']:.2f} ({daily_data['operation_count']} calls)")
```

### Performance Analysis

```python
# Analyze performance trends from metrics
performance_summary = rag.get_api_metrics_summary()['performance_summary']

# Response time analysis
recent_perf = performance_summary['recent_performance']
print(f"Recent Average Response Time: {recent_perf['avg_response_time_ms']:.1f}ms")
print(f"Sample Size: {recent_perf['sample_size']} calls")

# Error analysis
hourly_stats = performance_summary['current_hour']
if hourly_stats['total_calls'] > 0:
    error_rate = hourly_stats['error_rate_percent']
    if error_rate > 5.0:
        print(f"⚠️  High error rate detected: {error_rate:.1f}%")
    else:
        print(f"✅ Error rate normal: {error_rate:.1f}%")

# Research category distribution
print("\nAPI Calls by Research Category:")
for category, count in performance_summary['top_research_categories'].items():
    print(f"  {category.replace('_', ' ').title()}: {count} calls")
```

## Troubleshooting

### Common Issues

1. **Metrics not being logged**
   ```python
   # Check if API metrics logging is enabled
   if not rag.api_metrics_logger:
       print("❌ API metrics logging not initialized")
       print("✅ Set enable_api_metrics_logging=True in config")
   ```

2. **Log files not created**
   ```python
   # Verify log directory permissions and file logging settings
   import os
   log_dir = Path(rag.config.log_dir)
   if not log_dir.exists():
       print(f"❌ Log directory does not exist: {log_dir}")
       log_dir.mkdir(parents=True, exist_ok=True)
       print(f"✅ Created log directory: {log_dir}")
   ```

3. **Performance impact**
   ```python
   # Monitor metrics logging overhead
   summary = rag.get_api_metrics_summary()
   active_ops = summary['performance_summary']['system']['active_operations']
   if active_ops > 10:
       print(f"⚠️  High number of active operations: {active_ops}")
   ```

### Debug Information

```python
# Get detailed debug information
debug_info = {
    'config_status': {
        'api_metrics_enabled': bool(rag.api_metrics_logger),
        'cost_tracking_enabled': rag.cost_tracking_enabled,
        'file_logging_enabled': rag.config.enable_file_logging,
        'log_directory': str(rag.config.log_dir)
    },
    'integration_status': {
        'cost_persistence': bool(rag.cost_persistence),
        'budget_manager': bool(rag.budget_manager),
        'research_categorizer': bool(rag.research_categorizer),
        'audit_trail': bool(rag.audit_trail)
    },
    'session_info': {
        'session_id': getattr(rag.api_metrics_logger, 'session_id', 'Not available'),
        'total_cost': rag.total_cost,
        'query_count': len(rag.query_history)
    }
}

print("Debug Information:")
print(json.dumps(debug_info, indent=2))
```

## Best Practices

### 1. Resource Management

```python
# Always properly close resources
try:
    # Your RAG operations
    result = await rag.query("Your query here")
finally:
    # Clean shutdown
    rag.close_api_metrics_logger()
```

### 2. Batch Operations

```python
# For batch processing, use batch logging
async def process_documents(documents):
    start_time = time.time()
    total_cost = 0
    total_tokens = 0
    success_count = 0
    error_count = 0
    
    for doc in documents:
        try:
            result = await rag.query(doc, mode="hybrid")
            success_count += 1
            # Extract cost and tokens from result
            total_cost += result.get('cost', 0)
            total_tokens += result.get('token_usage', {}).get('total_tokens', 0)
        except Exception as e:
            error_count += 1
    
    # Log batch metrics
    processing_time = time.time() - start_time
    rag.log_batch_api_operation(
        operation_name="document_batch_processing",
        batch_size=len(documents),
        total_tokens=total_tokens,
        total_cost=total_cost,
        processing_time_seconds=processing_time,
        success_count=success_count,
        error_count=error_count,
        research_category=ResearchCategory.DOCUMENT_PROCESSING.value
    )
```

### 3. Monitoring and Alerting

```python
# Set up monitoring for key metrics
def check_system_health(rag):
    summary = rag.get_api_metrics_summary()
    
    # Check error rates
    current_hour = summary['performance_summary']['current_hour']
    if current_hour['error_rate_percent'] > 10.0:
        rag.log_system_event(
            event_type='high_error_rate',
            event_data={
                'error_rate': current_hour['error_rate_percent'],
                'threshold': 10.0,
                'total_calls': current_hour['total_calls']
            }
        )
    
    # Check budget utilization
    enhanced_summary = rag.get_enhanced_cost_summary()
    if 'budget_status' in enhanced_summary:
        daily_usage = enhanced_summary['budget_status']['daily']['percentage_used']
        if daily_usage > 80.0:
            rag.log_system_event(
                event_type='budget_warning',
                event_data={
                    'daily_usage_percent': daily_usage,
                    'threshold': 80.0
                }
            )

# Run health check periodically
import asyncio
async def periodic_health_check(rag, interval_seconds=300):  # 5 minutes
    while True:
        check_system_health(rag)
        await asyncio.sleep(interval_seconds)
```

## Conclusion

The API Usage Metrics Logging System provides comprehensive monitoring and analytics capabilities for the Clinical Metabolomics Oracle LightRAG integration. By seamlessly integrating with the enhanced cost tracking system and providing detailed, structured logging, it enables:

- **Real-time Performance Monitoring**: Track API usage, costs, and performance in real-time
- **Research Analytics**: Understand API usage patterns across different metabolomics research areas
- **Budget Management**: Monitor and control API costs with detailed budget utilization tracking
- **Compliance Logging**: Maintain detailed audit trails for research compliance requirements
- **System Optimization**: Identify performance bottlenecks and optimization opportunities

The system is designed to be lightweight, thread-safe, and seamlessly integrated with existing workflows while providing powerful analytics and monitoring capabilities for production environments.