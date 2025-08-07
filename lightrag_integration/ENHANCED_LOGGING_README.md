# Enhanced Logging System for Clinical Metabolomics Oracle

## Overview

The Enhanced Logging System provides comprehensive logging capabilities for troubleshooting ingestion issues and monitoring the performance of the Clinical Metabolomics Oracle LightRAG integration. This system builds upon the existing logging infrastructure with structured logging, correlation tracking, performance metrics, and specialized loggers for different components.

## Key Features

### 🔗 Correlation ID Tracking
- **Automatic correlation ID generation** for tracking related operations across the system
- **Nested operation context support** with parent-child relationship tracking
- **Thread-safe correlation management** for concurrent operations
- **Context propagation** across function calls and async operations

### 📊 Structured Logging
- **JSON-formatted log entries** for better parsing and analysis
- **Metadata enrichment** with contextual information
- **Consistent log schema** across all components
- **Separate structured log files** for machine processing

### ⚡ Performance Metrics
- **Automatic performance tracking** with decorators
- **Memory usage monitoring** during operations
- **CPU utilization tracking** for resource optimization
- **API call timing and cost tracking** for budget management
- **Batch processing metrics** for ingestion optimization

### 🎯 Specialized Loggers
- **Enhanced Logger**: Core structured logging with correlation IDs
- **Ingestion Logger**: Document and batch processing specific logging
- **Diagnostic Logger**: Configuration, storage, and system diagnostics
- **Performance Logger**: Detailed performance metrics and analysis

### 🔍 Error Context Capture
- **Detailed error information** with stack traces
- **Error correlation tracking** for related failures
- **Retry attempt logging** with backoff strategies
- **Recovery mechanism tracking** for resilience monitoring

## Architecture

```
Enhanced Logging System
├── enhanced_logging.py          # Core logging infrastructure
├── test_enhanced_logging_system.py  # Comprehensive test suite
├── demo_enhanced_logging.py     # Demonstration and examples
└── ENHANCED_LOGGING_README.md   # This documentation

Core Components:
├── CorrelationIDManager         # Thread-safe correlation tracking
├── StructuredLogRecord         # Structured log data containers
├── EnhancedLogger             # Core enhanced logging functionality
├── IngestionLogger           # Specialized for document processing
├── DiagnosticLogger         # System diagnostics and configuration
└── PerformanceTracker      # Performance metrics collection
```

## Integration with Existing System

The enhanced logging system seamlessly integrates with the existing Clinical Metabolomics Oracle infrastructure:

- **LightRAGConfig**: Extended logging configuration options
- **ClinicalMetabolomicsRAG**: Automatic enhanced logging initialization
- **Progress Tracking**: Integration with unified progress tracking
- **API Metrics**: Enhanced API call logging and monitoring
- **Cost Tracking**: Detailed cost logging with context

## Usage Examples

### Basic Enhanced Logging

```python
from enhanced_logging import EnhancedLogger, correlation_manager
import logging

# Create base logger
base_logger = logging.getLogger("my_application")
enhanced_logger = EnhancedLogger(base_logger, component="my_component")

# Basic structured logging
enhanced_logger.info(
    "Starting document processing",
    operation_name="document_processing",
    metadata={"document_count": 50, "batch_size": 10}
)

# Error logging with context
try:
    risky_operation()
except Exception as e:
    enhanced_logger.log_error_with_context(
        "Document processing failed",
        e,
        operation_name="document_processing",
        additional_context={"document_id": "doc123", "retry_count": 2}
    )
```

### Correlation Tracking

```python
from enhanced_logging import correlation_manager

# Track operations with correlation IDs
with correlation_manager.operation_context("knowledge_base_initialization") as context:
    enhanced_logger.info("Starting initialization")
    
    with correlation_manager.operation_context("pdf_processing") as sub_context:
        enhanced_logger.info("Processing PDFs")
        # All logs in this context will have the same correlation ID
        # and show parent-child relationship
    
    enhanced_logger.info("Initialization complete")
```

### Performance Monitoring

```python
from enhanced_logging import performance_logged, PerformanceTracker

# Automatic performance logging with decorator
@performance_logged("expensive_operation", enhanced_logger)
def expensive_operation(data):
    # Function implementation
    return processed_data

# Manual performance tracking
tracker = PerformanceTracker()
tracker.start_tracking()

# ... do work ...

metrics = tracker.get_metrics()
enhanced_logger.log_performance_metrics("my_operation", metrics)
```

### Specialized Loggers

```python
from enhanced_logging import create_enhanced_loggers

# Create all specialized loggers
loggers = create_enhanced_loggers(base_logger)

# Ingestion logging
loggers['ingestion'].log_document_start("doc123", "/path/to/doc.pdf")
loggers['ingestion'].log_document_complete("doc123", 1500.0, 10, 5000)

# Diagnostic logging  
loggers['diagnostic'].log_api_call_details("llm_completion", "gpt-4o-mini", 1000, 0.0025, 850.0, True)
loggers['diagnostic'].log_memory_usage("processing", 512.0, 45.2)
```

### Integration with ClinicalMetabolomicsRAG

```python
from clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from config import LightRAGConfig

# Enhanced logging is automatically initialized
config = LightRAGConfig(enable_file_logging=True, log_level="DEBUG")
rag = ClinicalMetabolomicsRAG(config)

# Enhanced loggers are available
if rag.enhanced_loggers:
    rag.enhanced_loggers['ingestion'].log_batch_start("batch_001", 10, 5, 0)
```

## Configuration

### Environment Variables

```bash
# Basic logging configuration
export LIGHTRAG_LOG_LEVEL=DEBUG
export LIGHTRAG_ENABLE_FILE_LOGGING=true
export LIGHTRAG_LOG_DIR=logs

# Enhanced logging specific
export LIGHTRAG_ENABLE_STRUCTURED_LOGGING=true
export LIGHTRAG_STRUCTURED_LOG_FILE=logs/structured.jsonl
export LIGHTRAG_ENABLE_PERFORMANCE_TRACKING=true
export LIGHTRAG_ENABLE_CORRELATION_TRACKING=true
```

### Programmatic Configuration

```python
from config import LightRAGConfig

config = LightRAGConfig(
    log_level="DEBUG",
    enable_file_logging=True,
    log_dir=Path("custom_logs"),
    log_max_bytes=20*1024*1024,  # 20MB
    log_backup_count=10
)
```

## Log Format and Schema

### Regular Log Format
```
2025-08-07 10:30:15,123 - clinical_metabolomics_rag - INFO - [abc-123-def] Starting document processing
```

### Structured Log Format (JSON)
```json
{
  "timestamp": "2025-08-07T10:30:15.123456Z",
  "level": "INFO",
  "message": "Starting document processing",
  "correlation_id": "abc-123-def-456",
  "operation_name": "document_processing",
  "component": "ingestion",
  "metadata": {
    "document_count": 50,
    "batch_size": 10
  },
  "performance_metrics": {
    "memory_mb": 512.5,
    "duration_ms": 1500.0
  }
}
```

## Performance Impact

The enhanced logging system is designed to have minimal performance impact:

- **Basic logging**: < 1ms per message
- **Structured logging**: < 5ms per message  
- **Error logging with context**: < 10ms per message
- **Performance tracking**: < 2ms overhead per operation

## Testing

### Running Tests

```bash
# Run all tests
pytest test_enhanced_logging_system.py -v

# Run specific test categories
python test_enhanced_logging_system.py --category basic
python test_enhanced_logging_system.py --category performance
python test_enhanced_logging_system.py --category integration
```

### Performance Benchmarking

```bash
# Run performance benchmarks
python test_enhanced_logging_system.py --category performance
```

## Demonstration

### Running Demos

```bash
# Run all demonstrations
python demo_enhanced_logging.py

# Run specific demonstrations
python demo_enhanced_logging.py --demo correlation
python demo_enhanced_logging.py --demo performance
python demo_enhanced_logging.py --demo ingestion
python demo_enhanced_logging.py --demo diagnostics
```

### Demo Output

The demo creates comprehensive log files showing:
- Correlation ID tracking across nested operations
- Performance metrics for different operations
- Structured logging with rich metadata
- Error handling with detailed context
- Integration scenarios with the main RAG system

## Troubleshooting

### Common Issues

1. **No structured logs generated**
   - Check `LIGHTRAG_ENABLE_FILE_LOGGING=true`
   - Verify log directory permissions
   - Check that structured logger is properly initialized

2. **Missing correlation IDs**
   - Ensure operations are wrapped in `correlation_manager.operation_context()`
   - Check that enhanced loggers are being used instead of base logger

3. **Performance metrics not collected**
   - Verify that `PerformanceTracker` is initialized
   - Check system permissions for memory/CPU monitoring
   - Ensure `psutil` is installed and working

4. **Log file rotation not working**
   - Check `log_max_bytes` and `log_backup_count` configuration
   - Verify disk space availability
   - Check file permissions for log directory

### Debug Mode

Enable debug mode for enhanced logging diagnostics:

```python
# Enable debug mode
enhanced_logger.debug("Enhanced logging debug info", 
                     metadata={"debug_mode": True})

# Check logger configuration
print(f"Enhanced loggers available: {bool(rag.enhanced_loggers)}")
print(f"Structured logger active: {bool(rag.structured_logger)}")
```

## Best Practices

### 1. Operation Context Management
Always wrap significant operations in correlation contexts:

```python
with correlation_manager.operation_context("operation_name") as context:
    # All logging within this block will have the same correlation ID
    enhanced_logger.info("Operation started")
    # ... do work ...
    enhanced_logger.info("Operation completed")
```

### 2. Metadata Usage
Include relevant metadata for better debugging:

```python
enhanced_logger.info(
    "Processing batch",
    metadata={
        "batch_id": batch_id,
        "document_count": len(documents),
        "estimated_duration": "5m",
        "memory_limit": "2GB"
    }
)
```

### 3. Error Context
Always include context when logging errors:

```python
enhanced_logger.log_error_with_context(
    "Failed to process document",
    error,
    operation_name="document_processing",
    additional_context={
        "document_id": doc_id,
        "document_path": doc_path,
        "retry_count": retry_count,
        "batch_id": batch_id
    }
)
```

### 4. Performance Monitoring
Use performance decorators for critical functions:

```python
@performance_logged("critical_operation", enhanced_logger)
def critical_operation(data):
    # Function automatically gets performance monitoring
    return process(data)
```

## Integration Points

### With Existing Systems

1. **LightRAG Integration**: Automatic initialization in `ClinicalMetabolomicsRAG`
2. **Progress Tracking**: Enhanced logging integrates with unified progress tracking
3. **Cost Monitoring**: API cost tracking includes enhanced logging context
4. **Error Recovery**: Recovery mechanisms include detailed logging

### With External Tools

1. **Log Aggregation**: Structured logs can be ingested by ELK stack, Splunk, etc.
2. **Monitoring**: Performance metrics can be exported to Prometheus, DataDog
3. **Alerting**: Error patterns can trigger alerts through log monitoring
4. **Analytics**: Correlation IDs enable end-to-end transaction tracking

## Future Enhancements

- **Distributed Tracing**: Integration with OpenTelemetry
- **Real-time Monitoring**: WebSocket-based real-time log streaming
- **Advanced Analytics**: Machine learning-based anomaly detection
- **Custom Metrics**: User-defined performance metrics
- **Log Sampling**: Intelligent log sampling for high-volume scenarios

## Support

For issues, questions, or feature requests related to the enhanced logging system:

1. Check this documentation and the demo scripts
2. Run the test suite to verify functionality
3. Review log outputs for diagnostic information
4. Check system resources and permissions

The enhanced logging system is designed to be self-diagnosing and will log its own initialization and configuration status.