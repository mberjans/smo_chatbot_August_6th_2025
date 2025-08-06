# LightRAG Integration Logging Configuration

## Overview

The LightRAG Integration system provides comprehensive logging capabilities designed for production use in the Clinical Metabolomics Oracle chatbot. The logging system supports multiple configuration sources, flexible log levels, file rotation, and robust error handling.

## Quick Start

### Basic Usage

```python
from lightrag_integration import LightRAGConfig, setup_lightrag_logging

# Method 1: Use the standalone function (simplest)
logger = setup_lightrag_logging()
logger.info("Hello from LightRAG!")

# Method 2: Create config and setup logging
config = LightRAGConfig.get_config()
logger = config.setup_lightrag_logging("my_component")
logger.info("Component initialized successfully")
```

### Environment Variable Configuration

Set environment variables for automatic configuration:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LIGHTRAG_LOG_LEVEL="INFO"
export LIGHTRAG_LOG_DIR="logs"
export LIGHTRAG_ENABLE_FILE_LOGGING="true"
export LIGHTRAG_LOG_MAX_BYTES="10485760"  # 10MB
export LIGHTRAG_LOG_BACKUP_COUNT="5"
```

## Configuration Options

### Logging Parameters

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `log_level` | `LIGHTRAG_LOG_LEVEL` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `log_dir` | `LIGHTRAG_LOG_DIR` | `"logs"` | Directory for log files |
| `enable_file_logging` | `LIGHTRAG_ENABLE_FILE_LOGGING` | `true` | Enable/disable file logging |
| `log_max_bytes` | `LIGHTRAG_LOG_MAX_BYTES` | `10485760` | Maximum log file size in bytes |
| `log_backup_count` | `LIGHTRAG_LOG_BACKUP_COUNT` | `5` | Number of backup files to keep |
| `log_filename` | N/A | `"lightrag_integration.log"` | Log file name |

### Boolean Environment Variables

The `LIGHTRAG_ENABLE_FILE_LOGGING` variable accepts various formats:
- True values: `"true"`, `"1"`, `"yes"`, `"t"`, `"on"` (case-insensitive)
- False values: `"false"`, `"0"`, `"no"`, `"f"`, `"off"` (case-insensitive)

## Usage Examples

### 1. Different Configuration Sources

```python
from lightrag_integration import LightRAGConfig

# From environment variables (recommended)
config = LightRAGConfig.get_config()

# From dictionary
config = LightRAGConfig.get_config(source={
    "api_key": "your-api-key",
    "log_level": "DEBUG",
    "log_dir": "custom_logs",
    "enable_file_logging": True
})

# From JSON file
config = LightRAGConfig.get_config(source="config.json")

# With parameter overrides
config = LightRAGConfig.get_config(
    source="config.json",
    log_level="WARNING",
    max_async=32
)
```

### 2. Multiple Logger Configuration

```python
from lightrag_integration import LightRAGConfig

# Component A: Debug level with file logging
config_a = LightRAGConfig.get_config(source={
    "api_key": "your-api-key",
    "log_level": "DEBUG",
    "log_dir": "logs/component_a",
    "log_filename": "component_a.log"
})
logger_a = config_a.setup_lightrag_logging("component_a")

# Component B: Error level, console only
config_b = LightRAGConfig.get_config(source={
    "api_key": "your-api-key", 
    "log_level": "ERROR",
    "enable_file_logging": False
})
logger_b = config_b.setup_lightrag_logging("component_b")

# Usage
logger_a.debug("Detailed debug information")
logger_a.info("Component A operation completed")
logger_b.error("Critical error in component B")
```

### 3. Log Rotation Configuration

```python
from lightrag_integration import LightRAGConfig

config = LightRAGConfig.get_config(source={
    "api_key": "your-api-key",
    "log_level": "INFO", 
    "log_max_bytes": 5242880,  # 5MB per file
    "log_backup_count": 10,    # Keep 10 backup files
    "log_dir": "logs/rotated"
})

logger = config.setup_lightrag_logging("rotating_logger")

# This will automatically rotate when files exceed 5MB
for i in range(10000):
    logger.info(f"Log message {i}: This is a test of log rotation functionality")
```

### 4. Error Handling and Graceful Degradation

```python
from lightrag_integration import LightRAGConfig, LightRAGConfigError

try:
    # Try to create config with potentially problematic settings
    config = LightRAGConfig.get_config(source={
        "api_key": "your-api-key",
        "log_dir": "/read-only-directory",  # This might fail
        "enable_file_logging": True
    })
    
    logger = config.setup_lightrag_logging("resilient_logger")
    logger.info("Logging setup successful")
    
except LightRAGConfigError as e:
    print(f"Configuration error: {e}")
    # Fall back to console-only logging
    config = LightRAGConfig.get_config(source={
        "api_key": "your-api-key",
        "enable_file_logging": False
    })
    logger = config.setup_lightrag_logging("fallback_logger")
    logger.warning("File logging failed, using console only")
```

### 5. Integration with LightRAG Components

```python
from lightrag_integration import LightRAGConfig

class LightRAGComponent:
    def __init__(self, config_source=None):
        self.config = LightRAGConfig.get_config(source=config_source)
        self.logger = self.config.setup_lightrag_logging(self.__class__.__name__)
        
        self.logger.info("Component initialized")
        self.logger.debug(f"Configuration: {self.config}")
    
    def process_document(self, document):
        self.logger.info(f"Processing document: {document.name}")
        try:
            # Process document
            result = self._internal_process(document)
            self.logger.info("Document processed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Failed to process document: {e}")
            raise
    
    def _internal_process(self, document):
        self.logger.debug("Starting internal processing")
        # Implementation here
        pass

# Usage
component = LightRAGComponent({
    "api_key": "your-api-key",
    "log_level": "DEBUG",
    "log_dir": "logs/components"
})
```

## Log Output Formats

### Console Output (Simple Format)
```
INFO: Component initialized successfully
WARNING: File not found, using default settings
ERROR: Failed to connect to API
```

### File Output (Detailed Format)
```
2025-08-06 04:32:00,334 - component_name - INFO - Component initialized successfully
2025-08-06 04:32:00,456 - component_name - WARNING - File not found, using default settings  
2025-08-06 04:32:00,567 - component_name - ERROR - Failed to connect to API
```

## Best Practices

### 1. Use Environment Variables in Production

```bash
# .env file or system environment
OPENAI_API_KEY=your-actual-api-key
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_LOG_DIR=/var/log/smo-chatbot
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_MAX_BYTES=50331648  # 48MB
LIGHTRAG_LOG_BACKUP_COUNT=30
```

### 2. Component-Specific Loggers

```python
from lightrag_integration import LightRAGConfig

class PDFProcessor:
    def __init__(self):
        config = LightRAGConfig.get_config(source={
            "log_filename": "pdf_processor.log"
        })
        self.logger = config.setup_lightrag_logging("PDFProcessor")

class GraphBuilder:
    def __init__(self):
        config = LightRAGConfig.get_config(source={
            "log_filename": "graph_builder.log"
        })
        self.logger = config.setup_lightrag_logging("GraphBuilder")
```

### 3. Log Levels by Environment

```python
import os
from lightrag_integration import LightRAGConfig

# Determine log level based on environment
env = os.getenv("ENVIRONMENT", "production")
log_levels = {
    "development": "DEBUG",
    "testing": "INFO", 
    "staging": "WARNING",
    "production": "ERROR"
}

config = LightRAGConfig.get_config(source={
    "log_level": log_levels.get(env, "INFO")
})
```

### 4. Structured Logging for Analysis

```python
import json
from lightrag_integration import LightRAGConfig

config = LightRAGConfig.get_config()
logger = config.setup_lightrag_logging("structured")

# Log structured data
def log_operation(operation, duration, success, metadata=None):
    log_data = {
        "operation": operation,
        "duration_ms": duration,
        "success": success,
        "metadata": metadata or {}
    }
    
    if success:
        logger.info(f"Operation completed: {json.dumps(log_data)}")
    else:
        logger.error(f"Operation failed: {json.dumps(log_data)}")

# Usage
log_operation("document_processing", 1247, True, {
    "document_id": "doc_123",
    "pages": 15,
    "size_mb": 2.3
})
```

## Troubleshooting

### Common Issues

1. **Log files not created**
   - Check directory permissions
   - Verify `LIGHTRAG_ENABLE_FILE_LOGGING=true`
   - Ensure parent directories exist or `auto_create_dirs=True`

2. **Log rotation not working**
   - Verify `log_max_bytes` and `log_backup_count` settings
   - Check file system permissions
   - Ensure sufficient disk space

3. **Missing log messages**
   - Check log level configuration
   - Verify logger name matches expected pattern
   - Ensure handlers are properly configured

### Debug Configuration

```python
from lightrag_integration import LightRAGConfig

# Enable debug mode
config = LightRAGConfig.get_config(source={
    "api_key": "your-api-key",
    "log_level": "DEBUG"
}, validate_config=False)

logger = config.setup_lightrag_logging("debug")
logger.debug("This will help diagnose logging issues")

# Print configuration for debugging
print(f"Log level: {config.log_level}")
print(f"Log dir: {config.log_dir}")
print(f"File logging: {config.enable_file_logging}")
```

## Testing Your Logging Setup

Run the demonstration script to verify your logging configuration:

```bash
python lightrag_integration/demo_logging.py
```

This script will test:
- ✓ Basic logging functionality
- ✓ Different log levels
- ✓ Environment variable configuration  
- ✓ File logging and directory creation
- ✓ Log rotation
- ✓ Multiple logger configurations
- ✓ Error handling and recovery
- ✓ Standalone function usage

## Performance Considerations

### Log Level Impact
- `DEBUG`: High performance impact, use only in development
- `INFO`: Moderate impact, good for production monitoring
- `WARNING`: Low impact, recommended for production
- `ERROR`: Minimal impact, critical errors only

### File vs Console Logging
- File logging has minimal performance impact
- Console logging can be slower, especially with high volume
- Disable console logging in production if not needed

### Log Rotation Settings
```python
# High-volume application
config = LightRAGConfig.get_config(source={
    "log_max_bytes": 104857600,  # 100MB
    "log_backup_count": 50,      # Keep more history
})

# Low-volume application  
config = LightRAGConfig.get_config(source={
    "log_max_bytes": 1048576,    # 1MB
    "log_backup_count": 5,       # Less history needed
})
```

## Integration with Monitoring Systems

### Log Aggregation (ELK Stack)
```python
import json
from lightrag_integration import LightRAGConfig

config = LightRAGConfig.get_config(source={
    "log_dir": "/var/log/smo-chatbot"
})
logger = config.setup_lightrag_logging("elk_compatible")

# Structured logging for Elasticsearch
def log_for_elk(level, message, **kwargs):
    log_entry = {
        "timestamp": time.time(),
        "service": "smo-chatbot",
        "component": "lightrag",
        "message": message,
        **kwargs
    }
    
    if level == "error":
        logger.error(json.dumps(log_entry))
    elif level == "warning": 
        logger.warning(json.dumps(log_entry))
    else:
        logger.info(json.dumps(log_entry))
```

### Metrics and Alerting
```python
from lightrag_integration import LightRAGConfig

config = LightRAGConfig.get_config()
logger = config.setup_lightrag_logging("metrics")

def log_metric(metric_name, value, tags=None):
    """Log metrics in a format suitable for monitoring systems."""
    tags_str = ",".join(f"{k}={v}" for k, v in (tags or {}).items())
    logger.info(f"METRIC {metric_name}={value} {tags_str}")

# Usage
log_metric("document_processed", 1, {"component": "pdf_processor"})
log_metric("api_response_time", 245, {"endpoint": "chat", "status": "success"})
```

---

For more examples and advanced usage, see:
- `lightrag_integration/demo_logging.py` - Comprehensive demonstration script
- `lightrag_integration/tests/test_lightrag_config.py` - Complete test suite with 223+ test cases
- `lightrag_integration/config.py` - Full implementation with detailed docstrings