# LightRAG Configuration Reference

## Overview

The `LightRAGConfig` dataclass provides comprehensive configuration management for the Clinical Metabolomics Oracle's LightRAG integration. This document serves as a complete reference for all configuration fields, validation rules, factory methods, and best practices.

## Table of Contents

- [Configuration Fields Reference](#configuration-fields-reference)
- [Validation Rules](#validation-rules)
- [Factory Methods](#factory-methods)
- [Directory Management](#directory-management)
- [Logging Configuration](#logging-configuration)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)

## Configuration Fields Reference

### Core Configuration Fields

#### `api_key: Optional[str]`
- **Environment Variable**: `OPENAI_API_KEY`
- **Default**: `None`
- **Required**: Yes (for production use)
- **Purpose**: OpenAI API key for LLM operations and embeddings
- **Validation**: Must be present and non-empty string
- **Security**: Automatically masked in string representations

**Example**:
```python
config = LightRAGConfig(api_key="sk-your_openai_api_key_here")
```

#### `model: str`
- **Environment Variable**: `LIGHTRAG_MODEL`
- **Default**: `"gpt-4o-mini"`
- **Purpose**: Primary LLM model for LightRAG operations
- **Validation**: No specific validation (any string accepted)
- **Common Values**: `"gpt-4o-mini"`, `"gpt-4o"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"`

**Example**:
```python
config = LightRAGConfig(model="gpt-4o")
```

#### `embedding_model: str`
- **Environment Variable**: `LIGHTRAG_EMBEDDING_MODEL`
- **Default**: `"text-embedding-3-small"`
- **Purpose**: Embedding model for semantic similarity operations
- **Validation**: No specific validation (any string accepted)
- **Common Values**: `"text-embedding-3-small"`, `"text-embedding-3-large"`, `"text-embedding-ada-002"`

**Example**:
```python
config = LightRAGConfig(embedding_model="text-embedding-3-large")
```

### Directory Configuration

#### `working_dir: Path`
- **Environment Variable**: `LIGHTRAG_WORKING_DIR`
- **Default**: Current working directory (`Path.cwd()`)
- **Type**: `pathlib.Path`
- **Purpose**: Base directory for all LightRAG operations and storage
- **Validation**: Must exist or be creatable as a directory
- **Auto-Creation**: Created automatically if `auto_create_dirs=True`

**Example**:
```python
config = LightRAGConfig(working_dir=Path("/app/lightrag_storage"))
```

#### `graph_storage_dir: Optional[Path]`
- **Environment Variable**: None (derived field)
- **Default**: `{working_dir}/lightrag`
- **Type**: `pathlib.Path`
- **Purpose**: Directory for storing the knowledge graph database and indexes
- **Validation**: Created if parent directory exists
- **Auto-Creation**: Created automatically if `auto_create_dirs=True`

**Example**:
```python
config = LightRAGConfig(graph_storage_dir=Path("/app/graph_data"))
```

### Performance Configuration

#### `max_async: int`
- **Environment Variable**: `LIGHTRAG_MAX_ASYNC`
- **Default**: `16`
- **Purpose**: Maximum number of concurrent async operations
- **Validation**: Must be positive integer
- **Impact**: Higher values increase parallelism but consume more resources

**Example**:
```python
config = LightRAGConfig(max_async=32)  # For high-performance scenarios
```

#### `max_tokens: int`
- **Environment Variable**: `LIGHTRAG_MAX_TOKENS`
- **Default**: `32768`
- **Purpose**: Maximum token limit for LLM responses
- **Validation**: Must be positive integer
- **Impact**: Affects response length and processing cost

**Example**:
```python
config = LightRAGConfig(max_tokens=8000)  # For shorter responses
```

### System Configuration

#### `auto_create_dirs: bool`
- **Environment Variable**: None
- **Default**: `True`
- **Purpose**: Whether to automatically create directories during initialization
- **Impact**: When `True`, creates all necessary directories in `__post_init__`

### Logging Configuration

#### `log_level: str`
- **Environment Variable**: `LIGHTRAG_LOG_LEVEL`
- **Default**: `"INFO"`
- **Purpose**: Logging level for the application
- **Validation**: Must be one of `{"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}`
- **Normalization**: Automatically converted to uppercase

**Example**:
```python
config = LightRAGConfig(log_level="DEBUG")  # For detailed debugging
```

#### `log_dir: Path`
- **Environment Variable**: `LIGHTRAG_LOG_DIR`
- **Default**: `"logs"`
- **Type**: `pathlib.Path`
- **Purpose**: Directory for storing log files
- **Auto-Creation**: Created automatically if file logging is enabled

#### `enable_file_logging: bool`
- **Environment Variable**: `LIGHTRAG_ENABLE_FILE_LOGGING`
- **Default**: `True`
- **Purpose**: Whether to enable file-based logging
- **Environment Parsing**: Accepts `"true"`, `"1"`, `"yes"`, `"t"`, `"on"` as true values

#### `log_max_bytes: int`
- **Environment Variable**: `LIGHTRAG_LOG_MAX_BYTES`
- **Default**: `10485760` (10MB)
- **Purpose**: Maximum size of individual log files before rotation
- **Validation**: Must be positive integer

#### `log_backup_count: int`
- **Environment Variable**: `LIGHTRAG_LOG_BACKUP_COUNT`
- **Default**: `5`
- **Purpose**: Number of backup log files to keep during rotation
- **Validation**: Must be non-negative integer

#### `log_filename: str`
- **Environment Variable**: None
- **Default**: `"lightrag_integration.log"`
- **Purpose**: Name of the main log file
- **Validation**: Must be non-empty and end with `.log` extension

## Validation Rules

The `validate()` method enforces the following rules:

### Required Fields
- **API Key**: Must be present and non-empty string
- **Log Filename**: Must be non-empty and end with `.log`

### Numeric Constraints
- **max_async**: Must be positive (> 0)
- **max_tokens**: Must be positive (> 0)
- **log_max_bytes**: Must be positive (> 0)
- **log_backup_count**: Must be non-negative (>= 0)

### String Validation
- **log_level**: Must be one of the valid logging levels
- **log_filename**: Must have `.log` extension

### Directory Validation
- **working_dir**: Must exist or be creatable as a directory

### Validation Example
```python
config = LightRAGConfig(api_key="sk-test", max_async=-1)  # This will fail
try:
    config.validate()
except LightRAGConfigError as e:
    print(f"Validation failed: {e}")
    # Output: "Validation failed: max_async must be positive"
```

## Factory Methods

### `get_config()` - Primary Factory Method

The recommended entry point for creating configurations with intelligent source detection and comprehensive error handling.

```python
@classmethod
def get_config(cls, 
               source: Optional[Union[str, Path, Dict[str, Any]]] = None,
               validate_config: bool = True,
               ensure_dirs: bool = True,
               **overrides) -> 'LightRAGConfig':
```

#### Parameters
- **source**: Configuration source (None=environment, str/Path=file, dict=dictionary)
- **validate_config**: Whether to validate before returning (default: True)
- **ensure_dirs**: Whether to create directories (default: True)
- **overrides**: Additional configuration values to override

#### Usage Examples

**Load from environment variables:**
```python
config = LightRAGConfig.get_config()
```

**Load from file with overrides:**
```python
config = LightRAGConfig.get_config(
    source="/path/to/config.json",
    max_async=32,
    log_level="DEBUG"
)
```

**Load from dictionary without validation:**
```python
config = LightRAGConfig.get_config(
    source={"api_key": "sk-test", "model": "gpt-4o"},
    validate_config=False
)
```

### `from_environment()` - Environment Factory

Creates configuration from environment variables only.

```python
config = LightRAGConfig.from_environment(auto_create_dirs=True)
```

### `from_dict()` - Dictionary Factory

Creates configuration from a dictionary with automatic Path handling.

```python
config_dict = {
    "api_key": "sk-test",
    "working_dir": "/app/storage",
    "max_async": 24
}
config = LightRAGConfig.from_dict(config_dict)
```

### `from_file()` - File Factory

Creates configuration from a JSON file with error handling.

```python
config = LightRAGConfig.from_file("/path/to/config.json")
```

**Example config.json:**
```json
{
    "api_key": "sk-your_key_here",
    "model": "gpt-4o",
    "working_dir": "/app/lightrag_storage",
    "max_async": 32,
    "log_level": "DEBUG"
}
```

## Directory Management

### Automatic Directory Creation

When `auto_create_dirs=True` (default), the following directories are created:

1. **Working Directory**: Base directory for all operations
2. **Graph Storage Directory**: Knowledge graph and indexes
3. **Log Directory**: Log files (if file logging enabled)

### Manual Directory Management

Use `ensure_directories()` method for explicit directory creation:

```python
config = LightRAGConfig(auto_create_dirs=False)
# ... configure as needed ...
config.ensure_directories()  # Create all necessary directories
```

### Path Resolution

The `get_absolute_path()` method resolves relative paths from the working directory:

```python
config = LightRAGConfig(working_dir="/app/storage")
abs_path = config.get_absolute_path("data/files")  # Returns /app/storage/data/files
```

## Logging Configuration

### Logger Setup

Use the `setup_lightrag_logging()` method to create a fully configured logger:

```python
config = LightRAGConfig.get_config()
logger = config.setup_lightrag_logging("my_app")
logger.info("LightRAG integration started")
```

### Logging Features

- **Console Handler**: Always enabled with simple formatting
- **File Handler**: Optional with detailed formatting and rotation
- **Log Rotation**: Automatic based on file size and backup count
- **Error Handling**: Graceful fallback to console-only logging

### Standalone Logging

Use the standalone function for quick logger setup:

```python
from lightrag_integration.config import setup_lightrag_logging

# Use environment configuration
logger = setup_lightrag_logging()

# Use custom configuration
config = LightRAGConfig(log_level="DEBUG", log_dir="custom_logs")
logger = setup_lightrag_logging(config, "custom_logger")
```

## Usage Examples

### Basic Usage

```python
from lightrag_integration.config import LightRAGConfig

# Simple configuration from environment
config = LightRAGConfig.get_config()
logger = config.setup_lightrag_logging()
logger.info(f"Configuration loaded: {config}")
```

### Production Configuration

```python
# Production setup with validation and directory creation
config = LightRAGConfig.get_config(
    source={
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o",
        "working_dir": "/app/production/storage",
        "max_async": 64,
        "log_level": "INFO",
        "log_dir": "/var/log/lightrag",
        "log_max_bytes": 50 * 1024 * 1024,  # 50MB
        "log_backup_count": 10
    },
    validate_config=True,
    ensure_dirs=True
)
```

### Development Configuration

```python
# Development setup with debugging
config = LightRAGConfig.get_config(
    source="dev_config.json",
    log_level="DEBUG",
    max_async=8,  # Lower for development
    validate_config=False  # Skip validation for testing
)
```

### Configuration File Template

```json
{
    "api_key": "sk-your_openai_api_key_here",
    "model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "working_dir": "./lightrag_storage",
    "max_async": 16,
    "max_tokens": 32768,
    "log_level": "INFO",
    "log_dir": "./logs",
    "enable_file_logging": true,
    "log_max_bytes": 10485760,
    "log_backup_count": 5
}
```

## Best Practices

### Security
1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Use masked representations** in logs and debugging
4. **Validate configurations** before use in production

### Performance
1. **Tune `max_async`** based on system resources
2. **Set appropriate `max_tokens`** to control costs
3. **Use log rotation** to prevent disk space issues
4. **Monitor directory sizes** in production

### Reliability
1. **Always validate configurations** in production
2. **Use `ensure_dirs=True`** to prevent path errors
3. **Handle `LightRAGConfigError`** exceptions appropriately
4. **Test configurations** before deployment

### Development
1. **Use separate configurations** for dev/staging/production
2. **Enable debug logging** during development
3. **Use lower resource limits** for local testing
4. **Create configuration templates** for team consistency

## Error Handling

### Custom Exception

The `LightRAGConfigError` exception is raised for all configuration-related errors:

```python
from lightrag_integration.config import LightRAGConfigError

try:
    config = LightRAGConfig.get_config()
except LightRAGConfigError as e:
    print(f"Configuration error: {e}")
    # Handle gracefully
```

### Common Error Scenarios

1. **Missing API Key**
   ```python
   # Error: API key is required and cannot be empty
   config = LightRAGConfig(api_key="")
   config.validate()  # Raises LightRAGConfigError
   ```

2. **Invalid Numeric Values**
   ```python
   # Error: max_async must be positive
   config = LightRAGConfig(max_async=0)
   config.validate()  # Raises LightRAGConfigError
   ```

3. **Invalid Log Level**
   ```python
   # Error: log_level must be one of {valid_levels}
   config = LightRAGConfig(log_level="INVALID")
   config.validate()  # Raises LightRAGConfigError
   ```

4. **Directory Creation Failures**
   ```python
   # Error: Failed to create required directories
   config = LightRAGConfig.get_config(
       working_dir="/read_only_path",
       ensure_dirs=True
   )  # Raises LightRAGConfigError
   ```

### Error Recovery

```python
def create_config_with_fallback():
    try:
        # Try primary configuration
        return LightRAGConfig.get_config("production.json")
    except FileNotFoundError:
        try:
            # Fall back to environment
            return LightRAGConfig.get_config()
        except LightRAGConfigError:
            # Use minimal default configuration
            return LightRAGConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                validate_config=False
            )
```

## Serialization and Utilities

### Dictionary Conversion

```python
config = LightRAGConfig.get_config()
config_dict = config.to_dict()  # Convert to dictionary
print(json.dumps(config_dict, indent=2))  # Pretty print
```

### Deep Copying

```python
config = LightRAGConfig.get_config()
config_copy = config.copy()  # Create deep copy
config_copy.max_async = 32  # Modify copy without affecting original
```

### String Representations

```python
config = LightRAGConfig(api_key="sk-secret")
print(str(config))   # Masked: api_key=***masked***
print(repr(config))  # Detailed masked representation
```

This configuration reference provides comprehensive coverage of all LightRAGConfig features, validation rules, and usage patterns for the Clinical Metabolomics Oracle's LightRAG integration.