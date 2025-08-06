# LightRAG Integration Environment Variables Reference

## Overview

This document provides comprehensive documentation for all environment variables used by the Clinical Metabolomics Oracle LightRAG integration. It serves as the definitive reference for configuration parameters, resolving inconsistencies between different configuration files and providing clear guidance for all deployment scenarios.

## Quick Reference

| Variable Name | Type | Required | Default Value | Description |
|---------------|------|----------|---------------|-------------|
| `OPENAI_API_KEY` | string | **Yes** | - | OpenAI API key for LLM and embedding operations |
| `LIGHTRAG_MODEL` | string | No | `"gpt-4o-mini"` | LLM model for LightRAG operations |
| `LIGHTRAG_EMBEDDING_MODEL` | string | No | `"text-embedding-3-small"` | OpenAI embedding model |
| `LIGHTRAG_WORKING_DIR` | path | No | Current directory | Working directory for LightRAG storage |
| `LIGHTRAG_MAX_ASYNC` | integer | No | `16` | Maximum concurrent async operations |
| `LIGHTRAG_MAX_TOKENS` | integer | No | `32768` | Maximum tokens for LLM responses |
| `LIGHTRAG_LOG_LEVEL` | string | No | `"INFO"` | Logging level |
| `LIGHTRAG_LOG_DIR` | path | No | `"logs"` | Directory for log files |
| `LIGHTRAG_ENABLE_FILE_LOGGING` | boolean | No | `true` | Enable file-based logging |
| `LIGHTRAG_LOG_MAX_BYTES` | integer | No | `10485760` | Maximum log file size (10MB) |
| `LIGHTRAG_LOG_BACKUP_COUNT` | integer | No | `5` | Number of backup log files |

## Detailed Environment Variables

### Core Configuration

#### `OPENAI_API_KEY` ⚠️ **Required**
- **Type**: String
- **Default**: None
- **Description**: OpenAI API key for accessing GPT models and embeddings
- **Validation**: Must be non-empty string starting with "sk-"
- **Security**: 🔒 **Sensitive** - Never expose in logs or version control
- **Example**: `sk-your_actual_api_key_here`
- **Obtainable from**: [OpenAI Platform](https://platform.openai.com/api-keys)

#### `LIGHTRAG_MODEL`
- **Type**: String
- **Default**: `"gpt-4o-mini"`
- **Description**: Primary LLM model for LightRAG knowledge graph operations
- **Validation**: Must be a valid OpenAI model name
- **Valid Options**:
  - `"gpt-4o-mini"` - Recommended for cost-effectiveness
  - `"gpt-4o"` - Best performance, higher cost
  - `"gpt-4-turbo"` - Good balance of performance and cost
  - `"gpt-3.5-turbo"` - Fastest, lower accuracy for complex tasks
- **Example**: `LIGHTRAG_MODEL=gpt-4o-mini`
- **Performance Impact**: Higher-tier models provide better entity extraction and relationship mapping

#### `LIGHTRAG_EMBEDDING_MODEL`
- **Type**: String
- **Default**: `"text-embedding-3-small"`
- **Description**: OpenAI embedding model for semantic similarity calculations
- **Validation**: Must be a valid OpenAI embedding model
- **Valid Options**:
  - `"text-embedding-3-small"` - 1536 dimensions, cost-effective
  - `"text-embedding-3-large"` - 3072 dimensions, higher accuracy
  - `"text-embedding-ada-002"` - Legacy model, 1536 dimensions
- **Example**: `LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small`
- **Storage Impact**: Larger embedding dimensions require more storage space

### Directory Configuration

#### `LIGHTRAG_WORKING_DIR`
- **Type**: Path (string)
- **Default**: Current working directory
- **Description**: Base directory for all LightRAG data storage
- **Validation**: Must be a valid, accessible directory path
- **Auto-Creation**: Directory created automatically if `auto_create_dirs=True`
- **Structure Created**:
  ```
  LIGHTRAG_WORKING_DIR/
  ├── lightrag/           # Graph storage (auto-created)
  │   ├── graph_db/
  │   ├── embeddings/
  │   └── cache/
  ```
- **Example**: `LIGHTRAG_WORKING_DIR=/opt/lightrag/storage`
- **Security**: Ensure proper file permissions for the application user

### Performance Configuration

#### `LIGHTRAG_MAX_ASYNC`
- **Type**: Integer
- **Default**: `16`
- **Description**: Maximum number of concurrent async operations for LightRAG processing
- **Validation**: Must be positive integer (> 0)
- **Range**: Recommended 4-64 depending on system resources
- **Performance Impact**: Higher values increase throughput but consume more memory and API quota
- **Example**: `LIGHTRAG_MAX_ASYNC=32`
- **Tuning Guidelines**:
  - Development: 4-8
  - Production (small): 16-24
  - Production (large): 32-64

#### `LIGHTRAG_MAX_TOKENS`
- **Type**: Integer
- **Default**: `32768`
- **Description**: Maximum token limit for LLM responses
- **Validation**: Must be positive integer (> 0)
- **Model Limits**:
  - `gpt-4o-mini`: 128k context, 16k output
  - `gpt-4o`: 128k context, 4k output
  - `gpt-3.5-turbo`: 16k context, 4k output
- **Example**: `LIGHTRAG_MAX_TOKENS=16384`
- **Cost Impact**: Higher token limits increase API costs

### Logging Configuration

#### `LIGHTRAG_LOG_LEVEL`
- **Type**: String
- **Default**: `"INFO"`
- **Description**: Logging verbosity level
- **Validation**: Must be valid logging level (case-insensitive)
- **Valid Options**:
  - `"DEBUG"` - Verbose debugging information
  - `"INFO"` - General operational information
  - `"WARNING"` - Warning messages only
  - `"ERROR"` - Error messages only
  - `"CRITICAL"` - Critical errors only
- **Example**: `LIGHTRAG_LOG_LEVEL=DEBUG`
- **Performance Impact**: DEBUG level may impact performance in production

#### `LIGHTRAG_LOG_DIR`
- **Type**: Path (string)
- **Default**: `"logs"`
- **Description**: Directory for storing log files
- **Validation**: Must be valid directory path
- **Auto-Creation**: Created automatically if `enable_file_logging=True`
- **Example**: `LIGHTRAG_LOG_DIR=/var/log/lightrag`
- **File Created**: `{LIGHTRAG_LOG_DIR}/lightrag_integration.log`

#### `LIGHTRAG_ENABLE_FILE_LOGGING`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable or disable file-based logging
- **Validation**: Accepts boolean strings: `true`/`false`, `1`/`0`, `yes`/`no`, `t`/`f`, `on`/`off`
- **Example**: `LIGHTRAG_ENABLE_FILE_LOGGING=true`
- **Impact**: When disabled, only console logging is active

#### `LIGHTRAG_LOG_MAX_BYTES`
- **Type**: Integer
- **Default**: `10485760` (10MB)
- **Description**: Maximum size of individual log files before rotation
- **Validation**: Must be positive integer (> 0)
- **Common Values**:
  - `1048576` (1MB) - Development
  - `10485760` (10MB) - Small production
  - `104857600` (100MB) - Large production
- **Example**: `LIGHTRAG_LOG_MAX_BYTES=52428800`

#### `LIGHTRAG_LOG_BACKUP_COUNT`
- **Type**: Integer
- **Default**: `5`
- **Description**: Number of backup log files to retain during rotation
- **Validation**: Must be non-negative integer (≥ 0)
- **Example**: `LIGHTRAG_LOG_BACKUP_COUNT=10`
- **Storage Impact**: Higher values retain more log history but use more disk space

## Configuration Inconsistencies Resolved

### Model Variable Name
**Issue**: Mismatch between `config.py` and `.env.example`
- `config.py` expects: `LIGHTRAG_MODEL`
- `.env.example` had: `LIGHTRAG_LLM_MODEL`

**Resolution**: Use `LIGHTRAG_MODEL` (as implemented in config.py)

### Missing Variables in .env.example
The following variables were missing from `.env.example` but are implemented in `config.py`:
- `LIGHTRAG_MAX_ASYNC`
- `LIGHTRAG_LOG_LEVEL`
- `LIGHTRAG_LOG_DIR`
- `LIGHTRAG_ENABLE_FILE_LOGGING`
- `LIGHTRAG_LOG_MAX_BYTES`
- `LIGHTRAG_LOG_BACKUP_COUNT`

### Token Limit Default Value
**Issue**: Different default values
- `config.py`: `32768` tokens
- `.env.example`: `8000` tokens

**Resolution**: Use `32768` (as implemented in config.py) for better performance with modern models

## Environment-Specific Configurations

### Development Environment
```bash
# Minimal configuration for development
OPENAI_API_KEY=sk-your_dev_key_here
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_WORKING_DIR=./dev_lightrag
LIGHTRAG_MAX_ASYNC=4
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_FILE_LOGGING=true
```

### Staging Environment
```bash
# Staging configuration with monitoring
OPENAI_API_KEY=sk-your_staging_key_here
LIGHTRAG_MODEL=gpt-4o-mini
LIGHTRAG_WORKING_DIR=/opt/lightrag/staging
LIGHTRAG_MAX_ASYNC=16
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_LOG_DIR=/var/log/lightrag-staging
LIGHTRAG_LOG_MAX_BYTES=10485760
LIGHTRAG_LOG_BACKUP_COUNT=5
```

### Production Environment
```bash
# Production configuration optimized for performance
OPENAI_API_KEY=sk-your_production_key_here
LIGHTRAG_MODEL=gpt-4o
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
LIGHTRAG_WORKING_DIR=/opt/lightrag/production
LIGHTRAG_MAX_ASYNC=32
LIGHTRAG_MAX_TOKENS=32768
LIGHTRAG_LOG_LEVEL=INFO
LIGHTRAG_LOG_DIR=/var/log/lightrag
LIGHTRAG_ENABLE_FILE_LOGGING=true
LIGHTRAG_LOG_MAX_BYTES=104857600
LIGHTRAG_LOG_BACKUP_COUNT=10
```

## Security Considerations

### API Key Management
- **Never commit API keys** to version control
- Use environment-specific `.env` files
- Consider using secret management services in production:
  - AWS Secrets Manager
  - Azure Key Vault
  - HashiCorp Vault
  - Kubernetes Secrets

### File Permissions
Ensure proper permissions for data directories:
```bash
# Set secure permissions
chmod 750 /opt/lightrag/production
chown -R lightrag:lightrag /opt/lightrag/production

# Log directory permissions
chmod 755 /var/log/lightrag
chown -R lightrag:lightrag /var/log/lightrag
```

### Network Security
- Restrict API access to required endpoints only
- Use HTTPS for all API communications
- Monitor API usage and rate limits

## Validation and Error Handling

### Configuration Validation
The `LightRAGConfig` class provides comprehensive validation:

```python
from lightrag_integration.config import LightRAGConfig

# Validate configuration
try:
    config = LightRAGConfig.get_config(validate_config=True)
    print("Configuration valid!")
except LightRAGConfigError as e:
    print(f"Configuration error: {e}")
```

### Common Validation Errors
- **Missing API Key**: `OPENAI_API_KEY` not set or empty
- **Invalid Numeric Values**: Negative values for `LIGHTRAG_MAX_ASYNC`, `LIGHTRAG_MAX_TOKENS`
- **Invalid Log Level**: Unsupported logging level
- **Directory Access**: Cannot create or access working directories
- **Invalid Model Names**: Non-existent OpenAI models

### Error Recovery
```python
# Example with fallback configuration
try:
    config = LightRAGConfig.get_config()
except LightRAGConfigError:
    # Use minimal safe configuration
    config = LightRAGConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",  # Fallback model
        max_async=4,  # Conservative async limit
        enable_file_logging=False  # Disable if directory issues
    )
```

## Testing Configuration

### Unit Testing
```python
# Test configuration with mock environment
import os
from unittest.mock import patch

@patch.dict(os.environ, {
    'OPENAI_API_KEY': 'sk-test-key',
    'LIGHTRAG_MODEL': 'gpt-4o-mini',
    'LIGHTRAG_MAX_ASYNC': '8'
})
def test_config_from_environment():
    config = LightRAGConfig.from_environment()
    assert config.model == 'gpt-4o-mini'
    assert config.max_async == 8
```

### Integration Testing
```bash
# Test with actual environment file
cp .env.example .env.test
# Edit .env.test with test values
export $(cat .env.test | xargs)
python -c "from lightrag_integration.config import LightRAGConfig; LightRAGConfig.get_config().validate()"
```

## Migration Guide

### From Legacy Configuration
If migrating from older configuration formats:

1. **Update variable names**:
   ```bash
   # Old format
   LIGHTRAG_LLM_MODEL=gpt-4o-mini
   
   # New format
   LIGHTRAG_MODEL=gpt-4o-mini
   ```

2. **Add missing logging variables**:
   ```bash
   # Add these to existing .env
   LIGHTRAG_LOG_LEVEL=INFO
   LIGHTRAG_LOG_DIR=logs
   LIGHTRAG_ENABLE_FILE_LOGGING=true
   LIGHTRAG_LOG_MAX_BYTES=10485760
   LIGHTRAG_LOG_BACKUP_COUNT=5
   ```

3. **Update token limits**:
   ```bash
   # Update from 8000 to 32768
   LIGHTRAG_MAX_TOKENS=32768
   ```

## Troubleshooting

### Common Issues

#### Configuration Not Loading
```bash
# Check environment variable is set
echo $OPENAI_API_KEY

# Test with Python
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

#### Directory Creation Failures
```bash
# Check permissions
ls -la /opt/lightrag/
stat /opt/lightrag/production

# Fix permissions
sudo chown -R $USER:$USER /opt/lightrag/
```

#### Logging Issues
```bash
# Check log directory exists and is writable
mkdir -p /var/log/lightrag
chmod 755 /var/log/lightrag

# Test log file creation
touch /var/log/lightrag/test.log
```

### Debug Mode
Enable comprehensive debugging:
```bash
LIGHTRAG_LOG_LEVEL=DEBUG
LIGHTRAG_ENABLE_FILE_LOGGING=true
```

## Best Practices

### Development
- Use separate API keys for development and production
- Enable debug logging during development
- Use smaller async limits to avoid quota exhaustion
- Test with minimal viable configurations first

### Production
- Use environment-specific secret management
- Monitor log file sizes and implement rotation
- Set conservative API limits initially, then optimize
- Implement health checks for configuration validity
- Use monitoring tools to track API usage and costs

### Maintenance
- Regularly rotate API keys
- Monitor log storage usage
- Review and optimize async settings based on usage patterns
- Keep backup copies of working configurations
- Document any custom configuration changes

## Related Files

- `/lightrag_integration/config.py` - Main configuration implementation
- `/.env.example` - Environment variable template
- `/lightrag_integration/tests/test_lightrag_config.py` - Configuration tests

## Version History

- **v1.0**: Initial environment variables documentation
- **v1.1**: Resolved inconsistencies between config.py and .env.example
- **v1.2**: Added comprehensive validation rules and security guidelines

---

*Last updated: August 6, 2025*  
*Task: CMO-LIGHTRAG-002-T10*