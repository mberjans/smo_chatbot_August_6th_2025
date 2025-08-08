# Feature Flags Guide: Clinical Metabolomics Oracle LightRAG Integration

## Overview

The LightRAG Integration module now supports comprehensive feature flags that allow for conditional imports, graceful degradation, and environment-based initialization. This system provides fine-grained control over which components are loaded and enabled, making the integration more flexible and suitable for different deployment scenarios.

## Core Concepts

### 1. Feature Flags
Environment variables that control which features are enabled:

```bash
# Core integration control
export LIGHTRAG_INTEGRATION_ENABLED=true

# Quality validation features  
export LIGHTRAG_ENABLE_QUALITY_VALIDATION=true
export LIGHTRAG_ENABLE_RELEVANCE_SCORING=true
export LIGHTRAG_ENABLE_ACCURACY_VALIDATION=false
export LIGHTRAG_ENABLE_FACTUAL_VALIDATION=false

# Performance monitoring
export LIGHTRAG_ENABLE_PERFORMANCE_MONITORING=false
export LIGHTRAG_ENABLE_BENCHMARKING=false

# Advanced features
export LIGHTRAG_ENABLE_DOCUMENT_INDEXING=false
export LIGHTRAG_ENABLE_RECOVERY_SYSTEM=false
```

### 2. Conditional Imports
Modules are only imported if their corresponding feature flag is enabled:

```python
# Only imports if LIGHTRAG_ENABLE_RELEVANCE_SCORING=true
if is_feature_enabled('relevance_scoring_enabled'):
    from .relevance_scorer import RelevanceScorer
else:
    RelevanceScorer = None
```

### 3. Graceful Degradation
When features are disabled, the system provides `None` stubs instead of raising import errors, allowing dependent code to check for availability.

## Available Feature Flags

### Core Integration Control
- `LIGHTRAG_INTEGRATION_ENABLED` - Master switch for LightRAG integration
- `LIGHTRAG_ENABLE_COST_TRACKING` - Enable cost tracking and budget management
- `LIGHTRAG_ENABLE_CIRCUIT_BREAKER` - Enable circuit breaker pattern for reliability

### Quality Validation Suite
- `LIGHTRAG_ENABLE_QUALITY_VALIDATION` - Master switch for quality validation features
- `LIGHTRAG_ENABLE_RELEVANCE_SCORING` - Enable relevance scoring for responses
- `LIGHTRAG_ENABLE_ACCURACY_VALIDATION` - Enable accuracy validation
- `LIGHTRAG_ENABLE_FACTUAL_VALIDATION` - Enable factual accuracy validation
- `LIGHTRAG_ENABLE_CLAIM_EXTRACTION` - Enable claim extraction and validation

### Performance Monitoring
- `LIGHTRAG_ENABLE_PERFORMANCE_MONITORING` - Enable performance monitoring features
- `LIGHTRAG_ENABLE_BENCHMARKING` - Enable benchmarking suites
- `LIGHTRAG_ENABLE_PROGRESS_TRACKING` - Enable progress tracking
- `LIGHTRAG_ENABLE_UNIFIED_PROGRESS_TRACKING` - Enable unified progress tracking

### Document Processing
- `LIGHTRAG_ENABLE_PDF_PROCESSING` - Enable PDF processing (enabled by default)
- `LIGHTRAG_ENABLE_DOCUMENT_INDEXING` - Enable document indexing features

### Advanced Features
- `LIGHTRAG_ENABLE_BUDGET_MONITORING` - Enable real-time budget monitoring
- `LIGHTRAG_ENABLE_RECOVERY_SYSTEM` - Enable advanced recovery system
- `LIGHTRAG_ENABLE_ALERT_SYSTEM` - Enable alert system
- `LIGHTRAG_ENABLE_AB_TESTING` - Enable A/B testing capabilities
- `LIGHTRAG_ENABLE_CONDITIONAL_ROUTING` - Enable conditional routing

### Development and Debugging
- `LIGHTRAG_DEBUG_MODE` - Enable debug mode
- `LIGHTRAG_ENABLE_DEVELOPMENT_FEATURES` - Enable development-only features

## Usage Examples

### Basic Usage

```python
import lightrag_integration

# Check if a feature is enabled
if lightrag_integration.is_feature_enabled('quality_validation_enabled'):
    print("Quality validation is available")

# Get all enabled features
enabled = lightrag_integration.get_enabled_features()
print(f"Enabled features: {list(enabled.keys())}")

# Get comprehensive integration status
status = lightrag_integration.get_integration_status()
print(f"Integration health: {status['integration_health']}")
```

### Feature-Aware Factory Functions

```python
import lightrag_integration

# Create system with feature-flag aware defaults
rag = lightrag_integration.create_clinical_rag_system_with_features()

# Create system optimized for quality validation (requires feature enabled)
if lightrag_integration.is_feature_enabled('quality_validation_enabled'):
    rag = lightrag_integration.create_quality_validation_system()

# Create system optimized for performance monitoring
if lightrag_integration.is_feature_enabled('performance_monitoring_enabled'):
    rag = lightrag_integration.create_performance_monitoring_system()
```

### Safe Component Usage

```python
import lightrag_integration

# Safe way to use optional components
if lightrag_integration.RelevanceScorer is not None:
    scorer = lightrag_integration.RelevanceScorer()
    # Use scorer...
else:
    print("Relevance scoring not available")

# Check availability before using
if lightrag_integration.is_feature_enabled('document_indexing_enabled'):
    indexer = lightrag_integration.DocumentIndexer()
    # Use indexer...
```

### Environment-Based Configuration

```bash
# Production environment - minimal features
export LIGHTRAG_INTEGRATION_ENABLED=true
export LIGHTRAG_ENABLE_COST_TRACKING=true
export LIGHTRAG_ENABLE_QUALITY_VALIDATION=false
export LIGHTRAG_ENABLE_PERFORMANCE_MONITORING=false

# Development environment - full features
export LIGHTRAG_INTEGRATION_ENABLED=true
export LIGHTRAG_ENABLE_QUALITY_VALIDATION=true
export LIGHTRAG_ENABLE_RELEVANCE_SCORING=true
export LIGHTRAG_ENABLE_PERFORMANCE_MONITORING=true
export LIGHTRAG_ENABLE_BENCHMARKING=true
export LIGHTRAG_DEBUG_MODE=true

# Research environment - quality focused
export LIGHTRAG_INTEGRATION_ENABLED=true
export LIGHTRAG_ENABLE_QUALITY_VALIDATION=true
export LIGHTRAG_ENABLE_RELEVANCE_SCORING=true
export LIGHTRAG_ENABLE_ACCURACY_VALIDATION=true
export LIGHTRAG_ENABLE_FACTUAL_VALIDATION=true
export LIGHTRAG_ENABLE_DOCUMENT_INDEXING=true
```

## Integration Status Monitoring

### Get Status Information

```python
import lightrag_integration

# Get comprehensive status
status = lightrag_integration.get_integration_status()

print(f"Integration Health: {status['integration_health']}")
print(f"Total Feature Flags: {len(status['feature_flags'])}")
print(f"Enabled Features: {len([f for f in status['feature_flags'].values() if f])}")

# Check individual modules
for module_name, module_status in status['modules'].items():
    print(f"{module_name}: enabled={module_status['enabled']}, available={module_status['available']}")
```

### Validation

```python
import lightrag_integration

# Validate integration setup
is_valid, issues = lightrag_integration.validate_integration_setup()

if not is_valid:
    print("Setup issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

## Dynamic Export Management

The module uses dynamic export management, meaning `__all__` is built based on available features:

```python
import lightrag_integration

# Only symbols for enabled features are exported
print(f"Available symbols: {len(lightrag_integration.__all__)}")

# Check if specific symbols are available
if 'RelevanceScorer' in lightrag_integration.__all__:
    print("RelevanceScorer is available for import")
```

## Migration Guide

### From Static Imports

**Before:**
```python
from lightrag_integration import RelevanceScorer, QualityReportGenerator

# This could fail if features are disabled
scorer = RelevanceScorer()
```

**After:**
```python
import lightrag_integration

# Safe approach with feature checking
if lightrag_integration.is_feature_enabled('relevance_scoring_enabled'):
    scorer = lightrag_integration.RelevanceScorer()
else:
    print("Relevance scoring not available")

# Or check for None
if lightrag_integration.RelevanceScorer is not None:
    scorer = lightrag_integration.RelevanceScorer()
```

### Factory Function Migration

**Before:**
```python
from lightrag_integration import create_clinical_rag_system

rag = create_clinical_rag_system()
```

**After:**
```python
from lightrag_integration import create_clinical_rag_system_with_features

# Automatically applies feature-flag aware defaults
rag = create_clinical_rag_system_with_features()
```

## Best Practices

### 1. Always Check Feature Availability

```python
# Good
if lightrag_integration.is_feature_enabled('quality_validation_enabled'):
    # Use quality validation features
    
# Bad - may fail if disabled
scorer = lightrag_integration.RelevanceScorer()
```

### 2. Use Feature-Aware Factory Functions

```python
# Good - respects feature flags
rag = lightrag_integration.create_clinical_rag_system_with_features()

# Also good - but manual configuration needed
rag = lightrag_integration.create_clinical_rag_system(
    enable_relevance_scoring=lightrag_integration.is_feature_enabled('relevance_scoring_enabled')
)
```

### 3. Handle Graceful Degradation

```python
# Provide fallbacks for disabled features
if lightrag_integration.QualityReportGenerator is not None:
    report = lightrag_integration.QualityReportGenerator().generate_report()
else:
    # Fallback to basic reporting
    report = {"status": "Quality reporting not available"}
```

### 4. Monitor Integration Health

```python
# Regular health checks in production
status = lightrag_integration.get_integration_status()
if status['integration_health'] != 'healthy':
    logger.warning(f"Integration degraded: {status.get('failed_required_modules', [])}")
```

## Troubleshooting

### Common Issues

1. **Module not loading**: Check `LIGHTRAG_INTEGRATION_ENABLED=true`
2. **Feature not available**: Verify corresponding feature flag is enabled
3. **Import errors**: Use safe checking patterns with `is not None`
4. **Unexpected behavior**: Check integration status and validation

### Debugging

```python
import lightrag_integration

# Debug feature flags
enabled = lightrag_integration.get_enabled_features()
print(f"Enabled features: {list(enabled.keys())}")

# Debug integration status
status = lightrag_integration.get_integration_status()
print(f"Integration health: {status['integration_health']}")

# Debug validation
is_valid, issues = lightrag_integration.validate_integration_setup()
if issues:
    print("Issues:", issues)
```

## Performance Considerations

- Feature flags are checked at module import time, not runtime
- Disabled features have minimal memory footprint (None stubs)
- Dynamic exports are built once during module initialization
- No runtime performance impact for checking enabled features

## Security Considerations

- Feature flags control which components are loaded, reducing attack surface
- Sensitive features can be disabled in production environments  
- API keys and credentials are masked in status reports
- Validation includes permission checks for directories and files