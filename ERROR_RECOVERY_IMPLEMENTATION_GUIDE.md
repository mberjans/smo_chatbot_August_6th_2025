# PDF Processor Error Recovery System - Implementation Guide

## Overview

The BiomedicalPDFProcessor now includes a comprehensive error recovery system designed to handle various failure scenarios that can occur during PDF processing. This system implements:

- **Automatic retry mechanisms** with exponential backoff
- **Error classification** (recoverable vs non-recoverable)
- **Recovery strategies** tailored to specific error types
- **Memory recovery** through garbage collection
- **File lock handling** with progressive delays
- **Timeout recovery** with dynamic timeout adjustment
- **Comprehensive statistics** and logging

## Key Components

### 1. ErrorRecoveryConfig

Configuration class that defines retry parameters and recovery strategies:

```python
from lightrag_integration.pdf_processor import ErrorRecoveryConfig

# Default configuration
config = ErrorRecoveryConfig()
# max_retries=3, base_delay=1.0s, max_delay=60.0s

# Custom configuration for production environments
production_config = ErrorRecoveryConfig(
    max_retries=5,              # More aggressive retry attempts
    base_delay=2.0,             # Start with 2s delay
    max_delay=120.0,            # Cap delays at 2 minutes
    exponential_base=2.0,       # Standard exponential backoff
    jitter=True,                # Add randomization to delays
    memory_recovery_enabled=True,    # Enable memory cleanup
    file_lock_retry_enabled=True,    # Enable file lock retry
    timeout_retry_enabled=True       # Enable timeout recovery
)
```

### 2. Enhanced BiomedicalPDFProcessor

The processor now accepts an `ErrorRecoveryConfig` parameter:

```python
from lightrag_integration.pdf_processor import BiomedicalPDFProcessor, ErrorRecoveryConfig
import logging

# Setup logging
logger = logging.getLogger("pdf_processor")
logger.setLevel(logging.INFO)

# Create processor with error recovery
recovery_config = ErrorRecoveryConfig(max_retries=3, base_delay=1.5)
processor = BiomedicalPDFProcessor(
    logger=logger,
    processing_timeout=300,
    memory_limit_mb=2048,
    error_recovery_config=recovery_config
)
```

## Error Classification System

The system automatically classifies errors into categories and applies appropriate recovery strategies:

### Recoverable Errors

| Error Type | Category | Recovery Strategy | Description |
|------------|----------|-------------------|-------------|
| PDFMemoryError | memory | memory_cleanup | Runs garbage collection and clears caches |
| PDFProcessingTimeoutError | timeout | timeout_retry | Increases timeout for next attempt |
| PDFFileAccessError (locked) | file_lock | file_lock_retry | Waits with progressive delays |
| PDFValidationError (minor) | validation | simple_retry | Standard exponential backoff |
| IOError | io_error | simple_retry | Standard retry for I/O issues |
| PyMuPDF errors | fitz_error | Varies by type | Memory, timeout, or simple retry |

### Non-Recoverable Errors

| Error Type | Category | Action | Description |
|------------|----------|--------|-------------|
| PDFValidationError (corrupt) | corruption | skip | File is corrupted, skip processing |
| PDFContentError | content | skip | Content extraction impossible |
| PDFFileAccessError (permission) | permission | skip | Permission denied errors |
| Disk space errors | disk_space | skip | No space left on device |

## Usage Examples

### 1. Basic Usage with Error Recovery

```python
import asyncio
from pathlib import Path
from lightrag_integration.pdf_processor import BiomedicalPDFProcessor, ErrorRecoveryConfig

async def process_pdfs_with_recovery():
    # Configure error recovery
    recovery_config = ErrorRecoveryConfig(
        max_retries=3,
        base_delay=1.0,
        memory_recovery_enabled=True,
        file_lock_retry_enabled=True
    )
    
    # Create processor
    processor = BiomedicalPDFProcessor(error_recovery_config=recovery_config)
    
    # Process all PDFs in directory
    documents = await processor.process_all_pdfs("papers/")
    
    # Get error recovery statistics
    recovery_stats = processor.get_error_recovery_stats()
    print(f"Files requiring retries: {recovery_stats['files_with_retries']}")
    print(f"Total recovery actions: {recovery_stats['total_recovery_actions']}")
    
    return documents

# Run the processing
documents = asyncio.run(process_pdfs_with_recovery())
```

### 2. Single File Processing with Detailed Error Handling

```python
def process_single_pdf_with_recovery(pdf_path):
    recovery_config = ErrorRecoveryConfig(max_retries=5, base_delay=0.5)
    processor = BiomedicalPDFProcessor(error_recovery_config=recovery_config)
    
    try:
        result = processor.extract_text_from_pdf(pdf_path)
        
        # Check if retries were needed
        retry_info = result['processing_info'].get('retry_info', {})
        if retry_info.get('total_attempts', 1) > 1:
            print(f"File required {retry_info['total_attempts']} attempts")
            print(f"Recovery strategies used: {[action['strategy'] for action in retry_info['recovery_actions']]}")
        
        return result
        
    except Exception as e:
        # Get enhanced error information
        error_info = processor._get_enhanced_error_info(Path(pdf_path), e)
        print(f"Processing failed: {error_info}")
        raise
```

### 3. Production Configuration

```python
def create_production_processor():
    """Create a processor optimized for production environments."""
    
    # Production-grade error recovery configuration
    recovery_config = ErrorRecoveryConfig(
        max_retries=5,              # Allow more retries for reliability
        base_delay=2.0,             # Start with longer delays
        max_delay=300.0,            # Allow up to 5 minutes between retries
        exponential_base=1.8,       # Moderate backoff progression
        jitter=True,                # Randomize delays to avoid thundering herd
        memory_recovery_enabled=True,     # Essential for long-running processes
        file_lock_retry_enabled=True,     # Handle file locks gracefully
        timeout_retry_enabled=True        # Recover from temporary timeouts
    )
    
    processor = BiomedicalPDFProcessor(
        processing_timeout=600,     # 10 minutes for complex PDFs
        memory_limit_mb=4096,       # 4GB memory limit
        max_page_text_size=2000000, # 2MB per page
        error_recovery_config=recovery_config
    )
    
    return processor
```

### 4. Monitoring and Statistics

```python
async def process_with_monitoring():
    processor = BiomedicalPDFProcessor()
    
    # Reset statistics for fresh batch
    processor.reset_error_recovery_stats()
    
    # Process documents
    documents = await processor.process_all_pdfs("papers/")
    
    # Get comprehensive statistics
    processing_stats = processor.get_processing_stats()
    recovery_stats = processor.get_error_recovery_stats()
    
    print("Processing Statistics:")
    print(f"  Memory usage: {processing_stats['current_memory_mb']:.2f} MB")
    print(f"  System memory: {processing_stats['system_memory_percent']:.1f}%")
    
    print("Error Recovery Statistics:")
    print(f"  Files with retries: {recovery_stats['files_with_retries']}")
    print(f"  Total recovery actions: {recovery_stats['total_recovery_actions']}")
    print(f"  Recovery actions breakdown: {recovery_stats['recovery_actions_by_type']}")
    
    # Detailed file-by-file retry information
    for file_path, retry_info in recovery_stats['retry_details_by_file'].items():
        file_name = Path(file_path).name
        print(f"  {file_name}: {retry_info['total_attempts']} attempts, "
              f"{len(retry_info['recovery_actions'])} recovery actions")
```

## Recovery Strategies Detail

### 1. Memory Cleanup Recovery

When memory-related errors occur:
- Triggers Python garbage collection (`gc.collect()`)
- Clears internal caches
- Waits briefly for memory to be freed
- Logs memory usage before and after recovery

```python
# Memory recovery is automatic, but you can check if it's enabled
config = ErrorRecoveryConfig(memory_recovery_enabled=True)
processor = BiomedicalPDFProcessor(error_recovery_config=config)
```

### 2. File Lock Recovery

For file access errors due to locks:
- Uses progressive delays (2s, 4s, 8s, up to 30s max)
- Logs file lock status and wait times
- Automatically retries after delays

### 3. Timeout Recovery

When processing timeouts occur:
- Dynamically increases timeout for next attempt
- Uses multiplier of 1.5x per attempt (up to 3x original)
- Restores original timeout after processing

### 4. Simple Retry Recovery

For general transient errors:
- Uses exponential backoff with configurable parameters
- Optional jitter to prevent thundering herd problems
- Respects maximum delay limits

## Logging and Monitoring

The error recovery system provides comprehensive logging:

```
INFO - Attempting recovery for memory error using memory_cleanup strategy (attempt 2)
INFO - Memory recovery completed. Current memory usage: 1247.23 MB
WARNING - Attempt 2 failed for document.pdf: PDFMemoryError - Memory allocation failed...
INFO - Retry attempt 2 for document.pdf
INFO - Successfully processed 45 pages, extracted 125,432 characters
```

## Integration with Progress Tracking

The error recovery system integrates seamlessly with the existing progress tracking:

```python
from lightrag_integration.progress_config import ProgressTrackingConfig
from lightrag_integration.progress_tracker import PDFProcessingProgressTracker

# Progress tracking will automatically capture retry information
progress_config = ProgressTrackingConfig(log_detailed_errors=True)
documents = await processor.process_all_pdfs(
    "papers/",
    progress_config=progress_config
)
```

## Best Practices

### 1. Configuration Guidelines

- **Development**: Use lower retry counts and shorter delays for faster feedback
- **Production**: Use higher retry counts and longer delays for reliability
- **High-load systems**: Enable jitter to prevent synchronized retries

### 2. Memory Management

- Monitor memory usage regularly using `get_processing_stats()`
- Enable memory recovery for long-running batch processes
- Consider reducing `memory_limit_mb` if running in constrained environments

### 3. File Lock Handling

- Enable file lock retry when processing shared directories
- Use appropriate delays based on expected lock duration
- Log file lock issues for debugging shared access problems

### 4. Monitoring and Alerting

- Monitor `error_recovery_stats` for trends in error types
- Alert on high retry rates or specific error categories
- Use detailed error information for debugging problematic files

## Troubleshooting

### High Retry Rates

If you see many retries:
1. Check system resources (memory, disk space)
2. Verify file permissions and locks
3. Adjust timeout settings if processing complex PDFs
4. Review PDF quality - corrupted files should be identified and skipped

### Memory Issues

For persistent memory problems:
1. Reduce batch size in `process_all_pdfs`
2. Lower `memory_limit_mb` setting
3. Enable memory recovery
4. Monitor system memory pressure

### File Access Problems

For file lock or permission issues:
1. Verify file permissions
2. Check if files are open in other applications
3. Enable file lock retry
4. Use longer delays for network storage

This error recovery system makes the PDF processor much more robust and suitable for production environments where reliability is crucial.