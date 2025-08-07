# Advanced Recovery and Graceful Degradation System

## Overview

The Advanced Recovery and Graceful Degradation System provides sophisticated failure handling and recovery mechanisms for the Clinical Metabolomics Oracle's document ingestion pipeline. This system ensures robust, resilient processing even under adverse conditions through progressive degradation, intelligent retry strategies, resource monitoring, and checkpoint/resume capabilities.

## Key Features

### 1. Progressive Degradation Strategies
- **Automatic fallback processing modes** when errors occur
- **Quality/complexity reduction** to maintain throughput
- **Partial ingestion success** scenarios handling
- **Document prioritization** in degraded modes

### 2. Resource-Aware Recovery
- **System resource monitoring** (memory, disk, CPU)
- **Automatic batch size reduction** under resource constraints
- **Memory cleanup strategies** between batches
- **Dynamic resource allocation** based on system state

### 3. Intelligent Retry Backoff
- **Multiple backoff strategies** (exponential, linear, fibonacci, adaptive)
- **Failure pattern analysis** for backoff calculation
- **Rate limit-aware scheduling**
- **Circuit breaker integration** with automatic recovery

### 4. Checkpoint and Resume Capability
- **Progress persistence** for long-running processes
- **Batch-boundary checkpointing**
- **Skip processed documents** on resume
- **Recovery state management**

### 5. Graceful Degradation Modes
- **Optimal**: Full processing with all features
- **Essential**: Process only critical documents
- **Minimal**: Reduced complexity for speed
- **Offline**: Queue documents when APIs unavailable
- **Safe**: Ultra-conservative with maximum error tolerance

## Architecture

### Core Components

```
AdvancedRecoverySystem
├── SystemResourceMonitor    # Resource monitoring and pressure detection
├── AdaptiveBackoffCalculator # Intelligent retry timing
├── CheckpointManager        # State persistence and recovery
└── RecoveryIntegratedProcessor # Integration with existing systems
```

### Integration Points

- **ClinicalMetabolomicsRAG**: Main RAG system integration
- **UnifiedProgressTracker**: Progress tracking integration
- **Existing Error Handling**: Circuit breakers and rate limiters

## Usage Examples

### Basic Setup

```python
from lightrag_integration.advanced_recovery_system import AdvancedRecoverySystem, DegradationMode
from lightrag_integration.recovery_integration import RecoveryIntegratedProcessor

# Initialize recovery system
recovery_system = AdvancedRecoverySystem(
    resource_thresholds=ResourceThresholds(
        memory_warning_percent=75.0,
        memory_critical_percent=90.0
    ),
    checkpoint_dir=Path("./checkpoints")
)

# Initialize integrated processor
processor = RecoveryIntegratedProcessor(
    rag_system=your_rag_system,
    recovery_system=recovery_system,
    enable_checkpointing=True,
    checkpoint_interval=10  # Checkpoint every 10 documents
)
```

### Document Processing with Recovery

```python
# Process documents with full recovery support
results = await processor.process_documents_with_recovery(
    documents=document_list,
    phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
    initial_batch_size=5,
    max_failures_per_batch=3,
    progress_callback=your_callback_function
)

# Results include comprehensive statistics
print(f"Processed: {len(results['processed_documents'])}")
print(f"Failed: {len(results['failed_documents'])}")
print(f"Recovery events: {len(results['recovery_events'])}")
print(f"Final mode: {results['final_degradation_mode']}")
```

### Manual Failure Handling

```python
# Handle specific failure types
recovery_strategy = recovery_system.handle_failure(
    failure_type=FailureType.API_RATE_LIMIT,
    error_message="Rate limit exceeded",
    document_id="problematic_doc_001",
    context={"batch_size": 5, "attempt": 2}
)

# Apply recommended backoff
if recovery_strategy.get('backoff_seconds'):
    await asyncio.sleep(recovery_strategy['backoff_seconds'])
```

### Checkpoint and Resume

```python
# Create checkpoint during processing
checkpoint_id = recovery_system.create_checkpoint({
    'processing_phase': 'mid_batch',
    'custom_metadata': 'important_session'
})

# Resume from checkpoint after restart
recovery_system.resume_from_checkpoint(checkpoint_id)

# List available checkpoints
checkpoints = recovery_system.checkpoint_manager.list_checkpoints()
```

### Resource Monitoring

```python
# Check current system resources
resources = recovery_system.resource_monitor.get_current_resources()
print(f"Memory usage: {resources['memory_percent']:.1f}%")
print(f"Disk usage: {resources['disk_percent']:.1f}%")

# Check for resource pressure
pressure = recovery_system.resource_monitor.check_resource_pressure()
if pressure:
    print(f"Resource pressure detected: {pressure}")
```

## Configuration Options

### Degradation Configuration

```python
degradation_config = DegradationConfig(
    mode=DegradationMode.OPTIMAL,
    enable_partial_processing=True,
    reduce_batch_size=True,
    skip_optional_metadata=False,
    disable_advanced_chunking=False,
    enable_document_priority=True,
    max_retry_attempts=3,
    backoff_multiplier=2.0,
    max_backoff_seconds=300.0
)
```

### Resource Thresholds

```python
resource_thresholds = ResourceThresholds(
    memory_warning_percent=75.0,    # Warning at 75% memory
    memory_critical_percent=90.0,   # Critical at 90% memory
    disk_warning_percent=80.0,      # Warning at 80% disk
    disk_critical_percent=95.0,     # Critical at 95% disk
    cpu_warning_percent=80.0,       # Warning at 80% CPU
    cpu_critical_percent=95.0       # Critical at 95% CPU
)
```

## Degradation Modes Explained

### Optimal Mode (Default)
- Full processing with all features enabled
- Maximum quality and completeness
- Highest resource usage
- Best for normal operating conditions

### Essential Mode
- Processes only documents marked as essential/critical
- Skips non-critical documents to focus resources
- Moderate resource usage
- Good for high-failure scenarios

### Minimal Mode
- Reduces processing complexity for speed
- Disables advanced features like complex chunking
- Minimal resource usage
- Best for resource-constrained environments

### Offline Mode  
- Queues documents when APIs are unavailable
- Processes one document at a time
- Handles network connectivity issues
- Good for intermittent connectivity scenarios

### Safe Mode
- Ultra-conservative processing with maximum error tolerance
- Returns success even for minor errors
- Extensive validation and error checking
- Best for critical production environments

## Error Handling Flow

```
Document Processing
         ↓
    Error Occurs
         ↓
   Classify Error Type
    ↙    ↓    ↘
Rate Limit  Memory  Other
    ↓       ↓       ↓
  Backoff  Reduce  Retry
    ↓    Resources   ↓
         ↘  ↓  ↙
    Check Degradation Needed
              ↓
     Apply Degradation Mode
              ↓
      Update Configuration
              ↓
       Continue Processing
```

## Recovery Strategies by Failure Type

| Failure Type | Recovery Action | Degradation Trigger | Backoff Strategy |
|--------------|----------------|-------------------|------------------|
| API_RATE_LIMIT | Exponential backoff + reduce batch | After 3 failures | Adaptive with jitter |
| MEMORY_PRESSURE | Aggressive batch reduction | Immediate | Minimal backoff |
| API_TIMEOUT | Retry with longer timeout | After 5 failures | Exponential |
| NETWORK_ERROR | Switch to offline mode | After 3 failures | Fibonacci |
| PROCESSING_ERROR | Retry with safe mode | After 10 failures | Linear |
| DISK_SPACE | Cleanup + reduce batch | Immediate | None |

## Monitoring and Metrics

### Recovery Status

```python
status = recovery_system.get_recovery_status()
```

Returns comprehensive status including:
- Current degradation mode
- Batch size adjustments
- Document processing progress
- Error counts by type
- System resource usage
- Available checkpoints

### Processing Statistics

```python
# From processing results
statistics = results['statistics']
print(f"Success rate: {statistics['success_rate']:.2%}")
print(f"Processing speed: {statistics['documents_per_second']:.2f} docs/sec")
print(f"Recovery actions: {statistics['total_recovery_actions']}")
```

## Best Practices

### 1. Configuration
- Set appropriate resource thresholds for your environment
- Configure checkpoint intervals based on processing time
- Choose initial batch sizes based on document complexity

### 2. Error Handling
- Implement custom error classification for domain-specific errors
- Use context information to improve recovery decisions
- Monitor error patterns to tune degradation thresholds

### 3. Resource Management
- Monitor system resources regularly
- Implement cleanup strategies for temporary files
- Use batch size reduction proactively

### 4. Checkpointing
- Create checkpoints at natural boundaries (batch completion)
- Include custom metadata for recovery context
- Regularly cleanup old checkpoints

### 5. Testing
- Test failure scenarios in development
- Validate checkpoint/resume functionality
- Verify degradation mode behavior

## Integration with Existing Systems

### With ClinicalMetabolomicsRAG

```python
# Initialize RAG with recovery integration
rag_system = ClinicalMetabolomicsRAG(config=your_config)

processor = RecoveryIntegratedProcessor(
    rag_system=rag_system,
    recovery_system=recovery_system
)

# Processing automatically uses existing error handling
results = await processor.process_documents_with_recovery(documents)
```

### With Progress Tracking

```python
# Recovery system integrates with unified progress tracking
progress_tracker = KnowledgeBaseProgressTracker(progress_config)
recovery_system = AdvancedRecoverySystem(progress_tracker=progress_tracker)

# Progress updates include recovery events
progress_tracker.start_phase(KnowledgeBasePhase.DOCUMENT_INGESTION)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce batch size
   - Enable memory cleanup
   - Switch to minimal mode

2. **API Rate Limits**
   - Increase backoff multipliers
   - Reduce concurrent requests
   - Enable intelligent retry

3. **Disk Space Issues**
   - Enable automatic cleanup
   - Monitor temporary file usage
   - Reduce checkpoint frequency

4. **Processing Failures**
   - Check error classification
   - Verify degradation thresholds
   - Review document prioritization

### Debugging Tools

```python
# Enable detailed logging
import logging
logging.getLogger('lightrag_integration.advanced_recovery_system').setLevel(logging.DEBUG)

# Monitor recovery events
def recovery_callback(event):
    print(f"Recovery event: {event}")

# Check system health
status = recovery_system.get_recovery_status()
if not status['system_resources']:
    print("Resource monitoring unavailable")
```

## Performance Considerations

- **Resource Monitoring**: ~1% CPU overhead
- **Checkpointing**: Proportional to document count
- **Degradation Switching**: Minimal overhead
- **Backoff Calculation**: Negligible impact

## Future Enhancements

- Machine learning-based failure prediction
- Dynamic resource allocation
- Distributed processing support
- Advanced document prioritization
- Real-time dashboard integration

## API Reference

See the inline documentation in the source code for complete API details:
- `advanced_recovery_system.py`: Core recovery components
- `recovery_integration.py`: Integration layer
- `test_advanced_recovery.py`: Comprehensive test suite

## Examples and Demonstrations

Run the demonstration script to see the system in action:

```bash
python lightrag_integration/demo_advanced_recovery.py
```

This demonstrates:
- Progressive degradation scenarios
- Resource-aware recovery
- Intelligent backoff strategies
- Checkpoint and resume capabilities
- Integrated processing workflows