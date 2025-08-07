# Async Testing Configuration for Clinical Metabolomics Oracle LightRAG Integration

**Task**: CMO-LIGHTRAG-008-T01 - Set up pytest configuration for async testing  
**Created**: August 7, 2025  
**Author**: Claude Code (Anthropic)  

## Overview

This document describes the comprehensive async testing configuration implemented for the Clinical Metabolomics Oracle LightRAG integration project. The configuration supports fully async test functions, fixtures, and concurrent operations testing.

## Configuration Files

### 1. pytest.ini Configuration

**Location**: `lightrag_integration/tests/pytest.ini`

Key async-related configurations:
```ini
# Async testing configuration
--asyncio-mode=auto  # Enables automatic async test detection

# Async-specific markers
async: Async tests requiring event loop
lightrag: Tests specifically for LightRAG integration
biomedical: Tests for biomedical-specific functionality
```

### 2. conftest.py Async Fixtures

**Location**: `lightrag_integration/tests/conftest.py`

#### Async Fixtures Available:

- **`async_test_context`**: Provides async test context with proper setup/cleanup
- **`async_mock_lightrag`**: Mock LightRAG system for async integration testing
- **`async_cost_tracker`**: Async cost tracking system with thread-safe operations
- **`async_progress_monitor`**: Async progress monitoring with event tracking
- **`event_loop_policy`**: Session-scoped event loop policy configuration
- **`async_timeout`**: Configurable timeout for async operations (30s default)

## Usage Patterns

### 1. Basic Async Test

```python
import pytest
import asyncio

@pytest.mark.asyncio
@pytest.mark.unit
async def test_basic_async_functionality():
    """Test basic async functionality."""
    
    async def async_operation(result="success"):
        await asyncio.sleep(0.01)
        return result
    
    result = await async_operation()
    assert result == "success"
```

### 2. Using Async Fixtures

```python
@pytest.mark.asyncio
@pytest.mark.unit
async def test_with_async_fixtures(async_cost_tracker, async_progress_monitor):
    """Test using async fixtures."""
    
    # Track cost asynchronously
    record = await async_cost_tracker.track_cost("test_operation", 0.05)
    assert record['cost'] == 0.05
    
    # Update progress asynchronously
    await async_progress_monitor.update(50.0, "processing")
    summary = await async_progress_monitor.get_summary()
    assert summary['current_progress'] == 50.0
```

### 3. Concurrent Operations Testing

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_operations():
    """Test concurrent async operations."""
    
    async def async_task(delay, result):
        await asyncio.sleep(delay)
        return result
    
    # Run multiple tasks concurrently
    tasks = [async_task(0.01, f"result_{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result == f"result_{i}"
```

### 4. LightRAG Integration Testing

```python
@pytest.mark.asyncio
@pytest.mark.lightrag
@pytest.mark.biomedical
async def test_lightrag_integration(async_mock_lightrag, async_cost_tracker):
    """Test LightRAG integration with cost tracking."""
    
    # Insert biomedical document
    insert_result = await async_mock_lightrag.ainsert(
        "Metabolomic analysis of diabetes patients using LC-MS."
    )
    assert insert_result['status'] == 'success'
    
    # Track insertion cost
    await async_cost_tracker.track_cost(
        "document_insertion",
        insert_result['cost']
    )
    
    # Query the system
    query_result = await async_mock_lightrag.aquery(
        "What metabolites are associated with diabetes?",
        mode="hybrid"
    )
    assert "Mock response" in query_result
```

## Required Dependencies

The following packages are required for async testing:

```text
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-timeout>=2.1.0
```

Install with:
```bash
pip install -r lightrag_integration/tests/test_requirements.txt
```

## Running Async Tests

### 1. Run all async tests:
```bash
cd lightrag_integration/tests
source ../../lightrag_env/bin/activate
python -m pytest -m "async" -v
```

### 2. Run specific async test categories:
```bash
# LightRAG integration tests
python -m pytest -m "lightrag" -v

# Biomedical-specific tests
python -m pytest -m "biomedical" -v

# Unit async tests
python -m pytest -m "async and unit" -v
```

### 3. Run performance tests:
```bash
python -m pytest -m "performance" -v --durations=10
```

### 4. Run with coverage:
```bash
python -m pytest test_async_configuration.py --cov=lightrag_integration --cov-report=html
```

## Configuration Verification

To verify the async testing configuration is working correctly:

```bash
cd lightrag_integration/tests
source ../../lightrag_env/bin/activate
python -m pytest test_async_configuration.py -v
```

Expected output:
- All 9 tests should pass
- Tests should complete in under 1 second (concurrent execution)
- No async-related errors or warnings

## Async Fixture Architecture

### AsyncCostTracker
- Thread-safe cost tracking using `asyncio.Lock`
- Async methods: `track_cost()`, `get_total()`, `get_costs()`, `reset()`
- Automatic cleanup in fixture teardown

### AsyncProgressMonitor
- Progress tracking with event logging
- Async methods: `update()`, `get_summary()`, `wait_for_completion()`
- Time-based progress monitoring

### MockLightRAGSystem
- Comprehensive async LightRAG simulation
- Methods: `ainsert()`, `aquery()`, `adelete()`
- Realistic biomedical content generation
- Cost and performance simulation

## Performance Considerations

1. **Concurrent Execution**: Tests run concurrently where possible
2. **Timeout Management**: 30-second default timeout for async operations
3. **Resource Cleanup**: Automatic cleanup of async resources
4. **Memory Management**: Efficient handling of async generators and coroutines

## Best Practices

1. **Always use `@pytest.mark.asyncio`** for async test functions
2. **Use `@pytest_asyncio.fixture`** for async fixtures
3. **Handle exceptions properly** in async contexts
4. **Use `asyncio.gather()`** for concurrent operations
5. **Set appropriate timeouts** for long-running operations
6. **Clean up resources** in fixture teardown

## Troubleshooting

### Common Issues:

1. **"async def functions are not natively supported"**
   - Solution: Add `@pytest.mark.asyncio` decorator

2. **"TypeError: object NoneType can't be used in 'await' expression"**
   - Solution: Ensure async fixtures use `@pytest_asyncio.fixture`

3. **Tests timing out**
   - Solution: Check `async_timeout` fixture or increase timeout values

4. **Import errors with async fixtures**
   - Solution: Ensure `pytest_asyncio` is imported in conftest.py

## Integration with Existing Tests

The async configuration is backward compatible with existing tests. Non-async tests continue to work as before, while new async tests benefit from the enhanced configuration.

## Future Enhancements

Planned improvements for async testing:
- Async database testing utilities
- Real-time monitoring dashboard for test execution
- Advanced async debugging tools
- Performance profiling for async operations

---

**Verification Status**: ✅ Configuration verified and all tests passing  
**Compatibility**: Python 3.8+ with pytest 7.0+ and pytest-asyncio 0.21+  
**Last Updated**: August 7, 2025