# Async Batch Processing Test Fixtures Guide

## Overview

This document provides a comprehensive guide to the async batch processing test fixtures implemented for the BiomedicalPDFProcessor. These fixtures enable thorough testing of batch PDF processing scenarios with various file types, sizes, and error conditions.

## Implementation Location

All fixtures are implemented in:
```
/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/tests/test_pdf_processor.py
```

## Key Components

### 1. Data Classes

#### `PerformanceMetrics`
```python
@dataclass
class PerformanceMetrics:
    start_time: float
    end_time: float
    peak_memory_mb: float
    total_files_processed: int
    successful_files: int
    failed_files: int
    average_processing_time_per_file: float
    
    @property
    def total_processing_time(self) -> float
    
    @property
    def success_rate(self) -> float
```

#### `MockPDFSpec`
```python
@dataclass
class MockPDFSpec:
    filename: str
    title: str
    page_count: int
    content_size: str  # 'small', 'medium', 'large'
    should_fail: bool = False
    failure_type: str = None  # 'validation', 'timeout', 'memory', 'access', 'content'
    processing_delay: float = 0.0  # Simulate processing time
```

### 2. Core Fixtures

#### `batch_test_environment`
**Purpose:** Creates a comprehensive test environment with multiple directories for different test scenarios.

**Returns:** Dict containing:
- `main_dir`: Main temporary directory
- `small_batch_dir`, `medium_batch_dir`, `large_batch_dir`: Size-specific directories
- `mixed_batch_dir`: Mixed success/failure scenarios
- `error_batch_dir`: Error testing scenarios
- `empty_dir`: Empty directory for edge case testing
- `real_pdf_dir`: Directory for real PDF copies
- `cleanup`: Function to clean up all directories

**Usage:**
```python
def test_example(batch_test_environment):
    env = batch_test_environment
    # Use env['small_batch_dir'] for testing
    # Cleanup is automatic via fixture teardown
```

#### `mock_pdf_generator`
**Purpose:** Generates mock PDF files based on specifications.

**Returns:** Function `generate_pdfs(directory: Path, specs: List[MockPDFSpec]) -> List[Path]`

**Usage:**
```python
def test_example(batch_test_environment, mock_pdf_generator, batch_test_data):
    env = batch_test_environment
    specs = batch_test_data['small_batch']
    created_files = mock_pdf_generator(env['small_batch_dir'], specs)
    # Files are automatically cleaned up
```

#### `batch_test_data`
**Purpose:** Provides pre-defined test data specifications for various scenarios.

**Returns:** Dict containing:
- `small_batch`: 3 small PDFs (1-3 pages, small content)
- `medium_batch`: 5 medium PDFs (4-8 pages, medium content)
- `large_batch`: 7 large PDFs (12-30 pages, large content)
- `mixed_success_failure`: Mix of successful and failing PDFs
- `all_failures`: All PDFs designed to fail with different error types
- `performance_stress`: 20 PDFs with processing delays for stress testing
- `diverse_sizes`: Range of PDF sizes from tiny to huge

#### `performance_monitor`
**Purpose:** Monitors performance metrics during batch processing tests.

**Returns:** PerformanceMonitor instance with methods:
- `start_monitoring()`: Begin performance tracking
- `update_peak_memory()`: Update peak memory usage
- `stop_monitoring(total_files, successful_files, failed_files)`: Stop and return metrics
- `get_current_memory_mb()`: Get current memory usage

**Usage:**
```python
def test_performance(performance_monitor):
    monitor = performance_monitor
    monitor.start_monitoring()
    
    # Run batch processing
    # ... test code ...
    
    metrics = monitor.stop_monitoring(total_files=5, successful_files=4, failed_files=1)
    assert metrics.success_rate == 0.8
```

#### `real_pdf_handler`
**Purpose:** Manages real PDF files for testing with actual documents.

**Returns:** RealPDFHandler instance with methods:
- `is_available()`: Check if real PDF is available
- `copy_to_directory(target_dir, new_name=None)`: Copy real PDF to directory
- `create_multiple_copies(target_dir, count)`: Create multiple copies
- `cleanup()`: Clean up all copied files

**Usage:**
```python
def test_real_pdfs(real_pdf_handler, batch_test_environment):
    handler = real_pdf_handler
    env = batch_test_environment
    
    if handler.is_available():
        copies = handler.create_multiple_copies(env['real_pdf_dir'], 3)
        # Test with real PDF copies
```

### 3. Specialized Fixtures

#### `corrupted_pdf_generator`
**Purpose:** Creates various types of corrupted PDF files for error testing.

**Corruption Types:**
- `truncated`: Valid PDF header but incomplete content
- `invalid_header`: Invalid PDF header
- `empty`: Completely empty file
- `binary_garbage`: Random binary data
- `incomplete_xref`: Incomplete cross-reference table

**Usage:**
```python
def test_error_handling(corrupted_pdf_generator, batch_test_environment):
    env = batch_test_environment
    corruption_types = ['truncated', 'empty', 'binary_garbage']
    corrupted_files = corrupted_pdf_generator(env['error_batch_dir'], corruption_types)
```

#### `mixed_file_generator`
**Purpose:** Creates mixed file types (PDFs and non-PDFs) for testing file filtering.

**Usage:**
```python
def test_file_filtering(mixed_file_generator, batch_test_environment):
    env = batch_test_environment
    file_specs = {'pdf': 3, 'txt': 2, 'doc': 1, 'image': 2}
    created_files = mixed_file_generator(env['mixed_batch_dir'], file_specs)
```

#### `async_mock_factory`
**Purpose:** Factory for creating async mocks compatible with batch processing tests.

**Returns:** Function `create_fitz_mock_side_effect(specs: List[MockPDFSpec]) -> callable`

**Usage:**
```python
def test_async_batch(async_mock_factory, batch_test_data):
    specs = batch_test_data['medium_batch']
    mock_side_effect = async_mock_factory(specs)
    
    with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz:
        mock_fitz.side_effect = mock_side_effect
        # Run async batch processing tests
```

#### `batch_processor_with_monitoring`
**Purpose:** Enhanced PDF processor with built-in monitoring capabilities.

**Returns:** MonitoringProcessor instance that tracks:
- Files processed
- Successful/failed extractions
- Total pages processed
- Processing time
- Memory snapshots

#### `async_test_helper`
**Purpose:** Utilities for async test execution and timing.

**Methods:**
- `run_with_timeout(coro, timeout_seconds=30.0)`: Run coroutine with timeout
- `measure_async_execution_time(coro)`: Measure execution time
- `create_async_context_manager()`: Create async context manager
- `parallel_execution_test(coroutines, max_concurrent=5)`: Controlled parallel execution

#### `directory_structure_validator`
**Purpose:** Validates directory structures and file organization.

**Methods:**
- `validate_batch_directory(directory, expected_pdf_count=None, should_contain_non_pdfs=False)`: Validate directory
- `get_directory_stats(directory)`: Get comprehensive directory statistics

## Usage Examples

### Basic Batch Processing Test

```python
def test_small_batch_processing(batch_test_environment, mock_pdf_generator, 
                               batch_test_data, async_mock_factory):
    env = batch_test_environment
    specs = batch_test_data['small_batch']
    
    # Create mock PDF files
    created_files = mock_pdf_generator(env['small_batch_dir'], specs)
    
    # Set up mocks
    mock_side_effect = async_mock_factory(specs)
    
    async def run_test():
        with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz:
            mock_fitz.side_effect = mock_side_effect
            
            processor = BiomedicalPDFProcessor()
            result = await processor.process_all_pdfs(env['small_batch_dir'])
            
            assert len(result) == 3
            for text, metadata in result:
                assert isinstance(text, str)
                assert isinstance(metadata, dict)
    
    asyncio.run(run_test())
```

### Performance Testing

```python
def test_performance_benchmark(batch_test_environment, mock_pdf_generator, 
                              batch_test_data, performance_monitor):
    env = batch_test_environment
    specs = batch_test_data['performance_stress']  # 20 PDFs with delays
    monitor = performance_monitor
    
    created_files = mock_pdf_generator(env['large_batch_dir'], specs)
    
    async def run_performance_test():
        monitor.start_monitoring()
        
        processor = BiomedicalPDFProcessor()
        result = await processor.process_all_pdfs(env['large_batch_dir'])
        
        metrics = monitor.stop_monitoring(
            total_files=len(specs),
            successful_files=len(result),
            failed_files=len(specs) - len(result)
        )
        
        # Performance assertions
        assert metrics.total_processing_time < 60.0  # Should complete within 60 seconds
        assert metrics.peak_memory_mb < 500.0  # Should not exceed 500MB
        assert metrics.success_rate >= 0.9  # 90% success rate minimum
    
    asyncio.run(run_performance_test())
```

### Error Handling Testing

```python
def test_comprehensive_error_handling(batch_test_environment, corrupted_pdf_generator,
                                    batch_test_data, async_mock_factory):
    env = batch_test_environment
    
    # Create corrupted files
    corruption_types = ['truncated', 'invalid_header', 'empty']
    corrupted_files = corrupted_pdf_generator(env['error_batch_dir'], corruption_types)
    
    # Mix with some successful files
    success_specs = batch_test_data['small_batch'][:2]  # 2 successful files
    mock_pdf_generator(env['error_batch_dir'], success_specs)
    
    async def run_error_test():
        processor = BiomedicalPDFProcessor()
        result = await processor.process_all_pdfs(env['error_batch_dir'])
        
        # Should only have successful results
        assert len(result) == 2  # Only 2 successful files
    
    asyncio.run(run_error_test())
```

### Real PDF Testing

```python
def test_with_real_pdfs(real_pdf_handler, batch_test_environment):
    handler = real_pdf_handler
    env = batch_test_environment
    
    if not handler.is_available():
        pytest.skip("Real PDF not available for testing")
    
    # Create multiple copies of real PDF
    copies = handler.create_multiple_copies(env['real_pdf_dir'], 3)
    
    async def run_real_pdf_test():
        processor = BiomedicalPDFProcessor()
        result = await processor.process_all_pdfs(env['real_pdf_dir'])
        
        assert len(result) == 3
        for text, metadata in result:
            assert len(text) > 0
            assert 'Clinical_Metabolomics_paper.pdf' in metadata['filename'] or \
                   'real_pdf_copy_' in metadata['filename']
    
    asyncio.run(run_real_pdf_test())
```

## Test Data Specifications

### Available Test Scenarios

1. **Small Batch** (3 PDFs): 1-3 pages, minimal content
2. **Medium Batch** (5 PDFs): 4-8 pages, moderate content  
3. **Large Batch** (7 PDFs): 12-30 pages, extensive content
4. **Mixed Success/Failure** (6 PDFs): 3 successful, 3 different failure types
5. **All Failures** (5 PDFs): All designed to fail with different error types
6. **Performance Stress** (20 PDFs): With processing delays for stress testing
7. **Diverse Sizes** (5 PDFs): Range from tiny (1 page) to huge (50 pages)

### Content Generation

The `AsyncBatchTestFixtures.create_biomedical_content()` method generates realistic biomedical content:

- **Small**: ~1KB of clinical metabolomics text
- **Medium**: ~5KB with repeated sections and variations
- **Large**: ~20KB with extensive sections and patient data variations

## Best Practices

1. **Always use fixtures for test isolation** - Each test gets clean directories and proper cleanup
2. **Use appropriate test data sizes** - Match test scenario complexity to test requirements
3. **Monitor performance** - Use `performance_monitor` for any test that processes multiple files
4. **Test error conditions** - Use `corrupted_pdf_generator` and failure specs to test error handling
5. **Validate directory states** - Use `directory_structure_validator` to verify test setup
6. **Clean async patterns** - Use `async_test_helper` for consistent async test execution

## Integration with Existing Tests

These fixtures are designed to work alongside the existing test patterns in `test_pdf_processor.py`. They follow the same naming conventions and cleanup patterns, ensuring consistency across the test suite.

The fixtures support all the test cases designed for:
- Basic batch processing functionality
- Performance benchmarking
- Error handling scenarios
- Real PDF integration
- Memory management testing
- Async execution timing
- File filtering and discovery

## Dependencies

Required packages for these fixtures:
- `pytest`
- `asyncio`
- `tempfile`
- `pathlib`
- `psutil` (for memory monitoring)
- `shutil` (for file operations)
- `dataclasses`
- `typing`
- `random`
- `time`

All fixtures include proper error handling and graceful degradation when optional dependencies are not available.