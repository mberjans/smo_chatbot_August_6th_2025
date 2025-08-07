# Test Data Fixtures Usage Guide

## Overview

The Test Data Fixtures system provides comprehensive pytest fixtures and utilities that integrate with the existing test infrastructure while utilizing the structured `test_data/` directory. This system bridges the gap between established testing patterns and modern test data management.

## Key Features

- **Seamless Integration**: Works with existing `conftest.py` and fixture infrastructure
- **Automatic Cleanup**: Proper teardown mechanisms for all test resources
- **Async Support**: Full async compatibility for LightRAG integration testing
- **Performance Optimized**: Efficient data loading and memory management
- **Comprehensive Validation**: Built-in data integrity and structure validation
- **Error Resilient**: Robust error handling and recovery mechanisms

## Quick Start

### Basic Usage

```python
import pytest
from tests.test_data_fixtures import (
    test_data_manager,
    sample_metabolomics_study,
    mock_metabolites_data,
    test_temp_dir
)

def test_basic_functionality(
    test_data_manager,
    sample_metabolomics_study: str,
    test_temp_dir: Path
):
    """Example of basic fixture usage."""
    # Sample data is automatically loaded
    assert "metabolomics" in sample_metabolomics_study.lower()
    
    # Temporary directory is ready for use
    test_file = test_temp_dir / "output.txt"
    test_file.write_text("test data")
    
    # Cleanup happens automatically when test completes
```

### Async Usage

```python
import pytest
from tests.test_data_integration import async_biomedical_data

@pytest.mark.asyncio
async def test_async_functionality(async_biomedical_data):
    """Example of async fixture usage."""
    # Data is loaded asynchronously
    compounds = async_biomedical_data["compounds"]
    studies = async_biomedical_data["studies"]
    
    # Process data
    assert len(compounds) > 0
    assert len(studies) > 0
```

## Available Fixtures

### Core Fixtures

#### `test_data_manager: TestDataManager`
- **Scope**: Function-level (default) or session-level
- **Purpose**: Central coordinator for test data operations
- **Features**: Automatic cleanup, resource tracking, lifecycle management

```python
def test_with_manager(test_data_manager):
    # Register temporary resources for cleanup
    temp_dir = Path(tempfile.mkdtemp())
    test_data_manager.register_temp_dir(temp_dir)
    
    # Resources are automatically cleaned up
```

#### `integrated_test_data_manager: TestDataManager`
- **Scope**: Function-level
- **Purpose**: Enhanced manager with integration capabilities
- **Features**: Legacy compatibility, performance optimization

### PDF Data Fixtures

#### `sample_metabolomics_study: str`
- **Source**: `test_data/pdfs/samples/sample_metabolomics_study.txt`
- **Content**: Realistic metabolomics research paper content
- **Usage**: Testing PDF processing, content analysis

#### `sample_clinical_trial: str`
- **Source**: `test_data/pdfs/samples/sample_clinical_trial.txt`
- **Content**: Clinical trial documentation
- **Usage**: Clinical research workflow testing

#### `pdf_test_files: Dict[str, str]`
- **Source**: All files in `test_data/pdfs/samples/`
- **Content**: Dictionary mapping filenames to content
- **Usage**: Batch PDF processing tests

#### `enhanced_pdf_data: Dict[str, Any]`
- **Source**: Generated using TestDataFactory
- **Content**: Enhanced PDF data with metadata and validation
- **Features**: Checksums, metadata, size tracking

```python
def test_pdf_processing(enhanced_pdf_data):
    for study_name, pdf_data in enhanced_pdf_data.items():
        content = pdf_data["content"]
        metadata = pdf_data["metadata"]
        checksum = pdf_data["checksum"]
        
        # Process PDF with metadata
        assert metadata["sample_size"] > 0
        assert len(content) > 100
```

### Database Fixtures

#### `test_cost_db: sqlite3.Connection`
- **Source**: Schema from `test_data/databases/schemas/cost_tracking_schema.sql`
- **Purpose**: Testing cost monitoring functionality
- **Features**: Automatic schema loading, cleanup on test completion

#### `test_knowledge_db: sqlite3.Connection`
- **Source**: Schema from `test_data/databases/schemas/knowledge_base_schema.sql`
- **Purpose**: Testing LightRAG knowledge base operations
- **Features**: Isolated test database, automatic cleanup

```python
def test_database_operations(test_cost_db):
    cursor = test_cost_db.cursor()
    
    # Database is ready with schema loaded
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    assert len(tables) > 0
```

### Mock Data Fixtures

#### `mock_metabolites_data: Dict[str, Any]`
- **Source**: `test_data/mocks/biomedical_data/mock_metabolites.json`
- **Content**: Comprehensive metabolite database with realistic data
- **Usage**: Testing biomedical data processing

#### `mock_openai_responses: Dict[str, Any]`
- **Source**: `test_data/mocks/api_responses/openai_api_responses.json`
- **Content**: Mock API responses for different scenarios
- **Usage**: Testing API integration without actual API calls

#### `comprehensive_mock_data: Dict[str, Any]`
- **Source**: Generated using MockDataGenerator
- **Content**: Complete mock data for all test scenarios
- **Categories**: API responses, system states, performance data

```python
def test_api_integration(comprehensive_mock_data):
    api_responses = comprehensive_mock_data["api_responses"]
    success_response = api_responses["success"]
    
    # Test success scenario
    assert success_response["status"] == "success"
    
    # Test error scenario
    failure_response = api_responses["failure"]
    assert failure_response["status"] == "error"
```

### Temporary Directory Fixtures

#### `test_temp_dir: Path`
- **Purpose**: General-purpose temporary directory
- **Features**: Automatic cleanup, unique per test

#### `test_staging_dir: Path`
- **Purpose**: Staging area for test data preparation
- **Location**: `test_data/temp/staging/`
- **Features**: Organized temporary workspace

#### `test_processing_dir: Path`
- **Purpose**: Processing workspace for test operations
- **Location**: `test_data/temp/processing/`
- **Features**: Isolated processing environment

#### `test_output_dir: Path`
- **Purpose**: Output directory for test results
- **Features**: Automatic cleanup, result isolation

### Async Fixtures

#### `async_test_data_manager: AsyncTestDataManager`
- **Purpose**: Async-compatible test data operations
- **Features**: Async data loading, concurrent operations support

#### `async_biomedical_data: Dict[str, Any]`
- **Purpose**: Asynchronously loaded biomedical test data
- **Content**: Compounds, studies, mock responses
- **Features**: Concurrent loading, comprehensive datasets

```python
@pytest.mark.asyncio
async def test_async_operations(async_test_data_manager):
    # Load data asynchronously
    async def load_test_data():
        return {"test": "data"}
    
    result = await async_test_data_manager.load_test_data_async(
        "test_type", "test_key", load_test_data
    )
    assert result["test"] == "data"
```

## Utility Classes

### TestDataFactory

Generates realistic test data for various scenarios.

```python
from tests.test_data_utilities import TestDataFactory

def test_data_generation():
    factory = TestDataFactory(seed=42)  # Reproducible results
    
    # Generate biochemical compound
    compound = factory.generate_compound()
    assert compound.molecular_weight > 0
    
    # Generate clinical study
    study = factory.generate_clinical_study(condition="Type 2 Diabetes")
    assert study.sample_size > 0
    
    # Generate compound database
    db = factory.generate_compound_database(count=10)
    assert len(db["metabolite_database"]["metabolites"]) == 10
```

### DataValidationSuite

Validates test data integrity and structure.

```python
from tests.test_data_utilities import DataValidationSuite

def test_data_validation():
    validator = DataValidationSuite()
    
    # Validate metabolite data
    metabolite = {
        "id": "met_001",
        "name": "glucose",
        "formula": "C6H12O6",
        "molecular_weight": 180.16
    }
    
    is_valid = validator.validate_metabolite_data(metabolite)
    assert is_valid
    
    # Get validation report
    report = validator.get_validation_report()
    assert report["success_rate"] > 0
```

### MockDataGenerator

Creates dynamic mock data for complex testing scenarios.

```python
from tests.test_data_utilities import MockDataGenerator

def test_mock_generation():
    generator = MockDataGenerator()
    
    # Generate API response mock
    response = generator.generate_api_response_mock(
        "openai_chat", success=True
    )
    assert response["status"] == "success"
    
    # Generate system state mock
    state = generator.generate_system_state_mock(
        "cost_monitor", healthy=True
    )
    assert state["status"] == "healthy"
```

## Advanced Usage Patterns

### Complex Test Scenarios

```python
def test_end_to_end_workflow(
    integrated_test_data_manager,
    enhanced_pdf_data,
    test_cost_db,
    test_processing_dir
):
    """Complex workflow testing with multiple fixtures."""
    
    # Process PDF data
    for study_name, pdf_data in enhanced_pdf_data.items():
        content = pdf_data["content"]
        
        # Save to processing directory
        pdf_file = test_processing_dir / f"{study_name}.txt"
        pdf_file.write_text(content)
        
        # Track in database
        cursor = test_cost_db.cursor()
        cursor.execute(
            "INSERT INTO processing_log (file_name, size_bytes) VALUES (?, ?)",
            (study_name, pdf_data["size_bytes"])
        )
        test_cost_db.commit()
    
    # Verify processing
    cursor.execute("SELECT COUNT(*) FROM processing_log")
    count = cursor.fetchone()[0]
    assert count == len(enhanced_pdf_data)
```

### Performance Testing

```python
@pytest.mark.performance
def test_performance_with_optimization(performance_optimizer):
    """Performance testing with optimization tracking."""
    
    def sample_operation():
        time.sleep(0.01)  # Simulate work
        return "result"
    
    # Profile the operation
    profiled_op = performance_optimizer.profile_data_loading(
        "sample_op", sample_operation
    )
    
    result = profiled_op()
    assert result == "result"
    
    # Get performance report
    report = performance_optimizer.get_performance_report()
    assert "sample_op" in [item[0] for item in report["slowest_loads"]]
```

### Error Handling Testing

```python
def test_error_scenarios(comprehensive_mock_data):
    """Test error handling with mock failure scenarios."""
    
    system_states = comprehensive_mock_data["system_states"]
    
    # Test budget exceeded scenario
    exceeded_state = system_states["cost_monitor_exceeded"]
    assert exceeded_state["status"] == "budget_exceeded"
    assert exceeded_state["utilization_percent"] > 100
    
    # Test degraded system scenario
    degraded_state = system_states["lightrag_degraded"]
    assert degraded_state["status"] == "degraded"
    assert "error_message" in degraded_state
```

## Migration from Existing Tests

### Step 1: Identify Current Patterns

```python
# Old pattern
def test_old_way():
    temp_dir = tempfile.mkdtemp()
    try:
        # Test logic here
        pass
    finally:
        shutil.rmtree(temp_dir)

# New pattern
def test_new_way(test_temp_dir):
    # Test logic here - cleanup is automatic
    pass
```

### Step 2: Use Migration Helper

```python
from tests.test_data_integration import migrate_existing_fixture

@migrate_existing_fixture("old_fixture_name", test_data_manager)
def old_fixture():
    return {"data": "value"}

def test_migrated_fixture(old_fixture):
    assert old_fixture["data"] == "value"
    assert old_fixture["_migrated"] is True
```

### Step 3: Update Test Dependencies

```python
# Before
def test_function():
    pdf_content = open("sample_file.txt").read()
    # Process content

# After  
def test_function(sample_metabolomics_study):
    pdf_content = sample_metabolomics_study
    # Process content - automatic loading and validation
```

## Best Practices

### 1. Choose Appropriate Fixture Scope

- **Function**: For unique test data (`test_temp_dir`)
- **Session**: For read-only shared data (`mock_metabolites_data`)
- **Module**: For test suite specific data

### 2. Use Async Fixtures for I/O Operations

```python
# Good for I/O heavy operations
@pytest.mark.asyncio
async def test_heavy_io(async_biomedical_data):
    # Data loaded asynchronously
    pass

# Good for simple operations
def test_simple_operation(sample_metabolomics_study):
    # Quick synchronous access
    pass
```

### 3. Leverage Integration Features

```python
def test_with_integration(integrated_test_data_manager):
    # Automatic integration with existing fixtures
    # Performance monitoring available
    # Legacy compatibility enabled
    pass
```

### 4. Validate Test Data

```python
def test_with_validation(mock_metabolites_data):
    from tests.test_data_utilities import DataValidationSuite
    
    validator = DataValidationSuite()
    
    # Validate before using
    for metabolite in mock_metabolites_data["metabolite_database"]["metabolites"]:
        is_valid = validator.validate_metabolite_data(metabolite)
        assert is_valid
```

### 5. Use Performance Monitoring

```python
@pytest.mark.performance
def test_performance_critical_operation(performance_optimizer):
    # Performance monitoring automatically enabled
    # Results available in test reports
    pass
```

## Troubleshooting

### Common Issues

#### 1. Fixture Not Found

```
E   fixture 'test_data_manager' not found
```

**Solution**: Import fixtures in test file or ensure `conftest.py` includes the fixtures.

```python
# Add to test file
from tests.test_data_fixtures import test_data_manager
```

#### 2. Async Fixture Issues

```
E   TypeError: object NoneType can't be used in 'await' expression
```

**Solution**: Use `pytest_asyncio.fixture` and mark test with `@pytest.mark.asyncio`.

```python
@pytest.mark.asyncio
async def test_async(async_test_data_manager):
    # Proper async test
    pass
```

#### 3. Test Data Not Found

```
E   pytest.skip.Exception: Sample metabolomics study not found
```

**Solution**: Ensure test data files exist in `test_data/` directory structure.

```bash
# Check test data structure
ls -la tests/test_data/pdfs/samples/
```

#### 4. Database Schema Issues

```
E   sqlite3.OperationalError: no such table: test_table
```

**Solution**: Verify schema files exist and contain valid SQL.

```bash
# Check schema files
ls -la tests/test_data/databases/schemas/
cat tests/test_data/databases/schemas/cost_tracking_schema.sql
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_debug_fixture(test_data_manager):
    # Debug information will be logged
    pass
```

### Performance Issues

Monitor fixture performance:

```python
def test_slow_fixture(performance_optimizer, test_data_manager):
    report = performance_optimizer.get_performance_report()
    print(f"Performance report: {report}")
```

## Integration Examples

### Example 1: Complete PDF Processing Test

```python
def test_complete_pdf_processing(
    enhanced_pdf_data,
    test_cost_db,
    test_processing_dir,
    integrated_test_data_manager
):
    """Complete PDF processing workflow test."""
    
    total_processed = 0
    total_cost = 0.0
    
    for study_name, pdf_data in enhanced_pdf_data.items():
        # Process PDF content
        content = pdf_data["content"]
        metadata = pdf_data["metadata"]
        
        # Save processed content
        output_file = test_processing_dir / f"processed_{study_name}.txt"
        processed_content = content.upper()  # Simple processing
        output_file.write_text(processed_content)
        
        # Track processing cost
        processing_cost = len(content) * 0.001  # Mock cost calculation
        total_cost += processing_cost
        
        # Log to database
        cursor = test_cost_db.cursor()
        cursor.execute("""
            INSERT INTO processing_log 
            (study_id, file_name, size_bytes, processing_cost) 
            VALUES (?, ?, ?, ?)
        """, (
            metadata["study_id"],
            study_name,
            pdf_data["size_bytes"],
            processing_cost
        ))
        
        total_processed += 1
    
    test_cost_db.commit()
    
    # Verify results
    assert total_processed > 0
    assert total_cost > 0
    
    # Verify database records
    cursor = test_cost_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM processing_log")
    db_count = cursor.fetchone()[0]
    assert db_count == total_processed
```

### Example 2: Async API Integration Test

```python
@pytest.mark.asyncio
async def test_async_api_integration(
    async_biomedical_data,
    comprehensive_mock_data
):
    """Async API integration test with mock responses."""
    
    compounds = async_biomedical_data["compounds"]
    mock_responses = comprehensive_mock_data["api_responses"]
    
    # Simulate API calls for each compound
    results = []
    
    for compound in compounds[:3]:  # Test with first 3 compounds
        # Mock API call
        if "glucose" in compound["name"].lower():
            response = mock_responses["success"]
        else:
            response = mock_responses["failure"]
        
        # Process response
        await asyncio.sleep(0.01)  # Simulate API delay
        
        if response["status"] == "success":
            results.append({
                "compound_id": compound["id"],
                "api_result": "success",
                "data": response["data"]
            })
        else:
            results.append({
                "compound_id": compound["id"],
                "api_result": "error",
                "error": response["error"]
            })
    
    # Verify results
    assert len(results) == 3
    success_count = sum(1 for r in results if r["api_result"] == "success")
    error_count = sum(1 for r in results if r["api_result"] == "error")
    
    assert success_count + error_count == len(results)
```

This comprehensive guide covers all aspects of using the new test data fixtures system while maintaining compatibility with existing test infrastructure. The fixtures provide a robust foundation for testing the Clinical Metabolomics Oracle LightRAG integration.