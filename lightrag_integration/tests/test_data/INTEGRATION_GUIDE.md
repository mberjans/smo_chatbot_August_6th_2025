# Test Data Integration Guide

This guide explains how to integrate the test data management system with existing test infrastructure in the Clinical Metabolomics Oracle LightRAG integration project.

## Overview

The test data management system provides:
- Structured test data for consistent testing
- Utilities for data lifecycle management
- Integration points with existing test frameworks
- Support for cleanup and maintenance operations

## Integration Points

### 1. Pytest Integration

#### Fixture Integration

```python
# In your test files or conftest.py
import pytest
from pathlib import Path
import sqlite3
import json

# Test data paths
TEST_DATA_PATH = Path(__file__).parent / "test_data"

@pytest.fixture
def sample_pdf_content():
    """Provide sample PDF content for testing"""
    with open(TEST_DATA_PATH / "pdfs" / "samples" / "sample_metabolomics_study.txt") as f:
        return f.read()

@pytest.fixture
def corrupted_pdf_content():
    """Provide corrupted PDF content for error testing"""
    with open(TEST_DATA_PATH / "pdfs" / "corrupted" / "corrupted_sample.txt") as f:
        return f.read()

@pytest.fixture
def test_database():
    """Provide test database connection"""
    db_path = TEST_DATA_PATH / "databases" / "samples" / "test_cost_tracking.db"
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()

@pytest.fixture
def mock_metabolites():
    """Provide mock metabolite data"""
    with open(TEST_DATA_PATH / "mocks" / "biomedical_data" / "mock_metabolites.json") as f:
        return json.load(f)

@pytest.fixture
def mock_api_responses():
    """Provide mock API responses"""
    with open(TEST_DATA_PATH / "mocks" / "api_responses" / "openai_api_responses.json") as f:
        return json.load(f)
```

#### Test Data Validation Fixture

```python
@pytest.fixture(scope="session")
def validate_test_data():
    """Validate test data before running tests"""
    import subprocess
    result = subprocess.run([
        "python", str(TEST_DATA_PATH / "utilities" / "validators" / "test_data_validator.py"),
        "--test-data-path", str(TEST_DATA_PATH),
        "--quiet"
    ])
    assert result.returncode == 0, "Test data validation failed"
```

#### Cleanup Fixtures

```python
@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Cleanup test data after test session"""
    yield
    import subprocess
    subprocess.run([
        "python", str(TEST_DATA_PATH / "utilities" / "cleanup_scripts" / "cleanup_test_data.py"),
        "--mode", "temp_only",
        "--base-path", str(TEST_DATA_PATH)
    ])

@pytest.fixture
def temp_workspace():
    """Provide clean temporary workspace for tests"""
    temp_dir = TEST_DATA_PATH / "temp" / "staging"
    temp_dir.mkdir(exist_ok=True)
    
    yield temp_dir
    
    # Cleanup after test
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)
```

### 2. Existing Test Integration

#### PDF Processing Tests

```python
# Integration with existing PDF processing tests
def test_pdf_processing_with_sample_data(sample_pdf_content):
    """Test PDF processing using sample data"""
    from lightrag_integration.pdf_processor import PDFProcessor
    
    processor = PDFProcessor()
    result = processor.process_content(sample_pdf_content)
    
    assert result is not None
    assert "metabolomics" in result.lower()

def test_pdf_error_handling_with_corrupted_data(corrupted_pdf_content):
    """Test error handling with corrupted data"""
    from lightrag_integration.pdf_processor import PDFProcessor
    
    processor = PDFProcessor()
    
    # Should handle corruption gracefully
    try:
        result = processor.process_content(corrupted_pdf_content)
        # Verify error recovery
        assert result is not None or processor.has_errors()
    except Exception as e:
        # Verify expected error types
        assert "corruption" in str(e).lower() or "malformed" in str(e).lower()
```

#### Database Tests

```python
def test_cost_tracking_with_test_db(test_database):
    """Test cost tracking using test database"""
    from lightrag_integration.cost_persistence import CostTracker
    
    tracker = CostTracker(database_connection=test_database)
    
    # Test with existing test data
    total_cost = tracker.get_total_cost()
    assert total_cost > 0  # Test data includes sample costs
    
    # Test adding new entries
    tracker.log_cost("test_operation", 0.50)
    new_total = tracker.get_total_cost()
    assert new_total > total_cost
```

#### Mock API Integration

```python
def test_openai_api_with_mock_responses(mock_api_responses, monkeypatch):
    """Test OpenAI API integration with mock responses"""
    
    class MockOpenAI:
        def __init__(self):
            self.responses = mock_api_responses
            
        def create_embedding(self, **kwargs):
            return self.responses["embedding_response_success"]
            
        def create_chat_completion(self, **kwargs):
            return self.responses["chat_completion_success"]
    
    monkeypatch.setattr("openai.OpenAI", MockOpenAI)
    
    # Test with mocked API
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    
    rag = ClinicalMetabolomicsRAG()
    result = rag.query("What are diabetes biomarkers?")
    
    assert "biomarkers" in result.lower()
    assert "diabetes" in result.lower()
```

### 3. Performance Test Integration

#### Benchmark Data Generation

```python
def test_performance_with_generated_data():
    """Test performance using generated test documents"""
    import subprocess
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate test documents
        subprocess.run([
            "python", str(TEST_DATA_PATH / "utilities" / "data_generators" / "generate_test_pdfs.py"),
            "--count", "50",
            "--output-dir", temp_dir
        ])
        
        # Run performance tests with generated data
        from lightrag_integration.pdf_processor import PDFProcessor
        import time
        
        processor = PDFProcessor()
        start_time = time.time()
        
        for pdf_file in Path(temp_dir).glob("*.txt"):
            with open(pdf_file) as f:
                processor.process_content(f.read())
        
        duration = time.time() - start_time
        avg_time = duration / 50
        
        assert avg_time < 5.0  # Should process each document in < 5 seconds
```

### 4. Error Handling Test Integration

#### Systematic Error Testing

```python
@pytest.mark.parametrize("corruption_type", [
    "incomplete_sections",
    "encoding_issues",
    "malformed_structure"
])
def test_error_handling_scenarios(corruption_type):
    """Test various error scenarios using test data"""
    from lightrag_integration.pdf_processor import PDFProcessor
    
    # Generate specific corruption type
    subprocess.run([
        "python", str(TEST_DATA_PATH / "utilities" / "data_generators" / "generate_test_pdfs.py"),
        "--count", "1",
        "--corruption-type", corruption_type,
        "--output-dir", str(TEST_DATA_PATH / "temp" / "staging")
    ])
    
    # Test error handling
    processor = PDFProcessor()
    corrupted_files = list((TEST_DATA_PATH / "temp" / "staging").glob("*.txt"))
    
    for file in corrupted_files:
        with open(file) as f:
            content = f.read()
            
        try:
            result = processor.process_content(content)
            # Verify graceful degradation
            assert result is not None or processor.has_warnings()
        except Exception as e:
            # Verify expected error handling
            assert hasattr(processor, 'error_recovery')
```

### 5. Configuration Integration

#### Test Configuration Override

```python
# test_config.py
import os
from pathlib import Path

TEST_DATA_PATH = Path(__file__).parent / "test_data"

# Override configuration for tests
TEST_CONFIG = {
    "database_path": str(TEST_DATA_PATH / "databases" / "samples" / "test_cost_tracking.db"),
    "log_config_path": str(TEST_DATA_PATH / "logs" / "configs" / "logging_config_template.json"),
    "mock_api_responses": True,
    "temp_directory": str(TEST_DATA_PATH / "temp" / "staging"),
    "cleanup_age_hours": 1,  # Aggressive cleanup for tests
}

@pytest.fixture(autouse=True)
def use_test_config(monkeypatch):
    """Apply test configuration"""
    for key, value in TEST_CONFIG.items():
        monkeypatch.setenv(f"LIGHTRAG_TEST_{key.upper()}", str(value))
```

## Continuous Integration Integration

### GitHub Actions Example

```yaml
name: Test with Data Management

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r lightrag_integration/tests/test_requirements.txt
        
    - name: Validate test data
      run: |
        cd lightrag_integration/tests/test_data
        python utilities/validators/test_data_validator.py --quiet
        
    - name: Generate additional test data
      run: |
        cd lightrag_integration/tests/test_data
        python utilities/data_generators/generate_test_pdfs.py --count 10 --output-dir temp/staging
        
    - name: Run tests
      run: |
        cd lightrag_integration/tests
        pytest --verbose --tb=short
        
    - name: Cleanup test data
      if: always()
      run: |
        cd lightrag_integration/tests/test_data
        python utilities/cleanup_scripts/cleanup_test_data.py --mode temp_only
        
    - name: Upload test reports
      if: failure()
      uses: actions/upload-artifact@v2
      with:
        name: test-reports
        path: lightrag_integration/tests/test_data/reports/
```

## Best Practices

### 1. Test Isolation

```python
@pytest.fixture
def isolated_test_environment():
    """Create isolated environment for each test"""
    import tempfile
    import shutil
    
    # Create temporary test data copy
    temp_dir = tempfile.mkdtemp()
    test_data_copy = Path(temp_dir) / "test_data"
    shutil.copytree(TEST_DATA_PATH, test_data_copy)
    
    yield test_data_copy
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### 2. Data Versioning

```python
def test_data_version_compatibility():
    """Ensure test data is compatible with current code version"""
    version_file = TEST_DATA_PATH / "VERSION"
    
    if version_file.exists():
        with open(version_file) as f:
            data_version = f.read().strip()
            
        # Compare with code version
        from lightrag_integration import __version__
        assert data_version.split('.')[0] == __version__.split('.')[0], \
            f"Test data version {data_version} incompatible with code version {__version__}"
```

### 3. Resource Management

```python
class TestDataManager:
    """Context manager for test data resources"""
    
    def __init__(self, cleanup_mode="temp_only"):
        self.cleanup_mode = cleanup_mode
        self.resources = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def get_sample_data(self, data_type):
        """Get sample data with automatic cleanup tracking"""
        if data_type == "pdf":
            path = TEST_DATA_PATH / "pdfs" / "samples" / "sample_metabolomics_study.txt"
        elif data_type == "database":
            path = TEST_DATA_PATH / "databases" / "samples" / "test_cost_tracking.db"
        # ... more data types
        
        self.resources.append(path)
        return path
        
    def cleanup(self):
        """Cleanup managed resources"""
        import subprocess
        subprocess.run([
            "python", str(TEST_DATA_PATH / "utilities" / "cleanup_scripts" / "cleanup_test_data.py"),
            "--mode", self.cleanup_mode
        ])

# Usage
def test_with_managed_resources():
    with TestDataManager() as manager:
        pdf_path = manager.get_sample_data("pdf")
        # Test code here
        pass
    # Automatic cleanup happens here
```

## Migration Guide

### From Existing Tests

1. **Identify Test Data**: Find existing hardcoded test data
2. **Move to Structure**: Move data to appropriate test_data directories
3. **Update Imports**: Change from hardcoded paths to fixture-based access
4. **Add Cleanup**: Ensure proper cleanup after tests
5. **Validate**: Run validation to ensure data integrity

### Example Migration

```python
# Before
def test_pdf_processing():
    content = """
    Title: Sample Study
    Abstract: This is a test...
    """
    # Test code

# After  
def test_pdf_processing(sample_pdf_content):
    # Uses fixture with structured test data
    # Test code remains the same
```

## Troubleshooting

### Common Integration Issues

1. **Path Issues**: Use absolute paths and Path objects
2. **Cleanup Not Running**: Check fixture scopes and autouse settings
3. **Data Validation Failures**: Run validator to identify issues
4. **Resource Conflicts**: Ensure proper test isolation

### Debug Tools

```python
@pytest.fixture
def debug_test_data():
    """Debug fixture for test data issues"""
    print(f"Test data path: {TEST_DATA_PATH}")
    print(f"Test data exists: {TEST_DATA_PATH.exists()}")
    
    for subdir in ["pdfs", "databases", "mocks"]:
        path = TEST_DATA_PATH / subdir
        print(f"{subdir}: exists={path.exists()}, files={len(list(path.rglob('*')))}")
```

This integration guide provides comprehensive support for incorporating the test data management system with existing test infrastructure while maintaining clean, maintainable test code.