# Test Data Utilities

This directory contains utility scripts for managing test data in the Clinical Metabolomics Oracle LightRAG integration project.

## Available Utilities

### cleanup_scripts/

**cleanup_test_data.py** - Comprehensive test data cleanup utility

**Purpose**: Manages cleanup of test data across the entire project, with intelligent preservation of templates and sample files.

**Key Features**:
- Multiple cleanup modes (temp_only, databases, logs, pdfs, states, all)
- Age-based cleanup for old files
- Template and sample preservation
- Database cleanup and optimization
- Dry-run mode for safe testing
- Comprehensive cleanup reporting

**Usage**:
```bash
# Clean temporary files only (safe)
python cleanup_test_data.py --mode temp_only

# Clean all generated files (preserves templates)
python cleanup_test_data.py --mode all

# Clean files older than 48 hours
python cleanup_test_data.py --mode all --age 48

# Dry run to see what would be cleaned
python cleanup_test_data.py --mode all --dry-run --verbose
```

**Cleanup Modes**:
- `temp_only`: Only temporary files and directories
- `databases`: Test databases (cleans data, preserves structure)
- `logs`: Log files (preserves templates and configs)
- `pdfs`: Generated PDF files (preserves samples and templates)
- `states`: Resets mock system states to defaults
- `all`: Comprehensive cleanup of all categories

### data_generators/

**generate_test_pdfs.py** - Test document generation utility

**Purpose**: Generates realistic biomedical research documents for testing PDF processing and content analysis.

**Key Features**:
- Realistic biomedical vocabulary and terminology
- Multiple complexity levels (minimal, medium, complex)
- Various document types (research papers, clinical trials)
- Controlled corruption for error testing
- Batch generation capabilities

**Usage**:
```bash
# Generate 10 test documents
python generate_test_pdfs.py --count 10

# Generate complex documents in specific directory
python generate_test_pdfs.py --count 5 --output-dir ../temp/staging

# Generate with multiple formats
python generate_test_pdfs.py --count 3 --formats txt pdf
```

**Document Types**:
- Research papers with metabolomics content
- Clinical trial protocols
- Corrupted documents for error testing
- Template-based documents with variable content

### validators/

**test_data_validator.py** - Test data integrity validation

**Purpose**: Validates the structure, content, and integrity of test data to ensure reliability of tests.

**Key Features**:
- Directory structure validation
- Content validation for different file types
- Database schema and integrity checks
- JSON format validation
- Checksum calculation for integrity verification
- Comprehensive reporting

**Usage**:
```bash
# Validate current test data
python test_data_validator.py

# Validate specific directory
python test_data_validator.py --test-data-path /path/to/test_data

# Generate detailed report
python test_data_validator.py --report-output validation_report.json

# Quiet mode for automated checks
python test_data_validator.py --quiet
```

**Validation Checks**:
- Directory structure completeness
- File format validity (JSON, SQL, etc.)
- Content quality (biomedical terms, required sections)
- Database connectivity and table existence
- Cross-references and dependencies

## Integration with Test Framework

### Pytest Integration

These utilities can be integrated with pytest for automated test data management:

```python
# In conftest.py
import subprocess

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Auto-cleanup after test session"""
    yield
    subprocess.run([
        "python", "utilities/cleanup_scripts/cleanup_test_data.py", 
        "--mode", "temp_only"
    ])

@pytest.fixture
def validate_test_data():
    """Validate test data before critical tests"""
    result = subprocess.run([
        "python", "utilities/validators/test_data_validator.py", 
        "--quiet"
    ])
    assert result.returncode == 0, "Test data validation failed"
```

### CI/CD Integration

```yaml
# Example GitHub Actions integration
- name: Validate Test Data
  run: |
    cd lightrag_integration/tests/test_data
    python utilities/validators/test_data_validator.py --quiet

- name: Cleanup Test Data
  if: always()
  run: |
    cd lightrag_integration/tests/test_data
    python utilities/cleanup_scripts/cleanup_test_data.py --mode temp_only
```

## Utility Configuration

### Environment Variables

- `TEST_DATA_PATH`: Default path for test data directory
- `CLEANUP_MAX_AGE`: Default age threshold for cleanup (hours)
- `VALIDATION_STRICT`: Enable strict validation mode

### Configuration Files

Utilities can be configured via JSON config files:

```json
{
  "cleanup": {
    "default_age_hours": 24,
    "preserve_patterns": ["*_template.*", "sample_*"],
    "temp_patterns": ["temp_*", "*.tmp", "__pycache__"]
  },
  "generation": {
    "default_count": 10,
    "biomedical_terms": ["metabolomics", "biomarker", "clinical"],
    "output_formats": ["txt"]
  },
  "validation": {
    "required_directories": ["pdfs", "databases", "mocks"],
    "min_sample_files": 2,
    "strict_mode": false
  }
}
```

## Development Guidelines

### Adding New Utilities

1. Follow the existing code structure and documentation standards
2. Include comprehensive error handling
3. Add command-line argument parsing with argparse
4. Provide both programmatic and CLI interfaces
5. Include logging with appropriate levels
6. Add usage examples in docstrings

### Testing Utilities

Utilities should include unit tests:

```python
# test_utilities.py
def test_cleanup_dry_run():
    """Test cleanup dry-run mode"""
    cleaner = TestDataCleanup("/path/to/test", dry_run=True)
    stats = cleaner.run_cleanup()
    assert stats['files_removed'] == 0  # No files should be removed in dry-run

def test_generator_creates_valid_content():
    """Test document generator creates valid content"""
    generator = TestPDFGenerator("/tmp/test")
    content = generator.generate_research_paper("TEST_001")
    assert "metabolomics" in content.lower()
    assert "ABSTRACT" in content
```

### Performance Considerations

- Use generators for large datasets
- Implement batch processing for multiple files
- Add progress reporting for long-running operations
- Consider memory usage when processing large files
- Implement incremental operations where possible

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure scripts have execute permissions
2. **Path Issues**: Use absolute paths or run from correct directory
3. **Database Locks**: Ensure no other processes are using test databases
4. **Disk Space**: Monitor space usage, especially in temp directories

### Debug Mode

Most utilities support verbose/debug modes:

```bash
python cleanup_test_data.py --verbose --dry-run
python test_data_validator.py --verbose
```

### Logging

Utilities log to both console and files. Check log files for detailed error information:

- Cleanup operations: Check cleanup logs in `reports/cleanup/`
- Validation: Check validation reports in `reports/validation/`
- Generation: Output includes generation statistics

## Best Practices

1. **Always test with dry-run**: Use `--dry-run` to preview operations
2. **Regular validation**: Run validators before important test runs
3. **Monitor disk usage**: Set up alerts for temp directory growth
4. **Backup important data**: Keep backups of template and sample files
5. **Document changes**: Update README when adding new utilities or features