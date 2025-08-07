# Test Data Management Directory

This directory contains the comprehensive test data management system for the Clinical Metabolomics Oracle LightRAG integration project. It provides structured test data, utilities, and management tools to support thorough testing of the system.

## Directory Structure

```
test_data/
├── pdfs/                    # PDF test files and templates
│   ├── samples/            # Sample biomedical PDF content files
│   ├── templates/          # PDF templates for test generation
│   └── corrupted/          # Intentionally corrupted files for error testing
├── databases/               # Database schemas and test databases
│   ├── schemas/            # SQL schema files
│   ├── samples/            # Sample databases with test data
│   └── test_dbs/          # Runtime test databases
├── logs/                   # Log file templates and configurations
│   ├── templates/          # Log file templates
│   ├── configs/           # Logging configuration files
│   └── samples/           # Sample log files
├── mocks/                  # Mock data for testing
│   ├── biomedical_data/   # Mock biomedical datasets
│   ├── api_responses/     # Mock API response data
│   └── state_data/        # System state mock data
├── temp/                   # Temporary file management
│   ├── staging/           # Temporary staging area
│   ├── processing/        # Processing workspace
│   └── cleanup/           # Cleanup workspace
├── utilities/              # Data management utilities
│   ├── cleanup_scripts/   # Cleanup and maintenance scripts
│   ├── data_generators/   # Test data generation tools
│   └── validators/        # Data validation tools
└── reports/               # Test reports and validation results
    ├── performance/       # Performance test reports
    ├── validation/        # Validation reports
    └── cleanup/           # Cleanup operation reports
```

## Key Components

### 1. PDF Test Data (`pdfs/`)

**Purpose**: Provides realistic biomedical research content for PDF processing tests.

- **samples/**: Contains sample biomedical research documents with realistic metabolomics content
- **templates/**: Provides templates for generating new test documents
- **corrupted/**: Contains intentionally corrupted files for error handling tests

**Key Files**:
- `sample_metabolomics_study.txt` - Sample metabolomics research paper
- `sample_clinical_trial.txt` - Sample clinical trial document
- `minimal_biomedical_template.txt` - Template for generating test documents
- `corrupted_sample.txt` - Corrupted file for error testing

### 2. Database Test Data (`databases/`)

**Purpose**: Supports database testing with realistic schemas and sample data.

**Key Files**:
- `cost_tracking_schema.sql` - Complete schema for cost tracking system
- `knowledge_base_schema.sql` - Schema for LightRAG knowledge base
- `test_cost_tracking.db` - Sample SQLite database with test data
- `test_knowledge_base.db` - Sample knowledge base with biomedical data

### 3. Mock Data (`mocks/`)

**Purpose**: Provides controlled mock data for testing various system components.

**Key Files**:
- `mock_metabolites.json` - Comprehensive metabolite database with realistic biomedical data
- `openai_api_responses.json` - Mock OpenAI API responses for different scenarios
- `mock_system_states.json` - System state data for testing monitoring and recovery

### 4. Utilities (`utilities/`)

**Purpose**: Provides tools for managing, generating, and validating test data.

**Key Scripts**:
- `cleanup_test_data.py` - Comprehensive cleanup utility for test data management
- `generate_test_pdfs.py` - Generates realistic biomedical test documents
- `test_data_validator.py` - Validates test data integrity and structure

## Usage Guidelines

### Test Data Lifecycle

1. **Generation**: Use data generators to create test content
2. **Validation**: Run validators to ensure data integrity
3. **Testing**: Use structured data in test suites
4. **Cleanup**: Regular cleanup of temporary and generated files

### Best Practices

1. **Keep Templates Static**: Never modify template files during tests
2. **Use Prefixes**: Generated files should use `generated_*` or `test_*` prefixes
3. **Regular Cleanup**: Run cleanup scripts after test runs
4. **Validation**: Validate test data integrity before major test runs

### Data Management Commands

```bash
# Generate test documents
python utilities/data_generators/generate_test_pdfs.py --count 10 --output-dir temp/staging

# Validate test data integrity
python utilities/validators/test_data_validator.py --test-data-path .

# Cleanup temporary files
python utilities/cleanup_scripts/cleanup_test_data.py --mode temp_only

# Full cleanup (excluding templates)
python utilities/cleanup_scripts/cleanup_test_data.py --mode all

# Dry run cleanup (see what would be cleaned)
python utilities/cleanup_scripts/cleanup_test_data.py --mode all --dry-run
```

## Integration with Test Infrastructure

### Existing Test Support

This test data structure integrates with the existing test infrastructure:

- **Fixtures**: Use structured data in pytest fixtures
- **Mock Systems**: Replace API calls with mock data
- **Error Testing**: Use corrupted data for error handling tests
- **Performance**: Use varied data sizes for performance tests

### Test Categories Supported

1. **PDF Processing Tests**: Use sample and corrupted PDF data
2. **Database Tests**: Use schema files and sample databases
3. **API Integration Tests**: Use mock API response data
4. **Error Handling Tests**: Use corrupted and invalid data
5. **Performance Tests**: Use datasets of varying sizes
6. **Recovery Tests**: Use system state data for recovery scenarios

## Git Integration

### Tracked Files
- All template files and schemas
- Sample data files
- Utility scripts
- Documentation

### Ignored Files (via .gitignore)
- Generated test files (`generated_*`, `test_*`)
- Temporary files in `temp/` directories
- Runtime databases
- Dynamic log files
- Performance reports

### File Naming Conventions
- **Templates**: `*_template.*`, `sample_*`
- **Generated**: `generated_*`, `test_*`
- **Runtime**: `runtime_*`, `dynamic_*`

## Maintenance

### Regular Tasks

1. **Weekly**: Run cleanup scripts to remove old temporary files
2. **Monthly**: Validate test data integrity
3. **Before Releases**: Full cleanup and validation
4. **After Major Changes**: Regenerate test datasets

### Monitoring

- Check disk usage in `temp/` directories
- Monitor database sizes in `test_dbs/`
- Validate that sample files remain uncorrupted

## Troubleshooting

### Common Issues

1. **Disk Space**: Run cleanup scripts to free space
2. **Corrupted Templates**: Restore from version control
3. **Invalid Test Data**: Run validator to identify issues
4. **Permission Errors**: Check file permissions on utility scripts

### Recovery Procedures

1. Reset to clean state: `git checkout -- test_data/`
2. Regenerate databases: Run schema files with SQLite
3. Validate integrity: Run validation scripts
4. Clean temporary files: Run cleanup with `--mode all`

## Contributing

When adding new test data:

1. Follow existing naming conventions
2. Place static data in appropriate template directories
3. Ensure generated data uses proper prefixes
4. Update this README if adding new categories
5. Run validation scripts before committing

## Related Documentation

- [Comprehensive Test Implementation Guide](../COMPREHENSIVE_TEST_IMPLEMENTATION_GUIDE.md)
- [Test Utilities Guide](../TEST_UTILITIES_GUIDE.md)
- [Error Handling Test Guide](../COMPREHENSIVE_ERROR_HANDLING_TEST_GUIDE.md)
- [Performance Testing Guide](../COMPREHENSIVE_PERFORMANCE_QUALITY_TESTING_GUIDE.md)