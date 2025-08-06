# ClinicalMetabolomicsRAG Test-Driven Development Guide

## Overview

This document explains the comprehensive test suite for the `ClinicalMetabolomicsRAG` class initialization, designed following Test-Driven Development (TDD) principles. The tests in `test_clinical_metabolomics_rag.py` define the expected behavior of the class **before** it is implemented, ensuring that the implementation meets all requirements from CMO-LIGHTRAG-005.

## Current Status: TDD Phase

**The ClinicalMetabolomicsRAG class does NOT exist yet.** This is intentional following TDD methodology:

1. ✅ **Write Tests First**: Comprehensive tests are written defining expected behavior
2. ⏳ **Run Tests (RED)**: Tests fail because implementation doesn't exist yet  
3. ⏳ **Implement Code (GREEN)**: Write minimal code to make tests pass
4. ⏳ **Refactor**: Improve code while keeping tests passing

## Test Suite Structure

### Test Classes and Coverage

#### 1. `TestClinicalMetabolomicsRAGInitialization` (5 tests)
- **Purpose**: Core initialization functionality
- **Key Tests**:
  - `test_initialization_with_valid_config`: Basic initialization with valid config
  - `test_initialization_with_none_config_raises_error`: Error handling for None config
  - `test_initialization_sets_up_required_attributes`: Verify all required attributes exist
  - `test_initialization_with_custom_working_directory`: Custom directory handling

#### 2. `TestClinicalMetabolomicsRAGConfiguration` (3 tests)
- **Purpose**: Configuration validation and parameter handling
- **Key Tests**:
  - `test_config_validation_during_initialization`: Config validation during init
  - `test_biomedical_parameters_setup`: Biomedical-specific parameter configuration
  - `test_config_parameter_override`: Parameter override functionality

#### 3. `TestClinicalMetabolomicsRAGLightRAGSetup` (3 tests)
- **Purpose**: LightRAG integration setup
- **Key Tests**:
  - `test_lightrag_instance_creation`: LightRAG instance creation with correct parameters
  - `test_lightrag_biomedical_configuration`: Biomedical-specific LightRAG config
  - `test_lightrag_storage_initialization`: Storage directory initialization

#### 4. `TestClinicalMetabolomicsRAGOpenAISetup` (3 tests)
- **Purpose**: OpenAI API integration
- **Key Tests**:
  - `test_openai_llm_function_setup`: OpenAI LLM function configuration
  - `test_openai_embedding_function_setup`: OpenAI embedding function setup
  - `test_openai_api_error_handling`: API error handling and recovery

#### 5. `TestClinicalMetabolomicsRAGErrorHandling` (4 tests)
- **Purpose**: Error handling and recovery mechanisms
- **Key Tests**:
  - `test_missing_api_key_error`: Missing API key error handling
  - `test_invalid_working_directory_error`: Invalid directory error handling
  - `test_lightrag_initialization_failure_handling`: LightRAG init failure handling
  - `test_rate_limit_error_handling`: Rate limiting and retry logic

#### 6. `TestClinicalMetabolomicsRAGBiomedicalConfig` (3 tests)
- **Purpose**: Biomedical-specific configuration
- **Key Tests**:
  - `test_biomedical_entity_types_configuration`: Entity types setup
  - `test_biomedical_relationship_types_configuration`: Relationship types setup
  - `test_clinical_metabolomics_specific_keywords`: Domain keywords configuration

#### 7. `TestClinicalMetabolomicsRAGMonitoring` (3 tests)
- **Purpose**: Logging and monitoring functionality
- **Key Tests**:
  - `test_logger_initialization`: Logger setup and configuration
  - `test_cost_monitoring_initialization`: API cost monitoring setup
  - `test_query_history_tracking`: Query history tracking functionality

#### 8. `TestClinicalMetabolomicsRAGQueryFunctionality` (3 tests)
- **Purpose**: Basic query processing functionality
- **Key Tests**:
  - `test_basic_query_functionality`: Basic query processing
  - `test_query_modes_support`: Multiple LightRAG query modes support
  - `test_cost_tracking_during_queries`: Cost tracking during queries

#### 9. `TestClinicalMetabolomicsRAGAsyncOperations` (2 tests)
- **Purpose**: Async functionality and resource management
- **Key Tests**:
  - `test_concurrent_query_processing`: Concurrent query handling
  - `test_resource_cleanup_after_operations`: Resource cleanup functionality

#### 10. `TestClinicalMetabolomicsRAGEdgeCases` (4 tests)
- **Purpose**: Edge cases and error conditions
- **Key Tests**:
  - `test_empty_working_directory_handling`: Empty directory handling
  - `test_query_with_empty_string`: Empty query handling
  - `test_query_with_very_long_input`: Long input handling
  - `test_multiple_initialization_calls`: Multiple init calls safety

#### 11. `TestClinicalMetabolomicsRAGIntegration` (2 tests)
- **Purpose**: Integration tests combining multiple components
- **Key Tests**:
  - `test_full_initialization_and_query_workflow`: Complete workflow test
  - `test_config_integration_with_pdf_processor`: PDF processor integration

#### 12. `TestClinicalMetabolomicsRAGPerformance` (2 tests)
- **Purpose**: Performance and benchmarking
- **Key Tests**:
  - `test_initialization_performance`: Initialization performance limits
  - `test_query_response_time`: Query response time performance

## Expected Class Interface (Based on Tests)

The tests define that `ClinicalMetabolomicsRAG` should have:

### Constructor
```python
def __init__(self, config: LightRAGConfig, **kwargs) -> None:
    """Initialize with LightRAGConfig and optional parameters."""
```

### Required Attributes
- `config`: LightRAGConfig instance
- `lightrag_instance`: LightRAG instance
- `logger`: Logger for the class
- `cost_monitor`: Cost monitoring functionality
- `is_initialized`: Initialization status
- `query_history`: List of query history
- `total_cost`: Total API costs
- `biomedical_params`: Biomedical-specific parameters

### Methods
```python
async def query(self, query: str, mode: str = 'hybrid') -> Dict[str, Any]:
    """Process a query and return response with metadata and cost."""

async def cleanup(self) -> None:
    """Clean up resources after operations."""

def track_api_cost(self, cost: float) -> None:
    """Track API costs for monitoring."""

def get_cost_summary(self) -> Dict[str, Any]:
    """Get summary of API costs."""
```

## Biomedical Configuration Requirements

### Entity Types
The tests expect these biomedical entity types:
- `METABOLITE`, `PROTEIN`, `GENE`, `DISEASE`, `PATHWAY`
- `ORGANISM`, `TISSUE`, `BIOMARKER`, `DRUG`, `CLINICAL_TRIAL`

### Relationship Types
The tests expect these biomedical relationship types:
- `METABOLIZES`, `REGULATES`, `INTERACTS_WITH`, `CAUSES`
- `TREATS`, `ASSOCIATED_WITH`, `PART_OF`, `EXPRESSED_IN`
- `TARGETS`, `MODULATES`

### Domain Keywords
Clinical metabolomics-specific keywords:
- `metabolomics`, `clinical`, `biomarker`, `mass spectrometry`
- `NMR`, `metabolite`, `pathway analysis`, `biofluid`

## Test Fixtures and Utilities

### Mock Classes
- `MockLightRAGResponse`: Mock LightRAG query response
- `MockOpenAIAPIUsage`: Mock OpenAI API usage for cost monitoring
- `MockLightRAGInstance`: Mock LightRAG instance for testing

### Fixtures
- `valid_config`: Valid LightRAGConfig for successful tests
- `invalid_config`: Invalid LightRAGConfig for error testing
- `mock_openai_client`: Mock OpenAI client
- `temp_working_dir`: Temporary working directory
- `mock_pdf_processor`: Mock PDF processor

## Running the Tests

### Current State (TDD RED Phase)
```bash
# Run all tests - should show failures/skips as implementation doesn't exist
python -m pytest lightrag_integration/tests/test_clinical_metabolomics_rag.py -v

# Expected output: 1 failed, 36 skipped (implementation missing)
```

### After Implementation (TDD GREEN Phase)
```bash
# Run all tests - should pass after implementation
python -m pytest lightrag_integration/tests/test_clinical_metabolomics_rag.py -v

# Expected output: 37 passed (or mostly passed with some failures to fix)
```

## Implementation Guidelines

### 1. Start with Minimal Implementation
Create `lightrag_integration/clinical_metabolomics_rag.py` with minimal class structure:
```python
from lightrag_integration.config import LightRAGConfig, LightRAGConfigError
import logging
from typing import Dict, Any, List, Optional

class ClinicalMetabolomicsRAG:
    def __init__(self, config: LightRAGConfig, **kwargs):
        if config is None:
            raise ValueError("config cannot be None")
        if not isinstance(config, LightRAGConfig):
            raise TypeError("config must be a LightRAGConfig instance")
        
        # Validate config
        config.validate()
        
        self.config = config
        self.is_initialized = False
        # ... minimal attributes to pass first tests
```

### 2. Iteratively Add Functionality
Add functionality incrementally to pass more tests:
1. Basic initialization and attribute setup
2. LightRAG integration
3. OpenAI API configuration
4. Biomedical parameters
5. Query functionality
6. Error handling
7. Async operations

### 3. Use Test Results to Guide Development
- Run tests frequently: `python -m pytest lightrag_integration/tests/test_clinical_metabolomics_rag.py::TestClinicalMetabolomicsRAGInitialization -v`
- Focus on one test class at a time
- Let test failures guide what to implement next

## CMO-LIGHTRAG-005 Requirements Traceability

| Requirement | Test Coverage | Status |
|-------------|---------------|---------|
| Initialization with LightRAGConfig | `TestClinicalMetabolomicsRAGInitialization` | ✅ Tested |
| LightRAG setup with biomedical parameters | `TestClinicalMetabolomicsRAGLightRAGSetup`, `TestClinicalMetabolomicsRAGBiomedicalConfig` | ✅ Tested |
| OpenAI LLM and embedding functions configuration | `TestClinicalMetabolomicsRAGOpenAISetup` | ✅ Tested |
| Error handling for API failures and rate limits | `TestClinicalMetabolomicsRAGErrorHandling` | ✅ Tested |
| Basic query functionality working | `TestClinicalMetabolomicsRAGQueryFunctionality` | ✅ Tested |
| API cost monitoring and logging | `TestClinicalMetabolomicsRAGMonitoring` | ✅ Tested |

## Next Steps

1. **Implement the Class**: Create `lightrag_integration/clinical_metabolomics_rag.py`
2. **Start Simple**: Make the first few tests pass with minimal implementation
3. **Iterate**: Gradually add functionality to pass more tests
4. **Refactor**: Improve code structure while keeping tests passing
5. **Integration**: Once all tests pass, integrate with the larger system

## Dependencies

The implementation will need:
- `lightrag`: LightRAG library
- `openai`: OpenAI API client
- `asyncio`: For async functionality
- `logging`: For logging functionality
- Existing project modules: `config.py`, `pdf_processor.py`

This comprehensive test suite ensures that the `ClinicalMetabolomicsRAG` class implementation will meet all requirements and handle edge cases properly, following TDD best practices.