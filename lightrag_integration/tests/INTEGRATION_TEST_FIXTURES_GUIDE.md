# Integration Test Fixtures and Sample Data Guide

This guide provides comprehensive documentation for the integration test fixtures and sample PDF data created for testing the PDF processor and LightRAG components integration.

## Overview

The integration test infrastructure consists of four main components:

1. **Extended conftest.py** - Base fixtures and utilities for all tests
2. **test_fixtures.py** - Advanced test fixtures and specialized mock systems
3. **test_configurations.py** - Comprehensive test configurations for various scenarios
4. **Sample PDF data generators** - Realistic biomedical content generators

## Files Created

### Core Files

- `/lightrag_integration/tests/conftest.py` (extended)
- `/lightrag_integration/tests/test_fixtures.py`
- `/lightrag_integration/tests/test_configurations.py`

## Feature Overview

### 1. Realistic Biomedical PDF Generation

The fixtures provide comprehensive biomedical PDF test data generation with:

- **Disease-specific content** for diabetes, cardiovascular, cancer, and liver disease
- **Technical complexity levels** (simple, medium, complex) 
- **Realistic metadata** including authors, journals, DOIs, keywords
- **Actual PDF file creation** using PyMuPDF when available
- **Failure scenario simulation** for robustness testing

### 2. Sophisticated Mock Systems

#### MockLightRAGSystem
- Realistic processing delays and costs
- Entity and relationship extraction simulation
- Query caching and response generation
- Performance statistics tracking
- Biomedical entity recognition patterns

#### RealisticLightRAGMock
- Advanced cost calculation based on token usage
- Contextual query responses based on content type
- Knowledge graph simulation
- Cache hit rate tracking
- Configurable failure injection

### 3. Comprehensive Test Configurations

#### Base Configuration Types
- `IntegrationTestConfig` - Base configuration for all tests
- `FailureTestConfig` - Failure scenario testing
- `PerformanceTestConfig` - Performance and load testing  
- `SecurityTestConfig` - Security and validation testing

#### Predefined Scenarios
- Basic integration testing
- Comprehensive feature coverage
- Performance benchmarking
- Memory-constrained environments
- High failure rate scenarios
- Budget-constrained testing
- Security validation
- Scalability testing

### 4. Advanced Testing Utilities

- **Error injection framework** for controlled failure testing
- **Performance monitoring** with detailed metrics tracking
- **Integration test scenario builders** for complex workflows
- **Resource usage monitoring** with memory and CPU tracking
- **Configuration validation** with optimization suggestions

## Usage Examples

### Basic Integration Test

```python
import pytest
from lightrag_integration.tests.test_fixtures import BiomedicalPDFGenerator
from lightrag_integration.tests.test_configurations import TestConfigurationLibrary

@pytest.mark.asyncio
async def test_basic_pdf_to_lightrag_integration(
    integration_test_environment, 
    basic_integration_config
):
    """Test basic PDF processing to LightRAG integration."""
    
    # Create test PDFs
    pdf_paths = integration_test_environment.create_test_pdf_collection(count=3)
    
    # Process PDFs
    for pdf_path in pdf_paths:
        result = await integration_test_environment.pdf_processor.process_pdf(pdf_path)
        assert result['success'] == True
        
        # Index in LightRAG
        documents = [result['text']]
        lightrag_result = await integration_test_environment.lightrag_system.ainsert(documents)
        assert lightrag_result['status'] == 'success'
    
    # Execute test queries
    query_result = await integration_test_environment.lightrag_system.aquery(
        "What metabolites are associated with diabetes?"
    )
    assert len(query_result) > 0
    
    # Verify cost tracking
    stats = integration_test_environment.lightrag_system.get_statistics()
    assert stats['total_cost'] <= basic_integration_config.daily_budget_limit
```

### Performance Testing

```python
@pytest.mark.asyncio
async def test_performance_benchmark(
    integration_test_environment,
    performance_test_config,
    performance_monitor
):
    """Test system performance under load."""
    
    async with performance_monitor.monitor_operation("full_workflow"):
        # Create large document collection
        pdf_paths = integration_test_environment.create_test_pdf_collection(
            count=performance_test_config.document_count
        )
        
        # Process in parallel
        tasks = []
        for pdf_path in pdf_paths:
            task = integration_test_environment.pdf_processor.process_pdf(pdf_path)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        successful_results = [r for r in results if r.get('success')]
        
        # Verify performance targets
        summary = performance_monitor.get_performance_summary()
        assert summary['success_rate'] >= 90  # 90% success rate
        assert summary['average_operation_time'] <= performance_test_config.max_response_time_seconds
```

### Failure Scenario Testing

```python
@pytest.mark.asyncio
async def test_failure_recovery(
    integration_test_environment,
    failure_test_config,
    error_injector
):
    """Test system resilience under failure conditions."""
    
    # Configure error injection
    error_injector.add_rule(
        target="pdf_processing",
        error_type=PDFProcessingTimeoutError("Simulated timeout"),
        trigger_after=2,
        probability=failure_test_config.pdf_failure_rate
    )
    
    pdf_paths = integration_test_environment.create_test_pdf_collection(count=10)
    successful_processing = 0
    failures = 0
    
    for pdf_path in pdf_paths:
        try:
            # Check if error should be injected
            error = error_injector.should_inject_error("pdf_processing")
            if error:
                raise error
            
            result = await integration_test_environment.pdf_processor.process_pdf(pdf_path)
            successful_processing += 1
            
        except Exception as e:
            failures += 1
            # Verify proper error handling
            assert failure_test_config.require_error_logging
    
    # Verify partial success handling
    if failure_test_config.expect_partial_success:
        assert successful_processing > 0
        assert failures > 0
    
    # Verify cleanup
    if failure_test_config.require_cleanup_on_failure:
        # Verify environment is clean
        assert integration_test_environment.working_dir.exists()
```

### Disease-Specific Content Testing

```python
@pytest.mark.asyncio  
async def test_diabetes_content_processing(
    integration_test_environment,
    biomedical_domain_config,
    disease_specific_content
):
    """Test processing of diabetes-specific biomedical content."""
    
    # Generate diabetes-specific content
    diabetes_content = disease_specific_content('diabetes', 'complex')
    
    # Create test PDF with diabetes content
    pdf_path = integration_test_environment.working_dir / "pdfs" / "diabetes_study.pdf"
    pdf_path.write_text(f"Title: Diabetes Metabolomics Study\n\n{diabetes_content}")
    
    # Process through pipeline
    pdf_result = await integration_test_environment.pdf_processor.process_pdf(pdf_path)
    assert pdf_result['success'] == True
    
    # Index in LightRAG
    lightrag_result = await integration_test_environment.lightrag_system.ainsert([pdf_result['text']])
    
    # Verify biomedical entities extracted
    assert lightrag_result['entities_extracted'] >= biomedical_domain_config.min_entities_per_document
    assert lightrag_result['relationships_found'] >= biomedical_domain_config.min_relationships_per_document
    
    # Test domain-specific queries
    queries = [
        "What metabolites are elevated in diabetes?",
        "How does glucose metabolism change in diabetes?",
        "What are the biomarkers for diabetes progression?"
    ]
    
    for query in queries:
        response = await integration_test_environment.lightrag_system.aquery(query)
        assert 'diabetes' in response.lower()
        assert any(metabolite in response.lower() for metabolite in ['glucose', 'insulin', 'lactate'])
```

### Configuration Testing

```python
def test_configuration_validation(configuration_validator, all_test_configurations):
    """Test configuration validation across all predefined configs."""
    
    for config_name, config in all_test_configurations.items():
        # Validate configuration
        issues = configuration_validator.validate_config(config)
        
        # Should have no critical issues
        critical_issues = [issue for issue in issues if 'must' in issue.lower()]
        assert len(critical_issues) == 0, f"Critical issues in {config_name}: {critical_issues}"
        
        # Get optimization suggestions
        suggestions = configuration_validator.suggest_optimizations(config)
        print(f"Optimizations for {config_name}: {suggestions}")
        
        # Verify budget settings are reasonable
        assert config.daily_budget_limit > 0
        assert config.monthly_budget_limit >= config.daily_budget_limit * 20  # At least 20 days
        
        # Verify resource settings
        assert config.max_memory_mb >= 100  # Minimum viable memory
        assert config.max_processing_time >= 10  # Minimum processing time
```

## Available Fixtures

### From conftest.py (Extended)

- `pdf_test_documents` - Collection of realistic PDF test documents
- `small_pdf_collection` - 3 PDFs for quick tests
- `large_pdf_collection` - 25 PDFs for performance tests  
- `temp_pdf_files` - Actual PDF files created on disk
- `mock_lightrag_system` - Basic LightRAG mock system
- `integration_config` - Basic integration configuration
- `mock_pdf_processor` - Comprehensive PDF processor mock
- `mock_cost_monitor` - Cost monitoring mock system
- `mock_progress_tracker` - Progress tracking mock system
- `error_injector` - Error injection utility
- `integration_test_environment` - Complete integration test environment

### From test_fixtures.py

- `advanced_pdf_scenarios` - Advanced PDF processing scenarios
- `realistic_lightrag_mock` - Sophisticated LightRAG mock
- `performance_monitor` - Performance monitoring utilities
- `scenario_builder` - Integration test scenario builder
- `disease_specific_content` - Disease-specific content generator
- `failure_simulation` - Failure simulation utilities

### From test_configurations.py

- `basic_integration_config` - Basic integration test configuration
- `comprehensive_integration_config` - Full feature test configuration
- `performance_test_config` - Performance testing configuration
- `failure_test_config` - Failure scenario configuration
- `security_test_config` - Security validation configuration
- `biomedical_domain_config` - Biomedical domain-specific configuration
- `memory_constrained_config` - Memory-constrained configuration
- `budget_constrained_config` - Budget-constrained configuration
- `all_test_configurations` - All predefined configurations
- `configuration_validator` - Configuration validation utility

## Test Data Characteristics

### PDF Document Types

1. **Metabolomics Studies**
   - Diabetes biomarker research
   - Liver disease metabolomics
   - Cancer metabolomics
   - Cardiovascular metabolomics

2. **Proteomics Studies**
   - Disease-specific protein profiling
   - Therapeutic target identification
   - Biomarker validation studies

3. **Genomics Studies**
   - GWAS studies
   - Gene expression analysis
   - Variant association studies

### Content Complexity Levels

- **Simple**: Basic studies with ELISA, colorimetric assays, 5-10 pages
- **Medium**: LC-MS/MS, GC-MS studies with statistical analysis, 10-20 pages
- **Complex**: Multi-omics, machine learning approaches, 20-40 pages

### Realistic Metadata

- Authentic journal names and DOI patterns
- Realistic author lists and affiliations
- Appropriate keywords and publication years
- File size and page count simulation

## Configuration Types

### Integration Test Configurations

- **basic_integration**: Minimal settings for quick tests
- **comprehensive_integration**: Full feature coverage
- **memory_constrained**: Limited memory scenarios
- **budget_constrained**: Cost-limited testing

### Performance Configurations

- **performance_benchmark**: High-load testing
- **scalability_test**: Large-scale processing
- **real_time_processing**: Low-latency requirements
- **batch_processing**: High-throughput scenarios

### Specialized Configurations

- **biomedical_domain**: Optimized for biomedical content
- **security_validation**: Security and robustness testing
- **edge_cases**: Boundary condition testing
- **multilingual**: Multi-language content support

## Best Practices

### 1. Test Structure

```python
@pytest.mark.asyncio
async def test_integration_scenario(
    integration_test_environment,
    appropriate_config,
    performance_monitor
):
    """Use descriptive test names and appropriate fixtures."""
    
    # Setup phase
    async with performance_monitor.monitor_operation("setup"):
        # Initialize test data
        pass
    
    # Execution phase  
    async with performance_monitor.monitor_operation("main_workflow"):
        # Execute main test logic
        pass
    
    # Validation phase
    # Assert results and verify expectations
    
    # Cleanup is handled by fixtures automatically
```

### 2. Configuration Selection

```python
# Select configuration based on test type
config = select_config_for_test_type('performance')

# Or create custom configuration
config = create_custom_config(
    base_type='basic',
    max_memory_mb=500,
    batch_size=10,
    daily_budget_limit=20.0
)

# Validate configuration before use
validator = ConfigurationValidator()
issues = validator.validate_config(config)
assert len(issues) == 0, f"Configuration issues: {issues}"
```

### 3. Error Handling Testing

```python
# Test with controlled error injection
error_injector.add_rule(
    target="lightrag_indexing",
    error_type=Exception("Network timeout"),
    trigger_after=3,
    probability=0.3
)

# Verify graceful failure handling
try:
    result = await process_documents(documents)
    # Should handle partial failures gracefully
except Exception as e:
    # Verify proper error logging and cleanup
    assert "Network timeout" in str(e)
```

### 4. Performance Monitoring

```python
async with performance_monitor.monitor_operation("document_processing", document_count=10):
    # Process documents
    results = await process_document_batch(documents)

# Get performance summary
summary = performance_monitor.get_performance_summary()
assert summary['average_operation_time'] <= target_time
assert summary['success_rate'] >= 0.95
```

## Common Usage Patterns

### Integration Workflow Testing

Test the complete pipeline from PDF processing to knowledge base querying:

```python
@pytest.mark.asyncio
async def test_complete_integration_workflow(integration_test_environment):
    # 1. Create test PDFs
    pdfs = integration_test_environment.create_test_pdf_collection(5)
    
    # 2. Process PDFs
    processed_docs = []
    for pdf in pdfs:
        result = await integration_test_environment.pdf_processor.process_pdf(pdf)
        processed_docs.append(result['text'])
    
    # 3. Index documents
    index_result = await integration_test_environment.lightrag_system.ainsert(processed_docs)
    
    # 4. Query knowledge base
    query_result = await integration_test_environment.lightrag_system.aquery("test query")
    
    # 5. Verify results
    assert len(processed_docs) == 5
    assert index_result['status'] == 'success'
    assert query_result is not None
```

### Performance Benchmarking

```python
@pytest.mark.asyncio
async def test_performance_under_load(
    integration_test_environment, 
    performance_test_config,
    performance_monitor
):
    # Create large document set
    documents = integration_test_environment.create_test_pdf_collection(
        performance_test_config.document_count
    )
    
    # Execute concurrent processing
    start_time = time.time()
    async with performance_monitor.monitor_operation("bulk_processing"):
        tasks = [
            integration_test_environment.pdf_processor.process_pdf(doc)
            for doc in documents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify performance targets
    processing_time = time.time() - start_time
    throughput = len(documents) / processing_time
    
    assert throughput >= performance_test_config.target_documents_per_second
    assert processing_time <= performance_test_config.max_processing_time
```

### Failure Recovery Testing

```python
@pytest.mark.asyncio
async def test_system_resilience(
    integration_test_environment,
    failure_test_config,
    error_injector
):
    # Configure failure scenarios
    error_injector.add_rule("pdf_processing", PDFProcessingError(), probability=0.3)
    error_injector.add_rule("lightrag_indexing", Exception("API Error"), probability=0.2)
    
    documents = integration_test_environment.create_test_pdf_collection(20)
    successful_count = 0
    
    for doc in documents:
        try:
            # Inject failures based on configuration
            if error_injector.should_inject_error("pdf_processing"):
                raise PDFProcessingError("Simulated failure")
            
            result = await integration_test_environment.pdf_processor.process_pdf(doc)
            successful_count += 1
            
        except Exception:
            # Expected failures - verify proper handling
            pass
    
    # Verify partial success and graceful degradation
    success_rate = successful_count / len(documents)
    expected_success_rate = 1.0 - failure_test_config.pdf_failure_rate
    assert success_rate >= expected_success_rate * 0.8  # Allow some variance
```

## Maintenance and Extension

### Adding New Content Types

To add support for new biomedical content types:

1. Extend `BiomedicalPDFGenerator.CONTENT_TEMPLATES` in `test_fixtures.py`
2. Add new entity patterns to `MockLightRAGSystem.entity_patterns`
3. Update response generation logic in `_generate_contextual_response`

### Adding New Configurations

To add new test configurations:

1. Create new configuration class in `test_configurations.py`
2. Add factory methods to `TestConfigurationLibrary`
3. Create corresponding pytest fixture
4. Add validation rules if needed

### Extending Mock Systems

To enhance mock system behavior:

1. Add new methods to `RealisticLightRAGMock` 
2. Implement additional entity/relationship extraction patterns
3. Add new response generation strategies
4. Include additional performance metrics

This comprehensive fixture system provides a solid foundation for thorough integration testing of the PDF processor and LightRAG components, enabling testing of normal operations, failure scenarios, performance characteristics, and security aspects.