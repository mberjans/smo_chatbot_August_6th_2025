#!/usr/bin/env python3
"""
Example Complete Test Framework Usage - CMO-LIGHTRAG-008-T06.

This example demonstrates how to use the complete test utilities framework
for real Clinical Metabolomics Oracle testing scenarios. It showcases:

1. How to use ConfigurationTestHelper for different test scenarios
2. Integration with all existing test utilities 
3. Resource cleanup and management
4. Performance testing with validation
5. Async test coordination
6. Biomedical content validation

This serves as a practical guide for implementing tests using the complete
test utilities framework.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Import the complete test framework
from configuration_test_utilities import (
    ConfigurationTestHelper, ResourceCleanupManager, TestScenarioType,
    create_complete_test_environment, managed_test_environment,
    validate_test_configuration
)
from test_utilities import SystemComponent, MockBehavior, TestComplexity
from async_test_utilities import AsyncTestCoordinator, AsyncOperationSpec
from performance_test_utilities import PerformanceThreshold


# =====================================================================
# UNIT TESTS WITH CONFIGURATION FRAMEWORK
# =====================================================================

class TestUnitTestsWithFramework:
    """Unit tests using the complete configuration framework."""
    
    def test_simple_unit_test_with_framework(self, standard_unit_test_config):
        """Test simple unit test scenario with configuration framework."""
        test_env = standard_unit_test_config
        
        # Validate configuration
        validation_errors = validate_test_configuration(test_env)
        assert not validation_errors, f"Configuration validation failed: {validation_errors}"
        
        # Test environment components
        assert test_env['environment_manager'] is not None
        assert test_env['cleanup_manager'] is not None
        assert test_env['working_dir'] is not None
        
        # Test working directory exists
        working_dir = Path(test_env['working_dir'])
        assert working_dir.exists()
        
        # Test mock system if available
        mock_system = test_env.get('mock_system', {})
        if 'logger' in mock_system:
            logger_mock = mock_system['logger']
            logger_mock.info("Test log message")
            assert logger_mock.info.called
    
    def test_unit_test_with_custom_configuration(self, configuration_test_helper):
        """Test unit test with custom configuration overrides."""
        # Create custom unit test configuration
        context_id = configuration_test_helper.create_test_configuration(
            TestScenarioType.UNIT_TEST,
            custom_overrides={
                'environment_vars': {'TEST_MODE': 'custom'},
                'performance_thresholds': {
                    'custom_metric': PerformanceThreshold(
                        metric_name='custom_metric',
                        threshold_value=2.0,
                        comparison_operator='lt',
                        unit='seconds'
                    )
                }
            }
        )
        
        try:
            test_env = configuration_test_helper.get_integrated_test_environment(context_id)
            
            # Test custom environment variable
            import os
            assert os.environ.get('TEST_MODE') == 'custom'
            
            # Test performance helper has custom threshold
            if test_env['performance_helper']:
                # Performance helper should have the custom threshold registered
                pass  # Implementation depends on PerformanceAssertionHelper interface
            
            # Validate the custom configuration
            validation_errors = validate_test_configuration(test_env)
            assert not validation_errors, f"Custom configuration validation failed: {validation_errors}"
            
        finally:
            configuration_test_helper.cleanup_configuration(context_id)


# =====================================================================
# INTEGRATION TESTS WITH COMPLETE FRAMEWORK
# =====================================================================

class TestIntegrationTestsWithFramework:
    """Integration tests using the complete configuration framework."""
    
    @pytest_asyncio.async_test
    async def test_biomedical_query_integration(self, biomedical_test_config):
        """Test biomedical query integration with complete framework."""
        test_env = biomedical_test_config
        
        # Get mock LightRAG system
        mock_system = test_env.get('mock_system', {})
        if 'lightrag_system' not in mock_system:
            pytest.skip("LightRAG mock not available")
        
        lightrag_mock = mock_system['lightrag_system']
        
        # Test biomedical query
        test_query = "What metabolites are associated with diabetes progression?"
        
        # Execute query using mock system
        start_time = time.time()
        response = await lightrag_mock.aquery(test_query)
        query_duration = time.time() - start_time
        
        # Validate response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 50  # Should be substantial response
        
        # Check for biomedical content
        response_lower = response.lower()
        biomedical_terms = ['metabolite', 'glucose', 'diabetes', 'biomarker', 'clinical']
        found_terms = [term for term in biomedical_terms if term in response_lower]
        assert len(found_terms) >= 2, f"Response should contain biomedical terms, found: {found_terms}"
        
        # Performance validation
        assert query_duration < 5.0, f"Query took too long: {query_duration:.3f}s"
        
        # Test resource usage
        if test_env['environment_manager']:
            health = test_env['environment_manager'].check_system_health()
            assert health['memory_usage_mb'] < 500, f"Memory usage too high: {health['memory_usage_mb']}MB"
    
    @pytest_asyncio.async_test
    async def test_pdf_processing_integration(self, standard_integration_test_config):
        """Test PDF processing integration with complete framework."""
        test_env = standard_integration_test_config
        
        # Get mock PDF processor
        mock_system = test_env.get('mock_system', {})
        if 'pdf_processor' not in mock_system:
            pytest.skip("PDF processor mock not available")
        
        pdf_processor_mock = mock_system['pdf_processor']
        
        # Test PDF processing
        test_pdf_path = "test_diabetes_study.pdf"
        
        start_time = time.time()
        result = await pdf_processor_mock.process_pdf(test_pdf_path)
        processing_duration = time.time() - start_time
        
        # Validate result structure
        assert result is not None
        assert 'text' in result
        assert 'metadata' in result
        assert 'success' in result
        assert result['success'] is True
        
        # Validate content
        text_content = result['text']
        assert len(text_content) > 100  # Should have substantial content
        
        # Check for clinical content
        content_lower = text_content.lower()
        clinical_terms = ['study', 'patients', 'analysis', 'clinical', 'research']
        found_terms = [term for term in clinical_terms if term in content_lower]
        assert len(found_terms) >= 2, f"Content should contain clinical terms, found: {found_terms}"
        
        # Performance validation
        assert processing_duration < 10.0, f"PDF processing took too long: {processing_duration:.3f}s"


# =====================================================================
# ASYNC TESTS WITH COORDINATION
# =====================================================================

class TestAsyncCoordinationWithFramework:
    """Async coordination tests using the complete framework."""
    
    @pytest_asyncio.async_test
    async def test_concurrent_query_processing(self):
        """Test concurrent query processing with async coordination."""
        async with managed_test_environment(TestScenarioType.ASYNC_TEST) as test_env:
            async_coordinator = test_env['async_coordinator']
            mock_system = test_env.get('mock_system', {})
            
            if not async_coordinator or 'lightrag_system' not in mock_system:
                pytest.skip("Async coordinator or LightRAG mock not available")
            
            lightrag_mock = mock_system['lightrag_system']
            
            # Create async session
            session_id = await async_coordinator.create_session("concurrent_queries")
            
            try:
                # Define concurrent queries
                queries = [
                    "What are diabetes biomarkers?",
                    "How does metabolomics help in cardiovascular disease?",
                    "What metabolic pathways are affected in kidney disease?",
                    "Clinical applications of proteomics in cancer research"
                ]
                
                # Execute queries concurrently
                start_time = time.time()
                
                async def execute_query(query):
                    return await lightrag_mock.aquery(query)
                
                results = await asyncio.gather(
                    *[execute_query(query) for query in queries],
                    return_exceptions=True
                )
                
                total_duration = time.time() - start_time
                
                # Validate results
                assert len(results) == len(queries)
                
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) >= 3, f"At least 3 queries should succeed, got {len(successful_results)}"
                
                # Validate content quality
                for result in successful_results:
                    assert isinstance(result, str)
                    assert len(result) > 50
                
                # Performance validation - concurrent should be faster than sequential
                estimated_sequential_time = len(queries) * 0.2  # Assume 0.2s per query
                efficiency_ratio = estimated_sequential_time / total_duration
                assert efficiency_ratio > 1.5, f"Concurrent execution should be more efficient, ratio: {efficiency_ratio:.2f}"
                
            finally:
                await async_coordinator.cancel_session(session_id)
    
    @pytest_asyncio.async_test
    async def test_async_resource_cleanup(self):
        """Test async resource cleanup coordination."""
        resource_ids = []
        
        async with managed_test_environment(TestScenarioType.ASYNC_TEST) as test_env:
            cleanup_manager = test_env['cleanup_manager']
            
            if not cleanup_manager:
                pytest.skip("Cleanup manager not available")
            
            # Create multiple async tasks
            async def long_running_task(task_id: int):
                try:
                    await asyncio.sleep(0.5)
                    return f"Task {task_id} completed"
                except asyncio.CancelledError:
                    return f"Task {task_id} cancelled"
            
            # Register tasks with cleanup manager
            tasks = []
            for i in range(5):
                task = asyncio.create_task(long_running_task(i))
                tasks.append(task)
                
                resource_id = cleanup_manager.register_async_task(task, cleanup_priority=i)
                resource_ids.append(resource_id)
            
            # Let some tasks run
            await asyncio.sleep(0.1)
            
            # Get initial stats
            initial_stats = cleanup_manager.get_cleanup_statistics()
            assert initial_stats['active_resources'] >= 5
            
        # After exiting context, cleanup should have occurred
        final_stats = cleanup_manager.get_cleanup_statistics()
        assert final_stats['resources_cleaned'] >= 5


# =====================================================================
# PERFORMANCE TESTS WITH VALIDATION
# =====================================================================

class TestPerformanceWithFramework:
    """Performance tests using the complete framework."""
    
    @pytest_asyncio.async_test
    async def test_query_performance_validation(self, standard_performance_test_config):
        """Test query performance with validation framework."""
        test_env = standard_performance_test_config
        
        performance_helper = test_env['performance_helper']
        mock_system = test_env.get('mock_system', {})
        
        if not performance_helper or 'lightrag_system' not in mock_system:
            pytest.skip("Performance helper or LightRAG mock not available")
        
        lightrag_mock = mock_system['lightrag_system']
        
        # Test multiple queries for statistical validation
        query_times = []
        memory_usage = []
        
        for i in range(10):
            # Measure memory before query
            if test_env['environment_manager']:
                health_before = test_env['environment_manager'].check_system_health()
                memory_before = health_before['memory_usage_mb']
            else:
                memory_before = 0
            
            # Execute query with timing
            start_time = time.time()
            response = await lightrag_mock.aquery(f"Test query {i} about metabolomics")
            duration = time.time() - start_time
            
            # Measure memory after query
            if test_env['environment_manager']:
                health_after = test_env['environment_manager'].check_system_health()
                memory_after = health_after['memory_usage_mb']
                memory_delta = memory_after - memory_before
            else:
                memory_delta = 0
            
            query_times.append(duration)
            memory_usage.append(memory_delta)
            
            # Validate individual query
            assert response is not None
            assert duration < 2.0, f"Query {i} too slow: {duration:.3f}s"
        
        # Statistical performance validation
        import statistics
        
        avg_time = statistics.mean(query_times)
        max_time = max(query_times)
        avg_memory = statistics.mean(memory_usage)
        
        # Performance assertions
        assert avg_time < 0.5, f"Average query time too high: {avg_time:.3f}s"
        assert max_time < 1.0, f"Maximum query time too high: {max_time:.3f}s"
        assert avg_memory < 10.0, f"Average memory usage per query too high: {avg_memory:.1f}MB"
        
        # Validate performance consistency (coefficient of variation)
        if len(query_times) > 1:
            time_std = statistics.stdev(query_times)
            time_cv = time_std / avg_time
            assert time_cv < 0.5, f"Query time too inconsistent, CV: {time_cv:.3f}"
    
    @pytest_asyncio.async_test
    async def test_resource_usage_monitoring(self):
        """Test resource usage monitoring during test execution."""
        async with managed_test_environment(TestScenarioType.PERFORMANCE_TEST) as test_env:
            environment_manager = test_env['environment_manager']
            cleanup_manager = test_env['cleanup_manager']
            
            if not environment_manager or not cleanup_manager:
                pytest.skip("Environment or cleanup manager not available")
            
            # Get initial resource state
            initial_health = environment_manager.check_system_health()
            initial_cleanup_stats = cleanup_manager.get_cleanup_statistics()
            
            # Simulate resource-intensive operations
            large_data = []
            temp_files = []
            
            try:
                # Create temporary resources
                for i in range(5):
                    # Create some memory usage
                    large_data.append([0] * 10000)
                    
                    # Create temporary files
                    temp_file = Path(test_env['working_dir']) / f"temp_{i}.txt"
                    temp_file.write_text("x" * 10000)  # 10KB file
                    temp_files.append(temp_file)
                    
                    # Register for cleanup
                    cleanup_manager.register_temporary_file(temp_file)
                
                # Monitor resource usage
                mid_health = environment_manager.check_system_health()
                mid_cleanup_stats = cleanup_manager.get_cleanup_statistics()
                
                # Validate resource increase
                memory_increase = mid_health['memory_usage_mb'] - initial_health['memory_usage_mb']
                resources_increase = mid_cleanup_stats['active_resources'] - initial_cleanup_stats['active_resources']
                
                assert resources_increase >= 5, f"Should track at least 5 new resources, got {resources_increase}"
                
                # Check that temp files exist
                for temp_file in temp_files:
                    assert temp_file.exists(), f"Temporary file should exist: {temp_file}"
                
            finally:
                # Clear large data
                large_data.clear()
            
            # Force cleanup
            cleanup_manager.cleanup_all_resources()
            
            # Validate cleanup
            final_cleanup_stats = cleanup_manager.get_cleanup_statistics()
            assert final_cleanup_stats['resources_cleaned'] >= 5
            
            # Check that temp files are cleaned up
            for temp_file in temp_files:
                assert not temp_file.exists(), f"Temporary file should be cleaned up: {temp_file}"


# =====================================================================
# COMPLETE INTEGRATION TEST
# =====================================================================

class TestCompleteIntegrationScenario:
    """Complete integration test using all framework components."""
    
    @pytest_asyncio.async_test
    async def test_complete_clinical_metabolomics_workflow(self):
        """Test complete clinical metabolomics workflow using entire framework."""
        # Use the convenience function to create complete environment
        test_env = create_complete_test_environment(
            TestScenarioType.BIOMEDICAL_TEST,
            custom_overrides={
                'environment_vars': {
                    'CLINICAL_MODE': 'true',
                    'VALIDATION_LEVEL': 'strict'
                },
                'performance_thresholds': {
                    'workflow_execution': PerformanceThreshold(
                        metric_name='workflow_execution',
                        threshold_value=30.0,
                        comparison_operator='lt',
                        unit='seconds'
                    )
                }
            }
        )
        
        try:
            # Validate initial configuration
            validation_errors = validate_test_configuration(test_env)
            assert not validation_errors, f"Initial configuration invalid: {validation_errors}"
            
            # Get components
            mock_system = test_env.get('mock_system', {})
            environment_manager = test_env['environment_manager']
            cleanup_manager = test_env['cleanup_manager']
            
            # Simulate complete workflow
            workflow_start = time.time()
            
            # Step 1: PDF Processing
            if 'pdf_processor' in mock_system:
                pdf_processor = mock_system['pdf_processor']
                
                # Process multiple PDFs concurrently
                pdf_files = [
                    "diabetes_metabolomics_study.pdf",
                    "cardiovascular_biomarkers.pdf",
                    "kidney_disease_metabolites.pdf"
                ]
                
                pdf_results = []
                for pdf_file in pdf_files:
                    result = await pdf_processor.process_pdf(pdf_file)
                    pdf_results.append(result)
                    
                    # Validate PDF result
                    assert result['success']
                    assert len(result['text']) > 100
                
                print(f"✓ Processed {len(pdf_results)} PDF files")
            
            # Step 2: Knowledge Base Population
            if 'lightrag_system' in mock_system:
                lightrag = mock_system['lightrag_system']
                
                # Insert documents
                documents = [result['text'] for result in pdf_results] if 'pdf_results' in locals() else [
                    "Clinical metabolomics study on diabetes patients",
                    "Cardiovascular biomarker analysis using LC-MS",
                    "Kidney disease metabolite profiling study"
                ]
                
                insert_result = await lightrag.ainsert(documents)
                assert insert_result['status'] == 'success'
                assert insert_result['documents_processed'] == len(documents)
                
                print(f"✓ Inserted {len(documents)} documents into knowledge base")
            
            # Step 3: Query Processing
            clinical_queries = [
                "What metabolites are elevated in diabetes patients?",
                "How do cardiovascular biomarkers correlate with metabolic dysfunction?",
                "What metabolic pathways are disrupted in kidney disease?"
            ]
            
            query_results = []
            for query in clinical_queries:
                if 'lightrag_system' in mock_system:
                    response = await mock_system['lightrag_system'].aquery(query)
                    query_results.append(response)
                    
                    # Validate response quality
                    assert len(response) > 100
                    
                    # Check for clinical content
                    response_lower = response.lower()
                    clinical_terms = ['metabolite', 'biomarker', 'clinical', 'patient']
                    found_terms = [term for term in clinical_terms if term in response_lower]
                    assert len(found_terms) >= 2, f"Query response should contain clinical terms"
            
            print(f"✓ Processed {len(query_results)} clinical queries")
            
            # Step 4: Resource and Performance Validation
            workflow_duration = time.time() - workflow_start
            
            # Check workflow performance
            assert workflow_duration < 30.0, f"Complete workflow too slow: {workflow_duration:.3f}s"
            
            # Check resource usage
            if environment_manager:
                final_health = environment_manager.check_system_health()
                assert final_health['memory_usage_mb'] < 512, f"Memory usage too high: {final_health['memory_usage_mb']}MB"
            
            # Check cleanup management
            if cleanup_manager:
                cleanup_stats = cleanup_manager.get_cleanup_statistics()
                print(f"✓ Cleanup manager tracked {cleanup_stats['active_resources']} resources")
            
            print(f"✓ Complete clinical metabolomics workflow succeeded in {workflow_duration:.3f}s")
            
        finally:
            # Cleanup is handled automatically by the configuration helper
            if test_env['config_helper']:
                test_env['config_helper'].cleanup_configuration(test_env['context_id'], force=True)


# =====================================================================
# EXAMPLE TEST EXECUTION
# =====================================================================

if __name__ == "__main__":
    """
    Example of how to run these tests.
    
    Run with pytest:
    pytest example_complete_test_framework.py -v -s
    
    Or run individual test classes:
    pytest example_complete_test_framework.py::TestCompleteIntegrationScenario -v -s
    """
    import sys
    print("This is an example test file showing how to use the complete test framework.")
    print("Run with pytest to execute the tests:")
    print("  pytest example_complete_test_framework.py -v -s")
    sys.exit(0)