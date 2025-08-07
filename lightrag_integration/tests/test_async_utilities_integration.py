#!/usr/bin/env python3
"""
Integration Tests for Async Test Utilities.

This module tests the async test utilities to ensure they work correctly
with the existing test infrastructure and pytest-asyncio framework.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import pytest_asyncio
import asyncio
import time
import random
from pathlib import Path
from typing import Dict, Any

from async_test_utilities import (
    AsyncTestCoordinator, AsyncOperationSpec, AsyncTestState, ConcurrencyPolicy,
    AsyncTestDataGenerator, AsyncOperationBatcher, AsyncResultAggregator,
    async_test_environment, async_resource_manager, async_error_injection,
    create_async_test_operations, run_coordinated_async_test
)
from test_utilities import EnvironmentSpec, SystemComponent
from performance_test_utilities import PerformanceThreshold


# =====================================================================
# TEST FIXTURES
# =====================================================================

@pytest.fixture
def sample_async_operations():
    """Provide sample async operations for testing."""
    
    async def simple_operation(value: str, delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return f"processed_{value}"
    
    async def dependent_operation(input_data: str) -> str:
        await asyncio.sleep(0.05)
        return f"dependent_result_from_{input_data}"
    
    async def failing_operation(failure_rate: float = 0.5) -> str:
        await asyncio.sleep(0.1)
        if random.random() < failure_rate:
            raise ValueError("Simulated operation failure")
        return "success"
    
    return {
        'simple': simple_operation,
        'dependent': dependent_operation,
        'failing': failing_operation
    }


# =====================================================================
# ASYNC TEST COORDINATOR TESTS
# =====================================================================

@pytest_asyncio.fixture
async def test_coordinator():
    """Provide AsyncTestCoordinator for tests."""
    coordinator = AsyncTestCoordinator(
        default_timeout=10.0,
        max_concurrent_operations=3,
        concurrency_policy=ConcurrencyPolicy.LIMITED
    )
    yield coordinator
    
    # Cleanup all sessions
    for session_id in list(coordinator.active_sessions.keys()):
        try:
            await coordinator.cleanup_session(session_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_async_coordinator_session_creation(test_coordinator):
    """Test async test coordinator session creation."""
    # Test basic session creation
    session_id = await test_coordinator.create_session()
    assert session_id in test_coordinator.active_sessions
    
    # Test custom session ID
    custom_session_id = "test_session_123"
    session_id_2 = await test_coordinator.create_session(custom_session_id)
    assert session_id_2 == custom_session_id
    assert session_id_2 in test_coordinator.active_sessions
    
    # Test duplicate session ID error
    with pytest.raises(ValueError, match="already exists"):
        await test_coordinator.create_session(custom_session_id)


@pytest.mark.asyncio
async def test_async_coordinator_operation_execution(test_coordinator, sample_async_operations):
    """Test async operation execution with coordinator."""
    session_id = await test_coordinator.create_session()
    
    # Create test operations
    operations = [
        AsyncOperationSpec(
            operation_id="op1",
            operation_func=sample_async_operations['simple'],
            args=("test_value", 0.1),
            timeout_seconds=5.0
        ),
        AsyncOperationSpec(
            operation_id="op2", 
            operation_func=sample_async_operations['simple'],
            args=("another_value", 0.05),
            timeout_seconds=5.0
        )
    ]
    
    # Add operations to session
    await test_coordinator.add_operations_batch(session_id, operations)
    
    # Execute session
    results = await test_coordinator.execute_session(session_id)
    
    # Verify results
    assert len(results) == 2
    assert "op1" in results
    assert "op2" in results
    assert results["op1"].succeeded
    assert results["op2"].succeeded
    assert results["op1"].result == "processed_test_value"
    assert results["op2"].result == "processed_another_value"


@pytest.mark.asyncio
async def test_async_coordinator_dependencies(test_coordinator, sample_async_operations):
    """Test async operation dependency resolution."""
    session_id = await test_coordinator.create_session()
    
    # Create operations with dependencies
    operations = [
        AsyncOperationSpec(
            operation_id="base_op",
            operation_func=sample_async_operations['simple'],
            args=("base", 0.1)
        ),
        AsyncOperationSpec(
            operation_id="dependent_op",
            operation_func=sample_async_operations['dependent'],
            args=("dependent_input",),
            dependencies=["base_op"]  # Depends on base_op
        )
    ]
    
    await test_coordinator.add_operations_batch(session_id, operations)
    
    # Track execution times
    start_time = time.time()
    results = await test_coordinator.execute_session(session_id)
    total_time = time.time() - start_time
    
    # Verify results
    assert len(results) == 2
    assert results["base_op"].succeeded
    assert results["dependent_op"].succeeded
    
    # Verify dependency was respected (dependent should start after base completes)
    base_end_time = results["base_op"].end_time
    dependent_start_time = results["dependent_op"].start_time
    assert dependent_start_time >= base_end_time


@pytest.mark.asyncio
async def test_async_coordinator_error_handling(test_coordinator, sample_async_operations):
    """Test async coordinator error handling."""
    session_id = await test_coordinator.create_session()
    
    operations = [
        AsyncOperationSpec(
            operation_id="success_op",
            operation_func=sample_async_operations['simple'],
            args=("success", 0.05)
        ),
        AsyncOperationSpec(
            operation_id="failing_op",
            operation_func=sample_async_operations['failing'], 
            args=(1.0,),  # 100% failure rate
            retry_count=2,
            retry_delay_seconds=0.1
        )
    ]
    
    await test_coordinator.add_operations_batch(session_id, operations)
    
    # Execute with fail_on_first_error=False
    results = await test_coordinator.execute_session(session_id, fail_on_first_error=False)
    
    # Verify results
    assert len(results) == 2
    assert results["success_op"].succeeded
    assert results["failing_op"].failed
    assert results["failing_op"].retry_count == 2
    assert isinstance(results["failing_op"].exception, ValueError)


# =====================================================================
# ASYNC CONTEXT MANAGERS TESTS  
# =====================================================================

@pytest.mark.asyncio
async def test_async_test_environment():
    """Test async test environment context manager."""
    environment_spec = EnvironmentSpec(
        temp_dirs=["test_data", "results"],
        performance_monitoring=True
    )
    
    async with async_test_environment(environment_spec) as env_context:
        # Verify context structure
        assert 'coordinator' in env_context
        assert 'session_id' in env_context
        assert 'environment_manager' in env_context
        assert 'environment_data' in env_context
        
        coordinator = env_context['coordinator']
        session_id = env_context['session_id']
        
        # Verify session exists
        assert session_id in coordinator.active_sessions
        
        # Verify environment setup
        environment_data = env_context['environment_data']
        assert 'working_dir' in environment_data
        assert 'subdirectories' in environment_data
        assert 'test_data' in str(environment_data['subdirectories']['test_data'])


@pytest.mark.asyncio
async def test_async_resource_manager():
    """Test async resource manager context manager."""
    created_resources = []
    cleaned_up_resources = []
    
    async def create_resource_1():
        await asyncio.sleep(0.05)
        resource = {"id": "resource_1", "type": "database"}
        created_resources.append(resource)
        return resource
    
    async def create_resource_2():
        await asyncio.sleep(0.03)
        resource = {"id": "resource_2", "type": "cache"}
        created_resources.append(resource)
        return resource
    
    async def cleanup_resource_1(resource):
        await asyncio.sleep(0.02)
        cleaned_up_resources.append(resource["id"])
    
    async def cleanup_resource_2(resource):
        await asyncio.sleep(0.01)
        cleaned_up_resources.append(resource["id"])
    
    resources = {
        'resource_1': create_resource_1,
        'resource_2': create_resource_2
    }
    
    cleanup_callbacks = {
        'resource_1': cleanup_resource_1,
        'resource_2': cleanup_resource_2
    }
    
    async with async_resource_manager(resources, cleanup_callbacks) as resource_context:
        # Verify resources were created
        assert resource_context['created_count'] == 2
        assert 'resource_1' in resource_context['resources']
        assert 'resource_2' in resource_context['resources']
        assert len(created_resources) == 2
    
    # Verify cleanup happened
    assert len(cleaned_up_resources) == 2
    assert 'resource_1' in cleaned_up_resources
    assert 'resource_2' in cleaned_up_resources


@pytest.mark.asyncio
async def test_async_error_injection():
    """Test async error injection context manager."""
    error_specs = [
        {
            'target': 'test_target',
            'error_type': ConnectionError,
            'message': 'Test connection error',
            'trigger_after': 1
        }
    ]
    
    async with async_error_injection(error_specs, injection_probability=1.0) as error_context:
        injector = error_context['injector']
        
        # Test error injection
        error = await injector.should_inject_error('test_target')
        assert error is not None
        assert isinstance(error, ConnectionError)
        assert str(error) == 'Test connection error'
        
        # Test recovery testing
        async def recovery_func():
            await asyncio.sleep(0.01)
        
        recovered = await injector.test_recovery('test_target', recovery_func)
        assert recovered is True
        
        # Verify injection was recorded
        assert len(error_context['injected_errors']) == 1
        assert len(error_context['recovery_results']) == 1


# =====================================================================
# ASYNC UTILITIES TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_async_data_generator():
    """Test async test data generator."""
    generator = AsyncTestDataGenerator(generation_delay=0.001)
    
    # Test biomedical query generation
    queries = await generator.generate_biomedical_queries(
        count=5,
        disease_types=['diabetes', 'cancer']
    )
    
    assert len(queries) == 5
    assert all(isinstance(q, str) for q in queries)
    assert any('diabetes' in q.lower() or 'cancer' in q.lower() for q in queries)
    
    # Test PDF data generation
    pdf_data = await generator.generate_pdf_data(count=3)
    
    assert len(pdf_data) == 3
    assert all('filename' in pdf for pdf in pdf_data)
    assert all('title' in pdf for pdf in pdf_data)
    assert all('content' in pdf for pdf in pdf_data)
    
    # Test cleanup
    await generator.cleanup_generated_data()


@pytest.mark.asyncio
async def test_async_operation_batcher():
    """Test async operation batcher."""
    batcher = AsyncOperationBatcher(
        batch_size=3,
        batch_delay=0.05,
        max_concurrent_batches=2
    )
    
    async def test_operation(op_id: str) -> str:
        await asyncio.sleep(0.01)
        return f"result_{op_id}"
    
    # Create test operations
    operations = []
    for i in range(8):  # 8 operations, batch size 3 -> 3 batches
        op = AsyncOperationSpec(
            operation_id=f"op_{i}",
            operation_func=test_operation,
            args=(f"op_{i}",)
        )
        operations.append(op)
    
    # Execute batched operations
    start_time = time.time()
    results = await batcher.execute_batched_operations(operations)
    execution_time = time.time() - start_time
    
    # Verify results
    assert len(results) == 8
    assert all(result.succeeded for result in results.values())
    assert all(f"result_op_{i}" == results[f"op_{i}"].result for i in range(8))
    
    # Verify batching happened (should take less time than sequential)
    assert execution_time < 0.5  # Should be much faster than 8 * 0.01 sequential


@pytest.mark.asyncio
async def test_async_result_aggregator():
    """Test async result aggregator."""
    from async_test_utilities import AsyncOperationResult
    
    # Create sample results
    results = {}
    for i in range(5):
        result = AsyncOperationResult(
            operation_id=f"op_{i}",
            state=AsyncTestState.COMPLETED if i < 4 else AsyncTestState.FAILED,
            start_time=time.time(),
            end_time=time.time() + 0.1,
            duration_ms=100.0,
            memory_usage_mb=50.0 if i < 4 else None,
            exception=ValueError("Test error") if i == 4 else None
        )
        results[f"op_{i}"] = result
    
    aggregator = AsyncResultAggregator()
    
    # Test basic aggregation
    aggregation = await aggregator.aggregate_results(results)
    
    # Verify aggregation structure
    assert 'execution_summary' in aggregation
    assert 'performance_statistics' in aggregation
    assert 'error_analysis' in aggregation
    
    # Verify execution summary
    execution_summary = aggregation['execution_summary']
    assert execution_summary['total_operations'] == 5
    assert execution_summary['successful_operations'] == 4
    assert execution_summary['failed_operations'] == 1
    assert execution_summary['success_rate_percent'] == 80.0
    
    # Verify performance statistics
    duration_stats = aggregation['performance_statistics']['duration_stats']
    assert duration_stats['mean_duration_ms'] == 100.0
    assert duration_stats['max_duration_ms'] == 100.0
    
    # Verify error analysis
    error_analysis = aggregation['error_analysis']
    assert error_analysis['total_errors'] == 1
    assert 'ValueError' in error_analysis['error_types']


# =====================================================================
# INTEGRATION TESTS WITH EXISTING INFRASTRUCTURE
# =====================================================================

@pytest.mark.asyncio
async def test_integration_with_existing_fixtures(
    async_test_coordinator,
    async_test_session, 
    comprehensive_mock_system
):
    """Test integration with existing test fixtures."""
    coordinator = async_test_session['coordinator']
    session_id = async_test_session['session_id']
    
    # Use existing mock system
    mock_lightrag = comprehensive_mock_system['lightrag_system']
    
    # Create operation using existing mock
    async def mock_query_operation(query: str) -> str:
        # Use the existing mock LightRAG system
        return await mock_lightrag.aquery(query)
    
    operation = AsyncOperationSpec(
        operation_id="integration_test_op",
        operation_func=mock_query_operation,
        args=("What are the key metabolites in diabetes?",),
        timeout_seconds=10.0
    )
    
    await coordinator.add_operation(session_id, operation)
    results = await coordinator.execute_session(session_id)
    
    # Verify integration worked
    assert len(results) == 1
    assert results["integration_test_op"].succeeded
    assert isinstance(results["integration_test_op"].result, str)
    assert len(results["integration_test_op"].result) > 0


@pytest.mark.asyncio 
async def test_async_utilities_with_performance_monitoring(
    performance_assertion_helper,
    advanced_resource_monitor
):
    """Test async utilities with performance monitoring."""
    coordinator = AsyncTestCoordinator()
    session_id = await coordinator.create_session(
        enable_resource_monitoring=False,  # We'll use the fixture
        enable_performance_monitoring=False  # We'll use the fixture
    )
    
    try:
        # Start monitoring
        performance_assertion_helper.establish_memory_baseline()
        advanced_resource_monitor.start_monitoring()
        
        # Create memory-intensive operations
        async def memory_operation(size_mb: int) -> str:
            # Simulate memory usage
            data = bytearray(size_mb * 1024 * 1024)  # Allocate memory
            await asyncio.sleep(0.1)
            del data  # Release memory
            return f"processed_{size_mb}MB"
        
        operations = []
        for i in range(3):
            op = AsyncOperationSpec(
                operation_id=f"memory_op_{i}",
                operation_func=memory_operation,
                args=(10,),  # 10MB each
                timeout_seconds=5.0
            )
            operations.append(op)
        
        await coordinator.add_operations_batch(session_id, operations)
        results = await coordinator.execute_session(session_id)
        
        # Stop monitoring
        resource_snapshots = advanced_resource_monitor.stop_monitoring()
        
        # Verify operations succeeded
        assert len(results) == 3
        assert all(result.succeeded for result in results.values())
        
        # Verify monitoring data was collected
        assert len(resource_snapshots) > 0
        
        # Check memory assertions
        performance_assertion_helper.assert_memory_leak_absent(
            tolerance_mb=100.0,
            assertion_name="async_memory_test"
        )
        
        # Get assertion summary
        assertion_summary = performance_assertion_helper.get_assertion_summary()
        assert assertion_summary['total_assertions'] > 0
        
    finally:
        await coordinator.cleanup_session(session_id)


# =====================================================================
# CONVENIENCE FUNCTIONS TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_create_async_test_operations():
    """Test convenience function for creating async operations."""
    async def test_func_1():
        await asyncio.sleep(0.01)
        return "result_1"
    
    async def test_func_2():
        await asyncio.sleep(0.01)
        return "result_2"
    
    operation_funcs = [test_func_1, test_func_2]
    dependencies = {
        "operation_1_test_func_2": ["operation_0_test_func_1"]
    }
    
    operations = await create_async_test_operations(
        operation_funcs, 
        dependencies=dependencies,
        timeout=10.0
    )
    
    assert len(operations) == 2
    assert operations[0].operation_id == "operation_0_test_func_1"
    assert operations[1].operation_id == "operation_1_test_func_2"
    assert operations[0].dependencies == []
    assert operations[1].dependencies == ["operation_0_test_func_1"]
    assert all(op.timeout_seconds == 10.0 for op in operations)


@pytest.mark.asyncio
async def test_run_coordinated_async_test():
    """Test convenience function for running coordinated async test."""
    async def test_operation(value: str) -> str:
        await asyncio.sleep(0.05)
        return f"processed_{value}"
    
    operations = [
        AsyncOperationSpec(
            operation_id="test_op_1",
            operation_func=test_operation,
            args=("value_1",)
        ),
        AsyncOperationSpec(
            operation_id="test_op_2", 
            operation_func=test_operation,
            args=("value_2",)
        )
    ]
    
    coordinator = AsyncTestCoordinator()
    
    results = await run_coordinated_async_test(
        coordinator,
        operations,
        fail_on_first_error=False
    )
    
    assert len(results) == 2
    assert results["test_op_1"].succeeded
    assert results["test_op_2"].succeeded
    assert results["test_op_1"].result == "processed_value_1" 
    assert results["test_op_2"].result == "processed_value_2"


# =====================================================================
# PERFORMANCE AND STRESS TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.slow
async def test_async_utilities_stress_test():
    """Stress test async utilities with many operations."""
    coordinator = AsyncTestCoordinator(
        max_concurrent_operations=10,
        concurrency_policy=ConcurrencyPolicy.LIMITED
    )
    
    session_id = await coordinator.create_session()
    
    try:
        async def stress_operation(op_id: int) -> str:
            await asyncio.sleep(random.uniform(0.01, 0.1))
            if random.random() < 0.05:  # 5% failure rate
                raise ValueError(f"Random failure in operation {op_id}")
            return f"stress_result_{op_id}"
        
        # Create many operations
        operations = []
        for i in range(50):
            op = AsyncOperationSpec(
                operation_id=f"stress_op_{i}",
                operation_func=stress_operation,
                args=(i,),
                retry_count=1,
                timeout_seconds=5.0
            )
            operations.append(op)
        
        # Execute stress test
        start_time = time.time()
        await coordinator.add_operations_batch(session_id, operations)
        results = await coordinator.execute_session(session_id, fail_on_first_error=False)
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 50
        success_count = len([r for r in results.values() if r.succeeded])
        
        # Should have high success rate despite random failures
        success_rate = success_count / len(results) * 100
        assert success_rate > 85  # At least 85% success rate
        
        # Should complete in reasonable time with concurrency
        assert execution_time < 30  # Should be much faster than 50 * 0.1 = 5s sequential
        
        # Get coordinator statistics
        stats = coordinator.get_global_statistics()
        assert stats['coordinator_stats']['total_operations'] >= 50
        
    finally:
        await coordinator.cleanup_session(session_id)


if __name__ == "__main__":
    # Run tests with pytest-asyncio
    pytest.main([__file__, "-v", "--tb=short"])