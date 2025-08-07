#!/usr/bin/env python3
"""
Test suite to verify async testing configuration is working correctly.

This module contains tests to verify that:
1. pytest-asyncio is properly configured
2. Async fixtures are working correctly
3. Event loop management is functioning
4. Async test markers are properly recognized
5. Concurrent async operations work as expected

Author: Claude Code (Anthropic)
Created: August 7, 2025
Task: CMO-LIGHTRAG-008-T01 - Set up pytest configuration for async testing
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock


@pytest.mark.asyncio
@pytest.mark.unit
async def test_basic_async_functionality():
    """Test basic async functionality with pytest-asyncio."""
    
    async def async_operation(delay: float = 0.1, result: str = "success"):
        """Simple async operation for testing."""
        await asyncio.sleep(delay)
        return result
    
    # Test single async operation
    result = await async_operation(0.01, "test_result")
    assert result == "test_result"
    
    # Test multiple concurrent operations
    tasks = [
        async_operation(0.01, f"result_{i}") 
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result == f"result_{i}"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_test_context_fixture(async_test_context):
    """Test async test context fixture is working."""
    
    assert 'start_time' in async_test_context
    assert 'tasks' in async_test_context
    assert 'cleanup_callbacks' in async_test_context
    
    # Test adding a task to the context
    async def sample_task():
        await asyncio.sleep(0.01)
        return "task_complete"
    
    task = asyncio.create_task(sample_task())
    async_test_context['tasks'].append(task)
    
    result = await task
    assert result == "task_complete"


@pytest.mark.asyncio
@pytest.mark.unit  
async def test_async_mock_lightrag_fixture(async_mock_lightrag):
    """Test async mock LightRAG fixture is working."""
    
    # Test ainsert method
    insert_result = await async_mock_lightrag.ainsert("test document content")
    assert insert_result['status'] == 'success'
    assert 'cost' in insert_result
    
    # Test aquery method
    query_result = await async_mock_lightrag.aquery("test query", mode="hybrid")
    assert isinstance(query_result, str)
    assert "Mock response" in query_result
    
    # Test adelete method
    delete_result = await async_mock_lightrag.adelete("test_id")
    assert delete_result['status'] == 'deleted'
    
    # Verify mock was called
    async_mock_lightrag.ainsert.assert_called_once()
    async_mock_lightrag.aquery.assert_called_once()
    async_mock_lightrag.adelete.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_cost_tracker_fixture(async_cost_tracker):
    """Test async cost tracker fixture is working."""
    
    # Track some costs
    record1 = await async_cost_tracker.track_cost("test_operation_1", 0.05)
    record2 = await async_cost_tracker.track_cost("test_operation_2", 0.03, model="gpt-4o-mini")
    
    assert record1['operation'] == "test_operation_1"
    assert record1['cost'] == 0.05
    assert 'timestamp' in record1
    
    assert record2['operation'] == "test_operation_2"
    assert record2['cost'] == 0.03
    assert record2['model'] == "gpt-4o-mini"
    
    # Check total cost
    total = await async_cost_tracker.get_total()
    assert total == 0.08
    
    # Check all costs
    costs = await async_cost_tracker.get_costs()
    assert len(costs) == 2
    
    # Test reset
    await async_cost_tracker.reset()
    total_after_reset = await async_cost_tracker.get_total()
    assert total_after_reset == 0.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_progress_monitor_fixture(async_progress_monitor):
    """Test async progress monitor fixture is working."""
    
    # Test initial state
    summary = await async_progress_monitor.get_summary()
    assert summary['current_progress'] == 0.0
    assert summary['current_status'] == "initialized"
    assert summary['total_events'] == 0
    
    # Update progress
    event1 = await async_progress_monitor.update(25.0, "processing")
    assert event1['progress'] == 25.0
    assert event1['status'] == "processing"
    
    event2 = await async_progress_monitor.update(75.0, "nearly_complete")
    assert event2['progress'] == 75.0
    
    # Check summary
    summary = await async_progress_monitor.get_summary()
    assert summary['current_progress'] == 75.0
    assert summary['current_status'] == "nearly_complete"
    assert summary['total_events'] == 2
    
    # Test completion
    await async_progress_monitor.update(100.0, "completed")
    completed = await async_progress_monitor.wait_for_completion(timeout=1.0)
    assert completed is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_async_operations(async_cost_tracker, async_progress_monitor):
    """Test concurrent async operations with multiple fixtures."""
    
    async def simulate_lightrag_operation(operation_id: int, cost: float):
        """Simulate a LightRAG operation with cost tracking and progress."""
        
        # Start operation
        await async_progress_monitor.update(
            operation_id * 20, 
            f"processing_operation_{operation_id}"
        )
        
        # Simulate processing delay
        await asyncio.sleep(0.01 * operation_id)
        
        # Track cost
        await async_cost_tracker.track_cost(
            f"lightrag_operation_{operation_id}",
            cost,
            operation_id=operation_id
        )
        
        # Complete operation
        await async_progress_monitor.update(
            (operation_id + 1) * 20,
            f"completed_operation_{operation_id}"
        )
        
        return f"operation_{operation_id}_result"
    
    # Run multiple operations concurrently
    operations = [
        simulate_lightrag_operation(i, 0.01 * (i + 1))
        for i in range(5)
    ]
    
    results = await asyncio.gather(*operations)
    
    # Verify results
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result == f"operation_{i}_result"
    
    # Check cost tracking
    total_cost = await async_cost_tracker.get_total()
    expected_cost = sum(0.01 * (i + 1) for i in range(5))
    assert abs(total_cost - expected_cost) < 0.001
    
    # Check progress tracking
    summary = await async_progress_monitor.get_summary()
    assert summary['current_progress'] == 100.0  # 5 * 20
    assert summary['total_events'] == 10  # 2 events per operation


@pytest.mark.asyncio
@pytest.mark.performance
async def test_async_performance_timing(async_timeout):
    """Test async performance and timing constraints."""
    
    start_time = time.time()
    
    async def timed_operation(duration: float):
        await asyncio.sleep(duration)
        return time.time()
    
    # Test that operations run concurrently
    tasks = [timed_operation(0.1) for _ in range(5)]
    end_times = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Should complete in ~0.1 seconds (concurrent), not ~0.5 seconds (sequential)
    assert total_time < 0.2, f"Operations took {total_time:.3f}s, expected < 0.2s"
    
    # All operations should complete around the same time
    time_spread = max(end_times) - min(end_times)
    assert time_spread < 0.05, f"Time spread {time_spread:.3f}s too large"
    
    # Verify we're within the async timeout
    assert total_time < async_timeout


@pytest.mark.asyncio
@pytest.mark.unit
async def test_exception_handling_in_async_tests():
    """Test that exceptions are properly handled in async tests."""
    
    async def failing_operation():
        await asyncio.sleep(0.01)
        raise ValueError("Test exception")
    
    # Test that exceptions are properly raised
    with pytest.raises(ValueError, match="Test exception"):
        await failing_operation()
    
    # Test exception handling in concurrent operations
    async def mixed_operations(should_fail: bool):
        await asyncio.sleep(0.01)
        if should_fail:
            raise RuntimeError("Intentional failure")
        return "success"
    
    # Test gather with return_exceptions=True
    results = await asyncio.gather(
        mixed_operations(False),
        mixed_operations(True),
        mixed_operations(False),
        return_exceptions=True
    )
    
    assert len(results) == 3
    assert results[0] == "success"
    assert isinstance(results[1], RuntimeError)
    assert results[2] == "success"


@pytest.mark.asyncio
@pytest.mark.lightrag
@pytest.mark.biomedical
async def test_biomedical_lightrag_simulation(async_mock_lightrag, async_cost_tracker):
    """Test simulation of biomedical LightRAG operations."""
    
    # Simulate inserting biomedical documents
    biomedical_docs = [
        "Metabolomic analysis of diabetes patients using LC-MS.",
        "Proteomic profiling in cardiovascular disease research.",
        "Genomic variants associated with metabolic disorders."
    ]
    
    insertion_tasks = []
    for doc in biomedical_docs:
        # Configure mock to return document-specific results
        async_mock_lightrag.ainsert.return_value = {
            'status': 'success',
            'cost': len(doc) * 0.0001,  # Cost based on content length
            'entities_extracted': len(doc.split()) // 2,
            'relationships_found': len(doc.split()) // 4
        }
        
        task = async_mock_lightrag.ainsert(doc)
        insertion_tasks.append(task)
    
    # Insert documents concurrently
    insertion_results = await asyncio.gather(*insertion_tasks)
    
    # Track costs
    for i, result in enumerate(insertion_results):
        await async_cost_tracker.track_cost(
            "biomedical_doc_insertion",
            result['cost'],
            document_id=i,
            entities=result['entities_extracted'],
            relationships=result['relationships_found']
        )
    
    # Simulate biomedical queries
    biomedical_queries = [
        "What metabolites are associated with diabetes?",
        "How do protein biomarkers relate to heart disease?",
        "Which genetic variants affect metabolic pathways?"
    ]
    
    query_tasks = []
    for query in biomedical_queries:
        # Configure mock to return query-specific results
        async_mock_lightrag.aquery.return_value = f"Biomedical analysis: {query[:20]}..."
        
        task = async_mock_lightrag.aquery(query, mode="hybrid")
        query_tasks.append(task)
    
    # Execute queries concurrently
    query_results = await asyncio.gather(*query_tasks)
    
    # Track query costs
    for i, result in enumerate(query_results):
        await async_cost_tracker.track_cost(
            "biomedical_query",
            0.02,  # Fixed query cost
            query_id=i,
            response_length=len(result)
        )
    
    # Verify results
    assert len(insertion_results) == 3
    assert len(query_results) == 3
    
    # Verify cost tracking
    total_cost = await async_cost_tracker.get_total()
    assert total_cost > 0.0
    
    costs = await async_cost_tracker.get_costs()
    insertion_costs = [c for c in costs if c['operation'] == 'biomedical_doc_insertion']
    query_costs = [c for c in costs if c['operation'] == 'biomedical_query']
    
    assert len(insertion_costs) == 3
    assert len(query_costs) == 3
    
    # Verify mock calls
    assert async_mock_lightrag.ainsert.call_count == 3
    assert async_mock_lightrag.aquery.call_count == 3


if __name__ == "__main__":
    # This allows the test file to be run directly for development
    pytest.main([__file__, "-v", "--tb=short"])