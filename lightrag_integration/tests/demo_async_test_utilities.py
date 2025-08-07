#!/usr/bin/env python3
"""
Demonstration of Async Test Utilities Usage.

This file demonstrates how to use the comprehensive async test utilities
for complex async testing scenarios in the Clinical Metabolomics Oracle project.

Examples include:
1. Basic async test coordination with dependencies
2. Async context managers for resource management
3. Error injection and recovery testing
4. Performance monitoring during async operations
5. Batched operation execution
6. Result aggregation and analysis

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import asyncio
import random
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Import async utilities
from async_test_utilities import (
    AsyncTestCoordinator, AsyncOperationSpec, AsyncTestState, ConcurrencyPolicy,
    AsyncTestDataGenerator, AsyncOperationBatcher, AsyncResultAggregator, AsyncRetryManager,
    async_test_environment, async_resource_manager, async_error_injection, async_performance_monitor,
    create_async_test_operations, run_coordinated_async_test
)

# Import existing test infrastructure
from test_utilities import EnvironmentSpec, SystemComponent
from performance_test_utilities import PerformanceThreshold

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# DEMO OPERATION FUNCTIONS
# =====================================================================

async def mock_pdf_processing(pdf_path: str, complexity: str = "medium") -> Dict[str, Any]:
    """Mock async PDF processing operation."""
    processing_time = {
        "simple": 0.5,
        "medium": 1.5, 
        "complex": 3.0
    }.get(complexity, 1.5)
    
    logger.info(f"Processing PDF: {pdf_path} (complexity: {complexity})")
    await asyncio.sleep(processing_time)
    
    return {
        'pdf_path': pdf_path,
        'processing_time': processing_time,
        'extracted_text_length': random.randint(1000, 10000),
        'metadata': {
            'pages': random.randint(5, 25),
            'complexity': complexity
        }
    }


async def mock_knowledge_base_insertion(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock async knowledge base insertion operation."""
    insertion_time = len(extracted_data.get('extracted_text_length', 1000)) / 1000
    
    logger.info(f"Inserting data into knowledge base (estimated time: {insertion_time:.2f}s)")
    await asyncio.sleep(insertion_time)
    
    return {
        'insertion_id': f"kb_{random.randint(1000, 9999)}",
        'insertion_time': insertion_time,
        'entities_extracted': random.randint(10, 100),
        'relationships_created': random.randint(5, 50)
    }


async def mock_query_processing(query: str, kb_data: Dict[str, Any] = None) -> str:
    """Mock async query processing operation."""
    query_time = len(query) / 100 + random.uniform(0.5, 2.0)
    
    logger.info(f"Processing query: {query[:50]}...")
    await asyncio.sleep(query_time)
    
    return f"Mock response for query: {query}. Based on {kb_data.get('entities_extracted', 0) if kb_data else 0} entities in knowledge base."


async def unreliable_operation(operation_id: str, failure_rate: float = 0.3) -> str:
    """Mock operation that fails randomly for error injection testing."""
    await asyncio.sleep(0.5)
    
    if random.random() < failure_rate:
        raise ValueError(f"Simulated failure in {operation_id}")
    
    return f"Success result from {operation_id}"


# =====================================================================
# BASIC ASYNC COORDINATION DEMO
# =====================================================================

async def demo_basic_async_coordination():
    """Demonstrate basic async test coordination with dependencies."""
    logger.info("=== Demo: Basic Async Coordination with Dependencies ===")
    
    coordinator = AsyncTestCoordinator(
        default_timeout=30.0,
        max_concurrent_operations=3,
        concurrency_policy=ConcurrencyPolicy.LIMITED
    )
    
    try:
        # Create session
        session_id = await coordinator.create_session()
        
        # Define operations with dependencies
        pdf_operation = AsyncOperationSpec(
            operation_id="pdf_processing",
            operation_func=mock_pdf_processing,
            args=("test_paper.pdf", "complex"),
            timeout_seconds=10.0
        )
        
        kb_operation = AsyncOperationSpec(
            operation_id="kb_insertion", 
            operation_func=mock_knowledge_base_insertion,
            dependencies=["pdf_processing"],  # Depends on PDF processing
            timeout_seconds=10.0
        )
        
        query1_operation = AsyncOperationSpec(
            operation_id="query1",
            operation_func=mock_query_processing,
            args=("What are the key metabolites in diabetes?",),
            dependencies=["kb_insertion"],  # Depends on KB insertion
            timeout_seconds=5.0
        )
        
        query2_operation = AsyncOperationSpec(
            operation_id="query2", 
            operation_func=mock_query_processing,
            args=("How does glucose metabolism change in disease?",),
            dependencies=["kb_insertion"],  # Also depends on KB insertion
            timeout_seconds=5.0
        )
        
        # Add operations to session
        operations = [pdf_operation, kb_operation, query1_operation, query2_operation]
        await coordinator.add_operations_batch(session_id, operations)
        
        # Execute session with progress tracking
        async def progress_callback(session_id: str, progress: float):
            logger.info(f"Session {session_id} progress: {progress:.1f}%")
        
        results = await coordinator.execute_session(
            session_id, 
            fail_on_first_error=False,
            progress_callback=progress_callback
        )
        
        # Display results
        logger.info(f"Execution completed with {len(results)} results:")
        for op_id, result in results.items():
            status = "SUCCESS" if result.succeeded else "FAILED"
            logger.info(f"  {op_id}: {status} ({result.duration_ms:.1f}ms)")
        
        # Get session statistics
        session_status = await coordinator.get_session_status(session_id)
        logger.info(f"Session statistics: {session_status['success_rate']:.1f}% success rate")
        
    finally:
        # Cleanup
        await coordinator.cleanup_session(session_id)


# =====================================================================
# ASYNC CONTEXT MANAGERS DEMO
# =====================================================================

async def demo_async_context_managers():
    """Demonstrate async context managers for resource management."""
    logger.info("=== Demo: Async Context Managers ===")
    
    # Demo 1: Async test environment
    environment_spec = EnvironmentSpec(
        temp_dirs=["async_test_data", "results"],
        performance_monitoring=True
    )
    
    async with async_test_environment(environment_spec, session_timeout=60.0) as env_context:
        coordinator = env_context['coordinator']
        session_id = env_context['session_id']
        
        logger.info(f"Environment created with session: {session_id}")
        
        # Demo 2: Resource management
        async def create_test_database():
            await asyncio.sleep(0.5)  # Simulate DB creation
            return {"db_connection": "mock_db", "tables_created": 5}
        
        async def create_cache_system():
            await asyncio.sleep(0.3)  # Simulate cache setup
            return {"cache_connection": "mock_cache", "cache_size": "100MB"}
        
        resources = {
            'database': create_test_database,
            'cache': create_cache_system
        }
        
        async def cleanup_database(db):
            logger.info(f"Cleaning up database: {db}")
            await asyncio.sleep(0.2)
        
        async def cleanup_cache(cache):
            logger.info(f"Cleaning up cache: {cache}")
            await asyncio.sleep(0.1)
        
        cleanup_callbacks = {
            'database': cleanup_database,
            'cache': cleanup_cache
        }
        
        async with async_resource_manager(resources, cleanup_callbacks) as resource_context:
            logger.info(f"Created {resource_context['created_count']} resources")
            
            # Demo 3: Performance monitoring
            performance_thresholds = {
                'memory_usage_mb': PerformanceThreshold(
                    'memory_usage_mb', 500, 'lte', 'MB', 'warning',
                    'Memory should stay under 500MB during test'
                )
            }
            
            async with async_performance_monitor(
                performance_thresholds, 
                sampling_interval=0.5
            ) as perf_context:
                
                # Simulate some work that uses resources
                for i in range(3):
                    operation = AsyncOperationSpec(
                        operation_id=f"resource_operation_{i}",
                        operation_func=mock_query_processing,
                        args=(f"Test query {i}",)
                    )
                    await coordinator.add_operation(session_id, operation)
                
                # Execute operations
                results = await coordinator.execute_session(session_id)
                
                # Check performance data
                logger.info(f"Performance samples collected: {len(perf_context['performance_samples'])}")
                logger.info(f"Threshold breaches: {len(perf_context['threshold_breaches'])}")


# =====================================================================
# ERROR INJECTION AND RECOVERY DEMO
# =====================================================================

async def demo_error_injection_recovery():
    """Demonstrate error injection and recovery testing."""
    logger.info("=== Demo: Error Injection and Recovery Testing ===")
    
    # Define error injection specs
    error_specs = [
        {
            'target': 'unreliable_op_1',
            'error_type': ConnectionError,
            'message': 'Simulated connection failure',
            'trigger_after': 1
        },
        {
            'target': 'unreliable_op_2', 
            'error_type': TimeoutError,
            'message': 'Simulated timeout',
            'trigger_after': 1
        }
    ]
    
    async with async_error_injection(
        error_specs, 
        injection_probability=0.8,
        recovery_testing=True
    ) as error_context:
        
        coordinator = AsyncTestCoordinator()
        session_id = await coordinator.create_session()
        
        try:
            # Create operations that might fail
            operations = []
            for i in range(3):
                op = AsyncOperationSpec(
                    operation_id=f"unreliable_op_{i}",
                    operation_func=unreliable_operation,
                    args=(f"unreliable_op_{i}", 0.4),
                    retry_count=2,
                    retry_delay_seconds=0.5
                )
                operations.append(op)
            
            await coordinator.add_operations_batch(session_id, operations)
            
            # Execute with error injection
            results = await coordinator.execute_session(session_id, fail_on_first_error=False)
            
            # Analyze error injection results
            injector = error_context['injector']
            injected_errors = error_context['injected_errors']
            recovery_results = error_context['recovery_results']
            
            logger.info(f"Operations executed: {len(results)}")
            logger.info(f"Errors injected: {len(injected_errors)}")
            logger.info(f"Recovery tests: {len(recovery_results)}")
            
            # Test recovery for each failed operation
            for op_id, result in results.items():
                if result.failed:
                    async def recovery_func():
                        logger.info(f"Attempting recovery for {op_id}")
                        await asyncio.sleep(0.2)  # Simulate recovery
                    
                    recovered = await injector.test_recovery(op_id, recovery_func)
                    logger.info(f"Recovery for {op_id}: {'SUCCESS' if recovered else 'FAILED'}")
        
        finally:
            await coordinator.cleanup_session(session_id)


# =====================================================================
# BATCHED OPERATIONS DEMO
# =====================================================================

async def demo_batched_operations():
    """Demonstrate batched async operation execution."""
    logger.info("=== Demo: Batched Async Operations ===")
    
    # Create data generator
    data_generator = AsyncTestDataGenerator()
    
    # Generate test queries
    queries = await data_generator.generate_biomedical_queries(
        count=20, 
        disease_types=['diabetes', 'cardiovascular', 'cancer']
    )
    
    # Create operations for all queries
    operations = []
    for i, query in enumerate(queries):
        operation = AsyncOperationSpec(
            operation_id=f"query_op_{i}",
            operation_func=mock_query_processing,
            args=(query,),
            timeout_seconds=10.0
        )
        operations.append(operation)
    
    # Execute operations in batches
    batcher = AsyncOperationBatcher(
        batch_size=5,
        batch_delay=0.2,
        max_concurrent_batches=3
    )
    
    async def progress_callback(progress: float):
        logger.info(f"Batch execution progress: {progress:.1f}%")
    
    results = await batcher.execute_batched_operations(
        operations, 
        progress_callback=progress_callback
    )
    
    # Aggregate results
    aggregator = AsyncResultAggregator()
    aggregation = await aggregator.aggregate_results(results)
    
    logger.info(f"Batched execution summary:")
    logger.info(f"  Total operations: {aggregation['execution_summary']['total_operations']}")
    logger.info(f"  Success rate: {aggregation['execution_summary']['success_rate_percent']:.1f}%")
    logger.info(f"  Average duration: {aggregation['performance_statistics']['duration_stats'].get('mean_duration_ms', 0):.1f}ms")
    
    # Export results to file
    output_file = Path("async_batch_results.json")
    await aggregator.export_results(aggregation, output_file, include_raw_results=True, raw_results=results)
    logger.info(f"Results exported to {output_file}")


# =====================================================================
# RETRY MECHANISMS DEMO
# =====================================================================

async def demo_retry_mechanisms():
    """Demonstrate async retry mechanisms."""
    logger.info("=== Demo: Async Retry Mechanisms ===")
    
    retry_manager = AsyncRetryManager()
    
    # Demo 1: Exponential backoff retry
    async def flaky_operation():
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Connection failed")
        return "Success after retries!"
    
    try:
        result = await retry_manager.retry_with_exponential_backoff(
            flaky_operation,
            max_retries=3,
            initial_delay=0.5,
            backoff_multiplier=2.0,
            exceptions=(ConnectionError,)
        )
        logger.info(f"Exponential backoff result: {result}")
    except ConnectionError as e:
        logger.error(f"Operation failed permanently: {e}")
    
    # Demo 2: Jitter retry
    async def another_flaky_operation():
        if random.random() < 0.6:  # 60% failure rate
            raise TimeoutError("Request timed out")
        return "Success with jitter!"
    
    try:
        result = await retry_manager.retry_with_jitter(
            another_flaky_operation,
            max_retries=4,
            base_delay=0.3,
            max_jitter=0.5,
            exceptions=(TimeoutError,)
        )
        logger.info(f"Jitter retry result: {result}")
    except TimeoutError as e:
        logger.error(f"Operation failed permanently: {e}")


# =====================================================================
# COMPREHENSIVE WORKFLOW DEMO
# =====================================================================

async def demo_comprehensive_workflow():
    """Demonstrate comprehensive async testing workflow."""
    logger.info("=== Demo: Comprehensive Async Testing Workflow ===")
    
    # Set up environment with all monitoring
    environment_spec = EnvironmentSpec(
        temp_dirs=["workflow_data", "results", "logs"],
        performance_monitoring=True,
        cleanup_on_exit=True
    )
    
    async with async_test_environment(environment_spec, session_timeout=120.0) as env_context:
        coordinator = env_context['coordinator']
        session_id = env_context['session_id']
        
        # Generate comprehensive test data
        data_generator = AsyncTestDataGenerator()
        pdf_data = await data_generator.generate_pdf_data(count=5)
        queries = await data_generator.generate_biomedical_queries(count=10)
        
        # Create comprehensive workflow operations
        workflow_operations = []
        
        # Phase 1: PDF processing operations
        for i, pdf in enumerate(pdf_data):
            pdf_op = AsyncOperationSpec(
                operation_id=f"pdf_process_{i}",
                operation_func=mock_pdf_processing,
                args=(pdf['filename'], "medium"),
                timeout_seconds=15.0
            )
            workflow_operations.append(pdf_op)
        
        # Phase 2: Knowledge base insertion (depends on PDF processing)
        for i in range(len(pdf_data)):
            kb_op = AsyncOperationSpec(
                operation_id=f"kb_insert_{i}",
                operation_func=mock_knowledge_base_insertion,
                args=({"extracted_text_length": 5000},),  # Mock data
                dependencies=[f"pdf_process_{i}"],
                timeout_seconds=10.0
            )
            workflow_operations.append(kb_op)
        
        # Phase 3: Query processing (depends on all KB insertions)
        kb_dependencies = [f"kb_insert_{i}" for i in range(len(pdf_data))]
        for i, query in enumerate(queries):
            query_op = AsyncOperationSpec(
                operation_id=f"query_process_{i}",
                operation_func=mock_query_processing,
                args=(query, {"entities_extracted": 50}),
                dependencies=kb_dependencies,
                timeout_seconds=8.0
            )
            workflow_operations.append(query_op)
        
        # Add performance monitoring
        performance_thresholds = {
            'memory_usage_mb': PerformanceThreshold(
                'memory_usage_mb', 800, 'lte', 'MB', 'warning',
                'Memory should stay under 800MB during workflow'
            )
        }
        
        async with async_performance_monitor(performance_thresholds) as perf_context:
            # Execute comprehensive workflow
            await coordinator.add_operations_batch(session_id, workflow_operations)
            
            async def workflow_progress(session_id: str, progress: float):
                logger.info(f"Workflow progress: {progress:.1f}%")
            
            results = await coordinator.execute_session(
                session_id,
                fail_on_first_error=False,
                progress_callback=workflow_progress
            )
            
            # Analyze comprehensive results
            aggregator = AsyncResultAggregator()
            
            # Custom analysis function
            async def phase_analysis(results):
                phases = {
                    'pdf_processing': [r for r in results.values() if r.operation_id.startswith('pdf_process')],
                    'kb_insertion': [r for r in results.values() if r.operation_id.startswith('kb_insert')], 
                    'query_processing': [r for r in results.values() if r.operation_id.startswith('query_process')]
                }
                
                phase_stats = {}
                for phase_name, phase_results in phases.items():
                    if phase_results:
                        success_rate = len([r for r in phase_results if r.succeeded]) / len(phase_results) * 100
                        avg_duration = sum(r.duration_ms or 0 for r in phase_results) / len(phase_results)
                        phase_stats[phase_name] = {
                            'operations': len(phase_results),
                            'success_rate': success_rate,
                            'avg_duration_ms': avg_duration
                        }
                
                return phase_stats
            
            aggregation = await aggregator.aggregate_results(
                results, 
                analysis_functions=[phase_analysis]
            )
            
            # Display comprehensive results
            logger.info("=== Comprehensive Workflow Results ===")
            execution_summary = aggregation['execution_summary']
            logger.info(f"Total operations: {execution_summary['total_operations']}")
            logger.info(f"Success rate: {execution_summary['success_rate_percent']:.1f}%")
            
            # Phase-specific results
            if 'custom_analysis' in aggregation and 'phase_analysis' in aggregation['custom_analysis']:
                phase_stats = aggregation['custom_analysis']['phase_analysis']
                for phase_name, stats in phase_stats.items():
                    logger.info(f"{phase_name}: {stats['operations']} ops, {stats['success_rate']:.1f}% success, {stats['avg_duration_ms']:.1f}ms avg")
            
            # Performance monitoring results
            logger.info(f"Performance samples: {len(perf_context['performance_samples'])}")
            if perf_context['threshold_breaches']:
                logger.warning(f"Performance threshold breaches: {len(perf_context['threshold_breaches'])}")
        
        # Get final session status
        final_status = await coordinator.get_session_status(session_id)
        logger.info(f"Final session statistics: {final_status}")


# =====================================================================
# MAIN DEMO RUNNER
# =====================================================================

async def main():
    """Run all async test utility demonstrations."""
    logger.info("Starting Async Test Utilities Demonstration")
    
    demos = [
        ("Basic Async Coordination", demo_basic_async_coordination),
        ("Async Context Managers", demo_async_context_managers), 
        ("Error Injection & Recovery", demo_error_injection_recovery),
        ("Batched Operations", demo_batched_operations),
        ("Retry Mechanisms", demo_retry_mechanisms),
        ("Comprehensive Workflow", demo_comprehensive_workflow)
    ]
    
    for demo_name, demo_func in demos:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {demo_name}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            await demo_func()
            duration = time.time() - start_time
            
            logger.info(f"✓ {demo_name} completed in {duration:.2f}s")
            
            # Small delay between demos
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.error(f"✗ {demo_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\nAsync Test Utilities Demonstration Complete!")


if __name__ == "__main__":
    asyncio.run(main())