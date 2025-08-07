#!/usr/bin/env python3
"""
Demonstration of Performance Testing Utilities and Benchmarking Helpers.

This script demonstrates how to use the comprehensive performance testing utilities
that integrate with the existing Clinical Metabolomics Oracle test infrastructure.

Key Features Demonstrated:
1. PerformanceAssertionHelper with timing decorators and memory validation
2. PerformanceBenchmarkSuite for standardized benchmarking
3. AdvancedResourceMonitor with threshold-based alerts
4. Integration with existing TestEnvironmentManager and MockSystemFactory
5. Comprehensive performance reporting and analysis

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List

# Import existing test infrastructure
from test_utilities import (
    TestEnvironmentManager, MockSystemFactory, SystemComponent, 
    MockBehavior, EnvironmentSpec, MockSpec, create_quick_test_environment
)

# Import new performance utilities
from performance_test_utilities import (
    PerformanceAssertionHelper, PerformanceBenchmarkSuite, AdvancedResourceMonitor,
    PerformanceThreshold, BenchmarkConfiguration
)

# Import performance test fixtures for mock operations
from performance_test_fixtures import (
    LoadTestScenarioGenerator, MockOperationGenerator, mock_clinical_query_operation
)


# =====================================================================
# SETUP LOGGING
# =====================================================================

def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_performance_utilities.log')
        ]
    )
    return logging.getLogger(__name__)


# =====================================================================
# DEMO 1: PERFORMANCE ASSERTION HELPER
# =====================================================================

async def demo_performance_assertion_helper(logger: logging.Logger):
    """Demonstrate PerformanceAssertionHelper capabilities."""
    
    logger.info("=" * 60)
    logger.info("DEMO 1: Performance Assertion Helper")
    logger.info("=" * 60)
    
    # Initialize assertion helper
    assertion_helper = PerformanceAssertionHelper(logger)
    assertion_helper.establish_memory_baseline()
    
    # Demo 1a: Timing decorator
    logger.info("\n1a. Timing Decorator Demo")
    
    @assertion_helper.time_operation("simulated_query_processing", 3000, auto_assert=True)
    async def simulate_query_processing(query: str) -> Dict[str, Any]:
        """Simulate query processing with variable duration."""
        processing_time = random.uniform(0.5, 2.5)  # 500ms to 2.5s
        await asyncio.sleep(processing_time)
        
        # Simulate memory allocation
        data = "x" * random.randint(1000, 10000)  # Varying memory usage
        
        return {
            'query': query,
            'response': f"Processed response for: {query}",
            'processing_time': processing_time,
            'data_size': len(data)
        }
    
    # Execute timed operations
    test_queries = [
        "What are the key metabolites in diabetes?",
        "How does cardiovascular disease affect lipid metabolism?",
        "What biomarkers are associated with liver disease?"
    ]
    
    for query in test_queries:
        try:
            result, metrics = await simulate_query_processing(query)
            logger.info(f"Query processed successfully: {metrics['duration_ms']:.1f}ms")
        except AssertionError as e:
            logger.warning(f"Performance assertion failed: {e}")
    
    # Demo 1b: Context manager timing
    logger.info("\n1b. Timing Context Manager Demo")
    
    with assertion_helper.timing_context("batch_processing", 5000):
        await asyncio.sleep(1.0)  # Simulate batch processing
        
        # Simulate memory-intensive operation
        large_data = []
        for i in range(1000):
            large_data.append("data" * 100)
        
        await asyncio.sleep(0.5)  # More processing
    
    # Demo 1c: Manual performance assertions
    logger.info("\n1c. Manual Performance Assertions Demo")
    
    # Throughput assertion
    start_time = time.time()
    operations_completed = 0
    
    for i in range(10):
        await asyncio.sleep(0.1)  # Simulate operation
        operations_completed += 1
    
    duration = time.time() - start_time
    
    try:
        assertion_helper.assert_throughput(
            operations_completed, duration, 5.0, "batch_throughput"
        )
        logger.info("Throughput assertion passed")
    except AssertionError as e:
        logger.warning(f"Throughput assertion failed: {e}")
    
    # Memory assertion
    try:
        assertion_helper.assert_memory_usage(200.0, "memory_check")
        logger.info("Memory assertion passed")
    except AssertionError as e:
        logger.warning(f"Memory assertion failed: {e}")
    
    # Memory leak check
    try:
        assertion_helper.assert_memory_leak_absent(50.0, "memory_leak_check")
        logger.info("Memory leak assertion passed")
    except AssertionError as e:
        logger.warning(f"Memory leak assertion failed: {e}")
    
    # Demo 1d: Error rate assertion
    logger.info("\n1d. Error Rate Assertion Demo")
    
    successful_ops = 95
    failed_ops = 5
    total_ops = successful_ops + failed_ops
    
    try:
        assertion_helper.assert_error_rate(failed_ops, total_ops, 8.0, "error_rate_check")
        logger.info("Error rate assertion passed")
    except AssertionError as e:
        logger.warning(f"Error rate assertion failed: {e}")
    
    # Show assertion summary
    logger.info("\n1e. Assertion Summary")
    summary = assertion_helper.get_assertion_summary()
    logger.info(f"Total assertions: {summary['total_assertions']}")
    logger.info(f"Passed: {summary['passed_assertions']}")
    logger.info(f"Failed: {summary['failed_assertions']}")
    logger.info(f"Success rate: {summary['success_rate_percent']:.1f}%")


# =====================================================================
# DEMO 2: ADVANCED RESOURCE MONITOR
# =====================================================================

async def demo_advanced_resource_monitor(logger: logging.Logger):
    """Demonstrate AdvancedResourceMonitor capabilities."""
    
    logger.info("=" * 60)
    logger.info("DEMO 2: Advanced Resource Monitor")
    logger.info("=" * 60)
    
    # Initialize resource monitor with custom thresholds
    custom_thresholds = {
        'cpu_percent': 70.0,
        'memory_mb': 300.0,
        'active_threads': 20
    }
    
    resource_monitor = AdvancedResourceMonitor(
        sampling_interval=0.5,
        alert_thresholds=custom_thresholds
    )
    
    logger.info("Starting resource monitoring...")
    resource_monitor.start_monitoring()
    
    # Demo 2a: Normal operations
    logger.info("\n2a. Normal Operations Monitoring")
    await asyncio.sleep(2.0)
    
    # Demo 2b: CPU intensive operation
    logger.info("\n2b. CPU Intensive Operation")
    start_time = time.time()
    result = 0
    while time.time() - start_time < 1.0:  # CPU intensive for 1 second
        result += sum(range(1000))
    
    await asyncio.sleep(1.0)
    
    # Demo 2c: Memory intensive operation
    logger.info("\n2c. Memory Intensive Operation")
    large_data_structures = []
    for i in range(5):
        large_data = ["data" * 1000 for _ in range(1000)]  # Create large data
        large_data_structures.append(large_data)
        await asyncio.sleep(0.5)
    
    # Demo 2d: Thread spawning (might trigger thread count alert)
    logger.info("\n2d. Thread Creation")
    import threading
    
    def worker():
        time.sleep(2.0)
    
    threads = []
    for i in range(10):  # Create multiple threads
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
        await asyncio.sleep(0.1)
    
    await asyncio.sleep(1.0)
    
    # Stop monitoring and get results
    logger.info("\nStopping resource monitoring...")
    snapshots = resource_monitor.stop_monitoring()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    # Analyze results
    logger.info("\n2e. Resource Monitoring Analysis")
    
    resource_summary = resource_monitor.get_resource_summary()
    logger.info(f"Monitoring duration: {resource_summary.get('monitoring_duration', 0):.1f}s")
    logger.info(f"Samples collected: {resource_summary.get('samples_collected', 0)}")
    
    cpu_stats = resource_summary.get('cpu_usage', {})
    logger.info(f"CPU - Avg: {cpu_stats.get('average', 0):.1f}%, Max: {cpu_stats.get('maximum', 0):.1f}%")
    
    memory_stats = resource_summary.get('memory_usage', {})
    logger.info(f"Memory - Avg: {memory_stats.get('average_mb', 0):.1f}MB, Peak: {memory_stats.get('maximum_mb', 0):.1f}MB")
    logger.info(f"Memory increase: {memory_stats.get('peak_increase_mb', 0):.1f}MB")
    
    # Show trends
    trends = resource_monitor.get_resource_trends()
    for trend_name, trend_data in trends.items():
        if 'direction' in trend_data:
            logger.info(f"Trend {trend_name}: {trend_data['direction']} (slope: {trend_data['slope']:.3f})")
    
    # Show alerts
    alert_summary = resource_monitor.get_alert_summary()
    logger.info(f"\nAlerts generated: {alert_summary['total_alerts']}")
    if alert_summary['total_alerts'] > 0:
        for alert_type, count in alert_summary.get('alert_counts_by_type', {}).items():
            logger.info(f"  {alert_type}: {count}")


# =====================================================================
# DEMO 3: PERFORMANCE BENCHMARK SUITE
# =====================================================================

async def demo_performance_benchmark_suite(logger: logging.Logger):
    """Demonstrate PerformanceBenchmarkSuite capabilities."""
    
    logger.info("=" * 60)
    logger.info("DEMO 3: Performance Benchmark Suite")
    logger.info("=" * 60)
    
    # Set up test environment
    env_manager, mock_factory = create_quick_test_environment(async_support=True)
    
    # Initialize benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite(
        output_dir=Path("demo_benchmarks"),
        environment_manager=env_manager
    )
    
    # Demo 3a: Run standard benchmarks
    logger.info("\n3a. Running Standard Benchmark Suite")
    
    # Use mock operation for demonstration
    mock_generator = MockOperationGenerator()
    
    benchmark_results = await benchmark_suite.run_benchmark_suite(
        benchmark_names=['clinical_query_performance'],  # Run just one for demo
        operation_func=mock_clinical_query_operation,
        data_generator=mock_generator.generate_query_data
    )
    
    # Show results summary
    summary = benchmark_results['suite_execution_summary']
    logger.info(f"Benchmark suite completed:")
    logger.info(f"  Total benchmarks: {summary['total_benchmarks']}")
    logger.info(f"  Passed: {summary['passed_benchmarks']}")
    logger.info(f"  Success rate: {summary['success_rate_percent']:.1f}%")
    
    stats = benchmark_results.get('overall_performance_statistics', {})
    logger.info(f"  Total operations: {stats.get('total_operations', 0):,}")
    logger.info(f"  Average response time: {stats.get('average_response_time_ms', 0):.1f}ms")
    logger.info(f"  Average throughput: {stats.get('average_throughput_ops_per_sec', 0):.2f} ops/sec")
    logger.info(f"  Overall error rate: {stats.get('overall_error_rate_percent', 0):.1f}%")
    
    # Demo 3b: Custom benchmark configuration
    logger.info("\n3b. Custom Benchmark Configuration")
    
    custom_thresholds = {
        'response_time_ms': PerformanceThreshold(
            'response_time_ms', 2000, 'lte', 'ms', 'error',
            'Custom: Response time should be under 2 seconds'
        ),
        'throughput_ops_per_sec': PerformanceThreshold(
            'throughput_ops_per_sec', 5.0, 'gte', 'ops/sec', 'error',
            'Custom: Should achieve at least 5 ops/sec'
        )
    }
    
    custom_benchmark = BenchmarkConfiguration(
        benchmark_name='custom_fast_benchmark',
        description='Custom benchmark with strict performance requirements',
        target_thresholds=custom_thresholds,
        test_scenarios=[LoadTestScenarioGenerator.create_light_load_scenario()]
    )
    
    # Add custom benchmark to suite
    benchmark_suite.standard_benchmarks['custom_fast_benchmark'] = custom_benchmark
    
    # Run custom benchmark
    custom_results = await benchmark_suite.run_benchmark_suite(
        benchmark_names=['custom_fast_benchmark'],
        operation_func=mock_clinical_query_operation,
        data_generator=mock_generator.generate_query_data
    )
    
    logger.info("Custom benchmark completed:")
    custom_summary = custom_results['suite_execution_summary']
    logger.info(f"  Passed: {custom_summary['passed_benchmarks']}/{custom_summary['total_benchmarks']}")
    
    # Demo 3c: Show recommendations
    logger.info("\n3c. Benchmark Recommendations")
    recommendations = benchmark_results.get('recommendations', [])
    for i, recommendation in enumerate(recommendations, 1):
        logger.info(f"  {i}. {recommendation}")
    
    # Cleanup
    env_manager.cleanup()


# =====================================================================
# DEMO 4: INTEGRATED PERFORMANCE TESTING
# =====================================================================

async def demo_integrated_performance_testing(logger: logging.Logger):
    """Demonstrate integrated performance testing with all utilities."""
    
    logger.info("=" * 60)
    logger.info("DEMO 4: Integrated Performance Testing")
    logger.info("=" * 60)
    
    # Set up comprehensive test environment
    env_spec = EnvironmentSpec(
        temp_dirs=["logs", "output", "performance_data"],
        performance_monitoring=True,
        memory_limits={'test_limit': 500}  # 500MB limit
    )
    
    env_manager = TestEnvironmentManager(env_spec)
    environment_data = env_manager.setup_environment()
    
    logger.info(f"Test environment setup: {environment_data['working_dir']}")
    
    # Initialize all performance utilities
    assertion_helper = PerformanceAssertionHelper(logger)
    resource_monitor = AdvancedResourceMonitor(sampling_interval=0.3)
    mock_factory = MockSystemFactory(env_manager)
    
    # Demo 4a: Complete mock system with performance monitoring
    logger.info("\n4a. Mock System Performance Test")
    
    # Create comprehensive mock system
    mock_components = [
        SystemComponent.LIGHTRAG_SYSTEM,
        SystemComponent.PDF_PROCESSOR,
        SystemComponent.COST_MONITOR,
        SystemComponent.PROGRESS_TRACKER
    ]
    
    mock_system = mock_factory.create_comprehensive_mock_set(
        mock_components, MockBehavior.SUCCESS
    )
    
    logger.info(f"Created mock system with {len(mock_system)} components")
    
    # Start comprehensive monitoring
    assertion_helper.establish_memory_baseline()
    resource_monitor.start_monitoring()
    
    # Demo 4b: Simulated workflow with performance validation
    logger.info("\n4b. Simulated Clinical Workflow")
    
    @assertion_helper.time_operation("clinical_workflow", 8000, memory_monitoring=True)
    async def simulate_clinical_workflow():
        """Simulate complete clinical metabolomics workflow."""
        
        # Step 1: PDF Processing
        pdf_processor = mock_system['pdf_processor']
        cost_monitor = mock_system['cost_monitor']
        progress_tracker = mock_system['progress_tracker']
        lightrag_system = mock_system['lightrag_system']
        
        # Process mock PDFs
        pdf_files = [f"clinical_study_{i}.pdf" for i in range(5)]
        processed_docs = []
        
        for pdf_file in pdf_files:
            progress_tracker.update_progress(
                len(processed_docs) / len(pdf_files) * 0.5,  # 50% for PDF processing
                f"Processing {pdf_file}"
            )
            
            doc_result = await pdf_processor.process_pdf(pdf_file)
            processed_docs.append(doc_result['text'])
            
            # Track costs
            cost_monitor.track_cost('pdf_processing', 0.02, file=pdf_file)
            
            await asyncio.sleep(0.1)  # Simulate processing time
        
        # Step 2: Knowledge Base Ingestion
        progress_tracker.update_progress(0.6, "Ingesting into knowledge base")
        
        ingestion_result = await lightrag_system.ainsert(processed_docs)
        cost_monitor.track_cost('kb_ingestion', ingestion_result['total_cost'])
        
        # Step 3: Query Processing
        test_queries = [
            "What metabolites are associated with diabetes?",
            "How does cardiovascular disease affect lipid metabolism?",
            "What are the key biomarkers for liver disease?"
        ]
        
        query_results = []
        for i, query in enumerate(test_queries):
            progress_tracker.update_progress(
                0.6 + ((i + 1) / len(test_queries)) * 0.4,  # 60% to 100%
                f"Processing query: {query}"
            )
            
            result = await lightrag_system.aquery(query, mode="hybrid")
            query_results.append(result)
            
            cost_monitor.track_cost('query_processing', 0.05, query=query)
            
            await asyncio.sleep(0.2)  # Simulate query processing
        
        progress_tracker.update_progress(1.0, "Workflow completed")
        
        return {
            'processed_documents': len(processed_docs),
            'ingestion_result': ingestion_result,
            'query_results': query_results,
            'total_cost': cost_monitor.get_total_cost(),
            'progress_summary': progress_tracker.get_summary()
        }
    
    # Execute workflow with monitoring
    try:
        workflow_result, workflow_metrics = await simulate_clinical_workflow()
        logger.info(f"Clinical workflow completed successfully")
        logger.info(f"  Duration: {workflow_metrics['duration_ms']:.1f}ms")
        logger.info(f"  Memory delta: {workflow_metrics.get('memory_delta_mb', 0):.1f}MB")
        logger.info(f"  Documents processed: {workflow_result['processed_documents']}")
        logger.info(f"  Total cost: ${workflow_result['total_cost']:.4f}")
    except Exception as e:
        logger.error(f"Clinical workflow failed: {e}")
    
    # Stop monitoring
    resource_snapshots = resource_monitor.stop_monitoring()
    
    # Demo 4c: Comprehensive performance validation
    logger.info("\n4c. Comprehensive Performance Validation")
    
    # Validate resource usage
    resource_summary = resource_monitor.get_resource_summary()
    logger.info(f"Peak memory usage: {resource_summary.get('memory_usage', {}).get('maximum_mb', 0):.1f}MB")
    
    try:
        assertion_helper.assert_memory_usage(400.0, "workflow_memory_check")
        logger.info("Memory usage validation passed")
    except AssertionError as e:
        logger.warning(f"Memory usage validation failed: {e}")
    
    # Check for memory leaks
    try:
        assertion_helper.assert_memory_leak_absent(100.0, "workflow_memory_leak_check")
        logger.info("Memory leak validation passed")
    except AssertionError as e:
        logger.warning(f"Memory leak validation failed: {e}")
    
    # Demo 4d: Generate comprehensive report
    logger.info("\n4d. Comprehensive Performance Report")
    
    # Export all results
    results_dir = environment_data['subdirectories']['performance_data']
    
    # Export assertion results
    assertion_results_path = results_dir / "integrated_test_assertions.json"
    assertion_helper.export_results_to_json(assertion_results_path)
    
    # Export resource monitoring report
    resource_report_path = results_dir / "integrated_test_resources.json"
    resource_monitor.export_monitoring_report(resource_report_path)
    
    # Show final summary
    assertion_summary = assertion_helper.get_assertion_summary()
    alert_summary = resource_monitor.get_alert_summary()
    
    logger.info("Integration test completed:")
    logger.info(f"  Assertions - Passed: {assertion_summary['passed_assertions']}, Failed: {assertion_summary['failed_assertions']}")
    logger.info(f"  Resource alerts: {alert_summary['total_alerts']}")
    logger.info(f"  Reports exported to: {results_dir}")
    
    # Cleanup
    env_manager.cleanup()


# =====================================================================
# MAIN DEMONSTRATION
# =====================================================================

async def main():
    """Run all performance utilities demonstrations."""
    
    logger = setup_logging()
    
    logger.info("Starting Clinical Metabolomics Oracle Performance Utilities Demonstration")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Performance Assertion Helper
        await demo_performance_assertion_helper(logger)
        await asyncio.sleep(1.0)
        
        # Demo 2: Advanced Resource Monitor
        await demo_advanced_resource_monitor(logger)
        await asyncio.sleep(1.0)
        
        # Demo 3: Performance Benchmark Suite
        await demo_performance_benchmark_suite(logger)
        await asyncio.sleep(1.0)
        
        # Demo 4: Integrated Performance Testing
        await demo_integrated_performance_testing(logger)
        
        logger.info("=" * 80)
        logger.info("All performance utilities demonstrations completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Demonstration failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())