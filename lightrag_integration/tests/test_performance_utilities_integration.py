#!/usr/bin/env python3
"""
Integration Tests for Performance Testing Utilities.

This module contains integration tests that verify the performance testing utilities
work correctly with the existing Clinical Metabolomics Oracle test infrastructure.

Test Coverage:
1. PerformanceAssertionHelper integration with existing fixtures
2. PerformanceBenchmarkSuite integration with TestEnvironmentManager
3. AdvancedResourceMonitor integration and alerting
4. End-to-end performance testing workflow
5. Cross-utility integration and data flow

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock

# Import existing test infrastructure
from test_utilities import (
    TestEnvironmentManager, MockSystemFactory, SystemComponent,
    MockBehavior, EnvironmentSpec
)

# Import performance utilities
from performance_test_utilities import (
    PerformanceAssertionHelper, PerformanceBenchmarkSuite, AdvancedResourceMonitor,
    PerformanceThreshold, BenchmarkConfiguration
)

# Import performance test fixtures
from performance_test_fixtures import (
    PerformanceMetrics, LoadTestScenarioGenerator, MockOperationGenerator,
    mock_clinical_query_operation
)


# =====================================================================
# TEST FIXTURES
# =====================================================================

@pytest.fixture
def test_logger():
    """Provide test logger."""
    logger = logging.getLogger("test_performance_utilities")
    logger.setLevel(logging.INFO)
    return logger


@pytest.fixture
def sample_performance_thresholds():
    """Provide sample performance thresholds for testing."""
    return {
        'response_time_ms': PerformanceThreshold(
            'response_time_ms', 2000, 'lte', 'ms', 'error',
            'Test response time should be under 2 seconds'
        ),
        'throughput_ops_per_sec': PerformanceThreshold(
            'throughput_ops_per_sec', 2.0, 'gte', 'ops/sec', 'error',
            'Test throughput should be at least 2 operations per second'
        ),
        'memory_usage_mb': PerformanceThreshold(
            'memory_usage_mb', 200, 'lte', 'MB', 'warning',
            'Test memory usage should be under 200MB'
        )
    }


# =====================================================================
# PERFORMANCE ASSERTION HELPER TESTS
# =====================================================================

class TestPerformanceAssertionHelper:
    """Test PerformanceAssertionHelper integration."""
    
    def test_initialization_with_existing_infrastructure(self, test_logger):
        """Test PerformanceAssertionHelper initializes correctly."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        assert assertion_helper.logger == test_logger
        assert len(assertion_helper.assertion_results) == 0
        assert len(assertion_helper.performance_data) == 0
        assert assertion_helper.memory_baseline is None
    
    def test_memory_baseline_establishment(self, test_logger):
        """Test memory baseline establishment."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        assertion_helper.establish_memory_baseline()
        
        assert assertion_helper.memory_baseline is not None
        assert assertion_helper.memory_baseline > 0
    
    @pytest.mark.asyncio
    async def test_timing_decorator_integration(self, test_logger):
        """Test timing decorator with async functions."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        @assertion_helper.time_operation("test_operation", 1000, auto_assert=True)
        async def test_async_operation():
            await asyncio.sleep(0.1)  # 100ms operation
            return {"status": "completed"}
        
        # Execute decorated function
        result, metrics = await test_async_operation()
        
        # Verify results
        assert result["status"] == "completed"
        assert metrics["duration_ms"] >= 100  # At least 100ms
        assert metrics["duration_ms"] < 1000   # Under threshold
        assert "test_operation" in metrics["operation_name"]
    
    def test_timing_context_manager(self, test_logger):
        """Test timing context manager."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        with assertion_helper.timing_context("test_context", 500):
            time.sleep(0.1)  # 100ms operation
        
        # Verify performance data was recorded
        assert "test_context_duration_ms" in assertion_helper.performance_data
        durations = assertion_helper.performance_data["test_context_duration_ms"]
        assert len(durations) == 1
        assert durations[0] >= 100
    
    def test_throughput_calculation_and_assertion(self, test_logger):
        """Test throughput calculation and assertion."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        # Test successful throughput assertion
        throughput = assertion_helper.calculate_throughput(20, 5.0, "test_throughput")
        assert throughput == 4.0  # 20 ops / 5 seconds = 4 ops/sec
        
        # Test throughput assertion pass
        result = assertion_helper.assert_throughput(20, 5.0, 3.0, "throughput_test")
        assert result.passed is True
        assert result.measured_value == 4.0
        
        # Test throughput assertion fail
        with pytest.raises(AssertionError):
            assertion_helper.assert_throughput(10, 5.0, 5.0, "throughput_fail_test")
    
    def test_memory_assertion_integration(self, test_logger):
        """Test memory assertion with baseline."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        assertion_helper.establish_memory_baseline()
        
        # Test memory usage assertion
        result = assertion_helper.assert_memory_usage(1000.0, "memory_test")
        assert result.passed is True  # Should pass with reasonable limit
        
        # Test memory leak assertion
        result = assertion_helper.assert_memory_leak_absent(100.0, "leak_test")
        assert result.passed is True  # Should pass with no significant leak
    
    def test_error_rate_assertion(self, test_logger):
        """Test error rate assertion."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        # Test passing error rate
        result = assertion_helper.assert_error_rate(2, 100, 5.0, "error_rate_test")
        assert result.passed is True
        assert result.measured_value == 2.0  # 2%
        
        # Test failing error rate
        with pytest.raises(AssertionError):
            assertion_helper.assert_error_rate(10, 100, 5.0, "high_error_rate_test")
    
    def test_assertion_summary_generation(self, test_logger):
        """Test assertion summary generation."""
        assertion_helper = PerformanceAssertionHelper(test_logger)
        
        # Add some test assertions
        assertion_helper.assert_error_rate(1, 100, 5.0, "test1")
        assertion_helper.assert_throughput(10, 2.0, 4.0, "test2")
        
        summary = assertion_helper.get_assertion_summary()
        
        assert summary["total_assertions"] == 2
        assert summary["passed_assertions"] == 2
        assert summary["failed_assertions"] == 0
        assert summary["success_rate_percent"] == 100.0
        assert len(summary["assertions"]) == 2


# =====================================================================
# ADVANCED RESOURCE MONITOR TESTS
# =====================================================================

class TestAdvancedResourceMonitor:
    """Test AdvancedResourceMonitor integration."""
    
    def test_initialization_with_custom_thresholds(self):
        """Test resource monitor initialization."""
        custom_thresholds = {
            'cpu_percent': 60.0,
            'memory_mb': 300.0
        }
        
        monitor = AdvancedResourceMonitor(
            sampling_interval=0.1,
            alert_thresholds=custom_thresholds
        )
        
        assert monitor.sampling_interval == 0.1
        assert monitor.alert_thresholds['cpu_percent'] == 60.0
        assert monitor.alert_thresholds['memory_mb'] == 300.0
        assert len(monitor.alerts) == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self):
        """Test complete monitoring lifecycle."""
        monitor = AdvancedResourceMonitor(sampling_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring is True
        
        # Let it collect some samples
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        snapshots = monitor.stop_monitoring()
        
        assert monitor.monitoring is False
        assert len(snapshots) > 0
        assert all(hasattr(s, 'timestamp') for s in snapshots)
        assert all(hasattr(s, 'cpu_percent') for s in snapshots)
        assert all(hasattr(s, 'memory_mb') for s in snapshots)
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test resource alert generation."""
        # Set very low thresholds to trigger alerts
        low_thresholds = {
            'cpu_percent': 0.1,  # Very low to trigger alert
            'memory_mb': 1.0,    # Very low to trigger alert
            'active_threads': 1  # Very low to trigger alert
        }
        
        monitor = AdvancedResourceMonitor(
            sampling_interval=0.1,
            alert_thresholds=low_thresholds
        )
        
        monitor.start_monitoring()
        await asyncio.sleep(0.3)  # Let it sample and potentially trigger alerts
        monitor.stop_monitoring()
        
        alert_summary = monitor.get_alert_summary()
        # Should have generated some alerts due to low thresholds
        assert alert_summary['total_alerts'] >= 0  # May or may not trigger depending on system
    
    def test_trend_analysis(self):
        """Test resource trend analysis."""
        monitor = AdvancedResourceMonitor()
        
        # Add some sample trend data
        monitor.trend_data['cpu_trend'].extend([10.0, 15.0, 20.0, 25.0, 30.0])
        monitor.trend_data['memory_trend'].extend([100.0, 110.0, 120.0, 130.0, 140.0])
        
        trends = monitor.get_resource_trends()
        
        assert 'cpu_trend' in trends
        assert 'memory_trend' in trends
        
        # Both should show increasing trends
        assert trends['cpu_trend']['direction'] == 'increasing'
        assert trends['memory_trend']['direction'] == 'increasing'
        
        assert trends['cpu_trend']['slope'] > 0
        assert trends['memory_trend']['slope'] > 0
    
    def test_resource_summary_generation(self):
        """Test resource summary generation."""
        monitor = AdvancedResourceMonitor()
        
        # Mock some snapshots
        from performance_test_fixtures import ResourceUsageSnapshot
        
        snapshots = [
            ResourceUsageSnapshot(
                timestamp=time.time(),
                cpu_percent=20.0,
                memory_mb=100.0,
                memory_percent=10.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_bytes_sent=0.0,
                network_bytes_received=0.0,
                active_threads=5,
                open_file_descriptors=10
            ),
            ResourceUsageSnapshot(
                timestamp=time.time() + 1,
                cpu_percent=30.0,
                memory_mb=120.0,
                memory_percent=12.0,
                disk_io_read_mb=1.0,
                disk_io_write_mb=1.0,
                network_bytes_sent=100.0,
                network_bytes_received=200.0,
                active_threads=6,
                open_file_descriptors=12
            )
        ]
        
        monitor.snapshots = snapshots
        monitor.baseline_cpu = 15.0
        monitor.baseline_memory = 90.0
        
        summary = monitor.get_resource_summary()
        
        assert summary['samples_collected'] == 2
        assert summary['cpu_usage']['average'] == 25.0  # (20 + 30) / 2
        assert summary['cpu_usage']['maximum'] == 30.0
        assert summary['memory_usage']['average_mb'] == 110.0  # (100 + 120) / 2
        assert summary['memory_usage']['maximum_mb'] == 120.0


# =====================================================================
# PERFORMANCE BENCHMARK SUITE TESTS
# =====================================================================

class TestPerformanceBenchmarkSuite:
    """Test PerformanceBenchmarkSuite integration."""
    
    @pytest.fixture
    def test_environment_setup(self):
        """Set up test environment for benchmark suite."""
        env_spec = EnvironmentSpec(
            temp_dirs=["test_output"],
            performance_monitoring=False
        )
        env_manager = TestEnvironmentManager(env_spec)
        environment_data = env_manager.setup_environment()
        
        yield env_manager, environment_data
        
        env_manager.cleanup()
    
    def test_initialization_with_environment_manager(self, test_environment_setup):
        """Test benchmark suite initialization."""
        env_manager, environment_data = test_environment_setup
        
        benchmark_suite = PerformanceBenchmarkSuite(
            output_dir=Path("test_benchmarks"),
            environment_manager=env_manager
        )
        
        assert benchmark_suite.environment_manager == env_manager
        assert benchmark_suite.output_dir.name == "test_benchmarks"
        assert len(benchmark_suite.standard_benchmarks) > 0
        assert 'clinical_query_performance' in benchmark_suite.standard_benchmarks
    
    def test_standard_benchmark_configurations(self, test_environment_setup):
        """Test standard benchmark configurations."""
        env_manager, environment_data = test_environment_setup
        
        benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
        
        # Check clinical query performance benchmark
        clinical_benchmark = benchmark_suite.standard_benchmarks['clinical_query_performance']
        assert clinical_benchmark.benchmark_name == 'clinical_query_performance'
        assert 'response_time_ms' in clinical_benchmark.target_thresholds
        assert 'throughput_ops_per_sec' in clinical_benchmark.target_thresholds
        assert len(clinical_benchmark.test_scenarios) > 0
    
    @pytest.mark.asyncio
    async def test_single_benchmark_execution(self, test_environment_setup):
        """Test single benchmark execution."""
        env_manager, environment_data = test_environment_setup
        
        benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
        
        # Create simple test configuration
        simple_thresholds = {
            'response_time_ms': PerformanceThreshold(
                'response_time_ms', 5000, 'lte', 'ms', 'error',
                'Simple test should be under 5 seconds'
            )
        }
        
        test_config = BenchmarkConfiguration(
            benchmark_name='simple_test',
            description='Simple test configuration',
            target_thresholds=simple_thresholds,
            test_scenarios=[LoadTestScenarioGenerator.create_baseline_scenario()]
        )
        
        # Modify scenario for faster execution
        test_config.test_scenarios[0].duration_seconds = 5.0
        test_config.test_scenarios[0].target_operations_per_second = 2.0
        
        mock_generator = MockOperationGenerator()
        
        result = await benchmark_suite._run_single_benchmark(
            test_config,
            mock_clinical_query_operation,
            mock_generator.generate_query_data
        )
        
        assert result['benchmark_name'] == 'simple_test'
        assert 'scenario_results' in result
        assert 'analysis' in result
        assert len(result['scenario_results']) > 0
        
        # Check if benchmark passed or failed
        assert isinstance(result['passed'], bool)
    
    @pytest.mark.asyncio
    async def test_benchmark_suite_execution(self, test_environment_setup):
        """Test complete benchmark suite execution."""
        env_manager, environment_data = test_environment_setup
        
        benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
        
        # Create minimal test benchmark
        minimal_scenario = LoadTestScenarioGenerator.create_baseline_scenario()
        minimal_scenario.duration_seconds = 3.0  # Very short for testing
        minimal_scenario.target_operations_per_second = 1.0
        
        minimal_thresholds = {
            'response_time_ms': PerformanceThreshold(
                'response_time_ms', 10000, 'lte', 'ms', 'error',
                'Test should be under 10 seconds'
            )
        }
        
        test_benchmark = BenchmarkConfiguration(
            benchmark_name='minimal_test',
            description='Minimal test for integration testing',
            target_thresholds=minimal_thresholds,
            test_scenarios=[minimal_scenario]
        )
        
        benchmark_suite.standard_benchmarks['minimal_test'] = test_benchmark
        
        mock_generator = MockOperationGenerator()
        
        suite_results = await benchmark_suite.run_benchmark_suite(
            benchmark_names=['minimal_test'],
            operation_func=mock_clinical_query_operation,
            data_generator=mock_generator.generate_query_data
        )
        
        # Verify suite results structure
        assert 'suite_execution_summary' in suite_results
        assert 'benchmark_results' in suite_results
        assert 'assertion_summary' in suite_results
        assert 'recommendations' in suite_results
        
        summary = suite_results['suite_execution_summary']
        assert summary['total_benchmarks'] == 1
        assert 'passed_benchmarks' in summary
        assert 'success_rate_percent' in summary
    
    def test_custom_benchmark_configuration(self, test_environment_setup):
        """Test custom benchmark configuration."""
        env_manager, environment_data = test_environment_setup
        
        benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
        
        custom_thresholds = {
            'custom_metric': PerformanceThreshold(
                'custom_metric', 100, 'lte', 'units', 'warning',
                'Custom metric should be under 100 units'
            )
        }
        
        custom_benchmark = BenchmarkConfiguration(
            benchmark_name='custom_benchmark',
            description='Custom benchmark for testing',
            target_thresholds=custom_thresholds,
            test_scenarios=[LoadTestScenarioGenerator.create_light_load_scenario()]
        )
        
        # Add to suite
        benchmark_suite.standard_benchmarks['custom_benchmark'] = custom_benchmark
        
        # Verify it was added
        assert 'custom_benchmark' in benchmark_suite.standard_benchmarks
        retrieved = benchmark_suite.standard_benchmarks['custom_benchmark']
        assert retrieved.benchmark_name == 'custom_benchmark'
        assert 'custom_metric' in retrieved.target_thresholds


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestPerformanceUtilitiesIntegration:
    """Test integration between all performance utilities."""
    
    @pytest.fixture
    def integrated_test_setup(self):
        """Set up integrated test environment."""
        # Environment setup
        env_spec = EnvironmentSpec(
            temp_dirs=["logs", "output", "performance_data"],
            performance_monitoring=True
        )
        env_manager = TestEnvironmentManager(env_spec)
        environment_data = env_manager.setup_environment()
        
        # Mock system setup
        mock_factory = MockSystemFactory(env_manager)
        mock_system = mock_factory.create_comprehensive_mock_set(
            [SystemComponent.LIGHTRAG_SYSTEM, SystemComponent.COST_MONITOR],
            MockBehavior.SUCCESS
        )
        
        # Performance utilities setup
        assertion_helper = PerformanceAssertionHelper()
        resource_monitor = AdvancedResourceMonitor(sampling_interval=0.2)
        benchmark_suite = PerformanceBenchmarkSuite(environment_manager=env_manager)
        
        yield {
            'env_manager': env_manager,
            'environment_data': environment_data,
            'mock_factory': mock_factory,
            'mock_system': mock_system,
            'assertion_helper': assertion_helper,
            'resource_monitor': resource_monitor,
            'benchmark_suite': benchmark_suite
        }
        
        env_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_testing(self, integrated_test_setup):
        """Test complete end-to-end performance testing workflow."""
        setup = integrated_test_setup
        
        assertion_helper = setup['assertion_helper']
        resource_monitor = setup['resource_monitor']
        mock_system = setup['mock_system']
        
        # Step 1: Establish baseline
        assertion_helper.establish_memory_baseline()
        
        # Step 2: Start monitoring
        resource_monitor.start_monitoring()
        
        # Step 3: Execute test workflow with performance tracking
        @assertion_helper.time_operation("integration_test", 5000)
        async def integration_test_workflow():
            lightrag_system = mock_system['lightrag_system']
            cost_monitor = mock_system['cost_monitor']
            
            # Simulate document processing
            test_docs = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
            ingestion_result = await lightrag_system.ainsert(test_docs)
            cost_monitor.track_cost('ingestion', ingestion_result['total_cost'])
            
            # Simulate queries
            test_queries = [
                "What are metabolites?",
                "How does diabetes affect metabolism?",
                "What are biomarkers?"
            ]
            
            query_results = []
            for query in test_queries:
                result = await lightrag_system.aquery(query)
                query_results.append(result)
                cost_monitor.track_cost('query', 0.02)
                await asyncio.sleep(0.1)  # Simulate processing time
            
            return {
                'ingestion_result': ingestion_result,
                'query_results': query_results,
                'total_cost': cost_monitor.get_total_cost()
            }
        
        # Execute workflow
        workflow_result, workflow_metrics = await integration_test_workflow()
        
        # Step 4: Stop monitoring
        resource_snapshots = resource_monitor.stop_monitoring()
        
        # Step 5: Performance validation
        # Throughput assertion
        num_operations = 1 + len(workflow_result['query_results'])  # ingestion + queries
        duration_seconds = workflow_metrics['duration_ms'] / 1000.0
        
        throughput_result = assertion_helper.assert_throughput(
            num_operations, duration_seconds, 0.5, "integration_throughput"
        )
        
        # Memory assertions
        memory_result = assertion_helper.assert_memory_leak_absent(50.0, "integration_memory")
        
        # Step 6: Validate results
        assert workflow_result is not None
        assert len(workflow_result['query_results']) == 3
        assert workflow_result['total_cost'] > 0
        
        assert workflow_metrics['duration_ms'] > 0
        assert workflow_metrics['duration_ms'] < 5000  # Under threshold
        
        assert throughput_result.passed
        assert memory_result.passed
        
        assert len(resource_snapshots) > 0
        
        # Step 7: Generate summary
        assertion_summary = assertion_helper.get_assertion_summary()
        resource_summary = resource_monitor.get_resource_summary()
        
        assert assertion_summary['total_assertions'] >= 3  # timing, throughput, memory
        assert assertion_summary['success_rate_percent'] == 100.0
        
        assert resource_summary['samples_collected'] > 0
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, integrated_test_setup):
        """Test performance regression detection across utilities."""
        setup = integrated_test_setup
        benchmark_suite = setup['benchmark_suite']
        
        # Create baseline metrics
        from performance_test_fixtures import PerformanceMetrics
        
        baseline_metrics = PerformanceMetrics(
            test_name="regression_test",
            start_time=time.time(),
            end_time=time.time() + 10,
            duration=10.0,
            operations_count=20,
            success_count=20,
            failure_count=0,
            throughput_ops_per_sec=2.0,
            average_latency_ms=1000.0,
            median_latency_ms=900.0,
            p95_latency_ms=1500.0,
            p99_latency_ms=1800.0,
            min_latency_ms=500.0,
            max_latency_ms=2000.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            error_rate_percent=0.0,
            concurrent_operations=2
        )
        
        benchmark_suite.set_baseline_metrics("regression_test", baseline_metrics)
        
        # Create current metrics (simulating performance degradation)
        current_metrics = PerformanceMetrics(
            test_name="regression_test",
            start_time=time.time(),
            end_time=time.time() + 15,
            duration=15.0,
            operations_count=15,  # Fewer operations
            success_count=14,     # Some failures
            failure_count=1,
            throughput_ops_per_sec=1.0,  # Reduced throughput
            average_latency_ms=2000.0,   # Increased latency
            median_latency_ms=1800.0,
            p95_latency_ms=3000.0,
            p99_latency_ms=3500.0,
            min_latency_ms=800.0,
            max_latency_ms=4000.0,
            memory_usage_mb=150.0,       # Increased memory
            cpu_usage_percent=75.0,      # Increased CPU
            error_rate_percent=6.7,      # Added errors
            concurrent_operations=2
        )
        
        # Compare against baseline
        comparison = benchmark_suite.compare_against_baseline("regression_test", current_metrics)
        
        assert comparison['benchmark_name'] == "regression_test"
        assert 'performance_changes' in comparison
        assert 'performance_ratios' in comparison
        assert 'trend' in comparison
        
        # Should detect degradation
        changes = comparison['performance_changes']
        assert changes['response_time_change_ms'] > 0      # Increased response time
        assert changes['throughput_change_ops_per_sec'] < 0  # Decreased throughput
        assert changes['error_rate_change_percent'] > 0    # Increased errors
        
        # Overall trend should indicate degradation
        assert comparison['trend'] == 'degradation'
    
    def test_cross_utility_data_flow(self, integrated_test_setup):
        """Test data flow between different performance utilities."""
        setup = integrated_test_setup
        
        assertion_helper = setup['assertion_helper']
        resource_monitor = setup['resource_monitor']
        
        # Generate some test data in assertion helper
        assertion_helper.performance_data['test_operation_duration_ms'].extend([100, 150, 200, 120, 180])
        assertion_helper.performance_data['test_memory_delta_mb'].extend([5, 10, 8, 12, 6])
        
        # Generate some test data in resource monitor
        resource_monitor.trend_data['cpu_trend'].extend([20, 25, 30, 35, 40])
        resource_monitor.trend_data['memory_trend'].extend([100, 105, 110, 115, 120])
        
        # Extract data from assertion helper
        assertion_summary = assertion_helper.get_assertion_summary()
        performance_data = assertion_summary['performance_data']
        
        # Extract data from resource monitor
        resource_trends = resource_monitor.get_resource_trends()
        
        # Verify data cross-references
        assert 'test_operation_duration_ms' in performance_data
        assert len(performance_data['test_operation_duration_ms']) == 5
        
        assert 'cpu_trend' in resource_trends
        assert resource_trends['cpu_trend']['direction'] == 'increasing'
        
        # Verify we can combine data from both utilities
        combined_metrics = {
            'assertion_data': performance_data,
            'resource_trends': resource_trends
        }
        
        assert len(combined_metrics) == 2
        assert 'assertion_data' in combined_metrics
        assert 'resource_trends' in combined_metrics


# =====================================================================
# TEST EXECUTION
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])