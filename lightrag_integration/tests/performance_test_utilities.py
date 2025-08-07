#!/usr/bin/env python3
"""
Performance Testing Utilities and Benchmarking Helpers.

This module provides comprehensive performance testing utilities that build on the
existing Clinical Metabolomics Oracle test infrastructure. It implements:

1. PerformanceAssertionHelper: Standard timing decorators and performance validation
2. PerformanceBenchmarkSuite: Standardized benchmarks across different scenarios  
3. Resource monitoring utilities with detailed diagnostics
4. Performance regression detection and analysis
5. Load testing coordination and management
6. Performance data visualization and reporting helpers

Key Features:
- Integrates seamlessly with TestEnvironmentManager and MockSystemFactory
- Provides standardized performance assertions with meaningful error messages
- Implements comprehensive benchmarking suites with baseline comparisons
- Advanced resource monitoring with threshold-based alerts
- Performance regression detection with statistical analysis
- Load testing coordination with concurrent operation management
- Performance data visualization helpers for reporting

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import threading
import statistics
import json
import warnings
import functools
import traceback
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Type, ContextManager
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import psutil
import logging

# Import from existing test infrastructure
from test_utilities import (
    TestEnvironmentManager, MockSystemFactory, SystemComponent, 
    TestComplexity, MockBehavior, EnvironmentSpec, MockSpec
)
from performance_test_fixtures import (
    PerformanceMetrics, LoadTestScenario, ResourceUsageSnapshot,
    ResourceMonitor, PerformanceTestExecutor, LoadTestScenarioGenerator
)
from performance_analysis_utilities import (
    PerformanceReport, PerformanceReportGenerator
)


# =====================================================================
# PERFORMANCE ASSERTION HELPER
# =====================================================================

@dataclass
class PerformanceThreshold:
    """Performance threshold specification."""
    metric_name: str
    threshold_value: Union[int, float]
    comparison_operator: str  # 'lt', 'lte', 'gt', 'gte', 'eq', 'neq'
    unit: str
    severity: str = 'error'  # 'warning', 'error', 'critical'
    description: str = ""
    
    def check(self, actual_value: Union[int, float]) -> Tuple[bool, str]:
        """Check if threshold is met."""
        operators = {
            'lt': lambda a, t: a < t,
            'lte': lambda a, t: a <= t, 
            'gt': lambda a, t: a > t,
            'gte': lambda a, t: a >= t,
            'eq': lambda a, t: abs(a - t) < 1e-9,
            'neq': lambda a, t: abs(a - t) >= 1e-9
        }
        
        if self.comparison_operator not in operators:
            return False, f"Invalid comparison operator: {self.comparison_operator}"
        
        passes = operators[self.comparison_operator](actual_value, self.threshold_value)
        
        if passes:
            message = f"✓ {self.metric_name}: {actual_value:.2f} {self.unit} meets threshold ({self.comparison_operator} {self.threshold_value} {self.unit})"
        else:
            message = f"✗ {self.metric_name}: {actual_value:.2f} {self.unit} fails threshold ({self.comparison_operator} {self.threshold_value} {self.unit})"
            if self.description:
                message += f" - {self.description}"
        
        return passes, message


@dataclass
class PerformanceAssertionResult:
    """Result of performance assertion."""
    assertion_name: str
    passed: bool
    measured_value: Union[int, float]
    threshold: PerformanceThreshold
    message: str
    timestamp: float
    duration_ms: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceAssertionHelper:
    """
    Comprehensive performance assertion helper with timing decorators,
    memory validation, throughput calculation, and threshold checking.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.assertion_results: List[PerformanceAssertionResult] = []
        self.active_timers: Dict[str, float] = {}
        self.memory_baseline: Optional[float] = None
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        
        # Default thresholds
        self.default_thresholds = {
            'response_time_ms': PerformanceThreshold(
                'response_time_ms', 10000, 'lte', 'ms', 'error',
                'Response time should be under 10 seconds'
            ),
            'memory_usage_mb': PerformanceThreshold(
                'memory_usage_mb', 1000, 'lte', 'MB', 'warning',
                'Memory usage should be under 1GB'
            ),
            'throughput_ops_per_sec': PerformanceThreshold(
                'throughput_ops_per_sec', 1.0, 'gte', 'ops/sec', 'error',
                'Throughput should be at least 1 operation per second'
            ),
            'error_rate_percent': PerformanceThreshold(
                'error_rate_percent', 5.0, 'lte', '%', 'error',
                'Error rate should be under 5%'
            ),
            'cpu_usage_percent': PerformanceThreshold(
                'cpu_usage_percent', 85.0, 'lte', '%', 'warning',
                'CPU usage should be under 85%'
            )
        }
    
    # Timing Decorators and Context Managers
    
    def time_operation(self, 
                      operation_name: str,
                      expected_max_duration_ms: Optional[float] = None,
                      memory_monitoring: bool = True,
                      auto_assert: bool = True):
        """
        Decorator for timing operations with automatic assertion.
        
        Args:
            operation_name: Name of the operation being timed
            expected_max_duration_ms: Maximum expected duration in milliseconds
            memory_monitoring: Whether to monitor memory usage
            auto_assert: Whether to automatically assert against thresholds
        """
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_timed_operation(
                        func, args, kwargs, operation_name,
                        expected_max_duration_ms, memory_monitoring, auto_assert
                    )
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return asyncio.run(self._execute_timed_operation(
                        lambda: func(*args, **kwargs), (), {}, operation_name,
                        expected_max_duration_ms, memory_monitoring, auto_assert
                    ))
                return sync_wrapper
        return decorator
    
    async def _execute_timed_operation(self,
                                     func: Callable,
                                     args: tuple,
                                     kwargs: dict,
                                     operation_name: str,
                                     expected_max_duration_ms: Optional[float],
                                     memory_monitoring: bool,
                                     auto_assert: bool):
        """Execute timed operation with comprehensive monitoring."""
        
        # Memory baseline
        memory_start = self._get_memory_usage() if memory_monitoring else None
        
        # Start timing
        start_time = time.time()
        exception_occurred = None
        result = None
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
        except Exception as e:
            exception_occurred = e
            self.logger.warning(f"Exception in timed operation {operation_name}: {e}")
        
        # End timing
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Memory usage
        memory_end = self._get_memory_usage() if memory_monitoring else None
        memory_delta = (memory_end - memory_start) if (memory_start and memory_end) else None
        
        # Record performance data
        self.performance_data[f"{operation_name}_duration_ms"].append(duration_ms)
        if memory_delta:
            self.performance_data[f"{operation_name}_memory_delta_mb"].append(memory_delta)
        
        # Create metrics
        metrics = {
            'duration_ms': duration_ms,
            'memory_start_mb': memory_start,
            'memory_end_mb': memory_end,
            'memory_delta_mb': memory_delta,
            'exception_occurred': exception_occurred is not None,
            'operation_name': operation_name
        }
        
        # Auto-assertion
        if auto_assert and expected_max_duration_ms:
            self.assert_response_time(
                duration_ms, 
                expected_max_duration_ms,
                f"{operation_name}_response_time"
            )
        
        # Log results
        status = "FAILED" if exception_occurred else "PASSED"
        self.logger.info(
            f"Performance [{status}] {operation_name}: "
            f"{duration_ms:.2f}ms"
            + (f", Memory Δ{memory_delta:+.2f}MB" if memory_delta else "")
        )
        
        if exception_occurred:
            raise exception_occurred
        
        return result, metrics
    
    @contextmanager
    def timing_context(self,
                      context_name: str,
                      expected_max_duration_ms: Optional[float] = None,
                      memory_monitoring: bool = True):
        """Context manager for timing operations."""
        memory_start = self._get_memory_usage() if memory_monitoring else None
        start_time = time.time()
        
        try:
            yield {
                'start_time': start_time,
                'context_name': context_name,
                'memory_start': memory_start
            }
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            memory_end = self._get_memory_usage() if memory_monitoring else None
            memory_delta = (memory_end - memory_start) if (memory_start and memory_end) else None
            
            # Record performance data
            self.performance_data[f"{context_name}_duration_ms"].append(duration_ms)
            if memory_delta:
                self.performance_data[f"{context_name}_memory_delta_mb"].append(memory_delta)
            
            # Auto-assertion
            if expected_max_duration_ms:
                self.assert_response_time(
                    duration_ms,
                    expected_max_duration_ms,
                    f"{context_name}_context"
                )
            
            self.logger.info(
                f"Performance Context {context_name}: "
                f"{duration_ms:.2f}ms"
                + (f", Memory Δ{memory_delta:+.2f}MB" if memory_delta else "")
            )
    
    # Memory Validation
    
    def establish_memory_baseline(self):
        """Establish memory usage baseline."""
        gc.collect()  # Force garbage collection
        self.memory_baseline = self._get_memory_usage()
        self.logger.info(f"Memory baseline established: {self.memory_baseline:.2f}MB")
    
    def assert_memory_usage(self,
                           max_memory_mb: float,
                           assertion_name: str = "memory_usage",
                           measure_from_baseline: bool = True) -> PerformanceAssertionResult:
        """Assert memory usage is within limits."""
        current_memory = self._get_memory_usage()
        
        if measure_from_baseline and self.memory_baseline:
            measured_value = current_memory - self.memory_baseline
            metric_name = "memory_increase_from_baseline_mb"
        else:
            measured_value = current_memory
            metric_name = "current_memory_usage_mb"
        
        threshold = PerformanceThreshold(
            metric_name, max_memory_mb, 'lte', 'MB', 'warning',
            f"Memory usage should be under {max_memory_mb}MB"
        )
        
        passed, message = threshold.check(measured_value)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=measured_value,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            additional_metrics={
                'current_memory_mb': current_memory,
                'baseline_memory_mb': self.memory_baseline
            }
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.warning(f"Memory assertion failed: {message}")
            if threshold.severity == 'error':
                raise AssertionError(f"Memory usage assertion failed: {message}")
        else:
            self.logger.info(f"Memory assertion passed: {message}")
        
        return result
    
    def assert_memory_leak_absent(self,
                                 tolerance_mb: float = 50.0,
                                 assertion_name: str = "memory_leak_check") -> PerformanceAssertionResult:
        """Assert no significant memory leak occurred."""
        if not self.memory_baseline:
            self.establish_memory_baseline()
            return self.assert_memory_usage(tolerance_mb, assertion_name, measure_from_baseline=True)
        
        gc.collect()  # Force cleanup before measuring
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.memory_baseline
        
        threshold = PerformanceThreshold(
            "memory_leak_detection_mb", tolerance_mb, 'lte', 'MB', 'warning',
            f"Memory increase should be under {tolerance_mb}MB to avoid memory leaks"
        )
        
        passed, message = threshold.check(memory_increase)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=memory_increase,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            additional_metrics={
                'baseline_memory_mb': self.memory_baseline,
                'current_memory_mb': current_memory,
                'memory_increase_mb': memory_increase
            }
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.warning(f"Memory leak assertion failed: {message}")
        else:
            self.logger.info(f"Memory leak assertion passed: {message}")
        
        return result
    
    # Throughput Calculation
    
    def calculate_throughput(self,
                           operation_count: int,
                           duration_seconds: float,
                           assertion_name: str = "throughput") -> float:
        """Calculate and optionally assert throughput."""
        if duration_seconds <= 0:
            self.logger.warning("Invalid duration for throughput calculation")
            return 0.0
        
        throughput = operation_count / duration_seconds
        self.performance_data['throughput_ops_per_sec'].append(throughput)
        
        self.logger.info(f"Calculated throughput: {throughput:.2f} ops/sec ({operation_count} ops in {duration_seconds:.2f}s)")
        
        return throughput
    
    def assert_throughput(self,
                         operation_count: int,
                         duration_seconds: float,
                         min_throughput_ops_per_sec: float,
                         assertion_name: str = "throughput") -> PerformanceAssertionResult:
        """Assert minimum throughput requirement."""
        throughput = self.calculate_throughput(operation_count, duration_seconds, assertion_name)
        
        threshold = PerformanceThreshold(
            "throughput_ops_per_sec", min_throughput_ops_per_sec, 'gte', 'ops/sec', 'error',
            f"Throughput should be at least {min_throughput_ops_per_sec} operations per second"
        )
        
        passed, message = threshold.check(throughput)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=throughput,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            additional_metrics={
                'operation_count': operation_count,
                'duration_seconds': duration_seconds
            }
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.error(f"Throughput assertion failed: {message}")
            raise AssertionError(f"Throughput assertion failed: {message}")
        else:
            self.logger.info(f"Throughput assertion passed: {message}")
        
        return result
    
    # Response Time Assertions
    
    def assert_response_time(self,
                           actual_duration_ms: float,
                           max_duration_ms: float,
                           assertion_name: str = "response_time") -> PerformanceAssertionResult:
        """Assert response time is within acceptable limits."""
        threshold = PerformanceThreshold(
            "response_time_ms", max_duration_ms, 'lte', 'ms', 'error',
            f"Response time should be under {max_duration_ms}ms"
        )
        
        passed, message = threshold.check(actual_duration_ms)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=actual_duration_ms,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            duration_ms=actual_duration_ms
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.error(f"Response time assertion failed: {message}")
            raise AssertionError(f"Response time assertion failed: {message}")
        else:
            self.logger.info(f"Response time assertion passed: {message}")
        
        return result
    
    def assert_percentile_response_time(self,
                                      durations_ms: List[float],
                                      percentile: int,
                                      max_duration_ms: float,
                                      assertion_name: str = "percentile_response_time") -> PerformanceAssertionResult:
        """Assert percentile response time."""
        if not durations_ms:
            raise ValueError("No durations provided for percentile calculation")
        
        percentile_value = np.percentile(durations_ms, percentile)
        
        threshold = PerformanceThreshold(
            f"p{percentile}_response_time_ms", max_duration_ms, 'lte', 'ms', 'error',
            f"P{percentile} response time should be under {max_duration_ms}ms"
        )
        
        passed, message = threshold.check(percentile_value)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=percentile_value,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            additional_metrics={
                'percentile': percentile,
                'total_samples': len(durations_ms),
                'min_duration_ms': min(durations_ms),
                'max_duration_ms': max(durations_ms),
                'avg_duration_ms': statistics.mean(durations_ms)
            }
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.error(f"Percentile response time assertion failed: {message}")
            raise AssertionError(f"Percentile response time assertion failed: {message}")
        else:
            self.logger.info(f"Percentile response time assertion passed: {message}")
        
        return result
    
    # Resource Usage Assertions
    
    def assert_cpu_usage(self,
                        max_cpu_percent: float,
                        duration_seconds: float = 5.0,
                        assertion_name: str = "cpu_usage") -> PerformanceAssertionResult:
        """Assert CPU usage over a monitoring period."""
        cpu_samples = []
        monitor_start = time.time()
        
        while time.time() - monitor_start < duration_seconds:
            try:
                cpu_percent = psutil.Process().cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
            except:
                break
            time.sleep(0.1)
        
        if not cpu_samples:
            cpu_percent = 0.0
        else:
            cpu_percent = statistics.mean(cpu_samples)
        
        threshold = PerformanceThreshold(
            "cpu_usage_percent", max_cpu_percent, 'lte', '%', 'warning',
            f"CPU usage should be under {max_cpu_percent}%"
        )
        
        passed, message = threshold.check(cpu_percent)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=cpu_percent,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            additional_metrics={
                'monitoring_duration_seconds': duration_seconds,
                'cpu_samples_count': len(cpu_samples),
                'max_cpu_sample': max(cpu_samples) if cpu_samples else 0,
                'min_cpu_sample': min(cpu_samples) if cpu_samples else 0
            }
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.warning(f"CPU usage assertion failed: {message}")
            if threshold.severity == 'error':
                raise AssertionError(f"CPU usage assertion failed: {message}")
        else:
            self.logger.info(f"CPU usage assertion passed: {message}")
        
        return result
    
    # Composite Performance Assertions
    
    def assert_performance_benchmark(self,
                                   metrics: PerformanceMetrics,
                                   benchmark_thresholds: Dict[str, PerformanceThreshold],
                                   assertion_name: str = "performance_benchmark") -> Dict[str, PerformanceAssertionResult]:
        """Assert multiple performance metrics against benchmarks."""
        results = {}
        
        # Map metrics to threshold checks
        metric_mappings = {
            'throughput_ops_per_sec': metrics.throughput_ops_per_sec,
            'average_latency_ms': metrics.average_latency_ms,
            'p95_latency_ms': metrics.p95_latency_ms,
            'error_rate_percent': metrics.error_rate_percent,
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_usage_percent': metrics.cpu_usage_percent
        }
        
        for threshold_name, threshold in benchmark_thresholds.items():
            if threshold_name in metric_mappings:
                measured_value = metric_mappings[threshold_name]
                passed, message = threshold.check(measured_value)
                
                result = PerformanceAssertionResult(
                    assertion_name=f"{assertion_name}_{threshold_name}",
                    passed=passed,
                    measured_value=measured_value,
                    threshold=threshold,
                    message=message,
                    timestamp=time.time(),
                    additional_metrics={'source_metrics': asdict(metrics)}
                )
                
                results[threshold_name] = result
                self.assertion_results.append(result)
                
                if not passed and threshold.severity == 'error':
                    self.logger.error(f"Benchmark assertion failed: {message}")
                    raise AssertionError(f"Benchmark assertion failed: {message}")
                elif not passed:
                    self.logger.warning(f"Benchmark assertion warning: {message}")
                else:
                    self.logger.info(f"Benchmark assertion passed: {message}")
        
        return results
    
    # Error Rate Assertions
    
    def assert_error_rate(self,
                         error_count: int,
                         total_count: int,
                         max_error_rate_percent: float,
                         assertion_name: str = "error_rate") -> PerformanceAssertionResult:
        """Assert error rate is within acceptable limits."""
        if total_count == 0:
            error_rate_percent = 0.0
        else:
            error_rate_percent = (error_count / total_count) * 100
        
        threshold = PerformanceThreshold(
            "error_rate_percent", max_error_rate_percent, 'lte', '%', 'error',
            f"Error rate should be under {max_error_rate_percent}%"
        )
        
        passed, message = threshold.check(error_rate_percent)
        
        result = PerformanceAssertionResult(
            assertion_name=assertion_name,
            passed=passed,
            measured_value=error_rate_percent,
            threshold=threshold,
            message=message,
            timestamp=time.time(),
            additional_metrics={
                'error_count': error_count,
                'total_count': total_count,
                'success_count': total_count - error_count
            }
        )
        
        self.assertion_results.append(result)
        
        if not passed:
            self.logger.error(f"Error rate assertion failed: {message}")
            raise AssertionError(f"Error rate assertion failed: {message}")
        else:
            self.logger.info(f"Error rate assertion passed: {message}")
        
        return result
    
    # Utility Methods
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_assertion_summary(self) -> Dict[str, Any]:
        """Get summary of all assertions."""
        total_assertions = len(self.assertion_results)
        passed_assertions = sum(1 for r in self.assertion_results if r.passed)
        failed_assertions = total_assertions - passed_assertions
        
        return {
            'total_assertions': total_assertions,
            'passed_assertions': passed_assertions,
            'failed_assertions': failed_assertions,
            'success_rate_percent': (passed_assertions / total_assertions * 100) if total_assertions > 0 else 100,
            'assertions': [asdict(r) for r in self.assertion_results],
            'performance_data': dict(self.performance_data)
        }
    
    def reset_assertions(self):
        """Reset all assertion results and performance data."""
        self.assertion_results.clear()
        self.performance_data.clear()
        self.memory_baseline = None
        self.active_timers.clear()
    
    def export_results_to_json(self, filepath: Path) -> None:
        """Export assertion results to JSON file."""
        summary = self.get_assertion_summary()
        summary['export_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Performance assertion results exported to {filepath}")


# =====================================================================
# PERFORMANCE BENCHMARK SUITE
# =====================================================================

@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark execution."""
    benchmark_name: str
    description: str
    target_thresholds: Dict[str, PerformanceThreshold]
    test_scenarios: List[LoadTestScenario]
    baseline_comparison: bool = True
    regression_detection: bool = True
    resource_monitoring: bool = True
    detailed_reporting: bool = True


class PerformanceBenchmarkSuite:
    """
    Comprehensive benchmark suite that runs standardized performance tests
    across different scenarios and tracks metrics over time.
    """
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 environment_manager: Optional[TestEnvironmentManager] = None):
        self.output_dir = output_dir or Path("performance_benchmarks")
        self.output_dir.mkdir(exist_ok=True)
        
        self.environment_manager = environment_manager
        self.assertion_helper = PerformanceAssertionHelper()
        self.report_generator = PerformanceReportGenerator(self.output_dir)
        
        self.benchmark_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Standard benchmark configurations
        self.standard_benchmarks = self._create_standard_benchmarks()
    
    def _create_standard_benchmarks(self) -> Dict[str, BenchmarkConfiguration]:
        """Create standard benchmark configurations."""
        return {
            'clinical_query_performance': BenchmarkConfiguration(
                benchmark_name='clinical_query_performance',
                description='Benchmark clinical query processing performance',
                target_thresholds={
                    'response_time_ms': PerformanceThreshold(
                        'response_time_ms', 5000, 'lte', 'ms', 'error',
                        'Clinical queries should respond within 5 seconds'
                    ),
                    'throughput_ops_per_sec': PerformanceThreshold(
                        'throughput_ops_per_sec', 2.0, 'gte', 'ops/sec', 'error',
                        'Should process at least 2 queries per second'
                    ),
                    'error_rate_percent': PerformanceThreshold(
                        'error_rate_percent', 5.0, 'lte', '%', 'error',
                        'Error rate should be under 5%'
                    ),
                    'memory_usage_mb': PerformanceThreshold(
                        'memory_usage_mb', 800, 'lte', 'MB', 'warning',
                        'Memory usage should be under 800MB'
                    )
                },
                test_scenarios=[
                    LoadTestScenarioGenerator.create_light_load_scenario(),
                    LoadTestScenarioGenerator.create_moderate_load_scenario()
                ]
            ),
            
            'pdf_processing_performance': BenchmarkConfiguration(
                benchmark_name='pdf_processing_performance',
                description='Benchmark PDF processing and ingestion performance',
                target_thresholds={
                    'response_time_ms': PerformanceThreshold(
                        'response_time_ms', 15000, 'lte', 'ms', 'error',
                        'PDF processing should complete within 15 seconds'
                    ),
                    'throughput_ops_per_sec': PerformanceThreshold(
                        'throughput_ops_per_sec', 0.5, 'gte', 'ops/sec', 'error',
                        'Should process at least 0.5 PDFs per second'
                    ),
                    'memory_usage_mb': PerformanceThreshold(
                        'memory_usage_mb', 1200, 'lte', 'MB', 'warning',
                        'Memory usage should be under 1.2GB for PDF processing'
                    )
                },
                test_scenarios=[
                    LoadTestScenarioGenerator.create_baseline_scenario(),
                    LoadTestScenarioGenerator.create_light_load_scenario()
                ]
            ),
            
            'scalability_benchmark': BenchmarkConfiguration(
                benchmark_name='scalability_benchmark',
                description='Benchmark system scalability under increasing load',
                target_thresholds={
                    'throughput_ops_per_sec': PerformanceThreshold(
                        'throughput_ops_per_sec', 10.0, 'gte', 'ops/sec', 'error',
                        'Should maintain at least 10 ops/sec under load'
                    ),
                    'p95_latency_ms': PerformanceThreshold(
                        'p95_latency_ms', 10000, 'lte', 'ms', 'error',
                        'P95 latency should be under 10 seconds'
                    ),
                    'error_rate_percent': PerformanceThreshold(
                        'error_rate_percent', 10.0, 'lte', '%', 'error',
                        'Error rate should be under 10% under heavy load'
                    )
                },
                test_scenarios=[
                    LoadTestScenarioGenerator.create_moderate_load_scenario(),
                    LoadTestScenarioGenerator.create_heavy_load_scenario(),
                    LoadTestScenarioGenerator.create_spike_test_scenario()
                ]
            ),
            
            'endurance_benchmark': BenchmarkConfiguration(
                benchmark_name='endurance_benchmark',
                description='Benchmark system endurance and stability over time',
                target_thresholds={
                    'memory_usage_mb': PerformanceThreshold(
                        'memory_usage_mb', 1000, 'lte', 'MB', 'error',
                        'Memory should remain stable under 1GB'
                    ),
                    'throughput_ops_per_sec': PerformanceThreshold(
                        'throughput_ops_per_sec', 3.0, 'gte', 'ops/sec', 'error',
                        'Should maintain consistent throughput over time'
                    ),
                    'error_rate_percent': PerformanceThreshold(
                        'error_rate_percent', 5.0, 'lte', '%', 'error',
                        'Error rate should remain low over extended periods'
                    )
                },
                test_scenarios=[
                    LoadTestScenarioGenerator.create_endurance_test_scenario()
                ]
            )
        }
    
    async def run_benchmark_suite(self,
                                benchmark_names: Optional[List[str]] = None,
                                operation_func: Optional[Callable] = None,
                                data_generator: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            benchmark_names: Names of benchmarks to run (None for all)
            operation_func: Function to execute during benchmarking
            data_generator: Function to generate test data
            
        Returns:
            Dictionary containing benchmark results and analysis
        """
        
        if benchmark_names is None:
            benchmark_names = list(self.standard_benchmarks.keys())
        
        # Import mock operation if none provided
        if operation_func is None:
            from performance_test_fixtures import mock_clinical_query_operation
            operation_func = mock_clinical_query_operation
        
        if data_generator is None:
            from performance_test_fixtures import MockOperationGenerator
            mock_generator = MockOperationGenerator()
            data_generator = mock_generator.generate_query_data
        
        self.logger.info(f"Starting benchmark suite: {benchmark_names}")
        
        # Reset assertion helper
        self.assertion_helper.reset_assertions()
        self.assertion_helper.establish_memory_baseline()
        
        benchmark_results = {}
        all_metrics = []
        
        for benchmark_name in benchmark_names:
            if benchmark_name not in self.standard_benchmarks:
                self.logger.warning(f"Unknown benchmark: {benchmark_name}")
                continue
            
            self.logger.info(f"Running benchmark: {benchmark_name}")
            
            benchmark_config = self.standard_benchmarks[benchmark_name]
            benchmark_result = await self._run_single_benchmark(
                benchmark_config, operation_func, data_generator
            )
            
            benchmark_results[benchmark_name] = benchmark_result
            all_metrics.extend(benchmark_result['scenario_metrics'])
        
        # Generate comprehensive report
        suite_report = self._generate_suite_report(benchmark_results, all_metrics)
        
        # Save results
        self._save_benchmark_results(suite_report)
        
        self.logger.info("Benchmark suite completed successfully")
        
        return suite_report
    
    async def _run_single_benchmark(self,
                                   config: BenchmarkConfiguration,
                                   operation_func: Callable,
                                   data_generator: Callable) -> Dict[str, Any]:
        """Run single benchmark configuration."""
        
        scenario_results = []
        scenario_metrics = []
        
        for scenario in config.test_scenarios:
            self.logger.info(f"Executing scenario: {scenario.scenario_name}")
            
            # Execute performance test
            executor = PerformanceTestExecutor()
            
            try:
                metrics = await executor.execute_load_test(
                    scenario, operation_func, data_generator
                )
                
                # Store metrics for history
                self.benchmark_history[config.benchmark_name].append(metrics)
                scenario_metrics.append(metrics)
                
                # Run assertions
                assertion_results = self.assertion_helper.assert_performance_benchmark(
                    metrics, config.target_thresholds,
                    f"{config.benchmark_name}_{scenario.scenario_name}"
                )
                
                scenario_result = {
                    'scenario_name': scenario.scenario_name,
                    'metrics': asdict(metrics),
                    'assertion_results': {k: asdict(v) for k, v in assertion_results.items()},
                    'passed': all(r.passed for r in assertion_results.values()),
                    'benchmark_name': config.benchmark_name
                }
                
                scenario_results.append(scenario_result)
                
                self.logger.info(
                    f"Scenario {scenario.scenario_name} completed - "
                    f"{'PASSED' if scenario_result['passed'] else 'FAILED'}"
                )
                
            except Exception as e:
                self.logger.error(f"Scenario {scenario.scenario_name} failed with exception: {e}")
                scenario_result = {
                    'scenario_name': scenario.scenario_name,
                    'error': str(e),
                    'passed': False,
                    'benchmark_name': config.benchmark_name
                }
                scenario_results.append(scenario_result)
        
        # Analyze benchmark results
        benchmark_analysis = self._analyze_benchmark_results(scenario_metrics, config)
        
        return {
            'benchmark_name': config.benchmark_name,
            'description': config.description,
            'scenario_results': scenario_results,
            'scenario_metrics': scenario_metrics,
            'analysis': benchmark_analysis,
            'passed': all(r['passed'] for r in scenario_results),
            'execution_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_benchmark_results(self,
                                 metrics_list: List[PerformanceMetrics],
                                 config: BenchmarkConfiguration) -> Dict[str, Any]:
        """Analyze benchmark results for trends and patterns."""
        
        if not metrics_list:
            return {'error': 'No metrics available for analysis'}
        
        # Aggregate statistics
        response_times = [m.average_latency_ms for m in metrics_list]
        throughputs = [m.throughput_ops_per_sec for m in metrics_list]
        error_rates = [m.error_rate_percent for m in metrics_list]
        memory_usage = [m.memory_usage_mb for m in metrics_list]
        
        analysis = {
            'total_scenarios': len(metrics_list),
            'aggregated_stats': {
                'avg_response_time_ms': statistics.mean(response_times),
                'median_response_time_ms': statistics.median(response_times),
                'max_response_time_ms': max(response_times),
                'avg_throughput_ops_per_sec': statistics.mean(throughputs),
                'total_operations': sum(m.operations_count for m in metrics_list),
                'total_errors': sum(m.failure_count for m in metrics_list),
                'avg_error_rate_percent': statistics.mean(error_rates),
                'peak_memory_mb': max(memory_usage),
                'avg_memory_mb': statistics.mean(memory_usage)
            },
            'performance_variance': {
                'response_time_std': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'throughput_std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                'consistency_score': self._calculate_consistency_score(response_times, throughputs)
            }
        }
        
        # Compare against thresholds
        threshold_analysis = {}
        for threshold_name, threshold in config.target_thresholds.items():
            if threshold_name in analysis['aggregated_stats']:
                value = analysis['aggregated_stats'][threshold_name]
                passed, message = threshold.check(value)
                threshold_analysis[threshold_name] = {
                    'passed': passed,
                    'message': message,
                    'measured_value': value,
                    'threshold_value': threshold.threshold_value
                }
        
        analysis['threshold_analysis'] = threshold_analysis
        
        # Regression analysis if historical data exists
        if len(self.benchmark_history[config.benchmark_name]) > 1:
            analysis['regression_analysis'] = self._analyze_regression(
                config.benchmark_name, metrics_list[-1]
            )
        
        return analysis
    
    def _calculate_consistency_score(self, 
                                   response_times: List[float], 
                                   throughputs: List[float]) -> float:
        """Calculate performance consistency score (0-100)."""
        if len(response_times) <= 1:
            return 100.0
        
        # Lower variance = higher consistency
        response_time_cv = statistics.stdev(response_times) / statistics.mean(response_times)
        throughput_cv = statistics.stdev(throughputs) / statistics.mean(throughputs) if statistics.mean(throughputs) > 0 else 0
        
        # Combined coefficient of variation (lower is better)
        combined_cv = (response_time_cv + throughput_cv) / 2
        
        # Convert to consistency score (higher is better)
        consistency_score = max(0, 100 - (combined_cv * 100))
        
        return min(consistency_score, 100.0)
    
    def _analyze_regression(self, 
                          benchmark_name: str, 
                          current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance regression compared to historical data."""
        
        historical_metrics = self.benchmark_history[benchmark_name][:-1]  # Exclude current
        
        if not historical_metrics:
            return {'status': 'no_historical_data'}
        
        # Calculate historical averages
        historical_response_times = [m.average_latency_ms for m in historical_metrics]
        historical_throughputs = [m.throughput_ops_per_sec for m in historical_metrics]
        historical_error_rates = [m.error_rate_percent for m in historical_metrics]
        
        avg_historical_response_time = statistics.mean(historical_response_times)
        avg_historical_throughput = statistics.mean(historical_throughputs)
        avg_historical_error_rate = statistics.mean(historical_error_rates)
        
        # Calculate changes
        response_time_change = ((current_metrics.average_latency_ms - avg_historical_response_time) 
                               / avg_historical_response_time * 100) if avg_historical_response_time > 0 else 0
        
        throughput_change = ((current_metrics.throughput_ops_per_sec - avg_historical_throughput) 
                            / avg_historical_throughput * 100) if avg_historical_throughput > 0 else 0
        
        error_rate_change = current_metrics.error_rate_percent - avg_historical_error_rate
        
        # Determine regression status
        regressions = []
        improvements = []
        
        if response_time_change > 20:  # 20% increase in response time
            regressions.append('response_time_degradation')
        elif response_time_change < -10:  # 10% improvement
            improvements.append('response_time_improvement')
        
        if throughput_change < -15:  # 15% decrease in throughput
            regressions.append('throughput_degradation')
        elif throughput_change > 10:  # 10% improvement
            improvements.append('throughput_improvement')
        
        if error_rate_change > 5:  # 5% increase in error rate
            regressions.append('error_rate_increase')
        elif error_rate_change < -2:  # 2% improvement
            improvements.append('error_rate_improvement')
        
        return {
            'status': 'regression_detected' if regressions else 'no_regression',
            'regressions': regressions,
            'improvements': improvements,
            'changes': {
                'response_time_change_percent': response_time_change,
                'throughput_change_percent': throughput_change,
                'error_rate_change_percent': error_rate_change
            },
            'historical_baseline': {
                'avg_response_time_ms': avg_historical_response_time,
                'avg_throughput_ops_per_sec': avg_historical_throughput,
                'avg_error_rate_percent': avg_historical_error_rate,
                'samples_count': len(historical_metrics)
            }
        }
    
    def _generate_suite_report(self, 
                              benchmark_results: Dict[str, Any], 
                              all_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate comprehensive benchmark suite report."""
        
        total_benchmarks = len(benchmark_results)
        passed_benchmarks = sum(1 for r in benchmark_results.values() if r['passed'])
        
        # Overall statistics
        overall_stats = {}
        if all_metrics:
            overall_stats = {
                'total_operations': sum(m.operations_count for m in all_metrics),
                'total_successful_operations': sum(m.success_count for m in all_metrics),
                'total_failed_operations': sum(m.failure_count for m in all_metrics),
                'average_response_time_ms': statistics.mean([m.average_latency_ms for m in all_metrics]),
                'average_throughput_ops_per_sec': statistics.mean([m.throughput_ops_per_sec for m in all_metrics]),
                'peak_memory_usage_mb': max([m.memory_usage_mb for m in all_metrics]),
                'overall_error_rate_percent': statistics.mean([m.error_rate_percent for m in all_metrics])
            }
        
        suite_report = {
            'suite_execution_summary': {
                'execution_timestamp': datetime.now().isoformat(),
                'total_benchmarks': total_benchmarks,
                'passed_benchmarks': passed_benchmarks,
                'failed_benchmarks': total_benchmarks - passed_benchmarks,
                'success_rate_percent': (passed_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 100
            },
            'overall_performance_statistics': overall_stats,
            'benchmark_results': benchmark_results,
            'assertion_summary': self.assertion_helper.get_assertion_summary(),
            'recommendations': self._generate_suite_recommendations(benchmark_results)
        }
        
        return suite_report
    
    def _generate_suite_recommendations(self, 
                                      benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        
        recommendations = []
        
        # Analyze failed benchmarks
        failed_benchmarks = [name for name, result in benchmark_results.items() if not result['passed']]
        
        if failed_benchmarks:
            recommendations.append(
                f"Address performance issues in failed benchmarks: {', '.join(failed_benchmarks)}"
            )
        
        # Analyze common patterns across benchmarks
        all_analyses = [result.get('analysis', {}) for result in benchmark_results.values()]
        
        # Check for consistent high response times
        high_response_times = [
            analysis.get('aggregated_stats', {}).get('avg_response_time_ms', 0)
            for analysis in all_analyses
        ]
        
        if high_response_times and statistics.mean(high_response_times) > 5000:
            recommendations.append(
                "Overall response times are high - consider optimizing query processing pipeline"
            )
        
        # Check for memory usage trends
        peak_memory_usage = [
            analysis.get('aggregated_stats', {}).get('peak_memory_mb', 0)
            for analysis in all_analyses
        ]
        
        if peak_memory_usage and max(peak_memory_usage) > 1000:
            recommendations.append(
                "Peak memory usage exceeds 1GB - implement memory optimization strategies"
            )
        
        # Check for consistency issues
        consistency_scores = [
            analysis.get('performance_variance', {}).get('consistency_score', 100)
            for analysis in all_analyses
        ]
        
        if consistency_scores and statistics.mean(consistency_scores) < 70:
            recommendations.append(
                "Performance consistency is low - investigate system instability or resource contention"
            )
        
        # Check for regressions
        regression_detected = any(
            analysis.get('regression_analysis', {}).get('status') == 'regression_detected'
            for analysis in all_analyses
        )
        
        if regression_detected:
            recommendations.append(
                "Performance regressions detected - review recent changes and optimize affected components"
            )
        
        if not recommendations:
            recommendations.append("All benchmarks passed - system performance is meeting expectations")
        
        return recommendations
    
    def _save_benchmark_results(self, suite_report: Dict[str, Any]):
        """Save benchmark results to files."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_path = self.output_dir / f"benchmark_suite_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(suite_report, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_path = self.output_dir / f"benchmark_suite_{timestamp}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_text(suite_report))
        
        # Export assertion results
        assertion_path = self.output_dir / f"benchmark_assertions_{timestamp}.json"
        self.assertion_helper.export_results_to_json(assertion_path)
        
        self.logger.info(f"Benchmark results saved to {json_path}")
    
    def _generate_summary_text(self, suite_report: Dict[str, Any]) -> str:
        """Generate human-readable summary text."""
        
        summary = suite_report['suite_execution_summary']
        stats = suite_report.get('overall_performance_statistics', {})
        recommendations = suite_report.get('recommendations', [])
        
        text = f"""
CLINICAL METABOLOMICS ORACLE - BENCHMARK SUITE REPORT
=====================================================

Execution Summary:
- Timestamp: {summary['execution_timestamp']}
- Total Benchmarks: {summary['total_benchmarks']}
- Passed: {summary['passed_benchmarks']}
- Failed: {summary['failed_benchmarks']}
- Success Rate: {summary['success_rate_percent']:.1f}%

Overall Performance Statistics:
- Total Operations: {stats.get('total_operations', 0):,}
- Successful Operations: {stats.get('total_successful_operations', 0):,}
- Failed Operations: {stats.get('total_failed_operations', 0):,}
- Average Response Time: {stats.get('average_response_time_ms', 0):.1f} ms
- Average Throughput: {stats.get('average_throughput_ops_per_sec', 0):.2f} ops/sec
- Peak Memory Usage: {stats.get('peak_memory_usage_mb', 0):.1f} MB
- Overall Error Rate: {stats.get('overall_error_rate_percent', 0):.1f}%

Benchmark Results:
"""
        
        for benchmark_name, result in suite_report['benchmark_results'].items():
            status = "PASSED" if result['passed'] else "FAILED"
            text += f"- {benchmark_name}: {status}\n"
        
        text += "\nRecommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            text += f"{i}. {rec}\n"
        
        text += "\nFor detailed results and metrics, see the complete JSON report.\n"
        
        return text
    
    def set_baseline_metrics(self, 
                           benchmark_name: str, 
                           metrics: PerformanceMetrics):
        """Set baseline metrics for comparison."""
        self.baseline_metrics[benchmark_name] = metrics
        self.logger.info(f"Baseline metrics set for benchmark: {benchmark_name}")
    
    def compare_against_baseline(self, 
                               benchmark_name: str, 
                               current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Compare current metrics against established baseline."""
        
        if benchmark_name not in self.baseline_metrics:
            return {'status': 'no_baseline_available'}
        
        baseline = self.baseline_metrics[benchmark_name]
        
        comparison = {
            'benchmark_name': benchmark_name,
            'baseline_timestamp': baseline.start_time,
            'current_timestamp': current_metrics.start_time,
            'performance_changes': {
                'response_time_change_ms': current_metrics.average_latency_ms - baseline.average_latency_ms,
                'throughput_change_ops_per_sec': current_metrics.throughput_ops_per_sec - baseline.throughput_ops_per_sec,
                'error_rate_change_percent': current_metrics.error_rate_percent - baseline.error_rate_percent,
                'memory_change_mb': current_metrics.memory_usage_mb - baseline.memory_usage_mb
            },
            'performance_ratios': {
                'response_time_ratio': current_metrics.average_latency_ms / baseline.average_latency_ms if baseline.average_latency_ms > 0 else 1.0,
                'throughput_ratio': current_metrics.throughput_ops_per_sec / baseline.throughput_ops_per_sec if baseline.throughput_ops_per_sec > 0 else 1.0,
                'memory_ratio': current_metrics.memory_usage_mb / baseline.memory_usage_mb if baseline.memory_usage_mb > 0 else 1.0
            }
        }
        
        # Determine overall trend
        response_ratio = comparison['performance_ratios']['response_time_ratio']
        throughput_ratio = comparison['performance_ratios']['throughput_ratio']
        
        if response_ratio <= 0.9 and throughput_ratio >= 1.1:
            comparison['trend'] = 'improvement'
        elif response_ratio >= 1.2 or throughput_ratio <= 0.8:
            comparison['trend'] = 'degradation'
        else:
            comparison['trend'] = 'stable'
        
        return comparison


# =====================================================================
# ADVANCED RESOURCE MONITORING UTILITIES
# =====================================================================

@dataclass
class ResourceAlert:
    """Resource usage alert."""
    alert_type: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    timestamp: float
    message: str
    suggested_action: str = ""


class AdvancedResourceMonitor(ResourceMonitor):
    """
    Advanced resource monitoring with threshold-based alerts,
    trend analysis, and detailed diagnostics.
    """
    
    def __init__(self, 
                 sampling_interval: float = 1.0,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        super().__init__(sampling_interval)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 85.0,
            'memory_mb': 1000.0,
            'memory_percent': 80.0,
            'disk_io_read_mb_per_sec': 100.0,
            'disk_io_write_mb_per_sec': 100.0,
            'active_threads': 50,
            'open_file_descriptors': 1000
        }
        
        self.alerts: List[ResourceAlert] = []
        self.trend_data: Dict[str, deque] = {
            'cpu_trend': deque(maxlen=10),
            'memory_trend': deque(maxlen=10),
            'io_trend': deque(maxlen=10)
        }
        
        # Previous values for rate calculations
        self.previous_snapshot: Optional[ResourceUsageSnapshot] = None
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start advanced resource monitoring with alerting."""
        self.alerts.clear()
        super().start_monitoring()
        self.logger.info("Advanced resource monitoring started with alerting")
    
    def _monitor_loop(self):
        """Enhanced monitoring loop with alerting and trend analysis."""
        if not self.system_available:
            return
        
        initial_net_io = psutil.net_io_counters()
        initial_disk_io = psutil.disk_io_counters()
        
        while self.monitoring:
            try:
                # Get standard snapshot
                snapshot = self._create_snapshot(initial_net_io, initial_disk_io)
                self.snapshots.append(snapshot)
                
                # Check alerts
                self._check_resource_alerts(snapshot)
                
                # Update trend data
                self._update_trends(snapshot)
                
                # Store for rate calculations
                self.previous_snapshot = snapshot
                
            except Exception as e:
                self.logger.debug(f"Monitoring sample failed: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _create_snapshot(self, initial_net_io, initial_disk_io) -> ResourceUsageSnapshot:
        """Create resource usage snapshot."""
        # CPU and memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Disk I/O
        try:
            current_disk_io = psutil.disk_io_counters()
            disk_read_mb = (current_disk_io.read_bytes - initial_disk_io.read_bytes) / 1024 / 1024
            disk_write_mb = (current_disk_io.write_bytes - initial_disk_io.write_bytes) / 1024 / 1024
        except:
            disk_read_mb = disk_write_mb = 0.0
        
        # Network I/O
        try:
            current_net_io = psutil.net_io_counters()
            net_sent = current_net_io.bytes_sent - initial_net_io.bytes_sent
            net_received = current_net_io.bytes_recv - initial_net_io.bytes_recv
        except:
            net_sent = net_received = 0.0
        
        # Thread and file descriptor count
        active_threads = threading.active_count()
        try:
            open_fds = self.process.num_fds()
        except:
            open_fds = 0
        
        return ResourceUsageSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=net_sent,
            network_bytes_received=net_received,
            active_threads=active_threads,
            open_file_descriptors=open_fds
        )
    
    def _check_resource_alerts(self, snapshot: ResourceUsageSnapshot):
        """Check for resource usage alerts."""
        current_time = time.time()
        
        # CPU alert
        if snapshot.cpu_percent > self.alert_thresholds.get('cpu_percent', 85.0):
            alert = ResourceAlert(
                alert_type='cpu_high',
                metric_name='cpu_percent',
                current_value=snapshot.cpu_percent,
                threshold_value=self.alert_thresholds['cpu_percent'],
                severity='warning',
                timestamp=current_time,
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                suggested_action="Consider optimizing CPU-intensive operations"
            )
            self.alerts.append(alert)
            self.logger.warning(alert.message)
        
        # Memory alert
        if snapshot.memory_mb > self.alert_thresholds.get('memory_mb', 1000.0):
            alert = ResourceAlert(
                alert_type='memory_high',
                metric_name='memory_mb',
                current_value=snapshot.memory_mb,
                threshold_value=self.alert_thresholds['memory_mb'],
                severity='warning',
                timestamp=current_time,
                message=f"High memory usage: {snapshot.memory_mb:.1f}MB",
                suggested_action="Consider implementing memory optimization or cleanup"
            )
            self.alerts.append(alert)
            self.logger.warning(alert.message)
        
        # Thread count alert
        if snapshot.active_threads > self.alert_thresholds.get('active_threads', 50):
            alert = ResourceAlert(
                alert_type='threads_high',
                metric_name='active_threads',
                current_value=snapshot.active_threads,
                threshold_value=self.alert_thresholds['active_threads'],
                severity='warning',
                timestamp=current_time,
                message=f"High thread count: {snapshot.active_threads}",
                suggested_action="Consider implementing thread pooling or cleanup"
            )
            self.alerts.append(alert)
            self.logger.warning(alert.message)
        
        # File descriptor alert
        if snapshot.open_file_descriptors > self.alert_thresholds.get('open_file_descriptors', 1000):
            alert = ResourceAlert(
                alert_type='file_descriptors_high',
                metric_name='open_file_descriptors',
                current_value=snapshot.open_file_descriptors,
                threshold_value=self.alert_thresholds['open_file_descriptors'],
                severity='error',
                timestamp=current_time,
                message=f"High file descriptor count: {snapshot.open_file_descriptors}",
                suggested_action="Check for resource leaks and ensure proper file cleanup"
            )
            self.alerts.append(alert)
            self.logger.error(alert.message)
    
    def _update_trends(self, snapshot: ResourceUsageSnapshot):
        """Update trend analysis data."""
        self.trend_data['cpu_trend'].append(snapshot.cpu_percent)
        self.trend_data['memory_trend'].append(snapshot.memory_mb)
        
        # Calculate I/O rate if we have previous snapshot
        if self.previous_snapshot:
            time_delta = snapshot.timestamp - self.previous_snapshot.timestamp
            if time_delta > 0:
                io_rate = (snapshot.disk_io_read_mb + snapshot.disk_io_write_mb) / time_delta
                self.trend_data['io_trend'].append(io_rate)
    
    def get_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends."""
        trends = {}
        
        for trend_name, trend_data in self.trend_data.items():
            if len(trend_data) < 3:
                trends[trend_name] = {'status': 'insufficient_data'}
                continue
            
            trend_values = list(trend_data)
            
            # Calculate linear trend
            x = list(range(len(trend_values)))
            try:
                slope = np.polyfit(x, trend_values, 1)[0]
                
                if abs(slope) < 0.1:
                    direction = 'stable'
                elif slope > 0:
                    direction = 'increasing'
                else:
                    direction = 'decreasing'
                
                trends[trend_name] = {
                    'direction': direction,
                    'slope': slope,
                    'current_value': trend_values[-1],
                    'min_value': min(trend_values),
                    'max_value': max(trend_values),
                    'avg_value': statistics.mean(trend_values)
                }
                
            except Exception:
                trends[trend_name] = {'status': 'calculation_failed'}
        
        return trends
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of resource alerts."""
        if not self.alerts:
            return {'status': 'no_alerts', 'total_alerts': 0}
        
        alert_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for alert in self.alerts:
            alert_counts[alert.alert_type] += 1
            severity_counts[alert.severity] += 1
        
        return {
            'status': 'alerts_generated',
            'total_alerts': len(self.alerts),
            'alert_counts_by_type': dict(alert_counts),
            'alert_counts_by_severity': dict(severity_counts),
            'latest_alert': asdict(self.alerts[-1]) if self.alerts else None,
            'alerts': [asdict(alert) for alert in self.alerts]
        }
    
    def export_monitoring_report(self, filepath: Path):
        """Export comprehensive monitoring report."""
        report = {
            'monitoring_session': {
                'start_time': self.snapshots[0].timestamp if self.snapshots else None,
                'end_time': self.snapshots[-1].timestamp if self.snapshots else None,
                'duration_seconds': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) if len(self.snapshots) > 1 else 0,
                'samples_collected': len(self.snapshots)
            },
            'resource_summary': self.get_resource_summary(),
            'trend_analysis': self.get_resource_trends(),
            'alert_summary': self.get_alert_summary(),
            'detailed_snapshots': [snapshot.to_dict() for snapshot in self.snapshots[-100:]]  # Last 100 samples
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Resource monitoring report exported to {filepath}")


# =====================================================================
# PYTEST FIXTURES FOR PERFORMANCE UTILITIES
# =====================================================================

@pytest.fixture
def performance_assertion_helper():
    """Provide PerformanceAssertionHelper for tests."""
    return PerformanceAssertionHelper()


@pytest.fixture
def performance_benchmark_suite(test_environment_manager):
    """Provide PerformanceBenchmarkSuite for tests."""
    return PerformanceBenchmarkSuite(environment_manager=test_environment_manager)


@pytest.fixture
def advanced_resource_monitor():
    """Provide AdvancedResourceMonitor for tests."""
    return AdvancedResourceMonitor(sampling_interval=0.5)


@pytest.fixture
def performance_thresholds():
    """Provide standard performance thresholds for testing."""
    return {
        'response_time_ms': PerformanceThreshold(
            'response_time_ms', 5000, 'lte', 'ms', 'error',
            'Response time should be under 5 seconds'
        ),
        'throughput_ops_per_sec': PerformanceThreshold(
            'throughput_ops_per_sec', 1.0, 'gte', 'ops/sec', 'error',
            'Throughput should be at least 1 operation per second'
        ),
        'memory_usage_mb': PerformanceThreshold(
            'memory_usage_mb', 500, 'lte', 'MB', 'warning',
            'Memory usage should be under 500MB'
        ),
        'error_rate_percent': PerformanceThreshold(
            'error_rate_percent', 5.0, 'lte', '%', 'error',
            'Error rate should be under 5%'
        )
    }


@pytest.fixture
async def performance_test_with_monitoring():
    """Provide performance test context with full monitoring."""
    
    async def run_monitored_test(
        test_func: Callable,
        assertion_helper: Optional[PerformanceAssertionHelper] = None,
        resource_monitor: Optional[AdvancedResourceMonitor] = None,
        expected_duration_ms: float = 10000
    ):
        """Run test with comprehensive performance monitoring."""
        
        if not assertion_helper:
            assertion_helper = PerformanceAssertionHelper()
        
        if not resource_monitor:
            resource_monitor = AdvancedResourceMonitor(sampling_interval=0.2)
        
        # Establish baselines
        assertion_helper.establish_memory_baseline()
        
        # Start monitoring
        resource_monitor.start_monitoring()
        
        test_start_time = time.time()
        test_exception = None
        test_result = None
        
        try:
            # Execute test
            if asyncio.iscoroutinefunction(test_func):
                test_result = await test_func()
            else:
                test_result = test_func()
                
        except Exception as e:
            test_exception = e
        
        test_end_time = time.time()
        test_duration_ms = (test_end_time - test_start_time) * 1000
        
        # Stop monitoring
        resource_snapshots = resource_monitor.stop_monitoring()
        
        # Run performance assertions
        assertion_helper.assert_response_time(test_duration_ms, expected_duration_ms, "test_execution")
        assertion_helper.assert_memory_leak_absent(100.0, "test_memory_leak")
        
        # Compile results
        results = {
            'test_result': test_result,
            'test_duration_ms': test_duration_ms,
            'test_exception': test_exception,
            'assertion_summary': assertion_helper.get_assertion_summary(),
            'resource_summary': resource_monitor.get_resource_summary(),
            'resource_trends': resource_monitor.get_resource_trends(),
            'alert_summary': resource_monitor.get_alert_summary(),
            'resource_snapshots': resource_snapshots
        }
        
        if test_exception:
            raise test_exception
        
        return results
    
    return run_monitored_test


# Make utilities available at module level
__all__ = [
    'PerformanceThreshold',
    'PerformanceAssertionResult', 
    'PerformanceAssertionHelper',
    'BenchmarkConfiguration',
    'PerformanceBenchmarkSuite',
    'ResourceAlert',
    'AdvancedResourceMonitor'
]