"""
Performance and load testing suite for ProductionCircuitBreaker.

This module provides comprehensive performance testing including:
- High concurrent request handling
- Circuit breaker performance overhead measurement
- Memory usage under load
- Recovery time under different conditions  
- Monitoring system performance impact
"""

import pytest
import asyncio
import time
import threading
import statistics
import random
import psutil
import concurrent.futures
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import gc
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lightrag_integration'))

from production_load_balancer import (
    ProductionCircuitBreaker,
    CircuitBreakerState,
    BackendInstanceConfig,
    BackendType,
    ProductionLoadBalancer,
    ProductionLoadBalancingConfig
)


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def high_performance_config():
    """Configuration optimized for high performance testing"""
    return BackendInstanceConfig(
        id="high_perf_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://localhost:8080",
        api_key="perf_test_key",
        failure_threshold=10,
        recovery_timeout_seconds=5,  # Fast recovery
        half_open_max_requests=20,
        expected_response_time_ms=100.0,  # Very fast expected time
        circuit_breaker_enabled=True
    )

@pytest.fixture
def memory_test_config():
    """Configuration for memory usage testing"""
    return BackendInstanceConfig(
        id="memory_test_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.test.com",
        api_key="memory_test_key",
        failure_threshold=5,
        recovery_timeout_seconds=30,
        expected_response_time_ms=1000.0,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def stress_test_config():
    """Configuration for stress testing scenarios"""
    return BackendInstanceConfig(
        id="stress_test_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.stress.com",
        api_key="stress_test_key", 
        failure_threshold=3,
        recovery_timeout_seconds=60,
        expected_response_time_ms=2000.0,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def benchmark_circuit_breaker(high_performance_config):
    """Create circuit breaker for benchmarking"""
    return ProductionCircuitBreaker(high_performance_config)

@pytest.fixture
def load_test_circuit_breakers():
    """Create multiple circuit breakers for load testing"""
    configs = []
    for i in range(10):
        config = BackendInstanceConfig(
            id=f"load_test_backend_{i}",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url=f"https://api.loadtest{i}.com",
            api_key=f"load_test_key_{i}",
            failure_threshold=5,
            recovery_timeout_seconds=30,
            expected_response_time_ms=1000.0
        )
        configs.append(config)
    
    return [ProductionCircuitBreaker(config) for config in configs]


# ============================================================================
# Performance Baseline Tests
# ============================================================================

class TestPerformanceBaselines:
    """Establish performance baselines for circuit breaker operations"""

    def test_single_request_processing_time(self, benchmark_circuit_breaker):
        """Test time for processing a single request"""
        cb = benchmark_circuit_breaker
        
        # Warm up
        for _ in range(100):
            cb.should_allow_request()
        
        # Measure baseline performance
        start_time = time.perf_counter()
        iterations = 10000
        
        for _ in range(iterations):
            cb.should_allow_request()
        
        end_time = time.perf_counter()
        avg_time_us = ((end_time - start_time) / iterations) * 1_000_000  # microseconds
        
        # Should be very fast - under 10 microseconds per check
        assert avg_time_us < 10.0, f"Single request check took {avg_time_us:.2f}μs, expected < 10μs"
        
        print(f"Average request check time: {avg_time_us:.2f}μs")

    def test_success_recording_performance(self, benchmark_circuit_breaker):
        """Test performance of success recording"""
        cb = benchmark_circuit_breaker
        
        start_time = time.perf_counter()
        iterations = 10000
        
        for i in range(iterations):
            cb.record_success(random.uniform(50, 200))
        
        end_time = time.perf_counter()
        avg_time_us = ((end_time - start_time) / iterations) * 1_000_000
        
        # Should be fast - under 50 microseconds per success recording
        assert avg_time_us < 50.0, f"Success recording took {avg_time_us:.2f}μs, expected < 50μs"
        
        print(f"Average success recording time: {avg_time_us:.2f}μs")

    def test_failure_recording_performance(self, benchmark_circuit_breaker):
        """Test performance of failure recording"""
        cb = benchmark_circuit_breaker
        
        start_time = time.perf_counter()
        iterations = 10000
        
        error_types = ["TimeoutError", "ServerError", "NetworkError", "ValidationError"]
        
        for i in range(iterations):
            error_type = error_types[i % len(error_types)]
            cb.record_failure(f"Test error {i}", error_type=error_type)
        
        end_time = time.perf_counter()
        avg_time_us = ((end_time - start_time) / iterations) * 1_000_000
        
        # Should be reasonably fast - under 100 microseconds per failure recording
        assert avg_time_us < 100.0, f"Failure recording took {avg_time_us:.2f}μs, expected < 100μs"
        
        print(f"Average failure recording time: {avg_time_us:.2f}μs")

    def test_metrics_collection_performance(self, benchmark_circuit_breaker):
        """Test performance of metrics collection"""
        cb = benchmark_circuit_breaker
        
        # Generate some data
        for i in range(1000):
            if i % 3 == 0:
                cb.record_failure(f"Error {i}", error_type="TestError")
            else:
                cb.record_success(random.uniform(100, 500))
        
        start_time = time.perf_counter()
        iterations = 1000
        
        for _ in range(iterations):
            metrics = cb.get_metrics()
        
        end_time = time.perf_counter()
        avg_time_us = ((end_time - start_time) / iterations) * 1_000_000
        
        # Metrics collection should be fast - under 500 microseconds
        assert avg_time_us < 500.0, f"Metrics collection took {avg_time_us:.2f}μs, expected < 500μs"
        
        print(f"Average metrics collection time: {avg_time_us:.2f}μs")


# ============================================================================
# Concurrent Access Performance Tests
# ============================================================================

class TestConcurrentPerformance:
    """Test circuit breaker performance under concurrent access"""

    def test_concurrent_request_checking(self, benchmark_circuit_breaker):
        """Test performance with concurrent request checking"""
        cb = benchmark_circuit_breaker
        results = []
        
        def worker_thread():
            start = time.perf_counter()
            for _ in range(1000):
                cb.should_allow_request()
            end = time.perf_counter()
            return end - start
        
        # Test with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_thread) for _ in range(10)]
            
            for future in concurrent.futures.as_completed(futures):
                duration = future.result()
                results.append(duration)
        
        # Calculate statistics
        avg_duration = statistics.mean(results)
        max_duration = max(results)
        
        # Should handle concurrent access efficiently
        assert avg_duration < 0.1, f"Average concurrent access took {avg_duration:.3f}s, expected < 0.1s"
        assert max_duration < 0.2, f"Max concurrent access took {max_duration:.3f}s, expected < 0.2s"
        
        print(f"Concurrent access - Avg: {avg_duration:.3f}s, Max: {max_duration:.3f}s")

    def test_concurrent_mixed_operations(self, benchmark_circuit_breaker):
        """Test performance with mixed concurrent operations"""
        cb = benchmark_circuit_breaker
        results = {'success': [], 'failure': [], 'check': []}
        
        def success_worker():
            start = time.perf_counter()
            for _ in range(500):
                cb.record_success(random.uniform(100, 300))
            end = time.perf_counter()
            return end - start
        
        def failure_worker():
            start = time.perf_counter()
            for _ in range(200):
                cb.record_failure(f"Concurrent error", error_type="ConcurrentError")
            end = time.perf_counter()
            return end - start
        
        def check_worker():
            start = time.perf_counter()
            for _ in range(1000):
                cb.should_allow_request()
            end = time.perf_counter()
            return end - start
        
        # Run mixed workload concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            
            # Submit different types of workers
            for _ in range(5):
                futures.append(('success', executor.submit(success_worker)))
            for _ in range(3):
                futures.append(('failure', executor.submit(failure_worker)))
            for _ in range(7):
                futures.append(('check', executor.submit(check_worker)))
            
            for operation, future in futures:
                duration = future.result()
                results[operation].append(duration)
        
        # Verify no significant performance degradation under mixed load
        for operation, durations in results.items():
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            # Performance thresholds per operation type
            if operation == 'check':
                assert avg_duration < 0.1, f"{operation} avg {avg_duration:.3f}s > 0.1s"
            elif operation == 'success':
                assert avg_duration < 0.2, f"{operation} avg {avg_duration:.3f}s > 0.2s"  
            elif operation == 'failure':
                assert avg_duration < 0.3, f"{operation} avg {avg_duration:.3f}s > 0.3s"
            
            print(f"{operation.title()} operations - Avg: {avg_duration:.3f}s, Max: {max_duration:.3f}s")

    def test_thread_safety_under_load(self, benchmark_circuit_breaker):
        """Test thread safety under high concurrent load"""
        cb = benchmark_circuit_breaker
        error_count = 0
        success_count = 0
        
        def mixed_operations_worker():
            nonlocal error_count, success_count
            local_errors = 0
            local_successes = 0
            
            for i in range(100):
                try:
                    # Mix of operations
                    if i % 3 == 0:
                        cb.record_success(random.uniform(50, 200))
                        local_successes += 1
                    elif i % 3 == 1:
                        cb.record_failure(f"Thread error {i}", error_type="ThreadError")
                        local_errors += 1
                    else:
                        cb.should_allow_request()
                        cb.get_metrics()
                except Exception as e:
                    print(f"Thread safety error: {e}")
                    raise
            
            return local_successes, local_errors
        
        # Run high concurrent load
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mixed_operations_worker) for _ in range(20)]
            
            for future in concurrent.futures.as_completed(futures):
                local_successes, local_errors = future.result()
                success_count += local_successes
                error_count += local_errors
        
        # Verify data consistency
        final_metrics = cb.get_metrics()
        
        # Should have recorded all operations without data races
        assert final_metrics['success_count'] <= success_count  # Some might be in windows
        assert final_metrics['failure_count'] <= error_count
        assert final_metrics['success_count'] >= 0
        assert final_metrics['failure_count'] >= 0
        
        print(f"Thread safety test - Successes: {success_count}, Errors: {error_count}")
        print(f"Final metrics - Successes: {final_metrics['success_count']}, Failures: {final_metrics['failure_count']}")


# ============================================================================
# Memory Usage Performance Tests
# ============================================================================

class TestMemoryPerformance:
    """Test memory usage characteristics of circuit breaker"""

    def test_memory_usage_baseline(self, memory_test_config):
        """Test baseline memory usage of circuit breaker"""
        # Measure memory before creating circuit breaker
        gc.collect()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create circuit breaker
        cb = ProductionCircuitBreaker(memory_test_config)
        
        gc.collect()
        memory_after_creation = process.memory_info().rss / 1024 / 1024
        
        # Add some data
        for i in range(1000):
            if i % 4 == 0:
                cb.record_failure(f"Memory test error {i}", error_type="MemoryTestError")
            else:
                cb.record_success(random.uniform(100, 500))
        
        gc.collect()
        memory_after_data = process.memory_info().rss / 1024 / 1024
        
        creation_overhead = memory_after_creation - memory_before
        data_overhead = memory_after_data - memory_after_creation
        
        # Circuit breaker should have minimal memory overhead
        assert creation_overhead < 5.0, f"Circuit breaker creation used {creation_overhead:.2f}MB"
        assert data_overhead < 10.0, f"1000 operations used {data_overhead:.2f}MB"
        
        print(f"Memory usage - Creation: {creation_overhead:.2f}MB, Data: {data_overhead:.2f}MB")

    def test_memory_growth_under_load(self, memory_test_config):
        """Test memory growth characteristics under sustained load"""
        cb = ProductionCircuitBreaker(memory_test_config)
        process = psutil.Process()
        
        memory_samples = []
        operation_counts = [1000, 5000, 10000, 20000, 50000]
        
        for count in operation_counts:
            # Add operations
            for i in range(count - len(cb.response_time_window)):
                if i % 5 == 0:
                    cb.record_failure(f"Load test error {i}", error_type="LoadTestError")
                else:
                    cb.record_success(random.uniform(50, 300))
            
            gc.collect()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append((count, memory_mb))
        
        # Calculate memory growth rate
        if len(memory_samples) >= 2:
            first_memory = memory_samples[0][1]
            last_memory = memory_samples[-1][1]
            total_operations = operation_counts[-1] - operation_counts[0]
            
            memory_per_1k_ops = ((last_memory - first_memory) / total_operations) * 1000
            
            # Memory growth should be bounded due to deque limits
            assert memory_per_1k_ops < 1.0, f"Memory grows by {memory_per_1k_ops:.3f}MB per 1K operations"
            
            print(f"Memory growth rate: {memory_per_1k_ops:.3f}MB per 1K operations")

    def test_memory_bounded_by_window_limits(self, memory_test_config):
        """Test that memory usage is bounded by window size limits"""
        cb = ProductionCircuitBreaker(memory_test_config)
        
        # Fill all windows to capacity
        window_size = max(len(cb.failure_rate_window.maxlen) if hasattr(cb.failure_rate_window, 'maxlen') else 100,
                         len(cb.response_time_window.maxlen) if hasattr(cb.response_time_window, 'maxlen') else 50)
        
        # Add more operations than window capacity
        for i in range(window_size * 5):  # 5x the window capacity
            cb.record_success(random.uniform(100, 400))
        
        # Windows should not exceed their limits
        assert len(cb.failure_rate_window) <= 100, f"Failure rate window: {len(cb.failure_rate_window)}"
        assert len(cb.response_time_window) <= 50, f"Response time window: {len(cb.response_time_window)}"
        
        # Error types dict might grow, but should be reasonable
        assert len(cb.error_types) < 100, f"Error types dict too large: {len(cb.error_types)}"

    def test_memory_cleanup_on_reset(self, memory_test_config):
        """Test memory cleanup when circuit breaker is reset"""
        cb = ProductionCircuitBreaker(memory_test_config)
        process = psutil.Process()
        
        # Fill with data
        for i in range(10000):
            if i % 3 == 0:
                cb.record_failure(f"Reset test error {i}", error_type=f"ErrorType{i%10}")
            else:
                cb.record_success(random.uniform(100, 500))
        
        gc.collect()
        memory_before_reset = process.memory_info().rss / 1024 / 1024
        
        # Reset circuit breaker
        cb.reset()
        
        gc.collect()
        memory_after_reset = process.memory_info().rss / 1024 / 1024
        
        # Verify data structures are cleared
        assert len(cb.failure_rate_window) == 0
        assert len(cb.response_time_window) == 0
        assert len(cb.error_types) == 0
        
        # Memory should be freed (allowing for some variance)
        memory_freed = memory_before_reset - memory_after_reset
        print(f"Memory freed on reset: {memory_freed:.2f}MB")
        
        # Note: Memory might not be immediately freed due to Python's memory management


# ============================================================================
# Load Testing Under Various Conditions
# ============================================================================

class TestLoadConditions:
    """Test circuit breaker under various load conditions"""

    def test_high_success_rate_load(self, load_test_circuit_breakers):
        """Test performance under high success rate load"""
        cbs = load_test_circuit_breakers
        start_time = time.perf_counter()
        
        # High success rate workload (95% success)
        operations_per_cb = 1000
        total_operations = len(cbs) * operations_per_cb
        
        def process_cb(cb, cb_index):
            for i in range(operations_per_cb):
                if i % 20 == 0:  # 5% failure rate
                    cb.record_failure(f"Load test failure {i}", error_type="LoadTestError")
                else:
                    cb.record_success(random.uniform(80, 200))
        
        # Process all circuit breakers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cbs)) as executor:
            futures = [executor.submit(process_cb, cb, i) for i, cb in enumerate(cbs)]
            concurrent.futures.wait(futures)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = total_operations / duration
        
        print(f"High success rate load - {total_operations} operations in {duration:.2f}s")
        print(f"Throughput: {throughput:.0f} ops/sec")
        
        # Should handle high success rate efficiently
        assert throughput > 10000, f"Throughput {throughput:.0f} ops/sec too low"
        
        # Verify circuit breakers remained closed
        closed_count = sum(1 for cb in cbs if cb.state == CircuitBreakerState.CLOSED)
        assert closed_count == len(cbs), f"Only {closed_count}/{len(cbs)} remained closed"

    def test_high_failure_rate_load(self, load_test_circuit_breakers):
        """Test performance under high failure rate load"""
        cbs = load_test_circuit_breakers
        start_time = time.perf_counter()
        
        # High failure rate workload (40% failure)
        operations_per_cb = 500  # Fewer operations due to higher processing cost
        total_operations = len(cbs) * operations_per_cb
        
        def process_cb(cb, cb_index):
            for i in range(operations_per_cb):
                if i % 5 < 2:  # 40% failure rate
                    error_types = ["TimeoutError", "ServerError", "NetworkError"]
                    error_type = error_types[i % len(error_types)]
                    cb.record_failure(f"High failure load {i}", error_type=error_type)
                else:
                    cb.record_success(random.uniform(100, 400))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cbs)) as executor:
            futures = [executor.submit(process_cb, cb, i) for i, cb in enumerate(cbs)]
            concurrent.futures.wait(futures)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = total_operations / duration
        
        print(f"High failure rate load - {total_operations} operations in {duration:.2f}s")
        print(f"Throughput: {throughput:.0f} ops/sec")
        
        # Should handle high failure rate load (lower throughput expected)
        assert throughput > 5000, f"Throughput {throughput:.0f} ops/sec too low for failure processing"
        
        # Some circuit breakers should have opened
        open_count = sum(1 for cb in cbs if cb.state == CircuitBreakerState.OPEN)
        assert open_count > 0, "No circuit breakers opened under high failure load"

    def test_bursty_load_patterns(self, load_test_circuit_breakers):
        """Test performance under bursty load patterns"""
        cbs = load_test_circuit_breakers[:3]  # Use subset for burst testing
        total_operations = 0
        
        start_time = time.perf_counter()
        
        def burst_worker(cb, burst_size, burst_delay):
            nonlocal total_operations
            operations = 0
            
            for burst in range(5):  # 5 bursts
                # Burst of operations
                for i in range(burst_size):
                    if i % 6 == 0:
                        cb.record_failure(f"Burst failure {i}", error_type="BurstError")
                    else:
                        cb.record_success(random.uniform(50, 150))
                    operations += 1
                
                # Wait between bursts
                time.sleep(burst_delay)
            
            return operations
        
        # Different burst patterns for each circuit breaker
        burst_patterns = [
            (100, 0.1),  # Small bursts, short delay
            (200, 0.2),  # Medium bursts, medium delay  
            (300, 0.3)   # Large bursts, long delay
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cbs)) as executor:
            futures = []
            for i, (cb, (burst_size, burst_delay)) in enumerate(zip(cbs, burst_patterns)):
                future = executor.submit(burst_worker, cb, burst_size, burst_delay)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                total_operations += future.result()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        print(f"Bursty load - {total_operations} operations in {duration:.2f}s")
        print(f"Average throughput: {total_operations/duration:.0f} ops/sec")
        
        # Should handle bursty patterns without significant issues
        assert duration < 10.0, f"Bursty load took too long: {duration:.2f}s"


# ============================================================================
# Recovery Time Performance Tests
# ============================================================================

class TestRecoveryPerformance:
    """Test circuit breaker recovery time performance"""

    def test_recovery_transition_time(self, stress_test_config):
        """Test time for recovery state transitions"""
        cb = ProductionCircuitBreaker(stress_test_config)
        
        # Open the circuit
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Recovery test {i}")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Measure transition to half-open
        cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        start_time = time.perf_counter()
        should_allow = cb.should_allow_request()
        transition_time = time.perf_counter() - start_time
        
        assert should_allow
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Transition should be very fast
        transition_time_us = transition_time * 1_000_000
        assert transition_time_us < 100, f"Transition took {transition_time_us:.2f}μs, expected < 100μs"
        
        print(f"Recovery transition time: {transition_time_us:.2f}μs")

    def test_half_open_to_closed_performance(self, stress_test_config):
        """Test performance of half-open to closed transition"""
        cb = ProductionCircuitBreaker(stress_test_config)
        
        # Set to half-open state
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.half_open_requests = 0
        
        # Measure time for successful half-open testing
        start_time = time.perf_counter()
        
        for i in range(cb.config.half_open_max_requests):
            cb.record_success(150.0)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Should transition to closed
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Half-open testing should be efficient
        avg_time_per_test = (duration / cb.config.half_open_max_requests) * 1000  # milliseconds
        assert avg_time_per_test < 1.0, f"Half-open test took {avg_time_per_test:.2f}ms per request"
        
        print(f"Half-open testing duration: {duration:.4f}s ({avg_time_per_test:.2f}ms per request)")

    def test_multiple_concurrent_recoveries(self):
        """Test performance when multiple circuit breakers recover simultaneously"""
        # Create multiple circuit breakers
        configs = []
        for i in range(5):
            config = BackendInstanceConfig(
                id=f"recovery_backend_{i}",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url=f"https://api.recovery{i}.com",
                api_key=f"recovery_key_{i}",
                failure_threshold=3,
                recovery_timeout_seconds=1,  # Very fast recovery
                half_open_max_requests=3
            )
            configs.append(config)
        
        cbs = [ProductionCircuitBreaker(config) for config in configs]
        
        # Open all circuit breakers
        for cb in cbs:
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Pre-recovery failure {i}")
        
        # Verify all are open
        for cb in cbs:
            assert cb.state == CircuitBreakerState.OPEN
        
        # Set recovery times to now (simulate time passing)
        for cb in cbs:
            cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Measure concurrent recovery performance
        start_time = time.perf_counter()
        
        def recovery_worker(cb):
            # Transition to half-open
            cb.should_allow_request()
            
            # Perform half-open testing
            for i in range(cb.config.half_open_max_requests):
                cb.record_success(120.0)
            
            return cb.state
        
        # Run recovery for all circuit breakers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cbs)) as executor:
            futures = [executor.submit(recovery_worker, cb) for cb in cbs]
            final_states = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # All should have recovered to closed
        assert all(state == CircuitBreakerState.CLOSED for state in final_states)
        
        # Concurrent recovery should be efficient
        assert duration < 1.0, f"Concurrent recovery took {duration:.3f}s, expected < 1.0s"
        
        print(f"Concurrent recovery of {len(cbs)} circuit breakers: {duration:.3f}s")


# ============================================================================
# Monitoring System Performance Impact
# ============================================================================

class TestMonitoringPerformance:
    """Test performance impact of monitoring systems"""

    def test_metrics_collection_overhead(self, benchmark_circuit_breaker):
        """Test performance overhead of metrics collection"""
        cb = benchmark_circuit_breaker
        
        # Generate baseline data
        for i in range(1000):
            if i % 4 == 0:
                cb.record_failure(f"Monitoring test {i}", error_type="MonitoringTestError")
            else:
                cb.record_success(random.uniform(100, 300))
        
        # Measure operations without metrics collection
        start_time = time.perf_counter()
        iterations = 1000
        
        for i in range(iterations):
            cb.record_success(200.0)
        
        baseline_time = time.perf_counter() - start_time
        
        # Measure operations with frequent metrics collection
        start_time = time.perf_counter()
        
        for i in range(iterations):
            cb.record_success(200.0)
            if i % 10 == 0:  # Collect metrics every 10 operations
                cb.get_metrics()
        
        with_metrics_time = time.perf_counter() - start_time
        
        # Calculate overhead
        overhead_percent = ((with_metrics_time - baseline_time) / baseline_time) * 100
        
        # Metrics collection overhead should be minimal
        assert overhead_percent < 50, f"Metrics collection adds {overhead_percent:.1f}% overhead"
        
        print(f"Metrics collection overhead: {overhead_percent:.1f}%")

    def test_large_metrics_payload_performance(self, benchmark_circuit_breaker):
        """Test performance with large metrics payloads"""
        cb = benchmark_circuit_breaker
        
        # Generate large amount of diverse data
        error_types = [f"ErrorType{i}" for i in range(50)]
        
        for i in range(10000):
            if i % 5 == 0:
                error_type = error_types[i % len(error_types)]
                cb.record_failure(f"Large payload error {i}", error_type=error_type)
            else:
                cb.record_success(random.uniform(50, 500))
        
        # Measure metrics collection time
        start_time = time.perf_counter()
        
        for _ in range(100):
            metrics = cb.get_metrics()
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Large metrics should still be collected efficiently
        assert avg_time_ms < 10, f"Large metrics collection took {avg_time_ms:.2f}ms, expected < 10ms"
        
        # Verify metrics completeness
        assert len(metrics['error_types']) == len(error_types)
        
        print(f"Large metrics collection time: {avg_time_ms:.2f}ms")

    def test_continuous_monitoring_impact(self, load_test_circuit_breakers):
        """Test impact of continuous monitoring on circuit breaker performance"""
        cbs = load_test_circuit_breakers[:3]  # Use subset
        monitoring_active = True
        metrics_collections = 0
        
        def monitoring_worker():
            nonlocal metrics_collections
            while monitoring_active:
                for cb in cbs:
                    cb.get_metrics()
                    metrics_collections += 1
                time.sleep(0.1)  # 10Hz monitoring
        
        def workload_worker(cb, operations):
            for i in range(operations):
                if i % 7 == 0:
                    cb.record_failure(f"Continuous monitor test {i}", error_type="ContinuousTestError")
                else:
                    cb.record_success(random.uniform(100, 250))
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitoring_worker)
        monitor_thread.start()
        
        # Run workload
        start_time = time.perf_counter()
        operations_per_cb = 2000
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cbs)) as executor:
            futures = [executor.submit(workload_worker, cb, operations_per_cb) for cb in cbs]
            concurrent.futures.wait(futures)
        
        end_time = time.perf_counter()
        
        # Stop monitoring
        monitoring_active = False
        monitor_thread.join()
        
        duration = end_time - start_time
        total_operations = len(cbs) * operations_per_cb
        throughput = total_operations / duration
        
        print(f"Workload with continuous monitoring:")
        print(f"  Operations: {total_operations} in {duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} ops/sec")
        print(f"  Metrics collections: {metrics_collections}")
        
        # Should maintain reasonable performance under monitoring
        assert throughput > 5000, f"Throughput {throughput:.0f} too low under monitoring"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements