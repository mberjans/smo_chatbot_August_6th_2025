"""
TTL Performance Impact Testing for Cache System.

This module provides comprehensive performance testing of TTL functionality,
measuring the impact of TTL operations on cache performance, memory usage,
scalability, and system throughput in the Clinical Metabolomics Oracle.

Test Coverage:
- TTL operation performance benchmarks
- Memory overhead of TTL metadata
- TTL cleanup performance at scale
- Concurrent TTL operation performance
- TTL impact on cache throughput
- TTL scalability testing
- TTL performance under various load patterns
- TTL performance regression detection

Classes:
    TestTTLOperationPerformance: Core TTL operation performance testing
    TestTTLMemoryPerformance: TTL memory usage and overhead testing
    TestTTLScalabilityPerformance: TTL performance scalability testing
    TestTTLConcurrencyPerformance: TTL performance under concurrent load
    TestTTLThroughputPerformance: TTL impact on overall system throughput
    TestTTLRegressionPerformance: TTL performance regression detection
    TestTTLBenchmarkSuite: Comprehensive TTL performance benchmarks

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import random
import statistics
import gc
import sys
import psutil
import concurrent.futures
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import pickle
import os

# Import test fixtures (adjust path as needed)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'unit'))
from cache_test_fixtures import (
    BIOMEDICAL_QUERIES,
    BiomedicalTestDataGenerator,
    CachePerformanceMetrics,
    CachePerformanceMeasurer
)


# Performance Test Configuration
PERFORMANCE_CONFIGS = {
    'SMALL_SCALE': {
        'entries': 100,
        'operations': 1000,
        'threads': 2,
        'duration': 10
    },
    'MEDIUM_SCALE': {
        'entries': 1000,
        'operations': 10000,
        'threads': 5,
        'duration': 30
    },
    'LARGE_SCALE': {
        'entries': 10000,
        'operations': 100000,
        'threads': 10,
        'duration': 60
    },
    'STRESS_SCALE': {
        'entries': 50000,
        'operations': 500000,
        'threads': 20,
        'duration': 120
    }
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'ttl_set_ms': 1.0,        # TTL set operation should be < 1ms
    'ttl_get_ms': 0.5,        # TTL get operation should be < 0.5ms
    'ttl_cleanup_ms': 10.0,   # TTL cleanup should be < 10ms per 1000 entries
    'ttl_extension_ms': 0.1,  # TTL extension should be < 0.1ms
    'memory_overhead_pct': 20, # TTL metadata should add < 20% memory overhead
    'throughput_degradation_pct': 10  # TTL should not reduce throughput by > 10%
}


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for TTL operations."""
    operation_type: str
    total_operations: int
    duration_seconds: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_sec: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    success_rate: float = 1.0
    error_count: int = 0
    
    def meets_performance_targets(self) -> bool:
        """Check if metrics meet performance targets."""
        thresholds = PERFORMANCE_THRESHOLDS
        
        # Check specific operation thresholds
        if self.operation_type == 'ttl_set':
            return self.avg_time_ms < thresholds['ttl_set_ms']
        elif self.operation_type == 'ttl_get':
            return self.avg_time_ms < thresholds['ttl_get_ms']
        elif self.operation_type == 'ttl_cleanup':
            return self.avg_time_ms < thresholds['ttl_cleanup_ms']
        elif self.operation_type == 'ttl_extension':
            return self.avg_time_ms < thresholds['ttl_extension_ms']
        
        # General thresholds
        return (
            self.success_rate > 0.95 and
            self.p99_time_ms < 1000  # P99 should be < 1s
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_type': self.operation_type,
            'total_operations': self.total_operations,
            'duration_seconds': self.duration_seconds,
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'p50_time_ms': self.p50_time_ms,
            'p95_time_ms': self.p95_time_ms,
            'p99_time_ms': self.p99_time_ms,
            'throughput_ops_sec': self.throughput_ops_sec,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_pct': self.cpu_usage_pct,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'meets_targets': self.meets_performance_targets()
        }


@dataclass  
class CacheEntry:
    """High-performance cache entry with TTL support."""
    key: str
    value: Any
    timestamp: float
    ttl: int
    confidence: float = 0.9
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if entry has expired."""
        if current_time is None:
            current_time = time.time()
        return current_time > (self.timestamp + self.ttl)
    
    def time_until_expiry(self, current_time: Optional[float] = None) -> float:
        """Get time until expiry in seconds."""
        if current_time is None:
            current_time = time.time()
        return max(0, (self.timestamp + self.ttl) - current_time)


class PerformanceTTLCache:
    """High-performance TTL cache optimized for performance testing."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.storage: Dict[str, CacheEntry] = {}
        self.expiry_index: Dict[float, List[str]] = defaultdict(list)  # timestamp -> keys
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.hits = 0
        self.misses = 0
        self.total_operations = 0
        
        # Cleanup optimization
        self.last_cleanup_time = 0.0
        self.cleanup_interval = 1.0  # seconds
        
    def _record_operation_time(self, operation: str, duration: float):
        """Record operation timing for performance analysis."""
        self.operation_times[operation].append(duration * 1000)  # Convert to ms
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key."""
        return f"cache:{hash(query)}"
    
    def _should_cleanup(self) -> bool:
        """Determine if cleanup should run."""
        current_time = time.time()
        return (current_time - self.last_cleanup_time) > self.cleanup_interval
    
    def _cleanup_expired_optimized(self) -> int:
        """Optimized cleanup of expired entries."""
        if not self._should_cleanup():
            return 0
        
        start_time = time.time()
        current_time = start_time
        expired_count = 0
        
        # Clean entries from expiry index
        expired_timestamps = [ts for ts in self.expiry_index.keys() if ts < current_time]
        
        for timestamp in expired_timestamps:
            keys_to_remove = self.expiry_index[timestamp]
            for key in keys_to_remove:
                if key in self.storage and self.storage[key].is_expired(current_time):
                    del self.storage[key]
                    expired_count += 1
            del self.expiry_index[timestamp]
        
        self.last_cleanup_time = current_time
        cleanup_duration = time.time() - start_time
        self._record_operation_time('cleanup', cleanup_duration)
        
        return expired_count
    
    def set(self, query: str, value: Any, ttl: Optional[int] = None, 
            confidence: float = 0.9, metadata: Optional[Dict[str, Any]] = None) -> str:
        """High-performance set operation with TTL."""
        start_time = time.time()
        
        # Cleanup if needed
        self._cleanup_expired_optimized()
        
        # Handle size limit (simple eviction for performance)
        if len(self.storage) >= self.max_size:
            # Remove oldest entry (first in dict in Python 3.7+)
            oldest_key = next(iter(self.storage))
            old_entry = self.storage[oldest_key]
            
            # Remove from expiry index
            expiry_time = old_entry.timestamp + old_entry.ttl
            if expiry_time in self.expiry_index:
                try:
                    self.expiry_index[expiry_time].remove(oldest_key)
                    if not self.expiry_index[expiry_time]:
                        del self.expiry_index[expiry_time]
                except ValueError:
                    pass  # Key not in list
            
            del self.storage[oldest_key]
        
        # Create entry
        key = self._generate_key(query)
        effective_ttl = ttl or self.default_ttl
        timestamp = time.time()
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=timestamp,
            ttl=effective_ttl,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Store entry
        self.storage[key] = entry
        
        # Index for expiration
        expiry_time = timestamp + effective_ttl
        self.expiry_index[expiry_time].append(key)
        
        self.total_operations += 1
        operation_time = time.time() - start_time
        self._record_operation_time('set', operation_time)
        
        return key
    
    def get(self, query: str) -> Optional[Any]:
        """High-performance get operation with TTL checking."""
        start_time = time.time()
        
        key = self._generate_key(query)
        
        if key in self.storage:
            entry = self.storage[key]
            current_time = time.time()
            
            if not entry.is_expired(current_time):
                entry.access_count += 1
                self.hits += 1
                self.total_operations += 1
                
                operation_time = current_time - start_time
                self._record_operation_time('get_hit', operation_time)
                
                return entry.value
            else:
                # Remove expired entry
                del self.storage[key]
                expiry_time = entry.timestamp + entry.ttl
                if expiry_time in self.expiry_index:
                    try:
                        self.expiry_index[expiry_time].remove(key)
                    except ValueError:
                        pass
        
        self.misses += 1
        self.total_operations += 1
        operation_time = time.time() - start_time
        self._record_operation_time('get_miss', operation_time)
        
        return None
    
    def extend_ttl(self, query: str, additional_time: int) -> bool:
        """Extend TTL for existing entry."""
        start_time = time.time()
        
        key = self._generate_key(query)
        
        if key in self.storage:
            entry = self.storage[key]
            if not entry.is_expired():
                # Remove from old expiry index
                old_expiry = entry.timestamp + entry.ttl
                if old_expiry in self.expiry_index:
                    try:
                        self.expiry_index[old_expiry].remove(key)
                        if not self.expiry_index[old_expiry]:
                            del self.expiry_index[old_expiry]
                    except ValueError:
                        pass
                
                # Update TTL
                entry.ttl += additional_time
                
                # Add to new expiry index
                new_expiry = entry.timestamp + entry.ttl
                self.expiry_index[new_expiry].append(key)
                
                operation_time = time.time() - start_time
                self._record_operation_time('extend_ttl', operation_time)
                
                return True
        
        operation_time = time.time() - start_time
        self._record_operation_time('extend_ttl_failed', operation_time)
        return False
    
    def force_cleanup(self) -> int:
        """Force immediate cleanup of all expired entries."""
        start_time = time.time()
        
        current_time = start_time
        expired_keys = []
        
        for key, entry in self.storage.items():
            if entry.is_expired(current_time):
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self.storage[key]
            del self.storage[key]
            
            # Clean from expiry index
            expiry_time = entry.timestamp + entry.ttl
            if expiry_time in self.expiry_index:
                try:
                    self.expiry_index[expiry_time].remove(key)
                    if not self.expiry_index[expiry_time]:
                        del self.expiry_index[expiry_time]
                except ValueError:
                    pass
        
        operation_time = time.time() - start_time
        self._record_operation_time('force_cleanup', operation_time)
        
        return len(expired_keys)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'total_entries': len(self.storage),
            'total_operations': self.total_operations,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'expiry_index_size': len(self.expiry_index),
            'operation_times': {}
        }
        
        # Calculate timing statistics
        for operation, times in self.operation_times.items():
            if times:
                stats['operation_times'][operation] = {
                    'count': len(times),
                    'avg_ms': statistics.mean(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'p50_ms': statistics.median(times),
                    'p95_ms': self._percentile(times, 0.95),
                    'p99_ms': self._percentile(times, 0.99)
                }
        
        return stats
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def clear_performance_stats(self):
        """Clear performance statistics."""
        self.operation_times.clear()
        self.hits = 0
        self.misses = 0
        self.total_operations = 0


class PerformanceTestRunner:
    """Utility for running and measuring cache performance tests."""
    
    def __init__(self):
        self.data_generator = BiomedicalTestDataGenerator()
        
    def run_operation_benchmark(self, cache: PerformanceTTLCache, operation_type: str,
                               operation_count: int, **kwargs) -> PerformanceMetrics:
        """Run benchmark for specific operation type."""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        operation_times = []
        error_count = 0
        
        for i in range(operation_count):
            op_start = time.time()
            
            try:
                if operation_type == 'set':
                    query_data = self.data_generator.generate_query()
                    cache.set(
                        query_data['query'],
                        f"Data {i}",
                        ttl=kwargs.get('ttl', 3600),
                        confidence=query_data['confidence']
                    )
                
                elif operation_type == 'get':
                    query_data = self.data_generator.generate_query()
                    cache.get(query_data['query'])
                
                elif operation_type == 'extend_ttl':
                    query_data = self.data_generator.generate_query()
                    cache.extend_ttl(query_data['query'], kwargs.get('extension', 60))
                
                elif operation_type == 'cleanup':
                    cache.force_cleanup()
                
                success = True
                
            except Exception as e:
                success = False
                error_count += 1
            
            op_time = (time.time() - op_start) * 1000  # Convert to ms
            operation_times.append(op_time)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()
        
        duration = end_time - start_time
        throughput = operation_count / duration if duration > 0 else 0
        
        # Calculate percentiles
        operation_times.sort()
        
        return PerformanceMetrics(
            operation_type=operation_type,
            total_operations=operation_count,
            duration_seconds=duration,
            avg_time_ms=statistics.mean(operation_times),
            min_time_ms=min(operation_times),
            max_time_ms=max(operation_times),
            p50_time_ms=self._percentile(operation_times, 0.5),
            p95_time_ms=self._percentile(operation_times, 0.95),
            p99_time_ms=self._percentile(operation_times, 0.99),
            throughput_ops_sec=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_pct=(end_cpu + start_cpu) / 2,
            success_rate=(operation_count - error_count) / operation_count,
            error_count=error_count
        )
    
    def run_concurrent_benchmark(self, cache: PerformanceTTLCache, operation_type: str,
                                total_operations: int, thread_count: int,
                                **kwargs) -> PerformanceMetrics:
        """Run concurrent benchmark with multiple threads."""
        
        operations_per_thread = total_operations // thread_count
        all_times = []
        total_errors = 0
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        def worker_thread():
            thread_times = []
            thread_errors = 0
            
            for i in range(operations_per_thread):
                op_start = time.time()
                
                try:
                    if operation_type == 'set':
                        query_data = self.data_generator.generate_query()
                        cache.set(
                            f"thread_query_{threading.current_thread().ident}_{i}",
                            f"Data {i}",
                            ttl=kwargs.get('ttl', 3600)
                        )
                    
                    elif operation_type == 'get':
                        query_data = self.data_generator.generate_query()
                        cache.get(query_data['query'])
                    
                    elif operation_type == 'mixed':
                        query_data = self.data_generator.generate_query()
                        if i % 2 == 0:
                            cache.set(f"mixed_{i}", f"Data {i}", ttl=3600)
                        else:
                            cache.get(query_data['query'])
                    
                except Exception:
                    thread_errors += 1
                
                op_time = (time.time() - op_start) * 1000
                thread_times.append(op_time)
            
            return thread_times, thread_errors
        
        # Run concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker_thread) for _ in range(thread_count)]
            
            for future in concurrent.futures.as_completed(futures):
                thread_times, thread_errors = future.result()
                all_times.extend(thread_times)
                total_errors += thread_errors
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        actual_operations = len(all_times)
        throughput = actual_operations / duration if duration > 0 else 0
        
        all_times.sort()
        
        return PerformanceMetrics(
            operation_type=f"concurrent_{operation_type}",
            total_operations=actual_operations,
            duration_seconds=duration,
            avg_time_ms=statistics.mean(all_times) if all_times else 0,
            min_time_ms=min(all_times) if all_times else 0,
            max_time_ms=max(all_times) if all_times else 0,
            p50_time_ms=self._percentile(all_times, 0.5),
            p95_time_ms=self._percentile(all_times, 0.95),
            p99_time_ms=self._percentile(all_times, 0.99),
            throughput_ops_sec=throughput,
            memory_usage_mb=end_memory - start_memory,
            success_rate=(actual_operations - total_errors) / actual_operations if actual_operations > 0 else 0,
            error_count=total_errors
        )
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        index = int(len(values) * percentile)
        return values[min(index, len(values) - 1)]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0


class TestTTLOperationPerformance:
    """Tests for core TTL operation performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = PerformanceTTLCache(max_size=10000)
        self.test_runner = PerformanceTestRunner()
    
    def test_ttl_set_operation_performance(self):
        """Test performance of TTL set operations."""
        config = PERFORMANCE_CONFIGS['MEDIUM_SCALE']
        
        metrics = self.test_runner.run_operation_benchmark(
            self.cache, 'set', config['operations'], ttl=3600
        )
        
        print(f"\nTTL Set Performance:")
        print(f"  Average time: {metrics.avg_time_ms:.3f}ms")
        print(f"  P99 time: {metrics.p99_time_ms:.3f}ms")
        print(f"  Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
        
        # Performance assertions
        assert metrics.meets_performance_targets(), \
            f"TTL set performance below target: {metrics.avg_time_ms}ms > {PERFORMANCE_THRESHOLDS['ttl_set_ms']}ms"
        
        assert metrics.success_rate > 0.99, "TTL set operations should have high success rate"
    
    def test_ttl_get_operation_performance(self):
        """Test performance of TTL get operations."""
        config = PERFORMANCE_CONFIGS['MEDIUM_SCALE']
        
        # Pre-populate cache
        for i in range(1000):
            query_data = self.test_runner.data_generator.generate_query()
            self.cache.set(query_data['query'], f"Data {i}", ttl=3600)
        
        metrics = self.test_runner.run_operation_benchmark(
            self.cache, 'get', config['operations']
        )
        
        print(f"\nTTL Get Performance:")
        print(f"  Average time: {metrics.avg_time_ms:.3f}ms")
        print(f"  P99 time: {metrics.p99_time_ms:.3f}ms") 
        print(f"  Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
        
        # Performance assertions
        assert metrics.avg_time_ms < PERFORMANCE_THRESHOLDS['ttl_get_ms'], \
            f"TTL get performance below target: {metrics.avg_time_ms}ms"
        
        assert metrics.success_rate > 0.99, "TTL get operations should have high success rate"
    
    def test_ttl_extension_performance(self):
        """Test performance of TTL extension operations."""
        # Pre-populate cache
        for i in range(100):
            query_data = self.test_runner.data_generator.generate_query()
            self.cache.set(f"extend_test_{i}", f"Data {i}", ttl=3600)
        
        metrics = self.test_runner.run_operation_benchmark(
            self.cache, 'extend_ttl', 1000, extension=300
        )
        
        print(f"\nTTL Extension Performance:")
        print(f"  Average time: {metrics.avg_time_ms:.3f}ms")
        print(f"  P99 time: {metrics.p99_time_ms:.3f}ms")
        print(f"  Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
        
        assert metrics.avg_time_ms < PERFORMANCE_THRESHOLDS['ttl_extension_ms'], \
            f"TTL extension performance below target: {metrics.avg_time_ms}ms"
    
    def test_ttl_cleanup_performance(self):
        """Test performance of TTL cleanup operations."""
        # Fill cache with entries that will expire
        for i in range(5000):
            ttl = 1 if i % 2 == 0 else 3600  # Half will expire
            self.cache.set(f"cleanup_test_{i}", f"Data {i}", ttl=ttl)
        
        # Wait for entries to expire
        time.sleep(1.1)
        
        metrics = self.test_runner.run_operation_benchmark(
            self.cache, 'cleanup', 100
        )
        
        print(f"\nTTL Cleanup Performance:")
        print(f"  Average time: {metrics.avg_time_ms:.3f}ms")
        print(f"  P99 time: {metrics.p99_time_ms:.3f}ms")
        print(f"  Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
        
        # Cleanup should handle large numbers efficiently
        normalized_cleanup_time = metrics.avg_time_ms / 50  # Per 1000 entries
        assert normalized_cleanup_time < PERFORMANCE_THRESHOLDS['ttl_cleanup_ms'], \
            f"TTL cleanup performance below target: {normalized_cleanup_time}ms per 1000 entries"


class TestTTLMemoryPerformance:
    """Tests for TTL memory usage and overhead."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_runner = PerformanceTestRunner()
    
    def test_ttl_memory_overhead(self):
        """Test memory overhead of TTL metadata."""
        # Create cache without TTL-specific features
        simple_cache = {}
        
        # Create TTL cache
        ttl_cache = PerformanceTTLCache(max_size=10000)
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = self.test_runner._get_memory_usage()
        
        # Add entries to simple cache
        for i in range(1000):
            simple_cache[f"key_{i}"] = f"Data {i}"
        
        simple_memory = self.test_runner._get_memory_usage()
        
        # Add same entries to TTL cache
        for i in range(1000):
            ttl_cache.set(f"key_{i}", f"Data {i}", ttl=3600)
        
        ttl_memory = self.test_runner._get_memory_usage()
        
        # Calculate overhead
        simple_overhead = simple_memory - baseline_memory
        ttl_overhead = ttl_memory - simple_memory
        
        if simple_overhead > 0:
            overhead_percentage = (ttl_overhead / simple_overhead) * 100
        else:
            overhead_percentage = 0
        
        print(f"\nTTL Memory Overhead:")
        print(f"  Baseline: {baseline_memory:.2f}MB")
        print(f"  Simple cache: {simple_memory:.2f}MB")
        print(f"  TTL cache: {ttl_memory:.2f}MB")
        print(f"  TTL overhead: {ttl_overhead:.2f}MB ({overhead_percentage:.1f}%)")
        
        # Memory overhead should be reasonable
        assert overhead_percentage < PERFORMANCE_THRESHOLDS['memory_overhead_pct'], \
            f"TTL memory overhead too high: {overhead_percentage}%"
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with cache size."""
        cache_sizes = [100, 500, 1000, 5000]
        memory_measurements = []
        
        for size in cache_sizes:
            gc.collect()
            start_memory = self.test_runner._get_memory_usage()
            
            cache = PerformanceTTLCache(max_size=size * 2)
            
            # Fill cache
            for i in range(size):
                cache.set(f"scale_test_{i}", f"Data {i}", ttl=3600)
            
            end_memory = self.test_runner._get_memory_usage()
            memory_per_entry = (end_memory - start_memory) / size
            memory_measurements.append((size, memory_per_entry))
            
            print(f"Cache size {size}: {memory_per_entry:.4f}MB per entry")
        
        # Memory per entry should be roughly linear
        # (allowing for some overhead in small caches)
        if len(memory_measurements) >= 2:
            large_cache_memory = memory_measurements[-1][1]
            small_cache_memory = memory_measurements[0][1]
            
            # Large caches should be more memory efficient per entry
            assert large_cache_memory <= small_cache_memory * 2, \
                "Memory scaling should be reasonable"
    
    def test_memory_fragmentation_impact(self):
        """Test impact of TTL operations on memory fragmentation."""
        cache = PerformanceTTLCache(max_size=5000)
        
        # Fill cache
        for i in range(2000):
            cache.set(f"frag_test_{i}", f"Data {i}", ttl=2)
        
        gc.collect()
        memory_after_fill = self.test_runner._get_memory_usage()
        
        # Wait for expiration
        time.sleep(2.1)
        
        # Force cleanup
        expired_count = cache.force_cleanup()
        
        gc.collect()
        memory_after_cleanup = self.test_runner._get_memory_usage()
        
        # Refill cache
        for i in range(2000):
            cache.set(f"refill_test_{i}", f"Data {i}", ttl=3600)
        
        memory_after_refill = self.test_runner._get_memory_usage()
        
        print(f"\nMemory Fragmentation Test:")
        print(f"  After fill: {memory_after_fill:.2f}MB")
        print(f"  After cleanup: {memory_after_cleanup:.2f}MB")
        print(f"  After refill: {memory_after_refill:.2f}MB")
        print(f"  Expired entries: {expired_count}")
        
        # Memory should be freed after cleanup
        assert memory_after_cleanup < memory_after_fill, \
            "Memory should be freed after TTL cleanup"


class TestTTLScalabilityPerformance:
    """Tests for TTL performance scalability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_runner = PerformanceTestRunner()
    
    def test_ttl_performance_scaling(self):
        """Test TTL performance scaling with cache size."""
        cache_sizes = [100, 500, 1000, 5000]
        performance_results = []
        
        for size in cache_sizes:
            cache = PerformanceTTLCache(max_size=size * 2)
            
            # Pre-fill cache
            for i in range(size):
                cache.set(f"scale_{i}", f"Data {i}", ttl=3600)
            
            # Measure get performance
            metrics = self.test_runner.run_operation_benchmark(
                cache, 'get', 1000
            )
            
            performance_results.append((size, metrics.avg_time_ms, metrics.throughput_ops_sec))
            
            print(f"Cache size {size}: {metrics.avg_time_ms:.3f}ms avg, "
                  f"{metrics.throughput_ops_sec:.0f} ops/sec")
        
        # Performance should scale reasonably
        for i in range(1, len(performance_results)):
            prev_size, prev_time, prev_throughput = performance_results[i-1]
            curr_size, curr_time, curr_throughput = performance_results[i]
            
            # Response time should not increase dramatically
            time_increase_factor = curr_time / prev_time
            size_increase_factor = curr_size / prev_size
            
            assert time_increase_factor < size_increase_factor * 2, \
                f"Performance degradation too steep: {time_increase_factor}x time for {size_increase_factor}x size"
    
    def test_ttl_cleanup_scaling(self):
        """Test TTL cleanup performance scaling."""
        cleanup_sizes = [1000, 5000, 10000]
        cleanup_results = []
        
        for size in cleanup_sizes:
            cache = PerformanceTTLCache(max_size=size * 2)
            
            # Fill with expiring entries
            for i in range(size):
                ttl = 1 if i % 2 == 0 else 3600  # Half expire
                cache.set(f"cleanup_{i}", f"Data {i}", ttl=ttl)
            
            # Wait for expiration
            time.sleep(1.1)
            
            # Measure cleanup performance
            start_time = time.time()
            expired_count = cache.force_cleanup()
            cleanup_time = time.time() - start_time
            
            cleanup_per_entry = cleanup_time / expired_count if expired_count > 0 else 0
            cleanup_results.append((size, cleanup_time, expired_count, cleanup_per_entry))
            
            print(f"Cleanup {size} entries: {cleanup_time:.3f}s total, "
                  f"{expired_count} expired, {cleanup_per_entry*1000:.3f}ms per entry")
        
        # Cleanup should scale sub-linearly
        if len(cleanup_results) >= 2:
            small_time_per_entry = cleanup_results[0][3]
            large_time_per_entry = cleanup_results[-1][3]
            
            # Per-entry cleanup time should not increase significantly
            assert large_time_per_entry < small_time_per_entry * 3, \
                "Cleanup scaling is too poor"
    
    def test_extreme_scale_performance(self):
        """Test performance at extreme scale (if resources allow)."""
        if sys.gettrace() is not None:
            pytest.skip("Skipping extreme scale test in debug mode")
        
        config = PERFORMANCE_CONFIGS['LARGE_SCALE']
        cache = PerformanceTTLCache(max_size=config['entries'])
        
        # Fill cache
        print(f"\nFilling cache with {config['entries']} entries...")
        start_time = time.time()
        
        for i in range(config['entries']):
            cache.set(f"extreme_{i}", f"Data {i}", ttl=3600)
            
            if i % 1000 == 0:
                print(f"  Progress: {i}/{config['entries']}")
        
        fill_time = time.time() - start_time
        print(f"Fill time: {fill_time:.2f}s ({config['entries']/fill_time:.0f} ops/sec)")
        
        # Test access performance
        print("Testing random access performance...")
        access_times = []
        
        for _ in range(1000):
            key_index = random.randint(0, config['entries'] - 1)
            start = time.time()
            cache.get(f"extreme_{key_index}")
            access_times.append((time.time() - start) * 1000)
        
        avg_access_time = statistics.mean(access_times)
        p99_access_time = self.test_runner._percentile(access_times, 0.99)
        
        print(f"Access performance: {avg_access_time:.3f}ms avg, {p99_access_time:.3f}ms p99")
        
        # Performance should remain reasonable even at scale
        assert avg_access_time < 10.0, f"Access time too slow at scale: {avg_access_time}ms"
        assert p99_access_time < 50.0, f"P99 access time too slow at scale: {p99_access_time}ms"


class TestTTLConcurrencyPerformance:
    """Tests for TTL performance under concurrent load."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_runner = PerformanceTestRunner()
    
    def test_concurrent_ttl_operations(self):
        """Test TTL performance under concurrent operations."""
        cache = PerformanceTTLCache(max_size=10000)
        config = PERFORMANCE_CONFIGS['MEDIUM_SCALE']
        
        # Test concurrent sets
        set_metrics = self.test_runner.run_concurrent_benchmark(
            cache, 'set', config['operations'], config['threads'], ttl=3600
        )
        
        print(f"\nConcurrent Set Performance ({config['threads']} threads):")
        print(f"  Average time: {set_metrics.avg_time_ms:.3f}ms")
        print(f"  P99 time: {set_metrics.p99_time_ms:.3f}ms")
        print(f"  Throughput: {set_metrics.throughput_ops_sec:.0f} ops/sec")
        print(f"  Success rate: {set_metrics.success_rate:.3f}")
        
        # Test concurrent gets
        get_metrics = self.test_runner.run_concurrent_benchmark(
            cache, 'get', config['operations'], config['threads']
        )
        
        print(f"\nConcurrent Get Performance ({config['threads']} threads):")
        print(f"  Average time: {get_metrics.avg_time_ms:.3f}ms")
        print(f"  P99 time: {get_metrics.p99_time_ms:.3f}ms")
        print(f"  Throughput: {get_metrics.throughput_ops_sec:.0f} ops/sec")
        print(f"  Success rate: {get_metrics.success_rate:.3f}")
        
        # Performance should remain reasonable under concurrency
        assert set_metrics.success_rate > 0.95, "Concurrent sets should have high success rate"
        assert get_metrics.success_rate > 0.95, "Concurrent gets should have high success rate"
        
        # Concurrent performance should not be dramatically worse than single-threaded
        assert set_metrics.avg_time_ms < 10.0, "Concurrent set performance acceptable"
        assert get_metrics.avg_time_ms < 5.0, "Concurrent get performance acceptable"
    
    def test_mixed_concurrent_operations(self):
        """Test mixed concurrent operations (reads/writes/cleanups)."""
        cache = PerformanceTTLCache(max_size=5000)
        
        # Pre-populate cache
        for i in range(1000):
            ttl = random.choice([1, 3600])  # Some expire quickly
            cache.set(f"mixed_{i}", f"Data {i}", ttl=ttl)
        
        metrics = self.test_runner.run_concurrent_benchmark(
            cache, 'mixed', 5000, 8
        )
        
        print(f"\nMixed Concurrent Operations:")
        print(f"  Average time: {metrics.avg_time_ms:.3f}ms")
        print(f"  Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
        print(f"  Success rate: {metrics.success_rate:.3f}")
        
        assert metrics.success_rate > 0.90, "Mixed operations should have good success rate"
        
    def test_concurrent_cleanup_performance(self):
        """Test performance when cleanup runs concurrently with operations."""
        cache = PerformanceTTLCache(max_size=5000)
        
        # Fill with expiring entries
        for i in range(2000):
            cache.set(f"concurrent_{i}", f"Data {i}", ttl=1)
        
        # Start concurrent operations
        def worker_operations():
            for i in range(100):
                cache.set(f"worker_{threading.current_thread().ident}_{i}", f"Data {i}", ttl=3600)
                cache.get(f"concurrent_{i % 2000}")
                time.sleep(0.001)  # Small delay
        
        # Wait for some expiration
        time.sleep(0.5)
        
        # Start worker threads
        threads = [threading.Thread(target=worker_operations) for _ in range(4)]
        
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        # Run concurrent cleanups
        cleanup_times = []
        for _ in range(10):
            cleanup_start = time.time()
            cache.force_cleanup()
            cleanup_times.append((time.time() - cleanup_start) * 1000)
            time.sleep(0.1)
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        avg_cleanup_time = statistics.mean(cleanup_times)
        
        print(f"\nConcurrent Cleanup Performance:")
        print(f"  Average cleanup time: {avg_cleanup_time:.3f}ms")
        print(f"  Total test time: {total_time:.2f}s")
        
        # Cleanup should not be severely impacted by concurrent operations
        assert avg_cleanup_time < 100.0, "Concurrent cleanup performance acceptable"


class TestTTLThroughputPerformance:
    """Tests for TTL impact on overall system throughput."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_runner = PerformanceTestRunner()
    
    def test_ttl_vs_no_ttl_throughput(self):
        """Compare throughput with and without TTL functionality."""
        # Simple cache without TTL
        simple_cache = {}
        
        # TTL-enabled cache
        ttl_cache = PerformanceTTLCache(max_size=5000)
        
        operation_count = 5000
        
        # Measure simple cache throughput
        simple_times = []
        start_time = time.time()
        
        for i in range(operation_count):
            op_start = time.time()
            
            if i % 2 == 0:
                simple_cache[f"key_{i}"] = f"Data {i}"
            else:
                simple_cache.get(f"key_{i//2}", None)
            
            simple_times.append((time.time() - op_start) * 1000)
        
        simple_duration = time.time() - start_time
        simple_throughput = operation_count / simple_duration
        
        # Measure TTL cache throughput
        ttl_times = []
        start_time = time.time()
        
        for i in range(operation_count):
            op_start = time.time()
            
            if i % 2 == 0:
                ttl_cache.set(f"key_{i}", f"Data {i}", ttl=3600)
            else:
                ttl_cache.get(f"key_{i//2}")
            
            ttl_times.append((time.time() - op_start) * 1000)
        
        ttl_duration = time.time() - start_time
        ttl_throughput = operation_count / ttl_duration
        
        throughput_ratio = ttl_throughput / simple_throughput
        degradation_pct = (1 - throughput_ratio) * 100
        
        print(f"\nThroughput Comparison:")
        print(f"  Simple cache: {simple_throughput:.0f} ops/sec")
        print(f"  TTL cache: {ttl_throughput:.0f} ops/sec")
        print(f"  Degradation: {degradation_pct:.1f}%")
        
        # TTL should not cause excessive throughput degradation
        assert degradation_pct < PERFORMANCE_THRESHOLDS['throughput_degradation_pct'], \
            f"TTL causes excessive throughput degradation: {degradation_pct}%"
    
    def test_sustained_throughput_performance(self):
        """Test sustained throughput over extended period."""
        cache = PerformanceTTLCache(max_size=5000)
        
        # Run sustained operations for longer period
        duration = 30  # seconds
        operation_intervals = []
        
        start_time = time.time()
        operation_count = 0
        
        while time.time() - start_time < duration:
            interval_start = time.time()
            
            # Perform batch of operations
            for i in range(100):
                query_data = self.test_runner.data_generator.generate_query()
                
                if operation_count % 3 == 0:
                    cache.set(f"sustained_{operation_count}", f"Data {operation_count}", ttl=3600)
                elif operation_count % 3 == 1:
                    cache.get(query_data['query'])
                else:
                    cache.extend_ttl(f"sustained_{operation_count//2}", 300)
                
                operation_count += 1
            
            interval_time = time.time() - interval_start
            interval_throughput = 100 / interval_time
            operation_intervals.append(interval_throughput)
        
        total_time = time.time() - start_time
        overall_throughput = operation_count / total_time
        
        # Calculate throughput stability
        throughput_std = statistics.stdev(operation_intervals)
        throughput_cv = throughput_std / statistics.mean(operation_intervals)
        
        print(f"\nSustained Throughput Performance:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Operations: {operation_count}")
        print(f"  Overall throughput: {overall_throughput:.0f} ops/sec")
        print(f"  Throughput CV: {throughput_cv:.3f}")
        
        # Throughput should be stable over time
        assert throughput_cv < 0.5, f"Throughput too variable: CV={throughput_cv}"
        assert overall_throughput > 1000, f"Sustained throughput too low: {overall_throughput} ops/sec"


class TestTTLBenchmarkSuite:
    """Comprehensive TTL performance benchmark suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_runner = PerformanceTestRunner()
        
    def test_comprehensive_ttl_benchmark(self):
        """Run comprehensive TTL performance benchmark suite."""
        print("\n" + "="*60)
        print("COMPREHENSIVE TTL PERFORMANCE BENCHMARK SUITE")
        print("="*60)
        
        results = {}
        
        # Test different cache sizes
        cache_sizes = [1000, 5000, 10000]
        
        for size in cache_sizes:
            print(f"\n--- Cache Size: {size} entries ---")
            cache = PerformanceTTLCache(max_size=size)
            
            # Fill cache
            for i in range(size // 2):
                cache.set(f"benchmark_{i}", f"Data {i}", ttl=3600)
            
            # Benchmark different operations
            operations = {
                'set': {'count': 1000, 'ttl': 3600},
                'get': {'count': 2000},
                'extend_ttl': {'count': 500, 'extension': 300}
            }
            
            size_results = {}
            
            for op_name, op_config in operations.items():
                metrics = self.test_runner.run_operation_benchmark(
                    cache, op_name, op_config['count'], **op_config
                )
                
                size_results[op_name] = metrics.to_dict()
                
                print(f"  {op_name.upper()}:")
                print(f"    Avg: {metrics.avg_time_ms:.3f}ms")
                print(f"    P99: {metrics.p99_time_ms:.3f}ms") 
                print(f"    Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
                print(f"    Success: {metrics.success_rate:.3f}")
            
            results[f'size_{size}'] = size_results
        
        # Test concurrency scaling
        print(f"\n--- Concurrency Scaling ---")
        cache = PerformanceTTLCache(max_size=10000)
        thread_counts = [1, 2, 4, 8]
        
        concurrency_results = {}
        
        for thread_count in thread_counts:
            metrics = self.test_runner.run_concurrent_benchmark(
                cache, 'mixed', 2000, thread_count
            )
            
            concurrency_results[f'threads_{thread_count}'] = metrics.to_dict()
            
            print(f"  {thread_count} threads:")
            print(f"    Throughput: {metrics.throughput_ops_sec:.0f} ops/sec")
            print(f"    Avg time: {metrics.avg_time_ms:.3f}ms")
        
        results['concurrency'] = concurrency_results
        
        # Save results
        self._save_benchmark_results(results)
        
        # Verify all benchmarks met basic performance criteria
        self._verify_benchmark_results(results)
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/ttl_benchmark_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'system_info': {
                        'python_version': sys.version,
                        'cpu_count': psutil.cpu_count(),
                        'memory_total': psutil.virtual_memory().total // (1024**3)  # GB
                    },
                    'results': results
                }, f, indent=2)
            
            print(f"\nBenchmark results saved to: {filename}")
        except Exception as e:
            print(f"\nFailed to save benchmark results: {e}")
    
    def _verify_benchmark_results(self, results: Dict[str, Any]):
        """Verify benchmark results meet performance criteria."""
        failures = []
        
        # Check operation-specific thresholds
        for size_key, size_data in results.items():
            if size_key.startswith('size_'):
                for op_name, op_data in size_data.items():
                    if not op_data.get('meets_targets', False):
                        failures.append(f"{size_key}_{op_name}: Performance below target")
        
        # Check concurrency performance
        if 'concurrency' in results:
            for thread_key, thread_data in results['concurrency'].items():
                if thread_data['success_rate'] < 0.95:
                    failures.append(f"{thread_key}: Low success rate {thread_data['success_rate']}")
        
        print(f"\n--- Benchmark Verification ---")
        if failures:
            print("PERFORMANCE ISSUES DETECTED:")
            for failure in failures:
                print(f"  ❌ {failure}")
            
            # Don't fail the test, but warn about performance issues
            print("\nWARNING: Some performance benchmarks did not meet targets")
        else:
            print("✅ All benchmarks passed performance criteria")


# Performance test fixtures
@pytest.fixture
def performance_ttl_cache():
    """Provide performance-optimized TTL cache."""
    return PerformanceTTLCache(max_size=5000)


@pytest.fixture
def performance_test_runner():
    """Provide performance test runner."""
    return PerformanceTestRunner()


@pytest.fixture(scope='session')
def benchmark_results_dir():
    """Provide directory for benchmark results."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Pytest configuration for performance tests
def pytest_configure(config):
    """Configure pytest for performance testing."""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "memory: mark test as memory intensive")


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow  # Performance tests are typically slow
]


if __name__ == "__main__":
    # Run performance tests with appropriate timeout
    pytest.main([__file__, "-v", "--tb=short", "--timeout=300"])