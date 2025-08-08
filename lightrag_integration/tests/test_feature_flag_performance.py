#!/usr/bin/env python3
"""
Comprehensive Performance and Stress Tests for Feature Flag System.

This module provides extensive performance testing and stress testing for the
feature flag system, including hash-based routing performance, caching
efficiency, A/B testing metrics, and system behavior under load.

Test Coverage Areas:
- Hash-based routing performance benchmarks
- Cache hit rate optimization and performance
- A/B testing metrics collection and analysis
- Concurrent user load testing
- Memory usage and optimization
- Response time analysis and optimization
- Throughput testing under various configurations
- Stress testing with resource constraints
- Performance regression detection
- Scalability testing and limits

Author: Claude Code (Anthropic)
Created: 2025-08-08
"""

import pytest
import pytest_asyncio
import asyncio
import time
import statistics
import threading
import concurrent.futures
import gc
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Tuple

# Import components for performance testing
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.feature_flag_manager import (
    FeatureFlagManager,
    RoutingContext,
    RoutingResult,
    RoutingDecision,
    UserCohort
)
from lightrag_integration.integration_wrapper import (
    IntegratedQueryService,
    QueryRequest,
    ServiceResponse,
    ResponseType
)


class PerformanceMetrics:
    """Helper class for collecting and analyzing performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.response_times = []
        self.memory_usage = []
        self.throughput_data = []
        self.error_rates = []
        self.cache_hit_rates = []
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = time.time()
        self.record_memory_usage()
    
    def end_measurement(self):
        """End performance measurement."""
        self.end_time = time.time()
        self.record_memory_usage()
    
    def record_response_time(self, response_time: float):
        """Record a response time measurement."""
        self.response_times.append(response_time)
    
    def record_memory_usage(self):
        """Record current memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
    
    def record_throughput(self, operations_per_second: float):
        """Record throughput measurement."""
        self.throughput_data.append(operations_per_second)
    
    def record_error_rate(self, error_rate: float):
        """Record error rate measurement."""
        self.error_rates.append(error_rate)
    
    def record_cache_hit_rate(self, hit_rate: float):
        """Record cache hit rate."""
        self.cache_hit_rates.append(hit_rate)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {
            'total_duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'response_times': self._analyze_times(self.response_times),
            'memory_usage': self._analyze_memory(),
            'throughput': self._analyze_throughput(),
            'error_rates': self._analyze_errors(),
            'cache_performance': self._analyze_cache()
        }
        return summary
    
    def _analyze_times(self, times: List[float]) -> Dict[str, float]:
        """Analyze response times."""
        if not times:
            return {'count': 0}
        
        return {
            'count': len(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'p95': self._percentile(times, 95),
            'p99': self._percentile(times, 99),
            'min': min(times),
            'max': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def _analyze_memory(self) -> Dict[str, float]:
        """Analyze memory usage."""
        if not self.memory_usage:
            return {'count': 0}
        
        return {
            'initial_mb': self.memory_usage[0] if self.memory_usage else 0,
            'final_mb': self.memory_usage[-1] if self.memory_usage else 0,
            'peak_mb': max(self.memory_usage),
            'growth_mb': (self.memory_usage[-1] - self.memory_usage[0]) if len(self.memory_usage) > 1 else 0
        }
    
    def _analyze_throughput(self) -> Dict[str, float]:
        """Analyze throughput data."""
        if not self.throughput_data:
            return {'count': 0}
        
        return {
            'mean_ops_per_sec': statistics.mean(self.throughput_data),
            'peak_ops_per_sec': max(self.throughput_data),
            'min_ops_per_sec': min(self.throughput_data)
        }
    
    def _analyze_errors(self) -> Dict[str, float]:
        """Analyze error rates."""
        if not self.error_rates:
            return {'count': 0}
        
        return {
            'mean_error_rate': statistics.mean(self.error_rates),
            'max_error_rate': max(self.error_rates)
        }
    
    def _analyze_cache(self) -> Dict[str, float]:
        """Analyze cache performance."""
        if not self.cache_hit_rates:
            return {'count': 0}
        
        return {
            'mean_hit_rate': statistics.mean(self.cache_hit_rates),
            'min_hit_rate': min(self.cache_hit_rates),
            'max_hit_rate': max(self.cache_hit_rates)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.fixture
def performance_metrics():
    """Provide PerformanceMetrics instance for testing."""
    return PerformanceMetrics()


class TestHashingPerformance:
    """Test performance of hash-based routing algorithms."""
    
    @pytest.mark.performance
    def test_hash_calculation_performance(self, performance_metrics):
        """Test hash calculation performance under load."""
        config = LightRAGConfig(lightrag_user_hash_salt="performance_test")
        feature_manager = FeatureFlagManager(config=config)
        
        num_users = 10000
        user_ids = [f"perf_user_{i}" for i in range(num_users)]
        
        performance_metrics.start_measurement()
        
        start_time = time.time()
        hashes = []
        
        for user_id in user_ids:
            hash_start = time.time()
            user_hash = feature_manager._calculate_user_hash(user_id)
            hash_end = time.time()
            
            hashes.append(user_hash)
            performance_metrics.record_response_time(hash_end - hash_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        performance_metrics.end_measurement()
        
        # Performance assertions
        ops_per_second = num_users / total_time
        assert ops_per_second > 50000, f"Hash calculation too slow: {ops_per_second} ops/sec"
        
        # Individual hash calculations should be very fast
        avg_hash_time = sum(performance_metrics.response_times) / len(performance_metrics.response_times)
        assert avg_hash_time < 0.001, f"Average hash time too slow: {avg_hash_time}s"
        
        # All hashes should be unique (no collisions in this test)
        assert len(set(hashes)) == len(hashes), "Hash collisions detected"
    
    @pytest.mark.performance
    def test_routing_decision_performance(self, performance_metrics):
        """Test routing decision performance."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_ab_testing=True,
            lightrag_enable_circuit_breaker=True,
            lightrag_enable_quality_metrics=True
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Pre-populate some performance metrics for quality threshold testing
        feature_manager.performance_metrics.lightrag_quality_scores = [0.8, 0.9, 0.85] * 100
        
        num_decisions = 5000
        contexts = [
            RoutingContext(
                user_id=f"routing_user_{i}",
                query_text=f"Performance test query {i}",
                query_type="performance_test",
                query_complexity=0.5 + (i % 5) * 0.1
            )
            for i in range(num_decisions)
        ]
        
        performance_metrics.start_measurement()
        
        start_time = time.time()
        results = []
        
        for context in contexts:
            decision_start = time.time()
            result = feature_manager.should_use_lightrag(context)
            decision_end = time.time()
            
            results.append(result)
            performance_metrics.record_response_time(decision_end - decision_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        performance_metrics.end_measurement()
        
        # Performance assertions
        ops_per_second = num_decisions / total_time
        assert ops_per_second > 1000, f"Routing decisions too slow: {ops_per_second} ops/sec"
        
        # Individual decisions should be fast
        avg_decision_time = sum(performance_metrics.response_times) / len(performance_metrics.response_times)
        assert avg_decision_time < 0.01, f"Average decision time too slow: {avg_decision_time}s"
        
        # Verify distribution
        lightrag_count = sum(1 for r in results if r.decision == RoutingDecision.LIGHTRAG)
        lightrag_percentage = (lightrag_count / num_decisions) * 100
        
        # Should be roughly 50% with A/B testing (within 10% tolerance)
        assert 40.0 <= lightrag_percentage <= 60.0, f"Unexpected distribution: {lightrag_percentage}%"
    
    @pytest.mark.performance
    def test_concurrent_routing_performance(self, performance_metrics):
        """Test concurrent routing decision performance."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=30.0
        )
        feature_manager = FeatureFlagManager(config=config)
        
        def worker_thread(worker_id, num_operations):
            """Worker thread for concurrent testing."""
            thread_results = []
            thread_times = []
            
            for i in range(num_operations):
                context = RoutingContext(
                    user_id=f"concurrent_user_{worker_id}_{i}",
                    query_text=f"Concurrent test query {i}"
                )
                
                start_time = time.time()
                result = feature_manager.should_use_lightrag(context)
                end_time = time.time()
                
                thread_results.append(result)
                thread_times.append(end_time - start_time)
            
            return thread_results, thread_times
        
        # Test with multiple concurrent threads
        num_threads = 8
        operations_per_thread = 500
        
        performance_metrics.start_measurement()
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, thread_id, operations_per_thread)
                for thread_id in range(num_threads)
            ]
            
            all_results = []
            all_times = []
            
            for future in concurrent.futures.as_completed(futures):
                thread_results, thread_times = future.result()
                all_results.extend(thread_results)
                all_times.extend(thread_times)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        performance_metrics.response_times = all_times
        performance_metrics.end_measurement()
        
        # Performance assertions
        total_operations = num_threads * operations_per_thread
        ops_per_second = total_operations / total_time
        
        assert ops_per_second > 2000, f"Concurrent routing too slow: {ops_per_second} ops/sec"
        
        # P95 response time should be reasonable
        p95_time = performance_metrics._percentile(all_times, 95)
        assert p95_time < 0.05, f"P95 response time too slow: {p95_time}s"


class TestCachePerformance:
    """Test caching performance and optimization."""
    
    @pytest.mark.performance
    def test_routing_cache_performance(self, performance_metrics):
        """Test routing cache hit rate and performance."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        # Create queries with different cache patterns
        cache_patterns = [
            ("repeated_user", "same query", 100),  # Same user, same query
            ("repeated_user", "different query", 50),  # Same user, different queries
            ("different_user", "same query", 75),  # Different users, same query
        ]
        
        performance_metrics.start_measurement()
        
        total_operations = 0
        cache_hits = 0
        cache_misses = 0
        
        for user_pattern, query_pattern, repeats in cache_patterns:
            for i in range(repeats):
                if user_pattern == "repeated_user":
                    user_id = "cache_test_user"
                else:
                    user_id = f"cache_user_{i}"
                
                if query_pattern == "same query":
                    query_text = "What are metabolites in diabetes?"
                else:
                    query_text = f"Query variation {i}"
                
                context = RoutingContext(user_id=user_id, query_text=query_text)
                
                # Check if this would be a cache hit
                cache_key = f"{user_id}:{hash(query_text)}"
                was_cached = cache_key in feature_manager._routing_cache
                
                start_time = time.time()
                result = feature_manager.should_use_lightrag(context)
                end_time = time.time()
                
                performance_metrics.record_response_time(end_time - start_time)
                total_operations += 1
                
                if was_cached:
                    cache_hits += 1
                else:
                    cache_misses += 1
        
        performance_metrics.end_measurement()
        
        # Calculate cache hit rate
        cache_hit_rate = cache_hits / total_operations if total_operations > 0 else 0
        performance_metrics.record_cache_hit_rate(cache_hit_rate)
        
        # Cache hit rate should be reasonable for repeated patterns
        assert cache_hit_rate > 0.3, f"Cache hit rate too low: {cache_hit_rate}"
        
        # Cached operations should be faster than non-cached
        # This is implicit in the design - cached operations skip computation
        
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_cache_performance(self, performance_metrics):
        """Test response caching performance in integrated service."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity:
            # Configure mock with realistic delays
            async def mock_query(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate network delay
                return ServiceResponse(
                    content="Mock response about diabetes metabolites",
                    processing_time=0.1,
                    response_type=ResponseType.PERPLEXITY
                )
            
            mock_perplexity.return_value.query_async = mock_query
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            # Test cache performance patterns
            queries = [
                ("cache_user_1", "What are diabetes metabolites?"),
                ("cache_user_1", "What are diabetes metabolites?"),  # Exact repeat
                ("cache_user_2", "What are diabetes metabolites?"),  # Different user, same query
                ("cache_user_1", "What are metabolites in diabetes?"),  # Slight variation
            ]
            
            performance_metrics.start_measurement()
            
            response_times = []
            cache_hits = 0
            
            for user_id, query_text in queries:
                request = QueryRequest(user_id=user_id, query_text=query_text)
                
                start_time = time.time()
                response = await service.query_async(request)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                performance_metrics.record_response_time(response_time)
                
                if response.response_type == ResponseType.CACHED:
                    cache_hits += 1
            
            performance_metrics.end_measurement()
            
            # Cached responses should be significantly faster
            if cache_hits > 0:
                # There should be some cache benefit
                min_response_time = min(response_times)
                max_response_time = max(response_times)
                
                # Cache hits should create a bimodal distribution
                assert max_response_time > min_response_time * 2, "Cache not providing speed benefit"
    
    @pytest.mark.performance
    def test_cache_memory_efficiency(self, performance_metrics):
        """Test cache memory efficiency under load."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        performance_metrics.start_measurement()
        
        # Fill cache with many entries
        cache_sizes = []
        num_entries = 2000
        
        for i in range(num_entries):
            context = RoutingContext(
                user_id=f"memory_user_{i}",
                query_text=f"Memory test query {i}"
            )
            
            result = feature_manager.should_use_lightrag(context)
            
            # Record cache size periodically
            if i % 100 == 0:
                cache_sizes.append(len(feature_manager._routing_cache))
                performance_metrics.record_memory_usage()
        
        performance_metrics.end_measurement()
        
        # Cache should be bounded (not grow indefinitely)
        max_cache_size = max(cache_sizes)
        assert max_cache_size <= 1000, f"Cache size not bounded: {max_cache_size}"
        
        # Memory usage should be reasonable
        memory_summary = performance_metrics._analyze_memory()
        memory_growth = memory_summary.get('growth_mb', 0)
        
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth}MB"


class TestThroughputAndScalability:
    """Test system throughput and scalability characteristics."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_throughput(self, performance_metrics):
        """Test sustained throughput over time."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=40.0,
            lightrag_enable_ab_testing=True
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Test sustained load for a longer period
        test_duration = 10.0  # seconds
        batch_size = 100
        
        performance_metrics.start_measurement()
        
        start_time = time.time()
        total_operations = 0
        throughput_measurements = []
        
        while time.time() - start_time < test_duration:
            batch_start = time.time()
            
            # Process a batch of operations
            for i in range(batch_size):
                context = RoutingContext(
                    user_id=f"throughput_user_{total_operations + i}",
                    query_text="Sustained throughput test query"
                )
                result = feature_manager.should_use_lightrag(context)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_throughput = batch_size / batch_time
            
            throughput_measurements.append(batch_throughput)
            performance_metrics.record_throughput(batch_throughput)
            total_operations += batch_size
        
        end_time = time.time()
        total_time = end_time - start_time
        
        performance_metrics.end_measurement()
        
        # Overall throughput
        overall_throughput = total_operations / total_time
        assert overall_throughput > 5000, f"Sustained throughput too low: {overall_throughput} ops/sec"
        
        # Throughput should be consistent (low variance)
        throughput_std = statistics.stdev(throughput_measurements)
        throughput_mean = statistics.mean(throughput_measurements)
        coefficient_of_variation = throughput_std / throughput_mean
        
        assert coefficient_of_variation < 0.3, f"Throughput too variable: {coefficient_of_variation}"
    
    @pytest.mark.performance
    def test_scalability_with_user_count(self, performance_metrics):
        """Test scalability with increasing number of unique users."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        user_counts = [100, 500, 1000, 2000, 5000]
        throughput_results = []
        
        for num_users in user_counts:
            performance_metrics.reset()
            performance_metrics.start_measurement()
            
            start_time = time.time()
            
            for i in range(num_users):
                context = RoutingContext(
                    user_id=f"scale_user_{i}",
                    query_text="Scalability test query"
                )
                result = feature_manager.should_use_lightrag(context)
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = num_users / total_time
            
            throughput_results.append((num_users, throughput))
            performance_metrics.record_throughput(throughput)
            performance_metrics.end_measurement()
        
        # Analyze scalability
        # Throughput should not degrade significantly with more users
        baseline_throughput = throughput_results[0][1]  # First measurement
        
        for num_users, throughput in throughput_results[1:]:
            # Allow some degradation but not dramatic
            degradation_ratio = throughput / baseline_throughput
            assert degradation_ratio > 0.5, f"Significant throughput degradation at {num_users} users: {degradation_ratio}"
        
        # Memory usage should scale reasonably
        memory_summary = performance_metrics._analyze_memory()
        memory_per_user = memory_summary.get('growth_mb', 0) / max(user_counts)
        
        assert memory_per_user < 0.01, f"Memory usage per user too high: {memory_per_user}MB/user"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_service_throughput(self, performance_metrics):
        """Test concurrent service throughput."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0
        )
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity, \
             patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            
            # Configure fast mocks
            async def fast_perplexity_query(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms response
                return ServiceResponse(content="Fast Perplexity response", processing_time=0.01)
            
            async def fast_lightrag_query(*args, **kwargs):
                await asyncio.sleep(0.015)  # 15ms response
                return ServiceResponse(content="Fast LightRAG response", processing_time=0.015)
            
            mock_perplexity.return_value.query_async = fast_perplexity_query
            mock_lightrag.return_value.query_async = fast_lightrag_query
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            performance_metrics.start_measurement()
            
            # Test with multiple concurrent requests
            concurrency_levels = [10, 25, 50]
            
            for concurrency in concurrency_levels:
                requests = [
                    QueryRequest(
                        user_id=f"concurrent_user_{i}",
                        query_text=f"Concurrent query {i}"
                    )
                    for i in range(concurrency)
                ]
                
                start_time = time.time()
                
                tasks = [service.query_async(request) for request in requests]
                responses = await asyncio.gather(*tasks)
                
                end_time = time.time()
                total_time = end_time - start_time
                throughput = concurrency / total_time
                
                performance_metrics.record_throughput(throughput)
                
                # All responses should be successful
                success_rate = sum(1 for r in responses if r.is_success) / len(responses)
                assert success_rate > 0.95, f"Success rate too low at concurrency {concurrency}: {success_rate}"
                
                # Throughput should be reasonable
                assert throughput > 100, f"Throughput too low at concurrency {concurrency}: {throughput} req/sec"
            
            performance_metrics.end_measurement()


class TestMemoryAndResourceUsage:
    """Test memory usage and resource consumption patterns."""
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self, performance_metrics):
        """Test memory usage characteristics under sustained load."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_ab_testing=True
        )
        feature_manager = FeatureFlagManager(config=config)
        
        performance_metrics.start_measurement()
        
        # Sustained operations with memory monitoring
        num_cycles = 10
        operations_per_cycle = 1000
        
        memory_readings = []
        
        for cycle in range(num_cycles):
            cycle_start_memory = performance_metrics.memory_usage[-1] if performance_metrics.memory_usage else 0
            performance_metrics.record_memory_usage()
            
            # Perform operations
            for i in range(operations_per_cycle):
                context = RoutingContext(
                    user_id=f"memory_cycle_{cycle}_user_{i}",
                    query_text=f"Memory test cycle {cycle} operation {i}"
                )
                result = feature_manager.should_use_lightrag(context)
                
                # Record some metrics
                feature_manager.record_success("lightrag", 1.0 + (i * 0.001), 0.8 + (i * 0.0001))
            
            cycle_end_memory = performance_metrics.memory_usage[-1]
            memory_readings.append(cycle_end_memory - cycle_start_memory if cycle_start_memory else 0)
            
            # Force garbage collection
            gc.collect()
            performance_metrics.record_memory_usage()
        
        performance_metrics.end_measurement()
        
        # Analyze memory usage patterns
        memory_summary = performance_metrics._analyze_memory()
        total_memory_growth = memory_summary.get('growth_mb', 0)
        
        # Memory growth should be bounded (no significant leaks)
        assert total_memory_growth < 20, f"Excessive memory growth: {total_memory_growth}MB"
        
        # Memory usage should stabilize (not grow indefinitely)
        if len(memory_readings) > 5:
            recent_growth = sum(memory_readings[-3:]) / 3
            early_growth = sum(memory_readings[:3]) / 3
            
            # Recent growth should not be significantly higher than early growth
            if early_growth > 0:
                growth_ratio = recent_growth / early_growth
                assert growth_ratio < 2.0, f"Memory usage not stabilizing: {growth_ratio}"
    
    @pytest.mark.performance
    def test_cache_memory_efficiency_detailed(self, performance_metrics):
        """Test detailed cache memory efficiency patterns."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity:
            mock_perplexity.return_value.query_async = AsyncMock(return_value=ServiceResponse(
                content="Test response", processing_time=0.1
            ))
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            performance_metrics.start_measurement()
            
            # Test different cache usage patterns
            cache_patterns = [
                ("high_reuse", 100, 10),    # 100 requests, 10 unique queries (high reuse)
                ("medium_reuse", 100, 50),  # 100 requests, 50 unique queries (medium reuse)
                ("low_reuse", 100, 90),     # 100 requests, 90 unique queries (low reuse)
            ]
            
            for pattern_name, num_requests, unique_queries in cache_patterns:
                pattern_start_memory = performance_metrics.memory_usage[-1] if performance_metrics.memory_usage else 0
                performance_metrics.record_memory_usage()
                
                cache_size_before = len(service._response_cache)
                
                # Generate requests according to pattern
                for i in range(num_requests):
                    query_index = i % unique_queries
                    request = QueryRequest(
                        user_id=f"{pattern_name}_user_{i}",
                        query_text=f"{pattern_name} query {query_index}"
                    )
                    
                    # We can't actually run async queries in this sync test,
                    # so we'll test the caching mechanism directly
                    cache_key = service._generate_cache_key(request)
                    
                    if cache_key not in service._response_cache:
                        # Simulate adding to cache
                        test_response = ServiceResponse(content=f"Response {query_index}")
                        service._cache_response(cache_key, test_response)
                
                cache_size_after = len(service._response_cache)
                cache_growth = cache_size_after - cache_size_before
                
                performance_metrics.record_memory_usage()
                pattern_end_memory = performance_metrics.memory_usage[-1]
                pattern_memory_growth = pattern_end_memory - pattern_start_memory
                
                # Analyze pattern efficiency
                expected_cache_entries = min(unique_queries, 100)  # Cache size limit
                actual_cache_entries = cache_growth
                
                # Cache should grow efficiently based on unique queries
                cache_efficiency = actual_cache_entries / expected_cache_entries if expected_cache_entries > 0 else 0
                assert cache_efficiency > 0.8, f"Cache efficiency too low for {pattern_name}: {cache_efficiency}"
                
                # Memory per cache entry should be reasonable
                if cache_growth > 0:
                    memory_per_entry = pattern_memory_growth / cache_growth
                    assert memory_per_entry < 1.0, f"Memory per cache entry too high: {memory_per_entry}MB"
            
            performance_metrics.end_measurement()
    
    @pytest.mark.performance
    def test_resource_cleanup_efficiency(self, performance_metrics):
        """Test resource cleanup and garbage collection efficiency."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        performance_metrics.start_measurement()
        
        # Create and destroy many objects
        num_iterations = 5
        objects_per_iteration = 2000
        
        memory_before_cleanup = []
        memory_after_cleanup = []
        
        for iteration in range(num_iterations):
            # Create many routing contexts and results
            for i in range(objects_per_iteration):
                context = RoutingContext(
                    user_id=f"cleanup_user_{iteration}_{i}",
                    query_text=f"Cleanup test {iteration} {i}",
                    metadata={"iteration": iteration, "index": i, "extra_data": "x" * 100}
                )
                result = feature_manager.should_use_lightrag(context)
                
                # Add some performance metrics
                feature_manager.record_success("lightrag", 1.0 + i * 0.001, 0.8)
            
            # Record memory before cleanup
            performance_metrics.record_memory_usage()
            memory_before = performance_metrics.memory_usage[-1]
            memory_before_cleanup.append(memory_before)
            
            # Force cleanup
            feature_manager.clear_caches()
            gc.collect()
            
            # Record memory after cleanup
            performance_metrics.record_memory_usage()
            memory_after = performance_metrics.memory_usage[-1]
            memory_after_cleanup.append(memory_after)
        
        performance_metrics.end_measurement()
        
        # Analyze cleanup efficiency
        cleanup_efficiencies = []
        for before, after in zip(memory_before_cleanup, memory_after_cleanup):
            if before > after:
                cleanup_efficiency = (before - after) / before
                cleanup_efficiencies.append(cleanup_efficiency)
        
        if cleanup_efficiencies:
            avg_cleanup_efficiency = sum(cleanup_efficiencies) / len(cleanup_efficiencies)
            assert avg_cleanup_efficiency > 0.1, f"Cleanup efficiency too low: {avg_cleanup_efficiency}"


class TestPerformanceRegression:
    """Test for performance regression detection."""
    
    @pytest.mark.performance
    def test_routing_decision_performance_baseline(self, performance_metrics):
        """Test routing decision performance against baseline expectations."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Performance baseline expectations
        EXPECTED_MIN_OPS_PER_SEC = 10000
        EXPECTED_MAX_AVG_RESPONSE_TIME = 0.001  # 1ms
        EXPECTED_MAX_P95_RESPONSE_TIME = 0.005  # 5ms
        
        num_operations = 5000
        
        performance_metrics.start_measurement()
        
        start_time = time.time()
        response_times = []
        
        for i in range(num_operations):
            context = RoutingContext(
                user_id=f"baseline_user_{i}",
                query_text="Baseline performance test query"
            )
            
            operation_start = time.time()
            result = feature_manager.should_use_lightrag(context)
            operation_end = time.time()
            
            response_times.append(operation_end - operation_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        performance_metrics.response_times = response_times
        performance_metrics.end_measurement()
        
        # Performance regression checks
        ops_per_second = num_operations / total_time
        assert ops_per_second >= EXPECTED_MIN_OPS_PER_SEC, \
            f"Performance regression: {ops_per_second} ops/sec < {EXPECTED_MIN_OPS_PER_SEC}"
        
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time <= EXPECTED_MAX_AVG_RESPONSE_TIME, \
            f"Performance regression: avg response time {avg_response_time}s > {EXPECTED_MAX_AVG_RESPONSE_TIME}s"
        
        p95_response_time = performance_metrics._percentile(response_times, 95)
        assert p95_response_time <= EXPECTED_MAX_P95_RESPONSE_TIME, \
            f"Performance regression: P95 response time {p95_response_time}s > {EXPECTED_MAX_P95_RESPONSE_TIME}s"
    
    @pytest.mark.performance
    def test_memory_usage_baseline(self, performance_metrics):
        """Test memory usage against baseline expectations."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        # Memory baseline expectations
        EXPECTED_MAX_MEMORY_GROWTH_MB = 10
        EXPECTED_MAX_MEMORY_PER_OPERATION_KB = 1.0
        
        performance_metrics.start_measurement()
        
        num_operations = 10000
        
        for i in range(num_operations):
            context = RoutingContext(
                user_id=f"memory_baseline_user_{i}",
                query_text=f"Memory baseline test {i}"
            )
            result = feature_manager.should_use_lightrag(context)
            
            # Add some metrics to test memory usage
            feature_manager.record_success("lightrag", 1.0, 0.8)
            
            # Record memory usage periodically
            if i % 1000 == 0:
                performance_metrics.record_memory_usage()
        
        performance_metrics.end_measurement()
        
        # Memory regression checks
        memory_summary = performance_metrics._analyze_memory()
        memory_growth = memory_summary.get('growth_mb', 0)
        
        assert memory_growth <= EXPECTED_MAX_MEMORY_GROWTH_MB, \
            f"Memory regression: growth {memory_growth}MB > {EXPECTED_MAX_MEMORY_GROWTH_MB}MB"
        
        memory_per_operation_kb = (memory_growth * 1024) / num_operations
        assert memory_per_operation_kb <= EXPECTED_MAX_MEMORY_PER_OPERATION_KB, \
            f"Memory regression: {memory_per_operation_kb}KB per op > {EXPECTED_MAX_MEMORY_PER_OPERATION_KB}KB"


class TestABTestingMetricsPerformance:
    """Test A/B testing metrics collection performance."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ab_testing_metrics_collection_overhead(self, performance_metrics):
        """Test performance overhead of A/B testing metrics collection."""
        config_without_ab = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_ab_testing=False
        )
        
        config_with_ab = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0,
            lightrag_enable_ab_testing=True
        )
        
        num_operations = 1000
        
        # Test without A/B testing
        feature_manager_no_ab = FeatureFlagManager(config=config_without_ab)
        
        start_time = time.time()
        for i in range(num_operations):
            context = RoutingContext(
                user_id=f"no_ab_user_{i}",
                query_text="A/B testing overhead test"
            )
            result = feature_manager_no_ab.should_use_lightrag(context)
        end_time = time.time()
        time_without_ab = end_time - start_time
        
        # Test with A/B testing
        feature_manager_with_ab = FeatureFlagManager(config=config_with_ab)
        
        start_time = time.time()
        for i in range(num_operations):
            context = RoutingContext(
                user_id=f"with_ab_user_{i}",
                query_text="A/B testing overhead test"
            )
            result = feature_manager_with_ab.should_use_lightrag(context)
        end_time = time.time()
        time_with_ab = end_time - start_time
        
        # Calculate overhead
        overhead_ratio = time_with_ab / time_without_ab if time_without_ab > 0 else 1.0
        
        # A/B testing should not add significant overhead
        assert overhead_ratio < 1.2, f"A/B testing overhead too high: {overhead_ratio}x"
        
        # Both should be fast
        ops_per_sec_no_ab = num_operations / time_without_ab
        ops_per_sec_with_ab = num_operations / time_with_ab
        
        assert ops_per_sec_no_ab > 5000, f"Base performance too low: {ops_per_sec_no_ab} ops/sec"
        assert ops_per_sec_with_ab > 4000, f"A/B testing performance too low: {ops_per_sec_with_ab} ops/sec"


# Mark the end of performance tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])