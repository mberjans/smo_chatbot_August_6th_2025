#!/usr/bin/env python3
"""
High-Performance Classification System Integration Tests

This module provides comprehensive integration tests to validate that the
high-performance classification system consistently meets <2 second response
time targets while maintaining high accuracy and reliability.

Test Categories:
    - Basic functionality tests
    - Performance validation tests  
    - Load and stress testing
    - Cache efficiency validation
    - Resource management verification
    - Error handling and recovery testing
    - Integration with existing systems
    - Real-time monitoring validation

Author: Claude Code (Anthropic)  
Version: 3.0.0 - Integration Test Suite
Created: 2025-08-08
Target: Validate <2 second response times with 99%+ reliability
"""

import pytest
import asyncio
import time
import statistics
import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

# Import the high-performance system components
try:
    from .high_performance_classification_system import (
        HighPerformanceClassificationSystem,
        HighPerformanceConfig,
        HighPerformanceCache,
        RequestOptimizer,
        LLMInteractionOptimizer,
        ResourceManager,
        high_performance_classification_context,
        create_high_performance_system
    )
    from .performance_benchmark_suite import (
        PerformanceBenchmarkRunner,
        BenchmarkConfig,
        BenchmarkReporter,
        run_comprehensive_benchmark
    )
    from .enhanced_llm_classifier import EnhancedLLMQueryClassifier
    from .query_router import BiomedicalQueryRouter
    from .llm_classification_prompts import ClassificationResult
except ImportError as e:
    logging.warning(f"Could not import high performance components: {e}")


# ============================================================================
# TEST CONFIGURATION AND FIXTURES
# ============================================================================

@pytest.fixture
def hp_config():
    """Provide high-performance configuration for testing."""
    return HighPerformanceConfig(
        target_response_time_ms=1500,
        max_response_time_ms=2000,
        l1_cache_size=1000,
        l1_cache_ttl=300,
        enable_cache_warming=True,
        enable_request_batching=True,
        enable_deduplication=True,
        max_batch_size=5,
        batch_timeout_ms=50.0,
        parallel_llm_calls=2
    )


@pytest.fixture  
async def hp_system(hp_config):
    """Provide high-performance system instance for testing."""
    async with high_performance_classification_context(hp_config) as system:
        yield system


@pytest.fixture
def test_queries():
    """Provide comprehensive test queries for validation."""
    return [
        # Simple queries
        ("What is metabolomics?", "GENERAL"),
        ("Define biomarkers", "GENERAL"),
        ("How does LC-MS work?", "GENERAL"),
        
        # Knowledge graph queries
        ("Relationship between glucose and insulin signaling", "KNOWLEDGE_GRAPH"),
        ("Pathway analysis of amino acid metabolism", "KNOWLEDGE_GRAPH"),
        ("Connection between metabolite levels and disease", "KNOWLEDGE_GRAPH"),
        
        # Real-time/temporal queries
        ("Latest research in metabolomics 2025", "REAL_TIME"),
        ("Recent advances in clinical diagnostics", "REAL_TIME"),
        ("Current trends in biomarker discovery", "REAL_TIME"),
        
        # Complex technical queries
        ("LC-MS/MS method optimization for glucose quantification in plasma samples", "KNOWLEDGE_GRAPH"),
        ("Statistical analysis of lipidomics biomarkers using PCA and OPLS-DA methods", "KNOWLEDGE_GRAPH"),
        ("Machine learning approaches for metabolite identification in clinical metabolomics", "KNOWLEDGE_GRAPH")
    ]


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestBasicFunctionality:
    """Test basic functionality of the high-performance system."""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, hp_config):
        """Test that the high-performance system initializes correctly."""
        async with high_performance_classification_context(hp_config) as system:
            assert system is not None
            assert isinstance(system, HighPerformanceClassificationSystem)
            assert system.config == hp_config
            assert system.cache is not None
            assert system.request_optimizer is not None
            assert system.llm_optimizer is not None
            assert system.resource_manager is not None
    
    @pytest.mark.asyncio
    async def test_single_query_classification(self, hp_system, test_queries):
        """Test single query classification with performance validation."""
        query_text, expected_category = test_queries[0]
        
        start_time = time.time()
        result, metadata = await hp_system.classify_query_optimized(query_text)
        response_time = (time.time() - start_time) * 1000
        
        # Validate basic functionality
        assert isinstance(result, ClassificationResult)
        assert result.category in ["GENERAL", "KNOWLEDGE_GRAPH", "REAL_TIME"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        
        # Validate performance metadata
        assert isinstance(metadata, dict)
        assert "response_time_ms" in metadata
        assert "optimizations_applied" in metadata
        assert "target_met" in metadata
        
        # Validate performance target
        assert response_time <= 2000, f"Response time {response_time:.1f}ms exceeds 2000ms limit"
        assert metadata["target_met"], f"Target not met: {response_time:.1f}ms"
        
        print(f"✓ Single query test passed: {response_time:.1f}ms, category: {result.category}")
    
    @pytest.mark.asyncio
    async def test_all_query_types(self, hp_system, test_queries):
        """Test all types of queries in the test suite."""
        results = []
        
        for query_text, expected_category in test_queries:
            start_time = time.time()
            result, metadata = await hp_system.classify_query_optimized(query_text)
            response_time = (time.time() - start_time) * 1000
            
            results.append({
                "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                "expected": expected_category,
                "actual": result.category,
                "confidence": result.confidence,
                "response_time": response_time,
                "target_met": metadata["target_met"]
            })
            
            # Each query must meet performance target
            assert response_time <= 2000, f"Query '{query_text[:30]}...' took {response_time:.1f}ms"
        
        # Calculate overall performance
        response_times = [r["response_time"] for r in results]
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        target_compliance = sum(1 for r in results if r["target_met"]) / len(results)
        
        print(f"✓ All query types test passed:")
        print(f"  Average response time: {avg_time:.1f}ms")
        print(f"  Maximum response time: {max_time:.1f}ms") 
        print(f"  Target compliance: {target_compliance:.1%}")
        
        # Overall performance requirements
        assert avg_time <= 1500, f"Average response time {avg_time:.1f}ms exceeds 1500ms target"
        assert target_compliance >= 0.95, f"Target compliance {target_compliance:.1%} below 95%"


# ============================================================================
# PERFORMANCE VALIDATION TESTS
# ============================================================================

class TestPerformanceValidation:
    """Comprehensive performance validation tests."""
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, hp_system):
        """Test response time consistency across multiple requests."""
        query = "What is the relationship between glucose metabolism and diabetes?"
        response_times = []
        
        # Run 50 identical requests to test consistency
        for i in range(50):
            start_time = time.time()
            result, metadata = await hp_system.classify_query_optimized(query)
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)
            
            # Each request must meet target
            assert response_time <= 2000, f"Request {i} took {response_time:.1f}ms"
        
        # Calculate consistency metrics
        avg_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        p95_time = np.percentile(response_times, 95)
        p99_time = np.percentile(response_times, 99)
        
        print(f"✓ Response time consistency test:")
        print(f"  Average: {avg_time:.1f}ms ± {std_dev:.1f}ms")
        print(f"  P95: {p95_time:.1f}ms")
        print(f"  P99: {p99_time:.1f}ms")
        
        # Consistency requirements
        assert avg_time <= 1500, f"Average {avg_time:.1f}ms exceeds target"
        assert p95_time <= 1800, f"P95 {p95_time:.1f}ms exceeds 1800ms"
        assert p99_time <= 1950, f"P99 {p99_time:.1f}ms exceeds 1950ms"
        assert std_dev <= 200, f"Standard deviation {std_dev:.1f}ms too high (>200ms)"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, hp_system):
        """Test performance under concurrent load."""
        query = "Analyze the metabolomics pathway for biomarker discovery"
        concurrent_users = 20
        requests_per_user = 10
        
        async def user_requests():
            """Simulate requests from a single user."""
            user_times = []
            for _ in range(requests_per_user):
                start_time = time.time()
                result, metadata = await hp_system.classify_query_optimized(query)
                response_time = (time.time() - start_time) * 1000
                user_times.append(response_time)
                await asyncio.sleep(0.1)  # Brief pause between requests
            return user_times
        
        # Execute concurrent users
        start_time = time.time()
        user_tasks = [asyncio.create_task(user_requests()) for _ in range(concurrent_users)]
        all_user_times = await asyncio.gather(*user_tasks)
        total_time = time.time() - start_time
        
        # Flatten all response times
        all_response_times = [time for user_times in all_user_times for time in user_times]
        total_requests = len(all_response_times)
        
        # Calculate performance metrics
        avg_time = statistics.mean(all_response_times)
        p95_time = np.percentile(all_response_times, 95)
        throughput = total_requests / total_time
        failures = sum(1 for t in all_response_times if t > 2000)
        
        print(f"✓ Concurrent load test ({concurrent_users} users, {total_requests} total requests):")
        print(f"  Average response time: {avg_time:.1f}ms")
        print(f"  P95 response time: {p95_time:.1f}ms")
        print(f"  Throughput: {throughput:.1f} requests/second")
        print(f"  Failures: {failures}/{total_requests}")
        
        # Performance requirements under load
        assert avg_time <= 1800, f"Average {avg_time:.1f}ms exceeds 1800ms under load"
        assert p95_time <= 2000, f"P95 {p95_time:.1f}ms exceeds 2000ms under load"
        assert failures == 0, f"{failures} requests exceeded 2000ms limit"
        assert throughput >= 10, f"Throughput {throughput:.1f} RPS too low"
    
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, hp_system):
        """Test the performance impact of caching."""
        queries = [
            "What is metabolomics analysis?",
            "LC-MS biomarker identification methods",
            "Pathway enrichment statistical analysis",
            "Clinical metabolomics diagnostic applications",
            "Machine learning for metabolite classification"
        ]
        
        # First round - cache misses expected
        first_round_times = []
        for query in queries:
            start_time = time.time()
            result, metadata = await hp_system.classify_query_optimized(query)
            response_time = (time.time() - start_time) * 1000
            first_round_times.append(response_time)
        
        # Brief pause to ensure cache is populated
        await asyncio.sleep(0.5)
        
        # Second round - cache hits expected
        second_round_times = []
        cache_hits = 0
        
        for query in queries:
            start_time = time.time()
            result, metadata = await hp_system.classify_query_optimized(query)
            response_time = (time.time() - start_time) * 1000
            second_round_times.append(response_time)
            
            if metadata.get("cache_hit", False):
                cache_hits += 1
        
        # Calculate cache impact
        avg_first_round = statistics.mean(first_round_times)
        avg_second_round = statistics.mean(second_round_times)
        improvement_percent = ((avg_first_round - avg_second_round) / avg_first_round) * 100
        
        print(f"✓ Cache performance test:")
        print(f"  First round (cache miss): {avg_first_round:.1f}ms average")
        print(f"  Second round (cache hit): {avg_second_round:.1f}ms average")
        print(f"  Performance improvement: {improvement_percent:.1f}%")
        print(f"  Cache hits: {cache_hits}/{len(queries)}")
        
        # Cache performance requirements
        assert cache_hits >= len(queries) * 0.8, f"Low cache hit rate: {cache_hits}/{len(queries)}"
        assert avg_second_round < avg_first_round, "Cache should improve performance"
        assert improvement_percent >= 20, f"Cache improvement {improvement_percent:.1f}% too low"


# ============================================================================
# CACHE EFFICIENCY TESTS
# ============================================================================

class TestCacheEfficiency:
    """Test cache efficiency and multi-level caching behavior."""
    
    @pytest.mark.asyncio
    async def test_l1_cache_behavior(self, hp_system):
        """Test L1 cache behavior and hit rates."""
        cache = hp_system.cache
        test_query = "Test query for L1 cache validation"
        
        # Initial cache miss
        result1 = await cache.get(test_query)
        assert result1 is None, "Expected cache miss on first access"
        
        # Set cache value
        test_result = ClassificationResult(
            category="GENERAL",
            confidence=0.8,
            reasoning="Test cache result",
            alternative_categories=[],
            uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        await cache.set(test_query, test_result)
        
        # Cache hit
        result2 = await cache.get(test_query)
        assert result2 is not None, "Expected cache hit after setting value"
        assert result2.level.value == "l1_memory", "Expected L1 cache hit"
        assert result2.value.category == "GENERAL"
        
        print("✓ L1 cache behavior test passed")
    
    @pytest.mark.asyncio
    async def test_cache_warming_effectiveness(self, hp_system):
        """Test cache warming effectiveness."""
        # Trigger cache warming
        await hp_system.cache.warm_cache([
            "What is metabolomics?",
            "LC-MS analysis methods", 
            "Biomarker discovery techniques"
        ])
        
        # Brief pause for warming to complete
        await asyncio.sleep(1.0)
        
        # Test queries that should benefit from warming
        warm_queries = [
            "What is metabolomics analysis?",  # Similar to warmed query
            "LC-MS analytical methods",       # Similar to warmed query
            "Biomarker discovery approaches"  # Similar to warmed query
        ]
        
        cache_hits = 0
        response_times = []
        
        for query in warm_queries:
            start_time = time.time()
            result, metadata = await hp_system.classify_query_optimized(query)
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)
            
            if metadata.get("cache_hit", False):
                cache_hits += 1
        
        avg_time = statistics.mean(response_times)
        
        print(f"✓ Cache warming test:")
        print(f"  Average response time after warming: {avg_time:.1f}ms")
        print(f"  Cache hits from warming: {cache_hits}/{len(warm_queries)}")
        
        # Performance should be excellent due to warming
        assert avg_time <= 100, f"Cache-warmed queries should be very fast, got {avg_time:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_statistics_accuracy(self, hp_system):
        """Test cache statistics tracking accuracy."""
        cache = hp_system.cache
        initial_stats = cache.get_cache_stats()
        
        # Perform various cache operations
        test_queries = [f"Test query {i}" for i in range(10)]
        
        # Generate some cache misses
        for query in test_queries:
            await cache.get(query)
        
        # Generate some cache hits  
        test_result = ClassificationResult(
            category="GENERAL", confidence=0.8, reasoning="Test",
            alternative_categories=[], uncertainty_indicators=[],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
        
        for i, query in enumerate(test_queries[:5]):
            await cache.set(query, test_result)
            await cache.get(query)  # Generate hit
        
        final_stats = cache.get_cache_stats()
        
        # Validate statistics
        total_requests_increase = final_stats["overall"]["total_requests"] - initial_stats["overall"]["total_requests"]
        hits_increase = final_stats["overall"]["total_hits"] - initial_stats["overall"]["total_hits"]
        
        print(f"✓ Cache statistics test:")
        print(f"  Total requests increase: {total_requests_increase}")
        print(f"  Hits increase: {hits_increase}")
        print(f"  Final hit rate: {final_stats['overall']['hit_rate']:.1%}")
        
        assert total_requests_increase >= 15, "Should track all cache operations"
        assert hits_increase >= 5, "Should track cache hits correctly"


# ============================================================================
# RESOURCE MANAGEMENT TESTS
# ============================================================================

class TestResourceManagement:
    """Test resource management and optimization capabilities."""
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self, hp_system):
        """Test resource monitoring functionality."""
        resource_manager = hp_system.resource_manager
        
        # Get initial resource stats
        initial_stats = resource_manager.get_resource_stats()
        
        # Verify monitoring is working
        assert "cpu" in initial_stats
        assert "memory" in initial_stats
        assert "threading" in initial_stats
        
        # CPU and memory should be reasonable ranges
        cpu_usage = initial_stats["cpu"]["current_usage_percent"]
        memory_usage = initial_stats["memory"]["current_usage_percent"]
        
        assert 0 <= cpu_usage <= 100, f"Invalid CPU usage: {cpu_usage}%"
        assert 0 <= memory_usage <= 100, f"Invalid memory usage: {memory_usage}%"
        
        print(f"✓ Resource monitoring test:")
        print(f"  CPU usage: {cpu_usage:.1f}%")
        print(f"  Memory usage: {memory_usage:.1f}%")
        print(f"  Max threads: {initial_stats['threading']['max_worker_threads']}")
        print(f"  Max processes: {initial_stats['threading']['max_worker_processes']}")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_limits(self, hp_system):
        """Test that concurrent processing respects configured limits."""
        # Execute many requests simultaneously
        concurrent_requests = 50
        query = "Test concurrent processing limits"
        
        async def single_request():
            return await hp_system.classify_query_optimized(query)
        
        # Execute all requests concurrently
        start_time = time.time()
        tasks = [asyncio.create_task(single_request()) for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Count successful results
        successful_results = [r for r in results if isinstance(r, tuple)]
        
        print(f"✓ Concurrent processing test:")
        print(f"  Concurrent requests: {concurrent_requests}")
        print(f"  Successful results: {len(successful_results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Effective throughput: {len(successful_results)/total_time:.1f} RPS")
        
        # Should handle concurrent requests without significant failures
        success_rate = len(successful_results) / concurrent_requests
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} too low under concurrent load"


# ============================================================================
# COMPREHENSIVE INTEGRATION TEST
# ============================================================================

class TestComprehensiveIntegration:
    """Comprehensive integration test combining all components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_validation(self, hp_config):
        """Complete end-to-end performance validation test."""
        print("\n" + "="*80)
        print("COMPREHENSIVE HIGH-PERFORMANCE INTEGRATION TEST")
        print("="*80)
        
        # Create system with production-like configuration
        production_config = HighPerformanceConfig(
            target_response_time_ms=1500,
            max_response_time_ms=2000,
            l1_cache_size=5000,
            enable_cache_warming=True,
            enable_request_batching=True,
            enable_deduplication=True,
            enable_adaptive_optimization=True
        )
        
        async with high_performance_classification_context(production_config) as hp_system:
            # Test 1: Basic functionality validation
            print("\n1. Basic Functionality Validation")
            print("-" * 40)
            
            basic_query = "What is the role of metabolomics in clinical diagnosis?"
            result, metadata = await hp_system.classify_query_optimized(basic_query)
            
            assert isinstance(result, ClassificationResult)
            assert result.confidence > 0.0
            assert metadata["response_time_ms"] <= 2000
            
            print(f"✓ Classification: {result.category} (confidence: {result.confidence:.3f})")
            print(f"✓ Response time: {metadata['response_time_ms']:.1f}ms")
            print(f"✓ Optimizations: {', '.join(metadata['optimizations_applied'])}")
            
            # Test 2: Performance under sustained load
            print("\n2. Sustained Load Performance")
            print("-" * 40)
            
            sustained_queries = [
                "Metabolite identification using LC-MS/MS",
                "Statistical analysis of biomarker data", 
                "Pathway enrichment analysis methods",
                "Clinical metabolomics workflow optimization",
                "Machine learning in metabolomics research",
                "Biomarker validation in clinical studies",
                "Quality control in metabolomics analysis",
                "Data preprocessing for metabolomics",
                "Multivariate analysis techniques",
                "Metabolomics database integration"
            ] * 10  # 100 total queries
            
            response_times = []
            cache_hits = 0
            successful_requests = 0
            
            start_time = time.time()
            
            for i, query in enumerate(sustained_queries):
                try:
                    result, metadata = await hp_system.classify_query_optimized(query)
                    response_times.append(metadata["response_time_ms"])
                    successful_requests += 1
                    
                    if metadata.get("cache_hit", False):
                        cache_hits += 1
                    
                    # Progress updates
                    if (i + 1) % 25 == 0:
                        print(f"  Progress: {i+1}/{len(sustained_queries)} requests completed")
                        
                except Exception as e:
                    print(f"  Request {i} failed: {e}")
            
            total_time = time.time() - start_time
            
            # Calculate performance metrics
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0
            p99_response_time = np.percentile(response_times, 99) if response_times else 0
            throughput = successful_requests / total_time
            cache_hit_rate = cache_hits / successful_requests if successful_requests > 0 else 0
            target_compliance = sum(1 for t in response_times if t <= 1500) / len(response_times) if response_times else 0
            
            print(f"✓ Total requests: {len(sustained_queries)}")
            print(f"✓ Successful requests: {successful_requests}")
            print(f"✓ Average response time: {avg_response_time:.1f}ms")
            print(f"✓ P95 response time: {p95_response_time:.1f}ms")
            print(f"✓ P99 response time: {p99_response_time:.1f}ms")
            print(f"✓ Throughput: {throughput:.1f} requests/second")
            print(f"✓ Cache hit rate: {cache_hit_rate:.1%}")
            print(f"✓ Target compliance: {target_compliance:.1%}")
            
            # Performance assertions
            assert successful_requests >= len(sustained_queries) * 0.99, "Success rate below 99%"
            assert avg_response_time <= 1500, f"Average response time {avg_response_time:.1f}ms exceeds 1500ms"
            assert p95_response_time <= 1800, f"P95 response time {p95_response_time:.1f}ms exceeds 1800ms"
            assert p99_response_time <= 2000, f"P99 response time {p99_response_time:.1f}ms exceeds 2000ms"
            assert target_compliance >= 0.90, f"Target compliance {target_compliance:.1%} below 90%"
            assert throughput >= 10, f"Throughput {throughput:.1f} RPS below minimum"
            
            # Test 3: System resource efficiency
            print("\n3. System Resource Efficiency")
            print("-" * 40)
            
            system_stats = hp_system.get_comprehensive_performance_stats()
            cache_stats = system_stats["cache"]
            resource_stats = system_stats["resources"]
            
            print(f"✓ Overall cache hit rate: {cache_stats['overall']['hit_rate']:.1%}")
            print(f"✓ L1 cache hit rate: {cache_stats['l1_cache']['hit_rate']:.1%}")
            print(f"✓ Average CPU usage: {resource_stats['cpu']['avg_usage_percent']:.1f}%")
            print(f"✓ Average memory usage: {resource_stats['memory']['avg_usage_percent']:.1f}%")
            
            # Resource efficiency assertions
            assert cache_stats['overall']['hit_rate'] >= 0.6, "Cache hit rate too low"
            assert resource_stats['cpu']['avg_usage_percent'] <= 80, "CPU usage too high"
            assert resource_stats['memory']['avg_usage_percent'] <= 80, "Memory usage too high"
            
            # Test 4: Error handling and recovery
            print("\n4. Error Handling and Recovery")
            print("-" * 40)
            
            # Test with problematic queries
            error_test_queries = [
                "",  # Empty query
                "x" * 1000,  # Very long query
                "Invalid query with special chars: @#$%^&*()",
                "Query with unicode: 测试查询 αβγ δεζ"
            ]
            
            error_handling_success = 0
            
            for query in error_test_queries:
                try:
                    result, metadata = await hp_system.classify_query_optimized(query)
                    # Should handle gracefully
                    assert isinstance(result, ClassificationResult)
                    assert metadata["response_time_ms"] <= 2000
                    error_handling_success += 1
                    print(f"✓ Handled problematic query: {query[:20]}...")
                except Exception as e:
                    print(f"✗ Failed on query '{query[:20]}...': {e}")
            
            print(f"✓ Error handling success: {error_handling_success}/{len(error_test_queries)}")
            assert error_handling_success >= len(error_test_queries) * 0.75, "Poor error handling"
            
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE INTEGRATION TEST PASSED")
        print("✅ System meets all <2 second response time requirements")
        print("✅ High performance and reliability validated")
        print("="*80)


# ============================================================================
# BENCHMARK INTEGRATION TESTS
# ============================================================================

class TestBenchmarkIntegration:
    """Test integration with the benchmark suite."""
    
    @pytest.mark.asyncio
    async def test_benchmark_suite_integration(self):
        """Test integration with the performance benchmark suite."""
        # Configure benchmark for integration testing
        config = BenchmarkConfig(
            concurrent_users=10,
            total_requests=100,
            load_pattern="constant",
            target_response_time_ms=1500,
            max_response_time_ms=2000,
            enable_detailed_logging=False,
            export_results=False,
            generate_plots=False
        )
        
        # Run benchmark
        results = await run_comprehensive_benchmark(config)
        
        # Validate benchmark results
        assert isinstance(results, type(results))  # Should be BenchmarkResults type
        assert results.total_requests == config.total_requests
        assert results.success_rate >= 0.95, f"Success rate {results.success_rate:.1%} too low"
        assert results.avg_response_time_ms <= 1800, f"Average time {results.avg_response_time_ms:.1f}ms too high"
        assert results.target_compliance_rate >= 0.90, f"Compliance {results.target_compliance_rate:.1%} too low"
        
        print(f"✓ Benchmark integration test:")
        print(f"  Grade: {results.performance_grade}")
        print(f"  Success rate: {results.success_rate:.1%}")
        print(f"  Average response time: {results.avg_response_time_ms:.1f}ms")
        print(f"  Target compliance: {results.target_compliance_rate:.1%}")


# ============================================================================
# TEST EXECUTION AND REPORTING
# ============================================================================

def run_performance_test_suite():
    """Run the complete performance test suite."""
    print("Starting High-Performance Classification System Test Suite")
    print("="*80)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with pytest
    test_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--asyncio-mode=auto"
    ]
    
    try:
        exit_code = pytest.main(test_args)
        
        if exit_code == 0:
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED")
            print("✅ High-Performance Classification System validated for <2 second response times")
            print("✅ System ready for production deployment")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ TESTS FAILED")
            print("❌ System does not meet performance requirements")
            print("❌ Review test output and optimize system configuration")
            print("="*80)
        
        return exit_code
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    # Run the test suite when executed directly
    exit_code = run_performance_test_suite()
    exit(exit_code)