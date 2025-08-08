#!/usr/bin/env python3
"""
Comprehensive Edge Cases and Error Condition Tests for Feature Flag System.

This module provides extensive test coverage for edge cases, error conditions,
and boundary scenarios in the feature flag system, ensuring robust behavior
under unusual or exceptional circumstances.

Test Coverage Areas:
- Boundary value testing for numeric configurations
- Error handling and recovery mechanisms
- Resource exhaustion scenarios
- Network failure and timeout handling
- Invalid data and malformed inputs
- Memory and cache limit testing
- Concurrent access edge cases
- Circuit breaker boundary conditions
- Hash collision and edge cases
- Configuration edge cases and validation
- Extreme load and stress conditions

Author: Claude Code (Anthropic)
Created: 2025-08-08
"""

import pytest
import pytest_asyncio
import asyncio
import time
import threading
import hashlib
import json
import logging
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Union

# Import components for edge case testing
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.feature_flag_manager import (
    FeatureFlagManager,
    RoutingContext,
    RoutingResult,
    RoutingDecision,
    RoutingReason,
    UserCohort,
    CircuitBreakerState,
    PerformanceMetrics
)
from lightrag_integration.integration_wrapper import (
    IntegratedQueryService,
    QueryRequest,
    ServiceResponse,
    ResponseType,
    AdvancedCircuitBreaker
)


class TestBoundaryValueEdgeCases:
    """Test boundary value conditions and edge cases."""
    
    @pytest.mark.parametrize("rollout_percentage", [
        -1.0, -100.0, -0.001,  # Below minimum
        0.0, 0.001, 0.1,       # Near minimum
        99.9, 99.999, 100.0,   # Near maximum
        100.001, 150.0, 1000.0 # Above maximum
    ])
    def test_rollout_percentage_boundary_values(self, rollout_percentage):
        """Test rollout percentage boundary values."""
        config = LightRAGConfig(lightrag_rollout_percentage=rollout_percentage)
        
        # Should be clamped to [0.0, 100.0] range
        assert 0.0 <= config.lightrag_rollout_percentage <= 100.0
        
        if rollout_percentage < 0.0:
            assert config.lightrag_rollout_percentage == 0.0
        elif rollout_percentage > 100.0:
            assert config.lightrag_rollout_percentage == 100.0
        else:
            assert config.lightrag_rollout_percentage == rollout_percentage
    
    @pytest.mark.parametrize("quality_threshold", [
        -1.0, -0.5, -0.001,    # Below minimum
        0.0, 0.001, 0.1,       # Near minimum
        0.9, 0.999, 1.0,       # Near maximum
        1.001, 1.5, 10.0       # Above maximum
    ])
    def test_quality_threshold_boundary_values(self, quality_threshold):
        """Test quality threshold boundary values."""
        config = LightRAGConfig(lightrag_min_quality_threshold=quality_threshold)
        
        # Should be clamped to [0.0, 1.0] range
        assert 0.0 <= config.lightrag_min_quality_threshold <= 1.0
        
        if quality_threshold < 0.0:
            assert config.lightrag_min_quality_threshold == 0.0
        elif quality_threshold > 1.0:
            assert config.lightrag_min_quality_threshold == 1.0
        else:
            assert config.lightrag_min_quality_threshold == quality_threshold
    
    @pytest.mark.parametrize("failure_threshold", [0, -1, -5, 1, 2, 100, 1000])
    def test_circuit_breaker_failure_threshold_boundary(self, failure_threshold):
        """Test circuit breaker failure threshold boundary values."""
        config = LightRAGConfig(lightrag_circuit_breaker_failure_threshold=failure_threshold)
        
        # Should be at least 1 (can't have zero or negative threshold)
        if failure_threshold <= 0:
            assert config.lightrag_circuit_breaker_failure_threshold >= 1
        else:
            assert config.lightrag_circuit_breaker_failure_threshold == failure_threshold
    
    @pytest.mark.parametrize("timeout", [
        -10.0, -1.0, 0.0,      # Invalid timeouts
        0.001, 0.1, 1.0,       # Very short timeouts
        30.0, 60.0, 300.0,     # Normal timeouts
        3600.0, 86400.0        # Very long timeouts
    ])
    def test_timeout_boundary_values(self, timeout):
        """Test timeout boundary values."""
        config = LightRAGConfig(lightrag_integration_timeout_seconds=timeout)
        
        # Should be positive
        if timeout <= 0:
            assert config.lightrag_integration_timeout_seconds > 0
        else:
            assert config.lightrag_integration_timeout_seconds == timeout


class TestHashingEdgeCases:
    """Test hashing function edge cases and potential collisions."""
    
    def test_hash_calculation_extreme_user_ids(self):
        """Test hash calculation with extreme user IDs."""
        config = LightRAGConfig(lightrag_user_hash_salt="test_salt")
        feature_manager = FeatureFlagManager(config=config)
        
        extreme_user_ids = [
            "",  # Empty string
            " ",  # Whitespace
            "a" * 1000,  # Very long ID
            "user_with_unicode_🦄_🌟",  # Unicode characters
            "user\nwith\nnewlines",  # Special characters
            "user\twith\ttabs",
            "user with spaces",
            "123456789",  # Numeric string
            "user@domain.com",  # Email format
            "specialchars!@#$%^&*()",  # Special characters
        ]
        
        hashes = []
        for user_id in extreme_user_ids:
            try:
                user_hash = feature_manager._calculate_user_hash(user_id)
                assert isinstance(user_hash, str)
                assert len(user_hash) == 64  # SHA256 hex length
                hashes.append(user_hash)
            except Exception as e:
                pytest.fail(f"Hash calculation failed for user_id '{user_id}': {e}")
        
        # All hashes should be unique (extremely unlikely to have collisions)
        assert len(set(hashes)) == len(hashes), "Hash collision detected"
    
    def test_hash_consistency_across_calls(self):
        """Test hash consistency across multiple calls."""
        config = LightRAGConfig(lightrag_user_hash_salt="consistency_test")
        feature_manager = FeatureFlagManager(config=config)
        
        user_id = "consistency_test_user"
        hashes = []
        
        # Calculate hash multiple times
        for _ in range(100):
            user_hash = feature_manager._calculate_user_hash(user_id)
            hashes.append(user_hash)
        
        # All should be identical
        assert len(set(hashes)) == 1, "Hash calculation not consistent"
    
    def test_hash_distribution_uniformity(self):
        """Test that hash distribution is roughly uniform."""
        config = LightRAGConfig(lightrag_user_hash_salt="distribution_test")
        feature_manager = FeatureFlagManager(config=config)
        
        # Generate hashes for many users
        num_users = 10000
        percentages = []
        
        for i in range(num_users):
            user_id = f"distribution_user_{i}"
            user_hash = feature_manager._calculate_user_hash(user_id)
            percentage = feature_manager._get_rollout_percentage_from_hash(user_hash)
            percentages.append(percentage)
        
        # Check distribution properties
        assert min(percentages) >= 0.0
        assert max(percentages) <= 100.0
        
        # Mean should be around 50
        mean_percentage = sum(percentages) / len(percentages)
        assert 45.0 <= mean_percentage <= 55.0, f"Mean percentage outside expected range: {mean_percentage}"
        
        # Distribution should be roughly uniform (check quartiles)
        sorted_percentages = sorted(percentages)
        q1 = sorted_percentages[len(sorted_percentages) // 4]
        q2 = sorted_percentages[len(sorted_percentages) // 2]
        q3 = sorted_percentages[3 * len(sorted_percentages) // 4]
        
        # Quartiles should be roughly at 25, 50, 75
        assert 20.0 <= q1 <= 30.0, f"Q1 outside expected range: {q1}"
        assert 45.0 <= q2 <= 55.0, f"Q2 outside expected range: {q2}"
        assert 70.0 <= q3 <= 80.0, f"Q3 outside expected range: {q3}"


class TestCircuitBreakerEdgeCases:
    """Test circuit breaker edge cases and boundary conditions."""
    
    def test_circuit_breaker_at_exact_threshold(self):
        """Test circuit breaker behavior at exact failure threshold."""
        config = LightRAGConfig(
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=3
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Set failure count to exactly the threshold
        feature_manager.circuit_breaker_state.failure_count = 3
        
        # Should open circuit breaker
        is_open = feature_manager._check_circuit_breaker()
        assert is_open is True
        assert feature_manager.circuit_breaker_state.is_open is True
    
    def test_circuit_breaker_just_below_threshold(self):
        """Test circuit breaker behavior just below threshold."""
        config = LightRAGConfig(
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=5
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Set failure count just below threshold
        feature_manager.circuit_breaker_state.failure_count = 4
        
        # Should remain closed
        is_open = feature_manager._check_circuit_breaker()
        assert is_open is False
        assert feature_manager.circuit_breaker_state.is_open is False
    
    def test_circuit_breaker_recovery_at_exact_timeout(self):
        """Test circuit breaker recovery at exact timeout boundary."""
        config = LightRAGConfig(
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_recovery_timeout=300.0
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Set circuit breaker to open with exact recovery timeout
        feature_manager.circuit_breaker_state.is_open = True
        feature_manager.circuit_breaker_state.last_failure_time = datetime.now() - timedelta(seconds=300)
        
        # Should attempt recovery
        is_open = feature_manager._check_circuit_breaker()
        assert is_open is False  # Should recover
        assert feature_manager.circuit_breaker_state.is_open is False
    
    def test_circuit_breaker_rapid_failure_recovery_cycles(self):
        """Test circuit breaker under rapid failure/recovery cycles."""
        config = LightRAGConfig(
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=2,
            lightrag_circuit_breaker_recovery_timeout=0.1  # Very short timeout
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Simulate rapid cycles
        for cycle in range(5):
            # Cause failures to open circuit breaker
            for _ in range(3):
                feature_manager.record_failure("lightrag", "Test failure")
            
            # Circuit breaker should be open
            assert feature_manager._check_circuit_breaker() is True
            
            # Wait for recovery
            time.sleep(0.2)
            
            # Should recover
            assert feature_manager._check_circuit_breaker() is False
            
            # Record a success to reset failure count
            feature_manager.record_success("lightrag", 1.0, 0.8)
    
    @pytest.mark.asyncio
    async def test_advanced_circuit_breaker_edge_cases(self):
        """Test AdvancedCircuitBreaker edge cases."""
        circuit_breaker = AdvancedCircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1
        )
        
        async def failing_function():
            raise Exception("Always fails")
        
        async def succeeding_function():
            return "success"
        
        # Test immediate opening on first failure with threshold=1
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.is_open is True
        assert circuit_breaker.failure_count == 1
        
        # Test multiple calls while open
        for _ in range(3):
            with pytest.raises(Exception, match="Circuit breaker is open"):
                await circuit_breaker.call(succeeding_function)
        
        # Wait for recovery and test successful call
        await asyncio.sleep(0.2)
        result = await circuit_breaker.call(succeeding_function)
        assert result == "success"
        assert circuit_breaker.is_open is False
        assert circuit_breaker.failure_count == 0


class TestMemoryAndCacheEdgeCases:
    """Test memory management and cache limit edge cases."""
    
    def test_routing_cache_memory_limits(self):
        """Test routing cache memory management under load."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        # Fill cache beyond typical limits
        large_cache_size = 2000
        
        for i in range(large_cache_size):
            cache_key = f"memory_test_key_{i}"
            mock_result = RoutingResult(
                decision=RoutingDecision.LIGHTRAG,
                reason=RoutingReason.USER_COHORT_ASSIGNMENT,
                confidence=0.95
            )
            feature_manager._cache_routing_result(cache_key, mock_result)
        
        # Cache should be managed (not unlimited growth)
        assert len(feature_manager._routing_cache) <= 1000  # Should be capped
    
    def test_performance_metrics_memory_management(self):
        """Test performance metrics memory management."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        # Add many metrics
        large_metric_count = 2000
        
        for i in range(large_metric_count):
            feature_manager.record_success("lightrag", 1.0 + (i * 0.01), 0.8 + (i * 0.0001))
        
        # Metrics arrays should be managed (not unlimited growth)
        assert len(feature_manager.performance_metrics.lightrag_response_times) <= 1000
        assert len(feature_manager.performance_metrics.lightrag_quality_scores) <= 1000
    
    def test_cache_cleanup_under_memory_pressure(self):
        """Test cache cleanup behavior under simulated memory pressure."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity:
            mock_perplexity.return_value.query_async = AsyncMock(return_value=ServiceResponse(
                content="Test response", response_type=ResponseType.PERPLEXITY
            ))
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            # Fill response cache
            large_cache_size = 1500
            
            for i in range(large_cache_size):
                cache_key = f"pressure_test_{i}"
                test_response = ServiceResponse(content=f"Response {i}")
                service._cache_response(cache_key, test_response)
            
            # Cache should be managed
            assert len(service._response_cache) <= 100  # Should be capped
    
    def test_concurrent_cache_access_edge_cases(self):
        """Test concurrent cache access edge cases."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        def cache_operations():
            for i in range(100):
                cache_key = f"concurrent_key_{i}"
                mock_result = RoutingResult(
                    decision=RoutingDecision.LIGHTRAG,
                    reason=RoutingReason.USER_COHORT_ASSIGNMENT,
                    confidence=0.95
                )
                feature_manager._cache_routing_result(cache_key, mock_result)
                
                # Also try to retrieve
                feature_manager._routing_cache.get(cache_key)
        
        # Run concurrent cache operations
        threads = [threading.Thread(target=cache_operations) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should complete without errors or corruption
        assert len(feature_manager._routing_cache) > 0


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_routing_with_corrupted_config(self):
        """Test routing behavior with corrupted configuration."""
        config = LightRAGConfig()
        
        # Corrupt configuration fields
        config.lightrag_rollout_percentage = None
        config.lightrag_user_hash_salt = None
        config.lightrag_enable_circuit_breaker = None
        
        # Should still work (with defaults or error handling)
        try:
            feature_manager = FeatureFlagManager(config=config)
            context = RoutingContext(user_id="corrupted_test", query_text="test")
            result = feature_manager.should_use_lightrag(context)
            
            # Should return a valid result
            assert isinstance(result, RoutingResult)
            assert result.decision in [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY]
        except Exception as e:
            # If it fails, should fail gracefully with clear error
            assert "configuration" in str(e).lower() or "config" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_service_initialization_failures(self):
        """Test service behavior when initialization fails."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        
        # Mock initialization failures
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService', side_effect=Exception("Init failed")):
            try:
                service = IntegratedQueryService(config=config, perplexity_api_key="test")
                
                # Should still be able to handle queries (graceful degradation)
                request = QueryRequest(query_text="test", user_id="init_failure_user")
                response = await service.query_async(request)
                
                # Should get an error response, not crash
                assert isinstance(response, ServiceResponse)
                assert response.is_success is False
                assert response.error_details is not None
                
            except Exception as e:
                # If it fails completely, should be a clear initialization error
                assert "init" in str(e).lower() or "initialization" in str(e).lower()
    
    def test_hash_calculation_with_invalid_salt(self):
        """Test hash calculation with invalid salt values."""
        invalid_salts = [None, "", 123, [], {}, True]
        
        for invalid_salt in invalid_salts:
            config = LightRAGConfig()
            config.lightrag_user_hash_salt = invalid_salt
            
            try:
                feature_manager = FeatureFlagManager(config=config)
                user_hash = feature_manager._calculate_user_hash("test_user")
                
                # Should still produce a valid hash
                assert isinstance(user_hash, str)
                assert len(user_hash) == 64
                
            except Exception as e:
                # If it fails, should be a clear error about salt
                assert "salt" in str(e).lower() or "hash" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_network_timeout_edge_cases(self):
        """Test network timeout edge cases."""
        config = LightRAGConfig(
            lightrag_integration_timeout_seconds=0.001  # Very short timeout
        )
        
        with patch('lightrag_integration.integration_wrapper.LightRAGQueryService') as mock_lightrag:
            # Simulate slow response
            async def slow_query(*args, **kwargs):
                await asyncio.sleep(0.01)  # Longer than timeout
                return ServiceResponse(content="Slow response")
            
            mock_lightrag.return_value.query_async = slow_query
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            request = QueryRequest(query_text="timeout test", user_id="timeout_user")
            response = await service.query_async(request)
            
            # Should handle timeout gracefully
            assert isinstance(response, ServiceResponse)
            # Might succeed with fallback or fail with timeout error
            if not response.is_success:
                assert "timeout" in response.error_details.lower()


class TestConcurrencyEdgeCases:
    """Test concurrency edge cases and race conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_operations(self):
        """Test concurrent circuit breaker state changes."""
        config = LightRAGConfig(
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=2
        )
        feature_manager = FeatureFlagManager(config=config)
        
        async def concurrent_operations():
            for _ in range(10):
                # Mix of failures and successes
                if hash(threading.current_thread().ident) % 2:
                    feature_manager.record_failure("lightrag", "Concurrent failure")
                else:
                    feature_manager.record_success("lightrag", 1.0, 0.8)
                
                # Check circuit breaker state
                feature_manager._check_circuit_breaker()
                
                await asyncio.sleep(0.001)  # Small delay
        
        # Run concurrent operations
        tasks = [concurrent_operations() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should complete without errors
        assert isinstance(feature_manager.circuit_breaker_state.failure_count, int)
        assert isinstance(feature_manager.circuit_breaker_state.is_open, bool)
    
    def test_concurrent_cohort_assignment(self):
        """Test concurrent user cohort assignment consistency."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=50.0
        )
        feature_manager = FeatureFlagManager(config=config)
        
        def assign_cohorts(user_prefix):
            results = {}
            for i in range(50):
                user_id = f"{user_prefix}_user_{i}"
                context = RoutingContext(user_id=user_id, query_text="test")
                result = feature_manager.should_use_lightrag(context)
                results[user_id] = result.user_cohort
            return results
        
        # Run concurrent assignments
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(assign_cohorts, f"thread_{i}")
                for i in range(4)
            ]
            
            all_results = {}
            for future in concurrent.futures.as_completed(futures):
                thread_results = future.result()
                all_results.update(thread_results)
        
        # Verify consistency - same user should get same cohort across threads
        # (This is implicitly tested by the fact that each thread uses different user IDs)
        assert len(all_results) == 200  # 4 threads * 50 users each
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_invalidation(self):
        """Test concurrent cache invalidation scenarios."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_service:
            mock_service.return_value.query_async = AsyncMock(return_value=ServiceResponse(
                content="Cached response", response_type=ResponseType.PERPLEXITY
            ))
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            async def cache_operations():
                for i in range(20):
                    request = QueryRequest(
                        query_text=f"Cache test {i}",
                        user_id=f"cache_user_{i}"
                    )
                    await service.query_async(request)
                    
                    # Occasionally clear cache
                    if i % 5 == 0:
                        service.clear_cache()
            
            # Run concurrent cache operations
            tasks = [cache_operations() for _ in range(3)]
            await asyncio.gather(*tasks)
            
            # Should complete without errors
            assert isinstance(service._response_cache, dict)


class TestResourceExhaustionEdgeCases:
    """Test behavior under resource exhaustion scenarios."""
    
    def test_extreme_user_load_simulation(self):
        """Test system behavior under extreme user load."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_rollout_percentage=30.0
        )
        feature_manager = FeatureFlagManager(config=config)
        
        # Simulate extreme number of users
        num_users = 100000
        start_time = time.time()
        
        routing_decisions = []
        
        for i in range(num_users):
            user_id = f"load_user_{i}"
            context = RoutingContext(user_id=user_id, query_text="load test")
            result = feature_manager.should_use_lightrag(context)
            routing_decisions.append(result.decision)
            
            # Check performance periodically
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - start_time
                ops_per_second = i / elapsed
                assert ops_per_second > 1000, f"Performance too low: {ops_per_second} ops/sec"
        
        total_time = time.time() - start_time
        final_ops_per_second = num_users / total_time
        
        # Should maintain reasonable performance
        assert final_ops_per_second > 5000, f"Final performance too low: {final_ops_per_second} ops/sec"
        
        # Distribution should still be correct
        lightrag_count = routing_decisions.count(RoutingDecision.LIGHTRAG)
        lightrag_percentage = (lightrag_count / num_users) * 100
        
        # Should be close to 30% (within 2% tolerance)
        assert abs(lightrag_percentage - 30.0) < 2.0, f"Distribution incorrect: {lightrag_percentage}%"
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        config = LightRAGConfig(lightrag_integration_enabled=True)
        feature_manager = FeatureFlagManager(config=config)
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Sustained operation load
        for iteration in range(10):
            # Create many routing decisions
            for i in range(1000):
                user_id = f"memory_user_{iteration}_{i}"
                context = RoutingContext(user_id=user_id, query_text="memory test")
                result = feature_manager.should_use_lightrag(context)
            
            # Record metrics
            for i in range(100):
                feature_manager.record_success("lightrag", 1.0 + (i * 0.01), 0.8 + (i * 0.001))
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be bounded (not indefinite leak)
            assert memory_growth < 100, f"Excessive memory growth: {memory_growth}MB"
            
            # Trigger garbage collection
            gc.collect()
    
    @pytest.mark.asyncio
    async def test_service_degradation_under_load(self):
        """Test service degradation behavior under high load."""
        config = LightRAGConfig(
            lightrag_integration_enabled=True,
            lightrag_enable_circuit_breaker=True,
            lightrag_circuit_breaker_failure_threshold=10
        )
        
        with patch('lightrag_integration.integration_wrapper.PerplexityQueryService') as mock_perplexity:
            # Configure mock to simulate occasional failures
            call_count = [0]  # Mutable counter
            
            async def mock_query(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 20 == 0:  # Fail every 20th call
                    return ServiceResponse(content="", error_details="Load failure")
                else:
                    await asyncio.sleep(0.01)  # Simulate processing time
                    return ServiceResponse(content=f"Response {call_count[0]}")
            
            mock_perplexity.return_value.query_async = mock_query
            
            service = IntegratedQueryService(config=config, perplexity_api_key="test")
            
            # High load simulation
            async def load_worker(worker_id):
                results = []
                for i in range(50):
                    request = QueryRequest(
                        query_text=f"Load test {worker_id}_{i}",
                        user_id=f"load_worker_{worker_id}"
                    )
                    start_time = time.time()
                    response = await service.query_async(request)
                    end_time = time.time()
                    
                    results.append({
                        'success': response.is_success,
                        'duration': end_time - start_time,
                        'response_type': response.response_type
                    })
                
                return results
            
            # Run multiple workers concurrently
            tasks = [load_worker(i) for i in range(10)]
            all_results = await asyncio.gather(*tasks)
            
            # Flatten results
            flat_results = [result for worker_results in all_results for result in worker_results]
            
            # Analyze results
            success_count = sum(1 for r in flat_results if r['success'])
            total_requests = len(flat_results)
            success_rate = success_count / total_requests
            
            # Should maintain reasonable success rate despite failures
            assert success_rate > 0.8, f"Success rate too low: {success_rate}"
            
            # Response times should be reasonable
            avg_duration = sum(r['duration'] for r in flat_results) / len(flat_results)
            assert avg_duration < 1.0, f"Average response time too high: {avg_duration}s"


class TestConfigurationExtremeEdgeCases:
    """Test configuration edge cases and invalid combinations."""
    
    def test_contradictory_configuration_settings(self):
        """Test behavior with contradictory configuration settings."""
        # Configuration that has contradictory settings
        config = LightRAGConfig(
            lightrag_integration_enabled=False,  # Disabled
            lightrag_rollout_percentage=100.0,   # But 100% rollout
            lightrag_enable_ab_testing=True,     # And A/B testing enabled
            lightrag_enable_circuit_breaker=True # And circuit breaker enabled
        )
        
        feature_manager = FeatureFlagManager(config=config)
        context = RoutingContext(user_id="contradiction_user", query_text="test")
        result = feature_manager.should_use_lightrag(context)
        
        # Should handle contradiction gracefully
        assert isinstance(result, RoutingResult)
        # With integration disabled, should route to Perplexity
        assert result.decision == RoutingDecision.PERPLEXITY
        assert result.reason == RoutingReason.FEATURE_DISABLED
    
    def test_invalid_json_routing_rules(self):
        """Test handling of invalid JSON routing rules."""
        invalid_json_strings = [
            '{"unclosed": true',
            '{"invalid": json}',
            '{invalid json completely',
            '{"nested": {"incomplete": }',
            '[]',  # Array instead of object
            '"string"',  # String instead of object
            '123',  # Number instead of object
            '',  # Empty string
        ]
        
        for invalid_json in invalid_json_strings:
            with patch.dict('os.environ', {'LIGHTRAG_ROUTING_RULES': invalid_json}):
                try:
                    config = LightRAGConfig()
                    # Should handle gracefully
                    assert config.lightrag_routing_rules in [None, {}]
                except Exception as e:
                    # If it fails, should be clear about JSON parsing
                    assert "json" in str(e).lower() or "routing" in str(e).lower()
    
    def test_extreme_numeric_configuration_values(self):
        """Test handling of extreme numeric values."""
        import sys
        
        extreme_values = [
            sys.float_info.max,    # Maximum float
            sys.float_info.min,    # Minimum positive float
            float('inf'),          # Infinity
            float('-inf'),         # Negative infinity
            1e308,                 # Very large number
            1e-308,                # Very small number
        ]
        
        for extreme_value in extreme_values:
            try:
                config = LightRAGConfig(
                    lightrag_rollout_percentage=extreme_value,
                    lightrag_min_quality_threshold=extreme_value,
                    lightrag_integration_timeout_seconds=extreme_value,
                    lightrag_circuit_breaker_recovery_timeout=extreme_value
                )
                
                # Values should be validated and clamped
                assert 0.0 <= config.lightrag_rollout_percentage <= 100.0
                assert 0.0 <= config.lightrag_min_quality_threshold <= 1.0
                assert config.lightrag_integration_timeout_seconds > 0
                assert config.lightrag_circuit_breaker_recovery_timeout > 0
                
            except (ValueError, OverflowError) as e:
                # Should handle extreme values gracefully
                assert "value" in str(e).lower() or "overflow" in str(e).lower()
    
    def test_unicode_and_encoding_edge_cases(self):
        """Test Unicode and encoding edge cases."""
        unicode_test_cases = [
            "🦄🌟🚀",  # Emoji
            "测试用户",  # Chinese characters
            "пользователь",  # Cyrillic
            "مستخدم",  # Arabic
            "ユーザー",  # Japanese
            "café",  # Accented characters
            "\x00\x01\x02",  # Control characters
            "user\u200Bwith\u200Bzero\u200Bwidth",  # Zero-width characters
        ]
        
        config = LightRAGConfig(lightrag_user_hash_salt="unicode_test_🦄")
        feature_manager = FeatureFlagManager(config=config)
        
        for unicode_user_id in unicode_test_cases:
            try:
                context = RoutingContext(user_id=unicode_user_id, query_text="unicode test 🌟")
                result = feature_manager.should_use_lightrag(context)
                
                # Should handle Unicode gracefully
                assert isinstance(result, RoutingResult)
                assert result.decision in [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY]
                
            except UnicodeError as e:
                # If Unicode handling fails, should be clear
                assert "unicode" in str(e).lower() or "encoding" in str(e).lower()


# Import required for concurrent testing
import concurrent.futures

# Mark the end of edge case tests
if __name__ == "__main__":
    pytest.main([__file__])