"""
Comprehensive failure scenario tests for ProductionCircuitBreaker.

This module tests various failure scenarios including API timeouts, rate limits,
service unavailability, cascading failures, budget exhaustion, memory pressure,
and network connectivity issues.
"""

import pytest
import asyncio
import time
import threading
import random
import statistics
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import concurrent.futures

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
# Failure Scenario Test Fixtures
# ============================================================================

@pytest.fixture
def timeout_prone_config():
    """Configuration for timeout-prone backend"""
    return BackendInstanceConfig(
        id="timeout_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.slow-service.com",
        api_key="test_key",
        timeout_seconds=5.0,  # Short timeout
        failure_threshold=2,  # Low threshold for timeouts
        recovery_timeout_seconds=30,
        expected_response_time_ms=1000.0
    )

@pytest.fixture
def rate_limited_config():
    """Configuration for rate-limited backend"""
    return BackendInstanceConfig(
        id="rate_limited_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.rate-limited.com",
        api_key="limited_key",
        max_requests_per_minute=10,  # Very low rate limit
        failure_threshold=3,
        recovery_timeout_seconds=120,  # Long recovery for rate limits
        expected_response_time_ms=2000.0
    )

@pytest.fixture
def unreliable_config():
    """Configuration for unreliable backend"""
    return BackendInstanceConfig(
        id="unreliable_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://unstable-service:8080",
        api_key="unreliable_key",
        failure_threshold=5,
        recovery_timeout_seconds=60,
        reliability_score=0.3,  # Very low reliability
        expected_response_time_ms=800.0
    )

@pytest.fixture
def memory_constrained_config():
    """Configuration for memory-constrained backend"""
    return BackendInstanceConfig(
        id="memory_constrained_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://memory-limited:8080",
        api_key="memory_key",
        failure_threshold=3,
        recovery_timeout_seconds=45,
        expected_response_time_ms=1500.0
    )


# ============================================================================
# API Timeout Scenario Tests
# ============================================================================

class TestAPITimeoutScenarios:
    """Test circuit breaker behavior under various timeout conditions"""

    def test_consecutive_timeouts_trigger_circuit_opening(self, timeout_prone_config):
        """Test that consecutive timeouts trigger circuit breaker opening"""
        cb = ProductionCircuitBreaker(timeout_prone_config)
        
        # Record consecutive timeouts
        for i in range(3):
            cb.record_failure(f"Request timeout {i}", response_time_ms=5000, error_type="TimeoutError")
        
        # Should open circuit due to timeout pattern
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.consecutive_timeouts >= 3

    def test_mixed_timeout_and_success_patterns(self, timeout_prone_config):
        """Test circuit breaker behavior with mixed timeout/success patterns"""
        cb = ProductionCircuitBreaker(timeout_prone_config)
        
        # Pattern: timeout, success, timeout, timeout
        cb.record_failure("Timeout 1", response_time_ms=5000, error_type="TimeoutError")
        cb.record_success(800.0)  # Quick success
        cb.record_failure("Timeout 2", response_time_ms=5000, error_type="TimeoutError")
        cb.record_failure("Timeout 3", response_time_ms=5000, error_type="TimeoutError")
        
        # Should track consecutive timeouts correctly
        assert cb.consecutive_timeouts == 2
        # But overall failure count might not trigger normal threshold
        assert cb.failure_count < cb.config.failure_threshold
        # Should still open due to timeout pattern
        assert cb.state == CircuitBreakerState.OPEN

    def test_timeout_adaptive_threshold_adjustment(self, timeout_prone_config):
        """Test adaptive threshold adjustment for timeout scenarios"""
        cb = ProductionCircuitBreaker(timeout_prone_config)
        
        # Record timeout pattern
        cb.record_failure("Timeout 1", error_type="TimeoutError")
        cb.record_failure("Timeout 2", error_type="TimeoutError")
        
        # Adaptive threshold should be lowered
        original_threshold = cb.config.failure_threshold
        adjusted_threshold = cb._adaptive_failure_threshold
        assert adjusted_threshold < original_threshold

    def test_timeout_recovery_with_extended_timeout(self, timeout_prone_config):
        """Test that timeout-triggered openings have extended recovery times"""
        cb = ProductionCircuitBreaker(timeout_prone_config)
        
        # Trigger opening with timeouts
        for i in range(4):
            cb.record_failure(f"Timeout {i}", error_type="TimeoutError")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Recovery time should be extended for timeout patterns
        base_timeout = cb.config.recovery_timeout_seconds
        actual_timeout = (cb.next_attempt_time - datetime.now()).total_seconds()
        assert actual_timeout > base_timeout * 1.5  # Should be significantly longer

    def test_timeout_vs_other_error_recovery_times(self, timeout_prone_config):
        """Test different recovery times for timeout vs other errors"""
        # Timeout-based circuit breaker
        cb_timeout = ProductionCircuitBreaker(timeout_prone_config)
        for i in range(4):
            cb_timeout.record_failure(f"Timeout {i}", error_type="TimeoutError")
        
        # Server error-based circuit breaker
        cb_server = ProductionCircuitBreaker(timeout_prone_config)
        for i in range(4):
            cb_server.record_failure(f"Server error {i}", error_type="ServerError")
        
        # Timeout recovery should be longer
        timeout_recovery = (cb_timeout.next_attempt_time - datetime.now()).total_seconds()
        server_recovery = (cb_server.next_attempt_time - datetime.now()).total_seconds()
        assert timeout_recovery > server_recovery


# ============================================================================
# Rate Limit Scenario Tests
# ============================================================================

class TestRateLimitScenarios:
    """Test circuit breaker behavior under rate limiting conditions"""

    def test_rate_limit_error_handling(self, rate_limited_config):
        """Test handling of rate limit errors"""
        cb = ProductionCircuitBreaker(rate_limited_config)
        
        # Record rate limit errors
        cb.record_failure("Rate limit exceeded", error_type="RateLimitError")
        cb.record_failure("Quota exceeded", error_type="RateLimitError")
        
        # Should track rate limit errors
        assert cb.error_types["RateLimitError"] == 2
        
        # One more should trigger circuit opening
        cb.record_failure("Rate limit hit again", error_type="RateLimitError")
        assert cb.state == CircuitBreakerState.OPEN

    def test_rate_limit_extended_recovery(self, rate_limited_config):
        """Test extended recovery times for rate limit scenarios"""
        cb = ProductionCircuitBreaker(rate_limited_config)
        
        # Trigger opening with rate limit errors
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Rate limit {i}", error_type="RateLimitError")
        
        # Recovery should account for rate limit cooldown
        recovery_time = (cb.next_attempt_time - datetime.now()).total_seconds()
        assert recovery_time >= cb.config.recovery_timeout_seconds

    def test_rate_limit_burst_detection(self, rate_limited_config):
        """Test detection of rate limit bursts"""
        cb = ProductionCircuitBreaker(rate_limited_config)
        
        # Simulate burst of rate limit errors in short time
        for i in range(5):
            cb.record_failure(f"Burst rate limit {i}", error_type="RateLimitError")
        
        # Should open circuit quickly due to burst pattern
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.error_types["RateLimitError"] == 5

    def test_gradual_rate_limit_vs_burst(self, rate_limited_config):
        """Test different handling of gradual vs burst rate limiting"""
        # Gradual rate limiting
        cb_gradual = ProductionCircuitBreaker(rate_limited_config)
        for i in range(3):
            cb_gradual.record_failure(f"Gradual rate limit {i}", error_type="RateLimitError")
            cb_gradual.record_success(1000.0)  # Success in between
        
        # Burst rate limiting
        cb_burst = ProductionCircuitBreaker(rate_limited_config)
        for i in range(3):
            cb_burst.record_failure(f"Burst rate limit {i}", error_type="RateLimitError")
        
        # Burst should be more likely to open circuit
        assert cb_burst.consecutive_server_errors == 0  # Rate limits don't count as server errors
        # But should still trigger opening due to failure pattern


# ============================================================================
# Service Unavailable Scenario Tests
# ============================================================================

class TestServiceUnavailableScenarios:
    """Test circuit breaker behavior when services are unavailable"""

    def test_service_unavailable_immediate_opening(self, unreliable_config):
        """Test immediate circuit opening for service unavailable errors"""
        cb = ProductionCircuitBreaker(unreliable_config)
        
        # Service unavailable errors should be treated seriously
        cb.record_failure("Service unavailable", error_type="ServiceUnavailable")
        cb.record_failure("Backend not responding", error_type="ServiceUnavailable")
        cb.record_failure("Connection refused", error_type="ServiceUnavailable")
        
        # Should open circuit
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.error_types["ServiceUnavailable"] == 3

    def test_service_unavailable_vs_temporary_errors(self, unreliable_config):
        """Test different handling of permanent vs temporary unavailability"""
        # Permanent unavailability
        cb_permanent = ProductionCircuitBreaker(unreliable_config)
        for i in range(3):
            cb_permanent.record_failure(f"Service down {i}", error_type="ServiceUnavailable")
        
        # Temporary errors
        cb_temporary = ProductionCircuitBreaker(unreliable_config)
        for i in range(3):
            cb_temporary.record_failure(f"Temp error {i}", error_type="TemporaryError")
        
        # Both should open, but recovery times might differ
        assert cb_permanent.state == CircuitBreakerState.OPEN
        assert cb_temporary.state == CircuitBreakerState.OPEN

    def test_network_connectivity_issues(self, unreliable_config):
        """Test handling of network connectivity issues"""
        cb = ProductionCircuitBreaker(unreliable_config)
        
        # Network-related errors
        network_errors = [
            "Connection timeout",
            "DNS resolution failed", 
            "Network unreachable",
            "Connection reset by peer"
        ]
        
        for error in network_errors:
            cb.record_failure(error, error_type="NetworkError")
        
        # Should open circuit due to network issues
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.error_types["NetworkError"] == 4

    def test_intermittent_service_availability(self, unreliable_config):
        """Test handling of intermittent service availability"""
        cb = ProductionCircuitBreaker(unreliable_config)
        
        # Pattern of intermittent failures
        pattern = [
            ("fail", "Service unavailable"),
            ("success", 800.0),
            ("fail", "Service unavailable"),
            ("fail", "Service unavailable"),
            ("success", 900.0),
            ("fail", "Service unavailable")
        ]
        
        for operation, data in pattern:
            if operation == "fail":
                cb.record_failure(data, error_type="ServiceUnavailable")
            else:
                cb.record_success(data)
        
        # Should track intermittent pattern
        assert cb.failure_count > 0
        assert cb.success_count > 0
        # Decision depends on failure threshold and pattern


# ============================================================================
# Cascading Failure Scenario Tests
# ============================================================================

class TestCascadingFailureScenarios:
    """Test circuit breaker behavior in cascading failure scenarios"""

    def test_primary_service_failure_cascade(self):
        """Test cascade prevention when primary service fails"""
        # Create multiple interconnected circuit breakers
        configs = {
            'primary': BackendInstanceConfig(
                id="primary_service",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url="http://primary:8080",
                api_key="primary_key",
                failure_threshold=3,
                recovery_timeout_seconds=60
            ),
            'secondary': BackendInstanceConfig(
                id="secondary_service",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.secondary.com",
                api_key="secondary_key",
                failure_threshold=4,
                recovery_timeout_seconds=90
            ),
            'tertiary': BackendInstanceConfig(
                id="tertiary_service", 
                backend_type=BackendType.CACHE,
                endpoint_url="redis://cache:6379",
                api_key="",
                failure_threshold=10,
                recovery_timeout_seconds=30
            )
        }
        
        cbs = {name: ProductionCircuitBreaker(config) for name, config in configs.items()}
        
        # Primary service fails
        for i in range(cbs['primary'].config.failure_threshold):
            cbs['primary'].record_failure(f"Primary failure {i}")
        
        assert cbs['primary'].state == CircuitBreakerState.OPEN
        
        # Secondary service experiences increased load
        for i in range(2):
            cbs['secondary'].record_failure(f"Secondary overload {i}")
        
        # Tertiary (cache) should remain stable
        cbs['tertiary'].record_success(50.0)
        cbs['tertiary'].record_success(45.0)
        
        # Should prevent full cascade
        assert cbs['secondary'].state != CircuitBreakerState.OPEN  # Not fully cascaded
        assert cbs['tertiary'].state == CircuitBreakerState.CLOSED

    def test_domino_effect_prevention(self):
        """Test prevention of domino effect in circuit breaker chain"""
        # Create chain of dependent services
        chain_configs = []
        for i in range(5):
            config = BackendInstanceConfig(
                id=f"service_chain_{i}",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url=f"https://api.chain{i}.com",
                api_key=f"chain_key_{i}",
                failure_threshold=3 + i,  # Increasing thresholds
                recovery_timeout_seconds=60 + (i * 30)  # Staggered recovery
            )
            chain_configs.append(config)
        
        chain_cbs = [ProductionCircuitBreaker(config) for config in chain_configs]
        
        # Simulate cascade starting from first service
        for i, cb in enumerate(chain_cbs):
            # Each service fails based on position in chain
            failure_count = min(i + 2, cb.config.failure_threshold)
            for j in range(failure_count):
                cb.record_failure(f"Cascade failure {j} in service {i}")
        
        # Should have varying states, not all open
        states = [cb.state for cb in chain_cbs]
        assert not all(state == CircuitBreakerState.OPEN for state in states)
        assert CircuitBreakerState.CLOSED in states  # Some should remain closed

    def test_load_redistribution_after_failure(self):
        """Test load redistribution after service failures"""
        # Simulate load balancing scenario
        load_configs = {
            'high_capacity': BackendInstanceConfig(
                id="high_capacity",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url="http://high-cap:8080",
                api_key="high_key",
                weight=2.0,  # High capacity
                failure_threshold=5
            ),
            'medium_capacity': BackendInstanceConfig(
                id="medium_capacity",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.medium.com",
                api_key="medium_key",
                weight=1.0,  # Medium capacity
                failure_threshold=3
            ),
            'low_capacity': BackendInstanceConfig(
                id="low_capacity",
                backend_type=BackendType.CACHE,
                endpoint_url="redis://low:6379",
                api_key="",
                weight=0.5,  # Low capacity
                failure_threshold=8  # But more resilient
            )
        }
        
        load_cbs = {name: ProductionCircuitBreaker(config) for name, config in load_configs.items()}
        
        # High capacity service fails
        for i in range(load_cbs['high_capacity'].config.failure_threshold):
            load_cbs['high_capacity'].record_failure(f"High capacity failure {i}")
        
        # Medium capacity experiences some load increase
        load_cbs['medium_capacity'].record_failure("Load increase failure 1")
        load_cbs['medium_capacity'].record_success(1200.0)  # Slower response
        load_cbs['medium_capacity'].record_failure("Load increase failure 2")
        
        # Low capacity should handle overflow gracefully
        load_cbs['low_capacity'].record_success(100.0)
        load_cbs['low_capacity'].record_success(95.0)
        
        # Should maintain system stability
        assert load_cbs['high_capacity'].state == CircuitBreakerState.OPEN
        assert load_cbs['low_capacity'].state == CircuitBreakerState.CLOSED


# ============================================================================
# Budget Exhaustion Scenario Tests
# ============================================================================

class TestBudgetExhaustionScenarios:
    """Test circuit breaker behavior during budget exhaustion"""

    def test_budget_threshold_circuit_activation(self):
        """Test circuit breaker activation based on budget thresholds"""
        expensive_config = BackendInstanceConfig(
            id="expensive_service",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://api.expensive.com",
            api_key="expensive_key",
            cost_per_1k_tokens=1.0,  # Very expensive
            failure_threshold=5
        )
        
        cb = ProductionCircuitBreaker(expensive_config)
        
        # Simulate budget pressure scenario
        # This would typically integrate with cost tracking system
        with patch.object(cb, '_check_budget_constraints') as mock_budget:
            mock_budget.return_value = {
                'budget_exceeded': True,
                'cost_escalation': True,
                'recommended_action': 'reduce_expensive_calls'
            }
            
            # Should influence failure threshold
            cb.record_failure("Expensive call failed")
            cb.record_failure("Another expensive failure")
            
            # Budget constraints should make circuit more sensitive
            # This would be implemented in actual integration

    def test_cost_escalation_circuit_behavior(self):
        """Test circuit breaker behavior during cost escalation"""
        cost_configs = {
            'premium': BackendInstanceConfig(
                id="premium_service",
                cost_per_1k_tokens=0.50,
                failure_threshold=3
            ),
            'standard': BackendInstanceConfig(
                id="standard_service", 
                cost_per_1k_tokens=0.20,
                failure_threshold=4
            ),
            'budget': BackendInstanceConfig(
                id="budget_service",
                cost_per_1k_tokens=0.05,
                failure_threshold=6
            )
        }
        
        cost_cbs = {name: ProductionCircuitBreaker(config) for name, config in cost_configs.items()}
        
        # Premium service should be more sensitive to failures during cost pressure
        # This demonstrates the concept - actual implementation would integrate with cost tracking


# ============================================================================
# Memory Pressure Scenario Tests
# ============================================================================

class TestMemoryPressureScenarios:
    """Test circuit breaker behavior under memory pressure"""

    def test_memory_pressure_failure_detection(self, memory_constrained_config):
        """Test detection of memory-related failures"""
        cb = ProductionCircuitBreaker(memory_constrained_config)
        
        # Memory-related errors
        memory_errors = [
            "Out of memory error",
            "Memory allocation failed",
            "Heap space exhausted",
            "GC overhead limit exceeded"
        ]
        
        for error in memory_errors:
            cb.record_failure(error, error_type="MemoryError")
        
        # Should open circuit due to memory issues
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.error_types["MemoryError"] == 4

    def test_memory_pressure_response_time_degradation(self, memory_constrained_config):
        """Test detection of response time degradation under memory pressure"""
        cb = ProductionCircuitBreaker(memory_constrained_config)
        
        # Simulate gradual memory pressure through response time increases
        baseline_time = cb._baseline_response_time
        pressure_times = [
            baseline_time * 1.2,  # 20% slower
            baseline_time * 1.5,  # 50% slower
            baseline_time * 2.0,  # 100% slower
            baseline_time * 3.0,  # 200% slower
            baseline_time * 4.0   # 300% slower - should trigger proactive opening
        ]
        
        for time_ms in pressure_times:
            cb.record_success(time_ms)
        
        # Should detect performance degradation
        assert not cb.should_allow_request() or cb.state == CircuitBreakerState.OPEN

    def test_memory_leak_detection_pattern(self, memory_constrained_config):
        """Test detection of memory leak patterns"""
        cb = ProductionCircuitBreaker(memory_constrained_config)
        
        # Simulate memory leak through gradually increasing response times
        base_time = 500.0
        leak_pattern = []
        
        for i in range(20):
            # Gradually increasing response times simulating memory leak
            response_time = base_time + (i * 100)  # +100ms per request
            cb.record_success(response_time)
            leak_pattern.append(response_time)
        
        # Should eventually trigger proactive circuit opening
        recent_avg = statistics.mean(leak_pattern[-10:])
        initial_avg = statistics.mean(leak_pattern[:5])
        
        assert recent_avg > initial_avg * 2  # Significant degradation
        
        # Final request should trigger opening
        cb.record_success(base_time * 5)  # Very slow response
        assert cb.state == CircuitBreakerState.OPEN


# ============================================================================
# Network Connectivity Issue Tests  
# ============================================================================

class TestNetworkConnectivityScenarios:
    """Test circuit breaker behavior under network connectivity issues"""

    def test_network_partition_handling(self):
        """Test handling of network partition scenarios"""
        network_config = BackendInstanceConfig(
            id="network_service",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://api.remote.com",
            api_key="network_key",
            timeout_seconds=10.0,
            failure_threshold=2  # Low threshold for network issues
        )
        
        cb = ProductionCircuitBreaker(network_config)
        
        # Network partition errors
        partition_errors = [
            "Connection timeout",
            "Network unreachable", 
            "No route to host"
        ]
        
        for error in partition_errors:
            cb.record_failure(error, error_type="NetworkError")
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.error_types["NetworkError"] == 3

    def test_intermittent_connectivity_patterns(self):
        """Test handling of intermittent network connectivity"""
        cb = ProductionCircuitBreaker(BackendInstanceConfig(
            id="intermittent_service",
            backend_type=BackendType.LIGHTRAG,
            endpoint_url="http://unreliable-network:8080",
            api_key="intermittent_key",
            failure_threshold=4
        ))
        
        # Intermittent pattern: fail, success, fail, fail, success, fail
        pattern = [
            ("fail", "Network timeout"),
            ("success", 800.0),
            ("fail", "Connection dropped"), 
            ("fail", "Network error"),
            ("success", 750.0),
            ("fail", "Packet loss")
        ]
        
        for operation, data in pattern:
            if operation == "fail":
                cb.record_failure(data, error_type="NetworkError")
            else:
                cb.record_success(data)
        
        # Should eventually open due to failure pattern
        assert cb.failure_count >= 4

    def test_dns_resolution_failures(self):
        """Test handling of DNS resolution failures"""
        cb = ProductionCircuitBreaker(BackendInstanceConfig(
            id="dns_service",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://nonexistent.domain.com",
            api_key="dns_key",
            failure_threshold=2
        ))
        
        # DNS-related failures
        cb.record_failure("DNS resolution failed", error_type="DNSError")
        cb.record_failure("Host not found", error_type="DNSError")
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.error_types["DNSError"] == 2

    def test_ssl_certificate_issues(self):
        """Test handling of SSL/TLS certificate issues"""
        cb = ProductionCircuitBreaker(BackendInstanceConfig(
            id="ssl_service",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://expired-cert.com",
            api_key="ssl_key",
            failure_threshold=1  # Very sensitive to security issues
        ))
        
        # SSL-related failures
        cb.record_failure("SSL certificate expired", error_type="SSLError")
        
        # Should open immediately for security issues
        assert cb.state == CircuitBreakerState.OPEN


# ============================================================================
# Complex Scenario Integration Tests
# ============================================================================

class TestComplexFailureScenarios:
    """Test complex, multi-faceted failure scenarios"""

    def test_multiple_simultaneous_failure_types(self):
        """Test handling multiple types of failures simultaneously"""
        cb = ProductionCircuitBreaker(BackendInstanceConfig(
            id="multi_failure_service",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://api.problematic.com",
            api_key="multi_key",
            failure_threshold=5
        ))
        
        # Mix of different failure types
        failures = [
            ("TimeoutError", "Request timeout"),
            ("ServerError", "Internal server error"),
            ("RateLimitError", "Rate limit exceeded"), 
            ("NetworkError", "Connection failed"),
            ("TimeoutError", "Another timeout"),
            ("ServerError", "Server overloaded")
        ]
        
        for error_type, error_msg in failures:
            cb.record_failure(error_msg, error_type=error_type)
        
        # Should handle mixed error types appropriately
        assert cb.state == CircuitBreakerState.OPEN
        assert len(cb.error_types) >= 4  # Multiple error types recorded

    def test_recovery_under_continued_pressure(self):
        """Test recovery behavior when pressure continues during recovery"""
        cb = ProductionCircuitBreaker(BackendInstanceConfig(
            id="pressure_service",
            backend_type=BackendType.LIGHTRAG,
            endpoint_url="http://under-pressure:8080",
            api_key="pressure_key",
            failure_threshold=3,
            recovery_timeout_seconds=10,
            half_open_max_requests=3
        ))
        
        # Initial failures to open circuit
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Initial failure {i}")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Fast-forward to recovery time
        cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Transition to half-open
        assert cb.should_allow_request()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Mixed results during half-open testing
        cb.record_success(800.0)  # First test succeeds
        cb.record_failure("Still under pressure", error_type="LoadError")  # But still problems
        
        # Should return to OPEN state
        assert cb.state == CircuitBreakerState.OPEN

    def test_gradual_service_degradation_detection(self):
        """Test detection of gradual service degradation"""
        cb = ProductionCircuitBreaker(BackendInstanceConfig(
            id="degrading_service",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://api.degrading.com",
            api_key="degrade_key",
            failure_threshold=6,
            expected_response_time_ms=1000.0
        ))
        
        # Simulate gradual degradation
        degradation_pattern = [
            (900.0, True),   # Good performance
            (1100.0, True),  # Slight slowdown
            (1300.0, True),  # More slowdown
            (1800.0, True),  # Significant slowdown
            (2500.0, False), # Too slow, starts failing
            (3000.0, False), # Much too slow
            (4000.0, False)  # Unacceptable
        ]
        
        for response_time, success in degradation_pattern:
            if success:
                cb.record_success(response_time)
            else:
                cb.record_failure("Too slow", response_time_ms=response_time, error_type="PerformanceError")
        
        # Should detect gradual degradation and open
        assert cb.state == CircuitBreakerState.OPEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])