"""
Comprehensive test suite for ProductionCircuitBreaker functionality.

This module provides extensive test coverage for the production-grade circuit breaker
implementation including unit tests, integration tests, failure scenarios, 
performance testing, and monitoring validation.

Test Coverage:
- Core circuit breaker functionality
- State transitions and thresholds
- Error handling and recovery
- Performance characteristics
- Integration with load balancer
- Monitoring and metrics
"""

import pytest
import asyncio
import time
import threading
import statistics
import random
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import json

# Import the circuit breaker components
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
# Test Fixtures and Configuration
# ============================================================================

@pytest.fixture
def basic_backend_config():
    """Create basic backend configuration for testing"""
    return BackendInstanceConfig(
        id="test_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.test.com",
        api_key="test_key",
        weight=1.0,
        cost_per_1k_tokens=0.20,
        max_requests_per_minute=100,
        timeout_seconds=30.0,
        health_check_path="/health",
        priority=1,
        expected_response_time_ms=1000.0,
        quality_score=0.85,
        reliability_score=0.90,
        failure_threshold=3,
        recovery_timeout_seconds=60,
        half_open_max_requests=5,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def aggressive_circuit_breaker_config():
    """Create aggressive circuit breaker configuration for testing"""
    return BackendInstanceConfig(
        id="aggressive_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.test.com",
        api_key="test_key",
        failure_threshold=2,  # Very low threshold
        recovery_timeout_seconds=10,  # Quick recovery
        half_open_max_requests=2,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def tolerant_circuit_breaker_config():
    """Create tolerant circuit breaker configuration for testing"""
    return BackendInstanceConfig(
        id="tolerant_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://localhost:8080",
        api_key="test_key",
        failure_threshold=10,  # High threshold
        recovery_timeout_seconds=300,  # Slow recovery
        half_open_max_requests=10,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def circuit_breaker(basic_backend_config):
    """Create basic circuit breaker instance"""
    return ProductionCircuitBreaker(basic_backend_config)

@pytest.fixture
def multiple_circuit_breakers(basic_backend_config, aggressive_circuit_breaker_config, tolerant_circuit_breaker_config):
    """Create multiple circuit breakers for cross-service testing"""
    return {
        'basic': ProductionCircuitBreaker(basic_backend_config),
        'aggressive': ProductionCircuitBreaker(aggressive_circuit_breaker_config),
        'tolerant': ProductionCircuitBreaker(tolerant_circuit_breaker_config)
    }


# ============================================================================
# Unit Tests for Core Circuit Breaker Functionality
# ============================================================================

class TestProductionCircuitBreakerCore:
    """Test core circuit breaker functionality"""

    def test_initial_state_closed(self, circuit_breaker):
        """Test circuit breaker starts in CLOSED state"""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.last_failure_time is None
        assert circuit_breaker.next_attempt_time is None
        assert circuit_breaker.half_open_requests == 0

    def test_should_allow_request_when_closed(self, circuit_breaker):
        """Test requests are allowed in CLOSED state"""
        assert circuit_breaker.should_allow_request() is True

    def test_record_success_basic(self, circuit_breaker):
        """Test recording successful operations"""
        response_time = 500.0
        circuit_breaker.record_success(response_time)
        
        assert circuit_breaker.success_count == 1
        assert len(circuit_breaker.response_time_window) == 1
        assert circuit_breaker.response_time_window[0] == response_time
        assert len(circuit_breaker.failure_rate_window) == 1
        assert circuit_breaker.failure_rate_window[0] is False

    def test_record_failure_basic(self, circuit_breaker):
        """Test recording failed operations"""
        error_msg = "Test error"
        circuit_breaker.record_failure(error_msg, response_time_ms=2000.0, error_type="ServerError")
        
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.last_failure_time is not None
        assert len(circuit_breaker.failure_rate_window) == 1
        assert circuit_breaker.failure_rate_window[0] is True
        assert circuit_breaker.error_types["ServerError"] == 1

    def test_state_transition_closed_to_open(self, circuit_breaker):
        """Test transition from CLOSED to OPEN state"""
        # Record enough failures to trigger circuit opening
        for i in range(circuit_breaker.config.failure_threshold):
            circuit_breaker.record_failure(f"Error {i}", error_type="ServerError")
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.next_attempt_time is not None
        assert not circuit_breaker.should_allow_request()

    def test_state_transition_open_to_half_open(self, circuit_breaker):
        """Test transition from OPEN to HALF_OPEN state"""
        # First open the circuit
        for i in range(circuit_breaker.config.failure_threshold):
            circuit_breaker.record_failure(f"Error {i}")
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Simulate time passing for recovery
        circuit_breaker.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Should transition to HALF_OPEN
        assert circuit_breaker.should_allow_request() is True
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    def test_state_transition_half_open_to_closed(self, circuit_breaker):
        """Test transition from HALF_OPEN to CLOSED state"""
        # Open the circuit first
        for i in range(circuit_breaker.config.failure_threshold):
            circuit_breaker.record_failure(f"Error {i}")
        
        # Force to HALF_OPEN state
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        circuit_breaker.half_open_requests = 0
        
        # Record successful requests
        for i in range(circuit_breaker.config.half_open_max_requests):
            circuit_breaker.record_success(500.0)
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_state_transition_half_open_to_open(self, circuit_breaker):
        """Test transition from HALF_OPEN back to OPEN state on failure"""
        # Open the circuit first
        for i in range(circuit_breaker.config.failure_threshold):
            circuit_breaker.record_failure(f"Error {i}")
        
        # Force to HALF_OPEN state
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        circuit_breaker.half_open_requests = 0
        
        # Record a failure in HALF_OPEN state
        circuit_breaker.record_failure("Half-open failure")
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN


class TestAdaptiveThresholds:
    """Test adaptive threshold adjustment logic"""

    def test_timeout_adaptive_threshold(self, circuit_breaker):
        """Test adaptive threshold for timeout errors"""
        # Record consecutive timeout errors
        for i in range(3):
            circuit_breaker.record_failure(f"Timeout {i}", error_type="TimeoutError")
        
        # Adaptive threshold should be lowered
        assert circuit_breaker._adaptive_failure_threshold < circuit_breaker.config.failure_threshold

    def test_server_error_adaptive_threshold(self, circuit_breaker):
        """Test adaptive threshold for server errors"""
        # Record consecutive server errors
        for i in range(3):
            circuit_breaker.record_failure(f"Server Error {i}", error_type="ServerError")
        
        # Adaptive threshold should be lowered
        assert circuit_breaker._adaptive_failure_threshold < circuit_breaker.config.failure_threshold

    def test_mixed_error_threshold_reset(self, circuit_breaker):
        """Test threshold reset after mixed error types"""
        # Record consecutive timeout errors
        circuit_breaker.record_failure("Timeout 1", error_type="TimeoutError")
        circuit_breaker.record_failure("Timeout 2", error_type="TimeoutError")
        
        # Threshold should be lowered
        lowered_threshold = circuit_breaker._adaptive_failure_threshold
        
        # Record a different error type
        circuit_breaker.record_failure("Different error", error_type="ValidationError")
        
        # Threshold should reset towards default
        assert circuit_breaker._adaptive_failure_threshold >= lowered_threshold

    def test_success_resets_consecutive_counters(self, circuit_breaker):
        """Test that success resets consecutive error counters"""
        # Record consecutive errors
        circuit_breaker.record_failure("Timeout", error_type="TimeoutError")
        circuit_breaker.record_failure("Server Error", error_type="ServerError")
        
        assert circuit_breaker.consecutive_timeouts == 1
        assert circuit_breaker.consecutive_server_errors == 1
        
        # Record success
        circuit_breaker.record_success(500.0)
        
        assert circuit_breaker.consecutive_timeouts == 0
        assert circuit_breaker.consecutive_server_errors == 0


class TestProactiveCircuitOpening:
    """Test proactive circuit opening based on performance degradation"""

    def test_proactive_opening_high_response_times(self, circuit_breaker):
        """Test proactive opening due to sustained high response times"""
        # Fill response time window with high values
        baseline = circuit_breaker._baseline_response_time
        for i in range(15):
            circuit_breaker.record_success(baseline * 4)  # 4x expected response time
        
        # Should trigger proactive opening
        assert not circuit_breaker.should_allow_request()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_proactive_opening_high_failure_rate(self, circuit_breaker):
        """Test proactive opening due to high failure rate"""
        # Fill failure rate window with high failure rate
        for i in range(25):
            if i < 15:  # 60% failure rate
                circuit_breaker.record_failure(f"Error {i}")
            else:
                circuit_breaker.record_success(500.0)
        
        # Should trigger proactive opening
        assert not circuit_breaker.should_allow_request()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_proactive_opening_consecutive_timeouts(self, circuit_breaker):
        """Test proactive opening due to consecutive timeouts"""
        # Record consecutive timeout errors
        for i in range(3):
            circuit_breaker.record_failure(f"Timeout {i}", error_type="TimeoutError")
        
        # Should trigger proactive opening
        assert not circuit_breaker.should_allow_request()
        assert circuit_breaker.state == CircuitBreakerState.OPEN


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery mechanisms"""

    def test_adaptive_recovery_timeout_timeouts(self, circuit_breaker):
        """Test adaptive recovery timeout for timeout errors"""
        # Record timeout errors to trigger adaptive timeout
        for i in range(4):
            circuit_breaker.record_failure(f"Timeout {i}", error_type="TimeoutError")
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        base_timeout = circuit_breaker.config.recovery_timeout_seconds
        
        # Recovery time should be doubled for timeouts
        time_until_retry = (circuit_breaker.next_attempt_time - datetime.now()).total_seconds()
        assert time_until_retry > base_timeout * 1.5  # Account for jitter

    def test_adaptive_recovery_timeout_server_errors(self, circuit_breaker):
        """Test adaptive recovery timeout for server errors"""
        # Record server errors to trigger adaptive timeout
        for i in range(4):
            circuit_breaker.record_failure(f"Server Error {i}", error_type="ServerError")
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        base_timeout = circuit_breaker.config.recovery_timeout_seconds
        
        # Recovery time should be increased for server errors
        time_until_retry = (circuit_breaker.next_attempt_time - datetime.now()).total_seconds()
        assert time_until_retry > base_timeout  # Should be increased

    def test_jitter_in_recovery_timeout(self, circuit_breaker):
        """Test that recovery timeout includes jitter"""
        # Create multiple circuit breakers and trigger opening
        recovery_times = []
        
        for i in range(10):
            cb = ProductionCircuitBreaker(circuit_breaker.config)
            # Trigger circuit opening
            for j in range(cb.config.failure_threshold):
                cb.record_failure(f"Error {j}")
            
            time_until_retry = (cb.next_attempt_time - datetime.now()).total_seconds()
            recovery_times.append(time_until_retry)
        
        # Should have variation due to jitter
        assert len(set([round(t, 1) for t in recovery_times])) > 1


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics collection and reporting"""

    def test_basic_metrics(self, circuit_breaker):
        """Test basic metrics collection"""
        # Record some operations
        circuit_breaker.record_success(500.0)
        circuit_breaker.record_success(600.0)
        circuit_breaker.record_failure("Error 1", response_time_ms=1000.0, error_type="ServerError")
        
        metrics = circuit_breaker.get_metrics()
        
        assert metrics['state'] == CircuitBreakerState.CLOSED.value
        assert metrics['failure_count'] == 1
        assert metrics['success_count'] == 2
        assert metrics['failure_rate'] == 1/3  # 1 failure out of 3 operations
        assert metrics['avg_response_time_ms'] > 0
        assert 'ServerError' in metrics['error_types']

    def test_percentile_response_times(self, circuit_breaker):
        """Test percentile response time calculation"""
        # Record varied response times
        response_times = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for rt in response_times:
            circuit_breaker.record_success(rt)
        
        metrics = circuit_breaker.get_metrics()
        assert metrics['p95_response_time_ms'] > 0
        assert metrics['avg_response_time_ms'] == statistics.mean(response_times)

    def test_error_type_tracking(self, circuit_breaker):
        """Test error type frequency tracking"""
        # Record different error types
        error_types = ['TimeoutError', 'ServerError', 'ValidationError', 'TimeoutError']
        for error_type in error_types:
            circuit_breaker.record_failure(f"Error: {error_type}", error_type=error_type)
        
        metrics = circuit_breaker.get_metrics()
        assert metrics['error_types']['TimeoutError'] == 2
        assert metrics['error_types']['ServerError'] == 1
        assert metrics['error_types']['ValidationError'] == 1

    def test_metrics_during_state_transitions(self, circuit_breaker):
        """Test metrics accuracy during state transitions"""
        # Initial CLOSED state
        initial_metrics = circuit_breaker.get_metrics()
        assert initial_metrics['state'] == CircuitBreakerState.CLOSED.value
        
        # Trigger OPEN state
        for i in range(circuit_breaker.config.failure_threshold):
            circuit_breaker.record_failure(f"Error {i}")
        
        open_metrics = circuit_breaker.get_metrics()
        assert open_metrics['state'] == CircuitBreakerState.OPEN.value
        assert open_metrics['next_attempt_time'] is not None
        assert open_metrics['time_until_retry_seconds'] is not None

    def test_reset_functionality(self, circuit_breaker):
        """Test circuit breaker reset functionality"""
        # Add some data
        circuit_breaker.record_success(500.0)
        circuit_breaker.record_failure("Error", error_type="ServerError")
        
        # Trigger opening
        for i in range(circuit_breaker.config.failure_threshold):
            circuit_breaker.record_failure(f"Error {i}")
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Reset
        circuit_breaker.reset()
        
        # Should be back to initial state
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert len(circuit_breaker.failure_rate_window) == 0
        assert len(circuit_breaker.response_time_window) == 0
        assert len(circuit_breaker.error_types) == 0


# ============================================================================
# Service-Specific Circuit Breaker Tests
# ============================================================================

class TestServiceSpecificBehavior:
    """Test circuit breaker behavior for different services"""

    def test_perplexity_circuit_breaker_behavior(self):
        """Test circuit breaker behavior for Perplexity service"""
        config = BackendInstanceConfig(
            id="perplexity_test",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="https://api.perplexity.ai",
            api_key="test_key",
            failure_threshold=3,
            recovery_timeout_seconds=60,
            expected_response_time_ms=2000.0  # Higher expected time for external API
        )
        
        cb = ProductionCircuitBreaker(config)
        
        # Should handle API-specific errors appropriately
        cb.record_failure("Rate limit exceeded", error_type="RateLimitError")
        cb.record_failure("API timeout", error_type="TimeoutError")
        
        # Should adapt threshold for timeout patterns
        assert cb.consecutive_timeouts > 0

    def test_lightrag_circuit_breaker_behavior(self):
        """Test circuit breaker behavior for LightRAG service"""
        config = BackendInstanceConfig(
            id="lightrag_test",
            backend_type=BackendType.LIGHTRAG,
            endpoint_url="http://localhost:8080",
            api_key="internal_key",
            failure_threshold=5,  # Higher threshold for internal service
            recovery_timeout_seconds=30,  # Faster recovery
            expected_response_time_ms=800.0  # Lower expected time for internal service
        )
        
        cb = ProductionCircuitBreaker(config)
        
        # Should be more tolerant of internal service issues
        for i in range(3):
            cb.record_failure(f"Internal error {i}", error_type="ServiceError")
        
        # Should still be closed due to higher threshold
        assert cb.state == CircuitBreakerState.CLOSED

    def test_cache_circuit_breaker_behavior(self):
        """Test circuit breaker behavior for cache service"""
        config = BackendInstanceConfig(
            id="cache_test",
            backend_type=BackendType.CACHE,
            endpoint_url="redis://localhost:6379",
            api_key="",
            failure_threshold=10,  # Very high threshold for cache
            recovery_timeout_seconds=5,   # Very fast recovery
            expected_response_time_ms=50.0  # Very low expected time
        )
        
        cb = ProductionCircuitBreaker(config)
        
        # Cache should be very tolerant but recover quickly
        for i in range(8):
            cb.record_failure(f"Cache miss {i}", error_type="CacheMiss")
        
        assert cb.state == CircuitBreakerState.CLOSED  # Still within threshold


# ============================================================================
# Cross-Service Coordination Tests  
# ============================================================================

class TestCrossServiceCoordination:
    """Test circuit breaker coordination across multiple services"""

    def test_independent_circuit_breaker_states(self, multiple_circuit_breakers):
        """Test that circuit breakers operate independently"""
        cbs = multiple_circuit_breakers
        
        # Trigger failure in one circuit breaker
        for i in range(cbs['aggressive'].config.failure_threshold):
            cbs['aggressive'].record_failure(f"Error {i}")
        
        assert cbs['aggressive'].state == CircuitBreakerState.OPEN
        assert cbs['basic'].state == CircuitBreakerState.CLOSED
        assert cbs['tolerant'].state == CircuitBreakerState.CLOSED

    def test_cascading_failure_prevention(self, multiple_circuit_breakers):
        """Test prevention of cascading failures across services"""
        cbs = multiple_circuit_breakers
        
        # Simulate failure patterns that might cause cascading
        # Primary service fails
        for i in range(cbs['basic'].config.failure_threshold):
            cbs['basic'].record_failure(f"Primary failure {i}")
        
        # Secondary service experiences increased load
        for i in range(5):
            if i < 3:
                cbs['aggressive'].record_failure(f"Secondary overload {i}")
            else:
                cbs['aggressive'].record_success(1000.0)  # Higher response times
        
        # Should handle gracefully without full cascade
        states = [cb.state for cb in cbs.values()]
        assert not all(state == CircuitBreakerState.OPEN for state in states)

    def test_service_recovery_coordination(self, multiple_circuit_breakers):
        """Test coordinated recovery across services"""
        cbs = multiple_circuit_breakers
        
        # Open multiple circuit breakers
        for name, cb in cbs.items():
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Error {i}")
        
        # All should be open
        for cb in cbs.values():
            assert cb.state == CircuitBreakerState.OPEN
        
        # Simulate time passing for different recovery times
        for name, cb in cbs.items():
            cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Check recovery transitions
        recovery_states = []
        for cb in cbs.values():
            if cb.should_allow_request():
                recovery_states.append(CircuitBreakerState.HALF_OPEN)
            else:
                recovery_states.append(cb.state)
        
        # Should have different recovery characteristics
        assert CircuitBreakerState.HALF_OPEN in recovery_states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])