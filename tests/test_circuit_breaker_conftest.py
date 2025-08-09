"""
Comprehensive test configuration and fixtures for circuit breaker testing.

This module provides shared pytest fixtures, test utilities, and configuration
for all circuit breaker test suites. It ensures consistent test setup and
provides reusable components across all test modules.
"""

import pytest
import asyncio
import random
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional
import tempfile
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lightrag_integration'))

from production_load_balancer import (
    ProductionCircuitBreaker,
    CircuitBreakerState,
    BackendInstanceConfig,
    BackendType,
    ProductionLoadBalancer,
    ProductionLoadBalancingConfig,
    LoadBalancingStrategy
)


# ============================================================================
# Test Configuration Settings
# ============================================================================

# Test timeouts
ASYNC_TEST_TIMEOUT = 30.0
PERFORMANCE_TEST_TIMEOUT = 60.0
LOAD_TEST_TIMEOUT = 120.0

# Test data sizes
SMALL_DATASET_SIZE = 100
MEDIUM_DATASET_SIZE = 1000
LARGE_DATASET_SIZE = 10000

# Performance thresholds
MAX_OPERATION_TIME_US = 100  # microseconds
MAX_MEMORY_GROWTH_MB = 10    # megabytes
MIN_THROUGHPUT_OPS_SEC = 5000  # operations per second


# ============================================================================
# Core Configuration Fixtures
# ============================================================================

@pytest.fixture
def base_backend_config():
    """Base backend configuration for testing"""
    return BackendInstanceConfig(
        id="test_base_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://localhost:8080",
        api_key="test_base_key",
        weight=1.0,
        cost_per_1k_tokens=0.05,
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
def lightrag_config():
    """LightRAG service configuration"""
    return BackendInstanceConfig(
        id="lightrag_test",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://lightrag-test:8080",
        api_key="lightrag_test_key",
        weight=1.5,
        cost_per_1k_tokens=0.05,
        max_requests_per_minute=200,
        timeout_seconds=20.0,
        health_check_path="/health",
        priority=1,
        expected_response_time_ms=800.0,
        quality_score=0.95,
        reliability_score=0.98,
        failure_threshold=5,
        recovery_timeout_seconds=30,
        half_open_max_requests=10,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def perplexity_config():
    """Perplexity API configuration"""
    return BackendInstanceConfig(
        id="perplexity_test",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.perplexity.ai",
        api_key="perplexity_test_key",
        weight=1.0,
        cost_per_1k_tokens=0.20,
        max_requests_per_minute=100,
        timeout_seconds=30.0,
        health_check_path="/models",
        priority=2,
        expected_response_time_ms=2000.0,
        quality_score=0.85,
        reliability_score=0.90,
        failure_threshold=3,
        recovery_timeout_seconds=60,
        half_open_max_requests=5,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def cache_config():
    """Cache service configuration"""
    return BackendInstanceConfig(
        id="cache_test",
        backend_type=BackendType.CACHE,
        endpoint_url="redis://cache-test:6379",
        api_key="",
        weight=0.5,
        cost_per_1k_tokens=0.0,
        max_requests_per_minute=1000,
        timeout_seconds=5.0,
        health_check_path="/ping",
        priority=0,
        expected_response_time_ms=50.0,
        quality_score=0.70,
        reliability_score=0.99,
        failure_threshold=10,
        recovery_timeout_seconds=10,
        half_open_max_requests=20,
        circuit_breaker_enabled=True
    )


# ============================================================================
# Specialized Configuration Fixtures
# ============================================================================

@pytest.fixture
def high_performance_config():
    """Configuration optimized for high performance"""
    return BackendInstanceConfig(
        id="high_perf_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://high-perf:8080",
        api_key="high_perf_key",
        weight=2.0,
        cost_per_1k_tokens=0.03,
        max_requests_per_minute=500,
        timeout_seconds=10.0,
        priority=1,
        expected_response_time_ms=200.0,
        quality_score=0.90,
        reliability_score=0.99,
        failure_threshold=10,
        recovery_timeout_seconds=15,
        half_open_max_requests=20,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def fragile_config():
    """Configuration for fragile/unreliable service"""
    return BackendInstanceConfig(
        id="fragile_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.fragile.com",
        api_key="fragile_key",
        weight=0.5,
        cost_per_1k_tokens=0.30,
        max_requests_per_minute=20,
        timeout_seconds=60.0,
        priority=5,
        expected_response_time_ms=5000.0,
        quality_score=0.60,
        reliability_score=0.70,
        failure_threshold=1,  # Very sensitive
        recovery_timeout_seconds=300,  # Long recovery
        half_open_max_requests=2,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def cost_optimized_config():
    """Configuration for cost-optimized service"""
    return BackendInstanceConfig(
        id="cost_optimized_backend",
        backend_type=BackendType.LIGHTRAG,
        endpoint_url="http://cost-optimized:8080",
        api_key="cost_opt_key",
        weight=1.0,
        cost_per_1k_tokens=0.01,  # Very cheap
        max_requests_per_minute=300,
        timeout_seconds=45.0,
        priority=3,
        expected_response_time_ms=1500.0,
        quality_score=0.75,
        reliability_score=0.85,
        failure_threshold=7,
        recovery_timeout_seconds=45,
        circuit_breaker_enabled=True
    )


# ============================================================================
# Circuit Breaker Instance Fixtures
# ============================================================================

@pytest.fixture
def basic_circuit_breaker(base_backend_config):
    """Basic circuit breaker instance"""
    return ProductionCircuitBreaker(base_backend_config)

@pytest.fixture
def lightrag_circuit_breaker(lightrag_config):
    """LightRAG circuit breaker instance"""
    return ProductionCircuitBreaker(lightrag_config)

@pytest.fixture
def perplexity_circuit_breaker(perplexity_config):
    """Perplexity circuit breaker instance"""
    return ProductionCircuitBreaker(perplexity_config)

@pytest.fixture
def cache_circuit_breaker(cache_config):
    """Cache circuit breaker instance"""
    return ProductionCircuitBreaker(cache_config)

@pytest.fixture
def multiple_circuit_breakers(lightrag_config, perplexity_config, cache_config):
    """Multiple circuit breakers for cross-service testing"""
    return {
        'lightrag': ProductionCircuitBreaker(lightrag_config),
        'perplexity': ProductionCircuitBreaker(perplexity_config),
        'cache': ProductionCircuitBreaker(cache_config)
    }

@pytest.fixture
def specialized_circuit_breakers(high_performance_config, fragile_config, cost_optimized_config):
    """Specialized circuit breakers for different scenarios"""
    return {
        'high_performance': ProductionCircuitBreaker(high_performance_config),
        'fragile': ProductionCircuitBreaker(fragile_config),
        'cost_optimized': ProductionCircuitBreaker(cost_optimized_config)
    }


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def test_utilities():
    """Collection of test utility functions"""
    
    class TestUtils:
        @staticmethod
        def generate_failures(circuit_breaker, count, error_type="TestError"):
            """Generate a specified number of failures"""
            for i in range(count):
                circuit_breaker.record_failure(f"Generated failure {i}", error_type=error_type)
        
        @staticmethod
        def generate_successes(circuit_breaker, count, base_response_time=500):
            """Generate a specified number of successes"""
            for i in range(count):
                response_time = base_response_time + random.uniform(-100, 100)
                circuit_breaker.record_success(response_time)
        
        @staticmethod
        def open_circuit_breaker(circuit_breaker):
            """Force circuit breaker to open"""
            TestUtils.generate_failures(circuit_breaker, circuit_breaker.config.failure_threshold)
            assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        @staticmethod
        def set_half_open(circuit_breaker):
            """Set circuit breaker to half-open state"""
            TestUtils.open_circuit_breaker(circuit_breaker)
            circuit_breaker.next_attempt_time = datetime.now() - timedelta(seconds=1)
            circuit_breaker.should_allow_request()
            assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        
        @staticmethod
        def simulate_time_passage(circuit_breaker, seconds):
            """Simulate time passage for recovery testing"""
            if circuit_breaker.next_attempt_time:
                circuit_breaker.next_attempt_time = datetime.now() - timedelta(seconds=seconds)
        
        @staticmethod
        def assert_metrics_consistency(circuit_breaker):
            """Assert that circuit breaker metrics are internally consistent"""
            metrics = circuit_breaker.get_metrics()
            
            # Basic consistency checks
            assert metrics["success_count"] >= 0
            assert metrics["failure_count"] >= 0
            assert 0 <= metrics["failure_rate"] <= 1
            assert metrics["avg_response_time_ms"] >= 0
            
            # State-specific checks
            if metrics["state"] == CircuitBreakerState.OPEN.value:
                assert metrics["failure_count"] >= circuit_breaker.config.failure_threshold or \
                       metrics["consecutive_timeouts"] >= 3
            
            return True
        
        @staticmethod
        def create_test_scenario(name, operations):
            """Create a named test scenario with predefined operations"""
            scenarios = {
                "healthy_service": [
                    ("success", 400), ("success", 500), ("success", 450),
                    ("success", 550), ("success", 480)
                ],
                "degraded_service": [
                    ("success", 1200), ("success", 1500), ("failure", "SlowResponse"),
                    ("success", 1800), ("success", 1300)
                ],
                "failing_service": [
                    ("failure", "ServiceError"), ("failure", "TimeoutError"),
                    ("failure", "NetworkError"), ("failure", "ServerError")
                ],
                "intermittent_issues": [
                    ("success", 400), ("failure", "TransientError"), ("success", 450),
                    ("failure", "TimeoutError"), ("success", 500), ("success", 480)
                ],
                "recovery_pattern": [
                    ("failure", "RecoveryTest"), ("success", 800), ("success", 600),
                    ("success", 500), ("success", 450)
                ]
            }
            return scenarios.get(name, operations)
    
    return TestUtils()


# ============================================================================
# Mock System Fixtures
# ============================================================================

@pytest.fixture
def mock_backend_client():
    """Mock backend client with common methods"""
    client = AsyncMock()
    client.query = AsyncMock()
    client.health_check = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client

@pytest.fixture
def mock_metrics_system():
    """Mock metrics collection system"""
    metrics = Mock()
    metrics.record_counter = Mock()
    metrics.record_gauge = Mock()
    metrics.record_histogram = Mock()
    metrics.record_timing = Mock()
    metrics.get_metrics = Mock(return_value={})
    return metrics

@pytest.fixture
def mock_alert_system():
    """Mock alerting system"""
    alerts = AsyncMock()
    alerts.trigger_alert = AsyncMock()
    alerts.resolve_alert = AsyncMock()
    alerts.send_notification = AsyncMock()
    alerts.get_active_alerts = AsyncMock(return_value=[])
    return alerts

@pytest.fixture
def sample_queries():
    """Sample queries for testing different scenarios"""
    return {
        "simple": "What is diabetes?",
        "complex": "Explain the molecular mechanisms of insulin signaling in hepatocytes",
        "recent": "What are the latest developments in metabolomics research in 2025?",
        "specialized": "Compare GC-MS and LC-MS approaches for metabolite identification",
        "cache_worthy": "Define metabolomics",
        "error_prone": "This is a query that might cause errors in processing",
        "long": "This is a very long query " * 20 + "that might stress the system",
        "medical": "What are the biomarkers for metabolic syndrome and how are they measured?"
    }

@pytest.fixture  
def mock_response_patterns():
    """Mock response patterns for different scenarios"""
    return {
        "success_fast": {
            "content": "Fast successful response",
            "confidence": 0.90,
            "tokens_used": 100,
            "response_time_ms": 200
        },
        "success_slow": {
            "content": "Slow but successful response",
            "confidence": 0.85,
            "tokens_used": 150,
            "response_time_ms": 2500
        },
        "success_cached": {
            "content": "Cached response",
            "confidence": 0.80,
            "tokens_used": 50,
            "response_time_ms": 25,
            "cached": True
        },
        "timeout_error": TimeoutError("Request timed out"),
        "server_error": Exception("Internal server error"),
        "rate_limit_error": Exception("Rate limit exceeded"),
        "network_error": Exception("Network connection failed"),
        "validation_error": ValueError("Invalid request format")
    }


if __name__ == "__main__":
    print("Circuit Breaker Test Configuration Module")
    print("This module provides fixtures and utilities for circuit breaker testing.")
    print(f"Async test timeout: {ASYNC_TEST_TIMEOUT}s")
    print(f"Performance test timeout: {PERFORMANCE_TEST_TIMEOUT}s") 
    print(f"Load test timeout: {LOAD_TEST_TIMEOUT}s")