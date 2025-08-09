"""
Integration tests for ProductionCircuitBreaker with ProductionLoadBalancer.

This module tests the integration between the circuit breaker system and the
production load balancer, including routing decisions, fallback mechanisms,
and performance monitoring.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lightrag_integration'))

from production_load_balancer import (
    ProductionLoadBalancer,
    ProductionLoadBalancingConfig,
    ProductionCircuitBreaker,
    CircuitBreakerState,
    BackendInstanceConfig,
    BackendType,
    LoadBalancingStrategy,
    BackendMetrics,
    create_default_production_config
)


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
async def production_config():
    """Create production configuration with multiple backends"""
    config = ProductionLoadBalancingConfig(
        strategy=LoadBalancingStrategy.ADAPTIVE_LEARNING,
        enable_adaptive_routing=True,
        enable_cost_optimization=True,
        enable_quality_based_routing=True,
        enable_real_time_monitoring=True,
        global_circuit_breaker_enabled=True,
        cascade_failure_prevention=True,
        backend_instances={
            "perplexity_1": BackendInstanceConfig(
                id="perplexity_1",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.perplexity.ai",
                api_key="test_key_1",
                weight=1.0,
                cost_per_1k_tokens=0.20,
                failure_threshold=3,
                recovery_timeout_seconds=60,
                circuit_breaker_enabled=True
            ),
            "perplexity_2": BackendInstanceConfig(
                id="perplexity_2",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.perplexity.ai",
                api_key="test_key_2",
                weight=1.0,
                cost_per_1k_tokens=0.20,
                failure_threshold=3,
                recovery_timeout_seconds=60,
                circuit_breaker_enabled=True
            ),
            "lightrag_1": BackendInstanceConfig(
                id="lightrag_1",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url="http://localhost:8080",
                api_key="internal_key",
                weight=1.5,
                cost_per_1k_tokens=0.05,
                failure_threshold=5,
                recovery_timeout_seconds=30,
                circuit_breaker_enabled=True
            ),
            "cache_1": BackendInstanceConfig(
                id="cache_1",
                backend_type=BackendType.CACHE,
                endpoint_url="redis://localhost:6379",
                api_key="",
                weight=0.1,
                cost_per_1k_tokens=0.0,
                failure_threshold=10,
                recovery_timeout_seconds=5,
                circuit_breaker_enabled=True
            )
        }
    )
    return config

@pytest.fixture
async def load_balancer(production_config):
    """Create production load balancer instance"""
    with patch.multiple(
        'production_load_balancer.ProductionLoadBalancer',
        _initialize_backend_clients=AsyncMock(),
        _start_monitoring_tasks=AsyncMock(),
        _initialize_metrics_collection=AsyncMock()
    ):
        lb = ProductionLoadBalancer(production_config)
        await lb.initialize()
        return lb

@pytest.fixture
def mock_backend_responses():
    """Mock backend response patterns"""
    return {
        "success": {"content": "Success response", "tokens_used": 100},
        "timeout": TimeoutError("Request timeout"),
        "server_error": Exception("Internal server error"),
        "rate_limit": Exception("Rate limit exceeded"),
        "invalid_response": {"error": "Invalid request format"}
    }


# ============================================================================
# Circuit Breaker Integration Tests
# ============================================================================

class TestCircuitBreakerLoadBalancerIntegration:
    """Test circuit breaker integration with load balancer"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, load_balancer):
        """Test circuit breakers are properly initialized for all backends"""
        assert hasattr(load_balancer, 'circuit_breakers')
        
        for backend_id in load_balancer.config.backend_instances.keys():
            assert backend_id in load_balancer.circuit_breakers
            cb = load_balancer.circuit_breakers[backend_id]
            assert isinstance(cb, ProductionCircuitBreaker)
            assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_request_filtering_by_circuit_breaker(self, load_balancer):
        """Test that circuit breakers filter requests appropriately"""
        # Open a circuit breaker
        backend_id = "perplexity_1"
        cb = load_balancer.circuit_breakers[backend_id]
        
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Test failure {i}")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Should not route to this backend
        with patch.object(load_balancer, '_select_backend_with_algorithm') as mock_select:
            # Mock selecting the open backend
            mock_select.return_value = backend_id
            
            # Should find an alternative backend due to circuit breaker
            available_backends = await load_balancer._get_available_backends("test query")
            assert backend_id not in available_backends

    @pytest.mark.asyncio
    async def test_fallback_routing_on_circuit_breaker_open(self, load_balancer):
        """Test fallback routing when primary circuit breaker opens"""
        # Open primary backend circuit breaker
        primary_backend = "lightrag_1"
        cb = load_balancer.circuit_breakers[primary_backend]
        
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Primary failure {i}")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Mock backend selection to prefer the failed backend initially
        with patch.object(load_balancer, '_calculate_backend_score') as mock_score:
            def mock_score_func(backend_id, metrics, query, context):
                if backend_id == primary_backend:
                    return 0.9  # High score but circuit is open
                else:
                    return 0.7  # Lower score but available
            
            mock_score.side_effect = mock_score_func
            
            # Should route to fallback backend
            selected_backend = await load_balancer._select_optimal_backend("test query", {})
            assert selected_backend != primary_backend
            assert selected_backend in load_balancer.config.backend_instances

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_updates_from_requests(self, load_balancer):
        """Test that circuit breaker states update based on actual request results"""
        backend_id = "perplexity_1"
        cb = load_balancer.circuit_breakers[backend_id]
        
        # Mock successful request
        with patch.object(load_balancer.backend_clients[backend_id], 'query') as mock_query:
            mock_query.return_value = {"content": "Success", "response_time_ms": 500}
            
            # Should update circuit breaker with success
            response = await load_balancer._execute_query_on_backend(
                backend_id, "test query", {}
            )
            
            assert cb.success_count > 0
            assert len(cb.response_time_window) > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_recording(self, load_balancer):
        """Test that circuit breaker records failures from actual requests"""
        backend_id = "perplexity_2"
        cb = load_balancer.circuit_breakers[backend_id]
        
        # Mock failing request
        with patch.object(load_balancer.backend_clients[backend_id], 'query') as mock_query:
            mock_query.side_effect = TimeoutError("Request timeout")
            
            # Should update circuit breaker with failure
            with pytest.raises(TimeoutError):
                await load_balancer._execute_query_on_backend(
                    backend_id, "test query", {}
                )
            
            assert cb.failure_count > 0
            assert cb.consecutive_timeouts > 0

    @pytest.mark.asyncio
    async def test_half_open_state_testing(self, load_balancer):
        """Test circuit breaker half-open state testing"""
        backend_id = "lightrag_1"
        cb = load_balancer.circuit_breakers[backend_id]
        
        # Open the circuit
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Failure {i}")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Simulate recovery time passing
        cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Should transition to HALF_OPEN
        assert cb.should_allow_request()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Test successful half-open requests
        with patch.object(load_balancer.backend_clients[backend_id], 'query') as mock_query:
            mock_query.return_value = {"content": "Recovery success", "response_time_ms": 300}
            
            # Execute half-open test requests
            for i in range(cb.config.half_open_max_requests):
                await load_balancer._execute_query_on_backend(
                    backend_id, f"test query {i}", {}
                )
            
            # Should transition back to CLOSED
            assert cb.state == CircuitBreakerState.CLOSED


class TestCascadeFailurePrevention:
    """Test cascade failure prevention mechanisms"""

    @pytest.mark.asyncio
    async def test_prevents_full_system_cascade(self, load_balancer):
        """Test that system prevents complete cascade failures"""
        # Gradually fail backends to simulate cascade scenario
        backend_ids = list(load_balancer.config.backend_instances.keys())
        
        # Fail all but one backend
        for backend_id in backend_ids[:-1]:
            cb = load_balancer.circuit_breakers[backend_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Cascade failure {i}")
        
        # Should still have one healthy backend available
        available_backends = await load_balancer._get_available_backends("test query")
        assert len(available_backends) >= 1
        
        # System should still be able to route requests
        selected_backend = await load_balancer._select_optimal_backend("test query", {})
        assert selected_backend is not None

    @pytest.mark.asyncio
    async def test_emergency_fallback_activation(self, load_balancer):
        """Test emergency fallback when most circuit breakers are open"""
        # Open most circuit breakers
        backend_ids = list(load_balancer.config.backend_instances.keys())
        
        for backend_id in backend_ids[:-1]:  # Leave one healthy
            cb = load_balancer.circuit_breakers[backend_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Emergency scenario {i}")
        
        # Should activate emergency protocols
        with patch.object(load_balancer, '_activate_emergency_fallback') as mock_fallback:
            await load_balancer.query("emergency test query")
            # Emergency fallback should be considered
            
        # System should still provide responses
        response = await load_balancer.query("test query after emergency")
        assert response is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_coordination(self, load_balancer):
        """Test coordinated recovery across multiple circuit breakers"""
        # Open all circuit breakers
        for backend_id, cb in load_balancer.circuit_breakers.items():
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"System-wide failure {i}")
        
        # All should be open
        for cb in load_balancer.circuit_breakers.values():
            assert cb.state == CircuitBreakerState.OPEN
        
        # Simulate staggered recovery times
        recovery_times = [10, 20, 30, 40]  # Different recovery times
        for i, (backend_id, cb) in enumerate(load_balancer.circuit_breakers.items()):
            cb.next_attempt_time = datetime.now() + timedelta(seconds=recovery_times[i % len(recovery_times)])
        
        # Should have coordinated recovery strategy
        await load_balancer._coordinate_system_recovery()
        
        # Recovery should be managed to prevent thundering herd
        recovery_states = [cb.state for cb in load_balancer.circuit_breakers.values()]
        assert not all(state == CircuitBreakerState.HALF_OPEN for state in recovery_states)


# ============================================================================
# Cost-Based Circuit Breaker Integration
# ============================================================================

class TestCostBasedCircuitBreakerIntegration:
    """Test integration with cost-based circuit breakers"""

    @pytest.mark.asyncio
    async def test_cost_threshold_circuit_breaker_integration(self, load_balancer):
        """Test integration between production and cost-based circuit breakers"""
        # Mock cost tracking
        with patch.object(load_balancer, 'cost_tracker') as mock_cost_tracker:
            mock_cost_tracker.check_budget_thresholds.return_value = {
                "budget_exceeded": True,
                "daily_limit_reached": False,
                "cost_escalation_detected": True
            }
            
            # Should influence routing decisions
            backend_selection = await load_balancer._select_cost_optimized_backend("test query")
            
            # Should prefer lower-cost backends when cost thresholds triggered
            selected_config = load_balancer.config.backend_instances[backend_selection]
            assert selected_config.cost_per_1k_tokens <= 0.10  # Prefer low-cost options

    @pytest.mark.asyncio
    async def test_budget_crisis_circuit_breaker_behavior(self, load_balancer):
        """Test circuit breaker behavior during budget crisis scenarios"""
        # Simulate budget crisis
        with patch.object(load_balancer, '_is_budget_crisis') as mock_crisis:
            mock_crisis.return_value = True
            
            # Should activate cost-protection circuit breakers
            for backend_id, config in load_balancer.config.backend_instances.items():
                if config.cost_per_1k_tokens > 0.15:  # High-cost backends
                    cb = load_balancer.circuit_breakers[backend_id]
                    # Should be more aggressive in opening
                    assert cb._adaptive_failure_threshold <= config.failure_threshold

    @pytest.mark.asyncio
    async def test_quality_vs_cost_circuit_breaker_decisions(self, load_balancer):
        """Test circuit breaker decisions balancing quality and cost"""
        # Set up quality vs cost scenario
        high_quality_expensive = "perplexity_1"  # High quality, high cost
        low_quality_cheap = "cache_1"           # Low quality, low cost
        
        # Degrade high-quality backend
        cb_expensive = load_balancer.circuit_breakers[high_quality_expensive]
        cb_expensive.record_failure("Quality degradation", response_time_ms=5000)
        cb_expensive.record_failure("High latency", response_time_ms=6000)
        
        # Should route to cheaper alternative despite quality difference
        with patch.object(load_balancer, '_calculate_quality_cost_score') as mock_score:
            mock_score.return_value = {"backend": low_quality_cheap, "score": 0.8}
            
            selected = await load_balancer._select_optimal_backend("cost-conscious query", {})
            
            # Should consider cost in circuit breaker decisions
            assert selected == low_quality_cheap or cb_expensive.state == CircuitBreakerState.OPEN


# ============================================================================
# Monitoring Integration Tests
# ============================================================================

class TestMonitoringIntegration:
    """Test circuit breaker integration with monitoring systems"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_collection(self, load_balancer):
        """Test that circuit breaker metrics are properly collected"""
        # Generate some activity
        for backend_id, cb in load_balancer.circuit_breakers.items():
            cb.record_success(500.0)
            cb.record_failure("Test failure", error_type="TestError")
        
        # Collect system metrics
        system_metrics = await load_balancer.get_system_metrics()
        
        assert 'circuit_breaker_metrics' in system_metrics
        
        for backend_id in load_balancer.config.backend_instances.keys():
            assert backend_id in system_metrics['circuit_breaker_metrics']
            cb_metrics = system_metrics['circuit_breaker_metrics'][backend_id]
            assert 'state' in cb_metrics
            assert 'failure_rate' in cb_metrics
            assert 'avg_response_time_ms' in cb_metrics

    @pytest.mark.asyncio
    async def test_circuit_breaker_alerting_triggers(self, load_balancer):
        """Test that circuit breaker state changes trigger appropriate alerts"""
        backend_id = "perplexity_1"
        cb = load_balancer.circuit_breakers[backend_id]
        
        with patch.object(load_balancer, 'alert_manager') as mock_alert_manager:
            # Trigger circuit opening
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Alert trigger {i}")
            
            # Should trigger circuit breaker alert
            await load_balancer._check_circuit_breaker_alerts()
            
            # Verify alert was triggered
            mock_alert_manager.trigger_alert.assert_called()
            alert_calls = mock_alert_manager.trigger_alert.call_args_list
            
            circuit_breaker_alerts = [
                call for call in alert_calls 
                if 'circuit_breaker' in str(call).lower()
            ]
            assert len(circuit_breaker_alerts) > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_dashboard_data(self, load_balancer):
        """Test circuit breaker data for dashboard visualization"""
        # Generate varied circuit breaker states
        backends = list(load_balancer.circuit_breakers.keys())
        
        # Keep one healthy, open one, put one in half-open
        cb_open = load_balancer.circuit_breakers[backends[0]]
        for i in range(cb_open.config.failure_threshold):
            cb_open.record_failure(f"Dashboard test {i}")
        
        cb_half_open = load_balancer.circuit_breakers[backends[1]]
        cb_half_open.state = CircuitBreakerState.HALF_OPEN
        
        # Get dashboard data
        dashboard_data = await load_balancer.get_dashboard_data()
        
        assert 'circuit_breaker_status' in dashboard_data
        status_data = dashboard_data['circuit_breaker_status']
        
        # Should have state distribution
        states = [backend_data['state'] for backend_data in status_data.values()]
        assert CircuitBreakerState.OPEN.value in states
        assert CircuitBreakerState.HALF_OPEN.value in states
        assert CircuitBreakerState.CLOSED.value in states

    @pytest.mark.asyncio
    async def test_circuit_breaker_health_check_integration(self, load_balancer):
        """Test circuit breaker integration with health check system"""
        backend_id = "lightrag_1"
        cb = load_balancer.circuit_breakers[backend_id]
        
        # Open circuit breaker
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Health check test {i}")
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Health check should reflect circuit breaker state
        health_status = await load_balancer.get_backend_health_status(backend_id)
        assert health_status['circuit_breaker_state'] == CircuitBreakerState.OPEN.value
        assert health_status['available'] is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance_impact_monitoring(self, load_balancer):
        """Test monitoring of circuit breaker performance impact"""
        # Execute requests with circuit breaker overhead
        start_time = time.time()
        
        for i in range(100):
            backend_id = await load_balancer._select_optimal_backend(f"query {i}", {})
            # Simulate circuit breaker decision overhead
            cb = load_balancer.circuit_breakers[backend_id]
            cb.should_allow_request()
        
        execution_time = time.time() - start_time
        
        # Performance metrics should track circuit breaker overhead
        perf_metrics = await load_balancer.get_performance_metrics()
        
        assert 'circuit_breaker_overhead_ms' in perf_metrics
        assert 'routing_decision_time_ms' in perf_metrics
        
        # Circuit breaker overhead should be minimal
        assert perf_metrics['circuit_breaker_overhead_ms'] < 10.0  # Less than 10ms per 100 decisions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])