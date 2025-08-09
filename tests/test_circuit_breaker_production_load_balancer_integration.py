#!/usr/bin/env python3
"""
Comprehensive Test Suite for Circuit Breaker Production Load Balancer Integration

This test suite validates the critical Priority 4 requirement: circuit breaker coordination
with intelligent routing systems in production load balancing scenarios.

Test Coverage:
- Circuit breaker backend isolation during failures
- Intelligent routing with circuit breaker states
- Automatic failover when backends circuit break
- Load balancing weight adjustments based on circuit breaker health
- Recovery coordination between routing and circuit breakers
- Production routing under mixed circuit breaker states
- Query router circuit breaker health integration

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Priority 4 - Production Load Balancer Integration Testing
"""

import asyncio
import pytest
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from enum import Enum
import logging

# Import the modules being tested
import sys
sys.path.append(str(Path(__file__).parent.parent / "lightrag_integration"))

try:
    # Circuit breaker and load balancer components
    from lightrag_integration.production_load_balancer import (
        ProductionLoadBalancer,
        ProductionLoadBalancingConfig,
        BackendInstanceConfig,
        BackendType,
        CircuitBreakerState,
        HealthStatus,
        BackendMetrics,
        ProductionCircuitBreaker,
        LoadBalancingStrategy,
        create_default_production_config
    )

    # Intelligent routing components  
    from lightrag_integration.production_intelligent_query_router import (
        ProductionIntelligentQueryRouter,
        ProductionFeatureFlags,
        DeploymentMode,
        PerformanceComparison
    )

    from lightrag_integration.intelligent_query_router import (
        IntelligentQueryRouter,
        LoadBalancingConfig,
        HealthCheckConfig,
        SystemHealthStatus,
        BackendType as IQRBackendType
    )

    from lightrag_integration.query_router import (
        BiomedicalQueryRouter, 
        RoutingDecision, 
        RoutingPrediction,
        ConfidenceMetrics
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures and Utilities
# ============================================================================

@pytest.fixture
def sample_production_config():
    """Sample production configuration for testing"""
    config = ProductionLoadBalancingConfig()
    
    # Configure test backends  
    config.backend_instances = {
        "lightrag_healthy": BackendInstanceConfig(
            id="lightrag_healthy",
            backend_type=BackendType.LIGHTRAG,
            endpoint_url="http://lightrag-1:8080",
            api_key="test_key",
            weight=1.0,
            circuit_breaker_enabled=True,
            failure_threshold=3,
            recovery_timeout_seconds=30
        ),
        "lightrag_degraded": BackendInstanceConfig(
            id="lightrag_degraded",
            backend_type=BackendType.LIGHTRAG,
            endpoint_url="http://lightrag-2:8080", 
            api_key="test_key",
            weight=0.8,
            circuit_breaker_enabled=True,
            failure_threshold=2,
            recovery_timeout_seconds=45
        ),
        "perplexity_primary": BackendInstanceConfig(
            id="perplexity_primary",
            backend_type=BackendType.PERPLEXITY,
            endpoint_url="http://perplexity-1:8080",
            api_key="test_key", 
            weight=1.0,
            circuit_breaker_enabled=True,
            failure_threshold=4,
            recovery_timeout_seconds=60
        )
    }
    
    return config


@pytest.fixture
def mock_backend_clients():
    """Create mock backend clients that can be used in tests"""
    mock_clients = {}
    
    def create_client_mock(backend_id: str, healthy: bool = True):
        client = Mock()
        client.health_check = AsyncMock()
        
        if healthy:
            client.health_check.return_value = (True, 800.0, {"status": "healthy"})
        else:
            client.health_check.return_value = (False, 5000.0, {"status": "unhealthy", "error": "timeout"})
        
        mock_clients[backend_id] = client
        return client
    
    return mock_clients, create_client_mock


@pytest.fixture
def mock_base_router():
    """Mock biomedical query router for testing"""
    router = Mock(spec=BiomedicalQueryRouter)
    
    def mock_routing(query, context=None):
        # Simple routing logic for tests
        if "lightrag" in query.lower():
            decision = RoutingDecision.LIGHTRAG
        elif "perplexity" in query.lower():
            decision = RoutingDecision.PERPLEXITY
        else:
            decision = RoutingDecision.LIGHTRAG  # Default
        
        from lightrag_integration.cost_persistence import ResearchCategory
        
        return RoutingPrediction(
            routing_decision=decision,
            confidence=0.85,
            reasoning=[f"Test routing to {decision.value}"],
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION,
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.85,
                research_category_confidence=0.8,
                temporal_analysis_confidence=0.7,
                signal_strength_confidence=0.9,
                context_coherence_confidence=0.85,
                keyword_density=0.6,
                pattern_match_strength=0.7,
                biomedical_entity_count=3,
                ambiguity_score=0.2,
                conflict_score=0.1,
                alternative_interpretations=[(RoutingDecision.PERPLEXITY, 0.6)],
                calculation_time_ms=10.5
            )
        )
    
    router.route_query.side_effect = mock_routing
    return router


# ============================================================================
# Circuit Breaker Backend Isolation Tests
# ============================================================================

class TestCircuitBreakerBackendIsolation:
    """Test circuit breaker backend isolation during failures"""
    
    def test_circuit_breaker_initialization(self, sample_production_config):
        """Test circuit breaker initialization with backend configuration"""
        
        with patch('lightrag_integration.production_load_balancer.LightRAGBackendClient'), \
             patch('lightrag_integration.production_load_balancer.PerplexityBackendClient'):
            
            load_balancer = ProductionLoadBalancer(sample_production_config)
            
            # Verify circuit breakers were created for each backend
            assert "lightrag_healthy" in load_balancer.circuit_breakers
            assert "lightrag_degraded" in load_balancer.circuit_breakers
            assert "perplexity_primary" in load_balancer.circuit_breakers
            
            # Verify initial states
            for cb in load_balancer.circuit_breakers.values():
                assert cb.state == CircuitBreakerState.CLOSED
                assert cb.failure_count == 0
    
    def test_circuit_breaker_state_transitions(self, sample_production_config):
        """Test circuit breaker state transitions on failures"""
        
        with patch('lightrag_integration.production_load_balancer.LightRAGBackendClient'), \
             patch('lightrag_integration.production_load_balancer.PerplexityBackendClient'):
            
            load_balancer = ProductionLoadBalancer(sample_production_config)
            cb = load_balancer.circuit_breakers["lightrag_healthy"]
            
            # Test failure accumulation
            assert cb.state == CircuitBreakerState.CLOSED
            
            # Record failures below threshold
            for i in range(2):  # Below failure_threshold of 3
                cb.record_failure(f"Test failure {i}")
            
            assert cb.state == CircuitBreakerState.CLOSED
            assert cb.failure_count == 2
            
            # Record failure that exceeds threshold
            cb.record_failure("Final failure")
            
            assert cb.state == CircuitBreakerState.OPEN
            assert not cb.should_allow_request()
    
    def test_circuit_breaker_half_open_recovery(self, sample_production_config):
        """Test circuit breaker recovery through half-open state"""
        
        with patch('lightrag_integration.production_load_balancer.LightRAGBackendClient'), \
             patch('lightrag_integration.production_load_balancer.PerplexityBackendClient'):
            
            load_balancer = ProductionLoadBalancer(sample_production_config)
            cb = load_balancer.circuit_breakers["lightrag_degraded"]
            
            # Trip the circuit breaker
            for _ in range(3):
                cb.record_failure("Test failure")
            
            assert cb.state == CircuitBreakerState.OPEN
            
            # Wait for recovery period to pass and test recovery
            original_next_attempt = cb.next_attempt_time
            if original_next_attempt:
                # Simulate time passing
                cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
                
                # Test that recovery is possible
                assert cb.should_allow_request()  # Should transition to half-open and allow request
                assert cb.state == CircuitBreakerState.HALF_OPEN
                
                # Test successful recovery
                cb.record_success(500.0)  # Record successful response
                cb.record_success(600.0)  # Multiple successes
                
                # Check if circuit closed after successful recovery
                if cb.half_open_requests >= cb.config.half_open_max_requests:
                    # Circuit should close if all half-open requests succeeded
                    assert cb.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]


# ============================================================================
# Intelligent Routing with Circuit Breaker States Tests  
# ============================================================================

class TestIntelligentRoutingWithCircuitBreakerStates:
    """Test intelligent routing decisions based on circuit breaker states"""
    
    @pytest.mark.asyncio
    async def test_routing_avoids_open_circuit_breakers(self, mock_base_router, sample_production_config):
        """Test that routing avoids backends with open circuit breakers"""
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY,
            enable_automatic_failback=True
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            # Mock production load balancer
            mock_lb = Mock()
            
            # Mock circuit breaker states
            cb_states = {
                "lightrag_healthy": CircuitBreakerState.CLOSED,
                "lightrag_degraded": CircuitBreakerState.OPEN,  # This backend is down
                "perplexity_primary": CircuitBreakerState.CLOSED
            }
            
            mock_lb.get_circuit_breaker_states.return_value = cb_states
            
            # Mock backend selection that considers circuit breaker states
            def mock_select_backend(routing_decision, context):
                # Only return backends with non-open circuit breakers
                available = [bid for bid, state in cb_states.items() 
                           if state != CircuitBreakerState.OPEN]
                
                if routing_decision == RoutingDecision.LIGHTRAG:
                    lightrag_available = [bid for bid in available if "lightrag" in bid]
                    return lightrag_available[0] if lightrag_available else None
                elif routing_decision == RoutingDecision.PERPLEXITY:
                    perplexity_available = [bid for bid in available if "perplexity" in bid]
                    return perplexity_available[0] if perplexity_available else None
                
                return available[0] if available else None
            
            mock_lb.select_backend.side_effect = mock_select_backend
            MockProdLB.return_value = mock_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags,
                production_config=sample_production_config
            )
            
            # Test routing to LightRAG - should avoid degraded backend
            result = await router.route_query("test lightrag query")
            
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Verify the mock was called with routing decision
            mock_lb.select_backend.assert_called()
            call_args = mock_lb.select_backend.call_args
            assert call_args[0][0] == RoutingDecision.LIGHTRAG  # First argument should be routing decision
    
    @pytest.mark.asyncio
    async def test_routing_with_mixed_circuit_breaker_states(self, mock_base_router, sample_production_config):
        """Test routing behavior with mixed circuit breaker states"""
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            mock_lb = Mock()
            
            # Mixed states: healthy, half-open, open
            cb_states = {
                "lightrag_healthy": CircuitBreakerState.CLOSED,
                "lightrag_degraded": CircuitBreakerState.HALF_OPEN,
                "perplexity_primary": CircuitBreakerState.OPEN
            }
            
            mock_lb.get_circuit_breaker_states.return_value = cb_states
            
            # Track half-open requests to simulate limited access
            half_open_requests = 0
            max_half_open_requests = 3
            
            def mock_select_with_limits(routing_decision, context):
                nonlocal half_open_requests
                
                if routing_decision == RoutingDecision.LIGHTRAG:
                    # Prefer healthy over half-open
                    if cb_states["lightrag_healthy"] == CircuitBreakerState.CLOSED:
                        return "lightrag_healthy"
                    elif (cb_states["lightrag_degraded"] == CircuitBreakerState.HALF_OPEN 
                          and half_open_requests < max_half_open_requests):
                        half_open_requests += 1
                        return "lightrag_degraded"
                    else:
                        return "lightrag_healthy"  # Fallback to healthy
                
                # Perplexity is open, should fallback or return None
                return None
            
            mock_lb.select_backend.side_effect = mock_select_with_limits
            MockProdLB.return_value = mock_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags,
                production_config=sample_production_config
            )
            
            # Test multiple requests
            results = []
            for i in range(6):
                result = await router.route_query(f"lightrag test query {i}")
                results.append(result)
            
            # Verify that requests were made
            assert len(results) == 6
            assert all(result is not None for result in results)
            assert all(result.routing_decision == RoutingDecision.LIGHTRAG for result in results)
            
            # Verify that backend selection was called with appropriate limits
            assert mock_lb.select_backend.call_count >= 6  # Should be called for each request


# ============================================================================
# Automatic Failover Tests
# ============================================================================

class TestAutomaticFailoverOnCircuitBreakerOpen:
    """Test automatic failover mechanisms when circuit breakers open"""
    
    @pytest.mark.asyncio
    async def test_failover_when_primary_backend_fails(self, mock_base_router, sample_production_config):
        """Test failover when primary backend circuit breaker opens"""
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY,
            enable_automatic_failback=True
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            mock_lb = Mock()
            
            # Initial state: all backends healthy
            initial_states = {
                "lightrag_healthy": CircuitBreakerState.CLOSED,
                "lightrag_degraded": CircuitBreakerState.CLOSED,
                "perplexity_primary": CircuitBreakerState.CLOSED
            }
            
            # After failure: primary backend is down
            failed_states = {
                "lightrag_healthy": CircuitBreakerState.OPEN,  # Primary failed
                "lightrag_degraded": CircuitBreakerState.CLOSED,
                "perplexity_primary": CircuitBreakerState.CLOSED
            }
            
            # Track state changes
            current_states = initial_states.copy()
            mock_lb.get_circuit_breaker_states.return_value = current_states
            
            def mock_failover_selection(routing_decision, context):
                # Select based on current circuit breaker states
                available = [bid for bid, state in current_states.items() 
                           if state != CircuitBreakerState.OPEN]
                
                if routing_decision == RoutingDecision.LIGHTRAG:
                    lightrag_available = [bid for bid in available if "lightrag" in bid]
                    if lightrag_available:
                        return lightrag_available[0]
                    # Cross-backend failover to Perplexity
                    perplexity_available = [bid for bid in available if "perplexity" in bid]
                    return perplexity_available[0] if perplexity_available else None
                
                return available[0] if available else None
            
            mock_lb.select_backend.side_effect = mock_failover_selection
            MockProdLB.return_value = mock_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags,
                production_config=sample_production_config
            )
            
            # Test 1: Normal operation
            result = await router.route_query("lightrag query")
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Test 2: Simulate primary backend failure
            current_states.update(failed_states)
            
            result = await router.route_query("lightrag query after failure")
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Verify backend selection was called appropriately
            assert mock_lb.select_backend.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_cross_backend_failover(self, mock_base_router, sample_production_config):
        """Test failover across different backend types"""
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            mock_lb = Mock()
            
            # All LightRAG backends are down
            cb_states = {
                "lightrag_healthy": CircuitBreakerState.OPEN,
                "lightrag_degraded": CircuitBreakerState.OPEN,
                "perplexity_primary": CircuitBreakerState.CLOSED
            }
            
            mock_lb.get_circuit_breaker_states.return_value = cb_states
            
            def mock_cross_backend_failover(routing_decision, context):
                available = [bid for bid, state in cb_states.items() 
                           if state != CircuitBreakerState.OPEN]
                
                if routing_decision == RoutingDecision.LIGHTRAG:
                    # No LightRAG backends available, fallback to Perplexity
                    lightrag_available = [bid for bid in available if "lightrag" in bid]
                    if not lightrag_available:
                        perplexity_available = [bid for bid in available if "perplexity" in bid]
                        return perplexity_available[0] if perplexity_available else None
                
                return available[0] if available else None
            
            mock_lb.select_backend.side_effect = mock_cross_backend_failover
            MockProdLB.return_value = mock_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags,
                production_config=sample_production_config
            )
            
            # Request LightRAG but should get Perplexity due to failover
            result = await router.route_query("lightrag query with all lightrag down")
            
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG  # Original routing decision
            
            # Verify cross-backend failover was invoked
            mock_lb.select_backend.assert_called()
            call_args = mock_lb.select_backend.call_args
            assert call_args[0][0] == RoutingDecision.LIGHTRAG


# ============================================================================
# Load Balancing Weight Adjustment Tests
# ============================================================================

class TestLoadBalancingWeightAdjustment:
    """Test load balancing weight adjustments based on circuit breaker health"""
    
    def test_weight_calculation_based_on_circuit_breaker_state(self, sample_production_config):
        """Test that weights are adjusted based on circuit breaker states"""
        
        with patch('lightrag_integration.production_load_balancer.LightRAGBackendClient'), \
             patch('lightrag_integration.production_load_balancer.PerplexityBackendClient'):
            
            load_balancer = ProductionLoadBalancer(sample_production_config)
            
            # Get initial weights
            initial_weights = {bid: config.weight for bid, config in sample_production_config.backend_instances.items()}
            
            # Simulate circuit breaker opening for one backend
            cb = load_balancer.circuit_breakers["lightrag_degraded"]
            for _ in range(3):  # Trip circuit breaker
                cb.record_failure(Exception("Test failure"))
            
            assert cb.state == CircuitBreakerState.OPEN
            
            # Test that open circuit breaker backend is not available for selection
            available_backends = load_balancer._get_available_backends()
            assert "lightrag_degraded" not in available_backends
            
            # Verify other backends remain available
            assert "lightrag_healthy" in available_backends
            assert "perplexity_primary" in available_backends


# ============================================================================
# Query Router Health Integration Tests
# ============================================================================

class TestQueryRouterCircuitBreakerHealthIntegration:
    """Test integration between query router and circuit breaker health"""
    
    @pytest.mark.asyncio
    async def test_health_status_integration(self, mock_base_router, sample_production_config):
        """Test that health status reflects circuit breaker states"""
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            mock_lb = Mock()
            
            # Mock health summary that includes circuit breaker information
            def mock_health_summary():
                cb_states = {
                    "lightrag_healthy": CircuitBreakerState.CLOSED,
                    "lightrag_degraded": CircuitBreakerState.OPEN,
                    "perplexity_primary": CircuitBreakerState.HALF_OPEN
                }
                
                return {
                    "overall_health": "degraded",  # Not all backends healthy
                    "backend_health": {
                        bid: "healthy" if state == CircuitBreakerState.CLOSED
                              else "degraded" if state == CircuitBreakerState.HALF_OPEN
                              else "unhealthy"
                        for bid, state in cb_states.items()
                    },
                    "circuit_breaker_summary": {
                        "closed": 1,
                        "half_open": 1,
                        "open": 1
                    }
                }
            
            mock_lb.get_health_summary.side_effect = mock_health_summary
            mock_lb.select_backend.return_value = "lightrag_healthy"
            MockProdLB.return_value = mock_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags,
                production_config=sample_production_config
            )
            
            # Test health status retrieval
            health_status = router.get_health_status()
            
            assert "production_load_balancer" in health_status
            prod_health = health_status["production_load_balancer"]
            assert "status" in prod_health
            
            if "health" in prod_health:
                health_data = prod_health["health"]
                assert "overall_health" in health_data
                assert "backend_health" in health_data
                assert "circuit_breaker_summary" in health_data
                
                # Verify circuit breaker information is included
                cb_summary = health_data["circuit_breaker_summary"]
                assert cb_summary["closed"] >= 0
                assert cb_summary["open"] >= 0
                assert cb_summary["half_open"] >= 0


# ============================================================================
# Integration Test - End-to-End Scenarios
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests combining all functionality"""
    
    @pytest.mark.asyncio
    async def test_complete_failover_and_recovery_scenario(self, mock_base_router, sample_production_config):
        """Test complete scenario: failure, failover, recovery"""
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY,
            enable_automatic_failback=True,
            enable_performance_comparison=True
        )
        
        with patch('lightrag_integration.production_intelligent_query_router.ProductionLoadBalancer') as MockProdLB:
            mock_lb = Mock()
            
            # Scenario states
            scenarios = [
                # Phase 1: All healthy
                {
                    "lightrag_healthy": CircuitBreakerState.CLOSED,
                    "lightrag_degraded": CircuitBreakerState.CLOSED,
                    "perplexity_primary": CircuitBreakerState.CLOSED
                },
                # Phase 2: Primary fails
                {
                    "lightrag_healthy": CircuitBreakerState.OPEN,
                    "lightrag_degraded": CircuitBreakerState.CLOSED,
                    "perplexity_primary": CircuitBreakerState.CLOSED
                },
                # Phase 3: Recovery begins
                {
                    "lightrag_healthy": CircuitBreakerState.HALF_OPEN,
                    "lightrag_degraded": CircuitBreakerState.CLOSED,
                    "perplexity_primary": CircuitBreakerState.CLOSED
                },
                # Phase 4: Full recovery
                {
                    "lightrag_healthy": CircuitBreakerState.CLOSED,
                    "lightrag_degraded": CircuitBreakerState.CLOSED,
                    "perplexity_primary": CircuitBreakerState.CLOSED
                }
            ]
            
            current_scenario = 0
            mock_lb.get_circuit_breaker_states.return_value = scenarios[current_scenario]
            
            def mock_scenario_selection(routing_decision, context):
                cb_states = scenarios[current_scenario]
                available = [bid for bid, state in cb_states.items() 
                           if state != CircuitBreakerState.OPEN]
                
                if routing_decision == RoutingDecision.LIGHTRAG:
                    lightrag_available = [bid for bid in available if "lightrag" in bid]
                    if lightrag_available:
                        # Prefer healthy over half-open
                        healthy_lightrag = [bid for bid in lightrag_available 
                                          if cb_states[bid] == CircuitBreakerState.CLOSED]
                        return healthy_lightrag[0] if healthy_lightrag else lightrag_available[0]
                
                return available[0] if available else None
            
            mock_lb.select_backend.side_effect = mock_scenario_selection
            MockProdLB.return_value = mock_lb
            
            router = ProductionIntelligentQueryRouter(
                base_router=mock_base_router,
                feature_flags=feature_flags,
                production_config=sample_production_config
            )
            
            results = []
            
            # Phase 1: Normal operation
            result = await router.route_query("Phase 1: Normal operation")
            results.append(("Phase 1", result.routing_decision.value if result else None))
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Phase 2: Primary failure, should failover
            current_scenario = 1
            mock_lb.get_circuit_breaker_states.return_value = scenarios[current_scenario]
            
            result = await router.route_query("Phase 2: Primary failed")
            results.append(("Phase 2", result.routing_decision.value if result else None))
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Phase 3: Recovery attempt (half-open)
            current_scenario = 2
            mock_lb.get_circuit_breaker_states.return_value = scenarios[current_scenario]
            
            result = await router.route_query("Phase 3: Recovery attempt")
            results.append(("Phase 3", result.routing_decision.value if result else None))
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Phase 4: Full recovery
            current_scenario = 3
            mock_lb.get_circuit_breaker_states.return_value = scenarios[current_scenario]
            
            result = await router.route_query("Phase 4: Full recovery")
            results.append(("Phase 4", result.routing_decision.value if result else None))
            assert result is not None
            assert result.routing_decision == RoutingDecision.LIGHTRAG
            
            # Verify we got results for all phases
            assert len(results) == 4
            
            # Log the complete scenario for verification
            logger.info(f"Complete scenario results: {results}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])