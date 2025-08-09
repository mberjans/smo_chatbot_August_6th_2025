"""
End-to-end workflow tests for ProductionCircuitBreaker integration.

This module provides comprehensive end-to-end testing of circuit breaker
functionality within complete query processing workflows, including:
- Complete query processing with circuit breaker protection
- Fallback system coordination during failures
- Recovery workflows after service restoration
- Multi-service failure and recovery scenarios
"""

import pytest
import asyncio
import time
import random
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
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
    create_default_production_config
)


# ============================================================================
# E2E Test Fixtures and Setup
# ============================================================================

@pytest.fixture
async def e2e_production_config():
    """Create comprehensive production configuration for E2E testing"""
    return ProductionLoadBalancingConfig(
        strategy=LoadBalancingStrategy.ADAPTIVE_LEARNING,
        enable_adaptive_routing=True,
        enable_cost_optimization=True,
        enable_quality_based_routing=True,
        enable_real_time_monitoring=True,
        global_circuit_breaker_enabled=True,
        cascade_failure_prevention=True,
        
        backend_instances={
            "lightrag_primary": BackendInstanceConfig(
                id="lightrag_primary",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url="http://lightrag-primary:8080",
                api_key="lightrag_primary_key",
                weight=2.0,
                cost_per_1k_tokens=0.05,
                max_requests_per_minute=200,
                timeout_seconds=20.0,
                priority=1,
                expected_response_time_ms=800.0,
                quality_score=0.95,
                reliability_score=0.98,
                failure_threshold=5,
                recovery_timeout_seconds=30,
                half_open_max_requests=10,
                circuit_breaker_enabled=True
            ),
            
            "lightrag_secondary": BackendInstanceConfig(
                id="lightrag_secondary",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url="http://lightrag-secondary:8080",
                api_key="lightrag_secondary_key",
                weight=1.5,
                cost_per_1k_tokens=0.05,
                max_requests_per_minute=150,
                timeout_seconds=25.0,
                priority=2,
                expected_response_time_ms=1000.0,
                quality_score=0.90,
                reliability_score=0.95,
                failure_threshold=4,
                recovery_timeout_seconds=45,
                circuit_breaker_enabled=True
            ),
            
            "perplexity_primary": BackendInstanceConfig(
                id="perplexity_primary",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.perplexity.ai",
                api_key="perplexity_primary_key",
                weight=1.0,
                cost_per_1k_tokens=0.20,
                max_requests_per_minute=100,
                timeout_seconds=30.0,
                priority=3,
                expected_response_time_ms=2000.0,
                quality_score=0.85,
                reliability_score=0.90,
                failure_threshold=3,
                recovery_timeout_seconds=60,
                circuit_breaker_enabled=True
            ),
            
            "perplexity_fallback": BackendInstanceConfig(
                id="perplexity_fallback",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.perplexity.ai",
                api_key="perplexity_fallback_key",
                weight=0.8,
                cost_per_1k_tokens=0.25,
                max_requests_per_minute=50,
                timeout_seconds=35.0,
                priority=4,
                expected_response_time_ms=2500.0,
                quality_score=0.80,
                reliability_score=0.85,
                failure_threshold=2,
                recovery_timeout_seconds=90,
                circuit_breaker_enabled=True
            ),
            
            "cache_primary": BackendInstanceConfig(
                id="cache_primary",
                backend_type=BackendType.CACHE,
                endpoint_url="redis://cache-primary:6379",
                api_key="",
                weight=0.5,
                cost_per_1k_tokens=0.0,
                max_requests_per_minute=1000,
                timeout_seconds=5.0,
                priority=0,  # Highest priority for cache
                expected_response_time_ms=50.0,
                quality_score=0.70,  # Lower quality but very fast
                reliability_score=0.99,
                failure_threshold=10,
                recovery_timeout_seconds=10,
                circuit_breaker_enabled=True
            )
        }
    )

@pytest.fixture
async def e2e_load_balancer(e2e_production_config):
    """Create fully configured production load balancer for E2E testing"""
    with patch.multiple(
        'production_load_balancer.ProductionLoadBalancer',
        _initialize_backend_clients=AsyncMock(),
        _start_monitoring_tasks=AsyncMock(),
        _initialize_metrics_collection=AsyncMock(),
        _initialize_cost_tracking=AsyncMock(),
        _initialize_alert_system=AsyncMock()
    ):
        lb = ProductionLoadBalancer(e2e_production_config)
        await lb.initialize()
        
        # Mock backend clients for testing
        for backend_id in lb.config.backend_instances.keys():
            mock_client = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client.health_check = AsyncMock(return_value=(True, 100.0, {}))
            lb.backend_clients[backend_id] = mock_client
        
        return lb

@pytest.fixture
def mock_query_scenarios():
    """Mock query scenarios for testing different response patterns"""
    return {
        "successful_lightrag": {
            "query": "What are the key biomarkers for metabolic syndrome?",
            "expected_backend": "lightrag_primary",
            "response": {
                "content": "Key biomarkers for metabolic syndrome include elevated glucose, insulin resistance markers, lipid profile changes, and inflammatory cytokines.",
                "confidence": 0.95,
                "tokens_used": 150,
                "response_time_ms": 750
            }
        },
        
        "fallback_to_perplexity": {
            "query": "Latest research on CRISPR gene editing in 2025",
            "expected_backend": "perplexity_primary",
            "response": {
                "content": "Recent advances in CRISPR technology include improved precision, reduced off-target effects, and new applications in therapeutic interventions.",
                "confidence": 0.88,
                "tokens_used": 200,
                "response_time_ms": 1800
            }
        },
        
        "cache_hit": {
            "query": "What is diabetes?",
            "expected_backend": "cache_primary",
            "response": {
                "content": "Diabetes is a group of metabolic disorders characterized by high blood sugar levels.",
                "confidence": 0.85,
                "tokens_used": 50,
                "response_time_ms": 25,
                "cached": True
            }
        },
        
        "complex_medical_query": {
            "query": "Explain the molecular mechanisms of insulin signaling in hepatocytes and how they are disrupted in type 2 diabetes",
            "expected_backend": "lightrag_primary",
            "response": {
                "content": "Insulin signaling in hepatocytes involves complex molecular pathways including IRS proteins, PI3K/Akt pathway, and regulation of gluconeogenesis and glycogen synthesis.",
                "confidence": 0.92,
                "tokens_used": 300,
                "response_time_ms": 1200
            }
        }
    }


# ============================================================================
# Complete Query Processing E2E Tests
# ============================================================================

class TestCompleteQueryProcessingWorkflows:
    """Test complete query processing workflows with circuit breaker protection"""

    @pytest.mark.asyncio
    async def test_successful_query_processing_flow(self, e2e_load_balancer, mock_query_scenarios):
        """Test successful end-to-end query processing"""
        lb = e2e_load_balancer
        scenario = mock_query_scenarios["successful_lightrag"]
        
        # Mock successful response from LightRAG primary
        mock_client = lb.backend_clients["lightrag_primary"]
        mock_client.query.return_value = scenario["response"]
        
        # Execute query
        result = await lb.query(scenario["query"])
        
        # Verify successful processing
        assert result is not None
        assert result["content"] == scenario["response"]["content"]
        assert result["confidence"] == scenario["response"]["confidence"]
        
        # Verify circuit breaker state remains healthy
        cb = lb.circuit_breakers["lightrag_primary"]
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.success_count > 0
        
        # Verify metrics were recorded
        metrics = cb.get_metrics()
        assert metrics["success_count"] == 1
        assert metrics["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_cache_first_then_fallback_flow(self, e2e_load_balancer, mock_query_scenarios):
        """Test cache-first strategy with fallback to primary services"""
        lb = e2e_load_balancer
        scenario = mock_query_scenarios["fallback_to_perplexity"]
        
        # Mock cache miss
        cache_client = lb.backend_clients["cache_primary"]
        cache_client.query.side_effect = KeyError("Cache miss")
        
        # Mock successful response from Perplexity
        perplexity_client = lb.backend_clients["perplexity_primary"]
        perplexity_client.query.return_value = scenario["response"]
        
        # Execute query
        result = await lb.query(scenario["query"])
        
        # Verify fallback worked
        assert result is not None
        assert result["content"] == scenario["response"]["content"]
        
        # Verify cache circuit breaker recorded the miss appropriately
        cache_cb = lb.circuit_breakers["cache_primary"]
        assert cache_cb.state == CircuitBreakerState.CLOSED  # Cache misses shouldn't open circuit
        
        # Verify primary service circuit breaker recorded success
        perplexity_cb = lb.circuit_breakers["perplexity_primary"]
        assert perplexity_cb.success_count > 0

    @pytest.mark.asyncio
    async def test_primary_service_failure_with_automatic_fallback(self, e2e_load_balancer):
        """Test automatic fallback when primary service fails"""
        lb = e2e_load_balancer
        
        # Mock primary LightRAG failure
        primary_client = lb.backend_clients["lightrag_primary"]
        primary_client.query.side_effect = TimeoutError("Primary service timeout")
        
        # Mock successful secondary LightRAG response
        secondary_client = lb.backend_clients["lightrag_secondary"]
        secondary_client.query.return_value = {
            "content": "Fallback response from secondary LightRAG",
            "confidence": 0.87,
            "tokens_used": 180,
            "response_time_ms": 950
        }
        
        # Execute query
        result = await lb.query("Test query for fallback")
        
        # Verify fallback occurred
        assert result is not None
        assert "secondary" in result["content"] or result["confidence"] == 0.87
        
        # Verify primary circuit breaker recorded failure
        primary_cb = lb.circuit_breakers["lightrag_primary"]
        assert primary_cb.failure_count > 0
        
        # Verify secondary circuit breaker recorded success
        secondary_cb = lb.circuit_breakers["lightrag_secondary"]
        assert secondary_cb.success_count > 0

    @pytest.mark.asyncio
    async def test_complex_medical_query_processing(self, e2e_load_balancer, mock_query_scenarios):
        """Test processing of complex medical queries with specialized routing"""
        lb = e2e_load_balancer
        scenario = mock_query_scenarios["complex_medical_query"]
        
        # Mock specialized LightRAG response for complex medical query
        lightrag_client = lb.backend_clients["lightrag_primary"]
        lightrag_client.query.return_value = scenario["response"]
        
        # Execute complex query
        result = await lb.query(scenario["query"])
        
        # Verify specialized handling
        assert result is not None
        assert result["confidence"] >= 0.90  # Should have high confidence for specialized knowledge
        assert result["tokens_used"] >= 250   # Complex queries use more tokens
        
        # Verify appropriate backend was selected
        cb = lb.circuit_breakers["lightrag_primary"]
        assert cb.success_count > 0
        
        # Verify response time was reasonable for complex query
        metrics = cb.get_metrics()
        assert metrics["avg_response_time_ms"] <= 2000  # Should be processed efficiently

    @pytest.mark.asyncio
    async def test_multi_stage_query_processing_workflow(self, e2e_load_balancer):
        """Test multi-stage query processing with different service types"""
        lb = e2e_load_balancer
        
        queries = [
            "What is metabolomics?",           # Simple query - should go to cache
            "Latest metabolomics research",    # Recent research - should go to Perplexity
            "Metabolomics data analysis methods", # Specialized - should go to LightRAG
        ]
        
        # Mock responses for different stages
        lb.backend_clients["cache_primary"].query.return_value = {
            "content": "Metabolomics is the study of metabolites", 
            "cached": True, "response_time_ms": 30
        }
        
        lb.backend_clients["perplexity_primary"].query.return_value = {
            "content": "Recent advances in metabolomics include...",
            "response_time_ms": 1900
        }
        
        lb.backend_clients["lightrag_primary"].query.return_value = {
            "content": "Advanced metabolomics analysis methods include...",
            "response_time_ms": 800
        }
        
        results = []
        for query in queries:
            result = await lb.query(query)
            results.append(result)
        
        # Verify all queries were processed
        assert len(results) == 3
        assert all(result is not None for result in results)
        
        # Verify different services were used appropriately
        cache_cb = lb.circuit_breakers["cache_primary"]
        perplexity_cb = lb.circuit_breakers["perplexity_primary"] 
        lightrag_cb = lb.circuit_breakers["lightrag_primary"]
        
        # At least one service should have recorded activity
        total_successes = (cache_cb.success_count + 
                          perplexity_cb.success_count + 
                          lightrag_cb.success_count)
        assert total_successes >= len(queries)


# ============================================================================
# Fallback System Coordination E2E Tests
# ============================================================================

class TestFallbackSystemCoordination:
    """Test coordination between circuit breakers and fallback systems"""

    @pytest.mark.asyncio
    async def test_cascade_prevention_during_primary_failure(self, e2e_load_balancer):
        """Test cascade prevention when primary services fail"""
        lb = e2e_load_balancer
        
        # Simulate primary LightRAG service failing
        primary_cb = lb.circuit_breakers["lightrag_primary"]
        for i in range(primary_cb.config.failure_threshold):
            primary_cb.record_failure(f"Primary cascade test {i}", error_type="ServiceFailure")
        
        assert primary_cb.state == CircuitBreakerState.OPEN
        
        # Mock secondary services to handle increased load
        lb.backend_clients["lightrag_secondary"].query.return_value = {
            "content": "Secondary handling increased load",
            "response_time_ms": 1200  # Slightly slower due to load
        }
        
        lb.backend_clients["perplexity_primary"].query.return_value = {
            "content": "Perplexity handling overflow",
            "response_time_ms": 2200
        }
        
        # Execute multiple queries to test fallback coordination
        queries = [f"Test query {i}" for i in range(10)]
        results = []
        
        for query in queries:
            result = await lb.query(query)
            results.append(result)
        
        # Verify all queries were handled despite primary failure
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) >= 8  # Most should succeed via fallback
        
        # Verify fallback services handled the load without cascading failure
        secondary_cb = lb.circuit_breakers["lightrag_secondary"]
        perplexity_cb = lb.circuit_breakers["perplexity_primary"]
        
        # Neither fallback should have opened (good cascade prevention)
        assert secondary_cb.state == CircuitBreakerState.CLOSED
        assert perplexity_cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_intelligent_fallback_routing_during_partial_failures(self, e2e_load_balancer):
        """Test intelligent routing when some services are degraded"""
        lb = e2e_load_balancer
        
        # Simulate partial degradation of LightRAG primary (slow but not failed)
        lightrag_primary_cb = lb.circuit_breakers["lightrag_primary"]
        for i in range(5):
            if i % 2 == 0:
                lightrag_primary_cb.record_success(3000.0)  # Very slow responses
            else:
                lightrag_primary_cb.record_failure(f"Intermittent failure {i}")
        
        # Perplexity primary experiencing rate limits
        perplexity_cb = lb.circuit_breakers["perplexity_primary"]
        for i in range(2):
            perplexity_cb.record_failure("Rate limit exceeded", error_type="RateLimitError")
        
        # Mock healthy secondary services
        lb.backend_clients["lightrag_secondary"].query.return_value = {
            "content": "Healthy secondary response",
            "response_time_ms": 900
        }
        
        lb.backend_clients["perplexity_fallback"].query.return_value = {
            "content": "Fallback Perplexity response", 
            "response_time_ms": 2400
        }
        
        # Execute queries and verify intelligent routing
        results = []
        for i in range(5):
            result = await lb.query(f"Intelligent routing test {i}")
            results.append(result)
        
        # Verify most queries succeeded
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) >= 4
        
        # Verify secondary services were preferred over degraded primary services
        secondary_cb = lb.circuit_breakers["lightrag_secondary"]
        fallback_cb = lb.circuit_breakers["perplexity_fallback"]
        
        # At least one secondary service should have handled requests
        assert secondary_cb.success_count > 0 or fallback_cb.success_count > 0

    @pytest.mark.asyncio
    async def test_emergency_cache_activation(self, e2e_load_balancer):
        """Test emergency cache activation when all primary services fail"""
        lb = e2e_load_balancer
        
        # Simulate failure of all primary services
        primary_services = ["lightrag_primary", "lightrag_secondary", "perplexity_primary"]
        
        for service_id in primary_services:
            cb = lb.circuit_breakers[service_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Emergency scenario failure {i}")
            assert cb.state == CircuitBreakerState.OPEN
        
        # Mock cache returning emergency responses
        cache_client = lb.backend_clients["cache_primary"]
        cache_client.query.return_value = {
            "content": "Emergency cached response - limited accuracy",
            "confidence": 0.60,  # Lower confidence for emergency cache
            "response_time_ms": 45,
            "emergency_mode": True
        }
        
        # Execute query in emergency scenario
        result = await lb.query("Emergency test query")
        
        # Verify emergency cache was activated
        assert result is not None
        assert "emergency" in result["content"].lower() or result.get("emergency_mode")
        
        # Verify cache circuit breaker handled the emergency load
        cache_cb = lb.circuit_breakers["cache_primary"]
        assert cache_cb.success_count > 0
        assert cache_cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_reduced_functionality(self, e2e_load_balancer):
        """Test graceful degradation when multiple services are unavailable"""
        lb = e2e_load_balancer
        
        # Simulate various levels of service degradation
        degradation_scenarios = {
            "lightrag_primary": CircuitBreakerState.OPEN,    # Complete failure
            "lightrag_secondary": CircuitBreakerState.OPEN,  # Complete failure  
            "perplexity_primary": CircuitBreakerState.HALF_OPEN,  # Partial recovery
            "perplexity_fallback": CircuitBreakerState.CLOSED,    # Healthy
            "cache_primary": CircuitBreakerState.CLOSED           # Healthy
        }
        
        # Set circuit breaker states
        for service_id, state in degradation_scenarios.items():
            cb = lb.circuit_breakers[service_id]
            cb.state = state
            if state == CircuitBreakerState.OPEN:
                cb.failure_count = cb.config.failure_threshold
            elif state == CircuitBreakerState.HALF_OPEN:
                cb.half_open_requests = 0
        
        # Mock limited responses from available services
        lb.backend_clients["perplexity_fallback"].query.return_value = {
            "content": "Limited response - reduced functionality",
            "confidence": 0.70,
            "response_time_ms": 2800,
            "degraded_mode": True
        }
        
        lb.backend_clients["cache_primary"].query.return_value = {
            "content": "Basic cached information",
            "confidence": 0.65,
            "response_time_ms": 50
        }
        
        # Execute query under degraded conditions
        result = await lb.query("Degradation test query")
        
        # Verify graceful degradation
        assert result is not None
        assert result["confidence"] >= 0.60  # Minimum acceptable confidence
        
        # Verify system communicated degraded state
        assert ("limited" in result["content"].lower() or 
                "basic" in result["content"].lower() or
                result.get("degraded_mode"))


# ============================================================================
# Recovery Workflow E2E Tests
# ============================================================================

class TestRecoveryWorkflows:
    """Test complete recovery workflows after service restoration"""

    @pytest.mark.asyncio
    async def test_gradual_service_recovery_workflow(self, e2e_load_balancer):
        """Test gradual recovery of services with circuit breaker coordination"""
        lb = e2e_load_balancer
        
        # Phase 1: Simulate service failures
        failed_services = ["lightrag_primary", "perplexity_primary"]
        
        for service_id in failed_services:
            cb = lb.circuit_breakers[service_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Pre-recovery failure {i}")
            assert cb.state == CircuitBreakerState.OPEN
        
        # Phase 2: Simulate time passing for recovery
        for service_id in failed_services:
            cb = lb.circuit_breakers[service_id]
            cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Phase 3: Mock services becoming healthy again
        lb.backend_clients["lightrag_primary"].query.return_value = {
            "content": "LightRAG primary recovered",
            "response_time_ms": 850
        }
        
        lb.backend_clients["perplexity_primary"].query.return_value = {
            "content": "Perplexity primary recovered",
            "response_time_ms": 1950
        }
        
        # Phase 4: Execute queries to trigger recovery testing
        recovery_queries = [f"Recovery test query {i}" for i in range(10)]
        recovery_results = []
        
        for query in recovery_queries:
            result = await lb.query(query)
            recovery_results.append(result)
            
            # Small delay to allow for half-open testing
            await asyncio.sleep(0.1)
        
        # Phase 5: Verify recovery
        successful_recoveries = [r for r in recovery_results if r is not None]
        assert len(successful_recoveries) >= 8  # Most should succeed during recovery
        
        # Verify circuit breakers recovered
        lightrag_cb = lb.circuit_breakers["lightrag_primary"]
        perplexity_cb = lb.circuit_breakers["perplexity_primary"]
        
        # Should be in CLOSED or HALF_OPEN state (recovery in progress)
        assert lightrag_cb.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]
        assert perplexity_cb.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]

    @pytest.mark.asyncio
    async def test_failed_recovery_handling(self, e2e_load_balancer):
        """Test handling of failed recovery attempts"""
        lb = e2e_load_balancer
        
        # Open circuit breaker
        cb = lb.circuit_breakers["lightrag_primary"]
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Pre-recovery failure {i}")
        assert cb.state == CircuitBreakerState.OPEN
        
        # Set for recovery attempt
        cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Mock service still failing during recovery attempt
        client = lb.backend_clients["lightrag_primary"]
        client.query.side_effect = TimeoutError("Still failing during recovery")
        
        # Mock fallback service
        fallback_client = lb.backend_clients["lightrag_secondary"]
        fallback_client.query.return_value = {
            "content": "Fallback during failed recovery",
            "response_time_ms": 1100
        }
        
        # Attempt query during failed recovery
        result = await lb.query("Failed recovery test")
        
        # Verify fallback worked during failed recovery
        assert result is not None
        assert "fallback" in result["content"].lower()
        
        # Verify circuit breaker returned to OPEN state
        assert cb.state == CircuitBreakerState.OPEN
        
        # Verify fallback service handled the request
        fallback_cb = lb.circuit_breakers["lightrag_secondary"]
        assert fallback_cb.success_count > 0

    @pytest.mark.asyncio
    async def test_coordinated_multi_service_recovery(self, e2e_load_balancer):
        """Test coordinated recovery of multiple services"""
        lb = e2e_load_balancer
        
        # Fail multiple services with different recovery times
        services_with_recovery = {
            "lightrag_primary": 1,    # Fast recovery
            "perplexity_primary": 5,  # Medium recovery  
            "lightrag_secondary": 10  # Slow recovery
        }
        
        # Open all circuit breakers
        for service_id, recovery_delay in services_with_recovery.items():
            cb = lb.circuit_breakers[service_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Multi-service failure {i}")
            assert cb.state == CircuitBreakerState.OPEN
            
            # Set different recovery times
            cb.next_attempt_time = datetime.now() + timedelta(seconds=recovery_delay)
        
        # Mock healthy responses for when services recover
        for service_id in services_with_recovery.keys():
            client = lb.backend_clients[service_id]
            client.query.return_value = {
                "content": f"Service {service_id} recovered",
                "response_time_ms": 800
            }
        
        # Test queries at different time intervals to catch staggered recovery
        recovery_timeline = []
        
        for time_offset in [2, 6, 12]:  # Test at different recovery stages
            # Simulate time passing
            current_time = datetime.now() + timedelta(seconds=time_offset)
            
            for service_id in services_with_recovery.keys():
                cb = lb.circuit_breakers[service_id]
                if current_time >= cb.next_attempt_time:
                    cb.next_attempt_time = current_time - timedelta(seconds=1)
            
            # Execute query
            result = await lb.query(f"Multi-recovery test at T+{time_offset}")
            recovery_timeline.append((time_offset, result))
        
        # Verify coordinated recovery
        successful_results = [(t, r) for t, r in recovery_timeline if r is not None]
        assert len(successful_results) == len(recovery_timeline)  # All should eventually succeed
        
        # Verify services recovered in expected order
        final_states = {}
        for service_id in services_with_recovery.keys():
            cb = lb.circuit_breakers[service_id]
            final_states[service_id] = cb.state
        
        # At least some services should have recovered
        recovered_services = [s for s, state in final_states.items() 
                            if state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]]
        assert len(recovered_services) >= 1

    @pytest.mark.asyncio
    async def test_recovery_under_continued_load(self, e2e_load_balancer):
        """Test service recovery while handling continued user load"""
        lb = e2e_load_balancer
        
        # Open circuit breaker
        cb = lb.circuit_breakers["lightrag_primary"]
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Initial failure {i}")
        assert cb.state == CircuitBreakerState.OPEN
        
        # Set for recovery
        cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Mock primary service recovering but handling load
        primary_client = lb.backend_clients["lightrag_primary"]
        call_count = 0
        
        def mock_recovering_service(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 3:  # First few calls succeed (half-open testing)
                return {
                    "content": f"Recovering service response {call_count}",
                    "response_time_ms": 900
                }
            elif call_count <= 6:  # Next few show some stress
                return {
                    "content": f"Service under load response {call_count}",
                    "response_time_ms": 1800
                }
            else:  # Stabilizes
                return {
                    "content": f"Service stabilized response {call_count}",
                    "response_time_ms": 750
                }
        
        primary_client.query.side_effect = mock_recovering_service
        
        # Mock secondary service as backup
        secondary_client = lb.backend_clients["lightrag_secondary"]
        secondary_client.query.return_value = {
            "content": "Secondary backup during recovery",
            "response_time_ms": 1050
        }
        
        # Simulate continued load during recovery
        load_queries = [f"Load during recovery {i}" for i in range(10)]
        load_results = []
        
        for query in load_queries:
            result = await lb.query(query)
            load_results.append(result)
            await asyncio.sleep(0.05)  # Small delay between queries
        
        # Verify system handled load during recovery
        successful_load_results = [r for r in load_results if r is not None]
        assert len(successful_load_results) >= 8  # Most should succeed
        
        # Verify recovery progressed appropriately
        final_cb_state = cb.state
        assert final_cb_state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]
        
        # Verify both services were used appropriately during recovery
        primary_successes = cb.success_count
        secondary_cb = lb.circuit_breakers["lightrag_secondary"]
        secondary_successes = secondary_cb.success_count
        
        # Total successes should account for the load
        total_successes = primary_successes + secondary_successes
        assert total_successes >= len(successful_load_results)


# ============================================================================
# Multi-Service Failure and Recovery E2E Tests
# ============================================================================

class TestMultiServiceFailureRecovery:
    """Test complex multi-service failure and recovery scenarios"""

    @pytest.mark.asyncio
    async def test_system_wide_failure_and_recovery_orchestration(self, e2e_load_balancer):
        """Test orchestrated recovery from system-wide failures"""
        lb = e2e_load_balancer
        
        # Phase 1: Simulate system-wide failure
        all_services = list(lb.circuit_breakers.keys())
        failed_services = all_services[:-1]  # Leave cache as last resort
        
        for service_id in failed_services:
            cb = lb.circuit_breakers[service_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"System-wide failure {i}")
            assert cb.state == CircuitBreakerState.OPEN
        
        # Phase 2: Only cache should be available
        cache_client = lb.backend_clients["cache_primary"]
        cache_client.query.return_value = {
            "content": "Emergency cache response during system failure",
            "confidence": 0.50,
            "response_time_ms": 60,
            "emergency_mode": True
        }
        
        # Verify system can still respond using cache
        emergency_result = await lb.query("Emergency system test")
        assert emergency_result is not None
        assert emergency_result.get("emergency_mode") or emergency_result["confidence"] <= 0.60
        
        # Phase 3: Orchestrated recovery - services come back in priority order
        recovery_order = [
            ("cache_primary", 0),      # Already healthy
            ("lightrag_primary", 1),   # Highest priority service
            ("lightrag_secondary", 3), # Secondary LightRAG
            ("perplexity_primary", 5), # Primary external service
            ("perplexity_fallback", 7) # Fallback external service
        ]
        
        # Mock services returning to health
        for service_id, delay in recovery_order:
            if service_id == "cache_primary":
                continue  # Already healthy
            
            cb = lb.circuit_breakers[service_id]
            cb.next_attempt_time = datetime.now() + timedelta(seconds=delay)
            
            client = lb.backend_clients[service_id]
            client.query.return_value = {
                "content": f"Service {service_id} restored",
                "response_time_ms": 800,
                "recovered": True
            }
        
        # Phase 4: Test recovery progression
        recovery_tests = []
        
        for test_time in [2, 4, 6, 8]:  # Test at different recovery stages
            # Simulate time progression
            current_time = datetime.now() + timedelta(seconds=test_time)
            
            for service_id in failed_services:
                cb = lb.circuit_breakers[service_id]
                recovery_time = next((delay for sid, delay in recovery_order if sid == service_id), 999)
                if test_time >= recovery_time:
                    cb.next_attempt_time = current_time - timedelta(seconds=1)
            
            # Execute test query
            result = await lb.query(f"Recovery progression test T+{test_time}")
            recovery_tests.append((test_time, result))
        
        # Verify progressive recovery
        all_successful = all(result is not None for _, result in recovery_tests)
        assert all_successful
        
        # Later tests should show improved service quality
        early_result = recovery_tests[0][1]
        late_result = recovery_tests[-1][1]
        
        if "confidence" in early_result and "confidence" in late_result:
            assert late_result["confidence"] >= early_result["confidence"]

    @pytest.mark.asyncio  
    async def test_partial_system_recovery_with_mixed_states(self, e2e_load_balancer):
        """Test system behavior with mixed service states during recovery"""
        lb = e2e_load_balancer
        
        # Create mixed service states
        mixed_states = {
            "lightrag_primary": CircuitBreakerState.CLOSED,    # Healthy
            "lightrag_secondary": CircuitBreakerState.HALF_OPEN, # Recovering
            "perplexity_primary": CircuitBreakerState.OPEN,     # Failed
            "perplexity_fallback": CircuitBreakerState.CLOSED,  # Healthy
            "cache_primary": CircuitBreakerState.CLOSED         # Healthy
        }
        
        # Set up circuit breaker states
        for service_id, state in mixed_states.items():
            cb = lb.circuit_breakers[service_id]
            cb.state = state
            
            if state == CircuitBreakerState.OPEN:
                cb.failure_count = cb.config.failure_threshold
                cb.next_attempt_time = datetime.now() + timedelta(seconds=60)
            elif state == CircuitBreakerState.HALF_OPEN:
                cb.half_open_requests = 0
        
        # Mock responses based on service states
        # Healthy services
        lb.backend_clients["lightrag_primary"].query.return_value = {
            "content": "Healthy LightRAG primary response",
            "confidence": 0.95, "response_time_ms": 750
        }
        
        lb.backend_clients["perplexity_fallback"].query.return_value = {
            "content": "Healthy Perplexity fallback response",
            "confidence": 0.82, "response_time_ms": 2100
        }
        
        lb.backend_clients["cache_primary"].query.return_value = {
            "content": "Healthy cache response",
            "confidence": 0.70, "response_time_ms": 45
        }
        
        # Recovering service (some successes, some issues)
        recovering_call_count = 0
        def mock_recovering_response(*args, **kwargs):
            nonlocal recovering_call_count
            recovering_call_count += 1
            
            if recovering_call_count % 3 == 0:  # Occasional issues
                raise TimeoutError("Still recovering")
            else:
                return {
                    "content": "LightRAG secondary recovering response",
                    "confidence": 0.88, "response_time_ms": 1200
                }
        
        lb.backend_clients["lightrag_secondary"].query.side_effect = mock_recovering_response
        
        # Failed service
        lb.backend_clients["perplexity_primary"].query.side_effect = Exception("Service still down")
        
        # Test multiple queries with mixed service states
        mixed_state_queries = [f"Mixed state test {i}" for i in range(15)]
        mixed_results = []
        
        for query in mixed_state_queries:
            try:
                result = await lb.query(query)
                mixed_results.append(result)
            except Exception as e:
                mixed_results.append(None)
        
        # Verify system handled mixed states gracefully
        successful_mixed = [r for r in mixed_results if r is not None]
        success_rate = len(successful_mixed) / len(mixed_results)
        
        # Should have high success rate despite mixed states
        assert success_rate >= 0.80, f"Success rate {success_rate:.2f} too low with mixed states"
        
        # Verify healthy services were preferred
        healthy_cb_successes = (lb.circuit_breakers["lightrag_primary"].success_count +
                               lb.circuit_breakers["perplexity_fallback"].success_count +
                               lb.circuit_breakers["cache_primary"].success_count)
        
        assert healthy_cb_successes > 0, "Healthy services should have been used"

    @pytest.mark.asyncio
    async def test_recovery_stress_testing_under_load(self, e2e_load_balancer):
        """Test recovery behavior under high concurrent load"""
        lb = e2e_load_balancer
        
        # Open multiple circuit breakers
        stressed_services = ["lightrag_primary", "perplexity_primary"]
        
        for service_id in stressed_services:
            cb = lb.circuit_breakers[service_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Stress test failure {i}")
            assert cb.state == CircuitBreakerState.OPEN
            
            # Set for immediate recovery attempt
            cb.next_attempt_time = datetime.now() - timedelta(seconds=1)
        
        # Mock services with varying recovery characteristics under load
        stress_call_counts = {"lightrag_primary": 0, "perplexity_primary": 0}
        
        def mock_stressed_recovery(service_id):
            def _mock_response(*args, **kwargs):
                stress_call_counts[service_id] += 1
                call_count = stress_call_counts[service_id]
                
                # Simulate gradual recovery under stress
                if call_count <= 3:  # Initial recovery attempts
                    if call_count % 2 == 0:
                        raise TimeoutError(f"Stress timeout in {service_id}")
                    else:
                        return {
                            "content": f"Stressed recovery {service_id} #{call_count}",
                            "response_time_ms": 2500  # Slow due to stress
                        }
                else:  # Stabilized recovery
                    return {
                        "content": f"Stable recovery {service_id} #{call_count}",
                        "response_time_ms": 1000
                    }
            return _mock_response
        
        for service_id in stressed_services:
            client = lb.backend_clients[service_id]
            client.query.side_effect = mock_stressed_recovery(service_id)
        
        # Mock healthy backup services
        lb.backend_clients["lightrag_secondary"].query.return_value = {
            "content": "Backup handling stress load",
            "response_time_ms": 1100
        }
        
        # Generate concurrent load during recovery
        concurrent_queries = [f"Stress recovery query {i}" for i in range(50)]
        
        # Use concurrent execution to stress test
        async def execute_query(query):
            try:
                return await lb.query(query)
            except Exception:
                return None
        
        # Execute all queries concurrently
        stress_results = await asyncio.gather(*[execute_query(q) for q in concurrent_queries])
        
        # Verify system handled stress during recovery
        successful_stress = [r for r in stress_results if r is not None]
        stress_success_rate = len(successful_stress) / len(stress_results)
        
        # Should maintain reasonable success rate under stress
        assert stress_success_rate >= 0.70, f"Stress success rate {stress_success_rate:.2f} too low"
        
        # Verify recovery progressed despite stress
        final_states = {}
        for service_id in stressed_services:
            cb = lb.circuit_breakers[service_id]
            final_states[service_id] = cb.state
        
        # At least one stressed service should show recovery progress
        recovering_services = [s for s, state in final_states.items() 
                             if state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]]
        assert len(recovering_services) >= 1, "No services showed recovery progress under stress"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])