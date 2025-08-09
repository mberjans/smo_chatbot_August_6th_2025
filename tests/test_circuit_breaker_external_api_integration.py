"""
Integration Tests for Circuit Breaker with External APIs

This module provides comprehensive integration tests for circuit breaker functionality
with external systems including OpenAI, Perplexity, and LightRAG APIs. Tests validate
real-world failure scenarios, recovery mechanisms, and integration with production
load balancer components.

Key Test Areas:
1. OpenAI API circuit breaker protection
2. Perplexity API failure handling
3. LightRAG service circuit breaking
4. Production load balancer integration
5. Cost-based circuit breaker scenarios
6. Real-world failure patterns
7. Recovery coordination

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Circuit Breaker External API Integration Tests
"""

import pytest
import asyncio
import aiohttp
import time
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import circuit breaker components
from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError
from lightrag_integration.cost_based_circuit_breaker import CostBasedCircuitBreaker, BudgetExhaustedError
from lightrag_integration.production_load_balancer import (
    ProductionLoadBalancer, BackendType, BackendInstanceConfig,
    ProductionLoadBalancingConfig, CircuitBreakerState, HealthStatus
)
from lightrag_integration.production_intelligent_query_router import ProductionIntelligentQueryRouter
from lightrag_integration.query_router import RoutingDecision

# Test utilities
from tests.conftest import (
    circuit_breaker_config, mock_openai_client, mock_perplexity_client,
    mock_lightrag_instance, load_generator
)


@dataclass
class MockAPIResponse:
    """Mock API response for testing"""
    status_code: int = 200
    content: str = "Mock response"
    headers: Dict[str, str] = None
    response_time_ms: float = 100.0
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}


class MockExternalAPI:
    """Mock external API for testing circuit breaker integration"""
    
    def __init__(self, name: str, fail_count: int = 0, failure_type: Exception = Exception):
        self.name = name
        self.fail_count = fail_count
        self.failure_type = failure_type
        self.call_count = 0
        self.response_time_ms = 100.0
        self.is_healthy = True
        
    async def call(self, request_data: Dict[str, Any]) -> MockAPIResponse:
        """Simulate API call with configurable failures"""
        self.call_count += 1
        
        # Simulate network latency
        await asyncio.sleep(self.response_time_ms / 1000.0)
        
        if not self.is_healthy:
            raise self.failure_type(f"{self.name} API unavailable")
        
        if self.call_count <= self.fail_count:
            raise self.failure_type(f"{self.name} API failure #{self.call_count}")
        
        return MockAPIResponse(
            content=f"{self.name} response for call #{self.call_count}",
            response_time_ms=self.response_time_ms
        )
    
    def set_failure_mode(self, fail_count: int, failure_type: Exception = Exception):
        """Set API to fail for specified number of calls"""
        self.fail_count = fail_count
        self.failure_type = failure_type
        self.call_count = 0
    
    def set_health_status(self, is_healthy: bool):
        """Set API health status"""
        self.is_healthy = is_healthy
    
    def set_response_time(self, response_time_ms: float):
        """Set API response time"""
        self.response_time_ms = response_time_ms


class TestCircuitBreakerOpenAIIntegration:
    """Test circuit breaker integration with OpenAI API"""
    
    @pytest.fixture
    def openai_api(self):
        """Mock OpenAI API"""
        return MockExternalAPI("OpenAI")
    
    @pytest.fixture
    def openai_circuit_breaker(self):
        """Circuit breaker configured for OpenAI API"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_openai_circuit_breaker_basic_protection(self, openai_api, openai_circuit_breaker):
        """Test basic circuit breaker protection for OpenAI API"""
        # Configure API to fail 3 times then succeed
        openai_api.set_failure_mode(3, Exception)
        
        # First 3 calls should fail and open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await openai_circuit_breaker.call(
                    openai_api.call, {"query": f"test query {i}"}
                )
        
        # Circuit should be open
        assert openai_circuit_breaker.state == 'open'
        assert openai_circuit_breaker.failure_count == 3
        
        # Next call should fail with CircuitBreakerError (fast fail)
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is open"):
            await openai_circuit_breaker.call(
                openai_api.call, {"query": "blocked query"}
            )
        
        # API shouldn't have been called (still at 3 calls)
        assert openai_api.call_count == 3
    
    @pytest.mark.asyncio
    async def test_openai_circuit_breaker_recovery(self, openai_api, openai_circuit_breaker):
        """Test circuit breaker recovery after timeout"""
        # Configure API to fail initially then succeed
        openai_api.set_failure_mode(3, Exception)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await openai_circuit_breaker.call(
                    openai_api.call, {"query": f"test query {i}"}
                )
        
        assert openai_circuit_breaker.state == 'open'
        
        # Advance time past recovery timeout
        openai_circuit_breaker.last_failure_time = time.time() - 1.5
        
        # Now API should succeed - circuit should close
        response = await openai_circuit_breaker.call(
            openai_api.call, {"query": "recovery test"}
        )
        
        assert response.content == "OpenAI response for call #4"
        assert openai_circuit_breaker.state == 'closed'
        assert openai_circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_openai_rate_limiting_with_circuit_breaker(self, openai_api, openai_circuit_breaker):
        """Test OpenAI rate limiting scenarios with circuit breaker"""
        from aiohttp import ClientResponseError
        
        # Configure API to simulate rate limiting
        openai_api.set_failure_mode(5, ClientResponseError(None, None, status=429))
        
        # Rate limit errors should trigger circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await openai_circuit_breaker.call(
                    openai_api.call, {"query": f"rate limited query {i}"}
                )
        
        # Circuit should be open due to rate limiting
        assert openai_circuit_breaker.state == 'open'
        
        # Subsequent calls should be blocked
        with pytest.raises(CircuitBreakerError):
            await openai_circuit_breaker.call(
                openai_api.call, {"query": "blocked by rate limit"}
            )
    
    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self, openai_api, openai_circuit_breaker):
        """Test OpenAI timeout handling with circuit breaker"""
        import asyncio
        
        # Configure API with high latency to simulate timeouts
        openai_api.set_response_time(5000.0)  # 5 seconds
        
        async def timeout_wrapper(*args, **kwargs):
            """Wrapper to simulate timeout"""
            try:
                return await asyncio.wait_for(openai_api.call(*args, **kwargs), timeout=0.1)
            except asyncio.TimeoutError:
                raise Exception("API timeout")
        
        # Timeouts should trigger circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await openai_circuit_breaker.call(timeout_wrapper, {"query": f"timeout test {i}"})
        
        assert openai_circuit_breaker.state == 'open'


class TestCircuitBreakerPerplexityIntegration:
    """Test circuit breaker integration with Perplexity API"""
    
    @pytest.fixture
    def perplexity_api(self):
        """Mock Perplexity API"""
        return MockExternalAPI("Perplexity")
    
    @pytest.fixture
    def perplexity_circuit_breaker(self):
        """Circuit breaker configured for Perplexity API"""
        return CircuitBreaker(
            failure_threshold=2,  # More sensitive for Perplexity
            recovery_timeout=2.0,
            expected_exception=(Exception, aiohttp.ClientError)
        )
    
    @pytest.mark.asyncio
    async def test_perplexity_connection_failures(self, perplexity_api, perplexity_circuit_breaker):
        """Test Perplexity connection failure handling"""
        # Simulate connection errors
        perplexity_api.set_failure_mode(3, aiohttp.ClientConnectionError("Connection failed"))
        
        # Connection errors should trigger circuit breaker
        for i in range(2):
            with pytest.raises(Exception):
                await perplexity_circuit_breaker.call(
                    perplexity_api.call, {"messages": [{"role": "user", "content": f"test {i}"}]}
                )
        
        assert perplexity_circuit_breaker.state == 'open'
        assert perplexity_circuit_breaker.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_perplexity_auth_failures(self, perplexity_api, perplexity_circuit_breaker):
        """Test Perplexity authentication failure scenarios"""
        # Simulate auth errors
        perplexity_api.set_failure_mode(2, aiohttp.ClientResponseError(None, None, status=401))
        
        # Auth errors should trigger circuit breaker
        for i in range(2):
            with pytest.raises(Exception):
                await perplexity_circuit_breaker.call(
                    perplexity_api.call, {"messages": [{"role": "user", "content": f"auth test {i}"}]}
                )
        
        assert perplexity_circuit_breaker.state == 'open'
    
    @pytest.mark.asyncio
    async def test_perplexity_partial_failures(self, perplexity_api, perplexity_circuit_breaker):
        """Test handling of partial failures with Perplexity API"""
        # Configure to fail only first call
        perplexity_api.set_failure_mode(1, Exception("Temporary failure"))
        
        # First call fails
        with pytest.raises(Exception):
            await perplexity_circuit_breaker.call(
                perplexity_api.call, {"messages": [{"role": "user", "content": "test 1"}]}
            )
        
        assert perplexity_circuit_breaker.failure_count == 1
        assert perplexity_circuit_breaker.state == 'closed'  # Not enough failures to open
        
        # Second call should succeed
        response = await perplexity_circuit_breaker.call(
            perplexity_api.call, {"messages": [{"role": "user", "content": "test 2"}]}
        )
        
        assert response.content == "Perplexity response for call #2"
        assert perplexity_circuit_breaker.failure_count == 0  # Reset on success


class TestCircuitBreakerLightRAGIntegration:
    """Test circuit breaker integration with LightRAG service"""
    
    @pytest.fixture
    def lightrag_api(self):
        """Mock LightRAG API"""
        return MockExternalAPI("LightRAG")
    
    @pytest.fixture
    def lightrag_circuit_breaker(self):
        """Circuit breaker configured for LightRAG"""
        return CircuitBreaker(
            failure_threshold=4,  # Higher threshold for LightRAG
            recovery_timeout=0.5,  # Fast recovery
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_lightrag_query_failures(self, lightrag_api, lightrag_circuit_breaker):
        """Test LightRAG query failure handling"""
        # Configure to fail 4 times
        lightrag_api.set_failure_mode(4, RuntimeError("LightRAG processing error"))
        
        # Process failures until circuit opens
        for i in range(4):
            with pytest.raises(RuntimeError):
                await lightrag_circuit_breaker.call(
                    lightrag_api.call, {"query": f"metabolomics query {i}", "mode": "hybrid"}
                )
        
        assert lightrag_circuit_breaker.state == 'open'
        assert lightrag_circuit_breaker.failure_count == 4
    
    @pytest.mark.asyncio
    async def test_lightrag_memory_issues(self, lightrag_api, lightrag_circuit_breaker):
        """Test LightRAG memory/resource failure scenarios"""
        # Simulate memory errors
        lightrag_api.set_failure_mode(2, MemoryError("Out of memory"))
        
        # Memory errors should be handled
        for i in range(2):
            with pytest.raises(MemoryError):
                await lightrag_circuit_breaker.call(
                    lightrag_api.call, {"query": f"large query {i}"}
                )
        
        assert lightrag_circuit_breaker.failure_count == 2
        assert lightrag_circuit_breaker.state == 'closed'  # Not enough to open
    
    @pytest.mark.asyncio
    async def test_lightrag_embedding_failures(self, lightrag_api, lightrag_circuit_breaker):
        """Test LightRAG embedding service failure handling"""
        # Simulate embedding service failures
        lightrag_api.set_failure_mode(4, ValueError("Embedding service unavailable"))
        
        for i in range(4):
            with pytest.raises(ValueError):
                await lightrag_circuit_breaker.call(
                    lightrag_api.call, {"operation": "embed", "texts": [f"text {i}"]}
                )
        
        # Circuit should open
        assert lightrag_circuit_breaker.state == 'open'
        
        # Fast recovery test
        lightrag_circuit_breaker.last_failure_time = time.time() - 0.6
        
        # Should allow recovery attempt
        response = await lightrag_circuit_breaker.call(
            lightrag_api.call, {"operation": "embed", "texts": ["recovery test"]}
        )
        
        assert lightrag_circuit_breaker.state == 'closed'


class TestProductionLoadBalancerCircuitBreakerIntegration:
    """Test circuit breaker integration with production load balancer"""
    
    @pytest.fixture
    def production_config(self):
        """Production load balancer configuration with circuit breakers"""
        config = ProductionLoadBalancingConfig()
        
        # Add backend instances with circuit breaker configs
        config.backend_instances = {
            "openai_1": BackendInstanceConfig(
                id="openai_1",
                backend_type=BackendType.OPENAI_DIRECT,
                endpoint_url="https://api.openai.com/v1",
                api_key="test-key",
                weight=1.0,
                circuit_breaker_enabled=True,
                failure_threshold=3,
                recovery_timeout_seconds=2
            ),
            "perplexity_1": BackendInstanceConfig(
                id="perplexity_1",
                backend_type=BackendType.PERPLEXITY,
                endpoint_url="https://api.perplexity.ai/chat/completions",
                api_key="test-key",
                weight=1.0,
                circuit_breaker_enabled=True,
                failure_threshold=2,
                recovery_timeout_seconds=3
            ),
            "lightrag_1": BackendInstanceConfig(
                id="lightrag_1",
                backend_type=BackendType.LIGHTRAG,
                endpoint_url="http://localhost:8000",
                api_key="test-key",
                weight=1.0,
                circuit_breaker_enabled=True,
                failure_threshold=4,
                recovery_timeout_seconds=1
            )
        }
        
        return config
    
    @pytest.fixture
    def mock_apis(self):
        """Mock APIs for testing"""
        return {
            "openai_1": MockExternalAPI("OpenAI"),
            "perplexity_1": MockExternalAPI("Perplexity"),
            "lightrag_1": MockExternalAPI("LightRAG")
        }
    
    @pytest.mark.asyncio
    async def test_load_balancer_circuit_breaker_coordination(self, production_config, mock_apis):
        """Test coordination between load balancer and circuit breakers"""
        # Mock the ProductionLoadBalancer to use our mock APIs
        with patch.object(ProductionLoadBalancer, '_create_backend_client') as mock_client:
            def create_client_side_effect(backend_id, config):
                return mock_apis[backend_id]
            
            mock_client.side_effect = create_client_side_effect
            
            load_balancer = ProductionLoadBalancer(production_config)
            
            # Configure first backend to fail
            mock_apis["openai_1"].set_failure_mode(5, Exception("API failure"))
            
            # Simulate requests that trigger circuit breaker
            routing_decision = RoutingDecision.EITHER
            context = {"query": "test query"}
            
            # First few requests should fail and trigger circuit breaker
            for i in range(3):
                try:
                    await load_balancer.select_backend(routing_decision, context)
                except Exception:
                    pass  # Expected failures
            
            # Circuit breaker should be open, load balancer should route to other backends
            selected_backend = await load_balancer.select_backend(routing_decision, context)
            
            # Should select a different backend (not openai_1)
            assert selected_backend in ["perplexity_1", "lightrag_1"]
    
    @pytest.mark.asyncio
    async def test_cascade_failure_prevention(self, production_config, mock_apis):
        """Test prevention of cascade failures across backends"""
        with patch.object(ProductionLoadBalancer, '_create_backend_client') as mock_client:
            def create_client_side_effect(backend_id, config):
                return mock_apis[backend_id]
            
            mock_client.side_effect = create_client_side_effect
            
            load_balancer = ProductionLoadBalancer(production_config)
            
            # Configure all backends to fail initially
            for api in mock_apis.values():
                api.set_failure_mode(10, Exception("Service degraded"))
            
            routing_decision = RoutingDecision.EITHER
            context = {"query": "cascade test"}
            
            # Multiple requests should eventually open all circuit breakers
            failure_count = 0
            for i in range(15):
                try:
                    await load_balancer.select_backend(routing_decision, context)
                except Exception:
                    failure_count += 1
            
            # Should have failed to find available backend
            assert failure_count > 0
            
            # Now fix one backend
            mock_apis["lightrag_1"].set_failure_mode(0)
            
            # After recovery timeout, should be able to route again
            await asyncio.sleep(1.1)  # Wait for recovery
            
            selected_backend = await load_balancer.select_backend(routing_decision, context)
            assert selected_backend == "lightrag_1"


class TestCostBasedCircuitBreakerIntegration:
    """Test integration with cost-based circuit breaker"""
    
    @pytest.fixture
    def cost_based_circuit_breaker(self):
        """Cost-based circuit breaker for testing"""
        return CostBasedCircuitBreaker(
            cost_threshold=10.0,
            failure_threshold=3,
            recovery_timeout=1.0,
            cost_window_hours=1.0
        )
    
    @pytest.fixture
    def expensive_api(self):
        """Mock expensive API"""
        api = MockExternalAPI("ExpensiveAPI")
        # Override call method to include cost
        original_call = api.call
        
        async def call_with_cost(*args, **kwargs):
            response = await original_call(*args, **kwargs)
            # Add cost information to response
            response.cost = 2.0  # $2 per call
            return response
        
        api.call = call_with_cost
        return api
    
    @pytest.mark.asyncio
    async def test_cost_based_circuit_breaker_api_integration(self, cost_based_circuit_breaker, expensive_api):
        """Test cost-based circuit breaker with expensive API calls"""
        
        # Make several expensive calls
        total_cost = 0.0
        for i in range(4):
            try:
                response = await cost_based_circuit_breaker.call(
                    expensive_api.call, {"query": f"expensive query {i}"}
                )
                total_cost += getattr(response, 'cost', 0.0)
                await cost_based_circuit_breaker.add_cost(getattr(response, 'cost', 0.0))
            except Exception:
                break
        
        # Should have spent $8 (4 calls × $2)
        assert total_cost == 8.0
        
        # Next call should trigger budget exhaustion
        with pytest.raises(BudgetExhaustedError):
            response = await cost_based_circuit_breaker.call(
                expensive_api.call, {"query": "budget exceeded"}
            )
            await cost_based_circuit_breaker.add_cost(2.0)
    
    @pytest.mark.asyncio
    async def test_cost_circuit_breaker_recovery(self, cost_based_circuit_breaker, expensive_api):
        """Test cost-based circuit breaker recovery after time window"""
        
        # Exhaust budget
        for i in range(5):
            try:
                response = await cost_based_circuit_breaker.call(
                    expensive_api.call, {"query": f"query {i}"}
                )
                await cost_based_circuit_breaker.add_cost(2.0)
            except BudgetExhaustedError:
                break
        
        # Should be in budget exhausted state
        assert cost_based_circuit_breaker.is_budget_exhausted()
        
        # Advance time past cost window
        cost_based_circuit_breaker.cost_window_start = time.time() - 3700  # > 1 hour
        
        # Should be able to make calls again
        response = await cost_based_circuit_breaker.call(
            expensive_api.call, {"query": "recovery test"}
        )
        await cost_based_circuit_breaker.add_cost(2.0)
        
        assert not cost_based_circuit_breaker.is_budget_exhausted()


class TestRealWorldFailureScenarios:
    """Test real-world failure patterns and recovery scenarios"""
    
    @pytest.fixture
    def multi_api_setup(self):
        """Setup with multiple APIs for comprehensive testing"""
        return {
            "openai": MockExternalAPI("OpenAI"),
            "perplexity": MockExternalAPI("Perplexity"),
            "lightrag": MockExternalAPI("LightRAG")
        }
    
    @pytest.fixture
    def circuit_breakers(self):
        """Circuit breakers for each API"""
        return {
            "openai": CircuitBreaker(failure_threshold=3, recovery_timeout=2.0),
            "perplexity": CircuitBreaker(failure_threshold=2, recovery_timeout=3.0),
            "lightrag": CircuitBreaker(failure_threshold=4, recovery_timeout=1.0)
        }
    
    @pytest.mark.asyncio
    async def test_intermittent_network_failures(self, multi_api_setup, circuit_breakers):
        """Test handling of intermittent network failures"""
        apis = multi_api_setup
        cbs = circuit_breakers
        
        # Simulate intermittent failures (fail every 3rd call)
        call_counts = {"openai": 0, "perplexity": 0, "lightrag": 0}
        
        async def intermittent_call(api_name):
            call_counts[api_name] += 1
            if call_counts[api_name] % 3 == 0:
                raise aiohttp.ClientConnectionError("Network timeout")
            return await apis[api_name].call({"query": f"test {call_counts[api_name]}"})
        
        # Make multiple calls - should handle intermittent failures without opening circuits
        successful_calls = 0
        for i in range(10):
            for api_name in ["openai", "perplexity", "lightrag"]:
                try:
                    await cbs[api_name].call(lambda: intermittent_call(api_name))
                    successful_calls += 1
                except Exception:
                    pass
        
        # Should have had some successful calls despite intermittent failures
        assert successful_calls > 15  # At least 50% success rate
        
        # Circuits should mostly remain closed due to intermittent nature
        closed_circuits = sum(1 for cb in cbs.values() if cb.state == 'closed')
        assert closed_circuits >= 2
    
    @pytest.mark.asyncio
    async def test_progressive_degradation_scenario(self, multi_api_setup, circuit_breakers):
        """Test scenario where services progressively degrade"""
        apis = multi_api_setup
        cbs = circuit_breakers
        
        # Phase 1: All services healthy
        for api_name in ["openai", "perplexity", "lightrag"]:
            response = await cbs[api_name].call(apis[api_name].call, {"query": "phase 1"})
            assert "response for call" in response.content
        
        # Phase 2: OpenAI starts failing
        apis["openai"].set_failure_mode(10, Exception("OpenAI degraded"))
        
        # Phase 3: Perplexity also starts failing after some time
        for i in range(3):
            try:
                await cbs["openai"].call(apis["openai"].call, {"query": f"failing {i}"})
            except Exception:
                pass
        
        apis["perplexity"].set_failure_mode(10, Exception("Perplexity degraded"))
        
        # Phase 4: Only LightRAG remains healthy
        for i in range(2):
            try:
                await cbs["perplexity"].call(apis["perplexity"].call, {"query": f"failing {i}"})
            except Exception:
                pass
        
        # Verify progressive circuit opening
        assert cbs["openai"].state == 'open'
        assert cbs["perplexity"].state == 'open'
        assert cbs["lightrag"].state == 'closed'
        
        # LightRAG should still work
        response = await cbs["lightrag"].call(apis["lightrag"].call, {"query": "last hope"})
        assert "LightRAG response" in response.content
    
    @pytest.mark.asyncio
    async def test_coordinated_recovery_scenario(self, multi_api_setup, circuit_breakers):
        """Test coordinated recovery of multiple services"""
        apis = multi_api_setup
        cbs = circuit_breakers
        
        # Break all services
        for api_name, api in apis.items():
            api.set_failure_mode(10, Exception(f"{api_name} down"))
        
        # Open all circuits
        for api_name in ["openai", "perplexity", "lightrag"]:
            threshold = cbs[api_name].failure_threshold
            for i in range(threshold):
                try:
                    await cbs[api_name].call(apis[api_name].call, {"query": f"break {i}"})
                except Exception:
                    pass
        
        # All circuits should be open
        for cb in cbs.values():
            assert cb.state == 'open'
        
        # Fix all services
        for api in apis.values():
            api.set_failure_mode(0)
        
        # Advance time for recovery
        recovery_time = time.time() - 5.0
        for cb in cbs.values():
            cb.last_failure_time = recovery_time
        
        # Test coordinated recovery
        recovery_results = {}
        for api_name in ["openai", "perplexity", "lightrag"]:
            try:
                response = await cbs[api_name].call(apis[api_name].call, {"query": "recovery"})
                recovery_results[api_name] = "success"
                assert cbs[api_name].state == 'closed'
            except Exception as e:
                recovery_results[api_name] = str(e)
        
        # All should recover successfully
        assert all(result == "success" for result in recovery_results.values())
    
    @pytest.mark.asyncio
    async def test_high_load_circuit_breaker_behavior(self, multi_api_setup, circuit_breakers, load_generator):
        """Test circuit breaker behavior under high load"""
        apis = multi_api_setup
        cbs = circuit_breakers
        
        # Configure APIs with different reliability under load
        apis["openai"].set_response_time(200.0)  # Slower response
        apis["perplexity"].set_response_time(100.0)  # Fast response
        apis["lightrag"].set_response_time(300.0)  # Slowest response
        
        # Set up intermittent failures under load
        async def load_test_call(api_name):
            # Simulate occasional failures under load
            if time.time() % 0.1 < 0.02:  # ~20% failure rate
                raise Exception(f"{api_name} overloaded")
            return await apis[api_name].call({"query": f"load test"})
        
        # Generate load using the load generator
        results = {"successes": 0, "failures": 0, "circuit_breaker_blocks": 0}
        
        async def single_request():
            api_name = ["openai", "perplexity", "lightrag"][int(time.time() * 1000) % 3]
            try:
                await cbs[api_name].call(lambda: load_test_call(api_name))
                results["successes"] += 1
            except CircuitBreakerError:
                results["circuit_breaker_blocks"] += 1
            except Exception:
                results["failures"] += 1
        
        # Run load test
        tasks = [single_request() for _ in range(100)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify circuit breakers protected against overload
        assert results["circuit_breaker_blocks"] > 0
        assert results["successes"] > 0
        
        # At least one circuit should have opened under load
        open_circuits = sum(1 for cb in cbs.values() if cb.state == 'open')
        assert open_circuits > 0


if __name__ == "__main__":
    # Run tests with proper configuration
    pytest.main([__file__, "-v", "--tb=short"])