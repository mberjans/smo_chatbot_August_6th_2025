"""
Integration Tests for Circuit Breaker with Multi-Level Fallback System

This module provides comprehensive integration tests for circuit breaker functionality
with the multi-level fallback system. Tests validate the coordination between circuit
breakers and fallback levels, ensuring proper escalation and recovery mechanisms.

Key Test Areas:
1. Circuit breaker triggers fallback levels
2. Fallback level progression with circuit breakers
3. Emergency cache activation
4. Recovery coordination between systems
5. Cascade failure prevention
6. Uncertainty-aware fallback integration
7. Performance optimization during fallbacks

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Circuit Breaker Fallback Integration Tests
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock, PropertyMock
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

# Import circuit breaker components
from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError
from lightrag_integration.cost_based_circuit_breaker import CostBasedCircuitBreaker, BudgetExhaustedError

# Import fallback system components
try:
    from lightrag_integration.comprehensive_fallback_system import (
        FallbackOrchestrator, FallbackResult, FallbackLevel, FallbackStrategy,
        FallbackConfig, create_fallback_orchestrator
    )
    from lightrag_integration.uncertainty_aware_cascade_system import (
        UncertaintyAwareFallbackCascade, CascadeResult, CascadeStepResult,
        CascadeStepType, CascadeFailureReason
    )
    from lightrag_integration.uncertainty_aware_fallback_implementation import (
        UncertaintyType, UncertaintyAnalysis, UncertaintyFallbackStrategies,
        UncertaintyAwareFallbackOrchestrator
    )
    FALLBACK_AVAILABLE = True
except ImportError:
    # Create mock classes if fallback system not available
    FALLBACK_AVAILABLE = False
    
    class FallbackLevel(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        TERTIARY = "tertiary"
        EMERGENCY_CACHE = "emergency_cache"
    
    class FallbackStrategy(Enum):
        FAIL_FAST = "fail_fast"
        RETRY_WITH_BACKOFF = "retry_with_backoff"
        CIRCUIT_BREAKER = "circuit_breaker"
        EMERGENCY_CACHE = "emergency_cache"
    
    @dataclass
    class FallbackResult:
        success: bool
        result: Any
        level_used: FallbackLevel
        strategy_used: FallbackStrategy
        response_time_ms: float
        error: Optional[Exception] = None

# Import production components
from lightrag_integration.production_load_balancer import ProductionLoadBalancer
from lightrag_integration.production_intelligent_query_router import ProductionIntelligentQueryRouter
from lightrag_integration.query_router import RoutingDecision, RoutingPrediction


class MockFallbackOrchestrator:
    """Mock fallback orchestrator for testing"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_levels = [
            FallbackLevel.PRIMARY,
            FallbackLevel.SECONDARY, 
            FallbackLevel.TERTIARY,
            FallbackLevel.EMERGENCY_CACHE
        ]
        self.current_level = FallbackLevel.PRIMARY
        self.emergency_cache = {}
        self.call_history = []
        
        # Initialize circuit breakers for each level
        for level in self.fallback_levels:
            self.circuit_breakers[level.value] = CircuitBreaker(
                failure_threshold=3 if level == FallbackLevel.PRIMARY else 2,
                recovery_timeout=2.0 if level == FallbackLevel.PRIMARY else 1.0
            )
    
    async def execute_with_fallback(self, operation: Callable, context: Dict[str, Any]) -> FallbackResult:
        """Execute operation with fallback support"""
        start_time = time.time()
        
        for level in self.fallback_levels:
            circuit_breaker = self.circuit_breakers[level.value]
            
            # Skip level if circuit breaker is open
            if circuit_breaker.state == 'open':
                continue
                
            try:
                if level == FallbackLevel.EMERGENCY_CACHE:
                    result = await self._get_from_emergency_cache(context)
                else:
                    result = await circuit_breaker.call(operation, context, level=level)
                
                response_time_ms = (time.time() - start_time) * 1000
                
                fallback_result = FallbackResult(
                    success=True,
                    result=result,
                    level_used=level,
                    strategy_used=FallbackStrategy.CIRCUIT_BREAKER,
                    response_time_ms=response_time_ms
                )
                
                self.call_history.append(fallback_result)
                return fallback_result
                
            except CircuitBreakerError:
                continue  # Try next level
            except Exception as e:
                if level == FallbackLevel.EMERGENCY_CACHE:
                    # Emergency cache is last resort
                    break
                continue
        
        # All levels failed
        response_time_ms = (time.time() - start_time) * 1000
        fallback_result = FallbackResult(
            success=False,
            result=None,
            level_used=FallbackLevel.EMERGENCY_CACHE,
            strategy_used=FallbackStrategy.FAIL_FAST,
            response_time_ms=response_time_ms,
            error=Exception("All fallback levels exhausted")
        )
        
        self.call_history.append(fallback_result)
        return fallback_result
    
    async def _get_from_emergency_cache(self, context: Dict[str, Any]) -> str:
        """Get response from emergency cache"""
        cache_key = str(hash(str(context.get("query", ""))))
        if cache_key in self.emergency_cache:
            return self.emergency_cache[cache_key]
        else:
            # Simulate cache miss with generic response
            return "Emergency cache response: Limited functionality available"
    
    def add_to_emergency_cache(self, query: str, response: str):
        """Add response to emergency cache"""
        cache_key = str(hash(query))
        self.emergency_cache[cache_key] = response
    
    def get_circuit_breaker_states(self) -> Dict[str, str]:
        """Get current states of all circuit breakers"""
        return {level: cb.state for level, cb in self.circuit_breakers.items()}


class MockUncertaintyAwareCascade:
    """Mock uncertainty-aware cascade system for testing"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.cascade_steps = ["high_confidence", "medium_confidence", "low_confidence", "consensus"]
        self.uncertainty_thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}
        
        for step in self.cascade_steps:
            self.circuit_breakers[step] = CircuitBreaker(
                failure_threshold=2,
                recovery_timeout=1.0
            )
    
    async def execute_cascade(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute uncertainty-aware cascade with circuit breaker protection"""
        
        for step in self.cascade_steps:
            circuit_breaker = self.circuit_breakers[step]
            
            if circuit_breaker.state == 'open':
                continue
            
            try:
                async def mock_step_execution(query, context, step=step):
                    if step == "consensus":
                        return {"result": f"Consensus result for: {query}", "confidence": 0.9}
                    else:
                        confidence = self.uncertainty_thresholds.get(step.split("_")[0], 0.5)
                        return {"result": f"{step} result for: {query}", "confidence": confidence}
                
                result = await circuit_breaker.call(mock_step_execution, query, context)
                
                # Check if result meets uncertainty threshold
                if result["confidence"] >= 0.7:  # Acceptable confidence
                    return {"success": True, "step_used": step, **result}
                    
            except CircuitBreakerError:
                continue
            except Exception:
                continue
        
        return {"success": False, "step_used": "none", "error": "All cascade steps failed"}


class TestCircuitBreakerFallbackOrchestration:
    """Test circuit breaker integration with fallback orchestration"""
    
    @pytest.fixture
    def fallback_orchestrator(self):
        """Fallback orchestrator with circuit breaker integration"""
        return MockFallbackOrchestrator()
    
    @pytest.fixture
    def mock_operation(self):
        """Mock operation that can fail or succeed"""
        class MockOperation:
            def __init__(self):
                self.call_count = 0
                self.fail_levels = set()
                
            async def __call__(self, context, level=FallbackLevel.PRIMARY):
                self.call_count += 1
                
                if level in self.fail_levels:
                    raise Exception(f"Operation failed at {level.value} level")
                
                return f"Success at {level.value} level (call #{self.call_count})"
            
            def set_failure_levels(self, levels: List[FallbackLevel]):
                self.fail_levels = set(levels)
        
        return MockOperation()
    
    @pytest.mark.asyncio
    async def test_fallback_level_progression(self, fallback_orchestrator, mock_operation):
        """Test progression through fallback levels when circuit breakers open"""
        
        # Configure primary level to fail
        mock_operation.set_failure_levels([FallbackLevel.PRIMARY])
        
        # Make requests that will fail at primary level
        for i in range(3):
            result = await fallback_orchestrator.execute_with_fallback(
                mock_operation, {"query": f"test query {i}"}
            )
            
            if i < 2:
                # Should still succeed at secondary level
                assert result.success
                assert result.level_used == FallbackLevel.SECONDARY
            else:
                # Primary circuit should be open, still using secondary
                assert result.success
                assert result.level_used == FallbackLevel.SECONDARY
        
        # Verify primary circuit breaker is open
        states = fallback_orchestrator.get_circuit_breaker_states()
        assert states[FallbackLevel.PRIMARY.value] == 'open'
        assert states[FallbackLevel.SECONDARY.value] == 'closed'
    
    @pytest.mark.asyncio
    async def test_cascade_failure_handling(self, fallback_orchestrator, mock_operation):
        """Test handling of cascade failures across multiple levels"""
        
        # Configure first two levels to fail
        mock_operation.set_failure_levels([FallbackLevel.PRIMARY, FallbackLevel.SECONDARY])
        
        # Open primary circuit breaker
        for i in range(3):
            try:
                await fallback_orchestrator.execute_with_fallback(
                    mock_operation, {"query": f"primary fail {i}"}
                )
            except:
                pass
        
        # Open secondary circuit breaker
        mock_operation.set_failure_levels([FallbackLevel.SECONDARY])
        for i in range(2):
            try:
                await fallback_orchestrator.execute_with_fallback(
                    mock_operation, {"query": f"secondary fail {i}"}
                )
            except:
                pass
        
        # Now only tertiary should work
        mock_operation.set_failure_levels([])  # Allow tertiary to succeed
        
        result = await fallback_orchestrator.execute_with_fallback(
            mock_operation, {"query": "tertiary test"}
        )
        
        assert result.success
        assert result.level_used == FallbackLevel.TERTIARY
        
        # Verify circuit breaker states
        states = fallback_orchestrator.get_circuit_breaker_states()
        assert states[FallbackLevel.PRIMARY.value] == 'open'
        assert states[FallbackLevel.SECONDARY.value] == 'open'
        assert states[FallbackLevel.TERTIARY.value] == 'closed'
    
    @pytest.mark.asyncio
    async def test_emergency_cache_activation(self, fallback_orchestrator, mock_operation):
        """Test emergency cache activation when all other levels fail"""
        
        # Pre-populate emergency cache
        fallback_orchestrator.add_to_emergency_cache(
            "emergency test", "Cached response for emergency test"
        )
        
        # Configure all levels to fail
        mock_operation.set_failure_levels([
            FallbackLevel.PRIMARY, FallbackLevel.SECONDARY, FallbackLevel.TERTIARY
        ])
        
        # Open all circuit breakers
        for level in [FallbackLevel.PRIMARY, FallbackLevel.SECONDARY, FallbackLevel.TERTIARY]:
            for i in range(3):
                try:
                    await fallback_orchestrator.execute_with_fallback(
                        mock_operation, {"query": f"{level.value} fail {i}"}
                    )
                except:
                    pass
        
        # Now request should fall back to emergency cache
        result = await fallback_orchestrator.execute_with_fallback(
            mock_operation, {"query": "emergency test"}
        )
        
        assert result.success
        assert result.level_used == FallbackLevel.EMERGENCY_CACHE
        assert "Limited functionality available" in result.result
    
    @pytest.mark.asyncio
    async def test_coordinated_recovery(self, fallback_orchestrator, mock_operation):
        """Test coordinated recovery of multiple fallback levels"""
        
        # Open all circuit breakers by failing operations
        mock_operation.set_failure_levels([
            FallbackLevel.PRIMARY, FallbackLevel.SECONDARY, FallbackLevel.TERTIARY
        ])
        
        for level in [FallbackLevel.PRIMARY, FallbackLevel.SECONDARY, FallbackLevel.TERTIARY]:
            threshold = fallback_orchestrator.circuit_breakers[level.value].failure_threshold
            for i in range(threshold):
                try:
                    await fallback_orchestrator.execute_with_fallback(
                        mock_operation, {"query": f"{level.value} break {i}"}
                    )
                except:
                    pass
        
        # Verify all are open
        states = fallback_orchestrator.get_circuit_breaker_states()
        for level in [FallbackLevel.PRIMARY, FallbackLevel.SECONDARY, FallbackLevel.TERTIARY]:
            assert states[level.value] == 'open'
        
        # Fix operations and advance time for recovery
        mock_operation.set_failure_levels([])
        recovery_time = time.time() - 3.0
        
        for circuit_breaker in fallback_orchestrator.circuit_breakers.values():
            circuit_breaker.last_failure_time = recovery_time
        
        # Test recovery at each level
        result = await fallback_orchestrator.execute_with_fallback(
            mock_operation, {"query": "recovery test"}
        )
        
        assert result.success
        assert result.level_used == FallbackLevel.PRIMARY  # Should recover to primary
        
        # Verify primary circuit is closed after successful call
        states = fallback_orchestrator.get_circuit_breaker_states()
        assert states[FallbackLevel.PRIMARY.value] == 'closed'


class TestUncertaintyAwareFallbackIntegration:
    """Test circuit breaker integration with uncertainty-aware fallback system"""
    
    @pytest.fixture
    def uncertainty_cascade(self):
        """Mock uncertainty-aware cascade system"""
        return MockUncertaintyAwareCascade()
    
    @pytest.mark.asyncio
    async def test_uncertainty_threshold_circuit_breaker_coordination(self, uncertainty_cascade):
        """Test coordination between uncertainty thresholds and circuit breakers"""
        
        # Configure high confidence step to fail
        def mock_high_confidence_failure(query, context, step="high_confidence"):
            raise Exception("High confidence model unavailable")
        
        # Replace the mock execution for high confidence
        high_cb = uncertainty_cascade.circuit_breakers["high_confidence"]
        
        # Fail high confidence step multiple times
        for i in range(2):
            try:
                await high_cb.call(mock_high_confidence_failure, "test", {})
            except Exception:
                pass
        
        # Circuit should be open
        assert high_cb.state == 'open'
        
        # Now cascade should skip to medium confidence
        result = await uncertainty_cascade.execute_cascade("uncertainty test", {})
        
        assert result["success"]
        assert result["step_used"] == "medium_confidence"
    
    @pytest.mark.asyncio
    async def test_confidence_degradation_with_circuit_protection(self, uncertainty_cascade):
        """Test graceful confidence degradation with circuit breaker protection"""
        
        # Create a sequence where high and medium confidence models fail
        async def failing_model(query, context, step):
            if step in ["high_confidence", "medium_confidence"]:
                raise Exception(f"{step} model overloaded")
            elif step == "low_confidence":
                return {"result": f"Low confidence result: {query}", "confidence": 0.4}
            else:  # consensus
                return {"result": f"Consensus result: {query}", "confidence": 0.9}
        
        # Open high and medium confidence circuit breakers
        for step in ["high_confidence", "medium_confidence"]:
            cb = uncertainty_cascade.circuit_breakers[step]
            for i in range(2):
                try:
                    await cb.call(failing_model, "test", {}, step)
                except Exception:
                    pass
        
        # Should fall back to consensus (skipping low confidence due to insufficient confidence)
        result = await uncertainty_cascade.execute_cascade("degradation test", {})
        
        assert result["success"]
        assert result["step_used"] == "consensus"
        assert result["confidence"] >= 0.7
    
    @pytest.mark.asyncio
    async def test_uncertainty_circuit_breaker_recovery_coordination(self, uncertainty_cascade):
        """Test coordinated recovery of uncertainty-aware cascade steps"""
        
        # Break all steps
        for step in uncertainty_cascade.cascade_steps:
            cb = uncertainty_cascade.circuit_breakers[step]
            for i in range(2):
                try:
                    async def failing_step(query, context, step=step):
                        raise Exception(f"{step} temporarily down")
                    await cb.call(failing_step, "test", {})
                except Exception:
                    pass
        
        # All circuits should be open
        for step in uncertainty_cascade.cascade_steps:
            assert uncertainty_cascade.circuit_breakers[step].state == 'open'
        
        # Advance time for recovery
        recovery_time = time.time() - 2.0
        for cb in uncertainty_cascade.circuit_breakers.values():
            cb.last_failure_time = recovery_time
        
        # Should be able to execute cascade again
        result = await uncertainty_cascade.execute_cascade("recovery test", {})
        
        assert result["success"]
        # Should use first available step (high_confidence)
        assert result["step_used"] == "high_confidence"


class TestProductionFallbackIntegration:
    """Test circuit breaker integration with production systems"""
    
    @pytest.fixture
    def production_router_with_fallback(self):
        """Production router with fallback integration"""
        from lightrag_integration.production_intelligent_query_router import (
            ProductionIntelligentQueryRouter, ProductionFeatureFlags, DeploymentMode
        )
        
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.PRODUCTION_ONLY,
            enable_automatic_failback=True
        )
        
        return ProductionIntelligentQueryRouter(feature_flags=feature_flags)
    
    @pytest.mark.asyncio
    async def test_production_router_fallback_coordination(self, production_router_with_fallback):
        """Test coordination between production router and fallback mechanisms"""
        
        router = production_router_with_fallback
        
        # Mock the production load balancer to simulate failures
        with patch.object(router, 'production_load_balancer') as mock_lb:
            mock_lb.select_backend.side_effect = [
                Exception("Load balancer failure"),  # First call fails
                Exception("Load balancer failure"),  # Second call fails
                "lightrag_1"  # Third call succeeds
            ]
            
            # Mock the legacy router as fallback
            with patch.object(router, '_route_with_legacy') as mock_legacy:
                mock_legacy.return_value = RoutingPrediction(
                    routing_decision=RoutingDecision.LIGHTRAG,
                    confidence_metrics={"overall_confidence": 0.8},
                    reasoning="Fallback to legacy router"
                )
                
                # First call should fail over to legacy
                result = await router.route_query("test query")
                
                assert result.routing_decision == RoutingDecision.LIGHTRAG
                assert "Fallback" in result.reasoning or "legacy" in result.reasoning.lower()
                
                # Verify legacy router was called
                mock_legacy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cost_based_fallback_integration(self):
        """Test integration with cost-based circuit breakers in fallback scenarios"""
        
        # Create cost-based circuit breaker
        cost_cb = CostBasedCircuitBreaker(
            cost_threshold=5.0,
            failure_threshold=2,
            recovery_timeout=1.0,
            cost_window_hours=1.0
        )
        
        # Create fallback orchestrator with cost awareness
        fallback_orchestrator = MockFallbackOrchestrator()
        
        async def expensive_operation(context, level=FallbackLevel.PRIMARY):
            # Simulate expensive API call
            cost = 2.0 if level == FallbackLevel.PRIMARY else 1.0
            await cost_cb.add_cost(cost)
            
            if cost_cb.is_budget_exhausted():
                raise BudgetExhaustedError("Budget exceeded")
            
            return f"Expensive operation result (cost: ${cost})"
        
        # Make calls that exhaust budget
        results = []
        for i in range(4):
            try:
                result = await fallback_orchestrator.execute_with_fallback(
                    expensive_operation, {"query": f"expensive query {i}"}
                )
                results.append(result)
            except BudgetExhaustedError:
                break
        
        # Should have 3 successful calls before budget exhaustion
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 2
        
        # Last successful call should have used lower-cost fallback level
        if successful_results:
            last_result = successful_results[-1]
            assert last_result.level_used in [FallbackLevel.SECONDARY, FallbackLevel.TERTIARY]


class TestComplexFailureScenarios:
    """Test complex failure scenarios involving multiple systems"""
    
    @pytest.fixture
    def complex_system_setup(self):
        """Setup complex system with multiple components"""
        return {
            "fallback_orchestrator": MockFallbackOrchestrator(),
            "uncertainty_cascade": MockUncertaintyAwareCascade(),
            "circuit_breakers": {
                "primary": CircuitBreaker(failure_threshold=2, recovery_timeout=1.0),
                "secondary": CircuitBreaker(failure_threshold=3, recovery_timeout=2.0),
                "emergency": CircuitBreaker(failure_threshold=1, recovery_timeout=0.5)
            }
        }
    
    @pytest.mark.asyncio
    async def test_nested_fallback_with_circuit_protection(self, complex_system_setup):
        """Test nested fallback scenarios with circuit breaker protection"""
        
        system = complex_system_setup
        orchestrator = system["fallback_orchestrator"]
        cascade = system["uncertainty_cascade"]
        
        # Create a nested operation that uses both systems
        async def nested_operation(context, level=FallbackLevel.PRIMARY):
            # First try uncertainty cascade
            cascade_result = await cascade.execute_cascade(
                context.get("query", ""), context
            )
            
            if not cascade_result["success"]:
                raise Exception("Cascade operation failed")
            
            return f"Nested result: {cascade_result['result']}"
        
        # Break some cascade steps
        cascade.circuit_breakers["high_confidence"].state = 'open'
        cascade.circuit_breakers["medium_confidence"].state = 'open'
        
        # Execute nested operation through fallback orchestrator
        result = await orchestrator.execute_with_fallback(
            nested_operation, {"query": "nested test"}
        )
        
        assert result.success
        assert "Nested result" in result.result
    
    @pytest.mark.asyncio
    async def test_performance_optimization_during_fallbacks(self, complex_system_setup):
        """Test performance optimization during fallback scenarios"""
        
        system = complex_system_setup
        orchestrator = system["fallback_orchestrator"]
        
        # Create operation with different performance characteristics per level
        async def performance_aware_operation(context, level=FallbackLevel.PRIMARY):
            # Simulate different response times per level
            delays = {
                FallbackLevel.PRIMARY: 0.1,      # 100ms - high quality, slow
                FallbackLevel.SECONDARY: 0.05,   # 50ms - medium quality, fast
                FallbackLevel.TERTIARY: 0.02,    # 20ms - low quality, very fast
                FallbackLevel.EMERGENCY_CACHE: 0.001  # 1ms - cached, instant
            }
            
            await asyncio.sleep(delays.get(level, 0.1))
            
            return f"Performance optimized result at {level.value} level"
        
        # Open primary circuit breaker to force fallback
        orchestrator.circuit_breakers[FallbackLevel.PRIMARY.value].state = 'open'
        
        start_time = time.time()
        
        result = await orchestrator.execute_with_fallback(
            performance_aware_operation, {"query": "performance test"}
        )
        
        end_time = time.time()
        
        assert result.success
        assert result.level_used == FallbackLevel.SECONDARY  # Should use faster secondary
        assert (end_time - start_time) < 0.1  # Should be faster than primary
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_with_monitoring(self, complex_system_setup):
        """Test graceful degradation with monitoring and alerting"""
        
        system = complex_system_setup
        orchestrator = system["fallback_orchestrator"]
        
        # Track degradation metrics
        degradation_metrics = {
            "circuit_breaker_openings": 0,
            "fallback_activations": 0,
            "emergency_cache_uses": 0,
            "performance_degradations": 0
        }
        
        async def monitored_operation(context, level=FallbackLevel.PRIMARY):
            # Simulate different failure patterns
            if level == FallbackLevel.PRIMARY:
                degradation_metrics["circuit_breaker_openings"] += 1
                raise Exception("Primary service degraded")
            elif level == FallbackLevel.SECONDARY:
                degradation_metrics["fallback_activations"] += 1
                return "Secondary service response"
            elif level == FallbackLevel.TERTIARY:
                degradation_metrics["performance_degradations"] += 1
                return "Tertiary service response (degraded quality)"
            else:  # Emergency cache
                degradation_metrics["emergency_cache_uses"] += 1
                return "Emergency cache response"
        
        # Execute multiple requests to trigger degradation
        results = []
        for i in range(5):
            result = await orchestrator.execute_with_fallback(
                monitored_operation, {"query": f"monitoring test {i}"}
            )
            results.append(result)
        
        # Verify graceful degradation occurred
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 4  # Most should succeed despite degradation
        
        # Verify monitoring metrics were collected
        assert degradation_metrics["circuit_breaker_openings"] > 0
        assert degradation_metrics["fallback_activations"] > 0
        
        # Verify different fallback levels were used
        levels_used = {r.level_used for r in successful_results}
        assert FallbackLevel.SECONDARY in levels_used


if __name__ == "__main__":
    # Run tests with proper configuration
    pytest.main([__file__, "-v", "--tb=short"])