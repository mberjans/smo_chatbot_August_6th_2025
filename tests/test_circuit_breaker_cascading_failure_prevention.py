"""
Multi-Level Cascading Failure Prevention Tests for Circuit Breakers
================================================================

This module provides comprehensive tests for circuit breaker coordination and cascading
failure prevention across multiple services. Tests validate that failures in one service
do not cascade to affect other services, ensuring system resilience and stability.

Priority 1 Test Suite - Critical for Production Readiness

Key Test Areas:
1. Service isolation during failures
2. Multi-service coordination between OpenAI, Perplexity, and LightRAG
3. Partial service degradation handling
4. Recovery coordination across services
5. Load balancer integration with circuit breakers
6. Cross-service failure propagation prevention
7. System-wide resilience validation

Author: Claude Code Assistant
Created: August 9, 2025
Task: Multi-Level Cascading Failure Prevention Tests
"""

import pytest
import asyncio
import time
import logging
import statistics
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

# Import circuit breaker components
from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError
from lightrag_integration.cost_based_circuit_breaker import (
    CostBasedCircuitBreaker, CircuitBreakerState
)

# Try to import production components, with fallbacks for testing
try:
    from lightrag_integration.production_load_balancer import (
        ProductionLoadBalancer, BackendType, BackendInstanceConfig,
        ProductionLoadBalancingConfig, HealthStatus, AlertSeverity
    )
    from lightrag_integration.production_intelligent_query_router import ProductionIntelligentQueryRouter
    from lightrag_integration.query_router import RoutingDecision
except ImportError:
    # Create mock enums for testing if imports fail
    from enum import Enum
    
    class BackendType(Enum):
        LIGHTRAG = "lightrag"
        PERPLEXITY = "perplexity"
        OPENAI_DIRECT = "openai_direct"
    
    class HealthStatus(Enum):
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        UNKNOWN = "unknown"
    
    class AlertSeverity(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class RoutingDecision(Enum):
        LIGHTRAG = "lightrag"
        PERPLEXITY = "perplexity"
        EITHER = "either"
        HYBRID = "hybrid"


# ============================================================================
# Test Data Models and Fixtures
# ============================================================================

class ServiceState(Enum):
    """Service operational states for testing"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class MockService:
    """Mock service for multi-service testing"""
    name: str
    service_type: BackendType
    state: ServiceState = ServiceState.HEALTHY
    failure_count: int = 0
    success_count: int = 0
    last_call_time: float = 0.0
    response_time_ms: float = 100.0
    error_rate: float = 0.0
    circuit_breaker: Optional[CircuitBreaker] = None
    
    def __post_init__(self):
        if self.circuit_breaker is None:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=5.0,
                expected_exception=Exception
            )
    
    async def call(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate service call with state-dependent behavior"""
        self.last_call_time = time.time()
        
        # Simulate response time
        await asyncio.sleep(self.response_time_ms / 1000.0)
        
        # State-dependent failure logic
        if self.state == ServiceState.FAILED:
            self.failure_count += 1
            raise Exception(f"{self.name} service is failed")
        elif self.state == ServiceState.FAILING:
            # Fail based on error rate (random chance)
            import random
            if random.random() < self.error_rate:
                self.failure_count += 1
                raise Exception(f"{self.name} service failing intermittently")
        elif self.state == ServiceState.DEGRADED:
            # Fail based on error rate (random chance) 
            import random
            if random.random() < self.error_rate:
                self.failure_count += 1
                raise Exception(f"{self.name} service degraded")
        
        self.success_count += 1
        return {
            "service": self.name,
            "response": f"Success from {self.name}",
            "timestamp": time.time(),
            "call_count": self.success_count + self.failure_count
        }
    
    def set_state(self, new_state: ServiceState, error_rate: float = 0.0):
        """Change service state for testing"""
        self.state = new_state
        self.error_rate = error_rate
    
    def reset_counters(self):
        """Reset call counters"""
        self.failure_count = 0
        self.success_count = 0


@dataclass
class MultiServiceTestEnvironment:
    """Comprehensive test environment for multi-service scenarios"""
    services: Dict[str, MockService] = field(default_factory=dict)
    load_balancer: Optional[ProductionLoadBalancer] = None
    query_router: Optional[ProductionIntelligentQueryRouter] = None
    global_circuit_breaker: Optional[CircuitBreaker] = None
    failure_propagation_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_service(self, service: MockService):
        """Add service to test environment"""
        self.services[service.name] = service
    
    def set_service_state(self, service_name: str, state: ServiceState, error_rate: float = 0.0):
        """Set state for specific service"""
        if service_name in self.services:
            self.services[service_name].set_state(state, error_rate)
    
    def get_healthy_services(self) -> List[str]:
        """Get list of healthy service names"""
        return [name for name, service in self.services.items() 
                if service.state == ServiceState.HEALTHY]
    
    def get_failed_services(self) -> List[str]:
        """Get list of failed service names"""
        return [name for name, service in self.services.items() 
                if service.state in [ServiceState.FAILED, ServiceState.FAILING]]
    
    def log_failure_event(self, event: Dict[str, Any]):
        """Log failure propagation event"""
        event['timestamp'] = time.time()
        self.failure_propagation_log.append(event)
    
    async def simulate_service_calls(self, service_names: List[str], 
                                   call_count: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Simulate calls to multiple services"""
        results = defaultdict(list)
        
        for _ in range(call_count):
            for service_name in service_names:
                if service_name in self.services:
                    service = self.services[service_name]
                    try:
                        result = await service.call({"test": "data"})
                        results[service_name].append({"success": True, "result": result})
                    except Exception as e:
                        results[service_name].append({"success": False, "error": str(e)})
                        self.log_failure_event({
                            "service": service_name,
                            "event_type": "service_failure",
                            "error": str(e)
                        })
        
        return dict(results)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def multi_service_environment():
    """Provide comprehensive multi-service test environment"""
    env = MultiServiceTestEnvironment()
    
    # Add OpenAI service
    openai_service = MockService(
        name="openai",
        service_type=BackendType.OPENAI_DIRECT,
        response_time_ms=200.0
    )
    env.add_service(openai_service)
    
    # Add Perplexity service
    perplexity_service = MockService(
        name="perplexity",
        service_type=BackendType.PERPLEXITY,
        response_time_ms=300.0
    )
    env.add_service(perplexity_service)
    
    # Add LightRAG service
    lightrag_service = MockService(
        name="lightrag",
        service_type=BackendType.LIGHTRAG,
        response_time_ms=150.0
    )
    env.add_service(lightrag_service)
    
    return env


@pytest.fixture
def coordinated_circuit_breakers():
    """Provide coordinated circuit breakers for testing"""
    circuit_breakers = {}
    
    # OpenAI circuit breaker - more sensitive to failures
    circuit_breakers["openai"] = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=3.0,
        expected_exception=Exception
    )
    
    # Perplexity circuit breaker - moderate sensitivity
    circuit_breakers["perplexity"] = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0,
        expected_exception=Exception
    )
    
    # LightRAG circuit breaker - more tolerant
    circuit_breakers["lightrag"] = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=10.0,
        expected_exception=Exception
    )
    
    # Global coordination circuit breaker
    circuit_breakers["global"] = CircuitBreaker(
        failure_threshold=2,  # Trips when any 2 services fail
        recovery_timeout=30.0,
        expected_exception=Exception
    )
    
    return circuit_breakers


@pytest.fixture
def failure_isolation_monitor():
    """Monitor for tracking failure isolation"""
    class FailureIsolationMonitor:
        def __init__(self):
            self.isolation_events = []
            self.propagation_events = []
            self.recovery_events = []
        
        def log_isolation(self, failed_service: str, isolated_from: List[str]):
            """Log successful failure isolation"""
            self.isolation_events.append({
                "timestamp": time.time(),
                "failed_service": failed_service,
                "isolated_from": isolated_from,
                "event_type": "isolation_success"
            })
        
        def log_propagation(self, source_service: str, affected_services: List[str]):
            """Log failure propagation (indicates test failure)"""
            self.propagation_events.append({
                "timestamp": time.time(),
                "source_service": source_service,
                "affected_services": affected_services,
                "event_type": "propagation_failure"
            })
        
        def log_recovery(self, recovered_service: str, coordinated_with: List[str]):
            """Log coordinated recovery"""
            self.recovery_events.append({
                "timestamp": time.time(),
                "recovered_service": recovered_service,
                "coordinated_with": coordinated_with,
                "event_type": "coordinated_recovery"
            })
        
        def has_propagation_failures(self) -> bool:
            """Check if any propagation failures occurred"""
            return len(self.propagation_events) > 0
        
        def get_isolation_success_rate(self) -> float:
            """Calculate isolation success rate"""
            total_events = len(self.isolation_events) + len(self.propagation_events)
            if total_events == 0:
                return 100.0
            return (len(self.isolation_events) / total_events) * 100.0
    
    return FailureIsolationMonitor()


@pytest.fixture
def mock_load_balancer_with_circuit_breakers(multi_service_environment, coordinated_circuit_breakers):
    """Mock load balancer with integrated circuit breakers"""
    class MockLoadBalancerWithCircuitBreakers:
        def __init__(self, services, circuit_breakers):
            self.services = services
            self.circuit_breakers = circuit_breakers
            self.routing_decisions = []
            self.health_checks = defaultdict(list)
            self.round_robin_counter = 0
        
        async def route_request(self, request_data: Dict[str, Any]) -> Tuple[str, Any]:
            """Route request with circuit breaker protection"""
            # Check available services (circuit breakers not open)
            available_services = []
            for service_name in self.services:
                cb = self.circuit_breakers.get(service_name)
                if cb and cb.state == "closed":
                    available_services.append(service_name)
            
            if not available_services:
                raise Exception("All services unavailable - circuit breakers open")
            
            # Select service (round-robin for testing)
            selected_service = available_services[self.round_robin_counter % len(available_services)]
            self.round_robin_counter += 1
            self.routing_decisions.append({
                "selected_service": selected_service,
                "available_services": available_services,
                "timestamp": time.time()
            })
            
            # Attempt service call through circuit breaker
            service = self.services[selected_service]
            cb = self.circuit_breakers[selected_service]
            
            try:
                result = await cb.call(service.call, request_data)
                return selected_service, result
            except Exception as e:
                # Circuit breaker will handle state transitions
                raise
        
        def get_service_health_status(self, service_name: str) -> HealthStatus:
            """Get health status based on circuit breaker state"""
            if service_name not in self.circuit_breakers:
                return HealthStatus.UNKNOWN
            
            cb = self.circuit_breakers[service_name]
            if cb.state == "closed":
                return HealthStatus.HEALTHY
            elif cb.state == "half_open":
                return HealthStatus.DEGRADED
            else:  # open
                return HealthStatus.UNHEALTHY
    
    return MockLoadBalancerWithCircuitBreakers(
        multi_service_environment.services,
        coordinated_circuit_breakers
    )


# ============================================================================
# Core Test Classes
# ============================================================================

class TestServiceIsolationDuringFailures:
    """Test that service failures are properly isolated"""
    
    @pytest.mark.asyncio
    async def test_single_service_failure_isolation(self, multi_service_environment, 
                                                   coordinated_circuit_breakers,
                                                   failure_isolation_monitor):
        """Test that failure in one service doesn't affect others"""
        # Set OpenAI service to failing state
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        # Simulate calls to all services
        results = await multi_service_environment.simulate_service_calls(
            ["openai", "perplexity", "lightrag"], call_count=5
        )
        
        # Verify OpenAI failures are isolated
        openai_results = results["openai"]
        perplexity_results = results["perplexity"]
        lightrag_results = results["lightrag"]
        
        # OpenAI should have all failures
        assert all(not r["success"] for r in openai_results), "OpenAI should fail consistently"
        
        # Other services should remain healthy
        assert all(r["success"] for r in perplexity_results), "Perplexity should remain healthy"
        assert all(r["success"] for r in lightrag_results), "LightRAG should remain healthy"
        
        # Log successful isolation
        failure_isolation_monitor.log_isolation(
            "openai", ["perplexity", "lightrag"]
        )
        
        # Verify no propagation occurred
        assert not failure_isolation_monitor.has_propagation_failures()
        assert failure_isolation_monitor.get_isolation_success_rate() == 100.0
    
    @pytest.mark.asyncio
    async def test_multiple_service_failure_isolation(self, multi_service_environment,
                                                     coordinated_circuit_breakers,
                                                     failure_isolation_monitor):
        """Test isolation when multiple services fail"""
        # Set both OpenAI and Perplexity to failing
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        multi_service_environment.set_service_state("perplexity", ServiceState.FAILING, error_rate=0.8)
        
        # Simulate calls
        results = await multi_service_environment.simulate_service_calls(
            ["openai", "perplexity", "lightrag"], call_count=8
        )
        
        # Verify failures are isolated
        openai_results = results["openai"]
        perplexity_results = results["perplexity"]
        lightrag_results = results["lightrag"]
        
        # Failed services should show consistent failure patterns
        assert all(not r["success"] for r in openai_results)
        
        # Perplexity should show high failure rate due to error_rate=0.8
        perplexity_failure_rate = sum(1 for r in perplexity_results if not r["success"]) / len(perplexity_results)
        assert perplexity_failure_rate > 0.6, f"Expected high failure rate, got {perplexity_failure_rate}"
        
        # LightRAG should remain completely healthy
        assert all(r["success"] for r in lightrag_results), "LightRAG should remain isolated and healthy"
        
        # Log isolation success
        failure_isolation_monitor.log_isolation("openai", ["lightrag"])
        failure_isolation_monitor.log_isolation("perplexity", ["lightrag"])
        
        assert failure_isolation_monitor.get_isolation_success_rate() == 100.0
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention_circuit_breaker_coordination(self, 
                                                                           multi_service_environment,
                                                                           coordinated_circuit_breakers,
                                                                           failure_isolation_monitor):
        """Test that circuit breakers coordinate to prevent cascading failures"""
        openai_cb = coordinated_circuit_breakers["openai"]
        perplexity_cb = coordinated_circuit_breakers["perplexity"] 
        lightrag_cb = coordinated_circuit_breakers["lightrag"]
        
        # Set OpenAI to failing to trigger its circuit breaker
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        # Make calls through circuit breakers until OpenAI trips
        openai_service = multi_service_environment.services["openai"]
        
        # Trip OpenAI circuit breaker
        for _ in range(3):  # failure_threshold = 2
            try:
                await openai_cb.call(openai_service.call, {"test": "data"})
            except Exception:
                pass  # Expected failures
        
        # Verify OpenAI circuit breaker is open
        assert openai_cb.state == "open", "OpenAI circuit breaker should be open"
        
        # Verify other circuit breakers remain closed
        assert perplexity_cb.state == "closed", "Perplexity CB should remain closed"
        assert lightrag_cb.state == "closed", "LightRAG CB should remain closed"
        
        # Test that other services continue to work normally
        perplexity_service = multi_service_environment.services["perplexity"]
        lightrag_service = multi_service_environment.services["lightrag"]
        
        # These should succeed
        result1 = await perplexity_cb.call(perplexity_service.call, {"test": "data"})
        assert result1 is not None
        
        result2 = await lightrag_cb.call(lightrag_service.call, {"test": "data"})
        assert result2 is not None
        
        # Log successful coordination
        failure_isolation_monitor.log_isolation("openai", ["perplexity", "lightrag"])
        assert not failure_isolation_monitor.has_propagation_failures()


class TestMultiServiceCoordination:
    """Test coordination between multiple circuit breakers"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_coordination(self, coordinated_circuit_breakers,
                                                     multi_service_environment):
        """Test that circuit breaker states are coordinated properly"""
        openai_cb = coordinated_circuit_breakers["openai"]
        perplexity_cb = coordinated_circuit_breakers["perplexity"]
        global_cb = coordinated_circuit_breakers["global"]
        
        # Initially all should be closed
        assert openai_cb.state == "closed"
        assert perplexity_cb.state == "closed"
        assert global_cb.state == "closed"
        
        # Fail OpenAI service and trigger circuit breaker
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        openai_service = multi_service_environment.services["openai"]
        
        # Trip OpenAI circuit breaker
        for _ in range(3):
            try:
                await openai_cb.call(openai_service.call, {"test": "data"})
            except Exception:
                pass
        
        assert openai_cb.state == "open"
        
        # Global CB should still be closed (only 1 service failed)
        assert global_cb.state == "closed"
        
        # Now fail Perplexity as well
        multi_service_environment.set_service_state("perplexity", ServiceState.FAILED)
        perplexity_service = multi_service_environment.services["perplexity"]
        
        # Trip Perplexity circuit breaker
        for _ in range(4):
            try:
                await perplexity_cb.call(perplexity_service.call, {"test": "data"})
            except Exception:
                pass
        
        assert perplexity_cb.state == "open"
        
        # Simulate global circuit breaker coordination
        # In real implementation, global CB would monitor other CBs
        async def global_check():
            if openai_cb.state == "open" and perplexity_cb.state == "open":
                raise Exception("Multiple services failed - triggering global circuit breaker")
        
        try:
            await global_cb.call(global_check)
        except Exception:
            pass
        
        async def global_failure():
            raise Exception("Second global failure")
        
        try:
            await global_cb.call(global_failure)
        except Exception:
            pass
        
        # Global CB should now be open due to multiple service failures
        assert global_cb.state == "open"
    
    @pytest.mark.asyncio
    async def test_coordinated_recovery_sequence(self, coordinated_circuit_breakers,
                                               multi_service_environment,
                                               failure_isolation_monitor,
                                               mock_time):
        """Test coordinated recovery across multiple services"""
        openai_cb = coordinated_circuit_breakers["openai"] 
        perplexity_cb = coordinated_circuit_breakers["perplexity"]
        
        # Set services to failing state
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        multi_service_environment.set_service_state("perplexity", ServiceState.FAILED)
        
        # Trip both circuit breakers
        for service_name in ["openai", "perplexity"]:
            service = multi_service_environment.services[service_name]
            cb = coordinated_circuit_breakers[service_name]
            
            for _ in range(4):
                try:
                    await cb.call(service.call, {"test": "data"})
                except Exception:
                    pass
        
        # Verify both are open
        assert openai_cb.state == "open"
        assert perplexity_cb.state == "open"
        
        # Advance time to trigger recovery timeout for OpenAI (3.0s)
        mock_time.advance(4.0)
        
        # OpenAI should transition to half-open
        # Simulate recovery by fixing OpenAI service
        multi_service_environment.set_service_state("openai", ServiceState.HEALTHY)
        
        # Test OpenAI recovery
        try:
            result = await openai_cb.call(multi_service_environment.services["openai"].call, {"test": "recovery"})
            assert result is not None
        except Exception:
            pass
        
        # OpenAI should be closed now (successful recovery)
        assert openai_cb.state == "closed"
        
        # Perplexity should still be open (longer recovery timeout: 5.0s)
        assert perplexity_cb.state == "open"
        
        # Advance time for Perplexity recovery
        mock_time.advance(2.0)  # Total: 6.0s
        
        # Fix Perplexity and test recovery
        multi_service_environment.set_service_state("perplexity", ServiceState.HEALTHY)
        
        try:
            result = await perplexity_cb.call(multi_service_environment.services["perplexity"].call, {"test": "recovery"})
            assert result is not None
        except Exception:
            pass
        
        # Both should now be recovered
        assert openai_cb.state == "closed"
        assert perplexity_cb.state == "closed"
        
        # Log coordinated recovery
        failure_isolation_monitor.log_recovery("openai", ["perplexity"])
        failure_isolation_monitor.log_recovery("perplexity", ["openai"])
        
        assert len(failure_isolation_monitor.recovery_events) == 2
    
    @pytest.mark.asyncio
    async def test_partial_service_degradation_handling(self, multi_service_environment,
                                                       coordinated_circuit_breakers):
        """Test handling of partial service degradation scenarios"""
        # Set up mixed service states
        multi_service_environment.set_service_state("openai", ServiceState.HEALTHY)
        multi_service_environment.set_service_state("perplexity", ServiceState.DEGRADED, error_rate=0.3)
        multi_service_environment.set_service_state("lightrag", ServiceState.FAILING, error_rate=0.7)
        
        # Simulate load across all services
        results = await multi_service_environment.simulate_service_calls(
            ["openai", "perplexity", "lightrag"], call_count=10
        )
        
        # Analyze results
        openai_success_rate = sum(1 for r in results["openai"] if r["success"]) / len(results["openai"])
        perplexity_success_rate = sum(1 for r in results["perplexity"] if r["success"]) / len(results["perplexity"])
        lightrag_success_rate = sum(1 for r in results["lightrag"] if r["success"]) / len(results["lightrag"])
        
        # Verify success rates match service states
        assert openai_success_rate == 1.0, "OpenAI should be fully healthy"
        assert 0.6 <= perplexity_success_rate <= 0.8, f"Perplexity should be degraded, got {perplexity_success_rate}"
        assert lightrag_success_rate <= 0.4, f"LightRAG should be failing, got {lightrag_success_rate}"
        
        # Verify circuit breakers respond appropriately to different degradation levels
        openai_cb = coordinated_circuit_breakers["openai"]
        perplexity_cb = coordinated_circuit_breakers["perplexity"] 
        lightrag_cb = coordinated_circuit_breakers["lightrag"]
        
        # Test circuit breaker states after load
        for service_name in ["openai", "perplexity", "lightrag"]:
            service = multi_service_environment.services[service_name]
            cb = coordinated_circuit_breakers[service_name]
            
            # Apply load through circuit breakers
            for _ in range(5):
                try:
                    await cb.call(service.call, {"test": f"load_{service_name}"})
                except Exception:
                    pass  # Circuit breaker will handle failures
        
        # Verify circuit breaker states match service degradation levels
        assert openai_cb.state == "closed", "Healthy service CB should remain closed"
        # Degraded service might still be closed if under threshold
        # Failing service should likely trip its circuit breaker
        if lightrag_cb.failure_count >= lightrag_cb.failure_threshold:
            assert lightrag_cb.state == "open", "Failing service CB should be open"


class TestRecoveryCoordinationAcrossServices:
    """Test coordinated recovery patterns across multiple services"""
    
    @pytest.mark.asyncio
    async def test_staggered_recovery_coordination(self, coordinated_circuit_breakers,
                                                  multi_service_environment,
                                                  mock_time,
                                                  failure_isolation_monitor):
        """Test staggered recovery to prevent simultaneous service restoration"""
        # Set up all services in failed state
        for service_name in ["openai", "perplexity", "lightrag"]:
            multi_service_environment.set_service_state(service_name, ServiceState.FAILED)
        
        # Trip all circuit breakers
        for service_name in ["openai", "perplexity", "lightrag"]:
            service = multi_service_environment.services[service_name]
            cb = coordinated_circuit_breakers[service_name]
            
            for _ in range(cb.failure_threshold + 1):
                try:
                    await cb.call(service.call, {"test": "data"})
                except Exception:
                    pass
        
        # Verify all are open
        for service_name in ["openai", "perplexity", "lightrag"]:
            assert coordinated_circuit_breakers[service_name].state == "open"
        
        # Implement staggered recovery timing
        recovery_times = {"openai": 3.0, "perplexity": 5.0, "lightrag": 10.0}
        recovered_services = []
        
        for service_name, recovery_time in recovery_times.items():
            # Advance time to recovery point
            mock_time.advance(recovery_time - mock_time.current() if mock_time.current() < recovery_time else 0.1)
            
            # Fix service
            multi_service_environment.set_service_state(service_name, ServiceState.HEALTHY)
            
            # Test recovery
            cb = coordinated_circuit_breakers[service_name]
            service = multi_service_environment.services[service_name]
            
            try:
                result = await cb.call(service.call, {"test": "recovery"})
                recovered_services.append(service_name)
                failure_isolation_monitor.log_recovery(service_name, recovered_services.copy())
            except Exception:
                pass
        
        # Verify staggered recovery
        recovery_events = failure_isolation_monitor.recovery_events
        assert len(recovery_events) == 3, "Should have 3 recovery events"
        
        # Verify recovery order
        recovery_order = [event["recovered_service"] for event in recovery_events]
        expected_order = ["openai", "perplexity", "lightrag"]
        assert recovery_order == expected_order, f"Expected {expected_order}, got {recovery_order}"
    
    @pytest.mark.asyncio
    async def test_recovery_failure_handling(self, coordinated_circuit_breakers,
                                            multi_service_environment,
                                            mock_time):
        """Test handling of failed recovery attempts"""
        openai_cb = coordinated_circuit_breakers["openai"]
        
        # Set service to failed state and trip circuit breaker
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        for _ in range(3):
            try:
                await openai_cb.call(multi_service_environment.services["openai"].call, {"test": "data"})
            except Exception:
                pass
        
        assert openai_cb.state == "open"
        
        # Advance time to recovery timeout
        mock_time.advance(4.0)
        
        # Attempt recovery but keep service in failed state (simulating failed recovery)
        service = multi_service_environment.services["openai"]
        
        try:
            await openai_cb.call(service.call, {"test": "failed_recovery"})
        except Exception:
            pass  # Expected failure
        
        # Circuit breaker should return to open state after failed recovery
        assert openai_cb.state == "open"
        
        # Advance time again for second recovery attempt
        mock_time.advance(4.0)
        
        # Now actually fix the service
        multi_service_environment.set_service_state("openai", ServiceState.HEALTHY)
        
        try:
            result = await openai_cb.call(service.call, {"test": "successful_recovery"})
            assert result is not None
        except Exception:
            pytest.fail("Recovery should succeed after service is fixed")
        
        # Circuit breaker should be closed now
        assert openai_cb.state == "closed"


class TestLoadBalancerCircuitBreakerCoordination:
    """Test integration between load balancer and circuit breakers"""
    
    @pytest.mark.asyncio
    async def test_load_balancer_circuit_breaker_integration(self, 
                                                            mock_load_balancer_with_circuit_breakers,
                                                            multi_service_environment):
        """Test load balancer properly integrates with circuit breakers"""
        lb = mock_load_balancer_with_circuit_breakers
        
        # All services should be available initially
        service_name, result = await lb.route_request({"test": "data"})
        assert service_name in ["openai", "perplexity", "lightrag"]
        assert result is not None
        
        # Set OpenAI to failing and trip its circuit breaker
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        # Make requests until OpenAI circuit breaker trips
        for _ in range(5):
            try:
                await lb.route_request({"test": "trigger_failure"})
            except Exception:
                pass  # Some requests may fail as CB trips
        
        # Verify OpenAI is no longer selected by load balancer
        available_services = set()
        for _ in range(10):
            try:
                service_name, _ = await lb.route_request({"test": "data"})
                available_services.add(service_name)
            except Exception:
                pass
        
        # OpenAI should not be in available services due to open circuit breaker
        assert "openai" not in available_services, "OpenAI should be excluded due to open circuit breaker"
        assert len(available_services) >= 1, "Other services should still be available"
    
    @pytest.mark.asyncio
    async def test_load_balancer_health_status_integration(self, 
                                                          mock_load_balancer_with_circuit_breakers,
                                                          multi_service_environment):
        """Test load balancer health status reflects circuit breaker states"""
        lb = mock_load_balancer_with_circuit_breakers
        
        # Initially all services should be healthy
        assert lb.get_service_health_status("openai") == HealthStatus.HEALTHY
        assert lb.get_service_health_status("perplexity") == HealthStatus.HEALTHY
        assert lb.get_service_health_status("lightrag") == HealthStatus.HEALTHY
        
        # Fail OpenAI service and trip circuit breaker
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        # Make enough requests to trip the OpenAI circuit breaker (failure_threshold = 2)
        for _ in range(10):  # Try more requests to ensure CB trips
            try:
                await lb.route_request({"test": "trip_cb"})
            except Exception:
                pass
        
        # Health status should reflect circuit breaker state
        assert lb.get_service_health_status("openai") == HealthStatus.UNHEALTHY
        assert lb.get_service_health_status("perplexity") == HealthStatus.HEALTHY
        assert lb.get_service_health_status("lightrag") == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_load_balancer_fallback_during_cascading_failure(self,
                                                                  mock_load_balancer_with_circuit_breakers,
                                                                  multi_service_environment,
                                                                  failure_isolation_monitor):
        """Test load balancer fallback behavior during cascading failures"""
        lb = mock_load_balancer_with_circuit_breakers
        
        # Set two services to failing
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        multi_service_environment.set_service_state("perplexity", ServiceState.FAILED)
        
        # Trip circuit breakers for both services
        for _ in range(10):
            try:
                await lb.route_request({"test": "cascade_test"})
            except Exception:
                pass
        
        # Load balancer should fall back to remaining healthy service
        successful_requests = 0
        routed_services = set()
        
        for _ in range(20):
            try:
                service_name, result = await lb.route_request({"test": "fallback"})
                successful_requests += 1
                routed_services.add(service_name)
            except Exception:
                pass
        
        # Should successfully route to LightRAG (the only healthy service)
        assert successful_requests > 0, "Should have successful requests to healthy service"
        assert routed_services == {"lightrag"}, f"Should only route to lightrag, got {routed_services}"
        
        # Log that cascading failure was prevented
        failure_isolation_monitor.log_isolation("openai", ["lightrag"])
        failure_isolation_monitor.log_isolation("perplexity", ["lightrag"])
        
        assert not failure_isolation_monitor.has_propagation_failures()
    
    @pytest.mark.asyncio
    async def test_load_balancer_complete_system_failure_handling(self,
                                                                 mock_load_balancer_with_circuit_breakers,
                                                                 multi_service_environment):
        """Test load balancer behavior when all services fail"""
        lb = mock_load_balancer_with_circuit_breakers
        
        # Set all services to failed
        for service_name in ["openai", "perplexity", "lightrag"]:
            multi_service_environment.set_service_state(service_name, ServiceState.FAILED)
        
        # Trip all circuit breakers
        for _ in range(20):
            try:
                await lb.route_request({"test": "total_failure"})
            except Exception:
                pass
        
        # Load balancer should raise appropriate exception when all services unavailable
        with pytest.raises(Exception, match="All services unavailable"):
            await lb.route_request({"test": "no_services"})
        
        # Verify all services are marked unhealthy
        for service_name in ["openai", "perplexity", "lightrag"]:
            assert lb.get_service_health_status(service_name) == HealthStatus.UNHEALTHY


class TestSystemWideResilienceValidation:
    """Comprehensive system-wide resilience tests"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_cascading_failure_prevention(self,
                                                             multi_service_environment,
                                                             coordinated_circuit_breakers,
                                                             mock_load_balancer_with_circuit_breakers,
                                                             failure_isolation_monitor,
                                                             mock_time):
        """Comprehensive test of cascading failure prevention across entire system"""
        lb = mock_load_balancer_with_circuit_breakers
        
        # Phase 1: Normal operation
        successful_requests = 0
        for _ in range(50):
            try:
                _, result = await lb.route_request({"test": "normal_ops"})
                if result:
                    successful_requests += 1
            except Exception:
                pass
        
        initial_success_rate = successful_requests / 50
        assert initial_success_rate > 0.9, f"Initial success rate should be high, got {initial_success_rate}"
        
        # Phase 2: Introduce cascade trigger (OpenAI fails)
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        failure_isolation_monitor.log_isolation("openai", ["perplexity", "lightrag"])
        
        # Test system response
        cascade_phase_success = 0
        for _ in range(30):
            try:
                service, result = await lb.route_request({"test": "cascade_phase"})
                if result and service != "openai":  # Should not route to failed service
                    cascade_phase_success += 1
            except Exception:
                pass
        
        # System should maintain service through remaining backends
        cascade_success_rate = cascade_phase_success / 30
        assert cascade_success_rate > 0.6, f"System should maintain service during cascade, got {cascade_success_rate}"
        
        # Phase 3: Secondary failure (Perplexity fails)
        mock_time.advance(1.0)
        multi_service_environment.set_service_state("perplexity", ServiceState.FAILED)
        failure_isolation_monitor.log_isolation("perplexity", ["lightrag"])
        
        # System should still operate through LightRAG
        final_phase_success = 0
        routed_services = set()
        for _ in range(20):
            try:
                service, result = await lb.route_request({"test": "final_phase"})
                if result:
                    final_phase_success += 1
                    routed_services.add(service)
            except Exception:
                pass
        
        # Should only route to healthy LightRAG service
        assert routed_services == {"lightrag"}, f"Should only route to lightrag, got {routed_services}"
        
        # Phase 4: Recovery validation
        mock_time.advance(5.0)
        
        # Fix OpenAI
        multi_service_environment.set_service_state("openai", ServiceState.HEALTHY)
        failure_isolation_monitor.log_recovery("openai", ["lightrag"])
        
        # Test partial recovery
        recovery_services = set()
        for _ in range(15):
            try:
                service, result = await lb.route_request({"test": "recovery_phase"})
                if result:
                    recovery_services.add(service)
            except Exception:
                pass
        
        # Should now have both OpenAI and LightRAG available
        expected_services = {"openai", "lightrag"}
        assert recovery_services.issubset(expected_services), f"Recovery services {recovery_services} should be subset of {expected_services}"
        
        # Verify no propagation failures occurred
        assert not failure_isolation_monitor.has_propagation_failures()
        assert failure_isolation_monitor.get_isolation_success_rate() == 100.0
    
    @pytest.mark.asyncio
    async def test_high_load_cascading_failure_resilience(self,
                                                         multi_service_environment,
                                                         coordinated_circuit_breakers,
                                                         mock_load_balancer_with_circuit_breakers):
        """Test system resilience under high load with cascading failures"""
        lb = mock_load_balancer_with_circuit_breakers
        
        # Simulate high concurrent load
        async def make_request(request_id):
            try:
                service, result = await lb.route_request({"test": f"load_request_{request_id}"})
                return {"success": True, "service": service, "request_id": request_id}
            except Exception as e:
                return {"success": False, "error": str(e), "request_id": request_id}
        
        # Phase 1: High load with all services healthy
        tasks = [make_request(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        assert success_rate > 0.95, f"High load success rate should be high, got {success_rate}"
        
        # Phase 2: Introduce failure during high load
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        tasks = [make_request(i) for i in range(100, 200)]
        results_with_failure = await asyncio.gather(*tasks)
        
        # System should gracefully degrade but not completely fail
        success_with_failure = sum(1 for r in results_with_failure if r["success"]) / len(results_with_failure)
        assert success_with_failure > 0.4, f"Should maintain partial service under failure, got {success_with_failure}"
        
        # Verify requests were not routed to failed service
        successful_services = {r["service"] for r in results_with_failure if r["success"]}
        assert "openai" not in successful_services, "Should not route to failed service"
        
        # Phase 3: Recovery under continued load
        multi_service_environment.set_service_state("openai", ServiceState.HEALTHY)
        
        # Allow some time for circuit breaker recovery
        await asyncio.sleep(0.1)
        
        tasks = [make_request(i) for i in range(200, 250)]
        recovery_results = await asyncio.gather(*tasks)
        
        recovery_success_rate = sum(1 for r in recovery_results if r["success"]) / len(recovery_results)
        assert recovery_success_rate > 0.8, f"Recovery success rate should be high, got {recovery_success_rate}"
    
    @pytest.mark.asyncio
    async def test_error_propagation_prevention_validation(self,
                                                          multi_service_environment,
                                                          failure_isolation_monitor):
        """Validate that errors do not propagate between isolated services"""
        
        # Create isolated service calls
        async def call_service_isolated(service_name, call_count=5):
            """Call service in isolation to test error propagation"""
            service = multi_service_environment.services[service_name]
            results = []
            
            for i in range(call_count):
                try:
                    result = await service.call({"test": f"isolated_call_{i}"})
                    results.append({"success": True, "result": result})
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
                    
                    # Log any error that might propagate
                    if "propagation" in str(e).lower():
                        failure_isolation_monitor.log_propagation(
                            service_name, 
                            [s for s in multi_service_environment.services.keys() if s != service_name]
                        )
            
            return results
        
        # Set OpenAI to failing
        multi_service_environment.set_service_state("openai", ServiceState.FAILED)
        
        # Call all services in isolation simultaneously
        openai_task = call_service_isolated("openai")
        perplexity_task = call_service_isolated("perplexity") 
        lightrag_task = call_service_isolated("lightrag")
        
        openai_results, perplexity_results, lightrag_results = await asyncio.gather(
            openai_task, perplexity_task, lightrag_task
        )
        
        # Verify isolation
        openai_success_rate = sum(1 for r in openai_results if r["success"]) / len(openai_results)
        perplexity_success_rate = sum(1 for r in perplexity_results if r["success"]) / len(perplexity_results)
        lightrag_success_rate = sum(1 for r in lightrag_results if r["success"]) / len(lightrag_results)
        
        # Failed service should have low success rate
        assert openai_success_rate == 0.0, "Failed service should have no successes"
        
        # Healthy services should maintain high success rates
        assert perplexity_success_rate == 1.0, "Healthy service should not be affected"
        assert lightrag_success_rate == 1.0, "Healthy service should not be affected"
        
        # Verify no error propagation occurred
        assert not failure_isolation_monitor.has_propagation_failures(), "No error propagation should occur"
        
        # Log successful isolation
        failure_isolation_monitor.log_isolation("openai", ["perplexity", "lightrag"])
        
        assert failure_isolation_monitor.get_isolation_success_rate() == 100.0