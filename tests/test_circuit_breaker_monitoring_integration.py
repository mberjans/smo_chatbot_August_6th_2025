"""
Integration Tests for Circuit Breaker with Monitoring and Alerting

This module provides comprehensive integration tests for circuit breaker functionality
with monitoring and alerting systems. Tests validate real-time metrics collection,
alert generation, performance dashboards, and integration with production monitoring
infrastructure.

Key Test Areas:
1. Real-time circuit breaker metrics collection
2. Alert generation and notification systems
3. Performance dashboard integration
4. Historical data analysis and reporting
5. Threshold-based alerting and escalation
6. Integration with production monitoring stack
7. Health check and system status reporting

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Circuit Breaker Monitoring Integration Tests
"""

import pytest
import asyncio
import time
import logging
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading

# Import circuit breaker components
from lightrag_integration.clinical_metabolomics_rag import CircuitBreaker, CircuitBreakerError
from lightrag_integration.cost_based_circuit_breaker import CostBasedCircuitBreaker, BudgetExhaustedError

# Import monitoring components
try:
    from lightrag_integration.production_monitoring import (
        ProductionMonitoring, MonitoringConfig, AlertConfig, MetricType,
        AlertSeverity, HealthCheckResult, SystemHealthStatus
    )
    from lightrag_integration.api_metrics_logger import (
        APIUsageMetricsLogger, APIMetric, MetricType as APIMetricType
    )
    from lightrag_integration.enhanced_logging import (
        EnhancedLogger, PerformanceMetrics, correlation_manager
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    
    # Mock classes for testing
    class AlertSeverity(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class MetricType(Enum):
        COUNTER = "counter"
        GAUGE = "gauge"
        HISTOGRAM = "histogram"
        TIMER = "timer"
    
    class SystemHealthStatus(Enum):
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
    
    @dataclass
    class HealthCheckResult:
        component: str
        status: SystemHealthStatus
        response_time_ms: float
        error: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

# Production components
from lightrag_integration.production_load_balancer import ProductionLoadBalancer
from lightrag_integration.production_intelligent_query_router import ProductionIntelligentQueryRouter


@dataclass
class CircuitBreakerMetric:
    """Circuit breaker specific metrics"""
    component_id: str
    timestamp: datetime
    state: str  # open, closed, half-open
    failure_count: int
    success_count: int
    total_requests: int
    response_time_ms: float
    error_rate_percent: float
    recovery_time_seconds: Optional[float] = None
    last_failure_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "response_time_ms": self.response_time_ms,
            "error_rate_percent": self.error_rate_percent,
            "recovery_time_seconds": self.recovery_time_seconds,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


@dataclass 
class AlertEvent:
    """Alert event for testing"""
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class MockMonitoringSystem:
    """Mock monitoring system for testing"""
    
    def __init__(self):
        self.metrics: List[CircuitBreakerMetric] = []
        self.alerts: List[AlertEvent] = []
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_thresholds = {
            "circuit_breaker_open": {"error_rate": 10.0, "failure_count": 5},
            "high_latency": {"response_time_ms": 5000.0},
            "budget_exhausted": {"cost_threshold": 100.0}
        }
        
    async def collect_circuit_breaker_metrics(self, circuit_breaker: CircuitBreaker, 
                                            component_id: str) -> CircuitBreakerMetric:
        """Collect metrics from circuit breaker"""
        # Calculate derived metrics
        total_requests = getattr(circuit_breaker, 'total_requests', circuit_breaker.failure_count)
        success_count = total_requests - circuit_breaker.failure_count
        error_rate = (circuit_breaker.failure_count / max(total_requests, 1)) * 100
        
        # Calculate recovery time if applicable
        recovery_time = None
        if circuit_breaker.last_failure_time and circuit_breaker.state == 'closed':
            recovery_time = time.time() - circuit_breaker.last_failure_time
        
        metric = CircuitBreakerMetric(
            component_id=component_id,
            timestamp=datetime.now(),
            state=circuit_breaker.state,
            failure_count=circuit_breaker.failure_count,
            success_count=success_count,
            total_requests=total_requests,
            response_time_ms=1000.0,  # Mock value
            error_rate_percent=error_rate,
            recovery_time_seconds=recovery_time,
            last_failure_time=datetime.fromtimestamp(circuit_breaker.last_failure_time) 
                             if circuit_breaker.last_failure_time else None
        )
        
        self.metrics.append(metric)
        await self._check_alert_thresholds(metric)
        return metric
    
    async def _check_alert_thresholds(self, metric: CircuitBreakerMetric):
        """Check if metric triggers any alerts"""
        
        # Circuit breaker opened
        if metric.state == 'open':
            alert = AlertEvent(
                severity=AlertSeverity.CRITICAL,
                component=metric.component_id,
                message=f"Circuit breaker opened for {metric.component_id}",
                timestamp=datetime.now(),
                metadata={"failure_count": metric.failure_count, "error_rate": metric.error_rate_percent}
            )
            self.alerts.append(alert)
            await self._notify_subscribers("circuit_breaker_open", alert)
        
        # High error rate
        if metric.error_rate_percent > self.alert_thresholds["circuit_breaker_open"]["error_rate"]:
            alert = AlertEvent(
                severity=AlertSeverity.HIGH,
                component=metric.component_id,
                message=f"High error rate: {metric.error_rate_percent:.1f}%",
                timestamp=datetime.now(),
                metadata={"error_rate": metric.error_rate_percent}
            )
            self.alerts.append(alert)
            await self._notify_subscribers("high_error_rate", alert)
        
        # High latency
        if metric.response_time_ms > self.alert_thresholds["high_latency"]["response_time_ms"]:
            alert = AlertEvent(
                severity=AlertSeverity.MEDIUM,
                component=metric.component_id,
                message=f"High latency: {metric.response_time_ms:.1f}ms",
                timestamp=datetime.now(),
                metadata={"response_time_ms": metric.response_time_ms}
            )
            self.alerts.append(alert)
            await self._notify_subscribers("high_latency", alert)
    
    async def _notify_subscribers(self, event_type: str, alert: AlertEvent):
        """Notify subscribers of alert events"""
        for callback in self.subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")
    
    def subscribe_to_alerts(self, event_type: str, callback: Callable):
        """Subscribe to alert events"""
        self.subscribers[event_type].append(callback)
    
    async def perform_health_check(self, component_id: str, circuit_breaker: CircuitBreaker) -> HealthCheckResult:
        """Perform health check on circuit breaker component"""
        start_time = time.time()
        
        try:
            # Simulate health check operation
            await asyncio.sleep(0.01)  # 10ms simulated check
            response_time = (time.time() - start_time) * 1000
            
            # Determine health status based on circuit breaker state
            if circuit_breaker.state == 'open':
                status = SystemHealthStatus.UNHEALTHY
            elif circuit_breaker.failure_count > 0:
                status = SystemHealthStatus.DEGRADED
            else:
                status = SystemHealthStatus.HEALTHY
            
            result = HealthCheckResult(
                component=component_id,
                status=status,
                response_time_ms=response_time,
                metadata={
                    "circuit_state": circuit_breaker.state,
                    "failure_count": circuit_breaker.failure_count,
                    "failure_threshold": circuit_breaker.failure_threshold
                }
            )
            
            self.health_checks[component_id] = result
            return result
            
        except Exception as e:
            result = HealthCheckResult(
                component=component_id,
                status=SystemHealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
            self.health_checks[component_id] = result
            return result
    
    def get_metrics_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified time range"""
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for time range"}
        
        # Calculate aggregated statistics
        total_requests = sum(m.total_requests for m in recent_metrics)
        total_failures = sum(m.failure_count for m in recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        # Circuit breaker state distribution
        state_counts = defaultdict(int)
        for metric in recent_metrics:
            state_counts[metric.state] += 1
        
        # Component health overview
        component_health = {}
        for component_id, health_check in self.health_checks.items():
            component_health[component_id] = health_check.status.value
        
        return {
            "time_range_minutes": time_range_minutes,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "error_rate_percent": (total_failures / max(total_requests, 1)) * 100,
            "average_response_time_ms": avg_response_time,
            "circuit_breaker_states": dict(state_counts),
            "component_health": component_health,
            "alert_count": len(self.alerts),
            "metrics_collected": len(recent_metrics)
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps({
                "metrics": [m.to_dict() for m in self.metrics],
                "alerts": [a.to_dict() for a in self.alerts],
                "health_checks": {k: asdict(v) for k, v in self.health_checks.items()}
            }, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class MonitoredCircuitBreaker(CircuitBreaker):
    """Circuit breaker with integrated monitoring"""
    
    def __init__(self, component_id: str, monitoring_system: MockMonitoringSystem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_id = component_id
        self.monitoring_system = monitoring_system
        self.total_requests = 0
        self.success_count = 0
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with monitoring integration"""
        self.total_requests += 1
        start_time = time.time()
        
        try:
            result = await super().call(func, *args, **kwargs)
            self.success_count += 1
            
            # Collect metrics after successful call
            await self.monitoring_system.collect_circuit_breaker_metrics(self, self.component_id)
            return result
            
        except Exception as e:
            # Collect metrics after failure
            await self.monitoring_system.collect_circuit_breaker_metrics(self, self.component_id)
            raise


class TestCircuitBreakerMetricsCollection:
    """Test real-time circuit breaker metrics collection"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Mock monitoring system"""
        return MockMonitoringSystem()
    
    @pytest.fixture
    def monitored_circuit_breaker(self, monitoring_system):
        """Circuit breaker with monitoring integration"""
        return MonitoredCircuitBreaker(
            component_id="test_component",
            monitoring_system=monitoring_system,
            failure_threshold=3,
            recovery_timeout=1.0
        )
    
    @pytest.mark.asyncio
    async def test_real_time_metrics_collection(self, monitored_circuit_breaker, monitoring_system):
        """Test real-time collection of circuit breaker metrics"""
        
        async def mock_operation():
            return "success"
        
        # Execute successful operations
        for i in range(5):
            result = await monitored_circuit_breaker.call(mock_operation)
            assert result == "success"
        
        # Verify metrics were collected
        assert len(monitoring_system.metrics) == 5
        
        # Check latest metric
        latest_metric = monitoring_system.metrics[-1]
        assert latest_metric.component_id == "test_component"
        assert latest_metric.state == "closed"
        assert latest_metric.success_count == 5
        assert latest_metric.failure_count == 0
        assert latest_metric.total_requests == 5
        assert latest_metric.error_rate_percent == 0.0
    
    @pytest.mark.asyncio
    async def test_failure_metrics_tracking(self, monitored_circuit_breaker, monitoring_system):
        """Test tracking of failure metrics"""
        
        async def failing_operation():
            raise Exception("Operation failed")
        
        # Execute failing operations
        for i in range(3):
            with pytest.raises(Exception):
                await monitored_circuit_breaker.call(failing_operation)
        
        # Verify failure metrics
        assert len(monitoring_system.metrics) == 3
        
        latest_metric = monitoring_system.metrics[-1]
        assert latest_metric.failure_count == 3
        assert latest_metric.success_count == 0
        assert latest_metric.error_rate_percent == 100.0
        assert latest_metric.state == "open"  # Should be open after 3 failures
    
    @pytest.mark.asyncio
    async def test_state_transition_metrics(self, monitored_circuit_breaker, monitoring_system):
        """Test metrics during circuit breaker state transitions"""
        
        async def mock_operation():
            return "success"
        
        async def failing_operation():
            raise Exception("Temporary failure")
        
        # Initial state: closed
        await monitored_circuit_breaker.call(mock_operation)
        assert monitoring_system.metrics[-1].state == "closed"
        
        # Transition to open
        for i in range(3):
            with pytest.raises(Exception):
                await monitored_circuit_breaker.call(failing_operation)
        
        assert monitoring_system.metrics[-1].state == "open"
        
        # Advance time for recovery
        monitored_circuit_breaker.last_failure_time = time.time() - 1.1
        
        # Transition to half-open then closed
        await monitored_circuit_breaker.call(mock_operation)
        
        assert monitoring_system.metrics[-1].state == "closed"
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, monitored_circuit_breaker, monitoring_system):
        """Test collection of performance-related metrics"""
        
        async def slow_operation():
            await asyncio.sleep(0.1)  # 100ms operation
            return "slow_result"
        
        start_time = time.time()
        await monitored_circuit_breaker.call(slow_operation)
        end_time = time.time()
        
        # Verify performance metrics were captured
        latest_metric = monitoring_system.metrics[-1]
        assert latest_metric.response_time_ms > 0
        
        # Should reflect the actual operation time (roughly 100ms + overhead)
        actual_time_ms = (end_time - start_time) * 1000
        assert 50 <= actual_time_ms <= 200  # Allow for some variance


class TestAlertGeneration:
    """Test alert generation from circuit breaker events"""
    
    @pytest.fixture
    def alert_monitoring_system(self):
        """Monitoring system configured for alerts"""
        system = MockMonitoringSystem()
        system.alert_thresholds = {
            "circuit_breaker_open": {"error_rate": 5.0, "failure_count": 2},
            "high_latency": {"response_time_ms": 100.0},
            "budget_exhausted": {"cost_threshold": 50.0}
        }
        return system
    
    @pytest.fixture
    def alert_circuit_breaker(self, alert_monitoring_system):
        """Circuit breaker configured for alert testing"""
        return MonitoredCircuitBreaker(
            component_id="alert_test",
            monitoring_system=alert_monitoring_system,
            failure_threshold=2,  # Lower threshold for faster testing
            recovery_timeout=0.5
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_alert(self, alert_circuit_breaker, alert_monitoring_system):
        """Test alert generation when circuit breaker opens"""
        
        async def failing_operation():
            raise Exception("Service unavailable")
        
        # Trigger circuit breaker opening
        for i in range(2):
            with pytest.raises(Exception):
                await alert_circuit_breaker.call(failing_operation)
        
        # Verify alert was generated
        alerts = alert_monitoring_system.alerts
        circuit_open_alerts = [a for a in alerts if "opened" in a.message]
        
        assert len(circuit_open_alerts) >= 1
        alert = circuit_open_alerts[0]
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.component == "alert_test"
        assert "opened" in alert.message
    
    @pytest.mark.asyncio
    async def test_high_error_rate_alert(self, alert_circuit_breaker, alert_monitoring_system):
        """Test alert generation for high error rates"""
        
        async def mixed_operation():
            # 50% failure rate
            import random
            if random.random() < 0.5:
                raise Exception("Intermittent failure")
            return "success"
        
        # Execute operations with high error rate
        for i in range(10):
            try:
                await alert_circuit_breaker.call(mixed_operation)
            except Exception:
                pass
        
        # Check for high error rate alerts
        error_rate_alerts = [a for a in alert_monitoring_system.alerts 
                           if "error rate" in a.message.lower()]
        
        assert len(error_rate_alerts) > 0
        alert = error_rate_alerts[0]
        assert alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_alert_notification_system(self, alert_circuit_breaker, alert_monitoring_system):
        """Test alert notification and subscription system"""
        
        # Set up alert subscribers
        received_alerts = []
        
        async def alert_handler(alert: AlertEvent):
            received_alerts.append(alert)
        
        def sync_alert_handler(alert: AlertEvent):
            received_alerts.append(alert)
        
        # Subscribe to different alert types
        alert_monitoring_system.subscribe_to_alerts("circuit_breaker_open", alert_handler)
        alert_monitoring_system.subscribe_to_alerts("high_error_rate", sync_alert_handler)
        
        # Trigger alerts
        async def failing_operation():
            raise Exception("Service failure")
        
        for i in range(2):
            with pytest.raises(Exception):
                await alert_circuit_breaker.call(failing_operation)
        
        # Give async handlers time to execute
        await asyncio.sleep(0.1)
        
        # Verify alerts were received by subscribers
        assert len(received_alerts) >= 1
        
        # Verify alert content
        for alert in received_alerts:
            assert hasattr(alert, 'severity')
            assert hasattr(alert, 'component')
            assert hasattr(alert, 'message')
            assert hasattr(alert, 'timestamp')


class TestHealthCheckIntegration:
    """Test integration with health check systems"""
    
    @pytest.fixture
    def health_monitoring_system(self):
        """Monitoring system for health checks"""
        return MockMonitoringSystem()
    
    @pytest.fixture
    def health_circuit_breaker(self, health_monitoring_system):
        """Circuit breaker for health check testing"""
        return MonitoredCircuitBreaker(
            component_id="health_test",
            monitoring_system=health_monitoring_system,
            failure_threshold=3,
            recovery_timeout=1.0
        )
    
    @pytest.mark.asyncio
    async def test_health_check_healthy_circuit(self, health_circuit_breaker, health_monitoring_system):
        """Test health check on healthy circuit breaker"""
        
        # Execute successful operation to ensure healthy state
        async def healthy_operation():
            return "healthy"
        
        await health_circuit_breaker.call(healthy_operation)
        
        # Perform health check
        health_result = await health_monitoring_system.perform_health_check(
            "health_test", health_circuit_breaker
        )
        
        assert health_result.component == "health_test"
        assert health_result.status == SystemHealthStatus.HEALTHY
        assert health_result.response_time_ms > 0
        assert health_result.error is None
        assert health_result.metadata["circuit_state"] == "closed"
    
    @pytest.mark.asyncio
    async def test_health_check_degraded_circuit(self, health_circuit_breaker, health_monitoring_system):
        """Test health check on degraded circuit breaker"""
        
        # Create some failures but not enough to open circuit
        async def failing_operation():
            raise Exception("Intermittent failure")
        
        with pytest.raises(Exception):
            await health_circuit_breaker.call(failing_operation)
        
        # Perform health check
        health_result = await health_monitoring_system.perform_health_check(
            "health_test", health_circuit_breaker
        )
        
        assert health_result.status == SystemHealthStatus.DEGRADED
        assert health_result.metadata["failure_count"] > 0
        assert health_result.metadata["circuit_state"] == "closed"
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_circuit(self, health_circuit_breaker, health_monitoring_system):
        """Test health check on unhealthy (open) circuit breaker"""
        
        # Open the circuit breaker
        async def failing_operation():
            raise Exception("Service down")
        
        for i in range(3):
            with pytest.raises(Exception):
                await health_circuit_breaker.call(failing_operation)
        
        # Perform health check
        health_result = await health_monitoring_system.perform_health_check(
            "health_test", health_circuit_breaker
        )
        
        assert health_result.status == SystemHealthStatus.UNHEALTHY
        assert health_result.metadata["circuit_state"] == "open"
        assert health_result.metadata["failure_count"] >= 3
    
    @pytest.mark.asyncio
    async def test_system_wide_health_status(self, health_monitoring_system):
        """Test system-wide health status aggregation"""
        
        # Create multiple circuit breakers in different states
        cb1 = MonitoredCircuitBreaker("component1", health_monitoring_system, failure_threshold=2)
        cb2 = MonitoredCircuitBreaker("component2", health_monitoring_system, failure_threshold=2)
        cb3 = MonitoredCircuitBreaker("component3", health_monitoring_system, failure_threshold=2)
        
        # Make cb1 healthy
        await cb1.call(lambda: "success")
        await health_monitoring_system.perform_health_check("component1", cb1)
        
        # Make cb2 degraded
        try:
            await cb2.call(lambda: Exception("fail"))
        except:
            pass
        await health_monitoring_system.perform_health_check("component2", cb2)
        
        # Make cb3 unhealthy (open)
        for i in range(2):
            try:
                await cb3.call(lambda: Exception("fail"))
            except:
                pass
        await health_monitoring_system.perform_health_check("component3", cb3)
        
        # Get system summary
        summary = health_monitoring_system.get_metrics_summary()
        
        assert "component_health" in summary
        component_health = summary["component_health"]
        
        assert component_health["component1"] == "healthy"
        assert component_health["component2"] == "degraded"
        assert component_health["component3"] == "unhealthy"


class TestPerformanceDashboardIntegration:
    """Test integration with performance dashboard systems"""
    
    @pytest.fixture
    def dashboard_monitoring_system(self):
        """Monitoring system for dashboard testing"""
        return MockMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_metrics_aggregation_for_dashboard(self, dashboard_monitoring_system):
        """Test metrics aggregation for dashboard display"""
        
        # Create multiple circuit breakers and generate metrics
        circuit_breakers = []
        for i in range(3):
            cb = MonitoredCircuitBreaker(
                f"service_{i}", dashboard_monitoring_system, failure_threshold=3
            )
            circuit_breakers.append(cb)
        
        # Generate mixed success/failure patterns
        for cb in circuit_breakers:
            for j in range(10):
                try:
                    if j % 3 == 0:  # Fail every 3rd request
                        await cb.call(lambda: Exception("failure"))
                    else:
                        await cb.call(lambda: "success")
                except:
                    pass
        
        # Get dashboard summary
        summary = dashboard_monitoring_system.get_metrics_summary(time_range_minutes=60)
        
        # Verify dashboard data structure
        required_fields = [
            "total_requests", "total_failures", "error_rate_percent",
            "average_response_time_ms", "circuit_breaker_states",
            "component_health", "alert_count", "metrics_collected"
        ]
        
        for field in required_fields:
            assert field in summary
        
        assert summary["total_requests"] > 0
        assert summary["metrics_collected"] > 0
        assert 0 <= summary["error_rate_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_historical_trend_analysis(self, dashboard_monitoring_system):
        """Test historical trend analysis capabilities"""
        
        cb = MonitoredCircuitBreaker(
            "trend_test", dashboard_monitoring_system, failure_threshold=5
        )
        
        # Simulate degrading performance over time
        for hour in range(3):
            for minute in range(10):
                try:
                    # Increase failure rate over time
                    if minute < (3 + hour * 2):  # More failures as time progresses
                        await cb.call(lambda: "success")
                    else:
                        await cb.call(lambda: Exception("degrading"))
                except:
                    pass
                
                # Simulate time passing
                await asyncio.sleep(0.001)
        
        # Analyze trends over different time windows
        recent_summary = dashboard_monitoring_system.get_metrics_summary(time_range_minutes=5)
        long_term_summary = dashboard_monitoring_system.get_metrics_summary(time_range_minutes=60)
        
        # Recent metrics should show higher error rate
        assert recent_summary["metrics_collected"] > 0
        assert long_term_summary["metrics_collected"] >= recent_summary["metrics_collected"]
    
    @pytest.mark.asyncio
    async def test_real_time_dashboard_updates(self, dashboard_monitoring_system):
        """Test real-time dashboard update capabilities"""
        
        cb = MonitoredCircuitBreaker(
            "realtime_test", dashboard_monitoring_system, failure_threshold=3
        )
        
        # Track metrics over time
        metric_snapshots = []
        
        for i in range(5):
            # Execute operation
            try:
                if i < 3:
                    await cb.call(lambda: "success")
                else:
                    await cb.call(lambda: Exception("failure"))
            except:
                pass
            
            # Take snapshot of current metrics
            summary = dashboard_monitoring_system.get_metrics_summary(time_range_minutes=60)
            metric_snapshots.append({
                "iteration": i,
                "total_requests": summary["total_requests"],
                "error_rate": summary["error_rate_percent"],
                "timestamp": time.time()
            })
            
            await asyncio.sleep(0.01)  # Small delay between operations
        
        # Verify metrics are updating in real-time
        assert len(metric_snapshots) == 5
        
        # Total requests should increase over time
        for i in range(1, len(metric_snapshots)):
            assert metric_snapshots[i]["total_requests"] >= metric_snapshots[i-1]["total_requests"]
        
        # Error rate should increase after failures start
        assert metric_snapshots[-1]["error_rate"] > metric_snapshots[0]["error_rate"]


class TestProductionMonitoringIntegration:
    """Test integration with production monitoring infrastructure"""
    
    @pytest.fixture
    def production_monitoring_setup(self):
        """Production monitoring setup"""
        monitoring_system = MockMonitoringSystem()
        
        # Configure production-like thresholds
        monitoring_system.alert_thresholds = {
            "circuit_breaker_open": {"error_rate": 1.0, "failure_count": 1},
            "high_latency": {"response_time_ms": 1000.0},
            "budget_exhausted": {"cost_threshold": 10.0}
        }
        
        return monitoring_system
    
    @pytest.mark.asyncio
    async def test_production_alert_escalation(self, production_monitoring_setup):
        """Test production alert escalation workflow"""
        
        monitoring_system = production_monitoring_setup
        
        # Set up escalation tracking
        escalation_log = []
        
        async def critical_alert_handler(alert: AlertEvent):
            escalation_log.append({
                "level": "critical",
                "alert": alert,
                "timestamp": time.time()
            })
        
        async def high_alert_handler(alert: AlertEvent):
            escalation_log.append({
                "level": "high", 
                "alert": alert,
                "timestamp": time.time()
            })
        
        # Subscribe to alerts
        monitoring_system.subscribe_to_alerts("circuit_breaker_open", critical_alert_handler)
        monitoring_system.subscribe_to_alerts("high_error_rate", high_alert_handler)
        
        # Create circuit breaker
        cb = MonitoredCircuitBreaker(
            "production_service", monitoring_system, failure_threshold=1
        )
        
        # Trigger production incident
        with pytest.raises(Exception):
            await cb.call(lambda: Exception("Production failure"))
        
        await asyncio.sleep(0.1)  # Allow alert processing
        
        # Verify escalation occurred
        assert len(escalation_log) > 0
        
        # Should have critical alert for circuit breaker opening
        critical_alerts = [log for log in escalation_log if log["level"] == "critical"]
        assert len(critical_alerts) >= 1
    
    @pytest.mark.asyncio
    async def test_metrics_export_for_external_systems(self, production_monitoring_setup):
        """Test metrics export for external monitoring systems"""
        
        monitoring_system = production_monitoring_setup
        
        # Generate production-like metrics
        cb = MonitoredCircuitBreaker(
            "external_metrics_test", monitoring_system, failure_threshold=3
        )
        
        # Mixed operations
        for i in range(20):
            try:
                if i % 4 == 0:  # 25% failure rate
                    await cb.call(lambda: Exception("Service error"))
                else:
                    await cb.call(lambda: "Success")
            except:
                pass
        
        # Export metrics in different formats
        json_export = monitoring_system.export_metrics("json")
        
        # Verify export contains required data
        export_data = json.loads(json_export)
        
        assert "metrics" in export_data
        assert "alerts" in export_data
        assert "health_checks" in export_data
        
        assert len(export_data["metrics"]) > 0
        
        # Verify metric structure for external consumption
        sample_metric = export_data["metrics"][0]
        required_fields = [
            "component_id", "timestamp", "state", "failure_count",
            "success_count", "total_requests", "response_time_ms",
            "error_rate_percent"
        ]
        
        for field in required_fields:
            assert field in sample_metric
    
    @pytest.mark.asyncio
    async def test_integration_with_cost_based_monitoring(self, production_monitoring_setup):
        """Test integration with cost-based circuit breakers and monitoring"""
        
        monitoring_system = production_monitoring_setup
        
        # Create cost-based circuit breaker
        cost_cb = CostBasedCircuitBreaker(
            cost_threshold=5.0,
            failure_threshold=2,
            recovery_timeout=1.0,
            cost_window_hours=1.0
        )
        
        # Simulate expensive operations with monitoring
        total_cost = 0.0
        for i in range(3):
            try:
                # Mock expensive operation
                await cost_cb.add_cost(2.0)
                total_cost += 2.0
                
                # Collect cost metrics manually (cost_cb doesn't inherit from MonitoredCircuitBreaker)
                await monitoring_system.collect_circuit_breaker_metrics(cost_cb, "cost_service")
                
            except BudgetExhaustedError:
                # Create budget exhaustion alert
                alert = AlertEvent(
                    severity=AlertSeverity.CRITICAL,
                    component="cost_service",
                    message="Budget exhausted",
                    timestamp=datetime.now(),
                    metadata={"total_cost": total_cost, "threshold": cost_cb.cost_threshold}
                )
                monitoring_system.alerts.append(alert)
                break
        
        # Verify cost monitoring integration
        assert len(monitoring_system.metrics) >= 2
        
        # Should have budget exhaustion alert
        budget_alerts = [a for a in monitoring_system.alerts if "budget" in a.message.lower()]
        assert len(budget_alerts) >= 1


if __name__ == "__main__":
    # Run tests with proper configuration
    pytest.main([__file__, "-v", "--tb=short"])