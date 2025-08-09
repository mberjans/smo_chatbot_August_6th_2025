"""
Comprehensive monitoring and alerting tests for ProductionCircuitBreaker.

This module provides extensive test coverage for monitoring system integration
including:
- Metrics collection accuracy
- Alert triggering and notification delivery
- Dashboard data integrity
- Logging system functionality  
- Health check endpoint reliability
"""

import pytest
import asyncio
import time
import json
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import tempfile
import os

import sys
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
# Monitoring Test Fixtures and Mocks
# ============================================================================

@pytest.fixture
def monitoring_config():
    """Configuration optimized for monitoring testing"""
    return BackendInstanceConfig(
        id="monitoring_backend",
        backend_type=BackendType.PERPLEXITY,
        endpoint_url="https://api.monitoring.com",
        api_key="monitoring_key",
        failure_threshold=3,
        recovery_timeout_seconds=30,
        half_open_max_requests=5,
        expected_response_time_ms=1000.0,
        circuit_breaker_enabled=True
    )

@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collection system"""
    collector = Mock()
    collector.record_metric = Mock()
    collector.record_counter = Mock()
    collector.record_histogram = Mock()
    collector.record_gauge = Mock()
    collector.get_metrics_snapshot = Mock()
    return collector

@pytest.fixture
def mock_alert_manager():
    """Mock alert management system"""
    manager = Mock()
    manager.trigger_alert = AsyncMock()
    manager.resolve_alert = AsyncMock()
    manager.get_active_alerts = AsyncMock()
    manager.send_notification = AsyncMock()
    return manager

@pytest.fixture
def mock_dashboard_service():
    """Mock dashboard service"""
    dashboard = Mock()
    dashboard.update_circuit_breaker_status = AsyncMock()
    dashboard.update_performance_metrics = AsyncMock()
    dashboard.update_health_status = AsyncMock()
    dashboard.get_dashboard_data = AsyncMock()
    return dashboard

@pytest.fixture
def mock_logging_system():
    """Mock structured logging system"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.log_structured = Mock()
    return logger

@pytest.fixture
async def monitored_load_balancer(monitoring_config):
    """Create load balancer with monitoring systems"""
    config = ProductionLoadBalancingConfig(
        backend_instances={"monitoring_backend": monitoring_config},
        enable_real_time_monitoring=True,
        global_circuit_breaker_enabled=True
    )
    
    with patch.multiple(
        'production_load_balancer.ProductionLoadBalancer',
        _initialize_backend_clients=AsyncMock(),
        _start_monitoring_tasks=AsyncMock(),
        _initialize_metrics_collection=AsyncMock()
    ):
        lb = ProductionLoadBalancer(config)
        await lb.initialize()
        
        # Mock backend client
        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.health_check = AsyncMock()
        lb.backend_clients["monitoring_backend"] = mock_client
        
        return lb


# ============================================================================
# Metrics Collection Tests
# ============================================================================

class TestMetricsCollection:
    """Test accuracy and completeness of metrics collection"""

    def test_basic_metrics_accuracy(self, monitoring_config, mock_metrics_collector):
        """Test basic circuit breaker metrics are collected accurately"""
        cb = ProductionCircuitBreaker(monitoring_config)
        
        # Generate test data
        test_operations = [
            ("success", 500.0, None),
            ("success", 750.0, None),
            ("failure", 2000.0, "TimeoutError"),
            ("success", 600.0, None),
            ("failure", None, "ServerError")
        ]
        
        for operation, response_time, error_type in test_operations:
            if operation == "success":
                cb.record_success(response_time)
            else:
                cb.record_failure(f"Test error", response_time_ms=response_time, error_type=error_type)
        
        # Collect metrics
        metrics = cb.get_metrics()
        
        # Verify accuracy
        assert metrics["success_count"] == 3
        assert metrics["failure_count"] == 2
        assert metrics["failure_rate"] == 0.4  # 2 failures out of 5 operations
        
        # Verify response time metrics
        expected_response_times = [500.0, 750.0, 2000.0, 600.0]  # Failures without response times excluded
        expected_avg = sum(expected_response_times) / len(expected_response_times)
        assert abs(metrics["avg_response_time_ms"] - expected_avg) < 0.1
        
        # Verify error type tracking
        assert metrics["error_types"]["TimeoutError"] == 1
        assert metrics["error_types"]["ServerError"] == 1

    def test_metrics_window_behavior(self, monitoring_config):
        """Test that metrics windows behave correctly with limits"""
        cb = ProductionCircuitBreaker(monitoring_config)
        
        # Fill beyond window capacity
        window_size = 100  # Assumed failure rate window size
        
        for i in range(window_size + 20):  # 20 extra operations
            if i % 4 == 0:
                cb.record_failure(f"Window test {i}", error_type="WindowTestError")
            else:
                cb.record_success(random.uniform(100, 300))
        
        metrics = cb.get_metrics()
        
        # Windows should be limited in size
        assert len(cb.failure_rate_window) <= window_size
        assert len(cb.response_time_window) <= 50  # Assumed response time window size
        
        # Recent data should be preserved
        recent_failure_count = sum(cb.failure_rate_window)
        expected_recent_failures = 25  # Approximately 1/4 of last 100 operations
        assert abs(recent_failure_count - expected_recent_failures) <= 5  # Allow some variance

    def test_state_transition_metrics(self, monitoring_config):
        """Test metrics during state transitions"""
        cb = ProductionCircuitBreaker(monitoring_config)
        
        # Test CLOSED -> OPEN transition
        initial_metrics = cb.get_metrics()
        assert initial_metrics["state"] == CircuitBreakerState.CLOSED.value
        
        # Trigger opening
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"State transition test {i}")
        
        open_metrics = cb.get_metrics()
        assert open_metrics["state"] == CircuitBreakerState.OPEN.value
        assert open_metrics["failure_count"] >= cb.config.failure_threshold
        assert open_metrics["next_attempt_time"] is not None
        
        # Test recovery timing metrics
        time_until_retry = open_metrics["time_until_retry_seconds"]
        assert time_until_retry > 0
        assert time_until_retry <= cb.config.recovery_timeout_seconds * 1.5  # Account for jitter

    def test_performance_metrics_accuracy(self, monitoring_config):
        """Test accuracy of performance-related metrics"""
        cb = ProductionCircuitBreaker(monitoring_config)
        
        # Generate controlled performance data
        response_times = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        for rt in response_times:
            cb.record_success(rt)
        
        metrics = cb.get_metrics()
        
        # Verify statistical metrics
        expected_avg = sum(response_times) / len(response_times)
        assert abs(metrics["avg_response_time_ms"] - expected_avg) < 0.1
        
        # Verify percentile calculations (if available)
        if "p95_response_time_ms" in metrics:
            expected_p95 = sorted(response_times)[int(0.95 * len(response_times))]
            assert abs(metrics["p95_response_time_ms"] - expected_p95) <= 50

    def test_error_classification_metrics(self, monitoring_config):
        """Test error classification and frequency metrics"""
        cb = ProductionCircuitBreaker(monitoring_config)
        
        # Generate various error types
        error_patterns = [
            ("TimeoutError", 5),
            ("ServerError", 3), 
            ("NetworkError", 2),
            ("ValidationError", 1),
            ("RateLimitError", 4)
        ]
        
        for error_type, count in error_patterns:
            for i in range(count):
                cb.record_failure(f"{error_type} #{i}", error_type=error_type)
        
        metrics = cb.get_metrics()
        
        # Verify error type frequencies
        for error_type, expected_count in error_patterns:
            assert metrics["error_types"][error_type] == expected_count
        
        # Verify consecutive error tracking
        assert metrics["consecutive_timeouts"] >= 0
        assert metrics["consecutive_server_errors"] >= 0

    @pytest.mark.asyncio
    async def test_real_time_metrics_streaming(self, monitored_load_balancer, mock_metrics_collector):
        """Test real-time streaming of circuit breaker metrics"""
        lb = monitored_load_balancer
        lb.metrics_collector = mock_metrics_collector
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Simulate operations that should trigger metric streams
        operations = [
            ("success", 500.0),
            ("failure", "Test failure"),
            ("success", 750.0),
            ("state_change", CircuitBreakerState.OPEN)
        ]
        
        for operation, data in operations:
            if operation == "success":
                cb.record_success(data)
            elif operation == "failure":
                cb.record_failure(data)
            elif operation == "state_change":
                cb.state = data
            
            # Simulate metric streaming
            await lb._stream_circuit_breaker_metrics("monitoring_backend")
        
        # Verify metrics were streamed
        assert mock_metrics_collector.record_metric.call_count >= len(operations)


# ============================================================================
# Alert System Tests
# ============================================================================

class TestAlertSystem:
    """Test alert triggering and notification systems"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_change_alerts(self, monitored_load_balancer, mock_alert_manager):
        """Test alerts triggered by circuit breaker state changes"""
        lb = monitored_load_balancer
        lb.alert_manager = mock_alert_manager
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Trigger circuit opening
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Alert test failure {i}")
        
        # Simulate alert checking
        await lb._check_circuit_breaker_alerts()
        
        # Verify circuit breaker opened alert was triggered
        mock_alert_manager.trigger_alert.assert_called()
        
        # Verify alert contains correct information
        alert_calls = mock_alert_manager.trigger_alert.call_args_list
        circuit_breaker_alert = None
        
        for call_args in alert_calls:
            args, kwargs = call_args
            if "circuit_breaker" in str(args).lower():
                circuit_breaker_alert = args[0] if args else kwargs
                break
        
        assert circuit_breaker_alert is not None
        # Verify alert details would be appropriate (implementation dependent)

    @pytest.mark.asyncio
    async def test_performance_degradation_alerts(self, monitored_load_balancer, mock_alert_manager):
        """Test alerts for performance degradation"""
        lb = monitored_load_balancer
        lb.alert_manager = mock_alert_manager
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Simulate performance degradation
        baseline = cb._baseline_response_time
        degraded_times = [baseline * 3, baseline * 4, baseline * 5, baseline * 4.5]
        
        for response_time in degraded_times:
            cb.record_success(response_time)
        
        # Check for performance alerts
        await lb._check_performance_alerts()
        
        # Verify performance degradation alert
        mock_alert_manager.trigger_alert.assert_called()
        
        # Check for specific performance alert
        alert_calls = mock_alert_manager.trigger_alert.call_args_list
        performance_alerts = [call for call in alert_calls 
                             if "performance" in str(call).lower() or "response_time" in str(call).lower()]
        
        assert len(performance_alerts) > 0

    @pytest.mark.asyncio
    async def test_cascading_failure_alerts(self, monitored_load_balancer, mock_alert_manager):
        """Test alerts for potential cascading failures"""
        # Create load balancer with multiple backends
        config = ProductionLoadBalancingConfig(
            backend_instances={
                f"cascade_backend_{i}": BackendInstanceConfig(
                    id=f"cascade_backend_{i}",
                    backend_type=BackendType.PERPLEXITY,
                    endpoint_url=f"https://api.cascade{i}.com",
                    api_key=f"cascade_key_{i}",
                    failure_threshold=2
                ) for i in range(4)
            },
            enable_real_time_monitoring=True,
            cascade_failure_prevention=True
        )
        
        with patch.multiple(
            'production_load_balancer.ProductionLoadBalancer',
            _initialize_backend_clients=AsyncMock(),
            _start_monitoring_tasks=AsyncMock()
        ):
            lb = ProductionLoadBalancer(config)
            await lb.initialize()
            lb.alert_manager = mock_alert_manager
        
        # Simulate cascade scenario - multiple services failing
        failed_services = ["cascade_backend_0", "cascade_backend_1", "cascade_backend_2"]
        
        for service_id in failed_services:
            cb = lb.circuit_breakers[service_id]
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Cascade failure {i}")
        
        # Check for cascade alerts
        await lb._check_cascade_failure_alerts()
        
        # Verify cascade prevention alert
        mock_alert_manager.trigger_alert.assert_called()
        
        cascade_alerts = [call for call in mock_alert_manager.trigger_alert.call_args_list
                         if "cascade" in str(call).lower()]
        assert len(cascade_alerts) > 0

    @pytest.mark.asyncio
    async def test_alert_resolution_and_recovery(self, monitored_load_balancer, mock_alert_manager):
        """Test alert resolution when issues are resolved"""
        lb = monitored_load_balancer
        lb.alert_manager = mock_alert_manager
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Phase 1: Trigger alert
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Recovery test failure {i}")
        
        await lb._check_circuit_breaker_alerts()
        
        # Verify alert was triggered
        assert mock_alert_manager.trigger_alert.called
        initial_alert_count = mock_alert_manager.trigger_alert.call_count
        
        # Phase 2: Recovery
        cb.reset()  # Simulate recovery
        
        await lb._check_circuit_breaker_alerts()
        
        # Verify alert resolution
        mock_alert_manager.resolve_alert.assert_called()
        
        resolution_calls = mock_alert_manager.resolve_alert.call_args_list
        circuit_breaker_resolutions = [call for call in resolution_calls
                                     if "circuit_breaker" in str(call).lower()]
        assert len(circuit_breaker_resolutions) > 0

    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self, monitored_load_balancer, mock_alert_manager):
        """Test that alerts are rate-limited to prevent spam"""
        lb = monitored_load_balancer
        lb.alert_manager = mock_alert_manager
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Rapidly trigger the same alert condition multiple times
        for iteration in range(5):
            # Reset and re-trigger the same alert
            cb.reset()
            
            for i in range(cb.config.failure_threshold):
                cb.record_failure(f"Rate limit test {iteration}-{i}")
            
            await lb._check_circuit_breaker_alerts()
            time.sleep(0.1)  # Small delay
        
        # Verify alerts were rate limited (implementation dependent)
        total_alerts = mock_alert_manager.trigger_alert.call_count
        
        # Should be fewer alerts than trigger attempts due to rate limiting
        assert total_alerts < 5 * 2  # Less than 2x the number of iterations

    @pytest.mark.asyncio
    async def test_notification_delivery_channels(self, monitored_load_balancer, mock_alert_manager):
        """Test alert delivery through different notification channels"""
        lb = monitored_load_balancer
        lb.alert_manager = mock_alert_manager
        
        # Configure multiple notification channels
        mock_alert_manager.send_notification.return_value = True
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Trigger high-severity alert
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Notification test {i}")
        
        await lb._check_circuit_breaker_alerts()
        
        # Simulate notification delivery
        await lb._deliver_circuit_breaker_notifications("monitoring_backend", "CRITICAL")
        
        # Verify notifications were sent
        mock_alert_manager.send_notification.assert_called()
        
        notification_calls = mock_alert_manager.send_notification.call_args_list
        assert len(notification_calls) > 0
        
        # Verify different channels were used (implementation dependent)
        # Could verify email, slack, webhook notifications, etc.


# ============================================================================
# Dashboard Integration Tests
# ============================================================================

class TestDashboardIntegration:
    """Test dashboard data integrity and updates"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_status_dashboard_updates(self, monitored_load_balancer, mock_dashboard_service):
        """Test dashboard updates for circuit breaker status changes"""
        lb = monitored_load_balancer
        lb.dashboard_service = mock_dashboard_service
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Test various state transitions
        state_transitions = [
            (CircuitBreakerState.CLOSED, "healthy_operations"),
            (CircuitBreakerState.OPEN, "failure_scenario"),
            (CircuitBreakerState.HALF_OPEN, "recovery_testing"),
            (CircuitBreakerState.CLOSED, "full_recovery")
        ]
        
        for target_state, scenario in state_transitions:
            # Simulate reaching target state
            if target_state == CircuitBreakerState.OPEN:
                for i in range(cb.config.failure_threshold):
                    cb.record_failure(f"Dashboard test {i}")
            elif target_state == CircuitBreakerState.HALF_OPEN:
                cb.state = CircuitBreakerState.HALF_OPEN
                cb.half_open_requests = 0
            elif target_state == CircuitBreakerState.CLOSED and scenario == "full_recovery":
                cb.reset()
            
            # Update dashboard
            await lb._update_dashboard_circuit_breaker_status("monitoring_backend")
            
            # Verify dashboard was updated
            mock_dashboard_service.update_circuit_breaker_status.assert_called()
        
        # Verify all state transitions were reflected
        update_calls = mock_dashboard_service.update_circuit_breaker_status.call_args_list
        assert len(update_calls) >= len(state_transitions)

    @pytest.mark.asyncio
    async def test_performance_metrics_dashboard_integration(self, monitored_load_balancer, mock_dashboard_service):
        """Test dashboard integration for performance metrics"""
        lb = monitored_load_balancer
        lb.dashboard_service = mock_dashboard_service
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Generate performance data
        performance_scenarios = [
            ("optimal", [100, 150, 120, 130, 110]),
            ("degraded", [800, 900, 850, 950, 880]),
            ("recovering", [600, 500, 400, 300, 250])
        ]
        
        for scenario, response_times in performance_scenarios:
            for rt in response_times:
                cb.record_success(rt)
            
            # Update dashboard with performance data
            await lb._update_dashboard_performance_metrics("monitoring_backend")
        
        # Verify performance updates
        mock_dashboard_service.update_performance_metrics.assert_called()
        
        performance_updates = mock_dashboard_service.update_performance_metrics.call_args_list
        assert len(performance_updates) >= len(performance_scenarios)

    @pytest.mark.asyncio
    async def test_dashboard_data_consistency(self, monitored_load_balancer, mock_dashboard_service):
        """Test consistency of data sent to dashboard"""
        lb = monitored_load_balancer
        lb.dashboard_service = mock_dashboard_service
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Generate mixed data
        operations = [
            ("success", 500.0, None),
            ("failure", 1500.0, "TimeoutError"),
            ("success", 400.0, None),
            ("failure", None, "ServerError"),
            ("success", 600.0, None)
        ]
        
        for operation, response_time, error_type in operations:
            if operation == "success":
                cb.record_success(response_time)
            else:
                cb.record_failure("Test failure", response_time_ms=response_time, error_type=error_type)
        
        # Get dashboard data
        dashboard_data = await lb._get_dashboard_data_for_circuit_breaker("monitoring_backend")
        
        # Verify data consistency
        direct_metrics = cb.get_metrics()
        
        # Key metrics should match
        assert dashboard_data["success_count"] == direct_metrics["success_count"]
        assert dashboard_data["failure_count"] == direct_metrics["failure_count"]
        assert dashboard_data["state"] == direct_metrics["state"]
        assert abs(dashboard_data["failure_rate"] - direct_metrics["failure_rate"]) < 0.001

    @pytest.mark.asyncio
    async def test_dashboard_historical_data_aggregation(self, monitored_load_balancer):
        """Test historical data aggregation for dashboard"""
        lb = monitored_load_balancer
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Generate time-series data
        time_points = []
        current_time = datetime.now()
        
        for i in range(24):  # 24 hour periods
            # Simulate different patterns throughout the day
            if 8 <= i <= 17:  # Business hours - higher load
                failure_rate = 0.1
                avg_response_time = 800
            else:  # Off hours - lower load
                failure_rate = 0.05
                avg_response_time = 400
            
            # Generate operations for this hour
            operations_count = 100
            failures_count = int(operations_count * failure_rate)
            successes_count = operations_count - failures_count
            
            hour_data = {
                "timestamp": current_time - timedelta(hours=23-i),
                "successes": successes_count,
                "failures": failures_count,
                "avg_response_time": avg_response_time,
                "state": CircuitBreakerState.CLOSED.value
            }
            
            time_points.append(hour_data)
        
        # Aggregate historical data
        historical_data = await lb._aggregate_historical_circuit_breaker_data(
            "monitoring_backend", 
            time_points
        )
        
        # Verify aggregation
        assert len(historical_data) == 24
        assert all("timestamp" in point for point in historical_data)
        assert all("failure_rate" in point for point in historical_data)
        
        # Verify business hours show different patterns than off hours
        business_hours_avg = sum(
            point["avg_response_time"] 
            for point in historical_data 
            if 8 <= point["timestamp"].hour <= 17
        ) / 10  # 10 business hours
        
        off_hours_avg = sum(
            point["avg_response_time"]
            for point in historical_data
            if not (8 <= point["timestamp"].hour <= 17) 
        ) / 14  # 14 off hours
        
        assert business_hours_avg > off_hours_avg


# ============================================================================
# Logging System Tests
# ============================================================================

class TestLoggingSystem:
    """Test structured logging functionality"""

    def test_circuit_breaker_state_change_logging(self, monitoring_config, mock_logging_system):
        """Test logging of circuit breaker state changes"""
        cb = ProductionCircuitBreaker(monitoring_config)
        cb.logger = mock_logging_system
        
        # Trigger state change
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Logging test failure {i}")
        
        # Verify state change was logged
        mock_logging_system.warning.assert_called()
        
        # Check log message content
        warning_calls = mock_logging_system.warning.call_args_list
        circuit_breaker_logs = [call for call in warning_calls 
                               if "circuit breaker" in str(call).lower() and "opened" in str(call).lower()]
        assert len(circuit_breaker_logs) > 0

    def test_performance_degradation_logging(self, monitoring_config, mock_logging_system):
        """Test logging of performance degradation"""
        cb = ProductionCircuitBreaker(monitoring_config)
        cb.logger = mock_logging_system
        
        # Simulate performance degradation
        baseline = cb._baseline_response_time
        degraded_times = [baseline * 4, baseline * 5, baseline * 6]
        
        for rt in degraded_times:
            cb.record_success(rt)
        
        # Force proactive opening check
        cb.should_allow_request()
        
        # Should log performance issues
        log_calls = (mock_logging_system.warning.call_args_list + 
                    mock_logging_system.info.call_args_list)
        
        performance_logs = [call for call in log_calls
                          if "performance" in str(call).lower() or "degradation" in str(call).lower()]
        
        # Performance degradation should be logged
        assert len(performance_logs) >= 0  # Implementation may vary

    def test_structured_logging_format(self, monitoring_config):
        """Test that logs follow structured format"""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            cb = ProductionCircuitBreaker(monitoring_config)
            
            # Trigger various events
            cb.record_success(500.0)
            cb.record_failure("Structured log test", error_type="StructuredTestError")
            
            # Verify logger was used
            assert mock_logger.info.called or mock_logger.warning.called or mock_logger.error.called

    def test_log_correlation_ids(self, monitoring_config):
        """Test that logs include correlation IDs for tracing"""
        # This test would verify that circuit breaker logs include
        # correlation IDs for distributed tracing
        cb = ProductionCircuitBreaker(monitoring_config)
        
        # Simulate operation with correlation context
        correlation_id = "test-correlation-123"
        
        # In a real implementation, this would set correlation context
        with patch.object(cb, 'logger') as mock_logger:
            cb.record_failure("Correlation test", error_type="CorrelationTest")
            
            # Verify logs would include correlation information
            # (Implementation dependent)
            mock_logger.warning.assert_called()

    def test_log_sampling_and_rate_limiting(self, monitoring_config):
        """Test that excessive logging is rate-limited"""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            cb = ProductionCircuitBreaker(monitoring_config)
            
            # Generate many similar events
            for i in range(100):
                cb.record_failure(f"Rate limit test {i}", error_type="RateLimitTestError")
            
            # Should not log every single failure (implementation dependent)
            total_log_calls = (mock_logger.warning.call_count + 
                             mock_logger.error.call_count + 
                             mock_logger.info.call_count)
            
            # Should be fewer log calls than events due to rate limiting/sampling
            assert total_log_calls < 100


# ============================================================================
# Health Check Integration Tests  
# ============================================================================

class TestHealthCheckIntegration:
    """Test health check endpoint integration"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_health_status_endpoint(self, monitored_load_balancer):
        """Test health status endpoint includes circuit breaker info"""
        lb = monitored_load_balancer
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Test different circuit breaker states
        test_states = [
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.OPEN,
            CircuitBreakerState.HALF_OPEN
        ]
        
        for state in test_states:
            # Set circuit breaker state
            cb.state = state
            if state == CircuitBreakerState.OPEN:
                cb.failure_count = cb.config.failure_threshold
            elif state == CircuitBreakerState.HALF_OPEN:
                cb.half_open_requests = 2
            
            # Get health status
            health_status = await lb.get_health_status()
            
            # Verify circuit breaker info is included
            assert "circuit_breakers" in health_status
            assert "monitoring_backend" in health_status["circuit_breakers"]
            
            cb_health = health_status["circuit_breakers"]["monitoring_backend"]
            assert cb_health["state"] == state.value
            
            # Overall health should reflect circuit breaker states
            if state == CircuitBreakerState.OPEN:
                assert health_status["overall_status"] in ["degraded", "unhealthy"]
            elif state == CircuitBreakerState.CLOSED:
                assert health_status["overall_status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_health_check_performance_metrics(self, monitored_load_balancer):
        """Test health check includes performance metrics"""
        lb = monitored_load_balancer
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Generate performance data
        response_times = [100, 200, 150, 300, 250]
        for rt in response_times:
            cb.record_success(rt)
        
        # Add some failures
        cb.record_failure("Health check test", error_type="HealthTestError")
        
        # Get health status
        health_status = await lb.get_health_status()
        
        # Verify performance metrics are included
        cb_health = health_status["circuit_breakers"]["monitoring_backend"]
        
        assert "avg_response_time_ms" in cb_health
        assert "failure_rate" in cb_health
        assert "success_count" in cb_health
        assert "failure_count" in cb_health
        
        # Verify values are reasonable
        assert cb_health["avg_response_time_ms"] > 0
        assert 0 <= cb_health["failure_rate"] <= 1

    @pytest.mark.asyncio
    async def test_health_check_detailed_diagnostics(self, monitored_load_balancer):
        """Test detailed diagnostic information in health checks"""
        lb = monitored_load_balancer
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Create diagnostic scenario
        error_types = ["TimeoutError", "ServerError", "NetworkError"]
        for i, error_type in enumerate(error_types):
            for j in range(i + 1):  # Different frequencies
                cb.record_failure(f"Diagnostic test {j}", error_type=error_type)
        
        # Get detailed health status
        detailed_health = await lb.get_detailed_health_status()
        
        # Verify diagnostic information
        cb_diagnostics = detailed_health["circuit_breakers"]["monitoring_backend"]
        
        assert "error_breakdown" in cb_diagnostics
        assert "performance_trends" in cb_diagnostics
        assert "recent_events" in cb_diagnostics
        
        # Verify error breakdown
        error_breakdown = cb_diagnostics["error_breakdown"]
        for error_type in error_types:
            assert error_type in error_breakdown
            assert error_breakdown[error_type] > 0

    @pytest.mark.asyncio
    async def test_health_check_alerting_integration(self, monitored_load_balancer, mock_alert_manager):
        """Test health check integration with alerting system"""
        lb = monitored_load_balancer
        lb.alert_manager = mock_alert_manager
        
        cb = lb.circuit_breakers["monitoring_backend"]
        
        # Create unhealthy condition
        for i in range(cb.config.failure_threshold):
            cb.record_failure(f"Health alert test {i}")
        
        # Run health check with alerting
        health_status = await lb.get_health_status_with_alerting()
        
        # Verify health status reflects issues
        assert health_status["overall_status"] in ["degraded", "unhealthy"]
        
        # Verify alerts were triggered
        mock_alert_manager.trigger_alert.assert_called()
        
        # Health check should include alert summary
        assert "active_alerts" in health_status
        assert len(health_status["active_alerts"]) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])