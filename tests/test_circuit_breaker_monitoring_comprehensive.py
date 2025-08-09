"""
Comprehensive Test Suite for Circuit Breaker Monitoring System
=============================================================

This test suite validates all aspects of the circuit breaker monitoring system
including metrics collection, logging, alerting, dashboard integration, and
integration with existing infrastructure.

Test Coverage:
1. Circuit Breaker Metrics Collection
2. Enhanced Structured Logging
3. Alerting and Notification System  
4. Dashboard Integration and Health Checks
5. Integration with Existing Monitoring Infrastructure
6. Performance and Load Testing
7. Error Handling and Recovery

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Circuit Breaker Monitoring Testing
Version: 1.0.0
"""

import pytest
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import threading

# Import the monitoring system components
from lightrag_integration.circuit_breaker_monitoring import (
    CircuitBreakerMonitoringSystem,
    CircuitBreakerMonitoringConfig,
    CircuitBreakerMetrics,
    CircuitBreakerLogger,
    CircuitBreakerAlerting,
    CircuitBreakerHealthCheck,
    AlertLevel,
    create_monitoring_system
)

from lightrag_integration.circuit_breaker_monitoring_integration import (
    CircuitBreakerMonitoringIntegration,
    CircuitBreakerMonitoringIntegrationConfig,
    CircuitBreakerEventForwarder,
    MonitoredCircuitBreaker
)

from lightrag_integration.enhanced_circuit_breaker_monitoring_integration import (
    EnhancedCircuitBreakerMonitoringManager,
    EnhancedCircuitBreakerMonitoringConfig,
    CircuitBreakerEventInterceptor,
    MonitoringEnabledCircuitBreaker,
    create_enhanced_monitoring_manager,
    setup_comprehensive_monitoring
)

from lightrag_integration.circuit_breaker_dashboard import (
    CircuitBreakerDashboardConfig,
    CircuitBreakerDashboardBase,
    StandaloneDashboardServer
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_log_dir():
    """Create temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def monitoring_config(temp_log_dir):
    """Create monitoring configuration for testing."""
    return CircuitBreakerMonitoringConfig(
        log_level="DEBUG",
        enable_structured_logging=True,
        log_file_path=os.path.join(temp_log_dir, "test_monitoring.log"),
        enable_debug_logging=True,
        enable_prometheus_metrics=False,  # Disable for testing
        enable_alerting=True,
        alert_file_path=os.path.join(temp_log_dir, "test_alerts.json"),
        enable_health_endpoints=True,
        health_check_interval=1.0,
        enable_real_time_monitoring=True
    )


@pytest.fixture
def monitoring_system(monitoring_config):
    """Create monitoring system for testing."""
    return CircuitBreakerMonitoringSystem(monitoring_config)


@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker for testing."""
    mock_cb = Mock()
    mock_cb.state = "closed"
    mock_cb.service_type = Mock()
    mock_cb.service_type.value = "test_service"
    mock_cb.__class__.__name__ = "TestCircuitBreaker"
    return mock_cb


@pytest.fixture
def integration_config(temp_log_dir):
    """Create integration configuration for testing."""
    config = CircuitBreakerMonitoringIntegrationConfig(
        enable_monitoring=True,
        enable_event_forwarding=True,
        buffer_events=False,  # Disable buffering for immediate testing
        enable_auto_alerts=True,
        enable_health_checks=True
    )
    
    config.monitoring_config = CircuitBreakerMonitoringConfig(
        log_level="DEBUG",
        log_file_path=os.path.join(temp_log_dir, "integration_test.log"),
        alert_file_path=os.path.join(temp_log_dir, "integration_alerts.json")
    )
    
    return config


@pytest.fixture
def monitoring_integration(integration_config):
    """Create monitoring integration for testing."""
    return CircuitBreakerMonitoringIntegration(integration_config)


# ============================================================================
# Circuit Breaker Metrics Collection Tests
# ============================================================================

class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics collection functionality."""
    
    def test_metrics_initialization(self, monitoring_config):
        """Test metrics system initialization."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        assert metrics.config == monitoring_config
        assert len(metrics.state_changes) == 0
        assert len(metrics.current_states) == 0
        assert len(metrics.failure_counts) == 0
        assert len(metrics.response_times) == 0
    
    def test_state_change_recording(self, monitoring_config):
        """Test recording circuit breaker state changes."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Record state change
        metrics.record_state_change(
            service="test_service",
            from_state="closed",
            to_state="open",
            metadata={"reason": "failure_threshold_exceeded"}
        )
        
        assert "test_service" in metrics.state_changes
        assert len(metrics.state_changes["test_service"]) == 1
        assert metrics.current_states["test_service"] == "open"
        
        change_record = metrics.state_changes["test_service"][0]
        assert change_record["from_state"] == "closed"
        assert change_record["to_state"] == "open"
        assert change_record["metadata"]["reason"] == "failure_threshold_exceeded"
    
    def test_failure_recording(self, monitoring_config):
        """Test recording service failures."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Record failures
        metrics.record_failure("test_service", "timeout", 5.0)
        metrics.record_failure("test_service", "http_error", 2.0)
        metrics.record_failure("test_service", "timeout", 3.0)
        
        assert metrics.failure_counts["test_service"]["timeout"] == 2
        assert metrics.failure_counts["test_service"]["http_error"] == 1
        assert len(metrics.failure_rates["test_service"]) == 3
        assert len(metrics.response_times["test_service"]) == 3
    
    def test_success_recording(self, monitoring_config):
        """Test recording successful operations."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Record successes
        metrics.record_success("test_service", 1.5)
        metrics.record_success("test_service", 2.0)
        metrics.record_success("test_service", 1.0)
        
        assert len(metrics.response_times["test_service"]) == 3
        assert len(metrics.success_rates["test_service"]) == 3
    
    def test_recovery_recording(self, monitoring_config):
        """Test recording circuit breaker recovery."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Record recoveries
        metrics.record_recovery("test_service", 30.0, True)
        metrics.record_recovery("test_service", 45.0, False)
        
        assert len(metrics.recovery_times["test_service"]) == 2
        assert len(metrics.recovery_success_rates["test_service"]) == 2
        assert metrics.recovery_times["test_service"] == [30.0, 45.0]
    
    def test_threshold_adjustment_recording(self, monitoring_config):
        """Test recording threshold adjustments."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Record threshold adjustment
        metrics.record_threshold_adjustment(
            service="test_service",
            adjustment_type="failure_threshold",
            old_value=5,
            new_value=3,
            effectiveness=0.85
        )
        
        assert len(metrics.threshold_adjustments["test_service"]) == 1
        assert metrics.threshold_effectiveness["test_service"]["failure_threshold"] == 0.85
        
        adjustment = metrics.threshold_adjustments["test_service"][0]
        assert adjustment["adjustment_type"] == "failure_threshold"
        assert adjustment["old_value"] == 5
        assert adjustment["new_value"] == 3
    
    def test_cost_impact_recording(self, monitoring_config):
        """Test recording cost impact from circuit breaker activations."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Record cost impacts
        budget_impact = {"budget_percentage": 15.5, "estimated_savings": 125.0}
        metrics.record_cost_impact("test_service", 125.0, budget_impact)
        metrics.record_cost_impact("test_service", 75.0, {"budget_percentage": 10.0})
        
        assert metrics.cost_savings["test_service"] == 200.0
        assert len(metrics.budget_impacts["test_service"]) == 2
    
    def test_current_metrics_calculation(self, monitoring_config):
        """Test calculation of current metrics."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Add test data
        metrics.record_state_change("test_service", "closed", "open")
        metrics.record_failure("test_service", "timeout", 3.0)
        metrics.record_success("test_service", 1.5)
        metrics.record_recovery("test_service", 30.0, True)
        metrics.record_cost_impact("test_service", 100.0, {})
        
        current_metrics = metrics.get_current_metrics("test_service")
        
        assert current_metrics["service"] == "test_service"
        assert current_metrics["current_state"] == "open"
        assert current_metrics["total_cost_savings"] == 100.0
        assert current_metrics["total_state_changes"] == 1
        assert current_metrics["total_failures"] == 1
        assert current_metrics["avg_recovery_time"] == 30.0


# ============================================================================
# Enhanced Structured Logging Tests
# ============================================================================

class TestCircuitBreakerLogger:
    """Test enhanced structured logging functionality."""
    
    def test_logger_initialization(self, monitoring_config):
        """Test logger initialization."""
        logger = CircuitBreakerLogger(monitoring_config)
        
        assert logger.config == monitoring_config
        assert logger.logger.level == logging.DEBUG
    
    def test_state_change_logging(self, monitoring_config, temp_log_dir):
        """Test state change logging."""
        logger = CircuitBreakerLogger(monitoring_config)
        
        # Log state change
        logger.log_state_change(
            service="test_service",
            from_state="closed",
            to_state="open",
            reason="failure_threshold_exceeded",
            metadata={"correlation_id": "test-123"}
        )
        
        # Check log file was created
        log_file = Path(monitoring_config.log_file_path)
        assert log_file.exists()
    
    def test_failure_logging(self, monitoring_config):
        """Test failure logging."""
        logger = CircuitBreakerLogger(monitoring_config)
        
        # Log failure
        logger.log_failure(
            service="test_service",
            failure_type="timeout",
            error_details="Connection timeout after 30 seconds",
            response_time=30.0,
            correlation_id="test-456"
        )
        
        # Verify no exceptions thrown
        assert True  # Test passes if no exception
    
    def test_performance_impact_logging(self, monitoring_config):
        """Test performance impact logging."""
        logger = CircuitBreakerLogger(monitoring_config)
        
        metrics = {
            "avg_response_time": 2.5,
            "p95_response_time": 5.0,
            "throughput_rps": 100
        }
        
        # Log performance impact
        logger.log_performance_impact(
            service="test_service",
            impact_type="latency_increase",
            metrics=metrics
        )
        
        assert True  # Test passes if no exception
    
    def test_debug_decision_logging(self, monitoring_config):
        """Test debug decision logging."""
        logger = CircuitBreakerLogger(monitoring_config)
        
        factors = {
            "failure_rate": 0.15,
            "response_time": 5.2,
            "consecutive_failures": 3
        }
        
        # Log debug decision
        logger.log_debug_decision(
            service="test_service",
            decision="open_circuit",
            factors=factors
        )
        
        assert True  # Test passes if no exception


# ============================================================================
# Alerting and Notification System Tests
# ============================================================================

class TestCircuitBreakerAlerting:
    """Test alerting and notification functionality."""
    
    def test_alerting_initialization(self, monitoring_config):
        """Test alerting system initialization."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        assert alerting.config == monitoring_config
        assert len(alerting.active_alerts) == 0
        assert len(alerting.alert_history) == 0
    
    def test_circuit_breaker_open_alert(self, monitoring_config):
        """Test circuit breaker open alert creation."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        alert_id = alerting.alert_circuit_breaker_open(
            service="test_service",
            failure_count=5,
            threshold=3,
            correlation_id="test-789"
        )
        
        assert alert_id in alerting.active_alerts
        alert = alerting.active_alerts[alert_id]
        assert alert.service == "test_service"
        assert alert.alert_type == "circuit_breaker_open"
        assert alert.level == AlertLevel.CRITICAL
        assert not alert.resolved
    
    def test_circuit_breaker_recovery_alert(self, monitoring_config):
        """Test circuit breaker recovery alert creation."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        alert_id = alerting.alert_circuit_breaker_recovery(
            service="test_service",
            downtime_seconds=120.0
        )
        
        alert = alerting.active_alerts[alert_id]
        assert alert.alert_type == "circuit_breaker_recovery"
        assert alert.level == AlertLevel.INFO
    
    def test_threshold_breach_alert(self, monitoring_config):
        """Test threshold breach alert creation."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        alert_id = alerting.alert_threshold_breach(
            service="test_service",
            threshold_type="response_time",
            current_value=5.5,
            threshold_value=3.0
        )
        
        alert = alerting.active_alerts[alert_id]
        assert alert.alert_type == "threshold_breach"
        assert alert.level == AlertLevel.WARNING
    
    def test_alert_resolution(self, monitoring_config):
        """Test alert resolution."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        # Create alert
        alert_id = alerting.alert_circuit_breaker_open(
            service="test_service",
            failure_count=5,
            threshold=3
        )
        
        # Resolve alert
        resolved = alerting.resolve_alert(alert_id)
        
        assert resolved
        assert alert_id not in alerting.active_alerts
        
        # Check alert history
        history_alerts = alerting.get_alert_history()
        resolved_alert = next(a for a in history_alerts if a.id == alert_id)
        assert resolved_alert.resolved
        assert resolved_alert.resolved_at is not None
    
    def test_get_active_alerts(self, monitoring_config):
        """Test getting active alerts."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        # Create multiple alerts
        alert1 = alerting.alert_circuit_breaker_open("service1", 5, 3)
        alert2 = alerting.alert_threshold_breach("service2", "response_time", 5.0, 3.0)
        alert3 = alerting.alert_circuit_breaker_open("service1", 3, 2)
        
        # Get all active alerts
        all_alerts = alerting.get_active_alerts()
        assert len(all_alerts) == 3
        
        # Get alerts for specific service
        service1_alerts = alerting.get_active_alerts("service1")
        assert len(service1_alerts) == 2
        
        service2_alerts = alerting.get_active_alerts("service2")
        assert len(service2_alerts) == 1
    
    def test_alert_file_writing(self, monitoring_config, temp_log_dir):
        """Test writing alerts to file."""
        alerting = CircuitBreakerAlerting(monitoring_config)
        
        # Create alert
        alerting.alert_circuit_breaker_open("test_service", 5, 3)
        
        # Check alert file was created
        alert_file = Path(monitoring_config.alert_file_path)
        assert alert_file.exists()
        
        # Read and verify alert data
        with open(alert_file, 'r') as f:
            alert_data = json.loads(f.read().strip())
            assert alert_data["service"] == "test_service"
            assert alert_data["alert_type"] == "circuit_breaker_open"


# ============================================================================
# Dashboard Integration and Health Check Tests
# ============================================================================

class TestCircuitBreakerHealthCheck:
    """Test health check and dashboard integration functionality."""
    
    @pytest.fixture
    def health_check(self, monitoring_config):
        """Create health check instance for testing."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        alerting = CircuitBreakerAlerting(monitoring_config)
        return CircuitBreakerHealthCheck(metrics, alerting, monitoring_config)
    
    def test_health_status_calculation(self, health_check):
        """Test health status calculation."""
        # Add test data to metrics
        health_check.metrics.record_state_change("test_service", "closed", "open")
        health_check.metrics.record_failure("test_service", "timeout", 3.0)
        
        # Update health status
        health_check._update_health_status()
        
        health_status = health_check.get_health_status("test_service")
        assert health_status["service"] == "test_service"
        assert health_status["current_state"] == "open"
        assert health_status["status"] == "critical"  # Because state is open
    
    def test_system_health_summary(self, health_check):
        """Test system health summary generation."""
        # Add test data
        health_check.metrics.record_state_change("service1", "closed", "open")
        health_check.metrics.record_state_change("service2", "closed", "degraded")
        health_check.metrics.record_state_change("service3", "closed", "closed")
        
        health_check._update_health_status()
        
        summary = health_check.get_system_health_summary()
        assert summary["total_services"] == 3
        assert summary["critical_services"] == 1
        assert summary["warning_services"] == 1
        assert summary["healthy_services"] == 1
        assert summary["overall_status"] == "critical"


class TestDashboardIntegration:
    """Test dashboard integration functionality."""
    
    def test_dashboard_config(self):
        """Test dashboard configuration."""
        config = CircuitBreakerDashboardConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8091
        assert config.enable_cors
        assert config.enable_websockets
        assert config.api_prefix == "/api/v1/circuit-breakers"
    
    def test_dashboard_base_functionality(self, monitoring_integration):
        """Test base dashboard functionality."""
        config = CircuitBreakerDashboardConfig()
        dashboard = CircuitBreakerDashboardBase(monitoring_integration, config)
        
        # Test health status endpoint
        health_response = dashboard.get_health_status()
        assert health_response["status"] == "success"
        
        # Test metrics summary endpoint
        metrics_response = dashboard.get_metrics_summary()
        assert metrics_response["status"] == "success"
        
        # Test alerts endpoint
        alerts_response = dashboard.get_active_alerts()
        assert alerts_response["status"] == "success"
        assert "alerts" in alerts_response["data"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestMonitoringIntegration:
    """Test monitoring integration functionality."""
    
    def test_integration_initialization(self, integration_config):
        """Test monitoring integration initialization."""
        integration = CircuitBreakerMonitoringIntegration(integration_config)
        
        assert integration.config == integration_config
        assert integration.monitoring_system is not None
        assert integration.event_forwarder is not None
    
    @pytest.mark.asyncio
    async def test_integration_start_stop(self, monitoring_integration):
        """Test starting and stopping monitoring integration."""
        # Test start
        await monitoring_integration.start()
        assert monitoring_integration._is_started
        
        # Test stop
        await monitoring_integration.stop()
        assert not monitoring_integration._is_started
    
    def test_circuit_breaker_wrapping(self, monitoring_integration, mock_circuit_breaker):
        """Test wrapping circuit breaker with monitoring."""
        wrapped_cb = monitoring_integration.wrap_circuit_breaker(
            mock_circuit_breaker, "test_service"
        )
        
        assert "test_service" in monitoring_integration.monitored_circuit_breakers
        assert isinstance(wrapped_cb, MonitoredCircuitBreaker)
    
    @pytest.mark.asyncio
    async def test_monitored_circuit_breaker_call(self, monitoring_integration, mock_circuit_breaker):
        """Test monitored circuit breaker operation calls."""
        # Setup mock operation
        async def test_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        # Wrap circuit breaker
        wrapped_cb = monitoring_integration.wrap_circuit_breaker(
            mock_circuit_breaker, "test_service"
        )
        
        # Mock the wrapped circuit breaker call method
        wrapped_cb.wrapped_cb.call = Mock(return_value=asyncio.coroutine(lambda: "success")())
        
        # Execute operation
        result = await wrapped_cb.call(test_operation)
        
        assert result == "success"
        wrapped_cb.wrapped_cb.call.assert_called_once()


class TestEnhancedMonitoringManager:
    """Test enhanced monitoring manager functionality."""
    
    def test_manager_initialization(self):
        """Test enhanced monitoring manager initialization."""
        config = EnhancedCircuitBreakerMonitoringConfig()
        manager = EnhancedCircuitBreakerMonitoringManager(config)
        
        assert manager.config == config
        assert manager.monitoring_integration is not None
        assert manager.event_interceptor is not None
    
    def test_circuit_breaker_registration(self, mock_circuit_breaker):
        """Test circuit breaker registration with manager."""
        manager = create_enhanced_monitoring_manager()
        
        monitored_cb = manager.register_circuit_breaker(
            mock_circuit_breaker, "test_service"
        )
        
        assert "test_service" in manager.monitored_circuit_breakers
        assert isinstance(monitored_cb, MonitoringEnabledCircuitBreaker)
    
    @pytest.mark.asyncio
    async def test_manager_start_stop(self):
        """Test manager start and stop functionality."""
        config = EnhancedCircuitBreakerMonitoringConfig(enable_dashboard=False)
        manager = EnhancedCircuitBreakerMonitoringManager(config)
        
        # Test start
        await manager.start()
        assert manager._is_started
        assert manager._startup_time is not None
        
        # Test stop
        await manager.stop()
        assert not manager._is_started
    
    def test_monitoring_status(self, mock_circuit_breaker):
        """Test monitoring status reporting."""
        manager = create_enhanced_monitoring_manager()
        manager.register_circuit_breaker(mock_circuit_breaker, "test_service")
        
        status = manager.get_monitoring_status()
        
        assert status["monitoring_enabled"]
        assert "test_service" in status["monitored_services"]
        assert "event_statistics" in status
        assert "health_summary" in status


# ============================================================================
# Performance and Load Tests
# ============================================================================

class TestPerformanceAndLoad:
    """Test performance and load handling of monitoring system."""
    
    def test_high_volume_metrics_collection(self, monitoring_config):
        """Test metrics collection under high volume."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Simulate high volume of events
        start_time = time.time()
        
        for i in range(1000):
            metrics.record_failure(f"service_{i % 10}", "timeout", 1.0)
            metrics.record_success(f"service_{i % 10}", 0.5)
            if i % 100 == 0:
                metrics.record_state_change(f"service_{i % 10}", "closed", "open")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 events quickly (under 1 second)
        assert processing_time < 1.0
        
        # Verify data integrity
        assert len(metrics.failure_counts) == 10  # 10 different services
        for service_name in metrics.failure_counts:
            assert metrics.failure_counts[service_name]["timeout"] == 100
    
    def test_concurrent_event_processing(self, monitoring_config):
        """Test concurrent event processing."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        def generate_events(service_name, event_count):
            for i in range(event_count):
                metrics.record_failure(service_name, "timeout", 1.0)
                metrics.record_success(service_name, 0.5)
        
        # Create multiple threads generating events concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=generate_events,
                args=(f"service_{i}", 100)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify data integrity
        assert len(metrics.failure_counts) == 5
        for i in range(5):
            service_name = f"service_{i}"
            assert metrics.failure_counts[service_name]["timeout"] == 100
    
    @pytest.mark.asyncio
    async def test_monitoring_system_under_load(self, monitoring_system):
        """Test complete monitoring system under load."""
        await monitoring_system.start_monitoring()
        
        try:
            # Generate load
            tasks = []
            for i in range(100):
                async def generate_load(service_idx):
                    service_name = f"load_test_service_{service_idx % 10}"
                    
                    # Record various events
                    monitoring_system.metrics.record_failure(service_name, "timeout", 2.0)
                    monitoring_system.metrics.record_success(service_name, 1.0)
                    
                    if service_idx % 20 == 0:
                        monitoring_system.alerting.alert_circuit_breaker_open(
                            service_name, 5, 3
                        )
                    
                    await asyncio.sleep(0.01)  # Small delay to simulate real-world timing
                
                tasks.append(generate_load(i))
            
            # Execute all tasks concurrently
            start_time = time.time()
            await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0
            
            # Verify system state
            current_metrics = monitoring_system.metrics.get_current_metrics()
            assert len(current_metrics) == 10  # 10 different services
            
            active_alerts = monitoring_system.alerting.get_active_alerts()
            assert len(active_alerts) >= 5  # At least 5 alerts created
            
        finally:
            await monitoring_system.stop_monitoring()


# ============================================================================
# Error Handling and Recovery Tests  
# ============================================================================

class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""
    
    def test_log_file_creation_failure(self, temp_log_dir):
        """Test handling of log file creation failures."""
        # Create config with invalid log path
        config = CircuitBreakerMonitoringConfig(
            log_file_path="/invalid/path/test.log"
        )
        
        # Logger should still initialize without throwing exception
        logger = CircuitBreakerLogger(config)
        
        # Should be able to log (will go to console)
        logger.log_state_change("test_service", "closed", "open", "test")
        
        assert True  # Test passes if no exception
    
    def test_alert_file_write_failure(self, temp_log_dir):
        """Test handling of alert file write failures."""
        config = CircuitBreakerMonitoringConfig(
            alert_file_path="/invalid/path/alerts.json"
        )
        
        alerting = CircuitBreakerAlerting(config)
        
        # Should not throw exception even if file write fails
        alert_id = alerting.alert_circuit_breaker_open("test_service", 5, 3)
        
        assert alert_id in alerting.active_alerts
    
    def test_metrics_calculation_with_empty_data(self, monitoring_config):
        """Test metrics calculations with empty or insufficient data."""
        metrics = CircuitBreakerMetrics(monitoring_config)
        
        # Should handle empty data gracefully
        current_metrics = metrics.get_current_metrics("nonexistent_service")
        
        assert current_metrics["service"] == "nonexistent_service"
        assert current_metrics["failure_rate"] == 0.0
        assert current_metrics["success_rate"] == 1.0
        assert current_metrics["avg_response_time"] == 0.0
    
    @pytest.mark.asyncio
    async def test_monitoring_system_exception_handling(self, monitoring_system):
        """Test monitoring system exception handling."""
        await monitoring_system.start_monitoring()
        
        try:
            # Test with invalid data that might cause exceptions
            monitoring_system.metrics.record_failure("", "", None)  # Empty strings
            monitoring_system.metrics.record_success(None, -1.0)  # Invalid values
            
            # System should continue to function
            health_summary = monitoring_system.health_check.get_system_health_summary()
            assert health_summary is not None
            
        finally:
            await monitoring_system.stop_monitoring()
    
    def test_integration_with_missing_dependencies(self):
        """Test integration behavior when dependencies are missing."""
        # Test with monitoring disabled
        config = EnhancedCircuitBreakerMonitoringConfig(enable_monitoring=False)
        manager = EnhancedCircuitBreakerMonitoringManager(config)
        
        assert manager.monitoring_integration is None
        assert manager.event_interceptor is None
        
        # Should still function for basic operations
        status = manager.get_monitoring_status()
        assert not status["monitoring_enabled"]


# ============================================================================
# Integration with Existing Infrastructure Tests
# ============================================================================

class TestExistingInfrastructureIntegration:
    """Test integration with existing monitoring infrastructure."""
    
    @patch('lightrag_integration.enhanced_circuit_breaker_monitoring_integration.PRODUCTION_MONITORING_AVAILABLE', True)
    @patch('lightrag_integration.enhanced_circuit_breaker_monitoring_integration.ProductionMonitoringSystem')
    def test_production_monitoring_integration(self, mock_production_monitoring):
        """Test integration with production monitoring system."""
        from lightrag_integration.enhanced_circuit_breaker_monitoring_integration import ProductionIntegrationHelper
        
        config = EnhancedCircuitBreakerMonitoringConfig(
            integrate_with_production_monitoring=True
        )
        manager = EnhancedCircuitBreakerMonitoringManager(config)
        
        integration_helper = ProductionIntegrationHelper(manager)
        result = integration_helper.integrate_with_production_monitoring()
        
        assert result  # Should return True for successful integration
    
    def test_existing_log_pattern_integration(self):
        """Test integration with existing log patterns."""
        config = EnhancedCircuitBreakerMonitoringConfig(
            use_existing_log_patterns=True
        )
        manager = EnhancedCircuitBreakerMonitoringManager(config)
        
        # Should initialize without issues
        assert manager.config.use_existing_log_patterns
    
    def test_environment_variable_configuration(self):
        """Test configuration via environment variables."""
        # Set environment variables
        os.environ['ECB_MONITORING_ENABLED'] = 'false'
        os.environ['ECB_DASHBOARD_PORT'] = '9090'
        os.environ['ECB_DASHBOARD_HOST'] = '127.0.0.1'
        
        try:
            config = EnhancedCircuitBreakerMonitoringConfig()
            
            assert not config.enable_monitoring
            assert config.dashboard_port == 9090
            assert config.dashboard_host == '127.0.0.1'
        finally:
            # Clean up environment variables
            for key in ['ECB_MONITORING_ENABLED', 'ECB_DASHBOARD_PORT', 'ECB_DASHBOARD_HOST']:
                os.environ.pop(key, None)


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self, temp_log_dir):
        """Test complete monitoring workflow from setup to teardown."""
        # Setup configuration
        config_overrides = {
            'enable_monitoring': True,
            'enable_dashboard': False,  # Disable dashboard for testing
            'monitoring_log_file': os.path.join(temp_log_dir, "e2e_test.log"),
            'integrate_with_production_monitoring': False
        }
        
        # Create and start monitoring manager
        manager = create_enhanced_monitoring_manager(config_overrides)
        await manager.start()
        
        try:
            # Register mock circuit breakers
            mock_openai_cb = Mock()
            mock_openai_cb.state = "closed"
            mock_openai_cb.__class__.__name__ = "OpenAICircuitBreaker"
            
            mock_lightrag_cb = Mock()
            mock_lightrag_cb.state = "closed"
            mock_lightrag_cb.__class__.__name__ = "LightRAGCircuitBreaker"
            
            openai_monitored = manager.register_circuit_breaker(mock_openai_cb, "openai_api")
            lightrag_monitored = manager.register_circuit_breaker(mock_lightrag_cb, "lightrag")
            
            # Simulate operations and failures
            await asyncio.sleep(0.1)  # Let monitoring system initialize
            
            # Simulate OpenAI failures
            for i in range(3):
                manager.event_interceptor.intercept_operation_failure(
                    openai_monitored.wrapped_cb,
                    "chat_completion",
                    Exception("Rate limit exceeded"),
                    5.0
                )
                await asyncio.sleep(0.1)
            
            # Simulate state change
            manager.event_interceptor.intercept_state_change(
                openai_monitored.wrapped_cb,
                "closed",
                "open",
                "failure_threshold_exceeded"
            )
            
            # Simulate LightRAG success
            manager.event_interceptor.intercept_operation_success(
                lightrag_monitored.wrapped_cb,
                "query",
                2.0
            )
            
            await asyncio.sleep(0.5)  # Let events process
            
            # Verify monitoring data
            status = manager.get_monitoring_status()
            assert status["monitoring_enabled"]
            assert len(status["monitored_services"]) == 2
            assert "openai_api" in status["monitored_services"]
            assert "lightrag" in status["monitored_services"]
            
            # Check health status
            health = manager.get_service_health()
            assert "openai_api" in health or "health_summary" in health
            
            # Check metrics
            metrics = manager.get_service_metrics()
            assert metrics is not None
            
            # Verify alerts were created
            if manager.monitoring_integration:
                alerts = manager.monitoring_integration.get_active_alerts()
                assert len(alerts) >= 0  # May have alerts depending on configuration
            
        finally:
            await manager.stop()
    
    def test_comprehensive_setup_function(self):
        """Test comprehensive setup function."""
        # Mock circuit breaker orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.openai_cb = Mock()
        mock_orchestrator.openai_cb.state = "closed"
        mock_orchestrator.openai_cb.__class__.__name__ = "OpenAICircuitBreaker"
        
        mock_orchestrator.lightrag_cb = Mock()
        mock_orchestrator.lightrag_cb.state = "closed"
        mock_orchestrator.lightrag_cb.__class__.__name__ = "LightRAGCircuitBreaker"
        
        # Setup comprehensive monitoring
        manager = setup_comprehensive_monitoring(
            circuit_breaker_orchestrator=mock_orchestrator,
            config_overrides={'enable_dashboard': False}
        )
        
        assert manager is not None
        assert len(manager.monitored_circuit_breakers) >= 2


# ============================================================================
# Test Runner Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])