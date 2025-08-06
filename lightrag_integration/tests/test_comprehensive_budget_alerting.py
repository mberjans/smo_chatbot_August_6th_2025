"""
Comprehensive Test Suite for Budget Alerting and Cost Management Systems

This module provides comprehensive testing for all components of the budget alerting
and cost management infrastructure, including:

- Alert notification system (email, webhooks, Slack)
- Real-time budget monitoring
- Cost-based circuit breakers
- Alert escalation management
- Dashboard API endpoints
- Integration between all components

Test Classes:
    - TestAlertNotificationSystem: Tests for alert delivery mechanisms
    - TestRealTimeBudgetMonitor: Tests for real-time monitoring
    - TestCostBasedCircuitBreaker: Tests for cost-aware circuit breaking
    - TestBudgetDashboardAPI: Tests for dashboard endpoints
    - TestIntegrationScenarios: End-to-end integration tests
"""

import pytest
import time
import json
import threading
import logging
import tempfile
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import the modules under test
from ..alert_system import (
    AlertNotificationSystem, AlertEscalationManager, AlertConfig, EmailAlertConfig,
    WebhookAlertConfig, SlackAlertConfig, AlertChannel
)
from ..realtime_budget_monitor import (
    RealTimeBudgetMonitor, CostProjectionEngine, BudgetHealthMetrics,
    BudgetMonitoringEvent, MonitoringEventType
)
from ..cost_based_circuit_breaker import (
    CostBasedCircuitBreaker, CostCircuitBreakerManager, OperationCostEstimator,
    CostThresholdRule, CostThresholdType, CircuitBreakerState
)
from ..budget_dashboard import (
    BudgetDashboardAPI, AnalyticsEngine, DashboardMetrics, DashboardTimeRange
)
from ..budget_manager import BudgetManager, BudgetAlert, AlertLevel, BudgetThreshold
from ..api_metrics_logger import APIUsageMetricsLogger, APIMetric, MetricType
from ..cost_persistence import CostPersistence, CostRecord, ResearchCategory


class TestAlertNotificationSystem:
    """Test suite for alert notification system."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_cost_persistence(self, temp_db_path):
        """Create mock cost persistence for testing."""
        return CostPersistence(db_path=temp_db_path)
    
    @pytest.fixture
    def mock_budget_manager(self, mock_cost_persistence):
        """Create mock budget manager for testing."""
        return BudgetManager(
            cost_persistence=mock_cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
    
    @pytest.fixture
    def basic_alert_config(self):
        """Create basic alert configuration for testing."""
        return AlertConfig(
            enabled_channels={AlertChannel.LOGGING, AlertChannel.CONSOLE},
            rate_limit_window=60.0,
            max_alerts_per_window=5
        )
    
    @pytest.fixture
    def alert_system(self, basic_alert_config):
        """Create alert notification system for testing."""
        return AlertNotificationSystem(basic_alert_config)
    
    def test_alert_system_initialization(self, alert_system):
        """Test alert system initialization."""
        assert alert_system.config is not None
        assert len(alert_system.config.enabled_channels) == 2
        assert AlertChannel.LOGGING in alert_system.config.enabled_channels
    
    def test_basic_alert_sending(self, alert_system):
        """Test basic alert sending functionality."""
        test_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=75.0,
            budget_limit=100.0,
            percentage_used=75.0,
            threshold_percentage=75.0,
            message="Test alert message"
        )
        
        result = alert_system.send_alert(test_alert, force=True)
        
        assert 'channels' in result
        assert 'logging' in result['channels']
        assert result['channels']['logging']['success'] is True
    
    def test_alert_rate_limiting(self, alert_system):
        """Test alert rate limiting functionality."""
        test_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=75.0,
            budget_limit=100.0,
            percentage_used=75.0,
            threshold_percentage=75.0,
            message="Rate limit test"
        )
        
        # Send first alert - should succeed
        result1 = alert_system.send_alert(test_alert, force=True)
        assert result1['channels']['logging']['success'] is True
        
        # Send same alert immediately - should be rate limited
        result2 = alert_system.send_alert(test_alert)
        assert result2.get('skipped') is True
    
    def test_email_alert_config_validation(self):
        """Test email alert configuration validation."""
        # Valid config
        email_config = EmailAlertConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            recipient_emails=["test@example.com"]
        )
        assert email_config.smtp_server == "smtp.gmail.com"
        
        # Invalid config - no recipients
        with pytest.raises(ValueError, match="At least one recipient email is required"):
            EmailAlertConfig(
                smtp_server="smtp.gmail.com",
                smtp_port=587,
                recipient_emails=[]
            )
    
    def test_webhook_alert_config_validation(self):
        """Test webhook alert configuration validation."""
        # Valid config
        webhook_config = WebhookAlertConfig(url="https://example.com/webhook")
        assert webhook_config.url == "https://example.com/webhook"
        
        # Invalid config - no URL
        with pytest.raises(ValueError, match="URL is required for webhook alerts"):
            WebhookAlertConfig(url="")
    
    def test_alert_escalation_manager(self, alert_system):
        """Test alert escalation functionality."""
        escalation_manager = AlertEscalationManager(alert_system)
        
        test_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=75.0,
            budget_limit=100.0,
            percentage_used=75.0,
            threshold_percentage=75.0,
            message="Escalation test"
        )
        
        result = escalation_manager.process_alert(test_alert)
        
        assert result['alert_processed'] is True
        assert 'delivery_result' in result
    
    def test_alert_template_rendering(self, alert_system):
        """Test alert template rendering functionality."""
        test_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.CRITICAL,
            period_type="monthly",
            period_key="2025-08",
            current_cost=2800.0,
            budget_limit=3000.0,
            percentage_used=93.3,
            threshold_percentage=90.0,
            message="Template test alert"
        )
        
        # Test text generation
        text_content = alert_system._generate_text_alert(test_alert)
        assert "Template test alert" in text_content
        assert "93.3%" in text_content
        
        # Test HTML generation
        html_content = alert_system._generate_html_alert(test_alert)
        assert "Template test alert" in html_content
        assert "93.3%" in html_content
    
    def test_alert_delivery_stats(self, alert_system):
        """Test alert delivery statistics tracking."""
        test_alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=AlertLevel.INFO,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=25.0,
            budget_limit=100.0,
            percentage_used=25.0,
            threshold_percentage=25.0,
            message="Stats test"
        )
        
        # Send alert
        alert_system.send_alert(test_alert, force=True)
        
        # Check stats
        stats = alert_system.get_delivery_stats()
        assert 'channels' in stats
        assert stats['alert_history_size'] >= 0


class TestRealTimeBudgetMonitor:
    """Test suite for real-time budget monitoring."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_cost_persistence(self, temp_db_path):
        """Create mock cost persistence for testing."""
        return CostPersistence(db_path=temp_db_path)
    
    @pytest.fixture
    def mock_budget_manager(self, mock_cost_persistence):
        """Create mock budget manager for testing."""
        return BudgetManager(
            cost_persistence=mock_cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
    
    @pytest.fixture
    def mock_api_metrics_logger(self):
        """Create mock API metrics logger."""
        mock_logger = Mock(spec=APIUsageMetricsLogger)
        mock_logger.get_performance_summary.return_value = {
            'current_hour': {
                'total_calls': 10,
                'total_tokens': 1000,
                'total_cost': 0.15,
                'avg_response_time_ms': 1200
            },
            'current_day': {
                'total_calls': 50,
                'total_tokens': 5000,
                'total_cost': 0.75,
                'avg_response_time_ms': 1100
            }
        }
        return mock_logger
    
    @pytest.fixture
    def real_time_monitor(self, mock_budget_manager, mock_api_metrics_logger, mock_cost_persistence):
        """Create real-time budget monitor for testing."""
        return RealTimeBudgetMonitor(
            budget_manager=mock_budget_manager,
            api_metrics_logger=mock_api_metrics_logger,
            cost_persistence=mock_cost_persistence,
            monitoring_interval=1.0  # Fast interval for testing
        )
    
    def test_cost_projection_engine_initialization(self):
        """Test cost projection engine initialization."""
        engine = CostProjectionEngine()
        assert engine._cost_history is not None
        assert len(engine._cost_history) == 0
    
    def test_cost_projection_with_data(self):
        """Test cost projection with sample data."""
        engine = CostProjectionEngine()
        
        # Add sample cost data points
        now = time.time()
        for i in range(10):
            engine.add_cost_datapoint(
                cost=0.1 + (i * 0.01),  # Increasing cost
                timestamp=now - (9 - i) * 3600,  # Hourly data points
                metadata={'test': True}
            )
        
        # Test daily projection
        projection = engine.project_daily_cost()
        assert 'projected_cost' in projection
        assert 'confidence' in projection
        assert projection['projected_cost'] > 0
    
    def test_anomaly_detection(self):
        """Test cost anomaly detection."""
        engine = CostProjectionEngine()
        
        # Add normal cost data
        now = time.time()
        for i in range(20):
            normal_cost = 0.1
            engine.add_cost_datapoint(normal_cost, now - i * 100)
        
        # Add anomalous cost
        engine.add_cost_datapoint(1.0, now)  # 10x normal cost
        
        anomalies = engine.detect_cost_anomalies()
        assert len(anomalies) > 0
        assert anomalies[0]['cost'] == 1.0
        assert anomalies[0]['severity'] in ['medium', 'high']
    
    def test_budget_health_metrics(self):
        """Test budget health metrics calculation."""
        health_metrics = BudgetHealthMetrics()
        
        # Mock budget status
        budget_status = {
            'daily_budget': {'percentage_used': 45.0},
            'monthly_budget': {'percentage_used': 60.0}
        }
        
        # Mock cost trends
        cost_trends = {
            'trend_percentage': 5.0,
            'confidence': 0.8
        }
        
        recent_alerts = []
        anomalies = []
        
        health_score = health_metrics.calculate_health_score(
            budget_status, cost_trends, recent_alerts, anomalies
        )
        
        assert 'overall_score' in health_score
        assert 'health_status' in health_score
        assert 'component_scores' in health_score
        assert 0 <= health_score['overall_score'] <= 100
    
    def test_monitoring_event_creation(self):
        """Test monitoring event creation and serialization."""
        event = BudgetMonitoringEvent(
            event_type=MonitoringEventType.THRESHOLD_WARNING,
            period_type="daily",
            period_key="2025-08-06",
            current_cost=75.0,
            budget_limit=100.0,
            percentage_used=75.0,
            severity="warning",
            message="Test monitoring event"
        )
        
        event_dict = event.to_dict()
        assert event_dict['event_type'] == 'threshold_warning'
        assert event_dict['current_cost'] == 75.0
        assert 'timestamp_iso' in event_dict
    
    def test_real_time_monitor_status(self, real_time_monitor):
        """Test real-time monitor status reporting."""
        status = real_time_monitor.get_monitoring_status()
        
        assert 'monitoring_active' in status
        assert 'statistics' in status
        assert 'health_score' in status
        assert 'projections' in status
        assert 'budget_status' in status
    
    def test_forced_monitoring_cycle(self, real_time_monitor):
        """Test forced monitoring cycle execution."""
        result = real_time_monitor.force_monitoring_cycle()
        
        assert 'success' in result
        assert 'timestamp' in result
        assert result['success'] is True
    
    def test_dashboard_metrics_generation(self, real_time_monitor):
        """Test dashboard metrics generation."""
        metrics = real_time_monitor.get_dashboard_metrics()
        
        assert 'system_health' in metrics
        assert 'budget_utilization' in metrics
        assert 'cost_projections' in metrics
        assert 'alerts' in metrics
        assert 'monitoring' in metrics


class TestCostBasedCircuitBreaker:
    """Test suite for cost-based circuit breaker system."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_cost_persistence(self, temp_db_path):
        """Create mock cost persistence for testing."""
        return CostPersistence(db_path=temp_db_path)
    
    @pytest.fixture
    def mock_budget_manager(self, mock_cost_persistence):
        """Create mock budget manager for testing."""
        manager = BudgetManager(
            cost_persistence=mock_cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
        return manager
    
    @pytest.fixture
    def operation_cost_estimator(self, mock_cost_persistence):
        """Create operation cost estimator for testing."""
        return OperationCostEstimator(mock_cost_persistence)
    
    @pytest.fixture
    def cost_threshold_rules(self):
        """Create sample cost threshold rules."""
        return [
            CostThresholdRule(
                rule_id="test_daily_limit",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=90.0,
                action="throttle",
                throttle_factor=0.5
            ),
            CostThresholdRule(
                rule_id="test_operation_cost",
                threshold_type=CostThresholdType.OPERATION_COST,
                threshold_value=1.0,
                action="block"
            )
        ]
    
    @pytest.fixture
    def cost_circuit_breaker(self, mock_budget_manager, operation_cost_estimator, cost_threshold_rules):
        """Create cost-based circuit breaker for testing."""
        return CostBasedCircuitBreaker(
            name="test_breaker",
            budget_manager=mock_budget_manager,
            cost_estimator=operation_cost_estimator,
            threshold_rules=cost_threshold_rules
        )
    
    def test_cost_threshold_rule_validation(self):
        """Test cost threshold rule validation."""
        # Valid rule
        rule = CostThresholdRule(
            rule_id="valid_rule",
            threshold_type=CostThresholdType.PERCENTAGE_DAILY,
            threshold_value=80.0,
            action="throttle"
        )
        assert rule.threshold_value == 80.0
        
        # Invalid threshold value
        with pytest.raises(ValueError, match="Threshold value must be positive"):
            CostThresholdRule(
                rule_id="invalid_rule",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=-10.0,
                action="throttle"
            )
        
        # Invalid action
        with pytest.raises(ValueError, match="Action must be 'block', 'throttle', or 'alert_only'"):
            CostThresholdRule(
                rule_id="invalid_action",
                threshold_type=CostThresholdType.PERCENTAGE_DAILY,
                threshold_value=80.0,
                action="invalid_action"
            )
    
    def test_operation_cost_estimation(self, operation_cost_estimator):
        """Test operation cost estimation."""
        # Test token-based estimation
        estimate = operation_cost_estimator.estimate_operation_cost(
            operation_type="llm_call",
            model_name="gpt-4o-mini",
            estimated_tokens={"input": 1000, "output": 500}
        )
        
        assert 'estimated_cost' in estimate
        assert 'confidence' in estimate
        assert estimate['estimated_cost'] > 0
        assert estimate['method'] == 'token_based'
    
    def test_cost_estimation_updates(self, operation_cost_estimator):
        """Test cost estimation updates with historical data."""
        # Add historical cost data
        operation_cost_estimator.update_historical_costs("llm_call", 0.05)
        operation_cost_estimator.update_historical_costs("llm_call", 0.08)
        operation_cost_estimator.update_historical_costs("llm_call", 0.06)
        
        # Test historical-based estimation
        estimate = operation_cost_estimator.estimate_operation_cost("llm_call")
        
        assert estimate['method'] == 'historical_average'
        assert 'samples' in estimate
        assert estimate['samples'] == 3
    
    def test_circuit_breaker_initialization(self, cost_circuit_breaker):
        """Test circuit breaker initialization."""
        assert cost_circuit_breaker.name == "test_breaker"
        assert cost_circuit_breaker.state == CircuitBreakerState.CLOSED
        assert len(cost_circuit_breaker.threshold_rules) == 2
    
    def test_circuit_breaker_operation_execution(self, cost_circuit_breaker):
        """Test circuit breaker operation execution."""
        def test_operation():
            return "success"
        
        # Normal operation should succeed
        result = cost_circuit_breaker.call(
            test_operation,
            operation_type="test_op",
            estimated_tokens={"input": 100, "output": 50}
        )
        
        assert result == "success"
    
    def test_circuit_breaker_cost_blocking(self, cost_circuit_breaker):
        """Test circuit breaker cost-based blocking."""
        def expensive_operation():
            return "expensive_result"
        
        # This should be blocked by the operation cost rule (>= $1.00)
        with pytest.raises(Exception, match="blocked by cost-based circuit breaker"):
            cost_circuit_breaker.call(
                expensive_operation,
                operation_type="expensive_op",
                estimated_tokens={"input": 10000, "output": 5000}  # High token count
            )
    
    def test_circuit_breaker_status(self, cost_circuit_breaker):
        """Test circuit breaker status reporting."""
        status = cost_circuit_breaker.get_status()
        
        assert 'name' in status
        assert 'state' in status
        assert 'statistics' in status
        assert 'cost_efficiency' in status
        assert status['name'] == "test_breaker"
    
    def test_circuit_breaker_manager(self, mock_budget_manager, mock_cost_persistence):
        """Test circuit breaker manager functionality."""
        manager = CostCircuitBreakerManager(
            budget_manager=mock_budget_manager,
            cost_persistence=mock_cost_persistence
        )
        
        # Create circuit breaker through manager
        breaker = manager.create_circuit_breaker("test_managed_breaker")
        assert breaker.name == "test_managed_breaker"
        
        # Test execution through manager
        def test_operation():
            return "managed_success"
        
        result = manager.execute_with_protection(
            "test_managed_breaker",
            test_operation,
            "test_operation"
        )
        
        assert result == "managed_success"
    
    def test_circuit_breaker_manager_system_status(self, mock_budget_manager, mock_cost_persistence):
        """Test circuit breaker manager system status."""
        manager = CostCircuitBreakerManager(
            budget_manager=mock_budget_manager,
            cost_persistence=mock_cost_persistence
        )
        
        # Create a few circuit breakers
        manager.create_circuit_breaker("breaker1")
        manager.create_circuit_breaker("breaker2")
        
        status = manager.get_system_status()
        
        assert 'circuit_breakers' in status
        assert 'manager_statistics' in status
        assert 'system_health' in status
        assert len(status['circuit_breakers']) == 2


class TestBudgetDashboardAPI:
    """Test suite for budget dashboard API."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_cost_persistence(self, temp_db_path):
        """Create mock cost persistence for testing."""
        return CostPersistence(db_path=temp_db_path)
    
    @pytest.fixture
    def mock_budget_manager(self, mock_cost_persistence):
        """Create mock budget manager for testing."""
        return BudgetManager(
            cost_persistence=mock_cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
    
    @pytest.fixture
    def mock_api_metrics_logger(self):
        """Create mock API metrics logger."""
        mock_logger = Mock(spec=APIUsageMetricsLogger)
        mock_logger.get_performance_summary.return_value = {
            'current_hour': {
                'total_calls': 10,
                'total_tokens': 1000,
                'total_cost': 0.15
            },
            'current_day': {
                'total_calls': 50,
                'total_tokens': 5000,
                'total_cost': 0.75
            }
        }
        return mock_logger
    
    @pytest.fixture
    def dashboard_api(self, mock_budget_manager, mock_api_metrics_logger, mock_cost_persistence):
        """Create budget dashboard API for testing."""
        return BudgetDashboardAPI(
            budget_manager=mock_budget_manager,
            api_metrics_logger=mock_api_metrics_logger,
            cost_persistence=mock_cost_persistence
        )
    
    def test_dashboard_metrics_initialization(self):
        """Test dashboard metrics initialization."""
        metrics = DashboardMetrics()
        assert metrics.budget_health_score == 0.0
        assert metrics.budget_health_status == "unknown"
        assert metrics.daily_cost == 0.0
    
    def test_dashboard_metrics_serialization(self):
        """Test dashboard metrics serialization."""
        metrics = DashboardMetrics(
            budget_health_score=85.5,
            budget_health_status="good",
            daily_cost=45.75,
            daily_budget=100.0,
            daily_percentage=45.75
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['budget_health_score'] == 85.5
        assert metrics_dict['budget_health_status'] == "good"
        assert 'last_update_time_iso' in metrics_dict
    
    def test_analytics_engine_initialization(self, mock_cost_persistence):
        """Test analytics engine initialization."""
        engine = AnalyticsEngine(mock_cost_persistence)
        assert engine.cost_persistence is mock_cost_persistence
        assert len(engine._analytics_cache) == 0
    
    def test_cost_trends_generation(self, mock_cost_persistence):
        """Test cost trends generation."""
        engine = AnalyticsEngine(mock_cost_persistence)
        
        # Mock the cost report
        with patch.object(mock_cost_persistence, 'generate_cost_report') as mock_report:
            mock_report.return_value = {
                'daily_costs': {
                    '2025-08-01': 10.0,
                    '2025-08-02': 12.0,
                    '2025-08-03': 15.0,
                    '2025-08-04': 11.0,
                    '2025-08-05': 14.0
                },
                'summary': {'total_cost': 62.0}
            }
            
            trends = engine.generate_cost_trends(DashboardTimeRange.LAST_7_DAYS)
            
            assert 'time_series' in trends
            assert 'total_cost' in trends
            assert 'trend_analysis' in trends
            assert trends['total_cost'] == 62.0
    
    def test_dashboard_overview_endpoint(self, dashboard_api):
        """Test dashboard overview endpoint."""
        overview = dashboard_api.get_dashboard_overview()
        
        assert overview['status'] == 'success'
        assert 'data' in overview
        assert 'metrics' in overview['data']
        assert 'system_health' in overview['data']
        assert 'meta' in overview
    
    def test_budget_status_endpoint(self, dashboard_api):
        """Test budget status endpoint."""
        status = dashboard_api.get_budget_status()
        
        assert status['status'] == 'success'
        assert 'data' in status
        assert 'budget_summary' in status['data']
        assert 'timestamp' in status['data']
    
    def test_cost_analytics_endpoint(self, dashboard_api):
        """Test cost analytics endpoint."""
        with patch.object(dashboard_api.analytics_engine, 'generate_cost_trends') as mock_trends:
            mock_trends.return_value = {
                'time_range': 'last_7_days',
                'total_cost': 100.0,
                'trend_analysis': {'trend_percentage': 5.0}
            }
            
            analytics = dashboard_api.get_cost_analytics()
            
            assert analytics['status'] == 'success'
            assert 'data' in analytics
            assert 'trends' in analytics['data']
    
    def test_alert_dashboard_endpoint(self, dashboard_api):
        """Test alert dashboard endpoint."""
        alert_dashboard = dashboard_api.get_alert_dashboard()
        
        assert alert_dashboard['status'] == 'success'
        assert 'data' in alert_dashboard
        assert 'recent_alerts' in alert_dashboard['data']
        assert 'alert_statistics' in alert_dashboard['data']
    
    def test_performance_metrics_endpoint(self, dashboard_api):
        """Test performance metrics endpoint."""
        metrics = dashboard_api.get_performance_metrics()
        
        assert metrics['status'] == 'success'
        assert 'data' in metrics
        assert 'performance_summary' in metrics['data']
    
    def test_health_check_endpoint(self, dashboard_api):
        """Test API health check endpoint."""
        health = dashboard_api.get_api_health_check()
        
        assert health['status'] == 'healthy'
        assert 'timestamp' in health
        assert 'components' in health
        assert health['components']['budget_manager'] is True
    
    def test_trigger_budget_check_endpoint(self, dashboard_api):
        """Test manual budget check trigger."""
        result = dashboard_api.trigger_budget_check()
        
        assert result['status'] == 'success'
        assert 'data' in result
        assert 'results' in result['data']


class TestIntegrationScenarios:
    """End-to-end integration tests for the complete budget management system."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def integrated_system(self, temp_db_path):
        """Create fully integrated budget management system."""
        # Initialize core components
        cost_persistence = CostPersistence(db_path=temp_db_path)
        
        budget_manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
        
        # Mock API metrics logger
        api_metrics_logger = Mock(spec=APIUsageMetricsLogger)
        api_metrics_logger.get_performance_summary.return_value = {
            'current_hour': {'total_calls': 10, 'total_tokens': 1000, 'total_cost': 0.15},
            'current_day': {'total_calls': 50, 'total_tokens': 5000, 'total_cost': 0.75}
        }
        
        # Initialize alert system
        alert_config = AlertConfig(
            enabled_channels={AlertChannel.LOGGING},
            rate_limit_window=60.0
        )
        alert_system = AlertNotificationSystem(alert_config)
        escalation_manager = AlertEscalationManager(alert_system)
        
        # Initialize real-time monitor
        real_time_monitor = RealTimeBudgetMonitor(
            budget_manager=budget_manager,
            api_metrics_logger=api_metrics_logger,
            cost_persistence=cost_persistence,
            alert_system=alert_system,
            escalation_manager=escalation_manager,
            monitoring_interval=0.1  # Fast for testing
        )
        
        # Initialize circuit breaker manager
        circuit_breaker_manager = CostCircuitBreakerManager(
            budget_manager=budget_manager,
            cost_persistence=cost_persistence
        )
        
        # Initialize dashboard
        dashboard_api = BudgetDashboardAPI(
            budget_manager=budget_manager,
            api_metrics_logger=api_metrics_logger,
            cost_persistence=cost_persistence,
            alert_system=alert_system,
            escalation_manager=escalation_manager,
            real_time_monitor=real_time_monitor,
            circuit_breaker_manager=circuit_breaker_manager
        )
        
        return {
            'cost_persistence': cost_persistence,
            'budget_manager': budget_manager,
            'api_metrics_logger': api_metrics_logger,
            'alert_system': alert_system,
            'escalation_manager': escalation_manager,
            'real_time_monitor': real_time_monitor,
            'circuit_breaker_manager': circuit_breaker_manager,
            'dashboard_api': dashboard_api
        }
    
    def test_complete_budget_alert_flow(self, integrated_system):
        """Test complete budget alert flow from cost recording to notification."""
        components = integrated_system
        
        # Record some costs to approach budget limit
        cost_record = CostRecord(
            timestamp=time.time(),
            session_id="test_session",
            operation_type="llm_call",
            model_name="gpt-4o-mini",
            cost_usd=85.0,  # Close to daily budget limit
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            research_category=ResearchCategory.GENERAL_QUERY.value
        )
        
        components['cost_persistence'].record_cost(cost_record)
        
        # Check budget status and trigger alerts
        budget_status = components['budget_manager'].check_budget_status(
            cost_amount=10.0,  # This should push over warning threshold
            operation_type="llm_call",
            research_category=ResearchCategory.GENERAL_QUERY
        )
        
        assert budget_status is not None
        assert 'alerts_generated' in budget_status
    
    def test_circuit_breaker_integration(self, integrated_system):
        """Test circuit breaker integration with budget system."""
        components = integrated_system
        
        # Create circuit breaker
        circuit_breaker = components['circuit_breaker_manager'].create_circuit_breaker(
            "integration_test_breaker"
        )
        
        def test_operation():
            return "operation_success"
        
        # Execute operation through circuit breaker
        result = components['circuit_breaker_manager'].execute_with_protection(
            "integration_test_breaker",
            test_operation,
            "test_operation"
        )
        
        assert result == "operation_success"
        
        # Check circuit breaker status
        status = circuit_breaker.get_status()
        assert status['statistics']['total_calls'] >= 1
        assert status['statistics']['allowed_calls'] >= 1
    
    def test_real_time_monitoring_integration(self, integrated_system):
        """Test real-time monitoring integration with other components."""
        components = integrated_system
        
        # Get monitoring status
        status = components['real_time_monitor'].get_monitoring_status()
        
        assert 'monitoring_active' in status
        assert 'health_score' in status
        assert 'budget_status' in status
        
        # Test forced monitoring cycle
        cycle_result = components['real_time_monitor'].force_monitoring_cycle()
        assert cycle_result['success'] is True
    
    def test_dashboard_integration(self, integrated_system):
        """Test dashboard integration with all components."""
        components = integrated_system
        
        # Test dashboard overview
        overview = components['dashboard_api'].get_dashboard_overview()
        assert overview['status'] == 'success'
        assert 'metrics' in overview['data']
        
        # Test budget status
        budget_status = components['dashboard_api'].get_budget_status()
        assert budget_status['status'] == 'success'
        
        # Test cost analytics
        analytics = components['dashboard_api'].get_cost_analytics()
        assert analytics['status'] == 'success'
        
        # Test alert dashboard
        alert_dashboard = components['dashboard_api'].get_alert_dashboard()
        assert alert_dashboard['status'] == 'success'
    
    def test_stress_scenario_high_usage(self, integrated_system):
        """Test system behavior under high usage scenario."""
        components = integrated_system
        
        # Simulate high usage by recording many costs
        for i in range(10):
            cost_record = CostRecord(
                timestamp=time.time() + i,
                session_id=f"stress_session_{i}",
                operation_type="llm_call",
                model_name="gpt-4o-mini",
                cost_usd=8.0,  # Will accumulate to 80.0
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                research_category=ResearchCategory.GENERAL_QUERY.value
            )
            components['cost_persistence'].record_cost(cost_record)
        
        # Check system still responds properly
        overview = components['dashboard_api'].get_dashboard_overview()
        assert overview['status'] == 'success'
        
        # Verify budget calculations
        budget_summary = components['budget_manager'].get_budget_summary()
        daily_cost = budget_summary.get('daily_budget', {}).get('total_cost', 0)
        assert daily_cost > 70.0  # Should have accumulated significant cost
    
    def test_error_recovery_scenario(self, integrated_system):
        """Test system recovery from error conditions."""
        components = integrated_system
        
        # Simulate error condition by forcing circuit breaker open
        circuit_breaker = components['circuit_breaker_manager'].create_circuit_breaker(
            "error_test_breaker"
        )
        circuit_breaker.force_open("Test error condition")
        
        # Verify circuit breaker is open
        status = circuit_breaker.get_status()
        assert status['state'] == 'open'
        
        # Test system health reporting
        overview = components['dashboard_api'].get_dashboard_overview()
        assert overview['status'] == 'success'
        
        # Recovery: force circuit breaker closed
        circuit_breaker.force_close("Test recovery")
        
        # Verify recovery
        status = circuit_breaker.get_status()
        assert status['state'] == 'closed'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])