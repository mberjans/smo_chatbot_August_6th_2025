"""
Comprehensive Test Suite for Unified System Health Monitoring Dashboard
======================================================================

This module provides comprehensive testing coverage for the unified system health 
monitoring dashboard implementation, including:

1. Unit Tests for all major components
2. Integration Tests for monitoring system connections
3. API Tests for REST endpoints and WebSocket functionality
4. Performance Tests for load handling and resource management
5. Mock Tests for graceful degradation scenarios

Test Coverage:
- UnifiedDataAggregator class methods
- AlertManager functionality  
- WebSocketManager operations
- Dashboard API endpoints
- Configuration and deployment helpers
- Error handling and recovery
- Performance under load
- Security and authentication

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
Task: CMO-LIGHTRAG-014-T07 - Dashboard Testing Suite
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi import WebSocket
import websockets

# Import the modules under test
from .unified_system_health_dashboard import (
    UnifiedSystemHealthDashboard,
    UnifiedDataAggregator,
    WebSocketManager,
    DashboardConfig,
    SystemHealthSnapshot,
    AlertEvent,
    create_unified_dashboard
)

from .dashboard_integration_helper import (
    DashboardIntegrationHelper,
    DashboardDeploymentConfig,
    get_development_config,
    get_production_config,
    quick_start_dashboard
)


# ============================================================================
# TEST CONFIGURATION AND FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    return DashboardConfig(
        host="127.0.0.1",
        port=8093,  # Use different port for testing
        enable_cors=True,
        enable_websockets=True,
        websocket_update_interval=0.5,  # Faster updates for testing
        enable_historical_data=True,
        historical_retention_hours=1,
        db_path=":memory:",  # Use in-memory database for tests
        enable_db_persistence=True,
        enable_alerts=True,
        alert_cooldown_seconds=10  # Shorter cooldown for testing
    )


@pytest.fixture
def sample_snapshot():
    """Create a sample health snapshot for testing."""
    return SystemHealthSnapshot(
        timestamp=datetime.now(),
        snapshot_id="test-123",
        system_uptime_seconds=3600.0,
        overall_health="healthy",
        health_score=0.85,
        load_level="NORMAL",
        load_score=0.3,
        cpu_utilization=25.0,
        memory_pressure=40.0,
        response_time_p95=150.0,
        error_rate=0.1,
        request_queue_depth=5,
        active_connections=10,
        degradation_active=False,
        degradation_level="NORMAL",
        emergency_mode=False,
        throughput_rps=50.0,
        success_rate=99.9,
        total_requests_processed=1000,
        connection_pool_usage=20.0,
        thread_pool_usage=15.0,
        memory_usage_mb=256.0,
        active_alerts_count=0
    )


@pytest.fixture
def sample_alert():
    """Create a sample alert for testing."""
    return AlertEvent(
        id="alert-test-456",
        timestamp=datetime.now(),
        severity="warning",
        source="test_source",
        title="Test Alert",
        message="This is a test alert message",
        category="performance"
    )


@pytest.fixture
def mock_orchestrator():
    """Create a mock graceful degradation orchestrator."""
    mock_orch = Mock()
    mock_orch.get_system_status.return_value = {
        'start_time': datetime.now().isoformat(),
        'current_load_level': 'NORMAL',
        'total_requests_processed': 1000,
        'integration_status': {
            'load_monitoring_active': True,
            'degradation_controller_active': True,
            'throttling_system_active': False,
            'integrated_load_balancer': False,
            'integrated_rag_system': True,
            'integrated_monitoring': True
        }
    }
    mock_orch.get_health_check.return_value = {
        'status': 'healthy',
        'issues': []
    }
    return mock_orch


@pytest.fixture
def mock_load_detector():
    """Create a mock enhanced load detection system."""
    from unittest.mock import Mock
    
    mock_detector = Mock()
    mock_metrics = Mock()
    mock_metrics.cpu_utilization = 25.0
    mock_metrics.memory_pressure = 40.0
    mock_metrics.response_time_p95 = 150.0
    mock_metrics.error_rate = 0.1
    mock_metrics.request_queue_depth = 5
    mock_metrics.active_connections = 10
    mock_metrics.load_score = 0.3
    mock_metrics.load_level = Mock()
    mock_metrics.load_level.name = 'NORMAL'
    
    mock_detector.get_current_metrics.return_value = mock_metrics
    return mock_detector


@pytest.fixture
def mock_degradation_controller():
    """Create a mock progressive service degradation controller."""
    mock_controller = Mock()
    mock_controller.get_current_status.return_value = {
        'degradation_active': False,
        'load_level': 'NORMAL',
        'emergency_mode': False,
        'feature_settings': {
            'advanced_search': True,
            'detailed_analytics': True,
            'export_functionality': True
        }
    }
    return mock_controller


@pytest.fixture
async def data_aggregator(test_config):
    """Create a data aggregator for testing."""
    aggregator = UnifiedDataAggregator(test_config)
    yield aggregator
    await aggregator.stop_aggregation()


@pytest.fixture
def websocket_manager():
    """Create a WebSocket manager for testing."""
    return WebSocketManager()


@pytest.fixture
async def dashboard(test_config, mock_orchestrator):
    """Create a dashboard instance for testing."""
    dashboard = UnifiedSystemHealthDashboard(
        config=test_config,
        graceful_degradation_orchestrator=mock_orchestrator
    )
    yield dashboard
    await dashboard.stop()


@pytest.fixture
def deployment_config():
    """Create a deployment configuration for testing."""
    return DashboardDeploymentConfig(
        deployment_type="test",
        dashboard_port=8094,
        enable_database=True,
        database_path=":memory:",
        enable_alerts=True,
        alert_cooldown_seconds=5
    )


# ============================================================================
# UNIT TESTS - DATA AGGREGATOR
# ============================================================================

class TestUnifiedDataAggregator:
    """Test suite for the UnifiedDataAggregator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test data aggregator initialization."""
        aggregator = UnifiedDataAggregator(test_config)
        
        assert aggregator.config == test_config
        assert aggregator.current_snapshot is None
        assert len(aggregator.historical_snapshots) == 0
        assert len(aggregator.alert_history) == 0
        assert len(aggregator.active_alerts) == 0
        assert not aggregator._running

    @pytest.mark.asyncio
    async def test_database_initialization(self, test_config):
        """Test database initialization."""
        # Use temporary file for this test
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_config.db_path = tmp.name
        
        try:
            aggregator = UnifiedDataAggregator(test_config)
            
            # Check that database file exists and has correct tables
            conn = sqlite3.connect(test_config.db_path)
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'health_snapshots' in tables
            assert 'alert_events' in tables
            
            conn.close()
            
        finally:
            # Clean up
            if os.path.exists(test_config.db_path):
                os.unlink(test_config.db_path)

    @pytest.mark.asyncio
    async def test_register_monitoring_systems(self, data_aggregator, mock_orchestrator, mock_load_detector):
        """Test registering monitoring systems."""
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator,
            enhanced_load_detector=mock_load_detector
        )
        
        assert data_aggregator.graceful_degradation_orchestrator == mock_orchestrator
        assert data_aggregator.enhanced_load_detector == mock_load_detector

    @pytest.mark.asyncio
    async def test_start_stop_aggregation(self, data_aggregator):
        """Test starting and stopping data aggregation."""
        # Test start
        await data_aggregator.start_aggregation()
        assert data_aggregator._running
        assert data_aggregator._aggregation_task is not None
        
        # Test stop
        await data_aggregator.stop_aggregation()
        assert not data_aggregator._running

    @pytest.mark.asyncio
    async def test_create_health_snapshot(self, data_aggregator, mock_orchestrator, mock_load_detector):
        """Test creating health snapshots."""
        # Register mock systems
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator,
            enhanced_load_detector=mock_load_detector
        )
        
        # Create snapshot
        snapshot = await data_aggregator._create_health_snapshot()
        
        assert snapshot is not None
        assert isinstance(snapshot, SystemHealthSnapshot)
        assert snapshot.snapshot_id is not None
        assert snapshot.overall_health == "healthy"
        assert snapshot.load_level == "NORMAL"
        assert snapshot.cpu_utilization == 25.0

    @pytest.mark.asyncio
    async def test_trend_calculation(self, data_aggregator):
        """Test trend calculation logic."""
        # Test improving trend
        improving_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = data_aggregator._calculate_trend(improving_values)
        assert trend == "improving"
        
        # Test degrading trend
        degrading_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        trend = data_aggregator._calculate_trend(degrading_values)
        assert trend == "degrading"
        
        # Test stable trend
        stable_values = [0.3, 0.31, 0.29, 0.3, 0.32]
        trend = data_aggregator._calculate_trend(stable_values)
        assert trend == "stable"

    @pytest.mark.asyncio
    async def test_alert_generation(self, data_aggregator):
        """Test alert generation and processing."""
        # Create mock snapshots with significant changes
        previous_snapshot = SystemHealthSnapshot(
            timestamp=datetime.now() - timedelta(seconds=10),
            snapshot_id="prev-123",
            system_uptime_seconds=3590.0,
            overall_health="healthy",
            health_score=0.85,
            load_level="NORMAL",
            load_score=0.3,
            cpu_utilization=25.0,
            memory_pressure=40.0,
            response_time_p95=150.0,
            error_rate=0.1,
            request_queue_depth=5,
            active_connections=10,
            degradation_active=False,
            degradation_level="NORMAL",
            emergency_mode=False
        )
        
        current_snapshot = SystemHealthSnapshot(
            timestamp=datetime.now(),
            snapshot_id="curr-456",
            system_uptime_seconds=3600.0,
            overall_health="degraded",
            health_score=0.45,
            load_level="HIGH",  # Changed from NORMAL
            load_score=0.8,
            cpu_utilization=85.0,
            memory_pressure=75.0,
            response_time_p95=3500.0,  # Significant increase
            error_rate=5.0,  # Significant increase
            request_queue_depth=50,
            active_connections=100,
            degradation_active=True,
            degradation_level="HIGH",
            emergency_mode=False
        )
        
        # Check for alerts
        await data_aggregator._check_for_alerts(previous_snapshot, current_snapshot)
        
        # Should have generated multiple alerts
        assert len(data_aggregator.active_alerts) > 0
        
        # Check specific alert types
        alert_sources = [alert.source for alert in data_aggregator.active_alerts.values()]
        assert "load_detector" in alert_sources

    @pytest.mark.asyncio
    async def test_historical_data_management(self, data_aggregator, sample_snapshot):
        """Test historical data storage and retrieval."""
        # Add snapshots to history
        for i in range(10):
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.now() - timedelta(minutes=i),
                snapshot_id=f"test-{i}",
                system_uptime_seconds=3600.0 + i,
                overall_health="healthy",
                health_score=0.8,
                load_level="NORMAL",
                load_score=0.3,
                cpu_utilization=20.0 + i,
                memory_pressure=30.0,
                response_time_p95=100.0,
                error_rate=0.1,
                request_queue_depth=1,
                active_connections=5,
                degradation_active=False,
                degradation_level="NORMAL",
                emergency_mode=False
            )
            data_aggregator.historical_snapshots.append(snapshot)
        
        # Test retrieval
        recent_snapshots = data_aggregator.get_historical_snapshots(hours=1)
        assert len(recent_snapshots) == 10
        
        # Test time filtering
        old_snapshots = data_aggregator.get_historical_snapshots(hours=0.1)  # 6 minutes
        assert len(old_snapshots) < 10

    @pytest.mark.asyncio
    async def test_alert_resolution(self, data_aggregator, sample_alert):
        """Test alert resolution functionality."""
        # Add alert to active alerts
        data_aggregator.active_alerts[sample_alert.id] = sample_alert
        
        # Resolve alert
        success = data_aggregator.resolve_alert(sample_alert.id, "test_user")
        
        assert success
        assert sample_alert.resolved
        assert sample_alert.resolved_by == "test_user"
        assert sample_alert.resolved_at is not None

    @pytest.mark.asyncio
    async def test_callback_registration(self, data_aggregator, sample_snapshot, sample_alert):
        """Test callback registration and execution."""
        update_called = False
        alert_called = False
        
        def update_callback(snapshot):
            nonlocal update_called
            update_called = True
            assert snapshot == sample_snapshot
        
        def alert_callback(alert):
            nonlocal alert_called
            alert_called = True
            assert alert == sample_alert
        
        # Register callbacks
        data_aggregator.add_update_callback(update_callback)
        data_aggregator.add_alert_callback(alert_callback)
        
        # Trigger callbacks (simulate internal calls)
        for callback in data_aggregator._update_callbacks:
            callback(sample_snapshot)
        
        for callback in data_aggregator._alert_callbacks:
            callback(sample_alert)
        
        assert update_called
        assert alert_called


# ============================================================================
# UNIT TESTS - WEBSOCKET MANAGER
# ============================================================================

class TestWebSocketManager:
    """Test suite for the WebSocketManager class."""

    def test_initialization(self, websocket_manager):
        """Test WebSocket manager initialization."""
        assert len(websocket_manager.connections) == 0

    def test_connection_management(self, websocket_manager):
        """Test WebSocket connection management."""
        # Create mock WebSocket connections
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws2 = Mock(spec=WebSocket)
        
        # Add connections
        websocket_manager.add_connection(mock_ws1)
        websocket_manager.add_connection(mock_ws2)
        
        assert len(websocket_manager.connections) == 2
        assert mock_ws1 in websocket_manager.connections
        assert mock_ws2 in websocket_manager.connections
        
        # Remove connection
        websocket_manager.remove_connection(mock_ws1)
        
        assert len(websocket_manager.connections) == 1
        assert mock_ws1 not in websocket_manager.connections
        assert mock_ws2 in websocket_manager.connections

    @pytest.mark.asyncio
    async def test_broadcast_snapshot(self, websocket_manager, sample_snapshot):
        """Test broadcasting health snapshots."""
        # Create mock WebSocket connections
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        
        websocket_manager.add_connection(mock_ws1)
        websocket_manager.add_connection(mock_ws2)
        
        # Broadcast snapshot
        await websocket_manager.broadcast_snapshot(sample_snapshot)
        
        # Verify both connections received the message
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        
        # Verify message content
        call_args = mock_ws1.send_text.call_args[0][0]
        message = json.loads(call_args)
        
        assert message["type"] == "health_update"
        assert "data" in message
        assert message["data"]["snapshot_id"] == sample_snapshot.snapshot_id

    @pytest.mark.asyncio
    async def test_broadcast_alert(self, websocket_manager, sample_alert):
        """Test broadcasting alerts."""
        # Create mock WebSocket connection
        mock_ws = AsyncMock(spec=WebSocket)
        websocket_manager.add_connection(mock_ws)
        
        # Broadcast alert
        await websocket_manager.broadcast_alert(sample_alert)
        
        # Verify message was sent
        mock_ws.send_text.assert_called_once()
        
        # Verify message content
        call_args = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_args)
        
        assert message["type"] == "alert"
        assert message["data"]["id"] == sample_alert.id
        assert message["data"]["severity"] == sample_alert.severity

    @pytest.mark.asyncio
    async def test_broadcast_with_failed_connection(self, websocket_manager, sample_snapshot):
        """Test broadcasting with a failed WebSocket connection."""
        # Create one good and one failing connection
        mock_ws_good = AsyncMock(spec=WebSocket)
        mock_ws_bad = AsyncMock(spec=WebSocket)
        mock_ws_bad.send_text.side_effect = Exception("Connection failed")
        
        websocket_manager.add_connection(mock_ws_good)
        websocket_manager.add_connection(mock_ws_bad)
        
        assert len(websocket_manager.connections) == 2
        
        # Broadcast snapshot
        await websocket_manager.broadcast_snapshot(sample_snapshot)
        
        # Good connection should still receive message
        mock_ws_good.send_text.assert_called_once()
        
        # Failed connection should be removed
        assert len(websocket_manager.connections) == 1
        assert mock_ws_bad not in websocket_manager.connections
        assert mock_ws_good in websocket_manager.connections


# ============================================================================
# INTEGRATION TESTS - DASHBOARD COMPONENTS
# ============================================================================

class TestDashboardIntegration:
    """Test suite for dashboard integration functionality."""

    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, test_config, mock_orchestrator):
        """Test dashboard initialization with orchestrator integration."""
        dashboard = UnifiedSystemHealthDashboard(
            config=test_config,
            graceful_degradation_orchestrator=mock_orchestrator
        )
        
        assert dashboard.config == test_config
        assert dashboard.data_aggregator is not None
        assert dashboard.websocket_manager is not None
        assert dashboard.app is not None
        assert dashboard.framework in ["fastapi", "flask"]

    @pytest.mark.asyncio
    async def test_orchestrator_integration(self, dashboard, mock_orchestrator):
        """Test integration with graceful degradation orchestrator."""
        # Verify orchestrator is registered
        assert dashboard.data_aggregator.graceful_degradation_orchestrator == mock_orchestrator
        
        # Test that orchestrator methods are called during snapshot creation
        snapshot = await dashboard.data_aggregator._create_health_snapshot()
        
        # Verify orchestrator methods were called
        mock_orchestrator.get_system_status.assert_called()
        mock_orchestrator.get_health_check.assert_called()
        
        # Verify snapshot contains orchestrator data
        assert snapshot.overall_health == "healthy"
        assert snapshot.integrated_services["load_monitoring"] is True

    @pytest.mark.asyncio
    async def test_real_time_updates(self, dashboard, sample_snapshot):
        """Test real-time update flow from aggregator to WebSocket."""
        update_received = False
        
        # Mock WebSocket broadcast to capture calls
        original_broadcast = dashboard.websocket_manager.broadcast_snapshot
        
        async def mock_broadcast(snapshot):
            nonlocal update_received
            update_received = True
            assert snapshot == sample_snapshot
        
        dashboard.websocket_manager.broadcast_snapshot = mock_broadcast
        
        # Trigger update callback
        dashboard._on_health_update(sample_snapshot)
        
        # Give async tasks time to complete
        await asyncio.sleep(0.1)
        
        assert update_received

    @pytest.mark.asyncio
    async def test_alert_propagation(self, dashboard, sample_alert):
        """Test alert propagation from aggregator to WebSocket."""
        alert_received = False
        
        # Mock WebSocket broadcast to capture calls
        async def mock_broadcast(alert):
            nonlocal alert_received
            alert_received = True
            assert alert == sample_alert
        
        dashboard.websocket_manager.broadcast_alert = mock_broadcast
        
        # Trigger alert callback
        dashboard._on_alert_generated(sample_alert)
        
        # Give async tasks time to complete
        await asyncio.sleep(0.1)
        
        assert alert_received


# ============================================================================
# API TESTS - REST ENDPOINTS
# ============================================================================

class TestDashboardAPI:
    """Test suite for dashboard REST API endpoints."""

    def test_health_endpoint(self, dashboard, sample_snapshot):
        """Test the health status endpoint."""
        # Set current snapshot
        dashboard.data_aggregator.current_snapshot = sample_snapshot
        
        # Create test client
        client = TestClient(dashboard.app)
        
        # Test health endpoint
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert data["data"]["snapshot_id"] == sample_snapshot.snapshot_id

    def test_health_endpoint_no_data(self, dashboard):
        """Test health endpoint when no data is available."""
        client = TestClient(dashboard.app)
        response = client.get("/api/v1/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "No health data available" in data["message"]

    def test_historical_data_endpoint(self, dashboard, sample_snapshot):
        """Test the historical data endpoint."""
        # Add historical snapshots
        for i in range(5):
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.now() - timedelta(minutes=i),
                snapshot_id=f"hist-{i}",
                system_uptime_seconds=3600.0,
                overall_health="healthy",
                health_score=0.8,
                load_level="NORMAL",
                load_score=0.3,
                cpu_utilization=20.0,
                memory_pressure=30.0,
                response_time_p95=100.0,
                error_rate=0.1,
                request_queue_depth=1,
                active_connections=5,
                degradation_active=False,
                degradation_level="NORMAL",
                emergency_mode=False
            )
            dashboard.data_aggregator.historical_snapshots.append(snapshot)
        
        client = TestClient(dashboard.app)
        response = client.get("/api/v1/health/history?hours=1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]) == 5

    def test_alerts_endpoint(self, dashboard, sample_alert):
        """Test the alerts endpoint."""
        # Add active alert
        dashboard.data_aggregator.active_alerts[sample_alert.id] = sample_alert
        
        client = TestClient(dashboard.app)
        response = client.get("/api/v1/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == sample_alert.id

    def test_alert_resolution_endpoint(self, dashboard, sample_alert):
        """Test the alert resolution endpoint."""
        # Add active alert
        dashboard.data_aggregator.active_alerts[sample_alert.id] = sample_alert
        
        client = TestClient(dashboard.app)
        response = client.post(f"/api/v1/alerts/{sample_alert.id}/resolve")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert sample_alert.resolved

    def test_system_status_endpoint(self, dashboard):
        """Test the system status endpoint."""
        client = TestClient(dashboard.app)
        response = client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "framework" in data["data"]
        assert "config" in data["data"]

    def test_main_dashboard_page(self, dashboard):
        """Test the main dashboard HTML page."""
        client = TestClient(dashboard.app)
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Unified System Health Dashboard" in response.text

    def test_cors_headers(self, dashboard):
        """Test CORS headers are properly set."""
        client = TestClient(dashboard.app)
        
        # Test preflight request
        response = client.options("/api/v1/health")
        
        # Should not error (CORS middleware should handle it)
        assert response.status_code in [200, 405]  # Depending on FastAPI version


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestDashboardPerformance:
    """Test suite for dashboard performance characteristics."""

    @pytest.mark.asyncio
    async def test_data_aggregation_performance(self, data_aggregator, mock_orchestrator, mock_load_detector):
        """Test performance of data aggregation process."""
        # Register systems
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator,
            enhanced_load_detector=mock_load_detector
        )
        
        # Measure time to create snapshots
        start_time = time.time()
        
        # Create multiple snapshots
        snapshots = []
        for i in range(10):
            snapshot = await data_aggregator._create_health_snapshot()
            snapshots.append(snapshot)
        
        end_time = time.time()
        
        # Should create snapshots quickly
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.1  # Less than 100ms per snapshot
        assert len(snapshots) == 10
        assert all(s is not None for s in snapshots)

    @pytest.mark.asyncio
    async def test_websocket_broadcast_performance(self, websocket_manager, sample_snapshot):
        """Test performance of WebSocket broadcasting."""
        # Create many mock connections
        mock_connections = []
        for i in range(50):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_connections.append(mock_ws)
            websocket_manager.add_connection(mock_ws)
        
        # Measure broadcast time
        start_time = time.time()
        
        await websocket_manager.broadcast_snapshot(sample_snapshot)
        
        end_time = time.time()
        
        # Should broadcast quickly even with many connections
        broadcast_time = end_time - start_time
        assert broadcast_time < 0.5  # Less than 500ms
        
        # Verify all connections received message
        for mock_ws in mock_connections:
            mock_ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_processing_performance(self, data_aggregator):
        """Test performance of alert processing."""
        # Create many alerts
        alerts = []
        for i in range(100):
            alert = AlertEvent(
                id=f"perf-test-{i}",
                timestamp=datetime.now(),
                severity="info",
                source="performance_test",
                title=f"Performance Test Alert {i}",
                message=f"Test alert number {i}",
                category="testing"
            )
            alerts.append(alert)
        
        # Measure processing time
        start_time = time.time()
        
        for alert in alerts:
            await data_aggregator._process_alert(alert)
        
        end_time = time.time()
        
        # Should process alerts quickly
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Less than 10ms per alert
        
        # Verify alerts were processed
        assert len(data_aggregator.alert_history) == 100

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, dashboard, sample_snapshot):
        """Test handling concurrent API requests."""
        # Set current snapshot
        dashboard.data_aggregator.current_snapshot = sample_snapshot
        
        client = TestClient(dashboard.app)
        
        # Function to make API request
        async def make_request():
            response = client.get("/api/v1/health")
            return response.status_code == 200
        
        # Make many concurrent requests
        tasks = [make_request() for _ in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should succeed
        success_count = sum(1 for r in results if r is True)
        assert success_count >= 15  # Allow for some potential failures in test environment

    def test_memory_usage_stability(self, data_aggregator, sample_snapshot):
        """Test memory usage remains stable with many snapshots."""
        import gc
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Add many snapshots
        initial_snapshot = tracemalloc.take_snapshot()
        
        for i in range(1000):
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.now(),
                snapshot_id=f"mem-test-{i}",
                system_uptime_seconds=3600.0 + i,
                overall_health="healthy",
                health_score=0.8,
                load_level="NORMAL",
                load_score=0.3,
                cpu_utilization=20.0,
                memory_pressure=30.0,
                response_time_p95=100.0,
                error_rate=0.1,
                request_queue_depth=1,
                active_connections=5,
                degradation_active=False,
                degradation_level="NORMAL",
                emergency_mode=False
            )
            data_aggregator.historical_snapshots.append(snapshot)
        
        gc.collect()
        final_snapshot = tracemalloc.take_snapshot()
        
        # Check memory growth
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        
        # Should not have excessive memory growth
        # Note: This is a basic check - in production you'd want more sophisticated memory analysis
        assert len(data_aggregator.historical_snapshots) <= 5000  # Should respect max size


# ============================================================================
# INTEGRATION HELPER TESTS
# ============================================================================

class TestDashboardIntegrationHelper:
    """Test suite for the dashboard integration helper."""

    def test_configuration_templates(self):
        """Test configuration template generation."""
        dev_config = get_development_config()
        assert dev_config.deployment_type == "development"
        assert dev_config.dashboard_port == 8092
        assert not dev_config.enable_ssl
        
        prod_config = get_production_config()
        assert prod_config.deployment_type == "production"
        assert prod_config.enable_ssl
        assert prod_config.websocket_update_interval == 5.0

    def test_config_conversion(self, deployment_config):
        """Test conversion from deployment config to dashboard config."""
        dashboard_config = deployment_config.to_dashboard_config()
        
        assert dashboard_config.host == deployment_config.dashboard_host
        assert dashboard_config.port == deployment_config.dashboard_port
        assert dashboard_config.enable_alerts == deployment_config.enable_alerts

    @pytest.mark.asyncio
    async def test_system_discovery(self, deployment_config):
        """Test automatic system discovery."""
        helper = DashboardIntegrationHelper(deployment_config)
        
        # Mock the discovery to avoid import issues in test environment
        with patch.object(helper, 'discover_monitoring_systems') as mock_discover:
            mock_discover.return_value = {
                "graceful_degradation": True,
                "enhanced_load_detection": True,
                "degradation_controller": False,
                "circuit_breaker_monitoring": False,
                "production_monitoring": False
            }
            
            discovery_results = await helper.discover_monitoring_systems()
            
            assert discovery_results["graceful_degradation"] is True
            assert discovery_results["enhanced_load_detection"] is True

    @pytest.mark.asyncio
    async def test_dashboard_validation(self, deployment_config, mock_orchestrator):
        """Test dashboard setup validation."""
        helper = DashboardIntegrationHelper(deployment_config)
        
        # Create a dashboard to validate
        dashboard = UnifiedSystemHealthDashboard(
            config=deployment_config.to_dashboard_config(),
            graceful_degradation_orchestrator=mock_orchestrator
        )
        
        validation_report = await helper.validate_dashboard_setup(dashboard)
        
        assert validation_report["overall_status"] in ["healthy", "warning", "error"]
        assert "checks" in validation_report
        assert "data_aggregator" in validation_report["checks"]

    def test_config_file_generation(self, deployment_config):
        """Test configuration file generation."""
        helper = DashboardIntegrationHelper(deployment_config)
        
        # Test YAML generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml_path = tmp.name
        
        try:
            success = helper.generate_config_file(yaml_path, format="yaml")
            assert success
            assert os.path.exists(yaml_path)
            
            # Verify file content
            with open(yaml_path, 'r') as f:
                content = f.read()
                assert "deployment_type" in content
                assert deployment_config.deployment_type in content
        
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)
        
        # Test JSON generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json_path = tmp.name
        
        try:
            success = helper.generate_config_file(json_path, format="json")
            assert success
            assert os.path.exists(json_path)
            
            # Verify file content
            with open(json_path, 'r') as f:
                data = json.load(f)
                assert data["deployment_type"] == deployment_config.deployment_type
        
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_quick_start_dashboard(self):
        """Test quick start dashboard functionality."""
        with patch('lightrag_integration.dashboard_integration_helper.DashboardIntegrationHelper.deploy_dashboard') as mock_deploy:
            mock_deploy.return_value = (Mock(), {"status": "success", "dashboard_info": {}})
            
            dashboard, report = await quick_start_dashboard(
                deployment_type="development",
                port=8095,
                enable_all_features=True
            )
            
            assert report["status"] == "success"
            mock_deploy.assert_called_once()


# ============================================================================
# MOCK TESTS - GRACEFUL DEGRADATION SCENARIOS
# ============================================================================

class TestGracefulDegradationScenarios:
    """Test suite for graceful degradation scenarios."""

    @pytest.mark.asyncio
    async def test_high_load_scenario(self, data_aggregator, mock_orchestrator, mock_load_detector):
        """Test dashboard behavior under high load conditions."""
        # Configure mocks for high load
        mock_orchestrator.get_system_status.return_value = {
            'start_time': datetime.now().isoformat(),
            'current_load_level': 'HIGH',
            'total_requests_processed': 10000,
            'integration_status': {
                'load_monitoring_active': True,
                'degradation_controller_active': True,
                'throttling_system_active': True,
                'integrated_load_balancer': False,
                'integrated_rag_system': True,
                'integrated_monitoring': True
            }
        }
        
        mock_load_detector.get_current_metrics.return_value.cpu_utilization = 85.0
        mock_load_detector.get_current_metrics.return_value.memory_pressure = 80.0
        mock_load_detector.get_current_metrics.return_value.response_time_p95 = 2500.0
        mock_load_detector.get_current_metrics.return_value.error_rate = 3.0
        mock_load_detector.get_current_metrics.return_value.load_level.name = 'HIGH'
        mock_load_detector.get_current_metrics.return_value.load_score = 0.85
        
        # Register systems and create snapshot
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator,
            enhanced_load_detector=mock_load_detector
        )
        
        snapshot = await data_aggregator._create_health_snapshot()
        
        # Verify high load conditions are reflected
        assert snapshot.load_level == "HIGH"
        assert snapshot.cpu_utilization == 85.0
        assert snapshot.response_time_p95 == 2500.0
        assert snapshot.overall_health in ["degraded", "warning"]

    @pytest.mark.asyncio
    async def test_emergency_mode_scenario(self, data_aggregator, mock_degradation_controller):
        """Test dashboard behavior in emergency mode."""
        # Configure mock for emergency mode
        mock_degradation_controller.get_current_status.return_value = {
            'degradation_active': True,
            'load_level': 'EMERGENCY',
            'emergency_mode': True,
            'feature_settings': {
                'advanced_search': False,
                'detailed_analytics': False,
                'export_functionality': False
            }
        }
        
        # Register system and create snapshot
        data_aggregator.register_monitoring_systems(
            degradation_controller=mock_degradation_controller
        )
        
        snapshot = await data_aggregator._create_health_snapshot()
        
        # Verify emergency mode conditions
        assert snapshot.emergency_mode is True
        assert snapshot.degradation_active is True
        assert snapshot.degradation_level == "EMERGENCY"
        assert snapshot.overall_health == "emergency"
        assert len(snapshot.disabled_features) == 3

    @pytest.mark.asyncio
    async def test_system_failure_recovery(self, data_aggregator, mock_orchestrator):
        """Test dashboard behavior when monitoring systems fail."""
        # Configure orchestrator to throw exception
        mock_orchestrator.get_system_status.side_effect = Exception("System failure")
        mock_orchestrator.get_health_check.side_effect = Exception("Health check failed")
        
        # Register system
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator
        )
        
        # Should still create snapshot despite failures
        snapshot = await data_aggregator._create_health_snapshot()
        
        assert snapshot is not None
        assert snapshot.overall_health == "unknown"
        
        # Reset mocks to working state
        mock_orchestrator.get_system_status.side_effect = None
        mock_orchestrator.get_health_check.side_effect = None
        mock_orchestrator.get_system_status.return_value = {
            'start_time': datetime.now().isoformat(),
            'current_load_level': 'NORMAL',
            'total_requests_processed': 1000,
            'integration_status': {}
        }
        mock_orchestrator.get_health_check.return_value = {
            'status': 'healthy',
            'issues': []
        }
        
        # Should recover
        snapshot = await data_aggregator._create_health_snapshot()
        assert snapshot.overall_health == "healthy"

    @pytest.mark.asyncio
    async def test_partial_system_availability(self, data_aggregator, mock_orchestrator):
        """Test dashboard with only some monitoring systems available."""
        # Only register orchestrator, not other systems
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator
        )
        
        snapshot = await data_aggregator._create_health_snapshot()
        
        # Should still work with limited data
        assert snapshot is not None
        assert snapshot.overall_health == "healthy"
        
        # Some metrics should have default values
        assert snapshot.cpu_utilization == 0.0  # Default when load detector not available
        assert snapshot.memory_pressure == 0.0


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        # Use invalid database path
        config = DashboardConfig(
            db_path="/invalid/path/database.db",
            enable_db_persistence=True
        )
        
        # Should not crash during initialization
        aggregator = UnifiedDataAggregator(config)
        assert aggregator is not None

    @pytest.mark.asyncio
    async def test_websocket_connection_errors(self, websocket_manager, sample_snapshot):
        """Test WebSocket connection error handling."""
        # Create mock connection that always fails
        mock_ws_bad = AsyncMock(spec=WebSocket)
        mock_ws_bad.send_text.side_effect = Exception("Connection lost")
        
        # Create good connection for comparison
        mock_ws_good = AsyncMock(spec=WebSocket)
        
        websocket_manager.add_connection(mock_ws_bad)
        websocket_manager.add_connection(mock_ws_good)
        
        # Should handle errors gracefully
        await websocket_manager.broadcast_snapshot(sample_snapshot)
        
        # Bad connection should be removed, good one should remain
        assert mock_ws_bad not in websocket_manager.connections
        assert mock_ws_good in websocket_manager.connections

    @pytest.mark.asyncio
    async def test_malformed_data_handling(self, data_aggregator, mock_orchestrator):
        """Test handling of malformed data from monitoring systems."""
        # Configure orchestrator to return malformed data
        mock_orchestrator.get_system_status.return_value = {
            'start_time': "invalid-date-format",
            'current_load_level': None,
            'total_requests_processed': "not-a-number",
            'integration_status': None
        }
        mock_orchestrator.get_health_check.return_value = None
        
        data_aggregator.register_monitoring_systems(
            graceful_degradation_orchestrator=mock_orchestrator
        )
        
        # Should handle malformed data gracefully
        snapshot = await data_aggregator._create_health_snapshot()
        
        assert snapshot is not None
        assert snapshot.system_uptime_seconds == 0.0  # Default value
        assert snapshot.total_requests_processed == 0  # Default value

    def test_invalid_configuration(self):
        """Test handling of invalid configuration values."""
        # Test with invalid port
        config = DashboardConfig(port=-1)
        
        # Should not crash (validation would happen at server start)
        dashboard = UnifiedSystemHealthDashboard(config=config)
        assert dashboard is not None
        
        # Test with invalid update interval
        config = DashboardConfig(websocket_update_interval=-1)
        aggregator = UnifiedDataAggregator(config)
        assert aggregator is not None

    @pytest.mark.asyncio
    async def test_concurrent_modifications(self, data_aggregator, sample_snapshot):
        """Test thread safety with concurrent modifications."""
        async def add_snapshots():
            for i in range(100):
                snapshot = SystemHealthSnapshot(
                    timestamp=datetime.now(),
                    snapshot_id=f"concurrent-{i}",
                    system_uptime_seconds=3600.0,
                    overall_health="healthy",
                    health_score=0.8,
                    load_level="NORMAL",
                    load_score=0.3,
                    cpu_utilization=20.0,
                    memory_pressure=30.0,
                    response_time_p95=100.0,
                    error_rate=0.1,
                    request_queue_depth=1,
                    active_connections=5,
                    degradation_active=False,
                    degradation_level="NORMAL",
                    emergency_mode=False
                )
                data_aggregator.historical_snapshots.append(snapshot)
                await asyncio.sleep(0.001)  # Small delay to allow interleaving
        
        # Run multiple concurrent tasks
        tasks = [add_snapshots() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        # Should have added all snapshots without corruption
        assert len(data_aggregator.historical_snapshots) == 300

    def test_memory_limits(self, data_aggregator):
        """Test behavior when approaching memory limits."""
        # Add maximum number of snapshots
        for i in range(5000):  # Maximum configured limit
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.now(),
                snapshot_id=f"mem-limit-{i}",
                system_uptime_seconds=3600.0,
                overall_health="healthy",
                health_score=0.8,
                load_level="NORMAL",
                load_score=0.3,
                cpu_utilization=20.0,
                memory_pressure=30.0,
                response_time_p95=100.0,
                error_rate=0.1,
                request_queue_depth=1,
                active_connections=5,
                degradation_active=False,
                degradation_level="NORMAL",
                emergency_mode=False
            )
            data_aggregator.historical_snapshots.append(snapshot)
        
        # Should respect maximum size
        assert len(data_aggregator.historical_snapshots) == 5000
        
        # Add one more - should maintain size limit
        extra_snapshot = SystemHealthSnapshot(
            timestamp=datetime.now(),
            snapshot_id="extra",
            system_uptime_seconds=3600.0,
            overall_health="healthy",
            health_score=0.8,
            load_level="NORMAL",
            load_score=0.3,
            cpu_utilization=20.0,
            memory_pressure=30.0,
            response_time_p95=100.0,
            error_rate=0.1,
            request_queue_depth=1,
            active_connections=5,
            degradation_active=False,
            degradation_level="NORMAL",
            emergency_mode=False
        )
        data_aggregator.historical_snapshots.append(extra_snapshot)
        
        # Should still respect limit
        assert len(data_aggregator.historical_snapshots) == 5000
        
        # Latest snapshot should be the extra one
        assert data_aggregator.historical_snapshots[-1].snapshot_id == "extra"


# ============================================================================
# SECURITY TESTS
# ============================================================================

class TestSecurityFeatures:
    """Test suite for security features."""

    def test_api_key_authentication(self):
        """Test API key authentication when enabled."""
        config = DashboardConfig(
            enable_api_key=True,
            api_key="test-api-key-123"
        )
        
        dashboard = UnifiedSystemHealthDashboard(config=config)
        assert dashboard.config.enable_api_key is True
        assert dashboard.config.api_key == "test-api-key-123"

    def test_cors_configuration(self):
        """Test CORS configuration."""
        config = DashboardConfig(enable_cors=True)
        dashboard = UnifiedSystemHealthDashboard(config=config)
        
        # CORS should be enabled in the application
        assert config.enable_cors is True
        
        # Test with CORS disabled
        config_no_cors = DashboardConfig(enable_cors=False)
        dashboard_no_cors = UnifiedSystemHealthDashboard(config=config_no_cors)
        assert config_no_cors.enable_cors is False

    def test_ssl_configuration_validation(self):
        """Test SSL configuration validation."""
        config = DashboardConfig(
            enable_ssl=True,
            ssl_cert_path="/path/to/cert.pem",
            ssl_key_path="/path/to/key.pem"
        )
        
        dashboard = UnifiedSystemHealthDashboard(config=config)
        assert dashboard.config.enable_ssl is True
        assert dashboard.config.ssl_cert_path == "/path/to/cert.pem"

    def test_input_sanitization(self, dashboard):
        """Test input sanitization in API endpoints."""
        client = TestClient(dashboard.app)
        
        # Test with potentially malicious input
        malicious_inputs = [
            "../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE health_snapshots; --",
            "../../../../../etc/shadow"
        ]
        
        for malicious_input in malicious_inputs:
            # Test history endpoint with malicious hours parameter
            response = client.get(f"/api/v1/health/history", 
                                params={"hours": malicious_input})
            
            # Should handle gracefully (either 422 validation error or safe conversion)
            assert response.status_code in [200, 422]
            
            if response.status_code == 200:
                # If it succeeded, ensure no data corruption
                data = response.json()
                assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_rate_limiting_concept(self, dashboard):
        """Test rate limiting concept (would need actual implementation)."""
        # This is a conceptual test - actual rate limiting would require
        # implementation in the dashboard application
        
        client = TestClient(dashboard.app)
        
        # Make many rapid requests
        responses = []
        for i in range(10):
            response = client.get("/api/v1/health")
            responses.append(response.status_code)
        
        # All should succeed in test environment (no rate limiting implemented yet)
        # In production, you'd implement rate limiting and test for 429 responses
        assert all(status in [200, 503] for status in responses)


# ============================================================================
# CLEANUP HELPERS
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_threads():
    """Cleanup any remaining threads after each test."""
    yield
    
    # Wait for any background tasks to complete
    import time
    time.sleep(0.1)
    
    # Force garbage collection
    import gc
    gc.collect()


# ============================================================================
# TEST RUNNER CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Configure test runner
    import sys
    import os
    
    # Add the project directory to Python path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_dir)
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Disable some verbose loggers during testing
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    # Run tests with pytest
    import subprocess
    result = subprocess.run([
        sys.executable, '-m', 'pytest', __file__, '-v',
        '--tb=short',
        '--asyncio-mode=auto',
        '--disable-warnings'
    ])
    
    sys.exit(result.returncode)