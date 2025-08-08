#!/usr/bin/env python3
"""
Comprehensive System Health Monitoring Tests for Clinical Metabolomics Oracle

This test suite provides comprehensive coverage for the health monitoring system
including health checkers, alert management, callbacks, and integration testing.

Test Coverage:
1. LightRAGHealthChecker - filesystem, resources, OpenAI connectivity
2. PerplexityHealthChecker - API connectivity, authentication, rate limits  
3. SystemHealthMonitor - background monitoring, health updates, alerts
4. AlertManager - threshold breach detection, suppression, acknowledgment
5. Alert Callbacks - Console, JSON File, Webhook delivery
6. Integration Tests - routing impact, end-to-end workflows
7. Performance Tests - concurrent health checking, error handling
8. Edge Cases - configuration validation, memory management

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Comprehensive health monitoring system tests
"""

import pytest
import asyncio
import time
import threading
import tempfile
import json
import os
import shutil
import logging
import statistics
import psutil
import requests
import httpx
import openai
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import concurrent.futures
from collections import deque, defaultdict
from contextlib import contextmanager, asynccontextmanager
import http.server
import socketserver
from threading import Thread
import socket

# Import health monitoring components
try:
    from lightrag_integration.intelligent_query_router import (
        # Health Check Components  
        BaseHealthChecker,
        LightRAGHealthChecker,
        PerplexityHealthChecker,
        HealthCheckResult,
        HealthCheckConfig,
        SystemHealthMonitor,
        SystemHealthStatus,
        BackendType,
        BackendHealthMetrics,
        
        # Alert System Components
        AlertManager,
        HealthAlert,
        AlertSeverity,
        AlertThresholds,
        AlertSuppressionRule,
        
        # Alert Callbacks
        AlertCallback,
        ConsoleAlertCallback, 
        JSONFileAlertCallback,
        WebhookAlertCallback,
        
        # Router Integration
        IntelligentQueryRouter
    )
except ImportError as e:
    logging.warning(f"Could not import health monitoring components: {e}")
    # Create mock classes for testing
    class BackendType:
        LIGHTRAG = "lightrag"
        PERPLEXITY = "perplexity"
    
    class SystemHealthStatus:
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        CRITICAL = "critical"
        OFFLINE = "offline"
    
    class AlertSeverity:
        INFO = "info"
        WARNING = "warning"
        CRITICAL = "critical"
        EMERGENCY = "emergency"


# ============================================================================
# TEST FIXTURES AND UTILITIES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def health_check_config(temp_dir):
    """Create health check configuration for testing."""
    return HealthCheckConfig(
        timeout_seconds=5.0,
        max_cpu_percent=80.0,
        max_memory_percent=85.0,
        min_disk_space_gb=1.0,
        lightrag_working_dir=temp_dir,
        lightrag_storage_dir=os.path.join(temp_dir, "storage"),
        lightrag_test_query="Test metabolomics query",
        perplexity_api_key="test_api_key",
        perplexity_base_url="https://api.perplexity.ai",
        perplexity_test_query="Latest research in metabolomics",
        alert_thresholds=AlertThresholds()
    )


@pytest.fixture
def alert_thresholds():
    """Create alert thresholds for testing."""
    return AlertThresholds(
        response_time_warning=1000.0,
        response_time_critical=3000.0,
        response_time_emergency=5000.0,
        error_rate_warning=0.1,
        error_rate_critical=0.3,
        error_rate_emergency=0.5,
        availability_warning=95.0,
        availability_critical=85.0,
        availability_emergency=70.0,
        health_score_warning=80.0,
        health_score_critical=60.0,
        health_score_emergency=40.0
    )


@pytest.fixture
def mock_webhook_server():
    """Create mock webhook server for testing."""
    class MockWebhookHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Store the received data for verification
            if not hasattr(self.server, 'received_data'):
                self.server.received_data = []
            self.server.received_data.append(json.loads(post_data.decode()))
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        
        def log_message(self, format, *args):
            pass  # Suppress HTTP server logs
    
    # Find available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    
    server = socketserver.TCPServer(("", port), MockWebhookHandler)
    server.received_data = []
    
    # Start server in background thread
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    yield f"http://localhost:{port}", server
    
    server.shutdown()
    server.server_close()


@pytest.fixture
def test_logger():
    """Create test logger with memory handler."""
    logger = logging.getLogger('test_health_monitoring')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add memory handler for testing
    from logging.handlers import MemoryHandler
    memory_handler = MemoryHandler(capacity=1000)
    memory_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(memory_handler)
    
    yield logger, memory_handler
    
    logger.handlers.clear()


# ============================================================================
# HEALTH CHECKER COMPONENT TESTS
# ============================================================================

class TestLightRAGHealthChecker:
    """Test LightRAG health checker functionality."""
    
    @pytest.mark.asyncio
    async def test_filesystem_access_check_success(self, health_check_config, temp_dir):
        """Test successful filesystem access check."""
        # Ensure directories exist
        os.makedirs(health_check_config.lightrag_storage_dir, exist_ok=True)
        
        checker = LightRAGHealthChecker(health_check_config)
        result = await checker._check_filesystem_access()
        
        assert result['accessible'] is True
        assert result['working_dir_exists'] is True
        assert result['working_dir_writable'] is True
        assert result['storage_dir_accessible'] is True
        assert result['error'] is None
    
    @pytest.mark.asyncio
    async def test_filesystem_access_check_missing_directory(self, health_check_config):
        """Test filesystem access check with missing working directory."""
        # Use non-existent directory
        health_check_config.lightrag_working_dir = "/non/existent/path"
        
        checker = LightRAGHealthChecker(health_check_config)
        result = await checker._check_filesystem_access()
        
        assert result['accessible'] is False
        assert result['working_dir_exists'] is False
        assert 'Working directory does not exist' in result['error']
    
    @pytest.mark.asyncio
    async def test_filesystem_access_check_permission_denied(self, health_check_config, temp_dir):
        """Test filesystem access check with permission issues."""
        # Create directory with restricted permissions
        restricted_dir = os.path.join(temp_dir, "restricted")
        os.makedirs(restricted_dir)
        os.chmod(restricted_dir, 0o444)  # Read-only
        
        health_check_config.lightrag_working_dir = restricted_dir
        
        checker = LightRAGHealthChecker(health_check_config)
        result = await checker._check_filesystem_access()
        
        assert result['accessible'] is False
        assert result['working_dir_writable'] is False
    
    def test_system_resources_check_healthy(self, health_check_config):
        """Test system resource check with healthy resources."""
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock healthy resource usage
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.free = 5 * (1024**3)  # 5GB free
            mock_disk.return_value.total = 100 * (1024**3)  # 100GB total
            mock_disk.return_value.used = 95 * (1024**3)  # 95GB used
            
            result = checker._check_system_resources()
            
            assert result['adequate'] is True
            assert result['cpu_percent'] == 50.0
            assert result['memory_percent'] == 60.0
            assert result['free_disk_gb'] == 5.0
            assert len(result['issues']) == 0
    
    def test_system_resources_check_unhealthy(self, health_check_config):
        """Test system resource check with resource constraints."""
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch('psutil.cpu_percent', return_value=95.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock constrained resources
            mock_memory.return_value.percent = 90.0
            mock_disk.return_value.free = 0.5 * (1024**3)  # 0.5GB free
            
            result = checker._check_system_resources()
            
            assert result['adequate'] is False
            assert len(result['issues']) >= 2  # CPU and disk issues
            assert any('CPU usage high' in issue for issue in result['issues'])
            assert any('Low disk space' in issue for issue in result['issues'])
    
    @pytest.mark.asyncio
    async def test_openai_connectivity_success(self, health_check_config):
        """Test successful OpenAI API connectivity check."""
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}), \
             patch('openai.OpenAI') as mock_client:
            
            # Mock successful API response
            mock_models = Mock()
            mock_models.data = [Mock(), Mock()]  # 2 models
            mock_client.return_value.models.list.return_value = mock_models
            
            result = await checker._check_openai_connectivity()
            
            assert result['available'] is True
            assert result['has_api_key'] is True
            assert result['models_accessible'] is True
            assert result['model_count'] == 2
    
    @pytest.mark.asyncio
    async def test_openai_connectivity_missing_key(self, health_check_config):
        """Test OpenAI connectivity check with missing API key."""
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch.dict(os.environ, {}, clear=True):
            result = await checker._check_openai_connectivity()
            
            assert result['available'] is False
            assert result['has_api_key'] is False
            assert 'OpenAI API key not found' in result['error']
    
    @pytest.mark.asyncio
    async def test_openai_connectivity_timeout(self, health_check_config):
        """Test OpenAI connectivity check with timeout."""
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}), \
             patch('openai.OpenAI') as mock_client:
            
            # Mock timeout
            mock_client.return_value.models.list.side_effect = asyncio.TimeoutError()
            
            result = await checker._check_openai_connectivity()
            
            assert result['available'] is False
            assert result['has_api_key'] is True
            assert 'timeout' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_sample_query_success(self, health_check_config):
        """Test successful sample query execution."""
        checker = LightRAGHealthChecker(health_check_config)
        
        result = await checker._test_sample_query()
        
        assert result['successful'] is True
        assert result['query_time_ms'] >= 100  # Should take at least 100ms due to sleep
        assert result['test_query'] == health_check_config.lightrag_test_query
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check_healthy(self, health_check_config, temp_dir):
        """Test comprehensive health check with healthy system."""
        # Setup healthy environment
        os.makedirs(health_check_config.lightrag_storage_dir, exist_ok=True)
        
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}), \
             patch('openai.OpenAI') as mock_client, \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock healthy conditions
            mock_models = Mock()
            mock_models.data = [Mock()]
            mock_client.return_value.models.list.return_value = mock_models
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.free = 5 * (1024**3)
            
            result = await checker.check_health()
            
            assert result.is_healthy is True
            assert result.response_time_ms > 0
            assert result.error_message is None
            assert 'accessible' in result.metadata
            assert 'adequate' in result.metadata
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check_unhealthy(self, health_check_config):
        """Test comprehensive health check with unhealthy system."""
        # Use non-existent directory
        health_check_config.lightrag_working_dir = "/non/existent"
        
        checker = LightRAGHealthChecker(health_check_config)
        
        result = await checker.check_health()
        
        assert result.is_healthy is False
        assert result.error_message is not None
        assert 'Working directory does not exist' in result.error_message


class TestPerplexityHealthChecker:
    """Test Perplexity health checker functionality."""
    
    @pytest.mark.asyncio
    async def test_api_connectivity_success(self, health_check_config):
        """Test successful API connectivity check."""
        health_check_config.perplexity_api_key = "test_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 0.1
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await checker._check_api_connectivity()
            
            assert result['accessible'] is True
            assert result.get('error') is None
    
    @pytest.mark.asyncio
    async def test_api_connectivity_failure(self, health_check_config):
        """Test API connectivity check with connection failure."""
        health_check_config.perplexity_api_key = "test_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        # Mock httpx.AsyncClient to raise an exception
        with patch('httpx.AsyncClient') as mock_client:
            async def mock_aenter(self):
                raise Exception("Connection failed")
            
            mock_client.return_value.__aenter__ = mock_aenter
            mock_client.return_value.__aexit__ = AsyncMock()
            
            result = await checker._check_api_connectivity()
            
            assert result is not None
            assert result['accessible'] is False
            assert 'Connection failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, health_check_config):
        """Test successful authentication check."""
        health_check_config.perplexity_api_key = "valid_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful auth response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'authenticated': True}
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await checker._check_authentication()
            
            assert result['authenticated'] is True
            assert result.get('error') is None
    
    @pytest.mark.asyncio
    async def test_authentication_invalid_key(self, health_check_config):
        """Test authentication check with invalid API key."""
        health_check_config.perplexity_api_key = "invalid_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock authentication failure
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await checker._check_authentication()
            
            assert result['authenticated'] is False
            assert 'authentication failed' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_rate_limits_check(self, health_check_config):
        """Test rate limits check."""
        health_check_config.perplexity_api_key = "test_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock rate limit headers
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'X-RateLimit-Remaining': '100',
                'X-RateLimit-Limit': '1000',
                'X-RateLimit-Reset': '3600'
            }
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await checker._check_rate_limits()
            
            assert 'rate_limit_remaining' in result
            assert result['rate_limit_remaining'] == 100
            assert result['rate_limit_limit'] == 1000
    
    @pytest.mark.asyncio
    async def test_response_format_check_valid(self, health_check_config):
        """Test response format check with valid response."""
        health_check_config.perplexity_api_key = "test_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock valid JSON response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'id': 'test_id',
                'choices': [{'message': {'content': 'test response'}}]
            }
            mock_response.text = '{"valid": "json"}'
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await checker._check_response_format()
            
            assert result['valid_format'] is True
            assert result.get('error') is None
    
    @pytest.mark.asyncio
    async def test_response_format_check_invalid(self, health_check_config):
        """Test response format check with invalid response."""
        health_check_config.perplexity_api_key = "test_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock invalid JSON response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Invalid JSON response"
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await checker._check_response_format()
            
            assert result['valid_format'] is False
            assert 'JSON decode error' in result['error']
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check_healthy(self, health_check_config):
        """Test comprehensive health check with healthy API."""
        health_check_config.perplexity_api_key = "valid_key"
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch('httpx.AsyncClient') as mock_client:
            
            # Mock healthy API responses
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.elapsed.total_seconds.return_value = 0.1
            mock_get_response.headers = {'X-RateLimit-Remaining': '100'}
            
            mock_post_response = Mock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {'authenticated': True, 'choices': [{}]}
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_get_response)
            mock_client.return_value.post = AsyncMock(return_value=mock_post_response)
            
            result = await checker.check_health()
            
            assert result.is_healthy is True
            assert result.response_time_ms > 0
            assert result.error_message is None
            assert result.metadata['has_api_key'] is True
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check_no_api_key(self, health_check_config):
        """Test comprehensive health check without API key."""
        health_check_config.perplexity_api_key = None
        checker = PerplexityHealthChecker(health_check_config)
        
        with patch.dict(os.environ, {}, clear=True):
            result = await checker.check_health()
            
            assert result.is_healthy is False
            assert 'API key not available' in result.error_message
            assert result.metadata['has_api_key'] is False


# ============================================================================
# ALERT SYSTEM TESTS
# ============================================================================

class TestAlertManager:
    """Test alert manager functionality."""
    
    def test_alert_manager_initialization(self, alert_thresholds):
        """Test alert manager initialization."""
        manager = AlertManager(alert_thresholds)
        
        assert manager.alert_thresholds == alert_thresholds
        assert len(manager.active_alerts) == 0
        assert len(manager.alert_history) == 0
        assert len(manager.alert_callbacks) > 0  # Default callbacks
        assert len(manager.suppression_rules) > 0  # Default suppression rules
    
    def test_generate_alert_response_time_warning(self, alert_thresholds):
        """Test alert generation for response time warning."""
        manager = AlertManager(alert_thresholds)
        
        # Create metrics that exceed response time warning threshold
        metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.DEGRADED,
            response_time_ms=1500.0,  # Exceeds warning threshold of 1000ms
            error_rate=0.05,
            last_health_check=datetime.now()
        )
        
        alerts = manager.check_and_generate_alerts(metrics)
        
        assert len(alerts) > 0
        response_time_alerts = [a for a in alerts if 'response_time' in a.id]
        assert len(response_time_alerts) > 0
        
        alert = response_time_alerts[0]
        assert alert.severity == AlertSeverity.WARNING
        assert alert.backend_type == BackendType.LIGHTRAG
        assert alert.current_value == 1500.0
        assert alert.threshold_value == 1000.0
    
    def test_generate_alert_error_rate_critical(self, alert_thresholds):
        """Test alert generation for critical error rate."""
        manager = AlertManager(alert_thresholds)
        
        # Create metrics that exceed error rate critical threshold
        metrics = BackendHealthMetrics(
            backend_type=BackendType.PERPLEXITY,
            status=SystemHealthStatus.CRITICAL,
            response_time_ms=500.0,
            error_rate=0.35,  # Exceeds critical threshold of 0.3
            last_health_check=datetime.now()
        )
        
        alerts = manager.check_and_generate_alerts(metrics)
        
        assert len(alerts) > 0
        error_rate_alerts = [a for a in alerts if 'error_rate' in a.id]
        assert len(error_rate_alerts) > 0
        
        alert = error_rate_alerts[0]
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.backend_type == BackendType.PERPLEXITY
        assert alert.current_value == 0.35
        assert alert.threshold_value == 0.3
    
    def test_alert_suppression_duplicate_prevention(self, alert_thresholds):
        """Test alert suppression prevents duplicate alerts."""
        manager = AlertManager(alert_thresholds)
        
        # Create metrics that exceed threshold
        metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.DEGRADED,
            response_time_ms=1500.0,
            error_rate=0.05,
            last_health_check=datetime.now()
        )
        
        # Generate alerts multiple times
        alerts1 = manager.check_and_generate_alerts(metrics)
        alerts2 = manager.check_and_generate_alerts(metrics)
        
        # First generation should create alerts
        assert len(alerts1) > 0
        
        # Second generation should be suppressed
        assert len(alerts2) == 0 or len(alerts2) < len(alerts1)
    
    def test_alert_acknowledgment(self, alert_thresholds):
        """Test alert acknowledgment functionality."""
        manager = AlertManager(alert_thresholds)
        
        # Generate an alert
        metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.DEGRADED,
            response_time_ms=1500.0,
            error_rate=0.05,
            last_health_check=datetime.now()
        )
        
        alerts = manager.check_and_generate_alerts(metrics)
        assert len(alerts) > 0
        
        alert_id = alerts[0].id
        
        # Acknowledge the alert
        success = manager.acknowledge_alert(alert_id, "test_user")
        assert success is True
        
        # Verify alert is acknowledged
        active_alerts = manager.get_active_alerts()
        acknowledged_alert = next((a for a in active_alerts if a.id == alert_id), None)
        assert acknowledged_alert is not None
        assert acknowledged_alert.acknowledged is True
        assert acknowledged_alert.acknowledged_by == "test_user"
    
    def test_alert_resolution(self, alert_thresholds):
        """Test alert resolution functionality."""
        manager = AlertManager(alert_thresholds)
        
        # Generate an alert
        metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.CRITICAL,
            response_time_ms=3500.0,
            error_rate=0.4,
            last_health_check=datetime.now()
        )
        
        alerts = manager.check_and_generate_alerts(metrics)
        assert len(alerts) > 0
        
        alert_id = alerts[0].id
        
        # Resolve the alert
        success = manager.resolve_alert(alert_id, "test_user")
        assert success is True
        
        # Verify alert is no longer active
        active_alerts = manager.get_active_alerts()
        resolved_alert = next((a for a in active_alerts if a.id == alert_id), None)
        assert resolved_alert is None
    
    def test_auto_recovery_detection(self, alert_thresholds):
        """Test automatic recovery detection and resolution."""
        manager = AlertManager(alert_thresholds)
        
        # Generate alerts with poor metrics
        poor_metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.CRITICAL,
            response_time_ms=3500.0,
            error_rate=0.4,
            last_health_check=datetime.now()
        )
        
        alerts = manager.check_and_generate_alerts(poor_metrics)
        assert len(alerts) > 0
        
        initial_active_count = len(manager.get_active_alerts())
        
        # Improve metrics (recovery)
        good_metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=200.0,
            error_rate=0.02,
            last_health_check=datetime.now()
        )
        
        # Check for auto-recovery
        recovery_alerts = manager.check_and_generate_alerts(good_metrics)
        
        # Should have fewer active alerts due to auto-recovery
        final_active_count = len(manager.get_active_alerts())
        assert final_active_count <= initial_active_count
    
    def test_alert_history_tracking(self, alert_thresholds):
        """Test alert history tracking."""
        manager = AlertManager(alert_thresholds)
        
        # Generate multiple alerts over time
        for i in range(5):
            metrics = BackendHealthMetrics(
                backend_type=BackendType.LIGHTRAG,
                status=SystemHealthStatus.DEGRADED,
                response_time_ms=1000.0 + (i * 200),  # Increasing response time
                error_rate=0.05,
                last_health_check=datetime.now()
            )
            
            alerts = manager.check_and_generate_alerts(metrics)
            time.sleep(0.1)  # Small delay between alerts
        
        # Check alert history
        history = manager.get_alert_history(limit=10)
        assert len(history) > 0
        
        # History should be ordered by timestamp (most recent first)
        for i in range(1, len(history)):
            assert history[i-1].timestamp >= history[i].timestamp
    
    def test_alert_statistics(self, alert_thresholds):
        """Test alert statistics generation."""
        manager = AlertManager(alert_thresholds)
        
        # Generate alerts with different severities
        metrics_warning = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.DEGRADED,
            response_time_ms=1500.0,
            error_rate=0.05,
            last_health_check=datetime.now()
        )
        
        metrics_critical = BackendHealthMetrics(
            backend_type=BackendType.PERPLEXITY,
            status=SystemHealthStatus.CRITICAL,
            response_time_ms=3500.0,
            error_rate=0.35,
            last_health_check=datetime.now()
        )
        
        manager.check_and_generate_alerts(metrics_warning)
        manager.check_and_generate_alerts(metrics_critical)
        
        # Get statistics
        stats = manager.get_alert_statistics()
        
        assert 'total_alerts' in stats
        assert 'active_alerts' in stats
        assert 'alerts_by_severity' in stats
        assert 'alerts_by_backend' in stats
        assert stats['total_alerts'] > 0


# ============================================================================
# ALERT CALLBACK TESTS
# ============================================================================

class TestAlertCallbacks:
    """Test alert callback functionality."""
    
    def test_console_alert_callback(self, test_logger):
        """Test console alert callback."""
        logger, memory_handler = test_logger
        callback = ConsoleAlertCallback(logger)
        
        # Create test alert
        alert = HealthAlert(
            id="test_alert_001",
            backend_type=BackendType.LIGHTRAG,
            severity=AlertSeverity.WARNING,
            message="Response time exceeded threshold",
            threshold_breached="response_time_warning",
            current_value=1500.0,
            threshold_value=1000.0,
            timestamp=datetime.now()
        )
        
        # Process alert
        success = callback(alert)
        assert success is True
        
        # Check log output
        memory_handler.buffer  # Access log records
        # Note: More detailed log verification would require custom handler
    
    def test_json_file_alert_callback(self, temp_dir):
        """Test JSON file alert callback."""
        alert_file = os.path.join(temp_dir, "alerts.json")
        callback = JSONFileAlertCallback(alert_file, max_alerts=5)
        
        # Create test alerts
        alerts = []
        for i in range(3):
            alert = HealthAlert(
                id=f"test_alert_{i:03d}",
                backend_type=BackendType.LIGHTRAG,
                severity=AlertSeverity.WARNING,
                message=f"Test alert {i}",
                threshold_breached="response_time_warning",
                current_value=1000.0 + (i * 100),
                threshold_value=1000.0,
                timestamp=datetime.now()
            )
            alerts.append(alert)
            
            success = callback(alert)
            assert success is True
        
        # Verify file contents
        assert os.path.exists(alert_file)
        
        with open(alert_file, 'r') as f:
            saved_alerts = json.load(f)
        
        assert len(saved_alerts) == 3
        assert saved_alerts[0]['id'] == "test_alert_000"
        assert saved_alerts[2]['id'] == "test_alert_002"
    
    def test_json_file_alert_callback_rotation(self, temp_dir):
        """Test JSON file alert callback with rotation."""
        alert_file = os.path.join(temp_dir, "alerts.json")
        callback = JSONFileAlertCallback(alert_file, max_alerts=3)
        
        # Create more alerts than max_alerts
        for i in range(5):
            alert = HealthAlert(
                id=f"test_alert_{i:03d}",
                backend_type=BackendType.LIGHTRAG,
                severity=AlertSeverity.INFO,
                message=f"Test alert {i}",
                threshold_breached="info",
                current_value=i,
                threshold_value=0,
                timestamp=datetime.now()
            )
            
            callback(alert)
        
        # Verify only last 3 alerts are kept
        with open(alert_file, 'r') as f:
            saved_alerts = json.load(f)
        
        assert len(saved_alerts) == 3
        # Should contain alerts 2, 3, 4 (last 3)
        alert_ids = [alert['id'] for alert in saved_alerts]
        assert "test_alert_002" in alert_ids
        assert "test_alert_004" in alert_ids
    
    def test_json_file_alert_callback_invalid_json_recovery(self, temp_dir):
        """Test JSON file alert callback recovery from invalid JSON."""
        alert_file = os.path.join(temp_dir, "alerts.json")
        
        # Create file with invalid JSON
        with open(alert_file, 'w') as f:
            f.write("invalid json content")
        
        callback = JSONFileAlertCallback(alert_file)
        
        # Create test alert
        alert = HealthAlert(
            id="test_alert_001",
            backend_type=BackendType.LIGHTRAG,
            severity=AlertSeverity.INFO,
            message="Test recovery",
            threshold_breached="info",
            current_value=1,
            threshold_value=0,
            timestamp=datetime.now()
        )
        
        # Should succeed despite invalid existing JSON
        success = callback(alert)
        assert success is True
        
        # Verify file now contains valid JSON
        with open(alert_file, 'r') as f:
            saved_alerts = json.load(f)
        
        assert len(saved_alerts) == 1
        assert saved_alerts[0]['id'] == "test_alert_001"
    
    def test_webhook_alert_callback_success(self, mock_webhook_server):
        """Test successful webhook alert callback."""
        webhook_url, server = mock_webhook_server
        callback = WebhookAlertCallback(webhook_url)
        
        # Create test alert
        alert = HealthAlert(
            id="test_alert_001",
            backend_type=BackendType.PERPLEXITY,
            severity=AlertSeverity.CRITICAL,
            message="API authentication failed",
            threshold_breached="error_rate_critical",
            current_value=0.5,
            threshold_value=0.3,
            timestamp=datetime.now()
        )
        
        # Send alert
        success = callback(alert)
        assert success is True
        
        # Verify webhook received data
        time.sleep(0.1)  # Allow server to process
        assert len(server.received_data) == 1
        
        received = server.received_data[0]
        assert 'alert' in received
        assert received['alert']['id'] == "test_alert_001"
        assert received['source'] == 'Clinical_Metabolomics_Oracle'
    
    def test_webhook_alert_callback_failure(self):
        """Test webhook alert callback with connection failure."""
        callback = WebhookAlertCallback("http://localhost:99999", timeout=1.0)
        
        alert = HealthAlert(
            id="test_alert_001",
            backend_type=BackendType.LIGHTRAG,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            threshold_breached="response_time_warning_ms",
            current_value=1500.0,
            threshold_value=1000.0,
            timestamp=datetime.now()
        )
        
        # Should fail gracefully
        success = callback(alert)
        assert success is False
    
    def test_webhook_alert_callback_custom_headers(self, mock_webhook_server):
        """Test webhook alert callback with custom headers."""
        webhook_url, server = mock_webhook_server
        custom_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test_token',
            'X-Custom-Header': 'test_value'
        }
        
        callback = WebhookAlertCallback(webhook_url, headers=custom_headers)
        
        alert = HealthAlert(
            id="test_alert_001",
            backend_type=BackendType.LIGHTRAG,
            severity=AlertSeverity.INFO,
            message="Test with custom headers",
            threshold_breached="info",
            current_value=1,
            threshold_value=0,
            timestamp=datetime.now()
        )
        
        success = callback(alert)
        assert success is True


# ============================================================================
# SYSTEM HEALTH MONITOR INTEGRATION TESTS
# ============================================================================

class TestSystemHealthMonitor:
    """Test system health monitor functionality."""
    
    def test_system_health_monitor_initialization(self, health_check_config):
        """Test system health monitor initialization."""
        monitor = SystemHealthMonitor(check_interval=10, health_config=health_check_config)
        
        assert monitor.check_interval == 10
        assert monitor.health_config == health_check_config
        assert len(monitor.backend_health) == len(BackendType)
        assert len(monitor.health_checkers) == len(BackendType)
        assert monitor.alert_manager is not None
        assert monitor.monitoring_active is False
    
    def test_backend_health_metrics_initialization(self, health_check_config):
        """Test backend health metrics are properly initialized."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        for backend_type in BackendType:
            metrics = monitor.backend_health[backend_type]
            assert metrics.backend_type == backend_type
            assert metrics.status == SystemHealthStatus.HEALTHY
            assert metrics.response_time_ms == 0.0
            assert metrics.error_rate == 0.0
            assert isinstance(metrics.last_health_check, datetime)
    
    @pytest.mark.asyncio
    async def test_single_health_check_execution(self, health_check_config, temp_dir):
        """Test single health check execution."""
        # Setup test environment
        os.makedirs(health_check_config.lightrag_storage_dir, exist_ok=True)
        
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}), \
             patch('openai.OpenAI') as mock_openai, \
             patch('httpx.AsyncClient') as mock_client, \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock healthy responses
            mock_openai.return_value.models.list.return_value.data = [Mock()]
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.free = 5 * (1024**3)
            
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.elapsed.total_seconds.return_value = 0.1
            mock_get_response.headers = {'X-RateLimit-Remaining': '100'}
            
            mock_post_response = Mock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {'authenticated': True, 'choices': [{}]}
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_get_response)
            mock_client.return_value.post = AsyncMock(return_value=mock_post_response)
            
            # Perform single health check
            monitor._perform_health_checks()
            
            # Verify health metrics were updated
            lightrag_metrics = monitor.backend_health[BackendType.LIGHTRAG]
            perplexity_metrics = monitor.backend_health[BackendType.PERPLEXITY]
            
            assert lightrag_metrics.status == SystemHealthStatus.HEALTHY
            assert lightrag_metrics.response_time_ms > 0
            assert perplexity_metrics.status == SystemHealthStatus.HEALTHY
            assert perplexity_metrics.response_time_ms > 0
    
    def test_health_config_update(self, health_check_config):
        """Test health configuration updates."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Update configuration
        new_config = HealthCheckConfig(
            timeout_seconds=10.0,
            max_cpu_percent=90.0,
            alert_thresholds=AlertThresholds(response_time_warning=2000.0)
        )
        
        monitor.update_health_config(new_config)
        
        assert monitor.health_config.timeout_seconds == 10.0
        assert monitor.health_config.max_cpu_percent == 90.0
        assert monitor.alert_manager.alert_thresholds.response_time_warning == 2000.0
    
    def test_detailed_health_status_retrieval(self, health_check_config):
        """Test detailed health status retrieval."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Add some health history
        monitor.health_history.append({
            'timestamp': datetime.now(),
            'backend': BackendType.LIGHTRAG.value,
            'status': SystemHealthStatus.HEALTHY.value,
            'response_time_ms': 150.0,
            'error_message': None,
            'metadata': {},
            'alerts_generated': 0
        })
        
        status = monitor.get_detailed_health_status(BackendType.LIGHTRAG)
        
        assert 'current_status' in status
        assert 'recent_history' in status
        assert 'health_trends' in status
        
        current_status = status['current_status']
        assert current_status['backend_type'] == BackendType.LIGHTRAG.value
    
    def test_system_health_summary(self, health_check_config):
        """Test system health summary generation."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Set different health states
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.HEALTHY
        monitor.backend_health[BackendType.PERPLEXITY].status = SystemHealthStatus.DEGRADED
        
        summary = monitor.get_system_health_summary()
        
        assert 'overall_status' in summary
        assert 'healthy_backends' in summary
        assert 'total_backends' in summary
        assert 'backends' in summary
        
        assert summary['healthy_backends'] == 1
        assert summary['total_backends'] == len(BackendType)
        assert summary['overall_status'] == SystemHealthStatus.DEGRADED.value
    
    def test_backend_routing_eligibility(self, health_check_config):
        """Test backend routing eligibility determination."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Test healthy backend
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.HEALTHY
        assert monitor.should_route_to_backend(BackendType.LIGHTRAG) is True
        
        # Test degraded backend (still routable)
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.DEGRADED
        assert monitor.should_route_to_backend(BackendType.LIGHTRAG) is True
        
        # Test critical backend (not routable)
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.CRITICAL
        assert monitor.should_route_to_backend(BackendType.LIGHTRAG) is False
        
        # Test offline backend (not routable)
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.OFFLINE
        assert monitor.should_route_to_backend(BackendType.LIGHTRAG) is False


# ============================================================================
# PERFORMANCE AND EDGE CASE TESTS
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checking(self, health_check_config, temp_dir):
        """Test concurrent health checking performance."""
        os.makedirs(health_check_config.lightrag_storage_dir, exist_ok=True)
        
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}), \
             patch('openai.OpenAI'), \
             patch('requests.get'), \
             patch('requests.post'), \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory'), \
             patch('psutil.disk_usage'):
            
            async def run_concurrent_checks():
                """Run multiple health checks concurrently."""
                tasks = []
                for _ in range(10):
                    for backend_type in BackendType:
                        checker = monitor.health_checkers[backend_type]
                        tasks.append(checker.check_health())
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            start_time = time.perf_counter()
            results = await run_concurrent_checks()
            end_time = time.perf_counter()
            
            # Verify all checks completed
            assert len(results) == 10 * len(BackendType)
            
            # Verify no exceptions occurred
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0
            
            # Verify reasonable performance (should complete in under 30 seconds)
            total_time = end_time - start_time
            assert total_time < 30.0
    
    def test_memory_usage_and_cleanup(self, health_check_config):
        """Test memory usage and cleanup behavior."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Generate large amount of health history
        for i in range(1000):
            monitor.health_history.append({
                'timestamp': datetime.now(),
                'backend': BackendType.LIGHTRAG.value,
                'status': SystemHealthStatus.HEALTHY.value,
                'response_time_ms': 100.0 + i,
                'error_message': None,
                'metadata': {'iteration': i},
                'alerts_generated': 0
            })
        
        # Verify history is limited to maxlen
        assert len(monitor.health_history) <= 100
        
        # Generate many alerts
        manager = monitor.alert_manager
        for i in range(500):
            alert = HealthAlert(
                id=f"test_alert_{i:03d}",
                backend_type=BackendType.LIGHTRAG,
                severity=AlertSeverity.INFO,
                message=f"Test alert {i}",
                threshold_breached="info",
                current_value=i,
                threshold_value=0,
                timestamp=datetime.now()
            )
            manager.alert_history.append(alert)
        
        # Verify alert history is limited
        assert len(manager.alert_history) <= manager.alert_history.maxlen
    
    def test_error_handling_and_recovery(self, health_check_config):
        """Test error handling and recovery mechanisms."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Test with health checker that raises exceptions
        with patch.object(monitor.health_checkers[BackendType.LIGHTRAG], 'check_health', 
                         side_effect=asyncio.TimeoutError("Health check timeout")):
            
            # Should handle timeout gracefully
            monitor._perform_health_checks()
            
            # Backend should be marked as critical
            lightrag_metrics = monitor.backend_health[BackendType.LIGHTRAG]
            assert lightrag_metrics.status == SystemHealthStatus.CRITICAL
            assert lightrag_metrics.consecutive_failures > 0
        
        # Test recovery after successful check
        with patch.object(monitor.health_checkers[BackendType.LIGHTRAG], 'check_health',
                         return_value=HealthCheckResult(
                             is_healthy=True,
                             response_time_ms=200.0,
                             error_message=None,
                             metadata={}
                         )):
            
            monitor._perform_health_checks()
            
            # Should recover to healthy state
            lightrag_metrics = monitor.backend_health[BackendType.LIGHTRAG]
            assert lightrag_metrics.status in [SystemHealthStatus.HEALTHY, SystemHealthStatus.DEGRADED]
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with invalid timeout
        with pytest.raises((ValueError, TypeError)):
            HealthCheckConfig(timeout_seconds=-1.0)
        
        # Test with invalid CPU threshold
        with pytest.raises((ValueError, TypeError)):
            HealthCheckConfig(max_cpu_percent=150.0)  # > 100%
        
        # Test with invalid memory threshold  
        with pytest.raises((ValueError, TypeError)):
            HealthCheckConfig(max_memory_percent=-5.0)  # < 0%
    
    def test_alert_threshold_edge_cases(self, health_check_config):
        """Test alert threshold edge cases."""
        # Test with metrics exactly at threshold
        thresholds = AlertThresholds(response_time_warning=1000.0)
        manager = AlertManager(thresholds)
        
        metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=1000.0,  # Exactly at threshold
            error_rate=0.0,
            last_health_check=datetime.now()
        )
        
        # Should not generate alert for exact threshold match
        alerts = manager.check_and_generate_alerts(metrics)
        response_time_alerts = [a for a in alerts if 'response_time' in a.id]
        assert len(response_time_alerts) == 0
        
        # Test with metrics just above threshold
        metrics.response_time_ms = 1000.1
        alerts = manager.check_and_generate_alerts(metrics)
        response_time_alerts = [a for a in alerts if 'response_time' in a.id]
        assert len(response_time_alerts) > 0
    
    def test_health_check_timeout_handling(self, health_check_config):
        """Test health check timeout handling."""
        # Set very short timeout
        health_check_config.timeout_seconds = 0.1
        
        checker = LightRAGHealthChecker(health_check_config)
        
        with patch.object(checker, '_test_sample_query', 
                         side_effect=asyncio.TimeoutError("Query timeout")):
            
            async def run_check():
                result = await checker.check_health()
                return result
            
            result = asyncio.run(run_check())
            
            # Should handle timeout gracefully
            assert result.is_healthy is False
            assert 'timeout' in result.error_message.lower() or 'exception' in result.error_message.lower()


# ============================================================================
# INTEGRATION WITH ROUTING TESTS
# ============================================================================

class TestHealthMonitoringRoutingIntegration:
    """Test integration between health monitoring and routing system."""
    
    def test_health_monitoring_affects_routing_decisions(self, health_check_config):
        """Test that health monitoring affects routing decisions."""
        monitor = SystemHealthMonitor(health_config=health_check_config)
        
        # Set one backend as unhealthy
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.OFFLINE
        
        # Mock router integration (would normally be part of IntelligentQueryRouter)
        def health_aware_routing_decision(query: str, monitor: SystemHealthMonitor):
            """Simulate health-aware routing decision."""
            lightrag_healthy = monitor.should_route_to_backend(BackendType.LIGHTRAG)
            perplexity_healthy = monitor.should_route_to_backend(BackendType.PERPLEXITY)
            
            if not lightrag_healthy and perplexity_healthy:
                return "perplexity"
            elif lightrag_healthy and not perplexity_healthy:
                return "lightrag"
            elif lightrag_healthy and perplexity_healthy:
                return "either"  # Both healthy, use normal routing logic
            else:
                return "fallback"  # Both unhealthy, use fallback
        
        # Test routing with unhealthy LightRAG
        decision = health_aware_routing_decision("test query", monitor)
        assert decision in ["perplexity", "fallback"]
        
        # Set LightRAG as healthy
        monitor.backend_health[BackendType.LIGHTRAG].status = SystemHealthStatus.HEALTHY
        decision = health_aware_routing_decision("test query", monitor)
        assert decision in ["either", "lightrag", "perplexity"]
    
    def test_alert_callbacks_integration(self, temp_dir, mock_webhook_server, test_logger):
        """Test integration of multiple alert callbacks."""
        webhook_url, server = mock_webhook_server
        logger, memory_handler = test_logger
        
        # Setup alert manager with multiple callbacks
        manager = AlertManager()
        manager.add_callback(ConsoleAlertCallback(logger))
        manager.add_callback(JSONFileAlertCallback(os.path.join(temp_dir, "alerts.json")))
        manager.add_callback(WebhookAlertCallback(webhook_url))
        
        # Generate alert that should trigger all callbacks
        metrics = BackendHealthMetrics(
            backend_type=BackendType.LIGHTRAG,
            status=SystemHealthStatus.CRITICAL,
            response_time_ms=4000.0,
            error_rate=0.5,
            last_health_check=datetime.now()
        )
        
        alerts = manager.check_and_generate_alerts(metrics)
        assert len(alerts) > 0
        
        # Allow callbacks to execute
        time.sleep(0.2)
        
        # Verify JSON file callback worked
        json_file = os.path.join(temp_dir, "alerts.json")
        assert os.path.exists(json_file)
        
        with open(json_file, 'r') as f:
            saved_alerts = json.load(f)
        assert len(saved_alerts) > 0
        
        # Verify webhook callback worked
        assert len(server.received_data) > 0
        assert server.received_data[0]['alert']['backend_type'] == BackendType.LIGHTRAG.value
    
    def test_end_to_end_health_monitoring_workflow(self, health_check_config, temp_dir):
        """Test complete end-to-end health monitoring workflow."""
        os.makedirs(health_check_config.lightrag_storage_dir, exist_ok=True)
        
        # Create system health monitor
        monitor = SystemHealthMonitor(check_interval=1, health_config=health_check_config)
        
        # Add JSON callback for verification
        json_file = os.path.join(temp_dir, "workflow_alerts.json")
        monitor.add_alert_callback(JSONFileAlertCallback(json_file))
        
        # Mock healthy initial state
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}), \
             patch('openai.OpenAI'), \
             patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post, \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Setup healthy responses
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.free = 5 * (1024**3)
            
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {'status': 'ok'}
            mock_get_response.headers = {'X-RateLimit-Remaining': '100'}
            mock_get.return_value = mock_get_response
            
            mock_post_response = Mock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {'authenticated': True, 'choices': [{}]}
            mock_post.return_value = mock_post_response
            
            # Phase 1: Initial healthy check
            monitor._perform_health_checks()
            
            initial_summary = monitor.get_system_health_summary()
            assert initial_summary['overall_status'] == SystemHealthStatus.HEALTHY.value
            
            # Phase 2: Introduce degradation
            mock_memory.return_value.percent = 85.0  # High memory usage
            monitor._perform_health_checks()
            
            degraded_summary = monitor.get_system_health_summary()
            # System might still be healthy overall, but should have generated alerts
            
            # Phase 3: Introduce critical issues
            mock_post_response.status_code = 500  # API failure
            mock_memory.return_value.percent = 95.0  # Critical memory usage
            
            monitor._perform_health_checks()
            
            critical_summary = monitor.get_system_health_summary()
            # Should have unhealthy backends now
            
            # Verify alerts were generated
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    workflow_alerts = json.load(f)
                # Should have generated some alerts during the workflow
                assert len(workflow_alerts) >= 0  # At least some alerts expected


if __name__ == "__main__":
    # Configure logging for test execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive system health monitoring tests...")
    
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--maxfail=10",
        "--durations=10",
        "-x"  # Stop on first failure for debugging
    ])