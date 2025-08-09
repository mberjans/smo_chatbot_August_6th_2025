"""
Comprehensive Test Suite for Enhanced Load Monitoring System
===========================================================

This test suite validates the enhanced load monitoring and detection mechanisms
including hysteresis behavior, production integration, and performance optimizations.

Test Categories:
1. Basic functionality tests
2. Hysteresis mechanism validation
3. Performance optimization tests
4. Production integration tests
5. Error handling and edge cases
6. Concurrency and thread safety tests

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import statistics

# Import the components to test
try:
    from ..enhanced_load_monitoring_system import (
        EnhancedLoadDetectionSystem,
        EnhancedSystemLoadMetrics,
        HysteresisConfig,
        TrendAnalyzer,
        create_enhanced_load_monitoring_system
    )
    from ..production_monitoring_integration import (
        ProductionMonitoringAdapter,
        GracefulDegradationIntegrator,
        IntegrationConfig,
        create_integrated_monitoring_system
    )
    from ..graceful_degradation_system import (
        SystemLoadLevel,
        LoadThresholds
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    pytest.skip(f"Required components not available: {e}", allow_module_level=True)


# ============================================================================
# TEST FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture
def load_thresholds():
    """Create test load thresholds."""
    return LoadThresholds(
        cpu_normal=50.0,
        cpu_elevated=65.0,
        cpu_high=75.0,
        cpu_critical=85.0,
        cpu_emergency=92.0,
        
        memory_normal=60.0,
        memory_elevated=70.0,
        memory_high=75.0,
        memory_critical=82.0,
        memory_emergency=88.0,
        
        queue_normal=10,
        queue_elevated=20,
        queue_high=40,
        queue_critical=80,
        queue_emergency=150
    )


@pytest.fixture
def hysteresis_config():
    """Create test hysteresis configuration."""
    return HysteresisConfig(
        enabled=True,
        down_factor=0.8,
        up_factor=1.0,
        stability_window=3,
        min_dwell_time=5.0
    )


@pytest.fixture
def enhanced_detector(load_thresholds, hysteresis_config):
    """Create enhanced load detection system for testing."""
    return EnhancedLoadDetectionSystem(
        thresholds=load_thresholds,
        hysteresis_config=hysteresis_config,
        monitoring_interval=0.5,  # Fast interval for testing
        enable_trend_analysis=True
    )


@pytest.fixture
def mock_production_monitoring():
    """Create mock production monitoring system."""
    mock = Mock()
    mock.update_system_metrics = Mock()
    mock.get_request_queue_depth = Mock(return_value=10)
    mock.get_active_connections = Mock(return_value=50)
    mock.get_connection_metrics = Mock(return_value={'queue_depth': 10, 'active_connections': 50})
    return mock


class MockPsutil:
    """Mock psutil for controlled testing."""
    
    def __init__(self):
        self.cpu_percent_value = 50.0
        self.memory_percent = 60.0
        self.memory_available = 2 * 1024 * 1024 * 1024  # 2GB
        self.memory_used = 1 * 1024 * 1024 * 1024  # 1GB
        self.swap_percent = 0.0
        self.connections_count = 50
    
    def cpu_percent(self, interval=None, percpu=False):
        if percpu:
            return [self.cpu_percent_value, self.cpu_percent_value + 5, self.cpu_percent_value - 5]
        return self.cpu_percent_value
    
    def virtual_memory(self):
        mock_memory = Mock()
        mock_memory.percent = self.memory_percent
        mock_memory.available = self.memory_available
        mock_memory.used = self.memory_used
        return mock_memory
    
    def swap_memory(self):
        mock_swap = Mock()
        mock_swap.percent = self.swap_percent
        mock_swap.total = 4 * 1024 * 1024 * 1024  # 4GB
        return mock_swap
    
    def disk_io_counters(self):
        mock_disk = Mock()
        mock_disk.read_time = 100
        mock_disk.write_time = 50
        return mock_disk
    
    def net_io_counters(self):
        mock_net = Mock()
        mock_net.dropin = 0
        mock_net.dropout = 0
        return mock_net
    
    def net_connections(self, kind='inet'):
        mock_conn = Mock()
        mock_conn.status = 'ESTABLISHED'
        return [mock_conn] * self.connections_count


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestBasicFunctionality:
    """Test basic enhanced load monitoring functionality."""
    
    def test_enhanced_detector_initialization(self, load_thresholds, hysteresis_config):
        """Test proper initialization of enhanced detector."""
        detector = EnhancedLoadDetectionSystem(
            thresholds=load_thresholds,
            hysteresis_config=hysteresis_config,
            monitoring_interval=1.0
        )
        
        assert detector.thresholds == load_thresholds
        assert detector.hysteresis_config == hysteresis_config
        assert detector.monitoring_interval == 1.0
        assert detector.current_load_level == SystemLoadLevel.NORMAL
        assert len(detector.metrics_history) == 0
        assert not detector._monitoring_active
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_basic_metrics_collection(self, mock_swap, mock_memory, mock_cpu, enhanced_detector):
        """Test basic system metrics collection."""
        # Setup mocks
        mock_cpu.return_value = 70.0
        mock_memory.return_value.percent = 65.0
        mock_memory.return_value.available = 2 * 1024 * 1024 * 1024
        mock_swap.return_value.percent = 0.0
        mock_swap.return_value.total = 0
        
        # Collect metrics
        metrics = asyncio.run(enhanced_detector._collect_enhanced_metrics())
        
        assert isinstance(metrics, EnhancedSystemLoadMetrics)
        assert metrics.cpu_utilization == 70.0
        assert metrics.memory_pressure == 65.0
        assert metrics.load_level in SystemLoadLevel
        assert 0.0 <= metrics.load_score <= 1.0
    
    def test_request_metrics_recording(self, enhanced_detector):
        """Test recording of request metrics."""
        # Record some requests
        enhanced_detector.record_request_metrics(500.0)
        enhanced_detector.record_request_metrics(750.0, "timeout")
        enhanced_detector.record_request_metrics(300.0)
        enhanced_detector.record_request_metrics(1200.0, "connection_error")
        
        assert len(enhanced_detector._response_times) == 4
        assert enhanced_detector._total_requests == 4
        assert enhanced_detector._error_counts["timeout"] == 1
        assert enhanced_detector._error_counts["connection_error"] == 1
    
    def test_queue_and_connection_updates(self, enhanced_detector):
        """Test updating queue depth and connection counts."""
        enhanced_detector.update_queue_depth(25)
        enhanced_detector.update_connection_count(75)
        
        assert enhanced_detector._request_queue_depth == 25
        assert enhanced_detector._active_connections == 75
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, enhanced_detector):
        """Test starting and stopping monitoring."""
        # Initially not active
        assert not enhanced_detector._monitoring_active
        
        # Start monitoring
        await enhanced_detector.start_monitoring()
        assert enhanced_detector._monitoring_active
        assert enhanced_detector._monitor_task is not None
        
        # Let it run briefly
        await asyncio.sleep(1.0)
        
        # Stop monitoring
        await enhanced_detector.stop_monitoring()
        assert not enhanced_detector._monitoring_active


# ============================================================================\n# HYSTERESIS MECHANISM TESTS\n# ============================================================================\n\nclass TestHysteresisMechanism:\n    \"\"\"Test hysteresis behavior for stable threshold management.\"\"\"\n    \n    def test_hysteresis_config_creation(self):\n        \"\"\"Test hysteresis configuration creation.\"\"\"\n        config = HysteresisConfig(\n            enabled=True,\n            down_factor=0.75,\n            up_factor=1.1,\n            min_dwell_time=10.0\n        )\n        \n        assert config.enabled\n        assert config.down_factor == 0.75\n        assert config.up_factor == 1.1\n        assert config.min_dwell_time == 10.0\n    \n    @patch('psutil.cpu_percent')\n    @patch('psutil.virtual_memory')\n    def test_hysteresis_prevents_rapid_level_drops(self, mock_memory, mock_cpu, enhanced_detector):\n        \"\"\"Test that hysteresis prevents rapid dropping of load levels.\"\"\"\n        # Setup high load initially\n        mock_cpu.return_value = 90.0  # CRITICAL level\n        mock_memory.return_value.percent = 85.0  # CRITICAL level\n        mock_memory.return_value.available = 1 * 1024 * 1024 * 1024\n        \n        # First measurement at critical level\n        metrics1 = asyncio.run(enhanced_detector._collect_enhanced_metrics())\n        enhanced_detector.current_load_level = metrics1.load_level\n        enhanced_detector.last_level_change_time = datetime.now()\n        \n        assert metrics1.load_level == SystemLoadLevel.CRITICAL\n        \n        # Immediately drop load significantly\n        mock_cpu.return_value = 60.0  # Would be ELEVATED without hysteresis\n        mock_memory.return_value.percent = 65.0  # Would be ELEVATED\n        \n        # Second measurement - should stay at CRITICAL due to min_dwell_time\n        metrics2 = asyncio.run(enhanced_detector._collect_enhanced_metrics())\n        \n        # Should stay at CRITICAL due to hysteresis\n        assert metrics2.load_level == SystemLoadLevel.CRITICAL\n        assert metrics2.hysteresis_factor_applied < 1.0\n    \n    @patch('psutil.cpu_percent')\n    @patch('psutil.virtual_memory')\n    def test_hysteresis_allows_level_drops_after_dwell_time(self, mock_memory, mock_cpu, enhanced_detector):\n        \"\"\"Test that hysteresis allows level drops after sufficient dwell time.\"\"\"\n        # Start at critical level\n        enhanced_detector.current_load_level = SystemLoadLevel.CRITICAL\n        enhanced_detector.last_level_change_time = datetime.now() - timedelta(seconds=10)\n        \n        # Setup lower load\n        mock_cpu.return_value = 55.0  # ELEVATED level\n        mock_memory.return_value.percent = 60.0  # NORMAL level\n        mock_memory.return_value.available = 2 * 1024 * 1024 * 1024\n        \n        # Should allow level drop after dwell time\n        metrics = asyncio.run(enhanced_detector._collect_enhanced_metrics())\n        \n        # Should drop to lower level after sufficient time\n        assert metrics.load_level < SystemLoadLevel.CRITICAL\n    \n    def test_hysteresis_threshold_adjustment(self, enhanced_detector, load_thresholds):\n        \"\"\"Test hysteresis threshold adjustment calculation.\"\"\"\n        factor = 0.8\n        adjusted_thresholds = enhanced_detector._apply_factor_to_thresholds(load_thresholds, factor)\n        \n        assert adjusted_thresholds.cpu_high == load_thresholds.cpu_high * factor\n        assert adjusted_thresholds.memory_critical == load_thresholds.memory_critical * factor\n        assert adjusted_thresholds.queue_emergency == int(load_thresholds.queue_emergency * factor)\n\n\n# ============================================================================\n# PERFORMANCE OPTIMIZATION TESTS\n# ============================================================================\n\nclass TestPerformanceOptimization:\n    \"\"\"Test performance optimizations in enhanced monitoring.\"\"\"\n    \n    def test_percentile_caching(self, enhanced_detector):\n        \"\"\"Test response time percentile caching.\"\"\"\n        # Add response times\n        response_times = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]\n        for rt in response_times:\n            enhanced_detector.record_request_metrics(rt)\n        \n        # First call should calculate and cache\n        current_time = time.time()\n        p95, p99, p99_9 = enhanced_detector._get_response_metrics_cached(current_time)\n        \n        assert p95 > 0\n        assert p99 > p95\n        assert p99_9 >= p99\n        \n        # Second call within cache TTL should use cached values\n        p95_2, p99_2, p99_9_2 = enhanced_detector._get_response_metrics_cached(current_time + 1.0)\n        \n        assert p95 == p95_2\n        assert p99 == p99_2\n        assert p99_9 == p99_9_2\n        \n        # Call after cache expiry should recalculate\n        p95_3, p99_3, p99_9_3 = enhanced_detector._get_response_metrics_cached(current_time + 3.0)\n        \n        # Values might be same but calculation was redone\n        assert enhanced_detector._cached_percentiles_timestamp > current_time + 2.0\n    \n    def test_metrics_history_size_limit(self, enhanced_detector):\n        \"\"\"Test that metrics history doesn't grow unbounded.\"\"\"\n        # Add more metrics than the max limit\n        for i in range(600):  # More than maxlen=500\n            metrics = EnhancedSystemLoadMetrics(\n                timestamp=datetime.now(),\n                cpu_utilization=50.0,\n                memory_pressure=60.0,\n                request_queue_depth=10,\n                response_time_p95=500.0,\n                response_time_p99=800.0,\n                error_rate=0.1,\n                active_connections=50,\n                disk_io_wait=10.0\n            )\n            enhanced_detector.metrics_history.append(metrics)\n        \n        # Should be limited to maxlen\n        assert len(enhanced_detector.metrics_history) == 500\n    \n    def test_error_rate_sliding_window(self, enhanced_detector):\n        \"\"\"Test error rate sliding window mechanism.\"\"\"\n        # Generate many requests to trigger sliding window\n        for i in range(25000):  # More than sliding window threshold\n            error_type = \"error\" if i % 100 == 0 else None\n            enhanced_detector.record_request_metrics(500.0, error_type)\n        \n        # Should have applied sliding window\n        assert enhanced_detector._total_requests < 25000\n        assert enhanced_detector._total_requests > 15000  # Should be reduced but not too much\n    \n    @pytest.mark.asyncio\n    async def test_concurrent_metrics_collection(self, enhanced_detector):\n        \"\"\"Test thread safety during concurrent metrics collection.\"\"\"\n        results = []\n        \n        async def collect_metrics():\n            for _ in range(50):\n                enhanced_detector.record_request_metrics(500.0)\n                await asyncio.sleep(0.01)\n        \n        # Run multiple concurrent collectors\n        tasks = [collect_metrics() for _ in range(5)]\n        await asyncio.gather(*tasks)\n        \n        # Should have recorded all requests safely\n        assert enhanced_detector._total_requests == 250\n\n\n# ============================================================================\n# TREND ANALYSIS TESTS\n# ============================================================================\n\nclass TestTrendAnalysis:\n    \"\"\"Test trend analysis functionality.\"\"\"\n    \n    def test_trend_analyzer_creation(self):\n        \"\"\"Test trend analyzer initialization.\"\"\"\n        analyzer = TrendAnalyzer(analysis_window=20)\n        assert analyzer.analysis_window == 20\n    \n    def test_trend_calculation(self):\n        \"\"\"Test trend calculation for various patterns.\"\"\"\n        analyzer = TrendAnalyzer()\n        \n        # Test increasing trend\n        increasing_values = [10, 20, 30, 40, 50]\n        trend = analyzer._calculate_trend(increasing_values)\n        assert trend > 0\n        \n        # Test decreasing trend\n        decreasing_values = [50, 40, 30, 20, 10]\n        trend = analyzer._calculate_trend(decreasing_values)\n        assert trend < 0\n        \n        # Test stable values\n        stable_values = [30, 30, 30, 30, 30]\n        trend = analyzer._calculate_trend(stable_values)\n        assert abs(trend) < 0.1  # Should be near zero\n    \n    def test_volatility_calculation(self):\n        \"\"\"Test volatility calculation.\"\"\"\n        analyzer = TrendAnalyzer()\n        \n        # Test stable values (low volatility)\n        stable_values = [50, 51, 49, 50, 52]\n        volatility = analyzer._calculate_volatility(stable_values)\n        assert 0 <= volatility < 0.1\n        \n        # Test volatile values (high volatility)\n        volatile_values = [10, 90, 20, 80, 30]\n        volatility = analyzer._calculate_volatility(volatile_values)\n        assert volatility > 0.5\n    \n    def test_trend_analysis_integration(self, enhanced_detector):\n        \"\"\"Test trend analysis integration with enhanced detector.\"\"\"\n        # Create history with trend\n        history = []\n        for i in range(10):\n            metrics = EnhancedSystemLoadMetrics(\n                timestamp=datetime.now(),\n                cpu_utilization=50.0 + i * 2,  # Increasing trend\n                memory_pressure=60.0,\n                request_queue_depth=10,\n                response_time_p95=500.0 + i * 50,  # Increasing trend\n                response_time_p99=800.0,\n                error_rate=0.1,\n                active_connections=50,\n                disk_io_wait=10.0,\n                load_score=0.5 + i * 0.02  # Increasing trend\n            )\n            history.append(metrics)\n        \n        current_metrics = history[-1]\n        \n        # Analyze trends\n        if enhanced_detector._trend_analyzer:\n            indicators = enhanced_detector._trend_analyzer.analyze(current_metrics, history)\n            \n            assert 'cpu_trend' in indicators\n            assert 'load_trend' in indicators\n            assert indicators['cpu_trend'] > 0  # Should detect increasing trend\n            assert indicators['load_trend'] > 0  # Should detect increasing trend\n\n\n# ============================================================================\n# PRODUCTION INTEGRATION TESTS\n# ============================================================================\n\nclass TestProductionIntegration:\n    \"\"\"Test production monitoring integration.\"\"\"\n    \n    def test_integration_config_creation(self):\n        \"\"\"Test integration configuration creation.\"\"\"\n        config = IntegrationConfig(\n            sync_interval=10.0,\n            bidirectional_sync=True,\n            enable_metrics_caching=True,\n            cache_ttl=5.0\n        )\n        \n        assert config.sync_interval == 10.0\n        assert config.bidirectional_sync\n        assert config.enable_metrics_caching\n        assert config.cache_ttl == 5.0\n    \n    def test_production_monitoring_adapter_creation(self, mock_production_monitoring, enhanced_detector):\n        \"\"\"Test production monitoring adapter initialization.\"\"\"\n        adapter = ProductionMonitoringAdapter(\n            production_monitoring=mock_production_monitoring,\n            enhanced_detector=enhanced_detector,\n            config=IntegrationConfig()\n        )\n        \n        assert adapter.production_monitoring == mock_production_monitoring\n        assert adapter.enhanced_detector == enhanced_detector\n        assert not adapter._integration_active\n    \n    @pytest.mark.asyncio\n    async def test_adapter_lifecycle(self, mock_production_monitoring, enhanced_detector):\n        \"\"\"Test adapter start/stop lifecycle.\"\"\"\n        adapter = ProductionMonitoringAdapter(\n            production_monitoring=mock_production_monitoring,\n            enhanced_detector=enhanced_detector\n        )\n        \n        # Start integration\n        await adapter.start_integration()\n        assert adapter._integration_active\n        \n        # Stop integration\n        await adapter.stop_integration()\n        assert not adapter._integration_active\n    \n    @pytest.mark.asyncio\n    async def test_metrics_pushing(self, mock_production_monitoring, enhanced_detector):\n        \"\"\"Test pushing enhanced metrics to production monitoring.\"\"\"\n        adapter = ProductionMonitoringAdapter(\n            production_monitoring=mock_production_monitoring,\n            enhanced_detector=enhanced_detector\n        )\n        \n        # Set up enhanced metrics\n        enhanced_detector.current_metrics = EnhancedSystemLoadMetrics(\n            timestamp=datetime.now(),\n            cpu_utilization=75.0,\n            memory_pressure=70.0,\n            request_queue_depth=25,\n            response_time_p95=1500.0,\n            response_time_p99=2500.0,\n            error_rate=1.5,\n            active_connections=100,\n            disk_io_wait=50.0,\n            memory_available_mb=1024.0,\n            load_level=SystemLoadLevel.HIGH\n        )\n        \n        # Push metrics\n        await adapter._push_enhanced_metrics()\n        \n        # Verify production monitoring was called\n        mock_production_monitoring.update_system_metrics.assert_called_once()\n    \n    @pytest.mark.asyncio\n    async def test_metrics_pulling(self, mock_production_monitoring, enhanced_detector):\n        \"\"\"Test pulling metrics from production monitoring.\"\"\"\n        adapter = ProductionMonitoringAdapter(\n            production_monitoring=mock_production_monitoring,\n            enhanced_detector=enhanced_detector\n        )\n        \n        # Pull metrics\n        await adapter._pull_production_metrics()\n        \n        # Verify production monitoring methods were called\n        mock_production_monitoring.get_request_queue_depth.assert_called()\n        mock_production_monitoring.get_active_connections.assert_called()\n    \n    def test_graceful_degradation_integrator(self, enhanced_detector):\n        \"\"\"Test graceful degradation manager integration.\"\"\"\n        # Create mock degradation manager\n        mock_degradation_manager = Mock()\n        mock_degradation_manager.load_detector = Mock()\n        mock_degradation_manager._handle_load_change = Mock()\n        \n        # Create integrator\n        integrator = GracefulDegradationIntegrator(\n            degradation_manager=mock_degradation_manager,\n            enhanced_detector=enhanced_detector\n        )\n        \n        # Perform integration\n        success = integrator.integrate()\n        assert success\n        \n        # Verify detector was replaced\n        assert mock_degradation_manager.load_detector == enhanced_detector\n        \n        # Test rollback\n        rollback_success = integrator.rollback()\n        assert rollback_success\n    \n    def test_complete_integration_factory(self, mock_production_monitoring):\n        \"\"\"Test complete integration factory function.\"\"\"\n        enhanced_detector, adapter, degradation_integrator = create_integrated_monitoring_system(\n            production_monitoring=mock_production_monitoring,\n            monitoring_interval=1.0\n        )\n        \n        assert enhanced_detector is not None\n        assert adapter is not None\n        assert degradation_integrator is None  # No degradation manager provided\n        \n        assert adapter.production_monitoring == mock_production_monitoring\n        assert adapter.enhanced_detector == enhanced_detector\n\n\n# ============================================================================\n# ERROR HANDLING AND EDGE CASES\n# ============================================================================\n\nclass TestErrorHandling:\n    \"\"\"Test error handling and edge cases.\"\"\"\n    \n    def test_safe_metrics_collection_on_psutil_error(self, enhanced_detector):\n        \"\"\"Test safe handling of psutil errors.\"\"\"\n        with patch('psutil.cpu_percent', side_effect=Exception(\"CPU error\")):\n            with patch('psutil.virtual_memory', side_effect=Exception(\"Memory error\")):\n                metrics = asyncio.run(enhanced_detector._collect_enhanced_metrics())\n                \n                # Should return safe defaults\n                assert isinstance(metrics, EnhancedSystemLoadMetrics)\n                assert metrics.load_level == SystemLoadLevel.NORMAL\n                assert metrics.load_score == 0.0\n    \n    def test_empty_response_times_handling(self, enhanced_detector):\n        \"\"\"Test handling of empty response times.\"\"\"\n        # No response times recorded\n        current_time = time.time()\n        p95, p99, p99_9 = enhanced_detector._get_response_metrics_cached(current_time)\n        \n        assert p95 == 0.0\n        assert p99 == 0.0\n        assert p99_9 == 0.0\n    \n    def test_zero_total_requests_error_rate(self, enhanced_detector):\n        \"\"\"Test error rate calculation with zero total requests.\"\"\"\n        # No requests recorded\n        error_metrics = enhanced_detector._get_error_metrics_categorized()\n        \n        assert error_metrics['total_rate'] == 0.0\n        assert error_metrics['categories'] == {}\n        assert error_metrics['total_errors'] == 0\n    \n    @pytest.mark.asyncio\n    async def test_adapter_fallback_mode(self, mock_production_monitoring, enhanced_detector):\n        \"\"\"Test adapter fallback mode on repeated sync failures.\"\"\"\n        adapter = ProductionMonitoringAdapter(\n            production_monitoring=mock_production_monitoring,\n            enhanced_detector=enhanced_detector,\n            config=IntegrationConfig(max_sync_failures=2, fallback_mode_timeout=1.0)\n        )\n        \n        # Make sync fail\n        with patch.object(adapter, '_perform_sync', side_effect=Exception(\"Sync error\")):\n            # Simulate multiple failures\n            adapter._sync_failures = 2\n            await adapter._enter_fallback_mode()\n            \n            assert adapter._fallback_mode\n    \n    def test_trend_analysis_with_insufficient_data(self):\n        \"\"\"Test trend analysis with insufficient historical data.\"\"\"\n        analyzer = TrendAnalyzer()\n        \n        # Empty history\n        current_metrics = EnhancedSystemLoadMetrics(\n            timestamp=datetime.now(),\n            cpu_utilization=50.0,\n            memory_pressure=60.0,\n            request_queue_depth=10,\n            response_time_p95=500.0,\n            response_time_p99=800.0,\n            error_rate=0.1,\n            active_connections=50,\n            disk_io_wait=10.0\n        )\n        \n        indicators = analyzer.analyze(current_metrics, [])\n        assert indicators == {}  # Should return empty dict\n    \n    def test_hysteresis_with_disabled_config(self, enhanced_detector, load_thresholds):\n        \"\"\"Test behavior when hysteresis is disabled.\"\"\"\n        # Disable hysteresis\n        enhanced_detector.hysteresis_config.enabled = False\n        \n        # Create test metrics\n        metrics = EnhancedSystemLoadMetrics(\n            timestamp=datetime.now(),\n            cpu_utilization=80.0,  # HIGH level\n            memory_pressure=65.0,\n            request_queue_depth=10,\n            response_time_p95=500.0,\n            response_time_p99=800.0,\n            error_rate=0.1,\n            active_connections=50,\n            disk_io_wait=10.0\n        )\n        \n        # Should use base calculation without hysteresis\n        level = enhanced_detector._calculate_load_level_with_hysteresis(metrics)\n        base_level = enhanced_detector._calculate_base_load_level(metrics)\n        \n        assert level == base_level  # Should be identical\n\n\n# ============================================================================\n# INTEGRATION TESTS\n# ============================================================================\n\nclass TestIntegrationScenarios:\n    \"\"\"Test complete integration scenarios.\"\"\"\n    \n    @pytest.mark.asyncio\n    async def test_complete_monitoring_workflow(self, mock_production_monitoring):\n        \"\"\"Test complete monitoring workflow with all components.\"\"\"\n        # Create integrated system\n        enhanced_detector, adapter, _ = create_integrated_monitoring_system(\n            production_monitoring=mock_production_monitoring,\n            monitoring_interval=0.5\n        )\n        \n        # Start monitoring\n        await enhanced_detector.start_monitoring()\n        await adapter.start_integration()\n        \n        # Simulate some activity\n        for i in range(5):\n            enhanced_detector.record_request_metrics(500 + i * 100)\n            enhanced_detector.update_queue_depth(10 + i * 5)\n            await asyncio.sleep(0.6)  # Wait for monitoring cycle\n        \n        # Verify metrics were collected\n        assert enhanced_detector.current_metrics is not None\n        assert len(enhanced_detector.metrics_history) > 0\n        \n        # Verify production monitoring was updated\n        assert mock_production_monitoring.update_system_metrics.called\n        \n        # Clean up\n        await adapter.stop_integration()\n        await enhanced_detector.stop_monitoring()\n    \n    @pytest.mark.asyncio\n    async def test_load_level_transitions_with_hysteresis(self, enhanced_detector):\n        \"\"\"Test load level transitions with hysteresis behavior.\"\"\"\n        transitions = []\n        \n        def record_transition(metrics):\n            transitions.append((datetime.now(), metrics.load_level))\n        \n        enhanced_detector.add_load_change_callback(record_transition)\n        \n        # Simulate load increases and decreases\n        with patch('psutil.cpu_percent') as mock_cpu, \\\n             patch('psutil.virtual_memory') as mock_memory:\n            \n            mock_memory.return_value.percent = 60.0\n            mock_memory.return_value.available = 2 * 1024 * 1024 * 1024\n            \n            # Start with normal load\n            mock_cpu.return_value = 45.0\n            await enhanced_detector._collect_enhanced_metrics()\n            \n            # Increase to high load\n            mock_cpu.return_value = 80.0\n            await enhanced_detector._collect_enhanced_metrics()\n            \n            # Immediately drop load (should be prevented by hysteresis)\n            mock_cpu.return_value = 50.0\n            await enhanced_detector._collect_enhanced_metrics()\n            \n            # After sufficient time, should allow drop\n            enhanced_detector.last_level_change_time = datetime.now() - timedelta(seconds=10)\n            await enhanced_detector._collect_enhanced_metrics()\n        \n        # Verify hysteresis behavior in transitions\n        if len(transitions) > 1:\n            # Should show hysteresis effect\n            assert any(trans[1] != SystemLoadLevel.NORMAL for trans in transitions)\n\n\n# ============================================================================\n# PERFORMANCE BENCHMARKS\n# ============================================================================\n\nclass TestPerformanceBenchmarks:\n    \"\"\"Performance benchmark tests for enhanced monitoring.\"\"\"\n    \n    @pytest.mark.asyncio\n    async def test_metrics_collection_performance(self, enhanced_detector):\n        \"\"\"Test performance of metrics collection under load.\"\"\"\n        start_time = time.time()\n        \n        # Collect metrics multiple times\n        for _ in range(100):\n            await enhanced_detector._collect_enhanced_metrics()\n        \n        end_time = time.time()\n        avg_time = (end_time - start_time) / 100\n        \n        # Should collect metrics quickly (< 50ms on average)\n        assert avg_time < 0.05, f\"Metrics collection too slow: {avg_time:.3f}s\"\n    \n    def test_percentile_calculation_performance(self, enhanced_detector):\n        \"\"\"Test performance of percentile calculations.\"\"\"\n        # Add large number of response times\n        for i in range(10000):\n            enhanced_detector.record_request_metrics(float(i))\n        \n        start_time = time.time()\n        \n        # Calculate percentiles multiple times\n        for _ in range(100):\n            enhanced_detector._get_response_metrics_cached(time.time())\n        \n        end_time = time.time()\n        avg_time = (end_time - start_time) / 100\n        \n        # Should be fast due to caching\n        assert avg_time < 0.01, f\"Percentile calculation too slow: {avg_time:.3f}s\"\n    \n    def test_memory_usage_bounds(self, enhanced_detector):\n        \"\"\"Test that memory usage stays within bounds.\"\"\"\n        import sys\n        \n        initial_size = sys.getsizeof(enhanced_detector)\n        \n        # Generate lots of data\n        for i in range(10000):\n            enhanced_detector.record_request_metrics(float(i))\n            \n            metrics = EnhancedSystemLoadMetrics(\n                timestamp=datetime.now(),\n                cpu_utilization=50.0,\n                memory_pressure=60.0,\n                request_queue_depth=10,\n                response_time_p95=500.0,\n                response_time_p99=800.0,\n                error_rate=0.1,\n                active_connections=50,\n                disk_io_wait=10.0\n            )\n            enhanced_detector.metrics_history.append(metrics)\n        \n        final_size = sys.getsizeof(enhanced_detector)\n        \n        # Should not grow excessively due to bounded collections\n        growth_factor = final_size / initial_size\n        assert growth_factor < 10, f\"Excessive memory growth: {growth_factor:.1f}x\"\n\n\nif __name__ == \"__main__\":\n    pytest.main([__file__, \"-v\"])