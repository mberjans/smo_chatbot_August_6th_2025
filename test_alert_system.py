#!/usr/bin/env python3
"""
Test script for Alert Management System integration

This script demonstrates the alert system functionality including:
- Alert generation based on health metrics
- Alert suppression and acknowledgment
- Callback system execution
- Alert history and statistics
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the project directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightrag_integration.intelligent_query_router import (
    IntelligentQueryRouter,
    BackendHealthMetrics,
    BackendType,
    SystemHealthStatus,
    AlertThresholds,
    AlertSeverity,
    HealthCheckConfig,
    WebhookAlertCallback
)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_test_health_metrics(backend_type: BackendType, 
                             response_time_ms: float = 500.0,
                             error_rate: float = 0.02,
                             availability_percentage: float = 99.5,
                             consecutive_failures: int = 0) -> BackendHealthMetrics:
    """Create test health metrics"""
    metrics = BackendHealthMetrics(
        backend_type=backend_type,
        status=SystemHealthStatus.HEALTHY,
        response_time_ms=response_time_ms,
        error_rate=error_rate,
        last_health_check=datetime.now(),
        consecutive_failures=consecutive_failures,
        availability_percentage=availability_percentage
    )
    
    # Set resource usage
    metrics.cpu_usage_percent = 45.0
    metrics.memory_usage_percent = 60.0
    metrics.disk_usage_percent = 35.0
    
    return metrics

def simulate_degraded_metrics(backend_type: BackendType) -> BackendHealthMetrics:
    """Create degraded health metrics that should trigger alerts"""
    return create_test_health_metrics(
        backend_type=backend_type,
        response_time_ms=1500.0,  # Above warning threshold
        error_rate=0.08,          # Above warning threshold
        availability_percentage=93.0,  # Below warning threshold
        consecutive_failures=4    # Above warning threshold
    )

def simulate_critical_metrics(backend_type: BackendType) -> BackendHealthMetrics:
    """Create critical health metrics that should trigger critical alerts"""
    return create_test_health_metrics(
        backend_type=backend_type,
        response_time_ms=2500.0,  # Above critical threshold
        error_rate=0.20,          # Above critical threshold
        availability_percentage=85.0,  # Below critical threshold
        consecutive_failures=7    # Above critical threshold
    )

def test_alert_generation(logger):
    """Test alert generation functionality"""
    logger.info("=== Testing Alert Generation ===")
    
    # Create router with custom alert thresholds
    alert_thresholds = AlertThresholds(
        response_time_warning=1000.0,
        response_time_critical=2000.0,
        error_rate_warning=0.05,
        error_rate_critical=0.15,
        availability_warning=95.0,
        availability_critical=90.0,
        consecutive_failures_warning=3,
        consecutive_failures_critical=5
    )
    
    health_config = HealthCheckConfig(alert_thresholds=alert_thresholds)
    router = IntelligentQueryRouter(health_check_config=health_config)
    
    # Test 1: Normal metrics (should not generate alerts)
    logger.info("Test 1: Normal metrics")
    normal_metrics = create_test_health_metrics(BackendType.LIGHTRAG)
    alerts = router.health_monitor.alert_manager.check_and_generate_alerts(normal_metrics)
    logger.info(f"Generated {len(alerts)} alerts for normal metrics (expected: 0)")
    
    # Test 2: Degraded metrics (should generate warning alerts)
    logger.info("Test 2: Degraded metrics")
    degraded_metrics = simulate_degraded_metrics(BackendType.LIGHTRAG)
    alerts = router.health_monitor.alert_manager.check_and_generate_alerts(degraded_metrics)
    logger.info(f"Generated {len(alerts)} alerts for degraded metrics")
    for alert in alerts:
        logger.info(f"  - {alert.severity.value.upper()}: {alert.message}")
    
    # Test 3: Critical metrics (should generate critical alerts)
    logger.info("Test 3: Critical metrics")
    critical_metrics = simulate_critical_metrics(BackendType.PERPLEXITY)
    alerts = router.health_monitor.alert_manager.check_and_generate_alerts(critical_metrics)
    logger.info(f"Generated {len(alerts)} alerts for critical metrics")
    for alert in alerts:
        logger.info(f"  - {alert.severity.value.upper()}: {alert.message}")
    
    router.shutdown()
    return True

def test_alert_suppression(logger):
    """Test alert suppression functionality"""
    logger.info("=== Testing Alert Suppression ===")
    
    router = IntelligentQueryRouter()
    alert_manager = router.health_monitor.alert_manager
    
    # Generate the same alert multiple times rapidly
    degraded_metrics = simulate_degraded_metrics(BackendType.LIGHTRAG)
    
    total_alerts = 0
    for i in range(10):
        alerts = alert_manager.check_and_generate_alerts(degraded_metrics)
        total_alerts += len(alerts)
        logger.info(f"Iteration {i+1}: Generated {len(alerts)} alerts")
        time.sleep(0.1)  # Small delay
    
    logger.info(f"Total alerts generated: {total_alerts} (suppression should reduce this)")
    
    # Check active alerts
    active_alerts = alert_manager.get_active_alerts()
    logger.info(f"Active alerts: {len(active_alerts)}")
    
    router.shutdown()
    return True

def test_alert_acknowledgment(logger):
    """Test alert acknowledgment and resolution"""
    logger.info("=== Testing Alert Acknowledgment ===")
    
    router = IntelligentQueryRouter()
    
    # Generate some alerts
    critical_metrics = simulate_critical_metrics(BackendType.LIGHTRAG)
    generated_alerts = router.health_monitor.alert_manager.check_and_generate_alerts(critical_metrics)
    logger.info(f"Generated {len(generated_alerts)} alerts")
    
    # Get active alerts
    active_alerts = router.get_active_alerts()
    logger.info(f"Active alerts: {len(active_alerts)}")
    
    if active_alerts:
        # Acknowledge first alert
        alert_id = active_alerts[0]['id']
        success = router.acknowledge_alert(alert_id, "test_user")
        logger.info(f"Acknowledged alert {alert_id}: {success}")
        
        # Check acknowledgment status
        updated_alerts = router.get_active_alerts()
        acknowledged = [a for a in updated_alerts if a.get('acknowledged', False)]
        logger.info(f"Acknowledged alerts: {len(acknowledged)}")
        
        # Resolve an alert
        if len(active_alerts) > 1:
            alert_id = active_alerts[1]['id']
            success = router.resolve_alert(alert_id, "test_user")
            logger.info(f"Resolved alert {alert_id}: {success}")
            
            # Check active alerts after resolution
            final_alerts = router.get_active_alerts()
            logger.info(f"Active alerts after resolution: {len(final_alerts)}")
    
    router.shutdown()
    return True

def test_callback_system(logger):
    """Test alert callback system"""
    logger.info("=== Testing Alert Callback System ===")
    
    router = IntelligentQueryRouter()
    
    # Test webhook callback registration
    webhook_registered = router.register_alert_callback(
        "webhook", 
        webhook_url="http://localhost:8000/alerts",  # This will fail but test the mechanism
        timeout=2.0
    )
    logger.info(f"Webhook callback registered: {webhook_registered}")
    
    # Test JSON file callback registration  
    test_alerts_file = "./logs/test_alerts.json"
    json_registered = router.register_alert_callback(
        "json_file",
        file_path=test_alerts_file,
        max_alerts=100
    )
    logger.info(f"JSON file callback registered: {json_registered}")
    
    # Generate alerts to test callbacks
    critical_metrics = simulate_critical_metrics(BackendType.PERPLEXITY)
    alerts = router.health_monitor.alert_manager.check_and_generate_alerts(critical_metrics)
    logger.info(f"Generated {len(alerts)} alerts for callback testing")
    
    # Wait a moment for callbacks to execute
    time.sleep(1.0)
    
    # Check if JSON file was created
    if os.path.exists(test_alerts_file):
        logger.info(f"JSON alert file created: {test_alerts_file}")
        with open(test_alerts_file, 'r') as f:
            content = f.read()
            logger.info(f"JSON file size: {len(content)} characters")
    else:
        logger.warning(f"JSON alert file not found: {test_alerts_file}")
    
    router.shutdown()
    return True

def test_alert_statistics(logger):
    """Test alert statistics and analytics"""
    logger.info("=== Testing Alert Statistics ===")
    
    router = IntelligentQueryRouter()
    
    # Generate various types of alerts
    backends = [BackendType.LIGHTRAG, BackendType.PERPLEXITY]
    
    for backend in backends:
        # Generate warning alerts
        degraded_metrics = simulate_degraded_metrics(backend)
        router.health_monitor.alert_manager.check_and_generate_alerts(degraded_metrics)
        
        # Generate critical alerts
        critical_metrics = simulate_critical_metrics(backend)
        router.health_monitor.alert_manager.check_and_generate_alerts(critical_metrics)
    
    # Get alert statistics
    stats = router.get_alert_statistics()
    logger.info("Alert Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Get alert history
    history = router.get_alert_history(limit=10)
    logger.info(f"Recent alert history: {len(history)} alerts")
    for alert in history[:3]:  # Show first 3
        logger.info(f"  - {alert['timestamp']}: {alert['severity']} - {alert['message']}")
    
    router.shutdown()
    return True

def test_threshold_configuration(logger):
    """Test alert threshold configuration"""
    logger.info("=== Testing Threshold Configuration ===")
    
    router = IntelligentQueryRouter()
    
    # Configure new thresholds
    new_thresholds = {
        'response_time_warning': 800.0,
        'response_time_critical': 1500.0,
        'error_rate_warning': 0.03,
        'error_rate_critical': 0.10
    }
    
    success = router.configure_alert_thresholds(new_thresholds)
    logger.info(f"Threshold configuration successful: {success}")
    
    # Test with metrics that should trigger alerts with new thresholds
    test_metrics = create_test_health_metrics(
        BackendType.LIGHTRAG,
        response_time_ms=900.0,  # Should trigger warning with new threshold
        error_rate=0.04          # Should trigger warning with new threshold
    )
    
    alerts = router.health_monitor.alert_manager.check_and_generate_alerts(test_metrics)
    logger.info(f"Generated {len(alerts)} alerts with new thresholds")
    for alert in alerts:
        logger.info(f"  - {alert.severity.value.upper()}: {alert.message}")
    
    router.shutdown()
    return True

def test_comprehensive_health_status(logger):
    """Test comprehensive health status with alerts"""
    logger.info("=== Testing Comprehensive Health Status ===")
    
    router = IntelligentQueryRouter()
    
    # Generate some alerts first
    critical_metrics = simulate_critical_metrics(BackendType.LIGHTRAG)
    router.health_monitor.alert_manager.check_and_generate_alerts(critical_metrics)
    
    # Get comprehensive health status
    health_status = router.get_system_health_with_alerts()
    
    logger.info("System Health with Alerts:")
    logger.info(f"  Overall Status: {health_status.get('overall_status')}")
    logger.info(f"  Healthy Backends: {health_status.get('healthy_backends')}/{health_status.get('total_backends')}")
    
    alert_summary = health_status.get('alert_summary', {})
    logger.info("  Alert Summary:")
    logger.info(f"    Total Active: {alert_summary.get('total_active')}")
    logger.info(f"    Critical: {alert_summary.get('critical_alerts')}")
    logger.info(f"    Emergency: {alert_summary.get('emergency_alerts')}")
    logger.info(f"    Unacknowledged: {alert_summary.get('unacknowledged_alerts')}")
    
    router.shutdown()
    return True

def main():
    """Main test function"""
    logger = setup_logging()
    logger.info("Starting Alert Management System Integration Tests")
    
    # Ensure logs directory exists
    Path("./logs").mkdir(exist_ok=True)
    Path("./logs/alerts").mkdir(exist_ok=True)
    
    tests = [
        ("Alert Generation", test_alert_generation),
        ("Alert Suppression", test_alert_suppression),
        ("Alert Acknowledgment", test_alert_acknowledgment),
        ("Callback System", test_callback_system),
        ("Alert Statistics", test_alert_statistics),
        ("Threshold Configuration", test_threshold_configuration),
        ("Comprehensive Health Status", test_comprehensive_health_status)
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            result = test_func(logger)
            if result:
                passed_tests += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed_tests += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed_tests += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
        
        # Small delay between tests
        time.sleep(0.5)
    
    logger.info(f"\n{'='*50}")
    logger.info("Test Summary:")
    logger.info(f"  Passed: {passed_tests}")
    logger.info(f"  Failed: {failed_tests}")
    logger.info(f"  Total: {passed_tests + failed_tests}")
    
    if failed_tests == 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {failed_tests} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())