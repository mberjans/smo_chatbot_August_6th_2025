#!/usr/bin/env python3
"""
Alert Management System Usage Demonstration

This script demonstrates how to use the Clinical Metabolomics Oracle Alert Management System
in production scenarios.

Key Features Demonstrated:
1. Configuring custom alert thresholds
2. Registering different types of alert callbacks
3. Monitoring system health with alerts
4. Managing active alerts (acknowledge/resolve)
5. Analyzing alert history and statistics
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Add the project directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightrag_integration.intelligent_query_router import (
    IntelligentQueryRouter,
    HealthCheckConfig,
    AlertThresholds,
    LoadBalancingConfig
)

def setup_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("AlertSystemDemo")

def demo_alert_configuration(logger):
    """Demonstrate alert threshold configuration"""
    logger.info("=== Alert Configuration Demo ===")
    
    # Create custom alert thresholds for production environment
    production_thresholds = {
        # Response time thresholds (ms) - stricter for production
        'response_time_warning': 500.0,
        'response_time_critical': 1000.0,
        'response_time_emergency': 3000.0,
        
        # Error rate thresholds - tighter control
        'error_rate_warning': 0.01,    # 1%
        'error_rate_critical': 0.05,   # 5%
        'error_rate_emergency': 0.10,  # 10%
        
        # Availability thresholds - high availability requirements
        'availability_warning': 99.0,  # 99%
        'availability_critical': 98.0, # 98%
        'availability_emergency': 95.0, # 95%
        
        # Health score thresholds
        'health_score_warning': 90.0,
        'health_score_critical': 75.0,
        'health_score_emergency': 50.0,
        
        # Resource usage thresholds
        'cpu_usage_warning': 60.0,
        'cpu_usage_critical': 80.0,
        'cpu_usage_emergency': 95.0,
        
        'memory_usage_warning': 70.0,
        'memory_usage_critical': 85.0,
        'memory_usage_emergency': 95.0
    }
    
    # Initialize router with production configuration
    health_config = HealthCheckConfig(
        timeout_seconds=10.0,
        retry_attempts=3,
        alert_thresholds=AlertThresholds(**production_thresholds)
    )
    
    load_config = LoadBalancingConfig(
        strategy="health_aware",
        health_check_interval=30,
        circuit_breaker_threshold=3,
        enable_adaptive_routing=True
    )
    
    router = IntelligentQueryRouter(
        health_check_config=health_config,
        load_balancing_config=load_config
    )
    
    logger.info("✅ Intelligent router initialized with production alert thresholds")
    
    # Update thresholds dynamically
    updated_thresholds = {
        'response_time_warning': 400.0,  # Even stricter
        'error_rate_warning': 0.005     # 0.5%
    }
    
    success = router.configure_alert_thresholds(updated_thresholds)
    logger.info(f"✅ Dynamic threshold update successful: {success}")
    
    return router

def demo_alert_callbacks(router, logger):
    """Demonstrate different alert callback configurations"""
    logger.info("=== Alert Callback Configuration Demo ===")
    
    # 1. Register webhook for Slack/Teams integration
    webhook_success = router.register_alert_callback(
        "webhook",
        webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
        timeout=5.0,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_TOKEN"
        }
    )
    logger.info(f"📡 Webhook callback registered: {webhook_success}")
    
    # 2. Register JSON file for persistent storage
    json_success = router.register_alert_callback(
        "json_file",
        file_path="./logs/alerts/production_alerts.json",
        max_alerts=50000  # Large capacity for production
    )
    logger.info(f"💾 JSON file callback registered: {json_success}")
    
    # 3. Console callback is already registered by default
    console_success = router.register_alert_callback("console")
    logger.info(f"🖥️  Console callback registered: {console_success}")
    
    logger.info("✅ All alert callbacks configured")

def demo_system_monitoring(router, logger):
    """Demonstrate system health monitoring with alerts"""
    logger.info("=== System Health Monitoring Demo ===")
    
    # Monitor system health for a short period
    monitoring_duration = 10  # seconds
    logger.info(f"🔍 Monitoring system health for {monitoring_duration} seconds...")
    
    start_time = time.time()
    while (time.time() - start_time) < monitoring_duration:
        # Get comprehensive health status
        health_status = router.get_system_health_with_alerts()
        
        logger.info(f"System Status: {health_status['overall_status']}")
        
        alert_summary = health_status.get('alert_summary', {})
        if alert_summary.get('total_active', 0) > 0:
            logger.warning(f"🚨 Active Alerts: {alert_summary['total_active']} "
                         f"(Critical: {alert_summary['critical_alerts']}, "
                         f"Emergency: {alert_summary['emergency_alerts']})")
        
        time.sleep(2)
    
    logger.info("✅ Health monitoring completed")

def demo_alert_management(router, logger):
    """Demonstrate alert management operations"""
    logger.info("=== Alert Management Demo ===")
    
    # Get current active alerts
    active_alerts = router.get_active_alerts()
    logger.info(f"📋 Current active alerts: {len(active_alerts)}")
    
    if active_alerts:
        # Demonstrate alert acknowledgment
        alert_to_ack = active_alerts[0]
        alert_id = alert_to_ack['id']
        
        ack_success = router.acknowledge_alert(alert_id, "production_admin")
        logger.info(f"✅ Alert acknowledged: {alert_id} -> {ack_success}")
        
        # Demonstrate alert resolution (if multiple alerts exist)
        if len(active_alerts) > 1:
            alert_to_resolve = active_alerts[1]
            resolve_id = alert_to_resolve['id']
            
            resolve_success = router.resolve_alert(resolve_id, "production_admin")
            logger.info(f"✅ Alert resolved: {resolve_id} -> {resolve_success}")
    
    # Get alert statistics
    stats = router.get_alert_statistics()
    if not stats.get('no_data', False):
        logger.info("📊 Alert Statistics:")
        logger.info(f"  Total alerts: {stats.get('total_alerts', 0)}")
        logger.info(f"  Active alerts: {stats.get('active_alerts', 0)}")
        logger.info(f"  Acknowledgment rate: {stats.get('acknowledgment_rate', 0):.2%}")
        logger.info(f"  Auto-recovery count: {stats.get('auto_recovery_count', 0)}")
        
        severity_dist = stats.get('severity_distribution', {})
        if severity_dist:
            logger.info("  Severity distribution:")
            for severity, count in severity_dist.items():
                logger.info(f"    {severity}: {count}")

def demo_alert_history(router, logger):
    """Demonstrate alert history analysis"""
    logger.info("=== Alert History Analysis Demo ===")
    
    # Get recent alert history
    recent_alerts = router.get_alert_history(limit=20)
    logger.info(f"📚 Recent alert history: {len(recent_alerts)} alerts")
    
    if recent_alerts:
        # Analyze alert patterns
        severity_counts = {}
        backend_counts = {}
        threshold_counts = {}
        
        for alert in recent_alerts:
            severity = alert.get('severity', 'unknown')
            backend = alert.get('backend_type', 'unknown')
            threshold = alert.get('threshold_breached', 'unknown')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            threshold_counts[threshold] = threshold_counts.get(threshold, 0) + 1
        
        logger.info("🔍 Alert Pattern Analysis:")
        logger.info(f"  Most frequent severity: {max(severity_counts, key=severity_counts.get)}")
        logger.info(f"  Most affected backend: {max(backend_counts, key=backend_counts.get)}")
        logger.info(f"  Most breached threshold: {max(threshold_counts, key=threshold_counts.get)}")
        
        # Show recent critical/emergency alerts
        critical_alerts = [a for a in recent_alerts 
                         if a.get('severity') in ['critical', 'emergency']]
        
        if critical_alerts:
            logger.warning(f"⚠️  Recent critical/emergency alerts: {len(critical_alerts)}")
            for alert in critical_alerts[:3]:  # Show top 3
                logger.warning(f"  - {alert['timestamp']}: {alert['message']}")
    
    # Get alerts from specific time period (last hour)
    one_hour_ago = datetime.now() - timedelta(hours=1)
    hour_alerts = router.get_alert_history(
        limit=100,
        start_time=one_hour_ago
    )
    logger.info(f"⏰ Alerts in last hour: {len(hour_alerts)}")

def demo_production_recommendations(logger):
    """Provide production deployment recommendations"""
    logger.info("=== Production Deployment Recommendations ===")
    
    recommendations = [
        "🔧 Tune alert thresholds based on your system's baseline performance",
        "📡 Configure webhook callbacks for Slack, Teams, or PagerDuty integration",
        "💾 Set up persistent JSON file storage with log rotation",
        "📊 Implement regular alert history analysis and trend monitoring",
        "🔄 Enable auto-recovery monitoring for self-healing capabilities",
        "⚡ Use health-aware load balancing for optimal performance",
        "🎯 Set up different alert thresholds for different environments (dev/staging/prod)",
        "📈 Monitor alert statistics to identify system performance trends",
        "🚨 Configure escalation policies for unacknowledged critical alerts",
        "🔍 Regularly review and adjust alert suppression rules to avoid noise"
    ]
    
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    logger.info("\n💡 Example Integration Code:")
    logger.info("""
    # Production setup example:
    router = IntelligentQueryRouter(
        health_check_config=HealthCheckConfig(
            timeout_seconds=10.0,
            alert_thresholds=AlertThresholds(
                response_time_warning=500.0,
                error_rate_warning=0.01,
                availability_warning=99.0
            )
        ),
        load_balancing_config=LoadBalancingConfig(
            strategy="health_aware",
            enable_adaptive_routing=True
        )
    )
    
    # Register production callbacks
    router.register_alert_callback("webhook", 
                                   webhook_url="YOUR_SLACK_WEBHOOK")
    router.register_alert_callback("json_file", 
                                   file_path="/var/log/cmo/alerts.json")
    
    # Monitor and manage alerts
    alerts = router.get_active_alerts(severity="critical")
    for alert in alerts:
        router.acknowledge_alert(alert['id'], "ops_team")
    """)

def main():
    """Main demonstration function"""
    logger = setup_logging()
    logger.info("🚀 Clinical Metabolomics Oracle - Alert System Demo")
    logger.info("=" * 60)
    
    try:
        # 1. Configure Alert System
        router = demo_alert_configuration(logger)
        
        # 2. Setup Alert Callbacks
        demo_alert_callbacks(router, logger)
        
        # 3. Monitor System Health
        demo_system_monitoring(router, logger)
        
        # 4. Manage Alerts
        demo_alert_management(router, logger)
        
        # 5. Analyze Alert History
        demo_alert_history(router, logger)
        
        # 6. Production Recommendations
        demo_production_recommendations(logger)
        
        logger.info("=" * 60)
        logger.info("🎉 Alert System Demo Completed Successfully!")
        logger.info("The alert management system is ready for production deployment.")
        
        # Cleanup
        router.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())