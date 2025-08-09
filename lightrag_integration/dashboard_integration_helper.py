"""
Dashboard Integration Helper for Unified System Health Dashboard
===============================================================

This module provides helper functions to easily integrate the Unified System Health Dashboard
with existing monitoring systems and production deployments. It simplifies the setup process
and provides production-ready configuration templates.

Key Features:
1. Automatic system discovery and integration
2. Configuration templates for different deployment scenarios  
3. Health check and validation utilities
4. Production deployment helpers
5. Docker and container integration support

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
Task: CMO-LIGHTRAG-014-T07 - Dashboard Integration Support
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import yaml

from .unified_system_health_dashboard import (
    UnifiedSystemHealthDashboard,
    DashboardConfig,
    create_unified_dashboard
)

# Import monitoring systems for integration
try:
    from .graceful_degradation_integration import (
        GracefulDegradationOrchestrator,
        create_and_start_graceful_degradation_system
    )
    GRACEFUL_DEGRADATION_AVAILABLE = True
except ImportError:
    GRACEFUL_DEGRADATION_AVAILABLE = False

try:
    from .production_monitoring import ProductionMonitoring
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False


# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

@dataclass
class DashboardDeploymentConfig:
    """Complete configuration for dashboard deployment scenarios."""
    
    # Deployment type
    deployment_type: str = "development"  # development, staging, production
    
    # Dashboard configuration
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8092
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Monitoring system configuration
    enable_graceful_degradation: bool = True
    graceful_degradation_config: Dict[str, Any] = None
    
    # Database and persistence
    enable_database: bool = True
    database_path: str = "unified_health_dashboard.db"
    retention_hours: int = 72
    
    # WebSocket and real-time features
    enable_websockets: bool = True
    websocket_update_interval: float = 2.0
    
    # Security and access control
    enable_api_key: bool = False
    api_key: Optional[str] = None
    enable_cors: bool = True
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 300
    enable_email_alerts: bool = False
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_recipients: List[str] = None
    
    # Performance tuning
    max_websocket_connections: int = 100
    max_historical_snapshots: int = 5000
    max_alert_history: int = 1000
    
    def __post_init__(self):
        if self.graceful_degradation_config is None:
            self.graceful_degradation_config = {}
        if self.alert_recipients is None:
            self.alert_recipients = []
    
    def to_dashboard_config(self) -> DashboardConfig:
        """Convert to DashboardConfig for the dashboard."""
        return DashboardConfig(
            host=self.dashboard_host,
            port=self.dashboard_port,
            enable_cors=self.enable_cors,
            enable_ssl=self.enable_ssl,
            ssl_cert_path=self.ssl_cert_path,
            ssl_key_path=self.ssl_key_path,
            enable_websockets=self.enable_websockets,
            websocket_update_interval=self.websocket_update_interval,
            enable_historical_data=self.enable_database,
            historical_retention_hours=self.retention_hours,
            db_path=self.database_path,
            enable_db_persistence=self.enable_database,
            enable_alerts=self.enable_alerts,
            alert_cooldown_seconds=self.alert_cooldown_seconds,
            enable_email_alerts=self.enable_email_alerts,
            enable_api_key=self.enable_api_key,
            api_key=self.api_key
        )


def get_development_config() -> DashboardDeploymentConfig:
    """Get configuration for development environment."""
    return DashboardDeploymentConfig(
        deployment_type="development",
        dashboard_port=8092,
        enable_ssl=False,
        enable_database=True,
        retention_hours=24,
        websocket_update_interval=1.0,  # More frequent updates for development
        enable_alerts=True,
        alert_cooldown_seconds=60,  # Shorter cooldown for testing
        enable_email_alerts=False
    )


def get_staging_config() -> DashboardDeploymentConfig:
    """Get configuration for staging environment."""
    return DashboardDeploymentConfig(
        deployment_type="staging",
        dashboard_port=8092,
        enable_ssl=True,
        ssl_cert_path="/etc/ssl/certs/dashboard.crt",
        ssl_key_path="/etc/ssl/private/dashboard.key",
        enable_database=True,
        retention_hours=48,
        websocket_update_interval=2.0,
        enable_alerts=True,
        alert_cooldown_seconds=300,
        enable_email_alerts=True,
        enable_api_key=True
    )


def get_production_config() -> DashboardDeploymentConfig:
    """Get configuration for production environment."""
    return DashboardDeploymentConfig(
        deployment_type="production",
        dashboard_port=8092,
        enable_ssl=True,
        ssl_cert_path="/etc/ssl/certs/dashboard.crt",
        ssl_key_path="/etc/ssl/private/dashboard.key",
        enable_database=True,
        retention_hours=72,
        websocket_update_interval=5.0,  # Less frequent updates for production
        enable_alerts=True,
        alert_cooldown_seconds=600,  # Longer cooldown to prevent spam
        enable_email_alerts=True,
        enable_api_key=True,
        max_websocket_connections=50,  # Limit connections in production
        graceful_degradation_config={
            "monitoring_interval": 5.0,
            "base_rate_per_second": 100.0,
            "max_queue_size": 1000,
            "max_concurrent_requests": 200
        }
    )


# ============================================================================
# INTEGRATION HELPER CLASS
# ============================================================================

class DashboardIntegrationHelper:
    """Helper class for integrating the dashboard with existing systems."""
    
    def __init__(self, config: Optional[DashboardDeploymentConfig] = None):
        self.config = config or get_development_config()
        self.logger = logging.getLogger(__name__)
        
        # Discovered systems
        self.discovered_systems: Dict[str, Any] = {}
        self.integration_status: Dict[str, bool] = {}
    
    async def discover_monitoring_systems(self) -> Dict[str, bool]:
        """
        Automatically discover available monitoring systems.
        
        Returns:
            Dict mapping system names to availability status
        """
        discovery_results = {}
        
        # Check for Graceful Degradation Orchestrator
        if GRACEFUL_DEGRADATION_AVAILABLE:
            try:
                # Try to import and create a test instance
                from .graceful_degradation_integration import create_graceful_degradation_system
                test_orchestrator = create_graceful_degradation_system()
                discovery_results["graceful_degradation"] = True
                self.discovered_systems["graceful_degradation"] = test_orchestrator
                self.logger.info("✅ Discovered Graceful Degradation Orchestrator")
            except Exception as e:
                discovery_results["graceful_degradation"] = False
                self.logger.warning(f"⚠️ Graceful Degradation Orchestrator not available: {e}")
        else:
            discovery_results["graceful_degradation"] = False
            self.logger.warning("⚠️ Graceful degradation module not available")
        
        # Check for Enhanced Load Detection System
        try:
            from .enhanced_load_monitoring_system import create_enhanced_load_monitoring_system
            test_detector = create_enhanced_load_monitoring_system()
            discovery_results["enhanced_load_detection"] = True
            self.discovered_systems["enhanced_load_detection"] = test_detector
            self.logger.info("✅ Discovered Enhanced Load Detection System")
        except Exception as e:
            discovery_results["enhanced_load_detection"] = False
            self.logger.warning(f"⚠️ Enhanced Load Detection System not available: {e}")
        
        # Check for Progressive Service Degradation Controller
        try:
            from .progressive_service_degradation_controller import create_progressive_degradation_controller
            test_controller = create_progressive_degradation_controller()
            discovery_results["degradation_controller"] = True
            self.discovered_systems["degradation_controller"] = test_controller
            self.logger.info("✅ Discovered Progressive Service Degradation Controller")
        except Exception as e:
            discovery_results["degradation_controller"] = False
            self.logger.warning(f"⚠️ Progressive Service Degradation Controller not available: {e}")
        
        # Check for Circuit Breaker Monitoring
        try:
            from .circuit_breaker_monitoring_integration import create_monitoring_integration
            test_cb_monitor = create_monitoring_integration()
            discovery_results["circuit_breaker_monitoring"] = True
            self.discovered_systems["circuit_breaker_monitoring"] = test_cb_monitor
            self.logger.info("✅ Discovered Circuit Breaker Monitoring")
        except Exception as e:
            discovery_results["circuit_breaker_monitoring"] = False
            self.logger.warning(f"⚠️ Circuit Breaker Monitoring not available: {e}")
        
        # Check for Production Monitoring
        if PRODUCTION_MONITORING_AVAILABLE:
            try:
                test_prod_monitor = ProductionMonitoring()
                discovery_results["production_monitoring"] = True
                self.discovered_systems["production_monitoring"] = test_prod_monitor
                self.logger.info("✅ Discovered Production Monitoring")
            except Exception as e:
                discovery_results["production_monitoring"] = False
                self.logger.warning(f"⚠️ Production Monitoring not available: {e}")
        else:
            discovery_results["production_monitoring"] = False
            self.logger.warning("⚠️ Production monitoring module not available")
        
        total_discovered = sum(discovery_results.values())
        self.logger.info(f"System discovery complete: {total_discovered}/{len(discovery_results)} systems available")
        
        return discovery_results
    
    async def create_integrated_orchestrator(self) -> Optional[Any]:
        """
        Create an integrated graceful degradation orchestrator with all discovered systems.
        
        Returns:
            Configured orchestrator or None if creation fails
        """
        if not GRACEFUL_DEGRADATION_AVAILABLE:
            self.logger.error("Cannot create orchestrator: graceful degradation not available")
            return None
        
        try:
            from .graceful_degradation_integration import (
                create_and_start_graceful_degradation_system,
                GracefulDegradationConfig
            )
            
            # Create configuration from deployment config
            gd_config = GracefulDegradationConfig(**self.config.graceful_degradation_config)
            
            # Get production systems from discovered systems
            production_systems = {}
            if "production_monitoring" in self.discovered_systems:
                production_systems["monitoring_system"] = self.discovered_systems["production_monitoring"]
            
            # Create and start the orchestrator
            orchestrator = await create_and_start_graceful_degradation_system(
                config=gd_config,
                **production_systems
            )
            
            self.logger.info("✅ Created integrated graceful degradation orchestrator")
            return orchestrator
            
        except Exception as e:
            self.logger.error(f"Failed to create integrated orchestrator: {e}")
            return None
    
    async def validate_dashboard_setup(self, dashboard: UnifiedSystemHealthDashboard) -> Dict[str, Any]:
        """
        Validate that the dashboard is properly configured and integrated.
        
        Args:
            dashboard: The dashboard instance to validate
            
        Returns:
            Validation report with status and issues
        """
        validation_report = {
            "overall_status": "unknown",
            "checks": {},
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Check data aggregator
            if dashboard.data_aggregator:
                validation_report["checks"]["data_aggregator"] = "✅ Present"
                
                # Check if any monitoring systems are registered
                has_systems = any([
                    dashboard.data_aggregator.graceful_degradation_orchestrator,
                    dashboard.data_aggregator.enhanced_load_detector,
                    dashboard.data_aggregator.degradation_controller,
                    dashboard.data_aggregator.circuit_breaker_monitor,
                    dashboard.data_aggregator.production_monitor
                ])
                
                if has_systems:
                    validation_report["checks"]["monitoring_systems"] = "✅ Registered"
                else:
                    validation_report["checks"]["monitoring_systems"] = "⚠️ None registered"
                    validation_report["warnings"].append("No monitoring systems registered")
            else:
                validation_report["checks"]["data_aggregator"] = "❌ Missing"
                validation_report["errors"].append("Data aggregator not found")
            
            # Check WebSocket manager
            if dashboard.websocket_manager:
                validation_report["checks"]["websocket_manager"] = "✅ Present"
            else:
                validation_report["checks"]["websocket_manager"] = "❌ Missing"
                validation_report["errors"].append("WebSocket manager not found")
            
            # Check web application
            if dashboard.app:
                validation_report["checks"]["web_app"] = f"✅ {dashboard.framework.upper()}"
            else:
                validation_report["checks"]["web_app"] = "❌ Not initialized"
                validation_report["errors"].append("Web application not initialized")
            
            # Check database setup if enabled
            if dashboard.config.enable_db_persistence:
                db_path = Path(dashboard.config.db_path)
                if db_path.exists():
                    validation_report["checks"]["database"] = "✅ Database file exists"
                else:
                    validation_report["checks"]["database"] = "⚠️ Database file not found"
                    validation_report["warnings"].append("Database file not found (will be created)")
            else:
                validation_report["checks"]["database"] = "ℹ️ Disabled"
            
            # Check SSL configuration if enabled
            if dashboard.config.enable_ssl:
                cert_exists = Path(dashboard.config.ssl_cert_path).exists() if dashboard.config.ssl_cert_path else False
                key_exists = Path(dashboard.config.ssl_key_path).exists() if dashboard.config.ssl_key_path else False
                
                if cert_exists and key_exists:
                    validation_report["checks"]["ssl"] = "✅ Certificates found"
                else:
                    validation_report["checks"]["ssl"] = "❌ Certificates missing"
                    validation_report["errors"].append("SSL enabled but certificate files not found")
            else:
                validation_report["checks"]["ssl"] = "ℹ️ Disabled"
            
            # Generate recommendations
            if validation_report["warnings"] or validation_report["errors"]:
                if not has_systems:
                    validation_report["recommendations"].append(
                        "Register monitoring systems using dashboard.data_aggregator.register_monitoring_systems()"
                    )
                
                if validation_report["errors"]:
                    validation_report["recommendations"].append(
                        "Resolve errors before deploying to production"
                    )
                
                if dashboard.config.enable_ssl and not (cert_exists and key_exists):
                    validation_report["recommendations"].append(
                        "Generate SSL certificates for secure deployment"
                    )
            
            # Determine overall status
            if validation_report["errors"]:
                validation_report["overall_status"] = "error"
            elif validation_report["warnings"]:
                validation_report["overall_status"] = "warning"
            else:
                validation_report["overall_status"] = "healthy"
            
        except Exception as e:
            validation_report["overall_status"] = "error"
            validation_report["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_report
    
    def generate_config_file(self, file_path: str, format: str = "yaml") -> bool:
        """
        Generate a configuration file for the dashboard.
        
        Args:
            file_path: Path to save the configuration file
            format: Format for the file ("yaml" or "json")
            
        Returns:
            True if file was generated successfully
        """
        try:
            config_data = asdict(self.config)
            
            if format.lower() == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration file generated: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate configuration file: {e}")
            return False
    
    def load_config_file(self, file_path: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            True if configuration was loaded successfully
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"Configuration file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    self.logger.error(f"Unsupported configuration file format: {file_path.suffix}")
                    return False
            
            # Update configuration
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.logger.info(f"Configuration loaded from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            return False
    
    async def deploy_dashboard(self) -> Tuple[Optional[UnifiedSystemHealthDashboard], Dict[str, Any]]:
        """
        Deploy a fully integrated dashboard with all available systems.
        
        Returns:
            Tuple of (dashboard_instance, deployment_report)
        """
        deployment_report = {
            "status": "unknown",
            "steps_completed": [],
            "errors": [],
            "warnings": [],
            "dashboard_info": {}
        }
        
        try:
            # Step 1: Discover monitoring systems
            self.logger.info("🔍 Discovering monitoring systems...")
            discovery_results = await self.discover_monitoring_systems()
            deployment_report["steps_completed"].append("System discovery")
            deployment_report["discovered_systems"] = discovery_results
            
            # Step 2: Create integrated orchestrator
            orchestrator = None
            if self.config.enable_graceful_degradation and discovery_results.get("graceful_degradation"):
                self.logger.info("🚀 Creating integrated orchestrator...")
                orchestrator = await self.create_integrated_orchestrator()
                if orchestrator:
                    deployment_report["steps_completed"].append("Orchestrator creation")
                else:
                    deployment_report["warnings"].append("Failed to create orchestrator")
            
            # Step 3: Create dashboard
            self.logger.info("📊 Creating unified dashboard...")
            dashboard_config = self.config.to_dashboard_config()
            dashboard = create_unified_dashboard(
                config=dashboard_config,
                graceful_degradation_orchestrator=orchestrator
            )
            deployment_report["steps_completed"].append("Dashboard creation")
            
            # Step 4: Validate setup
            self.logger.info("✅ Validating dashboard setup...")
            validation_report = await self.validate_dashboard_setup(dashboard)
            deployment_report["validation"] = validation_report
            
            if validation_report["overall_status"] == "error":
                deployment_report["status"] = "error"
                deployment_report["errors"].extend(validation_report["errors"])
                return None, deployment_report
            elif validation_report["overall_status"] == "warning":
                deployment_report["warnings"].extend(validation_report["warnings"])
            
            # Step 5: Prepare dashboard information
            deployment_report["dashboard_info"] = {
                "url": f"http{'s' if dashboard_config.enable_ssl else ''}://{dashboard_config.host}:{dashboard_config.port}",
                "websocket_url": f"ws{'s' if dashboard_config.enable_ssl else ''}://{dashboard_config.host}:{dashboard_config.port}{dashboard_config.websocket_endpoint}",
                "api_base_url": f"http{'s' if dashboard_config.enable_ssl else ''}://{dashboard_config.host}:{dashboard_config.port}{dashboard_config.api_prefix}",
                "framework": dashboard.framework,
                "config": {
                    "deployment_type": self.config.deployment_type,
                    "ssl_enabled": dashboard_config.enable_ssl,
                    "websockets_enabled": dashboard_config.enable_websockets,
                    "database_enabled": dashboard_config.enable_db_persistence,
                    "alerts_enabled": dashboard_config.enable_alerts
                }
            }
            
            deployment_report["status"] = "success"
            self.logger.info("✅ Dashboard deployment preparation completed successfully")
            
            return dashboard, deployment_report
            
        except Exception as e:
            deployment_report["status"] = "error"
            deployment_report["errors"].append(f"Deployment failed: {str(e)}")
            self.logger.error(f"Dashboard deployment failed: {e}")
            return None, deployment_report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def quick_start_dashboard(
    deployment_type: str = "development",
    port: int = 8092,
    enable_all_features: bool = True
) -> Tuple[Optional[UnifiedSystemHealthDashboard], Dict[str, Any]]:
    """
    Quick start function to deploy a dashboard with minimal configuration.
    
    Args:
        deployment_type: "development", "staging", or "production"
        port: Port to run the dashboard on
        enable_all_features: Whether to enable all available features
        
    Returns:
        Tuple of (dashboard_instance, deployment_report)
    """
    # Get base configuration for deployment type
    if deployment_type == "development":
        config = get_development_config()
    elif deployment_type == "staging":
        config = get_staging_config()
    elif deployment_type == "production":
        config = get_production_config()
    else:
        config = get_development_config()
    
    # Override port
    config.dashboard_port = port
    
    # Disable features if requested
    if not enable_all_features:
        config.enable_alerts = False
        config.enable_email_alerts = False
        config.enable_database = False
    
    # Create integration helper and deploy
    helper = DashboardIntegrationHelper(config)
    return await helper.deploy_dashboard()


def generate_docker_compose(
    config: DashboardDeploymentConfig,
    output_path: str = "docker-compose.yml"
) -> bool:
    """
    Generate a Docker Compose file for the dashboard deployment.
    
    Args:
        config: Deployment configuration
        output_path: Path to save the Docker Compose file
        
    Returns:
        True if file was generated successfully
    """
    try:
        docker_compose = {
            "version": "3.8",
            "services": {
                "unified-health-dashboard": {
                    "build": ".",
                    "ports": [f"{config.dashboard_port}:{config.dashboard_port}"],
                    "environment": {
                        "DASHBOARD_HOST": config.dashboard_host,
                        "DASHBOARD_PORT": str(config.dashboard_port),
                        "DEPLOYMENT_TYPE": config.deployment_type,
                        "ENABLE_SSL": str(config.enable_ssl).lower(),
                        "ENABLE_DATABASE": str(config.enable_database).lower(),
                        "ENABLE_ALERTS": str(config.enable_alerts).lower()
                    },
                    "volumes": [
                        "./data:/app/data",
                        "./logs:/app/logs"
                    ],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": [
                            "CMD", "curl", "-f", 
                            f"http://localhost:{config.dashboard_port}/api/v1/health"
                        ],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                        "start_period": "40s"
                    }
                }
            }
        }
        
        # Add SSL volume mapping if enabled
        if config.enable_ssl:
            docker_compose["services"]["unified-health-dashboard"]["volumes"].extend([
                "/etc/ssl/certs:/etc/ssl/certs:ro",
                "/etc/ssl/private:/etc/ssl/private:ro"
            ])
        
        # Save Docker Compose file
        with open(output_path, 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False, indent=2)
        
        logging.info(f"Docker Compose file generated: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to generate Docker Compose file: {e}")
        return False


def generate_dockerfile(output_path: str = "Dockerfile") -> bool:
    """
    Generate a Dockerfile for the dashboard.
    
    Args:
        output_path: Path to save the Dockerfile
        
    Returns:
        True if file was generated successfully
    """
    dockerfile_content = '''
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

# Expose dashboard port
EXPOSE 8092

# Set environment variables
ENV PYTHONPATH=/app
ENV DASHBOARD_HOST=0.0.0.0
ENV DASHBOARD_PORT=8092

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:$DASHBOARD_PORT/api/v1/health || exit 1

# Run the dashboard
CMD ["python", "-m", "lightrag_integration.unified_system_health_dashboard"]
'''.strip()
    
    try:
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        logging.info(f"Dockerfile generated: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to generate Dockerfile: {e}")
        return False


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Dashboard Integration Helper")
    parser.add_argument(
        "--deployment-type",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment type"
    )
    parser.add_argument("--port", type=int, default=8092, help="Dashboard port")
    parser.add_argument(
        "--config-file",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--generate-config",
        help="Generate configuration file at specified path"
    )
    parser.add_argument(
        "--generate-docker",
        action="store_true",
        help="Generate Docker configuration files"
    )
    parser.add_argument(
        "--start-dashboard",
        action="store_true",
        help="Start the dashboard after setup"
    )
    
    args = parser.parse_args()
    
    async def main():
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create integration helper
        if args.config_file:
            # Load configuration from file
            helper = DashboardIntegrationHelper()
            if helper.load_config_file(args.config_file):
                print(f"✅ Configuration loaded from {args.config_file}")
            else:
                print(f"❌ Failed to load configuration from {args.config_file}")
                sys.exit(1)
        else:
            # Use default configuration for deployment type
            if args.deployment_type == "development":
                config = get_development_config()
            elif args.deployment_type == "staging":
                config = get_staging_config()
            else:
                config = get_production_config()
            
            config.dashboard_port = args.port
            helper = DashboardIntegrationHelper(config)
        
        # Generate configuration file if requested
        if args.generate_config:
            if helper.generate_config_file(args.generate_config):
                print(f"✅ Configuration file generated: {args.generate_config}")
            else:
                print(f"❌ Failed to generate configuration file")
                sys.exit(1)
        
        # Generate Docker files if requested
        if args.generate_docker:
            if generate_dockerfile():
                print("✅ Dockerfile generated")
            
            if generate_docker_compose(helper.config):
                print("✅ Docker Compose file generated")
        
        # Deploy dashboard if requested
        if args.start_dashboard:
            print(f"🚀 Deploying dashboard ({args.deployment_type} mode)...")
            dashboard, report = await helper.deploy_dashboard()
            
            if report["status"] == "success":
                print("✅ Dashboard deployment successful!")
                print(f"   URL: {report['dashboard_info']['url']}")
                print(f"   WebSocket: {report['dashboard_info']['websocket_url']}")
                
                if dashboard:
                    try:
                        print("🌐 Starting dashboard server...")
                        await dashboard.start()
                    except KeyboardInterrupt:
                        print("\n🛑 Dashboard stopped by user")
                        await dashboard.stop()
            else:
                print("❌ Dashboard deployment failed!")
                if report["errors"]:
                    for error in report["errors"]:
                        print(f"   Error: {error}")
                sys.exit(1)
    
    # Run the main function
    asyncio.run(main())