"""
Budget Management Integration Module for Clinical Metabolomics Oracle LightRAG Integration

This module provides a comprehensive integration layer that combines all budget management
components into a unified system. It serves as the main entry point for initializing and
managing the complete cost alerting and budget management infrastructure.

Classes:
    - BudgetManagementConfig: Configuration for the integrated budget management system
    - BudgetManagementSystem: Main integration class combining all components
    - BudgetManagementFactory: Factory for creating configured budget management systems

The integration system provides:
    - Unified initialization and configuration of all budget components
    - Coordinated lifecycle management (startup, monitoring, shutdown)
    - Integration with existing LightRAG infrastructure
    - Configuration-driven setup with environment variable support
    - Health monitoring and system status reporting
    - Graceful error handling and recovery mechanisms
"""

import os
import time
import threading
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from contextlib import contextmanager

from .config import LightRAGConfig
from .budget_manager import BudgetManager, BudgetThreshold, AlertLevel
from .api_metrics_logger import APIUsageMetricsLogger
from .cost_persistence import CostPersistence
from .alert_system import (
    AlertNotificationSystem, AlertEscalationManager, AlertConfig,
    EmailAlertConfig, WebhookAlertConfig, SlackAlertConfig, AlertChannel
)
from .realtime_budget_monitor import RealTimeBudgetMonitor
from .cost_based_circuit_breaker import (
    CostCircuitBreakerManager, CostThresholdRule, CostThresholdType
)
from .budget_dashboard import BudgetDashboardAPI
from .research_categorizer import ResearchCategorizer
from .audit_trail import AuditTrail


@dataclass
class BudgetManagementConfig:
    """Configuration for the integrated budget management system."""
    
    # Core budget settings
    daily_budget_limit: Optional[float] = None
    monthly_budget_limit: Optional[float] = None
    budget_thresholds: Optional[BudgetThreshold] = None
    
    # Alert system configuration
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["logging"])
    email_config: Optional[Dict[str, Any]] = None
    webhook_config: Optional[Dict[str, Any]] = None
    slack_config: Optional[Dict[str, Any]] = None
    
    # Monitoring configuration
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: float = 60.0
    enable_cost_projections: bool = True
    
    # Circuit breaker configuration
    enable_circuit_breakers: bool = True
    default_circuit_breaker_rules: bool = True
    custom_circuit_breaker_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dashboard configuration
    enable_dashboard: bool = True
    dashboard_cache_ttl_seconds: float = 60.0
    
    # Advanced features
    enable_escalation: bool = True
    enable_cost_optimization_suggestions: bool = True
    enable_predictive_alerts: bool = True
    
    # Integration settings
    integrate_with_existing_circuit_breakers: bool = True
    circuit_breaker_callback_integration: bool = True
    
    @classmethod
    def from_lightrag_config(cls, lightrag_config: LightRAGConfig) -> 'BudgetManagementConfig':
        """Create budget management config from LightRAG config."""
        return cls(
            daily_budget_limit=lightrag_config.daily_budget_limit,
            monthly_budget_limit=lightrag_config.monthly_budget_limit,
            enable_alerts=lightrag_config.enable_budget_alerts,
            enable_real_time_monitoring=True,  # Always enabled for comprehensive monitoring
            enable_circuit_breakers=True,     # Always enabled for protection
            enable_dashboard=True             # Always enabled for visibility
        )
    
    @classmethod
    def from_environment(cls) -> 'BudgetManagementConfig':
        """Create budget management config from environment variables."""
        return cls(
            daily_budget_limit=float(os.getenv("BUDGET_DAILY_LIMIT")) if os.getenv("BUDGET_DAILY_LIMIT") else None,
            monthly_budget_limit=float(os.getenv("BUDGET_MONTHLY_LIMIT")) if os.getenv("BUDGET_MONTHLY_LIMIT") else None,
            enable_alerts=os.getenv("BUDGET_ENABLE_ALERTS", "true").lower() == "true",
            alert_channels=os.getenv("BUDGET_ALERT_CHANNELS", "logging").split(","),
            enable_real_time_monitoring=os.getenv("BUDGET_ENABLE_MONITORING", "true").lower() == "true",
            monitoring_interval_seconds=float(os.getenv("BUDGET_MONITORING_INTERVAL", "60.0")),
            enable_circuit_breakers=os.getenv("BUDGET_ENABLE_CIRCUIT_BREAKERS", "true").lower() == "true",
            enable_dashboard=os.getenv("BUDGET_ENABLE_DASHBOARD", "true").lower() == "true"
        )


class BudgetManagementSystem:
    """
    Main integration class for the complete budget management system.
    
    This class coordinates all budget management components and provides a unified
    interface for budget monitoring, alerting, and cost protection.
    """
    
    def __init__(self,
                 lightrag_config: LightRAGConfig,
                 budget_config: Optional[BudgetManagementConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the integrated budget management system.
        
        Args:
            lightrag_config: LightRAG configuration
            budget_config: Budget management configuration
            logger: Logger instance for operations
        """
        self.lightrag_config = lightrag_config
        self.budget_config = budget_config or BudgetManagementConfig.from_lightrag_config(lightrag_config)
        self.logger = logger or logging.getLogger(__name__)
        
        # Component references
        self.cost_persistence: Optional[CostPersistence] = None
        self.budget_manager: Optional[BudgetManager] = None
        self.api_metrics_logger: Optional[APIUsageMetricsLogger] = None
        self.alert_system: Optional[AlertNotificationSystem] = None
        self.escalation_manager: Optional[AlertEscalationManager] = None
        self.real_time_monitor: Optional[RealTimeBudgetMonitor] = None
        self.circuit_breaker_manager: Optional[CostCircuitBreakerManager] = None
        self.dashboard_api: Optional[BudgetDashboardAPI] = None
        self.research_categorizer: Optional[ResearchCategorizer] = None
        self.audit_trail: Optional[AuditTrail] = None
        
        # System state
        self._initialized = False
        self._running = False
        self._start_time: Optional[float] = None
        self._shutdown_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Integration with existing system
        self._existing_circuit_breaker_callback: Optional[Callable] = None
        
        self.logger.info("Budget management system created")
    
    def initialize(self) -> None:
        """Initialize all budget management components."""
        with self._lock:
            if self._initialized:
                self.logger.warning("Budget management system already initialized")
                return
            
            try:
                self.logger.info("Initializing budget management system...")
                
                # Initialize core persistence layer
                self._initialize_cost_persistence()
                
                # Initialize budget manager
                self._initialize_budget_manager()
                
                # Initialize API metrics logger
                self._initialize_api_metrics_logger()
                
                # Initialize alert system
                if self.budget_config.enable_alerts:
                    self._initialize_alert_system()
                
                # Initialize real-time monitoring
                if self.budget_config.enable_real_time_monitoring:
                    self._initialize_real_time_monitor()
                
                # Initialize circuit breakers
                if self.budget_config.enable_circuit_breakers:
                    self._initialize_circuit_breakers()
                
                # Initialize dashboard
                if self.budget_config.enable_dashboard:
                    self._initialize_dashboard()
                
                # Initialize supporting components
                self._initialize_supporting_components()
                
                # Set up integrations
                self._setup_integrations()
                
                # Set up shutdown handlers
                self._setup_shutdown_handlers()
                
                self._initialized = True
                self.logger.info("Budget management system initialization completed")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize budget management system: {e}")
                self._cleanup_partial_initialization()
                raise
    
    def start(self) -> None:
        """Start the budget management system."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("System must be initialized before starting")
            
            if self._running:
                self.logger.warning("Budget management system already running")
                return
            
            try:
                self.logger.info("Starting budget management system...")
                
                # Start real-time monitoring
                if self.real_time_monitor:
                    self.real_time_monitor.start_monitoring(background=True)
                
                # Log system start
                if self.audit_trail:
                    self.audit_trail.record_event(
                        event_type="system_start",
                        event_data={
                            "timestamp": time.time(),
                            "components_enabled": self._get_enabled_components()
                        },
                        session_id="system"
                    )
                
                self._running = True
                self._start_time = time.time()
                
                self.logger.info("Budget management system started successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to start budget management system: {e}")
                raise
    
    def stop(self) -> None:
        """Stop the budget management system."""
        with self._lock:
            if not self._running:
                self.logger.warning("Budget management system not running")
                return
            
            try:
                self.logger.info("Stopping budget management system...")
                
                # Stop real-time monitoring
                if self.real_time_monitor:
                    self.real_time_monitor.stop_monitoring()
                
                # Close alert system
                if self.alert_system:
                    self.alert_system.close()
                
                # Close API metrics logger
                if self.api_metrics_logger:
                    self.api_metrics_logger.close()
                
                # Close circuit breaker manager
                if self.circuit_breaker_manager:
                    self.circuit_breaker_manager.close()
                
                # Log system stop
                if self.audit_trail:
                    self.audit_trail.record_event(
                        event_type="system_stop",
                        event_data={
                            "timestamp": time.time(),
                            "uptime_seconds": time.time() - (self._start_time or 0)
                        },
                        session_id="system"
                    )
                
                self._running = False
                
                self.logger.info("Budget management system stopped successfully")
                
            except Exception as e:
                self.logger.error(f"Error stopping budget management system: {e}")
                raise
    
    @contextmanager
    def managed_operation(self, operation_type: str, **kwargs):
        """
        Context manager for executing operations with full budget management.
        
        This provides complete integration with cost tracking, circuit breaking,
        and budget monitoring for any operation.
        
        Usage:
            with budget_system.managed_operation("llm_call", model="gpt-4") as tracker:
                result = make_api_call()
                tracker.record_cost(0.05)
                return result
        """
        if not self._running:
            raise RuntimeError("Budget management system not running")
        
        class OperationTracker:
            def __init__(self, system: 'BudgetManagementSystem'):
                self.system = system
                self.operation_id = f"op_{time.time()}_{id(self)}"
                self.start_time = time.time()
                self.cost_recorded = False
            
            def record_cost(self, cost: float, **metadata):
                """Record the actual cost of the operation."""
                if self.system.cost_persistence and not self.cost_recorded:
                    from .cost_persistence import CostRecord, ResearchCategory
                    
                    cost_record = CostRecord(
                        timestamp=self.start_time,
                        session_id=kwargs.get('session_id'),
                        operation_type=operation_type,
                        model_name=kwargs.get('model'),
                        cost_usd=cost,
                        prompt_tokens=metadata.get('prompt_tokens', 0),
                        completion_tokens=metadata.get('completion_tokens', 0),
                        total_tokens=metadata.get('total_tokens', 0),
                        research_category=metadata.get('research_category', ResearchCategory.GENERAL_QUERY.value),
                        success=True,
                        response_time_seconds=(time.time() - self.start_time),
                        user_id=kwargs.get('user_id'),
                        project_id=kwargs.get('project_id'),
                        metadata=metadata
                    )
                    
                    self.system.cost_persistence.record_cost(cost_record)
                    
                    # Update circuit breaker cost estimator
                    if self.system.circuit_breaker_manager:
                        self.system.circuit_breaker_manager.update_operation_cost(
                            "default", self.operation_id, cost, operation_type
                        )
                    
                    self.cost_recorded = True
        
        tracker = OperationTracker(self)
        
        try:
            yield tracker
        except Exception as e:
            # Record failed operation if cost tracking is available
            if self.cost_persistence and not tracker.cost_recorded:
                from .cost_persistence import CostRecord, ResearchCategory
                
                cost_record = CostRecord(
                    timestamp=tracker.start_time,
                    session_id=kwargs.get('session_id'),
                    operation_type=operation_type,
                    model_name=kwargs.get('model'),
                    cost_usd=0.0,
                    success=False,
                    error_type=type(e).__name__,
                    response_time_seconds=(time.time() - tracker.start_time),
                    user_id=kwargs.get('user_id'),
                    project_id=kwargs.get('project_id'),
                    metadata={'error_message': str(e)}
                )
                
                self.cost_persistence.record_cost(cost_record)
            
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            status = {
                'initialized': self._initialized,
                'running': self._running,
                'uptime_seconds': time.time() - (self._start_time or 0) if self._start_time else 0,
                'components': {},
                'budget_health': {},
                'alerts': {},
                'circuit_breakers': {},
                'monitoring': {},
                'timestamp': time.time()
            }
            
            # Component status
            status['components'] = {
                'cost_persistence': self.cost_persistence is not None,
                'budget_manager': self.budget_manager is not None,
                'api_metrics_logger': self.api_metrics_logger is not None,
                'alert_system': self.alert_system is not None,
                'real_time_monitor': self.real_time_monitor is not None,
                'circuit_breaker_manager': self.circuit_breaker_manager is not None,
                'dashboard_api': self.dashboard_api is not None
            }
            
            # Budget health
            if self.budget_manager:
                budget_summary = self.budget_manager.get_budget_summary()
                status['budget_health'] = budget_summary
            
            # Alert status
            if self.alert_system:
                alert_stats = self.alert_system.get_delivery_stats()
                status['alerts'] = alert_stats
            
            # Circuit breaker status
            if self.circuit_breaker_manager:
                cb_status = self.circuit_breaker_manager.get_system_status()
                status['circuit_breakers'] = cb_status.get('system_health', {})
            
            # Monitoring status
            if self.real_time_monitor:
                monitor_status = self.real_time_monitor.get_monitoring_status()
                status['monitoring'] = {
                    'active': monitor_status.get('monitoring_active', False),
                    'health_score': monitor_status.get('health_score', {}),
                    'recent_events': len(monitor_status.get('recent_events', []))
                }
            
            return status
    
    def get_dashboard_api(self) -> Optional[BudgetDashboardAPI]:
        """Get the dashboard API instance for external use."""
        return self.dashboard_api
    
    def get_circuit_breaker_manager(self) -> Optional[CostCircuitBreakerManager]:
        """Get the circuit breaker manager for external integration.""" 
        return self.circuit_breaker_manager
    
    def register_circuit_breaker_callback(self, callback: Callable) -> None:
        """Register callback for existing circuit breaker integration."""
        self._existing_circuit_breaker_callback = callback
        
        if self.real_time_monitor:
            self.real_time_monitor.set_circuit_breaker_callback(callback)
    
    def _initialize_cost_persistence(self) -> None:
        """Initialize cost persistence layer."""
        if not self.lightrag_config.cost_persistence_enabled:
            self.logger.info("Cost persistence disabled, skipping initialization")
            return
        
        self.cost_persistence = CostPersistence(
            db_path=str(self.lightrag_config.cost_db_path),
            logger=self.logger
        )
        
        # Initialize database
        self.cost_persistence.initialize_database()
        
        self.logger.info("Cost persistence initialized")
    
    def _initialize_budget_manager(self) -> None:
        """Initialize budget manager."""
        if not self.cost_persistence:
            self.logger.warning("Cost persistence not available, budget manager will have limited functionality")
            return
        
        # Create budget thresholds
        thresholds = self.budget_config.budget_thresholds or BudgetThreshold()
        
        self.budget_manager = BudgetManager(
            cost_persistence=self.cost_persistence,
            daily_budget_limit=self.budget_config.daily_budget_limit,
            monthly_budget_limit=self.budget_config.monthly_budget_limit,
            thresholds=thresholds,
            logger=self.logger
        )
        
        self.logger.info("Budget manager initialized")
    
    def _initialize_api_metrics_logger(self) -> None:
        """Initialize API metrics logger."""
        self.api_metrics_logger = APIUsageMetricsLogger(
            config=self.lightrag_config,
            cost_persistence=self.cost_persistence,
            budget_manager=self.budget_manager,
            logger=self.logger
        )
        
        self.logger.info("API metrics logger initialized")
    
    def _initialize_alert_system(self) -> None:
        """Initialize alert notification system."""
        # Create alert configuration
        enabled_channels = set()
        for channel_name in self.budget_config.alert_channels:
            try:
                channel = AlertChannel(channel_name.lower())
                enabled_channels.add(channel)
            except ValueError:
                self.logger.warning(f"Unknown alert channel: {channel_name}")
        
        alert_config = AlertConfig(
            enabled_channels=enabled_channels,
            enable_escalation=self.budget_config.enable_escalation
        )
        
        # Add email configuration if provided
        if self.budget_config.email_config:
            email_config = EmailAlertConfig(**self.budget_config.email_config)
            alert_config.email_config = email_config
        
        # Add webhook configuration if provided
        if self.budget_config.webhook_config:
            webhook_config = WebhookAlertConfig(**self.budget_config.webhook_config)
            alert_config.webhook_config = webhook_config
        
        # Add Slack configuration if provided
        if self.budget_config.slack_config:
            slack_config = SlackAlertConfig(**self.budget_config.slack_config)
            alert_config.slack_config = slack_config
        
        self.alert_system = AlertNotificationSystem(alert_config, self.logger)
        
        # Initialize escalation manager if enabled
        if self.budget_config.enable_escalation:
            self.escalation_manager = AlertEscalationManager(self.alert_system, self.logger)
        
        self.logger.info("Alert system initialized")
    
    def _initialize_real_time_monitor(self) -> None:
        """Initialize real-time budget monitor."""
        if not self.budget_manager:
            self.logger.warning("Budget manager not available, real-time monitor disabled")
            return
        
        self.real_time_monitor = RealTimeBudgetMonitor(
            budget_manager=self.budget_manager,
            api_metrics_logger=self.api_metrics_logger,
            cost_persistence=self.cost_persistence,
            alert_system=self.alert_system,
            escalation_manager=self.escalation_manager,
            monitoring_interval=self.budget_config.monitoring_interval_seconds,
            logger=self.logger
        )
        
        self.logger.info("Real-time budget monitor initialized")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize cost-based circuit breaker system."""
        if not self.budget_manager or not self.cost_persistence:
            self.logger.warning("Required components not available, circuit breakers disabled")
            return
        
        self.circuit_breaker_manager = CostCircuitBreakerManager(
            budget_manager=self.budget_manager,
            cost_persistence=self.cost_persistence,
            logger=self.logger
        )
        
        # Create default circuit breakers if enabled
        if self.budget_config.default_circuit_breaker_rules:
            self.circuit_breaker_manager.create_circuit_breaker("llm_operations")
            self.circuit_breaker_manager.create_circuit_breaker("embedding_operations")
        
        self.logger.info("Circuit breaker system initialized")
    
    def _initialize_dashboard(self) -> None:
        """Initialize budget dashboard API."""
        self.dashboard_api = BudgetDashboardAPI(
            budget_manager=self.budget_manager,
            api_metrics_logger=self.api_metrics_logger,
            cost_persistence=self.cost_persistence,
            alert_system=self.alert_system,
            escalation_manager=self.escalation_manager,
            real_time_monitor=self.real_time_monitor,
            circuit_breaker_manager=self.circuit_breaker_manager,
            logger=self.logger
        )
        
        self.logger.info("Budget dashboard API initialized")
    
    def _initialize_supporting_components(self) -> None:
        """Initialize supporting components (research categorizer, audit trail)."""
        # Research categorizer
        if self.lightrag_config.enable_research_categorization:
            self.research_categorizer = ResearchCategorizer()
        
        # Audit trail
        if self.lightrag_config.enable_audit_trail:
            self.audit_trail = AuditTrail(
                db_path=str(self.lightrag_config.cost_db_path.parent / "audit_trail.db")
            )
        
        self.logger.info("Supporting components initialized")
    
    def _setup_integrations(self) -> None:
        """Set up integrations between components."""
        # Integrate budget manager with alert system
        if self.budget_manager and self.alert_system:
            def alert_callback(alert):
                self.alert_system.send_alert(alert)
            
            self.budget_manager.set_alert_callback(alert_callback)
        
        # Integrate API metrics logger with research categorizer
        if self.api_metrics_logger and self.research_categorizer:
            # This integration would be handled in the API metrics logger initialization
            pass
        
        self.logger.info("Component integrations completed")
    
    def _setup_shutdown_handlers(self) -> None:
        """Set up graceful shutdown handlers."""
        def shutdown_handler(signum, frame):
            self.logger.info(f"Received shutdown signal {signum}")
            self.stop()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
    
    def _get_enabled_components(self) -> Dict[str, bool]:
        """Get dictionary of enabled components."""
        return {
            'cost_persistence': self.cost_persistence is not None,
            'budget_manager': self.budget_manager is not None,
            'api_metrics_logger': self.api_metrics_logger is not None,
            'alert_system': self.alert_system is not None,
            'real_time_monitor': self.real_time_monitor is not None,
            'circuit_breaker_manager': self.circuit_breaker_manager is not None,
            'dashboard_api': self.dashboard_api is not None,
            'research_categorizer': self.research_categorizer is not None,
            'audit_trail': self.audit_trail is not None
        }
    
    def _cleanup_partial_initialization(self) -> None:
        """Clean up after partial initialization failure."""
        try:
            if self.alert_system:
                self.alert_system.close()
            if self.api_metrics_logger:
                self.api_metrics_logger.close()
            if self.circuit_breaker_manager:
                self.circuit_breaker_manager.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class BudgetManagementFactory:
    """Factory for creating configured budget management systems."""
    
    @classmethod
    def create_from_config(cls, 
                          lightrag_config: LightRAGConfig,
                          budget_config: Optional[BudgetManagementConfig] = None) -> BudgetManagementSystem:
        """Create budget management system from configuration."""
        if budget_config is None:
            budget_config = BudgetManagementConfig.from_lightrag_config(lightrag_config)
        
        system = BudgetManagementSystem(lightrag_config, budget_config)
        system.initialize()
        
        return system
    
    @classmethod
    def create_from_environment(cls) -> BudgetManagementSystem:
        """Create budget management system from environment variables."""
        # Create LightRAG config from environment
        lightrag_config = LightRAGConfig.get_config()
        
        # Create budget config from environment
        budget_config = BudgetManagementConfig.from_environment()
        
        return cls.create_from_config(lightrag_config, budget_config)
    
    @classmethod
    def create_minimal(cls, 
                      daily_budget: float = 100.0,
                      monthly_budget: float = 3000.0) -> BudgetManagementSystem:
        """Create minimal budget management system for testing or simple use cases."""
        # Create minimal LightRAG config
        lightrag_config = LightRAGConfig(
            api_key="test",
            daily_budget_limit=daily_budget,
            monthly_budget_limit=monthly_budget,
            enable_cost_tracking=True,
            enable_budget_alerts=True,
            auto_create_dirs=True
        )
        
        # Create minimal budget config
        budget_config = BudgetManagementConfig(
            daily_budget_limit=daily_budget,
            monthly_budget_limit=monthly_budget,
            enable_alerts=True,
            alert_channels=["logging"],
            enable_real_time_monitoring=True,
            enable_circuit_breakers=True,
            enable_dashboard=True
        )
        
        return cls.create_from_config(lightrag_config, budget_config)


# Convenience functions for common use cases
def create_budget_management_system(lightrag_config: LightRAGConfig) -> BudgetManagementSystem:
    """Convenience function to create and initialize budget management system."""
    return BudgetManagementFactory.create_from_config(lightrag_config)


def create_budget_management_from_env() -> BudgetManagementSystem:
    """Convenience function to create budget management system from environment."""
    return BudgetManagementFactory.create_from_environment()


if __name__ == "__main__":
    # Example usage
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and start budget management system
    try:
        system = BudgetManagementFactory.create_from_environment()
        system.start()
        
        # System is now running and monitoring budgets
        print("Budget management system started successfully")
        print("System status:", system.get_system_status())
        
        # Keep running (in real usage, this would be managed by the main application)
        import time
        time.sleep(10)
        
        # Stop the system
        system.stop()
        print("Budget management system stopped")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()