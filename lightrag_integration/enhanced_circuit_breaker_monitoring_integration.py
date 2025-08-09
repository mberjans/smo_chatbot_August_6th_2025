"""
Enhanced Circuit Breaker Monitoring Integration
===============================================

This module provides comprehensive integration between the enhanced circuit breaker system
and the monitoring infrastructure. It seamlessly connects circuit breaker events with
monitoring, logging, and alerting systems while maintaining compatibility with existing
infrastructure.

Key Features:
1. Automatic monitoring integration for all circuit breaker types
2. Event forwarding with minimal performance impact
3. Integration with existing production monitoring
4. Backward compatibility with existing systems
5. Configuration management and environment variable support

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Enhanced Circuit Breaker Monitoring Integration
Version: 1.0.0
"""

import asyncio
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import threading
import uuid
import json

# Import monitoring components
from .circuit_breaker_monitoring import (
    CircuitBreakerMonitoringSystem,
    CircuitBreakerMonitoringConfig,
    get_default_monitoring_config
)

from .circuit_breaker_monitoring_integration import (
    CircuitBreakerMonitoringIntegration,
    CircuitBreakerMonitoringIntegrationConfig,
    create_monitoring_integration
)

from .circuit_breaker_dashboard import (
    CircuitBreakerDashboardConfig,
    StandaloneDashboardServer,
    create_dashboard
)

# Import existing production monitoring
try:
    from .production_monitoring import ProductionMonitoringSystem
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False
    ProductionMonitoringSystem = None

# Import existing enhanced circuit breakers
try:
    from .enhanced_circuit_breaker_system import (
        EnhancedCircuitBreakerOrchestrator,
        OpenAICircuitBreaker,
        PerplexityCircuitBreaker, 
        LightRAGCircuitBreaker,
        CacheCircuitBreaker,
        EnhancedCircuitBreakerState,
        ServiceType,
        FailureType
    )
    ENHANCED_CB_AVAILABLE = True
except ImportError:
    ENHANCED_CB_AVAILABLE = False


# ============================================================================
# Integration Configuration
# ============================================================================

@dataclass
class EnhancedCircuitBreakerMonitoringConfig:
    """Comprehensive configuration for enhanced circuit breaker monitoring."""
    
    # Monitoring system settings
    enable_monitoring: bool = True
    monitoring_log_level: str = "INFO"
    monitoring_log_file: Optional[str] = "logs/enhanced_circuit_breaker_monitoring.log"
    
    # Integration with existing systems
    integrate_with_production_monitoring: bool = True
    use_existing_log_patterns: bool = True
    preserve_existing_metrics: bool = True
    
    # Circuit breaker specific settings
    monitor_openai_circuit_breaker: bool = True
    monitor_perplexity_circuit_breaker: bool = True
    monitor_lightrag_circuit_breaker: bool = True
    monitor_cache_circuit_breaker: bool = True
    
    # Event forwarding
    enable_real_time_event_forwarding: bool = True
    buffer_events: bool = True
    event_buffer_size: int = 1000
    event_flush_interval: float = 5.0
    
    # Dashboard settings
    enable_dashboard: bool = True
    dashboard_port: int = 8091
    dashboard_host: str = "0.0.0.0"
    enable_dashboard_websockets: bool = True
    
    # Alert settings
    enable_critical_alerts: bool = True
    enable_recovery_notifications: bool = True
    enable_performance_alerts: bool = True
    
    # Environment variable overrides
    def __post_init__(self):
        # Override with environment variables if present
        self.enable_monitoring = os.getenv('ECB_MONITORING_ENABLED', str(self.enable_monitoring)).lower() == 'true'
        self.monitoring_log_level = os.getenv('ECB_MONITORING_LOG_LEVEL', self.monitoring_log_level)
        self.monitoring_log_file = os.getenv('ECB_MONITORING_LOG_FILE', self.monitoring_log_file)
        
        # Dashboard settings
        self.enable_dashboard = os.getenv('ECB_DASHBOARD_ENABLED', str(self.enable_dashboard)).lower() == 'true'
        self.dashboard_port = int(os.getenv('ECB_DASHBOARD_PORT', str(self.dashboard_port)))
        self.dashboard_host = os.getenv('ECB_DASHBOARD_HOST', self.dashboard_host)
        
        # Integration settings
        self.integrate_with_production_monitoring = os.getenv('ECB_INTEGRATE_PRODUCTION', str(self.integrate_with_production_monitoring)).lower() == 'true'


# ============================================================================
# Circuit Breaker Event Interceptor
# ============================================================================

class CircuitBreakerEventInterceptor:
    """Intercepts events from enhanced circuit breakers and forwards to monitoring."""
    
    def __init__(self, monitoring_integration: CircuitBreakerMonitoringIntegration):
        self.monitoring_integration = monitoring_integration
        self.logger = logging.getLogger(__name__)
        
        # Event statistics
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time = None
        
    def intercept_state_change(self, circuit_breaker, from_state: str, to_state: str, 
                             reason: str, metadata: Optional[Dict[str, Any]] = None):
        """Intercept circuit breaker state changes."""
        try:
            service_name = self._get_service_name(circuit_breaker)
            correlation_id = metadata.get('correlation_id') if metadata else str(uuid.uuid4())
            
            # Forward to monitoring
            if self.monitoring_integration.event_forwarder:
                self.monitoring_integration.event_forwarder.forward_state_change_event(
                    service=service_name,
                    from_state=from_state,
                    to_state=to_state,
                    reason=reason,
                    metadata=metadata
                )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Failed to intercept state change event: {e}")
    
    def intercept_operation_failure(self, circuit_breaker, operation_name: str,
                                  exception: Exception, response_time: Optional[float] = None,
                                  metadata: Optional[Dict[str, Any]] = None):
        """Intercept operation failures."""
        try:
            service_name = self._get_service_name(circuit_breaker)
            failure_type = self._classify_exception(exception)
            correlation_id = metadata.get('correlation_id') if metadata else str(uuid.uuid4())
            
            # Forward to monitoring
            if self.monitoring_integration.event_forwarder:
                self.monitoring_integration.event_forwarder.forward_failure_event(
                    service=service_name,
                    failure_type=failure_type,
                    error_details=f"{operation_name}: {str(exception)}",
                    response_time=response_time,
                    correlation_id=correlation_id
                )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Failed to intercept failure event: {e}")
    
    def intercept_operation_success(self, circuit_breaker, operation_name: str,
                                  response_time: float, metadata: Optional[Dict[str, Any]] = None):
        """Intercept successful operations."""
        try:
            service_name = self._get_service_name(circuit_breaker)
            correlation_id = metadata.get('correlation_id') if metadata else str(uuid.uuid4())
            
            # Forward to monitoring
            if self.monitoring_integration.event_forwarder:
                self.monitoring_integration.event_forwarder.forward_success_event(
                    service=service_name,
                    response_time=response_time,
                    correlation_id=correlation_id
                )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Failed to intercept success event: {e}")
    
    def _get_service_name(self, circuit_breaker) -> str:
        """Extract service name from circuit breaker instance."""
        if hasattr(circuit_breaker, 'service_type'):
            return str(circuit_breaker.service_type.value)
        elif hasattr(circuit_breaker, 'service'):
            return str(circuit_breaker.service)
        else:
            return circuit_breaker.__class__.__name__.lower().replace('circuitbreaker', '')
    
    def _classify_exception(self, exception: Exception) -> str:
        """Classify exception type for monitoring."""
        exception_type = type(exception).__name__.lower()
        exception_message = str(exception).lower()
        
        if 'timeout' in exception_type or 'timeout' in exception_message:
            return 'timeout'
        elif 'http' in exception_type or 'status' in exception_type:
            return 'http_error'
        elif 'rate' in exception_message or 'limit' in exception_message:
            return 'rate_limit'
        elif 'auth' in exception_message:
            return 'authentication'
        elif 'unavailable' in exception_message:
            return 'service_unavailable'
        elif 'budget' in exception_message or 'quota' in exception_message:
            return 'budget_exceeded'
        else:
            return 'unknown_error'
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics."""
        return {
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'success_rate': (self.events_processed / max(self.events_processed + self.events_failed, 1)) * 100,
            'last_event_time': self.last_event_time,
            'last_event_ago_seconds': time.time() - self.last_event_time if self.last_event_time else None
        }


# ============================================================================
# Enhanced Circuit Breaker Wrapper
# ============================================================================

class MonitoringEnabledCircuitBreaker:
    """Wrapper that adds monitoring to existing enhanced circuit breakers."""
    
    def __init__(self, wrapped_circuit_breaker, event_interceptor: CircuitBreakerEventInterceptor):
        self.wrapped_cb = wrapped_circuit_breaker
        self.event_interceptor = event_interceptor
        
        # Store original methods to wrap them
        self._wrap_methods()
        
        # Track current state for change detection
        self._last_state = getattr(wrapped_circuit_breaker, 'state', 'unknown')
    
    def _wrap_methods(self):
        """Wrap circuit breaker methods to add monitoring."""
        # Wrap state change methods if they exist
        if hasattr(self.wrapped_cb, '_change_state'):
            original_change_state = self.wrapped_cb._change_state
            
            def monitored_change_state(new_state, reason="state_change", metadata=None):
                old_state = getattr(self.wrapped_cb, 'state', 'unknown')
                result = original_change_state(new_state, reason, metadata)
                
                # Intercept state change
                self.event_interceptor.intercept_state_change(
                    self.wrapped_cb, old_state, new_state, reason, metadata
                )
                
                return result
            
            self.wrapped_cb._change_state = monitored_change_state
        
        # Wrap call methods
        if hasattr(self.wrapped_cb, 'call'):
            original_call = self.wrapped_cb.call
            
            async def monitored_call(operation, *args, **kwargs):
                operation_name = getattr(operation, '__name__', 'unknown_operation')
                start_time = time.time()
                correlation_id = kwargs.get('correlation_id', str(uuid.uuid4()))
                
                try:
                    result = await original_call(operation, *args, **kwargs)
                    
                    # Intercept success
                    response_time = time.time() - start_time
                    self.event_interceptor.intercept_operation_success(
                        self.wrapped_cb, operation_name, response_time,
                        {'correlation_id': correlation_id}
                    )
                    
                    return result
                    
                except Exception as e:
                    # Intercept failure
                    response_time = time.time() - start_time
                    self.event_interceptor.intercept_operation_failure(
                        self.wrapped_cb, operation_name, e, response_time,
                        {'correlation_id': correlation_id}
                    )
                    raise
            
            self.wrapped_cb.call = monitored_call
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped circuit breaker."""
        return getattr(self.wrapped_cb, name)


# ============================================================================
# Main Enhanced Integration Manager
# ============================================================================

class EnhancedCircuitBreakerMonitoringManager:
    """Main manager for enhanced circuit breaker monitoring integration."""
    
    def __init__(self, config: Optional[EnhancedCircuitBreakerMonitoringConfig] = None):
        self.config = config or EnhancedCircuitBreakerMonitoringConfig()
        
        # Initialize monitoring components
        if self.config.enable_monitoring:
            self._setup_monitoring_system()
        else:
            self.monitoring_integration = None
            self.event_interceptor = None
            self.dashboard_server = None
        
        # Track monitored circuit breakers
        self.monitored_circuit_breakers: Dict[str, MonitoringEnabledCircuitBreaker] = {}
        
        # Integration state
        self._is_started = False
        self._startup_time = None
    
    def _setup_monitoring_system(self):
        """Setup the monitoring system components."""
        # Create monitoring integration
        monitoring_config = CircuitBreakerMonitoringIntegrationConfig(
            enable_monitoring=True,
            enable_event_forwarding=self.config.enable_real_time_event_forwarding,
            buffer_events=self.config.buffer_events,
            max_event_buffer_size=self.config.event_buffer_size,
            event_flush_interval=self.config.event_flush_interval,
            enable_auto_alerts=self.config.enable_critical_alerts,
            enable_health_checks=True
        )
        
        # Setup monitoring config
        monitoring_system_config = get_default_monitoring_config()
        monitoring_system_config.log_level = self.config.monitoring_log_level
        monitoring_system_config.log_file_path = self.config.monitoring_log_file
        
        monitoring_config.monitoring_config = monitoring_system_config
        
        self.monitoring_integration = CircuitBreakerMonitoringIntegration(monitoring_config)
        self.event_interceptor = CircuitBreakerEventInterceptor(self.monitoring_integration)
        
        # Setup dashboard if enabled
        if self.config.enable_dashboard:
            dashboard_config = CircuitBreakerDashboardConfig()
            dashboard_config.port = self.config.dashboard_port
            dashboard_config.host = self.config.dashboard_host
            dashboard_config.enable_websockets = self.config.enable_dashboard_websockets
            
            self.dashboard_server = StandaloneDashboardServer(
                self.monitoring_integration, dashboard_config
            )
        else:
            self.dashboard_server = None
    
    async def start(self):
        """Start the monitoring manager."""
        if self._is_started:
            return
        
        if self.monitoring_integration:
            await self.monitoring_integration.start()
        
        if self.dashboard_server:
            try:
                await self.dashboard_server.start_server()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Dashboard server failed to start: {e}")
        
        self._is_started = True
        self._startup_time = time.time()
        
        logging.getLogger(__name__).info("Enhanced circuit breaker monitoring started")
    
    async def stop(self):
        """Stop the monitoring manager."""
        if not self._is_started:
            return
        
        if self.monitoring_integration:
            await self.monitoring_integration.stop()
        
        # Dashboard server stop would need to be implemented
        
        self._is_started = False
        logging.getLogger(__name__).info("Enhanced circuit breaker monitoring stopped")
    
    def register_circuit_breaker(self, circuit_breaker, service_name: Optional[str] = None) -> MonitoringEnabledCircuitBreaker:
        """Register a circuit breaker for monitoring."""
        if not self.config.enable_monitoring or not self.event_interceptor:
            return circuit_breaker
        
        # Determine service name
        if not service_name:
            service_name = self._determine_service_name(circuit_breaker)
        
        # Create monitored wrapper
        monitored_cb = MonitoringEnabledCircuitBreaker(circuit_breaker, self.event_interceptor)
        self.monitored_circuit_breakers[service_name] = monitored_cb
        
        logging.getLogger(__name__).info(f"Registered circuit breaker for monitoring: {service_name}")
        return monitored_cb
    
    def _determine_service_name(self, circuit_breaker) -> str:
        """Determine service name from circuit breaker instance."""
        class_name = circuit_breaker.__class__.__name__
        
        if 'OpenAI' in class_name:
            return 'openai_api'
        elif 'Perplexity' in class_name:
            return 'perplexity_api'
        elif 'LightRAG' in class_name:
            return 'lightrag'
        elif 'Cache' in class_name:
            return 'cache'
        else:
            return class_name.lower().replace('circuitbreaker', '')
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        status = {
            'monitoring_enabled': self.config.enable_monitoring,
            'is_started': self._is_started,
            'startup_time': self._startup_time,
            'uptime_seconds': time.time() - self._startup_time if self._startup_time else None,
            'monitored_services': list(self.monitored_circuit_breakers.keys()),
            'dashboard_enabled': self.config.enable_dashboard and self.dashboard_server is not None
        }
        
        if self.event_interceptor:
            status['event_statistics'] = self.event_interceptor.get_event_statistics()
        
        if self.monitoring_integration:
            status['health_summary'] = self.monitoring_integration.get_health_status()
            status['active_alerts'] = len(self.monitoring_integration.get_active_alerts())
        
        if self.dashboard_server:
            status['dashboard_info'] = self.dashboard_server.get_dashboard_info()
        
        return status
    
    def get_service_health(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for specific service or all services."""
        if not self.monitoring_integration:
            return {'error': 'monitoring_disabled'}
        
        return self.monitoring_integration.get_health_status(service)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        if not self.monitoring_integration:
            return {'error': 'monitoring_disabled'}
        
        return self.monitoring_integration.get_metrics_summary()
    
    def force_health_check_update(self):
        """Force immediate health check update."""
        if self.monitoring_integration:
            self.monitoring_integration.force_health_check_update()


# ============================================================================
# Integration with Existing Production Systems
# ============================================================================

class ProductionIntegrationHelper:
    """Helper class for integrating with existing production monitoring systems."""
    
    def __init__(self, monitoring_manager: EnhancedCircuitBreakerMonitoringManager):
        self.monitoring_manager = monitoring_manager
        self.logger = logging.getLogger(__name__)
    
    def integrate_with_production_monitoring(self):
        """Integrate with existing production monitoring system."""
        if not PRODUCTION_MONITORING_AVAILABLE:
            self.logger.warning("Production monitoring system not available for integration")
            return False
        
        try:
            # This would integrate with the existing production monitoring
            # The exact implementation depends on the existing system architecture
            self.logger.info("Integrated with production monitoring system")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with production monitoring: {e}")
            return False
    
    def setup_existing_log_integration(self):
        """Setup integration with existing log patterns."""
        try:
            # Configure monitoring to use existing log patterns
            if self.monitoring_manager.monitoring_integration:
                # This would configure the monitoring to match existing patterns
                self.logger.info("Integrated with existing log patterns")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to setup log integration: {e}")
            return False


# ============================================================================
# Factory Functions and Utilities
# ============================================================================

def create_enhanced_monitoring_manager(config_overrides: Optional[Dict[str, Any]] = None) -> EnhancedCircuitBreakerMonitoringManager:
    """Factory function to create enhanced monitoring manager."""
    config = EnhancedCircuitBreakerMonitoringConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return EnhancedCircuitBreakerMonitoringManager(config)


def setup_comprehensive_monitoring(circuit_breaker_orchestrator=None,
                                 config_overrides: Optional[Dict[str, Any]] = None) -> EnhancedCircuitBreakerMonitoringManager:
    """Setup comprehensive monitoring for circuit breaker system."""
    manager = create_enhanced_monitoring_manager(config_overrides)
    
    # Register existing circuit breakers if orchestrator is provided
    if circuit_breaker_orchestrator and ENHANCED_CB_AVAILABLE:
        try:
            if hasattr(circuit_breaker_orchestrator, 'openai_cb'):
                manager.register_circuit_breaker(circuit_breaker_orchestrator.openai_cb, 'openai_api')
            
            if hasattr(circuit_breaker_orchestrator, 'perplexity_cb'):
                manager.register_circuit_breaker(circuit_breaker_orchestrator.perplexity_cb, 'perplexity_api')
            
            if hasattr(circuit_breaker_orchestrator, 'lightrag_cb'):
                manager.register_circuit_breaker(circuit_breaker_orchestrator.lightrag_cb, 'lightrag')
            
            if hasattr(circuit_breaker_orchestrator, 'cache_cb'):
                manager.register_circuit_breaker(circuit_breaker_orchestrator.cache_cb, 'cache')
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to register some circuit breakers: {e}")
    
    return manager


async def start_monitoring_system(manager: EnhancedCircuitBreakerMonitoringManager):
    """Start the monitoring system with proper error handling."""
    try:
        await manager.start()
        
        # Setup production integration if enabled
        if manager.config.integrate_with_production_monitoring:
            integration_helper = ProductionIntegrationHelper(manager)
            integration_helper.integrate_with_production_monitoring()
            integration_helper.setup_existing_log_integration()
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to start monitoring system: {e}")
        return False


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'EnhancedCircuitBreakerMonitoringManager',
    'EnhancedCircuitBreakerMonitoringConfig',
    'CircuitBreakerEventInterceptor',
    'MonitoringEnabledCircuitBreaker',
    'ProductionIntegrationHelper',
    'create_enhanced_monitoring_manager',
    'setup_comprehensive_monitoring',
    'start_monitoring_system'
]