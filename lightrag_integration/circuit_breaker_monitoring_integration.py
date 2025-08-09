"""
Circuit Breaker Monitoring Integration Module
============================================

This module provides seamless integration between the enhanced circuit breaker system
and the comprehensive monitoring system. It acts as a bridge between the circuit breaker
components and the monitoring infrastructure.

Key Features:
1. Automatic monitoring integration for circuit breaker instances
2. Event forwarding from circuit breakers to monitoring system
3. Performance tracking and metrics collection
4. Alert generation based on circuit breaker events
5. Integration with existing logging and monitoring infrastructure

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Circuit Breaker Monitoring Integration
Version: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import threading
import uuid

# Import monitoring system components
from .circuit_breaker_monitoring import (
    CircuitBreakerMonitoringSystem, CircuitBreakerMonitoringConfig,
    AlertLevel, create_monitoring_system, get_default_monitoring_config
)

# Import enhanced circuit breaker system components
try:
    from .enhanced_circuit_breaker_system import (
        EnhancedCircuitBreakerState, ServiceType, FailureType,
        CircuitBreakerEvent, ServiceMetrics
    )
    ENHANCED_CB_AVAILABLE = True
except ImportError:
    ENHANCED_CB_AVAILABLE = False
    
    # Fallback enums for when enhanced circuit breaker is not available
    from enum import Enum
    
    class EnhancedCircuitBreakerState(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
        DEGRADED = "degraded"
        RATE_LIMITED = "rate_limited"
    
    class ServiceType(Enum):
        OPENAI_API = "openai_api"
        PERPLEXITY_API = "perplexity_api"
        LIGHTRAG = "lightrag"
        CACHE = "cache"
    
    class FailureType(Enum):
        TIMEOUT = "timeout"
        HTTP_ERROR = "http_error"
        RATE_LIMIT = "rate_limit"
        SERVICE_UNAVAILABLE = "service_unavailable"


# ============================================================================
# Integration Configuration
# ============================================================================

@dataclass
class CircuitBreakerMonitoringIntegrationConfig:
    """Configuration for circuit breaker monitoring integration."""
    
    # Monitoring system config
    enable_monitoring: bool = True
    monitoring_config: Optional[CircuitBreakerMonitoringConfig] = None
    
    # Event forwarding settings
    enable_event_forwarding: bool = True
    buffer_events: bool = True
    max_event_buffer_size: int = 1000
    event_flush_interval: float = 5.0
    
    # Performance tracking
    enable_performance_tracking: bool = True
    track_response_times: bool = True
    track_success_rates: bool = True
    track_cost_impacts: bool = True
    
    # Alert configuration
    enable_auto_alerts: bool = True
    alert_on_state_changes: bool = True
    alert_on_threshold_breaches: bool = True
    alert_on_performance_degradation: bool = True
    
    # Health check integration
    enable_health_checks: bool = True
    health_check_services: List[str] = None
    
    def __post_init__(self):
        if self.health_check_services is None:
            self.health_check_services = ["openai_api", "perplexity_api", "lightrag", "cache"]
        
        if self.monitoring_config is None:
            self.monitoring_config = get_default_monitoring_config()


# ============================================================================
# Event Forwarding and Buffering
# ============================================================================

class CircuitBreakerEventForwarder:
    """Forwards circuit breaker events to the monitoring system."""
    
    def __init__(self, monitoring_system: CircuitBreakerMonitoringSystem,
                 config: CircuitBreakerMonitoringIntegrationConfig):
        self.monitoring_system = monitoring_system
        self.config = config
        
        # Event buffering
        self.event_buffer = []
        self.buffer_lock = threading.Lock()
        self.last_flush = time.time()
        
        # Performance tracking
        self.performance_data = {}
        self.performance_lock = threading.Lock()
        
        # Start background tasks if enabled
        if config.buffer_events:
            self._start_buffer_flush_task()
    
    def _start_buffer_flush_task(self):
        """Start background task to flush event buffer periodically."""
        def flush_loop():
            while True:
                try:
                    if time.time() - self.last_flush >= self.config.event_flush_interval:
                        self._flush_event_buffer()
                    time.sleep(1.0)
                except Exception as e:
                    logging.getLogger(__name__).error(f"Error in buffer flush loop: {e}")
                    time.sleep(5.0)
        
        flush_thread = threading.Thread(target=flush_loop, daemon=True)
        flush_thread.start()
    
    def forward_state_change_event(self, service: str, from_state: str, to_state: str,
                                  reason: str, metadata: Optional[Dict[str, Any]] = None):
        """Forward circuit breaker state change event."""
        if not self.config.enable_event_forwarding:
            return
        
        correlation_id = metadata.get('correlation_id') if metadata else str(uuid.uuid4())
        
        # Record in metrics
        self.monitoring_system.metrics.record_state_change(
            service=service,
            from_state=from_state,
            to_state=to_state,
            metadata=metadata
        )
        
        # Log the event
        self.monitoring_system.logger.log_state_change(
            service=service,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata
        )
        
        # Generate alerts if configured
        if self.config.enable_auto_alerts and self.config.alert_on_state_changes:
            self._generate_state_change_alerts(service, from_state, to_state, correlation_id)
    
    def forward_failure_event(self, service: str, failure_type: str, error_details: str,
                            response_time: Optional[float] = None, 
                            correlation_id: Optional[str] = None):
        """Forward service failure event."""
        if not self.config.enable_event_forwarding:
            return
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Record in metrics
        self.monitoring_system.metrics.record_failure(
            service=service,
            failure_type=failure_type,
            response_time=response_time
        )
        
        # Log the event
        self.monitoring_system.logger.log_failure(
            service=service,
            failure_type=failure_type,
            error_details=error_details,
            response_time=response_time,
            correlation_id=correlation_id
        )
    
    def forward_success_event(self, service: str, response_time: float,
                            correlation_id: Optional[str] = None):
        """Forward successful operation event."""
        if not self.config.enable_event_forwarding:
            return
        
        # Record in metrics
        self.monitoring_system.metrics.record_success(
            service=service,
            response_time=response_time
        )
        
        # Track performance data
        if self.config.enable_performance_tracking:
            with self.performance_lock:
                if service not in self.performance_data:
                    self.performance_data[service] = {
                        'response_times': [],
                        'last_success': time.time()
                    }
                
                self.performance_data[service]['response_times'].append(response_time)
                self.performance_data[service]['last_success'] = time.time()
                
                # Keep only recent data
                if len(self.performance_data[service]['response_times']) > 100:
                    self.performance_data[service]['response_times'] = \
                        self.performance_data[service]['response_times'][-100:]
    
    def forward_threshold_adjustment_event(self, service: str, adjustment_type: str,
                                         old_value: Any, new_value: Any, effectiveness: float,
                                         correlation_id: Optional[str] = None):
        """Forward threshold adjustment event."""
        if not self.config.enable_event_forwarding:
            return
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Record in metrics
        self.monitoring_system.metrics.record_threshold_adjustment(
            service=service,
            adjustment_type=adjustment_type,
            old_value=old_value,
            new_value=new_value,
            effectiveness=effectiveness
        )
        
        # Log the event
        self.monitoring_system.logger.log_threshold_adjustment(
            service=service,
            adjustment_type=adjustment_type,
            old_value=old_value,
            new_value=new_value,
            effectiveness=effectiveness,
            correlation_id=correlation_id
        )
        
        # Generate alert if configured
        if self.config.enable_auto_alerts and self.config.alert_on_threshold_breaches:
            self.monitoring_system.alerting.alert_threshold_breach(
                service=service,
                threshold_type=adjustment_type,
                current_value=new_value,
                threshold_value=old_value,
                correlation_id=correlation_id
            )
    
    def forward_cost_impact_event(self, service: str, cost_saved: float,
                                budget_impact: Dict[str, Any],
                                correlation_id: Optional[str] = None):
        """Forward cost impact event."""
        if not self.config.enable_event_forwarding:
            return
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Record in metrics
        self.monitoring_system.metrics.record_cost_impact(
            service=service,
            cost_saved=cost_saved,
            budget_impact=budget_impact
        )
        
        # Generate cost alert if significant
        budget_percentage = budget_impact.get('budget_percentage', 0)
        if cost_saved > 50 or budget_percentage > 10:  # Configurable thresholds
            self.monitoring_system.alerting.alert_cost_impact(
                service=service,
                cost_impact=cost_saved,
                budget_percentage=budget_percentage,
                correlation_id=correlation_id
            )
    
    def _generate_state_change_alerts(self, service: str, from_state: str, to_state: str,
                                    correlation_id: str):
        """Generate appropriate alerts for state changes."""
        if to_state == "open":
            # Critical alert when circuit breaker opens
            metrics = self.monitoring_system.metrics.get_current_metrics(service)
            failure_count = metrics.get('total_failures', 0) if metrics else 0
            
            self.monitoring_system.alerting.alert_circuit_breaker_open(
                service=service,
                failure_count=failure_count,
                threshold=5,  # Default threshold, should come from config
                correlation_id=correlation_id
            )
        
        elif to_state == "closed" and from_state in ["open", "half_open"]:
            # Recovery alert
            # Calculate downtime (simplified)
            downtime_seconds = 60.0  # Placeholder - should be calculated from actual downtime
            
            self.monitoring_system.alerting.alert_circuit_breaker_recovery(
                service=service,
                downtime_seconds=downtime_seconds,
                correlation_id=correlation_id
            )
    
    def _flush_event_buffer(self):
        """Flush buffered events to monitoring system."""
        with self.buffer_lock:
            if not self.event_buffer:
                return
            
            # Process buffered events
            # This would be implemented if event buffering is needed
            self.event_buffer.clear()
            self.last_flush = time.time()


# ============================================================================
# Circuit Breaker Monitoring Wrapper
# ============================================================================

class MonitoredCircuitBreaker:
    """Wrapper that adds monitoring to any circuit breaker implementation."""
    
    def __init__(self, wrapped_circuit_breaker, service_name: str,
                 monitoring_system: CircuitBreakerMonitoringSystem,
                 event_forwarder: CircuitBreakerEventForwarder):
        self.wrapped_cb = wrapped_circuit_breaker
        self.service_name = service_name
        self.monitoring_system = monitoring_system
        self.event_forwarder = event_forwarder
        
        # Track state changes
        self._last_state = getattr(wrapped_circuit_breaker, 'state', 'unknown')
        self._state_change_timestamps = {}
    
    async def call(self, operation: Callable, *args, **kwargs):
        """Execute operation through circuit breaker with monitoring."""
        correlation_id = kwargs.pop('correlation_id', str(uuid.uuid4()))
        start_time = time.time()
        
        try:
            # Check for state changes before operation
            current_state = getattr(self.wrapped_cb, 'state', 'unknown')
            if current_state != self._last_state:
                self.event_forwarder.forward_state_change_event(
                    service=self.service_name,
                    from_state=self._last_state,
                    to_state=current_state,
                    reason="state_check",
                    metadata={'correlation_id': correlation_id}
                )
                self._last_state = current_state
                self._state_change_timestamps[current_state] = time.time()
            
            # Execute operation through wrapped circuit breaker
            if hasattr(self.wrapped_cb, 'call'):
                result = await self.wrapped_cb.call(operation, *args, **kwargs)
            else:
                # Fallback for different circuit breaker interfaces
                result = await operation(*args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            self.event_forwarder.forward_success_event(
                service=self.service_name,
                response_time=response_time,
                correlation_id=correlation_id
            )
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            failure_type = self._classify_exception(e)
            
            self.event_forwarder.forward_failure_event(
                service=self.service_name,
                failure_type=failure_type,
                error_details=str(e),
                response_time=response_time,
                correlation_id=correlation_id
            )
            
            raise
    
    def _classify_exception(self, exception: Exception) -> str:
        """Classify exception type for monitoring purposes."""
        exception_type = type(exception).__name__.lower()
        
        if 'timeout' in exception_type or 'timeout' in str(exception).lower():
            return FailureType.TIMEOUT.value
        elif 'http' in exception_type or 'status' in exception_type:
            return FailureType.HTTP_ERROR.value
        elif 'rate' in str(exception).lower() or 'limit' in str(exception).lower():
            return FailureType.RATE_LIMIT.value
        elif 'auth' in str(exception).lower():
            return FailureType.AUTHENTICATION.value if hasattr(FailureType, 'AUTHENTICATION') else "authentication"
        elif 'unavailable' in str(exception).lower():
            return FailureType.SERVICE_UNAVAILABLE.value
        else:
            return "unknown_error"


# ============================================================================
# Main Integration Manager
# ============================================================================

class CircuitBreakerMonitoringIntegration:
    """Main integration manager for circuit breaker monitoring."""
    
    def __init__(self, config: Optional[CircuitBreakerMonitoringIntegrationConfig] = None):
        self.config = config or CircuitBreakerMonitoringIntegrationConfig()
        
        # Initialize monitoring system
        if self.config.enable_monitoring:
            self.monitoring_system = CircuitBreakerMonitoringSystem(
                self.config.monitoring_config
            )
            self.event_forwarder = CircuitBreakerEventForwarder(
                self.monitoring_system, self.config
            )
        else:
            self.monitoring_system = None
            self.event_forwarder = None
        
        # Track monitored circuit breakers
        self.monitored_circuit_breakers: Dict[str, MonitoredCircuitBreaker] = {}
        
        # Integration state
        self._is_started = False
    
    async def start(self):
        """Start the monitoring integration."""
        if self._is_started:
            return
        
        if self.monitoring_system:
            await self.monitoring_system.start_monitoring()
        
        self._is_started = True
    
    async def stop(self):
        """Stop the monitoring integration."""
        if not self._is_started:
            return
        
        if self.monitoring_system:
            await self.monitoring_system.stop_monitoring()
        
        self._is_started = False
    
    def wrap_circuit_breaker(self, circuit_breaker, service_name: str) -> MonitoredCircuitBreaker:
        """Wrap a circuit breaker with monitoring capabilities."""
        if not self.monitoring_system or not self.event_forwarder:
            # Return original circuit breaker if monitoring is disabled
            return circuit_breaker
        
        monitored_cb = MonitoredCircuitBreaker(
            wrapped_circuit_breaker=circuit_breaker,
            service_name=service_name,
            monitoring_system=self.monitoring_system,
            event_forwarder=self.event_forwarder
        )
        
        self.monitored_circuit_breakers[service_name] = monitored_cb
        return monitored_cb
    
    def get_health_status(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Get health status from monitoring system."""
        if not self.monitoring_system:
            return {'error': 'monitoring_disabled'}
        
        return self.monitoring_system.health_check.get_health_status(service)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.monitoring_system:
            return {'error': 'monitoring_disabled'}
        
        return self.monitoring_system.get_monitoring_summary()
    
    def get_active_alerts(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts."""
        if not self.monitoring_system:
            return []
        
        alerts = self.monitoring_system.alerting.get_active_alerts(service)
        return [
            {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'service': alert.service,
                'alert_type': alert.alert_type,
                'level': alert.level.value,
                'message': alert.message,
                'details': alert.details
            }
            for alert in alerts
        ]
    
    def get_prometheus_metrics(self) -> Optional[str]:
        """Get Prometheus metrics."""
        if not self.monitoring_system:
            return None
        
        return self.monitoring_system.get_prometheus_metrics()
    
    def force_health_check_update(self):
        """Force an immediate health check update."""
        if self.monitoring_system:
            self.monitoring_system.health_check._update_health_status()


# ============================================================================
# Factory Functions and Utilities
# ============================================================================

def create_monitoring_integration(config_overrides: Optional[Dict[str, Any]] = None) -> CircuitBreakerMonitoringIntegration:
    """Factory function to create monitoring integration with config overrides."""
    config = CircuitBreakerMonitoringIntegrationConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return CircuitBreakerMonitoringIntegration(config)


def get_integration_health_endpoint():
    """Get a simple health endpoint function for web frameworks."""
    integration = None
    
    def health_endpoint():
        nonlocal integration
        if not integration:
            integration = create_monitoring_integration()
        
        health_status = integration.get_health_status()
        system_summary = integration.monitoring_system.health_check.get_system_health_summary() if integration.monitoring_system else {}
        
        return {
            'status': 'ok',
            'timestamp': datetime.utcnow().isoformat(),
            'circuit_breaker_health': health_status,
            'system_summary': system_summary
        }
    
    return health_endpoint


def get_integration_metrics_endpoint():
    """Get a metrics endpoint function for Prometheus scraping."""
    integration = None
    
    def metrics_endpoint():
        nonlocal integration
        if not integration:
            integration = create_monitoring_integration()
        
        return integration.get_prometheus_metrics()
    
    return metrics_endpoint


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CircuitBreakerMonitoringIntegration',
    'CircuitBreakerMonitoringIntegrationConfig',
    'CircuitBreakerEventForwarder',
    'MonitoredCircuitBreaker',
    'create_monitoring_integration',
    'get_integration_health_endpoint',
    'get_integration_metrics_endpoint'
]