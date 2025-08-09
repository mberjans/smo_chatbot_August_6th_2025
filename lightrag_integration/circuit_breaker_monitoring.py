"""
Circuit Breaker Comprehensive Monitoring and Logging System
===========================================================

This module provides comprehensive monitoring, logging, and alerting for the enhanced
circuit breaker system. It integrates with existing monitoring infrastructure and
provides real-time visibility into circuit breaker behavior and effectiveness.

Key Features:
1. Circuit Breaker Metrics Collection - State changes, failure rates, performance metrics
2. Enhanced Structured Logging - Debug, warning, error logs for troubleshooting
3. Alerting and Notifications - Critical alerts, recovery notifications, threshold breaches
4. Dashboard Integration - Health checks, real-time status, service availability
5. Integration with Existing Infrastructure - Coordinated with current logging patterns

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Circuit Breaker Monitoring Implementation
Version: 1.0.0
"""

import asyncio
import logging
import time
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from pathlib import Path
import traceback
import statistics
from enum import Enum
import uuid

# Import existing monitoring infrastructure
try:
    from .production_monitoring import (
        ProductionLoggerConfig, StructuredLogFormatter, 
        PROMETHEUS_AVAILABLE
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False
    class ProductionLoggerConfig:
        pass
    class StructuredLogFormatter(logging.Formatter):
        pass

# Prometheus imports if available
if PROMETHEUS_AVAILABLE:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class CircuitBreakerMonitoringConfig:
    """Configuration for circuit breaker monitoring system."""
    
    # Logging configuration
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    log_file_path: Optional[str] = "logs/circuit_breaker_monitoring.log"
    enable_debug_logging: bool = False
    
    # Metrics collection
    enable_prometheus_metrics: bool = True
    metrics_port: int = 8090
    enable_custom_metrics: bool = True
    
    # Alerting configuration
    enable_alerting: bool = True
    alert_file_path: str = "logs/alerts/circuit_breaker_alerts.json"
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    
    # Dashboard integration
    enable_health_endpoints: bool = True
    health_check_interval: float = 30.0
    enable_real_time_monitoring: bool = True
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_window_size: int = 1000
    enable_trend_analysis: bool = True
    
    # Integration settings
    correlation_id_header: str = "X-Correlation-ID"
    enable_cross_service_correlation: bool = True


# ============================================================================
# Metrics Collection Classes
# ============================================================================

class CircuitBreakerMetrics:
    """Comprehensive metrics collection for circuit breakers."""
    
    def __init__(self, config: CircuitBreakerMonitoringConfig):
        self.config = config
        self.lock = threading.RLock()
        
        # State tracking
        self.state_changes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_states: Dict[str, str] = {}
        
        # Failure tracking
        self.failure_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.failure_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.performance_window_size))
        self.success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Recovery tracking
        self.recovery_times: Dict[str, List[float]] = defaultdict(list)
        self.recovery_success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Threshold tracking
        self.threshold_adjustments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.threshold_effectiveness: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Cost impact tracking
        self.cost_savings: Dict[str, float] = defaultdict(float)
        self.budget_impacts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and config.enable_prometheus_metrics:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.registry = CollectorRegistry()
        
        # State change metrics
        self.state_change_counter = Counter(
            'circuit_breaker_state_changes_total',
            'Total number of circuit breaker state changes',
            ['service', 'from_state', 'to_state'],
            registry=self.registry
        )
        
        # Failure metrics
        self.failure_counter = Counter(
            'circuit_breaker_failures_total',
            'Total number of failures per service',
            ['service', 'failure_type'],
            registry=self.registry
        )
        
        self.failure_rate_gauge = Gauge(
            'circuit_breaker_failure_rate',
            'Current failure rate per service',
            ['service'],
            registry=self.registry
        )
        
        # Performance metrics
        self.response_time_histogram = Histogram(
            'circuit_breaker_response_time_seconds',
            'Response time distribution per service',
            ['service'],
            registry=self.registry
        )
        
        self.success_rate_gauge = Gauge(
            'circuit_breaker_success_rate',
            'Current success rate per service',
            ['service'],
            registry=self.registry
        )
        
        # Recovery metrics
        self.recovery_time_histogram = Histogram(
            'circuit_breaker_recovery_time_seconds',
            'Time taken for circuit breaker recovery',
            ['service'],
            registry=self.registry
        )
        
        # Threshold metrics
        self.threshold_adjustment_counter = Counter(
            'circuit_breaker_threshold_adjustments_total',
            'Total threshold adjustments',
            ['service', 'adjustment_type'],
            registry=self.registry
        )
        
        # Cost metrics
        self.cost_savings_gauge = Gauge(
            'circuit_breaker_cost_savings_total',
            'Total cost savings from circuit breaker activations',
            ['service'],
            registry=self.registry
        )
    
    def record_state_change(self, service: str, from_state: str, to_state: str, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Record a circuit breaker state change."""
        with self.lock:
            timestamp = datetime.utcnow()
            
            change_record = {
                'timestamp': timestamp.isoformat(),
                'service': service,
                'from_state': from_state,
                'to_state': to_state,
                'metadata': metadata or {}
            }
            
            self.state_changes[service].append(change_record)
            self.current_states[service] = to_state
            
            # Update Prometheus metrics
            if hasattr(self, 'state_change_counter'):
                self.state_change_counter.labels(
                    service=service, 
                    from_state=from_state, 
                    to_state=to_state
                ).inc()
    
    def record_failure(self, service: str, failure_type: str, 
                      response_time: Optional[float] = None):
        """Record a service failure."""
        with self.lock:
            timestamp = time.time()
            
            self.failure_counts[service][failure_type] += 1
            self.failure_rates[service].append((timestamp, failure_type))
            
            if response_time is not None:
                self.response_times[service].append(response_time)
            
            # Update Prometheus metrics
            if hasattr(self, 'failure_counter'):
                self.failure_counter.labels(
                    service=service, 
                    failure_type=failure_type
                ).inc()
                
            if hasattr(self, 'response_time_histogram') and response_time:
                self.response_time_histogram.labels(service=service).observe(response_time)
    
    def record_success(self, service: str, response_time: float):
        """Record a successful operation."""
        with self.lock:
            timestamp = time.time()
            
            self.response_times[service].append(response_time)
            self.success_rates[service].append((timestamp, True))
            
            # Update Prometheus metrics
            if hasattr(self, 'response_time_histogram'):
                self.response_time_histogram.labels(service=service).observe(response_time)
    
    def record_recovery(self, service: str, recovery_time: float, successful: bool):
        """Record circuit breaker recovery attempt."""
        with self.lock:
            self.recovery_times[service].append(recovery_time)
            self.recovery_success_rates[service].append((time.time(), successful))
            
            # Update Prometheus metrics
            if hasattr(self, 'recovery_time_histogram'):
                self.recovery_time_histogram.labels(service=service).observe(recovery_time)
    
    def record_threshold_adjustment(self, service: str, adjustment_type: str, 
                                  old_value: Any, new_value: Any, effectiveness: float):
        """Record threshold adjustment and its effectiveness."""
        with self.lock:
            adjustment = {
                'timestamp': datetime.utcnow().isoformat(),
                'service': service,
                'adjustment_type': adjustment_type,
                'old_value': old_value,
                'new_value': new_value,
                'effectiveness': effectiveness
            }
            
            self.threshold_adjustments[service].append(adjustment)
            self.threshold_effectiveness[service][adjustment_type] = effectiveness
            
            # Update Prometheus metrics
            if hasattr(self, 'threshold_adjustment_counter'):
                self.threshold_adjustment_counter.labels(
                    service=service,
                    adjustment_type=adjustment_type
                ).inc()
    
    def record_cost_impact(self, service: str, cost_saved: float, 
                          budget_impact: Dict[str, Any]):
        """Record cost impact from circuit breaker activation."""
        with self.lock:
            self.cost_savings[service] += cost_saved
            self.budget_impacts[service].append({
                'timestamp': datetime.utcnow().isoformat(),
                'cost_saved': cost_saved,
                **budget_impact
            })
            
            # Update Prometheus metrics
            if hasattr(self, 'cost_savings_gauge'):
                self.cost_savings_gauge.labels(service=service).set(self.cost_savings[service])
    
    def get_current_metrics(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Get current metrics for a service or all services."""
        with self.lock:
            if service:
                return self._get_service_metrics(service)
            else:
                return {svc: self._get_service_metrics(svc) for svc in self.current_states.keys()}
    
    def _get_service_metrics(self, service: str) -> Dict[str, Any]:
        """Get metrics for a specific service."""
        now = time.time()
        
        # Calculate failure rate (last 5 minutes)
        recent_failures = [
            f for f in self.failure_rates[service] 
            if now - f[0] <= 300
        ]
        failure_rate = len(recent_failures) / 5.0 if recent_failures else 0.0
        
        # Calculate success rate (last hour)
        recent_successes = [
            s for s in self.success_rates[service] 
            if now - s[0] <= 3600
        ]
        success_rate = sum(1 for s in recent_successes if s[1]) / max(len(recent_successes), 1)
        
        # Calculate average response time
        recent_response_times = list(self.response_times[service])
        avg_response_time = statistics.mean(recent_response_times) if recent_response_times else 0.0
        p95_response_time = statistics.quantiles(recent_response_times, n=20)[18] if len(recent_response_times) >= 20 else avg_response_time
        
        # Calculate recovery metrics
        recent_recovery_times = self.recovery_times[service][-10:] if self.recovery_times[service] else []
        avg_recovery_time = statistics.mean(recent_recovery_times) if recent_recovery_times else 0.0
        
        return {
            'service': service,
            'current_state': self.current_states.get(service, 'unknown'),
            'failure_rate': failure_rate,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'avg_recovery_time': avg_recovery_time,
            'total_cost_savings': self.cost_savings[service],
            'total_state_changes': len(self.state_changes[service]),
            'total_failures': sum(self.failure_counts[service].values()),
            'threshold_effectiveness': dict(self.threshold_effectiveness[service])
        }


# ============================================================================
# Enhanced Logging System
# ============================================================================

class CircuitBreakerLogger:
    """Enhanced logging system for circuit breaker events."""
    
    def __init__(self, config: CircuitBreakerMonitoringConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger for circuit breaker events."""
        logger = logging.getLogger('circuit_breaker_monitoring')
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        if self.config.enable_structured_logging:
            console_handler.setFormatter(StructuredLogFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file_path:
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            if self.config.enable_structured_logging:
                file_handler.setFormatter(StructuredLogFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            logger.addHandler(file_handler)
        
        return logger
    
    def log_state_change(self, service: str, from_state: str, to_state: str, 
                        reason: str, metadata: Optional[Dict[str, Any]] = None):
        """Log circuit breaker state change."""
        correlation_id = metadata.get('correlation_id') if metadata else str(uuid.uuid4())
        
        log_data = {
            'event_type': 'state_change',
            'service': service,
            'from_state': from_state,
            'to_state': to_state,
            'reason': reason,
            'correlation_id': correlation_id,
            'metadata': metadata or {}
        }
        
        if to_state == 'open':
            self.logger.error(
                f"Circuit breaker OPENED for {service}: {from_state} -> {to_state}. Reason: {reason}",
                extra={'circuit_breaker_data': log_data}
            )
        elif to_state == 'closed' and from_state in ['open', 'half_open']:
            self.logger.info(
                f"Circuit breaker RECOVERED for {service}: {from_state} -> {to_state}. Reason: {reason}",
                extra={'circuit_breaker_data': log_data}
            )
        elif to_state == 'degraded':
            self.logger.warning(
                f"Circuit breaker DEGRADED for {service}: {from_state} -> {to_state}. Reason: {reason}",
                extra={'circuit_breaker_data': log_data}
            )
        else:
            self.logger.info(
                f"Circuit breaker state change for {service}: {from_state} -> {to_state}. Reason: {reason}",
                extra={'circuit_breaker_data': log_data}
            )
    
    def log_failure(self, service: str, failure_type: str, error_details: str,
                   response_time: Optional[float] = None, correlation_id: Optional[str] = None):
        """Log service failure that affects circuit breaker."""
        correlation_id = correlation_id or str(uuid.uuid4())
        
        log_data = {
            'event_type': 'failure',
            'service': service,
            'failure_type': failure_type,
            'error_details': error_details,
            'response_time': response_time,
            'correlation_id': correlation_id
        }
        
        self.logger.warning(
            f"Service failure recorded for {service}: {failure_type} - {error_details}",
            extra={'circuit_breaker_data': log_data}
        )
    
    def log_performance_impact(self, service: str, impact_type: str, 
                             metrics: Dict[str, Any], correlation_id: Optional[str] = None):
        """Log performance impact from circuit breaker decisions."""
        correlation_id = correlation_id or str(uuid.uuid4())
        
        log_data = {
            'event_type': 'performance_impact',
            'service': service,
            'impact_type': impact_type,
            'metrics': metrics,
            'correlation_id': correlation_id
        }
        
        self.logger.info(
            f"Performance impact for {service}: {impact_type}",
            extra={'circuit_breaker_data': log_data}
        )
    
    def log_threshold_adjustment(self, service: str, adjustment_type: str,
                               old_value: Any, new_value: Any, effectiveness: float,
                               correlation_id: Optional[str] = None):
        """Log threshold adjustment events."""
        correlation_id = correlation_id or str(uuid.uuid4())
        
        log_data = {
            'event_type': 'threshold_adjustment',
            'service': service,
            'adjustment_type': adjustment_type,
            'old_value': old_value,
            'new_value': new_value,
            'effectiveness': effectiveness,
            'correlation_id': correlation_id
        }
        
        self.logger.info(
            f"Threshold adjustment for {service}: {adjustment_type} changed from {old_value} to {new_value} (effectiveness: {effectiveness:.2f})",
            extra={'circuit_breaker_data': log_data}
        )
    
    def log_debug_decision(self, service: str, decision: str, factors: Dict[str, Any],
                          correlation_id: Optional[str] = None):
        """Log debug information for circuit breaker decisions."""
        if not self.config.enable_debug_logging:
            return
            
        correlation_id = correlation_id or str(uuid.uuid4())
        
        log_data = {
            'event_type': 'debug_decision',
            'service': service,
            'decision': decision,
            'factors': factors,
            'correlation_id': correlation_id
        }
        
        self.logger.debug(
            f"Circuit breaker decision for {service}: {decision}",
            extra={'circuit_breaker_data': log_data}
        )


# ============================================================================
# Alerting and Notification System
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class CircuitBreakerAlert:
    """Circuit breaker alert data structure."""
    id: str
    timestamp: datetime
    service: str
    alert_type: str
    level: AlertLevel
    message: str
    details: Dict[str, Any]
    correlation_id: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class CircuitBreakerAlerting:
    """Alerting and notification system for circuit breaker events."""
    
    def __init__(self, config: CircuitBreakerMonitoringConfig):
        self.config = config
        self.active_alerts: Dict[str, CircuitBreakerAlert] = {}
        self.alert_history: List[CircuitBreakerAlert] = []
        self.lock = threading.RLock()
        
        # Create alerts directory
        if config.alert_file_path:
            Path(config.alert_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    def create_alert(self, service: str, alert_type: str, level: AlertLevel,
                    message: str, details: Optional[Dict[str, Any]] = None,
                    correlation_id: Optional[str] = None) -> str:
        """Create a new alert."""
        alert_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())
        
        alert = CircuitBreakerAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            service=service,
            alert_type=alert_type,
            level=level,
            message=message,
            details=details or {},
            correlation_id=correlation_id
        )
        
        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Write to alert file
            if self.config.alert_file_path:
                self._write_alert_to_file(alert)
        
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                
                del self.active_alerts[alert_id]
                
                # Write resolution to file
                if self.config.alert_file_path:
                    self._write_alert_to_file(alert)
                
                return True
            return False
    
    def _write_alert_to_file(self, alert: CircuitBreakerAlert):
        """Write alert to JSON log file."""
        try:
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert.timestamp.isoformat()
            if alert.resolved_at:
                alert_data['resolved_at'] = alert.resolved_at.isoformat()
            alert_data['level'] = alert.level.value
            
            with open(self.config.alert_file_path, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            # Don't let alert logging failures break the system
            logging.getLogger(__name__).error(f"Failed to write alert to file: {e}")
    
    def alert_circuit_breaker_open(self, service: str, failure_count: int,
                                 threshold: int, correlation_id: Optional[str] = None) -> str:
        """Alert when circuit breaker opens."""
        return self.create_alert(
            service=service,
            alert_type="circuit_breaker_open",
            level=AlertLevel.CRITICAL,
            message=f"Circuit breaker opened for {service} after {failure_count} failures (threshold: {threshold})",
            details={
                'failure_count': failure_count,
                'threshold': threshold,
                'impact': 'service_unavailable'
            },
            correlation_id=correlation_id
        )
    
    def alert_circuit_breaker_recovery(self, service: str, downtime_seconds: float,
                                     correlation_id: Optional[str] = None) -> str:
        """Alert when circuit breaker recovers."""
        return self.create_alert(
            service=service,
            alert_type="circuit_breaker_recovery",
            level=AlertLevel.INFO,
            message=f"Circuit breaker recovered for {service} after {downtime_seconds:.1f} seconds",
            details={
                'downtime_seconds': downtime_seconds,
                'status': 'service_restored'
            },
            correlation_id=correlation_id
        )
    
    def alert_threshold_breach(self, service: str, threshold_type: str,
                             current_value: Any, threshold_value: Any,
                             correlation_id: Optional[str] = None) -> str:
        """Alert when threshold is breached."""
        return self.create_alert(
            service=service,
            alert_type="threshold_breach",
            level=AlertLevel.WARNING,
            message=f"Threshold breach for {service}: {threshold_type} = {current_value} (threshold: {threshold_value})",
            details={
                'threshold_type': threshold_type,
                'current_value': current_value,
                'threshold_value': threshold_value
            },
            correlation_id=correlation_id
        )
    
    def alert_performance_degradation(self, service: str, metric: str,
                                    current_value: float, baseline_value: float,
                                    correlation_id: Optional[str] = None) -> str:
        """Alert when performance degrades significantly."""
        degradation_percentage = ((current_value - baseline_value) / baseline_value) * 100
        
        return self.create_alert(
            service=service,
            alert_type="performance_degradation",
            level=AlertLevel.ERROR,
            message=f"Performance degradation for {service}: {metric} degraded by {degradation_percentage:.1f}%",
            details={
                'metric': metric,
                'current_value': current_value,
                'baseline_value': baseline_value,
                'degradation_percentage': degradation_percentage
            },
            correlation_id=correlation_id
        )
    
    def alert_cost_impact(self, service: str, cost_impact: float, budget_percentage: float,
                         correlation_id: Optional[str] = None) -> str:
        """Alert when circuit breaker activations have significant cost impact."""
        level = AlertLevel.WARNING if cost_impact > 100 else AlertLevel.INFO
        
        return self.create_alert(
            service=service,
            alert_type="cost_impact",
            level=level,
            message=f"Circuit breaker cost impact for {service}: ${cost_impact:.2f} ({budget_percentage:.1f}% of budget)",
            details={
                'cost_impact': cost_impact,
                'budget_percentage': budget_percentage
            },
            correlation_id=correlation_id
        )
    
    def get_active_alerts(self, service: Optional[str] = None) -> List[CircuitBreakerAlert]:
        """Get active alerts, optionally filtered by service."""
        with self.lock:
            alerts = list(self.active_alerts.values())
            if service:
                alerts = [alert for alert in alerts if alert.service == service]
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, service: Optional[str] = None, 
                         hours: int = 24) -> List[CircuitBreakerAlert]:
        """Get alert history, optionally filtered by service and time."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp >= cutoff_time
            ]
            
            if service:
                alerts = [alert for alert in alerts if alert.service == service]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)


# ============================================================================
# Dashboard Integration and Health Checks
# ============================================================================

class CircuitBreakerHealthCheck:
    """Health check system for circuit breaker monitoring."""
    
    def __init__(self, metrics: CircuitBreakerMetrics, 
                 alerting: CircuitBreakerAlerting,
                 config: CircuitBreakerMonitoringConfig):
        self.metrics = metrics
        self.alerting = alerting
        self.config = config
        
        self._health_status: Dict[str, Dict[str, Any]] = {}
        self._last_health_check = time.time()
    
    def get_health_status(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Get current health status for services."""
        now = time.time()
        
        # Update health status if needed
        if now - self._last_health_check >= self.config.health_check_interval:
            self._update_health_status()
            self._last_health_check = now
        
        if service:
            return self._health_status.get(service, {'status': 'unknown'})
        else:
            return dict(self._health_status)
    
    def _update_health_status(self):
        """Update health status for all services."""
        current_metrics = self.metrics.get_current_metrics()
        
        for service, service_metrics in current_metrics.items():
            health_status = self._calculate_service_health(service, service_metrics)
            self._health_status[service] = health_status
    
    def _calculate_service_health(self, service: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate health status for a specific service."""
        current_state = metrics.get('current_state', 'unknown')
        failure_rate = metrics.get('failure_rate', 0.0)
        success_rate = metrics.get('success_rate', 1.0)
        avg_response_time = metrics.get('avg_response_time', 0.0)
        
        # Determine overall health status
        if current_state == 'open':
            status = 'critical'
            health_score = 0.0
        elif current_state == 'degraded':
            status = 'warning'
            health_score = 0.3
        elif failure_rate > 0.1:  # More than 10% failure rate
            status = 'warning'
            health_score = max(0.1, 1.0 - failure_rate)
        elif success_rate < 0.9:  # Less than 90% success rate
            status = 'warning'
            health_score = success_rate
        else:
            status = 'healthy'
            health_score = success_rate
        
        # Get active alerts for this service
        active_alerts = self.alerting.get_active_alerts(service)
        
        return {
            'service': service,
            'status': status,
            'health_score': health_score,
            'current_state': current_state,
            'metrics': {
                'failure_rate': failure_rate,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'active_alerts': len(active_alerts)
            },
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        health_statuses = self.get_health_status()
        
        if not health_statuses:
            return {
                'overall_status': 'unknown',
                'healthy_services': 0,
                'warning_services': 0,
                'critical_services': 0,
                'total_services': 0
            }
        
        status_counts = {'healthy': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
        
        for service_health in health_statuses.values():
            status = service_health.get('status', 'unknown')
            status_counts[status] += 1
        
        # Determine overall status
        if status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        elif status_counts['healthy'] > 0:
            overall_status = 'healthy'
        else:
            overall_status = 'unknown'
        
        return {
            'overall_status': overall_status,
            'healthy_services': status_counts['healthy'],
            'warning_services': status_counts['warning'],
            'critical_services': status_counts['critical'],
            'total_services': sum(status_counts.values()),
            'last_updated': datetime.utcnow().isoformat()
        }


# ============================================================================
# Main Monitoring System Integration
# ============================================================================

class CircuitBreakerMonitoringSystem:
    """Main circuit breaker monitoring system that coordinates all components."""
    
    def __init__(self, config: Optional[CircuitBreakerMonitoringConfig] = None):
        self.config = config or CircuitBreakerMonitoringConfig()
        
        # Initialize components
        self.metrics = CircuitBreakerMetrics(self.config)
        self.logger = CircuitBreakerLogger(self.config)
        self.alerting = CircuitBreakerAlerting(self.config)
        self.health_check = CircuitBreakerHealthCheck(
            self.metrics, self.alerting, self.config
        )
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._is_running:
            return
        
        self._is_running = True
        
        if self.config.enable_real_time_monitoring:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._is_running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._is_running:
            try:
                # Update health checks
                self.health_check._update_health_status()
                
                # Update Prometheus metrics if available
                if hasattr(self.metrics, 'failure_rate_gauge'):
                    current_metrics = self.metrics.get_current_metrics()
                    for service, metrics in current_metrics.items():
                        self.metrics.failure_rate_gauge.labels(service=service).set(
                            metrics['failure_rate']
                        )
                        self.metrics.success_rate_gauge.labels(service=service).set(
                            metrics['success_rate']
                        )
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    def get_prometheus_metrics(self) -> Optional[str]:
        """Get Prometheus metrics in text format."""
        if hasattr(self.metrics, 'registry'):
            return generate_latest(self.metrics.registry)
        return None
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            'system_health': self.health_check.get_system_health_summary(),
            'active_alerts': len(self.alerting.get_active_alerts()),
            'metrics_summary': self.metrics.get_current_metrics(),
            'monitoring_config': {
                'prometheus_enabled': PROMETHEUS_AVAILABLE and self.config.enable_prometheus_metrics,
                'structured_logging': self.config.enable_structured_logging,
                'alerting_enabled': self.config.enable_alerting,
                'real_time_monitoring': self.config.enable_real_time_monitoring
            },
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# Integration Helper Functions
# ============================================================================

def create_monitoring_system(config_overrides: Optional[Dict[str, Any]] = None) -> CircuitBreakerMonitoringSystem:
    """Factory function to create a monitoring system with optional config overrides."""
    config = CircuitBreakerMonitoringConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return CircuitBreakerMonitoringSystem(config)


def get_default_monitoring_config() -> CircuitBreakerMonitoringConfig:
    """Get default monitoring configuration with environment variable overrides."""
    return CircuitBreakerMonitoringConfig(
        log_level=os.getenv('CB_MONITORING_LOG_LEVEL', 'INFO'),
        enable_structured_logging=os.getenv('CB_MONITORING_STRUCTURED_LOGS', 'true').lower() == 'true',
        log_file_path=os.getenv('CB_MONITORING_LOG_FILE', 'logs/circuit_breaker_monitoring.log'),
        enable_debug_logging=os.getenv('CB_MONITORING_DEBUG', 'false').lower() == 'true',
        enable_prometheus_metrics=os.getenv('CB_MONITORING_PROMETHEUS', 'true').lower() == 'true',
        metrics_port=int(os.getenv('CB_MONITORING_METRICS_PORT', '8090')),
        enable_alerting=os.getenv('CB_MONITORING_ALERTING', 'true').lower() == 'true',
        alert_file_path=os.getenv('CB_MONITORING_ALERT_FILE', 'logs/alerts/circuit_breaker_alerts.json'),
        enable_health_endpoints=os.getenv('CB_MONITORING_HEALTH_ENDPOINTS', 'true').lower() == 'true',
        health_check_interval=float(os.getenv('CB_MONITORING_HEALTH_INTERVAL', '30.0')),
        enable_real_time_monitoring=os.getenv('CB_MONITORING_REAL_TIME', 'true').lower() == 'true'
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CircuitBreakerMonitoringSystem',
    'CircuitBreakerMonitoringConfig',
    'CircuitBreakerMetrics',
    'CircuitBreakerLogger',
    'CircuitBreakerAlerting',
    'CircuitBreakerHealthCheck',
    'CircuitBreakerAlert',
    'AlertLevel',
    'create_monitoring_system',
    'get_default_monitoring_config'
]