#!/usr/bin/env python3
"""
IntelligentQueryRouter - Enhanced Wrapper for Biomedical Query Routing

This module provides an intelligent wrapper around the BiomedicalQueryRouter that
adds system health monitoring, load balancing, analytics, and enhanced decision logic.

Key Features:
- System health checks and monitoring integration
- Load balancing between multiple backends
- Routing decision logging and analytics
- Performance monitoring and optimization
- Enhanced uncertainty-aware routing decisions

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: CMO-LIGHTRAG-013-T01 Implementation
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import asyncio
import statistics
from contextlib import asynccontextmanager
import random
import os
import psutil
import requests
from pathlib import Path
import openai
import httpx
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction, ConfidenceMetrics
from .research_categorizer import ResearchCategorizer, CategoryPrediction
from .cost_persistence import ResearchCategory


class SystemHealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    RECOVERING = "recovering"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthTrend(Enum):
    """Health trend indicators"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


class BackendType(Enum):
    """Backend service types"""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"


@dataclass
class HealthAlert:
    """Health alert information"""
    id: str
    backend_type: BackendType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    threshold_breached: str
    current_value: float
    threshold_value: float
    suppressed: bool = False
    acknowledged: bool = False
    auto_recovery: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'backend_type': self.backend_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'threshold_breached': self.threshold_breached,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'suppressed': self.suppressed,
            'acknowledged': self.acknowledged,
            'auto_recovery': self.auto_recovery
        }


@dataclass
class PerformanceMetrics:
    """Advanced performance metrics"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    availability_history: deque = field(default_factory=lambda: deque(maxlen=100))
    recovery_times: List[float] = field(default_factory=list)
    
    # Percentile calculations
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Error analysis
    error_patterns: Dict[str, int] = field(default_factory=dict)
    error_categories: Dict[str, int] = field(default_factory=dict)
    
    # Trend analysis
    response_time_trend: HealthTrend = HealthTrend.STABLE
    error_rate_trend: HealthTrend = HealthTrend.STABLE
    availability_trend: HealthTrend = HealthTrend.STABLE
    
    def calculate_percentiles(self):
        """Calculate response time percentiles"""
        if len(self.response_times) > 0:
            times = sorted(self.response_times)
            self.p50_response_time = np.percentile(times, 50)
            self.p95_response_time = np.percentile(times, 95)
            self.p99_response_time = np.percentile(times, 99)
    
    def analyze_trends(self, window_size: int = 20):
        """Analyze performance trends"""
        if len(self.response_times) < window_size:
            return
        
        recent_times = list(self.response_times)[-window_size:]
        older_times = list(self.response_times)[-(window_size*2):-window_size] if len(self.response_times) >= window_size*2 else []
        
        if older_times:
            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)
            change_ratio = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            
            if abs(change_ratio) < 0.05:  # Less than 5% change
                self.response_time_trend = HealthTrend.STABLE
            elif change_ratio > 0.15:  # More than 15% increase
                self.response_time_trend = HealthTrend.DEGRADING
            elif change_ratio < -0.15:  # More than 15% decrease
                self.response_time_trend = HealthTrend.IMPROVING
            else:
                # Check volatility
                stdev = statistics.stdev(recent_times) if len(recent_times) > 1 else 0
                if stdev > recent_avg * 0.3:  # High volatility
                    self.response_time_trend = HealthTrend.VOLATILE
                else:
                    self.response_time_trend = HealthTrend.STABLE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'p50_response_time': self.p50_response_time,
            'p95_response_time': self.p95_response_time,
            'p99_response_time': self.p99_response_time,
            'response_time_trend': self.response_time_trend.value,
            'error_rate_trend': self.error_rate_trend.value,
            'availability_trend': self.availability_trend.value,
            'error_patterns': dict(self.error_patterns),
            'error_categories': dict(self.error_categories),
            'recovery_times_count': len(self.recovery_times),
            'avg_recovery_time_ms': statistics.mean(self.recovery_times) if self.recovery_times else 0.0
        }


@dataclass
class BackendHealthMetrics:
    """Enhanced health metrics for a backend service"""
    backend_type: BackendType
    status: SystemHealthStatus
    response_time_ms: float
    error_rate: float
    last_health_check: datetime
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    # Enhanced metrics
    health_score: float = 100.0  # 0-100 score
    availability_percentage: float = 100.0
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    last_failure_time: Optional[datetime] = None
    last_recovery_time: Optional[datetime] = None
    mean_time_to_recovery_ms: float = 0.0
    
    # API quota tracking
    api_quota_used: Optional[int] = None
    api_quota_limit: Optional[int] = None
    api_quota_reset_time: Optional[datetime] = None
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    def calculate_health_score(self) -> float:
        """Calculate weighted health score based on multiple factors"""
        factors = {
            'availability': (self.availability_percentage / 100.0) * 0.3,  # 30% weight
            'response_time': max(0, (2000 - self.response_time_ms) / 2000) * 0.25,  # 25% weight
            'error_rate': max(0, (1.0 - self.error_rate)) * 0.25,  # 25% weight
            'consecutive_failures': max(0, (5 - self.consecutive_failures) / 5) * 0.1,  # 10% weight
            'resource_usage': max(0, (100 - max(self.cpu_usage_percent, self.memory_usage_percent)) / 100) * 0.1  # 10% weight
        }
        
        self.health_score = sum(factors.values()) * 100
        return self.health_score
    
    def update_performance_metrics(self, response_time: float, success: bool, error_type: Optional[str] = None):
        """Update performance metrics with new data point"""
        self.performance_metrics.response_times.append(response_time)
        self.performance_metrics.error_counts.append(0 if success else 1)
        
        if not success and error_type:
            self.performance_metrics.error_patterns[error_type] = self.performance_metrics.error_patterns.get(error_type, 0) + 1
            
            # Categorize errors
            if 'timeout' in error_type.lower():
                category = 'timeout'
            elif 'auth' in error_type.lower() or 'unauthorized' in error_type.lower():
                category = 'authentication'
            elif 'rate' in error_type.lower() or 'quota' in error_type.lower():
                category = 'rate_limiting'
            elif 'network' in error_type.lower() or 'connection' in error_type.lower():
                category = 'network'
            else:
                category = 'other'
            
            self.performance_metrics.error_categories[category] = self.performance_metrics.error_categories.get(category, 0) + 1
        
        # Update calculated metrics
        self.performance_metrics.calculate_percentiles()
        self.performance_metrics.analyze_trends()
        
        # Update availability
        if len(self.performance_metrics.error_counts) >= 10:
            recent_errors = sum(list(self.performance_metrics.error_counts)[-10:])
            self.availability_percentage = ((10 - recent_errors) / 10) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = {
            'backend_type': self.backend_type.value,
            'status': self.status.value,
            'response_time_ms': self.response_time_ms,
            'error_rate': self.error_rate,
            'last_health_check': self.last_health_check.isoformat(),
            'consecutive_failures': self.consecutive_failures,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'health_score': self.health_score,
            'availability_percentage': self.availability_percentage,
            'mean_time_to_recovery_ms': self.mean_time_to_recovery_ms,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'disk_usage_percent': self.disk_usage_percent
        }
        
        if self.last_failure_time:
            base_dict['last_failure_time'] = self.last_failure_time.isoformat()
        if self.last_recovery_time:
            base_dict['last_recovery_time'] = self.last_recovery_time.isoformat()
        
        if self.api_quota_used is not None:
            base_dict['api_quota_used'] = self.api_quota_used
        if self.api_quota_limit is not None:
            base_dict['api_quota_limit'] = self.api_quota_limit
        if self.api_quota_reset_time:
            base_dict['api_quota_reset_time'] = self.api_quota_reset_time.isoformat()
        
        base_dict['performance_metrics'] = self.performance_metrics.to_dict()
        
        return base_dict


@dataclass 
class RoutingAnalytics:
    """Analytics data for routing decisions"""
    timestamp: datetime
    query: str
    routing_decision: RoutingDecision
    confidence: float
    response_time_ms: float
    backend_used: Optional[BackendType] = None
    fallback_triggered: bool = False
    system_health_impact: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'routing_decision': self.routing_decision.value,
            'confidence': self.confidence,
            'response_time_ms': self.response_time_ms,
            'backend_used': self.backend_used.value if self.backend_used else None,
            'fallback_triggered': self.fallback_triggered,
            'system_health_impact': self.system_health_impact,
            'metadata': self.metadata
        }


@dataclass
class HealthCheckResult:
    """Result from a health check operation"""
    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    check_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlertThresholds:
    """Configurable alert thresholds"""
    # Response time thresholds (ms)
    response_time_warning: float = 1000.0
    response_time_critical: float = 2000.0
    response_time_emergency: float = 5000.0
    
    # Error rate thresholds (0.0-1.0)
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.15  # 15%
    error_rate_emergency: float = 0.30  # 30%
    
    # Availability thresholds (0.0-100.0)
    availability_warning: float = 95.0  # 95%
    availability_critical: float = 90.0  # 90%
    availability_emergency: float = 80.0  # 80%
    
    # Health score thresholds (0.0-100.0)
    health_score_warning: float = 80.0
    health_score_critical: float = 60.0
    health_score_emergency: float = 40.0
    
    # Consecutive failure thresholds
    consecutive_failures_warning: int = 3
    consecutive_failures_critical: int = 5
    consecutive_failures_emergency: int = 10
    
    # API quota thresholds (0.0-1.0 as percentage of limit)
    api_quota_warning: float = 0.80  # 80%
    api_quota_critical: float = 0.90  # 90%
    api_quota_emergency: float = 0.95  # 95%
    
    # Resource usage thresholds
    cpu_usage_warning: float = 70.0
    cpu_usage_critical: float = 85.0
    cpu_usage_emergency: float = 95.0
    
    memory_usage_warning: float = 70.0
    memory_usage_critical: float = 85.0
    memory_usage_emergency: float = 95.0
    
    disk_usage_warning: float = 70.0
    disk_usage_critical: float = 85.0
    disk_usage_emergency: float = 90.0
    
    # Alert suppression (seconds)
    alert_suppression_window: int = 300  # 5 minutes
    
    def get_severity_for_metric(self, metric_name: str, value: float, higher_is_worse: bool = True) -> Optional[AlertSeverity]:
        """Determine alert severity for a metric value"""
        if not hasattr(self, f"{metric_name}_warning"):
            return None
        
        warning_threshold = getattr(self, f"{metric_name}_warning")
        critical_threshold = getattr(self, f"{metric_name}_critical")
        emergency_threshold = getattr(self, f"{metric_name}_emergency")
        
        if higher_is_worse:
            if value >= emergency_threshold:
                return AlertSeverity.EMERGENCY
            elif value >= critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= warning_threshold:
                return AlertSeverity.WARNING
        else:  # Lower is worse (e.g., availability, health score)
            if value <= emergency_threshold:
                return AlertSeverity.EMERGENCY
            elif value <= critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= warning_threshold:
                return AlertSeverity.WARNING
        
        return None


class AlertCallback:
    """Base class for alert callbacks"""
    
    def __call__(self, alert: HealthAlert) -> bool:
        """
        Process an alert
        
        Args:
            alert: The health alert to process
            
        Returns:
            bool: True if callback was successful, False otherwise
        """
        raise NotImplementedError


class ConsoleAlertCallback(AlertCallback):
    """Console/log alert callback"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def __call__(self, alert: HealthAlert) -> bool:
        """Log alert to console/logger"""
        try:
            severity_symbol = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️", 
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.EMERGENCY: "🆘"
            }.get(alert.severity, "❓")
            
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            message = (f"{severity_symbol} ALERT [{alert.severity.value.upper()}] "
                      f"{alert.backend_type.value.upper()}: {alert.message}")
            
            self.logger.log(log_level, message)
            
            # Add additional context for non-INFO alerts
            if alert.severity != AlertSeverity.INFO:
                context = (f"  📊 Current: {alert.current_value:.2f}, "
                          f"Threshold: {alert.threshold_value:.2f}, "
                          f"Metric: {alert.threshold_breached}")
                self.logger.log(log_level, context)
            
            return True
        except Exception as e:
            self.logger.error(f"Console alert callback failed: {e}")
            return False


class JSONFileAlertCallback(AlertCallback):
    """JSON file alert callback for persistence"""
    
    def __init__(self, file_path: str, max_alerts: int = 10000):
        self.file_path = Path(file_path)
        self.max_alerts = max_alerts
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, alert: HealthAlert) -> bool:
        """Save alert to JSON file"""
        try:
            # Load existing alerts
            alerts = []
            if self.file_path.exists():
                try:
                    with open(self.file_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            alerts = json.loads(content)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in alert file {self.file_path}, starting fresh")
                    alerts = []
            
            # Add new alert
            alerts.append(alert.to_dict())
            
            # Trim to max_alerts to prevent unlimited growth
            if len(alerts) > self.max_alerts:
                alerts = alerts[-self.max_alerts:]
            
            # Write back to file
            with open(self.file_path, 'w') as f:
                json.dump(alerts, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"JSON file alert callback failed: {e}")
            return False


class WebhookAlertCallback(AlertCallback):
    """Webhook alert callback"""
    
    def __init__(self, webhook_url: str, timeout: float = 5.0, 
                 headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.headers = headers or {'Content-Type': 'application/json'}
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, alert: HealthAlert) -> bool:
        """Send alert via webhook"""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'Clinical_Metabolomics_Oracle'
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code < 400:
                return True
            else:
                self.logger.error(f"Webhook alert failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Webhook alert callback failed: {e}")
            return False


@dataclass
class AlertSuppressionRule:
    """Rule for suppressing duplicate alerts"""
    alert_id_pattern: str  # Pattern to match alert IDs
    suppression_window_seconds: int  # How long to suppress duplicates
    max_occurrences: int = 1  # Max occurrences before suppression kicks in


class AlertManager:
    """
    Comprehensive alert management system for health monitoring.
    
    Features:
    - Generate alerts when health thresholds are breached
    - Alert suppression to prevent spam
    - Alert acknowledgment and auto-recovery
    - Configurable callback system for alert delivery
    - Alert history and analytics
    """
    
    def __init__(self, alert_thresholds: Optional[AlertThresholds] = None):
        self.alert_thresholds = alert_thresholds or AlertThresholds()
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_callbacks: List[AlertCallback] = []
        self.suppression_rules: List[AlertSuppressionRule] = []
        self.alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Threading for callback execution
        self.callback_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="AlertCallback")
        
        self.logger = logging.getLogger(__name__)
        
        # Default suppression rules
        self._setup_default_suppression_rules()
        
        # Default alert callbacks
        self._setup_default_callbacks()
    
    def _setup_default_suppression_rules(self):
        """Setup default alert suppression rules"""
        self.suppression_rules = [
            AlertSuppressionRule(
                alert_id_pattern="response_time_*",
                suppression_window_seconds=300,  # 5 minutes
                max_occurrences=3
            ),
            AlertSuppressionRule(
                alert_id_pattern="error_rate_*", 
                suppression_window_seconds=180,  # 3 minutes
                max_occurrences=2
            ),
            AlertSuppressionRule(
                alert_id_pattern="availability_*",
                suppression_window_seconds=600,  # 10 minutes  
                max_occurrences=1
            ),
            AlertSuppressionRule(
                alert_id_pattern="health_score_*",
                suppression_window_seconds=300,  # 5 minutes
                max_occurrences=2
            )
        ]
    
    def _setup_default_callbacks(self):
        """Setup default alert callbacks"""
        # Console callback
        self.alert_callbacks.append(ConsoleAlertCallback())
        
        # JSON file callback 
        alerts_dir = Path("./logs/alerts")
        alerts_dir.mkdir(parents=True, exist_ok=True)
        self.alert_callbacks.append(
            JSONFileAlertCallback(str(alerts_dir / "health_alerts.json"))
        )
    
    def add_callback(self, callback: AlertCallback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
        self.logger.info(f"Added alert callback: {callback.__class__.__name__}")
    
    def remove_callback(self, callback: AlertCallback):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info(f"Removed alert callback: {callback.__class__.__name__}")
    
    def add_suppression_rule(self, rule: AlertSuppressionRule):
        """Add alert suppression rule"""
        self.suppression_rules.append(rule)
        self.logger.info(f"Added suppression rule for pattern: {rule.alert_id_pattern}")
    
    def check_and_generate_alerts(self, backend_metrics: BackendHealthMetrics) -> List[HealthAlert]:
        """
        Check backend metrics against thresholds and generate alerts.
        
        Args:
            backend_metrics: Backend health metrics to check
            
        Returns:
            List of generated alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Response time alerts
        severity = self.alert_thresholds.get_severity_for_metric(
            "response_time", backend_metrics.response_time_ms, higher_is_worse=True
        )
        if severity:
            alert_id = f"response_time_{backend_metrics.backend_type.value}_{severity.value}"
            alert = HealthAlert(
                id=alert_id,
                backend_type=backend_metrics.backend_type,
                severity=severity,
                message=f"Response time {backend_metrics.response_time_ms:.0f}ms exceeds {severity.value} threshold",
                timestamp=current_time,
                threshold_breached="response_time",
                current_value=backend_metrics.response_time_ms,
                threshold_value=getattr(self.alert_thresholds, f"response_time_{severity.value}")
            )
            alerts.append(alert)
        
        # Error rate alerts
        severity = self.alert_thresholds.get_severity_for_metric(
            "error_rate", backend_metrics.error_rate, higher_is_worse=True
        )
        if severity:
            alert_id = f"error_rate_{backend_metrics.backend_type.value}_{severity.value}"
            alert = HealthAlert(
                id=alert_id,
                backend_type=backend_metrics.backend_type,
                severity=severity,
                message=f"Error rate {backend_metrics.error_rate:.1%} exceeds {severity.value} threshold",
                timestamp=current_time,
                threshold_breached="error_rate",
                current_value=backend_metrics.error_rate,
                threshold_value=getattr(self.alert_thresholds, f"error_rate_{severity.value}")
            )
            alerts.append(alert)
        
        # Availability alerts
        severity = self.alert_thresholds.get_severity_for_metric(
            "availability", backend_metrics.availability_percentage, higher_is_worse=False
        )
        if severity:
            alert_id = f"availability_{backend_metrics.backend_type.value}_{severity.value}"
            alert = HealthAlert(
                id=alert_id,
                backend_type=backend_metrics.backend_type,
                severity=severity,
                message=f"Availability {backend_metrics.availability_percentage:.1f}% below {severity.value} threshold",
                timestamp=current_time,
                threshold_breached="availability",
                current_value=backend_metrics.availability_percentage,
                threshold_value=getattr(self.alert_thresholds, f"availability_{severity.value}")
            )
            alerts.append(alert)
        
        # Health score alerts
        backend_metrics.calculate_health_score()  # Ensure it's up to date
        severity = self.alert_thresholds.get_severity_for_metric(
            "health_score", backend_metrics.health_score, higher_is_worse=False
        )
        if severity:
            alert_id = f"health_score_{backend_metrics.backend_type.value}_{severity.value}"
            alert = HealthAlert(
                id=alert_id,
                backend_type=backend_metrics.backend_type,
                severity=severity,
                message=f"Health score {backend_metrics.health_score:.1f} below {severity.value} threshold",
                timestamp=current_time,
                threshold_breached="health_score",
                current_value=backend_metrics.health_score,
                threshold_value=getattr(self.alert_thresholds, f"health_score_{severity.value}")
            )
            alerts.append(alert)
        
        # Consecutive failures alerts
        severity = self.alert_thresholds.get_severity_for_metric(
            "consecutive_failures", backend_metrics.consecutive_failures, higher_is_worse=True
        )
        if severity:
            alert_id = f"consecutive_failures_{backend_metrics.backend_type.value}_{severity.value}"
            alert = HealthAlert(
                id=alert_id,
                backend_type=backend_metrics.backend_type,
                severity=severity,
                message=f"Consecutive failures {backend_metrics.consecutive_failures} exceeds {severity.value} threshold",
                timestamp=current_time,
                threshold_breached="consecutive_failures",
                current_value=float(backend_metrics.consecutive_failures),
                threshold_value=float(getattr(self.alert_thresholds, f"consecutive_failures_{severity.value}"))
            )
            alerts.append(alert)
        
        # Resource usage alerts
        for resource in ['cpu_usage', 'memory_usage', 'disk_usage']:
            current_value = getattr(backend_metrics, f"{resource}_percent")
            severity = self.alert_thresholds.get_severity_for_metric(
                resource, current_value, higher_is_worse=True
            )
            if severity:
                alert_id = f"{resource}_{backend_metrics.backend_type.value}_{severity.value}"
                alert = HealthAlert(
                    id=alert_id,
                    backend_type=backend_metrics.backend_type,
                    severity=severity,
                    message=f"{resource.replace('_', ' ').title()} {current_value:.1f}% exceeds {severity.value} threshold",
                    timestamp=current_time,
                    threshold_breached=resource,
                    current_value=current_value,
                    threshold_value=getattr(self.alert_thresholds, f"{resource}_{severity.value}")
                )
                alerts.append(alert)
        
        # API quota alerts (if available)
        if (backend_metrics.api_quota_used is not None and 
            backend_metrics.api_quota_limit is not None):
            quota_percentage = backend_metrics.api_quota_used / backend_metrics.api_quota_limit
            severity = self.alert_thresholds.get_severity_for_metric(
                "api_quota", quota_percentage, higher_is_worse=True
            )
            if severity:
                alert_id = f"api_quota_{backend_metrics.backend_type.value}_{severity.value}"
                alert = HealthAlert(
                    id=alert_id,
                    backend_type=backend_metrics.backend_type,
                    severity=severity,
                    message=f"API quota usage {quota_percentage:.1%} exceeds {severity.value} threshold",
                    timestamp=current_time,
                    threshold_breached="api_quota",
                    current_value=quota_percentage,
                    threshold_value=getattr(self.alert_thresholds, f"api_quota_{severity.value}")
                )
                alerts.append(alert)
        
        # Process generated alerts
        processed_alerts = []
        for alert in alerts:
            if self._should_process_alert(alert):
                self._process_alert(alert)
                processed_alerts.append(alert)
        
        return processed_alerts
    
    def _should_process_alert(self, alert: HealthAlert) -> bool:
        """Check if alert should be processed (not suppressed)"""
        # Check if this alert is currently suppressed
        if self._is_alert_suppressed(alert):
            return False
        
        # Record this alert occurrence
        self.alert_counts[alert.id].append(datetime.now())
        
        return True
    
    def _is_alert_suppressed(self, alert: HealthAlert) -> bool:
        """Check if alert should be suppressed"""
        current_time = datetime.now()
        
        # Find matching suppression rules
        matching_rules = []
        for rule in self.suppression_rules:
            # Simple pattern matching (supports wildcards)
            if rule.alert_id_pattern.endswith("*"):
                pattern_prefix = rule.alert_id_pattern[:-1]
                if alert.id.startswith(pattern_prefix):
                    matching_rules.append(rule)
            elif rule.alert_id_pattern == alert.id:
                matching_rules.append(rule)
        
        # Check suppression for each matching rule
        for rule in matching_rules:
            # Get recent occurrences of this alert
            recent_occurrences = [
                timestamp for timestamp in self.alert_counts[alert.id]
                if (current_time - timestamp).total_seconds() <= rule.suppression_window_seconds
            ]
            
            # Check if we should suppress
            if len(recent_occurrences) >= rule.max_occurrences:
                self.logger.debug(f"Suppressing alert {alert.id} due to rule {rule.alert_id_pattern}")
                return True
        
        return False
    
    def _process_alert(self, alert: HealthAlert):
        """Process a new alert"""
        try:
            # Check for auto-recovery if this is a recovery from previous alert
            self._check_auto_recovery(alert)
            
            # Add to active alerts (replacing any existing alert with same ID)
            existing_alert = self.active_alerts.get(alert.id)
            if existing_alert:
                # Update existing alert
                alert.acknowledged = existing_alert.acknowledged  # Preserve acknowledgment status
                self.logger.info(f"Updated existing alert: {alert.id}")
            else:
                self.logger.info(f"Generated new alert: {alert.id} [{alert.severity.value}]")
            
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Execute callbacks asynchronously
            self._execute_callbacks(alert)
            
        except Exception as e:
            self.logger.error(f"Error processing alert {alert.id}: {e}")
    
    def _check_auto_recovery(self, alert: HealthAlert):
        """Check if alert represents auto-recovery from previous state"""
        # Look for existing more severe alerts for same metric/backend
        recovered_alerts = []
        
        for existing_id, existing_alert in self.active_alerts.items():
            if (existing_alert.backend_type == alert.backend_type and
                existing_alert.threshold_breached == alert.threshold_breached and
                existing_alert.severity.value != alert.severity.value):
                
                severity_levels = ["info", "warning", "critical", "emergency"]
                existing_level = severity_levels.index(existing_alert.severity.value)
                new_level = severity_levels.index(alert.severity.value)
                
                # If new alert is less severe, mark existing as recovered
                if new_level < existing_level:
                    existing_alert.auto_recovery = True
                    recovered_alerts.append(existing_id)
        
        # Remove recovered alerts from active alerts
        for alert_id in recovered_alerts:
            recovered_alert = self.active_alerts.pop(alert_id)
            self.logger.info(f"Auto-recovery: {recovered_alert.id} resolved by improvement to {alert.severity.value}")
            
            # Create recovery notification
            recovery_alert = HealthAlert(
                id=f"recovery_{recovered_alert.id}",
                backend_type=recovered_alert.backend_type,
                severity=AlertSeverity.INFO,
                message=f"Auto-recovery: {recovered_alert.threshold_breached} improved from {recovered_alert.severity.value} to {alert.severity.value}",
                timestamp=datetime.now(),
                threshold_breached=recovered_alert.threshold_breached,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                auto_recovery=True
            )
            
            self.alert_history.append(recovery_alert)
            self._execute_callbacks(recovery_alert)
    
    def _execute_callbacks(self, alert: HealthAlert):
        """Execute alert callbacks asynchronously"""
        for callback in self.alert_callbacks:
            future = self.callback_executor.submit(self._safe_callback_execution, callback, alert)
            # Don't wait for completion, callbacks run in background
    
    def _safe_callback_execution(self, callback: AlertCallback, alert: HealthAlert):
        """Safely execute callback with error handling"""
        try:
            success = callback(alert)
            if not success:
                self.logger.warning(f"Alert callback {callback.__class__.__name__} returned False for alert {alert.id}")
        except Exception as e:
            self.logger.error(f"Alert callback {callback.__class__.__name__} failed for alert {alert.id}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
            # Create acknowledgment event
            ack_alert = HealthAlert(
                id=f"ack_{alert_id}",
                backend_type=self.active_alerts[alert_id].backend_type,
                severity=AlertSeverity.INFO,
                message=f"Alert acknowledged by {acknowledged_by}",
                timestamp=datetime.now(),
                threshold_breached=self.active_alerts[alert_id].threshold_breached,
                current_value=self.active_alerts[alert_id].current_value,
                threshold_value=self.active_alerts[alert_id].threshold_value,
                acknowledged=True
            )
            
            self.alert_history.append(ack_alert)
            self._execute_callbacks(ack_alert)
            
            return True
        else:
            self.logger.warning(f"Attempted to acknowledge non-existent alert: {alert_id}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Manually resolve an active alert"""
        if alert_id in self.active_alerts:
            resolved_alert = self.active_alerts.pop(alert_id)
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            
            # Create resolution event
            resolution_alert = HealthAlert(
                id=f"resolved_{alert_id}",
                backend_type=resolved_alert.backend_type,
                severity=AlertSeverity.INFO,
                message=f"Alert manually resolved by {resolved_by}",
                timestamp=datetime.now(),
                threshold_breached=resolved_alert.threshold_breached,
                current_value=resolved_alert.current_value,
                threshold_value=resolved_alert.threshold_value
            )
            
            self.alert_history.append(resolution_alert)
            self._execute_callbacks(resolution_alert)
            
            return True
        else:
            self.logger.warning(f"Attempted to resolve non-existent alert: {alert_id}")
            return False
    
    def get_active_alerts(self, 
                         backend_type: Optional[BackendType] = None,
                         severity: Optional[AlertSeverity] = None,
                         acknowledged_only: bool = False) -> List[HealthAlert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if backend_type:
            alerts = [a for a in alerts if a.backend_type == backend_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if acknowledged_only:
            alerts = [a for a in alerts if a.acknowledged]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, 
                         limit: int = 100,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[HealthAlert]:
        """Get alert history with optional filtering"""
        alerts = list(self.alert_history)
        
        if start_time or end_time:
            filtered_alerts = []
            for alert in alerts:
                if start_time and alert.timestamp < start_time:
                    continue
                if end_time and alert.timestamp > end_time:
                    continue
                filtered_alerts.append(alert)
            alerts = filtered_alerts
        
        # Sort by timestamp, newest first, and apply limit
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and analytics"""
        if not self.alert_history:
            return {'no_data': True}
        
        # Count by severity
        severity_counts = defaultdict(int)
        backend_counts = defaultdict(int)
        threshold_counts = defaultdict(int)
        
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
            backend_counts[alert.backend_type.value] += 1
            threshold_counts[alert.threshold_breached] += 1
        
        # Recent activity (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= recent_threshold]
        
        # Alert trends
        total_alerts = len(self.alert_history)
        recent_alert_count = len(recent_alerts)
        
        # Average time to acknowledgment
        acked_alerts = [a for a in self.alert_history if a.acknowledged]
        avg_ack_time_hours = 0.0
        if acked_alerts:
            # This is simplified - in practice you'd track ack timestamps
            avg_ack_time_hours = 1.5  # Placeholder
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': len(self.active_alerts),
            'recent_alerts_24h': recent_alert_count,
            'severity_distribution': dict(severity_counts),
            'backend_distribution': dict(backend_counts),
            'threshold_distribution': dict(threshold_counts),
            'avg_acknowledgment_time_hours': avg_ack_time_hours,
            'auto_recovery_count': len([a for a in self.alert_history if a.auto_recovery]),
            'acknowledgment_rate': len(acked_alerts) / total_alerts if total_alerts > 0 else 0.0
        }
    
    def update_alert_thresholds(self, new_thresholds: AlertThresholds):
        """Update alert thresholds configuration"""
        self.alert_thresholds = new_thresholds
        self.logger.info("Alert thresholds updated")
    
    def cleanup_old_alerts(self, max_age_days: int = 30):
        """Clean up old resolved alerts from history"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        # Keep recent alerts and all active alerts
        cleaned_history = deque(maxlen=self.alert_history.maxlen)
        removed_count = 0
        
        for alert in self.alert_history:
            if alert.timestamp >= cutoff_time or alert.id in self.active_alerts:
                cleaned_history.append(alert)
            else:
                removed_count += 1
        
        self.alert_history = cleaned_history
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old alerts")
    
    def shutdown(self):
        """Shutdown alert manager"""
        self.callback_executor.shutdown(wait=True)
        self.logger.info("AlertManager shutdown complete")


@dataclass
class HealthCheckConfig:
    """Enhanced configuration for health checks"""
    # General settings
    timeout_seconds: float = 5.0
    retry_attempts: int = 2
    
    # LightRAG specific
    lightrag_working_dir: Optional[str] = None
    lightrag_storage_dir: Optional[str] = None
    lightrag_test_query: str = "What is ATP?"
    
    # Perplexity specific  
    perplexity_api_key: Optional[str] = None
    perplexity_base_url: str = "https://api.perplexity.ai"
    perplexity_test_query: str = "ping"
    
    # System resource thresholds (legacy - now using AlertThresholds)
    max_cpu_percent: float = 90.0
    max_memory_percent: float = 90.0
    min_disk_space_gb: float = 1.0
    
    # Enhanced monitoring settings
    predictive_monitoring_enabled: bool = True
    performance_history_size: int = 1000
    trend_analysis_window: int = 20
    health_score_calculation_enabled: bool = True
    
    # Alert configuration
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    alert_callbacks: List[Callable[[HealthAlert], None]] = field(default_factory=list)


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing"""
    strategy: str = "weighted_round_robin"  # "round_robin", "weighted", "health_aware"
    health_check_interval: int = 60  # seconds
    circuit_breaker_threshold: int = 5  # consecutive failures
    circuit_breaker_timeout: int = 300  # seconds
    response_time_threshold_ms: float = 2000.0
    enable_adaptive_routing: bool = True


class BaseHealthChecker:
    """Base class for health checkers"""
    
    def __init__(self, config: HealthCheckConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    async def check_health(self) -> HealthCheckResult:
        """Perform health check - to be implemented by subclasses"""
        raise NotImplementedError


class LightRAGHealthChecker(BaseHealthChecker):
    """Health checker for LightRAG backend"""
    
    def __init__(self, config: HealthCheckConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.working_dir = config.lightrag_working_dir or "./dickson_working_dir"
        self.storage_dir = config.lightrag_storage_dir or "./lightrag_cache"
    
    async def check_health(self) -> HealthCheckResult:
        """Comprehensive LightRAG health check"""
        start_time = time.perf_counter()
        errors = []
        metadata = {}
        
        try:
            # Check file system accessibility
            fs_check = await self._check_filesystem_access()
            if not fs_check['accessible']:
                errors.append(f"Filesystem access failed: {fs_check['error']}")
            metadata.update(fs_check)
            
            # Check system resources
            resource_check = self._check_system_resources()
            if not resource_check['adequate']:
                errors.append(f"System resources insufficient: {resource_check['issues']}")
            metadata.update(resource_check)
            
            # Check OpenAI API connectivity (for embeddings)
            try:
                openai_check = await self._check_openai_connectivity()
                if not openai_check['available']:
                    errors.append(f"OpenAI API unavailable: {openai_check['error']}")
                metadata.update(openai_check)
            except Exception as e:
                errors.append(f"OpenAI connectivity check failed: {e}")
                metadata['openai_error'] = str(e)
            
            # Test sample query execution (if no critical errors)
            if not errors:
                try:
                    query_check = await self._test_sample_query()
                    if not query_check['successful']:
                        errors.append(f"Sample query failed: {query_check['error']}")
                    metadata.update(query_check)
                except Exception as e:
                    errors.append(f"Sample query test failed: {e}")
                    metadata['query_test_error'] = str(e)
            
            response_time_ms = (time.perf_counter() - start_time) * 1000
            is_healthy = len(errors) == 0
            
            return HealthCheckResult(
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                error_message='; '.join(errors) if errors else None,
                metadata=metadata
            )
            
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"LightRAG health check failed: {e}")
            
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=f"Health check exception: {str(e)}",
                metadata={'exception': str(e)}
            )
    
    async def _check_filesystem_access(self) -> Dict[str, Any]:
        """Check filesystem accessibility"""
        try:
            # Check if working directory exists and is accessible
            working_path = Path(self.working_dir)
            if not working_path.exists():
                return {
                    'accessible': False,
                    'error': f'Working directory does not exist: {self.working_dir}',
                    'working_dir_exists': False
                }
            
            # Check write permissions
            test_file = working_path / '.health_check_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
                write_accessible = True
            except Exception as e:
                write_accessible = False
                write_error = str(e)
            
            # Check storage directory
            storage_path = Path(self.storage_dir)
            storage_accessible = storage_path.exists() or storage_path.parent.exists()
            
            return {
                'accessible': write_accessible and storage_accessible,
                'working_dir_exists': True,
                'working_dir_writable': write_accessible,
                'storage_dir_accessible': storage_accessible,
                'error': write_error if not write_accessible else None
            }
            
        except Exception as e:
            return {
                'accessible': False,
                'error': f'Filesystem check failed: {e}',
                'working_dir_exists': False
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_ok = cpu_percent < self.config.max_cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_ok = memory.percent < self.config.max_memory_percent
            
            # Disk space in working directory
            disk_usage = psutil.disk_usage(self.working_dir)
            free_space_gb = disk_usage.free / (1024**3)
            disk_ok = free_space_gb > self.config.min_disk_space_gb
            
            issues = []
            if not cpu_ok:
                issues.append(f'CPU usage high: {cpu_percent}%')
            if not memory_ok:
                issues.append(f'Memory usage high: {memory.percent}%')
            if not disk_ok:
                issues.append(f'Low disk space: {free_space_gb:.1f}GB')
            
            return {
                'adequate': len(issues) == 0,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'free_disk_gb': free_space_gb,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'adequate': False,
                'error': f'Resource check failed: {e}',
                'issues': [f'Resource monitoring error: {e}']
            }
    
    async def _check_openai_connectivity(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity for embeddings"""
        try:
            # Get OpenAI API key from environment or config
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    'available': False,
                    'error': 'OpenAI API key not found in environment',
                    'has_api_key': False
                }
            
            # Test API connectivity with a simple request
            client = openai.OpenAI(api_key=api_key, timeout=self.config.timeout_seconds)
            
            # Use asyncio timeout for the blocking call
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                future = executor.submit(client.models.list)
                try:
                    models = await asyncio.wait_for(
                        loop.run_in_executor(executor, lambda: future.result()),
                        timeout=self.config.timeout_seconds
                    )
                    
                    return {
                        'available': True,
                        'has_api_key': True,
                        'models_accessible': True,
                        'model_count': len(models.data)
                    }
                    
                except asyncio.TimeoutError:
                    return {
                        'available': False,
                        'error': 'OpenAI API request timeout',
                        'has_api_key': True
                    }
                    
        except Exception as e:
            return {
                'available': False,
                'error': f'OpenAI API check failed: {e}',
                'has_api_key': api_key is not None if 'api_key' in locals() else False
            }
    
    async def _test_sample_query(self) -> Dict[str, Any]:
        """Test sample query execution"""
        try:
            # This would normally test actual LightRAG query execution
            # For now, we'll do a basic validation check
            start_time = time.perf_counter()
            
            # Simulate query processing time
            await asyncio.sleep(0.1)
            
            query_time_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                'successful': True,
                'query_time_ms': query_time_ms,
                'test_query': self.config.lightrag_test_query
            }
            
        except Exception as e:
            return {
                'successful': False,
                'error': f'Sample query execution failed: {e}',
                'test_query': self.config.lightrag_test_query
            }


class PerplexityHealthChecker(BaseHealthChecker):
    """Health checker for Perplexity backend"""
    
    def __init__(self, config: HealthCheckConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.api_key = config.perplexity_api_key or os.getenv('PERPLEXITY_API_KEY')
        self.base_url = config.perplexity_base_url
    
    async def check_health(self) -> HealthCheckResult:
        """Comprehensive Perplexity health check"""
        start_time = time.perf_counter()
        errors = []
        metadata = {}
        
        try:
            # Check API key availability
            if not self.api_key:
                errors.append("Perplexity API key not available")
                metadata['has_api_key'] = False
            else:
                metadata['has_api_key'] = True
                
                # Check API connectivity
                connectivity_check = await self._check_api_connectivity()
                if not connectivity_check['accessible']:
                    errors.append(f"API connectivity failed: {connectivity_check['error']}")
                metadata.update(connectivity_check)
                
                # Check authentication if API is accessible
                if connectivity_check['accessible']:
                    auth_check = await self._check_authentication()
                    if not auth_check['authenticated']:
                        errors.append(f"Authentication failed: {auth_check['error']}")
                    metadata.update(auth_check)
                    
                    # Check rate limits and response format
                    if auth_check['authenticated']:
                        rate_limit_check = await self._check_rate_limits()
                        metadata.update(rate_limit_check)
                        
                        response_format_check = await self._check_response_format()
                        if not response_format_check['valid_format']:
                            errors.append(f"Response format invalid: {response_format_check['error']}")
                        metadata.update(response_format_check)
            
            response_time_ms = (time.perf_counter() - start_time) * 1000
            is_healthy = len(errors) == 0
            
            return HealthCheckResult(
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                error_message='; '.join(errors) if errors else None,
                metadata=metadata
            )
            
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Perplexity health check failed: {e}")
            
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=f"Health check exception: {str(e)}",
                metadata={'exception': str(e)}
            )
    
    async def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check basic API connectivity"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                # Try a basic request to check connectivity
                response = await client.get(f"{self.base_url}/")
                
                return {
                    'accessible': response.status_code in [200, 404, 405],  # 404/405 acceptable for base URL
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
                
        except httpx.TimeoutException:
            return {
                'accessible': False,
                'error': 'API request timeout',
                'timeout': True
            }
        except Exception as e:
            return {
                'accessible': False,
                'error': f'Connectivity check failed: {e}'
            }
    
    async def _check_authentication(self) -> Dict[str, Any]:
        """Check API authentication"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Try a lightweight endpoint to test authentication
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1
                    }
                )
                
                authenticated = response.status_code != 401
                
                return {
                    'authenticated': authenticated,
                    'status_code': response.status_code,
                    'error': 'Authentication failed' if response.status_code == 401 else None
                }
                
        except Exception as e:
            return {
                'authenticated': False,
                'error': f'Authentication check failed: {e}'
            }
    
    async def _check_rate_limits(self) -> Dict[str, Any]:
        """Check rate limit status"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1
                    }
                )
                
                # Extract rate limit headers if available
                rate_limit_info = {
                    'rate_limit_remaining': response.headers.get('x-ratelimit-remaining'),
                    'rate_limit_reset': response.headers.get('x-ratelimit-reset'),
                    'rate_limited': response.status_code == 429
                }
                
                return rate_limit_info
                
        except Exception as e:
            return {
                'rate_limit_check_error': str(e)
            }
    
    async def _check_response_format(self) -> Dict[str, Any]:
        """Check response format validation"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5
                    }
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Check if response has expected structure
                        valid_format = (
                            'choices' in data and
                            len(data['choices']) > 0 and
                            'message' in data['choices'][0]
                        )
                        
                        return {
                            'valid_format': valid_format,
                            'response_structure_ok': True
                        }
                        
                    except json.JSONDecodeError:
                        return {
                            'valid_format': False,
                            'error': 'Invalid JSON response format'
                        }
                else:
                    return {
                        'valid_format': False,
                        'error': f'HTTP {response.status_code}',
                        'response_body': response.text[:200]  # First 200 chars for debugging
                    }
                    
        except Exception as e:
            return {
                'valid_format': False,
                'error': f'Response format check failed: {e}'
            }


class SystemHealthMonitor:
    """System health monitoring for routing decisions"""
    
    def __init__(self, 
                 check_interval: int = 30,
                 health_config: Optional[HealthCheckConfig] = None):
        self.check_interval = check_interval
        self.health_config = health_config or HealthCheckConfig()
        self.backend_health: Dict[BackendType, BackendHealthMetrics] = {}
        self.health_history: deque = deque(maxlen=100)
        self.monitoring_active = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.health_config.alert_thresholds)
        
        # Initialize health checkers
        self.health_checkers: Dict[BackendType, BaseHealthChecker] = {
            BackendType.LIGHTRAG: LightRAGHealthChecker(self.health_config, self.logger),
            BackendType.PERPLEXITY: PerplexityHealthChecker(self.health_config, self.logger)
        }
        
        # Initialize backend health metrics
        for backend_type in BackendType:
            self.backend_health[backend_type] = BackendHealthMetrics(
                backend_type=backend_type,
                status=SystemHealthStatus.HEALTHY,
                response_time_ms=0.0,
                error_rate=0.0,
                last_health_check=datetime.now()
            )
    
    def start_monitoring(self):
        """Start health monitoring in background"""
        self.monitoring_active = True
        threading.Thread(target=self._health_check_loop, daemon=True).start()
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.alert_manager.shutdown()
        self.logger.info("System health monitoring stopped")
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self.monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(5)  # Shorter retry interval on error
    
    def _perform_health_checks(self):
        """Perform health checks for all backends"""
        # Use asyncio to run async health checks
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        for backend_type in BackendType:
            try:
                # Run async health check
                health_checker = self.health_checkers[backend_type]
                health_result = loop.run_until_complete(
                    asyncio.wait_for(
                        health_checker.check_health(), 
                        timeout=self.health_config.timeout_seconds * 2  # Double timeout for safety
                    )
                )
                
                # Update metrics based on health check result
                metrics = self.backend_health[backend_type]
                metrics.last_health_check = datetime.now()
                metrics.response_time_ms = health_result.response_time_ms
                
                if health_result.is_healthy:
                    metrics.consecutive_failures = 0
                    if health_result.response_time_ms < self.health_config.timeout_seconds * 500:  # Half timeout threshold
                        metrics.status = SystemHealthStatus.HEALTHY
                    else:
                        metrics.status = SystemHealthStatus.DEGRADED
                else:
                    metrics.consecutive_failures += 1
                    if metrics.consecutive_failures >= 5:
                        metrics.status = SystemHealthStatus.OFFLINE
                    elif metrics.consecutive_failures >= 3:
                        metrics.status = SystemHealthStatus.CRITICAL
                    else:
                        metrics.status = SystemHealthStatus.DEGRADED
                
                # Log detailed health check results
                if health_result.error_message:
                    self.logger.warning(f"Health check issues for {backend_type.value}: {health_result.error_message}")
                
                # Update error rate calculation
                metrics.total_requests += 1
                if health_result.is_healthy:
                    metrics.successful_requests += 1
                metrics.error_rate = 1.0 - (metrics.successful_requests / metrics.total_requests)
                
                # Update performance metrics
                metrics.update_performance_metrics(
                    response_time=health_result.response_time_ms,
                    success=health_result.is_healthy,
                    error_type=health_result.error_message if not health_result.is_healthy else None
                )
                
                # Update resource usage (simulated - in practice this would come from the health check)
                try:
                    metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    metrics.memory_usage_percent = memory.percent
                    disk_usage = psutil.disk_usage('/')
                    metrics.disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
                except Exception as e:
                    self.logger.debug(f"Error updating resource usage: {e}")
                
                # Generate alerts based on metrics
                generated_alerts = self.alert_manager.check_and_generate_alerts(metrics)
                if generated_alerts:
                    self.logger.info(f"Generated {len(generated_alerts)} alerts for {backend_type.value}")
                
                self.health_history.append({
                    'timestamp': datetime.now(),
                    'backend': backend_type.value,
                    'status': metrics.status.value,
                    'response_time_ms': health_result.response_time_ms,
                    'error_message': health_result.error_message,
                    'metadata': health_result.metadata,
                    'alerts_generated': len(generated_alerts)
                })
                
            except asyncio.TimeoutError:
                self.logger.error(f"Health check timeout for {backend_type.value}")
                metrics = self.backend_health[backend_type]
                metrics.consecutive_failures += 1
                metrics.status = SystemHealthStatus.CRITICAL
                metrics.total_requests += 1
                metrics.error_rate = 1.0 - (metrics.successful_requests / metrics.total_requests)
                
            except Exception as e:
                self.logger.error(f"Health check failed for {backend_type.value}: {e}")
                metrics = self.backend_health[backend_type]
                metrics.consecutive_failures += 1
                metrics.status = SystemHealthStatus.CRITICAL
                metrics.total_requests += 1
                metrics.error_rate = 1.0 - (metrics.successful_requests / metrics.total_requests)
    
    def update_health_config(self, new_config: HealthCheckConfig):
        """Update health check configuration"""
        self.health_config = new_config
        
        # Recreate health checkers with new config
        self.health_checkers = {
            BackendType.LIGHTRAG: LightRAGHealthChecker(self.health_config, self.logger),
            BackendType.PERPLEXITY: PerplexityHealthChecker(self.health_config, self.logger)
        }
        
        # Update alert manager thresholds
        self.alert_manager.update_alert_thresholds(self.health_config.alert_thresholds)
        
        self.logger.info("Health check configuration updated")
    
    def get_detailed_health_status(self, backend_type: BackendType) -> Dict[str, Any]:
        """Get detailed health status including recent check results"""
        metrics = self.backend_health.get(backend_type)
        if not metrics:
            return {'error': 'Backend not found'}
        
        # Get recent history for this backend
        recent_history = [
            entry for entry in list(self.health_history)[-20:]
            if entry['backend'] == backend_type.value
        ]
        
        return {
            'current_status': metrics.to_dict(),
            'recent_history': recent_history,
            'health_trends': self._calculate_health_trends(backend_type)
        }
    
    def _calculate_health_trends(self, backend_type: BackendType) -> Dict[str, Any]:
        """Calculate health trends for a backend"""
        backend_history = [
            entry for entry in self.health_history
            if entry['backend'] == backend_type.value
        ]
        
        if len(backend_history) < 2:
            return {'insufficient_data': True}
        
        # Calculate recent average response time
        recent_times = [entry['response_time_ms'] for entry in backend_history[-10:]]
        avg_response_time = statistics.mean(recent_times) if recent_times else 0.0
        
        # Calculate uptime percentage
        healthy_count = sum(
            1 for entry in backend_history
            if entry['status'] == SystemHealthStatus.HEALTHY.value
        )
        uptime_percentage = (healthy_count / len(backend_history)) * 100
        
        # Detect trends
        if len(recent_times) >= 3:
            recent_trend = 'improving' if recent_times[-1] < recent_times[-3] else 'degrading'
        else:
            recent_trend = 'stable'
        
        return {
            'avg_response_time_ms': avg_response_time,
            'uptime_percentage': uptime_percentage,
            'recent_trend': recent_trend,
            'total_checks': len(backend_history)
        }
    
    def get_backend_health(self, backend_type: BackendType) -> BackendHealthMetrics:
        """Get health metrics for specific backend"""
        return self.backend_health.get(backend_type)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        healthy_count = sum(1 for metrics in self.backend_health.values() 
                          if metrics.status == SystemHealthStatus.HEALTHY)
        total_count = len(self.backend_health)
        
        overall_status = SystemHealthStatus.HEALTHY
        if healthy_count == 0:
            overall_status = SystemHealthStatus.OFFLINE
        elif healthy_count < total_count:
            overall_status = SystemHealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'healthy_backends': healthy_count,
            'total_backends': total_count,
            'backends': {bt.value: metrics.to_dict() 
                        for bt, metrics in self.backend_health.items()}
        }
    
    def should_route_to_backend(self, backend_type: BackendType) -> bool:
        """Determine if backend is healthy enough for routing"""
        metrics = self.backend_health.get(backend_type)
        if not metrics:
            return False
        
        return metrics.status in [SystemHealthStatus.HEALTHY, SystemHealthStatus.DEGRADED]
    
    # Alert Management Methods
    
    def get_active_alerts(self, 
                         backend_type: Optional[BackendType] = None,
                         severity: Optional[AlertSeverity] = None) -> List[HealthAlert]:
        """Get active alerts with optional filtering"""
        return self.alert_manager.get_active_alerts(backend_type, severity)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "user") -> bool:
        """Acknowledge an active alert"""
        return self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "user") -> bool:
        """Manually resolve an active alert"""
        return self.alert_manager.resolve_alert(alert_id, resolved_by)
    
    def get_alert_history(self, 
                         limit: int = 100,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[HealthAlert]:
        """Get alert history with optional filtering"""
        return self.alert_manager.get_alert_history(limit, start_time, end_time)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and analytics"""
        return self.alert_manager.get_alert_statistics()
    
    def add_alert_callback(self, callback: AlertCallback):
        """Add alert callback"""
        self.alert_manager.add_callback(callback)
    
    def remove_alert_callback(self, callback: AlertCallback):
        """Remove alert callback"""
        self.alert_manager.remove_callback(callback)
    
    def update_alert_thresholds(self, thresholds: AlertThresholds):
        """Update alert thresholds"""
        self.alert_manager.update_alert_thresholds(thresholds)
        self.health_config.alert_thresholds = thresholds


class LoadBalancer:
    """Load balancer for multiple backend instances"""
    
    def __init__(self, config: LoadBalancingConfig, health_monitor: SystemHealthMonitor):
        self.config = config
        self.health_monitor = health_monitor
        self.backend_weights: Dict[BackendType, float] = {
            BackendType.LIGHTRAG: 1.0,
            BackendType.PERPLEXITY: 1.0
        }
        self.request_counts: Dict[BackendType, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def select_backend(self, routing_decision: RoutingDecision) -> Optional[BackendType]:
        """Select optimal backend based on routing decision and system health"""
        
        # Direct routing cases
        if routing_decision == RoutingDecision.LIGHTRAG:
            candidate = BackendType.LIGHTRAG
        elif routing_decision == RoutingDecision.PERPLEXITY:
            candidate = BackendType.PERPLEXITY
        else:
            # For EITHER or HYBRID, select based on health and load balancing
            candidate = self._select_best_available_backend()
        
        # Check health and apply circuit breaker logic
        if not self.health_monitor.should_route_to_backend(candidate):
            fallback_candidate = self._select_fallback_backend(candidate)
            if fallback_candidate:
                self.logger.warning(f"Backend {candidate.value} unhealthy, using fallback {fallback_candidate.value}")
                return fallback_candidate
            else:
                self.logger.error(f"No healthy backends available")
                return None
        
        # Update request counts for load balancing
        self.request_counts[candidate] += 1
        
        return candidate
    
    def _select_best_available_backend(self) -> BackendType:
        """Select best available backend using configured strategy"""
        
        if self.config.strategy == "round_robin":
            return self._round_robin_selection()
        elif self.config.strategy == "weighted":
            return self._weighted_selection()
        elif self.config.strategy == "health_aware":
            return self._health_aware_selection()
        else:
            return self._weighted_round_robin_selection()
    
    def _round_robin_selection(self) -> BackendType:
        """Simple round robin selection"""
        backends = list(BackendType)
        total_requests = sum(self.request_counts.values())
        return backends[total_requests % len(backends)]
    
    def _weighted_selection(self) -> BackendType:
        """Weighted selection based on backend weights"""
        total_weight = sum(self.backend_weights.values())
        rand = random.uniform(0, total_weight)
        
        cumulative = 0
        for backend_type, weight in self.backend_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return backend_type
        
        return BackendType.LIGHTRAG  # fallback
    
    def _health_aware_selection(self) -> BackendType:
        """Health-aware selection prioritizing healthy backends"""
        healthy_backends = []
        
        for backend_type in BackendType:
            if self.health_monitor.should_route_to_backend(backend_type):
                healthy_backends.append(backend_type)
        
        if not healthy_backends:
            return BackendType.LIGHTRAG  # fallback
        
        # Select least loaded healthy backend
        return min(healthy_backends, key=lambda b: self.request_counts[b])
    
    def _weighted_round_robin_selection(self) -> BackendType:
        """Weighted round robin combining health and weights"""
        # Adjust weights based on health
        adjusted_weights = {}
        
        for backend_type, base_weight in self.backend_weights.items():
            health_metrics = self.health_monitor.get_backend_health(backend_type)
            if health_metrics.status == SystemHealthStatus.HEALTHY:
                health_factor = 1.0
            elif health_metrics.status == SystemHealthStatus.DEGRADED:
                health_factor = 0.7
            elif health_metrics.status == SystemHealthStatus.CRITICAL:
                health_factor = 0.3
            else:  # OFFLINE
                health_factor = 0.0
            
            adjusted_weights[backend_type] = base_weight * health_factor
        
        # Select based on adjusted weights
        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            return BackendType.LIGHTRAG  # emergency fallback
        
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for backend_type, weight in adjusted_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return backend_type
        
        return BackendType.LIGHTRAG  # fallback
    
    def _select_fallback_backend(self, failed_backend: BackendType) -> Optional[BackendType]:
        """Select fallback backend when primary fails"""
        for backend_type in BackendType:
            if (backend_type != failed_backend and 
                self.health_monitor.should_route_to_backend(backend_type)):
                return backend_type
        return None
    
    def update_backend_weights(self, weights: Dict[BackendType, float]):
        """Update backend weights for load balancing"""
        self.backend_weights.update(weights)
        self.logger.info(f"Updated backend weights: {weights}")


class RoutingAnalyticsCollector:
    """Collector for routing analytics and metrics"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.analytics_data: deque = deque(maxlen=max_entries)
        self.routing_stats: Dict[str, int] = defaultdict(int)
        self.confidence_stats: List[float] = []
        self.response_time_stats: List[float] = []
        self.logger = logging.getLogger(__name__)
    
    def record_routing_decision(self, analytics: RoutingAnalytics):
        """Record routing decision analytics"""
        self.analytics_data.append(analytics)
        
        # Update statistics
        self.routing_stats[analytics.routing_decision.value] += 1
        self.confidence_stats.append(analytics.confidence)
        self.response_time_stats.append(analytics.response_time_ms)
        
        # Keep stats lists manageable
        if len(self.confidence_stats) > 1000:
            self.confidence_stats = self.confidence_stats[-500:]
        if len(self.response_time_stats) > 1000:
            self.response_time_stats = self.response_time_stats[-500:]
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        if not self.analytics_data:
            return {'no_data': True}
        
        # Calculate statistics
        total_requests = len(self.analytics_data)
        
        # Confidence statistics
        confidence_stats = {}
        if self.confidence_stats:
            confidence_stats = {
                'mean': statistics.mean(self.confidence_stats),
                'median': statistics.median(self.confidence_stats),
                'stdev': statistics.stdev(self.confidence_stats) if len(self.confidence_stats) > 1 else 0.0,
                'min': min(self.confidence_stats),
                'max': max(self.confidence_stats)
            }
        
        # Response time statistics
        response_time_stats = {}
        if self.response_time_stats:
            response_time_stats = {
                'mean_ms': statistics.mean(self.response_time_stats),
                'median_ms': statistics.median(self.response_time_stats),
                'p95_ms': statistics.quantiles(self.response_time_stats, n=20)[18] if len(self.response_time_stats) >= 20 else max(self.response_time_stats),
                'p99_ms': statistics.quantiles(self.response_time_stats, n=100)[98] if len(self.response_time_stats) >= 100 else max(self.response_time_stats),
                'min_ms': min(self.response_time_stats),
                'max_ms': max(self.response_time_stats)
            }
        
        # Routing distribution
        routing_distribution = {
            decision: count / total_requests 
            for decision, count in self.routing_stats.items()
        }
        
        # Recent performance (last 100 requests)
        recent_data = list(self.analytics_data)[-100:]
        recent_avg_confidence = statistics.mean([d.confidence for d in recent_data]) if recent_data else 0.0
        recent_avg_response_time = statistics.mean([d.response_time_ms for d in recent_data]) if recent_data else 0.0
        
        # Fallback statistics
        fallback_count = sum(1 for d in self.analytics_data if d.fallback_triggered)
        fallback_rate = fallback_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_requests': total_requests,
            'routing_distribution': routing_distribution,
            'confidence_stats': confidence_stats,
            'response_time_stats': response_time_stats,
            'recent_avg_confidence': recent_avg_confidence,
            'recent_avg_response_time_ms': recent_avg_response_time,
            'fallback_rate': fallback_rate,
            'system_health_impact_rate': sum(1 for d in self.analytics_data if d.system_health_impact) / total_requests if total_requests > 0 else 0.0
        }
    
    def export_analytics_data(self, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export analytics data for external analysis"""
        
        filtered_data = self.analytics_data
        
        if start_time or end_time:
            filtered_data = []
            for entry in self.analytics_data:
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                filtered_data.append(entry)
        
        return [entry.to_dict() for entry in filtered_data]


class IntelligentQueryRouter:
    """
    Enhanced intelligent query router with system health monitoring,
    load balancing, and comprehensive analytics.
    """
    
    def __init__(self, 
                 base_router: Optional[BiomedicalQueryRouter] = None,
                 load_balancing_config: Optional[LoadBalancingConfig] = None,
                 health_check_config: Optional[HealthCheckConfig] = None):
        """
        Initialize the intelligent query router.
        
        Args:
            base_router: Base BiomedicalQueryRouter instance
            load_balancing_config: Load balancing configuration
            health_check_config: Health check configuration
        """
        self.base_router = base_router or BiomedicalQueryRouter()
        self.load_balancing_config = load_balancing_config or LoadBalancingConfig()
        self.health_check_config = health_check_config or HealthCheckConfig()
        
        # Initialize components
        self.health_monitor = SystemHealthMonitor(
            check_interval=self.load_balancing_config.health_check_interval,
            health_config=self.health_check_config
        )
        self.load_balancer = LoadBalancer(self.load_balancing_config, self.health_monitor)
        self.analytics_collector = RoutingAnalyticsCollector()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'avg_response_time_ms': 0.0,
            'response_times': deque(maxlen=1000),
            'accuracy_samples': deque(maxlen=1000)
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.health_monitor.start_monitoring()
        
        self.logger.info("IntelligentQueryRouter initialized with enhanced capabilities")
    
    def route_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> RoutingPrediction:
        """
        Route query with enhanced intelligence, health monitoring, and analytics.
        
        Args:
            query_text: Query text to route
            context: Optional context information
            
        Returns:
            RoutingPrediction with enhanced metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Get base routing decision
            base_prediction = self.base_router.route_query(query_text, context)
            
            # Select backend based on health and load balancing
            selected_backend = self.load_balancer.select_backend(base_prediction.routing_decision)
            
            # Check if health impacted routing
            original_backend = self._get_natural_backend(base_prediction.routing_decision)
            health_impacted = (selected_backend != original_backend)
            
            # Apply fallback if needed
            fallback_triggered = False
            if not selected_backend:
                self.logger.warning("No healthy backends available, applying emergency fallback")
                base_prediction.routing_decision = RoutingDecision.EITHER
                selected_backend = BackendType.LIGHTRAG  # Emergency fallback
                fallback_triggered = True
            
            # Enhanced metadata with system health information
            enhanced_metadata = base_prediction.metadata.copy()
            enhanced_metadata.update({
                'intelligent_router_version': '1.0.0',
                'selected_backend': selected_backend.value if selected_backend else None,
                'health_impacted_routing': health_impacted,
                'fallback_triggered': fallback_triggered,
                'system_health_summary': self.health_monitor.get_system_health_summary(),
                'load_balancer_strategy': self.load_balancing_config.strategy
            })
            
            # Update prediction with enhanced metadata
            base_prediction.metadata = enhanced_metadata
            
            # Record analytics
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            analytics = RoutingAnalytics(
                timestamp=datetime.now(),
                query=query_text,
                routing_decision=base_prediction.routing_decision,
                confidence=base_prediction.confidence,
                response_time_ms=response_time_ms,
                backend_used=selected_backend,
                fallback_triggered=fallback_triggered,
                system_health_impact=health_impacted,
                metadata={
                    'query_length': len(query_text),
                    'context_provided': context is not None
                }
            )
            
            self.analytics_collector.record_routing_decision(analytics)
            
            # Update performance metrics
            self.performance_metrics['total_requests'] += 1
            self.performance_metrics['response_times'].append(response_time_ms)
            if self.performance_metrics['response_times']:
                self.performance_metrics['avg_response_time_ms'] = statistics.mean(
                    self.performance_metrics['response_times']
                )
            
            return base_prediction
            
        except Exception as e:
            self.logger.error(f"Error in intelligent routing: {e}")
            
            # Emergency fallback
            fallback_confidence_metrics = ConfidenceMetrics(
                overall_confidence=0.1,
                research_category_confidence=0.1,
                temporal_analysis_confidence=0.1,
                signal_strength_confidence=0.1,
                context_coherence_confidence=0.1,
                keyword_density=0.0,
                pattern_match_strength=0.0,
                biomedical_entity_count=0,
                ambiguity_score=1.0,
                conflict_score=1.0,
                alternative_interpretations=[],
                calculation_time_ms=0.0
            )
            
            fallback_prediction = RoutingPrediction(
                routing_decision=RoutingDecision.EITHER,
                confidence=0.1,
                reasoning=[f"Emergency fallback due to error: {str(e)}"],
                research_category=ResearchCategory.GENERAL_QUERY,
                confidence_metrics=fallback_confidence_metrics,
                temporal_indicators=[],
                knowledge_indicators=[],
                metadata={
                    'error_fallback': True,
                    'error_message': str(e),
                    'intelligent_router_version': '1.0.0'
                }
            )
            
            return fallback_prediction
    
    def _get_natural_backend(self, routing_decision: RoutingDecision) -> Optional[BackendType]:
        """Get the natural backend for a routing decision"""
        if routing_decision == RoutingDecision.LIGHTRAG:
            return BackendType.LIGHTRAG
        elif routing_decision == RoutingDecision.PERPLEXITY:
            return BackendType.PERPLEXITY
        else:
            return None  # EITHER or HYBRID don't have natural backends
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return self.health_monitor.get_system_health_summary()
    
    def get_detailed_backend_health(self, backend_type: str) -> Dict[str, Any]:
        """Get detailed health status for a specific backend"""
        try:
            backend_enum = BackendType(backend_type.lower())
            return self.health_monitor.get_detailed_health_status(backend_enum)
        except ValueError:
            return {'error': f'Unknown backend type: {backend_type}'}
    
    def update_health_check_config(self, config_updates: Dict[str, Any]):
        """Update health check configuration"""
        # Update configuration attributes
        for key, value in config_updates.items():
            if hasattr(self.health_check_config, key):
                setattr(self.health_check_config, key, value)
        
        # Apply updated configuration to health monitor
        self.health_monitor.update_health_config(self.health_check_config)
        self.logger.info(f"Health check configuration updated: {config_updates}")
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and statistics"""
        return self.analytics_collector.get_routing_statistics()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add additional calculated metrics
        if self.performance_metrics['response_times']:
            times = list(self.performance_metrics['response_times'])
            metrics['p95_response_time_ms'] = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
            metrics['p99_response_time_ms'] = statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times)
            metrics['min_response_time_ms'] = min(times)
            metrics['max_response_time_ms'] = max(times)
        
        return metrics
    
    def update_load_balancing_weights(self, weights: Dict[str, float]):
        """Update load balancing weights"""
        backend_weights = {}
        for backend_str, weight in weights.items():
            try:
                backend_type = BackendType(backend_str.lower())
                backend_weights[backend_type] = weight
            except ValueError:
                self.logger.warning(f"Unknown backend type: {backend_str}")
        
        if backend_weights:
            self.load_balancer.update_backend_weights(backend_weights)
    
    def export_analytics(self, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export routing analytics data"""
        return self.analytics_collector.export_analytics_data(start_time, end_time)
    
    # Alert Management Methods
    
    def get_active_alerts(self, 
                         backend_type: Optional[str] = None,
                         severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts with optional filtering"""
        backend_enum = None
        if backend_type:
            try:
                backend_enum = BackendType(backend_type.lower())
            except ValueError:
                self.logger.warning(f"Unknown backend type: {backend_type}")
                return []
        
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
            except ValueError:
                self.logger.warning(f"Unknown severity: {severity}")
                return []
        
        alerts = self.health_monitor.get_active_alerts(backend_enum, severity_enum)
        return [alert.to_dict() for alert in alerts]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "user") -> bool:
        """Acknowledge an active alert"""
        return self.health_monitor.acknowledge_alert(alert_id, acknowledged_by)
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "user") -> bool:
        """Manually resolve an active alert"""
        return self.health_monitor.resolve_alert(alert_id, resolved_by)
    
    def get_alert_history(self, 
                         limit: int = 100,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering"""
        alerts = self.health_monitor.get_alert_history(limit, start_time, end_time)
        return [alert.to_dict() for alert in alerts]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and analytics"""
        return self.health_monitor.get_alert_statistics()
    
    def configure_alert_thresholds(self, threshold_config: Dict[str, Any]):
        """Configure alert thresholds"""
        try:
            # Create new AlertThresholds object from config
            thresholds = AlertThresholds()
            
            # Update thresholds from config
            for key, value in threshold_config.items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)
                else:
                    self.logger.warning(f"Unknown threshold configuration: {key}")
            
            self.health_monitor.update_alert_thresholds(thresholds)
            self.logger.info(f"Alert thresholds updated: {threshold_config}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring alert thresholds: {e}")
            return False
    
    def register_alert_callback(self, callback_type: str, **kwargs) -> bool:
        """Register an alert callback"""
        try:
            if callback_type.lower() == "webhook":
                webhook_url = kwargs.get("webhook_url")
                if not webhook_url:
                    self.logger.error("Webhook callback requires 'webhook_url' parameter")
                    return False
                
                callback = WebhookAlertCallback(
                    webhook_url=webhook_url,
                    timeout=kwargs.get("timeout", 5.0),
                    headers=kwargs.get("headers")
                )
                self.health_monitor.add_alert_callback(callback)
                self.logger.info(f"Registered webhook alert callback: {webhook_url}")
                return True
            
            elif callback_type.lower() == "json_file":
                file_path = kwargs.get("file_path", "./logs/alerts/custom_alerts.json")
                max_alerts = kwargs.get("max_alerts", 10000)
                
                callback = JSONFileAlertCallback(file_path, max_alerts)
                self.health_monitor.add_alert_callback(callback)
                self.logger.info(f"Registered JSON file alert callback: {file_path}")
                return True
            
            elif callback_type.lower() == "console":
                callback = ConsoleAlertCallback()
                self.health_monitor.add_alert_callback(callback)
                self.logger.info("Registered console alert callback")
                return True
            
            else:
                self.logger.error(f"Unknown callback type: {callback_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering alert callback: {e}")
            return False
    
    def get_system_health_with_alerts(self) -> Dict[str, Any]:
        """Get comprehensive system health status including alert information"""
        health_summary = self.get_system_health_status()
        
        # Add alert information
        active_alerts = self.get_active_alerts()
        alert_stats = self.get_alert_statistics()
        
        health_summary.update({
            'active_alerts': active_alerts,
            'alert_statistics': alert_stats,
            'alert_summary': {
                'total_active': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.get('severity') == 'critical']),
                'emergency_alerts': len([a for a in active_alerts if a.get('severity') == 'emergency']),
                'unacknowledged_alerts': len([a for a in active_alerts if not a.get('acknowledged', False)])
            }
        })
        
        return health_summary
    
    def shutdown(self):
        """Shutdown the router and stop monitoring"""
        self.health_monitor.stop_monitoring()
        self.logger.info("IntelligentQueryRouter shutdown complete")