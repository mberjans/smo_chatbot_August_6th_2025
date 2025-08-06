"""
Real-time Budget Monitoring System for Clinical Metabolomics Oracle LightRAG Integration

This module provides real-time budget monitoring that integrates with the API metrics logging
system and cost tracking infrastructure to provide continuous oversight of budget utilization.

Classes:
    - BudgetMonitoringEvent: Data model for monitoring events
    - BudgetThresholdTrigger: Trigger configuration for threshold-based monitoring
    - RealTimeBudgetMonitor: Main real-time monitoring system
    - BudgetHealthMetrics: Comprehensive budget health assessment
    - CostProjectionEngine: Predictive cost analysis system

The real-time monitoring system supports:
    - Continuous budget monitoring with configurable intervals
    - Integration with API metrics logging for real-time cost tracking
    - Predictive cost analysis and budget forecasting
    - Automatic alert triggering based on usage patterns
    - Circuit breaker integration for cost-based operation limiting
    - Dashboard-ready metrics and health assessments
    - Thread-safe concurrent monitoring operations
"""

import asyncio
import threading
import time
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .budget_manager import BudgetManager, BudgetAlert, AlertLevel, BudgetThreshold
from .api_metrics_logger import APIUsageMetricsLogger, APIMetric, MetricType
from .cost_persistence import CostPersistence, ResearchCategory
from .alert_system import AlertNotificationSystem, AlertEscalationManager


class MonitoringEventType(Enum):
    """Types of budget monitoring events."""
    
    THRESHOLD_WARNING = "threshold_warning"
    THRESHOLD_CRITICAL = "threshold_critical"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    RATE_SPIKE_DETECTED = "rate_spike_detected"
    PROJECTION_ALERT = "projection_alert"
    COST_ANOMALY = "cost_anomaly"
    SYSTEM_HEALTH = "system_health"
    BUDGET_RESET = "budget_reset"


@dataclass
class BudgetMonitoringEvent:
    """Data model for budget monitoring events."""
    
    timestamp: float = field(default_factory=time.time)
    event_type: MonitoringEventType = MonitoringEventType.SYSTEM_HEALTH
    period_type: str = "daily"  # daily, monthly, hourly
    period_key: str = ""
    current_cost: float = 0.0
    budget_limit: Optional[float] = None
    percentage_used: float = 0.0
    
    # Rate and trend information
    cost_rate_per_hour: Optional[float] = None
    cost_trend_percentage: Optional[float] = None
    projected_cost: Optional[float] = None
    time_remaining_hours: Optional[float] = None
    
    # Alert information
    severity: str = "info"  # info, warning, critical, emergency
    message: str = ""
    suggested_action: Optional[str] = None
    
    # Context and metadata
    session_id: Optional[str] = None
    research_category: Optional[str] = None
    triggering_operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'event_type': self.event_type.value,
            'period_type': self.period_type,
            'period_key': self.period_key,
            'current_cost': self.current_cost,
            'budget_limit': self.budget_limit,
            'percentage_used': self.percentage_used,
            'cost_rate_per_hour': self.cost_rate_per_hour,
            'cost_trend_percentage': self.cost_trend_percentage,
            'projected_cost': self.projected_cost,
            'time_remaining_hours': self.time_remaining_hours,
            'severity': self.severity,
            'message': self.message,
            'suggested_action': self.suggested_action,
            'session_id': self.session_id,
            'research_category': self.research_category,
            'triggering_operation': self.triggering_operation,
            'metadata': self.metadata
        }


@dataclass
class BudgetThresholdTrigger:
    """Configuration for threshold-based monitoring triggers."""
    
    threshold_percentage: float
    event_type: MonitoringEventType
    severity: str = "warning"
    cooldown_minutes: float = 5.0
    action_required: bool = False
    auto_circuit_break: bool = False
    
    def __post_init__(self):
        """Validate trigger configuration."""
        if not 0 <= self.threshold_percentage <= 200:
            raise ValueError("Threshold percentage must be between 0 and 200")


class CostProjectionEngine:
    """
    Predictive cost analysis and forecasting system.
    
    Provides sophisticated cost projection capabilities based on usage patterns,
    trends, and statistical modeling for proactive budget management.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize cost projection engine."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Historical data for modeling
        self._cost_history: deque = deque(maxlen=1000)  # Recent cost data points
        self._hourly_aggregates: Dict[str, float] = {}  # Hour-key -> cost
        self._daily_aggregates: Dict[str, float] = {}   # Day-key -> cost
        
        # Projection models
        self._linear_trend_window = 24  # Hours for linear trend analysis
        self._moving_average_window = 12  # Hours for moving average
        self._seasonal_patterns = defaultdict(list)  # Hour-of-day -> [costs]
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def add_cost_datapoint(self, cost: float, timestamp: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add cost data point for analysis."""
        with self._lock:
            datapoint = {
                'timestamp': timestamp,
                'cost': cost,
                'metadata': metadata or {}
            }
            
            self._cost_history.append(datapoint)
            
            # Update aggregates
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            hour_key = dt.strftime('%Y-%m-%d-%H')
            day_key = dt.strftime('%Y-%m-%d')
            
            self._hourly_aggregates[hour_key] = self._hourly_aggregates.get(hour_key, 0) + cost
            self._daily_aggregates[day_key] = self._daily_aggregates.get(day_key, 0) + cost
            
            # Update seasonal patterns
            hour_of_day = dt.hour
            self._seasonal_patterns[hour_of_day].append(cost)
            if len(self._seasonal_patterns[hour_of_day]) > 100:
                self._seasonal_patterns[hour_of_day].pop(0)
    
    def project_daily_cost(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Project daily cost based on current trends."""
        with self._lock:
            if not self._cost_history:
                return {'projected_cost': 0.0, 'confidence': 0.0, 'method': 'no_data'}
            
            now = current_time or time.time()
            current_dt = datetime.fromtimestamp(now, tz=timezone.utc)
            day_start = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            day_start_timestamp = day_start.timestamp()
            
            # Get today's cost so far
            today_costs = [
                dp['cost'] for dp in self._cost_history
                if dp['timestamp'] >= day_start_timestamp
            ]
            
            current_daily_cost = sum(today_costs)
            hours_elapsed = (now - day_start_timestamp) / 3600
            
            if hours_elapsed < 0.5:  # Less than 30 minutes into the day
                return self._project_using_historical_average(current_daily_cost)
            
            # Calculate hourly rate for today
            hourly_rate = current_daily_cost / hours_elapsed if hours_elapsed > 0 else 0
            
            # Get recent trend
            trend_projection = self._project_using_trend_analysis(current_daily_cost, hours_elapsed)
            
            # Get seasonal projection
            seasonal_projection = self._project_using_seasonal_patterns(current_daily_cost, current_dt)
            
            # Combine projections with weights
            projections = [
                (hourly_rate * 24, 0.4, 'linear_extrapolation'),
                (trend_projection.get('projected_cost', hourly_rate * 24), 0.4, 'trend_analysis'),
                (seasonal_projection.get('projected_cost', hourly_rate * 24), 0.2, 'seasonal_patterns')
            ]
            
            # Weighted average
            total_weight = sum(weight for _, weight, _ in projections)
            weighted_projection = sum(proj * weight for proj, weight, _ in projections) / total_weight
            
            # Calculate confidence based on data availability and consistency
            confidence = self._calculate_projection_confidence(projections, hours_elapsed)
            
            return {
                'projected_cost': weighted_projection,
                'current_cost': current_daily_cost,
                'hours_elapsed': hours_elapsed,
                'hours_remaining': 24 - hours_elapsed,
                'hourly_rate': hourly_rate,
                'confidence': confidence,
                'method': 'weighted_ensemble',
                'projections': {
                    'linear': projections[0][0],
                    'trend': projections[1][0],
                    'seasonal': projections[2][0]
                }
            }
    
    def project_monthly_cost(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Project monthly cost based on daily patterns."""
        with self._lock:
            now = current_time or time.time()
            current_dt = datetime.fromtimestamp(now, tz=timezone.utc)
            month_start = current_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_start_timestamp = month_start.timestamp()
            
            # Get this month's cost so far
            month_costs = [
                dp['cost'] for dp in self._cost_history
                if dp['timestamp'] >= month_start_timestamp
            ]
            
            current_monthly_cost = sum(month_costs)
            days_elapsed = (now - month_start_timestamp) / (24 * 3600)
            
            # Get days in current month
            if current_dt.month == 12:
                next_month = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                next_month = current_dt.replace(month=current_dt.month + 1)
            
            days_in_month = (next_month - month_start).days
            
            if days_elapsed < 0.5:  # Less than half a day
                # Use historical monthly average if available
                return self._project_monthly_using_historical()
            
            # Simple linear projection
            daily_rate = current_monthly_cost / days_elapsed if days_elapsed > 0 else 0
            linear_projection = daily_rate * days_in_month
            
            # Use recent daily projections for more accuracy
            recent_daily_projection = self.project_daily_cost(now)
            daily_projection_rate = recent_daily_projection.get('projected_cost', daily_rate)
            trend_based_projection = daily_projection_rate * days_in_month
            
            # Weighted combination
            if days_elapsed < 3:  # Early in month, rely more on trend
                projected_cost = 0.3 * linear_projection + 0.7 * trend_based_projection
                confidence = 0.6
            elif days_elapsed < 10:  # Mid-early month
                projected_cost = 0.6 * linear_projection + 0.4 * trend_based_projection
                confidence = 0.75
            else:  # Later in month, linear projection more reliable
                projected_cost = 0.8 * linear_projection + 0.2 * trend_based_projection
                confidence = min(0.9, days_elapsed / days_in_month + 0.1)
            
            return {
                'projected_cost': projected_cost,
                'current_cost': current_monthly_cost,
                'days_elapsed': days_elapsed,
                'days_remaining': days_in_month - days_elapsed,
                'days_in_month': days_in_month,
                'daily_rate': daily_rate,
                'confidence': confidence,
                'method': 'weighted_daily_projection'
            }
    
    def _project_using_trend_analysis(self, current_cost: float, hours_elapsed: float) -> Dict[str, Any]:
        """Project using linear trend analysis."""
        if len(self._cost_history) < 5:
            return {'projected_cost': current_cost, 'confidence': 0.1}
        
        # Get recent hourly costs for trend analysis
        now = time.time()
        recent_window = now - (self._linear_trend_window * 3600)
        recent_costs = [
            dp for dp in self._cost_history
            if dp['timestamp'] >= recent_window
        ]
        
        if len(recent_costs) < 3:
            return {'projected_cost': current_cost, 'confidence': 0.2}
        
        # Calculate trend slope
        timestamps = [dp['timestamp'] for dp in recent_costs]
        costs = [dp['cost'] for dp in recent_costs]
        
        # Normalize timestamps to hours from start
        start_time = min(timestamps)
        hours = [(ts - start_time) / 3600 for ts in timestamps]
        
        # Simple linear regression
        n = len(hours)
        sum_x = sum(hours)
        sum_y = sum(costs)
        sum_xy = sum(h * c for h, c in zip(hours, costs))
        sum_x2 = sum(h * h for h in hours)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return {'projected_cost': current_cost, 'confidence': 0.1}
        
        # Calculate slope (cost per hour trend)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Project for remaining hours
        remaining_hours = 24 - hours_elapsed
        projected_additional_cost = slope * remaining_hours
        
        return {
            'projected_cost': current_cost + max(0, projected_additional_cost),
            'trend_slope': slope,
            'confidence': min(0.8, len(recent_costs) / 20)
        }
    
    def _project_using_seasonal_patterns(self, current_cost: float, current_dt: datetime) -> Dict[str, Any]:
        """Project using hour-of-day seasonal patterns."""
        if not self._seasonal_patterns:
            return {'projected_cost': current_cost, 'confidence': 0.1}
        
        current_hour = current_dt.hour
        remaining_cost = 0
        
        # Project remaining hours of the day
        for hour in range(current_hour + 1, 24):
            if hour in self._seasonal_patterns and self._seasonal_patterns[hour]:
                hour_avg = statistics.mean(self._seasonal_patterns[hour])
                remaining_cost += hour_avg
        
        confidence = len([h for h in self._seasonal_patterns if self._seasonal_patterns[h]]) / 24
        
        return {
            'projected_cost': current_cost + remaining_cost,
            'seasonal_remaining': remaining_cost,
            'confidence': min(0.7, confidence)
        }
    
    def _project_using_historical_average(self, current_cost: float) -> Dict[str, Any]:
        """Project using historical daily averages."""
        if not self._daily_aggregates:
            return {'projected_cost': current_cost, 'confidence': 0.1}
        
        daily_costs = list(self._daily_aggregates.values())
        if not daily_costs:
            return {'projected_cost': current_cost, 'confidence': 0.1}
        
        avg_daily_cost = statistics.mean(daily_costs)
        confidence = min(0.6, len(daily_costs) / 30)  # More confidence with more days
        
        return {
            'projected_cost': avg_daily_cost,
            'historical_average': avg_daily_cost,
            'confidence': confidence
        }
    
    def _project_monthly_using_historical(self) -> Dict[str, Any]:
        """Project monthly cost using historical patterns."""
        # This would use historical monthly data - simplified for now
        daily_costs = list(self._daily_aggregates.values()) if self._daily_aggregates else [0]
        avg_daily = statistics.mean(daily_costs) if daily_costs else 0
        
        return {
            'projected_cost': avg_daily * 30,  # Rough approximation
            'confidence': 0.3,
            'method': 'historical_daily_average'
        }
    
    def _calculate_projection_confidence(self, projections: List[Tuple], hours_elapsed: float) -> float:
        """Calculate confidence in projection based on consistency and data availability."""
        if len(projections) < 2:
            return 0.1
        
        # Check consistency between projections
        values = [proj for proj, _, _ in projections]
        if not values:
            return 0.1
        
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.1
        
        # Calculate coefficient of variation
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        cv = std_dev / mean_val if mean_val > 0 else 1
        
        # Base confidence on consistency and time elapsed
        consistency_score = max(0, 1 - cv)  # Lower CV = higher consistency
        time_score = min(1, hours_elapsed / 12)  # More confidence as day progresses
        data_score = min(1, len(self._cost_history) / 100)  # More confidence with more data
        
        return (consistency_score * 0.4 + time_score * 0.4 + data_score * 0.2)
    
    def detect_cost_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous cost patterns."""
        anomalies = []
        
        if len(self._cost_history) < 20:
            return anomalies
        
        # Get recent costs
        recent_costs = list(self._cost_history)[-20:]
        costs = [dp['cost'] for dp in recent_costs]
        
        if not costs:
            return anomalies
        
        mean_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
        
        # Detect outliers (cost > mean + 2*std)
        threshold = mean_cost + 2 * std_cost
        
        for dp in recent_costs[-5:]:  # Check last 5 data points
            if dp['cost'] > threshold and dp['cost'] > mean_cost * 1.5:
                anomalies.append({
                    'timestamp': dp['timestamp'],
                    'cost': dp['cost'],
                    'expected_range': (0, threshold),
                    'severity': 'high' if dp['cost'] > threshold * 1.5 else 'medium',
                    'description': f"Cost spike detected: {dp['cost']:.4f} (expected < {threshold:.4f})"
                })
        
        return anomalies


class BudgetHealthMetrics:
    """Comprehensive budget health assessment system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize budget health metrics."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Health scoring weights
        self.health_weights = {
            'budget_utilization': 0.3,
            'trend_stability': 0.25,
            'projection_accuracy': 0.2,
            'alert_frequency': 0.15,
            'cost_anomalies': 0.1
        }
    
    def calculate_health_score(self,
                             budget_status: Dict[str, Any],
                             cost_trends: Dict[str, Any],
                             recent_alerts: List[Dict[str, Any]],
                             anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive budget health score."""
        
        scores = {}
        
        # Budget utilization score (0-100)
        daily_util = budget_status.get('daily_budget', {}).get('percentage_used', 0)
        monthly_util = budget_status.get('monthly_budget', {}).get('percentage_used', 0)
        max_util = max(daily_util, monthly_util)
        
        if max_util <= 50:
            scores['budget_utilization'] = 100
        elif max_util <= 75:
            scores['budget_utilization'] = 100 - (max_util - 50) * 2
        elif max_util <= 90:
            scores['budget_utilization'] = 50 - (max_util - 75) * 2
        else:
            scores['budget_utilization'] = max(0, 20 - (max_util - 90))
        
        # Trend stability score
        trend_pct = abs(cost_trends.get('trend_percentage', 0))
        if trend_pct <= 10:
            scores['trend_stability'] = 100
        elif trend_pct <= 25:
            scores['trend_stability'] = 100 - (trend_pct - 10) * 3
        else:
            scores['trend_stability'] = max(0, 55 - (trend_pct - 25))
        
        # Projection accuracy (simplified)
        confidence = cost_trends.get('confidence', 0.5)
        scores['projection_accuracy'] = confidence * 100
        
        # Alert frequency score
        alert_count = len([a for a in recent_alerts if time.time() - a.get('timestamp', 0) < 3600])
        if alert_count == 0:
            scores['alert_frequency'] = 100
        elif alert_count <= 3:
            scores['alert_frequency'] = 100 - alert_count * 15
        else:
            scores['alert_frequency'] = max(0, 55 - (alert_count - 3) * 10)
        
        # Cost anomaly score
        high_severity_anomalies = len([a for a in anomalies if a.get('severity') == 'high'])
        if high_severity_anomalies == 0:
            scores['cost_anomalies'] = 100
        else:
            scores['cost_anomalies'] = max(0, 100 - high_severity_anomalies * 25)
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[component] * self.health_weights[component]
            for component in scores
        )
        
        # Determine health status
        if overall_score >= 80:
            health_status = "excellent"
        elif overall_score >= 65:
            health_status = "good"
        elif overall_score >= 45:
            health_status = "warning"
        elif overall_score >= 25:
            health_status = "critical"
        else:
            health_status = "emergency"
        
        return {
            'overall_score': round(overall_score, 1),
            'health_status': health_status,
            'component_scores': scores,
            'recommendations': self._generate_health_recommendations(scores, health_status),
            'timestamp': time.time()
        }
    
    def _generate_health_recommendations(self, scores: Dict[str, float], status: str) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if scores['budget_utilization'] < 50:
            recommendations.append("Budget utilization is high - consider increasing limits or reducing usage")
        
        if scores['trend_stability'] < 60:
            recommendations.append("Cost trends are volatile - investigate usage patterns")
        
        if scores['alert_frequency'] < 70:
            recommendations.append("Frequent alerts detected - review alert thresholds or investigate cost spikes")
        
        if scores['cost_anomalies'] < 80:
            recommendations.append("Cost anomalies detected - investigate unusual usage patterns")
        
        if status in ['critical', 'emergency']:
            recommendations.append("URGENT: Budget health is critical - immediate intervention required")
        elif status == 'warning':
            recommendations.append("Budget monitoring required - consider proactive measures")
        
        return recommendations


class RealTimeBudgetMonitor:
    """
    Main real-time budget monitoring system.
    
    Provides continuous monitoring of budget utilization with integration to
    API metrics, cost tracking, and alert systems for comprehensive oversight.
    """
    
    def __init__(self,
                 budget_manager: BudgetManager,
                 api_metrics_logger: APIUsageMetricsLogger,
                 cost_persistence: CostPersistence,
                 alert_system: Optional[AlertNotificationSystem] = None,
                 escalation_manager: Optional[AlertEscalationManager] = None,
                 monitoring_interval: float = 60.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize real-time budget monitor.
        
        Args:
            budget_manager: Budget management system
            api_metrics_logger: API metrics logging system
            cost_persistence: Cost persistence layer
            alert_system: Alert notification system
            escalation_manager: Alert escalation manager
            monitoring_interval: Monitoring interval in seconds
            logger: Logger instance for operations
        """
        self.budget_manager = budget_manager
        self.api_metrics_logger = api_metrics_logger
        self.cost_persistence = cost_persistence
        self.alert_system = alert_system
        self.escalation_manager = escalation_manager
        self.monitoring_interval = monitoring_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize sub-systems
        self.projection_engine = CostProjectionEngine(self.logger)
        self.health_metrics = BudgetHealthMetrics(self.logger)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Threshold triggers
        self._threshold_triggers = [
            BudgetThresholdTrigger(
                threshold_percentage=75.0,
                event_type=MonitoringEventType.THRESHOLD_WARNING,
                severity="warning",
                cooldown_minutes=15.0
            ),
            BudgetThresholdTrigger(
                threshold_percentage=90.0,
                event_type=MonitoringEventType.THRESHOLD_CRITICAL,
                severity="critical",
                cooldown_minutes=10.0
            ),
            BudgetThresholdTrigger(
                threshold_percentage=100.0,
                event_type=MonitoringEventType.THRESHOLD_EXCEEDED,
                severity="emergency",
                cooldown_minutes=5.0,
                auto_circuit_break=True
            )
        ]
        
        # Event history and statistics
        self._monitoring_events: deque = deque(maxlen=1000)
        self._trigger_cooldowns: Dict[str, float] = {}
        self._monitoring_stats = {
            'monitoring_cycles': 0,
            'events_generated': 0,
            'alerts_triggered': 0,
            'start_time': None,
            'last_cycle_time': None
        }
        
        # Circuit breaker callback
        self._circuit_breaker_callback: Optional[Callable] = None
        
        self.logger.info("Real-time budget monitor initialized")
    
    def start_monitoring(self, background: bool = True) -> None:
        """Start real-time budget monitoring."""
        with self._lock:
            if self._monitoring_active:
                self.logger.warning("Budget monitoring is already active")
                return
            
            self._monitoring_active = True
            self._stop_event.clear()
            self._monitoring_stats['start_time'] = time.time()
            
            if background:
                self._monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="BudgetMonitor",
                    daemon=True
                )
                self._monitoring_thread.start()
                self.logger.info("Real-time budget monitoring started in background")
            else:
                self.logger.info("Starting real-time budget monitoring in foreground")
                self._monitoring_loop()
    
    def stop_monitoring(self) -> None:
        """Stop real-time budget monitoring."""
        with self._lock:
            if not self._monitoring_active:
                self.logger.warning("Budget monitoring is not active")
                return
            
            self._monitoring_active = False
            self._stop_event.set()
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=10.0)
                if self._monitoring_thread.is_alive():
                    self.logger.warning("Monitoring thread did not stop gracefully")
            
            self.logger.info("Real-time budget monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info(f"Budget monitoring loop started (interval: {self.monitoring_interval}s)")
        
        while self._monitoring_active and not self._stop_event.is_set():
            try:
                cycle_start = time.time()
                
                # Perform monitoring cycle
                self._perform_monitoring_cycle()
                
                # Update statistics
                self._monitoring_stats['monitoring_cycles'] += 1
                self._monitoring_stats['last_cycle_time'] = cycle_start
                
                # Sleep until next cycle
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.monitoring_interval - cycle_time)
                
                if not self._stop_event.wait(sleep_time):
                    continue
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Continue monitoring despite errors
                if not self._stop_event.wait(10.0):  # Wait 10 seconds before retry
                    continue
                else:
                    break
        
        self.logger.info("Budget monitoring loop ended")
    
    def _perform_monitoring_cycle(self) -> None:
        """Perform one monitoring cycle."""
        now = time.time()
        
        try:
            # Get current budget status
            budget_status = self.budget_manager.get_budget_summary()
            
            # Get API metrics for cost projection
            performance_summary = self.api_metrics_logger.get_performance_summary()
            
            # Update projection engine with recent costs
            self._update_projection_data(performance_summary)
            
            # Generate cost projections
            daily_projection = self.projection_engine.project_daily_cost(now)
            monthly_projection = self.projection_engine.project_monthly_cost(now)
            
            # Detect anomalies
            anomalies = self.projection_engine.detect_cost_anomalies()
            
            # Check threshold triggers
            events = self._check_threshold_triggers(budget_status, daily_projection, monthly_projection)
            
            # Check for rate spikes and anomalies
            events.extend(self._check_cost_anomalies(anomalies))
            
            # Generate health assessment
            health_score = self.health_metrics.calculate_health_score(
                budget_status,
                daily_projection,
                [event.to_dict() for event in self._monitoring_events if now - event.timestamp < 3600],
                anomalies
            )
            
            # Process and record events
            for event in events:
                self._record_monitoring_event(event)
                self._process_monitoring_event(event)
            
            # Log periodic health status
            if self._monitoring_stats['monitoring_cycles'] % 10 == 0:  # Every 10 cycles
                self.logger.info(
                    f"Budget Health: {health_score['health_status']} "
                    f"(score: {health_score['overall_score']}) | "
                    f"Daily: {budget_status.get('daily_budget', {}).get('percentage_used', 0):.1f}% | "
                    f"Monthly: {budget_status.get('monthly_budget', {}).get('percentage_used', 0):.1f}%"
                )
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
    
    def _update_projection_data(self, performance_summary: Dict[str, Any]) -> None:
        """Update projection engine with recent cost data."""
        try:
            # Extract cost information from performance summary
            current_hour = performance_summary.get('current_hour', {})
            hour_cost = current_hour.get('total_cost', 0)
            
            if hour_cost > 0:
                self.projection_engine.add_cost_datapoint(
                    cost=hour_cost,
                    timestamp=time.time(),
                    metadata={
                        'source': 'api_metrics',
                        'total_calls': current_hour.get('total_calls', 0),
                        'total_tokens': current_hour.get('total_tokens', 0)
                    }
                )
        except Exception as e:
            self.logger.warning(f"Failed to update projection data: {e}")
    
    def _check_threshold_triggers(self,
                                budget_status: Dict[str, Any],
                                daily_projection: Dict[str, Any],
                                monthly_projection: Dict[str, Any]) -> List[BudgetMonitoringEvent]:
        """Check threshold triggers for current budget status."""
        events = []
        now = time.time()
        
        # Check daily budget thresholds
        daily_budget = budget_status.get('daily_budget')
        if daily_budget and daily_budget.get('budget_limit', 0) > 0:
            percentage_used = daily_budget.get('percentage_used', 0)
            events.extend(self._check_period_thresholds(
                'daily',
                datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                percentage_used,
                daily_budget.get('total_cost', 0),
                daily_budget.get('budget_limit', 0),
                daily_projection
            ))
        
        # Check monthly budget thresholds
        monthly_budget = budget_status.get('monthly_budget')
        if monthly_budget and monthly_budget.get('budget_limit', 0) > 0:
            percentage_used = monthly_budget.get('percentage_used', 0)
            events.extend(self._check_period_thresholds(
                'monthly',
                datetime.now(timezone.utc).strftime('%Y-%m'),
                percentage_used,
                monthly_budget.get('total_cost', 0),
                monthly_budget.get('budget_limit', 0),
                monthly_projection
            ))
        
        return events
    
    def _check_period_thresholds(self,
                               period_type: str,
                               period_key: str,
                               percentage_used: float,
                               current_cost: float,
                               budget_limit: float,
                               projection: Dict[str, Any]) -> List[BudgetMonitoringEvent]:
        """Check thresholds for a specific period."""
        events = []
        now = time.time()
        
        for trigger in self._threshold_triggers:
            if percentage_used >= trigger.threshold_percentage:
                # Check cooldown
                cooldown_key = f"{period_type}_{period_key}_{trigger.event_type.value}"
                last_trigger = self._trigger_cooldowns.get(cooldown_key, 0)
                
                if now - last_trigger >= (trigger.cooldown_minutes * 60):
                    # Create monitoring event
                    event = BudgetMonitoringEvent(
                        timestamp=now,
                        event_type=trigger.event_type,
                        period_type=period_type,
                        period_key=period_key,
                        current_cost=current_cost,
                        budget_limit=budget_limit,
                        percentage_used=percentage_used,
                        projected_cost=projection.get('projected_cost'),
                        severity=trigger.severity,
                        message=f"{period_type.title()} budget {trigger.event_type.value.replace('_', ' ')}: "
                               f"{percentage_used:.1f}% used (${current_cost:.2f} of ${budget_limit:.2f})",
                        metadata={
                            'trigger_threshold': trigger.threshold_percentage,
                            'projection_confidence': projection.get('confidence', 0),
                            'auto_circuit_break': trigger.auto_circuit_break
                        }
                    )
                    
                    events.append(event)
                    self._trigger_cooldowns[cooldown_key] = now
        
        return events
    
    def _check_cost_anomalies(self, anomalies: List[Dict[str, Any]]) -> List[BudgetMonitoringEvent]:
        """Check for cost anomalies and create monitoring events."""
        events = []
        
        for anomaly in anomalies:
            event = BudgetMonitoringEvent(
                timestamp=anomaly['timestamp'],
                event_type=MonitoringEventType.COST_ANOMALY,
                severity=anomaly['severity'],
                message=anomaly['description'],
                metadata={
                    'anomaly_cost': anomaly['cost'],
                    'expected_range': anomaly['expected_range'],
                    'detection_method': 'statistical_outlier'
                }
            )
            events.append(event)
        
        return events
    
    def _record_monitoring_event(self, event: BudgetMonitoringEvent) -> None:
        """Record monitoring event in history."""
        with self._lock:
            self._monitoring_events.append(event)
            self._monitoring_stats['events_generated'] += 1
    
    def _process_monitoring_event(self, event: BudgetMonitoringEvent) -> None:
        """Process monitoring event and trigger appropriate actions."""
        try:
            # Log event
            log_level = {
                'info': logging.INFO,
                'warning': logging.WARNING,
                'critical': logging.ERROR,
                'emergency': logging.CRITICAL
            }.get(event.severity, logging.INFO)
            
            self.logger.log(log_level, f"Budget Monitoring Event: {event.message}")
            
            # Convert to budget alert if needed
            if event.event_type in [MonitoringEventType.THRESHOLD_WARNING, 
                                   MonitoringEventType.THRESHOLD_CRITICAL,
                                   MonitoringEventType.THRESHOLD_EXCEEDED]:
                
                alert_level = {
                    MonitoringEventType.THRESHOLD_WARNING: AlertLevel.WARNING,
                    MonitoringEventType.THRESHOLD_CRITICAL: AlertLevel.CRITICAL,
                    MonitoringEventType.THRESHOLD_EXCEEDED: AlertLevel.EXCEEDED
                }.get(event.event_type, AlertLevel.WARNING)
                
                budget_alert = BudgetAlert(
                    timestamp=event.timestamp,
                    alert_level=alert_level,
                    period_type=event.period_type,
                    period_key=event.period_key,
                    current_cost=event.current_cost or 0,
                    budget_limit=event.budget_limit or 0,
                    percentage_used=event.percentage_used,
                    threshold_percentage=event.metadata.get('trigger_threshold', 0),
                    message=event.message,
                    metadata=event.metadata
                )
                
                # Send through alert system
                if self.alert_system:
                    self.alert_system.send_alert(budget_alert)
                    self._monitoring_stats['alerts_triggered'] += 1
                
                # Process through escalation manager
                if self.escalation_manager:
                    self.escalation_manager.process_alert(budget_alert)
            
            # Trigger circuit breaker if needed
            if (event.metadata.get('auto_circuit_break', False) and 
                self._circuit_breaker_callback):
                self.logger.warning("Triggering circuit breaker due to budget threshold exceeded")
                self._circuit_breaker_callback(event)
            
        except Exception as e:
            self.logger.error(f"Error processing monitoring event: {e}", exc_info=True)
    
    def set_circuit_breaker_callback(self, callback: Callable[[BudgetMonitoringEvent], None]) -> None:
        """Set callback for circuit breaker triggering."""
        self._circuit_breaker_callback = callback
        self.logger.info("Circuit breaker callback set")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics."""
        with self._lock:
            now = time.time()
            
            # Recent events
            recent_events = [
                event.to_dict() for event in self._monitoring_events
                if now - event.timestamp < 3600  # Last hour
            ]
            
            # Health metrics
            budget_status = self.budget_manager.get_budget_summary()
            daily_projection = self.projection_engine.project_daily_cost(now)
            monthly_projection = self.projection_engine.project_monthly_cost(now)
            anomalies = self.projection_engine.detect_cost_anomalies()
            
            health_score = self.health_metrics.calculate_health_score(
                budget_status, daily_projection, recent_events, anomalies
            )
            
            return {
                'monitoring_active': self._monitoring_active,
                'monitoring_interval': self.monitoring_interval,
                'statistics': self._monitoring_stats.copy(),
                'recent_events': recent_events,
                'health_score': health_score,
                'projections': {
                    'daily': daily_projection,
                    'monthly': monthly_projection
                },
                'anomalies': anomalies,
                'budget_status': budget_status,
                'uptime_seconds': now - self._monitoring_stats.get('start_time', now),
                'timestamp': now
            }
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for dashboard display."""
        status = self.get_monitoring_status()
        
        # Simplified dashboard metrics
        return {
            'system_health': {
                'status': status['health_score']['health_status'],
                'score': status['health_score']['overall_score'],
                'recommendations': status['health_score']['recommendations'][:3]  # Top 3
            },
            'budget_utilization': {
                'daily': status['budget_status'].get('daily_budget', {}),
                'monthly': status['budget_status'].get('monthly_budget', {})
            },
            'cost_projections': {
                'daily_projected': status['projections']['daily'].get('projected_cost', 0),
                'daily_confidence': status['projections']['daily'].get('confidence', 0),
                'monthly_projected': status['projections']['monthly'].get('projected_cost', 0),
                'monthly_confidence': status['projections']['monthly'].get('confidence', 0)
            },
            'alerts': {
                'recent_count': len(status['recent_events']),
                'critical_alerts': len([e for e in status['recent_events'] if e.get('severity') in ['critical', 'emergency']]),
                'anomalies_detected': len(status['anomalies'])
            },
            'monitoring': {
                'active': status['monitoring_active'],
                'uptime_hours': status['uptime_seconds'] / 3600,
                'cycles_completed': status['statistics']['monitoring_cycles']
            },
            'timestamp': status['timestamp']
        }
    
    def force_monitoring_cycle(self) -> Dict[str, Any]:
        """Force an immediate monitoring cycle and return results."""
        try:
            self.logger.info("Forcing immediate monitoring cycle")
            self._perform_monitoring_cycle()
            return {'success': True, 'timestamp': time.time()}
        except Exception as e:
            self.logger.error(f"Error in forced monitoring cycle: {e}")
            return {'success': False, 'error': str(e), 'timestamp': time.time()}
    
    def close(self) -> None:
        """Clean shutdown of real-time budget monitor."""
        try:
            self.stop_monitoring()
            self.logger.info("Real-time budget monitor shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during budget monitor shutdown: {e}")