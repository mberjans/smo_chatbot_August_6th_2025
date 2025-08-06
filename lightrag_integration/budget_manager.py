"""
Budget Management System for Clinical Metabolomics Oracle LightRAG Integration

This module provides comprehensive budget management with progressive alerting,
daily and monthly tracking, and integration with the cost persistence layer.

Classes:
    - AlertLevel: Enum for different alert severity levels
    - BudgetAlert: Data model for budget alert notifications
    - BudgetThreshold: Configuration for budget alert thresholds
    - BudgetManager: Main budget management and alerting system

The budget manager supports:
    - Real-time budget monitoring with configurable thresholds
    - Progressive alerts (warning, critical, exceeded)
    - Daily and monthly budget tracking
    - Integration with cost persistence layer
    - Thread-safe operations for concurrent access
"""

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .cost_persistence import CostPersistence, ResearchCategory


class AlertLevel(Enum):
    """Alert severity levels for budget management."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class BudgetAlert:
    """
    Data model for budget alert notifications.
    
    This dataclass represents a budget alert with all relevant information
    for notification and logging purposes.
    """
    
    timestamp: float
    alert_level: AlertLevel
    period_type: str  # 'daily' or 'monthly'
    period_key: str   # '2025-08-06' or '2025-08'
    current_cost: float
    budget_limit: float
    percentage_used: float
    threshold_percentage: float
    message: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'alert_level': self.alert_level.value,
            'period_type': self.period_type,
            'period_key': self.period_key,
            'current_cost': self.current_cost,
            'budget_limit': self.budget_limit,
            'percentage_used': self.percentage_used,
            'threshold_percentage': self.threshold_percentage,
            'message': self.message,
            'metadata': self.metadata or {}
        }


@dataclass
class BudgetThreshold:
    """Configuration for budget alert thresholds."""
    
    warning_percentage: float = 75.0    # Warning at 75% of budget
    critical_percentage: float = 90.0   # Critical alert at 90% of budget
    exceeded_percentage: float = 100.0  # Alert when budget is exceeded
    
    def __post_init__(self):
        """Validate threshold percentages."""
        thresholds = [self.warning_percentage, self.critical_percentage, self.exceeded_percentage]
        if not all(0 <= t <= 200 for t in thresholds):
            raise ValueError("All threshold percentages must be between 0 and 200")
        
        if not (self.warning_percentage <= self.critical_percentage <= self.exceeded_percentage):
            raise ValueError("Thresholds must be in ascending order: warning <= critical <= exceeded")


class BudgetManager:
    """
    Comprehensive budget management system with progressive alerting.
    
    This class provides real-time budget monitoring, progressive alerts,
    and integration with the cost persistence layer for historical tracking.
    """
    
    def __init__(self,
                 cost_persistence: CostPersistence,
                 daily_budget_limit: Optional[float] = None,
                 monthly_budget_limit: Optional[float] = None,
                 thresholds: Optional[BudgetThreshold] = None,
                 alert_callback: Optional[Callable[[BudgetAlert], None]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize budget manager.
        
        Args:
            cost_persistence: Cost persistence layer for data storage
            daily_budget_limit: Daily budget limit in USD
            monthly_budget_limit: Monthly budget limit in USD
            thresholds: Budget threshold configuration
            alert_callback: Optional callback function for alerts
            logger: Logger instance for operations
        """
        self.cost_persistence = cost_persistence
        self.daily_budget_limit = daily_budget_limit
        self.monthly_budget_limit = monthly_budget_limit
        self.thresholds = thresholds or BudgetThreshold()
        self.alert_callback = alert_callback
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Alert tracking to prevent spam
        self._last_alerts: Dict[str, float] = {}  # key -> timestamp
        self._alert_cooldown = 300.0  # 5 minutes between similar alerts
        
        # Performance optimization - cache recent budget status
        self._budget_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # key -> (status, timestamp)
        self._cache_ttl = 60.0  # Cache for 1 minute
        
        self.logger.info("Budget manager initialized")
        if daily_budget_limit:
            self.logger.info(f"Daily budget limit: ${daily_budget_limit:.2f}")
        if monthly_budget_limit:
            self.logger.info(f"Monthly budget limit: ${monthly_budget_limit:.2f}")
    
    def check_budget_status(self,
                          cost_amount: float,
                          operation_type: str,
                          research_category: Optional[ResearchCategory] = None) -> Dict[str, Any]:
        """
        Check current budget status and trigger alerts if necessary.
        
        Args:
            cost_amount: Cost of the current operation
            operation_type: Type of operation being performed
            research_category: Research category for the operation
            
        Returns:
            Dict containing budget status and any alerts generated
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            alerts_generated = []
            
            # Check daily budget
            daily_status = None
            if self.daily_budget_limit is not None:
                daily_status = self._check_period_budget(
                    'daily',
                    now,
                    self.daily_budget_limit,
                    cost_amount,
                    operation_type,
                    research_category
                )
                if daily_status.get('alerts'):
                    alerts_generated.extend(daily_status['alerts'])
            
            # Check monthly budget
            monthly_status = None
            if self.monthly_budget_limit is not None:
                monthly_status = self._check_period_budget(
                    'monthly',
                    now,
                    self.monthly_budget_limit,
                    cost_amount,
                    operation_type,
                    research_category
                )
                if monthly_status.get('alerts'):
                    alerts_generated.extend(monthly_status['alerts'])
            
            # Determine overall budget health
            budget_health = self._assess_budget_health(daily_status, monthly_status)
            
            return {
                'timestamp': time.time(),
                'daily_status': daily_status,
                'monthly_status': monthly_status,
                'budget_health': budget_health,
                'alerts_generated': alerts_generated,
                'operation_allowed': budget_health != 'exceeded'
            }
    
    def _check_period_budget(self,
                           period_type: str,
                           date: datetime,
                           budget_limit: float,
                           current_cost: float,
                           operation_type: str,
                           research_category: Optional[ResearchCategory]) -> Dict[str, Any]:
        """Check budget for a specific period (daily or monthly)."""
        # Get cached status if available and fresh
        cache_key = f"{period_type}_{date.strftime('%Y-%m-%d' if period_type == 'daily' else '%Y-%m')}"
        cached_status, cached_time = self._budget_cache.get(cache_key, (None, 0))
        
        if cached_status and (time.time() - cached_time) < self._cache_ttl:
            # Use cached status but add current cost
            status = cached_status.copy()
            status['projected_cost'] = status['current_cost'] + current_cost
            status['projected_percentage'] = (status['projected_cost'] / budget_limit) * 100 if budget_limit > 0 else 0
        else:
            # Get fresh status from persistence layer
            if period_type == 'daily':
                status = self.cost_persistence.get_daily_budget_status(date, budget_limit)
            else:
                status = self.cost_persistence.get_monthly_budget_status(date, budget_limit)
            
            # Add projected costs
            status['projected_cost'] = status['total_cost'] + current_cost
            status['projected_percentage'] = (status['projected_cost'] / budget_limit) * 100 if budget_limit > 0 else 0
            
            # Cache the status (without projected values)
            cached_status = {k: v for k, v in status.items() if not k.startswith('projected')}
            self._budget_cache[cache_key] = (cached_status, time.time())
        
        # Check for alerts based on projected costs
        alerts = self._check_threshold_alerts(
            period_type,
            status.get('date' if period_type == 'daily' else 'month'),
            status['projected_cost'],
            budget_limit,
            status['projected_percentage'],
            operation_type,
            research_category
        )
        
        status['alerts'] = alerts
        return status
    
    def _check_threshold_alerts(self,
                              period_type: str,
                              period_key: str,
                              current_cost: float,
                              budget_limit: float,
                              percentage_used: float,
                              operation_type: str,
                              research_category: Optional[ResearchCategory]) -> List[BudgetAlert]:
        """Check if any budget thresholds have been crossed and generate alerts."""
        alerts = []
        
        # Determine which thresholds have been crossed
        thresholds_crossed = []
        
        if percentage_used >= self.thresholds.exceeded_percentage:
            thresholds_crossed.append((AlertLevel.EXCEEDED, self.thresholds.exceeded_percentage))
        elif percentage_used >= self.thresholds.critical_percentage:
            thresholds_crossed.append((AlertLevel.CRITICAL, self.thresholds.critical_percentage))
        elif percentage_used >= self.thresholds.warning_percentage:
            thresholds_crossed.append((AlertLevel.WARNING, self.thresholds.warning_percentage))
        
        # Generate alerts for crossed thresholds (only if not in cooldown)
        for alert_level, threshold_percentage in thresholds_crossed:
            alert_key = f"{period_type}_{period_key}_{alert_level.value}"
            last_alert_time = self._last_alerts.get(alert_key, 0)
            
            if (time.time() - last_alert_time) >= self._alert_cooldown:
                alert = self._create_budget_alert(
                    alert_level,
                    period_type,
                    period_key,
                    current_cost,
                    budget_limit,
                    percentage_used,
                    threshold_percentage,
                    operation_type,
                    research_category
                )
                
                alerts.append(alert)
                self._last_alerts[alert_key] = time.time()
                
                # Trigger callback if provided
                if self.alert_callback:
                    try:
                        self.alert_callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")
        
        return alerts
    
    def _create_budget_alert(self,
                           alert_level: AlertLevel,
                           period_type: str,
                           period_key: str,
                           current_cost: float,
                           budget_limit: float,
                           percentage_used: float,
                           threshold_percentage: float,
                           operation_type: str,
                           research_category: Optional[ResearchCategory]) -> BudgetAlert:
        """Create a budget alert with appropriate messaging."""
        
        # Generate alert message based on level
        if alert_level == AlertLevel.EXCEEDED:
            message = f"{period_type.title()} budget exceeded! Current: ${current_cost:.2f}, Limit: ${budget_limit:.2f} ({percentage_used:.1f}%)"
        elif alert_level == AlertLevel.CRITICAL:
            message = f"{period_type.title()} budget critical! Current: ${current_cost:.2f}, Limit: ${budget_limit:.2f} ({percentage_used:.1f}%)"
        elif alert_level == AlertLevel.WARNING:
            message = f"{period_type.title()} budget warning! Current: ${current_cost:.2f}, Limit: ${budget_limit:.2f} ({percentage_used:.1f}%)"
        else:
            message = f"{period_type.title()} budget status: ${current_cost:.2f} of ${budget_limit:.2f} ({percentage_used:.1f}%)"
        
        # Create metadata
        metadata = {
            'operation_type': operation_type,
            'research_category': research_category.value if research_category else None,
            'remaining_budget': budget_limit - current_cost,
            'period_type': period_type,
            'threshold_crossed': threshold_percentage
        }
        
        alert = BudgetAlert(
            timestamp=time.time(),
            alert_level=alert_level,
            period_type=period_type,
            period_key=period_key,
            current_cost=current_cost,
            budget_limit=budget_limit,
            percentage_used=percentage_used,
            threshold_percentage=threshold_percentage,
            message=message,
            metadata=metadata
        )
        
        # Log the alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
            AlertLevel.EXCEEDED: logging.CRITICAL
        }.get(alert_level, logging.INFO)
        
        self.logger.log(log_level, f"Budget Alert: {message}")
        
        return alert
    
    def _assess_budget_health(self,
                            daily_status: Optional[Dict[str, Any]],
                            monthly_status: Optional[Dict[str, Any]]) -> str:
        """Assess overall budget health based on daily and monthly status."""
        
        # Check if any budget is exceeded
        if daily_status and daily_status.get('over_budget'):
            return 'exceeded'
        if monthly_status and monthly_status.get('over_budget'):
            return 'exceeded'
        
        # Check for critical status (>= 90%)
        daily_critical = daily_status and daily_status.get('percentage_used', 0) >= self.thresholds.critical_percentage
        monthly_critical = monthly_status and monthly_status.get('percentage_used', 0) >= self.thresholds.critical_percentage
        
        if daily_critical or monthly_critical:
            return 'critical'
        
        # Check for warning status (>= 75%)
        daily_warning = daily_status and daily_status.get('percentage_used', 0) >= self.thresholds.warning_percentage
        monthly_warning = monthly_status and monthly_status.get('percentage_used', 0) >= self.thresholds.warning_percentage
        
        if daily_warning or monthly_warning:
            return 'warning'
        
        return 'healthy'
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive budget summary.
        
        Returns:
            Dict containing current budget status for all configured limits
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            summary = {
                'timestamp': time.time(),
                'budget_health': 'healthy',
                'active_alerts': [],
                'daily_budget': None,
                'monthly_budget': None
            }
            
            # Get daily budget status
            if self.daily_budget_limit is not None:
                daily_status = self.cost_persistence.get_daily_budget_status(now, self.daily_budget_limit)
                summary['daily_budget'] = daily_status
                
                # Check for active alerts
                if daily_status.get('percentage_used', 0) >= self.thresholds.warning_percentage:
                    if daily_status.get('over_budget'):
                        summary['budget_health'] = 'exceeded'
                    elif daily_status.get('percentage_used', 0) >= self.thresholds.critical_percentage:
                        summary['budget_health'] = 'critical'
                    elif summary['budget_health'] == 'healthy':
                        summary['budget_health'] = 'warning'
            
            # Get monthly budget status
            if self.monthly_budget_limit is not None:
                monthly_status = self.cost_persistence.get_monthly_budget_status(now, self.monthly_budget_limit)
                summary['monthly_budget'] = monthly_status
                
                # Check for active alerts
                if monthly_status.get('percentage_used', 0) >= self.thresholds.warning_percentage:
                    if monthly_status.get('over_budget'):
                        summary['budget_health'] = 'exceeded'
                    elif monthly_status.get('percentage_used', 0) >= self.thresholds.critical_percentage and summary['budget_health'] != 'exceeded':
                        summary['budget_health'] = 'critical'
                    elif summary['budget_health'] == 'healthy':
                        summary['budget_health'] = 'warning'
            
            return summary
    
    def update_budget_limits(self,
                           daily_budget: Optional[float] = None,
                           monthly_budget: Optional[float] = None) -> None:
        """
        Update budget limits.
        
        Args:
            daily_budget: New daily budget limit
            monthly_budget: New monthly budget limit
        """
        with self._lock:
            if daily_budget is not None:
                self.daily_budget_limit = daily_budget
                self.logger.info(f"Daily budget limit updated to ${daily_budget:.2f}")
            
            if monthly_budget is not None:
                self.monthly_budget_limit = monthly_budget
                self.logger.info(f"Monthly budget limit updated to ${monthly_budget:.2f}")
            
            # Clear cache to force fresh budget calculations
            self._budget_cache.clear()
    
    def update_thresholds(self, thresholds: BudgetThreshold) -> None:
        """
        Update budget alert thresholds.
        
        Args:
            thresholds: New threshold configuration
        """
        with self._lock:
            self.thresholds = thresholds
            self.logger.info(f"Budget thresholds updated: warning={thresholds.warning_percentage}%, "
                           f"critical={thresholds.critical_percentage}%, exceeded={thresholds.exceeded_percentage}%")
            
            # Clear alert history to allow immediate alerts with new thresholds
            self._last_alerts.clear()
    
    def set_alert_callback(self, callback: Optional[Callable[[BudgetAlert], None]]) -> None:
        """
        Set or update the alert callback function.
        
        Args:
            callback: Function to call when alerts are generated
        """
        with self._lock:
            self.alert_callback = callback
            self.logger.info("Budget alert callback updated")
    
    def reset_alert_cooldowns(self) -> None:
        """Reset all alert cooldowns to allow immediate alerts."""
        with self._lock:
            self._last_alerts.clear()
            self.logger.info("Alert cooldowns reset")
    
    def clear_cache(self) -> None:
        """Clear budget status cache to force fresh calculations."""
        with self._lock:
            self._budget_cache.clear()
            self.logger.debug("Budget cache cleared")
    
    def get_spending_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze spending trends over the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict containing spending trend analysis
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get research analysis from cost persistence
        research_analysis = self.cost_persistence.get_research_analysis(days)
        
        # Generate cost report for the period
        cost_report = self.cost_persistence.generate_cost_report(start_date, end_date)
        
        # Calculate daily averages and trends
        daily_costs = cost_report.get('daily_costs', {})
        if daily_costs:
            daily_values = list(daily_costs.values())
            avg_daily_cost = sum(daily_values) / len(daily_values)
            
            # Simple trend calculation (last 7 days vs previous 7 days)
            if len(daily_values) >= 14:
                recent_avg = sum(daily_values[-7:]) / 7
                previous_avg = sum(daily_values[-14:-7]) / 7
                trend_percentage = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            else:
                trend_percentage = 0
        else:
            avg_daily_cost = 0
            trend_percentage = 0
        
        # Project monthly costs based on current trends
        current_month_days = datetime.now(timezone.utc).day
        days_in_month = 30  # Simplified assumption
        projected_monthly_cost = (cost_report['summary']['total_cost'] / current_month_days) * days_in_month if current_month_days > 0 else 0
        
        return {
            'period_days': days,
            'total_cost': cost_report['summary']['total_cost'],
            'average_daily_cost': avg_daily_cost,
            'trend_percentage': trend_percentage,
            'trend_direction': 'increasing' if trend_percentage > 5 else 'decreasing' if trend_percentage < -5 else 'stable',
            'projected_monthly_cost': projected_monthly_cost,
            'top_research_categories': research_analysis.get('top_categories', [])[:3],
            'daily_breakdown': daily_costs,
            'budget_projections': {
                'daily_limit_needed': avg_daily_cost * 1.2,  # 20% buffer
                'monthly_limit_needed': projected_monthly_cost * 1.2  # 20% buffer
            }
        }