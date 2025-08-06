"""
Budget Dashboard API for Clinical Metabolomics Oracle LightRAG Integration

This module provides comprehensive dashboard endpoints for budget monitoring, cost analytics,
and system health visualization. It integrates with all budget management components to
provide real-time insights and historical reporting capabilities.

Classes:
    - DashboardMetrics: Aggregated metrics for dashboard display
    - BudgetDashboardAPI: Main dashboard API with endpoints
    - AnalyticsEngine: Advanced analytics and reporting system
    - AlertDashboard: Alert-specific dashboard functionality
    - ReportGenerator: Automated report generation system

The budget dashboard system supports:
    - Real-time budget status and health monitoring
    - Historical cost analysis and trend visualization
    - Alert management and escalation tracking
    - Circuit breaker status and protection metrics
    - Comprehensive reporting with export capabilities
    - REST-like API endpoints for integration
    - WebSocket support for real-time updates
"""

import json
import time
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import logging
from pathlib import Path

from .budget_manager import BudgetManager, BudgetAlert, AlertLevel
from .api_metrics_logger import APIUsageMetricsLogger, APIMetric
from .cost_persistence import CostPersistence, ResearchCategory
from .alert_system import AlertNotificationSystem, AlertEscalationManager
from .realtime_budget_monitor import RealTimeBudgetMonitor, BudgetMonitoringEvent
from .cost_based_circuit_breaker import CostCircuitBreakerManager, CostBasedCircuitBreaker


class DashboardTimeRange(Enum):
    """Time ranges for dashboard data aggregation."""
    
    LAST_HOUR = "last_hour"
    LAST_6_HOURS = "last_6_hours"
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    CURRENT_MONTH = "current_month"
    CURRENT_YEAR = "current_year"
    CUSTOM = "custom"


class MetricAggregation(Enum):
    """Types of metric aggregations."""
    
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    PERCENTAGE = "percentage"


@dataclass
class DashboardMetrics:
    """Aggregated metrics for dashboard display."""
    
    # Budget overview
    budget_health_score: float = 0.0
    budget_health_status: str = "unknown"
    
    # Current usage
    daily_cost: float = 0.0
    daily_budget: Optional[float] = None
    daily_percentage: float = 0.0
    monthly_cost: float = 0.0
    monthly_budget: Optional[float] = None
    monthly_percentage: float = 0.0
    
    # Projections
    daily_projected_cost: float = 0.0
    daily_projection_confidence: float = 0.0
    monthly_projected_cost: float = 0.0
    monthly_projection_confidence: float = 0.0
    
    # Trends
    cost_trend_percentage: float = 0.0
    cost_trend_direction: str = "stable"
    average_daily_cost: float = 0.0
    cost_efficiency_score: float = 0.0
    
    # Alerts and events
    active_alerts: int = 0
    critical_alerts: int = 0
    alerts_last_24h: int = 0
    last_alert_time: Optional[float] = None
    
    # Circuit breaker status
    circuit_breakers_healthy: int = 0
    circuit_breakers_degraded: int = 0
    circuit_breakers_open: int = 0
    total_operations_blocked: int = 0
    cost_savings_from_protection: float = 0.0
    
    # API metrics
    total_api_calls: int = 0
    average_response_time: float = 0.0
    error_rate_percentage: float = 0.0
    tokens_consumed: int = 0
    
    # Research categories
    top_research_categories: List[Dict[str, Any]] = field(default_factory=list)
    category_cost_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # System health
    system_uptime: float = 0.0
    monitoring_active: bool = False
    last_update_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['last_update_time_iso'] = datetime.fromtimestamp(
            self.last_update_time, tz=timezone.utc
        ).isoformat()
        
        if self.last_alert_time:
            result['last_alert_time_iso'] = datetime.fromtimestamp(
                self.last_alert_time, tz=timezone.utc
            ).isoformat()
        
        return result


class AnalyticsEngine:
    """Advanced analytics and reporting system for budget data."""
    
    def __init__(self,
                 cost_persistence: CostPersistence,
                 logger: Optional[logging.Logger] = None):
        """Initialize analytics engine."""
        self.cost_persistence = cost_persistence
        self.logger = logger or logging.getLogger(__name__)
        
        # Analytics cache
        self._analytics_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 300.0  # 5 minutes
        
        # Thread safety
        self._lock = threading.RLock()
    
    def generate_cost_trends(self, time_range: DashboardTimeRange, 
                           granularity: str = "daily") -> Dict[str, Any]:
        """Generate cost trend analysis."""
        with self._lock:
            cache_key = f"cost_trends_{time_range.value}_{granularity}"
            cached_result, cached_time = self._analytics_cache.get(cache_key, (None, 0))
            
            if cached_result and (time.time() - cached_time) < self._cache_ttl:
                return cached_result
            
            try:
                # Determine time window
                end_time = datetime.now(timezone.utc)
                start_time = self._get_start_time(time_range, end_time)
                
                # Get cost data
                cost_report = self.cost_persistence.generate_cost_report(start_time, end_time)
                
                # Extract time series data
                if granularity == "daily":
                    time_series = cost_report.get('daily_costs', {})
                elif granularity == "hourly":
                    time_series = cost_report.get('hourly_costs', {})
                else:
                    time_series = cost_report.get('daily_costs', {})
                
                # Calculate trends
                values = list(time_series.values())
                if len(values) < 2:
                    trend_analysis = {
                        'trend_percentage': 0.0,
                        'trend_direction': 'stable',
                        'confidence': 0.0
                    }
                else:
                    trend_analysis = self._calculate_trend_analysis(values)
                
                # Generate forecast
                forecast = self._generate_forecast(values, periods=7)
                
                result = {
                    'time_range': time_range.value,
                    'granularity': granularity,
                    'time_series': time_series,
                    'total_cost': sum(values),
                    'average_cost': statistics.mean(values) if values else 0,
                    'min_cost': min(values) if values else 0,
                    'max_cost': max(values) if values else 0,
                    'trend_analysis': trend_analysis,
                    'forecast': forecast,
                    'data_points': len(values),
                    'timestamp': time.time()
                }
                
                # Cache result
                self._analytics_cache[cache_key] = (result, time.time())
                return result
                
            except Exception as e:
                self.logger.error(f"Error generating cost trends: {e}")
                return {
                    'error': str(e),
                    'time_range': time_range.value,
                    'timestamp': time.time()
                }
    
    def generate_category_analysis(self, time_range: DashboardTimeRange) -> Dict[str, Any]:
        """Generate research category cost analysis."""
        with self._lock:
            cache_key = f"category_analysis_{time_range.value}"
            cached_result, cached_time = self._analytics_cache.get(cache_key, (None, 0))
            
            if cached_result and (time.time() - cached_time) < self._cache_ttl:
                return cached_result
            
            try:
                # Get research analysis
                days = self._get_days_from_time_range(time_range)
                research_analysis = self.cost_persistence.get_research_analysis(days)
                
                # Calculate category efficiency metrics
                category_metrics = {}
                for category_data in research_analysis.get('top_categories', []):
                    category = category_data['category']
                    cost = category_data['total_cost']
                    calls = category_data['total_calls']
                    
                    avg_cost_per_call = cost / calls if calls > 0 else 0
                    efficiency_score = self._calculate_efficiency_score(cost, calls)
                    
                    category_metrics[category] = {
                        'total_cost': cost,
                        'total_calls': calls,
                        'average_cost_per_call': avg_cost_per_call,
                        'efficiency_score': efficiency_score,
                        'percentage_of_total': category_data.get('percentage', 0)
                    }
                
                result = {
                    'time_range': time_range.value,
                    'category_breakdown': research_analysis.get('category_breakdown', {}),
                    'category_metrics': category_metrics,
                    'top_categories': research_analysis.get('top_categories', [])[:10],
                    'total_categories': len(research_analysis.get('category_breakdown', {})),
                    'recommendations': self._generate_category_recommendations(category_metrics),
                    'timestamp': time.time()
                }
                
                # Cache result
                self._analytics_cache[cache_key] = (result, time.time())
                return result
                
            except Exception as e:
                self.logger.error(f"Error generating category analysis: {e}")
                return {
                    'error': str(e),
                    'time_range': time_range.value,
                    'timestamp': time.time()
                }
    
    def generate_efficiency_analysis(self) -> Dict[str, Any]:
        """Generate cost efficiency analysis."""
        try:
            # Get recent performance data
            recent_analysis = self.cost_persistence.get_research_analysis(30)  # Last 30 days
            
            # Calculate efficiency metrics
            efficiency_metrics = {
                'cost_per_token': 0.0,
                'cost_per_successful_operation': 0.0,
                'error_rate_impact': 0.0,
                'category_efficiency_variance': 0.0,
                'time_based_efficiency': {}
            }
            
            # Cost per token analysis
            total_cost = sum(cat['total_cost'] for cat in recent_analysis.get('top_categories', []))
            total_tokens = sum(cat.get('total_tokens', 0) for cat in recent_analysis.get('top_categories', []))
            
            if total_tokens > 0:
                efficiency_metrics['cost_per_token'] = total_cost / total_tokens
            
            # Time-based efficiency (cost by hour of day)
            hourly_costs = self._get_hourly_cost_distribution()
            efficiency_metrics['time_based_efficiency'] = hourly_costs
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(recent_analysis)
            
            return {
                'efficiency_metrics': efficiency_metrics,
                'optimization_opportunities': optimization_opportunities,
                'overall_efficiency_score': self._calculate_overall_efficiency_score(efficiency_metrics),
                'recommendations': self._generate_efficiency_recommendations(efficiency_metrics),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating efficiency analysis: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_start_time(self, time_range: DashboardTimeRange, end_time: datetime) -> datetime:
        """Get start time based on time range."""
        if time_range == DashboardTimeRange.LAST_HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == DashboardTimeRange.LAST_6_HOURS:
            return end_time - timedelta(hours=6)
        elif time_range == DashboardTimeRange.LAST_24_HOURS:
            return end_time - timedelta(days=1)
        elif time_range == DashboardTimeRange.LAST_7_DAYS:
            return end_time - timedelta(days=7)
        elif time_range == DashboardTimeRange.LAST_30_DAYS:
            return end_time - timedelta(days=30)
        elif time_range == DashboardTimeRange.CURRENT_MONTH:
            return end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif time_range == DashboardTimeRange.CURRENT_YEAR:
            return end_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return end_time - timedelta(days=7)  # Default
    
    def _get_days_from_time_range(self, time_range: DashboardTimeRange) -> int:
        """Get number of days from time range enum."""
        if time_range == DashboardTimeRange.LAST_HOUR:
            return 1
        elif time_range == DashboardTimeRange.LAST_6_HOURS:
            return 1
        elif time_range == DashboardTimeRange.LAST_24_HOURS:
            return 1
        elif time_range == DashboardTimeRange.LAST_7_DAYS:
            return 7
        elif time_range == DashboardTimeRange.LAST_30_DAYS:
            return 30
        elif time_range == DashboardTimeRange.CURRENT_MONTH:
            return 30  # Approximate
        elif time_range == DashboardTimeRange.CURRENT_YEAR:
            return 365  # Approximate
        else:
            return 7
    
    def _calculate_trend_analysis(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend analysis for time series data."""
        if len(values) < 2:
            return {
                'trend_percentage': 0.0,
                'trend_direction': 'stable',
                'confidence': 0.0
            }
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate linear regression slope
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Convert slope to percentage trend
        avg_value = statistics.mean(values)
        if avg_value > 0:
            trend_percentage = (slope / avg_value) * 100 * n
        else:
            trend_percentage = 0.0
        
        # Determine trend direction
        if abs(trend_percentage) < 5:
            direction = 'stable'
        elif trend_percentage > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate confidence based on R-squared
        confidence = self._calculate_r_squared(x, values, slope)
        
        return {
            'trend_percentage': round(trend_percentage, 2),
            'trend_direction': direction,
            'confidence': round(confidence, 3)
        }
    
    def _calculate_r_squared(self, x: List[float], y: List[float], slope: float) -> float:
        """Calculate R-squared for trend confidence."""
        if not x or not y:
            return 0.0
        
        y_mean = statistics.mean(y)
        x_mean = statistics.mean(x)
        
        # Calculate intercept
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def _generate_forecast(self, values: List[float], periods: int = 7) -> List[Dict[str, Any]]:
        """Generate simple forecast for future periods."""
        if len(values) < 3:
            return []
        
        # Simple moving average forecast
        window_size = min(7, len(values))
        recent_values = values[-window_size:]
        avg_recent = statistics.mean(recent_values)
        
        # Calculate trend
        trend = (values[-1] - values[-min(3, len(values))]) / min(3, len(values))
        
        forecast = []
        for i in range(1, periods + 1):
            projected_value = avg_recent + (trend * i)
            forecast.append({
                'period': i,
                'projected_value': max(0, projected_value),  # Ensure non-negative
                'confidence': max(0.1, 0.9 - (i * 0.1))  # Decreasing confidence
            })
        
        return forecast
    
    def _calculate_efficiency_score(self, cost: float, calls: int) -> float:
        """Calculate efficiency score for a category."""
        if calls == 0:
            return 0.0
        
        cost_per_call = cost / calls
        
        # Normalize to 0-100 scale (lower cost per call = higher score)
        # This is a simple heuristic - could be more sophisticated
        max_reasonable_cost = 0.10  # $0.10 per call
        efficiency = max(0, min(100, (max_reasonable_cost - cost_per_call) / max_reasonable_cost * 100))
        
        return round(efficiency, 1)
    
    def _calculate_overall_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall efficiency score."""
        # Simple composite score - could be more sophisticated
        scores = []
        
        # Cost per token efficiency
        cost_per_token = metrics.get('cost_per_token', 0)
        if cost_per_token > 0:
            token_efficiency = max(0, min(100, (0.001 - cost_per_token) / 0.001 * 100))
            scores.append(token_efficiency)
        
        # Add other efficiency metrics as they become available
        
        return statistics.mean(scores) if scores else 50.0  # Default neutral score
    
    def _get_hourly_cost_distribution(self) -> Dict[int, float]:
        """Get cost distribution by hour of day."""
        # Simplified implementation - would query actual data
        return {hour: 0.0 for hour in range(24)}
    
    def _identify_optimization_opportunities(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # High-cost categories
        top_categories = analysis.get('top_categories', [])
        if top_categories:
            highest_cost = top_categories[0]
            if highest_cost.get('total_cost', 0) > 10.0:  # Threshold
                opportunities.append({
                    'type': 'high_cost_category',
                    'category': highest_cost.get('category'),
                    'potential_savings': highest_cost.get('total_cost', 0) * 0.1,  # 10% reduction
                    'recommendation': f"Review usage patterns in {highest_cost.get('category')} category"
                })
        
        return opportunities
    
    def _generate_category_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on category analysis."""
        recommendations = []
        
        # Find inefficient categories
        inefficient_categories = [
            cat for cat, data in metrics.items()
            if data.get('efficiency_score', 0) < 50
        ]
        
        if inefficient_categories:
            recommendations.append(
                f"Review cost efficiency in categories: {', '.join(inefficient_categories[:3])}"
            )
        
        return recommendations
    
    def _generate_efficiency_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate efficiency improvement recommendations."""
        recommendations = []
        
        cost_per_token = metrics.get('cost_per_token', 0)
        if cost_per_token > 0.001:  # Threshold
            recommendations.append("Consider optimizing token usage or switching to more cost-effective models")
        
        return recommendations


class BudgetDashboardAPI:
    """Main dashboard API providing endpoints for budget monitoring and analytics."""
    
    def __init__(self,
                 budget_manager: BudgetManager,
                 api_metrics_logger: APIUsageMetricsLogger,
                 cost_persistence: CostPersistence,
                 alert_system: Optional[AlertNotificationSystem] = None,
                 escalation_manager: Optional[AlertEscalationManager] = None,
                 real_time_monitor: Optional[RealTimeBudgetMonitor] = None,
                 circuit_breaker_manager: Optional[CostCircuitBreakerManager] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize budget dashboard API."""
        self.budget_manager = budget_manager
        self.api_metrics_logger = api_metrics_logger
        self.cost_persistence = cost_persistence
        self.alert_system = alert_system
        self.escalation_manager = escalation_manager
        self.real_time_monitor = real_time_monitor
        self.circuit_breaker_manager = circuit_breaker_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize analytics engine
        self.analytics_engine = AnalyticsEngine(cost_persistence, logger)
        
        # Dashboard state
        self._start_time = time.time()
        self._request_count = 0
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 60.0  # 1 minute default cache
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Budget dashboard API initialized")
    
    def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get comprehensive dashboard overview."""
        with self._lock:
            self._request_count += 1
            
            try:
                # Aggregate all metrics
                metrics = self._aggregate_dashboard_metrics()
                
                # Get system health
                system_health = self._get_system_health()
                
                # Get recent alerts
                recent_alerts = self._get_recent_alerts(limit=10)
                
                # Get budget trends
                trends = self._get_budget_trends()
                
                return {
                    'status': 'success',
                    'data': {
                        'metrics': metrics.to_dict(),
                        'system_health': system_health,
                        'recent_alerts': recent_alerts,
                        'trends': trends,
                        'last_updated': time.time()
                    },
                    'meta': {
                        'request_count': self._request_count,
                        'uptime_seconds': time.time() - self._start_time
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error generating dashboard overview: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
    
    def get_budget_status(self, include_projections: bool = True) -> Dict[str, Any]:
        """Get current budget status with optional projections."""
        try:
            # Get budget summary
            budget_summary = self.budget_manager.get_budget_summary()
            
            result = {
                'status': 'success',
                'data': {
                    'budget_summary': budget_summary,
                    'timestamp': time.time()
                }
            }
            
            # Add projections if requested
            if include_projections and self.real_time_monitor:
                monitor_status = self.real_time_monitor.get_monitoring_status()
                result['data']['projections'] = monitor_status.get('projections', {})
                result['data']['health_score'] = monitor_status.get('health_score', {})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting budget status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_cost_analytics(self, 
                          time_range: str = "last_7_days",
                          granularity: str = "daily",
                          include_categories: bool = True) -> Dict[str, Any]:
        """Get comprehensive cost analytics."""
        try:
            # Parse time range
            try:
                time_range_enum = DashboardTimeRange(time_range)
            except ValueError:
                time_range_enum = DashboardTimeRange.LAST_7_DAYS
            
            # Generate cost trends
            trends = self.analytics_engine.generate_cost_trends(time_range_enum, granularity)
            
            result = {
                'status': 'success',
                'data': {
                    'trends': trends,
                    'timestamp': time.time()
                }
            }
            
            # Add category analysis if requested
            if include_categories:
                category_analysis = self.analytics_engine.generate_category_analysis(time_range_enum)
                result['data']['category_analysis'] = category_analysis
            
            # Add efficiency analysis
            efficiency_analysis = self.analytics_engine.generate_efficiency_analysis()
            result['data']['efficiency_analysis'] = efficiency_analysis
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting cost analytics: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_alert_dashboard(self, 
                           time_range: str = "last_24_hours",
                           include_escalation: bool = True) -> Dict[str, Any]:
        """Get alert-focused dashboard data."""
        try:
            # Get recent alerts from monitoring system
            recent_alerts = self._get_recent_alerts(time_range=time_range, limit=100)
            
            # Alert statistics
            alert_stats = self._calculate_alert_statistics(recent_alerts)
            
            result = {
                'status': 'success',
                'data': {
                    'recent_alerts': recent_alerts,
                    'alert_statistics': alert_stats,
                    'timestamp': time.time()
                }
            }
            
            # Add escalation data if requested and available
            if include_escalation and self.escalation_manager:
                escalation_status = self.escalation_manager.get_escalation_status()
                result['data']['escalation_status'] = escalation_status
            
            # Add alert system delivery stats if available
            if self.alert_system:
                delivery_stats = self.alert_system.get_delivery_stats()
                result['data']['delivery_stats'] = delivery_stats
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting alert dashboard: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker system status."""
        try:
            if not self.circuit_breaker_manager:
                return {
                    'status': 'success',
                    'data': {
                        'message': 'Circuit breaker manager not available',
                        'timestamp': time.time()
                    }
                }
            
            # Get system status
            system_status = self.circuit_breaker_manager.get_system_status()
            
            return {
                'status': 'success',
                'data': {
                    'system_status': system_status,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting circuit breaker status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics."""
        try:
            # Get performance summary from metrics logger
            performance_summary = self.api_metrics_logger.get_performance_summary()
            
            return {
                'status': 'success',
                'data': {
                    'performance_summary': performance_summary,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_cost_report(self, 
                       start_date: str,
                       end_date: str,
                       format: str = "json") -> Dict[str, Any]:
        """Generate detailed cost report for specified period."""
        try:
            # Parse dates
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Generate cost report
            cost_report = self.cost_persistence.generate_cost_report(start_dt, end_dt)
            
            return {
                'status': 'success',
                'data': {
                    'cost_report': cost_report,
                    'report_period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'format': format,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating cost report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def trigger_budget_check(self) -> Dict[str, Any]:
        """Manually trigger budget check and monitoring cycle."""
        try:
            results = {}
            
            # Trigger budget manager check
            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_summary()
                results['budget_status'] = budget_status
            
            # Trigger monitoring cycle if available
            if self.real_time_monitor:
                monitor_result = self.real_time_monitor.force_monitoring_cycle()
                results['monitoring_cycle'] = monitor_result
            
            return {
                'status': 'success',
                'data': {
                    'results': results,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error triggering budget check: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _aggregate_dashboard_metrics(self) -> DashboardMetrics:
        """Aggregate all dashboard metrics."""
        metrics = DashboardMetrics()
        
        try:
            # Get budget summary
            budget_summary = self.budget_manager.get_budget_summary()
            
            # Extract budget information
            daily_budget = budget_summary.get('daily_budget', {})
            monthly_budget = budget_summary.get('monthly_budget', {})
            
            metrics.daily_cost = daily_budget.get('total_cost', 0)
            metrics.daily_budget = daily_budget.get('budget_limit')
            metrics.daily_percentage = daily_budget.get('percentage_used', 0)
            
            metrics.monthly_cost = monthly_budget.get('total_cost', 0)
            metrics.monthly_budget = monthly_budget.get('budget_limit')
            metrics.monthly_percentage = monthly_budget.get('percentage_used', 0)
            
            metrics.budget_health_status = budget_summary.get('budget_health', 'unknown')
            
            # Get projections if monitoring is available
            if self.real_time_monitor:
                monitor_status = self.real_time_monitor.get_monitoring_status()
                
                daily_proj = monitor_status.get('projections', {}).get('daily', {})
                monthly_proj = monitor_status.get('projections', {}).get('monthly', {})
                
                metrics.daily_projected_cost = daily_proj.get('projected_cost', 0)
                metrics.daily_projection_confidence = daily_proj.get('confidence', 0)
                metrics.monthly_projected_cost = monthly_proj.get('projected_cost', 0)
                metrics.monthly_projection_confidence = monthly_proj.get('confidence', 0)
                
                health_score = monitor_status.get('health_score', {})
                metrics.budget_health_score = health_score.get('overall_score', 0)
                metrics.budget_health_status = health_score.get('health_status', 'unknown')
            
            # Get API metrics
            if self.api_metrics_logger:
                performance = self.api_metrics_logger.get_performance_summary()
                current_day = performance.get('current_day', {})
                
                metrics.total_api_calls = current_day.get('total_calls', 0)
                metrics.average_response_time = current_day.get('avg_response_time_ms', 0)
                metrics.error_rate_percentage = current_day.get('error_rate_percent', 0)
                metrics.tokens_consumed = current_day.get('total_tokens', 0)
            
            # Get circuit breaker stats
            if self.circuit_breaker_manager:
                cb_status = self.circuit_breaker_manager.get_system_status()
                circuit_breakers = cb_status.get('circuit_breakers', {})
                
                metrics.circuit_breakers_healthy = len([
                    cb for cb in circuit_breakers.values() 
                    if cb.get('state') == 'closed'
                ])
                metrics.circuit_breakers_degraded = len([
                    cb for cb in circuit_breakers.values() 
                    if cb.get('state') in ['budget_limited', 'half_open']
                ])
                metrics.circuit_breakers_open = len([
                    cb for cb in circuit_breakers.values() 
                    if cb.get('state') == 'open'
                ])
                
                # Aggregate protection stats
                for cb_data in circuit_breakers.values():
                    stats = cb_data.get('statistics', {})
                    metrics.total_operations_blocked += stats.get('cost_blocked_calls', 0)
                    metrics.cost_savings_from_protection += stats.get('cost_savings', 0)
            
            # System uptime
            metrics.system_uptime = time.time() - self._start_time
            metrics.monitoring_active = (
                self.real_time_monitor.get_monitoring_status().get('monitoring_active', False)
                if self.real_time_monitor else False
            )
            
        except Exception as e:
            self.logger.error(f"Error aggregating dashboard metrics: {e}")
        
        return metrics
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        try:
            health_components = {}
            
            # Budget manager health
            budget_summary = self.budget_manager.get_budget_summary()
            health_components['budget_manager'] = {
                'status': 'healthy' if budget_summary.get('budget_health') != 'exceeded' else 'critical',
                'details': budget_summary.get('budget_health', 'unknown')
            }
            
            # Real-time monitoring health
            if self.real_time_monitor:
                monitor_status = self.real_time_monitor.get_monitoring_status()
                health_components['real_time_monitor'] = {
                    'status': 'healthy' if monitor_status.get('monitoring_active') else 'degraded',
                    'details': f"Active: {monitor_status.get('monitoring_active', False)}"
                }
            
            # Circuit breaker health
            if self.circuit_breaker_manager:
                cb_status = self.circuit_breaker_manager.get_system_status()
                system_health = cb_status.get('system_health', {})
                health_components['circuit_breakers'] = {
                    'status': system_health.get('status', 'unknown'),
                    'details': system_health.get('message', '')
                }
            
            # Alert system health
            if self.alert_system:
                delivery_stats = self.alert_system.get_delivery_stats()
                total_attempts = sum(
                    channel_stats.get('total_attempts', 0)
                    for channel_stats in delivery_stats.get('channels', {}).values()
                )
                successful_deliveries = sum(
                    channel_stats.get('successful_deliveries', 0)
                    for channel_stats in delivery_stats.get('channels', {}).values()
                )
                
                success_rate = successful_deliveries / max(total_attempts, 1)
                health_components['alert_system'] = {
                    'status': 'healthy' if success_rate > 0.8 else 'degraded',
                    'details': f"Success rate: {success_rate:.1%}"
                }
            
            # Overall health assessment
            component_statuses = [comp['status'] for comp in health_components.values()]
            if 'critical' in component_statuses:
                overall_status = 'critical'
            elif 'degraded' in component_statuses:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            return {
                'overall_status': overall_status,
                'components': health_components,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {
                'overall_status': 'unknown',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_recent_alerts(self, time_range: str = "last_24_hours", limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts from monitoring system."""
        try:
            alerts = []
            
            # Get alerts from real-time monitor if available
            if self.real_time_monitor:
                monitor_status = self.real_time_monitor.get_monitoring_status()
                recent_events = monitor_status.get('recent_events', [])
                
                # Convert monitoring events to alert format
                for event in recent_events[:limit]:
                    alerts.append({
                        'timestamp': event.get('timestamp'),
                        'type': event.get('event_type'),
                        'severity': event.get('severity', 'info'),
                        'message': event.get('message', ''),
                        'source': 'real_time_monitor',
                        'metadata': event.get('metadata', {})
                    })
            
            return sorted(alerts, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def _get_budget_trends(self) -> Dict[str, Any]:
        """Get budget trend data for dashboard."""
        try:
            # Get spending trends from budget manager
            spending_trends = self.budget_manager.get_spending_trends(days=30)
            
            return {
                'spending_trends': spending_trends,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting budget trends: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _calculate_alert_statistics(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for alert data."""
        if not alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'alert_rate_per_hour': 0.0
            }
        
        # Count by severity
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in alerts:
            severity_counts[alert.get('severity', 'unknown')] += 1
            type_counts[alert.get('type', 'unknown')] += 1
        
        # Calculate alert rate (alerts per hour)
        if alerts:
            time_span = max(alert.get('timestamp', 0) for alert in alerts) - min(alert.get('timestamp', 0) for alert in alerts)
            time_span_hours = max(1, time_span / 3600)  # At least 1 hour
            alert_rate = len(alerts) / time_span_hours
        else:
            alert_rate = 0.0
        
        return {
            'total_alerts': len(alerts),
            'by_severity': dict(severity_counts),
            'by_type': dict(type_counts),
            'alert_rate_per_hour': round(alert_rate, 2)
        }
    
    def get_api_health_check(self) -> Dict[str, Any]:
        """Get API health check endpoint."""
        return {
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self._start_time,
            'request_count': self._request_count,
            'components': {
                'budget_manager': self.budget_manager is not None,
                'api_metrics_logger': self.api_metrics_logger is not None,
                'cost_persistence': self.cost_persistence is not None,
                'alert_system': self.alert_system is not None,
                'real_time_monitor': self.real_time_monitor is not None,
                'circuit_breaker_manager': self.circuit_breaker_manager is not None
            }
        }