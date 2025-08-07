#!/usr/bin/env python3
"""
Cleanup Validation and Monitoring System for Clinical Metabolomics Oracle LightRAG Integration.

This module provides comprehensive validation, monitoring, and reporting capabilities
for the advanced cleanup system. It ensures cleanup effectiveness and provides
detailed insights into resource usage patterns and cleanup performance.

Key Features:
1. Cleanup effectiveness validation
2. Resource usage monitoring and trending
3. Performance analysis and optimization
4. Automated reporting and alerting
5. Integration with existing logging systems
6. Dashboard-style monitoring capabilities
7. Cleanup failure analysis and recovery

Components:
- CleanupValidator: Validates cleanup effectiveness
- ResourceMonitor: Monitors resource usage patterns
- PerformanceAnalyzer: Analyzes cleanup performance
- CleanupReporter: Generates comprehensive reports
- AlertSystem: Automated alerting for cleanup issues
- DashboardMonitor: Real-time monitoring capabilities

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import logging
import sqlite3
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union, Callable, Tuple
import psutil
import weakref
from contextlib import contextmanager
import tempfile
import shutil
import os

# Import advanced cleanup system
try:
    from .advanced_cleanup_system import (
        AdvancedCleanupOrchestrator, ResourceType, CleanupStrategy, 
        CleanupScope, CleanupPolicy, ResourceThresholds
    )
except ImportError:
    # Handle import for standalone usage
    pass


# =====================================================================
# DATA STRUCTURES AND TYPES
# =====================================================================

@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: datetime
    resource_type: ResourceType
    usage_metrics: Dict[str, Any]
    threshold_violations: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class CleanupValidationResult:
    """Result of cleanup validation."""
    success: bool
    resource_type: ResourceType
    validation_timestamp: datetime
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for cleanup operations."""
    operation_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    resources_cleaned: Set[ResourceType]
    success_rate: float
    resource_metrics: Dict[ResourceType, Dict[str, Any]]
    memory_before_mb: float
    memory_after_mb: float
    memory_freed_mb: float


@dataclass
class AlertConfig:
    """Configuration for cleanup alerts."""
    enabled: bool = True
    memory_threshold_mb: float = 2048
    file_handle_threshold: int = 1000
    cleanup_failure_threshold: int = 3
    performance_degradation_threshold: float = 2.0  # 2x slower than baseline
    notification_cooldown_minutes: int = 30


# =====================================================================
# CLEANUP VALIDATOR
# =====================================================================

class CleanupValidator:
    """Validates cleanup effectiveness and identifies issues."""
    
    def __init__(self, thresholds: ResourceThresholds = None):
        self.thresholds = thresholds or ResourceThresholds()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validation_history = deque(maxlen=500)
        self._baseline_metrics = {}
        
    def validate_cleanup(self, orchestrator: 'AdvancedCleanupOrchestrator', 
                        resource_types: Set[ResourceType] = None) -> Dict[ResourceType, CleanupValidationResult]:
        """Validate cleanup effectiveness for specified resource types."""
        resource_types = resource_types or set(ResourceType)
        results = {}
        
        for resource_type in resource_types:
            result = self._validate_resource_cleanup(orchestrator, resource_type)
            results[resource_type] = result
            self._validation_history.append(result)
            
        return results
        
    def _validate_resource_cleanup(self, orchestrator: 'AdvancedCleanupOrchestrator', 
                                  resource_type: ResourceType) -> CleanupValidationResult:
        """Validate cleanup for a specific resource type."""
        result = CleanupValidationResult(
            success=True,
            resource_type=resource_type,
            validation_timestamp=datetime.now(),
        )
        
        try:
            # Get current usage from the orchestrator
            usage = orchestrator.get_resource_usage().get(resource_type, {})
            result.metrics = usage.copy()
            
            # Validate based on resource type
            if resource_type == ResourceType.MEMORY:
                self._validate_memory_cleanup(usage, result)
            elif resource_type == ResourceType.FILE_HANDLES:
                self._validate_file_handle_cleanup(usage, result)
            elif resource_type == ResourceType.DATABASE_CONNECTIONS:
                self._validate_db_connection_cleanup(usage, result)
            elif resource_type == ResourceType.TEMPORARY_FILES:
                self._validate_temp_file_cleanup(usage, result)
            elif resource_type == ResourceType.PROCESSES:
                self._validate_process_cleanup(usage, result)
                
            # Check for common issues
            self._check_common_issues(usage, result)
            
        except Exception as e:
            result.success = False
            result.issues.append(f"Validation error: {str(e)}")
            self.logger.error(f"Cleanup validation failed for {resource_type}: {e}")
            
        return result
        
    def _validate_memory_cleanup(self, usage: Dict[str, Any], result: CleanupValidationResult):
        """Validate memory cleanup effectiveness."""
        memory_mb = usage.get('memory_mb', 0)
        memory_percent = usage.get('memory_percent', 0)
        
        if self.thresholds.memory_mb and memory_mb > self.thresholds.memory_mb:
            result.success = False
            result.issues.append(f"Memory usage {memory_mb:.1f}MB exceeds threshold {self.thresholds.memory_mb}MB")
            
        if memory_percent > 80:
            result.warnings.append(f"High memory usage: {memory_percent:.1f}%")
            
        # Check garbage collection effectiveness
        gc_stats = usage.get('gc_stats')
        if gc_stats and len(gc_stats) > 0:
            for i, gen_stats in enumerate(gc_stats):
                uncollectable = gen_stats.get('uncollectable', 0)
                if uncollectable > 0:
                    result.warnings.append(f"Generation {i} has {uncollectable} uncollectable objects")
                    
    def _validate_file_handle_cleanup(self, usage: Dict[str, Any], result: CleanupValidationResult):
        """Validate file handle cleanup effectiveness."""
        open_files = usage.get('open_files', 0)
        tracked_files = usage.get('tracked_files', 0)
        
        if self.thresholds.file_handles and open_files > self.thresholds.file_handles:
            result.success = False
            result.issues.append(f"Open files {open_files} exceeds threshold {self.thresholds.file_handles}")
            
        if tracked_files > 0:
            result.warnings.append(f"Still tracking {tracked_files} file objects after cleanup")
            
    def _validate_db_connection_cleanup(self, usage: Dict[str, Any], result: CleanupValidationResult):
        """Validate database connection cleanup effectiveness."""
        active_connections = usage.get('active_connections', 0)
        
        if self.thresholds.db_connections and active_connections > self.thresholds.db_connections:
            result.success = False
            result.issues.append(f"Active DB connections {active_connections} exceeds threshold {self.thresholds.db_connections}")
            
        if active_connections > 0:
            result.recommendations.append("Consider implementing connection pooling for better resource management")
            
    def _validate_temp_file_cleanup(self, usage: Dict[str, Any], result: CleanupValidationResult):
        """Validate temporary file cleanup effectiveness."""
        temp_files = usage.get('temp_files', 0)
        temp_size_mb = usage.get('temp_size_mb', 0)
        
        if self.thresholds.temp_files and temp_files > self.thresholds.temp_files:
            result.success = False
            result.issues.append(f"Temporary files {temp_files} exceeds threshold {self.thresholds.temp_files}")
            
        if self.thresholds.temp_size_mb and temp_size_mb > self.thresholds.temp_size_mb:
            result.success = False
            result.issues.append(f"Temporary file size {temp_size_mb:.1f}MB exceeds threshold {self.thresholds.temp_size_mb}MB")
            
    def _validate_process_cleanup(self, usage: Dict[str, Any], result: CleanupValidationResult):
        """Validate process cleanup effectiveness."""
        tracked_processes = usage.get('tracked_processes', 0)
        
        if tracked_processes > 0:
            result.warnings.append(f"Still tracking {tracked_processes} processes after cleanup")
            
    def _check_common_issues(self, usage: Dict[str, Any], result: CleanupValidationResult):
        """Check for common cleanup issues across resource types."""
        # Check for error conditions in usage metrics
        if 'error' in usage:
            result.success = False
            result.issues.append(f"Resource manager error: {usage['error']}")
            
        # Check for anomalous patterns
        if not usage:
            result.warnings.append("No usage metrics available - may indicate monitoring issue")
            
    def get_validation_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get validation summary for the specified number of days."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_validations = [
            v for v in self._validation_history 
            if v.validation_timestamp >= cutoff
        ]
        
        if not recent_validations:
            return {'message': 'No validation data available for the specified period'}
            
        # Calculate success rates by resource type
        success_rates = defaultdict(list)
        issue_counts = defaultdict(int)
        
        for validation in recent_validations:
            resource_type = validation.resource_type
            success_rates[resource_type].append(validation.success)
            issue_counts[resource_type] += len(validation.issues)
            
        summary = {
            'period_days': days,
            'total_validations': len(recent_validations),
            'resource_type_summary': {}
        }
        
        for resource_type, successes in success_rates.items():
            success_rate = sum(successes) / len(successes) if successes else 0
            summary['resource_type_summary'][resource_type.name] = {
                'validations': len(successes),
                'success_rate': success_rate,
                'total_issues': issue_counts[resource_type],
                'avg_issues_per_validation': issue_counts[resource_type] / len(successes) if successes else 0
            }
            
        return summary


# =====================================================================
# RESOURCE MONITOR
# =====================================================================

class ResourceMonitor:
    """Monitors resource usage patterns and trends."""
    
    def __init__(self, sample_interval_seconds: int = 60):
        self.sample_interval = sample_interval_seconds
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._snapshots = defaultdict(deque)  # maxlen set per resource type
        self._monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Configure snapshot retention per resource type
        for resource_type in ResourceType:
            self._snapshots[resource_type] = deque(maxlen=1440)  # 24 hours at 1-minute intervals
            
    def start_monitoring(self, orchestrator: 'AdvancedCleanupOrchestrator'):
        """Start continuous resource monitoring."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self._monitoring_active = True
        self._orchestrator_ref = weakref.ref(orchestrator)
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self._monitoring_active:
            return
            
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            
        self.logger.info("Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active and not self._stop_event.is_set():
            try:
                orchestrator = self._orchestrator_ref() if self._orchestrator_ref else None
                if orchestrator:
                    self._take_snapshots(orchestrator)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            self._stop_event.wait(self.sample_interval)
            
    def _take_snapshots(self, orchestrator: 'AdvancedCleanupOrchestrator'):
        """Take resource usage snapshots."""
        timestamp = datetime.now()
        usage_data = orchestrator.get_resource_usage()
        
        for resource_type, usage_metrics in usage_data.items():
            snapshot = ResourceSnapshot(
                timestamp=timestamp,
                resource_type=resource_type,
                usage_metrics=usage_metrics.copy()
            )
            
            # Check for threshold violations
            self._check_thresholds(snapshot, orchestrator.thresholds)
            
            self._snapshots[resource_type].append(snapshot)
            
    def _check_thresholds(self, snapshot: ResourceSnapshot, thresholds: ResourceThresholds):
        """Check for threshold violations in snapshot."""
        resource_type = snapshot.resource_type
        usage = snapshot.usage_metrics
        
        if resource_type == ResourceType.MEMORY:
            memory_mb = usage.get('memory_mb', 0)
            if thresholds.memory_mb and memory_mb > thresholds.memory_mb:
                snapshot.threshold_violations.append(f"Memory usage {memory_mb:.1f}MB > {thresholds.memory_mb}MB")
                
        elif resource_type == ResourceType.FILE_HANDLES:
            open_files = usage.get('open_files', 0)
            if thresholds.file_handles and open_files > thresholds.file_handles:
                snapshot.threshold_violations.append(f"Open files {open_files} > {thresholds.file_handles}")
                
        elif resource_type == ResourceType.DATABASE_CONNECTIONS:
            active_connections = usage.get('active_connections', 0)
            if thresholds.db_connections and active_connections > thresholds.db_connections:
                snapshot.threshold_violations.append(f"DB connections {active_connections} > {thresholds.db_connections}")
                
        elif resource_type == ResourceType.TEMPORARY_FILES:
            temp_files = usage.get('temp_files', 0)
            temp_size_mb = usage.get('temp_size_mb', 0)
            
            if thresholds.temp_files and temp_files > thresholds.temp_files:
                snapshot.threshold_violations.append(f"Temp files {temp_files} > {thresholds.temp_files}")
                
            if thresholds.temp_size_mb and temp_size_mb > thresholds.temp_size_mb:
                snapshot.threshold_violations.append(f"Temp size {temp_size_mb:.1f}MB > {thresholds.temp_size_mb}MB")
                
    def get_trend_analysis(self, resource_type: ResourceType, hours: int = 24) -> Dict[str, Any]:
        """Get trend analysis for a resource type."""
        cutoff = datetime.now() - timedelta(hours=hours)
        snapshots = [
            s for s in self._snapshots[resource_type]
            if s.timestamp >= cutoff
        ]
        
        if len(snapshots) < 2:
            return {'message': 'Insufficient data for trend analysis'}
            
        # Extract time series data for key metrics
        trends = {}
        
        if resource_type == ResourceType.MEMORY:
            memory_values = [s.usage_metrics.get('memory_mb', 0) for s in snapshots]
            trends['memory_mb'] = self._calculate_trend_stats(memory_values)
            
        elif resource_type == ResourceType.FILE_HANDLES:
            file_values = [s.usage_metrics.get('open_files', 0) for s in snapshots]
            trends['open_files'] = self._calculate_trend_stats(file_values)
            
        elif resource_type == ResourceType.DATABASE_CONNECTIONS:
            db_values = [s.usage_metrics.get('active_connections', 0) for s in snapshots]
            trends['active_connections'] = self._calculate_trend_stats(db_values)
            
        elif resource_type == ResourceType.TEMPORARY_FILES:
            file_values = [s.usage_metrics.get('temp_files', 0) for s in snapshots]
            size_values = [s.usage_metrics.get('temp_size_mb', 0) for s in snapshots]
            trends['temp_files'] = self._calculate_trend_stats(file_values)
            trends['temp_size_mb'] = self._calculate_trend_stats(size_values)
            
        # Calculate threshold violation rate
        total_violations = sum(len(s.threshold_violations) for s in snapshots)
        trends['threshold_violation_rate'] = total_violations / len(snapshots)
        
        return {
            'resource_type': resource_type.name,
            'period_hours': hours,
            'sample_count': len(snapshots),
            'trends': trends,
            'threshold_violations': total_violations
        }
        
    def _calculate_trend_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for a series of values."""
        if not values:
            return {}
            
        return {
            'current': values[-1],
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'trend_slope': self._calculate_trend_slope(values)
        }
        
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0
            
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
        
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get overall monitoring summary."""
        summary = {
            'monitoring_active': self._monitoring_active,
            'sample_interval_seconds': self.sample_interval,
            'resource_summaries': {}
        }
        
        for resource_type in ResourceType:
            snapshots = self._snapshots[resource_type]
            if snapshots:
                latest = snapshots[-1]
                violation_count = sum(len(s.threshold_violations) for s in snapshots)
                
                summary['resource_summaries'][resource_type.name] = {
                    'snapshot_count': len(snapshots),
                    'latest_timestamp': latest.timestamp.isoformat(),
                    'latest_metrics': latest.usage_metrics,
                    'total_threshold_violations': violation_count,
                    'latest_violations': latest.threshold_violations
                }
                
        return summary


# =====================================================================
# PERFORMANCE ANALYZER
# =====================================================================

class PerformanceAnalyzer:
    """Analyzes cleanup performance and identifies optimization opportunities."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._performance_data = deque(maxlen=1000)
        self._baseline_performance = {}
        
    def record_cleanup_operation(self, orchestrator: 'AdvancedCleanupOrchestrator', 
                               resource_types: Set[ResourceType], 
                               start_time: datetime, end_time: datetime) -> PerformanceMetrics:
        """Record performance metrics for a cleanup operation."""
        operation_id = f"cleanup_{int(time.time())}_{hash(frozenset(resource_types))}"
        duration = (end_time - start_time).total_seconds()
        
        # Get resource usage before and after (approximated)
        current_usage = orchestrator.get_resource_usage()
        
        # Calculate success rate from cleanup statistics
        stats = orchestrator.get_cleanup_statistics()
        total_ops = stats.get('total_operations', 1)
        successful_ops = stats.get('successful_operations', 0)
        success_rate = successful_ops / total_ops if total_ops > 0 else 0
        
        # Estimate memory impact
        memory_metrics = current_usage.get(ResourceType.MEMORY, {})
        memory_current = memory_metrics.get('memory_mb', 0)
        
        # Create performance metrics record
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            resources_cleaned=resource_types,
            success_rate=success_rate,
            resource_metrics={rt: current_usage.get(rt, {}) for rt in resource_types},
            memory_before_mb=memory_current,  # Approximation
            memory_after_mb=memory_current,   # Approximation
            memory_freed_mb=0  # Would need before/after measurement
        )
        
        self._performance_data.append(metrics)
        self._update_baseline_performance()
        
        return metrics
        
    def _update_baseline_performance(self):
        """Update baseline performance metrics."""
        if len(self._performance_data) < 10:
            return
            
        recent_data = list(self._performance_data)[-10:]
        
        # Calculate baseline durations by resource type combinations
        duration_by_resources = defaultdict(list)
        
        for metrics in recent_data:
            resource_key = frozenset(metrics.resources_cleaned)
            duration_by_resources[resource_key].append(metrics.duration_seconds)
            
        # Update baselines
        for resource_key, durations in duration_by_resources.items():
            if len(durations) >= 5:  # Need at least 5 samples
                self._baseline_performance[resource_key] = {
                    'mean_duration': statistics.mean(durations),
                    'median_duration': statistics.median(durations),
                    'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0
                }
                
    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over the specified period."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            m for m in self._performance_data
            if m.start_time >= cutoff
        ]
        
        if not recent_data:
            return {'message': 'No performance data available for the specified period'}
            
        # Group by resource type combinations
        performance_by_resources = defaultdict(list)
        for metrics in recent_data:
            resource_key = frozenset(metrics.resources_cleaned)
            performance_by_resources[resource_key].append(metrics)
            
        analysis = {
            'period_days': days,
            'total_operations': len(recent_data),
            'resource_combinations': {}
        }
        
        for resource_key, metrics_list in performance_by_resources.items():
            resource_names = [rt.name for rt in resource_key]
            durations = [m.duration_seconds for m in metrics_list]
            success_rates = [m.success_rate for m in metrics_list]
            
            # Performance analysis
            perf_analysis = {
                'operations': len(metrics_list),
                'duration_stats': {
                    'mean': statistics.mean(durations),
                    'median': statistics.median(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0
                },
                'success_rate_stats': {
                    'mean': statistics.mean(success_rates),
                    'min': min(success_rates),
                    'max': max(success_rates)
                },
                'trend_slope': self._calculate_performance_trend(durations)
            }
            
            # Compare to baseline if available
            baseline = self._baseline_performance.get(resource_key)
            if baseline:
                current_mean = perf_analysis['duration_stats']['mean']
                baseline_mean = baseline['mean_duration']
                
                perf_analysis['baseline_comparison'] = {
                    'baseline_mean': baseline_mean,
                    'current_mean': current_mean,
                    'performance_ratio': current_mean / baseline_mean if baseline_mean > 0 else float('inf'),
                    'degradation_detected': current_mean > baseline_mean * 1.5  # 50% slower
                }
                
            analysis['resource_combinations'][str(resource_names)] = perf_analysis
            
        return analysis
        
    def _calculate_performance_trend(self, durations: List[float]) -> float:
        """Calculate performance trend (negative slope = improving performance)."""
        if len(durations) < 2:
            return 0
            
        n = len(durations)
        x_values = list(range(n))
        
        # Linear regression
        sum_x = sum(x_values)
        sum_y = sum(durations)
        sum_xy = sum(x * y for x, y in zip(x_values, durations))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
        
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for cleanup performance optimization."""
        opportunities = []
        
        if len(self._performance_data) < 20:
            return [{'message': 'Insufficient data for optimization analysis'}]
            
        # Analyze recent performance data
        recent_data = list(self._performance_data)[-50:]
        
        # Group by resource combinations
        perf_by_resources = defaultdict(list)
        for metrics in recent_data:
            resource_key = frozenset(metrics.resources_cleaned)
            perf_by_resources[resource_key].append(metrics)
            
        for resource_key, metrics_list in perf_by_resources.items():
            if len(metrics_list) < 5:
                continue
                
            durations = [m.duration_seconds for m in metrics_list]
            success_rates = [m.success_rate for m in metrics_list]
            
            resource_names = [rt.name for rt in resource_key]
            
            # Check for performance issues
            mean_duration = statistics.mean(durations)
            mean_success = statistics.mean(success_rates)
            duration_variation = statistics.stdev(durations) if len(durations) > 1 else 0
            
            # Identify specific opportunities
            if mean_duration > 5.0:  # Operations taking longer than 5 seconds
                opportunities.append({
                    'type': 'slow_cleanup',
                    'resources': resource_names,
                    'issue': f'Cleanup taking {mean_duration:.2f}s on average',
                    'recommendation': 'Consider parallel cleanup or resource-specific optimizations'
                })
                
            if mean_success < 0.9:  # Less than 90% success rate
                opportunities.append({
                    'type': 'low_success_rate',
                    'resources': resource_names,
                    'issue': f'Success rate only {mean_success*100:.1f}%',
                    'recommendation': 'Investigate cleanup failures and improve retry logic'
                })
                
            if duration_variation > mean_duration * 0.5:  # High variability
                opportunities.append({
                    'type': 'inconsistent_performance',
                    'resources': resource_names,
                    'issue': f'High performance variability (std_dev: {duration_variation:.2f}s)',
                    'recommendation': 'Investigate causes of performance inconsistency'
                })
                
        return opportunities if opportunities else [{'message': 'No optimization opportunities identified'}]


# =====================================================================
# CLEANUP REPORTER
# =====================================================================

class CleanupReporter:
    """Generates comprehensive cleanup reports and dashboards."""
    
    def __init__(self, report_dir: Path = None):
        self.report_dir = report_dir or Path("test_data/reports/cleanup")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def generate_comprehensive_report(self, 
                                    orchestrator: 'AdvancedCleanupOrchestrator',
                                    validator: CleanupValidator = None,
                                    monitor: ResourceMonitor = None,
                                    analyzer: PerformanceAnalyzer = None) -> Dict[str, Any]:
        """Generate a comprehensive cleanup system report."""
        timestamp = datetime.now()
        report_id = f"cleanup_report_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        report = {
            'report_id': report_id,
            'generated_at': timestamp.isoformat(),
            'system_overview': self._get_system_overview(orchestrator),
            'resource_status': self._get_resource_status(orchestrator),
            'cleanup_statistics': orchestrator.get_cleanup_statistics()
        }
        
        # Add validation results if validator is provided
        if validator:
            validation_results = validator.validate_cleanup(orchestrator)
            report['validation_results'] = {
                rt.name: asdict(result) for rt, result in validation_results.items()
            }
            report['validation_summary'] = validator.get_validation_summary()
            
        # Add monitoring data if monitor is provided
        if monitor:
            report['monitoring_summary'] = monitor.get_monitoring_summary()
            report['trend_analysis'] = {}
            for resource_type in ResourceType:
                trend = monitor.get_trend_analysis(resource_type, hours=24)
                if 'message' not in trend:  # Skip if no data
                    report['trend_analysis'][resource_type.name] = trend
                    
        # Add performance analysis if analyzer is provided
        if analyzer:
            report['performance_analysis'] = analyzer.analyze_performance_trends(days=7)
            report['optimization_opportunities'] = analyzer.identify_optimization_opportunities()
            
        # Add system health assessment
        report['health_assessment'] = self._assess_system_health(report)
        
        # Save report to file
        report_file = self.report_dir / f"{report_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate summary
        summary_file = self.report_dir / f"{report_id}_summary.txt"
        self._generate_text_summary(report, summary_file)
        
        self.logger.info(f"Comprehensive report generated: {report_file}")
        return report
        
    def _get_system_overview(self, orchestrator: 'AdvancedCleanupOrchestrator') -> Dict[str, Any]:
        """Get system overview information."""
        process = psutil.Process()
        
        return {
            'process_id': process.pid,
            'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'cleanup_policy': asdict(orchestrator.policy),
            'resource_thresholds': asdict(orchestrator.thresholds)
        }
        
    def _get_resource_status(self, orchestrator: 'AdvancedCleanupOrchestrator') -> Dict[str, Any]:
        """Get current resource status."""
        usage = orchestrator.get_resource_usage()
        return {rt.name: metrics for rt, metrics in usage.items()}
        
    def _assess_system_health(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health based on report data."""
        health_score = 100  # Start with perfect health
        issues = []
        recommendations = []
        
        # Check resource usage
        resource_status = report.get('resource_status', {})
        for resource_name, metrics in resource_status.items():
            if 'error' in metrics:
                health_score -= 20
                issues.append(f"Resource monitoring error for {resource_name}")
                
        # Check validation results
        validation_results = report.get('validation_results', {})
        for resource_name, result in validation_results.items():
            if not result.get('success', True):
                health_score -= 10
                issues.extend(result.get('issues', []))
                
        # Check cleanup statistics
        cleanup_stats = report.get('cleanup_statistics', {})
        total_ops = cleanup_stats.get('total_operations', 0)
        successful_ops = cleanup_stats.get('successful_operations', 0)
        
        if total_ops > 0:
            success_rate = successful_ops / total_ops
            if success_rate < 0.9:
                health_score -= 15
                issues.append(f"Low cleanup success rate: {success_rate*100:.1f}%")
                
        # Check performance issues
        optimization_opps = report.get('optimization_opportunities', [])
        if optimization_opps and 'message' not in optimization_opps[0]:
            health_score -= min(len(optimization_opps) * 5, 20)
            for opp in optimization_opps:
                issues.append(opp.get('issue', 'Performance optimization opportunity'))
                recommendations.append(opp.get('recommendation', 'See optimization opportunities'))
                
        # Ensure health score doesn't go below 0
        health_score = max(0, health_score)
        
        # Determine health status
        if health_score >= 90:
            status = "EXCELLENT"
        elif health_score >= 75:
            status = "GOOD"
        elif health_score >= 60:
            status = "FAIR"
        elif health_score >= 40:
            status = "POOR"
        else:
            status = "CRITICAL"
            
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'assessment_timestamp': datetime.now().isoformat()
        }
        
    def _generate_text_summary(self, report: Dict[str, Any], summary_file: Path):
        """Generate human-readable text summary."""
        health = report.get('health_assessment', {})
        
        summary_lines = [
            "=" * 60,
            "CLEANUP SYSTEM HEALTH REPORT SUMMARY",
            "=" * 60,
            f"Generated: {report['generated_at']}",
            f"Report ID: {report['report_id']}",
            "",
            f"OVERALL HEALTH: {health.get('status', 'UNKNOWN')} ({health.get('health_score', 0)}/100)",
            "",
        ]
        
        # Add system overview
        system = report.get('system_overview', {})
        summary_lines.extend([
            "SYSTEM OVERVIEW:",
            f"  Memory Usage: {system.get('memory_usage_mb', 0):.1f} MB",
            f"  CPU Usage: {system.get('cpu_percent', 0):.1f}%",
            f"  Open Files: {system.get('open_files', 0)}",
            f"  Threads: {system.get('num_threads', 0)}",
            ""
        ])
        
        # Add cleanup statistics
        stats = report.get('cleanup_statistics', {})
        total_ops = stats.get('total_operations', 0)
        successful_ops = stats.get('successful_operations', 0)
        success_rate = (successful_ops / total_ops * 100) if total_ops > 0 else 0
        
        summary_lines.extend([
            "CLEANUP STATISTICS:",
            f"  Total Operations: {total_ops}",
            f"  Successful Operations: {successful_ops}",
            f"  Success Rate: {success_rate:.1f}%",
            f"  Average Duration: {stats.get('average_duration', 0):.3f}s",
            ""
        ])
        
        # Add issues and recommendations
        issues = health.get('issues', [])
        recommendations = health.get('recommendations', [])
        
        if issues:
            summary_lines.extend([
                "ISSUES IDENTIFIED:",
                *[f"  - {issue}" for issue in issues[:5]],  # Limit to top 5
                ""
            ])
            
        if recommendations:
            summary_lines.extend([
                "RECOMMENDATIONS:",
                *[f"  - {rec}" for rec in recommendations[:5]],  # Limit to top 5
                ""
            ])
            
        summary_lines.extend([
            "=" * 60,
            f"For detailed information, see: {report['report_id']}.json",
            "=" * 60
        ])
        
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))


# =====================================================================
# ALERT SYSTEM
# =====================================================================

class AlertSystem:
    """Automated alerting for cleanup issues and performance problems."""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._alert_history = deque(maxlen=1000)
        self._last_notification_times = {}
        
    def check_and_alert(self, orchestrator: 'AdvancedCleanupOrchestrator', 
                       monitor: ResourceMonitor = None,
                       validator: CleanupValidator = None) -> List[Dict[str, Any]]:
        """Check system state and generate alerts if needed."""
        if not self.config.enabled:
            return []
            
        alerts = []
        current_time = datetime.now()
        
        # Check resource usage alerts
        alerts.extend(self._check_resource_alerts(orchestrator, current_time))
        
        # Check cleanup failure alerts
        alerts.extend(self._check_cleanup_failure_alerts(orchestrator, current_time))
        
        # Check monitoring alerts if monitor provided
        if monitor:
            alerts.extend(self._check_monitoring_alerts(monitor, current_time))
            
        # Check validation alerts if validator provided
        if validator:
            alerts.extend(self._check_validation_alerts(validator, current_time))
            
        # Process and send alerts
        for alert in alerts:
            self._process_alert(alert, current_time)
            
        return alerts
        
    def _check_resource_alerts(self, orchestrator: 'AdvancedCleanupOrchestrator', 
                             current_time: datetime) -> List[Dict[str, Any]]:
        """Check for resource usage alerts."""
        alerts = []
        usage = orchestrator.get_resource_usage()
        
        # Check memory usage
        memory_metrics = usage.get(ResourceType.MEMORY, {})
        memory_mb = memory_metrics.get('memory_mb', 0)
        
        if memory_mb > self.config.memory_threshold_mb:
            alerts.append({
                'type': 'resource_threshold',
                'resource': 'memory',
                'severity': 'warning',
                'message': f'Memory usage {memory_mb:.1f}MB exceeds threshold {self.config.memory_threshold_mb}MB',
                'current_value': memory_mb,
                'threshold': self.config.memory_threshold_mb
            })
            
        # Check file handles
        file_metrics = usage.get(ResourceType.FILE_HANDLES, {})
        open_files = file_metrics.get('open_files', 0)
        
        if open_files > self.config.file_handle_threshold:
            alerts.append({
                'type': 'resource_threshold',
                'resource': 'file_handles',
                'severity': 'warning',
                'message': f'Open files {open_files} exceeds threshold {self.config.file_handle_threshold}',
                'current_value': open_files,
                'threshold': self.config.file_handle_threshold
            })
            
        return alerts
        
    def _check_cleanup_failure_alerts(self, orchestrator: 'AdvancedCleanupOrchestrator', 
                                    current_time: datetime) -> List[Dict[str, Any]]:
        """Check for cleanup failure alerts."""
        alerts = []
        stats = orchestrator.get_cleanup_statistics()
        
        total_ops = stats.get('total_operations', 0)
        successful_ops = stats.get('successful_operations', 0)
        failed_ops = total_ops - successful_ops
        
        if failed_ops >= self.config.cleanup_failure_threshold:
            alerts.append({
                'type': 'cleanup_failures',
                'severity': 'error',
                'message': f'{failed_ops} cleanup failures detected (threshold: {self.config.cleanup_failure_threshold})',
                'failed_operations': failed_ops,
                'total_operations': total_ops,
                'success_rate': (successful_ops / total_ops * 100) if total_ops > 0 else 0
            })
            
        return alerts
        
    def _check_monitoring_alerts(self, monitor: ResourceMonitor, 
                               current_time: datetime) -> List[Dict[str, Any]]:
        """Check for monitoring-related alerts."""
        alerts = []
        
        if not monitor._monitoring_active:
            alerts.append({
                'type': 'monitoring_inactive',
                'severity': 'warning',
                'message': 'Resource monitoring is not active',
            })
            
        return alerts
        
    def _check_validation_alerts(self, validator: CleanupValidator, 
                               current_time: datetime) -> List[Dict[str, Any]]:
        """Check for validation-related alerts."""
        alerts = []
        
        # Check recent validation failures
        summary = validator.get_validation_summary(days=1)
        
        if 'resource_type_summary' in summary:
            for resource_name, resource_summary in summary['resource_type_summary'].items():
                success_rate = resource_summary.get('success_rate', 1.0)
                if success_rate < 0.8:  # Less than 80% success rate
                    alerts.append({
                        'type': 'validation_failure',
                        'resource': resource_name,
                        'severity': 'error',
                        'message': f'Validation success rate for {resource_name} is {success_rate*100:.1f}%',
                        'success_rate': success_rate
                    })
                    
        return alerts
        
    def _process_alert(self, alert: Dict[str, Any], current_time: datetime):
        """Process and potentially send an alert."""
        alert_key = f"{alert['type']}_{alert.get('resource', 'general')}"
        
        # Check cooldown period
        last_notification = self._last_notification_times.get(alert_key)
        if last_notification:
            cooldown = timedelta(minutes=self.config.notification_cooldown_minutes)
            if current_time - last_notification < cooldown:
                return  # Skip due to cooldown
                
        # Add timestamp and log the alert
        alert['timestamp'] = current_time.isoformat()
        alert['alert_id'] = f"alert_{int(time.time())}_{hash(alert_key)}"
        
        severity = alert.get('severity', 'info')
        message = alert.get('message', 'Unknown alert')
        
        if severity == 'error':
            self.logger.error(f"CLEANUP ALERT: {message}")
        elif severity == 'warning':
            self.logger.warning(f"CLEANUP ALERT: {message}")
        else:
            self.logger.info(f"CLEANUP ALERT: {message}")
            
        # Store alert in history
        self._alert_history.append(alert)
        
        # Update last notification time
        self._last_notification_times[alert_key] = current_time
        
        # Here you could add integration with external alerting systems
        # (email, Slack, PagerDuty, etc.)
        
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self._alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff
        ]
        
        if not recent_alerts:
            return {'message': 'No alerts in the specified period'}
            
        # Group by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for alert in recent_alerts:
            by_type[alert['type']] += 1
            by_severity[alert.get('severity', 'info')] += 1
            
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'most_recent': recent_alerts[-1] if recent_alerts else None
        }


# =====================================================================
# MAIN INTEGRATION CLASS
# =====================================================================

class CleanupValidationMonitor:
    """Integrated cleanup validation and monitoring system."""
    
    def __init__(self, 
                 cleanup_policy: CleanupPolicy = None,
                 thresholds: ResourceThresholds = None,
                 alert_config: AlertConfig = None,
                 report_dir: Path = None):
        
        self.orchestrator = AdvancedCleanupOrchestrator(cleanup_policy, thresholds)
        self.validator = CleanupValidator(thresholds)
        self.monitor = ResourceMonitor()
        self.analyzer = PerformanceAnalyzer()
        self.reporter = CleanupReporter(report_dir)
        self.alert_system = AlertSystem(alert_config)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._monitoring_active = False
        
    def start_monitoring(self):
        """Start the complete monitoring system."""
        self.monitor.start_monitoring(self.orchestrator)
        self._monitoring_active = True
        self.logger.info("Cleanup validation and monitoring system started")
        
    def stop_monitoring(self):
        """Stop the complete monitoring system."""
        self.monitor.stop_monitoring()
        self._monitoring_active = False
        self.logger.info("Cleanup validation and monitoring system stopped")
        
    def perform_cleanup_cycle(self, force: bool = False, 
                            resource_types: Set[ResourceType] = None) -> Dict[str, Any]:
        """Perform a complete cleanup cycle with validation and monitoring."""
        start_time = datetime.now()
        
        # Record performance start
        resource_types = resource_types or set(ResourceType)
        
        # Perform cleanup
        cleanup_success = self.orchestrator.cleanup(force=force, resource_types=resource_types)
        
        end_time = datetime.now()
        
        # Record performance metrics
        performance = self.analyzer.record_cleanup_operation(
            self.orchestrator, resource_types, start_time, end_time
        )
        
        # Validate cleanup
        validation_results = self.validator.validate_cleanup(
            self.orchestrator, resource_types
        )
        
        # Check for alerts
        alerts = self.alert_system.check_and_alert(
            self.orchestrator, self.monitor, self.validator
        )
        
        return {
            'cleanup_success': cleanup_success,
            'performance_metrics': asdict(performance),
            'validation_results': {rt.name: asdict(result) for rt, result in validation_results.items()},
            'alerts': alerts,
            'timestamp': end_time.isoformat()
        }
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system report."""
        return self.reporter.generate_comprehensive_report(
            self.orchestrator,
            self.validator,
            self.monitor,
            self.analyzer
        )
        
    @contextmanager
    def monitoring_context(self):
        """Context manager for automatic monitoring lifecycle."""
        try:
            self.start_monitoring()
            yield self
        finally:
            self.stop_monitoring()


# =====================================================================
# PYTEST INTEGRATION
# =====================================================================

@pytest.fixture(scope="session")
def cleanup_validation_monitor() -> Generator[CleanupValidationMonitor, None, None]:
    """Provide integrated cleanup validation and monitoring system."""
    policy = CleanupPolicy(
        strategy=CleanupStrategy.DEFERRED,
        scope=CleanupScope.SESSION,
        validate_cleanup=True,
        report_cleanup=True
    )
    
    thresholds = ResourceThresholds(
        memory_mb=1024,
        file_handles=500,
        db_connections=25
    )
    
    alert_config = AlertConfig(
        enabled=True,
        memory_threshold_mb=2048,
        file_handle_threshold=1000
    )
    
    monitor = CleanupValidationMonitor(policy, thresholds, alert_config)
    
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()
        # Generate final report
        try:
            final_report = monitor.generate_comprehensive_report()
            print(f"Final cleanup report generated: {final_report.get('report_id', 'unknown')}")
        except Exception as e:
            print(f"Error generating final report: {e}")


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create integrated monitoring system
    monitor = CleanupValidationMonitor()
    
    # Start monitoring
    with monitor.monitoring_context():
        # Simulate some cleanup cycles
        for i in range(3):
            result = monitor.perform_cleanup_cycle(force=True)
            print(f"Cleanup cycle {i+1}: {result['cleanup_success']}")
            time.sleep(1)
            
        # Generate report
        report = monitor.generate_comprehensive_report()
        print(f"Generated report: {report['report_id']}")