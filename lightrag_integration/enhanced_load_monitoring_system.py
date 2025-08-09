"""
Enhanced Load Monitoring and Detection System for Clinical Metabolomics Oracle
===============================================================================

This module provides advanced load monitoring capabilities that integrate with and enhance
the existing graceful degradation system. It implements:

1. Real-time metrics collection with minimal overhead
2. Hysteresis-based threshold management for stability
3. Advanced analytics and trend detection
4. Production monitoring system integration
5. Efficient caching and performance optimizations

Key Features:
- CPU utilization with multi-core awareness
- Memory pressure with swap monitoring
- Request queue depth tracking
- Response time percentiles (P95, P99, P99.9)
- Error rate categorization and trend analysis
- Network I/O and connection pool metrics
- Hysteresis mechanism for stable threshold transitions
- Integration with existing ProductionMonitoring

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
"""

import asyncio
import logging
import time
import statistics
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import json
import numpy as np

# Import from existing graceful degradation system
try:
    from .graceful_degradation_system import (
        SystemLoadLevel, SystemLoadMetrics, LoadThresholds,
        LoadDetectionSystem, GracefulDegradationManager
    )
    GRACEFUL_DEGRADATION_AVAILABLE = True
except ImportError:
    GRACEFUL_DEGRADATION_AVAILABLE = False
    logging.warning("Graceful degradation system not available - using standalone mode")
    
    # Define standalone versions if not available
    from enum import IntEnum
    
    class SystemLoadLevel(IntEnum):
        """System load levels for standalone mode."""
        NORMAL = 0
        ELEVATED = 1
        HIGH = 2
        CRITICAL = 3
        EMERGENCY = 4
    
    class LoadThresholds:
        """Basic load thresholds for standalone mode."""
        def __init__(self, **kwargs):
            # CPU thresholds
            self.cpu_normal = kwargs.get('cpu_normal', 50.0)
            self.cpu_elevated = kwargs.get('cpu_elevated', 65.0)
            self.cpu_high = kwargs.get('cpu_high', 80.0)
            self.cpu_critical = kwargs.get('cpu_critical', 90.0)
            self.cpu_emergency = kwargs.get('cpu_emergency', 95.0)
            
            # Memory thresholds
            self.memory_normal = kwargs.get('memory_normal', 60.0)
            self.memory_elevated = kwargs.get('memory_elevated', 70.0)
            self.memory_high = kwargs.get('memory_high', 75.0)
            self.memory_critical = kwargs.get('memory_critical', 85.0)
            self.memory_emergency = kwargs.get('memory_emergency', 90.0)
            
            # Queue thresholds
            self.queue_normal = kwargs.get('queue_normal', 10)
            self.queue_elevated = kwargs.get('queue_elevated', 20)
            self.queue_high = kwargs.get('queue_high', 40)
            self.queue_critical = kwargs.get('queue_critical', 80)
            self.queue_emergency = kwargs.get('queue_emergency', 150)
            
            # Response time thresholds
            self.response_p95_normal = kwargs.get('response_p95_normal', 1000.0)
            self.response_p95_elevated = kwargs.get('response_p95_elevated', 2000.0)
            self.response_p95_high = kwargs.get('response_p95_high', 3500.0)
            self.response_p95_critical = kwargs.get('response_p95_critical', 5000.0)
            self.response_p95_emergency = kwargs.get('response_p95_emergency', 8000.0)
            
            # Error rate thresholds
            self.error_rate_normal = kwargs.get('error_rate_normal', 0.1)
            self.error_rate_elevated = kwargs.get('error_rate_elevated', 0.5)
            self.error_rate_high = kwargs.get('error_rate_high', 1.0)
            self.error_rate_critical = kwargs.get('error_rate_critical', 2.0)
            self.error_rate_emergency = kwargs.get('error_rate_emergency', 5.0)

# Import production monitoring system
try:
    from .production_monitoring import ProductionMonitoring
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False
    logging.warning("Production monitoring not available")


# ============================================================================
# ENHANCED METRICS AND THRESHOLDS
# ============================================================================

@dataclass
class EnhancedSystemLoadMetrics:
    """Enhanced system load metrics with additional monitoring data."""
    
    # Base metrics (compatible with SystemLoadMetrics)
    timestamp: datetime
    cpu_utilization: float
    memory_pressure: float
    request_queue_depth: int
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    active_connections: int
    disk_io_wait: float
    
    # Enhanced metrics
    cpu_per_core: List[float] = field(default_factory=list)
    memory_available_mb: float = 0.0
    swap_pressure: float = 0.0
    response_time_p99_9: float = 0.0
    network_latency_estimate: float = 0.0
    error_categories: Dict[str, int] = field(default_factory=dict)
    trend_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Derived metrics
    load_level: SystemLoadLevel = SystemLoadLevel.NORMAL
    load_score: float = 0.0
    degradation_recommended: bool = False
    hysteresis_factor_applied: float = 1.0
    
    def to_base_metrics(self) -> 'SystemLoadMetrics':
        """Convert to base SystemLoadMetrics for compatibility."""
        if GRACEFUL_DEGRADATION_AVAILABLE:
            return SystemLoadMetrics(
                timestamp=self.timestamp,
                cpu_utilization=self.cpu_utilization,
                memory_pressure=self.memory_pressure,
                request_queue_depth=self.request_queue_depth,
                response_time_p95=self.response_time_p95,
                response_time_p99=self.response_time_p99,
                error_rate=self.error_rate,
                active_connections=self.active_connections,
                disk_io_wait=self.disk_io_wait,
                load_level=self.load_level,
                load_score=self.load_score,
                degradation_recommended=self.degradation_recommended
            )
        else:
            # Return as dict if base class not available
            return {
                'timestamp': self.timestamp,
                'cpu_utilization': self.cpu_utilization,
                'memory_pressure': self.memory_pressure,
                'request_queue_depth': self.request_queue_depth,
                'response_time_p95': self.response_time_p95,
                'response_time_p99': self.response_time_p99,
                'error_rate': self.error_rate,
                'active_connections': self.active_connections,
                'disk_io_wait': self.disk_io_wait,
                'load_level': self.load_level.name,
                'load_score': self.load_score,
                'degradation_recommended': self.degradation_recommended
            }


@dataclass
class HysteresisConfig:
    """Configuration for hysteresis behavior."""
    enabled: bool = True
    down_factor: float = 0.85  # Factor for lowering thresholds when moving down
    up_factor: float = 1.0     # Factor for raising thresholds when moving up
    stability_window: int = 3   # Number of measurements to consider for stability
    min_dwell_time: float = 30.0  # Minimum time to stay at a level (seconds)


# ============================================================================
# ENHANCED LOAD DETECTION SYSTEM
# ============================================================================

class EnhancedLoadDetectionSystem:
    """
    Advanced load detection system with production-ready features:
    
    - Real-time metrics with optimized collection
    - Hysteresis for threshold stability  
    - Trend analysis and prediction
    - Production system integration
    - Performance-optimized caching
    """
    
    def __init__(self,
                 thresholds: Optional[LoadThresholds] = None,
                 hysteresis_config: Optional[HysteresisConfig] = None,
                 monitoring_interval: float = 5.0,
                 production_monitoring: Optional[Any] = None,
                 enable_trend_analysis: bool = True):
        
        self.thresholds = thresholds or self._get_production_thresholds()
        self.hysteresis_config = hysteresis_config or HysteresisConfig()
        self.monitoring_interval = monitoring_interval
        self.production_monitoring = production_monitoring
        self.enable_trend_analysis = enable_trend_analysis
        self.logger = logging.getLogger(__name__)
        
        # Enhanced metrics history with categorization
        self.metrics_history: deque = deque(maxlen=500)  # Large history for trend analysis
        self.load_level_history: deque = deque(maxlen=100)
        self.level_transition_times: Dict[SystemLoadLevel, datetime] = {}
        
        # Current state tracking
        self.current_metrics: Optional[EnhancedSystemLoadMetrics] = None
        self.current_load_level: SystemLoadLevel = SystemLoadLevel.NORMAL
        self.previous_load_level: SystemLoadLevel = SystemLoadLevel.NORMAL
        self.last_level_change_time: datetime = datetime.now()
        
        # Monitoring state with thread safety
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[EnhancedSystemLoadMetrics], None]] = []
        self._metrics_lock = threading.Lock()
        
        # Performance optimization caches
        self._cached_cpu_times: Optional[Any] = None
        self._cached_network_io: Optional[Any] = None
        self._cached_percentiles: Optional[Tuple[float, float, float]] = None
        self._cached_percentiles_timestamp: float = 0.0
        self._percentile_cache_ttl: float = 2.0
        
        # Request tracking for realistic metrics
        self._response_times: deque = deque(maxlen=5000)  # Large sample for accuracy
        self._error_counts: defaultdict = defaultdict(int)  # Categorized error tracking
        self._total_requests: int = 0
        self._request_queue_depth: int = 0
        self._active_connections: int = 0
        
        # Trend analysis components
        self._trend_analyzer: Optional['TrendAnalyzer'] = None
        if enable_trend_analysis:
            self._trend_analyzer = TrendAnalyzer()
        
        # Initialize system integration
        self._initialize_system_integration()
        
        self.logger.info("Enhanced Load Detection System initialized")
    
    def _get_production_thresholds(self) -> LoadThresholds:
        """Get production-optimized thresholds."""
        if GRACEFUL_DEGRADATION_AVAILABLE:
            return LoadThresholds(
                # Aggressive CPU thresholds for production
                cpu_normal=45.0,
                cpu_elevated=60.0,
                cpu_high=75.0,
                cpu_critical=85.0,
                cpu_emergency=92.0,
                
                # Memory thresholds with swap consideration
                memory_normal=55.0,
                memory_elevated=65.0,
                memory_high=75.0,
                memory_critical=82.0,
                memory_emergency=88.0,
                
                # Request queue thresholds
                queue_normal=5,
                queue_elevated=15,
                queue_high=30,
                queue_critical=60,
                queue_emergency=120,
                
                # Response time thresholds (ms)
                response_p95_normal=1000.0,
                response_p95_elevated=2000.0,
                response_p95_high=3500.0,
                response_p95_critical=5000.0,
                response_p95_emergency=8000.0,
                
                # Error rate thresholds (%)
                error_rate_normal=0.1,
                error_rate_elevated=0.5,
                error_rate_high=1.0,
                error_rate_critical=2.0,
                error_rate_emergency=5.0
            )
        else:
            # Return dictionary if base class not available
            return {
                'cpu_thresholds': [45, 60, 75, 85, 92],
                'memory_thresholds': [55, 65, 75, 82, 88],
                'queue_thresholds': [5, 15, 30, 60, 120],
                'response_thresholds': [1000, 2000, 3500, 5000, 8000],
                'error_thresholds': [0.1, 0.5, 1.0, 2.0, 5.0]
            }
    
    def _initialize_system_integration(self):
        """Initialize integration with production systems."""
        if self.production_monitoring and PRODUCTION_MONITORING_AVAILABLE:
            try:
                # Register with production monitoring
                if hasattr(self.production_monitoring, 'register_enhanced_load_detector'):
                    self.production_monitoring.register_enhanced_load_detector(self)
                elif hasattr(self.production_monitoring, 'register_load_detector'):
                    self.production_monitoring.register_load_detector(self)
                
                self.logger.info("Integrated with production monitoring system")
            except Exception as e:
                self.logger.warning(f"Failed to integrate with production monitoring: {e}")
    
    def add_load_change_callback(self, callback: Callable[[EnhancedSystemLoadMetrics], None]):
        """Add callback for load level changes."""
        self._callbacks.append(callback)
    
    def record_request_metrics(self, response_time_ms: float, error_type: Optional[str] = None):
        """Record metrics for a completed request."""
        with self._metrics_lock:
            self._response_times.append(response_time_ms)
            self._total_requests += 1
            
            if error_type:
                self._error_counts[error_type] += 1
    
    def update_queue_depth(self, depth: int):
        """Update current request queue depth."""
        self._request_queue_depth = depth
    
    def update_connection_count(self, count: int):
        """Update active connection count."""
        self._active_connections = count
    
    async def start_monitoring(self):
        """Start the enhanced monitoring system."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Enhanced load monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Enhanced load monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop with optimized metrics collection."""
        while self._monitoring_active:
            try:
                # Collect enhanced metrics
                metrics = await self._collect_enhanced_metrics()
                
                # Update state
                with self._metrics_lock:
                    self.current_metrics = metrics
                    self.previous_load_level = self.current_load_level
                    self.current_load_level = metrics.load_level
                    
                    # Track level changes
                    if self.current_load_level != self.previous_load_level:
                        self.last_level_change_time = datetime.now()
                        self.level_transition_times[self.current_load_level] = self.last_level_change_time
                        
                        self.logger.info(f"Load level changed: {self.previous_load_level.name} → {self.current_load_level.name}")
                    
                    # Update history
                    self.metrics_history.append(metrics)
                    self.load_level_history.append((datetime.now(), self.current_load_level))
                
                # Notify callbacks if level changed
                if self.current_load_level != self.previous_load_level:
                    for callback in self._callbacks:
                        try:
                            callback(metrics)
                        except Exception as e:
                            self.logger.error(f"Error in load change callback: {e}")
                
                # Sync with production monitoring
                if self.production_monitoring:
                    await self._sync_with_production_system(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_enhanced_metrics(self) -> EnhancedSystemLoadMetrics:
        """Collect comprehensive system metrics with optimizations."""
        current_time = time.time()
        
        try:
            # Optimized CPU metrics with per-core data
            cpu_metrics = await self._get_cpu_metrics_async()
            
            # Enhanced memory metrics
            memory_metrics = self._get_memory_metrics_enhanced()
            
            # I/O and network metrics
            io_metrics = self._get_io_metrics_enhanced()
            
            # Response time percentiles with caching
            response_metrics = self._get_response_metrics_cached(current_time)
            
            # Error metrics with categorization
            error_metrics = self._get_error_metrics_categorized()
            
            # Connection and queue metrics
            connection_metrics = self._get_connection_metrics()
            
            # Create enhanced metrics object
            metrics = EnhancedSystemLoadMetrics(
                timestamp=datetime.now(),
                
                # Base metrics
                cpu_utilization=cpu_metrics['avg_utilization'],
                memory_pressure=memory_metrics['pressure'],
                request_queue_depth=connection_metrics['queue_depth'],
                response_time_p95=response_metrics['p95'],
                response_time_p99=response_metrics['p99'],
                error_rate=error_metrics['total_rate'],
                active_connections=connection_metrics['active_count'],
                disk_io_wait=io_metrics['disk_wait'],
                
                # Enhanced metrics
                cpu_per_core=cpu_metrics['per_core'],
                memory_available_mb=memory_metrics['available_mb'],
                swap_pressure=memory_metrics['swap_pressure'],
                response_time_p99_9=response_metrics['p99_9'],
                network_latency_estimate=io_metrics['network_latency'],
                error_categories=error_metrics['categories'],
            )
            
            # Apply hysteresis and calculate load level
            metrics.load_level = self._calculate_load_level_with_hysteresis(metrics)
            metrics.load_score = self._calculate_enhanced_load_score(metrics)
            metrics.degradation_recommended = metrics.load_level > SystemLoadLevel.NORMAL
            
            # Add trend analysis if enabled
            if self._trend_analyzer:
                metrics.trend_indicators = self._trend_analyzer.analyze(metrics, list(self.metrics_history))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting enhanced metrics: {e}")
            # Return safe defaults
            return self._get_safe_default_metrics()
    
    async def _get_cpu_metrics_async(self) -> Dict[str, Any]:
        """Get CPU metrics asynchronously with per-core data."""
        try:
            # Get per-CPU percentages
            cpu_percents = psutil.cpu_percent(percpu=True, interval=0.1)
            avg_utilization = statistics.mean(cpu_percents) if cpu_percents else 0.0
            
            return {
                'avg_utilization': avg_utilization,
                'per_core': cpu_percents,
                'core_count': len(cpu_percents)
            }
        except Exception as e:
            self.logger.warning(f"Error getting CPU metrics: {e}")
            return {
                'avg_utilization': 0.0,
                'per_core': [],
                'core_count': 1
            }
    
    def _get_memory_metrics_enhanced(self) -> Dict[str, float]:
        """Get comprehensive memory metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Enhanced memory pressure calculation
            memory_pressure = memory.percent
            swap_pressure = swap.percent if swap.total > 0 else 0.0
            
            # Adjust memory pressure based on swap usage
            if swap_pressure > 0:
                memory_pressure = min(100, memory_pressure + (swap_pressure * 0.3))
            
            return {
                'pressure': memory_pressure,
                'available_mb': memory.available / (1024 * 1024),
                'swap_pressure': swap_pressure,
                'used_mb': memory.used / (1024 * 1024),
                'cached_mb': getattr(memory, 'cached', 0) / (1024 * 1024)
            }
        except Exception as e:
            self.logger.warning(f"Error getting memory metrics: {e}")
            return {
                'pressure': 0.0,
                'available_mb': 1024.0,
                'swap_pressure': 0.0,
                'used_mb': 0.0,
                'cached_mb': 0.0
            }
    
    def _get_io_metrics_enhanced(self) -> Dict[str, float]:
        """Get enhanced I/O and network metrics."""
        try:
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_wait = 0.0
            if disk_io:
                disk_wait = (getattr(disk_io, 'read_time', 0) + 
                           getattr(disk_io, 'write_time', 0))
            
            # Network metrics with latency estimation
            network_io = psutil.net_io_counters()
            network_latency = 0.0
            
            if network_io and self._cached_network_io:
                # Calculate network pressure from drops and errors
                drops_delta = ((network_io.dropin + network_io.dropout) - 
                              (self._cached_network_io.dropin + self._cached_network_io.dropout))
                errors_delta = ((getattr(network_io, 'errin', 0) + getattr(network_io, 'errout', 0)) -
                               (getattr(self._cached_network_io, 'errin', 0) + getattr(self._cached_network_io, 'errout', 0)))
                
                if drops_delta > 0 or errors_delta > 0:
                    network_latency = min(100, (drops_delta + errors_delta) * 5)
            
            self._cached_network_io = network_io
            
            return {
                'disk_wait': disk_wait,
                'network_latency': network_latency,
                'disk_read_mb': getattr(disk_io, 'read_bytes', 0) / (1024 * 1024) if disk_io else 0,
                'disk_write_mb': getattr(disk_io, 'write_bytes', 0) / (1024 * 1024) if disk_io else 0
            }
        except Exception as e:
            self.logger.warning(f"Error getting I/O metrics: {e}")
            return {
                'disk_wait': 0.0,
                'network_latency': 0.0,
                'disk_read_mb': 0.0,
                'disk_write_mb': 0.0
            }
    
    def _get_response_metrics_cached(self, current_time: float) -> Dict[str, float]:
        """Get response time metrics with intelligent caching."""
        # Check cache validity
        if (self._cached_percentiles and 
            current_time - self._cached_percentiles_timestamp < self._percentile_cache_ttl):
            p95, p99, p99_9 = self._cached_percentiles
            return {'p95': p95, 'p99': p99, 'p99_9': p99_9}
        
        try:
            if not self._response_times:
                return {'p95': 0.0, 'p99': 0.0, 'p99_9': 0.0}
            
            # Convert to numpy array for efficient calculation
            times_array = np.array(list(self._response_times))
            
            # Calculate percentiles
            p95 = float(np.percentile(times_array, 95))
            p99 = float(np.percentile(times_array, 99))
            p99_9 = float(np.percentile(times_array, 99.9))
            
            # Cache results
            self._cached_percentiles = (p95, p99, p99_9)
            self._cached_percentiles_timestamp = current_time
            
            return {'p95': p95, 'p99': p99, 'p99_9': p99_9}
            
        except Exception as e:
            self.logger.warning(f"Error calculating response metrics: {e}")
            return {'p95': 0.0, 'p99': 0.0, 'p99_9': 0.0}
    
    def _get_error_metrics_categorized(self) -> Dict[str, Any]:
        """Get error metrics with categorization."""
        try:
            if self._total_requests == 0:
                return {
                    'total_rate': 0.0,
                    'categories': {},
                    'total_errors': 0
                }
            
            total_errors = sum(self._error_counts.values())
            error_rate = (total_errors / self._total_requests) * 100
            
            # Prepare categorized error data
            error_categories = dict(self._error_counts)
            
            # Apply sliding window to prevent unbounded growth
            if self._total_requests > 20000:
                self._total_requests = int(self._total_requests * 0.8)
                for error_type in self._error_counts:
                    self._error_counts[error_type] = int(self._error_counts[error_type] * 0.8)
            
            return {
                'total_rate': min(100.0, error_rate),
                'categories': error_categories,
                'total_errors': total_errors
            }
        except Exception as e:
            self.logger.warning(f"Error calculating error metrics: {e}")
            return {
                'total_rate': 0.0,
                'categories': {},
                'total_errors': 0
            }
    
    def _get_connection_metrics(self) -> Dict[str, int]:
        """Get connection and queue metrics."""
        try:
            # Try to get from production monitoring first
            queue_depth = self._request_queue_depth
            active_count = self._active_connections
            
            if (self.production_monitoring and 
                hasattr(self.production_monitoring, 'get_connection_metrics')):
                prod_metrics = self.production_monitoring.get_connection_metrics()
                if prod_metrics:
                    queue_depth = prod_metrics.get('queue_depth', queue_depth)
                    active_count = prod_metrics.get('active_connections', active_count)
            
            # Fallback to system-level connection counting if needed
            if active_count == 0:
                try:
                    connections = psutil.net_connections(kind='inet')
                    active_count = sum(1 for conn in connections 
                                     if conn.status == psutil.CONN_ESTABLISHED)
                except:
                    pass
            
            return {
                'queue_depth': queue_depth,
                'active_count': active_count
            }
        except Exception as e:
            self.logger.warning(f"Error getting connection metrics: {e}")
            return {
                'queue_depth': 0,
                'active_count': 0
            }
    
    def _calculate_load_level_with_hysteresis(self, metrics: EnhancedSystemLoadMetrics) -> SystemLoadLevel:
        """Calculate load level with hysteresis for stability."""
        if not self.hysteresis_config.enabled:
            return self._calculate_base_load_level(metrics)
        
        base_level = self._calculate_base_load_level(metrics)
        
        # Check if we're trying to move to a lower load level
        if base_level < self.current_load_level:
            # Apply hysteresis - check minimum dwell time
            time_at_current_level = (datetime.now() - self.last_level_change_time).total_seconds()
            if time_at_current_level < self.hysteresis_config.min_dwell_time:
                # Not enough time at current level, stay put
                return self.current_load_level
            
            # Apply hysteresis factor to thresholds
            adjusted_level = self._calculate_load_level_with_hysteresis_factor(
                metrics, self.hysteresis_config.down_factor)
            
            # Only move down if hysteresis-adjusted calculation agrees
            if adjusted_level < self.current_load_level:
                metrics.hysteresis_factor_applied = self.hysteresis_config.down_factor
                return adjusted_level
            else:
                return self.current_load_level
        
        # Moving up or staying same - apply immediately with potential up-factor
        elif base_level > self.current_load_level and self.hysteresis_config.up_factor != 1.0:
            adjusted_level = self._calculate_load_level_with_hysteresis_factor(
                metrics, self.hysteresis_config.up_factor)
            metrics.hysteresis_factor_applied = self.hysteresis_config.up_factor
            return adjusted_level
        
        return base_level
    
    def _calculate_base_load_level(self, metrics: EnhancedSystemLoadMetrics) -> SystemLoadLevel:
        """Calculate base load level without hysteresis."""
        if not GRACEFUL_DEGRADATION_AVAILABLE:
            # Simplified calculation if base system not available
            if metrics.cpu_utilization > 85 or metrics.memory_pressure > 80:
                return SystemLoadLevel.CRITICAL
            elif metrics.cpu_utilization > 75 or metrics.memory_pressure > 75:
                return SystemLoadLevel.HIGH
            elif metrics.cpu_utilization > 60 or metrics.memory_pressure > 65:
                return SystemLoadLevel.ELEVATED
            else:
                return SystemLoadLevel.NORMAL
        
        # Use thresholds to determine levels for each metric
        cpu_level = self._get_metric_level(
            metrics.cpu_utilization,
            [self.thresholds.cpu_normal, self.thresholds.cpu_elevated,
             self.thresholds.cpu_high, self.thresholds.cpu_critical,
             self.thresholds.cpu_emergency]
        )
        
        memory_level = self._get_metric_level(
            metrics.memory_pressure,
            [self.thresholds.memory_normal, self.thresholds.memory_elevated,
             self.thresholds.memory_high, self.thresholds.memory_critical,
             self.thresholds.memory_emergency]
        )
        
        queue_level = self._get_metric_level(
            metrics.request_queue_depth,
            [self.thresholds.queue_normal, self.thresholds.queue_elevated,
             self.thresholds.queue_high, self.thresholds.queue_critical,
             self.thresholds.queue_emergency]
        )
        
        response_level = self._get_metric_level(
            metrics.response_time_p95,
            [self.thresholds.response_p95_normal, self.thresholds.response_p95_elevated,
             self.thresholds.response_p95_high, self.thresholds.response_p95_critical,
             self.thresholds.response_p95_emergency]
        )
        
        error_level = self._get_metric_level(
            metrics.error_rate,
            [self.thresholds.error_rate_normal, self.thresholds.error_rate_elevated,
             self.thresholds.error_rate_high, self.thresholds.error_rate_critical,
             self.thresholds.error_rate_emergency]
        )
        
        # Take the highest (most critical) level
        max_level = max(cpu_level, memory_level, queue_level, response_level, error_level)
        return SystemLoadLevel(max_level)
    
    def _calculate_load_level_with_hysteresis_factor(self, metrics: EnhancedSystemLoadMetrics, factor: float) -> SystemLoadLevel:
        """Calculate load level with hysteresis factor applied to thresholds."""
        if not GRACEFUL_DEGRADATION_AVAILABLE:
            return self._calculate_base_load_level(metrics)
        
        # Apply factor to thresholds
        adjusted_thresholds = self._apply_factor_to_thresholds(self.thresholds, factor)
        
        # Calculate level with adjusted thresholds
        cpu_level = self._get_metric_level(
            metrics.cpu_utilization,
            [adjusted_thresholds.cpu_normal, adjusted_thresholds.cpu_elevated,
             adjusted_thresholds.cpu_high, adjusted_thresholds.cpu_critical,
             adjusted_thresholds.cpu_emergency]
        )
        
        memory_level = self._get_metric_level(
            metrics.memory_pressure,
            [adjusted_thresholds.memory_normal, adjusted_thresholds.memory_elevated,
             adjusted_thresholds.memory_high, adjusted_thresholds.memory_critical,
             adjusted_thresholds.memory_emergency]
        )
        
        queue_level = self._get_metric_level(
            metrics.request_queue_depth,
            [adjusted_thresholds.queue_normal, adjusted_thresholds.queue_elevated,
             adjusted_thresholds.queue_high, adjusted_thresholds.queue_critical,
             adjusted_thresholds.queue_emergency]
        )
        
        response_level = self._get_metric_level(
            metrics.response_time_p95,
            [adjusted_thresholds.response_p95_normal, adjusted_thresholds.response_p95_elevated,
             adjusted_thresholds.response_p95_high, adjusted_thresholds.response_p95_critical,
             adjusted_thresholds.response_p95_emergency]
        )
        
        error_level = self._get_metric_level(
            metrics.error_rate,
            [adjusted_thresholds.error_rate_normal, adjusted_thresholds.error_rate_elevated,
             adjusted_thresholds.error_rate_high, adjusted_thresholds.error_rate_critical,
             adjusted_thresholds.error_rate_emergency]
        )
        
        max_level = max(cpu_level, memory_level, queue_level, response_level, error_level)
        return SystemLoadLevel(max_level)
    
    def _apply_factor_to_thresholds(self, thresholds: LoadThresholds, factor: float) -> LoadThresholds:
        """Apply hysteresis factor to all thresholds."""
        return LoadThresholds(
            cpu_normal=thresholds.cpu_normal * factor,
            cpu_elevated=thresholds.cpu_elevated * factor,
            cpu_high=thresholds.cpu_high * factor,
            cpu_critical=thresholds.cpu_critical * factor,
            cpu_emergency=thresholds.cpu_emergency * factor,
            
            memory_normal=thresholds.memory_normal * factor,
            memory_elevated=thresholds.memory_elevated * factor,
            memory_high=thresholds.memory_high * factor,
            memory_critical=thresholds.memory_critical * factor,
            memory_emergency=thresholds.memory_emergency * factor,
            
            queue_normal=int(thresholds.queue_normal * factor),
            queue_elevated=int(thresholds.queue_elevated * factor),
            queue_high=int(thresholds.queue_high * factor),
            queue_critical=int(thresholds.queue_critical * factor),
            queue_emergency=int(thresholds.queue_emergency * factor),
            
            response_p95_normal=thresholds.response_p95_normal * factor,
            response_p95_elevated=thresholds.response_p95_elevated * factor,
            response_p95_high=thresholds.response_p95_high * factor,
            response_p95_critical=thresholds.response_p95_critical * factor,
            response_p95_emergency=thresholds.response_p95_emergency * factor,
            
            error_rate_normal=thresholds.error_rate_normal * factor,
            error_rate_elevated=thresholds.error_rate_elevated * factor,
            error_rate_high=thresholds.error_rate_high * factor,
            error_rate_critical=thresholds.error_rate_critical * factor,
            error_rate_emergency=thresholds.error_rate_emergency * factor
        )
    
    def _get_metric_level(self, value: float, thresholds: List[float]) -> int:
        """Get the level (0-4) for a metric value against thresholds."""
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                return i
        return len(thresholds) - 1  # Return highest level if above all thresholds
    
    def _calculate_enhanced_load_score(self, metrics: EnhancedSystemLoadMetrics) -> float:
        """Calculate enhanced load score with trend analysis."""
        # Base score calculation with weighted metrics
        cpu_score = min(metrics.cpu_utilization / 100.0, 1.0)
        memory_score = min(metrics.memory_pressure / 100.0, 1.0)
        queue_score = min(metrics.request_queue_depth / 200.0, 1.0)
        response_score = min(metrics.response_time_p95 / 10000.0, 1.0)
        error_score = min(metrics.error_rate / 10.0, 1.0)
        
        # Enhanced weights considering production priorities
        weights = {
            'cpu': 0.25,
            'memory': 0.25,
            'queue': 0.20,
            'response': 0.20,
            'error': 0.10
        }
        
        base_score = (
            cpu_score * weights['cpu'] +
            memory_score * weights['memory'] +
            queue_score * weights['queue'] +
            response_score * weights['response'] +
            error_score * weights['error']
        )
        
        # Apply trend analysis if available
        trend_factor = 1.0
        if self._trend_analyzer and len(self.metrics_history) >= 3:
            trend_indicators = metrics.trend_indicators
            if 'load_trend' in trend_indicators:
                # Increase score if trend is worsening
                trend_factor = 1.0 + max(0, trend_indicators['load_trend'] * 0.15)
        
        enhanced_score = min(1.0, base_score * trend_factor)
        return enhanced_score
    
    async def _sync_with_production_system(self, metrics: EnhancedSystemLoadMetrics):
        """Synchronize with production monitoring system."""
        if not self.production_monitoring:
            return
        
        try:
            # Push enhanced metrics to production system
            if hasattr(self.production_monitoring, 'update_enhanced_system_metrics'):
                await self.production_monitoring.update_enhanced_system_metrics(metrics)
            elif hasattr(self.production_monitoring, 'update_system_metrics'):
                # Fallback to basic metrics
                self.production_monitoring.update_system_metrics(
                    memory_usage_bytes=int(psutil.virtual_memory().used),
                    cpu_usage_percent=metrics.cpu_utilization
                )
            
            # Pull production-specific metrics
            if hasattr(self.production_monitoring, 'get_production_metrics'):
                prod_metrics = await self.production_monitoring.get_production_metrics()
                if prod_metrics:
                    # Update our tracking with production data
                    self._request_queue_depth = prod_metrics.get('queue_depth', self._request_queue_depth)
                    self._active_connections = prod_metrics.get('active_connections', self._active_connections)
                    
        except Exception as e:
            self.logger.debug(f"Production system sync error: {e}")
    
    def _get_safe_default_metrics(self) -> EnhancedSystemLoadMetrics:
        """Get safe default metrics for error conditions."""
        return EnhancedSystemLoadMetrics(
            timestamp=datetime.now(),
            cpu_utilization=0.0,
            memory_pressure=0.0,
            request_queue_depth=0,
            response_time_p95=0.0,
            response_time_p99=0.0,
            error_rate=0.0,
            active_connections=0,
            disk_io_wait=0.0,
            load_level=SystemLoadLevel.NORMAL,
            load_score=0.0,
            degradation_recommended=False
        )
    
    def get_current_metrics(self) -> Optional[EnhancedSystemLoadMetrics]:
        """Get current metrics snapshot."""
        return self.current_metrics
    
    def get_metrics_history(self, count: int = 100) -> List[EnhancedSystemLoadMetrics]:
        """Get recent metrics history."""
        return list(self.metrics_history)[-count:]
    
    def get_load_level_history(self, count: int = 50) -> List[Tuple[datetime, SystemLoadLevel]]:
        """Get load level change history."""
        return list(self.load_level_history)[-count:]
    
    def export_metrics_for_analysis(self) -> Dict[str, Any]:
        """Export metrics data for external analysis."""
        return {
            'current_metrics': self.current_metrics.to_base_metrics() if self.current_metrics else None,
            'current_load_level': self.current_load_level.name,
            'monitoring_interval': self.monitoring_interval,
            'hysteresis_config': {
                'enabled': self.hysteresis_config.enabled,
                'down_factor': self.hysteresis_config.down_factor,
                'up_factor': self.hysteresis_config.up_factor,
                'min_dwell_time': self.hysteresis_config.min_dwell_time
            },
            'metrics_count': len(self.metrics_history),
            'level_changes_count': len(self.load_level_history),
            'last_level_change': self.last_level_change_time.isoformat() if self.last_level_change_time else None
        }


# ============================================================================
# TREND ANALYSIS COMPONENT
# ============================================================================

class TrendAnalyzer:
    """Analyzes trends in system metrics for predictive load management."""
    
    def __init__(self, analysis_window: int = 10):
        self.analysis_window = analysis_window
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, current_metrics: EnhancedSystemLoadMetrics, 
                history: List[EnhancedSystemLoadMetrics]) -> Dict[str, float]:
        """Analyze trends and return trend indicators."""
        if len(history) < 3:
            return {}
        
        try:
            # Get recent history window
            recent_history = history[-self.analysis_window:] if len(history) >= self.analysis_window else history
            
            indicators = {}
            
            # CPU trend analysis
            cpu_values = [m.cpu_utilization for m in recent_history]
            indicators['cpu_trend'] = self._calculate_trend(cpu_values)
            
            # Memory trend analysis
            memory_values = [m.memory_pressure for m in recent_history]
            indicators['memory_trend'] = self._calculate_trend(memory_values)
            
            # Response time trend analysis
            response_values = [m.response_time_p95 for m in recent_history]
            indicators['response_trend'] = self._calculate_trend(response_values)
            
            # Overall load score trend
            load_values = [m.load_score for m in recent_history]
            indicators['load_trend'] = self._calculate_trend(load_values)
            
            # Volatility analysis
            indicators['cpu_volatility'] = self._calculate_volatility(cpu_values)
            indicators['memory_volatility'] = self._calculate_volatility(memory_values)
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f"Error in trend analysis: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, where 1 is strongly increasing)."""
        if len(values) < 2:
            return 0.0
        
        try:
            # Simple linear regression slope
            n = len(values)
            x = list(range(n))
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(values)
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            
            # Normalize slope to -1 to 1 range based on value scale
            value_range = max(values) - min(values) if max(values) != min(values) else 1.0
            normalized_slope = slope / value_range if value_range > 0 else 0.0
            
            return max(-1.0, min(1.0, normalized_slope))
            
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (0 to 1, where 1 is highly volatile)."""
        if len(values) < 2:
            return 0.0
        
        try:
            std_dev = statistics.stdev(values)
            mean_val = statistics.mean(values)
            
            # Coefficient of variation as volatility measure
            if mean_val > 0:
                volatility = std_dev / mean_val
            else:
                volatility = std_dev
            
            # Normalize to 0-1 range (cap at 1.0)
            return min(1.0, volatility)
            
        except Exception:
            return 0.0


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

def create_enhanced_load_monitoring_system(
    thresholds: Optional[LoadThresholds] = None,
    hysteresis_config: Optional[HysteresisConfig] = None,
    monitoring_interval: float = 5.0,
    production_monitoring: Optional[Any] = None,
    enable_trend_analysis: bool = True
) -> EnhancedLoadDetectionSystem:
    """Create a production-ready enhanced load monitoring system."""
    
    return EnhancedLoadDetectionSystem(
        thresholds=thresholds,
        hysteresis_config=hysteresis_config,
        monitoring_interval=monitoring_interval,
        production_monitoring=production_monitoring,
        enable_trend_analysis=enable_trend_analysis
    )


def integrate_with_graceful_degradation_manager(
    degradation_manager: Any,
    enhanced_detector: EnhancedLoadDetectionSystem
) -> bool:
    """Integrate enhanced detector with existing graceful degradation manager."""
    try:
        if hasattr(degradation_manager, 'load_detector'):
            # Replace existing detector with enhanced version
            old_detector = degradation_manager.load_detector
            degradation_manager.load_detector = enhanced_detector
            
            # Transfer callbacks from old detector
            if hasattr(old_detector, '_callbacks'):
                for callback in old_detector._callbacks:
                    enhanced_detector.add_load_change_callback(callback)
            
            return True
    except Exception as e:
        logging.error(f"Failed to integrate enhanced detector: {e}")
        return False


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

async def demonstrate_enhanced_monitoring():
    """Demonstrate the enhanced load monitoring system."""
    print("Enhanced Load Monitoring System Demonstration")
    print("=" * 60)
    
    # Create enhanced monitoring system
    hysteresis_config = HysteresisConfig(
        enabled=True,
        down_factor=0.8,
        min_dwell_time=10.0
    )
    
    enhanced_detector = create_enhanced_load_monitoring_system(
        monitoring_interval=2.0,
        hysteresis_config=hysteresis_config,
        enable_trend_analysis=True
    )
    
    # Add callback to monitor changes
    def on_load_change(metrics: EnhancedSystemLoadMetrics):
        print(f"Load Level: {metrics.load_level.name}")
        print(f"  CPU: {metrics.cpu_utilization:.1f}% (cores: {len(metrics.cpu_per_core)})")
        print(f"  Memory: {metrics.memory_pressure:.1f}% (available: {metrics.memory_available_mb:.0f}MB)")
        print(f"  Queue: {metrics.request_queue_depth} requests")
        print(f"  Response P95: {metrics.response_time_p95:.1f}ms")
        print(f"  Error Rate: {metrics.error_rate:.2f}%")
        print(f"  Load Score: {metrics.load_score:.3f}")
        if metrics.trend_indicators:
            print(f"  CPU Trend: {metrics.trend_indicators.get('cpu_trend', 0):.3f}")
            print(f"  Load Trend: {metrics.trend_indicators.get('load_trend', 0):.3f}")
        print(f"  Hysteresis Factor: {metrics.hysteresis_factor_applied:.2f}")
        print("-" * 40)
    
    enhanced_detector.add_load_change_callback(on_load_change)
    
    # Start monitoring
    await enhanced_detector.start_monitoring()
    print("Enhanced monitoring started. Running for 30 seconds...")
    
    # Simulate some load by recording fake request metrics
    for i in range(10):
        await asyncio.sleep(3)
        # Simulate varying response times and errors
        response_time = 500 + (i * 200)  # Increasing response times
        error_type = "timeout" if i > 5 else None
        enhanced_detector.record_request_metrics(response_time, error_type)
        
        # Update queue depth
        enhanced_detector.update_queue_depth(i * 5)
        enhanced_detector.update_connection_count(10 + i * 2)
    
    await enhanced_detector.stop_monitoring()
    
    # Export final metrics
    export_data = enhanced_detector.export_metrics_for_analysis()
    print("\nFinal System State:")
    print(json.dumps(export_data, indent=2, default=str))


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_enhanced_monitoring())