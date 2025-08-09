"""
Graceful Degradation System for Clinical Metabolomics Oracle

This module implements intelligent load-based degradation strategies that maintain system 
functionality while protecting against overload. It integrates with the existing production
load balancer, circuit breakers, and 5-level fallback hierarchy.

Key Features:
- Progressive degradation based on system load metrics
- Dynamic timeout management with adaptive scaling
- Query complexity reduction strategies
- Resource-aware feature control
- Integration with existing monitoring systems

Architecture:
1. LoadDetectionSystem: Monitors CPU, memory, queue depth, response times
2. DegradationLevelManager: Manages 4 progressive degradation levels
3. TimeoutManager: Dynamically adjusts API timeouts based on load
4. QuerySimplificationEngine: Reduces query complexity under load
5. FeatureToggleController: Disables non-essential features progressively

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
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

# Import existing components for integration
try:
    from .production_load_balancer import ProductionLoadBalancer
    LOAD_BALANCER_AVAILABLE = True
except ImportError:
    LOAD_BALANCER_AVAILABLE = False

try:
    from .production_monitoring import ProductionMonitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from .comprehensive_fallback_system import FallbackLevel, FallbackOrchestrator
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False

try:
    from .clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Check if all components are available
PRODUCTION_COMPONENTS_AVAILABLE = all([
    LOAD_BALANCER_AVAILABLE, MONITORING_AVAILABLE, FALLBACK_AVAILABLE, RAG_AVAILABLE
])

if not PRODUCTION_COMPONENTS_AVAILABLE:
    logging.warning("Some production components not available - using mock interfaces")
    
    # Create mock classes for missing components
    if not LOAD_BALANCER_AVAILABLE:
        class ProductionLoadBalancer:
            def __init__(self, *args, **kwargs):
                self.backend_instances = {}
                
    if not MONITORING_AVAILABLE:
        class ProductionMonitoring:
            def __init__(self, *args, **kwargs):
                pass
                
    if not FALLBACK_AVAILABLE:
        class FallbackOrchestrator:
            def __init__(self, *args, **kwargs):
                self.config = {}
                
    if not RAG_AVAILABLE:
        class ClinicalMetabolomicsRAG:
            def __init__(self, *args, **kwargs):
                self.config = {}


# ============================================================================
# LOAD DETECTION AND THRESHOLDS
# ============================================================================

class SystemLoadLevel(IntEnum):
    """System load levels for progressive degradation."""
    NORMAL = 0      # Normal operation
    ELEVATED = 1    # Slight load increase
    HIGH = 2        # High load - start degradation
    CRITICAL = 3    # Critical load - aggressive degradation
    EMERGENCY = 4   # Emergency - minimal functionality


class LoadMetricType(Enum):
    """Types of load metrics monitored."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_PRESSURE = "memory_pressure"
    REQUEST_QUEUE_DEPTH = "request_queue_depth"
    RESPONSE_TIME_P95 = "response_time_p95"
    RESPONSE_TIME_P99 = "response_time_p99"
    ERROR_RATE = "error_rate"
    ACTIVE_CONNECTIONS = "active_connections"
    DISK_IO_WAIT = "disk_io_wait"


@dataclass
class LoadThresholds:
    """Configurable thresholds for load detection."""
    
    # CPU utilization thresholds (percentage)
    cpu_normal: float = 50.0
    cpu_elevated: float = 65.0
    cpu_high: float = 80.0
    cpu_critical: float = 90.0
    cpu_emergency: float = 95.0
    
    # Memory pressure thresholds (percentage)
    memory_normal: float = 60.0
    memory_elevated: float = 70.0
    memory_high: float = 75.0
    memory_critical: float = 85.0
    memory_emergency: float = 90.0
    
    # Request queue depth thresholds
    queue_normal: int = 10
    queue_elevated: int = 25
    queue_high: int = 50
    queue_critical: int = 100
    queue_emergency: int = 200
    
    # Response time thresholds (milliseconds)
    response_p95_normal: float = 1000.0
    response_p95_elevated: float = 2000.0
    response_p95_high: float = 3000.0
    response_p95_critical: float = 5000.0
    response_p95_emergency: float = 8000.0
    
    # Error rate thresholds (percentage)
    error_rate_normal: float = 0.1
    error_rate_elevated: float = 0.5
    error_rate_high: float = 1.0
    error_rate_critical: float = 2.0
    error_rate_emergency: float = 5.0


@dataclass
class SystemLoadMetrics:
    """Current system load metrics."""
    
    timestamp: datetime
    cpu_utilization: float
    memory_pressure: float
    request_queue_depth: int
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    active_connections: int
    disk_io_wait: float
    
    # Derived metrics
    load_level: SystemLoadLevel = SystemLoadLevel.NORMAL
    load_score: float = 0.0
    degradation_recommended: bool = False


# ============================================================================
# DEGRADATION LEVELS AND STRATEGIES
# ============================================================================

@dataclass
class DegradationLevel:
    """Configuration for a specific degradation level."""
    
    level: SystemLoadLevel
    name: str
    description: str
    
    # Timeout adjustments (multipliers)
    lightrag_timeout_multiplier: float = 1.0
    literature_search_timeout_multiplier: float = 1.0
    openai_timeout_multiplier: float = 1.0
    perplexity_timeout_multiplier: float = 1.0
    
    # Feature toggles
    confidence_analysis_enabled: bool = True
    detailed_logging_enabled: bool = True
    complex_analytics_enabled: bool = True
    confidence_scoring_enabled: bool = True
    query_preprocessing_enabled: bool = True
    fallback_hierarchy_full: bool = True
    
    # Query simplification settings
    max_query_complexity: int = 100
    use_simplified_prompts: bool = False
    skip_context_enrichment: bool = False
    reduce_response_detail: bool = False
    
    # Resource limits
    max_concurrent_requests: int = 100
    max_memory_per_request_mb: int = 100
    batch_size_limit: int = 10


class DegradationStrategy:
    """Defines degradation strategies for different load levels."""
    
    def __init__(self):
        self.levels = {
            SystemLoadLevel.NORMAL: DegradationLevel(
                level=SystemLoadLevel.NORMAL,
                name="Normal Operation",
                description="Full functionality with all features enabled",
                lightrag_timeout_multiplier=1.0,
                literature_search_timeout_multiplier=1.0,
                openai_timeout_multiplier=1.0,
                perplexity_timeout_multiplier=1.0,
                confidence_analysis_enabled=True,
                detailed_logging_enabled=True,
                complex_analytics_enabled=True,
                confidence_scoring_enabled=True,
                query_preprocessing_enabled=True,
                fallback_hierarchy_full=True,
                max_query_complexity=100,
                max_concurrent_requests=100,
                max_memory_per_request_mb=100,
                batch_size_limit=10
            ),
            
            SystemLoadLevel.ELEVATED: DegradationLevel(
                level=SystemLoadLevel.ELEVATED,
                name="Elevated Load",
                description="Minor optimizations to reduce resource usage",
                lightrag_timeout_multiplier=0.9,
                literature_search_timeout_multiplier=0.9,
                openai_timeout_multiplier=0.95,
                perplexity_timeout_multiplier=0.95,
                confidence_analysis_enabled=True,
                detailed_logging_enabled=True,
                complex_analytics_enabled=True,
                confidence_scoring_enabled=True,
                query_preprocessing_enabled=True,
                fallback_hierarchy_full=True,
                max_query_complexity=90,
                max_concurrent_requests=80,
                max_memory_per_request_mb=80,
                batch_size_limit=8
            ),
            
            SystemLoadLevel.HIGH: DegradationLevel(
                level=SystemLoadLevel.HIGH,
                name="High Load Degradation",
                description="Significant performance optimizations",
                lightrag_timeout_multiplier=0.75,
                literature_search_timeout_multiplier=0.7,
                openai_timeout_multiplier=0.85,
                perplexity_timeout_multiplier=0.8,
                confidence_analysis_enabled=True,
                detailed_logging_enabled=False,  # Disable detailed logging
                complex_analytics_enabled=False,  # Disable complex analytics
                confidence_scoring_enabled=True,
                query_preprocessing_enabled=False,  # Skip preprocessing
                fallback_hierarchy_full=True,
                max_query_complexity=70,
                use_simplified_prompts=True,
                max_concurrent_requests=60,
                max_memory_per_request_mb=60,
                batch_size_limit=5
            ),
            
            SystemLoadLevel.CRITICAL: DegradationLevel(
                level=SystemLoadLevel.CRITICAL,
                name="Critical Load Protection",
                description="Aggressive degradation to maintain core functionality",
                lightrag_timeout_multiplier=0.5,
                literature_search_timeout_multiplier=0.5,
                openai_timeout_multiplier=0.7,
                perplexity_timeout_multiplier=0.6,
                confidence_analysis_enabled=False,  # Disable confidence analysis
                detailed_logging_enabled=False,
                complex_analytics_enabled=False,
                confidence_scoring_enabled=False,  # Disable detailed scoring
                query_preprocessing_enabled=False,
                fallback_hierarchy_full=False,  # Simplified fallback
                max_query_complexity=50,
                use_simplified_prompts=True,
                skip_context_enrichment=True,
                reduce_response_detail=True,
                max_concurrent_requests=30,
                max_memory_per_request_mb=40,
                batch_size_limit=3
            ),
            
            SystemLoadLevel.EMERGENCY: DegradationLevel(
                level=SystemLoadLevel.EMERGENCY,
                name="Emergency Mode",
                description="Minimal functionality to prevent system failure",
                lightrag_timeout_multiplier=0.3,
                literature_search_timeout_multiplier=0.3,
                openai_timeout_multiplier=0.5,
                perplexity_timeout_multiplier=0.4,
                confidence_analysis_enabled=False,
                detailed_logging_enabled=False,
                complex_analytics_enabled=False,
                confidence_scoring_enabled=False,
                query_preprocessing_enabled=False,
                fallback_hierarchy_full=False,
                max_query_complexity=30,
                use_simplified_prompts=True,
                skip_context_enrichment=True,
                reduce_response_detail=True,
                max_concurrent_requests=10,
                max_memory_per_request_mb=20,
                batch_size_limit=1
            )
        }


# ============================================================================
# LOAD DETECTION SYSTEM
# ============================================================================

class LoadDetectionSystem:
    """Monitors system metrics and determines appropriate load levels."""
    
    def __init__(self, thresholds: Optional[LoadThresholds] = None,
                 monitoring_interval: float = 5.0):
        self.thresholds = thresholds or LoadThresholds()
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
        # Metrics history for trend analysis
        self.metrics_history: deque = deque(maxlen=100)
        self.load_level_history: deque = deque(maxlen=20)
        
        # Current state
        self.current_metrics: Optional[SystemLoadMetrics] = None
        self.current_load_level: SystemLoadLevel = SystemLoadLevel.NORMAL
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[SystemLoadMetrics], None]] = []
        
        # Request queue simulation (would integrate with actual request handler)
        self._request_queue_depth = 0
        self._response_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._total_requests = 0
        
    def add_load_change_callback(self, callback: Callable[[SystemLoadMetrics], None]):
        """Add callback for load level changes."""
        self._callbacks.append(callback)
    
    def get_system_metrics(self) -> SystemLoadMetrics:
        """Collect current system metrics."""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_pressure = memory.percent
            
            # Disk I/O wait time
            try:
                disk_io = psutil.disk_io_counters()
                disk_io_wait = getattr(disk_io, 'read_time', 0) + getattr(disk_io, 'write_time', 0)
            except:
                disk_io_wait = 0.0
            
            # Response time calculations
            response_p95 = 0.0
            response_p99 = 0.0
            if self._response_times:
                sorted_times = sorted(self._response_times)
                p95_idx = int(len(sorted_times) * 0.95)
                p99_idx = int(len(sorted_times) * 0.99)
                response_p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0
                response_p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0
            
            # Error rate calculation
            error_rate = 0.0
            if self._total_requests > 0:
                error_rate = (self._error_count / self._total_requests) * 100
            
            # Create metrics object
            metrics = SystemLoadMetrics(
                timestamp=datetime.now(),
                cpu_utilization=cpu_percent,
                memory_pressure=memory_pressure,
                request_queue_depth=self._request_queue_depth,
                response_time_p95=response_p95,
                response_time_p99=response_p99,
                error_rate=error_rate,
                active_connections=0,  # Would integrate with actual connection counter
                disk_io_wait=disk_io_wait
            )
            
            # Determine load level and score
            metrics.load_level = self._calculate_load_level(metrics)
            metrics.load_score = self._calculate_load_score(metrics)
            metrics.degradation_recommended = metrics.load_level > SystemLoadLevel.NORMAL
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return safe defaults
            return SystemLoadMetrics(
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
    
    def _calculate_load_level(self, metrics: SystemLoadMetrics) -> SystemLoadLevel:
        """Calculate load level based on multiple metrics."""
        # Check each metric against thresholds
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
        
        # Use the highest level (most severe)
        max_level = max(cpu_level, memory_level, queue_level, response_level, error_level)
        
        # Apply hysteresis to prevent rapid oscillations
        if hasattr(self, 'current_load_level'):
            if max_level > self.current_load_level:
                # Load increasing - immediate response
                return SystemLoadLevel(max_level)
            elif max_level < self.current_load_level:
                # Load decreasing - wait for confirmation
                if len(self.load_level_history) >= 3:
                    recent_levels = list(self.load_level_history)[-3:]
                    if all(level <= max_level for level in recent_levels):
                        return SystemLoadLevel(max_level)
                    else:
                        return self.current_load_level
                else:
                    return self.current_load_level
        
        return SystemLoadLevel(max_level)
    
    def _get_metric_level(self, value: float, thresholds: List[float]) -> int:
        """Get load level for a specific metric."""
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                return i
        return len(thresholds)  # Emergency level
    
    def _calculate_load_score(self, metrics: SystemLoadMetrics) -> float:
        """Calculate overall load score (0.0 to 1.0)."""
        # Normalize each metric to 0-1 scale
        cpu_score = min(metrics.cpu_utilization / 100.0, 1.0)
        memory_score = min(metrics.memory_pressure / 100.0, 1.0)
        queue_score = min(metrics.request_queue_depth / 200.0, 1.0)
        response_score = min(metrics.response_time_p95 / 10000.0, 1.0)
        error_score = min(metrics.error_rate / 10.0, 1.0)
        
        # Weighted average (CPU and memory are most important)
        weights = {
            'cpu': 0.3,
            'memory': 0.3,
            'queue': 0.2,
            'response': 0.15,
            'error': 0.05
        }
        
        load_score = (
            cpu_score * weights['cpu'] +
            memory_score * weights['memory'] +
            queue_score * weights['queue'] +
            response_score * weights['response'] +
            error_score * weights['error']
        )
        
        return min(load_score, 1.0)
    
    async def start_monitoring(self):
        """Start continuous load monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Load monitoring started")
    
    async def stop_monitoring(self):
        """Stop load monitoring."""
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Load monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect metrics
                metrics = self.get_system_metrics()
                self.current_metrics = metrics
                
                # Check for load level changes
                previous_level = self.current_load_level
                self.current_load_level = metrics.load_level
                
                # Store in history
                self.metrics_history.append(metrics)
                self.load_level_history.append(metrics.load_level)
                
                # Notify callbacks if load level changed
                if previous_level != self.current_load_level:
                    self.logger.info(
                        f"Load level changed: {previous_level.name} -> {self.current_load_level.name}"
                    )
                    for callback in self._callbacks:
                        try:
                            callback(metrics)
                        except Exception as e:
                            self.logger.error(f"Error in load change callback: {e}")
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def record_request_start(self):
        """Record request start for queue depth tracking."""
        self._request_queue_depth += 1
        self._total_requests += 1
    
    def record_request_complete(self, response_time_ms: float, error: bool = False):
        """Record request completion."""
        self._request_queue_depth = max(0, self._request_queue_depth - 1)
        self._response_times.append(response_time_ms)
        if error:
            self._error_count += 1


# ============================================================================
# TIMEOUT MANAGEMENT SYSTEM
# ============================================================================

class TimeoutManager:
    """Manages dynamic timeout adjustments based on system load."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.degradation_strategy = DegradationStrategy()
        
        # Base timeouts (seconds) - matching existing system values
        self.base_timeouts = {
            'lightrag_query': 60.0,
            'literature_search': 90.0,
            'openai_api': 45.0,
            'perplexity_api': 35.0,
            'embedding_api': 30.0,
            'general_api': 30.0
        }
        
        # Current adjusted timeouts
        self.current_timeouts = self.base_timeouts.copy()
        self.current_load_level = SystemLoadLevel.NORMAL
    
    def update_timeouts_for_load_level(self, load_level: SystemLoadLevel):
        """Update timeouts based on current load level."""
        if load_level == self.current_load_level:
            return  # No change needed
        
        previous_level = self.current_load_level
        self.current_load_level = load_level
        
        # Get degradation configuration
        degradation_config = self.degradation_strategy.levels[load_level]
        
        # Calculate new timeouts
        self.current_timeouts = {
            'lightrag_query': self.base_timeouts['lightrag_query'] * degradation_config.lightrag_timeout_multiplier,
            'literature_search': self.base_timeouts['literature_search'] * degradation_config.literature_search_timeout_multiplier,
            'openai_api': self.base_timeouts['openai_api'] * degradation_config.openai_timeout_multiplier,
            'perplexity_api': self.base_timeouts['perplexity_api'] * degradation_config.perplexity_timeout_multiplier,
            'embedding_api': self.base_timeouts['embedding_api'] * degradation_config.openai_timeout_multiplier,
            'general_api': self.base_timeouts['general_api'] * degradation_config.openai_timeout_multiplier
        }
        
        self.logger.info(
            f"Timeouts adjusted for load level {load_level.name}: "
            f"LightRAG: {self.current_timeouts['lightrag_query']:.1f}s, "
            f"Literature: {self.current_timeouts['literature_search']:.1f}s, "
            f"OpenAI: {self.current_timeouts['openai_api']:.1f}s, "
            f"Perplexity: {self.current_timeouts['perplexity_api']:.1f}s"
        )
    
    def get_timeout(self, service: str) -> float:
        """Get current timeout for a specific service."""
        return self.current_timeouts.get(service, self.current_timeouts['general_api'])
    
    def get_all_timeouts(self) -> Dict[str, float]:
        """Get all current timeouts."""
        return self.current_timeouts.copy()


# ============================================================================
# QUERY SIMPLIFICATION ENGINE
# ============================================================================

class QuerySimplificationEngine:
    """Simplifies queries based on load levels to reduce processing complexity."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.degradation_strategy = DegradationStrategy()
        self.current_load_level = SystemLoadLevel.NORMAL
    
    def update_load_level(self, load_level: SystemLoadLevel):
        """Update current load level for query processing."""
        self.current_load_level = load_level
    
    def simplify_query_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify query parameters based on current load level."""
        degradation_config = self.degradation_strategy.levels[self.current_load_level]
        simplified_params = query_params.copy()
        
        if self.current_load_level >= SystemLoadLevel.HIGH:
            # High load simplifications
            if 'max_total_tokens' in simplified_params:
                # Reduce token limits
                original_tokens = simplified_params['max_total_tokens']
                simplified_params['max_total_tokens'] = min(
                    original_tokens,
                    int(original_tokens * 0.7)
                )
            
            if 'top_k' in simplified_params:
                # Reduce retrieval depth
                original_top_k = simplified_params['top_k']
                simplified_params['top_k'] = max(3, int(original_top_k * 0.6))
            
            # Use simpler response types
            if 'response_type' in simplified_params:
                simplified_params['response_type'] = "Short Answer"
        
        if self.current_load_level >= SystemLoadLevel.CRITICAL:
            # Critical load simplifications
            if 'max_total_tokens' in simplified_params:
                simplified_params['max_total_tokens'] = min(
                    simplified_params['max_total_tokens'], 2000
                )
            
            if 'top_k' in simplified_params:
                simplified_params['top_k'] = min(simplified_params['top_k'], 3)
            
            # Force minimal response
            simplified_params['response_type'] = "Short Answer"
            
            # Skip complex analysis modes
            if simplified_params.get('mode') == 'hybrid':
                simplified_params['mode'] = 'local'
        
        if self.current_load_level >= SystemLoadLevel.EMERGENCY:
            # Emergency mode - minimal processing
            simplified_params.update({
                'max_total_tokens': 1000,
                'top_k': 1,
                'response_type': "Short Answer",
                'mode': 'local'
            })
        
        return simplified_params
    
    def should_skip_feature(self, feature: str) -> bool:
        """Determine if a feature should be skipped based on load level."""
        degradation_config = self.degradation_strategy.levels[self.current_load_level]
        
        feature_toggles = {
            'confidence_analysis': degradation_config.confidence_analysis_enabled,
            'detailed_logging': degradation_config.detailed_logging_enabled,
            'complex_analytics': degradation_config.complex_analytics_enabled,
            'confidence_scoring': degradation_config.confidence_scoring_enabled,
            'query_preprocessing': degradation_config.query_preprocessing_enabled,
            'context_enrichment': not degradation_config.skip_context_enrichment,
            'detailed_response': not degradation_config.reduce_response_detail
        }
        
        return not feature_toggles.get(feature, True)
    
    def get_max_query_complexity(self) -> int:
        """Get maximum allowed query complexity for current load level."""
        degradation_config = self.degradation_strategy.levels[self.current_load_level]
        return degradation_config.max_query_complexity
    
    def should_use_simplified_prompts(self) -> bool:
        """Determine if simplified prompts should be used."""
        degradation_config = self.degradation_strategy.levels[self.current_load_level]
        return degradation_config.use_simplified_prompts


# ============================================================================
# FEATURE TOGGLE CONTROLLER
# ============================================================================

class FeatureToggleController:
    """Controls feature availability based on system load."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.degradation_strategy = DegradationStrategy()
        self.current_load_level = SystemLoadLevel.NORMAL
        
        # Feature state tracking
        self.feature_states: Dict[str, bool] = {}
        self.feature_change_callbacks: Dict[str, List[Callable[[bool], None]]] = defaultdict(list)
    
    def update_load_level(self, load_level: SystemLoadLevel):
        """Update load level and adjust feature availability."""
        if load_level == self.current_load_level:
            return
        
        previous_level = self.current_load_level
        self.current_load_level = load_level
        
        # Get new feature states
        degradation_config = self.degradation_strategy.levels[load_level]
        new_feature_states = {
            'confidence_analysis': degradation_config.confidence_analysis_enabled,
            'detailed_logging': degradation_config.detailed_logging_enabled,
            'complex_analytics': degradation_config.complex_analytics_enabled,
            'confidence_scoring': degradation_config.confidence_scoring_enabled,
            'query_preprocessing': degradation_config.query_preprocessing_enabled,
            'full_fallback_hierarchy': degradation_config.fallback_hierarchy_full,
            'context_enrichment': not degradation_config.skip_context_enrichment,
            'detailed_response': not degradation_config.reduce_response_detail
        }
        
        # Check for changes and notify callbacks
        for feature, new_state in new_feature_states.items():
            old_state = self.feature_states.get(feature, True)
            if old_state != new_state:
                self.feature_states[feature] = new_state
                self.logger.info(f"Feature '{feature}' {'enabled' if new_state else 'disabled'} for load level {load_level.name}")
                
                # Notify callbacks
                for callback in self.feature_change_callbacks[feature]:
                    try:
                        callback(new_state)
                    except Exception as e:
                        self.logger.error(f"Error in feature toggle callback for {feature}: {e}")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is currently enabled."""
        return self.feature_states.get(feature, True)
    
    def add_feature_callback(self, feature: str, callback: Callable[[bool], None]):
        """Add callback for feature state changes."""
        self.feature_change_callbacks[feature].append(callback)
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get current resource limits based on load level."""
        degradation_config = self.degradation_strategy.levels[self.current_load_level]
        return {
            'max_concurrent_requests': degradation_config.max_concurrent_requests,
            'max_memory_per_request_mb': degradation_config.max_memory_per_request_mb,
            'batch_size_limit': degradation_config.batch_size_limit
        }


# ============================================================================
# MAIN GRACEFUL DEGRADATION MANAGER
# ============================================================================

class GracefulDegradationManager:
    """Main orchestrator for graceful degradation system."""
    
    def __init__(self, 
                 load_thresholds: Optional[LoadThresholds] = None,
                 monitoring_interval: float = 5.0):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.load_detector = LoadDetectionSystem(load_thresholds, monitoring_interval)
        self.timeout_manager = TimeoutManager()
        self.query_simplifier = QuerySimplificationEngine()
        self.feature_controller = FeatureToggleController()
        
        # State tracking
        self.current_load_level = SystemLoadLevel.NORMAL
        self.degradation_active = False
        
        # Integration components
        self.production_load_balancer: Optional[Any] = None
        self.fallback_orchestrator: Optional[Any] = None
        self.clinical_rag: Optional[Any] = None
        
        # Callbacks for external integration
        self.load_change_callbacks: List[Callable[[SystemLoadLevel, SystemLoadMetrics], None]] = []
        
        # Register load change handler
        self.load_detector.add_load_change_callback(self._handle_load_change)
    
    def integrate_with_production_system(self,
                                       production_load_balancer: Optional[Any] = None,
                                       fallback_orchestrator: Optional[Any] = None,
                                       clinical_rag: Optional[Any] = None):
        """Integrate with existing production components."""
        self.production_load_balancer = production_load_balancer
        self.fallback_orchestrator = fallback_orchestrator
        self.clinical_rag = clinical_rag
        
        self.logger.info("Graceful degradation integrated with production components")
    
    def add_load_change_callback(self, callback: Callable[[SystemLoadLevel, SystemLoadMetrics], None]):
        """Add callback for load level changes."""
        self.load_change_callbacks.append(callback)
    
    async def start(self):
        """Start the graceful degradation system."""
        await self.load_detector.start_monitoring()
        self.logger.info("Graceful degradation system started")
    
    async def stop(self):
        """Stop the graceful degradation system."""
        await self.load_detector.stop_monitoring()
        self.logger.info("Graceful degradation system stopped")
    
    def _handle_load_change(self, metrics: SystemLoadMetrics):
        """Handle load level changes."""
        new_level = metrics.load_level
        previous_level = self.current_load_level
        self.current_load_level = new_level
        
        # Update all components
        self.timeout_manager.update_timeouts_for_load_level(new_level)
        self.query_simplifier.update_load_level(new_level)
        self.feature_controller.update_load_level(new_level)
        
        # Update degradation status
        self.degradation_active = new_level > SystemLoadLevel.NORMAL
        
        # Log the change
        if new_level != previous_level:
            self.logger.warning(
                f"System load level changed: {previous_level.name} -> {new_level.name} "
                f"(Load Score: {metrics.load_score:.3f}, CPU: {metrics.cpu_utilization:.1f}%, "
                f"Memory: {metrics.memory_pressure:.1f}%, Queue: {metrics.request_queue_depth})"
            )
        
        # Notify external callbacks
        for callback in self.load_change_callbacks:
            try:
                callback(new_level, metrics)
            except Exception as e:
                self.logger.error(f"Error in load change callback: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        current_metrics = self.load_detector.current_metrics
        return {
            'load_level': self.current_load_level.name,
            'load_score': current_metrics.load_score if current_metrics else 0.0,
            'degradation_active': self.degradation_active,
            'current_timeouts': self.timeout_manager.get_all_timeouts(),
            'resource_limits': self.feature_controller.get_resource_limits(),
            'metrics': {
                'cpu_utilization': current_metrics.cpu_utilization if current_metrics else 0.0,
                'memory_pressure': current_metrics.memory_pressure if current_metrics else 0.0,
                'request_queue_depth': current_metrics.request_queue_depth if current_metrics else 0,
                'response_time_p95': current_metrics.response_time_p95 if current_metrics else 0.0,
                'error_rate': current_metrics.error_rate if current_metrics else 0.0
            } if current_metrics else {}
        }
    
    def should_accept_request(self) -> bool:
        """Determine if new requests should be accepted."""
        if self.current_load_level >= SystemLoadLevel.EMERGENCY:
            # In emergency mode, only accept critical requests
            return False
        return True
    
    def get_timeout_for_service(self, service: str) -> float:
        """Get current timeout for a service."""
        return self.timeout_manager.get_timeout(service)
    
    def simplify_query_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify query parameters based on current load."""
        return self.query_simplifier.simplify_query_params(query_params)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled under current load."""
        return self.feature_controller.is_feature_enabled(feature)
    
    def record_request_metrics(self, response_time_ms: float, error: bool = False):
        """Record request metrics for load calculation."""
        self.load_detector.record_request_complete(response_time_ms, error)


# ============================================================================
# FACTORY AND CONFIGURATION FUNCTIONS
# ============================================================================

def create_production_degradation_system(
    load_thresholds: Optional[LoadThresholds] = None,
    monitoring_interval: float = 5.0,
    production_load_balancer: Optional[Any] = None,
    fallback_orchestrator: Optional[Any] = None,
    clinical_rag: Optional[Any] = None
) -> GracefulDegradationManager:
    """Create a production-ready graceful degradation system."""
    
    # Use production-optimized thresholds if none provided
    if load_thresholds is None:
        load_thresholds = LoadThresholds(
            # More aggressive thresholds for production
            cpu_high=75.0,
            cpu_critical=85.0,
            cpu_emergency=92.0,
            memory_high=70.0,
            memory_critical=80.0,
            memory_emergency=87.0,
            queue_high=30,
            queue_critical=75,
            queue_emergency=150,
            response_p95_high=2500.0,
            response_p95_critical=4000.0,
            response_p95_emergency=6000.0
        )
    
    # Create the manager
    manager = GracefulDegradationManager(load_thresholds, monitoring_interval)
    
    # Integrate with production components
    manager.integrate_with_production_system(
        production_load_balancer,
        fallback_orchestrator,
        clinical_rag
    )
    
    return manager


def create_development_degradation_system(
    monitoring_interval: float = 10.0
) -> GracefulDegradationManager:
    """Create a development-friendly graceful degradation system."""
    
    # More relaxed thresholds for development
    dev_thresholds = LoadThresholds(
        cpu_high=85.0,
        cpu_critical=92.0,
        cpu_emergency=97.0,
        memory_high=80.0,
        memory_critical=88.0,
        memory_emergency=95.0,
        queue_high=50,
        queue_critical=100,
        queue_emergency=200,
        response_p95_high=5000.0,
        response_p95_critical=8000.0,
        response_p95_emergency=12000.0
    )
    
    return GracefulDegradationManager(dev_thresholds, monitoring_interval)


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

async def demo_graceful_degradation():
    """Demonstration of the graceful degradation system."""
    
    # Create degradation manager
    degradation_manager = create_production_degradation_system()
    
    # Add callback to monitor load changes
    def on_load_change(load_level: SystemLoadLevel, metrics: SystemLoadMetrics):
        print(f"Load level changed to {load_level.name}")
        print(f"CPU: {metrics.cpu_utilization:.1f}%, Memory: {metrics.memory_pressure:.1f}%")
        print(f"Current timeouts: {degradation_manager.timeout_manager.get_all_timeouts()}")
        print("---")
    
    degradation_manager.add_load_change_callback(on_load_change)
    
    try:
        # Start monitoring
        await degradation_manager.start()
        print("Graceful degradation system started. Monitoring for 60 seconds...")
        
        # Simulate various load conditions
        await asyncio.sleep(60)
        
    finally:
        # Stop monitoring
        await degradation_manager.stop()
        print("Graceful degradation system stopped.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_graceful_degradation())