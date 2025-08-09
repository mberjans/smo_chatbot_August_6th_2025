"""
Comprehensive Performance Metrics Collection System for CMO Concurrent Load Tests
==================================================================================

This module provides advanced performance metrics collection, analysis, and integration
for the Clinical Metabolomics Oracle concurrent load testing framework. It builds upon
the existing concurrent testing infrastructure with sophisticated metrics collection,
real-time monitoring, and CMO-specific performance analysis.

Key Features:
1. **Advanced Metrics Collection**: Response time percentiles, throughput analysis, 
   success/failure categorization, resource usage with growth rate analysis
2. **Real-time Monitoring**: 100ms sampling intervals for live performance tracking
3. **CMO-Specific Metrics**: LightRAG performance, multi-tier cache effectiveness,
   circuit breaker analysis, fallback system usage tracking
4. **Performance Analytics**: Automated grading (A-F), trend detection, regression analysis
5. **Integration Layer**: Seamless integration with existing concurrent_performance_enhancer.py
6. **Advanced Analysis**: Component integration analysis, resource efficiency assessments

CMO-Specific Performance Targets:
- LightRAG Success Rate: >95% with hybrid mode optimization
- Multi-tier Cache Hit Rates: L1 >80%, L2 >70%, L3 >60%
- Circuit Breaker Threshold: Activate at >20% failure rate, recover <5%
- Fallback Success Chain: LightRAG → Perplexity → Cache with >90% overall success
- Resource Efficiency: <50MB memory growth per 100 concurrent users

Author: Claude Code (Anthropic)
Version: 3.0.0
Created: 2025-08-09
Integration: concurrent_performance_enhancer.py, concurrent_load_framework.py
"""

import asyncio
import logging
import time
import threading
import statistics
import json
import uuid
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from contextlib import asynccontextmanager
from enum import Enum, IntEnum
import concurrent.futures
from pathlib import Path
import tempfile
import gc
import psutil
import os

# Import existing framework components
from .concurrent_load_framework import ConcurrentLoadMetrics, LoadTestConfiguration
from .concurrent_performance_enhancer import (
    ConcurrentResourceMonitor, EnhancedHighPerformanceCache, 
    ConcurrentBiomedicalDataGenerator, create_enhanced_performance_suite
)


# ============================================================================
# ADVANCED PERFORMANCE METRICS CORE
# ============================================================================

class PerformanceGrade(Enum):
    """Performance grading system."""
    A_PLUS = "A+"      # Exceptional (>99% success, <500ms P95)
    A = "A"            # Excellent (>95% success, <1000ms P95)
    B = "B"            # Good (>90% success, <1500ms P95)
    C = "C"            # Fair (>85% success, <2500ms P95)
    D = "D"            # Poor (>75% success, <4000ms P95)
    F = "F"            # Failing (<75% success or >4000ms P95)


class MetricTrendDirection(Enum):
    """Performance trend analysis."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


class ComponentHealthStatus(Enum):
    """Component health status classification."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class LightRAGMetrics:
    """LightRAG-specific performance metrics."""
    
    # Query performance
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    hybrid_mode_queries: int = 0
    naive_mode_queries: int = 0
    local_mode_queries: int = 0
    global_mode_queries: int = 0
    
    # Response quality
    response_times: List[float] = field(default_factory=list)
    token_usage_input: List[int] = field(default_factory=list)
    token_usage_output: List[int] = field(default_factory=list)
    cost_per_query: List[float] = field(default_factory=list)
    
    # Mode effectiveness
    mode_success_rates: Dict[str, float] = field(default_factory=lambda: {
        'hybrid': 0.0, 'naive': 0.0, 'local': 0.0, 'global': 0.0
    })
    mode_response_times: Dict[str, List[float]] = field(default_factory=lambda: {
        'hybrid': [], 'naive': [], 'local': [], 'global': []
    })
    
    # Error categorization
    timeout_errors: int = 0
    api_errors: int = 0
    parsing_errors: int = 0
    cost_limit_errors: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate overall LightRAG success rate."""
        total = self.successful_queries + self.failed_queries
        return (self.successful_queries / total) if total > 0 else 0.0
    
    def get_mode_distribution(self) -> Dict[str, float]:
        """Get distribution of query modes."""
        total = self.total_queries
        if total == 0:
            return {'hybrid': 0.0, 'naive': 0.0, 'local': 0.0, 'global': 0.0}
        
        return {
            'hybrid': self.hybrid_mode_queries / total,
            'naive': self.naive_mode_queries / total,
            'local': self.local_mode_queries / total,
            'global': self.global_mode_queries / total
        }
    
    def get_average_cost(self) -> float:
        """Calculate average cost per query."""
        return statistics.mean(self.cost_per_query) if self.cost_per_query else 0.0
    
    def get_token_efficiency(self) -> Dict[str, float]:
        """Calculate token usage efficiency."""
        if not self.token_usage_input or not self.token_usage_output:
            return {'input_avg': 0.0, 'output_avg': 0.0, 'ratio': 0.0}
        
        input_avg = statistics.mean(self.token_usage_input)
        output_avg = statistics.mean(self.token_usage_output)
        ratio = output_avg / input_avg if input_avg > 0 else 0.0
        
        return {
            'input_avg': input_avg,
            'output_avg': output_avg,
            'ratio': ratio
        }


@dataclass
class MultiTierCacheMetrics:
    """Multi-tier cache performance metrics."""
    
    # L1 Cache (Memory)
    l1_hits: int = 0
    l1_misses: int = 0
    l1_evictions: int = 0
    l1_response_times: List[float] = field(default_factory=list)
    
    # L2 Cache (Extended Memory)
    l2_hits: int = 0
    l2_misses: int = 0
    l2_evictions: int = 0
    l2_response_times: List[float] = field(default_factory=list)
    
    # L3 Cache (Persistent)
    l3_hits: int = 0
    l3_misses: int = 0
    l3_evictions: int = 0
    l3_response_times: List[float] = field(default_factory=list)
    
    # Cache promotions (L3→L2→L1)
    l3_to_l2_promotions: int = 0
    l2_to_l1_promotions: int = 0
    
    # Cache pressure metrics
    cache_pressure_events: int = 0
    memory_pressure_evictions: int = 0
    
    # Access patterns
    sequential_access_patterns: int = 0
    random_access_patterns: int = 0
    hot_key_access_count: Dict[str, int] = field(default_factory=dict)
    
    def get_overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate across all tiers."""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_misses = self.l1_misses + self.l2_misses + self.l3_misses
        total = total_hits + total_misses
        return (total_hits / total) if total > 0 else 0.0
    
    def get_tier_hit_rates(self) -> Dict[str, float]:
        """Calculate hit rates for each cache tier."""
        def _tier_hit_rate(hits: int, misses: int) -> float:
            total = hits + misses
            return (hits / total) if total > 0 else 0.0
        
        return {
            'l1': _tier_hit_rate(self.l1_hits, self.l1_misses),
            'l2': _tier_hit_rate(self.l2_hits, self.l2_misses),
            'l3': _tier_hit_rate(self.l3_hits, self.l3_misses)
        }
    
    def get_average_response_times(self) -> Dict[str, float]:
        """Calculate average response times per cache tier."""
        return {
            'l1': statistics.mean(self.l1_response_times) if self.l1_response_times else 0.0,
            'l2': statistics.mean(self.l2_response_times) if self.l2_response_times else 0.0,
            'l3': statistics.mean(self.l3_response_times) if self.l3_response_times else 0.0
        }
    
    def get_cache_effectiveness_score(self) -> float:
        """Calculate overall cache effectiveness score (0-1)."""
        hit_rates = self.get_tier_hit_rates()
        response_times = self.get_average_response_times()
        
        # Weighted effectiveness based on hit rates and response times
        l1_effectiveness = hit_rates['l1'] * 1.0  # L1 most important
        l2_effectiveness = hit_rates['l2'] * 0.7  # L2 moderate importance
        l3_effectiveness = hit_rates['l3'] * 0.5  # L3 least important
        
        # Penalty for slow response times
        time_penalty = 0.0
        for tier, time_ms in response_times.items():
            if time_ms > 10:  # >10ms is considered slow for cache
                time_penalty += min(0.2, time_ms / 100)  # Cap penalty at 0.2
        
        effectiveness = (l1_effectiveness + l2_effectiveness + l3_effectiveness) / 2.2
        return max(0.0, effectiveness - time_penalty)


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker performance and behavior metrics."""
    
    # State transitions
    closed_to_open_transitions: int = 0
    open_to_half_open_transitions: int = 0
    half_open_to_closed_transitions: int = 0
    half_open_to_open_transitions: int = 0
    
    # Time in each state
    time_in_closed_state: float = 0.0
    time_in_open_state: float = 0.0
    time_in_half_open_state: float = 0.0
    
    # Failure tracking
    total_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    successful_requests: int = 0
    
    # Threshold monitoring
    failure_rate_threshold: float = 0.2  # 20% default
    current_failure_rate: float = 0.0
    threshold_violations: int = 0
    
    # Recovery tracking
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    
    # Cost-based metrics (CMO-specific)
    cost_threshold_activations: int = 0
    total_cost_blocked: float = 0.0
    average_cost_per_request: float = 0.0
    
    # Performance impact
    response_times_before_activation: List[float] = field(default_factory=list)
    response_times_after_recovery: List[float] = field(default_factory=list)
    
    def get_availability_percentage(self) -> float:
        """Calculate system availability percentage."""
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.blocked_requests) / self.total_requests) * 100
    
    def get_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        total_recoveries = self.successful_recoveries + self.failed_recoveries
        return (self.successful_recoveries / total_recoveries) if total_recoveries > 0 else 0.0
    
    def get_state_distribution(self) -> Dict[str, float]:
        """Get time distribution across circuit breaker states."""
        total_time = self.time_in_closed_state + self.time_in_open_state + self.time_in_half_open_state
        if total_time == 0:
            return {'closed': 0.0, 'open': 0.0, 'half_open': 0.0}
        
        return {
            'closed': self.time_in_closed_state / total_time,
            'open': self.time_in_open_state / total_time,
            'half_open': self.time_in_half_open_state / total_time
        }
    
    def is_functioning_properly(self) -> bool:
        """Assess if circuit breaker is functioning as expected."""
        availability = self.get_availability_percentage()
        recovery_rate = self.get_recovery_success_rate()
        
        # Circuit breaker should maintain >90% availability and >80% recovery rate
        return availability >= 90.0 and (recovery_rate >= 0.8 or self.recovery_attempts == 0)


@dataclass
class FallbackSystemMetrics:
    """Fallback system performance metrics for CMO (LightRAG → Perplexity → Cache)."""
    
    # Fallback chain usage
    primary_lightrag_attempts: int = 0
    primary_lightrag_successes: int = 0
    
    fallback_perplexity_attempts: int = 0
    fallback_perplexity_successes: int = 0
    
    fallback_cache_attempts: int = 0
    fallback_cache_successes: int = 0
    
    total_fallback_failures: int = 0
    
    # Response times by fallback level
    lightrag_response_times: List[float] = field(default_factory=list)
    perplexity_response_times: List[float] = field(default_factory=list)
    cache_response_times: List[float] = field(default_factory=list)
    
    # Cost tracking
    lightrag_costs: List[float] = field(default_factory=list)
    perplexity_costs: List[float] = field(default_factory=list)
    cache_costs: List[float] = field(default_factory=list)  # Usually 0
    
    # Quality degradation tracking
    response_quality_scores: Dict[str, List[float]] = field(default_factory=lambda: {
        'lightrag': [], 'perplexity': [], 'cache': []
    })
    
    def get_fallback_success_rates(self) -> Dict[str, float]:
        """Calculate success rates for each fallback level."""
        def _success_rate(successes: int, attempts: int) -> float:
            return (successes / attempts) if attempts > 0 else 0.0
        
        return {
            'lightrag': _success_rate(self.primary_lightrag_successes, self.primary_lightrag_attempts),
            'perplexity': _success_rate(self.fallback_perplexity_successes, self.fallback_perplexity_attempts),
            'cache': _success_rate(self.fallback_cache_successes, self.fallback_cache_attempts)
        }
    
    def get_overall_success_rate(self) -> float:
        """Calculate overall success rate of the fallback system."""
        total_attempts = self.primary_lightrag_attempts
        if total_attempts == 0:
            return 0.0
        
        total_successes = (self.primary_lightrag_successes + 
                          self.fallback_perplexity_successes + 
                          self.fallback_cache_successes)
        
        return total_successes / total_attempts
    
    def get_average_costs(self) -> Dict[str, float]:
        """Calculate average costs for each fallback level."""
        return {
            'lightrag': statistics.mean(self.lightrag_costs) if self.lightrag_costs else 0.0,
            'perplexity': statistics.mean(self.perplexity_costs) if self.perplexity_costs else 0.0,
            'cache': statistics.mean(self.cache_costs) if self.cache_costs else 0.0
        }
    
    def get_cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score (higher success rate, lower cost = better)."""
        success_rate = self.get_overall_success_rate()
        costs = self.get_average_costs()
        
        # Weight by usage - LightRAG successes are most cost-effective
        total_cost = (costs['lightrag'] * self.primary_lightrag_successes +
                     costs['perplexity'] * self.fallback_perplexity_successes)
        
        total_successes = self.primary_lightrag_successes + self.fallback_perplexity_successes
        avg_cost = total_cost / total_successes if total_successes > 0 else 0.0
        
        # Score: high success rate with low average cost
        if avg_cost == 0:
            return success_rate
        else:
            return success_rate / (1 + avg_cost)  # Penalize high costs


# ============================================================================
# ENHANCED CMO LOAD METRICS
# ============================================================================

class CMOLoadMetrics(ConcurrentLoadMetrics):
    """Enhanced load metrics specifically designed for CMO system performance analysis."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # CMO-specific metrics
        self.lightrag_metrics = LightRAGMetrics()
        self.cache_metrics = MultiTierCacheMetrics()
        self.circuit_breaker_metrics = CircuitBreakerMetrics()
        self.fallback_metrics = FallbackSystemMetrics()
        
        # Advanced performance tracking
        self.performance_samples: deque = deque(maxlen=10000)  # 100ms samples
        self.trend_analysis: Dict[str, List[float]] = defaultdict(list)
        self.component_health_status: Dict[str, ComponentHealthStatus] = {}
        
        # Resource efficiency tracking
        self.resource_efficiency_samples: List[float] = []
        self.memory_growth_rate: List[float] = []
        self.cpu_efficiency: List[float] = []
        
        # Performance grading
        self.current_grade: PerformanceGrade = PerformanceGrade.F
        self.grade_history: List[Tuple[datetime, PerformanceGrade]] = []
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 0.1  # 100ms
        self._last_sample_time = 0.0
        
        # Thread safety for real-time updates
        self._metrics_lock = asyncio.Lock()
        self._sample_buffer: deque = deque(maxlen=1000)


# ============================================================================
# REAL-TIME METRICS COLLECTION SYSTEM
# ============================================================================

@dataclass
class MetricSample:
    """Individual metric sample with timestamp."""
    timestamp: datetime
    metric_name: str
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


class RealTimeMetricsCollector:
    """Real-time metrics collector for concurrent load testing."""
    
    def __init__(self, config: CMOTestConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_metrics: Dict[str, float] = {}
        
        # Collection state
        self.collection_active = False
        self.collection_interval = 0.1  # 100ms sampling
        self.collection_tasks: List[asyncio.Task] = []
        
        # Thread safety
        self.metrics_lock = asyncio.Lock()
        
        # Metric definitions
        self.metric_definitions = {
            # Response time metrics
            'response_time_p50': {'unit': 'ms', 'category': 'performance'},
            'response_time_p95': {'unit': 'ms', 'category': 'performance'},
            'response_time_p99': {'unit': 'ms', 'category': 'performance'},
            
            # Throughput metrics
            'requests_per_second': {'unit': 'ops/sec', 'category': 'throughput'},
            'successful_requests_per_second': {'unit': 'ops/sec', 'category': 'throughput'},
            'failed_requests_per_second': {'unit': 'ops/sec', 'category': 'throughput'},
            
            # System resource metrics
            'cpu_usage_percent': {'unit': '%', 'category': 'resources'},
            'memory_usage_mb': {'unit': 'MB', 'category': 'resources'},
            'memory_growth_rate': {'unit': 'MB/min', 'category': 'resources'},
            
            # CMO-specific metrics
            'lightrag_success_rate': {'unit': '%', 'category': 'cmo'},
            'lightrag_avg_response_time': {'unit': 'ms', 'category': 'cmo'},
            'cache_hit_rate': {'unit': '%', 'category': 'cmo'},
            'cache_l1_hit_rate': {'unit': '%', 'category': 'cmo'},
            'cache_l2_hit_rate': {'unit': '%', 'category': 'cmo'},
            'cache_l3_hit_rate': {'unit': '%', 'category': 'cmo'},
            
            # Circuit breaker metrics
            'circuit_breaker_state': {'unit': 'state', 'category': 'reliability'},
            'circuit_breaker_failure_rate': {'unit': '%', 'category': 'reliability'},
            
            # Fallback system metrics
            'fallback_activation_rate': {'unit': '%', 'category': 'reliability'},
            'fallback_success_rate': {'unit': '%', 'category': 'reliability'},
            
            # Concurrent user metrics
            'active_concurrent_users': {'unit': 'count', 'category': 'concurrency'},
            'queued_requests': {'unit': 'count', 'category': 'concurrency'},
            'connection_pool_usage': {'unit': '%', 'category': 'concurrency'}
        }
    
    async def start_collection(self):
        """Start real-time metrics collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        
        # Start collection tasks
        self.collection_tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_performance_metrics()),
            asyncio.create_task(self._collect_cmo_specific_metrics()),
            asyncio.create_task(self._calculate_derived_metrics())
        ]
        
        self.logger.info("Real-time metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        # Cancel and wait for collection tasks
        for task in self.collection_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.collection_tasks.clear()
        self.logger.info("Real-time metrics collection stopped")
    
    async def record_metric(self, metric_name: str, value: float, 
                          context: Optional[Dict[str, Any]] = None,
                          tags: Optional[Dict[str, str]] = None):
        """Record a metric sample."""
        sample = MetricSample(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            context=context or {},
            tags=tags or {}
        )
        
        async with self.metrics_lock:
            self.metrics_buffer[metric_name].append(sample)
            self.current_metrics[metric_name] = value
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        process = psutil.Process(os.getpid())
        last_memory = 0
        last_time = time.time()
        
        while self.collection_active:
            try:
                # CPU and memory usage
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                await self.record_metric('cpu_usage_percent', cpu_percent)
                await self.record_metric('memory_usage_mb', memory_mb)
                
                # Calculate memory growth rate
                current_time = time.time()
                if last_memory > 0 and current_time > last_time:
                    time_delta = (current_time - last_time) / 60  # Convert to minutes
                    memory_delta = memory_mb - last_memory
                    growth_rate = memory_delta / time_delta if time_delta > 0 else 0
                    await self.record_metric('memory_growth_rate', growth_rate)
                
                last_memory = memory_mb
                last_time = current_time
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics from active tests."""
        while self.collection_active:
            try:
                # This would be called by the test framework to update performance metrics
                # The actual collection happens in the test execution context
                
                await asyncio.sleep(self.collection_interval * 5)  # Less frequent updates
                
            except Exception as e:
                self.logger.error(f"Error collecting performance metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_cmo_specific_metrics(self):
        """Collect CMO component-specific metrics."""
        while self.collection_active:
            try:
                # These would be updated by the CMO components during test execution
                # The collector provides the framework for receiving these updates
                
                await asyncio.sleep(self.collection_interval * 2)  # Moderate frequency
                
            except Exception as e:
                self.logger.error(f"Error collecting CMO metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _calculate_derived_metrics(self):
        """Calculate derived metrics from collected data."""
        while self.collection_active:
            try:
                async with self.metrics_lock:
                    # Calculate percentiles for response times if we have data
                    if 'response_times' in self.metrics_buffer:
                        response_times = [s.value for s in list(self.metrics_buffer['response_times'])[-100:]]  # Last 100 samples
                        if response_times:
                            await self._update_percentiles(response_times)
                    
                    # Calculate rates and ratios
                    await self._calculate_throughput_metrics()
                    await self._calculate_success_rates()
                
                await asyncio.sleep(self.collection_interval * 10)  # Every second
                
            except Exception as e:
                self.logger.error(f"Error calculating derived metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _update_percentiles(self, response_times: List[float]):
        """Update response time percentiles."""
        if len(response_times) >= 10:  # Need minimum samples
            p50 = np.percentile(response_times, 50)
            p95 = np.percentile(response_times, 95)
            p99 = np.percentile(response_times, 99)
            
            await self.record_metric('response_time_p50', p50)
            await self.record_metric('response_time_p95', p95)
            await self.record_metric('response_time_p99', p99)
    
    async def _calculate_throughput_metrics(self):
        """Calculate throughput metrics."""
        # This would calculate ops/sec based on recent request counts
        # Implementation depends on how request counts are tracked
        pass
    
    async def _calculate_success_rates(self):
        """Calculate various success rates."""
        # This would calculate success rates for different components
        # Implementation depends on how success/failure counts are tracked
        pass
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return self.current_metrics.copy()
    
    def get_metric_history(self, metric_name: str, time_window_seconds: int = 60) -> List[MetricSample]:
        """Get metric history for a specific time window."""
        if metric_name not in self.metrics_buffer:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
        return [sample for sample in self.metrics_buffer[metric_name] 
                if sample.timestamp >= cutoff_time]
    
    def generate_metrics_summary(self) -> Dict[str, Any]:
        """Generate comprehensive metrics summary."""
        summary = {
            'collection_period': {
                'start_time': None,
                'end_time': datetime.now(),
                'duration_seconds': 0
            },
            'metrics_collected': {},
            'performance_indicators': {},
            'resource_utilization': {},
            'cmo_specific_metrics': {}
        }
        
        # Find collection period
        all_samples = []
        for metric_buffer in self.metrics_buffer.values():
            all_samples.extend(metric_buffer)
        
        if all_samples:
            summary['collection_period']['start_time'] = min(s.timestamp for s in all_samples)
            summary['collection_period']['duration_seconds'] = (
                summary['collection_period']['end_time'] - 
                summary['collection_period']['start_time']
            ).total_seconds()
        
        # Summarize each metric
        for metric_name, samples in self.metrics_buffer.items():
            if samples:
                values = [s.value for s in samples]
                summary['metrics_collected'][metric_name] = {
                    'sample_count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stddev': statistics.stdev(values) if len(values) > 1 else 0,
                    'last_value': values[-1]
                }
        
        # Categorize metrics
        for metric_name, metric_info in summary['metrics_collected'].items():
            category = self.metric_definitions.get(metric_name, {}).get('category', 'other')
            
            if category == 'performance':
                summary['performance_indicators'][metric_name] = metric_info
            elif category == 'resources':
                summary['resource_utilization'][metric_name] = metric_info
            elif category == 'cmo':
                summary['cmo_specific_metrics'][metric_name] = metric_info
        
        return summary


# ============================================================================
# ADVANCED PERFORMANCE ANALYTICS
# ============================================================================

class CMOPerformanceAnalyzer:
    """Advanced analytics for CMO load test performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_functions = {
            'response_time_analysis': self._analyze_response_times,
            'throughput_analysis': self._analyze_throughput,
            'resource_efficiency': self._analyze_resource_efficiency,
            'cmo_component_analysis': self._analyze_cmo_components,
            'concurrent_user_impact': self._analyze_concurrent_user_impact,
            'performance_correlation': self._analyze_performance_correlations
        }
    
    async def analyze_test_results(self, 
                                 test_results: Dict[str, CMOLoadMetrics],
                                 metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Perform comprehensive analysis of test results."""
        
        analysis_results = {
            'test_summary': self._create_test_summary(test_results),
            'performance_analysis': {},
            'component_analysis': {},
            'recommendations': [],
            'performance_grade': 'Unknown'
        }
        
        # Run all analysis functions
        for analysis_name, analysis_func in self.analysis_functions.items():
            try:
                analysis_results['performance_analysis'][analysis_name] = await analysis_func(
                    test_results, metrics_collector
                )
            except Exception as e:
                self.logger.error(f"Error in {analysis_name}: {e}")
        
        # Generate recommendations
        analysis_results['recommendations'] = await self._generate_recommendations(
            test_results, analysis_results['performance_analysis']
        )
        
        # Assign overall performance grade
        analysis_results['performance_grade'] = self._calculate_performance_grade(
            test_results, analysis_results['performance_analysis']
        )
        
        return analysis_results
    
    def _create_test_summary(self, test_results: Dict[str, CMOLoadMetrics]) -> Dict[str, Any]:
        """Create high-level test summary."""
        if not test_results:
            return {}
        
        total_operations = sum(m.total_operations for m in test_results.values())
        total_successes = sum(m.successful_operations for m in test_results.values())
        total_failures = sum(m.failed_operations for m in test_results.values())
        
        return {
            'total_tests': len(test_results),
            'total_operations': total_operations,
            'overall_success_rate': total_successes / (total_successes + total_failures) if (total_successes + total_failures) > 0 else 0,
            'peak_concurrent_users': max(m.concurrent_peak for m in test_results.values()) if test_results else 0,
            'test_duration_range': {
                'min_seconds': min((m.end_time - m.start_time).total_seconds() for m in test_results.values() if m.end_time),
                'max_seconds': max((m.end_time - m.start_time).total_seconds() for m in test_results.values() if m.end_time)
            } if any(m.end_time for m in test_results.values()) else {}
        }
    
    async def _analyze_response_times(self, 
                                    test_results: Dict[str, CMOLoadMetrics],
                                    metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Analyze response time patterns."""
        response_time_analysis = {
            'overall_statistics': {},
            'percentile_analysis': {},
            'trend_analysis': {},
            'outlier_analysis': {}
        }
        
        # Collect all response times
        all_response_times = []
        for metrics in test_results.values():
            all_response_times.extend(metrics.response_times)
        
        if not all_response_times:
            return response_time_analysis
        
        # Overall statistics
        response_time_analysis['overall_statistics'] = {
            'count': len(all_response_times),
            'mean': statistics.mean(all_response_times),
            'median': statistics.median(all_response_times),
            'std_dev': statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0,
            'min': min(all_response_times),
            'max': max(all_response_times)
        }
        
        # Percentile analysis
        if len(all_response_times) >= 20:
            percentiles = [50, 75, 90, 95, 99, 99.9]
            response_time_analysis['percentile_analysis'] = {
                f'p{p}': np.percentile(all_response_times, p) for p in percentiles
            }
        
        # Trend analysis using time series data from metrics collector
        p95_history = metrics_collector.get_metric_history('response_time_p95', 300)  # Last 5 minutes
        if len(p95_history) >= 10:
            values = [sample.value for sample in p95_history]
            response_time_analysis['trend_analysis'] = {
                'trend_direction': 'improving' if values[-1] < values[0] else 'degrading' if values[-1] > values[0] else 'stable',
                'trend_magnitude': abs(values[-1] - values[0]) / values[0] if values[0] > 0 else 0,
                'volatility': statistics.stdev(values) / statistics.mean(values) if values and statistics.mean(values) > 0 else 0
            }
        
        # Outlier analysis
        mean_rt = response_time_analysis['overall_statistics']['mean']
        std_rt = response_time_analysis['overall_statistics']['std_dev']
        outliers = [rt for rt in all_response_times if abs(rt - mean_rt) > 3 * std_rt]
        
        response_time_analysis['outlier_analysis'] = {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(all_response_times) * 100,
            'outlier_impact': sum(outliers) / sum(all_response_times) * 100 if all_response_times else 0
        }
        
        return response_time_analysis
    
    async def _analyze_throughput(self, 
                                test_results: Dict[str, CMOLoadMetrics],
                                metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Analyze throughput patterns and scalability."""
        throughput_analysis = {
            'peak_throughput': 0,
            'average_throughput': 0,
            'throughput_efficiency': 0,
            'scalability_metrics': {}
        }
        
        # Calculate peak and average throughput
        throughput_values = []
        for metrics in test_results.values():
            throughput_values.extend(metrics.throughput_samples)
        
        if throughput_values:
            throughput_analysis['peak_throughput'] = max(throughput_values)
            throughput_analysis['average_throughput'] = statistics.mean(throughput_values)
            
            # Throughput efficiency (how consistent is the throughput)
            if len(throughput_values) > 1:
                cv = statistics.stdev(throughput_values) / statistics.mean(throughput_values)
                throughput_analysis['throughput_efficiency'] = max(0, 1 - cv)  # Lower coefficient of variation = higher efficiency
        
        # Scalability analysis
        user_throughput_correlation = []
        for test_name, metrics in test_results.items():
            if metrics.throughput_samples and metrics.concurrent_peak > 0:
                avg_throughput = statistics.mean(metrics.throughput_samples)
                user_throughput_correlation.append((metrics.concurrent_peak, avg_throughput))
        
        if len(user_throughput_correlation) >= 2:
            users = [x[0] for x in user_throughput_correlation]
            throughputs = [x[1] for x in user_throughput_correlation]
            
            # Calculate correlation coefficient
            if len(users) > 1:
                correlation = np.corrcoef(users, throughputs)[0, 1] if not np.isnan(np.corrcoef(users, throughputs)[0, 1]) else 0
                throughput_analysis['scalability_metrics'] = {
                    'user_throughput_correlation': correlation,
                    'scalability_rating': 'Excellent' if correlation > 0.8 else 'Good' if correlation > 0.5 else 'Fair' if correlation > 0.2 else 'Poor'
                }
        
        return throughput_analysis
    
    async def _analyze_resource_efficiency(self, 
                                         test_results: Dict[str, CMOLoadMetrics],
                                         metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Analyze resource utilization efficiency."""
        resource_analysis = {
            'memory_efficiency': {},
            'cpu_efficiency': {},
            'resource_scaling': {}
        }
        
        # Memory efficiency analysis
        memory_samples = []
        for metrics in test_results.values():
            memory_samples.extend(metrics.memory_samples)
        
        if memory_samples:
            memory_growth = max(memory_samples) - min(memory_samples)
            total_operations = sum(m.total_operations for m in test_results.values())
            
            resource_analysis['memory_efficiency'] = {
                'memory_growth_mb': memory_growth,
                'memory_per_operation_kb': (memory_growth * 1024) / total_operations if total_operations > 0 else 0,
                'memory_efficiency_grade': 'Excellent' if memory_growth < 50 else 'Good' if memory_growth < 100 else 'Fair' if memory_growth < 200 else 'Poor'
            }
        
        # CPU efficiency analysis
        cpu_samples = []
        for metrics in test_results.values():
            cpu_samples.extend(metrics.cpu_samples)
        
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            resource_analysis['cpu_efficiency'] = {
                'average_cpu_percent': avg_cpu,
                'peak_cpu_percent': max_cpu,
                'cpu_efficiency_grade': 'Excellent' if avg_cpu < 50 else 'Good' if avg_cpu < 70 else 'Fair' if avg_cpu < 85 else 'Poor'
            }
        
        return resource_analysis
    
    async def _analyze_cmo_components(self, 
                                    test_results: Dict[str, CMOLoadMetrics],
                                    metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Analyze CMO-specific component performance."""
        cmo_analysis = {
            'lightrag_performance': {},
            'cache_effectiveness': {},
            'circuit_breaker_analysis': {},
            'fallback_system_analysis': {}
        }
        
        # Aggregate CMO metrics
        for metrics in test_results.values():
            if isinstance(metrics, CMOLoadMetrics):
                # LightRAG analysis
                if metrics.lightrag_queries > 0:
                    cmo_analysis['lightrag_performance'] = {
                        'total_queries': metrics.lightrag_queries,
                        'success_rate': metrics.get_lightrag_success_rate(),
                        'average_response_time': statistics.mean(metrics.lightrag_response_times) if metrics.lightrag_response_times else 0,
                        'performance_grade': 'Excellent' if metrics.get_lightrag_success_rate() > 0.95 else 'Good' if metrics.get_lightrag_success_rate() > 0.90 else 'Fair'
                    }
                
                # Multi-tier cache analysis
                cache_analysis = metrics.get_multi_tier_cache_analysis()
                if cache_analysis:
                    cmo_analysis['cache_effectiveness'] = cache_analysis
                
                # Fallback analysis
                fallback_analysis = metrics.get_fallback_effectiveness()
                if fallback_analysis:
                    cmo_analysis['fallback_system_analysis'] = fallback_analysis
        
        return cmo_analysis
    
    async def _analyze_concurrent_user_impact(self, 
                                            test_results: Dict[str, CMOLoadMetrics],
                                            metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Analyze how concurrent users impact performance."""
        concurrency_analysis = {
            'concurrency_scaling': {},
            'performance_degradation': {},
            'optimal_user_count': None
        }
        
        # Analyze performance vs. user count relationship
        user_performance_data = []
        for test_name, metrics in test_results.values():
            if metrics.end_time and metrics.response_times:
                avg_response_time = statistics.mean(metrics.response_times)
                success_rate = metrics.get_success_rate()
                user_performance_data.append({
                    'concurrent_users': metrics.concurrent_peak,
                    'avg_response_time': avg_response_time,
                    'success_rate': success_rate,
                    'test_name': test_name
                })
        
        if len(user_performance_data) >= 2:
            # Sort by user count
            user_performance_data.sort(key=lambda x: x['concurrent_users'])
            
            # Calculate degradation
            baseline = user_performance_data[0]
            degradation_data = []
            
            for data in user_performance_data[1:]:
                response_time_degradation = (data['avg_response_time'] - baseline['avg_response_time']) / baseline['avg_response_time']
                success_rate_degradation = baseline['success_rate'] - data['success_rate']
                
                degradation_data.append({
                    'user_count': data['concurrent_users'],
                    'response_time_degradation': response_time_degradation,
                    'success_rate_degradation': success_rate_degradation
                })
            
            concurrency_analysis['performance_degradation'] = degradation_data
            
            # Find optimal user count (balance between throughput and performance)
            optimal_candidates = [
                data for data in user_performance_data
                if data['success_rate'] >= 0.90 and data['avg_response_time'] <= 3000  # 3 second threshold
            ]
            
            if optimal_candidates:
                optimal = max(optimal_candidates, key=lambda x: x['concurrent_users'])
                concurrency_analysis['optimal_user_count'] = optimal['concurrent_users']
        
        return concurrency_analysis
    
    async def _analyze_performance_correlations(self, 
                                              test_results: Dict[str, CMOLoadMetrics],
                                              metrics_collector: RealTimeMetricsCollector) -> Dict[str, Any]:
        """Analyze correlations between different performance metrics."""
        correlation_analysis = {
            'strong_correlations': [],
            'interesting_patterns': [],
            'performance_drivers': []
        }
        
        # This would perform correlation analysis between various metrics
        # For now, return placeholder structure
        
        return correlation_analysis
    
    async def _generate_recommendations(self, 
                                      test_results: Dict[str, CMOLoadMetrics],
                                      performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Response time recommendations
        response_analysis = performance_analysis.get('response_time_analysis', {})
        overall_stats = response_analysis.get('overall_statistics', {})
        
        if overall_stats.get('mean', 0) > 2000:
            recommendations.append(
                "Average response time exceeds 2 seconds. Consider optimizing query processing, "
                "implementing more aggressive caching, or scaling infrastructure."
            )
        
        # Throughput recommendations
        throughput_analysis = performance_analysis.get('throughput_analysis', {})
        scalability_metrics = throughput_analysis.get('scalability_metrics', {})
        
        if scalability_metrics.get('scalability_rating') in ['Fair', 'Poor']:
            recommendations.append(
                "System shows poor scalability characteristics. Investigate bottlenecks in "
                "concurrent processing, database connections, or resource contention."
            )
        
        # Resource efficiency recommendations
        resource_analysis = performance_analysis.get('resource_efficiency', {})
        memory_efficiency = resource_analysis.get('memory_efficiency', {})
        
        if memory_efficiency.get('memory_efficiency_grade') in ['Fair', 'Poor']:
            recommendations.append(
                "Memory usage shows concerning growth patterns. Review memory management, "
                "implement garbage collection tuning, or investigate potential memory leaks."
            )
        
        # CMO-specific recommendations
        cmo_analysis = performance_analysis.get('cmo_component_analysis', {})
        lightrag_perf = cmo_analysis.get('lightrag_performance', {})
        
        if lightrag_perf.get('success_rate', 1.0) < 0.90:
            recommendations.append(
                "LightRAG success rate below 90%. Review error handling, timeout configurations, "
                "and resource allocation for RAG processing components."
            )
        
        cache_effectiveness = cmo_analysis.get('cache_effectiveness', {})
        if cache_effectiveness.get('overall_hit_rate', 1.0) < 0.70:
            recommendations.append(
                "Cache hit rate below 70%. Optimize cache key strategies, review TTL settings, "
                "and consider increasing cache size or implementing smarter eviction policies."
            )
        
        return recommendations
    
    def _calculate_performance_grade(self, 
                                   test_results: Dict[str, CMOLoadMetrics],
                                   performance_analysis: Dict[str, Any]) -> str:
        """Calculate overall performance grade."""
        
        # Scoring factors
        scores = []
        
        # Success rate score
        test_summary = performance_analysis.get('test_summary', {})
        success_rate = test_summary.get('overall_success_rate', 0)
        success_score = min(success_rate * 100, 100)  # 0-100 scale
        scores.append(('success_rate', success_score, 0.3))  # 30% weight
        
        # Response time score
        response_analysis = performance_analysis.get('response_time_analysis', {})
        percentiles = response_analysis.get('percentile_analysis', {})
        p95 = percentiles.get('p95', float('inf'))
        
        # Score P95 response time (lower is better)
        if p95 <= 1000:
            response_score = 100
        elif p95 <= 2000:
            response_score = 80
        elif p95 <= 3000:
            response_score = 60
        elif p95 <= 5000:
            response_score = 40
        else:
            response_score = 20
        
        scores.append(('response_time', response_score, 0.25))  # 25% weight
        
        # Resource efficiency score
        resource_analysis = performance_analysis.get('resource_efficiency', {})
        memory_grade = resource_analysis.get('memory_efficiency', {}).get('memory_efficiency_grade', 'Poor')
        cpu_grade = resource_analysis.get('cpu_efficiency', {}).get('cpu_efficiency_grade', 'Poor')
        
        grade_scores = {'Excellent': 100, 'Good': 80, 'Fair': 60, 'Poor': 40}
        resource_score = (grade_scores[memory_grade] + grade_scores[cpu_grade]) / 2
        scores.append(('resource_efficiency', resource_score, 0.2))  # 20% weight
        
        # CMO components score
        cmo_analysis = performance_analysis.get('cmo_component_analysis', {})
        lightrag_perf = cmo_analysis.get('lightrag_performance', {})
        lightrag_grade = lightrag_perf.get('performance_grade', 'Fair')
        
        cmo_score = grade_scores.get(lightrag_grade, 60)
        scores.append(('cmo_components', cmo_score, 0.15))  # 15% weight
        
        # Scalability score
        throughput_analysis = performance_analysis.get('throughput_analysis', {})
        scalability_rating = throughput_analysis.get('scalability_metrics', {}).get('scalability_rating', 'Fair')
        scalability_score = grade_scores.get(scalability_rating, 60)
        scores.append(('scalability', scalability_score, 0.1))  # 10% weight
        
        # Calculate weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        # Convert to grade
        if weighted_score >= 90:
            return 'A (Excellent)'
        elif weighted_score >= 80:
            return 'B (Good)'
        elif weighted_score >= 70:
            return 'C (Satisfactory)'
        elif weighted_score >= 60:
            return 'D (Needs Improvement)'
        else:
            return 'F (Poor)'


# ============================================================================
# INTEGRATION WITH EXISTING FRAMEWORK
# ============================================================================

class CMOFrameworkIntegrator:
    """Integrates CMO testing with existing performance framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Existing framework components
        self.enhanced_performance_suite = None
        self.concurrent_resource_monitor = None
        self.performance_regression_detector = None
        
        # CMO-specific components
        self.cmo_metrics_collector = None
        self.cmo_performance_analyzer = None
    
    async def initialize_integration(self, cmo_config: CMOTestConfiguration):
        """Initialize integration with existing framework."""
        try:
            # Initialize enhanced performance suite from existing framework
            self.enhanced_performance_suite = create_enhanced_performance_suite(enable_monitoring=True)
            
            # Initialize concurrent resource monitor from existing framework
            self.concurrent_resource_monitor = ConcurrentResourceMonitor()
            await self.concurrent_resource_monitor.start_monitoring()
            
            # Initialize performance regression detector
            self.performance_regression_detector = ConcurrentPerformanceRegression()
            
            # Initialize CMO-specific components
            self.cmo_metrics_collector = RealTimeMetricsCollector(cmo_config)
            await self.cmo_metrics_collector.start_collection()
            
            self.cmo_performance_analyzer = CMOPerformanceAnalyzer()
            
            self.logger.info("CMO framework integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CMO framework integration: {e}")
            raise
    
    async def cleanup_integration(self):
        """Clean up integration components."""
        try:
            if self.cmo_metrics_collector:
                await self.cmo_metrics_collector.stop_collection()
            
            if self.concurrent_resource_monitor:
                await self.concurrent_resource_monitor.stop_monitoring()
            
            if self.enhanced_performance_suite and 'enhanced_cache' in self.enhanced_performance_suite:
                cache_system = self.enhanced_performance_suite['enhanced_cache']
                if hasattr(cache_system, 'base_cache') and hasattr(cache_system.base_cache, 'clear'):
                    await cache_system.base_cache.clear()
            
            self.logger.info("CMO framework integration cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during CMO integration cleanup: {e}")
    
    async def run_integrated_analysis(self, 
                                    test_results: Dict[str, CMOLoadMetrics]) -> Dict[str, Any]:
        """Run comprehensive analysis using both existing and CMO-specific frameworks."""
        
        integrated_analysis = {
            'cmo_specific_analysis': {},
            'existing_framework_analysis': {},
            'integrated_recommendations': [],
            'framework_comparison': {}
        }
        
        try:
            # Run CMO-specific analysis
            if self.cmo_performance_analyzer and self.cmo_metrics_collector:
                integrated_analysis['cmo_specific_analysis'] = await self.cmo_performance_analyzer.analyze_test_results(
                    test_results, self.cmo_metrics_collector
                )
            
            # Run existing framework analysis
            if self.enhanced_performance_suite and self.performance_regression_detector:
                # Convert CMO metrics to format expected by existing framework
                base_results = self._convert_cmo_to_base_format(test_results)
                
                # Run enhanced performance analysis from existing framework
                from .concurrent_performance_enhancer import run_enhanced_performance_analysis
                integrated_analysis['existing_framework_analysis'] = await run_enhanced_performance_analysis(
                    base_results, self.enhanced_performance_suite
                )
                
                # Run regression detection
                regression_analysis = self.performance_regression_detector.detect_regressions(base_results)
                integrated_analysis['existing_framework_analysis']['regression_analysis'] = regression_analysis
            
            # Generate integrated recommendations
            integrated_analysis['integrated_recommendations'] = self._generate_integrated_recommendations(
                integrated_analysis['cmo_specific_analysis'],
                integrated_analysis['existing_framework_analysis']
            )
            
            # Compare framework analyses
            integrated_analysis['framework_comparison'] = self._compare_framework_analyses(
                integrated_analysis['cmo_specific_analysis'],
                integrated_analysis['existing_framework_analysis']
            )
            
        except Exception as e:
            self.logger.error(f"Error running integrated analysis: {e}")
            raise
        
        return integrated_analysis
    
    def _convert_cmo_to_base_format(self, test_results: Dict[str, CMOLoadMetrics]) -> Dict[str, Any]:
        """Convert CMO metrics to base framework format."""
        base_results = {
            'individual_results': {}
        }
        
        for test_name, metrics in test_results.items():
            base_results['individual_results'][test_name] = {
                'basic_metrics': {
                    'success_rate': metrics.get_success_rate(),
                    'total_operations': metrics.total_operations,
                    'concurrent_peak': metrics.concurrent_peak,
                    'test_duration_seconds': (metrics.end_time - metrics.start_time).total_seconds() if metrics.end_time else 0
                },
                'performance_metrics': {
                    'percentiles': metrics.get_percentiles(),
                    'average_throughput': metrics.get_average_throughput(),
                    'cache_hit_rate': metrics.get_cache_hit_rate()
                },
                'resource_metrics': {
                    'memory_min_mb': min(metrics.memory_samples) if metrics.memory_samples else 0,
                    'memory_max_mb': max(metrics.memory_samples) if metrics.memory_samples else 0,
                    'memory_growth_mb': (max(metrics.memory_samples) - min(metrics.memory_samples)) if metrics.memory_samples else 0,
                    'cpu_avg_percent': statistics.mean(metrics.cpu_samples) if metrics.cpu_samples else 0
                }
            }
        
        return base_results
    
    def _generate_integrated_recommendations(self, 
                                           cmo_analysis: Dict[str, Any],
                                           existing_analysis: Dict[str, Any]) -> List[str]:
        """Generate integrated recommendations from both analyses."""
        recommendations = []
        
        # Get recommendations from both analyses
        cmo_recommendations = cmo_analysis.get('recommendations', [])
        existing_recommendations = existing_analysis.get('recommendations', [])
        
        # Combine and prioritize recommendations
        all_recommendations = cmo_recommendations + existing_recommendations
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        
        # Add integration-specific recommendations
        if len(unique_recommendations) > 10:
            recommendations.append(
                "Multiple performance issues identified across different analysis frameworks. "
                "Prioritize addressing high-impact issues first: response times, success rates, and resource usage."
            )
        
        return unique_recommendations[:15]  # Limit to top 15 recommendations
    
    def _compare_framework_analyses(self, 
                                  cmo_analysis: Dict[str, Any],
                                  existing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from different analysis frameworks."""
        comparison = {
            'agreement_score': 0.0,
            'conflicting_insights': [],
            'complementary_insights': [],
            'framework_strengths': {
                'cmo_specific': [],
                'existing_framework': []
            }
        }
        
        # This would perform a detailed comparison between the two analysis results
        # For now, return basic structure
        
        cmo_grade = cmo_analysis.get('performance_grade', 'Unknown')
        existing_recommendations_count = len(existing_analysis.get('recommendations', []))
        
        comparison['framework_strengths']['cmo_specific'] = [
            'Component-specific analysis for LightRAG and caching',
            'Multi-tier cache effectiveness analysis',
            'Fallback system performance evaluation'
        ]
        
        comparison['framework_strengths']['existing_framework'] = [
            'Comprehensive regression detection',
            'Advanced concurrent resource monitoring',
            'Established performance benchmarking'
        ]
        
        return comparison


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def create_integrated_cmo_testing_environment(config: CMOTestConfiguration) -> CMOFrameworkIntegrator:
    """Create a complete integrated CMO testing environment."""
    
    integrator = CMOFrameworkIntegrator()
    await integrator.initialize_integration(config)
    
    return integrator


def export_metrics_to_json(metrics_collector: RealTimeMetricsCollector, 
                          output_file: str):
    """Export collected metrics to JSON file."""
    
    metrics_summary = metrics_collector.generate_metrics_summary()
    
    with open(output_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    
    logging.getLogger(__name__).info(f"Metrics exported to {output_file}")


def create_performance_dashboard_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create data structure suitable for performance dashboard visualization."""
    
    dashboard_data = {
        'summary_cards': {},
        'time_series_data': {},
        'comparison_charts': {},
        'recommendation_list': []
    }
    
    # Extract summary card data
    cmo_analysis = analysis_results.get('cmo_specific_analysis', {})
    test_summary = cmo_analysis.get('test_summary', {})
    
    dashboard_data['summary_cards'] = {
        'overall_success_rate': test_summary.get('overall_success_rate', 0) * 100,
        'peak_concurrent_users': test_summary.get('peak_concurrent_users', 0),
        'total_operations': test_summary.get('total_operations', 0),
        'performance_grade': cmo_analysis.get('performance_grade', 'Unknown')
    }
    
    # Extract recommendations
    dashboard_data['recommendation_list'] = analysis_results.get('integrated_recommendations', [])
    
    return dashboard_data


if __name__ == "__main__":
    # Example usage
    async def main():
        from .cmo_test_configurations import CMOTestConfigurationFactory
        
        # Create test configuration
        config = CMOTestConfigurationFactory.create_basic_concurrent_config(users=25)
        
        # Create integrated testing environment
        integrator = await create_integrated_cmo_testing_environment(config)
        
        try:
            # Simulate some test results
            mock_metrics = CMOLoadMetrics(
                test_name="demo_test",
                start_time=datetime.now(),
                total_users=25
            )
            mock_metrics.end_time = datetime.now()
            mock_metrics.total_operations = 100
            mock_metrics.successful_operations = 95
            mock_metrics.failed_operations = 5
            mock_metrics.response_times = [0.5, 0.8, 1.2, 0.9, 1.1] * 20  # Mock response times
            
            test_results = {'demo_test': mock_metrics}
            
            # Run integrated analysis
            analysis_results = await integrator.run_integrated_analysis(test_results)
            
            print("Integrated CMO Analysis Results:")
            print(json.dumps(analysis_results, indent=2, default=str))
            
        finally:
            await integrator.cleanup_integration()
    
    asyncio.run(main())