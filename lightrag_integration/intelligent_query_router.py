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
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import asyncio
import statistics
from contextlib import asynccontextmanager
import random

from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction, ConfidenceMetrics
from .research_categorizer import ResearchCategorizer, CategoryPrediction
from .cost_persistence import ResearchCategory


class SystemHealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class BackendType(Enum):
    """Backend service types"""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"


@dataclass
class BackendHealthMetrics:
    """Health metrics for a backend service"""
    backend_type: BackendType
    status: SystemHealthStatus
    response_time_ms: float
    error_rate: float
    last_health_check: datetime
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'backend_type': self.backend_type.value,
            'status': self.status.value,
            'response_time_ms': self.response_time_ms,
            'error_rate': self.error_rate,
            'last_health_check': self.last_health_check.isoformat(),
            'consecutive_failures': self.consecutive_failures,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests
        }


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
class LoadBalancingConfig:
    """Configuration for load balancing"""
    strategy: str = "weighted_round_robin"  # "round_robin", "weighted", "health_aware"
    health_check_interval: int = 60  # seconds
    circuit_breaker_threshold: int = 5  # consecutive failures
    circuit_breaker_timeout: int = 300  # seconds
    response_time_threshold_ms: float = 2000.0
    enable_adaptive_routing: bool = True


class SystemHealthMonitor:
    """System health monitoring for routing decisions"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.backend_health: Dict[BackendType, BackendHealthMetrics] = {}
        self.health_history: deque = deque(maxlen=100)
        self.monitoring_active = False
        self.logger = logging.getLogger(__name__)
        
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
        for backend_type in BackendType:
            try:
                start_time = time.perf_counter()
                
                # Mock health check (in real implementation, would ping actual services)
                is_healthy = self._mock_backend_health_check(backend_type)
                
                response_time = (time.perf_counter() - start_time) * 1000
                
                metrics = self.backend_health[backend_type]
                metrics.last_health_check = datetime.now()
                metrics.response_time_ms = response_time
                
                if is_healthy:
                    metrics.consecutive_failures = 0
                    if response_time < 1000:
                        metrics.status = SystemHealthStatus.HEALTHY
                    else:
                        metrics.status = SystemHealthStatus.DEGRADED
                else:
                    metrics.consecutive_failures += 1
                    if metrics.consecutive_failures >= 3:
                        metrics.status = SystemHealthStatus.CRITICAL
                    elif metrics.consecutive_failures >= 5:
                        metrics.status = SystemHealthStatus.OFFLINE
                
                self.health_history.append({
                    'timestamp': datetime.now(),
                    'backend': backend_type.value,
                    'status': metrics.status.value,
                    'response_time_ms': response_time
                })
                
            except Exception as e:
                self.logger.error(f"Health check failed for {backend_type.value}: {e}")
                metrics = self.backend_health[backend_type]
                metrics.consecutive_failures += 1
                metrics.status = SystemHealthStatus.CRITICAL
    
    def _mock_backend_health_check(self, backend_type: BackendType) -> bool:
        """Mock health check for testing (replace with actual service checks)"""
        # Simulate occasional health issues for testing
        if backend_type == BackendType.LIGHTRAG:
            return random.random() > 0.1  # 90% healthy
        else:  # PERPLEXITY
            return random.random() > 0.05  # 95% healthy
    
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
                 load_balancing_config: Optional[LoadBalancingConfig] = None):
        """
        Initialize the intelligent query router.
        
        Args:
            base_router: Base BiomedicalQueryRouter instance
            load_balancing_config: Load balancing configuration
        """
        self.base_router = base_router or BiomedicalQueryRouter()
        self.load_balancing_config = load_balancing_config or LoadBalancingConfig()
        
        # Initialize components
        self.health_monitor = SystemHealthMonitor()
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
            fallback_prediction = RoutingPrediction(
                routing_decision=RoutingDecision.EITHER,
                confidence=0.1,
                reasoning=[f"Emergency fallback due to error: {str(e)}"],
                research_category=ResearchCategory.GENERAL_QUERY,
                confidence_metrics=None,
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
    
    def shutdown(self):
        """Shutdown the router and stop monitoring"""
        self.health_monitor.stop_monitoring()
        self.logger.info("IntelligentQueryRouter shutdown complete")