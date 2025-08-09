#!/usr/bin/env python3
"""
ProductionIntelligentQueryRouter - Enhanced Integration with Production Load Balancer

This module provides a production-ready integration of the IntelligentQueryRouter
with the advanced ProductionLoadBalancer. It maintains full backward compatibility
while adding enterprise-grade load balancing capabilities.

Key Features:
- Seamless integration with existing IntelligentQueryRouter interface
- Production load balancer with 10 advanced algorithms
- Feature flags for safe gradual rollout and A/B testing
- Configuration migration from existing systems
- Performance monitoring and comparison tools
- Rollback capabilities for safety

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: Production Load Balancer Integration
"""

import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import os
from pathlib import Path
import statistics
from contextlib import asynccontextmanager

from .intelligent_query_router import (
    IntelligentQueryRouter,
    LoadBalancingConfig,
    HealthCheckConfig,
    SystemHealthMonitor,
    BackendType,
    RoutingAnalyticsCollector,
    SystemHealthStatus,
    AlertSeverity
)
from .production_load_balancer import (
    ProductionLoadBalancer,
    ProductionLoadBalancingConfig,
    create_default_production_config,
    BackendMetrics,
    LoadBalancingAlgorithm,
    ProductionCircuitBreaker
)
from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction

# Enhanced Circuit Breaker Integration
try:
    from .enhanced_circuit_breaker_system import (
        EnhancedCircuitBreakerIntegration,
        CircuitBreakerOrchestrator,
        ServiceType,
        EnhancedCircuitBreakerState
    )
    from .enhanced_circuit_breaker_config import (
        EnhancedCircuitBreakerConfig,
        load_enhanced_circuit_breaker_config
    )
    ENHANCED_CIRCUIT_BREAKERS_AVAILABLE = True
except ImportError:
    ENHANCED_CIRCUIT_BREAKERS_AVAILABLE = False


class DeploymentMode(Enum):
    """Deployment mode for production rollout"""
    LEGACY_ONLY = "legacy_only"  # Use only existing load balancer
    PRODUCTION_ONLY = "production_only"  # Use only production load balancer
    A_B_TESTING = "a_b_testing"  # Split traffic between legacy and production
    CANARY = "canary"  # Small percentage to production, rest to legacy
    SHADOW = "shadow"  # Production runs in parallel for comparison


@dataclass
class ProductionFeatureFlags:
    """Feature flags for production deployment"""
    enable_production_load_balancer: bool = False
    deployment_mode: DeploymentMode = DeploymentMode.LEGACY_ONLY
    production_traffic_percentage: float = 0.0  # 0-100%
    enable_performance_comparison: bool = True
    enable_automatic_failback: bool = True
    enable_advanced_algorithms: bool = False
    enable_cost_optimization: bool = False
    enable_quality_metrics: bool = True
    rollback_threshold_error_rate: float = 5.0  # %
    rollback_threshold_latency_ms: float = 5000.0
    max_canary_duration_hours: int = 24
    
    @classmethod
    def from_env(cls) -> 'ProductionFeatureFlags':
        """Create feature flags from environment variables"""
        return cls(
            enable_production_load_balancer=os.getenv('PROD_LB_ENABLED', 'false').lower() == 'true',
            deployment_mode=DeploymentMode(os.getenv('PROD_LB_DEPLOYMENT_MODE', 'legacy_only')),
            production_traffic_percentage=float(os.getenv('PROD_LB_TRAFFIC_PERCENT', '0')),
            enable_performance_comparison=os.getenv('PROD_LB_PERF_COMPARISON', 'true').lower() == 'true',
            enable_automatic_failback=os.getenv('PROD_LB_AUTO_FAILBACK', 'true').lower() == 'true',
            enable_advanced_algorithms=os.getenv('PROD_LB_ADVANCED_ALGORITHMS', 'false').lower() == 'true',
            enable_cost_optimization=os.getenv('PROD_LB_COST_OPTIMIZATION', 'false').lower() == 'true',
            enable_quality_metrics=os.getenv('PROD_LB_QUALITY_METRICS', 'true').lower() == 'true',
            rollback_threshold_error_rate=float(os.getenv('PROD_LB_ROLLBACK_ERROR_RATE', '5.0')),
            rollback_threshold_latency_ms=float(os.getenv('PROD_LB_ROLLBACK_LATENCY_MS', '5000.0')),
            max_canary_duration_hours=int(os.getenv('PROD_LB_CANARY_MAX_HOURS', '24'))
        )


@dataclass
class PerformanceComparison:
    """Performance comparison between legacy and production systems"""
    timestamp: datetime
    legacy_response_time_ms: float
    production_response_time_ms: float
    legacy_success: bool
    production_success: bool
    legacy_backend: Optional[str]
    production_backend: Optional[str]
    query_complexity: float
    cost_difference: float = 0.0
    quality_difference: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'legacy_response_time_ms': self.legacy_response_time_ms,
            'production_response_time_ms': self.production_response_time_ms,
            'legacy_success': self.legacy_success,
            'production_success': self.production_success,
            'legacy_backend': self.legacy_backend,
            'production_backend': self.production_backend,
            'query_complexity': self.query_complexity,
            'cost_difference': self.cost_difference,
            'quality_difference': self.quality_difference,
            'performance_improvement': (
                (self.legacy_response_time_ms - self.production_response_time_ms) / self.legacy_response_time_ms * 100
                if self.legacy_response_time_ms > 0 else 0
            )
        }


class ConfigurationMigrator:
    """Migrates existing LoadBalancingConfig to ProductionLoadBalancingConfig"""
    
    @staticmethod
    def migrate_config(legacy_config: LoadBalancingConfig) -> ProductionLoadBalancingConfig:
        """Migrate legacy configuration to production configuration"""
        
        # Start with default production config
        prod_config = create_default_production_config()
        
        # Map legacy settings to production settings
        prod_config.health_monitoring.check_interval_seconds = legacy_config.health_check_interval
        prod_config.circuit_breaker.failure_threshold = legacy_config.circuit_breaker_threshold
        prod_config.circuit_breaker.recovery_timeout_seconds = legacy_config.circuit_breaker_timeout
        prod_config.performance_thresholds.response_time_ms = legacy_config.response_time_threshold_ms
        
        # Map strategy
        if legacy_config.strategy == "round_robin":
            prod_config.algorithm_config.primary_algorithm = "round_robin"
        elif legacy_config.strategy == "weighted":
            prod_config.algorithm_config.primary_algorithm = "weighted_round_robin"
        elif legacy_config.strategy == "health_aware":
            prod_config.algorithm_config.primary_algorithm = "health_aware"
        else:
            prod_config.algorithm_config.primary_algorithm = "weighted_round_robin"
        
        # Enable adaptive routing if it was enabled in legacy
        if legacy_config.enable_adaptive_routing:
            prod_config.algorithm_config.enable_adaptive_selection = True
            prod_config.algorithm_config.fallback_algorithms = [
                "least_connections", "response_time_based", "health_aware"
            ]
        
        return prod_config
    
    @staticmethod
    def validate_migration(legacy_config: LoadBalancingConfig, 
                          prod_config: ProductionLoadBalancingConfig) -> Dict[str, Any]:
        """Validate that migration preserved important settings"""
        validation_results = {
            'health_check_interval_preserved': (
                prod_config.health_monitoring.check_interval_seconds == legacy_config.health_check_interval
            ),
            'circuit_breaker_threshold_preserved': (
                prod_config.circuit_breaker.failure_threshold == legacy_config.circuit_breaker_threshold
            ),
            'response_time_threshold_preserved': (
                prod_config.performance_thresholds.response_time_ms == legacy_config.response_time_threshold_ms
            ),
            'adaptive_routing_preserved': (
                prod_config.algorithm_config.enable_adaptive_selection == legacy_config.enable_adaptive_routing
            )
        }
        
        validation_results['migration_successful'] = all(validation_results.values())
        return validation_results


class ProductionIntelligentQueryRouter:
    """
    Production-ready intelligent query router with enhanced load balancing.
    
    This class provides a drop-in replacement for IntelligentQueryRouter with
    production-grade load balancing capabilities, while maintaining full backward
    compatibility and providing safe deployment mechanisms.
    """
    
    def __init__(self, 
                 base_router: Optional[BiomedicalQueryRouter] = None,
                 load_balancing_config: Optional[LoadBalancingConfig] = None,
                 health_check_config: Optional[HealthCheckConfig] = None,
                 feature_flags: Optional[ProductionFeatureFlags] = None,
                 production_config: Optional[ProductionLoadBalancingConfig] = None):
        """
        Initialize the production intelligent query router.
        
        Args:
            base_router: Base BiomedicalQueryRouter instance
            load_balancing_config: Legacy load balancing configuration
            health_check_config: Health check configuration
            feature_flags: Production feature flags
            production_config: Production load balancer configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature flags
        self.feature_flags = feature_flags or ProductionFeatureFlags.from_env()
        
        # Initialize legacy system (always available for fallback)
        self.legacy_router = IntelligentQueryRouter(
            base_router=base_router,
            load_balancing_config=load_balancing_config,
            health_check_config=health_check_config
        )
        
        # Initialize production system if enabled
        self.production_load_balancer: Optional[ProductionLoadBalancer] = None
        if self.feature_flags.enable_production_load_balancer:
            try:
                # Migrate configuration if production config not provided
                if production_config is None:
                    legacy_config = load_balancing_config or LoadBalancingConfig()
                    production_config = ConfigurationMigrator.migrate_config(legacy_config)
                    
                    # Log migration results
                    migration_validation = ConfigurationMigrator.validate_migration(
                        legacy_config, production_config
                    )
                    self.logger.info(f"Configuration migration validation: {migration_validation}")
                
                self.production_load_balancer = ProductionLoadBalancer(production_config)
                self.logger.info("Production load balancer initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize production load balancer: {e}")
                if not self.feature_flags.enable_automatic_failback:
                    raise
                self.logger.warning("Falling back to legacy load balancer")
        
        # Performance monitoring
        self.performance_comparisons: deque = deque(maxlen=10000)
        self.deployment_start_time = datetime.now()
        self.request_counter = 0
        
        # Traffic routing state
        self._canary_start_time: Optional[datetime] = None
        
        self.logger.info(f"ProductionIntelligentQueryRouter initialized with deployment mode: {self.feature_flags.deployment_mode}")
    
    async def start_monitoring(self):
        """Start monitoring for production load balancer"""
        if self.production_load_balancer:
            await self.production_load_balancer.start_monitoring()
    
    async def stop_monitoring(self):
        """Stop monitoring for production load balancer"""
        if self.production_load_balancer:
            await self.production_load_balancer.stop_monitoring()
    
    def _should_use_production(self, query_complexity: float = 1.0) -> bool:
        """Determine if request should use production load balancer"""
        if not self.feature_flags.enable_production_load_balancer or not self.production_load_balancer:
            return False
        
        mode = self.feature_flags.deployment_mode
        
        if mode == DeploymentMode.LEGACY_ONLY:
            return False
        elif mode == DeploymentMode.PRODUCTION_ONLY:
            return True
        elif mode == DeploymentMode.A_B_TESTING:
            # Hash-based consistent routing for A/B testing
            import hashlib
            query_hash = int(hashlib.md5(str(self.request_counter).encode()).hexdigest()[:8], 16)
            return (query_hash % 100) < self.feature_flags.production_traffic_percentage
        elif mode == DeploymentMode.CANARY:
            # Check canary time limits
            if self._canary_start_time is None:
                self._canary_start_time = datetime.now()
            
            canary_duration = datetime.now() - self._canary_start_time
            if canary_duration.total_seconds() > self.feature_flags.max_canary_duration_hours * 3600:
                self.logger.warning("Canary deployment exceeded maximum duration, falling back to legacy")
                return False
            
            # Random routing based on percentage
            import random
            return random.random() * 100 < self.feature_flags.production_traffic_percentage
        elif mode == DeploymentMode.SHADOW:
            # Shadow mode: primary uses legacy, production runs in parallel
            return False  # We'll run production in parallel for comparison
        
        return False
    
    async def route_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> RoutingPrediction:
        """Route query using appropriate load balancer"""
        self.request_counter += 1
        start_time = time.time()
        
        # Calculate query complexity for routing decisions
        query_complexity = len(query_text.split()) / 10.0  # Simple complexity metric
        
        # Check for automatic rollback conditions
        if self._should_trigger_rollback():
            self.logger.critical("Automatic rollback triggered due to performance degradation")
            self.feature_flags.deployment_mode = DeploymentMode.LEGACY_ONLY
        
        try:
            if self.feature_flags.deployment_mode == DeploymentMode.SHADOW:
                # Shadow mode: run both systems and compare
                return await self._run_shadow_mode(query_text, context, query_complexity)
            elif self._should_use_production(query_complexity):
                # Use production load balancer
                return await self._route_with_production(query_text, context, query_complexity)
            else:
                # Use legacy load balancer
                return await self._route_with_legacy(query_text, context, query_complexity)
                
        except Exception as e:
            self.logger.error(f"Error in route_query: {e}")
            if self.feature_flags.enable_automatic_failback:
                self.logger.warning("Failing back to legacy router due to error")
                return await self._route_with_legacy(query_text, context, query_complexity)
            raise
    
    async def _route_with_legacy(self, query_text: str, context: Optional[Dict[str, Any]], 
                               query_complexity: float) -> RoutingPrediction:
        """Route query using legacy load balancer"""
        start_time = time.time()
        
        try:
            # Use the legacy intelligent query router
            prediction = await asyncio.get_event_loop().run_in_executor(
                None, self.legacy_router.route_query, query_text, context
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log performance metrics
            self.logger.debug(f"Legacy routing completed in {response_time_ms:.2f}ms")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Legacy routing failed: {e}")
            raise
    
    async def _route_with_production(self, query_text: str, context: Optional[Dict[str, Any]], 
                                   query_complexity: float) -> RoutingPrediction:
        """Route query using production load balancer"""
        start_time = time.time()
        
        try:
            # First get base routing decision from biomedical router
            base_prediction = await asyncio.get_event_loop().run_in_executor(
                None, self.legacy_router.base_router.route_query, query_text, context
            )
            
            # Use production load balancer for backend selection
            selected_backend = await self.production_load_balancer.select_backend(
                base_prediction.routing_decision, context or {}
            )
            
            if selected_backend is None:
                # No backend available, fallback to legacy
                self.logger.warning("No backend available from production load balancer, using legacy")
                return await self._route_with_legacy(query_text, context, query_complexity)
            
            # Update the prediction with production-selected backend
            if selected_backend == "lightrag":
                final_decision = RoutingDecision.LIGHTRAG
            elif selected_backend == "perplexity":
                final_decision = RoutingDecision.PERPLEXITY
            else:
                final_decision = base_prediction.routing_decision
            
            # Create enhanced prediction with production metrics
            enhanced_prediction = RoutingPrediction(
                routing_decision=final_decision,
                confidence_metrics=base_prediction.confidence_metrics,
                reasoning=f"Production LB: {base_prediction.reasoning}",
                backend_selected=selected_backend,
                load_balancer_metrics=await self._get_production_metrics()
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log performance metrics
            self.logger.debug(f"Production routing completed in {response_time_ms:.2f}ms, backend: {selected_backend}")
            
            return enhanced_prediction
            
        except Exception as e:
            self.logger.error(f"Production routing failed: {e}")
            if self.feature_flags.enable_automatic_failback:
                self.logger.warning("Failing back to legacy router")
                return await self._route_with_legacy(query_text, context, query_complexity)
            raise
    
    async def _run_shadow_mode(self, query_text: str, context: Optional[Dict[str, Any]], 
                             query_complexity: float) -> RoutingPrediction:
        """Run both legacy and production in parallel for comparison"""
        legacy_start = time.time()
        production_start = time.time()
        
        try:
            # Run both systems concurrently
            legacy_task = asyncio.create_task(
                self._route_with_legacy(query_text, context, query_complexity)
            )
            production_task = asyncio.create_task(
                self._route_with_production(query_text, context, query_complexity)
            )
            
            # Wait for both to complete with timeout
            legacy_result, production_result = await asyncio.gather(
                legacy_task, production_task, return_exceptions=True
            )
            
            legacy_time = (time.time() - legacy_start) * 1000
            production_time = (time.time() - production_start) * 1000
            
            # Record performance comparison
            comparison = PerformanceComparison(
                timestamp=datetime.now(),
                legacy_response_time_ms=legacy_time,
                production_response_time_ms=production_time,
                legacy_success=not isinstance(legacy_result, Exception),
                production_success=not isinstance(production_result, Exception),
                legacy_backend=getattr(legacy_result, 'backend_selected', None) if not isinstance(legacy_result, Exception) else None,
                production_backend=getattr(production_result, 'backend_selected', None) if not isinstance(production_result, Exception) else None,
                query_complexity=query_complexity
            )
            
            self.performance_comparisons.append(comparison)
            
            # Log comparison
            self.logger.info(f"Shadow mode comparison: Legacy {legacy_time:.2f}ms vs Production {production_time:.2f}ms")
            
            # Return legacy result (shadow mode uses legacy as primary)
            if isinstance(legacy_result, Exception):
                raise legacy_result
            return legacy_result
            
        except Exception as e:
            self.logger.error(f"Shadow mode execution failed: {e}")
            # Fallback to legacy only
            return await self._route_with_legacy(query_text, context, query_complexity)
    
    async def _get_production_metrics(self) -> Dict[str, Any]:
        """Get current production load balancer metrics"""
        if not self.production_load_balancer:
            return {}
        
        try:
            metrics = await self.production_load_balancer.get_metrics_summary()
            return {
                'active_backends': len(metrics.get('active_backends', [])),
                'total_requests': metrics.get('total_requests', 0),
                'average_response_time': metrics.get('average_response_time_ms', 0),
                'current_algorithm': metrics.get('current_algorithm', 'unknown')
            }
        except Exception as e:
            self.logger.warning(f"Failed to get production metrics: {e}")
            return {}
    
    def _should_trigger_rollback(self) -> bool:
        """Check if automatic rollback should be triggered"""
        if not self.feature_flags.enable_automatic_failback:
            return False
        
        if len(self.performance_comparisons) < 100:  # Need sufficient data
            return False
        
        # Check recent performance comparisons (last 100 requests)
        recent_comparisons = list(self.performance_comparisons)[-100:]
        
        # Calculate error rates
        production_errors = sum(1 for c in recent_comparisons if not c.production_success)
        error_rate = (production_errors / len(recent_comparisons)) * 100
        
        if error_rate > self.feature_flags.rollback_threshold_error_rate:
            self.logger.critical(f"Error rate {error_rate:.2f}% exceeds threshold {self.feature_flags.rollback_threshold_error_rate}%")
            return True
        
        # Check latency degradation
        production_latencies = [c.production_response_time_ms for c in recent_comparisons if c.production_success]
        if production_latencies:
            avg_production_latency = statistics.mean(production_latencies)
            if avg_production_latency > self.feature_flags.rollback_threshold_latency_ms:
                self.logger.critical(f"Average latency {avg_production_latency:.2f}ms exceeds threshold {self.feature_flags.rollback_threshold_latency_ms}ms")
                return True
        
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_comparisons:
            return {'error': 'No performance data available'}
        
        comparisons = list(self.performance_comparisons)
        
        # Calculate statistics
        legacy_times = [c.legacy_response_time_ms for c in comparisons if c.legacy_success]
        production_times = [c.production_response_time_ms for c in comparisons if c.production_success]
        
        legacy_success_rate = (sum(1 for c in comparisons if c.legacy_success) / len(comparisons)) * 100
        production_success_rate = (sum(1 for c in comparisons if c.production_success) / len(comparisons)) * 100
        
        report = {
            'deployment_mode': self.feature_flags.deployment_mode.value,
            'total_requests': len(comparisons),
            'deployment_duration_hours': (datetime.now() - self.deployment_start_time).total_seconds() / 3600,
            'legacy_stats': {
                'success_rate': legacy_success_rate,
                'avg_response_time_ms': statistics.mean(legacy_times) if legacy_times else 0,
                'median_response_time_ms': statistics.median(legacy_times) if legacy_times else 0,
                'p95_response_time_ms': statistics.quantiles(legacy_times, n=20)[18] if len(legacy_times) >= 20 else 0
            },
            'production_stats': {
                'success_rate': production_success_rate,
                'avg_response_time_ms': statistics.mean(production_times) if production_times else 0,
                'median_response_time_ms': statistics.median(production_times) if production_times else 0,
                'p95_response_time_ms': statistics.quantiles(production_times, n=20)[18] if len(production_times) >= 20 else 0
            },
            'performance_improvement': {
                'response_time_improvement_percent': (
                    ((statistics.mean(legacy_times) - statistics.mean(production_times)) / statistics.mean(legacy_times)) * 100
                    if legacy_times and production_times and statistics.mean(legacy_times) > 0 else 0
                ),
                'success_rate_difference': production_success_rate - legacy_success_rate
            },
            'recommendation': self._get_deployment_recommendation(comparisons)
        }
        
        return report
    
    def _get_deployment_recommendation(self, comparisons: List[PerformanceComparison]) -> str:
        """Generate deployment recommendation based on performance data"""
        if len(comparisons) < 100:
            return "Insufficient data for recommendation. Continue monitoring."
        
        legacy_times = [c.legacy_response_time_ms for c in comparisons if c.legacy_success]
        production_times = [c.production_response_time_ms for c in comparisons if c.production_success]
        
        if not legacy_times or not production_times:
            return "Insufficient success data for comparison. Review error logs."
        
        legacy_avg = statistics.mean(legacy_times)
        production_avg = statistics.mean(production_times)
        
        improvement_percent = ((legacy_avg - production_avg) / legacy_avg) * 100
        
        legacy_success_rate = (sum(1 for c in comparisons if c.legacy_success) / len(comparisons)) * 100
        production_success_rate = (sum(1 for c in comparisons if c.production_success) / len(comparisons)) * 100
        
        if improvement_percent > 10 and production_success_rate >= legacy_success_rate - 1:
            return "RECOMMENDED: Proceed with full production rollout. Significant performance improvement observed."
        elif improvement_percent > 5 and production_success_rate >= legacy_success_rate - 0.5:
            return "RECOMMENDED: Increase production traffic percentage. Moderate performance improvement observed."
        elif improvement_percent > 0 and production_success_rate >= legacy_success_rate:
            return "NEUTRAL: Continue current deployment. Slight improvement with stable reliability."
        elif production_success_rate < legacy_success_rate - 2:
            return "CAUTION: Consider rollback. Production showing reliability issues."
        else:
            return "CAUTION: No significant improvement. Consider optimization or rollback."
    
    def force_rollback(self, reason: str = "Manual rollback"):
        """Force rollback to legacy system"""
        self.logger.critical(f"Forcing rollback to legacy system: {reason}")
        self.feature_flags.deployment_mode = DeploymentMode.LEGACY_ONLY
        self.feature_flags.enable_production_load_balancer = False
    
    def export_performance_data(self, file_path: Optional[str] = None) -> str:
        """Export performance comparison data to JSON file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"production_performance_comparison_{timestamp}.json"
        
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'deployment_config': {
                'mode': self.feature_flags.deployment_mode.value,
                'traffic_percentage': self.feature_flags.production_traffic_percentage,
                'deployment_start': self.deployment_start_time.isoformat()
            },
            'performance_report': self.get_performance_report(),
            'raw_comparisons': [c.to_dict() for c in self.performance_comparisons]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Performance data exported to {file_path}")
        return file_path
    
    # Backward compatibility methods - delegate to legacy router
    def update_backend_weights(self, backend_weights: Dict[str, float]):
        """Update backend weights (backward compatibility)"""
        self.legacy_router.update_backend_weights(backend_weights)
        
        # Also update production load balancer if available
        if self.production_load_balancer:
            try:
                asyncio.create_task(self.production_load_balancer.update_backend_weights(backend_weights))
            except Exception as e:
                self.logger.warning(f"Failed to update production backend weights: {e}")
    
    def export_analytics(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Export analytics (backward compatibility)"""
        legacy_analytics = self.legacy_router.export_analytics(start_time, end_time)
        
        # Add production performance data
        legacy_analytics['production_integration'] = {
            'enabled': self.feature_flags.enable_production_load_balancer,
            'deployment_mode': self.feature_flags.deployment_mode.value,
            'performance_report': self.get_performance_report()
        }
        
        return legacy_analytics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status (backward compatibility)"""
        legacy_health = self.legacy_router.get_health_status()
        
        if self.production_load_balancer:
            try:
                production_health = asyncio.create_task(self.production_load_balancer.get_health_summary())
                legacy_health['production_load_balancer'] = {
                    'status': 'available',
                    'health': production_health
                }
            except Exception as e:
                legacy_health['production_load_balancer'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            legacy_health['production_load_balancer'] = {
                'status': 'disabled'
            }
        
        return legacy_health


# Factory function for easy migration
def create_production_intelligent_query_router(
    existing_router: Optional[IntelligentQueryRouter] = None,
    enable_production: bool = None,
    deployment_mode: str = None,
    traffic_percentage: float = None
) -> ProductionIntelligentQueryRouter:
    """
    Factory function to create ProductionIntelligentQueryRouter from existing router
    
    Args:
        existing_router: Existing IntelligentQueryRouter to migrate from
        enable_production: Override production enablement
        deployment_mode: Override deployment mode
        traffic_percentage: Override traffic percentage
    
    Returns:
        ProductionIntelligentQueryRouter instance
    """
    if existing_router:
        # Extract configuration from existing router
        base_router = existing_router.base_router
        load_balancing_config = existing_router.load_balancing_config
        health_check_config = existing_router.health_check_config
    else:
        base_router = None
        load_balancing_config = None
        health_check_config = None
    
    # Create feature flags with overrides
    feature_flags = ProductionFeatureFlags.from_env()
    if enable_production is not None:
        feature_flags.enable_production_load_balancer = enable_production
    if deployment_mode is not None:
        feature_flags.deployment_mode = DeploymentMode(deployment_mode)
    if traffic_percentage is not None:
        feature_flags.production_traffic_percentage = traffic_percentage
    
    return ProductionIntelligentQueryRouter(
        base_router=base_router,
        load_balancing_config=load_balancing_config,
        health_check_config=health_check_config,
        feature_flags=feature_flags
    )


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def main():
        # Create production router with canary deployment
        feature_flags = ProductionFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.CANARY,
            production_traffic_percentage=10.0,  # 10% to production
            enable_performance_comparison=True
        )
        
        router = ProductionIntelligentQueryRouter(
            feature_flags=feature_flags
        )
        
        await router.start_monitoring()
        
        # Test queries
        test_queries = [
            "What are the metabolic pathways involved in diabetes?",
            "Explain the role of mitochondria in cellular respiration.",
            "How do biomarkers help in disease diagnosis?"
        ]
        
        for query in test_queries:
            result = await router.route_query(query)
            print(f"Query: {query[:50]}...")
            print(f"Routing: {result.routing_decision}")
            print(f"Backend: {getattr(result, 'backend_selected', 'N/A')}")
            print("---")
        
        # Generate performance report
        report = router.get_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2))
        
        await router.stop_monitoring()
    
    asyncio.run(main())
