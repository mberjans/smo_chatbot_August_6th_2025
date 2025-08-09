#!/usr/bin/env python3
"""
Enhanced Production Intelligent Query Router with Comprehensive Logging and Analytics

This module extends the ProductionIntelligentQueryRouter with advanced logging and 
analytics capabilities for routing decisions. It provides detailed insight into
routing patterns, performance metrics, and system behavior.

Key Features:
- Comprehensive routing decision logging
- Real-time analytics and metrics collection
- Anomaly detection and alerting
- Integration with existing production features
- Performance impact monitoring
- Configurable logging levels and storage options

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Enhanced Routing Decision Logging Integration
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
import psutil

from .production_intelligent_query_router import (
    ProductionIntelligentQueryRouter,
    ProductionFeatureFlags,
    DeploymentMode,
    PerformanceComparison,
    ConfigurationMigrator
)
from .intelligent_query_router import (
    IntelligentQueryRouter,
    LoadBalancingConfig,
    HealthCheckConfig,
    SystemHealthMonitor,
    BackendType,
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
from .routing_decision_analytics import (
    RoutingDecisionLogger,
    RoutingAnalytics,
    LoggingConfig,
    RoutingDecisionLogEntry,
    AnalyticsMetrics,
    ProcessingMetrics,
    SystemState,
    LogLevel,
    StorageStrategy,
    create_routing_logger,
    create_routing_analytics,
    RoutingLoggingMixin,
    AnalyticsReport
)


@dataclass
class EnhancedFeatureFlags(ProductionFeatureFlags):
    """Extended feature flags with logging and analytics options"""
    
    # Logging configuration
    enable_routing_logging: bool = True
    routing_log_level: LogLevel = LogLevel.STANDARD
    routing_storage_strategy: StorageStrategy = StorageStrategy.HYBRID
    
    # Analytics configuration
    enable_real_time_analytics: bool = True
    analytics_aggregation_interval_minutes: int = 5
    enable_anomaly_detection: bool = True
    
    # Performance monitoring
    enable_performance_impact_monitoring: bool = True
    max_logging_overhead_ms: float = 5.0  # Max acceptable logging overhead
    
    # Privacy and compliance
    anonymize_query_content: bool = False
    hash_sensitive_data: bool = True
    
    @classmethod
    def from_env(cls) -> 'EnhancedFeatureFlags':
        """Create enhanced feature flags from environment variables"""
        base_flags = ProductionFeatureFlags.from_env()
        
        # Convert to enhanced flags
        enhanced = cls(**base_flags.__dict__)
        
        # Add logging-specific flags
        enhanced.enable_routing_logging = os.getenv('ROUTING_LOGGING_ENABLED', 'true').lower() == 'true'
        enhanced.routing_log_level = LogLevel(os.getenv('ROUTING_LOG_LEVEL', 'standard'))
        enhanced.routing_storage_strategy = StorageStrategy(os.getenv('ROUTING_STORAGE_STRATEGY', 'hybrid'))
        enhanced.enable_real_time_analytics = os.getenv('ROUTING_REAL_TIME_ANALYTICS', 'true').lower() == 'true'
        enhanced.analytics_aggregation_interval_minutes = int(os.getenv('ROUTING_ANALYTICS_INTERVAL_MINUTES', '5'))
        enhanced.enable_anomaly_detection = os.getenv('ROUTING_ANOMALY_DETECTION', 'true').lower() == 'true'
        enhanced.enable_performance_impact_monitoring = os.getenv('ROUTING_PERF_MONITORING', 'true').lower() == 'true'
        enhanced.max_logging_overhead_ms = float(os.getenv('ROUTING_MAX_LOGGING_OVERHEAD_MS', '5.0'))
        enhanced.anonymize_query_content = os.getenv('ROUTING_ANONYMIZE_QUERIES', 'false').lower() == 'true'
        enhanced.hash_sensitive_data = os.getenv('ROUTING_HASH_SENSITIVE_DATA', 'true').lower() == 'true'
        
        return enhanced


class EnhancedProductionIntelligentQueryRouter(ProductionIntelligentQueryRouter):
    """
    Enhanced Production Intelligent Query Router with comprehensive logging and analytics.
    
    This class extends the production router with detailed logging of routing decisions,
    real-time analytics, performance monitoring, and anomaly detection capabilities.
    """
    
    def __init__(self, 
                 base_router: Optional[BiomedicalQueryRouter] = None,
                 load_balancing_config: Optional[LoadBalancingConfig] = None,
                 health_check_config: Optional[HealthCheckConfig] = None,
                 feature_flags: Optional[EnhancedFeatureFlags] = None,
                 production_config: Optional[ProductionLoadBalancingConfig] = None,
                 logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the enhanced production intelligent query router.
        
        Args:
            base_router: Base BiomedicalQueryRouter instance
            load_balancing_config: Legacy load balancing configuration
            health_check_config: Health check configuration
            feature_flags: Enhanced feature flags with logging options
            production_config: Production load balancer configuration
            logging_config: Routing decision logging configuration
        """
        
        # Initialize enhanced feature flags
        self.enhanced_feature_flags = feature_flags or EnhancedFeatureFlags.from_env()
        
        # Initialize base production router
        super().__init__(
            base_router=base_router,
            load_balancing_config=load_balancing_config,
            health_check_config=health_check_config,
            feature_flags=self.enhanced_feature_flags,
            production_config=production_config
        )
        
        # Initialize logging system
        self._init_logging_system(logging_config)
        
        # Performance monitoring
        self.logging_overhead_metrics = deque(maxlen=1000)
        self.total_logged_decisions = 0
        
        # Anomaly detection
        self.last_anomaly_check = datetime.now()
        self.detected_anomalies: List[Dict[str, Any]] = []
        
        self.logger.info("EnhancedProductionIntelligentQueryRouter initialized with logging and analytics")
    
    def _init_logging_system(self, logging_config: Optional[LoggingConfig]):
        """Initialize the routing decision logging and analytics system"""
        
        if not self.enhanced_feature_flags.enable_routing_logging:
            self.routing_logger = None
            self.routing_analytics = None
            return
        
        try:
            # Create logging configuration if not provided
            if logging_config is None:
                logging_config = LoggingConfig(
                    enabled=self.enhanced_feature_flags.enable_routing_logging,
                    log_level=self.enhanced_feature_flags.routing_log_level,
                    storage_strategy=self.enhanced_feature_flags.routing_storage_strategy,
                    anonymize_queries=self.enhanced_feature_flags.anonymize_query_content,
                    hash_sensitive_data=self.enhanced_feature_flags.hash_sensitive_data,
                    enable_real_time_analytics=self.enhanced_feature_flags.enable_real_time_analytics,
                    analytics_aggregation_interval_minutes=self.enhanced_feature_flags.analytics_aggregation_interval_minutes
                )
            
            # Initialize logging components
            self.routing_logger = create_routing_logger(logging_config)
            self.routing_analytics = create_routing_analytics(self.routing_logger)
            
            self.logger.info(f"Routing logging system initialized: level={logging_config.log_level.value}, strategy={logging_config.storage_strategy.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize routing logging system: {e}")
            if not self.enhanced_feature_flags.enable_automatic_failback:
                raise
            
            # Disable logging on failure
            self.routing_logger = None
            self.routing_analytics = None
            self.enhanced_feature_flags.enable_routing_logging = False
            self.logger.warning("Routing logging disabled due to initialization failure")
    
    async def start_monitoring(self):
        """Start monitoring for both production load balancer and routing analytics"""
        # Start base monitoring
        await super().start_monitoring()
        
        # Start logging system
        if self.routing_logger and hasattr(self.routing_logger, 'start_async_logging'):
            await self.routing_logger.start_async_logging()
        
        # Start anomaly detection if enabled
        if self.enhanced_feature_flags.enable_anomaly_detection:
            self._start_anomaly_detection()
    
    async def stop_monitoring(self):
        """Stop monitoring for both production load balancer and routing analytics"""
        # Stop logging system
        if self.routing_logger and hasattr(self.routing_logger, 'stop_async_logging'):
            await self.routing_logger.stop_async_logging()
        
        # Stop base monitoring
        await super().stop_monitoring()
    
    def _start_anomaly_detection(self):
        """Start background anomaly detection"""
        def check_anomalies():
            try:
                if self.routing_analytics:
                    anomalies = self.routing_analytics.detect_anomalies()
                    if anomalies:
                        self.detected_anomalies.extend(anomalies)
                        for anomaly in anomalies:
                            self.logger.warning(f"Routing anomaly detected: {anomaly['type']} - {anomaly['description']}")
                
                # Schedule next check
                import threading
                threading.Timer(300, check_anomalies).start()  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {e}")
        
        # Start first check
        import threading
        threading.Timer(300, check_anomalies).start()
    
    async def route_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> RoutingPrediction:
        """Enhanced route_query with comprehensive logging"""
        routing_start_time = time.time()
        logging_start_time = None
        
        # Prepare system context for logging
        system_context = {
            'query_complexity': len(query_text.split()) / 10.0,
            'deployment_mode': self.enhanced_feature_flags.deployment_mode.value,
            'feature_flags': {
                'production_enabled': self.enhanced_feature_flags.enable_production_load_balancer,
                'logging_enabled': self.enhanced_feature_flags.enable_routing_logging,
                'analytics_enabled': self.enhanced_feature_flags.enable_real_time_analytics
            },
            'request_counter': self.request_counter
        }
        
        try:
            # Get base routing prediction
            prediction = await super().route_query(query_text, context)
            
            routing_time_ms = (time.time() - routing_start_time) * 1000
            
            # Add system context for logging
            await self._collect_system_context(system_context)
            
            # Log the routing decision with performance monitoring
            if self.routing_logger and self.enhanced_feature_flags.enable_routing_logging:
                logging_start_time = time.time()
                await self._log_routing_decision_comprehensive(
                    prediction, query_text, routing_start_time, system_context
                )
                logging_overhead_ms = (time.time() - logging_start_time) * 1000
                
                # Monitor logging performance impact
                self.logging_overhead_metrics.append(logging_overhead_ms)
                
                # Check for excessive logging overhead
                if (self.enhanced_feature_flags.enable_performance_impact_monitoring and 
                    logging_overhead_ms > self.enhanced_feature_flags.max_logging_overhead_ms):
                    self.logger.warning(f"High logging overhead detected: {logging_overhead_ms:.2f}ms")
            
            self.total_logged_decisions += 1
            return prediction
            
        except Exception as e:
            # Log error context
            system_context['errors'] = [str(e)]
            system_context['error_type'] = type(e).__name__
            
            if self.routing_logger and self.enhanced_feature_flags.enable_routing_logging:
                # Create minimal error log entry
                try:
                    await self._log_error_routing_decision(query_text, system_context, routing_start_time)
                except Exception as log_error:
                    self.logger.error(f"Failed to log error routing decision: {log_error}")
            
            raise
    
    async def _collect_system_context(self, context: Dict[str, Any]):
        """Collect comprehensive system context for logging"""
        try:
            # Backend health status
            if hasattr(self.legacy_router, 'health_monitor'):
                health_status = self.legacy_router.get_health_status()
                context['backend_health'] = health_status.get('backend_status', {})
            
            # Production load balancer metrics
            if self.production_load_balancer:
                lb_metrics = await self._get_production_metrics()
                context['load_balancer_metrics'] = lb_metrics
                context['selection_algorithm'] = lb_metrics.get('current_algorithm', 'unknown')
            
            # System resource usage
            context['resource_usage'] = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            # Backend weights if available
            if hasattr(self.legacy_router, 'load_balancer'):
                context['backend_weights'] = getattr(self.legacy_router.load_balancer, 'backend_weights', {})
            
        except Exception as e:
            self.logger.warning(f"Error collecting system context: {e}")
            context['context_collection_error'] = str(e)
    
    async def _log_routing_decision_comprehensive(self,
                                                prediction: RoutingPrediction,
                                                query_text: str,
                                                start_time: float,
                                                system_context: Dict[str, Any]):
        """Log routing decision with comprehensive context"""
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Collect processing metrics
        processing_metrics = {
            'decision_time_ms': getattr(prediction.confidence_metrics, 'calculation_time_ms', 0) 
                               if hasattr(prediction, 'confidence_metrics') else 0,
            'total_time_ms': processing_time_ms,
            'backend_selection_time_ms': system_context.get('backend_selection_time_ms', 0),
            'query_complexity': system_context.get('query_complexity', 0)
        }
        
        # Enhance system state with additional context
        system_state = {
            'backend_health': system_context.get('backend_health', {}),
            'backend_load': system_context.get('backend_load', {}),
            'resource_usage': system_context.get('resource_usage', {}),
            'selection_algorithm': system_context.get('selection_algorithm'),
            'load_balancer_metrics': system_context.get('load_balancer_metrics', {}),
            'backend_weights': system_context.get('backend_weights', {}),
            'errors': system_context.get('errors', []),
            'warnings': system_context.get('warnings', []),
            'fallback_used': system_context.get('fallback_used', False),
            'fallback_reason': system_context.get('fallback_reason'),
            'deployment_mode': system_context.get('deployment_mode'),
            'feature_flags': system_context.get('feature_flags', {}),
            'request_counter': system_context.get('request_counter', self.request_counter),
            'session_id': system_context.get('session_id')
        }
        
        # Log the decision
        await self.routing_logger.log_routing_decision(
            prediction, query_text, processing_metrics, system_state
        )
        
        # Record metrics for analytics
        if self.routing_analytics:
            log_entry = RoutingDecisionLogEntry.from_routing_prediction(
                prediction, query_text, processing_metrics, system_state, 
                self.routing_logger.config
            )
            self.routing_analytics.record_decision_metrics(log_entry)
    
    async def _log_error_routing_decision(self, 
                                        query_text: str, 
                                        system_context: Dict[str, Any], 
                                        start_time: float):
        """Log error routing decision with minimal overhead"""
        
        # Create minimal error prediction
        from .query_router import RoutingDecision, ConfidenceMetrics
        
        error_confidence = ConfidenceMetrics(
            overall_confidence=0.0,
            research_category_confidence=0.0,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=1.0,
            conflict_score=1.0,
            alternative_interpretations=[],
            calculation_time_ms=0.0
        )
        
        error_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.PERPLEXITY,  # Default fallback
            confidence_metrics=error_confidence,
            reasoning=["Error in routing decision"],
            research_category=None
        )
        
        # Log with error context
        await self._log_routing_decision_comprehensive(
            error_prediction, query_text, start_time, system_context
        )
    
    def get_routing_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics summary"""
        base_summary = super().get_performance_report()
        
        if not self.routing_analytics:
            return {**base_summary, 'routing_analytics': {'status': 'disabled'}}
        
        try:
            # Get analytics report
            analytics_report = self.routing_analytics.generate_analytics_report()
            real_time_metrics = self.routing_analytics.get_real_time_metrics()
            
            # Get logging performance metrics
            logging_overhead_stats = {}
            if self.logging_overhead_metrics:
                logging_overhead_stats = {
                    'avg_overhead_ms': statistics.mean(self.logging_overhead_metrics),
                    'max_overhead_ms': max(self.logging_overhead_metrics),
                    'overhead_p95_ms': statistics.quantiles(self.logging_overhead_metrics, n=20)[18] 
                                      if len(self.logging_overhead_metrics) >= 20 else 0
                }
            
            routing_analytics = {
                'status': 'enabled',
                'total_logged_decisions': self.total_logged_decisions,
                'logging_performance': logging_overhead_stats,
                'analytics_report': analytics_report.to_dict(),
                'real_time_metrics': real_time_metrics,
                'detected_anomalies': len(self.detected_anomalies),
                'recent_anomalies': self.detected_anomalies[-5:] if self.detected_anomalies else []
            }
            
        except Exception as e:
            self.logger.error(f"Error generating routing analytics summary: {e}")
            routing_analytics = {
                'status': 'error',
                'error': str(e)
            }
        
        return {
            **base_summary,
            'routing_analytics': routing_analytics
        }
    
    def export_comprehensive_analytics(self, 
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None) -> str:
        """Export comprehensive analytics including routing and performance data"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get base performance data
        performance_data = self.export_performance_data()
        
        # Get routing analytics data
        routing_data = None
        if self.routing_analytics:
            routing_data = self.routing_analytics.export_analytics(start_time, end_time)
        
        # Create comprehensive export
        comprehensive_data = {
            'export_timestamp': datetime.now().isoformat(),
            'export_type': 'comprehensive_analytics',
            'deployment_config': {
                'mode': self.enhanced_feature_flags.deployment_mode.value,
                'production_enabled': self.enhanced_feature_flags.enable_production_load_balancer,
                'logging_enabled': self.enhanced_feature_flags.enable_routing_logging,
                'analytics_enabled': self.enhanced_feature_flags.enable_real_time_analytics
            },
            'performance_report': self.get_performance_report(),
            'routing_analytics_summary': self.get_routing_analytics_summary(),
            'performance_data_file': performance_data,
            'routing_analytics_file': routing_data,
            'detected_anomalies': self.detected_anomalies,
            'logging_overhead_metrics': list(self.logging_overhead_metrics) if self.logging_overhead_metrics else []
        }
        
        export_file = f"comprehensive_analytics_report_{timestamp}.json"
        with open(export_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive analytics exported to {export_file}")
        return export_file
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get detailed anomaly detection report"""
        if not self.routing_analytics or not self.enhanced_feature_flags.enable_anomaly_detection:
            return {'status': 'disabled'}
        
        try:
            current_anomalies = self.routing_analytics.detect_anomalies()
            
            return {
                'status': 'enabled',
                'current_anomalies': current_anomalies,
                'total_detected': len(self.detected_anomalies),
                'detection_history': self.detected_anomalies,
                'last_check': self.last_anomaly_check.isoformat(),
                'anomaly_categories': {
                    'confidence_degradation': len([a for a in self.detected_anomalies if a.get('type') == 'confidence_degradation']),
                    'slow_decisions': len([a for a in self.detected_anomalies if a.get('type') == 'slow_decisions']),
                    'high_error_rate': len([a for a in self.detected_anomalies if a.get('type') == 'high_error_rate'])
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Enhanced health status including routing analytics health"""
        base_health = super().get_health_status()
        
        # Add routing analytics health
        routing_health = {
            'logging_enabled': self.enhanced_feature_flags.enable_routing_logging,
            'analytics_enabled': self.enhanced_feature_flags.enable_real_time_analytics,
            'total_logged_decisions': self.total_logged_decisions,
            'logging_errors': 0,  # Would track logging failures
            'anomaly_detection_status': 'enabled' if self.enhanced_feature_flags.enable_anomaly_detection else 'disabled',
            'detected_anomalies_count': len(self.detected_anomalies)
        }
        
        # Check logging overhead health
        if self.logging_overhead_metrics:
            avg_overhead = statistics.mean(self.logging_overhead_metrics)
            max_overhead = max(self.logging_overhead_metrics)
            
            routing_health['logging_performance'] = {
                'avg_overhead_ms': avg_overhead,
                'max_overhead_ms': max_overhead,
                'overhead_healthy': max_overhead < self.enhanced_feature_flags.max_logging_overhead_ms
            }
        
        base_health['routing_analytics'] = routing_health
        return base_health


# Factory function for easy creation
def create_enhanced_production_router(
    existing_router: Optional[IntelligentQueryRouter] = None,
    enable_production: bool = None,
    deployment_mode: str = None,
    traffic_percentage: float = None,
    enable_logging: bool = True,
    log_level: str = "standard"
) -> EnhancedProductionIntelligentQueryRouter:
    """
    Factory function to create EnhancedProductionIntelligentQueryRouter
    
    Args:
        existing_router: Existing IntelligentQueryRouter to migrate from
        enable_production: Override production enablement
        deployment_mode: Override deployment mode
        traffic_percentage: Override traffic percentage
        enable_logging: Enable routing decision logging
        log_level: Logging level (minimal, standard, detailed, debug)
    
    Returns:
        EnhancedProductionIntelligentQueryRouter instance
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
    
    # Create enhanced feature flags
    feature_flags = EnhancedFeatureFlags.from_env()
    if enable_production is not None:
        feature_flags.enable_production_load_balancer = enable_production
    if deployment_mode is not None:
        feature_flags.deployment_mode = DeploymentMode(deployment_mode)
    if traffic_percentage is not None:
        feature_flags.production_traffic_percentage = traffic_percentage
    
    # Set logging options
    feature_flags.enable_routing_logging = enable_logging
    feature_flags.routing_log_level = LogLevel(log_level)
    
    return EnhancedProductionIntelligentQueryRouter(
        base_router=base_router,
        load_balancing_config=load_balancing_config,
        health_check_config=health_check_config,
        feature_flags=feature_flags
    )


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def main():
        # Create enhanced router with logging enabled
        feature_flags = EnhancedFeatureFlags(
            enable_production_load_balancer=True,
            deployment_mode=DeploymentMode.A_B_TESTING,
            production_traffic_percentage=25.0,
            enable_routing_logging=True,
            routing_log_level=LogLevel.DETAILED,
            enable_real_time_analytics=True,
            enable_anomaly_detection=True
        )
        
        router = EnhancedProductionIntelligentQueryRouter(
            feature_flags=feature_flags
        )
        
        await router.start_monitoring()
        
        # Test queries with logging
        test_queries = [
            "What are the metabolic pathways involved in diabetes?",
            "Latest research on COVID-19 treatments",
            "Explain the role of mitochondria in cellular respiration",
            "Recent developments in cancer immunotherapy",
            "How do biomarkers help in disease diagnosis?"
        ]
        
        print("Testing Enhanced Router with Logging...")
        for i, query in enumerate(test_queries):
            result = await router.route_query(query)
            print(f"Query {i+1}: {query[:50]}...")
            print(f"  Routing: {result.routing_decision}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Backend: {getattr(result, 'backend_selected', 'N/A')}")
            print("---")
        
        # Wait for analytics processing
        await asyncio.sleep(2)
        
        # Generate comprehensive analytics report
        analytics_summary = router.get_routing_analytics_summary()
        print("\nRouting Analytics Summary:")
        print(json.dumps(analytics_summary.get('routing_analytics', {}), indent=2))
        
        # Check for anomalies
        anomaly_report = router.get_anomaly_report()
        print(f"\nAnomaly Detection: {anomaly_report['status']}")
        if anomaly_report['status'] == 'enabled':
            print(f"Current anomalies: {len(anomaly_report['current_anomalies'])}")
        
        # Export comprehensive analytics
        export_file = router.export_comprehensive_analytics()
        print(f"\nComprehensive analytics exported to: {export_file}")
        
        await router.stop_monitoring()
    
    asyncio.run(main())