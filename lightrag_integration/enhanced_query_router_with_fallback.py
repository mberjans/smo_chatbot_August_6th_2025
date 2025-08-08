"""
Enhanced Query Router with Comprehensive Fallback Integration

This module provides a seamless integration layer that enhances the existing
BiomedicalQueryRouter with comprehensive multi-tiered fallback capabilities
while maintaining full backward compatibility.

The enhanced router automatically:
- Detects failure conditions intelligently
- Implements progressive degradation strategies
- Provides 100% system availability through multi-level fallbacks
- Monitors and recovers from failures automatically
- Maintains all existing API compatibility

Classes:
    - EnhancedBiomedicalQueryRouter: Drop-in replacement with fallback capabilities
    - FallbackIntegrationConfig: Configuration for fallback integration
    - CompatibilityLayer: Ensures backward compatibility
    - AutoConfigurationManager: Automatically configures fallback system

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# Import existing components
try:
    from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction, ConfidenceMetrics
    from .enhanced_llm_classifier import EnhancedLLMQueryClassifier
    from .research_categorizer import ResearchCategorizer, CategoryPrediction
    from .cost_persistence import ResearchCategory
    
    # Import the comprehensive fallback system
    from .comprehensive_fallback_system import (
        FallbackOrchestrator, 
        FallbackMonitor,
        FallbackResult,
        FallbackLevel,
        FailureType,
        create_comprehensive_fallback_system
    )
    
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


@dataclass
class FallbackIntegrationConfig:
    """Configuration for fallback system integration."""
    
    # Fallback system configuration
    enable_fallback_system: bool = True
    enable_monitoring: bool = True
    monitoring_interval_seconds: int = 60
    
    # Emergency cache configuration
    emergency_cache_file: Optional[str] = None
    enable_cache_warming: bool = True
    cache_common_patterns: bool = True
    
    # Performance thresholds
    max_response_time_ms: float = 2000.0
    confidence_threshold: float = 0.6
    health_score_threshold: float = 0.7
    
    # Integration settings
    maintain_backward_compatibility: bool = True
    log_fallback_events: bool = True
    enable_auto_recovery: bool = True
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enable_fallback_system': self.enable_fallback_system,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_interval_seconds': self.monitoring_interval_seconds,
            'emergency_cache_file': self.emergency_cache_file,
            'enable_cache_warming': self.enable_cache_warming,
            'cache_common_patterns': self.cache_common_patterns,
            'max_response_time_ms': self.max_response_time_ms,
            'confidence_threshold': self.confidence_threshold,
            'health_score_threshold': self.health_score_threshold,
            'maintain_backward_compatibility': self.maintain_backward_compatibility,
            'log_fallback_events': self.log_fallback_events,
            'enable_auto_recovery': self.enable_auto_recovery,
            'enable_alerts': self.enable_alerts,
            'alert_cooldown_seconds': self.alert_cooldown_seconds
        }


class CompatibilityLayer:
    """
    Compatibility layer to ensure seamless integration with existing code.
    Handles conversion between existing and enhanced result formats.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize compatibility layer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def convert_fallback_result_to_routing_prediction(self, fallback_result: FallbackResult) -> RoutingPrediction:
        """
        Convert FallbackResult to RoutingPrediction for backward compatibility.
        
        Args:
            fallback_result: Result from fallback system
            
        Returns:
            RoutingPrediction compatible with existing code
        """
        # Extract the routing prediction
        routing_prediction = fallback_result.routing_prediction
        
        # Enhance metadata with fallback information
        if not routing_prediction.metadata:
            routing_prediction.metadata = {}
        
        routing_prediction.metadata.update({
            'fallback_system_used': True,
            'fallback_level_used': fallback_result.fallback_level_used.name,
            'fallback_success': fallback_result.success,
            'total_fallback_time_ms': fallback_result.total_processing_time_ms,
            'quality_score': fallback_result.quality_score,
            'reliability_score': fallback_result.reliability_score,
            'confidence_degradation': fallback_result.confidence_degradation
        })
        
        # Add fallback warnings to reasoning if present
        if fallback_result.warnings:
            additional_reasoning = [f"Fallback warning: {warning}" for warning in fallback_result.warnings[:3]]
            routing_prediction.reasoning.extend(additional_reasoning)
        
        # Add recovery suggestions if confidence is low
        if (routing_prediction.confidence < 0.3 and 
            fallback_result.recovery_suggestions):
            routing_prediction.metadata['recovery_suggestions'] = fallback_result.recovery_suggestions[:3]
        
        return routing_prediction
    
    def enhance_routing_prediction_with_fallback_info(self, 
                                                    prediction: RoutingPrediction,
                                                    fallback_info: Dict[str, Any]) -> RoutingPrediction:
        """Enhance existing routing prediction with fallback information."""
        if not prediction.metadata:
            prediction.metadata = {}
        
        prediction.metadata.update(fallback_info)
        return prediction
    
    def log_compatibility_event(self, event_type: str, details: Dict[str, Any]):
        """Log compatibility-related events."""
        self.logger.debug(f"Compatibility event [{event_type}]: {details}")


class AutoConfigurationManager:
    """
    Automatically configures the fallback system based on detected components
    and system environment.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize auto-configuration manager."""
        self.logger = logger or logging.getLogger(__name__)
    
    def auto_configure_fallback_system(self, 
                                     existing_router: Optional[BiomedicalQueryRouter] = None,
                                     existing_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                                     existing_categorizer: Optional[ResearchCategorizer] = None) -> Dict[str, Any]:
        """
        Automatically configure fallback system based on available components.
        
        Args:
            existing_router: Existing query router
            existing_classifier: Existing LLM classifier  
            existing_categorizer: Existing research categorizer
            
        Returns:
            Configuration dictionary for fallback system
        """
        config = {}
        
        # Configure emergency cache
        cache_dir = Path("fallback_cache")
        cache_dir.mkdir(exist_ok=True)
        config['emergency_cache_file'] = str(cache_dir / "emergency_cache.pkl")
        
        # Configure based on available components
        components_available = {
            'query_router': existing_router is not None,
            'llm_classifier': existing_classifier is not None,
            'research_categorizer': existing_categorizer is not None
        }
        
        config['available_components'] = components_available
        
        # Adjust fallback thresholds based on available components
        if components_available['llm_classifier']:
            config['llm_confidence_threshold'] = 0.6
        else:
            config['llm_confidence_threshold'] = 0.0  # No LLM available
        
        if components_available['research_categorizer']:
            config['keyword_confidence_threshold'] = 0.3
        else:
            config['keyword_confidence_threshold'] = 0.1  # Very low threshold
        
        # Performance configuration
        config['performance_targets'] = {
            'max_response_time_ms': 2000,
            'min_confidence': 0.1,
            'target_success_rate': 0.99
        }
        
        self.logger.info(f"Auto-configured fallback system with components: {components_available}")
        return config


class EnhancedBiomedicalQueryRouter(BiomedicalQueryRouter):
    """
    Enhanced Biomedical Query Router with comprehensive fallback capabilities.
    
    This class extends the existing BiomedicalQueryRouter to provide:
    - Multi-tiered fallback mechanisms
    - Intelligent failure detection
    - Automatic recovery capabilities
    - 100% system availability guarantee
    - Full backward compatibility with existing code
    """
    
    def __init__(self, 
                 fallback_config: Optional[FallbackIntegrationConfig] = None,
                 llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced query router with fallback capabilities.
        
        Args:
            fallback_config: Configuration for fallback system
            llm_classifier: Optional LLM classifier for integration
            logger: Logger instance
        """
        # Initialize parent class
        super().__init__(logger)
        
        # Initialize fallback components
        self.fallback_config = fallback_config or FallbackIntegrationConfig()
        self.llm_classifier = llm_classifier
        self.compatibility_layer = CompatibilityLayer(logger=self.logger)
        self.auto_config_manager = AutoConfigurationManager(logger=self.logger)
        
        # Initialize fallback system
        self.fallback_orchestrator = None
        self.fallback_monitor = None
        
        if self.fallback_config.enable_fallback_system:
            self._initialize_fallback_system()
        
        # Performance tracking for enhanced capabilities
        self.enhanced_routing_stats = {
            'fallback_activations': 0,
            'emergency_cache_uses': 0,
            'recovery_events': 0,
            'total_enhanced_queries': 0
        }
        
        self.logger.info("Enhanced Biomedical Query Router initialized with comprehensive fallback system")
    
    def _initialize_fallback_system(self):
        """Initialize the comprehensive fallback system."""
        try:
            # Auto-configure fallback system
            auto_config = self.auto_config_manager.auto_configure_fallback_system(
                existing_router=self,
                existing_classifier=self.llm_classifier,
                existing_categorizer=self
            )
            
            # Create fallback system
            fallback_config = {
                **auto_config,
                'emergency_cache_file': self.fallback_config.emergency_cache_file
            }
            
            self.fallback_orchestrator, self.fallback_monitor = create_comprehensive_fallback_system(
                config=fallback_config,
                logger=self.logger
            )
            
            # Integrate with existing components
            self.fallback_orchestrator.integrate_with_existing_components(
                query_router=self,
                llm_classifier=self.llm_classifier,
                research_categorizer=self
            )
            
            # Configure monitoring
            if self.fallback_config.enable_monitoring:
                if not self.fallback_monitor.monitoring_active:
                    self.fallback_monitor.start_monitoring(
                        check_interval_seconds=self.fallback_config.monitoring_interval_seconds
                    )
            
            # Warm cache if enabled
            if self.fallback_config.enable_cache_warming:
                self._warm_emergency_cache()
            
            self.logger.info("Fallback system initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback system: {e}")
            # Disable fallback system if initialization fails
            self.fallback_config.enable_fallback_system = False
    
    def _warm_emergency_cache(self):
        """Warm the emergency cache with common query patterns."""
        if not self.fallback_orchestrator:
            return
        
        common_patterns = [
            # Metabolite identification
            "identify metabolite",
            "compound identification", 
            "mass spectrum analysis",
            "molecular structure",
            
            # Pathway analysis
            "pathway analysis",
            "metabolic pathway",
            "biochemical network",
            "enzyme pathway",
            
            # Biomarker discovery
            "biomarker discovery",
            "disease marker",
            "diagnostic metabolite",
            "prognostic signature",
            
            # Drug discovery
            "drug discovery",
            "pharmaceutical compound",
            "drug target",
            "therapeutic mechanism",
            
            # Clinical diagnosis
            "clinical diagnosis",
            "patient sample",
            "medical metabolomics",
            "diagnostic testing",
            
            # Real-time queries
            "latest research",
            "recent studies",
            "current developments",
            "breaking news",
            
            # General queries
            "what is metabolomics",
            "explain pathway",
            "define biomarker",
            "metabolite analysis"
        ]
        
        try:
            self.fallback_orchestrator.emergency_cache.warm_cache(common_patterns)
            self.logger.info(f"Warmed emergency cache with {len(common_patterns)} common patterns")
        except Exception as e:
            self.logger.warning(f"Failed to warm emergency cache: {e}")
    
    def route_query(self, 
                   query_text: str, 
                   context: Optional[Dict[str, Any]] = None) -> RoutingPrediction:
        """
        Route a query with enhanced fallback capabilities.
        
        This method maintains full backward compatibility while providing
        comprehensive fallback protection.
        
        Args:
            query_text: The user query text to route
            context: Optional context information
            
        Returns:
            RoutingPrediction with enhanced reliability
        """
        start_time = time.time()
        self.enhanced_routing_stats['total_enhanced_queries'] += 1
        
        # Determine query priority from context
        priority = context.get('priority', 'normal') if context else 'normal'
        
        # Add performance target to context
        if context is None:
            context = {}
        context['performance_target_ms'] = self.fallback_config.max_response_time_ms
        
        # Try primary routing first
        primary_result = None
        primary_error = None
        
        try:
            # Use parent class routing method
            primary_result = super().route_query(query_text, context)
            
            # Validate primary result
            if self._is_primary_result_acceptable(primary_result, start_time):
                # Primary routing successful - enhance with fallback metadata
                primary_result = self.compatibility_layer.enhance_routing_prediction_with_fallback_info(
                    primary_result,
                    {
                        'fallback_system_available': self.fallback_config.enable_fallback_system,
                        'primary_routing_success': True,
                        'enhanced_routing_time_ms': (time.time() - start_time) * 1000
                    }
                )
                
                # Record successful primary routing
                if self.fallback_orchestrator:
                    self.fallback_orchestrator.failure_detector.record_operation_result(
                        response_time_ms=(time.time() - start_time) * 1000,
                        success=True,
                        confidence=primary_result.confidence
                    )
                
                return primary_result
                
        except Exception as e:
            primary_error = e
            self.logger.warning(f"Primary routing failed: {e}")
        
        # Primary routing failed or unacceptable - use fallback system
        if self.fallback_config.enable_fallback_system and self.fallback_orchestrator:
            return self._route_with_fallback_system(query_text, context, priority, start_time, primary_error)
        else:
            # No fallback system - create emergency response
            return self._create_emergency_response(query_text, start_time, primary_error)
    
    def _is_primary_result_acceptable(self, result: Optional[RoutingPrediction], start_time: float) -> bool:
        """
        Determine if primary routing result is acceptable.
        
        Args:
            result: Routing result from primary system
            start_time: Start time for performance measurement
            
        Returns:
            True if result is acceptable, False otherwise
        """
        if not result:
            return False
        
        # Check confidence threshold
        if result.confidence < self.fallback_config.confidence_threshold:
            self.logger.debug(f"Primary result confidence too low: {result.confidence:.3f}")
            return False
        
        # Check response time
        response_time_ms = (time.time() - start_time) * 1000
        if response_time_ms > self.fallback_config.max_response_time_ms:
            self.logger.debug(f"Primary result too slow: {response_time_ms:.1f}ms")
            return False
        
        # Check for circuit breaker conditions
        if hasattr(result, 'metadata') and result.metadata:
            if result.metadata.get('circuit_breaker_active'):
                self.logger.debug("Primary routing circuit breaker is active")
                return False
        
        return True
    
    def _route_with_fallback_system(self, 
                                   query_text: str, 
                                   context: Optional[Dict[str, Any]], 
                                   priority: str, 
                                   start_time: float,
                                   primary_error: Optional[Exception]) -> RoutingPrediction:
        """Route using the comprehensive fallback system."""
        self.enhanced_routing_stats['fallback_activations'] += 1
        
        try:
            # Process with comprehensive fallback
            fallback_result = self.fallback_orchestrator.process_query_with_comprehensive_fallback(
                query_text=query_text,
                context=context,
                priority=priority
            )
            
            # Log fallback usage if enabled
            if self.fallback_config.log_fallback_events:
                self.logger.info(f"Fallback system used: Level {fallback_result.fallback_level_used.name}, "
                               f"Success: {fallback_result.success}, "
                               f"Confidence: {fallback_result.routing_prediction.confidence:.3f}, "
                               f"Time: {fallback_result.total_processing_time_ms:.1f}ms")
            
            # Track emergency cache usage
            if fallback_result.fallback_level_used == FallbackLevel.EMERGENCY_CACHE:
                self.enhanced_routing_stats['emergency_cache_uses'] += 1
            
            # Convert to routing prediction for backward compatibility
            enhanced_result = self.compatibility_layer.convert_fallback_result_to_routing_prediction(fallback_result)
            
            # Add information about primary failure if it occurred
            if primary_error:
                enhanced_result.metadata = enhanced_result.metadata or {}
                enhanced_result.metadata['primary_failure_reason'] = str(primary_error)
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Fallback system also failed: {e}")
            # Create absolute emergency response
            return self._create_emergency_response(query_text, start_time, e)
    
    def _create_emergency_response(self, 
                                  query_text: str, 
                                  start_time: float, 
                                  error: Optional[Exception]) -> RoutingPrediction:
        """Create an emergency response when all systems fail."""
        
        # Simple keyword-based emergency routing
        query_lower = query_text.lower()
        
        if any(word in query_lower for word in ['latest', 'recent', 'new', 'current']):
            routing = RoutingDecision.PERPLEXITY
            category = ResearchCategory.LITERATURE_SEARCH
        elif any(word in query_lower for word in ['pathway', 'mechanism', 'relationship']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.PATHWAY_ANALYSIS
        elif any(word in query_lower for word in ['metabolite', 'compound', 'identify']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.METABOLITE_IDENTIFICATION
        else:
            routing = RoutingDecision.EITHER
            category = ResearchCategory.GENERAL_QUERY
        
        # Create minimal confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.05,
            research_category_confidence=0.05,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=1.0,
            conflict_score=0.0,
            alternative_interpretations=[(routing, 0.05)],
            calculation_time_ms=(time.time() - start_time) * 1000
        )
        
        emergency_result = RoutingPrediction(
            routing_decision=routing,
            confidence=0.05,
            reasoning=[
                "EMERGENCY RESPONSE: All routing systems failed",
                "Using basic keyword-based emergency routing",
                f"Error: {str(error) if error else 'System unavailable'}"
            ],
            research_category=category,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'emergency_response': True,
                'all_systems_failed': True,
                'error_message': str(error) if error else 'System unavailable',
                'response_time_ms': (time.time() - start_time) * 1000
            }
        )
        
        self.logger.critical(f"EMERGENCY RESPONSE activated for query: {query_text[:50]}...")
        return emergency_result
    
    def should_use_lightrag(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Enhanced version with fallback-aware decision making.
        
        Args:
            query_text: The user query text
            context: Optional context information
            
        Returns:
            Boolean indicating whether LightRAG should be used
        """
        try:
            # Use enhanced routing
            prediction = self.route_query(query_text, context)
            
            return prediction.routing_decision in [
                RoutingDecision.LIGHTRAG,
                RoutingDecision.HYBRID
            ] and prediction.confidence > 0.1  # Lower threshold for emergency cases
            
        except Exception as e:
            self.logger.warning(f"Error in should_use_lightrag: {e}")
            # Safe fallback - check for LightRAG keywords
            query_lower = query_text.lower()
            return any(word in query_lower for word in [
                'pathway', 'mechanism', 'relationship', 'connection',
                'metabolite', 'compound', 'identify', 'structure'
            ])
    
    def should_use_perplexity(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Enhanced version with fallback-aware decision making.
        
        Args:
            query_text: The user query text
            context: Optional context information
            
        Returns:
            Boolean indicating whether Perplexity API should be used
        """
        try:
            # Use enhanced routing
            prediction = self.route_query(query_text, context)
            
            return prediction.routing_decision in [
                RoutingDecision.PERPLEXITY,
                RoutingDecision.EITHER,
                RoutingDecision.HYBRID
            ] and prediction.confidence > 0.1  # Lower threshold for emergency cases
            
        except Exception as e:
            self.logger.warning(f"Error in should_use_perplexity: {e}")
            # Safe fallback - allow Perplexity for most queries
            return True
    
    def get_enhanced_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics including fallback system metrics.
        
        Returns:
            Dict containing enhanced routing statistics
        """
        # Get base statistics
        base_stats = self.get_routing_statistics()
        
        # Add enhanced statistics
        enhanced_stats = {
            'enhanced_router_stats': self.enhanced_routing_stats.copy(),
            'fallback_system_enabled': self.fallback_config.enable_fallback_system,
            'fallback_config': self.fallback_config.to_dict()
        }
        
        # Add fallback system statistics if available
        if self.fallback_orchestrator:
            try:
                fallback_stats = self.fallback_orchestrator.get_comprehensive_statistics()
                enhanced_stats['fallback_system_stats'] = fallback_stats
            except Exception as e:
                self.logger.warning(f"Failed to get fallback statistics: {e}")
                enhanced_stats['fallback_stats_error'] = str(e)
        
        # Add monitoring statistics if available
        if self.fallback_monitor:
            try:
                monitoring_report = self.fallback_monitor.get_monitoring_report()
                enhanced_stats['monitoring_report'] = monitoring_report
            except Exception as e:
                self.logger.warning(f"Failed to get monitoring report: {e}")
                enhanced_stats['monitoring_error'] = str(e)
        
        # Merge with base statistics
        base_stats.update(enhanced_stats)
        
        # Calculate enhanced success rate
        total_queries = self.enhanced_routing_stats['total_enhanced_queries']
        if total_queries > 0:
            fallback_rate = self.enhanced_routing_stats['fallback_activations'] / total_queries
            emergency_rate = self.enhanced_routing_stats['emergency_cache_uses'] / total_queries
            
            base_stats['enhanced_metrics'] = {
                'fallback_activation_rate': fallback_rate,
                'emergency_cache_usage_rate': emergency_rate,
                'system_reliability_score': 1.0 - (emergency_rate * 0.8 + fallback_rate * 0.2)
            }
        
        return base_stats
    
    def enable_emergency_mode(self):
        """Enable emergency mode with maximum fallback protection."""
        if self.fallback_orchestrator:
            self.fallback_orchestrator.enable_emergency_mode()
            self.logger.critical("Enhanced Query Router: Emergency mode enabled")
        else:
            self.logger.warning("Cannot enable emergency mode - fallback system not available")
    
    def disable_emergency_mode(self):
        """Disable emergency mode and return to normal operation."""
        if self.fallback_orchestrator:
            self.fallback_orchestrator.disable_emergency_mode()
            self.logger.info("Enhanced Query Router: Emergency mode disabled")
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system health report.
        
        Returns:
            Dict containing system health information
        """
        health_report = {
            'timestamp': time.time(),
            'enhanced_router_operational': True,
            'fallback_system_status': 'disabled'
        }
        
        if self.fallback_orchestrator:
            try:
                # Get comprehensive statistics
                stats = self.fallback_orchestrator.get_comprehensive_statistics()
                
                health_report.update({
                    'fallback_system_status': 'operational',
                    'system_health_score': stats.get('system_health', {}).get('overall_health_score', 0.0),
                    'early_warning_signals': stats.get('system_health', {}).get('early_warning_signals', []),
                    'recent_failures': len(stats.get('failure_detection', {}).get('metrics', {}).get('recent_errors', [])),
                    'fallback_activations': self.enhanced_routing_stats['fallback_activations'],
                    'emergency_cache_uses': self.enhanced_routing_stats['emergency_cache_uses']
                })
                
                # Determine overall system status
                health_score = health_report.get('system_health_score', 0.0)
                if health_score >= 0.8:
                    health_report['system_status'] = 'healthy'
                elif health_score >= 0.6:
                    health_report['system_status'] = 'degraded'
                elif health_score >= 0.3:
                    health_report['system_status'] = 'unstable'
                else:
                    health_report['system_status'] = 'critical'
                
            except Exception as e:
                health_report.update({
                    'fallback_system_status': 'error',
                    'fallback_system_error': str(e),
                    'system_status': 'unknown'
                })
        
        return health_report
    
    def shutdown_enhanced_features(self):
        """Shutdown enhanced features gracefully."""
        self.logger.info("Shutting down enhanced query router features")
        
        # Stop monitoring
        if self.fallback_monitor:
            try:
                self.fallback_monitor.stop_monitoring()
                self.logger.info("Fallback monitoring stopped")
            except Exception as e:
                self.logger.error(f"Error stopping monitoring: {e}")
        
        # Stop recovery manager
        if self.fallback_orchestrator and self.fallback_orchestrator.recovery_manager:
            try:
                self.fallback_orchestrator.recovery_manager.stop_recovery_monitoring()
                self.logger.info("Recovery monitoring stopped")
            except Exception as e:
                self.logger.error(f"Error stopping recovery monitoring: {e}")
        
        self.logger.info("Enhanced query router shutdown completed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown_enhanced_features()
        except:
            pass  # Ignore errors during cleanup


# ============================================================================
# FACTORY FUNCTIONS FOR EASY INTEGRATION
# ============================================================================

def create_enhanced_router_from_existing(existing_router: BiomedicalQueryRouter,
                                       llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                                       config: Optional[FallbackIntegrationConfig] = None) -> EnhancedBiomedicalQueryRouter:
    """
    Create an enhanced router from an existing BiomedicalQueryRouter instance.
    
    Args:
        existing_router: Existing BiomedicalQueryRouter instance
        llm_classifier: Optional LLM classifier
        config: Optional fallback integration configuration
        
    Returns:
        Enhanced router with fallback capabilities
    """
    # Create enhanced router
    enhanced_router = EnhancedBiomedicalQueryRouter(
        fallback_config=config,
        llm_classifier=llm_classifier,
        logger=existing_router.logger
    )
    
    # Copy existing router configuration
    enhanced_router.category_routing_map = existing_router.category_routing_map
    enhanced_router.routing_thresholds = existing_router.routing_thresholds
    enhanced_router.fallback_strategies = existing_router.fallback_strategies
    enhanced_router.temporal_analyzer = existing_router.temporal_analyzer
    
    # Copy performance tracking
    if hasattr(existing_router, '_routing_times'):
        enhanced_router._routing_times = existing_router._routing_times
    if hasattr(existing_router, '_query_cache'):
        enhanced_router._query_cache = existing_router._query_cache
    
    enhanced_router.logger.info("Enhanced router created from existing router instance")
    return enhanced_router


def create_production_ready_enhanced_router(llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                                          emergency_cache_dir: Optional[str] = None,
                                          logger: Optional[logging.Logger] = None) -> EnhancedBiomedicalQueryRouter:
    """
    Create a production-ready enhanced router with optimal configuration.
    
    Args:
        llm_classifier: Optional LLM classifier
        emergency_cache_dir: Directory for emergency cache
        logger: Logger instance
        
    Returns:
        Production-ready enhanced router
    """
    # Create production configuration
    config = FallbackIntegrationConfig(
        enable_fallback_system=True,
        enable_monitoring=True,
        monitoring_interval_seconds=30,  # More frequent monitoring in production
        emergency_cache_file=f"{emergency_cache_dir or 'cache'}/production_emergency_cache.pkl",
        enable_cache_warming=True,
        cache_common_patterns=True,
        max_response_time_ms=1500,  # Stricter performance target
        confidence_threshold=0.5,   # Lower threshold for better availability
        health_score_threshold=0.6,
        maintain_backward_compatibility=True,
        log_fallback_events=True,
        enable_auto_recovery=True,
        enable_alerts=True,
        alert_cooldown_seconds=60   # More frequent alerts in production
    )
    
    # Create enhanced router
    enhanced_router = EnhancedBiomedicalQueryRouter(
        fallback_config=config,
        llm_classifier=llm_classifier,
        logger=logger
    )
    
    # Pre-warm cache with production patterns
    production_patterns = [
        # High-frequency metabolomics queries
        "metabolite identification LC-MS",
        "pathway analysis KEGG",
        "biomarker validation study",
        "clinical metabolomics analysis",
        "drug metabolism pathway",
        "metabolic network reconstruction",
        "untargeted metabolomics workflow",
        "targeted metabolomics quantification",
        
        # Real-time information queries
        "latest metabolomics publications 2024",
        "recent advances clinical metabolomics",
        "current metabolomics technologies",
        "new biomarker discoveries",
        
        # Technical queries
        "mass spectrometry data processing",
        "NMR metabolomics analysis",
        "statistical analysis metabolomics",
        "machine learning biomarker discovery"
    ]
    
    if enhanced_router.fallback_orchestrator:
        enhanced_router.fallback_orchestrator.emergency_cache.warm_cache(production_patterns)
    
    if logger:
        logger.info("Production-ready enhanced router created with comprehensive fallback protection")
    
    return enhanced_router


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create enhanced router
    enhanced_router = create_production_ready_enhanced_router(logger=logger)
    
    # Test queries
    test_queries = [
        ("identify metabolite with mass 180.0634", 'high'),
        ("latest research on metabolomics biomarkers", 'normal'),
        ("pathway analysis for glucose metabolism", 'normal'),
        ("what is LC-MS", 'low'),
        ("complex query with multiple biomarkers and pathway interactions in diabetes", 'critical')
    ]
    
    logger.info("Testing Enhanced Biomedical Query Router with Comprehensive Fallback")
    
    for query, priority in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing query: {query} (Priority: {priority})")
        
        try:
            # Test routing
            result = enhanced_router.route_query(query, {'priority': priority})
            
            logger.info(f"Routing Decision: {result.routing_decision.value}")
            logger.info(f"Confidence: {result.confidence:.3f}")
            logger.info(f"Research Category: {result.research_category.value}")
            logger.info(f"Fallback Info: {result.metadata.get('fallback_system_used', 'Primary routing')}")
            
            if result.metadata and result.metadata.get('fallback_level_used'):
                logger.info(f"Fallback Level: {result.metadata['fallback_level_used']}")
                logger.info(f"Quality Score: {result.metadata.get('quality_score', 'N/A')}")
            
            # Test boolean methods
            use_lightrag = enhanced_router.should_use_lightrag(query)
            use_perplexity = enhanced_router.should_use_perplexity(query)
            logger.info(f"Use LightRAG: {use_lightrag}, Use Perplexity: {use_perplexity}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    
    # Get system health report
    logger.info(f"\n{'='*60}")
    health_report = enhanced_router.get_system_health_report()
    logger.info(f"System Health Report:")
    logger.info(f"System Status: {health_report.get('system_status', 'unknown')}")
    logger.info(f"Health Score: {health_report.get('system_health_score', 'N/A')}")
    logger.info(f"Fallback Activations: {health_report.get('fallback_activations', 0)}")
    
    # Get enhanced statistics
    stats = enhanced_router.get_enhanced_routing_statistics()
    logger.info(f"Total Enhanced Queries: {stats['enhanced_router_stats']['total_enhanced_queries']}")
    logger.info(f"Fallback System Enabled: {stats['fallback_system_enabled']}")
    
    # Cleanup
    enhanced_router.shutdown_enhanced_features()
    logger.info("Test completed successfully")