"""
Enhanced Query Router Integration with Comprehensive Confidence Scoring

This module provides an integration layer that enhances the existing BiomedicalQueryRouter
with the comprehensive confidence scoring system, maintaining full backward compatibility
while adding advanced confidence analysis capabilities.

Key Features:
    - Seamless integration with existing BiomedicalQueryRouter infrastructure
    - Enhanced RoutingPrediction with comprehensive confidence analysis
    - Backward compatibility with existing ConfidenceMetrics structure
    - Advanced LLM + keyword confidence fusion
    - Real-time confidence calibration and validation
    - Performance monitoring and optimization recommendations
    - Support for confidence interval estimation and uncertainty quantification

Classes:
    - EnhancedBiomedicalQueryRouter: Extended router with comprehensive confidence scoring
    - EnhancedRoutingPrediction: Extended prediction with detailed confidence analysis
    - ConfidenceIntegrationManager: Manages integration between systems
    - PerformanceOptimizer: Optimizes confidence calculation performance

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading

# Import existing components
try:
    from .query_router import (
        BiomedicalQueryRouter, RoutingPrediction, RoutingDecision, 
        ConfidenceMetrics, FallbackStrategy, TemporalAnalyzer
    )
    from .research_categorizer import CategoryPrediction, ResearchCategorizer
    from .enhanced_llm_classifier import EnhancedLLMQueryClassifier, ClassificationResult
    from .cost_persistence import ResearchCategory
    from .comprehensive_confidence_scorer import (
        HybridConfidenceScorer, ConfidenceValidator, HybridConfidenceResult,
        create_hybrid_confidence_scorer, integrate_with_existing_confidence_metrics
    )
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# ENHANCED ROUTING PREDICTION WITH COMPREHENSIVE CONFIDENCE
# ============================================================================

@dataclass
class EnhancedRoutingPrediction(RoutingPrediction):
    """
    Enhanced routing prediction with comprehensive confidence analysis.
    Extends the existing RoutingPrediction with advanced confidence metrics.
    """
    
    # Enhanced confidence analysis
    comprehensive_confidence: Optional['HybridConfidenceResult'] = None
    confidence_reliability_score: float = 0.0
    confidence_calibration_status: str = "unknown"
    
    # Advanced uncertainty metrics
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Data uncertainty
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    # LLM integration
    llm_classification_result: Optional[ClassificationResult] = None
    llm_confidence_contribution: float = 0.0
    keyword_confidence_contribution: float = 0.0
    
    # Performance metrics
    confidence_calculation_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of confidence analysis."""
        return {
            'overall_confidence': self.confidence,
            'confidence_level': self.confidence_level,
            'confidence_interval': self.confidence_interval,
            'reliability_score': self.confidence_reliability_score,
            'uncertainty_metrics': {
                'epistemic': self.epistemic_uncertainty,
                'aleatoric': self.aleatoric_uncertainty,
                'total': self.epistemic_uncertainty + self.aleatoric_uncertainty * 0.5
            },
            'source_contributions': {
                'llm': self.llm_confidence_contribution,
                'keyword': self.keyword_confidence_contribution
            },
            'calibration_status': self.confidence_calibration_status,
            'calculation_performance': {
                'confidence_time_ms': self.confidence_calculation_time_ms,
                'total_time_ms': self.total_processing_time_ms
            }
        }
    
    def is_high_quality_confidence(self) -> bool:
        """Determine if this is a high-quality confidence prediction."""
        return (
            self.confidence_reliability_score >= 0.7 and
            self.epistemic_uncertainty <= 0.3 and
            self.confidence_calculation_time_ms <= 100  # Reasonable performance
        )


# ============================================================================
# CONFIDENCE INTEGRATION MANAGER
# ============================================================================

class ConfidenceIntegrationManager:
    """
    Manages integration between existing confidence systems and comprehensive scoring.
    """
    
    def __init__(self, 
                 hybrid_scorer: HybridConfidenceScorer,
                 validator: Optional[ConfidenceValidator] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.hybrid_scorer = hybrid_scorer
        self.validator = validator
        self.logger = logger or logging.getLogger(__name__)
        
        # Integration configuration
        self.config = {
            'enable_llm_confidence': True,
            'enable_confidence_intervals': True,
            'enable_uncertainty_quantification': True,
            'enable_real_time_calibration': True,
            'fallback_to_keyword_only': True,
            'performance_target_ms': 150  # Target for comprehensive confidence calculation
        }
        
        # Performance tracking
        self.integration_stats = {
            'total_integrations': 0,
            'llm_successes': 0,
            'fallback_uses': 0,
            'average_processing_time_ms': 0.0
        }
        
        # Thread safety
        self._stats_lock = threading.Lock()
        
        self.logger.info("Confidence integration manager initialized")
    
    async def integrate_comprehensive_confidence(self, 
                                               query_text: str,
                                               base_routing_prediction: RoutingPrediction,
                                               llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                                               context: Optional[Dict[str, Any]] = None) -> EnhancedRoutingPrediction:
        """
        Integrate comprehensive confidence scoring with base routing prediction.
        
        Args:
            query_text: Original query text
            base_routing_prediction: Base routing prediction from existing system
            llm_classifier: Optional LLM classifier for enhanced analysis
            context: Additional context information
            
        Returns:
            EnhancedRoutingPrediction with comprehensive confidence analysis
        """
        
        start_time = time.time()
        integration_start = start_time
        
        try:
            # Extract components for comprehensive analysis
            keyword_prediction = CategoryPrediction(
                category=base_routing_prediction.research_category,
                confidence=base_routing_prediction.confidence,
                evidence=base_routing_prediction.knowledge_indicators or []
            )
            
            # Get LLM classification if available
            llm_result = None
            llm_response_metadata = None
            
            if self.config['enable_llm_confidence'] and llm_classifier:
                try:
                    llm_result, llm_response_metadata = await llm_classifier.classify_query(
                        query_text, context, priority="normal"
                    )
                    with self._stats_lock:
                        self.integration_stats['llm_successes'] += 1
                        
                except Exception as e:
                    self.logger.warning(f"LLM classification failed during integration: {e}")
                    if not self.config['fallback_to_keyword_only']:
                        raise
            
            # Calculate comprehensive confidence
            confidence_start = time.time()
            hybrid_result = await self.hybrid_scorer.calculate_comprehensive_confidence(
                query_text=query_text,
                llm_result=llm_result,
                keyword_prediction=keyword_prediction,
                context=context,
                llm_response_metadata=llm_response_metadata
            )
            confidence_time = (time.time() - confidence_start) * 1000
            
            # Create enhanced routing prediction
            enhanced_prediction = self._create_enhanced_prediction(
                base_routing_prediction, hybrid_result, llm_result, 
                confidence_time, integration_start
            )
            
            # Update integration statistics
            total_time = (time.time() - integration_start) * 1000
            self._update_integration_stats(total_time)
            
            # Log performance warning if needed
            if total_time > self.config['performance_target_ms']:
                self.logger.warning(f"Confidence integration took {total_time:.2f}ms "
                                  f"(target: {self.config['performance_target_ms']}ms)")
            
            self.logger.debug(f"Comprehensive confidence integration completed in {total_time:.2f}ms")
            
            return enhanced_prediction
            
        except Exception as e:
            # Fallback to base prediction with minimal enhancements
            self.logger.error(f"Confidence integration failed: {e}")
            with self._stats_lock:
                self.integration_stats['fallback_uses'] += 1
            
            return self._create_fallback_enhanced_prediction(
                base_routing_prediction, integration_start
            )
    
    def _create_enhanced_prediction(self, 
                                  base_prediction: RoutingPrediction,
                                  hybrid_result: HybridConfidenceResult,
                                  llm_result: Optional[ClassificationResult],
                                  confidence_time: float,
                                  integration_start: float) -> EnhancedRoutingPrediction:
        """Create enhanced routing prediction from analysis results."""
        
        # Calculate contribution weights
        llm_contribution = hybrid_result.llm_weight
        keyword_contribution = hybrid_result.keyword_weight
        
        # Determine calibration status
        calibration_status = "calibrated" if hybrid_result.calibration_adjustment != 0.0 else "uncalibrated"
        if abs(hybrid_result.calibration_adjustment) > 0.1:
            calibration_status = "significant_adjustment"
        
        # Create enhanced prediction
        enhanced = EnhancedRoutingPrediction(
            # Base fields (from RoutingPrediction)
            routing_decision=base_prediction.routing_decision,
            confidence=hybrid_result.overall_confidence,  # Use comprehensive confidence
            reasoning=base_prediction.reasoning + [
                f"Enhanced with comprehensive confidence analysis",
                f"LLM contribution: {llm_contribution:.1%}, Keyword contribution: {keyword_contribution:.1%}",
                f"Confidence reliability: {hybrid_result.confidence_reliability:.3f}"
            ],
            research_category=base_prediction.research_category,
            confidence_metrics=integrate_with_existing_confidence_metrics(hybrid_result, ""),
            fallback_strategy=base_prediction.fallback_strategy,
            temporal_indicators=base_prediction.temporal_indicators,
            knowledge_indicators=base_prediction.knowledge_indicators,
            metadata=base_prediction.metadata or {},
            
            # Enhanced fields
            comprehensive_confidence=hybrid_result,
            confidence_reliability_score=hybrid_result.confidence_reliability,
            confidence_calibration_status=calibration_status,
            epistemic_uncertainty=hybrid_result.epistemic_uncertainty,
            aleatoric_uncertainty=hybrid_result.aleatoric_uncertainty,
            confidence_interval=hybrid_result.confidence_interval,
            llm_classification_result=llm_result,
            llm_confidence_contribution=llm_contribution,
            keyword_confidence_contribution=keyword_contribution,
            confidence_calculation_time_ms=confidence_time,
            total_processing_time_ms=(time.time() - integration_start) * 1000
        )
        
        # Add enhanced metadata
        enhanced.metadata.update({
            'comprehensive_confidence_enabled': True,
            'hybrid_confidence_version': hybrid_result.calibration_version,
            'evidence_strength': hybrid_result.evidence_strength,
            'uncertainty_quantification': {
                'epistemic': hybrid_result.epistemic_uncertainty,
                'aleatoric': hybrid_result.aleatoric_uncertainty,
                'total': hybrid_result.total_uncertainty
            },
            'alternative_confidences': hybrid_result.alternative_confidences
        })
        
        return enhanced
    
    def _create_fallback_enhanced_prediction(self, 
                                           base_prediction: RoutingPrediction,
                                           integration_start: float) -> EnhancedRoutingPrediction:
        """Create fallback enhanced prediction when comprehensive analysis fails."""
        
        enhanced = EnhancedRoutingPrediction(
            # Copy base fields
            routing_decision=base_prediction.routing_decision,
            confidence=base_prediction.confidence,
            reasoning=base_prediction.reasoning + ["Used fallback confidence (comprehensive analysis failed)"],
            research_category=base_prediction.research_category,
            confidence_metrics=base_prediction.confidence_metrics,
            fallback_strategy=base_prediction.fallback_strategy,
            temporal_indicators=base_prediction.temporal_indicators,
            knowledge_indicators=base_prediction.knowledge_indicators,
            metadata=(base_prediction.metadata or {}).copy(),
            
            # Fallback enhanced fields
            comprehensive_confidence=None,
            confidence_reliability_score=base_prediction.confidence * 0.8,  # Reduced reliability
            confidence_calibration_status="fallback",
            epistemic_uncertainty=0.3,  # Default uncertainty
            aleatoric_uncertainty=0.2,
            confidence_interval=(max(0.0, base_prediction.confidence - 0.2), 
                               min(1.0, base_prediction.confidence + 0.2)),
            llm_classification_result=None,
            llm_confidence_contribution=0.0,
            keyword_confidence_contribution=1.0,  # Full keyword contribution
            confidence_calculation_time_ms=0.0,
            total_processing_time_ms=(time.time() - integration_start) * 1000
        )
        
        # Update metadata
        enhanced.metadata.update({
            'comprehensive_confidence_enabled': False,
            'fallback_reason': 'comprehensive_analysis_failed',
            'confidence_source': 'keyword_only'
        })
        
        return enhanced
    
    def _update_integration_stats(self, processing_time_ms: float):
        """Update integration performance statistics."""
        with self._stats_lock:
            self.integration_stats['total_integrations'] += 1
            
            # Update moving average of processing time
            current_avg = self.integration_stats['average_processing_time_ms']
            total_count = self.integration_stats['total_integrations']
            
            self.integration_stats['average_processing_time_ms'] = (
                (current_avg * (total_count - 1) + processing_time_ms) / total_count
            )
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        with self._stats_lock:
            stats = self.integration_stats.copy()
        
        # Calculate derived metrics
        if stats['total_integrations'] > 0:
            stats['llm_success_rate'] = stats['llm_successes'] / stats['total_integrations']
            stats['fallback_rate'] = stats['fallback_uses'] / stats['total_integrations']
        else:
            stats['llm_success_rate'] = 0.0
            stats['fallback_rate'] = 0.0
        
        stats['configuration'] = self.config.copy()
        return stats


# ============================================================================
# ENHANCED BIOMEDICAL QUERY ROUTER
# ============================================================================

class EnhancedBiomedicalQueryRouter(BiomedicalQueryRouter):
    """
    Enhanced biomedical query router with comprehensive confidence scoring.
    Extends the existing BiomedicalQueryRouter while maintaining full backward compatibility.
    """
    
    def __init__(self, 
                 llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                 enable_comprehensive_confidence: bool = True,
                 calibration_data_path: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced biomedical query router.
        
        Args:
            llm_classifier: Optional LLM classifier for enhanced analysis
            enable_comprehensive_confidence: Whether to enable comprehensive confidence scoring
            calibration_data_path: Path for confidence calibration data
            logger: Logger instance
        """
        
        # Initialize base router
        super().__init__(logger)
        
        self.llm_classifier = llm_classifier
        self.enable_comprehensive_confidence = enable_comprehensive_confidence
        
        # Initialize comprehensive confidence components
        if enable_comprehensive_confidence:
            self.hybrid_scorer = create_hybrid_confidence_scorer(
                biomedical_router=self,
                llm_classifier=llm_classifier,
                calibration_data_path=calibration_data_path,
                logger=self.logger
            )
            
            self.confidence_validator = ConfidenceValidator(
                self.hybrid_scorer, self.logger
            )
            
            self.integration_manager = ConfidenceIntegrationManager(
                self.hybrid_scorer, self.confidence_validator, self.logger
            )
        else:
            self.hybrid_scorer = None
            self.confidence_validator = None
            self.integration_manager = None
        
        # Enhanced configuration
        self.enhanced_config = {
            'comprehensive_confidence_enabled': enable_comprehensive_confidence,
            'llm_integration_enabled': llm_classifier is not None,
            'confidence_calibration_enabled': calibration_data_path is not None,
            'performance_monitoring_enabled': True,
            'auto_optimization_enabled': True
        }
        
        # Performance tracking for enhanced features
        self.enhanced_stats = {
            'enhanced_routes': 0,
            'comprehensive_confidence_calculations': 0,
            'llm_classifications': 0,
            'confidence_validations': 0,
            'fallback_routes': 0
        }
        
        self.logger.info(f"Enhanced biomedical query router initialized - "
                        f"Comprehensive confidence: {enable_comprehensive_confidence}")
    
    async def route_query_enhanced(self, 
                                 query_text: str,
                                 context: Optional[Dict[str, Any]] = None,
                                 enable_llm: bool = True,
                                 return_comprehensive_analysis: bool = True) -> EnhancedRoutingPrediction:
        """
        Route query with enhanced comprehensive confidence analysis.
        
        Args:
            query_text: The user query text to route
            context: Optional context information
            enable_llm: Whether to use LLM classification (if available)
            return_comprehensive_analysis: Whether to include comprehensive confidence analysis
            
        Returns:
            EnhancedRoutingPrediction with detailed confidence analysis
        """
        
        start_time = time.time()
        self.enhanced_stats['enhanced_routes'] += 1
        
        try:
            # Get base routing prediction
            base_prediction = self.route_query(query_text, context)
            
            # If comprehensive confidence is disabled, return basic enhanced prediction
            if not self.enable_comprehensive_confidence or not return_comprehensive_analysis:
                return self._create_basic_enhanced_prediction(base_prediction, start_time)
            
            # Use LLM classifier if available and enabled
            llm_classifier = self.llm_classifier if enable_llm else None
            
            # Integrate comprehensive confidence
            enhanced_prediction = await self.integration_manager.integrate_comprehensive_confidence(
                query_text=query_text,
                base_routing_prediction=base_prediction,
                llm_classifier=llm_classifier,
                context=context
            )
            
            self.enhanced_stats['comprehensive_confidence_calculations'] += 1
            
            if enhanced_prediction.llm_classification_result:
                self.enhanced_stats['llm_classifications'] += 1
            
            # Log performance
            total_time = (time.time() - start_time) * 1000
            if total_time > 200:  # Log if taking longer than 200ms
                self.logger.warning(f"Enhanced routing took {total_time:.2f}ms for query: {query_text[:50]}...")
            
            self.logger.debug(f"Enhanced query routing completed - "
                            f"Confidence: {enhanced_prediction.confidence:.3f}, "
                            f"Reliability: {enhanced_prediction.confidence_reliability_score:.3f}")
            
            return enhanced_prediction
            
        except Exception as e:
            self.logger.error(f"Enhanced query routing failed: {e}")
            self.enhanced_stats['fallback_routes'] += 1
            
            # Fallback to base prediction
            base_prediction = self.route_query(query_text, context)
            return self._create_basic_enhanced_prediction(base_prediction, start_time, error=str(e))
    
    def route_query(self, 
                   query_text: str,
                   context: Optional[Dict[str, Any]] = None) -> RoutingPrediction:
        """
        Override base route_query to maintain backward compatibility while adding enhancements.
        This method preserves the original interface while optionally adding enhanced features.
        """
        
        # Call parent implementation for base routing
        base_prediction = super().route_query(query_text, context)
        
        # If comprehensive confidence is disabled, return base prediction
        if not self.enable_comprehensive_confidence:
            return base_prediction
        
        # For synchronous calls, provide enhanced confidence if possible
        try:
            # Quick synchronous enhancement (no LLM to maintain performance)
            if self.hybrid_scorer:
                # Use keyword-only comprehensive confidence for backward compatibility
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    keyword_prediction = CategoryPrediction(
                        category=base_prediction.research_category,
                        confidence=base_prediction.confidence,
                        evidence=base_prediction.knowledge_indicators or []
                    )
                    
                    hybrid_result = loop.run_until_complete(
                        self.hybrid_scorer.calculate_comprehensive_confidence(
                            query_text=query_text,
                            llm_result=None,  # No LLM for sync calls
                            keyword_prediction=keyword_prediction,
                            context=context
                        )
                    )
                    
                    # Update confidence metrics with comprehensive analysis
                    base_prediction.confidence = hybrid_result.overall_confidence
                    base_prediction.confidence_metrics = integrate_with_existing_confidence_metrics(
                        hybrid_result, query_text
                    )
                    
                finally:
                    loop.close()
                    
        except Exception as e:
            self.logger.debug(f"Could not enhance base prediction: {e}")
        
        return base_prediction
    
    def validate_routing_accuracy(self, 
                                query_text: str,
                                predicted_routing: RoutingDecision,
                                predicted_confidence: float,
                                actual_accuracy: bool,
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate routing accuracy for confidence calibration.
        
        Args:
            query_text: Original query text
            predicted_routing: The routing decision that was made
            predicted_confidence: The confidence that was predicted
            actual_accuracy: Whether the routing was actually accurate
            context: Additional context information
            
        Returns:
            Validation results and recommendations
        """
        
        if not self.confidence_validator:
            return {'error': 'Confidence validation not available'}
        
        validation_result = self.confidence_validator.validate_confidence_accuracy(
            query_text=query_text,
            predicted_confidence=predicted_confidence,
            actual_routing_accuracy=actual_accuracy,
            context=context
        )
        
        self.enhanced_stats['confidence_validations'] += 1
        
        return validation_result
    
    def get_confidence_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive confidence validation report."""
        
        if not self.confidence_validator:
            return {'error': 'Confidence validation not available'}
        
        return self.confidence_validator.get_validation_report()
    
    def _create_basic_enhanced_prediction(self, 
                                        base_prediction: RoutingPrediction,
                                        start_time: float,
                                        error: Optional[str] = None) -> EnhancedRoutingPrediction:
        """Create basic enhanced prediction from base prediction."""
        
        enhanced = EnhancedRoutingPrediction(
            # Copy all base fields
            routing_decision=base_prediction.routing_decision,
            confidence=base_prediction.confidence,
            reasoning=base_prediction.reasoning + (["Enhanced routing (basic mode)"] if not error 
                     else [f"Enhanced routing failed: {error}"]),
            research_category=base_prediction.research_category,
            confidence_metrics=base_prediction.confidence_metrics,
            fallback_strategy=base_prediction.fallback_strategy,
            temporal_indicators=base_prediction.temporal_indicators,
            knowledge_indicators=base_prediction.knowledge_indicators,
            metadata=(base_prediction.metadata or {}).copy(),
            
            # Basic enhanced fields
            comprehensive_confidence=None,
            confidence_reliability_score=base_prediction.confidence * 0.9,  # Slight reduction for basic mode
            confidence_calibration_status="basic" if not error else "error",
            epistemic_uncertainty=0.2,
            aleatoric_uncertainty=0.1,
            confidence_interval=(max(0.0, base_prediction.confidence - 0.15), 
                               min(1.0, base_prediction.confidence + 0.15)),
            llm_classification_result=None,
            llm_confidence_contribution=0.0,
            keyword_confidence_contribution=1.0,
            confidence_calculation_time_ms=0.0,
            total_processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Update metadata
        enhanced.metadata.update({
            'comprehensive_confidence_enabled': False,
            'enhancement_mode': 'basic',
            'error': error
        })
        
        return enhanced
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced router statistics."""
        
        base_stats = self.get_routing_statistics()
        
        enhanced_stats = {
            'enhanced_routing_stats': self.enhanced_stats.copy(),
            'enhanced_configuration': self.enhanced_config.copy(),
            'base_routing_stats': base_stats
        }
        
        # Add integration stats if available
        if self.integration_manager:
            enhanced_stats['integration_stats'] = self.integration_manager.get_integration_stats()
        
        # Add confidence scorer stats if available
        if self.hybrid_scorer:
            enhanced_stats['confidence_scoring_stats'] = self.hybrid_scorer.get_comprehensive_stats()
        
        # Add validation stats if available
        if self.confidence_validator:
            enhanced_stats['validation_stats'] = self.confidence_validator.get_validation_report()
        
        return enhanced_stats
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize router performance based on current statistics."""
        
        optimization_results = {
            'actions_taken': [],
            'recommendations': [],
            'performance_improvements': {}
        }
        
        if not self.enable_comprehensive_confidence:
            optimization_results['recommendations'].append(
                "Enable comprehensive confidence scoring for advanced optimizations"
            )
            return optimization_results
        
        # Get current statistics
        stats = self.get_enhanced_statistics()
        
        # Optimize LLM classifier if available
        if self.llm_classifier:
            try:
                llm_optimization = await self.llm_classifier.optimize_system(auto_apply=True)
                optimization_results['actions_taken'].extend(llm_optimization['actions_taken'])
                optimization_results['recommendations'].extend(llm_optimization['recommendations_pending'])
            except Exception as e:
                self.logger.error(f"LLM optimization failed: {e}")
        
        # Check confidence scorer performance
        if self.hybrid_scorer:
            scoring_stats = stats.get('confidence_scoring_stats', {})
            avg_time = scoring_stats.get('scoring_performance', {}).get('average_scoring_time_ms', 0)
            
            if avg_time > 100:  # If confidence scoring is slow
                optimization_results['recommendations'].append({
                    'type': 'performance',
                    'issue': f'Slow confidence scoring ({avg_time:.1f}ms average)',
                    'suggestion': 'Consider disabling LLM integration for faster keyword-only confidence'
                })
        
        # Check fallback rates
        enhanced_stats = stats.get('enhanced_routing_stats', {})
        total_routes = enhanced_stats.get('enhanced_routes', 1)
        fallback_rate = enhanced_stats.get('fallback_routes', 0) / total_routes
        
        if fallback_rate > 0.1:  # More than 10% fallback rate
            optimization_results['recommendations'].append({
                'type': 'reliability',
                'issue': f'High fallback rate ({fallback_rate:.1%})',
                'suggestion': 'Review error logs and consider system stability improvements'
            })
        
        return optimization_results


# ============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ============================================================================

async def create_enhanced_biomedical_router(
    llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
    enable_comprehensive_confidence: bool = True,
    calibration_data_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedBiomedicalQueryRouter:
    """
    Factory function to create an enhanced biomedical query router.
    
    Args:
        llm_classifier: Optional LLM classifier for enhanced analysis
        enable_comprehensive_confidence: Whether to enable comprehensive confidence scoring
        calibration_data_path: Path for confidence calibration data
        logger: Logger instance
        
    Returns:
        Configured EnhancedBiomedicalQueryRouter instance
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Set default calibration path
    if calibration_data_path is None and enable_comprehensive_confidence:
        calibration_data_path = "/tmp/enhanced_router_calibration.json"
    
    router = EnhancedBiomedicalQueryRouter(
        llm_classifier=llm_classifier,
        enable_comprehensive_confidence=enable_comprehensive_confidence,
        calibration_data_path=calibration_data_path,
        logger=logger
    )
    
    logger.info("Enhanced biomedical query router created successfully")
    logger.info(f"Configuration: LLM={llm_classifier is not None}, "
               f"Comprehensive confidence={enable_comprehensive_confidence}")
    
    return router


def convert_enhanced_to_base_prediction(enhanced_prediction: EnhancedRoutingPrediction) -> RoutingPrediction:
    """
    Convert enhanced prediction back to base RoutingPrediction for backward compatibility.
    
    Args:
        enhanced_prediction: Enhanced routing prediction
        
    Returns:
        Base RoutingPrediction compatible with existing systems
    """
    
    return RoutingPrediction(
        routing_decision=enhanced_prediction.routing_decision,
        confidence=enhanced_prediction.confidence,
        reasoning=enhanced_prediction.reasoning,
        research_category=enhanced_prediction.research_category,
        confidence_metrics=enhanced_prediction.confidence_metrics,
        confidence_level=enhanced_prediction.confidence_level,
        fallback_strategy=enhanced_prediction.fallback_strategy,
        temporal_indicators=enhanced_prediction.temporal_indicators,
        knowledge_indicators=enhanced_prediction.knowledge_indicators,
        metadata=enhanced_prediction.metadata
    )


# ============================================================================
# ASYNC CONTEXT MANAGERS
# ============================================================================

import contextlib

@contextlib.asynccontextmanager
async def enhanced_router_context(
    llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
    enable_comprehensive_confidence: bool = True,
    calibration_data_path: Optional[str] = None
):
    """
    Async context manager for enhanced router with proper resource management.
    
    Usage:
        async with enhanced_router_context() as router:
            prediction = await router.route_query_enhanced("example query")
    """
    
    logger = logging.getLogger(__name__)
    router = None
    
    try:
        router = await create_enhanced_biomedical_router(
            llm_classifier=llm_classifier,
            enable_comprehensive_confidence=enable_comprehensive_confidence,
            calibration_data_path=calibration_data_path,
            logger=logger
        )
        
        logger.info("Enhanced router context initialized")
        yield router
        
    finally:
        if router:
            # Cleanup and save calibration data
            try:
                if router.hybrid_scorer and router.hybrid_scorer.calibrator:
                    router.hybrid_scorer.calibrator.save_calibration_data()
                
                stats = router.get_enhanced_statistics()
                logger.info(f"Enhanced router context cleanup - "
                           f"Total enhanced routes: {stats['enhanced_routing_stats']['enhanced_routes']}")
                logger.info(f"Comprehensive confidence calculations: "
                           f"{stats['enhanced_routing_stats']['comprehensive_confidence_calculations']}")
                           
            except Exception as e:
                logger.error(f"Error during enhanced router cleanup: {e}")


if __name__ == "__main__":
    # Example usage and integration testing
    import asyncio
    import logging
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def demo_enhanced_router():
        """Demonstrate enhanced biomedical query router."""
        
        print("=== Enhanced Biomedical Query Router Demo ===")
        
        async with enhanced_router_context(enable_comprehensive_confidence=True) as router:
            
            # Test queries with different characteristics
            test_queries = [
                "What is the relationship between glucose metabolism and insulin signaling?",
                "Latest metabolomics research on diabetes biomarkers 2025",
                "LC-MS analysis for metabolite identification",
                "pathway analysis",
                "biomarker discovery methods"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n--- Query {i}: {query} ---")
                
                try:
                    # Enhanced routing with comprehensive analysis
                    enhanced_prediction = await router.route_query_enhanced(query)
                    
                    # Display comprehensive confidence analysis
                    confidence_summary = enhanced_prediction.get_confidence_summary()
                    
                    print(f"Routing Decision: {enhanced_prediction.routing_decision.value}")
                    print(f"Overall Confidence: {confidence_summary['overall_confidence']:.3f}")
                    print(f"Confidence Level: {confidence_summary['confidence_level']}")
                    print(f"Confidence Interval: [{confidence_summary['confidence_interval'][0]:.3f}, {confidence_summary['confidence_interval'][1]:.3f}]")
                    print(f"Reliability Score: {confidence_summary['reliability_score']:.3f}")
                    print(f"LLM Contribution: {confidence_summary['source_contributions']['llm']:.1%}")
                    print(f"Keyword Contribution: {confidence_summary['source_contributions']['keyword']:.1%}")
                    print(f"Processing Time: {confidence_summary['calculation_performance']['total_time_ms']:.2f}ms")
                    print(f"High Quality Confidence: {enhanced_prediction.is_high_quality_confidence()}")
                    
                    # Test backward compatibility
                    base_prediction = convert_enhanced_to_base_prediction(enhanced_prediction)
                    print(f"Backward Compatible Confidence: {base_prediction.confidence:.3f}")
                    
                except Exception as e:
                    print(f"Error processing query: {e}")
            
            # Display router statistics
            print("\n--- Enhanced Router Statistics ---")
            stats = router.get_enhanced_statistics()
            enhanced_stats = stats['enhanced_routing_stats']
            
            print(f"Total Enhanced Routes: {enhanced_stats['enhanced_routes']}")
            print(f"Comprehensive Confidence Calculations: {enhanced_stats['comprehensive_confidence_calculations']}")
            print(f"LLM Classifications: {enhanced_stats['llm_classifications']}")
            print(f"Fallback Routes: {enhanced_stats['fallback_routes']}")
            
            # Test optimization
            print("\n--- Performance Optimization ---")
            optimization_results = await router.optimize_performance()
            print(f"Actions Taken: {len(optimization_results['actions_taken'])}")
            print(f"Recommendations: {len(optimization_results['recommendations'])}")
            
            for rec in optimization_results['recommendations'][:3]:  # Show first 3
                if isinstance(rec, dict):
                    print(f"- {rec.get('type', 'general')}: {rec.get('issue', 'N/A')}")
                else:
                    print(f"- {rec}")
    
    # Run demo
    print("Running enhanced biomedical query router demo...")
    try:
        asyncio.run(demo_enhanced_router())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()