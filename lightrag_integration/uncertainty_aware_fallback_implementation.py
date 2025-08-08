"""
Uncertainty-Aware Fallback Enhancement Implementation

This module provides implementation skeletons for the uncertainty-aware fallback
enhancements designed to integrate with the existing comprehensive fallback system.

The implementation focuses on proactive uncertainty detection and intelligent
routing strategies to handle uncertain classifications before they become failures.

Key Components:
    - UncertaintyDetector: Proactive uncertainty pattern detection
    - UncertaintyRoutingEngine: Intelligent routing for uncertain queries
    - UncertaintyFallbackStrategies: Specialized strategies for uncertainty types
    - Enhanced integration points with existing FallbackOrchestrator

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import existing components for integration
try:
    from .query_router import ConfidenceMetrics, RoutingPrediction, RoutingDecision, BiomedicalQueryRouter
    from .research_categorizer import CategoryPrediction, ResearchCategorizer
    from .enhanced_llm_classifier import ClassificationResult, EnhancedLLMQueryClassifier
    from .cost_persistence import ResearchCategory
    from .comprehensive_fallback_system import (
        FallbackOrchestrator, FallbackResult, FallbackLevel, FailureType,
        FailureDetectionMetrics, FallbackMonitor
    )
    from .comprehensive_confidence_scorer import HybridConfidenceResult, HybridConfidenceScorer
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# UNCERTAINTY DETECTION AND ANALYSIS
# ============================================================================

class UncertaintyType(Enum):
    """Types of uncertainty that can be detected in query classification."""
    
    LOW_CONFIDENCE = "low_confidence"              # Overall confidence too low
    HIGH_AMBIGUITY = "high_ambiguity"              # Query could fit multiple categories
    HIGH_CONFLICT = "high_conflict"                # Contradictory classification signals
    WEAK_EVIDENCE = "weak_evidence"                # Poor supporting evidence
    LLM_UNCERTAINTY = "llm_uncertainty"           # LLM expressing uncertainty
    WIDE_CONFIDENCE_INTERVAL = "wide_confidence_interval"  # Large uncertainty range
    INCONSISTENT_ALTERNATIVES = "inconsistent_alternatives"  # Alternative interpretations vary widely


class UncertaintyStrategy(Enum):
    """Strategies for handling different types of uncertainty."""
    
    UNCERTAINTY_CLARIFICATION = "uncertainty_clarification"
    HYBRID_CONSENSUS = "hybrid_consensus"
    CONFIDENCE_BOOSTING = "confidence_boosting"
    CONSERVATIVE_CLASSIFICATION = "conservative_classification"


@dataclass
class UncertaintyAnalysis:
    """Detailed analysis of query uncertainty with recommended actions."""
    
    # Uncertainty detection results
    detected_uncertainty_types: Set[UncertaintyType] = field(default_factory=set)
    uncertainty_severity: float = 0.0  # 0-1, higher = more uncertain
    confidence_degradation_risk: float = 0.0  # Risk of confidence degradation
    
    # Specific uncertainty metrics
    ambiguity_details: Dict[str, Any] = field(default_factory=dict)
    conflict_details: Dict[str, Any] = field(default_factory=dict)
    evidence_details: Dict[str, Any] = field(default_factory=dict)
    
    # Recommended actions
    requires_special_handling: bool = False
    recommended_strategy: Optional[UncertaintyStrategy] = None
    recommended_fallback_level: Optional[FallbackLevel] = None
    
    # Context for strategy selection
    query_characteristics: Dict[str, Any] = field(default_factory=dict)
    historical_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'detected_uncertainty_types': [ut.value for ut in self.detected_uncertainty_types],
            'uncertainty_severity': self.uncertainty_severity,
            'confidence_degradation_risk': self.confidence_degradation_risk,
            'ambiguity_details': self.ambiguity_details,
            'conflict_details': self.conflict_details,
            'evidence_details': self.evidence_details,
            'requires_special_handling': self.requires_special_handling,
            'recommended_strategy': self.recommended_strategy.value if self.recommended_strategy else None,
            'recommended_fallback_level': self.recommended_fallback_level.name if self.recommended_fallback_level else None,
            'query_characteristics': self.query_characteristics,
            'historical_patterns': self.historical_patterns
        }


@dataclass 
class UncertaintyFallbackConfig:
    """Configuration for uncertainty-aware fallback system."""
    
    # Uncertainty detection thresholds
    ambiguity_threshold_moderate: float = 0.4
    ambiguity_threshold_high: float = 0.7
    conflict_threshold_moderate: float = 0.3
    conflict_threshold_high: float = 0.6
    evidence_strength_threshold_weak: float = 0.3
    evidence_strength_threshold_very_weak: float = 0.1
    confidence_interval_width_threshold: float = 0.3
    
    # Strategy selection parameters
    clarification_min_alternatives: int = 2
    consensus_min_approaches: int = 3
    consensus_agreement_threshold: float = 0.7
    confidence_boost_max_adjustment: float = 0.2
    
    # Conservative classification settings
    conservative_confidence_threshold: float = 0.15
    conservative_default_routing: RoutingDecision = RoutingDecision.EITHER
    conservative_category: ResearchCategory = ResearchCategory.GENERAL_QUERY
    
    # Integration settings
    enable_proactive_detection: bool = True
    enable_uncertainty_learning: bool = True
    log_uncertainty_events: bool = True
    uncertainty_cache_size: int = 1000
    
    # Performance targets
    max_uncertainty_analysis_time_ms: float = 100.0
    max_clarification_generation_time_ms: float = 200.0
    min_confidence_improvement: float = 0.05


class UncertaintyDetector:
    """
    Proactive detection of uncertainty patterns before they become failures.
    Integrates with existing FailureDetector infrastructure.
    """
    
    def __init__(self, 
                 config: Optional[UncertaintyFallbackConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize uncertainty detector."""
        self.config = config or UncertaintyFallbackConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Historical pattern tracking
        self.uncertainty_patterns: deque = deque(maxlen=self.config.uncertainty_cache_size)
        self.pattern_success_rates: Dict[str, float] = {}
        
        # Performance metrics
        self.detection_metrics = {
            'total_analyses': 0,
            'uncertainty_detected': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'average_analysis_time_ms': 0.0
        }
        
        self.logger.info("UncertaintyDetector initialized with proactive detection enabled")
    
    def analyze_query_uncertainty(self, 
                                 query_text: str,
                                 confidence_metrics: ConfidenceMetrics,
                                 context: Optional[Dict[str, Any]] = None) -> UncertaintyAnalysis:
        """
        Comprehensive uncertainty analysis for a query.
        
        Args:
            query_text: The user query text
            confidence_metrics: Existing confidence metrics from classification
            context: Optional context information
            
        Returns:
            UncertaintyAnalysis with detected patterns and recommendations
        """
        start_time = time.time()
        self.detection_metrics['total_analyses'] += 1
        
        try:
            uncertainty_analysis = UncertaintyAnalysis()
            
            # Detect different types of uncertainty
            self._detect_low_confidence_uncertainty(confidence_metrics, uncertainty_analysis)
            self._detect_ambiguity_uncertainty(confidence_metrics, uncertainty_analysis)
            self._detect_conflict_uncertainty(confidence_metrics, uncertainty_analysis)
            self._detect_evidence_weakness_uncertainty(confidence_metrics, uncertainty_analysis)
            self._detect_confidence_interval_uncertainty(confidence_metrics, uncertainty_analysis)
            
            # Analyze query characteristics for context
            uncertainty_analysis.query_characteristics = self._analyze_query_characteristics(query_text)
            
            # Calculate overall uncertainty severity
            uncertainty_analysis.uncertainty_severity = self._calculate_uncertainty_severity(
                uncertainty_analysis, confidence_metrics
            )
            
            # Determine if special handling is required
            uncertainty_analysis.requires_special_handling = (
                uncertainty_analysis.uncertainty_severity > 0.5 or
                len(uncertainty_analysis.detected_uncertainty_types) >= 2
            )
            
            # Recommend strategy and fallback level
            if uncertainty_analysis.requires_special_handling:
                uncertainty_analysis.recommended_strategy = self._recommend_uncertainty_strategy(
                    uncertainty_analysis
                )
                uncertainty_analysis.recommended_fallback_level = self._recommend_fallback_level(
                    uncertainty_analysis
                )
                
                self.detection_metrics['uncertainty_detected'] += 1
            
            # Store pattern for future learning
            self._store_uncertainty_pattern(query_text, uncertainty_analysis, confidence_metrics)
            
            # Update performance metrics
            analysis_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(analysis_time)
            
            self.logger.debug(f"Uncertainty analysis completed in {analysis_time:.2f}ms: "
                            f"severity={uncertainty_analysis.uncertainty_severity:.3f}, "
                            f"types={len(uncertainty_analysis.detected_uncertainty_types)}")
            
            return uncertainty_analysis
            
        except Exception as e:
            self.logger.error(f"Error in uncertainty analysis: {e}")
            # Return safe default analysis
            return self._create_safe_default_analysis()
    
    def _detect_low_confidence_uncertainty(self, confidence_metrics: ConfidenceMetrics, 
                                         analysis: UncertaintyAnalysis):
        """Detect low confidence uncertainty patterns."""
        if confidence_metrics.overall_confidence < self.config.confidence_threshold_moderate:
            analysis.detected_uncertainty_types.add(UncertaintyType.LOW_CONFIDENCE)
            
            # Determine severity
            if confidence_metrics.overall_confidence < self.config.conservative_confidence_threshold:
                severity = "critical"
            elif confidence_metrics.overall_confidence < 0.3:
                severity = "severe"
            else:
                severity = "moderate"
            
            analysis.confidence_details = {
                'confidence_level': confidence_metrics.overall_confidence,
                'severity': severity,
                'threshold_used': self.config.confidence_threshold_moderate
            }
    
    def _detect_ambiguity_uncertainty(self, confidence_metrics: ConfidenceMetrics,
                                    analysis: UncertaintyAnalysis):
        """Detect high ambiguity uncertainty patterns."""
        if confidence_metrics.ambiguity_score > self.config.ambiguity_threshold_moderate:
            analysis.detected_uncertainty_types.add(UncertaintyType.HIGH_AMBIGUITY)
            
            # Analyze alternative interpretations
            alternatives_count = len(confidence_metrics.alternative_interpretations)
            max_alternative_confidence = 0.0
            
            if confidence_metrics.alternative_interpretations:
                max_alternative_confidence = max(
                    conf for _, conf in confidence_metrics.alternative_interpretations
                )
            
            analysis.ambiguity_details = {
                'ambiguity_score': confidence_metrics.ambiguity_score,
                'alternatives_count': alternatives_count,
                'max_alternative_confidence': max_alternative_confidence,
                'confidence_gap': confidence_metrics.overall_confidence - max_alternative_confidence
            }
    
    def _detect_conflict_uncertainty(self, confidence_metrics: ConfidenceMetrics,
                                   analysis: UncertaintyAnalysis):
        """Detect conflicting signals uncertainty."""
        if confidence_metrics.conflict_score > self.config.conflict_threshold_moderate:
            analysis.detected_uncertainty_types.add(UncertaintyType.HIGH_CONFLICT)
            
            analysis.conflict_details = {
                'conflict_score': confidence_metrics.conflict_score,
                'severity': 'high' if confidence_metrics.conflict_score > self.config.conflict_threshold_high else 'moderate',
                'biomedical_entity_count': confidence_metrics.biomedical_entity_count,
                'pattern_match_strength': confidence_metrics.pattern_match_strength
            }
    
    def _detect_evidence_weakness_uncertainty(self, confidence_metrics: ConfidenceMetrics,
                                            analysis: UncertaintyAnalysis):
        """Detect weak evidence uncertainty."""
        # Check if evidence strength is available (from HybridConfidenceResult)
        if hasattr(confidence_metrics, 'evidence_strength'):
            if confidence_metrics.evidence_strength < self.config.evidence_strength_threshold_weak:
                analysis.detected_uncertainty_types.add(UncertaintyType.WEAK_EVIDENCE)
                
                analysis.evidence_details = {
                    'evidence_strength': confidence_metrics.evidence_strength,
                    'severity': 'very_weak' if confidence_metrics.evidence_strength < self.config.evidence_strength_threshold_very_weak else 'weak',
                    'signal_strength_confidence': confidence_metrics.signal_strength_confidence
                }
    
    def _detect_confidence_interval_uncertainty(self, confidence_metrics: ConfidenceMetrics,
                                              analysis: UncertaintyAnalysis):
        """Detect wide confidence interval uncertainty."""
        # This would be available from HybridConfidenceResult
        if hasattr(confidence_metrics, 'confidence_interval'):
            interval_width = confidence_metrics.confidence_interval[1] - confidence_metrics.confidence_interval[0]
            
            if interval_width > self.config.confidence_interval_width_threshold:
                analysis.detected_uncertainty_types.add(UncertaintyType.WIDE_CONFIDENCE_INTERVAL)
                
                analysis.confidence_details = {
                    'interval_width': interval_width,
                    'confidence_interval': confidence_metrics.confidence_interval,
                    'relative_width': interval_width / confidence_metrics.overall_confidence if confidence_metrics.overall_confidence > 0 else float('inf')
                }
    
    def _analyze_query_characteristics(self, query_text: str) -> Dict[str, Any]:
        """Analyze query characteristics that might affect uncertainty."""
        characteristics = {
            'length': len(query_text),
            'word_count': len(query_text.split()),
            'has_question_words': any(word in query_text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which']),
            'has_temporal_indicators': any(word in query_text.lower() for word in ['recent', 'latest', 'current', 'new', 'today']),
            'has_technical_terms': any(word in query_text.lower() for word in ['metabolite', 'pathway', 'biomarker', 'compound']),
            'complexity_score': len(query_text.split()) / 10.0  # Simple complexity measure
        }
        
        return characteristics
    
    def _calculate_uncertainty_severity(self, analysis: UncertaintyAnalysis, 
                                       confidence_metrics: ConfidenceMetrics) -> float:
        """Calculate overall uncertainty severity (0-1)."""
        severity_factors = []
        
        # Base severity from confidence
        base_severity = 1.0 - confidence_metrics.overall_confidence
        severity_factors.append(base_severity * 0.4)  # 40% weight
        
        # Ambiguity contribution
        if UncertaintyType.HIGH_AMBIGUITY in analysis.detected_uncertainty_types:
            ambiguity_severity = confidence_metrics.ambiguity_score
            severity_factors.append(ambiguity_severity * 0.25)  # 25% weight
        
        # Conflict contribution
        if UncertaintyType.HIGH_CONFLICT in analysis.detected_uncertainty_types:
            conflict_severity = confidence_metrics.conflict_score
            severity_factors.append(conflict_severity * 0.2)  # 20% weight
        
        # Evidence weakness contribution
        if UncertaintyType.WEAK_EVIDENCE in analysis.detected_uncertainty_types:
            evidence_severity = 1.0 - getattr(confidence_metrics, 'evidence_strength', 0.5)
            severity_factors.append(evidence_severity * 0.15)  # 15% weight
        
        # Combine factors
        total_severity = sum(severity_factors) if severity_factors else base_severity
        
        # Normalize to 0-1 range
        return min(1.0, total_severity)
    
    def _recommend_uncertainty_strategy(self, analysis: UncertaintyAnalysis) -> UncertaintyStrategy:
        """Recommend the best strategy for handling detected uncertainty."""
        
        # High ambiguity with multiple alternatives -> Clarification
        if (UncertaintyType.HIGH_AMBIGUITY in analysis.detected_uncertainty_types and
            analysis.ambiguity_details.get('alternatives_count', 0) >= self.config.clarification_min_alternatives):
            return UncertaintyStrategy.UNCERTAINTY_CLARIFICATION
        
        # Conflicting signals or multiple uncertainty types -> Consensus
        if (UncertaintyType.HIGH_CONFLICT in analysis.detected_uncertainty_types or
            len(analysis.detected_uncertainty_types) >= 2):
            return UncertaintyStrategy.HYBRID_CONSENSUS
        
        # Weak evidence but decent confidence -> Confidence boosting
        if (UncertaintyType.WEAK_EVIDENCE in analysis.detected_uncertainty_types and
            UncertaintyType.LOW_CONFIDENCE not in analysis.detected_uncertainty_types):
            return UncertaintyStrategy.CONFIDENCE_BOOSTING
        
        # High severity uncertainty -> Conservative approach
        if analysis.uncertainty_severity > 0.8:
            return UncertaintyStrategy.CONSERVATIVE_CLASSIFICATION
        
        # Default to consensus for moderate uncertainty
        return UncertaintyStrategy.HYBRID_CONSENSUS
    
    def _recommend_fallback_level(self, analysis: UncertaintyAnalysis) -> FallbackLevel:
        """Recommend appropriate fallback level based on uncertainty analysis."""
        
        # Very high uncertainty -> Skip to keyword-based
        if analysis.uncertainty_severity > 0.8:
            return FallbackLevel.KEYWORD_BASED_ONLY
        
        # High uncertainty -> Simplified LLM
        elif analysis.uncertainty_severity > 0.6:
            return FallbackLevel.SIMPLIFIED_LLM
        
        # Moderate uncertainty -> Try full LLM first
        else:
            return FallbackLevel.FULL_LLM_WITH_CONFIDENCE
    
    def _store_uncertainty_pattern(self, query_text: str, analysis: UncertaintyAnalysis,
                                 confidence_metrics: ConfidenceMetrics):
        """Store uncertainty pattern for future learning."""
        pattern = {
            'timestamp': datetime.now(),
            'query_text': query_text[:100],  # Truncate for privacy
            'uncertainty_types': [ut.value for ut in analysis.detected_uncertainty_types],
            'severity': analysis.uncertainty_severity,
            'confidence': confidence_metrics.overall_confidence,
            'ambiguity_score': confidence_metrics.ambiguity_score,
            'conflict_score': confidence_metrics.conflict_score,
            'recommended_strategy': analysis.recommended_strategy.value if analysis.recommended_strategy else None
        }
        
        self.uncertainty_patterns.append(pattern)
    
    def _update_performance_metrics(self, analysis_time_ms: float):
        """Update performance metrics."""
        current_avg = self.detection_metrics['average_analysis_time_ms']
        total_analyses = self.detection_metrics['total_analyses']
        
        # Update running average
        if total_analyses > 1:
            self.detection_metrics['average_analysis_time_ms'] = (
                (current_avg * (total_analyses - 1) + analysis_time_ms) / total_analyses
            )
        else:
            self.detection_metrics['average_analysis_time_ms'] = analysis_time_ms
    
    def _create_safe_default_analysis(self) -> UncertaintyAnalysis:
        """Create safe default analysis when errors occur."""
        return UncertaintyAnalysis(
            detected_uncertainty_types={UncertaintyType.LOW_CONFIDENCE},
            uncertainty_severity=0.8,
            requires_special_handling=True,
            recommended_strategy=UncertaintyStrategy.CONSERVATIVE_CLASSIFICATION,
            recommended_fallback_level=FallbackLevel.KEYWORD_BASED_ONLY
        )
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        total_analyses = self.detection_metrics['total_analyses']
        
        return {
            'detection_metrics': self.detection_metrics.copy(),
            'detection_rate': (self.detection_metrics['uncertainty_detected'] / total_analyses 
                             if total_analyses > 0 else 0.0),
            'recent_patterns': list(self.uncertainty_patterns)[-10:],  # Last 10 patterns
            'pattern_success_rates': self.pattern_success_rates.copy(),
            'config_summary': {
                'ambiguity_threshold_high': self.config.ambiguity_threshold_high,
                'conflict_threshold_high': self.config.conflict_threshold_high,
                'evidence_strength_threshold_weak': self.config.evidence_strength_threshold_weak
            }
        }


# ============================================================================
# UNCERTAINTY-SPECIFIC FALLBACK STRATEGIES
# ============================================================================

class UncertaintyFallbackStrategies:
    """Collection of specialized strategies for handling uncertainty."""
    
    def __init__(self, 
                 config: Optional[UncertaintyFallbackConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize uncertainty fallback strategies."""
        self.config = config or UncertaintyFallbackConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Strategy performance tracking
        self.strategy_metrics = defaultdict(lambda: {
            'uses': 0,
            'successes': 0,
            'average_confidence_improvement': 0.0,
            'average_processing_time_ms': 0.0
        })
    
    def apply_clarification_strategy(self, 
                                   query_text: str,
                                   uncertainty_analysis: UncertaintyAnalysis,
                                   confidence_metrics: ConfidenceMetrics,
                                   context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """
        Apply uncertainty clarification strategy.
        
        Generates clarifying questions and provides multiple interpretation options.
        """
        start_time = time.time()
        strategy_name = "uncertainty_clarification"
        self.strategy_metrics[strategy_name]['uses'] += 1
        
        try:
            self.logger.info(f"Applying clarification strategy for query: {query_text[:50]}...")
            
            # Generate clarifying questions
            clarifying_questions = self._generate_clarifying_questions(
                query_text, uncertainty_analysis, confidence_metrics
            )
            
            # Provide multiple interpretation options
            interpretation_options = self._generate_interpretation_options(
                uncertainty_analysis, confidence_metrics
            )
            
            # Create enhanced routing prediction with clarification
            enhanced_prediction = self._create_clarification_prediction(
                query_text, confidence_metrics, clarifying_questions, interpretation_options
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create fallback result
            fallback_result = FallbackResult(
                routing_prediction=enhanced_prediction,
                fallback_level_used=FallbackLevel.FULL_LLM_WITH_CONFIDENCE,
                success=True,
                total_processing_time_ms=processing_time_ms,
                quality_score=0.8,  # High quality due to clarification
                reliability_score=0.9,
                warnings=[
                    f"Query ambiguity detected (score: {uncertainty_analysis.uncertainty_severity:.3f})",
                    "Clarifying questions generated to resolve uncertainty"
                ],
                fallback_chain=[strategy_name],
                debug_info={
                    'strategy_used': strategy_name,
                    'clarifying_questions': clarifying_questions,
                    'interpretation_options': interpretation_options,
                    'uncertainty_analysis': uncertainty_analysis.to_dict()
                }
            )
            
            # Update metrics
            self._update_strategy_metrics(strategy_name, processing_time_ms, success=True)
            
            self.logger.info(f"Clarification strategy completed successfully in {processing_time_ms:.2f}ms")
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Clarification strategy failed: {e}")
            self._update_strategy_metrics(strategy_name, 0, success=False)
            return self._create_strategy_failure_fallback(query_text, e)
    
    def apply_consensus_strategy(self,
                               query_text: str,
                               uncertainty_analysis: UncertaintyAnalysis,
                               confidence_metrics: ConfidenceMetrics,
                               context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """
        Apply hybrid consensus strategy.
        
        Uses multiple classification approaches and combines results.
        """
        start_time = time.time()
        strategy_name = "hybrid_consensus"
        self.strategy_metrics[strategy_name]['uses'] += 1
        
        try:
            self.logger.info(f"Applying consensus strategy for query: {query_text[:50]}...")
            
            # Collect multiple classification approaches
            classification_results = self._collect_multiple_classifications(
                query_text, confidence_metrics, context
            )
            
            # Calculate consensus
            consensus_result = self._calculate_consensus(classification_results)
            
            # Apply confidence boosting if strong consensus
            if consensus_result['consensus_strength'] > self.config.consensus_agreement_threshold:
                consensus_result['confidence'] *= 1.2  # Confidence boost
                consensus_result['confidence'] = min(1.0, consensus_result['confidence'])
            
            # Create enhanced prediction
            enhanced_prediction = self._create_consensus_prediction(
                query_text, confidence_metrics, consensus_result
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create fallback result
            fallback_result = FallbackResult(
                routing_prediction=enhanced_prediction,
                fallback_level_used=FallbackLevel.FULL_LLM_WITH_CONFIDENCE,
                success=True,
                total_processing_time_ms=processing_time_ms,
                quality_score=consensus_result['consensus_strength'],
                reliability_score=0.85,
                warnings=[
                    f"Consensus strategy applied with {len(classification_results)} approaches"
                ],
                fallback_chain=[strategy_name],
                debug_info={
                    'strategy_used': strategy_name,
                    'consensus_result': consensus_result,
                    'classification_approaches': len(classification_results),
                    'uncertainty_analysis': uncertainty_analysis.to_dict()
                }
            )
            
            # Update metrics
            confidence_improvement = enhanced_prediction.confidence - confidence_metrics.overall_confidence
            self._update_strategy_metrics(strategy_name, processing_time_ms, success=True, 
                                        confidence_improvement=confidence_improvement)
            
            self.logger.info(f"Consensus strategy completed successfully in {processing_time_ms:.2f}ms")
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Consensus strategy failed: {e}")
            self._update_strategy_metrics(strategy_name, 0, success=False)
            return self._create_strategy_failure_fallback(query_text, e)
    
    def apply_confidence_boosting_strategy(self,
                                         query_text: str,
                                         uncertainty_analysis: UncertaintyAnalysis,
                                         confidence_metrics: ConfidenceMetrics,
                                         context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """
        Apply confidence boosting strategy.
        
        Uses alternative analysis methods and historical calibration.
        """
        start_time = time.time()
        strategy_name = "confidence_boosting"
        self.strategy_metrics[strategy_name]['uses'] += 1
        
        try:
            self.logger.info(f"Applying confidence boosting strategy for query: {query_text[:50]}...")
            
            # Apply historical calibration
            calibrated_confidence = self._apply_historical_calibration(
                confidence_metrics, uncertainty_analysis
            )
            
            # Apply evidence strength adjustment
            evidence_adjusted_confidence = self._apply_evidence_strength_adjustment(
                calibrated_confidence, uncertainty_analysis
            )
            
            # Apply conservative boosting
            final_confidence = min(
                confidence_metrics.overall_confidence + self.config.confidence_boost_max_adjustment,
                evidence_adjusted_confidence
            )
            
            # Create boosted prediction
            boosted_prediction = self._create_boosted_prediction(
                query_text, confidence_metrics, final_confidence
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create fallback result
            fallback_result = FallbackResult(
                routing_prediction=boosted_prediction,
                fallback_level_used=FallbackLevel.FULL_LLM_WITH_CONFIDENCE,
                success=True,
                total_processing_time_ms=processing_time_ms,
                quality_score=0.75,
                reliability_score=0.8,
                confidence_degradation=confidence_metrics.overall_confidence - final_confidence,
                warnings=[
                    f"Confidence boosted from {confidence_metrics.overall_confidence:.3f} to {final_confidence:.3f}"
                ],
                fallback_chain=[strategy_name],
                debug_info={
                    'strategy_used': strategy_name,
                    'original_confidence': confidence_metrics.overall_confidence,
                    'calibrated_confidence': calibrated_confidence,
                    'evidence_adjusted_confidence': evidence_adjusted_confidence,
                    'final_confidence': final_confidence,
                    'uncertainty_analysis': uncertainty_analysis.to_dict()
                }
            )
            
            # Update metrics
            confidence_improvement = final_confidence - confidence_metrics.overall_confidence
            self._update_strategy_metrics(strategy_name, processing_time_ms, success=True,
                                        confidence_improvement=confidence_improvement)
            
            self.logger.info(f"Confidence boosting strategy completed successfully in {processing_time_ms:.2f}ms")
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Confidence boosting strategy failed: {e}")
            self._update_strategy_metrics(strategy_name, 0, success=False)
            return self._create_strategy_failure_fallback(query_text, e)
    
    def apply_conservative_classification_strategy(self,
                                                 query_text: str,
                                                 uncertainty_analysis: UncertaintyAnalysis,
                                                 confidence_metrics: ConfidenceMetrics,
                                                 context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """
        Apply conservative classification strategy.
        
        Defaults to broader categories and multiple routing options.
        """
        start_time = time.time()
        strategy_name = "conservative_classification"
        self.strategy_metrics[strategy_name]['uses'] += 1
        
        try:
            self.logger.info(f"Applying conservative classification strategy for query: {query_text[:50]}...")
            
            # Create conservative prediction
            conservative_prediction = self._create_conservative_prediction(
                query_text, uncertainty_analysis
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create fallback result
            fallback_result = FallbackResult(
                routing_prediction=conservative_prediction,
                fallback_level_used=FallbackLevel.KEYWORD_BASED_ONLY,
                success=True,
                total_processing_time_ms=processing_time_ms,
                quality_score=0.6,  # Lower quality but highly reliable
                reliability_score=0.95,
                confidence_degradation=confidence_metrics.overall_confidence - self.config.conservative_confidence_threshold,
                warnings=[
                    "Conservative classification applied due to high uncertainty",
                    f"Uncertainty severity: {uncertainty_analysis.uncertainty_severity:.3f}"
                ],
                fallback_chain=[strategy_name],
                recovery_suggestions=[
                    "Consider providing more specific query details",
                    "Multiple routing options provided for increased success probability"
                ],
                debug_info={
                    'strategy_used': strategy_name,
                    'conservative_confidence': self.config.conservative_confidence_threshold,
                    'uncertainty_analysis': uncertainty_analysis.to_dict()
                }
            )
            
            # Update metrics
            self._update_strategy_metrics(strategy_name, processing_time_ms, success=True)
            
            self.logger.info(f"Conservative classification strategy completed successfully in {processing_time_ms:.2f}ms")
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Conservative classification strategy failed: {e}")
            self._update_strategy_metrics(strategy_name, 0, success=False)
            return self._create_strategy_failure_fallback(query_text, e)
    
    # Helper methods for strategy implementations
    
    def _generate_clarifying_questions(self, 
                                     query_text: str,
                                     uncertainty_analysis: UncertaintyAnalysis,
                                     confidence_metrics: ConfidenceMetrics) -> List[str]:
        """Generate targeted clarifying questions based on uncertainty analysis."""
        questions = []
        
        # High ambiguity questions
        if UncertaintyType.HIGH_AMBIGUITY in uncertainty_analysis.detected_uncertainty_types:
            alternatives = [alt[0].value for alt in confidence_metrics.alternative_interpretations]
            if len(alternatives) >= 2:
                questions.append(
                    f"Your query could relate to multiple areas: {', '.join(alternatives[:3])}. "
                    f"Which specific aspect interests you most?"
                )
        
        # Temporal ambiguity questions
        if uncertainty_analysis.query_characteristics.get('has_temporal_indicators'):
            questions.append(
                "Are you looking for the most recent/current information, or general knowledge on this topic?"
            )
        
        # Technical depth questions
        if uncertainty_analysis.query_characteristics.get('has_technical_terms'):
            questions.append(
                "Do you need general information or specific technical details and mechanisms?"
            )
        
        # Scope clarification
        if uncertainty_analysis.query_characteristics.get('complexity_score', 0) > 1.5:
            questions.append(
                "Would you like a comprehensive overview or information focused on a specific aspect?"
            )
        
        return questions[:3]  # Limit to 3 questions to avoid overwhelming user
    
    def _generate_interpretation_options(self,
                                       uncertainty_analysis: UncertaintyAnalysis,
                                       confidence_metrics: ConfidenceMetrics) -> List[Dict[str, Any]]:
        """Generate multiple interpretation options for ambiguous queries."""
        options = []
        
        for routing_decision, confidence in confidence_metrics.alternative_interpretations:
            option = {
                'interpretation': routing_decision.value,
                'confidence': confidence,
                'description': self._get_interpretation_description(routing_decision),
                'recommended_for': self._get_recommendation_context(routing_decision)
            }
            options.append(option)
        
        return options[:3]  # Limit to top 3 options
    
    def _get_interpretation_description(self, routing_decision: RoutingDecision) -> str:
        """Get human-readable description for routing decision."""
        descriptions = {
            RoutingDecision.LIGHTRAG: "Knowledge graph analysis for relationships and mechanisms",
            RoutingDecision.PERPLEXITY: "Current information search and recent developments", 
            RoutingDecision.HYBRID: "Combined approach using both knowledge base and current information",
            RoutingDecision.EITHER: "Flexible routing based on query characteristics"
        }
        return descriptions.get(routing_decision, "Standard biomedical query processing")
    
    def _get_recommendation_context(self, routing_decision: RoutingDecision) -> str:
        """Get context for when this routing is recommended."""
        contexts = {
            RoutingDecision.LIGHTRAG: "Best for pathway analysis, metabolite relationships, and mechanism queries",
            RoutingDecision.PERPLEXITY: "Best for recent research, current developments, and time-sensitive information",
            RoutingDecision.HYBRID: "Best for comprehensive analysis requiring both established knowledge and recent findings",
            RoutingDecision.EITHER: "Suitable for general queries where multiple approaches could work"
        }
        return contexts.get(routing_decision, "General biomedical information queries")
    
    def _collect_multiple_classifications(self,
                                        query_text: str,
                                        confidence_metrics: ConfidenceMetrics,
                                        context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect classifications from multiple approaches."""
        # Placeholder implementation - would integrate with actual classifiers
        results = []
        
        # LLM-based classification (from existing result)
        results.append({
            'approach': 'llm_based',
            'routing_decision': RoutingDecision.LIGHTRAG,  # Would come from actual classification
            'confidence': confidence_metrics.overall_confidence,
            'weight': 0.6
        })
        
        # Keyword-based classification
        results.append({
            'approach': 'keyword_based',
            'routing_decision': RoutingDecision.PERPLEXITY,  # Would come from keyword analysis
            'confidence': confidence_metrics.pattern_match_strength,
            'weight': 0.3
        })
        
        # Pattern-based classification
        results.append({
            'approach': 'pattern_based',
            'routing_decision': RoutingDecision.HYBRID,  # Would come from pattern analysis
            'confidence': confidence_metrics.context_coherence_confidence,
            'weight': 0.1
        })
        
        return results
    
    def _calculate_consensus(self, classification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus from multiple classification approaches."""
        # Simple weighted voting implementation
        routing_votes = defaultdict(float)
        total_weight = 0
        
        for result in classification_results:
            routing_decision = result['routing_decision']
            weight = result['weight'] * result['confidence']
            routing_votes[routing_decision] += weight
            total_weight += weight
        
        # Find consensus decision
        if routing_votes:
            consensus_decision = max(routing_votes.items(), key=lambda x: x[1])
            consensus_strength = consensus_decision[1] / total_weight if total_weight > 0 else 0
            
            return {
                'routing_decision': consensus_decision[0],
                'confidence': consensus_strength,
                'consensus_strength': consensus_strength,
                'vote_distribution': dict(routing_votes),
                'total_approaches': len(classification_results)
            }
        else:
            return {
                'routing_decision': RoutingDecision.EITHER,
                'confidence': 0.1,
                'consensus_strength': 0.0,
                'vote_distribution': {},
                'total_approaches': 0
            }
    
    def _apply_historical_calibration(self,
                                    confidence_metrics: ConfidenceMetrics,
                                    uncertainty_analysis: UncertaintyAnalysis) -> float:
        """Apply historical calibration to confidence score."""
        # Placeholder implementation - would use actual historical data
        calibration_factor = 1.1  # Slight boost based on historical performance
        
        calibrated_confidence = confidence_metrics.overall_confidence * calibration_factor
        return min(1.0, calibrated_confidence)
    
    def _apply_evidence_strength_adjustment(self,
                                          confidence: float,
                                          uncertainty_analysis: UncertaintyAnalysis) -> float:
        """Adjust confidence based on evidence strength."""
        evidence_strength = uncertainty_analysis.evidence_details.get('evidence_strength', 0.5)
        
        # Apply evidence-based adjustment
        evidence_factor = 0.8 + (evidence_strength * 0.4)  # Range: 0.8 to 1.2
        adjusted_confidence = confidence * evidence_factor
        
        return min(1.0, adjusted_confidence)
    
    def _create_clarification_prediction(self,
                                       query_text: str,
                                       confidence_metrics: ConfidenceMetrics,
                                       clarifying_questions: List[str],
                                       interpretation_options: List[Dict[str, Any]]) -> RoutingPrediction:
        """Create routing prediction with clarification information."""
        # Create enhanced routing prediction
        return RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,  # Flexible routing
            confidence=0.7,  # Moderate confidence with clarification available
            reasoning=[
                "Query ambiguity detected - clarification recommended",
                f"Generated {len(clarifying_questions)} clarifying questions",
                f"Provided {len(interpretation_options)} interpretation options"
            ] + clarifying_questions,
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'uncertainty_strategy': 'clarification',
                'clarifying_questions': clarifying_questions,
                'interpretation_options': interpretation_options,
                'requires_user_input': True
            }
        )
    
    def _create_consensus_prediction(self,
                                   query_text: str,
                                   confidence_metrics: ConfidenceMetrics,
                                   consensus_result: Dict[str, Any]) -> RoutingPrediction:
        """Create routing prediction from consensus result."""
        return RoutingPrediction(
            routing_decision=consensus_result['routing_decision'],
            confidence=consensus_result['confidence'],
            reasoning=[
                f"Consensus achieved with {consensus_result['consensus_strength']:.3f} agreement",
                f"Multiple approaches considered: {consensus_result['total_approaches']}",
                "Enhanced confidence through ensemble voting"
            ],
            research_category=ResearchCategory.GENERAL_QUERY,  # Would be determined by consensus
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'uncertainty_strategy': 'consensus',
                'consensus_result': consensus_result,
                'ensemble_boost_applied': consensus_result['consensus_strength'] > 0.7
            }
        )
    
    def _create_boosted_prediction(self,
                                 query_text: str,
                                 confidence_metrics: ConfidenceMetrics,
                                 boosted_confidence: float) -> RoutingPrediction:
        """Create routing prediction with boosted confidence."""
        # Use original routing decision but with boosted confidence
        original_routing = RoutingDecision.LIGHTRAG  # Would come from original classification
        
        return RoutingPrediction(
            routing_decision=original_routing,
            confidence=boosted_confidence,
            reasoning=[
                f"Confidence boosted from {confidence_metrics.overall_confidence:.3f} to {boosted_confidence:.3f}",
                "Historical calibration and evidence analysis applied",
                "Conservative confidence adjustment within safe limits"
            ],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'uncertainty_strategy': 'confidence_boosting',
                'original_confidence': confidence_metrics.overall_confidence,
                'boost_applied': boosted_confidence - confidence_metrics.overall_confidence,
                'calibration_used': True
            }
        )
    
    def _create_conservative_prediction(self,
                                      query_text: str,
                                      uncertainty_analysis: UncertaintyAnalysis) -> RoutingPrediction:
        """Create conservative routing prediction."""
        return RoutingPrediction(
            routing_decision=self.config.conservative_default_routing,
            confidence=self.config.conservative_confidence_threshold,
            reasoning=[
                "Conservative classification applied due to high uncertainty",
                f"Uncertainty severity: {uncertainty_analysis.uncertainty_severity:.3f}",
                "Broad routing approach for maximum compatibility",
                "Multiple options available to increase success probability"
            ],
            research_category=self.config.conservative_category,
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=self.config.conservative_confidence_threshold,
                research_category_confidence=self.config.conservative_confidence_threshold,
                temporal_analysis_confidence=0.1,
                signal_strength_confidence=0.2,
                context_coherence_confidence=0.1,
                keyword_density=0.1,
                pattern_match_strength=0.1,
                biomedical_entity_count=1,
                ambiguity_score=uncertainty_analysis.uncertainty_severity,
                conflict_score=0.3,
                alternative_interpretations=[(RoutingDecision.EITHER, 0.1)],
                calculation_time_ms=10.0
            ),
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'uncertainty_strategy': 'conservative_classification',
                'high_uncertainty_detected': True,
                'conservative_routing': True,
                'multiple_options_recommended': True
            }
        )
    
    def _create_strategy_failure_fallback(self, query_text: str, error: Exception) -> FallbackResult:
        """Create fallback result when uncertainty strategy fails."""
        return FallbackResult(
            routing_prediction=RoutingPrediction(
                routing_decision=RoutingDecision.EITHER,
                confidence=0.1,
                reasoning=[
                    "Uncertainty strategy failed - using basic fallback",
                    f"Error: {str(error)}"
                ],
                research_category=ResearchCategory.GENERAL_QUERY,
                confidence_metrics=ConfidenceMetrics(
                    overall_confidence=0.1,
                    research_category_confidence=0.1,
                    temporal_analysis_confidence=0.0,
                    signal_strength_confidence=0.0,
                    context_coherence_confidence=0.0,
                    keyword_density=0.0,
                    pattern_match_strength=0.0,
                    biomedical_entity_count=0,
                    ambiguity_score=1.0,
                    conflict_score=0.0,
                    alternative_interpretations=[],
                    calculation_time_ms=0.0
                ),
                temporal_indicators=[],
                knowledge_indicators=[],
                metadata={'strategy_failure': True, 'error': str(error)}
            ),
            fallback_level_used=FallbackLevel.DEFAULT_ROUTING,
            success=False,
            failure_reasons=[FailureType.UNKNOWN_ERROR],
            warnings=[f"Uncertainty strategy failed: {str(error)}"],
            recovery_suggestions=["Manual query review may be needed"]
        )
    
    def _update_strategy_metrics(self, strategy_name: str, processing_time_ms: float, 
                               success: bool, confidence_improvement: float = 0.0):
        """Update performance metrics for a strategy."""
        metrics = self.strategy_metrics[strategy_name]
        
        if success:
            metrics['successes'] += 1
            
            # Update running average for confidence improvement
            current_improvement = metrics['average_confidence_improvement']
            successes = metrics['successes']
            metrics['average_confidence_improvement'] = (
                (current_improvement * (successes - 1) + confidence_improvement) / successes
            )
        
        # Update running average for processing time
        uses = metrics['uses']
        current_time = metrics['average_processing_time_ms']
        metrics['average_processing_time_ms'] = (
            (current_time * (uses - 1) + processing_time_ms) / uses
        )
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all uncertainty strategies."""
        stats = {}
        
        for strategy_name, metrics in self.strategy_metrics.items():
            success_rate = metrics['successes'] / metrics['uses'] if metrics['uses'] > 0 else 0.0
            stats[strategy_name] = {
                **metrics,
                'success_rate': success_rate
            }
        
        return {
            'strategy_statistics': stats,
            'total_strategies': len(self.strategy_metrics),
            'most_used_strategy': max(self.strategy_metrics.items(), 
                                    key=lambda x: x[1]['uses'])[0] if self.strategy_metrics else None,
            'best_performing_strategy': max(self.strategy_metrics.items(),
                                          key=lambda x: x[1]['successes'])[0] if self.strategy_metrics else None
        }


# ============================================================================
# ENHANCED FALLBACK ORCHESTRATOR WITH UNCERTAINTY AWARENESS
# ============================================================================

class UncertaintyAwareFallbackOrchestrator:
    """
    Enhanced fallback orchestrator that integrates uncertainty detection
    and specialized uncertainty handling strategies with the existing 
    comprehensive fallback system.
    """
    
    def __init__(self,
                 existing_orchestrator: FallbackOrchestrator,
                 config: Optional[UncertaintyFallbackConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize uncertainty-aware fallback orchestrator."""
        self.existing_orchestrator = existing_orchestrator
        self.config = config or UncertaintyFallbackConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize uncertainty components
        self.uncertainty_detector = UncertaintyDetector(config, logger)
        self.uncertainty_strategies = UncertaintyFallbackStrategies(config, logger)
        
        # Performance tracking
        self.uncertainty_processing_stats = {
            'total_queries_processed': 0,
            'uncertainty_detected_queries': 0,
            'uncertainty_strategies_applied': 0,
            'successful_uncertainty_resolutions': 0,
            'fallback_to_standard_system': 0
        }
        
        self.logger.info("UncertaintyAwareFallbackOrchestrator initialized")
    
    def process_query_with_uncertainty_awareness(self,
                                                query_text: str,
                                                context: Optional[Dict[str, Any]] = None,
                                                priority: str = 'normal') -> FallbackResult:
        """
        Main entry point for uncertainty-aware query processing.
        
        Args:
            query_text: The user query text
            context: Optional context information
            priority: Query priority level
            
        Returns:
            FallbackResult with uncertainty handling applied if needed
        """
        start_time = time.time()
        self.uncertainty_processing_stats['total_queries_processed'] += 1
        
        try:
            self.logger.debug(f"Processing query with uncertainty awareness: {query_text[:50]}...")
            
            # Step 1: Perform initial classification to get confidence metrics
            initial_prediction = self._get_initial_prediction(query_text, context)
            
            # Step 2: Analyze uncertainty patterns
            uncertainty_analysis = self.uncertainty_detector.analyze_query_uncertainty(
                query_text, initial_prediction.confidence_metrics, context
            )
            
            # Step 3: Determine if uncertainty-specific handling is needed
            if uncertainty_analysis.requires_special_handling:
                self.uncertainty_processing_stats['uncertainty_detected_queries'] += 1
                
                # Apply uncertainty-specific strategy
                uncertainty_result = self._apply_uncertainty_strategy(
                    query_text, uncertainty_analysis, initial_prediction.confidence_metrics, context
                )
                
                if uncertainty_result.success:
                    self.uncertainty_processing_stats['successful_uncertainty_resolutions'] += 1
                    total_time = (time.time() - start_time) * 1000
                    uncertainty_result.total_processing_time_ms += total_time
                    
                    self.logger.info(f"Uncertainty handling successful for query in {total_time:.2f}ms")
                    return uncertainty_result
                else:
                    # Uncertainty strategy failed - fall back to standard system
                    self.logger.warning("Uncertainty strategy failed - falling back to standard system")
                    self.uncertainty_processing_stats['fallback_to_standard_system'] += 1
            
            # Step 4: Use existing comprehensive fallback system
            # (either uncertainty not detected or uncertainty strategy failed)
            standard_result = self.existing_orchestrator.process_query_with_comprehensive_fallback(
                query_text=query_text,
                context=context,
                priority=priority
            )
            
            # Enhance standard result with uncertainty analysis if available
            if uncertainty_analysis.detected_uncertainty_types:
                self._enhance_standard_result_with_uncertainty_info(
                    standard_result, uncertainty_analysis
                )
            
            total_time = (time.time() - start_time) * 1000
            standard_result.total_processing_time_ms += total_time
            
            return standard_result
            
        except Exception as e:
            self.logger.error(f"Error in uncertainty-aware processing: {e}")
            # Fall back to existing system
            return self.existing_orchestrator.process_query_with_comprehensive_fallback(
                query_text=query_text,
                context=context,
                priority=priority
            )
    
    def _get_initial_prediction(self, query_text: str, 
                               context: Optional[Dict[str, Any]]) -> RoutingPrediction:
        """Get initial routing prediction for uncertainty analysis."""
        # This would integrate with existing routing system
        # Placeholder implementation
        return RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence=0.4,  # Intentionally low for demonstration
            reasoning=["Initial prediction for uncertainty analysis"],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.4,
                research_category_confidence=0.4,
                temporal_analysis_confidence=0.3,
                signal_strength_confidence=0.3,
                context_coherence_confidence=0.3,
                keyword_density=0.2,
                pattern_match_strength=0.2,
                biomedical_entity_count=2,
                ambiguity_score=0.6,  # High ambiguity for demonstration
                conflict_score=0.4,   # Some conflict
                alternative_interpretations=[
                    (RoutingDecision.PERPLEXITY, 0.35),
                    (RoutingDecision.HYBRID, 0.3)
                ],
                calculation_time_ms=50.0
            ),
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={}
        )
    
    def _apply_uncertainty_strategy(self,
                                  query_text: str,
                                  uncertainty_analysis: UncertaintyAnalysis,
                                  confidence_metrics: ConfidenceMetrics,
                                  context: Optional[Dict[str, Any]]) -> FallbackResult:
        """Apply appropriate uncertainty strategy based on analysis."""
        self.uncertainty_processing_stats['uncertainty_strategies_applied'] += 1
        
        strategy = uncertainty_analysis.recommended_strategy
        
        if strategy == UncertaintyStrategy.UNCERTAINTY_CLARIFICATION:
            return self.uncertainty_strategies.apply_clarification_strategy(
                query_text, uncertainty_analysis, confidence_metrics, context
            )
        
        elif strategy == UncertaintyStrategy.HYBRID_CONSENSUS:
            return self.uncertainty_strategies.apply_consensus_strategy(
                query_text, uncertainty_analysis, confidence_metrics, context
            )
        
        elif strategy == UncertaintyStrategy.CONFIDENCE_BOOSTING:
            return self.uncertainty_strategies.apply_confidence_boosting_strategy(
                query_text, uncertainty_analysis, confidence_metrics, context
            )
        
        elif strategy == UncertaintyStrategy.CONSERVATIVE_CLASSIFICATION:
            return self.uncertainty_strategies.apply_conservative_classification_strategy(
                query_text, uncertainty_analysis, confidence_metrics, context
            )
        
        else:
            # Default to consensus strategy
            return self.uncertainty_strategies.apply_consensus_strategy(
                query_text, uncertainty_analysis, confidence_metrics, context
            )
    
    def _enhance_standard_result_with_uncertainty_info(self,
                                                      standard_result: FallbackResult,
                                                      uncertainty_analysis: UncertaintyAnalysis):
        """Enhance standard fallback result with uncertainty analysis information."""
        if not standard_result.routing_prediction.metadata:
            standard_result.routing_prediction.metadata = {}
        
        # Add uncertainty analysis to metadata
        standard_result.routing_prediction.metadata.update({
            'uncertainty_analysis_performed': True,
            'detected_uncertainty_types': [ut.value for ut in uncertainty_analysis.detected_uncertainty_types],
            'uncertainty_severity': uncertainty_analysis.uncertainty_severity,
            'uncertainty_handling_attempted': True
        })
        
        # Add uncertainty warnings
        if uncertainty_analysis.uncertainty_severity > 0.5:
            warning = f"High uncertainty detected (severity: {uncertainty_analysis.uncertainty_severity:.3f})"
            if warning not in standard_result.warnings:
                standard_result.warnings.append(warning)
    
    def get_comprehensive_uncertainty_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for uncertainty-aware processing."""
        detector_stats = self.uncertainty_detector.get_detection_statistics()
        strategy_stats = self.uncertainty_strategies.get_strategy_statistics()
        
        # Calculate derived metrics
        total_processed = self.uncertainty_processing_stats['total_queries_processed']
        uncertainty_detection_rate = (
            self.uncertainty_processing_stats['uncertainty_detected_queries'] / total_processed
            if total_processed > 0 else 0.0
        )
        
        uncertainty_resolution_rate = (
            self.uncertainty_processing_stats['successful_uncertainty_resolutions'] / 
            self.uncertainty_processing_stats['uncertainty_strategies_applied']
            if self.uncertainty_processing_stats['uncertainty_strategies_applied'] > 0 else 0.0
        )
        
        return {
            'processing_statistics': self.uncertainty_processing_stats.copy(),
            'uncertainty_detection_rate': uncertainty_detection_rate,
            'uncertainty_resolution_rate': uncertainty_resolution_rate,
            'detector_statistics': detector_stats,
            'strategy_statistics': strategy_stats,
            'system_health': {
                'uncertainty_system_operational': True,
                'integration_with_standard_fallback': True,
                'performance_within_targets': detector_stats['detection_metrics']['average_analysis_time_ms'] < 100.0
            }
        }
    
    def enable_uncertainty_learning(self):
        """Enable learning mode for uncertainty pattern recognition."""
        self.config.enable_uncertainty_learning = True
        self.logger.info("Uncertainty learning mode enabled")
    
    def disable_uncertainty_learning(self):
        """Disable learning mode for uncertainty pattern recognition."""
        self.config.enable_uncertainty_learning = False
        self.logger.info("Uncertainty learning mode disabled")


# ============================================================================
# FACTORY FUNCTIONS FOR EASY INTEGRATION
# ============================================================================

def create_uncertainty_aware_fallback_system(
    existing_orchestrator: FallbackOrchestrator,
    config: Optional[UncertaintyFallbackConfig] = None,
    logger: Optional[logging.Logger] = None
) -> UncertaintyAwareFallbackOrchestrator:
    """
    Factory function to create uncertainty-aware fallback system.
    
    Args:
        existing_orchestrator: Existing FallbackOrchestrator instance
        config: Optional uncertainty fallback configuration
        logger: Optional logger instance
        
    Returns:
        UncertaintyAwareFallbackOrchestrator ready for use
    """
    return UncertaintyAwareFallbackOrchestrator(
        existing_orchestrator=existing_orchestrator,
        config=config,
        logger=logger
    )


def create_production_uncertainty_config() -> UncertaintyFallbackConfig:
    """Create production-ready uncertainty configuration."""
    return UncertaintyFallbackConfig(
        # More conservative thresholds for production
        ambiguity_threshold_moderate=0.3,
        ambiguity_threshold_high=0.6,
        conflict_threshold_moderate=0.25,
        conflict_threshold_high=0.5,
        evidence_strength_threshold_weak=0.4,
        evidence_strength_threshold_very_weak=0.15,
        
        # Production performance targets
        max_uncertainty_analysis_time_ms=80.0,
        max_clarification_generation_time_ms=150.0,
        min_confidence_improvement=0.03,
        
        # Enable all features for production
        enable_proactive_detection=True,
        enable_uncertainty_learning=True,
        log_uncertainty_events=True,
        uncertainty_cache_size=2000
    )


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Testing Uncertainty-Aware Fallback Enhancement System")
    
    # This would be replaced with actual existing orchestrator
    # For testing, we'll create a mock
    class MockFallbackOrchestrator:
        def process_query_with_comprehensive_fallback(self, query_text, context=None, priority='normal'):
            return FallbackResult(
                routing_prediction=RoutingPrediction(
                    routing_decision=RoutingDecision.LIGHTRAG,
                    confidence=0.3,
                    reasoning=["Mock standard fallback result"],
                    research_category=ResearchCategory.GENERAL_QUERY,
                    confidence_metrics=ConfidenceMetrics(
                        overall_confidence=0.3,
                        research_category_confidence=0.3,
                        temporal_analysis_confidence=0.2,
                        signal_strength_confidence=0.2,
                        context_coherence_confidence=0.2,
                        keyword_density=0.1,
                        pattern_match_strength=0.1,
                        biomedical_entity_count=1,
                        ambiguity_score=0.5,
                        conflict_score=0.3,
                        alternative_interpretations=[(RoutingDecision.PERPLEXITY, 0.25)],
                        calculation_time_ms=25.0
                    ),
                    temporal_indicators=[],
                    knowledge_indicators=[],
                    metadata={}
                ),
                fallback_level_used=FallbackLevel.FULL_LLM_WITH_CONFIDENCE,
                success=True,
                total_processing_time_ms=100.0,
                quality_score=0.7,
                reliability_score=0.8
            )
    
    # Test the uncertainty-aware system
    mock_orchestrator = MockFallbackOrchestrator()
    config = create_production_uncertainty_config()
    
    uncertainty_system = create_uncertainty_aware_fallback_system(
        existing_orchestrator=mock_orchestrator,
        config=config,
        logger=logger
    )
    
    # Test queries with different uncertainty patterns
    test_queries = [
        "What is metabolomics?",  # Low uncertainty
        "How does glucose metabolism work in diabetes?",  # Moderate uncertainty
        "Recent advances in metabolomics biomarker discovery",  # High ambiguity
        "Pathway analysis of metabolite interactions",  # Technical query
    ]
    
    logger.info("\n" + "="*60)
    logger.info("UNCERTAINTY-AWARE FALLBACK SYSTEM TESTING")
    logger.info("="*60)
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest {i}: {query}")
        logger.info("-" * 40)
        
        try:
            result = uncertainty_system.process_query_with_uncertainty_awareness(query)
            
            logger.info(f"Result: {result.routing_prediction.routing_decision.value}")
            logger.info(f"Confidence: {result.routing_prediction.confidence:.3f}")
            logger.info(f"Fallback Level: {result.fallback_level_used.name}")
            logger.info(f"Success: {result.success}")
            logger.info(f"Processing Time: {result.total_processing_time_ms:.2f}ms")
            
            if result.warnings:
                logger.info(f"Warnings: {result.warnings}")
            
            if result.routing_prediction.metadata:
                uncertainty_info = {k: v for k, v in result.routing_prediction.metadata.items() 
                                  if 'uncertainty' in k.lower()}
                if uncertainty_info:
                    logger.info(f"Uncertainty Info: {uncertainty_info}")
                    
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    
    # Get comprehensive statistics
    logger.info("\n" + "="*60)
    logger.info("SYSTEM STATISTICS")
    logger.info("="*60)
    
    stats = uncertainty_system.get_comprehensive_uncertainty_statistics()
    
    logger.info(f"Total Queries Processed: {stats['processing_statistics']['total_queries_processed']}")
    logger.info(f"Uncertainty Detection Rate: {stats['uncertainty_detection_rate']:.1%}")
    logger.info(f"Uncertainty Resolution Rate: {stats['uncertainty_resolution_rate']:.1%}")
    logger.info(f"System Health: {stats['system_health']['uncertainty_system_operational']}")
    
    logger.info("\nUncertainty-Aware Fallback Enhancement testing completed successfully")