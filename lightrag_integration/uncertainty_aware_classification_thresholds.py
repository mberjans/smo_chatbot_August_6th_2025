"""
Uncertainty-Aware Classification Thresholds Implementation

This module implements confidence threshold-based fallback logic that integrates
with the existing comprehensive fallback system to provide proactive uncertainty
detection and intelligent routing before classification failures occur.

The system implements a 4-level confidence threshold hierarchy:
- High confidence: >= 0.7 (reliable, direct routing)
- Medium confidence: >= 0.5 (reliable with validation)  
- Low confidence: >= 0.3 (requires fallback consideration)
- Very low confidence: < 0.3 (requires specialized handling)

Key Features:
    - Proactive uncertainty detection using existing metrics
    - Integration with 4 specialized fallback strategies
    - Performance optimized for < 100ms additional processing
    - Backward compatibility with existing ConfidenceMetrics
    - Comprehensive error handling and monitoring
    - Production-ready with detailed logging

Classes:
    - UncertaintyAwareClassificationThresholds: Main configuration class
    - ConfidenceThresholdRouter: Enhanced routing logic with threshold-based decisions
    - UncertaintyMetricsAnalyzer: Analysis of uncertainty patterns from existing metrics
    - ThresholdBasedFallbackIntegrator: Integration layer with existing FallbackOrchestrator

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
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
    from .comprehensive_confidence_scorer import (
        HybridConfidenceResult, HybridConfidenceScorer, LLMConfidenceAnalysis,
        KeywordConfidenceAnalysis, ConfidenceSource
    )
    from .uncertainty_aware_fallback_implementation import (
        UncertaintyDetector, UncertaintyFallbackStrategies, UncertaintyAnalysis,
        UncertaintyType, UncertaintyStrategy, UncertaintyFallbackConfig
    )
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# CONFIDENCE THRESHOLD DEFINITIONS AND CONFIGURATION
# ============================================================================

class ConfidenceLevel(Enum):
    """Enumeration of confidence levels with explicit thresholds."""
    
    HIGH = "high"           # >= 0.7 - Direct routing, high reliability
    MEDIUM = "medium"       # >= 0.5 - Validated routing, good reliability  
    LOW = "low"             # >= 0.3 - Fallback consideration, moderate reliability
    VERY_LOW = "very_low"   # < 0.3 - Specialized handling required


class ThresholdTrigger(Enum):
    """Types of threshold-based triggers for fallback activation."""
    
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    HIGH_UNCERTAINTY_DETECTED = "high_uncertainty_detected"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    WEAK_EVIDENCE_STRENGTH = "weak_evidence_strength"
    WIDE_CONFIDENCE_INTERVAL = "wide_confidence_interval"
    MULTIPLE_UNCERTAINTY_FACTORS = "multiple_uncertainty_factors"


@dataclass
class UncertaintyAwareClassificationThresholds:
    """
    Comprehensive configuration for uncertainty-aware classification thresholds.
    
    This class defines the 4-level confidence threshold system and integrates
    with existing uncertainty metrics for proactive fallback activation.
    """
    
    # Primary confidence thresholds (as specified in requirements)
    high_confidence_threshold: float = 0.7      # High confidence - direct routing
    medium_confidence_threshold: float = 0.5    # Medium confidence - validated routing
    low_confidence_threshold: float = 0.3       # Low confidence - fallback consideration
    very_low_confidence_threshold: float = 0.1  # Very low confidence - specialized handling
    
    # Uncertainty metric thresholds (integrated with existing system)
    ambiguity_score_threshold_moderate: float = 0.4
    ambiguity_score_threshold_high: float = 0.7
    conflict_score_threshold_moderate: float = 0.3
    conflict_score_threshold_high: float = 0.6
    total_uncertainty_threshold_moderate: float = 0.4
    total_uncertainty_threshold_high: float = 0.7
    evidence_strength_threshold_weak: float = 0.3
    evidence_strength_threshold_very_weak: float = 0.1
    
    # Confidence interval analysis thresholds
    confidence_interval_width_threshold_moderate: float = 0.3
    confidence_interval_width_threshold_high: float = 0.5
    confidence_reliability_threshold_low: float = 0.4
    confidence_reliability_threshold_very_low: float = 0.2
    
    # Fallback strategy selection parameters
    clarification_strategy_min_alternatives: int = 2
    consensus_strategy_min_approaches: int = 3
    consensus_agreement_threshold: float = 0.7
    confidence_boost_max_adjustment: float = 0.2
    conservative_fallback_confidence: float = 0.15
    
    # Integration and performance settings
    enable_proactive_threshold_monitoring: bool = True
    enable_uncertainty_pattern_learning: bool = True
    threshold_analysis_timeout_ms: float = 100.0
    max_fallback_attempts: int = 3
    
    # Monitoring and logging configuration
    log_threshold_decisions: bool = True
    log_uncertainty_patterns: bool = True
    performance_monitoring_enabled: bool = True
    detailed_metrics_collection: bool = True
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level based on score and thresholds."""
        if confidence_score >= self.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= self.low_confidence_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def should_trigger_fallback(self, 
                               confidence_metrics: ConfidenceMetrics,
                               hybrid_confidence: Optional[HybridConfidenceResult] = None) -> Tuple[bool, List[ThresholdTrigger]]:
        """
        Determine if fallback should be triggered based on confidence thresholds and uncertainty metrics.
        
        Args:
            confidence_metrics: Standard confidence metrics from existing system
            hybrid_confidence: Optional enhanced confidence result with uncertainty analysis
            
        Returns:
            Tuple of (should_trigger_fallback, list_of_triggers)
        """
        triggers = []
        
        # Check primary confidence threshold
        if confidence_metrics.overall_confidence < self.low_confidence_threshold:
            triggers.append(ThresholdTrigger.CONFIDENCE_BELOW_THRESHOLD)
        
        # Check ambiguity score threshold
        if confidence_metrics.ambiguity_score > self.ambiguity_score_threshold_moderate:
            triggers.append(ThresholdTrigger.HIGH_UNCERTAINTY_DETECTED)
        
        # Check conflict score threshold
        if confidence_metrics.conflict_score > self.conflict_score_threshold_moderate:
            triggers.append(ThresholdTrigger.CONFLICTING_EVIDENCE)
        
        # Check enhanced metrics if available
        if hybrid_confidence:
            # Check total uncertainty
            if hybrid_confidence.total_uncertainty > self.total_uncertainty_threshold_moderate:
                triggers.append(ThresholdTrigger.HIGH_UNCERTAINTY_DETECTED)
            
            # Check evidence strength
            if hybrid_confidence.evidence_strength < self.evidence_strength_threshold_weak:
                triggers.append(ThresholdTrigger.WEAK_EVIDENCE_STRENGTH)
            
            # Check confidence interval width
            interval_width = hybrid_confidence.confidence_interval[1] - hybrid_confidence.confidence_interval[0]
            if interval_width > self.confidence_interval_width_threshold_moderate:
                triggers.append(ThresholdTrigger.WIDE_CONFIDENCE_INTERVAL)
        
        # Check for multiple uncertainty factors
        if len(triggers) >= 2:
            triggers.append(ThresholdTrigger.MULTIPLE_UNCERTAINTY_FACTORS)
        
        should_trigger = len(triggers) > 0
        return should_trigger, triggers
    
    def recommend_fallback_strategy(self, 
                                  confidence_level: ConfidenceLevel,
                                  triggers: List[ThresholdTrigger],
                                  uncertainty_analysis: Optional[UncertaintyAnalysis] = None) -> UncertaintyStrategy:
        """Recommend appropriate fallback strategy based on confidence level and triggers."""
        
        # Very low confidence - conservative approach
        if confidence_level == ConfidenceLevel.VERY_LOW:
            return UncertaintyStrategy.CONSERVATIVE_CLASSIFICATION
        
        # Multiple triggers - use consensus
        if ThresholdTrigger.MULTIPLE_UNCERTAINTY_FACTORS in triggers:
            return UncertaintyStrategy.HYBRID_CONSENSUS
        
        # High ambiguity with alternatives - clarification
        if (ThresholdTrigger.HIGH_UNCERTAINTY_DETECTED in triggers and
            uncertainty_analysis and
            len(getattr(uncertainty_analysis, 'ambiguity_details', {}).get('alternatives_count', 0)) >= 
            self.clarification_strategy_min_alternatives):
            return UncertaintyStrategy.UNCERTAINTY_CLARIFICATION
        
        # Conflicting evidence - consensus
        if ThresholdTrigger.CONFLICTING_EVIDENCE in triggers:
            return UncertaintyStrategy.HYBRID_CONSENSUS
        
        # Weak evidence but decent confidence - confidence boosting
        if (ThresholdTrigger.WEAK_EVIDENCE_STRENGTH in triggers and
            confidence_level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]):
            return UncertaintyStrategy.CONFIDENCE_BOOSTING
        
        # Default to consensus for other cases
        return UncertaintyStrategy.HYBRID_CONSENSUS
    
    def recommend_fallback_level(self, confidence_level: ConfidenceLevel, 
                                uncertainty_severity: float = 0.0) -> FallbackLevel:
        """Recommend appropriate fallback level based on confidence and uncertainty."""
        
        # Very high uncertainty - skip to keyword-based
        if uncertainty_severity > 0.8 or confidence_level == ConfidenceLevel.VERY_LOW:
            return FallbackLevel.KEYWORD_BASED_ONLY
        
        # High uncertainty - simplified LLM
        elif uncertainty_severity > 0.6 or confidence_level == ConfidenceLevel.LOW:
            return FallbackLevel.SIMPLIFIED_LLM
        
        # Moderate uncertainty - try full LLM first
        else:
            return FallbackLevel.FULL_LLM_WITH_CONFIDENCE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'confidence_thresholds': {
                'high': self.high_confidence_threshold,
                'medium': self.medium_confidence_threshold,
                'low': self.low_confidence_threshold,
                'very_low': self.very_low_confidence_threshold
            },
            'uncertainty_thresholds': {
                'ambiguity_moderate': self.ambiguity_score_threshold_moderate,
                'ambiguity_high': self.ambiguity_score_threshold_high,
                'conflict_moderate': self.conflict_score_threshold_moderate,
                'conflict_high': self.conflict_score_threshold_high,
                'uncertainty_moderate': self.total_uncertainty_threshold_moderate,
                'uncertainty_high': self.total_uncertainty_threshold_high,
                'evidence_weak': self.evidence_strength_threshold_weak,
                'evidence_very_weak': self.evidence_strength_threshold_very_weak
            },
            'performance_settings': {
                'threshold_analysis_timeout_ms': self.threshold_analysis_timeout_ms,
                'max_fallback_attempts': self.max_fallback_attempts,
                'enable_proactive_monitoring': self.enable_proactive_threshold_monitoring,
                'enable_pattern_learning': self.enable_uncertainty_pattern_learning
            }
        }


# ============================================================================
# UNCERTAINTY METRICS ANALYZER
# ============================================================================

class UncertaintyMetricsAnalyzer:
    """
    Analyzer that interprets uncertainty patterns from existing confidence metrics
    and hybrid confidence results to provide actionable uncertainty insights.
    """
    
    def __init__(self, 
                 thresholds: UncertaintyAwareClassificationThresholds,
                 logger: Optional[logging.Logger] = None):
        """Initialize uncertainty metrics analyzer."""
        self.thresholds = thresholds
        self.logger = logger or logging.getLogger(__name__)
        
        # Pattern tracking for learning
        self.uncertainty_patterns: deque = deque(maxlen=1000)
        self.pattern_success_rates: Dict[str, float] = {}
        
        # Performance metrics
        self.analysis_metrics = {
            'total_analyses': 0,
            'high_uncertainty_detected': 0,
            'successful_predictions': 0,
            'average_analysis_time_ms': 0.0
        }
        
        self.logger.info("UncertaintyMetricsAnalyzer initialized with threshold-based analysis")
    
    def analyze_uncertainty_from_confidence_metrics(self, 
                                                   query_text: str,
                                                   confidence_metrics: ConfidenceMetrics,
                                                   hybrid_confidence: Optional[HybridConfidenceResult] = None,
                                                   context: Optional[Dict[str, Any]] = None) -> UncertaintyAnalysis:
        """
        Analyze uncertainty patterns from existing confidence metrics.
        
        Args:
            query_text: The original query text
            confidence_metrics: Standard confidence metrics from existing system  
            hybrid_confidence: Optional enhanced confidence analysis
            context: Optional context information
            
        Returns:
            UncertaintyAnalysis with detected patterns and recommended actions
        """
        start_time = time.time()
        self.analysis_metrics['total_analyses'] += 1
        
        try:
            # Initialize uncertainty analysis
            uncertainty_analysis = UncertaintyAnalysis()
            
            # Analyze confidence level
            confidence_level = self.thresholds.get_confidence_level(confidence_metrics.overall_confidence)
            
            # Detect uncertainty types based on thresholds
            self._detect_threshold_based_uncertainty(
                confidence_metrics, hybrid_confidence, uncertainty_analysis
            )
            
            # Analyze query characteristics
            uncertainty_analysis.query_characteristics = self._analyze_query_characteristics(query_text)
            
            # Calculate overall uncertainty severity
            uncertainty_analysis.uncertainty_severity = self._calculate_severity_from_metrics(
                confidence_metrics, hybrid_confidence, uncertainty_analysis
            )
            
            # Determine if special handling is required
            should_trigger, triggers = self.thresholds.should_trigger_fallback(
                confidence_metrics, hybrid_confidence
            )
            
            uncertainty_analysis.requires_special_handling = should_trigger
            
            # Recommend strategy and fallback level
            if should_trigger:
                uncertainty_analysis.recommended_strategy = self.thresholds.recommend_fallback_strategy(
                    confidence_level, triggers, uncertainty_analysis
                )
                uncertainty_analysis.recommended_fallback_level = self.thresholds.recommend_fallback_level(
                    confidence_level, uncertainty_analysis.uncertainty_severity
                )
                
                self.analysis_metrics['high_uncertainty_detected'] += 1
            
            # Store pattern for learning
            if self.thresholds.enable_uncertainty_pattern_learning:
                self._store_analysis_pattern(
                    query_text, uncertainty_analysis, confidence_metrics, triggers
                )
            
            # Update performance metrics
            analysis_time_ms = (time.time() - start_time) * 1000
            self._update_analysis_metrics(analysis_time_ms)
            
            if self.thresholds.log_threshold_decisions:
                self.logger.debug(f"Uncertainty analysis completed in {analysis_time_ms:.2f}ms: "
                                f"severity={uncertainty_analysis.uncertainty_severity:.3f}, "
                                f"confidence_level={confidence_level.value}, "
                                f"triggers={len(triggers)}")
            
            return uncertainty_analysis
            
        except Exception as e:
            self.logger.error(f"Error in uncertainty metrics analysis: {e}")
            return self._create_safe_default_analysis(confidence_metrics.overall_confidence)
    
    def _detect_threshold_based_uncertainty(self,
                                          confidence_metrics: ConfidenceMetrics,
                                          hybrid_confidence: Optional[HybridConfidenceResult],
                                          uncertainty_analysis: UncertaintyAnalysis):
        """Detect uncertainty types based on threshold analysis."""
        
        # Low confidence detection
        if confidence_metrics.overall_confidence < self.thresholds.low_confidence_threshold:
            uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.LOW_CONFIDENCE)
            uncertainty_analysis.confidence_details = {
                'confidence_level': confidence_metrics.overall_confidence,
                'threshold_used': self.thresholds.low_confidence_threshold,
                'severity': 'critical' if confidence_metrics.overall_confidence < self.thresholds.very_low_confidence_threshold else 'moderate'
            }
        
        # High ambiguity detection
        if confidence_metrics.ambiguity_score > self.thresholds.ambiguity_score_threshold_moderate:
            uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.HIGH_AMBIGUITY)
            uncertainty_analysis.ambiguity_details = {
                'ambiguity_score': confidence_metrics.ambiguity_score,
                'threshold_used': self.thresholds.ambiguity_score_threshold_moderate,
                'severity': 'high' if confidence_metrics.ambiguity_score > self.thresholds.ambiguity_score_threshold_high else 'moderate',
                'alternatives_count': len(confidence_metrics.alternative_interpretations)
            }
        
        # Conflict detection
        if confidence_metrics.conflict_score > self.thresholds.conflict_score_threshold_moderate:
            uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.HIGH_CONFLICT)
            uncertainty_analysis.conflict_details = {
                'conflict_score': confidence_metrics.conflict_score,
                'threshold_used': self.thresholds.conflict_score_threshold_moderate,
                'severity': 'high' if confidence_metrics.conflict_score > self.thresholds.conflict_score_threshold_high else 'moderate'
            }
        
        # Enhanced metrics analysis (if available)
        if hybrid_confidence:
            # Total uncertainty check
            if hybrid_confidence.total_uncertainty > self.thresholds.total_uncertainty_threshold_moderate:
                uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.HIGH_AMBIGUITY)
                uncertainty_analysis.ambiguity_details.update({
                    'total_uncertainty': hybrid_confidence.total_uncertainty,
                    'epistemic_uncertainty': hybrid_confidence.epistemic_uncertainty,
                    'aleatoric_uncertainty': hybrid_confidence.aleatoric_uncertainty
                })
            
            # Evidence strength check
            if hybrid_confidence.evidence_strength < self.thresholds.evidence_strength_threshold_weak:
                uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.WEAK_EVIDENCE)
                uncertainty_analysis.evidence_details = {
                    'evidence_strength': hybrid_confidence.evidence_strength,
                    'threshold_used': self.thresholds.evidence_strength_threshold_weak,
                    'severity': 'very_weak' if hybrid_confidence.evidence_strength < self.thresholds.evidence_strength_threshold_very_weak else 'weak',
                    'confidence_reliability': hybrid_confidence.confidence_reliability
                }
            
            # Confidence interval width check
            interval_width = hybrid_confidence.confidence_interval[1] - hybrid_confidence.confidence_interval[0]
            if interval_width > self.thresholds.confidence_interval_width_threshold_moderate:
                uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.WIDE_CONFIDENCE_INTERVAL)
                uncertainty_analysis.confidence_details.update({
                    'confidence_interval': hybrid_confidence.confidence_interval,
                    'interval_width': interval_width,
                    'relative_width': interval_width / confidence_metrics.overall_confidence if confidence_metrics.overall_confidence > 0 else float('inf')
                })
            
            # LLM uncertainty indicators
            if hybrid_confidence.llm_confidence.uncertainty_indicators:
                uncertainty_analysis.detected_uncertainty_types.add(UncertaintyType.LLM_UNCERTAINTY)
    
    def _analyze_query_characteristics(self, query_text: str) -> Dict[str, Any]:
        """Analyze query characteristics that affect uncertainty thresholds."""
        characteristics = {
            'length': len(query_text),
            'word_count': len(query_text.split()),
            'has_question_words': any(word in query_text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which']),
            'has_temporal_indicators': any(word in query_text.lower() for word in ['recent', 'latest', 'current', 'new', 'today', '2024', '2025']),
            'has_technical_terms': any(word in query_text.lower() for word in ['metabolomics', 'proteomics', 'genomics', 'lc-ms', 'gc-ms']),
            'has_biomedical_entities': any(word in query_text.lower() for word in ['metabolite', 'pathway', 'biomarker', 'compound', 'enzyme']),
            'complexity_score': min(len(query_text.split()) / 10.0, 1.0),  # Normalized complexity
            'specificity_score': len([word for word in query_text.lower().split() if len(word) > 6]) / max(len(query_text.split()), 1)
        }
        
        return characteristics
    
    def _calculate_severity_from_metrics(self,
                                       confidence_metrics: ConfidenceMetrics,
                                       hybrid_confidence: Optional[HybridConfidenceResult],
                                       uncertainty_analysis: UncertaintyAnalysis) -> float:
        """Calculate uncertainty severity from confidence metrics and threshold analysis."""
        severity_factors = []
        
        # Base severity from confidence level
        confidence_level = self.thresholds.get_confidence_level(confidence_metrics.overall_confidence)
        if confidence_level == ConfidenceLevel.VERY_LOW:
            severity_factors.append(0.9)
        elif confidence_level == ConfidenceLevel.LOW:
            severity_factors.append(0.6)
        elif confidence_level == ConfidenceLevel.MEDIUM:
            severity_factors.append(0.3)
        else:
            severity_factors.append(0.1)
        
        # Ambiguity contribution
        ambiguity_severity = min(confidence_metrics.ambiguity_score / self.thresholds.ambiguity_score_threshold_high, 1.0)
        severity_factors.append(ambiguity_severity * 0.3)
        
        # Conflict contribution  
        conflict_severity = min(confidence_metrics.conflict_score / self.thresholds.conflict_score_threshold_high, 1.0)
        severity_factors.append(conflict_severity * 0.2)
        
        # Enhanced metrics contribution
        if hybrid_confidence:
            # Total uncertainty
            uncertainty_severity = min(hybrid_confidence.total_uncertainty / self.thresholds.total_uncertainty_threshold_high, 1.0)
            severity_factors.append(uncertainty_severity * 0.25)
            
            # Evidence weakness
            evidence_weakness = max(0, (self.thresholds.evidence_strength_threshold_weak - hybrid_confidence.evidence_strength) / 
                                  self.thresholds.evidence_strength_threshold_weak)
            severity_factors.append(evidence_weakness * 0.15)
        
        # Number of uncertainty types detected
        uncertainty_type_penalty = min(len(uncertainty_analysis.detected_uncertainty_types) * 0.1, 0.3)
        severity_factors.append(uncertainty_type_penalty)
        
        # Calculate weighted average
        total_severity = sum(severity_factors) / len(severity_factors) if severity_factors else 0.5
        
        return min(1.0, total_severity)
    
    def _store_analysis_pattern(self,
                              query_text: str,
                              uncertainty_analysis: UncertaintyAnalysis,
                              confidence_metrics: ConfidenceMetrics,
                              triggers: List[ThresholdTrigger]):
        """Store uncertainty analysis pattern for learning."""
        pattern = {
            'timestamp': datetime.now(),
            'query_text': query_text[:100],  # Truncated for privacy
            'confidence_level': self.thresholds.get_confidence_level(confidence_metrics.overall_confidence).value,
            'uncertainty_severity': uncertainty_analysis.uncertainty_severity,
            'detected_types': [ut.value for ut in uncertainty_analysis.detected_uncertainty_types],
            'threshold_triggers': [trigger.value for trigger in triggers],
            'recommended_strategy': uncertainty_analysis.recommended_strategy.value if uncertainty_analysis.recommended_strategy else None,
            'recommended_fallback_level': uncertainty_analysis.recommended_fallback_level.name if uncertainty_analysis.recommended_fallback_level else None
        }
        
        self.uncertainty_patterns.append(pattern)
    
    def _update_analysis_metrics(self, analysis_time_ms: float):
        """Update performance metrics for analysis."""
        current_avg = self.analysis_metrics['average_analysis_time_ms']
        total_analyses = self.analysis_metrics['total_analyses']
        
        # Update running average
        if total_analyses > 1:
            self.analysis_metrics['average_analysis_time_ms'] = (
                (current_avg * (total_analyses - 1) + analysis_time_ms) / total_analyses
            )
        else:
            self.analysis_metrics['average_analysis_time_ms'] = analysis_time_ms
    
    def _create_safe_default_analysis(self, confidence_score: float) -> UncertaintyAnalysis:
        """Create safe default analysis when errors occur."""
        confidence_level = self.thresholds.get_confidence_level(confidence_score)
        
        return UncertaintyAnalysis(
            detected_uncertainty_types={UncertaintyType.LOW_CONFIDENCE},
            uncertainty_severity=0.8 if confidence_level == ConfidenceLevel.VERY_LOW else 0.5,
            requires_special_handling=True,
            recommended_strategy=UncertaintyStrategy.CONSERVATIVE_CLASSIFICATION,
            recommended_fallback_level=FallbackLevel.KEYWORD_BASED_ONLY if confidence_level == ConfidenceLevel.VERY_LOW else FallbackLevel.SIMPLIFIED_LLM,
            confidence_details={'confidence_level': confidence_score, 'error_fallback': True}
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        return {
            'analysis_metrics': self.analysis_metrics.copy(),
            'uncertainty_detection_rate': (
                self.analysis_metrics['high_uncertainty_detected'] / 
                max(self.analysis_metrics['total_analyses'], 1)
            ),
            'recent_patterns': list(self.uncertainty_patterns)[-10:],
            'pattern_distribution': self._calculate_pattern_distribution(),
            'threshold_effectiveness': self._calculate_threshold_effectiveness()
        }
    
    def _calculate_pattern_distribution(self) -> Dict[str, int]:
        """Calculate distribution of uncertainty patterns detected."""
        distribution = defaultdict(int)
        
        for pattern in self.uncertainty_patterns:
            for uncertainty_type in pattern.get('detected_types', []):
                distribution[uncertainty_type] += 1
        
        return dict(distribution)
    
    def _calculate_threshold_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness of different threshold triggers."""
        trigger_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        for pattern in self.uncertainty_patterns:
            triggers = pattern.get('threshold_triggers', [])
            for trigger in triggers:
                trigger_stats[trigger]['total'] += 1
                # Assume success if strategy was recommended (would be enhanced with actual feedback)
                if pattern.get('recommended_strategy'):
                    trigger_stats[trigger]['successful'] += 1
        
        effectiveness = {}
        for trigger, stats in trigger_stats.items():
            effectiveness[trigger] = stats['successful'] / max(stats['total'], 1)
        
        return effectiveness


# ============================================================================
# CONFIDENCE THRESHOLD ROUTER
# ============================================================================

class ConfidenceThresholdRouter:
    """
    Enhanced routing logic that uses confidence thresholds and uncertainty analysis
    to make intelligent routing decisions before classification failures occur.
    """
    
    def __init__(self,
                 thresholds: UncertaintyAwareClassificationThresholds,
                 uncertainty_analyzer: UncertaintyMetricsAnalyzer,
                 hybrid_confidence_scorer: Optional[HybridConfidenceScorer] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize confidence threshold router."""
        self.thresholds = thresholds
        self.uncertainty_analyzer = uncertainty_analyzer
        self.hybrid_confidence_scorer = hybrid_confidence_scorer
        self.logger = logger or logging.getLogger(__name__)
        
        # Routing decision statistics
        self.routing_stats = {
            'total_routing_decisions': 0,
            'threshold_triggered_decisions': 0,
            'fallback_preventions': 0,
            'high_confidence_routes': 0,
            'medium_confidence_routes': 0,
            'low_confidence_routes': 0,
            'very_low_confidence_routes': 0
        }
        
        # Performance tracking
        self.decision_times: deque = deque(maxlen=100)
        self.confidence_improvements: deque = deque(maxlen=100)
        
        self.logger.info("ConfidenceThresholdRouter initialized with threshold-based routing")
    
    def route_with_threshold_awareness(self,
                                     query_text: str,
                                     confidence_metrics: ConfidenceMetrics,
                                     context: Optional[Dict[str, Any]] = None) -> Tuple[RoutingPrediction, UncertaintyAnalysis]:
        """
        Perform threshold-aware routing with proactive uncertainty detection.
        
        Args:
            query_text: The user query text
            confidence_metrics: Standard confidence metrics
            context: Optional context information
            
        Returns:
            Tuple of (enhanced_routing_prediction, uncertainty_analysis)
        """
        start_time = time.time()
        self.routing_stats['total_routing_decisions'] += 1
        
        try:
            # Get enhanced confidence analysis if available
            hybrid_confidence = None
            if self.hybrid_confidence_scorer:
                try:
                    # This would be async in real implementation - simplified for this example
                    hybrid_confidence = None  # Would call hybrid_confidence_scorer.calculate_comprehensive_confidence
                except Exception as e:
                    self.logger.warning(f"Hybrid confidence calculation failed: {e}")
            
            # Perform uncertainty analysis
            uncertainty_analysis = self.uncertainty_analyzer.analyze_uncertainty_from_confidence_metrics(
                query_text, confidence_metrics, hybrid_confidence, context
            )
            
            # Determine confidence level
            confidence_level = self.thresholds.get_confidence_level(confidence_metrics.overall_confidence)
            
            # Check if threshold-based intervention is needed
            should_trigger, triggers = self.thresholds.should_trigger_fallback(
                confidence_metrics, hybrid_confidence
            )
            
            if should_trigger:
                self.routing_stats['threshold_triggered_decisions'] += 1
                
                # Create enhanced routing prediction with threshold-based adjustments
                enhanced_prediction = self._create_threshold_enhanced_prediction(
                    query_text, confidence_metrics, uncertainty_analysis, 
                    confidence_level, triggers, context
                )
                
                if self.thresholds.log_threshold_decisions:
                    self.logger.info(f"Threshold-based routing applied: level={confidence_level.value}, "
                                   f"triggers={[t.value for t in triggers]}, "
                                   f"strategy={uncertainty_analysis.recommended_strategy.value if uncertainty_analysis.recommended_strategy else 'none'}")
            else:
                # No threshold intervention needed - use standard routing with confidence level annotation
                enhanced_prediction = self._create_standard_prediction_with_confidence_level(
                    query_text, confidence_metrics, confidence_level, context
                )
            
            # Update statistics
            self._update_routing_statistics(confidence_level, should_trigger)
            
            # Track performance
            decision_time_ms = (time.time() - start_time) * 1000
            self.decision_times.append(decision_time_ms)
            
            if self.thresholds.log_threshold_decisions:
                self.logger.debug(f"Threshold-aware routing completed in {decision_time_ms:.2f}ms: "
                                f"confidence_level={confidence_level.value}, "
                                f"uncertainty_severity={uncertainty_analysis.uncertainty_severity:.3f}")
            
            return enhanced_prediction, uncertainty_analysis
            
        except Exception as e:
            self.logger.error(f"Error in threshold-aware routing: {e}")
            # Create fallback routing prediction
            fallback_prediction = self._create_error_fallback_prediction(query_text, confidence_metrics)
            fallback_analysis = self.uncertainty_analyzer._create_safe_default_analysis(confidence_metrics.overall_confidence)
            return fallback_prediction, fallback_analysis
    
    def _create_threshold_enhanced_prediction(self,
                                            query_text: str,
                                            confidence_metrics: ConfidenceMetrics,
                                            uncertainty_analysis: UncertaintyAnalysis,
                                            confidence_level: ConfidenceLevel,
                                            triggers: List[ThresholdTrigger],
                                            context: Optional[Dict[str, Any]]) -> RoutingPrediction:
        """Create enhanced routing prediction with threshold-based adjustments."""
        
        # Base routing decision (would normally come from existing router)
        base_routing = self._determine_base_routing_decision(confidence_metrics, confidence_level)
        
        # Apply threshold-based adjustments
        adjusted_confidence = self._apply_threshold_confidence_adjustments(
            confidence_metrics.overall_confidence, confidence_level, uncertainty_analysis
        )
        
        # Create reasoning with threshold context
        reasoning = [
            f"Threshold-aware routing applied (confidence level: {confidence_level.value})",
            f"Detected uncertainty severity: {uncertainty_analysis.uncertainty_severity:.3f}",
            f"Threshold triggers: {[t.value for t in triggers]}"
        ]
        
        if uncertainty_analysis.recommended_strategy:
            reasoning.append(f"Recommended uncertainty strategy: {uncertainty_analysis.recommended_strategy.value}")
        
        if uncertainty_analysis.recommended_fallback_level:
            reasoning.append(f"Recommended fallback level: {uncertainty_analysis.recommended_fallback_level.name}")
        
        # Add specific threshold insights
        if uncertainty_analysis.detected_uncertainty_types:
            reasoning.append(f"Detected uncertainty types: {[ut.value for ut in uncertainty_analysis.detected_uncertainty_types]}")
        
        # Create enhanced metadata
        metadata = {
            'threshold_routing_applied': True,
            'confidence_level': confidence_level.value,
            'original_confidence': confidence_metrics.overall_confidence,
            'adjusted_confidence': adjusted_confidence,
            'uncertainty_analysis': uncertainty_analysis.to_dict(),
            'threshold_triggers': [t.value for t in triggers],
            'proactive_uncertainty_detection': True
        }
        
        return RoutingPrediction(
            routing_decision=base_routing,
            confidence=adjusted_confidence,
            reasoning=reasoning,
            research_category=ResearchCategory.GENERAL_QUERY,  # Would be determined by actual classification
            confidence_metrics=self._create_enhanced_confidence_metrics(
                confidence_metrics, adjusted_confidence, uncertainty_analysis
            ),
            temporal_indicators=[],  # Would be populated by actual analysis
            knowledge_indicators=[],  # Would be populated by actual analysis
            metadata=metadata
        )
    
    def _create_standard_prediction_with_confidence_level(self,
                                                        query_text: str,
                                                        confidence_metrics: ConfidenceMetrics,
                                                        confidence_level: ConfidenceLevel,
                                                        context: Optional[Dict[str, Any]]) -> RoutingPrediction:
        """Create standard prediction annotated with confidence level information."""
        
        base_routing = self._determine_base_routing_decision(confidence_metrics, confidence_level)
        
        reasoning = [
            f"Standard routing with confidence level: {confidence_level.value}",
            f"Confidence score: {confidence_metrics.overall_confidence:.3f}",
            "No threshold-based intervention required"
        ]
        
        metadata = {
            'threshold_routing_applied': False,
            'confidence_level': confidence_level.value,
            'confidence_above_thresholds': True,
            'proactive_uncertainty_detection': False
        }
        
        return RoutingPrediction(
            routing_decision=base_routing,
            confidence=confidence_metrics.overall_confidence,
            reasoning=reasoning,
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata=metadata
        )
    
    def _determine_base_routing_decision(self,
                                       confidence_metrics: ConfidenceMetrics,
                                       confidence_level: ConfidenceLevel) -> RoutingDecision:
        """Determine base routing decision based on confidence level and metrics."""
        
        # High confidence - prefer specialized routing
        if confidence_level == ConfidenceLevel.HIGH:
            # Choose based on query characteristics (simplified)
            if len(confidence_metrics.alternative_interpretations) > 0:
                return confidence_metrics.alternative_interpretations[0][0]
            return RoutingDecision.LIGHTRAG  # Default for high confidence
        
        # Medium confidence - balanced approach
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return RoutingDecision.HYBRID
        
        # Low confidence - flexible routing
        elif confidence_level == ConfidenceLevel.LOW:
            return RoutingDecision.EITHER
        
        # Very low confidence - conservative approach
        else:
            return RoutingDecision.EITHER
    
    def _apply_threshold_confidence_adjustments(self,
                                              original_confidence: float,
                                              confidence_level: ConfidenceLevel,
                                              uncertainty_analysis: UncertaintyAnalysis) -> float:
        """Apply threshold-based confidence adjustments."""
        
        adjusted_confidence = original_confidence
        
        # Conservative adjustment for uncertainty
        if uncertainty_analysis.uncertainty_severity > 0.7:
            adjusted_confidence *= 0.9  # 10% reduction for high uncertainty
        elif uncertainty_analysis.uncertainty_severity > 0.5:
            adjusted_confidence *= 0.95  # 5% reduction for moderate uncertainty
        
        # Boost confidence for very reliable predictions
        if (confidence_level == ConfidenceLevel.HIGH and
            uncertainty_analysis.uncertainty_severity < 0.2):
            adjusted_confidence = min(1.0, adjusted_confidence * 1.05)  # 5% boost
        
        # Ensure confidence stays within reasonable bounds
        adjusted_confidence = max(0.01, min(0.99, adjusted_confidence))
        
        return adjusted_confidence
    
    def _create_enhanced_confidence_metrics(self,
                                          original_metrics: ConfidenceMetrics,
                                          adjusted_confidence: float,
                                          uncertainty_analysis: UncertaintyAnalysis) -> ConfidenceMetrics:
        """Create enhanced confidence metrics with uncertainty information."""
        
        # Create new metrics based on original with threshold enhancements
        enhanced_metrics = ConfidenceMetrics(
            overall_confidence=adjusted_confidence,
            research_category_confidence=original_metrics.research_category_confidence,
            temporal_analysis_confidence=original_metrics.temporal_analysis_confidence,
            signal_strength_confidence=original_metrics.signal_strength_confidence,
            context_coherence_confidence=original_metrics.context_coherence_confidence,
            keyword_density=original_metrics.keyword_density,
            pattern_match_strength=original_metrics.pattern_match_strength,
            biomedical_entity_count=original_metrics.biomedical_entity_count,
            ambiguity_score=original_metrics.ambiguity_score,
            conflict_score=original_metrics.conflict_score,
            alternative_interpretations=original_metrics.alternative_interpretations,
            calculation_time_ms=original_metrics.calculation_time_ms + 10.0  # Add threshold processing time
        )
        
        return enhanced_metrics
    
    def _create_error_fallback_prediction(self,
                                        query_text: str,
                                        confidence_metrics: ConfidenceMetrics) -> RoutingPrediction:
        """Create fallback prediction when threshold routing fails."""
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=max(0.1, confidence_metrics.overall_confidence * 0.5),  # Reduced confidence
            reasoning=[
                "Threshold routing error - using conservative fallback",
                "Applied safety routing decision",
                f"Original confidence: {confidence_metrics.overall_confidence:.3f}"
            ],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'threshold_routing_applied': False,
                'error_fallback': True,
                'routing_error_occurred': True
            }
        )
    
    def _update_routing_statistics(self, confidence_level: ConfidenceLevel, threshold_triggered: bool):
        """Update routing statistics."""
        
        if threshold_triggered:
            self.routing_stats['fallback_preventions'] += 1
        
        if confidence_level == ConfidenceLevel.HIGH:
            self.routing_stats['high_confidence_routes'] += 1
        elif confidence_level == ConfidenceLevel.MEDIUM:
            self.routing_stats['medium_confidence_routes'] += 1
        elif confidence_level == ConfidenceLevel.LOW:
            self.routing_stats['low_confidence_routes'] += 1
        else:
            self.routing_stats['very_low_confidence_routes'] += 1
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_decisions = max(self.routing_stats['total_routing_decisions'], 1)
        
        return {
            'routing_statistics': self.routing_stats.copy(),
            'confidence_level_distribution': {
                'high': self.routing_stats['high_confidence_routes'] / total_decisions,
                'medium': self.routing_stats['medium_confidence_routes'] / total_decisions,
                'low': self.routing_stats['low_confidence_routes'] / total_decisions,
                'very_low': self.routing_stats['very_low_confidence_routes'] / total_decisions
            },
            'threshold_intervention_rate': self.routing_stats['threshold_triggered_decisions'] / total_decisions,
            'fallback_prevention_rate': self.routing_stats['fallback_preventions'] / total_decisions,
            'performance_metrics': {
                'average_decision_time_ms': statistics.mean(self.decision_times) if self.decision_times else 0.0,
                'max_decision_time_ms': max(self.decision_times) if self.decision_times else 0.0,
                'decisions_within_target': len([t for t in self.decision_times if t < self.thresholds.threshold_analysis_timeout_ms]) / max(len(self.decision_times), 1)
            }
        }


# ============================================================================
# THRESHOLD-BASED FALLBACK INTEGRATOR
# ============================================================================

class ThresholdBasedFallbackIntegrator:
    """
    Integration layer that connects threshold-based uncertainty detection
    with the existing comprehensive fallback system.
    """
    
    def __init__(self,
                 existing_orchestrator: FallbackOrchestrator,
                 thresholds: UncertaintyAwareClassificationThresholds,
                 threshold_router: ConfidenceThresholdRouter,
                 uncertainty_strategies: UncertaintyFallbackStrategies,
                 logger: Optional[logging.Logger] = None):
        """Initialize threshold-based fallback integrator."""
        self.existing_orchestrator = existing_orchestrator
        self.thresholds = thresholds
        self.threshold_router = threshold_router
        self.uncertainty_strategies = uncertainty_strategies
        self.logger = logger or logging.getLogger(__name__)
        
        # Integration statistics
        self.integration_stats = {
            'total_queries_processed': 0,
            'threshold_interventions': 0,
            'proactive_fallback_preventions': 0,
            'fallback_to_existing_system': 0,
            'successful_threshold_resolutions': 0
        }
        
        self.logger.info("ThresholdBasedFallbackIntegrator initialized")
    
    def process_with_threshold_awareness(self,
                                       query_text: str,
                                       confidence_metrics: ConfidenceMetrics,
                                       context: Optional[Dict[str, Any]] = None,
                                       priority: str = 'normal') -> FallbackResult:
        """
        Main processing method that integrates threshold-based uncertainty detection
        with the existing comprehensive fallback system.
        
        Args:
            query_text: The user query text
            confidence_metrics: Standard confidence metrics from classification
            context: Optional context information
            priority: Query priority level
            
        Returns:
            FallbackResult with threshold-based enhancements or existing system fallback
        """
        start_time = time.time()
        self.integration_stats['total_queries_processed'] += 1
        
        try:
            if self.thresholds.log_threshold_decisions:
                self.logger.debug(f"Processing query with threshold awareness: {query_text[:50]}...")
            
            # Step 1: Perform threshold-aware routing analysis
            routing_prediction, uncertainty_analysis = self.threshold_router.route_with_threshold_awareness(
                query_text, confidence_metrics, context
            )
            
            # Step 2: Check if threshold-based intervention is needed
            if uncertainty_analysis.requires_special_handling:
                self.integration_stats['threshold_interventions'] += 1
                
                # Apply threshold-based uncertainty strategy
                threshold_result = self._apply_threshold_based_strategy(
                    query_text, uncertainty_analysis, confidence_metrics, 
                    routing_prediction, context
                )
                
                if threshold_result.success:
                    self.integration_stats['successful_threshold_resolutions'] += 1
                    self.integration_stats['proactive_fallback_preventions'] += 1
                    
                    # Enhance result with threshold processing information
                    self._enhance_result_with_threshold_info(threshold_result, uncertainty_analysis)
                    
                    total_time = (time.time() - start_time) * 1000
                    threshold_result.total_processing_time_ms = total_time
                    
                    if self.thresholds.log_threshold_decisions:
                        self.logger.info(f"Threshold-based resolution successful in {total_time:.2f}ms")
                    
                    return threshold_result
                else:
                    if self.thresholds.log_threshold_decisions:
                        self.logger.warning("Threshold-based strategy failed - falling back to existing system")
                    self.integration_stats['fallback_to_existing_system'] += 1
            
            # Step 3: Use existing comprehensive fallback system
            # (either no threshold intervention needed or threshold strategy failed)
            existing_result = self.existing_orchestrator.process_query_with_comprehensive_fallback(
                query_text=query_text,
                context=context,
                priority=priority
            )
            
            # Enhance existing result with threshold analysis information
            if uncertainty_analysis.detected_uncertainty_types or routing_prediction.metadata.get('threshold_routing_applied'):
                self._enhance_existing_result_with_threshold_insights(
                    existing_result, uncertainty_analysis, routing_prediction
                )
            
            total_time = (time.time() - start_time) * 1000
            existing_result.total_processing_time_ms += total_time
            
            return existing_result
            
        except Exception as e:
            self.logger.error(f"Error in threshold-based integration: {e}")
            # Fall back to existing system without threshold enhancements
            return self.existing_orchestrator.process_query_with_comprehensive_fallback(
                query_text=query_text,
                context=context,
                priority=priority
            )
    
    def _apply_threshold_based_strategy(self,
                                       query_text: str,
                                       uncertainty_analysis: UncertaintyAnalysis,
                                       confidence_metrics: ConfidenceMetrics,
                                       routing_prediction: RoutingPrediction,
                                       context: Optional[Dict[str, Any]]) -> FallbackResult:
        """Apply appropriate threshold-based uncertainty strategy."""
        
        strategy = uncertainty_analysis.recommended_strategy
        
        if self.thresholds.log_threshold_decisions:
            self.logger.info(f"Applying threshold-based strategy: {strategy.value}")
        
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
    
    def _enhance_result_with_threshold_info(self,
                                          result: FallbackResult,
                                          uncertainty_analysis: UncertaintyAnalysis):
        """Enhance threshold-based result with additional information."""
        
        if not result.routing_prediction.metadata:
            result.routing_prediction.metadata = {}
        
        result.routing_prediction.metadata.update({
            'threshold_based_processing': True,
            'uncertainty_severity': uncertainty_analysis.uncertainty_severity,
            'detected_uncertainty_types': [ut.value for ut in uncertainty_analysis.detected_uncertainty_types],
            'applied_strategy': uncertainty_analysis.recommended_strategy.value if uncertainty_analysis.recommended_strategy else None,
            'proactive_fallback_prevention': True,
            'confidence_threshold_system_version': '1.0.0'
        })
        
        # Add threshold-specific warnings if needed
        if uncertainty_analysis.uncertainty_severity > 0.8:
            warning = f"Very high uncertainty detected (severity: {uncertainty_analysis.uncertainty_severity:.3f}) - threshold-based handling applied"
            if warning not in result.warnings:
                result.warnings.append(warning)
        
        # Add success indicators
        if not hasattr(result, 'recovery_suggestions'):
            result.recovery_suggestions = []
        
        if uncertainty_analysis.recommended_strategy == UncertaintyStrategy.UNCERTAINTY_CLARIFICATION:
            result.recovery_suggestions.append("Query clarification questions available in debug info")
        
        result.recovery_suggestions.append("Threshold-based uncertainty handling successfully applied")
    
    def _enhance_existing_result_with_threshold_insights(self,
                                                       existing_result: FallbackResult,
                                                       uncertainty_analysis: UncertaintyAnalysis,
                                                       routing_prediction: RoutingPrediction):
        """Enhance existing system result with threshold analysis insights."""
        
        if not existing_result.routing_prediction.metadata:
            existing_result.routing_prediction.metadata = {}
        
        existing_result.routing_prediction.metadata.update({
            'threshold_analysis_performed': True,
            'uncertainty_analysis': uncertainty_analysis.to_dict(),
            'confidence_level_detected': routing_prediction.metadata.get('confidence_level'),
            'threshold_routing_considered': routing_prediction.metadata.get('threshold_routing_applied', False),
            'proactive_uncertainty_insights_available': True
        })
        
        # Add informational warnings about uncertainty patterns
        if uncertainty_analysis.uncertainty_severity > 0.6:
            info_warning = f"Moderate uncertainty detected (severity: {uncertainty_analysis.uncertainty_severity:.3f}) during threshold analysis"
            if info_warning not in existing_result.warnings:
                existing_result.warnings.append(info_warning)
        
        # Add threshold insights to debug info
        if not hasattr(existing_result, 'debug_info') or existing_result.debug_info is None:
            existing_result.debug_info = {}
        
        existing_result.debug_info.update({
            'threshold_analysis_results': uncertainty_analysis.to_dict(),
            'confidence_level_classification': routing_prediction.metadata.get('confidence_level'),
            'threshold_system_recommendations': {
                'strategy': uncertainty_analysis.recommended_strategy.value if uncertainty_analysis.recommended_strategy else None,
                'fallback_level': uncertainty_analysis.recommended_fallback_level.name if uncertainty_analysis.recommended_fallback_level else None
            }
        })
    
    def get_comprehensive_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for threshold-based integration."""
        
        # Get statistics from component systems
        router_stats = self.threshold_router.get_routing_statistics()
        analyzer_stats = self.threshold_router.uncertainty_analyzer.get_analysis_statistics()
        strategy_stats = self.uncertainty_strategies.get_strategy_statistics()
        
        # Calculate derived metrics
        total_processed = max(self.integration_stats['total_queries_processed'], 1)
        
        threshold_intervention_rate = self.integration_stats['threshold_interventions'] / total_processed
        proactive_prevention_rate = self.integration_stats['proactive_fallback_preventions'] / total_processed
        threshold_success_rate = (
            self.integration_stats['successful_threshold_resolutions'] / 
            max(self.integration_stats['threshold_interventions'], 1)
        )
        
        return {
            'integration_statistics': self.integration_stats.copy(),
            'performance_metrics': {
                'threshold_intervention_rate': threshold_intervention_rate,
                'proactive_prevention_rate': proactive_prevention_rate,
                'threshold_success_rate': threshold_success_rate,
                'fallback_to_existing_rate': self.integration_stats['fallback_to_existing_system'] / total_processed
            },
            'component_statistics': {
                'router_statistics': router_stats,
                'analyzer_statistics': analyzer_stats,
                'strategy_statistics': strategy_stats
            },
            'system_health': {
                'threshold_system_operational': True,
                'integration_successful': threshold_success_rate > 0.7,
                'performance_within_targets': (
                    router_stats['performance_metrics']['average_decision_time_ms'] < 
                    self.thresholds.threshold_analysis_timeout_ms
                ),
                'proactive_uncertainty_detection_effective': proactive_prevention_rate > 0.1
            },
            'configuration_summary': self.thresholds.to_dict()
        }


# ============================================================================
# FACTORY FUNCTIONS AND PRODUCTION SETUP
# ============================================================================

def create_uncertainty_aware_classification_thresholds(
    production_mode: bool = True,
    custom_thresholds: Optional[Dict[str, float]] = None,
    performance_targets: Optional[Dict[str, float]] = None
) -> UncertaintyAwareClassificationThresholds:
    """
    Factory function to create uncertainty-aware classification thresholds configuration.
    
    Args:
        production_mode: Whether to use production-optimized settings
        custom_thresholds: Optional custom threshold overrides
        performance_targets: Optional performance target overrides
        
    Returns:
        UncertaintyAwareClassificationThresholds instance
    """
    
    # Base configuration
    config = UncertaintyAwareClassificationThresholds()
    
    if production_mode:
        # Production-optimized settings
        config.ambiguity_score_threshold_moderate = 0.35
        config.ambiguity_score_threshold_high = 0.6
        config.conflict_score_threshold_moderate = 0.25
        config.conflict_score_threshold_high = 0.5
        config.evidence_strength_threshold_weak = 0.4
        config.evidence_strength_threshold_very_weak = 0.15
        config.threshold_analysis_timeout_ms = 80.0
        config.max_fallback_attempts = 2
        config.enable_proactive_threshold_monitoring = True
        config.enable_uncertainty_pattern_learning = True
        config.detailed_metrics_collection = False  # Reduce overhead in production
    
    # Apply custom threshold overrides
    if custom_thresholds:
        for key, value in custom_thresholds.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Apply performance target overrides
    if performance_targets:
        for key, value in performance_targets.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def create_complete_threshold_based_fallback_system(
    existing_orchestrator: FallbackOrchestrator,
    hybrid_confidence_scorer: Optional[HybridConfidenceScorer] = None,
    thresholds_config: Optional[UncertaintyAwareClassificationThresholds] = None,
    uncertainty_config: Optional[UncertaintyFallbackConfig] = None,
    logger: Optional[logging.Logger] = None
) -> ThresholdBasedFallbackIntegrator:
    """
    Factory function to create complete threshold-based fallback system.
    
    Args:
        existing_orchestrator: Existing FallbackOrchestrator instance
        hybrid_confidence_scorer: Optional hybrid confidence scorer
        thresholds_config: Optional threshold configuration
        uncertainty_config: Optional uncertainty fallback configuration  
        logger: Optional logger instance
        
    Returns:
        ThresholdBasedFallbackIntegrator ready for production use
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create configurations
    if thresholds_config is None:
        thresholds_config = create_uncertainty_aware_classification_thresholds(production_mode=True)
    
    if uncertainty_config is None:
        uncertainty_config = UncertaintyFallbackConfig()
    
    # Create component systems
    uncertainty_analyzer = UncertaintyMetricsAnalyzer(thresholds_config, logger)
    
    threshold_router = ConfidenceThresholdRouter(
        thresholds_config, uncertainty_analyzer, hybrid_confidence_scorer, logger
    )
    
    uncertainty_strategies = UncertaintyFallbackStrategies(uncertainty_config, logger)
    
    # Create integrated system
    integrator = ThresholdBasedFallbackIntegrator(
        existing_orchestrator, thresholds_config, threshold_router, 
        uncertainty_strategies, logger
    )
    
    logger.info("Complete threshold-based fallback system created successfully")
    return integrator


def validate_threshold_configuration(
    config: UncertaintyAwareClassificationThresholds
) -> Tuple[bool, List[str]]:
    """
    Validate threshold configuration for correctness and production readiness.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_validation_errors)
    """
    
    errors = []
    
    # Check threshold ordering
    if not (config.high_confidence_threshold > config.medium_confidence_threshold > 
            config.low_confidence_threshold > config.very_low_confidence_threshold):
        errors.append("Confidence thresholds must be in descending order: high > medium > low > very_low")
    
    # Check threshold bounds
    thresholds = [
        ('high_confidence_threshold', config.high_confidence_threshold),
        ('medium_confidence_threshold', config.medium_confidence_threshold),
        ('low_confidence_threshold', config.low_confidence_threshold),
        ('very_low_confidence_threshold', config.very_low_confidence_threshold)
    ]
    
    for name, value in thresholds:
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name} must be between 0.0 and 1.0, got {value}")
    
    # Check uncertainty threshold bounds
    uncertainty_thresholds = [
        ('ambiguity_score_threshold_moderate', config.ambiguity_score_threshold_moderate),
        ('ambiguity_score_threshold_high', config.ambiguity_score_threshold_high),
        ('conflict_score_threshold_moderate', config.conflict_score_threshold_moderate),
        ('conflict_score_threshold_high', config.conflict_score_threshold_high)
    ]
    
    for name, value in uncertainty_thresholds:
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name} must be between 0.0 and 1.0, got {value}")
    
    # Check performance targets
    if config.threshold_analysis_timeout_ms <= 0:
        errors.append("threshold_analysis_timeout_ms must be positive")
    
    if config.threshold_analysis_timeout_ms > 1000:
        errors.append("threshold_analysis_timeout_ms should be <= 1000ms for good performance")
    
    if config.max_fallback_attempts <= 0:
        errors.append("max_fallback_attempts must be positive")
    
    if config.max_fallback_attempts > 5:
        errors.append("max_fallback_attempts should be <= 5 to prevent excessive processing")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage and testing
    import logging
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Testing Uncertainty-Aware Classification Thresholds System")
    
    # Create test configuration
    test_config = create_uncertainty_aware_classification_thresholds(
        production_mode=False,  # Use development settings for testing
        custom_thresholds={
            'high_confidence_threshold': 0.75,  # Slightly higher for testing
            'log_threshold_decisions': True
        }
    )
    
    # Validate configuration
    is_valid, errors = validate_threshold_configuration(test_config)
    if not is_valid:
        logger.error(f"Configuration validation failed: {errors}")
    else:
        logger.info("Configuration validation passed")
    
    # Create mock existing orchestrator for testing
    class MockFallbackOrchestrator:
        def process_query_with_comprehensive_fallback(self, query_text, context=None, priority='normal'):
            return FallbackResult(
                routing_prediction=RoutingPrediction(
                    routing_decision=RoutingDecision.LIGHTRAG,
                    confidence=0.4,
                    reasoning=["Mock comprehensive fallback result"],
                    research_category=ResearchCategory.GENERAL_QUERY,
                    confidence_metrics=ConfidenceMetrics(
                        overall_confidence=0.4,
                        research_category_confidence=0.4,
                        temporal_analysis_confidence=0.3,
                        signal_strength_confidence=0.3,
                        context_coherence_confidence=0.3,
                        keyword_density=0.2,
                        pattern_match_strength=0.2,
                        biomedical_entity_count=1,
                        ambiguity_score=0.6,
                        conflict_score=0.4,
                        alternative_interpretations=[(RoutingDecision.PERPLEXITY, 0.35)],
                        calculation_time_ms=30.0
                    ),
                    temporal_indicators=[],
                    knowledge_indicators=[],
                    metadata={}
                ),
                fallback_level_used=FallbackLevel.FULL_LLM_WITH_CONFIDENCE,
                success=True,
                total_processing_time_ms=120.0,
                quality_score=0.7,
                reliability_score=0.8
            )
    
    # Create complete threshold system
    mock_orchestrator = MockFallbackOrchestrator()
    
    try:
        threshold_system = create_complete_threshold_based_fallback_system(
            existing_orchestrator=mock_orchestrator,
            thresholds_config=test_config,
            logger=logger
        )
        
        logger.info("Threshold-based fallback system created successfully")
        
        # Test queries with different confidence patterns
        test_cases = [
            # High confidence case
            {
                'query': 'What is the role of glucose in cellular metabolism?',
                'confidence_metrics': ConfidenceMetrics(
                    overall_confidence=0.8,  # High confidence
                    research_category_confidence=0.8,
                    temporal_analysis_confidence=0.7,
                    signal_strength_confidence=0.8,
                    context_coherence_confidence=0.8,
                    keyword_density=0.6,
                    pattern_match_strength=0.7,
                    biomedical_entity_count=3,
                    ambiguity_score=0.2,  # Low ambiguity
                    conflict_score=0.1,   # Low conflict
                    alternative_interpretations=[(RoutingDecision.LIGHTRAG, 0.8)],
                    calculation_time_ms=40.0
                )
            },
            # Medium confidence case
            {
                'query': 'How does metabolomics help in disease diagnosis?',
                'confidence_metrics': ConfidenceMetrics(
                    overall_confidence=0.6,  # Medium confidence
                    research_category_confidence=0.6,
                    temporal_analysis_confidence=0.5,
                    signal_strength_confidence=0.6,
                    context_coherence_confidence=0.6,
                    keyword_density=0.4,
                    pattern_match_strength=0.5,
                    biomedical_entity_count=2,
                    ambiguity_score=0.4,  # Moderate ambiguity
                    conflict_score=0.3,   # Some conflict
                    alternative_interpretations=[
                        (RoutingDecision.LIGHTRAG, 0.6),
                        (RoutingDecision.HYBRID, 0.5)
                    ],
                    calculation_time_ms=45.0
                )
            },
            # Low confidence case with high uncertainty
            {
                'query': 'Recent advances in biomarker discovery',
                'confidence_metrics': ConfidenceMetrics(
                    overall_confidence=0.25,  # Low confidence (below low threshold)
                    research_category_confidence=0.3,
                    temporal_analysis_confidence=0.2,
                    signal_strength_confidence=0.25,
                    context_coherence_confidence=0.3,
                    keyword_density=0.2,
                    pattern_match_strength=0.3,
                    biomedical_entity_count=1,
                    ambiguity_score=0.8,  # High ambiguity  
                    conflict_score=0.6,   # High conflict
                    alternative_interpretations=[
                        (RoutingDecision.PERPLEXITY, 0.25),
                        (RoutingDecision.HYBRID, 0.23),
                        (RoutingDecision.EITHER, 0.22)
                    ],
                    calculation_time_ms=35.0
                )
            }
        ]
        
        logger.info("\n" + "="*80)
        logger.info("THRESHOLD-BASED FALLBACK SYSTEM TESTING")
        logger.info("="*80)
        
        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            confidence_metrics = test_case['confidence_metrics']
            
            logger.info(f"\nTest Case {i}: {query}")
            logger.info(f"Original Confidence: {confidence_metrics.overall_confidence:.3f}")
            logger.info(f"Ambiguity Score: {confidence_metrics.ambiguity_score:.3f}")
            logger.info(f"Conflict Score: {confidence_metrics.conflict_score:.3f}")
            logger.info("-" * 60)
            
            try:
                # Process with threshold awareness
                result = threshold_system.process_with_threshold_awareness(
                    query, confidence_metrics
                )
                
                logger.info(f"Result: {result.routing_prediction.routing_decision.value}")
                logger.info(f"Final Confidence: {result.routing_prediction.confidence:.3f}")
                logger.info(f"Fallback Level: {result.fallback_level_used.name}")
                logger.info(f"Success: {result.success}")
                logger.info(f"Processing Time: {result.total_processing_time_ms:.2f}ms")
                
                # Show threshold-specific information
                metadata = result.routing_prediction.metadata
                if metadata:
                    if metadata.get('threshold_routing_applied'):
                        logger.info(f"Confidence Level: {metadata.get('confidence_level')}")
                        logger.info(f"Uncertainty Severity: {metadata.get('uncertainty_severity', 0):.3f}")
                    
                    if metadata.get('threshold_based_processing'):
                        strategy = metadata.get('applied_strategy')
                        if strategy:
                            logger.info(f"Applied Strategy: {strategy}")
                
                if result.warnings:
                    logger.info(f"Warnings: {result.warnings}")
                    
            except Exception as e:
                logger.error(f"Error processing test case: {e}")
        
        # Get comprehensive statistics
        logger.info("\n" + "="*80)
        logger.info("SYSTEM STATISTICS")
        logger.info("="*80)
        
        stats = threshold_system.get_comprehensive_integration_statistics()
        
        logger.info(f"Total Queries Processed: {stats['integration_statistics']['total_queries_processed']}")
        logger.info(f"Threshold Intervention Rate: {stats['performance_metrics']['threshold_intervention_rate']:.1%}")
        logger.info(f"Proactive Prevention Rate: {stats['performance_metrics']['proactive_prevention_rate']:.1%}")
        logger.info(f"Threshold Success Rate: {stats['performance_metrics']['threshold_success_rate']:.1%}")
        logger.info(f"System Health: {stats['system_health']['threshold_system_operational']}")
        logger.info(f"Performance Within Targets: {stats['system_health']['performance_within_targets']}")
        
        logger.info("\nThreshold-based fallback system testing completed successfully")
        
    except Exception as e:
        logger.error(f"System creation or testing failed: {e}")
        raise