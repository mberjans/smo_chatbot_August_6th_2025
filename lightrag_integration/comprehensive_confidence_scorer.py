"""
Comprehensive Confidence Scoring System for Clinical Metabolomics Oracle

This module provides a sophisticated confidence scoring system that integrates 
LLM-based semantic classification confidence with keyword-based confidence metrics,
providing multi-dimensional confidence analysis with calibration and historical accuracy tracking.

Key Features:
    - Hybrid confidence scoring integrating LLM and keyword-based approaches
    - LLM-specific confidence metrics with consistency analysis and reasoning quality
    - Confidence calibration based on historical performance and accuracy feedback
    - Multi-dimensional confidence analysis with component breakdown
    - Confidence intervals and uncertainty quantification
    - Adaptive weighting based on query characteristics and historical performance
    - Integration with existing ConfidenceMetrics infrastructure
    - Real-time confidence monitoring and validation

Classes:
    - LLMConfidenceAnalyzer: Advanced analysis of LLM response confidence
    - ConfidenceCalibrator: Historical accuracy tracking and confidence calibration  
    - HybridConfidenceScorer: Main hybrid confidence scoring engine
    - ConfidenceValidator: Validation and accuracy measurement framework
    - EnhancedConfidenceMetrics: Extended confidence metrics with LLM integration

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import json
import time
import statistics
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import hashlib
import numpy as np
from pathlib import Path
import threading

# Import existing components for integration
try:
    from .query_router import ConfidenceMetrics, RoutingPrediction, RoutingDecision, BiomedicalQueryRouter
    from .research_categorizer import CategoryPrediction, ResearchCategorizer
    from .enhanced_llm_classifier import ClassificationResult, EnhancedLLMQueryClassifier
    from .cost_persistence import ResearchCategory
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# ENHANCED CONFIDENCE METRICS AND DATACLASSES
# ============================================================================

class ConfidenceSource(Enum):
    """Sources of confidence information."""
    LLM_SEMANTIC = "llm_semantic"
    KEYWORD_BASED = "keyword_based" 
    PATTERN_MATCHING = "pattern_matching"
    HISTORICAL_CALIBRATION = "historical_calibration"
    ENSEMBLE_VOTING = "ensemble_voting"


@dataclass
class LLMConfidenceAnalysis:
    """Detailed analysis of LLM confidence with consistency metrics."""
    
    # Core LLM confidence metrics
    raw_confidence: float  # Original LLM confidence score
    calibrated_confidence: float  # Calibrated based on historical accuracy
    reasoning_quality_score: float  # Quality of LLM reasoning (0-1)
    consistency_score: float  # Consistency across multiple attempts (0-1)
    
    # Response analysis
    response_length: int  # Length of LLM response
    reasoning_depth: int  # Depth of reasoning provided (1-5)
    uncertainty_indicators: List[str]  # Explicit uncertainty expressions
    confidence_expressions: List[str]  # Confidence expressions found in response
    
    # Token-level analysis (if available)
    token_probabilities: Optional[List[float]] = None
    average_token_probability: Optional[float] = None
    min_token_probability: Optional[float] = None
    
    # Multi-attempt consistency
    alternative_responses: List[str] = field(default_factory=list)
    response_similarity: Optional[float] = None  # Similarity between attempts (0-1)
    
    # Temporal analysis
    response_time_ms: float = 0.0
    model_temperature: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass 
class KeywordConfidenceAnalysis:
    """Enhanced analysis of keyword-based confidence."""
    
    # Pattern matching confidence
    pattern_match_confidence: float
    keyword_density_confidence: float
    biomedical_entity_confidence: float
    domain_specificity_confidence: float
    
    # Signal strength analysis
    total_biomedical_signals: int
    strong_signals: int  # High-confidence indicators
    weak_signals: int   # Low-confidence indicators
    conflicting_signals: int  # Contradictory indicators
    
    # Context coherence
    semantic_coherence_score: float  # How well keywords relate to each other
    domain_alignment_score: float   # How well query aligns with biomedical domain
    query_completeness_score: float # How complete the query appears
    
    # Historical performance
    pattern_success_rate: float = 0.0  # Historical success rate for detected patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ConfidenceCalibrationData:
    """Data for confidence calibration and historical tracking."""
    
    # Historical accuracy data
    prediction_accuracies: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_bins: Dict[str, List[float]] = field(default_factory=dict)  # Binned accuracies
    
    # Calibration metrics
    calibration_slope: float = 1.0  # Slope of confidence vs accuracy
    calibration_intercept: float = 0.0  # Intercept of calibration curve
    brier_score: float = 0.0  # Measure of confidence calibration quality
    
    # Time-based degradation
    last_calibration_update: datetime = field(default_factory=datetime.now)
    time_decay_factor: float = 0.95  # Decay factor for historical data
    
    # Source-specific calibration
    llm_calibration_factor: float = 1.0
    keyword_calibration_factor: float = 1.0
    
    def update_accuracy(self, predicted_confidence: float, actual_accuracy: bool):
        """Update historical accuracy data."""
        self.prediction_accuracies.append({
            'confidence': predicted_confidence,
            'accuracy': 1.0 if actual_accuracy else 0.0,
            'timestamp': datetime.now()
        })
        
        # Update confidence bins
        confidence_bin = self._get_confidence_bin(predicted_confidence)
        if confidence_bin not in self.confidence_bins:
            self.confidence_bins[confidence_bin] = []
        self.confidence_bins[confidence_bin].append(1.0 if actual_accuracy else 0.0)
        
        # Limit bin size
        if len(self.confidence_bins[confidence_bin]) > 100:
            self.confidence_bins[confidence_bin] = self.confidence_bins[confidence_bin][-100:]
    
    def _get_confidence_bin(self, confidence: float) -> str:
        """Get confidence bin for given confidence score."""
        bin_size = 0.1
        bin_index = int(confidence / bin_size)
        bin_start = bin_index * bin_size
        return f"{bin_start:.1f}-{bin_start + bin_size:.1f}"


@dataclass
class HybridConfidenceResult:
    """Result of hybrid confidence scoring with detailed breakdown."""
    
    # Primary confidence scores
    overall_confidence: float  # Final weighted confidence (0-1)
    confidence_interval: Tuple[float, float]  # Confidence interval (lower, upper)
    
    # Component confidences
    llm_confidence: LLMConfidenceAnalysis
    keyword_confidence: KeywordConfidenceAnalysis
    
    # Weighting and combination
    llm_weight: float  # Weight given to LLM confidence (0-1)
    keyword_weight: float  # Weight given to keyword confidence (0-1)
    calibration_adjustment: float  # Adjustment from historical calibration
    
    # Uncertainty quantification
    epistemic_uncertainty: float  # Model uncertainty (what we don't know)
    aleatoric_uncertainty: float  # Data uncertainty (inherent noise)
    total_uncertainty: float  # Combined uncertainty
    
    # Quality indicators
    confidence_reliability: float  # How reliable this confidence estimate is (0-1)
    evidence_strength: float  # Strength of evidence for this confidence (0-1)
    
    # Alternative scenarios
    alternative_confidences: List[Tuple[str, float]] = field(default_factory=list)
    
    # Metadata
    calculation_time_ms: float = 0.0
    calibration_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert tuple to list for JSON serialization
        result['confidence_interval'] = list(self.confidence_interval)
        return result


# ============================================================================
# LLM CONFIDENCE ANALYZER
# ============================================================================

class LLMConfidenceAnalyzer:
    """
    Advanced analyzer for LLM response confidence with consistency analysis
    and reasoning quality assessment.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Confidence expression patterns
        self.confidence_patterns = {
            'high_confidence': [
                r'(?:very|extremely|highly)\s+(?:confident|certain|sure)',
                r'(?:definitely|certainly|absolutely)',
                r'(?:clear|obvious|evident)\s+(?:that|indication)',
                r'(?:strong|compelling)\s+evidence',
            ],
            'medium_confidence': [
                r'(?:likely|probably|appears?\s+to)',
                r'(?:suggests?\s+that|indicates?\s+that)',
                r'(?:reasonable|good)\s+(?:confidence|evidence)',
                r'(?:most\s+)?(?:likely|probable)'
            ],
            'low_confidence': [
                r'(?:might|may|could)\s+(?:be|indicate)',
                r'(?:possible|potential|uncertain)',
                r'(?:limited|weak)\s+evidence',
                r'(?:difficult\s+to\s+determine|hard\s+to\s+say)'
            ],
            'uncertainty': [
                r'(?:uncertain|unsure|unclear)',
                r'(?:don\'t\s+know|not\s+sure|can\'t\s+say)',
                r'(?:ambiguous|vague|inconclusive)',
                r'(?:need\s+more|insufficient)\s+(?:information|evidence)'
            ]
        }
        
        # Reasoning quality indicators
        self.reasoning_quality_indicators = {
            'structured_reasoning': [
                r'(?:first|second|third|finally)',
                r'(?:because|therefore|thus|hence)',
                r'(?:evidence\s+shows|data\s+suggests)',
                r'(?:based\s+on|according\s+to)'
            ],
            'domain_knowledge': [
                r'(?:metabolomics|proteomics|genomics)',
                r'(?:biomarker|pathway|mechanism)',
                r'(?:clinical|therapeutic|diagnostic)',
                r'(?:lc-ms|gc-ms|nmr|spectroscopy)'
            ],
            'logical_connections': [
                r'(?:leads\s+to|results\s+in|causes)',
                r'(?:relationship\s+between|connection\s+with)',
                r'(?:correlates?\s+with|associates?\s+with)',
                r'(?:if.*then|when.*then)'
            ]
        }
        
        # Response consistency tracking
        self.consistency_cache = {}
        self.cache_lock = threading.Lock()
        
    def analyze_llm_confidence(self, 
                              classification_result: ClassificationResult,
                              llm_response_text: Optional[str] = None,
                              alternative_responses: Optional[List[str]] = None,
                              response_metadata: Optional[Dict[str, Any]] = None) -> LLMConfidenceAnalysis:
        """
        Analyze LLM confidence with comprehensive metrics.
        
        Args:
            classification_result: LLM classification result
            llm_response_text: Full LLM response text for analysis
            alternative_responses: Alternative responses for consistency analysis
            response_metadata: Additional metadata about the response
            
        Returns:
            LLMConfidenceAnalysis with detailed confidence metrics
        """
        start_time = time.time()
        
        # Extract basic information
        raw_confidence = classification_result.confidence
        reasoning_text = classification_result.reasoning or ""
        response_text = llm_response_text or reasoning_text
        
        # Analyze reasoning quality
        reasoning_quality = self._analyze_reasoning_quality(response_text)
        
        # Analyze confidence expressions
        confidence_expressions = self._extract_confidence_expressions(response_text)
        uncertainty_indicators = self._extract_uncertainty_indicators(response_text)
        
        # Analyze response consistency if alternatives provided
        consistency_score = 1.0  # Default perfect consistency
        response_similarity = None
        if alternative_responses and len(alternative_responses) > 1:
            consistency_score, response_similarity = self._analyze_response_consistency(
                response_text, alternative_responses
            )
        
        # Calculate calibrated confidence
        calibrated_confidence = self._apply_confidence_calibration(
            raw_confidence, reasoning_quality, consistency_score, len(uncertainty_indicators)
        )
        
        # Extract response metadata
        response_time_ms = response_metadata.get('response_time_ms', 0.0) if response_metadata else 0.0
        model_temperature = response_metadata.get('temperature', 0.1) if response_metadata else 0.1
        
        analysis = LLMConfidenceAnalysis(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            reasoning_quality_score=reasoning_quality,
            consistency_score=consistency_score,
            response_length=len(response_text),
            reasoning_depth=self._assess_reasoning_depth(response_text),
            uncertainty_indicators=uncertainty_indicators,
            confidence_expressions=confidence_expressions,
            alternative_responses=alternative_responses or [],
            response_similarity=response_similarity,
            response_time_ms=response_time_ms,
            model_temperature=model_temperature
        )
        
        self.logger.debug(f"LLM confidence analysis completed in {(time.time() - start_time)*1000:.2f}ms")
        return analysis
    
    def _analyze_reasoning_quality(self, response_text: str) -> float:
        """Analyze the quality of LLM reasoning."""
        if not response_text:
            return 0.0
        
        text_lower = response_text.lower()
        quality_score = 0.0
        
        # Check for structured reasoning
        structured_count = sum(
            len([m for pattern in patterns for m in __import__('re').findall(pattern, text_lower)])
            for patterns in [self.reasoning_quality_indicators['structured_reasoning']]
        )
        quality_score += min(structured_count * 0.2, 0.4)
        
        # Check for domain knowledge
        domain_count = sum(
            len([m for pattern in patterns for m in __import__('re').findall(pattern, text_lower)])
            for patterns in [self.reasoning_quality_indicators['domain_knowledge']]
        )
        quality_score += min(domain_count * 0.15, 0.3)
        
        # Check for logical connections
        logical_count = sum(
            len([m for pattern in patterns for m in __import__('re').findall(pattern, text_lower)])
            for patterns in [self.reasoning_quality_indicators['logical_connections']]
        )
        quality_score += min(logical_count * 0.1, 0.3)
        
        return min(quality_score, 1.0)
    
    def _extract_confidence_expressions(self, response_text: str) -> List[str]:
        """Extract confidence expressions from response."""
        if not response_text:
            return []
        
        import re
        text_lower = response_text.lower()
        expressions = []
        
        for confidence_level, patterns in self.confidence_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                expressions.extend([f"{confidence_level}:{match}" for match in matches])
        
        return expressions[:10]  # Limit to prevent overflow
    
    def _extract_uncertainty_indicators(self, response_text: str) -> List[str]:
        """Extract uncertainty indicators from response."""
        if not response_text:
            return []
        
        import re
        text_lower = response_text.lower()
        indicators = []
        
        for pattern in self.confidence_patterns['uncertainty']:
            matches = re.findall(pattern, text_lower)
            indicators.extend(matches)
        
        return indicators[:5]  # Limit to prevent overflow
    
    def _analyze_response_consistency(self, primary_response: str, 
                                    alternative_responses: List[str]) -> Tuple[float, float]:
        """Analyze consistency between multiple LLM responses."""
        if not alternative_responses:
            return 1.0, None
        
        # Simple consistency metric based on text similarity
        similarities = []
        for alt_response in alternative_responses:
            similarity = self._calculate_text_similarity(primary_response, alt_response)
            similarities.append(similarity)
        
        consistency_score = statistics.mean(similarities) if similarities else 1.0
        avg_similarity = statistics.mean(similarities) if similarities else None
        
        return consistency_score, avg_similarity
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two responses."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity (can be enhanced with more sophisticated methods)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_reasoning_depth(self, response_text: str) -> int:
        """Assess the depth of reasoning provided (1-5 scale)."""
        if not response_text:
            return 1
        
        # Count reasoning indicators
        reasoning_indicators = [
            'because', 'therefore', 'thus', 'hence', 'since', 'due to',
            'leads to', 'results in', 'causes', 'explains',
            'evidence', 'data', 'research', 'study', 'analysis'
        ]
        
        text_lower = response_text.lower()
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in text_lower)
        
        # Map to 1-5 scale
        if indicator_count >= 8:
            return 5  # Very deep reasoning
        elif indicator_count >= 6:
            return 4  # Deep reasoning
        elif indicator_count >= 4:
            return 3  # Moderate reasoning
        elif indicator_count >= 2:
            return 2  # Basic reasoning
        else:
            return 1  # Minimal reasoning
    
    def _apply_confidence_calibration(self, raw_confidence: float, 
                                    reasoning_quality: float,
                                    consistency_score: float,
                                    uncertainty_count: int) -> float:
        """Apply calibration adjustments to raw confidence."""
        
        # Start with raw confidence
        calibrated = raw_confidence
        
        # Adjust based on reasoning quality
        reasoning_adjustment = (reasoning_quality - 0.5) * 0.1  # +/- 5%
        calibrated += reasoning_adjustment
        
        # Adjust based on consistency
        consistency_adjustment = (consistency_score - 0.8) * 0.1  # Penalty if inconsistent
        calibrated += consistency_adjustment
        
        # Adjust based on uncertainty indicators
        uncertainty_penalty = uncertainty_count * 0.05  # 5% penalty per uncertainty indicator
        calibrated -= uncertainty_penalty
        
        # Ensure bounds
        return max(0.0, min(1.0, calibrated))


# ============================================================================
# CONFIDENCE CALIBRATOR
# ============================================================================

class ConfidenceCalibrator:
    """
    Historical accuracy tracking and confidence calibration system.
    """
    
    def __init__(self, 
                 calibration_data_path: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.calibration_data_path = calibration_data_path
        
        # Initialize calibration data
        self.calibration_data = ConfidenceCalibrationData()
        
        # Load historical data if available
        if calibration_data_path:
            self._load_calibration_data()
        
        # Calibration update frequency
        self.update_frequency = 50  # Recalibrate every 50 predictions
        self.predictions_since_update = 0
        
    def record_prediction_outcome(self, 
                                 predicted_confidence: float,
                                 actual_accuracy: bool,
                                 confidence_source: ConfidenceSource,
                                 query_text: str = None) -> None:
        """
        Record the outcome of a confidence prediction for calibration.
        
        Args:
            predicted_confidence: The confidence that was predicted
            actual_accuracy: Whether the prediction was actually accurate
            confidence_source: Source of the confidence score
            query_text: Original query text for analysis
        """
        
        # Update calibration data
        self.calibration_data.update_accuracy(predicted_confidence, actual_accuracy)
        
        # Update source-specific calibration
        if confidence_source == ConfidenceSource.LLM_SEMANTIC:
            self._update_source_calibration('llm', predicted_confidence, actual_accuracy)
        elif confidence_source == ConfidenceSource.KEYWORD_BASED:
            self._update_source_calibration('keyword', predicted_confidence, actual_accuracy)
        
        self.predictions_since_update += 1
        
        # Recalibrate if needed
        if self.predictions_since_update >= self.update_frequency:
            self._recalibrate_confidence_scores()
            self.predictions_since_update = 0
        
        self.logger.debug(f"Recorded prediction outcome: conf={predicted_confidence:.3f}, "
                         f"accurate={actual_accuracy}, source={confidence_source.value}")
    
    def calibrate_confidence(self, 
                           raw_confidence: float,
                           confidence_source: ConfidenceSource) -> float:
        """
        Apply calibration to a raw confidence score.
        
        Args:
            raw_confidence: Raw confidence score (0-1)
            confidence_source: Source of the confidence
            
        Returns:
            Calibrated confidence score
        """
        
        # Apply general calibration curve
        calibrated = (raw_confidence * self.calibration_data.calibration_slope + 
                     self.calibration_data.calibration_intercept)
        
        # Apply source-specific calibration
        if confidence_source == ConfidenceSource.LLM_SEMANTIC:
            calibrated *= self.calibration_data.llm_calibration_factor
        elif confidence_source == ConfidenceSource.KEYWORD_BASED:
            calibrated *= self.calibration_data.keyword_calibration_factor
        
        # Apply time decay if calibration data is old
        time_since_update = datetime.now() - self.calibration_data.last_calibration_update
        if time_since_update.total_seconds() > 86400:  # More than 24 hours
            decay_factor = self.calibration_data.time_decay_factor ** (time_since_update.days)
            calibrated = raw_confidence * decay_factor + calibrated * (1 - decay_factor)
        
        return max(0.0, min(1.0, calibrated))
    
    def get_confidence_interval(self, 
                               calibrated_confidence: float,
                               evidence_strength: float = 1.0) -> Tuple[float, float]:
        """
        Calculate confidence interval for a confidence score.
        
        Args:
            calibrated_confidence: Calibrated confidence score
            evidence_strength: Strength of evidence (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        
        # Calculate interval width based on historical performance
        base_width = 0.1  # Base interval width
        
        # Adjust width based on calibration quality (Brier score)
        brier_adjustment = self.calibration_data.brier_score * 0.2
        
        # Adjust width based on evidence strength
        evidence_adjustment = (1.0 - evidence_strength) * 0.15
        
        interval_width = base_width + brier_adjustment + evidence_adjustment
        
        # Calculate bounds
        lower_bound = max(0.0, calibrated_confidence - interval_width / 2)
        upper_bound = min(1.0, calibrated_confidence + interval_width / 2)
        
        return (lower_bound, upper_bound)
    
    def _update_source_calibration(self, source: str, predicted: float, accurate: bool):
        """Update source-specific calibration factors."""
        
        # Simple learning rate for source calibration
        learning_rate = 0.01
        target = 1.0 if accurate else 0.0
        error = predicted - target
        
        if source == 'llm':
            self.calibration_data.llm_calibration_factor -= learning_rate * error
            self.calibration_data.llm_calibration_factor = max(0.1, min(2.0, 
                self.calibration_data.llm_calibration_factor))
        elif source == 'keyword':
            self.calibration_data.keyword_calibration_factor -= learning_rate * error
            self.calibration_data.keyword_calibration_factor = max(0.1, min(2.0,
                self.calibration_data.keyword_calibration_factor))
    
    def _recalibrate_confidence_scores(self):
        """Recalculate calibration parameters based on historical data."""
        
        if len(self.calibration_data.prediction_accuracies) < 10:
            return  # Need minimum data for calibration
        
        # Extract confidence and accuracy arrays
        confidences = []
        accuracies = []
        
        for prediction in self.calibration_data.prediction_accuracies:
            confidences.append(prediction['confidence'])
            accuracies.append(prediction['accuracy'])
        
        # Calculate calibration slope and intercept using simple linear regression
        if len(confidences) > 1:
            try:
                # Convert to numpy arrays for calculation
                conf_array = np.array(confidences) if 'numpy' in globals() else confidences
                acc_array = np.array(accuracies) if 'numpy' in globals() else accuracies
                
                if 'numpy' in globals():
                    # Use numpy for more accurate calculation
                    slope, intercept = np.polyfit(conf_array, acc_array, 1)
                else:
                    # Simple calculation without numpy
                    mean_conf = statistics.mean(confidences)
                    mean_acc = statistics.mean(accuracies)
                    
                    numerator = sum((c - mean_conf) * (a - mean_acc) 
                                  for c, a in zip(confidences, accuracies))
                    denominator = sum((c - mean_conf) ** 2 for c in confidences)
                    
                    slope = numerator / denominator if denominator != 0 else 1.0
                    intercept = mean_acc - slope * mean_conf
                
                self.calibration_data.calibration_slope = max(0.1, min(2.0, slope))
                self.calibration_data.calibration_intercept = max(-0.5, min(0.5, intercept))
                
                # Calculate Brier score for calibration quality
                brier_score = statistics.mean([(c - a) ** 2 for c, a in zip(confidences, accuracies)])
                self.calibration_data.brier_score = brier_score
                
                self.calibration_data.last_calibration_update = datetime.now()
                
                self.logger.debug(f"Confidence calibration updated: slope={slope:.3f}, "
                                f"intercept={intercept:.3f}, brier={brier_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to recalibrate confidence: {e}")
    
    def _load_calibration_data(self):
        """Load historical calibration data from file."""
        if not self.calibration_data_path or not Path(self.calibration_data_path).exists():
            return
        
        try:
            with open(self.calibration_data_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct calibration data
            self.calibration_data.calibration_slope = data.get('calibration_slope', 1.0)
            self.calibration_data.calibration_intercept = data.get('calibration_intercept', 0.0)
            self.calibration_data.brier_score = data.get('brier_score', 0.0)
            self.calibration_data.llm_calibration_factor = data.get('llm_calibration_factor', 1.0)
            self.calibration_data.keyword_calibration_factor = data.get('keyword_calibration_factor', 1.0)
            
            # Load recent predictions
            if 'recent_predictions' in data:
                for pred in data['recent_predictions'][-100:]:  # Last 100 predictions
                    self.calibration_data.prediction_accuracies.append(pred)
            
            self.logger.info(f"Loaded calibration data from {self.calibration_data_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration data: {e}")
    
    def save_calibration_data(self):
        """Save calibration data to file."""
        if not self.calibration_data_path:
            return
        
        try:
            data = {
                'calibration_slope': self.calibration_data.calibration_slope,
                'calibration_intercept': self.calibration_data.calibration_intercept,
                'brier_score': self.calibration_data.brier_score,
                'llm_calibration_factor': self.calibration_data.llm_calibration_factor,
                'keyword_calibration_factor': self.calibration_data.keyword_calibration_factor,
                'recent_predictions': list(self.calibration_data.prediction_accuracies)[-100:],
                'last_update': self.calibration_data.last_calibration_update.isoformat()
            }
            
            with open(self.calibration_data_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved calibration data to {self.calibration_data_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration data: {e}")
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get comprehensive calibration statistics."""
        
        total_predictions = len(self.calibration_data.prediction_accuracies)
        
        if total_predictions == 0:
            return {
                'total_predictions': 0,
                'overall_accuracy': 0.0,
                'calibration_slope': self.calibration_data.calibration_slope,
                'calibration_intercept': self.calibration_data.calibration_intercept,
                'brier_score': self.calibration_data.brier_score
            }
        
        # Calculate overall accuracy
        accuracies = [pred['accuracy'] for pred in self.calibration_data.prediction_accuracies]
        overall_accuracy = statistics.mean(accuracies)
        
        # Calculate confidence bin accuracies
        bin_accuracies = {}
        for bin_name, bin_accuracies_list in self.calibration_data.confidence_bins.items():
            if bin_accuracies_list:
                bin_accuracies[bin_name] = statistics.mean(bin_accuracies_list)
        
        return {
            'total_predictions': total_predictions,
            'overall_accuracy': overall_accuracy,
            'calibration_slope': self.calibration_data.calibration_slope,
            'calibration_intercept': self.calibration_data.calibration_intercept,
            'brier_score': self.calibration_data.brier_score,
            'llm_calibration_factor': self.calibration_data.llm_calibration_factor,
            'keyword_calibration_factor': self.calibration_data.keyword_calibration_factor,
            'confidence_bin_accuracies': bin_accuracies,
            'last_calibration_update': self.calibration_data.last_calibration_update.isoformat()
        }


# ============================================================================
# HYBRID CONFIDENCE SCORER - Main Engine
# ============================================================================

class HybridConfidenceScorer:
    """
    Main hybrid confidence scoring engine that integrates LLM and keyword-based
    confidence with sophisticated weighting, calibration, and uncertainty quantification.
    """
    
    def __init__(self, 
                 biomedical_router: Optional[BiomedicalQueryRouter] = None,
                 llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                 calibration_data_path: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or logging.getLogger(__name__)
        self.biomedical_router = biomedical_router
        self.llm_classifier = llm_classifier
        
        # Initialize component analyzers
        self.llm_analyzer = LLMConfidenceAnalyzer(self.logger)
        self.calibrator = ConfidenceCalibrator(calibration_data_path, self.logger)
        
        # Adaptive weighting parameters
        self.weighting_params = {
            'base_llm_weight': 0.6,  # Base weight for LLM confidence
            'base_keyword_weight': 0.4,  # Base weight for keyword confidence
            'query_length_factor': 0.1,  # Adjustment based on query length
            'domain_specificity_factor': 0.15,  # Adjustment based on domain specificity
            'consistency_factor': 0.2  # Adjustment based on response consistency
        }
        
        # Performance tracking
        self.scoring_times = deque(maxlen=100)
        self.confidence_predictions = deque(maxlen=1000)
        
        self.logger.info("Hybrid confidence scorer initialized")
    
    async def calculate_comprehensive_confidence(self, 
                                               query_text: str,
                                               llm_result: Optional[ClassificationResult] = None,
                                               keyword_prediction: Optional[CategoryPrediction] = None,
                                               context: Optional[Dict[str, Any]] = None,
                                               llm_response_metadata: Optional[Dict[str, Any]] = None) -> HybridConfidenceResult:
        """
        Calculate comprehensive confidence score integrating LLM and keyword approaches.
        
        Args:
            query_text: The original query text
            llm_result: LLM classification result (if available)
            keyword_prediction: Keyword-based prediction (if available)
            context: Additional context information
            llm_response_metadata: Metadata about LLM response
            
        Returns:
            HybridConfidenceResult with detailed confidence analysis
        """
        start_time = time.time()
        
        try:
            # Get LLM analysis if available
            if llm_result is None and self.llm_classifier is not None:
                try:
                    llm_result, llm_response_metadata = await self.llm_classifier.classify_query(
                        query_text, context
                    )
                except Exception as e:
                    self.logger.warning(f"LLM classification failed: {e}")
                    llm_result = None
            
            # Get keyword analysis if available
            if keyword_prediction is None and self.biomedical_router is not None:
                try:
                    routing_prediction = self.biomedical_router.route_query(query_text, context)
                    keyword_prediction = CategoryPrediction(
                        category=routing_prediction.research_category,
                        confidence=routing_prediction.confidence,
                        evidence=routing_prediction.knowledge_indicators or []
                    )
                except Exception as e:
                    self.logger.warning(f"Keyword analysis failed: {e}")
                    keyword_prediction = None
            
            # Analyze LLM confidence
            llm_confidence_analysis = None
            if llm_result:
                llm_confidence_analysis = self.llm_analyzer.analyze_llm_confidence(
                    llm_result,
                    llm_response_text=llm_result.reasoning,
                    response_metadata=llm_response_metadata
                )
            
            # Analyze keyword confidence
            keyword_confidence_analysis = self._analyze_keyword_confidence(
                query_text, keyword_prediction, context
            )
            
            # Calculate adaptive weights
            llm_weight, keyword_weight = self._calculate_adaptive_weights(
                query_text, llm_confidence_analysis, keyword_confidence_analysis
            )
            
            # Combine confidences
            combined_confidence = self._combine_confidences(
                llm_confidence_analysis, keyword_confidence_analysis,
                llm_weight, keyword_weight
            )
            
            # Apply calibration
            calibrated_confidence, calibration_adjustment = self._apply_calibration(
                combined_confidence, llm_confidence_analysis, keyword_confidence_analysis
            )
            
            # Calculate confidence interval
            evidence_strength = self._calculate_evidence_strength(
                llm_confidence_analysis, keyword_confidence_analysis
            )
            confidence_interval = self.calibrator.get_confidence_interval(
                calibrated_confidence, evidence_strength
            )
            
            # Calculate uncertainty metrics
            uncertainties = self._calculate_uncertainty_metrics(
                llm_confidence_analysis, keyword_confidence_analysis, evidence_strength
            )
            
            # Create comprehensive result
            result = HybridConfidenceResult(
                overall_confidence=calibrated_confidence,
                confidence_interval=confidence_interval,
                llm_confidence=llm_confidence_analysis or self._create_default_llm_analysis(),
                keyword_confidence=keyword_confidence_analysis,
                llm_weight=llm_weight,
                keyword_weight=keyword_weight,
                calibration_adjustment=calibration_adjustment,
                epistemic_uncertainty=uncertainties['epistemic'],
                aleatoric_uncertainty=uncertainties['aleatoric'],
                total_uncertainty=uncertainties['total'],
                confidence_reliability=self._calculate_confidence_reliability(
                    llm_confidence_analysis, keyword_confidence_analysis, evidence_strength
                ),
                evidence_strength=evidence_strength,
                alternative_confidences=self._generate_alternative_confidences(
                    llm_confidence_analysis, keyword_confidence_analysis
                ),
                calculation_time_ms=(time.time() - start_time) * 1000,
                calibration_version="1.0"
            )
            
            # Track performance
            self.scoring_times.append((time.time() - start_time) * 1000)
            self.confidence_predictions.append({
                'confidence': calibrated_confidence,
                'timestamp': datetime.now(),
                'query_length': len(query_text.split())
            })
            
            self.logger.debug(f"Comprehensive confidence calculated: {calibrated_confidence:.3f} "
                            f"(LLM: {llm_weight:.2f}, KW: {keyword_weight:.2f}) "
                            f"in {result.calculation_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate comprehensive confidence: {e}")
            # Return fallback result
            return self._create_fallback_confidence_result(query_text, start_time)
    
    def _analyze_keyword_confidence(self, 
                                  query_text: str,
                                  keyword_prediction: Optional[CategoryPrediction],
                                  context: Optional[Dict[str, Any]]) -> KeywordConfidenceAnalysis:
        """Analyze keyword-based confidence with enhanced metrics."""
        
        if not keyword_prediction:
            # Create minimal analysis for missing keyword prediction
            return KeywordConfidenceAnalysis(
                pattern_match_confidence=0.0,
                keyword_density_confidence=0.0,
                biomedical_entity_confidence=0.0,
                domain_specificity_confidence=0.0,
                total_biomedical_signals=0,
                strong_signals=0,
                weak_signals=0,
                conflicting_signals=0,
                semantic_coherence_score=0.0,
                domain_alignment_score=0.0,
                query_completeness_score=0.0
            )
        
        query_lower = query_text.lower()
        words = query_lower.split()
        word_count = len(words)
        
        # Analyze pattern matches
        evidence = keyword_prediction.evidence or []
        pattern_matches = len([e for e in evidence if e.startswith('pattern:')])
        keyword_matches = len([e for e in evidence if e.startswith('keyword:')])
        
        pattern_match_confidence = min(pattern_matches / 3.0, 1.0)  # Normalize to max 3 patterns
        keyword_density = min(keyword_matches / max(word_count, 1), 1.0)
        
        # Analyze biomedical entities
        biomedical_terms = [
            'metabolomics', 'proteomics', 'genomics', 'biomarker', 'metabolite',
            'pathway', 'lc-ms', 'gc-ms', 'nmr', 'spectroscopy', 'clinical'
        ]
        biomedical_count = sum(1 for term in biomedical_terms if term in query_lower)
        biomedical_entity_confidence = min(biomedical_count / 3.0, 1.0)
        
        # Domain specificity analysis
        domain_terms = [
            'analysis', 'identification', 'discovery', 'diagnosis', 'treatment',
            'research', 'study', 'investigation', 'assessment', 'evaluation'
        ]
        domain_count = sum(1 for term in domain_terms if term in query_lower)
        domain_specificity_confidence = min(domain_count / 2.0, 1.0)
        
        # Signal strength analysis
        strong_signals = biomedical_count + pattern_matches
        weak_signals = keyword_matches - strong_signals
        weak_signals = max(0, weak_signals)
        
        # Conflicting signals (simplified)
        temporal_terms = ['latest', 'recent', 'current', '2024', '2025']
        established_terms = ['established', 'known', 'traditional', 'mechanism']
        
        has_temporal = any(term in query_lower for term in temporal_terms)
        has_established = any(term in query_lower for term in established_terms)
        conflicting_signals = 1 if (has_temporal and has_established) else 0
        
        # Semantic coherence (simplified)
        semantic_coherence_score = keyword_prediction.confidence * 0.8  # Use prediction confidence as proxy
        
        # Domain alignment
        domain_alignment_score = min(biomedical_entity_confidence + domain_specificity_confidence, 1.0)
        
        # Query completeness
        has_action = any(action in query_lower for action in ['analyze', 'identify', 'find', 'determine'])
        has_object = any(obj in query_lower for obj in ['metabolite', 'biomarker', 'pathway', 'compound'])
        query_completeness_score = (0.3 if word_count > 3 else 0.0) + \
                                  (0.4 if has_action else 0.0) + \
                                  (0.3 if has_object else 0.0)
        
        return KeywordConfidenceAnalysis(
            pattern_match_confidence=pattern_match_confidence,
            keyword_density_confidence=keyword_density,
            biomedical_entity_confidence=biomedical_entity_confidence,
            domain_specificity_confidence=domain_specificity_confidence,
            total_biomedical_signals=strong_signals + weak_signals,
            strong_signals=strong_signals,
            weak_signals=weak_signals,
            conflicting_signals=conflicting_signals,
            semantic_coherence_score=semantic_coherence_score,
            domain_alignment_score=domain_alignment_score,
            query_completeness_score=query_completeness_score
        )
    
    def _calculate_adaptive_weights(self, 
                                  query_text: str,
                                  llm_analysis: Optional[LLMConfidenceAnalysis],
                                  keyword_analysis: KeywordConfidenceAnalysis) -> Tuple[float, float]:
        """Calculate adaptive weights for LLM vs keyword confidence."""
        
        # Start with base weights
        llm_weight = self.weighting_params['base_llm_weight']
        keyword_weight = self.weighting_params['base_keyword_weight']
        
        # Adjust based on query length
        word_count = len(query_text.split())
        if word_count <= 3:
            # Short queries - favor keywords
            llm_weight -= self.weighting_params['query_length_factor']
            keyword_weight += self.weighting_params['query_length_factor']
        elif word_count >= 15:
            # Long queries - favor LLM
            llm_weight += self.weighting_params['query_length_factor']
            keyword_weight -= self.weighting_params['query_length_factor']
        
        # Adjust based on domain specificity
        if keyword_analysis.domain_alignment_score > 0.7:
            # High domain alignment - favor keywords
            keyword_weight += self.weighting_params['domain_specificity_factor']
            llm_weight -= self.weighting_params['domain_specificity_factor']
        
        # Adjust based on LLM consistency (if available)
        if llm_analysis and llm_analysis.consistency_score < 0.8:
            # Low consistency - reduce LLM weight
            llm_weight -= self.weighting_params['consistency_factor']
            keyword_weight += self.weighting_params['consistency_factor']
        
        # Adjust based on conflicting signals
        if keyword_analysis.conflicting_signals > 0:
            # Conflicts - increase LLM weight (better at handling ambiguity)
            llm_weight += 0.1
            keyword_weight -= 0.1
        
        # Normalize weights
        total_weight = llm_weight + keyword_weight
        if total_weight > 0:
            llm_weight /= total_weight
            keyword_weight /= total_weight
        else:
            llm_weight, keyword_weight = 0.5, 0.5
        
        return llm_weight, keyword_weight
    
    def _combine_confidences(self, 
                           llm_analysis: Optional[LLMConfidenceAnalysis],
                           keyword_analysis: KeywordConfidenceAnalysis,
                           llm_weight: float,
                           keyword_weight: float) -> float:
        """Combine LLM and keyword confidences with adaptive weighting."""
        
        # Get LLM confidence
        llm_confidence = 0.5  # Default if no LLM analysis
        if llm_analysis:
            llm_confidence = llm_analysis.calibrated_confidence
        
        # Calculate keyword confidence as weighted average of components
        keyword_confidence = (
            keyword_analysis.pattern_match_confidence * 0.3 +
            keyword_analysis.keyword_density_confidence * 0.2 +
            keyword_analysis.biomedical_entity_confidence * 0.2 +
            keyword_analysis.domain_specificity_confidence * 0.1 +
            keyword_analysis.semantic_coherence_score * 0.2
        )
        
        # Weighted combination
        combined = llm_confidence * llm_weight + keyword_confidence * keyword_weight
        
        # Apply penalties for poor signal quality
        if keyword_analysis.conflicting_signals > 0:
            combined *= 0.9  # 10% penalty for conflicts
        
        if keyword_analysis.total_biomedical_signals == 0:
            combined *= 0.85  # 15% penalty for no biomedical signals
        
        return max(0.0, min(1.0, combined))
    
    def _apply_calibration(self, 
                         combined_confidence: float,
                         llm_analysis: Optional[LLMConfidenceAnalysis],
                         keyword_analysis: KeywordConfidenceAnalysis) -> Tuple[float, float]:
        """Apply calibration to combined confidence."""
        
        # Determine primary source for calibration
        if llm_analysis and llm_analysis.raw_confidence > 0:
            calibrated = self.calibrator.calibrate_confidence(
                combined_confidence, ConfidenceSource.LLM_SEMANTIC
            )
        else:
            calibrated = self.calibrator.calibrate_confidence(
                combined_confidence, ConfidenceSource.KEYWORD_BASED
            )
        
        calibration_adjustment = calibrated - combined_confidence
        
        return calibrated, calibration_adjustment
    
    def _calculate_evidence_strength(self, 
                                   llm_analysis: Optional[LLMConfidenceAnalysis],
                                   keyword_analysis: KeywordConfidenceAnalysis) -> float:
        """Calculate overall evidence strength."""
        
        evidence_factors = []
        
        # LLM evidence strength
        if llm_analysis:
            llm_evidence = (
                llm_analysis.reasoning_quality_score * 0.4 +
                llm_analysis.consistency_score * 0.3 +
                (1.0 - len(llm_analysis.uncertainty_indicators) * 0.1) * 0.3
            )
            evidence_factors.append(llm_evidence)
        
        # Keyword evidence strength
        keyword_evidence = (
            min(keyword_analysis.strong_signals / 3.0, 1.0) * 0.4 +
            keyword_analysis.domain_alignment_score * 0.3 +
            keyword_analysis.semantic_coherence_score * 0.3
        )
        evidence_factors.append(keyword_evidence)
        
        # Penalty for conflicts
        if keyword_analysis.conflicting_signals > 0:
            conflict_penalty = keyword_analysis.conflicting_signals * 0.2
            evidence_factors = [max(0.0, ef - conflict_penalty) for ef in evidence_factors]
        
        return max(0.1, statistics.mean(evidence_factors))
    
    def _calculate_uncertainty_metrics(self, 
                                     llm_analysis: Optional[LLMConfidenceAnalysis],
                                     keyword_analysis: KeywordConfidenceAnalysis,
                                     evidence_strength: float) -> Dict[str, float]:
        """Calculate epistemic and aleatoric uncertainty."""
        
        # Epistemic uncertainty (model uncertainty - what we don't know)
        epistemic_factors = []
        
        if llm_analysis:
            # LLM model uncertainty
            llm_uncertainty = (
                (1.0 - llm_analysis.consistency_score) * 0.4 +
                len(llm_analysis.uncertainty_indicators) * 0.1 +
                (1.0 - llm_analysis.reasoning_quality_score) * 0.3
            )
            epistemic_factors.append(llm_uncertainty)
        
        # Keyword model uncertainty
        keyword_uncertainty = (
            (1.0 - keyword_analysis.semantic_coherence_score) * 0.3 +
            (1.0 - keyword_analysis.domain_alignment_score) * 0.3 +
            keyword_analysis.conflicting_signals * 0.2
        )
        epistemic_factors.append(keyword_uncertainty)
        
        epistemic_uncertainty = min(1.0, statistics.mean(epistemic_factors))
        
        # Aleatoric uncertainty (data uncertainty - inherent noise)
        aleatoric_uncertainty = max(0.1, 1.0 - evidence_strength)
        
        # Total uncertainty
        total_uncertainty = min(1.0, epistemic_uncertainty + aleatoric_uncertainty * 0.5)
        
        return {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'total': total_uncertainty
        }
    
    def _calculate_confidence_reliability(self, 
                                        llm_analysis: Optional[LLMConfidenceAnalysis],
                                        keyword_analysis: KeywordConfidenceAnalysis,
                                        evidence_strength: float) -> float:
        """Calculate how reliable this confidence estimate is."""
        
        reliability_factors = []
        
        # Evidence strength factor
        reliability_factors.append(evidence_strength)
        
        # LLM reliability factors
        if llm_analysis:
            llm_reliability = (
                llm_analysis.consistency_score * 0.4 +
                llm_analysis.reasoning_quality_score * 0.3 +
                min(1.0, llm_analysis.response_length / 100) * 0.3  # Longer responses more reliable
            )
            reliability_factors.append(llm_reliability)
        
        # Keyword reliability factors
        keyword_reliability = (
            min(keyword_analysis.strong_signals / 2.0, 1.0) * 0.4 +
            keyword_analysis.semantic_coherence_score * 0.3 +
            (1.0 if keyword_analysis.conflicting_signals == 0 else 0.5) * 0.3
        )
        reliability_factors.append(keyword_reliability)
        
        return max(0.1, statistics.mean(reliability_factors))
    
    def _generate_alternative_confidences(self, 
                                        llm_analysis: Optional[LLMConfidenceAnalysis],
                                        keyword_analysis: KeywordConfidenceAnalysis) -> List[Tuple[str, float]]:
        """Generate alternative confidence scenarios."""
        
        alternatives = []
        
        # LLM-only confidence
        if llm_analysis:
            alternatives.append(("llm_only", llm_analysis.calibrated_confidence))
        
        # Keyword-only confidence
        keyword_conf = (
            keyword_analysis.pattern_match_confidence * 0.3 +
            keyword_analysis.biomedical_entity_confidence * 0.4 +
            keyword_analysis.semantic_coherence_score * 0.3
        )
        alternatives.append(("keyword_only", keyword_conf))
        
        # Conservative estimate
        if alternatives:
            min_conf = min(alt[1] for alt in alternatives)
            alternatives.append(("conservative", min_conf * 0.8))
        
        # Optimistic estimate
        if alternatives:
            max_conf = max(alt[1] for alt in alternatives)
            alternatives.append(("optimistic", min(max_conf * 1.2, 1.0)))
        
        return alternatives
    
    def _create_default_llm_analysis(self) -> LLMConfidenceAnalysis:
        """Create default LLM analysis when LLM is not available."""
        return LLMConfidenceAnalysis(
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_quality_score=0.0,
            consistency_score=1.0,
            response_length=0,
            reasoning_depth=1,
            uncertainty_indicators=[],
            confidence_expressions=[]
        )
    
    def _create_fallback_confidence_result(self, query_text: str, start_time: float) -> HybridConfidenceResult:
        """Create fallback confidence result when main calculation fails."""
        
        # Simple fallback based on query characteristics
        word_count = len(query_text.split())
        
        if word_count <= 2:
            fallback_confidence = 0.3  # Very short queries are uncertain
        elif word_count <= 5:
            fallback_confidence = 0.5  # Short queries
        else:
            fallback_confidence = 0.6  # Longer queries
        
        # Simple biomedical check
        biomedical_terms = ['metabolomics', 'biomarker', 'pathway', 'lc-ms', 'clinical']
        if any(term in query_text.lower() for term in biomedical_terms):
            fallback_confidence += 0.1
        
        fallback_confidence = min(1.0, fallback_confidence)
        
        return HybridConfidenceResult(
            overall_confidence=fallback_confidence,
            confidence_interval=(fallback_confidence - 0.2, fallback_confidence + 0.2),
            llm_confidence=self._create_default_llm_analysis(),
            keyword_confidence=KeywordConfidenceAnalysis(
                pattern_match_confidence=fallback_confidence,
                keyword_density_confidence=fallback_confidence,
                biomedical_entity_confidence=fallback_confidence,
                domain_specificity_confidence=fallback_confidence,
                total_biomedical_signals=0,
                strong_signals=0,
                weak_signals=0,
                conflicting_signals=0,
                semantic_coherence_score=fallback_confidence,
                domain_alignment_score=fallback_confidence,
                query_completeness_score=fallback_confidence
            ),
            llm_weight=0.5,
            keyword_weight=0.5,
            calibration_adjustment=0.0,
            epistemic_uncertainty=0.5,
            aleatoric_uncertainty=0.3,
            total_uncertainty=0.6,
            confidence_reliability=0.3,
            evidence_strength=0.4,
            alternative_confidences=[("fallback", fallback_confidence)],
            calculation_time_ms=(time.time() - start_time) * 1000,
            calibration_version="fallback"
        )
    
    def record_prediction_feedback(self, 
                                 query_text: str,
                                 predicted_confidence: float,
                                 actual_accuracy: bool,
                                 confidence_source: ConfidenceSource = ConfidenceSource.ENSEMBLE_VOTING):
        """Record feedback for confidence calibration."""
        
        self.calibrator.record_prediction_outcome(
            predicted_confidence, actual_accuracy, confidence_source, query_text
        )
        
        self.logger.debug(f"Recorded feedback: conf={predicted_confidence:.3f}, "
                         f"accurate={actual_accuracy}, query='{query_text[:50]}...'")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about confidence scoring performance."""
        
        stats = {
            'scoring_performance': {
                'total_scorings': len(self.confidence_predictions),
                'average_scoring_time_ms': statistics.mean(self.scoring_times) if self.scoring_times else 0.0,
                'max_scoring_time_ms': max(self.scoring_times) if self.scoring_times else 0.0,
                'min_scoring_time_ms': min(self.scoring_times) if self.scoring_times else 0.0
            },
            'confidence_distribution': {},
            'calibration_stats': self.calibrator.get_calibration_stats(),
            'weighting_parameters': self.weighting_params.copy()
        }
        
        # Calculate confidence distribution
        if self.confidence_predictions:
            confidences = [pred['confidence'] for pred in self.confidence_predictions]
            stats['confidence_distribution'] = {
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'std_dev': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                'min': min(confidences),
                'max': max(confidences)
            }
        
        return stats


# ============================================================================
# CONFIDENCE VALIDATOR
# ============================================================================

class ConfidenceValidator:
    """
    Validation and accuracy measurement framework for confidence predictions.
    """
    
    def __init__(self, 
                 hybrid_scorer: HybridConfidenceScorer,
                 logger: Optional[logging.Logger] = None):
        self.hybrid_scorer = hybrid_scorer
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation metrics
        self.validation_results = deque(maxlen=1000)
        self.accuracy_by_confidence_bin = defaultdict(list)
        
    def validate_confidence_accuracy(self, 
                                   query_text: str,
                                   predicted_confidence: float,
                                   actual_routing_accuracy: bool,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate confidence prediction accuracy and provide detailed analysis.
        
        Args:
            query_text: Original query text
            predicted_confidence: The confidence that was predicted
            actual_routing_accuracy: Whether the routing was actually correct
            context: Additional context information
            
        Returns:
            Dict with validation results and recommendations
        """
        
        start_time = time.time()
        
        # Record the validation
        validation_record = {
            'query_text': query_text,
            'predicted_confidence': predicted_confidence,
            'actual_accuracy': actual_routing_accuracy,
            'timestamp': datetime.now(),
            'query_length': len(query_text.split())
        }
        
        self.validation_results.append(validation_record)
        
        # Update confidence bin accuracy
        confidence_bin = self._get_confidence_bin(predicted_confidence)
        self.accuracy_by_confidence_bin[confidence_bin].append(actual_routing_accuracy)
        
        # Record feedback for calibration
        self.hybrid_scorer.record_prediction_feedback(
            query_text, predicted_confidence, actual_routing_accuracy
        )
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics()
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(validation_metrics)
        
        validation_time = (time.time() - start_time) * 1000
        
        result = {
            'validation_record': validation_record,
            'validation_metrics': validation_metrics,
            'recommendations': recommendations,
            'validation_time_ms': validation_time,
            'calibration_status': self._assess_calibration_status()
        }
        
        self.logger.debug(f"Confidence validation completed in {validation_time:.2f}ms")
        
        return result
    
    def _get_confidence_bin(self, confidence: float) -> str:
        """Get confidence bin for validation tracking."""
        bin_size = 0.1
        bin_index = int(confidence / bin_size)
        bin_start = bin_index * bin_size
        return f"{bin_start:.1f}-{bin_start + bin_size:.1f}"
    
    def _calculate_validation_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive validation metrics."""
        
        if not self.validation_results:
            return {'total_validations': 0}
        
        # Basic metrics
        total_validations = len(self.validation_results)
        accuracies = [v['actual_accuracy'] for v in self.validation_results]
        confidences = [v['predicted_confidence'] for v in self.validation_results]
        
        overall_accuracy = statistics.mean([1.0 if acc else 0.0 for acc in accuracies])
        
        # Calibration metrics
        calibration_error = self._calculate_calibration_error()
        
        # Confidence bin accuracies
        bin_accuracies = {}
        for bin_name, bin_results in self.accuracy_by_confidence_bin.items():
            if bin_results:
                bin_accuracy = statistics.mean([1.0 if acc else 0.0 for acc in bin_results])
                bin_accuracies[bin_name] = {
                    'accuracy': bin_accuracy,
                    'count': len(bin_results),
                    'expected_confidence': float(bin_name.split('-')[0]) + 0.05  # Mid-bin
                }
        
        # Recent performance (last 100 validations)
        recent_results = list(self.validation_results)[-100:]
        recent_accuracy = statistics.mean([1.0 if v['actual_accuracy'] else 0.0 for v in recent_results])
        
        return {
            'total_validations': total_validations,
            'overall_accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'calibration_error': calibration_error,
            'confidence_bin_accuracies': bin_accuracies,
            'confidence_stats': {
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'std_dev': statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            }
        }
    
    def _calculate_calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        
        if not self.accuracy_by_confidence_bin:
            return 0.0
        
        total_samples = sum(len(results) for results in self.accuracy_by_confidence_bin.values())
        if total_samples == 0:
            return 0.0
        
        weighted_error = 0.0
        
        for bin_name, bin_results in self.accuracy_by_confidence_bin.items():
            if not bin_results:
                continue
            
            bin_confidence = float(bin_name.split('-')[0]) + 0.05  # Mid-bin value
            bin_accuracy = statistics.mean([1.0 if acc else 0.0 for acc in bin_results])
            bin_weight = len(bin_results) / total_samples
            
            bin_error = abs(bin_confidence - bin_accuracy)
            weighted_error += bin_weight * bin_error
        
        return weighted_error
    
    def _generate_validation_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on validation metrics."""
        
        recommendations = []
        
        # Check calibration
        calibration_error = metrics.get('calibration_error', 0.0)
        if calibration_error > 0.1:
            recommendations.append({
                'type': 'calibration',
                'priority': 'high',
                'issue': f'High calibration error ({calibration_error:.3f})',
                'recommendation': 'Increase calibration data collection and consider adjusting calibration parameters'
            })
        
        # Check overall accuracy
        overall_accuracy = metrics.get('overall_accuracy', 0.0)
        if overall_accuracy < 0.7:
            recommendations.append({
                'type': 'accuracy',
                'priority': 'high',
                'issue': f'Low overall accuracy ({overall_accuracy:.1%})',
                'recommendation': 'Review confidence calculation weights and consider model improvements'
            })
        
        # Check recent performance trend
        recent_accuracy = metrics.get('recent_accuracy', 0.0)
        if recent_accuracy < overall_accuracy - 0.1:
            recommendations.append({
                'type': 'performance_trend',
                'priority': 'medium',
                'issue': 'Recent performance decline detected',
                'recommendation': 'Monitor for system degradation and consider recalibration'
            })
        
        # Check confidence distribution
        conf_stats = metrics.get('confidence_stats', {})
        if conf_stats.get('std_dev', 0) < 0.1:
            recommendations.append({
                'type': 'confidence_range',
                'priority': 'medium',
                'issue': 'Narrow confidence range detected',
                'recommendation': 'Consider adjusting confidence calculation to better differentiate query difficulty'
            })
        
        return recommendations
    
    def _assess_calibration_status(self) -> Dict[str, Any]:
        """Assess current calibration status."""
        
        calibration_stats = self.hybrid_scorer.calibrator.get_calibration_stats()
        
        # Assess calibration quality
        brier_score = calibration_stats.get('brier_score', 0.5)
        total_predictions = calibration_stats.get('total_predictions', 0)
        
        if total_predictions < 50:
            status = 'insufficient_data'
            quality = 'unknown'
        elif brier_score <= 0.1:
            status = 'well_calibrated'
            quality = 'excellent'
        elif brier_score <= 0.2:
            status = 'adequately_calibrated'
            quality = 'good'
        elif brier_score <= 0.3:
            status = 'poorly_calibrated'
            quality = 'fair'
        else:
            status = 'very_poorly_calibrated'
            quality = 'poor'
        
        return {
            'status': status,
            'quality': quality,
            'brier_score': brier_score,
            'calibration_data_points': total_predictions,
            'needs_recalibration': brier_score > 0.25 or total_predictions > 500
        }
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        metrics = self._calculate_validation_metrics()
        recommendations = self._generate_validation_recommendations(metrics)
        calibration_status = self._assess_calibration_status()
        
        # Calculate confidence reliability by bin
        reliability_by_bin = {}
        for bin_name, bin_data in metrics.get('confidence_bin_accuracies', {}).items():
            expected_conf = bin_data['expected_confidence']
            actual_acc = bin_data['accuracy']
            reliability = 1.0 - abs(expected_conf - actual_acc)
            reliability_by_bin[bin_name] = reliability
        
        overall_reliability = statistics.mean(reliability_by_bin.values()) if reliability_by_bin else 0.0
        
        return {
            'validation_summary': {
                'total_validations': metrics.get('total_validations', 0),
                'overall_accuracy': metrics.get('overall_accuracy', 0.0),
                'calibration_error': metrics.get('calibration_error', 0.0),
                'overall_reliability': overall_reliability,
                'validation_period_days': self._get_validation_period_days()
            },
            'detailed_metrics': metrics,
            'calibration_status': calibration_status,
            'recommendations': recommendations,
            'confidence_reliability_by_bin': reliability_by_bin,
            'system_health': self._assess_system_health(metrics, calibration_status)
        }
    
    def _get_validation_period_days(self) -> float:
        """Calculate validation period in days."""
        if not self.validation_results:
            return 0.0
        
        oldest = min(v['timestamp'] for v in self.validation_results)
        newest = max(v['timestamp'] for v in self.validation_results)
        
        return (newest - oldest).total_seconds() / 86400
    
    def _assess_system_health(self, metrics: Dict[str, Any], calibration_status: Dict[str, Any]) -> str:
        """Assess overall confidence system health."""
        
        health_factors = []
        
        # Accuracy factor
        accuracy = metrics.get('overall_accuracy', 0.0)
        health_factors.append(accuracy)
        
        # Calibration factor
        calibration_error = metrics.get('calibration_error', 0.0)
        calibration_health = max(0.0, 1.0 - calibration_error * 2)  # Scale calibration error
        health_factors.append(calibration_health)
        
        # Data sufficiency factor
        total_validations = metrics.get('total_validations', 0)
        data_health = min(1.0, total_validations / 100)  # Scale to 100 validations
        health_factors.append(data_health)
        
        # Calculate overall health
        overall_health = statistics.mean(health_factors)
        
        if overall_health >= 0.9:
            return 'excellent'
        elif overall_health >= 0.8:
            return 'good'
        elif overall_health >= 0.7:
            return 'fair'
        elif overall_health >= 0.6:
            return 'poor'
        else:
            return 'critical'


# ============================================================================
# INTEGRATION HELPER FUNCTIONS  
# ============================================================================

def create_hybrid_confidence_scorer(
    biomedical_router: Optional[BiomedicalQueryRouter] = None,
    llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
    calibration_data_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> HybridConfidenceScorer:
    """
    Factory function to create a hybrid confidence scorer with proper initialization.
    
    Args:
        biomedical_router: Existing biomedical router for keyword analysis
        llm_classifier: Enhanced LLM classifier for semantic analysis  
        calibration_data_path: Path to store calibration data
        logger: Logger instance
        
    Returns:
        Configured HybridConfidenceScorer instance
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create router if not provided
    if biomedical_router is None:
        try:
            biomedical_router = BiomedicalQueryRouter(logger)
            logger.info("Created biomedical router for hybrid confidence scoring")
        except Exception as e:
            logger.warning(f"Could not create biomedical router: {e}")
    
    # Set default calibration path
    if calibration_data_path is None:
        calibration_data_path = "/tmp/confidence_calibration.json"
    
    scorer = HybridConfidenceScorer(
        biomedical_router=biomedical_router,
        llm_classifier=llm_classifier,
        calibration_data_path=calibration_data_path,
        logger=logger
    )
    
    logger.info("Hybrid confidence scorer created successfully")
    return scorer


def integrate_with_existing_confidence_metrics(
    hybrid_result: HybridConfidenceResult,
    query_text: str
) -> ConfidenceMetrics:
    """
    Convert HybridConfidenceResult to existing ConfidenceMetrics format for backward compatibility.
    
    Args:
        hybrid_result: Result from hybrid confidence scoring
        query_text: Original query text
        
    Returns:
        ConfidenceMetrics compatible with existing infrastructure
    """
    
    # Create alternative interpretations from hybrid result
    alternative_interpretations = []
    for alt_name, alt_conf in hybrid_result.alternative_confidences:
        # Map alternative confidence types to routing decisions
        routing_mapping = {
            'llm_only': RoutingDecision.LIGHTRAG,
            'keyword_only': RoutingDecision.PERPLEXITY,
            'conservative': RoutingDecision.EITHER,
            'optimistic': RoutingDecision.HYBRID,
            'fallback': RoutingDecision.EITHER
        }
        routing_decision = routing_mapping.get(alt_name, RoutingDecision.EITHER)
        alternative_interpretations.append((routing_decision, alt_conf))
    
    return ConfidenceMetrics(
        overall_confidence=hybrid_result.overall_confidence,
        research_category_confidence=hybrid_result.llm_confidence.calibrated_confidence,
        temporal_analysis_confidence=hybrid_result.keyword_confidence.domain_alignment_score,
        signal_strength_confidence=hybrid_result.evidence_strength,
        context_coherence_confidence=hybrid_result.keyword_confidence.semantic_coherence_score,
        keyword_density=hybrid_result.keyword_confidence.keyword_density_confidence,
        pattern_match_strength=hybrid_result.keyword_confidence.pattern_match_confidence,
        biomedical_entity_count=hybrid_result.keyword_confidence.total_biomedical_signals,
        ambiguity_score=hybrid_result.total_uncertainty,
        conflict_score=hybrid_result.keyword_confidence.conflicting_signals * 0.5,
        alternative_interpretations=alternative_interpretations,
        calculation_time_ms=hybrid_result.calculation_time_ms
    )


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    async def demo_confidence_scoring():
        """Demonstrate comprehensive confidence scoring."""
        
        print("=== Comprehensive Confidence Scoring Demo ===")
        
        # Create hybrid scorer
        scorer = create_hybrid_confidence_scorer(logger=logger)
        
        # Test queries
        test_queries = [
            "What is the relationship between glucose metabolism and insulin signaling pathways?",
            "Latest research on metabolomics biomarkers for diabetes 2025",
            "LC-MS analysis methods for metabolite identification",
            "metabolomics"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            try:
                # Calculate comprehensive confidence
                result = await scorer.calculate_comprehensive_confidence(query)
                
                print(f"Overall Confidence: {result.overall_confidence:.3f}")
                print(f"Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
                print(f"LLM Weight: {result.llm_weight:.3f}, Keyword Weight: {result.keyword_weight:.3f}")
                print(f"Evidence Strength: {result.evidence_strength:.3f}")
                print(f"Total Uncertainty: {result.total_uncertainty:.3f}")
                print(f"Confidence Reliability: {result.confidence_reliability:.3f}")
                print(f"Calculation Time: {result.calculation_time_ms:.2f}ms")
                
                # Convert to legacy format
                legacy_metrics = integrate_with_existing_confidence_metrics(result, query)
                print(f"Legacy Overall Confidence: {legacy_metrics.overall_confidence:.3f}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        
        # Show system statistics
        print("\n--- System Statistics ---")
        stats = scorer.get_comprehensive_stats()
        print(f"Total Scorings: {stats['scoring_performance']['total_scorings']}")
        print(f"Average Scoring Time: {stats['scoring_performance']['average_scoring_time_ms']:.2f}ms")
        print(f"Calibration Data Points: {stats['calibration_stats']['total_predictions']}")
        
    
    # Run demo
    print("Running comprehensive confidence scoring demo...")
    try:
        asyncio.run(demo_confidence_scoring())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")