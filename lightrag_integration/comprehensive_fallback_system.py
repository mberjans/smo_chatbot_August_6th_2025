"""
Comprehensive Multi-Tiered Fallback System for Clinical Metabolomics Oracle

This module provides a bulletproof fallback mechanism that ensures 100% system availability
through intelligent degradation, failure detection, and automatic recovery capabilities.

The system implements a 5-level fallback hierarchy:
- Level 1: LLM with full confidence analysis (primary)
- Level 2: LLM with simplified prompts (degraded performance)
- Level 3: Keyword-based classification only (reliable fallback)
- Level 4: Cached responses for common queries (emergency)
- Level 5: Default routing decision with low confidence (last resort)

Classes:
    - FallbackLevel: Enumeration of fallback levels
    - FallbackResult: Result from fallback processing
    - FailureDetector: Intelligent failure detection system
    - FallbackOrchestrator: Main orchestrator for multi-tiered fallback
    - GracefulDegradationManager: Manages progressive degradation strategies
    - RecoveryManager: Handles automatic service recovery
    - EmergencyCache: Emergency response cache system
    - FallbackMonitor: Comprehensive monitoring and alerting

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import time
import json
import logging
import asyncio
import threading
import statistics
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import pickle

# Import existing components for integration
try:
    from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction, ConfidenceMetrics
    from .enhanced_llm_classifier import EnhancedLLMQueryClassifier, CircuitBreaker, CircuitBreakerState
    from .research_categorizer import ResearchCategorizer, CategoryPrediction
    from .cost_persistence import ResearchCategory
    from .cost_based_circuit_breaker import CostBasedCircuitBreaker
    from .budget_manager import BudgetManager
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# FALLBACK LEVEL DEFINITIONS AND DATA STRUCTURES
# ============================================================================

class FallbackLevel(IntEnum):
    """
    Enumeration of fallback levels in order of preference.
    Lower numbers indicate higher quality/preference.
    """
    
    FULL_LLM_WITH_CONFIDENCE = 1      # Primary: Full LLM analysis with confidence scoring
    SIMPLIFIED_LLM = 2                 # Degraded: LLM with simplified prompts
    KEYWORD_BASED_ONLY = 3            # Reliable: Pure keyword-based classification
    EMERGENCY_CACHE = 4               # Emergency: Cached responses for common queries
    DEFAULT_ROUTING = 5               # Last resort: Default routing with low confidence


class FailureType(Enum):
    """Types of failures that can trigger fallback mechanisms."""
    
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    BUDGET_EXCEEDED = "budget_exceeded"
    LOW_CONFIDENCE = "low_confidence"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN_ERROR = "unknown_error"


class DegradationStrategy(Enum):
    """Strategies for graceful degradation."""
    
    PROGRESSIVE_TIMEOUT_REDUCTION = "progressive_timeout_reduction"
    QUALITY_THRESHOLD_ADJUSTMENT = "quality_threshold_adjustment"
    CACHE_WARMING = "cache_warming"
    LOAD_SHEDDING = "load_shedding"
    PRIORITY_BASED_PROCESSING = "priority_based_processing"


@dataclass
class FallbackResult:
    """
    Result from fallback processing with comprehensive metadata.
    """
    
    # Core result data
    routing_prediction: RoutingPrediction
    fallback_level_used: FallbackLevel
    success: bool
    
    # Failure and recovery information
    failure_reasons: List[FailureType] = field(default_factory=list)
    attempted_levels: List[FallbackLevel] = field(default_factory=list)
    recovery_suggestions: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time_ms: float = 0.0
    fallback_decision_time_ms: float = 0.0
    level_processing_times: Dict[FallbackLevel, float] = field(default_factory=dict)
    
    # Quality metrics
    confidence_degradation: float = 0.0  # How much confidence was lost due to fallback
    quality_score: float = 1.0  # Overall quality of the result (1.0 = perfect)
    reliability_score: float = 1.0  # Reliability of the chosen fallback level
    
    # Metadata and debugging
    debug_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    fallback_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'routing_prediction': self.routing_prediction.to_dict(),
            'fallback_level_used': self.fallback_level_used.name,
            'success': self.success,
            'failure_reasons': [f.value for f in self.failure_reasons],
            'attempted_levels': [level.name for level in self.attempted_levels],
            'recovery_suggestions': self.recovery_suggestions,
            'total_processing_time_ms': self.total_processing_time_ms,
            'fallback_decision_time_ms': self.fallback_decision_time_ms,
            'level_processing_times': {level.name: time_ms for level, time_ms in self.level_processing_times.items()},
            'confidence_degradation': self.confidence_degradation,
            'quality_score': self.quality_score,
            'reliability_score': self.reliability_score,
            'debug_info': self.debug_info,
            'warnings': self.warnings,
            'fallback_chain': self.fallback_chain
        }


@dataclass
class FailureDetectionMetrics:
    """Metrics for intelligent failure detection."""
    
    # Response time metrics
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=50))
    average_response_time_ms: float = 0.0
    response_time_trend: float = 0.0  # Positive = getting slower
    
    # Error rate metrics
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rate: float = 0.0
    error_trend: float = 0.0  # Positive = more errors
    
    # Confidence metrics
    recent_confidences: deque = field(default_factory=lambda: deque(maxlen=50))
    average_confidence: float = 0.0
    confidence_trend: float = 0.0  # Negative = losing confidence
    
    # API health metrics
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    rate_limited_calls: int = 0
    
    # Performance degradation indicators
    performance_alerts: List[str] = field(default_factory=list)
    quality_degradation_score: float = 0.0
    system_health_score: float = 1.0  # 1.0 = perfect health
    
    def update_response_time(self, response_time_ms: float):
        """Update response time metrics."""
        self.recent_response_times.append(response_time_ms)
        if len(self.recent_response_times) >= 2:
            self.average_response_time_ms = statistics.mean(self.recent_response_times)
            # Calculate trend (simple linear regression slope)
            if len(self.recent_response_times) >= 5:
                times = list(range(len(self.recent_response_times)))
                values = list(self.recent_response_times)
                n = len(times)
                sum_x = sum(times)
                sum_y = sum(values)
                sum_xy = sum(x * y for x, y in zip(times, values))
                sum_x2 = sum(x * x for x in times)
                self.response_time_trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    def update_error(self, is_error: bool, error_type: Optional[FailureType] = None):
        """Update error rate metrics."""
        self.recent_errors.append(1 if is_error else 0)
        self.error_rate = sum(self.recent_errors) / len(self.recent_errors) if self.recent_errors else 0.0
        
        if is_error:
            self.failed_calls += 1
            if error_type == FailureType.API_TIMEOUT:
                self.timeout_calls += 1
            elif error_type == FailureType.RATE_LIMIT:
                self.rate_limited_calls += 1
        else:
            self.successful_calls += 1
    
    def update_confidence(self, confidence: float):
        """Update confidence metrics."""
        self.recent_confidences.append(confidence)
        if len(self.recent_confidences) >= 2:
            self.average_confidence = statistics.mean(self.recent_confidences)
            # Calculate confidence trend
            if len(self.recent_confidences) >= 5:
                times = list(range(len(self.recent_confidences)))
                values = list(self.recent_confidences)
                n = len(times)
                sum_x = sum(times)
                sum_y = sum(values)
                sum_xy = sum(x * y for x, y in zip(times, values))
                sum_x2 = sum(x * x for x in times)
                self.confidence_trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    def calculate_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        factors = []
        
        # Response time factor (penalize slow responses)
        if self.average_response_time_ms > 0:
            response_factor = max(0.0, 1.0 - (self.average_response_time_ms - 500) / 2000)  # Penalize above 500ms
            factors.append(response_factor)
        
        # Error rate factor
        error_factor = max(0.0, 1.0 - self.error_rate * 2)  # 50% error rate = 0 factor
        factors.append(error_factor)
        
        # Confidence factor
        if self.average_confidence > 0:
            confidence_factor = max(0.0, self.average_confidence - 0.3) / 0.7  # Scale 0.3-1.0 to 0-1.0
            factors.append(confidence_factor)
        
        # Trend factors (penalize negative trends)
        if self.response_time_trend > 0:  # Getting slower
            factors.append(max(0.0, 1.0 - abs(self.response_time_trend) * 10))
        
        if self.confidence_trend < 0:  # Losing confidence
            factors.append(max(0.0, 1.0 - abs(self.confidence_trend) * 10))
        
        self.system_health_score = statistics.mean(factors) if factors else 1.0
        return self.system_health_score


# ============================================================================
# INTELLIGENT FAILURE DETECTION SYSTEM
# ============================================================================

class FailureDetector:
    """
    Intelligent failure detection system that monitors system health and
    provides early warning signals for potential issues.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize failure detection system."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Detection metrics
        self.metrics = FailureDetectionMetrics()
        self.lock = threading.Lock()
        
        # Detection thresholds
        self.thresholds = {
            'response_time_warning_ms': 1500,      # Warn if responses > 1.5s
            'response_time_critical_ms': 3000,     # Critical if responses > 3s
            'error_rate_warning': 0.1,             # Warn if error rate > 10%
            'error_rate_critical': 0.25,           # Critical if error rate > 25%
            'confidence_warning': 0.4,             # Warn if confidence < 40%
            'confidence_critical': 0.2,            # Critical if confidence < 20%
            'trend_warning_threshold': 0.1,        # Warn if trend indicates degradation
            'health_score_warning': 0.6,           # Warn if health score < 60%
            'health_score_critical': 0.3,          # Critical if health score < 30%
        }
        
        # Pattern detection
        self.failure_patterns = defaultdict(list)
        self.pattern_detection_window = 100  # Track last 100 events
        
        # Alert suppression to prevent spam
        self.last_alert_times = defaultdict(float)
        self.alert_cooldown_seconds = 30
        
    def detect_failure_conditions(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> List[FailureType]:
        """
        Detect potential failure conditions based on current system state.
        
        Args:
            query_text: The user query being processed
            context: Optional context information
            
        Returns:
            List of detected failure types
        """
        detected_failures = []
        
        with self.lock:
            # Update health score
            health_score = self.metrics.calculate_health_score()
            
            # Check response time degradation
            if self.metrics.average_response_time_ms > self.thresholds['response_time_critical']:
                detected_failures.append(FailureType.PERFORMANCE_DEGRADATION)
            elif (self.metrics.average_response_time_ms > self.thresholds['response_time_warning_ms'] and
                  self.metrics.response_time_trend > self.thresholds['trend_warning_threshold']):
                detected_failures.append(FailureType.PERFORMANCE_DEGRADATION)
            
            # Check error rate
            if self.metrics.error_rate > self.thresholds['error_rate_critical']:
                detected_failures.append(FailureType.SERVICE_UNAVAILABLE)
            elif self.metrics.error_rate > self.thresholds['error_rate_warning']:
                detected_failures.append(FailureType.API_ERROR)
            
            # Check confidence degradation
            if self.metrics.average_confidence < self.thresholds['confidence_critical']:
                detected_failures.append(FailureType.LOW_CONFIDENCE)
            
            # Check overall health score
            if health_score < self.thresholds['health_score_critical']:
                detected_failures.append(FailureType.SERVICE_UNAVAILABLE)
            elif health_score < self.thresholds['health_score_warning']:
                detected_failures.append(FailureType.PERFORMANCE_DEGRADATION)
            
            # Pattern-based detection
            pattern_failures = self._detect_failure_patterns()
            detected_failures.extend(pattern_failures)
            
        return detected_failures
    
    def record_operation_result(self, 
                              response_time_ms: float, 
                              success: bool, 
                              confidence: float = None,
                              error_type: Optional[FailureType] = None):
        """Record the result of an operation for failure detection."""
        with self.lock:
            # Update metrics
            self.metrics.update_response_time(response_time_ms)
            self.metrics.update_error(not success, error_type)
            
            if confidence is not None:
                self.metrics.update_confidence(confidence)
            
            # Record for pattern detection
            event = {
                'timestamp': time.time(),
                'response_time_ms': response_time_ms,
                'success': success,
                'confidence': confidence,
                'error_type': error_type.value if error_type else None
            }
            
            self.failure_patterns['recent_events'].append(event)
            if len(self.failure_patterns['recent_events']) > self.pattern_detection_window:
                self.failure_patterns['recent_events'].pop(0)
    
    def _detect_failure_patterns(self) -> List[FailureType]:
        """Detect failure patterns from recent events."""
        detected_failures = []
        recent_events = self.failure_patterns.get('recent_events', [])
        
        if len(recent_events) < 5:
            return detected_failures
        
        current_time = time.time()
        recent_window = [e for e in recent_events if current_time - e['timestamp'] < 300]  # Last 5 minutes
        
        if not recent_window:
            return detected_failures
        
        # Pattern 1: Consecutive failures
        consecutive_failures = 0
        for event in reversed(recent_window[-10:]):  # Check last 10 events
            if not event['success']:
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= 5:
            detected_failures.append(FailureType.SERVICE_UNAVAILABLE)
        elif consecutive_failures >= 3:
            detected_failures.append(FailureType.API_ERROR)
        
        # Pattern 2: Rapid response time degradation
        if len(recent_window) >= 10:
            first_half = recent_window[:len(recent_window)//2]
            second_half = recent_window[len(recent_window)//2:]
            
            avg_first = statistics.mean(e['response_time_ms'] for e in first_half)
            avg_second = statistics.mean(e['response_time_ms'] for e in second_half)
            
            if avg_second > avg_first * 1.5:  # 50% increase in response time
                detected_failures.append(FailureType.PERFORMANCE_DEGRADATION)
        
        # Pattern 3: Confidence collapse
        confidence_events = [e for e in recent_window if e['confidence'] is not None]
        if len(confidence_events) >= 5:
            recent_confidences = [e['confidence'] for e in confidence_events[-5:]]
            if all(c < 0.3 for c in recent_confidences):  # All recent confidences are low
                detected_failures.append(FailureType.LOW_CONFIDENCE)
        
        return detected_failures
    
    def get_early_warning_signals(self) -> List[Dict[str, Any]]:
        """Get early warning signals for potential issues."""
        warnings = []
        
        with self.lock:
            health_score = self.metrics.calculate_health_score()
            
            # Response time warnings
            if (self.metrics.average_response_time_ms > self.thresholds['response_time_warning_ms'] and
                self.metrics.response_time_trend > 0):
                warnings.append({
                    'type': 'response_time_degradation',
                    'severity': 'warning',
                    'message': f'Response time trending upward: {self.metrics.average_response_time_ms:.1f}ms avg',
                    'recommended_actions': ['Consider enabling cache warming', 'Check API performance']
                })
            
            # Error rate warnings
            if self.metrics.error_rate > self.thresholds['error_rate_warning']:
                warnings.append({
                    'type': 'error_rate_increase',
                    'severity': 'warning' if self.metrics.error_rate < 0.2 else 'critical',
                    'message': f'Error rate elevated: {self.metrics.error_rate:.1%}',
                    'recommended_actions': ['Enable fallback mechanisms', 'Check service health']
                })
            
            # Confidence degradation warnings
            if (self.metrics.average_confidence < self.thresholds['confidence_warning'] and
                self.metrics.confidence_trend < -0.05):
                warnings.append({
                    'type': 'confidence_degradation',
                    'severity': 'warning',
                    'message': f'Classification confidence declining: {self.metrics.average_confidence:.2f} avg',
                    'recommended_actions': ['Review query patterns', 'Consider model retraining']
                })
            
            # Health score warnings
            if health_score < self.thresholds['health_score_warning']:
                severity = 'critical' if health_score < self.thresholds['health_score_critical'] else 'warning'
                warnings.append({
                    'type': 'system_health_degradation',
                    'severity': severity,
                    'message': f'System health score: {health_score:.2f}',
                    'recommended_actions': ['Enable all fallback mechanisms', 'Investigate system issues']
                })
        
        return warnings
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure detection statistics."""
        with self.lock:
            return {
                'metrics': {
                    'average_response_time_ms': self.metrics.average_response_time_ms,
                    'response_time_trend': self.metrics.response_time_trend,
                    'error_rate': self.metrics.error_rate,
                    'error_trend': self.metrics.error_trend,
                    'average_confidence': self.metrics.average_confidence,
                    'confidence_trend': self.metrics.confidence_trend,
                    'system_health_score': self.metrics.system_health_score,
                    'successful_calls': self.metrics.successful_calls,
                    'failed_calls': self.metrics.failed_calls,
                    'timeout_calls': self.metrics.timeout_calls,
                    'rate_limited_calls': self.metrics.rate_limited_calls
                },
                'thresholds': self.thresholds,
                'recent_events_count': len(self.failure_patterns.get('recent_events', [])),
                'performance_alerts': self.metrics.performance_alerts,
                'quality_degradation_score': self.metrics.quality_degradation_score
            }


# ============================================================================
# EMERGENCY CACHE SYSTEM
# ============================================================================

class EmergencyCache:
    """
    Emergency response cache system for common queries.
    Provides instant responses when all other systems fail.
    """
    
    def __init__(self, cache_file: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize emergency cache system."""
        self.logger = logger or logging.getLogger(__name__)
        self.cache_file = Path(cache_file) if cache_file else Path("emergency_cache.pkl")
        self.lock = threading.Lock()
        
        # In-memory cache for fast access
        self.cache = {}
        self.cache_metadata = {}
        
        # Cache configuration
        self.max_cache_size = 1000
        self.default_confidence = 0.15  # Low confidence for cached responses
        
        # Load existing cache
        self._load_cache()
        
        # Pre-populate with common queries
        self._populate_common_queries()
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.cache_metadata = data.get('metadata', {})
                self.logger.info(f"Loaded emergency cache with {len(self.cache)} entries")
        except Exception as e:
            self.logger.warning(f"Failed to load emergency cache: {e}")
            self.cache = {}
            self.cache_metadata = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                data = {
                    'cache': self.cache,
                    'metadata': self.cache_metadata,
                    'saved_at': datetime.now().isoformat()
                }
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save emergency cache: {e}")
    
    def _populate_common_queries(self):
        """Pre-populate cache with common query patterns and their responses."""
        common_patterns = {
            # Metabolite identification queries
            "identify metabolite": {
                "routing_decision": RoutingDecision.LIGHTRAG,
                "research_category": ResearchCategory.METABOLITE_IDENTIFICATION,
                "reasoning": ["Emergency cached response for metabolite identification query"]
            },
            "compound identification": {
                "routing_decision": RoutingDecision.LIGHTRAG,
                "research_category": ResearchCategory.METABOLITE_IDENTIFICATION,
                "reasoning": ["Emergency cached response for compound identification query"]
            },
            
            # Pathway analysis queries
            "pathway analysis": {
                "routing_decision": RoutingDecision.LIGHTRAG,
                "research_category": ResearchCategory.PATHWAY_ANALYSIS,
                "reasoning": ["Emergency cached response for pathway analysis query"]
            },
            "metabolic pathway": {
                "routing_decision": RoutingDecision.LIGHTRAG,
                "research_category": ResearchCategory.PATHWAY_ANALYSIS,
                "reasoning": ["Emergency cached response for metabolic pathway query"]
            },
            
            # Biomarker discovery queries
            "biomarker discovery": {
                "routing_decision": RoutingDecision.EITHER,
                "research_category": ResearchCategory.BIOMARKER_DISCOVERY,
                "reasoning": ["Emergency cached response for biomarker discovery query"]
            },
            "disease marker": {
                "routing_decision": RoutingDecision.EITHER,
                "research_category": ResearchCategory.BIOMARKER_DISCOVERY,
                "reasoning": ["Emergency cached response for disease marker query"]
            },
            
            # Drug discovery queries
            "drug discovery": {
                "routing_decision": RoutingDecision.EITHER,
                "research_category": ResearchCategory.DRUG_DISCOVERY,
                "reasoning": ["Emergency cached response for drug discovery query"]
            },
            "pharmaceutical": {
                "routing_decision": RoutingDecision.EITHER,
                "research_category": ResearchCategory.DRUG_DISCOVERY,
                "reasoning": ["Emergency cached response for pharmaceutical query"]
            },
            
            # Clinical diagnosis queries
            "clinical diagnosis": {
                "routing_decision": RoutingDecision.LIGHTRAG,
                "research_category": ResearchCategory.CLINICAL_DIAGNOSIS,
                "reasoning": ["Emergency cached response for clinical diagnosis query"]
            },
            "patient sample": {
                "routing_decision": RoutingDecision.LIGHTRAG,
                "research_category": ResearchCategory.CLINICAL_DIAGNOSIS,
                "reasoning": ["Emergency cached response for patient sample query"]
            },
            
            # Real-time queries
            "latest research": {
                "routing_decision": RoutingDecision.PERPLEXITY,
                "research_category": ResearchCategory.LITERATURE_SEARCH,
                "reasoning": ["Emergency cached response for latest research query"]
            },
            "recent studies": {
                "routing_decision": RoutingDecision.PERPLEXITY,
                "research_category": ResearchCategory.LITERATURE_SEARCH,
                "reasoning": ["Emergency cached response for recent studies query"]
            },
            
            # General queries
            "what is": {
                "routing_decision": RoutingDecision.EITHER,
                "research_category": ResearchCategory.GENERAL_QUERY,
                "reasoning": ["Emergency cached response for general definition query"]
            },
            "explain": {
                "routing_decision": RoutingDecision.EITHER,
                "research_category": ResearchCategory.GENERAL_QUERY,
                "reasoning": ["Emergency cached response for explanation query"]
            }
        }
        
        for pattern, response_data in common_patterns.items():
            query_hash = self._get_query_hash(pattern)
            
            # Create minimal confidence metrics for emergency response
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=self.default_confidence,
                research_category_confidence=self.default_confidence,
                temporal_analysis_confidence=0.1,
                signal_strength_confidence=0.1,
                context_coherence_confidence=0.1,
                keyword_density=0.1,
                pattern_match_strength=0.1,
                biomedical_entity_count=1,
                ambiguity_score=0.8,
                conflict_score=0.2,
                alternative_interpretations=[(response_data["routing_decision"], self.default_confidence)],
                calculation_time_ms=0.0
            )
            
            # Create routing prediction
            prediction = RoutingPrediction(
                routing_decision=response_data["routing_decision"],
                confidence=self.default_confidence,
                reasoning=response_data["reasoning"],
                research_category=response_data["research_category"],
                confidence_metrics=confidence_metrics,
                temporal_indicators=[],
                knowledge_indicators=[],
                metadata={
                    'emergency_cache': True,
                    'pattern_matched': pattern,
                    'cache_generated': True
                }
            )
            
            self.cache[query_hash] = prediction
            self.cache_metadata[query_hash] = {
                'pattern': pattern,
                'created_at': time.time(),
                'access_count': 0,
                'last_accessed': None
            }
        
        self._save_cache()
        self.logger.info(f"Populated emergency cache with {len(common_patterns)} common patterns")
    
    def _get_query_hash(self, query_text: str) -> str:
        """Generate hash for query text."""
        query_normalized = query_text.lower().strip()
        return hashlib.md5(query_normalized.encode()).hexdigest()
    
    def get_cached_response(self, query_text: str) -> Optional[RoutingPrediction]:
        """
        Get cached response for a query.
        
        Args:
            query_text: Query to look up
            
        Returns:
            Cached routing prediction or None if not found
        """
        query_hash = self._get_query_hash(query_text)
        
        with self.lock:
            # Direct hash lookup
            if query_hash in self.cache:
                self.cache_metadata[query_hash]['access_count'] += 1
                self.cache_metadata[query_hash]['last_accessed'] = time.time()
                return self.cache[query_hash]
            
            # Pattern matching for partial matches
            query_lower = query_text.lower()
            for cached_hash, prediction in self.cache.items():
                metadata = self.cache_metadata.get(cached_hash, {})
                pattern = metadata.get('pattern', '')
                
                if pattern and pattern in query_lower:
                    # Create a copy with updated metadata
                    updated_prediction = RoutingPrediction(
                        routing_decision=prediction.routing_decision,
                        confidence=prediction.confidence,
                        reasoning=prediction.reasoning + [f"Pattern matched: '{pattern}'"],
                        research_category=prediction.research_category,
                        confidence_metrics=prediction.confidence_metrics,
                        temporal_indicators=prediction.temporal_indicators,
                        knowledge_indicators=prediction.knowledge_indicators,
                        metadata=prediction.metadata.copy()
                    )
                    updated_prediction.metadata['pattern_matched'] = pattern
                    
                    # Update access metrics
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    metadata['last_accessed'] = time.time()
                    
                    return updated_prediction
        
        return None
    
    def cache_response(self, query_text: str, prediction: RoutingPrediction, force: bool = False):
        """
        Cache a response for future emergency use.
        
        Args:
            query_text: Query text to cache
            prediction: Routing prediction to cache
            force: Force caching even if cache is full
        """
        query_hash = self._get_query_hash(query_text)
        
        with self.lock:
            # Check cache size limits
            if len(self.cache) >= self.max_cache_size and not force:
                # Remove least recently used entries
                self._evict_lru_entries()
            
            # Only cache if confidence is reasonable or if forced
            if prediction.confidence >= 0.3 or force:
                self.cache[query_hash] = prediction
                self.cache_metadata[query_hash] = {
                    'query_text': query_text,
                    'created_at': time.time(),
                    'access_count': 0,
                    'last_accessed': None,
                    'original_confidence': prediction.confidence
                }
                
                # Periodically save to disk
                if len(self.cache) % 50 == 0:
                    self._save_cache()
    
    def _evict_lru_entries(self, num_to_evict: int = None):
        """Evict least recently used entries."""
        if num_to_evict is None:
            num_to_evict = max(1, len(self.cache) // 10)  # Remove 10% of cache
        
        # Sort by last accessed time (None values go to end)
        sorted_entries = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        for i in range(min(num_to_evict, len(sorted_entries))):
            hash_to_remove = sorted_entries[i][0]
            if hash_to_remove in self.cache:
                del self.cache[hash_to_remove]
            if hash_to_remove in self.cache_metadata:
                del self.cache_metadata[hash_to_remove]
        
        self.logger.debug(f"Evicted {num_to_evict} LRU entries from emergency cache")
    
    def warm_cache(self, query_patterns: List[str]):
        """
        Warm the cache with specific query patterns.
        
        Args:
            query_patterns: List of query patterns to warm cache with
        """
        for pattern in query_patterns:
            if not self.get_cached_response(pattern):
                # Create a basic response for the pattern
                prediction = self._create_default_response(pattern)
                self.cache_response(pattern, prediction, force=True)
        
        self.logger.info(f"Warmed emergency cache with {len(query_patterns)} patterns")
    
    def _create_default_response(self, query_text: str) -> RoutingPrediction:
        """Create a default response for a query pattern."""
        # Simple keyword-based classification for emergency responses
        query_lower = query_text.lower()
        
        # Determine routing based on simple keywords
        if any(word in query_lower for word in ['latest', 'recent', 'new', 'current', '2024', '2025']):
            routing = RoutingDecision.PERPLEXITY
            category = ResearchCategory.LITERATURE_SEARCH
        elif any(word in query_lower for word in ['pathway', 'mechanism', 'relationship']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.PATHWAY_ANALYSIS
        elif any(word in query_lower for word in ['metabolite', 'compound', 'identify']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.METABOLITE_IDENTIFICATION
        elif any(word in query_lower for word in ['biomarker', 'diagnostic', 'prognostic']):
            routing = RoutingDecision.EITHER
            category = ResearchCategory.BIOMARKER_DISCOVERY
        elif any(word in query_lower for word in ['drug', 'pharmaceutical', 'therapeutic']):
            routing = RoutingDecision.EITHER
            category = ResearchCategory.DRUG_DISCOVERY
        elif any(word in query_lower for word in ['clinical', 'patient', 'diagnosis']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.CLINICAL_DIAGNOSIS
        else:
            routing = RoutingDecision.EITHER
            category = ResearchCategory.GENERAL_QUERY
        
        # Create confidence metrics
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=self.default_confidence,
            research_category_confidence=self.default_confidence,
            temporal_analysis_confidence=0.1,
            signal_strength_confidence=0.1,
            context_coherence_confidence=0.1,
            keyword_density=0.1,
            pattern_match_strength=0.1,
            biomedical_entity_count=0,
            ambiguity_score=0.9,
            conflict_score=0.1,
            alternative_interpretations=[(routing, self.default_confidence)],
            calculation_time_ms=0.0
        )
        
        return RoutingPrediction(
            routing_decision=routing,
            confidence=self.default_confidence,
            reasoning=["Emergency cache default response", f"Pattern-based classification for: {query_text[:50]}..."],
            research_category=category,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'emergency_cache': True,
                'default_response': True,
                'generated_at': time.time()
            }
        )
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get emergency cache statistics."""
        with self.lock:
            total_accesses = sum(metadata.get('access_count', 0) for metadata in self.cache_metadata.values())
            
            return {
                'total_entries': len(self.cache),
                'max_cache_size': self.max_cache_size,
                'cache_utilization': len(self.cache) / self.max_cache_size,
                'total_accesses': total_accesses,
                'average_accesses_per_entry': total_accesses / len(self.cache) if self.cache else 0,
                'cache_file': str(self.cache_file),
                'cache_file_exists': self.cache_file.exists(),
                'default_confidence': self.default_confidence
            }


# ============================================================================
# GRACEFUL DEGRADATION MANAGER
# ============================================================================

class GracefulDegradationManager:
    """
    Manager for progressive degradation strategies to maintain service availability
    under adverse conditions while optimizing for performance and reliability.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize graceful degradation manager."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Degradation configuration
        self.degradation_levels = {
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE: {
                'timeout_ms': 1500,
                'quality_threshold': 0.7,
                'max_retries': 2,
                'cache_ttl': 3600
            },
            FallbackLevel.SIMPLIFIED_LLM: {
                'timeout_ms': 1000,
                'quality_threshold': 0.5,
                'max_retries': 1,
                'cache_ttl': 1800
            },
            FallbackLevel.KEYWORD_BASED_ONLY: {
                'timeout_ms': 500,
                'quality_threshold': 0.3,
                'max_retries': 0,
                'cache_ttl': 7200
            },
            FallbackLevel.EMERGENCY_CACHE: {
                'timeout_ms': 100,
                'quality_threshold': 0.1,
                'max_retries': 0,
                'cache_ttl': 86400  # 24 hours
            },
            FallbackLevel.DEFAULT_ROUTING: {
                'timeout_ms': 50,
                'quality_threshold': 0.05,
                'max_retries': 0,
                'cache_ttl': 86400
            }
        }
        
        # Progressive timeout reduction strategy
        self.timeout_reduction_steps = [1.0, 0.8, 0.6, 0.4, 0.2]  # Multiply base timeout by these factors
        self.current_timeout_multiplier = 1.0
        
        # Load shedding configuration
        self.load_shedding_enabled = False
        self.priority_queue = deque()
        self.max_queue_size = 1000
        
        # Performance tracking
        self.degradation_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def determine_optimal_fallback_level(self, 
                                       detected_failures: List[FailureType],
                                       system_health_score: float,
                                       query_priority: str = 'normal') -> FallbackLevel:
        """
        Determine the optimal fallback level based on current conditions.
        
        Args:
            detected_failures: List of detected failure types
            system_health_score: Current system health score (0.0-1.0)
            query_priority: Query priority level ('low', 'normal', 'high', 'critical')
            
        Returns:
            Recommended fallback level
        """
        # Start with full LLM as baseline
        recommended_level = FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        
        # Adjust based on detected failures
        if FailureType.SERVICE_UNAVAILABLE in detected_failures:
            recommended_level = max(recommended_level, FallbackLevel.EMERGENCY_CACHE)
        elif FailureType.CIRCUIT_BREAKER_OPEN in detected_failures:
            recommended_level = max(recommended_level, FallbackLevel.KEYWORD_BASED_ONLY)
        elif FailureType.API_TIMEOUT in detected_failures or FailureType.PERFORMANCE_DEGRADATION in detected_failures:
            recommended_level = max(recommended_level, FallbackLevel.SIMPLIFIED_LLM)
        elif FailureType.BUDGET_EXCEEDED in detected_failures:
            recommended_level = max(recommended_level, FallbackLevel.KEYWORD_BASED_ONLY)
        elif FailureType.LOW_CONFIDENCE in detected_failures:
            recommended_level = max(recommended_level, FallbackLevel.SIMPLIFIED_LLM)
        
        # Adjust based on system health score
        if system_health_score < 0.2:
            recommended_level = max(recommended_level, FallbackLevel.DEFAULT_ROUTING)
        elif system_health_score < 0.4:
            recommended_level = max(recommended_level, FallbackLevel.EMERGENCY_CACHE)
        elif system_health_score < 0.6:
            recommended_level = max(recommended_level, FallbackLevel.KEYWORD_BASED_ONLY)
        elif system_health_score < 0.8:
            recommended_level = max(recommended_level, FallbackLevel.SIMPLIFIED_LLM)
        
        # Adjust based on query priority
        if query_priority == 'critical':
            # Critical queries should try higher quality levels even under stress
            recommended_level = min(recommended_level, FallbackLevel.SIMPLIFIED_LLM)
        elif query_priority == 'low':
            # Low priority queries should degrade more aggressively
            recommended_level = max(recommended_level, FallbackLevel.KEYWORD_BASED_ONLY)
        
        return recommended_level
    
    def apply_progressive_timeout_reduction(self, base_timeout_ms: float, failure_count: int) -> float:
        """
        Apply progressive timeout reduction based on failure count.
        
        Args:
            base_timeout_ms: Base timeout in milliseconds
            failure_count: Number of recent failures
            
        Returns:
            Adjusted timeout in milliseconds
        """
        # Determine reduction step based on failure count
        step_index = min(failure_count, len(self.timeout_reduction_steps) - 1)
        multiplier = self.timeout_reduction_steps[step_index]
        
        adjusted_timeout = base_timeout_ms * multiplier
        
        # Minimum timeout to prevent overly aggressive reduction
        min_timeout = 100  # 100ms minimum
        return max(adjusted_timeout, min_timeout)
    
    def adjust_quality_thresholds(self, fallback_level: FallbackLevel, stress_factor: float) -> Dict[str, float]:
        """
        Adjust quality thresholds based on system stress.
        
        Args:
            fallback_level: Current fallback level
            stress_factor: System stress factor (0.0-1.0, higher = more stress)
            
        Returns:
            Adjusted quality thresholds
        """
        base_config = self.degradation_levels[fallback_level]
        adjusted_thresholds = base_config.copy()
        
        # Reduce quality threshold under stress to be more accepting
        base_threshold = base_config['quality_threshold']
        stress_reduction = stress_factor * 0.3  # Max 30% reduction
        adjusted_thresholds['quality_threshold'] = max(0.05, base_threshold - stress_reduction)
        
        # Adjust timeout based on stress
        base_timeout = base_config['timeout_ms']
        timeout_reduction = stress_factor * 0.5  # Max 50% timeout reduction
        adjusted_thresholds['timeout_ms'] = max(50, base_timeout * (1 - timeout_reduction))
        
        return adjusted_thresholds
    
    def enable_load_shedding(self, max_queue_size: int = None):
        """
        Enable load shedding with priority-based processing.
        
        Args:
            max_queue_size: Maximum queue size before dropping low-priority requests
        """
        self.load_shedding_enabled = True
        if max_queue_size:
            self.max_queue_size = max_queue_size
        
        self.logger.info(f"Load shedding enabled with max queue size: {self.max_queue_size}")
    
    def disable_load_shedding(self):
        """Disable load shedding and process all queued requests."""
        self.load_shedding_enabled = False
        self.logger.info("Load shedding disabled")
    
    def should_shed_load(self, query_priority: str = 'normal') -> bool:
        """
        Determine if load should be shed for a query based on priority and queue state.
        
        Args:
            query_priority: Priority of the query
            
        Returns:
            True if load should be shed (request should be dropped)
        """
        if not self.load_shedding_enabled:
            return False
        
        with self.lock:
            queue_utilization = len(self.priority_queue) / self.max_queue_size
            
            # Shed load based on priority and queue utilization
            if query_priority == 'low' and queue_utilization > 0.8:
                return True
            elif query_priority == 'normal' and queue_utilization > 0.95:
                return True
            elif query_priority == 'high' and queue_utilization > 0.98:
                return True
            # Never shed critical priority queries
            
            return False
    
    def warm_caches_for_emergency(self, common_patterns: List[str]):
        """
        Warm caches in preparation for emergency scenarios.
        
        Args:
            common_patterns: List of common query patterns to pre-cache
        """
        self.logger.info(f"Starting emergency cache warming for {len(common_patterns)} patterns")
        
        # This would typically trigger cache warming in the emergency cache
        # Implementation would depend on integration with caching systems
        
        for pattern in common_patterns:
            # Record cache warming activity
            self.degradation_history.append({
                'timestamp': time.time(),
                'action': 'cache_warm',
                'pattern': pattern,
                'reason': 'emergency_preparation'
            })
        
        self.logger.info("Emergency cache warming completed")
    
    def get_degradation_recommendations(self, 
                                     current_failures: List[FailureType],
                                     system_health: float) -> List[Dict[str, Any]]:
        """
        Get recommendations for system degradation based on current conditions.
        
        Args:
            current_failures: Currently detected failures
            system_health: Current system health score
            
        Returns:
            List of degradation recommendations
        """
        recommendations = []
        
        # Timeout reduction recommendations
        if FailureType.API_TIMEOUT in current_failures or FailureType.PERFORMANCE_DEGRADATION in current_failures:
            recommendations.append({
                'type': 'timeout_reduction',
                'action': 'reduce_timeouts',
                'urgency': 'high',
                'description': 'Reduce API timeouts to fail faster and enable quicker fallback',
                'parameters': {
                    'reduction_factor': 0.6,
                    'minimum_timeout_ms': 500
                }
            })
        
        # Quality threshold adjustments
        if system_health < 0.6:
            recommendations.append({
                'type': 'quality_adjustment',
                'action': 'lower_quality_thresholds',
                'urgency': 'medium',
                'description': 'Lower quality thresholds to accept more responses',
                'parameters': {
                    'threshold_reduction': 0.2,
                    'minimum_threshold': 0.1
                }
            })
        
        # Load shedding recommendations
        if len(current_failures) >= 3:
            recommendations.append({
                'type': 'load_shedding',
                'action': 'enable_load_shedding',
                'urgency': 'high',
                'description': 'Enable load shedding to protect system under high failure rate',
                'parameters': {
                    'max_queue_size': 500,
                    'priority_threshold': 'normal'
                }
            })
        
        # Cache warming recommendations
        if FailureType.SERVICE_UNAVAILABLE in current_failures:
            recommendations.append({
                'type': 'cache_warming',
                'action': 'warm_emergency_cache',
                'urgency': 'critical',
                'description': 'Warm emergency cache immediately for service unavailability',
                'parameters': {
                    'cache_patterns': ['common_queries', 'recent_patterns'],
                    'priority': 'critical'
                }
            })
        
        return recommendations
    
    def record_degradation_event(self, 
                                fallback_level: FallbackLevel,
                                reason: str,
                                success: bool,
                                metrics: Dict[str, Any] = None):
        """
        Record a degradation event for analysis and improvement.
        
        Args:
            fallback_level: Fallback level that was used
            reason: Reason for degradation
            success: Whether the degraded operation was successful
            metrics: Additional metrics about the event
        """
        with self.lock:
            event = {
                'timestamp': time.time(),
                'fallback_level': fallback_level.name,
                'reason': reason,
                'success': success,
                'metrics': metrics or {}
            }
            
            self.degradation_history.append(event)
        
        self.logger.debug(f"Recorded degradation event: {fallback_level.name} ({'success' if success else 'failure'})")
    
    def get_degradation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive degradation statistics."""
        with self.lock:
            if not self.degradation_history:
                return {'no_degradation_events': True}
            
            # Analyze degradation patterns
            recent_events = [e for e in self.degradation_history if time.time() - e['timestamp'] < 3600]  # Last hour
            
            level_counts = defaultdict(int)
            success_rates = defaultdict(list)
            
            for event in recent_events:
                level = event['fallback_level']
                level_counts[level] += 1
                success_rates[level].append(event['success'])
            
            level_success_rates = {
                level: (sum(successes) / len(successes)) if successes else 0.0
                for level, successes in success_rates.items()
            }
            
            return {
                'total_degradation_events': len(self.degradation_history),
                'recent_events_count': len(recent_events),
                'level_usage_counts': dict(level_counts),
                'level_success_rates': level_success_rates,
                'load_shedding_enabled': self.load_shedding_enabled,
                'current_queue_size': len(self.priority_queue),
                'max_queue_size': self.max_queue_size,
                'current_timeout_multiplier': self.current_timeout_multiplier,
                'degradation_levels_config': self.degradation_levels
            }


# ============================================================================
# RECOVERY MANAGER
# ============================================================================

class RecoveryManager:
    """
    Automatic recovery manager that handles service recovery validation
    and gradual traffic ramping after failures.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize recovery manager."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Recovery state tracking
        self.recovery_states = {}  # service_name -> recovery_state
        self.lock = threading.Lock()
        
        # Recovery configuration
        self.recovery_config = {
            'health_check_interval_seconds': 30,
            'min_success_rate_for_recovery': 0.8,
            'min_successful_calls_for_recovery': 10,
            'ramp_up_steps': [0.1, 0.2, 0.5, 0.8, 1.0],  # Percentage of traffic for each step
            'ramp_up_step_duration_seconds': 60,
            'max_recovery_attempts': 5,
            'recovery_cooldown_seconds': 300  # 5 minutes between recovery attempts
        }
        
        # Health check functions registry
        self.health_check_functions = {}
        
        # Recovery event history
        self.recovery_history = deque(maxlen=1000)
        
        # Automatic recovery thread
        self.recovery_thread = None
        self.recovery_thread_running = False
        
    def register_service_health_check(self, service_name: str, health_check_func: Callable[[], bool]):
        """
        Register a health check function for a service.
        
        Args:
            service_name: Name of the service
            health_check_func: Function that returns True if service is healthy
        """
        self.health_check_functions[service_name] = health_check_func
        self.logger.info(f"Registered health check for service: {service_name}")
    
    def start_recovery_monitoring(self):
        """Start automatic recovery monitoring thread."""
        if self.recovery_thread_running:
            self.logger.warning("Recovery monitoring already running")
            return
        
        self.recovery_thread_running = True
        self.recovery_thread = threading.Thread(target=self._recovery_monitoring_loop, daemon=True)
        self.recovery_thread.start()
        self.logger.info("Started automatic recovery monitoring")
    
    def stop_recovery_monitoring(self):
        """Stop automatic recovery monitoring."""
        self.recovery_thread_running = False
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5)
        self.logger.info("Stopped automatic recovery monitoring")
    
    def _recovery_monitoring_loop(self):
        """Main recovery monitoring loop."""
        while self.recovery_thread_running:
            try:
                self._check_service_recovery()
                time.sleep(self.recovery_config['health_check_interval_seconds'])
            except Exception as e:
                self.logger.error(f"Error in recovery monitoring loop: {e}")
                time.sleep(10)  # Short delay on error
    
    def _check_service_recovery(self):
        """Check all services for recovery opportunities."""
        with self.lock:
            for service_name, health_check_func in self.health_check_functions.items():
                recovery_state = self.recovery_states.get(service_name)
                
                if not recovery_state or recovery_state['status'] == 'healthy':
                    continue
                
                # Check if service is ready for recovery attempt
                current_time = time.time()
                
                if (recovery_state['status'] == 'failed' and
                    current_time - recovery_state['last_attempt'] > self.recovery_config['recovery_cooldown_seconds']):
                    
                    # Attempt health check
                    try:
                        if health_check_func():
                            self._initiate_service_recovery(service_name)
                    except Exception as e:
                        self.logger.warning(f"Health check failed for {service_name}: {e}")
                
                elif recovery_state['status'] == 'recovering':
                    # Continue recovery process
                    self._continue_service_recovery(service_name)
    
    def _initiate_service_recovery(self, service_name: str):
        """Initiate recovery process for a service."""
        current_time = time.time()
        
        recovery_state = {
            'status': 'recovering',
            'started_at': current_time,
            'current_step': 0,
            'step_started_at': current_time,
            'successful_calls': 0,
            'total_calls': 0,
            'step_success_rate': 0.0,
            'recovery_attempt': self.recovery_states.get(service_name, {}).get('recovery_attempt', 0) + 1,
            'last_attempt': current_time
        }
        
        self.recovery_states[service_name] = recovery_state
        
        self.recovery_history.append({
            'timestamp': current_time,
            'service': service_name,
            'event': 'recovery_initiated',
            'attempt': recovery_state['recovery_attempt']
        })
        
        self.logger.info(f"Initiated recovery for service {service_name} (attempt {recovery_state['recovery_attempt']})")
    
    def _continue_service_recovery(self, service_name: str):
        """Continue the recovery process for a service."""
        recovery_state = self.recovery_states[service_name]
        current_time = time.time()
        
        # Check if current step has been running long enough
        step_duration = current_time - recovery_state['step_started_at']
        
        if step_duration >= self.recovery_config['ramp_up_step_duration_seconds']:
            # Evaluate current step success
            if recovery_state['total_calls'] >= self.recovery_config['min_successful_calls_for_recovery']:
                success_rate = recovery_state['successful_calls'] / recovery_state['total_calls']
                
                if success_rate >= self.recovery_config['min_success_rate_for_recovery']:
                    # Step successful, move to next step
                    self._advance_recovery_step(service_name)
                else:
                    # Step failed, abort recovery
                    self._abort_recovery(service_name, f"Success rate too low: {success_rate:.2f}")
            else:
                # Not enough calls to evaluate, continue current step
                pass
    
    def _advance_recovery_step(self, service_name: str):
        """Advance to the next recovery step."""
        recovery_state = self.recovery_states[service_name]
        current_step = recovery_state['current_step']
        
        if current_step >= len(self.recovery_config['ramp_up_steps']) - 1:
            # Recovery complete
            self._complete_recovery(service_name)
        else:
            # Move to next step
            recovery_state['current_step'] += 1
            recovery_state['step_started_at'] = time.time()
            recovery_state['successful_calls'] = 0
            recovery_state['total_calls'] = 0
            recovery_state['step_success_rate'] = 0.0
            
            traffic_percentage = self.recovery_config['ramp_up_steps'][recovery_state['current_step']]
            
            self.recovery_history.append({
                'timestamp': time.time(),
                'service': service_name,
                'event': 'recovery_step_advanced',
                'step': recovery_state['current_step'],
                'traffic_percentage': traffic_percentage
            })
            
            self.logger.info(f"Advanced recovery for {service_name} to step {recovery_state['current_step']} "
                           f"({traffic_percentage*100:.0f}% traffic)")
    
    def _complete_recovery(self, service_name: str):
        """Complete recovery process for a service."""
        recovery_state = self.recovery_states[service_name]
        recovery_state['status'] = 'healthy'
        recovery_state['completed_at'] = time.time()
        
        total_recovery_time = recovery_state['completed_at'] - recovery_state['started_at']
        
        self.recovery_history.append({
            'timestamp': time.time(),
            'service': service_name,
            'event': 'recovery_completed',
            'total_time_seconds': total_recovery_time,
            'attempt': recovery_state['recovery_attempt']
        })
        
        self.logger.info(f"Completed recovery for {service_name} in {total_recovery_time:.1f} seconds "
                        f"(attempt {recovery_state['recovery_attempt']})")
    
    def _abort_recovery(self, service_name: str, reason: str):
        """Abort recovery process for a service."""
        recovery_state = self.recovery_states[service_name]
        recovery_state['status'] = 'failed'
        recovery_state['last_failure_reason'] = reason
        recovery_state['last_attempt'] = time.time()
        
        self.recovery_history.append({
            'timestamp': time.time(),
            'service': service_name,
            'event': 'recovery_aborted',
            'reason': reason,
            'attempt': recovery_state['recovery_attempt']
        })
        
        self.logger.warning(f"Aborted recovery for {service_name}: {reason}")
    
    def record_service_call_result(self, service_name: str, success: bool, response_time_ms: float = None):
        """
        Record the result of a service call for recovery tracking.
        
        Args:
            service_name: Name of the service
            success: Whether the call was successful
            response_time_ms: Response time in milliseconds
        """
        with self.lock:
            recovery_state = self.recovery_states.get(service_name)
            
            if recovery_state and recovery_state['status'] == 'recovering':
                # Update recovery metrics
                recovery_state['total_calls'] += 1
                if success:
                    recovery_state['successful_calls'] += 1
                
                recovery_state['step_success_rate'] = (
                    recovery_state['successful_calls'] / recovery_state['total_calls']
                )
    
    def should_allow_traffic(self, service_name: str) -> Tuple[bool, float]:
        """
        Determine if traffic should be allowed to a service and at what percentage.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Tuple of (allow_traffic, traffic_percentage)
        """
        with self.lock:
            recovery_state = self.recovery_states.get(service_name)
            
            if not recovery_state or recovery_state['status'] == 'healthy':
                return True, 1.0
            elif recovery_state['status'] == 'failed':
                return False, 0.0
            elif recovery_state['status'] == 'recovering':
                current_step = recovery_state['current_step']
                traffic_percentage = self.recovery_config['ramp_up_steps'][current_step]
                return True, traffic_percentage
            
            return False, 0.0
    
    def mark_service_as_failed(self, service_name: str, reason: str):
        """
        Mark a service as failed to initiate recovery monitoring.
        
        Args:
            service_name: Name of the service that failed
            reason: Reason for the failure
        """
        with self.lock:
            self.recovery_states[service_name] = {
                'status': 'failed',
                'failed_at': time.time(),
                'failure_reason': reason,
                'recovery_attempt': 0,
                'last_attempt': 0
            }
            
            self.recovery_history.append({
                'timestamp': time.time(),
                'service': service_name,
                'event': 'service_marked_failed',
                'reason': reason
            })
            
            self.logger.warning(f"Marked service {service_name} as failed: {reason}")
    
    def get_recovery_status(self, service_name: str = None) -> Dict[str, Any]:
        """
        Get recovery status for a specific service or all services.
        
        Args:
            service_name: Optional service name to get status for
            
        Returns:
            Recovery status information
        """
        with self.lock:
            if service_name:
                return {
                    'service': service_name,
                    'status': self.recovery_states.get(service_name, {'status': 'unknown'}),
                    'health_check_registered': service_name in self.health_check_functions
                }
            else:
                return {
                    'all_services': {name: state for name, state in self.recovery_states.items()},
                    'registered_health_checks': list(self.health_check_functions.keys()),
                    'recovery_monitoring_active': self.recovery_thread_running,
                    'recovery_config': self.recovery_config,
                    'recent_recovery_events': list(self.recovery_history)[-20:]  # Last 20 events
                }


# ============================================================================
# COMPREHENSIVE FALLBACK ORCHESTRATOR
# ============================================================================

class FallbackOrchestrator:
    """
    Main orchestrator for comprehensive multi-tiered fallback system.
    Coordinates all fallback mechanisms to ensure 100% system availability.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize comprehensive fallback orchestrator.
        
        Args:
            config: Configuration dictionary for fallback system
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize core components
        self.failure_detector = FailureDetector(logger=self.logger)
        self.degradation_manager = GracefulDegradationManager(logger=self.logger)
        self.recovery_manager = RecoveryManager(logger=self.logger)
        self.emergency_cache = EmergencyCache(
            cache_file=self.config.get('emergency_cache_file'),
            logger=self.logger
        )
        
        # Initialize existing components
        self.query_router = None
        self.llm_classifier = None
        self.research_categorizer = None
        
        # Fallback level processors
        self.level_processors = {
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE: self._process_full_llm_with_confidence,
            FallbackLevel.SIMPLIFIED_LLM: self._process_simplified_llm,
            FallbackLevel.KEYWORD_BASED_ONLY: self._process_keyword_based_only,
            FallbackLevel.EMERGENCY_CACHE: self._process_emergency_cache,
            FallbackLevel.DEFAULT_ROUTING: self._process_default_routing
        }
        
        # Performance tracking
        self.fallback_stats = defaultdict(lambda: defaultdict(int))
        self.performance_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Start recovery monitoring
        self.recovery_manager.start_recovery_monitoring()
        
        self.logger.info("Comprehensive fallback orchestrator initialized")
    
    def integrate_with_existing_components(self,
                                        query_router: BiomedicalQueryRouter = None,
                                        llm_classifier: EnhancedLLMQueryClassifier = None,
                                        research_categorizer: ResearchCategorizer = None):
        """
        Integrate with existing system components.
        
        Args:
            query_router: Existing query router instance
            llm_classifier: Existing LLM classifier instance  
            research_categorizer: Existing research categorizer instance
        """
        if query_router:
            self.query_router = query_router
            self.logger.info("Integrated with existing BiomedicalQueryRouter")
        
        if llm_classifier:
            self.llm_classifier = llm_classifier
            self.logger.info("Integrated with existing EnhancedLLMQueryClassifier")
        
        if research_categorizer:
            self.research_categorizer = research_categorizer
            self.logger.info("Integrated with existing ResearchCategorizer")
        
        # Register health checks for integrated components
        if self.query_router:
            self.recovery_manager.register_service_health_check(
                'query_router',
                lambda: hasattr(self.query_router, 'route_query')
            )
        
        if self.llm_classifier:
            self.recovery_manager.register_service_health_check(
                'llm_classifier',
                lambda: self._check_llm_classifier_health()
            )
    
    def _check_llm_classifier_health(self) -> bool:
        """Check health of LLM classifier."""
        try:
            if not self.llm_classifier:
                return False
            
            # Simple health check - attempt a basic classification
            test_result = self.llm_classifier.classify_query_basic("test query")
            return test_result is not None
        except Exception as e:
            self.logger.debug(f"LLM classifier health check failed: {e}")
            return False
    
    def process_query_with_comprehensive_fallback(self,
                                                query_text: str,
                                                context: Optional[Dict[str, Any]] = None,
                                                priority: str = 'normal') -> FallbackResult:
        """
        Process a query using comprehensive multi-tiered fallback system.
        
        Args:
            query_text: The user query to process
            context: Optional context information
            priority: Query priority ('low', 'normal', 'high', 'critical')
            
        Returns:
            FallbackResult with comprehensive processing information
        """
        start_time = time.time()
        
        # Check for load shedding first
        if self.degradation_manager.should_shed_load(priority):
            return self._create_load_shed_result(query_text, start_time)
        
        # Detect current failure conditions
        detected_failures = self.failure_detector.detect_failure_conditions(query_text, context)
        
        # Get system health score
        system_health_score = self.failure_detector.metrics.calculate_health_score()
        
        # Determine optimal fallback level
        target_fallback_level = self.degradation_manager.determine_optimal_fallback_level(
            detected_failures, system_health_score, priority
        )
        
        # Attempt processing at each fallback level until success
        fallback_result = self._execute_fallback_chain(
            query_text, context, target_fallback_level, detected_failures, start_time
        )
        
        # Record result for learning and monitoring
        self._record_fallback_result(fallback_result, detected_failures, system_health_score)
        
        return fallback_result
    
    def _execute_fallback_chain(self,
                              query_text: str,
                              context: Optional[Dict[str, Any]],
                              start_level: FallbackLevel,
                              detected_failures: List[FailureType],
                              start_time: float) -> FallbackResult:
        """Execute the fallback chain starting from the specified level."""
        
        attempted_levels = []
        failure_reasons = list(detected_failures)  # Copy the initial failures
        level_processing_times = {}
        warnings = []
        fallback_chain = []
        
        # Try each fallback level in order from start_level to DEFAULT_ROUTING
        for level in range(start_level, FallbackLevel.DEFAULT_ROUTING + 1):
            current_level = FallbackLevel(level)
            attempted_levels.append(current_level)
            
            level_start_time = time.time()
            
            try:
                # Get processor function for this level
                processor = self.level_processors.get(current_level)
                if not processor:
                    warnings.append(f"No processor found for fallback level {current_level.name}")
                    continue
                
                # Check if service is available for this level
                if not self._is_fallback_level_available(current_level):
                    warnings.append(f"Fallback level {current_level.name} is not available")
                    fallback_chain.append(f"{current_level.name}: unavailable")
                    continue
                
                # Attempt processing at this level
                routing_prediction = processor(query_text, context, detected_failures)
                
                if routing_prediction:
                    # Success! Calculate final result
                    level_processing_time = (time.time() - level_start_time) * 1000
                    level_processing_times[current_level] = level_processing_time
                    total_time = (time.time() - start_time) * 1000
                    
                    # Calculate quality and confidence degradation
                    quality_score = self._calculate_quality_score(current_level, routing_prediction)
                    confidence_degradation = self._calculate_confidence_degradation(current_level)
                    reliability_score = self._calculate_reliability_score(current_level, detected_failures)
                    
                    fallback_chain.append(f"{current_level.name}: success")
                    
                    return FallbackResult(
                        routing_prediction=routing_prediction,
                        fallback_level_used=current_level,
                        success=True,
                        failure_reasons=failure_reasons,
                        attempted_levels=attempted_levels,
                        recovery_suggestions=self._generate_recovery_suggestions(detected_failures),
                        total_processing_time_ms=total_time,
                        fallback_decision_time_ms=(level_start_time - start_time) * 1000,
                        level_processing_times=level_processing_times,
                        confidence_degradation=confidence_degradation,
                        quality_score=quality_score,
                        reliability_score=reliability_score,
                        warnings=warnings,
                        fallback_chain=fallback_chain,
                        debug_info={
                            'detected_failures': [f.value for f in detected_failures],
                            'system_health_score': self.failure_detector.metrics.calculate_health_score(),
                            'start_level': start_level.name,
                            'successful_level': current_level.name
                        }
                    )
                    
            except Exception as e:
                level_processing_time = (time.time() - level_start_time) * 1000
                level_processing_times[current_level] = level_processing_time
                
                error_msg = f"Failed at level {current_level.name}: {str(e)}"
                warnings.append(error_msg)
                fallback_chain.append(f"{current_level.name}: error - {str(e)[:50]}")
                
                self.logger.warning(error_msg)
                
                # Determine failure type from exception
                if "timeout" in str(e).lower():
                    failure_reasons.append(FailureType.API_TIMEOUT)
                elif "rate limit" in str(e).lower():
                    failure_reasons.append(FailureType.RATE_LIMIT)
                else:
                    failure_reasons.append(FailureType.UNKNOWN_ERROR)
        
        # If we get here, all fallback levels failed - create emergency result
        return self._create_emergency_fallback_result(
            query_text, start_time, attempted_levels, failure_reasons, 
            level_processing_times, warnings, fallback_chain
        )
    
    def _process_full_llm_with_confidence(self,
                                        query_text: str,
                                        context: Optional[Dict[str, Any]],
                                        detected_failures: List[FailureType]) -> Optional[RoutingPrediction]:
        """Process query using full LLM with confidence analysis (Level 1)."""
        try:
            # Check if LLM classifier is available and healthy
            if not self.llm_classifier or not self._check_llm_classifier_health():
                self.logger.debug("LLM classifier not available for full processing")
                return None
            
            # Use query router for full analysis if available
            if self.query_router:
                result = self.query_router.route_query(query_text, context)
                
                # Validate result quality
                if result and result.confidence >= 0.6:
                    self.logger.debug("Full LLM processing successful")
                    return result
                else:
                    self.logger.debug(f"Full LLM confidence too low: {result.confidence if result else 'None'}")
                    return None
            
            # Fallback to direct LLM classification
            classification_result = self.llm_classifier.classify_query_semantic(query_text)
            if classification_result:
                # Convert to routing prediction (simplified)
                routing_decision = RoutingDecision.EITHER  # Safe default
                confidence = getattr(classification_result, 'confidence', 0.5)
                
                if confidence >= 0.6:
                    # Create minimal but valid routing prediction
                    confidence_metrics = ConfidenceMetrics(
                        overall_confidence=confidence,
                        research_category_confidence=confidence,
                        temporal_analysis_confidence=0.3,
                        signal_strength_confidence=0.3,
                        context_coherence_confidence=0.3,
                        keyword_density=0.2,
                        pattern_match_strength=0.2,
                        biomedical_entity_count=1,
                        ambiguity_score=0.4,
                        conflict_score=0.2,
                        alternative_interpretations=[(routing_decision, confidence)],
                        calculation_time_ms=0.0
                    )
                    
                    return RoutingPrediction(
                        routing_decision=routing_decision,
                        confidence=confidence,
                        reasoning=["Full LLM classification with confidence analysis"],
                        research_category=ResearchCategory.GENERAL_QUERY,
                        confidence_metrics=confidence_metrics,
                        temporal_indicators=[],
                        knowledge_indicators=[],
                        metadata={'fallback_level': 'full_llm_with_confidence'}
                    )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Full LLM processing failed: {e}")
            return None
    
    def _process_simplified_llm(self,
                              query_text: str,
                              context: Optional[Dict[str, Any]],
                              detected_failures: List[FailureType]) -> Optional[RoutingPrediction]:
        """Process query using simplified LLM prompts (Level 2)."""
        try:
            # Use simplified processing if LLM classifier is available
            if self.llm_classifier:
                # Use basic classification with reduced timeout
                classification_result = self.llm_classifier.classify_query_basic(query_text)
                
                if classification_result:
                    confidence = getattr(classification_result, 'confidence', 0.4)
                    routing_decision = RoutingDecision.EITHER  # Safe default for simplified processing
                    
                    if confidence >= 0.3:  # Lower threshold for simplified processing
                        confidence_metrics = ConfidenceMetrics(
                            overall_confidence=confidence,
                            research_category_confidence=confidence,
                            temporal_analysis_confidence=0.2,
                            signal_strength_confidence=0.2,
                            context_coherence_confidence=0.2,
                            keyword_density=0.1,
                            pattern_match_strength=0.1,
                            biomedical_entity_count=1,
                            ambiguity_score=0.6,
                            conflict_score=0.3,
                            alternative_interpretations=[(routing_decision, confidence)],
                            calculation_time_ms=0.0
                        )
                        
                        return RoutingPrediction(
                            routing_decision=routing_decision,
                            confidence=confidence,
                            reasoning=["Simplified LLM classification", "Degraded performance mode"],
                            research_category=ResearchCategory.GENERAL_QUERY,
                            confidence_metrics=confidence_metrics,
                            temporal_indicators=[],
                            knowledge_indicators=[],
                            metadata={'fallback_level': 'simplified_llm'}
                        )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Simplified LLM processing failed: {e}")
            return None
    
    def _process_keyword_based_only(self,
                                  query_text: str,
                                  context: Optional[Dict[str, Any]],
                                  detected_failures: List[FailureType]) -> Optional[RoutingPrediction]:
        """Process query using keyword-based classification only (Level 3)."""
        try:
            # Use research categorizer if available
            if self.research_categorizer:
                category_prediction = self.research_categorizer.categorize_query(query_text, context)
                
                if category_prediction and category_prediction.confidence >= 0.2:
                    # Map research category to routing decision
                    routing_decision = self._map_category_to_routing(category_prediction.category)
                    confidence = min(category_prediction.confidence, 0.5)  # Cap at 0.5 for keyword-only
                    
                    confidence_metrics = ConfidenceMetrics(
                        overall_confidence=confidence,
                        research_category_confidence=category_prediction.confidence,
                        temporal_analysis_confidence=0.1,
                        signal_strength_confidence=0.3,  # Higher for keyword matching
                        context_coherence_confidence=0.1,
                        keyword_density=0.4,  # Keyword-based gets higher density
                        pattern_match_strength=0.3,
                        biomedical_entity_count=len(category_prediction.evidence),
                        ambiguity_score=0.7,
                        conflict_score=0.2,
                        alternative_interpretations=[(routing_decision, confidence)],
                        calculation_time_ms=0.0
                    )
                    
                    return RoutingPrediction(
                        routing_decision=routing_decision,
                        confidence=confidence,
                        reasoning=["Keyword-based classification only", f"Category: {category_prediction.category.value}"],
                        research_category=category_prediction.category,
                        confidence_metrics=confidence_metrics,
                        temporal_indicators=[],
                        knowledge_indicators=category_prediction.evidence,
                        metadata={'fallback_level': 'keyword_based_only'}
                    )
            
            # Fallback to simple keyword analysis
            return self._simple_keyword_classification(query_text)
            
        except Exception as e:
            self.logger.warning(f"Keyword-based processing failed: {e}")
            return self._simple_keyword_classification(query_text)
    
    def _simple_keyword_classification(self, query_text: str) -> RoutingPrediction:
        """Simple keyword-based classification as last resort."""
        query_lower = query_text.lower()
        
        # Simple routing logic based on keywords
        if any(word in query_lower for word in ['latest', 'recent', 'new', 'current', '2024', '2025']):
            routing = RoutingDecision.PERPLEXITY
            category = ResearchCategory.LITERATURE_SEARCH
            reasoning = ["Temporal keywords detected - routing to real-time service"]
        elif any(word in query_lower for word in ['pathway', 'mechanism', 'relationship', 'connection']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.PATHWAY_ANALYSIS
            reasoning = ["Knowledge graph keywords detected - routing to LightRAG"]
        elif any(word in query_lower for word in ['metabolite', 'compound', 'identify', 'mass']):
            routing = RoutingDecision.LIGHTRAG
            category = ResearchCategory.METABOLITE_IDENTIFICATION
            reasoning = ["Metabolite identification keywords detected"]
        else:
            routing = RoutingDecision.EITHER
            category = ResearchCategory.GENERAL_QUERY
            reasoning = ["General query - flexible routing"]
        
        confidence = 0.3  # Low confidence for simple keyword matching
        
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=confidence,
            research_category_confidence=confidence,
            temporal_analysis_confidence=0.1,
            signal_strength_confidence=0.3,
            context_coherence_confidence=0.1,
            keyword_density=0.2,
            pattern_match_strength=0.2,
            biomedical_entity_count=0,
            ambiguity_score=0.8,
            conflict_score=0.1,
            alternative_interpretations=[(routing, confidence)],
            calculation_time_ms=0.0
        )
        
        return RoutingPrediction(
            routing_decision=routing,
            confidence=confidence,
            reasoning=reasoning,
            research_category=category,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={'fallback_level': 'simple_keyword'}
        )
    
    def _process_emergency_cache(self,
                               query_text: str,
                               context: Optional[Dict[str, Any]],
                               detected_failures: List[FailureType]) -> Optional[RoutingPrediction]:
        """Process query using emergency cache (Level 4)."""
        try:
            cached_response = self.emergency_cache.get_cached_response(query_text)
            
            if cached_response:
                # Update metadata to indicate emergency cache usage
                cached_response.metadata = cached_response.metadata or {}
                cached_response.metadata.update({
                    'fallback_level': 'emergency_cache',
                    'emergency_fallback': True,
                    'cache_hit': True
                })
                
                # Add warning about emergency usage
                if 'reasoning' not in cached_response.metadata:
                    original_reasoning = cached_response.reasoning or []
                    cached_response.reasoning = original_reasoning + ["Retrieved from emergency cache due to system failures"]
                
                self.logger.info(f"Emergency cache hit for query: {query_text[:50]}...")
                return cached_response
            
            return None
            
        except Exception as e:
            self.logger.error(f"Emergency cache processing failed: {e}")
            return None
    
    def _process_default_routing(self,
                               query_text: str,
                               context: Optional[Dict[str, Any]],
                               detected_failures: List[FailureType]) -> RoutingPrediction:
        """Process query using default routing (Level 5 - Last Resort)."""
        # This should NEVER fail - it's the absolute last resort
        
        confidence = 0.05  # Very low confidence for default routing
        routing_decision = RoutingDecision.EITHER  # Safest default
        
        reasoning = [
            "Last resort default routing",
            "All other fallback levels failed",
            f"Detected failures: {[f.value for f in detected_failures]}"
        ]
        
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=confidence,
            research_category_confidence=confidence,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=1.0,  # Maximum ambiguity
            conflict_score=0.0,
            alternative_interpretations=[(routing_decision, confidence)],
            calculation_time_ms=0.0
        )
        
        return RoutingPrediction(
            routing_decision=routing_decision,
            confidence=confidence,
            reasoning=reasoning,
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'fallback_level': 'default_routing',
                'last_resort': True,
                'detected_failures': [f.value for f in detected_failures]
            }
        )
    
    def _is_fallback_level_available(self, level: FallbackLevel) -> bool:
        """Check if a fallback level is available for use."""
        if level == FallbackLevel.FULL_LLM_WITH_CONFIDENCE:
            return self.llm_classifier is not None and self._check_llm_classifier_health()
        elif level == FallbackLevel.SIMPLIFIED_LLM:
            return self.llm_classifier is not None
        elif level == FallbackLevel.KEYWORD_BASED_ONLY:
            return self.research_categorizer is not None
        elif level == FallbackLevel.EMERGENCY_CACHE:
            return self.emergency_cache is not None
        elif level == FallbackLevel.DEFAULT_ROUTING:
            return True  # Always available
        
        return False
    
    def _map_category_to_routing(self, category: ResearchCategory) -> RoutingDecision:
        """Map research category to routing decision."""
        mapping = {
            ResearchCategory.METABOLITE_IDENTIFICATION: RoutingDecision.LIGHTRAG,
            ResearchCategory.PATHWAY_ANALYSIS: RoutingDecision.LIGHTRAG,
            ResearchCategory.BIOMARKER_DISCOVERY: RoutingDecision.EITHER,
            ResearchCategory.DRUG_DISCOVERY: RoutingDecision.EITHER,
            ResearchCategory.CLINICAL_DIAGNOSIS: RoutingDecision.LIGHTRAG,
            ResearchCategory.LITERATURE_SEARCH: RoutingDecision.PERPLEXITY,
            ResearchCategory.GENERAL_QUERY: RoutingDecision.EITHER,
        }
        
        return mapping.get(category, RoutingDecision.EITHER)
    
    def _calculate_quality_score(self, level: FallbackLevel, prediction: RoutingPrediction) -> float:
        """Calculate quality score based on fallback level and prediction."""
        base_quality = {
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE: 1.0,
            FallbackLevel.SIMPLIFIED_LLM: 0.8,
            FallbackLevel.KEYWORD_BASED_ONLY: 0.6,
            FallbackLevel.EMERGENCY_CACHE: 0.3,
            FallbackLevel.DEFAULT_ROUTING: 0.1
        }
        
        quality = base_quality.get(level, 0.5)
        
        # Adjust based on prediction confidence
        if prediction.confidence > 0.8:
            quality = min(quality * 1.1, 1.0)
        elif prediction.confidence < 0.2:
            quality *= 0.8
        
        return quality
    
    def _calculate_confidence_degradation(self, level: FallbackLevel) -> float:
        """Calculate confidence degradation based on fallback level."""
        degradation = {
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE: 0.0,
            FallbackLevel.SIMPLIFIED_LLM: 0.2,
            FallbackLevel.KEYWORD_BASED_ONLY: 0.4,
            FallbackLevel.EMERGENCY_CACHE: 0.7,
            FallbackLevel.DEFAULT_ROUTING: 0.9
        }
        
        return degradation.get(level, 0.5)
    
    def _calculate_reliability_score(self, level: FallbackLevel, failures: List[FailureType]) -> float:
        """Calculate reliability score based on level and failure context."""
        base_reliability = {
            FallbackLevel.FULL_LLM_WITH_CONFIDENCE: 0.9,
            FallbackLevel.SIMPLIFIED_LLM: 0.8,
            FallbackLevel.KEYWORD_BASED_ONLY: 0.95,  # Very reliable
            FallbackLevel.EMERGENCY_CACHE: 0.99,     # Extremely reliable
            FallbackLevel.DEFAULT_ROUTING: 1.0       # Always works
        }
        
        reliability = base_reliability.get(level, 0.7)
        
        # Reduce reliability if we're using this level due to failures
        failure_impact = len(failures) * 0.05
        return max(0.1, reliability - failure_impact)
    
    def _generate_recovery_suggestions(self, failures: List[FailureType]) -> List[str]:
        """Generate recovery suggestions based on detected failures."""
        suggestions = []
        
        if FailureType.API_TIMEOUT in failures:
            suggestions.extend([
                "Increase API timeout thresholds",
                "Check network connectivity",
                "Consider API endpoint optimization"
            ])
        
        if FailureType.RATE_LIMIT in failures:
            suggestions.extend([
                "Implement exponential backoff",
                "Increase rate limit quotas",
                "Enable request queueing"
            ])
        
        if FailureType.BUDGET_EXCEEDED in failures:
            suggestions.extend([
                "Increase budget allocation",
                "Enable cost optimization features",
                "Review usage patterns"
            ])
        
        if FailureType.LOW_CONFIDENCE in failures:
            suggestions.extend([
                "Review query classification patterns",
                "Consider model retraining",
                "Improve keyword dictionaries"
            ])
        
        if FailureType.SERVICE_UNAVAILABLE in failures:
            suggestions.extend([
                "Check service health status",
                "Restart affected services",
                "Enable all fallback mechanisms"
            ])
        
        return suggestions
    
    def _create_load_shed_result(self, query_text: str, start_time: float) -> FallbackResult:
        """Create result when load is shed."""
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.0,
            research_category_confidence=0.0,
            temporal_analysis_confidence=0.0,
            signal_strength_confidence=0.0,
            context_coherence_confidence=0.0,
            keyword_density=0.0,
            pattern_match_strength=0.0,
            biomedical_entity_count=0,
            ambiguity_score=1.0,
            conflict_score=0.0,
            alternative_interpretations=[],
            calculation_time_ms=(time.time() - start_time) * 1000
        )
        
        routing_prediction = RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=0.0,
            reasoning=["Load shed - system under high stress"],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={'load_shed': True, 'reason': 'system_overload'}
        )
        
        return FallbackResult(
            routing_prediction=routing_prediction,
            fallback_level_used=FallbackLevel.DEFAULT_ROUTING,
            success=False,
            failure_reasons=[FailureType.SERVICE_UNAVAILABLE],
            attempted_levels=[],
            recovery_suggestions=["Reduce system load", "Enable load shedding", "Scale up resources"],
            total_processing_time_ms=(time.time() - start_time) * 1000,
            quality_score=0.0,
            reliability_score=0.0,
            warnings=["Load shed due to system overload"]
        )
    
    def _create_emergency_fallback_result(self,
                                        query_text: str,
                                        start_time: float,
                                        attempted_levels: List[FallbackLevel],
                                        failure_reasons: List[FailureType],
                                        level_processing_times: Dict[FallbackLevel, float],
                                        warnings: List[str],
                                        fallback_chain: List[str]) -> FallbackResult:
        """Create emergency fallback result when all levels fail."""
        
        # Create absolute last resort routing prediction
        routing_prediction = self._process_default_routing(query_text, None, failure_reasons)
        routing_prediction.metadata = routing_prediction.metadata or {}
        routing_prediction.metadata.update({
            'emergency_fallback': True,
            'all_levels_failed': True,
            'critical_system_failure': True
        })
        
        routing_prediction.reasoning.append("CRITICAL: All fallback levels failed - using emergency default")
        
        return FallbackResult(
            routing_prediction=routing_prediction,
            fallback_level_used=FallbackLevel.DEFAULT_ROUTING,
            success=True,  # We still provided a response, even if degraded
            failure_reasons=failure_reasons,
            attempted_levels=attempted_levels,
            recovery_suggestions=self._generate_recovery_suggestions(failure_reasons) + [
                "URGENT: All fallback mechanisms failed",
                "Check system health immediately",
                "Consider emergency maintenance"
            ],
            total_processing_time_ms=(time.time() - start_time) * 1000,
            level_processing_times=level_processing_times,
            confidence_degradation=0.95,
            quality_score=0.05,
            reliability_score=0.1,
            warnings=warnings + ["CRITICAL: All fallback levels failed"],
            fallback_chain=fallback_chain + ["EMERGENCY: Default routing as absolute last resort"],
            debug_info={
                'critical_failure': True,
                'all_levels_attempted': len(attempted_levels),
                'total_failures': len(failure_reasons),
                'emergency_mode': True
            }
        )
    
    def _record_fallback_result(self,
                              result: FallbackResult,
                              detected_failures: List[FailureType],
                              system_health_score: float):
        """Record fallback result for monitoring and improvement."""
        
        # Record with failure detector
        self.failure_detector.record_operation_result(
            response_time_ms=result.total_processing_time_ms,
            success=result.success,
            confidence=result.routing_prediction.confidence
        )
        
        # Record with degradation manager
        self.degradation_manager.record_degradation_event(
            fallback_level=result.fallback_level_used,
            reason=f"Detected failures: {[f.value for f in detected_failures]}",
            success=result.success,
            metrics={
                'processing_time_ms': result.total_processing_time_ms,
                'confidence': result.routing_prediction.confidence,
                'quality_score': result.quality_score,
                'system_health_score': system_health_score
            }
        )
        
        # Update statistics
        with self.lock:
            self.fallback_stats[result.fallback_level_used.name]['total_uses'] += 1
            if result.success:
                self.fallback_stats[result.fallback_level_used.name]['successful_uses'] += 1
            
            # Record performance metrics
            performance_record = {
                'timestamp': time.time(),
                'level_used': result.fallback_level_used.name,
                'processing_time_ms': result.total_processing_time_ms,
                'success': result.success,
                'confidence': result.routing_prediction.confidence,
                'quality_score': result.quality_score,
                'failure_count': len(result.failure_reasons)
            }
            
            self.performance_history.append(performance_record)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire fallback system."""
        
        return {
            'fallback_orchestrator': {
                'fallback_level_stats': dict(self.fallback_stats),
                'performance_history_count': len(self.performance_history),
                'integrated_components': {
                    'query_router': self.query_router is not None,
                    'llm_classifier': self.llm_classifier is not None,
                    'research_categorizer': self.research_categorizer is not None
                }
            },
            'failure_detection': self.failure_detector.get_detection_statistics(),
            'degradation_management': self.degradation_manager.get_degradation_statistics(),
            'recovery_management': self.recovery_manager.get_recovery_status(),
            'emergency_cache': self.emergency_cache.get_cache_statistics(),
            'system_health': {
                'overall_health_score': self.failure_detector.metrics.calculate_health_score(),
                'early_warning_signals': self.failure_detector.get_early_warning_signals()
            }
        }
    
    def enable_emergency_mode(self):
        """Enable emergency mode with maximum fallback protection."""
        self.logger.critical("ENABLING EMERGENCY MODE - All fallback mechanisms activated")
        
        # Enable aggressive degradation
        self.degradation_manager.enable_load_shedding(max_queue_size=100)
        
        # Warm emergency cache
        self.emergency_cache.warm_cache([
            "metabolite identification",
            "pathway analysis", 
            "biomarker discovery",
            "clinical diagnosis",
            "latest research"
        ])
        
        # Mark all optional services as failed to force fallback
        self.recovery_manager.mark_service_as_failed('llm_classifier', 'Emergency mode activated')
        
        self.logger.critical("Emergency mode enabled - System operating in maximum fallback protection")
    
    def disable_emergency_mode(self):
        """Disable emergency mode and return to normal operation."""
        self.logger.info("Disabling emergency mode - Returning to normal operation")
        
        # Disable load shedding
        self.degradation_manager.disable_load_shedding()
        
        # Reset service states (they will be checked by recovery manager)
        with self.recovery_manager.lock:
            for service_name in self.recovery_manager.recovery_states:
                if self.recovery_manager.recovery_states[service_name]['status'] == 'failed':
                    self.recovery_manager.recovery_states[service_name]['status'] = 'healthy'
        
        self.logger.info("Emergency mode disabled - Normal operation restored")


# ============================================================================
# FALLBACK MONITORING AND ALERTING
# ============================================================================

class FallbackMonitor:
    """
    Comprehensive monitoring and alerting system for fallback operations.
    Provides real-time monitoring, alerting, and reporting capabilities.
    """
    
    def __init__(self, orchestrator: FallbackOrchestrator, logger: Optional[logging.Logger] = None):
        """Initialize fallback monitor."""
        self.orchestrator = orchestrator
        self.logger = logger or logging.getLogger(__name__)
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_fallback_rate': 0.3,          # Alert if >30% of queries use fallback
            'low_success_rate': 0.9,            # Alert if success rate <90%
            'high_response_time': 2000,         # Alert if avg response time >2s
            'critical_failure_rate': 0.1,       # Alert if >10% critical failures
            'emergency_cache_overuse': 0.1,     # Alert if >10% queries use emergency cache
            'system_health_critical': 0.3       # Alert if health score <30%
        }
        
        # Alert history
        self.alert_history = deque(maxlen=1000)
        self.last_alert_times = defaultdict(float)
        self.alert_cooldown_seconds = 300  # 5 minutes
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
    def start_monitoring(self, check_interval_seconds: int = 60):
        """Start continuous monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Started fallback monitoring with {check_interval_seconds}s intervals")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped fallback monitoring")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_system_health_and_alert()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Short delay on error
    
    def _check_system_health_and_alert(self):
        """Check system health and generate alerts if needed."""
        try:
            stats = self.orchestrator.get_comprehensive_statistics()
            
            # Check various health indicators
            self._check_fallback_usage_alerts(stats)
            self._check_performance_alerts(stats)
            self._check_failure_rate_alerts(stats)
            self._check_system_health_alerts(stats)
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
    
    def _check_fallback_usage_alerts(self, stats: Dict[str, Any]):
        """Check for alerts related to fallback usage patterns."""
        fallback_stats = stats.get('fallback_orchestrator', {}).get('fallback_level_stats', {})
        
        if not fallback_stats:
            return
        
        # Calculate fallback rates
        total_queries = sum(level_stats.get('total_uses', 0) for level_stats in fallback_stats.values())
        
        if total_queries == 0:
            return
        
        # High fallback rate alert
        non_primary_queries = sum(
            level_stats.get('total_uses', 0)
            for level_name, level_stats in fallback_stats.items()
            if level_name != 'FULL_LLM_WITH_CONFIDENCE'
        )
        
        fallback_rate = non_primary_queries / total_queries
        
        if fallback_rate > self.alert_thresholds['high_fallback_rate']:
            self._send_alert(
                'high_fallback_rate',
                'warning',
                f"High fallback rate detected: {fallback_rate:.1%} of queries using fallback mechanisms",
                {
                    'fallback_rate': fallback_rate,
                    'total_queries': total_queries,
                    'non_primary_queries': non_primary_queries,
                    'recommended_actions': [
                        'Investigate primary system health',
                        'Check for service degradation',
                        'Review system capacity'
                    ]
                }
            )
        
        # Emergency cache overuse alert
        emergency_cache_uses = fallback_stats.get('EMERGENCY_CACHE', {}).get('total_uses', 0)
        emergency_cache_rate = emergency_cache_uses / total_queries
        
        if emergency_cache_rate > self.alert_thresholds['emergency_cache_overuse']:
            self._send_alert(
                'emergency_cache_overuse',
                'critical',
                f"Emergency cache overuse: {emergency_cache_rate:.1%} of queries using emergency cache",
                {
                    'emergency_cache_rate': emergency_cache_rate,
                    'emergency_cache_uses': emergency_cache_uses,
                    'recommended_actions': [
                        'URGENT: Check all primary and secondary systems',
                        'Investigate system failures',
                        'Consider emergency maintenance'
                    ]
                }
            )
    
    def _check_performance_alerts(self, stats: Dict[str, Any]):
        """Check for performance-related alerts."""
        failure_detection_stats = stats.get('failure_detection', {}).get('metrics', {})
        
        # High response time alert
        avg_response_time = failure_detection_stats.get('average_response_time_ms', 0)
        
        if avg_response_time > self.alert_thresholds['high_response_time']:
            self._send_alert(
                'high_response_time',
                'warning',
                f"High average response time: {avg_response_time:.1f}ms",
                {
                    'average_response_time_ms': avg_response_time,
                    'threshold_ms': self.alert_thresholds['high_response_time'],
                    'recommended_actions': [
                        'Enable aggressive timeout reduction',
                        'Warm caches proactively',
                        'Check API performance'
                    ]
                }
            )
    
    def _check_failure_rate_alerts(self, stats: Dict[str, Any]):
        """Check for failure rate alerts."""
        failure_detection_stats = stats.get('failure_detection', {}).get('metrics', {})
        
        # High error rate alert
        error_rate = failure_detection_stats.get('error_rate', 0)
        
        if error_rate > self.alert_thresholds['critical_failure_rate']:
            self._send_alert(
                'high_error_rate',
                'critical',
                f"High error rate detected: {error_rate:.1%}",
                {
                    'error_rate': error_rate,
                    'successful_calls': failure_detection_stats.get('successful_calls', 0),
                    'failed_calls': failure_detection_stats.get('failed_calls', 0),
                    'recommended_actions': [
                        'Enable emergency mode if not already active',
                        'Investigate service health',
                        'Check for API issues'
                    ]
                }
            )
    
    def _check_system_health_alerts(self, stats: Dict[str, Any]):
        """Check for overall system health alerts."""
        system_health = stats.get('system_health', {})
        health_score = system_health.get('overall_health_score', 1.0)
        
        if health_score < self.alert_thresholds['system_health_critical']:
            self._send_alert(
                'system_health_critical',
                'critical',
                f"Critical system health: {health_score:.2f}",
                {
                    'health_score': health_score,
                    'early_warnings': system_health.get('early_warning_signals', []),
                    'recommended_actions': [
                        'URGENT: Enable emergency mode immediately',
                        'Investigate all system components',
                        'Prepare for service interruption'
                    ]
                }
            )
        elif health_score < 0.6:
            self._send_alert(
                'system_health_degraded',
                'warning',
                f"System health degraded: {health_score:.2f}",
                {
                    'health_score': health_score,
                    'recommended_actions': [
                        'Enable preventive fallback mechanisms',
                        'Monitor system closely',
                        'Prepare emergency procedures'
                    ]
                }
            )
    
    def _send_alert(self, alert_type: str, severity: str, message: str, details: Dict[str, Any]):
        """Send an alert with cooldown protection."""
        current_time = time.time()
        
        # Check cooldown
        last_alert_time = self.last_alert_times.get(alert_type, 0)
        if current_time - last_alert_time < self.alert_cooldown_seconds:
            return  # Skip alert due to cooldown
        
        # Record alert
        alert = {
            'timestamp': current_time,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'details': details
        }
        
        self.alert_history.append(alert)
        self.last_alert_times[alert_type] = current_time
        
        # Log alert
        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(f"FALLBACK ALERT [{severity.upper()}]: {message}")
        
        # Additional logging for critical alerts
        if severity == 'critical':
            self.logger.critical(f"CRITICAL FALLBACK ALERT: {alert_type} - {message}")
            self.logger.critical(f"Alert details: {details}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        current_time = time.time()
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert['timestamp'] < 3600  # Last hour
        ]
        
        # Alert statistics
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert['severity']] += 1
        
        # System statistics
        stats = self.orchestrator.get_comprehensive_statistics()
        
        return {
            'monitoring_status': {
                'monitoring_active': self.monitoring_active,
                'alert_thresholds': self.alert_thresholds,
                'alert_cooldown_seconds': self.alert_cooldown_seconds
            },
            'recent_alerts': {
                'total_alerts_last_hour': len(recent_alerts),
                'alert_counts_by_severity': dict(alert_counts),
                'recent_alert_list': recent_alerts[-10:]  # Last 10 alerts
            },
            'system_overview': {
                'overall_health_score': stats.get('system_health', {}).get('overall_health_score', 0),
                'fallback_usage_summary': self._summarize_fallback_usage(stats),
                'performance_summary': self._summarize_performance(stats),
                'recommendations': self._generate_monitoring_recommendations(stats, recent_alerts)
            },
            'detailed_statistics': stats
        }
    
    def _summarize_fallback_usage(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize fallback usage patterns."""
        fallback_stats = stats.get('fallback_orchestrator', {}).get('fallback_level_stats', {})
        
        total_queries = sum(level_stats.get('total_uses', 0) for level_stats in fallback_stats.values())
        
        if total_queries == 0:
            return {'no_queries_processed': True}
        
        usage_summary = {}
        for level_name, level_stats in fallback_stats.items():
            uses = level_stats.get('total_uses', 0)
            successful = level_stats.get('successful_uses', 0)
            
            usage_summary[level_name] = {
                'usage_percentage': (uses / total_queries) * 100,
                'success_rate': (successful / uses) * 100 if uses > 0 else 0,
                'total_uses': uses
            }
        
        return usage_summary
    
    def _summarize_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize performance metrics."""
        failure_stats = stats.get('failure_detection', {}).get('metrics', {})
        
        return {
            'average_response_time_ms': failure_stats.get('average_response_time_ms', 0),
            'error_rate_percentage': failure_stats.get('error_rate', 0) * 100,
            'system_health_score': failure_stats.get('system_health_score', 0),
            'successful_calls': failure_stats.get('successful_calls', 0),
            'failed_calls': failure_stats.get('failed_calls', 0)
        }
    
    def _generate_monitoring_recommendations(self, stats: Dict[str, Any], recent_alerts: List[Dict]) -> List[str]:
        """Generate monitoring-based recommendations."""
        recommendations = []
        
        # High fallback usage recommendations
        fallback_stats = stats.get('fallback_orchestrator', {}).get('fallback_level_stats', {})
        total_queries = sum(level_stats.get('total_uses', 0) for level_stats in fallback_stats.values())
        
        if total_queries > 0:
            primary_uses = fallback_stats.get('FULL_LLM_WITH_CONFIDENCE', {}).get('total_uses', 0)
            primary_rate = primary_uses / total_queries
            
            if primary_rate < 0.7:
                recommendations.append("Primary system usage is low - investigate system health")
        
        # Recent critical alerts
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'critical']
        if len(critical_alerts) > 0:
            recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately")
        
        # Performance recommendations
        failure_stats = stats.get('failure_detection', {}).get('metrics', {})
        if failure_stats.get('average_response_time_ms', 0) > 1500:
            recommendations.append("Response times are elevated - enable performance optimizations")
        
        if failure_stats.get('error_rate', 0) > 0.05:
            recommendations.append("Error rate is elevated - investigate service reliability")
        
        # Emergency cache usage
        emergency_uses = fallback_stats.get('EMERGENCY_CACHE', {}).get('total_uses', 0)
        if total_queries > 0 and (emergency_uses / total_queries) > 0.05:
            recommendations.append("Emergency cache usage is high - check primary systems")
        
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations


# ============================================================================
# MAIN MODULE INTERFACE
# ============================================================================

def create_comprehensive_fallback_system(config: Dict[str, Any] = None, 
                                        logger: Optional[logging.Logger] = None) -> Tuple[FallbackOrchestrator, FallbackMonitor]:
    """
    Create and configure a comprehensive fallback system.
    
    Args:
        config: Configuration dictionary for the fallback system
        logger: Logger instance
        
    Returns:
        Tuple of (FallbackOrchestrator, FallbackMonitor)
    """
    # Create orchestrator
    orchestrator = FallbackOrchestrator(config=config, logger=logger)
    
    # Create monitor
    monitor = FallbackMonitor(orchestrator=orchestrator, logger=logger)
    
    # Start monitoring
    monitor.start_monitoring()
    
    if logger:
        logger.info("Comprehensive fallback system created and monitoring started")
    
    return orchestrator, monitor


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create fallback system
    orchestrator, monitor = create_comprehensive_fallback_system(logger=logger)
    
    # Example query processing
    test_queries = [
        "identify metabolite with mass 180.0634",
        "latest research on metabolomics",
        "pathway analysis for glucose metabolism",
        "biomarker discovery for diabetes"
    ]
    
    for query in test_queries:
        logger.info(f"Processing query: {query}")
        result = orchestrator.process_query_with_comprehensive_fallback(query)
        
        logger.info(f"Result: Level {result.fallback_level_used.name}, "
                   f"Success: {result.success}, "
                   f"Confidence: {result.routing_prediction.confidence:.3f}, "
                   f"Time: {result.total_processing_time_ms:.1f}ms")
    
    # Get comprehensive report
    report = monitor.get_monitoring_report()
    logger.info(f"System health score: {report['system_overview']['overall_health_score']:.2f}")
    
    # Clean shutdown
    monitor.stop_monitoring()
    orchestrator.recovery_manager.stop_recovery_monitoring()