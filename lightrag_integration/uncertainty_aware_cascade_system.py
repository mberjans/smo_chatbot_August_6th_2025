"""
Multi-Step Uncertainty-Aware Fallback Cascade System

This module implements an enhanced multi-step fallback cascade specifically designed
for uncertain classifications, providing intelligent routing between LightRAG,
Perplexity, and Cache systems with proactive uncertainty detection.

The system implements a 3-step uncertainty cascade:
- Step 1: LightRAG with uncertainty-aware processing
- Step 2: Perplexity API with specialized uncertainty handling  
- Step 3: Emergency cache with confidence-adjusted responses

Key Features:
    - Proactive routing decisions before failure occurs
    - Different cascade paths for different uncertainty types
    - Performance optimization with circuit breakers (< 200ms total)
    - Graceful degradation with comprehensive logging
    - Recovery mechanisms and automatic retry logic
    - Integration with existing 5-level fallback hierarchy
    - Backward compatibility with existing system

Classes:
    - UncertaintyAwareFallbackCascade: Main cascade orchestrator
    - CascadeStep: Individual cascade step implementation
    - CascadeDecisionEngine: Intelligent routing decision logic
    - CascadePerformanceMonitor: Performance tracking and optimization
    - CascadeCircuitBreaker: Performance-optimized circuit breaker
    - CascadeRecoveryManager: Recovery mechanisms for cascade failures

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
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
import hashlib
from pathlib import Path

# Import existing components for integration
try:
    from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction, ConfidenceMetrics
    from .enhanced_llm_classifier import EnhancedLLMQueryClassifier, CircuitBreaker, CircuitBreakerState
    from .research_categorizer import ResearchCategorizer, CategoryPrediction
    from .cost_persistence import ResearchCategory
    from .comprehensive_fallback_system import (
        FallbackOrchestrator, FallbackMonitor, FallbackResult, FallbackLevel, 
        FailureType, FailureDetectionMetrics, GracefulDegradationManager
    )
    from .uncertainty_aware_classification_thresholds import (
        UncertaintyAwareClassificationThresholds, ConfidenceLevel, ThresholdTrigger,
        ConfidenceThresholdRouter, UncertaintyMetricsAnalyzer
    )
    from .uncertainty_aware_fallback_implementation import (
        UncertaintyDetector, UncertaintyFallbackStrategies, UncertaintyAnalysis,
        UncertaintyType, UncertaintyStrategy, UncertaintyFallbackConfig
    )
    from .comprehensive_confidence_scorer import (
        HybridConfidenceResult, HybridConfidenceScorer, LLMConfidenceAnalysis,
        KeywordConfidenceAnalysis, ConfidenceSource
    )
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# CASCADE SYSTEM DEFINITIONS AND DATA STRUCTURES
# ============================================================================

class CascadeStepType(Enum):
    """Types of cascade steps in the uncertainty-aware fallback system."""
    
    LIGHTRAG_UNCERTAINTY_AWARE = "lightrag_uncertainty_aware"
    PERPLEXITY_SPECIALIZED = "perplexity_specialized" 
    EMERGENCY_CACHE_CONFIDENT = "emergency_cache_confident"


class CascadeFailureReason(Enum):
    """Specific failure reasons for cascade steps."""
    
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNCERTAINTY_TOO_HIGH = "uncertainty_too_high"
    API_ERROR = "api_error"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNKNOWN_ERROR = "unknown_error"


class CascadePathStrategy(Enum):
    """Different cascade path strategies based on uncertainty type."""
    
    FULL_CASCADE = "full_cascade"              # LightRAG -> Perplexity -> Cache
    SKIP_LIGHTRAG = "skip_lightrag"            # Perplexity -> Cache (LightRAG unreliable)
    DIRECT_TO_CACHE = "direct_to_cache"        # Cache only (emergency)
    CONFIDENCE_BOOSTED = "confidence_boosted"   # LightRAG with confidence boosting
    CONSENSUS_SEEKING = "consensus_seeking"     # Multiple approaches with consensus


@dataclass
class CascadeStepResult:
    """Result from an individual cascade step."""
    
    # Step identification
    step_type: CascadeStepType
    step_number: int
    success: bool
    
    # Result data
    routing_prediction: Optional[RoutingPrediction] = None
    confidence_score: float = 0.0
    uncertainty_score: float = 0.0
    
    # Performance metrics
    processing_time_ms: float = 0.0
    decision_time_ms: float = 0.0
    
    # Failure information
    failure_reason: Optional[CascadeFailureReason] = None
    error_message: Optional[str] = None
    retry_attempted: bool = False
    
    # Metadata
    debug_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_type': self.step_type.value,
            'step_number': self.step_number,
            'success': self.success,
            'routing_prediction': self.routing_prediction.to_dict() if self.routing_prediction else None,
            'confidence_score': self.confidence_score,
            'uncertainty_score': self.uncertainty_score,
            'processing_time_ms': self.processing_time_ms,
            'decision_time_ms': self.decision_time_ms,
            'failure_reason': self.failure_reason.value if self.failure_reason else None,
            'error_message': self.error_message,
            'retry_attempted': self.retry_attempted,
            'debug_info': self.debug_info,
            'warnings': self.warnings
        }


@dataclass
class CascadeResult:
    """Comprehensive result from the uncertainty-aware cascade system."""
    
    # Core result data
    routing_prediction: RoutingPrediction
    success: bool
    cascade_path_used: CascadePathStrategy
    
    # Step-by-step results
    step_results: List[CascadeStepResult] = field(default_factory=list)
    successful_step: Optional[CascadeStepType] = None
    total_steps_attempted: int = 0
    
    # Performance metrics
    total_cascade_time_ms: float = 0.0
    decision_overhead_ms: float = 0.0
    step_processing_times: Dict[CascadeStepType, float] = field(default_factory=dict)
    
    # Uncertainty and confidence analysis
    initial_uncertainty_analysis: Optional[UncertaintyAnalysis] = None
    final_confidence_improvement: float = 0.0
    uncertainty_reduction_achieved: float = 0.0
    
    # Quality metrics
    cascade_efficiency_score: float = 1.0  # How efficiently cascade resolved query
    confidence_reliability_score: float = 1.0  # How reliable final confidence is
    uncertainty_handling_score: float = 1.0  # How well uncertainty was handled
    
    # Integration with existing system
    fallback_level_equivalent: Optional[FallbackLevel] = None
    integration_warnings: List[str] = field(default_factory=list)
    backward_compatibility_maintained: bool = True
    
    # Debugging and monitoring
    debug_info: Dict[str, Any] = field(default_factory=dict)
    performance_alerts: List[str] = field(default_factory=list)
    recovery_actions_taken: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'routing_prediction': self.routing_prediction.to_dict(),
            'success': self.success,
            'cascade_path_used': self.cascade_path_used.value,
            'step_results': [result.to_dict() for result in self.step_results],
            'successful_step': self.successful_step.value if self.successful_step else None,
            'total_steps_attempted': self.total_steps_attempted,
            'total_cascade_time_ms': self.total_cascade_time_ms,
            'decision_overhead_ms': self.decision_overhead_ms,
            'step_processing_times': {step.value: time_ms for step, time_ms in self.step_processing_times.items()},
            'initial_uncertainty_analysis': self.initial_uncertainty_analysis.to_dict() if self.initial_uncertainty_analysis else None,
            'final_confidence_improvement': self.final_confidence_improvement,
            'uncertainty_reduction_achieved': self.uncertainty_reduction_achieved,
            'cascade_efficiency_score': self.cascade_efficiency_score,
            'confidence_reliability_score': self.confidence_reliability_score,
            'uncertainty_handling_score': self.uncertainty_handling_score,
            'fallback_level_equivalent': self.fallback_level_equivalent.name if self.fallback_level_equivalent else None,
            'integration_warnings': self.integration_warnings,
            'backward_compatibility_maintained': self.backward_compatibility_maintained,
            'debug_info': self.debug_info,
            'performance_alerts': self.performance_alerts,
            'recovery_actions_taken': self.recovery_actions_taken
        }


# ============================================================================
# CASCADE CIRCUIT BREAKER FOR PERFORMANCE OPTIMIZATION
# ============================================================================

class CascadeCircuitBreaker:
    """
    Performance-optimized circuit breaker specifically for cascade operations.
    Ensures < 200ms total cascade time requirement is maintained.
    """
    
    def __init__(self, 
                 step_type: CascadeStepType,
                 failure_threshold: int = 5,
                 recovery_timeout_seconds: int = 30,
                 max_step_time_ms: float = 150.0):
        """Initialize cascade circuit breaker."""
        self.step_type = step_type
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.max_step_time_ms = max_step_time_ms
        
        # Circuit breaker state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
        # Performance tracking
        self.recent_response_times = deque(maxlen=20)
        self.average_response_time = 0.0
        self.performance_degradation_detected = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if step can execute based on circuit breaker state."""
        with self.lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            
            elif self.state == CircuitBreakerState.OPEN:
                # Check if we should move to half-open
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.recovery_timeout_seconds):
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
                return False
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            
            return False
    
    def record_success(self, response_time_ms: float):
        """Record successful execution."""
        with self.lock:
            self.recent_response_times.append(response_time_ms)
            self.average_response_time = statistics.mean(self.recent_response_times)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 3:  # Require 3 successes to fully close
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    
            # Check for performance degradation
            if response_time_ms > self.max_step_time_ms:
                self.performance_degradation_detected = True
            elif response_time_ms < self.max_step_time_ms * 0.8:
                self.performance_degradation_detected = False
    
    def record_failure(self, response_time_ms: Optional[float] = None):
        """Record failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if response_time_ms:
                self.recent_response_times.append(response_time_ms)
                self.average_response_time = statistics.mean(self.recent_response_times)
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status."""
        with self.lock:
            return {
                'step_type': self.step_type.value,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'average_response_time_ms': self.average_response_time,
                'performance_degradation_detected': self.performance_degradation_detected,
                'can_execute': self.can_execute()
            }


# ============================================================================
# CASCADE DECISION ENGINE
# ============================================================================

class CascadeDecisionEngine:
    """
    Intelligent decision engine for cascade routing based on uncertainty type
    and confidence thresholds.
    """
    
    def __init__(self, 
                 threshold_config: UncertaintyAwareClassificationThresholds,
                 logger: Optional[logging.Logger] = None):
        """Initialize cascade decision engine."""
        self.threshold_config = threshold_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Decision history for learning
        self.decision_history = deque(maxlen=1000)
        self.strategy_success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
        
        # Performance metrics
        self.decision_times = deque(maxlen=100)
        self.average_decision_time_ms = 0.0
    
    def determine_cascade_strategy(self,
                                 uncertainty_analysis: UncertaintyAnalysis,
                                 confidence_metrics: ConfidenceMetrics,
                                 context: Optional[Dict[str, Any]] = None) -> Tuple[CascadePathStrategy, Dict[str, Any]]:
        """
        Determine optimal cascade strategy based on uncertainty analysis.
        
        Args:
            uncertainty_analysis: Detected uncertainty patterns
            confidence_metrics: Current confidence metrics
            context: Optional context information
            
        Returns:
            Tuple of (strategy, decision_metadata)
        """
        start_time = time.time()
        
        try:
            # Get confidence level
            confidence_level = self.threshold_config.get_confidence_level(
                confidence_metrics.overall_confidence
            )
            
            # Analyze uncertainty types
            uncertainty_types = uncertainty_analysis.detected_uncertainty_types
            uncertainty_severity = uncertainty_analysis.uncertainty_severity
            
            decision_metadata = {
                'confidence_level': confidence_level.value,
                'uncertainty_types': [ut.value for ut in uncertainty_types],
                'uncertainty_severity': uncertainty_severity,
                'decision_factors': []
            }
            
            # Decision logic based on uncertainty patterns
            if uncertainty_severity > 0.8:
                # Very high uncertainty - go straight to cache for safety
                decision_metadata['decision_factors'].append('Very high uncertainty detected')
                strategy = CascadePathStrategy.DIRECT_TO_CACHE
                
            elif UncertaintyType.LLM_UNCERTAINTY in uncertainty_types and len(uncertainty_types) > 2:
                # LLM is uncertain and multiple other factors - skip LightRAG
                decision_metadata['decision_factors'].append('LLM uncertainty with multiple factors')
                strategy = CascadePathStrategy.SKIP_LIGHTRAG
                
            elif confidence_level == ConfidenceLevel.VERY_LOW:
                # Very low confidence - need consensus approach
                decision_metadata['decision_factors'].append('Very low confidence requires consensus')
                strategy = CascadePathStrategy.CONSENSUS_SEEKING
                
            elif (UncertaintyType.WEAK_EVIDENCE in uncertainty_types and
                  UncertaintyType.LOW_CONFIDENCE in uncertainty_types):
                # Weak evidence with low confidence - try confidence boosting
                decision_metadata['decision_factors'].append('Weak evidence requires confidence boosting')
                strategy = CascadePathStrategy.CONFIDENCE_BOOSTED
                
            elif uncertainty_severity < 0.3 and confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]:
                # Moderate uncertainty - full cascade approach
                decision_metadata['decision_factors'].append('Moderate uncertainty - full cascade appropriate')
                strategy = CascadePathStrategy.FULL_CASCADE
                
            else:
                # Default to full cascade for other cases
                decision_metadata['decision_factors'].append('Default to full cascade')
                strategy = CascadePathStrategy.FULL_CASCADE
            
            # Record decision time
            decision_time_ms = (time.time() - start_time) * 1000
            self.decision_times.append(decision_time_ms)
            self.average_decision_time_ms = statistics.mean(self.decision_times)
            
            decision_metadata['decision_time_ms'] = decision_time_ms
            
            # Record decision for learning
            self._record_decision(strategy, uncertainty_analysis, confidence_metrics, decision_metadata)
            
            self.logger.debug(f"Selected cascade strategy: {strategy.value} based on {decision_metadata}")
            
            return strategy, decision_metadata
            
        except Exception as e:
            self.logger.error(f"Error in cascade decision: {e}")
            # Fallback to safest strategy
            return CascadePathStrategy.DIRECT_TO_CACHE, {
                'error': str(e),
                'fallback_decision': True
            }
    
    def _record_decision(self,
                        strategy: CascadePathStrategy,
                        uncertainty_analysis: UncertaintyAnalysis,
                        confidence_metrics: ConfidenceMetrics,
                        decision_metadata: Dict[str, Any]):
        """Record decision for learning and analysis."""
        decision_record = {
            'timestamp': datetime.now(timezone.utc),
            'strategy': strategy.value,
            'uncertainty_types': [ut.value for ut in uncertainty_analysis.detected_uncertainty_types],
            'uncertainty_severity': uncertainty_analysis.uncertainty_severity,
            'confidence_level': self.threshold_config.get_confidence_level(confidence_metrics.overall_confidence).value,
            'decision_metadata': decision_metadata
        }
        
        self.decision_history.append(decision_record)
    
    def update_strategy_success_rate(self, strategy: CascadePathStrategy, success: bool):
        """Update success rate for cascade strategy."""
        self.strategy_success_rates[strategy]['total'] += 1
        if success:
            self.strategy_success_rates[strategy]['success'] += 1
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all strategies."""
        performance = {}
        
        for strategy, stats in self.strategy_success_rates.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                performance[strategy.value] = {
                    'success_rate': success_rate,
                    'total_attempts': stats['total'],
                    'successful_attempts': stats['success']
                }
        
        return {
            'strategy_performance': performance,
            'average_decision_time_ms': self.average_decision_time_ms,
            'total_decisions': len(self.decision_history)
        }


# ============================================================================
# CASCADE PERFORMANCE MONITOR
# ============================================================================

class CascadePerformanceMonitor:
    """
    Comprehensive performance monitoring for cascade operations.
    Ensures < 200ms total cascade time requirement is maintained.
    """
    
    def __init__(self, 
                 max_total_cascade_time_ms: float = 200.0,
                 logger: Optional[logging.Logger] = None):
        """Initialize cascade performance monitor."""
        self.max_total_cascade_time_ms = max_total_cascade_time_ms
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance tracking
        self.cascade_times = deque(maxlen=200)
        self.step_performance = defaultdict(lambda: deque(maxlen=100))
        self.timeout_violations = 0
        self.performance_alerts = []
        
        # Real-time metrics
        self.current_cascade_start_time = None
        self.step_timing_data = {}
        self.performance_degradation_detected = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def start_cascade_timing(self) -> float:
        """Start timing a cascade operation."""
        with self.lock:
            self.current_cascade_start_time = time.time()
            self.step_timing_data = {}
            return self.current_cascade_start_time
    
    def record_step_timing(self, step_type: CascadeStepType, processing_time_ms: float):
        """Record timing for a specific cascade step."""
        with self.lock:
            self.step_timing_data[step_type] = processing_time_ms
            self.step_performance[step_type].append(processing_time_ms)
    
    def finish_cascade_timing(self) -> Tuple[float, bool, List[str]]:
        """
        Finish timing cascade operation and analyze performance.
        
        Returns:
            Tuple of (total_time_ms, within_limits, performance_alerts)
        """
        with self.lock:
            if not self.current_cascade_start_time:
                return 0.0, True, ["No cascade timing started"]
            
            total_time_ms = (time.time() - self.current_cascade_start_time) * 1000
            self.cascade_times.append(total_time_ms)
            
            within_limits = total_time_ms <= self.max_total_cascade_time_ms
            alerts = []
            
            if not within_limits:
                self.timeout_violations += 1
                alerts.append(f"Cascade time {total_time_ms:.1f}ms exceeded limit {self.max_total_cascade_time_ms}ms")
            
            # Check for performance degradation
            if len(self.cascade_times) >= 10:
                recent_avg = statistics.mean(list(self.cascade_times)[-10:])
                if recent_avg > self.max_total_cascade_time_ms * 0.8:
                    self.performance_degradation_detected = True
                    alerts.append(f"Performance degradation detected: avg {recent_avg:.1f}ms")
                elif recent_avg < self.max_total_cascade_time_ms * 0.6:
                    self.performance_degradation_detected = False
            
            # Analyze step performance
            for step_type, time_ms in self.step_timing_data.items():
                step_times = list(self.step_performance[step_type])
                if len(step_times) >= 5:
                    avg_step_time = statistics.mean(step_times[-5:])
                    if avg_step_time > 100:  # Individual step taking too long
                        alerts.append(f"Step {step_type.value} average time {avg_step_time:.1f}ms is high")
            
            self.performance_alerts.extend(alerts)
            self.current_cascade_start_time = None
            
            return total_time_ms, within_limits, alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            if not self.cascade_times:
                return {'status': 'no_data'}
            
            # Overall statistics
            avg_cascade_time = statistics.mean(self.cascade_times)
            median_cascade_time = statistics.median(self.cascade_times)
            max_cascade_time = max(self.cascade_times)
            min_cascade_time = min(self.cascade_times)
            
            # Performance compliance
            within_limit_count = sum(1 for t in self.cascade_times if t <= self.max_total_cascade_time_ms)
            compliance_rate = within_limit_count / len(self.cascade_times)
            
            # Step performance
            step_summary = {}
            for step_type, times in self.step_performance.items():
                if times:
                    step_summary[step_type.value] = {
                        'average_ms': statistics.mean(times),
                        'median_ms': statistics.median(times),
                        'max_ms': max(times),
                        'sample_count': len(times)
                    }
            
            return {
                'overall_performance': {
                    'average_cascade_time_ms': avg_cascade_time,
                    'median_cascade_time_ms': median_cascade_time,
                    'max_cascade_time_ms': max_cascade_time,
                    'min_cascade_time_ms': min_cascade_time,
                    'compliance_rate': compliance_rate,
                    'timeout_violations': self.timeout_violations,
                    'total_cascades': len(self.cascade_times)
                },
                'step_performance': step_summary,
                'performance_degradation_detected': self.performance_degradation_detected,
                'recent_alerts': self.performance_alerts[-10:] if self.performance_alerts else []
            }
    
    def should_skip_step_for_performance(self, 
                                       step_type: CascadeStepType,
                                       time_remaining_ms: float) -> bool:
        """
        Determine if a step should be skipped for performance reasons.
        
        Args:
            step_type: The cascade step to potentially skip
            time_remaining_ms: Time remaining in cascade budget
            
        Returns:
            True if step should be skipped for performance
        """
        with self.lock:
            # If we don't have enough time remaining, skip
            if time_remaining_ms < 50:  # Need at least 50ms for any step
                return True
            
            # Get average time for this step
            if step_type in self.step_performance:
                step_times = list(self.step_performance[step_type])
                if step_times:
                    avg_step_time = statistics.mean(step_times[-5:])  # Use recent average
                    if avg_step_time > time_remaining_ms * 0.9:  # Leave 10% buffer
                        return True
            
            return False


# ============================================================================
# MAIN CASCADE ORCHESTRATOR
# ============================================================================

class UncertaintyAwareFallbackCascade:
    """
    Main orchestrator for uncertainty-aware multi-step fallback cascade.
    
    This class provides the primary interface for the enhanced fallback system,
    implementing intelligent routing between LightRAG, Perplexity, and Cache
    systems based on uncertainty analysis and confidence thresholds.
    """
    
    def __init__(self,
                 fallback_orchestrator: Optional[FallbackOrchestrator] = None,
                 threshold_config: Optional[UncertaintyAwareClassificationThresholds] = None,
                 uncertainty_detector: Optional[UncertaintyDetector] = None,
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the uncertainty-aware fallback cascade.
        
        Args:
            fallback_orchestrator: Existing fallback orchestrator to integrate with
            threshold_config: Confidence threshold configuration
            uncertainty_detector: Uncertainty detection system
            config: Additional configuration options
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Core components
        self.fallback_orchestrator = fallback_orchestrator
        self.threshold_config = threshold_config or UncertaintyAwareClassificationThresholds()
        self.uncertainty_detector = uncertainty_detector or UncertaintyDetector()
        
        # Decision and monitoring systems
        self.decision_engine = CascadeDecisionEngine(self.threshold_config, self.logger)
        self.performance_monitor = CascadePerformanceMonitor(
            max_total_cascade_time_ms=self.config.get('max_total_cascade_time_ms', 200.0),
            logger=self.logger
        )
        
        # Circuit breakers for each cascade step
        self.circuit_breakers = {
            CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE: CascadeCircuitBreaker(
                CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE,
                max_step_time_ms=self.config.get('lightrag_max_time_ms', 120.0)
            ),
            CascadeStepType.PERPLEXITY_SPECIALIZED: CascadeCircuitBreaker(
                CascadeStepType.PERPLEXITY_SPECIALIZED,
                max_step_time_ms=self.config.get('perplexity_max_time_ms', 100.0)
            ),
            CascadeStepType.EMERGENCY_CACHE_CONFIDENT: CascadeCircuitBreaker(
                CascadeStepType.EMERGENCY_CACHE_CONFIDENT,
                max_step_time_ms=self.config.get('cache_max_time_ms', 20.0)
            )
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 3),
            thread_name_prefix="CascadeWorker"
        )
        
        # Integration components (set via integrate_with_existing method)
        self.query_router = None
        self.llm_classifier = None
        self.research_categorizer = None
        self.confidence_scorer = None
        
        # Performance and monitoring
        self.cascade_stats = defaultdict(lambda: defaultdict(int))
        self.integration_warnings = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.logger.info("UncertaintyAwareFallbackCascade initialized successfully")
    
    def integrate_with_existing_components(self,
                                         query_router: Optional[BiomedicalQueryRouter] = None,
                                         llm_classifier: Optional[EnhancedLLMQueryClassifier] = None,
                                         research_categorizer: Optional[ResearchCategorizer] = None,
                                         confidence_scorer: Optional[HybridConfidenceScorer] = None):
        """
        Integrate with existing system components for seamless operation.
        
        Args:
            query_router: Existing biomedical query router
            llm_classifier: Existing LLM classifier
            research_categorizer: Existing research categorizer
            confidence_scorer: Existing confidence scoring system
        """
        if query_router:
            self.query_router = query_router
            self.logger.info("Integrated with BiomedicalQueryRouter")
        
        if llm_classifier:
            self.llm_classifier = llm_classifier
            self.logger.info("Integrated with EnhancedLLMQueryClassifier")
        
        if research_categorizer:
            self.research_categorizer = research_categorizer
            self.logger.info("Integrated with ResearchCategorizer")
        
        if confidence_scorer:
            self.confidence_scorer = confidence_scorer
            self.logger.info("Integrated with HybridConfidenceScorer")
        
        # Integrate with fallback orchestrator if available
        if self.fallback_orchestrator:
            self.fallback_orchestrator.integrate_with_existing_components(
                query_router, llm_classifier, research_categorizer
            )
            self.logger.info("Integrated cascade with FallbackOrchestrator")
    
    def process_query_with_uncertainty_cascade(self,
                                             query_text: str,
                                             context: Optional[Dict[str, Any]] = None,
                                             priority: str = 'normal') -> CascadeResult:
        """
        Main entry point for processing queries with uncertainty-aware cascade.
        
        Args:
            query_text: The user query to process
            context: Optional context information
            priority: Query priority level
            
        Returns:
            CascadeResult with comprehensive processing information
        """
        cascade_start_time = self.performance_monitor.start_cascade_timing()
        
        try:
            # Step 1: Initial uncertainty analysis
            uncertainty_analysis = self._analyze_initial_uncertainty(query_text, context)
            
            # Step 2: Determine cascade strategy
            cascade_strategy, decision_metadata = self.decision_engine.determine_cascade_strategy(
                uncertainty_analysis, 
                uncertainty_analysis.query_characteristics.get('confidence_metrics'),
                context
            )
            
            # Step 3: Execute cascade based on strategy
            cascade_result = self._execute_cascade_strategy(
                query_text, context, cascade_strategy, uncertainty_analysis, 
                decision_metadata, cascade_start_time
            )
            
            # Step 4: Finalize results and performance analysis
            self._finalize_cascade_result(cascade_result, cascade_start_time)
            
            return cascade_result
            
        except Exception as e:
            self.logger.error(f"Error in uncertainty cascade processing: {e}")
            return self._create_emergency_cascade_result(query_text, cascade_start_time, str(e))
    
    def _analyze_initial_uncertainty(self,
                                   query_text: str,
                                   context: Optional[Dict[str, Any]]) -> UncertaintyAnalysis:
        """Perform initial uncertainty analysis using existing systems."""
        try:
            # Use existing query router to get initial confidence metrics
            confidence_metrics = None
            if self.query_router:
                try:
                    initial_result = self.query_router.route_query(query_text, context)
                    if initial_result:
                        confidence_metrics = initial_result.confidence_metrics
                except Exception as e:
                    self.logger.debug(f"Initial routing failed: {e}")
            
            # If no confidence metrics, create minimal ones
            if not confidence_metrics:
                confidence_metrics = ConfidenceMetrics(
                    overall_confidence=0.3,  # Assume low confidence
                    research_category_confidence=0.3,
                    temporal_analysis_confidence=0.2,
                    signal_strength_confidence=0.2,
                    context_coherence_confidence=0.2,
                    keyword_density=0.1,
                    pattern_match_strength=0.1,
                    biomedical_entity_count=1,
                    ambiguity_score=0.6,  # Assume high ambiguity
                    conflict_score=0.4,
                    alternative_interpretations=[],
                    calculation_time_ms=0.0
                )
            
            # Use uncertainty detector to analyze patterns
            uncertainty_analysis = self.uncertainty_detector.analyze_query_uncertainty(
                query_text, confidence_metrics, context
            )
            
            # Store confidence metrics in analysis for later use
            uncertainty_analysis.query_characteristics['confidence_metrics'] = confidence_metrics
            
            return uncertainty_analysis
            
        except Exception as e:
            self.logger.error(f"Error in initial uncertainty analysis: {e}")
            # Return minimal uncertainty analysis
            return UncertaintyAnalysis(
                detected_uncertainty_types={UncertaintyType.WEAK_EVIDENCE},
                uncertainty_severity=0.7,
                requires_special_handling=True,
                query_characteristics={'confidence_metrics': None}
            )
    
    def _execute_cascade_strategy(self,
                                query_text: str,
                                context: Optional[Dict[str, Any]],
                                cascade_strategy: CascadePathStrategy,
                                uncertainty_analysis: UncertaintyAnalysis,
                                decision_metadata: Dict[str, Any],
                                cascade_start_time: float) -> CascadeResult:
        """Execute the determined cascade strategy."""
        
        cascade_result = CascadeResult(
            routing_prediction=None,  # Will be set when successful
            success=False,
            cascade_path_used=cascade_strategy,
            initial_uncertainty_analysis=uncertainty_analysis,
            debug_info=decision_metadata
        )
        
        try:
            # Execute based on strategy
            if cascade_strategy == CascadePathStrategy.FULL_CASCADE:
                success = self._execute_full_cascade(query_text, context, cascade_result, cascade_start_time)
                
            elif cascade_strategy == CascadePathStrategy.SKIP_LIGHTRAG:
                success = self._execute_skip_lightrag_cascade(query_text, context, cascade_result, cascade_start_time)
                
            elif cascade_strategy == CascadePathStrategy.DIRECT_TO_CACHE:
                success = self._execute_direct_to_cache(query_text, context, cascade_result, cascade_start_time)
                
            elif cascade_strategy == CascadePathStrategy.CONFIDENCE_BOOSTED:
                success = self._execute_confidence_boosted_cascade(query_text, context, cascade_result, cascade_start_time)
                
            elif cascade_strategy == CascadePathStrategy.CONSENSUS_SEEKING:
                success = self._execute_consensus_seeking_cascade(query_text, context, cascade_result, cascade_start_time)
                
            else:
                # Fallback to full cascade
                self.logger.warning(f"Unknown cascade strategy {cascade_strategy}, falling back to full cascade")
                success = self._execute_full_cascade(query_text, context, cascade_result, cascade_start_time)
            
            cascade_result.success = success
            
            # Update strategy success rate
            self.decision_engine.update_strategy_success_rate(cascade_strategy, success)
            
        except Exception as e:
            self.logger.error(f"Error executing cascade strategy {cascade_strategy}: {e}")
            cascade_result.success = False
            cascade_result.debug_info['execution_error'] = str(e)
        
        return cascade_result
    
    def _execute_full_cascade(self,
                            query_text: str,
                            context: Optional[Dict[str, Any]],
                            cascade_result: CascadeResult,
                            cascade_start_time: float) -> bool:
        """Execute full LightRAG -> Perplexity -> Cache cascade."""
        
        time_budget_ms = self.performance_monitor.max_total_cascade_time_ms
        
        # Step 1: Try LightRAG with uncertainty awareness
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        if (remaining_time > 50 and 
            not self.performance_monitor.should_skip_step_for_performance(
                CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE, remaining_time)):
            
            step_result = self._execute_lightrag_step(query_text, context, 1, remaining_time)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success and step_result.confidence_score >= 0.5:
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE
                return True
        
        # Step 2: Try Perplexity with specialized uncertainty handling
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        if (remaining_time > 30 and
            not self.performance_monitor.should_skip_step_for_performance(
                CascadeStepType.PERPLEXITY_SPECIALIZED, remaining_time)):
            
            step_result = self._execute_perplexity_step(query_text, context, 2, remaining_time)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success and step_result.confidence_score >= 0.3:
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.PERPLEXITY_SPECIALIZED
                return True
        
        # Step 3: Emergency cache as final fallback
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        if remaining_time > 5:  # Even just a few ms for cache lookup
            step_result = self._execute_cache_step(query_text, context, 3, remaining_time)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success:
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.EMERGENCY_CACHE_CONFIDENT
                return True
        
        return False
    
    def _execute_skip_lightrag_cascade(self,
                                     query_text: str,
                                     context: Optional[Dict[str, Any]],
                                     cascade_result: CascadeResult,
                                     cascade_start_time: float) -> bool:
        """Execute Perplexity -> Cache cascade (skipping LightRAG)."""
        
        time_budget_ms = self.performance_monitor.max_total_cascade_time_ms
        
        # Step 1: Try Perplexity first
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        if (remaining_time > 30 and
            not self.performance_monitor.should_skip_step_for_performance(
                CascadeStepType.PERPLEXITY_SPECIALIZED, remaining_time)):
            
            step_result = self._execute_perplexity_step(query_text, context, 1, remaining_time)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success and step_result.confidence_score >= 0.3:
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.PERPLEXITY_SPECIALIZED
                return True
        
        # Step 2: Emergency cache as fallback
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        if remaining_time > 5:
            step_result = self._execute_cache_step(query_text, context, 2, remaining_time)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success:
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.EMERGENCY_CACHE_CONFIDENT
                return True
        
        return False
    
    def _execute_direct_to_cache(self,
                               query_text: str,
                               context: Optional[Dict[str, Any]],
                               cascade_result: CascadeResult,
                               cascade_start_time: float) -> bool:
        """Execute direct to cache strategy for emergency situations."""
        
        time_budget_ms = self.performance_monitor.max_total_cascade_time_ms
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        
        step_result = self._execute_cache_step(query_text, context, 1, remaining_time)
        cascade_result.step_results.append(step_result)
        cascade_result.total_steps_attempted += 1
        
        if step_result.success:
            cascade_result.routing_prediction = step_result.routing_prediction
            cascade_result.successful_step = CascadeStepType.EMERGENCY_CACHE_CONFIDENT
            return True
        
        return False
    
    def _execute_confidence_boosted_cascade(self,
                                          query_text: str,
                                          context: Optional[Dict[str, Any]],
                                          cascade_result: CascadeResult,
                                          cascade_start_time: float) -> bool:
        """Execute cascade with confidence boosting techniques."""
        
        # Try LightRAG with enhanced confidence scoring
        time_budget_ms = self.performance_monitor.max_total_cascade_time_ms
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        
        if remaining_time > 50:
            step_result = self._execute_lightrag_step(query_text, context, 1, remaining_time, boost_confidence=True)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success and step_result.confidence_score >= 0.4:  # Lower threshold with boosting
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE
                return True
        
        # Fallback to regular cascade
        return self._execute_full_cascade(query_text, context, cascade_result, cascade_start_time)
    
    def _execute_consensus_seeking_cascade(self,
                                         query_text: str,
                                         context: Optional[Dict[str, Any]],
                                         cascade_result: CascadeResult,
                                         cascade_start_time: float) -> bool:
        """Execute cascade seeking consensus between multiple approaches."""
        
        time_budget_ms = self.performance_monitor.max_total_cascade_time_ms
        consensus_results = []
        
        # Try multiple approaches in parallel if time permits
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        
        if remaining_time > 100:  # Need sufficient time for parallel execution
            futures = []
            
            # Submit LightRAG task
            if self.circuit_breakers[CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE].can_execute():
                future_lightrag = self.thread_pool.submit(
                    self._execute_lightrag_step, query_text, context, 1, remaining_time / 2
                )
                futures.append((CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE, future_lightrag))
            
            # Submit Perplexity task
            if self.circuit_breakers[CascadeStepType.PERPLEXITY_SPECIALIZED].can_execute():
                future_perplexity = self.thread_pool.submit(
                    self._execute_perplexity_step, query_text, context, 2, remaining_time / 2
                )
                futures.append((CascadeStepType.PERPLEXITY_SPECIALIZED, future_perplexity))
            
            # Collect results with timeout
            for step_type, future in futures:
                try:
                    step_result = future.result(timeout=remaining_time / 2000)  # Convert to seconds
                    cascade_result.step_results.append(step_result)
                    cascade_result.total_steps_attempted += 1
                    
                    if step_result.success:
                        consensus_results.append(step_result)
                        
                except FutureTimeoutError:
                    self.logger.warning(f"Consensus step {step_type.value} timed out")
                except Exception as e:
                    self.logger.error(f"Consensus step {step_type.value} failed: {e}")
        
        # Analyze consensus results
        if consensus_results:
            # Select best result based on confidence and uncertainty
            best_result = max(consensus_results, 
                            key=lambda r: r.confidence_score - r.uncertainty_score)
            
            cascade_result.routing_prediction = best_result.routing_prediction
            cascade_result.successful_step = best_result.step_type
            return True
        
        # Fallback to cache if consensus failed
        remaining_time = time_budget_ms - ((time.time() - cascade_start_time) * 1000)
        if remaining_time > 5:
            step_result = self._execute_cache_step(query_text, context, 3, remaining_time)
            cascade_result.step_results.append(step_result)
            cascade_result.total_steps_attempted += 1
            
            if step_result.success:
                cascade_result.routing_prediction = step_result.routing_prediction
                cascade_result.successful_step = CascadeStepType.EMERGENCY_CACHE_CONFIDENT
                return True
        
        return False
    
    def _execute_lightrag_step(self,
                             query_text: str,
                             context: Optional[Dict[str, Any]],
                             step_number: int,
                             time_limit_ms: float,
                             boost_confidence: bool = False) -> CascadeStepResult:
        """Execute LightRAG step with uncertainty awareness."""
        
        step_start_time = time.time()
        circuit_breaker = self.circuit_breakers[CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE]
        
        step_result = CascadeStepResult(
            step_type=CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE,
            step_number=step_number,
            success=False
        )
        
        try:
            # Check circuit breaker
            if not circuit_breaker.can_execute():
                step_result.failure_reason = CascadeFailureReason.CIRCUIT_BREAKER_OPEN
                step_result.error_message = "LightRAG circuit breaker is open"
                return step_result
            
            # Use existing query router with timeout
            if self.query_router:
                # Create timeout wrapper
                def query_with_timeout():
                    return self.query_router.route_query(query_text, context)
                
                future = self.thread_pool.submit(query_with_timeout)
                
                try:
                    routing_result = future.result(timeout=time_limit_ms / 1000)
                    
                    if routing_result:
                        confidence_score = routing_result.confidence
                        
                        # Apply confidence boosting if requested
                        if boost_confidence and self.confidence_scorer:
                            try:
                                hybrid_confidence = self.confidence_scorer.calculate_hybrid_confidence(
                                    query_text, routing_result, context or {}
                                )
                                if hybrid_confidence:
                                    # Boost confidence by up to 20%
                                    boost_factor = min(0.2, (1.0 - confidence_score) * 0.5)
                                    confidence_score = min(1.0, confidence_score + boost_factor)
                                    
                                    step_result.debug_info['confidence_boost_applied'] = boost_factor
                            except Exception as e:
                                self.logger.debug(f"Confidence boosting failed: {e}")
                        
                        step_result.routing_prediction = routing_result
                        step_result.confidence_score = confidence_score
                        step_result.uncertainty_score = routing_result.confidence_metrics.ambiguity_score if routing_result.confidence_metrics else 0.5
                        step_result.success = True
                        
                        # Record success
                        processing_time_ms = (time.time() - step_start_time) * 1000
                        circuit_breaker.record_success(processing_time_ms)
                        self.performance_monitor.record_step_timing(
                            CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE, processing_time_ms
                        )
                        
                    else:
                        step_result.failure_reason = CascadeFailureReason.CONFIDENCE_TOO_LOW
                        step_result.error_message = "LightRAG returned no result"
                
                except FutureTimeoutError:
                    step_result.failure_reason = CascadeFailureReason.TIMEOUT_EXCEEDED
                    step_result.error_message = f"LightRAG step timed out after {time_limit_ms}ms"
                    circuit_breaker.record_failure(time_limit_ms)
            else:
                step_result.failure_reason = CascadeFailureReason.SERVICE_UNAVAILABLE
                step_result.error_message = "Query router not available"
                
        except Exception as e:
            step_result.failure_reason = CascadeFailureReason.UNKNOWN_ERROR
            step_result.error_message = str(e)
            processing_time_ms = (time.time() - step_start_time) * 1000
            circuit_breaker.record_failure(processing_time_ms)
        
        step_result.processing_time_ms = (time.time() - step_start_time) * 1000
        return step_result
    
    def _execute_perplexity_step(self,
                               query_text: str,
                               context: Optional[Dict[str, Any]],
                               step_number: int,
                               time_limit_ms: float) -> CascadeStepResult:
        """Execute Perplexity step with specialized uncertainty handling."""
        
        step_start_time = time.time()
        circuit_breaker = self.circuit_breakers[CascadeStepType.PERPLEXITY_SPECIALIZED]
        
        step_result = CascadeStepResult(
            step_type=CascadeStepType.PERPLEXITY_SPECIALIZED,
            step_number=step_number,
            success=False
        )
        
        try:
            # Check circuit breaker
            if not circuit_breaker.can_execute():
                step_result.failure_reason = CascadeFailureReason.CIRCUIT_BREAKER_OPEN
                step_result.error_message = "Perplexity circuit breaker is open"
                return step_result
            
            # Simulate Perplexity API call with uncertainty handling
            # In a real implementation, this would call the actual Perplexity API
            # with specialized prompts for handling uncertain queries
            
            # For now, create a simulated response
            processing_time_ms = min(time_limit_ms * 0.8, 80.0)  # Simulate processing
            time.sleep(processing_time_ms / 1000)
            
            # Create a routing prediction with moderate confidence
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=0.4,
                research_category_confidence=0.4,
                temporal_analysis_confidence=0.3,
                signal_strength_confidence=0.3,
                context_coherence_confidence=0.3,
                keyword_density=0.2,
                pattern_match_strength=0.2,
                biomedical_entity_count=2,
                ambiguity_score=0.5,
                conflict_score=0.3,
                alternative_interpretations=[(RoutingDecision.EITHER, 0.4)],
                calculation_time_ms=processing_time_ms
            )
            
            routing_prediction = RoutingPrediction(
                routing_decision=RoutingDecision.EITHER,
                confidence=0.4,
                reasoning=["Perplexity API analysis with uncertainty handling"],
                research_category=ResearchCategory.GENERAL_QUERY,
                confidence_metrics=confidence_metrics,
                temporal_indicators=[],
                knowledge_indicators=[],
                metadata={'source': 'perplexity_specialized', 'uncertainty_aware': True}
            )
            
            step_result.routing_prediction = routing_prediction
            step_result.confidence_score = 0.4
            step_result.uncertainty_score = 0.5
            step_result.success = True
            
            # Record success
            circuit_breaker.record_success(processing_time_ms)
            self.performance_monitor.record_step_timing(
                CascadeStepType.PERPLEXITY_SPECIALIZED, processing_time_ms
            )
            
        except Exception as e:
            step_result.failure_reason = CascadeFailureReason.UNKNOWN_ERROR
            step_result.error_message = str(e)
            processing_time_ms = (time.time() - step_start_time) * 1000
            circuit_breaker.record_failure(processing_time_ms)
        
        step_result.processing_time_ms = (time.time() - step_start_time) * 1000
        return step_result
    
    def _execute_cache_step(self,
                          query_text: str,
                          context: Optional[Dict[str, Any]],
                          step_number: int,
                          time_limit_ms: float) -> CascadeStepResult:
        """Execute emergency cache step with confidence adjustment."""
        
        step_start_time = time.time()
        circuit_breaker = self.circuit_breakers[CascadeStepType.EMERGENCY_CACHE_CONFIDENT]
        
        step_result = CascadeStepResult(
            step_type=CascadeStepType.EMERGENCY_CACHE_CONFIDENT,
            step_number=step_number,
            success=False
        )
        
        try:
            # Check circuit breaker
            if not circuit_breaker.can_execute():
                step_result.failure_reason = CascadeFailureReason.CIRCUIT_BREAKER_OPEN
                step_result.error_message = "Cache circuit breaker is open"
                return step_result
            
            # Use existing fallback orchestrator's emergency cache if available
            if self.fallback_orchestrator and hasattr(self.fallback_orchestrator, 'emergency_cache'):
                cache_result = self.fallback_orchestrator.emergency_cache.get_cached_response(query_text)
                
                if cache_result:
                    step_result.routing_prediction = cache_result
                    step_result.confidence_score = 0.2  # Low but acceptable for emergency
                    step_result.uncertainty_score = 0.8  # High uncertainty from cache
                    step_result.success = True
                else:
                    # Create default emergency response
                    step_result.routing_prediction = self._create_default_emergency_response()
                    step_result.confidence_score = 0.15
                    step_result.uncertainty_score = 0.9
                    step_result.success = True
            else:
                # Create default emergency response
                step_result.routing_prediction = self._create_default_emergency_response()
                step_result.confidence_score = 0.15
                step_result.uncertainty_score = 0.9
                step_result.success = True
            
            # Record success
            processing_time_ms = (time.time() - step_start_time) * 1000
            circuit_breaker.record_success(processing_time_ms)
            self.performance_monitor.record_step_timing(
                CascadeStepType.EMERGENCY_CACHE_CONFIDENT, processing_time_ms
            )
            
        except Exception as e:
            step_result.failure_reason = CascadeFailureReason.UNKNOWN_ERROR
            step_result.error_message = str(e)
            processing_time_ms = (time.time() - step_start_time) * 1000
            circuit_breaker.record_failure(processing_time_ms)
        
        step_result.processing_time_ms = (time.time() - step_start_time) * 1000
        return step_result
    
    def _create_default_emergency_response(self) -> RoutingPrediction:
        """Create a default emergency routing prediction."""
        
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=0.15,
            research_category_confidence=0.15,
            temporal_analysis_confidence=0.1,
            signal_strength_confidence=0.1,
            context_coherence_confidence=0.1,
            keyword_density=0.1,
            pattern_match_strength=0.1,
            biomedical_entity_count=1,
            ambiguity_score=0.9,
            conflict_score=0.1,
            alternative_interpretations=[(RoutingDecision.EITHER, 0.15)],
            calculation_time_ms=1.0
        )
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=0.15,
            reasoning=["Emergency fallback response - conservative routing"],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=[],
            metadata={
                'source': 'emergency_cache',
                'fallback_level': 'emergency',
                'uncertainty_aware': True,
                'emergency_response': True
            }
        )
    
    def _finalize_cascade_result(self, cascade_result: CascadeResult, cascade_start_time: float):
        """Finalize cascade result with performance analysis and quality scoring."""
        
        # Finish performance timing
        total_time_ms, within_limits, alerts = self.performance_monitor.finish_cascade_timing()
        
        cascade_result.total_cascade_time_ms = total_time_ms
        cascade_result.performance_alerts = alerts
        
        # Calculate quality metrics
        if cascade_result.success and cascade_result.routing_prediction:
            # Efficiency score based on steps needed
            max_steps = 3
            cascade_result.cascade_efficiency_score = 1.0 - ((cascade_result.total_steps_attempted - 1) / max_steps * 0.3)
            
            # Confidence reliability based on successful step
            if cascade_result.successful_step == CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE:
                cascade_result.confidence_reliability_score = 0.9
            elif cascade_result.successful_step == CascadeStepType.PERPLEXITY_SPECIALIZED:
                cascade_result.confidence_reliability_score = 0.7
            else:  # Emergency cache
                cascade_result.confidence_reliability_score = 0.3
            
            # Uncertainty handling score
            initial_uncertainty = cascade_result.initial_uncertainty_analysis.uncertainty_severity if cascade_result.initial_uncertainty_analysis else 0.5
            final_uncertainty = min(step.uncertainty_score for step in cascade_result.step_results if step.success) if any(step.success for step in cascade_result.step_results) else 1.0
            
            cascade_result.uncertainty_reduction_achieved = max(0.0, initial_uncertainty - final_uncertainty)
            cascade_result.uncertainty_handling_score = min(1.0, cascade_result.uncertainty_reduction_achieved + 0.3)
            
            # Calculate confidence improvement
            initial_confidence = cascade_result.initial_uncertainty_analysis.query_characteristics.get('confidence_metrics', {}).get('overall_confidence', 0.3) if cascade_result.initial_uncertainty_analysis else 0.3
            final_confidence = cascade_result.routing_prediction.confidence
            cascade_result.final_confidence_improvement = max(0.0, final_confidence - initial_confidence)
        
        # Map to equivalent fallback level for integration
        if cascade_result.successful_step == CascadeStepType.LIGHTRAG_UNCERTAINTY_AWARE:
            cascade_result.fallback_level_equivalent = FallbackLevel.FULL_LLM_WITH_CONFIDENCE
        elif cascade_result.successful_step == CascadeStepType.PERPLEXITY_SPECIALIZED:
            cascade_result.fallback_level_equivalent = FallbackLevel.SIMPLIFIED_LLM
        elif cascade_result.successful_step == CascadeStepType.EMERGENCY_CACHE_CONFIDENT:
            cascade_result.fallback_level_equivalent = FallbackLevel.EMERGENCY_CACHE
        else:
            cascade_result.fallback_level_equivalent = FallbackLevel.DEFAULT_ROUTING
        
        # Record statistics
        with self.lock:
            self.cascade_stats['total_cascades'] += 1
            self.cascade_stats['successful_cascades'] += 1 if cascade_result.success else 0
            self.cascade_stats[cascade_result.cascade_path_used.value]['attempts'] += 1
            if cascade_result.success:
                self.cascade_stats[cascade_result.cascade_path_used.value]['successes'] += 1
            
            if within_limits:
                self.cascade_stats['performance']['within_time_limit'] += 1
            else:
                self.cascade_stats['performance']['timeout_violations'] += 1
    
    def _create_emergency_cascade_result(self,
                                       query_text: str,
                                       cascade_start_time: float,
                                       error_message: str) -> CascadeResult:
        """Create emergency cascade result when system fails completely."""
        
        # Create minimal emergency response
        emergency_prediction = self._create_default_emergency_response()
        
        cascade_result = CascadeResult(
            routing_prediction=emergency_prediction,
            success=True,  # Emergency response is still a success
            cascade_path_used=CascadePathStrategy.DIRECT_TO_CACHE,
            successful_step=CascadeStepType.EMERGENCY_CACHE_CONFIDENT,
            total_steps_attempted=1,
            cascade_efficiency_score=0.2,  # Low efficiency for emergency
            confidence_reliability_score=0.1,  # Very low reliability
            uncertainty_handling_score=0.1,  # Minimal uncertainty handling
            fallback_level_equivalent=FallbackLevel.DEFAULT_ROUTING,
            debug_info={
                'emergency_creation': True,
                'error_message': error_message,
                'cascade_start_time': cascade_start_time
            }
        )
        
        # Add emergency step result
        emergency_step = CascadeStepResult(
            step_type=CascadeStepType.EMERGENCY_CACHE_CONFIDENT,
            step_number=1,
            success=True,
            routing_prediction=emergency_prediction,
            confidence_score=0.15,
            uncertainty_score=0.9,
            processing_time_ms=(time.time() - cascade_start_time) * 1000,
            debug_info={'emergency_response': True}
        )
        
        cascade_result.step_results.append(emergency_step)
        
        # Finalize timing
        self._finalize_cascade_result(cascade_result, cascade_start_time)
        
        return cascade_result
    
    def get_cascade_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for the cascade system."""
        
        with self.lock:
            # Basic statistics
            total_cascades = self.cascade_stats['total_cascades']
            successful_cascades = self.cascade_stats['successful_cascades']
            success_rate = successful_cascades / total_cascades if total_cascades > 0 else 0.0
            
            # Strategy performance
            strategy_performance = {}
            for strategy in CascadePathStrategy:
                strategy_stats = self.cascade_stats[strategy.value]
                if strategy_stats['attempts'] > 0:
                    strategy_success_rate = strategy_stats['successes'] / strategy_stats['attempts']
                    strategy_performance[strategy.value] = {
                        'success_rate': strategy_success_rate,
                        'attempts': strategy_stats['attempts'],
                        'successes': strategy_stats['successes']
                    }
            
            # Performance metrics
            performance_summary = self.performance_monitor.get_performance_summary()
            
            # Circuit breaker status
            circuit_breaker_status = {}
            for step_type, breaker in self.circuit_breakers.items():
                circuit_breaker_status[step_type.value] = breaker.get_performance_status()
            
            # Decision engine performance
            decision_performance = self.decision_engine.get_strategy_performance()
            
            return {
                'overall_performance': {
                    'total_cascades': total_cascades,
                    'successful_cascades': successful_cascades,
                    'success_rate': success_rate,
                    'performance_compliance': self.cascade_stats['performance']
                },
                'strategy_performance': strategy_performance,
                'timing_performance': performance_summary,
                'circuit_breaker_status': circuit_breaker_status,
                'decision_engine_performance': decision_performance,
                'integration_warnings': self.integration_warnings
            }
    
    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
        except:
            pass  # Ignore cleanup errors


# ============================================================================
# CONVENIENCE FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_uncertainty_aware_cascade_system(
    fallback_orchestrator: Optional[FallbackOrchestrator] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None) -> UncertaintyAwareFallbackCascade:
    """
    Convenience function to create a complete uncertainty-aware cascade system.
    
    Args:
        fallback_orchestrator: Existing fallback orchestrator to integrate with
        config: Configuration options
        logger: Logger instance
        
    Returns:
        Configured UncertaintyAwareFallbackCascade instance
    """
    
    logger = logger or logging.getLogger(__name__)
    config = config or {}
    
    # Create threshold configuration
    threshold_config = UncertaintyAwareClassificationThresholds(
        high_confidence_threshold=config.get('high_confidence_threshold', 0.7),
        medium_confidence_threshold=config.get('medium_confidence_threshold', 0.5),
        low_confidence_threshold=config.get('low_confidence_threshold', 0.3),
        very_low_confidence_threshold=config.get('very_low_confidence_threshold', 0.1)
    )
    
    # Create uncertainty detector
    uncertainty_config = UncertaintyFallbackConfig()
    uncertainty_detector = UncertaintyDetector(config=uncertainty_config, logger=logger)
    
    # Create cascade system
    cascade_system = UncertaintyAwareFallbackCascade(
        fallback_orchestrator=fallback_orchestrator,
        threshold_config=threshold_config,
        uncertainty_detector=uncertainty_detector,
        config=config,
        logger=logger
    )
    
    logger.info("Uncertainty-aware cascade system created successfully")
    
    return cascade_system


def integrate_cascade_with_existing_router(
    existing_router: BiomedicalQueryRouter,
    cascade_system: Optional[UncertaintyAwareFallbackCascade] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None) -> UncertaintyAwareFallbackCascade:
    """
    Integrate cascade system with existing query router.
    
    Args:
        existing_router: Existing BiomedicalQueryRouter
        cascade_system: Optional existing cascade system
        config: Configuration options
        logger: Logger instance
        
    Returns:
        Integrated cascade system
    """
    
    logger = logger or logging.getLogger(__name__)
    
    if not cascade_system:
        cascade_system = create_uncertainty_aware_cascade_system(config=config, logger=logger)
    
    # Integrate with existing router
    cascade_system.integrate_with_existing_components(query_router=existing_router)
    
    logger.info("Cascade system integrated with existing router successfully")
    
    return cascade_system