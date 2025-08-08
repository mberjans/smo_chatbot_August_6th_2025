#!/usr/bin/env python3
"""
FeatureFlagManager: Advanced feature flag management for LightRAG integration.

This module provides comprehensive feature flag management for the Clinical Metabolomics 
Oracle LightRAG integration, supporting:

- Percentage-based rollout with hash-based consistent routing
- A/B testing capabilities with user cohort assignment  
- Circuit breaker integration for fallback scenarios
- Performance monitoring and quality metrics collection
- Conditional routing based on query characteristics
- Dynamic flag evaluation with real-time updates
- Integration with existing configuration patterns

Key Features:
- Hash-based consistent user assignment to maintain session consistency
- Gradual rollout with configurable percentage thresholds
- Quality-based routing decisions with fallback mechanisms
- Performance comparison between LightRAG and Perplexity responses
- Circuit breaker protection for unstable integrations
- Comprehensive logging and metrics collection
- Thread-safe flag evaluation with caching

Requirements:
- Compatible with existing LightRAGConfig architecture
- Integration with existing logging and monitoring systems
- Support for runtime flag updates without restart

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from pathlib import Path
import random

from .config import LightRAGConfig, LightRAGConfigError


class UserCohort(Enum):
    """User cohort assignments for A/B testing."""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"
    CONTROL = "control"


class RoutingDecision(Enum):
    """Routing decision outcomes."""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"
    DISABLED = "disabled"
    CIRCUIT_BREAKER = "circuit_breaker"


class RoutingReason(Enum):
    """Reasons for routing decisions."""
    FEATURE_DISABLED = "feature_disabled"
    ROLLOUT_PERCENTAGE = "rollout_percentage"
    USER_COHORT_ASSIGNMENT = "user_cohort_assignment"
    FORCED_COHORT = "forced_cohort"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    QUALITY_THRESHOLD = "quality_threshold"
    CONDITIONAL_RULE = "conditional_rule"
    TIMEOUT_PROTECTION = "timeout_protection"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_text: Optional[str] = None
    query_type: Optional[str] = None
    query_complexity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    decision: RoutingDecision
    reason: RoutingReason
    user_cohort: Optional[UserCohort] = None
    confidence: float = 1.0
    rollout_hash: Optional[str] = None
    circuit_breaker_state: Optional[str] = None
    quality_score: Optional[float] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert routing result to dictionary for logging."""
        return {
            'decision': self.decision.value,
            'reason': self.reason.value,
            'user_cohort': self.user_cohort.value if self.user_cohort else None,
            'confidence': self.confidence,
            'rollout_hash': self.rollout_hash,
            'circuit_breaker_state': self.circuit_breaker_state,
            'quality_score': self.quality_score,
            'processing_time_ms': self.processing_time_ms,
            'metadata': self.metadata,
            'timestamp': self.processing_time_ms
        }


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker functionality."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    recovery_attempts: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.successful_requests) / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.failure_rate


@dataclass
class PerformanceMetrics:
    """Performance metrics for routing decisions."""
    lightrag_response_times: List[float] = field(default_factory=list)
    perplexity_response_times: List[float] = field(default_factory=list)
    lightrag_quality_scores: List[float] = field(default_factory=list)
    perplexity_quality_scores: List[float] = field(default_factory=list)
    lightrag_success_count: int = 0
    perplexity_success_count: int = 0
    lightrag_error_count: int = 0
    perplexity_error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_lightrag_avg_response_time(self) -> float:
        """Get average LightRAG response time."""
        return sum(self.lightrag_response_times) / len(self.lightrag_response_times) if self.lightrag_response_times else 0.0
    
    def get_perplexity_avg_response_time(self) -> float:
        """Get average Perplexity response time."""
        return sum(self.perplexity_response_times) / len(self.perplexity_response_times) if self.perplexity_response_times else 0.0
    
    def get_lightrag_avg_quality(self) -> float:
        """Get average LightRAG quality score."""
        return sum(self.lightrag_quality_scores) / len(self.lightrag_quality_scores) if self.lightrag_quality_scores else 0.0
    
    def get_perplexity_avg_quality(self) -> float:
        """Get average Perplexity quality score."""
        return sum(self.perplexity_quality_scores) / len(self.perplexity_quality_scores) if self.perplexity_quality_scores else 0.0


class FeatureFlagManager:
    """
    Advanced feature flag manager for LightRAG integration.
    
    Provides comprehensive feature flag functionality including percentage-based rollout,
    A/B testing, circuit breaker protection, and performance-based routing decisions.
    
    Key capabilities:
    - Hash-based consistent user assignment for session stability
    - Gradual rollout with configurable percentage thresholds
    - A/B testing with cohort tracking and performance comparison
    - Circuit breaker protection for unstable integrations
    - Quality-based routing with dynamic thresholds
    - Conditional routing based on query characteristics
    - Real-time metrics collection and analysis
    - Thread-safe operations with optimized caching
    """
    
    def __init__(self, config: LightRAGConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the FeatureFlagManager.
        
        Args:
            config: LightRAGConfig instance with feature flag settings
            logger: Optional logger instance for debugging and metrics
        
        Raises:
            LightRAGConfigError: If configuration is invalid
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(config, LightRAGConfig):
            raise ValueError("config must be a LightRAGConfig instance")
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Circuit breaker state tracking
        self.circuit_breaker_state = CircuitBreakerState()
        
        # Performance metrics tracking
        self.performance_metrics = PerformanceMetrics()
        
        # Routing cache for performance optimization
        self._routing_cache: Dict[str, Tuple[RoutingResult, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)  # Cache TTL for routing decisions
        
        # User cohort assignments cache
        self._cohort_cache: Dict[str, UserCohort] = {}
        
        # Conditional routing rules
        self.routing_rules = self._parse_routing_rules(config.lightrag_routing_rules or {})
        
        self.logger.info(f"FeatureFlagManager initialized with rollout: {config.lightrag_rollout_percentage}%")
    
    def _parse_routing_rules(self, rules: Dict[str, Any]) -> Dict[str, Callable[[RoutingContext], bool]]:
        """
        Parse conditional routing rules from configuration.
        
        Args:
            rules: Dictionary of routing rules from configuration
        
        Returns:
            Dict of compiled routing rule functions
        """
        compiled_rules = {}
        
        for rule_name, rule_config in rules.items():
            try:
                if rule_config.get('type') == 'query_length':
                    min_length = rule_config.get('min_length', 0)
                    max_length = rule_config.get('max_length', float('inf'))
                    
                    def length_rule(context: RoutingContext, min_len=min_length, max_len=max_length) -> bool:
                        if not context.query_text:
                            return False
                        query_len = len(context.query_text)
                        return min_len <= query_len <= max_len
                    
                    compiled_rules[rule_name] = length_rule
                
                elif rule_config.get('type') == 'query_complexity':
                    min_complexity = rule_config.get('min_complexity', 0.0)
                    max_complexity = rule_config.get('max_complexity', 1.0)
                    
                    def complexity_rule(context: RoutingContext, min_comp=min_complexity, max_comp=max_complexity) -> bool:
                        if context.query_complexity is None:
                            return False
                        return min_comp <= context.query_complexity <= max_comp
                    
                    compiled_rules[rule_name] = complexity_rule
                
                elif rule_config.get('type') == 'query_type':
                    allowed_types = set(rule_config.get('allowed_types', []))
                    
                    def type_rule(context: RoutingContext, types=allowed_types) -> bool:
                        return context.query_type in types if context.query_type else False
                    
                    compiled_rules[rule_name] = type_rule
                
                self.logger.debug(f"Compiled routing rule: {rule_name}")
            
            except Exception as e:
                self.logger.warning(f"Failed to parse routing rule {rule_name}: {e}")
        
        return compiled_rules
    
    def _calculate_user_hash(self, user_identifier: str) -> str:
        """
        Calculate consistent hash for user assignment.
        
        Args:
            user_identifier: Unique identifier for the user/session
        
        Returns:
            Hexadecimal hash string for consistent assignment
        """
        hash_input = f"{user_identifier}:{self.config.lightrag_user_hash_salt}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _get_rollout_percentage_from_hash(self, user_hash: str) -> float:
        """
        Convert user hash to rollout percentage for consistent assignment.
        
        Args:
            user_hash: User's consistent hash value
        
        Returns:
            Percentage value (0-100) based on hash
        """
        # Use last 8 characters of hash for percentage calculation
        hash_suffix = user_hash[-8:]
        hash_int = int(hash_suffix, 16)
        max_hash = 16**8 - 1  # Maximum value for 8 hex characters
        return (hash_int / max_hash) * 100
    
    def _assign_user_cohort(self, user_identifier: str, user_hash: str) -> UserCohort:
        """
        Assign user to A/B testing cohort based on hash.
        
        Args:
            user_identifier: Unique identifier for the user/session
            user_hash: Pre-calculated user hash
        
        Returns:
            UserCohort assignment for the user
        """
        # Check cache first
        if user_identifier in self._cohort_cache:
            return self._cohort_cache[user_identifier]
        
        # Calculate cohort based on hash
        rollout_percentage = self._get_rollout_percentage_from_hash(user_hash)
        
        if not self.config.lightrag_enable_ab_testing:
            # Simple rollout without A/B testing
            cohort = UserCohort.LIGHTRAG if rollout_percentage <= self.config.lightrag_rollout_percentage else UserCohort.PERPLEXITY
        else:
            # A/B testing with equal split within rollout percentage
            if rollout_percentage <= self.config.lightrag_rollout_percentage:
                # Within rollout percentage, split 50/50 between LightRAG and Perplexity
                mid_point = rollout_percentage <= (self.config.lightrag_rollout_percentage / 2)
                cohort = UserCohort.LIGHTRAG if mid_point else UserCohort.PERPLEXITY
            else:
                # Outside rollout percentage, use control (Perplexity)
                cohort = UserCohort.CONTROL
        
        # Cache the assignment
        self._cohort_cache[user_identifier] = cohort
        
        self.logger.debug(f"Assigned user {user_identifier[:8]}... to cohort {cohort.value}")
        return cohort
    
    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should prevent LightRAG usage.
        
        Returns:
            True if circuit breaker is open (should block LightRAG), False otherwise
        """
        if not self.config.lightrag_enable_circuit_breaker:
            return False
        
        with self._lock:
            current_time = datetime.now()
            
            # If circuit breaker is open, check if recovery timeout has passed
            if self.circuit_breaker_state.is_open:
                if (self.circuit_breaker_state.last_failure_time and 
                    current_time - self.circuit_breaker_state.last_failure_time > 
                    timedelta(seconds=self.config.lightrag_circuit_breaker_recovery_timeout)):
                    
                    # Attempt recovery
                    self.circuit_breaker_state.is_open = False
                    self.circuit_breaker_state.recovery_attempts += 1
                    self.logger.info(f"Circuit breaker attempting recovery (attempt {self.circuit_breaker_state.recovery_attempts})")
                    return False
                
                return True
            
            # Check if failure threshold is exceeded
            if (self.circuit_breaker_state.failure_count >= 
                self.config.lightrag_circuit_breaker_failure_threshold):
                
                self.circuit_breaker_state.is_open = True
                self.circuit_breaker_state.last_failure_time = current_time
                self.logger.warning(f"Circuit breaker opened due to {self.circuit_breaker_state.failure_count} failures")
                return True
            
            return False
    
    def _evaluate_conditional_rules(self, context: RoutingContext) -> Tuple[bool, str]:
        """
        Evaluate conditional routing rules against context.
        
        Args:
            context: Routing context with query and user information
        
        Returns:
            Tuple of (should_use_lightrag, rule_name)
        """
        if not self.config.lightrag_enable_conditional_routing or not self.routing_rules:
            return True, "no_rules"
        
        for rule_name, rule_func in self.routing_rules.items():
            try:
                if rule_func(context):
                    self.logger.debug(f"Conditional rule {rule_name} triggered for LightRAG")
                    return True, rule_name
            except Exception as e:
                self.logger.warning(f"Error evaluating rule {rule_name}: {e}")
        
        return False, "no_matching_rules"
    
    def _check_quality_threshold(self) -> bool:
        """
        Check if LightRAG quality meets minimum threshold.
        
        Returns:
            True if quality is acceptable, False otherwise
        """
        if not self.config.lightrag_enable_quality_metrics:
            return True
        
        with self._lock:
            avg_quality = self.performance_metrics.get_lightrag_avg_quality()
            
            if avg_quality > 0 and avg_quality < self.config.lightrag_min_quality_threshold:
                self.logger.warning(f"LightRAG quality {avg_quality} below threshold {self.config.lightrag_min_quality_threshold}")
                return False
            
            return True
    
    def _get_cached_routing_result(self, cache_key: str) -> Optional[RoutingResult]:
        """
        Retrieve cached routing result if still valid.
        
        Args:
            cache_key: Key for routing cache lookup
        
        Returns:
            Cached RoutingResult if valid, None otherwise
        """
        with self._lock:
            if cache_key in self._routing_cache:
                result, timestamp = self._routing_cache[cache_key]
                if datetime.now() - timestamp < self._cache_ttl:
                    return result
                else:
                    # Remove expired entry
                    del self._routing_cache[cache_key]
            
            return None
    
    def _cache_routing_result(self, cache_key: str, result: RoutingResult) -> None:
        """
        Cache routing result for performance optimization.
        
        Args:
            cache_key: Key for routing cache storage
            result: RoutingResult to cache
        """
        with self._lock:
            # Limit cache size
            if len(self._routing_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(self._routing_cache.keys(), 
                                   key=lambda k: self._routing_cache[k][1])[:100]
                for key in oldest_keys:
                    del self._routing_cache[key]
            
            self._routing_cache[cache_key] = (result, datetime.now())
    
    def should_use_lightrag(self, context: RoutingContext) -> RoutingResult:
        """
        Determine whether to use LightRAG or fallback to Perplexity.
        
        This is the main routing decision method that evaluates all configured
        criteria including rollout percentage, circuit breaker, quality thresholds,
        and conditional rules.
        
        Args:
            context: RoutingContext with user and query information
        
        Returns:
            RoutingResult with decision and reasoning
        """
        start_time = time.time()
        
        # Generate cache key
        user_identifier = context.user_id or context.session_id or "anonymous"
        cache_key = f"{user_identifier}:{hash(context.query_text or '')}"
        
        # Check cache first (for performance)
        cached_result = self._get_cached_routing_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 1. Check if integration is globally enabled
            if not self.config.lightrag_integration_enabled:
                result = RoutingResult(
                    decision=RoutingDecision.PERPLEXITY,
                    reason=RoutingReason.FEATURE_DISABLED,
                    confidence=1.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._cache_routing_result(cache_key, result)
                return result
            
            # 2. Check forced cohort override
            if self.config.lightrag_force_user_cohort:
                forced_cohort = UserCohort.LIGHTRAG if self.config.lightrag_force_user_cohort == 'lightrag' else UserCohort.PERPLEXITY
                result = RoutingResult(
                    decision=RoutingDecision.LIGHTRAG if forced_cohort == UserCohort.LIGHTRAG else RoutingDecision.PERPLEXITY,
                    reason=RoutingReason.FORCED_COHORT,
                    user_cohort=forced_cohort,
                    confidence=1.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._cache_routing_result(cache_key, result)
                return result
            
            # 3. Check circuit breaker
            circuit_breaker_open = self._check_circuit_breaker()
            if circuit_breaker_open:
                result = RoutingResult(
                    decision=RoutingDecision.PERPLEXITY,
                    reason=RoutingReason.CIRCUIT_BREAKER_OPEN,
                    circuit_breaker_state="open",
                    confidence=1.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._cache_routing_result(cache_key, result)
                return result
            
            # 4. Check quality threshold
            quality_acceptable = self._check_quality_threshold()
            if not quality_acceptable:
                result = RoutingResult(
                    decision=RoutingDecision.PERPLEXITY,
                    reason=RoutingReason.QUALITY_THRESHOLD,
                    quality_score=self.performance_metrics.get_lightrag_avg_quality(),
                    confidence=0.8,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._cache_routing_result(cache_key, result)
                return result
            
            # 5. Evaluate conditional routing rules
            rules_pass, rule_name = self._evaluate_conditional_rules(context)
            if self.config.lightrag_enable_conditional_routing and not rules_pass:
                result = RoutingResult(
                    decision=RoutingDecision.PERPLEXITY,
                    reason=RoutingReason.CONDITIONAL_RULE,
                    confidence=0.9,
                    metadata={'failed_rule': rule_name},
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._cache_routing_result(cache_key, result)
                return result
            
            # 6. Calculate user assignment based on rollout percentage
            user_hash = self._calculate_user_hash(user_identifier)
            user_cohort = self._assign_user_cohort(user_identifier, user_hash)
            rollout_percentage = self._get_rollout_percentage_from_hash(user_hash)
            
            # Make routing decision based on cohort
            decision = RoutingDecision.LIGHTRAG if user_cohort == UserCohort.LIGHTRAG else RoutingDecision.PERPLEXITY
            reason = RoutingReason.USER_COHORT_ASSIGNMENT if self.config.lightrag_enable_ab_testing else RoutingReason.ROLLOUT_PERCENTAGE
            
            result = RoutingResult(
                decision=decision,
                reason=reason,
                user_cohort=user_cohort,
                confidence=0.95,
                rollout_hash=user_hash[:16],  # First 16 chars for logging
                circuit_breaker_state="closed",
                quality_score=self.performance_metrics.get_lightrag_avg_quality() or None,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    'rollout_percentage_achieved': rollout_percentage,
                    'rollout_threshold': self.config.lightrag_rollout_percentage,
                    'rule_triggered': rule_name if self.config.lightrag_enable_conditional_routing else None
                }
            )
            
            # Cache the result
            self._cache_routing_result(cache_key, result)
            
            self.logger.debug(f"Routing decision for {user_identifier[:8]}...: {decision.value} (reason: {reason.value})")
            return result
        
        except Exception as e:
            self.logger.error(f"Error in routing decision: {e}")
            # Fallback to Perplexity on any error
            result = RoutingResult(
                decision=RoutingDecision.PERPLEXITY,
                reason=RoutingReason.PERFORMANCE_DEGRADATION,
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
            return result
    
    def record_success(self, service: str, response_time: float, quality_score: Optional[float] = None) -> None:
        """
        Record successful request for performance tracking.
        
        Args:
            service: Service name ('lightrag' or 'perplexity')
            response_time: Response time in seconds
            quality_score: Optional quality score (0.0-1.0)
        """
        with self._lock:
            current_time = datetime.now()
            
            if service.lower() == 'lightrag':
                self.performance_metrics.lightrag_success_count += 1
                self.performance_metrics.lightrag_response_times.append(response_time)
                if quality_score is not None:
                    self.performance_metrics.lightrag_quality_scores.append(quality_score)
                
                # Reset circuit breaker failure count on success
                if self.circuit_breaker_state.failure_count > 0:
                    self.circuit_breaker_state.failure_count = max(0, self.circuit_breaker_state.failure_count - 1)
                    self.circuit_breaker_state.last_success_time = current_time
            
            elif service.lower() == 'perplexity':
                self.performance_metrics.perplexity_success_count += 1
                self.performance_metrics.perplexity_response_times.append(response_time)
                if quality_score is not None:
                    self.performance_metrics.perplexity_quality_scores.append(quality_score)
            
            self.circuit_breaker_state.total_requests += 1
            self.circuit_breaker_state.successful_requests += 1
            self.performance_metrics.last_updated = current_time
            
            # Limit metrics arrays to prevent memory growth
            max_history = 1000
            if len(self.performance_metrics.lightrag_response_times) > max_history:
                self.performance_metrics.lightrag_response_times = self.performance_metrics.lightrag_response_times[-max_history:]
            if len(self.performance_metrics.perplexity_response_times) > max_history:
                self.performance_metrics.perplexity_response_times = self.performance_metrics.perplexity_response_times[-max_history:]
            if len(self.performance_metrics.lightrag_quality_scores) > max_history:
                self.performance_metrics.lightrag_quality_scores = self.performance_metrics.lightrag_quality_scores[-max_history:]
            if len(self.performance_metrics.perplexity_quality_scores) > max_history:
                self.performance_metrics.perplexity_quality_scores = self.performance_metrics.perplexity_quality_scores[-max_history:]
    
    def record_failure(self, service: str, error_details: Optional[str] = None) -> None:
        """
        Record failed request for circuit breaker and metrics tracking.
        
        Args:
            service: Service name ('lightrag' or 'perplexity')
            error_details: Optional error details for logging
        """
        with self._lock:
            current_time = datetime.now()
            
            if service.lower() == 'lightrag':
                self.performance_metrics.lightrag_error_count += 1
                
                # Update circuit breaker state
                self.circuit_breaker_state.failure_count += 1
                self.circuit_breaker_state.last_failure_time = current_time
                
                self.logger.warning(f"LightRAG failure recorded (count: {self.circuit_breaker_state.failure_count}): {error_details}")
            
            elif service.lower() == 'perplexity':
                self.performance_metrics.perplexity_error_count += 1
                self.logger.warning(f"Perplexity failure recorded: {error_details}")
            
            self.circuit_breaker_state.total_requests += 1
            self.performance_metrics.last_updated = current_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for monitoring and debugging.
        
        Returns:
            Dictionary containing performance metrics and circuit breaker state
        """
        with self._lock:
            return {
                'circuit_breaker': {
                    'is_open': self.circuit_breaker_state.is_open,
                    'failure_count': self.circuit_breaker_state.failure_count,
                    'failure_rate': self.circuit_breaker_state.failure_rate,
                    'success_rate': self.circuit_breaker_state.success_rate,
                    'total_requests': self.circuit_breaker_state.total_requests,
                    'recovery_attempts': self.circuit_breaker_state.recovery_attempts,
                    'last_failure': self.circuit_breaker_state.last_failure_time.isoformat() if self.circuit_breaker_state.last_failure_time else None,
                    'last_success': self.circuit_breaker_state.last_success_time.isoformat() if self.circuit_breaker_state.last_success_time else None
                },
                'performance': {
                    'lightrag': {
                        'success_count': self.performance_metrics.lightrag_success_count,
                        'error_count': self.performance_metrics.lightrag_error_count,
                        'avg_response_time': self.performance_metrics.get_lightrag_avg_response_time(),
                        'avg_quality_score': self.performance_metrics.get_lightrag_avg_quality(),
                        'total_responses': len(self.performance_metrics.lightrag_response_times)
                    },
                    'perplexity': {
                        'success_count': self.performance_metrics.perplexity_success_count,
                        'error_count': self.performance_metrics.perplexity_error_count,
                        'avg_response_time': self.performance_metrics.get_perplexity_avg_response_time(),
                        'avg_quality_score': self.performance_metrics.get_perplexity_avg_quality(),
                        'total_responses': len(self.performance_metrics.perplexity_response_times)
                    },
                    'last_updated': self.performance_metrics.last_updated.isoformat()
                },
                'configuration': {
                    'integration_enabled': self.config.lightrag_integration_enabled,
                    'rollout_percentage': self.config.lightrag_rollout_percentage,
                    'ab_testing_enabled': self.config.lightrag_enable_ab_testing,
                    'circuit_breaker_enabled': self.config.lightrag_enable_circuit_breaker,
                    'quality_metrics_enabled': self.config.lightrag_enable_quality_metrics,
                    'conditional_routing_enabled': self.config.lightrag_enable_conditional_routing,
                    'force_user_cohort': self.config.lightrag_force_user_cohort
                },
                'cache_stats': {
                    'routing_cache_size': len(self._routing_cache),
                    'cohort_cache_size': len(self._cohort_cache),
                    'routing_rules_count': len(self.routing_rules)
                }
            }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state for manual recovery."""
        with self._lock:
            self.circuit_breaker_state = CircuitBreakerState()
            self.logger.info("Circuit breaker manually reset")
    
    def clear_caches(self) -> None:
        """Clear all caches for fresh state."""
        with self._lock:
            self._routing_cache.clear()
            self._cohort_cache.clear()
            self.logger.info("Feature flag caches cleared")
    
    def update_rollout_percentage(self, percentage: float) -> None:
        """
        Update rollout percentage dynamically.
        
        Args:
            percentage: New rollout percentage (0-100)
        
        Raises:
            ValueError: If percentage is out of valid range
        """
        if not (0 <= percentage <= 100):
            raise ValueError("Rollout percentage must be between 0 and 100")
        
        with self._lock:
            old_percentage = self.config.lightrag_rollout_percentage
            self.config.lightrag_rollout_percentage = percentage
            
            # Clear caches to ensure new percentage takes effect
            self.clear_caches()
            
            self.logger.info(f"Rollout percentage updated from {old_percentage}% to {percentage}%")