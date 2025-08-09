"""
Enhanced Circuit Breaker System for Clinical Metabolomics Oracle
===============================================================

This module implements a comprehensive enhanced circuit breaker system with service-specific 
breakers, cross-service coordination, and progressive degradation management. It builds upon 
the existing cost-based circuit breaker infrastructure and integrates with the production 
load balancer and fallback systems.

Key Components:
1. Service-Specific Circuit Breakers - Dedicated breakers for OpenAI, Perplexity, LightRAG, Cache
2. Circuit Breaker Orchestrator - System-wide coordination and state management
3. Failure Correlation Analyzer - Pattern detection and system-wide failure analysis
4. Progressive Degradation Manager - Graceful service degradation strategies

Features:
- Integration with existing cost-based circuit breakers
- Adaptive threshold adjustment based on service performance
- Comprehensive logging and metrics collection
- Cross-service coordination to prevent cascade failures
- Production-ready monitoring and alerting integration

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Enhanced Circuit Breaker Implementation
Version: 1.0.0
"""

import asyncio
import logging
import threading
import time
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import json
import uuid
import hashlib
from abc import ABC, abstractmethod

# Import existing system components
from .cost_based_circuit_breaker import (
    CostBasedCircuitBreaker, CircuitBreakerState as CostCircuitBreakerState,
    CostThresholdRule, OperationCostEstimator
)


# ============================================================================
# Core Enums and Types
# ============================================================================

class EnhancedCircuitBreakerState(Enum):
    """Enhanced circuit breaker states with additional service-specific states."""
    CLOSED = "closed"                    # Normal operation
    OPEN = "open"                       # Blocking operations due to failures
    HALF_OPEN = "half_open"             # Testing recovery
    DEGRADED = "degraded"               # Operating with reduced functionality
    RATE_LIMITED = "rate_limited"       # Rate limiting due to service constraints
    BUDGET_LIMITED = "budget_limited"    # Budget-based limitations
    MAINTENANCE = "maintenance"         # Planned maintenance mode


class ServiceType(Enum):
    """Supported service types for circuit breakers."""
    OPENAI_API = "openai_api"
    PERPLEXITY_API = "perplexity_api"
    LIGHTRAG = "lightrag"
    CACHE = "cache"
    EMBEDDING_SERVICE = "embedding_service"
    KNOWLEDGE_BASE = "knowledge_base"


class FailureType(Enum):
    """Types of failures that can trigger circuit breakers."""
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"
    BUDGET_EXCEEDED = "budget_exceeded"
    QUALITY_DEGRADATION = "quality_degradation"
    MEMORY_ERROR = "memory_error"
    UNKNOWN_ERROR = "unknown_error"


class AlertLevel(Enum):
    """Alert levels for circuit breaker events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ServiceMetrics:
    """Service performance and health metrics."""
    service_type: ServiceType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Service-specific metrics
    rate_limit_remaining: Optional[int] = None
    quota_usage_percentage: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 100.0 - self.get_success_rate()


@dataclass
class FailureEvent:
    """Represents a failure event in the system."""
    service_type: ServiceType
    failure_type: FailureType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    error_message: str = ""
    operation_context: Dict[str, Any] = field(default_factory=dict)
    recovery_time_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class CircuitBreakerConfig:
    """Configuration for enhanced circuit breakers."""
    service_type: ServiceType
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    degraded_threshold: int = 3
    rate_limit_window: float = 60.0
    
    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    min_failure_threshold: int = 3
    max_failure_threshold: int = 20
    threshold_adjustment_factor: float = 0.1
    
    # Service-specific configurations
    service_specific_config: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Base Enhanced Circuit Breaker
# ============================================================================

class BaseEnhancedCircuitBreaker(ABC):
    """Abstract base class for enhanced circuit breakers."""
    
    def __init__(self,
                 name: str,
                 config: CircuitBreakerConfig,
                 logger: Optional[logging.Logger] = None):
        """Initialize base enhanced circuit breaker."""
        self.name = name
        self.config = config
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
        
        # State management
        self.state = EnhancedCircuitBreakerState.CLOSED
        self.state_changed_time = time.time()
        self.half_open_calls = 0
        
        # Metrics
        self.metrics = ServiceMetrics(service_type=config.service_type)
        self.failure_events: deque = deque(maxlen=1000)
        
        # Adaptive threshold management
        self.current_failure_threshold = config.failure_threshold
        self.last_threshold_adjustment = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self._state_change_callbacks: List[Callable] = []
        
        self.logger.info(f"Enhanced circuit breaker '{name}' initialized for {config.service_type.value}")
    
    @abstractmethod
    def _check_service_health(self) -> bool:
        """Check service-specific health conditions."""
        pass
    
    @abstractmethod
    def _get_service_specific_metrics(self) -> Dict[str, Any]:
        """Get service-specific metrics."""
        pass
    
    def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation through circuit breaker protection."""
        with self._lock:
            # Update state based on current conditions
            self._update_state()
            
            # Check if operation is allowed
            if not self._is_operation_allowed():
                self._record_blocked_operation()
                from .clinical_metabolomics_rag import CircuitBreakerError
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is {self.state.value} - operation blocked"
                )
            
            # Execute operation
            start_time = time.time()
            try:
                result = operation(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record successful execution
                self._record_success(execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Classify and record failure
                failure_type = self._classify_failure(e)
                self._record_failure(failure_type, str(e), execution_time)
                raise
    
    def _is_operation_allowed(self) -> bool:
        """Check if operation is allowed based on current state."""
        if self.state == EnhancedCircuitBreakerState.CLOSED:
            return True
        elif self.state == EnhancedCircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        elif self.state == EnhancedCircuitBreakerState.DEGRADED:
            # Allow with reduced rate or functionality
            return True
        elif self.state == EnhancedCircuitBreakerState.RATE_LIMITED:
            # Check rate limiting conditions
            return self._check_rate_limit_allowance()
        else:
            return False
    
    def _update_state(self) -> None:
        """Update circuit breaker state based on current conditions."""
        now = time.time()
        
        # Check for state transitions
        if self.state == EnhancedCircuitBreakerState.OPEN:
            if now - self.state_changed_time >= self.config.recovery_timeout:
                self._transition_to_state(EnhancedCircuitBreakerState.HALF_OPEN)
        
        elif self.state == EnhancedCircuitBreakerState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.half_open_max_calls:
                self._transition_to_state(EnhancedCircuitBreakerState.CLOSED)
            elif self.metrics.consecutive_failures > 0:
                self._transition_to_state(EnhancedCircuitBreakerState.OPEN)
        
        elif self.state == EnhancedCircuitBreakerState.CLOSED:
            if self.metrics.consecutive_failures >= self.current_failure_threshold:
                # Check if should go to degraded mode first
                if self.metrics.consecutive_failures >= self.config.degraded_threshold:
                    if self._check_service_health():
                        self._transition_to_state(EnhancedCircuitBreakerState.DEGRADED)
                    else:
                        self._transition_to_state(EnhancedCircuitBreakerState.OPEN)
                else:
                    self._transition_to_state(EnhancedCircuitBreakerState.OPEN)
        
        # Check for rate limiting conditions
        if self._should_rate_limit():
            self._transition_to_state(EnhancedCircuitBreakerState.RATE_LIMITED)
    
    def _transition_to_state(self, new_state: EnhancedCircuitBreakerState) -> None:
        """Transition to a new state with proper logging and callbacks."""
        if new_state == self.state:
            return
        
        old_state = self.state
        self.state = new_state
        self.state_changed_time = time.time()
        
        if new_state == EnhancedCircuitBreakerState.HALF_OPEN:
            self.half_open_calls = 0
        
        # Log state change
        self.logger.warning(
            f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}"
        )
        
        # Trigger callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")
        
        # Adjust thresholds if needed
        if self.config.enable_adaptive_thresholds:
            self._adjust_adaptive_thresholds()
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful operation execution."""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes += 1
        
        # Update average response time
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics.average_response_time
            )
    
    def _record_failure(self, 
                       failure_type: FailureType, 
                       error_message: str, 
                       execution_time: float) -> None:
        """Record failed operation execution."""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = time.time()
        self.metrics.consecutive_successes = 0
        self.metrics.consecutive_failures += 1
        
        # Create failure event
        failure_event = FailureEvent(
            service_type=self.config.service_type,
            failure_type=failure_type,
            error_message=error_message,
            operation_context={'execution_time': execution_time}
        )
        
        self.failure_events.append(failure_event)
        
        # Log failure
        self.logger.error(
            f"Circuit breaker '{self.name}' recorded failure: "
            f"{failure_type.value} - {error_message[:200]}"
        )
    
    def _record_blocked_operation(self) -> None:
        """Record operation that was blocked by circuit breaker."""
        self.logger.info(f"Circuit breaker '{self.name}' blocked operation - state: {self.state.value}")
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure from exception."""
        error_str = str(exception).lower()
        exception_name = exception.__class__.__name__.lower()
        
        if 'timeout' in error_str or 'timeout' in exception_name:
            return FailureType.TIMEOUT
        elif 'rate limit' in error_str or 'ratelimit' in error_str:
            return FailureType.RATE_LIMIT
        elif 'quota' in error_str or 'quota exceeded' in error_str:
            return FailureType.QUOTA_EXCEEDED
        elif 'auth' in error_str or 'unauthorized' in error_str:
            return FailureType.AUTHENTICATION
        elif 'unavailable' in error_str or 'service unavailable' in error_str:
            return FailureType.SERVICE_UNAVAILABLE
        elif 'budget' in error_str or 'cost' in error_str:
            return FailureType.BUDGET_EXCEEDED
        elif 'memory' in error_str or 'memoryerror' in exception_name:
            return FailureType.MEMORY_ERROR
        elif 'http' in exception_name or hasattr(exception, 'status_code'):
            return FailureType.HTTP_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    def _should_rate_limit(self) -> bool:
        """Check if rate limiting should be applied."""
        # Check recent failure rate
        now = time.time()
        recent_failures = [
            event for event in self.failure_events
            if now - event.timestamp <= self.config.rate_limit_window
        ]
        
        if len(recent_failures) >= self.config.failure_threshold // 2:
            return True
        
        # Service-specific rate limiting logic
        if self.metrics.rate_limit_remaining is not None and self.metrics.rate_limit_remaining < 10:
            return True
        
        return False
    
    def _check_rate_limit_allowance(self) -> bool:
        """Check if operation is allowed under rate limiting."""
        # Simple rate limiting implementation
        now = time.time()
        
        # Allow one operation per second during rate limiting
        if hasattr(self, '_last_rate_limited_call'):
            if now - self._last_rate_limited_call < 1.0:
                return False
        
        self._last_rate_limited_call = now
        return True
    
    def _adjust_adaptive_thresholds(self) -> None:
        """Adjust failure thresholds based on system performance."""
        if not self.config.enable_adaptive_thresholds:
            return
        
        now = time.time()
        # Only adjust every 5 minutes
        if now - self.last_threshold_adjustment < 300:
            return
        
        current_error_rate = self.metrics.get_failure_rate()
        
        # Increase threshold if error rate is consistently low
        if current_error_rate < 5.0 and self.current_failure_threshold > self.config.min_failure_threshold:
            adjustment = max(1, int(self.current_failure_threshold * self.config.threshold_adjustment_factor))
            self.current_failure_threshold = max(
                self.config.min_failure_threshold,
                self.current_failure_threshold - adjustment
            )
            self.logger.info(f"Lowered failure threshold to {self.current_failure_threshold}")
        
        # Decrease threshold if error rate is high
        elif current_error_rate > 20.0 and self.current_failure_threshold < self.config.max_failure_threshold:
            adjustment = max(1, int(self.current_failure_threshold * self.config.threshold_adjustment_factor))
            self.current_failure_threshold = min(
                self.config.max_failure_threshold,
                self.current_failure_threshold + adjustment
            )
            self.logger.info(f"Raised failure threshold to {self.current_failure_threshold}")
        
        self.last_threshold_adjustment = now
    
    def add_state_change_callback(self, callback: Callable) -> None:
        """Add callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            now = time.time()
            
            return {
                'name': self.name,
                'service_type': self.config.service_type.value,
                'state': self.state.value,
                'state_duration': now - self.state_changed_time,
                'metrics': asdict(self.metrics),
                'config': {
                    'failure_threshold': self.current_failure_threshold,
                    'original_failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout
                },
                'recent_failures': len([
                    event for event in self.failure_events
                    if now - event.timestamp <= 300  # Last 5 minutes
                ]),
                'service_specific': self._get_service_specific_metrics(),
                'timestamp': now
            }
    
    def force_state(self, state: EnhancedCircuitBreakerState, reason: str = "Manual override") -> None:
        """Force circuit breaker to specific state."""
        with self._lock:
            self.logger.warning(f"Forcing circuit breaker '{self.name}' to state {state.value}: {reason}")
            self._transition_to_state(state)
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = EnhancedCircuitBreakerState.CLOSED
            self.state_changed_time = time.time()
            self.half_open_calls = 0
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
            self.current_failure_threshold = self.config.failure_threshold
            
            self.logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")


# ============================================================================
# Service-Specific Circuit Breakers
# ============================================================================

class OpenAICircuitBreaker(BaseEnhancedCircuitBreaker):
    """Circuit breaker specifically designed for OpenAI API integration."""
    
    def __init__(self,
                 name: str = "openai_circuit_breaker",
                 config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize OpenAI circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(
                service_type=ServiceType.OPENAI_API,
                failure_threshold=3,  # Lower threshold for API services
                recovery_timeout=30.0,  # Faster recovery for API services
                rate_limit_window=60.0,
                service_specific_config={
                    'model_health_tracking': True,
                    'token_usage_monitoring': True,
                    'cost_per_token_tracking': True,
                    'rate_limit_awareness': True
                }
            )
        
        super().__init__(name, config, logger)
        
        # OpenAI-specific state
        self.model_health: Dict[str, Dict[str, Any]] = {}
        self.token_usage_stats: Dict[str, int] = defaultdict(int)
        self.rate_limit_status: Dict[str, Any] = {}
    
    def _check_service_health(self) -> bool:
        """Check OpenAI API health conditions."""
        try:
            # Check rate limit status
            if self.rate_limit_status.get('requests_remaining', 100) < 10:
                return False
            
            # Check token usage patterns
            recent_token_usage = sum(self.token_usage_stats.values())
            if recent_token_usage > 100000:  # High token usage threshold
                return False
            
            # Check model-specific health
            for model, health_info in self.model_health.items():
                if health_info.get('consecutive_failures', 0) > 2:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking OpenAI service health: {e}")
            return False
    
    def _get_service_specific_metrics(self) -> Dict[str, Any]:
        """Get OpenAI-specific metrics."""
        return {
            'model_health': dict(self.model_health),
            'token_usage_stats': dict(self.token_usage_stats),
            'rate_limit_status': self.rate_limit_status,
            'total_token_usage': sum(self.token_usage_stats.values())
        }
    
    def update_model_health(self, model: str, success: bool, response_info: Dict[str, Any] = None) -> None:
        """Update health information for a specific model."""
        if model not in self.model_health:
            self.model_health[model] = {
                'consecutive_failures': 0,
                'consecutive_successes': 0,
                'total_requests': 0,
                'average_response_time': 0.0,
                'last_used': time.time()
            }
        
        health_info = self.model_health[model]
        health_info['total_requests'] += 1
        health_info['last_used'] = time.time()
        
        if success:
            health_info['consecutive_failures'] = 0
            health_info['consecutive_successes'] += 1
            
            if response_info:
                # Update response time
                response_time = response_info.get('response_time', 0)
                if health_info['average_response_time'] == 0:
                    health_info['average_response_time'] = response_time
                else:
                    alpha = 0.1
                    health_info['average_response_time'] = (
                        alpha * response_time + (1 - alpha) * health_info['average_response_time']
                    )
                
                # Track token usage
                if 'usage' in response_info:
                    usage = response_info['usage']
                    self.token_usage_stats['input_tokens'] += usage.get('prompt_tokens', 0)
                    self.token_usage_stats['output_tokens'] += usage.get('completion_tokens', 0)
                    self.token_usage_stats['total_tokens'] += usage.get('total_tokens', 0)
        else:
            health_info['consecutive_successes'] = 0
            health_info['consecutive_failures'] += 1
    
    def update_rate_limit_status(self, headers: Dict[str, str]) -> None:
        """Update rate limit status from API response headers."""
        try:
            self.rate_limit_status.update({
                'requests_remaining': int(headers.get('x-ratelimit-remaining-requests', 0)),
                'requests_limit': int(headers.get('x-ratelimit-limit-requests', 1000)),
                'tokens_remaining': int(headers.get('x-ratelimit-remaining-tokens', 0)),
                'tokens_limit': int(headers.get('x-ratelimit-limit-tokens', 100000)),
                'reset_time': headers.get('x-ratelimit-reset-requests'),
                'last_updated': time.time()
            })
            
            # Update metrics
            if self.rate_limit_status['requests_limit'] > 0:
                remaining_pct = (self.rate_limit_status['requests_remaining'] / 
                               self.rate_limit_status['requests_limit']) * 100
                self.metrics.rate_limit_remaining = remaining_pct
                
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error parsing rate limit headers: {e}")


class PerplexityCircuitBreaker(BaseEnhancedCircuitBreaker):
    """Circuit breaker specifically designed for Perplexity API integration."""
    
    def __init__(self,
                 name: str = "perplexity_circuit_breaker",
                 config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize Perplexity circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(
                service_type=ServiceType.PERPLEXITY_API,
                failure_threshold=4,
                recovery_timeout=45.0,
                rate_limit_window=300.0,  # Longer window for Perplexity
                service_specific_config={
                    'api_quota_management': True,
                    'query_complexity_analysis': True,
                    'search_result_quality_tracking': True,
                    'citation_accuracy_monitoring': True
                }
            )
        
        super().__init__(name, config, logger)
        
        # Perplexity-specific state
        self.query_complexity_stats: Dict[str, List[float]] = defaultdict(list)
        self.quota_usage: Dict[str, Any] = {}
        self.search_quality_metrics: Dict[str, float] = {}
    
    def _check_service_health(self) -> bool:
        """Check Perplexity API health conditions."""
        try:
            # Check quota usage
            quota_pct = self.quota_usage.get('percentage_used', 0)
            if quota_pct > 90:
                return False
            
            # Check search quality
            avg_quality = statistics.mean(self.search_quality_metrics.values()) if self.search_quality_metrics else 0
            if avg_quality < 0.7:  # Quality threshold
                return False
            
            # Check recent query complexity trends
            recent_complexities = []
            for complexities in self.query_complexity_stats.values():
                recent_complexities.extend(complexities[-10:])  # Last 10 queries per type
            
            if recent_complexities:
                avg_complexity = statistics.mean(recent_complexities)
                if avg_complexity > 0.8:  # High complexity threshold
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking Perplexity service health: {e}")
            return False
    
    def _get_service_specific_metrics(self) -> Dict[str, Any]:
        """Get Perplexity-specific metrics."""
        return {
            'query_complexity_stats': {k: list(v) for k, v in self.query_complexity_stats.items()},
            'quota_usage': self.quota_usage,
            'search_quality_metrics': self.search_quality_metrics,
            'average_search_quality': (
                statistics.mean(self.search_quality_metrics.values()) 
                if self.search_quality_metrics else 0.0
            )
        }
    
    def update_query_complexity(self, query_type: str, complexity_score: float) -> None:
        """Update query complexity statistics."""
        self.query_complexity_stats[query_type].append(complexity_score)
        
        # Keep only recent data
        if len(self.query_complexity_stats[query_type]) > 100:
            self.query_complexity_stats[query_type] = self.query_complexity_stats[query_type][-100:]
    
    def update_search_quality(self, query_id: str, quality_score: float) -> None:
        """Update search result quality metrics."""
        self.search_quality_metrics[query_id] = quality_score
        
        # Keep only recent quality scores
        if len(self.search_quality_metrics) > 1000:
            oldest_keys = sorted(self.search_quality_metrics.keys())[:100]
            for key in oldest_keys:
                del self.search_quality_metrics[key]
    
    def update_quota_status(self, quota_info: Dict[str, Any]) -> None:
        """Update API quota usage information."""
        self.quota_usage.update({
            'requests_used': quota_info.get('requests_used', 0),
            'requests_limit': quota_info.get('requests_limit', 1000),
            'percentage_used': quota_info.get('percentage_used', 0),
            'reset_time': quota_info.get('reset_time'),
            'last_updated': time.time()
        })
        
        # Update metrics
        self.metrics.quota_usage_percentage = self.quota_usage['percentage_used']


class LightRAGCircuitBreaker(BaseEnhancedCircuitBreaker):
    """Circuit breaker specifically designed for LightRAG system integration."""
    
    def __init__(self,
                 name: str = "lightrag_circuit_breaker",
                 config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize LightRAG circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(
                service_type=ServiceType.LIGHTRAG,
                failure_threshold=5,
                recovery_timeout=120.0,  # Longer recovery for complex systems
                rate_limit_window=60.0,
                service_specific_config={
                    'knowledge_base_health_monitoring': True,
                    'index_integrity_checking': True,
                    'embedding_service_health': True,
                    'retrieval_quality_tracking': True
                }
            )
        
        super().__init__(name, config, logger)
        
        # LightRAG-specific state
        self.knowledge_base_health: Dict[str, Any] = {}
        self.retrieval_quality_scores: deque = deque(maxlen=1000)
        self.index_status: Dict[str, Any] = {}
        self.embedding_service_status: Dict[str, Any] = {}
    
    def _check_service_health(self) -> bool:
        """Check LightRAG system health conditions."""
        try:
            # Check knowledge base integrity
            if not self.knowledge_base_health.get('index_accessible', True):
                return False
            
            # Check embedding service health
            embedding_health = self.embedding_service_status.get('consecutive_failures', 0)
            if embedding_health > 3:
                return False
            
            # Check retrieval quality
            if self.retrieval_quality_scores:
                recent_quality = statistics.mean(list(self.retrieval_quality_scores)[-20:])
                if recent_quality < 0.6:  # Quality threshold
                    return False
            
            # Check memory usage
            if self.metrics.memory_usage_mb > 8000:  # 8GB threshold
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking LightRAG service health: {e}")
            return False
    
    def _get_service_specific_metrics(self) -> Dict[str, Any]:
        """Get LightRAG-specific metrics."""
        return {
            'knowledge_base_health': self.knowledge_base_health,
            'retrieval_quality_scores': list(self.retrieval_quality_scores),
            'average_retrieval_quality': (
                statistics.mean(self.retrieval_quality_scores) 
                if self.retrieval_quality_scores else 0.0
            ),
            'index_status': self.index_status,
            'embedding_service_status': self.embedding_service_status,
            'memory_usage_mb': self.metrics.memory_usage_mb
        }
    
    def update_knowledge_base_health(self, health_info: Dict[str, Any]) -> None:
        """Update knowledge base health information."""
        self.knowledge_base_health.update(health_info)
        self.knowledge_base_health['last_updated'] = time.time()
    
    def update_retrieval_quality(self, quality_score: float) -> None:
        """Update retrieval quality score."""
        self.retrieval_quality_scores.append(quality_score)
    
    def update_embedding_service_status(self, success: bool, response_time: float = None) -> None:
        """Update embedding service status."""
        if 'total_requests' not in self.embedding_service_status:
            self.embedding_service_status = {
                'total_requests': 0,
                'successful_requests': 0,
                'consecutive_failures': 0,
                'average_response_time': 0.0
            }
        
        status = self.embedding_service_status
        status['total_requests'] += 1
        
        if success:
            status['successful_requests'] += 1
            status['consecutive_failures'] = 0
            
            if response_time:
                if status['average_response_time'] == 0:
                    status['average_response_time'] = response_time
                else:
                    alpha = 0.1
                    status['average_response_time'] = (
                        alpha * response_time + (1 - alpha) * status['average_response_time']
                    )
        else:
            status['consecutive_failures'] += 1
        
        status['last_updated'] = time.time()


class CacheCircuitBreaker(BaseEnhancedCircuitBreaker):
    """Circuit breaker specifically designed for Cache system integration."""
    
    def __init__(self,
                 name: str = "cache_circuit_breaker",
                 config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize Cache circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(
                service_type=ServiceType.CACHE,
                failure_threshold=8,  # Higher threshold for cache - can tolerate more failures
                recovery_timeout=30.0,
                rate_limit_window=60.0,
                service_specific_config={
                    'hit_rate_optimization': True,
                    'memory_pressure_handling': True,
                    'cache_invalidation_tracking': True,
                    'storage_backend_monitoring': True
                }
            )
        
        super().__init__(name, config, logger)
        
        # Cache-specific state
        self.cache_stats: Dict[str, Any] = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'evictions': 0
        }
        self.memory_pressure_alerts: deque = deque(maxlen=100)
        self.storage_backend_health: Dict[str, Any] = {}
    
    def _check_service_health(self) -> bool:
        """Check Cache system health conditions."""
        try:
            # Check hit rate
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            if total_requests > 0:
                hit_rate = self.cache_stats['hits'] / total_requests
                if hit_rate < 0.3:  # Low hit rate threshold
                    return False
            
            # Check memory pressure
            if self.metrics.memory_usage_mb > 4000:  # 4GB threshold for cache
                return False
            
            # Check recent memory pressure alerts
            recent_alerts = [
                alert for alert in self.memory_pressure_alerts
                if time.time() - alert['timestamp'] <= 300  # Last 5 minutes
            ]
            if len(recent_alerts) > 5:
                return False
            
            # Check storage backend health
            backend_failures = self.storage_backend_health.get('consecutive_failures', 0)
            if backend_failures > 5:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking Cache service health: {e}")
            return False
    
    def _get_service_specific_metrics(self) -> Dict[str, Any]:
        """Get Cache-specific metrics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests) if total_requests > 0 else 0
        
        return {
            'cache_stats': self.cache_stats.copy(),
            'hit_rate': hit_rate,
            'memory_pressure_alerts': list(self.memory_pressure_alerts),
            'storage_backend_health': self.storage_backend_health,
            'recent_memory_pressure_alerts': len([
                alert for alert in self.memory_pressure_alerts
                if time.time() - alert['timestamp'] <= 300
            ])
        }
    
    def record_cache_operation(self, operation: str, success: bool = True) -> None:
        """Record cache operation statistics."""
        if operation in self.cache_stats:
            self.cache_stats[operation] += 1
        
        # Update hit rate metric
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_requests > 0:
            self.metrics.cache_hit_rate = self.cache_stats['hits'] / total_requests
    
    def record_memory_pressure_alert(self, memory_usage_mb: float, pressure_level: str) -> None:
        """Record memory pressure alert."""
        alert = {
            'timestamp': time.time(),
            'memory_usage_mb': memory_usage_mb,
            'pressure_level': pressure_level
        }
        
        self.memory_pressure_alerts.append(alert)
        self.metrics.memory_usage_mb = memory_usage_mb
    
    def update_storage_backend_health(self, success: bool, operation_type: str = 'general') -> None:
        """Update storage backend health status."""
        if 'total_operations' not in self.storage_backend_health:
            self.storage_backend_health = {
                'total_operations': 0,
                'successful_operations': 0,
                'consecutive_failures': 0,
                'last_operation_time': time.time()
            }
        
        health = self.storage_backend_health
        health['total_operations'] += 1
        health['last_operation_time'] = time.time()
        
        if success:
            health['successful_operations'] += 1
            health['consecutive_failures'] = 0
        else:
            health['consecutive_failures'] += 1


# ============================================================================
# Circuit Breaker Orchestrator
# ============================================================================

class CircuitBreakerOrchestrator:
    """System-wide circuit breaker coordination and management."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize circuit breaker orchestrator."""
        self.logger = logger or logging.getLogger(f"{__name__}.orchestrator")
        
        # Circuit breaker registry
        self._circuit_breakers: Dict[str, BaseEnhancedCircuitBreaker] = {}
        self._service_dependencies: Dict[ServiceType, List[ServiceType]] = {}
        
        # System state
        self._system_state = "healthy"
        self._degradation_level = 0
        self._last_system_check = time.time()
        
        # Coordination logic
        self._coordination_rules: List[Dict[str, Any]] = []
        self._system_alerts: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default coordination rules
        self._initialize_coordination_rules()
        
        self.logger.info("Circuit breaker orchestrator initialized")
    
    def register_circuit_breaker(self, 
                                circuit_breaker: BaseEnhancedCircuitBreaker,
                                dependencies: Optional[List[ServiceType]] = None) -> None:
        """Register a circuit breaker with the orchestrator."""
        with self._lock:
            name = circuit_breaker.name
            service_type = circuit_breaker.config.service_type
            
            self._circuit_breakers[name] = circuit_breaker
            
            # Register dependencies
            if dependencies:
                self._service_dependencies[service_type] = dependencies
            
            # Add state change callback
            circuit_breaker.add_state_change_callback(self._handle_state_change)
            
            self.logger.info(f"Registered circuit breaker '{name}' for {service_type.value}")
    
    def _initialize_coordination_rules(self) -> None:
        """Initialize default coordination rules."""
        self._coordination_rules = [
            {
                'name': 'cascade_failure_prevention',
                'condition': 'multiple_services_failing',
                'action': 'progressive_degradation',
                'threshold': 2,
                'priority': 10
            },
            {
                'name': 'dependency_chain_protection',
                'condition': 'dependency_service_failed',
                'action': 'preemptive_degradation',
                'priority': 8
            },
            {
                'name': 'system_overload_protection',
                'condition': 'high_error_rates_across_services',
                'action': 'rate_limit_all_services',
                'threshold': 0.3,  # 30% error rate
                'priority': 9
            },
            {
                'name': 'recovery_coordination',
                'condition': 'services_recovering',
                'action': 'gradual_recovery',
                'priority': 5
            }
        ]
    
    def _handle_state_change(self, 
                           breaker_name: str,
                           old_state: EnhancedCircuitBreakerState,
                           new_state: EnhancedCircuitBreakerState) -> None:
        """Handle circuit breaker state changes."""
        with self._lock:
            # Log state change
            self.logger.info(
                f"Circuit breaker '{breaker_name}' changed state: {old_state.value} -> {new_state.value}"
            )
            
            # Create system alert
            alert = {
                'timestamp': time.time(),
                'type': 'state_change',
                'breaker_name': breaker_name,
                'old_state': old_state.value,
                'new_state': new_state.value,
                'alert_level': self._determine_alert_level(old_state, new_state)
            }
            self._system_alerts.append(alert)
            
            # Apply coordination rules
            self._apply_coordination_rules()
            
            # Update system state
            self._update_system_state()
    
    def _determine_alert_level(self,
                              old_state: EnhancedCircuitBreakerState,
                              new_state: EnhancedCircuitBreakerState) -> AlertLevel:
        """Determine alert level for state change."""
        if new_state == EnhancedCircuitBreakerState.OPEN:
            return AlertLevel.ERROR
        elif new_state == EnhancedCircuitBreakerState.DEGRADED:
            return AlertLevel.WARNING
        elif new_state == EnhancedCircuitBreakerState.CLOSED and old_state == EnhancedCircuitBreakerState.OPEN:
            return AlertLevel.INFO
        else:
            return AlertLevel.INFO
    
    def _apply_coordination_rules(self) -> None:
        """Apply coordination rules based on current system state."""
        # Sort rules by priority
        sorted_rules = sorted(self._coordination_rules, key=lambda r: r['priority'], reverse=True)
        
        for rule in sorted_rules:
            if self._evaluate_rule_condition(rule):
                self._execute_rule_action(rule)
    
    def _evaluate_rule_condition(self, rule: Dict[str, Any]) -> bool:
        """Evaluate if a coordination rule condition is met."""
        condition = rule['condition']
        
        if condition == 'multiple_services_failing':
            failed_breakers = [
                cb for cb in self._circuit_breakers.values()
                if cb.state in [EnhancedCircuitBreakerState.OPEN, EnhancedCircuitBreakerState.DEGRADED]
            ]
            return len(failed_breakers) >= rule.get('threshold', 2)
        
        elif condition == 'dependency_service_failed':
            return self._check_dependency_failures()
        
        elif condition == 'high_error_rates_across_services':
            total_error_rate = self._calculate_system_error_rate()
            return total_error_rate >= rule.get('threshold', 0.3)
        
        elif condition == 'services_recovering':
            recovering_breakers = [
                cb for cb in self._circuit_breakers.values()
                if cb.state == EnhancedCircuitBreakerState.HALF_OPEN
            ]
            return len(recovering_breakers) > 0
        
        return False
    
    def _execute_rule_action(self, rule: Dict[str, Any]) -> None:
        """Execute coordination rule action."""
        action = rule['action']
        rule_name = rule['name']
        
        if action == 'progressive_degradation':
            self._apply_progressive_degradation()
            self.logger.warning(f"Applied progressive degradation due to rule: {rule_name}")
        
        elif action == 'preemptive_degradation':
            self._apply_preemptive_degradation()
            self.logger.warning(f"Applied preemptive degradation due to rule: {rule_name}")
        
        elif action == 'rate_limit_all_services':
            self._apply_system_wide_rate_limiting()
            self.logger.warning(f"Applied system-wide rate limiting due to rule: {rule_name}")
        
        elif action == 'gradual_recovery':
            self._coordinate_gradual_recovery()
            self.logger.info(f"Coordinating gradual recovery due to rule: {rule_name}")
    
    def _check_dependency_failures(self) -> bool:
        """Check if any service dependencies have failed."""
        for service_type, dependencies in self._service_dependencies.items():
            for dependency in dependencies:
                # Find circuit breakers for this dependency
                dependent_breakers = [
                    cb for cb in self._circuit_breakers.values()
                    if cb.config.service_type == dependency
                ]
                
                # Check if any dependent service has failed
                for breaker in dependent_breakers:
                    if breaker.state == EnhancedCircuitBreakerState.OPEN:
                        return True
        
        return False
    
    def _calculate_system_error_rate(self) -> float:
        """Calculate overall system error rate."""
        total_requests = 0
        total_errors = 0
        
        for breaker in self._circuit_breakers.values():
            total_requests += breaker.metrics.total_requests
            total_errors += breaker.metrics.failed_requests
        
        if total_requests == 0:
            return 0.0
        
        return total_errors / total_requests
    
    def _apply_progressive_degradation(self) -> None:
        """Apply progressive degradation across services."""
        self._degradation_level = min(self._degradation_level + 1, 5)
        
        # Apply degradation based on level
        for breaker in self._circuit_breakers.values():
            if breaker.state == EnhancedCircuitBreakerState.CLOSED:
                # Move to degraded state if not critical service
                if breaker.config.service_type not in [ServiceType.CACHE]:  # Cache is less critical
                    breaker.force_state(EnhancedCircuitBreakerState.DEGRADED, "Progressive degradation")
    
    def _apply_preemptive_degradation(self) -> None:
        """Apply preemptive degradation to prevent cascade failures."""
        # Identify services that depend on failed services
        failed_service_types = {
            cb.config.service_type for cb in self._circuit_breakers.values()
            if cb.state == EnhancedCircuitBreakerState.OPEN
        }
        
        for service_type, dependencies in self._service_dependencies.items():
            if any(dep in failed_service_types for dep in dependencies):
                # Find breakers for this service type and degrade them
                for breaker in self._circuit_breakers.values():
                    if (breaker.config.service_type == service_type and 
                        breaker.state == EnhancedCircuitBreakerState.CLOSED):
                        breaker.force_state(EnhancedCircuitBreakerState.DEGRADED, "Preemptive degradation")
    
    def _apply_system_wide_rate_limiting(self) -> None:
        """Apply rate limiting across all services."""
        for breaker in self._circuit_breakers.values():
            if breaker.state in [EnhancedCircuitBreakerState.CLOSED, EnhancedCircuitBreakerState.DEGRADED]:
                breaker.force_state(EnhancedCircuitBreakerState.RATE_LIMITED, "System-wide rate limiting")
    
    def _coordinate_gradual_recovery(self) -> None:
        """Coordinate gradual recovery across services."""
        # Prioritize recovery order
        recovery_priority = [
            ServiceType.CACHE,           # Recover cache first (fastest)
            ServiceType.LIGHTRAG,       # Then LightRAG (core functionality)
            ServiceType.OPENAI_API,     # Then OpenAI (primary intelligence)
            ServiceType.PERPLEXITY_API  # Finally Perplexity (supplementary)
        ]
        
        for service_type in recovery_priority:
            breakers = [
                cb for cb in self._circuit_breakers.values()
                if cb.config.service_type == service_type and
                cb.state == EnhancedCircuitBreakerState.HALF_OPEN
            ]
            
            for breaker in breakers:
                # Give recovery services more time
                if time.time() - breaker.state_changed_time > breaker.config.recovery_timeout / 2:
                    # Allow more test calls during recovery
                    breaker.config.half_open_max_calls = min(breaker.config.half_open_max_calls + 1, 10)
    
    def _update_system_state(self) -> None:
        """Update overall system state based on circuit breaker states."""
        open_breakers = sum(1 for cb in self._circuit_breakers.values() 
                          if cb.state == EnhancedCircuitBreakerState.OPEN)
        degraded_breakers = sum(1 for cb in self._circuit_breakers.values() 
                              if cb.state == EnhancedCircuitBreakerState.DEGRADED)
        total_breakers = len(self._circuit_breakers)
        
        if total_breakers == 0:
            self._system_state = "unknown"
        elif open_breakers >= total_breakers / 2:
            self._system_state = "critical"
        elif open_breakers > 0 or degraded_breakers >= total_breakers / 2:
            self._system_state = "degraded"
        else:
            self._system_state = "healthy"
            self._degradation_level = max(0, self._degradation_level - 1)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            breaker_statuses = {}
            for name, breaker in self._circuit_breakers.items():
                breaker_statuses[name] = breaker.get_status()
            
            recent_alerts = [
                alert for alert in self._system_alerts
                if time.time() - alert['timestamp'] <= 3600  # Last hour
            ]
            
            return {
                'system_state': self._system_state,
                'degradation_level': self._degradation_level,
                'total_circuit_breakers': len(self._circuit_breakers),
                'circuit_breaker_states': {
                    state.value: sum(1 for cb in self._circuit_breakers.values() if cb.state == state)
                    for state in EnhancedCircuitBreakerState
                },
                'system_error_rate': self._calculate_system_error_rate(),
                'recent_alerts': recent_alerts,
                'service_dependencies': {
                    st.value: [dep.value for dep in deps] 
                    for st, deps in self._service_dependencies.items()
                },
                'circuit_breakers': breaker_statuses,
                'timestamp': time.time()
            }
    
    def force_system_recovery(self, reason: str = "Manual system recovery") -> None:
        """Force system-wide recovery."""
        with self._lock:
            self.logger.warning(f"Forcing system recovery: {reason}")
            
            for breaker in self._circuit_breakers.values():
                if breaker.state != EnhancedCircuitBreakerState.MAINTENANCE:
                    breaker.force_state(EnhancedCircuitBreakerState.CLOSED, reason)
            
            self._system_state = "healthy"
            self._degradation_level = 0
    
    def set_maintenance_mode(self, service_types: List[ServiceType], reason: str) -> None:
        """Set specific services to maintenance mode."""
        with self._lock:
            for breaker in self._circuit_breakers.values():
                if breaker.config.service_type in service_types:
                    breaker.force_state(EnhancedCircuitBreakerState.MAINTENANCE, reason)
                    self.logger.info(f"Set '{breaker.name}' to maintenance mode: {reason}")


# ============================================================================
# Failure Correlation Analyzer
# ============================================================================

class FailureCorrelationAnalyzer:
    """Analyzes failure patterns and correlations across services."""
    
    def __init__(self,
                 orchestrator: CircuitBreakerOrchestrator,
                 logger: Optional[logging.Logger] = None):
        """Initialize failure correlation analyzer."""
        self.orchestrator = orchestrator
        self.logger = logger or logging.getLogger(f"{__name__}.analyzer")
        
        # Analysis state
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.pattern_alerts: deque = deque(maxlen=500)
        
        # Analysis configuration
        self.analysis_window = 3600  # 1 hour
        self.correlation_threshold = 0.7
        self.pattern_detection_threshold = 3
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Failure correlation analyzer initialized")
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns across all circuit breakers."""
        with self._lock:
            now = time.time()
            analysis_results = {
                'correlations_detected': [],
                'failure_patterns': {},
                'system_wide_patterns': [],
                'recommendations': [],
                'timestamp': now
            }
            
            # Collect recent failure events
            recent_failures = self._collect_recent_failures(now - self.analysis_window)
            
            # Analyze temporal correlations
            correlations = self._analyze_temporal_correlations(recent_failures)
            analysis_results['correlations_detected'] = correlations
            
            # Detect failure patterns
            patterns = self._detect_failure_patterns(recent_failures)
            analysis_results['failure_patterns'] = patterns
            
            # Identify system-wide patterns
            system_patterns = self._identify_system_patterns(recent_failures)
            analysis_results['system_wide_patterns'] = system_patterns
            
            # Generate recommendations
            recommendations = self._generate_recommendations(correlations, patterns, system_patterns)
            analysis_results['recommendations'] = recommendations
            
            return analysis_results
    
    def _collect_recent_failures(self, since_time: float) -> List[Dict[str, Any]]:
        """Collect recent failure events from all circuit breakers."""
        recent_failures = []
        
        for breaker_name, breaker in self.orchestrator._circuit_breakers.items():
            for failure_event in breaker.failure_events:
                if failure_event.timestamp >= since_time:
                    failure_data = failure_event.to_dict()
                    failure_data['breaker_name'] = breaker_name
                    recent_failures.append(failure_data)
        
        # Sort by timestamp
        recent_failures.sort(key=lambda x: x['timestamp'])
        return recent_failures
    
    def _analyze_temporal_correlations(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze temporal correlations between service failures."""
        correlations = []
        
        # Group failures by service
        service_failures = defaultdict(list)
        for failure in failures:
            service_type = failure['service_type']
            service_failures[service_type].append(failure['timestamp'])
        
        # Analyze correlations between services
        services = list(service_failures.keys())
        for i in range(len(services)):
            for j in range(i + 1, len(services)):
                service_a, service_b = services[i], services[j]
                correlation = self._calculate_temporal_correlation(
                    service_failures[service_a],
                    service_failures[service_b]
                )
                
                if correlation >= self.correlation_threshold:
                    correlations.append({
                        'service_a': service_a,
                        'service_b': service_b,
                        'correlation_strength': correlation,
                        'failure_count_a': len(service_failures[service_a]),
                        'failure_count_b': len(service_failures[service_b])
                    })
        
        return correlations
    
    def _calculate_temporal_correlation(self,
                                      timestamps_a: List[float],
                                      timestamps_b: List[float],
                                      time_window: float = 300) -> float:
        """Calculate temporal correlation between two failure sequences."""
        if not timestamps_a or not timestamps_b:
            return 0.0
        
        correlated_pairs = 0
        total_pairs = 0
        
        for ts_a in timestamps_a:
            total_pairs += 1
            # Check if there's a failure in service B within time window
            for ts_b in timestamps_b:
                if abs(ts_a - ts_b) <= time_window:
                    correlated_pairs += 1
                    break
        
        return correlated_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _detect_failure_patterns(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect common failure patterns."""
        patterns = {
            'cascading_failures': [],
            'recurring_failures': [],
            'burst_failures': []
        }
        
        # Detect cascading failures (failures following dependencies)
        cascading = self._detect_cascading_failures(failures)
        patterns['cascading_failures'] = cascading
        
        # Detect recurring failures (same type repeating)
        recurring = self._detect_recurring_failures(failures)
        patterns['recurring_failures'] = recurring
        
        # Detect burst failures (many failures in short time)
        bursts = self._detect_burst_failures(failures)
        patterns['burst_failures'] = bursts
        
        return patterns
    
    def _detect_cascading_failures(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect cascading failure patterns."""
        cascading_patterns = []
        
        # Look for failure sequences that follow dependency chains
        for i in range(len(failures) - 1):
            current_failure = failures[i]
            next_failure = failures[i + 1]
            
            time_diff = next_failure['timestamp'] - current_failure['timestamp']
            
            # If failures are close in time and involve related services
            if time_diff <= 300:  # 5 minutes
                if self._are_services_related(
                    current_failure['service_type'],
                    next_failure['service_type']
                ):
                    cascading_patterns.append({
                        'trigger_service': current_failure['service_type'],
                        'affected_service': next_failure['service_type'],
                        'time_delay': time_diff,
                        'trigger_failure_type': current_failure['failure_type'],
                        'affected_failure_type': next_failure['failure_type']
                    })
        
        return cascading_patterns
    
    def _detect_recurring_failures(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect recurring failure patterns."""
        recurring_patterns = []
        
        # Group by service and failure type
        failure_groups = defaultdict(list)
        for failure in failures:
            key = (failure['service_type'], failure['failure_type'])
            failure_groups[key].append(failure['timestamp'])
        
        # Look for recurring patterns
        for (service, failure_type), timestamps in failure_groups.items():
            if len(timestamps) >= self.pattern_detection_threshold:
                # Calculate intervals
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = statistics.mean(intervals) if intervals else 0
                
                recurring_patterns.append({
                    'service_type': service,
                    'failure_type': failure_type,
                    'occurrence_count': len(timestamps),
                    'average_interval': avg_interval,
                    'interval_variance': statistics.variance(intervals) if len(intervals) > 1 else 0
                })
        
        return recurring_patterns
    
    def _detect_burst_failures(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect burst failure patterns."""
        burst_patterns = []
        burst_window = 300  # 5 minutes
        
        # Group failures by time windows
        time_buckets = defaultdict(list)
        for failure in failures:
            bucket = int(failure['timestamp'] // burst_window)
            time_buckets[bucket].append(failure)
        
        # Identify bursts
        for bucket, bucket_failures in time_buckets.items():
            if len(bucket_failures) >= 5:  # 5+ failures in 5 minutes
                burst_patterns.append({
                    'start_time': bucket * burst_window,
                    'failure_count': len(bucket_failures),
                    'affected_services': list(set(f['service_type'] for f in bucket_failures)),
                    'dominant_failure_types': self._get_dominant_failure_types(bucket_failures)
                })
        
        return burst_patterns
    
    def _are_services_related(self, service_a: str, service_b: str) -> bool:
        """Check if two services are related through dependencies."""
        # Check direct dependencies
        for service_type, dependencies in self.orchestrator._service_dependencies.items():
            if service_type.value == service_a:
                if any(dep.value == service_b for dep in dependencies):
                    return True
            elif service_type.value == service_b:
                if any(dep.value == service_a for dep in dependencies):
                    return True
        
        return False
    
    def _get_dominant_failure_types(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Get the most common failure types from a list of failures."""
        failure_type_counts = defaultdict(int)
        for failure in failures:
            failure_type_counts[failure['failure_type']] += 1
        
        # Return failure types that represent >20% of failures
        total_failures = len(failures)
        dominant_types = [
            failure_type for failure_type, count in failure_type_counts.items()
            if count / total_failures >= 0.2
        ]
        
        return dominant_types
    
    def _identify_system_patterns(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify system-wide failure patterns."""
        system_patterns = []
        
        # Pattern: All services failing simultaneously
        if len(failures) >= 10:  # Significant number of failures
            time_groups = defaultdict(list)
            for failure in failures:
                # Group by 1-minute windows
                minute_bucket = int(failure['timestamp'] // 60)
                time_groups[minute_bucket].append(failure)
            
            for minute_bucket, minute_failures in time_groups.items():
                if len(minute_failures) >= 5:
                    affected_services = set(f['service_type'] for f in minute_failures)
                    if len(affected_services) >= 3:  # Multiple services affected
                        system_patterns.append({
                            'pattern_type': 'system_wide_failure',
                            'time_window': minute_bucket * 60,
                            'affected_services': list(affected_services),
                            'failure_count': len(minute_failures),
                            'severity': 'high' if len(affected_services) >= 4 else 'medium'
                        })
        
        return system_patterns
    
    def _generate_recommendations(self,
                                correlations: List[Dict[str, Any]],
                                patterns: Dict[str, Any],
                                system_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Recommendations based on correlations
        for correlation in correlations:
            if correlation['correlation_strength'] >= 0.8:
                recommendations.append({
                    'type': 'dependency_review',
                    'priority': 'high',
                    'description': f"Strong correlation detected between {correlation['service_a']} and {correlation['service_b']}. Review dependency configuration.",
                    'affected_services': [correlation['service_a'], correlation['service_b']]
                })
        
        # Recommendations based on cascading failures
        if patterns['cascading_failures']:
            recommendations.append({
                'type': 'circuit_breaker_tuning',
                'priority': 'medium',
                'description': "Cascading failure patterns detected. Consider lowering failure thresholds for dependent services.",
                'affected_services': [p['affected_service'] for p in patterns['cascading_failures']]
            })
        
        # Recommendations based on burst patterns
        if patterns['burst_failures']:
            recommendations.append({
                'type': 'rate_limiting',
                'priority': 'high',
                'description': "Burst failure patterns detected. Implement or tune rate limiting mechanisms.",
                'affected_services': list(set(
                    service for pattern in patterns['burst_failures']
                    for service in pattern['affected_services']
                ))
            })
        
        # System-wide recommendations
        if system_patterns:
            recommendations.append({
                'type': 'system_monitoring',
                'priority': 'critical',
                'description': "System-wide failure patterns detected. Review overall system health and monitoring.",
                'affected_services': ['all']
            })
        
        return recommendations


# ============================================================================
# Progressive Degradation Manager
# ============================================================================

class ProgressiveDegradationManager:
    """Manages progressive degradation strategies across the system."""
    
    def __init__(self,
                 orchestrator: CircuitBreakerOrchestrator,
                 logger: Optional[logging.Logger] = None):
        """Initialize progressive degradation manager."""
        self.orchestrator = orchestrator
        self.logger = logger or logging.getLogger(f"{__name__}.degradation")
        
        # Degradation strategies
        self.degradation_strategies: Dict[ServiceType, Dict[str, Any]] = {}
        self.degradation_history: deque = deque(maxlen=1000)
        
        # Current degradation state
        self.current_degradation_level = 0
        self.max_degradation_level = 5
        self.degradation_policies: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default strategies
        self._initialize_degradation_strategies()
        
        self.logger.info("Progressive degradation manager initialized")
    
    def _initialize_degradation_strategies(self) -> None:
        """Initialize degradation strategies for each service type."""
        self.degradation_strategies = {
            ServiceType.OPENAI_API: {
                'level_1': {
                    'description': 'Use smaller models (gpt-4o-mini instead of gpt-4o)',
                    'actions': ['switch_to_smaller_model', 'reduce_max_tokens'],
                    'performance_impact': 0.15,  # 15% performance reduction
                    'cost_savings': 0.80  # 80% cost savings
                },
                'level_2': {
                    'description': 'Reduce response quality and length',
                    'actions': ['limit_response_length', 'simplify_prompts'],
                    'performance_impact': 0.30,
                    'cost_savings': 0.90
                },
                'level_3': {
                    'description': 'Cache-only responses with basic fallback',
                    'actions': ['cache_only_mode', 'basic_template_responses'],
                    'performance_impact': 0.60,
                    'cost_savings': 0.95
                }
            },
            
            ServiceType.PERPLEXITY_API: {
                'level_1': {
                    'description': 'Reduce search complexity and scope',
                    'actions': ['limit_search_scope', 'reduce_citation_depth'],
                    'performance_impact': 0.20,
                    'cost_savings': 0.40
                },
                'level_2': {
                    'description': 'Use cached search results when available',
                    'actions': ['prioritize_cache', 'reduce_real_time_searches'],
                    'performance_impact': 0.40,
                    'cost_savings': 0.70
                },
                'level_3': {
                    'description': 'Disable Perplexity, use alternative sources',
                    'actions': ['disable_service', 'use_backup_search'],
                    'performance_impact': 0.70,
                    'cost_savings': 1.0
                }
            },
            
            ServiceType.LIGHTRAG: {
                'level_1': {
                    'description': 'Reduce retrieval scope and complexity',
                    'actions': ['limit_retrieval_depth', 'reduce_context_window'],
                    'performance_impact': 0.10,
                    'cost_savings': 0.20
                },
                'level_2': {
                    'description': 'Use simplified retrieval algorithms',
                    'actions': ['basic_retrieval_only', 'disable_complex_reasoning'],
                    'performance_impact': 0.35,
                    'cost_savings': 0.50
                },
                'level_3': {
                    'description': 'Emergency mode with minimal functionality',
                    'actions': ['emergency_mode', 'basic_keyword_matching'],
                    'performance_impact': 0.70,
                    'cost_savings': 0.80
                }
            },
            
            ServiceType.CACHE: {
                'level_1': {
                    'description': 'Reduce cache TTL and optimize eviction',
                    'actions': ['reduce_ttl', 'aggressive_eviction'],
                    'performance_impact': 0.05,
                    'cost_savings': 0.30
                },
                'level_2': {
                    'description': 'Cache only critical responses',
                    'actions': ['selective_caching', 'priority_based_storage'],
                    'performance_impact': 0.15,
                    'cost_savings': 0.60
                },
                'level_3': {
                    'description': 'Minimal caching, memory optimization',
                    'actions': ['minimal_cache', 'emergency_cleanup'],
                    'performance_impact': 0.40,
                    'cost_savings': 0.80
                }
            }
        }
    
    def apply_degradation(self, 
                         target_services: Optional[List[ServiceType]] = None,
                         degradation_level: int = 1,
                         reason: str = "System overload") -> Dict[str, Any]:
        """Apply progressive degradation to specified services."""
        with self._lock:
            if target_services is None:
                target_services = list(self.degradation_strategies.keys())
            
            degradation_level = max(1, min(degradation_level, 3))  # Clamp to valid range
            
            degradation_result = {
                'applied_degradations': [],
                'performance_impact': 0.0,
                'cost_savings': 0.0,
                'affected_services': [],
                'timestamp': time.time(),
                'reason': reason
            }
            
            for service_type in target_services:
                if service_type in self.degradation_strategies:
                    service_result = self._apply_service_degradation(
                        service_type, degradation_level, reason
                    )
                    
                    degradation_result['applied_degradations'].append(service_result)
                    degradation_result['performance_impact'] += service_result.get('performance_impact', 0)
                    degradation_result['cost_savings'] += service_result.get('cost_savings', 0)
                    degradation_result['affected_services'].append(service_type.value)
            
            # Average the impacts
            if len(target_services) > 0:
                degradation_result['performance_impact'] /= len(target_services)
                degradation_result['cost_savings'] /= len(target_services)
            
            # Record degradation in history
            self.degradation_history.append(degradation_result)
            
            # Update current degradation level
            self.current_degradation_level = max(self.current_degradation_level, degradation_level)
            
            self.logger.warning(
                f"Applied degradation level {degradation_level} to {len(target_services)} services: {reason}"
            )
            
            return degradation_result
    
    def _apply_service_degradation(self,
                                  service_type: ServiceType,
                                  level: int,
                                  reason: str) -> Dict[str, Any]:
        """Apply degradation to a specific service."""
        strategy_key = f'level_{level}'
        strategy = self.degradation_strategies[service_type].get(strategy_key, {})
        
        service_result = {
            'service_type': service_type.value,
            'degradation_level': level,
            'strategy': strategy,
            'actions_taken': [],
            'performance_impact': strategy.get('performance_impact', 0),
            'cost_savings': strategy.get('cost_savings', 0),
            'success': False
        }
        
        try:
            # Execute degradation actions
            actions = strategy.get('actions', [])
            for action in actions:
                action_result = self._execute_degradation_action(service_type, action)
                service_result['actions_taken'].append({
                    'action': action,
                    'success': action_result,
                    'timestamp': time.time()
                })
            
            service_result['success'] = True
            
            # Update circuit breakers if needed
            self._update_circuit_breakers_for_degradation(service_type, level)
            
        except Exception as e:
            self.logger.error(f"Error applying degradation to {service_type.value}: {e}")
            service_result['error'] = str(e)
        
        return service_result
    
    def _execute_degradation_action(self, service_type: ServiceType, action: str) -> bool:
        """Execute a specific degradation action."""
        try:
            # Find circuit breakers for this service type
            target_breakers = [
                cb for cb in self.orchestrator._circuit_breakers.values()
                if cb.config.service_type == service_type
            ]
            
            if action == 'switch_to_smaller_model':
                # Update service configuration to use smaller models
                for breaker in target_breakers:
                    if isinstance(breaker, OpenAICircuitBreaker):
                        breaker.config.service_specific_config['preferred_model'] = 'gpt-4o-mini'
                        breaker.config.service_specific_config['max_tokens'] = 2000
                
            elif action == 'reduce_max_tokens':
                for breaker in target_breakers:
                    if isinstance(breaker, OpenAICircuitBreaker):
                        current_max = breaker.config.service_specific_config.get('max_tokens', 4000)
                        breaker.config.service_specific_config['max_tokens'] = max(500, int(current_max * 0.5))
                
            elif action == 'limit_search_scope':
                for breaker in target_breakers:
                    if isinstance(breaker, PerplexityCircuitBreaker):
                        breaker.config.service_specific_config['max_search_results'] = 5
                        breaker.config.service_specific_config['search_timeout'] = 15
                
            elif action == 'cache_only_mode':
                for breaker in target_breakers:
                    breaker.force_state(EnhancedCircuitBreakerState.DEGRADED, 
                                      "Degradation: Cache-only mode")
                
            elif action == 'limit_retrieval_depth':
                for breaker in target_breakers:
                    if isinstance(breaker, LightRAGCircuitBreaker):
                        breaker.config.service_specific_config['max_retrieval_depth'] = 3
                        breaker.config.service_specific_config['context_window_size'] = 2000
                
            elif action == 'reduce_ttl':
                for breaker in target_breakers:
                    if isinstance(breaker, CacheCircuitBreaker):
                        breaker.config.service_specific_config['default_ttl'] = 300  # 5 minutes
                        breaker.config.service_specific_config['max_cache_size'] = '500MB'
                
            elif action == 'disable_service':
                for breaker in target_breakers:
                    breaker.force_state(EnhancedCircuitBreakerState.OPEN, 
                                      "Degradation: Service disabled")
                
            else:
                self.logger.warning(f"Unknown degradation action: {action}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing degradation action {action}: {e}")
            return False
    
    def _update_circuit_breakers_for_degradation(self, service_type: ServiceType, level: int) -> None:
        """Update circuit breaker configurations for degradation."""
        target_breakers = [
            cb for cb in self.orchestrator._circuit_breakers.values()
            if cb.config.service_type == service_type
        ]
        
        for breaker in target_breakers:
            # Adjust thresholds based on degradation level
            if level >= 2:
                # More lenient thresholds during degradation
                breaker.current_failure_threshold = min(
                    breaker.current_failure_threshold + level,
                    breaker.config.max_failure_threshold
                )
                breaker.config.recovery_timeout = breaker.config.recovery_timeout * 1.5
            
            # Force to degraded state if not already worse
            if breaker.state == EnhancedCircuitBreakerState.CLOSED:
                breaker.force_state(EnhancedCircuitBreakerState.DEGRADED, 
                                  f"Progressive degradation level {level}")
    
    def recover_from_degradation(self, 
                                target_services: Optional[List[ServiceType]] = None,
                                recovery_level: int = 1) -> Dict[str, Any]:
        """Recover services from degradation."""
        with self._lock:
            if target_services is None:
                # Recover all currently degraded services
                target_services = [
                    cb.config.service_type for cb in self.orchestrator._circuit_breakers.values()
                    if cb.state == EnhancedCircuitBreakerState.DEGRADED
                ]
            
            recovery_result = {
                'recovered_services': [],
                'recovery_actions': [],
                'timestamp': time.time(),
                'recovery_level': recovery_level
            }
            
            for service_type in target_services:
                service_recovery = self._recover_service_from_degradation(service_type, recovery_level)
                recovery_result['recovered_services'].append(service_recovery)
            
            # Update system degradation level
            if recovery_level == self.current_degradation_level:
                self.current_degradation_level = max(0, self.current_degradation_level - 1)
            
            self.logger.info(f"Recovered {len(target_services)} services from degradation")
            
            return recovery_result
    
    def _recover_service_from_degradation(self, 
                                        service_type: ServiceType, 
                                        recovery_level: int) -> Dict[str, Any]:
        """Recover a specific service from degradation."""
        target_breakers = [
            cb for cb in self.orchestrator._circuit_breakers.values()
            if cb.config.service_type == service_type
        ]
        
        recovery_result = {
            'service_type': service_type.value,
            'recovery_level': recovery_level,
            'actions_taken': [],
            'success': False
        }
        
        try:
            # Reset service configurations to normal
            for breaker in target_breakers:
                # Reset service-specific configurations
                if isinstance(breaker, OpenAICircuitBreaker):
                    breaker.config.service_specific_config.pop('preferred_model', None)
                    breaker.config.service_specific_config['max_tokens'] = 4000
                
                elif isinstance(breaker, PerplexityCircuitBreaker):
                    breaker.config.service_specific_config['max_search_results'] = 10
                    breaker.config.service_specific_config['search_timeout'] = 30
                
                elif isinstance(breaker, LightRAGCircuitBreaker):
                    breaker.config.service_specific_config['max_retrieval_depth'] = 5
                    breaker.config.service_specific_config['context_window_size'] = 4000
                
                elif isinstance(breaker, CacheCircuitBreaker):
                    breaker.config.service_specific_config['default_ttl'] = 3600  # 1 hour
                    breaker.config.service_specific_config['max_cache_size'] = '2GB'
                
                # Reset circuit breaker thresholds
                breaker.current_failure_threshold = breaker.config.failure_threshold
                breaker.config.recovery_timeout = 60.0  # Reset to default
                
                # Transition to half-open for testing
                if breaker.state in [EnhancedCircuitBreakerState.DEGRADED, EnhancedCircuitBreakerState.OPEN]:
                    breaker.force_state(EnhancedCircuitBreakerState.HALF_OPEN, 
                                      "Recovery from degradation")
            
            recovery_result['success'] = True
            recovery_result['actions_taken'].append('reset_configurations')
            recovery_result['actions_taken'].append('reset_thresholds')
            recovery_result['actions_taken'].append('transition_to_half_open')
            
        except Exception as e:
            self.logger.error(f"Error recovering {service_type.value} from degradation: {e}")
            recovery_result['error'] = str(e)
        
        return recovery_result
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        with self._lock:
            degraded_services = []
            service_degradation_levels = {}
            
            for breaker in self.orchestrator._circuit_breakers.values():
                if breaker.state == EnhancedCircuitBreakerState.DEGRADED:
                    degraded_services.append(breaker.config.service_type.value)
                    # Estimate degradation level based on configuration changes
                    service_degradation_levels[breaker.config.service_type.value] = self._estimate_service_degradation_level(breaker)
            
            recent_degradations = [
                deg for deg in self.degradation_history
                if time.time() - deg['timestamp'] <= 3600  # Last hour
            ]
            
            return {
                'current_degradation_level': self.current_degradation_level,
                'max_degradation_level': self.max_degradation_level,
                'degraded_services': degraded_services,
                'service_degradation_levels': service_degradation_levels,
                'recent_degradations': recent_degradations,
                'total_degradation_events': len(self.degradation_history),
                'available_strategies': {
                    st.value: list(strategies.keys()) 
                    for st, strategies in self.degradation_strategies.items()
                },
                'timestamp': time.time()
            }
    
    def _estimate_service_degradation_level(self, breaker: BaseEnhancedCircuitBreaker) -> int:
        """Estimate the degradation level of a service based on its configuration."""
        # This is a simplified estimation - in a real system, you'd track this more precisely
        if breaker.state == EnhancedCircuitBreakerState.OPEN:
            return 3  # Maximum degradation
        elif breaker.state == EnhancedCircuitBreakerState.DEGRADED:
            # Check configuration changes to estimate level
            config = breaker.config.service_specific_config
            
            if isinstance(breaker, OpenAICircuitBreaker):
                if 'preferred_model' in config and config['preferred_model'] == 'gpt-4o-mini':
                    return 1 if config.get('max_tokens', 4000) > 1000 else 2
                elif config.get('max_tokens', 4000) < 2000:
                    return 2
            
            return 1  # Default to level 1 degradation
        
        return 0  # No degradation


# ============================================================================
# Integration with Existing Systems
# ============================================================================

class EnhancedCircuitBreakerIntegration:
    """Integration layer for enhanced circuit breakers with existing systems."""
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 cost_based_manager: Optional[Any] = None,  # CostCircuitBreakerManager
                 production_load_balancer: Optional[Any] = None,
                 budget_manager: Optional[Any] = None,
                 cost_persistence: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize integration layer."""
        self.config = config or {}
        self.cost_based_manager = cost_based_manager
        self.production_load_balancer = production_load_balancer
        self.budget_manager = budget_manager
        self.cost_persistence = cost_persistence
        self.logger = logger or logging.getLogger(f"{__name__}.integration")
        
        # Initialize enhanced circuit breaker system
        self.orchestrator = CircuitBreakerOrchestrator(logger)
        self.failure_analyzer = FailureCorrelationAnalyzer(self.orchestrator, logger)
        self.degradation_manager = ProgressiveDegradationManager(self.orchestrator, logger)
        
        # Service instances
        self.service_breakers: Dict[str, BaseEnhancedCircuitBreaker] = {}
        
        # Initialize service-specific circuit breakers based on config
        self._initialize_service_breakers()
        
        # Integration state
        self.integration_active = False
        self.integration_metrics: Dict[str, Any] = defaultdict(int)
    
    def _initialize_service_breakers(self):
        """Initialize service-specific circuit breakers based on configuration."""
        service_configs = {
            ServiceType.OPENAI_API: CircuitBreakerConfig(
                service_type=ServiceType.OPENAI_API,
                **self.config.get('openai_api', {})
            ),
            ServiceType.LIGHTRAG: CircuitBreakerConfig(
                service_type=ServiceType.LIGHTRAG,
                **self.config.get('lightrag', {})
            ),
            ServiceType.PERPLEXITY_API: CircuitBreakerConfig(
                service_type=ServiceType.PERPLEXITY_API,
                **self.config.get('perplexity_api', {})
            ),
            ServiceType.CACHE: CircuitBreakerConfig(
                service_type=ServiceType.CACHE,
                **self.config.get('cache', {})
            ),
        }
        
        for service_type, config in service_configs.items():
            if service_type == ServiceType.OPENAI_API:
                self.service_breakers[service_type.value] = OpenAICircuitBreaker(
                    service_type.value, config, self.logger
                )
            elif service_type == ServiceType.LIGHTRAG:
                self.service_breakers[service_type.value] = LightRAGCircuitBreaker(
                    service_type.value, config, self.logger
                )
            elif service_type == ServiceType.PERPLEXITY_API:
                self.service_breakers[service_type.value] = PerplexityCircuitBreaker(
                    service_type.value, config, self.logger
                )
            elif service_type == ServiceType.CACHE:
                self.service_breakers[service_type.value] = CacheCircuitBreaker(
                    service_type.value, config, self.logger
                )
        
        self.logger.info("Enhanced circuit breaker integration layer initialized")
    
    def get_service_breaker(self, service_type: ServiceType) -> Optional[BaseEnhancedCircuitBreaker]:
        """Get a service-specific circuit breaker."""
        return self.service_breakers.get(service_type.value)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'orchestrator_state': 'active' if self.integration_active else 'inactive',
            'service_breakers': {
                name: breaker.get_status() if hasattr(breaker, 'get_status') else {'state': 'unknown'}
                for name, breaker in self.service_breakers.items()
            },
            'system_health': {'status': 'healthy'},  # Simplified for now
            'total_protected_calls': sum(self.integration_metrics.values()),
            'total_blocked_calls': 0,  # Would need to be tracked
        }
    
    def initialize_service_breakers(self, 
                                  services_config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize service-specific circuit breakers."""
        if services_config is None:
            services_config = self._get_default_service_config()
        
        # Create OpenAI circuit breaker
        if 'openai' in services_config:
            openai_config = CircuitBreakerConfig(
                service_type=ServiceType.OPENAI_API,
                **services_config['openai']
            )
            openai_breaker = OpenAICircuitBreaker("openai_enhanced_cb", openai_config, self.logger)
            self.service_breakers['openai'] = openai_breaker
            self.orchestrator.register_circuit_breaker(openai_breaker)
        
        # Create Perplexity circuit breaker
        if 'perplexity' in services_config:
            perplexity_config = CircuitBreakerConfig(
                service_type=ServiceType.PERPLEXITY_API,
                **services_config['perplexity']
            )
            perplexity_breaker = PerplexityCircuitBreaker("perplexity_enhanced_cb", perplexity_config, self.logger)
            self.service_breakers['perplexity'] = perplexity_breaker
            self.orchestrator.register_circuit_breaker(perplexity_breaker)
        
        # Create LightRAG circuit breaker
        if 'lightrag' in services_config:
            lightrag_config = CircuitBreakerConfig(
                service_type=ServiceType.LIGHTRAG,
                **services_config['lightrag']
            )
            lightrag_breaker = LightRAGCircuitBreaker("lightrag_enhanced_cb", lightrag_config, self.logger)
            self.service_breakers['lightrag'] = lightrag_breaker
            self.orchestrator.register_circuit_breaker(
                lightrag_breaker, 
                dependencies=[ServiceType.OPENAI_API]  # LightRAG depends on OpenAI for embeddings
            )
        
        # Create Cache circuit breaker
        if 'cache' in services_config:
            cache_config = CircuitBreakerConfig(
                service_type=ServiceType.CACHE,
                **services_config['cache']
            )
            cache_breaker = CacheCircuitBreaker("cache_enhanced_cb", cache_config, self.logger)
            self.service_breakers['cache'] = cache_breaker
            self.orchestrator.register_circuit_breaker(cache_breaker)
        
        self.integration_active = True
        self.logger.info(f"Initialized {len(self.service_breakers)} enhanced circuit breakers")
    
    def _get_default_service_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default configuration for all services."""
        return {
            'openai': {
                'failure_threshold': 3,
                'recovery_timeout': 30.0,
                'enable_adaptive_thresholds': True
            },
            'perplexity': {
                'failure_threshold': 4,
                'recovery_timeout': 45.0,
                'rate_limit_window': 300.0
            },
            'lightrag': {
                'failure_threshold': 5,
                'recovery_timeout': 120.0,
                'enable_adaptive_thresholds': True
            },
            'cache': {
                'failure_threshold': 8,
                'recovery_timeout': 30.0
            }
        }
    
    def execute_with_enhanced_protection(self,
                                       service_name: str,
                                       operation: Callable,
                                       *args,
                                       **kwargs) -> Any:
        """Execute operation with enhanced circuit breaker protection."""
        if not self.integration_active:
            raise RuntimeError("Enhanced circuit breaker integration not initialized")
        
        if service_name not in self.service_breakers:
            raise ValueError(f"Unknown service: {service_name}")
        
        breaker = self.service_breakers[service_name]
        
        try:
            # Execute through enhanced circuit breaker
            result = breaker.call(operation, *args, **kwargs)
            
            # Update integration metrics
            self.integration_metrics['successful_calls'] += 1
            
            # Update service-specific metrics if available
            if hasattr(result, 'response_info'):
                self._update_service_metrics(service_name, True, result.response_info)
            
            return result
            
        except Exception as e:
            # Update integration metrics
            self.integration_metrics['failed_calls'] += 1
            
            # Update service-specific metrics
            self._update_service_metrics(service_name, False, {'error': str(e)})
            
            # Check if we need to apply system-wide measures
            self._check_system_health_after_failure(service_name, e)
            
            raise
    
    def _update_service_metrics(self, service_name: str, success: bool, response_info: Dict[str, Any]) -> None:
        """Update service-specific metrics."""
        breaker = self.service_breakers[service_name]
        
        if isinstance(breaker, OpenAICircuitBreaker):
            model_name = response_info.get('model', 'unknown')
            breaker.update_model_health(model_name, success, response_info)
            
            # Update rate limit info if available
            if 'headers' in response_info:
                breaker.update_rate_limit_status(response_info['headers'])
        
        elif isinstance(breaker, PerplexityCircuitBreaker):
            if 'query_complexity' in response_info:
                breaker.update_query_complexity('general', response_info['query_complexity'])
            
            if 'quality_score' in response_info:
                query_id = response_info.get('query_id', str(uuid.uuid4()))
                breaker.update_search_quality(query_id, response_info['quality_score'])
        
        elif isinstance(breaker, LightRAGCircuitBreaker):
            if 'retrieval_quality' in response_info:
                breaker.update_retrieval_quality(response_info['retrieval_quality'])
            
            if 'embedding_response_time' in response_info:
                breaker.update_embedding_service_status(success, response_info['embedding_response_time'])
        
        elif isinstance(breaker, CacheCircuitBreaker):
            if 'cache_operation' in response_info:
                breaker.record_cache_operation(response_info['cache_operation'], success)
    
    def _check_system_health_after_failure(self, service_name: str, exception: Exception) -> None:
        """Check system health after a failure and take appropriate action."""
        # Get current system status
        system_status = self.orchestrator.get_system_status()
        
        # Check if we need to trigger progressive degradation
        if system_status['system_error_rate'] > 0.3:  # 30% error rate
            self.degradation_manager.apply_degradation(
                degradation_level=1,
                reason=f"High system error rate after {service_name} failure"
            )
        
        # Check for cascade failure prevention
        open_breakers = system_status['circuit_breaker_states'].get('open', 0)
        if open_breakers >= 2:  # Multiple services failing
            self.degradation_manager.apply_degradation(
                degradation_level=2,
                reason="Multiple circuit breakers open - preventing cascade failure"
            )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all enhanced circuit breaker systems."""
        if not self.integration_active:
            return {'status': 'not_initialized'}
        
        return {
            'integration_active': self.integration_active,
            'integration_metrics': dict(self.integration_metrics),
            'orchestrator_status': self.orchestrator.get_system_status(),
            'degradation_status': self.degradation_manager.get_degradation_status(),
            'failure_analysis': self.failure_analyzer.analyze_failure_patterns(),
            'service_breakers': {
                name: breaker.get_status() 
                for name, breaker in self.service_breakers.items()
            },
            'timestamp': time.time()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the enhanced circuit breaker system."""
        self.logger.info("Shutting down enhanced circuit breaker integration")
        
        # Reset all circuit breakers
        for breaker in self.service_breakers.values():
            breaker.reset()
        
        self.integration_active = False
        self.logger.info("Enhanced circuit breaker integration shutdown complete")


# ============================================================================
# Factory Functions
# ============================================================================

def create_enhanced_circuit_breaker_system(
    cost_based_manager: Optional[Any] = None,
    production_load_balancer: Optional[Any] = None,
    services_config: Optional[Dict[str, Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedCircuitBreakerIntegration:
    """Factory function to create a complete enhanced circuit breaker system."""
    
    # Create integration layer
    integration = EnhancedCircuitBreakerIntegration(
        cost_based_manager=cost_based_manager,
        production_load_balancer=production_load_balancer,
        logger=logger
    )
    
    # Initialize service breakers
    integration.initialize_service_breakers(services_config)
    
    return integration


def create_service_specific_circuit_breaker(
    service_type: ServiceType,
    config: Optional[CircuitBreakerConfig] = None,
    logger: Optional[logging.Logger] = None
) -> BaseEnhancedCircuitBreaker:
    """Factory function to create service-specific circuit breakers."""
    
    if service_type == ServiceType.OPENAI_API:
        return OpenAICircuitBreaker(config=config, logger=logger)
    elif service_type == ServiceType.PERPLEXITY_API:
        return PerplexityCircuitBreaker(config=config, logger=logger)
    elif service_type == ServiceType.LIGHTRAG:
        return LightRAGCircuitBreaker(config=config, logger=logger)
    elif service_type == ServiceType.CACHE:
        return CacheCircuitBreaker(config=config, logger=logger)
    else:
        raise ValueError(f"Unsupported service type: {service_type}")


# ============================================================================
# Utility Functions
# ============================================================================

def monitor_circuit_breaker_health(integration: EnhancedCircuitBreakerIntegration,
                                 check_interval: float = 60.0) -> None:
    """Background monitoring function for circuit breaker health."""
    import threading
    import time
    
    def monitoring_loop():
        while integration.integration_active:
            try:
                # Get system status
                status = integration.get_comprehensive_status()
                
                # Check for critical issues
                orchestrator_status = status.get('orchestrator_status', {})
                if orchestrator_status.get('system_state') == 'critical':
                    integration.logger.critical("Circuit breaker system in critical state")
                
                # Run failure analysis
                failure_analysis = integration.failure_analyzer.analyze_failure_patterns()
                if failure_analysis['recommendations']:
                    integration.logger.warning(
                        f"Circuit breaker recommendations: {len(failure_analysis['recommendations'])} items"
                    )
                
                time.sleep(check_interval)
                
            except Exception as e:
                integration.logger.error(f"Error in circuit breaker monitoring: {e}")
                time.sleep(check_interval)
    
    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    
    integration.logger.info("Circuit breaker health monitoring started")