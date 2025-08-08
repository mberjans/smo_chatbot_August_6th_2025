"""
Enhanced LLM-Powered Query Classifier for Clinical Metabolomics Oracle

This module provides a production-ready LLM-based semantic classifier that integrates 
seamlessly with the existing Clinical Metabolomics Oracle infrastructure while adding 
advanced semantic understanding capabilities optimized for <2 second response times.

Key Features:
    - Circuit breaker patterns for API failure resilience
    - Advanced caching with LRU and TTL optimization for <2s response times
    - Comprehensive cost tracking and budget management with alerting
    - Intelligent fallback mechanisms and graceful degradation
    - Performance monitoring with real-time optimization recommendations
    - Full compatibility with existing ClassificationResult and RoutingPrediction structures
    - Async context management with timeout handling
    - Token optimization strategies for cost efficiency

Classes:
    - EnhancedLLMQueryClassifier: Main production-ready LLM classification engine
    - CircuitBreaker: API failure protection and automatic recovery
    - IntelligentCache: Advanced caching with LRU/TTL and performance optimization
    - CostManager: Budget tracking, alerting, and optimization
    - PerformanceMonitor: Real-time performance tracking and recommendations

Author: Claude Code (Anthropic)
Version: 2.0.0
Created: 2025-08-08
"""

import json
import time
import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import contextlib
import threading
from pathlib import Path

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    class AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenAI library not available")

# Import existing components for integration
try:
    from .llm_classification_prompts import (
        LLMClassificationPrompts,
        ClassificationCategory,
        ClassificationResult,
        CLASSIFICATION_RESULT_SCHEMA
    )
    from .query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction
    from .research_categorizer import CategoryPrediction
    from .cost_persistence import ResearchCategory
    from .query_classification_system import QueryClassificationCategories, ClassificationResult as SystemClassificationResult
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}. Some features may be limited.")


# ============================================================================
# ENHANCED CONFIGURATION CLASSES
# ============================================================================

class LLMProvider(Enum):
    """Supported LLM providers for classification."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker protection."""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before attempting recovery
    success_threshold: int = 2  # Successful calls needed to close circuit
    timeout_threshold: float = 10.0  # Timeout threshold for considering failure
    
    
@dataclass 
class CacheConfig:
    """Advanced caching configuration."""
    enable_caching: bool = True
    max_cache_size: int = 2000
    ttl_seconds: int = 3600  # 1 hour default
    lru_threshold: float = 0.8  # When to start LRU eviction
    performance_tracking: bool = True
    cache_warming: bool = True
    adaptive_ttl: bool = True  # Adjust TTL based on query patterns
    

@dataclass
class CostConfig:
    """Cost management configuration."""
    daily_budget: float = 5.0  # USD
    hourly_budget: float = 0.5  # USD
    cost_per_1k_tokens: Dict[str, float] = None
    enable_budget_alerts: bool = True
    budget_warning_threshold: float = 0.8  # 80% of budget
    automatic_budget_reset: bool = True
    cost_optimization: bool = True


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    target_response_time_ms: float = 2000.0  # <2 second target
    enable_monitoring: bool = True
    sample_rate: float = 1.0  # Track 100% of requests
    performance_window_size: int = 100  # Last N requests for averaging
    auto_optimization: bool = True
    benchmark_frequency: int = 50  # Run benchmarks every N requests


@dataclass
class EnhancedLLMConfig:
    """Comprehensive configuration for enhanced LLM classification."""
    
    # LLM Provider Settings
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    max_tokens: int = 200
    temperature: float = 0.1
    
    # Performance Settings  
    timeout_seconds: float = 1.5  # Aggressive timeout for <2s target
    max_retries: int = 2
    fallback_to_keywords: bool = True
    parallel_processing: bool = True
    
    # Prompt Strategy
    use_adaptive_prompts: bool = True
    confidence_threshold: float = 0.7
    validation_threshold: float = 0.5
    enable_prompt_caching: bool = True
    
    # Component Configurations
    circuit_breaker: CircuitBreakerConfig = None
    cache: CacheConfig = None  
    cost: CostConfig = None
    performance: PerformanceConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.circuit_breaker is None:
            self.circuit_breaker = CircuitBreakerConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.cost is None:
            self.cost = CostConfig()
            if self.cost.cost_per_1k_tokens is None:
                self.cost.cost_per_1k_tokens = {
                    "gpt-4o-mini": 0.0005,
                    "gpt-4o": 0.015,
                    "gpt-3.5-turbo": 0.0015
                }
        if self.performance is None:
            self.performance = PerformanceConfig()


# ============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# ============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, redirect to fallback
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for API calls with automatic recovery.
    Prevents cascading failures and enables graceful degradation.
    """
    
    def __init__(self, config: CircuitBreakerConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.total_calls = 0
        self.failed_calls = 0
        self.circuit_opens = 0
        
    def can_proceed(self) -> bool:
        """Check if call can proceed based on circuit state."""
        with self.lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if (time.time() - self.last_failure_time) > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker entering HALF_OPEN state for recovery attempt")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self, response_time: float = None):
        """Record a successful call."""
        with self.lock:
            self.total_calls += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info("Circuit breaker CLOSED - service recovered")
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, response_time: float = None):
        """Record a failed call."""
        with self.lock:
            self.total_calls += 1
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.circuit_opens += 1
                    self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                self.logger.warning("Circuit breaker returned to OPEN state after half-open failure")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls": self.total_calls,
                "failed_calls": self.failed_calls,
                "success_rate": (self.total_calls - self.failed_calls) / max(1, self.total_calls),
                "circuit_opens": self.circuit_opens,
                "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None
            }


# ============================================================================
# INTELLIGENT CACHING IMPLEMENTATION
# ============================================================================

class CacheEntry:
    """Enhanced cache entry with metadata."""
    
    def __init__(self, result: Any, ttl_seconds: int = 3600):
        self.result = result
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.access_count = 1
        self.last_accessed = self.created_at
        self.hit_rate = 1.0
        
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at
        
    def access(self):
        """Record access to this entry."""
        self.access_count += 1
        self.last_accessed = time.time()
        
    def extend_ttl(self, extension_seconds: int):
        """Extend TTL for frequently accessed entries."""
        self.expires_at += extension_seconds


class IntelligentCache:
    """
    Advanced caching system with LRU eviction, adaptive TTL, and performance optimization.
    Designed for <2 second response time targets.
    """
    
    def __init__(self, config: CacheConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache storage using OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_removals = 0
        
        # Adaptive TTL tracking
        self._access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
    def _get_cache_key(self, query_text: str, context: Dict[str, Any] = None) -> str:
        """Generate optimized cache key."""
        # Include relevant context in key for better hit rates
        key_data = {
            "query": query_text.lower().strip(),
            "context_hash": self._hash_context(context) if context else None
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash of relevant context elements."""
        # Only include relevant context fields that affect classification
        relevant_keys = ["user_id", "session_id", "domain", "source"]
        relevant_context = {k: v for k, v in context.items() if k in relevant_keys}
        return hashlib.md5(json.dumps(relevant_context, sort_keys=True).encode()).hexdigest()[:8]
    
    def get(self, query_text: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result with intelligent access tracking."""
        if not self.config.enable_caching:
            return None
            
        cache_key = self._get_cache_key(query_text, context)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired():
                self._cache.pop(cache_key, None)
                self.expired_removals += 1
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.access()
            self.hits += 1
            
            # Track access pattern for adaptive TTL
            if self.config.adaptive_ttl:
                self._access_patterns[cache_key].append(time.time())
                self._adjust_ttl_if_needed(cache_key, entry)
            
            self.logger.debug(f"Cache HIT for query hash: {cache_key[:8]}...")
            return entry.result
    
    def put(self, query_text: str, result: Any, context: Dict[str, Any] = None, 
            custom_ttl: int = None):
        """Cache result with intelligent eviction."""
        if not self.config.enable_caching:
            return
            
        cache_key = self._get_cache_key(query_text, context)
        ttl = custom_ttl or self.config.ttl_seconds
        
        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            # Check if eviction needed
            while len(self._cache) >= self.config.max_cache_size:
                self._evict_lru_entry()
            
            # Add new entry
            entry = CacheEntry(result, ttl)
            self._cache[cache_key] = entry
            
            self.logger.debug(f"Cache SET for query hash: {cache_key[:8]}... (TTL: {ttl}s)")
    
    def _evict_lru_entry(self):
        """Evict least recently used entry."""
        if self._cache:
            evicted_key, _ = self._cache.popitem(last=False)  # Remove oldest (first)
            self.evictions += 1
            self.logger.debug(f"Cache evicted LRU entry: {evicted_key[:8]}...")
    
    def _adjust_ttl_if_needed(self, cache_key: str, entry: CacheEntry):
        """Adjust TTL based on access patterns."""
        access_times = list(self._access_patterns[cache_key])
        
        if len(access_times) >= 3:
            # Calculate access frequency
            time_span = access_times[-1] - access_times[0]
            if time_span > 0:
                access_frequency = len(access_times) / time_span  # accesses per second
                
                # Extend TTL for frequently accessed entries
                if access_frequency > 0.01:  # More than 1 access per 100 seconds
                    extension = min(3600, self.config.ttl_seconds // 2)  # Extend by up to half TTL
                    entry.extend_ttl(extension)
                    self.logger.debug(f"Extended TTL for frequently accessed entry: {cache_key[:8]}")
    
    def clear_expired(self) -> int:
        """Remove expired entries and return count."""
        removed_count = 0
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
            self.expired_removals += removed_count
        
        if removed_count > 0:
            self.logger.debug(f"Cleared {removed_count} expired cache entries")
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                "cache_size": len(self._cache),
                "max_size": self.config.max_cache_size,
                "utilization": len(self._cache) / self.config.max_cache_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "expired_removals": self.expired_removals,
                "ttl_seconds": self.config.ttl_seconds,
                "adaptive_ttl": self.config.adaptive_ttl
            }
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Analyze and optimize cache performance."""
        stats = self.get_stats()
        recommendations = []
        
        # Hit rate analysis
        if stats["hit_rate"] < 0.3:
            recommendations.append({
                "type": "hit_rate",
                "issue": f"Low hit rate ({stats['hit_rate']:.1%})",
                "suggestion": "Consider increasing cache size or adjusting TTL"
            })
        
        # Utilization analysis
        if stats["utilization"] > 0.9:
            recommendations.append({
                "type": "utilization", 
                "issue": f"High cache utilization ({stats['utilization']:.1%})",
                "suggestion": "Consider increasing max_cache_size to reduce evictions"
            })
        
        # Eviction analysis
        if stats["evictions"] > stats["hits"] * 0.1:
            recommendations.append({
                "type": "evictions",
                "issue": f"High eviction rate ({stats['evictions']} evictions vs {stats['hits']} hits)",
                "suggestion": "Increase cache size or optimize TTL settings"
            })
        
        return {
            "current_stats": stats,
            "recommendations": recommendations,
            "cache_health": "good" if len(recommendations) <= 1 else "needs_attention"
        }


# ============================================================================
# COST MANAGEMENT IMPLEMENTATION  
# ============================================================================

class CostManager:
    """
    Comprehensive cost tracking and budget management with alerting.
    """
    
    def __init__(self, config: CostConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Cost tracking
        self.daily_cost = 0.0
        self.hourly_cost = 0.0
        self.total_tokens = 0
        self.request_count = 0
        
        # Budget tracking
        self.last_daily_reset = datetime.now().date()
        self.last_hourly_reset = datetime.now().hour
        
        # Alerting
        self.budget_warnings_sent = set()
        
        # Cost optimization
        self.model_costs = self.config.cost_per_1k_tokens or {}
        
    def can_make_request(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if request is within budget constraints."""
        self._reset_budgets_if_needed()
        
        # Check daily budget
        if self.daily_cost + estimated_cost > self.config.daily_budget:
            return False, f"Daily budget exceeded (${self.daily_cost:.4f} + ${estimated_cost:.4f} > ${self.config.daily_budget})"
        
        # Check hourly budget
        if self.hourly_cost + estimated_cost > self.config.hourly_budget:
            return False, f"Hourly budget exceeded (${self.hourly_cost:.4f} + ${estimated_cost:.4f} > ${self.config.hourly_budget})"
        
        return True, "Within budget"
    
    def record_request(self, model: str, prompt_tokens: int, completion_tokens: int, 
                      actual_cost: float = None):
        """Record API request cost."""
        self._reset_budgets_if_needed()
        
        # Calculate cost if not provided
        if actual_cost is None:
            cost_per_1k = self.model_costs.get(model, 0.001)  # Default fallback
            total_tokens = prompt_tokens + completion_tokens
            actual_cost = (total_tokens / 1000.0) * cost_per_1k
        
        # Update counters
        self.daily_cost += actual_cost
        self.hourly_cost += actual_cost
        self.total_tokens += prompt_tokens + completion_tokens
        self.request_count += 1
        
        # Check for budget alerts
        if self.config.enable_budget_alerts:
            self._check_budget_alerts()
        
        self.logger.debug(f"Recorded cost: ${actual_cost:.6f} for {prompt_tokens + completion_tokens} tokens")
    
    def _reset_budgets_if_needed(self):
        """Reset budget counters if time periods have passed."""
        now = datetime.now()
        
        # Reset daily budget
        if now.date() != self.last_daily_reset:
            self.daily_cost = 0.0
            self.last_daily_reset = now.date()
            self.budget_warnings_sent.clear()
            self.logger.info("Daily budget reset")
        
        # Reset hourly budget
        if now.hour != self.last_hourly_reset:
            self.hourly_cost = 0.0
            self.last_hourly_reset = now.hour
            self.logger.debug("Hourly budget reset")
    
    def _check_budget_alerts(self):
        """Check and send budget alerts if needed."""
        daily_utilization = self.daily_cost / self.config.daily_budget
        hourly_utilization = self.hourly_cost / self.config.hourly_budget
        
        # Daily budget warning
        if (daily_utilization >= self.config.budget_warning_threshold and 
            "daily_warning" not in self.budget_warnings_sent):
            self.budget_warnings_sent.add("daily_warning")
            self.logger.warning(f"Daily budget warning: {daily_utilization:.1%} utilization "
                              f"(${self.daily_cost:.4f}/${self.config.daily_budget})")
        
        # Hourly budget warning
        if (hourly_utilization >= self.config.budget_warning_threshold and 
            "hourly_warning" not in self.budget_warnings_sent):
            self.budget_warnings_sent.add("hourly_warning")
            self.logger.warning(f"Hourly budget warning: {hourly_utilization:.1%} utilization "
                              f"(${self.hourly_cost:.4f}/${self.config.hourly_budget})")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        self._reset_budgets_if_needed()
        
        return {
            "daily_cost": self.daily_cost,
            "daily_budget": self.config.daily_budget,
            "daily_utilization": self.daily_cost / self.config.daily_budget,
            "daily_remaining": self.config.daily_budget - self.daily_cost,
            
            "hourly_cost": self.hourly_cost, 
            "hourly_budget": self.config.hourly_budget,
            "hourly_utilization": self.hourly_cost / self.config.hourly_budget,
            "hourly_remaining": self.config.hourly_budget - self.hourly_cost,
            
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "avg_cost_per_request": self.daily_cost / max(1, self.request_count),
            
            "last_daily_reset": self.last_daily_reset.isoformat(),
            "budget_warnings_sent": list(self.budget_warnings_sent)
        }
    
    def estimate_cost(self, model: str, estimated_tokens: int) -> float:
        """Estimate cost for a request."""
        cost_per_1k = self.model_costs.get(model, 0.001)
        return (estimated_tokens / 1000.0) * cost_per_1k
    
    def optimize_model_selection(self, query_complexity: str) -> str:
        """Suggest optimal model based on cost and complexity."""
        if not self.config.cost_optimization:
            return "gpt-4o-mini"  # Default
        
        # Simple model selection logic based on complexity
        if query_complexity == "simple":
            return "gpt-4o-mini"  # Most cost-effective
        elif query_complexity == "complex":
            # Check if we can afford the better model
            budget_remaining = self.config.daily_budget - self.daily_cost
            if budget_remaining > 0.10:  # Save some budget
                return "gpt-4o"
            else:
                return "gpt-4o-mini"
        else:
            return "gpt-4o-mini"  # Default to cost-effective


# ============================================================================
# PERFORMANCE MONITORING IMPLEMENTATION
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    response_times: deque = None
    confidence_scores: deque = None
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    fallback_count: int = 0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = deque(maxlen=100)
        if self.confidence_scores is None:
            self.confidence_scores = deque(maxlen=100)


class PerformanceMonitor:
    """
    Real-time performance monitoring with optimization recommendations.
    """
    
    def __init__(self, config: PerformanceConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        
        # Performance thresholds
        self.target_response_time = config.target_response_time_ms
        self.performance_degradation_threshold = self.target_response_time * 1.5
        
        # Optimization tracking
        self.optimization_suggestions = []
        self.last_benchmark_time = 0
        
    def record_request(self, response_time_ms: float, confidence: float, 
                      success: bool, used_fallback: bool, timed_out: bool = False):
        """Record request performance metrics."""
        if not self.config.enable_monitoring:
            return
        
        # Update metrics
        self.metrics.response_times.append(response_time_ms)
        self.metrics.confidence_scores.append(confidence)
        
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.failure_count += 1
            
        if used_fallback:
            self.metrics.fallback_count += 1
            
        if timed_out:
            self.metrics.timeout_count += 1
        
        # Check for performance issues
        if self.config.auto_optimization:
            self._check_performance_issues(response_time_ms)
        
        # Run periodic benchmarks
        if (self.metrics.success_count + self.metrics.failure_count) % self.config.benchmark_frequency == 0:
            self._run_performance_benchmark()
    
    def _check_performance_issues(self, response_time_ms: float):
        """Check for performance issues and generate suggestions."""
        # Response time check
        if response_time_ms > self.performance_degradation_threshold:
            suggestion = {
                "type": "response_time",
                "issue": f"Response time {response_time_ms:.1f}ms exceeds degradation threshold",
                "suggestion": "Consider using fallback prompts or caching optimization",
                "timestamp": time.time()
            }
            self.optimization_suggestions.append(suggestion)
    
    def _run_performance_benchmark(self):
        """Run performance benchmark and generate recommendations."""
        current_time = time.time()
        if current_time - self.last_benchmark_time < 300:  # Don't benchmark more than every 5 minutes
            return
            
        self.last_benchmark_time = current_time
        
        stats = self.get_performance_stats()
        
        # Generate recommendations based on current performance
        if stats["avg_response_time"] > self.target_response_time:
            self.optimization_suggestions.append({
                "type": "benchmark",
                "issue": f"Average response time ({stats['avg_response_time']:.1f}ms) exceeds target",
                "suggestion": "Consider optimizing prompts, increasing cache hit rate, or using faster model",
                "timestamp": current_time
            })
        
        # Keep only recent suggestions (last hour)
        cutoff_time = current_time - 3600
        self.optimization_suggestions = [
            s for s in self.optimization_suggestions 
            if s["timestamp"] > cutoff_time
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_requests = self.metrics.success_count + self.metrics.failure_count
        
        if not self.metrics.response_times:
            return {
                "total_requests": total_requests,
                "avg_response_time": 0,
                "p95_response_time": 0,
                "success_rate": 0,
                "avg_confidence": 0,
                "target_response_time": self.target_response_time
            }
        
        response_times = list(self.metrics.response_times)
        confidence_scores = list(self.metrics.confidence_scores)
        
        # Calculate percentiles
        response_times_sorted = sorted(response_times)
        p95_index = int(0.95 * len(response_times_sorted))
        p99_index = int(0.99 * len(response_times_sorted))
        
        return {
            "total_requests": total_requests,
            "success_count": self.metrics.success_count,
            "failure_count": self.metrics.failure_count,
            "timeout_count": self.metrics.timeout_count,
            "fallback_count": self.metrics.fallback_count,
            
            "avg_response_time": sum(response_times) / len(response_times),
            "median_response_time": response_times_sorted[len(response_times_sorted) // 2],
            "p95_response_time": response_times_sorted[p95_index] if p95_index < len(response_times_sorted) else 0,
            "p99_response_time": response_times_sorted[p99_index] if p99_index < len(response_times_sorted) else 0,
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            
            "success_rate": self.metrics.success_count / max(1, total_requests),
            "fallback_rate": self.metrics.fallback_count / max(1, total_requests),
            "timeout_rate": self.metrics.timeout_count / max(1, total_requests),
            
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            
            "target_response_time": self.target_response_time,
            "target_compliance_rate": len([t for t in response_times if t <= self.target_response_time]) / len(response_times),
            
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        return self.optimization_suggestions[-10:]  # Last 10 suggestions


# ============================================================================
# ENHANCED LLM QUERY CLASSIFIER
# ============================================================================

class EnhancedLLMQueryClassifier:
    """
    Production-ready LLM-powered query classifier with advanced features:
    - Circuit breaker protection
    - Intelligent caching with LRU/TTL
    - Comprehensive cost management
    - Performance monitoring and optimization
    - Full integration with existing infrastructure
    """
    
    def __init__(self, 
                 config: EnhancedLLMConfig,
                 biomedical_router: Optional[BiomedicalQueryRouter] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the enhanced LLM query classifier.
        
        Args:
            config: Enhanced configuration object
            biomedical_router: Existing biomedical router for fallback
            logger: Logger instance
        """
        self.config = config
        self.biomedical_router = biomedical_router
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker, self.logger)
        self.cache = IntelligentCache(config.cache, self.logger)
        self.cost_manager = CostManager(config.cost, self.logger)
        self.performance_monitor = PerformanceMonitor(config.performance, self.logger)
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Classification state
        self.classification_count = 0
        self.initialization_time = time.time()
        
        self.logger.info(f"Enhanced LLM Query Classifier initialized with {config.provider.value} provider")
        self.logger.info(f"Target response time: {config.performance.target_response_time_ms}ms")
        self.logger.info(f"Daily budget: ${config.cost.daily_budget}")
    
    def _init_llm_client(self) -> None:
        """Initialize the LLM client based on provider configuration."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required but not available")
        
        if self.config.provider == LLMProvider.OPENAI:
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required")
            
            self.llm_client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds
            )
        else:
            raise NotImplementedError(f"Provider {self.config.provider.value} not yet implemented")
    
    async def classify_query(self, 
                           query_text: str,
                           context: Optional[Dict[str, Any]] = None,
                           force_llm: bool = False,
                           priority: str = "normal") -> Tuple[ClassificationResult, Dict[str, Any]]:
        """
        Classify a query with comprehensive error handling and optimization.
        
        Args:
            query_text: The query text to classify
            context: Optional context information
            force_llm: If True, skip cache and circuit breaker
            priority: Request priority ("low", "normal", "high")
            
        Returns:
            Tuple of (ClassificationResult, metadata)
        """
        start_time = time.time()
        self.classification_count += 1
        metadata = {
            "classification_id": self.classification_count,
            "start_time": start_time,
            "used_llm": False,
            "used_cache": False,
            "used_fallback": False,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "cost_estimate": 0.0,
            "priority": priority
        }
        
        try:
            # Check cache first (unless forced)
            if not force_llm:
                cached_result = self.cache.get(query_text, context)
                if cached_result:
                    metadata["used_cache"] = True
                    metadata["response_time_ms"] = (time.time() - start_time) * 1000
                    
                    self.performance_monitor.record_request(
                        metadata["response_time_ms"], cached_result.confidence,
                        True, False
                    )
                    
                    self.logger.debug(f"Cache hit for query: {query_text[:50]}...")
                    return cached_result, metadata
            
            # Check circuit breaker (unless forced or high priority)
            if not force_llm and priority != "high" and not self.circuit_breaker.can_proceed():
                metadata["used_fallback"] = True
                metadata["fallback_reason"] = "circuit_breaker_open"
                
                fallback_result = await self._fallback_classification(query_text, context)
                metadata["response_time_ms"] = (time.time() - start_time) * 1000
                
                self.performance_monitor.record_request(
                    metadata["response_time_ms"], fallback_result.confidence,
                    True, True
                )
                
                return fallback_result, metadata
            
            # Attempt LLM classification
            llm_result = await self._classify_with_llm(query_text, context, metadata)
            
            if llm_result:
                metadata["used_llm"] = True
                metadata["response_time_ms"] = (time.time() - start_time) * 1000
                
                # Record success
                self.circuit_breaker.record_success(metadata["response_time_ms"])
                self.performance_monitor.record_request(
                    metadata["response_time_ms"], llm_result.confidence,
                    True, False
                )
                
                # Cache successful result
                self.cache.put(query_text, llm_result, context)
                
                self.logger.debug(f"LLM classification successful: {llm_result.category}")
                return llm_result, metadata
            
        except asyncio.TimeoutError:
            metadata["error"] = "timeout"
            self.circuit_breaker.record_failure()
            self.performance_monitor.record_request(
                (time.time() - start_time) * 1000, 0.0, False, False, True
            )
            
        except Exception as e:
            metadata["error"] = str(e)
            self.circuit_breaker.record_failure()
            self.performance_monitor.record_request(
                (time.time() - start_time) * 1000, 0.0, False, False
            )
            self.logger.error(f"LLM classification failed: {str(e)}")
        
        # Fallback to keyword-based classification
        metadata["used_fallback"] = True
        metadata["fallback_reason"] = metadata.get("error", "llm_failure")
        
        fallback_result = await self._fallback_classification(query_text, context)
        metadata["response_time_ms"] = (time.time() - start_time) * 1000
        
        self.performance_monitor.record_request(
            metadata["response_time_ms"], fallback_result.confidence,
            True, True
        )
        
        return fallback_result, metadata
    
    async def _classify_with_llm(self, 
                                query_text: str,
                                context: Optional[Dict[str, Any]],
                                metadata: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Perform LLM-based classification with comprehensive error handling."""
        
        # Estimate cost and check budget
        estimated_tokens = self._estimate_tokens(query_text)
        estimated_cost = self.cost_manager.estimate_cost(self.config.model_name, estimated_tokens)
        metadata["cost_estimate"] = estimated_cost
        
        can_proceed, budget_message = self.cost_manager.can_make_request(estimated_cost)
        if not can_proceed:
            self.logger.warning(f"Budget check failed: {budget_message}")
            raise Exception(f"Budget exceeded: {budget_message}")
        
        # Select appropriate prompt strategy
        prompt_strategy = self._select_prompt_strategy(query_text, context)
        prompt = self._build_optimized_prompt(query_text, prompt_strategy)
        
        # Make API call with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature
                    ),
                    timeout=self.config.timeout_seconds
                )
                
                # Record actual cost
                if response.usage:
                    self.cost_manager.record_request(
                        self.config.model_name,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                    metadata["actual_tokens"] = response.usage.prompt_tokens + response.usage.completion_tokens
                    metadata["actual_cost"] = self.cost_manager.estimate_cost(
                        self.config.model_name, metadata["actual_tokens"]
                    )
                
                # Parse and validate response
                result = self._parse_and_validate_response(response.choices[0].message.content)
                return result
                
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries:
                    raise
                self.logger.warning(f"LLM request timeout on attempt {attempt + 1}")
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise
                self.logger.warning(f"LLM request failed on attempt {attempt + 1}: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        return None
    
    def _estimate_tokens(self, query_text: str) -> int:
        """Estimate token count for cost calculation."""
        # Simple estimation: ~4 characters per token
        base_prompt_tokens = 800  # Approximate prompt template size
        query_tokens = len(query_text) // 4
        response_tokens = 150  # Typical response size
        
        return base_prompt_tokens + query_tokens + response_tokens
    
    def _select_prompt_strategy(self, query_text: str, context: Optional[Dict[str, Any]]) -> str:
        """Select optimal prompt strategy based on query characteristics."""
        if not self.config.use_adaptive_prompts:
            return "standard"
        
        # Analyze query complexity
        query_length = len(query_text.split())
        
        if query_length < 5:
            return "simple"  # Use simplified prompt for short queries
        elif query_length > 20:
            return "detailed"  # Use detailed prompt for complex queries
        else:
            return "standard"  # Use standard prompt
    
    def _build_optimized_prompt(self, query_text: str, strategy: str) -> str:
        """Build optimized prompt based on strategy."""
        try:
            # Import prompts (with fallback)
            from .llm_classification_prompts import LLMClassificationPrompts
            
            if strategy == "simple":
                return LLMClassificationPrompts.build_fallback_prompt(query_text)
            else:
                return LLMClassificationPrompts.build_primary_prompt(query_text)
                
        except ImportError:
            # Fallback to basic prompt if import fails
            return self._build_basic_prompt(query_text)
    
    def _build_basic_prompt(self, query_text: str) -> str:
        """Build basic prompt as fallback."""
        return f"""
Classify this query into one of three categories:
- KNOWLEDGE_GRAPH: for established relationships and mechanisms
- REAL_TIME: for current/recent information  
- GENERAL: for basic definitions and explanations

Query: "{query_text}"

Respond with JSON:
{{"category": "KNOWLEDGE_GRAPH|REAL_TIME|GENERAL", "confidence": 0.8, "reasoning": "explanation"}}
"""
    
    def _parse_and_validate_response(self, response_text: str) -> ClassificationResult:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            result_data = json.loads(response_text)
            
            # Validate required fields
            if "category" not in result_data or "confidence" not in result_data:
                raise ValueError("Missing required fields in response")
            
            # Validate category
            valid_categories = ["KNOWLEDGE_GRAPH", "REAL_TIME", "GENERAL"]
            if result_data["category"] not in valid_categories:
                raise ValueError(f"Invalid category: {result_data['category']}")
            
            # Validate confidence
            confidence = float(result_data["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be 0-1, got {confidence}")
            
            # Set defaults for optional fields
            result_data.setdefault("reasoning", "LLM classification")
            result_data.setdefault("alternative_categories", [])
            result_data.setdefault("uncertainty_indicators", [])
            result_data.setdefault("biomedical_signals", {
                "entities": [], "relationships": [], "techniques": []
            })
            result_data.setdefault("temporal_signals", {
                "keywords": [], "patterns": [], "years": []
            })
            
            return ClassificationResult(**result_data)
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            self.logger.error(f"Response text: {response_text}")
            raise ValueError(f"Invalid LLM response format: {e}")
    
    async def _fallback_classification(self, 
                                     query_text: str,
                                     context: Optional[Dict[str, Any]]) -> ClassificationResult:
        """Fallback to keyword-based classification."""
        
        if self.biomedical_router:
            # Use existing biomedical router
            routing_prediction = self.biomedical_router.route_query(query_text, context)
            return self._convert_routing_to_classification(routing_prediction)
        else:
            # Simple pattern-based fallback
            return self._simple_pattern_classification(query_text)
    
    def _convert_routing_to_classification(self, routing_prediction: RoutingPrediction) -> ClassificationResult:
        """Convert routing prediction to classification result."""
        
        # Map routing decisions to categories
        category_mapping = {
            RoutingDecision.LIGHTRAG: "KNOWLEDGE_GRAPH",
            RoutingDecision.PERPLEXITY: "REAL_TIME", 
            RoutingDecision.EITHER: "GENERAL",
            RoutingDecision.HYBRID: "GENERAL"
        }
        
        category = category_mapping.get(routing_prediction.routing_decision, "GENERAL")
        
        return ClassificationResult(
            category=category,
            confidence=routing_prediction.confidence,
            reasoning=f"Keyword-based fallback: {', '.join(routing_prediction.reasoning[:2])}",
            alternative_categories=[],
            uncertainty_indicators=["fallback_classification"],
            biomedical_signals={
                "entities": routing_prediction.knowledge_indicators or [],
                "relationships": [],
                "techniques": []
            },
            temporal_signals={
                "keywords": routing_prediction.temporal_indicators or [],
                "patterns": [],
                "years": []
            }
        )
    
    def _simple_pattern_classification(self, query_text: str) -> ClassificationResult:
        """Simple pattern-based classification as last resort."""
        query_lower = query_text.lower()
        
        # Check for temporal indicators
        temporal_patterns = ["latest", "recent", "current", "2024", "2025", "new", "breaking"]
        if any(pattern in query_lower for pattern in temporal_patterns):
            return ClassificationResult(
                category="REAL_TIME",
                confidence=0.6,
                reasoning="Simple pattern: temporal indicators detected",
                alternative_categories=[],
                uncertainty_indicators=["simple_fallback"],
                biomedical_signals={"entities": [], "relationships": [], "techniques": []},
                temporal_signals={"keywords": temporal_patterns, "patterns": [], "years": []}
            )
        
        # Check for relationship patterns
        relationship_patterns = ["relationship", "connection", "pathway", "mechanism", "between"]
        if any(pattern in query_lower for pattern in relationship_patterns):
            return ClassificationResult(
                category="KNOWLEDGE_GRAPH",
                confidence=0.6,
                reasoning="Simple pattern: relationship indicators detected",
                alternative_categories=[],
                uncertainty_indicators=["simple_fallback"],
                biomedical_signals={"entities": [], "relationships": relationship_patterns, "techniques": []},
                temporal_signals={"keywords": [], "patterns": [], "years": []}
            )
        
        # Default to general
        return ClassificationResult(
            category="GENERAL",
            confidence=0.4,
            reasoning="Simple pattern: default classification",
            alternative_categories=[],
            uncertainty_indicators=["simple_fallback", "low_confidence"],
            biomedical_signals={"entities": [], "relationships": [], "techniques": []},
            temporal_signals={"keywords": [], "patterns": [], "years": []}
        )
    
    # ============================================================================
    # MONITORING AND OPTIMIZATION METHODS
    # ============================================================================
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "classification_stats": {
                "total_classifications": self.classification_count,
                "uptime_seconds": time.time() - self.initialization_time,
                "classifications_per_minute": (self.classification_count / max(1, (time.time() - self.initialization_time) / 60))
            },
            "circuit_breaker_stats": self.circuit_breaker.get_stats(),
            "cache_stats": self.cache.get_stats(),
            "cost_stats": self.cost_manager.get_budget_status(),
            "performance_stats": self.performance_monitor.get_performance_stats(),
            "configuration": {
                "provider": self.config.provider.value,
                "model_name": self.config.model_name,
                "target_response_time_ms": self.config.performance.target_response_time_ms,
                "daily_budget": self.config.cost.daily_budget
            }
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations."""
        
        # Collect recommendations from all components
        all_recommendations = []
        
        # Circuit breaker recommendations
        cb_stats = self.circuit_breaker.get_stats()
        if cb_stats["success_rate"] < 0.95:
            all_recommendations.append({
                "component": "circuit_breaker",
                "type": "reliability",
                "issue": f"Low success rate ({cb_stats['success_rate']:.1%})",
                "suggestion": "Review API reliability, increase timeout, or adjust failure threshold",
                "priority": "high"
            })
        
        # Cache recommendations
        cache_analysis = self.cache.optimize_cache()
        for rec in cache_analysis.get("recommendations", []):
            rec["component"] = "cache"
            rec["priority"] = "medium"
            all_recommendations.append(rec)
        
        # Performance recommendations
        perf_recs = self.performance_monitor.get_optimization_recommendations()
        for rec in perf_recs:
            rec["component"] = "performance"
            rec["priority"] = "high" if "response_time" in rec.get("type", "") else "medium"
            all_recommendations.append(rec)
        
        # Cost recommendations
        budget_status = self.cost_manager.get_budget_status()
        if budget_status["daily_utilization"] > 0.8:
            all_recommendations.append({
                "component": "cost",
                "type": "budget",
                "issue": f"High daily budget utilization ({budget_status['daily_utilization']:.1%})",
                "suggestion": "Consider increasing cache hit rate, using cheaper model, or optimizing prompts",
                "priority": "medium"
            })
        
        # Overall system health
        health_score = self._calculate_health_score()
        
        return {
            "overall_health": health_score["status"],
            "health_score": health_score["score"],
            "recommendations": sorted(all_recommendations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True),
            "optimization_summary": self._generate_optimization_summary(all_recommendations)
        }
    
    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        
        scores = []
        
        # Circuit breaker health (30% weight)
        cb_stats = self.circuit_breaker.get_stats()
        cb_score = min(1.0, cb_stats["success_rate"]) * 0.3
        scores.append(cb_score)
        
        # Performance health (40% weight)
        perf_stats = self.performance_monitor.get_performance_stats()
        target_compliance = perf_stats.get("target_compliance_rate", 0)
        perf_score = min(1.0, target_compliance) * 0.4
        scores.append(perf_score)
        
        # Cache health (15% weight)
        cache_stats = self.cache.get_stats()
        cache_score = min(1.0, cache_stats["hit_rate"]) * 0.15
        scores.append(cache_score)
        
        # Cost health (15% weight)
        budget_stats = self.cost_manager.get_budget_status()
        cost_score = max(0.0, 1.0 - budget_stats["daily_utilization"]) * 0.15
        scores.append(cost_score)
        
        total_score = sum(scores)
        
        if total_score >= 0.9:
            status = "excellent"
        elif total_score >= 0.7:
            status = "good"
        elif total_score >= 0.5:
            status = "fair"
        else:
            status = "needs_attention"
        
        return {
            "score": total_score,
            "status": status,
            "component_scores": {
                "circuit_breaker": cb_score / 0.3,
                "performance": perf_score / 0.4,
                "cache": cache_score / 0.15,
                "cost": cost_score / 0.15
            }
        }
    
    def _generate_optimization_summary(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate human-readable optimization summary."""
        
        if not recommendations:
            return "System is operating optimally with no immediate optimization needs."
        
        high_priority = len([r for r in recommendations if r["priority"] == "high"])
        medium_priority = len([r for r in recommendations if r["priority"] == "medium"])
        
        summary_parts = []
        
        if high_priority > 0:
            summary_parts.append(f"{high_priority} high-priority optimizations needed")
        
        if medium_priority > 0:
            summary_parts.append(f"{medium_priority} medium-priority improvements suggested")
        
        if not summary_parts:
            return "Minor optimization opportunities identified."
        
        return ". ".join(summary_parts) + "."
    
    async def optimize_system(self, auto_apply: bool = False) -> Dict[str, Any]:
        """Perform system optimization based on current performance."""
        
        optimization_results = {
            "actions_taken": [],
            "recommendations_pending": [],
            "performance_impact": {}
        }
        
        # Get current recommendations
        recommendations = self.get_optimization_recommendations()
        
        for rec in recommendations["recommendations"]:
            if auto_apply and rec["priority"] == "high":
                # Apply automatic optimizations for high-priority issues
                if rec["component"] == "cache" and "cache_size" in rec.get("suggestion", ""):
                    # Automatically increase cache size if recommended
                    old_size = self.config.cache.max_cache_size
                    self.config.cache.max_cache_size = min(5000, int(old_size * 1.5))
                    optimization_results["actions_taken"].append({
                        "action": "increased_cache_size",
                        "old_value": old_size,
                        "new_value": self.config.cache.max_cache_size
                    })
                
                elif rec["component"] == "performance" and "timeout" in rec.get("suggestion", ""):
                    # Automatically adjust timeout if performance is poor
                    old_timeout = self.config.timeout_seconds
                    self.config.timeout_seconds = min(3.0, old_timeout * 1.2)
                    optimization_results["actions_taken"].append({
                        "action": "increased_timeout",
                        "old_value": old_timeout,
                        "new_value": self.config.timeout_seconds
                    })
                
            else:
                optimization_results["recommendations_pending"].append(rec)
        
        return optimization_results


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

async def create_enhanced_llm_classifier(
    config: Optional[EnhancedLLMConfig] = None,
    api_key: Optional[str] = None,
    biomedical_router: Optional[BiomedicalQueryRouter] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedLLMQueryClassifier:
    """
    Factory function to create an enhanced LLM query classifier.
    
    Args:
        config: Enhanced configuration object
        api_key: OpenAI API key (overrides config)
        biomedical_router: Existing biomedical router for fallback
        logger: Logger instance
        
    Returns:
        Configured EnhancedLLMQueryClassifier instance
    """
    
    if config is None:
        config = EnhancedLLMConfig()
    
    if api_key:
        config.api_key = api_key
    
    if not config.api_key:
        import os
        config.api_key = os.getenv('OPENAI_API_KEY')
    
    if not config.api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config.")
    
    # Create biomedical router if not provided
    if biomedical_router is None:
        try:
            biomedical_router = BiomedicalQueryRouter(logger)
        except Exception as e:
            logger.warning(f"Could not create biomedical router: {e}")
    
    classifier = EnhancedLLMQueryClassifier(config, biomedical_router, logger)
    
    if logger:
        logger.info("Enhanced LLM query classifier created successfully")
        logger.info(f"Configuration: {config.provider.value} with {config.model_name}")
        logger.info(f"Performance target: {config.performance.target_response_time_ms}ms")
    
    return classifier


def convert_enhanced_result_to_routing_prediction(
    classification_result: ClassificationResult,
    metadata: Dict[str, Any],
    query_text: str
) -> RoutingPrediction:
    """
    Convert enhanced LLM classification result to RoutingPrediction for compatibility
    with existing infrastructure.
    
    Args:
        classification_result: Enhanced LLM classification result
        metadata: Classification metadata
        query_text: Original query text
        
    Returns:
        RoutingPrediction compatible with existing routing system
    """
    
    # Map categories to routing decisions
    category_mapping = {
        "KNOWLEDGE_GRAPH": RoutingDecision.LIGHTRAG,
        "REAL_TIME": RoutingDecision.PERPLEXITY,
        "GENERAL": RoutingDecision.EITHER
    }
    
    routing_decision = category_mapping.get(classification_result.category, RoutingDecision.EITHER)
    
    # Create reasoning list
    reasoning = [classification_result.reasoning]
    
    if metadata.get("used_llm"):
        reasoning.append("Enhanced LLM-powered semantic classification")
    elif metadata.get("used_cache"):
        reasoning.append("Cached LLM classification result")
    else:
        reasoning.append("Keyword-based fallback classification")
    
    # Add performance information
    if "response_time_ms" in metadata:
        reasoning.append(f"Response time: {metadata['response_time_ms']:.1f}ms")
    
    # Map to research category
    research_category_mapping = {
        "KNOWLEDGE_GRAPH": ResearchCategory.KNOWLEDGE_EXTRACTION,
        "REAL_TIME": ResearchCategory.LITERATURE_SEARCH,
        "GENERAL": ResearchCategory.GENERAL_QUERY
    }
    
    research_category = research_category_mapping.get(classification_result.category, ResearchCategory.GENERAL_QUERY)
    
    # Create enhanced confidence metrics
    try:
        from .query_router import ConfidenceMetrics
        
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=classification_result.confidence,
            research_category_confidence=classification_result.confidence,
            temporal_analysis_confidence=0.9 if classification_result.temporal_signals["keywords"] else 0.3,
            signal_strength_confidence=0.9 if classification_result.biomedical_signals["entities"] else 0.3,
            context_coherence_confidence=classification_result.confidence,
            keyword_density=len(classification_result.biomedical_signals["entities"]) / max(1, len(query_text.split())) * 10,
            pattern_match_strength=0.9 if classification_result.biomedical_signals["relationships"] else 0.3,
            biomedical_entity_count=len(classification_result.biomedical_signals["entities"]),
            ambiguity_score=len(classification_result.uncertainty_indicators) * 0.2,
            conflict_score=0.1 if classification_result.alternative_categories else 0.0,
            alternative_interpretations=[
                (category_mapping.get(alt.get("category"), RoutingDecision.EITHER), alt.get("confidence", 0.0))
                for alt in classification_result.alternative_categories
            ],
            calculation_time_ms=metadata.get("response_time_ms", 0.0)
        )
    except ImportError:
        confidence_metrics = None
    
    return RoutingPrediction(
        routing_decision=routing_decision,
        confidence=classification_result.confidence,
        reasoning=reasoning,
        research_category=research_category,
        confidence_metrics=confidence_metrics,
        temporal_indicators=classification_result.temporal_signals["keywords"],
        knowledge_indicators=classification_result.biomedical_signals["entities"],
        metadata={
            "enhanced_llm_classification": True,
            "classification_metadata": metadata,
            "biomedical_signals": classification_result.biomedical_signals,
            "temporal_signals": classification_result.temporal_signals,
            "uncertainty_indicators": classification_result.uncertainty_indicators
        }
    )


# ============================================================================
# ASYNC CONTEXT MANAGERS
# ============================================================================

@contextlib.asynccontextmanager
async def llm_classifier_context(config: EnhancedLLMConfig, 
                                 biomedical_router: Optional[BiomedicalQueryRouter] = None):
    """
    Async context manager for LLM classifier with proper resource management.
    
    Usage:
        async with llm_classifier_context(config) as classifier:
            result, metadata = await classifier.classify_query("example query")
    """
    
    logger = logging.getLogger(__name__)
    classifier = None
    
    try:
        classifier = await create_enhanced_llm_classifier(config, None, biomedical_router, logger)
        logger.info("LLM classifier context initialized")
        yield classifier
        
    finally:
        if classifier:
            # Cleanup operations
            stats = classifier.get_comprehensive_stats()
            logger.info(f"LLM classifier context cleanup - Total classifications: {stats['classification_stats']['total_classifications']}")
            logger.info(f"Final performance - Cache hit rate: {stats['cache_stats']['hit_rate']:.1%}")
            logger.info(f"Final costs - Daily spend: ${stats['cost_stats']['daily_cost']:.4f}")


if __name__ == "__main__":
    # Example usage
    import asyncio
    import os
    
    async def demo():
        config = EnhancedLLMConfig(
            api_key=os.getenv('OPENAI_API_KEY', 'demo-key'),
            performance=PerformanceConfig(target_response_time_ms=1500),
            cost=CostConfig(daily_budget=2.0)
        )
        
        async with llm_classifier_context(config) as classifier:
            # Demo classification
            result, metadata = await classifier.classify_query(
                "What is the relationship between glucose metabolism and insulin signaling?"
            )
            
            print(f"Classification: {result.category}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Used LLM: {metadata['used_llm']}")
            print(f"Response time: {metadata.get('response_time_ms', 0):.1f}ms")
            
            # Show system stats
            stats = classifier.get_comprehensive_stats()
            print(f"System health: {classifier.get_optimization_recommendations()['overall_health']}")
    
    # Run demo if executed directly
    if os.getenv('OPENAI_API_KEY'):
        asyncio.run(demo())
    else:
        print("Set OPENAI_API_KEY environment variable to run demo")