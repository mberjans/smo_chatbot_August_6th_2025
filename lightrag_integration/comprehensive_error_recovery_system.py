#!/usr/bin/env python3
"""
Comprehensive Error Recovery and Retry Logic System for Clinical Metabolomics Oracle

This module implements a sophisticated error recovery and retry system that integrates
with the existing advanced recovery system, circuit breakers, and graceful degradation.

Features:
    - Intelligent retry logic with multiple backoff strategies
    - Error classification and recovery strategy mapping  
    - State persistence for retry operations across failures
    - Integration with existing recovery systems
    - Monitoring and metrics for retry operations
    - Configuration-driven retry parameters
    - Thread-safe concurrent operation support
    - Recovery context preservation and restoration

Key Components:
    - ErrorRecoveryOrchestrator: Main coordination class
    - RetryStateManager: Persistent retry state management
    - IntelligentRetryEngine: Advanced retry logic implementation
    - RecoveryStrategyRouter: Routes errors to appropriate recovery strategies
    - RetryMetricsCollector: Monitoring and analytics for retry operations

Integration Points:
    - AdvancedRecoverySystem: Resource-aware recovery coordination
    - CostBasedCircuitBreaker: Budget-aware operation control
    - GracefulDegradationOrchestrator: System-wide degradation coordination
    - Enhanced logging and monitoring systems

Author: Claude Code (Anthropic)
Created: 2025-08-09
Version: 1.0.0
Task: CMO-LIGHTRAG-014-T06
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import random
import hashlib
import pickle

# Import existing error handling infrastructure
try:
    from .clinical_metabolomics_rag import (
        ClinicalMetabolomicsRAGError, QueryError, QueryRetryableError, 
        QueryNonRetryableError, QueryNetworkError, QueryAPIError,
        IngestionError, IngestionRetryableError, IngestionNonRetryableError,
        StorageInitializationError, CircuitBreakerError
    )
    from .advanced_recovery_system import (
        AdvancedRecoverySystem, FailureType, BackoffStrategy,
        DegradationMode, AdaptiveBackoffCalculator
    )
    from .cost_based_circuit_breaker import CostBasedCircuitBreaker
    ERROR_CLASSES_AVAILABLE = True
except ImportError:
    # Fallback for standalone operation
    ERROR_CLASSES_AVAILABLE = False
    
    class ClinicalMetabolomicsRAGError(Exception): pass
    class QueryError(ClinicalMetabolomicsRAGError): pass
    class QueryRetryableError(QueryError): pass
    class QueryNonRetryableError(QueryError): pass
    class IngestionError(ClinicalMetabolomicsRAGError): pass
    class CircuitBreakerError(Exception): pass
    
    # Mock failure types
    class FailureType(Enum):
        API_RATE_LIMIT = "api_rate_limit"
        API_TIMEOUT = "api_timeout"
        API_ERROR = "api_error"
        NETWORK_ERROR = "network_error"
        MEMORY_PRESSURE = "memory_pressure"
        PROCESSING_ERROR = "processing_error"
        RESOURCE_EXHAUSTION = "resource_exhaustion"
        
    class BackoffStrategy(Enum):
        EXPONENTIAL = "exponential"
        LINEAR = "linear"
        FIBONACCI = "fibonacci"
        ADAPTIVE = "adaptive"


class RetryStrategy(Enum):
    """Retry strategies for different error scenarios."""
    IMMEDIATE = "immediate"                    # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    LINEAR_BACKOFF = "linear_backoff"         # Linear backoff
    FIBONACCI_BACKOFF = "fibonacci_backoff"   # Fibonacci backoff
    ADAPTIVE_BACKOFF = "adaptive_backoff"     # Adaptive based on error patterns
    CIRCUIT_BREAKER = "circuit_breaker"       # Use circuit breaker logic
    DEGRADED_RETRY = "degraded_retry"         # Retry with degraded functionality
    NO_RETRY = "no_retry"                     # Don't retry


class ErrorSeverity(Enum):
    """Error severity levels for recovery prioritization."""
    CRITICAL = "critical"    # System-threatening errors
    HIGH = "high"           # Service-impacting errors
    MEDIUM = "medium"       # Feature-impacting errors
    LOW = "low"             # Minor errors
    INFO = "info"           # Informational errors


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    DEGRADE = "degrade"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    ABORT = "abort"
    CHECKPOINT = "checkpoint"
    NOTIFY = "notify"


@dataclass
class RetryAttempt:
    """Information about a single retry attempt."""
    attempt_number: int
    timestamp: datetime
    error_type: str
    error_message: str
    backoff_delay: float
    success: bool = False
    response_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryState:
    """Persistent state for retry operations."""
    operation_id: str
    operation_type: str
    operation_context: Dict[str, Any]
    
    # Retry tracking
    total_attempts: int = 0
    max_attempts: int = 3
    first_attempt_time: datetime = field(default_factory=datetime.now)
    last_attempt_time: Optional[datetime] = None
    next_retry_time: Optional[datetime] = None
    
    # Error tracking
    error_history: List[RetryAttempt] = field(default_factory=list)
    current_error_type: Optional[str] = None
    error_pattern: List[str] = field(default_factory=list)
    
    # Recovery state
    recovery_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    base_delay: float = 1.0
    max_delay: float = 300.0
    jitter_enabled: bool = True
    
    # Success tracking
    success_count: int = 0
    last_success_time: Optional[datetime] = None
    
    # Context preservation
    checkpoint_data: Optional[Dict[str, Any]] = None
    recovery_context: Dict[str, Any] = field(default_factory=dict)
    
    def add_attempt(self, attempt: RetryAttempt) -> None:
        """Add a retry attempt to the history."""
        self.error_history.append(attempt)
        self.total_attempts = len(self.error_history)
        self.last_attempt_time = attempt.timestamp
        
        # Update error pattern
        self.error_pattern.append(attempt.error_type)
        if len(self.error_pattern) > 20:  # Keep last 20 errors
            self.error_pattern = self.error_pattern[-20:]
        
        self.current_error_type = attempt.error_type
        
        if attempt.success:
            self.success_count += 1
            self.last_success_time = attempt.timestamp
    
    def should_retry(self) -> bool:
        """Check if operation should be retried."""
        if self.total_attempts >= self.max_attempts:
            return False
        
        if self.next_retry_time and datetime.now() < self.next_retry_time:
            return False
        
        return True
    
    def calculate_next_delay(self) -> float:
        """Calculate delay for next retry attempt."""
        if self.recovery_strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif self.recovery_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * self.total_attempts
        elif self.recovery_strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.base_delay * self._fibonacci(self.total_attempts)
        elif self.recovery_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** (self.total_attempts - 1))
        elif self.recovery_strategy == RetryStrategy.ADAPTIVE_BACKOFF:
            delay = self._adaptive_delay()
        else:
            delay = self.base_delay
        
        # Apply maximum delay
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter_enabled:
            jitter_factor = 0.1 + (random.random() * 0.2)  # 10-30% jitter
            delay *= (1.0 + jitter_factor)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 2:
            return 1
        a, b = 1, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
    
    def _adaptive_delay(self) -> float:
        """Calculate adaptive delay based on error patterns."""
        base_delay = self.base_delay * (self.backoff_multiplier ** (self.total_attempts - 1))
        
        # Adjust based on error pattern
        if len(self.error_pattern) >= 3:
            recent_errors = self.error_pattern[-3:]
            if len(set(recent_errors)) == 1:  # Same error type
                base_delay *= 1.5  # Increase delay for persistent errors
            
            # Check for alternating patterns
            if len(self.error_pattern) >= 6:
                pattern = self.error_pattern[-6:]
                if pattern[0::2] == pattern[0::2] and pattern[1::2] == pattern[1::2]:
                    base_delay *= 2.0  # Increase delay for alternating errors
        
        # Adjust based on success rate
        if self.total_attempts > 0:
            success_rate = self.success_count / self.total_attempts
            if success_rate < 0.3:  # Low success rate
                base_delay *= 2.0
            elif success_rate > 0.8:  # High success rate
                base_delay *= 0.7
        
        return base_delay


@dataclass
class ErrorRecoveryRule:
    """Rule for error recovery strategy selection."""
    rule_id: str
    error_patterns: List[str]  # Regex patterns to match error types/messages
    retry_strategy: RetryStrategy
    max_attempts: int = 3
    base_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 300.0
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)


class RetryStateManager:
    """Manages persistent retry state across system restarts."""
    
    def __init__(self,
                 state_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize retry state manager."""
        self.state_dir = state_dir or Path("logs/retry_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory state cache
        self._state_cache: Dict[str, RetryState] = {}
        self._cache_lock = threading.RLock()
        
        # Cleanup settings
        self._max_state_age_hours = 24
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()
        
        self.logger.info("Retry state manager initialized")
    
    def get_retry_state(self, operation_id: str) -> Optional[RetryState]:
        """Get retry state for an operation."""
        with self._cache_lock:
            # Check cache first
            if operation_id in self._state_cache:
                return self._state_cache[operation_id]
            
            # Load from disk
            state_file = self.state_dir / f"{operation_id}.pkl"
            if state_file.exists():
                try:
                    with open(state_file, 'rb') as f:
                        state = pickle.load(f)
                    
                    self._state_cache[operation_id] = state
                    return state
                    
                except Exception as e:
                    self.logger.error(f"Failed to load retry state {operation_id}: {e}")
                    
            return None
    
    def save_retry_state(self, state: RetryState) -> bool:
        """Save retry state persistently."""
        with self._cache_lock:
            try:
                # Update cache
                self._state_cache[state.operation_id] = state
                
                # Save to disk
                state_file = self.state_dir / f"{state.operation_id}.pkl"
                with open(state_file, 'wb') as f:
                    pickle.dump(state, f)
                
                # Periodic cleanup
                if time.time() - self._last_cleanup > self._cleanup_interval:
                    self._cleanup_old_states()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save retry state {state.operation_id}: {e}")
                return False
    
    def create_retry_state(self,
                          operation_id: str,
                          operation_type: str,
                          operation_context: Dict[str, Any],
                          max_attempts: int = 3,
                          recovery_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF) -> RetryState:
        """Create new retry state for an operation."""
        state = RetryState(
            operation_id=operation_id,
            operation_type=operation_type,
            operation_context=operation_context,
            max_attempts=max_attempts,
            recovery_strategy=recovery_strategy
        )
        
        self.save_retry_state(state)
        return state
    
    def delete_retry_state(self, operation_id: str) -> bool:
        """Delete retry state for completed or failed operation."""
        with self._cache_lock:
            try:
                # Remove from cache
                self._state_cache.pop(operation_id, None)
                
                # Remove from disk
                state_file = self.state_dir / f"{operation_id}.pkl"
                if state_file.exists():
                    state_file.unlink()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete retry state {operation_id}: {e}")
                return False
    
    def list_active_states(self) -> List[str]:
        """List all active retry states."""
        with self._cache_lock:
            active_states = []
            
            # Check disk files
            for state_file in self.state_dir.glob("*.pkl"):
                operation_id = state_file.stem
                state = self.get_retry_state(operation_id)
                if state and state.should_retry():
                    active_states.append(operation_id)
            
            return active_states
    
    def _cleanup_old_states(self) -> None:
        """Clean up old retry states."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self._max_state_age_hours)
            cleaned_count = 0
            
            for state_file in self.state_dir.glob("*.pkl"):
                try:
                    # Check file modification time
                    if datetime.fromtimestamp(state_file.stat().st_mtime) < cutoff_time:
                        state_file.unlink()
                        operation_id = state_file.stem
                        self._state_cache.pop(operation_id, None)
                        cleaned_count += 1
                        
                except Exception:
                    continue
            
            self._last_cleanup = time.time()
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old retry states")
                
        except Exception as e:
            self.logger.error(f"Error during retry state cleanup: {e}")


class IntelligentRetryEngine:
    """Advanced retry logic implementation with intelligent decision making."""
    
    def __init__(self,
                 state_manager: RetryStateManager,
                 recovery_rules: Optional[List[ErrorRecoveryRule]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize intelligent retry engine."""
        self.state_manager = state_manager
        self.recovery_rules = recovery_rules or self._create_default_rules()
        self.logger = logger or logging.getLogger(__name__)
        
        # Sort rules by priority
        self.recovery_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Statistics
        self._stats = {
            'total_operations': 0,
            'successful_retries': 0,
            'failed_operations': 0,
            'average_attempts': 0.0,
            'strategy_usage': defaultdict(int),
            'error_type_counts': defaultdict(int)
        }
        
        self.logger.info(f"Intelligent retry engine initialized with {len(self.recovery_rules)} rules")
    
    def _create_default_rules(self) -> List[ErrorRecoveryRule]:
        """Create default error recovery rules."""
        return [
            # API Rate Limiting
            ErrorRecoveryRule(
                rule_id="api_rate_limit",
                error_patterns=[r"rate.?limit", r"too.?many.?requests", r"quota.*exceeded"],
                retry_strategy=RetryStrategy.ADAPTIVE_BACKOFF,
                max_attempts=5,
                base_delay=10.0,
                backoff_multiplier=2.0,
                max_delay=600.0,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.DEGRADE],
                severity=ErrorSeverity.HIGH,
                priority=10
            ),
            
            # Network Errors
            ErrorRecoveryRule(
                rule_id="network_errors",
                error_patterns=[r"connection.*error", r"timeout", r"network.*unreachable"],
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=4,
                base_delay=2.0,
                backoff_multiplier=2.0,
                max_delay=120.0,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
                severity=ErrorSeverity.MEDIUM,
                priority=8
            ),
            
            # API Errors (5xx)
            ErrorRecoveryRule(
                rule_id="api_server_errors",
                error_patterns=[r"5\d\d", r"internal.*server.*error", r"service.*unavailable"],
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=5.0,
                backoff_multiplier=2.5,
                max_delay=300.0,
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK],
                severity=ErrorSeverity.HIGH,
                priority=9
            ),
            
            # Memory/Resource Issues
            ErrorRecoveryRule(
                rule_id="resource_exhaustion",
                error_patterns=[r"out.*of.*memory", r"resource.*exhausted", r"memory.*error"],
                retry_strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=2,
                base_delay=30.0,
                recovery_actions=[RecoveryAction.DEGRADE, RecoveryAction.CHECKPOINT],
                severity=ErrorSeverity.CRITICAL,
                priority=15
            ),
            
            # Circuit Breaker Errors
            ErrorRecoveryRule(
                rule_id="circuit_breaker",
                error_patterns=[r"circuit.*breaker.*open", r"circuit.*breaker.*blocked"],
                retry_strategy=RetryStrategy.NO_RETRY,
                max_attempts=0,
                recovery_actions=[RecoveryAction.FALLBACK, RecoveryAction.NOTIFY],
                severity=ErrorSeverity.HIGH,
                priority=12
            ),
            
            # Generic Retryable Errors
            ErrorRecoveryRule(
                rule_id="generic_retryable",
                error_patterns=[r"retryable", r"temporary.*error"],
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=1.0,
                backoff_multiplier=2.0,
                recovery_actions=[RecoveryAction.RETRY],
                severity=ErrorSeverity.MEDIUM,
                priority=5
            )
        ]
    
    def should_retry_operation(self,
                              operation_id: str,
                              error: Exception,
                              operation_type: str,
                              operation_context: Dict[str, Any]) -> Tuple[bool, RetryState]:
        """
        Determine if an operation should be retried and return retry state.
        
        Args:
            operation_id: Unique identifier for the operation
            error: Exception that occurred
            operation_type: Type of operation (query, ingestion, etc.)
            operation_context: Context information for the operation
            
        Returns:
            Tuple of (should_retry, retry_state)
        """
        # Get or create retry state
        retry_state = self.state_manager.get_retry_state(operation_id)
        if not retry_state:
            # Find matching rule and create state
            rule = self._find_matching_rule(error, operation_type)
            retry_state = self.state_manager.create_retry_state(
                operation_id=operation_id,
                operation_type=operation_type,
                operation_context=operation_context,
                max_attempts=rule.max_attempts if rule else 3,
                recovery_strategy=rule.retry_strategy if rule else RetryStrategy.EXPONENTIAL_BACKOFF
            )
            
            if rule:
                retry_state.base_delay = rule.base_delay
                retry_state.backoff_multiplier = rule.backoff_multiplier
                retry_state.max_delay = rule.max_delay
        
        # Record the current attempt
        attempt = RetryAttempt(
            attempt_number=retry_state.total_attempts + 1,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            backoff_delay=0.0  # Will be calculated later
        )
        
        retry_state.add_attempt(attempt)
        
        # Update statistics
        self._stats['total_operations'] += 1
        self._stats['error_type_counts'][type(error).__name__] += 1
        if retry_state.recovery_strategy:
            self._stats['strategy_usage'][retry_state.recovery_strategy.value] += 1
        
        # Check if we should retry
        should_retry = retry_state.should_retry()
        
        if should_retry:
            # Calculate next retry delay
            delay = retry_state.calculate_next_delay()
            retry_state.next_retry_time = datetime.now() + timedelta(seconds=delay)
            attempt.backoff_delay = delay
            
            self.logger.info(
                f"Operation {operation_id} will retry in {delay:.2f}s "
                f"(attempt {retry_state.total_attempts + 1}/{retry_state.max_attempts})"
            )
        else:
            self._stats['failed_operations'] += 1
            self.logger.warning(
                f"Operation {operation_id} exceeded retry limits "
                f"({retry_state.total_attempts}/{retry_state.max_attempts})"
            )
        
        # Save updated state
        self.state_manager.save_retry_state(retry_state)
        
        return should_retry, retry_state
    
    def _find_matching_rule(self, error: Exception, operation_type: str) -> Optional[ErrorRecoveryRule]:
        """Find the best matching recovery rule for an error."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        for rule in self.recovery_rules:
            # Check if any pattern matches
            for pattern in rule.error_patterns:
                import re
                if (re.search(pattern, error_message, re.IGNORECASE) or 
                    re.search(pattern, error_type, re.IGNORECASE)):
                    
                    # Check additional conditions if specified
                    if rule.conditions:
                        if not self._check_rule_conditions(rule.conditions, error, operation_type):
                            continue
                    
                    return rule
        
        return None
    
    def _check_rule_conditions(self,
                             conditions: Dict[str, Any],
                             error: Exception,
                             operation_type: str) -> bool:
        """Check if rule conditions are met."""
        # Example condition checks
        if 'operation_types' in conditions:
            if operation_type not in conditions['operation_types']:
                return False
        
        if 'error_types' in conditions:
            if type(error).__name__ not in conditions['error_types']:
                return False
        
        return True
    
    def record_operation_success(self, operation_id: str, response_time: float) -> None:
        """Record successful operation completion."""
        retry_state = self.state_manager.get_retry_state(operation_id)
        if retry_state:
            # Update the last attempt as successful
            if retry_state.error_history:
                retry_state.error_history[-1].success = True
                retry_state.error_history[-1].response_time = response_time
            
            retry_state.success_count += 1
            retry_state.last_success_time = datetime.now()
            
            # Update statistics
            if retry_state.total_attempts > 1:
                self._stats['successful_retries'] += 1
            
            # Calculate average attempts
            total_ops = self._stats['total_operations']
            if total_ops > 0:
                total_attempts = sum(
                    state.total_attempts 
                    for state in self.state_manager._state_cache.values()
                )
                self._stats['average_attempts'] = total_attempts / total_ops
            
            self.state_manager.save_retry_state(retry_state)
            
            self.logger.info(
                f"Operation {operation_id} succeeded after {retry_state.total_attempts} attempts"
            )
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get retry operation statistics."""
        return {
            'statistics': self._stats.copy(),
            'active_retries': len(self.state_manager.list_active_states()),
            'recovery_rules': len(self.recovery_rules),
            'timestamp': datetime.now().isoformat()
        }


class RecoveryStrategyRouter:
    """Routes errors to appropriate recovery strategies and coordinates actions."""
    
    def __init__(self,
                 retry_engine: IntelligentRetryEngine,
                 advanced_recovery: Optional['AdvancedRecoverySystem'] = None,
                 circuit_breaker_manager: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize recovery strategy router."""
        self.retry_engine = retry_engine
        self.advanced_recovery = advanced_recovery
        self.circuit_breaker_manager = circuit_breaker_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Recovery action handlers
        self._action_handlers = {
            RecoveryAction.RETRY: self._handle_retry_action,
            RecoveryAction.DEGRADE: self._handle_degrade_action,
            RecoveryAction.FALLBACK: self._handle_fallback_action,
            RecoveryAction.CIRCUIT_BREAK: self._handle_circuit_break_action,
            RecoveryAction.CHECKPOINT: self._handle_checkpoint_action,
            RecoveryAction.ESCALATE: self._handle_escalate_action,
            RecoveryAction.ABORT: self._handle_abort_action,
            RecoveryAction.NOTIFY: self._handle_notify_action
        }
        
        self.logger.info("Recovery strategy router initialized")
    
    def route_error_recovery(self,
                           operation_id: str,
                           error: Exception,
                           operation_type: str,
                           operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route error to appropriate recovery strategy.
        
        Args:
            operation_id: Unique operation identifier
            error: Exception that occurred
            operation_type: Type of operation
            operation_context: Operation context
            
        Returns:
            Recovery result dictionary
        """
        try:
            # Determine retry strategy
            should_retry, retry_state = self.retry_engine.should_retry_operation(
                operation_id, error, operation_type, operation_context
            )
            
            # Find recovery rule for additional actions
            rule = self.retry_engine._find_matching_rule(error, operation_type)
            
            recovery_result = {
                'operation_id': operation_id,
                'should_retry': should_retry,
                'retry_state': asdict(retry_state),
                'recovery_actions_taken': [],
                'next_retry_time': retry_state.next_retry_time.isoformat() if retry_state.next_retry_time else None,
                'error_severity': rule.severity.value if rule else ErrorSeverity.MEDIUM.value,
                'recovery_rule': rule.rule_id if rule else None
            }
            
            # Execute recovery actions
            if rule and rule.recovery_actions:
                for action in rule.recovery_actions:
                    try:
                        action_result = self._execute_recovery_action(
                            action, operation_id, error, operation_type, 
                            operation_context, retry_state
                        )
                        recovery_result['recovery_actions_taken'].append({
                            'action': action.value,
                            'result': action_result
                        })
                    except Exception as action_error:
                        self.logger.error(
                            f"Recovery action {action.value} failed for {operation_id}: {action_error}"
                        )
                        recovery_result['recovery_actions_taken'].append({
                            'action': action.value,
                            'result': {'success': False, 'error': str(action_error)}
                        })
            
            return recovery_result
            
        except Exception as e:
            self.logger.error(f"Error routing recovery for {operation_id}: {e}")
            return {
                'operation_id': operation_id,
                'should_retry': False,
                'error': str(e),
                'recovery_actions_taken': []
            }
    
    def _execute_recovery_action(self,
                               action: RecoveryAction,
                               operation_id: str,
                               error: Exception,
                               operation_type: str,
                               operation_context: Dict[str, Any],
                               retry_state: RetryState) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        handler = self._action_handlers.get(action)
        if handler:
            return handler(operation_id, error, operation_type, operation_context, retry_state)
        else:
            return {'success': False, 'error': f'No handler for action {action.value}'}
    
    def _handle_retry_action(self, operation_id: str, error: Exception, 
                           operation_type: str, operation_context: Dict[str, Any], 
                           retry_state: RetryState) -> Dict[str, Any]:
        """Handle retry recovery action."""
        return {
            'success': True,
            'action': 'retry_scheduled',
            'next_retry_time': retry_state.next_retry_time.isoformat() if retry_state.next_retry_time else None,
            'attempt_number': retry_state.total_attempts + 1
        }
    
    def _handle_degrade_action(self, operation_id: str, error: Exception,
                             operation_type: str, operation_context: Dict[str, Any],
                             retry_state: RetryState) -> Dict[str, Any]:
        """Handle degradation recovery action."""
        if self.advanced_recovery:
            try:
                # Map error to failure type for advanced recovery system
                failure_type = self._map_error_to_failure_type(error)
                recovery_strategy = self.advanced_recovery.handle_failure(
                    failure_type, str(error), operation_id, operation_context
                )
                return {
                    'success': True,
                    'action': 'degradation_applied',
                    'degradation_mode': self.advanced_recovery.current_degradation_mode.value,
                    'recovery_strategy': recovery_strategy
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': 'Advanced recovery system not available'}
    
    def _handle_fallback_action(self, operation_id: str, error: Exception,
                              operation_type: str, operation_context: Dict[str, Any],
                              retry_state: RetryState) -> Dict[str, Any]:
        """Handle fallback recovery action."""
        # Implement fallback logic here
        return {
            'success': True,
            'action': 'fallback_initiated',
            'fallback_strategy': 'graceful_degradation'
        }
    
    def _handle_circuit_break_action(self, operation_id: str, error: Exception,
                                   operation_type: str, operation_context: Dict[str, Any],
                                   retry_state: RetryState) -> Dict[str, Any]:
        """Handle circuit breaker recovery action."""
        if self.circuit_breaker_manager:
            try:
                # Force circuit breaker to open for this operation type
                breaker_name = f"{operation_type}_breaker"
                circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(breaker_name)
                if circuit_breaker:
                    circuit_breaker.force_open(f"Recovery action triggered by {operation_id}")
                    return {
                        'success': True,
                        'action': 'circuit_breaker_opened',
                        'breaker_name': breaker_name
                    }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Circuit breaker manager not available'}
    
    def _handle_checkpoint_action(self, operation_id: str, error: Exception,
                                operation_type: str, operation_context: Dict[str, Any],
                                retry_state: RetryState) -> Dict[str, Any]:
        """Handle checkpoint creation recovery action."""
        try:
            # Create checkpoint data
            checkpoint_data = {
                'operation_id': operation_id,
                'operation_type': operation_type,
                'operation_context': operation_context,
                'error_state': {
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'timestamp': datetime.now().isoformat()
                },
                'retry_state': asdict(retry_state)
            }
            
            # Save checkpoint
            retry_state.checkpoint_data = checkpoint_data
            self.retry_engine.state_manager.save_retry_state(retry_state)
            
            return {
                'success': True,
                'action': 'checkpoint_created',
                'checkpoint_id': operation_id
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_escalate_action(self, operation_id: str, error: Exception,
                              operation_type: str, operation_context: Dict[str, Any],
                              retry_state: RetryState) -> Dict[str, Any]:
        """Handle error escalation recovery action."""
        # Log escalation
        self.logger.critical(
            f"Error escalated for operation {operation_id}: {error}",
            extra={
                'operation_type': operation_type,
                'error_count': retry_state.total_attempts,
                'error_pattern': retry_state.error_pattern
            }
        )
        
        return {
            'success': True,
            'action': 'error_escalated',
            'escalation_level': 'critical'
        }
    
    def _handle_abort_action(self, operation_id: str, error: Exception,
                           operation_type: str, operation_context: Dict[str, Any],
                           retry_state: RetryState) -> Dict[str, Any]:
        """Handle operation abort recovery action."""
        # Mark operation as permanently failed
        self.retry_engine.state_manager.delete_retry_state(operation_id)
        
        return {
            'success': True,
            'action': 'operation_aborted',
            'reason': 'unrecoverable_error'
        }
    
    def _handle_notify_action(self, operation_id: str, error: Exception,
                            operation_type: str, operation_context: Dict[str, Any],
                            retry_state: RetryState) -> Dict[str, Any]:
        """Handle notification recovery action."""
        # Send notification (implementation depends on notification system)
        notification_message = (
            f"Operation {operation_id} ({operation_type}) failed after "
            f"{retry_state.total_attempts} attempts: {error}"
        )
        
        self.logger.warning(notification_message)
        
        return {
            'success': True,
            'action': 'notification_sent',
            'message': notification_message
        }
    
    def _map_error_to_failure_type(self, error: Exception) -> FailureType:
        """Map exception to failure type for advanced recovery system."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        if 'rate' in error_message and 'limit' in error_message:
            return FailureType.API_RATE_LIMIT
        elif 'timeout' in error_message:
            return FailureType.API_TIMEOUT
        elif 'network' in error_message or 'connection' in error_message:
            return FailureType.NETWORK_ERROR
        elif 'memory' in error_message:
            return FailureType.MEMORY_PRESSURE
        elif 'api' in error_message:
            return FailureType.API_ERROR
        else:
            return FailureType.PROCESSING_ERROR


class RetryMetricsCollector:
    """Collects and analyzes retry operation metrics."""
    
    def __init__(self,
                 state_manager: RetryStateManager,
                 logger: Optional[logging.Logger] = None):
        """Initialize retry metrics collector."""
        self.state_manager = state_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Metrics storage
        self._metrics_history = deque(maxlen=10000)
        self._current_metrics = defaultdict(float)
        self._lock = threading.RLock()
        
        # Periodic metrics calculation
        self._metrics_interval = 300  # 5 minutes
        self._last_metrics_update = time.time()
        
        self.logger.info("Retry metrics collector initialized")
    
    def record_retry_attempt(self,
                           operation_id: str,
                           operation_type: str,
                           error_type: str,
                           attempt_number: int,
                           backoff_delay: float,
                           success: bool) -> None:
        """Record a retry attempt for metrics."""
        with self._lock:
            metric_record = {
                'timestamp': time.time(),
                'operation_id': operation_id,
                'operation_type': operation_type,
                'error_type': error_type,
                'attempt_number': attempt_number,
                'backoff_delay': backoff_delay,
                'success': success
            }
            
            self._metrics_history.append(metric_record)
            
            # Update current metrics
            self._current_metrics['total_attempts'] += 1
            if success:
                self._current_metrics['successful_attempts'] += 1
            self._current_metrics[f'attempts_by_type_{operation_type}'] += 1
            self._current_metrics[f'errors_by_type_{error_type}'] += 1
    
    def get_retry_metrics(self) -> Dict[str, Any]:
        """Get comprehensive retry metrics."""
        with self._lock:
            now = time.time()
            
            # Update metrics if needed
            if now - self._last_metrics_update > self._metrics_interval:
                self._calculate_derived_metrics()
                self._last_metrics_update = now
            
            # Recent metrics (last hour)
            recent_cutoff = now - 3600
            recent_metrics = [
                m for m in self._metrics_history
                if m['timestamp'] > recent_cutoff
            ]
            
            # Calculate success rates
            total_recent = len(recent_metrics)
            successful_recent = sum(1 for m in recent_metrics if m['success'])
            
            # Error type distribution
            error_distribution = defaultdict(int)
            for metric in recent_metrics:
                error_distribution[metric['error_type']] += 1
            
            # Average backoff delays by error type
            backoff_by_error = defaultdict(list)
            for metric in recent_metrics:
                backoff_by_error[metric['error_type']].append(metric['backoff_delay'])
            
            average_backoffs = {
                error_type: sum(delays) / len(delays) if delays else 0
                for error_type, delays in backoff_by_error.items()
            }
            
            return {
                'current_metrics': dict(self._current_metrics),
                'recent_metrics': {
                    'total_attempts': total_recent,
                    'successful_attempts': successful_recent,
                    'success_rate': successful_recent / max(total_recent, 1),
                    'error_distribution': dict(error_distribution),
                    'average_backoff_by_error': average_backoffs
                },
                'active_retry_states': len(self.state_manager.list_active_states()),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from history."""
        if not self._metrics_history:
            return
        
        # Calculate moving averages, trends, etc.
        recent_window = list(self._metrics_history)[-1000:]  # Last 1000 entries
        
        # Average attempts per operation
        operations = defaultdict(list)
        for metric in recent_window:
            operations[metric['operation_id']].append(metric['attempt_number'])
        
        if operations:
            avg_attempts = sum(
                max(attempts) for attempts in operations.values()
            ) / len(operations)
            self._current_metrics['average_attempts_per_operation'] = avg_attempts
        
        # Success rate trends
        if len(recent_window) > 100:
            first_half = recent_window[:len(recent_window)//2]
            second_half = recent_window[len(recent_window)//2:]
            
            first_success_rate = sum(1 for m in first_half if m['success']) / len(first_half)
            second_success_rate = sum(1 for m in second_half if m['success']) / len(second_half)
            
            self._current_metrics['success_rate_trend'] = second_success_rate - first_success_rate


class ErrorRecoveryOrchestrator:
    """Main orchestrator for comprehensive error recovery and retry logic."""
    
    def __init__(self,
                 state_dir: Optional[Path] = None,
                 recovery_rules: Optional[List[ErrorRecoveryRule]] = None,
                 advanced_recovery: Optional['AdvancedRecoverySystem'] = None,
                 circuit_breaker_manager: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize error recovery orchestrator.
        
        Args:
            state_dir: Directory for persistent retry state storage
            recovery_rules: Custom recovery rules
            advanced_recovery: Advanced recovery system integration
            circuit_breaker_manager: Circuit breaker manager integration
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.state_manager = RetryStateManager(state_dir, self.logger)
        self.retry_engine = IntelligentRetryEngine(
            self.state_manager, recovery_rules, self.logger
        )
        self.strategy_router = RecoveryStrategyRouter(
            self.retry_engine, advanced_recovery, circuit_breaker_manager, self.logger
        )
        self.metrics_collector = RetryMetricsCollector(
            self.state_manager, self.logger
        )
        
        # Integration references
        self.advanced_recovery = advanced_recovery
        self.circuit_breaker_manager = circuit_breaker_manager
        
        # Orchestrator statistics
        self._orchestrator_stats = {
            'operations_handled': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'start_time': datetime.now().isoformat()
        }
        
        self.logger.info("Error recovery orchestrator initialized")
    
    def handle_operation_error(self,
                             operation_id: str,
                             error: Exception,
                             operation_type: str,
                             operation_context: Dict[str, Any],
                             operation_callable: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Main entry point for handling operation errors.
        
        Args:
            operation_id: Unique operation identifier
            error: Exception that occurred
            operation_type: Type of operation (query, ingestion, etc.)
            operation_context: Context information for the operation
            operation_callable: Optional callable to retry
            
        Returns:
            Recovery result dictionary
        """
        try:
            self._orchestrator_stats['operations_handled'] += 1
            
            # Route error through recovery strategy
            recovery_result = self.strategy_router.route_error_recovery(
                operation_id, error, operation_type, operation_context
            )
            
            # Record metrics
            retry_state = self.state_manager.get_retry_state(operation_id)
            if retry_state:
                self.metrics_collector.record_retry_attempt(
                    operation_id=operation_id,
                    operation_type=operation_type,
                    error_type=type(error).__name__,
                    attempt_number=retry_state.total_attempts,
                    backoff_delay=retry_state.error_history[-1].backoff_delay if retry_state.error_history else 0.0,
                    success=False
                )
            
            # Execute automatic retry if enabled and callable provided
            if (recovery_result.get('should_retry', False) and 
                operation_callable and 
                retry_state):
                
                retry_result = self._execute_retry(
                    operation_id, operation_callable, retry_state, operation_context
                )
                recovery_result['retry_execution'] = retry_result
            
            # Update statistics
            if recovery_result.get('should_retry', False):
                self._orchestrator_stats['successful_recoveries'] += 1
            else:
                self._orchestrator_stats['failed_recoveries'] += 1
            
            return recovery_result
            
        except Exception as orchestration_error:
            self.logger.error(
                f"Error in recovery orchestration for {operation_id}: {orchestration_error}"
            )
            return {
                'operation_id': operation_id,
                'orchestration_error': str(orchestration_error),
                'should_retry': False,
                'recovery_actions_taken': []
            }
    
    def _execute_retry(self,
                      operation_id: str,
                      operation_callable: Callable,
                      retry_state: RetryState,
                      operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automatic retry with proper delay."""
        try:
            # Wait for retry delay
            if retry_state.next_retry_time:
                delay = (retry_state.next_retry_time - datetime.now()).total_seconds()
                if delay > 0:
                    time.sleep(delay)
            
            # Execute operation
            start_time = time.time()
            result = operation_callable(**operation_context)
            execution_time = time.time() - start_time
            
            # Record success
            self.retry_engine.record_operation_success(operation_id, execution_time)
            self.metrics_collector.record_retry_attempt(
                operation_id=operation_id,
                operation_type=retry_state.operation_type,
                error_type='success',
                attempt_number=retry_state.total_attempts,
                backoff_delay=0.0,
                success=True
            )
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'total_attempts': retry_state.total_attempts
            }
            
        except Exception as retry_error:
            # Handle retry failure
            return {
                'success': False,
                'error': str(retry_error),
                'error_type': type(retry_error).__name__
            }
    
    def recover_operation(self,
                         operation_id: str,
                         operation_callable: Callable,
                         operation_context: Dict[str, Any]) -> Any:
        """
        Recover a failed operation using stored retry state.
        
        Args:
            operation_id: Operation identifier
            operation_callable: Function to execute
            operation_context: Operation context
            
        Returns:
            Operation result
            
        Raises:
            Exception: If operation cannot be recovered
        """
        retry_state = self.state_manager.get_retry_state(operation_id)
        if not retry_state:
            raise ValueError(f"No retry state found for operation {operation_id}")
        
        if not retry_state.should_retry():
            raise RuntimeError(f"Operation {operation_id} is not eligible for retry")
        
        # Execute retry
        retry_result = self._execute_retry(
            operation_id, operation_callable, retry_state, operation_context
        )
        
        if retry_result['success']:
            return retry_result['result']
        else:
            # Handle the retry error
            retry_error = Exception(retry_result['error'])
            recovery_result = self.handle_operation_error(
                operation_id, retry_error, retry_state.operation_type, operation_context
            )
            
            if recovery_result.get('should_retry', False):
                # Recursive retry
                return self.recover_operation(operation_id, operation_callable, operation_context)
            else:
                raise RuntimeError(f"Operation {operation_id} recovery failed: {retry_result['error']}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'orchestrator_statistics': self._orchestrator_stats.copy(),
            'retry_engine_statistics': self.retry_engine.get_retry_statistics(),
            'retry_metrics': self.metrics_collector.get_retry_metrics(),
            'active_retry_states': self.state_manager.list_active_states(),
            'recovery_rules_count': len(self.retry_engine.recovery_rules),
            'integrations': {
                'advanced_recovery': self.advanced_recovery is not None,
                'circuit_breaker_manager': self.circuit_breaker_manager is not None
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_completed_operations(self) -> Dict[str, int]:
        """Clean up completed retry states."""
        active_states = self.state_manager.list_active_states()
        all_states = []
        
        # Find all states
        for state_file in self.state_manager.state_dir.glob("*.pkl"):
            all_states.append(state_file.stem)
        
        # Clean up inactive states
        cleaned_count = 0
        for operation_id in all_states:
            if operation_id not in active_states:
                if self.state_manager.delete_retry_state(operation_id):
                    cleaned_count += 1
        
        return {
            'total_states': len(all_states),
            'active_states': len(active_states),
            'cleaned_states': cleaned_count
        }
    
    def close(self) -> None:
        """Clean shutdown of orchestrator."""
        try:
            self.logger.info("Shutting down error recovery orchestrator")
            # Components don't need explicit cleanup
        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}")


# Factory function for easy initialization
def create_error_recovery_orchestrator(
    state_dir: Optional[Path] = None,
    recovery_rules: Optional[List[ErrorRecoveryRule]] = None,
    advanced_recovery: Optional['AdvancedRecoverySystem'] = None,
    circuit_breaker_manager: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> ErrorRecoveryOrchestrator:
    """
    Factory function to create a configured error recovery orchestrator.
    
    Args:
        state_dir: Directory for retry state persistence
        recovery_rules: Custom recovery rules
        advanced_recovery: Advanced recovery system integration
        circuit_breaker_manager: Circuit breaker manager integration
        logger: Logger instance
        
    Returns:
        Configured ErrorRecoveryOrchestrator instance
    """
    return ErrorRecoveryOrchestrator(
        state_dir=state_dir or Path("logs/error_recovery"),
        recovery_rules=recovery_rules,
        advanced_recovery=advanced_recovery,
        circuit_breaker_manager=circuit_breaker_manager,
        logger=logger or logging.getLogger(__name__)
    )


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator
    orchestrator = create_error_recovery_orchestrator()
    
    # Example operation that might fail
    def example_operation(data: str) -> str:
        if random.random() < 0.7:  # 70% failure rate for testing
            raise Exception("Random failure for testing")
        return f"Processed: {data}"
    
    # Test error recovery
    operation_id = str(uuid.uuid4())
    operation_context = {"data": "test_data"}
    
    try:
        # Simulate operation failure
        result = example_operation(operation_context["data"])
        print(f"Operation succeeded: {result}")
    except Exception as e:
        # Handle error through orchestrator
        recovery_result = orchestrator.handle_operation_error(
            operation_id=operation_id,
            error=e,
            operation_type="example_operation",
            operation_context=operation_context,
            operation_callable=lambda **ctx: example_operation(ctx["data"])
        )
        
        print(f"Recovery result: {recovery_result}")
    
    # Get system status
    status = orchestrator.get_system_status()
    print(f"System status: {json.dumps(status, indent=2, default=str)}")
    
    orchestrator.close()