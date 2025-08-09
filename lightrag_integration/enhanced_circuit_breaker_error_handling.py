"""
Enhanced Circuit Breaker Error Handling Integration
=================================================

This module provides enhanced error handling that works with both traditional and enhanced
circuit breaker states. It provides unified error handling, recovery strategies, and
fallback mechanisms that are aware of the different circuit breaker states.

Classes:
    - EnhancedCircuitBreakerErrorHandler: Unified error handling for circuit breakers
    - CircuitBreakerStateMapper: Maps between traditional and enhanced states
    - EnhancedErrorRecoveryStrategy: Advanced recovery strategies
    - CircuitBreakerErrorAnalyzer: Analyzes circuit breaker errors for patterns

Features:
- Unified error handling for both traditional and enhanced circuit breakers
- Intelligent state mapping and translation
- Progressive recovery strategies based on circuit breaker state
- Error pattern analysis and prediction
- Integration with existing fallback systems
- Backward compatibility with existing error handling

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: CMO-LIGHTRAG-014-T04 - Enhanced Circuit Breaker Error Handling
Version: 1.0.0
"""

import logging
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque

# Import existing error types
from .clinical_metabolomics_rag import CircuitBreakerError

# Enhanced circuit breaker imports
try:
    from .enhanced_circuit_breaker_system import (
        EnhancedCircuitBreakerState,
        ServiceType,
        CircuitBreakerOrchestrator
    )
    ENHANCED_CIRCUIT_BREAKERS_AVAILABLE = True
except ImportError:
    ENHANCED_CIRCUIT_BREAKERS_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels for circuit breaker errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different circuit breaker states."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FALLBACK_TO_CACHE = "fallback_to_cache"
    FALLBACK_TO_ALTERNATIVE_SERVICE = "fallback_to_alternative_service"
    WAIT_FOR_RECOVERY = "wait_for_recovery"
    EMERGENCY_BYPASS = "emergency_bypass"
    NO_RECOVERY_POSSIBLE = "no_recovery_possible"


class EnhancedCircuitBreakerError(CircuitBreakerError):
    """Enhanced circuit breaker error with additional metadata."""
    
    def __init__(self, 
                 message: str,
                 service_type: Optional[str] = None,
                 enhanced_state: Optional[str] = None,
                 traditional_state: Optional[str] = None,
                 failure_count: int = 0,
                 recovery_time_estimate: Optional[float] = None,
                 recommended_strategy: Optional[RecoveryStrategy] = None,
                 error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 can_fallback: bool = True,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.service_type = service_type
        self.enhanced_state = enhanced_state
        self.traditional_state = traditional_state
        self.failure_count = failure_count
        self.recovery_time_estimate = recovery_time_estimate
        self.recommended_strategy = recommended_strategy
        self.error_severity = error_severity
        self.can_fallback = can_fallback
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """Enhanced string representation."""
        base_msg = super().__str__()
        
        if self.service_type:
            base_msg = f"[{self.service_type}] {base_msg}"
        
        if self.enhanced_state:
            base_msg += f" (state: {self.enhanced_state})"
        
        if self.recovery_time_estimate:
            base_msg += f" (recovery in ~{self.recovery_time_estimate:.1f}s)"
        
        if self.recommended_strategy:
            base_msg += f" (try: {self.recommended_strategy.value})"
        
        return base_msg


class CircuitBreakerStateMapper:
    """Maps between traditional and enhanced circuit breaker states."""
    
    # Mapping from enhanced states to traditional states
    ENHANCED_TO_TRADITIONAL = {
        'closed': 'closed',
        'open': 'open',
        'half_open': 'half-open',
        'degraded': 'open',  # Treat degraded as open for traditional systems
        'rate_limited': 'open',
        'budget_limited': 'open',
        'maintenance': 'open',
    }
    
    # Mapping from traditional states to enhanced states
    TRADITIONAL_TO_ENHANCED = {
        'closed': 'closed',
        'open': 'open',
        'half-open': 'half_open',
    }
    
    @classmethod
    def to_traditional_state(cls, enhanced_state: str) -> str:
        """Convert enhanced state to traditional state."""
        return cls.ENHANCED_TO_TRADITIONAL.get(enhanced_state.lower(), 'open')
    
    @classmethod
    def to_enhanced_state(cls, traditional_state: str) -> str:
        """Convert traditional state to enhanced state."""
        return cls.TRADITIONAL_TO_ENHANCED.get(traditional_state.lower(), 'open')
    
    @classmethod
    def is_blocking_state(cls, state: str, is_enhanced: bool = False) -> bool:
        """Check if a state blocks operations."""
        if is_enhanced:
            blocking_states = ['open', 'maintenance']
            return state.lower() in blocking_states
        else:
            return state.lower() == 'open'
    
    @classmethod
    def is_degraded_state(cls, state: str, is_enhanced: bool = False) -> bool:
        """Check if a state indicates degraded performance."""
        if is_enhanced:
            degraded_states = ['degraded', 'rate_limited', 'budget_limited', 'half_open']
            return state.lower() in degraded_states
        else:
            return state.lower() == 'half-open'


class EnhancedErrorRecoveryStrategy:
    """Advanced recovery strategies for enhanced circuit breaker errors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_attempts = defaultdict(int)
        self.last_recovery_attempt = defaultdict(float)
    
    def get_recovery_strategy(self, 
                             error: EnhancedCircuitBreakerError,
                             context: Dict[str, Any] = None) -> RecoveryStrategy:
        """Determine the best recovery strategy for an error."""
        context = context or {}
        
        # Analyze the error state
        if error.enhanced_state:
            state = error.enhanced_state.lower()
        else:
            state = CircuitBreakerStateMapper.to_enhanced_state(
                error.traditional_state or 'open'
            )
        
        # Determine strategy based on state and context
        if state == 'closed':
            return RecoveryStrategy.IMMEDIATE_RETRY
        
        elif state == 'half_open':
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        elif state == 'degraded':
            if error.can_fallback and context.get('cache_available'):
                return RecoveryStrategy.FALLBACK_TO_CACHE
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        elif state in ['rate_limited', 'budget_limited']:
            if context.get('alternative_service_available'):
                return RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_SERVICE
            return RecoveryStrategy.WAIT_FOR_RECOVERY
        
        elif state == 'open':
            if error.error_severity == ErrorSeverity.CRITICAL:
                if context.get('emergency_bypass_allowed'):
                    return RecoveryStrategy.EMERGENCY_BYPASS
                return RecoveryStrategy.NO_RECOVERY_POSSIBLE
            
            if error.can_fallback:
                if context.get('cache_available'):
                    return RecoveryStrategy.FALLBACK_TO_CACHE
                elif context.get('alternative_service_available'):
                    return RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_SERVICE
            
            return RecoveryStrategy.WAIT_FOR_RECOVERY
        
        elif state == 'maintenance':
            if context.get('alternative_service_available'):
                return RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_SERVICE
            return RecoveryStrategy.WAIT_FOR_RECOVERY
        
        else:
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
    
    def calculate_recovery_delay(self, 
                               error: EnhancedCircuitBreakerError,
                               strategy: RecoveryStrategy,
                               attempt_count: int = 0) -> float:
        """Calculate delay before attempting recovery."""
        base_delays = {
            RecoveryStrategy.IMMEDIATE_RETRY: 0.0,
            RecoveryStrategy.EXPONENTIAL_BACKOFF: min(2 ** attempt_count, 60.0),
            RecoveryStrategy.FALLBACK_TO_CACHE: 0.1,
            RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_SERVICE: 0.5,
            RecoveryStrategy.WAIT_FOR_RECOVERY: error.recovery_time_estimate or 30.0,
            RecoveryStrategy.EMERGENCY_BYPASS: 0.0,
            RecoveryStrategy.NO_RECOVERY_POSSIBLE: float('inf'),
        }
        
        return base_delays.get(strategy, 5.0)
    
    async def execute_recovery_strategy(self,
                                      error: EnhancedCircuitBreakerError,
                                      strategy: RecoveryStrategy,
                                      operation_func,
                                      *args,
                                      **kwargs) -> Any:
        """Execute a recovery strategy."""
        service_key = error.service_type or 'unknown'
        attempt_count = self.recovery_attempts[service_key]
        
        # Calculate and apply delay
        delay = self.calculate_recovery_delay(error, strategy, attempt_count)
        if delay > 0 and delay != float('inf'):
            self.logger.info(f"Applying recovery delay: {delay}s for strategy: {strategy.value}")
            await asyncio.sleep(delay)
        
        # Execute strategy
        try:
            if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                result = await operation_func(*args, **kwargs)
                self.recovery_attempts[service_key] = 0  # Reset on success
                return result
            
            elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                self.recovery_attempts[service_key] += 1
                result = await operation_func(*args, **kwargs)
                self.recovery_attempts[service_key] = 0  # Reset on success
                return result
            
            elif strategy == RecoveryStrategy.FALLBACK_TO_CACHE:
                # Try cache fallback (implementation would depend on available cache)
                cache_func = kwargs.get('cache_fallback_func')
                if cache_func:
                    return await cache_func(*args, **kwargs)
                raise error  # Re-raise if no cache available
            
            elif strategy == RecoveryStrategy.FALLBACK_TO_ALTERNATIVE_SERVICE:
                # Try alternative service
                alt_func = kwargs.get('alternative_service_func')
                if alt_func:
                    return await alt_func(*args, **kwargs)
                raise error  # Re-raise if no alternative available
            
            elif strategy == RecoveryStrategy.EMERGENCY_BYPASS:
                # Emergency bypass (use with caution)
                bypass_func = kwargs.get('emergency_bypass_func')
                if bypass_func:
                    self.logger.warning("Using emergency bypass - monitor carefully")
                    return await bypass_func(*args, **kwargs)
                raise error
            
            elif strategy == RecoveryStrategy.NO_RECOVERY_POSSIBLE:
                self.logger.error("No recovery possible for circuit breaker error")
                raise error
            
            else:  # WAIT_FOR_RECOVERY
                self.logger.info("Waiting for circuit breaker recovery")
                raise error
        
        except Exception as e:
            self.recovery_attempts[service_key] += 1
            self.last_recovery_attempt[service_key] = time.time()
            raise e


class CircuitBreakerErrorAnalyzer:
    """Analyzes circuit breaker errors for patterns and predictions."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history = deque(maxlen=1000)
        self.service_error_patterns = defaultdict(lambda: deque(maxlen=100))
    
    def analyze_error(self, error: Exception, service_type: str = None) -> EnhancedCircuitBreakerError:
        """Analyze an error and convert it to enhanced circuit breaker error if needed."""
        
        # If already enhanced, just record and return
        if isinstance(error, EnhancedCircuitBreakerError):
            self._record_error(error)
            return error
        
        # Convert traditional circuit breaker error
        if isinstance(error, CircuitBreakerError):
            enhanced_error = self._convert_traditional_error(error, service_type)
            self._record_error(enhanced_error)
            return enhanced_error
        
        # Not a circuit breaker error
        return error
    
    def _convert_traditional_error(self, 
                                 error: CircuitBreakerError, 
                                 service_type: str = None) -> EnhancedCircuitBreakerError:
        """Convert traditional circuit breaker error to enhanced version."""
        
        # Extract state information from error message if possible
        error_msg = str(error).lower()
        traditional_state = 'open'  # Default assumption
        
        if 'half-open' in error_msg or 'half_open' in error_msg:
            traditional_state = 'half-open'
        elif 'closed' in error_msg:
            traditional_state = 'closed'
        
        enhanced_state = CircuitBreakerStateMapper.to_enhanced_state(traditional_state)
        
        # Determine severity based on patterns
        severity = self._determine_error_severity(service_type, error_msg)
        
        # Estimate recovery time
        recovery_estimate = self._estimate_recovery_time(service_type, traditional_state)
        
        # Determine if fallback is possible
        can_fallback = self._can_use_fallback(service_type, error_msg)
        
        return EnhancedCircuitBreakerError(
            str(error),
            service_type=service_type,
            enhanced_state=enhanced_state,
            traditional_state=traditional_state,
            failure_count=self._get_recent_failure_count(service_type),
            recovery_time_estimate=recovery_estimate,
            error_severity=severity,
            can_fallback=can_fallback,
            metadata={
                'original_error_type': type(error).__name__,
                'original_error_message': str(error),
                'analysis_timestamp': time.time(),
            }
        )
    
    def _record_error(self, error: EnhancedCircuitBreakerError) -> None:
        """Record error for pattern analysis."""
        error_record = {
            'timestamp': time.time(),
            'service_type': error.service_type,
            'enhanced_state': error.enhanced_state,
            'traditional_state': error.traditional_state,
            'severity': error.error_severity.value,
            'can_fallback': error.can_fallback,
        }
        
        self.error_history.append(error_record)
        
        if error.service_type:
            self.service_error_patterns[error.service_type].append(error_record)
    
    def _determine_error_severity(self, service_type: str, error_msg: str) -> ErrorSeverity:
        """Determine error severity based on patterns and service type."""
        
        # Check for high-severity keywords
        high_severity_keywords = ['budget', 'quota', 'billing', 'payment', 'unauthorized']
        if any(keyword in error_msg for keyword in high_severity_keywords):
            return ErrorSeverity.HIGH
        
        # Check for critical keywords
        critical_keywords = ['emergency', 'critical', 'fatal', 'unrecoverable']
        if any(keyword in error_msg for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL
        
        # Check recent error frequency for this service
        if service_type:
            recent_errors = [
                e for e in self.service_error_patterns[service_type]
                if time.time() - e['timestamp'] < 300  # Last 5 minutes
            ]
            
            if len(recent_errors) > 10:
                return ErrorSeverity.HIGH
            elif len(recent_errors) > 5:
                return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _estimate_recovery_time(self, service_type: str, state: str) -> Optional[float]:
        """Estimate recovery time based on historical patterns."""
        
        # Default estimates by state
        default_estimates = {
            'open': 60.0,
            'half-open': 30.0,
            'closed': 0.0,
        }
        
        base_estimate = default_estimates.get(state, 30.0)
        
        # Adjust based on service type
        service_multipliers = {
            'openai_api': 1.0,
            'perplexity_api': 1.2,
            'lightrag': 0.8,
            'cache': 0.5,
        }
        
        multiplier = service_multipliers.get(service_type, 1.0)
        
        return base_estimate * multiplier
    
    def _can_use_fallback(self, service_type: str, error_msg: str) -> bool:
        """Determine if fallback mechanisms are available/appropriate."""
        
        # Check for non-fallback scenarios
        non_fallback_keywords = ['authentication', 'authorization', 'billing', 'quota']
        if any(keyword in error_msg for keyword in non_fallback_keywords):
            return False
        
        # Service-specific fallback availability
        fallback_available_services = ['openai_api', 'perplexity_api', 'lightrag']
        if service_type in fallback_available_services:
            return True
        
        return True  # Default to allowing fallback
    
    def _get_recent_failure_count(self, service_type: str) -> int:
        """Get recent failure count for service."""
        if not service_type:
            return 0
        
        recent_errors = [
            e for e in self.service_error_patterns[service_type]
            if time.time() - e['timestamp'] < 600  # Last 10 minutes
        ]
        
        return len(recent_errors)
    
    def get_error_patterns(self, service_type: str = None) -> Dict[str, Any]:
        """Get error patterns for analysis."""
        if service_type:
            patterns = list(self.service_error_patterns[service_type])
        else:
            patterns = list(self.error_history)
        
        if not patterns:
            return {'total_errors': 0}
        
        # Analyze patterns
        recent_patterns = [p for p in patterns if time.time() - p['timestamp'] < 3600]  # Last hour
        
        severity_counts = defaultdict(int)
        state_counts = defaultdict(int)
        
        for pattern in recent_patterns:
            severity_counts[pattern['severity']] += 1
            state_counts[pattern['enhanced_state']] += 1
        
        return {
            'total_errors': len(patterns),
            'recent_errors': len(recent_patterns),
            'severity_distribution': dict(severity_counts),
            'state_distribution': dict(state_counts),
            'error_rate_per_hour': len(recent_patterns),
            'most_common_severity': max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None,
            'most_common_state': max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else None,
        }


class EnhancedCircuitBreakerErrorHandler:
    """Unified error handling for enhanced circuit breaker systems."""
    
    def __init__(self, 
                 enable_enhanced_features: bool = True,
                 logger: Optional[logging.Logger] = None):
        self.enable_enhanced_features = enable_enhanced_features and ENHANCED_CIRCUIT_BREAKERS_AVAILABLE
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.error_analyzer = CircuitBreakerErrorAnalyzer(logger)
        self.recovery_strategy = EnhancedErrorRecoveryStrategy(logger)
        
        # Statistics
        self.handled_errors = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        
    async def handle_circuit_breaker_error(self,
                                         error: Exception,
                                         operation_func,
                                         service_type: str = None,
                                         recovery_context: Dict[str, Any] = None,
                                         *args,
                                         **kwargs) -> Any:
        """
        Handle circuit breaker errors with enhanced recovery strategies.
        
        Args:
            error: The circuit breaker error
            operation_func: Original operation to retry
            service_type: Service that generated the error
            recovery_context: Context for recovery decisions
            *args, **kwargs: Arguments for operation_func
            
        Returns:
            Result of successful recovery or re-raises enhanced error
        """
        
        self.handled_errors += 1
        recovery_context = recovery_context or {}
        
        try:
            # Analyze the error
            enhanced_error = self.error_analyzer.analyze_error(error, service_type)
            
            # Determine recovery strategy
            strategy = self.recovery_strategy.get_recovery_strategy(enhanced_error, recovery_context)
            enhanced_error.recommended_strategy = strategy
            
            self.logger.info(f"Handling circuit breaker error for {service_type}: {strategy.value}")
            
            # Attempt recovery
            result = await self.recovery_strategy.execute_recovery_strategy(
                enhanced_error, strategy, operation_func, *args, **kwargs
            )
            
            self.successful_recoveries += 1
            self.logger.info(f"Successfully recovered from circuit breaker error using {strategy.value}")
            return result
            
        except Exception as recovery_error:
            self.failed_recoveries += 1
            
            # If recovery failed, enhance the error with additional information
            if isinstance(error, CircuitBreakerError):
                enhanced_error = self.error_analyzer.analyze_error(error, service_type)
                enhanced_error.metadata['recovery_attempted'] = True
                enhanced_error.metadata['recovery_strategy'] = strategy.value if 'strategy' in locals() else None
                enhanced_error.metadata['recovery_error'] = str(recovery_error)
                
                self.logger.error(f"Failed to recover from circuit breaker error: {recovery_error}")
                raise enhanced_error
            else:
                self.logger.error(f"Recovery attempt failed: {recovery_error}")
                raise recovery_error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            'handled_errors': self.handled_errors,
            'successful_recoveries': self.successful_recoveries,
            'failed_recoveries': self.failed_recoveries,
            'success_rate': self.successful_recoveries / max(self.handled_errors, 1) * 100,
            'enhanced_features_enabled': self.enable_enhanced_features,
            'error_patterns': self.error_analyzer.get_error_patterns(),
        }
    
    def reset_statistics(self) -> None:
        """Reset error handling statistics."""
        self.handled_errors = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.error_analyzer = CircuitBreakerErrorAnalyzer(self.logger)


# Global error handler instance for easy access
_global_error_handler = None

def get_global_error_handler() -> EnhancedCircuitBreakerErrorHandler:
    """Get the global enhanced circuit breaker error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = EnhancedCircuitBreakerErrorHandler()
    return _global_error_handler


def set_global_error_handler(handler: EnhancedCircuitBreakerErrorHandler) -> None:
    """Set a custom global error handler."""
    global _global_error_handler
    _global_error_handler = handler


# Convenience function for common use case
async def handle_circuit_breaker_error(error: Exception,
                                     operation_func,
                                     service_type: str = None,
                                     recovery_context: Dict[str, Any] = None,
                                     *args,
                                     **kwargs) -> Any:
    """
    Convenience function to handle circuit breaker errors using global handler.
    
    This is a drop-in replacement for manual error handling that provides
    enhanced recovery capabilities.
    """
    handler = get_global_error_handler()
    return await handler.handle_circuit_breaker_error(
        error, operation_func, service_type, recovery_context, *args, **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def example_usage():
        # Create error handler
        handler = EnhancedCircuitBreakerErrorHandler()
        
        # Simulate a circuit breaker error
        error = CircuitBreakerError("Service is temporarily unavailable")
        
        # Mock operation function
        async def mock_operation():
            return "Success!"
        
        try:
            result = await handler.handle_circuit_breaker_error(
                error=error,
                operation_func=mock_operation,
                service_type="openai_api",
                recovery_context={
                    'cache_available': True,
                    'alternative_service_available': False
                }
            )
            print(f"Recovery successful: {result}")
        except Exception as e:
            print(f"Recovery failed: {e}")
        
        # Show statistics
        stats = handler.get_error_statistics()
        print(f"Error handling statistics: {stats}")
    
    asyncio.run(example_usage())