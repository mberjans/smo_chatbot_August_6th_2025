#!/usr/bin/env python3
"""
Error Recovery Integration Layer for Clinical Metabolomics Oracle

This module provides seamless integration of the comprehensive error recovery system
with the existing Clinical Metabolomics Oracle codebase through:

- Decorators for automatic error recovery
- Integration with existing error handling systems
- Context managers for operation protection
- Async operation support
- Easy-to-use helper functions and utilities

Features:
    - @retry_on_error decorator for automatic retry logic
    - ErrorRecoveryContext context manager for protected operations
    - Integration with existing ClinicalMetabolomicsRAG error classes
    - Async/await support for asynchronous operations
    - Configuration-driven behavior
    - Monitoring and metrics integration

Author: Claude Code (Anthropic)
Created: 2025-08-09
Version: 1.0.0
Task: CMO-LIGHTRAG-014-T06
"""

import asyncio
import functools
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path
import inspect
import time

# Import error recovery system components
from .comprehensive_error_recovery_system import (
    ErrorRecoveryOrchestrator, create_error_recovery_orchestrator
)
from .error_recovery_config import (
    ErrorRecoveryConfigManager, ConfigurationProfile, create_error_recovery_config_manager
)

# Import existing error classes
try:
    from .clinical_metabolomics_rag import (
        ClinicalMetabolomicsRAGError, QueryError, QueryRetryableError,
        IngestionError, IngestionRetryableError, CircuitBreakerError
    )
    ERROR_CLASSES_AVAILABLE = True
except ImportError:
    ERROR_CLASSES_AVAILABLE = False
    
    # Mock classes for standalone operation
    class ClinicalMetabolomicsRAGError(Exception): pass
    class QueryError(ClinicalMetabolomicsRAGError): pass
    class QueryRetryableError(QueryError): pass
    class IngestionError(ClinicalMetabolomicsRAGError): pass
    class IngestionRetryableError(IngestionError): pass
    class CircuitBreakerError(Exception): pass

# Import existing advanced recovery system
try:
    from .advanced_recovery_system import AdvancedRecoverySystem
    ADVANCED_RECOVERY_AVAILABLE = True
except ImportError:
    ADVANCED_RECOVERY_AVAILABLE = False
    AdvancedRecoverySystem = None

# Import circuit breaker
try:
    from .cost_based_circuit_breaker import CostCircuitBreakerManager
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CostCircuitBreakerManager = None


# Global orchestrator instance
_global_orchestrator: Optional[ErrorRecoveryOrchestrator] = None
_global_config_manager: Optional[ErrorRecoveryConfigManager] = None


def initialize_error_recovery_system(
    config_file: Optional[Path] = None,
    profile: Optional[ConfigurationProfile] = None,
    state_dir: Optional[Path] = None,
    advanced_recovery: Optional[AdvancedRecoverySystem] = None,
    circuit_breaker_manager: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> ErrorRecoveryOrchestrator:
    """
    Initialize the global error recovery system.
    
    Args:
        config_file: Path to configuration file
        profile: Configuration profile to use
        state_dir: Directory for retry state persistence
        advanced_recovery: Advanced recovery system integration
        circuit_breaker_manager: Circuit breaker manager integration
        logger: Logger instance
        
    Returns:
        Configured ErrorRecoveryOrchestrator instance
    """
    global _global_orchestrator, _global_config_manager
    
    if not logger:
        logger = logging.getLogger(__name__)
    
    # Initialize configuration manager
    _global_config_manager = create_error_recovery_config_manager(
        config_file=config_file,
        profile=profile,
        logger=logger
    )
    
    # Get recovery rules from configuration
    recovery_rules = _global_config_manager.get_recovery_rules()
    
    # Initialize orchestrator
    _global_orchestrator = create_error_recovery_orchestrator(
        state_dir=state_dir,
        recovery_rules=recovery_rules,
        advanced_recovery=advanced_recovery,
        circuit_breaker_manager=circuit_breaker_manager,
        logger=logger
    )
    
    logger.info("Global error recovery system initialized")
    return _global_orchestrator


def get_error_recovery_orchestrator() -> Optional[ErrorRecoveryOrchestrator]:
    """Get the global error recovery orchestrator."""
    return _global_orchestrator


def get_error_recovery_config_manager() -> Optional[ErrorRecoveryConfigManager]:
    """Get the global error recovery configuration manager."""
    return _global_config_manager


# Decorator for automatic error recovery
def retry_on_error(
    operation_type: str,
    max_attempts: Optional[int] = None,
    include_exceptions: Optional[tuple] = None,
    exclude_exceptions: Optional[tuple] = None,
    auto_retry: bool = True,
    preserve_context: bool = True
):
    """
    Decorator that adds automatic error recovery to functions.
    
    Args:
        operation_type: Type of operation for classification
        max_attempts: Override default max retry attempts
        include_exceptions: Tuple of exception types to handle
        exclude_exceptions: Tuple of exception types to exclude
        auto_retry: Whether to automatically retry on failure
        preserve_context: Whether to preserve function context for retries
        
    Usage:
        @retry_on_error("query_operation", max_attempts=3)
        def query_data(query: str) -> str:
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_with_recovery(
                func, args, kwargs, operation_type, max_attempts,
                include_exceptions, exclude_exceptions, auto_retry, preserve_context
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_async_with_recovery(
                func, args, kwargs, operation_type, max_attempts,
                include_exceptions, exclude_exceptions, auto_retry, preserve_context
            )
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _execute_with_recovery(
    func: Callable,
    args: tuple,
    kwargs: dict,
    operation_type: str,
    max_attempts: Optional[int],
    include_exceptions: Optional[tuple],
    exclude_exceptions: Optional[tuple],
    auto_retry: bool,
    preserve_context: bool
) -> Any:
    """Execute function with error recovery (synchronous)."""
    orchestrator = get_error_recovery_orchestrator()
    if not orchestrator:
        # Fallback to direct execution if orchestrator not initialized
        return func(*args, **kwargs)
    
    operation_id = str(uuid.uuid4())
    operation_context = _build_operation_context(
        func, args, kwargs, preserve_context
    )
    
    # Override max attempts if specified
    if max_attempts:
        operation_context['max_attempts'] = max_attempts
    
    try:
        result = func(*args, **kwargs)
        # Record success
        orchestrator.retry_engine.record_operation_success(operation_id, 0.0)
        return result
        
    except Exception as error:
        # Check if we should handle this exception
        if not _should_handle_exception(error, include_exceptions, exclude_exceptions):
            raise
        
        # Handle error through orchestrator
        recovery_result = orchestrator.handle_operation_error(
            operation_id=operation_id,
            error=error,
            operation_type=operation_type,
            operation_context=operation_context,
            operation_callable=func if auto_retry else None
        )
        
        # If automatic retry was performed and succeeded, return result
        if (recovery_result.get('retry_execution', {}).get('success', False)):
            return recovery_result['retry_execution']['result']
        
        # If retry is recommended but not auto-executed, provide recovery info
        if recovery_result.get('should_retry', False) and not auto_retry:
            # Add recovery info to exception
            error.recovery_info = recovery_result
        
        # Re-raise the original exception
        raise


async def _execute_async_with_recovery(
    func: Callable,
    args: tuple,
    kwargs: dict,
    operation_type: str,
    max_attempts: Optional[int],
    include_exceptions: Optional[tuple],
    exclude_exceptions: Optional[tuple],
    auto_retry: bool,
    preserve_context: bool
) -> Any:
    """Execute async function with error recovery."""
    orchestrator = get_error_recovery_orchestrator()
    if not orchestrator:
        # Fallback to direct execution if orchestrator not initialized
        return await func(*args, **kwargs)
    
    operation_id = str(uuid.uuid4())
    operation_context = _build_operation_context(
        func, args, kwargs, preserve_context
    )
    
    if max_attempts:
        operation_context['max_attempts'] = max_attempts
    
    try:
        result = await func(*args, **kwargs)
        orchestrator.retry_engine.record_operation_success(operation_id, 0.0)
        return result
        
    except Exception as error:
        if not _should_handle_exception(error, include_exceptions, exclude_exceptions):
            raise
        
        # For async functions, we need to create an async-compatible callable
        async def async_retry_callable(**ctx):
            return await func(*args, **kwargs)
        
        recovery_result = orchestrator.handle_operation_error(
            operation_id=operation_id,
            error=error,
            operation_type=operation_type,
            operation_context=operation_context,
            operation_callable=async_retry_callable if auto_retry else None
        )
        
        if (recovery_result.get('retry_execution', {}).get('success', False)):
            return recovery_result['retry_execution']['result']
        
        if recovery_result.get('should_retry', False) and not auto_retry:
            error.recovery_info = recovery_result
        
        raise


def _build_operation_context(
    func: Callable,
    args: tuple,
    kwargs: dict,
    preserve_context: bool
) -> Dict[str, Any]:
    """Build operation context for recovery."""
    context = {
        'function_name': func.__name__,
        'module': func.__module__,
        'timestamp': time.time()
    }
    
    if preserve_context:
        # Store arguments for retry (be careful with sensitive data)
        context['args'] = args
        context['kwargs'] = kwargs
    
    # Add function metadata
    if hasattr(func, '__doc__') and func.__doc__:
        context['function_doc'] = func.__doc__[:200]  # Truncate for storage
    
    return context


def _should_handle_exception(
    error: Exception,
    include_exceptions: Optional[tuple],
    exclude_exceptions: Optional[tuple]
) -> bool:
    """Check if exception should be handled by error recovery."""
    # Check exclusions first
    if exclude_exceptions and isinstance(error, exclude_exceptions):
        return False
    
    # If inclusions specified, only handle those
    if include_exceptions:
        return isinstance(error, include_exceptions)
    
    # Default: handle retryable errors from our error hierarchy
    retryable_types = (
        QueryRetryableError,
        IngestionRetryableError,
    )
    
    return isinstance(error, retryable_types)


@contextmanager
def error_recovery_context(
    operation_type: str,
    operation_id: Optional[str] = None,
    preserve_on_failure: bool = True
):
    """
    Context manager for protecting operations with error recovery.
    
    Args:
        operation_type: Type of operation
        operation_id: Optional operation identifier
        preserve_on_failure: Whether to preserve context on failure
        
    Usage:
        with error_recovery_context("data_processing") as ctx:
            # Protected operation
            result = process_data()
            ctx.set_result(result)
    """
    if not operation_id:
        operation_id = str(uuid.uuid4())
    
    context = ErrorRecoveryContext(operation_type, operation_id, preserve_on_failure)
    try:
        yield context
    except Exception as error:
        context.handle_error(error)
        raise


class ErrorRecoveryContext:
    """Context for error recovery operations."""
    
    def __init__(self,
                 operation_type: str,
                 operation_id: str,
                 preserve_on_failure: bool = True):
        """Initialize error recovery context."""
        self.operation_type = operation_type
        self.operation_id = operation_id
        self.preserve_on_failure = preserve_on_failure
        self.operation_context = {}
        self.result = None
        self.start_time = time.time()
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context value for recovery."""
        self.operation_context[key] = value
    
    def set_result(self, result: Any) -> None:
        """Set operation result."""
        self.result = result
        
        # Record success
        orchestrator = get_error_recovery_orchestrator()
        if orchestrator:
            execution_time = time.time() - self.start_time
            orchestrator.retry_engine.record_operation_success(
                self.operation_id, execution_time
            )
    
    def handle_error(self, error: Exception) -> None:
        """Handle error through recovery system."""
        orchestrator = get_error_recovery_orchestrator()
        if orchestrator:
            recovery_result = orchestrator.handle_operation_error(
                operation_id=self.operation_id,
                error=error,
                operation_type=self.operation_type,
                operation_context=self.operation_context.copy()
            )
            
            # Attach recovery info to exception for caller
            error.recovery_info = recovery_result


# Utility functions for common operations
def execute_with_retry(
    operation: Callable,
    operation_type: str,
    max_attempts: int = 3,
    operation_id: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Execute operation with retry logic.
    
    Args:
        operation: Function to execute
        operation_type: Type of operation
        max_attempts: Maximum retry attempts
        operation_id: Optional operation identifier
        **kwargs: Arguments for operation
        
    Returns:
        Operation result
    """
    if not operation_id:
        operation_id = str(uuid.uuid4())
    
    orchestrator = get_error_recovery_orchestrator()
    if not orchestrator:
        return operation(**kwargs)
    
    operation_context = {
        'max_attempts': max_attempts,
        'kwargs': kwargs,
        'timestamp': time.time()
    }
    
    try:
        result = operation(**kwargs)
        orchestrator.retry_engine.record_operation_success(operation_id, 0.0)
        return result
        
    except Exception as error:
        recovery_result = orchestrator.handle_operation_error(
            operation_id=operation_id,
            error=error,
            operation_type=operation_type,
            operation_context=operation_context,
            operation_callable=lambda: operation(**kwargs)
        )
        
        if recovery_result.get('retry_execution', {}).get('success', False):
            return recovery_result['retry_execution']['result']
        else:
            raise


async def execute_async_with_retry(
    operation: Callable,
    operation_type: str,
    max_attempts: int = 3,
    operation_id: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Execute async operation with retry logic.
    
    Args:
        operation: Async function to execute
        operation_type: Type of operation
        max_attempts: Maximum retry attempts
        operation_id: Optional operation identifier
        **kwargs: Arguments for operation
        
    Returns:
        Operation result
    """
    if not operation_id:
        operation_id = str(uuid.uuid4())
    
    orchestrator = get_error_recovery_orchestrator()
    if not orchestrator:
        return await operation(**kwargs)
    
    operation_context = {
        'max_attempts': max_attempts,
        'kwargs': kwargs,
        'timestamp': time.time()
    }
    
    try:
        result = await operation(**kwargs)
        orchestrator.retry_engine.record_operation_success(operation_id, 0.0)
        return result
        
    except Exception as error:
        async def async_retry():
            return await operation(**kwargs)
        
        recovery_result = orchestrator.handle_operation_error(
            operation_id=operation_id,
            error=error,
            operation_type=operation_type,
            operation_context=operation_context,
            operation_callable=async_retry
        )
        
        if recovery_result.get('retry_execution', {}).get('success', False):
            return recovery_result['retry_execution']['result']
        else:
            raise


# Integration helper for existing Clinical Metabolomics Oracle components
class ClinicalMetabolomicsErrorRecoveryMixin:
    """Mixin class to add error recovery to existing classes."""
    
    def __init__(self, *args, **kwargs):
        """Initialize mixin."""
        super().__init__(*args, **kwargs)
        self._recovery_orchestrator = get_error_recovery_orchestrator()
        self._recovery_enabled = True
    
    def enable_error_recovery(self) -> None:
        """Enable error recovery for this instance."""
        self._recovery_enabled = True
    
    def disable_error_recovery(self) -> None:
        """Disable error recovery for this instance."""
        self._recovery_enabled = False
    
    def execute_with_recovery(self,
                            operation: Callable,
                            operation_type: str,
                            **kwargs) -> Any:
        """Execute operation with error recovery."""
        if not self._recovery_enabled or not self._recovery_orchestrator:
            return operation(**kwargs)
        
        return execute_with_retry(
            operation=operation,
            operation_type=operation_type,
            **kwargs
        )
    
    async def execute_async_with_recovery(self,
                                        operation: Callable,
                                        operation_type: str,
                                        **kwargs) -> Any:
        """Execute async operation with error recovery."""
        if not self._recovery_enabled or not self._recovery_orchestrator:
            return await operation(**kwargs)
        
        return await execute_async_with_retry(
            operation=operation,
            operation_type=operation_type,
            **kwargs
        )


# System status and monitoring utilities
def get_error_recovery_status() -> Dict[str, Any]:
    """Get comprehensive error recovery system status."""
    orchestrator = get_error_recovery_orchestrator()
    config_manager = get_error_recovery_config_manager()
    
    status = {
        'orchestrator_available': orchestrator is not None,
        'config_manager_available': config_manager is not None,
        'timestamp': time.time()
    }
    
    if orchestrator:
        status['orchestrator_status'] = orchestrator.get_system_status()
    
    if config_manager:
        status['configuration_summary'] = config_manager.get_configuration_summary()
    
    return status


def shutdown_error_recovery_system() -> None:
    """Shutdown the error recovery system gracefully."""
    global _global_orchestrator, _global_config_manager
    
    if _global_orchestrator:
        _global_orchestrator.close()
        _global_orchestrator = None
    
    _global_config_manager = None


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize error recovery system
    orchestrator = initialize_error_recovery_system(
        profile=ConfigurationProfile.DEVELOPMENT
    )
    
    # Test synchronous operation with decorator
    @retry_on_error("test_sync_operation", max_attempts=3)
    def test_sync_operation(data: str) -> str:
        if random.random() < 0.7:  # 70% failure rate
            raise Exception(f"Random failure processing {data}")
        return f"Successfully processed: {data}"
    
    # Test asynchronous operation with decorator
    @retry_on_error("test_async_operation", max_attempts=3)
    async def test_async_operation(data: str) -> str:
        if random.random() < 0.7:  # 70% failure rate
            raise Exception(f"Random async failure processing {data}")
        return f"Successfully processed async: {data}"
    
    # Test context manager
    def test_context_manager():
        with error_recovery_context("context_test_operation") as ctx:
            ctx.set_context("test_data", "example")
            if random.random() < 0.5:
                raise Exception("Context manager test failure")
            ctx.set_result("Context manager success")
            return ctx.result
    
    # Run tests
    print("Testing synchronous operation:")
    try:
        result = test_sync_operation("test_data")
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    print("\nTesting asynchronous operation:")
    async def test_async():
        try:
            result = await test_async_operation("test_data")
            print(f"Async success: {result}")
        except Exception as e:
            print(f"Async failed: {e}")
    
    asyncio.run(test_async())
    
    print("\nTesting context manager:")
    try:
        result = test_context_manager()
        print(f"Context success: {result}")
    except Exception as e:
        print(f"Context failed: {e}")
    
    # Show system status
    print("\nError Recovery System Status:")
    status = get_error_recovery_status()
    print(f"System available: {status['orchestrator_available']}")
    
    # Shutdown
    shutdown_error_recovery_system()