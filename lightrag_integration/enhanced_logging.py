"""
Enhanced logging system for Clinical Metabolomics Oracle LightRAG integration.

This module provides comprehensive logging enhancements for troubleshooting ingestion issues,
including structured logging, correlation IDs, performance metrics, and detailed error context.

Classes:
    - CorrelationIDManager: Manages correlation IDs for tracking related operations
    - StructuredLogRecord: Enhanced log record with structured data
    - EnhancedLogger: Logger wrapper with structured logging capabilities
    - PerformanceLogger: Specialized logger for performance metrics
    - IngestionLogger: Specialized logger for ingestion processes
    - DiagnosticLogger: Specialized logger for diagnostic information

Features:
    - Correlation IDs for tracking related operations
    - Structured logging with JSON formatting
    - Performance metrics tracking
    - Memory usage monitoring
    - Detailed error context with stack traces
    - Integration with existing logging infrastructure
"""

import json
import uuid
import time
import psutil
import traceback
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import wraps
import logging
from pathlib import Path
import inspect
import sys


@dataclass
class CorrelationContext:
    """Context information for correlated operations."""
    correlation_id: str
    operation_name: str
    start_time: datetime
    parent_correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'correlation_id': self.correlation_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'parent_correlation_id': self.parent_correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'duration_ms': (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000,
            'metadata': self.metadata
        }


class CorrelationIDManager:
    """Thread-safe manager for correlation IDs and operation context."""
    
    def __init__(self):
        self._local = threading.local()
    
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    def set_context(self, context: CorrelationContext) -> None:
        """Set correlation context for current thread."""
        if not hasattr(self._local, 'context_stack'):
            self._local.context_stack = []
        self._local.context_stack.append(context)
    
    def get_context(self) -> Optional[CorrelationContext]:
        """Get current correlation context."""
        if hasattr(self._local, 'context_stack') and self._local.context_stack:
            return self._local.context_stack[-1]
        return None
    
    def pop_context(self) -> Optional[CorrelationContext]:
        """Remove and return current correlation context."""
        if hasattr(self._local, 'context_stack') and self._local.context_stack:
            return self._local.context_stack.pop()
        return None
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        context = self.get_context()
        return context.correlation_id if context else None
    
    @contextmanager
    def operation_context(self, operation_name: str, 
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """Context manager for correlation tracking."""
        current_context = self.get_context()
        parent_id = current_context.correlation_id if current_context else None
        
        context = CorrelationContext(
            correlation_id=self.generate_correlation_id(),
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc),
            parent_correlation_id=parent_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.set_context(context)
        try:
            yield context
        finally:
            self.pop_context()


# Global correlation ID manager
correlation_manager = CorrelationIDManager()


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None
    duration_ms: Optional[float] = None
    api_calls_count: int = 0
    tokens_used: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class StructuredLogRecord:
    """Enhanced log record with structured data."""
    
    def __init__(self, level: str, message: str, 
                 correlation_id: Optional[str] = None,
                 operation_name: Optional[str] = None,
                 component: Optional[str] = None,
                 error_details: Optional[Dict[str, Any]] = None,
                 performance_metrics: Optional[PerformanceMetrics] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.timestamp = datetime.now(timezone.utc)
        self.level = level
        self.message = message
        self.correlation_id = correlation_id or correlation_manager.get_correlation_id()
        self.operation_name = operation_name
        self.component = component
        self.error_details = error_details or {}
        self.performance_metrics = performance_metrics
        self.metadata = metadata or {}
        
        # Add context from correlation manager
        context = correlation_manager.get_context()
        if context:
            self.operation_name = self.operation_name or context.operation_name
            self.metadata.update(context.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        record = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'operation_name': self.operation_name,
            'component': self.component,
            'error_details': self.error_details,
            'metadata': self.metadata
        }
        
        if self.performance_metrics:
            record['performance_metrics'] = self.performance_metrics.to_dict()
        
        return {k: v for k, v in record.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=None, separators=(',', ':'))


class EnhancedLogger:
    """Enhanced logger wrapper with structured logging capabilities."""
    
    def __init__(self, base_logger: logging.Logger, component: str = None):
        self.base_logger = base_logger
        self.component = component
        self._performance_tracker = PerformanceTracker()
    
    def _log_structured(self, level: str, message: str, 
                       correlation_id: Optional[str] = None,
                       operation_name: Optional[str] = None,
                       error_details: Optional[Dict[str, Any]] = None,
                       performance_metrics: Optional[PerformanceMetrics] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       exc_info: bool = False):
        """Log structured record."""
        
        record = StructuredLogRecord(
            level=level,
            message=message,
            correlation_id=correlation_id,
            operation_name=operation_name,
            component=self.component,
            error_details=error_details,
            performance_metrics=performance_metrics,
            metadata=metadata
        )
        
        # Log as JSON for structured logs and regular message for console
        json_message = record.to_json()
        regular_message = f"[{record.correlation_id or 'N/A'}] {message}"
        
        # Log to base logger
        log_method = getattr(self.base_logger, level.lower())
        log_method(regular_message, exc_info=exc_info)
        
        # Also log structured data to a separate structured logger if available
        structured_logger = logging.getLogger(f"{self.base_logger.name}.structured")
        if structured_logger.handlers:
            structured_log_method = getattr(structured_logger, level.lower())
            structured_log_method(json_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log_structured('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log_structured('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log_structured('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log_structured('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self._log_structured('CRITICAL', message, **kwargs)
    
    def log_error_with_context(self, message: str, error: Exception,
                              operation_name: Optional[str] = None,
                              additional_context: Optional[Dict[str, Any]] = None):
        """Log error with detailed context and stack trace."""
        
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'file': inspect.currentframe().f_back.f_code.co_filename,
            'line': inspect.currentframe().f_back.f_lineno,
            'function': inspect.currentframe().f_back.f_code.co_name
        }
        
        if additional_context:
            error_details['additional_context'] = additional_context
        
        self.error(
            message,
            operation_name=operation_name,
            error_details=error_details,
            exc_info=True
        )
    
    def log_performance_metrics(self, operation_name: str, 
                               metrics: PerformanceMetrics,
                               level: str = 'INFO'):
        """Log performance metrics."""
        message = f"Performance metrics for {operation_name}"
        self._log_structured(level, message, 
                           operation_name=operation_name,
                           performance_metrics=metrics)


class PerformanceTracker:
    """Tracks performance metrics for operations."""
    
    def __init__(self):
        self._process = psutil.Process()
        self._start_stats = None
    
    def start_tracking(self) -> Dict[str, Any]:
        """Start performance tracking."""
        try:
            self._start_stats = {
                'cpu_times': self._process.cpu_times(),
                'memory_info': self._process.memory_info(),
                'io_counters': self._process.io_counters() if hasattr(self._process, 'io_counters') else None,
                'timestamp': time.time()
            }
            return self._start_stats
        except Exception:
            # If we can't get stats, just return empty dict
            return {}
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        try:
            current_time = time.time()
            memory_info = self._process.memory_info()
            
            metrics = PerformanceMetrics(
                cpu_percent=self._process.cpu_percent(),
                memory_mb=memory_info.rss / 1024 / 1024,
                memory_percent=self._process.memory_percent()
            )
            
            # Calculate duration if tracking was started
            if self._start_stats:
                metrics.duration_ms = (current_time - self._start_stats['timestamp']) * 1000
                
                # Calculate I/O differences if available
                if self._start_stats.get('io_counters') and hasattr(self._process, 'io_counters'):
                    try:
                        current_io = self._process.io_counters()
                        start_io = self._start_stats['io_counters']
                        metrics.disk_io_read_mb = (current_io.read_bytes - start_io.read_bytes) / 1024 / 1024
                        metrics.disk_io_write_mb = (current_io.write_bytes - start_io.write_bytes) / 1024 / 1024
                    except Exception:
                        pass
            
            return metrics
        except Exception:
            # Return empty metrics if we can't collect them
            return PerformanceMetrics()


class IngestionLogger:
    """Specialized logger for ingestion processes."""
    
    def __init__(self, base_logger: logging.Logger):
        self.enhanced_logger = EnhancedLogger(base_logger, component="ingestion")
        self._batch_stats = {}
    
    def log_document_start(self, document_id: str, document_path: str, 
                          batch_id: Optional[str] = None):
        """Log start of document processing."""
        metadata = {
            'document_id': document_id,
            'document_path': document_path,
            'batch_id': batch_id
        }
        
        self.enhanced_logger.info(
            f"Starting document processing: {document_id}",
            operation_name="document_processing",
            metadata=metadata
        )
    
    def log_document_complete(self, document_id: str, 
                            processing_time_ms: float,
                            pages_processed: int,
                            characters_extracted: int,
                            batch_id: Optional[str] = None):
        """Log successful document processing completion."""
        metadata = {
            'document_id': document_id,
            'processing_time_ms': processing_time_ms,
            'pages_processed': pages_processed,
            'characters_extracted': characters_extracted,
            'batch_id': batch_id
        }
        
        metrics = PerformanceMetrics(
            duration_ms=processing_time_ms
        )
        
        self.enhanced_logger.info(
            f"Document processing completed: {document_id} "
            f"({pages_processed} pages, {characters_extracted} chars, "
            f"{processing_time_ms:.1f}ms)",
            operation_name="document_processing",
            performance_metrics=metrics,
            metadata=metadata
        )
    
    def log_document_error(self, document_id: str, error: Exception,
                          batch_id: Optional[str] = None,
                          retry_count: int = 0):
        """Log document processing error."""
        metadata = {
            'document_id': document_id,
            'batch_id': batch_id,
            'retry_count': retry_count
        }
        
        self.enhanced_logger.log_error_with_context(
            f"Document processing failed: {document_id} (retry {retry_count})",
            error,
            operation_name="document_processing",
            additional_context=metadata
        )
    
    def log_batch_start(self, batch_id: str, batch_size: int, total_batches: int,
                       current_batch_index: int):
        """Log start of batch processing."""
        metadata = {
            'batch_id': batch_id,
            'batch_size': batch_size,
            'total_batches': total_batches,
            'current_batch_index': current_batch_index
        }
        
        self._batch_stats[batch_id] = {
            'start_time': time.time(),
            'batch_size': batch_size,
            'processed_count': 0,
            'error_count': 0
        }
        
        self.enhanced_logger.info(
            f"Starting batch processing: {batch_id} "
            f"({current_batch_index + 1}/{total_batches}, {batch_size} documents)",
            operation_name="batch_processing",
            metadata=metadata
        )
    
    def log_batch_progress(self, batch_id: str, completed_docs: int,
                          failed_docs: int, current_memory_mb: float):
        """Log batch processing progress."""
        stats = self._batch_stats.get(batch_id, {})
        batch_size = stats.get('batch_size', 0)
        progress_percent = (completed_docs + failed_docs) / batch_size * 100 if batch_size > 0 else 0
        
        metadata = {
            'batch_id': batch_id,
            'completed_docs': completed_docs,
            'failed_docs': failed_docs,
            'progress_percent': progress_percent,
            'current_memory_mb': current_memory_mb
        }
        
        metrics = PerformanceMetrics(
            memory_mb=current_memory_mb,
            error_count=failed_docs
        )
        
        self.enhanced_logger.info(
            f"Batch progress: {batch_id} - {completed_docs}/{batch_size} completed, "
            f"{failed_docs} failed ({progress_percent:.1f}%, {current_memory_mb:.1f}MB)",
            operation_name="batch_processing",
            performance_metrics=metrics,
            metadata=metadata
        )
    
    def log_batch_complete(self, batch_id: str, successful_docs: int, 
                          failed_docs: int, total_processing_time_ms: float):
        """Log batch processing completion."""
        stats = self._batch_stats.get(batch_id, {})
        batch_size = stats.get('batch_size', successful_docs + failed_docs)
        success_rate = (successful_docs / batch_size * 100) if batch_size > 0 else 0
        
        metadata = {
            'batch_id': batch_id,
            'successful_docs': successful_docs,
            'failed_docs': failed_docs,
            'batch_size': batch_size,
            'success_rate': success_rate,
            'total_processing_time_ms': total_processing_time_ms
        }
        
        metrics = PerformanceMetrics(
            duration_ms=total_processing_time_ms,
            error_count=failed_docs
        )
        
        self.enhanced_logger.info(
            f"Batch processing completed: {batch_id} - "
            f"{successful_docs}/{batch_size} successful ({success_rate:.1f}%), "
            f"took {total_processing_time_ms:.1f}ms",
            operation_name="batch_processing",
            performance_metrics=metrics,
            metadata=metadata
        )
        
        # Clean up batch stats
        self._batch_stats.pop(batch_id, None)


class DiagnosticLogger:
    """Specialized logger for diagnostic information."""
    
    def __init__(self, base_logger: logging.Logger):
        self.enhanced_logger = EnhancedLogger(base_logger, component="diagnostics")
    
    def log_configuration_validation(self, config_name: str, 
                                   validation_results: Dict[str, Any]):
        """Log configuration validation results."""
        metadata = {
            'config_name': config_name,
            'validation_results': validation_results
        }
        
        self.enhanced_logger.info(
            f"Configuration validation: {config_name}",
            operation_name="configuration_validation",
            metadata=metadata
        )
    
    def log_storage_initialization(self, storage_type: str, path: str,
                                 initialization_time_ms: float,
                                 success: bool, error_details: Optional[str] = None):
        """Log storage initialization results."""
        metadata = {
            'storage_type': storage_type,
            'path': path,
            'initialization_time_ms': initialization_time_ms,
            'success': success
        }
        
        if error_details:
            metadata['error_details'] = error_details
        
        metrics = PerformanceMetrics(
            duration_ms=initialization_time_ms
        )
        
        level = 'INFO' if success else 'ERROR'
        message = f"Storage initialization {'succeeded' if success else 'failed'}: {storage_type} at {path}"
        
        self.enhanced_logger._log_structured(
            level, message,
            operation_name="storage_initialization",
            performance_metrics=metrics,
            metadata=metadata
        )
    
    def log_api_call_details(self, api_type: str, model: str, 
                           tokens_used: int, cost: float,
                           response_time_ms: float, success: bool):
        """Log detailed API call information."""
        metadata = {
            'api_type': api_type,
            'model': model,
            'tokens_used': tokens_used,
            'cost': cost,
            'response_time_ms': response_time_ms,
            'success': success
        }
        
        metrics = PerformanceMetrics(
            duration_ms=response_time_ms,
            tokens_used=tokens_used,
            api_calls_count=1
        )
        
        self.enhanced_logger.info(
            f"API call: {api_type} with {model} - {tokens_used} tokens, "
            f"${cost:.4f}, {response_time_ms:.1f}ms",
            operation_name="api_call",
            performance_metrics=metrics,
            metadata=metadata
        )
    
    def log_memory_usage(self, operation_name: str, memory_mb: float, 
                        memory_percent: float, threshold_mb: Optional[float] = None):
        """Log memory usage information."""
        metadata = {
            'operation_name': operation_name,
            'memory_mb': memory_mb,
            'memory_percent': memory_percent,
            'threshold_mb': threshold_mb
        }
        
        metrics = PerformanceMetrics(
            memory_mb=memory_mb,
            memory_percent=memory_percent
        )
        
        # Determine log level based on memory usage
        level = 'DEBUG'
        if threshold_mb and memory_mb > threshold_mb:
            level = 'WARNING'
        elif memory_percent > 80:
            level = 'WARNING'
        elif memory_percent > 90:
            level = 'ERROR'
        
        self.enhanced_logger._log_structured(
            level,
            f"Memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%) in {operation_name}",
            operation_name="memory_monitoring",
            performance_metrics=metrics,
            metadata=metadata
        )


def performance_logged(operation_name: str, logger: Optional[EnhancedLogger] = None):
    """Decorator to automatically log performance metrics for functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from args if not provided
            actual_logger = logger
            if not actual_logger:
                # Try to find logger in the object (if it's a method)
                if args and hasattr(args[0], 'logger'):
                    base_logger = getattr(args[0], 'logger')
                    actual_logger = EnhancedLogger(base_logger)
            
            if not actual_logger:
                # Just call the function if no logger available
                return func(*args, **kwargs)
            
            correlation_id = correlation_manager.generate_correlation_id()
            
            with correlation_manager.operation_context(operation_name):
                tracker = PerformanceTracker()
                tracker.start_tracking()
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = e
                    result = None
                
                # Log performance metrics
                metrics = tracker.get_metrics()
                metrics.duration_ms = (time.time() - start_time) * 1000
                
                if success:
                    actual_logger.log_performance_metrics(operation_name, metrics, 'INFO')
                else:
                    actual_logger.log_performance_metrics(operation_name, metrics, 'ERROR')
                    actual_logger.log_error_with_context(
                        f"Operation failed: {operation_name}",
                        error,
                        operation_name=operation_name
                    )
                    raise error
                
                return result
        
        return wrapper
    return decorator


def create_enhanced_loggers(base_logger: logging.Logger) -> Dict[str, Union[EnhancedLogger, IngestionLogger, DiagnosticLogger]]:
    """
    Create a suite of enhanced loggers from a base logger.
    
    Args:
        base_logger: Base Python logger to enhance
        
    Returns:
        Dict containing enhanced logger instances
    """
    return {
        'enhanced': EnhancedLogger(base_logger),
        'ingestion': IngestionLogger(base_logger),
        'diagnostic': DiagnosticLogger(base_logger)
    }


def setup_structured_logging(logger_name: str, log_file_path: Optional[Path] = None) -> logging.Logger:
    """
    Set up a structured logger with JSON formatting.
    
    Args:
        logger_name: Name of the logger
        log_file_path: Optional path to log file for structured logs
        
    Returns:
        Configured structured logger
    """
    structured_logger = logging.getLogger(f"{logger_name}.structured")
    structured_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    structured_logger.handlers = []
    
    # Create JSON formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            # The message should already be JSON from StructuredLogRecord
            return record.getMessage()
    
    if log_file_path:
        # File handler for structured logs
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(JSONFormatter())
        structured_logger.addHandler(file_handler)
    
    return structured_logger