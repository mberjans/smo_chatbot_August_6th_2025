#!/usr/bin/env python3
"""
Comprehensive Logging Validation Framework for Clinical Metabolomics Oracle.

This module provides extensive testing and validation of the logging system to ensure
that all error scenarios produce appropriate, structured, and complete log entries
with proper correlation ID tracking and performance metrics.

Features:
- Structured log format validation
- Correlation ID tracking across complex scenarios
- Performance metrics logging validation
- Error context and stack trace verification
- Log aggregation and analysis tools
- Real-time log monitoring during error scenarios
- Log retention and rotation testing

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from unittest.mock import Mock, patch

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lightrag_integration.enhanced_logging import (
    EnhancedLogger, IngestionLogger, DiagnosticLogger, CorrelationIDManager,
    CorrelationContext, StructuredLogRecord, PerformanceMetrics, PerformanceTracker,
    correlation_manager
)

from lightrag_integration.clinical_metabolomics_rag import (
    IngestionError, IngestionRetryableError, IngestionAPIError, 
    IngestionNetworkError, StorageInitializationError
)


# =====================================================================
# LOG ANALYSIS AND VALIDATION FRAMEWORK
# =====================================================================

@dataclass
class LogEntry:
    """Represents a parsed log entry for analysis."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    correlation_id: Optional[str] = None
    operation_name: Optional[str] = None
    component: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_message: str = ""


@dataclass
class LogValidationResult:
    """Results of log validation."""
    total_entries: int = 0
    structured_entries: int = 0
    entries_with_correlation_id: int = 0
    error_entries: int = 0
    performance_entries: int = 0
    missing_required_fields: List[str] = field(default_factory=list)
    correlation_id_coverage: float = 0.0
    structured_format_compliance: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    passed: bool = False


class LogCapture:
    """Captures and analyzes log output for validation."""
    
    def __init__(self, logger_name: Optional[str] = None):
        """Initialize log capture."""
        self.logger_name = logger_name
        self.captured_logs: List[LogEntry] = []
        self.handler: Optional[logging.Handler] = None
        self._lock = threading.RLock()
    
    def start_capture(self) -> None:
        """Start capturing log messages."""
        self.handler = LogCaptureHandler(self)
        
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()  # Root logger
        
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)  # Capture all levels
    
    def stop_capture(self) -> None:
        """Stop capturing log messages."""
        if self.handler:
            if self.logger_name:
                logger = logging.getLogger(self.logger_name)
            else:
                logger = logging.getLogger()
            
            logger.removeHandler(self.handler)
            self.handler = None
    
    def add_log_entry(self, record: logging.LogRecord) -> None:
        """Add a captured log entry."""
        with self._lock:
            try:
                # Parse the log record
                entry = self._parse_log_record(record)
                self.captured_logs.append(entry)
            except Exception as e:
                # Don't let log parsing errors break the test
                pass
    
    def _parse_log_record(self, record: logging.LogRecord) -> LogEntry:
        """Parse a logging record into a LogEntry."""
        # Extract correlation ID from message if present
        correlation_id = None
        operation_name = None
        component = None
        error_details = None
        performance_metrics = None
        metadata = {}
        
        message = record.getMessage()
        
        # Try to extract correlation ID from message format [correlation_id]
        correlation_match = re.search(r'\[([a-f0-9-]+)\]', message)
        if correlation_match:
            correlation_id = correlation_match.group(1)
        
        # Try to extract structured data from message
        if hasattr(record, 'correlation_id'):
            correlation_id = record.correlation_id
        
        if hasattr(record, 'operation_name'):
            operation_name = record.operation_name
        
        if hasattr(record, 'component'):
            component = record.component
        
        if hasattr(record, 'error_details'):
            error_details = record.error_details
        
        if hasattr(record, 'performance_metrics'):
            performance_metrics = record.performance_metrics
        
        # Check for JSON-structured messages
        try:
            if message.strip().startswith('{'):
                parsed_message = json.loads(message)
                if isinstance(parsed_message, dict):
                    correlation_id = parsed_message.get('correlation_id', correlation_id)
                    operation_name = parsed_message.get('operation_name', operation_name)
                    component = parsed_message.get('component', component)
                    error_details = parsed_message.get('error_details', error_details)
                    performance_metrics = parsed_message.get('performance_metrics', performance_metrics)
                    metadata.update(parsed_message.get('metadata', {}))
        except (json.JSONDecodeError, TypeError):
            pass
        
        return LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            logger_name=record.name,
            message=message,
            correlation_id=correlation_id,
            operation_name=operation_name,
            component=component,
            error_details=error_details,
            performance_metrics=performance_metrics,
            metadata=metadata,
            raw_message=message
        )
    
    def get_logs_by_correlation_id(self, correlation_id: str) -> List[LogEntry]:
        """Get all log entries with a specific correlation ID."""
        with self._lock:
            return [entry for entry in self.captured_logs 
                   if entry.correlation_id == correlation_id]
    
    def get_logs_by_level(self, level: str) -> List[LogEntry]:
        """Get all log entries of a specific level."""
        with self._lock:
            return [entry for entry in self.captured_logs 
                   if entry.level == level]
    
    def get_logs_by_operation(self, operation_name: str) -> List[LogEntry]:
        """Get all log entries for a specific operation."""
        with self._lock:
            return [entry for entry in self.captured_logs 
                   if entry.operation_name == operation_name]
    
    def clear_logs(self) -> None:
        """Clear all captured logs."""
        with self._lock:
            self.captured_logs.clear()
    
    @contextmanager
    def capture_logs(self):
        """Context manager for capturing logs."""
        self.start_capture()
        try:
            yield self
        finally:
            self.stop_capture()


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures records for analysis."""
    
    def __init__(self, capture: LogCapture):
        """Initialize handler."""
        super().__init__()
        self.capture = capture
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the capture."""
        self.capture.add_log_entry(record)


class LogValidator:
    """Validates log output against requirements."""
    
    def __init__(self):
        """Initialize log validator."""
        self.required_fields = [
            'timestamp', 'level', 'logger_name', 'message'
        ]
        self.optional_structured_fields = [
            'correlation_id', 'operation_name', 'component',
            'error_details', 'performance_metrics', 'metadata'
        ]
    
    def validate_logs(self, logs: List[LogEntry]) -> LogValidationResult:
        """
        Validate a collection of log entries.
        
        Args:
            logs: List of log entries to validate
            
        Returns:
            Validation results
        """
        result = LogValidationResult()
        result.total_entries = len(logs)
        
        if result.total_entries == 0:
            result.validation_errors.append("No log entries found")
            return result
        
        structured_count = 0
        correlation_count = 0
        error_count = 0
        performance_count = 0
        
        for entry in logs:
            # Check for structured format
            if self._is_structured_entry(entry):
                structured_count += 1
            
            # Check for correlation ID
            if entry.correlation_id:
                correlation_count += 1
            
            # Check for error entries
            if entry.level in ['ERROR', 'CRITICAL'] or entry.error_details:
                error_count += 1
            
            # Check for performance entries
            if entry.performance_metrics:
                performance_count += 1
            
            # Validate required fields
            missing_fields = self._check_required_fields(entry)
            result.missing_required_fields.extend(missing_fields)
        
        # Calculate metrics
        result.structured_entries = structured_count
        result.entries_with_correlation_id = correlation_count
        result.error_entries = error_count
        result.performance_entries = performance_count
        
        result.structured_format_compliance = structured_count / result.total_entries
        result.correlation_id_coverage = correlation_count / result.total_entries
        
        # Determine if validation passed
        result.passed = (
            result.structured_format_compliance >= 0.8 and  # 80% structured
            result.correlation_id_coverage >= 0.7 and       # 70% with correlation ID
            len(result.missing_required_fields) == 0        # No missing required fields
        )
        
        if not result.passed:
            if result.structured_format_compliance < 0.8:
                result.validation_errors.append(
                    f"Structured format compliance too low: {result.structured_format_compliance:.1%} < 80%"
                )
            
            if result.correlation_id_coverage < 0.7:
                result.validation_errors.append(
                    f"Correlation ID coverage too low: {result.correlation_id_coverage:.1%} < 70%"
                )
            
            if result.missing_required_fields:
                result.validation_errors.append(
                    f"Missing required fields: {', '.join(set(result.missing_required_fields))}"
                )
        
        return result
    
    def _is_structured_entry(self, entry: LogEntry) -> bool:
        """Check if a log entry follows structured format."""
        # An entry is considered structured if it has at least one optional structured field
        return any([
            entry.correlation_id is not None,
            entry.operation_name is not None,
            entry.component is not None,
            entry.error_details is not None,
            entry.performance_metrics is not None,
            bool(entry.metadata)
        ])
    
    def _check_required_fields(self, entry: LogEntry) -> List[str]:
        """Check for missing required fields in an entry."""
        missing_fields = []
        
        if not entry.timestamp:
            missing_fields.append('timestamp')
        
        if not entry.level:
            missing_fields.append('level')
        
        if not entry.logger_name:
            missing_fields.append('logger_name')
        
        if not entry.message:
            missing_fields.append('message')
        
        return missing_fields
    
    def validate_correlation_id_tracking(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """
        Validate correlation ID tracking across operations.
        
        Args:
            logs: List of log entries to validate
            
        Returns:
            Correlation tracking validation results
        """
        correlation_tracking = defaultdict(list)
        
        # Group logs by correlation ID
        for entry in logs:
            if entry.correlation_id:
                correlation_tracking[entry.correlation_id].append(entry)
        
        results = {
            'total_correlation_ids': len(correlation_tracking),
            'correlation_chains': {},
            'orphaned_operations': [],
            'incomplete_chains': [],
            'tracking_quality': 0.0
        }
        
        complete_chains = 0
        
        for correlation_id, entries in correlation_tracking.items():
            # Sort entries by timestamp
            entries.sort(key=lambda e: e.timestamp)
            
            # Analyze the operation chain
            operations = [e.operation_name for e in entries if e.operation_name]
            
            chain_info = {
                'entries_count': len(entries),
                'duration_seconds': 0.0,
                'operations': operations,
                'has_start': False,
                'has_end': False,
                'error_occurred': False
            }
            
            if entries:
                chain_info['duration_seconds'] = (entries[-1].timestamp - entries[0].timestamp).total_seconds()
                
                # Check for start/end patterns
                first_message = entries[0].message.lower()
                last_message = entries[-1].message.lower()
                
                chain_info['has_start'] = any(keyword in first_message for keyword in ['start', 'begin', 'initialize'])
                chain_info['has_end'] = any(keyword in last_message for keyword in ['complete', 'finish', 'end'])
                
                # Check for errors
                chain_info['error_occurred'] = any(e.level in ['ERROR', 'CRITICAL'] for e in entries)
            
            results['correlation_chains'][correlation_id] = chain_info
            
            # A complete chain has start, some operations, and end
            if chain_info['has_start'] and chain_info['has_end'] and len(operations) > 0:
                complete_chains += 1
            elif not chain_info['has_start'] and not chain_info['has_end']:
                results['orphaned_operations'].append(correlation_id)
            else:
                results['incomplete_chains'].append(correlation_id)
        
        # Calculate tracking quality
        if results['total_correlation_ids'] > 0:
            results['tracking_quality'] = complete_chains / results['total_correlation_ids']
        
        return results


# =====================================================================
# LOGGING VALIDATION TEST SCENARIOS
# =====================================================================

class LoggingValidationScenarios:
    """Test scenarios for comprehensive logging validation."""
    
    def __init__(self, temp_dir: Path):
        """Initialize logging validation scenarios."""
        self.temp_dir = temp_dir
        self.log_capture = LogCapture()
        self.validator = LogValidator()
        
        # Create test loggers
        self.base_logger = logging.getLogger("test_validation")
        self.enhanced_logger = EnhancedLogger(self.base_logger, "logging_validator")
        self.ingestion_logger = IngestionLogger(self.base_logger)
        self.diagnostic_logger = DiagnosticLogger(self.base_logger)
    
    def test_basic_structured_logging(self) -> Dict[str, Any]:
        """Test basic structured logging functionality."""
        with self.log_capture.capture_logs():
            # Test all log levels with structured data
            self.enhanced_logger.debug("Debug message", correlation_id="debug-test-001")
            self.enhanced_logger.info("Info message", operation_name="test_operation")
            self.enhanced_logger.warning("Warning message", metadata={"key": "value"})
            self.enhanced_logger.error("Error message", correlation_id="error-test-001")
            self.enhanced_logger.critical("Critical message", component="test_component")
            
            # Test with correlation context
            with correlation_manager.operation_context("structured_logging_test") as context:
                self.enhanced_logger.info("Message within correlation context")
                self.enhanced_logger.error("Error within correlation context")
        
        # Validate results
        logs = self.log_capture.captured_logs
        validation_result = self.validator.validate_logs(logs)
        
        return {
            "scenario": "basic_structured_logging",
            "logs_captured": len(logs),
            "validation_result": validation_result,
            "structured_compliance": validation_result.structured_format_compliance,
            "correlation_coverage": validation_result.correlation_id_coverage,
            "passed": validation_result.passed
        }
    
    def test_error_logging_with_context(self) -> Dict[str, Any]:
        """Test error logging with full context and stack traces."""
        with self.log_capture.capture_logs():
            # Test different types of errors with context
            try:
                raise ValueError("Test validation error")
            except Exception as e:
                self.enhanced_logger.log_error_with_context(
                    "ValueError occurred during validation",
                    e,
                    operation_name="error_context_test",
                    additional_context={"document_id": "test-doc-001", "batch_id": "test-batch-001"}
                )
            
            try:
                raise IngestionAPIError("API rate limit exceeded", status_code=429, retry_after=60)
            except Exception as e:
                self.enhanced_logger.log_error_with_context(
                    "Ingestion API error",
                    e,
                    operation_name="api_error_test"
                )
            
            try:
                raise StorageInitializationError("Cannot create storage directory", storage_path="/invalid/path")
            except Exception as e:
                self.enhanced_logger.log_error_with_context(
                    "Storage initialization failed",
                    e,
                    operation_name="storage_error_test"
                )
        
        logs = self.log_capture.captured_logs
        error_logs = [log for log in logs if log.level in ['ERROR', 'CRITICAL']]
        
        # Check that error logs have required context
        errors_with_context = 0
        errors_with_stack_trace = 0
        
        for error_log in error_logs:
            if error_log.operation_name and error_log.correlation_id:
                errors_with_context += 1
            
            # Check for stack trace indicators in the message
            if any(indicator in error_log.message.lower() for indicator in ['traceback', 'exception', 'error']):
                errors_with_stack_trace += 1
        
        context_coverage = errors_with_context / len(error_logs) if error_logs else 0
        stack_trace_coverage = errors_with_stack_trace / len(error_logs) if error_logs else 0
        
        return {
            "scenario": "error_logging_with_context",
            "total_errors": len(error_logs),
            "errors_with_context": errors_with_context,
            "errors_with_stack_trace": errors_with_stack_trace,
            "context_coverage": context_coverage,
            "stack_trace_coverage": stack_trace_coverage,
            "passed": context_coverage >= 0.8 and stack_trace_coverage >= 0.8
        }
    
    def test_performance_metrics_logging(self) -> Dict[str, Any]:
        """Test performance metrics logging."""
        with self.log_capture.capture_logs():
            # Create performance metrics
            metrics = PerformanceMetrics(
                cpu_percent=45.5,
                memory_mb=256.0,
                duration_ms=1500.0,
                api_calls_count=3,
                tokens_used=150
            )
            
            self.enhanced_logger.log_performance_metrics("test_operation", metrics)
            
            # Test performance tracking decorator simulation
            with correlation_manager.operation_context("performance_test") as context:
                self.enhanced_logger.info("Starting performance test")
                time.sleep(0.1)  # Simulate work
                
                # Log completion with metrics
                completion_metrics = PerformanceMetrics(
                    cpu_percent=55.0,
                    memory_mb=280.0,
                    duration_ms=100.0
                )
                self.enhanced_logger.log_performance_metrics("performance_test", completion_metrics)
        
        logs = self.log_capture.captured_logs
        performance_logs = [log for log in logs if log.performance_metrics]
        
        # Validate performance log content
        valid_performance_logs = 0
        
        for perf_log in performance_logs:
            if (perf_log.performance_metrics and
                isinstance(perf_log.performance_metrics, dict) and
                'duration_ms' in str(perf_log.performance_metrics)):
                valid_performance_logs += 1
        
        performance_compliance = valid_performance_logs / len(performance_logs) if performance_logs else 0
        
        return {
            "scenario": "performance_metrics_logging",
            "performance_logs_captured": len(performance_logs),
            "valid_performance_logs": valid_performance_logs,
            "performance_compliance": performance_compliance,
            "passed": performance_compliance >= 0.8 and len(performance_logs) >= 2
        }
    
    def test_ingestion_logging_lifecycle(self) -> Dict[str, Any]:
        """Test complete ingestion logging lifecycle."""
        with self.log_capture.capture_logs():
            # Simulate batch processing lifecycle
            batch_id = "test-batch-001"
            documents = ["doc-001", "doc-002", "doc-003"]
            
            # Start batch
            self.ingestion_logger.log_batch_start(
                batch_id,
                batch_size=len(documents),
                total_batches=1,
                current_batch_index=0
            )
            
            # Process documents
            for i, doc_id in enumerate(documents):
                # Start document processing
                self.ingestion_logger.log_document_start(
                    doc_id,
                    f"/test/path/{doc_id}.pdf",
                    batch_id
                )
                
                if i == 1:  # Simulate error on second document
                    error = IngestionAPIError("Rate limit exceeded", status_code=429)
                    self.ingestion_logger.log_document_error(
                        doc_id,
                        error,
                        batch_id=batch_id,
                        retry_count=1
                    )
                else:
                    # Complete document processing
                    self.ingestion_logger.log_document_complete(
                        doc_id,
                        processing_time_ms=1000.0 + (i * 200),
                        pages_processed=5 + i,
                        characters_extracted=2000 + (i * 500),
                        batch_id=batch_id
                    )
            
            # Progress update
            self.ingestion_logger.log_batch_progress(
                batch_id,
                completed_docs=2,
                failed_docs=1,
                current_memory_mb=256.0
            )
            
            # Complete batch
            self.ingestion_logger.log_batch_complete(
                batch_id,
                successful_docs=2,
                failed_docs=1,
                total_processing_time_ms=3000.0
            )
        
        logs = self.log_capture.captured_logs
        
        # Analyze lifecycle completeness
        batch_start_logs = [log for log in logs if "batch start" in log.message.lower()]
        batch_complete_logs = [log for log in logs if "batch complete" in log.message.lower()]
        document_start_logs = [log for log in logs if "document start" in log.message.lower()]
        document_complete_logs = [log for log in logs if "document complete" in log.message.lower()]
        document_error_logs = [log for log in logs if "document error" in log.message.lower()]
        
        # Check correlation ID consistency within batch
        correlation_tracking = self.validator.validate_correlation_id_tracking(logs)
        
        lifecycle_complete = (
            len(batch_start_logs) >= 1 and
            len(batch_complete_logs) >= 1 and
            len(document_start_logs) >= 3 and
            len(document_complete_logs) >= 2 and
            len(document_error_logs) >= 1
        )
        
        return {
            "scenario": "ingestion_logging_lifecycle",
            "total_logs": len(logs),
            "batch_lifecycle_events": len(batch_start_logs) + len(batch_complete_logs),
            "document_lifecycle_events": len(document_start_logs) + len(document_complete_logs) + len(document_error_logs),
            "correlation_tracking_quality": correlation_tracking.get("tracking_quality", 0),
            "lifecycle_complete": lifecycle_complete,
            "passed": lifecycle_complete and correlation_tracking.get("tracking_quality", 0) >= 0.7
        }
    
    def test_diagnostic_logging_coverage(self) -> Dict[str, Any]:
        """Test diagnostic logging coverage."""
        with self.log_capture.capture_logs():
            # Configuration validation logging
            validation_results = {
                "api_key": "valid",
                "model": "gpt-4o-mini",
                "working_dir": "created",
                "errors": []
            }
            
            self.diagnostic_logger.log_configuration_validation(
                "test_config",
                validation_results
            )
            
            # Storage initialization logging
            self.diagnostic_logger.log_storage_initialization(
                storage_type="vector_store",
                path=str(self.temp_dir / "test_storage"),
                initialization_time_ms=500.0,
                success=True
            )
            
            # Failed storage initialization
            self.diagnostic_logger.log_storage_initialization(
                storage_type="graph_store", 
                path="/invalid/path",
                initialization_time_ms=100.0,
                success=False,
                error_details="Permission denied"
            )
            
            # API call logging
            self.diagnostic_logger.log_api_call_details(
                api_type="completion",
                model="gpt-4o-mini",
                tokens_used=150,
                cost=0.0075,
                response_time_ms=1200.0,
                success=True
            )
            
            # Memory usage logging at different levels
            self.diagnostic_logger.log_memory_usage(
                operation_name="normal_operation",
                memory_mb=512.0,
                memory_percent=60.0
            )
            
            self.diagnostic_logger.log_memory_usage(
                operation_name="high_memory_operation",
                memory_mb=2048.0,
                memory_percent=85.0,
                threshold_mb=1024.0
            )
        
        logs = self.log_capture.captured_logs
        
        # Categorize diagnostic logs
        config_logs = [log for log in logs if "configuration" in log.message.lower()]
        storage_logs = [log for log in logs if "storage" in log.message.lower()]
        api_logs = [log for log in logs if "api" in log.message.lower()]
        memory_logs = [log for log in logs if "memory" in log.message.lower()]
        
        diagnostic_coverage = {
            "configuration": len(config_logs) >= 1,
            "storage": len(storage_logs) >= 2,  # Success and failure
            "api_calls": len(api_logs) >= 1,
            "memory_usage": len(memory_logs) >= 2
        }
        
        coverage_score = sum(diagnostic_coverage.values()) / len(diagnostic_coverage)
        
        return {
            "scenario": "diagnostic_logging_coverage",
            "total_diagnostic_logs": len(logs),
            "coverage_by_category": diagnostic_coverage,
            "coverage_score": coverage_score,
            "passed": coverage_score >= 0.8
        }
    
    def test_correlation_id_tracking_complex_scenario(self) -> Dict[str, Any]:
        """Test correlation ID tracking across complex nested operations."""
        with self.log_capture.capture_logs():
            # Simulate complex nested operation
            with correlation_manager.operation_context("parent_operation") as parent_context:
                self.enhanced_logger.info("Starting parent operation")
                
                # Nested operation 1
                with correlation_manager.operation_context("child_operation_1") as child1_context:
                    self.enhanced_logger.info("Starting child operation 1")
                    
                    # Simulate some work with potential error
                    try:
                        raise ValueError("Simulated error in child 1")
                    except Exception as e:
                        self.enhanced_logger.log_error_with_context(
                            "Error in child operation 1",
                            e,
                            operation_name="child_operation_1"
                        )
                    
                    self.enhanced_logger.info("Completed child operation 1")
                
                # Nested operation 2
                with correlation_manager.operation_context("child_operation_2") as child2_context:
                    self.enhanced_logger.info("Starting child operation 2")
                    
                    # More nested levels
                    with correlation_manager.operation_context("grandchild_operation") as grandchild_context:
                        self.enhanced_logger.info("Performing grandchild operation")
                        
                        # Log performance metrics
                        metrics = PerformanceMetrics(
                            cpu_percent=65.0,
                            memory_mb=128.0,
                            duration_ms=800.0
                        )
                        self.enhanced_logger.log_performance_metrics("grandchild_operation", metrics)
                    
                    self.enhanced_logger.info("Completed child operation 2")
                
                self.enhanced_logger.info("Completed parent operation")
        
        logs = self.log_capture.captured_logs
        
        # Analyze correlation ID tracking
        correlation_tracking = self.validator.validate_correlation_id_tracking(logs)
        
        # Check for proper nesting
        correlation_ids = set()
        for log in logs:
            if log.correlation_id:
                correlation_ids.add(log.correlation_id)
        
        # Should have at least 4 different correlation IDs (parent + 2 children + grandchild)
        nested_operations_tracked = len(correlation_ids) >= 4
        
        # Check that each operation has start and completion logs
        operations_with_lifecycle = 0
        for correlation_id in correlation_ids:
            operation_logs = [log for log in logs if log.correlation_id == correlation_id]
            has_start = any("starting" in log.message.lower() for log in operation_logs)
            has_complete = any("completed" in log.message.lower() for log in operation_logs)
            
            if has_start and has_complete:
                operations_with_lifecycle += 1
        
        lifecycle_tracking_quality = operations_with_lifecycle / len(correlation_ids) if correlation_ids else 0
        
        return {
            "scenario": "correlation_id_tracking_complex",
            "total_logs": len(logs),
            "unique_correlation_ids": len(correlation_ids),
            "nested_operations_tracked": nested_operations_tracked,
            "operations_with_complete_lifecycle": operations_with_lifecycle,
            "lifecycle_tracking_quality": lifecycle_tracking_quality,
            "correlation_tracking": correlation_tracking,
            "passed": (nested_operations_tracked and 
                      lifecycle_tracking_quality >= 0.75 and 
                      correlation_tracking.get("tracking_quality", 0) >= 0.6)
        }
    
    def generate_logging_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive logging validation report."""
        scenarios = [
            self.test_basic_structured_logging(),
            self.test_error_logging_with_context(),
            self.test_performance_metrics_logging(),
            self.test_ingestion_logging_lifecycle(),
            self.test_diagnostic_logging_coverage(),
            self.test_correlation_id_tracking_complex_scenario()
        ]
        
        # Clear logs between scenarios
        self.log_capture.clear_logs()
        
        # Calculate overall results
        total_scenarios = len(scenarios)
        passed_scenarios = sum(1 for s in scenarios if s.get("passed", False))
        
        overall_pass_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        report = {
            "logging_validation_report": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "failed_scenarios": total_scenarios - passed_scenarios,
                "overall_pass_rate": overall_pass_rate,
                "overall_status": "PASS" if overall_pass_rate >= 0.8 else "FAIL"
            },
            "scenario_results": {s["scenario"]: s for s in scenarios},
            "recommendations": self._generate_logging_recommendations(scenarios)
        }
        
        return report
    
    def _generate_logging_recommendations(self, scenarios: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on logging validation results."""
        recommendations = []
        
        for scenario in scenarios:
            if not scenario.get("passed", False):
                scenario_name = scenario.get("scenario", "unknown")
                
                if scenario_name == "basic_structured_logging":
                    if scenario.get("structured_compliance", 0) < 0.8:
                        recommendations.append("Improve structured logging compliance - more logs should include structured data")
                    if scenario.get("correlation_coverage", 0) < 0.7:
                        recommendations.append("Increase correlation ID usage in log messages")
                
                elif scenario_name == "error_logging_with_context":
                    if scenario.get("context_coverage", 0) < 0.8:
                        recommendations.append("Ensure all error logs include operation context and correlation IDs")
                    if scenario.get("stack_trace_coverage", 0) < 0.8:
                        recommendations.append("Include stack traces in error logs for debugging")
                
                elif scenario_name == "performance_metrics_logging":
                    recommendations.append("Improve performance metrics logging - ensure all performance-critical operations are logged")
                
                elif scenario_name == "ingestion_logging_lifecycle":
                    if not scenario.get("lifecycle_complete", False):
                        recommendations.append("Ensure complete lifecycle logging for batch and document processing")
                    if scenario.get("correlation_tracking_quality", 0) < 0.7:
                        recommendations.append("Improve correlation ID consistency across ingestion operations")
                
                elif scenario_name == "diagnostic_logging_coverage":
                    recommendations.append("Expand diagnostic logging coverage for system components")
                
                elif scenario_name == "correlation_id_tracking_complex":
                    recommendations.append("Improve correlation ID tracking for nested operations")
        
        # Add general recommendations
        overall_pass_rate = len([s for s in scenarios if s.get("passed", False)]) / len(scenarios)
        
        if overall_pass_rate < 0.8:
            recommendations.append("Overall logging quality needs improvement - review logging practices")
        
        return recommendations


# =====================================================================
# PYTEST INTEGRATION FOR LOGGING VALIDATION
# =====================================================================

import pytest

@pytest.fixture
def logging_scenarios(temp_dir):
    """Create logging validation scenarios."""
    return LoggingValidationScenarios(temp_dir)


class TestLoggingValidation:
    """Pytest integration for logging validation."""
    
    def test_basic_structured_logging(self, logging_scenarios):
        """Test basic structured logging."""
        result = logging_scenarios.test_basic_structured_logging()
        
        assert result["passed"] == True
        assert result["structured_compliance"] >= 0.8
        assert result["correlation_coverage"] >= 0.7
        assert result["logs_captured"] > 0
    
    def test_error_logging_with_context(self, logging_scenarios):
        """Test error logging with context."""
        result = logging_scenarios.test_error_logging_with_context()
        
        assert result["passed"] == True
        assert result["context_coverage"] >= 0.8
        assert result["total_errors"] > 0
    
    def test_performance_metrics_logging(self, logging_scenarios):
        """Test performance metrics logging."""
        result = logging_scenarios.test_performance_metrics_logging()
        
        assert result["passed"] == True
        assert result["performance_logs_captured"] >= 2
    
    def test_ingestion_logging_lifecycle(self, logging_scenarios):
        """Test ingestion logging lifecycle."""
        result = logging_scenarios.test_ingestion_logging_lifecycle()
        
        assert result["passed"] == True
        assert result["lifecycle_complete"] == True
    
    def test_diagnostic_logging_coverage(self, logging_scenarios):
        """Test diagnostic logging coverage."""
        result = logging_scenarios.test_diagnostic_logging_coverage()
        
        assert result["passed"] == True
        assert result["coverage_score"] >= 0.8
    
    def test_correlation_id_tracking_complex(self, logging_scenarios):
        """Test complex correlation ID tracking."""
        result = logging_scenarios.test_correlation_id_tracking_complex_scenario()
        
        assert result["passed"] == True
        assert result["nested_operations_tracked"] == True
        assert result["unique_correlation_ids"] >= 4
    
    def test_comprehensive_logging_validation(self, logging_scenarios):
        """Run comprehensive logging validation."""
        report = logging_scenarios.generate_logging_validation_report()
        
        assert report["logging_validation_report"]["overall_status"] == "PASS"
        assert report["logging_validation_report"]["overall_pass_rate"] >= 0.8


if __name__ == "__main__":
    # Run as standalone script
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        scenarios = LoggingValidationScenarios(Path(temp_dir))
        report = scenarios.generate_logging_validation_report()
        
        print(json.dumps(report, indent=2, default=str))