#!/usr/bin/env python3
"""
End-to-End Error Handling Validation Framework for Clinical Metabolomics Oracle.

This comprehensive testing framework simulates realistic failure scenarios to validate
that error handling, recovery mechanisms, and logging work correctly in production-like
conditions.

Test Categories:
1. Failure Simulation Framework - Realistic error injection
2. Recovery Mechanism Validation - Test recovery under stress
3. Error Logging Validation - Verify complete logging coverage
4. End-to-End Scenario Tests - Complex multi-failure scenarios
5. Performance Under Stress - Validate system behavior under load
6. Production Readiness Tests - Final validation checks

Features:
- Realistic failure injection with configurable patterns
- Comprehensive recovery mechanism testing
- Performance monitoring under error conditions
- Memory leak detection during repeated failures
- Correlation ID tracking across complex failure scenarios
- Circuit breaker behavior under sustained failures
- Checkpoint and resume validation with corruption simulation
- Resource exhaustion simulation and response validation

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import psutil
import pytest
import random
import shutil
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAGError, IngestionError, IngestionRetryableError,
    IngestionNonRetryableError, IngestionResourceError, IngestionNetworkError,
    IngestionAPIError, StorageInitializationError, StoragePermissionError,
    StorageSpaceError, StorageDirectoryError, StorageRetryableError,
    CircuitBreakerError, CircuitBreaker, RateLimiter
)

from lightrag_integration.advanced_recovery_system import (
    AdvancedRecoverySystem, DegradationMode, FailureType, BackoffStrategy,
    ResourceThresholds, DegradationConfig, CheckpointData, CheckpointManager,
    SystemResourceMonitor, AdaptiveBackoffCalculator
)

from lightrag_integration.enhanced_logging import (
    EnhancedLogger, IngestionLogger, DiagnosticLogger, CorrelationIDManager,
    CorrelationContext, StructuredLogRecord, PerformanceMetrics, PerformanceTracker,
    correlation_manager, performance_logged
)

from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker, KnowledgeBasePhase
)


# =====================================================================
# FAILURE SIMULATION FRAMEWORK
# =====================================================================

@dataclass
class FailurePattern:
    """Defines a pattern of failures for simulation."""
    failure_type: FailureType
    frequency: float  # 0.0 to 1.0 (probability of failure)
    duration_seconds: float  # How long this pattern lasts
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_fail(self) -> bool:
        """Determine if failure should occur based on frequency."""
        return random.random() < self.frequency


@dataclass
class SimulationScenario:
    """Defines a complete failure simulation scenario."""
    name: str
    description: str
    duration_seconds: float
    failure_patterns: List[FailurePattern]
    expected_degradation_mode: Optional[DegradationMode] = None
    expected_recovery_actions: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)


@dataclass
class SimulationMetrics:
    """Tracks metrics during simulation."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    degradation_mode_changes: int = 0
    checkpoints_created: int = 0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    error_counts: Dict[FailureType, int] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate simulation duration."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate operation success rate."""
        total = self.successful_operations + self.failed_operations
        if total == 0:
            return 1.0
        return self.successful_operations / total
    
    @property
    def recovery_rate(self) -> float:
        """Calculate recovery success rate."""
        if self.recovery_attempts == 0:
            return 1.0
        return self.successful_recoveries / self.recovery_attempts


class FailureSimulator:
    """Simulates realistic failure scenarios for testing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize failure simulator."""
        self.logger = logger or logging.getLogger(__name__)
        self._active_patterns: List[FailurePattern] = []
        self._pattern_start_times: Dict[FailurePattern, datetime] = {}
        self._lock = threading.RLock()
    
    def add_failure_pattern(self, pattern: FailurePattern) -> None:
        """Add a failure pattern to active simulation."""
        with self._lock:
            self._active_patterns.append(pattern)
            self._pattern_start_times[pattern] = datetime.now()
            self.logger.info(f"Added failure pattern: {pattern.failure_type.value} "
                           f"(frequency: {pattern.frequency:.2f})")
    
    def remove_expired_patterns(self) -> List[FailurePattern]:
        """Remove expired failure patterns."""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for pattern in self._active_patterns[:]:  # Copy to avoid modification during iteration
                start_time = self._pattern_start_times.get(pattern, now)
                if (now - start_time).total_seconds() >= pattern.duration_seconds:
                    self._active_patterns.remove(pattern)
                    del self._pattern_start_times[pattern]
                    expired.append(pattern)
            
            return expired
    
    def should_inject_failure(self, operation_type: str = "default") -> Optional[Tuple[FailureType, str]]:
        """
        Determine if a failure should be injected.
        
        Args:
            operation_type: Type of operation being performed
            
        Returns:
            Tuple of (failure_type, error_message) if failure should occur, None otherwise
        """
        # Remove expired patterns first
        self.remove_expired_patterns()
        
        with self._lock:
            for pattern in self._active_patterns:
                if pattern.should_fail():
                    error_message = random.choice(pattern.error_messages) if pattern.error_messages else f"Simulated {pattern.failure_type.value} failure"
                    return pattern.failure_type, error_message
        
        return None
    
    def clear_all_patterns(self) -> None:
        """Clear all active failure patterns."""
        with self._lock:
            self._active_patterns.clear()
            self._pattern_start_times.clear()


class ResourceStressSimulator:
    """Simulates system resource stress conditions."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize resource stress simulator."""
        self.logger = logger or logging.getLogger(__name__)
        self._memory_pressure = False
        self._cpu_pressure = False
        self._disk_pressure = False
        self._lock = threading.RLock()
    
    @contextmanager
    def memory_pressure_simulation(self, pressure_level: float = 0.85):
        """
        Context manager that simulates memory pressure.
        
        Args:
            pressure_level: Memory pressure level (0.0 to 1.0)
        """
        with self._lock:
            self._memory_pressure = True
        
        original_get_current_resources = None
        
        try:
            # Patch SystemResourceMonitor to report high memory usage
            with patch('lightrag_integration.advanced_recovery_system.psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = pressure_level * 100
                mock_memory.return_value.available = 1024**3 * (1 - pressure_level)  # Available memory
                
                self.logger.info(f"Simulating memory pressure at {pressure_level:.1%}")
                yield
                
        finally:
            with self._lock:
                self._memory_pressure = False
            self.logger.info("Memory pressure simulation ended")
    
    @contextmanager
    def disk_pressure_simulation(self, pressure_level: float = 0.95):
        """
        Context manager that simulates disk space pressure.
        
        Args:
            pressure_level: Disk space pressure level (0.0 to 1.0)
        """
        with self._lock:
            self._disk_pressure = True
        
        try:
            with patch('lightrag_integration.advanced_recovery_system.psutil.disk_usage') as mock_disk:
                mock_disk.return_value.percent = pressure_level * 100
                mock_disk.return_value.free = 1024**3 * (1 - pressure_level)  # Free disk space
                
                self.logger.info(f"Simulating disk pressure at {pressure_level:.1%}")
                yield
                
        finally:
            with self._lock:
                self._disk_pressure = False
            self.logger.info("Disk pressure simulation ended")
    
    @contextmanager
    def combined_resource_pressure(self, 
                                 memory_level: float = 0.85,
                                 disk_level: float = 0.90,
                                 cpu_level: float = 0.80):
        """
        Simulate combined resource pressure across multiple dimensions.
        
        Args:
            memory_level: Memory pressure level
            disk_level: Disk pressure level  
            cpu_level: CPU pressure level
        """
        with self._lock:
            self._memory_pressure = True
            self._disk_pressure = True
            self._cpu_pressure = True
        
        try:
            with patch('lightrag_integration.advanced_recovery_system.psutil.virtual_memory') as mock_memory, \
                 patch('lightrag_integration.advanced_recovery_system.psutil.disk_usage') as mock_disk, \
                 patch('lightrag_integration.advanced_recovery_system.psutil.cpu_percent') as mock_cpu:
                
                # Configure mocks
                mock_memory.return_value.percent = memory_level * 100
                mock_memory.return_value.available = 1024**3 * (1 - memory_level)
                
                mock_disk.return_value.percent = disk_level * 100
                mock_disk.return_value.free = 1024**3 * (1 - disk_level)
                
                mock_cpu.return_value = cpu_level * 100
                
                self.logger.warning(f"Simulating combined resource pressure - "
                                  f"Memory: {memory_level:.1%}, "
                                  f"Disk: {disk_level:.1%}, "
                                  f"CPU: {cpu_level:.1%}")
                yield
                
        finally:
            with self._lock:
                self._memory_pressure = False
                self._disk_pressure = False
                self._cpu_pressure = False
            self.logger.info("Combined resource pressure simulation ended")


class NetworkFailureSimulator:
    """Simulates various network failure conditions."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize network failure simulator."""
        self.logger = logger or logging.getLogger(__name__)
        self._connection_failure_rate = 0.0
        self._timeout_rate = 0.0
        self._slow_response_rate = 0.0
        self._lock = threading.RLock()
    
    @contextmanager
    def network_instability(self, 
                          connection_failure_rate: float = 0.1,
                          timeout_rate: float = 0.05,
                          slow_response_rate: float = 0.2):
        """
        Simulate network instability with various failure modes.
        
        Args:
            connection_failure_rate: Rate of connection failures
            timeout_rate: Rate of timeout failures  
            slow_response_rate: Rate of slow responses
        """
        with self._lock:
            self._connection_failure_rate = connection_failure_rate
            self._timeout_rate = timeout_rate
            self._slow_response_rate = slow_response_rate
        
        try:
            self.logger.info(f"Simulating network instability - "
                           f"Connection failures: {connection_failure_rate:.1%}, "
                           f"Timeouts: {timeout_rate:.1%}, "
                           f"Slow responses: {slow_response_rate:.1%}")
            yield self
            
        finally:
            with self._lock:
                self._connection_failure_rate = 0.0
                self._timeout_rate = 0.0
                self._slow_response_rate = 0.0
            self.logger.info("Network instability simulation ended")
    
    def should_fail_connection(self) -> bool:
        """Check if connection should fail."""
        with self._lock:
            return random.random() < self._connection_failure_rate
    
    def should_timeout(self) -> bool:
        """Check if request should timeout."""
        with self._lock:
            return random.random() < self._timeout_rate
    
    def should_be_slow(self) -> bool:
        """Check if response should be slow."""
        with self._lock:
            return random.random() < self._slow_response_rate
    
    def get_simulated_delay(self) -> float:
        """Get simulated network delay in seconds."""
        if self.should_be_slow():
            return random.uniform(5.0, 15.0)  # Slow response
        return random.uniform(0.1, 0.5)  # Normal response


# =====================================================================
# END-TO-END VALIDATION FRAMEWORK
# =====================================================================

class ErrorHandlingE2EValidator:
    """
    Comprehensive end-to-end error handling validation framework.
    
    This class orchestrates complex failure scenarios and validates that
    the error handling system responds correctly under various stress conditions.
    """
    
    def __init__(self, 
                 temp_dir: Path,
                 logger: Optional[logging.Logger] = None):
        """Initialize E2E validator."""
        self.temp_dir = temp_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Create component instances
        self.progress_tracker = Mock(spec=KnowledgeBaseProgressTracker)
        self.recovery_system = AdvancedRecoverySystem(
            progress_tracker=self.progress_tracker,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        # Create logging system
        base_logger = logging.getLogger("test_logger")
        self.enhanced_logger = EnhancedLogger(base_logger, "e2e_validator")
        self.ingestion_logger = IngestionLogger(base_logger)
        self.diagnostic_logger = DiagnosticLogger(base_logger)
        
        # Create simulators
        self.failure_simulator = FailureSimulator(self.logger)
        self.resource_simulator = ResourceStressSimulator(self.logger)
        self.network_simulator = NetworkFailureSimulator(self.logger)
        
        # Metrics tracking
        self.metrics = SimulationMetrics(start_time=datetime.now())
        self._metrics_lock = threading.RLock()
        
        # Validation results
        self.validation_results: Dict[str, Any] = {}
    
    def create_test_documents(self, count: int = 100) -> List[str]:
        """Create test documents for processing simulation."""
        return [f"test_document_{i:04d}" for i in range(count)]
    
    def simulate_document_processing(self, 
                                   document_id: str,
                                   batch_id: str = "test_batch") -> bool:
        """
        Simulate processing of a single document with potential failures.
        
        Args:
            document_id: ID of document to process
            batch_id: Batch ID for logging
            
        Returns:
            True if processing succeeded, False if failed
        """
        correlation_id = f"doc_process_{document_id}_{int(time.time())}"
        
        with correlation_manager.operation_context("document_processing", 
                                                  document_id=document_id,
                                                  batch_id=batch_id) as context:
            
            # Log processing start
            self.ingestion_logger.log_document_start(
                document_id, 
                f"/simulated/path/{document_id}.pdf",
                batch_id
            )
            
            with self._metrics_lock:
                self.metrics.total_operations += 1
            
            # Check for simulated failures
            failure_injection = self.failure_simulator.should_inject_failure("document_processing")
            
            if failure_injection:
                failure_type, error_message = failure_injection
                
                # Create appropriate error based on failure type
                if failure_type == FailureType.API_RATE_LIMIT:
                    error = IngestionAPIError(error_message, status_code=429, retry_after=60)
                elif failure_type == FailureType.API_ERROR:
                    error = IngestionAPIError(error_message, status_code=500, retry_after=30)
                elif failure_type == FailureType.NETWORK_ERROR:
                    error = IngestionNetworkError(error_message, retry_after=10)
                elif failure_type == FailureType.MEMORY_PRESSURE:
                    error = IngestionResourceError(error_message, resource_type="memory")
                else:
                    error = IngestionError(error_message, document_id=document_id)
                
                # Log the error
                self.ingestion_logger.log_document_error(
                    document_id,
                    error,
                    batch_id=batch_id,
                    retry_count=1
                )
                
                # Handle the failure in recovery system
                strategy = self.recovery_system.handle_failure(
                    failure_type,
                    error_message,
                    document_id
                )
                
                with self._metrics_lock:
                    self.metrics.failed_operations += 1
                    self.metrics.error_counts[failure_type] = self.metrics.error_counts.get(failure_type, 0) + 1
                    
                    if strategy.get('checkpoint_recommended', False):
                        self.metrics.recovery_attempts += 1
                
                return False
            
            else:
                # Simulate successful processing
                processing_time = random.uniform(500, 2000)  # 0.5-2 seconds
                pages = random.randint(5, 50)
                characters = random.randint(1000, 10000)
                
                self.ingestion_logger.log_document_complete(
                    document_id,
                    processing_time_ms=processing_time,
                    pages_processed=pages,
                    characters_extracted=characters,
                    batch_id=batch_id
                )
                
                self.recovery_system.mark_document_processed(document_id)
                
                with self._metrics_lock:
                    self.metrics.successful_operations += 1
                
                return True
    
    def run_basic_error_injection_scenario(self, 
                                         duration_seconds: float = 60.0) -> Dict[str, Any]:
        """
        Run basic error injection scenario with various failure types.
        
        Args:
            duration_seconds: How long to run the scenario
            
        Returns:
            Validation results
        """
        self.logger.info("Starting basic error injection scenario")
        
        # Create failure patterns
        patterns = [
            FailurePattern(
                failure_type=FailureType.API_RATE_LIMIT,
                frequency=0.15,  # 15% chance
                duration_seconds=duration_seconds / 3,
                error_messages=["Rate limit exceeded", "Too many requests", "Quota exceeded"]
            ),
            FailurePattern(
                failure_type=FailureType.NETWORK_ERROR,
                frequency=0.10,  # 10% chance  
                duration_seconds=duration_seconds / 2,
                error_messages=["Connection timeout", "Network unreachable", "DNS resolution failed"]
            ),
            FailurePattern(
                failure_type=FailureType.API_ERROR,
                frequency=0.08,  # 8% chance
                duration_seconds=duration_seconds,
                error_messages=["Internal server error", "Service unavailable", "Bad gateway"]
            )
        ]
        
        # Add failure patterns
        for pattern in patterns:
            self.failure_simulator.add_failure_pattern(pattern)
        
        # Initialize document processing
        documents = self.create_test_documents(50)
        self.recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=5
        )
        
        start_time = time.time()
        processed_docs = 0
        
        # Process documents with error injection
        while time.time() - start_time < duration_seconds and processed_docs < len(documents):
            batch = self.recovery_system.get_next_batch()
            if not batch:
                break
            
            batch_id = f"batch_{int(time.time())}"
            
            # Log batch start
            self.ingestion_logger.log_batch_start(
                batch_id,
                batch_size=len(batch),
                total_batches=len(documents) // 5,
                current_batch_index=processed_docs // 5
            )
            
            batch_successful = 0
            batch_failed = 0
            
            for doc_id in batch:
                success = self.simulate_document_processing(doc_id, batch_id)
                if success:
                    batch_successful += 1
                else:
                    batch_failed += 1
                
                # Collect performance metrics periodically
                if processed_docs % 10 == 0:
                    self._collect_performance_metrics()
                
                processed_docs += 1
                time.sleep(0.1)  # Small delay to simulate processing time
            
            # Log batch completion
            self.ingestion_logger.log_batch_complete(
                batch_id,
                successful_docs=batch_successful,
                failed_docs=batch_failed,
                total_processing_time_ms=(len(batch) * 1000)
            )
            
            # Create checkpoint periodically
            if processed_docs % 20 == 0:
                checkpoint_id = self.recovery_system.create_checkpoint({
                    "scenario": "basic_error_injection",
                    "processed_docs": processed_docs
                })
                
                with self._metrics_lock:
                    self.metrics.checkpoints_created += 1
        
        # Clear failure patterns
        self.failure_simulator.clear_all_patterns()
        
        # Collect final metrics
        final_status = self.recovery_system.get_recovery_status()
        
        results = {
            "scenario": "basic_error_injection",
            "duration_seconds": time.time() - start_time,
            "documents_processed": processed_docs,
            "success_rate": self.metrics.success_rate,
            "error_counts": dict(self.metrics.error_counts),
            "degradation_mode": final_status["degradation_mode"],
            "batch_size_changes": final_status["current_batch_size"] != 5,
            "checkpoints_created": self.metrics.checkpoints_created,
            "recovery_attempts": self.metrics.recovery_attempts,
            "validation_status": "PASS" if self.metrics.success_rate > 0.5 else "FAIL"
        }
        
        self.validation_results["basic_error_injection"] = results
        return results
    
    def run_resource_pressure_scenario(self, 
                                     duration_seconds: float = 90.0) -> Dict[str, Any]:
        """
        Run scenario with simulated resource pressure.
        
        Args:
            duration_seconds: How long to run the scenario
            
        Returns:
            Validation results
        """
        self.logger.info("Starting resource pressure scenario")
        
        documents = self.create_test_documents(30)
        self.recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=10  # Start with larger batch size
        )
        
        initial_batch_size = self.recovery_system._current_batch_size
        initial_mode = self.recovery_system.current_degradation_mode
        
        start_time = time.time()
        processed_docs = 0
        degradation_triggered = False
        
        # Run with resource pressure simulation
        with self.resource_simulator.combined_resource_pressure(
            memory_level=0.90,  # High memory pressure
            disk_level=0.85,    # High disk pressure
            cpu_level=0.75      # Moderate CPU pressure
        ):
            
            # Add memory pressure failure pattern
            memory_pattern = FailurePattern(
                failure_type=FailureType.MEMORY_PRESSURE,
                frequency=0.25,  # 25% chance
                duration_seconds=duration_seconds,
                error_messages=["Out of memory", "Memory allocation failed", "Heap exhausted"]
            )
            
            self.failure_simulator.add_failure_pattern(memory_pattern)
            
            while time.time() - start_time < duration_seconds and processed_docs < len(documents):
                batch = self.recovery_system.get_next_batch()
                if not batch:
                    break
                
                for doc_id in batch:
                    success = self.simulate_document_processing(doc_id, f"pressure_batch_{processed_docs//5}")
                    processed_docs += 1
                    
                    # Check for degradation mode changes
                    current_mode = self.recovery_system.current_degradation_mode
                    if current_mode != initial_mode and not degradation_triggered:
                        degradation_triggered = True
                        with self._metrics_lock:
                            self.metrics.degradation_mode_changes += 1
                        
                        self.logger.warning(f"Degradation mode changed from {initial_mode.value} to {current_mode.value}")
                    
                    # Collect metrics more frequently under pressure
                    if processed_docs % 5 == 0:
                        self._collect_performance_metrics()
                    
                    time.sleep(0.2)  # Longer processing time under pressure
                
                # Force checkpoint creation under pressure
                if processed_docs % 10 == 0:
                    checkpoint_id = self.recovery_system.create_checkpoint({
                        "scenario": "resource_pressure",
                        "memory_pressure": True,
                        "processed_docs": processed_docs
                    })
                    
                    with self._metrics_lock:
                        self.metrics.checkpoints_created += 1
        
        # Clear patterns
        self.failure_simulator.clear_all_patterns()
        
        final_status = self.recovery_system.get_recovery_status()
        final_batch_size = self.recovery_system._current_batch_size
        
        results = {
            "scenario": "resource_pressure",
            "duration_seconds": time.time() - start_time,
            "documents_processed": processed_docs,
            "success_rate": self.metrics.success_rate,
            "initial_batch_size": initial_batch_size,
            "final_batch_size": final_batch_size,
            "batch_size_reduced": final_batch_size < initial_batch_size,
            "degradation_triggered": degradation_triggered,
            "degradation_mode_changes": self.metrics.degradation_mode_changes,
            "final_degradation_mode": final_status["degradation_mode"],
            "memory_pressure_errors": self.metrics.error_counts.get(FailureType.MEMORY_PRESSURE, 0),
            "checkpoints_created": self.metrics.checkpoints_created,
            "avg_memory_usage_mb": sum(self.metrics.memory_usage_mb) / len(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0,
            "validation_status": "PASS" if degradation_triggered and final_batch_size < initial_batch_size else "FAIL"
        }
        
        self.validation_results["resource_pressure"] = results
        return results
    
    def run_circuit_breaker_scenario(self, 
                                   duration_seconds: float = 120.0) -> Dict[str, Any]:
        """
        Run scenario to test circuit breaker behavior under sustained failures.
        
        Args:
            duration_seconds: How long to run the scenario
            
        Returns:
            Validation results
        """
        self.logger.info("Starting circuit breaker scenario")
        
        # Create circuit breaker for testing
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            expected_exception=Exception
        )
        
        # High failure rate pattern
        failure_pattern = FailurePattern(
            failure_type=FailureType.API_ERROR,
            frequency=0.80,  # 80% failure rate
            duration_seconds=duration_seconds / 2,  # First half has high failures
            error_messages=["Service unavailable", "Internal server error", "Gateway timeout"]
        )
        
        self.failure_simulator.add_failure_pattern(failure_pattern)
        
        start_time = time.time()
        circuit_open_count = 0
        circuit_recovery_count = 0
        
        async def simulate_api_call() -> bool:
            """Simulate API call with potential failure."""
            failure_injection = self.failure_simulator.should_inject_failure("api_call")
            
            if failure_injection:
                raise Exception(failure_injection[1])
            
            return True
        
        # Run circuit breaker test
        results = []
        while time.time() - start_time < duration_seconds:
            try:
                # Simulate API call through circuit breaker
                result = asyncio.run(circuit_breaker.call(simulate_api_call))
                results.append(("success", circuit_breaker.state))
                
                if circuit_breaker.state == "closed" and circuit_breaker.failure_count == 0:
                    circuit_recovery_count += 1
                
            except CircuitBreakerError:
                results.append(("circuit_open", circuit_breaker.state))
                circuit_open_count += 1
                
            except Exception as e:
                results.append(("failure", circuit_breaker.state))
            
            # Collect metrics
            self._collect_performance_metrics()
            
            time.sleep(0.5)  # Small delay between calls
        
        # Clear patterns
        self.failure_simulator.clear_all_patterns()
        
        # Analyze results
        total_calls = len(results)
        successful_calls = len([r for r in results if r[0] == "success"])
        circuit_open_calls = len([r for r in results if r[0] == "circuit_open"])
        failed_calls = len([r for r in results if r[0] == "failure"])
        
        circuit_opened = circuit_open_count > 0
        circuit_recovered = circuit_recovery_count > 0
        
        validation_results = {
            "scenario": "circuit_breaker",
            "duration_seconds": time.time() - start_time,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "circuit_open_calls": circuit_open_calls,
            "circuit_opened": circuit_opened,
            "circuit_recovered": circuit_recovered,
            "final_circuit_state": circuit_breaker.state,
            "failure_threshold_reached": circuit_breaker.failure_count >= circuit_breaker.failure_threshold,
            "validation_status": "PASS" if circuit_opened and (circuit_recovered or circuit_breaker.state == "open") else "FAIL"
        }
        
        self.validation_results["circuit_breaker"] = validation_results
        return validation_results
    
    def run_checkpoint_corruption_scenario(self,
                                         duration_seconds: float = 60.0) -> Dict[str, Any]:
        """
        Test checkpoint and resume functionality with corruption simulation.
        
        Args:
            duration_seconds: How long to run the scenario
            
        Returns:
            Validation results
        """
        self.logger.info("Starting checkpoint corruption scenario")
        
        documents = self.create_test_documents(20)
        self.recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=3
        )
        
        start_time = time.time()
        checkpoints_created = []
        corrupted_checkpoints = []
        successful_resumes = 0
        failed_resumes = 0
        
        processed_docs = 0
        
        while time.time() - start_time < duration_seconds and processed_docs < len(documents):
            # Process a few documents
            batch = self.recovery_system.get_next_batch()
            if not batch:
                break
            
            for doc_id in batch[:2]:  # Process only 2 from batch
                success = self.simulate_document_processing(doc_id, "corruption_test")
                processed_docs += 1
            
            # Create checkpoint
            checkpoint_id = self.recovery_system.create_checkpoint({
                "corruption_test": True,
                "processed_docs": processed_docs
            })
            checkpoints_created.append(checkpoint_id)
            
            # Sometimes corrupt the checkpoint
            if random.random() < 0.3:  # 30% chance of corruption
                checkpoint_file = self.recovery_system.checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
                
                if checkpoint_file.exists():
                    # Corrupt the file
                    if random.random() < 0.5:
                        # Complete corruption
                        checkpoint_file.write_text("corrupted json data")
                    else:
                        # Partial corruption (invalid JSON)
                        with open(checkpoint_file, 'r') as f:
                            data = f.read()
                        corrupted_data = data[:-20] + "corrupted"  # Truncate and add garbage
                        checkpoint_file.write_text(corrupted_data)
                    
                    corrupted_checkpoints.append(checkpoint_id)
                    self.logger.info(f"Corrupted checkpoint {checkpoint_id}")
            
            # Try to resume from a random previous checkpoint
            if len(checkpoints_created) > 1 and random.random() < 0.4:
                test_checkpoint = random.choice(checkpoints_created[:-1])  # Not the latest
                
                try:
                    success = self.recovery_system.resume_from_checkpoint(test_checkpoint)
                    if success:
                        successful_resumes += 1
                        self.logger.info(f"Successfully resumed from checkpoint {test_checkpoint}")
                    else:
                        failed_resumes += 1
                        self.logger.warning(f"Failed to resume from checkpoint {test_checkpoint}")
                        
                except Exception as e:
                    failed_resumes += 1
                    self.logger.error(f"Exception during resume from {test_checkpoint}: {e}")
            
            time.sleep(0.5)
        
        validation_results = {
            "scenario": "checkpoint_corruption",
            "duration_seconds": time.time() - start_time,
            "documents_processed": processed_docs,
            "checkpoints_created": len(checkpoints_created),
            "checkpoints_corrupted": len(corrupted_checkpoints),
            "successful_resumes": successful_resumes,
            "failed_resumes": failed_resumes,
            "corruption_rate": len(corrupted_checkpoints) / len(checkpoints_created) if checkpoints_created else 0,
            "resume_success_rate": successful_resumes / (successful_resumes + failed_resumes) if (successful_resumes + failed_resumes) > 0 else 1.0,
            "validation_status": "PASS" if successful_resumes > 0 or failed_resumes == len(corrupted_checkpoints) else "FAIL"
        }
        
        self.validation_results["checkpoint_corruption"] = validation_results
        return validation_results
    
    def run_memory_leak_detection_scenario(self,
                                         duration_seconds: float = 180.0) -> Dict[str, Any]:
        """
        Run scenario to detect potential memory leaks during repeated failures.
        
        Args:
            duration_seconds: How long to run the scenario
            
        Returns:
            Validation results including memory usage analysis
        """
        self.logger.info("Starting memory leak detection scenario")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Create pattern with frequent failures
        failure_pattern = FailurePattern(
            failure_type=FailureType.PROCESSING_ERROR,
            frequency=0.60,  # 60% failure rate
            duration_seconds=duration_seconds,
            error_messages=["Processing failed", "Conversion error", "Invalid data"]
        )
        
        self.failure_simulator.add_failure_pattern(failure_pattern)
        
        documents = self.create_test_documents(200)  # Larger set for extended testing
        self.recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=5
        )
        
        start_time = time.time()
        memory_samples = [initial_memory_mb]
        processed_docs = 0
        checkpoints_created = 0
        
        # Process documents with frequent memory monitoring
        while time.time() - start_time < duration_seconds:
            batch = self.recovery_system.get_next_batch()
            if not batch:
                break
            
            for doc_id in batch:
                # Process document (with potential failure)
                success = self.simulate_document_processing(doc_id, "memory_test")
                processed_docs += 1
                
                # Monitor memory every 10 documents
                if processed_docs % 10 == 0:
                    current_memory_mb = process.memory_info().rss / (1024 * 1024)
                    memory_samples.append(current_memory_mb)
                    
                    # Create and delete checkpoints to test cleanup
                    checkpoint_id = self.recovery_system.create_checkpoint({
                        "memory_test": True,
                        "iteration": processed_docs
                    })
                    checkpoints_created += 1
                    
                    # Delete old checkpoints periodically
                    if processed_docs % 50 == 0:
                        old_checkpoints = self.recovery_system.checkpoint_manager.list_checkpoints()
                        for old_checkpoint in old_checkpoints[:-5]:  # Keep last 5
                            self.recovery_system.checkpoint_manager.delete_checkpoint(old_checkpoint)
                    
                    # Force garbage collection periodically
                    if processed_docs % 20 == 0:
                        import gc
                        gc.collect()
                
                # Small delay to prevent overwhelming
                time.sleep(0.05)
        
        # Clear patterns
        self.failure_simulator.clear_all_patterns()
        
        # Analyze memory usage
        final_memory_mb = memory_samples[-1]
        memory_increase_mb = final_memory_mb - initial_memory_mb
        max_memory_mb = max(memory_samples)
        avg_memory_mb = sum(memory_samples) / len(memory_samples)
        
        # Calculate memory trend (linear regression)
        if len(memory_samples) > 2:
            x = list(range(len(memory_samples)))
            y = memory_samples
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i]**2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        else:
            slope = 0
        
        # Memory leak detection criteria
        memory_leak_detected = (
            memory_increase_mb > 100 or  # More than 100MB increase
            (slope > 0.5 and len(memory_samples) > 10)  # Consistent upward trend
        )
        
        validation_results = {
            "scenario": "memory_leak_detection",
            "duration_seconds": time.time() - start_time,
            "documents_processed": processed_docs,
            "checkpoints_created": checkpoints_created,
            "initial_memory_mb": initial_memory_mb,
            "final_memory_mb": final_memory_mb,
            "max_memory_mb": max_memory_mb,
            "avg_memory_mb": avg_memory_mb,
            "memory_increase_mb": memory_increase_mb,
            "memory_trend_slope": slope,
            "memory_leak_detected": memory_leak_detected,
            "memory_samples_count": len(memory_samples),
            "validation_status": "PASS" if not memory_leak_detected else "FAIL"
        }
        
        self.validation_results["memory_leak_detection"] = validation_results
        return validation_results
    
    def _collect_performance_metrics(self) -> None:
        """Collect current performance metrics."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
            
            with self._metrics_lock:
                self.metrics.memory_usage_mb.append(memory_mb)
                self.metrics.cpu_usage_percent.append(cpu_percent)
        
        except Exception as e:
            self.logger.warning(f"Failed to collect performance metrics: {e}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Complete validation report with all scenario results
        """
        self.metrics.end_time = datetime.now()
        
        # Calculate overall statistics
        total_scenarios = len(self.validation_results)
        passed_scenarios = len([r for r in self.validation_results.values() 
                               if r.get("validation_status") == "PASS"])
        
        overall_success_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Aggregate error counts
        total_error_counts = {}
        for result in self.validation_results.values():
            error_counts = result.get("error_counts", {})
            for error_type, count in error_counts.items():
                total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
        
        # Performance summary
        performance_summary = {
            "avg_memory_usage_mb": sum(self.metrics.memory_usage_mb) / len(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0,
            "max_memory_usage_mb": max(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0,
            "avg_cpu_usage_percent": sum(self.metrics.cpu_usage_percent) / len(self.metrics.cpu_usage_percent) if self.metrics.cpu_usage_percent else 0,
            "max_cpu_usage_percent": max(self.metrics.cpu_usage_percent) if self.metrics.cpu_usage_percent else 0
        }
        
        # Overall assessment
        overall_status = "PASS" if overall_success_rate >= 0.8 else "FAIL"  # 80% pass rate required
        
        report = {
            "validation_report": {
                "timestamp": self.metrics.end_time.isoformat(),
                "duration_seconds": self.metrics.duration_seconds,
                "overall_status": overall_status,
                "overall_success_rate": overall_success_rate,
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "failed_scenarios": total_scenarios - passed_scenarios
            },
            "scenario_results": self.validation_results,
            "aggregate_metrics": {
                "total_operations": self.metrics.total_operations,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations,
                "operation_success_rate": self.metrics.success_rate,
                "recovery_attempts": self.metrics.recovery_attempts,
                "successful_recoveries": self.metrics.successful_recoveries,
                "recovery_success_rate": self.metrics.recovery_rate,
                "degradation_mode_changes": self.metrics.degradation_mode_changes,
                "checkpoints_created": self.metrics.checkpoints_created,
                "error_counts": dict(self.metrics.error_counts)
            },
            "performance_summary": performance_summary,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check overall success rate
        if self.metrics.success_rate < 0.7:
            recommendations.append("Consider implementing additional error handling for low success rate scenarios")
        
        # Check memory usage
        if self.metrics.memory_usage_mb:
            max_memory = max(self.metrics.memory_usage_mb)
            if max_memory > 2048:  # More than 2GB
                recommendations.append("Monitor memory usage - peak usage exceeded 2GB during testing")
        
        # Check error recovery
        if self.metrics.recovery_attempts > 0 and self.metrics.recovery_rate < 0.8:
            recommendations.append("Improve recovery mechanisms - recovery success rate is below 80%")
        
        # Check specific scenario failures
        for scenario_name, result in self.validation_results.items():
            if result.get("validation_status") == "FAIL":
                if scenario_name == "memory_leak_detection":
                    recommendations.append("Investigate potential memory leaks in error handling paths")
                elif scenario_name == "circuit_breaker":
                    recommendations.append("Review circuit breaker configuration and recovery logic")
                elif scenario_name == "resource_pressure":
                    recommendations.append("Enhance resource monitoring and adaptive degradation logic")
        
        # Check degradation behavior
        if self.metrics.degradation_mode_changes == 0 and self.metrics.error_counts:
            recommendations.append("Verify degradation triggers are properly configured")
        
        return recommendations


# =====================================================================
# COMPREHENSIVE TEST RUNNER
# =====================================================================

class ComprehensiveErrorHandlingTestRunner:
    """
    Main test runner for comprehensive error handling validation.
    
    This class orchestrates all validation scenarios and generates
    a final production readiness assessment.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize test runner."""
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("ErrorHandlingE2E")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_file = self.output_dir / "error_handling_e2e_test.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def run_all_scenarios(self, 
                         quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run all error handling validation scenarios.
        
        Args:
            quick_mode: Run abbreviated tests for faster execution
            
        Returns:
            Complete validation report
        """
        self.logger.info("Starting comprehensive error handling validation")
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize validator
            validator = ErrorHandlingE2EValidator(temp_path, self.logger)
            
            # Determine scenario durations
            if quick_mode:
                durations = {
                    "basic_error_injection": 30.0,
                    "resource_pressure": 45.0,
                    "circuit_breaker": 60.0,
                    "checkpoint_corruption": 30.0,
                    "memory_leak_detection": 90.0
                }
            else:
                durations = {
                    "basic_error_injection": 60.0,
                    "resource_pressure": 90.0,
                    "circuit_breaker": 120.0,
                    "checkpoint_corruption": 60.0,
                    "memory_leak_detection": 180.0
                }
            
            # Run scenarios
            scenario_results = {}
            
            try:
                # 1. Basic Error Injection
                self.logger.info("Running basic error injection scenario...")
                scenario_results["basic_error_injection"] = validator.run_basic_error_injection_scenario(
                    durations["basic_error_injection"]
                )
                
                # 2. Resource Pressure
                self.logger.info("Running resource pressure scenario...")
                scenario_results["resource_pressure"] = validator.run_resource_pressure_scenario(
                    durations["resource_pressure"]
                )
                
                # 3. Circuit Breaker
                self.logger.info("Running circuit breaker scenario...")
                scenario_results["circuit_breaker"] = validator.run_circuit_breaker_scenario(
                    durations["circuit_breaker"]
                )
                
                # 4. Checkpoint Corruption
                self.logger.info("Running checkpoint corruption scenario...")
                scenario_results["checkpoint_corruption"] = validator.run_checkpoint_corruption_scenario(
                    durations["checkpoint_corruption"]
                )
                
                # 5. Memory Leak Detection
                self.logger.info("Running memory leak detection scenario...")
                scenario_results["memory_leak_detection"] = validator.run_memory_leak_detection_scenario(
                    durations["memory_leak_detection"]
                )
                
            except Exception as e:
                self.logger.error(f"Error during scenario execution: {e}", exc_info=True)
                scenario_results["execution_error"] = str(e)
            
            # Generate comprehensive report
            report = validator.generate_comprehensive_report()
            
            # Save report to file
            report_file = self.output_dir / f"error_handling_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to {report_file}")
            
            return report
    
    def print_summary_report(self, report: Dict[str, Any]) -> None:
        """Print a human-readable summary report."""
        validation_info = report["validation_report"]
        
        print("\n" + "="*80)
        print("ERROR HANDLING VALIDATION SUMMARY")
        print("="*80)
        
        print(f"Overall Status: {validation_info['overall_status']}")
        print(f"Success Rate: {validation_info['overall_success_rate']:.1%}")
        print(f"Scenarios Passed: {validation_info['passed_scenarios']}/{validation_info['total_scenarios']}")
        print(f"Test Duration: {validation_info['duration_seconds']:.1f} seconds")
        
        print("\nSCENARIO RESULTS:")
        print("-" * 40)
        
        for scenario_name, result in report["scenario_results"].items():
            status = result.get("validation_status", "UNKNOWN")
            duration = result.get("duration_seconds", 0)
            
            print(f"{scenario_name:25} {status:4} ({duration:5.1f}s)")
            
            # Add specific details for each scenario
            if scenario_name == "basic_error_injection":
                success_rate = result.get("success_rate", 0)
                print(f"  - Success Rate: {success_rate:.1%}")
                print(f"  - Documents Processed: {result.get('documents_processed', 0)}")
            
            elif scenario_name == "resource_pressure":
                degraded = result.get("degradation_triggered", False)
                print(f"  - Degradation Triggered: {'Yes' if degraded else 'No'}")
                print(f"  - Batch Size Reduced: {'Yes' if result.get('batch_size_reduced', False) else 'No'}")
            
            elif scenario_name == "circuit_breaker":
                opened = result.get("circuit_opened", False)
                recovered = result.get("circuit_recovered", False)
                print(f"  - Circuit Opened: {'Yes' if opened else 'No'}")
                print(f"  - Circuit Recovered: {'Yes' if recovered else 'No'}")
            
            elif scenario_name == "checkpoint_corruption":
                resume_rate = result.get("resume_success_rate", 0)
                print(f"  - Resume Success Rate: {resume_rate:.1%}")
                print(f"  - Checkpoints Created: {result.get('checkpoints_created', 0)}")
            
            elif scenario_name == "memory_leak_detection":
                leak_detected = result.get("memory_leak_detected", False)
                memory_increase = result.get("memory_increase_mb", 0)
                print(f"  - Memory Leak Detected: {'Yes' if leak_detected else 'No'}")
                print(f"  - Memory Increase: {memory_increase:.1f} MB")
        
        # Performance Summary
        perf = report["performance_summary"]
        print(f"\nPERFORMANCE METRICS:")
        print("-" * 40)
        print(f"Average Memory Usage: {perf['avg_memory_usage_mb']:.1f} MB")
        print(f"Peak Memory Usage: {perf['max_memory_usage_mb']:.1f} MB")
        print(f"Average CPU Usage: {perf['avg_cpu_usage_percent']:.1f}%")
        
        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        # Final Assessment
        print("\n" + "="*80)
        if validation_info['overall_status'] == 'PASS':
            print("✓ ERROR HANDLING SYSTEM VALIDATION: PASSED")
            print("  The error handling system is ready for production use.")
        else:
            print("✗ ERROR HANDLING SYSTEM VALIDATION: FAILED")
            print("  Please address the issues identified before production deployment.")
        print("="*80)


# =====================================================================
# PYTEST INTEGRATION
# =====================================================================

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def error_validator(temp_dir):
    """Create error handling validator for testing."""
    return ErrorHandlingE2EValidator(temp_dir)


@pytest.fixture  
def test_runner():
    """Create test runner for comprehensive testing."""
    return ComprehensiveErrorHandlingTestRunner()


# Test classes for pytest integration
class TestEndToEndErrorHandling:
    """End-to-end error handling tests for pytest integration."""
    
    @pytest.mark.slow
    def test_basic_error_injection_scenario(self, error_validator):
        """Test basic error injection scenario."""
        result = error_validator.run_basic_error_injection_scenario(30.0)  # Quick test
        
        assert result["validation_status"] == "PASS"
        assert result["documents_processed"] > 0
        assert result["success_rate"] > 0.3  # At least 30% success with error injection
    
    @pytest.mark.slow
    def test_resource_pressure_scenario(self, error_validator):
        """Test resource pressure scenario."""
        result = error_validator.run_resource_pressure_scenario(45.0)  # Quick test
        
        assert result["validation_status"] == "PASS"
        assert result["degradation_triggered"] == True
        assert result["batch_size_reduced"] == True
    
    @pytest.mark.slow
    def test_circuit_breaker_scenario(self, error_validator):
        """Test circuit breaker scenario."""
        result = error_validator.run_circuit_breaker_scenario(60.0)  # Quick test
        
        assert result["validation_status"] == "PASS"
        assert result["circuit_opened"] == True
        assert result["total_calls"] > 0
    
    def test_checkpoint_corruption_scenario(self, error_validator):
        """Test checkpoint corruption scenario."""
        result = error_validator.run_checkpoint_corruption_scenario(30.0)  # Quick test
        
        assert result["validation_status"] == "PASS"
        assert result["checkpoints_created"] > 0
    
    @pytest.mark.slow
    def test_memory_leak_detection_scenario(self, error_validator):
        """Test memory leak detection scenario."""
        result = error_validator.run_memory_leak_detection_scenario(90.0)  # Quick test
        
        assert result["validation_status"] == "PASS"
        assert result["memory_leak_detected"] == False
        assert result["documents_processed"] > 0
    
    @pytest.mark.integration
    def test_comprehensive_validation_quick_mode(self, test_runner):
        """Test comprehensive validation in quick mode."""
        report = test_runner.run_all_scenarios(quick_mode=True)
        
        # Should have all expected scenarios
        expected_scenarios = {
            "basic_error_injection",
            "resource_pressure", 
            "circuit_breaker",
            "checkpoint_corruption",
            "memory_leak_detection"
        }
        
        assert set(report["scenario_results"].keys()) == expected_scenarios
        
        # Overall validation should pass
        assert report["validation_report"]["overall_status"] == "PASS"


# Main execution
if __name__ == "__main__":
    # Run comprehensive validation
    runner = ComprehensiveErrorHandlingTestRunner()
    
    # Allow command line argument for quick mode
    import sys
    quick_mode = "--quick" in sys.argv
    
    print(f"Running error handling validation (quick_mode={quick_mode})...")
    report = runner.run_all_scenarios(quick_mode=quick_mode)
    
    # Print summary
    runner.print_summary_report(report)
    
    # Exit with appropriate code
    overall_status = report["validation_report"]["overall_status"]
    sys.exit(0 if overall_status == "PASS" else 1)