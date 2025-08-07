#!/usr/bin/env python3
"""
Comprehensive Error Handling Unit Tests for Clinical Metabolomics Oracle.

This test suite provides complete coverage of all error handling scenarios
implemented for ingestion failures, storage initialization, recovery mechanisms,
and enhanced logging.

Test Categories:
1. Ingestion Error Classification Tests
2. Document Ingestion Error Handling Tests  
3. Storage Initialization Error Tests
4. Advanced Recovery System Tests
5. Enhanced Logging Error Tests
6. Integration Error Scenario Tests

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import pytest
import asyncio
import logging
import json
import time
import psutil
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

# Import the components we're testing
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
# ERROR CLASSIFICATION TESTS
# =====================================================================

class TestIngestionErrorClassification:
    """Test ingestion error hierarchy and classification."""
    
    def test_base_ingestion_error_creation(self):
        """Test basic IngestionError creation with context."""
        error = IngestionError(
            "Test error message",
            document_id="doc123",
            error_code="E001"
        )
        
        assert str(error) == "Test error message"
        assert error.document_id == "doc123"
        assert error.error_code == "E001"
        assert isinstance(error.timestamp, datetime)
    
    def test_ingestion_retryable_error(self):
        """Test IngestionRetryableError with retry delay."""
        error = IngestionRetryableError(
            "Rate limit exceeded",
            document_id="doc456",
            retry_after=60,
            error_code="RATE_LIMIT"
        )
        
        assert str(error) == "Rate limit exceeded"
        assert error.document_id == "doc456"
        assert error.retry_after == 60
        assert error.error_code == "RATE_LIMIT"
        assert isinstance(error, IngestionError)
    
    def test_ingestion_non_retryable_error(self):
        """Test IngestionNonRetryableError for permanent failures."""
        error = IngestionNonRetryableError(
            "Invalid document format",
            document_id="doc789",
            error_code="INVALID_FORMAT"
        )
        
        assert str(error) == "Invalid document format"
        assert error.document_id == "doc789"
        assert error.error_code == "INVALID_FORMAT"
        assert isinstance(error, IngestionError)
    
    def test_ingestion_resource_error(self):
        """Test IngestionResourceError with resource type."""
        error = IngestionResourceError(
            "Insufficient memory",
            document_id="doc101",
            resource_type="memory",
            error_code="OUT_OF_MEMORY"
        )
        
        assert str(error) == "Insufficient memory"
        assert error.resource_type == "memory"
        assert isinstance(error, IngestionError)
    
    def test_ingestion_network_error(self):
        """Test IngestionNetworkError inheritance."""
        error = IngestionNetworkError(
            "Connection timeout",
            retry_after=30
        )
        
        assert str(error) == "Connection timeout"
        assert error.retry_after == 30
        assert isinstance(error, IngestionRetryableError)
    
    def test_ingestion_api_error(self):
        """Test IngestionAPIError with status code."""
        error = IngestionAPIError(
            "API server error",
            status_code=500,
            retry_after=120
        )
        
        assert str(error) == "API server error"
        assert error.status_code == 500
        assert error.retry_after == 120
        assert isinstance(error, IngestionRetryableError)
    
    def test_error_hierarchy_inheritance(self):
        """Test that all error types inherit properly."""
        api_error = IngestionAPIError("API error")
        network_error = IngestionNetworkError("Network error")
        resource_error = IngestionResourceError("Resource error")
        
        # Test inheritance chain
        assert isinstance(api_error, IngestionRetryableError)
        assert isinstance(api_error, IngestionError)
        assert isinstance(api_error, ClinicalMetabolomicsRAGError)
        
        assert isinstance(network_error, IngestionRetryableError)
        assert isinstance(resource_error, IngestionError)


class TestStorageErrorClassification:
    """Test storage error hierarchy and classification."""
    
    def test_base_storage_error_creation(self):
        """Test basic StorageInitializationError creation."""
        error = StorageInitializationError(
            "Storage initialization failed",
            storage_path="/tmp/storage",
            error_code="INIT_FAILED"
        )
        
        assert str(error) == "Storage initialization failed"
        assert error.storage_path == "/tmp/storage"
        assert error.error_code == "INIT_FAILED"
    
    def test_storage_permission_error(self):
        """Test StoragePermissionError with permission details."""
        error = StoragePermissionError(
            "Permission denied",
            storage_path="/restricted/path",
            required_permission="write"
        )
        
        assert str(error) == "Permission denied"
        assert error.storage_path == "/restricted/path"
        assert error.required_permission == "write"
        assert isinstance(error, StorageInitializationError)
    
    def test_storage_space_error(self):
        """Test StorageSpaceError with space details."""
        error = StorageSpaceError(
            "Insufficient disk space",
            storage_path="/full/disk",
            available_space=1024,
            required_space=2048
        )
        
        assert str(error) == "Insufficient disk space"
        assert error.available_space == 1024
        assert error.required_space == 2048
        assert isinstance(error, StorageInitializationError)
    
    def test_storage_directory_error(self):
        """Test StorageDirectoryError with operation details."""
        error = StorageDirectoryError(
            "Cannot create directory",
            storage_path="/readonly/path",
            directory_operation="create"
        )
        
        assert str(error) == "Cannot create directory"
        assert error.directory_operation == "create"
        assert isinstance(error, StorageInitializationError)


# =====================================================================
# CIRCUIT BREAKER TESTS  
# =====================================================================

class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with test settings."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed (normal) state."""
        async def successful_func():
            return "success"
        
        result = await circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self, circuit_breaker):
        """Test circuit breaker counts failures correctly."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Should allow failures up to threshold
        for i in range(2):  # 2 failures, below threshold of 3
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
            assert circuit_breaker.state == "closed"
        
        # Third failure should open circuit
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        assert circuit_breaker.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker blocks calls when open."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Force circuit open by exceeding threshold
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "open"
        
        # Should raise CircuitBreakerError instead of calling function
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery mechanism."""
        async def failing_func():
            raise Exception("Test failure")
        
        async def successful_func():
            return "recovered"
        
        # Force circuit open
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should transition to half-open and succeed
        result = await circuit_breaker.call(successful_func)
        assert result == "recovered"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_rate_limiter_token_acquisition(self):
        """Test rate limiter allows requests within limits."""
        limiter = RateLimiter(max_requests=5, time_window=1.0)
        
        # Should allow up to max_requests
        for _ in range(5):
            assert limiter.acquire() == True
        
        # Should deny additional requests
        assert limiter.acquire() == False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_token_refill(self):
        """Test rate limiter refills tokens over time."""
        limiter = RateLimiter(max_requests=2, time_window=0.5)
        
        # Consume all tokens
        assert limiter.acquire() == True
        assert limiter.acquire() == True
        assert limiter.acquire() == False
        
        # Wait for refill
        await asyncio.sleep(0.6)
        
        # Should have tokens again
        assert limiter.acquire() == True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_token(self):
        """Test rate limiter wait mechanism."""
        limiter = RateLimiter(max_requests=1, time_window=0.2)
        
        # Consume token
        assert limiter.acquire() == True
        
        # Wait for next token (should take ~0.2 seconds)
        start_time = time.time()
        await limiter.wait_for_token()
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.2
        assert elapsed < 0.5  # Reasonable upper bound


# =====================================================================
# ADVANCED RECOVERY SYSTEM TESTS
# =====================================================================

class TestSystemResourceMonitor:
    """Test system resource monitoring."""
    
    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor with test thresholds."""
        thresholds = ResourceThresholds(
            memory_warning_percent=60.0,
            memory_critical_percent=80.0,
            disk_warning_percent=70.0,
            disk_critical_percent=90.0,
            cpu_warning_percent=70.0,
            cpu_critical_percent=85.0
        )
        return SystemResourceMonitor(thresholds)
    
    def test_get_current_resources(self, resource_monitor):
        """Test resource collection."""
        resources = resource_monitor.get_current_resources()
        
        # Should contain expected keys
        expected_keys = ['memory_percent', 'memory_available_gb', 'disk_percent', 
                        'disk_free_gb', 'cpu_percent']
        
        for key in expected_keys:
            assert key in resources
            assert isinstance(resources[key], (int, float))
    
    @patch('psutil.virtual_memory')
    def test_memory_pressure_detection(self, mock_memory, resource_monitor):
        """Test memory pressure detection."""
        # Mock high memory usage
        mock_memory.return_value.percent = 85.0
        mock_memory.return_value.available = 1024**3  # 1GB
        
        with patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_disk.return_value.percent = 50.0
            mock_disk.return_value.free = 10 * 1024**3  # 10GB
            mock_cpu.return_value = 30.0
            
            recommendations = resource_monitor.check_resource_pressure()
            assert 'memory' in recommendations
            assert 'critical' in recommendations['memory']
    
    @patch('psutil.disk_usage')
    def test_disk_pressure_detection(self, mock_disk, resource_monitor):
        """Test disk pressure detection."""
        # Mock high disk usage
        mock_disk.return_value.percent = 95.0
        mock_disk.return_value.free = 512 * 1024**2  # 512MB
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value.percent = 40.0
            mock_memory.return_value.available = 4 * 1024**3  # 4GB
            mock_cpu.return_value = 25.0
            
            recommendations = resource_monitor.check_resource_pressure()
            assert 'disk' in recommendations
            assert 'critical' in recommendations['disk']


class TestAdaptiveBackoffCalculator:
    """Test adaptive backoff calculation."""
    
    @pytest.fixture
    def backoff_calculator(self):
        """Create backoff calculator."""
        return AdaptiveBackoffCalculator()
    
    def test_exponential_backoff(self, backoff_calculator):
        """Test exponential backoff strategy."""
        delays = []
        for attempt in range(1, 5):
            delay = backoff_calculator.calculate_backoff(
                FailureType.API_ERROR,
                attempt,
                BackoffStrategy.EXPONENTIAL,
                base_delay=1.0,
                jitter=False
            )
            delays.append(delay)
        
        # Should increase exponentially: 1, 2, 4, 8
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0
        assert delays[3] == 8.0
    
    def test_linear_backoff(self, backoff_calculator):
        """Test linear backoff strategy."""
        delays = []
        for attempt in range(1, 5):
            delay = backoff_calculator.calculate_backoff(
                FailureType.NETWORK_ERROR,
                attempt,
                BackoffStrategy.LINEAR,
                base_delay=2.0,
                jitter=False
            )
            delays.append(delay)
        
        # Should increase linearly: 2, 4, 6, 8
        assert delays[0] == 2.0
        assert delays[1] == 4.0
        assert delays[2] == 6.0
        assert delays[3] == 8.0
    
    def test_fibonacci_backoff(self, backoff_calculator):
        """Test Fibonacci backoff strategy."""
        delays = []
        for attempt in range(1, 6):
            delay = backoff_calculator.calculate_backoff(
                FailureType.PROCESSING_ERROR,
                attempt,
                BackoffStrategy.FIBONACCI,
                base_delay=1.0,
                jitter=False
            )
            delays.append(delay)
        
        # Should follow Fibonacci: 1, 1, 2, 3, 5
        assert delays[0] == 1.0
        assert delays[1] == 1.0
        assert delays[2] == 2.0
        assert delays[3] == 3.0
        assert delays[4] == 5.0
    
    def test_adaptive_backoff_with_failure_history(self, backoff_calculator):
        """Test adaptive backoff adjusts based on failure history."""
        # Record multiple failures
        for _ in range(15):  # High failure count
            backoff_calculator.calculate_backoff(
                FailureType.API_RATE_LIMIT,
                1,
                BackoffStrategy.ADAPTIVE
            )
        
        # Next backoff should be higher due to failure history
        high_failure_delay = backoff_calculator.calculate_backoff(
            FailureType.API_RATE_LIMIT,
            1,
            BackoffStrategy.ADAPTIVE,
            jitter=False
        )
        
        # Compare with fresh calculator
        fresh_calculator = AdaptiveBackoffCalculator()
        normal_delay = fresh_calculator.calculate_backoff(
            FailureType.API_RATE_LIMIT,
            1,
            BackoffStrategy.ADAPTIVE,
            jitter=False
        )
        
        assert high_failure_delay > normal_delay
    
    def test_backoff_max_delay_limit(self, backoff_calculator):
        """Test backoff respects maximum delay."""
        delay = backoff_calculator.calculate_backoff(
            FailureType.API_ERROR,
            10,  # High attempt count
            BackoffStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=60.0,
            jitter=False
        )
        
        assert delay <= 60.0
    
    def test_success_recording(self, backoff_calculator):
        """Test success recording affects backoff."""
        # Record failures
        for _ in range(5):
            backoff_calculator.calculate_backoff(FailureType.API_ERROR, 1)
        
        # Record successes
        for _ in range(10):
            backoff_calculator.record_success()
        
        # Should result in better backoff due to good success rate
        delay_with_successes = backoff_calculator.calculate_backoff(
            FailureType.API_ERROR,
            1,
            BackoffStrategy.ADAPTIVE,
            jitter=False
        )
        
        # Compare with calculator that has no success history
        no_success_calculator = AdaptiveBackoffCalculator()
        for _ in range(5):
            no_success_calculator.calculate_backoff(FailureType.API_ERROR, 1)
        
        delay_no_successes = no_success_calculator.calculate_backoff(
            FailureType.API_ERROR,
            1,
            BackoffStrategy.ADAPTIVE,
            jitter=False
        )
        
        # Should be similar or better with success history
        assert delay_with_successes <= delay_no_successes * 1.5


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager with temp directory."""
        checkpoint_dir = temp_dir / "checkpoints"
        return CheckpointManager(checkpoint_dir)
    
    def test_create_checkpoint(self, checkpoint_manager):
        """Test checkpoint creation."""
        checkpoint_id = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            processed_documents=["doc1", "doc2"],
            failed_documents={"doc3": "Error message"},
            pending_documents=["doc4", "doc5"],
            current_batch_size=5,
            degradation_mode=DegradationMode.OPTIMAL,
            system_resources={"memory_percent": 50.0},
            error_counts={FailureType.API_ERROR: 2},
            metadata={"test": "data"}
        )
        
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        
        # Verify checkpoint file exists
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
        assert checkpoint_file.exists()
    
    def test_load_checkpoint(self, checkpoint_manager):
        """Test checkpoint loading."""
        # Create checkpoint first
        checkpoint_id = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.STORAGE_INIT,
            processed_documents=["doc1"],
            failed_documents={"doc2": "Failed"},
            pending_documents=["doc3"],
            current_batch_size=3,
            degradation_mode=DegradationMode.ESSENTIAL,
            system_resources={"cpu_percent": 75.0},
            error_counts={FailureType.MEMORY_PRESSURE: 1}
        )
        
        # Load checkpoint
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_id)
        
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == checkpoint_id
        assert checkpoint.phase == KnowledgeBasePhase.STORAGE_INIT
        assert checkpoint.processed_documents == ["doc1"]
        assert checkpoint.failed_documents == {"doc2": "Failed"}
        assert checkpoint.pending_documents == ["doc3"]
        assert checkpoint.current_batch_size == 3
        assert checkpoint.degradation_mode == DegradationMode.ESSENTIAL
    
    def test_load_nonexistent_checkpoint(self, checkpoint_manager):
        """Test loading non-existent checkpoint returns None."""
        result = checkpoint_manager.load_checkpoint("nonexistent-id")
        assert result is None
    
    def test_list_checkpoints(self, checkpoint_manager):
        """Test checkpoint listing."""
        # Create multiple checkpoints
        id1 = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            processed_documents=[],
            failed_documents={},
            pending_documents=["doc1"],
            current_batch_size=1,
            degradation_mode=DegradationMode.OPTIMAL,
            system_resources={},
            error_counts={}
        )
        
        id2 = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.STORAGE_INIT,
            processed_documents=[],
            failed_documents={},
            pending_documents=["doc2"],
            current_batch_size=1,
            degradation_mode=DegradationMode.MINIMAL,
            system_resources={},
            error_counts={}
        )
        
        # List all checkpoints
        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert id1 in all_checkpoints
        assert id2 in all_checkpoints
        
        # List by phase
        ingestion_checkpoints = checkpoint_manager.list_checkpoints(
            KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        assert id1 in ingestion_checkpoints
        assert id2 not in ingestion_checkpoints
    
    def test_delete_checkpoint(self, checkpoint_manager):
        """Test checkpoint deletion."""
        # Create checkpoint
        checkpoint_id = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            processed_documents=[],
            failed_documents={},
            pending_documents=[],
            current_batch_size=1,
            degradation_mode=DegradationMode.OPTIMAL,
            system_resources={},
            error_counts={}
        )
        
        # Verify exists
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
        assert checkpoint_file.exists()
        
        # Delete checkpoint
        result = checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert result == True
        assert not checkpoint_file.exists()
        
        # Try to delete again
        result = checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert result == False
    
    def test_cleanup_old_checkpoints(self, checkpoint_manager):
        """Test cleanup of old checkpoints."""
        # Create checkpoint
        checkpoint_id = checkpoint_manager.create_checkpoint(
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            processed_documents=[],
            failed_documents={},
            pending_documents=[],
            current_batch_size=1,
            degradation_mode=DegradationMode.OPTIMAL,
            system_resources={},
            error_counts={}
        )
        
        # Modify file timestamp to make it old
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        checkpoint_file.touch(exist_ok=True)
        import os
        os.utime(checkpoint_file, (old_time, old_time))
        
        # Cleanup with 24 hour threshold
        deleted_count = checkpoint_manager.cleanup_old_checkpoints(max_age_hours=24)
        
        assert deleted_count == 1
        assert not checkpoint_file.exists()


class TestAdvancedRecoverySystem:
    """Test the complete advanced recovery system."""
    
    @pytest.fixture
    def recovery_system(self, temp_dir):
        """Create recovery system for testing."""
        progress_tracker = Mock(spec=KnowledgeBaseProgressTracker)
        return AdvancedRecoverySystem(
            progress_tracker=progress_tracker,
            checkpoint_dir=temp_dir / "checkpoints"
        )
    
    def test_initialize_ingestion_session(self, recovery_system):
        """Test ingestion session initialization."""
        documents = ["doc1", "doc2", "doc3"]
        recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=2
        )
        
        assert recovery_system._pending_documents == documents
        assert len(recovery_system._processed_documents) == 0
        assert len(recovery_system._failed_documents) == 0
        assert recovery_system._current_phase == KnowledgeBasePhase.DOCUMENT_INGESTION
        assert recovery_system._current_batch_size == 2
        assert recovery_system.current_degradation_mode == DegradationMode.OPTIMAL
    
    def test_handle_api_rate_limit_failure(self, recovery_system):
        """Test handling of API rate limit failures."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        strategy = recovery_system.handle_failure(
            failure_type=FailureType.API_RATE_LIMIT,
            error_message="Rate limit exceeded",
            document_id="doc1"
        )
        
        assert strategy['action'] == 'backoff_and_retry'
        assert strategy['batch_size_adjustment'] == 0.5
        assert 'backoff_seconds' in strategy
        assert recovery_system._error_counts[FailureType.API_RATE_LIMIT] == 1
        assert "doc1" in recovery_system._failed_documents
    
    def test_handle_memory_pressure_failure(self, recovery_system):
        """Test handling of memory pressure failures."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        with patch.object(recovery_system.resource_monitor, 'check_resource_pressure') as mock_check:
            mock_check.return_value = {'memory': 'critical_reduce_batch_size'}
            
            strategy = recovery_system.handle_failure(
                failure_type=FailureType.MEMORY_PRESSURE,
                error_message="Out of memory",
                document_id="doc1"
            )
        
        assert strategy['action'] == 'reduce_resources'
        assert strategy['batch_size_adjustment'] == 0.3
        assert strategy['degradation_needed'] == True
        assert strategy['checkpoint_recommended'] == True
    
    def test_degradation_mode_progression(self, recovery_system):
        """Test degradation mode progression under stress."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=10
        )
        
        # Simulate multiple API errors to trigger degradation
        for i in range(6):  # Exceed threshold
            recovery_system.handle_failure(
                failure_type=FailureType.API_ERROR,
                error_message=f"API error {i}",
                document_id=f"doc{i}"
            )
        
        # Should have degraded from OPTIMAL
        assert recovery_system.current_degradation_mode != DegradationMode.OPTIMAL
        assert recovery_system._current_batch_size < 10  # Should have reduced
    
    def test_get_next_batch_respects_degradation(self, recovery_system):
        """Test batch sizing respects degradation mode."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3", "doc4", "doc5"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=5
        )
        
        # Normal mode - should get full batch
        batch = recovery_system.get_next_batch()
        assert len(batch) == 5
        
        # Force degradation to SAFE mode
        recovery_system.current_degradation_mode = DegradationMode.SAFE
        recovery_system._update_degradation_config()
        
        batch = recovery_system.get_next_batch()
        assert len(batch) == 1  # SAFE mode uses batch size 1
    
    def test_mark_document_processed(self, recovery_system):
        """Test marking documents as processed."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        recovery_system.mark_document_processed("doc1")
        
        assert "doc1" in recovery_system._processed_documents
        assert "doc1" not in recovery_system._pending_documents
    
    def test_checkpoint_and_resume_functionality(self, recovery_system):
        """Test checkpoint creation and resume functionality."""
        # Initialize session
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=2
        )
        
        # Process some documents
        recovery_system.mark_document_processed("doc1")
        recovery_system.handle_failure(
            FailureType.NETWORK_ERROR,
            "Network timeout",
            "doc2"
        )
        
        # Create checkpoint
        checkpoint_id = recovery_system.create_checkpoint({"test": "metadata"})
        assert checkpoint_id is not None
        
        # Reset system (simulate restart)
        recovery_system.initialize_ingestion_session(
            documents=["doc4", "doc5"],  # Different documents
            phase=KnowledgeBasePhase.STORAGE_INIT,
            batch_size=1
        )
        
        # Resume from checkpoint
        success = recovery_system.resume_from_checkpoint(checkpoint_id)
        assert success == True
        
        # Verify state restored
        assert len(recovery_system._processed_documents) == 1
        assert "doc1" in recovery_system._processed_documents
        assert "doc2" in recovery_system._failed_documents
        assert "doc3" in recovery_system._pending_documents
        assert recovery_system._current_phase == KnowledgeBasePhase.DOCUMENT_INGESTION
    
    def test_get_recovery_status(self, recovery_system):
        """Test recovery status reporting."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        recovery_system.mark_document_processed("doc1")
        recovery_system.handle_failure(
            FailureType.API_ERROR,
            "API failed",
            "doc2"
        )
        
        status = recovery_system.get_recovery_status()
        
        assert status['degradation_mode'] == DegradationMode.OPTIMAL.value
        assert status['document_progress']['total'] == 3
        assert status['document_progress']['processed'] == 1
        assert status['document_progress']['failed'] == 1
        assert status['document_progress']['pending'] == 1
        assert status['error_counts'][FailureType.API_ERROR.value] == 1
        assert 'system_resources' in status
        assert 'resource_pressure' in status


# =====================================================================
# ENHANCED LOGGING TESTS
# =====================================================================

class TestCorrelationIDManager:
    """Test correlation ID management."""
    
    @pytest.fixture
    def correlation_manager(self):
        """Create fresh correlation manager."""
        return CorrelationIDManager()
    
    def test_generate_correlation_id(self, correlation_manager):
        """Test correlation ID generation."""
        id1 = correlation_manager.generate_correlation_id()
        id2 = correlation_manager.generate_correlation_id()
        
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0
    
    def test_context_management(self, correlation_manager):
        """Test context setting and retrieval."""
        context = CorrelationContext(
            correlation_id="test-id",
            operation_name="test_operation",
            start_time=datetime.now(),
            metadata={"key": "value"}
        )
        
        # Initially no context
        assert correlation_manager.get_context() is None
        
        # Set context
        correlation_manager.set_context(context)
        retrieved = correlation_manager.get_context()
        
        assert retrieved is not None
        assert retrieved.correlation_id == "test-id"
        assert retrieved.operation_name == "test_operation"
        assert retrieved.metadata["key"] == "value"
    
    def test_context_stack(self, correlation_manager):
        """Test context stacking for nested operations."""
        context1 = CorrelationContext(
            correlation_id="id1",
            operation_name="op1",
            start_time=datetime.now()
        )
        
        context2 = CorrelationContext(
            correlation_id="id2",
            operation_name="op2",
            start_time=datetime.now()
        )
        
        # Set first context
        correlation_manager.set_context(context1)
        assert correlation_manager.get_context().correlation_id == "id1"
        
        # Set second context (stacked)
        correlation_manager.set_context(context2)
        assert correlation_manager.get_context().correlation_id == "id2"
        
        # Pop second context
        popped = correlation_manager.pop_context()
        assert popped.correlation_id == "id2"
        assert correlation_manager.get_context().correlation_id == "id1"
    
    def test_operation_context_manager(self, correlation_manager):
        """Test operation context manager."""
        with correlation_manager.operation_context(
            "test_operation",
            user_id="user123",
            metadata={"test": "data"}
        ) as context:
            assert context.operation_name == "test_operation"
            assert context.user_id == "user123"
            assert context.metadata["test"] == "data"
            
            # Context should be active
            current = correlation_manager.get_context()
            assert current.correlation_id == context.correlation_id
        
        # Context should be cleared after exiting
        assert correlation_manager.get_context() is None


class TestStructuredLogRecord:
    """Test structured log record creation."""
    
    def test_structured_log_record_creation(self):
        """Test basic log record creation."""
        record = StructuredLogRecord(
            level="INFO",
            message="Test message",
            correlation_id="test-id",
            operation_name="test_op",
            component="test_component"
        )
        
        assert record.level == "INFO"
        assert record.message == "Test message"
        assert record.correlation_id == "test-id"
        assert record.operation_name == "test_op"
        assert record.component == "test_component"
        assert isinstance(record.timestamp, datetime)
    
    def test_structured_log_record_with_error_details(self):
        """Test log record with error details."""
        error_details = {
            "error_type": "ValueError",
            "error_message": "Invalid input",
            "stack_trace": "Stack trace here"
        }
        
        record = StructuredLogRecord(
            level="ERROR",
            message="Error occurred",
            error_details=error_details
        )
        
        assert record.error_details == error_details
    
    def test_structured_log_record_serialization(self):
        """Test log record JSON serialization."""
        metrics = PerformanceMetrics(
            cpu_percent=25.0,
            memory_mb=512.0,
            duration_ms=1500.0
        )
        
        record = StructuredLogRecord(
            level="INFO",
            message="Performance test",
            performance_metrics=metrics,
            metadata={"test": "data"}
        )
        
        # Test dictionary conversion
        record_dict = record.to_dict()
        assert record_dict["level"] == "INFO"
        assert record_dict["message"] == "Performance test"
        assert "performance_metrics" in record_dict
        assert record_dict["metadata"]["test"] == "data"
        
        # Test JSON serialization
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["level"] == "INFO"


class TestEnhancedLogger:
    """Test enhanced logger functionality."""
    
    @pytest.fixture
    def enhanced_logger(self):
        """Create enhanced logger for testing."""
        base_logger = Mock(spec=logging.Logger)
        return EnhancedLogger(base_logger, component="test_component")
    
    def test_structured_logging_methods(self, enhanced_logger):
        """Test structured logging methods."""
        # Test each log level
        enhanced_logger.debug("Debug message", correlation_id="debug-id")
        enhanced_logger.info("Info message", operation_name="test_op")
        enhanced_logger.warning("Warning message", metadata={"key": "value"})
        enhanced_logger.error("Error message", correlation_id="error-id")
        enhanced_logger.critical("Critical message")
        
        # Verify base logger was called
        assert enhanced_logger.base_logger.debug.called
        assert enhanced_logger.base_logger.info.called
        assert enhanced_logger.base_logger.warning.called
        assert enhanced_logger.base_logger.error.called
        assert enhanced_logger.base_logger.critical.called
    
    def test_error_logging_with_context(self, enhanced_logger):
        """Test error logging with full context."""
        error = ValueError("Test error")
        
        enhanced_logger.log_error_with_context(
            "Operation failed",
            error,
            operation_name="test_operation",
            additional_context={"document_id": "doc123"}
        )
        
        # Verify error logging was called
        enhanced_logger.base_logger.error.assert_called()
        call_args = enhanced_logger.base_logger.error.call_args
        
        # Should include correlation ID in message
        assert "[" in call_args[0][0]  # Message should contain correlation ID
        assert call_args[1]["exc_info"] == True  # Should include exception info
    
    def test_performance_metrics_logging(self, enhanced_logger):
        """Test performance metrics logging."""
        metrics = PerformanceMetrics(
            cpu_percent=50.0,
            memory_mb=1024.0,
            duration_ms=2000.0,
            api_calls_count=5,
            tokens_used=100
        )
        
        enhanced_logger.log_performance_metrics("test_operation", metrics)
        
        # Verify info logging was called
        enhanced_logger.base_logger.info.assert_called()


class TestIngestionLogger:
    """Test ingestion-specific logging."""
    
    @pytest.fixture
    def ingestion_logger(self):
        """Create ingestion logger for testing."""
        base_logger = Mock(spec=logging.Logger)
        return IngestionLogger(base_logger)
    
    def test_document_processing_lifecycle(self, ingestion_logger):
        """Test complete document processing logging."""
        # Start processing
        ingestion_logger.log_document_start(
            "doc123",
            "/path/to/doc.pdf",
            "batch-456"
        )
        
        # Complete processing
        ingestion_logger.log_document_complete(
            "doc123",
            processing_time_ms=1500.0,
            pages_processed=10,
            characters_extracted=5000,
            batch_id="batch-456"
        )
        
        # Verify logging calls
        assert ingestion_logger.enhanced_logger.base_logger.info.call_count == 2
    
    def test_document_error_logging(self, ingestion_logger):
        """Test document error logging."""
        error = Exception("Processing failed")
        
        ingestion_logger.log_document_error(
            "doc123",
            error,
            batch_id="batch-456",
            retry_count=2
        )
        
        # Should call error logging
        ingestion_logger.enhanced_logger.base_logger.error.assert_called()
    
    def test_batch_processing_lifecycle(self, ingestion_logger):
        """Test batch processing logging."""
        # Start batch
        ingestion_logger.log_batch_start(
            "batch-123",
            batch_size=5,
            total_batches=10,
            current_batch_index=2
        )
        
        # Progress update
        ingestion_logger.log_batch_progress(
            "batch-123",
            completed_docs=3,
            failed_docs=1,
            current_memory_mb=512.0
        )
        
        # Complete batch
        ingestion_logger.log_batch_complete(
            "batch-123",
            successful_docs=4,
            failed_docs=1,
            total_processing_time_ms=10000.0
        )
        
        # Verify all logging calls
        assert ingestion_logger.enhanced_logger.base_logger.info.call_count == 3


class TestDiagnosticLogger:
    """Test diagnostic logging functionality."""
    
    @pytest.fixture
    def diagnostic_logger(self):
        """Create diagnostic logger for testing."""
        base_logger = Mock(spec=logging.Logger)
        return DiagnosticLogger(base_logger)
    
    def test_configuration_validation_logging(self, diagnostic_logger):
        """Test configuration validation logging."""
        validation_results = {
            "api_key": "valid",
            "model": "valid",
            "working_dir": "created",
            "errors": []
        }
        
        diagnostic_logger.log_configuration_validation(
            "lightrag_config",
            validation_results
        )
        
        diagnostic_logger.enhanced_logger.base_logger.info.assert_called()
    
    def test_storage_initialization_logging(self, diagnostic_logger):
        """Test storage initialization logging."""
        # Successful initialization
        diagnostic_logger.log_storage_initialization(
            storage_type="vector_store",
            path="/tmp/storage",
            initialization_time_ms=500.0,
            success=True
        )
        
        # Failed initialization
        diagnostic_logger.log_storage_initialization(
            storage_type="graph_store",
            path="/invalid/path",
            initialization_time_ms=100.0,
            success=False,
            error_details="Permission denied"
        )
        
        # Should have called both info and error logging
        call_count = (diagnostic_logger.enhanced_logger.base_logger.info.call_count + 
                     diagnostic_logger.enhanced_logger.base_logger.error.call_count)
        assert call_count == 2
    
    def test_api_call_logging(self, diagnostic_logger):
        """Test API call logging."""
        diagnostic_logger.log_api_call_details(
            api_type="completion",
            model="gpt-4o-mini",
            tokens_used=150,
            cost=0.0075,
            response_time_ms=1200.0,
            success=True
        )
        
        diagnostic_logger.enhanced_logger.base_logger.info.assert_called()
    
    def test_memory_usage_logging(self, diagnostic_logger):
        """Test memory usage logging with different levels."""
        # Normal memory usage (debug level)
        diagnostic_logger.log_memory_usage(
            operation_name="normal_operation",
            memory_mb=512.0,
            memory_percent=60.0
        )
        
        # High memory usage (warning level)
        diagnostic_logger.log_memory_usage(
            operation_name="high_memory_operation",
            memory_mb=2048.0,
            memory_percent=85.0,
            threshold_mb=1024.0
        )
        
        # Critical memory usage (error level)
        diagnostic_logger.log_memory_usage(
            operation_name="critical_memory_operation",
            memory_mb=4096.0,
            memory_percent=95.0
        )
        
        # Should have made multiple logging calls at different levels
        total_calls = (diagnostic_logger.enhanced_logger.base_logger.debug.call_count +
                      diagnostic_logger.enhanced_logger.base_logger.warning.call_count +
                      diagnostic_logger.enhanced_logger.base_logger.error.call_count)
        assert total_calls == 3


class TestPerformanceTracker:
    """Test performance tracking functionality."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Create performance tracker for testing."""
        return PerformanceTracker()
    
    @patch('psutil.Process')
    def test_start_tracking(self, mock_process, performance_tracker):
        """Test starting performance tracking."""
        # Mock process stats
        mock_cpu_times = Mock()
        mock_cpu_times.user = 1.5
        mock_cpu_times.system = 0.5
        
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 512  # 512MB
        
        mock_process.return_value.cpu_times.return_value = mock_cpu_times
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.io_counters.return_value = None
        
        # Start tracking
        start_stats = performance_tracker.start_tracking()
        
        assert 'cpu_times' in start_stats
        assert 'memory_info' in start_stats
        assert 'timestamp' in start_stats
    
    @patch('psutil.Process')
    def test_get_metrics(self, mock_process, performance_tracker):
        """Test getting performance metrics."""
        # Mock current stats
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 256  # 256MB
        
        mock_process.return_value.cpu_percent.return_value = 45.0
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.memory_percent.return_value = 30.0
        
        # Get metrics
        metrics = performance_tracker.get_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.cpu_percent == 45.0
        assert metrics.memory_mb == 256.0
        assert metrics.memory_percent == 30.0


class TestPerformanceLoggedDecorator:
    """Test performance logging decorator."""
    
    def test_performance_logged_decorator(self):
        """Test performance logging decorator."""
        base_logger = Mock(spec=logging.Logger)
        enhanced_logger = EnhancedLogger(base_logger)
        
        @performance_logged("test_operation", enhanced_logger)
        def test_function(x, y):
            time.sleep(0.01)  # Small delay for measurable duration
            return x + y
        
        result = test_function(5, 3)
        
        assert result == 8
        # Should have logged performance metrics
        enhanced_logger.base_logger.info.assert_called()
    
    def test_performance_logged_decorator_with_error(self):
        """Test performance logging decorator handles errors."""
        base_logger = Mock(spec=logging.Logger)
        enhanced_logger = EnhancedLogger(base_logger)
        
        @performance_logged("failing_operation", enhanced_logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Should have logged error
        enhanced_logger.base_logger.error.assert_called()


# =====================================================================
# INTEGRATION ERROR SCENARIO TESTS
# =====================================================================

class TestIngestionErrorIntegration:
    """Test integration of error handling components."""
    
    @pytest.fixture
    def integrated_system(self, temp_dir):
        """Create integrated system for testing."""
        progress_tracker = Mock(spec=KnowledgeBaseProgressTracker)
        recovery_system = AdvancedRecoverySystem(
            progress_tracker=progress_tracker,
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        base_logger = Mock(spec=logging.Logger)
        ingestion_logger = IngestionLogger(base_logger)
        
        return {
            'recovery_system': recovery_system,
            'ingestion_logger': ingestion_logger,
            'progress_tracker': progress_tracker
        }
    
    def test_full_error_recovery_workflow(self, integrated_system):
        """Test complete error recovery workflow."""
        recovery = integrated_system['recovery_system']
        logger = integrated_system['ingestion_logger']
        
        # Initialize processing session
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        recovery.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=3
        )
        
        # Simulate processing with errors
        batch1 = recovery.get_next_batch()
        assert len(batch1) == 3
        
        # Process first document successfully
        logger.log_document_start(batch1[0], f"/path/{batch1[0]}.pdf")
        recovery.mark_document_processed(batch1[0])
        logger.log_document_complete(batch1[0], 1000.0, 5, 2000)
        
        # Second document fails with retryable error
        error = IngestionAPIError("Rate limit exceeded", status_code=429, retry_after=60)
        logger.log_document_error(batch1[1], error, retry_count=1)
        strategy = recovery.handle_failure(
            FailureType.API_RATE_LIMIT,
            str(error),
            batch1[1]
        )
        
        # Verify recovery strategy
        assert strategy['action'] == 'backoff_and_retry'
        assert recovery._current_batch_size < 3  # Batch size reduced
        
        # Create checkpoint
        checkpoint_id = recovery.create_checkpoint({"integration_test": True})
        
        # Verify system state
        status = recovery.get_recovery_status()
        assert status['document_progress']['processed'] == 1
        assert status['document_progress']['failed'] == 1
        assert status['document_progress']['pending'] == 3
    
    def test_multiple_error_types_handling(self, integrated_system):
        """Test handling multiple error types in sequence."""
        recovery = integrated_system['recovery_system']
        
        recovery.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # API error
        recovery.handle_failure(
            FailureType.API_ERROR,
            "Server error",
            "doc1"
        )
        
        # Network error  
        recovery.handle_failure(
            FailureType.NETWORK_ERROR,
            "Connection timeout",
            "doc2"
        )
        
        # Memory pressure
        with patch.object(recovery.resource_monitor, 'check_resource_pressure') as mock_check:
            mock_check.return_value = {'memory': 'critical_reduce_batch_size'}
            
            recovery.handle_failure(
                FailureType.MEMORY_PRESSURE,
                "Out of memory",
                "doc3"
            )
        
        # Verify degradation occurred
        assert recovery.current_degradation_mode != DegradationMode.OPTIMAL
        
        # Verify error counts
        error_counts = recovery._error_counts
        assert error_counts[FailureType.API_ERROR] == 1
        assert error_counts[FailureType.NETWORK_ERROR] == 1
        assert error_counts[FailureType.MEMORY_PRESSURE] == 1
    
    def test_storage_error_handling_integration(self, integrated_system):
        """Test storage error handling integration."""
        recovery = integrated_system['recovery_system']
        logger = integrated_system['ingestion_logger']
        
        # Simulate storage initialization phase
        recovery.initialize_ingestion_session(
            documents=[],
            phase=KnowledgeBasePhase.STORAGE_INIT
        )
        
        # Test different storage errors
        permission_error = StoragePermissionError(
            "Cannot write to directory",
            storage_path="/readonly/path",
            required_permission="write"
        )
        
        space_error = StorageSpaceError(
            "Insufficient disk space",
            storage_path="/full/disk",
            available_space=1024,
            required_space=2048
        )
        
        directory_error = StorageDirectoryError(
            "Cannot create directory",
            storage_path="/invalid/path",
            directory_operation="create"
        )
        
        # Each error should be handled appropriately
        for error in [permission_error, space_error, directory_error]:
            # Log the error (in real scenario, this would be done by storage system)
            logger.enhanced_logger.log_error_with_context(
                f"Storage error: {error}",
                error,
                operation_name="storage_initialization"
            )
        
        # Verify error logging occurred
        assert logger.enhanced_logger.base_logger.error.call_count == 3


class TestErrorHandlingEdgeCases:
    """Test error handling edge cases and boundary conditions."""
    
    @pytest.fixture
    def recovery_system(self, temp_dir):
        """Create recovery system for edge case testing."""
        return AdvancedRecoverySystem(checkpoint_dir=temp_dir / "checkpoints")
    
    def test_empty_document_list_handling(self, recovery_system):
        """Test handling of empty document list."""
        recovery_system.initialize_ingestion_session(
            documents=[],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        batch = recovery_system.get_next_batch()
        assert len(batch) == 0
        
        status = recovery_system.get_recovery_status()
        assert status['document_progress']['total'] == 0
    
    def test_all_documents_failed_scenario(self, recovery_system):
        """Test scenario where all documents fail."""
        documents = ["doc1", "doc2", "doc3"]
        recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Fail all documents
        for doc_id in documents:
            recovery_system.handle_failure(
                FailureType.PROCESSING_ERROR,
                f"Processing failed for {doc_id}",
                doc_id
            )
        
        status = recovery_system.get_recovery_status()
        assert status['document_progress']['failed'] == 3
        assert status['document_progress']['pending'] == 0
        assert status['document_progress']['success_rate'] == 0.0
    
    def test_rapid_consecutive_failures(self, recovery_system):
        """Test handling of rapid consecutive failures."""
        recovery_system.initialize_ingestion_session(
            documents=["doc1"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
            batch_size=10
        )
        
        # Generate many failures quickly
        for i in range(20):
            recovery_system.handle_failure(
                FailureType.API_RATE_LIMIT,
                f"Failure {i}",
                "doc1"
            )
        
        # Should have heavily degraded
        assert recovery_system.current_degradation_mode in [
            DegradationMode.SAFE, DegradationMode.MINIMAL
        ]
        assert recovery_system._current_batch_size == 1
    
    def test_checkpoint_corruption_handling(self, recovery_system):
        """Test handling of corrupted checkpoint files."""
        # Create checkpoint
        checkpoint_id = recovery_system.create_checkpoint(
            {"test": "data"}
        )
        
        # Corrupt the checkpoint file
        checkpoint_file = recovery_system.checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.json"
        checkpoint_file.write_text("corrupted json content")
        
        # Attempt to load corrupted checkpoint
        result = recovery_system.resume_from_checkpoint(checkpoint_id)
        assert result == False  # Should fail gracefully
    
    def test_extreme_resource_pressure(self, recovery_system):
        """Test behavior under extreme resource pressure."""
        with patch.object(recovery_system.resource_monitor, 'check_resource_pressure') as mock_check:
            # Simulate critical resource pressure on all fronts
            mock_check.return_value = {
                'memory': 'critical_reduce_batch_size',
                'cpu': 'critical_reduce_concurrency',
                'disk': 'critical_cleanup_required'
            }
            
            recovery_system.initialize_ingestion_session(
                documents=["doc1", "doc2"],
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                batch_size=10
            )
            
            # Trigger resource-based degradation
            recovery_system.handle_failure(
                FailureType.RESOURCE_EXHAUSTION,
                "System resources exhausted"
            )
            
            # Should degrade to most restrictive mode
            assert recovery_system.current_degradation_mode == DegradationMode.SAFE
            
            batch = recovery_system.get_next_batch()
            assert len(batch) <= 1  # Should use minimal batch size
    
    def test_concurrent_checkpoint_operations(self, recovery_system):
        """Test concurrent checkpoint operations (thread safety)."""
        import threading
        
        recovery_system.initialize_ingestion_session(
            documents=["doc1", "doc2", "doc3"],
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        checkpoint_ids = []
        errors = []
        
        def create_checkpoint(index):
            try:
                checkpoint_id = recovery_system.create_checkpoint({"thread": index})
                checkpoint_ids.append(checkpoint_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple checkpoints concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_checkpoint, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have created all checkpoints without errors
        assert len(errors) == 0
        assert len(checkpoint_ids) == 5
        assert len(set(checkpoint_ids)) == 5  # All unique


# =====================================================================
# PERFORMANCE AND STRESS TESTS
# =====================================================================

@pytest.mark.performance
class TestErrorHandlingPerformance:
    """Test performance characteristics of error handling."""
    
    def test_error_classification_performance(self):
        """Test performance of error classification."""
        errors_to_test = [
            IngestionAPIError("API error", status_code=500),
            IngestionNetworkError("Network error"),
            IngestionResourceError("Memory error", resource_type="memory"),
            StoragePermissionError("Permission error", required_permission="write"),
            StorageSpaceError("Space error", available_space=1024, required_space=2048)
        ]
        
        start_time = time.time()
        
        for _ in range(1000):  # Test with many error objects
            for error in errors_to_test:
                # Simulate error handling operations
                assert isinstance(error, Exception)
                error_message = str(error)
                error_type = type(error).__name__
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete in under 1 second
    
    def test_backoff_calculation_performance(self):
        """Test performance of backoff calculations."""
        calculator = AdaptiveBackoffCalculator()
        
        start_time = time.time()
        
        # Simulate many backoff calculations
        for attempt in range(1, 100):
            for failure_type in FailureType:
                delay = calculator.calculate_backoff(
                    failure_type,
                    attempt,
                    BackoffStrategy.ADAPTIVE
                )
                assert delay > 0
        
        elapsed = time.time() - start_time
        assert elapsed < 2.0  # Should complete in under 2 seconds
    
    def test_checkpoint_creation_performance(self, temp_dir):
        """Test performance of checkpoint operations."""
        checkpoint_manager = CheckpointManager(temp_dir / "checkpoints")
        
        # Create large document lists for checkpoint
        large_processed = [f"doc_{i}" for i in range(1000)]
        large_failed = {f"failed_{i}": f"Error {i}" for i in range(100)}
        large_pending = [f"pending_{i}" for i in range(500)]
        
        start_time = time.time()
        
        # Create multiple checkpoints with large data
        checkpoint_ids = []
        for i in range(10):
            checkpoint_id = checkpoint_manager.create_checkpoint(
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                processed_documents=large_processed[:100 * i],  # Increasing size
                failed_documents=dict(list(large_failed.items())[:10 * i]),
                pending_documents=large_pending[:50 * i],
                current_batch_size=5,
                degradation_mode=DegradationMode.OPTIMAL,
                system_resources={"memory_percent": 50.0},
                error_counts={FailureType.API_ERROR: i}
            )
            checkpoint_ids.append(checkpoint_id)
        
        creation_time = time.time() - start_time
        
        # Load all checkpoints
        start_time = time.time()
        for checkpoint_id in checkpoint_ids:
            checkpoint = checkpoint_manager.load_checkpoint(checkpoint_id)
            assert checkpoint is not None
        
        loading_time = time.time() - start_time
        
        # Performance assertions
        assert creation_time < 5.0  # Creation should be fast
        assert loading_time < 2.0   # Loading should be fast
    
    @pytest.mark.slow
    def test_memory_usage_under_stress(self, temp_dir):
        """Test memory usage under stress conditions."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create recovery system
        recovery_system = AdvancedRecoverySystem(
            checkpoint_dir=temp_dir / "stress_checkpoints"
        )
        
        # Generate stress conditions
        large_document_list = [f"stress_doc_{i}" for i in range(10000)]
        
        recovery_system.initialize_ingestion_session(
            documents=large_document_list,
            phase=KnowledgeBasePhase.DOCUMENT_INGESTION
        )
        
        # Simulate many failures and recoveries
        for i in range(1000):
            if i % 100 == 0:  # Periodic checkpoint
                recovery_system.create_checkpoint({"stress_test": i})
            
            recovery_system.handle_failure(
                FailureType.API_ERROR,
                f"Stress error {i}",
                f"stress_doc_{i % 100}"
            )
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])