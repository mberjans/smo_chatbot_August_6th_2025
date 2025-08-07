#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Unified Progress Tracking System.

This test suite provides thorough validation of the unified progress tracking
functionality implemented for the Clinical Metabolomics Oracle knowledge base
construction process.

Test Coverage:
1. Core functionality tests (UnifiedProgressTracker class)
2. Phase-based progress calculation with correct weights
3. Progress state management and transitions
4. Callback system invocation and functionality
5. Integration with existing PDF progress tracking
6. Configuration loading and validation
7. Error handling and recovery scenarios
8. Callback system variations (console, file, logger, custom)
9. Edge cases and thread safety
10. Performance tests for minimal overhead

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import json
import logging
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Import unified progress tracking components
from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker,
    KnowledgeBasePhase, 
    PhaseWeights,
    PhaseProgressInfo,
    UnifiedProgressState,
    UnifiedProgressCallback
)
from lightrag_integration.progress_config import ProgressTrackingConfig, ProcessingMetrics
from lightrag_integration.progress_tracker import PDFProcessingProgressTracker


# =====================================================================
# TEST FIXTURES AND UTILITIES
# =====================================================================

@pytest.fixture
def phase_weights():
    """Standard phase weights for testing."""
    return PhaseWeights(
        storage_init=0.10,
        pdf_processing=0.60,
        document_ingestion=0.25,
        finalization=0.05
    )

@pytest.fixture
def custom_phase_weights():
    """Custom phase weights for testing edge cases."""
    return PhaseWeights(
        storage_init=0.2,
        pdf_processing=0.5,
        document_ingestion=0.25,
        finalization=0.05
    )

@pytest.fixture
def mock_progress_config():
    """Mock progress tracking configuration."""
    config = Mock(spec=ProgressTrackingConfig)
    config.enable_progress_tracking = True
    config.enable_unified_progress_tracking = True
    config.enable_phase_based_progress = True
    config.save_unified_progress_to_file = False
    config.unified_progress_file_path = None
    config.progress_log_level = "INFO"
    config.error_log_level = "ERROR"
    config.get_log_level_value = Mock(side_effect=lambda level: getattr(logging, level, logging.INFO))
    return config

@pytest.fixture
def temp_progress_file(temp_dir):
    """Temporary progress file for testing persistence."""
    return temp_dir / "test_progress.json"

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock(spec=logging.Logger)

@pytest.fixture
def callback_capture():
    """Utility for capturing callback invocations."""
    class CallbackCapture:
        def __init__(self):
            self.calls = []
            self.call_count = 0
        
        def __call__(self, overall_progress, current_phase, phase_progress, 
                     status_message, phase_details, all_phases):
            self.call_count += 1
            self.calls.append({
                'overall_progress': overall_progress,
                'current_phase': current_phase,
                'phase_progress': phase_progress,
                'status_message': status_message,
                'phase_details': phase_details.copy(),
                'all_phases': {k: v.to_dict() for k, v in all_phases.items()},
                'timestamp': time.time(),
                'call_number': self.call_count
            })
        
        def get_last_call(self):
            return self.calls[-1] if self.calls else None
        
        def get_calls_for_phase(self, phase: KnowledgeBasePhase):
            return [call for call in self.calls if call['current_phase'] == phase]
        
        def reset(self):
            self.calls.clear()
            self.call_count = 0
    
    return CallbackCapture()

@pytest.fixture
def mock_pdf_processor():
    """Mock PDF processor for integration tests."""
    processor = Mock()
    processor.get_current_metrics = Mock(return_value=ProcessingMetrics(
        total_files=10,
        completed_files=5,
        failed_files=1,
        skipped_files=0
    ))
    return processor

@pytest.fixture
def error_injection_tracker():
    """Utility for testing error injection scenarios."""
    class ErrorInjectionTracker:
        def __init__(self):
            self.should_fail = False
            self.failure_count = 0
            self.error_message = "Test error"
        
        def set_failure_mode(self, should_fail: bool, error_message: str = "Test error"):
            self.should_fail = should_fail
            self.error_message = error_message
        
        def maybe_fail(self):
            if self.should_fail:
                self.failure_count += 1
                raise Exception(self.error_message)
    
    return ErrorInjectionTracker()


# =====================================================================
# CORE FUNCTIONALITY TESTS
# =====================================================================

class TestUnifiedProgressTrackerCore:
    """Test core functionality of the UnifiedProgressTracker class."""
    
    def test_tracker_initialization_default(self):
        """Test tracker initialization with default parameters."""
        tracker = KnowledgeBaseProgressTracker()
        
        # Check default initialization
        assert tracker.state is not None
        assert isinstance(tracker.state.phase_weights, PhaseWeights)
        assert tracker.state.overall_progress == 0.0
        assert tracker.state.current_phase is None
        assert len(tracker.state.phase_info) == len(KnowledgeBasePhase)
        assert tracker.state.total_documents == 0
        assert tracker.state.processed_documents == 0
        assert tracker.state.failed_documents == 0
        
        # Check all phases are initialized
        for phase in KnowledgeBasePhase:
            assert phase in tracker.state.phase_info
            phase_info = tracker.state.phase_info[phase]
            assert phase_info.phase == phase
            assert phase_info.current_progress == 0.0
            assert not phase_info.is_active
            assert not phase_info.is_completed
            assert not phase_info.is_failed
    
    def test_tracker_initialization_with_config(self, mock_progress_config, phase_weights, mock_logger):
        """Test tracker initialization with custom configuration."""
        callback_called = []
        
        def test_callback(*args):
            callback_called.append(args)
        
        tracker = KnowledgeBaseProgressTracker(
            progress_config=mock_progress_config,
            phase_weights=phase_weights,
            progress_callback=test_callback,
            logger=mock_logger
        )
        
        assert tracker.progress_config == mock_progress_config
        assert tracker.phase_weights == phase_weights
        assert tracker.progress_callback == test_callback
        assert tracker.logger == mock_logger
        assert tracker.state.phase_weights == phase_weights
    
    def test_start_initialization(self, mock_progress_config):
        """Test starting initialization process."""
        tracker = KnowledgeBaseProgressTracker(progress_config=mock_progress_config)
        
        # Test start initialization
        total_docs = 25
        tracker.start_initialization(total_documents=total_docs)
        
        assert tracker.state.start_time is not None
        assert tracker.state.total_documents == total_docs
        assert isinstance(tracker.state.start_time, datetime)
        
        # Check that start time is recent
        time_diff = datetime.now() - tracker.state.start_time
        assert time_diff.total_seconds() < 1.0  # Should be very recent
    
    def test_phase_lifecycle(self):
        """Test complete phase lifecycle: start -> update -> complete."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=5)
        
        phase = KnowledgeBasePhase.PDF_PROCESSING
        
        # Test starting phase
        status_msg = "Starting PDF processing"
        estimated_duration = 30.0
        details = {'total_files': 5}
        
        tracker.start_phase(phase, status_msg, estimated_duration, details)
        
        phase_info = tracker.state.phase_info[phase]
        assert tracker.state.current_phase == phase
        assert phase_info.is_active
        assert not phase_info.is_completed
        assert phase_info.status_message == status_msg
        assert phase_info.estimated_duration == estimated_duration
        assert phase_info.details == details
        
        # Test updating phase progress
        progress_update = 0.4
        update_msg = "Processing files..."
        update_details = {'completed_files': 2}
        
        tracker.update_phase_progress(phase, progress_update, update_msg, update_details)
        
        assert phase_info.current_progress == progress_update
        assert phase_info.status_message == update_msg
        assert 'completed_files' in phase_info.details
        assert phase_info.details['completed_files'] == 2
        
        # Test completing phase
        completion_msg = "PDF processing complete"
        tracker.complete_phase(phase, completion_msg)
        
        assert not phase_info.is_active
        assert phase_info.is_completed
        assert phase_info.current_progress == 1.0
        assert phase_info.status_message == completion_msg
        assert phase_info.end_time is not None
    
    def test_phase_failure_handling(self):
        """Test phase failure handling."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=3)
        
        phase = KnowledgeBasePhase.DOCUMENT_INGESTION
        
        # Start phase
        tracker.start_phase(phase, "Starting ingestion")
        tracker.update_phase_progress(phase, 0.3, "Processing...")
        
        # Fail phase
        error_message = "Database connection failed"
        tracker.fail_phase(phase, error_message)
        
        phase_info = tracker.state.phase_info[phase]
        assert not phase_info.is_active
        assert phase_info.is_failed
        assert not phase_info.is_completed
        assert phase_info.error_message == error_message
        assert phase_info.end_time is not None
        
        # Check that error was added to global errors list
        error_found = any(error_message in error for error in tracker.state.errors)
        assert error_found
    
    def test_get_current_state_deep_copy(self):
        """Test that get_current_state returns deep copy."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=2)
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test")
        
        # Get state copy
        state_copy = tracker.get_current_state()
        
        # Modify original state
        tracker.state.total_documents = 100
        tracker.state.phase_info[KnowledgeBasePhase.STORAGE_INIT].current_progress = 0.8
        
        # Verify copy wasn't affected
        assert state_copy.total_documents == 2
        assert state_copy.phase_info[KnowledgeBasePhase.STORAGE_INIT].current_progress == 0.0
    
    def test_document_counts_tracking(self):
        """Test document counts tracking."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=10)
        
        # Update counts incrementally
        tracker.update_document_counts(processed=3)
        assert tracker.state.processed_documents == 3
        
        tracker.update_document_counts(processed=2, failed=1)
        assert tracker.state.processed_documents == 5
        assert tracker.state.failed_documents == 1
        
        # Update total
        tracker.update_document_counts(total=15)
        assert tracker.state.total_documents == 15
        assert tracker.state.processed_documents == 5
        assert tracker.state.failed_documents == 1


# =====================================================================
# PHASE WEIGHTS AND PROGRESS CALCULATION TESTS
# =====================================================================

class TestPhaseWeightsAndProgress:
    """Test phase weights validation and progress calculation."""
    
    def test_default_phase_weights(self):
        """Test default phase weights sum to 1.0."""
        weights = PhaseWeights()
        total = (weights.storage_init + weights.pdf_processing + 
                weights.document_ingestion + weights.finalization)
        assert abs(total - 1.0) < 0.001
    
    def test_custom_phase_weights_validation(self):
        """Test custom phase weights validation."""
        # Valid weights
        valid_weights = PhaseWeights(
            storage_init=0.15,
            pdf_processing=0.50,
            document_ingestion=0.30,
            finalization=0.05
        )
        total = (valid_weights.storage_init + valid_weights.pdf_processing + 
                valid_weights.document_ingestion + valid_weights.finalization)
        assert abs(total - 1.0) < 0.001
        
        # Invalid weights should raise ValueError
        with pytest.raises(ValueError, match="Phase weights must sum to 1.0"):
            PhaseWeights(
                storage_init=0.2,
                pdf_processing=0.5,
                document_ingestion=0.2,
                finalization=0.2  # Total = 1.1
            )
    
    def test_phase_weight_getter(self):
        """Test get_weight method for all phases."""
        weights = PhaseWeights(
            storage_init=0.1,
            pdf_processing=0.6,
            document_ingestion=0.25,
            finalization=0.05
        )
        
        assert weights.get_weight(KnowledgeBasePhase.STORAGE_INIT) == 0.1
        assert weights.get_weight(KnowledgeBasePhase.PDF_PROCESSING) == 0.6
        assert weights.get_weight(KnowledgeBasePhase.DOCUMENT_INGESTION) == 0.25
        assert weights.get_weight(KnowledgeBasePhase.FINALIZATION) == 0.05
    
    def test_overall_progress_calculation(self, custom_phase_weights):
        """Test overall progress calculation with different phase completions."""
        tracker = KnowledgeBaseProgressTracker(phase_weights=custom_phase_weights)
        tracker.start_initialization(total_documents=5)
        
        # Complete storage init (20% weight)
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Done")
        
        expected_progress = 0.2  # 20% of total
        actual_progress = tracker.state.calculate_overall_progress()
        assert abs(actual_progress - expected_progress) < 0.001
        
        # Half complete PDF processing (50% weight)
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF")
        tracker.update_phase_progress(KnowledgeBasePhase.PDF_PROCESSING, 0.5, "Half done")
        
        expected_progress = 0.2 + (0.5 * 0.5)  # 20% + 25%
        actual_progress = tracker.state.calculate_overall_progress()
        assert abs(actual_progress - expected_progress) < 0.001
        
        # Complete PDF processing
        tracker.complete_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF done")
        
        expected_progress = 0.2 + 0.5  # 70% total
        actual_progress = tracker.state.calculate_overall_progress()
        assert abs(actual_progress - expected_progress) < 0.001
    
    def test_progress_calculation_with_failures(self):
        """Test progress calculation when some phases fail."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=3)
        
        # Complete storage init
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Done")
        
        # Fail PDF processing at 30%
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF")
        tracker.update_phase_progress(KnowledgeBasePhase.PDF_PROCESSING, 0.3, "Working")
        tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, "Failed")
        
        # Progress should include failed phase's partial progress
        expected_progress = 0.1 + (0.6 * 0.3)  # Storage + partial PDF
        actual_progress = tracker.state.calculate_overall_progress()
        assert abs(actual_progress - expected_progress) < 0.001
    
    def test_progress_bounds_validation(self):
        """Test that progress values are properly bounded."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization()
        
        phase = KnowledgeBasePhase.STORAGE_INIT
        tracker.start_phase(phase, "Test")
        
        # Test negative progress (should be clamped to 0)
        tracker.update_phase_progress(phase, -0.5, "Negative test")
        assert tracker.state.phase_info[phase].current_progress == 0.0
        
        # Test progress > 1 (should be clamped to 1)
        tracker.update_phase_progress(phase, 1.5, "Over-progress test")
        assert tracker.state.phase_info[phase].current_progress == 1.0


# =====================================================================
# CALLBACK SYSTEM TESTS  
# =====================================================================

class TestCallbackSystem:
    """Test the progress callback system functionality."""
    
    def test_callback_invocation_on_progress_updates(self, callback_capture):
        """Test that callbacks are invoked on progress updates."""
        tracker = KnowledgeBaseProgressTracker(progress_callback=callback_capture)
        tracker.start_initialization(total_documents=2)
        
        # Should trigger callback on phase start
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Starting storage")
        assert callback_capture.call_count >= 1
        
        # Should trigger callback on progress update
        tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Half done")
        assert callback_capture.call_count >= 2
        
        # Should trigger callback on phase completion
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage complete")
        assert callback_capture.call_count >= 3
    
    def test_callback_parameters(self, callback_capture):
        """Test that callback receives correct parameters."""
        tracker = KnowledgeBaseProgressTracker(progress_callback=callback_capture)
        tracker.start_initialization(total_documents=5)
        
        phase = KnowledgeBasePhase.PDF_PROCESSING
        status_msg = "Processing PDFs"
        details = {'files_processed': 2, 'total_files': 5}
        
        tracker.start_phase(phase, status_msg, details=details)
        tracker.update_phase_progress(phase, 0.4, status_msg, details)
        
        last_call = callback_capture.get_last_call()
        assert last_call is not None
        assert last_call['current_phase'] == phase
        assert last_call['phase_progress'] == 0.4
        assert last_call['status_message'] == status_msg
        assert last_call['phase_details']['files_processed'] == 2
        assert isinstance(last_call['all_phases'], dict)
        assert len(last_call['all_phases']) == len(KnowledgeBasePhase)
    
    def test_callback_failure_handling(self, mock_logger):
        """Test that callback failures don't break progress tracking."""
        def failing_callback(*args):
            raise Exception("Callback failed")
        
        tracker = KnowledgeBaseProgressTracker(
            progress_callback=failing_callback,
            logger=mock_logger
        )
        tracker.start_initialization(total_documents=1)
        
        # This should not raise an exception despite callback failure
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test")
        tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Test")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Test")
        
        # Verify that warnings were logged
        mock_logger.warning.assert_called()
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "Progress callback failed" in str(call)]
        assert len(warning_calls) >= 1
    
    def test_multiple_callback_invocations_sequence(self, callback_capture):
        """Test callback invocation sequence during full initialization."""
        tracker = KnowledgeBaseProgressTracker(progress_callback=callback_capture)
        tracker.start_initialization(total_documents=2)
        
        # Go through all phases
        for phase in KnowledgeBasePhase:
            tracker.start_phase(phase, f"Starting {phase.value}")
            tracker.update_phase_progress(phase, 0.5, f"Halfway {phase.value}")
            tracker.complete_phase(phase, f"Completed {phase.value}")
        
        # Should have at least 3 calls per phase (start, update, complete)
        assert callback_capture.call_count >= len(KnowledgeBasePhase) * 3
        
        # Verify progress increases over time
        progress_values = [call['overall_progress'] for call in callback_capture.calls]
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1]  # Should be non-decreasing
    
    def test_console_callback_integration(self, capsys):
        """Test integration with console callback functionality."""
        # Simple console-style callback
        def console_callback(overall_progress, current_phase, phase_progress, 
                           status_message, phase_details, all_phases):
            print(f"Progress: {overall_progress:.1%} | {current_phase.value}: {status_message}")
        
        tracker = KnowledgeBaseProgressTracker(progress_callback=console_callback)
        tracker.start_initialization(total_documents=1)
        
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Initializing storage")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage ready")
        
        # Capture console output
        captured = capsys.readouterr()
        assert "Progress:" in captured.out
        assert "storage_initialization" in captured.out
        assert "Storage ready" in captured.out


# =====================================================================
# CONFIGURATION TESTS
# =====================================================================

class TestProgressTrackingConfiguration:
    """Test progress tracking configuration validation and behavior."""
    
    def test_config_default_values(self):
        """Test default configuration values."""
        config = ProgressTrackingConfig()
        
        # Test default values
        assert config.enable_progress_tracking is True
        assert config.enable_unified_progress_tracking is True
        assert config.enable_phase_based_progress is True
        assert config.save_unified_progress_to_file is True  # Default behavior
        
        # Test log levels are valid
        assert config.progress_log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.error_log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        # Test numeric parameters
        assert config.log_progress_interval > 0
        assert config.memory_check_interval > 0
        assert config.max_error_details_length > 0
        assert config.phase_progress_update_interval > 0
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        config = ProgressTrackingConfig(
            log_progress_interval=-5,  # Should be corrected
            memory_check_interval=0,   # Should be corrected
            max_error_details_length=-100,  # Should be corrected
            phase_progress_update_interval=0  # Should be corrected
        )
        
        # Negative/zero values should be corrected to defaults
        assert config.log_progress_interval > 0
        assert config.memory_check_interval > 0
        assert config.max_error_details_length > 0
        assert config.phase_progress_update_interval > 0
    
    def test_config_with_file_persistence(self, temp_progress_file):
        """Test configuration with file persistence enabled."""
        config = ProgressTrackingConfig(
            save_unified_progress_to_file=True,
            unified_progress_file_path=temp_progress_file
        )
        
        tracker = KnowledgeBaseProgressTracker(progress_config=config)
        tracker.start_initialization(total_documents=1)
        
        # Make some progress to trigger file write
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test persistence")
        tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Testing")
        
        # File should be created and contain progress data
        assert temp_progress_file.exists()
        
        with open(temp_progress_file, 'r') as f:
            progress_data = json.load(f)
        
        assert 'state' in progress_data
        assert 'timestamp' in progress_data
        assert 'config' in progress_data
        assert progress_data['state']['overall_progress'] > 0
    
    def test_config_serialization(self, temp_progress_file):
        """Test configuration serialization and deserialization."""
        original_config = ProgressTrackingConfig(
            enable_progress_tracking=True,
            log_progress_interval=3,
            save_unified_progress_to_file=True,
            unified_progress_file_path=temp_progress_file
        )
        
        # Convert to dict
        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['enable_progress_tracking'] is True
        assert config_dict['log_progress_interval'] == 3
        
        # Recreate from dict
        recreated_config = ProgressTrackingConfig.from_dict(config_dict)
        assert recreated_config.enable_progress_tracking == original_config.enable_progress_tracking
        assert recreated_config.log_progress_interval == original_config.log_progress_interval
        assert recreated_config.unified_progress_file_path == original_config.unified_progress_file_path


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestProgressTrackingIntegration:
    """Test integration with existing systems."""
    
    def test_pdf_progress_tracker_integration(self):
        """Test integration with existing PDF progress tracker."""
        # Create mock PDF progress tracker
        pdf_tracker = Mock(spec=PDFProcessingProgressTracker)
        pdf_metrics = ProcessingMetrics(
            total_files=20,
            completed_files=12,
            failed_files=2,
            skipped_files=1
        )
        pdf_tracker.get_current_metrics.return_value = pdf_metrics
        
        # Create unified tracker and integrate
        unified_tracker = KnowledgeBaseProgressTracker()
        unified_tracker.integrate_pdf_progress_tracker(pdf_tracker)
        unified_tracker.start_initialization(total_documents=20)
        
        # Start PDF processing phase
        unified_tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Processing PDFs")
        
        # Sync PDF progress
        unified_tracker.sync_pdf_progress()
        
        # Verify PDF progress was synced
        pdf_phase_info = unified_tracker.state.phase_info[KnowledgeBasePhase.PDF_PROCESSING]
        
        # Progress should reflect PDF tracker metrics
        processed_files = pdf_metrics.completed_files + pdf_metrics.failed_files + pdf_metrics.skipped_files
        expected_progress = processed_files / pdf_metrics.total_files
        
        assert abs(pdf_phase_info.current_progress - expected_progress) < 0.001
        assert pdf_phase_info.details['completed_files'] == pdf_metrics.completed_files
        assert pdf_phase_info.details['failed_files'] == pdf_metrics.failed_files
        assert pdf_phase_info.details['total_files'] == pdf_metrics.total_files
    
    @pytest.mark.asyncio
    async def test_initialize_knowledge_base_integration(self, callback_capture):
        """Test integration with initialize_knowledge_base method simulation."""
        # This simulates how the unified progress tracker would be used
        # in the actual initialize_knowledge_base method
        
        tracker = KnowledgeBaseProgressTracker(progress_callback=callback_capture)
        
        # Simulate knowledge base initialization process
        total_documents = 5
        tracker.start_initialization(total_documents=total_documents)
        
        # Phase 1: Storage Initialization
        tracker.start_phase(
            KnowledgeBasePhase.STORAGE_INIT, 
            "Initializing storage directories"
        )
        await asyncio.sleep(0.01)  # Simulate work
        tracker.complete_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Storage initialization complete"
        )
        
        # Phase 2: PDF Processing  
        tracker.start_phase(
            KnowledgeBasePhase.PDF_PROCESSING,
            "Processing PDF documents"
        )
        
        # Simulate processing documents one by one
        for i in range(total_documents):
            await asyncio.sleep(0.005)  # Simulate processing time
            progress = (i + 1) / total_documents
            tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                progress,
                f"Processed {i + 1}/{total_documents} documents"
            )
            tracker.update_document_counts(processed=1)
        
        tracker.complete_phase(
            KnowledgeBasePhase.PDF_PROCESSING,
            "PDF processing complete"
        )
        
        # Phase 3: Document Ingestion
        tracker.start_phase(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            "Ingesting documents into knowledge graph"
        )
        await asyncio.sleep(0.01)
        tracker.update_phase_progress(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            0.5,
            "Batch ingestion in progress"
        )
        await asyncio.sleep(0.01)
        tracker.complete_phase(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            "Document ingestion complete"
        )
        
        # Phase 4: Finalization
        tracker.start_phase(
            KnowledgeBasePhase.FINALIZATION,
            "Finalizing knowledge base"
        )
        await asyncio.sleep(0.005)
        tracker.complete_phase(
            KnowledgeBasePhase.FINALIZATION,
            "Knowledge base ready"
        )
        
        # Verify final state
        final_state = tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        assert final_state.processed_documents == total_documents
        assert all(phase_info.is_completed for phase_info in final_state.phase_info.values())
        
        # Verify callbacks were called appropriately
        assert callback_capture.call_count > 0
        
        # Check progress increased monotonically
        progress_values = [call['overall_progress'] for call in callback_capture.calls]
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1]


# =====================================================================
# ERROR HANDLING AND EDGE CASES TESTS
# =====================================================================

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_no_documents_to_process(self):
        """Test handling when there are no documents to process."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=0)
        
        # Complete all phases with no work
        for phase in KnowledgeBasePhase:
            tracker.start_phase(phase, f"Empty {phase.value}")
            tracker.complete_phase(phase, f"No work for {phase.value}")
        
        final_state = tracker.get_current_state()
        assert abs(final_state.overall_progress - 1.0) < 0.001
        assert final_state.total_documents == 0
        assert final_state.processed_documents == 0
        assert final_state.failed_documents == 0
    
    def test_pdf_processing_complete_failure(self, callback_capture):
        """Test handling when PDF processing completely fails."""
        tracker = KnowledgeBaseProgressTracker(progress_callback=callback_capture)
        tracker.start_initialization(total_documents=10)
        
        # Complete storage init normally
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage init")
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage complete")
        
        # Fail PDF processing immediately
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Starting PDF processing")
        tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, "Complete PDF processing failure")
        
        # Continue with document ingestion (recovery scenario)
        tracker.start_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Attempting recovery")
        tracker.complete_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Recovery successful")
        
        # Complete finalization
        tracker.start_phase(KnowledgeBasePhase.FINALIZATION, "Finalizing")
        tracker.complete_phase(KnowledgeBasePhase.FINALIZATION, "Complete")
        
        final_state = tracker.get_current_state()
        
        # Overall progress should reflect the failure
        storage_weight = 0.10
        pdf_weight = 0.60  # Failed, so 0 progress
        ingestion_weight = 0.25  # Completed
        finalization_weight = 0.05  # Completed
        expected_progress = storage_weight + ingestion_weight + finalization_weight  # 0.40
        
        assert abs(final_state.overall_progress - expected_progress) < 0.001
        assert len(final_state.errors) > 0
        assert final_state.phase_info[KnowledgeBasePhase.PDF_PROCESSING].is_failed
    
    def test_multiple_phase_failures(self):
        """Test handling multiple phase failures."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=5)
        
        # Fail multiple phases
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage init")
        tracker.fail_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage failure")
        
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF processing")
        tracker.update_phase_progress(KnowledgeBasePhase.PDF_PROCESSING, 0.2, "Some progress")
        tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF failure")
        
        # Complete remaining phases
        tracker.start_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Ingestion")
        tracker.complete_phase(KnowledgeBasePhase.DOCUMENT_INGESTION, "Ingestion complete")
        
        tracker.start_phase(KnowledgeBasePhase.FINALIZATION, "Finalization")
        tracker.complete_phase(KnowledgeBasePhase.FINALIZATION, "Finalization complete")
        
        final_state = tracker.get_current_state()
        
        # Check that multiple errors were recorded
        assert len(final_state.errors) >= 2
        
        # Check failed phases
        assert final_state.phase_info[KnowledgeBasePhase.STORAGE_INIT].is_failed
        assert final_state.phase_info[KnowledgeBasePhase.PDF_PROCESSING].is_failed
        
        # Check completed phases
        assert final_state.phase_info[KnowledgeBasePhase.DOCUMENT_INGESTION].is_completed
        assert final_state.phase_info[KnowledgeBasePhase.FINALIZATION].is_completed
    
    def test_progress_file_write_failure(self, temp_progress_file, mock_logger):
        """Test handling when progress file write fails."""
        # Create config that points to invalid path
        invalid_path = temp_progress_file.parent / "nonexistent_dir" / "progress.json"
        
        config = ProgressTrackingConfig(
            save_unified_progress_to_file=True,
            unified_progress_file_path=invalid_path
        )
        
        tracker = KnowledgeBaseProgressTracker(
            progress_config=config,
            logger=mock_logger
        )
        
        # This should not raise an exception even though file write will fail
        tracker.start_initialization(total_documents=1)
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test")
        
        # Should have logged warning about file write failure
        mock_logger.warning.assert_called()
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "Failed to save unified progress to file" in str(call)]
        assert len(warning_calls) >= 1
    
    def test_time_estimation_edge_cases(self):
        """Test time estimation in edge cases."""
        tracker = KnowledgeBaseProgressTracker()
        
        # Before starting, no time estimate available
        state = tracker.get_current_state()
        assert state.get_estimated_time_remaining() is None
        
        tracker.start_initialization(total_documents=1)
        
        # Right after starting with no progress, no estimate
        state = tracker.get_current_state()
        assert state.get_estimated_time_remaining() is None
        
        # With 100% progress, remaining time should be 0
        for phase in KnowledgeBasePhase:
            tracker.start_phase(phase, f"Phase {phase.value}")
            tracker.complete_phase(phase, f"Phase {phase.value} done")
        
        state = tracker.get_current_state()
        remaining = state.get_estimated_time_remaining()
        assert remaining is not None
        assert remaining == 0.0


# =====================================================================
# THREAD SAFETY TESTS
# =====================================================================

class TestThreadSafety:
    """Test thread safety of progress tracking."""
    
    def test_concurrent_progress_updates(self):
        """Test concurrent progress updates from multiple threads."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=100)
        
        # Track results from concurrent operations
        results = []
        errors = []
        
        def update_progress_worker(worker_id, phase):
            try:
                for i in range(10):
                    tracker.update_phase_progress(
                        phase, 
                        (i + 1) / 10, 
                        f"Worker {worker_id} update {i+1}",
                        {'worker_id': worker_id, 'iteration': i}
                    )
                    time.sleep(0.001)  # Small delay to increase chances of contention
                results.append(f"Worker {worker_id} completed")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Start PDF processing phase
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Concurrent processing")
        
        # Create multiple threads updating the same phase
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(
                target=update_progress_worker,
                args=(worker_id, KnowledgeBasePhase.PDF_PROCESSING)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors in concurrent updates: {errors}"
        assert len(results) == 5, f"Not all workers completed: {results}"
        
        # Verify final state is consistent
        final_state = tracker.get_current_state()
        pdf_phase_info = final_state.phase_info[KnowledgeBasePhase.PDF_PROCESSING]
        
        # Progress should be between 0 and 1
        assert 0.0 <= pdf_phase_info.current_progress <= 1.0
        # Details should contain some worker information
        assert 'worker_id' in pdf_phase_info.details
    
    def test_concurrent_phase_transitions(self):
        """Test concurrent phase transitions."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=10)
        
        results = []
        errors = []
        
        def phase_worker(phase, worker_id):
            try:
                tracker.start_phase(phase, f"Worker {worker_id} starting {phase.value}")
                time.sleep(0.01)
                
                for i in range(3):
                    tracker.update_phase_progress(
                        phase, 
                        (i + 1) / 3, 
                        f"Worker {worker_id} progress {i+1}"
                    )
                    time.sleep(0.005)
                
                tracker.complete_phase(phase, f"Worker {worker_id} completed {phase.value}")
                results.append(f"Worker {worker_id} phase {phase.value} completed")
                
            except Exception as e:
                errors.append(f"Worker {worker_id} phase {phase.value} error: {e}")
        
        # Create threads for different phases
        threads = []
        phases = list(KnowledgeBasePhase)
        
        for i, phase in enumerate(phases):
            thread = threading.Thread(
                target=phase_worker,
                args=(phase, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All operations should complete successfully
        assert len(errors) == 0, f"Errors in concurrent phase transitions: {errors}"
        assert len(results) == len(phases), f"Not all phases completed: {results}"
        
        # Verify all phases ended up completed
        final_state = tracker.get_current_state()
        for phase in phases:
            phase_info = final_state.phase_info[phase]
            assert phase_info.is_completed or phase_info.is_failed  # Should be in final state


# =====================================================================
# PERFORMANCE TESTS
# =====================================================================

class TestPerformance:
    """Test performance characteristics of progress tracking."""
    
    def test_minimal_overhead_progress_updates(self):
        """Test that progress updates have minimal overhead."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=1000)
        
        phase = KnowledgeBasePhase.PDF_PROCESSING
        tracker.start_phase(phase, "Performance test")
        
        # Measure time for many progress updates
        num_updates = 1000
        start_time = time.time()
        
        for i in range(num_updates):
            progress = i / num_updates
            tracker.update_phase_progress(
                phase, 
                progress, 
                f"Update {i}",
                {'iteration': i, 'data': 'test_data_' * 10}  # Some details
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be very fast - less than 1 second for 1000 updates
        assert total_time < 1.0, f"Too slow: {total_time:.3f}s for {num_updates} updates"
        
        # Average time per update should be very small
        avg_time_per_update = total_time / num_updates
        assert avg_time_per_update < 0.001, f"Too slow per update: {avg_time_per_update:.6f}s"
    
    def test_callback_overhead_with_complex_callbacks(self):
        """Test overhead when using complex callbacks."""
        complex_callback_calls = []
        
        def complex_callback(overall_progress, current_phase, phase_progress, 
                           status_message, phase_details, all_phases):
            # Simulate some processing in callback
            import json
            
            # Create complex data structure
            callback_data = {
                'timestamp': time.time(),
                'overall_progress': overall_progress,
                'current_phase': current_phase.value,
                'phase_progress': phase_progress,
                'status_message': status_message,
                'phase_details': phase_details.copy(),
                'all_phases_summary': {
                    phase.value: {
                        'progress': info.current_progress,
                        'status': info.status_message,
                        'is_active': info.is_active,
                        'is_completed': info.is_completed
                    }
                    for phase, info in all_phases.items()
                }
            }
            
            # Serialize to JSON (simulate logging to file)
            json_str = json.dumps(callback_data, indent=2)
            complex_callback_calls.append(json_str)
        
        tracker = KnowledgeBaseProgressTracker(progress_callback=complex_callback)
        tracker.start_initialization(total_documents=100)
        
        # Measure time with complex callback
        num_updates = 100
        start_time = time.time()
        
        phase = KnowledgeBasePhase.PDF_PROCESSING
        tracker.start_phase(phase, "Performance test with complex callback")
        
        for i in range(num_updates):
            progress = i / num_updates
            tracker.update_phase_progress(
                phase,
                progress,
                f"Processing item {i}",
                {
                    'item_id': i,
                    'batch_data': [f"item_{j}" for j in range(5)],  # Some complex data
                    'metadata': {'type': 'test', 'size': i * 10}
                }
            )
        
        tracker.complete_phase(phase, "Complex callback test complete")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Even with complex callbacks, should be reasonably fast
        assert total_time < 2.0, f"Too slow with complex callback: {total_time:.3f}s"
        
        # Verify callbacks were executed
        assert len(complex_callback_calls) > 0
        
        # Verify callback data is properly formatted JSON
        for json_str in complex_callback_calls[:3]:  # Check first few
            parsed = json.loads(json_str)
            assert 'overall_progress' in parsed
            assert 'all_phases_summary' in parsed
    
    def test_memory_usage_with_many_updates(self):
        """Test that memory usage doesn't grow excessively with many updates."""
        import gc
        import sys
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=500)
        
        phase = KnowledgeBasePhase.PDF_PROCESSING
        tracker.start_phase(phase, "Memory test")
        
        # Perform many updates
        num_updates = 500
        for i in range(num_updates):
            tracker.update_phase_progress(
                phase,
                i / num_updates,
                f"Memory test update {i}",
                {
                    'update_id': i,
                    'timestamp': time.time(),
                    'data': f"test_data_{i}" * 5  # Some variable data
                }
            )
            
            # Every 100 updates, check memory hasn't grown excessively
            if i % 100 == 0 and i > 0:
                gc.collect()
                current_objects = len(gc.get_objects())
                
                # Object count shouldn't grow excessively
                # Allow some growth but not proportional to update count
                max_allowed_objects = initial_objects + (i // 10)  # Very generous limit
                
                if current_objects > max_allowed_objects:
                    pytest.fail(f"Memory leak detected: {current_objects} objects vs initial {initial_objects} at update {i}")
        
        tracker.complete_phase(phase, "Memory test complete")
        
        # Final memory check
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Final object count should be reasonable
        max_final_objects = initial_objects + 100  # Allow some overhead
        assert final_objects < max_final_objects, f"Memory leak: {final_objects} vs initial {initial_objects}"


# =====================================================================
# INTEGRATION WITH KNOWLEDGE BASE INITIALIZATION TESTS
# =====================================================================

class TestKnowledgeBaseIntegration:
    """Test integration with knowledge base initialization process."""
    
    @pytest.mark.asyncio
    async def test_full_initialization_simulation(self, callback_capture, temp_progress_file):
        """Test complete knowledge base initialization simulation."""
        config = ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            save_unified_progress_to_file=True,
            unified_progress_file_path=temp_progress_file,
            progress_log_level="INFO"
        )
        
        tracker = KnowledgeBaseProgressTracker(
            progress_config=config,
            progress_callback=callback_capture
        )
        
        # Simulate realistic initialization process
        total_documents = 15
        batch_size = 5
        
        # Start initialization
        tracker.start_initialization(total_documents=total_documents)
        
        # === PHASE 1: STORAGE INITIALIZATION ===
        tracker.start_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Initializing LightRAG storage directories",
            estimated_duration=5.0,
            details={'directories_to_create': 4}
        )
        
        # Simulate creating directories
        directories = ['graph_storage', 'vector_storage', 'cache', 'logs']
        for i, directory in enumerate(directories):
            await asyncio.sleep(0.01)  # Simulate work
            tracker.update_phase_progress(
                KnowledgeBasePhase.STORAGE_INIT,
                (i + 1) / len(directories),
                f"Created directory: {directory}",
                {'current_directory': directory}
            )
        
        tracker.complete_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Storage initialization completed successfully"
        )
        
        # === PHASE 2: PDF PROCESSING ===
        tracker.start_phase(
            KnowledgeBasePhase.PDF_PROCESSING,
            f"Processing {total_documents} PDF documents",
            estimated_duration=30.0,
            details={'total_files': total_documents, 'batch_size': batch_size}
        )
        
        # Simulate batch processing
        processed_count = 0
        failed_count = 0
        
        for batch_start in range(0, total_documents, batch_size):
            batch_end = min(batch_start + batch_size, total_documents)
            batch_files = batch_end - batch_start
            
            await asyncio.sleep(0.02)  # Simulate batch processing time
            
            # Simulate some failures (realistic scenario)
            batch_failed = 1 if batch_start == 10 else 0  # Fail one file in third batch
            batch_successful = batch_files - batch_failed
            
            processed_count += batch_successful
            failed_count += batch_failed
            
            tracker.update_phase_progress(
                KnowledgeBasePhase.PDF_PROCESSING,
                batch_end / total_documents,
                f"Processed batch {(batch_start // batch_size) + 1}: {batch_successful} successful, {batch_failed} failed",
                {
                    'batch_number': (batch_start // batch_size) + 1,
                    'files_in_batch': batch_files,
                    'successful_in_batch': batch_successful,
                    'failed_in_batch': batch_failed,
                    'total_processed': processed_count,
                    'total_failed': failed_count
                }
            )
            
            # Update document counts
            tracker.update_document_counts(
                processed=batch_successful,
                failed=batch_failed
            )
        
        tracker.complete_phase(
            KnowledgeBasePhase.PDF_PROCESSING,
            f"PDF processing completed: {processed_count} successful, {failed_count} failed"
        )
        
        # === PHASE 3: DOCUMENT INGESTION ===
        tracker.start_phase(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            "Ingesting processed documents into knowledge graph",
            estimated_duration=20.0,
            details={'documents_to_ingest': processed_count}
        )
        
        # Simulate knowledge graph ingestion in chunks
        ingestion_batch_size = 3
        ingested_count = 0
        
        for chunk_start in range(0, processed_count, ingestion_batch_size):
            chunk_end = min(chunk_start + ingestion_batch_size, processed_count)
            chunk_size = chunk_end - chunk_start
            
            await asyncio.sleep(0.015)  # Simulate ingestion time
            
            ingested_count += chunk_size
            
            tracker.update_phase_progress(
                KnowledgeBasePhase.DOCUMENT_INGESTION,
                ingested_count / processed_count,
                f"Ingested chunk {(chunk_start // ingestion_batch_size) + 1}: {chunk_size} documents",
                {
                    'chunk_number': (chunk_start // ingestion_batch_size) + 1,
                    'documents_in_chunk': chunk_size,
                    'total_ingested': ingested_count,
                    'remaining': processed_count - ingested_count
                }
            )
        
        tracker.complete_phase(
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            f"Document ingestion completed: {ingested_count} documents ingested"
        )
        
        # === PHASE 4: FINALIZATION ===
        tracker.start_phase(
            KnowledgeBasePhase.FINALIZATION,
            "Finalizing knowledge base and optimizing indices",
            estimated_duration=3.0,
            details={'optimization_tasks': 3}
        )
        
        # Simulate finalization tasks
        finalization_tasks = ['index_optimization', 'cache_warming', 'validation']
        for i, task in enumerate(finalization_tasks):
            await asyncio.sleep(0.01)
            tracker.update_phase_progress(
                KnowledgeBasePhase.FINALIZATION,
                (i + 1) / len(finalization_tasks),
                f"Completed {task}",
                {'current_task': task}
            )
        
        tracker.complete_phase(
            KnowledgeBasePhase.FINALIZATION,
            "Knowledge base initialization completed successfully"
        )
        
        # === VERIFICATION ===
        final_state = tracker.get_current_state()
        
        # Verify overall completion
        assert abs(final_state.overall_progress - 1.0) < 0.001
        
        # Verify document counts
        assert final_state.total_documents == total_documents
        assert final_state.processed_documents == processed_count
        assert final_state.failed_documents == failed_count
        
        # Verify all phases completed
        for phase in KnowledgeBasePhase:
            phase_info = final_state.phase_info[phase]
            assert phase_info.is_completed
            assert not phase_info.is_active
            assert not phase_info.is_failed
            assert phase_info.current_progress == 1.0
        
        # Verify timing information
        assert final_state.start_time is not None
        total_elapsed = (datetime.now() - final_state.start_time).total_seconds()
        assert total_elapsed > 0
        
        # Verify callbacks were called extensively
        assert callback_capture.call_count > 20  # Should have many calls
        
        # Verify progress was monotonically increasing
        progress_values = [call['overall_progress'] for call in callback_capture.calls]
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1]
        
        # Verify final callback has 100% progress
        final_call = callback_capture.get_last_call()
        assert final_call['overall_progress'] >= 0.99
        
        # Verify progress file was saved
        assert temp_progress_file.exists()
        
        with open(temp_progress_file, 'r') as f:
            progress_data = json.load(f)
        
        assert progress_data['state']['overall_progress'] >= 0.99
        assert progress_data['state']['processed_documents'] == processed_count
        assert progress_data['state']['failed_documents'] == failed_count
        
        # Verify phase completion in progress file
        for phase_value, phase_data in progress_data['state']['phase_info'].items():
            assert phase_data['is_completed'] is True
            assert phase_data['current_progress'] == 1.0
    
    def test_progress_summary_generation(self):
        """Test human-readable progress summary generation."""
        tracker = KnowledgeBaseProgressTracker()
        tracker.start_initialization(total_documents=10)
        
        # Test initial summary
        summary = tracker.get_progress_summary()
        assert "Overall Progress: 0.0%" in summary
        
        # Make some progress
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Initializing storage")
        tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.7, "Almost done with storage")
        
        summary = tracker.get_progress_summary()
        assert "Overall Progress:" in summary
        assert "storage_initialization" in summary
        assert "Almost done with storage" in summary
        assert "Documents: 0/10" in summary
        assert "Elapsed:" in summary
        
        # Complete some documents
        tracker.update_document_counts(processed=3, failed=1)
        tracker.complete_phase(KnowledgeBasePhase.STORAGE_INIT, "Storage complete")
        
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Processing PDFs")
        
        summary = tracker.get_progress_summary()
        assert "Documents: 3/10" in summary
        assert "1 failed" in summary
        assert "pdf_processing" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])