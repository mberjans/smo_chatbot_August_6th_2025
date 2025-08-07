#!/usr/bin/env python3
"""
Test Suite for Unified Progress Tracking Integration

This test suite verifies that the unified progress tracking system is properly
integrated with the ClinicalMetabolomicsRAG initialize_knowledge_base method.

Run with:
    python test_unified_progress_integration.py
    or
    pytest test_unified_progress_integration.py -v
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Test imports
from clinical_metabolomics_rag import ClinicalMetabolomicsRAG
from progress_config import ProgressTrackingConfig
from unified_progress_tracker import KnowledgeBaseProgressTracker, KnowledgeBasePhase
from progress_integration import create_unified_progress_tracker


class TestUnifiedProgressIntegration:
    """Test suite for unified progress tracking integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def progress_config(self, temp_dir):
        """Create progress configuration for testing."""
        return ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            save_unified_progress_to_file=True,
            unified_progress_file_path=temp_dir / "progress.json",
            enable_progress_tracking=True,
            log_progress_interval=1
        )
    
    @pytest.fixture
    def mock_rag_system(self):
        """Create a mocked RAG system for testing."""
        rag = Mock(spec=ClinicalMetabolomicsRAG)
        rag.logger = Mock()
        rag.is_initialized = True
        rag._knowledge_base_initialized = False
        rag.pdf_processor = None
        return rag
    
    def test_progress_config_unified_tracking_fields(self):
        """Test that progress configuration includes unified tracking fields."""
        config = ProgressTrackingConfig()
        
        # Check that new unified progress fields exist
        assert hasattr(config, 'enable_unified_progress_tracking')
        assert hasattr(config, 'enable_phase_based_progress')
        assert hasattr(config, 'save_unified_progress_to_file')
        assert hasattr(config, 'unified_progress_file_path')
        assert hasattr(config, 'enable_progress_callbacks')
        assert hasattr(config, 'phase_progress_update_interval')
        
        # Check default values
        assert config.enable_unified_progress_tracking is True
        assert config.enable_phase_based_progress is True
        assert isinstance(config.phase_progress_update_interval, float)
    
    def test_unified_progress_tracker_creation(self, progress_config):
        """Test creation of unified progress tracker."""
        tracker = create_unified_progress_tracker(
            progress_config=progress_config,
            enable_console_output=False
        )
        
        assert isinstance(tracker, KnowledgeBaseProgressTracker)
        assert tracker.progress_config == progress_config
        assert tracker.logger is not None
    
    def test_phase_weights_configuration(self):
        """Test phase weights configuration."""
        from unified_progress_tracker import PhaseWeights
        
        # Test default weights
        weights = PhaseWeights()
        assert weights.storage_init == 0.10
        assert weights.pdf_processing == 0.60
        assert weights.document_ingestion == 0.25
        assert weights.finalization == 0.05
        
        # Verify weights sum to 1.0
        total = (weights.storage_init + weights.pdf_processing + 
                weights.document_ingestion + weights.finalization)
        assert abs(total - 1.0) < 0.001
    
    def test_knowledge_base_phases_enum(self):
        """Test KnowledgeBasePhase enum values."""
        phases = list(KnowledgeBasePhase)
        expected_phases = [
            KnowledgeBasePhase.STORAGE_INIT,
            KnowledgeBasePhase.PDF_PROCESSING,
            KnowledgeBasePhase.DOCUMENT_INGESTION,
            KnowledgeBasePhase.FINALIZATION
        ]
        
        assert len(phases) == 4
        for phase in expected_phases:
            assert phase in phases
    
    @pytest.mark.asyncio
    async def test_progress_tracker_initialization(self, progress_config, temp_dir):
        """Test progress tracker initialization and basic functionality."""
        tracker = create_unified_progress_tracker(
            progress_config=progress_config,
            enable_console_output=False
        )
        
        # Test initialization
        tracker.start_initialization(total_documents=5)
        state = tracker.get_current_state()
        
        assert state.start_time is not None
        assert state.total_documents == 5
        assert state.overall_progress == 0.0
        assert state.current_phase is None
        
        # Test phase tracking
        tracker.start_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Starting storage initialization"
        )
        
        state = tracker.get_current_state()
        assert state.current_phase == KnowledgeBasePhase.STORAGE_INIT
        
        phase_info = state.phase_info[KnowledgeBasePhase.STORAGE_INIT]
        assert phase_info.is_active
        assert not phase_info.is_completed
        assert not phase_info.is_failed
        
        # Test progress update
        tracker.update_phase_progress(
            KnowledgeBasePhase.STORAGE_INIT,
            0.5,
            "Storage directories created"
        )
        
        state = tracker.get_current_state()
        phase_info = state.phase_info[KnowledgeBasePhase.STORAGE_INIT]
        assert phase_info.current_progress == 0.5
        assert "Storage directories created" in phase_info.status_message
        
        # Test phase completion
        tracker.complete_phase(
            KnowledgeBasePhase.STORAGE_INIT,
            "Storage initialization completed"
        )
        
        state = tracker.get_current_state()
        phase_info = state.phase_info[KnowledgeBasePhase.STORAGE_INIT]
        assert phase_info.is_completed
        assert not phase_info.is_active
        assert phase_info.current_progress == 1.0
    
    def test_progress_callback_integration(self, progress_config):
        """Test progress callback integration."""
        callback_calls = []
        
        def test_callback(overall_progress, current_phase, phase_progress, 
                         status_message, phase_details, all_phases):
            callback_calls.append({
                'overall_progress': overall_progress,
                'current_phase': current_phase,
                'phase_progress': phase_progress,
                'status_message': status_message
            })
        
        tracker = create_unified_progress_tracker(
            progress_config=progress_config,
            progress_callback=test_callback
        )
        
        # Start initialization and phase
        tracker.start_initialization(total_documents=3)
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Starting")
        
        # Check that callback was triggered
        assert len(callback_calls) > 0
        
        last_call = callback_calls[-1]
        assert last_call['current_phase'] == KnowledgeBasePhase.STORAGE_INIT
        assert last_call['status_message'] == "Starting"
    
    @pytest.mark.asyncio
    async def test_progress_persistence(self, progress_config, temp_dir):
        """Test progress persistence to file."""
        # Ensure progress file path is set
        progress_config.save_unified_progress_to_file = True
        progress_config.unified_progress_file_path = temp_dir / "test_progress.json"
        
        tracker = create_unified_progress_tracker(
            progress_config=progress_config,
            enable_console_output=False
        )
        
        # Start tracking and update progress
        tracker.start_initialization(total_documents=2)
        tracker.start_phase(KnowledgeBasePhase.STORAGE_INIT, "Test phase")
        tracker.update_phase_progress(KnowledgeBasePhase.STORAGE_INIT, 0.5, "Mid progress")
        
        # Check that progress file was created
        progress_file = progress_config.unified_progress_file_path
        assert progress_file.exists()
        
        # Verify file contents
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        assert 'timestamp' in progress_data
        assert 'state' in progress_data
        assert 'config' in progress_data
        
        state_data = progress_data['state']
        assert state_data['total_documents'] == 2
        assert state_data['current_phase'] == KnowledgeBasePhase.STORAGE_INIT.value
        assert KnowledgeBasePhase.STORAGE_INIT.value in state_data['phase_info']
    
    def test_phase_progress_calculation(self):
        """Test overall progress calculation with phase weights."""
        from unified_progress_tracker import UnifiedProgressState, PhaseWeights
        
        weights = PhaseWeights()
        state = UnifiedProgressState(phase_weights=weights)
        
        # Set progress for each phase
        state.phase_info[KnowledgeBasePhase.STORAGE_INIT].current_progress = 1.0  # Complete
        state.phase_info[KnowledgeBasePhase.PDF_PROCESSING].current_progress = 0.5  # Half done
        state.phase_info[KnowledgeBasePhase.DOCUMENT_INGESTION].current_progress = 0.0  # Not started
        state.phase_info[KnowledgeBasePhase.FINALIZATION].current_progress = 0.0  # Not started
        
        # Calculate expected progress
        expected_progress = (1.0 * 0.10) + (0.5 * 0.60) + (0.0 * 0.25) + (0.0 * 0.05)
        expected_progress = 0.10 + 0.30  # = 0.40
        
        calculated_progress = state.calculate_overall_progress()
        assert abs(calculated_progress - expected_progress) < 0.001
    
    def test_error_handling_in_progress_tracking(self, progress_config):
        """Test error handling in progress tracking components."""
        tracker = create_unified_progress_tracker(
            progress_config=progress_config,
            enable_console_output=False
        )
        
        # Test phase failure
        tracker.start_initialization(total_documents=1)
        tracker.start_phase(KnowledgeBasePhase.PDF_PROCESSING, "Starting PDF processing")
        tracker.fail_phase(KnowledgeBasePhase.PDF_PROCESSING, "PDF processing failed")
        
        state = tracker.get_current_state()
        phase_info = state.phase_info[KnowledgeBasePhase.PDF_PROCESSING]
        
        assert phase_info.is_failed
        assert not phase_info.is_completed
        assert not phase_info.is_active
        assert "PDF processing failed" in phase_info.error_message
        assert len(state.errors) > 0
    
    def test_document_count_tracking(self, progress_config):
        """Test document count tracking functionality."""
        tracker = create_unified_progress_tracker(
            progress_config=progress_config,
            enable_console_output=False
        )
        
        # Start with initial document count
        tracker.start_initialization(total_documents=10)
        state = tracker.get_current_state()
        assert state.total_documents == 10
        assert state.processed_documents == 0
        assert state.failed_documents == 0
        
        # Update document counts
        tracker.update_document_counts(processed=5, failed=2)
        state = tracker.get_current_state()
        assert state.processed_documents == 5
        assert state.failed_documents == 2
        
        # Update total if needed
        tracker.update_document_counts(total=12)
        state = tracker.get_current_state()
        assert state.total_documents == 12

def run_tests():
    """Run all tests manually if not using pytest."""
    import sys
    
    print("🧪 Running Unified Progress Tracking Integration Tests")
    print("=" * 60)
    
    test_class = TestUnifiedProgressIntegration()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        print(f"Running {method_name}...", end=" ")
        
        try:
            method = getattr(test_class, method_name)
            
            # Handle async tests
            if asyncio.iscoroutinefunction(method):
                # Create fixtures for async test
                with tempfile.TemporaryDirectory() as tmp_dir:
                    temp_dir = Path(tmp_dir)
                    progress_config = ProgressTrackingConfig(
                        enable_unified_progress_tracking=True,
                        save_unified_progress_to_file=True,
                        unified_progress_file_path=temp_dir / "progress.json"
                    )
                    asyncio.run(method(progress_config, temp_dir))
            else:
                # Regular test - create simple fixtures if needed
                if 'progress_config' in method.__code__.co_varnames:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        temp_dir = Path(tmp_dir)
                        progress_config = ProgressTrackingConfig(
                            enable_unified_progress_tracking=True,
                            save_unified_progress_to_file=True,
                            unified_progress_file_path=temp_dir / "progress.json"
                        )
                        method(progress_config, temp_dir)
                elif 'temp_dir' in method.__code__.co_varnames:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        temp_dir = Path(tmp_dir)
                        method(temp_dir)
                else:
                    method()
            
            print("✅ PASSED")
            passed += 1
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("❌ Some tests failed!")
        sys.exit(1)
    else:
        print("✅ All tests passed!")

if __name__ == "__main__":
    try:
        # Try to run with pytest if available
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Fall back to manual test runner
        run_tests()