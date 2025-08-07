#!/usr/bin/env python3
"""
Comprehensive Tests for Advanced Recovery and Graceful Degradation System.

This module provides thorough testing of the advanced recovery mechanisms,
including unit tests, integration tests, and scenario-based testing.

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from lightrag_integration.advanced_recovery_system import (
    AdvancedRecoverySystem, DegradationMode, FailureType, BackoffStrategy,
    ResourceThresholds, DegradationConfig, CheckpointData,
    SystemResourceMonitor, AdaptiveBackoffCalculator, CheckpointManager
)
from lightrag_integration.recovery_integration import RecoveryIntegratedProcessor
from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker, KnowledgeBasePhase
)
from lightrag_integration.progress_config import ProgressTrackingConfig


class TestSystemResourceMonitor:
    """Tests for system resource monitoring."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        thresholds = ResourceThresholds(
            memory_warning_percent=75.0,
            memory_critical_percent=90.0
        )
        monitor = SystemResourceMonitor(thresholds)
        
        assert monitor.thresholds.memory_warning_percent == 75.0
        assert monitor.thresholds.memory_critical_percent == 90.0
    
    def test_get_current_resources(self):
        """Test getting current system resources."""
        monitor = SystemResourceMonitor()
        resources = monitor.get_current_resources()
        
        # Check that we get the expected resource keys
        expected_keys = {
            'memory_percent', 'memory_available_gb', 'disk_percent', 
            'disk_free_gb', 'cpu_percent', 'load_average'
        }
        assert set(resources.keys()) == expected_keys
        
        # Check that values are reasonable
        assert 0.0 <= resources['memory_percent'] <= 100.0
        assert resources['memory_available_gb'] >= 0.0
        assert 0.0 <= resources['disk_percent'] <= 100.0
        assert resources['disk_free_gb'] >= 0.0
        assert resources['cpu_percent'] >= 0.0
    
    def test_check_resource_pressure(self):
        """Test resource pressure detection."""
        # Create monitor with low thresholds for testing
        thresholds = ResourceThresholds(
            memory_warning_percent=10.0,
            memory_critical_percent=20.0,
            disk_warning_percent=10.0,
            disk_critical_percent=20.0
        )
        monitor = SystemResourceMonitor(thresholds)
        
        # Mock the resource data to simulate pressure
        with patch.object(monitor, 'get_current_resources') as mock_resources:
            mock_resources.return_value = {
                'memory_percent': 25.0,  # Above critical threshold
                'disk_percent': 15.0,    # Above warning threshold
                'cpu_percent': 5.0       # Normal
            }
            
            recommendations = monitor.check_resource_pressure()
            
            assert 'memory' in recommendations
            assert 'critical' in recommendations['memory']
            assert 'disk' in recommendations
            assert 'warning' in recommendations['disk']
            assert 'cpu' not in recommendations


class TestAdaptiveBackoffCalculator:
    """Tests for adaptive backoff calculation."""
    
    def test_backoff_calculator_initialization(self):
        """Test backoff calculator initialization."""
        calc = AdaptiveBackoffCalculator()
        assert calc._failure_history == {}
        assert calc._success_history == []
        assert calc._last_api_response_time == 1.0
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        calc = AdaptiveBackoffCalculator()
        
        # Test exponential backoff
        delay1 = calc.calculate_backoff(
            FailureType.API_ERROR, 1, strategy=BackoffStrategy.EXPONENTIAL, 
            base_delay=1.0, jitter=False
        )
        delay2 = calc.calculate_backoff(
            FailureType.API_ERROR, 2, strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=1.0, jitter=False
        )
        delay3 = calc.calculate_backoff(
            FailureType.API_ERROR, 3, strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=1.0, jitter=False
        )
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
    
    def test_fibonacci_backoff(self):
        """Test Fibonacci backoff calculation."""
        calc = AdaptiveBackoffCalculator()
        
        # Test first few Fibonacci values
        assert calc._fibonacci(1) == 1
        assert calc._fibonacci(2) == 1
        assert calc._fibonacci(3) == 2
        assert calc._fibonacci(4) == 3
        assert calc._fibonacci(5) == 5
        assert calc._fibonacci(6) == 8
    
    def test_adaptive_backoff_with_failure_history(self):
        """Test adaptive backoff considering failure history."""
        calc = AdaptiveBackoffCalculator()
        
        # Record some failures
        for _ in range(12):  # High failure rate
            calc.calculate_backoff(FailureType.API_RATE_LIMIT, 1, jitter=False)
        
        # This should result in higher delay due to high failure rate
        high_failure_delay = calc.calculate_backoff(
            FailureType.API_RATE_LIMIT, 1, jitter=False
        )
        
        # Create new calculator for comparison
        calc_new = AdaptiveBackoffCalculator()
        normal_delay = calc_new.calculate_backoff(
            FailureType.API_RATE_LIMIT, 1, jitter=False
        )
        
        assert high_failure_delay > normal_delay
    
    def test_record_success(self):
        """Test recording successful operations."""
        calc = AdaptiveBackoffCalculator()
        
        initial_count = len(calc._success_history)
        calc.record_success()
        
        assert len(calc._success_history) == initial_count + 1
        assert isinstance(calc._success_history[-1], datetime)


class TestCheckpointManager:
    """Tests for checkpoint management."""
    
    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            assert manager.checkpoint_dir == checkpoint_dir
            assert checkpoint_dir.exists()
    
    def test_create_and_load_checkpoint(self):
        """Test creating and loading checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Create checkpoint
            checkpoint_id = manager.create_checkpoint(
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                processed_documents=["doc1", "doc2"],
                failed_documents={"doc3": "error message"},
                pending_documents=["doc4", "doc5"],
                current_batch_size=5,
                degradation_mode=DegradationMode.MINIMAL,
                system_resources={"memory_percent": 75.0},
                error_counts={FailureType.API_ERROR: 2},
                metadata={"test": "value"}
            )
            
            assert checkpoint_id is not None
            assert len(checkpoint_id) > 0
            
            # Load checkpoint
            loaded_checkpoint = manager.load_checkpoint(checkpoint_id)
            
            assert loaded_checkpoint is not None
            assert loaded_checkpoint.phase == KnowledgeBasePhase.DOCUMENT_INGESTION
            assert loaded_checkpoint.processed_documents == ["doc1", "doc2"]
            assert loaded_checkpoint.failed_documents == {"doc3": "error message"}
            assert loaded_checkpoint.pending_documents == ["doc4", "doc5"]
            assert loaded_checkpoint.current_batch_size == 5
            assert loaded_checkpoint.degradation_mode == DegradationMode.MINIMAL
            assert loaded_checkpoint.metadata == {"test": "value"}
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Create multiple checkpoints
            checkpoint1 = manager.create_checkpoint(
                phase=KnowledgeBasePhase.PDF_PROCESSING,
                processed_documents=["doc1"],
                failed_documents={},
                pending_documents=["doc2"],
                current_batch_size=3,
                degradation_mode=DegradationMode.OPTIMAL,
                system_resources={},
                error_counts={}
            )
            
            checkpoint2 = manager.create_checkpoint(
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                processed_documents=["doc1", "doc2"],
                failed_documents={},
                pending_documents=["doc3"],
                current_batch_size=2,
                degradation_mode=DegradationMode.SAFE,
                system_resources={},
                error_counts={}
            )
            
            # List all checkpoints
            all_checkpoints = manager.list_checkpoints()
            assert len(all_checkpoints) == 2
            assert checkpoint1 in all_checkpoints
            assert checkpoint2 in all_checkpoints
            
            # List checkpoints by phase
            pdf_checkpoints = manager.list_checkpoints(KnowledgeBasePhase.PDF_PROCESSING)
            assert len(pdf_checkpoints) == 1
            assert checkpoint1 in pdf_checkpoints
            
            ingestion_checkpoints = manager.list_checkpoints(KnowledgeBasePhase.DOCUMENT_INGESTION)
            assert len(ingestion_checkpoints) == 1
            assert checkpoint2 in ingestion_checkpoints
    
    def test_delete_checkpoint(self):
        """Test deleting checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Create checkpoint
            checkpoint_id = manager.create_checkpoint(
                phase=KnowledgeBasePhase.STORAGE_INIT,
                processed_documents=[],
                failed_documents={},
                pending_documents=["doc1"],
                current_batch_size=1,
                degradation_mode=DegradationMode.OPTIMAL,
                system_resources={},
                error_counts={}
            )
            
            # Verify it exists
            assert manager.load_checkpoint(checkpoint_id) is not None
            
            # Delete it
            success = manager.delete_checkpoint(checkpoint_id)
            assert success
            
            # Verify it's gone
            assert manager.load_checkpoint(checkpoint_id) is None


class TestAdvancedRecoverySystem:
    """Tests for the advanced recovery system."""
    
    def test_recovery_system_initialization(self):
        """Test recovery system initialization."""
        recovery_system = AdvancedRecoverySystem()
        
        assert recovery_system.current_degradation_mode == DegradationMode.OPTIMAL
        assert isinstance(recovery_system.degradation_config, DegradationConfig)
        assert recovery_system._current_batch_size == 10
        assert recovery_system._original_batch_size == 10
    
    def test_initialize_ingestion_session(self):
        """Test initializing an ingestion session."""
        recovery_system = AdvancedRecoverySystem()
        documents = ["doc1", "doc2", "doc3"]
        
        recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=KnowledgeBasePhase.PDF_PROCESSING,
            batch_size=5
        )
        
        assert recovery_system._pending_documents == documents
        assert len(recovery_system._processed_documents) == 0
        assert len(recovery_system._failed_documents) == 0
        assert recovery_system._current_phase == KnowledgeBasePhase.PDF_PROCESSING
        assert recovery_system._current_batch_size == 5
    
    def test_handle_failure_api_rate_limit(self):
        """Test handling API rate limit failures."""
        recovery_system = AdvancedRecoverySystem()
        recovery_system.initialize_ingestion_session(["doc1"], KnowledgeBasePhase.DOCUMENT_INGESTION)
        
        strategy = recovery_system.handle_failure(
            FailureType.API_RATE_LIMIT,
            "Rate limit exceeded",
            document_id="doc1"
        )
        
        assert strategy['action'] == 'backoff_and_retry'
        assert 'backoff_seconds' in strategy
        assert strategy['batch_size_adjustment'] == 0.5
        assert recovery_system._error_counts[FailureType.API_RATE_LIMIT] == 1
        assert "doc1" in recovery_system._failed_documents
    
    def test_handle_failure_memory_pressure(self):
        """Test handling memory pressure failures."""
        recovery_system = AdvancedRecoverySystem()
        recovery_system.initialize_ingestion_session(["doc1"], KnowledgeBasePhase.DOCUMENT_INGESTION)
        
        strategy = recovery_system.handle_failure(
            FailureType.MEMORY_PRESSURE,
            "High memory usage detected",
            document_id="doc1"
        )
        
        assert strategy['action'] == 'reduce_resources'
        assert strategy['batch_size_adjustment'] == 0.3
        assert strategy['degradation_needed']
        assert strategy['checkpoint_recommended']
    
    def test_progressive_degradation(self):
        """Test progressive degradation through multiple failures."""
        recovery_system = AdvancedRecoverySystem()
        recovery_system.initialize_ingestion_session(["doc1"], KnowledgeBasePhase.DOCUMENT_INGESTION)
        
        # Start in optimal mode
        assert recovery_system.current_degradation_mode == DegradationMode.OPTIMAL
        
        # Multiple API errors should trigger degradation
        for i in range(6):
            recovery_system.handle_failure(
                FailureType.API_ERROR,
                f"API error {i+1}",
                document_id=f"doc{i+1}"
            )
        
        # Should have degraded by now
        assert recovery_system.current_degradation_mode != DegradationMode.OPTIMAL
    
    def test_get_next_batch(self):
        """Test getting next batch with dynamic sizing."""
        recovery_system = AdvancedRecoverySystem()
        documents = [f"doc{i}" for i in range(20)]
        recovery_system.initialize_ingestion_session(documents, KnowledgeBasePhase.DOCUMENT_INGESTION, 5)
        
        # Get first batch
        batch1 = recovery_system.get_next_batch()
        assert len(batch1) <= 5  # May be reduced due to resource monitoring
        assert all(doc in documents for doc in batch1)
        
        # Simulate memory pressure to reduce batch size
        recovery_system.handle_failure(FailureType.MEMORY_PRESSURE, "Memory pressure")
        
        # Next batch should be smaller
        batch2 = recovery_system.get_next_batch()
        assert len(batch2) < 5
    
    def test_mark_document_processed(self):
        """Test marking documents as processed."""
        recovery_system = AdvancedRecoverySystem()
        recovery_system.initialize_ingestion_session(["doc1", "doc2"], KnowledgeBasePhase.DOCUMENT_INGESTION)
        
        recovery_system.mark_document_processed("doc1")
        
        assert "doc1" in recovery_system._processed_documents
        assert "doc1" not in recovery_system._pending_documents
        assert len(recovery_system._pending_documents) == 1
    
    def test_checkpoint_and_resume(self):
        """Test creating checkpoints and resuming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recovery_system = AdvancedRecoverySystem(checkpoint_dir=Path(temp_dir))
            
            # Initialize session and process some documents
            documents = ["doc1", "doc2", "doc3", "doc4"]
            recovery_system.initialize_ingestion_session(documents, KnowledgeBasePhase.DOCUMENT_INGESTION)
            
            recovery_system.mark_document_processed("doc1")
            recovery_system.handle_failure(FailureType.PROCESSING_ERROR, "Error", "doc2")
            
            # Create checkpoint
            checkpoint_id = recovery_system.create_checkpoint({"test": "data"})
            assert checkpoint_id is not None
            
            # Create new recovery system and resume
            new_recovery_system = AdvancedRecoverySystem(checkpoint_dir=Path(temp_dir))
            success = new_recovery_system.resume_from_checkpoint(checkpoint_id)
            
            assert success
            assert "doc1" in new_recovery_system._processed_documents
            assert "doc2" in new_recovery_system._failed_documents
            assert len(new_recovery_system._pending_documents) == 2
    
    def test_get_recovery_status(self):
        """Test getting recovery status."""
        recovery_system = AdvancedRecoverySystem()
        recovery_system.initialize_ingestion_session(["doc1", "doc2"], KnowledgeBasePhase.DOCUMENT_INGESTION)
        
        recovery_system.mark_document_processed("doc1")
        recovery_system.handle_failure(FailureType.API_ERROR, "Error", "doc2")
        
        status = recovery_system.get_recovery_status()
        
        assert status['degradation_mode'] == DegradationMode.OPTIMAL.value
        assert status['document_progress']['processed'] == 1
        assert status['document_progress']['failed'] == 1
        # pending count may vary based on document prioritization
        assert FailureType.API_ERROR.value in status['error_counts']


class TestRecoveryIntegratedProcessor:
    """Tests for the integrated recovery processor."""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Create a mock RAG system."""
        mock = Mock()
        mock.process_document = AsyncMock(return_value=True)
        return mock
    
    @pytest.fixture
    def processor(self, mock_rag_system):
        """Create a recovery integrated processor for testing."""
        return RecoveryIntegratedProcessor(
            rag_system=mock_rag_system,
            enable_checkpointing=True,
            checkpoint_interval=2
        )
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.rag_system is not None
        assert processor.recovery_system is not None
        assert processor.progress_tracker is not None
        assert processor.enable_checkpointing
        assert processor.checkpoint_interval == 2
    
    @pytest.mark.asyncio
    async def test_process_documents_success(self, processor):
        """Test successful document processing."""
        documents = ["doc1", "doc2", "doc3"]
        
        results = await processor.process_documents_with_recovery(
            documents=documents,
            initial_batch_size=2
        )
        
        assert len(results['processed_documents']) == 3
        assert len(results['failed_documents']) == 0
        assert results['final_degradation_mode'] == DegradationMode.OPTIMAL
    
    @pytest.mark.asyncio
    async def test_process_documents_with_failures(self, processor):
        """Test document processing with simulated failures."""
        # Mock the processing to simulate failures
        with patch.object(processor, '_process_single_document_with_recovery') as mock_process:
            # First document succeeds, second fails, third succeeds
            mock_process.side_effect = [
                {'success': True, 'error': None, 'recovery_events': [], 'attempts': 1},
                {'success': False, 'error': 'Processing failed', 'recovery_events': [{'action': 'retry'}], 'attempts': 3},
                {'success': True, 'error': None, 'recovery_events': [], 'attempts': 1}
            ]
            
            documents = ["doc1", "doc2", "doc3"]
            results = await processor.process_documents_with_recovery(
                documents=documents,
                initial_batch_size=3
            )
            
            assert len(results['processed_documents']) == 2
            assert len(results['failed_documents']) == 1
            assert "doc2" in results['failed_documents']
            assert len(results['recovery_events']) > 0
    
    def test_classify_error(self, processor):
        """Test error classification."""
        test_cases = [
            ("Rate limit exceeded", FailureType.API_RATE_LIMIT),
            ("Request timed out", FailureType.API_TIMEOUT),
            ("Out of memory", FailureType.MEMORY_PRESSURE),
            ("Network connection failed", FailureType.NETWORK_ERROR),
            ("OpenAI API error", FailureType.API_ERROR),
            ("Unknown error", FailureType.PROCESSING_ERROR)
        ]
        
        for error_msg, expected_type in test_cases:
            result = processor._classify_error(error_msg)
            assert result == expected_type
    
    def test_is_essential_document(self, processor):
        """Test essential document detection."""
        assert processor._is_essential_document("essential_doc_001")
        assert processor._is_essential_document("critical_analysis")
        assert processor._is_essential_document("important_data")
        assert not processor._is_essential_document("regular_document")
        assert not processor._is_essential_document("doc_123")
    
    @pytest.mark.asyncio
    async def test_degraded_processing_modes(self, processor):
        """Test different degraded processing modes."""
        # Test optimal mode
        result = await processor._process_document_optimal("test_doc")
        assert result is True
        
        # Test minimal mode
        result = await processor._process_document_minimal("test_doc")
        assert result is True
        
        # Test safe mode (should always succeed)
        result = await processor._process_document_safe("test_doc")
        assert result is True
        
        # Test offline mode
        result = await processor._queue_document_offline("test_doc")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_resource_pressure_handling(self, processor):
        """Test handling of resource pressure."""
        pressure = {
            'memory': 'critical_reduce_batch_size',
            'disk': 'warning_cleanup_recommended'
        }
        
        recovery_event = await processor._handle_resource_pressure(pressure, "test_doc")
        
        assert recovery_event is not None
        assert recovery_event['action'] == 'reduce_resources'


class TestIntegrationScenarios:
    """Integration tests for complex recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_recovery_workflow(self):
        """Test a complete recovery workflow from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up integrated system
            mock_rag = Mock()
            mock_rag.process_document = AsyncMock()
            
            progress_config = ProgressTrackingConfig(
                enable_progress_tracking=True,
                save_unified_progress_to_file=True,
                unified_progress_file_path=Path(temp_dir) / "progress.json"
            )
            
            progress_tracker = KnowledgeBaseProgressTracker(progress_config)
            
            recovery_system = AdvancedRecoverySystem(
                progress_tracker=progress_tracker,
                checkpoint_dir=Path(temp_dir) / "checkpoints"
            )
            
            processor = RecoveryIntegratedProcessor(
                rag_system=mock_rag,
                recovery_system=recovery_system,
                progress_tracker=progress_tracker
            )
            
            # Define test scenario with mixed success/failure
            documents = [f"doc_{i:03d}" for i in range(10)]
            
            # Mock processing to simulate various scenarios
            with patch.object(processor, '_process_single_document_with_recovery') as mock_process:
                # Simulate mixed results with recovery events
                results = []
                for i, doc in enumerate(documents):
                    if i % 4 == 0:  # Every 4th document fails
                        results.append({
                            'success': False,
                            'error': 'Simulated failure',
                            'recovery_events': [{'action': 'retry', 'degradation_needed': i > 4}],
                            'attempts': 3
                        })
                    else:
                        results.append({
                            'success': True,
                            'error': None,
                            'recovery_events': [],
                            'attempts': 1
                        })
                
                mock_process.side_effect = results
                
                # Process documents
                final_results = await processor.process_documents_with_recovery(
                    documents=documents,
                    initial_batch_size=3
                )
                
                # Verify results
                assert len(final_results['processed_documents']) == 7  # 70% success rate
                assert len(final_results['failed_documents']) == 3     # 30% failure rate
                assert len(final_results['recovery_events']) > 0
                
                # Verify progress tracking worked
                assert progress_config.unified_progress_file_path.exists()
                
                # Verify checkpoints were created
                assert len(final_results['checkpoints']) > 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery_scenario(self):
        """Test checkpoint creation and recovery in failure scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            
            # First processing session - simulate crash after partial processing
            recovery_system1 = AdvancedRecoverySystem(checkpoint_dir=checkpoint_dir)
            documents = [f"critical_doc_{i:03d}" for i in range(20)]
            
            recovery_system1.initialize_ingestion_session(
                documents=documents,
                phase=KnowledgeBasePhase.DOCUMENT_INGESTION,
                batch_size=5
            )
            
            # Process some documents successfully
            for i in range(8):
                recovery_system1.mark_document_processed(documents[i])
            
            # Simulate some failures
            for i in range(8, 12):
                recovery_system1.handle_failure(
                    FailureType.PROCESSING_ERROR,
                    f"Processing failed for {documents[i]}",
                    document_id=documents[i]
                )
            
            # Create checkpoint before "crash"
            checkpoint_id = recovery_system1.create_checkpoint({
                'session_info': 'Before simulated crash',
                'processed_count': 8,
                'failed_count': 4
            })
            
            # Simulate system restart with new recovery system
            recovery_system2 = AdvancedRecoverySystem(checkpoint_dir=checkpoint_dir)
            
            # Verify checkpoint exists
            checkpoints = recovery_system2.checkpoint_manager.list_checkpoints()
            assert checkpoint_id in checkpoints
            
            # Resume from checkpoint
            success = recovery_system2.resume_from_checkpoint(checkpoint_id)
            assert success
            
            # Verify state was restored correctly
            status = recovery_system2.get_recovery_status()
            assert status['document_progress']['processed'] == 8
            assert status['document_progress']['failed'] == 4
            assert status['document_progress']['pending'] == 8  # Remaining documents
            
            # Continue processing from checkpoint
            next_batch = recovery_system2.get_next_batch()
            assert len(next_batch) > 0
            assert all(doc in documents[12:] for doc in next_batch)  # Only unprocessed docs


# Test fixtures and utilities
@pytest.fixture
def temp_checkpoint_dir():
    """Provide a temporary directory for checkpoint testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "checkpoints"


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "essential_metabolomics_001.pdf",
        "critical_pathway_analysis.pdf", 
        "biomarker_study_003.pdf",
        "important_protocol_004.pdf",
        "regular_document_005.pdf",
        "standard_analysis_006.pdf",
        "key_findings_007.pdf",
        "routine_report_008.pdf",
        "essential_methodology_009.pdf",
        "general_document_010.pdf"
    ]


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    pytest.main([__file__, "-v", "--tb=short"])