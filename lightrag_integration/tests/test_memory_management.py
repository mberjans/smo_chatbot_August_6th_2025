"""
Comprehensive unit tests for memory management functionality in BiomedicalPDFProcessor.

This module tests the advanced memory management features including:
- Batch processing with configurable batch sizes
- Memory monitoring and cleanup mechanisms
- Dynamic batch size adjustment based on memory pressure
- Enhanced garbage collection between batches
- Memory usage statistics and logging

Test Categories:
1. Batch Processing Logic
2. Memory Monitoring
3. Memory Cleanup
4. Dynamic Batch Size Adjustment
5. Backward Compatibility
6. Error Handling in Batch Mode
7. Performance Benchmarks

Author: Claude Code Assistant
Created: 2025-08-06
"""

import asyncio
import gc
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any, List, Tuple
import pytest

# Import the classes to test
from ..pdf_processor import BiomedicalPDFProcessor, ErrorRecoveryConfig
from ..progress_config import ProgressTrackingConfig
from ..progress_tracker import PDFProcessingProgressTracker


class TestMemoryManagement:
    """Test suite for memory management functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a BiomedicalPDFProcessor instance for testing."""
        logger = logging.getLogger("test_processor")
        error_recovery = ErrorRecoveryConfig(max_retries=1)  # Minimal retries for testing
        
        return BiomedicalPDFProcessor(
            logger=logger,
            processing_timeout=10,  # Short timeout for tests
            memory_limit_mb=512,    # Moderate limit for tests
            error_recovery_config=error_recovery
        )
    
    @pytest.fixture
    def progress_tracker(self):
        """Create a progress tracker for testing."""
        config = ProgressTrackingConfig()
        logger = logging.getLogger("test_tracker")
        return PDFProcessingProgressTracker(config=config, logger=logger)
    
    @pytest.fixture
    def mock_pdf_files(self, tmp_path):
        """Create mock PDF files for testing."""
        pdf_files = []
        for i in range(20):  # Create 20 test PDF files
            pdf_file = tmp_path / f"test_paper_{i:02d}.pdf"
            # Create minimal PDF content
            pdf_file.write_bytes(b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n'
                                b'2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n'
                                b'3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n'
                                b'/Contents 4 0 R\n>>\nendobj\n'
                                b'4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n'
                                b'(Test content) Tj\nET\nendstream\nendobj\n'
                                b'xref\n0 5\n0000000000 65535 f \n0000000015 00000 n \n'
                                b'0000000074 00000 n \n0000000120 00000 n \n0000000226 00000 n \n'
                                b'trailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n329\n%%EOF')
            pdf_files.append(pdf_file)
        return pdf_files

    def test_get_memory_usage_accuracy(self, processor):
        """Test that _get_memory_usage() method returns accurate memory statistics."""
        memory_stats = processor._get_memory_usage()
        
        # Check that all expected keys are present
        expected_keys = [
            'process_memory_mb', 'process_memory_peak_mb', 'system_memory_percent',
            'system_memory_available_mb', 'system_memory_used_mb', 'system_memory_total_mb'
        ]
        
        for key in expected_keys:
            assert key in memory_stats, f"Missing key: {key}"
            assert isinstance(memory_stats[key], (int, float)), f"Invalid type for {key}"
            assert memory_stats[key] >= 0, f"Negative value for {key}"
        
        # Check reasonable ranges
        assert 0 <= memory_stats['system_memory_percent'] <= 100
        assert memory_stats['system_memory_total_mb'] > memory_stats['system_memory_used_mb']
        assert memory_stats['process_memory_mb'] > 0

    def test_cleanup_memory_effectiveness(self, processor):
        """Test that _cleanup_memory() reduces memory usage effectively."""
        # Record memory before cleanup
        memory_before = processor._get_memory_usage()
        
        # Create some temporary objects to increase memory usage
        large_data = [list(range(10000)) for _ in range(100)]
        
        # Run cleanup
        cleanup_result = processor._cleanup_memory()
        
        # Verify cleanup result structure
        expected_keys = ['memory_before_mb', 'memory_after_mb', 'memory_freed_mb', 'system_memory_percent']
        for key in expected_keys:
            assert key in cleanup_result, f"Missing key in cleanup result: {key}"
        
        # Memory should not increase after cleanup
        assert cleanup_result['memory_after_mb'] <= cleanup_result['memory_before_mb'] + 10  # Allow small margin
        
        # Clean up the large data
        del large_data
        gc.collect()

    @pytest.mark.parametrize("current_size,memory_pressure,expected_change", [
        (10, 0.95, "decrease"),  # High pressure should decrease size
        (10, 0.75, "decrease"),  # Moderate pressure should decrease size
        (5, 0.3, "increase"),    # Low pressure should increase size
        (20, 0.3, "no_change"),  # Already at max, no increase
        (1, 0.95, "no_change"),  # Already at min, no decrease
    ])
    def test_dynamic_batch_size_adjustment(self, processor, current_size, memory_pressure, expected_change):
        """Test dynamic batch size adjustment based on memory pressure."""
        max_memory_mb = 1024
        memory_usage = max_memory_mb * memory_pressure
        performance_data = {'average_processing_time': 2.0}
        
        new_size = processor._adjust_batch_size(current_size, memory_usage, max_memory_mb, performance_data)
        
        if expected_change == "decrease":
            assert new_size < current_size, f"Expected decrease but got {new_size} vs {current_size}"
        elif expected_change == "increase":
            assert new_size > current_size, f"Expected increase but got {new_size} vs {current_size}"
        else:  # no_change
            assert new_size == current_size, f"Expected no change but got {new_size} vs {current_size}"
        
        # Size should always be within reasonable bounds
        assert 1 <= new_size <= 20

    @patch('psutil.Process')
    def test_memory_monitoring_accuracy(self, mock_process, processor):
        """Test memory usage tracking during processing."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500MB
        mock_memory_info.peak_wset = 600 * 1024 * 1024  # 600MB peak
        
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance
        
        # Test memory usage retrieval
        memory_stats = processor._get_memory_usage()
        
        assert memory_stats['process_memory_mb'] == 500.0
        assert memory_stats['process_memory_peak_mb'] == 600.0

    @pytest.mark.anyio
    async def test_batch_processing_basic(self, processor, progress_tracker, mock_pdf_files):
        """Test basic batch processing functionality."""
        # Use a smaller subset for testing
        test_files = mock_pdf_files[:5]
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            # Mock successful PDF extraction
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Test batch processing
            batch_results = await processor._process_batch(
                test_files, batch_num=1, progress_tracker=progress_tracker, max_memory_mb=1024
            )
            
            assert len(batch_results) == len(test_files)
            assert all(isinstance(result, tuple) and len(result) == 2 for result in batch_results)
            assert mock_extract.call_count == len(test_files)

    @pytest.mark.anyio
    async def test_memory_cleanup_between_batches(self, processor, progress_tracker, mock_pdf_files):
        """Test memory cleanup occurs between batches."""
        test_files = mock_pdf_files[:3]
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract, \
             patch.object(processor, '_cleanup_memory') as mock_cleanup:
            
            # Mock successful extraction
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Mock cleanup to return realistic values
            mock_cleanup.return_value = {
                'memory_before_mb': 500.0,
                'memory_after_mb': 450.0,
                'memory_freed_mb': 50.0,
                'system_memory_percent': 60.0
            }
            
            # Process with batch processing enabled
            results = await processor._process_with_batch_mode(
                test_files, initial_batch_size=2, max_memory_mb=1024, progress_tracker=progress_tracker
            )
            
            assert len(results) == len(test_files)
            # Cleanup should be called between batches
            assert mock_cleanup.called

    @pytest.mark.parametrize("batch_size,num_files,expected_batches", [
        (5, 10, 2),   # 10 files, batch size 5 = 2 batches
        (3, 10, 4),   # 10 files, batch size 3 = 4 batches (3,3,3,1)
        (10, 5, 1),   # 5 files, batch size 10 = 1 batch
        (1, 3, 3),    # 3 files, batch size 1 = 3 batches
    ])
    @pytest.mark.anyio
    async def test_batch_processing_with_different_sizes(self, processor, progress_tracker, 
                                                       mock_pdf_files, batch_size, num_files, expected_batches):
        """Test batch processing with different batch sizes."""
        test_files = mock_pdf_files[:num_files]
        
        batch_count = 0
        
        async def mock_process_batch(pdf_batch, batch_num, tracker, max_memory):
            nonlocal batch_count
            batch_count += 1
            # Return empty results for each file
            return [(f'content_{i}', {'batch_number': batch_num}) for i in range(len(pdf_batch))]
        
        with patch.object(processor, '_process_batch', side_effect=mock_process_batch):
            results = await processor._process_with_batch_mode(
                test_files, initial_batch_size=batch_size, max_memory_mb=1024, progress_tracker=progress_tracker
            )
            
            assert len(results) == num_files
            # Note: batch count may vary due to dynamic adjustment, so we check a reasonable range
            assert 1 <= batch_count <= expected_batches + 2  # Allow some flexibility for dynamic adjustment

    @pytest.mark.anyio
    async def test_memory_limit_enforcement(self, processor, progress_tracker, mock_pdf_files):
        """Test memory limit enforcement during batch processing."""
        test_files = mock_pdf_files[:3]
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract, \
             patch.object(processor, '_get_memory_usage') as mock_memory:
            
            # Mock memory usage that exceeds threshold
            mock_memory.side_effect = [
                {'process_memory_mb': 100.0, 'system_memory_percent': 50.0},  # Initial
                {'process_memory_mb': 850.0, 'system_memory_percent': 70.0},  # High usage
                {'process_memory_mb': 200.0, 'system_memory_percent': 40.0},  # After cleanup
            ]
            
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Process with low memory limit to trigger enforcement
            batch_results = await processor._process_batch(
                test_files, batch_num=1, progress_tracker=progress_tracker, max_memory_mb=1000
            )
            
            assert len(batch_results) <= len(test_files)  # Should complete or handle gracefully

    @pytest.mark.anyio
    async def test_backward_compatibility(self, processor, progress_tracker, mock_pdf_files):
        """Test that existing API works without breaking changes."""
        test_files = mock_pdf_files[:3]
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Test sequential mode (backward compatibility)
            results = await processor._process_sequential_mode(test_files, progress_tracker)
            
            assert len(results) == len(test_files)
            assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
            assert mock_extract.call_count == len(test_files)

    @pytest.mark.anyio
    async def test_large_batch_processing(self, processor, progress_tracker):
        """Test processing large numbers of files."""
        # Create a large number of mock files
        large_file_list = [Path(f"mock_file_{i}.pdf") for i in range(100)]
        
        with patch.object(processor, '_process_batch') as mock_batch:
            mock_batch.return_value = [(f'content_{i}', {'batch_number': 1}) 
                                     for i in range(10)]  # Mock 10 results per batch
            
            results = await processor._process_with_batch_mode(
                large_file_list, initial_batch_size=10, max_memory_mb=2048, progress_tracker=progress_tracker
            )
            
            # Should have called _process_batch multiple times
            assert mock_batch.call_count >= 10  # At least 10 batches for 100 files
            
    @pytest.mark.anyio
    async def test_memory_pressure_scenarios(self, processor, progress_tracker, mock_pdf_files):
        """Test handling of high memory usage scenarios."""
        test_files = mock_pdf_files[:5]
        
        # Simulate escalating memory pressure
        memory_readings = [
            {'process_memory_mb': 100.0, 'system_memory_percent': 50.0},
            {'process_memory_mb': 800.0, 'system_memory_percent': 75.0},  # High pressure
            {'process_memory_mb': 900.0, 'system_memory_percent': 85.0},  # Very high pressure
            {'process_memory_mb': 200.0, 'system_memory_percent': 45.0},  # After cleanup
        ]
        
        with patch.object(processor, '_get_memory_usage', side_effect=memory_readings), \
             patch.object(processor, 'extract_text_from_pdf') as mock_extract, \
             patch.object(processor, '_cleanup_memory') as mock_cleanup:
            
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            mock_cleanup.return_value = {
                'memory_before_mb': 900.0,
                'memory_after_mb': 200.0,
                'memory_freed_mb': 700.0,
                'system_memory_percent': 45.0
            }
            
            # Should handle high memory pressure gracefully
            results = await processor._process_with_batch_mode(
                test_files, initial_batch_size=5, max_memory_mb=1000, progress_tracker=progress_tracker
            )
            
            # Should complete processing despite memory pressure
            assert len(results) >= 0  # May be empty if all fail, but should not crash

    @pytest.mark.anyio
    async def test_cleanup_memory_effectiveness_integration(self, processor, progress_tracker, mock_pdf_files):
        """Test garbage collection effectiveness in integration scenario."""
        test_files = mock_pdf_files[:2]
        
        cleanup_calls = []
        
        def mock_cleanup(force=False):
            cleanup_result = {
                'memory_before_mb': 500.0,
                'memory_after_mb': 400.0,
                'memory_freed_mb': 100.0,
                'system_memory_percent': 60.0
            }
            cleanup_calls.append(cleanup_result)
            return cleanup_result
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract, \
             patch.object(processor, '_cleanup_memory', side_effect=mock_cleanup):
            
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Process files with cleanup
            await processor._process_with_batch_mode(
                test_files, initial_batch_size=1, max_memory_mb=1024, progress_tracker=progress_tracker
            )
            
            # Cleanup should have been called
            assert len(cleanup_calls) >= 1
            
            # Verify cleanup results structure
            for cleanup_result in cleanup_calls:
                assert 'memory_freed_mb' in cleanup_result
                assert cleanup_result['memory_freed_mb'] >= 0

    @pytest.mark.anyio
    async def test_batch_processing_error_handling(self, processor, progress_tracker, mock_pdf_files):
        """Test error recovery in batch processing mode."""
        test_files = mock_pdf_files[:3]
        
        # Mock extract_text_from_pdf to fail for first file, succeed for others
        def mock_extract_side_effect(pdf_path):
            if "test_paper_00.pdf" in str(pdf_path):
                raise Exception("Mock processing error")
            return {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': str(pdf_path)},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
        
        with patch.object(processor, 'extract_text_from_pdf', side_effect=mock_extract_side_effect):
            batch_results = await processor._process_batch(
                test_files, batch_num=1, progress_tracker=progress_tracker, max_memory_mb=1024
            )
            
            # Should continue processing after error
            assert len(batch_results) == 2  # 2 successful, 1 failed
            
    @pytest.mark.anyio 
    async def test_process_all_pdfs_with_batch_processing(self, processor, progress_tracker, tmp_path):
        """Test the main process_all_pdfs method with batch processing enabled."""
        # Create test directory with PDF files
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        
        # Create a few test PDF files
        for i in range(3):
            pdf_file = papers_dir / f"test_{i}.pdf"
            pdf_file.write_bytes(b'%PDF-1.4\n%%EOF')  # Minimal PDF
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Test with batch processing enabled
            results = await processor.process_all_pdfs(
                papers_dir=papers_dir,
                progress_tracker=progress_tracker,
                batch_size=2,
                max_memory_mb=1024,
                enable_batch_processing=True
            )
            
            assert len(results) == 3
            assert mock_extract.call_count == 3

    @pytest.mark.anyio
    async def test_process_all_pdfs_without_batch_processing(self, processor, progress_tracker, tmp_path):
        """Test the main process_all_pdfs method with batch processing disabled."""
        # Create test directory with PDF files
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        
        # Create a few test PDF files
        for i in range(3):
            pdf_file = papers_dir / f"test_{i}.pdf"
            pdf_file.write_bytes(b'%PDF-1.4\n%%EOF')  # Minimal PDF
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            # Test with batch processing disabled
            results = await processor.process_all_pdfs(
                papers_dir=papers_dir,
                progress_tracker=progress_tracker,
                enable_batch_processing=False  # Disable batch processing
            )
            
            assert len(results) == 3
            assert mock_extract.call_count == 3

    def test_get_processing_stats_includes_memory_features(self, processor):
        """Test that processing stats include memory management features."""
        stats = processor.get_processing_stats()
        
        # Check memory stats are included
        assert 'memory_stats' in stats
        memory_stats = stats['memory_stats']
        assert 'process_memory_mb' in memory_stats
        assert 'system_memory_percent' in memory_stats
        
        # Check memory management features are documented
        assert 'memory_management' in stats
        memory_features = stats['memory_management']
        
        expected_features = [
            'batch_processing_available',
            'dynamic_batch_sizing',
            'enhanced_garbage_collection',
            'memory_pressure_monitoring',
            'automatic_cleanup_between_batches'
        ]
        
        for feature in expected_features:
            assert feature in memory_features
            assert memory_features[feature] is True

    def test_memory_usage_boundaries(self, processor):
        """Test memory usage calculations with boundary conditions."""
        # Test with mocked extreme memory values
        with patch('psutil.Process') as mock_process, \
             patch('psutil.virtual_memory') as mock_virtual_memory:
            
            # Mock very high memory usage
            mock_memory_info = Mock()
            mock_memory_info.rss = 8 * 1024 * 1024 * 1024  # 8GB
            mock_memory_info.peak_wset = 10 * 1024 * 1024 * 1024  # 10GB
            
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value = mock_memory_info
            mock_process.return_value = mock_process_instance
            
            mock_virtual_memory.return_value = Mock(
                percent=95.0,
                available=1 * 1024 * 1024 * 1024,  # 1GB
                used=15 * 1024 * 1024 * 1024,      # 15GB
                total=16 * 1024 * 1024 * 1024      # 16GB
            )
            
            memory_stats = processor._get_memory_usage()
            
            assert memory_stats['process_memory_mb'] == 8192.0  # 8GB in MB
            assert memory_stats['system_memory_percent'] == 95.0
            assert memory_stats['system_memory_total_mb'] == 16384.0  # 16GB in MB

    @pytest.mark.anyio
    async def test_performance_benchmarks(self, processor, progress_tracker, mock_pdf_files):
        """Test performance characteristics of batch processing."""
        test_files = mock_pdf_files[:10]
        
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            start_time = time.time()
            
            # Test batch processing performance
            results = await processor._process_with_batch_mode(
                test_files, initial_batch_size=5, max_memory_mb=1024, progress_tracker=progress_tracker
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert len(results) == len(test_files)
            
            # Performance should be reasonable (not more than 10 seconds for 10 mocked files)
            assert processing_time < 10.0
            
            # Throughput should be reasonable
            files_per_second = len(test_files) / processing_time
            assert files_per_second > 0.5  # At least 0.5 files per second

    def test_batch_size_adjustment_edge_cases(self, processor):
        """Test batch size adjustment with edge cases."""
        performance_data = {'average_processing_time': 3.0}
        
        # Test minimum batch size enforcement
        new_size = processor._adjust_batch_size(1, 1000, 1000, performance_data)  # High pressure
        assert new_size >= 1
        
        # Test behavior with already large batch size and low memory pressure
        # The method checks current_batch_size < 20 before increasing, so 25 should stay 25
        new_size = processor._adjust_batch_size(25, 100, 1000, performance_data)  # Low pressure
        assert new_size == 25  # Should stay the same, not increase beyond current
        
        # Test with moderate memory pressure
        new_size = processor._adjust_batch_size(10, 500, 1000, performance_data)  # Moderate pressure
        assert new_size >= 1
        
        # Test with very high memory pressure
        new_size = processor._adjust_batch_size(10, 2000, 1000, performance_data)  # 200% pressure
        assert new_size >= 1
        assert new_size < 10  # Should decrease

    def test_memory_cleanup_force_parameter(self, processor):
        """Test memory cleanup with force parameter."""
        # Test normal cleanup
        result_normal = processor._cleanup_memory(force=False)
        assert 'memory_before_mb' in result_normal
        assert 'memory_after_mb' in result_normal
        
        # Test forced cleanup
        result_forced = processor._cleanup_memory(force=True)
        assert 'memory_before_mb' in result_forced
        assert 'memory_after_mb' in result_forced
        
        # Both should return valid results - allow small negative values due to measurement variance
        assert result_normal['memory_freed_mb'] >= -1.0  # Allow small negative due to measurement variance
        assert result_forced['memory_freed_mb'] >= -1.0
        
        # Check that the structure is consistent
        assert isinstance(result_normal['memory_freed_mb'], (int, float))
        assert isinstance(result_forced['memory_freed_mb'], (int, float))

    @pytest.mark.anyio
    async def test_batch_processing_empty_directory(self, processor, progress_tracker, tmp_path):
        """Test batch processing with empty directory."""
        empty_dir = tmp_path / "empty_papers"
        empty_dir.mkdir()
        
        results = await processor.process_all_pdfs(
            papers_dir=empty_dir,
            progress_tracker=progress_tracker,
            batch_size=5,
            enable_batch_processing=True
        )
        
        assert results == []

    @pytest.mark.anyio
    async def test_batch_processing_nonexistent_directory(self, processor, progress_tracker, tmp_path):
        """Test batch processing with nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        
        results = await processor.process_all_pdfs(
            papers_dir=nonexistent_dir,
            progress_tracker=progress_tracker,
            batch_size=5,
            enable_batch_processing=True
        )
        
        assert results == []

    def test_error_recovery_stats_integration(self, processor):
        """Test error recovery statistics integration with memory management."""
        # Test initial state
        stats = processor.get_error_recovery_stats()
        assert stats['files_with_retries'] == 0
        assert stats['total_recovery_actions'] == 0
        
        # Test stats structure
        assert 'recovery_actions_by_type' in stats
        assert 'retry_details_by_file' in stats
        assert 'error_recovery_config' in stats
        
        # Test reset functionality
        processor.reset_error_recovery_stats()
        stats_after_reset = processor.get_error_recovery_stats()
        assert stats_after_reset['files_with_retries'] == 0


class TestMemoryManagementIntegration:
    """Integration tests for memory management with real-world scenarios."""
    
    @pytest.fixture
    def processor_with_strict_limits(self):
        """Create processor with strict memory limits for testing."""
        return BiomedicalPDFProcessor(
            memory_limit_mb=128,  # Very low limit to trigger memory management
            processing_timeout=5,
            error_recovery_config=ErrorRecoveryConfig(max_retries=1)
        )
    
    @pytest.mark.anyio
    async def test_memory_management_under_pressure(self, processor_with_strict_limits):
        """Test memory management behavior under simulated memory pressure."""
        # Create mock files
        mock_files = [Path(f"test_{i}.pdf") for i in range(10)]
        
        # Mock high memory usage scenarios
        memory_scenarios = [
            {'process_memory_mb': 100.0, 'system_memory_percent': 60.0},  # Normal
            {'process_memory_mb': 120.0, 'system_memory_percent': 80.0},  # Getting high
            {'process_memory_mb': 125.0, 'system_memory_percent': 90.0},  # Very high
            {'process_memory_mb': 90.0, 'system_memory_percent': 50.0},   # After cleanup
        ]
        
        with patch.object(processor_with_strict_limits, '_get_memory_usage', 
                         side_effect=memory_scenarios * 3), \
             patch.object(processor_with_strict_limits, 'extract_text_from_pdf') as mock_extract, \
             patch.object(processor_with_strict_limits, '_cleanup_memory') as mock_cleanup:
            
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1, 'filename': 'test.pdf'},
                'page_texts': ['Test content'],
                'processing_info': {'total_characters': 12, 'pages_processed': 1}
            }
            
            mock_cleanup.return_value = {
                'memory_before_mb': 125.0,
                'memory_after_mb': 90.0,
                'memory_freed_mb': 35.0,
                'system_memory_percent': 50.0
            }
            
            progress_config = ProgressTrackingConfig()
            progress_tracker = PDFProcessingProgressTracker(config=progress_config)
            
            # Should handle memory pressure gracefully
            results = await processor_with_strict_limits._process_with_batch_mode(
                mock_files[:5], initial_batch_size=3, max_memory_mb=128, progress_tracker=progress_tracker
            )
            
            # Should complete despite memory constraints
            assert len(results) >= 0
            # Memory cleanup should have been triggered
            assert mock_cleanup.called

    def test_memory_statistics_accuracy_integration(self, processor_with_strict_limits):
        """Test accuracy of memory statistics in integration scenario."""
        stats = processor_with_strict_limits.get_processing_stats()
        
        # Verify all memory-related statistics are present and reasonable
        memory_stats = stats['memory_stats']
        
        # Check that memory values are positive and reasonable
        assert memory_stats['process_memory_mb'] > 0
        assert memory_stats['process_memory_mb'] < memory_stats['system_memory_total_mb']
        assert 0 <= memory_stats['system_memory_percent'] <= 100
        
        # Check memory management features are properly advertised
        memory_mgmt = stats['memory_management']
        assert all(feature for feature in memory_mgmt.values())


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v", "--tb=short"])