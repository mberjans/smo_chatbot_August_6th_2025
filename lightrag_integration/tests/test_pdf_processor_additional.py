"""
Comprehensive additional tests for BiomedicalPDFProcessor to achieve 90%+ coverage.

This test file focuses on previously untested functionality including:
- ErrorRecoveryConfig edge cases and boundary conditions
- Error classification and recovery mechanisms
- Memory management and cleanup functionality
- Text preprocessing and validation edge cases
- Timeout and memory monitoring
- Concurrent processing scenarios
- Error recovery statistics and logging
- Performance monitoring and resource utilization
- Integration tests for complex processing scenarios
- Edge cases and boundary conditions

These tests complement the existing test_pdf_processor.py file to achieve
comprehensive coverage of the pdf_processor module.
"""

import pytest
import asyncio
import time
import gc
import random
import tempfile
import shutil
import psutil
import threading
import signal
import os
import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, AsyncMock, call
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF

from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor, BiomedicalPDFProcessorError, 
    PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError, 
    PDFFileAccessError, PDFContentError, ErrorRecoveryConfig
)


# =====================================================================
# TEST FIXTURES FOR ADDITIONAL COVERAGE
# =====================================================================

@pytest.fixture
def mock_memory_stats():
    """Mock memory statistics for testing."""
    return {
        'process_memory_mb': 256.5,
        'process_memory_peak_mb': 512.0,
        'system_memory_percent': 45.2,
        'system_memory_available_mb': 4096.0,
        'system_memory_used_mb': 2048.0,
        'system_memory_total_mb': 8192.0
    }

@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        # Create a minimal valid PDF structure
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000010 00000 n 
0000000053 00000 n 
0000000100 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
156
%%EOF"""
        temp_file.write(pdf_content)
        temp_file.flush()
        
        yield Path(temp_file.name)
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except (OSError, FileNotFoundError):
            pass

@pytest.fixture
def corrupted_pdf_file():
    """Create a corrupted PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        # Write invalid PDF content
        temp_file.write(b"This is not a valid PDF file content")
        temp_file.flush()
        
        yield Path(temp_file.name)
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except (OSError, FileNotFoundError):
            pass

@pytest.fixture
def large_text_content():
    """Generate large text content for testing memory limits."""
    base_text = "This is a test sentence with biomedical content. "
    # Create ~2MB of text
    return base_text * 50000

@pytest.fixture
def processor_with_recovery():
    """Create processor with custom error recovery config."""
    recovery_config = ErrorRecoveryConfig(
        max_retries=5,
        base_delay=0.1,
        max_delay=2.0,
        exponential_base=1.5,
        jitter=True,
        memory_recovery_enabled=True,
        file_lock_retry_enabled=True,
        timeout_retry_enabled=True
    )
    return BiomedicalPDFProcessor(
        processing_timeout=30,
        memory_limit_mb=512,
        max_page_text_size=500000,
        error_recovery_config=recovery_config
    )


# =====================================================================
# TESTS FOR ERRORRECOVERYCONFIG CLASS
# =====================================================================

class TestErrorRecoveryConfig:
    """Comprehensive tests for ErrorRecoveryConfig class."""

    @pytest.mark.pdf_processor
    def test_error_recovery_config_default_initialization(self):
        """Test ErrorRecoveryConfig with default parameters."""
        config = ErrorRecoveryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.memory_recovery_enabled is True
        assert config.file_lock_retry_enabled is True
        assert config.timeout_retry_enabled is True

    @pytest.mark.pdf_processor
    def test_error_recovery_config_custom_initialization(self):
        """Test ErrorRecoveryConfig with custom parameters."""
        config = ErrorRecoveryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            memory_recovery_enabled=False,
            file_lock_retry_enabled=False,
            timeout_retry_enabled=False
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.memory_recovery_enabled is False
        assert config.file_lock_retry_enabled is False
        assert config.timeout_retry_enabled is False

    @pytest.mark.pdf_processor
    def test_error_recovery_config_calculate_delay_exponential_backoff(self):
        """Test delay calculation with exponential backoff."""
        config = ErrorRecoveryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
        # Test exponential progression
        assert config.calculate_delay(0) == 1.0  # 1.0 * 2^0
        assert config.calculate_delay(1) == 2.0  # 1.0 * 2^1
        assert config.calculate_delay(2) == 4.0  # 1.0 * 2^2
        assert config.calculate_delay(3) == 8.0  # 1.0 * 2^3
        assert config.calculate_delay(4) == 10.0  # Capped at max_delay

    @pytest.mark.pdf_processor
    def test_error_recovery_config_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter enabled."""
        config = ErrorRecoveryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # With jitter, delay should be base delay + random jitter
        delay_0 = config.calculate_delay(0)
        delay_1 = config.calculate_delay(1)
        
        # Should be at least the base delay
        assert delay_0 >= 1.0
        assert delay_1 >= 2.0
        
        # Should not exceed base delay + 25% jitter
        assert delay_0 <= 1.25
        assert delay_1 <= 2.5
        
        # Multiple calls should return different values due to jitter
        delays = [config.calculate_delay(0) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have variation

    @pytest.mark.pdf_processor
    def test_error_recovery_config_calculate_delay_max_limit(self):
        """Test that calculated delay respects max_delay limit."""
        config = ErrorRecoveryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=3.0,
            jitter=False
        )
        
        # High attempt numbers should be capped at max_delay
        assert config.calculate_delay(10) == 5.0
        assert config.calculate_delay(100) == 5.0

    @pytest.mark.pdf_processor
    def test_error_recovery_config_edge_cases(self):
        """Test ErrorRecoveryConfig edge cases and boundary conditions."""
        # Zero base delay
        config = ErrorRecoveryConfig(base_delay=0.0, jitter=False)
        assert config.calculate_delay(1) == 0.0
        
        # Very small exponential base
        config = ErrorRecoveryConfig(
            base_delay=1.0, 
            exponential_base=1.01, 
            jitter=False
        )
        delay = config.calculate_delay(1)
        assert delay == pytest.approx(1.01, rel=1e-3)
        
        # Max delay smaller than base delay
        config = ErrorRecoveryConfig(
            base_delay=10.0, 
            max_delay=5.0, 
            jitter=False
        )
        assert config.calculate_delay(0) == 5.0  # Should be capped


# =====================================================================
# TESTS FOR ERROR CLASSIFICATION AND RECOVERY MECHANISMS
# =====================================================================

class TestErrorClassificationAndRecovery:
    """Tests for error classification and recovery mechanisms."""

    @pytest.mark.pdf_processor
    def test_classify_error_memory_errors(self, processor_with_recovery):
        """Test classification of memory-related errors."""
        # Test PDFMemoryError
        memory_error = PDFMemoryError("Out of memory")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(memory_error)
        assert is_recoverable is True
        assert category == "memory"
        assert strategy == "memory_cleanup"
        
        # Test standard MemoryError
        mem_error = MemoryError("Cannot allocate memory")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(mem_error)
        assert is_recoverable is True
        assert category == "memory"
        assert strategy == "memory_cleanup"
        
        # Test error with "memory" in message
        generic_error = Exception("Memory allocation failed")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(generic_error)
        assert is_recoverable is True
        assert category == "memory"
        assert strategy == "memory_cleanup"

    @pytest.mark.pdf_processor
    def test_classify_error_timeout_errors(self, processor_with_recovery):
        """Test classification of timeout-related errors."""
        # Test PDFProcessingTimeoutError
        timeout_error = PDFProcessingTimeoutError("Processing timed out")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(timeout_error)
        assert is_recoverable is True
        assert category == "timeout"
        assert strategy == "timeout_retry"
        
        # Test error with "timeout" in message
        generic_error = Exception("Operation timeout occurred")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(generic_error)
        assert is_recoverable is True
        assert category == "timeout"
        assert strategy == "timeout_retry"

    @pytest.mark.pdf_processor
    def test_classify_error_file_access_errors(self, processor_with_recovery):
        """Test classification of file access errors."""
        # Test file lock error
        lock_error = PDFFileAccessError("File is locked or in use by another process")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(lock_error)
        assert is_recoverable is True
        assert category == "file_lock"
        assert strategy == "file_lock_retry"
        
        # Test permission error
        perm_error = PDFFileAccessError("Permission denied")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(perm_error)
        assert is_recoverable is False
        assert category == "permission"
        assert strategy == "skip"
        
        # Test generic file access error
        access_error = PDFFileAccessError("Cannot access file")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(access_error)
        assert is_recoverable is True
        assert category == "file_access"
        assert strategy == "simple_retry"

    @pytest.mark.pdf_processor
    def test_classify_error_validation_errors(self, processor_with_recovery):
        """Test classification of validation errors."""
        # Test corrupted PDF
        corrupt_error = PDFValidationError("PDF is corrupted or invalid")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(corrupt_error)
        assert is_recoverable is False
        assert category == "corruption"
        assert strategy == "skip"
        
        # Test generic validation error
        validation_error = PDFValidationError("Validation failed")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(validation_error)
        assert is_recoverable is True
        assert category == "validation"
        assert strategy == "simple_retry"

    @pytest.mark.pdf_processor
    def test_classify_error_content_errors(self, processor_with_recovery):
        """Test classification of content errors."""
        content_error = PDFContentError("No extractable content")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(content_error)
        assert is_recoverable is False
        assert category == "content"
        assert strategy == "skip"

    @pytest.mark.pdf_processor
    def test_classify_error_io_errors(self, processor_with_recovery):
        """Test classification of IO errors."""
        # Test disk space error
        disk_error = IOError("No space left on device")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(disk_error)
        assert is_recoverable is False
        assert category == "disk_space"
        assert strategy == "skip"
        
        # Test generic IO error
        io_error = IOError("Input/output error")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(io_error)
        assert is_recoverable is True
        assert category == "io_error"
        assert strategy == "simple_retry"

    @pytest.mark.pdf_processor
    def test_classify_error_fitz_errors(self, processor_with_recovery):
        """Test classification of PyMuPDF (fitz) specific errors."""
        # Create mock fitz error with timeout message - use a class that inherits from Exception
        class FitzTimeoutError(Exception):
            pass
        
        fitz_timeout_error = FitzTimeoutError("fitz timeout occurred")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(fitz_timeout_error)
        assert is_recoverable is True
        assert category == "fitz_timeout"
        assert strategy == "timeout_retry"
        
        # Create mock fitz error with memory message
        fitz_memory_error = Exception("mupdf memory allocation failed")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(fitz_memory_error)
        assert is_recoverable is True
        assert category == "fitz_memory"
        assert strategy == "memory_cleanup"
        
        # Create generic fitz error
        class FitzError(Exception):
            pass
            
        fitz_error = FitzError("fitz processing error")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(fitz_error)
        assert is_recoverable is True
        assert category == "fitz_error"
        assert strategy == "simple_retry"

    @pytest.mark.pdf_processor
    def test_classify_error_unknown_errors(self, processor_with_recovery):
        """Test classification of unknown errors."""
        unknown_error = RuntimeError("Unknown processing error")
        is_recoverable, category, strategy = processor_with_recovery._classify_error(unknown_error)
        assert is_recoverable is True
        assert category == "unknown"
        assert strategy == "simple_retry"

    @pytest.mark.pdf_processor
    @patch('time.sleep')
    def test_attempt_memory_recovery(self, mock_sleep, processor_with_recovery):
        """Test memory recovery attempt."""
        with patch('gc.collect') as mock_gc, \
             patch('psutil.Process') as mock_process:
            
            mock_process_instance = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB
            mock_process_instance.memory_info.return_value = mock_memory_info
            mock_process.return_value = mock_process_instance
            
            result = processor_with_recovery._attempt_memory_recovery()
            
            assert result is True
            mock_gc.assert_called()
            mock_sleep.assert_called_with(0.5)

    @pytest.mark.pdf_processor
    @patch('time.sleep')
    def test_attempt_file_lock_recovery(self, mock_sleep, processor_with_recovery, temp_pdf_file):
        """Test file lock recovery attempt."""
        result = processor_with_recovery._attempt_file_lock_recovery(temp_pdf_file, 1)
        
        assert result is True
        mock_sleep.assert_called()
        # Should sleep for at least base delay * 2^attempt
        args = mock_sleep.call_args[0]
        assert args[0] >= 2.0  # base_delay * 2^1

    @pytest.mark.pdf_processor
    def test_attempt_timeout_recovery(self, processor_with_recovery):
        """Test timeout recovery attempt."""
        original_timeout = processor_with_recovery.processing_timeout
        
        result = processor_with_recovery._attempt_timeout_recovery(1)
        
        assert result is True
        # Timeout should be increased
        assert processor_with_recovery.processing_timeout > original_timeout

    @pytest.mark.pdf_processor
    @patch('time.sleep')
    def test_attempt_simple_recovery(self, mock_sleep, processor_with_recovery):
        """Test simple recovery attempt."""
        result = processor_with_recovery._attempt_simple_recovery(2)
        
        assert result is True
        mock_sleep.assert_called()
        # Should use exponential backoff
        args = mock_sleep.call_args[0]
        assert args[0] > 0

    @pytest.mark.pdf_processor
    def test_attempt_error_recovery_disabled_strategies(self):
        """Test error recovery when specific strategies are disabled."""
        config = ErrorRecoveryConfig(
            memory_recovery_enabled=False,
            file_lock_retry_enabled=False,
            timeout_retry_enabled=False
        )
        processor = BiomedicalPDFProcessor(error_recovery_config=config)
        
        # Test disabled memory recovery
        result = processor._attempt_memory_recovery()
        assert result is False
        
        # Test disabled file lock recovery
        result = processor._attempt_file_lock_recovery(Path("test.pdf"), 0)
        assert result is False
        
        # Test disabled timeout recovery
        result = processor._attempt_timeout_recovery(0)
        assert result is False

    @pytest.mark.pdf_processor
    def test_attempt_error_recovery_unknown_strategy(self, processor_with_recovery, temp_pdf_file, caplog):
        """Test error recovery with unknown strategy."""
        result = processor_with_recovery._attempt_error_recovery(
            "test_category", "unknown_strategy", temp_pdf_file, 0
        )
        
        assert result is False
        assert "Unknown recovery strategy: unknown_strategy" in caplog.text


# =====================================================================
# TESTS FOR MEMORY MANAGEMENT AND CLEANUP FUNCTIONALITY
# =====================================================================

class TestMemoryManagementAndCleanup:
    """Tests for memory management and cleanup functionality."""

    @pytest.mark.pdf_processor
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_get_memory_usage(self, mock_virtual_memory, mock_process, processor_with_recovery):
        """Test memory usage statistics collection."""
        # Mock process memory info
        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB
        mock_memory_info.peak_wset = 512 * 1024 * 1024  # 512 MB (Windows-specific)
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance
        
        # Mock system memory info
        mock_vm = Mock()
        mock_vm.percent = 45.2
        mock_vm.available = 4096 * 1024 * 1024  # 4 GB
        mock_vm.used = 2048 * 1024 * 1024  # 2 GB
        mock_vm.total = 8192 * 1024 * 1024  # 8 GB
        mock_virtual_memory.return_value = mock_vm
        
        memory_stats = processor_with_recovery._get_memory_usage()
        
        assert memory_stats['process_memory_mb'] == 256.0
        assert memory_stats['process_memory_peak_mb'] == 512.0
        assert memory_stats['system_memory_percent'] == 45.2
        assert memory_stats['system_memory_available_mb'] == 4096.0
        assert memory_stats['system_memory_used_mb'] == 2048.0
        assert memory_stats['system_memory_total_mb'] == 8192.0

    @pytest.mark.pdf_processor
    @patch('time.sleep')
    @patch('gc.collect')
    @patch.object(BiomedicalPDFProcessor, '_get_memory_usage')
    def test_cleanup_memory_basic(self, mock_get_memory, mock_gc, mock_sleep, processor_with_recovery):
        """Test basic memory cleanup functionality."""
        # Mock memory usage before and after cleanup
        mock_get_memory.side_effect = [
            {'process_memory_mb': 512.0},  # Before cleanup
            {'process_memory_mb': 256.0, 'system_memory_percent': 40.0}  # After cleanup
        ]
        
        result = processor_with_recovery._cleanup_memory()
        
        # Verify garbage collection was called multiple times (3 initial calls)
        assert mock_gc.call_count >= 3
        
        # Verify sleep was called for delays (at least 3 times between GC cycles + final delay)
        assert mock_sleep.call_count >= 4
        
        # Verify result contains memory statistics
        assert result['memory_before_mb'] == 512.0
        assert result['memory_after_mb'] == 256.0
        assert result['memory_freed_mb'] == 256.0
        assert result['system_memory_percent'] == 40.0

    @pytest.mark.pdf_processor
    @patch('gc.get_count')
    @patch('gc.collect')
    def test_cleanup_memory_with_generations(self, mock_gc_collect, mock_get_count, processor_with_recovery):
        """Test memory cleanup with garbage collection generations."""
        # Mock GC count to return 3 generations
        mock_get_count.return_value = [100, 50, 25]  # 3 generations
        
        with patch.object(processor_with_recovery, '_get_memory_usage') as mock_get_memory:
            mock_get_memory.side_effect = [
                {'process_memory_mb': 512.0},
                {'process_memory_mb': 256.0, 'system_memory_percent': 40.0}
            ]
            
            processor_with_recovery._cleanup_memory()
            
            # Should call collect for each generation plus initial calls
            expected_calls = 3 + 3  # 3 initial calls + 3 generation calls
            assert mock_gc_collect.call_count >= expected_calls

    @pytest.mark.pdf_processor
    def test_adjust_batch_size_high_memory_pressure(self, processor_with_recovery):
        """Test batch size adjustment under high memory pressure."""
        performance_data = {'average_processing_time': 2.0}
        
        # High memory pressure (>90%)
        new_batch_size = processor_with_recovery._adjust_batch_size(
            current_batch_size=10,
            memory_usage=1024,  # 1024 MB usage
            max_memory_mb=1000,  # 1000 MB limit (>100% pressure)
            performance_data=performance_data
        )
        
        assert new_batch_size == 5  # Should be halved

    @pytest.mark.pdf_processor
    def test_adjust_batch_size_moderate_memory_pressure(self, processor_with_recovery):
        """Test batch size adjustment under moderate memory pressure."""
        performance_data = {'average_processing_time': 2.0}
        
        # Moderate memory pressure (70-90%)
        new_batch_size = processor_with_recovery._adjust_batch_size(
            current_batch_size=10,
            memory_usage=800,  # 80% of max
            max_memory_mb=1000,
            performance_data=performance_data
        )
        
        assert new_batch_size == 8  # Should be reduced to 80%

    @pytest.mark.pdf_processor
    def test_adjust_batch_size_low_memory_pressure_good_performance(self, processor_with_recovery):
        """Test batch size increase under low memory pressure with good performance."""
        performance_data = {'average_processing_time': 3.0}  # Fast processing
        
        # Low memory pressure (<40%) with good performance
        new_batch_size = processor_with_recovery._adjust_batch_size(
            current_batch_size=5,
            memory_usage=300,  # 30% of max
            max_memory_mb=1000,
            performance_data=performance_data
        )
        
        assert new_batch_size == 7  # Should increase by 2

    @pytest.mark.pdf_processor
    def test_adjust_batch_size_low_memory_pressure_poor_performance(self, processor_with_recovery):
        """Test batch size stability under low memory pressure with poor performance."""
        performance_data = {'average_processing_time': 10.0}  # Slow processing
        
        # Low memory pressure but poor performance
        new_batch_size = processor_with_recovery._adjust_batch_size(
            current_batch_size=5,
            memory_usage=300,  # 30% of max
            max_memory_mb=1000,
            performance_data=performance_data
        )
        
        assert new_batch_size == 5  # Should remain unchanged

    @pytest.mark.pdf_processor
    def test_adjust_batch_size_maximum_limits(self, processor_with_recovery):
        """Test batch size adjustment respects maximum limits."""
        performance_data = {'average_processing_time': 1.0}
        
        # Test minimum limit (high memory pressure)
        new_batch_size = processor_with_recovery._adjust_batch_size(
            current_batch_size=2,
            memory_usage=1500,  # Very high memory usage
            max_memory_mb=1000,
            performance_data=performance_data
        )
        
        assert new_batch_size == 1  # Should not go below 1
        
        # Test maximum limit (low memory pressure)
        new_batch_size = processor_with_recovery._adjust_batch_size(
            current_batch_size=18,
            memory_usage=200,  # Very low memory usage
            max_memory_mb=1000,
            performance_data=performance_data
        )
        
        assert new_batch_size == 20  # Should be capped at 20

    @pytest.mark.pdf_processor
    @patch.object(BiomedicalPDFProcessor, '_get_memory_usage')
    @patch.object(BiomedicalPDFProcessor, '_cleanup_memory')
    def test_monitor_memory_context_manager(self, mock_cleanup, mock_get_memory, processor_with_recovery):
        """Test memory monitoring context manager."""
        # Mock memory usage
        mock_get_memory.side_effect = [
            {'process_memory_mb': 256.0},  # Initial
            {'process_memory_mb': 768.0, 'system_memory_percent': 85.0}  # Final
        ]
        
        with processor_with_recovery._monitor_memory():
            assert processor_with_recovery._memory_monitor_active is True
        
        assert processor_with_recovery._memory_monitor_active is False
        
        # Should detect memory increase but not clean up (processing complete)
        mock_cleanup.assert_not_called()

    @pytest.mark.pdf_processor
    @patch.object(BiomedicalPDFProcessor, '_get_memory_usage')
    @patch('psutil.virtual_memory')
    def test_monitor_memory_high_system_usage_warning(self, mock_virtual_memory, mock_get_memory, processor_with_recovery, caplog):
        """Test memory monitoring warns about high system memory usage."""
        # Mock high system memory usage
        mock_vm = Mock()
        mock_vm.percent = 95.0  # High system memory usage
        mock_virtual_memory.return_value = mock_vm
        
        mock_get_memory.side_effect = [
            {'process_memory_mb': 256.0},
            {'process_memory_mb': 512.0, 'system_memory_percent': 95.0}
        ]
        
        with processor_with_recovery._monitor_memory():
            pass
        
        assert "System memory usage high: 95.0%" in caplog.text


# =====================================================================
# TESTS FOR TEXT PREPROCESSING AND VALIDATION
# =====================================================================

class TestTextPreprocessingAndValidation:
    """Comprehensive tests for text preprocessing and validation functionality."""

    @pytest.mark.pdf_processor
    def test_validate_and_clean_page_text_empty_page(self, processor_with_recovery):
        """Test validation of empty page text."""
        result = processor_with_recovery._validate_and_clean_page_text("", 1)
        assert result == ""

    @pytest.mark.pdf_processor
    def test_validate_and_clean_page_text_large_content(self, processor_with_recovery, large_text_content):
        """Test validation of oversized page text."""
        # Set a small max page text size for testing
        processor_with_recovery.max_page_text_size = 1000
        
        result = processor_with_recovery._validate_and_clean_page_text(large_text_content, 1)
        
        assert len(result) <= 1000 + len("\n[TEXT TRUNCATED DUE TO SIZE LIMIT]")
        assert "[TEXT TRUNCATED DUE TO SIZE LIMIT]" in result

    @pytest.mark.pdf_processor
    def test_validate_and_clean_page_text_encoding_issues(self, processor_with_recovery, caplog):
        """Test handling of text with encoding issues."""
        # Create text with problematic Unicode characters
        problematic_text = "Test text with \udcff invalid surrogate"
        
        result = processor_with_recovery._validate_and_clean_page_text(problematic_text, 1)
        
        # Should handle encoding issues gracefully
        assert result is not None
        assert "Encoding issues detected" in caplog.text

    @pytest.mark.pdf_processor
    def test_validate_and_clean_page_text_excessive_control_chars(self, processor_with_recovery, caplog):
        """Test handling of text with excessive control characters."""
        # Create text with many control characters
        control_heavy_text = "Normal text" + "\x00\x01\x02\x03\x04\x05" * 20
        
        result = processor_with_recovery._validate_and_clean_page_text(control_heavy_text, 1)
        
        # Control characters should be cleaned up
        assert "\x00" not in result
        assert "Excessive control characters" in caplog.text

    @pytest.mark.pdf_processor
    def test_validate_text_encoding_unicode_replacements(self, processor_with_recovery):
        """Test Unicode character replacements in text encoding validation."""
        # Test text with various Unicode characters that should be replaced
        unicode_text = "Test\u2013dash\u2014em\u2018quote\u2019end\u201cleft\u201dright\u2026ellipsis\u00a0space\u00b7dot\u2022bullet"
        
        result = processor_with_recovery._validate_text_encoding(unicode_text)
        
        # Check replacements
        assert "\u2013" not in result  # en dash should be replaced
        assert "\u2014" not in result  # em dash should be replaced
        assert "\u2018" not in result  # left single quote should be replaced
        assert "\u2019" not in result  # right single quote should be replaced
        assert "\u201c" not in result  # left double quote should be replaced
        assert "\u201d" not in result  # right double quote should be replaced
        assert "\u2026" not in result  # ellipsis should be replaced
        assert "\u00a0" not in result  # non-breaking space should be replaced
        assert "\u00b7" not in result  # middle dot should be replaced
        assert "\u2022" not in result  # bullet should be replaced
        
        # Check that replacements are correct
        assert "-" in result or "--" in result  # dashes replaced
        assert "'" in result  # quotes replaced
        assert '"' in result  # double quotes replaced
        assert "..." in result  # ellipsis replaced
        assert " " in result  # spaces handled
        assert "*" in result  # bullets and dots replaced

    @pytest.mark.pdf_processor
    def test_remove_pdf_artifacts_page_numbers(self, processor_with_recovery):
        """Test removal of page number artifacts."""
        text_with_page_numbers = """
        Some content here.
        
        123
        
        More content.
        Page 45
        Additional content.
        Page 67 of 100
        Final content.
        """
        
        result = processor_with_recovery._remove_pdf_artifacts(text_with_page_numbers)
        
        # Page numbers should be removed
        assert "123" not in result.split('\n')  # Standalone number
        assert "Page 45" not in result
        assert "Page 67 of 100" not in result

    @pytest.mark.pdf_processor
    def test_remove_pdf_artifacts_headers_footers(self, processor_with_recovery):
        """Test removal of header and footer artifacts."""
        text_with_headers = """
        Content here.
        
        Nature Journal 2023 Volume 15
        
        More content.
        
        doi: 10.1038/example
        
        Final content.
        
        © 2023 Nature Publishing Group
        
        Downloaded from nature.com on 2023-10-15
        """
        
        result = processor_with_recovery._remove_pdf_artifacts(text_with_headers)
        
        # Headers and footers should be removed
        assert "Nature Journal 2023" not in result
        assert "doi: 10.1038/example" not in result
        assert "© 2023 Nature Publishing Group" not in result
        assert "Downloaded from nature.com" not in result

    @pytest.mark.pdf_processor
    def test_fix_text_extraction_issues_hyphenated_words(self, processor_with_recovery):
        """Test fixing of hyphenated words broken across lines."""
        broken_text = """
        This is a sen-
        tence with broken words.
        Another bro-
        ken word here.
        """
        
        result = processor_with_recovery._fix_text_extraction_issues(broken_text)
        
        # Hyphenated words should be fixed
        assert "sentence" in result
        assert "broken" in result
        assert "sen-\n        tence" not in result

    @pytest.mark.pdf_processor
    def test_fix_text_extraction_issues_spacing_punctuation(self, processor_with_recovery):
        """Test fixing of spacing issues around punctuation."""
        spaced_text = """
        Word1.Word2 should be fixed .
        Also fix spacing around ( parentheses ) and [ brackets ] .
        """
        
        result = processor_with_recovery._fix_text_extraction_issues(spaced_text)
        
        # Spacing should be fixed - check that punctuation is properly spaced
        assert "Word1. Word2" in result
        # Check that excessive spacing around punctuation is reduced
        assert result.count(" .") < spaced_text.count(" .")
        assert result.count(" (") < spaced_text.count(" (")
        assert result.count(" )") < spaced_text.count(" )")
        assert result.count(" [") < spaced_text.count(" [")
        assert result.count(" ]") < spaced_text.count(" ]")

    @pytest.mark.pdf_processor
    def test_preserve_scientific_notation_p_values(self, processor_with_recovery):
        """Test preservation of p-values in scientific notation."""
        scientific_text = """
        The results showed p < 0.001 and p-value = 0.05.
        Statistical significance was p = 0.0001.
        """
        
        result = processor_with_recovery._preserve_scientific_notation(scientific_text)
        
        # P-values should have consistent formatting
        assert "p<0.001" in result or "p < 0.001" in result
        assert "p-value=0.05" in result or "p-value = 0.05" in result

    @pytest.mark.pdf_processor
    def test_preserve_scientific_notation_chemical_formulas(self, processor_with_recovery):
        """Test preservation of chemical formulas."""
        chemical_text = """
        The compound H 2 O and Ca Cl 2 were analyzed.
        Glucose C 6 H 12 O 6 was measured.
        """
        
        result = processor_with_recovery._preserve_scientific_notation(chemical_text)
        
        # Chemical formulas should be fixed - check that spacing is reduced
        assert "H2O" in result or "H 2 O" not in result
        assert "CaCl2" in result or "Ca Cl 2" not in result  
        assert "C6H12O6" in result or "C 6 H 12 O 6" not in result

    @pytest.mark.pdf_processor
    def test_preserve_scientific_notation_units_measurements(self, processor_with_recovery):
        """Test preservation of units and measurements."""
        measurement_text = """
        Temperature was 37 ° C and pH 7.4.
        Concentration was 10 mM and 5 μM.
        Molecular weight was 150 kDa.
        """
        
        result = processor_with_recovery._preserve_scientific_notation(measurement_text)
        
        # Units should be properly formatted
        assert "37°C" in result
        assert "pH 7.4" in result
        assert "10 mM" in result
        assert "5 μM" in result
        assert "150 kDa" in result

    @pytest.mark.pdf_processor
    def test_handle_biomedical_formatting_references(self, processor_with_recovery):
        """Test handling of biomedical references and citations."""
        ref_text = """
        Previous studies [ 1 , 2 - 4 ] showed that Smith et al. , 2020 found significant results.
        See Fig. 1 and Table 2 for details.
        As shown in Supplementary Figure S1.
        """
        
        result = processor_with_recovery._handle_biomedical_formatting(ref_text)
        
        # References should be cleaned up - check that excessive spacing is reduced
        assert "[ 1 , 2 - 4 ]" not in result  # Should be cleaned up
        assert "Smith et al., 2020" in result or "Smith et al. , 2020" not in result
        assert "Fig. 1" in result
        assert "Table 2" in result
        assert "Supplementary Figure" in result

    @pytest.mark.pdf_processor
    def test_clean_text_flow_paragraph_breaks(self, processor_with_recovery):
        """Test cleaning of text flow and paragraph breaks."""
        messy_text = """
        
        
        This is a paragraph.
        
        
        
        
        This is another paragraph.
        
        
        """
        
        result = processor_with_recovery._clean_text_flow(messy_text)
        
        # Should normalize paragraph breaks to maximum of 2 newlines
        assert "\n\n\n" not in result

    @pytest.mark.pdf_processor
    def test_normalize_biomedical_terms_abbreviations(self, processor_with_recovery):
        """Test normalization of biomedical abbreviations."""
        abbrev_text = """
        We used d n a sequencing and r n a analysis.
        The q p c r results showed significant differences.
        ELISA and western blot were performed.
        """
        
        result = processor_with_recovery._normalize_biomedical_terms(abbrev_text)
        
        # Abbreviations should be normalized
        assert "DNA" in result
        assert "RNA" in result
        assert "qPCR" in result
        assert "d n a" not in result
        assert "r n a" not in result

    @pytest.mark.pdf_processor
    def test_normalize_biomedical_terms_greek_letters(self, processor_with_recovery):
        """Test normalization of Greek letters in biomedical context."""
        greek_text = """
        The alpha - ketoglutarate and beta - oxidation pathways.
        Measured delta - 13C values and gamma - radiation.
        """
        
        result = processor_with_recovery._normalize_biomedical_terms(greek_text)
        
        # Greek letters should be normalized
        assert "α-ketoglutarate" in result
        assert "β-oxidation" in result
        assert "δ-13C" in result or "delta-13C" in result
        assert "γ-radiation" in result or "gamma-radiation" in result

    @pytest.mark.pdf_processor
    def test_preprocess_biomedical_text_complete_pipeline(self, processor_with_recovery):
        """Test complete biomedical text preprocessing pipeline."""
        complex_text = """
        
        Page 123
        
        Clinical Metabolomics Analysis
        
        Abstract: This study presents comprehensive metabolomic analysis using L C - M S / M S.
        We analyzed H 2 O extracts with p < 0.001 significance.
        Temperature was maintained at 37 ° C and pH 7.4.
        
        Results showed significant differences [ 1 - 3 ] in biomarker concentrations.
        The d n a and r n a samples were processed using q p c r.
        
        © 2023 Nature Publishing
        Downloaded from nature.com
        
        """
        
        result = processor_with_recovery._preprocess_biomedical_text(complex_text)
        
        # Should apply all preprocessing steps - check that processing occurred
        assert "Page 123" not in result  # Artifacts removed
        assert "© 2023 Nature Publishing" not in result  # Copyright removed
        assert "Downloaded from" not in result  # Download info removed
        # Check that some processing occurred for abbreviations and formulas
        assert "Clinical Metabolomics Analysis" in result  # Main content preserved
        assert len(result) > 0  # Should not be empty
        assert result.strip()  # Should not be just whitespace
        # Check that some scientific notation processing occurred
        assert "p <" in result or "p<" in result  # P-value formatting
        assert "37" in result and "°C" in result or "°" in result  # Temperature processing


# =====================================================================
# TESTS FOR TIMEOUT AND MEMORY MONITORING EDGE CASES
# =====================================================================

class TestTimeoutAndMemoryMonitoring:
    """Tests for timeout and memory monitoring edge cases."""

    @pytest.mark.pdf_processor
    def test_check_processing_timeout_not_started(self, processor_with_recovery):
        """Test timeout checking when processing hasn't started."""
        processor_with_recovery._processing_start_time = None
        
        # Should not raise exception when start time is None
        processor_with_recovery._check_processing_timeout()

    @pytest.mark.pdf_processor
    def test_check_processing_timeout_within_limit(self, processor_with_recovery):
        """Test timeout checking within time limit."""
        processor_with_recovery._processing_start_time = time.time() - 5  # 5 seconds ago
        processor_with_recovery.processing_timeout = 30  # 30 second timeout
        
        # Should not raise exception
        processor_with_recovery._check_processing_timeout()

    @pytest.mark.pdf_processor
    def test_check_processing_timeout_exceeded(self, processor_with_recovery):
        """Test timeout checking when time limit is exceeded."""
        processor_with_recovery._processing_start_time = time.time() - 35  # 35 seconds ago
        processor_with_recovery.processing_timeout = 30  # 30 second timeout
        
        with pytest.raises(PDFProcessingTimeoutError, match="PDF processing timed out"):
            processor_with_recovery._check_processing_timeout()

    @pytest.mark.pdf_processor
    @patch('fitz.open')
    @patch('time.time')
    def test_open_pdf_with_timeout_warning(self, mock_time, mock_fitz_open, processor_with_recovery, temp_pdf_file, caplog):
        """Test PDF opening with slow opening time warning."""
        # Mock slow PDF opening (>30 seconds)
        mock_time.side_effect = [100.0, 135.0]  # 35 seconds elapsed
        mock_doc = Mock()
        mock_fitz_open.return_value = mock_doc
        
        result = processor_with_recovery._open_pdf_with_timeout(temp_pdf_file)
        
        assert result == mock_doc
        assert "PDF opening took" in caplog.text
        assert "35.0 seconds" in caplog.text

    @pytest.mark.pdf_processor
    @patch('fitz.open')
    def test_open_pdf_with_timeout_fitz_error(self, mock_fitz_open, processor_with_recovery, temp_pdf_file):
        """Test PDF opening with fitz FileDataError."""
        mock_fitz_open.side_effect = fitz.FileDataError("Invalid PDF file")
        
        with pytest.raises(PDFValidationError, match="Invalid or corrupted PDF file"):
            processor_with_recovery._open_pdf_with_timeout(temp_pdf_file)

    @pytest.mark.pdf_processor
    @patch('fitz.open')
    def test_open_pdf_with_timeout_generic_error_with_timeout(self, mock_fitz_open, processor_with_recovery, temp_pdf_file):
        """Test PDF opening with generic error during timeout condition."""
        processor_with_recovery._processing_start_time = time.time() - 35  # Simulate timeout
        processor_with_recovery.processing_timeout = 30
        
        mock_fitz_open.side_effect = Exception("Generic error")
        
        with pytest.raises(PDFProcessingTimeoutError, match="PDF opening timed out"):
            processor_with_recovery._open_pdf_with_timeout(temp_pdf_file)

    @pytest.mark.pdf_processor
    @patch('fitz.open')
    def test_open_pdf_with_timeout_generic_error_no_timeout(self, mock_fitz_open, processor_with_recovery, temp_pdf_file):
        """Test PDF opening with generic error without timeout condition."""
        processor_with_recovery._processing_start_time = time.time() - 5  # Within timeout
        processor_with_recovery.processing_timeout = 30
        
        mock_fitz_open.side_effect = Exception("Generic error")
        
        with pytest.raises(PDFValidationError, match="Failed to open PDF"):
            processor_with_recovery._open_pdf_with_timeout(temp_pdf_file)


# =====================================================================
# TESTS FOR CONCURRENT PROCESSING AND BATCH SIZE ADJUSTMENT
# =====================================================================

class TestConcurrentProcessingAndBatchAdjustment:
    """Tests for concurrent processing scenarios and batch size adjustment."""

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_batch_memory_monitoring(self, processor_with_recovery, temp_pdf_file):
        """Test batch processing with memory monitoring."""
        pdf_batch = [temp_pdf_file]
        
        with patch.object(processor_with_recovery, '_get_memory_usage') as mock_memory, \
             patch.object(processor_with_recovery, '_cleanup_memory') as mock_cleanup, \
             patch.object(processor_with_recovery, 'extract_text_from_pdf') as mock_extract:
            
            # Mock high memory usage to trigger cleanup
            mock_memory.side_effect = [
                {'process_memory_mb': 256.0, 'system_memory_percent': 45.0},  # Initial
                {'process_memory_mb': 450.0},  # Before file processing (high)
                {'process_memory_mb': 300.0, 'system_memory_percent': 50.0}   # Final
            ]
            
            mock_extract.return_value = {
                'text': 'Test content',
                'metadata': {'pages': 1},
                'processing_info': {'total_characters': 12},
                'page_texts': ['Test content']
            }
            
            results = await processor_with_recovery._process_batch(
                pdf_batch, batch_num=1, progress_tracker=None, max_memory_mb=500
            )
            
            # Should trigger cleanup due to high memory usage (450 > 500 * 0.8)
            mock_cleanup.assert_called_once()
            assert len(results) == 1
            assert results[0][0] == 'Test content'

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, processor_with_recovery, temp_pdf_file):
        """Test batch processing with file processing errors."""
        pdf_batch = [temp_pdf_file, temp_pdf_file]  # Two identical files for testing
        
        with patch.object(processor_with_recovery, 'extract_text_from_pdf') as mock_extract:
            # First file succeeds, second file fails
            mock_extract.side_effect = [
                {
                    'text': 'Success content',
                    'metadata': {'pages': 1},
                    'processing_info': {'total_characters': 15},
                    'page_texts': ['Success content']
                },
                PDFValidationError("Validation failed")
            ]
            
            results = await processor_with_recovery._process_batch(
                pdf_batch, batch_num=1, progress_tracker=None, max_memory_mb=1000
            )
            
            # Should only return successful result
            assert len(results) == 1
            assert results[0][0] == 'Success content'

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio 
    async def test_process_with_batch_mode_dynamic_sizing(self, processor_with_recovery, temp_pdf_file):
        """Test batch processing with dynamic batch size adjustment."""
        pdf_files = [temp_pdf_file] * 10  # 10 identical files for testing
        
        with patch.object(processor_with_recovery, '_process_batch') as mock_process_batch, \
             patch.object(processor_with_recovery, '_get_memory_usage') as mock_memory, \
             patch.object(processor_with_recovery, '_cleanup_memory') as mock_cleanup:
            
            # Mock memory usage to trigger batch size adjustment
            mock_memory.side_effect = [
                {'process_memory_mb': 200.0, 'system_memory_percent': 40.0},  # Initial low
                {'process_memory_mb': 800.0},  # High usage to trigger reduction
                {'process_memory_mb': 400.0, 'system_memory_percent': 50.0}   # After cleanup
            ]
            
            # Mock successful batch processing
            mock_process_batch.return_value = [('test', {'pages': 1})]
            mock_cleanup.return_value = {'memory_freed_mb': 100.0, 'system_memory_percent': 45.0}
            
            # Create a simple progress tracker mock
            progress_tracker = Mock()
            progress_tracker.start_batch_processing = Mock()
            progress_tracker.finish_batch_processing = Mock()
            progress_tracker.finish_batch_processing.return_value = Mock(
                completed_files=10, failed_files=0, skipped_files=0, total_files=10,
                total_characters=100, total_pages=10, processing_time=1.0,
                average_processing_time=0.1
            )
            
            results = await processor_with_recovery._process_with_batch_mode(
                pdf_files, initial_batch_size=5, max_memory_mb=1000, progress_tracker=progress_tracker
            )
            
            # Should have processed files in batches
            assert mock_process_batch.called
            assert mock_cleanup.called

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_sequential_mode(self, processor_with_recovery, temp_pdf_file):
        """Test sequential processing mode."""
        pdf_files = [temp_pdf_file]
        
        with patch.object(processor_with_recovery, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                'text': 'Sequential content',
                'metadata': {'pages': 1},
                'processing_info': {'total_characters': 18},
                'page_texts': ['Sequential content']
            }
            
            # Create progress tracker mock with proper context manager
            progress_tracker = Mock()
            context_manager = Mock()
            context_manager.__enter__ = Mock()
            context_manager.__exit__ = Mock(return_value=None)
            progress_tracker.track_file_processing.return_value = context_manager
            progress_tracker.record_file_success = Mock()
            
            results = await processor_with_recovery._process_sequential_mode(
                pdf_files, progress_tracker
            )
            
            assert len(results) == 1
            assert results[0][0] == 'Sequential content'
            progress_tracker.record_file_success.assert_called_once()

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_all_pdfs_batch_mode_enabled(self, processor_with_recovery, tmp_path):
        """Test process_all_pdfs with batch processing enabled."""
        # Create test directory with PDF files
        test_dir = tmp_path / "test_pdfs"
        test_dir.mkdir()
        
        # Create mock PDF files
        for i in range(3):
            pdf_file = test_dir / f"test_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        
        with patch.object(processor_with_recovery, '_process_with_batch_mode') as mock_batch_mode:
            mock_batch_mode.return_value = [('test', {'pages': 1})] * 3
            
            results = await processor_with_recovery.process_all_pdfs(
                papers_dir=str(test_dir),
                batch_size=2,
                max_memory_mb=1000,
                enable_batch_processing=True
            )
            
            mock_batch_mode.assert_called_once()
            assert len(results) == 3

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_all_pdfs_sequential_mode_enabled(self, processor_with_recovery, tmp_path):
        """Test process_all_pdfs with sequential processing."""
        # Create test directory with PDF files
        test_dir = tmp_path / "test_pdfs"
        test_dir.mkdir()
        
        pdf_file = test_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        
        with patch.object(processor_with_recovery, '_process_sequential_mode') as mock_sequential:
            mock_sequential.return_value = [('test', {'pages': 1})]
            
            results = await processor_with_recovery.process_all_pdfs(
                papers_dir=str(test_dir),
                enable_batch_processing=False
            )
            
            mock_sequential.assert_called_once()
            assert len(results) == 1


# =====================================================================
# TESTS FOR ERROR RECOVERY STATISTICS AND LOGGING
# =====================================================================

class TestErrorRecoveryStatisticsAndLogging:
    """Tests for error recovery statistics and logging functionality."""

    @pytest.mark.pdf_processor
    def test_get_error_recovery_stats_empty(self, processor_with_recovery):
        """Test error recovery stats when no errors occurred."""
        stats = processor_with_recovery.get_error_recovery_stats()
        
        assert stats['files_with_retries'] == 0
        assert stats['total_recovery_actions'] == 0
        assert stats['recovery_actions_by_type'] == {}
        assert stats['retry_details_by_file'] == {}
        assert 'error_recovery_config' in stats
        
        config = stats['error_recovery_config']
        assert config['max_retries'] == 5  # From processor_with_recovery fixture
        assert config['memory_recovery_enabled'] is True

    @pytest.mark.pdf_processor
    def test_get_error_recovery_stats_with_retries(self, processor_with_recovery):
        """Test error recovery stats after retry attempts."""
        # Simulate retry statistics
        processor_with_recovery._retry_stats['test_file.pdf'] = {
            'total_attempts': 3,
            'recoverable_attempts': 2,
            'recovery_actions': [
                {'attempt': 2, 'strategy': 'memory_cleanup', 'category': 'memory'}
            ],
            'error_history': [
                {'attempt': 1, 'error_type': 'MemoryError', 'error_message': 'Out of memory'}
            ]
        }
        
        processor_with_recovery._recovery_actions_attempted = {
            'memory_memory_cleanup': 2,
            'timeout_timeout_retry': 1
        }
        
        stats = processor_with_recovery.get_error_recovery_stats()
        
        assert stats['files_with_retries'] == 1
        assert stats['total_recovery_actions'] == 3
        assert stats['recovery_actions_by_type']['memory_memory_cleanup'] == 2
        assert 'test_file.pdf' in stats['retry_details_by_file']

    @pytest.mark.pdf_processor
    def test_reset_error_recovery_stats(self, processor_with_recovery):
        """Test resetting error recovery statistics."""
        # Add some stats first
        processor_with_recovery._retry_stats['test.pdf'] = {'attempts': 1}
        processor_with_recovery._recovery_actions_attempted['test_action'] = 1
        
        processor_with_recovery.reset_error_recovery_stats()
        
        assert processor_with_recovery._retry_stats == {}
        assert processor_with_recovery._recovery_actions_attempted == {}

    @pytest.mark.pdf_processor
    def test_get_enhanced_error_info_with_retry_details(self, processor_with_recovery, temp_pdf_file):
        """Test enhanced error info with retry details."""
        # Add retry information
        file_key = str(temp_pdf_file)
        processor_with_recovery._retry_stats[file_key] = {
            'total_attempts': 3,
            'recovery_actions': [
                {'strategy': 'memory_cleanup'},
                {'strategy': 'timeout_retry'}
            ]
        }
        
        error = ValueError("Test error message")
        error_info = processor_with_recovery._get_enhanced_error_info(temp_pdf_file, error)
        
        assert "ValueError: Test error message" in error_info
        assert "Attempts: 3" in error_info
        assert "Recovery strategies used:" in error_info
        assert "memory_cleanup" in error_info
        assert "timeout_retry" in error_info

    @pytest.mark.pdf_processor
    def test_get_enhanced_error_info_without_retry_details(self, processor_with_recovery, temp_pdf_file):
        """Test enhanced error info without retry details."""
        error = ValueError("Test error message")
        error_info = processor_with_recovery._get_enhanced_error_info(temp_pdf_file, error)
        
        assert error_info == "ValueError: Test error message"

    @pytest.mark.pdf_processor
    def test_log_error_recovery_summary_no_retries(self, processor_with_recovery, caplog):
        """Test logging recovery summary with no retries."""
        processor_with_recovery._log_error_recovery_summary()
        
        # Should not log anything when no retries occurred
        assert "Error recovery summary" not in caplog.text

    @pytest.mark.pdf_processor
    def test_log_error_recovery_summary_with_retries(self, processor_with_recovery, caplog):
        """Test logging recovery summary with retry information."""
        # Clear any existing log records
        caplog.clear()
        
        # Add retry statistics
        processor_with_recovery._retry_stats = {
            'file1.pdf': {'total_attempts': 3, 'recoverable_attempts': 2},
            'file2.pdf': {'total_attempts': 2, 'recoverable_attempts': 1}
        }
        processor_with_recovery._recovery_actions_attempted = {
            'memory_cleanup': 3,
            'timeout_retry': 1
        }
        
        processor_with_recovery._log_error_recovery_summary()
        
        # Check that recovery summary is logged
        assert "Error recovery summary" in caplog.text
        assert "files required retries" in caplog.text
        assert "total recovery actions" in caplog.text

    @pytest.mark.pdf_processor
    def test_log_error_recovery_summary_problematic_files(self, processor_with_recovery, caplog):
        """Test logging of most problematic files."""
        # Add retry statistics with varying attempt counts
        processor_with_recovery._retry_stats = {
            '/path/to/problematic1.pdf': {'total_attempts': 5, 'recoverable_attempts': 4},
            '/path/to/problematic2.pdf': {'total_attempts': 3, 'recoverable_attempts': 2},
            '/path/to/easy.pdf': {'total_attempts': 1, 'recoverable_attempts': 0}
        }
        
        processor_with_recovery._log_error_recovery_summary()
        
        # Should log most problematic files first
        assert "Problematic file: problematic1.pdf required 5 attempts" in caplog.text
        assert "Problematic file: problematic2.pdf required 3 attempts" in caplog.text


# =====================================================================
# TESTS FOR PERFORMANCE MONITORING AND RESOURCE UTILIZATION  
# =====================================================================

class TestPerformanceMonitoringAndResourceUtilization:
    """Tests for performance monitoring and resource utilization functionality."""

    @pytest.mark.pdf_processor
    @patch('gc.get_count')
    @patch('gc.isenabled')
    @patch('gc.get_threshold') 
    def test_get_processing_stats_complete(self, mock_threshold, mock_enabled, mock_count, processor_with_recovery, mock_memory_stats):
        """Test comprehensive processing statistics collection."""
        # Mock garbage collection info
        mock_count.return_value = [100, 50, 25]
        mock_enabled.return_value = True
        mock_threshold.return_value = (700, 10, 10)
        
        with patch.object(processor_with_recovery, '_get_memory_usage', return_value=mock_memory_stats):
            # Add some error recovery stats
            processor_with_recovery._retry_stats['test.pdf'] = {'attempts': 2}
            processor_with_recovery._recovery_actions_attempted['memory_cleanup'] = 1
            
            stats = processor_with_recovery.get_processing_stats()
            
            # Check basic configuration
            assert stats['processing_timeout'] == 30
            assert stats['memory_limit_mb'] == 512
            assert stats['max_page_text_size'] == 500000
            assert stats['memory_monitor_active'] is False
            
            # Check memory statistics
            memory_stats = stats['memory_stats']
            assert memory_stats['process_memory_mb'] == 256.5
            assert memory_stats['system_memory_percent'] == 45.2
            assert memory_stats['system_memory_total_mb'] == 8192.0
            
            # Check memory management features
            memory_mgmt = stats['memory_management']
            assert memory_mgmt['batch_processing_available'] is True
            assert memory_mgmt['dynamic_batch_sizing'] is True
            assert memory_mgmt['enhanced_garbage_collection'] is True
            
            # Check error recovery info
            error_recovery = stats['error_recovery']
            assert error_recovery['files_with_retries'] == 1
            assert error_recovery['total_recovery_actions'] == 1
            assert error_recovery['error_recovery_enabled'] is True
            
            # Check garbage collection info
            gc_stats = stats['garbage_collection']
            assert gc_stats['collections_count'] == [100, 50, 25]
            assert gc_stats['gc_enabled'] is True
            assert gc_stats['gc_thresholds'] == (700, 10, 10)

    @pytest.mark.pdf_processor
    @patch('psutil.Process')
    def test_get_memory_usage_without_peak_wset(self, mock_process, processor_with_recovery):
        """Test memory usage collection on systems without peak_wset attribute."""
        # Mock process without peak_wset attribute (non-Windows systems)
        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB
        
        # Mock getattr to return rss when peak_wset is not available
        def mock_getattr(obj, name, default):
            if name == 'peak_wset':
                return default  # Return the default (rss value)
            return getattr(obj, name, default)
        
        with patch('builtins.getattr', side_effect=mock_getattr):
            mock_process_instance.memory_info.return_value = mock_memory_info
            mock_process.return_value = mock_process_instance
            
            with patch('psutil.virtual_memory') as mock_virtual_memory:
                mock_vm = Mock()
                mock_vm.percent = 45.2
                mock_vm.available = 4096 * 1024 * 1024
                mock_vm.used = 2048 * 1024 * 1024
                mock_vm.total = 8192 * 1024 * 1024
                mock_virtual_memory.return_value = mock_vm
                
                memory_stats = processor_with_recovery._get_memory_usage()
                
                # Should use rss as fallback for peak memory
                assert memory_stats['process_memory_mb'] == 256.0
                assert memory_stats['process_memory_peak_mb'] == 256.0  # Falls back to rss


# =====================================================================
# TESTS FOR INTEGRATION AND COMPLEX PROCESSING SCENARIOS
# =====================================================================

class TestIntegrationAndComplexScenarios:
    """Integration tests for complex processing scenarios."""

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_complete_error_recovery_workflow(self, processor_with_recovery, temp_pdf_file):
        """Test complete error recovery workflow from classification to recovery."""
        with patch.object(processor_with_recovery, '_extract_text_internal') as mock_extract:
            # First attempt fails with memory error, second succeeds
            mock_extract.side_effect = [
                PDFMemoryError("Out of memory during processing"),
                {
                    'text': 'Recovered content',
                    'metadata': {'pages': 1, 'filename': temp_pdf_file.name},
                    'page_texts': ['Recovered content'],
                    'processing_info': {'total_characters': 17, 'pages_processed': 1}
                }
            ]
            
            with patch.object(processor_with_recovery, '_attempt_memory_recovery', return_value=True):
                result = processor_with_recovery.extract_text_from_pdf(temp_pdf_file)
                
                # Should succeed after recovery
                assert result['text'] == 'Recovered content'
                assert 'retry_info' in result['processing_info']
                
                retry_info = result['processing_info']['retry_info']
                assert retry_info['total_attempts'] == 2
                assert retry_info['recoverable_attempts'] == 1
                assert len(retry_info['recovery_actions']) == 1
                assert retry_info['recovery_actions'][0]['strategy'] == 'memory_cleanup'

    @pytest.mark.pdf_processor
    def test_pdf_validation_comprehensive_workflow(self, processor_with_recovery, temp_pdf_file):
        """Test comprehensive PDF validation workflow."""
        with patch('fitz.open') as mock_open:
            mock_doc = Mock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 5
            mock_doc.metadata = {
                'title': 'Test Document',
                'author': 'Test Author',
                'creator': 'Test Creator'
            }
            mock_open.return_value = mock_doc
            
            # Mock page loading for validation
            mock_page = Mock()
            mock_page.get_text.return_value = "Sample text content"
            mock_doc.load_page.return_value = mock_page
            
            result = processor_with_recovery.validate_pdf(temp_pdf_file)
            
            assert result['valid'] is True
            assert result['error'] is None
            assert result['pages'] == 5
            assert result['encrypted'] is False
            assert result['metadata']['title'] == 'Test Document'
            assert result['metadata']['author'] == 'Test Author'

    @pytest.mark.pdf_processor
    def test_pdf_validation_encrypted_pdf(self, processor_with_recovery, temp_pdf_file):
        """Test PDF validation with encrypted document."""
        with patch('fitz.open') as mock_open:
            mock_doc = Mock()
            mock_doc.needs_pass = True
            mock_open.return_value = mock_doc
            
            result = processor_with_recovery.validate_pdf(temp_pdf_file)
            
            assert result['valid'] is False
            assert result['encrypted'] is True
            assert result['error'] == "PDF is password protected"

    @pytest.mark.pdf_processor
    def test_get_page_count_comprehensive(self, processor_with_recovery, temp_pdf_file):
        """Test comprehensive page count functionality."""
        with patch('fitz.open') as mock_open:
            mock_doc = Mock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 42
            mock_open.return_value = mock_doc
            
            page_count = processor_with_recovery.get_page_count(temp_pdf_file)
            
            assert page_count == 42
            mock_doc.close.assert_called_once()

    @pytest.mark.pdf_processor
    def test_get_page_count_encrypted_pdf(self, processor_with_recovery, temp_pdf_file):
        """Test page count with encrypted PDF."""
        with patch('fitz.open') as mock_open:
            mock_doc = Mock()
            mock_doc.needs_pass = True
            mock_open.return_value = mock_doc
            
            with pytest.raises(PDFFileAccessError, match="password protected"):
                processor_with_recovery.get_page_count(temp_pdf_file)
            
            mock_doc.close.assert_called_once()

    @pytest.mark.pdf_processor 
    @patch('mimetypes.guess_type')
    def test_validate_pdf_file_mime_type_fallback(self, mock_guess_type, processor_with_recovery, temp_pdf_file):
        """Test PDF validation with MIME type fallback to header check."""
        # Mock incorrect MIME type detection
        mock_guess_type.return_value = ('text/plain', None)
        
        # The temp_pdf_file fixture creates a file with proper PDF header
        # So validation should pass despite incorrect MIME type
        processor_with_recovery._validate_pdf_file(temp_pdf_file)
        # Should not raise exception

    @pytest.mark.pdf_processor
    def test_validate_pdf_file_corrupted_header(self, processor_with_recovery, corrupted_pdf_file):
        """Test PDF validation with corrupted header."""
        with pytest.raises(PDFValidationError, match="invalid header"):
            processor_with_recovery._validate_pdf_file(corrupted_pdf_file)

    @pytest.mark.pdf_processor
    def test_validate_pdf_file_file_access_permission_error(self, processor_with_recovery):
        """Test PDF validation with permission errors."""
        # Create a path that will cause permission error
        restricted_path = Path("/root/restricted.pdf") if os.name != 'nt' else Path("C:/System Volume Information/restricted.pdf")
        
        # Mock the file to exist but be inaccessible
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'stat') as mock_stat, \
             patch('builtins.open', side_effect=PermissionError("Permission denied")):
            
            mock_stat.return_value = Mock(st_size=1000)
            
            with pytest.raises(PDFFileAccessError, match="Permission denied"):
                processor_with_recovery._validate_pdf_file(restricted_path)

    @pytest.mark.pdf_processor
    def test_validate_pdf_file_large_file_warning(self, processor_with_recovery, temp_pdf_file, caplog):
        """Test PDF validation with large file size warning."""
        # Clear any existing log records
        caplog.clear()
        
        with patch.object(temp_pdf_file, 'stat') as mock_stat:
            # Mock large file size (>100MB)
            mock_stat.return_value = Mock(st_size=150 * 1024 * 1024)
            
            processor_with_recovery._validate_pdf_file(temp_pdf_file)
            
            assert "Large PDF file detected" in caplog.text


# =====================================================================
# TESTS FOR EDGE CASES AND BOUNDARY CONDITIONS
# =====================================================================

class TestEdgeCasesAndBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.pdf_processor
    def test_parse_pdf_date_edge_cases(self, processor_with_recovery):
        """Test PDF date parsing edge cases."""
        # Empty string
        assert processor_with_recovery._parse_pdf_date("") is None
        
        # None input
        assert processor_with_recovery._parse_pdf_date(None) is None
        
        # String without D: prefix
        result = processor_with_recovery._parse_pdf_date("20231215143000")
        assert result == "2023-12-15T14:30:00"
        
        # String too short
        assert processor_with_recovery._parse_pdf_date("2023") is None
        
        # Invalid date values
        assert processor_with_recovery._parse_pdf_date("20231315143000") is None  # Invalid month
        assert processor_with_recovery._parse_pdf_date("20231232143000") is None  # Invalid day
        
        # Edge of valid date ranges
        result = processor_with_recovery._parse_pdf_date("20000101000000")  # Y2K
        assert result == "2000-01-01T00:00:00"

    @pytest.mark.pdf_processor
    def test_extract_metadata_edge_cases(self, processor_with_recovery, temp_pdf_file):
        """Test metadata extraction edge cases."""
        with patch('fitz.open') as mock_open:
            mock_doc = Mock()
            mock_doc.page_count = 0  # Edge case: PDF with no pages
            
            # Mock metadata with None and empty values
            mock_doc.metadata = {
                'title': None,  # None value
                'author': '',   # Empty string
                'subject': '   ',  # Whitespace only
                'creator': 'Valid Creator',  # Valid value
                'producer': None,
                'keywords': 'keyword1, keyword2',  # Valid keywords
                'creationDate': '',  # Empty creation date
                'modDate': 'invalid-date'  # Invalid modification date
            }
            
            metadata = processor_with_recovery._extract_metadata(mock_doc, temp_pdf_file)
            
            # Should only include non-empty, valid values
            assert 'title' not in metadata
            assert 'author' not in metadata
            assert 'subject' not in metadata
            assert metadata['creator'] == 'Valid Creator'
            assert 'producer' not in metadata
            assert metadata['keywords'] == 'keyword1, keyword2'
            assert 'creation_date' not in metadata
            assert 'modification_date' not in metadata
            
            # Basic file info should always be present
            assert metadata['filename'] == temp_pdf_file.name
            assert metadata['pages'] == 0
            assert 'file_size_bytes' in metadata

    @pytest.mark.pdf_processor
    def test_validate_and_clean_page_text_extreme_sizes(self, processor_with_recovery):
        """Test page text validation with extreme sizes."""
        # Test with exactly at limit
        processor_with_recovery.max_page_text_size = 100
        text_at_limit = "x" * 100
        
        result = processor_with_recovery._validate_and_clean_page_text(text_at_limit, 1)
        assert len(result) == 100
        assert "TRUNCATED" not in result
        
        # Test with one character over limit
        text_over_limit = "x" * 101
        result = processor_with_recovery._validate_and_clean_page_text(text_over_limit, 1)
        assert len(result) <= 100 + len("\n[TEXT TRUNCATED DUE TO SIZE LIMIT]")
        assert "TRUNCATED" in result

    @pytest.mark.pdf_processor
    def test_text_preprocessing_empty_and_whitespace(self, processor_with_recovery):
        """Test text preprocessing with empty and whitespace-only content."""
        # Empty text
        assert processor_with_recovery._preprocess_biomedical_text("") == ""
        
        # Whitespace only
        result = processor_with_recovery._preprocess_biomedical_text("   \n\t  \n  ")
        assert result == ""
        
        # Single character
        result = processor_with_recovery._preprocess_biomedical_text("a")
        assert result == "a"

    @pytest.mark.pdf_processor
    def test_error_recovery_boundary_conditions(self, processor_with_recovery):
        """Test error recovery with boundary conditions."""
        # Test with zero retries
        config = ErrorRecoveryConfig(max_retries=0)
        processor = BiomedicalPDFProcessor(error_recovery_config=config)
        
        # Should not attempt any retries
        assert processor.error_recovery.max_retries == 0

    @pytest.mark.pdf_processor
    def test_memory_cleanup_exception_handling(self, processor_with_recovery):
        """Test memory cleanup with exception handling."""
        with patch('gc.collect', side_effect=Exception("GC failed")), \
             patch('time.sleep') as mock_sleep, \
             patch.object(processor_with_recovery, '_get_memory_usage') as mock_memory:
            
            mock_memory.side_effect = [
                {'process_memory_mb': 256.0},
                {'process_memory_mb': 256.0, 'system_memory_percent': 40.0}
            ]
            
            # Should handle GC exceptions gracefully and continue processing
            result = processor_with_recovery._cleanup_memory()
            
            # Should still return memory statistics even when GC fails
            assert 'memory_before_mb' in result
            assert 'memory_after_mb' in result
            assert result['memory_before_mb'] == 256.0
            assert result['memory_after_mb'] == 256.0

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_all_pdfs_no_pdf_files(self, processor_with_recovery, tmp_path):
        """Test process_all_pdfs with directory containing no PDF files."""
        test_dir = tmp_path / "empty_pdfs"
        test_dir.mkdir()
        
        # Create non-PDF files
        (test_dir / "document.txt").write_text("Not a PDF")
        (test_dir / "image.png").write_bytes(b"fake png data")
        
        results = await processor_with_recovery.process_all_pdfs(str(test_dir))
        
        assert results == []

    @pytest.mark.pdf_processor
    @pytest.mark.asyncio
    async def test_process_all_pdfs_nonexistent_directory(self, processor_with_recovery):
        """Test process_all_pdfs with non-existent directory."""
        nonexistent_dir = "/path/that/does/not/exist"
        
        results = await processor_with_recovery.process_all_pdfs(nonexistent_dir)
        
        assert results == []

    @pytest.mark.pdf_processor
    def test_processor_initialization_edge_cases(self):
        """Test processor initialization with edge cases."""
        # Test with zero/negative values
        processor = BiomedicalPDFProcessor(
            processing_timeout=0,
            memory_limit_mb=0,
            max_page_text_size=1
        )
        
        assert processor.processing_timeout == 0
        assert processor.memory_limit_mb == 0
        assert processor.max_page_text_size == 1

    @pytest.mark.pdf_processor
    def test_file_not_found_edge_cases(self, processor_with_recovery):
        """Test file not found handling in various methods."""
        nonexistent_file = Path("/path/that/does/not/exist.pdf")
        
        # extract_text_from_pdf
        with pytest.raises(FileNotFoundError):
            processor_with_recovery.extract_text_from_pdf(nonexistent_file)
        
        # validate_pdf
        with pytest.raises(FileNotFoundError):
            processor_with_recovery.validate_pdf(nonexistent_file)
        
        # get_page_count
        with pytest.raises(FileNotFoundError):
            processor_with_recovery.get_page_count(nonexistent_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])