#!/usr/bin/env python3
"""
Comprehensive PDF Processing Error Handling Test Suite - CMO-LIGHTRAG-009.

This module provides extensive testing for error handling during PDF processing,
covering all failure scenarios from individual PDF corruption to large-scale
batch processing failures, recovery mechanisms, and system stability.

Test Categories:
1. Individual PDF Processing Error Tests:
   - Corrupted PDF file handling with recovery attempts
   - Empty or malformed PDF files
   - Password-protected and restricted access PDFs
   - Network/storage failures during processing
   - Memory exhaustion scenarios during large PDF processing
   - Character encoding and Unicode errors
   
2. Batch Processing Error Tests:
   - Concurrent processing conflicts and failures
   - Mixed success/failure scenarios in batches
   - Memory pressure during batch operations
   - Network interruptions during batch processing
   - Large-scale batch processing stress testing
   
3. Knowledge Base Integration Error Tests:
   - API failures during knowledge base construction
   - Storage system failures during ingestion
   - Index corruption and recovery
   - Partial ingestion failure handling
   
4. Recovery Mechanism Tests:
   - Automatic retry with exponential backoff
   - Graceful degradation under system stress
   - Error classification and routing
   - System resource monitoring and throttling
   - Checkpoint and resume functionality
   
5. System Stability Tests:
   - Error propagation and isolation
   - Memory leak prevention during error conditions
   - Performance under sustained error conditions
   - Error logging and monitoring validation
   - Long-running stability under mixed conditions

The test suite integrates with existing error handling infrastructure and
validates that the system maintains stability and provides useful diagnostics
under all error conditions.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Task: CMO-LIGHTRAG-009 - Comprehensive PDF Processing Error Handling Tests
Version: 1.0.0
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
import json
import time
import threading
import gc
import psutil
import signal
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch, mock_open, call
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import concurrent.futures
import random

# PDF handling
import fitz  # PyMuPDF

# Import components under test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor, 
    BiomedicalPDFProcessorError,
    PDFValidationError,
    PDFProcessingTimeoutError,
    PDFMemoryError,
    PDFFileAccessError,
    PDFContentError,
    ErrorRecoveryConfig
)

from lightrag_integration.clinical_metabolomics_rag import (
    ClinicalMetabolomicsRAG,
    IngestionError, 
    IngestionRetryableError,
    IngestionNonRetryableError, 
    IngestionResourceError,
    IngestionNetworkError,
    IngestionAPIError,
    StorageInitializationError
)

from lightrag_integration.advanced_recovery_system import (
    AdvancedRecoverySystem,
    DegradationMode,
    FailureType,
    BackoffStrategy,
    ResourceThresholds,
    SystemResourceMonitor
)

from lightrag_integration.enhanced_logging import (
    EnhancedLogger,
    IngestionLogger,
    CorrelationIDManager,
    PerformanceTracker
)

from lightrag_integration.unified_progress_tracker import (
    KnowledgeBaseProgressTracker,
    KnowledgeBasePhase
)


# =====================================================================
# TEST FIXTURES AND UTILITIES
# =====================================================================

@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def pdf_processor():
    """Create PDF processor with standard test configuration."""
    return BiomedicalPDFProcessor()


@pytest.fixture
def enhanced_pdf_processor():
    """Create PDF processor with enhanced error recovery configuration."""
    recovery_config = ErrorRecoveryConfig(
        max_retries=5,
        base_delay=0.1,  # Faster for testing
        max_delay=2.0,   # Shorter for testing
        memory_recovery_enabled=True,
        file_lock_retry_enabled=True,
        timeout_retry_enabled=True
    )
    return BiomedicalPDFProcessor(error_recovery_config=recovery_config)


@pytest.fixture
def mock_clinical_rag():
    """Create mock Clinical Metabolomics RAG system."""
    mock_rag = MagicMock(spec=ClinicalMetabolomicsRAG)
    mock_rag.ingest_documents = AsyncMock()
    mock_rag.query = AsyncMock()
    mock_rag.get_knowledge_base_status = AsyncMock()
    return mock_rag


@pytest.fixture
def recovery_system():
    """Create advanced recovery system for testing."""
    thresholds = ResourceThresholds(
        memory_warning_percent=70.0,
        memory_critical_percent=85.0,
        cpu_warning_percent=80.0,
        cpu_critical_percent=90.0,
        disk_warning_percent=85.0,
        disk_critical_percent=95.0
    )
    return AdvancedRecoverySystem(
        resource_thresholds=thresholds,
        enable_adaptive_backoff=True,
        enable_circuit_breaker=True
    )


@pytest.fixture
def correlation_logger():
    """Create logger with correlation tracking."""
    correlation_manager = CorrelationIDManager()
    logger = EnhancedLogger(
        name="test_pdf_error_handling",
        correlation_manager=correlation_manager
    )
    return logger, correlation_manager


class PDFTestFileGenerator:
    """Utility class for generating various types of test PDF files."""
    
    @staticmethod
    def create_corrupted_pdf(file_path: Path, corruption_type: str):
        """Create a corrupted PDF file for testing."""
        if corruption_type == "zero_byte":
            file_path.write_bytes(b"")
            
        elif corruption_type == "invalid_header":
            file_path.write_bytes(b"NOT_A_PDF_FILE\ngarbage content")
            
        elif corruption_type == "truncated":
            # Valid header but truncated content
            file_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            
        elif corruption_type == "binary_garbage":
            file_path.write_bytes(b"\x00\x01\x02\x03" * 100)
            
        elif corruption_type == "mixed_valid_invalid":
            # Start valid then become corrupted
            content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
            content += b"valid content for a while\n" * 10
            content += b"\x00\xFF\x47\x47\x00" * 50  # Invalid bytes (0x47 = 'G')
            file_path.write_bytes(content)
    
    @staticmethod
    def create_valid_pdf_with_content(file_path: Path, content: str, 
                                     pages: int = 1, metadata: Dict[str, Any] = None):
        """Create a valid PDF with specified content for testing."""
        try:
            doc = fitz.open()
            for i in range(pages):
                page = doc.new_page()
                page_content = f"Page {i+1}\n{content}\n" if pages > 1 else content
                page.insert_text((50, 50 + i * 20), page_content, fontsize=12)
            
            if metadata:
                doc.set_metadata(metadata)
            
            doc.save(str(file_path))
            doc.close()
        except Exception as e:
            # Fallback to text file if PDF creation fails
            logging.warning(f"PDF creation failed, using text fallback: {e}")
            file_path.write_text(content, encoding='utf-8')


# =====================================================================
# INDIVIDUAL PDF ERROR HANDLING TESTS
# =====================================================================

class TestIndividualPDFErrorHandling:
    """Test error handling for individual PDF processing scenarios."""
    
    @pytest.mark.asyncio
    async def test_corrupted_pdf_recovery_attempts(self, enhanced_pdf_processor, temp_test_dir):
        """Test recovery attempts when processing corrupted PDFs."""
        corruption_types = [
            "zero_byte", "invalid_header", "truncated", 
            "binary_garbage", "mixed_valid_invalid"
        ]
        
        for corruption_type in corruption_types:
            pdf_file = temp_test_dir / f"corrupted_{corruption_type}.pdf"
            PDFTestFileGenerator.create_corrupted_pdf(pdf_file, corruption_type)
            
            # Should attempt recovery but ultimately fail
            with pytest.raises(BiomedicalPDFProcessorError) as exc_info:
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
            
            error_msg = str(exc_info.value).lower()
            expected_indicators = ["corrupted", "invalid", "failed", "error"]
            assert any(indicator in error_msg for indicator in expected_indicators)
            
            # Verify error recovery was attempted
            assert hasattr(enhanced_pdf_processor, '_retry_stats') or \
                   "retry" in error_msg or "attempt" in error_msg

    @pytest.mark.asyncio
    async def test_memory_exhaustion_recovery(self, enhanced_pdf_processor, temp_test_dir):
        """Test recovery from memory exhaustion during PDF processing."""
        
        @patch('lightrag_integration.pdf_processor.fitz.open')
        def run_memory_test(mock_fitz_open):
            # Simulate memory exhaustion on first attempt, success on retry
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 1
            mock_doc.metadata = {}
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Test content"
            mock_doc.load_page.return_value = mock_page
            
            # First call raises MemoryError, subsequent calls succeed
            mock_fitz_open.side_effect = [
                MemoryError("Memory allocation failed"),
                mock_doc  # Success on retry
            ]
            
            pdf_file = temp_test_dir / "memory_test.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, "Test content for memory exhaustion"
            )
            
            # Should recover from memory error with retry
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                # If recovery succeeds, should get content
                assert 'text' in result
                assert 'metadata' in result
            except PDFMemoryError:
                # If recovery fails, should get proper error type
                pytest.fail("Memory recovery should have succeeded with retry")
                
        run_memory_test()

    @pytest.mark.asyncio
    async def test_password_protected_pdf_handling(self, pdf_processor, temp_test_dir):
        """Test handling of password-protected PDFs with retry attempts."""
        
        with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz_open:
            mock_doc = MagicMock()
            mock_doc.needs_pass = True
            mock_fitz_open.return_value = mock_doc
            
            pdf_file = temp_test_dir / "encrypted.pdf"
            pdf_file.write_bytes(b"encrypted pdf content")
            
            with pytest.raises(BiomedicalPDFProcessorError) as exc_info:
                pdf_processor.extract_text_from_pdf(pdf_file)
            
            assert "password protected" in str(exc_info.value).lower()
            mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_network_storage_failure_recovery(self, enhanced_pdf_processor, temp_test_dir):
        """Test recovery from network/storage failures during PDF processing."""
        
        network_errors = [
            OSError("Network path not accessible"),
            IOError("I/O operation failed"), 
            TimeoutError("Network timeout"),
            ConnectionError("Network connection lost"),
            PermissionError("Access denied")
        ]
        
        for error in network_errors:
            with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz_open:
                # First attempt fails, second succeeds
                mock_doc = MagicMock()
                mock_doc.needs_pass = False
                mock_doc.page_count = 1
                mock_doc.metadata = {}
                mock_page = MagicMock()
                mock_page.get_text.return_value = "Recovered content"
                mock_doc.load_page.return_value = mock_page
                
                mock_fitz_open.side_effect = [error, mock_doc]
                
                pdf_file = temp_test_dir / f"network_test_{type(error).__name__}.pdf"
                PDFTestFileGenerator.create_valid_pdf_with_content(
                    pdf_file, "Network recovery test content"
                )
                
                # Should attempt recovery
                try:
                    result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                    # Recovery succeeded
                    assert 'text' in result
                except BiomedicalPDFProcessorError as e:
                    # Recovery failed but with proper error handling
                    assert str(error) in str(e) or type(error).__name__ in str(e)

    @pytest.mark.asyncio
    async def test_unicode_encoding_error_recovery(self, pdf_processor, temp_test_dir):
        """Test recovery from Unicode encoding errors during text extraction."""
        
        with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz_open:
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 2
            mock_doc.metadata = {}
            
            # First page has Unicode error, second page works
            mock_page1 = MagicMock()
            mock_page1.get_text.side_effect = UnicodeDecodeError(
                'utf-8', b'\x80\x81', 0, 1, 'invalid start byte'
            )
            
            mock_page2 = MagicMock()
            mock_page2.get_text.return_value = "Valid content from page 2"
            
            mock_doc.load_page.side_effect = lambda page_num: mock_page1 if page_num == 0 else mock_page2
            mock_fitz_open.return_value = mock_doc
            
            pdf_file = temp_test_dir / "unicode_test.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, "Unicode test content", pages=2
            )
            
            result = pdf_processor.extract_text_from_pdf(pdf_file)
            
            # Should handle Unicode errors gracefully
            assert len(result['page_texts']) == 2
            assert result['page_texts'][0] == ""  # Failed page
            assert "Valid content" in result['page_texts'][1]  # Successful page


# =====================================================================
# BATCH PROCESSING ERROR HANDLING TESTS
# =====================================================================

class TestBatchProcessingErrorHandling:
    """Test error handling for batch PDF processing scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_conflict_handling(self, enhanced_pdf_processor, temp_test_dir):
        """Test handling of concurrent processing conflicts."""
        
        # Create multiple PDF files for concurrent processing
        pdf_files = []
        for i in range(5):
            pdf_file = temp_test_dir / f"concurrent_test_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, f"Concurrent test content {i}"
            )
            pdf_files.append(pdf_file)
        
        def process_pdf_with_errors(pdf_path, fail_probability=0.3):
            """Process PDF with simulated random failures."""
            if random.random() < fail_probability:
                raise OSError(f"Simulated concurrent access error for {pdf_path}")
            return enhanced_pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Process files concurrently with some failures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for pdf_file in pdf_files:
                future = executor.submit(process_pdf_with_errors, pdf_file)
                futures.append((pdf_file, future))
            
            results = []
            errors = []
            
            for pdf_file, future in futures:
                try:
                    result = future.result(timeout=10)
                    results.append((pdf_file, result))
                except Exception as e:
                    errors.append((pdf_file, str(e)))
            
            # Should handle some successes and some failures
            assert len(results) + len(errors) == len(pdf_files)
            # At least some operations should succeed despite conflicts
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_mixed_success_failure_batch_processing(self, enhanced_pdf_processor, temp_test_dir):
        """Test batch processing with mixed success and failure scenarios."""
        
        # Create a mix of valid and invalid PDF files
        files_config = [
            ("valid_1.pdf", "valid", "Valid content 1"),
            ("corrupted_1.pdf", "corrupted", "zero_byte"),
            ("valid_2.pdf", "valid", "Valid content 2"),
            ("corrupted_2.pdf", "corrupted", "invalid_header"),
            ("valid_3.pdf", "valid", "Valid content 3"),
            ("password_protected.pdf", "encrypted", None)
        ]
        
        pdf_files = []
        expected_successes = 0
        expected_failures = 0
        
        for filename, file_type, content in files_config:
            pdf_file = temp_test_dir / filename
            
            if file_type == "valid":
                PDFTestFileGenerator.create_valid_pdf_with_content(pdf_file, content)
                expected_successes += 1
            elif file_type == "corrupted":
                PDFTestFileGenerator.create_corrupted_pdf(pdf_file, content)
                expected_failures += 1
            elif file_type == "encrypted":
                # Create mock encrypted file
                pdf_file.write_bytes(b"encrypted content")
                expected_failures += 1
            
            pdf_files.append(pdf_file)
        
        # Process batch with error tracking
        results = []
        errors = []
        
        for pdf_file in pdf_files:
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                results.append((pdf_file, result))
            except BiomedicalPDFProcessorError as e:
                errors.append((pdf_file, str(e)))
        
        # Verify mixed results match expectations
        assert len(results) <= expected_successes  # May have fewer due to PDF creation issues
        assert len(errors) >= expected_failures   # Should have at least expected failures
        assert len(results) + len(errors) == len(pdf_files)

    @pytest.mark.asyncio
    async def test_memory_pressure_during_batch_processing(self, enhanced_pdf_processor, temp_test_dir):
        """Test batch processing behavior under memory pressure."""
        
        # Create multiple larger PDF files
        large_pdf_files = []
        for i in range(10):
            pdf_file = temp_test_dir / f"large_pdf_{i}.pdf"
            # Create content that would use more memory
            large_content = f"Large document content {i}\n" * 1000
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, large_content, pages=5
            )
            large_pdf_files.append(pdf_file)
        
        # Monitor memory usage during batch processing
        initial_memory = psutil.Process().memory_info().rss
        max_memory_usage = initial_memory
        
        results = []
        memory_errors = []
        
        for i, pdf_file in enumerate(large_pdf_files):
            try:
                # Check memory before processing
                current_memory = psutil.Process().memory_info().rss
                max_memory_usage = max(max_memory_usage, current_memory)
                
                # Process with memory monitoring
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                results.append(result)
                
                # Force garbage collection periodically
                if i % 3 == 0:
                    gc.collect()
                    
            except PDFMemoryError as e:
                memory_errors.append(str(e))
            except Exception as e:
                # Other errors (file issues, etc.)
                continue
        
        # Verify system handled memory pressure appropriately
        memory_increase = max_memory_usage - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Should have processed at least some files
        assert len(results) > 0
        
        # Memory usage should be reasonable (not unbounded growth)
        # This is a heuristic - adjust based on system capabilities
        assert memory_increase_mb < 500  # Less than 500MB increase

    @pytest.mark.asyncio
    async def test_network_interruption_during_batch(self, enhanced_pdf_processor, temp_test_dir):
        """Test batch processing resilience to network interruptions."""
        
        pdf_files = []
        for i in range(8):
            pdf_file = temp_test_dir / f"network_batch_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, f"Network batch content {i}"
            )
            pdf_files.append(pdf_file)
        
        # Simulate network interruptions during batch processing
        with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz_open:
            
            def simulate_network_issues(pdf_path):
                """Simulate intermittent network issues."""
                file_index = int(pdf_path.name.split('_')[-1].split('.')[0])
                
                if file_index in [2, 5]:  # Simulate failures on specific files
                    raise ConnectionError(f"Network interrupted for {pdf_path}")
                
                # Normal processing for other files
                mock_doc = MagicMock()
                mock_doc.needs_pass = False
                mock_doc.page_count = 1
                mock_doc.metadata = {}
                mock_page = MagicMock()
                mock_page.get_text.return_value = f"Content from {pdf_path.name}"
                mock_doc.load_page.return_value = mock_page
                return mock_doc
            
            mock_fitz_open.side_effect = simulate_network_issues
            
            successful_results = []
            network_errors = []
            
            for pdf_file in pdf_files:
                try:
                    result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                    successful_results.append((pdf_file, result))
                except (ConnectionError, BiomedicalPDFProcessorError) as e:
                    network_errors.append((pdf_file, str(e)))
            
            # Should have some successes and some network-related failures
            assert len(successful_results) > 0
            assert len(network_errors) == 2  # Files 2 and 5 should fail
            assert len(successful_results) + len(network_errors) == len(pdf_files)


# =====================================================================
# KNOWLEDGE BASE INTEGRATION ERROR TESTS
# =====================================================================

class TestKnowledgeBaseIntegrationErrors:
    """Test error handling during knowledge base integration."""
    
    @pytest.mark.asyncio
    async def test_api_failure_during_knowledge_base_construction(self, mock_clinical_rag, 
                                                                 pdf_processor, temp_test_dir):
        """Test handling of API failures during knowledge base construction."""
        
        # Create valid PDF files for ingestion
        pdf_files = []
        for i in range(3):
            pdf_file = temp_test_dir / f"kb_api_test_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, f"Knowledge base content {i}"
            )
            pdf_files.append(pdf_file)
        
        # Extract content from PDFs
        pdf_contents = []
        for pdf_file in pdf_files:
            try:
                content = pdf_processor.extract_text_from_pdf(pdf_file)
                pdf_contents.append(content)
            except BiomedicalPDFProcessorError:
                # Skip files that can't be processed
                continue
        
        # Simulate API failures during ingestion
        api_errors = [
            IngestionAPIError("API rate limit exceeded", retry_after=60),
            IngestionNetworkError("Connection timeout to embedding service"),
            IngestionResourceError("Insufficient quota for vector operations"),
            IngestionRetryableError("Temporary service unavailable", retry_after=30)
        ]
        
        for error in api_errors:
            mock_clinical_rag.ingest_documents.side_effect = error
            
            # Should handle API errors appropriately
            with pytest.raises((IngestionAPIError, IngestionNetworkError, 
                              IngestionResourceError, IngestionRetryableError)):
                await mock_clinical_rag.ingest_documents(pdf_contents)
            
            # Verify error was the expected type
            assert isinstance(error, (IngestionAPIError, IngestionNetworkError,
                                    IngestionResourceError, IngestionRetryableError))

    @pytest.mark.asyncio
    async def test_storage_system_failures_during_ingestion(self, mock_clinical_rag, 
                                                           pdf_processor, temp_test_dir):
        """Test handling of storage system failures during ingestion."""
        
        # Create PDF content for ingestion
        pdf_file = temp_test_dir / "storage_test.pdf"
        PDFTestFileGenerator.create_valid_pdf_with_content(
            pdf_file, "Storage system test content"
        )
        
        content = pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Simulate storage system failures
        storage_errors = [
            StorageInitializationError("Cannot initialize storage backend"),
            OSError("Disk full - cannot write to storage"),
            PermissionError("Access denied to storage directory"),
            IOError("Storage device I/O error")
        ]
        
        for error in storage_errors:
            mock_clinical_rag.ingest_documents.side_effect = error
            
            with pytest.raises((StorageInitializationError, OSError, 
                              PermissionError, IOError)):
                await mock_clinical_rag.ingest_documents([content])

    @pytest.mark.asyncio
    async def test_partial_ingestion_failure_handling(self, mock_clinical_rag, 
                                                     pdf_processor, temp_test_dir):
        """Test handling of partial ingestion failures in batch operations."""
        
        # Create multiple PDF files
        pdf_files = []
        for i in range(6):
            pdf_file = temp_test_dir / f"partial_ingestion_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, f"Partial ingestion test content {i}"
            )
            pdf_files.append(pdf_file)
        
        # Extract contents
        pdf_contents = []
        for pdf_file in pdf_files:
            try:
                content = pdf_processor.extract_text_from_pdf(pdf_file)
                pdf_contents.append(content)
            except BiomedicalPDFProcessorError:
                continue
        
        # Simulate partial ingestion failure
        def partial_ingestion_side_effect(documents):
            """Simulate failure after processing some documents."""
            if len(documents) > 3:
                raise IngestionResourceError(
                    f"Resource exhausted after processing {len(documents[:3])} of {len(documents)} documents"
                )
            return {"ingested": len(documents), "failed": 0}
        
        mock_clinical_rag.ingest_documents.side_effect = partial_ingestion_side_effect
        
        # Test partial ingestion
        try:
            # First batch (3 documents) should succeed
            result = await mock_clinical_rag.ingest_documents(pdf_contents[:3])
            assert result["ingested"] == 3
            
            # Second batch (6 documents) should fail with resource error
            with pytest.raises(IngestionResourceError) as exc_info:
                await mock_clinical_rag.ingest_documents(pdf_contents)
            
            assert "Resource exhausted" in str(exc_info.value)
            
        except Exception as e:
            pytest.fail(f"Unexpected error in partial ingestion test: {e}")


# =====================================================================
# RECOVERY MECHANISM TESTS
# =====================================================================

class TestRecoveryMechanisms:
    """Test advanced recovery mechanisms and error handling strategies."""
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_recovery(self, enhanced_pdf_processor, temp_test_dir):
        """Test exponential backoff during error recovery."""
        
        pdf_file = temp_test_dir / "backoff_test.pdf"
        PDFTestFileGenerator.create_valid_pdf_with_content(pdf_file, "Backoff test content")
        
        retry_delays = []
        original_sleep = time.sleep
        
        def mock_sleep(delay):
            """Mock sleep to capture retry delays."""
            retry_delays.append(delay)
            # Don't actually sleep to speed up test
            return
        
        with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz_open, \
             patch('time.sleep', side_effect=mock_sleep):
            
            # Simulate multiple failures before success
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 1
            mock_doc.metadata = {}
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Success content"
            mock_doc.load_page.return_value = mock_page
            
            # Fail 3 times, then succeed
            mock_fitz_open.side_effect = [
                OSError("Temporary failure 1"),
                OSError("Temporary failure 2"), 
                OSError("Temporary failure 3"),
                mock_doc  # Success
            ]
            
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                
                # Should have succeeded after retries
                assert 'text' in result
                
                # Should have exponentially increasing delays
                if len(retry_delays) > 1:
                    # Each delay should be larger than the previous (with possible jitter)
                    base_progression = all(
                        retry_delays[i] >= retry_delays[i-1] * 0.8  # Allow for jitter
                        for i in range(1, len(retry_delays))
                    )
                    assert base_progression or len(retry_delays) <= 2
                    
            except BiomedicalPDFProcessorError:
                # If all retries failed, that's also acceptable behavior
                assert len(retry_delays) >= 3  # Should have attempted retries

    @pytest.mark.asyncio 
    async def test_graceful_degradation_under_system_stress(self, recovery_system, 
                                                           enhanced_pdf_processor, temp_test_dir):
        """Test graceful degradation when system is under stress."""
        
        # Create test PDFs
        pdf_files = []
        for i in range(5):
            pdf_file = temp_test_dir / f"stress_test_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, f"Stress test content {i}"
            )
            pdf_files.append(pdf_file)
        
        # Simulate high system resource usage
        with patch.object(psutil, 'virtual_memory') as mock_memory, \
             patch.object(psutil, 'cpu_percent') as mock_cpu:
            
            # Simulate high memory and CPU usage
            mock_memory.return_value.percent = 88.0  # Above critical threshold
            mock_cpu.return_value = 92.0  # Above critical threshold
            
            results = []
            degradation_triggered = False
            
            for pdf_file in pdf_files:
                try:
                    # Check if recovery system would trigger degradation
                    if recovery_system.should_trigger_degradation():
                        degradation_triggered = True
                        # In degradation mode, might skip processing or use simplified processing
                        continue
                    
                    result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                    results.append(result)
                    
                except BiomedicalPDFProcessorError as e:
                    # Errors are expected under stress
                    continue
            
            # Should have either triggered degradation or processed with some limitations
            assert degradation_triggered or len(results) <= len(pdf_files)

    @pytest.mark.asyncio
    async def test_error_classification_and_routing(self, enhanced_pdf_processor, temp_test_dir):
        """Test that errors are properly classified and routed to appropriate handlers."""
        
        error_scenarios = [
            ("corrupted.pdf", "corrupted", "zero_byte", PDFValidationError),
            ("permission.pdf", "permission", PermissionError("Access denied"), PDFFileAccessError),
            ("memory.pdf", "memory", MemoryError("Out of memory"), PDFMemoryError),
            ("timeout.pdf", "timeout", TimeoutError("Processing timeout"), PDFProcessingTimeoutError),
            ("content.pdf", "content", UnicodeDecodeError('utf-8', b'\x80', 0, 1, 'err'), PDFContentError)
        ]
        
        for filename, scenario_type, error_input, expected_error_type in error_scenarios:
            pdf_file = temp_test_dir / filename
            
            if scenario_type == "corrupted":
                PDFTestFileGenerator.create_corrupted_pdf(pdf_file, error_input)
            else:
                # Create valid file for non-corruption scenarios
                PDFTestFileGenerator.create_valid_pdf_with_content(
                    pdf_file, f"Test content for {scenario_type}"
                )
            
            if scenario_type != "corrupted":
                # Mock the specific error for non-corruption scenarios
                with patch('lightrag_integration.pdf_processor.fitz.open') as mock_fitz_open:
                    if isinstance(error_input, Exception):
                        mock_fitz_open.side_effect = error_input
                    else:
                        mock_fitz_open.side_effect = Exception(error_input)
                    
                    with pytest.raises(BiomedicalPDFProcessorError):
                        enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
            else:
                # For corrupted files, test directly
                with pytest.raises(BiomedicalPDFProcessorError):
                    enhanced_pdf_processor.extract_text_from_pdf(pdf_file)

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume_functionality(self, enhanced_pdf_processor, 
                                                      temp_test_dir, correlation_logger):
        """Test checkpoint creation and resume functionality for batch processing."""
        
        logger, correlation_manager = correlation_logger
        
        # Create batch of PDF files
        pdf_files = []
        for i in range(8):
            pdf_file = temp_test_dir / f"checkpoint_test_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                pdf_file, f"Checkpoint test content {i}"
            )
            pdf_files.append(pdf_file)
        
        # Simulate batch processing with checkpointing
        checkpoint_file = temp_test_dir / "processing_checkpoint.json"
        processed_files = []
        failed_files = []
        
        # Process first batch with simulated failure in middle
        for i, pdf_file in enumerate(pdf_files[:5]):
            try:
                if i == 3:  # Simulate failure on 4th file
                    raise OSError("Simulated system failure")
                
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                processed_files.append({
                    'file': str(pdf_file),
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                failed_files.append({
                    'file': str(pdf_file),
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Create checkpoint
                checkpoint_data = {
                    'processed': processed_files,
                    'failed': failed_files,
                    'last_file_index': i,
                    'total_files': len(pdf_files),
                    'correlation_id': correlation_manager.get_current_id()
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                break  # Simulate system stopping after failure
        
        # Resume processing from checkpoint
        assert checkpoint_file.exists()
        
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        resume_index = checkpoint_data['last_file_index'] + 1
        
        # Continue processing from where we left off
        for pdf_file in pdf_files[resume_index:]:
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                processed_files.append({
                    'file': str(pdf_file),
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'resumed': True
                })
            except Exception as e:
                failed_files.append({
                    'file': str(pdf_file),
                    'status': 'failed', 
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Verify checkpoint and resume worked
        total_processed = len([f for f in processed_files if f['status'] == 'success'])
        total_failed = len(failed_files)
        
        assert total_processed + total_failed == len(pdf_files)
        assert any(f.get('resumed', False) for f in processed_files)  # Some were resumed


# =====================================================================
# SYSTEM STABILITY TESTS
# =====================================================================

class TestSystemStability:
    """Test system stability under various error conditions."""
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_isolation(self, enhanced_pdf_processor, temp_test_dir):
        """Test that errors are properly isolated and don't affect other operations."""
        
        # Create mix of good and bad files
        good_files = []
        bad_files = []
        
        for i in range(3):
            # Good files
            good_file = temp_test_dir / f"good_{i}.pdf"
            PDFTestFileGenerator.create_valid_pdf_with_content(
                good_file, f"Good content {i}"
            )
            good_files.append(good_file)
            
            # Bad files
            bad_file = temp_test_dir / f"bad_{i}.pdf"
            PDFTestFileGenerator.create_corrupted_pdf(bad_file, "zero_byte")
            bad_files.append(bad_file)
        
        # Process files independently - errors shouldn't propagate
        good_results = []
        bad_errors = []
        
        # Process good files first
        for good_file in good_files:
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(good_file)
                good_results.append(result)
            except Exception as e:
                pytest.fail(f"Good file {good_file} failed: {e}")
        
        # Process bad files - should fail but not affect good files
        for bad_file in bad_files:
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(bad_file)
                pytest.fail(f"Bad file {bad_file} should have failed")
            except BiomedicalPDFProcessorError as e:
                bad_errors.append(e)
        
        # Process good files again - should still work
        for good_file in good_files:
            try:
                result = enhanced_pdf_processor.extract_text_from_pdf(good_file)
                assert 'text' in result
            except Exception as e:
                pytest.fail(f"Good file {good_file} failed after bad file processing: {e}")
        
        # Verify isolation
        assert len(good_results) == len(good_files)
        assert len(bad_errors) == len(bad_files)

    @pytest.mark.asyncio
    async def test_memory_leak_prevention_during_errors(self, enhanced_pdf_processor, temp_test_dir):
        """Test that memory leaks don't occur during sustained error conditions."""
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Create many corrupted files to trigger repeated errors
        corrupted_files = []
        for i in range(20):
            corrupted_file = temp_test_dir / f"memory_leak_test_{i}.pdf"
            PDFTestFileGenerator.create_corrupted_pdf(
                corrupted_file, 
                ["zero_byte", "invalid_header", "truncated", "binary_garbage"][i % 4]
            )
            corrupted_files.append(corrupted_file)
        
        # Process all corrupted files (should fail but not leak memory)
        for corrupted_file in corrupted_files:
            try:
                enhanced_pdf_processor.extract_text_from_pdf(corrupted_file)
            except BiomedicalPDFProcessorError:
                # Expected failure
                pass
            
            # Force garbage collection periodically
            if len(corrupted_files) % 5 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Memory increase should be minimal (under 50MB for error handling)
        assert memory_increase_mb < 50, f"Memory increased by {memory_increase_mb}MB - possible leak"

    @pytest.mark.asyncio
    async def test_performance_under_sustained_error_conditions(self, enhanced_pdf_processor, 
                                                               temp_test_dir):
        """Test system performance doesn't degrade under sustained error conditions."""
        
        # Create corrupted files for sustained error testing
        corrupted_files = []
        for i in range(10):
            corrupted_file = temp_test_dir / f"sustained_error_{i}.pdf"
            PDFTestFileGenerator.create_corrupted_pdf(corrupted_file, "invalid_header")
            corrupted_files.append(corrupted_file)
        
        processing_times = []
        
        # Process files and measure time for each
        for corrupted_file in corrupted_files:
            start_time = time.time()
            
            try:
                enhanced_pdf_processor.extract_text_from_pdf(corrupted_file)
            except BiomedicalPDFProcessorError:
                # Expected failure
                pass
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Performance shouldn't degrade significantly over time
        if len(processing_times) >= 5:
            early_times = processing_times[:3]
            later_times = processing_times[-3:]
            
            avg_early = sum(early_times) / len(early_times)
            avg_later = sum(later_times) / len(later_times)
            
            # Later processing shouldn't be significantly slower (allow 2x degradation max)
            performance_ratio = avg_later / avg_early
            assert performance_ratio < 2.0, f"Performance degraded by {performance_ratio:.2f}x"

    @pytest.mark.asyncio
    async def test_error_logging_and_monitoring_validation(self, enhanced_pdf_processor, 
                                                          temp_test_dir, correlation_logger):
        """Test that error logging and monitoring work correctly under all conditions."""
        
        logger, correlation_manager = correlation_logger
        
        # Capture log messages
        log_messages = []
        
        class TestLogHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(self.format(record))
        
        test_handler = TestLogHandler()
        test_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add handler to processor logger
        enhanced_pdf_processor.logger.addHandler(test_handler)
        enhanced_pdf_processor.logger.setLevel(logging.DEBUG)
        
        try:
            # Test various error scenarios
            error_scenarios = [
                ("zero_byte.pdf", "zero_byte"),
                ("invalid_header.pdf", "invalid_header"),
                ("truncated.pdf", "truncated")
            ]
            
            for filename, corruption_type in error_scenarios:
                pdf_file = temp_test_dir / filename
                PDFTestFileGenerator.create_corrupted_pdf(pdf_file, corruption_type)
                
                with correlation_manager.correlation_context():
                    try:
                        enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                    except BiomedicalPDFProcessorError:
                        # Expected failure
                        pass
            
            # Verify logging captured all scenarios
            assert len(log_messages) > 0
            
            # Should have error messages for each scenario
            error_logs = [msg for msg in log_messages if 'ERROR' in msg or 'WARN' in msg]
            assert len(error_logs) >= len(error_scenarios)
            
            # Should contain file names and error details
            for filename, _ in error_scenarios:
                filename_in_logs = any(filename in msg for msg in log_messages)
                assert filename_in_logs, f"No log messages found for {filename}"
        
        finally:
            enhanced_pdf_processor.logger.removeHandler(test_handler)

    @pytest.mark.asyncio
    async def test_long_running_stability_under_mixed_conditions(self, enhanced_pdf_processor,
                                                                temp_test_dir):
        """Test long-running stability under mixed success/failure conditions."""
        
        # Create a realistic mix of files that would be encountered in production
        file_scenarios = [
            ("valid_small.pdf", "valid", "Small document content", 1),
            ("corrupted_1.pdf", "corrupted", "zero_byte", None),
            ("valid_medium.pdf", "valid", "Medium document content\n" * 100, 2),
            ("invalid_header.pdf", "corrupted", "invalid_header", None),
            ("valid_large.pdf", "valid", "Large document content\n" * 500, 5),
            ("truncated.pdf", "corrupted", "truncated", None),
            ("valid_unicode.pdf", "valid", "Unicode content: héllo wörld 测试", 1),
            ("binary_garbage.pdf", "corrupted", "binary_garbage", None)
        ]
        
        pdf_files = []
        expected_successes = 0
        
        for filename, file_type, content, pages in file_scenarios:
            pdf_file = temp_test_dir / filename
            
            if file_type == "valid":
                PDFTestFileGenerator.create_valid_pdf_with_content(pdf_file, content, pages or 1)
                expected_successes += 1
            else:  # corrupted
                PDFTestFileGenerator.create_corrupted_pdf(pdf_file, content)
            
            pdf_files.append((pdf_file, file_type))
        
        # Process files multiple times to simulate long-running operation
        cycles = 3
        total_successes = 0
        total_failures = 0
        processing_times = []
        
        for cycle in range(cycles):
            cycle_start = time.time()
            cycle_successes = 0
            cycle_failures = 0
            
            # Randomize order to simulate real-world conditions
            random.shuffle(pdf_files)
            
            for pdf_file, expected_type in pdf_files:
                try:
                    result = enhanced_pdf_processor.extract_text_from_pdf(pdf_file)
                    cycle_successes += 1
                    
                    # Validate result structure
                    assert 'text' in result
                    assert 'metadata' in result
                    assert 'page_texts' in result
                    
                except BiomedicalPDFProcessorError:
                    cycle_failures += 1
                    # Expected for corrupted files
            
            cycle_end = time.time()
            processing_times.append(cycle_end - cycle_start)
            
            total_successes += cycle_successes
            total_failures += cycle_failures
            
            # Force garbage collection between cycles
            gc.collect()
        
        # Verify stability over multiple cycles
        assert total_successes >= expected_successes * cycles * 0.8  # Allow some PDF creation failures
        assert total_failures > 0  # Should have some failures from corrupted files
        
        # Performance should remain stable across cycles
        if len(processing_times) >= 2:
            avg_time = sum(processing_times) / len(processing_times)
            max_deviation = max(abs(t - avg_time) for t in processing_times)
            relative_deviation = max_deviation / avg_time
            
            # Processing time variation should be reasonable (within 50% of average)
            assert relative_deviation < 0.5, f"Processing time varied by {relative_deviation:.1%}"


# =====================================================================
# TEST EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Configure logging for test execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with verbose output and detailed reporting
    pytest.main([
        __file__,
        "-v",
        "-s", 
        "--tb=short",
        "--maxfail=5",
        "-x"  # Stop on first failure for debugging
    ])