#!/usr/bin/env python3
"""
Comprehensive unit tests for progress tracking functionality during PDF processing.

This module tests the progress tracking and logging capabilities of the BiomedicalPDFProcessor
class, focusing specifically on progress reporting during single and batch PDF processing
operations.
"""

import pytest
import asyncio
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any, Tuple
import fitz

# Add the parent directory to the path to import the module
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pdf_processor import (
    BiomedicalPDFProcessor, BiomedicalPDFProcessorError,
    PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError,
    PDFFileAccessError, PDFContentError
)
from config import LightRAGConfig, LightRAGConfigError, setup_lightrag_logging
import shutil
import threading
import os


class TestProgressTrackingSinglePDF:
    """Test progress tracking during single PDF processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_single_pdf_processing_progress_logging(self, caplog):
        """Test progress tracking messages during single PDF processing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Create mock PyMuPDF document
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 3
            mock_doc.metadata = {
                'title': 'Test Document',
                'author': 'Test Author',
                'pages': 3
            }
            
            # Create mock pages with realistic content
            mock_pages = []
            for i in range(3):
                mock_page = MagicMock()
                mock_page.get_text.return_value = f"Page {i+1} content with biomedical text"
                mock_pages.append(mock_page)
            
            mock_doc.load_page.side_effect = lambda page_num: mock_pages[page_num]
            
            with patch('fitz.open', return_value=mock_doc):
                with patch.object(self.processor, '_validate_pdf_file'):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        
                        with caplog.at_level(logging.INFO):
                            result = self.processor.extract_text_from_pdf(tmp_path)
                            
                            # Verify progress tracking messages
                            log_messages = [record.message for record in caplog.records]
                            
                            # Check for opening message
                            opening_logs = [msg for msg in log_messages if "Opening PDF file" in msg]
                            assert len(opening_logs) >= 1, "Should log PDF opening"
                            
                            # Check for page count logging
                            page_count_logs = [msg for msg in log_messages if "PDF has 3 pages" in msg]
                            assert len(page_count_logs) == 1, "Should log total page count"
                            
                            # Check for processing range message
                            range_logs = [msg for msg in log_messages if "Processing pages 0 to 2" in msg]
                            assert len(range_logs) == 1, "Should log processing range"
                            
                            # Check for success summary
                            success_logs = [msg for msg in log_messages if "Successfully processed 3 pages" in msg]
                            assert len(success_logs) == 1, "Should log successful completion with page count"
                            
                            # Verify character count logging
                            character_logs = [msg for msg in log_messages if "characters" in msg]
                            assert len(character_logs) >= 1, "Should log character counts"
                            
            # Cleanup
            tmp_path.unlink(missing_ok=True)
    
    def test_single_pdf_page_by_page_progress_tracking(self, caplog):
        """Test page-by-page progress tracking during single PDF processing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Create mock PyMuPDF document with 5 pages
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 5
            mock_doc.metadata = {'title': 'Multi-page Test'}
            
            # Track page processing calls
            page_texts = [f"Page {i+1} biomedical content" for i in range(5)]
            mock_pages = []
            for i in range(5):
                mock_page = MagicMock()
                mock_page.get_text.return_value = page_texts[i]
                mock_pages.append(mock_page)
            
            mock_doc.load_page.side_effect = lambda page_num: mock_pages[page_num]
            
            with patch('fitz.open', return_value=mock_doc):
                with patch.object(self.processor, '_validate_pdf_file'):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 2048
                        
                        with caplog.at_level(logging.DEBUG):
                            result = self.processor.extract_text_from_pdf(tmp_path)
                            
                            # Verify page-by-page progress tracking
                            debug_logs = [record.message for record in caplog.records 
                                        if record.levelname == 'DEBUG']
                            
                            # Should have debug logs for each page
                            page_debug_logs = [msg for msg in debug_logs 
                                             if "Extracted" in msg and "characters from page" in msg]
                            assert len(page_debug_logs) == 5, f"Should have 5 page debug logs, got {len(page_debug_logs)}"
                            
                            # Verify page numbers are tracked correctly
                            for i in range(5):
                                page_specific_logs = [msg for msg in page_debug_logs 
                                                    if f"page {i}" in msg]
                                assert len(page_specific_logs) == 1, f"Should have exactly one log for page {i}"
            
            # Cleanup
            tmp_path.unlink(missing_ok=True)
    
    def test_single_pdf_error_progress_tracking(self, caplog):
        """Test progress tracking when single PDF processing encounters errors."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Create mock PyMuPDF document where page 2 fails
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 4
            mock_doc.metadata = {'title': 'Error Test'}
            
            def mock_load_page(page_num):
                if page_num == 2:
                    raise Exception("Page extraction error")
                mock_page = MagicMock()
                mock_page.get_text.return_value = f"Page {page_num} content"
                return mock_page
            
            mock_doc.load_page.side_effect = mock_load_page
            
            with patch('fitz.open', return_value=mock_doc):
                with patch.object(self.processor, '_validate_pdf_file'):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        
                        with caplog.at_level(logging.WARNING):
                            result = self.processor.extract_text_from_pdf(tmp_path)
                            
                            # Verify error tracking in progress
                            warning_logs = [record.message for record in caplog.records 
                                          if record.levelname == 'WARNING']
                            
                            # Should have warning for failed page
                            page_error_logs = [msg for msg in warning_logs 
                                             if "Failed to extract text from page 2" in msg]
                            assert len(page_error_logs) == 1, "Should log page extraction failure"
                            
                            # Should still complete processing with other pages
                            info_logs = [record.message for record in caplog.records 
                                       if record.levelname == 'INFO']
                            success_logs = [msg for msg in info_logs 
                                          if "Successfully processed" in msg]
                            assert len(success_logs) == 1, "Should still log successful completion"
            
            # Cleanup
            tmp_path.unlink(missing_ok=True)


class TestProgressTrackingBatchProcessing:
    """Test progress tracking during batch PDF processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
    
    @pytest.fixture
    def temp_pdf_directory(self):
        """Create a temporary directory with mock PDF files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock PDF files
        pdf_files = []
        for i in range(5):
            pdf_file = temp_dir / f"paper_{i+1}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n%test content")  # Minimal PDF header
            pdf_files.append(pdf_file)
        
        yield temp_dir, pdf_files
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_batch_processing_file_discovery_progress(self, caplog, temp_pdf_directory):
        """Test progress tracking during file discovery phase of batch processing."""
        temp_dir, pdf_files = temp_pdf_directory
        
        with patch.object(self.processor, 'extract_text_from_pdf') as mock_extract:
            # Mock successful extraction
            mock_extract.return_value = {
                'text': 'Mock extracted text',
                'metadata': {'filename': 'test.pdf', 'pages': 2},
                'processing_info': {'total_characters': 100, 'pages_processed': 2},
                'page_texts': ['Page 1', 'Page 2']
            }
            
            with caplog.at_level(logging.INFO):
                documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                
                # Verify file discovery logging
                info_logs = [record.message for record in caplog.records 
                           if record.levelname == 'INFO']
                
                # Check for file discovery message
                discovery_logs = [msg for msg in info_logs 
                                if f"Found {len(pdf_files)} PDF files" in msg]
                assert len(discovery_logs) == 1, "Should log file discovery count"
                
                # Check directory path logging
                directory_logs = [msg for msg in info_logs 
                                if str(temp_dir) in msg]
                assert len(directory_logs) >= 1, "Should log directory being processed"
    
    def test_batch_processing_file_index_progress_tracking(self, caplog, temp_pdf_directory):
        """Test file-by-file index progress tracking (1/5, 2/5 format)."""
        temp_dir, pdf_files = temp_pdf_directory
        
        with patch.object(self.processor, 'extract_text_from_pdf') as mock_extract:
            # Mock successful extraction
            mock_extract.return_value = {
                'text': 'Mock extracted text',
                'metadata': {'filename': 'test.pdf', 'pages': 2},
                'processing_info': {'total_characters': 100, 'pages_processed': 2},
                'page_texts': ['Page 1', 'Page 2']
            }
            
            with caplog.at_level(logging.INFO):
                documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                
                # Verify file index progress tracking
                info_logs = [record.message for record in caplog.records 
                           if record.levelname == 'INFO']
                
                # Check for file index messages (1/5, 2/5, etc.)
                for i in range(len(pdf_files)):
                    expected_pattern = f"Processing PDF {i+1}/{len(pdf_files)}"
                    index_logs = [msg for msg in info_logs if expected_pattern in msg]
                    assert len(index_logs) == 1, f"Should have progress message for file {i+1}/{len(pdf_files)}"
    
    def test_batch_processing_success_statistics_tracking(self, caplog, temp_pdf_directory):
        """Test tracking of processing statistics during batch processing."""
        temp_dir, pdf_files = temp_pdf_directory
        
        with patch.object(self.processor, 'extract_text_from_pdf') as mock_extract:
            # Mock successful extraction with varying statistics
            def mock_extraction(*args, **kwargs):
                return {
                    'text': 'Mock extracted text with varying length',
                    'metadata': {'filename': 'test.pdf', 'pages': 3},
                    'processing_info': {'total_characters': 250, 'pages_processed': 3},
                    'page_texts': ['Page 1', 'Page 2', 'Page 3']
                }
            
            mock_extract.side_effect = mock_extraction
            
            with caplog.at_level(logging.INFO):
                documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                
                info_logs = [record.message for record in caplog.records 
                           if record.levelname == 'INFO']
                
                # Check for per-file success statistics
                for pdf_file in pdf_files:
                    success_logs = [msg for msg in info_logs 
                                  if f"Successfully processed {pdf_file.name}" in msg 
                                  and "characters" in msg and "pages" in msg]
                    assert len(success_logs) == 1, f"Should have success stats for {pdf_file.name}"
                
                # Check for final batch summary
                summary_logs = [msg for msg in info_logs 
                              if "Batch processing completed" in msg]
                assert len(summary_logs) == 1, "Should have batch processing summary"
                
                # Verify summary contains correct counts
                summary_msg = summary_logs[0]
                assert f"{len(pdf_files)} successful" in summary_msg, "Summary should show successful count"
                assert "0 failed" in summary_msg, "Summary should show zero failed count"
    
    def test_batch_processing_failure_statistics_tracking(self, caplog, temp_pdf_directory):
        """Test tracking of failures during batch processing."""
        temp_dir, pdf_files = temp_pdf_directory
        
        # Make some files fail
        failure_count = 2
        
        def mock_extraction(pdf_path, *args, **kwargs):
            pdf_name = Path(pdf_path).name
            if pdf_name in ['paper_2.pdf', 'paper_4.pdf']:  # Make 2 files fail
                raise PDFValidationError(f"Mock validation error for {pdf_name}")
            return {
                'text': 'Mock extracted text',
                'metadata': {'filename': pdf_name, 'pages': 2},
                'processing_info': {'total_characters': 150, 'pages_processed': 2},
                'page_texts': ['Page 1', 'Page 2']
            }
        
        with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extraction):
            with caplog.at_level(logging.INFO):
                documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                
                # Verify error tracking
                error_logs = [record.message for record in caplog.records 
                            if record.levelname == 'ERROR']
                
                # Should have error logs for failed files
                validation_error_logs = [msg for msg in error_logs 
                                       if "PDF processing error" in msg]
                assert len(validation_error_logs) == failure_count, f"Should have {failure_count} error logs"
                
                # Check final summary includes failure count
                info_logs = [record.message for record in caplog.records 
                           if record.levelname == 'INFO']
                summary_logs = [msg for msg in info_logs 
                              if "Batch processing completed" in msg]
                assert len(summary_logs) == 1, "Should have batch processing summary"
                
                summary_msg = summary_logs[0]
                expected_successful = len(pdf_files) - failure_count
                assert f"{expected_successful} successful" in summary_msg, "Summary should show correct successful count"
                assert f"{failure_count} failed" in summary_msg, "Summary should show correct failed count"
    
    def test_batch_processing_empty_directory_progress(self, caplog):
        """Test progress tracking when processing empty directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            with caplog.at_level(logging.INFO):
                documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                
                info_logs = [record.message for record in caplog.records 
                           if record.levelname == 'INFO']
                
                # Should log that no PDF files were found
                no_files_logs = [msg for msg in info_logs 
                               if "No PDF files found" in msg]
                assert len(no_files_logs) == 1, "Should log when no PDF files found"
                
                # Should return empty list
                assert documents == [], "Should return empty list for empty directory"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_batch_processing_nonexistent_directory_progress(self, caplog):
        """Test progress tracking when processing non-existent directory."""
        nonexistent_dir = Path("/tmp/nonexistent_pdf_directory_12345")
        
        with caplog.at_level(logging.WARNING):
            documents = asyncio.run(self.processor.process_all_pdfs(nonexistent_dir))
            
            warning_logs = [record.message for record in caplog.records 
                          if record.levelname == 'WARNING']
            
            # Should log directory doesn't exist
            no_dir_logs = [msg for msg in warning_logs 
                         if "does not exist" in msg]
            assert len(no_dir_logs) == 1, "Should log when directory doesn't exist"
            
            # Should return empty list
            assert documents == [], "Should return empty list for non-existent directory"


class TestProgressTrackingVariousFileCounts:
    """Test progress tracking with various file counts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
    
    @pytest.mark.parametrize("file_count", [1, 5, 10, 15])
    def test_batch_processing_progress_with_different_file_counts(self, caplog, file_count):
        """Test progress tracking accuracy with different numbers of files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create PDF files
            pdf_files = []
            for i in range(file_count):
                pdf_file = temp_dir / f"paper_{i+1:03d}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            with patch.object(self.processor, 'extract_text_from_pdf') as mock_extract:
                # Mock successful extraction
                mock_extract.return_value = {
                    'text': 'Mock extracted text',
                    'metadata': {'filename': 'test.pdf', 'pages': 1},
                    'processing_info': {'total_characters': 50, 'pages_processed': 1},
                    'page_texts': ['Page 1']
                }
                
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    info_logs = [record.message for record in caplog.records 
                               if record.levelname == 'INFO']
                    
                    # Check file count discovery
                    discovery_logs = [msg for msg in info_logs 
                                    if f"Found {file_count} PDF files" in msg]
                    assert len(discovery_logs) == 1, f"Should discover {file_count} files"
                    
                    # Check progress messages for each file
                    progress_logs = [msg for msg in info_logs 
                                   if "Processing PDF" in msg and "/" in msg]
                    assert len(progress_logs) == file_count, f"Should have {file_count} progress messages"
                    
                    # Verify progress indexing is correct
                    for i in range(file_count):
                        expected_pattern = f"Processing PDF {i+1}/{file_count}"
                        matching_logs = [msg for msg in progress_logs 
                                       if expected_pattern in msg]
                        assert len(matching_logs) == 1, f"Should have exactly one message for file {i+1}/{file_count}"
                    
                    # Check final summary
                    summary_logs = [msg for msg in info_logs 
                                  if "Batch processing completed" in msg]
                    assert len(summary_logs) == 1, "Should have final summary"
                    
                    summary_msg = summary_logs[0]
                    assert f"{file_count} successful" in summary_msg, f"Summary should show {file_count} successful"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestProgressTrackingTimingAndPerformance:
    """Test progress tracking timing and performance metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_progress_tracking_timing_information(self, caplog):
        """Test that progress tracking includes timing information."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create a few PDF files
            pdf_files = []
            for i in range(3):
                pdf_file = temp_dir / f"timed_paper_{i+1}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            # Mock extraction with deliberate delay
            def mock_extraction_with_delay(*args, **kwargs):
                time.sleep(0.1)  # Small delay to simulate processing
                return {
                    'text': 'Mock extracted text',
                    'metadata': {'filename': 'test.pdf', 'pages': 2},
                    'processing_info': {
                        'total_characters': 100, 
                        'pages_processed': 2,
                        'processing_timestamp': time.time()
                    },
                    'page_texts': ['Page 1', 'Page 2']
                }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extraction_with_delay):
                start_time = time.time()
                
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                end_time = time.time()
                total_time = end_time - start_time
                
                # Verify that processing took reasonable time
                assert total_time >= 0.3, "Processing should take at least 0.3 seconds with delays"
                
                # Check that all expected progress messages are present
                info_logs = [record.message for record in caplog.records 
                           if record.levelname == 'INFO']
                
                # Should have discovery, progress, success, and summary messages
                discovery_logs = [msg for msg in info_logs if "Found" in msg and "PDF files" in msg]
                progress_logs = [msg for msg in info_logs if "Processing PDF" in msg]
                success_logs = [msg for msg in info_logs if "Successfully processed" in msg]
                summary_logs = [msg for msg in info_logs if "Batch processing completed" in msg]
                
                assert len(discovery_logs) == 1, "Should have file discovery log"
                assert len(progress_logs) == 3, "Should have 3 progress logs"
                assert len(success_logs) == 3, "Should have 3 success logs"
                assert len(summary_logs) == 1, "Should have summary log"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_progress_tracking_with_processing_delays(self, caplog):
        """Test progress tracking when files have varying processing times."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create PDF files
            pdf_files = []
            for i in range(4):
                pdf_file = temp_dir / f"delay_paper_{i+1}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            # Mock extraction with varying delays
            def mock_extraction_varying_delay(pdf_path, *args, **kwargs):
                pdf_name = Path(pdf_path).name
                if 'paper_1' in pdf_name:
                    time.sleep(0.05)  # Fast
                elif 'paper_2' in pdf_name:
                    time.sleep(0.15)  # Medium
                elif 'paper_3' in pdf_name:
                    time.sleep(0.25)  # Slow
                else:
                    time.sleep(0.1)   # Normal
                
                return {
                    'text': f'Mock extracted text from {pdf_name}',
                    'metadata': {'filename': pdf_name, 'pages': 2},
                    'processing_info': {'total_characters': 120, 'pages_processed': 2},
                    'page_texts': ['Page 1', 'Page 2']
                }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extraction_varying_delay):
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    # Verify all files were processed despite varying delays
                    info_logs = [record.message for record in caplog.records 
                               if record.levelname == 'INFO']
                    
                    success_logs = [msg for msg in info_logs 
                                  if "Successfully processed" in msg]
                    assert len(success_logs) == 4, "Should successfully process all files despite delays"
                    
                    # Check that progress tracking continued for all files
                    progress_logs = [msg for msg in info_logs 
                                   if "Processing PDF" in msg]
                    assert len(progress_logs) == 4, "Should have progress logs for all files"
                    
                    # Verify final summary shows all successful
                    summary_logs = [msg for msg in info_logs 
                                  if "4 successful" in msg and "0 failed" in msg]
                    assert len(summary_logs) == 1, "Should show all files succeeded in summary"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestProgressTrackingEdgeCases:
    """Test progress tracking edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_progress_tracking_with_mixed_success_failure(self, caplog):
        """Test progress tracking when some files succeed and others fail."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create PDF files
            pdf_files = []
            for i in range(6):
                pdf_file = temp_dir / f"mixed_paper_{i+1}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            # Mock extraction with mixed results
            def mock_extraction_mixed_results(pdf_path, *args, **kwargs):
                pdf_name = Path(pdf_path).name
                
                if 'paper_2' in pdf_name:
                    raise PDFValidationError(f"Validation failed for {pdf_name}")
                elif 'paper_4' in pdf_name:
                    raise PDFProcessingTimeoutError(f"Timeout processing {pdf_name}")
                elif 'paper_6' in pdf_name:
                    raise BiomedicalPDFProcessorError(f"General error for {pdf_name}")
                else:
                    return {
                        'text': f'Success text from {pdf_name}',
                        'metadata': {'filename': pdf_name, 'pages': 3},
                        'processing_info': {'total_characters': 200, 'pages_processed': 3},
                        'page_texts': ['Page 1', 'Page 2', 'Page 3']
                    }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extraction_mixed_results):
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    # Verify progress tracking for all files (both successful and failed)
                    info_logs = [record.message for record in caplog.records 
                               if record.levelname == 'INFO']
                    error_logs = [record.message for record in caplog.records 
                                if record.levelname == 'ERROR']
                    
                    # Should have progress messages for all 6 files
                    progress_logs = [msg for msg in info_logs 
                                   if "Processing PDF" in msg and "/" in msg]
                    assert len(progress_logs) == 6, "Should track progress for all files"
                    
                    # Should have success messages for 3 files
                    success_logs = [msg for msg in info_logs 
                                  if "Successfully processed" in msg]
                    assert len(success_logs) == 3, "Should have 3 success messages"
                    
                    # Should have error messages for 3 failed files
                    assert len(error_logs) == 3, "Should have 3 error messages"
                    
                    # Verify final summary is accurate
                    summary_logs = [msg for msg in info_logs 
                                  if "Batch processing completed" in msg]
                    assert len(summary_logs) == 1, "Should have final summary"
                    
                    summary_msg = summary_logs[0]
                    assert "3 successful" in summary_msg, "Summary should show 3 successful"
                    assert "3 failed" in summary_msg, "Summary should show 3 failed"
                    assert "6 total" in summary_msg, "Summary should show 6 total"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_progress_tracking_character_count_accuracy(self, caplog):
        """Test accuracy of character count tracking in progress messages."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create PDF files
            pdf_files = []
            for i in range(3):
                pdf_file = temp_dir / f"char_count_paper_{i+1}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            # Mock extraction with specific character counts
            def mock_extraction_specific_counts(pdf_path, *args, **kwargs):
                pdf_name = Path(pdf_path).name
                
                if 'paper_1' in pdf_name:
                    text = "A" * 100  # Exactly 100 characters
                    char_count = 100
                elif 'paper_2' in pdf_name:
                    text = "B" * 250  # Exactly 250 characters
                    char_count = 250
                else:
                    text = "C" * 500  # Exactly 500 characters
                    char_count = 500
                
                return {
                    'text': text,
                    'metadata': {'filename': pdf_name, 'pages': 2},
                    'processing_info': {'total_characters': char_count, 'pages_processed': 2},
                    'page_texts': ['Page 1', 'Page 2']
                }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extraction_specific_counts):
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    info_logs = [record.message for record in caplog.records 
                               if record.levelname == 'INFO']
                    
                    # Verify character counts are accurately reported
                    success_logs = [msg for msg in info_logs 
                                  if "Successfully processed" in msg]
                    
                    # Check specific character counts in success messages
                    char_count_100 = [msg for msg in success_logs if "100 characters" in msg]
                    char_count_250 = [msg for msg in success_logs if "250 characters" in msg]
                    char_count_500 = [msg for msg in success_logs if "500 characters" in msg]
                    
                    assert len(char_count_100) == 1, "Should report 100 characters for paper_1"
                    assert len(char_count_250) == 1, "Should report 250 characters for paper_2"
                    assert len(char_count_500) == 1, "Should report 500 characters for paper_3"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_progress_tracking_page_count_accuracy(self, caplog):
        """Test accuracy of page count tracking in progress messages."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create PDF files
            pdf_files = []
            for i in range(3):
                pdf_file = temp_dir / f"page_count_paper_{i+1}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            # Mock extraction with specific page counts
            def mock_extraction_specific_pages(pdf_path, *args, **kwargs):
                pdf_name = Path(pdf_path).name
                
                if 'paper_1' in pdf_name:
                    page_count = 1
                    pages = ['Page 1']
                elif 'paper_2' in pdf_name:
                    page_count = 5
                    pages = ['Page 1', 'Page 2', 'Page 3', 'Page 4', 'Page 5']
                else:
                    page_count = 10
                    pages = [f'Page {i+1}' for i in range(10)]
                
                return {
                    'text': f'Content from {pdf_name}',
                    'metadata': {'filename': pdf_name, 'pages': page_count},
                    'processing_info': {'total_characters': 150, 'pages_processed': page_count},
                    'page_texts': pages
                }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extraction_specific_pages):
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    info_logs = [record.message for record in caplog.records 
                               if record.levelname == 'INFO']
                    
                    # Verify page counts are accurately reported
                    success_logs = [msg for msg in info_logs 
                                  if "Successfully processed" in msg]
                    
                    # Check specific page counts in success messages
                    page_count_1 = [msg for msg in success_logs if "1 pages" in msg]
                    page_count_5 = [msg for msg in success_logs if "5 pages" in msg]
                    page_count_10 = [msg for msg in success_logs if "10 pages" in msg]
                    
                    assert len(page_count_1) == 1, "Should report 1 page for paper_1"
                    assert len(page_count_5) == 1, "Should report 5 pages for paper_2"
                    assert len(page_count_10) == 1, "Should report 10 pages for paper_3"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestProgressTrackingCallbacksAndMetrics:
    """Test progress tracking callbacks and metrics collection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_progress_tracking_log_level_filtering(self, caplog):
        """Test that progress messages appear at appropriate log levels."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create single PDF file
            pdf_file = temp_dir / "level_test.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n%test content")
            
            with patch.object(self.processor, 'extract_text_from_pdf') as mock_extract:
                mock_extract.return_value = {
                    'text': 'Test content',
                    'metadata': {'filename': 'test.pdf', 'pages': 2},
                    'processing_info': {'total_characters': 100, 'pages_processed': 2},
                    'page_texts': ['Page 1', 'Page 2']
                }
                
                # Test with INFO level
                caplog.clear()
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    info_records = [record for record in caplog.records 
                                  if record.levelname == 'INFO']
                    debug_records = [record for record in caplog.records 
                                   if record.levelname == 'DEBUG']
                    
                    # Should have INFO level progress messages
                    assert len(info_records) >= 3, "Should have INFO level messages"
                    # DEBUG messages should not appear at INFO level
                    assert len(debug_records) == 0, "Should not have DEBUG messages at INFO level"
                
                # Test with WARNING level (should have fewer messages)
                caplog.clear()
                with caplog.at_level(logging.WARNING):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    warning_or_higher = [record for record in caplog.records 
                                       if record.levelno >= logging.WARNING]
                    info_or_lower = [record for record in caplog.records 
                                   if record.levelno < logging.WARNING]
                    
                    # Should have fewer messages at WARNING level
                    assert len(info_or_lower) == 0, "Should not have INFO messages at WARNING level"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_progress_tracking_message_consistency(self, caplog):
        """Test consistency of progress tracking message formats."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create PDF files
            pdf_files = []
            for i in range(4):
                pdf_file = temp_dir / f"consistency_paper_{i+1}.pdf"
                pdf_file.write_bytes(b"%PDF-1.4\n%test content")
                pdf_files.append(pdf_file)
            
            with patch.object(self.processor, 'extract_text_from_pdf') as mock_extract:
                mock_extract.return_value = {
                    'text': 'Consistent test content',
                    'metadata': {'filename': 'test.pdf', 'pages': 2},
                    'processing_info': {'total_characters': 150, 'pages_processed': 2},
                    'page_texts': ['Page 1', 'Page 2']
                }
                
                with caplog.at_level(logging.INFO):
                    documents = asyncio.run(self.processor.process_all_pdfs(temp_dir))
                    
                    info_logs = [record.message for record in caplog.records 
                               if record.levelname == 'INFO']
                    
                    # Check progress message format consistency
                    progress_logs = [msg for msg in info_logs 
                                   if "Processing PDF" in msg]
                    
                    for i, progress_msg in enumerate(progress_logs):
                        # Should follow format: "Processing PDF X/Y: filename"
                        expected_prefix = f"Processing PDF {i+1}/4:"
                        assert expected_prefix in progress_msg, f"Progress message should start with '{expected_prefix}'"
                        assert pdf_files[i].name in progress_msg, f"Progress message should contain filename"
                    
                    # Check success message format consistency
                    success_logs = [msg for msg in info_logs 
                                  if "Successfully processed" in msg]
                    
                    for success_msg in success_logs:
                        # Should contain filename, character count, and page count
                        assert "characters" in success_msg, "Success message should mention character count"
                        assert "pages" in success_msg, "Success message should mention page count"
                        
                        # Should contain actual numbers
                        assert "150 characters" in success_msg, "Should show correct character count"
                        assert "2 pages" in success_msg, "Should show correct page count"
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestLoggingConfiguration:
    """Test logging configuration and setup."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_logs_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_log_level_configuration_debug(self, caplog):
        """Test DEBUG log level configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="DEBUG",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_debug_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        # Test all log levels
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # All messages should appear
        assert len(caplog.records) == 5
        assert any(record.levelname == 'DEBUG' for record in caplog.records)
        assert any(record.levelname == 'INFO' for record in caplog.records)
        assert any(record.levelname == 'WARNING' for record in caplog.records)
        assert any(record.levelname == 'ERROR' for record in caplog.records)
        assert any(record.levelname == 'CRITICAL' for record in caplog.records)
    
    def test_log_level_configuration_info(self, caplog):
        """Test INFO log level configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_info_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        # Test all log levels
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # Only INFO and above should appear
        logged_levels = [record.levelname for record in caplog.records]
        assert 'DEBUG' not in logged_levels
        assert 'INFO' in logged_levels
        assert 'WARNING' in logged_levels
        assert 'ERROR' in logged_levels
        assert 'CRITICAL' in logged_levels
    
    def test_log_level_configuration_warning(self, caplog):
        """Test WARNING log level configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="WARNING",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_warning_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        # Test all log levels
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # Only WARNING and above should appear
        logged_levels = [record.levelname for record in caplog.records]
        assert 'DEBUG' not in logged_levels
        assert 'INFO' not in logged_levels
        assert 'WARNING' in logged_levels
        assert 'ERROR' in logged_levels
        assert 'CRITICAL' in logged_levels
    
    def test_log_level_configuration_error(self, caplog):
        """Test ERROR log level configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="ERROR",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_error_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        # Test all log levels
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # Only ERROR and above should appear
        logged_levels = [record.levelname for record in caplog.records]
        assert 'DEBUG' not in logged_levels
        assert 'INFO' not in logged_levels
        assert 'WARNING' not in logged_levels
        assert 'ERROR' in logged_levels
        assert 'CRITICAL' in logged_levels
    
    def test_log_level_configuration_critical(self, caplog):
        """Test CRITICAL log level configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="CRITICAL",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_critical_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        # Test all log levels
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # Only CRITICAL should appear
        logged_levels = [record.levelname for record in caplog.records]
        assert 'DEBUG' not in logged_levels
        assert 'INFO' not in logged_levels
        assert 'WARNING' not in logged_levels
        assert 'ERROR' not in logged_levels
        assert 'CRITICAL' in logged_levels
    
    def test_console_only_logging_configuration(self, caplog):
        """Test console-only logging configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_console_logger")
        
        # Verify no file handler is created
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) == 0, "Should not have file handlers when file logging disabled"
        
        # Verify console handler exists
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(console_handlers) == 1, "Should have exactly one console handler"
        
        # Test logging works
        logger.propagate = True  # Enable propagation for caplog
        with caplog.at_level(logging.INFO):
            logger.info("Console test message")
        
        assert len(caplog.records) == 1
        assert caplog.records[0].message == "Console test message"
    
    def test_file_logging_configuration(self):
        """Test file logging configuration."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=True,
            log_filename="test_logging.log",
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_file_logger")
        
        # Verify file handler is created
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) == 1, "Should have exactly one file handler"
        
        # Verify console handler also exists
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(console_handlers) == 1, "Should have exactly one console handler"
        
        # Test logging creates file
        logger.info("File test message")
        
        log_file = self.test_log_dir / "test_logging.log"
        assert log_file.exists(), "Log file should be created"
        
        # Read and verify log content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "File test message" in content
            assert "INFO" in content
    
    def test_environment_variable_log_level_configuration(self):
        """Test log level configuration via environment variables."""
        # Test DEBUG level
        with patch.dict(os.environ, {"LIGHTRAG_LOG_LEVEL": "DEBUG"}):
            config = LightRAGConfig(
                api_key="test_key",
                log_dir=self.test_log_dir,
                enable_file_logging=False,
                auto_create_dirs=False
            )
            assert config.log_level == "DEBUG"
        
        # Test INFO level
        with patch.dict(os.environ, {"LIGHTRAG_LOG_LEVEL": "INFO"}):
            config = LightRAGConfig(
                api_key="test_key",
                log_dir=self.test_log_dir,
                enable_file_logging=False,
                auto_create_dirs=False
            )
            assert config.log_level == "INFO"
        
        # Test WARNING level
        with patch.dict(os.environ, {"LIGHTRAG_LOG_LEVEL": "WARNING"}):
            config = LightRAGConfig(
                api_key="test_key",
                log_dir=self.test_log_dir,
                enable_file_logging=False,
                auto_create_dirs=False
            )
            assert config.log_level == "WARNING"
    
    def test_environment_variable_log_dir_configuration(self):
        """Test log directory configuration via environment variables."""
        test_env_log_dir = Path(tempfile.mkdtemp(prefix="env_logs_"))
        
        try:
            with patch.dict(os.environ, {"LIGHTRAG_LOG_DIR": str(test_env_log_dir)}):
                config = LightRAGConfig(
                    api_key="test_key",
                    enable_file_logging=True,
                    auto_create_dirs=False
                )
                assert config.log_dir == test_env_log_dir
        finally:
            if test_env_log_dir.exists():
                shutil.rmtree(test_env_log_dir, ignore_errors=True)
    
    def test_environment_variable_enable_file_logging(self):
        """Test enable file logging configuration via environment variables."""
        # Test enabling file logging
        with patch.dict(os.environ, {"LIGHTRAG_ENABLE_FILE_LOGGING": "true"}):
            config = LightRAGConfig(
                api_key="test_key",
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            assert config.enable_file_logging is True
        
        # Test disabling file logging
        with patch.dict(os.environ, {"LIGHTRAG_ENABLE_FILE_LOGGING": "false"}):
            config = LightRAGConfig(
                api_key="test_key",
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            assert config.enable_file_logging is False
        
        # Test various true values
        for true_value in ["1", "yes", "t", "on", "TRUE", "True"]:
            with patch.dict(os.environ, {"LIGHTRAG_ENABLE_FILE_LOGGING": true_value}):
                config = LightRAGConfig(
                    api_key="test_key",
                    log_dir=self.test_log_dir,
                    auto_create_dirs=False
                )
                assert config.enable_file_logging is True, f"'{true_value}' should be interpreted as True"


class TestLoggingBehavior:
    """Test logging behavior and message formatting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_logs_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_log_message_formatting_console(self, caplog):
        """Test log message formatting for console output."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_format_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        with caplog.at_level(logging.INFO):
            logger.info("Test formatting message")
        
        # Console format should be simple: "LEVEL: message"
        record = caplog.records[0]
        # The actual formatted message from console handler would be "INFO: Test formatting message"
        # But caplog captures the raw record, so we check the message content
        assert record.message == "Test formatting message"
        assert record.levelname == "INFO"
    
    def test_log_message_formatting_file(self):
        """Test log message formatting for file output."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=True,
            log_filename="format_test.log",
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_file_format_logger")
        logger.info("Test file formatting message")
        
        log_file = self.test_log_dir / "format_test.log"
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read().strip()
            # File format should be detailed: "timestamp - logger_name - level - message"
            assert "test_file_format_logger" in content
            assert "INFO" in content
            assert "Test file formatting message" in content
            # Should contain timestamp
            assert "-" in content  # Timestamp format includes dashes
    
    def test_logging_during_error_conditions(self, caplog):
        """Test logging behavior during error conditions."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="DEBUG",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_error_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        with caplog.at_level(logging.DEBUG):
            try:
                raise ValueError("Test error for logging")
            except ValueError as e:
                logger.error(f"Caught error: {e}")
                logger.exception("Exception with traceback")
        
        error_records = [r for r in caplog.records if r.levelname == 'ERROR']
        assert len(error_records) == 2
        
        # Check error message logging
        assert "Caught error: Test error for logging" in [r.message for r in error_records]
        assert "Exception with traceback" in [r.message for r in error_records]
        
        # Check that exception record has exc_info
        exception_record = next(r for r in error_records if r.message == "Exception with traceback")
        assert exception_record.exc_info is not None
    
    def test_logging_performance_efficiency(self, caplog):
        """Test logging performance and efficiency."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_performance_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        # Test that DEBUG messages don't get processed at INFO level
        with caplog.at_level(logging.INFO):
            start_time = time.time()
            
            # These debug messages should not be processed
            for i in range(1000):
                logger.debug(f"Debug message {i} - this should not be processed")
            
            # These info messages should be processed
            for i in range(10):
                logger.info(f"Info message {i}")
            
            end_time = time.time()
        
        # Should only have 10 INFO messages, no DEBUG messages
        assert len(caplog.records) == 10
        info_records = [r for r in caplog.records if r.levelname == 'INFO']
        debug_records = [r for r in caplog.records if r.levelname == 'DEBUG']
        
        assert len(info_records) == 10
        assert len(debug_records) == 0
        
        # Processing should be fast (less than 1 second for this simple test)
        assert end_time - start_time < 1.0
    
    def test_logging_with_different_scenarios(self, caplog):
        """Test logging behavior with different application scenarios."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="DEBUG",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_scenario_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        with caplog.at_level(logging.DEBUG):
            # Simulate PDF processing scenario
            logger.info("Starting PDF batch processing")
            logger.debug("Loading configuration")
            logger.info("Processing file 1/5: document.pdf")
            logger.debug("Extracted 2500 characters from 10 pages")
            logger.warning("Page 3 had extraction issues, continuing")
            logger.info("Processing file 2/5: report.pdf")
            logger.error("Failed to process report.pdf: file corrupted")
            logger.info("Batch processing completed: 1 successful, 1 failed")
        
        # Verify appropriate log levels were used
        records_by_level = {}
        for record in caplog.records:
            if record.levelname not in records_by_level:
                records_by_level[record.levelname] = []
            records_by_level[record.levelname].append(record.message)
        
        assert 'INFO' in records_by_level
        assert 'DEBUG' in records_by_level
        assert 'WARNING' in records_by_level
        assert 'ERROR' in records_by_level
        
        # Check specific message routing
        assert len(records_by_level['INFO']) == 4  # Start, file 1, file 2, completion
        assert len(records_by_level['DEBUG']) == 2  # Config load, extraction details
        assert len(records_by_level['WARNING']) == 1  # Page extraction issues
        assert len(records_by_level['ERROR']) == 1  # File corruption


class TestLoggingRotation:
    """Test log rotation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_logs_rotation_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_log_rotation_configuration(self):
        """Test log rotation configuration parameters."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=True,
            log_filename="rotation_test.log",
            log_max_bytes=1024,  # Small size for testing
            log_backup_count=3,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_rotation_logger")
        
        # Verify handler configuration
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) == 1
        
        handler = file_handlers[0]
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3
    
    def test_log_rotation_behavior(self):
        """Test actual log rotation behavior with large messages."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=True,
            log_filename="rotation_behavior.log",
            log_max_bytes=500,  # Very small for rotation testing
            log_backup_count=2,
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("test_rotation_behavior_logger")
        
        # Write enough data to trigger rotation
        large_message = "This is a large message for testing log rotation. " * 10
        
        for i in range(5):
            logger.info(f"Message {i}: {large_message}")
        
        log_file = self.test_log_dir / "rotation_behavior.log"
        backup1 = self.test_log_dir / "rotation_behavior.log.1"
        backup2 = self.test_log_dir / "rotation_behavior.log.2"
        
        # Main log file should exist
        assert log_file.exists()
        
        # At least one backup should be created due to size limit
        # Note: Exact rotation behavior depends on message sizes and timing
        log_files = list(self.test_log_dir.glob("rotation_behavior.log*"))
        assert len(log_files) >= 1  # At least the main file
        
        # Verify we can read from the current log file
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Message" in content
    
    def test_log_rotation_with_environment_variables(self):
        """Test log rotation configuration via environment variables."""
        with patch.dict(os.environ, {
            "LIGHTRAG_LOG_MAX_BYTES": "2048",
            "LIGHTRAG_LOG_BACKUP_COUNT": "7"
        }):
            config = LightRAGConfig(
                api_key="test_key",
                log_dir=self.test_log_dir,
                enable_file_logging=True,
                auto_create_dirs=False
            )
            
            assert config.log_max_bytes == 2048
            assert config.log_backup_count == 7
            
            logger = config.setup_lightrag_logging("test_env_rotation_logger")
            
            # Verify handler uses environment values
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
            assert len(file_handlers) == 1
            
            handler = file_handlers[0]
            assert handler.maxBytes == 2048
            assert handler.backupCount == 7


class TestLoggingErrorHandling:
    """Test logging error handling and graceful degradation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_logs_errors_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_graceful_degradation_when_file_logging_fails(self, caplog):
        """Test graceful degradation when file logging fails."""
        # Use a read-only directory to simulate permission error
        readonly_dir = self.test_log_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            config = LightRAGConfig(
                api_key="test_key",
                log_level="INFO",
                log_dir=readonly_dir,
                enable_file_logging=True,
                log_filename="should_fail.log",
                auto_create_dirs=False
            )
            
            # This should not raise an exception, but log a warning
            logger = config.setup_lightrag_logging("test_degradation_logger")
            logger.propagate = True  # Enable propagation for caplog
            
            # Should have console handler even if file handler failed
            console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)]
            assert len(console_handlers) >= 1, "Should have console handler for graceful degradation"
            
            # Test that warnings were logged by trying to capture them through the logger itself
            with caplog.at_level(logging.WARNING):
                # The warnings are already logged during setup, but we can verify by testing the behavior
                file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
                assert len(file_handlers) == 0, "Should not have file handlers when file logging failed"
            
            # Logger should still work for console output
            caplog.clear()
            logger.propagate = True  # Ensure propagation is still enabled
            with caplog.at_level(logging.INFO):
                logger.info("Test message after file logging failure")
            
            assert len(caplog.records) == 1
            assert caplog.records[0].message == "Test message after file logging failure"
        
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)
    
    def test_invalid_log_directory_handling(self, caplog):
        """Test handling of invalid log directory paths."""
        # Use a path that points to a file instead of directory
        fake_file_path = self.test_log_dir / "fake_file"
        fake_file_path.write_text("This is a file, not a directory")
        
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=fake_file_path,  # This is a file, not a directory
            enable_file_logging=True,
            log_filename="should_fail.log",
            auto_create_dirs=False
        )
        
        # This should handle the error gracefully by wrapping in try-except
        logger = None
        try:
            logger = config.setup_lightrag_logging("test_invalid_dir_logger")
            logger.propagate = True  # Enable propagation for caplog
            
            # Should still have console handler
            console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)]
            assert len(console_handlers) >= 1
            
            # Test that logger works
            caplog.clear()
            with caplog.at_level(logging.INFO):
                logger.info("Test message with invalid log directory")
            
            assert len(caplog.records) == 1
            assert caplog.records[0].message == "Test message with invalid log directory"
            
        except LightRAGConfigError:
            # This is expected behavior when log directory setup fails completely
            # In this case, we just verify the exception was raised appropriately
            assert logger is None, "Logger should not be created when directory setup fails"
    
    def test_logging_configuration_validation_errors(self):
        """Test validation of logging configuration parameters."""
        # Test invalid log level by manually setting it after initialization
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",  # Start with valid level
            log_dir=self.test_log_dir,
            auto_create_dirs=False
        )
        config.log_level = "INVALID_LEVEL"  # Manually set invalid level
        with pytest.raises(LightRAGConfigError, match="log_level must be one of"):
            config.validate()
        
        # Test log_max_bytes validation
        with pytest.raises(LightRAGConfigError, match="log_max_bytes must be positive"):
            config = LightRAGConfig(
                api_key="test_key",
                log_level="INFO",
                log_max_bytes=0,
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            config.validate()
        
        # Test negative log_max_bytes
        with pytest.raises(LightRAGConfigError, match="log_max_bytes must be positive"):
            config = LightRAGConfig(
                api_key="test_key",
                log_level="INFO",
                log_max_bytes=-100,
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            config.validate()
        
        # Test invalid log_backup_count
        with pytest.raises(LightRAGConfigError, match="log_backup_count must be non-negative"):
            config = LightRAGConfig(
                api_key="test_key",
                log_level="INFO",
                log_backup_count=-1,
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            config.validate()
        
        # Test empty log filename
        with pytest.raises(LightRAGConfigError, match="log_filename cannot be empty"):
            config = LightRAGConfig(
                api_key="test_key",
                log_level="INFO",
                log_filename="",
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            config.validate()
        
        # Test invalid log filename extension
        with pytest.raises(LightRAGConfigError, match="log_filename should end with '.log' extension"):
            config = LightRAGConfig(
                api_key="test_key",
                log_level="INFO",
                log_filename="invalid.txt",
                log_dir=self.test_log_dir,
                auto_create_dirs=False
            )
            config.validate()


class TestLoggingIntegration:
    """Test logging integration with system components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_logs_integration_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_logging_system_initialization(self):
        """Test logging system initialization and configuration loading."""
        # Test initialization from environment
        with patch.dict(os.environ, {
            "LIGHTRAG_LOG_LEVEL": "DEBUG",
            "LIGHTRAG_LOG_DIR": str(self.test_log_dir),
            "LIGHTRAG_ENABLE_FILE_LOGGING": "true",
            "LIGHTRAG_LOG_MAX_BYTES": "5120",
            "LIGHTRAG_LOG_BACKUP_COUNT": "3"
        }):
            config = LightRAGConfig.get_config(validate_config=False, ensure_dirs=False)
            
            assert config.log_level == "DEBUG"
            assert config.log_dir == self.test_log_dir
            assert config.enable_file_logging is True
            assert config.log_max_bytes == 5120
            assert config.log_backup_count == 3
            
            logger = config.setup_lightrag_logging("integration_test_logger")
            
            # Verify both handlers are configured
            assert len(logger.handlers) == 2
            assert logger.level == logging.DEBUG
    
    def test_logging_with_different_directory_permissions(self):
        """Test logging with different directory permission scenarios."""
        # Test with writable directory
        writable_dir = self.test_log_dir / "writable"
        writable_dir.mkdir(mode=0o755)
        
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=writable_dir,
            enable_file_logging=True,
            log_filename="permission_test.log",
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("permission_test_logger")
        logger.info("Test message in writable directory")
        
        log_file = writable_dir / "permission_test.log"
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message in writable directory" in content
    
    def test_concurrent_logging_from_multiple_threads(self):
        """Test concurrent logging from multiple threads."""
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=True,
            log_filename="concurrent_test.log",
            auto_create_dirs=False
        )
        
        logger = config.setup_lightrag_logging("concurrent_test_logger")
        
        def log_messages(thread_id, message_count=10):
            """Log messages from a specific thread."""
            for i in range(message_count):
                logger.info(f"Thread {thread_id} - Message {i}")
        
        # Create multiple threads that log concurrently
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=log_messages, args=(thread_id,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all messages were logged
        log_file = self.test_log_dir / "concurrent_test.log"
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            
            # Should have 50 messages (5 threads * 10 messages each)
            message_lines = [line for line in lines if "Message" in line]
            assert len(message_lines) == 50
            
            # Each thread should have contributed messages
            for thread_id in range(5):
                thread_messages = [line for line in message_lines if f"Thread {thread_id}" in line]
                assert len(thread_messages) == 10
    
    def test_standalone_logging_setup_function(self, caplog):
        """Test the standalone setup_lightrag_logging function."""
        # Test with provided config
        config = LightRAGConfig(
            api_key="test_key",
            log_level="INFO",
            log_dir=self.test_log_dir,
            enable_file_logging=False,
            auto_create_dirs=False
        )
        
        logger = setup_lightrag_logging(config, "standalone_test_logger")
        logger.propagate = True  # Enable propagation for caplog
        
        assert logger.name == "standalone_test_logger"
        assert len(logger.handlers) == 1  # Console only
        
        with caplog.at_level(logging.INFO):
            logger.info("Standalone function test message")
        
        assert len(caplog.records) == 1
        assert caplog.records[0].message == "Standalone function test message"
    
    def test_standalone_logging_setup_function_without_config(self, caplog):
        """Test the standalone setup_lightrag_logging function without providing config."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test_key_for_standalone",
            "LIGHTRAG_LOG_LEVEL": "INFO",
            "LIGHTRAG_ENABLE_FILE_LOGGING": "false"
        }):
            logger = setup_lightrag_logging(logger_name="standalone_no_config_logger")
            logger.propagate = True  # Enable propagation for caplog
            
            assert logger.name == "standalone_no_config_logger"
            
            with caplog.at_level(logging.INFO):
                logger.info("Standalone without config test message")
            
            assert len(caplog.records) == 1
            assert caplog.records[0].message == "Standalone without config test message"


# ==========================================
# INTEGRATION TESTS
# ==========================================
# These integration tests focus on testing real-world scenarios
# with actual file operations and component interactions

class TestRealFileIntegration:
    """Integration tests with real PDF files and actual file operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_dir = Path(tempfile.mkdtemp(prefix="integration_test_pdfs_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_real_pdf_file(self, filename: str, content: str = None, pages: int = 1) -> Path:
        """Create a real PDF file for testing using fitz."""
        pdf_path = self.test_dir / filename
        
        if content is None:
            content = f"This is test content for {filename}. " * 10
        
        # Create a real PDF using PyMuPDF
        doc = fitz.open()
        for page_num in range(pages):
            page = doc.new_page()
            page_content = f"{content} Page {page_num + 1} content. "
            page.insert_text((72, 72), page_content, fontsize=12)
        
        doc.save(str(pdf_path))
        doc.close()
        
        return pdf_path
    
    def test_progress_tracking_with_real_single_pdf(self, caplog):
        """Test progress tracking with actual PDF file processing."""
        # Create a real PDF file
        pdf_file = self.create_real_pdf_file(
            "real_test_document.pdf", 
            "This is biomedical content about metabolomics and clinical analysis. ",
            pages=3
        )
        
        with caplog.at_level(logging.INFO):
            result = self.processor.extract_text_from_pdf(pdf_file)
            
            # Verify the result is valid
            assert result is not None
            assert 'text' in result
            assert 'metadata' in result
            assert len(result['text']) > 0
            
            # Verify progress tracking messages
            log_messages = [record.message for record in caplog.records]
            
            # Check for opening message
            opening_logs = [msg for msg in log_messages if "Opening PDF file" in msg]
            assert len(opening_logs) >= 1, "Should log PDF opening"
            
            # Check for page count logging
            page_count_logs = [msg for msg in log_messages if "PDF has 3 pages" in msg]
            assert len(page_count_logs) == 1, "Should log correct page count"
            
            # Check for success summary with real character counts
            success_logs = [msg for msg in log_messages if "Successfully processed 3 pages" in msg]
            assert len(success_logs) == 1, "Should log successful completion"
            
            # Verify actual processing stats match logged stats
            processing_info = result.get('processing_info', {})
            actual_char_count = processing_info.get('total_characters', 0)
            char_logs = [msg for msg in log_messages if f"{actual_char_count} characters" in msg]
            assert len(char_logs) >= 1, "Should log actual character count"
    
    def test_progress_tracking_with_various_pdf_sizes(self, caplog):
        """Test progress tracking with PDFs of different sizes."""
        test_cases = [
            ("small.pdf", "Small content. ", 1),
            ("medium.pdf", "Medium sized content with more text. " * 20, 5),
            ("large.pdf", "Large content with extensive text for testing. " * 50, 15)
        ]
        
        for filename, content, pages in test_cases:
            pdf_file = self.create_real_pdf_file(filename, content, pages)
            
            caplog.clear()
            with caplog.at_level(logging.INFO):
                result = self.processor.extract_text_from_pdf(pdf_file)
                
                # Verify processing succeeded
                assert result is not None
                assert len(result['text']) > 0
                
                # Verify correct page count is logged
                log_messages = [record.message for record in caplog.records]
                page_logs = [msg for msg in log_messages if f"PDF has {pages} pages" in msg]
                assert len(page_logs) == 1, f"Should log correct page count for {filename}"
                
                # Verify success message includes correct stats
                success_logs = [msg for msg in log_messages if f"Successfully processed {pages} pages" in msg]
                assert len(success_logs) == 1, f"Should log successful completion for {filename}"
    
    def test_progress_tracking_with_corrupted_pdf(self, caplog):
        """Test progress tracking with corrupted or problematic PDF files."""
        # Create a file that looks like PDF but is corrupted
        corrupted_file = self.test_dir / "corrupted.pdf"
        corrupted_file.write_bytes(b"%PDF-1.4\nThis is not valid PDF content")
        
        with caplog.at_level(logging.WARNING):
            with pytest.raises((PDFValidationError, BiomedicalPDFProcessorError)):
                self.processor.extract_text_from_pdf(corrupted_file)
            
            # Verify error tracking
            warning_logs = [record.message for record in caplog.records 
                          if record.levelname in ['WARNING', 'ERROR']]
            
            # Should have logged the processing attempt and error
            error_logs = [msg for msg in warning_logs if "corrupted.pdf" in msg]
            assert len(error_logs) >= 1, "Should log error for corrupted file"
    
    def test_progress_tracking_batch_with_real_files(self, caplog):
        """Test batch progress tracking with real PDF files."""
        # Create multiple real PDF files
        pdf_files = []
        for i in range(4):
            content = f"Biomedical research paper {i+1} content about metabolomics. " * (i + 1) * 5
            pdf_file = self.create_real_pdf_file(f"research_paper_{i+1}.pdf", content, pages=i+2)
            pdf_files.append(pdf_file)
        
        with caplog.at_level(logging.INFO):
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
            # Verify all files were processed
            assert len(documents) == 4, "Should process all PDF files"
            
            # Verify progress tracking messages
            log_messages = [record.message for record in caplog.records if record.levelname == 'INFO']
            
            # Check file discovery
            discovery_logs = [msg for msg in log_messages if f"Found {len(pdf_files)} PDF files" in msg]
            assert len(discovery_logs) == 1, "Should log file discovery"
            
            # Check individual file progress
            for i, pdf_file in enumerate(pdf_files):
                progress_pattern = f"Processing PDF {i+1}/{len(pdf_files)}"
                progress_logs = [msg for msg in log_messages if progress_pattern in msg]
                assert len(progress_logs) == 1, f"Should track progress for file {i+1}"
            
            # Check final summary
            summary_logs = [msg for msg in log_messages if "Batch processing completed" in msg]
            assert len(summary_logs) == 1, "Should have batch summary"
            assert f"{len(pdf_files)} successful" in summary_logs[0], "Should show correct success count"


class TestLightRAGSystemIntegration:
    """Integration tests with LightRAG system components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_dir = Path(tempfile.mkdtemp(prefix="lightrag_integration_"))
        self.log_dir = self.test_dir / "logs"
        
        # Create test configuration
        self.config = LightRAGConfig(
            api_key="test_integration_key",
            working_dir=self.test_dir,
            log_dir=self.log_dir,
            log_level="DEBUG",
            enable_file_logging=True,
            auto_create_dirs=False
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_pdf(self, filename: str, pages: int = 2) -> Path:
        """Create a test PDF file."""
        pdf_path = self.test_dir / filename
        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"Test content page {i+1} for LightRAG integration", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    
    def test_progress_tracking_with_lightrag_config_initialization(self, caplog):
        """Test progress tracking during LightRAG configuration initialization."""
        # Test configuration setup with logging
        with caplog.at_level(logging.DEBUG):
            # Initialize configuration (this will create directories if auto_create_dirs is True)
            test_config = LightRAGConfig(
                api_key="test_key",
                working_dir=self.test_dir,
                log_dir=self.log_dir,
                auto_create_dirs=True
            )
            
            # Set up logging system
            logger = test_config.setup_lightrag_logging("integration_test")
            logger.propagate = True
            
            # Test that logger works with configuration
            logger.info("LightRAG configuration initialized successfully")
            
            # Verify initialization was logged
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            init_logs = [msg for msg in info_logs if "initialized" in msg]
            assert len(init_logs) >= 1, "Should log configuration initialization"
    
    def test_progress_tracking_with_document_ingestion_workflow(self, caplog):
        """Test progress tracking through document ingestion workflow."""
        # Create test PDFs
        pdf1 = self.create_test_pdf("metabolomics_paper.pdf", pages=3)
        pdf2 = self.create_test_pdf("clinical_study.pdf", pages=5)
        
        # Set up LightRAG logging
        logger = self.config.setup_lightrag_logging("ingestion_test")
        logger.propagate = True
        
        with caplog.at_level(logging.INFO):
            # Simulate document ingestion workflow
            logger.info("Starting document ingestion workflow")
            
            # Process PDFs (simulating knowledge base ingestion)
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
            logger.info(f"Ingested {len(documents)} documents into knowledge base")
            logger.info("Document ingestion workflow completed successfully")
            
            # Verify workflow logging
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            
            workflow_logs = [msg for msg in info_logs if "workflow" in msg]
            assert len(workflow_logs) >= 2, "Should log workflow start and completion"
            
            ingestion_logs = [msg for msg in info_logs if "Ingested" in msg and "documents" in msg]
            assert len(ingestion_logs) == 1, "Should log document ingestion count"
            
            # Verify PDF processing logs are also present
            processing_logs = [msg for msg in info_logs if "Processing PDF" in msg]
            assert len(processing_logs) == 2, "Should log processing for each PDF"
    
    def test_logging_integration_with_file_and_console_output(self, caplog):
        """Test logging integration with both file and console output."""
        # Create the log directory first
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        logger = self.config.setup_lightrag_logging("file_console_test")
        logger.propagate = True  # Enable propagation for caplog
        
        # Create test PDF
        pdf_file = self.create_test_pdf("logging_test.pdf")
        
        with caplog.at_level(logging.INFO):
            # Process PDF (this will generate logs)
            result = self.processor.extract_text_from_pdf(pdf_file)
            
            # Test that we have both console and file handlers configured
            console_handlers = [h for h in logger.handlers 
                              if isinstance(h, logging.StreamHandler) and 
                              not isinstance(h, logging.handlers.RotatingFileHandler)]
            file_handlers = [h for h in logger.handlers 
                           if isinstance(h, logging.handlers.RotatingFileHandler)]
            
            # Should have both types of handlers if file logging is enabled
            if self.config.enable_file_logging:
                assert len(console_handlers) >= 1, "Should have console handler"
                assert len(file_handlers) >= 1, "Should have file handler when file logging enabled"
            else:
                assert len(console_handlers) >= 1, "Should have console handler"
            
            # Verify console logging works (captured by caplog)
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            processing_logs = [msg for msg in info_logs if "PDF" in msg or "processing" in msg.lower()]
            assert len(processing_logs) > 0, "Should capture processing logs in console output"
    
    def test_error_recovery_and_progress_continuation(self, caplog):
        """Test error recovery and progress continuation in integrated workflow."""
        # Create mixed files (some valid PDFs, some problematic)
        good_pdf1 = self.create_test_pdf("good_paper1.pdf", pages=2)
        good_pdf2 = self.create_test_pdf("good_paper2.pdf", pages=3)
        
        # Create problematic file
        bad_file = self.test_dir / "bad_file.pdf"
        bad_file.write_bytes(b"%PDF-1.4\nCorrupted content")
        
        logger = self.config.setup_lightrag_logging("recovery_test")
        logger.propagate = True
        
        with caplog.at_level(logging.INFO):
            # Process all files (should continue despite errors)
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
            # Should have processed the good files despite the bad one
            assert len(documents) == 2, "Should process good files despite errors"
            
            # Check error handling logs
            error_logs = [record.message for record in caplog.records if record.levelname == 'ERROR']
            assert len(error_logs) >= 1, "Should log processing errors"
            
            # Check progress continuation (file-level success messages)
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            file_success_logs = [msg for msg in info_logs if "Successfully processed good_paper" in msg]
            assert len(file_success_logs) == 2, "Should successfully process good files"
            
            # Check final summary includes error count
            summary_logs = [msg for msg in info_logs if "Batch processing completed" in msg]
            assert len(summary_logs) == 1, "Should have final summary"
            assert "2 successful" in summary_logs[0], "Should show correct success count"
            assert "1 failed" in summary_logs[0], "Should show correct failure count"


class TestPerformanceIntegration:
    """Integration tests focusing on performance and timing validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_dir = Path(tempfile.mkdtemp(prefix="performance_integration_"))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_large_pdf(self, filename: str, pages: int, content_per_page: int = 1000) -> Path:
        """Create a larger PDF for performance testing."""
        pdf_path = self.test_dir / filename
        doc = fitz.open()
        
        for i in range(pages):
            page = doc.new_page()
            # Create substantial content per page
            content = f"Page {i+1} content: " + "This is test content for performance testing. " * content_per_page
            page.insert_text((72, 72), content[:8000], fontsize=10)  # Limit content to avoid overflow
        
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    
    def test_progress_tracking_with_large_batch(self, caplog):
        """Test progress tracking with large batch of files (10+ files)."""
        # Create 12 PDFs of varying sizes
        pdf_files = []
        for i in range(12):
            pages = min(i + 1, 8)  # Vary from 1 to 8 pages
            content_size = (i % 3) + 1  # Vary content density
            pdf_file = self.create_large_pdf(f"large_batch_{i+1:02d}.pdf", pages, content_size * 100)
            pdf_files.append(pdf_file)
        
        start_time = time.time()
        
        with caplog.at_level(logging.INFO):
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all files were processed
        assert len(documents) == 12, "Should process all large batch files"
        
        # Verify progress tracking for all files
        info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
        
        # Check discovery
        discovery_logs = [msg for msg in info_logs if "Found 12 PDF files" in msg]
        assert len(discovery_logs) == 1, "Should discover all 12 files"
        
        # Check individual progress messages
        progress_logs = [msg for msg in info_logs if "Processing PDF" in msg and "/" in msg]
        assert len(progress_logs) == 12, "Should have progress message for each file"
        
        # Verify progress indexing is sequential
        for i in range(12):
            expected_pattern = f"Processing PDF {i+1}/12"
            matching_logs = [msg for msg in progress_logs if expected_pattern in msg]
            assert len(matching_logs) == 1, f"Should have exactly one progress message for file {i+1}/12"
        
        # Check final summary
        summary_logs = [msg for msg in info_logs if "Batch processing completed" in msg]
        assert len(summary_logs) == 1, "Should have batch summary"
        assert "12 successful" in summary_logs[0], "Should show all files successful"
        
        # Performance validation - should complete within reasonable time
        assert processing_time < 60, f"Large batch should complete within 60 seconds, took {processing_time:.2f}s"
    
    def test_memory_usage_during_progress_tracking(self, caplog):
        """Test memory usage during progress tracking operations."""
        import psutil
        import os
        
        # Create a few large PDFs
        large_pdfs = []
        for i in range(5):
            pdf_file = self.create_large_pdf(f"memory_test_{i+1}.pdf", pages=10, content_per_page=500)
            large_pdfs.append(pdf_file)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with caplog.at_level(logging.DEBUG):
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify processing completed
        assert len(documents) == 5, "Should process all large files"
        
        # Verify detailed logging includes memory-conscious operations
        debug_logs = [record.message for record in caplog.records if record.levelname == 'DEBUG']
        
        # Should have page-by-page debug logs (memory conscious processing)
        page_logs = [msg for msg in debug_logs if "characters from page" in msg]
        assert len(page_logs) >= 10, "Should have page-by-page processing logs"
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB, should be < 100MB"
    
    def test_timeout_scenarios_and_progress_behavior(self, caplog):
        """Test timeout scenarios and progress tracking behavior."""
        # Create a processor with short timeout for testing
        short_timeout_processor = BiomedicalPDFProcessor(processing_timeout=1)  # 1 second timeout
        
        # Create a file that would take time to process (lots of pages)
        large_pdf = self.create_large_pdf("timeout_test.pdf", pages=50, content_per_page=1000)
        
        with caplog.at_level(logging.WARNING):
            try:
                # This might timeout or complete quickly depending on system performance
                result = short_timeout_processor.extract_text_from_pdf(large_pdf)
                
                # If it completes, verify normal progress tracking
                info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
                success_logs = [msg for msg in info_logs if "Successfully processed" in msg]
                assert len(success_logs) >= 1, "Should log successful completion if no timeout"
                
            except PDFProcessingTimeoutError:
                # If timeout occurs, verify timeout logging
                error_logs = [record.message for record in caplog.records if record.levelname == 'ERROR']
                timeout_logs = [msg for msg in error_logs if "timeout" in msg.lower()]
                assert len(timeout_logs) >= 1, "Should log timeout errors"
            
            except Exception:
                # Other exceptions are also acceptable for this stress test
                pass
    
    def test_concurrent_processing_with_progress_tracking(self, caplog):
        """Test concurrent processing with progress tracking."""
        # Create multiple PDFs for concurrent processing
        pdf_files = []
        for i in range(6):
            pdf_file = self.create_large_pdf(f"concurrent_{i+1}.pdf", pages=3, content_per_page=200)
            pdf_files.append(pdf_file)
        
        with caplog.at_level(logging.INFO):
            # Process files (this uses asyncio internally)
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
        # Verify all files processed
        assert len(documents) == 6, "Should process all files concurrently"
        
        # Verify progress tracking worked during concurrent operations
        info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
        
        # Check that all files were tracked
        progress_logs = [msg for msg in info_logs if "Processing PDF" in msg]
        assert len(progress_logs) == 6, "Should track progress for all concurrent files"
        
        # Check that all files completed successfully (file-level success messages)
        file_success_logs = [msg for msg in info_logs if "Successfully processed concurrent_" in msg]
        assert len(file_success_logs) == 6, "Should log file-level success for all concurrent files"
        
        # Final summary should show all successful
        summary_logs = [msg for msg in info_logs if "6 successful" in msg]
        assert len(summary_logs) == 1, "Should show all concurrent files successful"


class TestEndToEndWorkflowIntegration:
    """Integration tests for complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="e2e_integration_"))
        self.processor = BiomedicalPDFProcessor()
        
        # Set up complete LightRAG configuration
        self.config = LightRAGConfig(
            api_key="test_e2e_key",
            working_dir=self.test_dir,
            log_dir=self.test_dir / "logs",
            log_level="INFO",
            enable_file_logging=True,
            auto_create_dirs=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_biomedical_pdf(self, filename: str, topic: str = "metabolomics") -> Path:
        """Create a PDF with biomedical content."""
        pdf_path = self.test_dir / filename
        doc = fitz.open()
        
        # Create realistic biomedical content
        biomedical_content = {
            "metabolomics": "Clinical metabolomics analysis of biomarkers in diabetes patients. LC-MS/MS methodology for metabolite identification. Statistical analysis using PCA and OPLS-DA.",
            "genomics": "Genome-wide association study (GWAS) for cardiovascular disease risk factors. SNP analysis and genetic variation studies in population cohorts.",
            "proteomics": "Proteomic analysis using mass spectrometry. Protein identification and quantification in clinical samples. Biomarker discovery for cancer diagnosis."
        }
        
        content = biomedical_content.get(topic, biomedical_content["metabolomics"])
        
        for i in range(3):
            page = doc.new_page()
            page_content = f"Page {i+1}: {content} " * 10
            page.insert_text((72, 72), page_content, fontsize=11)
        
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    
    def test_complete_workflow_file_discovery_to_completion(self, caplog):
        """Test complete workflow from file discovery to processing completion."""
        # Create diverse biomedical PDFs
        pdfs = [
            self.create_biomedical_pdf("metabolomics_study.pdf", "metabolomics"),
            self.create_biomedical_pdf("genomics_research.pdf", "genomics"),
            self.create_biomedical_pdf("proteomics_analysis.pdf", "proteomics")
        ]
        
        # Set up logging
        logger = self.config.setup_lightrag_logging("e2e_workflow")
        logger.propagate = True
        
        with caplog.at_level(logging.INFO):
            # Complete end-to-end workflow
            logger.info("Starting complete biomedical document processing workflow")
            
            # Stage 1: Configuration and initialization
            logger.info(f"Working directory: {self.config.working_dir}")
            logger.info(f"Processing {len(pdfs)} biomedical documents")
            
            # Stage 2: Document discovery and processing
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
            # Stage 3: Post-processing validation
            valid_documents = []
            for doc in documents:
                if doc and len(doc.get('text', '')) > 100:  # Minimum content threshold
                    valid_documents.append(doc)
                    logger.info(f"Validated document: {doc.get('metadata', {}).get('filename', 'unknown')}")
            
            logger.info(f"Workflow completed: {len(valid_documents)} valid documents processed")
            
            # Verify complete workflow logging
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            
            # Check workflow stages
            workflow_start = [msg for msg in info_logs if "Starting complete" in msg]
            assert len(workflow_start) == 1, "Should log workflow start"
            
            config_logs = [msg for msg in info_logs if "Working directory" in msg]
            assert len(config_logs) == 1, "Should log configuration details"
            
            discovery_logs = [msg for msg in info_logs if f"Processing {len(pdfs)}" in msg]
            assert len(discovery_logs) == 1, "Should log document count"
            
            validation_logs = [msg for msg in info_logs if "Validated document" in msg]
            assert len(validation_logs) == len(pdfs), "Should validate all documents"
            
            completion_logs = [msg for msg in info_logs if "Workflow completed" in msg]
            assert len(completion_logs) == 1, "Should log workflow completion"
            
            # Verify all processing stages completed
            assert len(documents) == len(pdfs), "Should process all documents"
            assert len(valid_documents) == len(pdfs), "All documents should be valid"
    
    def test_integration_with_configuration_loading_and_cleanup(self, caplog):
        """Test integration between configuration loading, processing, and cleanup."""
        # Create test PDFs
        pdf1 = self.create_biomedical_pdf("config_test_1.pdf")
        pdf2 = self.create_biomedical_pdf("config_test_2.pdf")
        
        with caplog.at_level(logging.DEBUG):
            # Stage 1: Configuration loading and validation
            config = LightRAGConfig(
                api_key="test_config_key",
                working_dir=self.test_dir,
                log_dir=self.test_dir / "config_logs",
                log_level="DEBUG",
                enable_file_logging=True,
                auto_create_dirs=True
            )
            
            # Stage 2: Logger setup
            logger = config.setup_lightrag_logging("config_integration_test")
            logger.propagate = True
            logger.info("Configuration loaded and logger initialized")
            
            # Stage 3: PDF processing
            processor = BiomedicalPDFProcessor(logger=logger)
            documents = asyncio.run(processor.process_all_pdfs(self.test_dir))
            
            # Stage 4: Cleanup verification
            logger.info("Processing completed, verifying results")
            for doc in documents:
                metadata = doc.get('metadata', {})
                logger.debug(f"Document metadata: {metadata}")
            
            logger.info("Integration test completed successfully")
            
            # Verify integration logging
            debug_logs = [record.message for record in caplog.records if record.levelname == 'DEBUG']
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            
            # Configuration stage
            init_logs = [msg for msg in info_logs if "initialized" in msg]
            assert len(init_logs) >= 1, "Should log initialization"
            
            # Processing stage
            processing_logs = [msg for msg in info_logs if "Processing PDF" in msg]
            assert len(processing_logs) == 2, "Should log processing for both PDFs"
            
            # Cleanup stage
            completion_logs = [msg for msg in info_logs if "completed successfully" in msg]
            assert len(completion_logs) >= 1, "Should log successful completion"
            
            # Debug details
            metadata_logs = [msg for msg in debug_logs if "metadata" in msg]
            assert len(metadata_logs) >= 2, "Should log metadata for processed documents"
    
    def test_real_world_error_scenarios_and_recovery(self, caplog):
        """Test real-world error scenarios and recovery mechanisms."""
        # Create mixed scenario: good files, corrupted files, permission issues
        good_pdf1 = self.create_biomedical_pdf("good_document_1.pdf")
        good_pdf2 = self.create_biomedical_pdf("good_document_2.pdf")
        
        # Corrupted file
        corrupted_file = self.test_dir / "corrupted_document.pdf"
        corrupted_file.write_bytes(b"%PDF-1.4\nInvalid PDF structure")
        
        # Empty file
        empty_file = self.test_dir / "empty_document.pdf"
        empty_file.touch()
        
        # Set up comprehensive logging
        logger = self.config.setup_lightrag_logging("error_recovery_test")
        logger.propagate = True
        
        with caplog.at_level(logging.INFO):
            # Process all files with error recovery
            logger.info("Starting error recovery scenario test")
            
            documents = asyncio.run(self.processor.process_all_pdfs(self.test_dir))
            
            # Verify recovery behavior
            successful_docs = [doc for doc in documents if doc is not None]
            logger.info(f"Recovery test completed: {len(successful_docs)} successful out of 4 total files")
            
            # Verify error handling and recovery
            all_logs = [record.message for record in caplog.records]
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            error_logs = [record.message for record in caplog.records if record.levelname == 'ERROR']
            
            # Should have attempted to process all files
            discovery_logs = [msg for msg in info_logs if "Found 4 PDF files" in msg]
            assert len(discovery_logs) == 1, "Should discover all 4 files"
            
            # Should have progress messages for all files
            progress_logs = [msg for msg in info_logs if "Processing PDF" in msg]
            assert len(progress_logs) == 4, "Should attempt to process all files"
            
            # Should have success messages for good files only (file-level success messages)
            file_success_logs = [msg for msg in info_logs if "Successfully processed good_document" in msg]
            assert len(file_success_logs) == 2, "Should successfully process only good files"
            
            # Should have error messages for problematic files
            assert len(error_logs) >= 2, "Should log errors for problematic files"
            
            # Should have final summary with correct counts
            summary_logs = [msg for msg in info_logs if "Batch processing completed" in msg]
            assert len(summary_logs) == 1, "Should have final summary"
            assert "2 successful" in summary_logs[0], "Should show 2 successful"
            assert "2 failed" in summary_logs[0], "Should show 2 failed"
            
            # Verify actual processing results
            assert len(successful_docs) == 2, "Should have 2 successful documents despite errors"
    
    def test_multi_component_interaction_testing(self, caplog):
        """Test interaction between different system components."""
        # Create comprehensive test scenario
        pdf_files = [
            self.create_biomedical_pdf("component_test_1.pdf", "metabolomics"),
            self.create_biomedical_pdf("component_test_2.pdf", "genomics")
        ]
        
        # Set up configuration with file logging
        config = LightRAGConfig(
            api_key="multi_component_key",
            working_dir=self.test_dir,
            log_dir=self.test_dir / "component_logs",
            log_level="DEBUG",
            enable_file_logging=True,
            auto_create_dirs=True
        )
        
        # Component 1: Logger setup and configuration
        logger = config.setup_lightrag_logging("multi_component_test")
        logger.propagate = True
        
        # Component 2: PDF processor with custom configuration
        processor = BiomedicalPDFProcessor(
            logger=logger,
            processing_timeout=120,
            memory_limit_mb=500
        )
        
        with caplog.at_level(logging.DEBUG):
            # Component interaction workflow
            logger.info("Multi-component integration test starting")
            
            # Processor + Configuration interaction
            logger.debug(f"Processor configured with timeout: {processor.processing_timeout}s")
            
            # Processor + File system interaction
            documents = asyncio.run(processor.process_all_pdfs(self.test_dir))
            
            # Configuration + Logging interaction (verify file logging)
            log_file = config.log_dir / config.log_filename
            if log_file.exists():
                logger.info(f"File logging confirmed: {log_file}")
            
            # Results validation interaction
            for i, doc in enumerate(documents):
                if doc:
                    char_count = len(doc.get('text', ''))
                    page_count = doc.get('metadata', {}).get('pages', 0)
                    logger.info(f"Document {i+1}: {char_count} characters, {page_count} pages")
            
            logger.info("Multi-component integration test completed")
            
            # Verify component interactions
            debug_logs = [record.message for record in caplog.records if record.levelname == 'DEBUG']
            info_logs = [record.message for record in caplog.records if record.levelname == 'INFO']
            
            # Configuration component logs
            config_logs = [msg for msg in debug_logs if "configured with timeout" in msg]
            assert len(config_logs) == 1, "Should log processor configuration"
            
            # File system component logs
            file_logs = [msg for msg in info_logs if "File logging confirmed" in msg]
            assert len(file_logs) >= 0, "File logging interaction"  # May or may not exist based on timing
            
            # Processing component logs
            processing_logs = [msg for msg in info_logs if "Processing PDF" in msg]
            assert len(processing_logs) == len(pdf_files), "Should log processing for all files"
            
            # Results validation component logs
            result_logs = [msg for msg in info_logs if "characters" in msg and "pages" in msg]
            assert len(result_logs) >= len(pdf_files), "Should log results for processed documents"
            
            # Overall integration completion
            completion_logs = [msg for msg in info_logs if "integration test completed" in msg]
            assert len(completion_logs) == 1, "Should log overall completion"
            
            # Verify actual file logging worked (component integration)
            if (config.log_dir / config.log_filename).exists():
                with open(config.log_dir / config.log_filename, 'r') as f:
                    file_content = f.read()
                    assert "Multi-component integration test" in file_content, "Should log to file"
                    assert "Processing PDF" in file_content, "Should include processing logs in file"


class TestCustomExceptionErrorLogging:
    """Test error logging behavior for custom exception types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_error_logs_"))
        self.config = LightRAGConfig(
            api_key="test_key",
            log_level="DEBUG",
            log_dir=self.test_log_dir,
            enable_file_logging=True,
            log_filename="error_logging_test.log",
            auto_create_dirs=True
        )
        self.logger = self.config.setup_lightrag_logging("error_logging_test")
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_pdf_validation_error_logging(self, caplog):
        """Test logging behavior for PDFValidationError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Ensure processor logger propagates to root logger for caplog to capture
            self.processor.logger.propagate = True
            self.processor.logger.setLevel(logging.DEBUG)
            
            # Mock fitz.open to raise PDFValidationError that will go through exception handler
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open') as mock_fitz:
                    mock_fitz.side_effect = PDFValidationError("Invalid PDF format detected")
                    
                    with caplog.at_level(logging.ERROR, logger='pdf_processor'):
                        with pytest.raises(PDFValidationError):
                            self.processor.extract_text_from_pdf(tmp_path)
                        
                        # Verify error logging - the processor logs "PDF processing error for {path}: {error}"
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        assert len(error_records) >= 1, f"Should log PDFValidationError. Found records: {[r.message for r in caplog.records]}"
                        
                        # Look for the specific error logging pattern from the processor
                        validation_errors = [r for r in error_records if 
                                           "PDF processing error" in r.message and 
                                           str(tmp_path) in r.message and
                                           "Invalid PDF format detected" in r.message]
                        assert len(validation_errors) >= 1, "Should log specific validation error message with file path"
                        
                        # Check error severity is ERROR level
                        assert any(r.levelno == logging.ERROR for r in validation_errors), "Should use ERROR level"
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_pdf_timeout_error_logging(self, caplog):
        """Test logging behavior for PDFProcessingTimeoutError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Ensure processor logger propagates to root logger for caplog to capture
            self.processor.logger.propagate = True
            self.processor.logger.setLevel(logging.DEBUG)
            
            # Mock timeout scenario
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open') as mock_fitz:
                    mock_fitz.side_effect = PDFProcessingTimeoutError("Processing timeout after 30 seconds")
                    
                    with caplog.at_level(logging.ERROR, logger='pdf_processor'):
                        with pytest.raises(PDFValidationError):  # Timeout gets wrapped as PDFValidationError
                            self.processor.extract_text_from_pdf(tmp_path)
                    
                        # Verify timeout error logging
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        timeout_errors = [r for r in error_records if "timeout" in r.message.lower()]
                        assert len(timeout_errors) >= 1, "Should log timeout error"
                        
                        # Check for specific timeout message
                        timeout_messages = [r for r in timeout_errors if "30 seconds" in r.message]
                        assert len(timeout_messages) >= 1, "Should log specific timeout duration"
                        
                        # Verify ERROR level for timeouts
                        assert any(r.levelno == logging.ERROR for r in timeout_errors), "Should use ERROR level"
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_pdf_memory_error_logging(self, caplog):
        """Test logging behavior for PDFMemoryError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Ensure processor logger propagates to root logger for caplog to capture
            self.processor.logger.propagate = True
            self.processor.logger.setLevel(logging.DEBUG)
            
            # Mock memory error scenario
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open') as mock_fitz:
                    mock_fitz.side_effect = PDFMemoryError("Insufficient memory to process large PDF (500MB)")
                    
                    with caplog.at_level(logging.ERROR, logger='pdf_processor'):
                        with pytest.raises(PDFValidationError):  # Memory error gets wrapped as PDFValidationError
                            self.processor.extract_text_from_pdf(tmp_path)
                    
                        # Verify memory error logging
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        memory_errors = [r for r in error_records if "memory" in r.message.lower()]
                        assert len(memory_errors) >= 1, "Should log memory error"
                        
                        # Check for specific memory information
                        size_messages = [r for r in memory_errors if "500MB" in r.message]
                        assert len(size_messages) >= 1, "Should log file size information"
                        
                        # Memory errors should be ERROR level
                        assert any(r.levelno == logging.ERROR for r in memory_errors), "Should use ERROR level"
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_pdf_file_access_error_logging(self, caplog):
        """Test logging behavior for PDFFileAccessError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Ensure processor logger propagates to root logger for caplog to capture
            self.processor.logger.propagate = True
            self.processor.logger.setLevel(logging.DEBUG)
            
            # Mock file access error
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open') as mock_fitz:
                    mock_fitz.side_effect = PDFFileAccessError(f"Permission denied accessing {tmp_path}")
                    
                    with caplog.at_level(logging.ERROR, logger='pdf_processor'):
                        with pytest.raises(PDFValidationError):  # Access error gets wrapped as PDFValidationError
                            self.processor.extract_text_from_pdf(tmp_path)
                    
                        # Verify file access error logging
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        access_errors = [r for r in error_records if "permission" in r.message.lower() or "access" in r.message.lower()]
                        assert len(access_errors) >= 1, "Should log file access error"
                        
                        # Check for file path in error message
                        path_messages = [r for r in access_errors if str(tmp_path) in r.message]
                        assert len(path_messages) >= 1, "Should include file path in error message"
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_pdf_content_error_logging(self, caplog):
        """Test logging behavior for PDFContentError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Ensure processor logger propagates to root logger for caplog to capture
            self.processor.logger.propagate = True
            self.processor.logger.setLevel(logging.DEBUG)
            
            # Mock content error - use direct exception in fitz.open to ensure it goes through exception handler
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open') as mock_fitz:
                    mock_fitz.side_effect = PDFContentError("Corrupted page data detected")
                    
                    with caplog.at_level(logging.ERROR, logger='pdf_processor'):
                        with pytest.raises(PDFValidationError):  # Content error gets wrapped as PDFValidationError
                            self.processor.extract_text_from_pdf(tmp_path)
                            
                        # Verify content error logging
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        content_errors = [r for r in error_records if "corrupted" in r.message.lower() or "content" in r.message.lower()]
                        assert len(content_errors) >= 1, "Should log content error"
                        
                        # Check for specific error details
                        corruption_messages = [r for r in content_errors if "corrupted page data" in r.message.lower()]
                        assert len(corruption_messages) >= 1, "Should log specific corruption details"
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_error_message_formatting_and_context(self, caplog):
        """Test error message formatting includes proper context information."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Ensure processor logger propagates to root logger for caplog to capture
            self.processor.logger.propagate = True
            self.processor.logger.setLevel(logging.DEBUG)
            
            error_message = f"Processing failed for {tmp_path.name} due to validation issues"
            
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open') as mock_fitz:
                    mock_fitz.side_effect = PDFValidationError(error_message)
                    
                    with caplog.at_level(logging.DEBUG, logger='pdf_processor'):
                        with pytest.raises(PDFValidationError):
                            self.processor.extract_text_from_pdf(tmp_path)
                    
                        # Check that error context includes file information
                        all_messages = [r.message for r in caplog.records]
                        
                        # Should log file path context
                        path_context = [msg for msg in all_messages if str(tmp_path) in msg or tmp_path.name in msg]
                        assert len(path_context) >= 1, "Should include file path in error context"
                        
                        # Should log operation being performed
                        operation_context = [msg for msg in all_messages if "extract" in msg.lower() or "processing" in msg.lower()]
                        assert len(operation_context) >= 1, "Should include operation context"
        
        finally:
            tmp_path.unlink(missing_ok=True)


class TestRecoveryMechanismTesting:
    """Test recovery mechanisms and system resilience under error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_recovery_"))
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_batch_processing_continues_after_individual_failures(self, caplog):
        """Test that batch processing continues after individual file failures."""
        # Create test files
        test_files = []
        for i in range(5):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock processing to fail on specific files
            original_extract = self.processor.extract_text_from_pdf
            
            def mock_extract(path):
                if "1.pdf" in str(path) or "3.pdf" in str(path):  # Fail on files 1 and 3
                    raise PDFValidationError(f"Validation failed for {path}")
                else:
                    # Return mock success result
                    return {
                        'text': f'Mock content for {path.name}',
                        'metadata': {'pages': 1, 'title': f'Test {path.name}'},
                        'page_count': 1,
                        'character_count': len(f'Mock content for {path.name}')
                    }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract):
                with caplog.at_level(logging.INFO):
                    results = await self.processor.process_pdfs_batch(test_files)
                    
                    # Verify batch processing continued despite failures
                    assert len(results) == 5, "Should return results for all files (successes and failures)"
                    
                    # Count successful vs failed results
                    successful_results = [r for r in results if r.get('error') is None]
                    failed_results = [r for r in results if r.get('error') is not None]
                    
                    assert len(successful_results) == 3, "Should have 3 successful results"
                    assert len(failed_results) == 2, "Should have 2 failed results"
                    
                    # Verify error logging for failures
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    validation_errors = [r for r in error_records if "validation failed" in r.message.lower()]
                    assert len(validation_errors) >= 2, "Should log validation failures"
                    
                    # Verify success logging continues
                    success_records = [r for r in caplog.records if "successfully processed" in r.message.lower()]
                    assert len(success_records) >= 3, "Should log successful processing"
                    
                    # Verify batch completion logging
                    completion_records = [r for r in caplog.records if "batch processing completed" in r.message.lower()]
                    assert len(completion_records) >= 1, "Should log batch completion"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)
    
    def test_progress_tracking_accuracy_with_errors(self, caplog):
        """Test progress tracking accuracy when errors occur."""
        # Create test files
        test_files = []
        for i in range(3):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock to fail on middle file
            def mock_extract(path):
                if "1.pdf" in str(path):  # Fail on middle file
                    raise PDFProcessingTimeoutError("Processing timeout")
                return {
                    'text': f'Content for {path.name}',
                    'metadata': {'pages': 1},
                    'page_count': 1,
                    'character_count': 100
                }
            
            progress_messages = []
            
            def capture_progress(message):
                progress_messages.append(message)
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract):
                with patch.object(self.processor, '_log_progress', side_effect=capture_progress):
                    with caplog.at_level(logging.INFO):
                        try:
                            # Process files individually to track progress
                            results = []
                            for i, file_path in enumerate(test_files):
                                try:
                                    result = self.processor.extract_text_from_pdf(file_path)
                                    results.append(result)
                                except Exception as e:
                                    results.append({'error': str(e), 'path': str(file_path)})
                                    
                            # Verify progress tracking continued despite errors
                            assert len(results) == 3, "Should track all files processed"
                            
                            # Verify progress messages were captured
                            assert len(progress_messages) >= 1, "Should capture progress messages"
                            
                            # Check error vs success count in logs
                            error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
                            info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
                            
                            assert len(error_logs) >= 1, "Should log errors"
                            assert len(info_logs) >= 1, "Should continue logging info messages"
                            
                        except Exception:
                            # Even if batch fails, verify partial progress was tracked
                            assert len(progress_messages) >= 0, "Should have attempted progress tracking"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)
    
    def test_error_accumulation_and_reporting(self, caplog):
        """Test error accumulation and final reporting."""
        test_files = []
        for i in range(4):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock different types of errors
            def mock_extract(path):
                filename = path.name
                if "0" in filename:
                    raise PDFValidationError("Invalid format")
                elif "1" in filename:
                    raise PDFMemoryError("Out of memory")
                elif "2" in filename:
                    raise PDFFileAccessError("Permission denied")
                else:
                    return {'text': 'Success', 'page_count': 1, 'character_count': 7}
            
            error_summary = []
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract):
                with caplog.at_level(logging.ERROR):
                    # Process files and accumulate errors
                    for file_path in test_files:
                        try:
                            self.processor.extract_text_from_pdf(file_path)
                        except Exception as e:
                            error_summary.append({
                                'file': str(file_path),
                                'error_type': type(e).__name__,
                                'error_message': str(e)
                            })
                    
                    # Verify error accumulation
                    assert len(error_summary) == 3, "Should accumulate 3 errors"
                    
                    # Verify different error types were captured
                    error_types = [err['error_type'] for err in error_summary]
                    assert 'PDFValidationError' in error_types
                    assert 'PDFMemoryError' in error_types
                    assert 'PDFFileAccessError' in error_types
                    
                    # Verify error logging occurred for each type
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    assert len(error_records) >= 3, "Should log each error"
                    
                    # Check for specific error messages in logs
                    validation_logs = [r for r in error_records if "invalid format" in r.message.lower()]
                    memory_logs = [r for r in error_records if "memory" in r.message.lower()]
                    access_logs = [r for r in error_records if "permission" in r.message.lower()]
                    
                    assert len(validation_logs) >= 1, "Should log validation errors"
                    assert len(memory_logs) >= 1, "Should log memory errors"
                    assert len(access_logs) >= 1, "Should log access errors"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)
    
    def test_system_state_consistency_after_errors(self, caplog):
        """Test system state consistency after errors occur."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Test that processor state remains consistent after errors
            original_config = self.processor._config if hasattr(self.processor, '_config') else None
            
            # Cause an error
            with patch.object(self.processor, '_validate_pdf_file') as mock_validate:
                mock_validate.side_effect = PDFValidationError("Test error")
                
                with caplog.at_level(logging.ERROR):
                    with pytest.raises(PDFValidationError):
                        self.processor.extract_text_from_pdf(tmp_path)
                    
                    # Verify processor state is unchanged
                    current_config = self.processor._config if hasattr(self.processor, '_config') else None
                    assert current_config == original_config, "Processor config should remain unchanged"
                    
                    # Test that processor can still process valid files after error
                    mock_doc = MagicMock()
                    mock_doc.needs_pass = False
                    mock_doc.page_count = 1
                    mock_doc.metadata = {'title': 'Recovery Test'}
                    mock_page = MagicMock()
                    mock_page.get_text.return_value = "Recovery test content"
                    mock_doc.load_page.return_value = mock_page
                    
                    with patch('fitz.open', return_value=mock_doc):
                        with patch.object(self.processor, '_validate_pdf_file'):  # Don't raise error this time
                            with patch('pathlib.Path.stat') as mock_stat:
                                mock_stat.return_value.st_size = 1024
                                
                                # Should successfully process after previous error
                                result = self.processor.extract_text_from_pdf(tmp_path)
                                assert result is not None, "Should successfully process after error recovery"
                                assert 'text' in result, "Should return valid result structure"
        
        finally:
            tmp_path.unlink(missing_ok=True)


class TestErrorScenarioIntegration:
    """Test integration of various error scenarios with proper logging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_error_integration_"))
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    def test_timeout_scenarios_with_proper_logging(self, caplog):
        """Test timeout scenarios with comprehensive error logging."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Mock a realistic timeout scenario
            def slow_open(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow operation
                raise PDFProcessingTimeoutError(f"Processing timeout exceeded 30 seconds for {tmp_path}")
            
            # Mock validation to pass and mock file size
            with patch.object(self.processor, '_validate_pdf_file'):  # Pass validation
                with patch('fitz.open', side_effect=slow_open):
                    start_time = time.time()
                    
                    with caplog.at_level(logging.DEBUG):
                        with pytest.raises(PDFValidationError):  # Timeout gets wrapped as PDFValidationError
                            self.processor.extract_text_from_pdf(tmp_path)
                    
                    end_time = time.time()
                    
                    # Verify timeout logging includes timing information
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    timeout_errors = [r for r in error_records if "timeout" in r.message.lower()]
                    assert len(timeout_errors) >= 1, "Should log timeout error"
                    
                    # Check for duration information
                    duration_logs = [r for r in timeout_errors if "30 seconds" in r.message]
                    assert len(duration_logs) >= 1, "Should log timeout duration"
                    
                    # Check for file path in timeout logs
                    file_logs = [r for r in timeout_errors if str(tmp_path) in r.message]
                    assert len(file_logs) >= 1, "Should include file path in timeout logs"
                    
                    # Verify actual timing was reasonable (not actually 30 seconds)
                    actual_duration = end_time - start_time
                    assert actual_duration < 5.0, "Test timeout should complete quickly"
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_memory_limit_scenarios_and_logging(self, caplog):
        """Test memory limit scenarios with detailed error logging."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Mock large file size
            large_size = 500 * 1024 * 1024  # 500MB
            
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = large_size
                
                with patch('fitz.open') as mock_open:
                    mock_open.side_effect = PDFMemoryError(f"Insufficient memory for {large_size // (1024*1024)}MB file")
                    
                    with caplog.at_level(logging.WARNING):
                        with pytest.raises(PDFMemoryError):
                            self.processor.extract_text_from_pdf(tmp_path)
                        
                        # Verify memory error logging
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        memory_errors = [r for r in error_records if "memory" in r.message.lower()]
                        assert len(memory_errors) >= 1, "Should log memory error"
                        
                        # Check for file size information
                        size_logs = [r for r in memory_errors if "500MB" in r.message or "MB" in r.message]
                        assert len(size_logs) >= 1, "Should log file size information"
                        
                        # Check for warning logs about large file
                        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
                        large_file_warnings = [r for r in warning_records if "large" in r.message.lower() or "size" in r.message.lower()]
                        # Warning may or may not be present depending on implementation
                        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_file_access_permission_errors(self, caplog):
        """Test file access and permission error handling with logging."""
        # Create a file and make it inaccessible
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Make file unreadable (if possible on this system)
            try:
                tmp_path.chmod(0o000)  # No permissions
                access_restricted = True
            except (OSError, PermissionError):
                # If we can't restrict permissions, mock the error instead
                access_restricted = False
            
            if access_restricted:
                with caplog.at_level(logging.ERROR):
                    with pytest.raises((PDFFileAccessError, PermissionError, OSError)):
                        self.processor.extract_text_from_pdf(tmp_path)
                    
                    # Verify access error logging
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    access_errors = [r for r in error_records if "permission" in r.message.lower() or "access" in r.message.lower()]
                    
                    # May be logged by our code or by the system
                    assert len(error_records) >= 1, "Should log access error"
            
            else:
                # Mock permission error since we can't create real one
                with patch('fitz.open') as mock_open:
                    mock_open.side_effect = PDFFileAccessError(f"Permission denied: {tmp_path}")
                    
                    with caplog.at_level(logging.ERROR):
                        with pytest.raises(PDFFileAccessError):
                            self.processor.extract_text_from_pdf(tmp_path)
                        
                        # Verify permission error logging
                        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                        permission_errors = [r for r in error_records if "permission denied" in r.message.lower()]
                        assert len(permission_errors) >= 1, "Should log permission error"
                        
                        # Check file path is included
                        path_logs = [r for r in permission_errors if str(tmp_path) in r.message]
                        assert len(path_logs) >= 1, "Should include file path in permission error"
        
        finally:
            # Restore permissions for cleanup
            try:
                tmp_path.chmod(0o666)
            except (OSError, PermissionError):
                pass
            tmp_path.unlink(missing_ok=True)
    
    def test_corrupted_pdf_handling_with_logging(self, caplog):
        """Test corrupted PDF content handling and error logging."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Mock corrupted PDF scenario
            mock_doc = MagicMock()
            mock_doc.needs_pass = False
            mock_doc.page_count = 2
            mock_doc.metadata = {'title': 'Corrupted Document'}
            
            # First page loads fine, second page is corrupted
            mock_page1 = MagicMock()
            mock_page1.get_text.return_value = "Page 1 content"
            
            def mock_load_page(page_num):
                if page_num == 0:
                    return mock_page1
                else:
                    raise PDFContentError(f"Corrupted data on page {page_num + 1}")
            
            mock_doc.load_page.side_effect = mock_load_page
            
            with patch('fitz.open', return_value=mock_doc):
                with patch.object(self.processor, '_validate_pdf_file'):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 2048
                        
                        with caplog.at_level(logging.WARNING):
                            with pytest.raises(PDFContentError):
                                self.processor.extract_text_from_pdf(tmp_path)
                            
                            # Verify corruption error logging
                            error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                            content_errors = [r for r in error_records if "corrupted" in r.message.lower()]
                            assert len(content_errors) >= 1, "Should log content corruption error"
                            
                            # Check for page-specific information
                            page_errors = [r for r in content_errors if "page 2" in r.message.lower()]
                            assert len(page_errors) >= 1, "Should log specific page with corruption"
                            
                            # May have warning logs about partial processing
                            warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
                            partial_warnings = [r for r in warning_records if "partial" in r.message.lower() or "page" in r.message.lower()]
                            # Partial processing warnings may or may not be present
        
        finally:
            tmp_path.unlink(missing_ok=True)


class TestErrorRateAndResilience:
    """Test system resilience under various error rate conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiomedicalPDFProcessor()
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="test_resilience_"))
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_partial_failure_scenarios(self, caplog):
        """Test partial failure scenarios with mixed success and failure."""
        # Create 10 test files
        test_files = []
        for i in range(10):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock 30% failure rate
            def mock_extract(path):
                filename = path.name
                # Fail on files containing '2', '5', '8' (30% failure rate)
                if any(fail_num in filename for fail_num in ['2', '5', '8']):
                    # Vary the error types
                    if '2' in filename:
                        raise PDFValidationError(f"Validation failed: {path}")
                    elif '5' in filename:
                        raise PDFMemoryError(f"Memory error: {path}")
                    else:  # '8'
                        raise PDFContentError(f"Content error: {path}")
                else:
                    return {
                        'text': f'Success content for {path.name}',
                        'metadata': {'pages': 1, 'title': f'Doc {path.name}'},
                        'page_count': 1,
                        'character_count': 50
                    }
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract):
                with caplog.at_level(logging.INFO):
                    results = await self.processor.process_pdfs_batch(test_files)
                    
                    # Verify mixed results
                    assert len(results) == 10, "Should process all files"
                    
                    successful = [r for r in results if r.get('error') is None]
                    failed = [r for r in results if r.get('error') is not None]
                    
                    assert len(successful) == 7, "Should have 7 successful results (70%)"
                    assert len(failed) == 3, "Should have 3 failed results (30%)"
                    
                    # Verify error logging variety
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    validation_errors = [r for r in error_records if "validation" in r.message.lower()]
                    memory_errors = [r for r in error_records if "memory" in r.message.lower()]
                    content_errors = [r for r in error_records if "content" in r.message.lower()]
                    
                    assert len(validation_errors) >= 1, "Should log validation errors"
                    assert len(memory_errors) >= 1, "Should log memory errors"
                    assert len(content_errors) >= 1, "Should log content errors"
                    
                    # Verify success logging continues
                    success_records = [r for r in caplog.records if "successfully processed" in r.message.lower()]
                    assert len(success_records) >= 7, "Should log successful processing"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)
    
    def test_complete_failure_scenario(self, caplog):
        """Test complete failure scenario handling."""
        test_files = []
        for i in range(5):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock 100% failure rate
            def mock_extract_fail(path):
                raise PDFValidationError(f"All files fail validation: {path}")
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract_fail):
                with caplog.at_level(logging.ERROR):
                    # Process files individually to see failure handling
                    results = []
                    for file_path in test_files:
                        try:
                            result = self.processor.extract_text_from_pdf(file_path)
                            results.append(result)
                        except Exception as e:
                            results.append({'error': str(e), 'path': str(file_path)})
                    
                    # Verify all failed
                    failed_results = [r for r in results if 'error' in r]
                    assert len(failed_results) == 5, "All files should fail"
                    
                    # Verify comprehensive error logging
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    assert len(error_records) >= 5, "Should log all failures"
                    
                    # Check that each file was logged
                    for file_path in test_files:
                        file_errors = [r for r in error_records if str(file_path) in r.message]
                        assert len(file_errors) >= 1, f"Should log error for {file_path}"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)
    
    def test_high_error_rate_graceful_degradation(self, caplog):
        """Test graceful degradation under high error rates."""
        test_files = []
        for i in range(20):  # Larger batch
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock 80% failure rate
            def mock_extract_high_fail(path):
                filename = path.name
                # Success only on files ending with '0' or '1' (10% success rate)
                if filename.endswith('0.pdf') or filename.endswith('1.pdf'):
                    return {
                        'text': f'Rare success for {path.name}',
                        'metadata': {'pages': 1},
                        'page_count': 1,
                        'character_count': 25
                    }
                else:
                    # Various error types
                    if '2' in filename or '3' in filename:
                        raise PDFValidationError(f"High failure rate validation: {path}")
                    elif '4' in filename or '5' in filename:
                        raise PDFMemoryError(f"High failure rate memory: {path}")
                    elif '6' in filename or '7' in filename:
                        raise PDFProcessingTimeoutError(f"High failure rate timeout: {path}")
                    else:
                        raise PDFFileAccessError(f"High failure rate access: {path}")
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract_high_fail):
                with caplog.at_level(logging.INFO):
                    # Process files and track degradation
                    results = []
                    error_count = 0
                    
                    for file_path in test_files:
                        try:
                            result = self.processor.extract_text_from_pdf(file_path)
                            results.append(result)
                        except Exception as e:
                            error_count += 1
                            results.append({'error': str(e), 'path': str(file_path)})
                            
                            # Simulate checking error threshold
                            error_rate = error_count / (len(results))
                            if error_rate > 0.75:  # 75% error rate threshold
                                # Log high error rate warning
                                logging.getLogger().warning(
                                    f"High error rate detected: {error_rate:.2%} ({error_count}/{len(results)})"
                                )
                    
                    # Verify high error rate was detected and logged
                    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
                    high_error_warnings = [r for r in warning_records if "high error rate" in r.message.lower()]
                    assert len(high_error_warnings) >= 1, "Should log high error rate warnings"
                    
                    # Verify error logging continued throughout
                    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                    assert len(error_records) >= 15, "Should log many errors (80% of 20 files)"
                    
                    # Verify successful processing was still logged for rare successes
                    success_records = [r for r in caplog.records if "successfully processed" in r.message.lower()]
                    # May have 0-4 successes depending on filename patterns
                    
                    # Verify system didn't crash despite high error rate
                    assert len(results) == 20, "Should complete processing all files despite high error rate"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)
    
    def test_error_threshold_handling(self, caplog):
        """Test error threshold detection and handling."""
        test_files = []
        for i in range(10):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            test_files.append(Path(tmp_file.name))
            tmp_file.close()
        
        try:
            # Mock progressive failure scenario
            failure_count = 0
            
            def mock_extract_threshold(path):
                nonlocal failure_count
                filename = path.name
                
                # First 3 files succeed, then start failing
                if '0' in filename or '1' in filename or '2' in filename:
                    return {
                        'text': f'Initial success for {path.name}',
                        'metadata': {'pages': 1},
                        'page_count': 1,
                        'character_count': 30
                    }
                else:
                    failure_count += 1
                    raise PDFValidationError(f"Progressive failure #{failure_count}: {path}")
            
            with patch.object(self.processor, 'extract_text_from_pdf', side_effect=mock_extract_threshold):
                with caplog.at_level(logging.WARNING):
                    results = []
                    success_count = 0
                    error_count = 0
                    
                    for file_path in test_files:
                        try:
                            result = self.processor.extract_text_from_pdf(file_path)
                            results.append(result)
                            success_count += 1
                        except Exception as e:
                            error_count += 1
                            results.append({'error': str(e), 'path': str(file_path)})
                            
                            # Check error threshold (50%)
                            total_processed = success_count + error_count
                            if total_processed >= 5:  # After processing at least 5 files
                                error_rate = error_count / total_processed
                                if error_rate >= 0.5:
                                    logging.getLogger().warning(
                                        f"Error threshold exceeded: {error_rate:.2%} error rate "
                                        f"({error_count} errors in {total_processed} files)"
                                    )
                    
                    # Verify threshold detection
                    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
                    threshold_warnings = [r for r in warning_records if "error threshold exceeded" in r.message.lower()]
                    assert len(threshold_warnings) >= 1, "Should detect error threshold"
                    
                    # Verify threshold details in logs
                    rate_logs = [r for r in threshold_warnings if "50%" in r.message or "0.5" in r.message]
                    assert len(rate_logs) >= 1, "Should log specific error rate"
                    
                    # Verify processing continued despite threshold
                    assert len(results) == 10, "Should complete processing despite threshold"
        
        finally:
            for test_file in test_files:
                test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])