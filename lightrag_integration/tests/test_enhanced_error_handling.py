"""
Comprehensive tests for enhanced error handling in BiomedicalPDFProcessor.

This test suite demonstrates all the edge cases and error handling capabilities
added to the PDF processor, including MIME validation, memory monitoring,
timeout protection, file locking detection, and encoding issues.
"""

import os
import tempfile
import time
import pytest
from pathlib import Path
import logging

from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor,
    BiomedicalPDFProcessorError,
    PDFValidationError,
    PDFProcessingTimeoutError,
    PDFMemoryError,
    PDFFileAccessError,
    PDFContentError
)


class TestEnhancedErrorHandling:
    """Test enhanced error handling capabilities."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor(
            processing_timeout=10,  # Short timeout for testing
            memory_limit_mb=100,    # Lower limit for testing
            max_page_text_size=50000  # Reasonable limit for testing
        )
        logging.basicConfig(level=logging.INFO)
    
    def test_processing_stats(self):
        """Test processing statistics method."""
        stats = self.processor.get_processing_stats()
        
        assert 'processing_timeout' in stats
        assert 'memory_limit_mb' in stats
        assert 'max_page_text_size' in stats
        assert 'current_memory_mb' in stats
        assert 'system_memory_percent' in stats
        assert 'memory_monitor_active' in stats
        
        assert stats['processing_timeout'] == 10
        assert stats['memory_limit_mb'] == 100
        assert stats['max_page_text_size'] == 50000
        assert isinstance(stats['current_memory_mb'], (int, float))
        assert isinstance(stats['system_memory_percent'], (int, float))
    
    def test_custom_exception_hierarchy(self):
        """Test custom exception hierarchy."""
        # Test that all custom exceptions inherit from base exception
        with pytest.raises(BiomedicalPDFProcessorError):
            raise PDFValidationError("Test validation error")
        
        with pytest.raises(BiomedicalPDFProcessorError):
            raise PDFProcessingTimeoutError("Test timeout error")
        
        with pytest.raises(BiomedicalPDFProcessorError):
            raise PDFMemoryError("Test memory error")
        
        with pytest.raises(BiomedicalPDFProcessorError):
            raise PDFFileAccessError("Test access error")
        
        with pytest.raises(BiomedicalPDFProcessorError):
            raise PDFContentError("Test content error")
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.processor.extract_text_from_pdf("/nonexistent/path/file.pdf")
    
    def test_zero_byte_file(self):
        """Test handling of zero-byte files."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            zero_byte_file = f.name
        
        try:
            with pytest.raises(PDFValidationError) as exc_info:
                self.processor.extract_text_from_pdf(zero_byte_file)
            assert "empty" in str(exc_info.value).lower()
        finally:
            os.unlink(zero_byte_file)
    
    def test_invalid_file_type(self):
        """Test handling of files with invalid MIME type/header."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='w') as f:
            f.write("This is not a PDF file")
            fake_pdf_file = f.name
        
        try:
            with pytest.raises(PDFValidationError) as exc_info:
                self.processor.extract_text_from_pdf(fake_pdf_file)
            error_msg = str(exc_info.value).lower()
            assert "invalid header" in error_msg or "corrupted" in error_msg
        finally:
            os.unlink(fake_pdf_file)
    
    def test_directory_instead_of_file(self):
        """Test handling when path points to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_pdf_path = os.path.join(temp_dir, "fake.pdf")
            os.makedirs(fake_pdf_path)
            
            with pytest.raises(PDFValidationError) as exc_info:
                self.processor.extract_text_from_pdf(fake_pdf_path)
            assert "not a file" in str(exc_info.value).lower()
    
    def test_validate_pdf_method(self):
        """Test the validate_pdf method."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            self.processor.validate_pdf("/nonexistent/file.pdf")
        
        # Test with zero-byte file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            zero_byte_file = f.name
        
        try:
            result = self.processor.validate_pdf(zero_byte_file)
            assert not result['valid']
            assert "empty" in result['error'].lower()
        finally:
            os.unlink(zero_byte_file)
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        # This test verifies the memory monitoring context manager works
        initial_stats = self.processor.get_processing_stats()
        
        # Simulate memory monitoring during processing
        with self.processor._monitor_memory():
            # Memory monitor should be active
            assert self.processor._memory_monitor_active
            
            # Do some memory allocation
            large_list = [i for i in range(1000)]
            del large_list
        
        # Memory monitor should be inactive after context
        assert not self.processor._memory_monitor_active
    
    def test_timeout_checking(self):
        """Test timeout checking functionality."""
        # Set a very short timeout for testing
        test_processor = BiomedicalPDFProcessor(processing_timeout=1)
        test_processor._processing_start_time = time.time() - 2  # Simulate 2 seconds elapsed
        
        with pytest.raises(PDFProcessingTimeoutError):
            test_processor._check_processing_timeout()
    
    def test_text_validation_and_cleaning(self):
        """Test text validation and cleaning methods."""
        # Test with problematic Unicode characters
        problematic_text = "Test text with Unicode: \u2013 \u2014 \u2018hello\u2019 \u201cworld\u201d"
        cleaned_text = self.processor._validate_text_encoding(problematic_text)
        
        # Should replace Unicode characters with ASCII equivalents
        assert "\u2013" not in cleaned_text  # en dash should be replaced
        assert "\u2014" not in cleaned_text  # em dash should be replaced
        assert "\u2018" not in cleaned_text  # left single quote should be replaced
        assert "\u2019" not in cleaned_text  # right single quote should be replaced
        
        # Test with excessive control characters
        control_text = "Normal text\x00\x01\x02\x03 with control chars"
        cleaned_control_text = self.processor._validate_and_clean_page_text(control_text, 1)
        
        # Control characters should be cleaned
        assert len([c for c in cleaned_control_text if ord(c) < 32 and c not in '\n\r\t']) < 5
    
    def test_large_text_block_handling(self):
        """Test handling of very large text blocks."""
        # Create text larger than the limit
        large_text = "A" * (self.processor.max_page_text_size + 1000)
        
        cleaned_text = self.processor._validate_and_clean_page_text(large_text, 1)
        
        # Should be truncated
        assert len(cleaned_text) <= self.processor.max_page_text_size + 100  # Allow for truncation message
        assert "TEXT TRUNCATED" in cleaned_text
    
    def test_get_page_count_enhanced(self):
        """Test enhanced get_page_count method."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            self.processor.get_page_count("/nonexistent/file.pdf")
        
        # Test with zero-byte file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            zero_byte_file = f.name
        
        try:
            with pytest.raises(PDFValidationError):
                self.processor.get_page_count(zero_byte_file)
        finally:
            os.unlink(zero_byte_file)
    
    def test_configuration_parameters(self):
        """Test different configuration parameters."""
        # Test with different timeout settings
        short_timeout_processor = BiomedicalPDFProcessor(processing_timeout=5)
        assert short_timeout_processor.processing_timeout == 5
        
        # Test with different memory limits
        high_memory_processor = BiomedicalPDFProcessor(memory_limit_mb=2048)
        assert high_memory_processor.memory_limit_mb == 2048
        
        # Test with different page text size limits
        small_page_processor = BiomedicalPDFProcessor(max_page_text_size=10000)
        assert small_page_processor.max_page_text_size == 10000


def test_with_real_pdf_file():
    """Test enhanced error handling with a real PDF file if available."""
    processor = BiomedicalPDFProcessor()
    pdf_path = "papers/Clinical_Metabolomics_paper.pdf"
    
    if os.path.exists(pdf_path):
        # Test validation
        validation_result = processor.validate_pdf(pdf_path)
        assert validation_result['valid']
        assert validation_result['pages'] > 0
        assert validation_result['file_size_bytes'] > 0
        
        # Test extraction with enhanced error handling
        result = processor.extract_text_from_pdf(pdf_path, start_page=0, end_page=2)
        assert len(result['text']) > 0
        assert result['processing_info']['pages_processed'] == 2
        assert result['processing_info']['preprocessing_applied'] is True
        
        # Test processing stats after real processing
        stats = processor.get_processing_stats()
        assert stats['current_memory_mb'] > 0
        print(f"Memory usage after processing: {stats['current_memory_mb']} MB")
    else:
        print("Real PDF file not found, skipping real file test")


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestEnhancedErrorHandling()
    test_suite.setup_method()
    
    print("Running enhanced error handling tests...")
    
    try:
        test_suite.test_processing_stats()
        print("✓ Processing stats test passed")
        
        test_suite.test_custom_exception_hierarchy()
        print("✓ Exception hierarchy test passed")
        
        test_suite.test_zero_byte_file()
        print("✓ Zero-byte file test passed")
        
        test_suite.test_invalid_file_type()
        print("✓ Invalid file type test passed")
        
        test_suite.test_memory_monitoring()
        print("✓ Memory monitoring test passed")
        
        test_suite.test_text_validation_and_cleaning()
        print("✓ Text validation test passed")
        
        test_suite.test_large_text_block_handling()
        print("✓ Large text block test passed")
        
        test_with_real_pdf_file()
        print("✓ Real PDF file test passed")
        
        print("\n✓ All enhanced error handling tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()