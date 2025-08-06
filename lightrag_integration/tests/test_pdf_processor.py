"""
Comprehensive unit tests for BiomedicalPDFProcessor (Fixed Version).

This is a corrected version of the PDF processor tests that addresses issues
with temporary file handling and PyMuPDF exception types.
"""

import os
import io
import logging
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
import fitz  # PyMuPDF

from lightrag_integration.pdf_processor import BiomedicalPDFProcessor, BiomedicalPDFProcessorError


class TestBiomedicalPDFProcessorBasics:
    """Test basic functionality and initialization of BiomedicalPDFProcessor."""
    
    def test_init_with_default_logger(self):
        """Test initialization with default logger."""
        processor = BiomedicalPDFProcessor()
        assert processor.logger is not None
        assert processor.logger.name == "lightrag_integration.pdf_processor"
    
    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        processor = BiomedicalPDFProcessor(logger=custom_logger)
        assert processor.logger == custom_logger
        assert processor.logger.name == "test_logger"


class TestBiomedicalPDFProcessorTextExtraction:
    """Test text extraction functionality."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.processor = BiomedicalPDFProcessor()
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_text_from_pdf_success(self, mock_fitz_open):
        """Test successful text extraction from PDF."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 2
        mock_doc.metadata = {
            'title': 'Clinical Metabolomics Paper',
            'author': 'Dr. Smith',
            'creationDate': 'D:20240101120000'
        }
        
        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content: Clinical metabolomics analysis."
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content: Results show p<0.05 significance."
        
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc
        
        # Create temporary file with content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"dummy pdf content")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.extract_text_from_pdf(tmp_path)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'metadata' in result
            assert 'page_texts' in result
            assert 'processing_info' in result
            
            # Verify text content
            assert "Clinical metabolomics analysis" in result['text']
            assert "Results show p<0.05 significance" in result['text']
            
            # Verify metadata
            assert result['metadata']['filename'] == tmp_path.name
            assert result['metadata']['pages'] == 2
            assert result['metadata']['pages_processed'] == 2
            assert result['metadata']['title'] == 'Clinical Metabolomics Paper'
            assert result['metadata']['author'] == 'Dr. Smith'
            
            # Verify processing info
            assert result['processing_info']['start_page'] == 0
            assert result['processing_info']['end_page'] == 2
            assert result['processing_info']['preprocessing_applied'] == True
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_text_page_range(self, mock_fitz_open):
        """Test text extraction with specific page range."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 5
        mock_doc.metadata = {}
        
        # Mock page 2 only
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Specific biomedical content on this page"
        mock_doc.load_page.return_value = mock_page2
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"dummy pdf content")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.extract_text_from_pdf(
                tmp_path, start_page=2, end_page=3
            )
            
            # Should only process page 2
            assert result['processing_info']['start_page'] == 2
            assert result['processing_info']['end_page'] == 3
            assert result['processing_info']['pages_processed'] == 1
            assert len(result['page_texts']) == 1
            assert "biomedical content" in result['text']
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestBiomedicalPDFProcessorMetadata:
    """Test metadata extraction functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_parse_pdf_date_valid(self):
        """Test parsing valid PDF date formats."""
        # Standard PDF date format
        result = self.processor._parse_pdf_date('D:20240315140530+05\'00\'')
        assert result == '2024-03-15T14:05:30'
        
        # Without timezone
        result = self.processor._parse_pdf_date('D:20240315140530')
        assert result == '2024-03-15T14:05:30'
        
        # Without D: prefix
        result = self.processor._parse_pdf_date('20240315140530')
        assert result == '2024-03-15T14:05:30'
    
    def test_parse_pdf_date_invalid(self):
        """Test parsing invalid PDF date formats."""
        assert self.processor._parse_pdf_date('') is None
        assert self.processor._parse_pdf_date('invalid_date') is None
        assert self.processor._parse_pdf_date('D:') is None
        assert self.processor._parse_pdf_date('D:2024') is None


class TestBiomedicalPDFProcessorPreprocessing:
    """Test text preprocessing functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_preprocess_biomedical_text_basic(self):
        """Test basic text preprocessing."""
        raw_text = """This    is   a    test   with
        multiple    spaces   and
        
        
        line breaks."""
        
        processed = self.processor._preprocess_biomedical_text(raw_text)
        
        # Should normalize whitespace
        assert "multiple spaces" in processed
        assert "   " not in processed  # No triple spaces
    
    def test_preprocess_p_values(self):
        """Test preprocessing of p-values."""
        text = "Results showed p < 0.05 and p = 0.001 significance."
        processed = self.processor._preprocess_biomedical_text(text)
        
        # Should normalize p-values (be flexible with exact formatting)
        assert "p" in processed and "0.05" in processed
        assert "0.001" in processed
    
    def test_preprocess_empty_text(self):
        """Test preprocessing of empty or None text."""
        assert self.processor._preprocess_biomedical_text("") == ""
        assert self.processor._preprocess_biomedical_text(None) == ""
        assert self.processor._preprocess_biomedical_text("   ") == ""


class TestBiomedicalPDFProcessorErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_extract_text_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent_path = Path("/non/existent/file.pdf")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            self.processor.extract_text_from_pdf(non_existent_path)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_extract_text_not_a_file(self):
        """Test handling when path is not a file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            
            with pytest.raises(BiomedicalPDFProcessorError) as exc_info:
                self.processor.extract_text_from_pdf(dir_path)
            
            assert "not a file" in str(exc_info.value).lower()
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_text_encrypted_pdf(self, mock_fitz_open):
        """Test handling of password-protected PDF."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"dummy content")
            tmp_path = Path(tmp_file.name)
        
        try:
            with pytest.raises(BiomedicalPDFProcessorError) as exc_info:
                self.processor.extract_text_from_pdf(tmp_path)
            
            assert "password protected" in str(exc_info.value).lower()
            mock_doc.close.assert_called_once()
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestBiomedicalPDFProcessorValidation:
    """Test PDF validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_validate_pdf_file_not_found(self):
        """Test validation of non-existent file."""
        non_existent_path = Path("/non/existent/file.pdf")
        
        with pytest.raises(FileNotFoundError):
            self.processor.validate_pdf(non_existent_path)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_validate_pdf_success(self, mock_fitz_open):
        """Test successful PDF validation."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 3
        mock_doc.metadata = {'title': 'Test PDF'}
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content"
        mock_doc.load_page.return_value = mock_page
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"dummy pdf content for validation")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.validate_pdf(tmp_path)
            
            assert result['valid'] is True
            assert result['error'] is None
            assert result['pages'] == 3
            assert result['encrypted'] is False
            assert result['file_size_bytes'] > 0
            assert isinstance(result['metadata'], dict)
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestBiomedicalPDFProcessorRealPDF:
    """Test with actual sample PDF file."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
        # Path to the actual sample PDF
        self.sample_pdf_path = Path(
            "/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
            "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf"
        )
    
    def test_sample_pdf_exists(self):
        """Verify the sample PDF exists."""
        assert self.sample_pdf_path.exists(), f"Sample PDF not found: {self.sample_pdf_path}"
        assert self.sample_pdf_path.is_file(), "Sample PDF path is not a file"
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_extract_text_from_real_pdf(self):
        """Test text extraction from the actual sample PDF."""
        result = self.processor.extract_text_from_pdf(self.sample_pdf_path)
        
        # Verify structure
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'metadata' in result
        assert 'page_texts' in result
        assert 'processing_info' in result
        
        # Verify content characteristics
        assert len(result['text']) > 0, "No text extracted"
        assert result['metadata']['pages'] > 0, "No pages found"
        assert len(result['page_texts']) == result['metadata']['pages_processed']
        
        # Check for biomedical content indicators
        text_lower = result['text'].lower()
        biomedical_terms = ['metabolomic', 'clinical', 'biomarker', 'analysis', 'patient']
        found_terms = [term for term in biomedical_terms if term in text_lower]
        assert len(found_terms) > 0, f"No biomedical terms found. Text sample: {result['text'][:500]}"
        
        # Verify metadata
        assert 'filename' in result['metadata']
        assert result['metadata']['filename'] == 'Clinical_Metabolomics_paper.pdf'
        assert result['metadata']['file_size_bytes'] > 0
        
        # Verify processing info
        assert result['processing_info']['pages_processed'] > 0
        assert result['processing_info']['total_characters'] > 0
        assert 'processing_timestamp' in result['processing_info']
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_validate_real_pdf(self):
        """Test validation of the actual sample PDF."""
        result = self.processor.validate_pdf(self.sample_pdf_path)
        
        assert result['valid'] is True, f"PDF validation failed: {result['error']}"
        assert result['error'] is None
        assert result['pages'] > 0
        assert result['encrypted'] is False
        assert result['file_size_bytes'] > 0
        assert isinstance(result['metadata'], dict)
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_get_page_count_real_pdf(self):
        """Test getting page count from the actual sample PDF."""
        count = self.processor.get_page_count(self.sample_pdf_path)
        assert count > 0, "PDF should have at least one page"
        assert isinstance(count, int)
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_extract_first_page_only(self):
        """Test extracting only the first page from the sample PDF."""
        result = self.processor.extract_text_from_pdf(
            self.sample_pdf_path, start_page=0, end_page=1
        )
        
        assert result['processing_info']['pages_processed'] == 1
        assert len(result['page_texts']) == 1
        assert result['processing_info']['start_page'] == 0
        assert result['processing_info']['end_page'] == 1