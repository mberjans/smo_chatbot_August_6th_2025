"""
Comprehensive unit tests for PDF metadata extraction functionality.

This module contains focused tests for the metadata extraction capabilities
of the BiomedicalPDFProcessor, covering all metadata-related methods and 
edge cases specific to PDF document metadata handling.

Test Categories:
- Basic metadata extraction from PDF documents
- PDF document metadata (title, author, creation date, modification date, subject, creator, producer, keywords)
- PDF date parsing functionality (_parse_pdf_date method)
- File system metadata extraction (size, path, etc.)
- Integration tests with extract_text_from_pdf and validate_pdf methods
- Error handling edge cases and missing metadata scenarios
- Real PDF file metadata extraction tests
- Metadata field type validation and format checking
- Edge cases and boundary conditions

Author: Clinical Metabolomics Oracle Team
Date: August 6th, 2025
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import fitz  # PyMuPDF

from lightrag_integration.pdf_processor import BiomedicalPDFProcessor, BiomedicalPDFProcessorError


class TestPDFMetadataExtractionCore:
    """Test core metadata extraction functionality."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.processor = BiomedicalPDFProcessor()
        
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_metadata_complete_metadata(self, mock_fitz_open):
        """Test extraction of complete PDF metadata with all fields present."""
        # Mock complete PDF metadata
        mock_doc = MagicMock()
        mock_doc.page_count = 15
        mock_doc.metadata = {
            'title': 'Clinical Metabolomics: A Comprehensive Analysis of Biomarkers',
            'author': 'Dr. Jane Smith, Dr. John Doe, Dr. Emily Johnson',
            'subject': 'Biomedical Research - Clinical Metabolomics and Biomarker Discovery',
            'creator': 'LaTeX with hyperref package version 7.00v',
            'producer': 'pdfTeX-1.40.21 with Adobe Acrobat Distiller 10.1.16',
            'keywords': 'metabolomics, clinical, biomarkers, NMR, LC-MS, mass spectrometry, precision medicine',
            'creationDate': 'D:20240315140530+05\'00\'',
            'modDate': 'D:20240316091245-08\'00\''
        }
        
        # Create temporary file for file system metadata
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"dummy pdf content for testing comprehensive metadata extraction")
            tmp_path = Path(tmp_file.name)
        
        try:
            # Extract metadata
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Verify basic file metadata
            assert metadata['filename'] == tmp_path.name
            assert metadata['file_path'] == str(tmp_path.absolute())
            assert metadata['pages'] == 15
            assert metadata['file_size_bytes'] > 0
            
            # Verify PDF document metadata - all fields present
            assert metadata['title'] == 'Clinical Metabolomics: A Comprehensive Analysis of Biomarkers'
            assert metadata['author'] == 'Dr. Jane Smith, Dr. John Doe, Dr. Emily Johnson'
            assert metadata['subject'] == 'Biomedical Research - Clinical Metabolomics and Biomarker Discovery'
            assert metadata['creator'] == 'LaTeX with hyperref package version 7.00v'
            assert metadata['producer'] == 'pdfTeX-1.40.21 with Adobe Acrobat Distiller 10.1.16'
            assert metadata['keywords'] == 'metabolomics, clinical, biomarkers, NMR, LC-MS, mass spectrometry, precision medicine'
            
            # Verify parsed dates
            assert metadata['creation_date'] == '2024-03-15T14:05:30'
            assert metadata['modification_date'] == '2024-03-16T09:12:45'
            
            # Verify metadata completeness - should have all possible fields
            expected_fields = [
                'filename', 'file_path', 'pages', 'file_size_bytes',
                'title', 'author', 'subject', 'creator', 'producer', 'keywords',
                'creation_date', 'modification_date'
            ]
            for field in expected_fields:
                assert field in metadata, f"Expected field '{field}' missing from metadata"
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_metadata_partial_metadata(self, mock_fitz_open):
        """Test extraction when only some metadata fields are present."""
        mock_doc = MagicMock()
        mock_doc.page_count = 8
        mock_doc.metadata = {
            'title': 'Partial Metadata Research Paper',
            'author': '',  # Empty string - should be excluded
            'subject': 'Medical Research and Clinical Studies',
            'creator': '',  # Empty string - should be excluded
            'producer': 'Adobe PDF Library 15.0',
            'keywords': '',  # Empty string - should be excluded
            'creationDate': 'D:20230510120000Z',
            # modDate intentionally missing
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"partial metadata test content for validation")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Verify present non-empty fields are included
            assert metadata['title'] == 'Partial Metadata Research Paper'
            assert metadata['subject'] == 'Medical Research and Clinical Studies'
            assert metadata['producer'] == 'Adobe PDF Library 15.0'
            assert metadata['creation_date'] == '2023-05-10T12:00:00'
            
            # Verify empty/missing fields are not included
            assert 'author' not in metadata  # Empty string excluded
            assert 'creator' not in metadata  # Empty string excluded
            assert 'keywords' not in metadata  # Empty string excluded
            assert 'modification_date' not in metadata  # Not present in source
            
            # Verify basic fields are always present
            assert metadata['filename'] == tmp_path.name
            assert metadata['pages'] == 8
            assert metadata['file_size_bytes'] > 0
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_metadata_empty_metadata(self, mock_fitz_open):
        """Test extraction when PDF has completely empty metadata."""
        mock_doc = MagicMock()
        mock_doc.page_count = 3
        mock_doc.metadata = {}  # Completely empty metadata dictionary
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"minimal PDF with no metadata")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Should still have basic file system info
            assert metadata['filename'] == tmp_path.name
            assert metadata['file_path'] == str(tmp_path.absolute())
            assert metadata['pages'] == 3
            assert metadata['file_size_bytes'] > 0
            
            # Should not have any PDF document metadata
            pdf_metadata_fields = ['title', 'author', 'subject', 'creator', 'producer', 'keywords']
            for field in pdf_metadata_fields:
                assert field not in metadata, f"Field '{field}' should not be present in empty metadata"
            
            # Should not have date fields
            assert 'creation_date' not in metadata
            assert 'modification_date' not in metadata
            
            # Verify only basic fields are present
            expected_basic_fields = {'filename', 'file_path', 'pages', 'file_size_bytes'}
            assert set(metadata.keys()) == expected_basic_fields
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_metadata_whitespace_handling(self, mock_fitz_open):
        """Test that whitespace in metadata fields is properly trimmed."""
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.metadata = {
            'title': '  \t  Whitespace Test Paper  \n  ',
            'author': '\n\n  Dr. Test Author  \t\t',
            'subject': '   Medical Research with Spaces   \r\n',
            'creator': '  ',  # Only whitespace - should be excluded
            'producer': '\t\n\r  ',  # Only whitespace - should be excluded
            'keywords': '  metabolomics, clinical  ',  # Should be trimmed but kept
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"whitespace handling test content")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Verify whitespace is properly trimmed
            assert metadata['title'] == 'Whitespace Test Paper'
            assert metadata['author'] == 'Dr. Test Author'
            assert metadata['subject'] == 'Medical Research with Spaces'
            assert metadata['keywords'] == 'metabolomics, clinical'
            
            # Verify whitespace-only fields are excluded
            assert 'creator' not in metadata
            assert 'producer' not in metadata
            
            # Verify no extra whitespace remains
            for key, value in metadata.items():
                if isinstance(value, str):
                    assert not value.startswith(' '), f"Field '{key}' has leading whitespace"
                    assert not value.endswith(' '), f"Field '{key}' has trailing whitespace"
                    assert not value.startswith('\t'), f"Field '{key}' has leading tab"
                    assert not value.endswith('\n'), f"Field '{key}' has trailing newline"
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_metadata_large_file_handling(self, mock_fitz_open):
        """Test metadata extraction for large files."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1000  # Large document
        mock_doc.metadata = {
            'title': 'Large Document Test',
            'author': 'Performance Test Author'
        }
        
        # Create a larger temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            large_content = b"large document content " * 10000  # ~230KB
            tmp_file.write(large_content)
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Verify metadata extraction works for large files
            assert metadata['pages'] == 1000
            assert metadata['file_size_bytes'] > 200000  # Should be substantial
            assert metadata['title'] == 'Large Document Test'
            assert metadata['author'] == 'Performance Test Author'
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestPDFDateParsingComprehensive:
    """Test PDF date parsing functionality with comprehensive coverage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    def test_parse_pdf_date_standard_formats(self):
        """Test parsing all standard PDF date formats."""
        # Full format with positive timezone
        result = self.processor._parse_pdf_date('D:20240315140530+05\'00\'')
        assert result == '2024-03-15T14:05:30'
        
        # Full format with negative timezone
        result = self.processor._parse_pdf_date('D:20231201235959-08\'00\'')
        assert result == '2023-12-01T23:59:59'
        
        # Without timezone info
        result = self.processor._parse_pdf_date('D:20240315140530')
        assert result == '2024-03-15T14:05:30'
        
        # With Z timezone (UTC)
        result = self.processor._parse_pdf_date('D:20240315140530Z')
        assert result == '2024-03-15T14:05:30'
    
    def test_parse_pdf_date_no_d_prefix(self):
        """Test parsing PDF dates without 'D:' prefix."""
        result = self.processor._parse_pdf_date('20240315140530')
        assert result == '2024-03-15T14:05:30'
        
        result = self.processor._parse_pdf_date('20231225120000')
        assert result == '2023-12-25T12:00:00'
        
        # With timezone but no D: prefix
        result = self.processor._parse_pdf_date('20240315140530+05\'00\'')
        assert result == '2024-03-15T14:05:30'
    
    def test_parse_pdf_date_minimal_format(self):
        """Test parsing minimal 14-character format."""
        result = self.processor._parse_pdf_date('D:20240101000000')
        assert result == '2024-01-01T00:00:00'
        
        # Exactly 14 characters without D:
        result = self.processor._parse_pdf_date('20240101000000')
        assert result == '2024-01-01T00:00:00'
        
        # Midnight on New Year
        result = self.processor._parse_pdf_date('D:20250101000000')
        assert result == '2025-01-01T00:00:00'
    
    def test_parse_pdf_date_invalid_formats(self):
        """Test handling of invalid PDF date formats."""
        invalid_dates = [
            '',  # Empty string
            None,  # None input
            'D:2024',  # Too short
            '20240315',  # Too short (only 8 chars)
            'D:20241332140530',  # Invalid month (13)
            'D:20240230140530',  # Invalid day (Feb 30)
            'D:20240315250530',  # Invalid hour (25)
            'D:20240315146030',  # Invalid minute (60)
            'D:20240315144570',  # Invalid second (70)
            'D:202a0315140530',  # Non-numeric characters
            'D:INVALID_DATE',  # Completely invalid
            'NotADate',  # Random string
            'D:',  # Only prefix
            'D:20240315',  # Date only, too short for time
            'D:2024031514',  # Incomplete time
        ]
        
        for invalid_date in invalid_dates:
            result = self.processor._parse_pdf_date(invalid_date)
            assert result is None, f"Expected None for invalid date '{invalid_date}', got '{result}'"
    
    def test_parse_pdf_date_edge_cases(self):
        """Test edge cases in PDF date parsing."""
        # Leap year February 29th (2024 is a leap year)
        result = self.processor._parse_pdf_date('D:20240229120000')
        assert result == '2024-02-29T12:00:00'
        
        # Non-leap year February 28th (2023 is not a leap year)
        result = self.processor._parse_pdf_date('D:20230228235959')
        assert result == '2023-02-28T23:59:59'
        
        # Year 2000 (Y2K edge case - was a leap year)
        result = self.processor._parse_pdf_date('D:20000229120000')
        assert result == '2000-02-29T12:00:00'
        
        # Future date
        result = self.processor._parse_pdf_date('D:20301225180000')
        assert result == '2030-12-25T18:00:00'
        
        # End of month/year boundary
        result = self.processor._parse_pdf_date('D:20231231235959')
        assert result == '2023-12-31T23:59:59'
        
        # Start of year
        result = self.processor._parse_pdf_date('D:20240101000000')
        assert result == '2024-01-01T00:00:00'
        
        # Different months with varying days
        result = self.processor._parse_pdf_date('D:20240331180000')  # March 31
        assert result == '2024-03-31T18:00:00'
        
        result = self.processor._parse_pdf_date('D:20240430180000')  # April 30
        assert result == '2024-04-30T18:00:00'

    def test_parse_pdf_date_boundary_conditions(self):
        """Test boundary conditions for date parsing."""
        # Test February 29 on non-leap year (should fail)
        assert self.processor._parse_pdf_date('D:20230229120000') is None
        
        # Test April 31 (invalid - April has only 30 days)
        assert self.processor._parse_pdf_date('D:20240431120000') is None
        
        # Test month 00 (invalid)
        assert self.processor._parse_pdf_date('D:20240015120000') is None
        
        # Test day 00 (invalid)
        assert self.processor._parse_pdf_date('D:20240300120000') is None


class TestExtractTextMetadataIntegration:
    """Test metadata extraction through the main extract_text_from_pdf method."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_text_includes_complete_metadata(self, mock_fitz_open):
        """Test that extract_text_from_pdf includes proper comprehensive metadata."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 10
        mock_doc.metadata = {
            'title': 'Integration Test Paper - Comprehensive Metadata',
            'author': 'Integration Test Author',
            'subject': 'Software Testing and Quality Assurance',
            'creator': 'Test Suite Generator v1.0',
            'producer': 'Mock PDF Producer 2024',
            'keywords': 'testing, integration, metadata, validation',
            'creationDate': 'D:20240201100000+00\'00\'',
            'modDate': 'D:20240202150000-05\'00\''
        }
        
        # Mock pages for text extraction
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test page content for integration testing"
        mock_doc.load_page.return_value = mock_page
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"integration test content with comprehensive metadata")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.extract_text_from_pdf(tmp_path)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'metadata' in result
            assert 'text' in result
            assert 'page_texts' in result
            assert 'processing_info' in result
            
            # Verify comprehensive metadata is present and correct
            metadata = result['metadata']
            
            # Basic file metadata
            assert metadata['filename'] == tmp_path.name
            assert metadata['file_path'] == str(tmp_path.absolute())
            assert metadata['pages'] == 10
            assert metadata['file_size_bytes'] > 0
            assert metadata['pages_processed'] == 10  # Added by extract_text_from_pdf
            
            # Complete PDF metadata
            assert metadata['title'] == 'Integration Test Paper - Comprehensive Metadata'
            assert metadata['author'] == 'Integration Test Author'
            assert metadata['subject'] == 'Software Testing and Quality Assurance'
            assert metadata['creator'] == 'Test Suite Generator v1.0'
            assert metadata['producer'] == 'Mock PDF Producer 2024'
            assert metadata['keywords'] == 'testing, integration, metadata, validation'
            assert metadata['creation_date'] == '2024-02-01T10:00:00'
            assert metadata['modification_date'] == '2024-02-02T15:00:00'
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_text_partial_page_processing_metadata(self, mock_fitz_open):
        """Test metadata when processing only partial pages."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 20
        mock_doc.metadata = {
            'title': 'Partial Processing Test Document',
            'author': 'Partial Test Author'
        }
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page content for partial processing"
        mock_doc.load_page.return_value = mock_page
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"partial processing test content")
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process only pages 5-10 (5 pages total)
            result = self.processor.extract_text_from_pdf(tmp_path, start_page=5, end_page=10)
            
            metadata = result['metadata']
            assert metadata['pages'] == 20  # Total pages in document
            assert metadata['pages_processed'] == 5  # Pages actually processed (10-5)
            assert metadata['title'] == 'Partial Processing Test Document'
            assert metadata['author'] == 'Partial Test Author'
            
            # Verify processing info consistency
            assert result['processing_info']['pages_processed'] == 5
            assert result['processing_info']['start_page'] == 5
            assert result['processing_info']['end_page'] == 10
            assert len(result['page_texts']) == 5
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_extract_text_metadata_with_preprocessing(self, mock_fitz_open):
        """Test that metadata is preserved when text preprocessing is enabled/disabled."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 3
        mock_doc.metadata = {
            'title': 'Preprocessing Test Document',
            'creationDate': 'D:20240315120000'
        }
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content   with  spaces\nand\n\nbreaks"
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"preprocessing test")
            tmp_path = Path(tmp_file.name)
        
        try:
            # Test with preprocessing enabled
            result_with_prep = self.processor.extract_text_from_pdf(tmp_path, preprocess_text=True)
            
            # Test with preprocessing disabled
            result_without_prep = self.processor.extract_text_from_pdf(tmp_path, preprocess_text=False)
            
            # Metadata should be identical regardless of preprocessing
            metadata_with = result_with_prep['metadata']
            metadata_without = result_without_prep['metadata']
            
            assert metadata_with['title'] == metadata_without['title']
            assert metadata_with['pages'] == metadata_without['pages']
            assert metadata_with['creation_date'] == metadata_without['creation_date']
            assert metadata_with['file_size_bytes'] == metadata_without['file_size_bytes']
            
            # Processing info should reflect the preprocessing setting
            assert result_with_prep['processing_info']['preprocessing_applied'] is True
            assert result_without_prep['processing_info']['preprocessing_applied'] is False
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestValidatePDFMetadata:
    """Test metadata extraction through validate_pdf method."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_validate_pdf_includes_complete_metadata(self, mock_fitz_open):
        """Test that validate_pdf includes proper comprehensive metadata."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 7
        mock_doc.metadata = {
            'title': 'Validation Test Document - Complete Metadata',
            'author': 'Validation Test Author',
            'subject': 'PDF Validation and Testing',
            'creator': 'Validation Test Creator',
            'producer': 'Mock Validation Producer',
            'keywords': 'validation, testing, pdf, metadata',
            'creationDate': 'D:20240401140000',
            'modDate': 'D:20240401150000'
        }
        
        # Mock successful page access
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Validation test content"
        mock_doc.load_page.return_value = mock_page
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"validation test content with complete metadata")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.validate_pdf(tmp_path)
            
            # Verify validation result structure
            assert result['valid'] is True
            assert result['error'] is None
            assert result['pages'] == 7
            assert result['encrypted'] is False
            assert result['file_size_bytes'] > 0
            
            # Verify comprehensive metadata is included
            metadata = result['metadata']
            assert isinstance(metadata, dict)
            
            # Basic metadata
            assert metadata['filename'] == tmp_path.name
            assert metadata['file_path'] == str(tmp_path.absolute())
            assert metadata['pages'] == 7
            assert metadata['file_size_bytes'] > 0
            
            # Complete PDF metadata
            assert metadata['title'] == 'Validation Test Document - Complete Metadata'
            assert metadata['author'] == 'Validation Test Author'
            assert metadata['subject'] == 'PDF Validation and Testing'
            assert metadata['creator'] == 'Validation Test Creator'
            assert metadata['producer'] == 'Mock Validation Producer'
            assert metadata['keywords'] == 'validation, testing, pdf, metadata'
            assert metadata['creation_date'] == '2024-04-01T14:00:00'
            assert metadata['modification_date'] == '2024-04-01T15:00:00'
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_validate_pdf_encrypted_metadata_handling(self, mock_fitz_open):
        """Test metadata handling for encrypted PDFs in validation."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"encrypted pdf test content")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.validate_pdf(tmp_path)
            
            # Verify validation properly handles encrypted PDFs
            assert result['valid'] is False
            assert result['encrypted'] is True
            assert result['error'] == "PDF is password protected"
            assert result['file_size_bytes'] > 0  # File size should still be available
            assert result['pages'] is None  # Cannot access pages in encrypted PDF
            assert result['metadata'] == {}  # No metadata for encrypted files
            
            # Verify mock was called and closed
            mock_doc.close.assert_called_once()
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_validate_pdf_corrupted_file_metadata(self, mock_fitz_open):
        """Test metadata handling for corrupted PDF files."""
        # Mock corrupted file that raises FileDataError
        mock_fitz_open.side_effect = fitz.FileDataError("file is damaged")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"corrupted pdf content")
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.processor.validate_pdf(tmp_path)
            
            # Verify corrupted file handling
            assert result['valid'] is False
            assert result['encrypted'] is False
            assert "Invalid or corrupted PDF" in result['error']
            assert result['file_size_bytes'] > 0  # File size should still be available
            assert result['pages'] is None
            assert result['metadata'] == {}
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestMetadataErrorHandling:
    """Test error handling in metadata extraction."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_extraction_with_date_parsing_errors(self, mock_fitz_open):
        """Test that metadata extraction continues even if date parsing fails."""
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.metadata = {
            'title': 'Date Error Test Document',
            'author': 'Date Error Test Author',
            'creationDate': 'INVALID_DATE_FORMAT_12345',
            'modDate': 'D:ALSO_COMPLETELY_INVALID_FORMAT'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"date parsing error test content")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Should still extract other valid metadata
            assert metadata['title'] == 'Date Error Test Document'
            assert metadata['author'] == 'Date Error Test Author'
            assert metadata['filename'] == tmp_path.name
            assert metadata['pages'] == 5
            assert metadata['file_size_bytes'] > 0
            
            # Invalid dates should not be included (parsing failed, filtered out)
            assert 'creation_date' not in metadata
            assert 'modification_date' not in metadata
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_metadata_extraction_file_access_error(self):
        """Test metadata extraction when file access fails."""
        # Use a non-existent file path
        non_existent_path = Path("/non/existent/path/test_metadata.pdf")
        
        # Mock document but file access will fail
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.metadata = {'title': 'Test Document'}
        
        # This should raise an error when trying to get file stats
        with pytest.raises((FileNotFoundError, OSError)):
            self.processor._extract_metadata(mock_doc, non_existent_path)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_extraction_with_none_values(self, mock_fitz_open):
        """Test handling of None values in PDF metadata."""
        mock_doc = MagicMock()
        mock_doc.page_count = 3
        mock_doc.metadata = {
            'title': None,  # Explicitly None
            'author': 'Valid Author',
            'subject': None,  # Explicitly None
            'creationDate': None,  # Explicitly None
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"none values test")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # None values should be handled gracefully
            assert metadata['author'] == 'Valid Author'  # Valid field preserved
            assert 'title' not in metadata  # None value excluded
            assert 'subject' not in metadata  # None value excluded  
            assert 'creation_date' not in metadata  # None value excluded
            
            # Basic metadata should still be present
            assert metadata['pages'] == 3
            assert metadata['file_size_bytes'] > 0
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestRealPDFMetadataExtraction:
    """Test metadata extraction with real PDF files."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
        self.sample_pdf_path = Path(
            "/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
            "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf"
        )
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_real_pdf_comprehensive_metadata_extraction(self):
        """Test comprehensive metadata extraction from actual sample PDF."""
        result = self.processor.extract_text_from_pdf(self.sample_pdf_path)
        metadata = result['metadata']
        
        # Verify basic file metadata is always present
        assert metadata['filename'] == 'Clinical_Metabolomics_paper.pdf'
        assert metadata['file_path'] == str(self.sample_pdf_path.absolute())
        assert isinstance(metadata['pages'], int)
        assert metadata['pages'] > 0
        assert isinstance(metadata['file_size_bytes'], int)
        assert metadata['file_size_bytes'] > 0
        assert isinstance(metadata['pages_processed'], int)
        assert metadata['pages_processed'] > 0
        assert metadata['pages_processed'] <= metadata['pages']
        
        # Log the actual metadata for inspection and debugging
        print(f"\n=== Real PDF Comprehensive Metadata ===")
        for key, value in sorted(metadata.items()):
            print(f"  {key}: {value} ({type(value).__name__})")
        print("=" * 45)
        
        # Test that required file-based metadata is always present
        required_fields = ['filename', 'file_path', 'pages', 'file_size_bytes', 'pages_processed']
        for field in required_fields:
            assert field in metadata, f"Required field '{field}' missing from metadata"
            assert metadata[field] is not None, f"Required field '{field}' is None"
        
        # Test optional PDF metadata fields (if present, they should be valid)
        optional_fields = ['title', 'author', 'subject', 'creator', 'producer', 'keywords']
        for field in optional_fields:
            if field in metadata:
                assert isinstance(metadata[field], str), f"Field '{field}' should be string"
                assert len(metadata[field]) > 0, f"Field '{field}' should not be empty"
                assert metadata[field].strip() == metadata[field], f"Field '{field}' has whitespace"
        
        # Test date fields (if present, they should be valid ISO format)
        date_fields = ['creation_date', 'modification_date']
        for field in date_fields:
            if field in metadata:
                date_str = metadata[field]
                assert isinstance(date_str, str), f"Date field '{field}' should be string"
                assert len(date_str) == 19, f"Date field '{field}' should be 19 chars (ISO format)"
                # Verify ISO 8601 format: YYYY-MM-DDTHH:MM:SS
                try:
                    datetime.fromisoformat(date_str)
                except ValueError:
                    pytest.fail(f"Date field '{field}' is not valid ISO format: {date_str}")
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_real_pdf_validation_metadata_consistency(self):
        """Test metadata consistency between validate_pdf and extract_text_from_pdf."""
        # Get metadata from both methods
        validation_result = self.processor.validate_pdf(self.sample_pdf_path)
        extraction_result = self.processor.extract_text_from_pdf(self.sample_pdf_path)
        
        # Both should succeed
        assert validation_result['valid'] is True
        assert validation_result['error'] is None
        assert 'metadata' in extraction_result
        
        validation_metadata = validation_result['metadata']
        extraction_metadata = extraction_result['metadata']
        
        # Core metadata should be consistent
        consistent_fields = ['filename', 'file_path', 'pages', 'file_size_bytes']
        for field in consistent_fields:
            assert validation_metadata[field] == extraction_metadata[field], \
                f"Field '{field}' inconsistent: validation={validation_metadata[field]}, " \
                f"extraction={extraction_metadata[field]}"
        
        # Optional PDF metadata should be consistent (if present in both)
        pdf_fields = ['title', 'author', 'subject', 'creator', 'producer', 'keywords', 
                      'creation_date', 'modification_date']
        for field in pdf_fields:
            val_has_field = field in validation_metadata
            ext_has_field = field in extraction_metadata
            
            if val_has_field and ext_has_field:
                assert validation_metadata[field] == extraction_metadata[field], \
                    f"PDF field '{field}' inconsistent between methods"
            elif val_has_field != ext_has_field:
                pytest.fail(f"Field '{field}' present in one method but not the other")
        
        print(f"\n=== Metadata Consistency Check ===")
        print(f"Validation metadata keys: {set(validation_metadata.keys())}")
        print(f"Extraction metadata keys: {set(extraction_metadata.keys())}")
        print("All core fields consistent ✓")
    
    @pytest.mark.skipif(
        not Path("/Users/Mark/Research/Clinical_Metabolomics_Oracle/"
                "smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf").exists(),
        reason="Sample PDF not available"
    )
    def test_real_pdf_metadata_performance(self):
        """Test that metadata extraction is performant with real PDF."""
        import time
        
        # Test extract_text_from_pdf metadata performance
        start_time = time.time()
        result = self.processor.extract_text_from_pdf(self.sample_pdf_path)
        extract_time = time.time() - start_time
        
        # Test validate_pdf metadata performance  
        start_time = time.time()
        validation = self.processor.validate_pdf(self.sample_pdf_path)
        validate_time = time.time() - start_time
        
        # Should complete reasonably quickly (adjust thresholds as needed)
        assert extract_time < 30.0, f"Text extraction took too long: {extract_time:.2f}s"
        assert validate_time < 10.0, f"Validation took too long: {validate_time:.2f}s"
        
        # Validation should be faster than full extraction
        assert validate_time < extract_time, \
            f"Validation ({validate_time:.2f}s) should be faster than extraction ({extract_time:.2f}s)"
        
        print(f"\n=== Performance Results ===")
        print(f"Text extraction: {extract_time:.3f}s")
        print(f"Validation: {validate_time:.3f}s")
        print(f"Pages processed: {result['metadata']['pages']}")
        print(f"File size: {result['metadata']['file_size_bytes']:,} bytes")


class TestMetadataFieldTypesAndValidation:
    """Test that metadata fields have correct types and formats."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()
    
    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_field_types_comprehensive(self, mock_fitz_open):
        """Test that all metadata fields have expected types."""
        mock_doc = MagicMock()
        mock_doc.page_count = 42
        mock_doc.metadata = {
            'title': 'Type Validation Test Document',
            'author': 'Type Validation Author',
            'subject': 'Comprehensive Type Testing',
            'creator': 'Type Test Creator',
            'producer': 'Type Test Producer v2.0',
            'keywords': 'types, validation, testing, comprehensive',
            'creationDate': 'D:20240101120000+00\'00\'',
            'modDate': 'D:20240102180000-05\'00\''
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"comprehensive type validation test content")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Test string fields
            string_fields = ['filename', 'file_path', 'title', 'author', 'subject', 
                           'creator', 'producer', 'keywords', 'creation_date', 'modification_date']
            for field in string_fields:
                if field in metadata:
                    assert isinstance(metadata[field], str), f"Field '{field}' should be string"
                    assert len(metadata[field]) > 0, f"Field '{field}' should not be empty"
            
            # Test integer fields
            integer_fields = ['pages', 'file_size_bytes']
            for field in integer_fields:
                assert isinstance(metadata[field], int), f"Field '{field}' should be integer"
                assert metadata[field] > 0, f"Field '{field}' should be positive"
            
            # Test date format (ISO 8601) 
            date_fields = ['creation_date', 'modification_date']
            for field in date_fields:
                if field in metadata:
                    date_str = metadata[field]
                    assert len(date_str) == 19, f"Date field '{field}' should be 19 chars"
                    assert date_str[4] == '-', f"Date field '{field}' missing year-month separator"
                    assert date_str[7] == '-', f"Date field '{field}' missing month-day separator"
                    assert date_str[10] == 'T', f"Date field '{field}' missing date-time separator"
                    assert date_str[13] == ':', f"Date field '{field}' missing hour-minute separator"
                    assert date_str[16] == ':', f"Date field '{field}' missing minute-second separator"
                    
                    # Verify it's a valid datetime
                    try:
                        parsed_date = datetime.fromisoformat(date_str)
                        assert isinstance(parsed_date, datetime)
                    except ValueError as e:
                        pytest.fail(f"Date field '{field}' is not valid ISO format: {e}")
            
            # Test specific expected values
            assert metadata['title'] == 'Type Validation Test Document'
            assert metadata['pages'] == 42
            assert metadata['creation_date'] == '2024-01-01T12:00:00'
            assert metadata['modification_date'] == '2024-01-02T18:00:00'
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_path_handling(self, mock_fitz_open):
        """Test that file paths are handled correctly in metadata."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.metadata = {'title': 'Path Test'}
        
        # Test with different path types
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"path test content")
            tmp_path = Path(tmp_file.name)
        
        try:
            # Test with Path object
            metadata_path = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Test with string path
            metadata_str = self.processor._extract_metadata(mock_doc, str(tmp_path))
            
            # Results should be identical
            assert metadata_path['filename'] == metadata_str['filename']
            assert metadata_path['file_path'] == metadata_str['file_path']
            
            # Verify path properties
            assert metadata_path['filename'] == tmp_path.name
            assert metadata_path['file_path'] == str(tmp_path.absolute())
            assert metadata_path['file_path'].startswith('/')  # Should be absolute
            assert metadata_path['file_path'].endswith('.pdf')
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_special_characters_handling(self, mock_fitz_open):
        """Test handling of special characters in metadata fields."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.metadata = {
            'title': 'Special Characters: àáâãäåæçèéêë & "quotes" & (parentheses) & [brackets]',
            'author': 'José María García-López, François Müller, 北京大学',
            'subject': 'Research with émoticons 😀 and symbols ∑∆∫',
            'keywords': 'unicode, special-chars, test, åøæ, 中文, русский',
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"special characters test")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Special characters should be preserved
            assert 'àáâãäåæçèéêë' in metadata['title']
            assert 'José María García-López' in metadata['author']
            assert '北京大学' in metadata['author']
            assert '😀' in metadata['subject']
            assert '∑∆∫' in metadata['subject']
            assert '中文' in metadata['keywords']
            assert 'русский' in metadata['keywords']
            
            # All fields should still be valid strings
            for field in ['title', 'author', 'subject', 'keywords']:
                assert isinstance(metadata[field], str)
                assert len(metadata[field]) > 0
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestMetadataEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions for metadata extraction."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = BiomedicalPDFProcessor()

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_very_large_fields(self, mock_fitz_open):
        """Test handling of very large metadata field values."""
        # Create very long strings
        very_long_title = "Very Long Title " * 1000  # ~17KB
        very_long_keywords = "keyword" + ", keyword" * 5000  # ~50KB
        
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.metadata = {
            'title': very_long_title,
            'author': 'Normal Author',
            'keywords': very_long_keywords
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"large fields test")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Large fields should be handled without truncation (unless explicitly implemented)
            assert metadata['title'] == very_long_title.strip()
            assert metadata['keywords'] == very_long_keywords.strip()
            assert metadata['author'] == 'Normal Author'
            
            # Should still be valid strings
            assert isinstance(metadata['title'], str)
            assert isinstance(metadata['keywords'], str)
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_zero_page_document(self, mock_fitz_open):
        """Test handling of document with zero pages."""
        mock_doc = MagicMock()
        mock_doc.page_count = 0  # Unusual but possible
        mock_doc.metadata = {'title': 'Zero Page Document'}
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"zero pages test")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Should handle zero pages gracefully
            assert metadata['pages'] == 0
            assert metadata['title'] == 'Zero Page Document'
            assert metadata['file_size_bytes'] > 0  # File still has size
            
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch('lightrag_integration.pdf_processor.fitz.open')
    def test_metadata_maximum_pages(self, mock_fitz_open):
        """Test handling of document with very large page count."""
        mock_doc = MagicMock()
        mock_doc.page_count = 999999  # Very large document
        mock_doc.metadata = {'title': 'Massive Document'}
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"maximum pages test")
            tmp_path = Path(tmp_file.name)
        
        try:
            metadata = self.processor._extract_metadata(mock_doc, tmp_path)
            
            # Should handle large page counts
            assert metadata['pages'] == 999999
            assert metadata['title'] == 'Massive Document'
            assert isinstance(metadata['pages'], int)
            
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Run specific test classes for development/debugging
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test class if provided as argument
        test_class = sys.argv[1]
        pytest.main([__file__ + "::" + test_class, "-v", "-s"])
    else:
        # Run all tests with verbose output
        pytest.main([__file__, "-v", "-s"])