"""
BiomedicalPDFProcessor for Clinical Metabolomics Oracle - LightRAG integration.

This module provides specialized PDF processing capabilities for biomedical documents
using PyMuPDF (fitz). It includes text extraction, metadata retrieval, and 
preprocessing specifically tailored for clinical metabolomics and biomedical literature.

Classes:
    - BiomedicalPDFProcessorError: Custom exception for PDF processing errors
    - BiomedicalPDFProcessor: Main class for processing biomedical PDF documents

The processor handles:
    - Text extraction from PDF documents using PyMuPDF
    - Metadata extraction (creation date, modification date, title, author, etc.)
    - Text preprocessing for biomedical content
    - Error handling for corrupted, encrypted, or invalid PDFs
    - Page-by-page processing with optional page range specification
    - Cleaning and normalization of extracted text
"""

import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import fitz  # PyMuPDF


class BiomedicalPDFProcessorError(Exception):
    """Custom exception for biomedical PDF processing errors."""
    pass


class BiomedicalPDFProcessor:
    """
    Specialized PDF processor for biomedical documents.
    
    This class provides comprehensive PDF processing capabilities specifically
    designed for biomedical and clinical metabolomics literature. It uses
    PyMuPDF for robust PDF handling and includes specialized text preprocessing
    for scientific content.
    
    Attributes:
        logger: Logger instance for tracking processing activities
        
    Example:
        processor = BiomedicalPDFProcessor()
        result = processor.extract_text_from_pdf("paper.pdf")
        print(f"Extracted {len(result['text'])} characters from {result['metadata']['pages']} pages")
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the BiomedicalPDFProcessor.
        
        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, 
                             pdf_path: Union[str, Path],
                             start_page: Optional[int] = None,
                             end_page: Optional[int] = None,
                             preprocess_text: bool = True) -> Dict[str, Any]:
        """
        Extract text and metadata from a biomedical PDF document.
        
        This method processes the entire PDF or a specified page range,
        extracting both text content and document metadata. The extracted
        text can be optionally preprocessed to improve quality for
        downstream LightRAG processing.
        
        Args:
            pdf_path: Path to the PDF file (string or Path object)
            start_page: Starting page number (0-indexed). If None, starts from page 0.
            end_page: Ending page number (0-indexed, exclusive). If None, processes all pages.
            preprocess_text: Whether to apply biomedical text preprocessing
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'text': Extracted text content (str)
                - 'metadata': PDF metadata including:
                  - 'filename': Original filename
                  - 'file_path': Full file path
                  - 'pages': Total number of pages
                  - 'pages_processed': Number of pages actually processed
                  - 'creation_date': PDF creation date (if available)
                  - 'modification_date': PDF modification date (if available)
                  - 'title': Document title (if available)
                  - 'author': Document author (if available)
                  - 'subject': Document subject (if available)
                  - 'creator': PDF creator application (if available)
                  - 'producer': PDF producer (if available)
                  - 'file_size_bytes': File size in bytes
                - 'page_texts': List of text from each processed page (List[str])
                - 'processing_info': Dictionary with processing statistics
                
        Raises:
            BiomedicalPDFProcessorError: If PDF cannot be processed
            FileNotFoundError: If PDF file doesn't exist
            PermissionError: If file cannot be accessed
        """
        pdf_path = Path(pdf_path)
        
        # Validate file existence and accessibility
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.is_file():
            raise BiomedicalPDFProcessorError(f"Path is not a file: {pdf_path}")
        
        try:
            # Open the PDF document
            self.logger.info(f"Opening PDF file: {pdf_path}")
            doc = fitz.open(str(pdf_path))
            
            # Handle encrypted PDFs
            if doc.needs_pass:
                doc.close()
                raise BiomedicalPDFProcessorError(f"PDF is password protected: {pdf_path}")
            
            # Get basic document information
            total_pages = doc.page_count
            self.logger.info(f"PDF has {total_pages} pages")
            
            # Determine page range
            start_page = start_page or 0
            end_page = end_page or total_pages
            
            # Validate page range
            if start_page < 0 or start_page >= total_pages:
                doc.close()
                raise BiomedicalPDFProcessorError(
                    f"Invalid start_page {start_page}. Must be 0 <= start_page < {total_pages}"
                )
            
            if end_page <= start_page or end_page > total_pages:
                doc.close()
                raise BiomedicalPDFProcessorError(
                    f"Invalid end_page {end_page}. Must be {start_page} < end_page <= {total_pages}"
                )
            
            # Extract metadata
            metadata = self._extract_metadata(doc, pdf_path)
            metadata['pages_processed'] = end_page - start_page
            
            # Extract text from specified pages
            page_texts = []
            full_text_parts = []
            
            self.logger.info(f"Processing pages {start_page} to {end_page-1}")
            
            for page_num in range(start_page, end_page):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Apply preprocessing if requested
                    if preprocess_text:
                        page_text = self._preprocess_biomedical_text(page_text)
                    
                    page_texts.append(page_text)
                    full_text_parts.append(page_text)
                    
                    self.logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    page_texts.append("")  # Keep page index consistent
            
            # Combine all text
            full_text = "\n\n".join(full_text_parts)
            
            # Create processing information
            processing_info = {
                'start_page': start_page,
                'end_page': end_page,
                'pages_processed': len(page_texts),
                'total_characters': len(full_text),
                'preprocessing_applied': preprocess_text,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            doc.close()
            
            self.logger.info(
                f"Successfully processed {processing_info['pages_processed']} pages, "
                f"extracted {processing_info['total_characters']} characters"
            )
            
            return {
                'text': full_text,
                'metadata': metadata,
                'page_texts': page_texts,
                'processing_info': processing_info
            }
            
        except fitz.FileDataError as e:
            raise BiomedicalPDFProcessorError(f"Invalid or corrupted PDF file: {e}")
        except Exception as fitz_error:
            # Handle various PyMuPDF errors that don't have specific exception types
            if 'fitz' in str(type(fitz_error)).lower() or 'mupdf' in str(fitz_error).lower():
                raise BiomedicalPDFProcessorError(f"PyMuPDF processing error: {fitz_error}")
            raise  # Re-raise if not a PyMuPDF error
        except PermissionError as e:
            raise BiomedicalPDFProcessorError(f"Permission denied accessing PDF: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error processing PDF {pdf_path}: {e}")
            raise BiomedicalPDFProcessorError(f"Failed to process PDF: {e}")
    
    def _extract_metadata(self, doc: fitz.Document, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from the PDF document.
        
        Args:
            doc: Opened PyMuPDF document
            pdf_path: Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing all available metadata
        """
        metadata = {
            'filename': pdf_path.name,
            'file_path': str(pdf_path.absolute()),
            'pages': doc.page_count,
            'file_size_bytes': pdf_path.stat().st_size
        }
        
        # Extract PDF metadata
        pdf_metadata = doc.metadata
        
        # Map PDF metadata fields with safe extraction
        metadata_fields = {
            'title': pdf_metadata.get('title', '').strip(),
            'author': pdf_metadata.get('author', '').strip(),
            'subject': pdf_metadata.get('subject', '').strip(),
            'creator': pdf_metadata.get('creator', '').strip(),
            'producer': pdf_metadata.get('producer', '').strip(),
            'keywords': pdf_metadata.get('keywords', '').strip()
        }
        
        # Only include non-empty metadata fields
        for field, value in metadata_fields.items():
            if value:
                metadata[field] = value
        
        # Handle creation and modification dates
        try:
            creation_date = pdf_metadata.get('creationDate', '')
            if creation_date:
                # PyMuPDF returns dates in format: D:YYYYMMDDHHmmSSOHH'mm'
                metadata['creation_date'] = self._parse_pdf_date(creation_date)
        except Exception as e:
            self.logger.debug(f"Could not parse creation date: {e}")
        
        try:
            mod_date = pdf_metadata.get('modDate', '')
            if mod_date:
                metadata['modification_date'] = self._parse_pdf_date(mod_date)
        except Exception as e:
            self.logger.debug(f"Could not parse modification date: {e}")
        
        return metadata
    
    def _parse_pdf_date(self, pdf_date: str) -> Optional[str]:
        """
        Parse PDF date format to ISO format.
        
        PDF dates are typically in format: D:YYYYMMDDHHmmSSOHH'mm'
        
        Args:
            pdf_date: Date string from PDF metadata
            
        Returns:
            Optional[str]: ISO formatted date string or None if parsing fails
        """
        if not pdf_date:
            return None
        
        try:
            # Remove 'D:' prefix if present
            if pdf_date.startswith('D:'):
                pdf_date = pdf_date[2:]
            
            # Extract basic components (YYYYMMDDHHMMSS)
            if len(pdf_date) >= 14:
                year = int(pdf_date[0:4])
                month = int(pdf_date[4:6])
                day = int(pdf_date[6:8])
                hour = int(pdf_date[8:10])
                minute = int(pdf_date[10:12])
                second = int(pdf_date[12:14])
                
                # Create datetime object and return ISO format
                dt = datetime(year, month, day, hour, minute, second)
                return dt.isoformat()
        except (ValueError, IndexError) as e:
            self.logger.debug(f"Failed to parse PDF date '{pdf_date}': {e}")
        
        return None
    
    def _preprocess_biomedical_text(self, text: str) -> str:
        """
        Preprocess extracted text for biomedical content analysis.
        
        This method applies specialized cleaning and normalization techniques
        suitable for biomedical and clinical metabolomics literature.
        
        Args:
            text: Raw extracted text from PDF
            
        Returns:
            str: Preprocessed text optimized for LightRAG processing
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Clean up common PDF extraction artifacts
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation across lines
        text = re.sub(r'\s*\n\s*([a-z])', r' \1', text)  # Join broken words
        
        # Normalize chemical formulas and compound names
        text = re.sub(r'([A-Z][a-z]?)(\d+)', r'\1\2', text)  # Ensure proper chemical notation
        
        # Preserve important biomedical formatting
        text = re.sub(r'\bp\s*<\s*0\.\d+', lambda m: m.group(0).replace(' ', ''), text)  # p-values
        text = re.sub(r'\bp\s*=\s*0\.\d+', lambda m: m.group(0).replace(' ', ''), text)  # p-values
        
        # Clean up reference markers while preserving structure
        text = re.sub(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]', r'[\1]', text)
        
        # Normalize units and measurements
        text = re.sub(r'(\d+)\s*(mg|kg|g|ml|l|mol|M)\b', r'\1 \2', text)
        
        # Remove header/footer artifacts (page numbers, etc.)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Clean up excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{2,}', '--', text)
        
        # Trim whitespace and return
        return text.strip()
    
    def validate_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a PDF file and return basic information without full processing.
        
        This method performs quick validation checks to determine if a PDF
        can be processed without doing the full text extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'valid': Boolean indicating if PDF is processable
                - 'error': Error message if PDF is invalid (None if valid)
                - 'file_size_bytes': File size in bytes (if accessible)
                - 'pages': Number of pages (if PDF is valid)
                - 'encrypted': Boolean indicating if PDF is password protected
                - 'metadata': Basic metadata (if available)
                
        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        result = {
            'valid': False,
            'error': None,
            'file_size_bytes': None,
            'pages': None,
            'encrypted': False,
            'metadata': {}
        }
        
        try:
            # Get file size
            result['file_size_bytes'] = pdf_path.stat().st_size
            
            # Try to open the PDF
            doc = fitz.open(str(pdf_path))
            
            # Check if encrypted
            if doc.needs_pass:
                result['encrypted'] = True
                result['error'] = "PDF is password protected"
                doc.close()
                return result
            
            # Get basic information
            result['pages'] = doc.page_count
            result['metadata'] = self._extract_metadata(doc, pdf_path)
            
            # Test if we can read at least one page
            if doc.page_count > 0:
                try:
                    page = doc.load_page(0)
                    test_text = page.get_text()
                    result['valid'] = True
                except Exception as e:
                    result['error'] = f"Cannot extract text from PDF: {e}"
            else:
                result['error'] = "PDF has no pages"
            
            doc.close()
            
        except fitz.FileDataError as e:
            result['error'] = f"Invalid or corrupted PDF: {e}"
        except PermissionError as e:
            result['error'] = f"Permission denied: {e}"
        except Exception as e:
            result['error'] = f"Unexpected error: {e}"
        
        return result
    
    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """
        Get the number of pages in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            int: Number of pages in the PDF
            
        Raises:
            BiomedicalPDFProcessorError: If PDF cannot be opened or is invalid
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            if doc.needs_pass:
                doc.close()
                raise BiomedicalPDFProcessorError(f"PDF is password protected: {pdf_path}")
            
            page_count = doc.page_count
            doc.close()
            return page_count
            
        except fitz.FileDataError as e:
            raise BiomedicalPDFProcessorError(f"Invalid or corrupted PDF: {e}")
        except Exception as e:
            raise BiomedicalPDFProcessorError(f"Failed to get page count: {e}")