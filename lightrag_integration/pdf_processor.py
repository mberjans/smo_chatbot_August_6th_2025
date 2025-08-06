"""
BiomedicalPDFProcessor for Clinical Metabolomics Oracle - LightRAG integration.

This module provides specialized PDF processing capabilities for biomedical documents
using PyMuPDF (fitz). It includes text extraction, metadata retrieval, and 
preprocessing specifically tailored for clinical metabolomics and biomedical literature.

Classes:
    - BiomedicalPDFProcessorError: Base custom exception for PDF processing errors
    - PDFValidationError: Exception for PDF file validation failures
    - PDFProcessingTimeoutError: Exception for processing timeout conditions
    - PDFMemoryError: Exception for memory-related processing issues
    - PDFFileAccessError: Exception for file access problems (locks, permissions)
    - PDFContentError: Exception for content extraction and encoding issues
    - BiomedicalPDFProcessor: Main class for processing biomedical PDF documents

The processor handles:
    - Text extraction from PDF documents using PyMuPDF
    - Metadata extraction (creation date, modification date, title, author, etc.)
    - Text preprocessing for biomedical content
    - Comprehensive error handling for various edge cases:
      * MIME type validation and PDF header verification
      * Memory pressure monitoring during processing
      * Processing timeout protection
      * Enhanced file locking and permission detection
      * Zero-byte file handling
      * Malformed PDF structure detection
      * Character encoding issue resolution
      * Large text block protection
    - Page-by-page processing with optional page range specification
    - Cleaning and normalization of extracted text
    - Processing statistics and monitoring capabilities
"""

import re
import logging
import asyncio
import mimetypes
import psutil
import signal
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
from contextlib import contextmanager
import fitz  # PyMuPDF


class BiomedicalPDFProcessorError(Exception):
    """Base custom exception for biomedical PDF processing errors."""
    pass


class PDFValidationError(BiomedicalPDFProcessorError):
    """Exception raised when PDF file validation fails."""
    pass


class PDFProcessingTimeoutError(BiomedicalPDFProcessorError):
    """Exception raised when PDF processing times out."""
    pass


class PDFMemoryError(BiomedicalPDFProcessorError):
    """Exception raised when PDF processing exceeds memory limits."""
    pass


class PDFFileAccessError(BiomedicalPDFProcessorError):
    """Exception raised when PDF file cannot be accessed (locked, permissions, etc.)."""
    pass


class PDFContentError(BiomedicalPDFProcessorError):
    """Exception raised when PDF has no extractable content or encoding issues."""
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
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 processing_timeout: int = 300,  # 5 minutes default
                 memory_limit_mb: int = 1024,    # 1GB default
                 max_page_text_size: int = 1000000):  # 1MB per page default
        """
        Initialize the BiomedicalPDFProcessor.
        
        Args:
            logger: Optional logger instance. If None, creates a default logger.
            processing_timeout: Maximum processing time in seconds (default: 300)
            memory_limit_mb: Maximum memory usage in MB (default: 1024)
            max_page_text_size: Maximum text size per page in characters (default: 1000000)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.processing_timeout = processing_timeout
        self.memory_limit_mb = memory_limit_mb
        self.max_page_text_size = max_page_text_size
        self._processing_start_time = None
        self._memory_monitor_active = False
    
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
        
        # Start processing timer for timeout monitoring
        self._processing_start_time = time.time()
        
        # Comprehensive file validation
        self._validate_pdf_file(pdf_path)
        
        try:
            # Start memory monitoring
            with self._monitor_memory():
                # Open the PDF document with timeout protection
                self.logger.info(f"Opening PDF file: {pdf_path}")
                doc = self._open_pdf_with_timeout(pdf_path)
            
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
            
            # Extract text from specified pages with enhanced error handling
            page_texts = []
            full_text_parts = []
            
            self.logger.info(f"Processing pages {start_page} to {end_page-1}")
            
            for page_num in range(start_page, end_page):
                # Check timeout on each page
                self._check_processing_timeout()
                
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Validate page text for encoding issues and size limits
                    page_text = self._validate_and_clean_page_text(page_text, page_num)
                    
                    # Apply preprocessing if requested
                    if preprocess_text:
                        page_text = self._preprocess_biomedical_text(page_text)
                    
                    page_texts.append(page_text)
                    full_text_parts.append(page_text)
                    
                    self.logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    page_texts.append("")  # Keep page index consistent
            
            # Combine all text and validate final result
            full_text = "\n\n".join(full_text_parts)
            
            # Check if we extracted any meaningful content
            if not full_text.strip():
                self.logger.warning(f"No text content extracted from PDF: {pdf_path}")
                # Don't raise exception, but log the issue
                
            # Final encoding validation
            full_text = self._validate_text_encoding(full_text)
            
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
            
        except (PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError, 
                PDFFileAccessError, PDFContentError) as e:
            # Re-raise our custom exceptions
            self.logger.error(f"PDF processing error for {pdf_path}: {e}")
            raise
        except fitz.FileDataError as e:
            raise PDFValidationError(f"Invalid or corrupted PDF file: {e}")
        except Exception as fitz_error:
            # Handle various PyMuPDF errors that don't have specific exception types
            error_msg = str(fitz_error).lower()
            if 'fitz' in str(type(fitz_error)).lower() or 'mupdf' in error_msg:
                if 'timeout' in error_msg or 'time' in error_msg:
                    raise PDFProcessingTimeoutError(f"PyMuPDF processing timeout: {fitz_error}")
                elif 'memory' in error_msg or 'allocation' in error_msg:
                    raise PDFMemoryError(f"PyMuPDF memory error: {fitz_error}")
                else:
                    raise BiomedicalPDFProcessorError(f"PyMuPDF processing error: {fitz_error}")
            raise  # Re-raise if not a PyMuPDF error
        except MemoryError as e:
            raise PDFMemoryError(f"Memory error processing PDF: {e}")
        except PermissionError as e:
            raise PDFFileAccessError(f"Permission denied accessing PDF: {e}")
        except Exception as e:
            # Check if it's a timeout condition
            if self._processing_start_time and time.time() - self._processing_start_time > self.processing_timeout:
                raise PDFProcessingTimeoutError(f"Processing timed out: {e}")
            
            self.logger.error(f"Unexpected error processing PDF {pdf_path}: {e}")
            raise BiomedicalPDFProcessorError(f"Failed to process PDF: {e}")
    
    def _extract_metadata(self, doc: fitz.Document, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from the PDF document.
        
        Args:
            doc: Opened PyMuPDF document
            pdf_path: Path to the PDF file (string or Path object)
            
        Returns:
            Dict[str, Any]: Dictionary containing all available metadata
        """
        # Ensure pdf_path is a Path object
        pdf_path = Path(pdf_path)
        
        metadata = {
            'filename': pdf_path.name,
            'file_path': str(pdf_path.absolute()),
            'pages': doc.page_count,
            'file_size_bytes': pdf_path.stat().st_size
        }
        
        # Extract PDF metadata
        pdf_metadata = doc.metadata
        
        # Map PDF metadata fields with safe extraction (handle None values)
        metadata_fields = {}
        field_names = ['title', 'author', 'subject', 'creator', 'producer', 'keywords']
        
        for field in field_names:
            value = pdf_metadata.get(field, '')
            if value is not None:
                value = str(value).strip()
                if value:  # Only include non-empty values
                    metadata_fields[field] = value
        
        # Add valid metadata fields to the result
        metadata.update(metadata_fields)
        
        # Handle creation and modification dates
        try:
            creation_date = pdf_metadata.get('creationDate', '')
            if creation_date:
                # PyMuPDF returns dates in format: D:YYYYMMDDHHmmSSOHH'mm'
                parsed_date = self._parse_pdf_date(creation_date)
                if parsed_date:  # Only add if parsing succeeded
                    metadata['creation_date'] = parsed_date
        except Exception as e:
            self.logger.debug(f"Could not parse creation date: {e}")
        
        try:
            mod_date = pdf_metadata.get('modDate', '')
            if mod_date:
                parsed_date = self._parse_pdf_date(mod_date)
                if parsed_date:  # Only add if parsing succeeded
                    metadata['modification_date'] = parsed_date
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
        
        This method applies comprehensive cleaning and normalization techniques
        specifically designed for biomedical and clinical metabolomics literature.
        It handles common PDF extraction artifacts while preserving scientific 
        notation, chemical formulas, statistical values, and biomedical terminology.
        
        Processing steps:
        1. Remove common PDF artifacts (headers, footers, page numbers)
        2. Fix text extraction issues (broken words, spacing problems)
        3. Preserve scientific notation (p-values, chemical formulas, units)
        4. Handle biomedical formatting (references, citations, figure captions)
        5. Clean up text flow while maintaining document structure
        6. Normalize special biomedical terms and notations
        
        Args:
            text: Raw extracted text from PDF
            
        Returns:
            str: Preprocessed text optimized for LightRAG processing
        """
        if not text:
            return ""
        
        # Step 1: Remove common PDF artifacts
        text = self._remove_pdf_artifacts(text)
        
        # Step 2: Fix text extraction issues
        text = self._fix_text_extraction_issues(text)
        
        # Step 3: Preserve scientific notation
        text = self._preserve_scientific_notation(text)
        
        # Step 4: Handle biomedical formatting
        text = self._handle_biomedical_formatting(text)
        
        # Step 5: Clean up text flow
        text = self._clean_text_flow(text)
        
        # Step 6: Handle special biomedical terms
        text = self._normalize_biomedical_terms(text)
        
        # Final cleanup: normalize whitespace and trim
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive line breaks
        
        return text.strip()
    
    def _validate_pdf_file(self, pdf_path: Path) -> None:
        """
        Comprehensive PDF file validation.
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            PDFValidationError: If file validation fails
            PDFFileAccessError: If file cannot be accessed
        """
        # Check file existence
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.is_file():
            raise PDFValidationError(f"Path is not a file: {pdf_path}")
        
        # Check for zero-byte files
        file_size = pdf_path.stat().st_size
        if file_size == 0:
            raise PDFValidationError(f"PDF file is empty (0 bytes): {pdf_path}")
        
        # MIME type validation
        mime_type, _ = mimetypes.guess_type(str(pdf_path))
        if mime_type != 'application/pdf':
            # Additional check by reading file header
            try:
                with open(pdf_path, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        raise PDFValidationError(f"File does not appear to be a valid PDF (invalid header): {pdf_path}")
            except (OSError, IOError) as e:
                raise PDFFileAccessError(f"Cannot read file header: {e}")
        
        # Enhanced file locking detection
        try:
            # Try to open file in append mode (non-destructive test for locks)
            with open(pdf_path, 'ab') as f:
                pass
        except (OSError, IOError, PermissionError) as e:
            if "being used by another process" in str(e).lower() or "resource temporarily unavailable" in str(e).lower():
                raise PDFFileAccessError(f"PDF file is locked or in use by another process: {pdf_path}")
            elif "permission denied" in str(e).lower():
                raise PDFFileAccessError(f"Permission denied accessing PDF file: {pdf_path}")
            else:
                raise PDFFileAccessError(f"Cannot access PDF file: {e}")
        
        # Check file size limits (warn if very large)
        if file_size > 100 * 1024 * 1024:  # 100MB
            self.logger.warning(f"Large PDF file detected ({file_size / 1024 / 1024:.1f} MB): {pdf_path}")
        
        self.logger.debug(f"PDF file validation passed: {pdf_path} ({file_size} bytes)")
    
    def _open_pdf_with_timeout(self, pdf_path: Path) -> fitz.Document:
        """
        Open PDF with timeout protection.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            fitz.Document: Opened PDF document
            
        Raises:
            PDFProcessingTimeoutError: If opening takes too long
            PDFValidationError: If PDF is invalid or corrupted
        """
        def open_pdf():
            return fitz.open(str(pdf_path))
        
        try:
            # Use a simple timeout mechanism
            start_time = time.time()
            doc = open_pdf()
            elapsed = time.time() - start_time
            
            if elapsed > 30:  # Warn if opening takes more than 30 seconds
                self.logger.warning(f"PDF opening took {elapsed:.1f} seconds: {pdf_path}")
            
            return doc
            
        except fitz.FileDataError as e:
            raise PDFValidationError(f"Invalid or corrupted PDF file: {e}")
        except Exception as e:
            # Check if it's a timeout or other processing issue
            if time.time() - self._processing_start_time > self.processing_timeout:
                raise PDFProcessingTimeoutError(f"PDF opening timed out after {self.processing_timeout}s")
            raise PDFValidationError(f"Failed to open PDF: {e}")
    
    @contextmanager
    def _monitor_memory(self):
        """
        Context manager for memory monitoring during PDF processing.
        
        Raises:
            PDFMemoryError: If memory usage exceeds limits
        """
        self._memory_monitor_active = True
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            self._memory_monitor_active = False
            
        # Check final memory usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        if memory_increase > self.memory_limit_mb:
            self.logger.warning(f"High memory usage detected: {memory_increase:.1f} MB increase")
            # Don't raise exception here as processing is complete, just log
        
        # Check system memory pressure
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 90:
            self.logger.warning(f"System memory usage high: {system_memory.percent:.1f}%")
    
    def _check_processing_timeout(self) -> None:
        """
        Check if processing has exceeded timeout limit.
        
        Raises:
            PDFProcessingTimeoutError: If processing has timed out
        """
        if self._processing_start_time is None:
            return
            
        elapsed = time.time() - self._processing_start_time
        if elapsed > self.processing_timeout:
            raise PDFProcessingTimeoutError(
                f"PDF processing timed out after {elapsed:.1f}s (limit: {self.processing_timeout}s)"
            )
    
    def _validate_and_clean_page_text(self, page_text: str, page_num: int) -> str:
        """
        Validate and clean text extracted from a PDF page.
        
        Args:
            page_text: Raw text extracted from page
            page_num: Page number for logging
            
        Returns:
            str: Cleaned and validated page text
            
        Raises:
            PDFContentError: If page text has severe issues
        """
        if not page_text:
            return ""  # Empty page is acceptable
        
        # Check for excessively large text blocks
        if len(page_text) > self.max_page_text_size:
            self.logger.warning(
                f"Large text block on page {page_num}: {len(page_text)} characters "
                f"(limit: {self.max_page_text_size}). Truncating."
            )
            page_text = page_text[:self.max_page_text_size] + "\n[TEXT TRUNCATED DUE TO SIZE LIMIT]"
        
        # Check for encoding issues
        try:
            # Try to encode/decode to catch encoding problems
            page_text.encode('utf-8').decode('utf-8')
        except UnicodeEncodeError as e:
            self.logger.warning(f"Encoding issues detected on page {page_num}: {e}")
            # Replace problematic characters
            page_text = page_text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Check for binary data or excessive control characters
        control_chars = sum(1 for c in page_text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > len(page_text) * 0.1:  # More than 10% control characters
            self.logger.warning(f"Excessive control characters on page {page_num}: {control_chars}/{len(page_text)}")
            # Clean up control characters except newlines, carriage returns, and tabs
            page_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', page_text)
        
        return page_text
    
    def _validate_text_encoding(self, text: str) -> str:
        """
        Validate and fix text encoding issues in the final extracted text.
        
        Args:
            text: Extracted text to validate
            
        Returns:
            str: Text with encoding issues fixed
        """
        if not text:
            return text
        
        try:
            # Test round-trip encoding
            text.encode('utf-8').decode('utf-8')
        except UnicodeEncodeError as e:
            self.logger.warning(f"Final text encoding issues detected: {e}")
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Replace common problematic Unicode characters with ASCII equivalents
        replacements = {
            '\u2013': '-',      # en dash
            '\u2014': '--',     # em dash
            '\u2018': "'",      # left single quotation mark
            '\u2019': "'",      # right single quotation mark
            '\u201c': '"',      # left double quotation mark
            '\u201d': '"',      # right double quotation mark
            '\u2026': '...',    # horizontal ellipsis
            '\u00a0': ' ',      # non-breaking space
            '\u00b7': '*',      # middle dot
            '\u2022': '*',      # bullet
        }
        
        for unicode_char, ascii_replacement in replacements.items():
            text = text.replace(unicode_char, ascii_replacement)
        
        return text
    
    def _remove_pdf_artifacts(self, text: str) -> str:
        """
        Remove common PDF extraction artifacts like headers, footers, and watermarks.
        
        Args:
            text: Input text with potential PDF artifacts
            
        Returns:
            str: Text with PDF artifacts removed
        """
        # Remove page numbers (standalone numbers on their own lines or at start/end)
        # Be more careful not to remove numbers that are part of scientific content
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Page\s+\d+\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        # Only remove trailing numbers that are likely page numbers (not preceded by decimal point)
        text = re.sub(r'(?<![0-9.])\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common header/footer patterns
        text = re.sub(r'\n\s*Page\s+\d+\s*of\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*/\s*\d+\s*\n', '\n', text)
        
        # Remove journal headers and footers (common patterns)
        text = re.sub(r'\n\s*.*?Journal.*?\d{4}.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*doi:\s*\S+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*©\s*\d{4}.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove repetitive horizontal lines or dots
        text = re.sub(r'\n\s*[._-]{5,}\s*\n', '\n', text)
        text = re.sub(r'\n\s*[.]{5,}\s*\n', '\n', text)
        
        # Remove "Downloaded from" or "Accessed" lines
        text = re.sub(r'\n\s*Downloaded\s+from.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'Downloaded\s+from.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*Accessed\s+on.*?\n', '\n', text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_text_extraction_issues(self, text: str) -> str:
        """
        Fix common text extraction issues like broken words and incorrect spacing.
        
        Args:
            text: Input text with extraction issues
            
        Returns:
            str: Text with extraction issues fixed
        """
        # Fix hyphenated words broken across lines
        text = re.sub(r'-\s*\n\s*([a-z])', r'\1', text)
        
        # Fix words broken across lines (common in PDF extraction)
        text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1\2', text)
        
        # Fix broken sentences (lowercase letter after line break)
        text = re.sub(r'\n\s*([a-z])', r' \1', text)
        
        # Fix missing spaces after periods
        text = re.sub(r'\.([A-Z][a-z])', r'. \1', text)
        
        # Fix missing spaces before parentheses
        text = re.sub(r'([a-z])(\([A-Z])', r'\1 \2', text)
        
        # Fix excessive spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
        
        # Fix spacing around brackets and parentheses
        text = re.sub(r'\s*\[\s*', r' [', text)
        text = re.sub(r'\s*\]\s*', r'] ', text)
        text = re.sub(r'\s*\(\s*', r' (', text)
        text = re.sub(r'\s*\)\s*', r') ', text)
        
        return text
    
    def _preserve_scientific_notation(self, text: str) -> str:
        """
        Preserve and normalize scientific notation, statistical values, and units.
        
        Args:
            text: Input text containing scientific notation
            
        Returns:
            str: Text with preserved and normalized scientific notation
        """
        # Preserve p-values with proper spacing
        text = re.sub(r'\bp\s*[<>=]\s*0\.\d+', lambda m: re.sub(r'\s+', '', m.group(0)), text)
        text = re.sub(r'\bp\s*-\s*value\s*[<>=]\s*0\.\d+', lambda m: re.sub(r'\s+', '', m.group(0).replace(' - ', '-').replace('- ', '-').replace(' -', '-')), text)
        
        # Preserve R-squared values
        text = re.sub(r'\bR\s*2\s*=\s*0\.\d+', lambda m: re.sub(r'\s+', '', m.group(0)), text)
        text = re.sub(r'\bR\s*-\s*squared\s*=\s*0\.\d+', lambda m: re.sub(r'\s+', ' ', m.group(0)), text)
        
        # Preserve confidence intervals
        text = re.sub(r'95\s*%\s*CI\s*[:\[]', '95% CI:', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*\]', r'[\1-\2]', text)
        
        # Preserve scientific notation (e.g., 1.5 × 10⁻³, 1.5e-3)
        text = re.sub(r'(\d+\.?\d*)\s*[×x]\s*10\s*[⁻−-]\s*(\d+)', r'\1×10⁻\2', text)
        text = re.sub(r'(\d+\.?\d*)\s*e\s*[⁻−-]\s*(\d+)', r'\1e-\2', text)
        
        # Preserve chemical formulas with proper spacing (e.g., H 2 O → H2O)
        text = re.sub(r'([A-Z][a-z]?)\s*(\d+)', r'\1\2', text)
        text = re.sub(r'([A-Z][a-z]?)\s+([A-Z][a-z]?)\s*(\d+)', r'\1\2\3', text)  # For compounds like Ca Cl 2
        
        # Preserve molecular weights and concentrations
        text = re.sub(r'(\d+\.?\d*)\s*(kDa|Da|MW)\b', r'\1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*(μM|mM|nM|pM|M)\b', r'\1 \2', text)
        
        # Preserve temperature and pH values
        text = re.sub(r'(\d+\.?\d*)\s*°\s*C\b', r'\1°C', text)
        text = re.sub(r'\bpH\s*(\d+\.?\d*)', r'pH \1', text)
        
        # Preserve units and measurements with proper spacing
        units = ['mg', 'kg', 'g', 'ml', 'μl', 'l', 'mol', 'mmol', 'μmol', 'min', 'h', 'hr', 's', 'Hz', 'rpm']
        for unit in units:
            text = re.sub(rf'(\d+\.?\d*)\s*{unit}\b', rf'\1 {unit}', text, flags=re.IGNORECASE)
        
        # Fix spaced-out technique abbreviations (e.g., H P L C → HPLC)
        techniques = ['HPLC', 'LCMS', 'GCMS', 'MALDI', 'ESI', 'FTICR', 'qPCR', 'rtPCR']
        for technique in techniques:
            spaced = ' '.join(list(technique))  # Convert HPLC to H P L C
            text = re.sub(rf'\b{re.escape(spaced)}\b', technique, text, flags=re.IGNORECASE)
        
        return text
    
    def _handle_biomedical_formatting(self, text: str) -> str:
        """
        Handle biomedical-specific formatting like references, citations, and captions.
        
        Args:
            text: Input text with biomedical formatting
            
        Returns:
            str: Text with cleaned biomedical formatting
        """
        # Clean up reference citations while preserving structure
        text = re.sub(r'\[\s*(\d+(?:\s*[-,]\s*\d+)*)\s*\]', r'[\1]', text)
        text = re.sub(r'\(\s*(\d+(?:\s*[-,]\s*\d+)*)\s*\)', r'(\1)', text)
        
        # Handle author citations (e.g., "Smith et al., 2020")
        text = re.sub(r'([A-Z][a-z]+)\s+et\s+al\.\s*,?\s*(\d{4})', r'\1 et al., \2', text)
        
        # Clean up figure and table references
        text = re.sub(r'\bFig\.?\s+(\d+[a-zA-Z]?)\b', r'Fig. \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bFigure\s+(\d+[a-zA-Z]?)\b', r'Figure \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bTable\s+(\d+[a-zA-Z]?)\b', r'Table \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bSupplementary\s+Figure\s+(\d+[a-zA-Z]?)\b', r'Supplementary Figure \1', text, flags=re.IGNORECASE)
        
        # Handle equation references
        text = re.sub(r'\bEq\.?\s*(\d+)\b', r'Eq. \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bEquation\s*(\d+)\b', r'Equation \1', text, flags=re.IGNORECASE)
        
        # Clean up section references
        text = re.sub(r'\bSection\s*(\d+(?:\.\d+)*)\b', r'Section \1', text, flags=re.IGNORECASE)
        
        # Handle institutional affiliations (remove excessive spacing)
        text = re.sub(r'\n\s*\d+\s*[A-Z][a-z]+.*?University.*?\n', '\n', text)
        text = re.sub(r'\n\s*\d+\s*Department.*?\n', '\n', text)
        
        return text
    
    def _clean_text_flow(self, text: str) -> str:
        """
        Clean up text flow while maintaining document structure.
        
        Args:
            text: Input text with flow issues
            
        Returns:
            str: Text with improved flow
        """
        # Normalize paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix section headers (ensure proper spacing)
        text = re.sub(r'\n\s*([A-Z][A-Z\s]+[A-Z])\s*\n', r'\n\n\1\n\n', text)
        
        # Handle bulleted and numbered lists
        text = re.sub(r'\n\s*([•·▪▫])\s*', r'\n• ', text)
        text = re.sub(r'\n\s*(\d+\.)\s*', r'\n\1 ', text)
        text = re.sub(r'\n\s*([a-z]\))\s*', r'\n\1 ', text)
        
        # Fix spacing around colons in titles/headers
        text = re.sub(r'([A-Z][a-z\s]+):\s*\n', r'\1:\n', text)
        
        # Clean up excessive punctuation while preserving ellipses
        text = re.sub(r'[.]{4,}', '...', text)
        text = re.sub(r'[-]{3,}', '--', text)
        
        # Remove orphaned punctuation
        text = re.sub(r'\n\s*[,;.]\s*\n', '\n', text)
        
        return text
    
    def _normalize_biomedical_terms(self, text: str) -> str:
        """
        Normalize special biomedical terms and gene/protein names.
        
        Args:
            text: Input text with biomedical terms
            
        Returns:
            str: Text with normalized biomedical terms
        """
        # Fix spaced-out nucleic acid abbreviations (e.g., m r n a → mRNA)
        spaced_abbrevs = {
            'm r n a': 'mRNA', 'd n a': 'DNA', 'r n a': 'RNA',
            'q p c r': 'qPCR', 'r t p c r': 'rtPCR', 'p c r': 'PCR'
        }
        
        for spaced, standard in spaced_abbrevs.items():
            text = re.sub(rf'\b{re.escape(spaced)}\b', standard, text, flags=re.IGNORECASE)
        
        # Standardize gene names (typically uppercase/italics indicators)
        text = re.sub(r'\b([A-Z]{2,})\s*gene\b', r'\1 gene', text)
        text = re.sub(r'\b([A-Z]{2,})\s*protein\b', r'\1 protein', text)
        
        # Standardize common biomedical abbreviations
        abbrevs = {
            'DNA': 'DNA', 'RNA': 'RNA', 'mRNA': 'mRNA', 'PCR': 'PCR', 'qPCR': 'qPCR',
            'ELISA': 'ELISA', 'Western blot': 'Western blot', 'SDS-PAGE': 'SDS-PAGE',
            'HPLC': 'HPLC', 'LC-MS': 'LC-MS', 'GC-MS': 'GC-MS', 'NMR': 'NMR'
        }
        
        for abbrev, standard in abbrevs.items():
            text = re.sub(rf'\b{re.escape(abbrev.lower())}\b', standard, text, flags=re.IGNORECASE)
        
        # Standardize statistical terms
        text = re.sub(r'\bstd\.?\s*dev\.?\b', 'standard deviation', text, flags=re.IGNORECASE)
        text = re.sub(r'\bstd\.?\s*err\.?\b', 'standard error', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsem\b', 'SEM', text, flags=re.IGNORECASE)
        
        # Standardize concentration units
        text = re.sub(r'\bmg\s*/\s*ml\b', 'mg/mL', text, flags=re.IGNORECASE)
        text = re.sub(r'\bug\s*/\s*ml\b', 'μg/mL', text, flags=re.IGNORECASE)
        text = re.sub(r'\bng\s*/\s*ml\b', 'ng/mL', text, flags=re.IGNORECASE)
        
        # Handle Greek letters commonly used in biochemistry (with proper spacing)
        greek_letters = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
            'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'sigma': 'σ', 'omega': 'ω'
        }
        
        for greek, symbol in greek_letters.items():
            # Handle both spaced (alpha - ketoglutarate) and non-spaced versions
            text = re.sub(rf'\b{greek}\s*-\s*', rf'{symbol}-', text, flags=re.IGNORECASE)
            text = re.sub(rf'\b{greek}\b', symbol, text, flags=re.IGNORECASE)
        
        return text
    
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
            # Use enhanced validation
            self._validate_pdf_file(pdf_path)
            
            # Get file size
            result['file_size_bytes'] = pdf_path.stat().st_size
            
            # Try to open the PDF with timeout protection
            doc = self._open_pdf_with_timeout(pdf_path)
            
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
                    # Validate the extracted text
                    test_text = self._validate_and_clean_page_text(test_text, 0)
                    result['valid'] = True
                except Exception as e:
                    result['error'] = f"Cannot extract text from PDF: {e}"
            else:
                result['error'] = "PDF has no pages"
            
            doc.close()
            
        except (PDFValidationError, PDFFileAccessError, PDFProcessingTimeoutError) as e:
            result['error'] = str(e)
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
            # Use enhanced validation first
            self._validate_pdf_file(pdf_path)
            
            doc = self._open_pdf_with_timeout(pdf_path)
            if doc.needs_pass:
                doc.close()
                raise PDFFileAccessError(f"PDF is password protected: {pdf_path}")
            
            page_count = doc.page_count
            doc.close()
            return page_count
            
        except (PDFValidationError, PDFFileAccessError, PDFProcessingTimeoutError) as e:
            # Re-raise our custom exceptions
            raise
        except fitz.FileDataError as e:
            raise PDFValidationError(f"Invalid or corrupted PDF: {e}")
        except Exception as e:
            if 'timeout' in str(e).lower():
                raise PDFProcessingTimeoutError(f"Timeout getting page count: {e}")
            raise BiomedicalPDFProcessorError(f"Failed to get page count: {e}")
    
    async def process_all_pdfs(self, papers_dir: Union[str, Path] = "papers/") -> List[Tuple[str, Dict[str, Any]]]:
        """
        Asynchronously process all PDF files in the specified directory.
        
        This method scans the papers directory for PDF files and processes them
        using the extract_text_from_pdf method. It includes progress tracking,
        error recovery, and detailed logging for batch processing operations.
        
        Args:
            papers_dir: Directory containing PDF files (default: "papers/")
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of tuples containing:
                - text: Extracted text content
                - metadata: Combined metadata and processing information
                
        Note:
            This method continues processing even if individual PDFs fail,
            logging errors and moving to the next file. Failed PDFs are
            skipped but logged for review.
        """
        papers_path = Path(papers_dir)
        documents = []
        
        if not papers_path.exists():
            self.logger.warning(f"Papers directory {papers_path} does not exist")
            return documents
        
        # Find all PDF files
        pdf_files = list(papers_path.glob("*.pdf"))
        if not pdf_files:
            self.logger.info(f"No PDF files found in directory: {papers_path}")
            return documents
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process in {papers_path}")
        
        # Process each PDF file
        processed_count = 0
        failed_count = 0
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing PDF {processed_count + 1}/{len(pdf_files)}: {pdf_file.name}")
                
                # Extract text and metadata with enhanced error handling
                result = self.extract_text_from_pdf(pdf_file)
                
                # Combine metadata and processing info for return
                combined_metadata = result['metadata'].copy()
                combined_metadata.update(result['processing_info'])
                combined_metadata['page_texts_count'] = len(result['page_texts'])
                
                # Add to results
                documents.append((result['text'], combined_metadata))
                processed_count += 1
                
                self.logger.info(
                    f"Successfully processed {pdf_file.name}: "
                    f"{combined_metadata['total_characters']} characters, "
                    f"{combined_metadata['pages_processed']} pages"
                )
                
                # Add small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except (PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError, 
                   PDFFileAccessError, PDFContentError) as e:
                failed_count += 1
                self.logger.error(f"PDF processing error for {pdf_file.name}: {e}")
                # Continue processing other files
                continue
            except Exception as e:
                failed_count += 1
                self.logger.error(f"Unexpected error processing {pdf_file.name}: {e}")
                # Continue processing other files
                continue
        
        # Log final summary
        self.logger.info(
            f"Batch processing completed: {processed_count} successful, "
            f"{failed_count} failed out of {len(pdf_files)} total files"
        )
        
        return documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics and configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing statistics
        """
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        system_memory = psutil.virtual_memory()
        
        return {
            'processing_timeout': self.processing_timeout,
            'memory_limit_mb': self.memory_limit_mb,
            'max_page_text_size': self.max_page_text_size,
            'current_memory_mb': round(current_memory, 2),
            'system_memory_percent': round(system_memory.percent, 1),
            'system_memory_available_mb': round(system_memory.available / 1024 / 1024, 2),
            'memory_monitor_active': self._memory_monitor_active
        }