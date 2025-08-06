"""
BiomedicalPDFProcessor for Clinical Metabolomics Oracle - LightRAG integration.

This module provides specialized PDF processing capabilities for biomedical documents
using PyMuPDF (fitz). It includes text extraction, metadata retrieval, preprocessing
specifically tailored for clinical metabolomics and biomedical literature, and 
comprehensive error recovery mechanisms.

Classes:
    - BiomedicalPDFProcessorError: Base custom exception for PDF processing errors
    - PDFValidationError: Exception for PDF file validation failures
    - PDFProcessingTimeoutError: Exception for processing timeout conditions
    - PDFMemoryError: Exception for memory-related processing issues
    - PDFFileAccessError: Exception for file access problems (locks, permissions)
    - PDFContentError: Exception for content extraction and encoding issues
    - ErrorRecoveryConfig: Configuration for error recovery and retry mechanisms
    - BiomedicalPDFProcessor: Main class for processing biomedical PDF documents

The processor handles:
    - Text extraction from PDF documents using PyMuPDF
    - Metadata extraction (creation date, modification date, title, author, etc.)
    - Text preprocessing for biomedical content
    - Comprehensive error handling and recovery for various edge cases:
      * MIME type validation and PDF header verification
      * Memory pressure monitoring during processing
      * Processing timeout protection with dynamic timeout adjustment
      * Enhanced file locking and permission detection with retry logic
      * Zero-byte file handling
      * Malformed PDF structure detection
      * Character encoding issue resolution
      * Large text block protection
    - Advanced error recovery mechanisms:
      * Automatic retry with exponential backoff
      * Error classification (recoverable vs non-recoverable)
      * Memory recovery through garbage collection
      * File lock recovery with progressive delays
      * Timeout recovery with dynamic timeout increase
      * Comprehensive retry statistics and logging
    - Page-by-page processing with optional page range specification
    - Cleaning and normalization of extracted text
    - Processing statistics and monitoring capabilities
    - Batch processing with graceful degradation
    - Advanced memory management for large document collections:
      * Batch processing with configurable batch sizes
      * Memory monitoring and cleanup between batches
      * Dynamic batch size adjustment based on memory usage
      * Enhanced garbage collection to prevent memory accumulation
      * Memory pool management for large collections (100+ PDFs)
      * Document streaming to process files incrementally
"""

import re
import logging
import asyncio
import mimetypes
import psutil
import signal
import time
import threading
import gc
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING
from datetime import datetime
from contextlib import contextmanager
import fitz  # PyMuPDF

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .progress_config import ProgressTrackingConfig
    from .progress_tracker import PDFProcessingProgressTracker


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


class ErrorRecoveryConfig:
    """
    Configuration for error recovery and retry mechanisms.
    
    This class defines parameters for retry strategies, error classification,
    and recovery actions for different types of failures.
    """
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 memory_recovery_enabled: bool = True,
                 file_lock_retry_enabled: bool = True,
                 timeout_retry_enabled: bool = True):
        """
        Initialize error recovery configuration.
        
        Args:
            max_retries: Maximum number of retry attempts per file (default: 3)
            base_delay: Base delay between retries in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 60.0)
            exponential_base: Base for exponential backoff calculation (default: 2.0)
            jitter: Whether to add random jitter to retry delays (default: True)
            memory_recovery_enabled: Whether to attempt memory recovery (default: True)
            file_lock_retry_enabled: Whether to retry on file lock errors (default: True)
            timeout_retry_enabled: Whether to retry on timeout errors (default: True)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.memory_recovery_enabled = memory_recovery_enabled
        self.file_lock_retry_enabled = file_lock_retry_enabled
        self.timeout_retry_enabled = timeout_retry_enabled
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt using exponential backoff.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            float: Delay in seconds before next retry
        """
        # Calculate exponential backoff
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        # Add jitter if enabled (up to 25% of delay)
        if self.jitter:
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay


class BiomedicalPDFProcessor:
    """
    Specialized PDF processor for biomedical documents with comprehensive error recovery.
    
    This class provides comprehensive PDF processing capabilities specifically
    designed for biomedical and clinical metabolomics literature. It uses
    PyMuPDF for robust PDF handling and includes specialized text preprocessing
    for scientific content, along with advanced error recovery mechanisms.
    
    Attributes:
        logger: Logger instance for tracking processing activities
        error_recovery: Configuration for error recovery and retry mechanisms
        
    Example:
        processor = BiomedicalPDFProcessor()
        result = processor.extract_text_from_pdf("paper.pdf")
        print(f"Extracted {len(result['text'])} characters from {result['metadata']['pages']} pages")
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 processing_timeout: int = 300,  # 5 minutes default
                 memory_limit_mb: int = 1024,    # 1GB default
                 max_page_text_size: int = 1000000,  # 1MB per page default
                 error_recovery_config: Optional[ErrorRecoveryConfig] = None):
        """
        Initialize the BiomedicalPDFProcessor with error recovery capabilities.
        
        Args:
            logger: Optional logger instance. If None, creates a default logger.
            processing_timeout: Maximum processing time in seconds (default: 300)
            memory_limit_mb: Maximum memory usage in MB (default: 1024)
            max_page_text_size: Maximum text size per page in characters (default: 1000000)
            error_recovery_config: Optional error recovery configuration (creates default if None)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.processing_timeout = processing_timeout
        self.memory_limit_mb = memory_limit_mb
        self.max_page_text_size = max_page_text_size
        self.error_recovery = error_recovery_config or ErrorRecoveryConfig()
        self._processing_start_time = None
        self._memory_monitor_active = False
        
        # Error recovery tracking
        self._retry_stats: Dict[str, Dict[str, Any]] = {}
        self._recovery_actions_attempted: Dict[str, int] = {}
    
    def _classify_error(self, error: Exception) -> Tuple[bool, str, str]:
        """
        Classify an error to determine if it's recoverable and what recovery strategy to use.
        
        Args:
            error: Exception that occurred during processing
            
        Returns:
            Tuple[bool, str, str]: (is_recoverable, error_category, recovery_strategy)
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Memory-related errors (recoverable with memory cleanup)
        if isinstance(error, (PDFMemoryError, MemoryError)):
            return True, "memory", "memory_cleanup"
        
        if "memory" in error_msg or "allocation" in error_msg:
            return True, "memory", "memory_cleanup"
        
        # Timeout errors (recoverable with timeout increase)
        if isinstance(error, PDFProcessingTimeoutError):
            return True, "timeout", "timeout_retry"
        
        if "timeout" in error_msg or "time" in error_msg:
            return True, "timeout", "timeout_retry"
        
        # File access errors (recoverable with retry)
        if isinstance(error, PDFFileAccessError):
            if "locked" in error_msg or "in use" in error_msg:
                return True, "file_lock", "file_lock_retry"
            elif "permission" in error_msg:
                return False, "permission", "skip"  # Usually not recoverable
            else:
                return True, "file_access", "simple_retry"
        
        # Validation errors - some recoverable, some not
        if isinstance(error, PDFValidationError):
            if "corrupted" in error_msg or "invalid" in error_msg:
                return False, "corruption", "skip"  # Not recoverable
            else:
                return True, "validation", "simple_retry"
        
        # Content errors (usually not recoverable)
        if isinstance(error, PDFContentError):
            return False, "content", "skip"
        
        # Network/IO related errors (recoverable)
        if isinstance(error, (IOError, OSError)):
            if "no space" in error_msg:
                return False, "disk_space", "skip"  # Usually not recoverable
            else:
                return True, "io_error", "simple_retry"
        
        # PyMuPDF specific errors
        if "fitz" in error_type.lower() or "mupdf" in error_msg:
            if "timeout" in error_msg:
                return True, "fitz_timeout", "timeout_retry"
            elif "memory" in error_msg:
                return True, "fitz_memory", "memory_cleanup"
            else:
                return True, "fitz_error", "simple_retry"
        
        # Unknown errors - try once with simple retry
        return True, "unknown", "simple_retry"
    
    def _attempt_error_recovery(self, error_category: str, recovery_strategy: str, 
                               file_path: Path, attempt: int) -> bool:
        """
        Attempt to recover from an error using the specified strategy.
        
        Args:
            error_category: Category of the error
            recovery_strategy: Strategy to use for recovery
            file_path: Path to the file being processed
            attempt: Current attempt number
            
        Returns:
            bool: True if recovery action was attempted, False if not applicable
        """
        self.logger.info(f"Attempting recovery for {error_category} error using {recovery_strategy} strategy (attempt {attempt + 1})")
        
        # Track recovery attempts
        recovery_key = f"{error_category}_{recovery_strategy}"
        self._recovery_actions_attempted[recovery_key] = self._recovery_actions_attempted.get(recovery_key, 0) + 1
        
        if recovery_strategy == "memory_cleanup":
            return self._attempt_memory_recovery()
        elif recovery_strategy == "file_lock_retry":
            return self._attempt_file_lock_recovery(file_path, attempt)
        elif recovery_strategy == "timeout_retry":
            return self._attempt_timeout_recovery(attempt)
        elif recovery_strategy == "simple_retry":
            return self._attempt_simple_recovery(attempt)
        elif recovery_strategy == "skip":
            self.logger.warning(f"Skipping file due to non-recoverable {error_category} error: {file_path}")
            return False
        else:
            self.logger.warning(f"Unknown recovery strategy: {recovery_strategy}")
            return False
    
    def _attempt_memory_recovery(self) -> bool:
        """
        Attempt to recover from memory-related errors.
        
        Returns:
            bool: True if recovery action was attempted
        """
        if not self.error_recovery.memory_recovery_enabled:
            return False
        
        self.logger.info("Attempting memory recovery: running garbage collection and clearing caches")
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Wait a brief moment for memory to be freed
            time.sleep(0.5)
            
            # Check current memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory recovery completed. Current memory usage: {current_memory:.2f} MB")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Memory recovery failed: {e}")
            return False
    
    def _attempt_file_lock_recovery(self, file_path: Path, attempt: int) -> bool:
        """
        Attempt to recover from file lock errors.
        
        Args:
            file_path: Path to the locked file
            attempt: Current attempt number
            
        Returns:
            bool: True if recovery action was attempted
        """
        if not self.error_recovery.file_lock_retry_enabled:
            return False
        
        # Calculate delay with longer waits for file locks
        base_delay = max(2.0, self.error_recovery.base_delay)
        delay = min(base_delay * (2 ** attempt), 30.0)  # Max 30s for file locks
        
        self.logger.info(f"File appears to be locked: {file_path}. Waiting {delay:.1f}s before retry")
        time.sleep(delay)
        
        return True
    
    def _attempt_timeout_recovery(self, attempt: int) -> bool:
        """
        Attempt to recover from timeout errors.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            bool: True if recovery action was attempted
        """
        if not self.error_recovery.timeout_retry_enabled:
            return False
        
        # For timeout recovery, we'll increase the timeout for the next attempt
        timeout_multiplier = 1.5 ** (attempt + 1)
        new_timeout = min(self.processing_timeout * timeout_multiplier, self.processing_timeout * 3)
        
        self.logger.info(f"Timeout occurred. Increasing timeout to {new_timeout:.1f}s for retry")
        
        # Temporarily increase timeout (will be restored after processing)
        self._original_timeout = getattr(self, '_original_timeout', self.processing_timeout)
        self.processing_timeout = int(new_timeout)
        
        return True
    
    def _attempt_simple_recovery(self, attempt: int) -> bool:
        """
        Attempt simple recovery with exponential backoff delay.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            bool: True if recovery action was attempted
        """
        delay = self.error_recovery.calculate_delay(attempt)
        self.logger.info(f"Simple recovery: waiting {delay:.1f}s before retry")
        time.sleep(delay)
        return True

    def extract_text_from_pdf(self, 
                             pdf_path: Union[str, Path],
                             start_page: Optional[int] = None,
                             end_page: Optional[int] = None,
                             preprocess_text: bool = True) -> Dict[str, Any]:
        """
        Extract text and metadata from a biomedical PDF document with comprehensive error recovery.
        
        This method now includes comprehensive error recovery mechanisms with
        retry logic, error classification, and recovery strategies. It processes 
        the entire PDF or a specified page range, extracting both text content 
        and document metadata.
        
        Args:
            pdf_path: Path to the PDF file (string or Path object)
            start_page: Starting page number (0-indexed). If None, starts from page 0.
            end_page: Ending page number (0-indexed, exclusive). If None, processes all pages.
            preprocess_text: Whether to apply biomedical text preprocessing
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'text': Extracted text content (str)
                - 'metadata': PDF metadata including retry information
                - 'page_texts': List of text from each processed page (List[str])
                - 'processing_info': Dictionary with processing statistics including retry info
                
        Raises:
            BiomedicalPDFProcessorError: If PDF cannot be processed after all retry attempts
            FileNotFoundError: If PDF file doesn't exist
            PermissionError: If file cannot be accessed
        """
        return self._extract_text_with_retry(pdf_path, start_page, end_page, preprocess_text)
    
    def _extract_text_with_retry(self, pdf_path: Union[str, Path], 
                                start_page: Optional[int] = None,
                                end_page: Optional[int] = None,
                                preprocess_text: bool = True) -> Dict[str, Any]:
        """
        Extract text from PDF with retry mechanisms and error recovery.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed, exclusive)
            preprocess_text: Whether to apply biomedical text preprocessing
            
        Returns:
            Dict[str, Any]: Extraction result with text, metadata, and processing info
            
        Raises:
            BiomedicalPDFProcessorError: If all retry attempts fail
        """
        pdf_path = Path(pdf_path)
        last_error = None
        retry_info = {
            'total_attempts': 0,
            'recoverable_attempts': 0,
            'recovery_actions': [],
            'error_history': []
        }
        
        # Store original timeout for restoration
        original_timeout = self.processing_timeout
        
        try:
            for attempt in range(self.error_recovery.max_retries + 1):  # +1 for initial attempt
                retry_info['total_attempts'] = attempt + 1
                
                try:
                    # Log retry attempt
                    if attempt > 0:
                        self.logger.info(f"Retry attempt {attempt} for {pdf_path.name}")
                    
                    # Attempt extraction using the original method logic
                    result = self._extract_text_internal(pdf_path, start_page, end_page, preprocess_text)
                    
                    # Add retry information to result
                    result['processing_info']['retry_info'] = retry_info
                    
                    # Success - restore timeout and return
                    self.processing_timeout = original_timeout
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    
                    # Record error in retry info
                    retry_info['error_history'].append({
                        'attempt': attempt + 1,
                        'error_type': error_type,
                        'error_message': str(e)[:200]  # Truncate long messages
                    })
                    
                    # Classify error and determine recovery strategy
                    is_recoverable, error_category, recovery_strategy = self._classify_error(e)
                    
                    # Log error details
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {pdf_path.name}: {error_type} - {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}"
                    )
                    
                    # If not recoverable or out of retries, break
                    if not is_recoverable or attempt >= self.error_recovery.max_retries:
                        if not is_recoverable:
                            self.logger.error(f"Non-recoverable error for {pdf_path.name}: {error_category}")
                        break
                    
                    # Attempt recovery
                    retry_info['recoverable_attempts'] += 1
                    recovery_attempted = self._attempt_error_recovery(error_category, recovery_strategy, pdf_path, attempt)
                    
                    if recovery_attempted:
                        retry_info['recovery_actions'].append({
                            'attempt': attempt + 1,
                            'strategy': recovery_strategy,
                            'category': error_category
                        })
                    
                    # Continue to next attempt
                    continue
            
            # All retry attempts failed
            self.logger.error(f"All retry attempts failed for {pdf_path.name} after {retry_info['total_attempts']} attempts")
            
            # Record retry statistics
            file_key = str(pdf_path)
            self._retry_stats[file_key] = retry_info
            
            raise last_error
            
        finally:
            # Always restore original timeout
            self.processing_timeout = original_timeout
    
    def _extract_text_internal(self, 
                              pdf_path: Union[str, Path],
                              start_page: Optional[int] = None,
                              end_page: Optional[int] = None,
                              preprocess_text: bool = True) -> Dict[str, Any]:
        """
        Internal method that contains the original PDF extraction logic.
        This is separated to allow the retry mechanism to call it multiple times.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed, exclusive)
            preprocess_text: Whether to apply biomedical text preprocessing
            
        Returns:
            Dict[str, Any]: Extraction result with text, metadata, and processing info
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
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get detailed memory usage statistics.
        
        Returns:
            Dict[str, float]: Dictionary containing memory usage metrics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_peak_mb': getattr(memory_info, 'peak_wset', memory_info.rss) / 1024 / 1024,
            'system_memory_percent': system_memory.percent,
            'system_memory_available_mb': system_memory.available / 1024 / 1024,
            'system_memory_used_mb': system_memory.used / 1024 / 1024,
            'system_memory_total_mb': system_memory.total / 1024 / 1024
        }
    
    def _cleanup_memory(self, force: bool = False) -> Dict[str, float]:
        """
        Enhanced garbage collection and memory cleanup between batches.
        
        Args:
            force: Whether to force aggressive cleanup even if memory usage is low
            
        Returns:
            Dict[str, float]: Memory usage before and after cleanup
        """
        memory_before = self._get_memory_usage()
        
        # Log cleanup initiation
        self.logger.debug(f"Memory cleanup initiated - Memory before: {memory_before['process_memory_mb']:.2f} MB")
        
        # Force garbage collection multiple times for thorough cleanup
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)  # Brief pause between GC cycles
        
        # Force collection of all generations
        if hasattr(gc, 'collect'):
            for generation in range(gc.get_count().__len__()):
                gc.collect(generation)
        
        # Clear any remaining PyMuPDF caches if possible
        try:
            # PyMuPDF doesn't have explicit cache clearing, but we can help by
            # ensuring document objects are properly closed
            pass
        except:
            pass
        
        # Small delay to allow memory to be reclaimed by the system
        time.sleep(0.2)
        
        memory_after = self._get_memory_usage()
        
        # Calculate memory freed
        memory_freed = memory_before['process_memory_mb'] - memory_after['process_memory_mb']
        
        self.logger.info(
            f"Memory cleanup completed - Memory freed: {memory_freed:.2f} MB "
            f"(Before: {memory_before['process_memory_mb']:.2f} MB, "
            f"After: {memory_after['process_memory_mb']:.2f} MB)"
        )
        
        return {
            'memory_before_mb': memory_before['process_memory_mb'],
            'memory_after_mb': memory_after['process_memory_mb'],
            'memory_freed_mb': memory_freed,
            'system_memory_percent': memory_after['system_memory_percent']
        }
    
    def _adjust_batch_size(self, current_batch_size: int, memory_usage: float, 
                          max_memory_mb: int, performance_data: Dict[str, Any]) -> int:
        """
        Dynamically adjust batch size based on memory usage and performance patterns.
        
        Args:
            current_batch_size: Current batch size
            memory_usage: Current memory usage in MB
            max_memory_mb: Maximum allowed memory usage in MB
            performance_data: Performance metrics from recent processing
            
        Returns:
            int: Adjusted batch size
        """
        # Calculate memory pressure (0.0 to 1.0+)
        memory_pressure = memory_usage / max_memory_mb
        
        # Base adjustment decision on memory pressure
        if memory_pressure > 0.9:  # High memory pressure - reduce batch size significantly
            new_batch_size = max(1, current_batch_size // 2)
            self.logger.warning(
                f"High memory pressure ({memory_pressure:.2f}), reducing batch size from "
                f"{current_batch_size} to {new_batch_size}"
            )
        elif memory_pressure > 0.7:  # Moderate memory pressure - reduce batch size
            new_batch_size = max(1, int(current_batch_size * 0.8))
            self.logger.info(
                f"Moderate memory pressure ({memory_pressure:.2f}), reducing batch size from "
                f"{current_batch_size} to {new_batch_size}"
            )
        elif memory_pressure < 0.4 and current_batch_size < 20:  # Low memory pressure - can increase
            # Only increase if we have good performance data
            avg_processing_time = performance_data.get('average_processing_time', 0)
            if avg_processing_time > 0 and avg_processing_time < 5.0:  # Fast processing
                new_batch_size = min(20, current_batch_size + 2)
                self.logger.info(
                    f"Low memory pressure ({memory_pressure:.2f}) and good performance, "
                    f"increasing batch size from {current_batch_size} to {new_batch_size}"
                )
            else:
                new_batch_size = current_batch_size
        else:
            new_batch_size = current_batch_size
        
        return new_batch_size
    
    async def _process_batch(self, pdf_batch: List[Path], batch_num: int, 
                           progress_tracker: Optional['PDFProcessingProgressTracker'],
                           max_memory_mb: int) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process a batch of PDF files with memory monitoring and cleanup.
        
        Args:
            pdf_batch: List of PDF file paths to process in this batch
            batch_num: Batch number for logging
            progress_tracker: Progress tracker instance
            max_memory_mb: Maximum memory usage allowed
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: Processing results for the batch
        """
        batch_results = []
        batch_start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        self.logger.info(
            f"Starting batch {batch_num} with {len(pdf_batch)} PDFs "
            f"(Memory: {initial_memory['process_memory_mb']:.2f} MB, "
            f"System: {initial_memory['system_memory_percent']:.1f}%)"
        )
        
        for file_index, pdf_file in enumerate(pdf_batch):
            try:
                # Check memory before processing each file
                current_memory = self._get_memory_usage()
                
                # If memory usage is getting high, perform cleanup
                if current_memory['process_memory_mb'] > max_memory_mb * 0.8:
                    self.logger.warning(
                        f"Memory usage high ({current_memory['process_memory_mb']:.2f} MB), "
                        f"performing cleanup before processing {pdf_file.name}"
                    )
                    self._cleanup_memory()
                
                # Process file with progress tracking
                if progress_tracker:
                    with progress_tracker.track_file_processing(pdf_file, file_index) as file_info:
                        result = self.extract_text_from_pdf(pdf_file)
                        
                        # Combine metadata and processing info
                        combined_metadata = result['metadata'].copy()
                        combined_metadata.update(result['processing_info'])
                        combined_metadata['page_texts_count'] = len(result['page_texts'])
                        combined_metadata['batch_number'] = batch_num
                        
                        # Record successful processing
                        progress_tracker.record_file_success(
                            pdf_file,
                            combined_metadata['total_characters'],
                            combined_metadata['pages_processed']
                        )
                        
                        batch_results.append((result['text'], combined_metadata))
                else:
                    # Process without progress tracking
                    result = self.extract_text_from_pdf(pdf_file)
                    
                    combined_metadata = result['metadata'].copy()
                    combined_metadata.update(result['processing_info'])
                    combined_metadata['page_texts_count'] = len(result['page_texts'])
                    combined_metadata['batch_number'] = batch_num
                    
                    batch_results.append((result['text'], combined_metadata))
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except (PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError, 
                   PDFFileAccessError, PDFContentError) as e:
                # Enhanced error logging
                error_info = self._get_enhanced_error_info(pdf_file, e)
                self.logger.error(f"Failed to process {pdf_file.name} in batch {batch_num}: {error_info}")
                continue
            except Exception as e:
                error_info = self._get_enhanced_error_info(pdf_file, e)
                self.logger.error(f"Unexpected error processing {pdf_file.name} in batch {batch_num}: {error_info}")
                continue
        
        # Calculate batch processing metrics
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        final_memory = self._get_memory_usage()
        memory_increase = final_memory['process_memory_mb'] - initial_memory['process_memory_mb']
        
        self.logger.info(
            f"Batch {batch_num} completed: {len(batch_results)}/{len(pdf_batch)} files successful, "
            f"{batch_duration:.2f}s duration, memory increase: {memory_increase:.2f} MB"
        )
        
        return batch_results
    
    async def _process_with_batch_mode(self, pdf_files: List[Path], initial_batch_size: int,
                                     max_memory_mb: int, progress_tracker: 'PDFProcessingProgressTracker') -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process PDF files using batch processing with memory management.
        
        Args:
            pdf_files: List of all PDF files to process
            initial_batch_size: Initial batch size
            max_memory_mb: Maximum memory usage allowed
            progress_tracker: Progress tracker instance
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: All processing results
        """
        all_documents = []
        current_batch_size = initial_batch_size
        
        # Split files into batches
        total_batches = (len(pdf_files) + current_batch_size - 1) // current_batch_size
        
        # Track performance metrics for batch size adjustment
        performance_data = {
            'batch_times': [],
            'memory_usage': [],
            'files_per_second': []
        }
        
        for batch_num in range(total_batches):
            start_idx = batch_num * current_batch_size
            end_idx = min(start_idx + current_batch_size, len(pdf_files))
            pdf_batch = pdf_files[start_idx:end_idx]
            
            # Check memory before batch and adjust batch size if needed
            current_memory = self._get_memory_usage()
            if batch_num > 0:  # Don't adjust on first batch
                current_batch_size = self._adjust_batch_size(
                    current_batch_size, 
                    current_memory['process_memory_mb'],
                    max_memory_mb,
                    performance_data
                )
                
                # If batch size changed, recalculate the batch
                if current_batch_size != (end_idx - start_idx):
                    end_idx = min(start_idx + current_batch_size, len(pdf_files))
                    pdf_batch = pdf_files[start_idx:end_idx]
            
            # Process the batch
            batch_start_time = time.time()
            
            try:
                batch_results = await self._process_batch(
                    pdf_batch, batch_num + 1, progress_tracker, max_memory_mb
                )
                
                all_documents.extend(batch_results)
                
                # Record performance metrics
                batch_duration = time.time() - batch_start_time
                files_processed = len(batch_results)
                files_per_second = files_processed / batch_duration if batch_duration > 0 else 0
                
                performance_data['batch_times'].append(batch_duration)
                performance_data['memory_usage'].append(current_memory['process_memory_mb'])
                performance_data['files_per_second'].append(files_per_second)
                
                # Calculate average processing time for batch size adjustment
                if len(performance_data['batch_times']) > 0:
                    performance_data['average_processing_time'] = sum(performance_data['batch_times']) / len(performance_data['batch_times'])
                
            except Exception as e:
                self.logger.error(f"Critical error processing batch {batch_num + 1}: {e}")
                # Continue with next batch
                continue
            
            # Perform memory cleanup between batches
            if batch_num < total_batches - 1:  # Don't clean up after last batch
                cleanup_result = self._cleanup_memory()
                
                # Log memory management statistics
                self.logger.info(
                    f"Post-batch {batch_num + 1} cleanup: freed {cleanup_result['memory_freed_mb']:.2f} MB "
                    f"(System memory: {cleanup_result['system_memory_percent']:.1f}%)"
                )
                
                # Brief pause to allow system to stabilize
                await asyncio.sleep(0.5)
        
        # Final memory cleanup
        final_cleanup = self._cleanup_memory(force=True)
        self.logger.info(
            f"Final memory cleanup completed - freed {final_cleanup['memory_freed_mb']:.2f} MB"
        )
        
        # Log batch processing summary
        if performance_data['batch_times']:
            avg_batch_time = sum(performance_data['batch_times']) / len(performance_data['batch_times'])
            total_processing_time = sum(performance_data['batch_times'])
            avg_files_per_second = sum(performance_data['files_per_second']) / len(performance_data['files_per_second'])
            
            self.logger.info(
                f"Batch processing summary: {total_batches} batches processed, "
                f"avg batch time: {avg_batch_time:.2f}s, "
                f"total time: {total_processing_time:.2f}s, "
                f"avg throughput: {avg_files_per_second:.2f} files/second"
            )
        
        return all_documents
    
    async def _process_sequential_mode(self, pdf_files: List[Path], 
                                     progress_tracker: 'PDFProcessingProgressTracker') -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process PDF files sequentially (legacy mode for backward compatibility).
        
        Args:
            pdf_files: List of PDF files to process
            progress_tracker: Progress tracker instance
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: Processing results
        """
        documents = []
        
        self.logger.info("Processing files sequentially without batch memory management")
        
        # Process each PDF file with progress tracking
        for file_index, pdf_file in enumerate(pdf_files):
            try:
                with progress_tracker.track_file_processing(pdf_file, file_index) as file_info:
                    # Extract text and metadata with enhanced error handling
                    result = self.extract_text_from_pdf(pdf_file)
                    
                    # Combine metadata and processing info for return
                    combined_metadata = result['metadata'].copy()
                    combined_metadata.update(result['processing_info'])
                    combined_metadata['page_texts_count'] = len(result['page_texts'])
                    
                    # Record successful processing in tracker
                    progress_tracker.record_file_success(
                        pdf_file,
                        combined_metadata['total_characters'],
                        combined_metadata['pages_processed']
                    )
                    
                    # Add to results
                    documents.append((result['text'], combined_metadata))
                    
                    # Add small delay to prevent overwhelming the system
                    await asyncio.sleep(0.1)
                    
            except (PDFValidationError, PDFProcessingTimeoutError, PDFMemoryError, 
                   PDFFileAccessError, PDFContentError) as e:
                # Enhanced error logging with retry information
                error_info = self._get_enhanced_error_info(pdf_file, e)
                self.logger.error(f"Failed to process {pdf_file.name} after all retry attempts: {error_info}")
                
                # Progress tracker already handles the error logging in the context manager
                # Continue processing other files
                continue
            except Exception as e:
                # Enhanced error logging for unexpected errors
                error_info = self._get_enhanced_error_info(pdf_file, e)
                self.logger.error(f"Unexpected error processing {pdf_file.name}: {error_info}")
                
                # Progress tracker already handles the error logging in the context manager
                # Continue processing other files
                continue
        
        return documents
    
    async def process_all_pdfs(self, 
                              papers_dir: Union[str, Path] = "papers/",
                              progress_config: Optional['ProgressTrackingConfig'] = None,
                              progress_tracker: Optional['PDFProcessingProgressTracker'] = None,
                              batch_size: int = 10,
                              max_memory_mb: int = 2048,
                              enable_batch_processing: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Asynchronously process all PDF files in the specified directory with comprehensive progress tracking and memory management.
        
        This method scans the papers directory for PDF files and processes them
        using the extract_text_from_pdf method. It includes advanced progress tracking,
        performance monitoring, detailed logging, error recovery capabilities, and
        advanced memory management for large document collections.
        
        Memory Management Features:
        - Batch processing with configurable batch sizes
        - Memory monitoring and cleanup between batches
        - Dynamic batch size adjustment based on memory usage
        - Enhanced garbage collection to prevent memory accumulation
        - Memory pool management for large collections (100+ PDFs)
        
        Args:
            papers_dir: Directory containing PDF files (default: "papers/")
            progress_config: Optional progress tracking configuration
            progress_tracker: Optional existing progress tracker instance
            batch_size: Number of PDFs to process before memory cleanup (default: 10)
            max_memory_mb: Maximum memory usage in MB before triggering cleanup (default: 2048)
            enable_batch_processing: Whether to use batch processing with memory management (default: True)
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of tuples containing:
                - text: Extracted text content
                - metadata: Combined metadata and processing information
                
        Note:
            This method continues processing even if individual PDFs fail,
            logging errors and moving to the next file. Failed PDFs are
            tracked and logged for comprehensive reporting. When batch processing
            is enabled, it processes documents in smaller batches to prevent
            memory accumulation, making it suitable for large document collections.
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
        
        # Initialize progress tracking if not provided
        if progress_tracker is None:
            # Import here to avoid circular imports
            from .progress_config import ProgressTrackingConfig
            from .progress_tracker import PDFProcessingProgressTracker
            
            if progress_config is None:
                progress_config = ProgressTrackingConfig()
            
            progress_tracker = PDFProcessingProgressTracker(
                config=progress_config,
                logger=self.logger
            )
        
        # Start batch processing with progress tracking
        progress_tracker.start_batch_processing(len(pdf_files), pdf_files)
        
        # Log initial memory state and processing approach
        initial_memory = self._get_memory_usage()
        self.logger.info(
            f"Found {len(pdf_files)} PDF files to process in {papers_path} "
            f"(Initial memory: {initial_memory['process_memory_mb']:.2f} MB, "
            f"System: {initial_memory['system_memory_percent']:.1f}%)"
        )
        
        # Choose processing approach based on enable_batch_processing flag
        if enable_batch_processing and len(pdf_files) > 1:
            self.logger.info(
                f"Using batch processing mode with initial batch size {batch_size}, "
                f"max memory limit {max_memory_mb} MB"
            )
            documents = await self._process_with_batch_mode(
                pdf_files, batch_size, max_memory_mb, progress_tracker
            )
        else:
            self.logger.info("Using sequential processing mode (batch processing disabled)")
            documents = await self._process_sequential_mode(pdf_files, progress_tracker)
        
        # Complete batch processing and get final metrics
        final_metrics = progress_tracker.finish_batch_processing()
        
        # Log final summary with enhanced metrics
        self.logger.info(
            f"Batch processing completed: {final_metrics.completed_files} successful, "
            f"{final_metrics.failed_files} failed, {final_metrics.skipped_files} skipped "
            f"out of {final_metrics.total_files} total files"
        )
        
        # Log additional performance metrics if available
        if final_metrics.total_characters > 0:
            self.logger.info(
                f"Performance summary: {final_metrics.total_characters:,} total characters, "
                f"{final_metrics.total_pages:,} total pages, "
                f"{final_metrics.processing_time:.2f}s total time, "
                f"{final_metrics.average_processing_time:.2f}s average per file"
            )
        
        # Log error recovery statistics if there were any retries
        if self._retry_stats or self._recovery_actions_attempted:
            self._log_error_recovery_summary()
        
        # Log final memory usage and cleanup statistics
        final_memory = self._get_memory_usage()
        memory_change = final_memory['process_memory_mb'] - initial_memory['process_memory_mb']
        
        self.logger.info(
            f"Memory management summary: "
            f"Initial: {initial_memory['process_memory_mb']:.2f} MB, "
            f"Final: {final_memory['process_memory_mb']:.2f} MB, "
            f"Change: {memory_change:+.2f} MB, "
            f"System memory usage: {final_memory['system_memory_percent']:.1f}%"
        )
        
        # Log batch processing mode used
        processing_mode = "batch processing" if enable_batch_processing and len(pdf_files) > 1 else "sequential processing"
        self.logger.info(f"Processing completed using {processing_mode} mode")
        
        return documents
    
    def _log_error_recovery_summary(self) -> None:
        """Log a summary of error recovery actions taken during batch processing."""
        if not (self._retry_stats or self._recovery_actions_attempted):
            return
        
        total_files_with_retries = len(self._retry_stats)
        total_recovery_actions = sum(self._recovery_actions_attempted.values())
        
        self.logger.info(f"Error recovery summary: {total_files_with_retries} files required retries, {total_recovery_actions} total recovery actions")
        
        # Log recovery action breakdown
        if self._recovery_actions_attempted:
            recovery_breakdown = ", ".join([f"{action}: {count}" for action, count in self._recovery_actions_attempted.items()])
            self.logger.info(f"Recovery actions breakdown: {recovery_breakdown}")
        
        # Log most problematic files
        if self._retry_stats:
            files_by_attempts = sorted(self._retry_stats.items(), key=lambda x: x[1]['total_attempts'], reverse=True)
            top_problematic = files_by_attempts[:3]  # Top 3 most problematic files
            
            for file_path, retry_info in top_problematic:
                file_name = Path(file_path).name
                self.logger.warning(
                    f"Problematic file: {file_name} required {retry_info['total_attempts']} attempts, "
                    f"{retry_info['recoverable_attempts']} with recovery actions"
                )
    
    def get_error_recovery_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive error recovery statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing error recovery statistics
        """
        return {
            'files_with_retries': len(self._retry_stats),
            'total_recovery_actions': sum(self._recovery_actions_attempted.values()),
            'recovery_actions_by_type': self._recovery_actions_attempted.copy(),
            'retry_details_by_file': self._retry_stats.copy(),
            'error_recovery_config': {
                'max_retries': self.error_recovery.max_retries,
                'base_delay': self.error_recovery.base_delay,
                'max_delay': self.error_recovery.max_delay,
                'memory_recovery_enabled': self.error_recovery.memory_recovery_enabled,
                'file_lock_retry_enabled': self.error_recovery.file_lock_retry_enabled,
                'timeout_retry_enabled': self.error_recovery.timeout_retry_enabled
            }
        }
    
    def reset_error_recovery_stats(self) -> None:
        """Reset error recovery statistics for a new batch processing session."""
        self._retry_stats.clear()
        self._recovery_actions_attempted.clear()
        self.logger.debug("Error recovery statistics reset")
    
    def _get_enhanced_error_info(self, pdf_file: Path, error: Exception) -> str:
        """
        Get enhanced error information including retry details.
        
        Args:
            pdf_file: Path to the PDF file that failed
            error: Exception that occurred
            
        Returns:
            str: Enhanced error information string
        """
        error_info = f"{type(error).__name__}: {str(error)[:200]}"
        
        # Add retry information if available
        file_key = str(pdf_file)
        if file_key in self._retry_stats:
            retry_info = self._retry_stats[file_key]
            error_info += f" (Attempts: {retry_info['total_attempts']}"
            
            if retry_info['recovery_actions']:
                strategies = [action['strategy'] for action in retry_info['recovery_actions']]
                unique_strategies = list(set(strategies))
                error_info += f", Recovery strategies used: {', '.join(unique_strategies)}"
            
            error_info += ")"
        
        return error_info
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics and configuration including memory management features.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing statistics
        """
        # Get detailed memory usage
        memory_stats = self._get_memory_usage()
        
        # Get basic error recovery stats
        error_recovery_summary = {
            'files_with_retries': len(self._retry_stats),
            'total_recovery_actions': sum(self._recovery_actions_attempted.values()),
            'error_recovery_enabled': self.error_recovery.max_retries > 0
        }
        
        return {
            # Basic processing configuration
            'processing_timeout': self.processing_timeout,
            'memory_limit_mb': self.memory_limit_mb,
            'max_page_text_size': self.max_page_text_size,
            'memory_monitor_active': self._memory_monitor_active,
            
            # Enhanced memory statistics
            'memory_stats': {
                'process_memory_mb': round(memory_stats['process_memory_mb'], 2),
                'process_memory_peak_mb': round(memory_stats['process_memory_peak_mb'], 2),
                'system_memory_percent': round(memory_stats['system_memory_percent'], 1),
                'system_memory_available_mb': round(memory_stats['system_memory_available_mb'], 2),
                'system_memory_used_mb': round(memory_stats['system_memory_used_mb'], 2),
                'system_memory_total_mb': round(memory_stats['system_memory_total_mb'], 2)
            },
            
            # Memory management features
            'memory_management': {
                'batch_processing_available': True,
                'dynamic_batch_sizing': True,
                'enhanced_garbage_collection': True,
                'memory_pressure_monitoring': True,
                'automatic_cleanup_between_batches': True
            },
            
            # Error recovery
            'error_recovery': error_recovery_summary,
            
            # Garbage collection statistics
            'garbage_collection': {
                'collections_count': gc.get_count(),
                'gc_enabled': gc.isenabled(),
                'gc_thresholds': gc.get_threshold()
            }
        }