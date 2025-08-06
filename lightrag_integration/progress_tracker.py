"""
Progress tracking system for Clinical Metabolomics Oracle LightRAG integration.

This module provides comprehensive progress tracking capabilities for PDF
processing operations, including detailed logging, performance monitoring,
and error tracking with thread-safe operations.

Classes:
    - PDFProcessingProgressTracker: Main progress tracking class
    - FileProcessingInfo: Information about individual file processing
"""

import json
import time
import psutil
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from contextlib import contextmanager

from .progress_config import ProgressTrackingConfig, ProcessingMetrics, FileProcessingStatus


class FileProcessingInfo:
    """
    Information about individual file processing for detailed tracking.
    
    This class maintains detailed information about the processing of
    individual PDF files, including timing, status, and error details.
    
    Attributes:
        filename: Name of the PDF file
        file_path: Full path to the PDF file
        file_size: Size of the file in bytes
        status: Current processing status
        start_time: When processing started
        end_time: When processing completed (None if still processing)
        characters_extracted: Number of characters extracted
        pages_processed: Number of pages processed
        error_message: Error message if processing failed
        error_type: Type of error that occurred
        memory_usage_mb: Memory usage during processing (MB)
        processing_time: Time taken to process the file (seconds)
    """
    
    def __init__(self, filename: str, file_path: Union[str, Path], file_size: int = 0):
        """
        Initialize file processing information.
        
        Args:
            filename: Name of the PDF file
            file_path: Full path to the PDF file
            file_size: Size of the file in bytes
        """
        self.filename = filename
        self.file_path = str(file_path)
        self.file_size = file_size
        self.status = FileProcessingStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.characters_extracted = 0
        self.pages_processed = 0
        self.error_message: Optional[str] = None
        self.error_type: Optional[str] = None
        self.memory_usage_mb = 0.0
        self.processing_time = 0.0
    
    def start_processing(self) -> None:
        """Mark file as processing and record start time."""
        self.status = FileProcessingStatus.PROCESSING
        self.start_time = datetime.now()
    
    def complete_processing(self, characters_extracted: int, pages_processed: int) -> None:
        """
        Mark file as completed and record results.
        
        Args:
            characters_extracted: Number of characters extracted
            pages_processed: Number of pages processed
        """
        self.status = FileProcessingStatus.COMPLETED
        self.end_time = datetime.now()
        self.characters_extracted = characters_extracted
        self.pages_processed = pages_processed
        
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def fail_processing(self, error_message: str, error_type: str = "unknown") -> None:
        """
        Mark file as failed and record error details.
        
        Args:
            error_message: Error message describing the failure
            error_type: Type/category of the error
        """
        self.status = FileProcessingStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
        self.error_type = error_type
        
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def skip_processing(self, reason: str) -> None:
        """
        Mark file as skipped.
        
        Args:
            reason: Reason for skipping the file
        """
        self.status = FileProcessingStatus.SKIPPED
        self.end_time = datetime.now()
        self.error_message = reason
        self.error_type = "skipped"
        
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def update_memory_usage(self) -> None:
        """Update current memory usage."""
        try:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.memory_usage_mb = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'characters_extracted': self.characters_extracted,
            'pages_processed': self.pages_processed,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'memory_usage_mb': self.memory_usage_mb,
            'processing_time': self.processing_time
        }


class PDFProcessingProgressTracker:
    """
    Comprehensive progress tracking system for PDF batch processing operations.
    
    This class provides thread-safe progress tracking, detailed logging,
    performance monitoring, and error tracking for PDF processing workflows.
    It integrates seamlessly with the existing LightRAG infrastructure.
    
    Features:
        - Thread-safe progress tracking
        - Detailed logging with configurable levels
        - Performance monitoring and memory tracking
        - Error categorization and reporting
        - Progress persistence to file
        - Real-time statistics calculation
        - Integration with existing logging systems
    """
    
    def __init__(self, 
                 config: Optional[ProgressTrackingConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the progress tracker.
        
        Args:
            config: Progress tracking configuration (creates default if None)
            logger: Logger instance (creates default if None)
        """
        self.config = config or ProgressTrackingConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread-safe data structures
        self._lock = threading.RLock()
        self.metrics = ProcessingMetrics()
        self.file_info: Dict[str, FileProcessingInfo] = {}
        self.current_file: Optional[str] = None
        
        # Performance monitoring
        self._initial_memory_mb = 0.0
        self._peak_memory_mb = 0.0
        
        # Ensure progress file directory exists if needed
        if self.config.save_progress_to_file:
            self.config.ensure_progress_file_directory()
    
    def start_batch_processing(self, total_files: int, file_list: Optional[List[Path]] = None) -> None:
        """
        Start batch processing and initialize tracking.
        
        Args:
            total_files: Total number of files to process
            file_list: Optional list of file paths for detailed tracking
        """
        with self._lock:
            self.metrics = ProcessingMetrics()
            self.metrics.total_files = total_files
            self.file_info.clear()
            
            # Initialize file info if file list provided
            if file_list:
                for file_path in file_list:
                    try:
                        file_size = file_path.stat().st_size if file_path.exists() else 0
                    except (OSError, PermissionError):
                        file_size = 0
                    
                    self.file_info[str(file_path)] = FileProcessingInfo(
                        filename=file_path.name,
                        file_path=file_path,
                        file_size=file_size
                    )
            
            # Record initial memory usage
            try:
                process = psutil.Process()
                self._initial_memory_mb = process.memory_info().rss / 1024 / 1024
                self._peak_memory_mb = self._initial_memory_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._initial_memory_mb = 0.0
                self._peak_memory_mb = 0.0
            
            # Log batch start
            if self.config.enable_progress_tracking:
                self.logger.log(
                    self.config.get_log_level_value(self.config.progress_log_level),
                    f"Starting batch processing: {total_files} files"
                )
                
                if self.config.log_processing_stats:
                    self.logger.log(
                        self.config.get_log_level_value(self.config.stats_log_level),
                        f"Initial memory usage: {self._initial_memory_mb:.2f} MB"
                    )
    
    @contextmanager
    def track_file_processing(self, file_path: Union[str, Path], file_index: int):
        """
        Context manager for tracking individual file processing.
        
        Args:
            file_path: Path to the file being processed
            file_index: Index of the file in the batch (0-based)
            
        Yields:
            FileProcessingInfo: Information object for the file
        """
        file_path_str = str(file_path)
        
        with self._lock:
            # Get or create file info
            if file_path_str not in self.file_info:
                file_path_obj = Path(file_path)
                try:
                    file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
                except (OSError, PermissionError):
                    file_size = 0
                
                self.file_info[file_path_str] = FileProcessingInfo(
                    filename=file_path_obj.name,
                    file_path=file_path,
                    file_size=file_size
                )
            
            file_info = self.file_info[file_path_str]
            self.current_file = file_path_str
            
            # Start processing
            file_info.start_processing()
            
            # Log progress if needed
            if self.config.should_log_progress(file_index):
                self.logger.log(
                    self.config.get_log_level_value(self.config.progress_log_level),
                    f"Processing file {file_index + 1}/{self.metrics.total_files}: {file_info.filename}"
                )
            
            # Check memory if needed
            if self.config.should_check_memory(file_index):
                file_info.update_memory_usage()
                self._update_peak_memory(file_info.memory_usage_mb)
                
                if self.config.log_processing_stats:
                    self.logger.log(
                        self.config.get_log_level_value(self.config.stats_log_level),
                        f"Memory usage at file {file_index + 1}: {file_info.memory_usage_mb:.2f} MB"
                    )
        
        try:
            yield file_info
        except Exception as e:
            # Handle processing error
            error_type = type(e).__name__
            error_message = str(e)
            
            if self.config.log_detailed_errors:
                error_message = self.config.truncate_error_details(error_message)
            
            with self._lock:
                file_info.fail_processing(error_message, error_type)
                self.metrics.failed_files += 1
                self.metrics.add_error(error_type)
                
                # Log error
                self.logger.log(
                    self.config.get_log_level_value(self.config.error_log_level),
                    f"Failed to process {file_info.filename}: {error_message}"
                )
            
            raise
        else:
            # Processing completed successfully (caller should call record_success)
            pass
        finally:
            with self._lock:
                self.current_file = None
                
                # Save progress if enabled
                if self.config.save_progress_to_file:
                    self._save_progress_to_file()
    
    def record_file_success(self, file_path: Union[str, Path], 
                           characters_extracted: int, pages_processed: int) -> None:
        """
        Record successful processing of a file.
        
        Args:
            file_path: Path to the successfully processed file
            characters_extracted: Number of characters extracted
            pages_processed: Number of pages processed
        """
        file_path_str = str(file_path)
        
        with self._lock:
            if file_path_str in self.file_info:
                file_info = self.file_info[file_path_str]
                file_info.complete_processing(characters_extracted, pages_processed)
                
                # Update metrics
                self.metrics.completed_files += 1
                self.metrics.total_characters += characters_extracted
                self.metrics.total_pages += pages_processed
                self.metrics.update_processing_time()
                
                # Log file details if enabled
                if self.config.log_file_details:
                    self.logger.log(
                        self.config.get_log_level_value(self.config.progress_log_level),
                        f"Successfully processed {file_info.filename}: "
                        f"{characters_extracted} chars, {pages_processed} pages, "
                        f"{file_info.processing_time:.2f}s"
                    )
    
    def record_file_skip(self, file_path: Union[str, Path], reason: str) -> None:
        """
        Record that a file was skipped.
        
        Args:
            file_path: Path to the skipped file
            reason: Reason for skipping the file
        """
        file_path_str = str(file_path)
        
        with self._lock:
            if file_path_str not in self.file_info:
                file_path_obj = Path(file_path)
                self.file_info[file_path_str] = FileProcessingInfo(
                    filename=file_path_obj.name,
                    file_path=file_path,
                    file_size=0
                )
            
            file_info = self.file_info[file_path_str]
            file_info.skip_processing(reason)
            self.metrics.skipped_files += 1
            
            # Log skip if detailed errors enabled
            if self.config.log_detailed_errors:
                self.logger.log(
                    self.config.get_log_level_value(self.config.error_log_level),
                    f"Skipped {file_info.filename}: {reason}"
                )
    
    def finish_batch_processing(self) -> ProcessingMetrics:
        """
        Complete batch processing and return final metrics.
        
        Returns:
            ProcessingMetrics: Final processing metrics
        """
        with self._lock:
            self.metrics.end_time = datetime.now()
            self.metrics.update_processing_time()
            
            # Log final statistics
            if self.config.log_processing_stats:
                self._log_final_statistics()
            
            # Save final progress if enabled
            if self.config.save_progress_to_file:
                self._save_progress_to_file()
            
            return self.metrics
    
    def get_current_metrics(self) -> ProcessingMetrics:
        """
        Get current processing metrics.
        
        Returns:
            ProcessingMetrics: Current metrics (copy)
        """
        with self._lock:
            # Update processing time for current metrics
            current_metrics = ProcessingMetrics(
                total_files=self.metrics.total_files,
                completed_files=self.metrics.completed_files,
                failed_files=self.metrics.failed_files,
                skipped_files=self.metrics.skipped_files,
                total_characters=self.metrics.total_characters,
                total_pages=self.metrics.total_pages,
                start_time=self.metrics.start_time,
                end_time=self.metrics.end_time,
                average_processing_time=self.metrics.average_processing_time,
                errors_by_type=self.metrics.errors_by_type.copy()
            )
            current_metrics.update_processing_time()
            return current_metrics
    
    def get_file_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all files processed.
        
        Returns:
            List[Dict[str, Any]]: List of file processing details
        """
        with self._lock:
            return [file_info.to_dict() for file_info in self.file_info.values()]
    
    def _update_peak_memory(self, current_memory_mb: float) -> None:
        """Update peak memory usage tracking."""
        if current_memory_mb > self._peak_memory_mb:
            self._peak_memory_mb = current_memory_mb
    
    def _log_final_statistics(self) -> None:
        """Log comprehensive final statistics."""
        metrics = self.metrics
        
        # Basic statistics
        self.logger.log(
            self.config.get_log_level_value(self.config.stats_log_level),
            f"Batch processing completed: {metrics.completed_files} successful, "
            f"{metrics.failed_files} failed, {metrics.skipped_files} skipped "
            f"out of {metrics.total_files} total files"
        )
        
        # Performance statistics
        if metrics.completed_files > 0:
            self.logger.log(
                self.config.get_log_level_value(self.config.stats_log_level),
                f"Processing statistics: {metrics.success_rate:.1f}% success rate, "
                f"Total characters: {metrics.total_characters:,}, "
                f"Total pages: {metrics.total_pages:,}, "
                f"Average time per file: {metrics.average_processing_time:.2f}s"
            )
        
        # Memory statistics
        if self.config.enable_memory_monitoring:
            memory_increase = self._peak_memory_mb - self._initial_memory_mb
            self.logger.log(
                self.config.get_log_level_value(self.config.stats_log_level),
                f"Memory statistics: Initial: {self._initial_memory_mb:.2f} MB, "
                f"Peak: {self._peak_memory_mb:.2f} MB, "
                f"Increase: {memory_increase:.2f} MB"
            )
        
        # Error statistics
        if metrics.errors_by_type:
            error_summary = ", ".join([f"{error_type}: {count}" for error_type, count in metrics.errors_by_type.items()])
            self.logger.log(
                self.config.get_log_level_value(self.config.error_log_level),
                f"Error summary: {error_summary}"
            )
    
    def _save_progress_to_file(self) -> None:
        """Save current progress to file."""
        if not self.config.progress_file_path:
            return
        
        try:
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics.to_dict(),
                'file_details': self.get_file_details(),
                'memory_stats': {
                    'initial_memory_mb': self._initial_memory_mb,
                    'peak_memory_mb': self._peak_memory_mb,
                    'memory_increase_mb': self._peak_memory_mb - self._initial_memory_mb
                },
                'config': self.config.to_dict()
            }
            
            with open(self.config.progress_file_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
        except (OSError, IOError, json.JSONEncodeError) as e:
            # Log error but don't fail processing
            self.logger.warning(f"Failed to save progress to file: {e}")
    
    def get_progress_summary(self) -> str:
        """
        Get a human-readable progress summary.
        
        Returns:
            str: Formatted progress summary
        """
        with self._lock:
            metrics = self.get_current_metrics()
            processed = metrics.completed_files + metrics.failed_files + metrics.skipped_files
            
            summary = f"Progress: {processed}/{metrics.total_files} files processed"
            
            if metrics.total_files > 0:
                progress_percent = (processed / metrics.total_files) * 100
                summary += f" ({progress_percent:.1f}%)"
            
            if metrics.completed_files > 0:
                summary += f", Success rate: {metrics.success_rate:.1f}%"
            
            if metrics.processing_time > 0:
                summary += f", Time elapsed: {metrics.processing_time:.1f}s"
            
            if self.current_file:
                current_filename = Path(self.current_file).name
                summary += f", Currently processing: {current_filename}"
            
            return summary