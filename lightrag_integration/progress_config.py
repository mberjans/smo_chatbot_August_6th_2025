"""
Progress tracking configuration for Clinical Metabolomics Oracle LightRAG integration.

This module provides configuration classes for managing progress tracking
during PDF processing operations. It integrates with the existing LightRAGConfig
system to provide comprehensive logging and monitoring capabilities.

Classes:
    - ProgressTrackingConfig: Configuration for progress tracking and logging
    - ProcessingMetrics: Data class for tracking processing metrics
    - FileProcessingStatus: Enum for file processing states
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime


class FileProcessingStatus(Enum):
    """Enumeration for file processing status tracking."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ProcessingMetrics:
    """
    Data class for tracking PDF processing metrics.
    
    This class maintains counters and timing information for batch
    processing operations, providing detailed insights into processing
    performance and success rates.
    
    Attributes:
        total_files: Total number of files to process
        completed_files: Number of successfully processed files
        failed_files: Number of files that failed processing
        skipped_files: Number of files that were skipped
        total_characters: Total characters extracted from all PDFs
        total_pages: Total pages processed across all PDFs
        start_time: Processing start timestamp
        end_time: Processing end timestamp (None if still running)
        average_processing_time: Average time per file in seconds
        errors_by_type: Dictionary mapping error types to counts
    """
    
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_characters: int = 0
    total_pages: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    average_processing_time: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize start_time if not provided."""
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def processing_time(self) -> float:
        """
        Get total processing time in seconds.
        
        Returns:
            float: Total processing time, or time elapsed if still running
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """
        Get success rate as a percentage.
        
        Returns:
            float: Success rate (0.0 to 100.0)
        """
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """
        Get failure rate as a percentage.
        
        Returns:
            float: Failure rate (0.0 to 100.0)
        """
        if self.total_files == 0:
            return 0.0
        return (self.failed_files / self.total_files) * 100.0
    
    def add_error(self, error_type: str) -> None:
        """
        Add an error to the error tracking.
        
        Args:
            error_type: Type/category of the error
        """
        if error_type in self.errors_by_type:
            self.errors_by_type[error_type] += 1
        else:
            self.errors_by_type[error_type] = 1
    
    def update_processing_time(self) -> None:
        """Update average processing time based on completed files."""
        if self.completed_files > 0 and self.processing_time > 0:
            self.average_processing_time = self.processing_time / self.completed_files
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of metrics
        """
        return {
            'total_files': self.total_files,
            'completed_files': self.completed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'total_characters': self.total_characters,
            'total_pages': self.total_pages,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'processing_time': self.processing_time,
            'average_processing_time': self.average_processing_time,
            'success_rate': self.success_rate,
            'failure_rate': self.failure_rate,
            'errors_by_type': self.errors_by_type.copy()
        }


@dataclass
class ProgressTrackingConfig:
    """
    Configuration for PDF processing progress tracking and logging.
    
    This class extends the base LightRAG configuration with specific
    settings for progress tracking, detailed logging, and performance
    monitoring during batch PDF processing operations.
    
    Attributes:
        enable_progress_tracking: Whether to enable detailed progress tracking
        log_progress_interval: Interval for logging progress (number of files)
        log_detailed_errors: Whether to log detailed error information
        log_processing_stats: Whether to log processing statistics
        log_file_details: Whether to log individual file processing details
        progress_log_level: Log level for progress messages (default: INFO)
        error_log_level: Log level for error messages (default: ERROR)
        stats_log_level: Log level for statistics messages (default: INFO)
        max_error_details_length: Maximum length of error details in logs
        enable_memory_monitoring: Whether to monitor memory usage during processing
        memory_check_interval: Interval for memory checks (number of files)
        enable_timing_details: Whether to track detailed timing information
        save_progress_to_file: Whether to save progress to a file
        progress_file_path: Path to save progress information (if enabled)
    """
    
    enable_progress_tracking: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_PROGRESS_TRACKING", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    log_progress_interval: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_LOG_PROGRESS_INTERVAL", "5"))
    )
    log_detailed_errors: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_LOG_DETAILED_ERRORS", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    log_processing_stats: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_LOG_PROCESSING_STATS", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    log_file_details: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_LOG_FILE_DETAILS", "false").lower() in ("true", "1", "yes", "t", "on")
    )
    
    # Log levels for different types of messages
    progress_log_level: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_PROGRESS_LOG_LEVEL", "INFO")
    )
    error_log_level: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ERROR_LOG_LEVEL", "ERROR")
    )
    stats_log_level: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_STATS_LOG_LEVEL", "INFO")
    )
    
    # Error handling configuration
    max_error_details_length: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ERROR_DETAILS_LENGTH", "500"))
    )
    
    # Performance monitoring
    enable_memory_monitoring: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_MEMORY_MONITORING", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    memory_check_interval: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_MEMORY_CHECK_INTERVAL", "10"))
    )
    enable_timing_details: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_TIMING_DETAILS", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    
    # Progress persistence
    save_progress_to_file: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_SAVE_PROGRESS_TO_FILE", "false").lower() in ("true", "1", "yes", "t", "on")
    )
    progress_file_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("LIGHTRAG_PROGRESS_FILE_PATH", "logs/processing_progress.json")) if os.getenv("LIGHTRAG_SAVE_PROGRESS_TO_FILE", "false").lower() in ("true", "1", "yes", "t", "on") else None
    )
    
    # Knowledge base progress tracking extensions
    enable_unified_progress_tracking: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_UNIFIED_PROGRESS", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    enable_phase_based_progress: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_PHASE_PROGRESS", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    phase_progress_update_interval: float = field(
        default_factory=lambda: float(os.getenv("LIGHTRAG_PHASE_UPDATE_INTERVAL", "2.0"))
    )
    enable_progress_callbacks: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_PROGRESS_CALLBACKS", "false").lower() in ("true", "1", "yes", "t", "on")
    )
    save_unified_progress_to_file: bool = field(
        default_factory=lambda: os.getenv("LIGHTRAG_SAVE_UNIFIED_PROGRESS", "true").lower() in ("true", "1", "yes", "t", "on")
    )
    unified_progress_file_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("LIGHTRAG_UNIFIED_PROGRESS_FILE_PATH", "logs/knowledge_base_progress.json")) if os.getenv("LIGHTRAG_SAVE_UNIFIED_PROGRESS", "true").lower() in ("true", "1", "yes", "t", "on") else None
    )
    
    def __post_init__(self):
        """Post-initialization processing to handle log level validation and path conversion."""
        # Normalize and validate log levels
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        
        if self.progress_log_level.upper() not in valid_levels:
            self.progress_log_level = "INFO"
        else:
            self.progress_log_level = self.progress_log_level.upper()
        
        if self.error_log_level.upper() not in valid_levels:
            self.error_log_level = "ERROR"
        else:
            self.error_log_level = self.error_log_level.upper()
        
        if self.stats_log_level.upper() not in valid_levels:
            self.stats_log_level = "INFO"
        else:
            self.stats_log_level = self.stats_log_level.upper()
        
        # Convert progress_file_path to Path object if it's a string
        if isinstance(self.progress_file_path, str):
            self.progress_file_path = Path(self.progress_file_path)
        
        # Convert unified_progress_file_path to Path object if it's a string
        if isinstance(self.unified_progress_file_path, str):
            self.unified_progress_file_path = Path(self.unified_progress_file_path)
        
        # Validate numerical parameters
        if self.log_progress_interval <= 0:
            self.log_progress_interval = 5
        
        if self.memory_check_interval <= 0:
            self.memory_check_interval = 10
        
        if self.max_error_details_length <= 0:
            self.max_error_details_length = 500
        
        if self.phase_progress_update_interval <= 0:
            self.phase_progress_update_interval = 2.0
    
    def get_log_level_value(self, level_name: str) -> int:
        """
        Get numeric log level value for a log level name.
        
        Args:
            level_name: Name of the log level (e.g., 'INFO', 'DEBUG')
            
        Returns:
            int: Numeric log level value
        """
        return getattr(logging, level_name.upper(), logging.INFO)
    
    def should_log_progress(self, file_index: int) -> bool:
        """
        Determine if progress should be logged for the given file index.
        
        Args:
            file_index: Current file index (0-based)
            
        Returns:
            bool: True if progress should be logged
        """
        if not self.enable_progress_tracking:
            return False
        
        # Always log for the first file and at regular intervals
        return file_index == 0 or (file_index + 1) % self.log_progress_interval == 0
    
    def should_check_memory(self, file_index: int) -> bool:
        """
        Determine if memory should be checked for the given file index.
        
        Args:
            file_index: Current file index (0-based)
            
        Returns:
            bool: True if memory should be checked
        """
        if not self.enable_memory_monitoring:
            return False
        
        # Always check for the first file and at regular intervals
        return file_index == 0 or (file_index + 1) % self.memory_check_interval == 0
    
    def truncate_error_details(self, error_message: str) -> str:
        """
        Truncate error details to maximum allowed length.
        
        Args:
            error_message: Original error message
            
        Returns:
            str: Truncated error message if necessary
        """
        if len(error_message) <= self.max_error_details_length:
            return error_message
        
        return error_message[:self.max_error_details_length - 10] + "...[truncated]"
    
    def ensure_progress_file_directory(self) -> None:
        """Ensure the progress file directory exists if progress saving is enabled."""
        if self.save_progress_to_file and self.progress_file_path:
            self.progress_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.save_unified_progress_to_file and self.unified_progress_file_path:
            self.unified_progress_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        return {
            'enable_progress_tracking': self.enable_progress_tracking,
            'log_progress_interval': self.log_progress_interval,
            'log_detailed_errors': self.log_detailed_errors,
            'log_processing_stats': self.log_processing_stats,
            'log_file_details': self.log_file_details,
            'progress_log_level': self.progress_log_level,
            'error_log_level': self.error_log_level,
            'stats_log_level': self.stats_log_level,
            'max_error_details_length': self.max_error_details_length,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'memory_check_interval': self.memory_check_interval,
            'enable_timing_details': self.enable_timing_details,
            'save_progress_to_file': self.save_progress_to_file,
            'progress_file_path': str(self.progress_file_path) if self.progress_file_path else None,
            'enable_unified_progress_tracking': self.enable_unified_progress_tracking,
            'enable_phase_based_progress': self.enable_phase_based_progress,
            'phase_progress_update_interval': self.phase_progress_update_interval,
            'enable_progress_callbacks': self.enable_progress_callbacks,
            'save_unified_progress_to_file': self.save_unified_progress_to_file,
            'unified_progress_file_path': str(self.unified_progress_file_path) if self.unified_progress_file_path else None
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProgressTrackingConfig':
        """
        Create ProgressTrackingConfig instance from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            ProgressTrackingConfig: Configuration instance
        """
        config_dict = config_dict.copy()  # Don't modify original
        
        # Convert progress_file_path to Path object if present
        if 'progress_file_path' in config_dict and config_dict['progress_file_path']:
            config_dict['progress_file_path'] = Path(config_dict['progress_file_path'])
        
        # Convert unified_progress_file_path to Path object if present
        if 'unified_progress_file_path' in config_dict and config_dict['unified_progress_file_path']:
            config_dict['unified_progress_file_path'] = Path(config_dict['unified_progress_file_path'])
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ProgressTrackingConfig("
            f"enable_progress_tracking={self.enable_progress_tracking}, "
            f"log_progress_interval={self.log_progress_interval}, "
            f"log_detailed_errors={self.log_detailed_errors}, "
            f"log_processing_stats={self.log_processing_stats}, "
            f"enable_memory_monitoring={self.enable_memory_monitoring})"
        )