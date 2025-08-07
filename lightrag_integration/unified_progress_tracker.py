"""
Unified Progress Tracking System for Clinical Metabolomics Oracle Knowledge Base Construction.

This module provides comprehensive progress tracking capabilities across all phases
of knowledge base initialization, integrating with existing progress tracking infrastructure
while adding phase-based progress calculation and unified reporting.

Classes:
    - KnowledgeBaseProgressTracker: Main unified progress tracking class
    - PhaseProgressInfo: Information about individual phase progress
    - UnifiedProgressCallback: Callback interface for progress updates
    - PhaseWeights: Configuration for phase-based progress weighting
"""

import json
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum

from .progress_config import ProgressTrackingConfig, ProcessingMetrics
from .progress_tracker import PDFProcessingProgressTracker


class KnowledgeBasePhase(Enum):
    """Enumeration for knowledge base initialization phases."""
    STORAGE_INIT = "storage_initialization"
    PDF_PROCESSING = "pdf_processing" 
    DOCUMENT_INGESTION = "document_ingestion"
    FINALIZATION = "finalization"


@dataclass
class PhaseWeights:
    """
    Configuration for phase-based progress weighting.
    
    Attributes:
        storage_init: Weight for storage initialization phase (default: 10%)
        pdf_processing: Weight for PDF processing phase (default: 60%)
        document_ingestion: Weight for document ingestion phase (default: 25%)
        finalization: Weight for finalization phase (default: 5%)
    """
    storage_init: float = 0.10
    pdf_processing: float = 0.60
    document_ingestion: float = 0.25
    finalization: float = 0.05
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.storage_init + self.pdf_processing + self.document_ingestion + self.finalization
        if abs(total - 1.0) > 0.001:  # Allow small floating point tolerance
            raise ValueError(f"Phase weights must sum to 1.0, got {total}")
    
    def get_weight(self, phase: KnowledgeBasePhase) -> float:
        """Get weight for a specific phase."""
        weight_map = {
            KnowledgeBasePhase.STORAGE_INIT: self.storage_init,
            KnowledgeBasePhase.PDF_PROCESSING: self.pdf_processing,
            KnowledgeBasePhase.DOCUMENT_INGESTION: self.document_ingestion,
            KnowledgeBasePhase.FINALIZATION: self.finalization
        }
        return weight_map[phase]


@dataclass
class PhaseProgressInfo:
    """
    Information about individual phase progress.
    
    Attributes:
        phase: The phase being tracked
        start_time: When the phase started
        end_time: When the phase completed (None if still running)
        current_progress: Current progress within the phase (0.0 to 1.0)
        status_message: Current status message for the phase
        details: Additional phase-specific details
        error_message: Error message if phase failed
        estimated_duration: Estimated duration for the phase in seconds
    """
    phase: KnowledgeBasePhase
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_progress: float = 0.0
    status_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    estimated_duration: Optional[float] = None
    
    @property
    def is_active(self) -> bool:
        """Check if phase is currently active."""
        return self.start_time is not None and self.end_time is None
    
    @property
    def is_completed(self) -> bool:
        """Check if phase is completed."""
        return self.end_time is not None and self.error_message is None
    
    @property
    def is_failed(self) -> bool:
        """Check if phase failed."""
        return self.error_message is not None
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time for the phase in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def start_phase(self, status_message: str = "", estimated_duration: Optional[float] = None):
        """Start the phase."""
        self.start_time = datetime.now()
        self.status_message = status_message
        self.estimated_duration = estimated_duration
        self.current_progress = 0.0
        self.error_message = None
    
    def update_progress(self, progress: float, status_message: str = "", details: Optional[Dict[str, Any]] = None):
        """Update phase progress."""
        self.current_progress = max(0.0, min(1.0, progress))
        if status_message:
            self.status_message = status_message
        if details:
            self.details.update(details)
    
    def complete_phase(self, status_message: str = ""):
        """Complete the phase successfully."""
        self.end_time = datetime.now()
        self.current_progress = 1.0
        if status_message:
            self.status_message = status_message
    
    def fail_phase(self, error_message: str):
        """Mark phase as failed."""
        self.end_time = datetime.now()
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'phase': self.phase.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'current_progress': self.current_progress,
            'status_message': self.status_message,
            'details': self.details.copy(),
            'error_message': self.error_message,
            'estimated_duration': self.estimated_duration,
            'elapsed_time': self.elapsed_time,
            'is_active': self.is_active,
            'is_completed': self.is_completed,
            'is_failed': self.is_failed
        }


class UnifiedProgressCallback(Protocol):
    """
    Protocol defining the unified progress callback interface.
    
    This callback is invoked whenever progress is updated across any phase
    of knowledge base initialization, providing a unified view of overall progress.
    """
    
    def __call__(self, 
                 overall_progress: float,
                 current_phase: KnowledgeBasePhase,
                 phase_progress: float,
                 status_message: str,
                 phase_details: Dict[str, Any],
                 all_phases: Dict[KnowledgeBasePhase, PhaseProgressInfo]) -> None:
        """
        Called when progress is updated.
        
        Args:
            overall_progress: Overall progress across all phases (0.0 to 1.0)
            current_phase: Currently active phase
            phase_progress: Progress within the current phase (0.0 to 1.0)
            status_message: Current status message
            phase_details: Details specific to current phase
            all_phases: Complete phase information for all phases
        """
        ...


@dataclass
class UnifiedProgressState:
    """
    Unified state tracking for knowledge base construction progress.
    
    Attributes:
        overall_progress: Overall progress across all phases (0.0 to 1.0)
        current_phase: Currently active phase
        phase_info: Progress information for each phase
        phase_weights: Weighting configuration for phases
        start_time: Overall initialization start time
        estimated_completion_time: Estimated completion time
        total_documents: Total number of documents to process
        processed_documents: Number of documents processed so far
        failed_documents: Number of documents that failed processing
        errors: List of errors encountered
    """
    overall_progress: float = 0.0
    current_phase: Optional[KnowledgeBasePhase] = None
    phase_info: Dict[KnowledgeBasePhase, PhaseProgressInfo] = field(default_factory=dict)
    phase_weights: PhaseWeights = field(default_factory=PhaseWeights)
    start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize phase info for all phases."""
        if not self.phase_info:
            for phase in KnowledgeBasePhase:
                self.phase_info[phase] = PhaseProgressInfo(phase=phase)
    
    def calculate_overall_progress(self) -> float:
        """Calculate overall progress based on phase weights and individual progress."""
        total_progress = 0.0
        for phase, phase_info in self.phase_info.items():
            weight = self.phase_weights.get_weight(phase)
            phase_progress = phase_info.current_progress
            total_progress += weight * phase_progress
        return total_progress
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time based on current progress and elapsed time."""
        if not self.start_time or self.overall_progress <= 0.0:
            return None
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if self.overall_progress >= 1.0:
            return 0.0
        
        estimated_total_time = elapsed_time / self.overall_progress
        return max(0.0, estimated_total_time - elapsed_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_progress': self.overall_progress,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'phase_info': {phase.value: info.to_dict() for phase, info in self.phase_info.items()},
            'phase_weights': {
                'storage_init': self.phase_weights.storage_init,
                'pdf_processing': self.phase_weights.pdf_processing,
                'document_ingestion': self.phase_weights.document_ingestion,
                'finalization': self.phase_weights.finalization
            },
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'estimated_time_remaining': self.get_estimated_time_remaining(),
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'failed_documents': self.failed_documents,
            'errors': self.errors.copy(),
            'elapsed_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        }


class KnowledgeBaseProgressTracker:
    """
    Unified progress tracking system for knowledge base construction.
    
    This class provides comprehensive progress tracking across all phases of knowledge
    base initialization, integrating with existing PDF progress tracking while adding
    phase-based progress calculation and unified reporting capabilities.
    
    Features:
        - Phase-based progress tracking with configurable weights
        - Integration with existing PDFProcessingProgressTracker
        - Unified callback interface for progress updates
        - Thread-safe progress state management
        - Detailed logging and metrics collection
        - Progress persistence and recovery
        - Estimated time remaining calculation
    """
    
    def __init__(self,
                 progress_config: Optional[ProgressTrackingConfig] = None,
                 phase_weights: Optional[PhaseWeights] = None,
                 progress_callback: Optional[UnifiedProgressCallback] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the unified progress tracker.
        
        Args:
            progress_config: Configuration for progress tracking (creates default if None)
            phase_weights: Phase weight configuration (uses default if None)
            progress_callback: Callback for progress updates
            logger: Logger instance (creates default if None)
        """
        self.progress_config = progress_config or ProgressTrackingConfig()
        self.phase_weights = phase_weights or PhaseWeights()
        self.progress_callback = progress_callback
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread-safe state management
        self._lock = threading.RLock()
        self.state = UnifiedProgressState(phase_weights=self.phase_weights)
        
        # Integration with existing PDF progress tracker
        self.pdf_progress_tracker: Optional[PDFProcessingProgressTracker] = None
        
        # Progress persistence
        self._progress_file_path: Optional[Path] = None
        if self.progress_config.save_unified_progress_to_file:
            if self.progress_config.unified_progress_file_path:
                self._progress_file_path = self.progress_config.unified_progress_file_path
            else:
                # Fallback to default path
                default_path = Path("logs/knowledge_base_progress.json")
                self._progress_file_path = default_path
            self._progress_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_initialization(self, total_documents: int = 0) -> None:
        """
        Start knowledge base initialization tracking.
        
        Args:
            total_documents: Total number of documents expected to process
        """
        with self._lock:
            self.state = UnifiedProgressState(phase_weights=self.phase_weights)
            self.state.start_time = datetime.now()
            self.state.total_documents = total_documents
            
            # Log initialization start
            if self.progress_config.enable_progress_tracking:
                self.logger.log(
                    self.progress_config.get_log_level_value(self.progress_config.progress_log_level),
                    f"Starting knowledge base initialization: {total_documents} documents expected"
                )
            
            self._save_progress()
    
    def start_phase(self, 
                   phase: KnowledgeBasePhase,
                   status_message: str = "",
                   estimated_duration: Optional[float] = None,
                   details: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a specific phase of initialization.
        
        Args:
            phase: The phase being started
            status_message: Status message for the phase
            estimated_duration: Estimated duration in seconds
            details: Additional phase-specific details
        """
        with self._lock:
            self.state.current_phase = phase
            phase_info = self.state.phase_info[phase]
            
            phase_info.start_phase(status_message, estimated_duration)
            if details:
                phase_info.details.update(details)
            
            # Update overall progress
            self.state.overall_progress = self.state.calculate_overall_progress()
            
            # Log phase start
            if self.progress_config.enable_progress_tracking:
                self.logger.log(
                    self.progress_config.get_log_level_value(self.progress_config.progress_log_level),
                    f"Starting phase: {phase.value} - {status_message}"
                )
            
            # Trigger callback
            self._trigger_callback()
            self._save_progress()
    
    def update_phase_progress(self,
                             phase: KnowledgeBasePhase,
                             progress: float,
                             status_message: str = "",
                             details: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress for a specific phase.
        
        Args:
            phase: The phase being updated
            progress: Progress within the phase (0.0 to 1.0)
            status_message: Updated status message
            details: Additional phase-specific details
        """
        with self._lock:
            phase_info = self.state.phase_info[phase]
            phase_info.update_progress(progress, status_message, details)
            
            # Update overall progress
            self.state.overall_progress = self.state.calculate_overall_progress()
            
            # Trigger callback
            self._trigger_callback()
            
            # Save progress periodically
            if self.progress_config.save_unified_progress_to_file:
                self._save_progress()
    
    def complete_phase(self,
                      phase: KnowledgeBasePhase,
                      status_message: str = "") -> None:
        """
        Mark a phase as completed.
        
        Args:
            phase: The phase being completed
            status_message: Final status message for the phase
        """
        with self._lock:
            phase_info = self.state.phase_info[phase]
            phase_info.complete_phase(status_message)
            
            # Update overall progress
            self.state.overall_progress = self.state.calculate_overall_progress()
            
            # Log phase completion
            if self.progress_config.enable_progress_tracking:
                elapsed_time = phase_info.elapsed_time
                self.logger.log(
                    self.progress_config.get_log_level_value(self.progress_config.progress_log_level),
                    f"Completed phase: {phase.value} in {elapsed_time:.2f}s - {status_message}"
                )
            
            # Check if this was the last phase
            if phase == KnowledgeBasePhase.FINALIZATION:
                self._complete_initialization()
            
            # Trigger callback
            self._trigger_callback()
            self._save_progress()
    
    def fail_phase(self,
                  phase: KnowledgeBasePhase,
                  error_message: str) -> None:
        """
        Mark a phase as failed.
        
        Args:
            phase: The phase that failed
            error_message: Error message describing the failure
        """
        with self._lock:
            phase_info = self.state.phase_info[phase]
            phase_info.fail_phase(error_message)
            error_entry = f"{phase.value}: {error_message}"
            self.state.errors.append(error_entry)
            
            # Log phase failure
            self.logger.log(
                self.progress_config.get_log_level_value(self.progress_config.error_log_level),
                f"Phase failed: {phase.value} - {error_message}"
            )
            
            # Trigger callback
            self._trigger_callback()
            self._save_progress()
    
    def integrate_pdf_progress_tracker(self, pdf_tracker: PDFProcessingProgressTracker) -> None:
        """
        Integrate with existing PDF progress tracker.
        
        Args:
            pdf_tracker: The PDF processing progress tracker to integrate with
        """
        self.pdf_progress_tracker = pdf_tracker
        
        # Set up bridge to translate PDF progress to phase progress
        self._setup_pdf_progress_bridge()
    
    def update_document_counts(self,
                              processed: int = 0,
                              failed: int = 0,
                              total: Optional[int] = None) -> None:
        """
        Update document processing counts.
        
        Args:
            processed: Number of documents processed
            failed: Number of documents that failed
            total: Total number of documents (updates existing total if provided)
        """
        with self._lock:
            if processed > 0:
                self.state.processed_documents += processed
            if failed > 0:
                self.state.failed_documents += failed
            if total is not None:
                self.state.total_documents = total
    
    def get_current_state(self) -> UnifiedProgressState:
        """
        Get current progress state.
        
        Returns:
            Current unified progress state (copy)
        """
        with self._lock:
            # Create a deep copy by recreating the state
            new_state = UnifiedProgressState(phase_weights=self.phase_weights)
            
            # Copy basic attributes
            new_state.overall_progress = self.state.overall_progress
            new_state.current_phase = self.state.current_phase
            new_state.start_time = self.state.start_time
            new_state.estimated_completion_time = self.state.estimated_completion_time
            new_state.total_documents = self.state.total_documents
            new_state.processed_documents = self.state.processed_documents
            new_state.failed_documents = self.state.failed_documents
            new_state.errors = self.state.errors.copy()
            
            # Deep copy phase info
            for phase, phase_info in self.state.phase_info.items():
                new_phase_info = PhaseProgressInfo(phase=phase)
                new_phase_info.start_time = phase_info.start_time
                new_phase_info.end_time = phase_info.end_time
                new_phase_info.current_progress = phase_info.current_progress
                new_phase_info.status_message = phase_info.status_message
                new_phase_info.details = phase_info.details.copy()
                new_phase_info.error_message = phase_info.error_message
                new_phase_info.estimated_duration = phase_info.estimated_duration
                new_state.phase_info[phase] = new_phase_info
            
            return new_state
    
    def get_progress_summary(self) -> str:
        """
        Get human-readable progress summary.
        
        Returns:
            Formatted progress summary string
        """
        with self._lock:
            current_state = self.get_current_state()
            
            summary_parts = []
            
            # Overall progress
            summary_parts.append(f"Overall Progress: {current_state.overall_progress:.1%}")
            
            # Current phase
            if current_state.current_phase:
                phase_info = current_state.phase_info[current_state.current_phase]
                summary_parts.append(f"Current Phase: {current_state.current_phase.value} ({phase_info.current_progress:.1%})")
                if phase_info.status_message:
                    summary_parts.append(f"Status: {phase_info.status_message}")
            
            # Document counts
            if current_state.total_documents > 0:
                summary_parts.append(f"Documents: {current_state.processed_documents}/{current_state.total_documents} processed")
                if current_state.failed_documents > 0:
                    summary_parts.append(f"{current_state.failed_documents} failed")
            
            # Time information
            if current_state.start_time:
                elapsed = (datetime.now() - current_state.start_time).total_seconds()
                summary_parts.append(f"Elapsed: {elapsed:.1f}s")
                
                remaining = current_state.get_estimated_time_remaining()
                if remaining:
                    summary_parts.append(f"Remaining: {remaining:.1f}s")
            
            # Errors
            if current_state.errors:
                summary_parts.append(f"Errors: {len(current_state.errors)}")
            
            return " | ".join(summary_parts)
    
    def _setup_pdf_progress_bridge(self) -> None:
        """Set up bridge to translate PDF progress to unified progress."""
        if not self.pdf_progress_tracker:
            return
        
        # This would be called periodically to sync PDF progress
        def sync_pdf_progress():
            if not self.pdf_progress_tracker:
                return
                
            metrics = self.pdf_progress_tracker.get_current_metrics()
            
            # Calculate PDF processing progress
            if metrics.total_files > 0:
                processed_files = metrics.completed_files + metrics.failed_files + metrics.skipped_files
                pdf_progress = processed_files / metrics.total_files
                
                # Update PDF processing phase
                status_msg = f"Processing {processed_files}/{metrics.total_files} files"
                self.update_phase_progress(
                    KnowledgeBasePhase.PDF_PROCESSING,
                    pdf_progress,
                    status_msg,
                    {
                        'completed_files': metrics.completed_files,
                        'failed_files': metrics.failed_files,
                        'skipped_files': metrics.skipped_files,
                        'total_files': metrics.total_files,
                        'success_rate': metrics.success_rate
                    }
                )
                
                # Update document counts
                self.update_document_counts(
                    processed=metrics.completed_files,
                    failed=metrics.failed_files,
                    total=metrics.total_files
                )
        
        # Store the sync function for periodic calling
        self._pdf_progress_sync = sync_pdf_progress
    
    def sync_pdf_progress(self) -> None:
        """Manually trigger PDF progress synchronization."""
        if hasattr(self, '_pdf_progress_sync'):
            self._pdf_progress_sync()
    
    def _trigger_callback(self) -> None:
        """Trigger the progress callback with current state."""
        if not self.progress_callback:
            return
        
        try:
            current_state = self.get_current_state()
            
            # Prepare callback parameters
            current_phase = current_state.current_phase or KnowledgeBasePhase.STORAGE_INIT
            phase_info = current_state.phase_info[current_phase]
            
            self.progress_callback(
                overall_progress=current_state.overall_progress,
                current_phase=current_phase,
                phase_progress=phase_info.current_progress,
                status_message=phase_info.status_message,
                phase_details=phase_info.details.copy(),
                all_phases=current_state.phase_info.copy()
            )
        except Exception as e:
            self.logger.warning(f"Progress callback failed: {e}")
    
    def _complete_initialization(self) -> None:
        """Handle completion of entire initialization process."""
        elapsed_time = (datetime.now() - self.state.start_time).total_seconds() if self.state.start_time else 0.0
        
        self.logger.log(
            self.progress_config.get_log_level_value(self.progress_config.progress_log_level),
            f"Knowledge base initialization completed in {elapsed_time:.2f}s - "
            f"{self.state.processed_documents}/{self.state.total_documents} documents processed"
        )
    
    def _save_progress(self) -> None:
        """Save current progress to file if enabled."""
        if not self.progress_config.save_unified_progress_to_file or not self._progress_file_path:
            return
        
        try:
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'state': self.state.to_dict(),
                'config': self.progress_config.to_dict()
            }
            
            with open(self._progress_file_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
        except (OSError, IOError, json.JSONEncodeError) as e:
            self.logger.warning(f"Failed to save unified progress to file: {e}")