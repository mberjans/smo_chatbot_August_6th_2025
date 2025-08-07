"""
Progress Integration Module for Clinical Metabolomics Oracle Knowledge Base Construction.

This module provides integration utilities and helper functions to seamlessly
integrate the unified progress tracking system with existing knowledge base
initialization workflows.

Functions:
    - create_unified_progress_tracker: Factory function for creating progress trackers
    - setup_progress_integration: Setup function for integrating with existing code
    - ProgressCallbackBuilder: Helper class for building progress callbacks
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

from .unified_progress_tracker import (
    KnowledgeBaseProgressTracker, 
    KnowledgeBasePhase, 
    PhaseWeights,
    UnifiedProgressCallback,
    UnifiedProgressState
)
from .progress_config import ProgressTrackingConfig
from .progress_tracker import PDFProcessingProgressTracker


def create_unified_progress_tracker(
    progress_config: Optional[ProgressTrackingConfig] = None,
    phase_weights: Optional[PhaseWeights] = None,
    progress_callback: Optional[UnifiedProgressCallback] = None,
    logger: Optional[logging.Logger] = None,
    enable_console_output: bool = False,
    console_update_interval: float = 5.0
) -> KnowledgeBaseProgressTracker:
    """
    Factory function for creating unified progress trackers with sensible defaults.
    
    Args:
        progress_config: Progress tracking configuration
        phase_weights: Custom phase weights (uses defaults if None)
        progress_callback: Custom progress callback
        logger: Logger instance
        enable_console_output: Whether to enable console progress output
        console_update_interval: Interval for console updates in seconds
    
    Returns:
        Configured KnowledgeBaseProgressTracker instance
    """
    # Create default configuration if none provided
    if progress_config is None:
        progress_config = ProgressTrackingConfig(
            enable_unified_progress_tracking=True,
            enable_phase_based_progress=True,
            save_unified_progress_to_file=True,
            enable_progress_callbacks=progress_callback is not None or enable_console_output
        )
    
    # Create console callback if requested
    if enable_console_output and progress_callback is None:
        progress_callback = ConsoleProgressCallback(update_interval=console_update_interval)
    
    # Create and return tracker
    return KnowledgeBaseProgressTracker(
        progress_config=progress_config,
        phase_weights=phase_weights,
        progress_callback=progress_callback,
        logger=logger
    )


def setup_progress_integration(
    rag_instance: Any,
    progress_config: Optional[ProgressTrackingConfig] = None,
    enable_callbacks: bool = False,
    callback_interval: float = 2.0
) -> KnowledgeBaseProgressTracker:
    """
    Setup progress integration for an existing ClinicalMetabolomicsRAG instance.
    
    Args:
        rag_instance: The ClinicalMetabolomicsRAG instance
        progress_config: Progress tracking configuration
        enable_callbacks: Whether to enable progress callbacks
        callback_interval: Callback update interval in seconds
    
    Returns:
        Configured progress tracker attached to the RAG instance
    """
    # Create unified progress tracker
    progress_tracker = create_unified_progress_tracker(
        progress_config=progress_config,
        logger=getattr(rag_instance, 'logger', None),
        enable_console_output=enable_callbacks,
        console_update_interval=callback_interval
    )
    
    # Attach to RAG instance
    rag_instance._unified_progress_tracker = progress_tracker
    
    # Setup integration with PDF processor if it exists
    if hasattr(rag_instance, 'pdf_processor') and rag_instance.pdf_processor:
        pdf_tracker = getattr(rag_instance.pdf_processor, 'progress_tracker', None)
        if pdf_tracker:
            progress_tracker.integrate_pdf_progress_tracker(pdf_tracker)
    
    return progress_tracker


class ConsoleProgressCallback:
    """
    Simple console-based progress callback implementation.
    
    This callback provides basic progress output to the console, suitable for
    command-line applications and debugging.
    """
    
    def __init__(self, 
                 update_interval: float = 5.0,
                 show_phase_details: bool = True,
                 show_time_estimates: bool = True):
        """
        Initialize console progress callback.
        
        Args:
            update_interval: Minimum interval between console updates
            show_phase_details: Whether to show detailed phase information
            show_time_estimates: Whether to show time estimates
        """
        self.update_interval = update_interval
        self.show_phase_details = show_phase_details
        self.show_time_estimates = show_time_estimates
        self.last_update = 0.0
    
    def __call__(self, 
                 overall_progress: float,
                 current_phase: KnowledgeBasePhase,
                 phase_progress: float,
                 status_message: str,
                 phase_details: Dict[str, Any],
                 all_phases: Dict[KnowledgeBasePhase, Any]) -> None:
        """Handle progress update callback."""
        import time
        current_time = time.time()
        
        # Rate limit console updates
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Format progress bar
        progress_bar = self._format_progress_bar(overall_progress)
        
        # Basic progress line
        print(f"\r{progress_bar} {overall_progress:.1%} | {current_phase.value}: {phase_progress:.1%}", end="")
        
        # Add status message if available
        if status_message:
            print(f" | {status_message}", end="")
        
        # Add time estimates if enabled
        if self.show_time_estimates and 'estimated_time_remaining' in phase_details:
            remaining = phase_details.get('estimated_time_remaining')
            if remaining and remaining > 0:
                print(f" | ETA: {remaining:.0f}s", end="")
        
        # Flush output
        print(flush=True)
        
        # Show phase details on new line if enabled
        if self.show_phase_details and phase_details:
            detail_items = []
            if 'completed_files' in phase_details:
                detail_items.append(f"Files: {phase_details['completed_files']}")
            if 'failed_files' in phase_details and phase_details['failed_files'] > 0:
                detail_items.append(f"Failed: {phase_details['failed_files']}")
            if 'success_rate' in phase_details:
                detail_items.append(f"Success: {phase_details['success_rate']:.1f}%")
            
            if detail_items:
                print(f"\n  └─ {' | '.join(detail_items)}")
    
    def _format_progress_bar(self, progress: float, width: int = 20) -> str:
        """Format a text-based progress bar."""
        filled = int(progress * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"


class ProgressCallbackBuilder:
    """
    Builder class for creating custom progress callbacks with various features.
    
    This class provides a fluent interface for building progress callbacks
    with logging, file output, metrics collection, and custom formatting.
    """
    
    def __init__(self):
        """Initialize callback builder."""
        self.callbacks = []
        self._logger = None
        self._log_level = logging.INFO
        self._log_interval = 10.0
        self._file_path = None
        self._file_interval = 30.0
        self._metrics_collector = None
    
    def with_logging(self, 
                    logger: logging.Logger,
                    log_level: int = logging.INFO,
                    log_interval: float = 10.0) -> 'ProgressCallbackBuilder':
        """Add logging callback."""
        self._logger = logger
        self._log_level = log_level
        self._log_interval = log_interval
        return self
    
    def with_file_output(self, 
                        file_path: Union[str, Path],
                        update_interval: float = 30.0) -> 'ProgressCallbackBuilder':
        """Add file output callback."""
        self._file_path = Path(file_path)
        self._file_interval = update_interval
        return self
    
    def with_metrics_collection(self, 
                               metrics_collector: Callable) -> 'ProgressCallbackBuilder':
        """Add metrics collection callback."""
        self._metrics_collector = metrics_collector
        return self
    
    def with_console_output(self, 
                           update_interval: float = 5.0,
                           show_details: bool = True) -> 'ProgressCallbackBuilder':
        """Add console output callback."""
        console_callback = ConsoleProgressCallback(
            update_interval=update_interval,
            show_phase_details=show_details
        )
        self.callbacks.append(console_callback)
        return self
    
    def with_custom_callback(self, 
                           callback: UnifiedProgressCallback) -> 'ProgressCallbackBuilder':
        """Add custom callback."""
        self.callbacks.append(callback)
        return self
    
    def build(self) -> UnifiedProgressCallback:
        """Build the composite callback."""
        # Add built-in callbacks based on configuration
        if self._logger:
            self.callbacks.append(
                LoggingProgressCallback(
                    logger=self._logger,
                    log_level=self._log_level,
                    update_interval=self._log_interval
                )
            )
        
        if self._file_path:
            self.callbacks.append(
                FileOutputProgressCallback(
                    file_path=self._file_path,
                    update_interval=self._file_interval
                )
            )
        
        if self._metrics_collector:
            self.callbacks.append(
                MetricsCollectionCallback(self._metrics_collector)
            )
        
        # Return composite callback
        return CompositeProgressCallback(self.callbacks)


class LoggingProgressCallback:
    """Progress callback that outputs to a logger."""
    
    def __init__(self, 
                 logger: logging.Logger,
                 log_level: int = logging.INFO,
                 update_interval: float = 10.0):
        """Initialize logging callback."""
        self.logger = logger
        self.log_level = log_level
        self.update_interval = update_interval
        self.last_update = 0.0
    
    def __call__(self, 
                 overall_progress: float,
                 current_phase: KnowledgeBasePhase,
                 phase_progress: float,
                 status_message: str,
                 phase_details: Dict[str, Any],
                 all_phases: Dict[KnowledgeBasePhase, Any]) -> None:
        """Handle progress update callback."""
        import time
        current_time = time.time()
        
        # Rate limit log updates
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Format log message
        message = f"Progress: {overall_progress:.1%} | Phase: {current_phase.value} ({phase_progress:.1%})"
        if status_message:
            message += f" | {status_message}"
        
        self.logger.log(self.log_level, message)


class FileOutputProgressCallback:
    """Progress callback that writes to a file."""
    
    def __init__(self, 
                 file_path: Path,
                 update_interval: float = 30.0):
        """Initialize file output callback."""
        self.file_path = file_path
        self.update_interval = update_interval
        self.last_update = 0.0
        
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, 
                 overall_progress: float,
                 current_phase: KnowledgeBasePhase,
                 phase_progress: float,
                 status_message: str,
                 phase_details: Dict[str, Any],
                 all_phases: Dict[KnowledgeBasePhase, Any]) -> None:
        """Handle progress update callback."""
        import time
        import json
        
        current_time = time.time()
        
        # Rate limit file updates
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Prepare progress data
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_progress': overall_progress,
            'current_phase': current_phase.value,
            'phase_progress': phase_progress,
            'status_message': status_message,
            'phase_details': phase_details,
            'all_phases': {phase.value: phase_info.to_dict() 
                          for phase, phase_info in all_phases.items()}
        }
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        except (OSError, IOError) as e:
            # Silently ignore file write errors to avoid disrupting progress
            pass


class MetricsCollectionCallback:
    """Progress callback that collects metrics via a custom function."""
    
    def __init__(self, metrics_collector: Callable):
        """Initialize metrics collection callback."""
        self.metrics_collector = metrics_collector
    
    def __call__(self, 
                 overall_progress: float,
                 current_phase: KnowledgeBasePhase,
                 phase_progress: float,
                 status_message: str,
                 phase_details: Dict[str, Any],
                 all_phases: Dict[KnowledgeBasePhase, Any]) -> None:
        """Handle progress update callback."""
        try:
            self.metrics_collector({
                'overall_progress': overall_progress,
                'current_phase': current_phase.value,
                'phase_progress': phase_progress,
                'status_message': status_message,
                'phase_details': phase_details,
                'timestamp': datetime.now().isoformat()
            })
        except Exception:
            # Silently ignore metrics collection errors
            pass


class CompositeProgressCallback:
    """Composite callback that delegates to multiple callbacks."""
    
    def __init__(self, callbacks: List[UnifiedProgressCallback]):
        """Initialize composite callback."""
        self.callbacks = callbacks
    
    def __call__(self, 
                 overall_progress: float,
                 current_phase: KnowledgeBasePhase,
                 phase_progress: float,
                 status_message: str,
                 phase_details: Dict[str, Any],
                 all_phases: Dict[KnowledgeBasePhase, Any]) -> None:
        """Handle progress update callback by delegating to all callbacks."""
        for callback in self.callbacks:
            try:
                callback(
                    overall_progress=overall_progress,
                    current_phase=current_phase,
                    phase_progress=phase_progress,
                    status_message=status_message,
                    phase_details=phase_details,
                    all_phases=all_phases
                )
            except Exception:
                # Continue with other callbacks even if one fails
                pass


def estimate_phase_durations(total_documents: int) -> Dict[KnowledgeBasePhase, float]:
    """
    Estimate phase durations based on document count and empirical data.
    
    Args:
        total_documents: Total number of documents to process
    
    Returns:
        Dictionary mapping phases to estimated durations in seconds
    """
    # Base estimates (for reference workload)
    base_estimates = {
        KnowledgeBasePhase.STORAGE_INIT: 5.0,  # Usually quick
        KnowledgeBasePhase.PDF_PROCESSING: 30.0,  # Per document
        KnowledgeBasePhase.DOCUMENT_INGESTION: 10.0,  # Per document  
        KnowledgeBasePhase.FINALIZATION: 10.0,  # Usually quick
    }
    
    # Scale by document count
    estimates = {}
    for phase, base_time in base_estimates.items():
        if phase in [KnowledgeBasePhase.PDF_PROCESSING, KnowledgeBasePhase.DOCUMENT_INGESTION]:
            estimates[phase] = base_time * total_documents
        else:
            estimates[phase] = base_time
    
    return estimates


# Utility functions for easy integration with existing code

def add_progress_tracking_to_method(method_name: str = "initialize_knowledge_base"):
    """
    Decorator factory for adding progress tracking to methods.
    
    Args:
        method_name: Name of the method being decorated
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Extract progress configuration from kwargs
            progress_config = kwargs.get('progress_config')
            
            # Setup progress tracker if unified tracking is enabled
            if (progress_config and 
                getattr(progress_config, 'enable_unified_progress_tracking', False)):
                
                progress_tracker = setup_progress_integration(
                    rag_instance=self,
                    progress_config=progress_config,
                    enable_callbacks=getattr(progress_config, 'enable_progress_callbacks', False)
                )
                
                # Store tracker for method access
                kwargs['_unified_progress_tracker'] = progress_tracker
            
            # Call original method
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator