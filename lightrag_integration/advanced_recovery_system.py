#!/usr/bin/env python3
"""
Advanced Recovery and Graceful Degradation System for Clinical Metabolomics Oracle.

This module implements sophisticated recovery mechanisms and graceful degradation
strategies for handling ingestion failures in the Clinical Metabolomics Oracle system.

Features:
    - Progressive degradation strategies with fallback processing modes
    - Resource-aware recovery with system monitoring
    - Intelligent retry backoff with adaptive strategies
    - Checkpoint and resume capability for long-running processes
    - Multiple graceful degradation modes (essential, minimal, offline, safe)
    - Integration with existing progress tracking and error handling

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import asyncio
import json
import logging
import psutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import random
import hashlib

from .progress_config import ProgressTrackingConfig
from .unified_progress_tracker import KnowledgeBaseProgressTracker, KnowledgeBasePhase


class DegradationMode(Enum):
    """Enumeration for different degradation modes."""
    OPTIMAL = "optimal"              # Full processing with all features
    ESSENTIAL = "essential"          # Process only critical documents
    MINIMAL = "minimal"              # Reduce processing complexity for speed
    OFFLINE = "offline"              # Queue documents when APIs unavailable
    SAFE = "safe"                    # Ultra-conservative with maximum tolerance


class FailureType(Enum):
    """Types of failures that can trigger recovery mechanisms."""
    API_RATE_LIMIT = "api_rate_limit"
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_SPACE = "disk_space"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class BackoffStrategy(Enum):
    """Different backoff strategies for retries."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"


@dataclass
class ResourceThresholds:
    """System resource monitoring thresholds."""
    memory_warning_percent: float = 75.0    # Warning at 75% memory usage
    memory_critical_percent: float = 90.0   # Critical at 90% memory usage
    disk_warning_percent: float = 80.0      # Warning at 80% disk usage
    disk_critical_percent: float = 95.0     # Critical at 95% disk usage
    cpu_warning_percent: float = 80.0       # Warning at 80% CPU usage
    cpu_critical_percent: float = 95.0      # Critical at 95% CPU usage


@dataclass
class DegradationConfig:
    """Configuration for degradation strategies."""
    mode: DegradationMode = DegradationMode.OPTIMAL
    enable_partial_processing: bool = True
    reduce_batch_size: bool = True
    skip_optional_metadata: bool = False
    disable_advanced_chunking: bool = False
    enable_document_priority: bool = True
    max_retry_attempts: int = 3
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 300.0


@dataclass
class CheckpointData:
    """Data structure for process checkpoints."""
    checkpoint_id: str
    timestamp: datetime
    phase: KnowledgeBasePhase
    processed_documents: List[str]
    failed_documents: Dict[str, str]  # document_id -> error_message
    pending_documents: List[str]
    current_batch_size: int
    degradation_mode: DegradationMode
    system_resources: Dict[str, float]
    error_counts: Dict[FailureType, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp.isoformat(),
            'phase': self.phase.value,
            'processed_documents': self.processed_documents.copy(),
            'failed_documents': self.failed_documents.copy(),
            'pending_documents': self.pending_documents.copy(),
            'current_batch_size': self.current_batch_size,
            'degradation_mode': self.degradation_mode.value,
            'system_resources': self.system_resources.copy(),
            'error_counts': {k.value: v for k, v in self.error_counts.items()},
            'metadata': self.metadata.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create checkpoint from dictionary."""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            phase=KnowledgeBasePhase(data['phase']),
            processed_documents=data['processed_documents'],
            failed_documents=data['failed_documents'],
            pending_documents=data['pending_documents'],
            current_batch_size=data['current_batch_size'],
            degradation_mode=DegradationMode(data['degradation_mode']),
            system_resources=data['system_resources'],
            error_counts={FailureType(k): v for k, v in data['error_counts'].items()},
            metadata=data.get('metadata', {})
        )


class SystemResourceMonitor:
    """Monitors system resources and triggers degradation when thresholds are exceeded."""
    
    def __init__(self, 
                 thresholds: Optional[ResourceThresholds] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize resource monitor.
        
        Args:
            thresholds: Resource threshold configuration
            logger: Logger instance
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.logger = logger or logging.getLogger(__name__)
        self._last_check = datetime.now()
        self._check_interval = 30.0  # Check every 30 seconds
        
    def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu = psutil.cpu_percent(interval=1)
            
            return {
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'cpu_percent': cpu,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system resources: {e}")
            return {}
    
    def check_resource_pressure(self) -> Dict[str, str]:
        """
        Check for resource pressure and return recommendations.
        
        Returns:
            Dictionary with resource type -> recommendation
        """
        resources = self.get_current_resources()
        recommendations = {}
        
        # Check memory
        if 'memory_percent' in resources:
            mem_pct = resources['memory_percent']
            if mem_pct >= self.thresholds.memory_critical_percent:
                recommendations['memory'] = 'critical_reduce_batch_size'
            elif mem_pct >= self.thresholds.memory_warning_percent:
                recommendations['memory'] = 'warning_monitor_usage'
        
        # Check disk
        if 'disk_percent' in resources:
            disk_pct = resources['disk_percent']
            if disk_pct >= self.thresholds.disk_critical_percent:
                recommendations['disk'] = 'critical_cleanup_required'
            elif disk_pct >= self.thresholds.disk_warning_percent:
                recommendations['disk'] = 'warning_cleanup_recommended'
        
        # Check CPU
        if 'cpu_percent' in resources:
            cpu_pct = resources['cpu_percent']
            if cpu_pct >= self.thresholds.cpu_critical_percent:
                recommendations['cpu'] = 'critical_reduce_concurrency'
            elif cpu_pct >= self.thresholds.cpu_warning_percent:
                recommendations['cpu'] = 'warning_consider_throttling'
        
        return recommendations


class AdaptiveBackoffCalculator:
    """Calculates intelligent retry backoff based on error patterns and system state."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize backoff calculator."""
        self.logger = logger or logging.getLogger(__name__)
        self._failure_history: Dict[FailureType, List[datetime]] = {}
        self._success_history: List[datetime] = []
        self._last_api_response_time = 1.0
        
    def calculate_backoff(self, 
                         failure_type: FailureType,
                         attempt: int,
                         strategy: BackoffStrategy = BackoffStrategy.ADAPTIVE,
                         base_delay: float = 1.0,
                         max_delay: float = 300.0,
                         jitter: bool = True) -> float:
        """
        Calculate intelligent backoff delay.
        
        Args:
            failure_type: Type of failure that occurred
            attempt: Current attempt number (1-based)
            strategy: Backoff strategy to use
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add jitter
            
        Returns:
            Backoff delay in seconds
        """
        # Record failure
        now = datetime.now()
        if failure_type not in self._failure_history:
            self._failure_history[failure_type] = []
        self._failure_history[failure_type].append(now)
        
        # Clean old history (keep last hour)
        cutoff = now - timedelta(hours=1)
        for failure_list in self._failure_history.values():
            failure_list[:] = [ts for ts in failure_list if ts > cutoff]
        self._success_history[:] = [ts for ts in self._success_history if ts > cutoff]
        
        # Calculate base delay based on strategy
        if strategy == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** (attempt - 1))
        elif strategy == BackoffStrategy.LINEAR:
            delay = base_delay * attempt
        elif strategy == BackoffStrategy.FIBONACCI:
            delay = base_delay * self._fibonacci(attempt)
        elif strategy == BackoffStrategy.ADAPTIVE:
            delay = self._adaptive_backoff(failure_type, attempt, base_delay)
        else:
            delay = base_delay
        
        # Apply maximum
        delay = min(delay, max_delay)
        
        # Add jitter if requested
        if jitter:
            jitter_factor = 0.1 + (random.random() * 0.2)  # 10-30% jitter
            delay *= (1.0 + jitter_factor)
        
        return delay
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self._success_history.append(datetime.now())
    
    def update_api_response_time(self, response_time: float) -> None:
        """Update the last observed API response time."""
        self._last_api_response_time = response_time
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 2:
            return 1
        a, b = 1, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
    
    def _adaptive_backoff(self, failure_type: FailureType, attempt: int, base_delay: float) -> float:
        """Calculate adaptive backoff based on failure patterns."""
        # Start with exponential base
        delay = base_delay * (2 ** (attempt - 1))
        
        # Adjust based on failure frequency
        recent_failures = len(self._failure_history.get(failure_type, []))
        if recent_failures > 10:  # High failure rate
            delay *= 2.0
        elif recent_failures > 5:  # Moderate failure rate
            delay *= 1.5
        
        # Adjust based on success rate
        recent_successes = len(self._success_history)
        if recent_successes > 0 and recent_failures > 0:
            success_rate = recent_successes / (recent_successes + recent_failures)
            if success_rate < 0.5:  # Low success rate
                delay *= 1.5
        
        # Adjust based on API response time
        if self._last_api_response_time > 5.0:  # Slow API
            delay *= 1.2
        elif self._last_api_response_time > 10.0:  # Very slow API
            delay *= 1.5
        
        # Special handling for specific failure types
        if failure_type == FailureType.API_RATE_LIMIT:
            delay *= 3.0  # Rate limits need longer waits
        elif failure_type == FailureType.MEMORY_PRESSURE:
            delay *= 0.5  # Memory issues might resolve quickly
        
        return delay


class CheckpointManager:
    """Manages checkpoints for resumable long-running processes."""
    
    def __init__(self, 
                 checkpoint_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            logger: Logger instance
        """
        self.checkpoint_dir = checkpoint_dir or Path("logs/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        
    def create_checkpoint(self,
                         phase: KnowledgeBasePhase,
                         processed_documents: List[str],
                         failed_documents: Dict[str, str],
                         pending_documents: List[str],
                         current_batch_size: int,
                         degradation_mode: DegradationMode,
                         system_resources: Dict[str, float],
                         error_counts: Dict[FailureType, int],
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new checkpoint.
        
        Args:
            phase: Current processing phase
            processed_documents: List of successfully processed document IDs
            failed_documents: Map of failed document IDs to error messages
            pending_documents: List of documents still to process
            current_batch_size: Current batch size being used
            degradation_mode: Current degradation mode
            system_resources: Current system resource usage
            error_counts: Count of errors by type
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = self._generate_checkpoint_id(phase, processed_documents)
        
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            phase=phase,
            processed_documents=processed_documents.copy(),
            failed_documents=failed_documents.copy(),
            pending_documents=pending_documents.copy(),
            current_batch_size=current_batch_size,
            degradation_mode=degradation_mode,
            system_resources=system_resources.copy(),
            error_counts=error_counts.copy(),
            metadata=metadata or {}
        )
        
        # Save to disk
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        with self._lock:
            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Created checkpoint {checkpoint_id} with {len(pending_documents)} pending documents")
                return checkpoint_id
                
            except (OSError, IOError, json.JSONDecodeError, TypeError) as e:
                self.logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
                raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        with self._lock:
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                checkpoint = CheckpointData.from_dict(data)
                self.logger.info(f"Loaded checkpoint {checkpoint_id} from {checkpoint.timestamp}")
                return checkpoint
                
            except (OSError, IOError, json.JSONDecodeError, TypeError) as e:
                self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
                return None
    
    def list_checkpoints(self, phase: Optional[KnowledgeBasePhase] = None) -> List[str]:
        """
        List available checkpoints.
        
        Args:
            phase: Optional phase filter
            
        Returns:
            List of checkpoint IDs
        """
        checkpoints = []
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                if phase is None:
                    checkpoints.append(checkpoint_file.stem)
                else:
                    # Load and check phase
                    try:
                        with open(checkpoint_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if data.get('phase') == phase.value:
                            checkpoints.append(checkpoint_file.stem)
                    except Exception:
                        continue  # Skip invalid checkpoints
                        
        except OSError as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
        
        return sorted(checkpoints)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        with self._lock:
            try:
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    self.logger.info(f"Deleted checkpoint {checkpoint_id}")
                    return True
                return False
                
            except OSError as e:
                self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
                return False
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        Clean up old checkpoints.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of checkpoints deleted
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    # Check file modification time
                    if datetime.fromtimestamp(checkpoint_file.stat().st_mtime) < cutoff:
                        checkpoint_file.unlink()
                        deleted_count += 1
                        
                except OSError:
                    continue  # Skip files we can't process
                    
        except OSError as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old checkpoints")
        
        return deleted_count
    
    def _generate_checkpoint_id(self, phase: KnowledgeBasePhase, processed_documents: List[str]) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(''.join(sorted(processed_documents)).encode()).hexdigest()[:8]
        return f"{phase.value}_{timestamp}_{content_hash}"


class AdvancedRecoverySystem:
    """
    Advanced recovery and graceful degradation system.
    
    This system provides sophisticated recovery mechanisms including:
    - Progressive degradation with multiple fallback strategies
    - Resource-aware recovery and adaptive processing
    - Intelligent retry with multiple backoff strategies
    - Checkpoint and resume capabilities
    - Multiple degradation modes for different scenarios
    """
    
    def __init__(self,
                 progress_tracker: Optional[KnowledgeBaseProgressTracker] = None,
                 resource_thresholds: Optional[ResourceThresholds] = None,
                 checkpoint_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize advanced recovery system.
        
        Args:
            progress_tracker: Progress tracking system to integrate with
            resource_thresholds: Resource monitoring thresholds
            checkpoint_dir: Directory for checkpoint storage
            logger: Logger instance
        """
        self.progress_tracker = progress_tracker
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize subsystems
        self.resource_monitor = SystemResourceMonitor(resource_thresholds, self.logger)
        self.backoff_calculator = AdaptiveBackoffCalculator(self.logger)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, self.logger)
        
        # Current state
        self.current_degradation_mode = DegradationMode.OPTIMAL
        self.degradation_config = DegradationConfig()
        self._error_counts: Dict[FailureType, int] = {}
        self._current_batch_size = 10  # Default batch size
        self._original_batch_size = 10
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Document tracking
        self._processed_documents: Set[str] = set()
        self._failed_documents: Dict[str, str] = {}
        self._pending_documents: List[str] = []
        self._current_phase = KnowledgeBasePhase.STORAGE_INIT
    
    def initialize_ingestion_session(self,
                                   documents: List[str],
                                   phase: KnowledgeBasePhase,
                                   batch_size: int = 10) -> None:
        """
        Initialize a new ingestion session.
        
        Args:
            documents: List of document IDs to process
            phase: Processing phase
            batch_size: Initial batch size
        """
        with self._lock:
            self._pending_documents = documents.copy()
            self._processed_documents.clear()
            self._failed_documents.clear()
            self._current_phase = phase
            self._current_batch_size = batch_size
            self._original_batch_size = batch_size
            self.current_degradation_mode = DegradationMode.OPTIMAL
            self._error_counts.clear()
            
        self.logger.info(f"Initialized ingestion session: {len(documents)} documents in {phase.value}")
    
    def handle_failure(self, 
                      failure_type: FailureType,
                      error_message: str,
                      document_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a failure and determine recovery strategy.
        
        Args:
            failure_type: Type of failure that occurred
            error_message: Error message
            document_id: ID of document that failed (if applicable)
            context: Additional context information
            
        Returns:
            Recovery strategy recommendations
        """
        with self._lock:
            # Record the failure
            self._error_counts[failure_type] = self._error_counts.get(failure_type, 0) + 1
            
            if document_id:
                self._failed_documents[document_id] = error_message
                # Remove from pending if it was there
                if document_id in self._pending_documents:
                    self._pending_documents.remove(document_id)
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(failure_type, error_message, context)
            
            # Apply immediate mitigations
            self._apply_recovery_strategy(recovery_strategy)
            
            # Log the failure and response
            self.logger.warning(
                f"Handling {failure_type.value} failure: {error_message} "
                f"-> Strategy: {recovery_strategy.get('action', 'unknown')}"
            )
            
            return recovery_strategy
    
    def get_next_batch(self) -> List[str]:
        """
        Get the next batch of documents to process based on current strategy.
        
        Returns:
            List of document IDs to process next
        """
        with self._lock:
            if not self._pending_documents:
                return []
            
            # Adjust batch size based on current conditions
            batch_size = self._calculate_optimal_batch_size()
            
            # Apply document prioritization if enabled
            if self.degradation_config.enable_document_priority:
                self._pending_documents = self._prioritize_documents(self._pending_documents)
            
            # Extract next batch
            next_batch = self._pending_documents[:batch_size]
            return next_batch
    
    def mark_document_processed(self, document_id: str) -> None:
        """Mark a document as successfully processed."""
        with self._lock:
            self._processed_documents.add(document_id)
            if document_id in self._pending_documents:
                self._pending_documents.remove(document_id)
            
            # Record success for backoff calculation
            self.backoff_calculator.record_success()
    
    def create_checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a checkpoint of current processing state.
        
        Args:
            metadata: Additional metadata to store
            
        Returns:
            Checkpoint ID
        """
        with self._lock:
            system_resources = self.resource_monitor.get_current_resources()
            
            return self.checkpoint_manager.create_checkpoint(
                phase=self._current_phase,
                processed_documents=list(self._processed_documents),
                failed_documents=self._failed_documents.copy(),
                pending_documents=self._pending_documents.copy(),
                current_batch_size=self._current_batch_size,
                degradation_mode=self.current_degradation_mode,
                system_resources=system_resources,
                error_counts=self._error_counts.copy(),
                metadata=metadata
            )
    
    def resume_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Resume processing from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to resume from
            
        Returns:
            True if resumed successfully
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if not checkpoint:
            self.logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        with self._lock:
            self._processed_documents = set(checkpoint.processed_documents)
            self._failed_documents = checkpoint.failed_documents.copy()
            self._pending_documents = checkpoint.pending_documents.copy()
            self._current_phase = checkpoint.phase
            self._current_batch_size = checkpoint.current_batch_size
            self.current_degradation_mode = checkpoint.degradation_mode
            self._error_counts = checkpoint.error_counts.copy()
        
        self.logger.info(
            f"Resumed from checkpoint {checkpoint_id}: "
            f"{len(self._pending_documents)} pending, "
            f"{len(self._processed_documents)} processed, "
            f"{len(self._failed_documents)} failed"
        )
        
        return True
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """
        Get current recovery system status.
        
        Returns:
            Status information dictionary
        """
        with self._lock:
            system_resources = self.resource_monitor.get_current_resources()
            resource_pressure = self.resource_monitor.check_resource_pressure()
            
            total_documents = len(self._processed_documents) + len(self._failed_documents) + len(self._pending_documents)
            
            return {
                'degradation_mode': self.current_degradation_mode.value,
                'current_batch_size': self._current_batch_size,
                'original_batch_size': self._original_batch_size,
                'document_progress': {
                    'total': total_documents,
                    'processed': len(self._processed_documents),
                    'failed': len(self._failed_documents),
                    'pending': len(self._pending_documents),
                    'success_rate': len(self._processed_documents) / max(1, total_documents - len(self._pending_documents))
                },
                'error_counts': {k.value: v for k, v in self._error_counts.items()},
                'system_resources': system_resources,
                'resource_pressure': resource_pressure,
                'current_phase': self._current_phase.value,
                'available_checkpoints': len(self.checkpoint_manager.list_checkpoints())
            }
    
    def _determine_recovery_strategy(self, 
                                   failure_type: FailureType,
                                   error_message: str,
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the appropriate recovery strategy for a failure."""
        strategy = {
            'action': 'retry',
            'degradation_needed': False,
            'batch_size_adjustment': 1.0,
            'backoff_seconds': 1.0,
            'checkpoint_recommended': False
        }
        
        # Check system resources
        resource_pressure = self.resource_monitor.check_resource_pressure()
        
        # Strategy based on failure type
        if failure_type == FailureType.API_RATE_LIMIT:
            strategy.update({
                'action': 'backoff_and_retry',
                'backoff_seconds': self.backoff_calculator.calculate_backoff(
                    failure_type, self._error_counts.get(failure_type, 1), BackoffStrategy.ADAPTIVE
                ),
                'batch_size_adjustment': 0.5,  # Reduce batch size
                'degradation_needed': self._error_counts.get(failure_type, 0) > 3
            })
        
        elif failure_type == FailureType.MEMORY_PRESSURE or 'memory' in resource_pressure:
            strategy.update({
                'action': 'reduce_resources',
                'batch_size_adjustment': 0.3,  # Aggressive reduction
                'degradation_needed': True,
                'checkpoint_recommended': True
            })
        
        elif failure_type == FailureType.API_ERROR:
            error_count = self._error_counts.get(failure_type, 0)
            if error_count > 5:
                strategy.update({
                    'action': 'degrade_to_safe_mode',
                    'degradation_needed': True,
                    'checkpoint_recommended': True
                })
        
        elif failure_type == FailureType.NETWORK_ERROR:
            strategy.update({
                'action': 'switch_to_offline_mode',
                'degradation_needed': True,
                'backoff_seconds': self.backoff_calculator.calculate_backoff(failure_type, 1)
            })
        
        # Adjust based on current degradation mode
        if self.current_degradation_mode != DegradationMode.OPTIMAL:
            strategy['backoff_seconds'] *= 0.5  # Shorter waits in degraded mode
        
        return strategy
    
    def _apply_recovery_strategy(self, strategy: Dict[str, Any]) -> None:
        """Apply the determined recovery strategy."""
        action = strategy.get('action', 'retry')
        
        # Adjust batch size if needed
        batch_adjustment = strategy.get('batch_size_adjustment', 1.0)
        if batch_adjustment != 1.0:
            new_batch_size = max(1, int(self._current_batch_size * batch_adjustment))
            self._current_batch_size = new_batch_size
            self.logger.info(f"Adjusted batch size to {new_batch_size}")
        
        # Apply degradation if needed
        if strategy.get('degradation_needed', False):
            self._apply_degradation(action)
        
        # Create checkpoint if recommended
        if strategy.get('checkpoint_recommended', False):
            checkpoint_id = self.create_checkpoint({'recovery_action': action})
            self.logger.info(f"Created recovery checkpoint: {checkpoint_id}")
    
    def _apply_degradation(self, trigger_action: str) -> None:
        """Apply appropriate degradation based on current conditions."""
        current_mode = self.current_degradation_mode
        
        # Determine target degradation mode
        if trigger_action == 'reduce_resources':
            target_mode = DegradationMode.MINIMAL
        elif trigger_action == 'degrade_to_safe_mode':
            target_mode = DegradationMode.SAFE
        elif trigger_action == 'switch_to_offline_mode':
            target_mode = DegradationMode.OFFLINE
        else:
            target_mode = DegradationMode.ESSENTIAL
        
        # Don't degrade further if already in more restrictive mode
        mode_hierarchy = {
            DegradationMode.OPTIMAL: 0,
            DegradationMode.ESSENTIAL: 1,
            DegradationMode.MINIMAL: 2,
            DegradationMode.OFFLINE: 3,
            DegradationMode.SAFE: 4
        }
        
        if mode_hierarchy[target_mode] > mode_hierarchy[current_mode]:
            self.current_degradation_mode = target_mode
            self._update_degradation_config()
            self.logger.warning(f"Degraded to {target_mode.value} mode due to {trigger_action}")
    
    def _update_degradation_config(self) -> None:
        """Update degradation configuration based on current mode."""
        mode = self.current_degradation_mode
        
        if mode == DegradationMode.ESSENTIAL:
            self.degradation_config.skip_optional_metadata = True
            self.degradation_config.reduce_batch_size = True
            self.degradation_config.max_retry_attempts = 2
        
        elif mode == DegradationMode.MINIMAL:
            self.degradation_config.skip_optional_metadata = True
            self.degradation_config.disable_advanced_chunking = True
            self.degradation_config.reduce_batch_size = True
            self.degradation_config.max_retry_attempts = 2
            self._current_batch_size = max(1, self._current_batch_size // 2)
        
        elif mode == DegradationMode.OFFLINE:
            self.degradation_config.max_retry_attempts = 1
            self._current_batch_size = 1  # Process one at a time
        
        elif mode == DegradationMode.SAFE:
            self.degradation_config.skip_optional_metadata = True
            self.degradation_config.disable_advanced_chunking = True
            self.degradation_config.reduce_batch_size = True
            self.degradation_config.max_retry_attempts = 5
            self.degradation_config.backoff_multiplier = 3.0
            self._current_batch_size = 1  # Ultra-conservative
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on current conditions."""
        base_size = self._current_batch_size
        
        # Adjust based on resource pressure
        resource_pressure = self.resource_monitor.check_resource_pressure()
        
        if 'memory' in resource_pressure and 'critical' in resource_pressure['memory']:
            base_size = max(1, base_size // 4)
        elif 'memory' in resource_pressure and 'warning' in resource_pressure['memory']:
            base_size = max(1, base_size // 2)
        
        if 'cpu' in resource_pressure and 'critical' in resource_pressure['cpu']:
            base_size = max(1, base_size // 2)
        
        # Apply degradation mode limits
        if self.current_degradation_mode == DegradationMode.SAFE:
            base_size = 1
        elif self.current_degradation_mode == DegradationMode.MINIMAL:
            base_size = min(base_size, 3)
        elif self.current_degradation_mode == DegradationMode.ESSENTIAL:
            base_size = min(base_size, 5)
        
        return max(1, base_size)
    
    def _prioritize_documents(self, documents: List[str]) -> List[str]:
        """
        Prioritize documents for processing based on current degradation mode.
        
        Args:
            documents: List of document IDs
            
        Returns:
            Prioritized list of document IDs
        """
        if self.current_degradation_mode == DegradationMode.ESSENTIAL:
            # In essential mode, prioritize documents with "essential" or "critical" keywords
            # This is a simplified implementation - in practice, you'd have more sophisticated prioritization
            essential_docs = []
            other_docs = []
            
            for doc_id in documents:
                if any(keyword in doc_id.lower() for keyword in ['essential', 'critical', 'important']):
                    essential_docs.append(doc_id)
                else:
                    other_docs.append(doc_id)
            
            return essential_docs + other_docs
        
        return documents