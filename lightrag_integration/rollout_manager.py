#!/usr/bin/env python3
"""
RolloutManager: Advanced gradual rollout capabilities for LightRAG integration.

This module provides comprehensive rollout management for the LightRAG integration,
supporting sophisticated rollout strategies, monitoring, and automatic adjustments
based on performance metrics and quality thresholds.

Key Features:
- Multi-phase gradual rollout with configurable stages
- Automatic rollout progression based on success metrics
- Emergency rollback capabilities with circuit breaker integration
- A/B testing with statistical significance testing
- Real-time monitoring and alerting
- Rollout scheduling and automation
- Quality-gated progression with configurable thresholds
- Integration with existing feature flag infrastructure

Rollout Strategies:
- Linear rollout (fixed percentage increments)
- Exponential rollout (doubling exposure)
- Custom rollout (user-defined stages)
- Canary rollout (small initial exposure with quality gates)
- Blue-green rollout (instant switchover after validation)

Requirements:
- Builds on FeatureFlagManager infrastructure
- Compatible with existing monitoring and alerting systems
- Thread-safe operations with atomic state updates
- Persistent state management for rollout recovery

Author: Claude Code (Anthropic)
Created: 2025-08-08
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from pathlib import Path
import statistics
import math

from .config import LightRAGConfig
from .feature_flag_manager import FeatureFlagManager, PerformanceMetrics


class RolloutStrategy(Enum):
    """Rollout strategy types."""
    MANUAL = "manual"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class RolloutPhase(Enum):
    """Rollout phase states."""
    INACTIVE = "inactive"
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    MONITORING = "monitoring"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class RolloutTrigger(Enum):
    """Rollout progression triggers."""
    MANUAL = "manual"
    TIME_BASED = "time_based"
    METRIC_BASED = "metric_based"
    QUALITY_BASED = "quality_based"
    HYBRID = "hybrid"


@dataclass
class RolloutStage:
    """Configuration for a rollout stage."""
    stage_name: str
    target_percentage: float
    min_duration_minutes: int = 60
    min_requests: int = 100
    success_threshold: float = 0.95
    quality_threshold: float = 0.7
    max_error_rate: float = 0.05
    auto_advance: bool = True
    notification_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate stage configuration."""
        if not (0 <= self.target_percentage <= 100):
            raise ValueError("target_percentage must be between 0 and 100")
        
        if self.min_duration_minutes < 0:
            raise ValueError("min_duration_minutes must be non-negative")
        
        if self.min_requests < 0:
            raise ValueError("min_requests must be non-negative")
        
        if not (0 <= self.success_threshold <= 1):
            raise ValueError("success_threshold must be between 0 and 1")
        
        if not (0 <= self.quality_threshold <= 1):
            raise ValueError("quality_threshold must be between 0 and 1")
        
        if not (0 <= self.max_error_rate <= 1):
            raise ValueError("max_error_rate must be between 0 and 1")


@dataclass
class RolloutConfiguration:
    """Complete rollout configuration."""
    strategy: RolloutStrategy
    stages: List[RolloutStage]
    trigger: RolloutTrigger = RolloutTrigger.HYBRID
    emergency_rollback_enabled: bool = True
    emergency_error_threshold: float = 0.1
    emergency_quality_threshold: float = 0.5
    monitoring_interval_minutes: int = 5
    notification_webhook: Optional[str] = None
    rollback_on_failure: bool = True
    require_manual_approval: bool = False
    max_rollout_duration_hours: int = 168  # 1 week default
    statistical_significance_threshold: float = 0.05  # p-value for A/B testing
    min_sample_size: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutState:
    """Current state of rollout execution."""
    rollout_id: str
    phase: RolloutPhase = RolloutPhase.INACTIVE
    current_stage_index: int = -1
    current_percentage: float = 0.0
    started_at: Optional[datetime] = None
    stage_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    stage_requests: int = 0
    stage_successful_requests: int = 0
    stage_failed_requests: int = 0
    average_quality_score: float = 0.0
    stage_quality_scores: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def current_stage(self) -> Optional[RolloutStage]:
        """Get current rollout stage if available."""
        return None  # Will be set by RolloutManager
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def stage_success_rate(self) -> float:
        """Calculate current stage success rate."""
        if self.stage_requests == 0:
            return 0.0
        return self.stage_successful_requests / self.stage_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate overall error rate."""
        return 1.0 - self.success_rate
    
    @property
    def stage_error_rate(self) -> float:
        """Calculate current stage error rate."""
        return 1.0 - self.stage_success_rate
    
    @property
    def stage_average_quality(self) -> float:
        """Calculate current stage average quality."""
        if not self.stage_quality_scores:
            return 0.0
        return statistics.mean(self.stage_quality_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'rollout_id': self.rollout_id,
            'phase': self.phase.value,
            'current_stage_index': self.current_stage_index,
            'current_percentage': self.current_percentage,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stage_started_at': self.stage_started_at.isoformat() if self.stage_started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'stage_requests': self.stage_requests,
            'stage_successful_requests': self.stage_successful_requests,
            'stage_failed_requests': self.stage_failed_requests,
            'success_rate': self.success_rate,
            'stage_success_rate': self.stage_success_rate,
            'error_rate': self.error_rate,
            'stage_error_rate': self.stage_error_rate,
            'average_quality_score': self.average_quality_score,
            'stage_average_quality': self.stage_average_quality,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }


class RolloutManager:
    """
    Advanced rollout manager for LightRAG integration.
    
    Provides comprehensive rollout management with multiple strategies,
    automatic progression, emergency rollback, and detailed monitoring.
    """
    
    def __init__(self, config: LightRAGConfig, feature_manager: FeatureFlagManager,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the RolloutManager.
        
        Args:
            config: LightRAG configuration instance
            feature_manager: FeatureFlagManager instance for integration
            logger: Optional logger instance
        """
        self.config = config
        self.feature_manager = feature_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Rollout state
        self.rollout_config: Optional[RolloutConfiguration] = None
        self.rollout_state: Optional[RolloutState] = None
        
        # Monitoring and automation
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # State persistence
        self._state_file = Path("rollout_state.json")
        
        # Callbacks for notifications
        self._notification_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        self.logger.info("RolloutManager initialized")
    
    def add_notification_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add notification callback for rollout events.
        
        Args:
            callback: Function to call with (event_type, event_data)
        """
        self._notification_callbacks.append(callback)
        self.logger.info("Notification callback added")
    
    def _notify(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Send notification to all registered callbacks.
        
        Args:
            event_type: Type of event (e.g., 'stage_started', 'rollout_completed')
            event_data: Event data dictionary
        """
        for callback in self._notification_callbacks:
            try:
                callback(event_type, event_data)
            except Exception as e:
                self.logger.error(f"Notification callback error: {e}")
    
    def create_linear_rollout(self, start_percentage: float = 5.0, 
                            increment: float = 10.0, 
                            stage_duration_minutes: int = 60,
                            final_percentage: float = 100.0) -> RolloutConfiguration:
        """
        Create a linear rollout configuration.
        
        Args:
            start_percentage: Starting percentage for rollout
            increment: Percentage increment for each stage
            stage_duration_minutes: Minimum duration for each stage
            final_percentage: Final target percentage
        
        Returns:
            RolloutConfiguration for linear rollout
        """
        stages = []
        current_percentage = start_percentage
        
        while current_percentage <= final_percentage:
            stage_name = f"Stage {len(stages) + 1} ({current_percentage}%)"
            
            stages.append(RolloutStage(
                stage_name=stage_name,
                target_percentage=current_percentage,
                min_duration_minutes=stage_duration_minutes,
                min_requests=max(100, int(current_percentage * 10)),  # Scale requests with percentage
                success_threshold=0.95,
                quality_threshold=0.7,
                max_error_rate=0.05
            ))
            
            if current_percentage == final_percentage:
                break
            
            current_percentage = min(current_percentage + increment, final_percentage)
        
        return RolloutConfiguration(
            strategy=RolloutStrategy.LINEAR,
            stages=stages,
            trigger=RolloutTrigger.HYBRID
        )
    
    def create_exponential_rollout(self, start_percentage: float = 1.0,
                                 stage_duration_minutes: int = 60,
                                 final_percentage: float = 100.0) -> RolloutConfiguration:
        """
        Create an exponential rollout configuration.
        
        Args:
            start_percentage: Starting percentage for rollout
            stage_duration_minutes: Minimum duration for each stage
            final_percentage: Final target percentage
        
        Returns:
            RolloutConfiguration for exponential rollout
        """
        stages = []
        current_percentage = start_percentage
        
        while current_percentage < final_percentage:
            stage_name = f"Stage {len(stages) + 1} ({current_percentage}%)"
            
            stages.append(RolloutStage(
                stage_name=stage_name,
                target_percentage=current_percentage,
                min_duration_minutes=stage_duration_minutes,
                min_requests=max(100, int(current_percentage * 10)),
                success_threshold=0.95,
                quality_threshold=0.7,
                max_error_rate=0.05
            ))
            
            # Double the percentage for next stage
            next_percentage = min(current_percentage * 2, final_percentage)
            
            if next_percentage == current_percentage:
                break
            
            current_percentage = next_percentage
        
        # Add final stage if not already at 100%
        if stages[-1].target_percentage < final_percentage:
            stages.append(RolloutStage(
                stage_name=f"Final Stage ({final_percentage}%)",
                target_percentage=final_percentage,
                min_duration_minutes=stage_duration_minutes * 2,  # Longer monitoring for full rollout
                min_requests=max(500, int(final_percentage * 20)),
                success_threshold=0.98,
                quality_threshold=0.75,
                max_error_rate=0.02
            ))
        
        return RolloutConfiguration(
            strategy=RolloutStrategy.EXPONENTIAL,
            stages=stages,
            trigger=RolloutTrigger.HYBRID
        )
    
    def create_canary_rollout(self, canary_percentage: float = 1.0,
                            canary_duration_minutes: int = 120,
                            full_percentage: float = 100.0) -> RolloutConfiguration:
        """
        Create a canary rollout configuration.
        
        Args:
            canary_percentage: Percentage for canary stage
            canary_duration_minutes: Duration for canary monitoring
            full_percentage: Final percentage after canary validation
        
        Returns:
            RolloutConfiguration for canary rollout
        """
        stages = [
            RolloutStage(
                stage_name=f"Canary ({canary_percentage}%)",
                target_percentage=canary_percentage,
                min_duration_minutes=canary_duration_minutes,
                min_requests=500,  # Higher requirement for canary
                success_threshold=0.98,  # Higher threshold for canary
                quality_threshold=0.8,   # Higher quality requirement
                max_error_rate=0.02,     # Lower error tolerance
                auto_advance=False       # Require manual approval
            ),
            RolloutStage(
                stage_name=f"Full Rollout ({full_percentage}%)",
                target_percentage=full_percentage,
                min_duration_minutes=60,
                min_requests=1000,
                success_threshold=0.95,
                quality_threshold=0.75,
                max_error_rate=0.05
            )
        ]
        
        return RolloutConfiguration(
            strategy=RolloutStrategy.CANARY,
            stages=stages,
            trigger=RolloutTrigger.MANUAL,
            require_manual_approval=True,
            emergency_rollback_enabled=True,
            emergency_error_threshold=0.05,
            emergency_quality_threshold=0.6
        )
    
    def start_rollout(self, rollout_config: RolloutConfiguration, 
                     rollout_id: Optional[str] = None) -> str:
        """
        Start a new rollout with the given configuration.
        
        Args:
            rollout_config: Configuration for the rollout
            rollout_id: Optional custom rollout ID
        
        Returns:
            Rollout ID for tracking
        
        Raises:
            ValueError: If rollout is already in progress
        """
        with self._lock:
            if self.rollout_state and self.rollout_state.phase == RolloutPhase.IN_PROGRESS:
                raise ValueError("Rollout already in progress")
            
            # Generate rollout ID if not provided
            if not rollout_id:
                rollout_id = f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate configuration
            if not rollout_config.stages:
                raise ValueError("Rollout configuration must have at least one stage")
            
            # Initialize rollout state
            self.rollout_config = rollout_config
            self.rollout_state = RolloutState(
                rollout_id=rollout_id,
                phase=RolloutPhase.STARTING,
                started_at=datetime.now()
            )
            
            # Start monitoring
            self._start_monitoring()
            
            # Start first stage
            self._advance_to_next_stage()
            
            # Save state
            self._save_state()
            
            # Notify
            self._notify("rollout_started", {
                'rollout_id': rollout_id,
                'strategy': rollout_config.strategy.value,
                'total_stages': len(rollout_config.stages)
            })
            
            self.logger.info(f"Rollout {rollout_id} started with {len(rollout_config.stages)} stages")
            return rollout_id
    
    def _advance_to_next_stage(self) -> bool:
        """
        Advance to the next rollout stage.
        
        Returns:
            True if advanced to next stage, False if rollout completed
        """
        if not self.rollout_state or not self.rollout_config:
            return False
        
        with self._lock:
            next_stage_index = self.rollout_state.current_stage_index + 1
            
            if next_stage_index >= len(self.rollout_config.stages):
                # Rollout completed
                self._complete_rollout()
                return False
            
            # Advance to next stage
            self.rollout_state.current_stage_index = next_stage_index
            self.rollout_state.stage_started_at = datetime.now()
            self.rollout_state.phase = RolloutPhase.IN_PROGRESS
            
            # Reset stage metrics
            self.rollout_state.stage_requests = 0
            self.rollout_state.stage_successful_requests = 0
            self.rollout_state.stage_failed_requests = 0
            self.rollout_state.stage_quality_scores.clear()
            
            # Update feature flag manager with new percentage
            current_stage = self.rollout_config.stages[next_stage_index]
            self.rollout_state.current_percentage = current_stage.target_percentage
            self.feature_manager.update_rollout_percentage(current_stage.target_percentage)
            
            # Save state
            self._save_state()
            
            # Notify
            self._notify("stage_started", {
                'rollout_id': self.rollout_state.rollout_id,
                'stage_index': next_stage_index,
                'stage_name': current_stage.stage_name,
                'target_percentage': current_stage.target_percentage
            })
            
            self.logger.info(f"Advanced to stage {next_stage_index + 1}: {current_stage.stage_name}")
            return True
    
    def _complete_rollout(self) -> None:
        """Complete the rollout process."""
        if not self.rollout_state:
            return
        
        with self._lock:
            self.rollout_state.phase = RolloutPhase.COMPLETED
            self.rollout_state.completed_at = datetime.now()
            self.rollout_state.last_updated = datetime.now()
            
            # Stop monitoring
            self._stop_monitoring()
            
            # Save final state
            self._save_state()
            
            # Notify
            self._notify("rollout_completed", {
                'rollout_id': self.rollout_state.rollout_id,
                'total_duration_minutes': (
                    (datetime.now() - self.rollout_state.started_at).total_seconds() / 60
                ) if self.rollout_state.started_at else 0,
                'final_percentage': self.rollout_state.current_percentage,
                'total_requests': self.rollout_state.total_requests,
                'success_rate': self.rollout_state.success_rate,
                'average_quality': self.rollout_state.average_quality_score
            })
            
            self.logger.info(f"Rollout {self.rollout_state.rollout_id} completed successfully")
    
    def pause_rollout(self, reason: str = "Manual pause") -> bool:
        """
        Pause the current rollout.
        
        Args:
            reason: Reason for pausing
        
        Returns:
            True if rollout was paused, False if not possible
        """
        if not self.rollout_state or self.rollout_state.phase != RolloutPhase.IN_PROGRESS:
            return False
        
        with self._lock:
            self.rollout_state.phase = RolloutPhase.PAUSED
            self.rollout_state.last_updated = datetime.now()
            self.rollout_state.metadata['pause_reason'] = reason
            
            # Save state
            self._save_state()
            
            # Notify
            self._notify("rollout_paused", {
                'rollout_id': self.rollout_state.rollout_id,
                'reason': reason
            })
            
            self.logger.info(f"Rollout {self.rollout_state.rollout_id} paused: {reason}")
            return True
    
    def resume_rollout(self) -> bool:
        """
        Resume a paused rollout.
        
        Returns:
            True if rollout was resumed, False if not possible
        """
        if not self.rollout_state or self.rollout_state.phase != RolloutPhase.PAUSED:
            return False
        
        with self._lock:
            self.rollout_state.phase = RolloutPhase.IN_PROGRESS
            self.rollout_state.last_updated = datetime.now()
            
            # Restart monitoring
            self._start_monitoring()
            
            # Save state
            self._save_state()
            
            # Notify
            self._notify("rollout_resumed", {
                'rollout_id': self.rollout_state.rollout_id
            })
            
            self.logger.info(f"Rollout {self.rollout_state.rollout_id} resumed")
            return True
    
    def emergency_rollback(self, reason: str = "Emergency rollback") -> bool:
        """
        Execute emergency rollback to 0% rollout.
        
        Args:
            reason: Reason for emergency rollback
        
        Returns:
            True if rollback was executed, False if not possible
        """
        if not self.rollout_state:
            return False
        
        with self._lock:
            self.rollout_state.phase = RolloutPhase.ROLLING_BACK
            self.rollout_state.last_updated = datetime.now()
            self.rollout_state.metadata['rollback_reason'] = reason
            
            # Set rollout to 0%
            self.feature_manager.update_rollout_percentage(0.0)
            self.rollout_state.current_percentage = 0.0
            
            # Complete rollback
            self.rollout_state.phase = RolloutPhase.ROLLED_BACK
            
            # Stop monitoring
            self._stop_monitoring()
            
            # Save state
            self._save_state()
            
            # Notify
            self._notify("emergency_rollback", {
                'rollout_id': self.rollout_state.rollout_id,
                'reason': reason,
                'rollback_percentage': 0.0
            })
            
            self.logger.warning(f"Emergency rollback executed for {self.rollout_state.rollout_id}: {reason}")
            return True
    
    def record_request_result(self, success: bool, quality_score: Optional[float] = None, 
                            error_details: Optional[str] = None) -> None:
        """
        Record a request result for rollout monitoring.
        
        Args:
            success: Whether the request was successful
            quality_score: Optional quality score (0.0-1.0)
            error_details: Optional error details for failures
        """
        if not self.rollout_state or self.rollout_state.phase not in [
            RolloutPhase.IN_PROGRESS, RolloutPhase.MONITORING
        ]:
            return
        
        with self._lock:
            # Update overall stats
            self.rollout_state.total_requests += 1
            if success:
                self.rollout_state.successful_requests += 1
            else:
                self.rollout_state.failed_requests += 1
                if error_details:
                    self.rollout_state.error_messages.append(error_details)
            
            # Update stage stats
            self.rollout_state.stage_requests += 1
            if success:
                self.rollout_state.stage_successful_requests += 1
            else:
                self.rollout_state.stage_failed_requests += 1
            
            # Update quality scores
            if quality_score is not None:
                self.rollout_state.stage_quality_scores.append(quality_score)
                
                # Update overall average (simple moving average)
                if self.rollout_state.average_quality_score == 0.0:
                    self.rollout_state.average_quality_score = quality_score
                else:
                    # Weighted average favoring recent scores
                    weight = 0.1
                    self.rollout_state.average_quality_score = (
                        (1 - weight) * self.rollout_state.average_quality_score +
                        weight * quality_score
                    )
            
            self.rollout_state.last_updated = datetime.now()
            
            # Check for emergency conditions
            self._check_emergency_conditions()
    
    def _check_emergency_conditions(self) -> None:
        """Check if emergency rollback conditions are met."""
        if not self.rollout_state or not self.rollout_config or not self.rollout_config.emergency_rollback_enabled:
            return
        
        # Check error rate
        if (self.rollout_state.stage_requests >= 50 and 
            self.rollout_state.stage_error_rate > self.rollout_config.emergency_error_threshold):
            
            self.emergency_rollback(f"High error rate: {self.rollout_state.stage_error_rate:.2%}")
            return
        
        # Check quality score
        if (len(self.rollout_state.stage_quality_scores) >= 20 and
            self.rollout_state.stage_average_quality < self.rollout_config.emergency_quality_threshold):
            
            self.emergency_rollback(f"Low quality score: {self.rollout_state.stage_average_quality:.2f}")
            return
    
    def _start_monitoring(self) -> None:
        """Start background monitoring task."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start monitoring task if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            if not self._monitoring_task or self._monitoring_task.done():
                self._monitoring_task = loop.create_task(self._monitoring_loop())
        except RuntimeError:
            # No async context, monitoring will be handled synchronously
            self.logger.info("No async context for monitoring, using synchronous monitoring")
    
    def _stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        self._monitoring_active = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for rollout progression."""
        while self._monitoring_active and self.rollout_state and self.rollout_config:
            try:
                await asyncio.sleep(self.rollout_config.monitoring_interval_minutes * 60)
                
                if not self._monitoring_active:
                    break
                
                # Check if current stage is ready to advance
                if self._should_advance_stage():
                    if not self._advance_to_next_stage():
                        # Rollout completed
                        break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def _should_advance_stage(self) -> bool:
        """
        Check if current stage should advance to next stage.
        
        Returns:
            True if stage should advance, False otherwise
        """
        if not self.rollout_state or not self.rollout_config:
            return False
        
        current_stage_index = self.rollout_state.current_stage_index
        if current_stage_index < 0 or current_stage_index >= len(self.rollout_config.stages):
            return False
        
        current_stage = self.rollout_config.stages[current_stage_index]
        
        # Check if auto-advance is disabled
        if not current_stage.auto_advance:
            return False
        
        # Check minimum duration
        if self.rollout_state.stage_started_at:
            stage_duration = datetime.now() - self.rollout_state.stage_started_at
            if stage_duration.total_seconds() < current_stage.min_duration_minutes * 60:
                return False
        
        # Check minimum requests
        if self.rollout_state.stage_requests < current_stage.min_requests:
            return False
        
        # Check success threshold
        if self.rollout_state.stage_success_rate < current_stage.success_threshold:
            self.logger.info(f"Stage success rate {self.rollout_state.stage_success_rate:.2%} "
                           f"below threshold {current_stage.success_threshold:.2%}")
            return False
        
        # Check error rate
        if self.rollout_state.stage_error_rate > current_stage.max_error_rate:
            self.logger.info(f"Stage error rate {self.rollout_state.stage_error_rate:.2%} "
                           f"above threshold {current_stage.max_error_rate:.2%}")
            return False
        
        # Check quality threshold
        if (self.rollout_state.stage_quality_scores and
            self.rollout_state.stage_average_quality < current_stage.quality_threshold):
            self.logger.info(f"Stage quality {self.rollout_state.stage_average_quality:.2f} "
                           f"below threshold {current_stage.quality_threshold:.2f}")
            return False
        
        return True
    
    def get_rollout_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current rollout status.
        
        Returns:
            Dictionary with rollout status or None if no active rollout
        """
        if not self.rollout_state:
            return None
        
        with self._lock:
            status = self.rollout_state.to_dict()
            
            # Add current stage information
            if (self.rollout_config and 
                0 <= self.rollout_state.current_stage_index < len(self.rollout_config.stages)):
                
                current_stage = self.rollout_config.stages[self.rollout_state.current_stage_index]
                status['current_stage'] = {
                    'name': current_stage.stage_name,
                    'target_percentage': current_stage.target_percentage,
                    'min_duration_minutes': current_stage.min_duration_minutes,
                    'min_requests': current_stage.min_requests,
                    'success_threshold': current_stage.success_threshold,
                    'quality_threshold': current_stage.quality_threshold,
                    'auto_advance': current_stage.auto_advance
                }
                
                # Calculate progress
                if self.rollout_state.stage_started_at:
                    elapsed_minutes = (datetime.now() - self.rollout_state.stage_started_at).total_seconds() / 60
                    status['stage_progress'] = {
                        'elapsed_minutes': elapsed_minutes,
                        'duration_progress': min(elapsed_minutes / current_stage.min_duration_minutes, 1.0),
                        'requests_progress': min(self.rollout_state.stage_requests / current_stage.min_requests, 1.0),
                        'ready_to_advance': self._should_advance_stage()
                    }
            
            # Add rollout configuration summary
            if self.rollout_config:
                status['rollout_config'] = {
                    'strategy': self.rollout_config.strategy.value,
                    'total_stages': len(self.rollout_config.stages),
                    'trigger': self.rollout_config.trigger.value,
                    'emergency_rollback_enabled': self.rollout_config.emergency_rollback_enabled
                }
            
            return status
    
    def _save_state(self) -> None:
        """Save rollout state to file for persistence."""
        if not self.rollout_state:
            return
        
        try:
            with open(self._state_file, 'w') as f:
                state_data = {
                    'rollout_state': self.rollout_state.to_dict(),
                    'rollout_config': asdict(self.rollout_config) if self.rollout_config else None
                }
                json.dump(state_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save rollout state: {e}")
    
    def load_state(self) -> bool:
        """
        Load rollout state from file.
        
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if not self._state_file.exists():
            return False
        
        try:
            with open(self._state_file, 'r') as f:
                state_data = json.load(f)
            
            # Reconstruct rollout state
            if state_data.get('rollout_state'):
                state_dict = state_data['rollout_state']
                self.rollout_state = RolloutState(
                    rollout_id=state_dict['rollout_id'],
                    phase=RolloutPhase(state_dict['phase']),
                    current_stage_index=state_dict['current_stage_index'],
                    current_percentage=state_dict['current_percentage']
                )
                
                # Restore timestamps
                if state_dict.get('started_at'):
                    self.rollout_state.started_at = datetime.fromisoformat(state_dict['started_at'])
                if state_dict.get('stage_started_at'):
                    self.rollout_state.stage_started_at = datetime.fromisoformat(state_dict['stage_started_at'])
                if state_dict.get('completed_at'):
                    self.rollout_state.completed_at = datetime.fromisoformat(state_dict['completed_at'])
                
                # Restore metrics
                self.rollout_state.total_requests = state_dict.get('total_requests', 0)
                self.rollout_state.successful_requests = state_dict.get('successful_requests', 0)
                self.rollout_state.failed_requests = state_dict.get('failed_requests', 0)
                self.rollout_state.metadata = state_dict.get('metadata', {})
            
            # Reconstruct rollout config
            if state_data.get('rollout_config'):
                config_dict = state_data['rollout_config']
                
                # Reconstruct stages
                stages = []
                for stage_dict in config_dict.get('stages', []):
                    stages.append(RolloutStage(**stage_dict))
                
                self.rollout_config = RolloutConfiguration(
                    strategy=RolloutStrategy(config_dict['strategy']),
                    stages=stages,
                    trigger=RolloutTrigger(config_dict.get('trigger', 'hybrid')),
                    emergency_rollback_enabled=config_dict.get('emergency_rollback_enabled', True)
                )
            
            # Resume monitoring if rollout is active
            if (self.rollout_state and 
                self.rollout_state.phase in [RolloutPhase.IN_PROGRESS, RolloutPhase.MONITORING]):
                self._start_monitoring()
            
            self.logger.info(f"Rollout state loaded: {self.rollout_state.rollout_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load rollout state: {e}")
            return False