#!/usr/bin/env python3
"""
Routing Decision Analytics System for IntelligentQueryRouter

This module provides comprehensive logging and analytics for routing decisions,
including structured logging, performance metrics collection, and decision analysis.

Key Features:
- Structured JSON logging of all routing decisions
- Real-time analytics collection and aggregation
- Performance monitoring and trend analysis
- Configurable logging levels and storage options
- Integration with existing IntelligentQueryRouter
- Log rotation and archival management
- Decision accuracy tracking and optimization insights

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Routing Decision Logging and Analytics System Design
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque, Counter
from enum import Enum
from pathlib import Path
import statistics
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import asynccontextmanager
import os
import gzip
import shutil
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class LogLevel(Enum):
    """Logging verbosity levels"""
    MINIMAL = "minimal"      # Only log final decisions
    STANDARD = "standard"    # Log decisions with basic metrics
    DETAILED = "detailed"    # Log full decision process
    DEBUG = "debug"         # Log everything including internal state


class StorageStrategy(Enum):
    """Storage strategies for analytics data"""
    FILE_ONLY = "file_only"        # Only file-based storage
    MEMORY_ONLY = "memory_only"    # Only in-memory storage
    HYBRID = "hybrid"              # Both file and memory
    DISTRIBUTED = "distributed"   # Multiple storage backends


class RoutingMetricType(Enum):
    """Types of routing metrics to track"""
    DECISION_TIME = "decision_time"
    CONFIDENCE_SCORE = "confidence_score"
    BACKEND_SELECTION = "backend_selection"
    ERROR_RATE = "error_rate"
    ACCURACY_SCORE = "accuracy_score"
    LOAD_BALANCE_RATIO = "load_balance_ratio"


@dataclass
class LoggingConfig:
    """Configuration for routing decision logging"""
    enabled: bool = True
    log_level: LogLevel = LogLevel.STANDARD
    storage_strategy: StorageStrategy = StorageStrategy.HYBRID
    
    # File logging configuration
    log_directory: str = "logs/routing_decisions"
    log_filename_pattern: str = "routing_decisions_{date}.jsonl"
    max_file_size_mb: int = 100
    max_files_to_keep: int = 30
    compress_old_logs: bool = True
    
    # Memory storage configuration
    max_memory_entries: int = 10000
    memory_retention_hours: int = 24
    
    # Performance configuration
    async_logging: bool = True
    batch_size: int = 100
    flush_interval_seconds: int = 30
    
    # Privacy and security
    anonymize_queries: bool = False
    hash_sensitive_data: bool = True
    exclude_fields: List[str] = field(default_factory=lambda: [])
    
    # Analytics configuration
    enable_real_time_analytics: bool = True
    analytics_aggregation_interval_minutes: int = 5
    enable_performance_alerts: bool = True
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create configuration from environment variables"""
        return cls(
            enabled=os.getenv('ROUTING_LOGGING_ENABLED', 'true').lower() == 'true',
            log_level=LogLevel(os.getenv('ROUTING_LOG_LEVEL', 'standard')),
            storage_strategy=StorageStrategy(os.getenv('ROUTING_STORAGE_STRATEGY', 'hybrid')),
            log_directory=os.getenv('ROUTING_LOG_DIR', 'logs/routing_decisions'),
            max_file_size_mb=int(os.getenv('ROUTING_MAX_FILE_SIZE_MB', '100')),
            max_files_to_keep=int(os.getenv('ROUTING_MAX_FILES', '30')),
            compress_old_logs=os.getenv('ROUTING_COMPRESS_LOGS', 'true').lower() == 'true',
            max_memory_entries=int(os.getenv('ROUTING_MAX_MEMORY_ENTRIES', '10000')),
            memory_retention_hours=int(os.getenv('ROUTING_MEMORY_RETENTION_HOURS', '24')),
            async_logging=os.getenv('ROUTING_ASYNC_LOGGING', 'true').lower() == 'true',
            anonymize_queries=os.getenv('ROUTING_ANONYMIZE_QUERIES', 'false').lower() == 'true',
            enable_real_time_analytics=os.getenv('ROUTING_REAL_TIME_ANALYTICS', 'true').lower() == 'true'
        )


@dataclass
class RoutingDecisionLogEntry:
    """Structured log entry for routing decisions"""
    
    # Basic identifiers
    entry_id: str
    timestamp: datetime
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Query information
    query_text: Optional[str] = None  # May be anonymized/hashed
    query_hash: str = ""
    query_length: int = 0
    query_complexity_score: float = 0.0
    query_context: Dict[str, Any] = field(default_factory=dict)
    
    # Routing decision
    routing_decision: str = ""
    confidence_score: float = 0.0
    confidence_level: str = ""
    decision_reasoning: List[str] = field(default_factory=list)
    alternative_routes: List[Tuple[str, float]] = field(default_factory=list)
    
    # Performance metrics
    decision_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    backend_selection_time_ms: float = 0.0
    
    # System state during decision
    backend_health_status: Dict[str, str] = field(default_factory=dict)
    backend_load_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    system_resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Load balancing information
    selected_backend: Optional[str] = None
    backend_selection_algorithm: Optional[str] = None
    load_balancer_metrics: Dict[str, Any] = field(default_factory=dict)
    backend_weights: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics (if available)
    accuracy_score: Optional[float] = None
    user_satisfaction_score: Optional[float] = None
    response_quality_score: Optional[float] = None
    
    # Error information
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    
    # Metadata
    router_version: str = "1.0.0"
    environment: str = "production"
    deployment_mode: Optional[str] = None
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime to ISO string
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_routing_prediction(cls, 
                               prediction,  # RoutingPrediction
                               query_text: str,
                               processing_metrics: Dict[str, float],
                               system_state: Dict[str, Any],
                               config: LoggingConfig) -> 'RoutingDecisionLogEntry':
        """Create log entry from RoutingPrediction and system state"""
        
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Handle query anonymization/hashing
        processed_query = query_text
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()[:16]
        
        if config.anonymize_queries:
            processed_query = f"<anonymized:{len(query_text)}>"
        elif config.hash_sensitive_data:
            # Keep first/last few words, hash middle
            words = query_text.split()
            if len(words) > 6:
                processed_query = f"{' '.join(words[:2])} ... <hashed> ... {' '.join(words[-2:])}"
        
        return cls(
            entry_id=entry_id,
            timestamp=timestamp,
            query_text=processed_query,
            query_hash=query_hash,
            query_length=len(query_text),
            query_complexity_score=processing_metrics.get('query_complexity', 0.0),
            routing_decision=prediction.routing_decision.value,
            confidence_score=prediction.confidence,
            confidence_level=prediction.confidence_level,
            decision_reasoning=prediction.reasoning if hasattr(prediction, 'reasoning') else [],
            alternative_routes=[(r.value, c) for r, c in prediction.get_alternative_routes()] if hasattr(prediction, 'get_alternative_routes') else [],
            decision_time_ms=processing_metrics.get('decision_time_ms', 0.0),
            total_processing_time_ms=processing_metrics.get('total_time_ms', 0.0),
            backend_selection_time_ms=processing_metrics.get('backend_selection_time_ms', 0.0),
            backend_health_status=system_state.get('backend_health', {}),
            backend_load_metrics=system_state.get('backend_load', {}),
            system_resource_usage=system_state.get('resource_usage', {}),
            selected_backend=getattr(prediction, 'backend_selected', None),
            backend_selection_algorithm=system_state.get('selection_algorithm'),
            load_balancer_metrics=getattr(prediction, 'load_balancer_metrics', {}),
            backend_weights=system_state.get('backend_weights', {}),
            errors_encountered=system_state.get('errors', []),
            warnings=system_state.get('warnings', []),
            fallback_used=system_state.get('fallback_used', False),
            fallback_reason=system_state.get('fallback_reason'),
            deployment_mode=system_state.get('deployment_mode'),
            feature_flags=system_state.get('feature_flags', {})
        )


@dataclass
class AnalyticsMetrics:
    """Aggregated analytics metrics for routing decisions"""
    
    # Time period
    start_time: datetime
    end_time: datetime
    total_requests: int = 0
    
    # Decision distribution
    decision_distribution: Dict[str, int] = field(default_factory=dict)
    decision_percentages: Dict[str, float] = field(default_factory=dict)
    
    # Confidence metrics
    avg_confidence_score: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    low_confidence_requests: int = 0
    
    # Performance metrics
    avg_decision_time_ms: float = 0.0
    p95_decision_time_ms: float = 0.0
    p99_decision_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    
    # Backend utilization
    backend_utilization: Dict[str, float] = field(default_factory=dict)
    backend_response_times: Dict[str, float] = field(default_factory=dict)
    backend_error_rates: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    avg_accuracy_score: Optional[float] = None
    user_satisfaction_rate: Optional[float] = None
    fallback_rate: float = 0.0
    error_rate: float = 0.0
    
    # Trends
    decision_time_trend: str = "stable"  # improving, stable, degrading
    confidence_trend: str = "stable"
    error_rate_trend: str = "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class RoutingDecisionLogger:
    """
    Comprehensive logging system for routing decisions with multiple storage backends
    """
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RoutingDecisionLogger")
        
        # Initialize storage backends
        self._init_file_logging()
        self._init_memory_storage()
        
        # Async logging queue
        self._log_queue = asyncio.Queue() if config.async_logging else None
        self._log_worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Batch processing
        self._batch_buffer: List[RoutingDecisionLogEntry] = []
        self._last_flush = time.time()
        self._batch_lock = threading.Lock()
        
        self.logger.info(f"RoutingDecisionLogger initialized with config: {config.log_level}, {config.storage_strategy}")
    
    def _init_file_logging(self):
        """Initialize file-based logging"""
        if self.config.storage_strategy in [StorageStrategy.FILE_ONLY, StorageStrategy.HYBRID]:
            # Create log directory
            Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
            
            # Set up rotating file handler
            log_file = Path(self.config.log_directory) / self.config.log_filename_pattern.format(
                date=datetime.now().strftime("%Y%m%d")
            )
            
            # Configure file handler with rotation
            self.file_handler = RotatingFileHandler(
                str(log_file),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.max_files_to_keep
            )
            
            # Set up formatter for structured JSON logs
            formatter = logging.Formatter('%(message)s')
            self.file_handler.setFormatter(formatter)
            
            # Create dedicated logger for file output
            self.file_logger = logging.getLogger(f"{__name__}.FileLogger")
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.addHandler(self.file_handler)
            self.file_logger.propagate = False
    
    def _init_memory_storage(self):
        """Initialize in-memory storage"""
        if self.config.storage_strategy in [StorageStrategy.MEMORY_ONLY, StorageStrategy.HYBRID]:
            self.memory_storage = deque(maxlen=self.config.max_memory_entries)
            self._memory_lock = threading.Lock()
            
            # Start cleanup task for memory retention
            if self.config.memory_retention_hours > 0:
                self._start_memory_cleanup_task()
    
    def _start_memory_cleanup_task(self):
        """Start background task for memory cleanup"""
        def cleanup_memory():
            cutoff_time = datetime.now() - timedelta(hours=self.config.memory_retention_hours)
            with self._memory_lock:
                # Remove entries older than retention period
                # Note: deque doesn't support efficient filtering, so we rebuild it
                self.memory_storage = deque(
                    [entry for entry in self.memory_storage if entry.timestamp > cutoff_time],
                    maxlen=self.config.max_memory_entries
                )
        
        # Schedule cleanup every hour
        threading.Timer(3600, cleanup_memory).start()
    
    async def start_async_logging(self):
        """Start async logging worker"""
        if self.config.async_logging and self._log_worker_task is None:
            self._log_worker_task = asyncio.create_task(self._async_log_worker())
    
    async def stop_async_logging(self):
        """Stop async logging worker and flush remaining logs"""
        if self._log_worker_task:
            self._shutdown_event.set()
            await self._log_worker_task
            self._log_worker_task = None
        
        # Flush any remaining batched logs
        await self._flush_batch()
    
    async def _async_log_worker(self):
        """Async worker for processing log entries"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for log entry or timeout
                    entry = await asyncio.wait_for(self._log_queue.get(), timeout=1.0)
                    await self._process_log_entry(entry)
                except asyncio.TimeoutError:
                    # Check if we need to flush batch
                    if (time.time() - self._last_flush) > self.config.flush_interval_seconds:
                        await self._flush_batch()
                except Exception as e:
                    self.logger.error(f"Error in async log worker: {e}")
        except asyncio.CancelledError:
            # Flush remaining logs on cancellation
            await self._flush_batch()
            raise
    
    async def log_routing_decision(self, 
                                 prediction,  # RoutingPrediction
                                 query_text: str,
                                 processing_metrics: Dict[str, float],
                                 system_state: Dict[str, Any]):
        """Log a routing decision"""
        if not self.config.enabled:
            return
        
        # Create log entry
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            prediction, query_text, processing_metrics, system_state, self.config
        )
        
        # Apply log level filtering
        if not self._should_log_entry(log_entry):
            return
        
        # Process log entry
        if self.config.async_logging:
            await self._log_queue.put(log_entry)
        else:
            await self._process_log_entry(log_entry)
    
    def _should_log_entry(self, entry: RoutingDecisionLogEntry) -> bool:
        """Determine if entry should be logged based on log level"""
        if self.config.log_level == LogLevel.MINIMAL:
            # Only log final decisions, no detailed metrics
            return True
        elif self.config.log_level == LogLevel.STANDARD:
            # Log decisions with basic metrics
            return True
        elif self.config.log_level == LogLevel.DETAILED:
            # Log full decision process
            return True
        else:  # DEBUG
            # Log everything
            return True
    
    async def _process_log_entry(self, entry: RoutingDecisionLogEntry):
        """Process a single log entry"""
        try:
            # Apply field exclusions
            if self.config.exclude_fields:
                entry_dict = entry.to_dict()
                for field in self.config.exclude_fields:
                    entry_dict.pop(field, None)
            else:
                entry_dict = entry.to_dict()
            
            # Store in file if configured
            if self.config.storage_strategy in [StorageStrategy.FILE_ONLY, StorageStrategy.HYBRID]:
                await self._store_to_file(entry_dict)
            
            # Store in memory if configured
            if self.config.storage_strategy in [StorageStrategy.MEMORY_ONLY, StorageStrategy.HYBRID]:
                await self._store_to_memory(entry)
            
        except Exception as e:
            self.logger.error(f"Error processing log entry: {e}")
    
    async def _store_to_file(self, entry_dict: Dict[str, Any]):
        """Store entry to file"""
        try:
            if self.config.async_logging:
                # Add to batch
                with self._batch_lock:
                    self._batch_buffer.append(entry_dict)
                    if len(self._batch_buffer) >= self.config.batch_size:
                        await self._flush_batch()
            else:
                # Write immediately
                self.file_logger.info(json.dumps(entry_dict, default=str))
        except Exception as e:
            self.logger.error(f"Error storing to file: {e}")
    
    async def _store_to_memory(self, entry: RoutingDecisionLogEntry):
        """Store entry to memory"""
        try:
            with self._memory_lock:
                self.memory_storage.append(entry)
        except Exception as e:
            self.logger.error(f"Error storing to memory: {e}")
    
    async def _flush_batch(self):
        """Flush batched log entries to file"""
        if not self._batch_buffer:
            return
        
        try:
            with self._batch_lock:
                batch = self._batch_buffer.copy()
                self._batch_buffer.clear()
                self._last_flush = time.time()
            
            # Write batch to file
            for entry in batch:
                self.file_logger.info(json.dumps(entry, default=str))
            
        except Exception as e:
            self.logger.error(f"Error flushing batch: {e}")
    
    def get_recent_entries(self, limit: int = 100) -> List[RoutingDecisionLogEntry]:
        """Get recent log entries from memory storage"""
        if self.config.storage_strategy not in [StorageStrategy.MEMORY_ONLY, StorageStrategy.HYBRID]:
            return []
        
        with self._memory_lock:
            # Return most recent entries
            return list(self.memory_storage)[-limit:]
    
    def query_entries(self, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     routing_decision: Optional[str] = None,
                     min_confidence: Optional[float] = None,
                     limit: int = 1000) -> List[RoutingDecisionLogEntry]:
        """Query log entries with filters"""
        if self.config.storage_strategy not in [StorageStrategy.MEMORY_ONLY, StorageStrategy.HYBRID]:
            return []
        
        with self._memory_lock:
            results = []
            for entry in self.memory_storage:
                # Apply filters
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                if routing_decision and entry.routing_decision != routing_decision:
                    continue
                if min_confidence and entry.confidence_score < min_confidence:
                    continue
                
                results.append(entry)
                
                if len(results) >= limit:
                    break
            
            return results


class RoutingAnalytics:
    """
    Advanced analytics engine for routing decisions with real-time metrics
    """
    
    def __init__(self, logger: RoutingDecisionLogger):
        self.logger = logger
        self.config = logger.config
        self.analytics_logger = logging.getLogger(f"{__name__}.RoutingAnalytics")
        
        # Real-time metrics
        self._metrics_cache: Dict[str, Any] = {}
        self._metrics_lock = threading.Lock()
        self._last_aggregation = datetime.now()
        
        # Performance tracking
        self._decision_times = deque(maxlen=1000)
        self._confidence_scores = deque(maxlen=1000)
        self._backend_counters = defaultdict(int)
        self._error_counter = 0
        self._total_requests = 0
        
        # Start real-time analytics if enabled
        if self.config.enable_real_time_analytics:
            self._start_real_time_analytics()
    
    def _start_real_time_analytics(self):
        """Start background real-time analytics processing"""
        def process_analytics():
            try:
                self._aggregate_metrics()
                # Schedule next run
                threading.Timer(
                    self.config.analytics_aggregation_interval_minutes * 60,
                    process_analytics
                ).start()
            except Exception as e:
                self.analytics_logger.error(f"Error in real-time analytics: {e}")
        
        # Start first run
        threading.Timer(
            self.config.analytics_aggregation_interval_minutes * 60,
            process_analytics
        ).start()
    
    def record_decision_metrics(self, 
                              entry: RoutingDecisionLogEntry):
        """Record metrics from a routing decision"""
        with self._metrics_lock:
            self._total_requests += 1
            self._decision_times.append(entry.decision_time_ms)
            self._confidence_scores.append(entry.confidence_score)
            self._backend_counters[entry.selected_backend or 'unknown'] += 1
            
            if entry.errors_encountered:
                self._error_counter += 1
    
    def _aggregate_metrics(self):
        """Aggregate current metrics"""
        with self._metrics_lock:
            if not self._decision_times:
                return
            
            # Calculate aggregated metrics
            now = datetime.now()
            self._metrics_cache = {
                'timestamp': now.isoformat(),
                'total_requests': self._total_requests,
                'avg_decision_time_ms': statistics.mean(self._decision_times),
                'p95_decision_time_ms': statistics.quantiles(self._decision_times, n=20)[18] if len(self._decision_times) >= 20 else 0,
                'avg_confidence_score': statistics.mean(self._confidence_scores),
                'backend_distribution': dict(self._backend_counters),
                'error_rate': (self._error_counter / self._total_requests * 100) if self._total_requests > 0 else 0,
                'requests_per_minute': self._total_requests / max((now - self._last_aggregation).total_seconds() / 60, 1)
            }
            
            self._last_aggregation = now
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        with self._metrics_lock:
            return self._metrics_cache.copy()
    
    def generate_analytics_report(self,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> AnalyticsMetrics:
        """Generate comprehensive analytics report"""
        
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Query log entries for the time period
        entries = self.logger.query_entries(start_time=start_time, end_time=end_time, limit=10000)
        
        if not entries:
            return AnalyticsMetrics(start_time=start_time, end_time=end_time)
        
        # Aggregate metrics
        total_requests = len(entries)
        decision_distribution = Counter(entry.routing_decision for entry in entries)
        decision_percentages = {k: (v / total_requests * 100) for k, v in decision_distribution.items()}
        
        confidence_scores = [entry.confidence_score for entry in entries]
        decision_times = [entry.decision_time_ms for entry in entries]
        
        confidence_distribution = Counter()
        low_confidence_count = 0
        for score in confidence_scores:
            if score >= 0.8:
                confidence_distribution['high'] += 1
            elif score >= 0.6:
                confidence_distribution['medium'] += 1
            elif score >= 0.4:
                confidence_distribution['low'] += 1
            else:
                confidence_distribution['very_low'] += 1
                low_confidence_count += 1
        
        # Backend utilization
        backend_counts = Counter(entry.selected_backend for entry in entries if entry.selected_backend)
        backend_utilization = {k: (v / total_requests * 100) for k, v in backend_counts.items()}
        
        # Error metrics
        error_count = sum(1 for entry in entries if entry.errors_encountered)
        fallback_count = sum(1 for entry in entries if entry.fallback_used)
        
        return AnalyticsMetrics(
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            decision_distribution=dict(decision_distribution),
            decision_percentages=decision_percentages,
            avg_confidence_score=statistics.mean(confidence_scores) if confidence_scores else 0,
            confidence_distribution=dict(confidence_distribution),
            low_confidence_requests=low_confidence_count,
            avg_decision_time_ms=statistics.mean(decision_times) if decision_times else 0,
            p95_decision_time_ms=statistics.quantiles(decision_times, n=20)[18] if len(decision_times) >= 20 else 0,
            p99_decision_time_ms=statistics.quantiles(decision_times, n=100)[98] if len(decision_times) >= 100 else 0,
            backend_utilization=backend_utilization,
            error_rate=(error_count / total_requests * 100) if total_requests > 0 else 0,
            fallback_rate=(fallback_count / total_requests * 100) if total_requests > 0 else 0
        )
    
    def detect_anomalies(self, window_hours: int = 1) -> List[Dict[str, Any]]:
        """Detect anomalies in routing patterns"""
        anomalies = []
        
        # Get recent metrics
        recent_entries = self.logger.get_recent_entries(limit=1000)
        if len(recent_entries) < 50:  # Need sufficient data
            return anomalies
        
        # Check for confidence score anomalies
        recent_confidence = [entry.confidence_score for entry in recent_entries[-100:]]
        historical_confidence = [entry.confidence_score for entry in recent_entries[:-100]]
        
        if len(historical_confidence) > 0:
            historical_mean = statistics.mean(historical_confidence)
            recent_mean = statistics.mean(recent_confidence)
            
            # Check for significant confidence degradation
            if recent_mean < historical_mean - 0.15:  # 15% drop
                anomalies.append({
                    'type': 'confidence_degradation',
                    'severity': 'warning',
                    'description': f'Average confidence dropped from {historical_mean:.3f} to {recent_mean:.3f}',
                    'metric': 'confidence_score',
                    'threshold_breached': 0.15,
                    'current_value': recent_mean,
                    'historical_value': historical_mean
                })
        
        # Check for decision time anomalies
        recent_times = [entry.decision_time_ms for entry in recent_entries[-100:]]
        if len(recent_times) > 0:
            avg_time = statistics.mean(recent_times)
            if avg_time > 1000:  # > 1 second
                anomalies.append({
                    'type': 'slow_decisions',
                    'severity': 'warning',
                    'description': f'Average decision time increased to {avg_time:.1f}ms',
                    'metric': 'decision_time_ms',
                    'threshold_breached': 1000,
                    'current_value': avg_time
                })
        
        # Check for error rate anomalies
        recent_errors = sum(1 for entry in recent_entries[-100:] if entry.errors_encountered)
        error_rate = (recent_errors / min(100, len(recent_entries))) * 100
        
        if error_rate > 5:  # > 5% error rate
            anomalies.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'description': f'Error rate increased to {error_rate:.1f}%',
                'metric': 'error_rate',
                'threshold_breached': 5.0,
                'current_value': error_rate
            })
        
        return anomalies
    
    def export_analytics(self, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        file_path: Optional[str] = None) -> str:
        """Export analytics report to JSON file"""
        
        # Generate report
        report = self.generate_analytics_report(start_time, end_time)
        
        # Add real-time metrics
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analytics_report': report.to_dict(),
            'real_time_metrics': self.get_real_time_metrics(),
            'anomalies': self.detect_anomalies()
        }
        
        # Generate file path if not provided
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"routing_analytics_report_{timestamp}.json"
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dumps(export_data, f, indent=2, default=str)
        
        self.analytics_logger.info(f"Analytics report exported to {file_path}")
        return file_path


# Factory functions for easy integration
def create_routing_logger(config: Optional[LoggingConfig] = None) -> RoutingDecisionLogger:
    """Create routing decision logger with configuration"""
    if config is None:
        config = LoggingConfig.from_env()
    
    return RoutingDecisionLogger(config)


def create_routing_analytics(logger: RoutingDecisionLogger) -> RoutingAnalytics:
    """Create routing analytics engine"""
    return RoutingAnalytics(logger)


# Integration helper for existing router
class RoutingLoggingMixin:
    """
    Mixin class to add logging capabilities to existing routers
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize logging components
        self.routing_logging_config = LoggingConfig.from_env()
        self.routing_logger = create_routing_logger(self.routing_logging_config)
        self.routing_analytics = create_routing_analytics(self.routing_logger)
        
        # Start async logging if enabled
        if self.routing_logging_config.async_logging:
            asyncio.create_task(self.routing_logger.start_async_logging())
    
    async def _log_routing_decision_with_context(self,
                                               prediction,  # RoutingPrediction
                                               query_text: str,
                                               start_time: float,
                                               system_context: Dict[str, Any]):
        """Log routing decision with full context"""
        
        processing_time = (time.time() - start_time) * 1000
        
        # Collect processing metrics
        processing_metrics = {
            'decision_time_ms': getattr(prediction.confidence_metrics, 'calculation_time_ms', 0) if hasattr(prediction, 'confidence_metrics') else 0,
            'total_time_ms': processing_time,
            'backend_selection_time_ms': system_context.get('backend_selection_time_ms', 0),
            'query_complexity': system_context.get('query_complexity', len(query_text.split()) / 10.0)
        }
        
        # Collect system state
        system_state = {
            'backend_health': system_context.get('backend_health', {}),
            'backend_load': system_context.get('backend_load', {}),
            'resource_usage': system_context.get('resource_usage', {}),
            'selection_algorithm': system_context.get('selection_algorithm'),
            'backend_weights': system_context.get('backend_weights', {}),
            'errors': system_context.get('errors', []),
            'warnings': system_context.get('warnings', []),
            'fallback_used': system_context.get('fallback_used', False),
            'fallback_reason': system_context.get('fallback_reason'),
            'deployment_mode': getattr(self, 'deployment_mode', None) if hasattr(self, 'deployment_mode') else None,
            'feature_flags': system_context.get('feature_flags', {})
        }
        
        # Log the decision
        await self.routing_logger.log_routing_decision(
            prediction, query_text, processing_metrics, system_state
        )
        
        # Record metrics for analytics
        log_entry = RoutingDecisionLogEntry.from_routing_prediction(
            prediction, query_text, processing_metrics, system_state, self.routing_logging_config
        )
        self.routing_analytics.record_decision_metrics(log_entry)
    
    def get_routing_analytics_report(self,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get routing analytics report"""
        return self.routing_analytics.generate_analytics_report(start_time, end_time).to_dict()
    
    def export_routing_analytics(self,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> str:
        """Export routing analytics to file"""
        return self.routing_analytics.export_analytics(start_time, end_time)
    
    async def shutdown_routing_logging(self):
        """Shutdown routing logging system gracefully"""
        if self.routing_logging_config.async_logging:
            await self.routing_logger.stop_async_logging()