"""
API Usage Metrics Logging System for Clinical Metabolomics Oracle LightRAG Integration

This module provides comprehensive API usage metrics logging that integrates with
the existing logging infrastructure and enhanced cost tracking system.

Classes:
    - MetricType: Enum for different types of API metrics
    - APIMetric: Data model for individual API metrics
    - APIUsageMetricsLogger: Main metrics logging system
    - MetricsAggregator: System for aggregating and analyzing metrics

The metrics logging system supports:
    - Integration with existing LightRAG logging infrastructure
    - Detailed API usage tracking (tokens, costs, performance)
    - Research-specific metrics categorization
    - Structured logging for analytics and monitoring
    - Audit-friendly compliance logging
    - Thread-safe concurrent operations
    - Automatic metric aggregation and reporting
"""

import json
import time
import threading
import logging
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import uuid

from .cost_persistence import CostRecord, ResearchCategory, CostPersistence
from .budget_manager import BudgetManager, AlertLevel, BudgetAlert
from .research_categorizer import ResearchCategorizer
from .audit_trail import AuditTrail


class MetricType(Enum):
    """Types of API metrics tracked by the system."""
    
    # Core API operations
    LLM_CALL = "llm_call"
    EMBEDDING_CALL = "embedding_call"
    HYBRID_OPERATION = "hybrid_operation"
    
    # Performance metrics
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    
    # Cost metrics
    TOKEN_USAGE = "token_usage"
    COST_TRACKING = "cost_tracking"
    BUDGET_UTILIZATION = "budget_utilization"
    
    # Research-specific metrics
    RESEARCH_CATEGORY = "research_category"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    DOCUMENT_PROCESSING = "document_processing"
    
    # System metrics
    MEMORY_USAGE = "memory_usage"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    RETRY_PATTERNS = "retry_patterns"


@dataclass
class APIMetric:
    """
    Data model for individual API usage metrics.
    
    This comprehensive dataclass captures all relevant information
    about API usage for detailed analysis and audit purposes.
    """
    
    # Core identification
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    metric_type: MetricType = MetricType.LLM_CALL
    
    # API operation details
    operation_name: str = "unknown"
    model_name: Optional[str] = None
    api_provider: str = "openai"
    endpoint_used: Optional[str] = None
    
    # Token and cost metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_per_token: Optional[float] = None
    
    # Performance metrics
    response_time_ms: Optional[float] = None
    queue_time_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    
    # Quality and success metrics
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    final_attempt: bool = True
    
    # Research categorization
    research_category: str = ResearchCategory.GENERAL_QUERY.value
    query_type: Optional[str] = None
    subject_area: Optional[str] = None
    document_type: Optional[str] = None  # pdf, text, structured_data
    
    # System resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    concurrent_operations: int = 1
    
    # Request/response characteristics
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    context_length: Optional[int] = None
    temperature_used: Optional[float] = None
    
    # Budget and compliance
    daily_budget_used_percent: Optional[float] = None
    monthly_budget_used_percent: Optional[float] = None
    compliance_level: str = "standard"  # standard, high, critical
    
    # User and project context
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing for calculated fields."""
        # Calculate total tokens if not already set
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens + self.embedding_tokens
        
        if self.cost_per_token is None and self.total_tokens > 0:
            self.cost_per_token = self.cost_usd / self.total_tokens
        
        if self.throughput_tokens_per_sec is None and self.response_time_ms and self.total_tokens > 0:
            self.throughput_tokens_per_sec = self.total_tokens / (self.response_time_ms / 1000.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        result = asdict(self)
        result['metric_type'] = self.metric_type.value
        result['timestamp_iso'] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        return result
    
    def to_cost_record(self) -> CostRecord:
        """Convert API metric to cost record for persistence."""
        return CostRecord(
            timestamp=self.timestamp,
            session_id=self.session_id,
            operation_type=self.operation_name,
            model_name=self.model_name,
            cost_usd=self.cost_usd,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            embedding_tokens=self.embedding_tokens,
            total_tokens=self.total_tokens,
            research_category=self.research_category,
            query_type=self.query_type,
            subject_area=self.subject_area,
            response_time_seconds=self.response_time_ms / 1000.0 if self.response_time_ms else None,
            success=self.success,
            error_type=self.error_type,
            user_id=self.user_id,
            project_id=self.project_id,
            metadata=self.metadata
        )


class MetricsAggregator:
    """
    System for aggregating and analyzing API usage metrics.
    
    Provides real-time aggregation of metrics for monitoring,
    alerting, and analysis purposes.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize metrics aggregator."""
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._metrics_buffer: List[APIMetric] = []
        self._hourly_stats = defaultdict(lambda: defaultdict(float))
        self._daily_stats = defaultdict(lambda: defaultdict(float))
        self._category_stats = defaultdict(lambda: defaultdict(float))
        self._error_patterns = Counter()
        
        # Performance tracking
        self._performance_window = []  # Recent performance metrics
        self._max_window_size = 1000
    
    def add_metric(self, metric: APIMetric) -> None:
        """Add a metric to the aggregation system."""
        with self._lock:
            self._metrics_buffer.append(metric)
            self._update_aggregations(metric)
            
            # Maintain performance window
            if len(self._performance_window) >= self._max_window_size:
                self._performance_window.pop(0)
            self._performance_window.append(metric)
    
    def _update_aggregations(self, metric: APIMetric) -> None:
        """Update internal aggregations with new metric."""
        timestamp = datetime.fromtimestamp(metric.timestamp, tz=timezone.utc)
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        day_key = timestamp.strftime('%Y-%m-%d')
        
        # Update hourly statistics
        hourly = self._hourly_stats[hour_key]
        hourly['total_calls'] += 1
        hourly['total_tokens'] += metric.total_tokens
        hourly['total_cost'] += metric.cost_usd
        if metric.response_time_ms:
            hourly['total_response_time'] += metric.response_time_ms
            hourly['response_time_count'] += 1
        if not metric.success:
            hourly['error_count'] += 1
        
        # Update daily statistics
        daily = self._daily_stats[day_key]
        daily['total_calls'] += 1
        daily['total_tokens'] += metric.total_tokens
        daily['total_cost'] += metric.cost_usd
        if metric.response_time_ms:
            daily['total_response_time'] += metric.response_time_ms
            daily['response_time_count'] += 1
        if not metric.success:
            daily['error_count'] += 1
        
        # Update category statistics
        category = self._category_stats[metric.research_category]
        category['total_calls'] += 1
        category['total_tokens'] += metric.total_tokens
        category['total_cost'] += metric.cost_usd
        
        # Track error patterns
        if not metric.success and metric.error_type:
            self._error_patterns[metric.error_type] += 1
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current aggregated statistics."""
        with self._lock:
            now = datetime.now(tz=timezone.utc)
            current_hour = now.strftime('%Y-%m-%d-%H')
            current_day = now.strftime('%Y-%m-%d')
            
            # Calculate average response times
            hourly = self._hourly_stats.get(current_hour, {})
            daily = self._daily_stats.get(current_day, {})
            
            hourly_avg_response = (
                hourly.get('total_response_time', 0) / max(hourly.get('response_time_count', 1), 1)
            )
            daily_avg_response = (
                daily.get('total_response_time', 0) / max(daily.get('response_time_count', 1), 1)
            )
            
            # Recent performance metrics
            recent_metrics = self._performance_window[-100:] if self._performance_window else []
            recent_avg_response = (
                sum(m.response_time_ms for m in recent_metrics if m.response_time_ms) /
                max(len([m for m in recent_metrics if m.response_time_ms]), 1)
            )
            
            return {
                'current_hour': {
                    'total_calls': hourly.get('total_calls', 0),
                    'total_tokens': int(hourly.get('total_tokens', 0)),
                    'total_cost': round(hourly.get('total_cost', 0), 6),
                    'avg_response_time_ms': round(hourly_avg_response, 2),
                    'error_count': int(hourly.get('error_count', 0)),
                    'error_rate_percent': round(
                        hourly.get('error_count', 0) / max(hourly.get('total_calls', 1), 1) * 100, 2
                    )
                },
                'current_day': {
                    'total_calls': daily.get('total_calls', 0),
                    'total_tokens': int(daily.get('total_tokens', 0)),
                    'total_cost': round(daily.get('total_cost', 0), 6),
                    'avg_response_time_ms': round(daily_avg_response, 2),
                    'error_count': int(daily.get('error_count', 0)),
                    'error_rate_percent': round(
                        daily.get('error_count', 0) / max(daily.get('total_calls', 1), 1) * 100, 2
                    )
                },
                'recent_performance': {
                    'avg_response_time_ms': round(recent_avg_response, 2),
                    'sample_size': len(recent_metrics)
                },
                'top_research_categories': dict(
                    sorted(
                        [(cat, stats['total_calls']) for cat, stats in self._category_stats.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                ),
                'top_error_types': dict(self._error_patterns.most_common(5)),
                'buffer_size': len(self._metrics_buffer)
            }


class APIUsageMetricsLogger:
    """
    Main API usage metrics logging system.
    
    Provides comprehensive logging of API usage metrics with integration
    to existing logging infrastructure and cost tracking systems.
    """
    
    def __init__(self, 
                 config: Any = None,
                 cost_persistence: Optional[CostPersistence] = None,
                 budget_manager: Optional[BudgetManager] = None,
                 research_categorizer: Optional[ResearchCategorizer] = None,
                 audit_trail: Optional[AuditTrail] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize API usage metrics logger.
        
        Args:
            config: Configuration object with logging settings
            cost_persistence: Cost persistence layer for integration
            budget_manager: Budget manager for cost tracking
            research_categorizer: Research categorizer for metrics
            audit_trail: Audit trail for compliance logging
            logger: Logger instance for metrics logging
        """
        self.config = config
        self.cost_persistence = cost_persistence
        self.budget_manager = budget_manager
        self.research_categorizer = research_categorizer
        self.audit_trail = audit_trail
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        self._setup_metrics_logging()
        
        # Initialize metrics aggregator
        self.metrics_aggregator = MetricsAggregator(self.logger)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Performance tracking
        self._active_operations = {}
        self._operation_counter = 0
        
        self.logger.info(f"API Usage Metrics Logger initialized with session ID: {self.session_id}")
    
    def _setup_metrics_logging(self) -> None:
        """Set up specialized metrics logging handlers."""
        try:
            # Create metrics-specific logger
            self.metrics_logger = logging.getLogger(f"{self.logger.name}.metrics")
            self.metrics_logger.setLevel(logging.INFO)
            self.metrics_logger.propagate = False
            
            # Set up metrics file handler if file logging is enabled
            if hasattr(self.config, 'enable_file_logging') and self.config.enable_file_logging:
                log_dir = getattr(self.config, 'log_dir', Path('logs'))
                if isinstance(log_dir, str):
                    log_dir = Path(log_dir)
                
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Create specialized metrics log file
                metrics_log_path = log_dir / "api_metrics.log"
                
                metrics_handler = logging.handlers.RotatingFileHandler(
                    filename=str(metrics_log_path),
                    maxBytes=getattr(self.config, 'log_max_bytes', 10 * 1024 * 1024),
                    backupCount=getattr(self.config, 'log_backup_count', 5),
                    encoding="utf-8"
                )
                
                # Structured JSON formatter for metrics
                metrics_formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "logger": "%(name)s", '
                    '"level": "%(levelname)s", "message": %(message)s}'
                )
                metrics_handler.setFormatter(metrics_formatter)
                self.metrics_logger.addHandler(metrics_handler)
                
                # Create audit log file for compliance
                audit_log_path = log_dir / "api_audit.log"
                
                audit_handler = logging.handlers.RotatingFileHandler(
                    filename=str(audit_log_path),
                    maxBytes=getattr(self.config, 'log_max_bytes', 10 * 1024 * 1024),
                    backupCount=getattr(self.config, 'log_backup_count', 10),  # Keep more audit logs
                    encoding="utf-8"
                )
                
                audit_formatter = logging.Formatter(
                    '%(asctime)s - AUDIT - %(message)s'
                )
                audit_handler.setFormatter(audit_formatter)
                
                self.audit_logger = logging.getLogger(f"{self.logger.name}.audit")
                self.audit_logger.setLevel(logging.INFO)
                self.audit_logger.propagate = False
                self.audit_logger.addHandler(audit_handler)
            else:
                self.audit_logger = self.logger
            
        except Exception as e:
            self.logger.warning(f"Could not set up specialized metrics logging: {e}")
            self.metrics_logger = self.logger
            self.audit_logger = self.logger
    
    @contextmanager
    def track_api_call(self, 
                       operation_name: str,
                       model_name: Optional[str] = None,
                       research_category: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking individual API calls.
        
        Usage:
            with metrics_logger.track_api_call("llm_generate", "gpt-4") as tracker:
                # Make API call
                result = api_call()
                tracker.set_tokens(prompt=100, completion=50)
                tracker.set_cost(0.01)
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Create metric template
        metric = APIMetric(
            session_id=self.session_id,
            operation_name=operation_name,
            model_name=model_name,
            research_category=research_category or ResearchCategory.GENERAL_QUERY.value,
            metadata=metadata or {},
            memory_usage_mb=start_memory
        )
        
        # Track active operation
        with self._lock:
            self._operation_counter += 1
            self._active_operations[operation_id] = {
                'metric': metric,
                'start_time': start_time,
                'start_memory': start_memory
            }
            metric.concurrent_operations = len(self._active_operations)
        
        class APICallTracker:
            def __init__(self, metric: APIMetric, logger: 'APIUsageMetricsLogger'):
                self.metric = metric
                self.logger = logger
                self.completed = False
            
            def set_tokens(self, prompt: int = 0, completion: int = 0, embedding: int = 0):
                self.metric.prompt_tokens = prompt
                self.metric.completion_tokens = completion
                self.metric.embedding_tokens = embedding
                self.metric.total_tokens = prompt + completion + embedding
            
            def set_cost(self, cost_usd: float):
                self.metric.cost_usd = cost_usd
            
            def set_response_details(self, 
                                   response_time_ms: Optional[float] = None,
                                   request_size: Optional[int] = None,
                                   response_size: Optional[int] = None):
                if response_time_ms:
                    self.metric.response_time_ms = response_time_ms
                if request_size:
                    self.metric.request_size_bytes = request_size
                if response_size:
                    self.metric.response_size_bytes = response_size
            
            def set_error(self, error_type: str, error_message: str):
                self.metric.success = False
                self.metric.error_type = error_type
                self.metric.error_message = error_message
            
            def add_metadata(self, key: str, value: Any):
                self.metric.metadata[key] = value
            
            def complete(self):
                if not self.completed:
                    self.logger._complete_operation(operation_id, self.metric)
                    self.completed = True
        
        tracker = APICallTracker(metric, self)
        
        try:
            yield tracker
        except Exception as e:
            tracker.set_error(type(e).__name__, str(e))
            raise
        finally:
            if not tracker.completed:
                tracker.complete()
    
    def _complete_operation(self, operation_id: str, metric: APIMetric) -> None:
        """Complete an API operation and log metrics."""
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Update metric with completion details
        if not metric.response_time_ms:
            metric.response_time_ms = (end_time - metric.timestamp) * 1000
        
        # Memory usage delta
        with self._lock:
            if operation_id in self._active_operations:
                start_memory = self._active_operations[operation_id]['start_memory']
                del self._active_operations[operation_id]
                metric.memory_usage_mb = max(end_memory - start_memory, 0)
        
        # Add to aggregator
        self.metrics_aggregator.add_metric(metric)
        
        # Log structured metrics
        self._log_metric(metric)
        
        # Integrate with cost tracking systems
        self._integrate_with_cost_systems(metric)
        
        # Log audit trail
        self._log_audit_trail(metric)
    
    def _log_metric(self, metric: APIMetric) -> None:
        """Log structured metric data."""
        try:
            metric_data = metric.to_dict()
            
            # Log to metrics logger as JSON
            self.metrics_logger.info(json.dumps(metric_data, default=str))
            
            # Log summary to main logger
            summary = (
                f"API Call: {metric.operation_name} | "
                f"Model: {metric.model_name} | "
                f"Tokens: {metric.total_tokens} | "
                f"Cost: ${metric.cost_usd:.6f} | "
                f"Time: {metric.response_time_ms:.1f}ms | "
                f"Success: {metric.success}"
            )
            
            if metric.research_category != ResearchCategory.GENERAL_QUERY.value:
                summary += f" | Category: {metric.research_category}"
            
            log_level = logging.INFO if metric.success else logging.WARNING
            self.logger.log(log_level, summary)
            
        except Exception as e:
            self.logger.error(f"Error logging metric: {e}")
    
    def _integrate_with_cost_systems(self, metric: APIMetric) -> None:
        """Integrate metric with existing cost tracking systems."""
        try:
            # Convert to cost record and persist
            if self.cost_persistence:
                cost_record = metric.to_cost_record()
                self.cost_persistence.record_cost(cost_record)
            
            # Check budget constraints
            if self.budget_manager and metric.cost_usd > 0:
                current_usage = self.budget_manager.check_budget_status()
                
                # Update metric with budget utilization
                if 'daily' in current_usage:
                    daily = current_usage['daily']
                    if daily.get('budget_limit', 0) > 0:
                        metric.daily_budget_used_percent = (
                            daily.get('current_cost', 0) / daily['budget_limit'] * 100
                        )
                
                if 'monthly' in current_usage:
                    monthly = current_usage['monthly']
                    if monthly.get('budget_limit', 0) > 0:
                        metric.monthly_budget_used_percent = (
                            monthly.get('current_cost', 0) / monthly['budget_limit'] * 100
                        )
            
            # Enhance with research categorization
            if self.research_categorizer and metric.query_type:
                enhanced_category = self.research_categorizer.categorize_query(
                    metric.query_type, metric.subject_area
                )
                if enhanced_category:
                    metric.research_category = enhanced_category.value
                    metric.metadata['enhanced_categorization'] = True
            
        except Exception as e:
            self.logger.error(f"Error integrating with cost systems: {e}")
    
    def _log_audit_trail(self, metric: APIMetric) -> None:
        """Log audit trail for compliance."""
        try:
            audit_data = {
                'event_type': 'api_usage',
                'timestamp': metric.timestamp,
                'session_id': metric.session_id,
                'operation': metric.operation_name,
                'model': metric.model_name,
                'tokens': metric.total_tokens,
                'cost_usd': metric.cost_usd,
                'success': metric.success,
                'research_category': metric.research_category,
                'user_id': metric.user_id,
                'project_id': metric.project_id,
                'compliance_level': metric.compliance_level
            }
            
            if not metric.success:
                audit_data['error_type'] = metric.error_type
                audit_data['error_message'] = metric.error_message
            
            self.audit_logger.info(json.dumps(audit_data))
            
            # Record in audit trail system if available
            if self.audit_trail:
                self.audit_trail.record_event(
                    event_type='api_usage',
                    event_data=audit_data,
                    user_id=metric.user_id,
                    session_id=metric.session_id
                )
                
        except Exception as e:
            self.logger.error(f"Error logging audit trail: {e}")
    
    def log_batch_operation(self, 
                           operation_name: str,
                           batch_size: int,
                           total_tokens: int,
                           total_cost: float,
                           processing_time_ms: float,
                           success_count: int,
                           error_count: int,
                           research_category: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log metrics for batch operations.
        
        Args:
            operation_name: Name of the batch operation
            batch_size: Number of items in the batch
            total_tokens: Total tokens consumed
            total_cost: Total cost in USD
            processing_time_ms: Total processing time in milliseconds
            success_count: Number of successful operations
            error_count: Number of failed operations
            research_category: Research category for the batch
            metadata: Additional metadata
        """
        metric = APIMetric(
            session_id=self.session_id,
            metric_type=MetricType.HYBRID_OPERATION,
            operation_name=f"batch_{operation_name}",
            total_tokens=total_tokens,
            cost_usd=total_cost,
            response_time_ms=processing_time_ms,
            success=error_count == 0,
            research_category=research_category or ResearchCategory.GENERAL_QUERY.value,
            metadata={
                **(metadata or {}),
                'batch_size': batch_size,
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_count / batch_size if batch_size > 0 else 0
            }
        )
        
        # Add to aggregator and log
        self.metrics_aggregator.add_metric(metric)
        self._log_metric(metric)
        self._integrate_with_cost_systems(metric)
        self._log_audit_trail(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        try:
            current_stats = self.metrics_aggregator.get_current_stats()
            
            # Add system information
            current_stats['system'] = {
                'memory_usage_mb': self._get_memory_usage(),
                'active_operations': len(self._active_operations),
                'session_uptime_seconds': time.time() - self.start_time,
                'session_id': self.session_id
            }
            
            # Add integration status
            current_stats['integration_status'] = {
                'cost_persistence': self.cost_persistence is not None,
                'budget_manager': self.budget_manager is not None,
                'research_categorizer': self.research_categorizer is not None,
                'audit_trail': self.audit_trail is not None
            }
            
            return current_stats
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def log_system_event(self, 
                        event_type: str, 
                        event_data: Dict[str, Any],
                        user_id: Optional[str] = None) -> None:
        """
        Log system events for monitoring and debugging.
        
        Args:
            event_type: Type of system event
            event_data: Event data dictionary
            user_id: Optional user ID for audit purposes
        """
        try:
            event_record = {
                'timestamp': time.time(),
                'event_type': event_type,
                'session_id': self.session_id,
                'user_id': user_id,
                'data': event_data
            }
            
            self.logger.info(f"System Event: {event_type} - {json.dumps(event_data)}")
            
            # Log to audit trail if available
            if self.audit_trail:
                self.audit_trail.record_event(
                    event_type=f'system_{event_type}',
                    event_data=event_record,
                    user_id=user_id,
                    session_id=self.session_id
                )
                
        except Exception as e:
            self.logger.error(f"Error logging system event: {e}")
    
    def close(self) -> None:
        """Clean shutdown of metrics logging."""
        try:
            # Log final performance summary
            final_summary = self.get_performance_summary()
            self.logger.info(f"API Metrics Logger shutdown - Final Summary: {json.dumps(final_summary)}")
            
            # Close handlers
            for handler in self.metrics_logger.handlers[:]:
                handler.close()
                self.metrics_logger.removeHandler(handler)
            
            for handler in self.audit_logger.handlers[:]:
                if handler not in self.logger.handlers:  # Don't close shared handlers
                    handler.close()
                    self.audit_logger.removeHandler(handler)
            
        except Exception as e:
            self.logger.error(f"Error during metrics logger shutdown: {e}")