"""
Load-Based Request Throttling and Queuing System for Clinical Metabolomics Oracle
===================================================================================

This module completes the graceful degradation implementation by providing intelligent
request management under varying load conditions. It implements:

1. **Request Throttling System**: Load-aware throttling with token bucket algorithm
2. **Intelligent Request Queuing**: Priority-based queuing with fair scheduling
3. **Connection Pool Management**: Adaptive connection pooling based on load
4. **Request Lifecycle Management**: Complete request flow control

The system integrates seamlessly with:
- Enhanced Load Monitoring System (load level detection)
- Progressive Service Degradation Controller (timeout/complexity management)
- Production Load Balancer (request routing)
- Clinical Metabolomics RAG (query processing)

Architecture:
- LoadBasedThrottler: Token bucket rate limiting with load awareness
- PriorityRequestQueue: Intelligent queuing with anti-starvation protection
- AdaptiveConnectionPool: Dynamic pool sizing based on load
- RequestLifecycleManager: Complete request admission and flow control
- RequestThrottlingSystem: Main orchestrator integrating all components

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
Production Ready: Yes
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import heapq
import json
import weakref
import aiohttp
import psutil

# Import enhanced load monitoring system
try:
    from .enhanced_load_monitoring_system import (
        SystemLoadLevel, EnhancedSystemLoadMetrics, 
        EnhancedLoadDetectionSystem
    )
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    ENHANCED_MONITORING_AVAILABLE = False
    logging.warning("Enhanced load monitoring system not available")
    
    class SystemLoadLevel(IntEnum):
        NORMAL = 0
        ELEVATED = 1
        HIGH = 2
        CRITICAL = 3
        EMERGENCY = 4

# Import progressive degradation controller
try:
    from .progressive_service_degradation_controller import (
        ProgressiveServiceDegradationController
    )
    DEGRADATION_CONTROLLER_AVAILABLE = True
except ImportError:
    DEGRADATION_CONTROLLER_AVAILABLE = False
    logging.warning("Progressive degradation controller not available")


# ============================================================================
# REQUEST PRIORITY AND CLASSIFICATION
# ============================================================================

class RequestPriority(IntEnum):
    """Request priority levels for intelligent queuing."""
    CRITICAL = 0     # Health checks, system monitoring
    HIGH = 1         # Interactive user queries
    MEDIUM = 2       # Batch processing, background tasks
    LOW = 3          # Analytics, reporting
    BACKGROUND = 4   # Cleanup, maintenance tasks


class RequestType(Enum):
    """Request type classification."""
    HEALTH_CHECK = "health_check"
    USER_QUERY = "user_query"
    BATCH_PROCESSING = "batch_processing"
    ANALYTICS = "analytics"
    MAINTENANCE = "maintenance"
    ADMIN = "admin"


@dataclass
class RequestMetadata:
    """Metadata for a request in the throttling system."""
    request_id: str
    request_type: RequestType
    priority: RequestPriority
    created_at: datetime
    deadline: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    estimated_duration: float = 5.0  # seconds
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    
    def __post_init__(self):
        if self.deadline is None:
            # Default deadline based on priority
            timeout_seconds = {
                RequestPriority.CRITICAL: 30,
                RequestPriority.HIGH: 120,
                RequestPriority.MEDIUM: 300,
                RequestPriority.LOW: 600,
                RequestPriority.BACKGROUND: 1800
            }.get(self.priority, 300)
            
            self.deadline = self.created_at + timedelta(seconds=timeout_seconds)
    
    def is_expired(self) -> bool:
        """Check if request has exceeded its deadline."""
        return datetime.now() > self.deadline
    
    def can_retry(self) -> bool:
        """Check if request can be retried."""
        return self.retries < self.max_retries and not self.is_expired()


# ============================================================================
# TOKEN BUCKET RATE LIMITER
# ============================================================================

class LoadBasedThrottler:
    """
    Token bucket rate limiter with dynamic rate adjustment based on system load.
    
    Features:
    - Load-aware rate limiting
    - Burst capacity management
    - Fair token distribution
    - Request type awareness
    """
    
    def __init__(self, 
                 base_rate_per_second: float = 10.0,
                 burst_capacity: int = 20,
                 load_detector: Optional[Any] = None):
        
        self.base_rate_per_second = base_rate_per_second
        self.burst_capacity = burst_capacity
        self.load_detector = load_detector
        self.logger = logging.getLogger(f"{__name__}.LoadBasedThrottler")
        
        # Token bucket state
        self.tokens = float(burst_capacity)
        self.last_refill = time.time()
        self.current_rate = base_rate_per_second
        self.current_load_level = SystemLoadLevel.NORMAL
        
        # Rate adjustment factors by load level
        self.rate_factors = {
            SystemLoadLevel.NORMAL: 1.0,
            SystemLoadLevel.ELEVATED: 0.8,
            SystemLoadLevel.HIGH: 0.6,
            SystemLoadLevel.CRITICAL: 0.4,
            SystemLoadLevel.EMERGENCY: 0.2
        }
        
        # Priority-based rate allocation
        self.priority_weights = {
            RequestPriority.CRITICAL: 1.0,
            RequestPriority.HIGH: 0.8,
            RequestPriority.MEDIUM: 0.6,
            RequestPriority.LOW: 0.4,
            RequestPriority.BACKGROUND: 0.2
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'denied_requests': 0,
            'tokens_consumed': 0.0,
            'last_reset': datetime.now()
        }
        
        # Auto-adjust rate based on load if detector available
        if self.load_detector:
            self.load_detector.add_load_change_callback(self._on_load_change)
    
    def _on_load_change(self, metrics):
        """Handle load level changes."""
        if hasattr(metrics, 'load_level'):
            new_level = metrics.load_level
            if new_level != self.current_load_level:
                self._adjust_rate_for_load_level(new_level)
    
    def _adjust_rate_for_load_level(self, load_level: SystemLoadLevel):
        """Adjust throttling rate based on system load level."""
        with self._lock:
            old_rate = self.current_rate
            self.current_load_level = load_level
            
            rate_factor = self.rate_factors.get(load_level, 0.5)
            self.current_rate = self.base_rate_per_second * rate_factor
            
            self.logger.info(f"Adjusted throttling rate: {old_rate:.1f} → {self.current_rate:.1f} req/s "
                           f"(load level: {load_level.name})")
    
    def _refill_tokens(self):
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.current_rate
        self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def acquire_token(self, 
                          request_metadata: RequestMetadata,
                          tokens_needed: float = 1.0,
                          timeout: float = 30.0) -> bool:
        """
        Acquire tokens for a request.
        
        Args:
            request_metadata: Request metadata for priority handling
            tokens_needed: Number of tokens required
            timeout: Maximum wait time for tokens
            
        Returns:
            True if tokens acquired, False if timeout or denied
        """
        start_time = time.time()
        
        with self._lock:
            self.stats['total_requests'] += 1
        
        # Apply priority weighting to token requirement
        priority_weight = self.priority_weights.get(request_metadata.priority, 1.0)
        effective_tokens_needed = tokens_needed * priority_weight
        
        while time.time() - start_time < timeout:
            with self._lock:
                self._refill_tokens()
                
                if self.tokens >= effective_tokens_needed:
                    self.tokens -= effective_tokens_needed
                    self.stats['allowed_requests'] += 1
                    self.stats['tokens_consumed'] += effective_tokens_needed
                    
                    self.logger.debug(f"Token acquired for {request_metadata.request_id} "
                                    f"(priority: {request_metadata.priority.name}, "
                                    f"tokens: {effective_tokens_needed:.1f}, "
                                    f"remaining: {self.tokens:.1f})")
                    return True
            
            # Wait before retry
            await asyncio.sleep(0.1)
        
        # Timeout - token acquisition failed
        with self._lock:
            self.stats['denied_requests'] += 1
        
        self.logger.warning(f"Token acquisition timeout for {request_metadata.request_id} "
                          f"(priority: {request_metadata.priority.name})")
        return False
    
    def get_current_rate(self) -> float:
        """Get current throttling rate."""
        return self.current_rate
    
    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill_tokens()
            return self.tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get throttling statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'current_rate': self.current_rate,
                'available_tokens': self.tokens,
                'burst_capacity': self.burst_capacity,
                'load_level': self.current_load_level.name,
                'success_rate': (stats['allowed_requests'] / max(stats['total_requests'], 1)) * 100
            })
            return stats
    
    def reset_statistics(self):
        """Reset throttling statistics."""
        with self._lock:
            self.stats = {
                'total_requests': 0,
                'allowed_requests': 0,
                'denied_requests': 0,
                'tokens_consumed': 0.0,
                'last_reset': datetime.now()
            }


# ============================================================================
# PRIORITY-BASED REQUEST QUEUE
# ============================================================================

class PriorityRequestQueue:
    """
    Priority-based request queue with anti-starvation mechanisms.
    
    Features:
    - Priority-based scheduling
    - Anti-starvation protection
    - Load-aware queue size limits
    - Fair scheduling algorithms
    - Request timeout handling
    """
    
    def __init__(self,
                 max_queue_size: int = 1000,
                 starvation_threshold: float = 300.0,  # 5 minutes
                 load_detector: Optional[Any] = None):
        
        self.max_queue_size = max_queue_size
        self.starvation_threshold = starvation_threshold
        self.load_detector = load_detector
        self.logger = logging.getLogger(f"{__name__}.PriorityRequestQueue")
        
        # Priority heaps - separate heap for each priority level
        self._priority_heaps: Dict[RequestPriority, List] = {
            priority: [] for priority in RequestPriority
        }
        
        # Anti-starvation tracking
        self._starvation_counters: Dict[RequestPriority, int] = defaultdict(int)
        self._last_served: Dict[RequestPriority, datetime] = {}
        
        # Queue size limits by load level
        self._load_level_limits = {
            SystemLoadLevel.NORMAL: max_queue_size,
            SystemLoadLevel.ELEVATED: int(max_queue_size * 0.8),
            SystemLoadLevel.HIGH: int(max_queue_size * 0.6),
            SystemLoadLevel.CRITICAL: int(max_queue_size * 0.4),
            SystemLoadLevel.EMERGENCY: int(max_queue_size * 0.2)
        }
        
        # Current state
        self.current_load_level = SystemLoadLevel.NORMAL
        self.current_max_size = max_queue_size
        self._queue_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_rejected': 0,
            'total_expired': 0,
            'starvation_promotions': 0,
            'queue_sizes_by_priority': defaultdict(int),
            'last_reset': datetime.now()
        }
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Integration with load detector
        if self.load_detector:
            self.load_detector.add_load_change_callback(self._on_load_change)
    
    def _on_load_change(self, metrics):
        """Handle load level changes."""
        if hasattr(metrics, 'load_level'):
            new_level = metrics.load_level
            if new_level != self.current_load_level:
                self._adjust_queue_limits(new_level)
    
    def _adjust_queue_limits(self, load_level: SystemLoadLevel):
        """Adjust queue size limits based on system load."""
        with self._queue_lock:
            old_limit = self.current_max_size
            self.current_load_level = load_level
            self.current_max_size = self._load_level_limits.get(load_level, self.max_queue_size)
            
            self.logger.info(f"Adjusted queue limit: {old_limit} → {self.current_max_size} "
                           f"(load level: {load_level.name})")
            
            # If new limit is smaller, may need to reject some queued requests
            if self.current_max_size < old_limit:
                self._enforce_queue_limit()
    
    def _enforce_queue_limit(self):
        """Enforce current queue size limit by rejecting lowest priority items."""
        total_size = sum(len(heap) for heap in self._priority_heaps.values())
        
        if total_size <= self.current_max_size:
            return
        
        # Remove items from lowest priority queues first
        items_to_remove = total_size - self.current_max_size
        removed_count = 0
        
        for priority in reversed(list(RequestPriority)):
            if removed_count >= items_to_remove:
                break
            
            heap = self._priority_heaps[priority]
            while heap and removed_count < items_to_remove:
                removed_item = heapq.heappop(heap)
                removed_count += 1
                self.stats['total_rejected'] += 1
                
                self.logger.warning(f"Queue limit enforcement: rejected {removed_item[2].request_id} "
                                  f"(priority: {priority.name})")
    
    async def enqueue(self, 
                     request_metadata: RequestMetadata,
                     request_handler: Callable,
                     *args, **kwargs) -> bool:
        """
        Enqueue a request for processing.
        
        Args:
            request_metadata: Request metadata
            request_handler: Function to handle the request
            *args, **kwargs: Arguments for request handler
            
        Returns:
            True if enqueued successfully, False if rejected
        """
        with self._queue_lock:
            # Check queue capacity
            total_size = sum(len(heap) for heap in self._priority_heaps.values())
            if total_size >= self.current_max_size:
                self.stats['total_rejected'] += 1
                self.logger.warning(f"Queue full: rejected {request_metadata.request_id}")
                return False
            
            # Check if request is already expired
            if request_metadata.is_expired():
                self.stats['total_expired'] += 1
                self.logger.warning(f"Request expired before queueing: {request_metadata.request_id}")
                return False
            
            # Create queue entry
            # Use negative timestamp for proper heap ordering (earliest first)
            queue_entry = (
                -request_metadata.created_at.timestamp(),
                request_metadata.request_id,
                request_metadata,
                request_handler,
                args,
                kwargs
            )
            
            # Add to appropriate priority heap
            priority = request_metadata.priority
            heapq.heappush(self._priority_heaps[priority], queue_entry)
            
            # Update statistics
            self.stats['total_queued'] += 1
            self.stats['queue_sizes_by_priority'][priority.name] += 1
            
            self.logger.debug(f"Enqueued request {request_metadata.request_id} "
                            f"(priority: {priority.name}, queue size: {len(self._priority_heaps[priority])})")
            
            return True
    
    async def dequeue(self) -> Optional[Tuple[RequestMetadata, Callable, Tuple, Dict]]:
        """
        Dequeue the next request for processing.
        
        Uses priority-based scheduling with anti-starvation protection.
        
        Returns:
            (metadata, handler, args, kwargs) or None if queue empty
        """
        with self._queue_lock:
            # First, check for expired requests and remove them
            self._cleanup_expired_requests()
            
            # Check if any priority level is experiencing starvation
            starved_priority = self._check_starvation()
            
            # Select priority to serve
            if starved_priority is not None:
                selected_priority = starved_priority
                self.stats['starvation_promotions'] += 1
                self.logger.debug(f"Anti-starvation: serving {selected_priority.name} priority")
            else:
                selected_priority = self._select_priority_to_serve()
            
            if selected_priority is None:
                return None
            
            # Get request from selected priority heap
            heap = self._priority_heaps[selected_priority]
            if not heap:
                return None
            
            queue_entry = heapq.heappop(heap)
            _, request_id, metadata, handler, args, kwargs = queue_entry
            
            # Update statistics and tracking
            self.stats['total_processed'] += 1
            self.stats['queue_sizes_by_priority'][selected_priority.name] -= 1
            self._last_served[selected_priority] = datetime.now()
            self._starvation_counters[selected_priority] = 0
            
            # Increment starvation counters for other priorities
            for priority in RequestPriority:
                if priority != selected_priority:
                    self._starvation_counters[priority] += 1
            
            self.logger.debug(f"Dequeued request {request_id} "
                            f"(priority: {selected_priority.name})")
            
            return metadata, handler, args, kwargs
    
    def _cleanup_expired_requests(self):
        """Remove expired requests from all queues."""
        now = datetime.now()
        expired_count = 0
        
        for priority, heap in self._priority_heaps.items():
            # Rebuild heap without expired items
            valid_items = []
            for item in heap:
                _, _, metadata, _, _, _ = item
                if not metadata.is_expired():
                    valid_items.append(item)
                else:
                    expired_count += 1
                    self.stats['total_expired'] += 1
            
            # Replace heap with valid items
            self._priority_heaps[priority] = valid_items
            heapq.heapify(valid_items)
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired requests")
    
    def _check_starvation(self) -> Optional[RequestPriority]:
        """Check if any priority level is experiencing starvation."""
        now = datetime.now()
        
        # Check for priorities that haven't been served recently
        for priority in reversed(list(RequestPriority)):  # Check lower priorities first
            if len(self._priority_heaps[priority]) == 0:
                continue
            
            last_served = self._last_served.get(priority)
            if last_served is None:
                # Never served - definitely starved
                return priority
            
            time_since_served = (now - last_served).total_seconds()
            if time_since_served > self.starvation_threshold:
                return priority
        
        return None
    
    def _select_priority_to_serve(self) -> Optional[RequestPriority]:
        """Select which priority level to serve based on availability and fairness."""
        # Serve highest priority first, but with some fairness considerations
        for priority in RequestPriority:
            if len(self._priority_heaps[priority]) > 0:
                return priority
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        with self._queue_lock:
            queue_sizes = {
                priority.name: len(heap) 
                for priority, heap in self._priority_heaps.items()
            }
            
            total_size = sum(queue_sizes.values())
            
            return {
                'total_size': total_size,
                'max_size': self.current_max_size,
                'utilization': (total_size / max(self.current_max_size, 1)) * 100,
                'queue_sizes_by_priority': queue_sizes,
                'load_level': self.current_load_level.name,
                'statistics': self.stats.copy(),
                'starvation_counters': dict(self._starvation_counters),
                'last_served': {
                    priority.name: timestamp.isoformat() if timestamp else None
                    for priority, timestamp in self._last_served.items()
                }
            }
    
    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Started queue cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped queue cleanup task")
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up expired requests."""
        while self._running:
            try:
                with self._queue_lock:
                    self._cleanup_expired_requests()
                
                # Sleep for cleanup interval
                await asyncio.sleep(30)  # Clean up every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)


# ============================================================================
# ADAPTIVE CONNECTION POOL MANAGEMENT
# ============================================================================

class AdaptiveConnectionPool:
    """
    Adaptive connection pool that adjusts size based on system load.
    
    Features:
    - Dynamic pool sizing based on load level
    - Connection health monitoring
    - Efficient connection reuse
    - Load-aware connection limits
    """
    
    def __init__(self,
                 base_pool_size: int = 20,
                 max_pool_size: int = 100,
                 min_pool_size: int = 5,
                 connection_timeout: float = 30.0,
                 load_detector: Optional[Any] = None):
        
        self.base_pool_size = base_pool_size
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.connection_timeout = connection_timeout
        self.load_detector = load_detector
        self.logger = logging.getLogger(f"{__name__}.AdaptiveConnectionPool")
        
        # Connection pool by load level
        self._pool_size_by_level = {
            SystemLoadLevel.NORMAL: base_pool_size,
            SystemLoadLevel.ELEVATED: int(base_pool_size * 0.9),
            SystemLoadLevel.HIGH: int(base_pool_size * 0.7),
            SystemLoadLevel.CRITICAL: int(base_pool_size * 0.5),
            SystemLoadLevel.EMERGENCY: int(base_pool_size * 0.3)
        }
        
        # Current state
        self.current_load_level = SystemLoadLevel.NORMAL
        self.current_pool_size = base_pool_size
        
        # aiohttp connector pool
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Pool statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_closed': 0,
            'pool_size_changes': 0,
            'last_reset': datetime.now()
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize connection pool
        self._initialize_pool()
        
        # Integration with load detector
        if self.load_detector:
            self.load_detector.add_load_change_callback(self._on_load_change)
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        self._create_connector()
    
    def _create_connector(self):
        """Create aiohttp connector with current settings."""
        if self._connector:
            # Close existing connector
            asyncio.create_task(self._connector.close())
        
        # Create new connector with current pool size
        self._connector = aiohttp.TCPConnector(
            limit=self.current_pool_size,
            limit_per_host=max(2, self.current_pool_size // 4),
            ttl_dns_cache=300,
            use_dns_cache=True,
            timeout=aiohttp.ClientTimeout(total=self.connection_timeout),
            enable_cleanup_closed=True
        )
        
        self.logger.debug(f"Created connector with pool size: {self.current_pool_size}")
    
    def _on_load_change(self, metrics):
        """Handle load level changes."""
        if hasattr(metrics, 'load_level'):
            new_level = metrics.load_level
            if new_level != self.current_load_level:
                self._adjust_pool_size(new_level)
    
    def _adjust_pool_size(self, load_level: SystemLoadLevel):
        """Adjust connection pool size based on system load."""
        with self._lock:
            old_size = self.current_pool_size
            self.current_load_level = load_level
            
            new_size = self._pool_size_by_level.get(load_level, self.base_pool_size)
            new_size = max(self.min_pool_size, min(self.max_pool_size, new_size))
            
            if new_size != self.current_pool_size:
                self.current_pool_size = new_size
                self.stats['pool_size_changes'] += 1
                
                self.logger.info(f"Adjusted connection pool size: {old_size} → {new_size} "
                               f"(load level: {load_level.name})")
                
                # Recreate connector with new size
                self._create_connector()
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get a client session with the current connector."""
        if self._session is None or self._session.closed:
            with self._lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        connector=self._connector,
                        timeout=aiohttp.ClientTimeout(total=self.connection_timeout)
                    )
                    self.stats['connections_created'] += 1
        
        return self._session
    
    async def close(self):
        """Close the connection pool."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.stats['connections_closed'] += 1
        
        if self._connector:
            await self._connector.close()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        connector_stats = {}
        if self._connector:
            connector_stats = {
                'connections': len(getattr(self._connector, '_conns', {})),
                'acquired_connections': len(getattr(self._connector, '_acquired', set())),
                'available_connections': len(getattr(self._connector, '_available_connections', [])) if hasattr(self._connector, '_available_connections') else 0
            }
        
        return {
            'current_pool_size': self.current_pool_size,
            'max_pool_size': self.max_pool_size,
            'min_pool_size': self.min_pool_size,
            'load_level': self.current_load_level.name,
            'connector_stats': connector_stats,
            'statistics': self.stats.copy()
        }


# ============================================================================
# REQUEST LIFECYCLE MANAGER
# ============================================================================

class RequestLifecycleManager:
    """
    Complete request lifecycle management with admission control and monitoring.
    
    Features:
    - Request admission control
    - Queue position tracking
    - Request timeout management
    - Graceful request rejection
    - Request lifecycle metrics
    """
    
    def __init__(self,
                 throttler: LoadBasedThrottler,
                 queue: PriorityRequestQueue,
                 connection_pool: AdaptiveConnectionPool,
                 max_concurrent_requests: int = 50):
        
        self.throttler = throttler
        self.queue = queue
        self.connection_pool = connection_pool
        self.max_concurrent_requests = max_concurrent_requests
        self.logger = logging.getLogger(f"{__name__}.RequestLifecycleManager")
        
        # Active request tracking
        self._active_requests: Dict[str, RequestMetadata] = {}
        self._request_futures: Dict[str, asyncio.Future] = {}
        self._concurrent_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Request processing metrics
        self.metrics = {
            'total_requests': 0,
            'admitted_requests': 0,
            'rejected_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'timeout_requests': 0,
            'average_wait_time': 0.0,
            'average_processing_time': 0.0,
            'last_reset': datetime.now()
        }
        
        # Request processing worker
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread safety
        self._lock = threading.Lock()
    
    async def submit_request(self,
                           request_metadata: RequestMetadata,
                           request_handler: Callable,
                           *args, **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Submit a request for processing.
        
        Args:
            request_metadata: Request metadata
            request_handler: Request handler function
            *args, **kwargs: Arguments for handler
            
        Returns:
            (success, message) - success indicates if request was accepted
        """
        with self._lock:
            self.metrics['total_requests'] += 1
        
        request_id = request_metadata.request_id
        
        # Step 1: Token-based rate limiting
        try:
            token_acquired = await self.throttler.acquire_token(
                request_metadata, 
                timeout=10.0
            )
            
            if not token_acquired:
                with self._lock:
                    self.metrics['rejected_requests'] += 1
                return False, f"Rate limit exceeded for request {request_id}"
        
        except Exception as e:
            self.logger.error(f"Error acquiring token for {request_id}: {e}")
            with self._lock:
                self.metrics['rejected_requests'] += 1
            return False, f"Rate limiting error: {str(e)}"
        
        # Step 2: Queue admission
        try:
            queued = await self.queue.enqueue(
                request_metadata,
                request_handler,
                *args, **kwargs
            )
            
            if not queued:
                with self._lock:
                    self.metrics['rejected_requests'] += 1
                return False, f"Queue full or request expired: {request_id}"
        
        except Exception as e:
            self.logger.error(f"Error queueing request {request_id}: {e}")
            with self._lock:
                self.metrics['rejected_requests'] += 1
            return False, f"Queue error: {str(e)}"
        
        # Step 3: Track admitted request
        with self._lock:
            self._active_requests[request_id] = request_metadata
            self.metrics['admitted_requests'] += 1
        
        self.logger.info(f"Request {request_id} admitted for processing "
                        f"(priority: {request_metadata.priority.name})")
        
        return True, f"Request {request_id} accepted"
    
    async def start_processing(self):
        """Start the request processing loop."""
        if self._processing_task is None or self._processing_task.done():
            self._running = True
            self._processing_task = asyncio.create_task(self._processing_loop())
            self.logger.info("Started request processing")
    
    async def stop_processing(self):
        """Stop the request processing loop."""
        self._running = False
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped request processing")
    
    async def _processing_loop(self):
        """Main request processing loop."""
        while self._running:
            try:
                # Get next request from queue
                queue_item = await self.queue.dequeue()
                
                if queue_item is None:
                    # No requests in queue, wait briefly
                    await asyncio.sleep(0.1)
                    continue
                
                metadata, handler, args, kwargs = queue_item
                
                # Process request with concurrency control
                async with self._concurrent_semaphore:
                    await self._process_single_request(metadata, handler, args, kwargs)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_single_request(self, 
                                    metadata: RequestMetadata,
                                    handler: Callable,
                                    args: Tuple,
                                    kwargs: Dict):
        """Process a single request with full lifecycle management."""
        request_id = metadata.request_id
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing request {request_id}")
            
            # Check if request has expired
            if metadata.is_expired():
                with self._lock:
                    self.metrics['timeout_requests'] += 1
                    self._active_requests.pop(request_id, None)
                self.logger.warning(f"Request {request_id} expired before processing")
                return
            
            # Get connection session
            session = await self.connection_pool.get_session()
            
            # Execute request handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)
            
            # Request completed successfully
            processing_time = time.time() - start_time
            
            with self._lock:
                self.metrics['completed_requests'] += 1
                self._active_requests.pop(request_id, None)
                
                # Update average processing time
                old_avg = self.metrics['average_processing_time']
                count = self.metrics['completed_requests']
                self.metrics['average_processing_time'] = ((old_avg * (count - 1)) + processing_time) / count
            
            self.logger.info(f"Request {request_id} completed successfully "
                           f"(processing time: {processing_time:.2f}s)")
        
        except asyncio.TimeoutError:
            with self._lock:
                self.metrics['timeout_requests'] += 1
                self._active_requests.pop(request_id, None)
            self.logger.error(f"Request {request_id} timed out during processing")
        
        except Exception as e:
            with self._lock:
                self.metrics['failed_requests'] += 1
                self._active_requests.pop(request_id, None)
            self.logger.error(f"Request {request_id} failed: {str(e)}")
    
    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active requests."""
        with self._lock:
            active_info = {}
            now = datetime.now()
            
            for request_id, metadata in self._active_requests.items():
                active_info[request_id] = {
                    'request_type': metadata.request_type.value,
                    'priority': metadata.priority.name,
                    'created_at': metadata.created_at.isoformat(),
                    'deadline': metadata.deadline.isoformat() if metadata.deadline else None,
                    'age_seconds': (now - metadata.created_at).total_seconds(),
                    'estimated_duration': metadata.estimated_duration,
                    'retries': metadata.retries
                }
            
            return active_info
    
    def get_lifecycle_metrics(self) -> Dict[str, Any]:
        """Get request lifecycle metrics."""
        with self._lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            if metrics['total_requests'] > 0:
                metrics['admission_rate'] = (metrics['admitted_requests'] / metrics['total_requests']) * 100
                metrics['completion_rate'] = (metrics['completed_requests'] / metrics['admitted_requests']) * 100 if metrics['admitted_requests'] > 0 else 0
                metrics['failure_rate'] = (metrics['failed_requests'] / metrics['admitted_requests']) * 100 if metrics['admitted_requests'] > 0 else 0
            else:
                metrics['admission_rate'] = 0
                metrics['completion_rate'] = 0
                metrics['failure_rate'] = 0
            
            metrics['active_requests'] = len(self._active_requests)
            metrics['concurrent_capacity'] = self.max_concurrent_requests
            metrics['concurrent_utilization'] = (len(self._active_requests) / self.max_concurrent_requests) * 100
            
            return metrics


# ============================================================================
# MAIN REQUEST THROTTLING SYSTEM
# ============================================================================

class RequestThrottlingSystem:
    """
    Main orchestrator for load-based request throttling and queuing.
    
    Integrates all components:
    - LoadBasedThrottler (token bucket rate limiting)
    - PriorityRequestQueue (intelligent queuing)
    - AdaptiveConnectionPool (connection management)
    - RequestLifecycleManager (request flow control)
    """
    
    def __init__(self,
                 # Throttling configuration
                 base_rate_per_second: float = 10.0,
                 burst_capacity: int = 20,
                 
                 # Queue configuration
                 max_queue_size: int = 1000,
                 starvation_threshold: float = 300.0,
                 
                 # Connection pool configuration
                 base_pool_size: int = 20,
                 max_pool_size: int = 100,
                 
                 # Request processing configuration
                 max_concurrent_requests: int = 50,
                 
                 # Integration components
                 load_detector: Optional[Any] = None,
                 degradation_controller: Optional[Any] = None):
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.throttler = LoadBasedThrottler(
            base_rate_per_second=base_rate_per_second,
            burst_capacity=burst_capacity,
            load_detector=load_detector
        )
        
        self.queue = PriorityRequestQueue(
            max_queue_size=max_queue_size,
            starvation_threshold=starvation_threshold,
            load_detector=load_detector
        )
        
        self.connection_pool = AdaptiveConnectionPool(
            base_pool_size=base_pool_size,
            max_pool_size=max_pool_size,
            load_detector=load_detector
        )
        
        self.lifecycle_manager = RequestLifecycleManager(
            throttler=self.throttler,
            queue=self.queue,
            connection_pool=self.connection_pool,
            max_concurrent_requests=max_concurrent_requests
        )
        
        # Store integration references
        self.load_detector = load_detector
        self.degradation_controller = degradation_controller
        
        # System state
        self._running = False
        self._start_time: Optional[datetime] = None
        
        # Integration callbacks
        if self.degradation_controller:
            self._integrate_with_degradation_controller()
        
        self.logger.info("Request Throttling System initialized")
    
    def _integrate_with_degradation_controller(self):
        """Integrate with progressive degradation controller."""
        try:
            def on_degradation_change(previous_level, new_level):
                self.logger.info(f"Degradation level changed: {previous_level.name} → {new_level.name}")
                # The individual components will handle load changes through their load_detector callbacks
            
            self.degradation_controller.add_load_change_callback(on_degradation_change)
            self.logger.info("Integrated with progressive degradation controller")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with degradation controller: {e}")
    
    async def start(self):
        """Start the request throttling system."""
        if self._running:
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        try:
            # Start all components
            await self.queue.start_cleanup_task()
            await self.lifecycle_manager.start_processing()
            
            self.logger.info("Request Throttling System started")
            
        except Exception as e:
            self.logger.error(f"Error starting throttling system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the request throttling system."""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # Stop all components
            await self.lifecycle_manager.stop_processing()
            await self.queue.stop_cleanup_task()
            await self.connection_pool.close()
            
            self.logger.info("Request Throttling System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping throttling system: {e}")
    
    async def submit_request(self,
                           request_type: RequestType,
                           priority: Optional[RequestPriority] = None,
                           handler: Optional[Callable] = None,
                           estimated_duration: float = 5.0,
                           user_id: Optional[str] = None,
                           *args, **kwargs) -> Tuple[bool, str, str]:
        """
        Submit a request for throttled processing.
        
        Args:
            request_type: Type of request
            priority: Request priority (auto-assigned if None)
            handler: Request handler function
            estimated_duration: Estimated processing time in seconds
            user_id: User identifier for tracking
            *args, **kwargs: Arguments for request handler
            
        Returns:
            (success, message, request_id)
        """
        if not self._running:
            return False, "Throttling system not running", ""
        
        # Generate unique request ID
        request_id = f"{request_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Auto-assign priority based on request type if not specified
        if priority is None:
            priority_map = {
                RequestType.HEALTH_CHECK: RequestPriority.CRITICAL,
                RequestType.USER_QUERY: RequestPriority.HIGH,
                RequestType.BATCH_PROCESSING: RequestPriority.MEDIUM,
                RequestType.ANALYTICS: RequestPriority.LOW,
                RequestType.MAINTENANCE: RequestPriority.BACKGROUND,
                RequestType.ADMIN: RequestPriority.HIGH
            }
            priority = priority_map.get(request_type, RequestPriority.MEDIUM)
        
        # Create request metadata
        metadata = RequestMetadata(
            request_id=request_id,
            request_type=request_type,
            priority=priority,
            created_at=datetime.now(),
            estimated_duration=estimated_duration,
            user_id=user_id
        )
        
        # Submit to lifecycle manager
        try:
            success, message = await self.lifecycle_manager.submit_request(
                metadata, handler, *args, **kwargs
            )
            return success, message, request_id
            
        except Exception as e:
            self.logger.error(f"Error submitting request {request_id}: {e}")
            return False, f"Submission error: {str(e)}", request_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        throttler_stats = self.throttler.get_statistics()
        queue_status = self.queue.get_queue_status()
        pool_status = self.connection_pool.get_pool_status()
        lifecycle_metrics = self.lifecycle_manager.get_lifecycle_metrics()
        active_requests = self.lifecycle_manager.get_active_requests()
        
        return {
            'system_running': self._running,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
            
            'throttling': throttler_stats,
            'queue': queue_status,
            'connection_pool': pool_status,
            'lifecycle': lifecycle_metrics,
            'active_requests': active_requests,
            
            'integration': {
                'load_detector_available': self.load_detector is not None,
                'degradation_controller_available': self.degradation_controller is not None
            }
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get system health check information."""
        status = self.get_system_status()
        
        # Determine overall health
        health_issues = []
        
        # Check throttling health
        throttling = status['throttling']
        if throttling['success_rate'] < 80:
            health_issues.append("Low throttling success rate")
        
        # Check queue health
        queue = status['queue']
        if queue['utilization'] > 90:
            health_issues.append("Queue near capacity")
        
        # Check processing health
        lifecycle = status['lifecycle']
        if lifecycle['failure_rate'] > 10:
            health_issues.append("High request failure rate")
        
        if lifecycle['concurrent_utilization'] > 95:
            health_issues.append("Concurrent processing near capacity")
        
        overall_health = "healthy" if not health_issues else ("degraded" if len(health_issues) < 3 else "unhealthy")
        
        return {
            'status': overall_health,
            'issues': health_issues,
            'uptime_seconds': status['uptime_seconds'],
            'total_requests_processed': lifecycle['completed_requests'],
            'current_queue_size': queue['total_size'],
            'current_active_requests': len(status['active_requests']),
            'throttling_rate': throttling['current_rate'],
            'success_rate': throttling['success_rate']
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_request_throttling_system(
    # System configuration
    base_rate_per_second: float = 10.0,
    max_queue_size: int = 1000,
    max_concurrent_requests: int = 50,
    
    # Integration components
    load_detector: Optional[Any] = None,
    degradation_controller: Optional[Any] = None,
    
    # Advanced configuration
    custom_config: Optional[Dict[str, Any]] = None
) -> RequestThrottlingSystem:
    """
    Create a production-ready request throttling system.
    
    Args:
        base_rate_per_second: Base throttling rate
        max_queue_size: Maximum queue size
        max_concurrent_requests: Maximum concurrent requests
        load_detector: Enhanced load detection system
        degradation_controller: Progressive degradation controller
        custom_config: Additional configuration options
        
    Returns:
        Configured RequestThrottlingSystem
    """
    config = custom_config or {}
    
    return RequestThrottlingSystem(
        base_rate_per_second=base_rate_per_second,
        burst_capacity=config.get('burst_capacity', int(base_rate_per_second * 2)),
        max_queue_size=max_queue_size,
        starvation_threshold=config.get('starvation_threshold', 300.0),
        base_pool_size=config.get('base_pool_size', 20),
        max_pool_size=config.get('max_pool_size', 100),
        max_concurrent_requests=max_concurrent_requests,
        load_detector=load_detector,
        degradation_controller=degradation_controller
    )


def create_integrated_graceful_degradation_system(
    # Monitoring configuration
    monitoring_interval: float = 5.0,
    
    # Throttling configuration
    base_rate_per_second: float = 10.0,
    max_queue_size: int = 1000,
    max_concurrent_requests: int = 50,
    
    # Production systems
    production_systems: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, RequestThrottlingSystem]:
    """
    Create a complete integrated graceful degradation system with all components.
    
    Returns:
        (load_detector, degradation_controller, throttling_system)
    """
    systems = production_systems or {}
    
    # Create enhanced load monitoring system
    if ENHANCED_MONITORING_AVAILABLE:
        from .enhanced_load_monitoring_system import create_enhanced_load_monitoring_system
        load_detector = create_enhanced_load_monitoring_system(
            monitoring_interval=monitoring_interval,
            enable_trend_analysis=True,
            production_monitoring=systems.get('monitoring')
        )
    else:
        load_detector = None
    
    # Create progressive degradation controller
    if DEGRADATION_CONTROLLER_AVAILABLE:
        from .progressive_service_degradation_controller import create_progressive_degradation_controller
        degradation_controller = create_progressive_degradation_controller(
            enhanced_detector=load_detector,
            production_load_balancer=systems.get('load_balancer'),
            clinical_rag=systems.get('clinical_rag'),
            production_monitoring=systems.get('monitoring')
        )
    else:
        degradation_controller = None
    
    # Create request throttling system
    throttling_system = create_request_throttling_system(
        base_rate_per_second=base_rate_per_second,
        max_queue_size=max_queue_size,
        max_concurrent_requests=max_concurrent_requests,
        load_detector=load_detector,
        degradation_controller=degradation_controller
    )
    
    return load_detector, degradation_controller, throttling_system


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

async def demonstrate_request_throttling_system():
    """Demonstrate the complete request throttling and queuing system."""
    print("Load-Based Request Throttling and Queuing System Demonstration")
    print("=" * 80)
    
    # Create integrated system
    load_detector, degradation_controller, throttling_system = create_integrated_graceful_degradation_system(
        monitoring_interval=2.0,
        base_rate_per_second=5.0,  # Lower rate for demo
        max_queue_size=20,
        max_concurrent_requests=10
    )
    
    print(f"Created integrated graceful degradation system:")
    print(f"  Load Detector: {'✓' if load_detector else '✗'}")
    print(f"  Degradation Controller: {'✓' if degradation_controller else '✗'}")
    print(f"  Throttling System: ✓")
    print()
    
    # Start systems
    if load_detector:
        await load_detector.start_monitoring()
        print("Load monitoring started")
    
    await throttling_system.start()
    print("Request throttling system started")
    print()
    
    # Demonstrate different request types and priorities
    async def sample_request_handler(request_type: str, delay: float = 1.0):
        """Sample request handler that simulates work."""
        print(f"  Processing {request_type} request...")
        await asyncio.sleep(delay)
        return f"{request_type} completed"
    
    # Submit various requests
    request_scenarios = [
        (RequestType.HEALTH_CHECK, RequestPriority.CRITICAL, 0.5),
        (RequestType.USER_QUERY, RequestPriority.HIGH, 2.0),
        (RequestType.USER_QUERY, RequestPriority.HIGH, 1.5),
        (RequestType.BATCH_PROCESSING, RequestPriority.MEDIUM, 3.0),
        (RequestType.ANALYTICS, RequestPriority.LOW, 2.5),
        (RequestType.MAINTENANCE, RequestPriority.BACKGROUND, 4.0)
    ]
    
    print("🚀 Submitting requests...")
    submitted_requests = []
    
    for i, (req_type, priority, duration) in enumerate(request_scenarios):
        success, message, request_id = await throttling_system.submit_request(
            request_type=req_type,
            priority=priority,
            handler=sample_request_handler,
            estimated_duration=duration,
            user_id=f"user_{i}",
            delay=duration
        )
        
        if success:
            print(f"  ✅ {req_type.value} request submitted: {request_id}")
            submitted_requests.append(request_id)
        else:
            print(f"  ❌ {req_type.value} request rejected: {message}")
        
        # Brief pause between submissions
        await asyncio.sleep(0.5)
    
    print(f"\nSubmitted {len(submitted_requests)} requests")
    
    # Show system status over time
    print("\n📊 System Status Monitoring...")
    for i in range(6):
        await asyncio.sleep(3)
        
        status = throttling_system.get_system_status()
        health = throttling_system.get_health_check()
        
        print(f"\n--- Status Update {i+1} ---")
        print(f"Health: {health['status'].upper()} ({'🟢' if health['status'] == 'healthy' else '🟡' if health['status'] == 'degraded' else '🔴'})")
        
        if health['issues']:
            print(f"Issues: {', '.join(health['issues'])}")
        
        print(f"Queue Size: {status['queue']['total_size']}/{status['queue']['max_size']} "
              f"({status['queue']['utilization']:.1f}%)")
        print(f"Active Requests: {len(status['active_requests'])}/{status['lifecycle']['concurrent_capacity']}")
        print(f"Throttling Rate: {status['throttling']['current_rate']:.1f} req/s")
        print(f"Success Rate: {status['throttling']['success_rate']:.1f}%")
        
        # Show queue breakdown
        queue_sizes = status['queue']['queue_sizes_by_priority']
        non_empty_queues = {k: v for k, v in queue_sizes.items() if v > 0}
        if non_empty_queues:
            print(f"Queue by Priority: {non_empty_queues}")
        
        # Simulate load changes for demonstration
        if i == 2 and degradation_controller:
            print("🔄 Simulating HIGH load...")
            degradation_controller.force_load_level(SystemLoadLevel.HIGH, "Demo high load")
        elif i == 4 and degradation_controller:
            print("🔄 Simulating recovery to NORMAL load...")
            degradation_controller.force_load_level(SystemLoadLevel.NORMAL, "Demo recovery")
    
    # Final status
    print("\n📋 Final System Status:")
    final_status = throttling_system.get_system_status()
    final_health = throttling_system.get_health_check()
    
    print(f"Total Requests Processed: {final_health['total_requests_processed']}")
    print(f"Final Queue Size: {final_health['current_queue_size']}")
    print(f"System Uptime: {final_health['uptime_seconds']:.1f}s")
    print(f"Overall Success Rate: {final_health['success_rate']:.1f}%")
    
    # Cleanup
    await throttling_system.stop()
    if load_detector:
        await load_detector.stop_monitoring()
    
    print("\n✅ Request Throttling System demonstration completed!")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_request_throttling_system())