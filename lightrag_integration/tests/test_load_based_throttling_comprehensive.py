"""
Comprehensive Tests for Load-Based Request Throttling and Queuing System
=========================================================================

This test suite provides comprehensive coverage for all components of the
load-based request throttling and queuing system:

1. LoadBasedThrottler (token bucket rate limiting)
2. PriorityRequestQueue (intelligent queuing with anti-starvation)
3. AdaptiveConnectionPool (dynamic connection management)
4. RequestLifecycleManager (complete request flow control)
5. RequestThrottlingSystem (main orchestrator)
6. GracefulDegradationOrchestrator (integration layer)

Test Categories:
- Unit tests for individual components
- Integration tests for component interaction
- Performance tests under various load conditions
- Stress tests for extreme scenarios
- Production scenario simulation

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import pytest
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_based_request_throttling_system import (
    LoadBasedThrottler, PriorityRequestQueue, AdaptiveConnectionPool,
    RequestLifecycleManager, RequestThrottlingSystem,
    RequestType, RequestPriority, RequestMetadata,
    SystemLoadLevel
)

from graceful_degradation_integration import (
    GracefulDegradationOrchestrator, GracefulDegradationConfig
)


# ============================================================================
# TEST FIXTURES AND UTILITIES
# ============================================================================

@pytest.fixture
def mock_load_detector():
    """Mock load detector for testing."""
    detector = Mock()
    detector.add_load_change_callback = Mock()
    return detector


@pytest.fixture
def sample_request_metadata():
    """Sample request metadata for testing."""
    return RequestMetadata(
        request_id="test_request_001",
        request_type=RequestType.USER_QUERY,
        priority=RequestPriority.HIGH,
        created_at=datetime.now(),
        estimated_duration=2.0,
        user_id="test_user"
    )


@pytest.fixture
async def throttler(mock_load_detector):
    """Create a throttler for testing."""
    return LoadBasedThrottler(
        base_rate_per_second=10.0,
        burst_capacity=20,
        load_detector=mock_load_detector
    )


@pytest.fixture
async def request_queue(mock_load_detector):
    """Create a request queue for testing."""
    queue = PriorityRequestQueue(
        max_queue_size=100,
        starvation_threshold=60.0,
        load_detector=mock_load_detector
    )
    await queue.start_cleanup_task()
    yield queue
    await queue.stop_cleanup_task()


@pytest.fixture
async def connection_pool(mock_load_detector):
    """Create a connection pool for testing."""
    pool = AdaptiveConnectionPool(
        base_pool_size=10,
        max_pool_size=50,
        load_detector=mock_load_detector
    )
    yield pool
    await pool.close()


@pytest.fixture
async def throttling_system(mock_load_detector):
    """Create a complete throttling system for testing."""
    system = RequestThrottlingSystem(
        base_rate_per_second=10.0,
        burst_capacity=20,
        max_queue_size=100,
        max_concurrent_requests=10,
        load_detector=mock_load_detector
    )
    await system.start()
    yield system
    await system.stop()


# ============================================================================
# UNIT TESTS - LoadBasedThrottler
# ============================================================================

class TestLoadBasedThrottler:
    """Test the token bucket rate limiter."""
    
    @pytest.mark.asyncio
    async def test_throttler_initialization(self, mock_load_detector):
        """Test throttler initialization."""
        throttler = LoadBasedThrottler(
            base_rate_per_second=5.0,
            burst_capacity=10,
            load_detector=mock_load_detector
        )
        
        assert throttler.base_rate_per_second == 5.0
        assert throttler.burst_capacity == 10
        assert throttler.current_rate == 5.0
        assert throttler.tokens == 10.0
        assert throttler.current_load_level == SystemLoadLevel.NORMAL
    
    @pytest.mark.asyncio
    async def test_token_acquisition_success(self, throttler, sample_request_metadata):
        """Test successful token acquisition."""
        # Should succeed immediately with full bucket
        success = await throttler.acquire_token(
            sample_request_metadata,
            tokens_needed=1.0,
            timeout=1.0
        )
        
        assert success is True
        assert throttler.tokens < throttler.burst_capacity
    
    @pytest.mark.asyncio
    async def test_token_acquisition_priority_weighting(self, throttler):
        """Test that priority affects token requirements."""
        critical_request = RequestMetadata(
            request_id="critical_001",
            request_type=RequestType.HEALTH_CHECK,
            priority=RequestPriority.CRITICAL,
            created_at=datetime.now()
        )
        
        low_request = RequestMetadata(
            request_id="low_001",
            request_type=RequestType.ANALYTICS,
            priority=RequestPriority.LOW,
            created_at=datetime.now()
        )
        
        # Drain most tokens
        throttler.tokens = 1.0
        
        # Critical request should succeed
        success_critical = await throttler.acquire_token(critical_request, timeout=0.1)
        assert success_critical is True
        
        # Low priority request should fail (needs more tokens due to weighting)
        success_low = await throttler.acquire_token(low_request, timeout=0.1)
        assert success_low is False
    
    @pytest.mark.asyncio
    async def test_rate_adjustment_on_load_change(self, throttler):
        """Test rate adjustment when load level changes."""
        # Simulate load change callback
        mock_metrics = Mock()
        mock_metrics.load_level = SystemLoadLevel.HIGH
        
        throttler._on_load_change(mock_metrics)
        
        # Rate should be reduced for HIGH load
        expected_rate = throttler.base_rate_per_second * throttler.rate_factors[SystemLoadLevel.HIGH]
        assert throttler.current_rate == expected_rate
        assert throttler.current_load_level == SystemLoadLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_token_refill_mechanism(self, throttler):
        """Test token bucket refill over time."""
        # Drain tokens
        throttler.tokens = 0.0
        initial_time = time.time()
        throttler.last_refill = initial_time
        
        # Simulate time passage
        throttler.last_refill = initial_time - 1.0  # 1 second ago
        
        # Refill should add tokens based on rate and elapsed time
        throttler._refill_tokens()
        
        expected_tokens = min(throttler.burst_capacity, throttler.current_rate * 1.0)
        assert throttler.tokens == expected_tokens
    
    @pytest.mark.asyncio
    async def test_throttler_statistics(self, throttler, sample_request_metadata):
        """Test statistics collection."""
        # Acquire some tokens
        await throttler.acquire_token(sample_request_metadata)
        await throttler.acquire_token(sample_request_metadata)
        
        stats = throttler.get_statistics()
        
        assert stats['total_requests'] == 2
        assert stats['allowed_requests'] == 2
        assert stats['denied_requests'] == 0
        assert stats['success_rate'] == 100.0
        assert 'current_rate' in stats
        assert 'available_tokens' in stats


# ============================================================================
# UNIT TESTS - PriorityRequestQueue
# ============================================================================

class TestPriorityRequestQueue:
    """Test the priority-based request queue."""
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self, mock_load_detector):
        """Test queue initialization."""
        queue = PriorityRequestQueue(
            max_queue_size=50,
            starvation_threshold=120.0,
            load_detector=mock_load_detector
        )
        
        assert queue.max_queue_size == 50
        assert queue.starvation_threshold == 120.0
        assert queue.current_max_size == 50
        assert queue.current_load_level == SystemLoadLevel.NORMAL
    
    @pytest.mark.asyncio
    async def test_request_enqueue_success(self, request_queue):
        """Test successful request enqueueing."""
        async def dummy_handler():
            return "success"
        
        metadata = RequestMetadata(
            request_id="enqueue_test_001",
            request_type=RequestType.USER_QUERY,
            priority=RequestPriority.HIGH,
            created_at=datetime.now()
        )
        
        success = await request_queue.enqueue(metadata, dummy_handler)
        assert success is True
        
        status = request_queue.get_queue_status()
        assert status['total_size'] == 1
        assert status['queue_sizes_by_priority']['HIGH'] == 1
    
    @pytest.mark.asyncio
    async def test_queue_capacity_enforcement(self, mock_load_detector):
        """Test queue capacity enforcement."""
        small_queue = PriorityRequestQueue(
            max_queue_size=2,
            load_detector=mock_load_detector
        )
        
        async def dummy_handler():
            return "success"
        
        # Fill queue to capacity
        for i in range(2):
            metadata = RequestMetadata(
                request_id=f"capacity_test_{i}",
                request_type=RequestType.USER_QUERY,
                priority=RequestPriority.MEDIUM,
                created_at=datetime.now()
            )
            success = await small_queue.enqueue(metadata, dummy_handler)
            assert success is True
        
        # Next request should be rejected
        metadata = RequestMetadata(
            request_id="capacity_test_overflow",
            request_type=RequestType.USER_QUERY,
            priority=RequestPriority.MEDIUM,
            created_at=datetime.now()
        )
        success = await small_queue.enqueue(metadata, dummy_handler)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, request_queue):
        """Test that higher priority requests are dequeued first."""
        async def dummy_handler():
            return "success"
        
        # Enqueue requests with different priorities
        priorities = [RequestPriority.LOW, RequestPriority.CRITICAL, RequestPriority.MEDIUM]
        
        for i, priority in enumerate(priorities):
            metadata = RequestMetadata(
                request_id=f"priority_test_{i}",
                request_type=RequestType.USER_QUERY,
                priority=priority,
                created_at=datetime.now()
            )
            await request_queue.enqueue(metadata, dummy_handler)
        
        # Dequeue and verify order (CRITICAL should come first)
        first_item = await request_queue.dequeue()
        assert first_item is not None
        assert first_item[0].priority == RequestPriority.CRITICAL
        
        second_item = await request_queue.dequeue()
        assert second_item is not None
        assert second_item[0].priority == RequestPriority.MEDIUM
        
        third_item = await request_queue.dequeue()
        assert third_item is not None
        assert third_item[0].priority == RequestPriority.LOW
    
    @pytest.mark.asyncio
    async def test_expired_request_cleanup(self, request_queue):
        """Test cleanup of expired requests."""
        async def dummy_handler():
            return "success"
        
        # Create an expired request
        expired_metadata = RequestMetadata(
            request_id="expired_test",
            request_type=RequestType.USER_QUERY,
            priority=RequestPriority.MEDIUM,
            created_at=datetime.now() - timedelta(hours=1),
            deadline=datetime.now() - timedelta(minutes=30)
        )
        
        # Enqueue should reject expired request
        success = await request_queue.enqueue(expired_metadata, dummy_handler)
        assert success is False
        
        # Verify no items in queue
        status = request_queue.get_queue_status()
        assert status['total_size'] == 0
    
    @pytest.mark.asyncio
    async def test_load_level_queue_limit_adjustment(self, request_queue):
        """Test queue limit adjustment based on load level."""
        # Simulate HIGH load level
        mock_metrics = Mock()
        mock_metrics.load_level = SystemLoadLevel.HIGH
        
        request_queue._on_load_change(mock_metrics)
        
        # Queue limit should be reduced
        expected_limit = int(request_queue.max_queue_size * 0.6)  # HIGH load factor
        assert request_queue.current_max_size == expected_limit
        assert request_queue.current_load_level == SystemLoadLevel.HIGH


# ============================================================================
# UNIT TESTS - AdaptiveConnectionPool
# ============================================================================

class TestAdaptiveConnectionPool:
    """Test the adaptive connection pool."""
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, mock_load_detector):
        """Test connection pool initialization."""
        pool = AdaptiveConnectionPool(
            base_pool_size=15,
            max_pool_size=75,
            min_pool_size=3,
            load_detector=mock_load_detector
        )
        
        assert pool.base_pool_size == 15
        assert pool.max_pool_size == 75
        assert pool.min_pool_size == 3
        assert pool.current_pool_size == 15
        assert pool.current_load_level == SystemLoadLevel.NORMAL
        
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_pool_size_adjustment(self, connection_pool):
        """Test pool size adjustment based on load level."""
        # Test adjustment to CRITICAL load
        mock_metrics = Mock()
        mock_metrics.load_level = SystemLoadLevel.CRITICAL
        
        connection_pool._on_load_change(mock_metrics)
        
        # Pool size should be reduced for CRITICAL load
        expected_size = int(connection_pool.base_pool_size * 0.5)  # CRITICAL load factor
        assert connection_pool.current_pool_size == expected_size
        assert connection_pool.current_load_level == SystemLoadLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_session_creation(self, connection_pool):
        """Test client session creation."""
        session = await connection_pool.get_session()
        
        assert session is not None
        assert not session.closed
        
        # Should return same session on subsequent calls
        session2 = await connection_pool.get_session()
        assert session is session2
    
    @pytest.mark.asyncio
    async def test_pool_status(self, connection_pool):
        """Test pool status reporting."""
        status = connection_pool.get_pool_status()
        
        assert 'current_pool_size' in status
        assert 'max_pool_size' in status
        assert 'min_pool_size' in status
        assert 'load_level' in status
        assert 'statistics' in status
        
        assert status['current_pool_size'] == connection_pool.current_pool_size
        assert status['load_level'] == connection_pool.current_load_level.name


# ============================================================================
# INTEGRATION TESTS - RequestThrottlingSystem
# ============================================================================

class TestRequestThrottlingSystem:
    """Test the complete request throttling system."""
    
    @pytest.mark.asyncio
    async def test_system_initialization_and_startup(self, mock_load_detector):
        """Test system initialization and startup."""
        system = RequestThrottlingSystem(
            base_rate_per_second=5.0,
            max_queue_size=50,
            max_concurrent_requests=10,
            load_detector=mock_load_detector
        )
        
        assert system.throttler is not None
        assert system.queue is not None
        assert system.connection_pool is not None
        assert system.lifecycle_manager is not None
        
        # Start system
        await system.start()
        assert system._running is True
        
        # Stop system
        await system.stop()
        assert system._running is False
    
    @pytest.mark.asyncio
    async def test_request_submission_and_processing(self, throttling_system):
        """Test request submission and processing."""
        async def test_handler(message: str):
            await asyncio.sleep(0.1)  # Simulate work
            return f"Processed: {message}"
        
        # Submit a request
        success, message, request_id = await throttling_system.submit_request(
            request_type=RequestType.USER_QUERY,
            priority=RequestPriority.HIGH,
            handler=test_handler,
            estimated_duration=1.0,
            message="Test query"
        )
        
        assert success is True
        assert "accepted" in message.lower()
        assert request_id != ""
        
        # Give some time for processing
        await asyncio.sleep(0.5)
        
        # Check system status
        status = throttling_system.get_system_status()
        assert status['system_running'] is True
        assert status['lifecycle']['total_requests'] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_request_types(self, throttling_system):
        """Test handling multiple request types with different priorities."""
        async def test_handler(req_type: str):
            await asyncio.sleep(0.1)
            return f"Completed {req_type}"
        
        request_types = [
            (RequestType.HEALTH_CHECK, RequestPriority.CRITICAL),
            (RequestType.USER_QUERY, RequestPriority.HIGH),
            (RequestType.BATCH_PROCESSING, RequestPriority.MEDIUM),
            (RequestType.ANALYTICS, RequestPriority.LOW),
            (RequestType.MAINTENANCE, RequestPriority.BACKGROUND)
        ]
        
        submitted_requests = []
        for req_type, priority in request_types:
            success, message, request_id = await throttling_system.submit_request(
                request_type=req_type,
                priority=priority,
                handler=test_handler,
                req_type=req_type.value
            )
            
            if success:
                submitted_requests.append(request_id)
        
        # Should have submitted all requests
        assert len(submitted_requests) == len(request_types)
        
        # Give time for processing
        await asyncio.sleep(1.0)
        
        # Check that requests were processed
        status = throttling_system.get_system_status()
        assert status['lifecycle']['total_requests'] == len(request_types)
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, throttling_system):
        """Test system health check functionality."""
        health = throttling_system.get_health_check()
        
        assert 'status' in health
        assert 'issues' in health
        assert 'uptime_seconds' in health
        assert 'total_requests_processed' in health
        assert 'current_queue_size' in health
        assert 'throttling_rate' in health
        assert 'success_rate' in health
        
        # Should be healthy initially
        assert health['status'] in ['healthy', 'degraded']  # Could be degraded if no requests processed yet
    
    @pytest.mark.asyncio
    async def test_load_level_impact_on_throttling(self, throttling_system):
        """Test that load level changes affect throttling behavior."""
        # Get initial throttling rate
        initial_status = throttling_system.get_system_status()
        initial_rate = initial_status['throttling']['current_rate']
        
        # Simulate HIGH load level change
        mock_metrics = Mock()
        mock_metrics.load_level = SystemLoadLevel.HIGH
        
        throttling_system.throttler._on_load_change(mock_metrics)
        
        # Check that rate was adjusted
        updated_status = throttling_system.get_system_status()
        updated_rate = updated_status['throttling']['current_rate']
        
        # Rate should be reduced for HIGH load
        assert updated_rate < initial_rate


# ============================================================================
# INTEGRATION TESTS - GracefulDegradationOrchestrator
# ============================================================================

class TestGracefulDegradationOrchestrator:
    """Test the complete graceful degradation integration."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = GracefulDegradationConfig(
            monitoring_interval=1.0,
            base_rate_per_second=5.0,
            max_queue_size=50
        )
        
        orchestrator = GracefulDegradationOrchestrator(config=config)
        
        assert orchestrator.config.monitoring_interval == 1.0
        assert orchestrator.config.base_rate_per_second == 5.0
        assert orchestrator.config.max_queue_size == 50
        assert orchestrator._integration_status is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_startup_and_shutdown(self):
        """Test orchestrator startup and shutdown."""
        config = GracefulDegradationConfig(
            auto_start_monitoring=True,
            base_rate_per_second=2.0,  # Low rate for testing
            max_queue_size=20
        )
        
        orchestrator = GracefulDegradationOrchestrator(config=config)
        
        # Start the system
        await orchestrator.start()
        assert orchestrator._running is True
        
        # Check status
        status = orchestrator.get_system_status()
        assert status['running'] is True
        assert status['integration_status']['throttling_system_active'] is True
        
        # Stop the system
        await orchestrator.stop()
        assert orchestrator._running is False
    
    @pytest.mark.asyncio
    async def test_integrated_request_processing(self):
        """Test request processing through the integrated system."""
        config = GracefulDegradationConfig(
            base_rate_per_second=10.0,
            max_queue_size=50,
            max_concurrent_requests=5
        )
        
        orchestrator = GracefulDegradationOrchestrator(config=config)
        await orchestrator.start()
        
        try:
            async def test_handler(message: str):
                await asyncio.sleep(0.1)
                return f"Integrated processing: {message}"
            
            # Submit requests of different types
            request_scenarios = [
                ('health_check', 'critical'),
                ('user_query', 'high'),
                ('batch_processing', 'medium'),
                ('analytics', 'low')
            ]
            
            successful_submissions = 0
            for req_type, priority in request_scenarios:
                success, message, request_id = await orchestrator.submit_request(
                    request_type=req_type,
                    priority=priority,
                    handler=test_handler,
                    message=f"Test {req_type} request"
                )
                
                if success:
                    successful_submissions += 1
            
            assert successful_submissions == len(request_scenarios)
            
            # Allow processing time
            await asyncio.sleep(1.0)
            
            # Check final status
            health = orchestrator.get_health_check()
            assert health['total_requests_processed'] > 0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test health monitoring and reporting."""
        orchestrator = GracefulDegradationOrchestrator()
        await orchestrator.start()
        
        try:
            # Get initial health check
            health = orchestrator.get_health_check()
            
            assert 'status' in health
            assert 'issues' in health
            assert 'component_status' in health
            assert 'production_integration' in health
            
            # Health should be reasonable initially
            assert health['status'] in ['healthy', 'degraded']
            
            # Check component status
            components = health['component_status']
            assert 'throttling_system' in components
            
        finally:
            await orchestrator.stop()


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStressScenarios:
    """Test system behavior under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_high_request_volume_stress(self):
        """Test system behavior under high request volume."""
        config = GracefulDegradationConfig(
            base_rate_per_second=20.0,  # Higher rate for stress test
            max_queue_size=200,
            max_concurrent_requests=20
        )
        
        orchestrator = GracefulDegradationOrchestrator(config=config)
        await orchestrator.start()
        
        try:
            async def fast_handler(req_id: int):
                await asyncio.sleep(0.05)  # Fast processing
                return f"Completed {req_id}"
            
            # Submit many requests rapidly
            submit_tasks = []
            for i in range(100):  # 100 requests
                task = orchestrator.submit_request(
                    request_type='user_query',
                    priority='high',
                    handler=fast_handler,
                    req_id=i
                )
                submit_tasks.append(task)
            
            # Wait for all submissions
            results = await asyncio.gather(*submit_tasks, return_exceptions=True)
            
            # Count successful submissions
            successful = sum(1 for success, _, _ in results if success)
            
            # Should have reasonable success rate even under stress
            success_rate = successful / len(results)
            assert success_rate > 0.7  # At least 70% success rate
            
            # Allow processing time
            await asyncio.sleep(3.0)
            
            # Check system health after stress
            health = orchestrator.get_health_check()
            assert health['status'] in ['healthy', 'degraded']  # Should still be functional
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test system behavior when queue overflows."""
        config = GracefulDegradationConfig(
            base_rate_per_second=1.0,  # Very low rate to cause backup
            max_queue_size=10,  # Small queue
            max_concurrent_requests=1  # Single concurrent request
        )
        
        orchestrator = GracefulDegradationOrchestrator(config=config)
        await orchestrator.start()
        
        try:
            async def slow_handler(req_id: int):
                await asyncio.sleep(1.0)  # Slow processing to cause backup
                return f"Slow completed {req_id}"
            
            # Submit more requests than queue can handle
            results = []
            for i in range(20):  # More than queue capacity
                success, message, request_id = await orchestrator.submit_request(
                    request_type='user_query',
                    handler=slow_handler,
                    req_id=i
                )
                results.append((success, message, request_id))
            
            # Some requests should be rejected due to queue overflow
            successful = sum(1 for success, _, _ in results if success)
            rejected = sum(1 for success, _, _ in results if not success)
            
            assert rejected > 0  # Some requests should be rejected
            assert successful <= config.max_queue_size + config.max_concurrent_requests
            
            # System should still be healthy
            health = orchestrator.get_health_check()
            assert health['status'] in ['healthy', 'degraded']
            
        finally:
            await orchestrator.stop()


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for the throttling system."""
    
    @pytest.mark.asyncio
    async def test_token_acquisition_performance(self):
        """Benchmark token acquisition performance."""
        throttler = LoadBasedThrottler(
            base_rate_per_second=100.0,
            burst_capacity=200
        )
        
        metadata = RequestMetadata(
            request_id="perf_test",
            request_type=RequestType.USER_QUERY,
            priority=RequestPriority.MEDIUM,
            created_at=datetime.now()
        )
        
        # Benchmark token acquisition
        iterations = 1000
        start_time = time.time()
        
        successful_acquisitions = 0
        for _ in range(iterations):
            success = await throttler.acquire_token(metadata, timeout=0.1)
            if success:
                successful_acquisitions += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should be able to process many token requests quickly
        requests_per_second = iterations / elapsed
        assert requests_per_second > 100  # At least 100 token requests per second
        
        # Should have reasonable success rate
        success_rate = successful_acquisitions / iterations
        assert success_rate > 0.5  # At least 50% success (limited by burst capacity)
    
    @pytest.mark.asyncio
    async def test_queue_operation_performance(self):
        """Benchmark queue operation performance."""
        queue = PriorityRequestQueue(max_queue_size=1000)
        await queue.start_cleanup_task()
        
        try:
            async def dummy_handler():
                return "success"
            
            # Benchmark enqueue operations
            enqueue_iterations = 500
            start_time = time.time()
            
            for i in range(enqueue_iterations):
                metadata = RequestMetadata(
                    request_id=f"perf_enqueue_{i}",
                    request_type=RequestType.USER_QUERY,
                    priority=RequestPriority.MEDIUM,
                    created_at=datetime.now()
                )
                await queue.enqueue(metadata, dummy_handler)
            
            enqueue_time = time.time() - start_time
            
            # Benchmark dequeue operations
            start_time = time.time()
            dequeued = 0
            
            for _ in range(enqueue_iterations):
                item = await queue.dequeue()
                if item is not None:
                    dequeued += 1
            
            dequeue_time = time.time() - start_time
            
            # Performance assertions
            enqueue_rate = enqueue_iterations / enqueue_time
            dequeue_rate = dequeued / dequeue_time
            
            assert enqueue_rate > 100  # At least 100 enqueues per second
            assert dequeue_rate > 100  # At least 100 dequeues per second
            assert dequeued == enqueue_iterations  # Should dequeue all items
            
        finally:
            await queue.stop_cleanup_task()


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])