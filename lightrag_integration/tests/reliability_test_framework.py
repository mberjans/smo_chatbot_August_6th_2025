#!/usr/bin/env python3
"""
Reliability Test Framework for Clinical Metabolomics Oracle
==========================================================

This module provides the core framework for executing comprehensive reliability
validation tests as defined in CMO-LIGHTRAG-014-T08.

Features:
- Test orchestration and isolation
- Failure injection mechanisms
- Performance monitoring during tests
- Metrics collection and analysis
- Result reporting and validation

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import logging
import time
import threading
import psutil
import json
import statistics
import traceback
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from enum import Enum, IntEnum
import weakref
import random

# Import existing system components
try:
    from ..graceful_degradation_integration import (
        GracefulDegradationOrchestrator, 
        GracefulDegradationConfig,
        create_graceful_degradation_system
    )
    from ..enhanced_load_monitoring_system import SystemLoadLevel
    GRACEFUL_DEGRADATION_AVAILABLE = True
except ImportError:
    GRACEFUL_DEGRADATION_AVAILABLE = False
    logging.warning("Graceful degradation system not available - using mock implementations")
    
    class SystemLoadLevel(IntEnum):
        NORMAL = 0
        ELEVATED = 1
        HIGH = 2
        CRITICAL = 3
        EMERGENCY = 4

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CORE RELIABILITY METRICS AND DATA STRUCTURES
# ============================================================================

@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability metrics for validation."""
    
    # Performance metrics
    success_rate: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    p99_9_response_time: float = 0.0
    throughput_rps: float = 0.0
    
    # Resource utilization
    peak_memory_usage: float = 0.0
    avg_cpu_utilization: float = 0.0
    max_queue_depth: int = 0
    avg_queue_depth: float = 0.0
    
    # Reliability indicators
    mtbf_hours: float = 0.0  # Mean Time Between Failures
    mttr_seconds: float = 0.0  # Mean Time To Recovery
    availability_percentage: float = 0.0
    
    # System behavior
    fallback_usage_distribution: Dict[str, float] = field(default_factory=dict)
    circuit_breaker_activations: int = 0
    load_level_transitions: Dict[str, int] = field(default_factory=dict)
    
    # Error analysis
    error_rate_by_category: Dict[str, float] = field(default_factory=dict)
    recovery_success_rate: float = 0.0
    cascade_prevention_score: float = 0.0
    
    # Test metadata
    test_duration_seconds: float = 0.0
    test_start_time: datetime = field(default_factory=datetime.now)
    test_end_time: Optional[datetime] = None


@dataclass
class TestResult:
    """Individual test result container."""
    
    test_name: str
    category: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    duration: float
    metrics: ReliabilityMetrics
    details: str = ""
    error: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass  
class ReliabilityTestConfig:
    """Configuration for reliability testing."""
    
    # Test execution settings
    max_test_duration_minutes: int = 30
    isolation_recovery_time_seconds: int = 10
    monitoring_interval_seconds: float = 1.0
    
    # System thresholds
    min_success_rate: float = 0.85
    max_response_time_ms: float = 5000.0
    max_memory_usage_percentage: float = 0.90
    max_cpu_usage_percentage: float = 0.90
    
    # Failure injection settings
    failure_injection_enabled: bool = True
    max_concurrent_failures: int = 2
    failure_duration_range: Tuple[int, int] = (10, 120)  # seconds
    
    # Load testing settings
    base_rps: float = 10.0
    max_rps: float = 1000.0
    ramp_up_time_seconds: int = 60


# ============================================================================
# FAILURE INJECTION MECHANISMS
# ============================================================================

class FailureInjector:
    """Base class for failure injection mechanisms."""
    
    def __init__(self, name: str):
        self.name = name
        self.active = False
        self.start_time = None
        self.end_time = None
        
    async def inject_failure(self):
        """Inject the failure condition."""
        self.active = True
        self.start_time = time.time()
        logger.info(f"Injecting failure: {self.name}")
        
    async def restore_normal(self):
        """Restore normal operation."""
        self.active = False
        self.end_time = time.time()
        logger.info(f"Restoring normal operation: {self.name}")
        
    @asynccontextmanager
    async def failure_context(self):
        """Context manager for temporary failure injection."""
        try:
            await self.inject_failure()
            yield self
        finally:
            await self.restore_normal()


class ServiceOutageInjector(FailureInjector):
    """Simulates complete service outages."""
    
    def __init__(self, service_name: str, orchestrator=None):
        super().__init__(f"service_outage_{service_name}")
        self.service_name = service_name
        self.orchestrator = orchestrator
        self.original_handler = None
        
    async def inject_failure(self):
        await super().inject_failure()
        
        # Mock the service to always fail
        if self.orchestrator and hasattr(self.orchestrator, '_mock_service_failure'):
            await self.orchestrator._mock_service_failure(self.service_name, failure_rate=1.0)
        
    async def restore_normal(self):
        await super().restore_normal()
        
        # Restore normal service behavior
        if self.orchestrator and hasattr(self.orchestrator, '_restore_service_health'):
            await self.orchestrator._restore_service_health(self.service_name)


class NetworkLatencyInjector(FailureInjector):
    """Simulates network latency conditions."""
    
    def __init__(self, delay_ms: int, jitter_ms: int = 0):
        super().__init__(f"network_latency_{delay_ms}ms")
        self.delay_ms = delay_ms
        self.jitter_ms = jitter_ms
        self.original_network_delay = 0
        
    async def inject_failure(self):
        await super().inject_failure()
        # In a real implementation, this would configure network delays
        # For testing, we'll simulate this through request delays
        logger.info(f"Simulating network latency: {self.delay_ms}ms ± {self.jitter_ms}ms")
        
    async def restore_normal(self):
        await super().restore_normal()
        logger.info("Restoring normal network conditions")


class MemoryPressureInjector(FailureInjector):
    """Simulates memory pressure conditions."""
    
    def __init__(self, target_usage_percentage: float):
        super().__init__(f"memory_pressure_{target_usage_percentage:.0f}%")
        self.target_usage = target_usage_percentage
        self.memory_consumer = []
        self.consumer_thread = None
        
    async def inject_failure(self):
        await super().inject_failure()
        
        # Calculate how much memory to consume
        total_memory = psutil.virtual_memory().total
        current_usage = psutil.virtual_memory().percent / 100.0
        
        if current_usage < self.target_usage:
            target_bytes = int(total_memory * (self.target_usage - current_usage))
            
            # Consume memory gradually
            await self._consume_memory(target_bytes)
            
    async def _consume_memory(self, target_bytes: int):
        """Gradually consume memory to reach target usage."""
        chunk_size = min(target_bytes // 10, 100 * 1024 * 1024)  # 100MB max chunks
        
        try:
            while len(self.memory_consumer) * chunk_size < target_bytes and self.active:
                chunk = bytearray(chunk_size)
                self.memory_consumer.append(chunk)
                await asyncio.sleep(0.1)  # Small delay between allocations
                
        except MemoryError:
            logger.warning("Memory allocation failed - reached system limits")
            
    async def restore_normal(self):
        await super().restore_normal()
        
        # Release consumed memory
        self.memory_consumer.clear()
        
        # Force garbage collection
        import gc
        gc.collect()


# ============================================================================
# PERFORMANCE MONITORING AND METRICS COLLECTION
# ============================================================================

class ReliabilityTestMonitor:
    """Comprehensive monitoring during reliability tests."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.monitoring = False
        self.monitor_task = None
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k data points
        self.start_time = None
        
    async def start_monitoring(self):
        """Start continuous monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started reliability test monitoring")
        
    async def stop_monitoring(self):
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Stopped reliability test monitoring")
        return self._calculate_aggregate_metrics()
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            logger.debug("Monitoring loop cancelled")
            
    async def _collect_system_metrics(self):
        """Collect current system metrics."""
        timestamp = time.time()
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        # Network metrics (if available)
        try:
            network_io = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
        except:
            network_io = None
            network_connections = 0
            
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_available_gb': memory_info.available / (1024**3),
            'network_connections': network_connections,
            'network_bytes_sent': network_io.bytes_sent if network_io else 0,
            'network_bytes_recv': network_io.bytes_recv if network_io else 0,
        }
        
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics from monitoring history."""
        if not self.metrics_history:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_percent'] for m in self.metrics_history]
        connection_values = [m['network_connections'] for m in self.metrics_history]
        
        return {
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': statistics.mean(memory_values),
            'max_memory_percent': max(memory_values),
            'avg_network_connections': statistics.mean(connection_values),
            'max_network_connections': max(connection_values),
            'sample_count': len(self.metrics_history)
        }


class LoadGenerator:
    """Generate realistic load patterns for testing."""
    
    def __init__(self, target_rps: float, duration: int, pattern: str = 'constant'):
        self.target_rps = target_rps
        self.duration = duration
        self.pattern = pattern
        self.active = False
        self.results = []
        
    async def run(self, orchestrator) -> List[Dict]:
        """Run load generation against the orchestrator."""
        self.active = True
        self.results = []
        
        start_time = time.time()
        end_time = start_time + self.duration
        
        # Calculate request interval
        base_interval = 1.0 / self.target_rps if self.target_rps > 0 else 1.0
        
        request_tasks = []
        request_count = 0
        
        try:
            while time.time() < end_time and self.active:
                
                # Adjust RPS based on pattern
                current_rps = self._calculate_current_rps(
                    time.time() - start_time, 
                    self.duration
                )
                current_interval = 1.0 / current_rps if current_rps > 0 else 1.0
                
                # Create and submit request
                request_task = self._create_load_request(orchestrator, request_count)
                request_tasks.append(request_task)
                request_count += 1
                
                # Wait for next request
                await asyncio.sleep(current_interval)
                
                # Clean up completed tasks periodically
                if len(request_tasks) > 100:
                    completed_tasks = [t for t in request_tasks if t.done()]
                    for task in completed_tasks:
                        try:
                            result = await task
                            self.results.append(result)
                        except Exception as e:
                            self.results.append({
                                'success': False,
                                'error': str(e),
                                'timestamp': time.time()
                            })
                    
                    request_tasks = [t for t in request_tasks if not t.done()]
            
            # Wait for remaining tasks to complete
            if request_tasks:
                remaining_results = await asyncio.gather(*request_tasks, return_exceptions=True)
                
                for result in remaining_results:
                    if isinstance(result, Exception):
                        self.results.append({
                            'success': False,
                            'error': str(result),
                            'timestamp': time.time()
                        })
                    else:
                        self.results.append(result)
                        
        finally:
            self.active = False
            
        return self.results
        
    def _calculate_current_rps(self, elapsed_time: float, total_duration: float) -> float:
        """Calculate current RPS based on load pattern."""
        if self.pattern == 'constant':
            return self.target_rps
        elif self.pattern == 'ramp_up':
            progress = min(elapsed_time / total_duration, 1.0)
            return self.target_rps * progress
        elif self.pattern == 'spike':
            # Spike pattern: low -> high -> low
            progress = elapsed_time / total_duration
            if progress < 0.3:
                return self.target_rps * 0.2
            elif progress < 0.7:
                return self.target_rps
            else:
                return self.target_rps * 0.3
        else:
            return self.target_rps
            
    async def _create_load_request(self, orchestrator, request_id: int):
        """Create and execute a single load request."""
        start_time = time.time()
        
        try:
            # Create realistic handler
            async def test_handler():
                # Simulate variable processing time
                processing_time = random.uniform(0.1, 0.5)  # 100-500ms
                await asyncio.sleep(processing_time)
                return f"Response for request {request_id}"
            
            # Submit request
            success, message, response_id = await orchestrator.submit_request(
                request_type='user_query',
                priority='medium',
                handler=test_handler,
                timeout=10.0
            )
            
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'success': success,
                'response_time': end_time - start_time,
                'message': message,
                'timestamp': start_time
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'request_id': request_id,
                'success': False,
                'response_time': end_time - start_time,
                'error': str(e),
                'timestamp': start_time
            }


# ============================================================================
# TEST UTILITIES AND HELPERS
# ============================================================================

class ReliabilityTestUtils:
    """Utility functions for reliability testing."""
    
    @staticmethod
    def calculate_success_rate(results: List[Dict]) -> float:
        """Calculate success rate from test results."""
        if not results:
            return 0.0
            
        successful = sum(1 for r in results if r.get('success', False))
        return successful / len(results)
        
    @staticmethod
    def calculate_response_time_percentiles(results: List[Dict]) -> Dict[str, float]:
        """Calculate response time percentiles."""
        response_times = [
            r['response_time'] * 1000  # Convert to milliseconds
            for r in results 
            if 'response_time' in r and r.get('success', False)
        ]
        
        if not response_times:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p99_9': 0}
            
        response_times.sort()
        
        def percentile(data, p):
            index = int(len(data) * p / 100)
            if index >= len(data):
                index = len(data) - 1
            return data[index]
            
        return {
            'p50': percentile(response_times, 50),
            'p95': percentile(response_times, 95),
            'p99': percentile(response_times, 99),
            'p99_9': percentile(response_times, 99.9)
        }
        
    @staticmethod
    def calculate_throughput(results: List[Dict], duration: float) -> float:
        """Calculate throughput in requests per second."""
        if duration <= 0:
            return 0.0
            
        successful_requests = sum(1 for r in results if r.get('success', False))
        return successful_requests / duration
        
    @staticmethod
    def analyze_error_distribution(results: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of error types."""
        error_counts = defaultdict(int)
        
        for result in results:
            if not result.get('success', False):
                error_type = result.get('error', 'unknown_error')
                error_counts[error_type] += 1
                
        return dict(error_counts)


# ============================================================================
# ORCHESTRATOR CREATION AND CONFIGURATION
# ============================================================================

async def create_test_orchestrator(config: Optional[ReliabilityTestConfig] = None) -> Any:
    """Create a test orchestrator for reliability testing."""
    
    if config is None:
        config = ReliabilityTestConfig()
    
    if GRACEFUL_DEGRADATION_AVAILABLE:
        # Use real graceful degradation system
        orchestrator_config = GracefulDegradationConfig(
            monitoring_interval=config.monitoring_interval_seconds,
            base_rate_per_second=config.base_rps,
            max_queue_size=200,
            max_concurrent_requests=50
        )
        
        orchestrator = GracefulDegradationOrchestrator(config=orchestrator_config)
        
        # Add test-specific methods for failure injection
        orchestrator._test_failure_injectors = {}
        orchestrator._mock_service_failure = _mock_service_failure
        orchestrator._restore_service_health = _restore_service_health
        
        return orchestrator
        
    else:
        # Use mock implementation for testing
        return MockOrchestrator(config)


async def _mock_service_failure(orchestrator, service_name: str, failure_rate: float = 1.0):
    """Mock service failure for testing purposes."""
    logger.info(f"Mocking {service_name} failure with rate {failure_rate}")
    # In real implementation, this would configure the orchestrator
    # to simulate service failures


async def _restore_service_health(orchestrator, service_name: str):
    """Restore service health after failure simulation."""
    logger.info(f"Restoring {service_name} service health")
    # In real implementation, this would restore normal service behavior


class MockOrchestrator:
    """Mock orchestrator for testing when real system isn't available."""
    
    def __init__(self, config: ReliabilityTestConfig):
        self.config = config
        self.request_count = 0
        self.start_time = time.time()
        self._running = False
        
    async def submit_request(self, request_type: str, priority: str, handler: Callable, **kwargs):
        """Mock request submission."""
        self.request_count += 1
        
        try:
            # Simulate processing
            start_time = time.time()
            result = await handler() if asyncio.iscoroutinefunction(handler) else handler()
            end_time = time.time()
            
            return (True, result, f"mock_request_{self.request_count}")
            
        except Exception as e:
            return (False, str(e), f"mock_request_{self.request_count}")
            
    async def start(self):
        """Start mock orchestrator."""
        self._running = True
        
    async def stop(self):
        """Stop mock orchestrator."""
        self._running = False
        
    def get_health_check(self):
        """Get mock health check."""
        return {
            'status': 'healthy',
            'uptime_seconds': time.time() - self.start_time,
            'total_requests_processed': self.request_count
        }
        
    def get_system_status(self):
        """Get mock system status."""
        return {
            'running': self._running,
            'current_load_level': SystemLoadLevel.NORMAL,
            'total_requests_processed': self.request_count
        }


# ============================================================================
# MAIN RELIABILITY TEST FRAMEWORK CLASS
# ============================================================================

class ReliabilityValidationFramework:
    """Main framework for executing reliability validation tests."""
    
    def __init__(self, config: Optional[ReliabilityTestConfig] = None):
        self.config = config or ReliabilityTestConfig()
        self.test_orchestrator = None
        self.monitoring_system = None
        self.test_results = {}
        
    async def setup_test_environment(self):
        """Initialize test environment with monitoring."""
        logger.info("Setting up reliability test environment")
        
        self.test_orchestrator = await create_test_orchestrator(self.config)
        self.monitoring_system = ReliabilityTestMonitor(
            collection_interval=self.config.monitoring_interval_seconds
        )
        
        if hasattr(self.test_orchestrator, 'start'):
            await self.test_orchestrator.start()
            
        logger.info("Reliability test environment ready")
        
    async def cleanup_test_environment(self):
        """Clean up test environment."""
        logger.info("Cleaning up reliability test environment")
        
        if self.monitoring_system:
            await self.monitoring_system.stop_monitoring()
            
        if self.test_orchestrator and hasattr(self.test_orchestrator, 'stop'):
            await self.test_orchestrator.stop()
            
        logger.info("Reliability test environment cleaned up")
        
    async def execute_monitored_test(
        self, 
        test_name: str, 
        test_func: Callable,
        category: str = "general"
    ) -> TestResult:
        """Execute a single test with comprehensive monitoring."""
        
        logger.info(f"Starting reliability test: {test_name}")
        start_time = time.time()
        
        # Start monitoring
        await self.monitoring_system.start_monitoring()
        
        try:
            # Execute the test
            test_start = time.time()
            await test_func(self.test_orchestrator, self.config)
            test_duration = time.time() - test_start
            
            # Stop monitoring and collect metrics
            monitoring_metrics = await self.monitoring_system.stop_monitoring()
            
            # Create reliability metrics
            reliability_metrics = ReliabilityMetrics(
                test_duration_seconds=test_duration,
                test_start_time=datetime.fromtimestamp(test_start),
                test_end_time=datetime.fromtimestamp(time.time()),
                avg_cpu_utilization=monitoring_metrics.get('avg_cpu_percent', 0) / 100.0,
                peak_memory_usage=monitoring_metrics.get('max_memory_percent', 0) / 100.0
            )
            
            return TestResult(
                test_name=test_name,
                category=category,
                status='passed',
                duration=test_duration,
                metrics=reliability_metrics,
                details=f"Test completed successfully in {test_duration:.2f}s"
            )
            
        except Exception as e:
            # Stop monitoring
            await self.monitoring_system.stop_monitoring()
            
            duration = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            return TestResult(
                test_name=test_name,
                category=category,
                status='failed',
                duration=duration,
                metrics=ReliabilityMetrics(test_duration_seconds=duration),
                error=error_msg,
                details="Test failed with exception"
            )


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

async def example_reliability_test(orchestrator, config: ReliabilityTestConfig):
    """Example reliability test implementation."""
    
    # Generate test load
    load_generator = LoadGenerator(
        target_rps=config.base_rps * 2,
        duration=30,  # 30 seconds
        pattern='constant'
    )
    
    # Run load test
    results = await load_generator.run(orchestrator)
    
    # Analyze results
    success_rate = ReliabilityTestUtils.calculate_success_rate(results)
    response_times = ReliabilityTestUtils.calculate_response_time_percentiles(results)
    throughput = ReliabilityTestUtils.calculate_throughput(results, 30)
    
    # Validate against thresholds
    assert success_rate >= config.min_success_rate, f"Success rate {success_rate:.2f} below threshold {config.min_success_rate}"
    assert response_times['p95'] <= config.max_response_time_ms, f"P95 response time {response_times['p95']:.0f}ms above threshold"
    
    logger.info(f"Example test completed - Success rate: {success_rate:.2f}, P95: {response_times['p95']:.0f}ms, Throughput: {throughput:.1f} RPS")


async def main():
    """Main function for testing the reliability framework."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        
        # Execute example test
        result = await framework.execute_monitored_test(
            test_name="example_reliability_test",
            test_func=example_reliability_test,
            category="framework_validation"
        )
        
        print(f"Test result: {result.status}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Details: {result.details}")
        
    finally:
        await framework.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(main())