#!/usr/bin/env python3
"""
Stress Testing Scenarios for Reliability Validation
==================================================

Implementation of stress testing scenarios ST-001 through ST-004 as defined in
CMO-LIGHTRAG-014-T08 reliability validation design.

Test Scenarios:
- ST-001: Progressive Load Escalation
- ST-002: Burst Load Handling  
- ST-003: Memory Pressure Endurance
- ST-004: Maximum Concurrent Request Handling

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import pytest
import logging
import time
import psutil
import statistics
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import reliability test framework
from .reliability_test_framework import (
    ReliabilityValidationFramework,
    ReliabilityTestConfig,
    LoadGenerator,
    MemoryPressureInjector,
    ReliabilityTestUtils,
    SystemLoadLevel,
    create_test_orchestrator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STRESS TEST UTILITIES AND HELPERS
# ============================================================================

@dataclass
class LoadEscalationPhase:
    """Configuration for a load escalation phase."""
    name: str
    target_rps: float
    duration_seconds: int
    expected_load_level: SystemLoadLevel
    min_success_rate: float = 0.85
    max_p95_response_time: float = 5000.0  # milliseconds


class ConcurrentRequestTester:
    """Utility for testing concurrent request handling."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.active_requests = 0
        self.max_concurrent_reached = 0
        
    async def generate_concurrent_requests(self, target_concurrent: int, duration: int):
        """Generate specified number of concurrent requests."""
        
        async def request_handler(request_id: int):
            """Handler for individual concurrent request."""
            self.active_requests += 1
            self.max_concurrent_reached = max(self.max_concurrent_reached, self.active_requests)
            
            try:
                # Simulate realistic processing complexity
                complexity = random.choice(['low', 'medium', 'high'])
                processing_time = {
                    'low': random.uniform(0.1, 0.3),
                    'medium': random.uniform(0.5, 1.5), 
                    'high': random.uniform(2.0, 4.0)
                }[complexity]
                
                await asyncio.sleep(processing_time)
                return f"Processed concurrent request {request_id}"
                
            finally:
                self.active_requests -= 1
        
        # Create all concurrent requests
        request_tasks = []
        start_time = time.time()
        
        for i in range(target_concurrent):
            task = self.orchestrator.submit_request(
                request_type='user_query',
                priority='medium',
                handler=lambda req_id=i: request_handler(req_id),
                timeout=30.0
            )
            request_tasks.append(task)
        
        # Wait for completion with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*request_tasks, return_exceptions=True),
                timeout=duration + 30
            )
            
            # Analyze results
            successful = 0
            failed = 0
            total_time = time.time() - start_time
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    success, message, request_id = result
                    if success:
                        successful += 1
                    else:
                        failed += 1
            
            return {
                'total_requests': target_concurrent,
                'successful_requests': successful,
                'failed_requests': failed,
                'success_rate': successful / target_concurrent,
                'total_duration': total_time,
                'max_concurrent_achieved': self.max_concurrent_reached
            }
            
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in request_tasks:
                if not task.done():
                    task.cancel()
                    
            return {
                'total_requests': target_concurrent,
                'successful_requests': 0,
                'failed_requests': target_concurrent,
                'success_rate': 0.0,
                'total_duration': duration + 30,
                'max_concurrent_achieved': self.max_concurrent_reached,
                'timeout': True
            }


class RealisticUserSessionSimulator:
    """Simulate realistic user interaction patterns for stress testing."""
    
    @staticmethod
    def generate_realistic_handler(complexity: str = 'medium'):
        """Generate realistic request handler with variable complexity."""
        
        async def handler():
            # Simulate different types of operations
            if complexity == 'low':
                # Simple query processing
                await asyncio.sleep(random.uniform(0.1, 0.3))
                return "Simple query result"
                
            elif complexity == 'medium':
                # Standard metabolomics analysis
                await asyncio.sleep(random.uniform(0.5, 1.5))
                # Simulate some CPU work
                data = [random.random() for _ in range(1000)]
                statistics.mean(data)
                return "Standard analysis complete"
                
            elif complexity == 'high':
                # Complex cross-document analysis
                await asyncio.sleep(random.uniform(2.0, 4.0))
                # Simulate more intensive CPU work
                data = [random.random() for _ in range(10000)]
                statistics.stdev(data)
                return "Complex analysis complete"
                
            else:  # 'mixed'
                # Random complexity
                sub_complexity = random.choice(['low', 'medium', 'high'])
                return await RealisticUserSessionSimulator.generate_realistic_handler(sub_complexity)()
        
        return handler


# ============================================================================
# ST-001: PROGRESSIVE LOAD ESCALATION TEST
# ============================================================================

async def test_progressive_load_escalation(orchestrator, config: ReliabilityTestConfig):
    """
    ST-001: Test system response to gradually increasing load levels.
    
    Validates that the system:
    - Maintains acceptable performance as load increases
    - Triggers appropriate load level transitions
    - Implements controlled response time degradation
    - Maintains minimum success rates at all levels
    """
    logger.info("Starting ST-001: Progressive Load Escalation Test")
    
    # Define escalation phases
    escalation_phases = [
        LoadEscalationPhase(
            name="baseline",
            target_rps=config.base_rps,
            duration_seconds=60,
            expected_load_level=SystemLoadLevel.NORMAL,
            min_success_rate=0.95,
            max_p95_response_time=2000.0
        ),
        LoadEscalationPhase(
            name="elevated",
            target_rps=config.base_rps * 5,
            duration_seconds=120,
            expected_load_level=SystemLoadLevel.ELEVATED,
            min_success_rate=0.90,
            max_p95_response_time=3000.0
        ),
        LoadEscalationPhase(
            name="high",
            target_rps=config.base_rps * 20,
            duration_seconds=180,
            expected_load_level=SystemLoadLevel.HIGH,
            min_success_rate=0.85,
            max_p95_response_time=4000.0
        ),
        LoadEscalationPhase(
            name="critical",
            target_rps=config.base_rps * 50,
            duration_seconds=240,
            expected_load_level=SystemLoadLevel.CRITICAL,
            min_success_rate=0.80,
            max_p95_response_time=5000.0
        )
    ]
    
    phase_results = {}
    baseline_metrics = None
    
    for phase in escalation_phases:
        logger.info(f"Executing escalation phase: {phase.name} ({phase.target_rps} RPS)")
        
        # Generate load for this phase
        load_generator = LoadGenerator(
            target_rps=phase.target_rps,
            duration=phase.duration_seconds,
            pattern='constant'
        )
        
        phase_start_time = time.time()
        
        # Execute load generation
        results = await load_generator.run(orchestrator)
        
        phase_duration = time.time() - phase_start_time
        
        # Analyze phase results
        success_rate = ReliabilityTestUtils.calculate_success_rate(results)
        response_times = ReliabilityTestUtils.calculate_response_time_percentiles(results)
        throughput = ReliabilityTestUtils.calculate_throughput(results, phase_duration)
        
        # Get system status
        system_status = orchestrator.get_system_status() if hasattr(orchestrator, 'get_system_status') else {}
        current_load_level = system_status.get('current_load_level', SystemLoadLevel.NORMAL)
        
        phase_metrics = {
            'success_rate': success_rate,
            'p95_response_time': response_times['p95'],
            'p99_response_time': response_times['p99'],
            'throughput_rps': throughput,
            'load_level': current_load_level,
            'total_requests': len(results),
            'failed_requests': len([r for r in results if not r.get('success', False)])
        }
        
        phase_results[phase.name] = phase_metrics
        
        # Store baseline for comparison
        if phase.name == "baseline":
            baseline_metrics = phase_metrics
        
        # Validate phase requirements
        assert success_rate >= phase.min_success_rate, \
            f"Phase {phase.name}: Success rate {success_rate:.2f} below minimum {phase.min_success_rate}"
        
        assert response_times['p95'] <= phase.max_p95_response_time, \
            f"Phase {phase.name}: P95 response time {response_times['p95']:.0f}ms exceeds maximum {phase.max_p95_response_time}ms"
        
        # Allow recovery time between phases
        await asyncio.sleep(10)
        
        logger.info(f"Phase {phase.name} completed: {success_rate:.2f} success rate, {response_times['p95']:.0f}ms P95")
    
    # Validate progressive behavior
    if baseline_metrics:
        # Response time degradation should be controlled
        critical_metrics = phase_results.get('critical', {})
        if critical_metrics:
            response_time_ratio = critical_metrics['p95_response_time'] / baseline_metrics['p95_response_time']
            assert response_time_ratio <= 4.0, \
                f"Response time degradation too severe: {response_time_ratio:.1f}x baseline"
        
        # Success rate degradation should be controlled
        if critical_metrics:
            success_rate_drop = baseline_metrics['success_rate'] - critical_metrics['success_rate']
            assert success_rate_drop <= 0.15, \
                f"Success rate drop too large: {success_rate_drop:.2f}"
    
    logger.info("ST-001: Progressive Load Escalation Test completed successfully")
    return phase_results


# ============================================================================
# ST-002: BURST LOAD HANDLING TEST  
# ============================================================================

async def test_burst_load_handling(orchestrator, config: ReliabilityTestConfig):
    """
    ST-002: Test system resilience to sudden traffic spikes.
    
    Validates that the system:
    - Handles sudden traffic bursts without crashing
    - Maintains acceptable performance during bursts
    - Recovers quickly after burst ends
    - Queue management prevents system overload
    """
    logger.info("Starting ST-002: Burst Load Handling Test")
    
    # Establish baseline load
    baseline_generator = LoadGenerator(
        target_rps=config.base_rps * 2,
        duration=30,
        pattern='constant'
    )
    
    baseline_results = await baseline_generator.run(orchestrator)
    baseline_metrics = {
        'success_rate': ReliabilityTestUtils.calculate_success_rate(baseline_results),
        'p95_response_time': ReliabilityTestUtils.calculate_response_time_percentiles(baseline_results)['p95']
    }
    
    logger.info(f"Baseline established: {baseline_metrics['success_rate']:.2f} success rate")
    
    # Define burst scenarios
    burst_scenarios = [
        {
            'name': 'short_intense_burst',
            'baseline_rps': config.base_rps * 2,
            'burst_rps': config.base_rps * 50,
            'burst_duration': 10,
            'recovery_duration': 30,
            'min_success_rate': 0.70,
            'max_recovery_time': 30
        },
        {
            'name': 'medium_burst',
            'baseline_rps': config.base_rps * 2,
            'burst_rps': config.base_rps * 100,
            'burst_duration': 5,
            'recovery_duration': 45,
            'min_success_rate': 0.65,
            'max_recovery_time': 45
        },
        {
            'name': 'sustained_burst',
            'baseline_rps': config.base_rps * 2,
            'burst_rps': config.base_rps * 20,
            'burst_duration': 60,
            'recovery_duration': 60,
            'min_success_rate': 0.75,
            'max_recovery_time': 60
        }
    ]
    
    burst_results = {}
    
    for scenario in burst_scenarios:
        logger.info(f"Executing burst scenario: {scenario['name']}")
        
        # Start baseline load
        baseline_task = asyncio.create_task(
            LoadGenerator(
                target_rps=scenario['baseline_rps'],
                duration=scenario['burst_duration'] + scenario['recovery_duration'] + 20,
                pattern='constant'
            ).run(orchestrator)
        )
        
        # Wait a moment for baseline to establish
        await asyncio.sleep(5)
        
        # Execute burst
        burst_start_time = time.time()
        burst_generator = LoadGenerator(
            target_rps=scenario['burst_rps'],
            duration=scenario['burst_duration'],
            pattern='constant'
        )
        
        burst_task = asyncio.create_task(burst_generator.run(orchestrator))
        
        # Monitor system during burst
        burst_system_metrics = []
        burst_end_time = burst_start_time + scenario['burst_duration']
        
        while time.time() < burst_end_time:
            if hasattr(orchestrator, 'get_health_check'):
                health = orchestrator.get_health_check()
                burst_system_metrics.append({
                    'timestamp': time.time(),
                    'health_status': health.get('status', 'unknown')
                })
            await asyncio.sleep(1)
        
        # Wait for burst completion
        burst_results_data = await burst_task
        burst_actual_end_time = time.time()
        
        # Measure recovery
        recovery_start_time = time.time()
        recovery_complete = False
        recovery_time = 0
        
        while time.time() - recovery_start_time < scenario['max_recovery_time']:
            if hasattr(orchestrator, 'get_system_status'):
                status = orchestrator.get_system_status()
                load_level = status.get('current_load_level', SystemLoadLevel.NORMAL)
                
                if load_level <= SystemLoadLevel.ELEVATED:
                    recovery_time = time.time() - recovery_start_time
                    recovery_complete = True
                    break
            
            await asyncio.sleep(2)
        
        # Cancel baseline task
        baseline_task.cancel()
        try:
            await baseline_task
        except asyncio.CancelledError:
            pass
        
        # Analyze burst performance
        burst_success_rate = ReliabilityTestUtils.calculate_success_rate(burst_results_data)
        burst_response_times = ReliabilityTestUtils.calculate_response_time_percentiles(burst_results_data)
        
        scenario_results = {
            'burst_success_rate': burst_success_rate,
            'burst_p95_response_time': burst_response_times['p95'],
            'recovery_time': recovery_time,
            'recovery_complete': recovery_complete,
            'system_stability': len([m for m in burst_system_metrics 
                                   if m['health_status'] != 'unhealthy']) / len(burst_system_metrics) if burst_system_metrics else 1.0
        }
        
        burst_results[scenario['name']] = scenario_results
        
        # Validate burst handling
        assert burst_success_rate >= scenario['min_success_rate'], \
            f"Burst {scenario['name']}: Success rate {burst_success_rate:.2f} below minimum {scenario['min_success_rate']}"
        
        assert recovery_time <= scenario['max_recovery_time'], \
            f"Burst {scenario['name']}: Recovery time {recovery_time:.1f}s exceeds maximum {scenario['max_recovery_time']}s"
        
        # Allow full recovery before next scenario
        await asyncio.sleep(30)
        
        logger.info(f"Burst {scenario['name']} completed: {burst_success_rate:.2f} success rate, {recovery_time:.1f}s recovery")
    
    logger.info("ST-002: Burst Load Handling Test completed successfully")
    return burst_results


# ============================================================================
# ST-003: MEMORY PRESSURE ENDURANCE TEST
# ============================================================================

async def test_memory_pressure_endurance(orchestrator, config: ReliabilityTestConfig):
    """
    ST-003: Test system endurance under sustained memory pressure.
    
    Validates that the system:
    - Operates correctly under high memory usage
    - Implements appropriate memory management
    - Avoids out-of-memory crashes
    - Maintains performance under memory constraints
    """
    logger.info("Starting ST-003: Memory Pressure Endurance Test")
    
    memory_pressure_levels = [0.60, 0.70, 0.75, 0.80, 0.85]
    pressure_results = {}
    
    for pressure_level in memory_pressure_levels:
        logger.info(f"Testing memory pressure level: {pressure_level:.0%}")
        
        # Create memory pressure injector
        pressure_injector = MemoryPressureInjector(pressure_level)
        
        try:
            # Inject memory pressure
            await pressure_injector.inject_failure()
            
            # Allow system to adapt to memory pressure
            await asyncio.sleep(10)
            
            # Run workload under memory pressure
            workload_generator = LoadGenerator(
                target_rps=config.base_rps * 5,
                duration=180,  # 3 minutes
                pattern='constant'
            )
            
            # Monitor memory during workload
            memory_monitoring_task = asyncio.create_task(
                monitor_memory_usage(duration=200)
            )
            
            # Execute workload
            workload_results = await workload_generator.run(orchestrator)
            
            # Stop memory monitoring
            memory_monitoring_task.cancel()
            try:
                memory_stats = await memory_monitoring_task
            except asyncio.CancelledError:
                memory_stats = {'peak_usage': psutil.virtual_memory().percent / 100.0}
            
            # Analyze results under memory pressure
            success_rate = ReliabilityTestUtils.calculate_success_rate(workload_results)
            response_times = ReliabilityTestUtils.calculate_response_time_percentiles(workload_results)
            
            # Get system health
            system_health = orchestrator.get_health_check() if hasattr(orchestrator, 'get_health_check') else {'status': 'unknown'}
            
            pressure_results[f"{pressure_level:.0%}"] = {
                'target_pressure': pressure_level,
                'actual_peak_memory': memory_stats.get('peak_usage', 0),
                'success_rate': success_rate,
                'p95_response_time': response_times['p95'],
                'system_health': system_health.get('status', 'unknown'),
                'total_requests': len(workload_results),
                'oom_events': 0  # Would be detected from logs in real implementation
            }
            
            # Validate memory pressure handling
            assert success_rate >= 0.80, \
                f"Memory pressure {pressure_level:.0%}: Success rate {success_rate:.2f} too low"
            
            assert memory_stats.get('peak_usage', 0) < 0.95, \
                f"Memory pressure {pressure_level:.0%}: Actual usage {memory_stats.get('peak_usage', 0):.0%} too high"
            
            logger.info(f"Memory pressure {pressure_level:.0%} completed: {success_rate:.2f} success rate")
            
        finally:
            # Restore normal memory conditions
            await pressure_injector.restore_normal()
            
            # Allow memory recovery
            await asyncio.sleep(15)
    
    # Test memory pressure with burst load
    logger.info("Testing memory pressure with burst load")
    
    pressure_injector = MemoryPressureInjector(0.80)
    
    try:
        await pressure_injector.inject_failure()
        await asyncio.sleep(10)
        
        # Execute burst under memory pressure
        burst_generator = LoadGenerator(
            target_rps=config.base_rps * 20,
            duration=60,
            pattern='constant'
        )
        
        burst_results = await burst_generator.run(orchestrator)
        burst_success_rate = ReliabilityTestUtils.calculate_success_rate(burst_results)
        
        pressure_results['burst_under_pressure'] = {
            'success_rate': burst_success_rate,
            'memory_level': 0.80,
            'burst_handling': burst_success_rate >= 0.70
        }
        
        assert burst_success_rate >= 0.70, \
            f"Burst under memory pressure: Success rate {burst_success_rate:.2f} too low"
        
    finally:
        await pressure_injector.restore_normal()
    
    logger.info("ST-003: Memory Pressure Endurance Test completed successfully")
    return pressure_results


async def monitor_memory_usage(duration: int) -> Dict[str, float]:
    """Monitor memory usage during testing."""
    peak_usage = 0
    samples = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        current_usage = psutil.virtual_memory().percent / 100.0
        peak_usage = max(peak_usage, current_usage)
        samples.append(current_usage)
        await asyncio.sleep(1)
    
    return {
        'peak_usage': peak_usage,
        'avg_usage': statistics.mean(samples) if samples else 0,
        'sample_count': len(samples)
    }


# ============================================================================
# ST-004: MAXIMUM CONCURRENT REQUEST HANDLING TEST
# ============================================================================

async def test_maximum_concurrent_requests(orchestrator, config: ReliabilityTestConfig):
    """
    ST-004: Determine and validate maximum concurrent request capacity.
    
    Validates that the system:
    - Handles high levels of concurrent requests
    - Gracefully rejects requests beyond capacity
    - Maintains system stability at capacity limits
    - Provides accurate capacity determination
    """
    logger.info("Starting ST-004: Maximum Concurrent Request Handling Test")
    
    concurrent_tester = ConcurrentRequestTester(orchestrator)
    concurrent_levels = [10, 25, 50, 100, 200, 500, 1000]
    capacity_results = {}
    max_viable_capacity = 0
    
    # Test increasing concurrent levels
    for level in concurrent_levels:
        logger.info(f"Testing {level} concurrent requests")
        
        # Execute concurrent request test
        results = await concurrent_tester.generate_concurrent_requests(
            target_concurrent=level,
            duration=300  # 5 minutes max wait
        )
        
        capacity_results[level] = results
        
        # Check if this level is viable
        if results['success_rate'] >= 0.85 and not results.get('timeout', False):
            max_viable_capacity = level
            logger.info(f"Level {level}: {results['success_rate']:.2f} success rate - VIABLE")
        else:
            logger.info(f"Level {level}: {results['success_rate']:.2f} success rate - NOT VIABLE")
            break
        
        # Allow recovery between tests
        await asyncio.sleep(10)
    
    # Validate determined capacity with sustained load
    if max_viable_capacity > 0:
        logger.info(f"Validating capacity {max_viable_capacity} with sustained load")
        
        validation_results = await validate_sustained_concurrent_capacity(
            orchestrator,
            concurrent_requests=max_viable_capacity,
            duration=300  # 5 minutes
        )
        
        capacity_results['sustained_validation'] = validation_results
        
        assert validation_results['success_rate'] >= 0.85, \
            f"Sustained capacity validation failed: {validation_results['success_rate']:.2f} success rate"
        
        assert validation_results['stability_score'] >= 0.90, \
            f"Sustained capacity stability too low: {validation_results['stability_score']:.2f}"
    
    # Test behavior beyond capacity
    if max_viable_capacity > 0:
        beyond_capacity_level = min(max_viable_capacity * 2, 2000)
        logger.info(f"Testing behavior beyond capacity: {beyond_capacity_level} concurrent requests")
        
        beyond_results = await concurrent_tester.generate_concurrent_requests(
            target_concurrent=beyond_capacity_level,
            duration=60  # Shorter duration for overload test
        )
        
        capacity_results['beyond_capacity'] = beyond_results
        
        # System should gracefully handle overload
        # Either by queueing or rejecting requests, but not crashing
        system_health = orchestrator.get_health_check() if hasattr(orchestrator, 'get_health_check') else {'status': 'healthy'}
        
        assert system_health.get('status', 'unhealthy') != 'crashed', \
            "System crashed under overload conditions"
    
    final_results = {
        'max_viable_capacity': max_viable_capacity,
        'capacity_test_results': capacity_results,
        'recommendation': f"System can handle up to {max_viable_capacity} concurrent requests reliably"
    }
    
    logger.info(f"ST-004: Maximum Concurrent Request Handling Test completed - Capacity: {max_viable_capacity}")
    return final_results


async def validate_sustained_concurrent_capacity(
    orchestrator, 
    concurrent_requests: int, 
    duration: int
) -> Dict[str, Any]:
    """Validate that determined capacity can be sustained over time."""
    
    # Generate sustained concurrent load
    stability_samples = []
    start_time = time.time()
    end_time = start_time + duration
    
    async def sustained_request_handler(request_id: int):
        """Handler for sustained concurrent requests."""
        processing_time = random.uniform(2.0, 5.0)  # Longer processing for sustained test
        await asyncio.sleep(processing_time)
        return f"Sustained request {request_id} processed"
    
    # Launch concurrent requests in batches to maintain target concurrency
    active_tasks = []
    total_requests = 0
    successful_requests = 0
    
    while time.time() < end_time:
        # Maintain target concurrency
        while len(active_tasks) < concurrent_requests and time.time() < end_time:
            task = orchestrator.submit_request(
                request_type='user_query',
                priority='medium',
                handler=lambda req_id=total_requests: sustained_request_handler(req_id),
                timeout=30.0
            )
            active_tasks.append(asyncio.create_task(task))
            total_requests += 1
        
        # Check for completed tasks
        completed_tasks = [t for t in active_tasks if t.done()]
        for task in completed_tasks:
            try:
                result = await task
                if result[0]:  # Success
                    successful_requests += 1
            except Exception:
                pass  # Failed request
        
        # Remove completed tasks
        active_tasks = [t for t in active_tasks if not t.done()]
        
        # Sample system stability
        if hasattr(orchestrator, 'get_health_check'):
            health = orchestrator.get_health_check()
            stability_samples.append({
                'timestamp': time.time(),
                'health_status': health.get('status', 'unknown'),
                'concurrent_requests': len(active_tasks)
            })
        
        await asyncio.sleep(1)
    
    # Wait for remaining requests to complete
    if active_tasks:
        try:
            remaining_results = await asyncio.wait_for(
                asyncio.gather(*active_tasks, return_exceptions=True),
                timeout=30
            )
            
            for result in remaining_results:
                if not isinstance(result, Exception) and result[0]:
                    successful_requests += 1
                    
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in active_tasks:
                task.cancel()
    
    # Calculate stability score
    healthy_samples = [s for s in stability_samples if s['health_status'] in ['healthy', 'operational']]
    stability_score = len(healthy_samples) / len(stability_samples) if stability_samples else 1.0
    
    return {
        'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
        'total_requests': total_requests,
        'successful_requests': successful_requests,
        'stability_score': stability_score,
        'avg_concurrent_achieved': statistics.mean([s['concurrent_requests'] for s in stability_samples]) if stability_samples else 0,
        'duration': duration
    }


# ============================================================================
# PYTEST TEST WRAPPER FUNCTIONS
# ============================================================================

@pytest.mark.asyncio
async def test_st_001_progressive_load_escalation():
    """Pytest wrapper for ST-001."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="ST-001-Progressive-Load-Escalation",
            test_func=test_progressive_load_escalation,
            category="stress_testing"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


@pytest.mark.asyncio
async def test_st_002_burst_load_handling():
    """Pytest wrapper for ST-002."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="ST-002-Burst-Load-Handling",
            test_func=test_burst_load_handling,
            category="stress_testing"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


@pytest.mark.asyncio
async def test_st_003_memory_pressure_endurance():
    """Pytest wrapper for ST-003."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="ST-003-Memory-Pressure-Endurance",
            test_func=test_memory_pressure_endurance,
            category="stress_testing"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


@pytest.mark.asyncio
async def test_st_004_maximum_concurrent_requests():
    """Pytest wrapper for ST-004."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="ST-004-Maximum-Concurrent-Requests",
            test_func=test_maximum_concurrent_requests,
            category="stress_testing"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_all_stress_tests():
    """Run all stress testing scenarios."""
    framework = ReliabilityValidationFramework()
    
    stress_tests = [
        ("ST-001", test_progressive_load_escalation),
        ("ST-002", test_burst_load_handling),
        ("ST-003", test_memory_pressure_endurance),
        ("ST-004", test_maximum_concurrent_requests)
    ]
    
    results = {}
    
    try:
        await framework.setup_test_environment()
        
        for test_name, test_func in stress_tests:
            logger.info(f"Executing {test_name}")
            
            result = await framework.execute_monitored_test(
                test_name=test_name,
                test_func=test_func,
                category="stress_testing"
            )
            
            results[test_name] = result
            logger.info(f"{test_name} completed: {result.status}")
            
            # Brief recovery between tests
            await asyncio.sleep(30)
            
    finally:
        await framework.cleanup_test_environment()
    
    # Report summary
    passed_tests = sum(1 for r in results.values() if r.status == 'passed')
    total_tests = len(results)
    
    print(f"\nStress Testing Summary: {passed_tests}/{total_tests} tests passed")
    
    for test_name, result in results.items():
        status_emoji = "✅" if result.status == 'passed' else "❌"
        print(f"{status_emoji} {test_name}: {result.status} ({result.duration:.1f}s)")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_stress_tests())