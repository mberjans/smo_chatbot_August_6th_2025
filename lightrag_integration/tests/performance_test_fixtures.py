#!/usr/bin/env python3
"""
Performance Test Fixtures for Clinical Metabolomics Oracle.

This module provides comprehensive performance testing fixtures for evaluating
scalability, throughput, latency, and resource utilization of the Clinical
Metabolomics Oracle LightRAG integration under various load conditions.

Components:
- LoadTestScenarioGenerator: Creates realistic load testing scenarios
- PerformanceBenchmarkSuite: Comprehensive performance benchmarking tools
- ScalabilityTestBuilder: Tests system behavior under increasing loads
- ResourceMonitoringFixtures: Monitors system resources during testing
- ConcurrencyTestManager: Tests concurrent operations and thread safety
- PerformanceRegressionDetector: Detects performance regressions

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import random
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import psutil
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    test_name: str
    start_time: float
    end_time: float
    duration: float
    operations_count: int
    success_count: int
    failure_count: int
    throughput_ops_per_sec: float
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate_percent: float
    concurrent_operations: int
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'test_name': self.test_name,
            'duration_seconds': self.duration,
            'operations_completed': self.operations_count,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'average_latency_ms': self.average_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'error_rate_percent': self.error_rate_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent
        }


@dataclass
class LoadTestScenario:
    """Defines a comprehensive load testing scenario."""
    scenario_name: str
    description: str
    target_operations_per_second: float
    duration_seconds: float
    concurrent_users: int
    ramp_up_duration: float
    operation_types: Dict[str, float]  # operation_type -> probability
    data_size_range: Tuple[int, int]  # min, max data size in bytes
    success_criteria: Dict[str, Any]
    resource_limits: Dict[str, Any]
    warmup_duration: float = 10.0
    cooldown_duration: float = 5.0
    
    @property
    def total_expected_operations(self) -> int:
        """Calculate total expected operations."""
        return int(self.target_operations_per_second * self.duration_seconds)
    
    @property
    def scenario_summary(self) -> str:
        """Generate scenario summary."""
        return f"{self.scenario_name}: {self.concurrent_users} users, {self.target_operations_per_second} ops/sec for {self.duration_seconds}s"


@dataclass
class ResourceUsageSnapshot:
    """Snapshot of system resource usage."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_received: float
    active_threads: int
    open_file_descriptors: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_received': self.network_bytes_received,
            'active_threads': self.active_threads,
            'open_file_descriptors': self.open_file_descriptors
        }


class ResourceMonitor:
    """
    Monitors system resources during performance testing.
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.snapshots: List[ResourceUsageSnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Baseline measurements
        self.baseline_cpu = 0.0
        self.baseline_memory = 0.0
        
        # Get process handle
        try:
            self.process = psutil.Process()
            self.system_available = True
        except:
            self.system_available = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.system_available:
            return
        
        self.monitoring = True
        self.snapshots.clear()
        
        # Take baseline measurements
        self.baseline_cpu = self.process.cpu_percent()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[ResourceUsageSnapshot]:
        """Stop monitoring and return snapshots."""
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        return self.snapshots.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        if not self.system_available:
            return
        
        initial_net_io = psutil.net_io_counters()
        initial_disk_io = psutil.disk_io_counters()
        
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()
                
                # Disk I/O
                try:
                    current_disk_io = psutil.disk_io_counters()
                    disk_read_mb = (current_disk_io.read_bytes - initial_disk_io.read_bytes) / 1024 / 1024
                    disk_write_mb = (current_disk_io.write_bytes - initial_disk_io.write_bytes) / 1024 / 1024
                except:
                    disk_read_mb = disk_write_mb = 0.0
                
                # Network I/O
                try:
                    current_net_io = psutil.net_io_counters()
                    net_sent = current_net_io.bytes_sent - initial_net_io.bytes_sent
                    net_received = current_net_io.bytes_recv - initial_net_io.bytes_recv
                except:
                    net_sent = net_received = 0.0
                
                # Thread count
                active_threads = threading.active_count()
                
                # File descriptors
                try:
                    open_fds = self.process.num_fds()
                except:
                    open_fds = 0
                
                snapshot = ResourceUsageSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_bytes_sent=net_sent,
                    network_bytes_received=net_received,
                    active_threads=active_threads,
                    open_file_descriptors=open_fds
                )
                
                self.snapshots.append(snapshot)
                
            except Exception as e:
                # Continue monitoring even if some metrics fail
                pass
            
            time.sleep(self.sampling_interval)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.snapshots:
            return {'error': 'No resource data available'}
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]
        
        return {
            'monitoring_duration': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'samples_collected': len(self.snapshots),
            'cpu_usage': {
                'average': np.mean(cpu_values),
                'maximum': np.max(cpu_values),
                'minimum': np.min(cpu_values),
                'baseline': self.baseline_cpu
            },
            'memory_usage': {
                'average_mb': np.mean(memory_values),
                'maximum_mb': np.max(memory_values),
                'minimum_mb': np.min(memory_values),
                'baseline_mb': self.baseline_memory,
                'peak_increase_mb': np.max(memory_values) - self.baseline_memory
            },
            'thread_count': {
                'average': np.mean([s.active_threads for s in self.snapshots]),
                'maximum': np.max([s.active_threads for s in self.snapshots])
            }
        }


class PerformanceTestExecutor:
    """
    Executes performance tests with comprehensive metrics collection.
    """
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.operation_latencies: List[float] = []
        self.operation_results: List[bool] = []
        self.test_start_time = 0.0
        self.test_end_time = 0.0
    
    async def execute_load_test(self, 
                              scenario: LoadTestScenario,
                              operation_func: Callable,
                              operation_data_generator: Callable) -> PerformanceMetrics:
        """Execute a comprehensive load test scenario."""
        
        print(f"Starting load test: {scenario.scenario_summary}")
        
        # Initialize tracking
        self.operation_latencies.clear()
        self.operation_results.clear()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Warmup phase
        if scenario.warmup_duration > 0:
            print(f"Warmup phase: {scenario.warmup_duration}s")
            await self._execute_warmup(scenario.warmup_duration, operation_func, operation_data_generator)
        
        # Main test phase
        self.test_start_time = time.time()
        print(f"Main test phase: {scenario.duration_seconds}s with {scenario.concurrent_users} concurrent users")
        
        await self._execute_main_test(scenario, operation_func, operation_data_generator)
        
        self.test_end_time = time.time()
        
        # Cooldown phase
        if scenario.cooldown_duration > 0:
            print(f"Cooldown phase: {scenario.cooldown_duration}s")
            await asyncio.sleep(scenario.cooldown_duration)
        
        # Stop monitoring and collect results
        resource_snapshots = self.resource_monitor.stop_monitoring()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(scenario, resource_snapshots)
        
        print(f"Load test completed: {metrics.summary}")
        
        return metrics
    
    async def _execute_warmup(self, 
                            duration: float, 
                            operation_func: Callable,
                            data_generator: Callable):
        """Execute warmup phase."""
        warmup_end_time = time.time() + duration
        warmup_operations = 0
        
        # Light load during warmup
        concurrent_ops = min(5, max(1, int(duration / 2)))
        
        while time.time() < warmup_end_time:
            tasks = []
            for _ in range(concurrent_ops):
                data = data_generator()
                task = asyncio.create_task(self._execute_single_operation(operation_func, data, warmup=True))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            warmup_operations += len(tasks)
            
            # Brief pause
            await asyncio.sleep(0.1)
        
        print(f"Warmup completed: {warmup_operations} operations")
    
    async def _execute_main_test(self, 
                               scenario: LoadTestScenario,
                               operation_func: Callable,
                               data_generator: Callable):
        """Execute main test phase with proper load distribution."""
        
        # Calculate operations per batch
        batch_interval = 1.0  # 1 second batches
        ops_per_batch = max(1, int(scenario.target_operations_per_second * batch_interval))
        
        test_end_time = time.time() + scenario.duration_seconds
        batch_count = 0
        
        while time.time() < test_end_time:
            batch_start = time.time()
            
            # Create batch of operations
            tasks = []
            for _ in range(min(ops_per_batch, scenario.concurrent_users)):
                # Select operation type based on probabilities
                operation_type = self._select_operation_type(scenario.operation_types)
                data = data_generator(operation_type)
                
                task = asyncio.create_task(self._execute_single_operation(operation_func, data))
                tasks.append(task)
            
            # Execute batch
            await asyncio.gather(*tasks, return_exceptions=True)
            batch_count += 1
            
            # Control rate
            elapsed = time.time() - batch_start
            sleep_time = max(0, batch_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        print(f"Main test completed: {batch_count} batches executed")
    
    async def _execute_single_operation(self, 
                                      operation_func: Callable,
                                      data: Any,
                                      warmup: bool = False) -> bool:
        """Execute single operation with timing."""
        start_time = time.time()
        success = False
        
        try:
            # Execute the operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(data)
            else:
                result = operation_func(data)
            
            success = True
            
        except Exception as e:
            success = False
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Record metrics (only for main test, not warmup)
        if not warmup:
            self.operation_latencies.append(latency_ms)
            self.operation_results.append(success)
        
        return success
    
    def _select_operation_type(self, operation_types: Dict[str, float]) -> str:
        """Select operation type based on probabilities."""
        random_value = random.random()
        cumulative = 0.0
        
        for op_type, probability in operation_types.items():
            cumulative += probability
            if random_value <= cumulative:
                return op_type
        
        # Fallback to first operation type
        return list(operation_types.keys())[0]
    
    def _calculate_performance_metrics(self, 
                                     scenario: LoadTestScenario,
                                     resource_snapshots: List[ResourceUsageSnapshot]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        test_duration = self.test_end_time - self.test_start_time
        total_operations = len(self.operation_results)
        successful_operations = sum(self.operation_results)
        failed_operations = total_operations - successful_operations
        
        # Calculate latency statistics
        if self.operation_latencies:
            latencies = np.array(self.operation_latencies)
            avg_latency = np.mean(latencies)
            median_latency = np.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
        else:
            avg_latency = median_latency = p95_latency = p99_latency = min_latency = max_latency = 0.0
        
        # Calculate throughput
        throughput = successful_operations / test_duration if test_duration > 0 else 0.0
        
        # Calculate error rate
        error_rate = (failed_operations / total_operations * 100) if total_operations > 0 else 0.0
        
        # Get resource usage
        if resource_snapshots:
            avg_memory = np.mean([s.memory_mb for s in resource_snapshots])
            avg_cpu = np.mean([s.cpu_percent for s in resource_snapshots])
        else:
            avg_memory = avg_cpu = 0.0
        
        return PerformanceMetrics(
            test_name=scenario.scenario_name,
            start_time=self.test_start_time,
            end_time=self.test_end_time,
            duration=test_duration,
            operations_count=total_operations,
            success_count=successful_operations,
            failure_count=failed_operations,
            throughput_ops_per_sec=throughput,
            average_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            error_rate_percent=error_rate,
            concurrent_operations=scenario.concurrent_users,
            additional_metrics={
                'target_ops_per_sec': scenario.target_operations_per_second,
                'achieved_ops_per_sec': throughput,
                'throughput_ratio': throughput / scenario.target_operations_per_second if scenario.target_operations_per_second > 0 else 0,
                'resource_efficiency': successful_operations / avg_memory if avg_memory > 0 else 0
            }
        )


class LoadTestScenarioGenerator:
    """
    Generates various load testing scenarios for different performance testing needs.
    """
    
    @staticmethod
    def create_baseline_scenario() -> LoadTestScenario:
        """Create baseline performance scenario."""
        return LoadTestScenario(
            scenario_name="baseline_performance",
            description="Baseline performance test with single user",
            target_operations_per_second=1.0,
            duration_seconds=30.0,
            concurrent_users=1,
            ramp_up_duration=0.0,
            operation_types={
                'simple_query': 0.6,
                'medium_query': 0.3,
                'complex_query': 0.1
            },
            data_size_range=(100, 1000),
            success_criteria={
                'min_throughput_ops_per_sec': 0.8,
                'max_average_latency_ms': 2000,
                'max_error_rate_percent': 5.0,
                'max_memory_mb': 500
            },
            resource_limits={
                'max_memory_mb': 1000,
                'max_cpu_percent': 80
            }
        )
    
    @staticmethod
    def create_light_load_scenario() -> LoadTestScenario:
        """Create light load scenario."""
        return LoadTestScenario(
            scenario_name="light_load_test",
            description="Light load test with moderate concurrent users",
            target_operations_per_second=5.0,
            duration_seconds=60.0,
            concurrent_users=3,
            ramp_up_duration=10.0,
            operation_types={
                'simple_query': 0.5,
                'medium_query': 0.4,
                'complex_query': 0.1
            },
            data_size_range=(200, 2000),
            success_criteria={
                'min_throughput_ops_per_sec': 4.0,
                'max_average_latency_ms': 3000,
                'max_error_rate_percent': 5.0,
                'max_memory_mb': 800
            },
            resource_limits={
                'max_memory_mb': 1500,
                'max_cpu_percent': 85
            }
        )
    
    @staticmethod
    def create_moderate_load_scenario() -> LoadTestScenario:
        """Create moderate load scenario."""
        return LoadTestScenario(
            scenario_name="moderate_load_test",
            description="Moderate load test simulating typical usage",
            target_operations_per_second=10.0,
            duration_seconds=120.0,
            concurrent_users=8,
            ramp_up_duration=20.0,
            operation_types={
                'simple_query': 0.4,
                'medium_query': 0.4,
                'complex_query': 0.2
            },
            data_size_range=(500, 5000),
            success_criteria={
                'min_throughput_ops_per_sec': 8.0,
                'max_average_latency_ms': 5000,
                'max_error_rate_percent': 8.0,
                'max_memory_mb': 1200
            },
            resource_limits={
                'max_memory_mb': 2000,
                'max_cpu_percent': 90
            },
            warmup_duration=15.0
        )
    
    @staticmethod
    def create_heavy_load_scenario() -> LoadTestScenario:
        """Create heavy load scenario for stress testing."""
        return LoadTestScenario(
            scenario_name="heavy_load_test",
            description="Heavy load stress test",
            target_operations_per_second=20.0,
            duration_seconds=180.0,
            concurrent_users=15,
            ramp_up_duration=30.0,
            operation_types={
                'simple_query': 0.3,
                'medium_query': 0.4,
                'complex_query': 0.3
            },
            data_size_range=(1000, 10000),
            success_criteria={
                'min_throughput_ops_per_sec': 15.0,
                'max_average_latency_ms': 8000,
                'max_error_rate_percent': 15.0,
                'max_memory_mb': 2000
            },
            resource_limits={
                'max_memory_mb': 4000,
                'max_cpu_percent': 95
            },
            warmup_duration=20.0,
            cooldown_duration=10.0
        )
    
    @staticmethod
    def create_spike_test_scenario() -> LoadTestScenario:
        """Create spike test scenario."""
        return LoadTestScenario(
            scenario_name="spike_test",
            description="Spike test with sudden load increase",
            target_operations_per_second=50.0,
            duration_seconds=60.0,
            concurrent_users=25,
            ramp_up_duration=5.0,  # Rapid ramp-up
            operation_types={
                'simple_query': 0.6,
                'medium_query': 0.3,
                'complex_query': 0.1
            },
            data_size_range=(100, 2000),
            success_criteria={
                'min_throughput_ops_per_sec': 30.0,
                'max_average_latency_ms': 10000,
                'max_error_rate_percent': 25.0,
                'max_memory_mb': 3000
            },
            resource_limits={
                'max_memory_mb': 5000,
                'max_cpu_percent': 98
            },
            warmup_duration=10.0,
            cooldown_duration=15.0
        )
    
    @staticmethod
    def create_endurance_test_scenario() -> LoadTestScenario:
        """Create endurance test scenario for long-duration testing."""
        return LoadTestScenario(
            scenario_name="endurance_test",
            description="Long-duration endurance test",
            target_operations_per_second=5.0,
            duration_seconds=600.0,  # 10 minutes
            concurrent_users=5,
            ramp_up_duration=30.0,
            operation_types={
                'simple_query': 0.5,
                'medium_query': 0.3,
                'complex_query': 0.2
            },
            data_size_range=(300, 3000),
            success_criteria={
                'min_throughput_ops_per_sec': 4.0,
                'max_average_latency_ms': 4000,
                'max_error_rate_percent': 5.0,
                'max_memory_mb': 1000,
                'memory_stability': True  # Memory should not continuously grow
            },
            resource_limits={
                'max_memory_mb': 2000,
                'max_cpu_percent': 85
            },
            warmup_duration=30.0,
            cooldown_duration=30.0
        )
    
    @staticmethod
    def create_custom_scenario(
        name: str,
        ops_per_sec: float,
        duration: float,
        concurrent_users: int,
        operation_mix: Dict[str, float]
    ) -> LoadTestScenario:
        """Create custom load test scenario."""
        return LoadTestScenario(
            scenario_name=name,
            description=f"Custom scenario: {ops_per_sec} ops/sec, {concurrent_users} users",
            target_operations_per_second=ops_per_sec,
            duration_seconds=duration,
            concurrent_users=concurrent_users,
            ramp_up_duration=duration * 0.1,  # 10% ramp-up
            operation_types=operation_mix,
            data_size_range=(500, 5000),
            success_criteria={
                'min_throughput_ops_per_sec': ops_per_sec * 0.8,
                'max_average_latency_ms': 5000,
                'max_error_rate_percent': 10.0,
                'max_memory_mb': 1500
            },
            resource_limits={
                'max_memory_mb': 3000,
                'max_cpu_percent': 90
            },
            warmup_duration=max(10.0, duration * 0.05),
            cooldown_duration=5.0
        )


class MockOperationGenerator:
    """
    Generates mock operations for performance testing.
    """
    
    def __init__(self):
        self.query_templates = {
            'simple_query': [
                "What is {}?",
                "Find information about {}",
                "Tell me about {}",
                "Describe {}"
            ],
            'medium_query': [
                "Compare {} and {} in terms of clinical significance",
                "Analyze the pathway involving {} and {}",
                "What are the biomarkers for {} disease?",
                "How is {} metabolized in {}?"
            ],
            'complex_query': [
                "Provide comprehensive analysis of {} pathway including key enzymes, regulation, and clinical relevance",
                "Integrate metabolomics and proteomics data for {} biomarker discovery in {} disease",
                "Develop diagnostic panel for {} using {} analytical methods with statistical validation",
                "Design clinical trial for {} therapeutic intervention monitoring using metabolomics"
            ]
        }
        
        self.biomedical_terms = [
            'glucose', 'lactate', 'cholesterol', 'creatinine', 'insulin',
            'diabetes', 'cardiovascular disease', 'cancer', 'liver disease',
            'glycolysis', 'TCA cycle', 'fatty acid metabolism', 'amino acid metabolism'
        ]
    
    def generate_query_data(self, operation_type: str = 'medium_query') -> Dict[str, Any]:
        """Generate query data for testing."""
        
        templates = self.query_templates.get(operation_type, self.query_templates['medium_query'])
        template = random.choice(templates)
        
        # Fill template with biomedical terms
        if '{}' in template:
            if template.count('{}') == 1:
                query_text = template.format(random.choice(self.biomedical_terms))
            elif template.count('{}') == 2:
                terms = random.sample(self.biomedical_terms, 2)
                query_text = template.format(terms[0], terms[1])
            else:
                # Multiple placeholders - fill with random terms
                terms = random.choices(self.biomedical_terms, k=template.count('{}'))
                query_text = template.format(*terms)
        else:
            query_text = template
        
        return {
            'query_text': query_text,
            'operation_type': operation_type,
            'expected_response_length': self._estimate_response_length(operation_type),
            'complexity_score': self._calculate_complexity_score(operation_type),
            'processing_priority': random.choice(['normal', 'high']),
            'client_id': f"test_client_{random.randint(1, 100)}",
            'session_id': f"session_{random.randint(1000, 9999)}"
        }
    
    def _estimate_response_length(self, operation_type: str) -> int:
        """Estimate expected response length."""
        length_ranges = {
            'simple_query': (100, 300),
            'medium_query': (300, 800),
            'complex_query': (800, 2000)
        }
        
        range_tuple = length_ranges.get(operation_type, length_ranges['medium_query'])
        return random.randint(*range_tuple)
    
    def _calculate_complexity_score(self, operation_type: str) -> float:
        """Calculate complexity score for operation."""
        complexity_scores = {
            'simple_query': random.uniform(0.1, 0.3),
            'medium_query': random.uniform(0.3, 0.7),
            'complex_query': random.uniform(0.7, 1.0)
        }
        
        return complexity_scores.get(operation_type, 0.5)


# Mock operation function for testing
async def mock_clinical_query_operation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock clinical query operation for performance testing."""
    
    # Simulate processing time based on complexity
    complexity = data.get('complexity_score', 0.5)
    base_delay = 0.1
    complexity_delay = complexity * 0.5
    random_variation = random.uniform(0.8, 1.2)
    
    delay = (base_delay + complexity_delay) * random_variation
    await asyncio.sleep(delay)
    
    # Simulate occasional failures
    if random.random() < 0.02:  # 2% failure rate
        raise Exception("Simulated operation failure")
    
    # Generate mock response
    response_length = data.get('expected_response_length', 500)
    response = "Mock clinical response. " * (response_length // 25)
    
    return {
        'query': data['query_text'],
        'response': response,
        'processing_time': delay,
        'tokens_used': response_length // 4,
        'cost': delay * 0.01,
        'confidence': random.uniform(0.7, 0.95)
    }


# Pytest fixtures for performance testing
@pytest.fixture
def performance_test_executor():
    """Provide performance test executor."""
    return PerformanceTestExecutor()

@pytest.fixture
def resource_monitor():
    """Provide resource monitor."""
    return ResourceMonitor(sampling_interval=0.5)

@pytest.fixture
def load_test_scenarios():
    """Provide collection of load test scenarios."""
    generator = LoadTestScenarioGenerator()
    
    return {
        'baseline': generator.create_baseline_scenario(),
        'light_load': generator.create_light_load_scenario(),
        'moderate_load': generator.create_moderate_load_scenario(),
        'heavy_load': generator.create_heavy_load_scenario(),
        'spike_test': generator.create_spike_test_scenario(),
        'endurance_test': generator.create_endurance_test_scenario()
    }

@pytest.fixture
def mock_operation_generator():
    """Provide mock operation generator."""
    return MockOperationGenerator()

@pytest.fixture
def performance_benchmarks():
    """Provide performance benchmarks for comparison."""
    return {
        'acceptable_latency_ms': {
            'simple_query': 1000,
            'medium_query': 3000,
            'complex_query': 8000
        },
        'minimum_throughput_ops_per_sec': {
            'light_load': 4.0,
            'moderate_load': 8.0,
            'heavy_load': 15.0
        },
        'resource_limits': {
            'max_memory_mb': 2000,
            'max_cpu_percent': 85,
            'max_error_rate_percent': 10.0
        },
        'scalability_targets': {
            'concurrent_users_supported': 20,
            'max_operations_per_hour': 36000,
            'memory_per_user_mb': 100
        }
    }

@pytest.fixture
async def sample_performance_test_results(performance_test_executor, mock_operation_generator):
    """Provide sample performance test results."""
    
    # Run a quick baseline test
    baseline_scenario = LoadTestScenarioGenerator.create_baseline_scenario()
    baseline_scenario.duration_seconds = 10.0  # Shortened for fixture
    
    metrics = await performance_test_executor.execute_load_test(
        scenario=baseline_scenario,
        operation_func=mock_clinical_query_operation,
        operation_data_generator=mock_operation_generator.generate_query_data
    )
    
    return metrics

@pytest.fixture
def performance_regression_detector():
    """Provide performance regression detection."""
    
    class PerformanceRegressionDetector:
        def __init__(self):
            self.baseline_metrics = {}
            self.regression_thresholds = {
                'throughput_degradation_percent': 10.0,
                'latency_increase_percent': 20.0,
                'error_rate_increase_percent': 5.0,
                'memory_increase_percent': 15.0
            }
        
        def set_baseline(self, metrics: PerformanceMetrics):
            """Set baseline performance metrics."""
            self.baseline_metrics = metrics.summary
        
        def detect_regressions(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
            """Detect performance regressions."""
            if not self.baseline_metrics:
                return {'status': 'no_baseline'}
            
            current = current_metrics.summary
            regressions = []
            
            # Check throughput regression
            baseline_throughput = self.baseline_metrics['throughput_ops_per_sec']
            current_throughput = current['throughput_ops_per_sec']
            
            if baseline_throughput > 0:
                throughput_change = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
                if throughput_change > self.regression_thresholds['throughput_degradation_percent']:
                    regressions.append({
                        'type': 'throughput_degradation',
                        'baseline': baseline_throughput,
                        'current': current_throughput,
                        'change_percent': throughput_change
                    })
            
            # Check latency regression
            baseline_latency = self.baseline_metrics['average_latency_ms']
            current_latency = current['average_latency_ms']
            
            if baseline_latency > 0:
                latency_increase = ((current_latency - baseline_latency) / baseline_latency) * 100
                if latency_increase > self.regression_thresholds['latency_increase_percent']:
                    regressions.append({
                        'type': 'latency_increase',
                        'baseline': baseline_latency,
                        'current': current_latency,
                        'change_percent': latency_increase
                    })
            
            # Check error rate regression
            baseline_error_rate = self.baseline_metrics['error_rate_percent']
            current_error_rate = current['error_rate_percent']
            
            error_rate_increase = current_error_rate - baseline_error_rate
            if error_rate_increase > self.regression_thresholds['error_rate_increase_percent']:
                regressions.append({
                    'type': 'error_rate_increase',
                    'baseline': baseline_error_rate,
                    'current': current_error_rate,
                    'change_percent': error_rate_increase
                })
            
            return {
                'status': 'regression_detected' if regressions else 'no_regression',
                'regressions': regressions,
                'summary': {
                    'total_regressions': len(regressions),
                    'severity': 'critical' if any(r['change_percent'] > 50 for r in regressions) else 'moderate'
                }
            }
    
    return PerformanceRegressionDetector()

@pytest.fixture
def concurrent_operation_test_suite():
    """Provide concurrent operation test suite."""
    
    async def test_concurrent_queries(num_concurrent: int = 10):
        """Test concurrent query execution."""
        generator = MockOperationGenerator()
        
        tasks = []
        for _ in range(num_concurrent):
            data = generator.generate_query_data('medium_query')
            task = asyncio.create_task(mock_clinical_query_operation(data))
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        return {
            'total_operations': num_concurrent,
            'successful_operations': len(successful_results),
            'failed_operations': len(failed_results),
            'total_time': end_time - start_time,
            'operations_per_second': num_concurrent / (end_time - start_time),
            'success_rate': len(successful_results) / num_concurrent * 100
        }
    
    return {
        'test_concurrent_queries': test_concurrent_queries,
        'concurrency_levels': [1, 5, 10, 20, 50],
        'expected_linear_scaling_threshold': 0.8  # 80% efficiency expected
    }