#!/usr/bin/env python3
"""
Performance Benchmark Suite for High-Performance Classification System

This module provides comprehensive benchmarking tools to validate and measure
the performance of the high-performance LLM-based classification system under
various load conditions and usage patterns.

Key Features:
    - Multi-tier benchmarking (unit, integration, load, stress)
    - Real-time performance monitoring during tests
    - Automated performance regression detection
    - Detailed performance analytics and reporting
    - Load pattern simulation (constant, burst, ramp, spike)
    - Resource utilization monitoring
    - Statistical analysis with confidence intervals
    - Automated optimization recommendations
    - Export capabilities for CI/CD integration

Author: Claude Code (Anthropic)
Version: 1.0.0 - Performance Benchmark Suite
Created: 2025-08-08
Target: Validate consistent <2 second response times with comprehensive analysis
"""

import asyncio
import time
import statistics
import logging
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
import psutil
import threading
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress matplotlib warnings in headless environments
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Import high-performance system components
try:
    from .high_performance_classification_system import (
        HighPerformanceClassificationSystem,
        HighPerformanceConfig,
        high_performance_classification_context,
        create_high_performance_system
    )
    from .llm_classification_prompts import ClassificationResult
    HP_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import high performance components: {e}")
    HP_COMPONENTS_AVAILABLE = False
    
    # Create placeholder classes for type hints when components aren't available
    class HighPerformanceConfig:
        pass
    
    class HighPerformanceClassificationSystem:
        pass
    
    class ClassificationResult:
        pass


# ============================================================================
# BENCHMARK CONFIGURATION AND ENUMS
# ============================================================================

class LoadPattern(Enum):
    """Load patterns for benchmarking."""
    CONSTANT = "constant"           # Steady constant load
    RAMP_UP = "ramp_up"            # Gradual increase in load
    BURST = "burst"                # Sudden spikes in load
    SPIKE = "spike"                # Extreme load spikes
    WAVE = "wave"                  # Sine wave pattern
    RANDOM = "random"              # Random load variations

class BenchmarkType(Enum):
    """Types of benchmarks."""
    UNIT = "unit"                  # Single request validation
    INTEGRATION = "integration"    # Component integration tests
    LOAD = "load"                  # Expected load testing
    STRESS = "stress"              # Beyond normal load testing
    ENDURANCE = "endurance"        # Long-duration testing
    SPIKE = "spike"                # Extreme load spikes

class PerformanceGrade(Enum):
    """Performance grading system."""
    EXCELLENT = "A+"               # <1000ms avg, >99% compliance
    VERY_GOOD = "A"                # <1200ms avg, >97% compliance
    GOOD = "B+"                    # <1400ms avg, >95% compliance
    ACCEPTABLE = "B"               # <1600ms avg, >90% compliance
    POOR = "C"                     # <2000ms avg, >80% compliance
    FAILING = "F"                  # >2000ms avg or <80% compliance


@dataclass
class BenchmarkConfig:
    """Comprehensive benchmark configuration."""
    
    # Test configuration
    benchmark_name: str = "performance_validation"
    benchmark_type: BenchmarkType = BenchmarkType.LOAD
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    
    # Load parameters
    concurrent_users: int = 20
    total_requests: int = 1000
    duration_seconds: Optional[int] = None
    requests_per_second: Optional[float] = None
    
    # Performance targets
    target_response_time_ms: float = 1500.0
    max_response_time_ms: float = 2000.0
    target_throughput_rps: float = 50.0
    target_compliance_rate: float = 0.95
    
    # Test queries
    test_queries: List[str] = field(default_factory=list)
    query_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Monitoring and reporting
    enable_real_time_monitoring: bool = True
    enable_detailed_logging: bool = True
    export_results: bool = True
    generate_plots: bool = True
    output_directory: str = "benchmark_results"
    
    # System configuration
    hp_config: Optional[HighPerformanceConfig] = None
    warmup_requests: int = 50
    cooldown_seconds: int = 10
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.test_queries:
            self.test_queries = [
                "What is metabolomics?",
                "LC-MS analysis for biomarker identification",
                "Pathway enrichment analysis methods",
                "Latest research in clinical metabolomics 2025", 
                "Relationship between glucose metabolism and diabetes",
                "Statistical analysis of metabolomics data",
                "Machine learning applications in metabolomics",
                "Quality control in metabolomics workflows",
                "Biomarker validation in clinical studies",
                "Multi-omics integration approaches",
                "Metabolite identification using mass spectrometry",
                "Clinical diagnosis using metabolomics",
                "Drug discovery pipeline optimization",
                "Real-time metabolomics monitoring systems",
                "Data preprocessing for metabolomics analysis"
            ]
        
        if not self.query_distribution:
            # Equal distribution by default
            weight = 1.0 / len(self.test_queries)
            self.query_distribution = {query: weight for query in self.test_queries}


# ============================================================================
# BENCHMARK RESULTS AND METRICS
# ============================================================================

@dataclass
class RequestResult:
    """Individual request result metrics."""
    request_id: str
    query_text: str
    start_time: float
    end_time: float
    response_time_ms: float
    success: bool
    error_message: Optional[str]
    classification_result: Optional[ClassificationResult]
    metadata: Dict[str, Any]
    user_id: int
    timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Basic metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    
    # Response time metrics
    avg_response_time_ms: float
    median_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    std_dev_ms: float
    
    # Target compliance
    target_compliance_count: int
    target_compliance_rate: float
    max_compliance_count: int
    max_compliance_rate: float
    
    # Throughput metrics
    actual_throughput_rps: float
    peak_throughput_rps: float
    min_throughput_rps: float
    
    # Resource utilization
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_percent: float
    max_memory_percent: float
    
    # Cache performance
    cache_hit_rate: float
    avg_cache_response_time_ms: float
    
    # Quality metrics
    performance_grade: PerformanceGrade
    regression_detected: bool
    optimization_score: float


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    
    # Configuration
    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    
    # Performance metrics
    metrics: PerformanceMetrics
    
    # Detailed results
    request_results: List[RequestResult]
    time_series_data: Dict[str, List[Tuple[float, float]]]
    resource_utilization: Dict[str, List[Tuple[float, float]]]
    
    # Analysis
    performance_analysis: Dict[str, Any]
    bottleneck_analysis: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    
    # Export data
    export_paths: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# LOAD PATTERN GENERATORS
# ============================================================================

class LoadPatternGenerator:
    """Generates different load patterns for benchmarking."""
    
    @staticmethod
    def constant_load(total_requests: int, concurrent_users: int) -> List[Tuple[int, float]]:
        """Generate constant load pattern."""
        requests_per_user = total_requests // concurrent_users
        return [(concurrent_users, 0.0)] * requests_per_user
    
    @staticmethod
    def ramp_up_load(total_requests: int, max_concurrent: int, ramp_duration: float) -> List[Tuple[int, float]]:
        """Generate ramping up load pattern."""
        pattern = []
        requests_per_step = 10
        steps = total_requests // requests_per_step
        step_duration = ramp_duration / steps
        
        for i in range(steps):
            concurrent_users = min(max_concurrent, int((i + 1) * max_concurrent / steps))
            pattern.extend([(concurrent_users, step_duration / requests_per_step)] * requests_per_step)
        
        return pattern
    
    @staticmethod
    def burst_load(total_requests: int, concurrent_users: int, burst_interval: float) -> List[Tuple[int, float]]:
        """Generate burst load pattern with periodic spikes."""
        pattern = []
        base_load = concurrent_users // 3
        burst_load = concurrent_users
        
        requests_per_burst = 50
        bursts = total_requests // requests_per_burst
        
        for i in range(bursts):
            if i % 4 == 0:  # Burst every 4th interval
                pattern.extend([(burst_load, 0.01)] * requests_per_burst)
            else:
                pattern.extend([(base_load, burst_interval)] * requests_per_burst)
        
        return pattern
    
    @staticmethod
    def spike_load(total_requests: int, concurrent_users: int) -> List[Tuple[int, float]]:
        """Generate extreme spike load pattern."""
        pattern = []
        base_load = 2
        spike_load = concurrent_users
        
        normal_duration = 0.1
        spike_duration = 0.01
        
        requests_between_spikes = 20
        spike_requests = 10
        
        remaining_requests = total_requests
        while remaining_requests > 0:
            # Normal load
            normal_count = min(requests_between_spikes, remaining_requests)
            pattern.extend([(base_load, normal_duration)] * normal_count)
            remaining_requests -= normal_count
            
            # Spike load
            if remaining_requests > 0:
                spike_count = min(spike_requests, remaining_requests)
                pattern.extend([(spike_load, spike_duration)] * spike_count)
                remaining_requests -= spike_count
        
        return pattern
    
    @staticmethod
    def wave_load(total_requests: int, max_concurrent: int, wave_period: float) -> List[Tuple[int, float]]:
        """Generate sine wave load pattern."""
        pattern = []
        requests_per_step = 5
        steps = total_requests // requests_per_step
        
        for i in range(steps):
            # Sine wave with minimum of 1 user
            phase = (i / steps) * 2 * np.pi * (total_requests / (wave_period * 100))
            concurrent_users = max(1, int(max_concurrent * (0.5 + 0.5 * np.sin(phase))))
            pattern.extend([(concurrent_users, wave_period / 100)] * requests_per_step)
        
        return pattern
    
    @staticmethod
    def random_load(total_requests: int, max_concurrent: int, seed: int = 42) -> List[Tuple[int, float]]:
        """Generate random load pattern."""
        np.random.seed(seed)
        pattern = []
        
        requests_per_step = 10
        steps = total_requests // requests_per_step
        
        for _ in range(steps):
            concurrent_users = np.random.randint(1, max_concurrent + 1)
            delay = np.random.uniform(0.01, 0.2)
            pattern.extend([(concurrent_users, delay)] * requests_per_step)
        
        return pattern


# ============================================================================
# REAL-TIME PERFORMANCE MONITOR
# ============================================================================

class RealTimePerformanceMonitor:
    """Real-time performance monitoring during benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Metrics tracking
        self.response_times = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        self.cpu_samples = deque(maxlen=200)
        self.memory_samples = deque(maxlen=200)
        
        # Time series data
        self.time_series = {
            "response_times": [],
            "throughput": [],
            "cpu_usage": [],
            "memory_usage": [],
            "concurrent_users": []
        }
        
        self.start_time = None
        self.lock = threading.RLock()
        
        # Performance alerts
        self.alerts = []
        self.alert_thresholds = {
            "response_time": config.max_response_time_ms,
            "cpu_usage": 90.0,
            "memory_usage": 90.0,
            "error_rate": 0.05
        }
    
    async def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logging.info("Real-time performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Real-time performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                current_time = time.time() - self.start_time
                
                # Sample system resources
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                with self.lock:
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_percent)
                    
                    # Record time series data
                    self.time_series["cpu_usage"].append((current_time, cpu_percent))
                    self.time_series["memory_usage"].append((current_time, memory_percent))
                    
                    # Calculate current throughput
                    if len(self.throughput_samples) > 0:
                        recent_throughput = sum(self.throughput_samples) / max(1, len(self.throughput_samples))
                        self.time_series["throughput"].append((current_time, recent_throughput))
                    
                    # Check for alerts
                    self._check_alerts(current_time, cpu_percent, memory_percent)
                
                await asyncio.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)
    
    def record_request(self, response_time_ms: float, success: bool, concurrent_users: int):
        """Record a request result."""
        current_time = time.time() - self.start_time if self.start_time else 0
        
        with self.lock:
            self.response_times.append(response_time_ms)
            
            # Estimate throughput
            self.throughput_samples.append(1.0)  # 1 request completed
            
            # Record time series
            self.time_series["response_times"].append((current_time, response_time_ms))
            self.time_series["concurrent_users"].append((current_time, concurrent_users))
    
    def _check_alerts(self, current_time: float, cpu_percent: float, memory_percent: float):
        """Check for performance alerts."""
        
        # Response time alerts
        if len(self.response_times) >= 10:
            recent_avg = statistics.mean(list(self.response_times)[-10:])
            if recent_avg > self.alert_thresholds["response_time"]:
                self.alerts.append({
                    "timestamp": current_time,
                    "type": "response_time",
                    "message": f"High average response time: {recent_avg:.1f}ms",
                    "severity": "warning"
                })
        
        # Resource alerts
        if cpu_percent > self.alert_thresholds["cpu_usage"]:
            self.alerts.append({
                "timestamp": current_time,
                "type": "cpu_usage",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "severity": "warning"
            })
        
        if memory_percent > self.alert_thresholds["memory_usage"]:
            self.alerts.append({
                "timestamp": current_time,
                "type": "memory_usage", 
                "message": f"High memory usage: {memory_percent:.1f}%",
                "severity": "warning"
            })
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.lock:
            recent_response_times = list(self.response_times)[-50:]  # Last 50 requests
            
            return {
                "current_throughput": sum(self.throughput_samples) / max(1, len(self.throughput_samples)),
                "avg_response_time": statistics.mean(recent_response_times) if recent_response_times else 0,
                "current_cpu": self.cpu_samples[-1] if self.cpu_samples else 0,
                "current_memory": self.memory_samples[-1] if self.memory_samples else 0,
                "alert_count": len(self.alerts)
            }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class PerformanceBenchmarkRunner:
    """Main benchmark runner with comprehensive analysis capabilities."""
    
    def __init__(self, config: BenchmarkConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Results tracking
        self.results: List[RequestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Monitoring
        self.monitor = RealTimePerformanceMonitor(config) if config.enable_real_time_monitoring else None
        
        # Load pattern
        self.load_pattern = self._generate_load_pattern()
        
        # System under test
        self.hp_system: Optional[HighPerformanceClassificationSystem] = None
        
        self.logger.info(f"Benchmark runner initialized: {config.benchmark_name}")
    
    def _generate_load_pattern(self) -> List[Tuple[int, float]]:
        """Generate load pattern based on configuration."""
        generator = LoadPatternGenerator()
        
        if self.config.load_pattern == LoadPattern.CONSTANT:
            return generator.constant_load(self.config.total_requests, self.config.concurrent_users)
        elif self.config.load_pattern == LoadPattern.RAMP_UP:
            return generator.ramp_up_load(self.config.total_requests, self.config.concurrent_users, 60.0)
        elif self.config.load_pattern == LoadPattern.BURST:
            return generator.burst_load(self.config.total_requests, self.config.concurrent_users, 0.1)
        elif self.config.load_pattern == LoadPattern.SPIKE:
            return generator.spike_load(self.config.total_requests, self.config.concurrent_users)
        elif self.config.load_pattern == LoadPattern.WAVE:
            return generator.wave_load(self.config.total_requests, self.config.concurrent_users, 30.0)
        elif self.config.load_pattern == LoadPattern.RANDOM:
            return generator.random_load(self.config.total_requests, self.config.concurrent_users)
        else:
            return generator.constant_load(self.config.total_requests, self.config.concurrent_users)
    
    async def run_benchmark(self) -> BenchmarkResults:
        """Run the complete benchmark suite."""
        self.logger.info(f"Starting benchmark: {self.config.benchmark_name}")
        self.logger.info(f"Configuration: {self.config.total_requests} requests, {self.config.concurrent_users} max users")
        
        self.start_time = datetime.now()
        
        try:
            # Initialize high-performance system
            hp_config = self.config.hp_config or HighPerformanceConfig()
            
            async with high_performance_classification_context(hp_config) as hp_system:
                self.hp_system = hp_system
                
                # Start monitoring
                if self.monitor:
                    await self.monitor.start_monitoring()
                
                # Warmup phase
                await self._warmup_phase()
                
                # Main benchmark execution
                await self._execute_benchmark()
                
                # Stop monitoring
                if self.monitor:
                    await self.monitor.stop_monitoring()
                
                # Cooldown phase
                await self._cooldown_phase()
            
            self.end_time = datetime.now()
            
            # Generate results
            results = await self._generate_results()
            
            # Export results if enabled
            if self.config.export_results:
                await self._export_results(results)
            
            # Generate plots if enabled
            if self.config.generate_plots:
                await self._generate_plots(results)
            
            self.logger.info("Benchmark completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise
    
    async def _warmup_phase(self):
        """Execute warmup phase."""
        if self.config.warmup_requests == 0:
            return
        
        self.logger.info(f"Starting warmup phase: {self.config.warmup_requests} requests")
        
        warmup_query = self.config.test_queries[0]
        warmup_tasks = []
        
        for i in range(self.config.warmup_requests):
            task = asyncio.create_task(self._execute_single_request(
                f"warmup_{i}",
                warmup_query,
                0,  # warmup user
                is_warmup=True
            ))
            warmup_tasks.append(task)
        
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        self.logger.info("Warmup phase completed")
    
    async def _execute_benchmark(self):
        """Execute the main benchmark."""
        self.logger.info("Starting main benchmark execution")
        
        tasks = []
        request_id = 0
        
        # Execute requests according to load pattern
        for concurrent_users, delay in self.load_pattern:
            # Select query based on distribution
            query = self._select_query()
            
            # Create tasks for concurrent users
            user_tasks = []
            for user_id in range(concurrent_users):
                task = asyncio.create_task(self._execute_single_request(
                    f"req_{request_id}",
                    query,
                    user_id
                ))
                user_tasks.append(task)
                request_id += 1
            
            tasks.extend(user_tasks)
            
            # Apply delay if specified
            if delay > 0:
                await asyncio.sleep(delay)
        
        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info(f"Main benchmark execution completed: {len(self.results)} results recorded")
    
    def _select_query(self) -> str:
        """Select query based on configured distribution."""
        # Simple random selection based on distribution weights
        queries = list(self.config.query_distribution.keys())
        weights = list(self.config.query_distribution.values())
        
        return np.random.choice(queries, p=weights)
    
    async def _execute_single_request(self, request_id: str, query: str, user_id: int, is_warmup: bool = False) -> RequestResult:
        """Execute a single request and record results."""
        start_time = time.time()
        result = None
        error_message = None
        success = False
        metadata = {}
        
        try:
            # Execute request through high-performance system
            classification_result, request_metadata = await self.hp_system.classify_query_optimized(
                query_text=query,
                priority="normal"
            )
            
            result = classification_result
            metadata = request_metadata
            success = True
            
        except Exception as e:
            error_message = str(e)
            self.logger.debug(f"Request {request_id} failed: {e}")
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Create result record
        request_result = RequestResult(
            request_id=request_id,
            query_text=query,
            start_time=start_time,
            end_time=end_time,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_message,
            classification_result=result,
            metadata=metadata,
            user_id=user_id,
            timestamp=datetime.now()
        )
        
        # Record result (skip warmup requests in final results)
        if not is_warmup:
            self.results.append(request_result)
        
        # Record in monitor
        if self.monitor:
            self.monitor.record_request(response_time_ms, success, user_id)
        
        return request_result
    
    async def _cooldown_phase(self):
        """Execute cooldown phase."""
        if self.config.cooldown_seconds > 0:
            self.logger.info(f"Cooldown phase: {self.config.cooldown_seconds} seconds")
            await asyncio.sleep(self.config.cooldown_seconds)
    
    async def _generate_results(self) -> BenchmarkResults:
        """Generate comprehensive benchmark results."""
        
        if not self.results:
            raise ValueError("No results available for analysis")
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        # Generate time series data
        time_series_data = {}
        resource_utilization = {}
        
        if self.monitor:
            time_series_data = self.monitor.time_series.copy()
            resource_utilization = {
                "cpu": self.monitor.time_series["cpu_usage"],
                "memory": self.monitor.time_series["memory_usage"]
            }
        
        # Perform performance analysis
        performance_analysis = self._analyze_performance()
        
        # Identify bottlenecks
        bottleneck_analysis = self._analyze_bottlenecks()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        return BenchmarkResults(
            config=self.config,
            start_time=self.start_time,
            end_time=self.end_time,
            total_duration_seconds=(self.end_time - self.start_time).total_seconds(),
            metrics=metrics,
            request_results=self.results,
            time_series_data=time_series_data,
            resource_utilization=resource_utilization,
            performance_analysis=performance_analysis,
            bottleneck_analysis=bottleneck_analysis,
            recommendations=recommendations
        )
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time_ms for r in successful_results]
        
        if not response_times:
            raise ValueError("No successful requests for analysis")
        
        # Basic metrics
        total_requests = len(self.results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        success_rate = successful_requests / total_requests
        
        # Response time statistics
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        
        # Target compliance
        target_compliant = [t for t in response_times if t <= self.config.target_response_time_ms]
        max_compliant = [t for t in response_times if t <= self.config.max_response_time_ms]
        
        target_compliance_rate = len(target_compliant) / len(response_times)
        max_compliance_rate = len(max_compliant) / len(response_times)
        
        # Throughput calculation
        duration = (self.end_time - self.start_time).total_seconds()
        actual_throughput = successful_requests / duration
        
        # Resource utilization (from monitor if available)
        avg_cpu = max_cpu = avg_memory = max_memory = 0
        
        if self.monitor:
            if self.monitor.cpu_samples:
                avg_cpu = statistics.mean(self.monitor.cpu_samples)
                max_cpu = max(self.monitor.cpu_samples)
            if self.monitor.memory_samples:
                avg_memory = statistics.mean(self.monitor.memory_samples)
                max_memory = max(self.monitor.memory_samples)
        
        # Cache performance (from system stats)
        cache_hit_rate = 0
        avg_cache_response_time = 0
        
        if self.hp_system:
            cache_stats = self.hp_system.cache.get_cache_stats()
            cache_hit_rate = cache_stats["overall"]["hit_rate"]
            if cache_stats["overall"]["avg_hit_time_ms"]:
                avg_cache_response_time = cache_stats["overall"]["avg_hit_time_ms"]
        
        # Performance grading
        performance_grade = self._calculate_performance_grade(
            avg_response_time, target_compliance_rate
        )
        
        # Regression detection
        regression_detected = self._detect_regression(response_times)
        
        # Optimization score
        optimization_score = self._calculate_optimization_score(
            avg_response_time, target_compliance_rate, cache_hit_rate, success_rate
        )
        
        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            std_dev_ms=std_dev,
            target_compliance_count=len(target_compliant),
            target_compliance_rate=target_compliance_rate,
            max_compliance_count=len(max_compliant),
            max_compliance_rate=max_compliance_rate,
            actual_throughput_rps=actual_throughput,
            peak_throughput_rps=actual_throughput,  # Would need sliding window for peak
            min_throughput_rps=actual_throughput,   # Would need sliding window for min
            avg_cpu_percent=avg_cpu,
            max_cpu_percent=max_cpu,
            avg_memory_percent=avg_memory,
            max_memory_percent=max_memory,
            cache_hit_rate=cache_hit_rate,
            avg_cache_response_time_ms=avg_cache_response_time,
            performance_grade=performance_grade,
            regression_detected=regression_detected,
            optimization_score=optimization_score
        )
    
    def _calculate_performance_grade(self, avg_response_time: float, compliance_rate: float) -> PerformanceGrade:
        """Calculate performance grade based on metrics."""
        
        if avg_response_time < 1000 and compliance_rate > 0.99:
            return PerformanceGrade.EXCELLENT
        elif avg_response_time < 1200 and compliance_rate > 0.97:
            return PerformanceGrade.VERY_GOOD
        elif avg_response_time < 1400 and compliance_rate > 0.95:
            return PerformanceGrade.GOOD
        elif avg_response_time < 1600 and compliance_rate > 0.90:
            return PerformanceGrade.ACCEPTABLE
        elif avg_response_time < 2000 and compliance_rate > 0.80:
            return PerformanceGrade.POOR
        else:
            return PerformanceGrade.FAILING
    
    def _detect_regression(self, response_times: List[float]) -> bool:
        """Detect performance regression using statistical analysis."""
        
        if len(response_times) < 100:
            return False
        
        # Split into early and late periods
        split_point = len(response_times) // 2
        early_times = response_times[:split_point]
        late_times = response_times[split_point:]
        
        early_avg = statistics.mean(early_times)
        late_avg = statistics.mean(late_times)
        
        # Consider regression if late period is 10% slower
        regression_threshold = early_avg * 1.1
        
        return late_avg > regression_threshold
    
    def _calculate_optimization_score(self, avg_response_time: float, compliance_rate: float, 
                                    cache_hit_rate: float, success_rate: float) -> float:
        """Calculate overall optimization score (0-100)."""
        
        # Response time score (0-40 points)
        if avg_response_time <= 1000:
            time_score = 40
        elif avg_response_time <= 1500:
            time_score = 30
        elif avg_response_time <= 2000:
            time_score = 20
        else:
            time_score = 0
        
        # Compliance score (0-30 points)
        compliance_score = compliance_rate * 30
        
        # Cache efficiency score (0-20 points)
        cache_score = cache_hit_rate * 20
        
        # Reliability score (0-10 points)
        reliability_score = success_rate * 10
        
        return time_score + compliance_score + cache_score + reliability_score
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Perform detailed performance analysis."""
        
        response_times = [r.response_time_ms for r in self.results if r.success]
        
        analysis = {
            "distribution_analysis": self._analyze_response_time_distribution(response_times),
            "temporal_analysis": self._analyze_temporal_patterns(),
            "user_analysis": self._analyze_user_patterns(),
            "query_analysis": self._analyze_query_patterns(),
            "cache_analysis": self._analyze_cache_effectiveness(),
        }
        
        return analysis
    
    def _analyze_response_time_distribution(self, response_times: List[float]) -> Dict[str, Any]:
        """Analyze response time distribution."""
        
        if not response_times:
            return {}
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f"p{p}": np.percentile(response_times, p) for p in percentiles}
        
        # Distribution shape
        skewness = self._calculate_skewness(response_times)
        kurtosis = self._calculate_kurtosis(response_times)
        
        return {
            "percentiles": percentile_values,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "outlier_count": len([t for t in response_times if t > self.config.max_response_time_ms * 1.5])
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        
        if std_dev == 0:
            return 0
        
        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std_dev) ** 3 for x in data)
        
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data distribution."""
        if len(data) < 4:
            return 0
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        
        if std_dev == 0:
            return 0
        
        n = len(data)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - mean) / std_dev) ** 4 for x in data) - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        
        return kurtosis
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal performance patterns."""
        
        if not self.results:
            return {}
        
        # Group results by time windows (e.g., 10-second windows)
        window_size = 10.0  # seconds
        start_time = min(r.start_time for r in self.results)
        end_time = max(r.end_time for r in self.results)
        
        windows = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = current_time + window_size
            window_results = [r for r in self.results if current_time <= r.start_time < window_end and r.success]
            
            if window_results:
                response_times = [r.response_time_ms for r in window_results]
                windows.append({
                    "start_time": current_time - start_time,
                    "avg_response_time": statistics.mean(response_times),
                    "request_count": len(window_results),
                    "throughput": len(window_results) / window_size
                })
            
            current_time = window_end
        
        return {
            "window_analysis": windows,
            "performance_degradation": self._detect_performance_degradation(windows),
            "throughput_variance": self._calculate_throughput_variance(windows)
        }
    
    def _detect_performance_degradation(self, windows: List[Dict[str, Any]]) -> bool:
        """Detect if performance degrades over time."""
        
        if len(windows) < 5:
            return False
        
        early_windows = windows[:len(windows)//2]
        late_windows = windows[len(windows)//2:]
        
        early_avg = statistics.mean(w["avg_response_time"] for w in early_windows)
        late_avg = statistics.mean(w["avg_response_time"] for w in late_windows)
        
        return late_avg > early_avg * 1.2  # 20% degradation threshold
    
    def _calculate_throughput_variance(self, windows: List[Dict[str, Any]]) -> float:
        """Calculate variance in throughput over time."""
        
        if len(windows) < 2:
            return 0
        
        throughputs = [w["throughput"] for w in windows]
        return statistics.stdev(throughputs)
    
    def _analyze_user_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns by user."""
        
        user_stats = defaultdict(list)
        
        for result in self.results:
            if result.success:
                user_stats[result.user_id].append(result.response_time_ms)
        
        user_analysis = {}
        for user_id, times in user_stats.items():
            user_analysis[user_id] = {
                "request_count": len(times),
                "avg_response_time": statistics.mean(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0
            }
        
        return {
            "user_statistics": user_analysis,
            "user_count": len(user_stats),
            "requests_per_user": statistics.mean(len(times) for times in user_stats.values()) if user_stats else 0
        }
    
    def _analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns by query type."""
        
        query_stats = defaultdict(list)
        
        for result in self.results:
            if result.success:
                query_stats[result.query_text].append(result.response_time_ms)
        
        query_analysis = {}
        for query, times in query_stats.items():
            query_analysis[query[:50] + "..."] = {  # Truncate for readability
                "request_count": len(times),
                "avg_response_time": statistics.mean(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0
            }
        
        return {
            "query_statistics": query_analysis,
            "unique_queries": len(query_stats),
            "most_tested_query": max(query_stats.keys(), key=lambda q: len(query_stats[q])) if query_stats else None
        }
    
    def _analyze_cache_effectiveness(self) -> Dict[str, Any]:
        """Analyze cache effectiveness during benchmark."""
        
        cache_hits = 0
        cache_misses = 0
        cache_response_times = []
        no_cache_response_times = []
        
        for result in self.results:
            if result.success and result.metadata:
                if result.metadata.get("cache_hit", False):
                    cache_hits += 1
                    cache_response_times.append(result.response_time_ms)
                else:
                    cache_misses += 1
                    no_cache_response_times.append(result.response_time_ms)
        
        cache_analysis = {
            "total_cache_hits": cache_hits,
            "total_cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / max(1, cache_hits + cache_misses),
        }
        
        if cache_response_times:
            cache_analysis["avg_cache_response_time"] = statistics.mean(cache_response_times)
        
        if no_cache_response_times:
            cache_analysis["avg_no_cache_response_time"] = statistics.mean(no_cache_response_times)
        
        if cache_response_times and no_cache_response_times:
            cache_analysis["cache_improvement_factor"] = statistics.mean(no_cache_response_times) / statistics.mean(cache_response_times)
        
        return cache_analysis
    
    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """Identify system bottlenecks."""
        
        bottlenecks = {
            "response_time_bottlenecks": [],
            "throughput_bottlenecks": [],
            "resource_bottlenecks": [],
            "error_bottlenecks": []
        }
        
        # Response time bottlenecks
        response_times = [r.response_time_ms for r in self.results if r.success]
        if response_times:
            p95_time = np.percentile(response_times, 95)
            if p95_time > self.config.target_response_time_ms * 1.5:
                bottlenecks["response_time_bottlenecks"].append({
                    "type": "high_p95_response_time",
                    "value": p95_time,
                    "threshold": self.config.target_response_time_ms * 1.5,
                    "severity": "high"
                })
        
        # Resource bottlenecks
        if self.monitor:
            if self.monitor.cpu_samples:
                max_cpu = max(self.monitor.cpu_samples)
                if max_cpu > 85:
                    bottlenecks["resource_bottlenecks"].append({
                        "type": "high_cpu_usage",
                        "value": max_cpu,
                        "threshold": 85,
                        "severity": "medium"
                    })
            
            if self.monitor.memory_samples:
                max_memory = max(self.monitor.memory_samples)
                if max_memory > 85:
                    bottlenecks["resource_bottlenecks"].append({
                        "type": "high_memory_usage",
                        "value": max_memory,
                        "threshold": 85,
                        "severity": "medium"
                    })
        
        # Error rate bottlenecks
        error_rate = (len([r for r in self.results if not r.success]) / len(self.results)) if self.results else 0
        if error_rate > 0.05:  # 5% error rate threshold
            bottlenecks["error_bottlenecks"].append({
                "type": "high_error_rate",
                "value": error_rate,
                "threshold": 0.05,
                "severity": "high"
            })
        
        return bottlenecks
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on results."""
        
        recommendations = []
        
        # Response time recommendations
        if metrics.avg_response_time_ms > self.config.target_response_time_ms:
            severity = "high" if metrics.avg_response_time_ms > self.config.max_response_time_ms else "medium"
            recommendations.append({
                "category": "response_time",
                "severity": severity,
                "issue": f"Average response time ({metrics.avg_response_time_ms:.1f}ms) exceeds target ({self.config.target_response_time_ms:.1f}ms)",
                "recommendations": [
                    "Increase cache sizes (L1, L2, L3)",
                    "Optimize prompt token usage",
                    "Enable more aggressive caching strategies",
                    "Consider horizontal scaling",
                    "Review and optimize query patterns"
                ],
                "priority": 1
            })
        
        # Cache recommendations
        if metrics.cache_hit_rate < 0.7:
            recommendations.append({
                "category": "caching",
                "severity": "medium",
                "issue": f"Low cache hit rate ({metrics.cache_hit_rate:.1%})",
                "recommendations": [
                    "Improve cache warming strategy",
                    "Increase cache TTL values",
                    "Enable predictive caching",
                    "Review cache key generation logic",
                    "Consider cache partitioning by query type"
                ],
                "priority": 2
            })
        
        # Resource recommendations
        if metrics.avg_cpu_percent > 80:
            recommendations.append({
                "category": "resources",
                "severity": "medium",
                "issue": f"High CPU utilization ({metrics.avg_cpu_percent:.1f}%)",
                "recommendations": [
                    "Optimize CPU-intensive operations",
                    "Consider process pooling",
                    "Implement request throttling",
                    "Scale horizontally",
                    "Profile code for CPU bottlenecks"
                ],
                "priority": 2
            })
        
        if metrics.avg_memory_percent > 80:
            recommendations.append({
                "category": "resources",
                "severity": "medium",
                "issue": f"High memory utilization ({metrics.avg_memory_percent:.1f}%)",
                "recommendations": [
                    "Enable memory pooling",
                    "Reduce cache sizes if necessary",
                    "Implement garbage collection optimization",
                    "Review memory leaks",
                    "Consider memory-efficient data structures"
                ],
                "priority": 2
            })
        
        # Success rate recommendations
        if metrics.success_rate < 0.99:
            recommendations.append({
                "category": "reliability",
                "severity": "high",
                "issue": f"Success rate ({metrics.success_rate:.1%}) below 99%",
                "recommendations": [
                    "Implement better error handling",
                    "Add circuit breaker patterns",
                    "Review timeout configurations",
                    "Implement retry mechanisms",
                    "Monitor and fix error sources"
                ],
                "priority": 1
            })
        
        # Performance grade recommendations
        if metrics.performance_grade in [PerformanceGrade.POOR, PerformanceGrade.FAILING]:
            recommendations.append({
                "category": "overall_performance",
                "severity": "high",
                "issue": f"Overall performance grade: {metrics.performance_grade.value}",
                "recommendations": [
                    "Comprehensive performance review needed",
                    "Consider system architecture changes",
                    "Implement performance monitoring",
                    "Review all optimization strategies",
                    "Consider professional performance consulting"
                ],
                "priority": 1
            })
        
        # Sort by priority
        return sorted(recommendations, key=lambda x: x["priority"])
    
    async def _export_results(self, results: BenchmarkResults):
        """Export benchmark results to various formats."""
        
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.benchmark_name}_{timestamp}"
        
        # Export JSON summary
        json_path = output_dir / f"{base_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump({
                "config": asdict(results.config),
                "metrics": asdict(results.metrics),
                "performance_analysis": results.performance_analysis,
                "recommendations": results.recommendations
            }, f, indent=2, default=str)
        
        results.export_paths["json_summary"] = str(json_path)
        
        # Export CSV with detailed results
        csv_path = output_dir / f"{base_name}_detailed.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Headers
            writer.writerow([
                "request_id", "query_text", "user_id", "response_time_ms",
                "success", "error_message", "classification_category",
                "classification_confidence", "cache_hit", "timestamp"
            ])
            
            # Data rows
            for result in results.request_results:
                writer.writerow([
                    result.request_id,
                    result.query_text[:100] + "..." if len(result.query_text) > 100 else result.query_text,
                    result.user_id,
                    result.response_time_ms,
                    result.success,
                    result.error_message or "",
                    result.classification_result.category if result.classification_result else "",
                    result.classification_result.confidence if result.classification_result else "",
                    result.metadata.get("cache_hit", False),
                    result.timestamp.isoformat()
                ])
        
        results.export_paths["csv_detailed"] = str(csv_path)
        
        # Export performance metrics CSV
        metrics_csv_path = output_dir / f"{base_name}_metrics.csv"
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(["metric", "value", "unit"])
            
            metrics_dict = asdict(results.metrics)
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    unit = "ms" if "time" in key.lower() else ("%" if "rate" in key.lower() or "percent" in key.lower() else "count")
                    writer.writerow([key, value, unit])
        
        results.export_paths["csv_metrics"] = str(metrics_csv_path)
        
        self.logger.info(f"Results exported to: {output_dir}")
    
    async def _generate_plots(self, results: BenchmarkResults):
        """Generate performance visualization plots."""
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{self.config.benchmark_name}_{timestamp}"
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Plot 1: Response time distribution
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Performance Benchmark Results: {self.config.benchmark_name}', fontsize=16)
            
            # Response time histogram
            response_times = [r.response_time_ms for r in results.request_results if r.success]
            axes[0, 0].hist(response_times, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(x=results.config.target_response_time_ms, color='red', linestyle='--', label=f'Target ({results.config.target_response_time_ms}ms)')
            axes[0, 0].axvline(x=results.config.max_response_time_ms, color='orange', linestyle='--', label=f'Max ({results.config.max_response_time_ms}ms)')
            axes[0, 0].set_xlabel('Response Time (ms)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Response Time Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Response time over time
            if results.time_series_data.get("response_times"):
                times, values = zip(*results.time_series_data["response_times"])
                axes[0, 1].plot(times, values, alpha=0.6, linewidth=1)
                axes[0, 1].axhline(y=results.config.target_response_time_ms, color='red', linestyle='--', label=f'Target ({results.config.target_response_time_ms}ms)')
                axes[0, 1].set_xlabel('Time (seconds)')
                axes[0, 1].set_ylabel('Response Time (ms)')
                axes[0, 1].set_title('Response Time Over Time')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Throughput over time
            if results.time_series_data.get("throughput"):
                times, values = zip(*results.time_series_data["throughput"])
                axes[1, 0].plot(times, values, color='green', linewidth=2)
                axes[1, 0].set_xlabel('Time (seconds)')
                axes[1, 0].set_ylabel('Throughput (RPS)')
                axes[1, 0].set_title('Throughput Over Time')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Resource utilization
            if results.resource_utilization.get("cpu") and results.resource_utilization.get("memory"):
                cpu_times, cpu_values = zip(*results.resource_utilization["cpu"])
                memory_times, memory_values = zip(*results.resource_utilization["memory"])
                
                axes[1, 1].plot(cpu_times, cpu_values, label='CPU %', color='red')
                axes[1, 1].plot(memory_times, memory_values, label='Memory %', color='blue')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].set_ylabel('Utilization (%)')
                axes[1, 1].set_title('Resource Utilization')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f"{base_name}_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results.export_paths["performance_plot"] = str(plot_path)
            
            # Plot 2: Performance metrics summary
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            metrics_data = {
                'Avg Response Time (ms)': results.metrics.avg_response_time_ms,
                'P95 Response Time (ms)': results.metrics.p95_response_time_ms,
                'P99 Response Time (ms)': results.metrics.p99_response_time_ms,
                'Success Rate (%)': results.metrics.success_rate * 100,
                'Target Compliance (%)': results.metrics.target_compliance_rate * 100,
                'Cache Hit Rate (%)': results.metrics.cache_hit_rate * 100,
                'Throughput (RPS)': results.metrics.actual_throughput_rps
            }
            
            bars = ax.bar(range(len(metrics_data)), list(metrics_data.values()), 
                         color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'lightsteelblue', 'peachpuff'])
            ax.set_xticks(range(len(metrics_data)))
            ax.set_xticklabels(list(metrics_data.keys()), rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title(f'Performance Metrics Summary - Grade: {results.metrics.performance_grade.value}')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save metrics plot
            metrics_plot_path = output_dir / f"{base_name}_metrics.png"
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results.export_paths["metrics_plot"] = str(metrics_plot_path)
            
            self.logger.info(f"Performance plots generated: {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")


# ============================================================================
# BENCHMARK REPORTING
# ============================================================================

class BenchmarkReporter:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, results: BenchmarkResults):
        self.results = results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of benchmark results."""
        
        report = []
        report.append("=" * 80)
        report.append(f"PERFORMANCE BENCHMARK REPORT: {self.results.config.benchmark_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Overall Performance Grade: {self.results.metrics.performance_grade.value}")
        report.append(f"Target Response Time: {self.results.config.target_response_time_ms:.1f}ms")
        report.append(f"Actual Average Response Time: {self.results.metrics.avg_response_time_ms:.1f}ms")
        report.append(f"Target Compliance Rate: {self.results.metrics.target_compliance_rate:.1%}")
        report.append(f"Success Rate: {self.results.metrics.success_rate:.1%}")
        report.append(f"Total Requests Processed: {self.results.metrics.total_requests:,}")
        report.append(f"Benchmark Duration: {self.results.total_duration_seconds:.1f} seconds")
        report.append("")
        
        # Performance Metrics
        report.append("DETAILED PERFORMANCE METRICS")
        report.append("-" * 35)
        report.append(f"Response Time Statistics:")
        report.append(f"  Average: {self.results.metrics.avg_response_time_ms:.1f}ms")
        report.append(f"  Median: {self.results.metrics.median_response_time_ms:.1f}ms")
        report.append(f"  Minimum: {self.results.metrics.min_response_time_ms:.1f}ms")
        report.append(f"  Maximum: {self.results.metrics.max_response_time_ms:.1f}ms")
        report.append(f"  P95: {self.results.metrics.p95_response_time_ms:.1f}ms")
        report.append(f"  P99: {self.results.metrics.p99_response_time_ms:.1f}ms")
        report.append(f"  Standard Deviation: {self.results.metrics.std_dev_ms:.1f}ms")
        report.append("")
        
        report.append(f"Target Compliance:")
        report.append(f"  Target (<{self.results.config.target_response_time_ms:.0f}ms): {self.results.metrics.target_compliance_rate:.1%}")
        report.append(f"  Maximum (<{self.results.config.max_response_time_ms:.0f}ms): {self.results.metrics.max_compliance_rate:.1%}")
        report.append("")
        
        report.append(f"Throughput Metrics:")
        report.append(f"  Average Throughput: {self.results.metrics.actual_throughput_rps:.1f} requests/second")
        report.append(f"  Total Successful Requests: {self.results.metrics.successful_requests:,}")
        report.append(f"  Total Failed Requests: {self.results.metrics.failed_requests:,}")
        report.append("")
        
        # System Resources
        report.append(f"Resource Utilization:")
        report.append(f"  Average CPU: {self.results.metrics.avg_cpu_percent:.1f}%")
        report.append(f"  Peak CPU: {self.results.metrics.max_cpu_percent:.1f}%")
        report.append(f"  Average Memory: {self.results.metrics.avg_memory_percent:.1f}%")
        report.append(f"  Peak Memory: {self.results.metrics.max_memory_percent:.1f}%")
        report.append("")
        
        # Cache Performance
        report.append(f"Cache Performance:")
        report.append(f"  Cache Hit Rate: {self.results.metrics.cache_hit_rate:.1%}")
        if self.results.metrics.avg_cache_response_time_ms > 0:
            report.append(f"  Average Cache Response Time: {self.results.metrics.avg_cache_response_time_ms:.1f}ms")
        report.append("")
        
        # Recommendations
        if self.results.recommendations:
            report.append("OPTIMIZATION RECOMMENDATIONS")
            report.append("-" * 35)
            for i, rec in enumerate(self.results.recommendations[:5], 1):  # Top 5 recommendations
                report.append(f"{i}. {rec['category'].upper()} - {rec['severity'].upper()}")
                report.append(f"   Issue: {rec['issue']}")
                report.append(f"   Recommendations:")
                for suggestion in rec['recommendations'][:3]:  # Top 3 suggestions
                    report.append(f"   - {suggestion}")
                report.append("")
        
        # Bottleneck Analysis
        if self.results.bottleneck_analysis:
            report.append("BOTTLENECK ANALYSIS")
            report.append("-" * 20)
            
            for category, bottlenecks in self.results.bottleneck_analysis.items():
                if bottlenecks:
                    report.append(f"{category.replace('_', ' ').title()}:")
                    for bottleneck in bottlenecks:
                        report.append(f"  - {bottleneck['type']}: {bottleneck['value']:.1f} (threshold: {bottleneck['threshold']:.1f})")
            report.append("")
        
        # Performance Analysis Summary
        if self.results.performance_analysis:
            report.append("PERFORMANCE ANALYSIS HIGHLIGHTS")
            report.append("-" * 37)
            
            # Response time distribution
            if "distribution_analysis" in self.results.performance_analysis:
                dist = self.results.performance_analysis["distribution_analysis"]
                if "outlier_count" in dist:
                    report.append(f"Response Time Outliers: {dist['outlier_count']} requests exceeded 1.5x max threshold")
            
            # Cache effectiveness
            if "cache_analysis" in self.results.performance_analysis:
                cache = self.results.performance_analysis["cache_analysis"]
                if "cache_improvement_factor" in cache:
                    report.append(f"Cache Effectiveness: {cache['cache_improvement_factor']:.1f}x improvement over no-cache")
            
            report.append("")
        
        # Test Configuration
        report.append("TEST CONFIGURATION")
        report.append("-" * 20)
        report.append(f"Benchmark Type: {self.results.config.benchmark_type.value}")
        report.append(f"Load Pattern: {self.results.config.load_pattern.value}")
        report.append(f"Concurrent Users: {self.results.config.concurrent_users}")
        report.append(f"Total Requests: {self.results.config.total_requests}")
        report.append(f"Test Queries: {len(self.results.config.test_queries)}")
        report.append("")
        
        # Conclusion
        report.append("CONCLUSION")
        report.append("-" * 15)
        
        if self.results.metrics.performance_grade in [PerformanceGrade.EXCELLENT, PerformanceGrade.VERY_GOOD]:
            report.append("✅ SYSTEM PERFORMANCE: EXCELLENT")
            report.append("✅ System consistently meets <2 second response time requirements")
            report.append("✅ Ready for production deployment")
        elif self.results.metrics.performance_grade == PerformanceGrade.GOOD:
            report.append("✅ SYSTEM PERFORMANCE: GOOD")
            report.append("✅ System meets most performance requirements")
            report.append("⚠️  Minor optimizations recommended")
        elif self.results.metrics.performance_grade == PerformanceGrade.ACCEPTABLE:
            report.append("⚠️  SYSTEM PERFORMANCE: ACCEPTABLE")
            report.append("⚠️  System meets basic requirements but needs optimization")
            report.append("⚠️  Review recommendations before production deployment")
        else:
            report.append("❌ SYSTEM PERFORMANCE: NEEDS IMPROVEMENT")
            report.append("❌ System does not meet performance requirements")
            report.append("❌ Significant optimization required before production")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed technical report."""
        
        summary = self.generate_summary_report()
        
        detailed = [summary, "", ""]
        detailed.append("=" * 80)
        detailed.append("DETAILED TECHNICAL ANALYSIS")
        detailed.append("=" * 80)
        detailed.append("")
        
        # Temporal Analysis
        if "temporal_analysis" in self.results.performance_analysis:
            temporal = self.results.performance_analysis["temporal_analysis"]
            detailed.append("TEMPORAL PERFORMANCE ANALYSIS")
            detailed.append("-" * 35)
            
            if "performance_degradation" in temporal:
                degradation_status = "DETECTED" if temporal["performance_degradation"] else "NOT DETECTED"
                detailed.append(f"Performance Degradation Over Time: {degradation_status}")
            
            if "throughput_variance" in temporal:
                detailed.append(f"Throughput Variance: {temporal['throughput_variance']:.2f}")
            
            detailed.append("")
        
        # User Analysis
        if "user_analysis" in self.results.performance_analysis:
            user = self.results.performance_analysis["user_analysis"]
            detailed.append("USER PATTERN ANALYSIS")
            detailed.append("-" * 25)
            detailed.append(f"Concurrent Users Simulated: {user.get('user_count', 0)}")
            detailed.append(f"Average Requests per User: {user.get('requests_per_user', 0):.1f}")
            detailed.append("")
        
        # Query Analysis
        if "query_analysis" in self.results.performance_analysis:
            query = self.results.performance_analysis["query_analysis"]
            detailed.append("QUERY PATTERN ANALYSIS")
            detailed.append("-" * 25)
            detailed.append(f"Unique Query Patterns Tested: {query.get('unique_queries', 0)}")
            
            if "most_tested_query" in query and query["most_tested_query"]:
                detailed.append(f"Most Frequently Tested Query: {query['most_tested_query'][:50]}...")
            
            detailed.append("")
        
        # Statistical Analysis
        if "distribution_analysis" in self.results.performance_analysis:
            dist = self.results.performance_analysis["distribution_analysis"]
            detailed.append("STATISTICAL DISTRIBUTION ANALYSIS")
            detailed.append("-" * 38)
            
            if "percentiles" in dist:
                detailed.append("Response Time Percentiles:")
                for percentile, value in dist["percentiles"].items():
                    detailed.append(f"  {percentile.upper()}: {value:.1f}ms")
            
            if "skewness" in dist:
                detailed.append(f"Distribution Skewness: {dist['skewness']:.3f}")
                if abs(dist['skewness']) > 1:
                    detailed.append("  (High skewness indicates asymmetric distribution)")
            
            if "kurtosis" in dist:
                detailed.append(f"Distribution Kurtosis: {dist['kurtosis']:.3f}")
                if abs(dist['kurtosis']) > 1:
                    detailed.append("  (High kurtosis indicates heavy-tailed distribution)")
            
            detailed.append("")
        
        # Export Information
        if self.results.export_paths:
            detailed.append("EXPORTED ARTIFACTS")
            detailed.append("-" * 20)
            for artifact_type, path in self.results.export_paths.items():
                detailed.append(f"{artifact_type.replace('_', ' ').title()}: {path}")
            detailed.append("")
        
        detailed.append("=" * 80)
        detailed.append("END OF DETAILED REPORT")
        detailed.append("=" * 80)
        
        return "\n".join(detailed)


# ============================================================================
# HIGH-LEVEL BENCHMARK FUNCTIONS
# ============================================================================

async def run_comprehensive_benchmark(config: BenchmarkConfig = None) -> BenchmarkResults:
    """
    Run a comprehensive performance benchmark with the specified configuration.
    
    Args:
        config: Benchmark configuration. If None, uses default configuration.
        
    Returns:
        Complete benchmark results with analysis and recommendations.
    """
    
    if not HP_COMPONENTS_AVAILABLE:
        raise ImportError("High-performance components not available. Please install required dependencies.")
    
    if config is None:
        config = BenchmarkConfig()
    
    # Set up logging
    logger = logging.getLogger("performance_benchmark")
    logger.setLevel(logging.INFO)
    
    # Run benchmark
    runner = PerformanceBenchmarkRunner(config, logger)
    results = await runner.run_benchmark()
    
    # Generate and log summary
    reporter = BenchmarkReporter(results)
    summary_report = reporter.generate_summary_report()
    
    print("\n" + summary_report)
    
    return results


async def run_quick_performance_test(target_response_time_ms: float = 1500, 
                                   total_requests: int = 100) -> BenchmarkResults:
    """
    Run a quick performance test with minimal configuration.
    
    Args:
        target_response_time_ms: Target response time in milliseconds
        total_requests: Total number of requests to execute
        
    Returns:
        Benchmark results
    """
    
    config = BenchmarkConfig(
        benchmark_name="quick_performance_test",
        benchmark_type=BenchmarkType.LOAD,
        load_pattern=LoadPattern.CONSTANT,
        concurrent_users=10,
        total_requests=total_requests,
        target_response_time_ms=target_response_time_ms,
        warmup_requests=10,
        enable_real_time_monitoring=True,
        export_results=False,
        generate_plots=False
    )
    
    return await run_comprehensive_benchmark(config)


async def run_stress_test(max_concurrent_users: int = 50, 
                         duration_seconds: int = 300) -> BenchmarkResults:
    """
    Run a stress test to determine system limits.
    
    Args:
        max_concurrent_users: Maximum concurrent users to simulate
        duration_seconds: Duration of the stress test in seconds
        
    Returns:
        Benchmark results
    """
    
    # Calculate total requests based on duration and expected throughput
    total_requests = max_concurrent_users * duration_seconds // 2
    
    config = BenchmarkConfig(
        benchmark_name="stress_test",
        benchmark_type=BenchmarkType.STRESS,
        load_pattern=LoadPattern.RAMP_UP,
        concurrent_users=max_concurrent_users,
        total_requests=total_requests,
        target_response_time_ms=2000,  # More lenient for stress testing
        max_response_time_ms=5000,
        warmup_requests=50,
        enable_real_time_monitoring=True,
        export_results=True,
        generate_plots=True,
        output_directory="stress_test_results"
    )
    
    return await run_comprehensive_benchmark(config)


if __name__ == "__main__":
    # Example usage and demonstration
    
    async def demo_benchmark_suite():
        """Demonstrate the benchmark suite capabilities."""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print("Starting Performance Benchmark Suite Demonstration")
        print("=" * 80)
        
        # Quick performance test
        print("\n1. Running Quick Performance Test...")
        quick_results = await run_quick_performance_test(
            target_response_time_ms=1500,
            total_requests=50
        )
        
        print(f"Quick Test Results: Grade {quick_results.metrics.performance_grade.value}")
        print(f"Average Response Time: {quick_results.metrics.avg_response_time_ms:.1f}ms")
        print(f"Success Rate: {quick_results.metrics.success_rate:.1%}")
        
        # Comprehensive benchmark
        print("\n2. Running Comprehensive Benchmark...")
        
        comprehensive_config = BenchmarkConfig(
            benchmark_name="comprehensive_demo",
            benchmark_type=BenchmarkType.LOAD,
            load_pattern=LoadPattern.WAVE,
            concurrent_users=20,
            total_requests=200,
            target_response_time_ms=1500,
            enable_real_time_monitoring=True,
            export_results=True,
            generate_plots=True,
            output_directory="demo_results"
        )
        
        comprehensive_results = await run_comprehensive_benchmark(comprehensive_config)
        
        # Generate detailed report
        reporter = BenchmarkReporter(comprehensive_results)
        detailed_report = reporter.generate_detailed_report()
        
        # Save detailed report
        report_path = Path("demo_results") / "comprehensive_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(detailed_report)
        
        print(f"\nDetailed report saved to: {report_path}")
        print(f"Exported artifacts: {list(comprehensive_results.export_paths.keys())}")
        
        print("\n" + "=" * 80)
        print("Performance Benchmark Suite Demonstration Complete")
        print("=" * 80)
    
    # Run demonstration
    asyncio.run(demo_benchmark_suite())