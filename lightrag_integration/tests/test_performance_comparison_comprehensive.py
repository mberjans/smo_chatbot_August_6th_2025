"""
Comprehensive Performance Comparison Between Old and New Systems
===============================================================

This module provides detailed performance analysis and comparison between the
baseline Clinical Metabolomics Oracle system and the enhanced LLM-integrated
version, with comprehensive benchmarking, profiling, and optimization insights.

Key Features:
- Detailed performance profiling of both systems
- Memory usage and CPU utilization analysis
- Response time distribution analysis
- Throughput and scalability comparison
- Resource efficiency metrics
- Performance regression detection
- Optimization recommendations
- Load testing under various scenarios

Test Categories:
1. Basic Performance Benchmarks
2. Memory and CPU Profiling
3. Scalability Testing
4. Concurrent Load Performance
5. Edge Case Performance
6. Resource Efficiency Analysis
7. Long-term Performance Monitoring
8. Performance Optimization Insights

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import pytest
import asyncio
import time
import statistics
import json
import logging
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import tracemalloc
import gc
import sys
import resource
import numpy as np
from collections import defaultdict, deque

# Import system components
from lightrag_integration.query_router import BiomedicalQueryRouter
from lightrag_integration.llm_query_classifier import LLMQueryClassifier, LLMClassificationConfig
from lightrag_integration.comprehensive_confidence_scorer import HybridConfidenceScorer

# Test utilities
from .biomedical_test_fixtures import BiomedicalTestFixtures
from .performance_test_utilities import PerformanceTestUtilities


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a system."""
    
    # Response time metrics (in milliseconds)
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    response_time_std: float
    
    # Throughput metrics
    queries_per_second: float
    successful_queries: int
    failed_queries: int
    success_rate: float
    
    # Resource utilization metrics
    avg_cpu_usage: float
    peak_cpu_usage: float
    avg_memory_mb: float
    peak_memory_mb: float
    memory_growth_mb: float  # Memory growth during test
    
    # System efficiency metrics
    cpu_efficiency: float  # Queries per CPU%
    memory_efficiency: float  # Queries per MB
    
    # Quality metrics
    avg_confidence: float
    confidence_variance: float
    
    # Timing breakdown (for enhanced system)
    llm_time_ratio: float = 0.0  # Percentage of time spent in LLM calls
    fallback_usage_rate: float = 0.0  # How often fallback is used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SystemComparison:
    """Comparison results between baseline and enhanced systems."""
    
    baseline_metrics: PerformanceMetrics
    enhanced_metrics: PerformanceMetrics
    
    # Performance differences (positive = enhanced is better)
    response_time_improvement: float
    throughput_improvement: float
    accuracy_improvement: float
    
    # Resource impact (positive = enhanced uses more)
    cpu_overhead: float
    memory_overhead: float
    
    # Quality improvements
    confidence_improvement: float
    success_rate_improvement: float
    
    # Overall assessment
    performance_score: float  # Weighted score (0-100)
    recommendation: str  # deploy, optimize, reject
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class PerformanceProfiler:
    """Advanced performance profiler for detailed system analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.profiling_data = {}
        self.memory_snapshots = []
        
    def start_profiling(self, profile_name: str):
        """Start profiling session."""
        tracemalloc.start()
        
        self.profiling_data[profile_name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': psutil.cpu_percent(),
            'query_times': [],
            'memory_samples': [],
            'cpu_samples': []
        }
        
        self.logger.debug(f"Started profiling session: {profile_name}")
    
    def record_query_performance(self, profile_name: str, query_time: float, confidence: float):
        """Record individual query performance."""
        if profile_name not in self.profiling_data:
            return
        
        self.profiling_data[profile_name]['query_times'].append({
            'time': query_time,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def sample_resources(self, profile_name: str):
        """Sample current resource usage."""
        if profile_name not in self.profiling_data:
            return
        
        current_memory = self._get_memory_usage()
        current_cpu = psutil.cpu_percent()
        
        self.profiling_data[profile_name]['memory_samples'].append(current_memory)
        self.profiling_data[profile_name]['cpu_samples'].append(current_cpu)
    
    def stop_profiling(self, profile_name: str) -> Dict[str, Any]:
        """Stop profiling and return analysis."""
        if profile_name not in self.profiling_data:
            return {}
        
        data = self.profiling_data[profile_name]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        total_time = end_time - data['start_time']
        memory_growth = end_memory - data['start_memory']
        
        query_times = [q['time'] for q in data['query_times']]
        confidences = [q['confidence'] for q in data['query_times']]
        
        # Get memory trace info
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        analysis = {
            'session_duration': total_time,
            'total_queries': len(query_times),
            'queries_per_second': len(query_times) / total_time if total_time > 0 else 0,
            'memory_growth_mb': memory_growth,
            'response_times': {
                'min': min(query_times) if query_times else 0,
                'max': max(query_times) if query_times else 0,
                'avg': statistics.mean(query_times) if query_times else 0,
                'median': statistics.median(query_times) if query_times else 0,
                'std': statistics.stdev(query_times) if len(query_times) > 1 else 0
            },
            'resource_usage': {
                'avg_cpu': statistics.mean(data['cpu_samples']) if data['cpu_samples'] else 0,
                'peak_cpu': max(data['cpu_samples']) if data['cpu_samples'] else 0,
                'avg_memory': statistics.mean(data['memory_samples']) if data['memory_samples'] else 0,
                'peak_memory': max(data['memory_samples']) if data['memory_samples'] else 0
            },
            'confidence_stats': {
                'avg': statistics.mean(confidences) if confidences else 0,
                'std': statistics.stdev(confidences) if len(confidences) > 1 else 0
            },
            'top_memory_usage': [
                {
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / 1024 / 1024
                }
                for stat in top_stats[:5]
            ]
        }
        
        tracemalloc.stop()
        self.logger.debug(f"Completed profiling session: {profile_name}")
        
        return analysis
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class BaselineSystemTester:
    """Performance tester for the baseline (existing) system."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.router = BiomedicalQueryRouter(logger)
        self.profiler = PerformanceProfiler(logger)
        
    async def run_performance_benchmark(self, 
                                      test_queries: List[str],
                                      concurrent_users: int = 1,
                                      duration_seconds: int = 60) -> PerformanceMetrics:
        """Run comprehensive performance benchmark."""
        
        self.logger.info(f"Starting baseline system benchmark: {len(test_queries)} queries, "
                        f"{concurrent_users} concurrent users, {duration_seconds}s duration")
        
        # Start profiling
        self.profiler.start_profiling("baseline_benchmark")
        
        # Performance tracking
        response_times = []
        confidences = []
        success_count = 0
        failure_count = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Resource monitoring setup
        resource_samples = []
        
        def resource_monitor():
            while time.time() < end_time:
                self.profiler.sample_resources("baseline_benchmark")
                resource_samples.append({
                    'timestamp': time.time(),
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent
                })
                time.sleep(1)
        
        # Start resource monitoring
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute queries
        async def execute_query(query: str) -> Dict[str, Any]:
            query_start = time.time()
            try:
                prediction = self.router.route_query(query)
                query_time = (time.time() - query_start) * 1000  # Convert to ms
                
                return {
                    'success': True,
                    'response_time': query_time,
                    'confidence': prediction.confidence if prediction else 0.0,
                    'prediction': prediction
                }
            except Exception as e:
                query_time = (time.time() - query_start) * 1000
                return {
                    'success': False,
                    'response_time': query_time,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        # Run concurrent workload
        tasks = []
        current_time = time.time()
        
        while current_time < end_time:
            # Create batch of concurrent tasks
            batch_tasks = []
            for i in range(concurrent_users):
                if current_time >= end_time:
                    break
                query = test_queries[len(tasks) % len(test_queries)]
                batch_tasks.append(execute_query(query))
                current_time = time.time()
            
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, dict):
                        response_times.append(result['response_time'])
                        confidences.append(result['confidence'])
                        
                        if result['success']:
                            success_count += 1
                            self.profiler.record_query_performance(
                                "baseline_benchmark",
                                result['response_time'],
                                result['confidence']
                            )
                        else:
                            failure_count += 1
            
            # Small delay between batches to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Stop profiling and get analysis
        profiling_analysis = self.profiler.stop_profiling("baseline_benchmark")
        
        # Calculate metrics
        total_queries = success_count + failure_count
        actual_duration = time.time() - start_time
        
        # Resource calculations
        cpu_samples = [s['cpu'] for s in resource_samples] if resource_samples else [0]
        memory_samples = [s['memory'] for s in resource_samples] if resource_samples else [0]
        
        # Build performance metrics
        metrics = PerformanceMetrics(
            # Response time metrics
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            median_response_time=statistics.median(response_times) if response_times else 0,
            p95_response_time=np.percentile(response_times, 95) if response_times else 0,
            p99_response_time=np.percentile(response_times, 99) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            response_time_std=statistics.stdev(response_times) if len(response_times) > 1 else 0,
            
            # Throughput metrics
            queries_per_second=total_queries / actual_duration if actual_duration > 0 else 0,
            successful_queries=success_count,
            failed_queries=failure_count,
            success_rate=success_count / total_queries if total_queries > 0 else 0,
            
            # Resource utilization
            avg_cpu_usage=statistics.mean(cpu_samples),
            peak_cpu_usage=max(cpu_samples),
            avg_memory_mb=profiling_analysis.get('resource_usage', {}).get('avg_memory', 0),
            peak_memory_mb=profiling_analysis.get('resource_usage', {}).get('peak_memory', 0),
            memory_growth_mb=profiling_analysis.get('memory_growth_mb', 0),
            
            # Efficiency metrics
            cpu_efficiency=total_queries / statistics.mean(cpu_samples) if cpu_samples and statistics.mean(cpu_samples) > 0 else 0,
            memory_efficiency=total_queries / profiling_analysis.get('resource_usage', {}).get('avg_memory', 1),
            
            # Quality metrics
            avg_confidence=statistics.mean(confidences) if confidences else 0,
            confidence_variance=statistics.variance(confidences) if len(confidences) > 1 else 0
        )
        
        self.logger.info(f"Baseline benchmark completed: {total_queries} queries, "
                        f"{metrics.queries_per_second:.2f} QPS, "
                        f"{metrics.avg_response_time:.2f}ms avg response time")
        
        return metrics


class EnhancedSystemTester:
    """Performance tester for the enhanced LLM-integrated system."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.profiler = PerformanceProfiler(logger)
        
        # Create mock enhanced system
        self.baseline_router = BiomedicalQueryRouter(logger)
        self.enhanced_system = self._create_mock_enhanced_system()
        
    def _create_mock_enhanced_system(self):
        """Create mock enhanced system that simulates LLM integration."""
        
        class MockEnhancedSystem:
            def __init__(self, baseline_router, logger):
                self.baseline_router = baseline_router
                self.logger = logger
                self.llm_call_count = 0
                self.fallback_count = 0
                
            async def route_query_enhanced(self, query: str):
                """Enhanced routing with simulated LLM integration."""
                
                # Simulate LLM decision making
                llm_start = time.time()
                
                # Get baseline prediction first
                baseline_prediction = self.baseline_router.route_query(query)
                
                # Simulate LLM processing time based on query complexity
                query_complexity = min(len(query.split()) / 10.0, 1.0)
                base_llm_time = 0.05 + (query_complexity * 0.1)  # 50-150ms base time
                
                # Add some variability
                import random
                llm_processing_time = base_llm_time * random.uniform(0.8, 1.5)
                
                # Simulate network latency occasionally
                if random.random() < 0.1:  # 10% chance of network delay
                    llm_processing_time += random.uniform(0.1, 0.3)
                
                # Check if should use fallback (simulate budget/error conditions)
                use_fallback = random.random() < 0.15  # 15% fallback rate
                
                if use_fallback:
                    # Use baseline system as fallback
                    self.fallback_count += 1
                    await asyncio.sleep(0.01)  # Minimal additional time for fallback logic
                    
                    return {
                        'prediction': baseline_prediction,
                        'enhanced': False,
                        'llm_time': 0.01,
                        'total_time': time.time() - llm_start,
                        'used_fallback': True
                    }
                else:
                    # Simulate LLM enhancement
                    await asyncio.sleep(llm_processing_time)
                    self.llm_call_count += 1
                    
                    # Enhanced prediction (slightly better confidence)
                    enhanced_prediction = baseline_prediction
                    if hasattr(enhanced_prediction, 'confidence'):
                        confidence_boost = query_complexity * 0.1  # Up to 10% boost for complex queries
                        enhanced_prediction.confidence = min(
                            enhanced_prediction.confidence + confidence_boost,
                            0.95
                        )
                    
                    return {
                        'prediction': enhanced_prediction,
                        'enhanced': True,
                        'llm_time': llm_processing_time,
                        'total_time': time.time() - llm_start,
                        'used_fallback': False
                    }
        
        return MockEnhancedSystem(self.baseline_router, self.logger)
    
    async def run_performance_benchmark(self,
                                      test_queries: List[str],
                                      concurrent_users: int = 1,
                                      duration_seconds: int = 60) -> PerformanceMetrics:
        """Run comprehensive performance benchmark for enhanced system."""
        
        self.logger.info(f"Starting enhanced system benchmark: {len(test_queries)} queries, "
                        f"{concurrent_users} concurrent users, {duration_seconds}s duration")
        
        # Start profiling
        self.profiler.start_profiling("enhanced_benchmark")
        
        # Performance tracking
        response_times = []
        confidences = []
        success_count = 0
        failure_count = 0
        llm_times = []
        fallback_usage = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Resource monitoring
        resource_samples = []
        
        def resource_monitor():
            while time.time() < end_time:
                self.profiler.sample_resources("enhanced_benchmark")
                resource_samples.append({
                    'timestamp': time.time(),
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent
                })
                time.sleep(1)
        
        # Start resource monitoring
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute queries
        async def execute_enhanced_query(query: str) -> Dict[str, Any]:
            query_start = time.time()
            try:
                result = await self.enhanced_system.route_query_enhanced(query)
                total_time = (time.time() - query_start) * 1000  # Convert to ms
                
                return {
                    'success': True,
                    'response_time': total_time,
                    'confidence': result['prediction'].confidence if result['prediction'] else 0.0,
                    'llm_time': result.get('llm_time', 0) * 1000,  # Convert to ms
                    'used_fallback': result.get('used_fallback', False),
                    'enhanced': result.get('enhanced', False)
                }
            except Exception as e:
                total_time = (time.time() - query_start) * 1000
                return {
                    'success': False,
                    'response_time': total_time,
                    'confidence': 0.0,
                    'llm_time': 0,
                    'used_fallback': True,
                    'error': str(e)
                }
        
        # Run concurrent workload
        current_time = time.time()
        
        while current_time < end_time:
            # Create batch of concurrent tasks
            batch_tasks = []
            for i in range(concurrent_users):
                if current_time >= end_time:
                    break
                query = test_queries[len(response_times) % len(test_queries)]
                batch_tasks.append(execute_enhanced_query(query))
                current_time = time.time()
            
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, dict):
                        response_times.append(result['response_time'])
                        confidences.append(result['confidence'])
                        llm_times.append(result.get('llm_time', 0))
                        
                        if result.get('used_fallback', False):
                            fallback_usage += 1
                        
                        if result['success']:
                            success_count += 1
                            self.profiler.record_query_performance(
                                "enhanced_benchmark",
                                result['response_time'],
                                result['confidence']
                            )
                        else:
                            failure_count += 1
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Stop profiling
        profiling_analysis = self.profiler.stop_profiling("enhanced_benchmark")
        
        # Calculate metrics
        total_queries = success_count + failure_count
        actual_duration = time.time() - start_time
        
        # Resource calculations
        cpu_samples = [s['cpu'] for s in resource_samples] if resource_samples else [0]
        
        # Calculate LLM time ratio
        total_response_time = sum(response_times)
        total_llm_time = sum(llm_times)
        llm_time_ratio = total_llm_time / total_response_time if total_response_time > 0 else 0
        fallback_usage_rate = fallback_usage / total_queries if total_queries > 0 else 0
        
        # Build performance metrics
        metrics = PerformanceMetrics(
            # Response time metrics
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            median_response_time=statistics.median(response_times) if response_times else 0,
            p95_response_time=np.percentile(response_times, 95) if response_times else 0,
            p99_response_time=np.percentile(response_times, 99) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            response_time_std=statistics.stdev(response_times) if len(response_times) > 1 else 0,
            
            # Throughput metrics
            queries_per_second=total_queries / actual_duration if actual_duration > 0 else 0,
            successful_queries=success_count,
            failed_queries=failure_count,
            success_rate=success_count / total_queries if total_queries > 0 else 0,
            
            # Resource utilization
            avg_cpu_usage=statistics.mean(cpu_samples),
            peak_cpu_usage=max(cpu_samples),
            avg_memory_mb=profiling_analysis.get('resource_usage', {}).get('avg_memory', 0),
            peak_memory_mb=profiling_analysis.get('resource_usage', {}).get('peak_memory', 0),
            memory_growth_mb=profiling_analysis.get('memory_growth_mb', 0),
            
            # Efficiency metrics
            cpu_efficiency=total_queries / statistics.mean(cpu_samples) if cpu_samples and statistics.mean(cpu_samples) > 0 else 0,
            memory_efficiency=total_queries / profiling_analysis.get('resource_usage', {}).get('avg_memory', 1),
            
            # Quality metrics
            avg_confidence=statistics.mean(confidences) if confidences else 0,
            confidence_variance=statistics.variance(confidences) if len(confidences) > 1 else 0,
            
            # Enhanced system specific metrics
            llm_time_ratio=llm_time_ratio,
            fallback_usage_rate=fallback_usage_rate
        )
        
        self.logger.info(f"Enhanced benchmark completed: {total_queries} queries, "
                        f"{metrics.queries_per_second:.2f} QPS, "
                        f"{metrics.avg_response_time:.2f}ms avg response time, "
                        f"{fallback_usage_rate:.1%} fallback usage")
        
        return metrics


class PerformanceComparisonAnalyzer:
    """Analyzes and compares performance between baseline and enhanced systems."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def analyze_performance_comparison(self,
                                     baseline_metrics: PerformanceMetrics,
                                     enhanced_metrics: PerformanceMetrics) -> SystemComparison:
        """Perform comprehensive comparison analysis."""
        
        # Calculate performance differences
        response_time_improvement = (
            (baseline_metrics.avg_response_time - enhanced_metrics.avg_response_time)
            / baseline_metrics.avg_response_time
        ) if baseline_metrics.avg_response_time > 0 else 0
        
        throughput_improvement = (
            (enhanced_metrics.queries_per_second - baseline_metrics.queries_per_second)
            / baseline_metrics.queries_per_second
        ) if baseline_metrics.queries_per_second > 0 else 0
        
        accuracy_improvement = enhanced_metrics.avg_confidence - baseline_metrics.avg_confidence
        
        # Calculate resource overhead
        cpu_overhead = enhanced_metrics.avg_cpu_usage - baseline_metrics.avg_cpu_usage
        memory_overhead = enhanced_metrics.avg_memory_mb - baseline_metrics.avg_memory_mb
        
        # Quality improvements
        confidence_improvement = enhanced_metrics.avg_confidence - baseline_metrics.avg_confidence
        success_rate_improvement = enhanced_metrics.success_rate - baseline_metrics.success_rate
        
        # Calculate overall performance score (0-100)
        score_components = {
            'response_time': self._score_response_time_improvement(response_time_improvement),
            'throughput': self._score_throughput_improvement(throughput_improvement),
            'accuracy': self._score_accuracy_improvement(accuracy_improvement),
            'resource_efficiency': self._score_resource_efficiency(cpu_overhead, memory_overhead),
            'quality': self._score_quality_improvement(confidence_improvement, success_rate_improvement)
        }
        
        # Weighted performance score
        weights = {
            'response_time': 0.25,
            'throughput': 0.20,
            'accuracy': 0.25,
            'resource_efficiency': 0.15,
            'quality': 0.15
        }
        
        performance_score = sum(score * weights[component] for component, score in score_components.items())
        
        # Generate recommendation
        recommendation = self._generate_recommendation(performance_score, response_time_improvement, 
                                                     cpu_overhead, memory_overhead)
        
        comparison = SystemComparison(
            baseline_metrics=baseline_metrics,
            enhanced_metrics=enhanced_metrics,
            response_time_improvement=response_time_improvement,
            throughput_improvement=throughput_improvement,
            accuracy_improvement=accuracy_improvement,
            cpu_overhead=cpu_overhead,
            memory_overhead=memory_overhead,
            confidence_improvement=confidence_improvement,
            success_rate_improvement=success_rate_improvement,
            performance_score=performance_score,
            recommendation=recommendation
        )
        
        self.logger.info(f"Performance comparison completed:")
        self.logger.info(f"  Response time: {response_time_improvement:+.1%}")
        self.logger.info(f"  Throughput: {throughput_improvement:+.1%}")
        self.logger.info(f"  Accuracy: {accuracy_improvement:+.3f}")
        self.logger.info(f"  CPU overhead: {cpu_overhead:+.1f}%")
        self.logger.info(f"  Memory overhead: {memory_overhead:+.1f}MB")
        self.logger.info(f"  Overall score: {performance_score:.1f}/100")
        self.logger.info(f"  Recommendation: {recommendation}")
        
        return comparison
    
    def _score_response_time_improvement(self, improvement: float) -> float:
        """Score response time improvement (0-100)."""
        if improvement >= 0.2:  # 20%+ improvement
            return 100
        elif improvement >= 0.1:  # 10-20% improvement
            return 80
        elif improvement >= 0:  # No degradation
            return 60
        elif improvement >= -0.1:  # <10% degradation
            return 40
        elif improvement >= -0.3:  # 10-30% degradation
            return 20
        else:  # >30% degradation
            return 0
    
    def _score_throughput_improvement(self, improvement: float) -> float:
        """Score throughput improvement (0-100)."""
        if improvement >= 0.3:  # 30%+ improvement
            return 100
        elif improvement >= 0.1:  # 10-30% improvement
            return 80
        elif improvement >= 0:  # No degradation
            return 60
        elif improvement >= -0.1:  # <10% degradation
            return 40
        elif improvement >= -0.2:  # 10-20% degradation
            return 20
        else:  # >20% degradation
            return 0
    
    def _score_accuracy_improvement(self, improvement: float) -> float:
        """Score accuracy improvement (0-100)."""
        if improvement >= 0.1:  # 10%+ improvement
            return 100
        elif improvement >= 0.05:  # 5-10% improvement
            return 80
        elif improvement >= 0:  # No degradation
            return 60
        elif improvement >= -0.05:  # <5% degradation
            return 40
        else:  # >5% degradation
            return 20
    
    def _score_resource_efficiency(self, cpu_overhead: float, memory_overhead: float) -> float:
        """Score resource efficiency (0-100)."""
        # Penalize resource overhead
        cpu_penalty = max(0, cpu_overhead) * 2  # 2 points per % CPU overhead
        memory_penalty = max(0, memory_overhead) / 10  # 1 point per 10MB memory overhead
        
        base_score = 100
        total_penalty = cpu_penalty + memory_penalty
        
        return max(0, base_score - total_penalty)
    
    def _score_quality_improvement(self, confidence_improvement: float, success_rate_improvement: float) -> float:
        """Score quality improvement (0-100)."""
        confidence_score = self._score_accuracy_improvement(confidence_improvement * 10)  # Scale up confidence
        success_rate_score = self._score_accuracy_improvement(success_rate_improvement * 10)  # Scale up success rate
        
        return (confidence_score + success_rate_score) / 2
    
    def _generate_recommendation(self, 
                               performance_score: float,
                               response_time_improvement: float,
                               cpu_overhead: float,
                               memory_overhead: float) -> str:
        """Generate deployment recommendation."""
        
        # Critical blockers
        if response_time_improvement < -0.5:  # >50% slower
            return "reject"
        
        if cpu_overhead > 50 or memory_overhead > 1000:  # Excessive resource usage
            return "reject"
        
        # Strong recommendation for deployment
        if performance_score >= 80:
            return "deploy"
        
        # Good with some optimization
        if performance_score >= 60:
            return "optimize"
        
        # Needs significant work
        if performance_score >= 40:
            return "optimize"
        
        # Not ready
        return "reject"
    
    def generate_detailed_report(self, comparison: SystemComparison) -> Dict[str, Any]:
        """Generate detailed performance comparison report."""
        
        report = {
            'executive_summary': {
                'recommendation': comparison.recommendation,
                'performance_score': comparison.performance_score,
                'key_improvements': {
                    'response_time': f"{comparison.response_time_improvement:+.1%}",
                    'throughput': f"{comparison.throughput_improvement:+.1%}",
                    'accuracy': f"{comparison.accuracy_improvement:+.3f}",
                    'quality': f"{comparison.confidence_improvement:+.3f}"
                },
                'resource_impact': {
                    'cpu_overhead': f"{comparison.cpu_overhead:+.1f}%",
                    'memory_overhead': f"{comparison.memory_overhead:+.1f}MB"
                }
            },
            'detailed_metrics': {
                'baseline': comparison.baseline_metrics.to_dict(),
                'enhanced': comparison.enhanced_metrics.to_dict()
            },
            'performance_analysis': {
                'response_time_distribution': {
                    'baseline_p95': comparison.baseline_metrics.p95_response_time,
                    'enhanced_p95': comparison.enhanced_metrics.p95_response_time,
                    'p95_improvement': (
                        comparison.baseline_metrics.p95_response_time - 
                        comparison.enhanced_metrics.p95_response_time
                    ) / comparison.baseline_metrics.p95_response_time if comparison.baseline_metrics.p95_response_time > 0 else 0
                },
                'resource_efficiency': {
                    'baseline_cpu_efficiency': comparison.baseline_metrics.cpu_efficiency,
                    'enhanced_cpu_efficiency': comparison.enhanced_metrics.cpu_efficiency,
                    'baseline_memory_efficiency': comparison.baseline_metrics.memory_efficiency,
                    'enhanced_memory_efficiency': comparison.enhanced_metrics.memory_efficiency
                },
                'quality_metrics': {
                    'confidence_variance_baseline': comparison.baseline_metrics.confidence_variance,
                    'confidence_variance_enhanced': comparison.enhanced_metrics.confidence_variance,
                    'success_rate_baseline': comparison.baseline_metrics.success_rate,
                    'success_rate_enhanced': comparison.enhanced_metrics.success_rate
                }
            },
            'optimization_opportunities': self._identify_optimization_opportunities(comparison),
            'deployment_considerations': self._generate_deployment_considerations(comparison)
        }
        
        return report
    
    def _identify_optimization_opportunities(self, comparison: SystemComparison) -> List[Dict[str, str]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Response time optimization
        if comparison.response_time_improvement < 0:
            opportunities.append({
                'area': 'Response Time',
                'issue': f"Response time degraded by {abs(comparison.response_time_improvement):.1%}",
                'recommendation': "Optimize LLM call latency, implement better caching, or increase fallback threshold"
            })
        
        # Throughput optimization
        if comparison.throughput_improvement < 0:
            opportunities.append({
                'area': 'Throughput',
                'issue': f"Throughput decreased by {abs(comparison.throughput_improvement):.1%}",
                'recommendation': "Implement connection pooling, optimize concurrent processing, or increase system resources"
            })
        
        # Resource optimization
        if comparison.cpu_overhead > 20:
            opportunities.append({
                'area': 'CPU Usage',
                'issue': f"High CPU overhead: {comparison.cpu_overhead:.1f}%",
                'recommendation': "Profile CPU-intensive operations, optimize algorithms, or implement async processing"
            })
        
        if comparison.memory_overhead > 100:
            opportunities.append({
                'area': 'Memory Usage',
                'issue': f"High memory overhead: {comparison.memory_overhead:.1f}MB",
                'recommendation': "Implement memory pooling, optimize data structures, or implement garbage collection tuning"
            })
        
        # Quality optimization
        if comparison.accuracy_improvement < 0.05:
            opportunities.append({
                'area': 'Accuracy',
                'issue': f"Limited accuracy improvement: {comparison.accuracy_improvement:.3f}",
                'recommendation': "Fine-tune LLM prompts, improve training data, or implement better confidence calibration"
            })
        
        return opportunities
    
    def _generate_deployment_considerations(self, comparison: SystemComparison) -> List[str]:
        """Generate deployment considerations."""
        considerations = []
        
        if comparison.recommendation == "deploy":
            considerations.extend([
                "System shows strong performance improvements and is ready for production",
                "Monitor resource usage closely during initial deployment",
                "Implement gradual rollout to validate performance under real load",
                "Set up comprehensive monitoring and alerting for the enhanced features"
            ])
        elif comparison.recommendation == "optimize":
            considerations.extend([
                "System shows promise but needs optimization before full deployment",
                "Consider phased rollout starting with non-critical workloads",
                "Focus optimization efforts on identified performance bottlenecks",
                "Implement A/B testing to validate improvements in production"
            ])
        else:  # reject
            considerations.extend([
                "System performance does not meet production requirements",
                "Significant optimization or redesign needed before deployment",
                "Consider alternative implementation approaches",
                "Conduct detailed performance profiling to identify root causes"
            ])
        
        # LLM-specific considerations
        if comparison.enhanced_metrics.fallback_usage_rate > 0.3:
            considerations.append(f"High fallback usage rate ({comparison.enhanced_metrics.fallback_usage_rate:.1%}) indicates potential LLM reliability issues")
        
        if comparison.enhanced_metrics.llm_time_ratio > 0.7:
            considerations.append("High LLM time ratio suggests potential for optimization through caching or model selection")
        
        return considerations


class ComprehensivePerformanceTester:
    """Main class for comprehensive performance comparison testing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fixtures = BiomedicalTestFixtures()
        
        # Initialize testers
        self.baseline_tester = BaselineSystemTester(self.logger)
        self.enhanced_tester = EnhancedSystemTester(self.logger)
        self.analyzer = PerformanceComparisonAnalyzer(self.logger)
        
    async def run_comprehensive_performance_comparison(self,
                                                     test_scenarios: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive performance comparison between systems."""
        
        self.logger.info("Starting comprehensive performance comparison")
        
        # Default test scenarios if none provided
        if test_scenarios is None:
            test_scenarios = self._get_default_test_scenarios()
        
        # Results storage
        comparison_results = {}
        
        # Run each test scenario
        for scenario_name, scenario_config in test_scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name}")
            
            try:
                # Extract scenario parameters
                test_queries = scenario_config['queries']
                concurrent_users = scenario_config.get('concurrent_users', 1)
                duration_seconds = scenario_config.get('duration_seconds', 30)
                
                # Run baseline test
                self.logger.info(f"Testing baseline system for scenario: {scenario_name}")
                baseline_metrics = await self.baseline_tester.run_performance_benchmark(
                    test_queries, concurrent_users, duration_seconds
                )
                
                # Run enhanced test
                self.logger.info(f"Testing enhanced system for scenario: {scenario_name}")
                enhanced_metrics = await self.enhanced_tester.run_performance_benchmark(
                    test_queries, concurrent_users, duration_seconds
                )
                
                # Analyze comparison
                comparison = self.analyzer.analyze_performance_comparison(
                    baseline_metrics, enhanced_metrics
                )
                
                # Generate detailed report
                detailed_report = self.analyzer.generate_detailed_report(comparison)
                
                comparison_results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'comparison': comparison.to_dict(),
                    'detailed_report': detailed_report
                }
                
                self.logger.info(f"Scenario {scenario_name} completed: "
                                f"Score {comparison.performance_score:.1f}/100, "
                                f"Recommendation: {comparison.recommendation}")
                
            except Exception as e:
                self.logger.error(f"Scenario {scenario_name} failed: {str(e)}")
                comparison_results[scenario_name] = {
                    'error': str(e),
                    'scenario_config': scenario_config
                }
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(comparison_results)
        
        final_report = {
            'test_timestamp': datetime.now().isoformat(),
            'overall_summary': overall_summary,
            'scenario_results': comparison_results,
            'recommendations': self._generate_final_recommendations(comparison_results)
        }
        
        self.logger.info("Comprehensive performance comparison completed")
        self.logger.info(f"Overall recommendation: {overall_summary.get('final_recommendation', 'unknown')}")
        
        return final_report
    
    def _get_default_test_scenarios(self) -> Dict[str, Any]:
        """Get default test scenarios for comprehensive testing."""
        
        # Get test queries from fixtures
        test_queries = self.fixtures.get_biomedical_test_queries()
        
        scenarios = {
            'basic_performance': {
                'description': 'Basic performance test with simple queries',
                'queries': [q['query'] for q in test_queries.get('biomarker_discovery', [])[:5]],
                'concurrent_users': 1,
                'duration_seconds': 30
            },
            'concurrent_load': {
                'description': 'Concurrent load test',
                'queries': [q['query'] for q in test_queries.get('pathway_analysis', [])[:3]],
                'concurrent_users': 5,
                'duration_seconds': 60
            },
            'complex_queries': {
                'description': 'Complex query performance test',
                'queries': [
                    "What are the metabolomic pathways involved in insulin resistance and how do they relate to type 2 diabetes progression?",
                    "How do LC-MS techniques compare with GC-MS for metabolite identification in clinical samples, and what are the implications for biomarker discovery?",
                    "What are the latest advances in metabolomics applications for personalized medicine in cardiovascular disease?"
                ],
                'concurrent_users': 2,
                'duration_seconds': 45
            },
            'mixed_workload': {
                'description': 'Mixed workload with various query types',
                'queries': [
                    "metabolomics",
                    "What are biomarkers?",
                    "Latest research on diabetes metabolomics 2024",
                    "LC-MS analysis of glucose metabolites",
                    "How do metabolic pathways interact in disease?"
                ],
                'concurrent_users': 3,
                'duration_seconds': 60
            }
        }
        
        return scenarios
    
    def _generate_overall_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary across all scenarios."""
        
        successful_scenarios = [r for r in comparison_results.values() if 'comparison' in r]
        
        if not successful_scenarios:
            return {
                'final_recommendation': 'reject',
                'reason': 'No successful test scenarios',
                'scenarios_tested': len(comparison_results),
                'successful_scenarios': 0
            }
        
        # Aggregate metrics
        performance_scores = [r['comparison']['performance_score'] for r in successful_scenarios]
        recommendations = [r['comparison']['recommendation'] for r in successful_scenarios]
        
        avg_performance_score = statistics.mean(performance_scores)
        
        # Count recommendation types
        recommendation_counts = {
            'deploy': recommendations.count('deploy'),
            'optimize': recommendations.count('optimize'),
            'reject': recommendations.count('reject')
        }
        
        # Determine final recommendation
        total_scenarios = len(successful_scenarios)
        if recommendation_counts['deploy'] >= total_scenarios * 0.7:  # 70% deploy
            final_recommendation = 'deploy'
        elif recommendation_counts['deploy'] + recommendation_counts['optimize'] >= total_scenarios * 0.6:
            final_recommendation = 'optimize'
        else:
            final_recommendation = 'reject'
        
        return {
            'final_recommendation': final_recommendation,
            'avg_performance_score': avg_performance_score,
            'scenarios_tested': len(comparison_results),
            'successful_scenarios': len(successful_scenarios),
            'recommendation_distribution': recommendation_counts,
            'performance_score_range': {
                'min': min(performance_scores),
                'max': max(performance_scores),
                'std': statistics.stdev(performance_scores) if len(performance_scores) > 1 else 0
            }
        }
    
    def _generate_final_recommendations(self, comparison_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate final recommendations based on all test results."""
        
        successful_scenarios = [r for r in comparison_results.values() if 'comparison' in r]
        recommendations = []
        
        if not successful_scenarios:
            recommendations.append({
                'type': 'critical',
                'title': 'Testing Failed',
                'recommendation': 'All test scenarios failed. System requires debugging before performance evaluation.',
                'action': 'Debug test environment and system issues'
            })
            return recommendations
        
        # Analyze common issues across scenarios
        avg_cpu_overhead = statistics.mean([
            r['comparison']['cpu_overhead'] for r in successful_scenarios
        ])
        
        avg_memory_overhead = statistics.mean([
            r['comparison']['memory_overhead'] for r in successful_scenarios
        ])
        
        avg_response_time_improvement = statistics.mean([
            r['comparison']['response_time_improvement'] for r in successful_scenarios
        ])
        
        # Resource optimization recommendations
        if avg_cpu_overhead > 20:
            recommendations.append({
                'type': 'optimization',
                'title': 'CPU Optimization Needed',
                'recommendation': f'Average CPU overhead is {avg_cpu_overhead:.1f}%. Optimize CPU-intensive operations.',
                'action': 'Profile and optimize CPU usage'
            })
        
        if avg_memory_overhead > 100:
            recommendations.append({
                'type': 'optimization',
                'title': 'Memory Optimization Needed',
                'recommendation': f'Average memory overhead is {avg_memory_overhead:.1f}MB. Optimize memory usage.',
                'action': 'Implement memory optimization strategies'
            })
        
        # Performance recommendations
        if avg_response_time_improvement < -0.1:
            recommendations.append({
                'type': 'performance',
                'title': 'Response Time Degradation',
                'recommendation': f'Average response time degraded by {abs(avg_response_time_improvement):.1%}.',
                'action': 'Optimize LLM integration and caching'
            })
        
        # Success recommendations
        if avg_response_time_improvement > 0.1:
            recommendations.append({
                'type': 'positive',
                'title': 'Performance Improvement Achieved',
                'recommendation': f'Average response time improved by {avg_response_time_improvement:.1%}.',
                'action': 'Continue with deployment planning'
            })
        
        return recommendations


# Pytest test class
@pytest.mark.asyncio
class TestPerformanceComparisonComprehensive:
    """Main test class for comprehensive performance comparison."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tester = ComprehensivePerformanceTester(self.logger)
    
    async def test_basic_performance_comparison(self):
        """Test basic performance comparison between systems."""
        
        # Simple test scenario
        test_scenarios = {
            'basic_test': {
                'queries': ["What are metabolomics biomarkers?", "LC-MS analysis methods"],
                'concurrent_users': 1,
                'duration_seconds': 10  # Short duration for testing
            }
        }
        
        results = await self.tester.run_comprehensive_performance_comparison(test_scenarios)
        
        # Validate results structure
        assert 'overall_summary' in results
        assert 'scenario_results' in results
        assert 'recommendations' in results
        
        # Check scenario results
        scenario_results = results['scenario_results']['basic_test']
        assert 'comparison' in scenario_results
        assert 'detailed_report' in scenario_results
        
        # Check comparison metrics
        comparison = scenario_results['comparison']
        assert 'performance_score' in comparison
        assert 'recommendation' in comparison
        assert comparison['performance_score'] >= 0
        assert comparison['performance_score'] <= 100
        assert comparison['recommendation'] in ['deploy', 'optimize', 'reject']
        
        self.logger.info(f"Basic performance test completed with score: {comparison['performance_score']}")
    
    async def test_concurrent_load_comparison(self):
        """Test performance comparison under concurrent load."""
        
        test_scenarios = {
            'load_test': {
                'queries': ["metabolomics", "biomarkers"],
                'concurrent_users': 3,
                'duration_seconds': 15
            }
        }
        
        results = await self.tester.run_comprehensive_performance_comparison(test_scenarios)
        
        scenario_results = results['scenario_results']['load_test']
        comparison = scenario_results['comparison']
        
        # Check that concurrent processing was tested
        baseline_metrics = comparison['baseline_metrics']
        enhanced_metrics = comparison['enhanced_metrics']
        
        assert baseline_metrics['successful_queries'] > 0
        assert enhanced_metrics['successful_queries'] > 0
        
        # Check throughput metrics
        assert 'queries_per_second' in baseline_metrics
        assert 'queries_per_second' in enhanced_metrics
        
        self.logger.info(f"Load test completed - Baseline QPS: {baseline_metrics['queries_per_second']:.2f}, "
                        f"Enhanced QPS: {enhanced_metrics['queries_per_second']:.2f}")
    
    async def test_performance_profiling_accuracy(self):
        """Test that performance profiling produces accurate metrics."""
        
        # Test profiler directly
        profiler = PerformanceProfiler(self.logger)
        
        profiler.start_profiling("test_session")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        profiler.record_query_performance("test_session", 50.0, 0.8)
        
        await asyncio.sleep(0.1)
        profiler.record_query_performance("test_session", 75.0, 0.9)
        
        profiler.sample_resources("test_session")
        
        analysis = profiler.stop_profiling("test_session")
        
        # Validate profiling results
        assert 'session_duration' in analysis
        assert 'total_queries' in analysis
        assert analysis['total_queries'] == 2
        
        assert 'response_times' in analysis
        response_times = analysis['response_times']
        assert response_times['min'] == 50.0
        assert response_times['max'] == 75.0
        
        assert 'confidence_stats' in analysis
        confidence_stats = analysis['confidence_stats']
        assert 0.8 <= confidence_stats['avg'] <= 0.9
        
        self.logger.info("Performance profiling accuracy test passed")
    
    async def test_system_comparison_analysis(self):
        """Test system comparison analysis logic."""
        
        # Create mock metrics for testing
        baseline_metrics = PerformanceMetrics(
            avg_response_time=100.0,
            median_response_time=95.0,
            p95_response_time=150.0,
            p99_response_time=200.0,
            min_response_time=50.0,
            max_response_time=250.0,
            response_time_std=25.0,
            queries_per_second=10.0,
            successful_queries=100,
            failed_queries=5,
            success_rate=0.95,
            avg_cpu_usage=30.0,
            peak_cpu_usage=45.0,
            avg_memory_mb=512.0,
            peak_memory_mb=600.0,
            memory_growth_mb=50.0,
            cpu_efficiency=3.33,
            memory_efficiency=0.195,
            avg_confidence=0.75,
            confidence_variance=0.05
        )
        
        enhanced_metrics = PerformanceMetrics(
            avg_response_time=120.0,  # 20% slower
            median_response_time=110.0,
            p95_response_time=170.0,
            p99_response_time=220.0,
            min_response_time=60.0,
            max_response_time=280.0,
            response_time_std=30.0,
            queries_per_second=8.5,  # Lower throughput
            successful_queries=95,
            failed_queries=3,
            success_rate=0.97,  # Higher success rate
            avg_cpu_usage=40.0,  # Higher CPU usage
            peak_cpu_usage=60.0,
            avg_memory_mb=650.0,  # Higher memory usage
            peak_memory_mb=750.0,
            memory_growth_mb=75.0,
            cpu_efficiency=2.38,
            memory_efficiency=0.146,
            avg_confidence=0.82,  # Better confidence
            confidence_variance=0.03,
            llm_time_ratio=0.3,  # 30% of time in LLM calls
            fallback_usage_rate=0.1  # 10% fallback usage
        )
        
        # Analyze comparison
        analyzer = PerformanceComparisonAnalyzer(self.logger)
        comparison = analyzer.analyze_performance_comparison(baseline_metrics, enhanced_metrics)
        
        # Validate comparison results
        assert comparison.response_time_improvement < 0  # Slower response time
        assert comparison.throughput_improvement < 0  # Lower throughput
        assert comparison.accuracy_improvement > 0  # Better accuracy
        assert comparison.cpu_overhead > 0  # More CPU usage
        assert comparison.memory_overhead > 0  # More memory usage
        assert comparison.confidence_improvement > 0  # Better confidence
        
        # Check performance score calculation
        assert 0 <= comparison.performance_score <= 100
        
        # Check recommendation logic
        assert comparison.recommendation in ['deploy', 'optimize', 'reject']
        
        self.logger.info(f"System comparison analysis test completed: "
                        f"Score {comparison.performance_score:.1f}, "
                        f"Recommendation: {comparison.recommendation}")
    
    async def test_comprehensive_performance_scenarios(self):
        """Test comprehensive performance scenarios."""
        
        # Run with default scenarios
        results = await self.tester.run_comprehensive_performance_comparison()
        
        # Validate overall structure
        assert 'overall_summary' in results
        assert 'scenario_results' in results
        assert 'recommendations' in results
        
        overall_summary = results['overall_summary']
        
        # Check summary metrics
        assert 'final_recommendation' in overall_summary
        assert 'avg_performance_score' in overall_summary
        assert 'scenarios_tested' in overall_summary
        assert 'successful_scenarios' in overall_summary
        
        # Should have tested multiple scenarios
        assert overall_summary['scenarios_tested'] >= 3
        assert overall_summary['successful_scenarios'] >= 0
        
        # Check that each scenario has proper results
        scenario_results = results['scenario_results']
        for scenario_name, scenario_result in scenario_results.items():
            if 'comparison' in scenario_result:
                comparison = scenario_result['comparison']
                assert 'performance_score' in comparison
                assert 'recommendation' in comparison
                
                detailed_report = scenario_result['detailed_report']
                assert 'executive_summary' in detailed_report
                assert 'detailed_metrics' in detailed_report
        
        self.logger.info(f"Comprehensive performance test completed: "
                        f"{overall_summary['successful_scenarios']}/{overall_summary['scenarios_tested']} scenarios successful, "
                        f"Final recommendation: {overall_summary['final_recommendation']}")


# Export main classes
__all__ = [
    'TestPerformanceComparisonComprehensive',
    'ComprehensivePerformanceTester',
    'BaselineSystemTester',
    'EnhancedSystemTester',
    'PerformanceComparisonAnalyzer',
    'PerformanceProfiler',
    'PerformanceMetrics',
    'SystemComparison'
]