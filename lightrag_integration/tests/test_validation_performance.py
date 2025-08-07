#!/usr/bin/env python3
"""
Comprehensive Performance and Scalability Tests for Factual Accuracy Validation System.

This test suite provides thorough performance testing for the factual accuracy validation
pipeline including benchmarking, load testing, scalability analysis, and resource monitoring.

Test Categories:
1. Component-level performance tests
2. System-level performance benchmarks
3. Scalability and load testing
4. Memory usage and resource monitoring
5. Concurrent processing performance
6. Performance regression testing
7. Optimization validation tests

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import statistics
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import gc
import sys
import json

# Import test fixtures
from .factual_validation_test_fixtures import *

# Import the modules to test
try:
    from ..accuracy_scorer import (
        FactualAccuracyScorer, AccuracyScore, AccuracyReport
    )
    from ..factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationResult, VerificationStatus
    )
    from ..claim_extractor import (
        BiomedicalClaimExtractor, ExtractedClaim
    )
    from ..document_indexer import (
        SourceDocumentIndex
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_ops_per_second: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]


class ResourceMonitor:
    """Monitor system resources during test execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []
    
    def start_monitoring(self, interval: float = 0.1):
        """Start resource monitoring."""
        self.monitoring = True
        self.measurements = []
        
        def monitor():
            while self.monitoring:
                try:
                    memory_info = self.process.memory_info()
                    cpu_percent = self.process.cpu_percent()
                    
                    measurement = {
                        'timestamp': time.time(),
                        'memory_rss_mb': memory_info.rss / 1024 / 1024,
                        'memory_vms_mb': memory_info.vms / 1024 / 1024,
                        'cpu_percent': cpu_percent,
                        'num_threads': self.process.num_threads()
                    }
                    
                    self.measurements.append(measurement)
                    time.sleep(interval)
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        
        if not self.measurements:
            return {}
        
        memory_values = [m['memory_rss_mb'] for m in self.measurements]
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        
        return {
            'duration_seconds': self.measurements[-1]['timestamp'] - self.measurements[0]['timestamp'],
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'peak_cpu_percent': max(cpu_values),
            'avg_cpu_percent': statistics.mean(cpu_values),
            'measurement_count': len(self.measurements),
            'measurements': self.measurements[-10:]  # Last 10 measurements
        }


@pytest.mark.performance_validation
class TestComponentPerformance:
    """Test suite for individual component performance."""
    
    @pytest.fixture
    def resource_monitor(self):
        """Provide resource monitor for tests."""
        return ResourceMonitor()
    
    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for testing."""
        return {
            'claim_extraction': {
                'max_time_ms': 500,
                'max_memory_mb': 50,
                'min_throughput': 10  # claims per second
            },
            'claim_verification': {
                'max_time_ms': 1000,
                'max_memory_mb': 100,
                'min_throughput': 5  # claims per second
            },
            'accuracy_scoring': {
                'max_time_ms': 750,
                'max_memory_mb': 75,
                'min_throughput': 8  # reports per second
            },
            'report_generation': {
                'max_time_ms': 1500,
                'max_memory_mb': 150,
                'min_throughput': 3  # reports per second
            }
        }
    
    @pytest.mark.asyncio
    async def test_claim_extractor_performance(self, mock_claim_extractor, performance_thresholds, resource_monitor):
        """Test claim extractor performance with various input sizes."""
        
        # Test data of different sizes
        test_responses = [
            "Short response with glucose 150 mg/dL.",
            " ".join(SAMPLE_BIOMEDICAL_RESPONSES[:3]),  # Medium response
            " ".join(SAMPLE_BIOMEDICAL_RESPONSES * 2),  # Large response
            " ".join(SAMPLE_BIOMEDICAL_RESPONSES * 5)   # Very large response
        ]
        
        results = []
        threshold = performance_thresholds['claim_extraction']
        
        for i, response in enumerate(test_responses):
            # Configure mock to simulate processing time
            processing_delay = 0.05 + (i * 0.02)  # Increasing delay for larger responses
            
            async def delayed_extract(text, context=None):
                await asyncio.sleep(processing_delay)
                return [Mock(claim_id=f"perf_claim_{j}") for j in range(1 + i * 2)]
            
            mock_claim_extractor.extract_claims = delayed_extract
            
            # Measure performance
            resource_monitor.start_monitoring()
            start_time = time.time()
            
            claims = await mock_claim_extractor.extract_claims(response)
            
            execution_time = (time.time() - start_time) * 1000
            resource_stats = resource_monitor.stop_monitoring()
            
            # Calculate throughput
            throughput = len(claims) / (execution_time / 1000) if execution_time > 0 else 0
            
            benchmark = PerformanceBenchmark(
                operation_name=f"claim_extraction_size_{len(response)}",
                execution_time_ms=execution_time,
                memory_usage_mb=resource_stats.get('peak_memory_mb', 0),
                cpu_usage_percent=resource_stats.get('peak_cpu_percent', 0),
                throughput_ops_per_second=throughput,
                success_rate=1.0,
                error_count=0,
                metadata={'response_length': len(response), 'claims_extracted': len(claims)}
            )
            
            results.append(benchmark)
            
            # Check against thresholds for smaller responses
            if i < 2:  # Only check thresholds for small/medium responses
                assert execution_time <= threshold['max_time_ms']
                assert resource_stats.get('peak_memory_mb', 0) <= threshold['max_memory_mb']
        
        # Performance should scale reasonably
        small_time = results[0].execution_time_ms
        large_time = results[-1].execution_time_ms
        assert large_time <= small_time * 10  # Should not be more than 10x slower
    
    @pytest.mark.asyncio
    async def test_validator_performance(self, mock_factual_validator, sample_verification_results, 
                                       performance_thresholds, resource_monitor):
        """Test factual validator performance with various claim loads."""
        
        # Create claim batches of different sizes
        batch_sizes = [1, 5, 10, 25, 50]
        threshold = performance_thresholds['claim_verification']
        
        results = []
        
        for batch_size in batch_sizes:
            # Create test claims
            test_claims = []
            for i in range(batch_size):
                claim = Mock()
                claim.claim_id = f"perf_test_claim_{i}"
                claim.claim_type = "numeric"
                claim.claim_text = f"Test claim {i} with value 150 mg/dL"
                claim.confidence = Mock(overall_confidence=75.0)
                test_claims.append(claim)
            
            # Configure mock validator
            mock_results = sample_verification_results[:batch_size]
            report = Mock()
            report.verification_results = mock_results
            report.total_claims = batch_size
            
            # Simulate processing time based on batch size
            processing_delay = 0.02 * batch_size  # 20ms per claim
            
            async def delayed_verify(claims):
                await asyncio.sleep(processing_delay)
                return report
            
            mock_factual_validator.verify_claims = delayed_verify
            
            # Measure performance
            resource_monitor.start_monitoring()
            start_time = time.time()
            
            verification_report = await mock_factual_validator.verify_claims(test_claims)
            
            execution_time = (time.time() - start_time) * 1000
            resource_stats = resource_monitor.stop_monitoring()
            
            # Calculate throughput
            throughput = batch_size / (execution_time / 1000) if execution_time > 0 else 0
            
            benchmark = PerformanceBenchmark(
                operation_name=f"claim_verification_batch_{batch_size}",
                execution_time_ms=execution_time,
                memory_usage_mb=resource_stats.get('peak_memory_mb', 0),
                cpu_usage_percent=resource_stats.get('peak_cpu_percent', 0),
                throughput_ops_per_second=throughput,
                success_rate=1.0,
                error_count=0,
                metadata={'batch_size': batch_size, 'claims_verified': len(verification_report.verification_results)}
            )
            
            results.append(benchmark)
            
            # Check thresholds for smaller batches
            if batch_size <= 10:
                assert execution_time <= threshold['max_time_ms']
                assert resource_stats.get('peak_memory_mb', 0) <= threshold['max_memory_mb']
                assert throughput >= threshold['min_throughput']
        
        # Throughput should remain reasonable across batch sizes
        throughputs = [r.throughput_ops_per_second for r in results]
        avg_throughput = statistics.mean(throughputs)
        assert avg_throughput >= threshold['min_throughput'] * 0.5  # At least half the minimum
    
    @pytest.mark.asyncio
    async def test_accuracy_scorer_performance(self, sample_verification_results, performance_thresholds, 
                                             resource_monitor):
        """Test accuracy scorer performance with various result sets."""
        
        scorer = FactualAccuracyScorer()
        threshold = performance_thresholds['accuracy_scoring']
        
        # Test with different numbers of verification results
        result_counts = [1, 5, 10, 20, 50]
        benchmarks = []
        
        for count in result_counts:
            # Create result set
            test_results = sample_verification_results[:count] if count <= len(sample_verification_results) else sample_verification_results * (count // len(sample_verification_results) + 1)
            test_results = test_results[:count]
            
            # Measure performance
            resource_monitor.start_monitoring()
            start_time = time.time()
            
            accuracy_score = await scorer.score_accuracy(test_results)
            
            execution_time = (time.time() - start_time) * 1000
            resource_stats = resource_monitor.stop_monitoring()
            
            # Calculate throughput
            throughput = count / (execution_time / 1000) if execution_time > 0 else 0
            
            benchmark = PerformanceBenchmark(
                operation_name=f"accuracy_scoring_{count}_results",
                execution_time_ms=execution_time,
                memory_usage_mb=resource_stats.get('peak_memory_mb', 0),
                cpu_usage_percent=resource_stats.get('peak_cpu_percent', 0),
                throughput_ops_per_second=throughput,
                success_rate=1.0,
                error_count=0,
                metadata={'result_count': count, 'overall_score': accuracy_score.overall_score}
            )
            
            benchmarks.append(benchmark)
            
            # Check thresholds for smaller result sets
            if count <= 20:
                assert execution_time <= threshold['max_time_ms']
                assert resource_stats.get('peak_memory_mb', 0) <= threshold['max_memory_mb']
        
        # Performance should scale sub-linearly
        small_benchmark = benchmarks[0]
        large_benchmark = benchmarks[-1]
        
        # Time should not increase linearly with result count
        time_ratio = large_benchmark.execution_time_ms / small_benchmark.execution_time_ms
        count_ratio = large_benchmark.metadata['result_count'] / small_benchmark.metadata['result_count']
        
        assert time_ratio <= count_ratio  # Should scale better than linearly
    
    @pytest.mark.asyncio
    async def test_report_generation_performance(self, sample_verification_results, sample_extracted_claims,
                                               performance_thresholds, resource_monitor):
        """Test report generation performance."""
        
        scorer = FactualAccuracyScorer()
        threshold = performance_thresholds['report_generation']
        
        # Test comprehensive report generation
        resource_monitor.start_monitoring()
        start_time = time.time()
        
        report = await scorer.generate_comprehensive_report(
            sample_verification_results,
            claims=sample_extracted_claims,
            query="Performance test query",
            response="Performance test response"
        )
        
        execution_time = (time.time() - start_time) * 1000
        resource_stats = resource_monitor.stop_monitoring()
        
        # Calculate throughput (reports per second)
        throughput = 1 / (execution_time / 1000) if execution_time > 0 else 0
        
        benchmark = PerformanceBenchmark(
            operation_name="comprehensive_report_generation",
            execution_time_ms=execution_time,
            memory_usage_mb=resource_stats.get('peak_memory_mb', 0),
            cpu_usage_percent=resource_stats.get('peak_cpu_percent', 0),
            throughput_ops_per_second=throughput,
            success_rate=1.0,
            error_count=0,
            metadata={
                'claims_analyzed': len(sample_verification_results),
                'report_sections': len(report.to_dict()),
                'recommendations_count': len(report.quality_recommendations)
            }
        )
        
        # Check against thresholds
        assert execution_time <= threshold['max_time_ms']
        assert resource_stats.get('peak_memory_mb', 0) <= threshold['max_memory_mb']
        assert throughput >= threshold['min_throughput']
        
        # Report should be comprehensive
        assert isinstance(report, AccuracyReport)
        assert len(report.quality_recommendations) > 0
        assert len(report.claims_analysis) == len(sample_verification_results)


@pytest.mark.performance_validation
class TestSystemPerformance:
    """Test suite for system-level performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_performance(self, integrated_pipeline, resource_monitor):
        """Test end-to-end pipeline performance."""
        
        await integrated_pipeline.initialize()
        
        test_scenarios = [
            {
                "query": "What are glucose levels in diabetes?",
                "response": "Glucose levels in diabetic patients are typically elevated above 126 mg/dL fasting.",
                "expected_max_time_ms": 2000
            },
            {
                "query": "How does LC-MS work in metabolomics?", 
                "response": "LC-MS combines liquid chromatography with mass spectrometry for metabolite identification and quantification.",
                "expected_max_time_ms": 2500
            },
            {
                "query": "What statistical methods are used?",
                "response": "Statistical analysis includes PCA, PLS-DA, t-tests, and ANOVA with FDR correction at p<0.05.",
                "expected_max_time_ms": 3000
            }
        ]
        
        performance_results = []
        
        for scenario in test_scenarios:
            # Measure end-to-end performance
            resource_monitor.start_monitoring()
            start_time = time.time()
            
            result = await integrated_pipeline.process_response(
                scenario["query"], scenario["response"]
            )
            
            execution_time = (time.time() - start_time) * 1000
            resource_stats = resource_monitor.stop_monitoring()
            
            performance_result = {
                'scenario': scenario["query"][:30] + "...",
                'success': result['success'],
                'execution_time_ms': execution_time,
                'claims_extracted': result.get('claims_extracted', 0),
                'peak_memory_mb': resource_stats.get('peak_memory_mb', 0),
                'avg_cpu_percent': resource_stats.get('avg_cpu_percent', 0),
                'expected_max_time': scenario["expected_max_time_ms"]
            }
            
            performance_results.append(performance_result)
            
            # Check performance expectations
            assert result['success'] is True
            assert execution_time <= scenario["expected_max_time_ms"]
        
        # Overall system performance should be consistent
        execution_times = [r['execution_time_ms'] for r in performance_results]
        avg_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Standard deviation should not be too high (performance should be consistent)
        assert std_dev <= avg_time * 0.5  # Max 50% variation
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, integrated_pipeline, resource_monitor):
        """Test for memory leaks during repeated operations."""
        
        await integrated_pipeline.initialize()
        
        # Monitor memory over multiple iterations
        memory_measurements = []
        num_iterations = 10
        
        for i in range(num_iterations):
            resource_monitor.start_monitoring(interval=0.05)
            
            result = await integrated_pipeline.process_response(
                f"Query {i+1}", f"Response {i+1} about metabolomics analysis."
            )
            
            resource_stats = resource_monitor.stop_monitoring()
            memory_measurements.append(resource_stats.get('peak_memory_mb', 0))
            
            # Force garbage collection
            gc.collect()
            
            assert result['success'] is True
        
        # Check for memory growth trend
        if len(memory_measurements) >= 5:
            early_avg = statistics.mean(memory_measurements[:3])
            late_avg = statistics.mean(memory_measurements[-3:])
            
            # Memory should not grow significantly (allowing for some fluctuation)
            memory_growth = late_avg - early_avg
            assert memory_growth <= early_avg * 0.3  # Max 30% growth
    
    @pytest.mark.asyncio
    async def test_cpu_utilization_efficiency(self, integrated_pipeline, resource_monitor):
        """Test CPU utilization efficiency."""
        
        await integrated_pipeline.initialize()
        
        # Test CPU usage during processing
        resource_monitor.start_monitoring(interval=0.02)  # High frequency monitoring
        
        # Process multiple requests
        tasks = []
        for i in range(5):
            task = integrated_pipeline.process_response(
                f"CPU test query {i+1}",
                f"CPU test response {i+1} with metabolomics data analysis."
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        resource_stats = resource_monitor.stop_monitoring()
        
        # All should succeed
        assert all(r['success'] for r in results)
        
        # CPU usage should be reasonable
        avg_cpu = resource_stats.get('avg_cpu_percent', 0)
        peak_cpu = resource_stats.get('peak_cpu_percent', 0)
        
        # Should use CPU but not monopolize it
        assert 0 < avg_cpu < 80  # Should use some CPU but not too much
        assert peak_cpu < 95     # Should not max out CPU


@pytest.mark.performance_validation
class TestScalabilityAndLoad:
    """Test suite for scalability and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_scalability(self, integrated_pipeline):
        """Test scalability with increasing concurrent loads."""
        
        await integrated_pipeline.initialize()
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}
        
        for concurrency in concurrency_levels:
            # Create concurrent requests
            tasks = []
            for i in range(concurrency):
                task = integrated_pipeline.process_response(
                    f"Concurrent query {i+1}",
                    f"Concurrent response {i+1} about metabolomics research."
                )
                tasks.append(task)
            
            # Measure performance
            start_time = time.time()
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = (time.time() - start_time) * 1000
            
            # Analyze results
            successful_results = [r for r in concurrent_results if isinstance(r, dict) and r.get('success')]
            error_count = len(concurrent_results) - len(successful_results)
            
            success_rate = len(successful_results) / len(concurrent_results)
            throughput = len(successful_results) / (total_time / 1000) if total_time > 0 else 0
            
            results[concurrency] = {
                'total_time_ms': total_time,
                'success_rate': success_rate,
                'error_count': error_count,
                'throughput': throughput,
                'avg_response_time': total_time / concurrency if concurrency > 0 else 0
            }
            
            # Success rate should remain high
            assert success_rate >= 0.8  # At least 80% success rate
        
        # Throughput should improve with concurrency (up to a point)
        throughput_1 = results[1]['throughput']
        throughput_5 = results[5]['throughput']
        
        # 5 concurrent should be faster than 1
        assert throughput_5 > throughput_1 * 2
    
    @pytest.mark.asyncio
    async def test_load_testing_sustained(self, integrated_pipeline):
        """Test sustained load over time."""
        
        await integrated_pipeline.initialize()
        
        # Run sustained load for a period
        duration_seconds = 30
        concurrent_requests = 3
        
        start_time = time.time()
        completed_requests = 0
        error_count = 0
        response_times = []
        
        async def make_request(request_id: int):
            nonlocal completed_requests, error_count
            
            req_start = time.time()
            try:
                result = await integrated_pipeline.process_response(
                    f"Load test query {request_id}",
                    f"Load test response {request_id} with detailed metabolomics analysis."
                )
                
                response_time = (time.time() - req_start) * 1000
                response_times.append(response_time)
                
                if result['success']:
                    completed_requests += 1
                else:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        # Generate continuous load
        request_id = 0
        active_tasks = set()
        
        while time.time() - start_time < duration_seconds:
            # Maintain target concurrency
            if len(active_tasks) < concurrent_requests:
                task = asyncio.create_task(make_request(request_id))
                active_tasks.add(task)
                request_id += 1
            
            # Clean up completed tasks
            completed_tasks = [task for task in active_tasks if task.done()]
            for task in completed_tasks:
                active_tasks.remove(task)
            
            await asyncio.sleep(0.1)  # Small delay
        
        # Wait for remaining tasks to complete
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Performance metrics
        throughput = completed_requests / total_time
        error_rate = error_count / (completed_requests + error_count) if completed_requests + error_count > 0 else 1
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Performance expectations
        assert throughput > 1  # At least 1 request per second
        assert error_rate < 0.1  # Less than 10% error rate
        assert avg_response_time < 5000  # Average response under 5 seconds
        
        # Response times should be reasonably consistent
        if len(response_times) > 5:
            std_dev = statistics.stdev(response_times)
            assert std_dev < avg_response_time  # Standard deviation should be less than mean
    
    @pytest.mark.asyncio
    async def test_burst_load_handling(self, integrated_pipeline):
        """Test handling of burst loads."""
        
        await integrated_pipeline.initialize()
        
        # Simulate burst load - many requests at once
        burst_size = 50
        
        # Create burst requests
        tasks = []
        for i in range(burst_size):
            task = integrated_pipeline.process_response(
                f"Burst query {i+1}",
                f"Burst response {i+1} with metabolomics analysis."
            )
            tasks.append(task)
        
        # Execute burst
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        # Analyze burst results
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        error_results = [r for r in results if not isinstance(r, dict) or not r.get('success')]
        
        success_rate = len(successful_results) / len(results)
        throughput = len(successful_results) / (total_time / 1000) if total_time > 0 else 0
        
        # Burst handling expectations
        assert success_rate >= 0.7  # At least 70% success under burst load
        assert throughput > 5  # Should handle at least 5 requests per second in burst
        assert total_time < 15000  # Should complete burst within 15 seconds
    
    @pytest.mark.asyncio
    async def test_resource_limit_behavior(self, integrated_pipeline, resource_monitor):
        """Test behavior under resource constraints."""
        
        await integrated_pipeline.initialize()
        
        # Create resource-intensive requests
        large_responses = [" ".join(SAMPLE_BIOMEDICAL_RESPONSES * 10) for _ in range(10)]
        
        resource_monitor.start_monitoring()
        
        # Process resource-intensive requests
        tasks = []
        for i, response in enumerate(large_responses):
            task = integrated_pipeline.process_response(
                f"Resource test query {i+1}",
                response
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        resource_stats = resource_monitor.stop_monitoring()
        
        # Analyze resource usage
        peak_memory = resource_stats.get('peak_memory_mb', 0)
        avg_cpu = resource_stats.get('avg_cpu_percent', 0)
        
        # Should handle resource-intensive tasks
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.5  # At least 50% success under high resource usage
        assert peak_memory < 1000   # Should not exceed 1GB memory usage
        assert total_time < 30000   # Should complete within 30 seconds


@pytest.mark.performance_validation
class TestPerformanceRegression:
    """Test suite for performance regression detection."""
    
    @pytest.fixture
    def performance_baseline(self):
        """Provide performance baseline measurements."""
        return {
            'claim_extraction_ms': 100,
            'single_claim_verification_ms': 200,
            'accuracy_scoring_ms': 150,
            'report_generation_ms': 300,
            'end_to_end_pipeline_ms': 800,
            'memory_usage_mb': 100,
            'throughput_ops_per_second': 10
        }
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, integrated_pipeline, performance_baseline):
        """Test for performance regression against baseline."""
        
        await integrated_pipeline.initialize()
        
        # Test key operations and compare to baseline
        test_query = "What are the key metabolites in diabetes research?"
        test_response = "Diabetes research shows elevated glucose levels and altered amino acid metabolism."
        
        # Measure current performance
        start_time = time.time()
        result = await integrated_pipeline.process_response(test_query, test_response)
        execution_time = (time.time() - start_time) * 1000
        
        # Performance comparison
        baseline_time = performance_baseline['end_to_end_pipeline_ms']
        performance_ratio = execution_time / baseline_time
        
        # Allow some performance variation but detect significant regression
        assert performance_ratio <= 2.0, f"Performance regression detected: {execution_time}ms vs baseline {baseline_time}ms"
        
        # Log performance for tracking
        print(f"Performance comparison: Current={execution_time:.1f}ms, Baseline={baseline_time}ms, Ratio={performance_ratio:.2f}")
    
    @pytest.mark.asyncio
    async def test_memory_usage_regression(self, integrated_pipeline, performance_baseline, resource_monitor):
        """Test for memory usage regression."""
        
        await integrated_pipeline.initialize()
        
        # Measure memory usage during typical operation
        resource_monitor.start_monitoring()
        
        # Process several requests to get stable memory measurement
        for i in range(5):
            await integrated_pipeline.process_response(
                f"Memory test query {i+1}",
                f"Memory test response {i+1} about metabolomics research."
            )
        
        resource_stats = resource_monitor.stop_monitoring()
        current_memory = resource_stats.get('peak_memory_mb', 0)
        baseline_memory = performance_baseline['memory_usage_mb']
        
        memory_ratio = current_memory / baseline_memory if baseline_memory > 0 else 1
        
        # Allow reasonable memory increase but detect significant regression
        assert memory_ratio <= 2.5, f"Memory usage regression: {current_memory}MB vs baseline {baseline_memory}MB"
        
        print(f"Memory comparison: Current={current_memory:.1f}MB, Baseline={baseline_memory}MB, Ratio={memory_ratio:.2f}")
    
    @pytest.mark.asyncio
    async def test_throughput_regression(self, integrated_pipeline, performance_baseline):
        """Test for throughput regression."""
        
        await integrated_pipeline.initialize()
        
        # Measure throughput
        num_requests = 20
        
        tasks = []
        for i in range(num_requests):
            task = integrated_pipeline.process_response(
                f"Throughput test {i+1}",
                "Throughput test response about metabolomics analysis."
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_results = [r for r in results if r.get('success')]
        current_throughput = len(successful_results) / total_time
        baseline_throughput = performance_baseline['throughput_ops_per_second']
        
        throughput_ratio = current_throughput / baseline_throughput if baseline_throughput > 0 else 1
        
        # Throughput should not significantly decrease
        assert throughput_ratio >= 0.5, f"Throughput regression: {current_throughput:.2f} ops/s vs baseline {baseline_throughput} ops/s"
        
        print(f"Throughput comparison: Current={current_throughput:.2f} ops/s, Baseline={baseline_throughput} ops/s, Ratio={throughput_ratio:.2f}")


@pytest.mark.performance_validation
class TestOptimizationValidation:
    """Test suite for validating performance optimizations."""
    
    @pytest.mark.asyncio
    async def test_caching_optimization(self, integrated_pipeline):
        """Test caching optimization effectiveness."""
        
        await integrated_pipeline.initialize()
        
        # Same request multiple times to test caching
        test_query = "What are glucose levels in diabetes?"
        test_response = "Glucose levels are elevated in diabetes patients."
        
        # First request (cold cache)
        start_time = time.time()
        result1 = await integrated_pipeline.process_response(test_query, test_response)
        first_time = (time.time() - start_time) * 1000
        
        # Second request (warm cache)
        start_time = time.time()
        result2 = await integrated_pipeline.process_response(test_query, test_response)
        second_time = (time.time() - start_time) * 1000
        
        # Third request (warm cache)
        start_time = time.time()
        result3 = await integrated_pipeline.process_response(test_query, test_response)
        third_time = (time.time() - start_time) * 1000
        
        # All should succeed
        assert result1['success'] and result2['success'] and result3['success']
        
        # Subsequent requests should be faster due to caching (if implemented)
        # If no caching, this test documents the potential for optimization
        avg_cached_time = (second_time + third_time) / 2
        print(f"Cache optimization potential: First={first_time:.1f}ms, Cached avg={avg_cached_time:.1f}ms")
        
        # Performance should be reasonably consistent
        time_variation = abs(second_time - third_time)
        assert time_variation < first_time * 0.5  # Variation should be less than 50% of first request
    
    @pytest.mark.asyncio
    async def test_batch_processing_optimization(self, integrated_pipeline):
        """Test batch processing optimization."""
        
        await integrated_pipeline.initialize()
        
        # Compare sequential vs concurrent processing
        test_cases = [
            ("Query 1", "Response 1 about glucose metabolism."),
            ("Query 2", "Response 2 about LC-MS analysis."),
            ("Query 3", "Response 3 about statistical methods."),
            ("Query 4", "Response 4 about biomarker discovery."),
            ("Query 5", "Response 5 about clinical applications.")
        ]
        
        # Sequential processing
        sequential_start = time.time()
        sequential_results = []
        for query, response in test_cases:
            result = await integrated_pipeline.process_response(query, response)
            sequential_results.append(result)
        sequential_time = (time.time() - sequential_start) * 1000
        
        # Concurrent processing
        concurrent_start = time.time()
        tasks = [integrated_pipeline.process_response(q, r) for q, r in test_cases]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = (time.time() - concurrent_start) * 1000
        
        # Both should succeed
        assert all(r['success'] for r in sequential_results)
        assert all(r['success'] for r in concurrent_results)
        
        # Concurrent should be faster
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
        print(f"Batch processing speedup: Sequential={sequential_time:.1f}ms, Concurrent={concurrent_time:.1f}ms, Speedup={speedup:.2f}x")
        
        # Should achieve some speedup from concurrency
        assert speedup >= 1.2  # At least 20% speedup
    
    @pytest.mark.asyncio
    async def test_resource_pooling_optimization(self, integrated_pipeline, resource_monitor):
        """Test resource pooling optimization."""
        
        await integrated_pipeline.initialize()
        
        # Test resource usage with many rapid requests
        num_requests = 30
        
        resource_monitor.start_monitoring(interval=0.1)
        
        # Rapid fire requests
        tasks = []
        for i in range(num_requests):
            task = integrated_pipeline.process_response(
                f"Resource pool test {i+1}",
                f"Resource pool response {i+1} with analysis."
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        resource_stats = resource_monitor.stop_monitoring()
        
        # All should succeed
        successful_count = sum(1 for r in results if r.get('success'))
        success_rate = successful_count / num_requests
        
        assert success_rate >= 0.9  # At least 90% success with resource pooling
        
        # Memory usage should be reasonable despite many requests
        peak_memory = resource_stats.get('peak_memory_mb', 0)
        assert peak_memory < 500  # Should not exceed 500MB even with many requests
        
        print(f"Resource pooling test: {successful_count}/{num_requests} succeeded, Peak memory: {peak_memory:.1f}MB")


if __name__ == "__main__":
    # Run the performance test suite
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance_validation"])