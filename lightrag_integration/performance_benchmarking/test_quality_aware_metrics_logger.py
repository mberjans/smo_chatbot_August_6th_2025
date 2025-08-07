#!/usr/bin/env python3
"""
Unit tests for Quality Aware Metrics Logger module.

This module provides comprehensive testing for the QualityAwareAPIMetricsLogger
and related components, including metrics collection, aggregation, analysis,
and reporting functionality.

Author: Claude Code (Anthropic)
Created: August 7, 2025
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from dataclasses import asdict

# Import the modules under test
from quality_aware_metrics_logger import (
    QualityAPIMetric,
    QualityMetricsAggregator,
    QualityAwareAPIMetricsLogger
)

# Import parent classes and dependencies
try:
    from api_metrics_logger import APIMetric
    from tests.performance_test_fixtures import ResourceUsageSnapshot
except ImportError:
    # Create mock classes for testing
    class APIMetric:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ResourceUsageSnapshot:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class TestQualityAPIMetric:
    """Test suite for QualityAPIMetric class."""
    
    def test_metric_initialization(self):
        """Test proper initialization of QualityAPIMetric."""
        metric = QualityAPIMetric(
            timestamp=time.time(),
            endpoint="/api/validate",
            method="POST",
            response_time_ms=1500.0,
            status_code=200,
            cost_usd=0.05,
            quality_validation_type="factual_accuracy",
            validation_accuracy_score=85.0,
            confidence_score=0.8,
            claims_processed=10,
            validation_duration_ms=800.0
        )
        
        assert metric.endpoint == "/api/validate"
        assert metric.method == "POST"
        assert metric.response_time_ms == 1500.0
        assert metric.quality_validation_type == "factual_accuracy"
        assert metric.validation_accuracy_score == 85.0
        assert metric.confidence_score == 0.8
        assert metric.claims_processed == 10
    
    def test_calculate_quality_efficiency_ratio(self):
        """Test quality efficiency ratio calculation."""
        # Test with valid data
        metric = QualityAPIMetric(
            response_time_ms=2000.0,
            validation_accuracy_score=90.0,
            confidence_score=0.85
        )
        
        ratio = metric.calculate_quality_efficiency_ratio()
        
        assert isinstance(ratio, float)
        assert ratio > 0
        
        # Should be (90 * 0.85) / 2000 = 0.03825
        expected_ratio = (90.0 * 0.85) / 2000.0
        assert abs(ratio - expected_ratio) < 0.0001
    
    def test_calculate_quality_efficiency_ratio_edge_cases(self):
        """Test quality efficiency ratio with edge case values."""
        # Test with zero response time
        metric_zero_time = QualityAPIMetric(
            response_time_ms=0.0,
            validation_accuracy_score=85.0,
            confidence_score=0.8
        )
        ratio_zero = metric_zero_time.calculate_quality_efficiency_ratio()
        assert ratio_zero == 0.0
        
        # Test with zero quality scores
        metric_zero_quality = QualityAPIMetric(
            response_time_ms=1000.0,
            validation_accuracy_score=0.0,
            confidence_score=0.0
        )
        ratio_zero_quality = metric_zero_quality.calculate_quality_efficiency_ratio()
        assert ratio_zero_quality == 0.0
    
    def test_get_cost_per_claim(self):
        """Test cost per claim calculation."""
        # Test with valid data
        metric = QualityAPIMetric(
            cost_usd=0.10,
            claims_processed=5
        )
        
        cost_per_claim = metric.get_cost_per_claim()
        assert cost_per_claim == 0.02  # 0.10 / 5
        
        # Test with zero claims
        metric_zero_claims = QualityAPIMetric(
            cost_usd=0.05,
            claims_processed=0
        )
        cost_zero_claims = metric_zero_claims.get_cost_per_claim()
        assert cost_zero_claims == 0.0
    
    def test_get_quality_validation_cost_usd(self):
        """Test quality validation cost calculation."""
        # Test with quality validation percentage
        metric = QualityAPIMetric(
            cost_usd=0.20,
            quality_validation_cost_percentage=30.0
        )
        
        validation_cost = metric.get_quality_validation_cost_usd()
        assert validation_cost == 0.06  # 20% of 0.20
        
        # Test without percentage (should return 0)
        metric_no_percentage = QualityAPIMetric(cost_usd=0.15)
        validation_cost_no_pct = metric_no_percentage.get_quality_validation_cost_usd()
        assert validation_cost_no_pct == 0.0
    
    def test_is_quality_threshold_met(self):
        """Test quality threshold checking."""
        metric = QualityAPIMetric(
            validation_accuracy_score=88.0,
            confidence_score=0.82
        )
        
        # Test with thresholds met
        assert metric.is_quality_threshold_met(
            accuracy_threshold=85.0,
            confidence_threshold=0.8
        ) is True
        
        # Test with accuracy threshold not met
        assert metric.is_quality_threshold_met(
            accuracy_threshold=90.0,
            confidence_threshold=0.8
        ) is False
        
        # Test with confidence threshold not met
        assert metric.is_quality_threshold_met(
            accuracy_threshold=85.0,
            confidence_threshold=0.85
        ) is False
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization of QualityAPIMetric."""
        timestamp = time.time()
        metric = QualityAPIMetric(
            timestamp=timestamp,
            endpoint="/test",
            quality_validation_type="relevance",
            validation_accuracy_score=92.5
        )
        
        metric_dict = metric.to_dict()
        
        assert isinstance(metric_dict, dict)
        assert metric_dict['timestamp'] == timestamp
        assert metric_dict['endpoint'] == "/test"
        assert metric_dict['quality_validation_type'] == "relevance"
        assert metric_dict['validation_accuracy_score'] == 92.5
    
    def test_from_dict_deserialization(self):
        """Test dictionary deserialization to QualityAPIMetric."""
        metric_data = {
            'timestamp': time.time(),
            'endpoint': '/deserialize/test',
            'quality_validation_type': 'integrated',
            'validation_accuracy_score': 87.3,
            'confidence_score': 0.79,
            'claims_processed': 8
        }
        
        metric = QualityAPIMetric.from_dict(metric_data)
        
        assert metric.endpoint == '/deserialize/test'
        assert metric.quality_validation_type == 'integrated'
        assert metric.validation_accuracy_score == 87.3
        assert metric.confidence_score == 0.79
        assert metric.claims_processed == 8


class TestQualityMetricsAggregator:
    """Test suite for QualityMetricsAggregator class."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample quality metrics for testing."""
        base_time = time.time()
        return [
            QualityAPIMetric(
                timestamp=base_time - 3600,  # 1 hour ago
                endpoint="/api/factual",
                response_time_ms=1200.0,
                status_code=200,
                cost_usd=0.08,
                quality_validation_type="factual_accuracy",
                validation_accuracy_score=88.5,
                confidence_score=0.82,
                claims_processed=6,
                validation_duration_ms=750.0
            ),
            QualityAPIMetric(
                timestamp=base_time - 1800,  # 30 minutes ago
                endpoint="/api/relevance",
                response_time_ms=900.0,
                status_code=200,
                cost_usd=0.06,
                quality_validation_type="relevance_scoring",
                validation_accuracy_score=91.2,
                confidence_score=0.87,
                claims_processed=4,
                validation_duration_ms=600.0
            ),
            QualityAPIMetric(
                timestamp=base_time - 900,  # 15 minutes ago
                endpoint="/api/integrated",
                response_time_ms=1800.0,
                status_code=200,
                cost_usd=0.12,
                quality_validation_type="integrated_workflow",
                validation_accuracy_score=85.7,
                confidence_score=0.79,
                claims_processed=8,
                validation_duration_ms=1200.0
            )
        ]
    
    def test_aggregator_initialization(self):
        """Test QualityMetricsAggregator initialization."""
        aggregator = QualityMetricsAggregator(buffer_size=100, aggregation_window_seconds=300)
        
        assert aggregator.buffer_size == 100
        assert aggregator.aggregation_window_seconds == 300
        assert len(aggregator._metrics_buffer) == 0
        assert isinstance(aggregator.aggregated_metrics, dict)
    
    def test_add_metric(self, sample_metrics):
        """Test adding metrics to aggregator."""
        aggregator = QualityMetricsAggregator(buffer_size=10)
        
        # Add metrics
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        assert len(aggregator._metrics_buffer) == 3
        assert aggregator._metrics_buffer[0] == sample_metrics[0]
    
    def test_buffer_overflow_handling(self, sample_metrics):
        """Test buffer overflow handling."""
        aggregator = QualityMetricsAggregator(buffer_size=2)  # Small buffer
        
        # Add more metrics than buffer size
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        # Should keep only the most recent metrics
        assert len(aggregator._metrics_buffer) == 2
        assert aggregator._metrics_buffer[-1] == sample_metrics[-1]  # Most recent
    
    def test_get_metrics_by_endpoint(self, sample_metrics):
        """Test filtering metrics by endpoint."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        factual_metrics = aggregator.get_metrics_by_endpoint("/api/factual")
        assert len(factual_metrics) == 1
        assert factual_metrics[0].endpoint == "/api/factual"
        
        relevance_metrics = aggregator.get_metrics_by_endpoint("/api/relevance")
        assert len(relevance_metrics) == 1
        assert relevance_metrics[0].quality_validation_type == "relevance_scoring"
    
    def test_get_metrics_by_validation_type(self, sample_metrics):
        """Test filtering metrics by validation type."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        factual_metrics = aggregator.get_metrics_by_validation_type("factual_accuracy")
        assert len(factual_metrics) == 1
        assert factual_metrics[0].quality_validation_type == "factual_accuracy"
        
        integrated_metrics = aggregator.get_metrics_by_validation_type("integrated_workflow")
        assert len(integrated_metrics) == 1
        assert integrated_metrics[0].validation_duration_ms == 1200.0
    
    def test_get_recent_metrics(self, sample_metrics):
        """Test retrieval of recent metrics within time window."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        # Get metrics from last 45 minutes
        recent_metrics = aggregator.get_recent_metrics(time_window_seconds=2700)
        
        # Should include the two most recent metrics
        assert len(recent_metrics) == 2
        assert all(metric.timestamp > time.time() - 2700 for metric in recent_metrics)
    
    def test_calculate_aggregate_statistics(self, sample_metrics):
        """Test aggregate statistics calculation."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        stats = aggregator.calculate_aggregate_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'average_response_time_ms' in stats
        assert 'average_validation_accuracy' in stats
        assert 'average_confidence_score' in stats
        assert 'total_cost_usd' in stats
        assert 'total_claims_processed' in stats
        
        assert stats['total_requests'] == 3
        assert stats['total_cost_usd'] == 0.26  # 0.08 + 0.06 + 0.12
        assert stats['total_claims_processed'] == 18  # 6 + 4 + 8
    
    def test_calculate_quality_performance_metrics(self, sample_metrics):
        """Test quality-performance metrics calculation."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        qp_metrics = aggregator.calculate_quality_performance_metrics()
        
        assert isinstance(qp_metrics, dict)
        assert 'quality_efficiency_ratio_avg' in qp_metrics
        assert 'cost_per_claim_avg' in qp_metrics
        assert 'validation_duration_ratio_avg' in qp_metrics
        assert 'quality_vs_speed_correlation' in qp_metrics
        
        # Verify calculations
        assert qp_metrics['quality_efficiency_ratio_avg'] > 0
        assert qp_metrics['cost_per_claim_avg'] > 0
    
    def test_get_performance_trends(self, sample_metrics):
        """Test performance trend analysis."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        trends = aggregator.get_performance_trends()
        
        assert isinstance(trends, dict)
        assert 'response_time_trend' in trends
        assert 'accuracy_trend' in trends
        assert 'cost_trend' in trends
        assert 'confidence_trend' in trends
        
        # Each trend should have direction and magnitude
        for trend_name, trend_data in trends.items():
            if trend_data:  # If there's trend data
                assert 'direction' in trend_data
                assert trend_data['direction'] in ['increasing', 'decreasing', 'stable']
    
    def test_identify_quality_issues(self, sample_metrics):
        """Test quality issue identification."""
        aggregator = QualityMetricsAggregator()
        
        # Add a metric with low quality scores to trigger issues
        low_quality_metric = QualityAPIMetric(
            timestamp=time.time(),
            endpoint="/api/problematic",
            response_time_ms=3000.0,  # Slow
            status_code=200,
            cost_usd=0.20,  # Expensive
            quality_validation_type="factual_accuracy",
            validation_accuracy_score=65.0,  # Low accuracy
            confidence_score=0.4,  # Low confidence
            claims_processed=2,  # Low productivity
            validation_duration_ms=2500.0  # Slow validation
        )
        
        for metric in sample_metrics + [low_quality_metric]:
            aggregator.add_metric(metric)
        
        issues = aggregator.identify_quality_issues()
        
        assert isinstance(issues, list)
        # Should identify multiple issues with the problematic metric
        assert len(issues) > 0
        
        # Check that issues contain relevant information
        for issue in issues:
            assert 'issue_type' in issue
            assert 'description' in issue
            assert 'affected_metrics' in issue
            assert 'severity' in issue
    
    def test_export_aggregated_data(self, sample_metrics):
        """Test exporting aggregated data."""
        aggregator = QualityMetricsAggregator()
        
        for metric in sample_metrics:
            aggregator.add_metric(metric)
        
        # Calculate aggregated metrics first
        aggregator.calculate_aggregate_statistics()
        aggregator.calculate_quality_performance_metrics()
        
        exported_data = aggregator.export_aggregated_data()
        
        assert isinstance(exported_data, dict)
        assert 'aggregated_metrics' in exported_data
        assert 'quality_performance_metrics' in exported_data
        assert 'raw_metrics' in exported_data
        assert 'export_timestamp' in exported_data
        
        # Raw metrics should be serializable
        raw_metrics = exported_data['raw_metrics']
        assert len(raw_metrics) == 3
        assert all(isinstance(metric, dict) for metric in raw_metrics)


class TestQualityAwareAPIMetricsLogger:
    """Test suite for QualityAwareAPIMetricsLogger class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response for testing."""
        response = Mock()
        response.status_code = 200
        response.headers = {'content-type': 'application/json'}
        response.json.return_value = {
            'validation_result': {
                'accuracy_score': 87.5,
                'confidence_score': 0.83,
                'claims_processed': 5
            }
        }
        response.text = json.dumps(response.json.return_value)
        return response
    
    def test_logger_initialization(self, temp_output_dir):
        """Test QualityAwareAPIMetricsLogger initialization."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_metrics.log",
            buffer_size=50,
            auto_flush_interval_seconds=30
        )
        
        assert logger.log_file == temp_output_dir / "test_metrics.log"
        assert isinstance(logger.metrics_aggregator, QualityMetricsAggregator)
        assert logger.metrics_aggregator.buffer_size == 50
        assert logger._auto_flush_interval == 30
    
    @pytest.mark.asyncio
    async def test_log_quality_api_call_async(self, temp_output_dir, mock_response):
        """Test asynchronous quality API call logging."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_async.log"
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            metric = await logger.log_quality_api_call_async(
                endpoint="/api/validate",
                method="POST",
                request_data={"text": "test content"},
                quality_validation_type="factual_accuracy"
            )
            
            assert isinstance(metric, QualityAPIMetric)
            assert metric.endpoint == "/api/validate"
            assert metric.method == "POST"
            assert metric.quality_validation_type == "factual_accuracy"
            assert metric.validation_accuracy_score == 87.5
            assert metric.confidence_score == 0.83
            assert metric.claims_processed == 5
    
    def test_log_quality_api_call_sync(self, temp_output_dir, mock_response):
        """Test synchronous quality API call logging."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_sync.log"
        )
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            metric = logger.log_quality_api_call(
                endpoint="/api/sync/validate",
                method="POST",
                request_data={"query": "test query", "response": "test response"},
                quality_validation_type="relevance_scoring"
            )
            
            assert isinstance(metric, QualityAPIMetric)
            assert metric.endpoint == "/api/sync/validate"
            assert metric.quality_validation_type == "relevance_scoring"
            assert metric.response_time_ms > 0  # Should have measured time
    
    def test_log_existing_quality_metric(self, temp_output_dir):
        """Test logging of existing quality metric."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_existing.log"
        )
        
        existing_metric = QualityAPIMetric(
            timestamp=time.time(),
            endpoint="/api/existing",
            method="GET",
            response_time_ms=850.0,
            status_code=200,
            cost_usd=0.04,
            quality_validation_type="integrated_workflow",
            validation_accuracy_score=89.2,
            confidence_score=0.86,
            claims_processed=3
        )
        
        logger.log_quality_metric(existing_metric)
        
        # Check that metric was added to aggregator
        recent_metrics = logger.metrics_aggregator.get_recent_metrics(3600)
        assert len(recent_metrics) == 1
        assert recent_metrics[0] == existing_metric
    
    def test_get_quality_metrics_summary(self, temp_output_dir):
        """Test quality metrics summary generation."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_summary.log"
        )
        
        # Add sample metrics
        sample_metrics = [
            QualityAPIMetric(
                timestamp=time.time() - 1800,
                endpoint="/api/test1",
                response_time_ms=1100.0,
                validation_accuracy_score=88.0,
                confidence_score=0.82,
                claims_processed=4
            ),
            QualityAPIMetric(
                timestamp=time.time() - 900,
                endpoint="/api/test2",
                response_time_ms=1300.0,
                validation_accuracy_score=91.5,
                confidence_score=0.87,
                claims_processed=6
            )
        ]
        
        for metric in sample_metrics:
            logger.log_quality_metric(metric)
        
        summary = logger.get_quality_metrics_summary(time_window_seconds=3600)
        
        assert isinstance(summary, dict)
        assert 'total_requests' in summary
        assert 'time_window_hours' in summary
        assert 'average_quality_scores' in summary
        assert 'performance_statistics' in summary
        assert 'quality_issues' in summary
        
        assert summary['total_requests'] == 2
        assert summary['time_window_hours'] == 1.0
    
    def test_analyze_quality_performance_correlation(self, temp_output_dir):
        """Test quality-performance correlation analysis."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_correlation.log"
        )
        
        # Add metrics with varied quality/performance characteristics
        test_metrics = [
            QualityAPIMetric(
                timestamp=time.time() - 3000,
                response_time_ms=1000.0,  # Fast
                validation_accuracy_score=95.0,  # High quality
                confidence_score=0.92,
                claims_processed=8
            ),
            QualityAPIMetric(
                timestamp=time.time() - 2000,
                response_time_ms=2000.0,  # Slow
                validation_accuracy_score=78.0,  # Lower quality
                confidence_score=0.71,
                claims_processed=4
            ),
            QualityAPIMetric(
                timestamp=time.time() - 1000,
                response_time_ms=1500.0,  # Medium
                validation_accuracy_score=85.0,  # Medium quality
                confidence_score=0.81,
                claims_processed=6
            )
        ]
        
        for metric in test_metrics:
            logger.log_quality_metric(metric)
        
        correlation_analysis = logger.analyze_quality_performance_correlation()
        
        assert isinstance(correlation_analysis, dict)
        assert 'response_time_vs_accuracy' in correlation_analysis
        assert 'response_time_vs_confidence' in correlation_analysis
        assert 'cost_efficiency_analysis' in correlation_analysis
        assert 'quality_consistency_analysis' in correlation_analysis
    
    def test_generate_quality_optimization_recommendations(self, temp_output_dir):
        """Test quality optimization recommendations generation."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_recommendations.log"
        )
        
        # Add metrics with identifiable issues
        problematic_metrics = [
            QualityAPIMetric(
                timestamp=time.time() - 2000,
                response_time_ms=4000.0,  # Very slow
                validation_accuracy_score=60.0,  # Low quality
                confidence_score=0.45,  # Low confidence
                cost_usd=0.25,  # Expensive
                claims_processed=2  # Low productivity
            ),
            QualityAPIMetric(
                timestamp=time.time() - 1000,
                response_time_ms=3500.0,  # Slow
                validation_accuracy_score=65.0,  # Low quality
                confidence_score=0.50,
                cost_usd=0.22,
                claims_processed=3
            )
        ]
        
        for metric in problematic_metrics:
            logger.log_quality_metric(metric)
        
        recommendations = logger.generate_quality_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should identify performance and quality issues
        recommendation_types = [rec.get('recommendation_type') for rec in recommendations]
        assert any('performance' in rec_type.lower() for rec_type in recommendation_types)
        assert any('quality' in rec_type.lower() for rec_type in recommendation_types)
    
    @pytest.mark.asyncio
    async def test_export_quality_metrics_async(self, temp_output_dir):
        """Test asynchronous quality metrics export."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_export.log"
        )
        
        # Add sample metrics
        sample_metric = QualityAPIMetric(
            timestamp=time.time(),
            endpoint="/api/export_test",
            validation_accuracy_score=90.0,
            confidence_score=0.88
        )
        logger.log_quality_metric(sample_metric)
        
        export_path = await logger.export_quality_metrics_async(
            output_file=temp_output_dir / "exported_metrics.json",
            include_raw_data=True
        )
        
        assert export_path.exists()
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert 'aggregated_metrics' in exported_data
        assert 'quality_performance_metrics' in exported_data
        assert 'raw_metrics' in exported_data
        assert len(exported_data['raw_metrics']) == 1
    
    def test_auto_flush_functionality(self, temp_output_dir):
        """Test automatic flushing of metrics."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_autoflush.log",
            auto_flush_interval_seconds=1  # Very short interval for testing
        )
        
        # Add metric
        test_metric = QualityAPIMetric(
            timestamp=time.time(),
            endpoint="/api/flush_test",
            validation_accuracy_score=85.0
        )
        logger.log_quality_metric(test_metric)
        
        # Start auto-flush
        logger.start_auto_flush()
        
        # Wait for flush
        time.sleep(1.2)
        
        # Stop auto-flush
        logger.stop_auto_flush()
        
        # Check that log file was created and contains data
        assert logger.log_file.exists()
        assert logger.log_file.stat().st_size > 0
    
    def test_flush_metrics_to_file(self, temp_output_dir):
        """Test manual flushing of metrics to file."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "test_manual_flush.log"
        )
        
        # Add multiple metrics
        for i in range(5):
            metric = QualityAPIMetric(
                timestamp=time.time() - (i * 300),  # 5 minutes apart
                endpoint=f"/api/test_{i}",
                validation_accuracy_score=80.0 + i,
                confidence_score=0.75 + (i * 0.05)
            )
            logger.log_quality_metric(metric)
        
        # Flush to file
        flush_result = logger.flush_metrics_to_file()
        
        assert flush_result is True
        assert logger.log_file.exists()
        
        # Verify file contents
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 5  # Should have 5 metric entries


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_logging_workflow(self, temp_output_dir):
        """Test complete end-to-end quality logging workflow."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "e2e_test.log",
            buffer_size=10,
            auto_flush_interval_seconds=2
        )
        
        # Start auto-flush
        logger.start_auto_flush()
        
        try:
            # Simulate multiple API calls
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'validation_result': {
                        'accuracy_score': 89.0,
                        'confidence_score': 0.84,
                        'claims_processed': 6
                    }
                }
                mock_post.return_value.__aenter__.return_value = mock_response
                
                # Log multiple async calls
                tasks = []
                for i in range(3):
                    task = logger.log_quality_api_call_async(
                        endpoint=f"/api/test_{i}",
                        method="POST",
                        request_data={"test": f"data_{i}"},
                        quality_validation_type="factual_accuracy"
                    )
                    tasks.append(task)
                
                metrics = await asyncio.gather(*tasks)
                
                # Verify all metrics were logged
                assert len(metrics) == 3
                assert all(isinstance(m, QualityAPIMetric) for m in metrics)
                
                # Generate summary
                summary = logger.get_quality_metrics_summary()
                assert summary['total_requests'] >= 3
                
                # Export data
                export_path = await logger.export_quality_metrics_async(
                    output_file=temp_output_dir / "e2e_export.json"
                )
                assert export_path.exists()
                
        finally:
            logger.stop_auto_flush()
    
    def test_error_handling_invalid_api_responses(self, temp_output_dir):
        """Test error handling with invalid API responses."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "error_test.log"
        )
        
        # Test with invalid JSON response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Invalid JSON response"
            mock_post.return_value = mock_response
            
            # Should handle gracefully
            metric = logger.log_quality_api_call(
                endpoint="/api/invalid_json",
                method="POST",
                request_data={"test": "data"},
                quality_validation_type="factual_accuracy"
            )
            
            assert isinstance(metric, QualityAPIMetric)
            # Should have default quality values when parsing fails
            assert metric.validation_accuracy_score == 0.0
            assert metric.confidence_score == 0.0
    
    def test_error_handling_http_errors(self, temp_output_dir):
        """Test error handling with HTTP errors."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "http_error_test.log"
        )
        
        # Test with HTTP 500 error
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response
            
            metric = logger.log_quality_api_call(
                endpoint="/api/server_error",
                method="POST",
                request_data={"test": "data"},
                quality_validation_type="relevance_scoring"
            )
            
            assert isinstance(metric, QualityAPIMetric)
            assert metric.status_code == 500
            assert metric.error_occurred is True
    
    def test_concurrent_logging_operations(self, temp_output_dir):
        """Test concurrent logging operations."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "concurrent_test.log"
        )
        
        # Simulate concurrent metric logging
        def log_test_metric(i):
            metric = QualityAPIMetric(
                timestamp=time.time(),
                endpoint=f"/api/concurrent_{i}",
                validation_accuracy_score=80.0 + i,
                confidence_score=0.7 + (i * 0.01)
            )
            logger.log_quality_metric(metric)
        
        # Use threading to simulate concurrent operations
        import threading
        threads = []
        for i in range(10):
            thread = threading.Thread(target=log_test_metric, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all metrics were logged
        recent_metrics = logger.metrics_aggregator.get_recent_metrics(3600)
        assert len(recent_metrics) == 10
    
    def test_memory_management_large_datasets(self, temp_output_dir):
        """Test memory management with large datasets."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "memory_test.log",
            buffer_size=100  # Reasonable buffer size
        )
        
        # Add large number of metrics
        for i in range(500):
            metric = QualityAPIMetric(
                timestamp=time.time() - i,
                endpoint=f"/api/load_test_{i}",
                validation_accuracy_score=75.0 + (i % 20),
                confidence_score=0.6 + ((i % 30) * 0.01)
            )
            logger.log_quality_metric(metric)
        
        # Buffer should maintain size limit
        assert len(logger.metrics_aggregator._metrics_buffer) <= 100
        
        # Should still be able to generate summary
        summary = logger.get_quality_metrics_summary()
        assert isinstance(summary, dict)
        assert summary['total_requests'] <= 100
    
    def test_metric_serialization_edge_cases(self, temp_output_dir):
        """Test metric serialization with edge case values."""
        logger = QualityAwareAPIMetricsLogger(
            log_file=temp_output_dir / "serialization_test.log"
        )
        
        # Test with extreme values
        extreme_metric = QualityAPIMetric(
            timestamp=time.time(),
            endpoint="/api/extreme",
            response_time_ms=float('inf'),  # Infinite response time
            cost_usd=0.0,  # Zero cost
            validation_accuracy_score=100.0,  # Perfect accuracy
            confidence_score=0.0,  # Zero confidence
            claims_processed=-1  # Invalid negative count
        )
        
        # Should handle extreme values gracefully
        logger.log_quality_metric(extreme_metric)
        
        # Export and verify serialization works
        export_path = temp_output_dir / "extreme_export.json"
        success = logger.export_quality_metrics(
            output_file=export_path,
            include_raw_data=True
        )
        
        assert success is True
        assert export_path.exists()
        
        # Verify exported data is valid JSON
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert isinstance(exported_data, dict)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])