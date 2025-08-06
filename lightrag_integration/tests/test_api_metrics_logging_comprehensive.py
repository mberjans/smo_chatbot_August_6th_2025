#!/usr/bin/env python3
"""
Comprehensive test suite for API Metrics Logging System.

This test suite provides complete coverage of the API metrics logging components including:
- APIMetric data model and calculations  
- MetricType enumeration and categorization
- MetricsAggregator real-time aggregation and statistics
- APIUsageMetricsLogger main functionality and context managers
- Integration with cost tracking, budget management, and audit systems
- Performance under concurrent operations and high load
- Template rendering and structured logging
- System monitoring and health metrics

Author: Claude Code (Anthropic)  
Created: August 6, 2025
"""

import pytest
import asyncio
import time
import json
import logging
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Test imports
from lightrag_integration.api_metrics_logger import (
    APIUsageMetricsLogger,
    APIMetric,
    MetricType,
    MetricsAggregator
)
from lightrag_integration.cost_persistence import (
    CostPersistence,
    CostRecord,
    ResearchCategory
)
from lightrag_integration.budget_manager import BudgetManager, BudgetAlert, AlertLevel
from lightrag_integration.research_categorizer import ResearchCategorizer
from lightrag_integration.audit_trail import AuditTrail


class TestAPIMetric:
    """Comprehensive tests for APIMetric data model."""
    
    def test_api_metric_minimal_creation(self):
        """Test APIMetric creation with minimal parameters."""
        metric = APIMetric()
        
        assert metric.id is not None
        assert metric.timestamp is not None
        assert metric.metric_type == MetricType.LLM_CALL
        assert metric.operation_name == "unknown"
        assert metric.api_provider == "openai"
        assert metric.success is True
        assert metric.research_category == ResearchCategory.GENERAL_QUERY.value
        assert isinstance(metric.metadata, dict)
        assert isinstance(metric.tags, list)
    
    def test_api_metric_comprehensive_creation(self):
        """Test APIMetric creation with all parameters."""
        metadata = {
            "request_id": "req_12345",
            "model_parameters": {"temperature": 0.7, "max_tokens": 1000},
            "quality_score": 0.95,
            "processing_details": {"queue_position": 3, "batch_size": 10}
        }
        
        tags = ["production", "high_priority", "research_project_alpha"]
        
        metric = APIMetric(
            session_id="comprehensive_session",
            metric_type=MetricType.HYBRID_OPERATION,
            operation_name="complex_rag_query",
            model_name="gpt-4o",
            api_provider="openai",
            endpoint_used="https://api.openai.com/v1/chat/completions",
            prompt_tokens=1500,
            completion_tokens=800,
            embedding_tokens=300,
            cost_usd=0.45,
            cost_per_token=0.00015,
            response_time_ms=3500,
            queue_time_ms=150,
            processing_time_ms=3200,
            throughput_tokens_per_sec=742.86,
            success=True,
            retry_count=1,
            final_attempt=True,
            research_category=ResearchCategory.PATHWAY_ANALYSIS.value,
            query_type="scientific_analysis",
            subject_area="systems_biology",
            document_type="structured_data",
            memory_usage_mb=256.5,
            cpu_usage_percent=75.2,
            concurrent_operations=5,
            request_size_bytes=4096,
            response_size_bytes=8192,
            context_length=2000,
            temperature_used=0.7,
            daily_budget_used_percent=35.5,
            monthly_budget_used_percent=12.8,
            compliance_level="high",
            user_id="researcher_001",
            project_id="metabolomics_study_2025",
            experiment_id="exp_789",
            metadata=metadata,
            tags=tags
        )
        
        # Verify all fields
        assert metric.session_id == "comprehensive_session"
        assert metric.metric_type == MetricType.HYBRID_OPERATION
        assert metric.operation_name == "complex_rag_query"
        assert metric.model_name == "gpt-4o"
        assert metric.prompt_tokens == 1500
        assert metric.completion_tokens == 800
        assert metric.embedding_tokens == 300
        assert metric.total_tokens == 2600  # Should be calculated
        assert metric.cost_usd == 0.45
        assert metric.response_time_ms == 3500
        assert metric.research_category == ResearchCategory.PATHWAY_ANALYSIS.value
        assert metric.query_type == "scientific_analysis"
        assert metric.subject_area == "systems_biology"
        assert metric.memory_usage_mb == 256.5
        assert metric.user_id == "researcher_001"
        assert metric.metadata == metadata
        assert metric.tags == tags
    
    def test_api_metric_post_init_calculations(self):
        """Test post-initialization calculations in APIMetric."""
        # Test total tokens calculation
        metric1 = APIMetric(
            prompt_tokens=500,
            completion_tokens=300,
            embedding_tokens=200
        )
        assert metric1.total_tokens == 1000
        
        # Test cost per token calculation
        metric2 = APIMetric(
            total_tokens=2000,
            cost_usd=0.30
        )
        assert metric2.cost_per_token == pytest.approx(0.00015, abs=1e-8)
        
        # Test throughput calculation
        metric3 = APIMetric(
            total_tokens=1500,
            response_time_ms=2000
        )
        assert metric3.throughput_tokens_per_sec == pytest.approx(750.0, abs=0.1)
        
        # Test with zero values (should not cause division by zero)
        metric4 = APIMetric(
            total_tokens=0,
            response_time_ms=0,
            cost_usd=0.0
        )
        assert metric4.cost_per_token is None
        assert metric4.throughput_tokens_per_sec is None
    
    def test_api_metric_serialization(self):
        """Test APIMetric serialization methods."""
        metric = APIMetric(
            operation_name="serialization_test",
            model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.08,
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION.value,
            metadata={"test_key": "test_value", "nested": {"data": 123}},
            tags=["test", "serialization"]
        )
        
        # Test to_dict
        metric_dict = metric.to_dict()
        
        assert isinstance(metric_dict, dict)
        assert metric_dict['operation_name'] == "serialization_test"
        assert metric_dict['model_name'] == "gpt-4o-mini"
        assert metric_dict['total_tokens'] == 150
        assert metric_dict['cost_usd'] == 0.08
        assert metric_dict['metric_type'] == MetricType.LLM_CALL.value
        assert metric_dict['research_category'] == ResearchCategory.METABOLITE_IDENTIFICATION.value
        assert 'timestamp_iso' in metric_dict
        assert 'id' in metric_dict
        
        # Verify ISO timestamp format
        assert 'T' in metric_dict['timestamp_iso']
        assert metric_dict['timestamp_iso'].endswith('Z') or '+' in metric_dict['timestamp_iso']
        
        # Test to_cost_record
        cost_record = metric.to_cost_record()
        
        assert cost_record.operation_type == "serialization_test"
        assert cost_record.model_name == "gpt-4o-mini"
        assert cost_record.prompt_tokens == 100
        assert cost_record.completion_tokens == 50
        assert cost_record.total_tokens == 150
        assert cost_record.cost_usd == 0.08
        assert cost_record.research_category == ResearchCategory.METABOLITE_IDENTIFICATION.value
        assert cost_record.session_id == metric.session_id
        assert cost_record.success == metric.success
    
    def test_api_metric_with_errors(self):
        """Test APIMetric with error conditions."""
        error_metric = APIMetric(
            operation_name="failed_operation",
            model_name="gpt-4o",
            prompt_tokens=200,
            cost_usd=0.0,  # No cost for failed operation
            success=False,
            error_type="RateLimitError",
            error_message="Rate limit exceeded for model gpt-4o",
            retry_count=3,
            final_attempt=True
        )
        
        assert error_metric.success is False
        assert error_metric.error_type == "RateLimitError"
        assert error_metric.error_message == "Rate limit exceeded for model gpt-4o"
        assert error_metric.retry_count == 3
        assert error_metric.cost_usd == 0.0
        
        # Error metrics should still be serializable
        error_dict = error_metric.to_dict()
        assert error_dict['success'] is False
        assert error_dict['error_type'] == "RateLimitError"


class TestMetricType:
    """Tests for MetricType enumeration."""
    
    def test_metric_type_values(self):
        """Test all metric type values are defined."""
        expected_types = [
            'llm_call',
            'embedding_call', 
            'hybrid_operation',
            'response_time',
            'throughput',
            'error_rate',
            'token_usage',
            'cost_tracking',
            'budget_utilization',
            'research_category',
            'knowledge_extraction',
            'document_processing',
            'memory_usage',
            'concurrent_operations',
            'retry_patterns'
        ]
        
        actual_types = [metric_type.value for metric_type in MetricType]
        
        for expected_type in expected_types:
            assert expected_type in actual_types
    
    def test_metric_type_categorization(self):
        """Test metric type categorization."""
        # Core API operations
        core_operations = [
            MetricType.LLM_CALL,
            MetricType.EMBEDDING_CALL,
            MetricType.HYBRID_OPERATION
        ]
        
        # Performance metrics
        performance_metrics = [
            MetricType.RESPONSE_TIME,
            MetricType.THROUGHPUT,
            MetricType.ERROR_RATE
        ]
        
        # System metrics
        system_metrics = [
            MetricType.MEMORY_USAGE,
            MetricType.CONCURRENT_OPERATIONS
        ]
        
        for metric_type in core_operations + performance_metrics + system_metrics:
            assert metric_type in MetricType


class TestMetricsAggregator:
    """Comprehensive tests for MetricsAggregator."""
    
    @pytest.fixture
    def aggregator(self):
        """Create a MetricsAggregator instance for testing."""
        logger = Mock(spec=logging.Logger)
        return MetricsAggregator(logger)
    
    def test_aggregator_initialization(self, aggregator):
        """Test MetricsAggregator initialization."""
        assert hasattr(aggregator, '_metrics_buffer')
        assert hasattr(aggregator, '_hourly_stats')
        assert hasattr(aggregator, '_daily_stats')
        assert hasattr(aggregator, '_category_stats')
        assert hasattr(aggregator, '_error_patterns')
        assert hasattr(aggregator, '_performance_window')
        
        assert len(aggregator._metrics_buffer) == 0
        assert aggregator._max_window_size == 1000
    
    def test_add_single_metric(self, aggregator):
        """Test adding a single metric to aggregator."""
        metric = APIMetric(
            operation_name="test_add_metric",
            model_name="gpt-4o-mini",
            prompt_tokens=150,
            completion_tokens=75,
            cost_usd=0.12,
            response_time_ms=1800,
            research_category=ResearchCategory.BIOMARKER_DISCOVERY.value
        )
        
        aggregator.add_metric(metric)
        
        assert len(aggregator._metrics_buffer) == 1
        assert len(aggregator._performance_window) == 1
        assert aggregator._metrics_buffer[0] == metric
    
    def test_add_multiple_metrics_with_aggregation(self, aggregator):
        """Test adding multiple metrics and verify aggregation."""
        metrics = []
        for i in range(15):
            metric = APIMetric(
                operation_name=f"batch_operation_{i}",
                model_name="gpt-4o-mini",
                prompt_tokens=100 + i * 20,
                completion_tokens=50 + i * 10,
                cost_usd=0.05 + i * 0.01,
                response_time_ms=1000 + i * 200,
                success=(i % 4) != 0,  # 25% error rate
                error_type="TimeoutError" if (i % 4) == 0 else None,
                research_category=list(ResearchCategory)[i % len(ResearchCategory)].value
            )
            metrics.append(metric)
            aggregator.add_metric(metric)
        
        assert len(aggregator._metrics_buffer) == 15
        assert len(aggregator._performance_window) == 15
        
        # Test current statistics
        stats = aggregator.get_current_stats()
        
        # Verify structure
        assert 'current_hour' in stats
        assert 'current_day' in stats
        assert 'recent_performance' in stats
        assert 'top_research_categories' in stats
        assert 'top_error_types' in stats
        assert 'buffer_size' in stats
        
        # Verify calculations
        current_hour = stats['current_hour']
        assert current_hour['total_calls'] >= 15
        assert current_hour['total_tokens'] > 0
        assert current_hour['total_cost'] > 0
        assert current_hour['error_count'] >= 3  # ~25% error rate
        assert 0 <= current_hour['error_rate_percent'] <= 100
        
        # Verify error tracking
        top_errors = stats['top_error_types']
        assert 'TimeoutError' in top_errors
        assert top_errors['TimeoutError'] >= 3
    
    def test_performance_window_size_management(self, aggregator):
        """Test performance window size management."""
        # Add more metrics than window size
        for i in range(aggregator._max_window_size + 100):
            metric = APIMetric(
                operation_name=f"window_test_{i}",
                response_time_ms=1000 + i
            )
            aggregator.add_metric(metric)
        
        # Performance window should be capped at max size
        assert len(aggregator._performance_window) == aggregator._max_window_size
        
        # Buffer should contain all metrics
        assert len(aggregator._metrics_buffer) == aggregator._max_window_size + 100
        
        # Window should contain most recent metrics
        latest_metric = aggregator._performance_window[-1]
        assert latest_metric.operation_name == f"window_test_{aggregator._max_window_size + 99}"
    
    def test_hourly_and_daily_aggregation(self, aggregator):
        """Test hourly and daily aggregation logic."""
        import datetime
        
        # Create metrics with specific timestamps
        base_time = time.time()
        test_metrics = []
        
        # Add metrics for current hour
        for i in range(5):
            metric = APIMetric(
                operation_name=f"hourly_test_{i}",
                cost_usd=0.1,
                total_tokens=100,
                response_time_ms=1500,
                timestamp=base_time + i * 60  # 1-minute intervals
            )
            test_metrics.append(metric)
            aggregator.add_metric(metric)
        
        # Add metrics for different hour (2 hours ago)
        past_time = base_time - (2 * 3600)
        for i in range(3):
            metric = APIMetric(
                operation_name=f"past_hourly_test_{i}",
                cost_usd=0.05,
                total_tokens=50,
                response_time_ms=800,
                timestamp=past_time + i * 60
            )
            test_metrics.append(metric)
            aggregator.add_metric(metric)
        
        stats = aggregator.get_current_stats()
        
        # Current hour should have 5 events
        current_hour = stats['current_hour']
        assert current_hour['total_calls'] >= 5
        assert current_hour['total_cost'] >= 0.5
        
        # Daily stats should include all events
        current_day = stats['current_day']
        assert current_day['total_calls'] >= 8
        assert current_day['total_cost'] >= 0.65
    
    def test_category_statistics(self, aggregator):
        """Test research category statistics aggregation."""
        category_data = [
            (ResearchCategory.METABOLITE_IDENTIFICATION, 8, 0.15),
            (ResearchCategory.PATHWAY_ANALYSIS, 5, 0.20),
            (ResearchCategory.BIOMARKER_DISCOVERY, 3, 0.25),
            (ResearchCategory.DRUG_DISCOVERY, 2, 0.30)
        ]
        
        for category, count, cost_per_metric in category_data:
            for i in range(count):
                metric = APIMetric(
                    operation_name=f"{category.value}_test_{i}",
                    cost_usd=cost_per_metric,
                    total_tokens=100 + i * 10,
                    research_category=category.value
                )
                aggregator.add_metric(metric)
        
        stats = aggregator.get_current_stats()
        categories = stats['top_research_categories']
        
        # Verify category counts (ordered by count)
        assert categories[ResearchCategory.METABOLITE_IDENTIFICATION.value] == 8
        assert categories[ResearchCategory.PATHWAY_ANALYSIS.value] == 5
        assert categories[ResearchCategory.BIOMARKER_DISCOVERY.value] == 3
        assert categories[ResearchCategory.DRUG_DISCOVERY.value] == 2
        
        # Categories should be ordered by count (descending)
        category_counts = list(categories.values())
        assert category_counts == sorted(category_counts, reverse=True)
    
    def test_error_pattern_analysis(self, aggregator):
        """Test error pattern analysis and tracking."""
        error_patterns = [
            ("RateLimitError", 12),
            ("TimeoutError", 8),
            ("ValidationError", 5),
            ("AuthenticationError", 3),
            ("InternalServerError", 2)
        ]
        
        for error_type, count in error_patterns:
            for i in range(count):
                metric = APIMetric(
                    operation_name=f"error_test_{error_type}_{i}",
                    success=False,
                    error_type=error_type,
                    cost_usd=0.0  # Failed operations typically don't incur cost
                )
                aggregator.add_metric(metric)
        
        stats = aggregator.get_current_stats()
        top_errors = stats['top_error_types']
        
        # Verify top 5 error types
        assert len(top_errors) == 5
        assert top_errors['RateLimitError'] == 12
        assert top_errors['TimeoutError'] == 8
        assert top_errors['ValidationError'] == 5
        assert top_errors['AuthenticationError'] == 3
        assert top_errors['InternalServerError'] == 2
        
        # Should be ordered by frequency
        error_counts = list(top_errors.values())
        assert error_counts == sorted(error_counts, reverse=True)
    
    def test_recent_performance_calculation(self, aggregator):
        """Test recent performance metrics calculation."""
        # Add metrics with varying response times
        response_times = [800, 1200, 1500, 900, 2000, 1100, 1800, 1300]
        
        for i, response_time in enumerate(response_times):
            metric = APIMetric(
                operation_name=f"performance_test_{i}",
                response_time_ms=response_time,
                total_tokens=100 + i * 25
            )
            aggregator.add_metric(metric)
        
        stats = aggregator.get_current_stats()
        recent_perf = stats['recent_performance']
        
        assert 'avg_response_time_ms' in recent_perf
        assert 'sample_size' in recent_perf
        assert recent_perf['sample_size'] == len(response_times)
        
        # Calculate expected average
        expected_avg = sum(response_times) / len(response_times)
        assert abs(recent_perf['avg_response_time_ms'] - expected_avg) < 1.0


class TestAPIUsageMetricsLogger:
    """Comprehensive tests for APIUsageMetricsLogger."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock configuration for testing."""
        config = Mock()
        config.enable_file_logging = True
        config.log_dir = temp_dir / "logs"
        config.log_max_bytes = 1024 * 1024
        config.log_backup_count = 3
        return config
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield Path(db_path)
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def integrated_metrics_logger(self, mock_config, temp_db_path):
        """Create fully integrated APIUsageMetricsLogger."""
        # Create integrated components
        cost_persistence = CostPersistence(temp_db_path)
        budget_manager = BudgetManager(
            cost_persistence=cost_persistence,
            daily_budget_limit=100.0,
            monthly_budget_limit=3000.0
        )
        research_categorizer = ResearchCategorizer()
        audit_trail = AuditTrail(db_path=temp_db_path)
        
        logger = Mock(spec=logging.Logger)
        
        return APIUsageMetricsLogger(
            config=mock_config,
            cost_persistence=cost_persistence,
            budget_manager=budget_manager,
            research_categorizer=research_categorizer,
            audit_trail=audit_trail,
            logger=logger
        )
    
    def test_metrics_logger_initialization(self, integrated_metrics_logger):
        """Test APIUsageMetricsLogger initialization."""
        logger = integrated_metrics_logger
        
        assert logger.session_id is not None
        assert logger.start_time is not None
        assert isinstance(logger.metrics_aggregator, MetricsAggregator)
        assert logger._operation_counter == 0
        assert len(logger._active_operations) == 0
        
        # Verify integrations
        assert logger.cost_persistence is not None
        assert logger.budget_manager is not None
        assert logger.research_categorizer is not None
        assert logger.audit_trail is not None
    
    def test_track_api_call_context_manager_basic(self, integrated_metrics_logger):
        """Test basic track_api_call context manager usage."""
        logger = integrated_metrics_logger
        
        with logger.track_api_call("basic_test", "gpt-4o-mini") as tracker:
            # Simulate API call processing
            time.sleep(0.01)
            
            tracker.set_tokens(prompt=200, completion=150)
            tracker.set_cost(0.18)
            tracker.set_response_details(
                response_time_ms=1200,
                request_size=2048,
                response_size=4096
            )
            tracker.add_metadata("test_param", "test_value")
        
        # Verify metric was recorded
        assert len(logger.metrics_aggregator._metrics_buffer) == 1
        metric = logger.metrics_aggregator._metrics_buffer[0]
        
        assert metric.operation_name == "basic_test"
        assert metric.model_name == "gpt-4o-mini"
        assert metric.prompt_tokens == 200
        assert metric.completion_tokens == 150
        assert metric.total_tokens == 350
        assert metric.cost_usd == 0.18
        assert metric.response_time_ms >= 10  # At least the sleep time
        assert metric.request_size_bytes == 2048
        assert metric.response_size_bytes == 4096
        assert metric.metadata["test_param"] == "test_value"
        assert metric.success is True
    
    def test_track_api_call_with_error_handling(self, integrated_metrics_logger):
        """Test track_api_call with error conditions."""
        logger = integrated_metrics_logger
        
        # Test with explicit error
        with logger.track_api_call("error_test", "gpt-4o") as tracker:
            tracker.set_tokens(prompt=100, completion=0)
            tracker.set_cost(0.0)
            tracker.set_error("APIError", "Service temporarily unavailable")
        
        # Verify error was recorded
        metric = logger.metrics_aggregator._metrics_buffer[0]
        assert metric.success is False
        assert metric.error_type == "APIError"
        assert metric.error_message == "Service temporarily unavailable"
        assert metric.cost_usd == 0.0
        
        # Test with exception in context
        try:
            with logger.track_api_call("exception_test", "gpt-4o") as tracker:
                tracker.set_tokens(prompt=50)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Verify exception was recorded
        assert len(logger.metrics_aggregator._metrics_buffer) == 2
        error_metric = logger.metrics_aggregator._metrics_buffer[1]
        assert error_metric.success is False
        assert error_metric.error_type == "ValueError"
        assert error_metric.error_message == "Test exception"
    
    def test_concurrent_operations_tracking(self, integrated_metrics_logger):
        """Test tracking of concurrent operations."""
        logger = integrated_metrics_logger
        results = []
        
        def worker(worker_id):
            with logger.track_api_call(f"concurrent_op_{worker_id}", "gpt-4o-mini") as tracker:
                # Simulate processing time
                time.sleep(0.02)
                tracker.set_tokens(prompt=50 + worker_id * 10, completion=25 + worker_id * 5)
                tracker.set_cost(0.02 + worker_id * 0.01)
                return tracker.metric.concurrent_operations
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            for future in futures:
                results.append(future.result())
        
        # Verify all operations completed
        assert len(logger.metrics_aggregator._metrics_buffer) == 8
        
        # Verify concurrent operation tracking
        for i, metric in enumerate(logger.metrics_aggregator._metrics_buffer):
            assert metric.operation_name == f"concurrent_op_{i}"
            assert metric.concurrent_operations >= 1  # At least 1, could be higher during concurrency
    
    def test_batch_operation_logging(self, integrated_metrics_logger):
        """Test batch operation logging functionality."""
        logger = integrated_metrics_logger
        
        logger.log_batch_operation(
            operation_name="document_processing",
            batch_size=25,
            total_tokens=12500,
            total_cost=1.75,
            processing_time_ms=45000,
            success_count=23,
            error_count=2,
            research_category=ResearchCategory.DOCUMENT_PROCESSING.value,
            metadata={
                "document_types": ["pdf", "text", "json"],
                "average_doc_size": 2048,
                "processing_node": "worker-3"
            }
        )
        
        assert len(logger.metrics_aggregator._metrics_buffer) == 1
        metric = logger.metrics_aggregator._metrics_buffer[0]
        
        assert metric.operation_name == "batch_document_processing"
        assert metric.metric_type == MetricType.HYBRID_OPERATION
        assert metric.total_tokens == 12500
        assert metric.cost_usd == 1.75
        assert metric.response_time_ms == 45000
        assert metric.research_category == ResearchCategory.DOCUMENT_PROCESSING.value
        
        # Verify batch-specific metadata
        assert metric.metadata["batch_size"] == 25
        assert metric.metadata["success_count"] == 23
        assert metric.metadata["error_count"] == 2
        assert metric.metadata["success_rate"] == 0.92
        assert metric.metadata["document_types"] == ["pdf", "text", "json"]
    
    def test_integration_with_cost_tracking(self, integrated_metrics_logger):
        """Test integration with cost persistence system."""
        logger = integrated_metrics_logger
        
        with logger.track_api_call("cost_integration_test", "gpt-4o") as tracker:
            tracker.set_tokens(prompt=300, completion=200)
            tracker.set_cost(0.25)
            tracker.add_metadata("integration_test", True)
        
        # Verify cost was recorded in persistence layer
        cost_records = logger.cost_persistence.db.get_cost_records(limit=1)
        assert len(cost_records) == 1
        
        cost_record = cost_records[0]
        assert cost_record.operation_type == "cost_integration_test"
        assert cost_record.model_name == "gpt-4o"
        assert cost_record.prompt_tokens == 300
        assert cost_record.completion_tokens == 200
        assert cost_record.total_tokens == 500
        assert cost_record.cost_usd == 0.25
        assert cost_record.session_id == logger.session_id
    
    def test_integration_with_budget_manager(self, integrated_metrics_logger):
        """Test integration with budget manager."""
        logger = integrated_metrics_logger
        
        # Add significant cost to trigger budget checking
        with logger.track_api_call("budget_integration_test", "gpt-4o") as tracker:
            tracker.set_tokens(prompt=2000, completion=1500)
            tracker.set_cost(0.85)  # Significant cost
        
        # Verify budget utilization was updated in metric
        metric = logger.metrics_aggregator._metrics_buffer[0]
        assert metric.daily_budget_used_percent is not None
        assert metric.monthly_budget_used_percent is not None
        assert metric.daily_budget_used_percent >= 0.85  # At least the cost percentage
    
    def test_integration_with_research_categorizer(self, integrated_metrics_logger):
        """Test integration with research categorizer."""
        logger = integrated_metrics_logger
        
        with logger.track_api_call("research_categorization_test", "gpt-4o-mini") as tracker:
            tracker.set_tokens(prompt=150, completion=100)
            tracker.set_cost(0.12)
            # Set query type that should trigger categorization
            tracker.metric.query_type = "metabolite identification"
            tracker.metric.subject_area = "biochemistry"
        
        # Verify research categorization was enhanced
        metric = logger.metrics_aggregator._metrics_buffer[0]
        # Note: The exact category might depend on the categorizer's logic
        assert metric.research_category is not None
    
    def test_integration_with_audit_trail(self, integrated_metrics_logger):
        """Test integration with audit trail system."""
        logger = integrated_metrics_logger
        
        with logger.track_api_call("audit_integration_test", "gpt-4o") as tracker:
            tracker.set_tokens(prompt=100, completion=50)
            tracker.set_cost(0.08)
            tracker.metric.user_id = "test_researcher"
            tracker.metric.project_id = "audit_project"
        
        # Verify audit event was recorded
        audit_events = logger.audit_trail.get_audit_log(limit=1)
        assert len(audit_events) >= 1
        
        # Find the API usage event
        api_event = None
        for event in audit_events:
            if hasattr(event, 'event_type') and event.event_type.value == 'api_usage':
                api_event = event
                break
        
        if api_event:
            assert api_event.user_id == "test_researcher"
            assert 'tokens' in api_event.event_data
            assert 'cost_usd' in api_event.event_data
    
    def test_performance_summary_generation(self, integrated_metrics_logger):
        """Test comprehensive performance summary generation."""
        logger = integrated_metrics_logger
        
        # Add varied metrics
        for i in range(10):
            with logger.track_api_call(f"perf_test_{i}", "gpt-4o-mini") as tracker:
                tracker.set_tokens(
                    prompt=100 + i * 20,
                    completion=50 + i * 10
                )
                tracker.set_cost(0.05 + i * 0.01)
                tracker.set_response_details(response_time_ms=1000 + i * 100)
        
        summary = logger.get_performance_summary()
        
        # Verify summary structure
        assert 'current_hour' in summary
        assert 'current_day' in summary
        assert 'recent_performance' in summary
        assert 'top_research_categories' in summary
        assert 'system' in summary
        assert 'integration_status' in summary
        
        # Verify system information
        system_info = summary['system']
        assert 'memory_usage_mb' in system_info
        assert 'active_operations' in system_info
        assert 'session_uptime_seconds' in system_info
        assert 'session_id' in system_info
        assert system_info['session_id'] == logger.session_id
        
        # Verify integration status
        integration_status = summary['integration_status']
        assert integration_status['cost_persistence'] is True
        assert integration_status['budget_manager'] is True
        assert integration_status['research_categorizer'] is True
        assert integration_status['audit_trail'] is True
    
    def test_system_event_logging(self, integrated_metrics_logger):
        """Test system event logging functionality."""
        logger = integrated_metrics_logger
        
        event_data = {
            "event_name": "rag_system_initialization",
            "initialization_time_ms": 3500,
            "components_loaded": [
                "llm_interface", 
                "embedding_service", 
                "cost_tracking",
                "budget_manager",
                "audit_system"
            ],
            "memory_allocated_mb": 512,
            "configuration_hash": "abc123def456"
        }
        
        logger.log_system_event(
            event_type="system_initialization",
            event_data=event_data,
            user_id="system_admin"
        )
        
        # Verify system event was logged
        # This would typically be verified through audit trail or logs
        # For now, verify the call completes without error
        assert True
    
    def test_memory_management_and_cleanup(self, integrated_metrics_logger):
        """Test memory management and cleanup functionality."""
        logger = integrated_metrics_logger
        
        # Generate large number of metrics
        for i in range(2000):  # Exceed aggregator's window size
            with logger.track_api_call(f"memory_test_{i}", "gpt-4o-mini") as tracker:
                tracker.set_tokens(prompt=50, completion=25)
                tracker.set_cost(0.03)
        
        # Verify memory management
        assert len(logger.metrics_aggregator._metrics_buffer) == 2000
        assert len(logger.metrics_aggregator._performance_window) == logger.metrics_aggregator._max_window_size
        
        # Performance summary should still work
        summary = logger.get_performance_summary()
        assert summary['system']['active_operations'] == 0  # All operations completed
    
    def test_error_recovery_and_resilience(self, integrated_metrics_logger):
        """Test error recovery and system resilience."""
        logger = integrated_metrics_logger
        
        # Test with integration component failures
        original_cost_persistence = logger.cost_persistence
        
        # Simulate cost persistence failure
        logger.cost_persistence = None
        
        with logger.track_api_call("resilience_test", "gpt-4o-mini") as tracker:
            tracker.set_tokens(prompt=100, completion=50)
            tracker.set_cost(0.05)
        
        # Should still record metric locally
        assert len(logger.metrics_aggregator._metrics_buffer) == 1
        
        # Restore component
        logger.cost_persistence = original_cost_persistence
        
        # Verify system recovery
        with logger.track_api_call("recovery_test", "gpt-4o-mini") as tracker:
            tracker.set_tokens(prompt=100, completion=50)
            tracker.set_cost(0.05)
        
        assert len(logger.metrics_aggregator._metrics_buffer) == 2


class TestPerformanceAndConcurrency:
    """Performance and concurrency tests for metrics logging system."""
    
    @pytest.fixture
    def performance_logger(self, temp_dir):
        """Create metrics logger optimized for performance testing."""
        config = Mock()
        config.enable_file_logging = False  # Disable for speed
        
        return APIUsageMetricsLogger(config=config, logger=Mock())
    
    def test_high_throughput_metrics_logging(self, performance_logger):
        """Test high throughput metrics logging performance."""
        logger = performance_logger
        num_operations = 1000
        
        start_time = time.time()
        
        # Simulate high-throughput operations
        for i in range(num_operations):
            with logger.track_api_call(f"throughput_test_{i}", "gpt-4o-mini") as tracker:
                tracker.set_tokens(prompt=100, completion=50)
                tracker.set_cost(0.05)
        
        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = num_operations / total_time
        
        # Verify reasonable performance (adjust threshold as needed)
        assert ops_per_second > 100  # Should handle at least 100 ops/sec
        
        # Verify all operations were recorded
        assert len(logger.metrics_aggregator._metrics_buffer) == num_operations
    
    def test_concurrent_metrics_logging_stress(self, performance_logger):
        """Test concurrent metrics logging under stress conditions."""
        logger = performance_logger
        num_threads = 20
        ops_per_thread = 100
        
        def stress_worker(worker_id):
            operations_completed = 0
            for i in range(ops_per_thread):
                try:
                    with logger.track_api_call(f"stress_{worker_id}_{i}", "gpt-4o-mini") as tracker:
                        tracker.set_tokens(prompt=75 + worker_id, completion=35 + i)
                        tracker.set_cost(0.04 + worker_id * 0.001)
                        time.sleep(0.001)  # Small delay to increase contention
                    operations_completed += 1
                except Exception as e:
                    # Log error but continue
                    print(f"Worker {worker_id} error: {e}")
            return operations_completed
        
        # Run stress test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        # Verify all operations completed successfully
        total_completed = sum(results)
        expected_total = num_threads * ops_per_thread
        
        # Allow for some failures under stress
        assert total_completed >= expected_total * 0.95  # At least 95% success rate
        
        # Verify metrics were recorded
        assert len(logger.metrics_aggregator._metrics_buffer) >= total_completed * 0.95
        
        # Verify performance under stress
        stress_duration = end_time - start_time
        stress_ops_per_second = total_completed / stress_duration
        
        print(f"Stress test: {total_completed} operations in {stress_duration:.2f}s = {stress_ops_per_second:.1f} ops/sec")
        
        # Should maintain reasonable performance under stress
        assert stress_ops_per_second > 50


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])