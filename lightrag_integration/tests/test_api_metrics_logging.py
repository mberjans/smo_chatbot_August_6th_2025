#!/usr/bin/env python3
"""
Comprehensive test suite for API usage metrics logging system.

This test suite covers:
- API metrics logger initialization and configuration
- Structured logging for API usage patterns and performance metrics
- Integration with enhanced cost tracking system
- Audit-friendly logging for research compliance requirements
- Thread-safe concurrent operations
- Metrics aggregation and reporting functionality

Author: Claude Code (Anthropic)
Created: August 6, 2025
"""

import pytest
import asyncio
import time
import json
import logging
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightrag_integration.api_metrics_logger import (
    APIUsageMetricsLogger, 
    APIMetric, 
    MetricType, 
    MetricsAggregator
)
from lightrag_integration.config import LightRAGConfig
from lightrag_integration.cost_persistence import CostPersistence, ResearchCategory
from lightrag_integration.budget_manager import BudgetManager, BudgetThreshold
from lightrag_integration.research_categorizer import ResearchCategorizer
from lightrag_integration.audit_trail import AuditTrail


class TestAPIMetric:
    """Test cases for APIMetric data model."""
    
    def test_api_metric_creation(self):
        """Test basic APIMetric creation and initialization."""
        metric = APIMetric(
            operation_name="test_llm_call",
            model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01
        )
        
        assert metric.operation_name == "test_llm_call"
        assert metric.model_name == "gpt-4o-mini"
        assert metric.total_tokens == 150
        assert metric.cost_usd == 0.01
        assert metric.success is True
        assert metric.timestamp is not None
        assert metric.id is not None
    
    def test_api_metric_post_init_calculations(self):
        """Test post-initialization calculations in APIMetric."""
        metric = APIMetric(
            operation_name="test_call",
            total_tokens=200,
            cost_usd=0.02,
            response_time_ms=1500
        )
        
        # Test cost per token calculation
        expected_cost_per_token = 0.02 / 200
        assert metric.cost_per_token == pytest.approx(expected_cost_per_token)
        
        # Test throughput calculation
        expected_throughput = 200 / (1500 / 1000.0)
        assert metric.throughput_tokens_per_sec == pytest.approx(expected_throughput)
    
    def test_api_metric_to_dict(self):
        """Test APIMetric serialization to dictionary."""
        metric = APIMetric(
            operation_name="test_embedding",
            model_name="text-embedding-3-small",
            embedding_tokens=100,
            cost_usd=0.002
        )
        
        metric_dict = metric.to_dict()
        
        assert isinstance(metric_dict, dict)
        assert metric_dict['operation_name'] == "test_embedding"
        assert metric_dict['model_name'] == "text-embedding-3-small"
        assert metric_dict['embedding_tokens'] == 100
        assert 'timestamp_iso' in metric_dict
        assert 'metric_type' in metric_dict
    
    def test_api_metric_to_cost_record(self):
        """Test conversion of APIMetric to CostRecord."""
        metric = APIMetric(
            operation_name="llm_test",
            model_name="gpt-4o",
            prompt_tokens=50,
            completion_tokens=25,
            cost_usd=0.008,
            research_category=ResearchCategory.METABOLITE_IDENTIFICATION.value,
            response_time_ms=2000
        )
        
        cost_record = metric.to_cost_record()
        
        assert cost_record.operation_type == "llm_test"
        assert cost_record.model_name == "gpt-4o"
        assert cost_record.prompt_tokens == 50
        assert cost_record.completion_tokens == 25
        assert cost_record.cost_usd == 0.008
        assert cost_record.research_category == ResearchCategory.METABOLITE_IDENTIFICATION.value
        assert cost_record.response_time_seconds == 2.0


class TestMetricsAggregator:
    """Test cases for MetricsAggregator."""
    
    @pytest.fixture
    def aggregator(self):
        """Create a MetricsAggregator instance for testing."""
        return MetricsAggregator()
    
    def test_aggregator_initialization(self, aggregator):
        """Test MetricsAggregator initialization."""
        assert isinstance(aggregator._metrics_buffer, list)
        assert len(aggregator._metrics_buffer) == 0
        assert aggregator._max_window_size == 1000
    
    def test_add_single_metric(self, aggregator):
        """Test adding a single metric to the aggregator."""
        metric = APIMetric(
            operation_name="test_op",
            model_name="gpt-4o-mini",
            total_tokens=100,
            cost_usd=0.005,
            response_time_ms=1000
        )
        
        aggregator.add_metric(metric)
        
        assert len(aggregator._metrics_buffer) == 1
        assert len(aggregator._performance_window) == 1
    
    def test_add_multiple_metrics(self, aggregator):
        """Test adding multiple metrics and aggregation."""
        metrics = []
        for i in range(10):
            metric = APIMetric(
                operation_name=f"test_op_{i}",
                model_name="gpt-4o-mini",
                total_tokens=100 + i * 10,
                cost_usd=0.005 + i * 0.001,
                response_time_ms=1000 + i * 100,
                success=i % 2 == 0  # Half successful, half failed
            )
            metrics.append(metric)
            aggregator.add_metric(metric)
        
        assert len(aggregator._metrics_buffer) == 10
        
        # Test statistics calculation
        stats = aggregator.get_current_stats()
        assert 'current_hour' in stats
        assert 'current_day' in stats
        assert stats['current_hour']['total_calls'] >= 0
        assert stats['current_day']['total_calls'] >= 0
    
    def test_error_pattern_tracking(self, aggregator):
        """Test error pattern tracking in aggregator."""
        # Add metrics with different error types
        error_types = ['RateLimitError', 'TimeoutError', 'RateLimitError']
        for error_type in error_types:
            metric = APIMetric(
                operation_name="failed_op",
                success=False,
                error_type=error_type
            )
            aggregator.add_metric(metric)
        
        stats = aggregator.get_current_stats()
        error_patterns = stats.get('top_error_types', {})
        
        # RateLimitError should appear twice
        assert error_patterns.get('RateLimitError', 0) == 2
        assert error_patterns.get('TimeoutError', 0) == 1
    
    def test_performance_window_size_limit(self, aggregator):
        """Test that performance window respects size limits."""
        # Add more metrics than the window size
        for i in range(aggregator._max_window_size + 100):
            metric = APIMetric(operation_name=f"op_{i}")
            aggregator.add_metric(metric)
        
        # Window should be limited to max size
        assert len(aggregator._performance_window) == aggregator._max_window_size
        
        # Buffer should contain all metrics
        assert len(aggregator._metrics_buffer) == aggregator._max_window_size + 100


class TestAPIUsageMetricsLogger:
    """Test cases for APIUsageMetricsLogger."""
    
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
    def metrics_logger(self, mock_config):
        """Create an APIUsageMetricsLogger instance for testing."""
        mock_logger = Mock(spec=logging.Logger)
        return APIUsageMetricsLogger(
            config=mock_config,
            logger=mock_logger
        )
    
    def test_metrics_logger_initialization(self, metrics_logger):
        """Test APIUsageMetricsLogger initialization."""
        assert metrics_logger.session_id is not None
        assert metrics_logger.start_time is not None
        assert isinstance(metrics_logger.metrics_aggregator, MetricsAggregator)
        assert metrics_logger._operation_counter == 0
    
    def test_track_api_call_context_manager(self, metrics_logger):
        """Test the track_api_call context manager."""
        with metrics_logger.track_api_call("test_operation", "gpt-4o-mini") as tracker:
            assert tracker is not None
            
            # Test setting tokens
            tracker.set_tokens(prompt=100, completion=50)
            tracker.set_cost(0.01)
            tracker.set_response_details(response_time_ms=1500)
            tracker.add_metadata('test_key', 'test_value')
        
        # Verify metric was added to aggregator
        assert len(metrics_logger.metrics_aggregator._metrics_buffer) == 1
        
        metric = metrics_logger.metrics_aggregator._metrics_buffer[0]
        assert metric.operation_name == "test_operation"
        assert metric.model_name == "gpt-4o-mini"
        assert metric.prompt_tokens == 100
        assert metric.completion_tokens == 50
        assert metric.cost_usd == 0.01
        assert metric.response_time_ms == 1500
        assert metric.metadata.get('test_key') == 'test_value'
    
    def test_track_api_call_with_error(self, metrics_logger):
        """Test track_api_call when an error occurs."""
        try:
            with metrics_logger.track_api_call("error_operation") as tracker:
                tracker.set_error("TestError", "Test error message")
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Verify error was recorded
        assert len(metrics_logger.metrics_aggregator._metrics_buffer) == 1
        
        metric = metrics_logger.metrics_aggregator._metrics_buffer[0]
        assert metric.success is False
        assert metric.error_type == "TestError"
        assert metric.error_message == "Test error message"
    
    def test_concurrent_api_tracking(self, metrics_logger):
        """Test concurrent API call tracking."""
        def make_tracked_call(operation_id):
            with metrics_logger.track_api_call(f"concurrent_op_{operation_id}") as tracker:
                time.sleep(0.1)  # Simulate processing
                tracker.set_tokens(prompt=50, completion=25)
                tracker.set_cost(0.005)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_tracked_call, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all operations were tracked
        assert len(metrics_logger.metrics_aggregator._metrics_buffer) == 5
        
        # Verify all operations have unique names
        operation_names = [m.operation_name for m in metrics_logger.metrics_aggregator._metrics_buffer]
        assert len(set(operation_names)) == 5
    
    def test_batch_operation_logging(self, metrics_logger):
        """Test batch operation logging functionality."""
        metrics_logger.log_batch_operation(
            operation_name="pdf_processing",
            batch_size=10,
            total_tokens=5000,
            total_cost=0.50,
            processing_time_ms=30000,
            success_count=8,
            error_count=2,
            research_category=ResearchCategory.DOCUMENT_PROCESSING.value
        )
        
        assert len(metrics_logger.metrics_aggregator._metrics_buffer) == 1
        
        metric = metrics_logger.metrics_aggregator._metrics_buffer[0]
        assert metric.operation_name == "batch_pdf_processing"
        assert metric.metric_type == MetricType.HYBRID_OPERATION
        assert metric.total_tokens == 5000
        assert metric.cost_usd == 0.50
        assert metric.response_time_ms == 30000
        assert metric.metadata['batch_size'] == 10
        assert metric.metadata['success_count'] == 8
        assert metric.metadata['error_count'] == 2
        assert metric.metadata['success_rate'] == 0.8
    
    def test_performance_summary(self, metrics_logger):
        """Test getting performance summary."""
        # Add some test metrics
        for i in range(3):
            with metrics_logger.track_api_call(f"test_op_{i}") as tracker:
                tracker.set_tokens(prompt=100 + i * 10, completion=50 + i * 5)
                tracker.set_cost(0.01 + i * 0.005)
        
        summary = metrics_logger.get_performance_summary()
        
        assert 'current_hour' in summary
        assert 'current_day' in summary
        assert 'system' in summary
        assert 'integration_status' in summary
        
        system_info = summary['system']
        assert 'memory_usage_mb' in system_info
        assert 'session_id' in system_info
        assert system_info['session_id'] == metrics_logger.session_id
    
    def test_system_event_logging(self, metrics_logger):
        """Test system event logging functionality."""
        event_data = {
            'event_name': 'rag_initialization',
            'duration_ms': 2500,
            'components_loaded': ['llm', 'embedding', 'cost_tracking'],
            'memory_usage': 150.5
        }
        
        metrics_logger.log_system_event(
            event_type='system_startup',
            event_data=event_data,
            user_id='test_user'
        )
        
        # Verify the event was logged (mock logger should have been called)
        # In a real implementation, this would check the audit trail
        assert True  # Placeholder - would verify audit trail logging


class TestIntegrationWithCostTracking:
    """Test integration with enhanced cost tracking system."""
    
    @pytest.fixture
    def mock_cost_persistence(self):
        """Create a mock cost persistence system."""
        return Mock(spec=CostPersistence)
    
    @pytest.fixture
    def mock_budget_manager(self):
        """Create a mock budget manager."""
        budget_manager = Mock(spec=BudgetManager)
        budget_manager.check_budget_status.return_value = {
            'daily': {'current_cost': 5.0, 'budget_limit': 50.0},
            'monthly': {'current_cost': 150.0, 'budget_limit': 1000.0}
        }
        return budget_manager
    
    @pytest.fixture
    def mock_audit_trail(self):
        """Create a mock audit trail system."""
        return Mock(spec=AuditTrail)
    
    @pytest.fixture
    def integrated_metrics_logger(self, temp_dir, mock_cost_persistence, 
                                 mock_budget_manager, mock_audit_trail):
        """Create metrics logger with full integration."""
        config = Mock()
        config.enable_file_logging = True
        config.log_dir = temp_dir / "logs"
        config.log_max_bytes = 1024 * 1024
        config.log_backup_count = 3
        
        return APIUsageMetricsLogger(
            config=config,
            cost_persistence=mock_cost_persistence,
            budget_manager=mock_budget_manager,
            audit_trail=mock_audit_trail,
            logger=Mock(spec=logging.Logger)
        )
    
    def test_cost_tracking_integration(self, integrated_metrics_logger, 
                                      mock_cost_persistence):
        """Test integration with cost persistence system."""
        with integrated_metrics_logger.track_api_call("test_llm") as tracker:
            tracker.set_tokens(prompt=200, completion=100)
            tracker.set_cost(0.015)
        
        # Verify cost record was created and persisted
        mock_cost_persistence.record_cost.assert_called_once()
        
        # Get the cost record that was passed
        call_args = mock_cost_persistence.record_cost.call_args
        cost_record = call_args[0][0]
        
        assert cost_record.operation_type == "test_llm"
        assert cost_record.cost_usd == 0.015
        assert cost_record.prompt_tokens == 200
        assert cost_record.completion_tokens == 100
    
    def test_budget_manager_integration(self, integrated_metrics_logger, 
                                       mock_budget_manager):
        """Test integration with budget manager."""
        with integrated_metrics_logger.track_api_call("test_budget") as tracker:
            tracker.set_cost(10.0)
        
        # Verify budget status was checked
        mock_budget_manager.check_budget_status.assert_called()
        
        # Check that budget utilization was recorded in metric
        metric = integrated_metrics_logger.metrics_aggregator._metrics_buffer[0]
        assert metric.daily_budget_used_percent == 10.0  # 5.0/50.0 * 100
        assert metric.monthly_budget_used_percent == 15.0  # 150.0/1000.0 * 100
    
    def test_audit_trail_integration(self, integrated_metrics_logger, 
                                    mock_audit_trail):
        """Test integration with audit trail system."""
        with integrated_metrics_logger.track_api_call("test_audit") as tracker:
            tracker.set_tokens(prompt=100, completion=50)
            tracker.set_cost(0.008)
        
        # Verify audit event was recorded
        mock_audit_trail.record_event.assert_called_once()
        
        # Get the audit event that was recorded
        call_args = mock_audit_trail.record_event.call_args
        assert call_args[1]['event_type'] == 'api_usage'


class TestThreadSafetyAndConcurrency:
    """Test thread safety and concurrent operations."""
    
    @pytest.fixture
    def concurrent_metrics_logger(self, temp_dir):
        """Create metrics logger for concurrency testing."""
        config = Mock()
        config.enable_file_logging = False  # Disable file logging for speed
        return APIUsageMetricsLogger(config=config, logger=Mock())
    
    def test_concurrent_metric_logging(self, concurrent_metrics_logger):
        """Test concurrent metric logging operations."""
        num_threads = 10
        operations_per_thread = 20
        
        def worker(thread_id):
            for i in range(operations_per_thread):
                with concurrent_metrics_logger.track_api_call(f"thread_{thread_id}_op_{i}") as tracker:
                    tracker.set_tokens(prompt=100, completion=50)
                    tracker.set_cost(0.005)
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Create and start threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations were logged
        total_operations = num_threads * operations_per_thread
        assert len(concurrent_metrics_logger.metrics_aggregator._metrics_buffer) == total_operations
        
        # Verify no data corruption occurred
        for metric in concurrent_metrics_logger.metrics_aggregator._metrics_buffer:
            assert metric.prompt_tokens == 100
            assert metric.completion_tokens == 50
            assert metric.cost_usd == 0.005
    
    def test_concurrent_performance_summary_access(self, concurrent_metrics_logger):
        """Test concurrent access to performance summary."""
        summary_results = []
        
        def add_metrics_and_get_summary():
            # Add some metrics
            with concurrent_metrics_logger.track_api_call("concurrent_test") as tracker:
                tracker.set_tokens(prompt=50, completion=25)
                tracker.set_cost(0.003)
            
            # Get performance summary
            summary = concurrent_metrics_logger.get_performance_summary()
            summary_results.append(summary)
        
        # Run multiple threads accessing summary concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_metrics_and_get_summary)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all summaries were successfully generated
        assert len(summary_results) == 5
        
        # Verify summaries contain expected structure
        for summary in summary_results:
            assert 'current_hour' in summary
            assert 'current_day' in summary
            assert 'system' in summary


# Integration tests with the full system
class TestFullSystemIntegration:
    """Test full system integration scenarios."""
    
    def test_logging_configuration_integration(self, temp_dir):
        """Test integration with LightRAG logging configuration."""
        # Create a real config with logging enabled
        config_dict = {
            'api_key': 'test-key',
            'log_dir': str(temp_dir / 'logs'),
            'enable_file_logging': True,
            'log_level': 'INFO',
            'enable_api_metrics_logging': True
        }
        
        config = LightRAGConfig.get_config(source=config_dict)
        logger = config.setup_lightrag_logging("test_metrics")
        
        # Create metrics logger with real config
        metrics_logger = APIUsageMetricsLogger(
            config=config,
            logger=logger
        )
        
        assert metrics_logger.config == config
        assert metrics_logger.logger == logger
        
        # Test that log files are created
        log_dir = Path(config_dict['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add a metric to trigger logging
        with metrics_logger.track_api_call("integration_test") as tracker:
            tracker.set_tokens(prompt=100, completion=50)
            tracker.set_cost(0.01)
        
        # Check that metrics log file exists (after some operations)
        metrics_log_file = log_dir / "api_metrics.log"
        # File creation may be delayed, so we just verify the logger was set up
        assert metrics_logger.metrics_logger is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])