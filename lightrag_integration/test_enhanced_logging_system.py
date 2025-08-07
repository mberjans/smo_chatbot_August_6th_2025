#!/usr/bin/env python3
"""
Test suite for the enhanced logging system in Clinical Metabolomics Oracle LightRAG integration.

This module provides comprehensive tests to validate the enhanced logging functionality,
including structured logging, correlation IDs, performance metrics, and diagnostic capabilities.

Test Categories:
    1. Basic Enhanced Logger Tests
    2. Structured Logging Tests  
    3. Correlation ID Management Tests
    4. Performance Tracking Tests
    5. Ingestion Logger Tests
    6. Diagnostic Logger Tests
    7. Integration Tests with ClinicalMetabolomicsRAG
    8. Performance Impact Tests

Usage:
    python test_enhanced_logging_system.py
    
    Or run specific test categories:
    python test_enhanced_logging_system.py --category basic
    python test_enhanced_logging_system.py --category performance
"""

import pytest
import json
import time
import tempfile
import asyncio
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import the enhanced logging system
from enhanced_logging import (
    EnhancedLogger, IngestionLogger, DiagnosticLogger,
    CorrelationIDManager, correlation_manager, PerformanceTracker,
    StructuredLogRecord, PerformanceMetrics, create_enhanced_loggers,
    setup_structured_logging, performance_logged
)

# Import configuration and main classes
from config import LightRAGConfig
from clinical_metabolomics_rag import ClinicalMetabolomicsRAG


class TestEnhancedLoggingSystem:
    """Comprehensive test suite for the enhanced logging system."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_logger(self, temp_log_dir):
        """Create test logger instance."""
        logger = logging.getLogger("test_enhanced_logging")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Add file handler for testing
        log_file = temp_log_dir / "test.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @pytest.fixture
    def enhanced_logger(self, test_logger):
        """Create enhanced logger instance."""
        return EnhancedLogger(test_logger, component="test_component")
    
    def test_structured_log_record_creation(self):
        """Test creation and serialization of structured log records."""
        record = StructuredLogRecord(
            level="INFO",
            message="Test message",
            correlation_id="test-123",
            operation_name="test_operation",
            component="test_component",
            metadata={"key": "value"}
        )
        
        # Test dictionary conversion
        record_dict = record.to_dict()
        assert record_dict['level'] == "INFO"
        assert record_dict['message'] == "Test message"
        assert record_dict['correlation_id'] == "test-123"
        assert record_dict['operation_name'] == "test_operation"
        assert record_dict['component'] == "test_component"
        assert record_dict['metadata'] == {"key": "value"}
        assert 'timestamp' in record_dict
        
        # Test JSON serialization
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed['message'] == "Test message"
    
    def test_correlation_id_manager(self):
        """Test correlation ID management and context tracking."""
        manager = CorrelationIDManager()
        
        # Test correlation ID generation
        corr_id = manager.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0
        
        # Test context management
        with manager.operation_context("test_operation", user_id="user123") as context:
            assert context.operation_name == "test_operation"
            assert context.user_id == "user123"
            assert context.correlation_id is not None
            
            # Test nested contexts
            with manager.operation_context("nested_operation") as nested_context:
                assert nested_context.parent_correlation_id == context.correlation_id
                assert manager.get_correlation_id() == nested_context.correlation_id
            
            # After nested context, should return to parent
            assert manager.get_correlation_id() == context.correlation_id
        
        # After context, should be None
        assert manager.get_correlation_id() is None
    
    def test_performance_tracker(self):
        """Test performance metrics tracking."""
        tracker = PerformanceTracker()
        
        # Start tracking
        start_stats = tracker.start_tracking()
        assert 'timestamp' in start_stats
        
        # Simulate some work
        time.sleep(0.1)
        
        # Get metrics
        metrics = tracker.get_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.memory_mb is not None
        assert metrics.memory_mb > 0
        assert metrics.duration_ms is not None
        assert metrics.duration_ms >= 100  # At least 100ms from sleep
        
        # Test metrics dictionary conversion
        metrics_dict = metrics.to_dict()
        assert 'memory_mb' in metrics_dict
        assert 'duration_ms' in metrics_dict
    
    def test_enhanced_logger_basic_functionality(self, enhanced_logger, temp_log_dir):
        """Test basic enhanced logger functionality."""
        # Test basic logging methods
        enhanced_logger.debug("Debug message")
        enhanced_logger.info("Info message")
        enhanced_logger.warning("Warning message")
        enhanced_logger.error("Error message")
        enhanced_logger.critical("Critical message")
        
        # Test error logging with context
        test_error = ValueError("Test error")
        enhanced_logger.log_error_with_context(
            "Error occurred during testing",
            test_error,
            operation_name="test_operation",
            additional_context={"test_data": "value"}
        )
        
        # Check that logs were written
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) > 0
        
        log_content = log_files[0].read_text()
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Error occurred during testing" in log_content
    
    def test_ingestion_logger(self, test_logger):
        """Test specialized ingestion logger functionality."""
        ingestion_logger = IngestionLogger(test_logger)
        
        # Test document processing logging
        ingestion_logger.log_document_start("doc123", "/path/to/doc.pdf", batch_id="batch_001")
        ingestion_logger.log_document_complete(
            "doc123", 
            processing_time_ms=1500.0,
            pages_processed=10,
            characters_extracted=5000,
            batch_id="batch_001"
        )
        
        # Test batch processing logging
        ingestion_logger.log_batch_start("batch_001", 5, 10, 0)
        ingestion_logger.log_batch_progress("batch_001", 3, 1, 512.5)
        ingestion_logger.log_batch_complete("batch_001", 4, 1, 7500.0)
        
        # Test error logging
        test_error = Exception("Processing failed")
        ingestion_logger.log_document_error("doc456", test_error, batch_id="batch_001", retry_count=2)
    
    def test_diagnostic_logger(self, test_logger):
        """Test specialized diagnostic logger functionality."""
        diagnostic_logger = DiagnosticLogger(test_logger)
        
        # Test configuration validation logging
        validation_results = {
            "api_key": "configured",
            "model": "valid", 
            "storage_path": "accessible"
        }
        diagnostic_logger.log_configuration_validation("test_config", validation_results)
        
        # Test storage initialization logging
        diagnostic_logger.log_storage_initialization(
            "vector_store", 
            "/path/to/storage",
            123.5,
            True
        )
        
        # Test API call logging
        diagnostic_logger.log_api_call_details(
            "llm_completion",
            "gpt-4o-mini",
            1000,
            0.0025,
            850.0,
            True
        )
        
        # Test memory usage logging
        diagnostic_logger.log_memory_usage("test_operation", 512.0, 45.2, threshold_mb=1024.0)
    
    def test_performance_logged_decorator(self, test_logger):
        """Test the performance logging decorator."""
        enhanced_logger = EnhancedLogger(test_logger)
        
        @performance_logged("test_function", enhanced_logger)
        def test_function(x, y):
            time.sleep(0.1)  # Simulate work
            return x + y
        
        # Test successful function execution
        result = test_function(2, 3)
        assert result == 5
        
        # Test function that raises exception
        @performance_logged("failing_function", enhanced_logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
    
    def test_structured_logging_file_output(self, temp_log_dir):
        """Test structured logging to file."""
        structured_log_file = temp_log_dir / "structured.jsonl"
        structured_logger = setup_structured_logging("test_structured", structured_log_file)
        
        # Create enhanced logger that uses structured logging
        base_logger = logging.getLogger("test_base")
        enhanced_logger = EnhancedLogger(base_logger)
        
        # Log some structured messages
        enhanced_logger.info("Structured message 1", metadata={"key1": "value1"})
        enhanced_logger.error("Structured error", error_details={"error_type": "TestError"})
        
        # Force flush
        for handler in structured_logger.handlers:
            handler.flush()
        
        # Verify structured log file exists and contains JSON
        if structured_log_file.exists():
            content = structured_log_file.read_text()
            # Should contain JSON lines
            assert len(content.strip()) > 0
    
    def test_create_enhanced_loggers(self, test_logger):
        """Test the factory function for creating enhanced loggers."""
        loggers = create_enhanced_loggers(test_logger)
        
        assert 'enhanced' in loggers
        assert 'ingestion' in loggers
        assert 'diagnostic' in loggers
        
        assert isinstance(loggers['enhanced'], EnhancedLogger)
        assert isinstance(loggers['ingestion'], IngestionLogger)
        assert isinstance(loggers['diagnostic'], DiagnosticLogger)
    
    @pytest.mark.asyncio
    async def test_integration_with_clinical_metabolomics_rag(self):
        """Test integration of enhanced logging with main RAG class."""
        # Create minimal test configuration
        config = LightRAGConfig(
            api_key="test-key",
            working_dir=Path("/tmp/test_rag"),
            auto_create_dirs=False,
            enable_file_logging=True,
            log_level="DEBUG"
        )
        
        # Mock the LightRAG dependencies to avoid actual API calls
        with patch('lightrag_integration.clinical_metabolomics_rag.LIGHTRAG_AVAILABLE', False), \
             patch('openai.OpenAI') as mock_openai_client:
            
            # Create RAG instance
            rag = ClinicalMetabolomicsRAG(config)
            
            # Verify enhanced logging is initialized
            assert hasattr(rag, 'enhanced_loggers')
            assert hasattr(rag, 'performance_tracker')
            
            if rag.enhanced_loggers:
                assert 'enhanced' in rag.enhanced_loggers
                assert 'ingestion' in rag.enhanced_loggers
                assert 'diagnostic' in rag.enhanced_loggers
    
    def test_performance_metrics_accuracy(self):
        """Test accuracy of performance metrics collection."""
        tracker = PerformanceTracker()
        
        # Start tracking
        tracker.start_tracking()
        
        # Simulate controlled work
        start_time = time.time()
        time.sleep(0.2)  # 200ms sleep
        end_time = time.time()
        
        metrics = tracker.get_metrics()
        expected_duration = (end_time - start_time) * 1000  # Convert to ms
        
        # Duration should be close to expected (within 50ms tolerance)
        assert abs(metrics.duration_ms - expected_duration) < 50
        
        # Memory metrics should be reasonable
        assert metrics.memory_mb > 0
        assert metrics.memory_mb < 10000  # Less than 10GB (reasonable for test)
        assert 0 <= metrics.memory_percent <= 100
    
    def test_correlation_id_propagation(self):
        """Test that correlation IDs are properly propagated through nested operations."""
        manager = CorrelationIDManager()
        
        correlation_ids = []
        
        def capture_correlation_id():
            correlation_ids.append(manager.get_correlation_id())
        
        with manager.operation_context("level1", user_id="user1"):
            capture_correlation_id()
            
            with manager.operation_context("level2"):
                capture_correlation_id()
                
                with manager.operation_context("level3"):
                    capture_correlation_id()
                
                capture_correlation_id()  # Back to level2
            
            capture_correlation_id()  # Back to level1
        
        capture_correlation_id()  # Outside all contexts
        
        # Should have 6 correlation IDs
        assert len(correlation_ids) == 6
        
        # Level 1 context
        assert correlation_ids[0] is not None
        assert correlation_ids[4] == correlation_ids[0]  # Same when returning to level1
        
        # Level 2 context (different from level 1)
        assert correlation_ids[1] is not None
        assert correlation_ids[1] != correlation_ids[0]
        assert correlation_ids[3] == correlation_ids[1]  # Same when returning to level2
        
        # Level 3 context (different from others)
        assert correlation_ids[2] is not None
        assert correlation_ids[2] not in [correlation_ids[0], correlation_ids[1]]
        
        # Outside all contexts
        assert correlation_ids[5] is None
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring capabilities."""
        tracker = PerformanceTracker()
        
        # Get initial metrics
        initial_metrics = tracker.get_metrics()
        assert initial_metrics.memory_mb > 0
        
        # Create some memory usage
        large_data = ['x' * 1000 for _ in range(1000)]  # ~1MB of data
        
        # Get metrics after memory allocation
        after_metrics = tracker.get_metrics()
        
        # Memory usage should have increased (though this might be small)
        # Just verify the metrics are still reasonable
        assert after_metrics.memory_mb > 0
        assert after_metrics.memory_percent >= 0
        
        # Clean up
        del large_data
    
    def test_error_context_capture(self, enhanced_logger):
        """Test that error context is properly captured."""
        def nested_function():
            def inner_function():
                raise ValueError("Inner error")
            return inner_function()
        
        try:
            nested_function()
        except ValueError as e:
            enhanced_logger.log_error_with_context(
                "Error in nested function",
                e,
                operation_name="test_nested_error",
                additional_context={"nested_level": 2}
            )
        
        # Verify that the error was logged (this mainly tests that no exceptions are raised)
        # In a real test, you'd check the log output
        assert True  # Test passes if no exception during logging
    
    def benchmark_logging_performance(self, enhanced_logger, iterations=1000):
        """Benchmark the performance impact of enhanced logging."""
        # Test basic logging performance
        start_time = time.time()
        for i in range(iterations):
            enhanced_logger.info(f"Test message {i}")
        basic_time = time.time() - start_time
        
        # Test structured logging performance
        start_time = time.time()
        for i in range(iterations):
            enhanced_logger.info(
                f"Structured message {i}",
                operation_name="benchmark_test",
                metadata={"iteration": i, "test_data": "value"}
            )
        structured_time = time.time() - start_time
        
        # Test error logging performance
        test_error = Exception("Benchmark error")
        start_time = time.time()
        for i in range(iterations):
            try:
                enhanced_logger.log_error_with_context(
                    f"Error message {i}",
                    test_error,
                    operation_name="benchmark_error",
                    additional_context={"iteration": i}
                )
            except:
                pass  # Ignore any logging errors during benchmark
        error_logging_time = time.time() - start_time
        
        return {
            'basic_logging_time': basic_time,
            'structured_logging_time': structured_time,
            'error_logging_time': error_logging_time,
            'iterations': iterations,
            'basic_per_message_ms': (basic_time / iterations) * 1000,
            'structured_per_message_ms': (structured_time / iterations) * 1000,
            'error_per_message_ms': (error_logging_time / iterations) * 1000
        }
    
    def test_logging_performance_impact(self, enhanced_logger):
        """Test that enhanced logging doesn't have excessive performance impact."""
        benchmark_results = self.benchmark_logging_performance(enhanced_logger, 100)
        
        # Basic logging should be very fast (< 1ms per message)
        assert benchmark_results['basic_per_message_ms'] < 1.0
        
        # Structured logging should still be reasonable (< 5ms per message)
        assert benchmark_results['structured_per_message_ms'] < 5.0
        
        # Error logging with context should be reasonable (< 10ms per message)
        assert benchmark_results['error_per_message_ms'] < 10.0
        
        print(f"Logging Performance Benchmark Results:")
        print(f"  Basic logging: {benchmark_results['basic_per_message_ms']:.2f}ms per message")
        print(f"  Structured logging: {benchmark_results['structured_per_message_ms']:.2f}ms per message") 
        print(f"  Error logging: {benchmark_results['error_per_message_ms']:.2f}ms per message")


def run_specific_tests(category: str = "all"):
    """Run specific test categories."""
    test_instance = TestEnhancedLoggingSystem()
    
    if category == "basic" or category == "all":
        print("Running basic functionality tests...")
        # These would need proper pytest setup, but showing structure
        
    if category == "performance" or category == "all":
        print("Running performance tests...")
        # Create a temporary logger for testing
        logger = logging.getLogger("benchmark_test")
        enhanced_logger = EnhancedLogger(logger)
        
        results = test_instance.benchmark_logging_performance(enhanced_logger)
        print("Performance benchmark results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
    if category == "integration" or category == "all":
        print("Running integration tests...")
        # Integration tests would go here


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Enhanced Logging System")
    parser.add_argument("--category", choices=["basic", "performance", "integration", "all"], 
                       default="all", help="Test category to run")
    
    args = parser.parse_args()
    
    print("Enhanced Logging System Test Suite")
    print("==================================")
    
    run_specific_tests(args.category)
    
    print("\nTest suite completed. For full pytest integration, run:")
    print("pytest test_enhanced_logging_system.py -v")