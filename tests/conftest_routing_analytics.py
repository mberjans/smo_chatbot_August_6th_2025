#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures for Routing Decision Analytics Tests

This module provides shared fixtures, configuration, and utilities for
testing the routing decision logging and analytics system.

Key Features:
- Shared test fixtures for mock objects
- Database and file system setup/teardown
- Performance testing utilities
- Test data generation
- Async test support

Author: Claude Code (Anthropic)
Created: August 9, 2025
Task: Test Configuration and Fixtures for Routing Analytics
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, AsyncGenerator
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pytest
import pytest_asyncio

# Import modules under test
from lightrag_integration.routing_decision_analytics import (
    RoutingDecisionLogger,
    RoutingAnalytics,
    LoggingConfig,
    RoutingDecisionLogEntry,
    AnalyticsMetrics,
    LogLevel,
    StorageStrategy,
    RoutingMetricType,
    create_routing_logger,
    create_routing_analytics
)

from lightrag_integration.enhanced_production_router import (
    EnhancedProductionIntelligentQueryRouter,
    EnhancedFeatureFlags
)

from lightrag_integration.query_router import (
    BiomedicalQueryRouter,
    RoutingDecision,
    RoutingPrediction,
    ConfidenceMetrics
)

from lightrag_integration.production_intelligent_query_router import (
    ProductionIntelligentQueryRouter,
    DeploymentMode,
    ProductionFeatureFlags
)


# Configure pytest-asyncio
pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for all test files"""
    temp_dir = tempfile.mkdtemp(prefix="routing_analytics_tests_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_log_dir(temp_test_dir):
    """Create a temporary log directory for individual tests"""
    log_dir = Path(temp_test_dir) / f"logs_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    yield str(log_dir)
    # Individual test cleanup happens in teardown


@pytest.fixture
def sample_confidence_metrics():
    """Create sample confidence metrics for testing"""
    return ConfidenceMetrics(
        overall_confidence=0.85,
        research_category_confidence=0.90,
        temporal_analysis_confidence=0.80,
        signal_strength_confidence=0.88,
        context_coherence_confidence=0.82,
        keyword_density=0.75,
        pattern_match_strength=0.85,
        biomedical_entity_count=5,
        ambiguity_score=0.15,
        conflict_score=0.10,
        alternative_interpretations=["diabetes research", "metabolic studies"],
        calculation_time_ms=12.5
    )


@pytest.fixture
def sample_routing_prediction(sample_confidence_metrics):
    """Create sample routing prediction for testing"""
    return RoutingPrediction(
        routing_decision=RoutingDecision.LIGHTRAG,
        confidence_metrics=sample_confidence_metrics,
        reasoning=["High biomedical entity count", "Strong keyword density"],
        research_category="metabolic_disorders"
    )


@pytest.fixture
def mock_base_router():
    """Create mock BiomedicalQueryRouter for testing"""
    router = Mock(spec=BiomedicalQueryRouter)
    router.route_query = AsyncMock()
    return router


@pytest.fixture
def mock_processing_metrics():
    """Create mock processing metrics"""
    return {
        'decision_time_ms': 15.2,
        'total_time_ms': 45.6,
        'backend_selection_time_ms': 5.1,
        'query_complexity': 0.8,
        'preprocessing_time_ms': 8.2,
        'postprocessing_time_ms': 3.1
    }


@pytest.fixture
def mock_system_state():
    """Create mock system state for testing"""
    return {
        'backend_health': {'lightrag': 'healthy', 'perplexity': 'healthy'},
        'backend_load': {
            'lightrag': {'cpu': 45.2, 'memory': 62.1, 'requests': 150},
            'perplexity': {'cpu': 32.1, 'memory': 48.5, 'requests': 85}
        },
        'resource_usage': {
            'cpu_percent': 25.5,
            'memory_percent': 58.3,
            'memory_available_gb': 6.2
        },
        'selection_algorithm': 'weighted_round_robin',
        'backend_weights': {'lightrag': 0.7, 'perplexity': 0.3},
        'errors': [],
        'warnings': ['High memory usage detected'],
        'fallback_used': False,
        'fallback_reason': None,
        'deployment_mode': 'production',
        'feature_flags': {'analytics_enabled': True, 'logging_enabled': True}
    }


@pytest.fixture(params=[
    LogLevel.MINIMAL,
    LogLevel.STANDARD,
    LogLevel.DETAILED,
    LogLevel.DEBUG
])
def log_level(request):
    """Parametrized fixture for different log levels"""
    return request.param


@pytest.fixture(params=[
    StorageStrategy.FILE_ONLY,
    StorageStrategy.MEMORY_ONLY,
    StorageStrategy.HYBRID
])
def storage_strategy(request):
    """Parametrized fixture for different storage strategies"""
    return request.param


@pytest.fixture
def default_logging_config(temp_log_dir):
    """Create default logging configuration for testing"""
    return LoggingConfig(
        enabled=True,
        log_level=LogLevel.STANDARD,
        storage_strategy=StorageStrategy.HYBRID,
        log_directory=temp_log_dir,
        max_file_size_mb=10,
        max_files_to_keep=5,
        compress_old_logs=True,
        max_memory_entries=100,
        memory_retention_hours=1,
        async_logging=False,  # Sync by default for easier testing
        batch_size=10,
        flush_interval_seconds=2,
        anonymize_queries=False,
        hash_sensitive_data=False,
        enable_real_time_analytics=True,
        analytics_aggregation_interval_minutes=1
    )


@pytest.fixture
def async_logging_config(temp_log_dir):
    """Create async logging configuration for testing"""
    return LoggingConfig(
        enabled=True,
        log_level=LogLevel.DETAILED,
        storage_strategy=StorageStrategy.HYBRID,
        log_directory=temp_log_dir,
        async_logging=True,
        batch_size=5,
        flush_interval_seconds=1,
        max_memory_entries=50
    )


@pytest.fixture
def production_logging_config(temp_log_dir):
    """Create production-like logging configuration"""
    return LoggingConfig(
        enabled=True,
        log_level=LogLevel.STANDARD,
        storage_strategy=StorageStrategy.FILE_ONLY,
        log_directory=temp_log_dir,
        max_file_size_mb=100,
        max_files_to_keep=30,
        compress_old_logs=True,
        async_logging=True,
        batch_size=50,
        flush_interval_seconds=10,
        anonymize_queries=True,
        hash_sensitive_data=True,
        enable_real_time_analytics=True,
        analytics_aggregation_interval_minutes=5
    )


@pytest.fixture
def enhanced_feature_flags():
    """Create enhanced feature flags for testing"""
    return EnhancedFeatureFlags(
        enable_production_load_balancer=True,
        deployment_mode=DeploymentMode.A_B_TESTING,
        production_traffic_percentage=50.0,
        enable_routing_logging=True,
        routing_log_level=LogLevel.DETAILED,
        routing_storage_strategy=StorageStrategy.HYBRID,
        enable_real_time_analytics=True,
        analytics_aggregation_interval_minutes=2,
        enable_anomaly_detection=True,
        enable_performance_impact_monitoring=True,
        max_logging_overhead_ms=10.0,
        anonymize_query_content=False,
        hash_sensitive_data=True
    )


@pytest.fixture
async def routing_logger(default_logging_config):
    """Create routing logger with default configuration"""
    logger = RoutingDecisionLogger(default_logging_config)
    if default_logging_config.async_logging:
        await logger.start_async_logging()
    
    yield logger
    
    if default_logging_config.async_logging:
        await logger.stop_async_logging()


@pytest.fixture
def routing_analytics(routing_logger):
    """Create routing analytics with logger"""
    return RoutingAnalytics(routing_logger)


@pytest.fixture
async def enhanced_router(mock_base_router, enhanced_feature_flags, default_logging_config):
    """Create enhanced production router for testing"""
    router = EnhancedProductionIntelligentQueryRouter(
        base_router=mock_base_router,
        feature_flags=enhanced_feature_flags,
        logging_config=default_logging_config
    )
    
    await router.start_monitoring()
    yield router
    await router.stop_monitoring()


class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def create_routing_predictions(count: int) -> List[RoutingPrediction]:
        """Generate multiple routing predictions for testing"""
        predictions = []
        routing_decisions = [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY]
        
        for i in range(count):
            # Vary confidence and performance metrics
            base_confidence = 0.5 + (i % 5) * 0.1
            decision_time = 8.0 + (i % 3) * 2.0
            routing_decision = routing_decisions[i % len(routing_decisions)]
            
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=base_confidence,
                research_category_confidence=base_confidence + 0.05,
                temporal_analysis_confidence=base_confidence - 0.05,
                signal_strength_confidence=base_confidence,
                context_coherence_confidence=base_confidence - 0.02,
                keyword_density=base_confidence * 0.8,
                pattern_match_strength=base_confidence,
                biomedical_entity_count=max(1, int(base_confidence * 8)),
                ambiguity_score=1 - base_confidence,
                conflict_score=1 - base_confidence,
                alternative_interpretations=[f"alternative_{i}"],
                calculation_time_ms=decision_time
            )
            
            prediction = RoutingPrediction(
                routing_decision=routing_decision,
                confidence_metrics=confidence_metrics,
                reasoning=[f"Test reasoning {i}"],
                research_category=f"category_{i % 3}"
            )
            
            predictions.append(prediction)
        
        return predictions
    
    @staticmethod
    def create_log_entries(count: int, config: LoggingConfig) -> List[RoutingDecisionLogEntry]:
        """Generate multiple log entries for testing"""
        predictions = TestDataGenerator.create_routing_predictions(count)
        entries = []
        base_time = datetime.now()
        
        for i, prediction in enumerate(predictions):
            processing_metrics = {
                'decision_time_ms': 10.0 + i * 0.5,
                'total_time_ms': 35.0 + i * 1.2,
                'query_complexity': 0.5 + (i % 5) * 0.1
            }
            
            system_state = {
                'backend_health': {'lightrag': 'healthy', 'perplexity': 'healthy'},
                'resource_usage': {'cpu_percent': 20 + i % 30},
                'backend_weights': {'lightrag': 0.8, 'perplexity': 0.2}
            }
            
            entry = RoutingDecisionLogEntry.from_routing_prediction(
                prediction,
                f"Test query {i}: biomedical research question",
                processing_metrics,
                system_state,
                config
            )
            
            # Set varied timestamps
            entry.timestamp = base_time - timedelta(minutes=i)
            entry.selected_backend = f"backend_{i % 2}"
            
            entries.append(entry)
        
        return entries


@pytest.fixture
def test_data_generator():
    """Provide test data generator utility"""
    return TestDataGenerator()


@pytest.fixture
def sample_log_entries(test_data_generator, default_logging_config):
    """Generate sample log entries for testing"""
    return test_data_generator.create_log_entries(20, default_logging_config)


class PerformanceTestUtils:
    """Utilities for performance testing"""
    
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure execution time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            return result, execution_time
        return wrapper
    
    @staticmethod
    async def measure_async_execution_time(coro):
        """Measure execution time for async function"""
        start_time = time.time()
        result = await coro
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        return result, execution_time
    
    @staticmethod
    def create_performance_baseline(operation_name: str, max_time_ms: float):
        """Create performance baseline checker"""
        def check_performance(execution_time_ms: float):
            assert execution_time_ms <= max_time_ms, \
                f"{operation_name} took {execution_time_ms:.2f}ms, expected <= {max_time_ms}ms"
        return check_performance


@pytest.fixture
def performance_utils():
    """Provide performance testing utilities"""
    return PerformanceTestUtils()


class MockEnvironmentManager:
    """Utility for managing mock environment variables"""
    
    def __init__(self):
        self.original_env = {}
        self.mock_env = {}
    
    def set_env_var(self, key: str, value: str):
        """Set environment variable for testing"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        self.mock_env[key] = value
        os.environ[key] = value
    
    def set_routing_env_vars(self, **kwargs):
        """Set routing-specific environment variables"""
        env_mapping = {
            'logging_enabled': 'ROUTING_LOGGING_ENABLED',
            'log_level': 'ROUTING_LOG_LEVEL',
            'storage_strategy': 'ROUTING_STORAGE_STRATEGY',
            'log_dir': 'ROUTING_LOG_DIR',
            'max_file_size_mb': 'ROUTING_MAX_FILE_SIZE_MB',
            'async_logging': 'ROUTING_ASYNC_LOGGING',
            'anonymize_queries': 'ROUTING_ANONYMIZE_QUERIES',
            'real_time_analytics': 'ROUTING_REAL_TIME_ANALYTICS'
        }
        
        for param, env_var in env_mapping.items():
            if param in kwargs:
                self.set_env_var(env_var, str(kwargs[param]))
    
    def cleanup(self):
        """Restore original environment variables"""
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        
        self.original_env.clear()
        self.mock_env.clear()


@pytest.fixture
def mock_env():
    """Provide environment variable management for tests"""
    manager = MockEnvironmentManager()
    yield manager
    manager.cleanup()


class AsyncTestUtils:
    """Utilities for async testing"""
    
    @staticmethod
    async def wait_for_condition(condition, timeout_seconds=5, check_interval=0.1):
        """Wait for a condition to become true"""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if condition():
                return True
            await asyncio.sleep(check_interval)
        return False
    
    @staticmethod
    async def run_concurrent_operations(operations: List, max_concurrent=10):
        """Run multiple async operations concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation
        
        tasks = [run_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks)


@pytest.fixture
def async_test_utils():
    """Provide async testing utilities"""
    return AsyncTestUtils()


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "async_test: marks tests that require async support")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)
        
        # Mark integration tests based on file name
        if "integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests based on function name
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)


# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for test environment"""
    logging.getLogger("lightrag_integration").setLevel(logging.WARNING)
    logging.getLogger("test").setLevel(logging.DEBUG)
    yield
    # Cleanup logging handlers if needed


# Test database cleanup (if needed for future extensions)
@pytest.fixture(autouse=True)
def cleanup_test_artifacts(temp_test_dir):
    """Cleanup test artifacts after each test"""
    yield
    # Perform any necessary cleanup
    # This runs after each test completes