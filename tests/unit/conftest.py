"""
Pytest configuration and fixtures for cache unit tests.

This module provides pytest configuration, shared fixtures, and utilities
for all cache unit tests. It sets up test environments, provides mock
objects, and configures test execution parameters.

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import test fixtures
from .cache_test_fixtures import (
    CacheTestFixtures, 
    BiomedicalTestDataGenerator,
    MockCacheBackends,
    CachePerformanceMeasurer,
    BIOMEDICAL_QUERIES,
    PERFORMANCE_TEST_QUERIES,
    EMERGENCY_RESPONSE_PATTERNS
)

# Test configuration
TEST_CONFIG = {
    'cache_sizes': {
        'small': 5,
        'medium': 50,
        'large': 500
    },
    'performance_thresholds': {
        'fast_operation_ms': 10,
        'medium_operation_ms': 100,
        'slow_operation_ms': 1000
    },
    'confidence_thresholds': {
        'high': 0.9,
        'medium': 0.7,
        'low': 0.5
    },
    'ttl_values': {
        'short': 300,    # 5 minutes
        'medium': 3600,  # 1 hour  
        'long': 86400    # 24 hours
    }
}


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_cache_dir():
    """Provide temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp(prefix='cache_test_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture
def biomedical_test_data():
    """Provide biomedical test data."""
    return BIOMEDICAL_QUERIES


@pytest.fixture
def performance_test_queries():
    """Provide performance test queries."""
    return PERFORMANCE_TEST_QUERIES


@pytest.fixture
def emergency_response_patterns():
    """Provide emergency response patterns."""
    return EMERGENCY_RESPONSE_PATTERNS


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return BiomedicalTestDataGenerator()


@pytest.fixture
def mock_cache_backends():
    """Provide mock cache backends."""
    return MockCacheBackends()


@pytest.fixture
def performance_measurer():
    """Provide performance measurement utility."""
    measurer = CachePerformanceMeasurer()
    yield measurer
    measurer.clear_measurements()


@pytest.fixture
def cache_test_fixtures():
    """Provide main cache test fixtures."""
    return CacheTestFixtures()


@pytest.fixture
def mock_redis_client():
    """Provide mock Redis client."""
    mock_redis = AsyncMock()
    storage = {}
    
    async def mock_get(key):
        return storage.get(key)
    
    async def mock_set(key, value, ex=None):
        storage[key] = value
        return True
    
    async def mock_delete(key):
        return storage.pop(key, None) is not None
    
    async def mock_exists(key):
        return key in storage
    
    async def mock_ttl(key):
        return -1 if key in storage else -2
    
    mock_redis.get = mock_get
    mock_redis.set = mock_set
    mock_redis.delete = mock_delete
    mock_redis.exists = mock_exists
    mock_redis.ttl = mock_ttl
    
    # Add storage reference for test access
    mock_redis._test_storage = storage
    
    return mock_redis


@pytest.fixture
def mock_disk_cache(temp_cache_dir):
    """Provide mock disk cache."""
    mock_cache = Mock()
    storage = {}
    
    def mock_get(key, default=None):
        return storage.get(key, default)
    
    def mock_set(key, value):
        storage[key] = value
        return True
    
    def mock_delete(key):
        return storage.pop(key, None) is not None
    
    def mock_clear():
        storage.clear()
    
    def mock_size():
        return len(storage)
    
    mock_cache.get = mock_get
    mock_cache.set = mock_set
    mock_cache.delete = mock_delete
    mock_cache.clear = mock_clear
    mock_cache.size = mock_size
    mock_cache.__len__ = lambda: len(storage)
    mock_cache.__contains__ = lambda key: key in storage
    
    # Add storage reference for test access
    mock_cache._test_storage = storage
    mock_cache._cache_dir = temp_cache_dir
    
    return mock_cache


@pytest.fixture
def failing_cache_backend():
    """Provide cache backend that simulates failures."""
    class FailingBackend:
        def __init__(self, failure_rate=0.5):
            self.failure_rate = failure_rate
            self.storage = {}
            self.call_count = 0
        
        async def get(self, key):
            self.call_count += 1
            if self._should_fail():
                raise Exception("Simulated cache failure")
            return self.storage.get(key)
        
        async def set(self, key, value):
            self.call_count += 1
            if self._should_fail():
                raise Exception("Simulated cache failure")
            self.storage[key] = value
            return True
        
        async def delete(self, key):
            self.call_count += 1
            if self._should_fail():
                raise Exception("Simulated cache failure")
            return self.storage.pop(key, None) is not None
        
        def _should_fail(self):
            import random
            return random.random() < self.failure_rate
        
        def set_failure_rate(self, rate):
            self.failure_rate = rate
    
    return FailingBackend()


@pytest.fixture(params=['small', 'medium', 'large'])
def cache_size_config(request, test_config):
    """Provide different cache size configurations."""
    size_name = request.param
    return {
        'size_name': size_name,
        'max_size': test_config['cache_sizes'][size_name],
        'description': f'{size_name} cache configuration'
    }


@pytest.fixture(params=[0.5, 0.7, 0.9])
def confidence_threshold_config(request):
    """Provide different confidence threshold configurations."""
    threshold = request.param
    return {
        'threshold': threshold,
        'description': f'Confidence threshold {threshold}'
    }


@pytest.fixture(params=[300, 3600, 86400])
def ttl_config(request):
    """Provide different TTL configurations."""
    ttl_seconds = request.param
    ttl_names = {300: 'short', 3600: 'medium', 86400: 'long'}
    
    return {
        'ttl_seconds': ttl_seconds,
        'ttl_name': ttl_names[ttl_seconds],
        'description': f'{ttl_names[ttl_seconds]} TTL ({ttl_seconds}s)'
    }


@pytest.fixture
def mock_time():
    """Provide controllable time mock."""
    class MockTime:
        def __init__(self):
            self.current_time = 1000000.0  # Start at a fixed time
        
        def time(self):
            return self.current_time
        
        def advance(self, seconds):
            self.current_time += seconds
        
        def set_time(self, timestamp):
            self.current_time = timestamp
    
    mock_time = MockTime()
    
    with patch('time.time', side_effect=mock_time.time):
        yield mock_time


@pytest.fixture
def memory_usage_tracker():
    """Provide memory usage tracking utility."""
    import psutil
    import gc
    
    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.initial_memory = None
            self.measurements = []
        
        def start_tracking(self):
            gc.collect()  # Force garbage collection
            self.initial_memory = self.process.memory_info().rss
            self.measurements = []
        
        def measure(self, label=""):
            gc.collect()
            current_memory = self.process.memory_info().rss
            delta_mb = (current_memory - self.initial_memory) / (1024 * 1024)
            self.measurements.append({
                'label': label,
                'memory_mb': current_memory / (1024 * 1024),
                'delta_mb': delta_mb
            })
            return delta_mb
        
        def get_peak_usage(self):
            if not self.measurements:
                return 0
            return max(m['delta_mb'] for m in self.measurements)
        
        def get_final_usage(self):
            if not self.measurements:
                return 0
            return self.measurements[-1]['delta_mb']
    
    return MemoryTracker()


@pytest.fixture
def concurrent_test_helper():
    """Provide utilities for concurrent testing."""
    class ConcurrentTestHelper:
        def __init__(self):
            self.results = []
            self.errors = []
        
        async def run_concurrent_operations(self, operations, max_workers=5):
            """Run operations concurrently and collect results."""
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all operations
                futures = []
                for operation in operations:
                    if asyncio.iscoroutinefunction(operation):
                        future = executor.submit(asyncio.run, operation())
                    else:
                        future = executor.submit(operation)
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        self.errors.append(str(e))
        
        def get_success_rate(self):
            total = len(self.results) + len(self.errors)
            return len(self.results) / total if total > 0 else 0
        
        def clear(self):
            self.results.clear()
            self.errors.clear()
    
    return ConcurrentTestHelper()


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "performance: Performance and benchmarking tests")  
    config.addinivalue_line("markers", "integration: Integration tests between components")
    config.addinivalue_line("markers", "reliability: Reliability and failure scenario tests")
    config.addinivalue_line("markers", "slow: Tests that take longer than normal")
    config.addinivalue_line("markers", "concurrent: Tests for concurrent access patterns")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark concurrent tests
        if "concurrent" in item.nodeid.lower() or "thread" in item.nodeid.lower():
            item.add_marker(pytest.mark.concurrent)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ['stress', 'load', 'benchmark']):
            item.add_marker(pytest.mark.slow)


# Test data validation
def validate_test_data():
    """Validate test data integrity."""
    assert BIOMEDICAL_QUERIES, "Biomedical queries should not be empty"
    assert PERFORMANCE_TEST_QUERIES, "Performance test queries should not be empty"
    assert EMERGENCY_RESPONSE_PATTERNS, "Emergency response patterns should not be empty"
    
    # Validate biomedical queries structure
    for category, queries in BIOMEDICAL_QUERIES.items():
        assert isinstance(queries, list), f"Category {category} should contain list of queries"
        for query_data in queries:
            assert 'query' in query_data, f"Query data missing 'query' field in {category}"
            assert 'response' in query_data, f"Query data missing 'response' field in {category}"
            assert 'confidence' in query_data['response'], f"Response missing confidence in {category}"


# Run validation on import
validate_test_data()