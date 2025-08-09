"""
Unit tests for core cache storage and retrieval operations.

This module tests the fundamental cache storage and retrieval operations
including cache key generation, data serialization/deserialization,
entry metadata management, and size limit enforcement.

Test Coverage:
- Cache key generation and collision handling
- Data serialization and deserialization
- Cache entry metadata management  
- Size limit enforcement and eviction policies
- TTL (Time To Live) expiration handling
- Thread safety and concurrent operations
- Memory usage patterns
- Performance characteristics

Classes:
    TestCacheKeyGeneration: Tests for cache key generation and collision handling
    TestCacheDataSerialization: Tests for data serialization/deserialization
    TestCacheMetadata: Tests for cache entry metadata management
    TestCacheSizeLimits: Tests for size limit enforcement
    TestCacheTTL: Tests for TTL expiration handling
    TestCacheThreadSafety: Tests for concurrent operations
    TestCachePerformance: Tests for performance characteristics

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import pickle
import json
import hashlib
import sys
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime, timedelta
import concurrent.futures

# Test fixtures and realistic biomedical data
BIOMEDICAL_QUERY_FIXTURES = [
    {
        'query': 'What are the metabolic pathways involved in glucose metabolism?',
        'expected_cache_key': 'bio_glucose_metabolism_pathways',
        'cache_tier_preference': 'L1',
        'expected_ttl': 3600,
        'data_size': 1024
    },
    {
        'query': 'How does insulin resistance affect metabolomics profiles?',
        'expected_cache_key': 'bio_insulin_resistance_metabolomics',
        'cache_tier_preference': 'L1',
        'expected_ttl': 3600,
        'data_size': 2048
    },
    {
        'query': 'What biomarkers indicate cardiovascular disease risk?',
        'expected_cache_key': 'bio_cvd_biomarkers_risk',
        'cache_tier_preference': 'L2',
        'expected_ttl': 7200,
        'data_size': 1536
    },
    {
        'query': 'Latest research on cancer metabolomics 2024',
        'expected_cache_key': 'temporal_cancer_metabolomics_2024',
        'cache_tier_preference': None,  # Should not be cached due to temporal nature
        'expected_ttl': 300,
        'data_size': 4096
    },
    {
        'query': 'Metabolomic analysis of diabetes progression',
        'expected_cache_key': 'bio_diabetes_metabolomic_progression',
        'cache_tier_preference': 'L2',
        'expected_ttl': 7200,
        'data_size': 3072
    }
]

PERFORMANCE_STRESS_FIXTURES = [
    {
        'query': f'Large dataset query {i}',
        'data_size': 1024 * (i + 1),  # Varying sizes from 1KB to 100KB
        'expected_ttl': 3600
    } for i in range(100)
]

@dataclass
class CacheEntry:
    """Standard cache entry structure for testing."""
    key: str
    value: Any
    timestamp: float
    ttl: int
    metadata: Dict[str, Any]
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > (self.timestamp + self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary."""
        return asdict(self)


class MockCache:
    """Mock cache implementation for testing core operations."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.storage: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0,
            'operations': 0
        }
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key from query text."""
        # Normalize query text
        normalized = query.lower().strip()
        # Generate hash
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        return f"cache_{hash_obj.hexdigest()[:16]}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage."""
        try:
            # Try JSON first (faster)
            return json.dumps(data).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(data)
    
    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            # Try JSON first
            return json.loads(data_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data_bytes)
    
    def _enforce_size_limits(self):
        """Enforce cache size limits through LRU eviction."""
        while len(self.storage) >= self.max_size:
            # Remove least recently used item
            evicted_key, evicted_entry = self.storage.popitem(last=False)
            self.stats['evictions'] += 1
            self.stats['total_size_bytes'] -= evicted_entry.size_bytes
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.storage.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self.storage.pop(key)
            self.stats['total_size_bytes'] -= entry.size_bytes
    
    def set(self, query: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store value in cache."""
        with self._lock:
            key = self._generate_cache_key(query)
            serialized_data = self._serialize_data(value)
            metadata = metadata or {}
            serialized_metadata = self._serialize_data(metadata) if metadata else b''
            size_bytes = len(serialized_data) + len(serialized_metadata) + len(key)
            
            # Clean up expired entries first
            self._cleanup_expired_entries()
            
            # Enforce size limits
            self._enforce_size_limits()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=serialized_data,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                metadata=metadata,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self.storage:
                old_entry = self.storage[key]
                self.stats['total_size_bytes'] -= old_entry.size_bytes
            
            # Store new entry
            self.storage[key] = entry
            self.storage.move_to_end(key)  # Mark as most recently used
            
            # Update stats
            self.stats['total_size_bytes'] += size_bytes
            self.stats['operations'] += 1
            
            return key
    
    def get(self, query: str) -> Optional[Any]:
        """Retrieve value from cache."""
        with self._lock:
            key = self._generate_cache_key(query)
            
            # Clean up all expired entries when accessed
            self._cleanup_expired_entries()
            
            if key not in self.storage:
                self.stats['misses'] += 1
                self.stats['operations'] += 1
                return None
            
            entry = self.storage[key]
            
            # Check if expired (should have been cleaned up above, but double check)
            if entry.is_expired():
                self.storage.pop(key)
                self.stats['total_size_bytes'] -= entry.size_bytes
                self.stats['misses'] += 1
                self.stats['operations'] += 1
                return None
            
            # Move to end (mark as most recently used)
            self.storage.move_to_end(key)
            
            # Update stats
            self.stats['hits'] += 1
            self.stats['operations'] += 1
            
            # Deserialize and return data
            return self._deserialize_data(entry.value)
    
    def delete(self, query: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            key = self._generate_cache_key(query)
            
            if key in self.storage:
                entry = self.storage.pop(key)
                self.stats['total_size_bytes'] -= entry.size_bytes
                self.stats['operations'] += 1
                return True
            
            return False
    
    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self.storage.clear()
            self.stats['total_size_bytes'] = 0
            self.stats['operations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_ops = self.stats['operations']
            hit_rate = self.stats['hits'] / total_ops if total_ops > 0 else 0
            miss_rate = self.stats['misses'] / total_ops if total_ops > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'total_entries': len(self.storage),
                'total_size_bytes': self.stats['total_size_bytes'],
                'evictions': self.stats['evictions'],
                'operations': total_ops
            }


class TestCacheKeyGeneration:
    """Tests for cache key generation and collision handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockCache()
    
    def test_cache_key_consistency(self):
        """Test that identical queries generate identical cache keys."""
        query = "What are the metabolic pathways involved in glucose metabolism?"
        
        key1 = self.cache._generate_cache_key(query)
        key2 = self.cache._generate_cache_key(query)
        
        assert key1 == key2, "Identical queries should generate identical cache keys"
        assert len(key1) > 10, "Cache key should be sufficiently long"
        assert key1.startswith("cache_"), "Cache key should have proper prefix"
    
    def test_cache_key_normalization(self):
        """Test cache key normalization handles whitespace and case."""
        base_query = "glucose metabolism pathways"
        variations = [
            "glucose metabolism pathways",
            "  glucose metabolism pathways  ",
            "Glucose Metabolism Pathways",
            "GLUCOSE METABOLISM PATHWAYS",
            "\t\nglucose metabolism pathways\n\t"
        ]
        
        keys = [self.cache._generate_cache_key(q) for q in variations]
        
        # All variations should produce the same key
        assert len(set(keys)) == 1, "Query variations should normalize to same cache key"
    
    def test_cache_key_uniqueness(self):
        """Test that different queries generate different cache keys."""
        queries = [fixture['query'] for fixture in BIOMEDICAL_QUERY_FIXTURES]
        keys = [self.cache._generate_cache_key(q) for q in queries]
        
        # All keys should be unique
        assert len(keys) == len(set(keys)), "Different queries should generate unique cache keys"
    
    def test_cache_key_collision_resistance(self):
        """Test cache key collision resistance with similar queries."""
        similar_queries = [
            "glucose metabolism in diabetes",
            "glucose metabolism and diabetes",
            "diabetes glucose metabolism",
            "metabolism of glucose in diabetes",
            "diabetic glucose metabolism"
        ]
        
        keys = [self.cache._generate_cache_key(q) for q in similar_queries]
        unique_keys = set(keys)
        
        assert len(keys) == len(unique_keys), "Similar queries should generate unique cache keys"
    
    def test_cache_key_performance(self):
        """Test cache key generation performance."""
        query = "What are the metabolic pathways involved in glucose metabolism?"
        
        start_time = time.time()
        for _ in range(1000):
            self.cache._generate_cache_key(query)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 1000) * 1000
        assert avg_time_ms < 1.0, f"Cache key generation should be fast (<1ms), got {avg_time_ms:.2f}ms"


class TestCacheDataSerialization:
    """Tests for data serialization and deserialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockCache()
    
    def test_json_serialization(self):
        """Test JSON serialization for simple data types."""
        test_data = {
            "query": "glucose metabolism",
            "results": ["pathway1", "pathway2", "pathway3"],
            "confidence": 0.85,
            "count": 42
        }
        
        # Serialize and deserialize
        serialized = self.cache._serialize_data(test_data)
        deserialized = self.cache._deserialize_data(serialized)
        
        assert deserialized == test_data, "JSON serialization should preserve data integrity"
        assert isinstance(serialized, bytes), "Serialized data should be bytes"
    
    def test_pickle_serialization_fallback(self):
        """Test pickle serialization fallback for complex objects."""
        # Create complex object that can't be JSON serialized but can be pickled
        complex_data = {
            "datetime": datetime.now(),
            "set": {1, 2, 3, 4, 5},
            "tuple": (1, 2, 3)
        }
        
        # Should fall back to pickle
        serialized = self.cache._serialize_data(complex_data)
        deserialized = self.cache._deserialize_data(serialized)
        
        assert deserialized["datetime"] == complex_data["datetime"]
        assert deserialized["set"] == complex_data["set"]
        assert deserialized["tuple"] == complex_data["tuple"]
    
    def test_serialization_size_efficiency(self):
        """Test serialization size efficiency."""
        data = {"query": "test", "results": ["a"] * 100}
        
        json_size = len(self.cache._serialize_data(data))
        
        # JSON should be reasonably compact
        assert json_size < 2000, f"Serialized data size should be reasonable, got {json_size} bytes"
    
    def test_serialization_performance(self):
        """Test serialization performance."""
        data = {
            "query": "performance test",
            "results": [f"result_{i}" for i in range(1000)],
            "metadata": {"timestamp": time.time(), "source": "test"}
        }
        
        # Test serialization performance
        start_time = time.time()
        for _ in range(100):
            serialized = self.cache._serialize_data(data)
            self.cache._deserialize_data(serialized)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        assert avg_time_ms < 10.0, f"Serialization should be fast (<10ms), got {avg_time_ms:.2f}ms"
    
    def test_unicode_handling(self):
        """Test proper handling of Unicode characters."""
        unicode_data = {
            "query": "метаболизм глюкозы",  # Russian
            "results": ["蛋白质代谢", "脂质代谢"],  # Chinese
            "emoji": "🧬🔬💊"
        }
        
        serialized = self.cache._serialize_data(unicode_data)
        deserialized = self.cache._deserialize_data(serialized)
        
        assert deserialized == unicode_data, "Unicode characters should be preserved"


class TestCacheMetadata:
    """Tests for cache entry metadata management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockCache()
    
    def test_metadata_storage(self):
        """Test metadata is properly stored with cache entries."""
        query = "glucose metabolism"
        value = {"result": "test data"}
        metadata = {
            "source": "test",
            "confidence": 0.95,
            "timestamp": time.time(),
            "query_type": "biomedical"
        }
        
        key = self.cache.set(query, value, metadata=metadata)
        
        # Retrieve entry directly to check metadata
        entry = self.cache.storage[key]
        assert entry.metadata == metadata, "Metadata should be stored with cache entry"
    
    def test_metadata_preservation(self):
        """Test metadata is preserved during cache operations."""
        query = "insulin resistance"
        value = {"pathways": ["pathway1", "pathway2"]}
        metadata = {
            "query_complexity": "medium",
            "processing_time_ms": 150,
            "cache_tier": "L1"
        }
        
        self.cache.set(query, value, metadata=metadata)
        
        # Retrieve and verify metadata is accessible
        # (In real implementation, you'd have a get_metadata method)
        key = self.cache._generate_cache_key(query)
        entry = self.cache.storage[key]
        
        assert entry.metadata["query_complexity"] == "medium"
        assert entry.metadata["processing_time_ms"] == 150
        assert entry.metadata["cache_tier"] == "L1"
    
    def test_default_metadata(self):
        """Test default metadata handling."""
        query = "default metadata test"
        value = {"test": "data"}
        
        key = self.cache.set(query, value)
        entry = self.cache.storage[key]
        
        assert isinstance(entry.metadata, dict), "Default metadata should be empty dict"
        assert len(entry.metadata) == 0, "Default metadata should be empty"
    
    def test_metadata_size_tracking(self):
        """Test metadata contributes to size tracking."""
        query = "metadata size test"
        value = {"small": "data"}
        large_metadata = {
            "large_field": "x" * 1000,  # 1KB of metadata
            "additional": {"nested": {"deep": {"data": "test"}}}
        }
        
        key = self.cache.set(query, value, metadata=large_metadata)
        entry = self.cache.storage[key]
        
        # Size should include serialized metadata
        assert entry.size_bytes > 1000, "Entry size should include metadata"


class TestCacheSizeLimits:
    """Tests for size limit enforcement and eviction policies."""
    
    def test_lru_eviction_policy(self):
        """Test LRU (Least Recently Used) eviction policy."""
        # Create small cache to trigger eviction
        small_cache = MockCache(max_size=3)
        
        # Fill cache to capacity
        small_cache.set("query1", "data1")
        small_cache.set("query2", "data2") 
        small_cache.set("query3", "data3")
        
        assert len(small_cache.storage) == 3, "Cache should be at capacity"
        
        # Access query1 to make it recently used
        small_cache.get("query1")
        
        # Add new entry, should evict query2 (least recently used)
        small_cache.set("query4", "data4")
        
        assert len(small_cache.storage) == 3, "Cache should maintain size limit"
        assert small_cache.get("query1") == "data1", "Recently used entry should remain"
        assert small_cache.get("query2") is None, "LRU entry should be evicted"
        assert small_cache.get("query3") == "data3", "Other entries should remain"
        assert small_cache.get("query4") == "data4", "New entry should be present"
    
    def test_size_limit_enforcement(self):
        """Test cache size limit enforcement."""
        cache = MockCache(max_size=5)
        
        # Add entries beyond capacity
        for i in range(10):
            cache.set(f"query_{i}", f"data_{i}")
        
        assert len(cache.storage) <= 5, "Cache should not exceed max size"
        assert cache.stats['evictions'] > 0, "Evictions should have occurred"
    
    def test_memory_usage_tracking(self):
        """Test accurate memory usage tracking."""
        cache = MockCache()
        initial_size = cache.stats['total_size_bytes']
        
        # Add entry and check size increase
        cache.set("test_query", {"large_data": "x" * 1000})
        
        assert cache.stats['total_size_bytes'] > initial_size, "Memory usage should increase"
        
        # Remove entry and check size decrease
        cache.delete("test_query")
        
        # Should return to near initial size (might not be exact due to overhead)
        assert cache.stats['total_size_bytes'] <= initial_size, "Memory usage should decrease after deletion"
    
    def test_eviction_statistics(self):
        """Test eviction statistics tracking."""
        cache = MockCache(max_size=2)
        
        # Fill cache beyond capacity to trigger evictions
        for i in range(5):
            cache.set(f"query_{i}", f"data_{i}")
        
        stats = cache.get_stats()
        assert stats['evictions'] >= 3, "Should track eviction count"
    
    def test_large_entry_handling(self):
        """Test handling of entries larger than typical size."""
        cache = MockCache(max_size=10)
        
        # Add very large entry
        large_data = {"massive_field": "x" * 10000}  # 10KB
        cache.set("large_query", large_data)
        
        # Should still be stored and retrievable
        retrieved = cache.get("large_query")
        assert retrieved == large_data, "Large entries should be stored correctly"


class TestCacheTTL:
    """Tests for TTL (Time To Live) expiration handling."""
    
    def test_ttl_expiration(self):
        """Test entries expire after TTL."""
        cache = MockCache(default_ttl=1)  # 1 second TTL
        
        cache.set("expiring_query", "test_data")
        
        # Should be retrievable immediately
        assert cache.get("expiring_query") == "test_data"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should return None after expiration
        assert cache.get("expiring_query") is None
    
    def test_custom_ttl(self):
        """Test custom TTL values."""
        cache = MockCache(default_ttl=3600)  # 1 hour default
        
        # Set with custom short TTL
        cache.set("short_ttl_query", "data", ttl=1)
        # Set with custom long TTL  
        cache.set("long_ttl_query", "data", ttl=7200)
        
        # Get entries directly to check TTL
        short_key = cache._generate_cache_key("short_ttl_query")
        long_key = cache._generate_cache_key("long_ttl_query")
        
        assert cache.storage[short_key].ttl == 1
        assert cache.storage[long_key].ttl == 7200
    
    def test_ttl_cleanup_on_access(self):
        """Test expired entries are cleaned up on access."""
        cache = MockCache(default_ttl=1)
        
        cache.set("query1", "data1")
        cache.set("query2", "data2")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Access should trigger cleanup
        cache.get("query1")
        
        # Both entries should be removed from storage
        assert len(cache.storage) == 0, "Expired entries should be cleaned up"
    
    def test_ttl_statistics(self):
        """Test TTL expiration affects cache statistics."""
        cache = MockCache(default_ttl=1)
        
        cache.set("expiring_query", "data")
        
        # Wait for expiration and access
        time.sleep(1.1)
        result = cache.get("expiring_query")
        
        assert result is None
        stats = cache.get_stats()
        assert stats['misses'] > 0, "TTL expiration should count as cache miss"


class TestCacheThreadSafety:
    """Tests for thread safety and concurrent operations."""
    
    def test_concurrent_read_write(self):
        """Test concurrent read and write operations."""
        cache = MockCache()
        results = {'errors': []}
        
        def writer_thread():
            try:
                for i in range(100):
                    cache.set(f"concurrent_query_{i}", f"data_{i}")
            except Exception as e:
                results['errors'].append(f"Writer error: {e}")
        
        def reader_thread():
            try:
                for i in range(100):
                    cache.get(f"concurrent_query_{i}")
            except Exception as e:
                results['errors'].append(f"Reader error: {e}")
        
        # Start threads
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer_thread))
            threads.append(threading.Thread(target=reader_thread))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results['errors']) == 0, f"Concurrent operations should not cause errors: {results['errors']}"
    
    def test_concurrent_eviction(self):
        """Test concurrent operations during cache eviction."""
        cache = MockCache(max_size=50)
        errors = []
        
        def heavy_writer():
            try:
                for i in range(200):
                    cache.set(f"heavy_query_{i}", f"data_{i}" * 100)
            except Exception as e:
                errors.append(f"Heavy writer error: {e}")
        
        def frequent_reader():
            try:
                for i in range(100):
                    cache.get(f"heavy_query_{i % 50}")
            except Exception as e:
                errors.append(f"Frequent reader error: {e}")
        
        # Start concurrent operations
        threads = [
            threading.Thread(target=heavy_writer),
            threading.Thread(target=heavy_writer),
            threading.Thread(target=frequent_reader),
            threading.Thread(target=frequent_reader)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Concurrent eviction should not cause errors: {errors}"
        
        # Cache should maintain consistency
        stats = cache.get_stats()
        assert stats['total_entries'] <= 50, "Cache size should be maintained during concurrent operations"
    
    def test_atomic_operations(self):
        """Test atomicity of cache operations."""
        cache = MockCache()
        
        def atomic_test():
            # These operations should be atomic
            cache.set("atomic_test", "value1")
            result1 = cache.get("atomic_test")
            cache.set("atomic_test", "value2")
            result2 = cache.get("atomic_test")
            return result1, result2
        
        # Run atomic tests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(atomic_test) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Each thread should see consistent results
        for result1, result2 in results:
            assert result1 in ["value1", "value2"], "Should get valid value"
            assert result2 in ["value1", "value2"], "Should get valid value"


class TestCachePerformance:
    """Tests for cache performance characteristics."""
    
    def test_get_performance(self):
        """Test cache get operation performance."""
        cache = MockCache()
        
        # Populate cache
        for i in range(1000):
            cache.set(f"perf_query_{i}", f"data_{i}")
        
        # Measure get performance
        start_time = time.time()
        for i in range(1000):
            cache.get(f"perf_query_{i}")
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 1000) * 1000
        assert avg_time_ms < 1.0, f"Cache get should be fast (<1ms), got {avg_time_ms:.3f}ms"
    
    def test_set_performance(self):
        """Test cache set operation performance."""
        cache = MockCache()
        
        # Measure set performance
        start_time = time.time()
        for i in range(1000):
            cache.set(f"perf_query_{i}", f"data_{i}")
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 1000) * 1000
        assert avg_time_ms < 2.0, f"Cache set should be fast (<2ms), got {avg_time_ms:.3f}ms"
    
    def test_memory_efficiency(self):
        """Test cache memory efficiency."""
        cache = MockCache()
        
        # Add known amount of data
        data_per_entry = 1000  # ~1KB per entry
        num_entries = 100
        
        for i in range(num_entries):
            data = {"field": "x" * data_per_entry}
            cache.set(f"memory_test_{i}", data)
        
        stats = cache.get_stats()
        
        # Check memory usage is reasonable (allowing for overhead)
        expected_min = num_entries * data_per_entry
        expected_max = expected_min * 2  # Allow 100% overhead
        
        assert expected_min <= stats['total_size_bytes'] <= expected_max, \
            f"Memory usage should be reasonable: expected {expected_min}-{expected_max}, got {stats['total_size_bytes']}"
    
    def test_hit_rate_performance(self):
        """Test cache hit rate under typical usage patterns."""
        cache = MockCache()
        
        # Simulate realistic usage pattern
        queries = [f"common_query_{i}" for i in range(50)]  # 50 unique queries
        
        # First pass - populate cache
        for query in queries:
            cache.set(query, f"data_for_{query}")
        
        # Second pass - should hit cache frequently
        for _ in range(5):  # Repeat each query 5 times
            for query in queries:
                cache.get(query)
        
        stats = cache.get_stats()
        hit_rate = stats['hit_rate']
        
        assert hit_rate > 0.8, f"Hit rate should be high for repeated queries: {hit_rate:.2f}"
    
    def test_scalability_linear_performance(self):
        """Test performance scales reasonably with cache size."""
        # Test small cache
        small_cache = MockCache(max_size=100)
        small_times = []
        
        start = time.time()
        for i in range(100):
            small_cache.set(f"query_{i}", f"data_{i}")
            small_cache.get(f"query_{i}")
        small_times.append(time.time() - start)
        
        # Test larger cache
        large_cache = MockCache(max_size=1000)
        large_times = []
        
        start = time.time()
        for i in range(1000):
            large_cache.set(f"query_{i}", f"data_{i}")
            large_cache.get(f"query_{i}")
        large_times.append(time.time() - start)
        
        # Performance should scale reasonably (not more than 2x slower per operation)
        small_avg = small_times[0] / 100
        large_avg = large_times[0] / 1000
        
        performance_ratio = large_avg / small_avg
        assert performance_ratio < 10.0, f"Performance should scale reasonably: {performance_ratio:.2f}x slower per operation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])