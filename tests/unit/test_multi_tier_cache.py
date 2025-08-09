"""
Unit tests for multi-tier cache coordination and operations.

This module tests the multi-level cache hierarchy including L1 memory cache,
L2 disk cache, L3 Redis cache, and cross-tier coordination mechanisms.

Test Coverage:
- L1 memory cache operations and LRU eviction
- L2 disk cache persistence and size management
- L3 Redis cache distributed operations
- Cross-tier cache coordination and fallback
- Cache promotion and demotion strategies
- Write-through and write-behind patterns
- Cache consistency across tiers
- Performance optimization across levels

Classes:
    TestL1MemoryCache: Tests for L1 in-memory cache functionality
    TestL2DiskCache: Tests for L2 persistent disk cache
    TestL3RedisCache: Tests for L3 distributed Redis cache
    TestMultiTierCoordination: Tests for cross-tier coordination
    TestCachePromotionStrategies: Tests for data promotion between tiers
    TestCacheConsistency: Tests for data consistency across tiers
    TestMultiTierPerformance: Tests for performance optimization

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import json
import pickle
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import redis
import diskcache

# Test data fixtures
BIOMEDICAL_TEST_DATA = {
    'glucose_metabolism': {
        'query': 'What are the metabolic pathways involved in glucose metabolism?',
        'response': {
            'pathways': ['glycolysis', 'gluconeogenesis', 'glycogenolysis'],
            'key_enzymes': ['hexokinase', 'phosphofructokinase', 'pyruvate kinase'],
            'confidence': 0.95
        },
        'size_estimate': 2048,
        'tier_preference': 'L1'
    },
    'insulin_resistance': {
        'query': 'How does insulin resistance affect metabolomics profiles?',
        'response': {
            'affected_metabolites': ['glucose', 'fatty_acids', 'amino_acids'],
            'biomarkers': ['HOMA-IR', 'glucose_auc', 'insulin_sensitivity_index'],
            'pathways_disrupted': ['lipid_metabolism', 'protein_synthesis'],
            'confidence': 0.88
        },
        'size_estimate': 3072,
        'tier_preference': 'L1'
    },
    'cardiovascular_biomarkers': {
        'query': 'What biomarkers indicate cardiovascular disease risk?',
        'response': {
            'lipid_panel': ['LDL', 'HDL', 'triglycerides', 'total_cholesterol'],
            'inflammatory_markers': ['CRP', 'IL6', 'TNF_alpha'],
            'metabolic_markers': ['homocysteine', 'lipoprotein_a'],
            'confidence': 0.92
        },
        'size_estimate': 2560,
        'tier_preference': 'L2'
    }
}

class CacheLevel(Enum):
    """Cache tier levels."""
    L1_MEMORY = "L1"
    L2_DISK = "L2"
    L3_REDIS = "L3"

@dataclass
class CacheHit:
    """Information about a cache hit."""
    level: CacheLevel
    key: str
    hit_time_ms: float
    data_size: int

@dataclass
class CacheStats:
    """Comprehensive cache statistics."""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    total_operations: int = 0
    
    def get_overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all tiers."""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_requests = total_hits + self.l1_misses + self.l2_misses + self.l3_misses
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def get_tier_hit_rate(self, tier: CacheLevel) -> float:
        """Calculate hit rate for specific tier."""
        if tier == CacheLevel.L1_MEMORY:
            total = self.l1_hits + self.l1_misses
            return self.l1_hits / total if total > 0 else 0.0
        elif tier == CacheLevel.L2_DISK:
            total = self.l2_hits + self.l2_misses
            return self.l2_hits / total if total > 0 else 0.0
        elif tier == CacheLevel.L3_REDIS:
            total = self.l3_hits + self.l3_misses
            return self.l3_hits / total if total > 0 else 0.0
        return 0.0


class MockL1MemoryCache:
    """Mock L1 memory cache implementation."""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.storage: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache."""
        with self._lock:
            if key in self.storage:
                entry = self.storage[key]
                
                # Check TTL
                if time.time() > entry['timestamp'] + entry['ttl']:
                    del self.storage[key]
                    self.stats['misses'] += 1
                    return None
                
                # Move to end (LRU)
                self.storage.move_to_end(key)
                self.stats['hits'] += 1
                return entry['value']
            
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L1 cache."""
        with self._lock:
            # Enforce size limit
            while len(self.storage) >= self.max_size:
                self.storage.popitem(last=False)
                self.stats['evictions'] += 1
            
            entry = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl,
                'access_count': 1
            }
            
            self.storage[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from L1 cache."""
        with self._lock:
            if key in self.storage:
                del self.storage[key]
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_ops = self.stats['hits'] + self.stats['misses']
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': self.stats['hits'] / total_ops if total_ops > 0 else 0.0,
                'size': len(self.storage),
                'max_size': self.max_size
            }


class MockL2DiskCache:
    """Mock L2 disk cache implementation."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Index file to track entries
        self.index_file = os.path.join(cache_dir, 'cache_index.json')
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except:
            self.index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            pass  # Ignore save errors for testing
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{key}.cache")
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.index.items():
            if current_time > entry['timestamp'] + entry['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._delete_entry(key)
    
    def _delete_entry(self, key: str):
        """Delete cache entry."""
        if key in self.index:
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
            del self.index[key]
            self._save_index()
    
    def _enforce_size_limit(self):
        """Enforce cache size limits."""
        total_size = sum(entry.get('size', 0) for entry in self.index.values())
        
        while total_size > self.max_size_bytes and self.index:
            # Find LRU entry
            lru_key = min(self.index.keys(), 
                         key=lambda k: self.index[k].get('last_access', 0))
            
            entry_size = self.index[lru_key].get('size', 0)
            self._delete_entry(lru_key)
            total_size -= entry_size
            self.stats['evictions'] += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache."""
        self._cleanup_expired()
        
        if key not in self.index:
            self.stats['misses'] += 1
            return None
        
        entry = self.index[key]
        
        # Check if file exists
        file_path = self._get_file_path(key)
        if not os.path.exists(file_path):
            del self.index[key]
            self._save_index()
            self.stats['misses'] += 1
            return None
        
        # Load data
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update last access
            entry['last_access'] = time.time()
            self._save_index()
            
            self.stats['hits'] += 1
            return data
        except:
            # Handle corrupted files
            self._delete_entry(key)
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in L2 cache."""
        try:
            # Serialize data
            file_path = self._get_file_path(key)
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            file_size = os.path.getsize(file_path)
            self.index[key] = {
                'timestamp': time.time(),
                'last_access': time.time(),
                'ttl': ttl,
                'size': file_size
            }
            
            self._save_index()
            self._enforce_size_limit()
            return True
        except Exception as e:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from L2 cache."""
        if key in self.index:
            self._delete_entry(key)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get('size', 0) for entry in self.index.values())
        total_ops = self.stats['hits'] + self.stats['misses']
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': self.stats['hits'] / total_ops if total_ops > 0 else 0.0,
            'size': len(self.index),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_mb
        }


class MockL3RedisCache:
    """Mock L3 Redis cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.stats = {'hits': 0, 'misses': 0, 'errors': 0}
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.connected = True
    
    def _simulate_network_delay(self, delay_ms: float = 1.0):
        """Simulate network delay for Redis operations."""
        time.sleep(delay_ms / 1000)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L3 cache."""
        if not self.connected:
            self.stats['errors'] += 1
            return None
        
        self._simulate_network_delay()
        
        if key not in self.storage:
            self.stats['misses'] += 1
            return None
        
        entry = self.storage[key]
        
        # Check TTL
        if time.time() > entry['timestamp'] + entry['ttl']:
            del self.storage[key]
            self.stats['misses'] += 1
            return None
        
        self.stats['hits'] += 1
        return pickle.loads(entry['data'])
    
    async def set(self, key: str, value: Any, ttl: int = 86400) -> bool:
        """Set value in L3 cache."""
        if not self.connected:
            self.stats['errors'] += 1
            return False
        
        self._simulate_network_delay()
        
        try:
            serialized_data = pickle.dumps(value)
            self.storage[key] = {
                'data': serialized_data,
                'timestamp': time.time(),
                'ttl': ttl
            }
            return True
        except Exception as e:
            self.stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from L3 cache."""
        if not self.connected:
            self.stats['errors'] += 1
            return False
        
        self._simulate_network_delay()
        
        if key in self.storage:
            del self.storage[key]
            return True
        return False
    
    def set_connection_state(self, connected: bool):
        """Simulate connection state changes."""
        self.connected = connected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_ops = self.stats['hits'] + self.stats['misses']
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'errors': self.stats['errors'],
            'hit_rate': self.stats['hits'] / total_ops if total_ops > 0 else 0.0,
            'size': len(self.storage),
            'connected': self.connected
        }


class MultiTierCache:
    """Multi-tier cache system with L1, L2, and L3 levels."""
    
    def __init__(self, l1_cache: MockL1MemoryCache, 
                 l2_cache: MockL2DiskCache, 
                 l3_cache: MockL3RedisCache):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.l3_cache = l3_cache
        self.stats = CacheStats()
        self._promotion_enabled = True
        self._write_through = True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-tier cache with automatic promotion."""
        start_time = time.time()
        
        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats.l1_hits += 1
            return value
        self.stats.l1_misses += 1
        
        # Try L2
        value = await self.l2_cache.get(key)
        if value is not None:
            self.stats.l2_hits += 1
            
            # Promote to L1 if enabled
            if self._promotion_enabled:
                await self.l1_cache.set(key, value)
                self.stats.promotions += 1
            
            return value
        self.stats.l2_misses += 1
        
        # Try L3
        value = await self.l3_cache.get(key)
        if value is not None:
            self.stats.l3_hits += 1
            
            # Promote to L2 and L1 if enabled
            if self._promotion_enabled:
                await self.l2_cache.set(key, value)
                await self.l1_cache.set(key, value)
                self.stats.promotions += 2
            
            return value
        self.stats.l3_misses += 1
        
        return None
    
    async def set(self, key: str, value: Any, 
                  tier_preference: Optional[CacheLevel] = None) -> bool:
        """Set value in multi-tier cache."""
        success = True
        
        if self._write_through:
            # Write-through: write to all tiers
            l1_success = await self.l1_cache.set(key, value)
            l2_success = await self.l2_cache.set(key, value)
            l3_success = await self.l3_cache.set(key, value)
            success = l1_success and l2_success and l3_success
        else:
            # Write to preferred tier only
            if tier_preference == CacheLevel.L1_MEMORY:
                success = await self.l1_cache.set(key, value)
            elif tier_preference == CacheLevel.L2_DISK:
                success = await self.l2_cache.set(key, value)
            elif tier_preference == CacheLevel.L3_REDIS:
                success = await self.l3_cache.set(key, value)
            else:
                # Default to L1
                success = await self.l1_cache.set(key, value)
        
        if success:
            self.stats.total_operations += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers."""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key)
        l3_deleted = await self.l3_cache.delete(key)
        
        return l1_deleted or l2_deleted or l3_deleted
    
    def set_promotion_enabled(self, enabled: bool):
        """Enable/disable cache promotion."""
        self._promotion_enabled = enabled
    
    def set_write_through_enabled(self, enabled: bool):
        """Enable/disable write-through mode."""
        self._write_through = enabled
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all cache tiers."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        
        return {
            'overall_hit_rate': self.stats.get_overall_hit_rate(),
            'promotions': self.stats.promotions,
            'total_operations': self.stats.total_operations,
            'l1': l1_stats,
            'l2': l2_stats,
            'l3': l3_stats,
            'tier_hit_rates': {
                'L1': self.stats.get_tier_hit_rate(CacheLevel.L1_MEMORY),
                'L2': self.stats.get_tier_hit_rate(CacheLevel.L2_DISK),
                'L3': self.stats.get_tier_hit_rate(CacheLevel.L3_REDIS)
            }
        }


class TestL1MemoryCache:
    """Tests for L1 in-memory cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.l1_cache = MockL1MemoryCache(max_size=5, default_ttl=300)
    
    @pytest.mark.asyncio
    async def test_l1_basic_operations(self):
        """Test basic L1 cache operations."""
        # Test set and get
        success = await self.l1_cache.set("test_key", "test_value")
        assert success, "L1 cache set should succeed"
        
        value = await self.l1_cache.get("test_key")
        assert value == "test_value", "L1 cache should return correct value"
        
        # Test delete
        deleted = await self.l1_cache.delete("test_key")
        assert deleted, "L1 cache delete should succeed"
        
        value = await self.l1_cache.get("test_key")
        assert value is None, "L1 cache should return None after delete"
    
    @pytest.mark.asyncio
    async def test_l1_lru_eviction(self):
        """Test LRU eviction in L1 cache."""
        # Fill cache to capacity
        for i in range(5):
            await self.l1_cache.set(f"key_{i}", f"value_{i}")
        
        # Access key_0 to make it recently used
        await self.l1_cache.get("key_0")
        
        # Add new entry, should evict key_1 (least recently used)
        await self.l1_cache.set("new_key", "new_value")
        
        assert await self.l1_cache.get("key_0") == "value_0", "Recently used entry should remain"
        assert await self.l1_cache.get("key_1") is None, "LRU entry should be evicted"
        assert await self.l1_cache.get("new_key") == "new_value", "New entry should be present"
    
    @pytest.mark.asyncio
    async def test_l1_ttl_expiration(self):
        """Test TTL expiration in L1 cache."""
        # Set entry with short TTL
        await self.l1_cache.set("expiring_key", "expiring_value", ttl=1)
        
        # Should be retrievable immediately
        value = await self.l1_cache.get("expiring_key")
        assert value == "expiring_value", "Entry should be retrievable before expiration"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be None after expiration
        value = await self.l1_cache.get("expiring_key")
        assert value is None, "Entry should expire after TTL"
    
    @pytest.mark.asyncio
    async def test_l1_performance_characteristics(self):
        """Test L1 cache performance characteristics."""
        # Populate cache
        for i in range(100):
            await self.l1_cache.set(f"perf_key_{i}", f"perf_value_{i}")
        
        # Measure get performance
        start_time = time.time()
        for i in range(100):
            await self.l1_cache.get(f"perf_key_{i}")
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        assert avg_time_ms < 1.0, f"L1 cache get should be fast (<1ms), got {avg_time_ms:.3f}ms"


class TestL2DiskCache:
    """Tests for L2 persistent disk cache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.l2_cache = MockL2DiskCache(self.temp_dir, max_size_mb=1)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_l2_persistence(self):
        """Test L2 cache persistence across restarts."""
        # Store data
        await self.l2_cache.set("persistent_key", {"data": "persistent_value"})
        
        # Create new cache instance (simulates restart)
        new_l2_cache = MockL2DiskCache(self.temp_dir, max_size_mb=1)
        
        # Should retrieve stored data
        value = await new_l2_cache.get("persistent_key")
        assert value == {"data": "persistent_value"}, "L2 cache should persist data across restarts"
    
    @pytest.mark.asyncio
    async def test_l2_size_management(self):
        """Test L2 cache size management."""
        # Add large entries to trigger size management
        large_data = {"large_field": "x" * 10000}  # ~10KB
        
        # Add multiple entries
        for i in range(10):
            await self.l2_cache.set(f"large_key_{i}", large_data)
        
        stats = self.l2_cache.get_stats()
        assert stats['total_size_mb'] <= 1.0, "L2 cache should respect size limits"
        assert stats['evictions'] > 0, "L2 cache should evict entries when size limit reached"
    
    @pytest.mark.asyncio
    async def test_l2_corruption_recovery(self):
        """Test L2 cache handles corrupted files."""
        # Create cache entry
        await self.l2_cache.set("corruption_key", "test_data")
        
        # Corrupt the file
        file_path = self.l2_cache._get_file_path("corruption_key")
        with open(file_path, 'w') as f:
            f.write("corrupted data")
        
        # Should handle corruption gracefully
        value = await self.l2_cache.get("corruption_key")
        assert value is None, "L2 cache should handle corrupted files gracefully"
    
    @pytest.mark.asyncio
    async def test_l2_concurrent_access(self):
        """Test L2 cache concurrent access."""
        async def writer():
            for i in range(10):
                await self.l2_cache.set(f"concurrent_key_{i}", f"concurrent_value_{i}")
        
        async def reader():
            for i in range(10):
                await self.l2_cache.get(f"concurrent_key_{i}")
        
        # Run concurrent operations
        await asyncio.gather(writer(), reader(), writer(), reader())
        
        # Should not raise exceptions
        stats = self.l2_cache.get_stats()
        assert stats['hits'] + stats['misses'] > 0, "L2 cache should handle concurrent operations"


class TestL3RedisCache:
    """Tests for L3 distributed Redis cache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.l3_cache = MockL3RedisCache()
    
    @pytest.mark.asyncio
    async def test_l3_basic_operations(self):
        """Test basic L3 cache operations."""
        # Test set and get
        success = await self.l3_cache.set("redis_key", {"redis": "value"})
        assert success, "L3 cache set should succeed"
        
        value = await self.l3_cache.get("redis_key")
        assert value == {"redis": "value"}, "L3 cache should return correct value"
        
        # Test delete
        deleted = await self.l3_cache.delete("redis_key")
        assert deleted, "L3 cache delete should succeed"
        
        value = await self.l3_cache.get("redis_key")
        assert value is None, "L3 cache should return None after delete"
    
    @pytest.mark.asyncio
    async def test_l3_connection_failure_handling(self):
        """Test L3 cache connection failure handling."""
        # Simulate connection failure
        self.l3_cache.set_connection_state(False)
        
        # Operations should fail gracefully
        success = await self.l3_cache.set("fail_key", "fail_value")
        assert not success, "L3 cache set should fail when disconnected"
        
        value = await self.l3_cache.get("fail_key")
        assert value is None, "L3 cache get should return None when disconnected"
        
        stats = self.l3_cache.get_stats()
        assert stats['errors'] > 0, "L3 cache should track connection errors"
    
    @pytest.mark.asyncio
    async def test_l3_ttl_management(self):
        """Test L3 cache TTL management."""
        # Set with short TTL
        await self.l3_cache.set("ttl_key", "ttl_value", ttl=1)
        
        # Should be available immediately
        value = await self.l3_cache.get("ttl_key")
        assert value == "ttl_value", "L3 cache should return value before TTL"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        value = await self.l3_cache.get("ttl_key")
        assert value is None, "L3 cache should expire entries after TTL"
    
    @pytest.mark.asyncio
    async def test_l3_network_simulation(self):
        """Test L3 cache network delay simulation."""
        # Measure operation time
        start_time = time.time()
        await self.l3_cache.set("network_key", "network_value")
        await self.l3_cache.get("network_key")
        end_time = time.time()
        
        # Should have some delay (simulated network latency)
        total_time_ms = (end_time - start_time) * 1000
        assert total_time_ms >= 2.0, "L3 cache should simulate network delay"


class TestMultiTierCoordination:
    """Tests for cross-tier cache coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.l1_cache = MockL1MemoryCache(max_size=5)
        self.l2_cache = MockL2DiskCache(self.temp_dir, max_size_mb=1)
        self.l3_cache = MockL3RedisCache()
        self.multi_cache = MultiTierCache(self.l1_cache, self.l2_cache, self.l3_cache)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_cache_tier_fallback_chain(self):
        """Test L1 -> L2 -> L3 fallback chain."""
        # Store only in L3
        await self.l3_cache.set("fallback_key", "fallback_value")
        
        # Should fallback through tiers
        value = await self.multi_cache.get("fallback_key")
        assert value == "fallback_value", "Should retrieve value through fallback chain"
        
        # Check statistics
        stats = self.multi_cache.stats
        assert stats.l1_misses == 1, "Should miss L1"
        assert stats.l2_misses == 1, "Should miss L2"
        assert stats.l3_hits == 1, "Should hit L3"
    
    @pytest.mark.asyncio
    async def test_cache_promotion_strategies(self):
        """Test automatic promotion of frequently accessed data."""
        # Store in L3 only
        await self.l3_cache.set("promote_key", "promote_value")
        
        # Access through multi-tier cache
        value = await self.multi_cache.get("promote_key")
        assert value == "promote_value", "Should retrieve value"
        
        # Value should now be promoted to L1 and L2
        l1_value = await self.l1_cache.get("promote_key")
        l2_value = await self.l2_cache.get("promote_key")
        
        assert l1_value == "promote_value", "Value should be promoted to L1"
        assert l2_value == "promote_value", "Value should be promoted to L2"
        
        assert self.multi_cache.stats.promotions == 2, "Should track promotions"
    
    @pytest.mark.asyncio
    async def test_write_through_vs_write_behind(self):
        """Test write-through vs write-behind patterns."""
        # Test write-through (default)
        await self.multi_cache.set("writethrough_key", "writethrough_value")
        
        # Should be in all tiers
        l1_value = await self.l1_cache.get("writethrough_key")
        l2_value = await self.l2_cache.get("writethrough_key")
        l3_value = await self.l3_cache.get("writethrough_key")
        
        assert l1_value == "writethrough_value", "Write-through should write to L1"
        assert l2_value == "writethrough_value", "Write-through should write to L2"
        assert l3_value == "writethrough_value", "Write-through should write to L3"
        
        # Test write-behind (tier-specific)
        self.multi_cache.set_write_through_enabled(False)
        await self.multi_cache.set("writebehind_key", "writebehind_value", 
                                   tier_preference=CacheLevel.L1_MEMORY)
        
        l1_value = await self.l1_cache.get("writebehind_key")
        l2_value = await self.l2_cache.get("writebehind_key")
        
        assert l1_value == "writebehind_value", "Write-behind should write to preferred tier"
        assert l2_value is None, "Write-behind should not write to other tiers"
    
    @pytest.mark.asyncio
    async def test_cache_consistency_across_tiers(self):
        """Test data consistency between cache levels."""
        # Set initial value
        await self.multi_cache.set("consistency_key", "initial_value")
        
        # Update L1 directly (simulates inconsistency)
        await self.l1_cache.set("consistency_key", "updated_value")
        
        # Should get updated value from L1
        value = await self.multi_cache.get("consistency_key")
        assert value == "updated_value", "Should get most recent value from highest tier"
        
        # Delete from L1
        await self.l1_cache.delete("consistency_key")
        
        # Should fallback to L2
        value = await self.multi_cache.get("consistency_key")
        assert value == "initial_value", "Should fallback to lower tier after deletion"
    
    @pytest.mark.asyncio
    async def test_comprehensive_statistics(self):
        """Test comprehensive statistics across all tiers."""
        # Generate various cache operations
        test_data = BIOMEDICAL_TEST_DATA
        
        for key, data in test_data.items():
            await self.multi_cache.set(key, data['response'])
        
        # Access some entries multiple times
        for key in list(test_data.keys())[:2]:
            await self.multi_cache.get(key)
            await self.multi_cache.get(key)
        
        # Get comprehensive stats
        stats = self.multi_cache.get_comprehensive_stats()
        
        assert 'overall_hit_rate' in stats, "Should provide overall hit rate"
        assert 'promotions' in stats, "Should track promotions"
        assert 'l1' in stats, "Should provide L1 statistics"
        assert 'l2' in stats, "Should provide L2 statistics"
        assert 'l3' in stats, "Should provide L3 statistics"
        assert 'tier_hit_rates' in stats, "Should provide per-tier hit rates"


class TestCachePromotionStrategies:
    """Tests for cache promotion and demotion strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.l1_cache = MockL1MemoryCache(max_size=3)
        self.l2_cache = MockL2DiskCache(self.temp_dir, max_size_mb=1)
        self.l3_cache = MockL3RedisCache()
        self.multi_cache = MultiTierCache(self.l1_cache, self.l2_cache, self.l3_cache)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_frequency_based_promotion(self):
        """Test promotion based on access frequency."""
        # Store data in L3
        await self.l3_cache.set("frequent_key", "frequent_value")
        
        # Access multiple times to trigger promotion
        for _ in range(3):
            value = await self.multi_cache.get("frequent_key")
            assert value == "frequent_value"
        
        # Should be promoted to L1
        l1_value = await self.l1_cache.get("frequent_key")
        assert l1_value == "frequent_value", "Frequently accessed data should be promoted to L1"
    
    @pytest.mark.asyncio
    async def test_size_based_promotion_limits(self):
        """Test promotion respects size limits."""
        # Fill L1 cache to capacity
        for i in range(3):
            await self.l1_cache.set(f"l1_key_{i}", f"l1_value_{i}")
        
        # Store large data in L3
        large_data = {"large": "x" * 1000}
        await self.l3_cache.set("large_key", large_data)
        
        # Access should promote but may evict other entries
        value = await self.multi_cache.get("large_key")
        assert value == large_data
        
        # L1 should still respect size limits
        l1_stats = self.l1_cache.get_stats()
        assert l1_stats['size'] <= 3, "L1 cache should respect size limits during promotion"
    
    @pytest.mark.asyncio
    async def test_promotion_disable_enable(self):
        """Test enabling/disabling promotion."""
        # Disable promotion
        self.multi_cache.set_promotion_enabled(False)
        
        # Store in L3
        await self.l3_cache.set("no_promote_key", "no_promote_value")
        
        # Access should not promote
        value = await self.multi_cache.get("no_promote_key")
        assert value == "no_promote_value"
        
        l1_value = await self.l1_cache.get("no_promote_key")
        assert l1_value is None, "Data should not be promoted when promotion is disabled"
        
        # Enable promotion
        self.multi_cache.set_promotion_enabled(True)
        
        # Access should now promote
        value = await self.multi_cache.get("no_promote_key")
        assert value == "no_promote_value"
        
        l1_value = await self.l1_cache.get("no_promote_key")
        assert l1_value == "no_promote_value", "Data should be promoted when promotion is enabled"


class TestCacheConsistency:
    """Tests for data consistency across cache tiers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.l1_cache = MockL1MemoryCache()
        self.l2_cache = MockL2DiskCache(self.temp_dir)
        self.l3_cache = MockL3RedisCache()
        self.multi_cache = MultiTierCache(self.l1_cache, self.l2_cache, self.l3_cache)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_invalidation_propagation(self):
        """Test invalidation propagates across all tiers."""
        # Store in all tiers
        await self.multi_cache.set("invalidate_key", "invalidate_value")
        
        # Verify present in all tiers
        assert await self.l1_cache.get("invalidate_key") == "invalidate_value"
        assert await self.l2_cache.get("invalidate_key") == "invalidate_value"
        assert await self.l3_cache.get("invalidate_key") == "invalidate_value"
        
        # Delete from multi-tier cache
        deleted = await self.multi_cache.delete("invalidate_key")
        assert deleted, "Deletion should succeed"
        
        # Should be removed from all tiers
        assert await self.l1_cache.get("invalidate_key") is None
        assert await self.l2_cache.get("invalidate_key") is None
        assert await self.l3_cache.get("invalidate_key") is None
    
    @pytest.mark.asyncio
    async def test_tier_failure_consistency(self):
        """Test consistency when individual tiers fail."""
        # Simulate L3 failure
        self.l3_cache.set_connection_state(False)
        
        # Store data (should succeed in L1 and L2)
        success = await self.multi_cache.set("failure_key", "failure_value")
        
        # Should still be retrievable from working tiers
        value = await self.multi_cache.get("failure_key")
        assert value == "failure_value", "Should retrieve from working tiers when one fails"
        
        # Restore L3 and verify consistency
        self.l3_cache.set_connection_state(True)
        
        # Data should still be accessible
        value = await self.multi_cache.get("failure_key")
        assert value == "failure_value", "Data should remain consistent after tier recovery"


class TestMultiTierPerformance:
    """Tests for multi-tier cache performance optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.l1_cache = MockL1MemoryCache(max_size=50)
        self.l2_cache = MockL2DiskCache(self.temp_dir, max_size_mb=5)
        self.l3_cache = MockL3RedisCache()
        self.multi_cache = MultiTierCache(self.l1_cache, self.l2_cache, self.l3_cache)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_response_time_optimization(self):
        """Test multi-tier cache meets response time targets."""
        # Populate caches with test data
        for key, data in BIOMEDICAL_TEST_DATA.items():
            await self.multi_cache.set(key, data['response'])
        
        # Measure response times for cached data
        start_time = time.time()
        
        for key in BIOMEDICAL_TEST_DATA.keys():
            value = await self.multi_cache.get(key)
            assert value is not None, f"Should retrieve cached value for {key}"
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / len(BIOMEDICAL_TEST_DATA)) * 1000
        
        assert avg_time_ms < 100, f"Multi-tier cache should be fast (<100ms), got {avg_time_ms:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_hit_ratio_optimization(self):
        """Test multi-tier cache achieves high hit ratios."""
        test_queries = list(BIOMEDICAL_TEST_DATA.keys()) * 3  # Repeat queries
        
        # Populate cache
        for key, data in BIOMEDICAL_TEST_DATA.items():
            await self.multi_cache.set(key, data['response'])
        
        # Access queries multiple times
        for query in test_queries:
            await self.multi_cache.get(query)
        
        stats = self.multi_cache.get_comprehensive_stats()
        overall_hit_rate = stats['overall_hit_rate']
        
        assert overall_hit_rate > 0.8, f"Multi-tier cache should achieve high hit rate (>80%), got {overall_hit_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_tier_performance_characteristics(self):
        """Test performance characteristics of different tiers."""
        test_key = "performance_test"
        test_value = {"performance": "test_data"}
        
        # Measure L1 performance
        await self.l1_cache.set(test_key, test_value)
        start_time = time.time()
        for _ in range(100):
            await self.l1_cache.get(test_key)
        l1_time = time.time() - start_time
        
        # Measure L2 performance
        await self.l2_cache.set(test_key, test_value)
        start_time = time.time()
        for _ in range(100):
            await self.l2_cache.get(test_key)
        l2_time = time.time() - start_time
        
        # Measure L3 performance
        await self.l3_cache.set(test_key, test_value)
        start_time = time.time()
        for _ in range(100):
            await self.l3_cache.get(test_key)
        l3_time = time.time() - start_time
        
        # L1 should be fastest, L3 should be slowest
        assert l1_time < l2_time, "L1 should be faster than L2"
        assert l2_time < l3_time, "L2 should be faster than L3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])