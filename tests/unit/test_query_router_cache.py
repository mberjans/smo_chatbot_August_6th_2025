"""
Unit tests for query router LRU cache functionality.

This module tests the query router's LRU cache system that stores
high-confidence routing decisions to improve performance and reduce
redundant classification operations.

Test Coverage:
- LRU eviction policy implementation
- Query hash consistency for cache keys
- Confidence-based caching decisions
- Cache invalidation on routing logic updates
- Performance impact measurement
- Thread safety and concurrent access
- Memory usage optimization
- Cache warming strategies

Classes:
    TestQueryRouterLRUCache: Tests for LRU cache functionality
    TestQueryHashConsistency: Tests for consistent query hashing
    TestConfidenceBasedCaching: Tests for confidence threshold caching
    TestCacheInvalidation: Tests for cache invalidation mechanisms
    TestQueryRouterPerformance: Tests for performance improvements
    TestCacheThreadSafety: Tests for concurrent access safety
    TestCacheMemoryManagement: Tests for memory usage optimization

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import hashlib
import gc
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
from enum import Enum
import concurrent.futures

# Query routing test data
class RoutingDecision(Enum):
    """Routing decision options."""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"
    FALLBACK = "fallback"
    DIRECT_RESPONSE = "direct"

@dataclass
class RoutingPrediction:
    """Routing prediction with confidence metrics."""
    decision: RoutingDecision
    confidence: float
    reasoning: str
    timestamp: float
    processing_time_ms: float
    features: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['decision'] = self.decision.value
        return result

# Test data for query router caching
QUERY_ROUTER_TEST_DATA = [
    {
        'query': 'What are the metabolic pathways involved in glucose metabolism?',
        'routing_decision': RoutingDecision.LIGHTRAG,
        'confidence': 0.95,
        'reasoning': 'High biomedical content, suitable for RAG system',
        'should_cache': True,
        'category': 'biomedical_factual'
    },
    {
        'query': 'How does insulin resistance affect metabolomics profiles?',
        'routing_decision': RoutingDecision.LIGHTRAG,
        'confidence': 0.92,
        'reasoning': 'Complex biomedical query requiring knowledge base',
        'should_cache': True,
        'category': 'biomedical_complex'
    },
    {
        'query': 'Latest COVID-19 metabolomics research 2024',
        'routing_decision': RoutingDecision.PERPLEXITY,
        'confidence': 0.88,
        'reasoning': 'Temporal query requiring current information',
        'should_cache': False,  # Temporal queries shouldn't be cached
        'category': 'temporal'
    },
    {
        'query': 'What is 2+2?',
        'routing_decision': RoutingDecision.DIRECT_RESPONSE,
        'confidence': 0.99,
        'reasoning': 'Simple arithmetic, direct answer',
        'should_cache': True,
        'category': 'simple_factual'
    },
    {
        'query': 'Tell me about current drug prices',
        'routing_decision': RoutingDecision.PERPLEXITY,
        'confidence': 0.75,
        'reasoning': 'Current market information needed',
        'should_cache': False,  # Market data changes frequently
        'category': 'market_data'
    },
    {
        'query': 'Explain diabetes pathophysiology',
        'routing_decision': RoutingDecision.LIGHTRAG,
        'confidence': 0.91,
        'reasoning': 'Medical education content in knowledge base',
        'should_cache': True,
        'category': 'medical_education'
    },
    {
        'query': 'Low confidence biomedical query example',
        'routing_decision': RoutingDecision.FALLBACK,
        'confidence': 0.45,  # Below typical caching threshold
        'reasoning': 'Uncertain classification, use fallback',
        'should_cache': False,  # Low confidence shouldn't be cached
        'category': 'uncertain'
    }
]

# Performance test data
PERFORMANCE_TEST_QUERIES = [
    f"Test query for performance measurement {i}"
    for i in range(1000)
]

@dataclass
class CacheEntry:
    """Query router cache entry."""
    query_hash: str
    routing_prediction: RoutingPrediction
    access_count: int
    last_access: float
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_hash': self.query_hash,
            'routing_prediction': self.routing_prediction.to_dict(),
            'access_count': self.access_count,
            'last_access': self.last_access,
            'created_at': self.created_at
        }


class QueryRouterLRUCache:
    """LRU cache for query routing decisions."""
    
    def __init__(self, max_size: int = 1000, confidence_threshold: float = 0.7):
        self.max_size = max_size
        self.confidence_threshold = confidence_threshold
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0,
            'low_confidence_skips': 0,
            'total_queries': 0
        }
        
        self._lock = threading.RLock()
        self._routing_version = 1  # Track routing logic version for invalidation
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate consistent hash for query."""
        # Normalize query text
        normalized = ' '.join(query.lower().split())  # Normalize whitespace
        
        # Generate hash
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
    
    def _enforce_lru_policy(self):
        """Enforce LRU eviction policy."""
        while len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key, lru_entry = self.cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def _should_cache_prediction(self, prediction: RoutingPrediction) -> bool:
        """Determine if prediction should be cached based on confidence."""
        # Don't cache low confidence predictions
        if prediction.confidence < self.confidence_threshold:
            self.stats['low_confidence_skips'] += 1
            return False
        
        # Don't cache temporal queries (these change over time)
        temporal_keywords = ['latest', 'current', 'recent', '2024', '2025', 'today']
        if any(keyword in prediction.reasoning.lower() for keyword in temporal_keywords):
            return False
        
        # Cache high-confidence, stable predictions
        return True
    
    async def get_cached_routing(self, query: str) -> Optional[RoutingPrediction]:
        """Get cached routing decision for query."""
        with self._lock:
            self.stats['total_queries'] += 1
            query_hash = self._generate_query_hash(query)
            
            if query_hash in self.cache:
                entry = self.cache[query_hash]
                
                # Move to end (mark as most recently used)
                self.cache.move_to_end(query_hash)
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                
                self.stats['hits'] += 1
                return entry.routing_prediction
            
            self.stats['misses'] += 1
            return None
    
    async def cache_routing_decision(self, query: str, prediction: RoutingPrediction) -> bool:
        """Cache routing decision if it meets criteria."""
        with self._lock:
            # Check if should cache
            if not self._should_cache_prediction(prediction):
                return False
            
            query_hash = self._generate_query_hash(query)
            
            # Enforce size limits
            self._enforce_lru_policy()
            
            # Create cache entry
            entry = CacheEntry(
                query_hash=query_hash,
                routing_prediction=prediction,
                access_count=0,
                last_access=time.time(),
                created_at=time.time()
            )
            
            # Store in cache
            self.cache[query_hash] = entry
            
            return True
    
    def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern or all entries."""
        with self._lock:
            if pattern is None:
                # Invalidate all
                invalidated_count = len(self.cache)
                self.cache.clear()
            else:
                # Invalidate matching pattern
                keys_to_remove = []
                for key, entry in self.cache.items():
                    if pattern.lower() in entry.routing_prediction.reasoning.lower():
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.cache[key]
                
                invalidated_count = len(keys_to_remove)
            
            self.stats['invalidations'] += invalidated_count
            self._routing_version += 1
            
            return invalidated_count
    
    def update_routing_version(self):
        """Update routing version to trigger cache invalidation."""
        with self._lock:
            self._routing_version += 1
            # In a real implementation, this might invalidate entries
            # based on version mismatch
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            # Memory usage estimation
            estimated_memory_kb = len(self.cache) * 0.5  # Rough estimate
            
            return {
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'invalidations': self.stats['invalidations'],
                'low_confidence_skips': self.stats['low_confidence_skips'],
                'total_queries': self.stats['total_queries'],
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'confidence_threshold': self.confidence_threshold,
                'routing_version': self._routing_version,
                'estimated_memory_kb': estimated_memory_kb,
                'utilization': len(self.cache) / self.max_size
            }
    
    def get_cache_contents(self) -> List[Dict[str, Any]]:
        """Get current cache contents for debugging."""
        with self._lock:
            return [entry.to_dict() for entry in self.cache.values()]
    
    def clear_cache(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            # Reset stats except version
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'invalidations': 0,
                'low_confidence_skips': 0,
                'total_queries': 0
            }


class TestQueryRouterLRUCache:
    """Tests for query router LRU cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache(max_size=5, confidence_threshold=0.7)
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self):
        """Test basic cache set and get operations."""
        query = "test biomedical query"
        prediction = RoutingPrediction(
            decision=RoutingDecision.LIGHTRAG,
            confidence=0.9,
            reasoning="High biomedical content",
            timestamp=time.time(),
            processing_time_ms=50.0
        )
        
        # Should cache high-confidence prediction
        cached = await self.cache.cache_routing_decision(query, prediction)
        assert cached, "Should cache high-confidence prediction"
        
        # Should retrieve cached prediction
        retrieved = await self.cache.get_cached_routing(query)
        assert retrieved is not None, "Should retrieve cached prediction"
        assert retrieved.decision == RoutingDecision.LIGHTRAG
        assert retrieved.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_lru_eviction_policy(self):
        """Test LRU eviction when cache reaches capacity."""
        # Fill cache to capacity (max_size = 5)
        queries = []
        for i in range(5):
            query = f"test query {i}"
            queries.append(query)
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"Test reasoning {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        stats = self.cache.get_cache_statistics()
        assert stats['cache_size'] == 5, "Cache should be at capacity"
        
        # Access first query to make it recently used
        await self.cache.get_cached_routing(queries[0])
        
        # Add new query, should evict LRU (queries[1])
        new_query = "new test query"
        new_prediction = RoutingPrediction(
            decision=RoutingDecision.PERPLEXITY,
            confidence=0.85,
            reasoning="New test reasoning",
            timestamp=time.time(),
            processing_time_ms=60.0
        )
        await self.cache.cache_routing_decision(new_query, new_prediction)
        
        # Verify eviction occurred
        stats = self.cache.get_cache_statistics()
        assert stats['cache_size'] == 5, "Cache should maintain size limit"
        assert stats['evictions'] == 1, "Should track eviction"
        
        # Recently used query should still be cached
        retrieved = await self.cache.get_cached_routing(queries[0])
        assert retrieved is not None, "Recently used entry should remain"
        
        # LRU query should be evicted
        retrieved = await self.cache.get_cached_routing(queries[1])
        assert retrieved is None, "LRU entry should be evicted"
        
        # New query should be cached
        retrieved = await self.cache.get_cached_routing(new_query)
        assert retrieved is not None, "New entry should be cached"
    
    @pytest.mark.asyncio
    async def test_confidence_based_caching(self):
        """Test caching decisions based on confidence thresholds."""
        # High confidence - should cache
        high_conf_query = "high confidence query"
        high_conf_prediction = RoutingPrediction(
            decision=RoutingDecision.LIGHTRAG,
            confidence=0.9,  # Above threshold (0.7)
            reasoning="High confidence reasoning",
            timestamp=time.time(),
            processing_time_ms=40.0
        )
        
        cached = await self.cache.cache_routing_decision(high_conf_query, high_conf_prediction)
        assert cached, "Should cache high confidence prediction"
        
        # Low confidence - should not cache
        low_conf_query = "low confidence query"
        low_conf_prediction = RoutingPrediction(
            decision=RoutingDecision.FALLBACK,
            confidence=0.5,  # Below threshold (0.7)
            reasoning="Low confidence reasoning",
            timestamp=time.time(),
            processing_time_ms=80.0
        )
        
        cached = await self.cache.cache_routing_decision(low_conf_query, low_conf_prediction)
        assert not cached, "Should not cache low confidence prediction"
        
        # Verify statistics
        stats = self.cache.get_cache_statistics()
        assert stats['low_confidence_skips'] == 1, "Should track low confidence skips"
    
    @pytest.mark.asyncio
    async def test_temporal_query_filtering(self):
        """Test that temporal queries are not cached."""
        temporal_queries = [
            ("latest metabolomics research", "latest research updates"),
            ("current drug prices 2024", "current market information"),
            ("recent COVID studies", "recent study results"),
            ("today's biomarker news", "today's news updates")
        ]
        
        for query, reasoning in temporal_queries:
            prediction = RoutingPrediction(
                decision=RoutingDecision.PERPLEXITY,
                confidence=0.85,  # High confidence
                reasoning=reasoning,
                timestamp=time.time(),
                processing_time_ms=60.0
            )
            
            cached = await self.cache.cache_routing_decision(query, prediction)
            assert not cached, f"Should not cache temporal query: {query}"
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation functionality."""
        # Cache several predictions
        test_data = QUERY_ROUTER_TEST_DATA[:3]  # Use first 3 entries
        
        for data in test_data:
            if data['should_cache']:
                prediction = RoutingPrediction(
                    decision=data['routing_decision'],
                    confidence=data['confidence'],
                    reasoning=data['reasoning'],
                    timestamp=time.time(),
                    processing_time_ms=50.0
                )
                await self.cache.cache_routing_decision(data['query'], prediction)
        
        initial_stats = self.cache.get_cache_statistics()
        initial_size = initial_stats['cache_size']
        
        # Invalidate entries matching pattern
        invalidated = self.cache.invalidate_cache(pattern="biomedical")
        
        # Should invalidate biomedical entries
        assert invalidated > 0, "Should invalidate matching entries"
        
        final_stats = self.cache.get_cache_statistics()
        assert final_stats['cache_size'] < initial_size, "Cache size should decrease"
        assert final_stats['invalidations'] == invalidated, "Should track invalidations"
        assert final_stats['routing_version'] > initial_stats['routing_version'], "Should increment version"
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_optimization(self):
        """Test cache improves hit rates for repeated queries."""
        test_queries = [data['query'] for data in QUERY_ROUTER_TEST_DATA if data['should_cache']]
        
        # First pass - cache predictions
        for query in test_queries:
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.85,
                reasoning="Test reasoning",
                timestamp=time.time(),
                processing_time_ms=100.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        initial_stats = self.cache.get_cache_statistics()
        initial_hits = initial_stats['hits']
        
        # Second pass - should hit cache
        hit_count = 0
        for query in test_queries:
            result = await self.cache.get_cached_routing(query)
            if result is not None:
                hit_count += 1
        
        final_stats = self.cache.get_cache_statistics()
        
        assert hit_count > 0, "Should get cache hits on repeated queries"
        assert final_stats['hits'] > initial_hits, "Hit count should increase"
        
        # Hit rate should be high for repeated queries
        assert final_stats['hit_rate'] > 0.5, "Hit rate should be reasonable for repeated queries"


class TestQueryHashConsistency:
    """Tests for consistent query hashing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache()
    
    def test_hash_consistency(self):
        """Test identical queries generate identical hashes."""
        query = "What are the metabolic pathways involved in glucose metabolism?"
        
        hash1 = self.cache._generate_query_hash(query)
        hash2 = self.cache._generate_query_hash(query)
        
        assert hash1 == hash2, "Identical queries should generate identical hashes"
        assert len(hash1) == 16, "Hash should be 16 characters"
    
    def test_hash_normalization(self):
        """Test query normalization produces consistent hashes."""
        base_query = "glucose metabolism pathways"
        variations = [
            "glucose metabolism pathways",
            "  glucose   metabolism   pathways  ",
            "glucose\tmetabolism\npathways",
            "Glucose Metabolism Pathways",
            "GLUCOSE METABOLISM PATHWAYS"
        ]
        
        hashes = [self.cache._generate_query_hash(q) for q in variations]
        unique_hashes = set(hashes)
        
        assert len(unique_hashes) == 1, "Query variations should normalize to same hash"
    
    def test_hash_uniqueness(self):
        """Test different queries generate different hashes."""
        queries = [
            "glucose metabolism",
            "insulin resistance", 
            "diabetes pathophysiology",
            "metabolomics analysis",
            "biomarker discovery"
        ]
        
        hashes = [self.cache._generate_query_hash(q) for q in queries]
        unique_hashes = set(hashes)
        
        assert len(hashes) == len(unique_hashes), "Different queries should generate unique hashes"
    
    def test_hash_collision_resistance(self):
        """Test hash collision resistance with similar queries."""
        similar_queries = [
            "glucose metabolism in diabetes",
            "glucose metabolism and diabetes",
            "diabetes glucose metabolism",
            "metabolic glucose in diabetes",
            "diabetic glucose metabolism"
        ]
        
        hashes = [self.cache._generate_query_hash(q) for q in similar_queries]
        unique_hashes = set(hashes)
        
        assert len(hashes) == len(unique_hashes), "Similar queries should generate unique hashes"


class TestConfidenceBasedCaching:
    """Tests for confidence-based caching decisions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache(confidence_threshold=0.75)
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_enforcement(self):
        """Test confidence threshold is properly enforced."""
        test_cases = [
            (0.9, True, "Very high confidence should be cached"),
            (0.8, True, "High confidence should be cached"),
            (0.75, True, "Threshold confidence should be cached"),
            (0.74, False, "Just below threshold should not be cached"),
            (0.6, False, "Medium confidence should not be cached"),
            (0.4, False, "Low confidence should not be cached")
        ]
        
        for confidence, should_cache, message in test_cases:
            query = f"test query with confidence {confidence}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=confidence,
                reasoning="Test reasoning",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            
            cached = await self.cache.cache_routing_decision(query, prediction)
            assert cached == should_cache, message
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_adjustment(self):
        """Test adjusting confidence threshold affects caching."""
        query = "test query"
        prediction = RoutingPrediction(
            decision=RoutingDecision.LIGHTRAG,
            confidence=0.7,
            reasoning="Test reasoning",
            timestamp=time.time(),
            processing_time_ms=50.0
        )
        
        # With threshold 0.75, should not cache
        cache_high_threshold = QueryRouterLRUCache(confidence_threshold=0.75)
        cached = await cache_high_threshold.cache_routing_decision(query, prediction)
        assert not cached, "Should not cache with high threshold"
        
        # With threshold 0.6, should cache
        cache_low_threshold = QueryRouterLRUCache(confidence_threshold=0.6)
        cached = await cache_low_threshold.cache_routing_decision(query, prediction)
        assert cached, "Should cache with low threshold"
    
    @pytest.mark.asyncio
    async def test_confidence_statistics_tracking(self):
        """Test confidence-based statistics are tracked."""
        # Try to cache several low-confidence predictions
        for i in range(5):
            query = f"low confidence query {i}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.FALLBACK,
                confidence=0.5,  # Below threshold
                reasoning="Low confidence reasoning",
                timestamp=time.time(),
                processing_time_ms=100.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        stats = self.cache.get_cache_statistics()
        assert stats['low_confidence_skips'] == 5, "Should track low confidence skips"
        assert stats['cache_size'] == 0, "No entries should be cached"


class TestCacheInvalidation:
    """Tests for cache invalidation mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache(max_size=10)
    
    @pytest.mark.asyncio
    async def test_pattern_based_invalidation(self):
        """Test pattern-based cache invalidation."""
        # Cache entries with different reasoning patterns
        test_entries = [
            ("biomedical query 1", "biomedical content analysis"),
            ("biomedical query 2", "medical knowledge required"),
            ("temporal query", "current information needed"),
            ("general query", "general purpose response")
        ]
        
        for query, reasoning in test_entries:
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.85,
                reasoning=reasoning,
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        initial_size = self.cache.get_cache_statistics()['cache_size']
        
        # Invalidate entries with "biomedical" pattern
        invalidated = self.cache.invalidate_cache(pattern="biomedical")
        
        assert invalidated == 1, "Should invalidate one biomedical entry"  # Only first one has "biomedical" in reasoning
        
        final_size = self.cache.get_cache_statistics()['cache_size']
        assert final_size == initial_size - invalidated, "Cache size should decrease by invalidated count"
    
    @pytest.mark.asyncio
    async def test_full_cache_invalidation(self):
        """Test complete cache invalidation."""
        # Cache several entries
        for i in range(5):
            query = f"test query {i}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"Test reasoning {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        assert self.cache.get_cache_statistics()['cache_size'] == 5, "Should have 5 cached entries"
        
        # Invalidate all
        invalidated = self.cache.invalidate_cache()
        
        assert invalidated == 5, "Should invalidate all entries"
        assert self.cache.get_cache_statistics()['cache_size'] == 0, "Cache should be empty"
        assert self.cache.get_cache_statistics()['invalidations'] == 5, "Should track invalidations"
    
    def test_routing_version_updates(self):
        """Test routing version updates for cache invalidation."""
        initial_version = self.cache.get_cache_statistics()['routing_version']
        
        self.cache.update_routing_version()
        
        updated_version = self.cache.get_cache_statistics()['routing_version']
        assert updated_version > initial_version, "Routing version should increment"


class TestQueryRouterPerformance:
    """Tests for query router cache performance improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache(max_size=100)
    
    @pytest.mark.asyncio
    async def test_cache_improves_performance(self):
        """Test cache improves query routing performance."""
        test_queries = PERFORMANCE_TEST_QUERIES[:50]  # Use subset for testing
        
        # Simulate initial routing with cache population
        initial_times = []
        for query in test_queries:
            start_time = time.time()
            
            # Simulate routing decision
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning="Cached test reasoning",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            
            await self.cache.cache_routing_decision(query, prediction)
            
            end_time = time.time()
            initial_times.append(end_time - start_time)
        
        # Measure cached performance
        cached_times = []
        cache_hits = 0
        
        for query in test_queries:
            start_time = time.time()
            
            cached_result = await self.cache.get_cached_routing(query)
            if cached_result:
                cache_hits += 1
            
            end_time = time.time()
            cached_times.append(end_time - start_time)
        
        # Cache should provide faster access
        avg_initial_time = sum(initial_times) / len(initial_times)
        avg_cached_time = sum(cached_times) / len(cached_times)
        
        assert avg_cached_time < avg_initial_time, "Cached access should be faster"
        assert cache_hits > 0, "Should get cache hits"
        
        # High hit rate indicates good caching performance
        hit_rate = cache_hits / len(test_queries)
        assert hit_rate > 0.8, f"Hit rate should be high for repeated queries: {hit_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self):
        """Test cache memory usage is reasonable."""
        # Fill cache with realistic data
        for i, query in enumerate(PERFORMANCE_TEST_QUERIES[:100]):
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"Test reasoning for query {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        stats = self.cache.get_cache_statistics()
        
        # Memory usage should be reasonable
        assert stats['estimated_memory_kb'] < 100, "Memory usage should be reasonable"
        assert stats['cache_size'] <= self.cache.max_size, "Should not exceed max size"
        assert stats['utilization'] <= 1.0, "Utilization should not exceed 100%"
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self):
        """Test cache performance under concurrent load."""
        async def query_worker(worker_id: int, query_count: int):
            """Worker that performs cache operations."""
            hits = 0
            total_time = 0
            
            for i in range(query_count):
                query = f"worker_{worker_id}_query_{i}"
                
                start_time = time.time()
                
                # Try cache first
                cached = await self.cache.get_cached_routing(query)
                
                if cached is None:
                    # Simulate routing and cache
                    prediction = RoutingPrediction(
                        decision=RoutingDecision.LIGHTRAG,
                        confidence=0.8,
                        reasoning=f"Worker {worker_id} reasoning",
                        timestamp=time.time(),
                        processing_time_ms=50.0
                    )
                    await self.cache.cache_routing_decision(query, prediction)
                else:
                    hits += 1
                
                end_time = time.time()
                total_time += (end_time - start_time)
            
            return hits, total_time
        
        # Run concurrent workers
        workers = [query_worker(i, 20) for i in range(5)]
        results = await asyncio.gather(*workers)
        
        total_hits = sum(r[0] for r in results)
        total_time = sum(r[1] for r in results)
        total_queries = 5 * 20  # 5 workers * 20 queries each
        
        avg_time_per_query = total_time / total_queries
        
        # Performance should remain good under load
        assert avg_time_per_query < 0.01, f"Average query time should be <10ms: {avg_time_per_query*1000:.1f}ms"
        
        # Should handle concurrent access without errors
        final_stats = self.cache.get_cache_statistics()
        assert final_stats['total_queries'] > 0, "Should track all queries"


class TestCacheThreadSafety:
    """Tests for cache thread safety under concurrent access."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache(max_size=50)
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write_operations(self):
        """Test concurrent cache read and write operations."""
        errors = []
        
        async def writer_worker():
            """Worker that writes to cache."""
            try:
                for i in range(20):
                    query = f"writer_query_{i}"
                    prediction = RoutingPrediction(
                        decision=RoutingDecision.LIGHTRAG,
                        confidence=0.8,
                        reasoning=f"Writer reasoning {i}",
                        timestamp=time.time(),
                        processing_time_ms=50.0
                    )
                    await self.cache.cache_routing_decision(query, prediction)
                    await asyncio.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Writer error: {e}")
        
        async def reader_worker():
            """Worker that reads from cache."""
            try:
                for i in range(20):
                    query = f"writer_query_{i}"
                    await self.cache.get_cached_routing(query)
                    await asyncio.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Reader error: {e}")
        
        # Run concurrent workers
        await asyncio.gather(
            writer_worker(), writer_worker(),
            reader_worker(), reader_worker(), reader_worker()
        )
        
        assert len(errors) == 0, f"Concurrent operations should not cause errors: {errors}"
        
        # Cache should maintain consistent state
        stats = self.cache.get_cache_statistics()
        assert stats['cache_size'] >= 0, "Cache size should be valid"
        assert stats['hits'] + stats['misses'] > 0, "Should have processed queries"
    
    @pytest.mark.asyncio
    async def test_concurrent_eviction_scenarios(self):
        """Test cache behavior during concurrent eviction scenarios."""
        errors = []
        
        async def cache_filler():
            """Fill cache to trigger evictions."""
            try:
                for i in range(100):  # More than max_size to trigger evictions
                    query = f"eviction_query_{i}"
                    prediction = RoutingPrediction(
                        decision=RoutingDecision.LIGHTRAG,
                        confidence=0.8,
                        reasoning=f"Eviction test {i}",
                        timestamp=time.time(),
                        processing_time_ms=50.0
                    )
                    await self.cache.cache_routing_decision(query, prediction)
            except Exception as e:
                errors.append(f"Filler error: {e}")
        
        async def cache_reader():
            """Read from cache during evictions."""
            try:
                for i in range(50):
                    query = f"eviction_query_{i}"
                    await self.cache.get_cached_routing(query)
                    await asyncio.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader error: {e}")
        
        # Run concurrent operations
        await asyncio.gather(cache_filler(), cache_reader(), cache_reader())
        
        assert len(errors) == 0, f"Concurrent eviction should not cause errors: {errors}"
        
        # Cache should respect size limits
        stats = self.cache.get_cache_statistics()
        assert stats['cache_size'] <= self.cache.max_size, "Should maintain size limits"
        assert stats['evictions'] > 0, "Should track evictions"
    
    def test_thread_safe_statistics(self):
        """Test statistics remain consistent under concurrent access."""
        def statistics_worker():
            """Worker that accesses statistics."""
            for _ in range(100):
                stats = self.cache.get_cache_statistics()
                # Verify statistics are internally consistent
                assert stats['hits'] >= 0
                assert stats['misses'] >= 0
                assert stats['cache_size'] >= 0
                assert stats['cache_size'] <= stats['max_size']
        
        # Run multiple threads accessing statistics
        threads = [threading.Thread(target=statistics_worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        final_stats = self.cache.get_cache_statistics()
        assert final_stats is not None, "Statistics should remain accessible"


class TestCacheMemoryManagement:
    """Tests for cache memory usage optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryRouterLRUCache(max_size=100)
    
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self):
        """Test memory usage scales reasonably with cache size."""
        # Start with empty cache
        empty_stats = self.cache.get_cache_statistics()
        empty_memory = empty_stats['estimated_memory_kb']
        
        # Fill cache partially
        for i in range(50):
            query = f"memory_test_query_{i}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"Memory test reasoning {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        half_full_stats = self.cache.get_cache_statistics()
        half_full_memory = half_full_stats['estimated_memory_kb']
        
        # Fill cache completely
        for i in range(50, 100):
            query = f"memory_test_query_{i}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"Memory test reasoning {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        full_stats = self.cache.get_cache_statistics()
        full_memory = full_stats['estimated_memory_kb']
        
        # Memory should scale reasonably
        assert half_full_memory > empty_memory, "Memory usage should increase with entries"
        assert full_memory > half_full_memory, "Memory usage should continue to increase"
        assert full_memory < 1000, "Memory usage should remain reasonable (<1MB)"
    
    @pytest.mark.asyncio 
    async def test_garbage_collection_impact(self):
        """Test cache performance with garbage collection."""
        # Fill cache
        for i in range(100):
            query = f"gc_test_query_{i}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"GC test reasoning {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            await self.cache.cache_routing_decision(query, prediction)
        
        # Force garbage collection
        gc.collect()
        
        # Cache should still function correctly
        test_query = "gc_test_query_50"
        result = await self.cache.get_cached_routing(test_query)
        assert result is not None, "Cache should survive garbage collection"
        
        stats = self.cache.get_cache_statistics()
        assert stats['cache_size'] > 0, "Cache should retain entries after GC"
    
    def test_cache_clear_memory_cleanup(self):
        """Test cache clear properly frees memory."""
        # Fill cache
        for i in range(50):
            query = f"clear_test_query_{i}"
            prediction = RoutingPrediction(
                decision=RoutingDecision.LIGHTRAG,
                confidence=0.8,
                reasoning=f"Clear test reasoning {i}",
                timestamp=time.time(),
                processing_time_ms=50.0
            )
            asyncio.run(self.cache.cache_routing_decision(query, prediction))
        
        filled_stats = self.cache.get_cache_statistics()
        assert filled_stats['cache_size'] == 50, "Should have 50 entries"
        
        # Clear cache
        self.cache.clear_cache()
        
        cleared_stats = self.cache.get_cache_statistics()
        assert cleared_stats['cache_size'] == 0, "Should have no entries after clear"
        assert cleared_stats['estimated_memory_kb'] < filled_stats['estimated_memory_kb'], \
            "Memory usage should decrease after clear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])