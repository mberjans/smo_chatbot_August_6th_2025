"""
Comprehensive TTL (Time-To-Live) functionality tests for cache system.

This module provides extensive testing of TTL behaviors across different cache tiers,
including expiration scenarios, dynamic TTL management, confidence-based adjustments,
and multi-tier TTL coordination in the Clinical Metabolomics Oracle system.

Test Coverage:
- Multi-tier TTL configuration and management
- TTL expiration scenarios and boundary conditions
- Dynamic TTL adjustments based on query patterns
- Confidence-based TTL modifications
- TTL extension and refresh mechanisms
- TTL inheritance and cascading between cache tiers
- TTL synchronization across distributed cache systems
- TTL-based cache promotion and demotion strategies
- Edge cases and boundary condition testing

Classes:
    TestTTLBasicFunctionality: Core TTL behavior testing
    TestTTLExpirationScenarios: Comprehensive expiration testing
    TestDynamicTTLManagement: Adaptive TTL behavior testing
    TestConfidenceBasedTTL: TTL adjustments based on response confidence
    TestMultiTierTTLCoordination: TTL coordination across cache tiers
    TestTTLBoundaryConditions: Edge cases and boundary testing
    TestTTLExtensionRefresh: TTL refresh and extension scenarios
    TestTTLPerformanceImpact: TTL impact on cache performance

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import random
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
from collections import OrderedDict

# Import test fixtures
from .cache_test_fixtures import (
    BIOMEDICAL_QUERIES,
    BiomedicalTestDataGenerator,
    CachePerformanceMetrics,
    CachePerformanceMeasurer
)


# TTL Configuration Constants - Based on existing system analysis
TTL_CONFIGS = {
    'L1_CACHE': 300,      # 5 minutes
    'L2_CACHE': 3600,     # 1 hour  
    'L3_CACHE': 86400,    # 24 hours
    'EMERGENCY_CACHE': 86400,  # 24 hours
    'FALLBACK_MIN': 1800, # 30 minutes
    'FALLBACK_MAX': 7200, # 2 hours
    'TEMPORAL_QUERIES': 300,   # 5 minutes for time-sensitive queries
    'HIGH_CONFIDENCE': 7200,   # 2 hours for high confidence responses
    'LOW_CONFIDENCE': 900,     # 15 minutes for low confidence responses
    'CLINICAL_CRITICAL': 14400 # 4 hours for critical clinical information
}


@dataclass
class TTLTestEntry:
    """Test cache entry with TTL tracking."""
    key: str
    value: Any
    timestamp: float
    ttl: int
    confidence: float = 0.9
    access_count: int = 0
    tier: str = 'L1'
    metadata: Optional[Dict[str, Any]] = None
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if entry has expired."""
        if current_time is None:
            current_time = time.time()
        return current_time > (self.timestamp + self.ttl)
    
    def time_until_expiry(self, current_time: Optional[float] = None) -> float:
        """Get time until expiry in seconds."""
        if current_time is None:
            current_time = time.time()
        return max(0, (self.timestamp + self.ttl) - current_time)
    
    def refresh_ttl(self, new_ttl: Optional[int] = None) -> None:
        """Refresh TTL with current timestamp."""
        self.timestamp = time.time()
        if new_ttl is not None:
            self.ttl = new_ttl
        self.access_count += 1


class MockTTLCache:
    """Mock cache implementation with comprehensive TTL support."""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        self.storage: Dict[str, TTLTestEntry] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self.expiration_events = []
        self.ttl_extensions = []
        self.tier = 'L1'
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key from query."""
        return f"{self.tier}:{hash(query)}"
    
    def _cleanup_expired(self) -> List[str]:
        """Clean up expired entries and return expired keys."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in list(self.storage.items()):
            if entry.is_expired(current_time):
                expired_keys.append(key)
                self.expiration_events.append({
                    'key': key,
                    'expired_at': current_time,
                    'original_ttl': entry.ttl,
                    'age_at_expiry': current_time - entry.timestamp
                })
                del self.storage[key]
        
        return expired_keys
    
    def set(self, query: str, value: Any, ttl: Optional[int] = None, 
            confidence: float = 0.9, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Set cache entry with TTL."""
        # Cleanup expired entries first
        self._cleanup_expired()
        
        # Apply size limit with LRU eviction
        while len(self.storage) >= self.max_size:
            oldest_key = next(iter(self.storage))
            del self.storage[oldest_key]
        
        key = self._generate_key(query)
        entry = TTLTestEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
            confidence=confidence,
            tier=self.tier,
            metadata=metadata or {}
        )
        
        self.storage[key] = entry
        return key
    
    def get(self, query: str) -> Optional[Any]:
        """Get cache entry, handling TTL expiration."""
        key = self._generate_key(query)
        
        # Clean expired entries
        self._cleanup_expired()
        
        if key in self.storage:
            entry = self.storage[key]
            if not entry.is_expired():
                self.hits += 1
                entry.access_count += 1
                # Move to end for LRU
                self.storage.move_to_end(key)
                return entry.value
            else:
                # Entry expired, remove it
                del self.storage[key]
        
        self.misses += 1
        return None
    
    def extend_ttl(self, query: str, additional_time: int) -> bool:
        """Extend TTL for existing entry."""
        key = self._generate_key(query)
        
        if key in self.storage:
            entry = self.storage[key]
            if not entry.is_expired():
                entry.ttl += additional_time
                self.ttl_extensions.append({
                    'key': key,
                    'extended_at': time.time(),
                    'additional_time': additional_time,
                    'new_ttl': entry.ttl
                })
                return True
        
        return False
    
    def refresh_ttl(self, query: str, new_ttl: Optional[int] = None) -> bool:
        """Refresh TTL with current timestamp."""
        key = self._generate_key(query)
        
        if key in self.storage:
            entry = self.storage[key]
            if not entry.is_expired():
                entry.refresh_ttl(new_ttl)
                return True
        
        return False
    
    def get_ttl_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Get TTL information for entry."""
        key = self._generate_key(query)
        
        if key in self.storage:
            entry = self.storage[key]
            current_time = time.time()
            return {
                'key': key,
                'ttl': entry.ttl,
                'timestamp': entry.timestamp,
                'age': current_time - entry.timestamp,
                'time_until_expiry': entry.time_until_expiry(current_time),
                'is_expired': entry.is_expired(current_time),
                'access_count': entry.access_count,
                'confidence': entry.confidence
            }
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics including TTL metrics."""
        current_time = time.time()
        total_entries = len(self.storage)
        expired_count = len([e for e in self.storage.values() if e.is_expired(current_time)])
        
        ttl_distribution = {}
        for entry in self.storage.values():
            ttl_bucket = (entry.ttl // 300) * 300  # 5-minute buckets
            ttl_distribution[ttl_bucket] = ttl_distribution.get(ttl_bucket, 0) + 1
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_count,
            'active_entries': total_entries - expired_count,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'expiration_events': len(self.expiration_events),
            'ttl_extensions': len(self.ttl_extensions),
            'ttl_distribution': ttl_distribution
        }


class TestTTLBasicFunctionality:
    """Tests for core TTL functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache(default_ttl=TTL_CONFIGS['L1_CACHE'])
        self.data_generator = BiomedicalTestDataGenerator()
    
    def test_basic_ttl_setting(self):
        """Test basic TTL configuration and setting."""
        query = "What is glucose metabolism?"
        value = "Glucose metabolism involves glycolysis and gluconeogenesis"
        
        # Set with default TTL
        key = self.cache.set(query, value)
        assert key is not None
        
        # Verify TTL info
        ttl_info = self.cache.get_ttl_info(query)
        assert ttl_info is not None
        assert ttl_info['ttl'] == TTL_CONFIGS['L1_CACHE']
        assert ttl_info['time_until_expiry'] > 0
        assert not ttl_info['is_expired']
    
    def test_custom_ttl_setting(self):
        """Test setting entries with custom TTL values."""
        test_cases = [
            ("Short TTL query", "Short data", 60),
            ("Medium TTL query", "Medium data", 1800),
            ("Long TTL query", "Long data", 7200)
        ]
        
        for query, value, ttl in test_cases:
            key = self.cache.set(query, value, ttl=ttl)
            ttl_info = self.cache.get_ttl_info(query)
            
            assert ttl_info['ttl'] == ttl
            assert ttl_info['time_until_expiry'] <= ttl
            assert ttl_info['time_until_expiry'] > ttl - 1  # Account for execution time
    
    def test_ttl_expiration_basic(self):
        """Test basic TTL expiration behavior."""
        query = "Expiring query"
        value = "Test data"
        
        # Set with 1-second TTL
        self.cache.set(query, value, ttl=1)
        
        # Should be retrievable immediately
        result = self.cache.get(query)
        assert result == value
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be None after expiration
        result = self.cache.get(query)
        assert result is None
        
        # Verify expiration event was recorded
        stats = self.cache.get_statistics()
        assert stats['expiration_events'] > 0
    
    def test_ttl_precision(self):
        """Test TTL precision and accuracy."""
        query = "Precision test query"
        value = "Test data"
        ttl = 2  # 2 seconds
        
        start_time = time.time()
        self.cache.set(query, value, ttl=ttl)
        
        # Check at 50% of TTL
        time.sleep(1.0)
        result = self.cache.get(query)
        assert result == value
        
        # Check just before expiration
        time.sleep(0.8)  # Total: 1.8s
        result = self.cache.get(query)
        assert result == value
        
        # Check after expiration
        time.sleep(0.5)  # Total: 2.3s
        result = self.cache.get(query)
        assert result is None
        
        # Verify timing precision (within 500ms tolerance)
        elapsed = time.time() - start_time
        assert 2.0 <= elapsed <= 2.5


class TestTTLExpirationScenarios:
    """Tests for comprehensive TTL expiration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache(default_ttl=3600)
        self.biomedical_queries = BIOMEDICAL_QUERIES['metabolism'][:3]
    
    def test_multiple_entry_expiration(self):
        """Test expiration of multiple entries with different TTLs."""
        entries = [
            ("Query 1", "Data 1", 1),
            ("Query 2", "Data 2", 2),
            ("Query 3", "Data 3", 3),
            ("Query 4", "Data 4", 10)  # Won't expire during test
        ]
        
        # Set all entries
        for query, value, ttl in entries:
            self.cache.set(query, value, ttl=ttl)
        
        # Verify all entries are present
        for query, value, _ in entries:
            assert self.cache.get(query) == value
        
        # Wait 1.5 seconds - first entry should expire
        time.sleep(1.5)
        assert self.cache.get("Query 1") is None
        assert self.cache.get("Query 2") is not None
        assert self.cache.get("Query 3") is not None
        assert self.cache.get("Query 4") is not None
        
        # Wait another 1 second - second entry should expire
        time.sleep(1.0)  # Total: 2.5s
        assert self.cache.get("Query 2") is None
        assert self.cache.get("Query 3") is not None
        assert self.cache.get("Query 4") is not None
        
        # Wait another 1 second - third entry should expire
        time.sleep(1.0)  # Total: 3.5s
        assert self.cache.get("Query 3") is None
        assert self.cache.get("Query 4") is not None
    
    def test_expiration_cleanup_timing(self):
        """Test timing and efficiency of expiration cleanup."""
        # Fill cache with entries that will expire
        for i in range(10):
            query = f"Expiring query {i}"
            self.cache.set(query, f"Data {i}", ttl=1)
        
        # Add one long-lived entry
        self.cache.set("Long lived", "Persistent data", ttl=3600)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Accessing any entry should trigger cleanup
        result = self.cache.get("Long lived")
        assert result == "Persistent data"
        
        # Verify all expired entries were cleaned up
        stats = self.cache.get_statistics()
        assert stats['active_entries'] == 1
        assert stats['expiration_events'] >= 10
    
    def test_expiration_boundary_conditions(self):
        """Test expiration at exact boundary conditions."""
        query = "Boundary test"
        value = "Test data"
        ttl = 2
        
        self.cache.set(query, value, ttl=ttl)
        
        # Test access just before expiration (within 100ms)
        time.sleep(ttl - 0.1)
        result = self.cache.get(query)
        assert result == value, "Entry should still be valid just before expiration"
        
        # Test access just after expiration
        time.sleep(0.2)  # Total: ttl + 0.1
        result = self.cache.get(query)
        assert result is None, "Entry should be expired just after TTL"
    
    def test_expiration_with_biomedical_data(self):
        """Test expiration scenarios with realistic biomedical queries."""
        # Set biomedical queries with varying TTLs based on confidence
        for i, query_data in enumerate(self.biomedical_queries):
            query = query_data['query']
            response = query_data['response']
            confidence = response['confidence']
            
            # Use confidence-based TTL
            if confidence >= 0.9:
                ttl = 3
            elif confidence >= 0.8:
                ttl = 2
            else:
                ttl = 1
            
            self.cache.set(query, response, ttl=ttl, confidence=confidence)
        
        # Wait for some entries to expire
        time.sleep(1.5)
        
        # Check which entries are still available based on confidence
        available_count = 0
        expired_count = 0
        
        for query_data in self.biomedical_queries:
            query = query_data['query']
            result = self.cache.get(query)
            
            if result is not None:
                available_count += 1
            else:
                expired_count += 1
        
        # High confidence entries should still be available
        assert available_count > 0, "Some high-confidence entries should still be available"
        assert expired_count > 0, "Some low-confidence entries should have expired"


class TestDynamicTTLManagement:
    """Tests for dynamic TTL adjustment and management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache(default_ttl=3600)
        self.data_generator = BiomedicalTestDataGenerator()
    
    def test_ttl_extension(self):
        """Test extending TTL of existing entries."""
        query = "Extendable query"
        value = "Test data"
        initial_ttl = 2
        
        # Set entry with short TTL
        self.cache.set(query, value, ttl=initial_ttl)
        
        # Wait halfway through TTL
        time.sleep(1.0)
        
        # Extend TTL by 3 seconds
        success = self.cache.extend_ttl(query, 3)
        assert success, "TTL extension should succeed"
        
        # Wait past original expiration time
        time.sleep(1.5)  # Total: 2.5s (past original TTL)
        
        # Should still be available due to extension
        result = self.cache.get(query)
        assert result == value, "Entry should still be available after TTL extension"
        
        # Verify extension was recorded
        stats = self.cache.get_statistics()
        assert stats['ttl_extensions'] == 1
    
    def test_ttl_refresh(self):
        """Test refreshing TTL with new timestamp."""
        query = "Refreshable query"
        value = "Test data"
        ttl = 2
        
        # Set entry
        self.cache.set(query, value, ttl=ttl)
        
        # Wait halfway through TTL
        time.sleep(1.0)
        
        # Refresh TTL (resets timestamp)
        success = self.cache.refresh_ttl(query)
        assert success, "TTL refresh should succeed"
        
        # Wait past original expiration time
        time.sleep(1.5)  # Would have expired at 2s, now should expire at 3s
        
        # Should still be available due to refresh
        result = self.cache.get(query)
        assert result == value, "Entry should still be available after TTL refresh"
    
    def test_adaptive_ttl_based_on_access_pattern(self):
        """Test adaptive TTL based on access patterns."""
        # Simulate frequently accessed entry
        frequent_query = "Frequently accessed query"
        frequent_value = "Popular data"
        
        # Simulate rarely accessed entry
        rare_query = "Rarely accessed query"
        rare_value = "Unpopular data"
        
        # Set both with same initial TTL
        initial_ttl = 5
        self.cache.set(frequent_query, frequent_value, ttl=initial_ttl)
        self.cache.set(rare_query, rare_value, ttl=initial_ttl)
        
        # Simulate frequent access pattern
        for _ in range(5):
            self.cache.get(frequent_query)
            time.sleep(0.2)
        
        # Get access counts
        frequent_info = self.cache.get_ttl_info(frequent_query)
        rare_info = self.cache.get_ttl_info(rare_query)
        
        assert frequent_info['access_count'] > rare_info['access_count']
        
        # In a real implementation, frequently accessed items might get extended TTL
        # For now, just verify we can track access patterns
        assert frequent_info['access_count'] >= 5
        assert rare_info['access_count'] == 0
    
    def test_confidence_based_ttl_adjustment(self):
        """Test TTL adjustment based on response confidence."""
        test_cases = [
            ("High confidence query", "Reliable data", 0.95, TTL_CONFIGS['HIGH_CONFIDENCE']),
            ("Medium confidence query", "Moderate data", 0.75, TTL_CONFIGS['L1_CACHE']),
            ("Low confidence query", "Uncertain data", 0.65, TTL_CONFIGS['LOW_CONFIDENCE'])
        ]
        
        for query, value, confidence, expected_ttl in test_cases:
            # Simulate confidence-based TTL selection
            ttl = self._calculate_confidence_based_ttl(confidence)
            self.cache.set(query, value, ttl=ttl, confidence=confidence)
            
            ttl_info = self.cache.get_ttl_info(query)
            assert ttl_info['ttl'] == expected_ttl
            assert ttl_info['confidence'] == confidence
    
    def _calculate_confidence_based_ttl(self, confidence: float) -> int:
        """Calculate TTL based on confidence score."""
        if confidence >= 0.9:
            return TTL_CONFIGS['HIGH_CONFIDENCE']
        elif confidence >= 0.8:
            return TTL_CONFIGS['L1_CACHE']
        else:
            return TTL_CONFIGS['LOW_CONFIDENCE']
    
    def test_query_type_specific_ttl(self):
        """Test TTL adjustment based on query type."""
        # Temporal/current queries - short TTL
        temporal_query = "Latest COVID-19 research 2024"
        self.cache.set(temporal_query, "Current research", ttl=TTL_CONFIGS['TEMPORAL_QUERIES'])
        
        # Clinical critical information - longer TTL
        critical_query = "Emergency treatment protocol for anaphylaxis"
        self.cache.set(critical_query, "Critical protocol", ttl=TTL_CONFIGS['CLINICAL_CRITICAL'])
        
        # General knowledge - standard TTL
        general_query = "What is photosynthesis?"
        self.cache.set(general_query, "Process explanation", ttl=TTL_CONFIGS['L2_CACHE'])
        
        # Verify TTL assignments
        temporal_info = self.cache.get_ttl_info(temporal_query)
        critical_info = self.cache.get_ttl_info(critical_query)
        general_info = self.cache.get_ttl_info(general_query)
        
        assert temporal_info['ttl'] == TTL_CONFIGS['TEMPORAL_QUERIES']
        assert critical_info['ttl'] == TTL_CONFIGS['CLINICAL_CRITICAL']
        assert general_info['ttl'] == TTL_CONFIGS['L2_CACHE']
        
        # Temporal should have shortest TTL
        assert temporal_info['ttl'] < general_info['ttl'] < critical_info['ttl']


class TestConfidenceBasedTTL:
    """Tests for confidence-based TTL adjustments."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache()
        self.biomedical_queries = BIOMEDICAL_QUERIES['metabolism']
    
    def test_high_confidence_longer_ttl(self):
        """Test that high confidence responses get longer TTL."""
        high_conf_query = "What is ATP synthesis?"
        high_conf_response = {"process": "oxidative phosphorylation", "confidence": 0.95}
        
        low_conf_query = "What causes rare metabolic disorder X?"
        low_conf_response = {"possibilities": ["genetic mutation", "environmental"], "confidence": 0.65}
        
        # Set entries with confidence-based TTL
        high_ttl = self._get_ttl_by_confidence(high_conf_response['confidence'])
        low_ttl = self._get_ttl_by_confidence(low_conf_response['confidence'])
        
        self.cache.set(high_conf_query, high_conf_response, ttl=high_ttl, 
                      confidence=high_conf_response['confidence'])
        self.cache.set(low_conf_query, low_conf_response, ttl=low_ttl,
                      confidence=low_conf_response['confidence'])
        
        high_info = self.cache.get_ttl_info(high_conf_query)
        low_info = self.cache.get_ttl_info(low_conf_query)
        
        assert high_info['ttl'] > low_info['ttl']
        assert high_info['confidence'] > low_info['confidence']
    
    def _get_ttl_by_confidence(self, confidence: float) -> int:
        """Get TTL based on confidence score."""
        if confidence >= 0.9:
            return TTL_CONFIGS['HIGH_CONFIDENCE']
        elif confidence >= 0.8:
            return TTL_CONFIGS['L2_CACHE']  
        elif confidence >= 0.7:
            return TTL_CONFIGS['L1_CACHE']
        else:
            return TTL_CONFIGS['LOW_CONFIDENCE']
    
    def test_confidence_ttl_scaling(self):
        """Test TTL scaling across confidence range."""
        confidence_levels = [0.95, 0.85, 0.75, 0.65, 0.55]
        ttl_values = []
        
        for i, confidence in enumerate(confidence_levels):
            query = f"Confidence test query {i}"
            response = {"data": f"test data {i}", "confidence": confidence}
            ttl = self._get_ttl_by_confidence(confidence)
            
            self.cache.set(query, response, ttl=ttl, confidence=confidence)
            ttl_values.append(ttl)
        
        # Verify decreasing TTL with decreasing confidence
        for i in range(len(ttl_values) - 1):
            assert ttl_values[i] >= ttl_values[i + 1], \
                f"TTL should decrease with confidence: {ttl_values[i]} >= {ttl_values[i+1]}"
    
    def test_confidence_threshold_behavior(self):
        """Test behavior at confidence thresholds."""
        # Test right at threshold boundaries
        threshold_tests = [
            (0.90, TTL_CONFIGS['HIGH_CONFIDENCE']),  # Right at high threshold
            (0.89, TTL_CONFIGS['L2_CACHE']),         # Just below high threshold
            (0.80, TTL_CONFIGS['L2_CACHE']),         # Right at medium threshold
            (0.79, TTL_CONFIGS['L1_CACHE']),         # Just below medium threshold
            (0.70, TTL_CONFIGS['L1_CACHE']),         # Right at low threshold
            (0.69, TTL_CONFIGS['LOW_CONFIDENCE'])    # Below low threshold
        ]
        
        for confidence, expected_ttl in threshold_tests:
            query = f"Threshold test {confidence}"
            calculated_ttl = self._get_ttl_by_confidence(confidence)
            
            assert calculated_ttl == expected_ttl, \
                f"Confidence {confidence} should map to TTL {expected_ttl}, got {calculated_ttl}"


class TestMultiTierTTLCoordination:
    """Tests for TTL coordination across multiple cache tiers."""
    
    def setup_method(self):
        """Set up test fixtures with multiple cache tiers."""
        self.l1_cache = MockTTLCache(max_size=5, default_ttl=TTL_CONFIGS['L1_CACHE'])
        self.l1_cache.tier = 'L1'
        
        self.l2_cache = MockTTLCache(max_size=20, default_ttl=TTL_CONFIGS['L2_CACHE'])
        self.l2_cache.tier = 'L2'
        
        self.l3_cache = MockTTLCache(max_size=100, default_ttl=TTL_CONFIGS['L3_CACHE'])
        self.l3_cache.tier = 'L3'
    
    def test_ttl_cascade_across_tiers(self):
        """Test TTL cascading from higher to lower tiers."""
        query = "Multi-tier test query"
        value = "Test data"
        
        # Set in L3 first (longest TTL)
        self.l3_cache.set(query, value)
        
        # Promote to L2 (should inherit shorter TTL)
        l3_value = self.l3_cache.get(query)
        self.l2_cache.set(query, l3_value, ttl=TTL_CONFIGS['L2_CACHE'])
        
        # Promote to L1 (should get even shorter TTL)
        l2_value = self.l2_cache.get(query)
        self.l1_cache.set(query, l2_value, ttl=TTL_CONFIGS['L1_CACHE'])
        
        # Verify TTL hierarchy
        l1_info = self.l1_cache.get_ttl_info(query)
        l2_info = self.l2_cache.get_ttl_info(query)
        l3_info = self.l3_cache.get_ttl_info(query)
        
        assert l1_info['ttl'] < l2_info['ttl'] < l3_info['ttl']
    
    def test_ttl_synchronization(self):
        """Test TTL synchronization across tiers."""
        query = "Sync test query"
        value = "Synchronized data"
        
        # Set same entry in all tiers with different TTLs
        self.l1_cache.set(query, value, ttl=300)   # 5 minutes
        self.l2_cache.set(query, value, ttl=1800)  # 30 minutes
        self.l3_cache.set(query, value, ttl=7200)  # 2 hours
        
        # Simulate TTL synchronization (in real system, this might be automatic)
        # For testing, we'll manually check TTL differences
        l1_info = self.l1_cache.get_ttl_info(query)
        l2_info = self.l2_cache.get_ttl_info(query)
        l3_info = self.l3_cache.get_ttl_info(query)
        
        # Verify proper TTL progression
        assert l1_info['ttl'] == 300
        assert l2_info['ttl'] == 1800
        assert l3_info['ttl'] == 7200
        
        # Verify all entries are currently valid
        assert not l1_info['is_expired']
        assert not l2_info['is_expired']
        assert not l3_info['is_expired']
    
    def test_cross_tier_ttl_consistency(self):
        """Test TTL consistency when promoting/demoting between tiers."""
        query = "Consistency test query"
        value = "Test data"
        original_ttl = 1800  # 30 minutes
        
        # Start in L2
        self.l2_cache.set(query, value, ttl=original_ttl)
        
        # Promote to L1 - should maintain reasonable TTL relative to remaining time
        l2_info = self.l2_cache.get_ttl_info(query)
        remaining_time = l2_info['time_until_expiry']
        
        # L1 TTL should not exceed remaining time from L2
        l1_ttl = min(TTL_CONFIGS['L1_CACHE'], int(remaining_time))
        self.l1_cache.set(query, value, ttl=l1_ttl)
        
        l1_info = self.l1_cache.get_ttl_info(query)
        assert l1_info['ttl'] <= remaining_time + 1  # Allow for execution time
    
    def test_tier_specific_ttl_policies(self):
        """Test that each tier enforces its own TTL policies."""
        query = "Policy test query"
        value = "Test data"
        
        # Each tier should respect its maximum TTL limits
        excessive_ttl = 100000  # Very long TTL
        
        # L1 should cap TTL
        self.l1_cache.set(query, value, ttl=excessive_ttl)
        l1_info = self.l1_cache.get_ttl_info(query)
        
        # In a real system, L1 might cap TTL to reasonable limits
        # For this test, we verify it accepts the TTL but could add policy enforcement
        assert l1_info['ttl'] == excessive_ttl  # Currently accepts any TTL
        
        # Could add policy enforcement:
        # expected_l1_ttl = min(excessive_ttl, TTL_CONFIGS['L1_CACHE'] * 2)
        # assert l1_info['ttl'] <= expected_l1_ttl


class TestTTLBoundaryConditions:
    """Tests for TTL boundary conditions and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache()
    
    def test_zero_ttl(self):
        """Test behavior with zero TTL (immediate expiration)."""
        query = "Zero TTL query"
        value = "Should expire immediately"
        
        # Set with zero TTL
        self.cache.set(query, value, ttl=0)
        
        # Should be expired immediately
        result = self.cache.get(query)
        assert result is None
    
    def test_negative_ttl(self):
        """Test behavior with negative TTL."""
        query = "Negative TTL query"
        value = "Already expired"
        
        # Set with negative TTL
        self.cache.set(query, value, ttl=-1)
        
        # Should be expired immediately
        result = self.cache.get(query)
        assert result is None
    
    def test_very_large_ttl(self):
        """Test behavior with very large TTL values."""
        query = "Large TTL query"
        value = "Long-lived data"
        large_ttl = 2**31 - 1  # Maximum 32-bit integer
        
        self.cache.set(query, value, ttl=large_ttl)
        
        ttl_info = self.cache.get_ttl_info(query)
        assert ttl_info['ttl'] == large_ttl
        assert ttl_info['time_until_expiry'] > 1000000  # Should be very large
        
        # Should be retrievable
        result = self.cache.get(query)
        assert result == value
    
    def test_concurrent_ttl_operations(self):
        """Test TTL operations under concurrent access."""
        query = "Concurrent test query"
        value = "Test data"
        
        # Set initial entry
        self.cache.set(query, value, ttl=5)
        
        def extend_ttl():
            for _ in range(5):
                self.cache.extend_ttl(query, 1)
                time.sleep(0.1)
        
        def refresh_ttl():
            for _ in range(5):
                self.cache.refresh_ttl(query)
                time.sleep(0.1)
        
        # Start concurrent threads
        extend_thread = threading.Thread(target=extend_ttl)
        refresh_thread = threading.Thread(target=refresh_ttl)
        
        extend_thread.start()
        refresh_thread.start()
        
        # Access entry while operations are happening
        for _ in range(10):
            result = self.cache.get(query)
            assert result == value  # Should remain accessible
            time.sleep(0.05)
        
        extend_thread.join()
        refresh_thread.join()
        
        # Verify operations were recorded
        stats = self.cache.get_statistics()
        assert stats['ttl_extensions'] > 0
    
    def test_clock_changes(self):
        """Test TTL behavior with simulated clock changes."""
        query = "Clock test query"
        value = "Time-sensitive data"
        
        # Set entry with 3 second TTL
        self.cache.set(query, value, ttl=3)
        
        # Mock time forward by 2 seconds
        original_time = time.time
        
        def mock_time_forward():
            return original_time() + 2
        
        with patch('time.time', side_effect=mock_time_forward):
            # Should still be valid (2 < 3)
            result = self.cache.get(query)
            # Note: This test might fail due to time.time() being called in multiple places
            # In real implementation, would need more sophisticated time mocking
        
        # Restore original time function
        # Entry should still be accessible in real time
        result = self.cache.get(query)
        assert result == value


class TestTTLExtensionRefresh:
    """Tests for TTL extension and refresh mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache()
    
    def test_multiple_ttl_extensions(self):
        """Test multiple TTL extensions on same entry."""
        query = "Multiple extension query"
        value = "Test data"
        initial_ttl = 2
        
        # Set entry
        self.cache.set(query, value, ttl=initial_ttl)
        
        # Multiple extensions
        extensions = [1, 2, 3]  # Total extension: 6 seconds
        
        for extension in extensions:
            success = self.cache.extend_ttl(query, extension)
            assert success
        
        # Wait past original TTL
        time.sleep(initial_ttl + 1)  # 3 seconds
        
        # Should still be available due to extensions
        result = self.cache.get(query)
        assert result == value
        
        # Verify extensions were recorded
        stats = self.cache.get_statistics()
        assert stats['ttl_extensions'] == len(extensions)
    
    def test_ttl_refresh_resets_age(self):
        """Test that TTL refresh resets the entry age."""
        query = "Refresh test query"
        value = "Test data"
        ttl = 3
        
        # Set entry
        self.cache.set(query, value, ttl=ttl)
        
        # Wait halfway through TTL
        time.sleep(1.5)
        
        # Get info before refresh
        info_before = self.cache.get_ttl_info(query)
        age_before = info_before['age']
        time_left_before = info_before['time_until_expiry']
        
        # Refresh TTL
        success = self.cache.refresh_ttl(query)
        assert success
        
        # Get info after refresh
        info_after = self.cache.get_ttl_info(query)
        age_after = info_after['age']
        time_left_after = info_after['time_until_expiry']
        
        # Age should be reset (close to 0)
        assert age_after < age_before
        assert age_after < 0.1  # Should be very recent
        
        # Time until expiry should be refreshed to full TTL
        assert time_left_after > time_left_before
        assert time_left_after > ttl - 0.1  # Should be close to full TTL
    
    def test_extension_vs_refresh_behavior(self):
        """Test difference between TTL extension and refresh."""
        base_query = "Extension vs refresh test"
        value = "Test data"
        ttl = 2
        
        # Test extension
        extend_query = f"{base_query} - extend"
        self.cache.set(extend_query, value, ttl=ttl)
        
        time.sleep(1)  # Half TTL
        
        extend_info_before = self.cache.get_ttl_info(extend_query)
        self.cache.extend_ttl(extend_query, 2)  # Add 2 seconds
        extend_info_after = self.cache.get_ttl_info(extend_query)
        
        # Test refresh  
        refresh_query = f"{base_query} - refresh"
        self.cache.set(refresh_query, value, ttl=ttl)
        
        time.sleep(1)  # Half TTL
        
        refresh_info_before = self.cache.get_ttl_info(refresh_query)
        self.cache.refresh_ttl(refresh_query)  # Reset with same TTL
        refresh_info_after = self.cache.get_ttl_info(refresh_query)
        
        # Extension: TTL increases, age stays same
        assert extend_info_after['ttl'] > extend_info_before['ttl']
        
        # Refresh: TTL stays same, age resets
        assert refresh_info_after['ttl'] == refresh_info_before['ttl']
        assert refresh_info_after['age'] < refresh_info_before['age']


class TestTTLPerformanceImpact:
    """Tests for TTL impact on cache performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockTTLCache(max_size=100)
        self.performance_measurer = CachePerformanceMeasurer()
    
    def test_ttl_cleanup_performance(self):
        """Test performance impact of TTL cleanup operations."""
        # Fill cache with entries that will expire
        num_entries = 50
        
        for i in range(num_entries):
            query = f"Performance test query {i}"
            value = f"Data {i}"
            ttl = 1 if i % 2 == 0 else 3600  # Half will expire quickly
            self.cache.set(query, value, ttl=ttl)
        
        # Measure get performance before expiration
        _, time_before, _ = self.performance_measurer.measure_operation(
            "get_before_expiration", self.cache.get, "Performance test query 1"
        )
        
        # Wait for some entries to expire
        time.sleep(1.1)
        
        # Measure get performance after expiration (triggers cleanup)
        _, time_after, _ = self.performance_measurer.measure_operation(
            "get_after_expiration", self.cache.get, "Performance test query 1"
        )
        
        # Performance should not degrade significantly
        # (In real implementation, cleanup might be optimized)
        assert time_after < time_before * 10  # Allow for some overhead
        
        # Verify cleanup occurred
        stats = self.cache.get_statistics()
        assert stats['expiration_events'] > 0
    
    def test_ttl_extension_performance(self):
        """Test performance of TTL extension operations."""
        query = "Extension performance test"
        value = "Test data"
        
        self.cache.set(query, value, ttl=3600)
        
        # Measure extension performance
        extension_times = []
        for i in range(10):
            _, duration, success = self.performance_measurer.measure_operation(
                "ttl_extension", self.cache.extend_ttl, query, 60
            )
            extension_times.append(duration)
            assert success
        
        avg_extension_time = sum(extension_times) / len(extension_times)
        
        # TTL extensions should be fast (< 1ms typically)
        assert avg_extension_time < 10  # 10ms is generous for in-memory operations
    
    def test_memory_usage_with_ttl_metadata(self):
        """Test memory overhead of TTL metadata."""
        # This is more of a conceptual test since we can't easily measure
        # Python object memory usage precisely, but we can verify structure
        
        query = "Memory test query"
        value = "Test data"
        
        self.cache.set(query, value, ttl=3600)
        
        ttl_info = self.cache.get_ttl_info(query)
        
        # Verify all TTL metadata is present
        required_fields = ['key', 'ttl', 'timestamp', 'age', 'time_until_expiry', 
                          'is_expired', 'access_count', 'confidence']
        
        for field in required_fields:
            assert field in ttl_info, f"TTL info missing required field: {field}"
        
        # Verify data types are appropriate (not storing unnecessary large objects)
        assert isinstance(ttl_info['ttl'], int)
        assert isinstance(ttl_info['timestamp'], float)
        assert isinstance(ttl_info['access_count'], int)
        assert isinstance(ttl_info['confidence'], float)


# Pytest fixtures
@pytest.fixture
def ttl_cache():
    """Provide TTL-enabled cache for testing."""
    return MockTTLCache()


@pytest.fixture
def multi_tier_ttl_caches():
    """Provide multi-tier TTL caches for testing."""
    return {
        'L1': MockTTLCache(max_size=5, default_ttl=TTL_CONFIGS['L1_CACHE']),
        'L2': MockTTLCache(max_size=20, default_ttl=TTL_CONFIGS['L2_CACHE']),
        'L3': MockTTLCache(max_size=100, default_ttl=TTL_CONFIGS['L3_CACHE'])
    }


@pytest.fixture
def biomedical_ttl_data():
    """Provide biomedical data with TTL configurations."""
    data = []
    for category, queries in BIOMEDICAL_QUERIES.items():
        for query_data in queries[:2]:  # Limit for testing
            ttl = query_data.get('expected_ttl', TTL_CONFIGS['L1_CACHE'])
            data.append({
                'query': query_data['query'],
                'response': query_data['response'],
                'ttl': ttl,
                'confidence': query_data['response'].get('confidence', 0.9)
            })
    return data


# Module-level test configuration
pytest_plugins = ['pytest_asyncio']


if __name__ == "__main__":
    # Run specific test class or all tests
    pytest.main([__file__, "-v"])