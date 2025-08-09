"""
Comprehensive cache invalidation strategy tests for the Clinical Metabolomics Oracle system.

This module provides extensive testing of cache invalidation mechanisms across different
triggers, patterns, and strategies. It covers time-based expiration, size-limit based
LRU eviction, manual invalidation operations, pattern-based invalidation, and system
state change invalidation scenarios.

Test Coverage:
- Invalidation trigger testing (time, size, manual, pattern, system state)
- Manual invalidation operations and bulk operations
- Pattern-based invalidation (query type changes, metadata updates)
- Access-based invalidation strategies and confidence-score based invalidation
- Cache hit ratio optimization through strategic invalidation
- Resource utilization-based invalidation policies
- Immediate vs deferred invalidation strategies
- Conditional invalidation based on metadata and system state
- Background cleanup and garbage collection processes
- Edge cases and boundary conditions for invalidation scenarios

Classes:
    TestInvalidationTriggers: Core invalidation trigger mechanism testing
    TestManualInvalidation: Manual and programmatic invalidation operations
    TestPatternBasedInvalidation: Pattern and metadata-based invalidation
    TestAccessBasedInvalidation: Usage pattern and confidence-based invalidation
    TestInvalidationStrategies: Different invalidation strategy implementations
    TestBulkInvalidation: Bulk and batch invalidation operations
    TestConditionalInvalidation: Conditional and rule-based invalidation
    TestInvalidationPerformance: Performance impact of invalidation operations
    TestInvalidationEdgeCases: Edge cases and boundary condition testing

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
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import re

# Import test fixtures
from .cache_test_fixtures import (
    BIOMEDICAL_QUERIES,
    BiomedicalTestDataGenerator,
    CachePerformanceMetrics,
    CachePerformanceMeasurer,
    EMERGENCY_RESPONSE_PATTERNS
)


# Invalidation strategy constants
INVALIDATION_STRATEGIES = {
    'IMMEDIATE': 'immediate',
    'DEFERRED': 'deferred',
    'BATCH': 'batch',
    'BACKGROUND': 'background'
}

INVALIDATION_TRIGGERS = {
    'TIME_BASED': 'time_expiration',
    'SIZE_BASED': 'size_limit',
    'MANUAL': 'manual_request',
    'PATTERN_BASED': 'pattern_match',
    'SYSTEM_STATE': 'system_change',
    'ACCESS_BASED': 'access_pattern',
    'CONFIDENCE_BASED': 'confidence_threshold',
    'RESOURCE_BASED': 'resource_utilization'
}

INVALIDATION_POLICIES = {
    'LRU': 'least_recently_used',
    'LFU': 'least_frequently_used',
    'FIFO': 'first_in_first_out',
    'CONFIDENCE_WEIGHTED': 'confidence_weighted',
    'ACCESS_COUNT_WEIGHTED': 'access_count_weighted',
    'SIZE_BASED': 'size_based',
    'TTL_BASED': 'ttl_based'
}


@dataclass
class InvalidationEvent:
    """Track invalidation events for analysis."""
    event_id: str
    trigger: str
    strategy: str
    timestamp: float
    affected_keys: List[str]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Optional[float] = None
    success: bool = True
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class InvalidationRule:
    """Define invalidation rules and conditions."""
    rule_id: str
    trigger: str
    condition: str
    action: str
    priority: int = 100
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockInvalidatingCache:
    """
    Mock cache implementation with comprehensive invalidation strategies.
    
    This mock extends the basic TTL cache with advanced invalidation
    mechanisms for testing various invalidation scenarios.
    """
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600,
                 invalidation_strategy: str = INVALIDATION_STRATEGIES['IMMEDIATE']):
        # Storage and basic configuration
        self.storage: Dict[str, Any] = OrderedDict()
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.invalidation_strategy = invalidation_strategy
        
        # Statistics tracking
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.evictions = 0
        
        # Invalidation tracking
        self.invalidation_events: List[InvalidationEvent] = []
        self.invalidation_rules: List[InvalidationRule] = []
        self.pending_invalidations: Set[str] = set()
        self.invalidation_queue: List[Tuple[str, str, Dict[str, Any]]] = []
        
        # Access pattern tracking
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        self.confidence_scores: Dict[str, float] = {}
        
        # Background processing
        self._background_cleanup_enabled = True
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()
        
        # Pattern matching
        self.pattern_rules: Dict[str, List[str]] = defaultdict(list)
        self.tag_to_keys: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.performance_measurer = CachePerformanceMeasurer()
    
    def _generate_key(self, query: str, prefix: str = "") -> str:
        """Generate cache key from query."""
        return f"{prefix}:{hash(query)}" if prefix else str(hash(query))
    
    def _record_invalidation(self, event: InvalidationEvent):
        """Record invalidation event for analysis."""
        self.invalidation_events.append(event)
        self.invalidations += 1
    
    def _should_invalidate_by_rule(self, key: str, metadata: Dict[str, Any]) -> List[InvalidationRule]:
        """Check if entry should be invalidated by rules."""
        applicable_rules = []
        
        for rule in self.invalidation_rules:
            if not rule.enabled:
                continue
                
            if self._evaluate_rule_condition(rule, key, metadata):
                applicable_rules.append(rule)
        
        return sorted(applicable_rules, key=lambda r: r.priority)
    
    def _evaluate_rule_condition(self, rule: InvalidationRule, key: str, metadata: Dict[str, Any]) -> bool:
        """Evaluate if rule condition is met."""
        try:
            # Simple condition evaluation (in production, this would be more sophisticated)
            condition = rule.condition
            
            # Time-based conditions
            if 'age >' in condition:
                age_limit = float(condition.split('age >')[1].strip())
                entry_age = metadata.get('age', 0)
                return entry_age > age_limit
            
            # Access count conditions
            if 'access_count <' in condition:
                access_limit = int(condition.split('access_count <')[1].strip())
                access_count = metadata.get('access_count', 0)
                return access_count < access_limit
            
            # Confidence conditions
            if 'confidence <' in condition:
                conf_limit = float(condition.split('confidence <')[1].strip())
                confidence = metadata.get('confidence', 1.0)
                return confidence < conf_limit
            
            # Pattern matching
            if 'pattern:' in condition:
                pattern = condition.split('pattern:')[1].strip()
                query = metadata.get('original_query', '')
                return bool(re.search(pattern, query, re.IGNORECASE))
            
            # Tag-based conditions
            if 'tag:' in condition:
                tag = condition.split('tag:')[1].strip()
                tags = metadata.get('tags', [])
                return tag in tags
            
            return False
        except Exception:
            return False
    
    def set(self, query: str, value: Any, ttl: Optional[int] = None,
            confidence: float = 0.9, tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Set cache entry with invalidation metadata."""
        key = self._generate_key(query)
        
        # Enforce size limits with invalidation
        if len(self.storage) >= self.max_size:
            self._invalidate_by_policy(INVALIDATION_POLICIES['LRU'], 1)
        
        # Store entry
        entry_data = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl or self.default_ttl,
            'access_count': 0,
            'confidence': confidence,
            'tags': tags or [],
            'original_query': query
        }
        
        if metadata:
            entry_data.update(metadata)
        
        self.storage[key] = entry_data
        self.metadata[key] = entry_data.copy()
        self.confidence_scores[key] = confidence
        self.access_counts[key] = 0
        self.last_access[key] = time.time()
        
        # Update tag mappings
        for tag in (tags or []):
            self.tag_to_keys[tag].add(key)
        
        # Check invalidation rules
        rules = self._should_invalidate_by_rule(key, entry_data)
        if rules and self.invalidation_strategy == INVALIDATION_STRATEGIES['IMMEDIATE']:
            self._apply_invalidation_rules(key, rules)
        
        return key
    
    def get(self, query: str) -> Optional[Any]:
        """Get cache entry with invalidation checks."""
        key = self._generate_key(query)
        
        # Check for expired entries
        self._cleanup_expired()
        
        if key in self.storage:
            entry = self.storage[key]
            
            # Check TTL expiration
            if self._is_expired(entry):
                self._invalidate_entry(key, INVALIDATION_TRIGGERS['TIME_BASED'], 
                                     'TTL expired')
                self.misses += 1
                return None
            
            # Update access patterns
            self.hits += 1
            self.access_counts[key] += 1
            self.last_access[key] = time.time()
            entry['access_count'] = self.access_counts[key]
            
            # Move to end for LRU
            self.storage.move_to_end(key)
            
            # Check for rule-based invalidation after access
            rules = self._should_invalidate_by_rule(key, entry)
            if rules:
                self._apply_invalidation_rules(key, rules)
                return None
            
            return entry['value']
        
        self.misses += 1
        return None
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry has expired."""
        return time.time() > (entry['timestamp'] + entry['ttl'])
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in list(self.storage.items()):
            if self._is_expired(entry):
                expired_keys.append(key)
        
        if expired_keys:
            self._invalidate_entries(expired_keys, INVALIDATION_TRIGGERS['TIME_BASED'],
                                   'TTL expiration cleanup')
    
    def invalidate(self, query: str, reason: str = "Manual invalidation") -> bool:
        """Manually invalidate cache entry."""
        key = self._generate_key(query)
        return self._invalidate_entry(key, INVALIDATION_TRIGGERS['MANUAL'], reason)
    
    def invalidate_by_pattern(self, pattern: str, reason: str = "Pattern invalidation") -> int:
        """Invalidate entries matching pattern."""
        matching_keys = []
        
        for key in list(self.storage.keys()):
            entry = self.storage[key]
            original_query = entry.get('original_query', '')
            
            if re.search(pattern, original_query, re.IGNORECASE):
                matching_keys.append(key)
        
        if matching_keys:
            self._invalidate_entries(matching_keys, INVALIDATION_TRIGGERS['PATTERN_BASED'],
                                   f"Pattern match: {pattern}")
        
        return len(matching_keys)
    
    def invalidate_by_tags(self, tags: List[str], reason: str = "Tag invalidation") -> int:
        """Invalidate entries with specified tags."""
        keys_to_invalidate = set()
        
        for tag in tags:
            if tag in self.tag_to_keys:
                keys_to_invalidate.update(self.tag_to_keys[tag])
        
        if keys_to_invalidate:
            self._invalidate_entries(list(keys_to_invalidate), 
                                   INVALIDATION_TRIGGERS['PATTERN_BASED'],
                                   f"Tag invalidation: {tags}")
        
        return len(keys_to_invalidate)
    
    def invalidate_by_confidence(self, threshold: float, 
                               reason: str = "Confidence threshold") -> int:
        """Invalidate entries below confidence threshold."""
        keys_to_invalidate = []
        
        for key, confidence in self.confidence_scores.items():
            if confidence < threshold and key in self.storage:
                keys_to_invalidate.append(key)
        
        if keys_to_invalidate:
            self._invalidate_entries(keys_to_invalidate,
                                   INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
                                   f"Confidence below {threshold}")
        
        return len(keys_to_invalidate)
    
    def invalidate_by_access_count(self, max_count: int,
                                 reason: str = "Low access count") -> int:
        """Invalidate entries with low access counts."""
        keys_to_invalidate = []
        
        for key, count in self.access_counts.items():
            if count <= max_count and key in self.storage:
                keys_to_invalidate.append(key)
        
        if keys_to_invalidate:
            self._invalidate_entries(keys_to_invalidate,
                                   INVALIDATION_TRIGGERS['ACCESS_BASED'],
                                   f"Access count <= {max_count}")
        
        return len(keys_to_invalidate)
    
    def _invalidate_by_policy(self, policy: str, count: int) -> int:
        """Invalidate entries according to policy."""
        if not self.storage:
            return 0
        
        keys_to_invalidate = []
        
        if policy == INVALIDATION_POLICIES['LRU']:
            # Get least recently used entries
            sorted_by_access = sorted(
                self.last_access.items(),
                key=lambda x: x[1]
            )
            keys_to_invalidate = [k for k, _ in sorted_by_access[:count]]
        
        elif policy == INVALIDATION_POLICIES['LFU']:
            # Get least frequently used entries
            sorted_by_count = sorted(
                self.access_counts.items(),
                key=lambda x: x[1]
            )
            keys_to_invalidate = [k for k, _ in sorted_by_count[:count]]
        
        elif policy == INVALIDATION_POLICIES['CONFIDENCE_WEIGHTED']:
            # Get lowest confidence entries
            sorted_by_confidence = sorted(
                self.confidence_scores.items(),
                key=lambda x: x[1]
            )
            keys_to_invalidate = [k for k, _ in sorted_by_confidence[:count]]
        
        elif policy == INVALIDATION_POLICIES['FIFO']:
            # Get oldest entries
            keys_to_invalidate = list(self.storage.keys())[:count]
        
        if keys_to_invalidate:
            self._invalidate_entries(keys_to_invalidate,
                                   INVALIDATION_TRIGGERS['SIZE_BASED'],
                                   f"Policy eviction: {policy}")
        
        return len(keys_to_invalidate)
    
    def _invalidate_entry(self, key: str, trigger: str, reason: str) -> bool:
        """Invalidate single cache entry."""
        if key not in self.storage:
            return False
        
        return self._invalidate_entries([key], trigger, reason) > 0
    
    def _invalidate_entries(self, keys: List[str], trigger: str, reason: str) -> int:
        """Invalidate multiple cache entries."""
        start_time = time.time()
        invalidated_keys = []
        
        for key in keys:
            if key in self.storage:
                # Remove from storage
                entry = self.storage.pop(key, None)
                self.metadata.pop(key, None)
                
                # Clean up tracking data
                self.access_counts.pop(key, None)
                self.last_access.pop(key, None)
                self.confidence_scores.pop(key, None)
                
                # Clean up tag mappings
                if entry and 'tags' in entry:
                    for tag in entry['tags']:
                        if tag in self.tag_to_keys:
                            self.tag_to_keys[tag].discard(key)
                            if not self.tag_to_keys[tag]:
                                del self.tag_to_keys[tag]
                
                invalidated_keys.append(key)
        
        # Record invalidation event
        if invalidated_keys:
            event = InvalidationEvent(
                event_id=f"inv_{int(time.time() * 1000)}_{len(self.invalidation_events)}",
                trigger=trigger,
                strategy=self.invalidation_strategy,
                timestamp=start_time,
                affected_keys=invalidated_keys,
                reason=reason,
                performance_impact=time.time() - start_time,
                success=True
            )
            self._record_invalidation(event)
        
        return len(invalidated_keys)
    
    def _apply_invalidation_rules(self, key: str, rules: List[InvalidationRule]):
        """Apply invalidation rules to entry."""
        for rule in rules:
            if rule.action == 'invalidate':
                self._invalidate_entry(key, rule.trigger, f"Rule: {rule.rule_id}")
            elif rule.action == 'defer':
                self.pending_invalidations.add(key)
            elif rule.action == 'queue':
                self.invalidation_queue.append((key, rule.trigger, rule.metadata))
    
    def add_invalidation_rule(self, rule: InvalidationRule):
        """Add invalidation rule."""
        self.invalidation_rules.append(rule)
    
    def remove_invalidation_rule(self, rule_id: str) -> bool:
        """Remove invalidation rule."""
        initial_count = len(self.invalidation_rules)
        self.invalidation_rules = [r for r in self.invalidation_rules if r.rule_id != rule_id]
        return len(self.invalidation_rules) < initial_count
    
    def bulk_invalidate(self, queries: List[str], reason: str = "Bulk invalidation") -> int:
        """Invalidate multiple queries in batch."""
        keys = [self._generate_key(query) for query in queries]
        return self._invalidate_entries(keys, INVALIDATION_TRIGGERS['MANUAL'], reason)
    
    def clear_cache(self, reason: str = "Cache cleared") -> int:
        """Clear entire cache."""
        keys = list(self.storage.keys())
        return self._invalidate_entries(keys, INVALIDATION_TRIGGERS['MANUAL'], reason)
    
    def process_deferred_invalidations(self) -> int:
        """Process pending invalidations."""
        if not self.pending_invalidations:
            return 0
        
        keys = list(self.pending_invalidations)
        self.pending_invalidations.clear()
        
        return self._invalidate_entries(keys, INVALIDATION_TRIGGERS['SYSTEM_STATE'],
                                      "Deferred invalidation processing")
    
    def background_cleanup(self) -> Dict[str, int]:
        """Perform background cleanup operations."""
        if not self._background_cleanup_enabled:
            return {}
        
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return {}
        
        results = {}
        
        # Clean up expired entries
        expired_count = len([k for k, v in self.storage.items() if self._is_expired(v)])
        if expired_count > 0:
            self._cleanup_expired()
            results['expired_cleaned'] = expired_count
        
        # Process deferred invalidations
        deferred_count = self.process_deferred_invalidations()
        if deferred_count > 0:
            results['deferred_processed'] = deferred_count
        
        # Process invalidation queue
        queue_count = len(self.invalidation_queue)
        if queue_count > 0:
            for key, trigger, metadata in self.invalidation_queue:
                self._invalidate_entry(key, trigger, "Queued invalidation")
            self.invalidation_queue.clear()
            results['queued_processed'] = queue_count
        
        self._last_cleanup = current_time
        return results
    
    def get_invalidation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive invalidation statistics."""
        trigger_counts = defaultdict(int)
        strategy_counts = defaultdict(int)
        
        for event in self.invalidation_events:
            trigger_counts[event.trigger] += 1
            strategy_counts[event.strategy] += 1
        
        return {
            'total_invalidations': self.invalidations,
            'invalidation_events': len(self.invalidation_events),
            'trigger_breakdown': dict(trigger_counts),
            'strategy_breakdown': dict(strategy_counts),
            'pending_invalidations': len(self.pending_invalidations),
            'invalidation_queue_size': len(self.invalidation_queue),
            'active_rules': len([r for r in self.invalidation_rules if r.enabled]),
            'total_rules': len(self.invalidation_rules),
            'cache_size': len(self.storage),
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
    
    def get_invalidation_events(self, limit: Optional[int] = None) -> List[InvalidationEvent]:
        """Get recent invalidation events."""
        events = sorted(self.invalidation_events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit] if limit else events


class TestInvalidationTriggers:
    """Tests for cache invalidation trigger mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache(max_size=10, default_ttl=3600)
        self.data_generator = BiomedicalTestDataGenerator()
    
    def test_time_based_invalidation(self):
        """Test time-based TTL expiration invalidation."""
        query = "What is glucose metabolism?"
        value = "Glucose metabolism involves glycolysis and gluconeogenesis"
        
        # Set entry with short TTL
        self.cache.set(query, value, ttl=1)
        
        # Should be retrievable immediately
        result = self.cache.get(query)
        assert result == value
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should trigger time-based invalidation
        result = self.cache.get(query)
        assert result is None
        
        # Verify invalidation event was recorded
        stats = self.cache.get_invalidation_statistics()
        assert stats['total_invalidations'] > 0
        
        events = self.cache.get_invalidation_events()
        assert any(e.trigger == INVALIDATION_TRIGGERS['TIME_BASED'] for e in events)
    
    def test_size_based_invalidation_lru(self):
        """Test size-limit based LRU eviction invalidation."""
        # Fill cache to capacity
        queries = [f"Query {i}" for i in range(self.cache.max_size)]
        values = [f"Value {i}" for i in range(self.cache.max_size)]
        
        for query, value in zip(queries, values):
            self.cache.set(query, value)
        
        # Cache should be full
        assert len(self.cache.storage) == self.cache.max_size
        
        # Add one more entry - should trigger LRU eviction
        overflow_query = "Overflow query"
        self.cache.set(overflow_query, "Overflow value")
        
        # Cache size should remain at max
        assert len(self.cache.storage) == self.cache.max_size
        
        # First query should be evicted (LRU)
        result = self.cache.get(queries[0])
        assert result is None
        
        # Verify size-based invalidation event
        events = self.cache.get_invalidation_events()
        assert any(e.trigger == INVALIDATION_TRIGGERS['SIZE_BASED'] for e in events)
    
    def test_manual_invalidation(self):
        """Test manual invalidation operations."""
        query = "Test query for manual invalidation"
        value = "Test value"
        
        # Set entry
        self.cache.set(query, value)
        assert self.cache.get(query) == value
        
        # Manually invalidate
        success = self.cache.invalidate(query, "Testing manual invalidation")
        assert success
        
        # Should be invalidated
        result = self.cache.get(query)
        assert result is None
        
        # Verify manual invalidation event
        events = self.cache.get_invalidation_events()
        manual_events = [e for e in events if e.trigger == INVALIDATION_TRIGGERS['MANUAL']]
        assert len(manual_events) > 0
        assert "Testing manual invalidation" in manual_events[0].reason
    
    def test_pattern_based_invalidation(self):
        """Test pattern-based invalidation."""
        # Set entries with different patterns
        queries = [
            "What is glucose metabolism?",
            "How does glucose regulation work?",
            "What are lipid pathways?",
            "How does protein synthesis work?"
        ]
        
        for i, query in enumerate(queries):
            self.cache.set(query, f"Value {i}")
        
        # All should be retrievable
        for query in queries:
            assert self.cache.get(query) is not None
        
        # Invalidate all glucose-related queries
        invalidated_count = self.cache.invalidate_by_pattern("glucose")
        assert invalidated_count == 2  # Two queries contain "glucose"
        
        # Glucose queries should be invalidated
        assert self.cache.get("What is glucose metabolism?") is None
        assert self.cache.get("How does glucose regulation work?") is None
        
        # Non-glucose queries should remain
        assert self.cache.get("What are lipid pathways?") is not None
        assert self.cache.get("How does protein synthesis work?") is not None
        
        # Verify pattern-based invalidation events
        events = self.cache.get_invalidation_events()
        pattern_events = [e for e in events if e.trigger == INVALIDATION_TRIGGERS['PATTERN_BASED']]
        assert len(pattern_events) > 0
    
    def test_system_state_invalidation(self):
        """Test system state change invalidation."""
        # Set up invalidation rule based on system state
        rule = InvalidationRule(
            rule_id="system_state_test",
            trigger=INVALIDATION_TRIGGERS['SYSTEM_STATE'],
            condition="tag:temporal",
            action="invalidate",
            priority=50
        )
        
        self.cache.add_invalidation_rule(rule)
        
        # Add entries with temporal tags
        self.cache.set("Current COVID research", "Latest findings", 
                      tags=["temporal", "current"])
        self.cache.set("Historical data", "Past research", 
                      tags=["historical"])
        
        # Both should be retrievable initially
        assert self.cache.get("Current COVID research") is not None
        assert self.cache.get("Historical data") is not None
        
        # Invalidate by temporal tag (simulating system state change)
        invalidated_count = self.cache.invalidate_by_tags(["temporal"])
        assert invalidated_count == 1
        
        # Temporal entry should be invalidated
        assert self.cache.get("Current COVID research") is None
        # Non-temporal entry should remain
        assert self.cache.get("Historical data") is not None


class TestManualInvalidation:
    """Tests for manual and programmatic invalidation operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache()
        self.biomedical_queries = BIOMEDICAL_QUERIES['metabolism'][:5]
    
    def test_single_entry_invalidation(self):
        """Test invalidating single cache entry."""
        query = "Single invalidation test"
        value = "Test data"
        
        self.cache.set(query, value)
        assert self.cache.get(query) == value
        
        # Manual invalidation
        success = self.cache.invalidate(query, "Single entry test")
        assert success
        assert self.cache.get(query) is None
        
        # Verify invalidation statistics
        stats = self.cache.get_invalidation_statistics()
        assert stats['total_invalidations'] == 1
    
    def test_bulk_invalidation(self):
        """Test bulk invalidation of multiple entries."""
        queries = ["Bulk query 1", "Bulk query 2", "Bulk query 3"]
        values = ["Value 1", "Value 2", "Value 3"]
        
        # Set all entries
        for query, value in zip(queries, values):
            self.cache.set(query, value)
        
        # Verify all are retrievable
        for query, value in zip(queries, values):
            assert self.cache.get(query) == value
        
        # Bulk invalidation
        invalidated_count = self.cache.bulk_invalidate(queries, "Bulk invalidation test")
        assert invalidated_count == len(queries)
        
        # All should be invalidated
        for query in queries:
            assert self.cache.get(query) is None
        
        # Verify bulk invalidation event
        events = self.cache.get_invalidation_events()
        bulk_events = [e for e in events if "Bulk invalidation" in e.reason]
        assert len(bulk_events) > 0
        assert len(bulk_events[0].affected_keys) == len(queries)
    
    def test_invalidation_with_biomedical_data(self):
        """Test invalidation with realistic biomedical queries."""
        # Set biomedical queries
        for query_data in self.biomedical_queries:
            query = query_data['query']
            response = query_data['response']
            self.cache.set(query, response, confidence=response['confidence'])
        
        initial_count = len(self.cache.storage)
        assert initial_count == len(self.biomedical_queries)
        
        # Invalidate specific metabolic queries
        metabolic_queries = [q['query'] for q in self.biomedical_queries if 'metabol' in q['query'].lower()]
        invalidated_count = self.cache.bulk_invalidate(metabolic_queries, "Metabolic data update")
        
        assert invalidated_count > 0
        assert len(self.cache.storage) == initial_count - invalidated_count
        
        # Verify metabolic queries are invalidated
        for query in metabolic_queries:
            assert self.cache.get(query) is None
    
    def test_nonexistent_entry_invalidation(self):
        """Test invalidating non-existent entries."""
        # Attempt to invalidate non-existent entry
        success = self.cache.invalidate("Non-existent query", "Test non-existent")
        assert not success
        
        # Statistics should not change
        stats = self.cache.get_invalidation_statistics()
        assert stats['total_invalidations'] == 0
    
    def test_clear_entire_cache(self):
        """Test clearing entire cache."""
        # Fill cache
        for i in range(10):
            self.cache.set(f"Query {i}", f"Value {i}")
        
        assert len(self.cache.storage) == 10
        
        # Clear cache
        cleared_count = self.cache.clear_cache("Testing cache clear")
        assert cleared_count == 10
        assert len(self.cache.storage) == 0
        
        # Verify clear event
        events = self.cache.get_invalidation_events()
        clear_events = [e for e in events if "cache clear" in e.reason.lower()]
        assert len(clear_events) > 0
        assert len(clear_events[0].affected_keys) == 10


class TestPatternBasedInvalidation:
    """Tests for pattern and metadata-based invalidation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache()
    
    def test_regex_pattern_invalidation(self):
        """Test invalidation using regex patterns."""
        # Set entries with different query patterns
        test_data = [
            ("What causes diabetes?", "Diabetes causes"),
            ("How to treat diabetes?", "Diabetes treatment"),
            ("What is cancer?", "Cancer information"),
            ("How does insulin work?", "Insulin mechanism"),
            ("Latest diabetes research 2024", "Current diabetes research")
        ]
        
        for query, value in test_data:
            self.cache.set(query, value)
        
        # Invalidate diabetes-related queries using pattern
        invalidated_count = self.cache.invalidate_by_pattern(r"diabetes")
        assert invalidated_count == 3
        
        # Verify diabetes queries are invalidated
        assert self.cache.get("What causes diabetes?") is None
        assert self.cache.get("How to treat diabetes?") is None
        assert self.cache.get("Latest diabetes research 2024") is None
        
        # Non-diabetes queries should remain
        assert self.cache.get("What is cancer?") is not None
        assert self.cache.get("How does insulin work?") is not None
    
    def test_case_insensitive_pattern_matching(self):
        """Test case-insensitive pattern matching."""
        test_queries = [
            "GLUCOSE metabolism pathway",
            "glucose transport mechanism", 
            "Glucose regulation in diabetes",
            "protein synthesis pathway"
        ]
        
        for i, query in enumerate(test_queries):
            self.cache.set(query, f"Value {i}")
        
        # Case-insensitive pattern matching
        invalidated_count = self.cache.invalidate_by_pattern(r"glucose")
        assert invalidated_count == 3
        
        # Verify case-insensitive matching worked
        assert self.cache.get("GLUCOSE metabolism pathway") is None
        assert self.cache.get("glucose transport mechanism") is None
        assert self.cache.get("Glucose regulation in diabetes") is None
        assert self.cache.get("protein synthesis pathway") is not None
    
    def test_tag_based_invalidation(self):
        """Test invalidation based on entry tags."""
        # Set entries with different tags
        self.cache.set("Current COVID research", "Latest COVID findings", 
                      tags=["temporal", "covid", "current"])
        self.cache.set("COVID historical data", "Past COVID research",
                      tags=["covid", "historical"])
        self.cache.set("Cancer research", "Cancer findings",
                      tags=["oncology", "current"])
        self.cache.set("Diabetes treatment", "Treatment options",
                      tags=["diabetes", "treatment"])
        
        # Invalidate current research (temporal tag)
        invalidated_count = self.cache.invalidate_by_tags(["temporal"])
        assert invalidated_count == 1
        assert self.cache.get("Current COVID research") is None
        
        # Other entries should remain
        assert self.cache.get("COVID historical data") is not None
        assert self.cache.get("Cancer research") is not None
        
        # Invalidate all COVID-related entries
        invalidated_count = self.cache.invalidate_by_tags(["covid"])
        assert invalidated_count == 1  # Only historical COVID left
        assert self.cache.get("COVID historical data") is None
        
        # Non-COVID entries should remain
        assert self.cache.get("Cancer research") is not None
        assert self.cache.get("Diabetes treatment") is not None
    
    def test_multi_tag_invalidation(self):
        """Test invalidation with multiple tags."""
        # Set entries with overlapping tags
        self.cache.set("Entry 1", "Value 1", tags=["tag1", "tag2"])
        self.cache.set("Entry 2", "Value 2", tags=["tag2", "tag3"])
        self.cache.set("Entry 3", "Value 3", tags=["tag3", "tag4"])
        self.cache.set("Entry 4", "Value 4", tags=["tag4"])
        
        # Invalidate by multiple tags
        invalidated_count = self.cache.invalidate_by_tags(["tag1", "tag3"])
        
        # Should invalidate entries with either tag1 OR tag3
        assert invalidated_count == 3  # Entry 1 (tag1), Entry 2 (tag3), Entry 3 (tag3)
        
        assert self.cache.get("Entry 1") is None
        assert self.cache.get("Entry 2") is None  
        assert self.cache.get("Entry 3") is None
        assert self.cache.get("Entry 4") is not None  # Only tag4
    
    def test_complex_pattern_matching(self):
        """Test complex regex patterns for invalidation."""
        # Set entries with various biomedical terms
        biomedical_queries = [
            "What are the effects of drug-123 on metabolism?",
            "How does drug-456 interact with proteins?",
            "What is the mechanism of compound-XYZ?",
            "How does medication-ABC affect glucose levels?",
            "What are the side effects of therapy-789?"
        ]
        
        for i, query in enumerate(biomedical_queries):
            self.cache.set(query, f"Response {i}")
        
        # Complex pattern: drug followed by dash and numbers
        drug_pattern = r"drug-\d+"
        invalidated_count = self.cache.invalidate_by_pattern(drug_pattern)
        assert invalidated_count == 2
        
        # Pattern matching word boundaries
        compound_pattern = r"\bcompound-"
        invalidated_count = self.cache.invalidate_by_pattern(compound_pattern)
        assert invalidated_count == 1
        
        # Only medication and therapy entries should remain
        remaining_entries = sum(1 for query in biomedical_queries 
                              if self.cache.get(query) is not None)
        assert remaining_entries == 2


class TestAccessBasedInvalidation:
    """Tests for access pattern and confidence-based invalidation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache()
    
    def test_confidence_based_invalidation(self):
        """Test invalidation based on confidence scores."""
        # Set entries with different confidence levels
        confidence_data = [
            ("High confidence query", "Reliable data", 0.95),
            ("Medium confidence query", "Moderate data", 0.75),
            ("Low confidence query 1", "Uncertain data 1", 0.45),
            ("Low confidence query 2", "Uncertain data 2", 0.35),
            ("Very low confidence query", "Very uncertain data", 0.25)
        ]
        
        for query, value, confidence in confidence_data:
            self.cache.set(query, value, confidence=confidence)
        
        # Invalidate entries below confidence threshold
        invalidated_count = self.cache.invalidate_by_confidence(0.5)
        assert invalidated_count == 3  # Three queries below 0.5 confidence
        
        # High and medium confidence should remain
        assert self.cache.get("High confidence query") is not None
        assert self.cache.get("Medium confidence query") is not None
        
        # Low confidence queries should be invalidated
        assert self.cache.get("Low confidence query 1") is None
        assert self.cache.get("Low confidence query 2") is None
        assert self.cache.get("Very low confidence query") is None
        
        # Verify confidence-based invalidation events
        events = self.cache.get_invalidation_events()
        confidence_events = [e for e in events if e.trigger == INVALIDATION_TRIGGERS['CONFIDENCE_BASED']]
        assert len(confidence_events) > 0
    
    def test_access_count_based_invalidation(self):
        """Test invalidation based on access count patterns."""
        # Set multiple entries
        queries = [f"Access test query {i}" for i in range(5)]
        for query in queries:
            self.cache.set(query, f"Value for {query}")
        
        # Simulate different access patterns
        access_patterns = [5, 3, 1, 0, 2]  # Number of accesses per query
        
        for query, access_count in zip(queries, access_patterns):
            for _ in range(access_count):
                self.cache.get(query)  # This increments access count
        
        # Invalidate entries with low access count (≤ 1)
        invalidated_count = self.cache.invalidate_by_access_count(1)
        assert invalidated_count == 2  # Two queries with access count ≤ 1
        
        # High-access queries should remain
        assert self.cache.get(queries[0]) is not None  # 5 accesses
        assert self.cache.get(queries[1]) is not None  # 3 accesses
        assert self.cache.get(queries[4]) is not None  # 2 accesses
        
        # Low-access queries should be invalidated
        assert self.cache.get(queries[2]) is None  # 1 access
        assert self.cache.get(queries[3]) is None  # 0 accesses
    
    def test_lru_access_pattern_invalidation(self):
        """Test LRU-based access pattern invalidation."""
        # Fill cache to capacity
        queries = [f"LRU test query {i}" for i in range(self.cache.max_size)]
        for query in queries:
            self.cache.set(query, f"Value for {query}")
        
        # Access some entries to update their LRU order
        recently_accessed = queries[-3:]  # Access last 3 entries
        for query in recently_accessed:
            self.cache.get(query)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Force LRU eviction by adding new entry
        new_query = "New overflow query"
        self.cache.set(new_query, "New value")
        
        # Least recently used entry should be evicted
        # (First entry that wasn't recently accessed)
        lru_candidate = queries[0]
        assert self.cache.get(lru_candidate) is None
        
        # Recently accessed entries should still be there
        for query in recently_accessed:
            assert self.cache.get(query) is not None
    
    def test_combined_access_confidence_invalidation(self):
        """Test invalidation combining access patterns and confidence."""
        # Set entries with different access/confidence combinations
        test_data = [
            ("Popular high confidence", "Data 1", 0.9, 5),  # High conf, high access
            ("Popular low confidence", "Data 2", 0.4, 5),   # Low conf, high access
            ("Unpopular high confidence", "Data 3", 0.9, 1), # High conf, low access
            ("Unpopular low confidence", "Data 4", 0.4, 1)   # Low conf, low access
        ]
        
        for query, value, confidence, access_count in test_data:
            self.cache.set(query, value, confidence=confidence)
            # Simulate access pattern
            for _ in range(access_count):
                self.cache.get(query)
        
        # First, invalidate by low confidence
        confidence_invalidated = self.cache.invalidate_by_confidence(0.7)
        assert confidence_invalidated == 2  # Two low confidence entries
        
        # Then, invalidate by low access count among remaining
        access_invalidated = self.cache.invalidate_by_access_count(2)
        # Should invalidate "Unpopular high confidence" (1 access)
        assert access_invalidated == 1
        
        # Only "Popular high confidence" should remain
        assert self.cache.get("Popular high confidence") is not None
        assert self.cache.get("Popular low confidence") is None
        assert self.cache.get("Unpopular high confidence") is None
        assert self.cache.get("Unpopular low confidence") is None


class TestInvalidationStrategies:
    """Tests for different invalidation strategy implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def test_immediate_invalidation_strategy(self):
        """Test immediate invalidation strategy."""
        cache = MockInvalidatingCache(invalidation_strategy=INVALIDATION_STRATEGIES['IMMEDIATE'])
        
        # Add rule for immediate invalidation
        rule = InvalidationRule(
            rule_id="immediate_test",
            trigger=INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
            condition="confidence < 0.5",
            action="invalidate",
            priority=100
        )
        cache.add_invalidation_rule(rule)
        
        # Set low confidence entry - should be immediately invalidated
        cache.set("Low confidence entry", "Uncertain data", confidence=0.3)
        
        # Should not be retrievable due to immediate invalidation
        result = cache.get("Low confidence entry")
        assert result is None
        
        # Verify immediate invalidation event
        events = cache.get_invalidation_events()
        assert len(events) > 0
        assert events[0].strategy == INVALIDATION_STRATEGIES['IMMEDIATE']
    
    def test_deferred_invalidation_strategy(self):
        """Test deferred invalidation strategy."""
        cache = MockInvalidatingCache(invalidation_strategy=INVALIDATION_STRATEGIES['DEFERRED'])
        
        # Add rule for deferred invalidation
        rule = InvalidationRule(
            rule_id="deferred_test",
            trigger=INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
            condition="confidence < 0.5",
            action="defer",
            priority=100
        )
        cache.add_invalidation_rule(rule)
        
        # Set low confidence entry - should be deferred, not immediately invalidated
        cache.set("Deferred entry", "Uncertain data", confidence=0.3)
        
        # Should still be retrievable (deferred invalidation)
        result = cache.get("Deferred entry")
        assert result == "Uncertain data"
        
        # Should be in pending invalidations
        stats = cache.get_invalidation_statistics()
        assert stats['pending_invalidations'] > 0
        
        # Process deferred invalidations
        processed_count = cache.process_deferred_invalidations()
        assert processed_count > 0
        
        # Now should be invalidated
        result = cache.get("Deferred entry")
        assert result is None
    
    def test_batch_invalidation_strategy(self):
        """Test batch invalidation strategy."""
        cache = MockInvalidatingCache(invalidation_strategy=INVALIDATION_STRATEGIES['BATCH'])
        
        # Set multiple entries for batch invalidation
        queries = [f"Batch test query {i}" for i in range(5)]
        for query in queries:
            cache.set(query, f"Value for {query}")
        
        # Batch invalidate multiple entries
        invalidated_count = cache.bulk_invalidate(queries, "Batch strategy test")
        assert invalidated_count == len(queries)
        
        # All should be invalidated
        for query in queries:
            assert cache.get(query) is None
        
        # Should have single batch invalidation event
        events = cache.get_invalidation_events()
        batch_events = [e for e in events if "Batch strategy" in e.reason]
        assert len(batch_events) == 1
        assert len(batch_events[0].affected_keys) == len(queries)
    
    def test_background_invalidation_strategy(self):
        """Test background invalidation strategy."""
        cache = MockInvalidatingCache(invalidation_strategy=INVALIDATION_STRATEGIES['BACKGROUND'])
        cache._cleanup_interval = 0.1  # Very short interval for testing
        
        # Set entries that will expire
        for i in range(3):
            cache.set(f"Expiring query {i}", f"Value {i}", ttl=0.5)
        
        # Set entry for deferred invalidation
        cache.pending_invalidations.add("deferred_key")
        
        # Set entry for queued invalidation
        cache.invalidation_queue.append(("queued_key", INVALIDATION_TRIGGERS['MANUAL'], {}))
        
        # Wait for entries to expire
        time.sleep(0.6)
        
        # Trigger background cleanup
        cleanup_results = cache.background_cleanup()
        
        # Should have cleaned up expired, deferred, and queued entries
        assert 'expired_cleaned' in cleanup_results or 'deferred_processed' in cleanup_results
        
        # Verify background cleanup worked
        stats = cache.get_invalidation_statistics()
        assert stats['total_invalidations'] > 0


class TestBulkInvalidation:
    """Tests for bulk and batch invalidation operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache()
    
    def test_bulk_invalidation_performance(self):
        """Test performance of bulk invalidation operations."""
        # Set large number of entries
        num_entries = 100
        queries = [f"Bulk performance test {i}" for i in range(num_entries)]
        
        # Measure set performance
        start_time = time.time()
        for query in queries:
            self.cache.set(query, f"Value for {query}")
        set_time = time.time() - start_time
        
        # Measure bulk invalidation performance
        start_time = time.time()
        invalidated_count = self.cache.bulk_invalidate(queries, "Performance test")
        bulk_invalidation_time = time.time() - start_time
        
        assert invalidated_count == num_entries
        
        # Bulk invalidation should be reasonably fast
        assert bulk_invalidation_time < set_time * 2  # At most 2x slower than setting
        
        # Verify all are invalidated
        for query in queries:
            assert self.cache.get(query) is None
    
    def test_partial_bulk_invalidation(self):
        """Test bulk invalidation with some non-existent entries."""
        existing_queries = ["Existing 1", "Existing 2", "Existing 3"]
        non_existing_queries = ["Non-existing 1", "Non-existing 2"]
        
        # Set only existing queries
        for query in existing_queries:
            self.cache.set(query, f"Value for {query}")
        
        # Attempt bulk invalidation including non-existing entries
        all_queries = existing_queries + non_existing_queries
        invalidated_count = self.cache.bulk_invalidate(all_queries, "Partial bulk test")
        
        # Should only invalidate existing entries
        assert invalidated_count == len(existing_queries)
        
        # Existing queries should be invalidated
        for query in existing_queries:
            assert self.cache.get(query) is None
    
    def test_bulk_invalidation_with_patterns(self):
        """Test bulk invalidation combined with pattern matching."""
        # Set entries with different categories
        diabetes_queries = [
            "What causes diabetes type 1?",
            "How to manage diabetes type 2?",
            "Diabetes complications overview"
        ]
        
        cancer_queries = [
            "What are cancer risk factors?",
            "How is cancer diagnosed?",
            "Cancer treatment options"
        ]
        
        other_queries = [
            "What is hypertension?",
            "How does insulin work?"
        ]
        
        all_queries = diabetes_queries + cancer_queries + other_queries
        for query in all_queries:
            self.cache.set(query, f"Response for {query}")
        
        # Bulk invalidate specific categories
        diabetes_invalidated = self.cache.invalidate_by_pattern("diabetes")
        assert diabetes_invalidated == len(diabetes_queries)
        
        cancer_invalidated = self.cache.invalidate_by_pattern("cancer")
        assert cancer_invalidated == len(cancer_queries)
        
        # Only other queries should remain
        for query in other_queries:
            assert self.cache.get(query) is not None
        
        for query in diabetes_queries + cancer_queries:
            assert self.cache.get(query) is None
    
    def test_bulk_operations_thread_safety(self):
        """Test thread safety of bulk invalidation operations."""
        num_threads = 5
        entries_per_thread = 10
        
        def worker(thread_id):
            # Each thread adds entries
            queries = [f"Thread {thread_id} Query {i}" for i in range(entries_per_thread)]
            for query in queries:
                self.cache.set(query, f"Value from thread {thread_id}")
            
            # Each thread bulk invalidates its own entries
            invalidated = self.cache.bulk_invalidate(queries, f"Thread {thread_id} cleanup")
            return thread_id, invalidated
        
        # Run concurrent bulk operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all threads completed successfully
        assert len(results) == num_threads
        
        for thread_id, invalidated_count in results:
            assert invalidated_count == entries_per_thread
        
        # Cache should be empty or nearly empty
        assert len(self.cache.storage) == 0
        
        # Verify invalidation events from all threads
        events = self.cache.get_invalidation_events()
        thread_events = [e for e in events if "Thread" in e.reason]
        assert len(thread_events) == num_threads


class TestConditionalInvalidation:
    """Tests for conditional and rule-based invalidation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache()
    
    def test_rule_based_invalidation(self):
        """Test invalidation using custom rules."""
        # Add invalidation rules
        rules = [
            InvalidationRule(
                rule_id="old_entries",
                trigger=INVALIDATION_TRIGGERS['TIME_BASED'],
                condition="age > 2",  # Entries older than 2 seconds
                action="invalidate",
                priority=100
            ),
            InvalidationRule(
                rule_id="low_access",
                trigger=INVALIDATION_TRIGGERS['ACCESS_BASED'],
                condition="access_count < 2",
                action="invalidate",
                priority=200
            )
        ]
        
        for rule in rules:
            self.cache.add_invalidation_rule(rule)
        
        # Set test entries
        self.cache.set("Old entry", "Old data")
        time.sleep(0.1)
        self.cache.set("Recent entry", "Recent data")
        
        # Access recent entry multiple times
        for _ in range(3):
            self.cache.get("Recent entry")
        
        # Wait for age rule to trigger
        time.sleep(2.1)
        
        # Access entries to trigger rule evaluation
        old_result = self.cache.get("Old entry")  # Should trigger age rule
        recent_result = self.cache.get("Recent entry")  # Should not trigger any rule
        
        # Old entry should be invalidated by age rule
        assert old_result is None
        # Recent entry should remain (high access count, recent)
        assert recent_result is not None
    
    def test_conditional_invalidation_priority(self):
        """Test rule priority handling in conditional invalidation."""
        # Add rules with different priorities
        high_priority_rule = InvalidationRule(
            rule_id="high_priority",
            trigger=INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
            condition="confidence < 0.8",
            action="invalidate",
            priority=50  # Higher priority (lower number)
        )
        
        low_priority_rule = InvalidationRule(
            rule_id="low_priority",
            trigger=INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
            condition="confidence < 0.6",
            action="defer",
            priority=200  # Lower priority (higher number)
        )
        
        self.cache.add_invalidation_rule(high_priority_rule)
        self.cache.add_invalidation_rule(low_priority_rule)
        
        # Set entry that matches both rules
        self.cache.set("Priority test", "Test data", confidence=0.5)
        
        # High priority rule should execute first (invalidate action)
        result = self.cache.get("Priority test")
        assert result is None  # Should be immediately invalidated
        
        # Verify that high priority rule was applied
        events = self.cache.get_invalidation_events()
        rule_events = [e for e in events if "Rule:" in e.reason]
        assert any("high_priority" in e.reason for e in rule_events)
    
    def test_rule_enabling_disabling(self):
        """Test enabling and disabling invalidation rules."""
        rule = InvalidationRule(
            rule_id="toggle_test",
            trigger=INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
            condition="confidence < 0.7",
            action="invalidate",
            priority=100,
            enabled=True
        )
        
        self.cache.add_invalidation_rule(rule)
        
        # Set low confidence entry - should be invalidated
        self.cache.set("Test with rule enabled", "Data 1", confidence=0.5)
        result = self.cache.get("Test with rule enabled")
        assert result is None
        
        # Disable rule
        rule.enabled = False
        
        # Set another low confidence entry - should NOT be invalidated
        self.cache.set("Test with rule disabled", "Data 2", confidence=0.5)
        result = self.cache.get("Test with rule disabled")
        assert result is not None
        
        # Re-enable rule
        rule.enabled = True
        
        # Access the entry again - should now be invalidated
        result = self.cache.get("Test with rule disabled")
        assert result is None  # Should be invalidated when rule is re-enabled
    
    def test_complex_conditional_logic(self):
        """Test complex conditional invalidation logic."""
        # Add rule with complex condition
        complex_rule = InvalidationRule(
            rule_id="complex_condition",
            trigger=INVALIDATION_TRIGGERS['SYSTEM_STATE'],
            condition="tag:temporal",  # Entries tagged as temporal
            action="invalidate",
            priority=100
        )
        
        self.cache.add_invalidation_rule(complex_rule)
        
        # Set entries with different characteristics
        test_entries = [
            ("Current research", "Latest findings", ["temporal", "current"]),
            ("Historical data", "Past research", ["historical"]),
            ("Real-time data", "Live updates", ["temporal", "realtime"]),
            ("Static reference", "Reference data", ["reference"])
        ]
        
        for query, value, tags in test_entries:
            self.cache.set(query, value, tags=tags)
        
        # All should be initially retrievable
        for query, _, _ in test_entries:
            assert self.cache.get(query) is not None
        
        # Simulate system state change affecting temporal data
        temporal_invalidated = self.cache.invalidate_by_tags(["temporal"])
        assert temporal_invalidated == 2  # Two temporal entries
        
        # Temporal entries should be invalidated
        assert self.cache.get("Current research") is None
        assert self.cache.get("Real-time data") is None
        
        # Non-temporal entries should remain
        assert self.cache.get("Historical data") is not None
        assert self.cache.get("Static reference") is not None
    
    def test_rule_removal(self):
        """Test removal of invalidation rules."""
        # Add multiple rules
        rule1 = InvalidationRule("rule1", INVALIDATION_TRIGGERS['TIME_BASED'], 
                                "age > 1", "invalidate", 100)
        rule2 = InvalidationRule("rule2", INVALIDATION_TRIGGERS['CONFIDENCE_BASED'], 
                                "confidence < 0.5", "invalidate", 200)
        
        self.cache.add_invalidation_rule(rule1)
        self.cache.add_invalidation_rule(rule2)
        
        stats = self.cache.get_invalidation_statistics()
        assert stats['total_rules'] == 2
        
        # Remove one rule
        removed = self.cache.remove_invalidation_rule("rule1")
        assert removed
        
        stats = self.cache.get_invalidation_statistics()
        assert stats['total_rules'] == 1
        assert stats['active_rules'] == 1
        
        # Try to remove non-existent rule
        removed = self.cache.remove_invalidation_rule("non_existent")
        assert not removed
        
        stats = self.cache.get_invalidation_statistics()
        assert stats['total_rules'] == 1


class TestInvalidationEdgeCases:
    """Tests for edge cases and boundary conditions in invalidation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MockInvalidatingCache(max_size=5)
    
    def test_invalidation_during_iteration(self):
        """Test invalidation while iterating over cache entries."""
        # Fill cache
        queries = [f"Iteration test {i}" for i in range(self.cache.max_size)]
        for query in queries:
            self.cache.set(query, f"Value for {query}")
        
        # Simulate invalidation during iteration (common in cleanup operations)
        keys_to_process = list(self.cache.storage.keys())
        invalidated_during_iteration = []
        
        for key in keys_to_process:
            if key in self.cache.storage:
                # Simulate some condition that triggers invalidation
                if len(invalidated_during_iteration) < 2:
                    query = self.cache.storage[key]['original_query']
                    success = self.cache.invalidate(query, "Invalidated during iteration")
                    if success:
                        invalidated_during_iteration.append(key)
        
        assert len(invalidated_during_iteration) == 2
        assert len(self.cache.storage) == self.cache.max_size - 2
    
    def test_concurrent_invalidation_race_conditions(self):
        """Test race conditions in concurrent invalidation."""
        # Set entries for concurrent access
        test_queries = [f"Concurrent test {i}" for i in range(10)]
        for query in test_queries:
            self.cache.set(query, f"Value for {query}")
        
        results = []
        
        def invalidate_worker(queries_subset):
            worker_results = []
            for query in queries_subset:
                success = self.cache.invalidate(query, "Concurrent invalidation")
                worker_results.append((query, success))
            return worker_results
        
        # Split queries among concurrent workers
        mid_point = len(test_queries) // 2
        worker1_queries = test_queries[:mid_point]
        worker2_queries = test_queries[mid_point:]
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(invalidate_worker, worker1_queries)
            future2 = executor.submit(invalidate_worker, worker2_queries)
            
            results1 = future1.result()
            results2 = future2.result()
        
        # Count successful invalidations
        successful_invalidations = sum(1 for _, success in results1 + results2 if success)
        
        # All queries should have been successfully invalidated
        assert successful_invalidations == len(test_queries)
        
        # Cache should be empty
        assert len(self.cache.storage) == 0
    
    def test_invalidation_with_circular_references(self):
        """Test invalidation with complex object references."""
        # Create objects with potential circular references
        class MockBiomedicalData:
            def __init__(self, name: str):
                self.name = name
                self.references = []
            
            def add_reference(self, other):
                self.references.append(other)
        
        # Create circular reference structure
        data1 = MockBiomedicalData("Glucose metabolism")
        data2 = MockBiomedicalData("Insulin pathway")
        data3 = MockBiomedicalData("Diabetes pathophysiology")
        
        data1.add_reference(data2)
        data2.add_reference(data3)
        data3.add_reference(data1)  # Circular reference
        
        # Store in cache
        self.cache.set("Glucose query", data1)
        self.cache.set("Insulin query", data2)
        self.cache.set("Diabetes query", data3)
        
        # Invalidate entries with circular references
        invalidated_count = self.cache.bulk_invalidate(
            ["Glucose query", "Insulin query", "Diabetes query"],
            "Circular reference cleanup"
        )
        
        assert invalidated_count == 3
        
        # Verify all are invalidated
        assert self.cache.get("Glucose query") is None
        assert self.cache.get("Insulin query") is None
        assert self.cache.get("Diabetes query") is None
    
    def test_invalidation_memory_cleanup(self):
        """Test proper memory cleanup during invalidation."""
        # Create large objects to test memory cleanup
        large_data = "x" * 10000  # 10KB string
        
        queries = [f"Memory test {i}" for i in range(10)]
        for query in queries:
            self.cache.set(query, large_data)
        
        # Verify objects are stored
        assert len(self.cache.storage) == 10
        
        # Create weak references to track garbage collection
        weak_refs = []
        for key in list(self.cache.storage.keys()):
            entry = self.cache.storage[key]
            weak_refs.append(weakref.ref(entry))
        
        # Bulk invalidate all entries
        invalidated_count = self.cache.bulk_invalidate(queries, "Memory cleanup test")
        assert invalidated_count == 10
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check that weak references are cleaned up
        # (This test is somewhat implementation-dependent)
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        
        # At least some references should be cleaned up
        assert alive_refs < len(weak_refs)
    
    def test_invalidation_with_malformed_patterns(self):
        """Test invalidation with malformed regex patterns."""
        # Set test entries
        self.cache.set("Test query 1", "Value 1")
        self.cache.set("Test query 2", "Value 2")
        
        # Test with malformed regex patterns
        malformed_patterns = [
            "[unclosed",
            "(?P<incomplete",
            "*invalid",
            "(?invalid)",
            "\\invalid_escape"
        ]
        
        for pattern in malformed_patterns:
            # Should not crash, should return 0 invalidations
            try:
                invalidated_count = self.cache.invalidate_by_pattern(pattern)
                assert invalidated_count == 0
            except Exception as e:
                # If it throws an exception, it should be handled gracefully
                assert "pattern" in str(e).lower() or "regex" in str(e).lower()
        
        # Original entries should still be present
        assert self.cache.get("Test query 1") is not None
        assert self.cache.get("Test query 2") is not None
    
    def test_invalidation_statistics_accuracy(self):
        """Test accuracy of invalidation statistics tracking."""
        initial_stats = self.cache.get_invalidation_statistics()
        assert initial_stats['total_invalidations'] == 0
        
        # Perform various invalidation operations
        operations = [
            ("Manual", lambda: self.cache.invalidate("Test 1", "Manual test")),
            ("Pattern", lambda: self.cache.invalidate_by_pattern("non_existent")),
            ("Confidence", lambda: self.cache.invalidate_by_confidence(0.5)),
            ("Bulk", lambda: self.cache.bulk_invalidate(["Test 2"], "Bulk test"))
        ]
        
        # Set some test data first
        self.cache.set("Test 1", "Value 1")
        self.cache.set("Test 2", "Value 2", confidence=0.3)
        
        expected_invalidations = 0
        for op_name, operation in operations:
            result = operation()
            if op_name in ["Manual", "Bulk"] or (op_name == "Confidence" and result > 0):
                expected_invalidations += result if isinstance(result, int) else 1
        
        final_stats = self.cache.get_invalidation_statistics()
        assert final_stats['total_invalidations'] == expected_invalidations
        
        # Verify event tracking accuracy
        events = self.cache.get_invalidation_events()
        total_affected_keys = sum(len(event.affected_keys) for event in events)
        assert total_affected_keys == final_stats['total_invalidations']


# Pytest fixtures for invalidation testing
@pytest.fixture
def invalidating_cache():
    """Provide cache with invalidation capabilities."""
    return MockInvalidatingCache()


@pytest.fixture
def cache_with_rules():
    """Provide cache with pre-configured invalidation rules."""
    cache = MockInvalidatingCache()
    
    # Add common invalidation rules
    rules = [
        InvalidationRule("low_confidence", INVALIDATION_TRIGGERS['CONFIDENCE_BASED'],
                        "confidence < 0.6", "invalidate", 100),
        InvalidationRule("old_entries", INVALIDATION_TRIGGERS['TIME_BASED'],
                        "age > 3600", "invalidate", 200),
        InvalidationRule("unused_entries", INVALIDATION_TRIGGERS['ACCESS_BASED'],
                        "access_count < 1", "defer", 300)
    ]
    
    for rule in rules:
        cache.add_invalidation_rule(rule)
    
    return cache


@pytest.fixture
def biomedical_invalidation_cache():
    """Provide cache pre-loaded with biomedical test data for invalidation testing."""
    cache = MockInvalidatingCache()
    
    # Load biomedical test data
    for category, queries in BIOMEDICAL_QUERIES.items():
        for query_data in queries[:2]:  # Limit for testing
            query = query_data['query']
            response = query_data['response']
            confidence = response.get('confidence', 0.9)
            
            # Add category-based tags
            tags = [category]
            if 'temporal' in query.lower() or '2024' in query:
                tags.append('temporal')
            
            cache.set(query, response, confidence=confidence, tags=tags)
    
    return cache


# Module-level test configuration
pytest_plugins = ['pytest_asyncio']


if __name__ == "__main__":
    # Run specific test class or all tests
    pytest.main([__file__, "-v", "--tb=short"])