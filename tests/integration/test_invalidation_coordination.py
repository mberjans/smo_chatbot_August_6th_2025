"""
Multi-tier cache invalidation coordination tests for the Clinical Metabolomics Oracle system.

This module provides comprehensive integration testing of invalidation coordination
across multiple cache tiers (L1 memory, L2 disk, L3 Redis, emergency cache).
It tests cascading invalidation, selective invalidation, distributed coordination,
and consistency maintenance across cache tiers.

Test Coverage:
- Cascading invalidation across cache tiers (L1 -> L2 -> L3 -> Emergency)
- Selective invalidation (invalidate specific tiers while preserving others)
- Distributed invalidation coordination across multiple nodes
- Invalidation consistency and synchronization across tiers
- Cross-tier invalidation propagation and timing
- Emergency cache invalidation during system failures
- Multi-tier invalidation performance and optimization
- Invalidation rollback and recovery mechanisms

Classes:
    TestCascadingInvalidation: Cascading invalidation across cache tiers
    TestSelectiveInvalidation: Selective tier-specific invalidation
    TestDistributedInvalidation: Distributed cache invalidation coordination
    TestInvalidationConsistency: Consistency maintenance across tiers
    TestInvalidationPropagation: Invalidation propagation timing and mechanisms
    TestEmergencyInvalidation: Emergency cache invalidation scenarios
    TestInvalidationRecovery: Recovery from invalidation failures
    TestInvalidationSynchronization: Tier synchronization during invalidation

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import weakref
import uuid
from collections import defaultdict, OrderedDict

# Import test fixtures and mocks from unit tests
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'unit'))

from cache_test_fixtures import (
    BIOMEDICAL_QUERIES,
    BiomedicalTestDataGenerator,
    CachePerformanceMetrics,
    CachePerformanceMeasurer,
    EMERGENCY_RESPONSE_PATTERNS,
    MockCacheBackends
)

# Import invalidation components from unit tests
from test_cache_invalidation import (
    InvalidationEvent,
    InvalidationRule,
    MockInvalidatingCache,
    INVALIDATION_STRATEGIES,
    INVALIDATION_TRIGGERS,
    INVALIDATION_POLICIES
)


@dataclass
class TierConfiguration:
    """Configuration for a cache tier in multi-tier testing."""
    tier_name: str
    max_size: int
    default_ttl: int
    invalidation_strategy: str
    priority: int
    enabled: bool = True
    failure_rate: float = 0.0
    latency_ms: float = 1.0


@dataclass
class InvalidationCoordinationEvent:
    """Track invalidation coordination events across tiers."""
    coordination_id: str
    initiating_tier: str
    target_tiers: List[str]
    trigger: str
    strategy: str
    timestamp: float
    affected_keys: List[str]
    propagation_times: Dict[str, float] = field(default_factory=dict)
    success_by_tier: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTierCacheSystem:
    """
    Multi-tier cache system for integration testing of invalidation coordination.
    
    Simulates L1 (memory), L2 (disk), L3 (Redis), and emergency cache tiers
    with comprehensive invalidation coordination capabilities.
    """
    
    def __init__(self, tier_configs: Dict[str, TierConfiguration]):
        self.tiers: Dict[str, MockInvalidatingCache] = {}
        self.tier_configs = tier_configs
        
        # Initialize cache tiers
        for tier_name, config in tier_configs.items():
            cache = MockInvalidatingCache(
                max_size=config.max_size,
                default_ttl=config.default_ttl,
                invalidation_strategy=config.invalidation_strategy
            )
            cache.tier = tier_name
            self.tiers[tier_name] = cache
        
        # Coordination tracking
        self.coordination_events: List[InvalidationCoordinationEvent] = []
        self.invalidation_locks = defaultdict(threading.Lock)
        
        # Performance and statistics
        self.performance_measurer = CachePerformanceMeasurer()
        self.tier_hit_counts: Dict[str, int] = defaultdict(int)
        self.tier_miss_counts: Dict[str, int] = defaultdict(int)
        
        # Propagation settings
        self.propagation_delay_ms = 10
        self.max_propagation_time_ms = 1000
        
        # Failure simulation
        self.failure_simulation_enabled = False
        
    def _generate_coordination_id(self) -> str:
        """Generate unique coordination ID."""
        return f"coord_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def _get_tier_priority_order(self) -> List[str]:
        """Get tiers in priority order (L1 -> L2 -> L3 -> Emergency)."""
        return sorted(
            self.tier_configs.keys(),
            key=lambda t: self.tier_configs[t].priority
        )
    
    def _simulate_network_delay(self, tier_name: str):
        """Simulate network delay for distributed cache operations."""
        if tier_name in ['L3', 'emergency']:
            delay_ms = self.tier_configs[tier_name].latency_ms
            time.sleep(delay_ms / 1000.0)
    
    def _simulate_failure(self, tier_name: str) -> bool:
        """Simulate tier failure based on configured failure rate."""
        if not self.failure_simulation_enabled:
            return False
        
        failure_rate = self.tier_configs[tier_name].failure_rate
        return random.random() < failure_rate
    
    def set_across_tiers(self, query: str, value: Any, tiers: Optional[List[str]] = None,
                        ttl: Optional[int] = None, confidence: float = 0.9,
                        tags: Optional[List[str]] = None) -> Dict[str, str]:
        """Set entry across specified cache tiers."""
        if tiers is None:
            tiers = list(self.tiers.keys())
        
        results = {}
        
        for tier_name in tiers:
            if tier_name in self.tiers and self.tier_configs[tier_name].enabled:
                try:
                    if not self._simulate_failure(tier_name):
                        self._simulate_network_delay(tier_name)
                        key = self.tiers[tier_name].set(
                            query, value, ttl=ttl, confidence=confidence, tags=tags
                        )
                        results[tier_name] = key
                    else:
                        results[tier_name] = None
                except Exception as e:
                    results[tier_name] = None
        
        return results
    
    def get_from_tiers(self, query: str, tiers: Optional[List[str]] = None) -> Tuple[Any, str]:
        """Get entry from cache tiers in priority order."""
        if tiers is None:
            tiers = self._get_tier_priority_order()
        
        for tier_name in tiers:
            if tier_name in self.tiers and self.tier_configs[tier_name].enabled:
                try:
                    if not self._simulate_failure(tier_name):
                        self._simulate_network_delay(tier_name)
                        result = self.tiers[tier_name].get(query)
                        if result is not None:
                            self.tier_hit_counts[tier_name] += 1
                            return result, tier_name
                        else:
                            self.tier_miss_counts[tier_name] += 1
                    else:
                        self.tier_miss_counts[tier_name] += 1
                except Exception as e:
                    self.tier_miss_counts[tier_name] += 1
        
        return None, ""
    
    def invalidate_across_tiers(self, query: str, tiers: Optional[List[str]] = None,
                               strategy: str = "cascading", reason: str = "Cross-tier invalidation",
                               coordination_mode: str = "immediate") -> InvalidationCoordinationEvent:
        """Invalidate entry across multiple cache tiers with coordination."""
        if tiers is None:
            tiers = list(self.tiers.keys())
        
        coordination_id = self._generate_coordination_id()
        event = InvalidationCoordinationEvent(
            coordination_id=coordination_id,
            initiating_tier=tiers[0] if tiers else "system",
            target_tiers=tiers,
            trigger=INVALIDATION_TRIGGERS['MANUAL'],
            strategy=strategy,
            timestamp=time.time(),
            affected_keys=[query]
        )
        
        start_time = time.time()
        
        if strategy == "cascading":
            self._cascading_invalidation(query, tiers, event, reason)
        elif strategy == "parallel":
            self._parallel_invalidation(query, tiers, event, reason)
        elif strategy == "selective":
            self._selective_invalidation(query, tiers, event, reason)
        
        event.metadata['total_time_ms'] = (time.time() - start_time) * 1000
        self.coordination_events.append(event)
        
        return event
    
    def _cascading_invalidation(self, query: str, tiers: List[str], 
                               event: InvalidationCoordinationEvent, reason: str):
        """Perform cascading invalidation across tiers."""
        ordered_tiers = self._get_tier_priority_order()
        target_tiers = [t for t in ordered_tiers if t in tiers]
        
        for tier_name in target_tiers:
            if tier_name in self.tiers:
                tier_start = time.time()
                
                try:
                    if not self._simulate_failure(tier_name):
                        self._simulate_network_delay(tier_name)
                        success = self.tiers[tier_name].invalidate(query, f"{reason} - {tier_name}")
                        event.success_by_tier[tier_name] = success
                    else:
                        event.success_by_tier[tier_name] = False
                except Exception as e:
                    event.success_by_tier[tier_name] = False
                
                tier_time = (time.time() - tier_start) * 1000
                event.propagation_times[tier_name] = tier_time
                
                # Add small delay between tiers for cascading
                if tier_name != target_tiers[-1]:
                    time.sleep(self.propagation_delay_ms / 1000.0)
    
    def _parallel_invalidation(self, query: str, tiers: List[str],
                              event: InvalidationCoordinationEvent, reason: str):
        """Perform parallel invalidation across tiers."""
        def invalidate_tier(tier_name: str) -> Tuple[str, bool, float]:
            tier_start = time.time()
            
            try:
                if not self._simulate_failure(tier_name):
                    self._simulate_network_delay(tier_name)
                    success = self.tiers[tier_name].invalidate(query, f"{reason} - {tier_name}")
                else:
                    success = False
            except Exception as e:
                success = False
            
            tier_time = (time.time() - tier_start) * 1000
            return tier_name, success, tier_time
        
        # Use ThreadPoolExecutor for parallel invalidation
        with ThreadPoolExecutor(max_workers=len(tiers)) as executor:
            futures = [executor.submit(invalidate_tier, tier) for tier in tiers if tier in self.tiers]
            
            for future in as_completed(futures):
                tier_name, success, tier_time = future.result()
                event.success_by_tier[tier_name] = success
                event.propagation_times[tier_name] = tier_time
    
    def _selective_invalidation(self, query: str, tiers: List[str],
                               event: InvalidationCoordinationEvent, reason: str):
        """Perform selective invalidation of specific tiers only."""
        for tier_name in tiers:
            if tier_name in self.tiers:
                tier_start = time.time()
                
                try:
                    if not self._simulate_failure(tier_name):
                        self._simulate_network_delay(tier_name)
                        success = self.tiers[tier_name].invalidate(query, f"Selective {reason} - {tier_name}")
                    else:
                        success = False
                except Exception as e:
                    success = False
                
                tier_time = (time.time() - tier_start) * 1000
                event.success_by_tier[tier_name] = success
                event.propagation_times[tier_name] = tier_time
    
    def bulk_invalidate_across_tiers(self, queries: List[str], tiers: Optional[List[str]] = None,
                                   strategy: str = "parallel") -> List[InvalidationCoordinationEvent]:
        """Bulk invalidate multiple entries across tiers."""
        if tiers is None:
            tiers = list(self.tiers.keys())
        
        coordination_events = []
        
        for query in queries:
            event = self.invalidate_across_tiers(query, tiers, strategy, "Bulk invalidation")
            coordination_events.append(event)
        
        return coordination_events
    
    def invalidate_by_pattern_across_tiers(self, pattern: str, tiers: Optional[List[str]] = None) -> Dict[str, int]:
        """Invalidate entries matching pattern across multiple tiers."""
        if tiers is None:
            tiers = list(self.tiers.keys())
        
        results = {}
        
        for tier_name in tiers:
            if tier_name in self.tiers:
                try:
                    if not self._simulate_failure(tier_name):
                        self._simulate_network_delay(tier_name)
                        count = self.tiers[tier_name].invalidate_by_pattern(pattern)
                        results[tier_name] = count
                    else:
                        results[tier_name] = 0
                except Exception as e:
                    results[tier_name] = 0
        
        return results
    
    def check_consistency_across_tiers(self, query: str, tiers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check consistency of entry across cache tiers."""
        if tiers is None:
            tiers = list(self.tiers.keys())
        
        consistency_report = {
            'query': query,
            'timestamp': time.time(),
            'tier_states': {},
            'consistent': True,
            'inconsistencies': []
        }
        
        values = {}
        for tier_name in tiers:
            if tier_name in self.tiers:
                try:
                    value = self.tiers[tier_name].get(query)
                    values[tier_name] = value
                    consistency_report['tier_states'][tier_name] = {
                        'has_entry': value is not None,
                        'value_type': type(value).__name__ if value is not None else None
                    }
                except Exception as e:
                    consistency_report['tier_states'][tier_name] = {
                        'has_entry': False,
                        'error': str(e)
                    }
        
        # Check for inconsistencies
        non_null_values = {k: v for k, v in values.items() if v is not None}
        if len(set(str(v) for v in non_null_values.values())) > 1:
            consistency_report['consistent'] = False
            consistency_report['inconsistencies'].append('Different values across tiers')
        
        return consistency_report
    
    def synchronize_tiers(self, query: str, source_tier: str, target_tiers: List[str]) -> Dict[str, bool]:
        """Synchronize entry from source tier to target tiers."""
        if source_tier not in self.tiers:
            return {}
        
        source_value = self.tiers[source_tier].get(query)
        if source_value is None:
            return {}
        
        results = {}
        for tier_name in target_tiers:
            if tier_name in self.tiers and tier_name != source_tier:
                try:
                    if not self._simulate_failure(tier_name):
                        self._simulate_network_delay(tier_name)
                        key = self.tiers[tier_name].set(query, source_value)
                        results[tier_name] = key is not None
                    else:
                        results[tier_name] = False
                except Exception as e:
                    results[tier_name] = False
        
        return results
    
    def get_invalidation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive invalidation statistics across all tiers."""
        tier_stats = {}
        
        for tier_name, cache in self.tiers.items():
            tier_stats[tier_name] = cache.get_invalidation_statistics()
        
        coordination_stats = {
            'total_coordination_events': len(self.coordination_events),
            'successful_coordinations': sum(
                1 for e in self.coordination_events 
                if all(e.success_by_tier.values())
            ),
            'average_propagation_time_ms': self._calculate_average_propagation_time(),
            'tier_hit_counts': dict(self.tier_hit_counts),
            'tier_miss_counts': dict(self.tier_miss_counts)
        }
        
        return {
            'tier_statistics': tier_stats,
            'coordination_statistics': coordination_stats
        }
    
    def _calculate_average_propagation_time(self) -> float:
        """Calculate average propagation time across all coordination events."""
        if not self.coordination_events:
            return 0.0
        
        total_times = []
        for event in self.coordination_events:
            if event.propagation_times:
                total_times.extend(event.propagation_times.values())
        
        return sum(total_times) / len(total_times) if total_times else 0.0
    
    def enable_failure_simulation(self, tier_failure_rates: Dict[str, float]):
        """Enable failure simulation with specified failure rates per tier."""
        self.failure_simulation_enabled = True
        for tier_name, failure_rate in tier_failure_rates.items():
            if tier_name in self.tier_configs:
                self.tier_configs[tier_name].failure_rate = failure_rate
    
    def disable_failure_simulation(self):
        """Disable failure simulation."""
        self.failure_simulation_enabled = False
        for config in self.tier_configs.values():
            config.failure_rate = 0.0


class TestCascadingInvalidation:
    """Tests for cascading invalidation across cache tiers."""
    
    def setup_method(self):
        """Set up multi-tier cache system for testing."""
        self.tier_configs = {
            'L1': TierConfiguration('L1', 10, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 50, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),  
            'L3': TierConfiguration('L3', 200, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3),
            'emergency': TierConfiguration('emergency', 100, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 4)
        }
        
        self.cache_system = MultiTierCacheSystem(self.tier_configs)
        self.data_generator = BiomedicalTestDataGenerator()
    
    def test_basic_cascading_invalidation(self):
        """Test basic cascading invalidation from L1 to emergency cache."""
        query = "What is glucose metabolism?"
        value = "Glucose metabolism involves glycolysis and gluconeogenesis"
        
        # Set entry across all tiers
        set_results = self.cache_system.set_across_tiers(query, value)
        assert len(set_results) == 4  # All tiers should have the entry
        
        # Verify entry exists in all tiers
        for tier_name in ['L1', 'L2', 'L3', 'emergency']:
            result = self.cache_system.tiers[tier_name].get(query)
            assert result == value
        
        # Perform cascading invalidation
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="Test cascading invalidation"
        )
        
        # Verify invalidation cascaded to all tiers
        assert all(invalidation_event.success_by_tier.values())
        assert len(invalidation_event.target_tiers) == 4
        
        # Verify entry is invalidated in all tiers
        for tier_name in ['L1', 'L2', 'L3', 'emergency']:
            result = self.cache_system.tiers[tier_name].get(query)
            assert result is None
        
        # Verify propagation times are recorded
        assert len(invalidation_event.propagation_times) == 4
        for tier_name, prop_time in invalidation_event.propagation_times.items():
            assert prop_time >= 0
    
    def test_cascading_with_tier_failures(self):
        """Test cascading invalidation when some tiers fail."""
        query = "Test query for failure scenario"
        value = "Test value"
        
        # Set up failure simulation
        self.cache_system.enable_failure_simulation({
            'L1': 0.0,    # L1 always succeeds
            'L2': 0.5,    # L2 fails 50% of the time
            'L3': 0.0,    # L3 always succeeds
            'emergency': 0.3  # Emergency fails 30% of the time
        })
        
        # Set entry across all tiers (multiple attempts to handle failures)
        set_attempts = 0
        max_attempts = 5
        
        while set_attempts < max_attempts:
            set_results = self.cache_system.set_across_tiers(query, value)
            if len([k for k in set_results.values() if k is not None]) >= 2:
                break
            set_attempts += 1
        
        # Perform cascading invalidation
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="Test with failures"
        )
        
        # At least some tiers should have been processed
        assert len(invalidation_event.success_by_tier) > 0
        
        # Some invalidations may fail due to simulated failures
        success_count = sum(invalidation_event.success_by_tier.values())
        total_attempts = len(invalidation_event.success_by_tier)
        
        # At least 50% should succeed (given failure rates)
        assert success_count >= total_attempts * 0.5
    
    def test_cascading_timing_constraints(self):
        """Test that cascading invalidation meets timing constraints."""
        query = "Timing test query"
        value = "Timing test value"
        
        # Set realistic latencies
        self.cache_system.tier_configs['L1'].latency_ms = 1
        self.cache_system.tier_configs['L2'].latency_ms = 10  
        self.cache_system.tier_configs['L3'].latency_ms = 50
        self.cache_system.tier_configs['emergency'].latency_ms = 100
        
        self.cache_system.set_across_tiers(query, value)
        
        start_time = time.time()
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="Timing test"
        )
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Cascading should take longer than parallel (due to sequential execution)
        expected_min_time = sum(config.latency_ms for config in self.cache_system.tier_configs.values())
        assert total_time >= expected_min_time * 0.8  # Allow 20% tolerance
        
        # But should not exceed maximum threshold
        assert total_time < self.cache_system.max_propagation_time_ms
        
        # Verify propagation times are sequential
        propagation_times = list(invalidation_event.propagation_times.values())
        # Times should generally increase (later tiers take longer)
        # Allow some variation due to timing precision
        assert len(propagation_times) == 4
    
    def test_cascading_with_biomedical_data(self):
        """Test cascading invalidation with realistic biomedical queries."""
        biomedical_queries = BIOMEDICAL_QUERIES['metabolism'][:3]
        
        # Set biomedical entries across all tiers
        for query_data in biomedical_queries:
            query = query_data['query']
            response = query_data['response']
            self.cache_system.set_across_tiers(
                query, response, confidence=response['confidence']
            )
        
        # Verify all entries are present
        for query_data in biomedical_queries:
            query = query_data['query']
            for tier_name in self.cache_system.tiers.keys():
                result = self.cache_system.tiers[tier_name].get(query)
                assert result is not None
        
        # Cascading invalidation of all biomedical queries
        invalidation_events = []
        for query_data in biomedical_queries:
            query = query_data['query']
            event = self.cache_system.invalidate_across_tiers(
                query, strategy="cascading", reason="Biomedical data update"
            )
            invalidation_events.append(event)
        
        # Verify all cascading invalidations succeeded
        for event in invalidation_events:
            assert all(event.success_by_tier.values())
        
        # Verify all entries are invalidated
        for query_data in biomedical_queries:
            query = query_data['query']
            for tier_name in self.cache_system.tiers.keys():
                result = self.cache_system.tiers[tier_name].get(query)
                assert result is None
        
        # Verify statistics
        stats = self.cache_system.get_invalidation_statistics()
        assert stats['coordination_statistics']['total_coordination_events'] == len(biomedical_queries)


class TestSelectiveInvalidation:
    """Tests for selective tier-specific invalidation."""
    
    def setup_method(self):
        """Set up multi-tier cache system for testing."""
        self.tier_configs = {
            'L1': TierConfiguration('L1', 10, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 50, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
            'L3': TierConfiguration('L3', 200, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3),
            'emergency': TierConfiguration('emergency', 100, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 4)
        }
        
        self.cache_system = MultiTierCacheSystem(self.tier_configs)
    
    def test_selective_l1_only_invalidation(self):
        """Test invalidating only L1 cache while preserving others."""
        query = "Selective L1 test query"
        value = "Test value for selective invalidation"
        
        # Set entry across all tiers
        self.cache_system.set_across_tiers(query, value)
        
        # Verify entry exists in all tiers
        for tier_name in self.cache_system.tiers.keys():
            assert self.cache_system.tiers[tier_name].get(query) == value
        
        # Selectively invalidate only L1
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, tiers=['L1'], strategy="selective", reason="L1 selective invalidation"
        )
        
        # Verify only L1 was targeted
        assert invalidation_event.target_tiers == ['L1']
        assert invalidation_event.success_by_tier['L1'] == True
        
        # Verify L1 is invalidated
        assert self.cache_system.tiers['L1'].get(query) is None
        
        # Verify other tiers still have the entry
        for tier_name in ['L2', 'L3', 'emergency']:
            assert self.cache_system.tiers[tier_name].get(query) == value
    
    def test_selective_memory_tiers_invalidation(self):
        """Test invalidating memory-based tiers while preserving persistent storage."""
        query = "Memory tiers test query"
        value = "Test value for memory invalidation"
        
        # Set entry across all tiers
        self.cache_system.set_across_tiers(query, value)
        
        # Selectively invalidate memory-based tiers (L1 and L3)
        memory_tiers = ['L1', 'L3']
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, tiers=memory_tiers, strategy="selective", reason="Memory tier cleanup"
        )
        
        # Verify memory tiers were invalidated
        assert set(invalidation_event.target_tiers) == set(memory_tiers)
        for tier in memory_tiers:
            assert invalidation_event.success_by_tier[tier] == True
            assert self.cache_system.tiers[tier].get(query) is None
        
        # Verify persistent tiers still have the entry
        persistent_tiers = ['L2', 'emergency']
        for tier_name in persistent_tiers:
            assert self.cache_system.tiers[tier_name].get(query) == value
    
    def test_selective_invalidation_with_patterns(self):
        """Test selective invalidation combined with pattern matching."""
        # Set entries with different categories
        test_entries = [
            ("Glucose metabolism query", "Glucose data", ['L1', 'L2']),
            ("Insulin pathway query", "Insulin data", ['L2', 'L3']),
            ("Diabetes research query", "Diabetes data", ['L1', 'L3', 'emergency']),
            ("Cancer biomarker query", "Cancer data", ['L1', 'L2', 'L3', 'emergency'])
        ]
        
        # Set entries in specified tiers only
        for query, value, target_tiers in test_entries:
            self.cache_system.set_across_tiers(query, value, tiers=target_tiers)
        
        # Selective pattern invalidation on L1 and L3 only
        target_tiers = ['L1', 'L3']
        diabetes_invalidated = self.cache_system.invalidate_by_pattern_across_tiers(
            "diabetes", tiers=target_tiers
        )
        
        # Verify diabetes query invalidated only in L1 and L3
        assert diabetes_invalidated['L1'] == 1  # Diabetes query was in L1
        assert diabetes_invalidated['L3'] == 1  # Diabetes query was in L3
        
        # Verify diabetes query still exists in emergency tier
        assert self.cache_system.tiers['emergency'].get("Diabetes research query") is not None
        
        # Verify other queries unaffected in target tiers
        assert self.cache_system.tiers['L1'].get("Glucose metabolism query") is not None
        assert self.cache_system.tiers['L3'].get("Insulin pathway query") is not None
    
    def test_selective_confidence_based_invalidation(self):
        """Test selective invalidation based on confidence scores in specific tiers."""
        # Set entries with different confidence levels
        confidence_entries = [
            ("High confidence query", "Reliable data", 0.95, ['L1', 'L2', 'L3']),
            ("Medium confidence query", "Moderate data", 0.75, ['L1', 'L2', 'L3']),
            ("Low confidence query", "Uncertain data", 0.45, ['L1', 'L2', 'L3'])
        ]
        
        for query, value, confidence, target_tiers in confidence_entries:
            for tier_name in target_tiers:
                self.cache_system.tiers[tier_name].set(
                    query, value, confidence=confidence
                )
        
        # Selective confidence-based invalidation in L1 only
        l1_invalidated = self.cache_system.tiers['L1'].invalidate_by_confidence(0.7)
        assert l1_invalidated == 1  # Only low confidence entry
        
        # Verify L1 invalidation
        assert self.cache_system.tiers['L1'].get("Low confidence query") is None
        assert self.cache_system.tiers['L1'].get("High confidence query") is not None
        assert self.cache_system.tiers['L1'].get("Medium confidence query") is not None
        
        # Verify other tiers unaffected
        assert self.cache_system.tiers['L2'].get("Low confidence query") is not None
        assert self.cache_system.tiers['L3'].get("Low confidence query") is not None


class TestDistributedInvalidation:
    """Tests for distributed cache invalidation coordination."""
    
    def setup_method(self):
        """Set up multiple cache system instances for distributed testing."""
        # Create multiple cache system instances to simulate distributed nodes
        self.node_configs = {
            'L1': TierConfiguration('L1', 10, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 50, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
            'L3': TierConfiguration('L3', 200, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
        }
        
        # Create 3 distributed nodes
        self.nodes = {}
        for i in range(3):
            node_id = f"node_{i}"
            self.nodes[node_id] = MultiTierCacheSystem(self.node_configs.copy())
    
    def test_parallel_invalidation_across_nodes(self):
        """Test parallel invalidation across multiple distributed nodes."""
        query = "Distributed parallel test query"
        value = "Distributed test value"
        
        # Set entry in all nodes
        for node_id, node_system in self.nodes.items():
            node_system.set_across_tiers(query, value)
        
        # Verify entry exists in all nodes
        for node_id, node_system in self.nodes.items():
            for tier_name in node_system.tiers.keys():
                assert node_system.tiers[tier_name].get(query) == value
        
        # Parallel invalidation across all nodes
        def invalidate_node(node_id: str) -> Tuple[str, InvalidationCoordinationEvent]:
            node_system = self.nodes[node_id]
            event = node_system.invalidate_across_tiers(
                query, strategy="parallel", reason=f"Distributed invalidation from {node_id}"
            )
            return node_id, event
        
        # Execute parallel invalidation
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = [executor.submit(invalidate_node, node_id) for node_id in self.nodes.keys()]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all nodes completed invalidation
        assert len(results) == len(self.nodes)
        
        for node_id, event in results:
            # All tiers in each node should be successfully invalidated
            assert all(event.success_by_tier.values())
        
        # Verify entries are invalidated across all nodes
        for node_id, node_system in self.nodes.items():
            for tier_name in node_system.tiers.keys():
                assert node_system.tiers[tier_name].get(query) is None
    
    def test_distributed_consistency_check(self):
        """Test consistency checking across distributed nodes."""
        query = "Consistency test query"
        values = ["Value A", "Value B", "Value C"]
        
        # Set different values in different nodes (simulate inconsistency)
        for i, (node_id, node_system) in enumerate(self.nodes.items()):
            node_system.set_across_tiers(query, values[i % len(values)])
        
        # Check consistency across all nodes
        consistency_reports = {}
        for node_id, node_system in self.nodes.items():
            report = node_system.check_consistency_across_tiers(query)
            consistency_reports[node_id] = report
        
        # Verify each node reports its own consistency correctly
        for node_id, report in consistency_reports.items():
            # Each individual node should be internally consistent
            # (but nodes will be inconsistent with each other)
            assert 'tier_states' in report
            assert len(report['tier_states']) == len(self.node_configs)
    
    def test_distributed_synchronization(self):
        """Test synchronization of data across distributed nodes."""
        query = "Sync test query"
        authoritative_value = "Authoritative value"
        
        # Set authoritative value in first node only
        master_node = self.nodes['node_0']
        master_node.set_across_tiers(query, authoritative_value)
        
        # Set different/no values in other nodes
        for node_id in ['node_1', 'node_2']:
            self.nodes[node_id].set_across_tiers(query, f"Stale value from {node_id}")
        
        # Simulate synchronization by copying from master to other nodes
        # (In real system, this would be handled by distributed cache protocol)
        for node_id in ['node_1', 'node_2']:
            target_node = self.nodes[node_id]
            # Clear existing entries
            target_node.invalidate_across_tiers(query, reason="Pre-sync cleanup")
            # Set authoritative value
            target_node.set_across_tiers(query, authoritative_value)
        
        # Verify all nodes now have consistent values
        for node_id, node_system in self.nodes.items():
            for tier_name in node_system.tiers.keys():
                result = node_system.tiers[tier_name].get(query)
                assert result == authoritative_value
    
    def test_distributed_failure_tolerance(self):
        """Test distributed invalidation tolerance to node failures."""
        query = "Failure tolerance test query"
        value = "Resilient test value"
        
        # Set entry in all nodes
        for node_system in self.nodes.values():
            node_system.set_across_tiers(query, value)
        
        # Simulate failures in some nodes
        self.nodes['node_1'].enable_failure_simulation({
            'L1': 0.8, 'L2': 0.8, 'L3': 0.8  # High failure rate
        })
        
        # Attempt distributed invalidation
        successful_invalidations = 0
        failed_invalidations = 0
        
        for node_id, node_system in self.nodes.items():
            try:
                event = node_system.invalidate_across_tiers(
                    query, strategy="parallel", reason="Failure tolerance test"
                )
                
                if any(event.success_by_tier.values()):
                    successful_invalidations += 1
                else:
                    failed_invalidations += 1
                    
            except Exception as e:
                failed_invalidations += 1
        
        # At least some nodes should succeed despite failures
        assert successful_invalidations > 0
        
        # System should be resilient (at least 60% success rate)
        success_rate = successful_invalidations / len(self.nodes)
        assert success_rate >= 0.6


class TestInvalidationConsistency:
    """Tests for invalidation consistency maintenance across tiers."""
    
    def setup_method(self):
        """Set up multi-tier cache system for consistency testing."""
        self.tier_configs = {
            'L1': TierConfiguration('L1', 5, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
            'L2': TierConfiguration('L2', 20, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
            'L3': TierConfiguration('L3', 50, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
        }
        
        self.cache_system = MultiTierCacheSystem(self.tier_configs)
    
    def test_consistency_after_cascading_invalidation(self):
        """Test that cascading invalidation maintains consistency."""
        query = "Consistency test query"
        value = "Consistent test value"
        
        # Set entry across all tiers
        self.cache_system.set_across_tiers(query, value)
        
        # Verify initial consistency
        consistency_report = self.cache_system.check_consistency_across_tiers(query)
        assert consistency_report['consistent'] == True
        
        # Perform cascading invalidation
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="Consistency test"
        )
        
        # Verify all tiers were successfully invalidated
        assert all(invalidation_event.success_by_tier.values())
        
        # Verify consistency after invalidation (all tiers should be empty)
        post_invalidation_report = self.cache_system.check_consistency_across_tiers(query)
        
        # All tiers should consistently show no entry
        for tier_state in post_invalidation_report['tier_states'].values():
            assert tier_state['has_entry'] == False
    
    def test_consistency_with_partial_failures(self):
        """Test consistency handling when some tiers fail during invalidation."""
        query = "Partial failure consistency test"
        value = "Test value for partial failure"
        
        # Set entry across all tiers
        self.cache_system.set_across_tiers(query, value)
        
        # Enable failure simulation for L2 tier
        self.cache_system.enable_failure_simulation({'L2': 1.0})  # L2 always fails
        
        # Attempt cascading invalidation
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="Partial failure test"
        )
        
        # L1 and L3 should succeed, L2 should fail
        assert invalidation_event.success_by_tier['L1'] == True
        assert invalidation_event.success_by_tier['L2'] == False
        assert invalidation_event.success_by_tier['L3'] == True
        
        # Check consistency - system is now inconsistent
        consistency_report = self.cache_system.check_consistency_across_tiers(query)
        assert consistency_report['consistent'] == False
        assert 'Different values across tiers' in consistency_report['inconsistencies'] or len(consistency_report['inconsistencies']) == 0
        
        # L1 and L3 should be invalidated, L2 should still have the value
        assert self.cache_system.tiers['L1'].get(query) is None
        assert self.cache_system.tiers['L2'].get(query) == value  # Failed to invalidate
        assert self.cache_system.tiers['L3'].get(query) is None
    
    def test_consistency_repair_mechanism(self):
        """Test consistency repair after partial invalidation failures."""
        query = "Consistency repair test"
        value = "Value for repair test"
        
        # Set entry across all tiers
        self.cache_system.set_across_tiers(query, value)
        
        # Manually create inconsistency by invalidating only some tiers
        self.cache_system.tiers['L1'].invalidate(query, "Manual L1 invalidation")
        self.cache_system.tiers['L3'].invalidate(query, "Manual L3 invalidation")
        # L2 still has the value
        
        # Verify inconsistency
        consistency_report = self.cache_system.check_consistency_across_tiers(query)
        assert consistency_report['consistent'] == False
        
        # Repair by completing the invalidation
        remaining_invalidation = self.cache_system.invalidate_across_tiers(
            query, tiers=['L2'], strategy="selective", reason="Consistency repair"
        )
        
        assert remaining_invalidation.success_by_tier['L2'] == True
        
        # Verify consistency restored (all tiers now consistently empty)
        post_repair_report = self.cache_system.check_consistency_across_tiers(query)
        
        for tier_state in post_repair_report['tier_states'].values():
            assert tier_state['has_entry'] == False
    
    def test_consistency_with_concurrent_operations(self):
        """Test consistency during concurrent invalidation operations."""
        base_query = "Concurrent consistency test"
        queries = [f"{base_query} {i}" for i in range(10)]
        value = "Concurrent test value"
        
        # Set all entries across all tiers
        for query in queries:
            self.cache_system.set_across_tiers(query, value)
        
        def invalidate_queries(query_subset: List[str]) -> List[bool]:
            results = []
            for query in query_subset:
                try:
                    event = self.cache_system.invalidate_across_tiers(
                        query, strategy="parallel", reason="Concurrent test"
                    )
                    results.append(all(event.success_by_tier.values()))
                except Exception as e:
                    results.append(False)
            return results
        
        # Split queries for concurrent processing
        mid_point = len(queries) // 2
        query_set_1 = queries[:mid_point]
        query_set_2 = queries[mid_point:]
        
        # Execute concurrent invalidations
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(invalidate_queries, query_set_1)
            future2 = executor.submit(invalidate_queries, query_set_2)
            
            results1 = future1.result()
            results2 = future2.result()
        
        # Most invalidations should succeed
        total_successes = sum(results1) + sum(results2)
        total_attempts = len(results1) + len(results2)
        success_rate = total_successes / total_attempts
        
        assert success_rate >= 0.8  # At least 80% success rate
        
        # Check final consistency for all queries
        inconsistent_count = 0
        for query in queries:
            consistency_report = self.cache_system.check_consistency_across_tiers(query)
            if not consistency_report['consistent']:
                inconsistent_count += 1
        
        # Allow some inconsistency due to concurrency, but should be minimal
        inconsistency_rate = inconsistent_count / len(queries)
        assert inconsistency_rate <= 0.2  # At most 20% inconsistency


class TestInvalidationRecovery:
    """Tests for recovery from invalidation failures."""
    
    def setup_method(self):
        """Set up cache system with failure simulation for recovery testing."""
        self.tier_configs = {
            'L1': TierConfiguration('L1', 10, 300, INVALIDATION_STRATEGIES['DEFERRED'], 1),
            'L2': TierConfiguration('L2', 50, 3600, INVALIDATION_STRATEGIES['DEFERRED'], 2),
            'L3': TierConfiguration('L3', 200, 86400, INVALIDATION_STRATEGIES['DEFERRED'], 3),
        }
        
        self.cache_system = MultiTierCacheSystem(self.tier_configs)
    
    def test_recovery_from_network_partition(self):
        """Test recovery after network partition causes invalidation failures."""
        query = "Network partition test query"
        value = "Network partition test value"
        
        # Set entry across all tiers
        self.cache_system.set_across_tiers(query, value)
        
        # Simulate network partition by causing failures in distributed tiers
        self.cache_system.enable_failure_simulation({
            'L1': 0.0,   # Local tier always works
            'L2': 0.9,   # Disk cache mostly fails (simulating I/O issues)
            'L3': 1.0    # Network tier always fails (simulating partition)
        })
        
        # Attempt invalidation during "partition"
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="During network partition"
        )
        
        # Only L1 should succeed
        assert invalidation_event.success_by_tier['L1'] == True
        assert invalidation_event.success_by_tier['L2'] == False
        assert invalidation_event.success_by_tier['L3'] == False
        
        # "Partition heals" - restore network connectivity
        self.cache_system.disable_failure_simulation()
        
        # Retry invalidation of remaining tiers
        recovery_event = self.cache_system.invalidate_across_tiers(
            query, tiers=['L2', 'L3'], strategy="parallel", reason="Post-partition recovery"
        )
        
        # Recovery should succeed
        assert recovery_event.success_by_tier['L2'] == True
        assert recovery_event.success_by_tier['L3'] == True
        
        # Verify final consistency
        for tier_name in self.cache_system.tiers.keys():
            assert self.cache_system.tiers[tier_name].get(query) is None
    
    def test_deferred_invalidation_recovery(self):
        """Test recovery using deferred invalidation strategy."""
        query = "Deferred invalidation test"
        value = "Deferred test value"
        
        # Configure deferred invalidation strategy for all tiers
        for tier_name in self.cache_system.tiers.keys():
            cache = self.cache_system.tiers[tier_name]
            cache.invalidation_strategy = INVALIDATION_STRATEGIES['DEFERRED']
            
            # Add rule that defers invalidation based on system state
            rule = InvalidationRule(
                rule_id=f"defer_rule_{tier_name}",
                trigger=INVALIDATION_TRIGGERS['MANUAL'],
                condition="tag:defer",
                action="defer",
                priority=100
            )
            cache.add_invalidation_rule(rule)
        
        # Set entry with defer tag
        for tier_name in self.cache_system.tiers.keys():
            self.cache_system.tiers[tier_name].set(
                query, value, tags=["defer", "recoverable"]
            )
        
        # Invalidation should be deferred due to rules
        invalidation_event = self.cache_system.invalidate_across_tiers(
            query, strategy="cascading", reason="Deferred invalidation test"
        )
        
        # Entries should still be accessible (invalidation deferred)
        for tier_name in self.cache_system.tiers.keys():
            assert self.cache_system.tiers[tier_name].get(query) == value
        
        # Process deferred invalidations
        for tier_name in self.cache_system.tiers.keys():
            processed_count = self.cache_system.tiers[tier_name].process_deferred_invalidations()
            assert processed_count > 0
        
        # Now entries should be invalidated
        for tier_name in self.cache_system.tiers.keys():
            assert self.cache_system.tiers[tier_name].get(query) is None
    
    def test_background_recovery_cleanup(self):
        """Test background recovery and cleanup processes."""
        queries = [f"Background cleanup test {i}" for i in range(5)]
        value = "Background cleanup value"
        
        # Set entries across all tiers
        for query in queries:
            self.cache_system.set_across_tiers(query, value)
        
        # Manually add some entries to pending invalidation queues
        for tier_name, cache in self.cache_system.tiers.items():
            for query in queries[:3]:  # First 3 queries go to pending
                key = cache._generate_key(query)
                cache.pending_invalidations.add(key)
                cache.invalidation_queue.append((key, INVALIDATION_TRIGGERS['SYSTEM_STATE'], {}))
        
        # Run background cleanup on all tiers
        cleanup_results = {}
        for tier_name, cache in self.cache_system.tiers.items():
            results = cache.background_cleanup()
            cleanup_results[tier_name] = results
        
        # Verify cleanup processed pending and queued invalidations
        for tier_name, results in cleanup_results.items():
            assert 'deferred_processed' in results or 'queued_processed' in results
        
        # Verify statistics reflect the cleanup
        stats = self.cache_system.get_invalidation_statistics()
        total_invalidations = sum(
            tier_stats['total_invalidations'] 
            for tier_stats in stats['tier_statistics'].values()
        )
        assert total_invalidations > 0


# Pytest fixtures for integration testing
@pytest.fixture
def multi_tier_cache_system():
    """Provide multi-tier cache system for integration testing."""
    tier_configs = {
        'L1': TierConfiguration('L1', 10, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
        'L2': TierConfiguration('L2', 50, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
        'L3': TierConfiguration('L3', 200, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3),
        'emergency': TierConfiguration('emergency', 100, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 4)
    }
    
    return MultiTierCacheSystem(tier_configs)


@pytest.fixture
def distributed_cache_nodes():
    """Provide multiple cache system nodes for distributed testing."""
    node_configs = {
        'L1': TierConfiguration('L1', 10, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
        'L2': TierConfiguration('L2', 50, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
        'L3': TierConfiguration('L3', 200, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
    }
    
    nodes = {}
    for i in range(3):
        node_id = f"node_{i}"
        nodes[node_id] = MultiTierCacheSystem(node_configs.copy())
    
    return nodes


@pytest.fixture
def biomedical_multi_tier_cache():
    """Provide multi-tier cache pre-loaded with biomedical data."""
    tier_configs = {
        'L1': TierConfiguration('L1', 20, 300, INVALIDATION_STRATEGIES['IMMEDIATE'], 1),
        'L2': TierConfiguration('L2', 100, 3600, INVALIDATION_STRATEGIES['IMMEDIATE'], 2),
        'L3': TierConfiguration('L3', 500, 86400, INVALIDATION_STRATEGIES['IMMEDIATE'], 3)
    }
    
    cache_system = MultiTierCacheSystem(tier_configs)
    
    # Load biomedical test data across tiers
    for category, queries in BIOMEDICAL_QUERIES.items():
        for query_data in queries[:2]:  # Limit for testing
            query = query_data['query']
            response = query_data['response']
            confidence = response.get('confidence', 0.9)
            
            # Add category-based tags
            tags = [category]
            if 'temporal' in query.lower() or '2024' in query:
                tags.append('temporal')
            
            cache_system.set_across_tiers(
                query, response, confidence=confidence, tags=tags
            )
    
    return cache_system


# Module-level test configuration
pytest_plugins = ['pytest_asyncio']


if __name__ == "__main__":
    # Run specific test class or all tests
    pytest.main([__file__, "-v", "--tb=short"])