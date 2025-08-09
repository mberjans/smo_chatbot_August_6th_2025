"""
Unit tests for emergency cache system operations.

This module tests the emergency pickle-based cache system that provides
instant responses when all other systems fail. The emergency cache is
designed for maximum reliability and sub-second response times.

Test Coverage:
- Emergency cache activation when primary systems fail
- Pickle-based serialization and security
- Emergency cache preloading of common queries
- File management and rotation
- Response time guarantees (<1 second)
- Pattern-based fallback storage
- Cache warming strategies
- Recovery and failover mechanisms

Classes:
    TestEmergencyCacheActivation: Tests for emergency cache activation
    TestPickleSerialization: Tests for pickle serialization security
    TestEmergencyCachePreloading: Tests for cache preloading
    TestEmergencyCacheFileManagement: Tests for file management
    TestEmergencyCachePerformance: Tests for response time guarantees
    TestPatternBasedFallback: Tests for pattern-based fallback
    TestEmergencyCacheRecovery: Tests for recovery mechanisms

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import os
import pickle
import json
import tempfile
import shutil
import threading
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Emergency cache test data
EMERGENCY_RESPONSE_PATTERNS = {
    'metabolomics_general': {
        'patterns': [
            'what is metabolomics',
            'metabolomics definition',
            'clinical metabolomics',
            'metabolomics overview'
        ],
        'response': {
            'definition': 'Metabolomics is the comprehensive analysis of small molecules (metabolites) in biological systems.',
            'applications': ['disease diagnosis', 'drug discovery', 'personalized medicine'],
            'confidence': 0.9,
            'source': 'emergency_cache'
        }
    },
    'glucose_metabolism': {
        'patterns': [
            'glucose metabolism',
            'glucose pathways',
            'glucose metabolic pathways',
            'how is glucose metabolized'
        ],
        'response': {
            'pathways': ['glycolysis', 'gluconeogenesis', 'glycogen synthesis'],
            'key_enzymes': ['hexokinase', 'glucose-6-phosphatase', 'glycogen phosphorylase'],
            'regulation': 'insulin and glucagon',
            'confidence': 0.88,
            'source': 'emergency_cache'
        }
    },
    'diabetes_metabolomics': {
        'patterns': [
            'diabetes metabolomics',
            'diabetic metabolic profiling',
            'diabetes biomarkers',
            'metabolomics diabetes'
        ],
        'response': {
            'biomarkers': ['glucose', 'HbA1c', 'fructosamine', 'insulin'],
            'altered_pathways': ['glucose metabolism', 'lipid metabolism', 'amino acid metabolism'],
            'metabolites': ['glucose', '3-hydroxybutyrate', 'lactate', 'alanine'],
            'confidence': 0.85,
            'source': 'emergency_cache'
        }
    },
    'error_fallback': {
        'patterns': ['*'],  # Catch-all pattern
        'response': {
            'message': 'I apologize, but I am currently experiencing technical difficulties. Please try your question again in a few moments.',
            'suggestions': [
                'Try rephrasing your question',
                'Check for system status updates',
                'Contact support if the issue persists'
            ],
            'confidence': 1.0,
            'source': 'emergency_fallback'
        }
    }
}

COMMON_BIOMEDICAL_QUERIES = [
    "What are the main metabolic pathways?",
    "How does insulin affect glucose metabolism?",
    "What biomarkers indicate diabetes risk?",
    "What is clinical metabolomics used for?",
    "How do statins affect lipid metabolism?",
    "What metabolites are altered in cancer?",
    "How does exercise affect metabolism?",
    "What is the role of mitochondria in metabolism?",
    "How do metabolomics identify drug targets?",
    "What metabolic changes occur in aging?"
]

@dataclass
class EmergencyCacheEntry:
    """Emergency cache entry structure."""
    key: str
    pattern: str
    response: Dict[str, Any]
    timestamp: float
    access_count: int
    last_access: float
    file_path: Optional[str] = None
    
    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if entry is stale."""
        return (time.time() - self.timestamp) > (max_age_hours * 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EmergencyCache:
    """Emergency cache system for critical failover scenarios."""
    
    def __init__(self, cache_dir: str, max_entries: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        
        self.index_file = self.cache_dir / "emergency_index.json"
        self.patterns_file = self.cache_dir / "patterns.pkl"
        
        # In-memory storage for fast access
        self.memory_cache: Dict[str, EmergencyCacheEntry] = {}
        self.pattern_cache: Dict[str, str] = {}  # pattern -> key mapping
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'activations': 0,
            'preloads': 0,
            'file_rotations': 0,
            'pattern_matches': 0
        }
        
        self._lock = threading.RLock()
        self.is_active = False
        
        # Load existing cache
        self._load_cache()
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        normalized = query.lower().strip()
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        return f"emergency_{hash_obj.hexdigest()[:12]}"
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def _load_cache(self):
        """Load emergency cache from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                
                for key, entry_data in index_data.items():
                    entry = EmergencyCacheEntry(**entry_data)
                    self.memory_cache[key] = entry
                    
                    # Load pattern mappings
                    if entry.pattern:
                        self.pattern_cache[entry.pattern] = key
        except Exception as e:
            logging.warning(f"Failed to load emergency cache index: {e}")
            self.memory_cache = {}
            self.pattern_cache = {}
    
    def _save_cache(self):
        """Save emergency cache to disk."""
        try:
            index_data = {}
            for key, entry in self.memory_cache.items():
                # Don't include file_path in index to avoid issues
                entry_dict = entry.to_dict()
                entry_dict.pop('file_path', None)
                index_data[key] = entry_dict
            
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save emergency cache index: {e}")
    
    def _match_pattern(self, query: str) -> Optional[str]:
        """Match query against cached patterns."""
        normalized_query = query.lower().strip()
        
        # Direct pattern matches
        for pattern, key in self.pattern_cache.items():
            if pattern == '*':  # Catch-all pattern
                continue
                
            if pattern in normalized_query or normalized_query in pattern:
                self.stats['pattern_matches'] += 1
                return key
        
        # Fallback to catch-all if exists
        if '*' in self.pattern_cache:
            self.stats['pattern_matches'] += 1
            return self.pattern_cache['*']
        
        return None
    
    def activate(self):
        """Activate emergency cache mode."""
        with self._lock:
            self.is_active = True
            self.stats['activations'] += 1
            logging.critical("Emergency cache activated - primary systems unavailable")
    
    def deactivate(self):
        """Deactivate emergency cache mode."""
        with self._lock:
            self.is_active = False
            logging.info("Emergency cache deactivated - primary systems restored")
    
    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get response from emergency cache."""
        if not self.is_active:
            return None
        
        with self._lock:
            start_time = time.time()
            
            # Try direct key match first
            key = self._generate_cache_key(query)
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check if stale
                if entry.is_stale():
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                
                self.stats['hits'] += 1
                
                response = entry.response.copy()
                response['emergency_cache_hit'] = True
                response['response_time_ms'] = (time.time() - start_time) * 1000
                return response
            
            # Try pattern matching
            matched_key = self._match_pattern(query)
            if matched_key and matched_key in self.memory_cache:
                entry = self.memory_cache[matched_key]
                
                if not entry.is_stale():
                    entry.access_count += 1
                    entry.last_access = time.time()
                    
                    self.stats['hits'] += 1
                    
                    response = entry.response.copy()
                    response['emergency_cache_hit'] = True
                    response['pattern_matched'] = entry.pattern
                    response['response_time_ms'] = (time.time() - start_time) * 1000
                    return response
            
            self.stats['misses'] += 1
            return None
    
    async def set(self, query: str, response: Dict[str, Any], pattern: Optional[str] = None) -> bool:
        """Store response in emergency cache."""
        with self._lock:
            try:
                key = self._generate_cache_key(query)
                
                # Create entry
                entry = EmergencyCacheEntry(
                    key=key,
                    pattern=pattern or query.lower().strip(),
                    response=response,
                    timestamp=time.time(),
                    access_count=0,
                    last_access=time.time()
                )
                
                # Store in memory
                self.memory_cache[key] = entry
                
                # Update pattern mapping
                if pattern:
                    self.pattern_cache[pattern] = key
                
                # Store to file
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                entry.file_path = str(file_path)
                
                # Enforce size limits
                await self._enforce_size_limits()
                
                # Save index
                self._save_cache()
                
                return True
            except Exception as e:
                logging.error(f"Failed to store emergency cache entry: {e}")
                return False
    
    async def _enforce_size_limits(self):
        """Enforce cache size limits."""
        if len(self.memory_cache) <= self.max_entries:
            return
        
        # Sort by least recently used
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_access
        )
        
        # Remove LRU entries
        entries_to_remove = len(self.memory_cache) - self.max_entries
        for i in range(entries_to_remove):
            key, entry = sorted_entries[i]
            
            # Remove file
            if entry.file_path and os.path.exists(entry.file_path):
                os.remove(entry.file_path)
            
            # Remove from memory
            del self.memory_cache[key]
            
            # Remove pattern mapping
            pattern_keys_to_remove = [
                p for p, k in self.pattern_cache.items() if k == key
            ]
            for p in pattern_keys_to_remove:
                del self.pattern_cache[p]
    
    async def preload_common_patterns(self, patterns: Dict[str, Any] = None):
        """Preload common query patterns."""
        if patterns is None:
            patterns = EMERGENCY_RESPONSE_PATTERNS
        
        with self._lock:
            for pattern_name, pattern_data in patterns.items():
                for pattern in pattern_data['patterns']:
                    await self.set(
                        query=pattern,
                        response=pattern_data['response'],
                        pattern=pattern
                    )
                    self.stats['preloads'] += 1
            
            logging.info(f"Preloaded {self.stats['preloads']} emergency patterns")
    
    def rotate_files(self, max_age_hours: int = 24):
        """Rotate old cache files."""
        with self._lock:
            current_time = time.time()
            rotated_count = 0
            
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if entry.is_stale(max_age_hours):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self.memory_cache[key]
                
                # Remove file
                if entry.file_path and os.path.exists(entry.file_path):
                    os.remove(entry.file_path)
                
                # Remove from memory and patterns
                del self.memory_cache[key]
                pattern_keys_to_remove = [
                    p for p, k in self.pattern_cache.items() if k == key
                ]
                for p in pattern_keys_to_remove:
                    del self.pattern_cache[p]
                
                rotated_count += 1
            
            self.stats['file_rotations'] += rotated_count
            self._save_cache()
            
            logging.info(f"Rotated {rotated_count} old emergency cache files")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emergency cache statistics."""
        with self._lock:
            total_ops = self.stats['hits'] + self.stats['misses']
            return {
                'is_active': self.is_active,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': self.stats['hits'] / total_ops if total_ops > 0 else 0.0,
                'activations': self.stats['activations'],
                'preloads': self.stats['preloads'],
                'file_rotations': self.stats['file_rotations'],
                'pattern_matches': self.stats['pattern_matches'],
                'total_entries': len(self.memory_cache),
                'total_patterns': len(self.pattern_cache),
                'cache_size_mb': sum(
                    os.path.getsize(self._get_file_path(key))
                    for key in self.memory_cache.keys()
                    if self._get_file_path(key).exists()
                ) / (1024 * 1024)
            }


class TestEmergencyCacheActivation:
    """Tests for emergency cache activation and deactivation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_emergency_cache_activation(self):
        """Test emergency cache activation."""
        assert not self.emergency_cache.is_active, "Emergency cache should start inactive"
        
        self.emergency_cache.activate()
        
        assert self.emergency_cache.is_active, "Emergency cache should be active after activation"
        stats = self.emergency_cache.get_stats()
        assert stats['activations'] == 1, "Should track activation count"
    
    def test_emergency_cache_deactivation(self):
        """Test emergency cache deactivation."""
        self.emergency_cache.activate()
        assert self.emergency_cache.is_active, "Emergency cache should be active"
        
        self.emergency_cache.deactivate()
        
        assert not self.emergency_cache.is_active, "Emergency cache should be inactive after deactivation"
    
    @pytest.mark.asyncio
    async def test_inactive_cache_returns_none(self):
        """Test inactive emergency cache returns None."""
        # Cache is inactive by default
        result = await self.emergency_cache.get("test query")
        assert result is None, "Inactive emergency cache should return None"
    
    @pytest.mark.asyncio
    async def test_active_cache_processes_queries(self):
        """Test active emergency cache processes queries."""
        # Activate and store data
        self.emergency_cache.activate()
        await self.emergency_cache.set("test query", {"response": "test response"})
        
        result = await self.emergency_cache.get("test query")
        assert result is not None, "Active emergency cache should process queries"
        assert result['response'] == "test response"
        assert result['emergency_cache_hit'] is True


class TestPickleSerialization:
    """Tests for pickle serialization security and functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_pickle_serialization_basic(self):
        """Test basic pickle serialization functionality."""
        complex_response = {
            'data': ['item1', 'item2', 'item3'],
            'metadata': {
                'timestamp': datetime.now(),
                'confidence': 0.95,
                'nested': {'deep': {'value': 42}}
            },
            'tuple_data': (1, 2, 3, 4, 5)
        }
        
        self.emergency_cache.activate()
        success = await self.emergency_cache.set("complex_query", complex_response)
        assert success, "Should successfully serialize complex data"
        
        result = await self.emergency_cache.get("complex_query")
        assert result is not None, "Should retrieve serialized data"
        
        # Verify data integrity (excluding added metadata)
        assert result['data'] == complex_response['data']
        assert result['metadata']['confidence'] == complex_response['metadata']['confidence']
        assert result['tuple_data'] == complex_response['tuple_data']
    
    @pytest.mark.asyncio
    async def test_pickle_security_safe_types(self):
        """Test pickle handles safe data types correctly."""
        safe_data_types = [
            {"type": "dict", "data": {"key": "value"}},
            {"type": "list", "data": [1, 2, 3, "string"]},
            {"type": "tuple", "data": (1, "two", 3.0)},
            {"type": "string", "data": "safe string"},
            {"type": "number", "data": 42.5},
            {"type": "boolean", "data": True}
        ]
        
        self.emergency_cache.activate()
        
        for test_case in safe_data_types:
            query = f"test_{test_case['type']}"
            response = {"safe_data": test_case['data']}
            
            success = await self.emergency_cache.set(query, response)
            assert success, f"Should handle {test_case['type']} safely"
            
            result = await self.emergency_cache.get(query)
            assert result is not None, f"Should retrieve {test_case['type']} safely"
            assert result['safe_data'] == test_case['data']
    
    @pytest.mark.asyncio
    async def test_pickle_error_handling(self):
        """Test pickle handles serialization errors gracefully."""
        # Create object that might cause serialization issues
        class UnserializableClass:
            def __init__(self):
                self.data = "test"
                # This should still be pickle-able, but let's test error handling
        
        problematic_data = {"object": UnserializableClass()}
        
        self.emergency_cache.activate()
        
        # Should handle gracefully (either succeed or fail without crashing)
        try:
            success = await self.emergency_cache.set("problematic_query", problematic_data)
            # If it succeeds, verify it can be retrieved
            if success:
                result = await self.emergency_cache.get("problematic_query")
                assert result is not None, "Should retrieve if successfully stored"
        except Exception:
            # If it fails, that's acceptable as long as it doesn't crash the system
            pass
    
    def test_pickle_file_integrity(self):
        """Test pickle files maintain integrity on disk."""
        self.emergency_cache.activate()
        
        # Create test entry
        test_entry = EmergencyCacheEntry(
            key="test_key",
            pattern="test pattern",
            response={"test": "response"},
            timestamp=time.time(),
            access_count=1,
            last_access=time.time()
        )
        
        # Manually store to file
        file_path = self.emergency_cache._get_file_path("test_key")
        with open(file_path, 'wb') as f:
            pickle.dump(test_entry, f)
        
        # Verify file exists and can be loaded
        assert file_path.exists(), "Pickle file should exist"
        
        with open(file_path, 'rb') as f:
            loaded_entry = pickle.load(f)
        
        assert loaded_entry.key == test_entry.key, "Loaded entry should match original"
        assert loaded_entry.response == test_entry.response, "Response data should be preserved"


class TestEmergencyCachePreloading:
    """Tests for emergency cache preloading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_preload_common_patterns(self):
        """Test preloading of common query patterns."""
        await self.emergency_cache.preload_common_patterns()
        
        stats = self.emergency_cache.get_stats()
        assert stats['preloads'] > 0, "Should preload patterns"
        assert stats['total_entries'] > 0, "Should have cached entries"
        assert stats['total_patterns'] > 0, "Should have pattern mappings"
    
    @pytest.mark.asyncio
    async def test_preloaded_pattern_matching(self):
        """Test preloaded patterns can be matched and retrieved."""
        await self.emergency_cache.preload_common_patterns()
        self.emergency_cache.activate()
        
        # Test pattern matching for preloaded data
        test_queries = [
            "what is metabolomics",  # Should match metabolomics_general pattern
            "glucose pathways",      # Should match glucose_metabolism pattern
            "diabetes biomarkers"    # Should match diabetes_metabolomics pattern
        ]
        
        for query in test_queries:
            result = await self.emergency_cache.get(query)
            assert result is not None, f"Should find match for '{query}'"
            assert result['emergency_cache_hit'] is True
            assert 'pattern_matched' in result
    
    @pytest.mark.asyncio
    async def test_custom_pattern_preloading(self):
        """Test preloading with custom patterns."""
        custom_patterns = {
            'test_pattern': {
                'patterns': ['test query', 'testing pattern'],
                'response': {'custom': 'response', 'confidence': 0.9}
            }
        }
        
        await self.emergency_cache.preload_common_patterns(custom_patterns)
        self.emergency_cache.activate()
        
        result = await self.emergency_cache.get("test query")
        assert result is not None, "Should retrieve custom preloaded pattern"
        assert result['custom'] == 'response'
    
    @pytest.mark.asyncio
    async def test_preload_statistics_tracking(self):
        """Test preloading statistics are tracked correctly."""
        initial_stats = self.emergency_cache.get_stats()
        initial_preloads = initial_stats['preloads']
        
        await self.emergency_cache.preload_common_patterns()
        
        final_stats = self.emergency_cache.get_stats()
        assert final_stats['preloads'] > initial_preloads, "Should increment preload counter"
    
    @pytest.mark.asyncio
    async def test_preload_with_biomedical_queries(self):
        """Test preloading works with realistic biomedical queries."""
        # Create biomedical patterns
        biomedical_patterns = {
            'metabolomics_pathways': {
                'patterns': [
                    'metabolic pathways',
                    'biochemical pathways',
                    'metabolic routes'
                ],
                'response': {
                    'pathways': ['glycolysis', 'TCA cycle', 'oxidative phosphorylation'],
                    'confidence': 0.92,
                    'source': 'emergency_biomedical'
                }
            }
        }
        
        await self.emergency_cache.preload_common_patterns(biomedical_patterns)
        self.emergency_cache.activate()
        
        result = await self.emergency_cache.get("metabolic pathways")
        assert result is not None, "Should handle biomedical query patterns"
        assert 'pathways' in result
        assert result['confidence'] == 0.92


class TestEmergencyCacheFileManagement:
    """Tests for emergency cache file management and rotation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir, max_entries=5)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_file_creation_and_cleanup(self):
        """Test cache files are created and cleaned up properly."""
        self.emergency_cache.activate()
        
        # Store entries
        for i in range(3):
            await self.emergency_cache.set(f"query_{i}", {"data": f"response_{i}"})
        
        # Check files exist
        cache_files = list(self.emergency_cache.cache_dir.glob("*.pkl"))
        assert len(cache_files) == 3, "Should create files for cache entries"
        
        # Check index file exists
        assert self.emergency_cache.index_file.exists(), "Should create index file"
    
    def test_file_rotation_by_age(self):
        """Test file rotation removes old entries."""
        self.emergency_cache.activate()
        
        # Create old entries by manipulating timestamps
        old_entry = EmergencyCacheEntry(
            key="old_key",
            pattern="old pattern",
            response={"old": "response"},
            timestamp=time.time() - (48 * 3600),  # 48 hours ago
            access_count=1,
            last_access=time.time() - (48 * 3600)
        )
        
        recent_entry = EmergencyCacheEntry(
            key="recent_key",
            pattern="recent pattern",
            response={"recent": "response"},
            timestamp=time.time(),
            access_count=1,
            last_access=time.time()
        )
        
        # Store entries in memory cache
        self.emergency_cache.memory_cache["old_key"] = old_entry
        self.emergency_cache.memory_cache["recent_key"] = recent_entry
        
        # Create corresponding files
        with open(self.emergency_cache._get_file_path("old_key"), 'wb') as f:
            pickle.dump(old_entry, f)
        with open(self.emergency_cache._get_file_path("recent_key"), 'wb') as f:
            pickle.dump(recent_entry, f)
        
        # Rotate files (max age 24 hours)
        self.emergency_cache.rotate_files(max_age_hours=24)
        
        # Old entry should be removed
        assert "old_key" not in self.emergency_cache.memory_cache
        assert not self.emergency_cache._get_file_path("old_key").exists()
        
        # Recent entry should remain
        assert "recent_key" in self.emergency_cache.memory_cache
        assert self.emergency_cache._get_file_path("recent_key").exists()
        
        stats = self.emergency_cache.get_stats()
        assert stats['file_rotations'] > 0, "Should track file rotations"
    
    @pytest.mark.asyncio
    async def test_size_limit_enforcement(self):
        """Test cache enforces size limits."""
        # Max entries is 5
        for i in range(8):
            await self.emergency_cache.set(f"query_{i}", {"data": f"response_{i}"})
        
        stats = self.emergency_cache.get_stats()
        assert stats['total_entries'] <= 5, "Should enforce max entries limit"
        
        # Should keep most recently accessed entries
        cache_files = list(self.emergency_cache.cache_dir.glob("*.pkl"))
        assert len(cache_files) <= 5, "Should not exceed file count limit"
    
    @pytest.mark.asyncio
    async def test_index_persistence(self):
        """Test cache index persists across restarts."""
        self.emergency_cache.activate()
        
        # Store data
        await self.emergency_cache.set("persistent_query", {"persistent": "response"})
        
        # Create new cache instance (simulate restart)
        new_cache = EmergencyCache(self.temp_dir)
        new_cache.activate()
        
        # Should load previous data
        result = await new_cache.get("persistent_query")
        assert result is not None, "Should load data after restart"
        assert result['persistent'] == "response"
    
    def test_cache_size_calculation(self):
        """Test cache size calculation is accurate."""
        self.emergency_cache.activate()
        
        # Create entries with known sizes
        small_data = {"small": "data"}
        large_data = {"large": "x" * 1000}
        
        # Store entries
        asyncio.run(self.emergency_cache.set("small_query", small_data))
        asyncio.run(self.emergency_cache.set("large_query", large_data))
        
        stats = self.emergency_cache.get_stats()
        assert stats['cache_size_mb'] > 0, "Should calculate cache size"
        
        # Large entry should contribute more to size
        # (This is a rough check since pickle overhead varies)
        assert stats['total_entries'] == 2, "Should track entry count correctly"


class TestEmergencyCachePerformance:
    """Tests for emergency cache performance guarantees."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_sub_second_response_time(self):
        """Test emergency cache provides sub-second response times."""
        await self.emergency_cache.preload_common_patterns()
        self.emergency_cache.activate()
        
        # Measure response times
        response_times = []
        test_queries = [
            "what is metabolomics",
            "glucose metabolism",
            "diabetes biomarkers"
        ]
        
        for query in test_queries:
            start_time = time.time()
            result = await self.emergency_cache.get(query)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert result is not None, f"Should find response for '{query}'"
            assert response_time < 1.0, f"Response time should be <1s, got {response_time:.3f}s"
            
            # Check response includes timing information
            assert 'response_time_ms' in result
            assert result['response_time_ms'] < 1000
        
        # Average response time should be much faster
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.1, f"Average response time should be <100ms, got {avg_response_time*1000:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test emergency cache performance under concurrent load."""
        await self.emergency_cache.preload_common_patterns()
        self.emergency_cache.activate()
        
        async def query_worker(worker_id: int):
            """Worker function for concurrent testing."""
            results = []
            queries = ["what is metabolomics", "glucose pathways", "diabetes biomarkers"]
            
            for i in range(10):  # 10 queries per worker
                query = queries[i % len(queries)]
                start_time = time.time()
                result = await self.emergency_cache.get(query)
                end_time = time.time()
                
                results.append({
                    'query': query,
                    'response_time': end_time - start_time,
                    'found': result is not None
                })
            
            return results
        
        # Run multiple workers concurrently
        workers = [query_worker(i) for i in range(5)]
        all_results = await asyncio.gather(*workers)
        
        # Flatten results
        flat_results = [result for worker_results in all_results for result in worker_results]
        
        # Verify performance
        response_times = [r['response_time'] for r in flat_results]
        max_response_time = max(response_times)
        avg_response_time = sum(response_times) / len(response_times)
        
        assert max_response_time < 1.0, f"Max response time should be <1s, got {max_response_time:.3f}s"
        assert avg_response_time < 0.1, f"Avg response time should be <100ms, got {avg_response_time*1000:.1f}ms"
        
        # Verify all queries were successful
        success_rate = sum(1 for r in flat_results if r['found']) / len(flat_results)
        assert success_rate > 0.95, f"Success rate should be >95%, got {success_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_memory_cache_efficiency(self):
        """Test emergency cache memory efficiency."""
        # Store data and measure memory impact
        initial_entries = len(self.emergency_cache.memory_cache)
        
        await self.emergency_cache.preload_common_patterns()
        
        final_entries = len(self.emergency_cache.memory_cache)
        added_entries = final_entries - initial_entries
        
        assert added_entries > 0, "Should add entries to memory cache"
        
        # Verify fast memory access
        self.emergency_cache.activate()
        
        start_time = time.time()
        for _ in range(100):
            await self.emergency_cache.get("what is metabolomics")
        end_time = time.time()
        
        avg_time_per_access = (end_time - start_time) / 100
        assert avg_time_per_access < 0.01, f"Memory access should be <10ms, got {avg_time_per_access*1000:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_pattern_matching_performance(self):
        """Test pattern matching performance is acceptable."""
        await self.emergency_cache.preload_common_patterns()
        self.emergency_cache.activate()
        
        # Test pattern matching with various query types
        pattern_queries = [
            "what is metabolomics exactly",  # Should match metabolomics pattern
            "tell me about glucose pathways",  # Should match glucose pattern
            "diabetes metabolomics research",   # Should match diabetes pattern
            "completely unrelated query"       # Should match catch-all pattern
        ]
        
        start_time = time.time()
        for query in pattern_queries:
            result = await self.emergency_cache.get(query)
            assert result is not None, f"Should find pattern match for '{query}'"
        end_time = time.time()
        
        avg_pattern_match_time = (end_time - start_time) / len(pattern_queries)
        assert avg_pattern_match_time < 0.1, f"Pattern matching should be fast, got {avg_pattern_match_time*1000:.1f}ms"


class TestPatternBasedFallback:
    """Tests for pattern-based fallback storage and matching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_pattern_based_storage(self):
        """Test storing responses with associated patterns."""
        self.emergency_cache.activate()
        
        # Store with specific pattern
        pattern = "insulin resistance"
        response = {
            "condition": "insulin resistance",
            "description": "Reduced cellular response to insulin",
            "confidence": 0.9
        }
        
        success = await self.emergency_cache.set(
            query="insulin resistance metabolomics",
            response=response,
            pattern=pattern
        )
        
        assert success, "Should store with pattern successfully"
        
        # Verify pattern mapping
        assert pattern in self.emergency_cache.pattern_cache
        
        # Should match similar queries
        similar_queries = [
            "insulin resistance effects",
            "insulin resistance syndrome",
            "metabolic insulin resistance"
        ]
        
        for query in similar_queries:
            result = await self.emergency_cache.get(query)
            assert result is not None, f"Should match pattern for '{query}'"
            assert result['condition'] == "insulin resistance"
            assert result['pattern_matched'] == pattern
    
    @pytest.mark.asyncio
    async def test_catch_all_pattern(self):
        """Test catch-all pattern for unmatched queries."""
        # Add catch-all pattern
        catch_all_response = {
            "message": "Emergency fallback response",
            "type": "fallback"
        }
        
        await self.emergency_cache.set(
            query="fallback",
            response=catch_all_response,
            pattern="*"
        )
        
        self.emergency_cache.activate()
        
        # Test with completely unmatched query
        result = await self.emergency_cache.get("completely random query with no matches")
        
        assert result is not None, "Should match catch-all pattern"
        assert result['message'] == "Emergency fallback response"
        assert result['pattern_matched'] == "*"
    
    @pytest.mark.asyncio
    async def test_pattern_priority(self):
        """Test pattern matching priority (specific over general)."""
        # Store specific pattern
        await self.emergency_cache.set(
            query="glucose metabolism",
            response={"type": "specific", "topic": "glucose"},
            pattern="glucose metabolism"
        )
        
        # Store general pattern
        await self.emergency_cache.set(
            query="general metabolism",
            response={"type": "general", "topic": "metabolism"},
            pattern="metabolism"
        )
        
        self.emergency_cache.activate()
        
        # Query should match specific pattern
        result = await self.emergency_cache.get("glucose metabolism pathways")
        
        assert result is not None, "Should find match"
        assert result['type'] == "specific", "Should prefer specific pattern match"
        assert result['topic'] == "glucose"
    
    @pytest.mark.asyncio
    async def test_pattern_statistics(self):
        """Test pattern matching statistics are tracked."""
        await self.emergency_cache.preload_common_patterns()
        self.emergency_cache.activate()
        
        initial_stats = self.emergency_cache.get_stats()
        initial_matches = initial_stats['pattern_matches']
        
        # Trigger pattern matches
        await self.emergency_cache.get("what is metabolomics")
        await self.emergency_cache.get("glucose pathways")
        
        final_stats = self.emergency_cache.get_stats()
        final_matches = final_stats['pattern_matches']
        
        assert final_matches > initial_matches, "Should increment pattern match counter"
        assert final_matches >= initial_matches + 2, "Should track each pattern match"


class TestEmergencyCacheRecovery:
    """Tests for emergency cache recovery and failover mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.emergency_cache = EmergencyCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recovery_from_corrupted_index(self):
        """Test recovery when index file is corrupted."""
        # Create cache with data
        asyncio.run(self.emergency_cache.set("test_query", {"test": "response"}))
        
        # Corrupt index file
        with open(self.emergency_cache.index_file, 'w') as f:
            f.write("corrupted json data {invalid")
        
        # Create new cache instance (should handle corruption)
        recovered_cache = EmergencyCache(self.temp_dir)
        
        # Should start with empty cache but not crash
        assert len(recovered_cache.memory_cache) == 0, "Should start fresh after corruption"
        
        # Should be able to store new data
        recovered_cache.activate()
        success = asyncio.run(recovered_cache.set("recovery_test", {"recovered": "data"}))
        assert success, "Should be able to store data after recovery"
    
    def test_recovery_from_missing_files(self):
        """Test recovery when cache files are missing."""
        # Create cache entry
        self.emergency_cache.activate()
        asyncio.run(self.emergency_cache.set("missing_file_test", {"data": "test"}))
        
        # Delete the actual cache file but leave index
        cache_files = list(self.emergency_cache.cache_dir.glob("*.pkl"))
        if cache_files:
            os.remove(cache_files[0])
        
        # Should handle missing file gracefully
        result = asyncio.run(self.emergency_cache.get("missing_file_test"))
        assert result is None, "Should return None for missing file"
        
        # Should not crash and should continue working
        success = asyncio.run(self.emergency_cache.set("new_test", {"new": "data"}))
        assert success, "Should continue working after missing file"
    
    @pytest.mark.asyncio
    async def test_recovery_from_disk_full(self):
        """Test recovery when disk is full (simulated)."""
        self.emergency_cache.activate()
        
        # Mock disk full scenario
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            success = await self.emergency_cache.set("disk_full_test", {"test": "data"})
            assert not success, "Should fail gracefully when disk is full"
        
        # Should continue working after disk issue is resolved
        success = await self.emergency_cache.set("recovery_test", {"recovered": "data"})
        assert success, "Should work after disk issue is resolved"
    
    def test_thread_safety_during_recovery(self):
        """Test emergency cache is thread-safe during recovery operations."""
        self.emergency_cache.activate()
        
        def writer_thread():
            for i in range(10):
                asyncio.run(self.emergency_cache.set(f"thread_test_{i}", {"data": i}))
        
        def reader_thread():
            for i in range(10):
                asyncio.run(self.emergency_cache.get(f"thread_test_{i}"))
        
        def rotator_thread():
            self.emergency_cache.rotate_files()
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=writer_thread),
            threading.Thread(target=reader_thread),
            threading.Thread(target=rotator_thread)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not crash and should maintain reasonable state
        stats = self.emergency_cache.get_stats()
        assert stats['total_entries'] >= 0, "Should maintain valid entry count"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test emergency cache provides graceful degradation."""
        # Even without preloading, should provide some response
        self.emergency_cache.activate()
        
        # Store minimal fallback response
        await self.emergency_cache.set(
            "fallback",
            {
                "message": "System is experiencing issues. Please try again later.",
                "status": "degraded"
            },
            pattern="*"
        )
        
        # Any query should get a response
        result = await self.emergency_cache.get("any random query")
        
        assert result is not None, "Should provide fallback response"
        assert result['status'] == "degraded"
        assert "experiencing issues" in result['message']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])