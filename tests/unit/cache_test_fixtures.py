"""
Comprehensive test fixtures and utilities for cache testing.

This module provides shared test fixtures, mock objects, and realistic
biomedical test data for use across all cache unit tests.

Classes:
    CacheTestFixtures: Main test fixture provider
    BiomedicalTestDataGenerator: Generator for realistic biomedical queries
    MockCacheBackends: Mock implementations for testing
    CachePerformanceMetrics: Performance measurement utilities

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import time
import random
import string
from typing import Dict, List, Any, Optional, Generator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import json
import pickle
from unittest.mock import Mock, AsyncMock

# Comprehensive biomedical test data
BIOMEDICAL_QUERIES = {
    'metabolism': [
        {
            'query': 'What are the key metabolic pathways in cellular respiration?',
            'response': {
                'pathways': ['glycolysis', 'citric acid cycle', 'electron transport chain'],
                'location': ['cytoplasm', 'mitochondria'],
                'products': ['ATP', 'CO2', 'H2O'],
                'confidence': 0.95
            },
            'cache_preference': 'L1',
            'expected_ttl': 3600
        },
        {
            'query': 'How does insulin regulate glucose metabolism?',
            'response': {
                'mechanism': 'promotes glucose uptake and glycogen synthesis',
                'target_tissues': ['muscle', 'liver', 'adipose'],
                'pathways_affected': ['glycolysis', 'gluconeogenesis', 'glycogenesis'],
                'confidence': 0.92
            },
            'cache_preference': 'L1',
            'expected_ttl': 3600
        },
        {
            'query': 'What metabolites are elevated in diabetes?',
            'response': {
                'elevated_metabolites': ['glucose', 'ketone bodies', 'advanced glycation end products'],
                'biomarkers': ['HbA1c', 'fructosamine', 'glucose'],
                'pathways': ['ketogenesis', 'protein glycation'],
                'confidence': 0.88
            },
            'cache_preference': 'L2',
            'expected_ttl': 7200
        }
    ],
    'clinical_applications': [
        {
            'query': 'How is metabolomics used in drug discovery?',
            'response': {
                'applications': ['biomarker identification', 'toxicity screening', 'efficacy assessment'],
                'methods': ['LC-MS', 'GC-MS', 'NMR spectroscopy'],
                'advantages': ['comprehensive profiling', 'phenotype characterization'],
                'confidence': 0.90
            },
            'cache_preference': 'L1',
            'expected_ttl': 3600
        },
        {
            'query': 'What biomarkers indicate cardiovascular disease risk?',
            'response': {
                'lipid_markers': ['LDL', 'HDL', 'triglycerides', 'apolipoprotein B'],
                'inflammatory_markers': ['CRP', 'IL-6', 'TNF-alpha'],
                'metabolic_markers': ['homocysteine', 'lipoprotein(a)'],
                'confidence': 0.93
            },
            'cache_preference': 'L2',
            'expected_ttl': 7200
        }
    ],
    'disease_metabolomics': [
        {
            'query': 'What metabolic changes occur in cancer?',
            'response': {
                'altered_pathways': ['glycolysis', 'glutaminolysis', 'lipid synthesis'],
                'key_metabolites': ['lactate', 'glutamine', 'fatty acids'],
                'mechanisms': ['Warburg effect', 'metabolic reprogramming'],
                'confidence': 0.87
            },
            'cache_preference': 'L2',
            'expected_ttl': 7200
        },
        {
            'query': 'How does Alzheimer\'s disease affect brain metabolism?',
            'response': {
                'metabolic_dysfunction': ['glucose hypometabolism', 'mitochondrial dysfunction'],
                'affected_neurotransmitters': ['acetylcholine', 'glutamate', 'GABA'],
                'biomarkers': ['tau', 'amyloid-beta', 'neurofilament light'],
                'confidence': 0.84
            },
            'cache_preference': 'L2',
            'expected_ttl': 7200
        }
    ],
    'temporal_queries': [
        {
            'query': 'Latest COVID-19 metabolomics research 2024',
            'response': {
                'recent_findings': ['altered amino acid metabolism', 'lipid dysregulation'],
                'biomarkers': ['kynurenine', 'tryptophan', 'sphingolipids'],
                'status': 'current_research',
                'confidence': 0.78
            },
            'cache_preference': None,  # Temporal queries shouldn't be cached
            'expected_ttl': 300
        },
        {
            'query': 'Current drug metabolomics studies 2024',
            'response': {
                'active_studies': ['personalized medicine', 'pharmacokinetics', 'adverse effects'],
                'methods': ['precision dosing', 'biomarker discovery'],
                'status': 'ongoing',
                'confidence': 0.75
            },
            'cache_preference': None,
            'expected_ttl': 300
        }
    ]
}

PERFORMANCE_TEST_QUERIES = [
    f"Performance test query {i}: metabolomics analysis of biomarker {chr(65 + i % 26)}"
    for i in range(1000)
]

EMERGENCY_RESPONSE_PATTERNS = {
    'general_metabolomics': {
        'patterns': [
            'what is metabolomics',
            'metabolomics definition',
            'clinical metabolomics',
            'metabolomics overview'
        ],
        'response': {
            'definition': 'Metabolomics is the comprehensive analysis of small molecules in biological systems',
            'applications': ['disease diagnosis', 'drug discovery', 'personalized medicine'],
            'methods': ['mass spectrometry', 'NMR spectroscopy'],
            'confidence': 0.9,
            'source': 'emergency_cache'
        }
    },
    'glucose_metabolism': {
        'patterns': [
            'glucose metabolism',
            'glucose pathways',
            'glucose processing',
            'sugar metabolism'
        ],
        'response': {
            'pathways': ['glycolysis', 'gluconeogenesis', 'glycogenolysis'],
            'regulation': ['insulin', 'glucagon', 'epinephrine'],
            'organs': ['liver', 'muscle', 'brain'],
            'confidence': 0.88,
            'source': 'emergency_cache'
        }
    },
    'error_fallback': {
        'patterns': ['*'],
        'response': {
            'message': 'System temporarily unavailable. Please try again later.',
            'suggestions': ['rephrase your question', 'try again in a few minutes'],
            'confidence': 1.0,
            'source': 'emergency_fallback'
        }
    }
}


@dataclass
class CachePerformanceMetrics:
    """Performance metrics for cache testing."""
    operation_type: str
    total_operations: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    success_rate: float
    memory_usage_kb: Optional[float] = None
    
    def meets_performance_targets(self) -> bool:
        """Check if metrics meet performance targets."""
        return (
            self.avg_time_ms < 100 and  # Average < 100ms
            self.p99_time_ms < 1000 and  # P99 < 1s
            self.success_rate > 0.95  # >95% success rate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BiomedicalTestDataGenerator:
    """Generator for realistic biomedical test data."""
    
    def __init__(self):
        self.metabolites = [
            'glucose', 'lactate', 'pyruvate', 'citrate', 'succinate',
            'acetyl-CoA', 'ATP', 'NADH', 'glutamine', 'alanine',
            'cholesterol', 'fatty acids', 'triglycerides', 'phospholipids'
        ]
        
        self.pathways = [
            'glycolysis', 'gluconeogenesis', 'citric acid cycle',
            'electron transport chain', 'fatty acid oxidation',
            'lipid synthesis', 'amino acid metabolism', 'ketogenesis'
        ]
        
        self.diseases = [
            'diabetes', 'cancer', 'cardiovascular disease',
            'Alzheimer\'s disease', 'obesity', 'metabolic syndrome',
            'liver disease', 'kidney disease'
        ]
        
        self.methods = [
            'LC-MS/MS', 'GC-MS', 'NMR spectroscopy', 'UPLC-MS',
            'targeted metabolomics', 'untargeted metabolomics',
            'lipidomics', 'glycomics'
        ]
    
    def generate_query(self, category: str = 'random') -> Dict[str, Any]:
        """Generate a realistic biomedical query."""
        if category == 'metabolism':
            metabolite = random.choice(self.metabolites)
            pathway = random.choice(self.pathways)
            query = f"How does {metabolite} participate in {pathway}?"
            
        elif category == 'disease':
            disease = random.choice(self.diseases)
            metabolite = random.choice(self.metabolites)
            query = f"What role does {metabolite} play in {disease}?"
            
        elif category == 'methods':
            method = random.choice(self.methods)
            metabolite = random.choice(self.metabolites)
            query = f"How is {method} used to measure {metabolite}?"
            
        else:  # random
            templates = [
                "What is the relationship between {metabolite} and {disease}?",
                "How does {pathway} affect {metabolite} levels?",
                "What {method} techniques measure {metabolite}?",
                "How is {metabolite} regulated in {disease}?"
            ]
            template = random.choice(templates)
            query = template.format(
                metabolite=random.choice(self.metabolites),
                pathway=random.choice(self.pathways),
                disease=random.choice(self.diseases),
                method=random.choice(self.methods)
            )
        
        return {
            'query': query,
            'category': category,
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'expected_cache': category != 'temporal'
        }
    
    def generate_batch(self, count: int, category: str = 'random') -> List[Dict[str, Any]]:
        """Generate a batch of queries."""
        return [self.generate_query(category) for _ in range(count)]
    
    def generate_performance_dataset(self, size: int = 1000) -> List[Dict[str, Any]]:
        """Generate large dataset for performance testing."""
        categories = ['metabolism', 'disease', 'methods', 'random']
        dataset = []
        
        for i in range(size):
            category = categories[i % len(categories)]
            query_data = self.generate_query(category)
            query_data['id'] = i
            dataset.append(query_data)
        
        return dataset


class MockCacheBackends:
    """Collection of mock cache backend implementations."""
    
    @staticmethod
    def create_mock_redis():
        """Create mock Redis backend."""
        mock_redis = AsyncMock()
        
        # Mock storage
        storage = {}
        
        async def mock_get(key):
            return storage.get(key)
        
        async def mock_set(key, value, ex=None):
            storage[key] = value
            return True
        
        async def mock_delete(key):
            return storage.pop(key, None) is not None
        
        mock_redis.get = mock_get
        mock_redis.set = mock_set
        mock_redis.delete = mock_delete
        
        return mock_redis, storage
    
    @staticmethod
    def create_mock_diskcache(temp_dir: str):
        """Create mock disk cache backend."""
        mock_cache = Mock()
        
        # Mock storage
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
        
        mock_cache.get = mock_get
        mock_cache.set = mock_set
        mock_cache.delete = mock_delete
        mock_cache.clear = mock_clear
        mock_cache.__len__ = lambda: len(storage)
        
        return mock_cache, storage
    
    @staticmethod
    def create_failing_backend(failure_rate: float = 0.5):
        """Create backend that fails randomly."""
        mock = AsyncMock()
        
        async def maybe_fail(*args, **kwargs):
            if random.random() < failure_rate:
                raise Exception("Simulated backend failure")
            return True
        
        mock.get = maybe_fail
        mock.set = maybe_fail
        mock.delete = maybe_fail
        
        return mock


class CacheTestFixtures:
    """Main test fixture provider for cache testing."""
    
    def __init__(self):
        self.data_generator = BiomedicalTestDataGenerator()
        self.mock_backends = MockCacheBackends()
    
    @pytest.fixture
    def biomedical_queries(self):
        """Provide biomedical test queries."""
        return BIOMEDICAL_QUERIES
    
    @pytest.fixture
    def performance_queries(self):
        """Provide performance test queries."""
        return PERFORMANCE_TEST_QUERIES
    
    @pytest.fixture
    def emergency_patterns(self):
        """Provide emergency response patterns."""
        return EMERGENCY_RESPONSE_PATTERNS
    
    @pytest.fixture
    def test_data_generator(self):
        """Provide test data generator."""
        return self.data_generator
    
    @pytest.fixture
    def mock_redis_backend(self):
        """Provide mock Redis backend."""
        return self.mock_backends.create_mock_redis()
    
    @pytest.fixture
    def mock_disk_backend(self, tmp_path):
        """Provide mock disk cache backend."""
        return self.mock_backends.create_mock_diskcache(str(tmp_path))
    
    @pytest.fixture
    def failing_backend(self):
        """Provide failing backend for error testing."""
        return self.mock_backends.create_failing_backend()
    
    @pytest.fixture
    def cache_performance_measurer(self):
        """Provide performance measurement utilities."""
        return CachePerformanceMeasurer()


class CachePerformanceMeasurer:
    """Utility for measuring cache performance."""
    
    def __init__(self):
        self.measurements = {}
    
    def measure_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Measure a single cache operation."""
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        
        self.measurements[operation_name].append({
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': start_time
        })
        
        return result, duration_ms, success
    
    async def measure_async_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Measure an async cache operation."""
        start_time = time.time()
        
        try:
            result = await operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        
        self.measurements[operation_name].append({
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': start_time
        })
        
        return result, duration_ms, success
    
    def measure_batch_operations(self, operation_name: str, operations: List[Tuple[callable, tuple, dict]]):
        """Measure a batch of operations."""
        results = []
        
        for operation_func, args, kwargs in operations:
            result, duration, success = self.measure_operation(operation_name, operation_func, *args, **kwargs)
            results.append((result, duration, success))
        
        return results
    
    def get_performance_metrics(self, operation_name: str) -> CachePerformanceMetrics:
        """Calculate performance metrics for an operation."""
        if operation_name not in self.measurements:
            raise ValueError(f"No measurements found for operation: {operation_name}")
        
        measurements = self.measurements[operation_name]
        durations = [m['duration_ms'] for m in measurements]
        successes = [m['success'] for m in measurements]
        
        # Calculate statistics
        total_ops = len(measurements)
        avg_time = sum(durations) / len(durations)
        min_time = min(durations)
        max_time = max(durations)
        
        # Percentiles
        sorted_durations = sorted(durations)
        p95_index = int(len(sorted_durations) * 0.95)
        p99_index = int(len(sorted_durations) * 0.99)
        p95_time = sorted_durations[p95_index] if p95_index < len(sorted_durations) else max_time
        p99_time = sorted_durations[p99_index] if p99_index < len(sorted_durations) else max_time
        
        success_rate = sum(successes) / len(successes)
        
        return CachePerformanceMetrics(
            operation_type=operation_name,
            total_operations=total_ops,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            success_rate=success_rate
        )
    
    def clear_measurements(self):
        """Clear all measurements."""
        self.measurements.clear()
    
    def export_measurements(self, filename: str):
        """Export measurements to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.measurements, f, indent=2)


def generate_cache_test_scenarios():
    """Generate comprehensive test scenarios for cache testing."""
    scenarios = {
        'basic_operations': {
            'description': 'Basic cache CRUD operations',
            'queries': [
                'What is glucose metabolism?',
                'How does insulin work?',
                'What are diabetes biomarkers?'
            ],
            'expected_hits': 3,
            'expected_misses': 3
        },
        'lru_eviction': {
            'description': 'LRU eviction policy testing',
            'cache_size': 3,
            'queries': [f'Query {i}' for i in range(6)],
            'access_pattern': [0, 1, 2, 0, 3, 4, 5],
            'expected_evictions': 3
        },
        'confidence_filtering': {
            'description': 'Confidence-based caching decisions',
            'queries': [
                ('High confidence query', 0.9, True),
                ('Medium confidence query', 0.7, True),
                ('Low confidence query', 0.5, False)
            ]
        },
        'concurrent_access': {
            'description': 'Concurrent cache access patterns',
            'thread_count': 5,
            'operations_per_thread': 20,
            'query_pool_size': 10
        },
        'performance_stress': {
            'description': 'Performance under stress conditions',
            'query_count': 1000,
            'cache_size': 100,
            'target_avg_time_ms': 10,
            'target_p99_time_ms': 100
        }
    }
    
    return scenarios


# Global test data for easy import
BIOMEDICAL_TEST_DATA = BIOMEDICAL_QUERIES
EMERGENCY_TEST_PATTERNS = EMERGENCY_RESPONSE_PATTERNS
CACHE_TEST_SCENARIOS = generate_cache_test_scenarios()

__all__ = [
    'CacheTestFixtures',
    'BiomedicalTestDataGenerator', 
    'MockCacheBackends',
    'CachePerformanceMetrics',
    'CachePerformanceMeasurer',
    'BIOMEDICAL_QUERIES',
    'PERFORMANCE_TEST_QUERIES',
    'EMERGENCY_RESPONSE_PATTERNS',
    'BIOMEDICAL_TEST_DATA',
    'EMERGENCY_TEST_PATTERNS',
    'CACHE_TEST_SCENARIOS',
    'generate_cache_test_scenarios'
]