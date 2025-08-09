"""
Test Configuration and Utilities for Multi-Level Fallback Testing
=================================================================

This module provides configuration, utilities, and shared fixtures for testing
the multi-level fallback system (LightRAG → Perplexity → Cache).

Features:
- Test configuration management
- Mock backend factories
- Performance benchmarking utilities
- Failure simulation helpers
- Analytics validation tools

Author: Claude Code (Anthropic)
Task: CMO-LIGHTRAG-014-T01-TEST Support Module
Created: August 9, 2025
"""

import pytest
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

# Import test dependencies
try:
    from lightrag_integration.query_router import RoutingDecision, RoutingPrediction, ConfidenceMetrics
    from lightrag_integration.research_categorizer import CategoryPrediction  
    from lightrag_integration.cost_persistence import ResearchCategory
    from lightrag_integration.comprehensive_fallback_system import FallbackLevel, FailureType
except ImportError:
    # Fallback definitions for testing
    class RoutingDecision(Enum):
        LIGHTRAG = "lightrag"
        PERPLEXITY = "perplexity"
        EITHER = "either"
        HYBRID = "hybrid"
    
    class FallbackLevel(Enum):
        FULL_LLM_WITH_CONFIDENCE = 1
        SIMPLIFIED_LLM = 2
        KEYWORD_BASED_ONLY = 3
        EMERGENCY_CACHE = 4
        DEFAULT_ROUTING = 5
    
    class FailureType(Enum):
        API_TIMEOUT = "api_timeout"
        API_ERROR = "api_error"
        SERVICE_UNAVAILABLE = "service_unavailable"


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@dataclass
class FallbackTestConfig:
    """Configuration for fallback testing scenarios."""
    
    # Performance thresholds
    max_primary_response_time_ms: float = 1000
    max_fallback_response_time_ms: float = 2000
    max_emergency_response_time_ms: float = 500
    
    # Failure simulation parameters
    simulate_timeouts: bool = True
    simulate_rate_limits: bool = True
    simulate_connection_errors: bool = True
    timeout_duration_seconds: float = 2.0
    
    # Load testing parameters
    concurrent_queries_count: int = 20
    rapid_queries_count: int = 50
    max_concurrent_test_time_seconds: float = 5.0
    
    # Confidence thresholds
    min_primary_confidence: float = 0.7
    min_fallback_confidence: float = 0.3
    min_emergency_confidence: float = 0.1
    
    # Success rate requirements
    required_success_rate: float = 0.95
    required_fallback_success_rate: float = 0.9
    
    # Monitoring and analytics
    enable_performance_monitoring: bool = True
    enable_failure_analytics: bool = True
    log_level: str = "INFO"


# ============================================================================
# MOCK BACKEND FACTORIES
# ============================================================================

class MockBackendFactory:
    """Factory for creating mock backends with configurable behavior."""
    
    @staticmethod
    def create_lightrag_backend(failure_rate=0.0, response_time_ms=800, **kwargs):
        """Create a mock LightRAG backend."""
        backend = Mock()
        backend.name = "lightrag"
        backend.backend_type = "lightrag"
        backend.failure_rate = failure_rate
        backend.base_response_time_ms = response_time_ms
        backend.failure_count = 0
        backend.call_count = 0
        backend.is_healthy = Mock(return_value=True)
        
        def mock_query(query_text, context=None):
            backend.call_count += 1
            
            # Simulate failures based on failure rate or manual trigger
            if backend.failure_count > 0:
                backend.failure_count -= 1
                if "timeout" in query_text.lower():
                    time.sleep(0.1)  # Brief delay to simulate timeout attempt
                    raise TimeoutError("LightRAG timeout simulation")
                elif "error" in query_text.lower():
                    raise ConnectionError("LightRAG connection error simulation")
                elif "slow" in query_text.lower():
                    time.sleep(0.5)  # Simulate slow response
            
            # Simulate response time variation
            import random
            actual_time = backend.base_response_time_ms + random.randint(-100, 200)
            time.sleep(actual_time / 10000)  # Brief delay to simulate processing
            
            # Create successful response
            return MockResponseFactory.create_lightrag_response(
                query_text, 
                confidence=0.8 + random.random() * 0.15,
                response_time_ms=actual_time
            )
        
        backend.query = Mock(side_effect=mock_query)
        return backend
    
    @staticmethod
    def create_perplexity_backend(failure_rate=0.0, response_time_ms=1200, **kwargs):
        """Create a mock Perplexity backend."""
        backend = Mock()
        backend.name = "perplexity"
        backend.backend_type = "perplexity"
        backend.failure_rate = failure_rate
        backend.base_response_time_ms = response_time_ms
        backend.failure_count = 0
        backend.call_count = 0
        backend.is_healthy = Mock(return_value=True)
        
        def mock_query(query_text, context=None):
            backend.call_count += 1
            
            # Simulate failures
            if backend.failure_count > 0:
                backend.failure_count -= 1
                if "timeout" in query_text.lower():
                    time.sleep(0.2)  # Brief delay
                    raise TimeoutError("Perplexity timeout simulation")
                elif "rate_limit" in query_text.lower():
                    raise Exception("Rate limit exceeded - Perplexity API")
                elif "error" in query_text.lower():
                    raise Exception("Perplexity API error simulation")
            
            # Simulate response time
            import random
            actual_time = backend.base_response_time_ms + random.randint(-200, 300)
            time.sleep(actual_time / 10000)
            
            return MockResponseFactory.create_perplexity_response(
                query_text,
                confidence=0.7 + random.random() * 0.2,
                response_time_ms=actual_time
            )
        
        backend.query = Mock(side_effect=mock_query)
        return backend
    
    @staticmethod
    def create_cache_backend(hit_rate=0.6, response_time_ms=50, **kwargs):
        """Create a mock cache backend."""
        backend = Mock()
        backend.name = "cache"
        backend.backend_type = "cache"
        backend.hit_rate = hit_rate
        backend.base_response_time_ms = response_time_ms
        backend.cache_data = {}
        backend.call_count = 0
        backend.hit_count = 0
        backend.miss_count = 0
        backend.is_available = Mock(return_value=True)
        
        # Predefined cache patterns for common queries
        cache_patterns = [
            "what is metabolomics",
            "clinical metabolomics",
            "biomarker discovery", 
            "pathway analysis",
            "metabolite identification",
            "mass spectrometry"
        ]
        
        def mock_get(query_key):
            backend.call_count += 1
            query_lower = str(query_key).lower()
            
            # Check for cache hits based on patterns
            cache_hit = any(pattern in query_lower for pattern in cache_patterns)
            
            if cache_hit:
                backend.hit_count += 1
                time.sleep(backend.base_response_time_ms / 10000)
                return MockResponseFactory.create_cache_response(
                    query_key,
                    confidence=0.3 + (backend.hit_rate * 0.4),  # Variable confidence
                    response_time_ms=backend.base_response_time_ms
                )
            else:
                backend.miss_count += 1
                return None  # Cache miss
        
        backend.get = Mock(side_effect=mock_get)
        backend.set = Mock(return_value=True)
        return backend


class MockResponseFactory:
    """Factory for creating mock responses from different backends."""
    
    @staticmethod
    def create_base_confidence_metrics(confidence, response_time_ms=100):
        """Create base confidence metrics for testing."""
        import random
        
        # Add some realistic variation
        base_conf = confidence
        variation = random.random() * 0.1 - 0.05  # ±5% variation
        
        return ConfidenceMetrics(
            overall_confidence=max(0, min(1, base_conf + variation)),
            research_category_confidence=max(0, min(1, base_conf + variation * 0.8)),
            temporal_analysis_confidence=max(0, min(1, base_conf - 0.1 + variation)),
            signal_strength_confidence=max(0, min(1, base_conf - 0.05 + variation)),
            context_coherence_confidence=max(0, min(1, base_conf + variation * 0.6)),
            keyword_density=random.random() * 0.5 + 0.2,
            pattern_match_strength=random.random() * 0.4 + 0.3,
            biomedical_entity_count=random.randint(1, 5),
            ambiguity_score=max(0, min(1, 1 - base_conf + variation)),
            conflict_score=random.random() * 0.4,
            alternative_interpretations=[
                (RoutingDecision.LIGHTRAG, base_conf + random.random() * 0.1),
                (RoutingDecision.PERPLEXITY, base_conf - 0.1 + random.random() * 0.2),
                (RoutingDecision.EITHER, base_conf - 0.05 + random.random() * 0.1)
            ],
            calculation_time_ms=response_time_ms * random.random() * 0.5
        )
    
    @staticmethod
    def create_lightrag_response(query_text, confidence=0.8, response_time_ms=800):
        """Create a mock LightRAG response."""
        confidence_metrics = MockResponseFactory.create_base_confidence_metrics(
            confidence, response_time_ms
        )
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.LIGHTRAG,
            confidence=confidence,
            reasoning=[
                f"LightRAG knowledge graph analysis for: {query_text[:50]}...",
                "High-quality response from curated knowledge base",
                "Strong entity relationships identified"
            ],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=["knowledge_graph", "entity_relationships"],
            metadata={
                'backend': 'lightrag',
                'response_time_ms': response_time_ms,
                'knowledge_graph_used': True,
                'entity_count': 3
            }
        )
    
    @staticmethod
    def create_perplexity_response(query_text, confidence=0.75, response_time_ms=1200):
        """Create a mock Perplexity API response."""
        confidence_metrics = MockResponseFactory.create_base_confidence_metrics(
            confidence, response_time_ms
        )
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.PERPLEXITY,
            confidence=confidence,
            reasoning=[
                f"Perplexity real-time analysis for: {query_text[:50]}...",
                "Current information from web sources", 
                "Recent research findings incorporated"
            ],
            research_category=ResearchCategory.LITERATURE_SEARCH,
            confidence_metrics=confidence_metrics,
            temporal_indicators=["recent", "current"],
            knowledge_indicators=["web_sources", "recent_research"],
            metadata={
                'backend': 'perplexity', 
                'response_time_ms': response_time_ms,
                'real_time_data': True,
                'source_count': 5
            }
        )
    
    @staticmethod
    def create_cache_response(query_text, confidence=0.3, response_time_ms=50):
        """Create a mock cache response."""
        confidence_metrics = MockResponseFactory.create_base_confidence_metrics(
            confidence, response_time_ms
        )
        
        return RoutingPrediction(
            routing_decision=RoutingDecision.EITHER,
            confidence=confidence,
            reasoning=[
                f"Cached response for common pattern: {query_text[:50]}...",
                "Fallback to emergency cache",
                "Reduced confidence but reliable response"
            ],
            research_category=ResearchCategory.GENERAL_QUERY,
            confidence_metrics=confidence_metrics,
            temporal_indicators=[],
            knowledge_indicators=["cached_pattern"],
            metadata={
                'backend': 'cache',
                'response_time_ms': response_time_ms,
                'cache_hit': True,
                'emergency_response': True
            }
        )


# ============================================================================
# TEST UTILITIES
# ============================================================================

class FallbackTestUtils:
    """Utilities for fallback testing scenarios."""
    
    @staticmethod
    def create_test_query_scenarios():
        """Create standardized test query scenarios."""
        return [
            {
                'name': 'primary_success',
                'query': 'What are the metabolic pathways in diabetes?',
                'expected_backend': 'lightrag',
                'expected_confidence_min': 0.7,
                'expected_max_time_ms': 1000,
                'failure_setup': {}
            },
            {
                'name': 'primary_failure_secondary_success', 
                'query': 'Recent advances in metabolomics research',
                'expected_backend': 'perplexity',
                'expected_confidence_min': 0.5,
                'expected_max_time_ms': 1500,
                'failure_setup': {'lightrag_failures': 1}
            },
            {
                'name': 'both_primary_fail_cache_success',
                'query': 'What is clinical metabolomics?',
                'expected_backend': 'cache',
                'expected_confidence_min': 0.2,
                'expected_max_time_ms': 800,
                'failure_setup': {'lightrag_failures': 1, 'perplexity_failures': 1}
            },
            {
                'name': 'all_fail_default_routing',
                'query': 'Complex rare metabolite analysis',
                'expected_backend': 'default',
                'expected_confidence_min': 0.05,
                'expected_max_time_ms': 1000,
                'failure_setup': {
                    'lightrag_failures': 1, 
                    'perplexity_failures': 1, 
                    'cache_unavailable': True
                }
            }
        ]
    
    @staticmethod
    def setup_failure_conditions(backends, failure_setup):
        """Set up failure conditions on mock backends."""
        if 'lightrag_failures' in failure_setup and 'lightrag' in backends:
            backends['lightrag'].failure_count = failure_setup['lightrag_failures']
        
        if 'perplexity_failures' in failure_setup and 'perplexity' in backends:
            backends['perplexity'].failure_count = failure_setup['perplexity_failures']
        
        if 'cache_unavailable' in failure_setup and 'cache' in backends:
            backends['cache'].is_available.return_value = not failure_setup['cache_unavailable']
    
    @staticmethod
    def validate_fallback_result(result, expected_scenario):
        """Validate a fallback result against expected scenario."""
        validations = {
            'success': result.success is True,
            'has_prediction': result.routing_prediction is not None,
            'minimum_confidence': result.routing_prediction.confidence >= expected_scenario['expected_confidence_min'],
            'reasonable_time': result.total_processing_time_ms <= expected_scenario['expected_max_time_ms'],
            'has_reasoning': len(result.routing_prediction.reasoning) > 0
        }
        
        # Additional validations based on backend
        if expected_scenario['expected_backend'] == 'cache':
            validations['used_fallback'] = result.fallback_level_used in [
                FallbackLevel.EMERGENCY_CACHE, FallbackLevel.KEYWORD_BASED_ONLY
            ]
        elif expected_scenario['expected_backend'] == 'default':
            validations['used_default'] = result.fallback_level_used == FallbackLevel.DEFAULT_ROUTING
        
        return validations
    
    @staticmethod
    def measure_performance_characteristics(func, *args, **kwargs):
        """Measure performance characteristics of a function call."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Pre-execution measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu_percent = process.cpu_percent()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Post-execution measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu_percent = process.cpu_percent()
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'execution_time_ms': (end_time - start_time) * 1000,
            'memory_used_mb': end_memory - start_memory,
            'cpu_usage_percent': max(start_cpu_percent, end_cpu_percent),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def create_load_test_scenario(num_queries=20, concurrent=True):
        """Create a load testing scenario."""
        import random
        
        query_templates = [
            "What is {topic} in metabolomics?",
            "How does {topic} affect cellular metabolism?", 
            "Recent research on {topic} and biomarkers",
            "Clinical applications of {topic} analysis",
            "Pathway interactions involving {topic}"
        ]
        
        topics = [
            "glucose", "amino acids", "lipids", "nucleotides", 
            "TCA cycle", "glycolysis", "fatty acid oxidation",
            "oxidative stress", "insulin signaling", "mitochondrial function"
        ]
        
        queries = []
        for i in range(num_queries):
            template = random.choice(query_templates)
            topic = random.choice(topics)
            query = template.format(topic=topic)
            
            queries.append({
                'id': i,
                'query': query,
                'priority': random.choice(['normal', 'high', 'low']),
                'context': {
                    'load_test': True,
                    'query_id': i,
                    'topic': topic,
                    'batch_processing': concurrent
                }
            })
        
        return queries


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

class FallbackPerformanceBenchmark:
    """Performance benchmarking utilities for fallback testing."""
    
    def __init__(self, config: FallbackTestConfig):
        self.config = config
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def start_benchmark(self):
        """Start performance benchmarking."""
        self.start_time = time.time()
        self.results = []
    
    def record_query_result(self, query_info, result, performance_data):
        """Record the result of a query for benchmarking."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'query_id': query_info.get('id', 'unknown'),
            'query_text': query_info.get('query', '')[:100],  # Truncate for storage
            'success': result.success if result else False,
            'fallback_level': result.fallback_level_used.name if result else 'FAILED',
            'confidence': result.routing_prediction.confidence if result and result.routing_prediction else 0,
            'processing_time_ms': result.total_processing_time_ms if result else 0,
            'confidence_degradation': getattr(result, 'confidence_degradation', 0) if result else 1.0,
            'performance_data': performance_data
        }
        self.results.append(record)
    
    def finish_benchmark(self):
        """Finish benchmarking and calculate summary statistics."""
        self.end_time = time.time()
        return self.generate_performance_report()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        total_queries = len(self.results)
        successful_queries = [r for r in self.results if r['success']]
        failed_queries = [r for r in self.results if not r['success']]
        
        # Calculate statistics
        success_rate = len(successful_queries) / total_queries
        
        if successful_queries:
            processing_times = [r['processing_time_ms'] for r in successful_queries]
            confidences = [r['confidence'] for r in successful_queries]
            degradations = [r['confidence_degradation'] for r in successful_queries]
            
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
            
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            
            avg_degradation = sum(degradations) / len(degradations)
            max_degradation = max(degradations)
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0
            avg_confidence = min_confidence = 0
            avg_degradation = max_degradation = 1.0
        
        # Fallback level analysis
        fallback_levels = {}
        for result in successful_queries:
            level = result['fallback_level']
            fallback_levels[level] = fallback_levels.get(level, 0) + 1
        
        # Performance assessment
        performance_assessment = {
            'meets_success_rate_requirement': success_rate >= self.config.required_success_rate,
            'meets_performance_requirements': avg_processing_time <= self.config.max_fallback_response_time_ms,
            'maintains_minimum_confidence': min_confidence >= self.config.min_emergency_confidence,
            'degradation_acceptable': avg_degradation <= 0.6
        }
        
        overall_score = sum(performance_assessment.values()) / len(performance_assessment)
        
        return {
            'benchmark_summary': {
                'total_queries': total_queries,
                'successful_queries': len(successful_queries),
                'failed_queries': len(failed_queries),
                'success_rate': success_rate,
                'benchmark_duration_seconds': self.end_time - self.start_time if self.end_time else 0
            },
            'performance_metrics': {
                'avg_processing_time_ms': avg_processing_time,
                'max_processing_time_ms': max_processing_time,
                'min_processing_time_ms': min_processing_time,
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'avg_confidence_degradation': avg_degradation,
                'max_confidence_degradation': max_degradation
            },
            'fallback_level_distribution': fallback_levels,
            'performance_assessment': performance_assessment,
            'overall_performance_score': overall_score,
            'detailed_results': self.results[-10:] if len(self.results) > 10 else self.results  # Last 10 results
        }


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def fallback_test_config():
    """Provide test configuration for fallback scenarios."""
    return FallbackTestConfig()


@pytest.fixture
def mock_backends():
    """Provide mock backends for testing."""
    return {
        'lightrag': MockBackendFactory.create_lightrag_backend(),
        'perplexity': MockBackendFactory.create_perplexity_backend(),
        'cache': MockBackendFactory.create_cache_backend()
    }


@pytest.fixture
def performance_benchmark(fallback_test_config):
    """Provide performance benchmarking capability."""
    return FallbackPerformanceBenchmark(fallback_test_config)


@pytest.fixture
def test_query_scenarios():
    """Provide standardized test query scenarios."""
    return FallbackTestUtils.create_test_query_scenarios()


# ============================================================================
# EXAMPLE USAGE AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Example usage of the test configuration system
    
    # Create test config
    config = FallbackTestConfig()
    print(f"Test Config: {config}")
    
    # Create mock backends
    lightrag = MockBackendFactory.create_lightrag_backend()
    perplexity = MockBackendFactory.create_perplexity_backend()
    cache = MockBackendFactory.create_cache_backend()
    
    # Test mock functionality
    print(f"LightRAG backend: {lightrag.name}")
    print(f"Perplexity backend: {perplexity.name}")
    print(f"Cache backend: {cache.name}")
    
    # Create test scenarios
    scenarios = FallbackTestUtils.create_test_query_scenarios()
    print(f"Created {len(scenarios)} test scenarios")
    
    # Demo performance measurement
    def sample_function(x):
        time.sleep(0.01)  # Simulate processing
        return x * 2
    
    perf_result = FallbackTestUtils.measure_performance_characteristics(
        sample_function, 42
    )
    print(f"Performance measurement example: {perf_result}")
    
    print("Test configuration system validation complete!")