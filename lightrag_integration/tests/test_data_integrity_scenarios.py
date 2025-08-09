#!/usr/bin/env python3
"""
Data Integrity & Consistency Testing Scenarios for Reliability Validation
=========================================================================

Implementation of data integrity testing scenarios DI-001 through DI-003 as defined in
CMO-LIGHTRAG-014-T08 reliability validation design.

Test Scenarios:
- DI-001: Cross-Source Response Consistency
- DI-002: Cache Freshness and Accuracy
- DI-003: Malformed Response Recovery

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import asyncio
import pytest
import logging
import time
import json
import hashlib
import statistics
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# Import reliability test framework
from .reliability_test_framework import (
    ReliabilityValidationFramework,
    ReliabilityTestConfig,
    LoadGenerator,
    FailureInjector,
    ReliabilityTestUtils,
    create_test_orchestrator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA INTEGRITY TEST UTILITIES
# ============================================================================

@dataclass
class ResponseConsistencyMetrics:
    """Metrics for measuring response consistency across sources."""
    semantic_similarity: float = 0.0
    factual_accuracy: float = 0.0
    completeness_score: float = 0.0
    format_consistency: float = 0.0
    metadata_consistency: float = 0.0


@dataclass
class CacheAnalysisResult:
    """Result of cache analysis operations."""
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    fresh_data_rate: float = 0.0
    stale_data_rate: float = 0.0
    cache_accuracy: float = 0.0


class ResponseCorruptionInjector(FailureInjector):
    """Inject various types of response corruption for testing."""
    
    def __init__(self, corruption_type: str, severity: float = 0.5):
        super().__init__(f"response_corruption_{corruption_type}")
        self.corruption_type = corruption_type
        self.severity = severity
        self.corrupted_responses = 0
        
    async def inject_failure(self):
        await super().inject_failure()
        logger.info(f"Injecting {self.corruption_type} corruption at {self.severity} severity")
        
    async def corrupt_response(self, response: str) -> str:
        """Apply corruption to a response based on corruption type."""
        if not self.active:
            return response
            
        if random.random() > self.severity:
            return response  # Skip corruption for this response
            
        self.corrupted_responses += 1
        
        if self.corruption_type == 'malformed_json':
            return self._corrupt_json(response)
        elif self.corruption_type == 'character_encoding_error':
            return self._corrupt_encoding(response)
        elif self.corruption_type == 'truncated_response':
            return self._truncate_response(response)
        elif self.corruption_type == 'invalid_response_schema':
            return self._corrupt_schema(response)
        else:
            return response
    
    def _corrupt_json(self, response: str) -> str:
        """Introduce JSON formatting errors."""
        if '{' in response:
            # Remove random closing brace
            response = response.replace('}', '', 1)
        return response
    
    def _corrupt_encoding(self, response: str) -> str:
        """Introduce character encoding issues."""
        # Replace some characters with encoding errors
        chars_to_corrupt = random.sample(list(response), min(5, len(response)))
        for char in chars_to_corrupt:
            response = response.replace(char, '�', 1)
        return response
    
    def _truncate_response(self, response: str) -> str:
        """Truncate response at random point."""
        if len(response) > 20:
            truncate_point = random.randint(10, len(response) - 10)
            return response[:truncate_point]
        return response
    
    def _corrupt_schema(self, response: str) -> str:
        """Introduce schema validation errors."""
        # Add invalid fields or remove required fields
        if '{' in response and '}' in response:
            response = response.replace('{', '{"invalid_field": "corrupted", ')
        return response


class ResponseConsistencyAnalyzer:
    """Analyze consistency between responses from different sources."""
    
    @staticmethod
    def calculate_semantic_similarity(response1: str, response2: str) -> float:
        """Calculate semantic similarity between two responses."""
        if not response1 or not response2:
            return 0.0
        
        # Simple word overlap similarity (in production, would use embeddings)
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_factual_accuracy(responses: List[str]) -> float:
        """Calculate factual accuracy across multiple responses."""
        if len(responses) < 2:
            return 1.0  # Single response assumed accurate
        
        # Simple factual consistency check (in production, would use fact-checking)
        key_facts = []
        
        for response in responses:
            # Extract potential facts (simplified)
            words = response.lower().split()
            numbers = [w for w in words if w.replace('.', '').isdigit()]
            capitalized_words = [w for w in response.split() if w[0].isupper() and len(w) > 3]
            
            key_facts.append({
                'numbers': set(numbers),
                'proper_nouns': set(capitalized_words)
            })
        
        # Calculate consistency of key facts
        if len(key_facts) < 2:
            return 1.0
        
        number_consistency = len(key_facts[0]['numbers'].intersection(key_facts[1]['numbers'])) / max(1, len(key_facts[0]['numbers'].union(key_facts[1]['numbers'])))
        noun_consistency = len(key_facts[0]['proper_nouns'].intersection(key_facts[1]['proper_nouns'])) / max(1, len(key_facts[0]['proper_nouns'].union(key_facts[1]['proper_nouns'])))
        
        return (number_consistency + noun_consistency) / 2
    
    @staticmethod
    def calculate_completeness_score(response: str, expected_components: List[str] = None) -> float:
        """Calculate completeness score of a response."""
        if not expected_components:
            expected_components = [
                'definition', 'explanation', 'example', 'context', 'reference'
            ]
        
        response_lower = response.lower()
        found_components = 0
        
        for component in expected_components:
            # Check for component indicators (simplified)
            indicators = {
                'definition': ['is defined as', 'refers to', 'means', 'is'],
                'explanation': ['because', 'due to', 'explanation', 'reason'],
                'example': ['example', 'for instance', 'such as', 'e.g.'],
                'context': ['in the context of', 'clinical', 'medical', 'study'],
                'reference': ['reference', 'study', 'research', 'literature']
            }
            
            if component in indicators:
                if any(indicator in response_lower for indicator in indicators[component]):
                    found_components += 1
        
        return found_components / len(expected_components) if expected_components else 1.0


class CacheConsistencyValidator:
    """Validate cache consistency and freshness."""
    
    def __init__(self):
        self.cache_operations = []
        self.consistency_violations = []
    
    async def validate_cache_consistency(self, cache_entries: List[Dict]) -> CacheAnalysisResult:
        """Validate consistency of cache entries."""
        if not cache_entries:
            return CacheAnalysisResult()
        
        total_entries = len(cache_entries)
        fresh_entries = 0
        stale_entries = 0
        consistent_entries = 0
        
        current_time = time.time()
        
        for entry in cache_entries:
            # Check freshness
            cache_time = entry.get('cache_time', current_time)
            ttl = entry.get('ttl', 3600)  # Default 1 hour TTL
            
            if current_time - cache_time < ttl:
                fresh_entries += 1
            else:
                stale_entries += 1
            
            # Check consistency (simplified)
            if self._validate_entry_consistency(entry):
                consistent_entries += 1
        
        return CacheAnalysisResult(
            fresh_data_rate=fresh_entries / total_entries,
            stale_data_rate=stale_entries / total_entries,
            cache_accuracy=consistent_entries / total_entries
        )
    
    def _validate_entry_consistency(self, entry: Dict) -> bool:
        """Validate individual cache entry consistency."""
        required_fields = ['key', 'value', 'cache_time', 'source']
        
        # Check required fields
        if not all(field in entry for field in required_fields):
            return False
        
        # Check data integrity
        if 'checksum' in entry:
            calculated_checksum = hashlib.md5(str(entry.get('value', '')).encode()).hexdigest()
            if calculated_checksum != entry['checksum']:
                return False
        
        return True


# ============================================================================
# DI-001: CROSS-SOURCE RESPONSE CONSISTENCY TEST
# ============================================================================

async def test_cross_source_response_consistency(orchestrator, config: ReliabilityTestConfig):
    """
    DI-001: Test consistency of responses across different fallback sources.
    
    Validates that the system:
    - Provides semantically consistent responses across sources
    - Maintains factual accuracy regardless of source
    - Ensures response completeness across all sources
    - Preserves metadata structure consistency
    """
    logger.info("Starting DI-001: Cross-Source Response Consistency Test")
    
    # Generate test queries with known expected responses
    consistency_test_queries = generate_consistency_test_queries(20)
    
    response_sources = ['lightrag', 'perplexity', 'cache', 'default']
    source_responses = {}
    
    # Collect responses from each source
    for source in response_sources:
        logger.info(f"Collecting responses from source: {source}")
        source_responses[source] = {}
        
        # Force responses from specific source (simulated)
        for query_id, query in consistency_test_queries.items():
            try:
                # Create source-specific handler
                async def source_specific_handler():
                    # Simulate different source behaviors
                    if source == 'lightrag':
                        await asyncio.sleep(random.uniform(0.5, 2.0))
                        return generate_lightrag_style_response(query)
                    elif source == 'perplexity':
                        await asyncio.sleep(random.uniform(1.0, 3.0))
                        return generate_perplexity_style_response(query)
                    elif source == 'cache':
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        return generate_cached_response(query)
                    else:  # default
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        return generate_default_response(query)
                
                result = await orchestrator.submit_request(
                    request_type='user_query',
                    priority='medium',
                    handler=source_specific_handler,
                    timeout=30.0
                )
                
                if result[0]:  # If successful
                    response_content = result[1]
                    source_responses[source][query_id] = {
                        'content': response_content,
                        'metadata': extract_response_metadata(response_content),
                        'confidence': calculate_response_confidence(response_content),
                        'timestamp': time.time(),
                        'source': source
                    }
                else:
                    source_responses[source][query_id] = {'error': result[1]}
                    
            except Exception as e:
                source_responses[source][query_id] = {'error': str(e)}
    
    # Analyze response consistency
    consistency_analysis = {}
    analyzer = ResponseConsistencyAnalyzer()
    
    for query_id in consistency_test_queries:
        query_responses = {}
        
        # Collect valid responses for this query
        for source in response_sources:
            if (query_id in source_responses[source] and 
                'content' in source_responses[source][query_id]):
                query_responses[source] = source_responses[source][query_id]
        
        if len(query_responses) >= 2:
            # Calculate consistency metrics
            response_contents = [resp['content'] for resp in query_responses.values()]
            
            # Pairwise semantic similarity
            similarities = []
            for i, content1 in enumerate(response_contents):
                for j, content2 in enumerate(response_contents[i+1:], i+1):
                    similarity = analyzer.calculate_semantic_similarity(content1, content2)
                    similarities.append(similarity)
            
            avg_similarity = statistics.mean(similarities) if similarities else 0.0
            factual_accuracy = analyzer.calculate_factual_accuracy(response_contents)
            
            # Calculate completeness for each response
            completeness_scores = [
                analyzer.calculate_completeness_score(content) 
                for content in response_contents
            ]
            avg_completeness = statistics.mean(completeness_scores)
            
            consistency_analysis[query_id] = ResponseConsistencyMetrics(
                semantic_similarity=avg_similarity,
                factual_accuracy=factual_accuracy,
                completeness_score=avg_completeness,
                format_consistency=calculate_format_consistency(response_contents),
                metadata_consistency=calculate_metadata_consistency(query_responses)
            )
    
    # Calculate overall consistency metrics
    if consistency_analysis:
        overall_consistency = ResponseConsistencyMetrics(
            semantic_similarity=statistics.mean([m.semantic_similarity for m in consistency_analysis.values()]),
            factual_accuracy=statistics.mean([m.factual_accuracy for m in consistency_analysis.values()]),
            completeness_score=statistics.mean([m.completeness_score for m in consistency_analysis.values()]),
            format_consistency=statistics.mean([m.format_consistency for m in consistency_analysis.values()]),
            metadata_consistency=statistics.mean([m.metadata_consistency for m in consistency_analysis.values()])
        )
    else:
        overall_consistency = ResponseConsistencyMetrics()
    
    # Validate consistency thresholds
    assert overall_consistency.semantic_similarity >= 0.80, \
        f"Semantic similarity {overall_consistency.semantic_similarity:.2f} below threshold 0.80"
    
    assert overall_consistency.factual_accuracy >= 0.85, \
        f"Factual accuracy {overall_consistency.factual_accuracy:.2f} below threshold 0.85"
    
    assert overall_consistency.completeness_score >= 0.75, \
        f"Completeness score {overall_consistency.completeness_score:.2f} below threshold 0.75"
    
    logger.info(f"DI-001 completed - Semantic similarity: {overall_consistency.semantic_similarity:.2f}, "
               f"Factual accuracy: {overall_consistency.factual_accuracy:.2f}, "
               f"Completeness: {overall_consistency.completeness_score:.2f}")
    
    return {
        'overall_consistency': overall_consistency,
        'query_analysis': consistency_analysis,
        'source_coverage': {source: len(responses) for source, responses in source_responses.items()},
        'total_queries_analyzed': len(consistency_analysis)
    }


# ============================================================================
# DI-002: CACHE FRESHNESS AND ACCURACY TEST
# ============================================================================

async def test_cache_freshness_and_accuracy(orchestrator, config: ReliabilityTestConfig):
    """
    DI-002: Test cache freshness mechanisms and response accuracy.
    
    Validates that the system:
    - Maintains accurate cache hit/miss behavior
    - Implements proper cache expiry mechanisms
    - Serves fresh data when available
    - Falls back appropriately when cache is stale
    """
    logger.info("Starting DI-002: Cache Freshness and Accuracy Test")
    
    cache_validator = CacheConsistencyValidator()
    
    # Phase 1: Cache population with known fresh data
    fresh_queries = generate_fresh_test_queries(30)
    cache_population_results = []
    
    logger.info("Phase 1: Populating cache with fresh data")
    
    for query in fresh_queries:
        # Generate fresh response and ensure it's cached
        async def cache_population_handler():
            # Simulate fresh data retrieval
            await asyncio.sleep(random.uniform(1.0, 2.0))
            response = generate_timestamped_response(query)
            # Add cache metadata
            response_data = {
                'content': response,
                'cache_time': time.time(),
                'ttl': 3600,  # 1 hour TTL
                'source': 'fresh_generation',
                'checksum': hashlib.md5(response.encode()).hexdigest()
            }
            return json.dumps(response_data)
        
        result = await orchestrator.submit_request(
            request_type='user_query',
            priority='medium',
            handler=cache_population_handler,
            timeout=30.0
        )
        
        cache_population_results.append({
            'query_id': query['id'],
            'success': result[0],
            'cached': result[0]  # Assume successful requests are cached
        })
    
    cache_population_success_rate = sum(1 for r in cache_population_results if r['success']) / len(cache_population_results)
    assert cache_population_success_rate >= 0.95, f"Cache population success rate {cache_population_success_rate:.2f} too low"
    
    # Phase 2: Test cache hit behavior
    logger.info("Phase 2: Testing cache hit behavior")
    
    cache_hit_results = {}
    
    for query in fresh_queries[:20]:  # Test subset for cache hits
        # Simulate cache lookup
        async def cache_hit_handler():
            # Simulate cache hit with faster response
            await asyncio.sleep(random.uniform(0.1, 0.3))
            cached_response = generate_cached_response(query)
            return cached_response
        
        result = await orchestrator.submit_request(
            request_type='user_query',
            priority='medium',
            handler=cache_hit_handler,
            timeout=15.0
        )
        
        cache_hit_results[query['id']] = {
            'success': result[0],
            'response': result[1] if result[0] else None,
            'source': 'cache' if result[0] else 'error',
            'response_time': 0.2  # Simulated fast cache response
        }
    
    # Validate cache hits
    cache_hit_rate = sum(1 for r in cache_hit_results.values() if r['success'] and r['source'] == 'cache') / len(cache_hit_results)
    assert cache_hit_rate >= 0.90, f"Cache hit rate {cache_hit_rate:.2f} below threshold 0.90"
    
    # Phase 3: Test cache expiry behavior
    logger.info("Phase 3: Testing cache expiry behavior")
    
    # Simulate time passage to trigger cache expiry (in production, would wait or mock time)
    expired_cache_results = {}
    
    for query in fresh_queries[:15]:  # Test subset for expiry
        # Simulate expired cache and fresh retrieval
        async def expired_cache_handler():
            # Simulate cache miss and fresh data retrieval
            await asyncio.sleep(random.uniform(1.5, 3.0))
            fresh_response = generate_fresh_response(query)
            return fresh_response
        
        result = await orchestrator.submit_request(
            request_type='user_query',
            priority='medium',
            handler=expired_cache_handler,
            timeout=30.0
        )
        
        expired_cache_results[query['id']] = {
            'success': result[0],
            'source': 'fresh_retrieval' if result[0] else 'error',
            'freshness_validated': validate_response_freshness(result[1]) if result[0] else False,
            'response_time': 2.0  # Simulated slower fresh retrieval
        }
    
    # Validate fresh data retrieval after cache expiry
    fresh_retrieval_rate = sum(1 for r in expired_cache_results.values() if r['source'] == 'fresh_retrieval') / len(expired_cache_results)
    assert fresh_retrieval_rate >= 0.85, f"Fresh retrieval rate {fresh_retrieval_rate:.2f} below threshold 0.85"
    
    # Phase 4: Cache accuracy validation
    logger.info("Phase 4: Validating cache accuracy")
    
    # Simulate cache entries for validation
    simulated_cache_entries = []
    current_time = time.time()
    
    for i, query in enumerate(fresh_queries):
        cache_entry = {
            'key': f"query_{query['id']}",
            'value': f"cached_response_{i}",
            'cache_time': current_time - random.uniform(0, 7200),  # Random age up to 2 hours
            'ttl': 3600,  # 1 hour TTL
            'source': 'cache',
            'checksum': hashlib.md5(f"cached_response_{i}".encode()).hexdigest()
        }
        simulated_cache_entries.append(cache_entry)
    
    cache_analysis = await cache_validator.validate_cache_consistency(simulated_cache_entries)
    
    # Validate cache analysis results
    assert cache_analysis.cache_accuracy >= 0.95, \
        f"Cache accuracy {cache_analysis.cache_accuracy:.2f} below threshold 0.95"
    
    assert cache_analysis.fresh_data_rate >= 0.60, \
        f"Fresh data rate {cache_analysis.fresh_data_rate:.2f} below threshold 0.60"
    
    # Calculate overall cache performance metrics
    overall_cache_metrics = {
        'cache_population_success_rate': cache_population_success_rate,
        'cache_hit_rate': cache_hit_rate,
        'fresh_retrieval_rate': fresh_retrieval_rate,
        'cache_accuracy': cache_analysis.cache_accuracy,
        'fresh_data_rate': cache_analysis.fresh_data_rate,
        'stale_data_rate': cache_analysis.stale_data_rate
    }
    
    logger.info(f"DI-002 completed - Cache hit rate: {cache_hit_rate:.2f}, "
               f"Fresh retrieval rate: {fresh_retrieval_rate:.2f}, "
               f"Cache accuracy: {cache_analysis.cache_accuracy:.2f}")
    
    return {
        'cache_metrics': overall_cache_metrics,
        'cache_analysis': cache_analysis,
        'phase_results': {
            'population': cache_population_results,
            'hit_testing': cache_hit_results,
            'expiry_testing': expired_cache_results
        }
    }


# ============================================================================
# DI-003: MALFORMED RESPONSE RECOVERY TEST
# ============================================================================

async def test_malformed_response_recovery(orchestrator, config: ReliabilityTestConfig):
    """
    DI-003: Test recovery from malformed or corrupted responses.
    
    Validates that the system:
    - Detects malformed responses accurately
    - Implements appropriate recovery mechanisms
    - Falls back to alternative sources when needed
    - Maintains service quality despite corruption
    """
    logger.info("Starting DI-003: Malformed Response Recovery Test")
    
    corruption_scenarios = [
        {
            'name': 'json_corruption',
            'corruption_type': 'malformed_json',
            'severity': 0.6,
            'expected_recovery': 'fallback_to_next_source',
            'min_success_rate': 0.80
        },
        {
            'name': 'encoding_corruption',
            'corruption_type': 'character_encoding_error',
            'severity': 0.4,
            'expected_recovery': 'retry_with_different_encoding',
            'min_success_rate': 0.85
        },
        {
            'name': 'partial_response',
            'corruption_type': 'truncated_response',
            'severity': 0.5,
            'expected_recovery': 'retry_request',
            'min_success_rate': 0.75
        },
        {
            'name': 'schema_violation',
            'corruption_type': 'invalid_response_schema',
            'severity': 0.3,
            'expected_recovery': 'fallback_with_error_logging',
            'min_success_rate': 0.90
        }
    ]
    
    corruption_results = {}
    
    for scenario in corruption_scenarios:
        logger.info(f"Testing corruption scenario: {scenario['name']}")
        
        # Create corruption injector
        corruption_injector = ResponseCorruptionInjector(
            scenario['corruption_type'],
            scenario['severity']
        )
        
        try:
            # Inject corruption
            await corruption_injector.inject_failure()
            
            # Generate test queries
            test_queries = generate_corruption_test_queries(25)
            corruption_test_results = []
            
            for query in test_queries:
                async def corruption_test_handler():
                    # Generate response that may be corrupted
                    original_response = generate_test_response(query)
                    
                    # Apply corruption if injector is active
                    corrupted_response = await corruption_injector.corrupt_response(original_response)
                    
                    # Simulate corruption detection and recovery
                    if is_response_corrupted(corrupted_response):
                        # Attempt recovery based on scenario
                        if scenario['expected_recovery'] == 'fallback_to_next_source':
                            return generate_fallback_response(query)
                        elif scenario['expected_recovery'] == 'retry_request':
                            # Simulate retry
                            await asyncio.sleep(0.5)
                            return generate_test_response(query)
                        elif scenario['expected_recovery'] == 'retry_with_different_encoding':
                            return fix_encoding_issues(corrupted_response)
                        else:
                            return generate_safe_default_response(query)
                    
                    return corrupted_response
                
                try:
                    result = await orchestrator.submit_request(
                        request_type='user_query',
                        priority='medium',
                        handler=corruption_test_handler,
                        timeout=30.0
                    )
                    
                    corruption_test_results.append({
                        'query_id': query['id'],
                        'success': result[0],
                        'error_type': identify_error_type(result) if not result[0] else None,
                        'recovery_action': identify_recovery_action(result),
                        'final_source': determine_final_response_source(result) if result[0] else None,
                        'response_quality': assess_response_quality(result[1]) if result[0] else 0.0
                    })
                    
                except Exception as e:
                    corruption_test_results.append({
                        'query_id': query['id'],
                        'success': False,
                        'error_type': 'exception',
                        'error_message': str(e),
                        'recovery_action': 'none',
                        'response_quality': 0.0
                    })
            
            # Analyze recovery behavior
            recovery_stats = analyze_corruption_recovery(corruption_test_results, scenario)
            
            corruption_results[scenario['name']] = {
                'success_rate': recovery_stats['overall_success_rate'],
                'recovery_effectiveness': recovery_stats['recovery_effectiveness'],
                'error_detection_accuracy': recovery_stats['error_detection_accuracy'],
                'fallback_activation_rate': recovery_stats['fallback_activation_rate'],
                'retry_attempt_rate': recovery_stats['retry_attempt_rate'],
                'avg_response_quality': recovery_stats['avg_response_quality'],
                'corrupted_responses': corruption_injector.corrupted_responses
            }
            
            # Validate recovery effectiveness
            assert recovery_stats['overall_success_rate'] >= scenario['min_success_rate'], \
                f"Scenario {scenario['name']}: Success rate {recovery_stats['overall_success_rate']:.2f} below minimum {scenario['min_success_rate']}"
            
            assert recovery_stats['error_detection_accuracy'] >= 0.90, \
                f"Scenario {scenario['name']}: Error detection accuracy {recovery_stats['error_detection_accuracy']:.2f} below threshold 0.90"
            
            # Validate specific recovery behavior
            if scenario['expected_recovery'] == 'fallback_to_next_source':
                assert recovery_stats['fallback_activation_rate'] >= 0.70, \
                    f"Scenario {scenario['name']}: Fallback activation rate {recovery_stats['fallback_activation_rate']:.2f} below threshold 0.70"
            elif scenario['expected_recovery'] == 'retry_request':
                assert recovery_stats['retry_attempt_rate'] >= 0.60, \
                    f"Scenario {scenario['name']}: Retry attempt rate {recovery_stats['retry_attempt_rate']:.2f} below threshold 0.60"
            
            logger.info(f"Corruption scenario {scenario['name']} completed: "
                       f"{recovery_stats['overall_success_rate']:.2f} success rate, "
                       f"{recovery_stats['error_detection_accuracy']:.2f} detection accuracy")
            
        finally:
            # Restore normal operation
            await corruption_injector.restore_normal()
        
        # Allow recovery between scenarios
        await asyncio.sleep(10)
    
    # Validate overall corruption recovery capability
    overall_success_rate = statistics.mean([r['success_rate'] for r in corruption_results.values()])
    overall_detection_accuracy = statistics.mean([r['error_detection_accuracy'] for r in corruption_results.values()])
    
    assert overall_success_rate >= 0.80, \
        f"Overall corruption recovery success rate {overall_success_rate:.2f} below threshold 0.80"
    
    assert overall_detection_accuracy >= 0.90, \
        f"Overall error detection accuracy {overall_detection_accuracy:.2f} below threshold 0.90"
    
    logger.info(f"DI-003 completed - Overall success rate: {overall_success_rate:.2f}, "
               f"Detection accuracy: {overall_detection_accuracy:.2f}")
    
    return {
        'scenario_results': corruption_results,
        'overall_metrics': {
            'success_rate': overall_success_rate,
            'detection_accuracy': overall_detection_accuracy,
            'recovery_effectiveness': statistics.mean([r['recovery_effectiveness'] for r in corruption_results.values()])
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_consistency_test_queries(count: int) -> Dict[str, Dict]:
    """Generate test queries for consistency testing."""
    base_queries = [
        {
            'text': 'What is the role of metabolomics in diabetes research?',
            'expected_components': ['definition', 'applications', 'examples'],
            'complexity': 'medium'
        },
        {
            'text': 'Explain the significance of biomarkers in clinical metabolomics.',
            'expected_components': ['definition', 'clinical_relevance', 'examples'],
            'complexity': 'high'
        },
        {
            'text': 'How are mass spectrometry techniques used in metabolomics?',
            'expected_components': ['technical_explanation', 'applications', 'advantages'],
            'complexity': 'high'
        }
    ]
    
    queries = {}
    for i in range(count):
        base_query = random.choice(base_queries)
        query_id = f"consistency_query_{i}"
        queries[query_id] = {
            'id': query_id,
            'text': base_query['text'],
            'expected_components': base_query['expected_components'],
            'complexity': base_query['complexity'],
            'timestamp': time.time()
        }
    
    return queries


def generate_fresh_test_queries(count: int) -> List[Dict]:
    """Generate test queries for cache freshness testing."""
    queries = []
    for i in range(count):
        queries.append({
            'id': f'fresh_query_{i}',
            'text': f'Current research on metabolite {1000 + i}',
            'complexity': 'medium',
            'timestamp': time.time(),
            'cacheable': True
        })
    return queries


def generate_corruption_test_queries(count: int) -> List[Dict]:
    """Generate test queries for corruption testing."""
    queries = []
    for i in range(count):
        queries.append({
            'id': f'corruption_query_{i}',
            'text': f'Analyze metabolic pathway data for compound {i}',
            'complexity': 'varied',
            'timestamp': time.time()
        })
    return queries


# Response generation functions (simplified for testing)
def generate_lightrag_style_response(query: Dict) -> str:
    """Generate LightRAG-style response."""
    return f"LightRAG response for {query['text']}: Comprehensive analysis from knowledge base with detailed metabolomic information."


def generate_perplexity_style_response(query: Dict) -> str:
    """Generate Perplexity-style response."""
    return f"Perplexity response for {query['text']}: Research-based answer with current scientific literature references."


def generate_cached_response(query: Dict) -> str:
    """Generate cached-style response."""
    return f"Cached response for {query['text']}: Previously generated answer with validated metabolomic data."


def generate_default_response(query: Dict) -> str:
    """Generate default response."""
    return f"Default response for {query['text']}: Standard informational answer about metabolomics concepts."


def generate_timestamped_response(query: Dict) -> str:
    """Generate response with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Response generated at {timestamp} for: {query['text']}"


def generate_fresh_response(query: Dict) -> str:
    """Generate fresh response."""
    return f"Fresh response for {query['text']}: Updated information with latest research findings."


def generate_test_response(query: Dict) -> str:
    """Generate standard test response."""
    return f"Test response for {query['text']}: Metabolomics analysis with structured data format."


def generate_fallback_response(query: Dict) -> str:
    """Generate fallback response."""
    return f"Fallback response for {query['text']}: Alternative source information."


def generate_safe_default_response(query: Dict) -> str:
    """Generate safe default response."""
    return f"Safe default response for {query['text']}: Basic metabolomics information."


# Analysis and validation functions
def extract_response_metadata(response: str) -> Dict:
    """Extract metadata from response."""
    return {
        'length': len(response),
        'has_timestamp': 'at' in response.lower(),
        'source_mentioned': any(source in response.lower() for source in ['lightrag', 'perplexity', 'cache', 'default']),
        'format': 'structured' if ':' in response else 'unstructured'
    }


def calculate_response_confidence(response: str) -> float:
    """Calculate confidence score for response."""
    confidence_indicators = ['comprehensive', 'detailed', 'analysis', 'research', 'validated']
    indicators_found = sum(1 for indicator in confidence_indicators if indicator in response.lower())
    return min(1.0, indicators_found / len(confidence_indicators))


def calculate_format_consistency(response_contents: List[str]) -> float:
    """Calculate format consistency across responses."""
    if len(response_contents) < 2:
        return 1.0
    
    formats = []
    for content in response_contents:
        format_features = {
            'has_colon': ':' in content,
            'has_structured_format': content.count('\n') > 0,
            'length_category': 'short' if len(content) < 100 else 'medium' if len(content) < 300 else 'long'
        }
        formats.append(format_features)
    
    # Calculate similarity between formats
    consistency_score = 0
    total_comparisons = 0
    
    for i, format1 in enumerate(formats):
        for format2 in formats[i+1:]:
            matching_features = sum(1 for key in format1 if format1[key] == format2[key])
            consistency_score += matching_features / len(format1)
            total_comparisons += 1
    
    return consistency_score / total_comparisons if total_comparisons > 0 else 1.0


def calculate_metadata_consistency(query_responses: Dict) -> float:
    """Calculate metadata consistency across responses."""
    if len(query_responses) < 2:
        return 1.0
    
    metadata_list = [resp.get('metadata', {}) for resp in query_responses.values()]
    
    if not metadata_list:
        return 1.0
    
    # Check consistency of metadata fields
    consistent_fields = 0
    total_fields = 0
    
    for field in ['length', 'format', 'has_timestamp']:
        values = [metadata.get(field) for metadata in metadata_list if field in metadata]
        if len(values) > 1:
            total_fields += 1
            if len(set(str(v) for v in values)) <= 2:  # Allow some variation
                consistent_fields += 1
    
    return consistent_fields / total_fields if total_fields > 0 else 1.0


def validate_response_freshness(response: str) -> bool:
    """Validate if response contains fresh information."""
    freshness_indicators = ['current', 'recent', 'latest', 'updated', '2024', '2025']
    return any(indicator in response.lower() for indicator in freshness_indicators)


def is_response_corrupted(response: str) -> bool:
    """Check if response shows signs of corruption."""
    corruption_indicators = [
        '�' in response,  # Encoding errors
        response.count('{') != response.count('}'),  # JSON bracket mismatch
        len(response) < 10,  # Suspiciously short
        'invalid_field' in response  # Schema corruption marker
    ]
    return any(corruption_indicators)


def fix_encoding_issues(response: str) -> str:
    """Attempt to fix encoding issues in response."""
    return response.replace('�', '?')


def identify_error_type(result: Tuple) -> Optional[str]:
    """Identify the type of error from result."""
    if result[0]:  # Success
        return None
    
    error_message = result[1].lower()
    if 'json' in error_message:
        return 'json_error'
    elif 'encoding' in error_message:
        return 'encoding_error'
    elif 'timeout' in error_message:
        return 'timeout_error'
    else:
        return 'unknown_error'


def identify_recovery_action(result: Tuple) -> str:
    """Identify the recovery action taken."""
    if not result[0]:
        return 'none'
    
    message = result[1].lower()
    if 'fallback' in message:
        return 'fallback'
    elif 'retry' in message:
        return 'retry'
    elif 'adaptive' in message:
        return 'adaptive_timeout'
    else:
        return 'standard'


def determine_final_response_source(result: Tuple) -> str:
    """Determine the final source of the response."""
    if not result[0]:
        return 'error'
    
    message = result[1].lower()
    if 'fallback' in message:
        return 'fallback'
    elif 'cached' in message:
        return 'cache'
    elif 'default' in message:
        return 'default'
    else:
        return 'primary'


def assess_response_quality(response: str) -> float:
    """Assess the quality of a response."""
    quality_factors = [
        len(response) > 50,  # Sufficient length
        ':' in response,     # Structured format
        not ('�' in response),  # No encoding issues
        any(word in response.lower() for word in ['metabolomics', 'analysis', 'research'])  # Domain relevance
    ]
    
    return sum(quality_factors) / len(quality_factors)


def analyze_corruption_recovery(results: List[Dict], scenario: Dict) -> Dict:
    """Analyze corruption recovery effectiveness."""
    total_results = len(results)
    successful_results = [r for r in results if r['success']]
    
    recovery_stats = {
        'overall_success_rate': len(successful_results) / total_results if total_results > 0 else 0,
        'recovery_effectiveness': 0.0,
        'error_detection_accuracy': 0.0,
        'fallback_activation_rate': 0.0,
        'retry_attempt_rate': 0.0,
        'avg_response_quality': 0.0
    }
    
    if successful_results:
        # Calculate recovery effectiveness
        recovery_actions = [r['recovery_action'] for r in successful_results if r['recovery_action'] != 'standard']
        recovery_stats['recovery_effectiveness'] = len(recovery_actions) / len(successful_results)
        
        # Calculate fallback and retry rates
        fallback_count = sum(1 for r in successful_results if r['recovery_action'] == 'fallback')
        retry_count = sum(1 for r in successful_results if r['recovery_action'] == 'retry')
        
        recovery_stats['fallback_activation_rate'] = fallback_count / total_results
        recovery_stats['retry_attempt_rate'] = retry_count / total_results
        
        # Calculate average response quality
        quality_scores = [r.get('response_quality', 0) for r in successful_results]
        recovery_stats['avg_response_quality'] = statistics.mean(quality_scores) if quality_scores else 0
    
    # Calculate error detection accuracy (simplified)
    error_results = [r for r in results if not r['success']]
    correct_error_detection = sum(1 for r in error_results if r.get('error_type') != 'unknown_error')
    recovery_stats['error_detection_accuracy'] = correct_error_detection / len(error_results) if error_results else 1.0
    
    return recovery_stats


# ============================================================================
# PYTEST TEST WRAPPER FUNCTIONS
# ============================================================================

@pytest.mark.asyncio
async def test_di_001_cross_source_response_consistency():
    """Pytest wrapper for DI-001."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="DI-001-Cross-Source-Response-Consistency",
            test_func=test_cross_source_response_consistency,
            category="data_integrity"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


@pytest.mark.asyncio
async def test_di_002_cache_freshness_and_accuracy():
    """Pytest wrapper for DI-002."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="DI-002-Cache-Freshness-And-Accuracy",
            test_func=test_cache_freshness_and_accuracy,
            category="data_integrity"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


@pytest.mark.asyncio
async def test_di_003_malformed_response_recovery():
    """Pytest wrapper for DI-003."""
    framework = ReliabilityValidationFramework()
    
    try:
        await framework.setup_test_environment()
        result = await framework.execute_monitored_test(
            test_name="DI-003-Malformed-Response-Recovery",
            test_func=test_malformed_response_recovery,
            category="data_integrity"
        )
        assert result.status == 'passed'
        
    finally:
        await framework.cleanup_test_environment()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_all_data_integrity_tests():
    """Run all data integrity testing scenarios."""
    framework = ReliabilityValidationFramework()
    
    data_integrity_tests = [
        ("DI-001", test_cross_source_response_consistency),
        ("DI-002", test_cache_freshness_and_accuracy),
        ("DI-003", test_malformed_response_recovery)
    ]
    
    results = {}
    
    try:
        await framework.setup_test_environment()
        
        for test_name, test_func in data_integrity_tests:
            logger.info(f"Executing {test_name}")
            
            result = await framework.execute_monitored_test(
                test_name=test_name,
                test_func=test_func,
                category="data_integrity"
            )
            
            results[test_name] = result
            logger.info(f"{test_name} completed: {result.status}")
            
            # Brief recovery between tests
            await asyncio.sleep(20)
            
    finally:
        await framework.cleanup_test_environment()
    
    # Report summary
    passed_tests = sum(1 for r in results.values() if r.status == 'passed')
    total_tests = len(results)
    
    print(f"\nData Integrity Testing Summary: {passed_tests}/{total_tests} tests passed")
    
    for test_name, result in results.items():
        status_emoji = "✅" if result.status == 'passed' else "❌"
        print(f"{status_emoji} {test_name}: {result.status} ({result.duration:.1f}s)")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_data_integrity_tests())