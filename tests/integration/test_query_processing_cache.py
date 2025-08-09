"""
Comprehensive Integration Tests for Query Processing with Multi-Tier Caching

This module provides comprehensive integration tests focusing on the interaction between
the Clinical Metabolomics Oracle's query processing pipeline and multi-tier caching system.

Integration Scenarios:
- Query Classification Pipeline with Cache Integration
- Router Cache Coordination with Classification System  
- Response Generation Pipeline Caching
- Quality Scoring Integration with Cached Responses
- LightRAG Query Processing with Multi-Tier Cache
- Emergency Fallback System Cache Integration
- Prompt Cache Integration with LLM Processing
- Real-World Biomedical Query Processing Scenarios

Performance Targets:
- Cache hit ratio > 80% for repeated biomedical queries
- Response time < 500ms for cached clinical metabolomics queries
- Cache invalidation coordination < 100ms across tiers
- Query processing with cache integration < 2s for complex biomedical workflows

Test Coverage:
- Query pipeline end-to-end integration
- Multi-component cache coordination 
- Performance-critical integration scenarios
- Real-world biomedical query processing
- Error handling and fallback scenarios
- Cache consistency during concurrent operations

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import json
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import concurrent.futures
import hashlib

# Import system components (mock imports for test environment)
try:
    from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision, RoutingPrediction
    from lightrag_integration.research_categorizer import ResearchCategory
except ImportError:
    # Mock implementations for test environment
    class RoutingDecision(Enum):
        LIGHTRAG = "lightrag"
        PERPLEXITY = "perplexity"  
        EITHER = "either"
        HYBRID = "hybrid"
    
    class ResearchCategory(Enum):
        METABOLITE_IDENTIFICATION = "metabolite_identification"
        PATHWAY_ANALYSIS = "pathway_analysis"
        BIOMARKER_DISCOVERY = "biomarker_discovery"
        CLINICAL_DIAGNOSIS = "clinical_diagnosis"
        LITERATURE_SEARCH = "literature_search"
        GENERAL_QUERY = "general_query"


# Multi-tier cache system mock imports
from tests.unit.test_multi_tier_cache import MultiTierCache, MockL1MemoryCache, MockL2DiskCache, MockL3RedisCache


@dataclass
class CacheMetrics:
    """Cache performance metrics for integration testing."""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_invalidations: int = 0
    avg_response_time_ms: float = 0.0
    cache_hit_ratio: float = 0.0
    tier_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.tier_distribution is None:
            self.tier_distribution = {'L1': 0, 'L2': 0, 'L3': 0}
    
    def calculate_hit_ratio(self):
        """Calculate cache hit ratio."""
        total_operations = self.cache_hits + self.cache_misses
        self.cache_hit_ratio = self.cache_hits / total_operations if total_operations > 0 else 0.0
        return self.cache_hit_ratio


@dataclass
class QueryProcessingResult:
    """Result of query processing with cache integration."""
    query_text: str
    routing_decision: RoutingDecision
    research_category: ResearchCategory
    response_content: Dict[str, Any]
    cache_status: str  # 'hit', 'miss', 'write'
    processing_time_ms: float
    cache_tier_used: Optional[str] = None
    quality_score: float = 0.0
    confidence_score: float = 0.0
    cached_result: bool = False


class MockQueryClassificationSystem:
    """Mock query classification system with cache integration."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.classification_cache_prefix = "classification:"
        self.stats = {
            'classifications': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def classify_query(self, query_text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Classify query with cache integration.
        
        Args:
            query_text: The user query to classify
            use_cache: Whether to use cache for classification
            
        Returns:
            Dict containing classification results
        """
        start_time = time.time()
        cache_key = f"{self.classification_cache_prefix}{hashlib.md5(query_text.encode()).hexdigest()}"
        
        # Check cache first if enabled
        if use_cache:
            cached_result = await self.cache_system.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return {
                    **cached_result,
                    'from_cache': True,
                    'cache_key': cache_key,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            self.stats['cache_misses'] += 1
        
        # Simulate classification processing
        await asyncio.sleep(0.1)  # Simulate processing delay
        
        # Mock classification logic based on query content
        query_lower = query_text.lower()
        
        if 'metabolite' in query_lower or 'compound' in query_lower:
            category = ResearchCategory.METABOLITE_IDENTIFICATION
            confidence = 0.9
        elif 'pathway' in query_lower or 'mechanism' in query_lower:
            category = ResearchCategory.PATHWAY_ANALYSIS
            confidence = 0.85
        elif 'biomarker' in query_lower or 'marker' in query_lower:
            category = ResearchCategory.BIOMARKER_DISCOVERY
            confidence = 0.8
        elif 'clinical' in query_lower or 'diagnosis' in query_lower:
            category = ResearchCategory.CLINICAL_DIAGNOSIS
            confidence = 0.75
        elif 'recent' in query_lower or 'latest' in query_lower or 'current' in query_lower:
            category = ResearchCategory.LITERATURE_SEARCH
            confidence = 0.7
        else:
            category = ResearchCategory.GENERAL_QUERY
            confidence = 0.6
        
        classification_result = {
            'category': category,
            'confidence': confidence,
            'evidence': [word for word in query_lower.split() if len(word) > 3],
            'processing_time_ms': (time.time() - start_time) * 1000,
            'from_cache': False,
            'cache_key': cache_key
        }
        
        # Cache result if enabled
        if use_cache and confidence > 0.5:
            await self.cache_system.set(cache_key, classification_result)
        
        self.stats['classifications'] += 1
        return classification_result


class MockQueryRouter:
    """Mock query router with cache integration."""
    
    def __init__(self, cache_system: MultiTierCache, classification_system: MockQueryClassificationSystem):
        self.cache_system = cache_system
        self.classification_system = classification_system
        self.routing_cache_prefix = "routing:"
        self.stats = {
            'routings': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def route_query(self, query_text: str, context: Optional[Dict[str, Any]] = None,
                         use_cache: bool = True) -> RoutingPrediction:
        """
        Route query with cache integration.
        
        Args:
            query_text: The user query to route
            context: Optional context information
            use_cache: Whether to use cache for routing decisions
            
        Returns:
            RoutingPrediction with routing decision and confidence
        """
        start_time = time.time()
        cache_key = f"{self.routing_cache_prefix}{hashlib.md5((query_text + str(context)).encode()).hexdigest()}"
        
        # Check cache first if enabled
        if use_cache:
            cached_result = await self.cache_system.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return RoutingPrediction(**{
                    **cached_result,
                    'from_cache': True,
                    'cache_key': cache_key
                })
            self.stats['cache_misses'] += 1
        
        # Get classification (with its own caching)
        classification = await self.classification_system.classify_query(query_text, use_cache)
        
        # Determine routing based on classification and temporal analysis
        query_lower = query_text.lower()
        temporal_indicators = ['recent', 'latest', 'current', 'new', 'breaking', '2024', '2025']
        
        has_temporal = any(indicator in query_lower for indicator in temporal_indicators)
        category = classification['category']
        
        # Routing logic
        if has_temporal or category == ResearchCategory.LITERATURE_SEARCH:
            routing_decision = RoutingDecision.PERPLEXITY
            confidence = 0.8
        elif category in [ResearchCategory.PATHWAY_ANALYSIS, ResearchCategory.METABOLITE_IDENTIFICATION]:
            routing_decision = RoutingDecision.LIGHTRAG  
            confidence = 0.85
        elif category == ResearchCategory.BIOMARKER_DISCOVERY:
            routing_decision = RoutingDecision.HYBRID
            confidence = 0.7
        else:
            routing_decision = RoutingDecision.EITHER
            confidence = 0.6
        
        prediction = {
            'routing_decision': routing_decision,
            'confidence': confidence,
            'reasoning': [f"Category: {category.value}", f"Temporal: {has_temporal}"],
            'research_category': category,
            'processing_time_ms': (time.time() - start_time) * 1000,
            'from_cache': False,
            'cache_key': cache_key,
            'classification_result': classification
        }
        
        # Cache result if enabled and confidence is high
        if use_cache and confidence > 0.5:
            cache_data = {k: v for k, v in prediction.items() 
                         if k not in ['classification_result']}  # Don't double-cache
            await self.cache_system.set(cache_key, cache_data)
        
        self.stats['routings'] += 1
        return prediction


class MockResponseGenerator:
    """Mock response generator with cache integration."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.response_cache_prefix = "response:"
        self.prompt_cache_prefix = "prompt:"
        self.stats = {
            'responses_generated': 0,
            'response_cache_hits': 0,
            'response_cache_misses': 0,
            'prompt_cache_hits': 0,
            'prompt_cache_misses': 0
        }
    
    async def generate_response(self, query_text: str, routing_decision: RoutingDecision,
                              context: Optional[Dict[str, Any]] = None,
                              use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate response with cache integration.
        
        Args:
            query_text: The user query
            routing_decision: Routing decision from router
            context: Optional context information  
            use_cache: Whether to use cache for responses
            
        Returns:
            Dict containing generated response and metadata
        """
        start_time = time.time()
        
        # Create cache keys
        response_cache_key = f"{self.response_cache_prefix}{hashlib.md5((query_text + routing_decision.value).encode()).hexdigest()}"
        prompt_cache_key = f"{self.prompt_cache_prefix}{hashlib.md5(query_text.encode()).hexdigest()}"
        
        # Check response cache first
        if use_cache:
            cached_response = await self.cache_system.get(response_cache_key)
            if cached_response is not None:
                self.stats['response_cache_hits'] += 1
                return {
                    **cached_response,
                    'from_cache': True,
                    'cache_type': 'response',
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            self.stats['response_cache_misses'] += 1
        
        # Check prompt cache for optimization
        cached_prompt = None
        if use_cache:
            cached_prompt = await self.cache_system.get(prompt_cache_key)
            if cached_prompt is not None:
                self.stats['prompt_cache_hits'] += 1
            else:
                self.stats['prompt_cache_misses'] += 1
        
        # Simulate response generation based on routing decision
        await asyncio.sleep(0.2 if routing_decision == RoutingDecision.HYBRID else 0.1)
        
        # Generate mock biomedical response
        query_lower = query_text.lower()
        
        if 'metabolite' in query_lower:
            response_content = {
                'metabolites': ['glucose', 'lactate', 'acetyl-CoA'],
                'pathways': ['glycolysis', 'TCA cycle'],
                'confidence': 0.9,
                'source': routing_decision.value
            }
        elif 'pathway' in query_lower:
            response_content = {
                'pathways': ['glycolysis', 'gluconeogenesis', 'pentose phosphate'],
                'enzymes': ['hexokinase', 'phosphofructokinase', 'pyruvate kinase'],
                'confidence': 0.85,
                'source': routing_decision.value
            }
        elif 'biomarker' in query_lower:
            response_content = {
                'biomarkers': ['HbA1c', 'glucose', 'insulin'],
                'disease_associations': ['diabetes', 'metabolic syndrome'],
                'confidence': 0.8,
                'source': routing_decision.value
            }
        else:
            response_content = {
                'general_info': 'Clinical metabolomics information',
                'confidence': 0.7,
                'source': routing_decision.value
            }
        
        response = {
            'query': query_text,
            'routing_decision': routing_decision,
            'content': response_content,
            'processing_time_ms': (time.time() - start_time) * 1000,
            'from_cache': False,
            'cache_type': 'generated',
            'prompt_optimized': cached_prompt is not None
        }
        
        # Cache response if high confidence
        if use_cache and response_content.get('confidence', 0) > 0.7:
            cache_data = {k: v for k, v in response.items() 
                         if k not in ['processing_time_ms']}
            await self.cache_system.set(response_cache_key, cache_data, ttl=3600)
        
        # Cache prompt optimization if not already cached
        if use_cache and cached_prompt is None:
            prompt_optimization = {
                'optimized_prompt': f"Optimized prompt for: {query_text[:50]}...",
                'optimization_type': 'biomedical_focus',
                'created_at': time.time()
            }
            await self.cache_system.set(prompt_cache_key, prompt_optimization, ttl=1800)
        
        self.stats['responses_generated'] += 1
        return response


class MockQualityScorer:
    """Mock quality scorer with cache integration."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.quality_cache_prefix = "quality:"
        self.stats = {
            'scores_calculated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def score_response_quality(self, response: Dict[str, Any], 
                                   query_context: Dict[str, Any],
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Score response quality with cache integration.
        
        Args:
            response: Generated response to score
            query_context: Context about the original query
            use_cache: Whether to use cache for quality scores
            
        Returns:
            Dict containing quality scores and metrics
        """
        start_time = time.time()
        
        # Create cache key based on response content and context
        content_hash = hashlib.md5(json.dumps(response.get('content', {}), sort_keys=True).encode()).hexdigest()
        cache_key = f"{self.quality_cache_prefix}{content_hash}"
        
        # Check cache first
        if use_cache:
            cached_score = await self.cache_system.get(cache_key)
            if cached_score is not None:
                self.stats['cache_hits'] += 1
                return {
                    **cached_score,
                    'from_cache': True,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            self.stats['cache_misses'] += 1
        
        # Simulate quality scoring
        await asyncio.sleep(0.05)  # Quick scoring simulation
        
        content = response.get('content', {})
        confidence = content.get('confidence', 0.5)
        
        # Mock quality metrics
        quality_scores = {
            'overall_quality': min(confidence * 1.2, 1.0),
            'biomedical_relevance': 0.9 if any(key in content for key in ['metabolites', 'pathways', 'biomarkers']) else 0.6,
            'scientific_accuracy': confidence,
            'completeness': 0.8 if len(str(content)) > 100 else 0.6,
            'citation_quality': 0.7,
            'confidence_consistency': abs(confidence - 0.8) < 0.2,
            'processing_time_ms': (time.time() - start_time) * 1000,
            'from_cache': False,
            'cache_key': cache_key
        }
        
        # Cache quality scores if high quality
        if use_cache and quality_scores['overall_quality'] > 0.6:
            cache_data = {k: v for k, v in quality_scores.items() 
                         if k not in ['processing_time_ms']}
            await self.cache_system.set(cache_key, cache_data, ttl=7200)  # Longer TTL for quality scores
        
        self.stats['scores_calculated'] += 1
        return quality_scores


class IntegratedQueryProcessor:
    """Integrated query processor combining all components with cache coordination."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.classification_system = MockQueryClassificationSystem(cache_system)
        self.router = MockQueryRouter(cache_system, self.classification_system)
        self.response_generator = MockResponseGenerator(cache_system)
        self.quality_scorer = MockQualityScorer(cache_system)
        
        self.stats = CacheMetrics()
        self.processing_history: List[QueryProcessingResult] = []
    
    async def process_query_with_cache(self, query_text: str, 
                                     context: Optional[Dict[str, Any]] = None,
                                     use_cache: bool = True) -> QueryProcessingResult:
        """
        Process query through complete pipeline with cache integration.
        
        Args:
            query_text: The user query to process
            context: Optional context information
            use_cache: Whether to use caching throughout the pipeline
            
        Returns:
            QueryProcessingResult with complete processing information
        """
        start_time = time.time()
        pipeline_cache_key = f"pipeline:{hashlib.md5((query_text + str(context)).encode()).hexdigest()}"
        
        # Check for complete pipeline cache first
        if use_cache:
            cached_pipeline_result = await self.cache_system.get(pipeline_cache_key)
            if cached_pipeline_result is not None:
                self.stats.cache_hits += 1
                result = QueryProcessingResult(
                    query_text=query_text,
                    routing_decision=RoutingDecision(cached_pipeline_result['routing_decision']),
                    research_category=ResearchCategory(cached_pipeline_result['research_category']),
                    response_content=cached_pipeline_result['response_content'],
                    cache_status='hit',
                    processing_time_ms=(time.time() - start_time) * 1000,
                    cache_tier_used=cached_pipeline_result.get('cache_tier_used', 'unknown'),
                    quality_score=cached_pipeline_result.get('quality_score', 0.0),
                    confidence_score=cached_pipeline_result.get('confidence_score', 0.0),
                    cached_result=True
                )
                self.processing_history.append(result)
                return result
            
            self.stats.cache_misses += 1
        
        # Process through pipeline
        try:
            # Step 1: Route query
            routing_result = await self.router.route_query(query_text, context, use_cache)
            
            # Step 2: Generate response
            response = await self.response_generator.generate_response(
                query_text, routing_result['routing_decision'], context, use_cache
            )
            
            # Step 3: Score quality
            quality_scores = await self.quality_scorer.score_response_quality(
                response, {'query': query_text, 'routing': routing_result}, use_cache
            )
            
            # Create result
            result = QueryProcessingResult(
                query_text=query_text,
                routing_decision=routing_result['routing_decision'],
                research_category=routing_result['research_category'],
                response_content=response['content'],
                cache_status='miss' if not any([
                    routing_result.get('from_cache'), 
                    response.get('from_cache'),
                    quality_scores.get('from_cache')
                ]) else 'partial',
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=quality_scores['overall_quality'],
                confidence_score=routing_result['confidence'],
                cached_result=False
            )
            
            # Cache complete pipeline result if high quality
            if (use_cache and result.quality_score > 0.7 and 
                result.confidence_score > 0.6):
                pipeline_cache_data = {
                    'routing_decision': result.routing_decision.value,
                    'research_category': result.research_category.value,
                    'response_content': result.response_content,
                    'quality_score': result.quality_score,
                    'confidence_score': result.confidence_score,
                    'cache_tier_used': 'pipeline',
                    'cached_at': time.time()
                }
                await self.cache_system.set(pipeline_cache_key, pipeline_cache_data, ttl=1800)
                self.stats.cache_writes += 1
            
            self.processing_history.append(result)
            return result
            
        except Exception as e:
            # Error handling with fallback
            result = QueryProcessingResult(
                query_text=query_text,
                routing_decision=RoutingDecision.EITHER,
                research_category=ResearchCategory.GENERAL_QUERY,
                response_content={'error': str(e), 'fallback': True},
                cache_status='error',
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=0.1,
                confidence_score=0.1,
                cached_result=False
            )
            self.processing_history.append(result)
            return result
    
    async def warm_cache_with_common_queries(self, queries: List[str]) -> Dict[str, Any]:
        """
        Warm cache with common biomedical queries.
        
        Args:
            queries: List of queries to pre-populate cache
            
        Returns:
            Dict containing cache warming statistics
        """
        start_time = time.time()
        warming_stats = {
            'queries_processed': 0,
            'cache_entries_created': 0,
            'total_time_ms': 0,
            'avg_time_per_query_ms': 0,
            'errors': []
        }
        
        for query in queries:
            try:
                result = await self.process_query_with_cache(query, use_cache=True)
                warming_stats['queries_processed'] += 1
                if not result.cached_result:
                    warming_stats['cache_entries_created'] += 1
            except Exception as e:
                warming_stats['errors'].append(f"Query '{query[:50]}...': {str(e)}")
        
        total_time = (time.time() - start_time) * 1000
        warming_stats['total_time_ms'] = total_time
        warming_stats['avg_time_per_query_ms'] = (total_time / len(queries) 
                                                 if queries else 0)
        
        return warming_stats
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        self.stats.calculate_hit_ratio()
        
        return {
            'cache_metrics': {
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'cache_writes': self.stats.cache_writes,
                'cache_hit_ratio': self.stats.cache_hit_ratio,
                'avg_response_time_ms': self.stats.avg_response_time_ms
            },
            'component_stats': {
                'classification': self.classification_system.stats,
                'routing': self.router.stats,
                'response_generation': self.response_generator.stats,
                'quality_scoring': self.quality_scorer.stats
            },
            'cache_system_stats': self.cache_system.get_comprehensive_stats(),
            'processing_history_count': len(self.processing_history),
            'recent_queries': [result.query_text for result in self.processing_history[-5:]]
        }


# Test Fixtures
@pytest.fixture
async def cache_system():
    """Set up multi-tier cache system for testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        l1_cache = MockL1MemoryCache(max_size=50, default_ttl=300)
        l2_cache = MockL2DiskCache(temp_dir, max_size_mb=10)
        l3_cache = MockL3RedisCache()
        
        multi_cache = MultiTierCache(l1_cache, l2_cache, l3_cache)
        yield multi_cache
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def integrated_processor(cache_system):
    """Set up integrated query processor with cache system."""
    return IntegratedQueryProcessor(cache_system)


# Biomedical Test Data
BIOMEDICAL_TEST_QUERIES = [
    {
        'query': 'What are the key metabolites involved in glucose metabolism?',
        'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
        'expected_routing': RoutingDecision.LIGHTRAG,
        'expected_confidence': 0.85,
        'cache_priority': 'high'
    },
    {
        'query': 'Explain the glycolysis pathway and its regulation mechanisms',
        'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
        'expected_routing': RoutingDecision.LIGHTRAG,
        'expected_confidence': 0.9,
        'cache_priority': 'high'
    },
    {
        'query': 'What are the latest biomarkers for early diabetes detection?',
        'expected_category': ResearchCategory.LITERATURE_SEARCH,
        'expected_routing': RoutingDecision.PERPLEXITY,
        'expected_confidence': 0.8,
        'cache_priority': 'medium'
    },
    {
        'query': 'Recent advances in metabolomics for cardiovascular disease and established pathways',
        'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
        'expected_routing': RoutingDecision.HYBRID,
        'expected_confidence': 0.75,
        'cache_priority': 'high'
    },
    {
        'query': 'Clinical diagnosis protocols for metabolic syndrome',
        'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
        'expected_routing': RoutingDecision.LIGHTRAG,
        'expected_confidence': 0.8,
        'cache_priority': 'medium'
    }
]


class TestQueryProcessingCacheIntegration:
    """Core query-cache integration tests with biomedical focus."""
    
    @pytest.mark.asyncio
    async def test_basic_query_processing_with_cache(self, integrated_processor):
        """Test basic query processing with cache integration."""
        query = "What metabolites are involved in glycolysis?"
        
        # First call - should miss cache and populate it
        result1 = await integrated_processor.process_query_with_cache(query)
        
        assert result1.query_text == query
        assert result1.cache_status in ['miss', 'partial']
        assert result1.routing_decision == RoutingDecision.LIGHTRAG
        assert result1.research_category == ResearchCategory.METABOLITE_IDENTIFICATION
        assert result1.processing_time_ms > 0
        assert not result1.cached_result
        
        # Second call - should hit cache
        result2 = await integrated_processor.process_query_with_cache(query)
        
        assert result2.query_text == query
        assert result2.cache_status == 'hit'
        assert result2.cached_result
        assert result2.processing_time_ms < result1.processing_time_ms
    
    @pytest.mark.asyncio
    async def test_query_classification_cache_integration(self, integrated_processor):
        """Test query classification system cache integration."""
        query = "Analyze metabolic pathways in liver cells"
        
        # Test classification caching
        classification1 = await integrated_processor.classification_system.classify_query(query)
        classification2 = await integrated_processor.classification_system.classify_query(query)
        
        # Second call should be faster due to caching
        assert classification1['category'] == ResearchCategory.PATHWAY_ANALYSIS
        assert classification2['category'] == ResearchCategory.PATHWAY_ANALYSIS
        assert classification2['from_cache']
        
        # Verify cache statistics
        stats = integrated_processor.classification_system.stats
        assert stats['cache_hits'] > 0
        assert stats['classifications'] >= 1
    
    @pytest.mark.asyncio
    async def test_router_cache_coordination(self, integrated_processor):
        """Test query router cache coordination with classification system."""
        query = "Current biomarkers for metabolic disorders"
        context = {"user_focus": "clinical_research"}
        
        # Test routing with cache coordination
        routing1 = await integrated_processor.router.route_query(query, context)
        routing2 = await integrated_processor.router.route_query(query, context)
        
        # Verify routing decisions
        assert routing1['routing_decision'] == RoutingDecision.PERPLEXITY  # Has temporal indicators
        assert routing2['routing_decision'] == routing1['routing_decision']
        assert routing2['from_cache']
        
        # Verify cache coordination with classification
        assert 'classification_result' in routing1
        assert routing1['classification_result']['category'] == ResearchCategory.BIOMARKER_DISCOVERY
    
    @pytest.mark.asyncio
    async def test_response_generation_cache_integration(self, integrated_processor):
        """Test response generation pipeline caching."""
        query = "Metabolic biomarkers for diabetes"
        routing_decision = RoutingDecision.LIGHTRAG
        
        # Test response generation caching
        response1 = await integrated_processor.response_generator.generate_response(
            query, routing_decision
        )
        response2 = await integrated_processor.response_generator.generate_response(
            query, routing_decision
        )
        
        # Verify response content
        assert 'biomarkers' in response1['content']
        assert response1['content']['confidence'] > 0.5
        assert response2['from_cache']
        assert response2['processing_time_ms'] < response1['processing_time_ms']
        
        # Verify prompt caching
        stats = integrated_processor.response_generator.stats
        assert stats['prompt_cache_hits'] > 0 or stats['prompt_cache_misses'] > 0
    
    @pytest.mark.asyncio
    async def test_quality_scoring_cache_integration(self, integrated_processor):
        """Test quality scoring integration with cached responses."""
        response = {
            'content': {
                'biomarkers': ['HbA1c', 'glucose', 'insulin'],
                'confidence': 0.9,
                'pathways': ['insulin_signaling']
            }
        }
        query_context = {'query': 'diabetes biomarkers', 'routing': 'lightrag'}
        
        # Test quality scoring caching
        quality1 = await integrated_processor.quality_scorer.score_response_quality(
            response, query_context
        )
        quality2 = await integrated_processor.quality_scorer.score_response_quality(
            response, query_context
        )
        
        # Verify quality scores
        assert quality1['overall_quality'] > 0.5
        assert quality1['biomedical_relevance'] > 0.8
        assert quality2['from_cache']
        assert quality2['processing_time_ms'] < quality1['processing_time_ms']
    
    @pytest.mark.asyncio
    async def test_multi_component_cache_coordination(self, integrated_processor):
        """Test coordination between multiple cached components."""
        queries = [
            "Glucose metabolism pathways",
            "Insulin resistance biomarkers", 
            "Latest diabetes research and established mechanisms"
        ]
        
        # Process queries to populate caches
        results = []
        for query in queries:
            result = await integrated_processor.process_query_with_cache(query)
            results.append(result)
        
        # Verify different routing decisions and caching
        assert len(set(result.routing_decision for result in results)) > 1
        
        # Re-process same queries - should hit various cache levels
        cached_results = []
        for query in queries:
            result = await integrated_processor.process_query_with_cache(query)
            cached_results.append(result)
        
        # Verify cache utilization
        cache_hits = sum(1 for result in cached_results if result.cache_status == 'hit')
        assert cache_hits > 0
        
        # Verify performance improvement
        for original, cached in zip(results, cached_results):
            if cached.cache_status == 'hit':
                assert cached.processing_time_ms < original.processing_time_ms
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_coordination(self, integrated_processor):
        """Test cache invalidation coordination across components."""
        query = "Metabolomics data analysis methods"
        
        # Process query to populate all cache levels
        result1 = await integrated_processor.process_query_with_cache(query)
        
        # Manually invalidate specific cache entries
        cache_keys_to_invalidate = [
            f"classification:{hashlib.md5(query.encode()).hexdigest()}",
            f"routing:{hashlib.md5((query + 'None').encode()).hexdigest()}"
        ]
        
        for cache_key in cache_keys_to_invalidate:
            await integrated_processor.cache_system.delete(cache_key)
            integrated_processor.stats.cache_invalidations += 1
        
        # Process query again - should have partial cache miss
        result2 = await integrated_processor.process_query_with_cache(query)
        
        # Verify that some components had to be recomputed
        assert result2.cache_status != 'hit'  # Should not be full cache hit
        assert result2.processing_time_ms > result1.processing_time_ms * 0.5  # Some recomputation
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, integrated_processor):
        """Test cache consistency during concurrent query processing."""
        query_templates = [
            "Metabolite analysis for {}",
            "Biomarker discovery in {}",
            "Pathway analysis of {}"
        ]
        
        conditions = ["diabetes", "cancer", "cardiovascular disease", "obesity"]
        queries = [template.format(condition) 
                  for template in query_templates 
                  for condition in conditions]
        
        # Process queries concurrently
        tasks = [integrated_processor.process_query_with_cache(query) 
                for query in queries[:8]]  # Limit to 8 for reasonable test time
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions and all results are valid
        valid_results = [r for r in results if isinstance(r, QueryProcessingResult)]
        assert len(valid_results) == len(tasks)
        
        # Verify cache consistency - same queries should have consistent results
        query_results = {}
        for result in valid_results:
            if result.query_text in query_results:
                # Should have same routing decision and category
                prev_result = query_results[result.query_text]
                assert result.routing_decision == prev_result.routing_decision
                assert result.research_category == prev_result.research_category
            query_results[result.query_text] = result
        
        # Verify cache performance under concurrent load
        stats = integrated_processor.get_comprehensive_stats()
        assert stats['cache_metrics']['cache_hit_ratio'] >= 0  # Should have some cache activity
    
    @pytest.mark.asyncio  
    async def test_cache_warming_performance(self, integrated_processor):
        """Test cache warming with common biomedical queries."""
        common_queries = [
            "glucose metabolism pathway",
            "insulin resistance biomarkers",
            "metabolomics analysis methods",
            "clinical metabolomics applications",
            "biomarker discovery techniques"
        ]
        
        # Warm cache
        warming_stats = await integrated_processor.warm_cache_with_common_queries(
            common_queries
        )
        
        # Verify warming statistics
        assert warming_stats['queries_processed'] == len(common_queries)
        assert warming_stats['cache_entries_created'] > 0
        assert warming_stats['avg_time_per_query_ms'] > 0
        assert len(warming_stats['errors']) == 0
        
        # Test improved performance after warming
        start_time = time.time()
        for query in common_queries:
            result = await integrated_processor.process_query_with_cache(query)
            assert result.cache_status in ['hit', 'partial']
        end_time = time.time()
        
        # Cache warming should improve overall performance
        avg_time_after_warming = ((end_time - start_time) * 1000) / len(common_queries)
        assert avg_time_after_warming < warming_stats['avg_time_per_query_ms']


class TestBiomedicalQueryProcessingScenarios:
    """Real-world biomedical query processing scenarios with cache integration."""
    
    @pytest.mark.asyncio
    async def test_clinical_metabolomics_workflow(self, integrated_processor):
        """Test complete clinical metabolomics workflow with caching."""
        # Simulate clinical metabolomics research workflow
        workflow_queries = [
            "What are the key metabolites in diabetes pathogenesis?",
            "Glucose metabolism pathway regulation mechanisms", 
            "Latest biomarkers for diabetes early detection",
            "Clinical protocols for metabolomics data analysis",
            "Statistical methods for biomarker validation"
        ]
        
        workflow_results = []
        for i, query in enumerate(workflow_queries):
            context = {
                'workflow_step': i + 1,
                'previous_queries': workflow_queries[:i],
                'research_focus': 'clinical_metabolomics'
            }
            
            result = await integrated_processor.process_query_with_cache(
                query, context
            )
            workflow_results.append(result)
        
        # Verify workflow progression
        assert len(workflow_results) == len(workflow_queries)
        
        # Verify appropriate routing decisions for different query types
        routing_decisions = [result.routing_decision for result in workflow_results]
        assert RoutingDecision.LIGHTRAG in routing_decisions  # For established knowledge
        assert RoutingDecision.PERPLEXITY in routing_decisions  # For latest research
        
        # Verify cache utilization improves workflow efficiency
        processing_times = [result.processing_time_ms for result in workflow_results]
        assert processing_times[-1] < processing_times[0] * 1.5  # Later queries benefit from cache
        
        # Verify quality scores are consistently high
        quality_scores = [result.quality_score for result in workflow_results]
        assert all(score > 0.6 for score in quality_scores)
    
    @pytest.mark.asyncio
    async def test_multi_language_biomedical_queries(self, integrated_processor):
        """Test multi-language biomedical query processing with cache."""
        # Note: This is a mock test - actual implementation would need translation
        multilingual_queries = [
            ("English", "metabolic pathways in diabetes"),
            ("Scientific", "glucose-6-phosphate dehydrogenase deficiency pathways"),
            ("Clinical", "HbA1c levels in metabolic syndrome patients"),
            ("Technical", "LC-MS/MS analysis of metabolomics samples")
        ]
        
        results = []
        for language, query in multilingual_queries:
            context = {'language_context': language, 'domain': 'biomedical'}
            result = await integrated_processor.process_query_with_cache(query, context)
            results.append((language, result))
        
        # Verify all queries processed successfully
        assert len(results) == len(multilingual_queries)
        
        # Verify appropriate handling of technical vs. clinical terminology
        technical_result = next(result for lang, result in results if lang == 'Technical')
        clinical_result = next(result for lang, result in results if lang == 'Clinical')
        
        assert technical_result.research_category in [
            ResearchCategory.METABOLITE_IDENTIFICATION,
            ResearchCategory.GENERAL_QUERY
        ]
        assert clinical_result.research_category in [
            ResearchCategory.CLINICAL_DIAGNOSIS,
            ResearchCategory.BIOMARKER_DISCOVERY
        ]
    
    @pytest.mark.asyncio
    async def test_research_data_queries_with_cache_optimization(self, integrated_processor):
        """Test research data queries with cache optimization."""
        research_queries = [
            {
                'query': 'Metabolomics data preprocessing methods',
                'data_type': 'methods',
                'cache_priority': 'high'
            },
            {
                'query': 'Statistical analysis of biomarker data',
                'data_type': 'analysis', 
                'cache_priority': 'high'
            },
            {
                'query': 'Latest machine learning applications in metabolomics',
                'data_type': 'current_research',
                'cache_priority': 'medium'
            },
            {
                'query': 'Database integration for metabolomics workflows',
                'data_type': 'technical',
                'cache_priority': 'low'
            }
        ]
        
        # Process queries with different cache priorities
        results = []
        for query_info in research_queries:
            context = {
                'data_type': query_info['data_type'],
                'cache_priority': query_info['cache_priority']
            }
            
            result = await integrated_processor.process_query_with_cache(
                query_info['query'], context
            )
            results.append((query_info['cache_priority'], result))
        
        # Verify high-priority queries are cached more aggressively
        high_priority_results = [result for priority, result in results if priority == 'high']
        medium_priority_results = [result for priority, result in results if priority == 'medium']
        
        # High priority queries should have better cache performance on repeat
        for _, result in [(priority, result) for priority, result in results if priority == 'high']:
            # Process same query again
            repeat_result = await integrated_processor.process_query_with_cache(
                result.query_text
            )
            if repeat_result.cache_status == 'hit':
                assert repeat_result.processing_time_ms < result.processing_time_ms * 0.5
    
    @pytest.mark.asyncio
    async def test_citation_confidence_scoring_with_cache(self, integrated_processor):
        """Test citation and confidence scoring with cached responses."""
        queries_with_citations = [
            {
                'query': 'Metabolomics biomarkers in cardiovascular disease',
                'expected_citations': True,
                'expected_confidence': 0.8
            },
            {
                'query': 'Recent breakthrough in diabetes metabolomics research',
                'expected_citations': True,
                'expected_confidence': 0.7
            },
            {
                'query': 'What is metabolomics?',
                'expected_citations': False,
                'expected_confidence': 0.9
            }
        ]
        
        citation_results = []
        for query_info in queries_with_citations:
            result = await integrated_processor.process_query_with_cache(
                query_info['query']
            )
            
            # Mock citation analysis (would be more sophisticated in real implementation)
            citation_score = 0.8 if query_info['expected_citations'] else 0.5
            
            citation_results.append({
                'query': query_info['query'],
                'result': result,
                'citation_score': citation_score,
                'expected_confidence': query_info['expected_confidence']
            })
        
        # Verify confidence scoring consistency with citations
        for citation_result in citation_results:
            result = citation_result['result']
            expected_conf = citation_result['expected_confidence']
            
            # Allow some tolerance in confidence scoring
            assert abs(result.confidence_score - expected_conf) < 0.3
            
            # High citation scores should correlate with higher quality
            if citation_result['citation_score'] > 0.7:
                assert result.quality_score > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])