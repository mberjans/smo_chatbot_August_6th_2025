"""
Performance-Critical Cache Integration Tests for Clinical Metabolomics Oracle

This module provides comprehensive performance-focused integration tests for the 
multi-tier caching system integrated with query processing pipeline, with emphasis
on performance optimization, scalability, and efficiency under realistic load conditions.

Performance Test Areas:
- Cache warming performance and optimization strategies
- Predictive caching accuracy and efficiency under load
- Cache hit optimization in high-frequency query processing  
- Performance monitoring with real-time cache statistics
- Concurrent access patterns and cache consistency
- Memory and disk usage optimization
- Network latency impact on distributed caching
- Cache eviction and promotion performance

Load Testing Scenarios:
- High-frequency biomedical query bursts (100+ queries/second)
- Large-scale clinical metabolomics research simulations
- Multi-tenant research environment simulation
- Peak usage pattern simulation (conference, research deadline periods)
- Long-running research workflow performance
- Cache system resilience under memory pressure

Performance Targets:
- Cache hit ratio > 85% for common biomedical query patterns
- Query processing latency < 100ms for cached responses
- Cache warming completion < 30s for 1000 common queries
- Memory usage < 500MB for L1 cache under normal load
- Disk cache growth rate < 100MB/hour under continuous use
- Cache invalidation propagation < 50ms across all tiers
- Concurrent query processing > 50 queries/second sustained

Optimization Validation:
- Cache size optimization for biomedical query patterns
- TTL optimization for different types of biomedical content
- Cache promotion/demotion strategy effectiveness
- Predictive caching accuracy and resource efficiency
- Cache compression and storage optimization
- Network bandwidth optimization for distributed cache

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
import tempfile
import shutil
import json
import statistics
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
import hashlib
import random
import logging
from contextlib import asynccontextmanager

# Import system components (mock imports for test environment)
try:
    from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
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
        DATA_PREPROCESSING = "data_preprocessing"
        GENERAL_QUERY = "general_query"

# Import cache system and previous test components
from tests.unit.test_multi_tier_cache import MultiTierCache, MockL1MemoryCache, MockL2DiskCache, MockL3RedisCache


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for cache integration testing."""
    
    # Timing metrics
    response_times: List[float] = field(default_factory=list)
    cache_lookup_times: List[float] = field(default_factory=list)
    cache_write_times: List[float] = field(default_factory=list)
    query_processing_times: List[float] = field(default_factory=list)
    
    # Throughput metrics
    queries_per_second: float = 0.0
    cache_operations_per_second: float = 0.0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Cache efficiency metrics
    cache_hit_ratio: float = 0.0
    cache_miss_ratio: float = 0.0
    cache_write_ratio: float = 0.0
    predictive_hit_accuracy: float = 0.0
    
    # Resource metrics
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    peak_disk_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    
    # Quality metrics  
    avg_confidence_score: float = 0.0
    avg_quality_score: float = 0.0
    error_rate: float = 0.0
    
    def add_response_time(self, response_time: float):
        """Add response time measurement."""
        self.response_times.append(response_time)
        
    def calculate_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        return {
            'p50': statistics.median(sorted_times),
            'p90': statistics.quantiles(sorted_times, n=10)[8] if len(sorted_times) >= 10 else max(sorted_times),
            'p95': statistics.quantiles(sorted_times, n=20)[18] if len(sorted_times) >= 20 else max(sorted_times),
            'p99': statistics.quantiles(sorted_times, n=100)[98] if len(sorted_times) >= 100 else max(sorted_times),
            'min': min(sorted_times),
            'max': max(sorted_times),
            'avg': statistics.mean(sorted_times)
        }
        
    def calculate_throughput_metrics(self, duration_seconds: float):
        """Calculate throughput metrics over given duration."""
        if duration_seconds > 0:
            total_queries = self.successful_queries + self.failed_queries
            self.queries_per_second = total_queries / duration_seconds
            
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary report."""
        percentiles = self.calculate_percentiles()
        
        return {
            'timing_performance': {
                'response_times': percentiles,
                'avg_cache_lookup_ms': statistics.mean(self.cache_lookup_times) if self.cache_lookup_times else 0,
                'avg_cache_write_ms': statistics.mean(self.cache_write_times) if self.cache_write_times else 0,
                'avg_query_processing_ms': statistics.mean(self.query_processing_times) if self.query_processing_times else 0
            },
            'throughput_performance': {
                'queries_per_second': self.queries_per_second,
                'cache_operations_per_second': self.cache_operations_per_second,
                'successful_queries': self.successful_queries,
                'failed_queries': self.failed_queries,
                'success_rate': self.successful_queries / max(self.successful_queries + self.failed_queries, 1)
            },
            'cache_efficiency': {
                'cache_hit_ratio': self.cache_hit_ratio,
                'cache_miss_ratio': self.cache_miss_ratio,
                'predictive_hit_accuracy': self.predictive_hit_accuracy
            },
            'resource_utilization': {
                'peak_memory_mb': self.peak_memory_usage_mb,
                'avg_memory_mb': self.avg_memory_usage_mb,
                'peak_disk_mb': self.peak_disk_usage_mb,
                'cpu_percent': self.cpu_utilization_percent
            },
            'quality_metrics': {
                'avg_confidence': self.avg_confidence_score,
                'avg_quality': self.avg_quality_score,
                'error_rate': self.error_rate
            }
        }


@dataclass
class LoadTestConfiguration:
    """Configuration for load testing scenarios."""
    
    test_name: str
    duration_seconds: int
    concurrent_users: int
    queries_per_user: int
    ramp_up_seconds: int = 30
    cache_warming_enabled: bool = True
    predictive_caching_enabled: bool = True
    target_qps: float = 50.0
    target_cache_hit_ratio: float = 0.85
    target_p95_response_time_ms: float = 500.0
    memory_limit_mb: float = 500.0
    
    def get_total_queries(self) -> int:
        """Calculate total number of queries for this test."""
        return self.concurrent_users * self.queries_per_user


class PerformanceMonitor:
    """Real-time performance monitoring for cache integration tests."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_samples = []
        
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                # Memory monitoring
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.resource_samples.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': process.cpu_percent()
                })
                
                # Update metrics
                if memory_mb > self.metrics.peak_memory_usage_mb:
                    self.metrics.peak_memory_usage_mb = memory_mb
                    
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                pass  # Ignore monitoring errors
    
    def record_operation(self, operation_type: str, duration_ms: float, success: bool):
        """Record performance data for cache operation."""
        if operation_type == 'response':
            self.metrics.add_response_time(duration_ms)
            if success:
                self.metrics.successful_queries += 1
            else:
                self.metrics.failed_queries += 1
        elif operation_type == 'cache_lookup':
            self.metrics.cache_lookup_times.append(duration_ms)
        elif operation_type == 'cache_write':
            self.metrics.cache_write_times.append(duration_ms)
        elif operation_type == 'query_processing':
            self.metrics.query_processing_times.append(duration_ms)
    
    def finalize_metrics(self, test_duration_seconds: float):
        """Finalize metrics calculation after test completion."""
        self.metrics.calculate_throughput_metrics(test_duration_seconds)
        
        # Calculate average resource usage
        if self.resource_samples:
            memory_values = [sample['memory_mb'] for sample in self.resource_samples]
            cpu_values = [sample['cpu_percent'] for sample in self.resource_samples]
            
            self.metrics.avg_memory_usage_mb = statistics.mean(memory_values)
            self.metrics.cpu_utilization_percent = statistics.mean(cpu_values)


class HighPerformanceCacheIntegrationSystem:
    """High-performance cache integration system for performance testing."""
    
    def __init__(self, cache_system: MultiTierCache, enable_optimizations: bool = True):
        self.cache_system = cache_system
        self.enable_optimizations = enable_optimizations
        
        # Import processors from previous test files
        from .test_query_processing_cache import IntegratedQueryProcessor
        from .test_end_to_end_cache_flow import EndToEndQueryProcessor
        
        self.base_processor = IntegratedQueryProcessor(cache_system)
        self.e2e_processor = EndToEndQueryProcessor(cache_system)
        
        # Performance optimizations
        self.query_batch_processor = QueryBatchProcessor(cache_system)
        self.cache_optimizer = CacheOptimizer(cache_system)
        self.performance_predictor = PerformancePredictor()
        
        # Performance tracking
        self.query_stats = defaultdict(int)
        self.cache_patterns = defaultdict(list)
        self.optimization_history = []
        
    async def process_query_optimized(self, query_text: str,
                                    context: Optional[Dict[str, Any]] = None,
                                    performance_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query with performance optimizations enabled.
        
        Args:
            query_text: The user query to process
            context: Optional context information
            performance_hints: Optional performance optimization hints
            
        Returns:
            Dict containing query result and performance metadata
        """
        start_time = time.time()
        
        # Apply performance hints if provided
        if performance_hints and self.enable_optimizations:
            if performance_hints.get('use_fast_cache_lookup'):
                # Use L1 cache only for extremely fast lookups
                context = context or {}
                context['cache_preference'] = 'L1_only'
            
            if performance_hints.get('batch_related_queries'):
                # Batch process related queries for efficiency
                related_queries = performance_hints.get('related_queries', [])
                if related_queries:
                    return await self._process_query_batch([query_text] + related_queries)
        
        # Standard optimized processing
        result = await self.base_processor.process_query_with_cache(
            query_text, context, use_cache=True
        )
        
        # Performance metadata
        processing_time = (time.time() - start_time) * 1000
        performance_metadata = {
            'processing_time_ms': processing_time,
            'cache_tier_used': getattr(result, 'cache_tier_used', 'unknown'),
            'optimization_applied': self.enable_optimizations,
            'query_complexity_score': len(query_text.split()) * 0.1,
            'cache_efficiency_score': 1.0 if getattr(result, 'cached_result', False) else 0.3
        }
        
        # Update performance tracking
        self.query_stats[query_text[:50]] += 1  # Track query frequency (truncated)
        self.cache_patterns[result.research_category.value].append(processing_time)
        
        return {
            'query_result': result,
            'performance_metadata': performance_metadata
        }
    
    async def _process_query_batch(self, queries: List[str]) -> Dict[str, Any]:
        """Process batch of related queries with shared optimizations."""
        return await self.query_batch_processor.process_batch(queries)
    
    async def warm_cache_optimized(self, warming_queries: List[str],
                                 warming_strategy: str = 'adaptive') -> Dict[str, Any]:
        """
        Perform optimized cache warming based on query patterns.
        
        Args:
            warming_queries: List of queries to use for cache warming
            warming_strategy: Warming strategy ('adaptive', 'aggressive', 'conservative')
            
        Returns:
            Dict containing cache warming performance statistics
        """
        return await self.cache_optimizer.warm_cache_with_strategy(
            warming_queries, warming_strategy
        )
    
    async def optimize_cache_configuration(self, workload_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize cache configuration based on workload profile.
        
        Args:
            workload_profile: Profile of expected workload characteristics
            
        Returns:
            Dict containing optimization recommendations and performance impact
        """
        return await self.cache_optimizer.optimize_configuration(workload_profile)
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and optimization recommendations."""
        return {
            'query_frequency_analysis': dict(self.query_stats),
            'cache_pattern_analysis': {
                category: {
                    'avg_time_ms': statistics.mean(times),
                    'query_count': len(times),
                    'p95_time_ms': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times) if times else 0
                }
                for category, times in self.cache_patterns.items()
            },
            'optimization_recommendations': self.performance_predictor.get_recommendations(
                self.query_stats, self.cache_patterns
            )
        }


class QueryBatchProcessor:
    """Batch processor for related queries with shared cache optimizations."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.batch_stats = {'batches_processed': 0, 'queries_batched': 0, 'time_saved_ms': 0}
    
    async def process_batch(self, queries: List[str]) -> Dict[str, Any]:
        """Process batch of queries with shared optimizations."""
        start_time = time.time()
        
        # Group queries by similarity for optimization
        query_groups = self._group_similar_queries(queries)
        
        batch_results = []
        shared_context = {}
        
        for group in query_groups:
            # Process group with shared context
            group_results = []
            for query in group:
                # Use accumulated context from previous queries in group
                result = await self._process_single_with_context(query, shared_context)
                group_results.append(result)
                
                # Update shared context
                if hasattr(result, 'entities'):
                    shared_context['accumulated_entities'] = shared_context.get('accumulated_entities', [])
                    shared_context['accumulated_entities'].extend(result.entities[:3])  # Limit context size
            
            batch_results.extend(group_results)
        
        processing_time = (time.time() - start_time) * 1000
        
        self.batch_stats['batches_processed'] += 1
        self.batch_stats['queries_batched'] += len(queries)
        
        return {
            'batch_results': batch_results,
            'batch_processing_time_ms': processing_time,
            'queries_in_batch': len(queries),
            'optimization_applied': True,
            'estimated_time_saved_ms': max(0, len(queries) * 100 - processing_time)
        }
    
    def _group_similar_queries(self, queries: List[str]) -> List[List[str]]:
        """Group similar queries for batch optimization."""
        groups = []
        
        # Simple similarity grouping based on common keywords
        biomedical_categories = {
            'metabolite': [],
            'pathway': [],
            'biomarker': [],
            'clinical': [],
            'other': []
        }
        
        for query in queries:
            query_lower = query.lower()
            categorized = False
            
            for category in biomedical_categories:
                if category in query_lower:
                    biomedical_categories[category].append(query)
                    categorized = True
                    break
            
            if not categorized:
                biomedical_categories['other'].append(query)
        
        # Convert to groups, filtering empty categories
        for category, category_queries in biomedical_categories.items():
            if category_queries:
                groups.append(category_queries)
        
        return groups if groups else [queries]  # Fallback to single group
    
    async def _process_single_with_context(self, query: str, shared_context: Dict[str, Any]):
        """Process single query with shared context optimization."""
        # Simplified processing with context
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Mock result with shared context utilization
        return type('MockResult', (), {
            'query_text': query,
            'cached_result': False,
            'processing_time_ms': 50,
            'confidence_score': 0.8,
            'entities': shared_context.get('accumulated_entities', [])[:2],  # Use shared entities
            'research_category': type('Category', (), {'value': 'general_query'})()
        })()


class CacheOptimizer:
    """Cache configuration optimizer for performance tuning."""
    
    def __init__(self, cache_system: MultiTierCache):
        self.cache_system = cache_system
        self.optimization_stats = {'optimizations_applied': 0, 'performance_improvements': []}
    
    async def warm_cache_with_strategy(self, warming_queries: List[str],
                                     strategy: str = 'adaptive') -> Dict[str, Any]:
        """
        Warm cache using specified strategy.
        
        Args:
            warming_queries: Queries to use for warming
            strategy: Warming strategy ('adaptive', 'aggressive', 'conservative')
            
        Returns:
            Dict containing warming performance statistics
        """
        start_time = time.time()
        
        warming_stats = {
            'strategy_used': strategy,
            'queries_processed': 0,
            'cache_entries_created': 0,
            'warming_time_ms': 0,
            'estimated_hit_ratio_improvement': 0.0
        }
        
        # Apply strategy-specific parameters
        if strategy == 'aggressive':
            # Process all queries, high parallelism
            batch_size = 10
            ttl_multiplier = 2.0
        elif strategy == 'conservative':
            # Process selectively, low resource usage
            batch_size = 3
            ttl_multiplier = 0.5
            warming_queries = warming_queries[:len(warming_queries)//2]  # Use fewer queries
        else:  # adaptive
            # Balanced approach
            batch_size = 5
            ttl_multiplier = 1.0
        
        # Process warming queries in batches
        for i in range(0, len(warming_queries), batch_size):
            batch = warming_queries[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for query in batch:
                # Simulate cache warming operation
                task = self._warm_single_query(query, ttl_multiplier)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statistics
            successful_warmings = sum(1 for result in batch_results 
                                    if isinstance(result, dict) and result.get('success', False))
            warming_stats['queries_processed'] += len(batch)
            warming_stats['cache_entries_created'] += successful_warmings
        
        warming_stats['warming_time_ms'] = (time.time() - start_time) * 1000
        
        # Estimate hit ratio improvement
        cache_coverage = warming_stats['cache_entries_created'] / max(len(warming_queries), 1)
        warming_stats['estimated_hit_ratio_improvement'] = cache_coverage * 0.6  # Heuristic estimate
        
        self.optimization_stats['optimizations_applied'] += 1
        self.optimization_stats['performance_improvements'].append(warming_stats)
        
        return warming_stats
    
    async def _warm_single_query(self, query: str, ttl_multiplier: float) -> Dict[str, Any]:
        """Warm cache for single query."""
        # Simulate query processing and caching
        await asyncio.sleep(0.02)  # Simulate processing time
        
        # Mock cache warming
        cache_key = f"warm:{hashlib.md5(query.encode()).hexdigest()}"
        cache_data = {
            'query': query,
            'warmed_at': time.time(),
            'ttl_multiplier': ttl_multiplier
        }
        
        # Simulate cache write with TTL
        base_ttl = 1800  # 30 minutes
        adjusted_ttl = int(base_ttl * ttl_multiplier)
        
        await self.cache_system.set(cache_key, cache_data, ttl=adjusted_ttl)
        
        return {'success': True, 'cache_key': cache_key}
    
    async def optimize_configuration(self, workload_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize cache configuration based on workload profile.
        
        Args:
            workload_profile: Expected workload characteristics
            
        Returns:
            Dict containing optimization recommendations
        """
        optimization_recommendations = {
            'current_config': await self._get_current_cache_config(),
            'workload_analysis': self._analyze_workload_profile(workload_profile),
            'recommendations': [],
            'estimated_performance_impact': {}
        }
        
        # Analyze workload profile
        query_rate = workload_profile.get('queries_per_second', 10)
        cache_hit_ratio = workload_profile.get('current_cache_hit_ratio', 0.5)
        memory_budget_mb = workload_profile.get('memory_budget_mb', 100)
        
        # Generate recommendations
        if cache_hit_ratio < 0.8:
            optimization_recommendations['recommendations'].append({
                'type': 'increase_cache_size',
                'description': 'Increase L1 cache size to improve hit ratio',
                'current_value': '50 entries',
                'recommended_value': '100 entries',
                'estimated_improvement': '+15% hit ratio'
            })
        
        if query_rate > 20:
            optimization_recommendations['recommendations'].append({
                'type': 'enable_predictive_caching',
                'description': 'Enable predictive caching for high query rate',
                'estimated_improvement': '+10% performance'
            })
        
        if memory_budget_mb > 200:
            optimization_recommendations['recommendations'].append({
                'type': 'increase_l1_memory',
                'description': 'Allocate more memory to L1 cache',
                'current_value': f'{memory_budget_mb/4}MB',
                'recommended_value': f'{memory_budget_mb/2}MB',
                'estimated_improvement': '+20% L1 hit ratio'
            })
        
        return optimization_recommendations
    
    async def _get_current_cache_config(self) -> Dict[str, Any]:
        """Get current cache configuration."""
        cache_stats = self.cache_system.get_comprehensive_stats()
        
        return {
            'l1_size': cache_stats.get('l1', {}).get('size', 0),
            'l2_size_mb': cache_stats.get('l2', {}).get('total_size_mb', 0),
            'l3_connected': cache_stats.get('l3', {}).get('connected', False),
            'overall_hit_rate': cache_stats.get('overall_hit_rate', 0)
        }
    
    def _analyze_workload_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workload profile for optimization insights."""
        return {
            'workload_type': self._classify_workload_type(profile),
            'resource_intensity': self._assess_resource_intensity(profile),
            'optimization_potential': self._estimate_optimization_potential(profile)
        }
    
    def _classify_workload_type(self, profile: Dict[str, Any]) -> str:
        """Classify workload type based on profile characteristics."""
        query_rate = profile.get('queries_per_second', 10)
        
        if query_rate > 50:
            return 'high_throughput'
        elif query_rate > 20:
            return 'medium_throughput'
        else:
            return 'low_throughput'
    
    def _assess_resource_intensity(self, profile: Dict[str, Any]) -> str:
        """Assess resource intensity of workload."""
        memory_budget = profile.get('memory_budget_mb', 100)
        
        if memory_budget > 500:
            return 'high_resource'
        elif memory_budget > 200:
            return 'medium_resource'
        else:
            return 'low_resource'
    
    def _estimate_optimization_potential(self, profile: Dict[str, Any]) -> float:
        """Estimate potential performance improvement from optimization."""
        current_hit_ratio = profile.get('current_cache_hit_ratio', 0.5)
        
        # Simple heuristic: more improvement potential if hit ratio is low
        return min((0.9 - current_hit_ratio) * 100, 50)  # Max 50% improvement potential


class PerformancePredictor:
    """Performance prediction and recommendation system."""
    
    def __init__(self):
        self.prediction_history = []
        
    def get_recommendations(self, query_stats: Dict[str, int],
                          cache_patterns: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze query frequency patterns
        total_queries = sum(query_stats.values())
        if total_queries > 0:
            # Find most frequent queries
            frequent_queries = sorted(query_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for query_prefix, count in frequent_queries:
                if count > total_queries * 0.1:  # > 10% of all queries
                    recommendations.append({
                        'type': 'cache_warming',
                        'priority': 'high',
                        'description': f'Pre-warm cache for frequent query pattern: {query_prefix}',
                        'estimated_benefit': f'{count} queries would benefit'
                    })
        
        # Analyze cache performance patterns
        for category, times in cache_patterns.items():
            if times:
                avg_time = statistics.mean(times)
                if avg_time > 200:  # Slow category
                    recommendations.append({
                        'type': 'cache_optimization',
                        'priority': 'medium',
                        'description': f'Optimize caching for {category} queries (avg: {avg_time:.1f}ms)',
                        'estimated_benefit': f'Potential 30% reduction in response time'
                    })
        
        return recommendations


# Biomedical Query Generators for Load Testing
class BiomedicalQueryGenerator:
    """Generate realistic biomedical queries for load testing."""
    
    def __init__(self):
        self.metabolite_queries = [
            "What are the key metabolites in glucose metabolism?",
            "How do insulin levels affect metabolic pathways?",
            "Identify biomarkers for diabetes diagnosis",
            "Metabolic profiling of cardiovascular disease",
            "Lipidomics analysis of metabolic syndrome"
        ]
        
        self.pathway_queries = [
            "Explain the glycolysis pathway regulation",
            "How does the TCA cycle connect to fatty acid synthesis?",
            "Metabolic pathway interactions in cancer",
            "Insulin signaling pathway components",
            "Amino acid metabolism pathways"
        ]
        
        self.clinical_queries = [
            "Clinical applications of metabolomics in diagnostics",
            "Biomarker validation protocols for clinical use",
            "Metabolomics-based drug discovery approaches",
            "Personalized medicine using metabolic profiling",
            "Clinical metabolomics workflow optimization"
        ]
        
        self.literature_queries = [
            "Recent advances in metabolomics research",
            "Latest biomarker discoveries in 2024",
            "Current trends in clinical metabolomics",
            "New metabolomics technologies and methods",
            "Breakthrough metabolomics applications"
        ]
        
        self.all_queries = (self.metabolite_queries + self.pathway_queries + 
                           self.clinical_queries + self.literature_queries)
    
    def generate_query_sequence(self, num_queries: int, pattern: str = 'mixed') -> List[str]:
        """Generate sequence of queries for load testing."""
        if pattern == 'metabolite_focused':
            query_pool = self.metabolite_queries * 3 + self.pathway_queries
        elif pattern == 'clinical_focused':
            query_pool = self.clinical_queries * 3 + self.literature_queries
        elif pattern == 'research_focused':
            query_pool = self.literature_queries * 2 + self.pathway_queries * 2
        else:  # mixed
            query_pool = self.all_queries
        
        # Generate queries with some repetition for cache testing
        queries = []
        for i in range(num_queries):
            if i < len(query_pool):
                queries.append(query_pool[i % len(query_pool)])
            else:
                # Add variations for additional queries
                base_query = random.choice(query_pool)
                variation = self._create_query_variation(base_query, i)
                queries.append(variation)
        
        return queries
    
    def _create_query_variation(self, base_query: str, variation_id: int) -> str:
        """Create variation of base query."""
        variations = [
            f"{base_query} in humans",
            f"{base_query} and clinical significance",
            f"How does {base_query.lower()}",
            f"Recent research on {base_query.lower()}",
            f"{base_query} - detailed analysis"
        ]
        
        return variations[variation_id % len(variations)]


# Test Fixtures
@pytest.fixture
async def high_performance_cache_system():
    """Set up high-performance cache system for testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Larger cache sizes for performance testing
        l1_cache = MockL1MemoryCache(max_size=200, default_ttl=1200)
        l2_cache = MockL2DiskCache(temp_dir, max_size_mb=100)
        l3_cache = MockL3RedisCache()
        
        multi_cache = MultiTierCache(l1_cache, l2_cache, l3_cache)
        yield multi_cache
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def performance_system(high_performance_cache_system):
    """Set up high-performance integration system."""
    return HighPerformanceCacheIntegrationSystem(
        high_performance_cache_system, enable_optimizations=True
    )


@pytest.fixture
def query_generator():
    """Set up biomedical query generator."""
    return BiomedicalQueryGenerator()


@pytest.fixture
def performance_monitor():
    """Set up performance monitor."""
    monitor = PerformanceMonitor()
    yield monitor
    monitor.stop_monitoring()


# Performance Test Classes
class TestCachePerformanceOptimization:
    """Tests for cache performance optimization and monitoring."""
    
    @pytest.mark.asyncio
    async def test_cache_warming_performance(self, performance_system, query_generator, 
                                           performance_monitor):
        """Test cache warming performance with different strategies."""
        warming_queries = query_generator.generate_query_sequence(50, 'mixed')
        
        performance_monitor.start_monitoring()
        
        # Test different warming strategies
        strategies = ['conservative', 'adaptive', 'aggressive']
        strategy_results = {}
        
        for strategy in strategies:
            start_time = time.time()
            
            warming_result = await performance_system.warm_cache_optimized(
                warming_queries[:20], strategy  # Limit for test speed
            )
            
            strategy_time = (time.time() - start_time) * 1000
            strategy_results[strategy] = {
                'warming_time_ms': strategy_time,
                'cache_entries_created': warming_result['cache_entries_created'],
                'estimated_improvement': warming_result['estimated_hit_ratio_improvement']
            }
            
            # Clear cache between strategies
            await asyncio.sleep(0.1)
        
        performance_monitor.stop_monitoring()
        
        # Verify performance characteristics
        assert 'conservative' in strategy_results
        assert 'adaptive' in strategy_results
        assert 'aggressive' in strategy_results
        
        # Conservative should be fastest but create fewer entries
        conservative = strategy_results['conservative']
        aggressive = strategy_results['aggressive']
        
        assert conservative['warming_time_ms'] < aggressive['warming_time_ms'] * 1.5
        assert aggressive['cache_entries_created'] >= conservative['cache_entries_created']
        
        # All strategies should complete within reasonable time
        for strategy_result in strategy_results.values():
            assert strategy_result['warming_time_ms'] < 10000  # 10 seconds max
    
    @pytest.mark.asyncio
    async def test_high_frequency_query_processing(self, performance_system, query_generator,
                                                  performance_monitor):
        """Test high-frequency query processing performance."""
        test_queries = query_generator.generate_query_sequence(100, 'metabolite_focused')
        
        performance_monitor.start_monitoring()
        start_time = time.time()
        
        # Process queries with high frequency
        results = []
        for i, query in enumerate(test_queries[:50]):  # Limit for test performance
            result = await performance_system.process_query_optimized(
                query,
                performance_hints={'use_fast_cache_lookup': i > 10}  # Enable fast lookup after warmup
            )
            results.append(result)
            
            performance_monitor.record_operation(
                'response', 
                result['performance_metadata']['processing_time_ms'],
                True
            )
        
        test_duration = time.time() - start_time
        performance_monitor.finalize_metrics(test_duration)
        performance_monitor.stop_monitoring()
        
        # Verify performance targets
        metrics_summary = performance_monitor.metrics.get_summary_report()
        
        # Check throughput
        assert metrics_summary['throughput_performance']['queries_per_second'] > 10
        assert metrics_summary['throughput_performance']['success_rate'] > 0.95
        
        # Check response times
        timing_perf = metrics_summary['timing_performance']['response_times']
        assert timing_perf['avg'] < 300  # Average response time < 300ms
        assert timing_perf['p95'] < 500  # 95th percentile < 500ms
        
        # Check resource usage
        resource_perf = metrics_summary['resource_utilization']
        assert resource_perf['peak_memory_mb'] < 100  # Memory usage reasonable
    
    @pytest.mark.asyncio
    async def test_predictive_caching_accuracy(self, performance_system, query_generator):
        """Test predictive caching accuracy and efficiency."""
        # Generate base queries to establish patterns
        base_queries = query_generator.generate_query_sequence(30, 'clinical_focused')
        
        # Process base queries to train predictive system
        for query in base_queries[:15]:
            await performance_system.process_query_optimized(query)
        
        # Test predictive caching with follow-up queries
        test_query = "What are diabetes biomarkers?"
        follow_up_queries = [
            "How are diabetes biomarkers validated?",
            "Clinical applications of diabetes biomarkers",
            "Statistical analysis of diabetes biomarker data"
        ]
        
        # Process main query
        main_result = await performance_system.process_query_optimized(
            test_query,
            performance_hints={
                'batch_related_queries': True,
                'related_queries': follow_up_queries
            }
        )
        
        # Verify predictive processing was applied
        assert main_result['performance_metadata']['optimization_applied']
        
        # Process follow-up queries and check for cache benefits
        follow_up_times = []
        for follow_up in follow_up_queries:
            start_time = time.time()
            follow_up_result = await performance_system.process_query_optimized(follow_up)
            follow_up_time = (time.time() - start_time) * 1000
            follow_up_times.append(follow_up_time)
        
        # Follow-up queries should be faster on average due to predictive caching
        avg_follow_up_time = statistics.mean(follow_up_times)
        assert avg_follow_up_time < 150  # Should be fast due to prediction
    
    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, performance_system, query_generator):
        """Test cache performance under concurrent access."""
        concurrent_queries = query_generator.generate_query_sequence(60, 'mixed')
        
        # Create concurrent tasks
        async def process_query_batch(queries_batch):
            results = []
            for query in queries_batch:
                result = await performance_system.process_query_optimized(query)
                results.append(result)
            return results
        
        # Split queries into batches for concurrent processing
        batch_size = 10
        batches = [concurrent_queries[i:i + batch_size] 
                  for i in range(0, len(concurrent_queries), batch_size)][:3]  # Limit batches
        
        # Process batches concurrently
        start_time = time.time()
        batch_tasks = [process_query_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        concurrent_duration = time.time() - start_time
        
        # Verify concurrent performance
        total_queries = sum(len(batch_result) for batch_result in batch_results)
        concurrent_qps = total_queries / concurrent_duration
        
        assert concurrent_qps > 5  # Should maintain reasonable throughput
        
        # Verify cache consistency - same queries should have consistent results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        # Check for duplicate queries and consistent results
        query_results = {}
        for result in all_results:
            query_text = result['query_result'].query_text
            if query_text in query_results:
                # Should have consistent routing decisions
                prev_result = query_results[query_text]
                assert (result['query_result'].routing_decision == 
                       prev_result['query_result'].routing_decision)
            query_results[query_text] = result
    
    @pytest.mark.asyncio
    async def test_cache_configuration_optimization(self, performance_system):
        """Test cache configuration optimization recommendations."""
        # Simulate different workload profiles
        workload_profiles = [
            {
                'name': 'high_throughput',
                'queries_per_second': 100,
                'current_cache_hit_ratio': 0.6,
                'memory_budget_mb': 500
            },
            {
                'name': 'memory_constrained', 
                'queries_per_second': 20,
                'current_cache_hit_ratio': 0.4,
                'memory_budget_mb': 50
            },
            {
                'name': 'research_intensive',
                'queries_per_second': 15,
                'current_cache_hit_ratio': 0.8,
                'memory_budget_mb': 200
            }
        ]
        
        optimization_results = []
        for profile in workload_profiles:
            optimization = await performance_system.optimize_cache_configuration(profile)
            optimization_results.append((profile['name'], optimization))
        
        # Verify optimization recommendations
        for profile_name, optimization in optimization_results:
            assert 'recommendations' in optimization
            assert 'estimated_performance_impact' in optimization
            assert len(optimization['recommendations']) > 0
            
            # High throughput profile should get more aggressive recommendations
            if profile_name == 'high_throughput':
                recommendation_types = [rec['type'] for rec in optimization['recommendations']]
                assert 'enable_predictive_caching' in recommendation_types or 'increase_cache_size' in recommendation_types
    
    @pytest.mark.asyncio 
    async def test_memory_usage_optimization(self, performance_system, query_generator, 
                                           performance_monitor):
        """Test memory usage optimization under sustained load."""
        sustained_queries = query_generator.generate_query_sequence(80, 'mixed')
        
        performance_monitor.start_monitoring()
        
        # Process sustained load
        for i, query in enumerate(sustained_queries[:40]):  # Limit for test
            await performance_system.process_query_optimized(query)
            
            # Force garbage collection periodically to test memory efficiency
            if i % 10 == 0:
                gc.collect()
        
        performance_monitor.stop_monitoring()
        
        # Verify memory usage stayed within reasonable bounds
        metrics_summary = performance_monitor.metrics.get_summary_report()
        resource_metrics = metrics_summary['resource_utilization']
        
        assert resource_metrics['peak_memory_mb'] < 200  # Should not exceed 200MB
        assert resource_metrics['avg_memory_mb'] < 150   # Average should be reasonable
        
        # Get performance insights
        insights = performance_system.get_performance_insights()
        
        assert 'query_frequency_analysis' in insights
        assert 'cache_pattern_analysis' in insights  
        assert 'optimization_recommendations' in insights
        
        # Should have some optimization recommendations for sustained load
        recommendations = insights['optimization_recommendations']
        assert len(recommendations) > 0


class TestLoadTestingScenarios:
    """Load testing scenarios for cache performance validation."""
    
    @pytest.mark.asyncio
    async def test_research_conference_peak_load(self, performance_system, query_generator):
        """Simulate peak load during research conference (high concurrent queries)."""
        # Conference scenario: Many researchers accessing system simultaneously
        conference_config = LoadTestConfiguration(
            test_name="research_conference_peak",
            duration_seconds=30,  # Short test for CI
            concurrent_users=15,
            queries_per_user=8,
            ramp_up_seconds=5,
            target_qps=20.0,
            target_cache_hit_ratio=0.7
        )
        
        # Generate conference-style queries (mix of common and specialized)
        common_queries = query_generator.generate_query_sequence(30, 'metabolite_focused')
        specialized_queries = query_generator.generate_query_sequence(20, 'clinical_focused')
        
        # Simulate concurrent users
        async def simulate_user(user_id: int, queries: List[str]):
            user_results = []
            for query in queries:
                start_time = time.time()
                result = await performance_system.process_query_optimized(
                    f"{query} (user_{user_id})"  # Slight variation per user
                )
                response_time = (time.time() - start_time) * 1000
                user_results.append((query, response_time, result))
            return user_results
        
        # Create user simulation tasks
        user_tasks = []
        for user_id in range(conference_config.concurrent_users):
            # Mix of common and specialized queries per user
            user_queries = (common_queries[:4] + specialized_queries[:2] + 
                          common_queries[4:6])[:conference_config.queries_per_user]
            user_task = simulate_user(user_id, user_queries)
            user_tasks.append(user_task)
        
        # Execute load test
        start_time = time.time()
        user_results = await asyncio.gather(*user_tasks)
        test_duration = time.time() - start_time
        
        # Analyze results
        all_response_times = []
        total_queries = 0
        successful_queries = 0
        
        for user_result in user_results:
            for query, response_time, result in user_result:
                all_response_times.append(response_time)
                total_queries += 1
                if result and result.get('query_result'):
                    successful_queries += 1
        
        # Calculate performance metrics
        qps_achieved = total_queries / test_duration
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        p95_response_time = (statistics.quantiles(all_response_times, n=20)[18] 
                           if len(all_response_times) >= 20 else max(all_response_times))
        
        # Verify performance targets
        assert qps_achieved > conference_config.target_qps * 0.8  # 80% of target QPS
        assert success_rate > 0.95  # 95% success rate
        assert avg_response_time < 300  # Average < 300ms
        assert p95_response_time < conference_config.target_p95_response_time_ms
    
    @pytest.mark.asyncio
    async def test_sustained_research_workload(self, performance_system, query_generator):
        """Test sustained research workload over extended period."""
        # Sustained workload: Steady research activity over time
        sustained_config = LoadTestConfiguration(
            test_name="sustained_research_workload",
            duration_seconds=45,  # Longer test
            concurrent_users=8,
            queries_per_user=15,
            ramp_up_seconds=10,
            target_qps=12.0,
            target_cache_hit_ratio=0.85  # Higher hit ratio expected for sustained load
        )
        
        # Generate research workflow queries
        workflow_queries = (
            query_generator.generate_query_sequence(25, 'metabolite_focused') +
            query_generator.generate_query_sequence(25, 'pathway_queries') +
            query_generator.generate_query_sequence(25, 'clinical_focused')
        )
        
        # Simulate sustained research activity
        async def sustained_researcher(researcher_id: int, query_sequence: List[str]):
            researcher_metrics = []
            
            for i, query in enumerate(query_sequence):
                start_time = time.time()
                
                # Add some realistic delays between queries (research thinking time)
                if i > 0:
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                
                result = await performance_system.process_query_optimized(query)
                response_time = (time.time() - start_time) * 1000
                
                researcher_metrics.append({
                    'query_index': i,
                    'response_time_ms': response_time,
                    'success': result is not None,
                    'cached': getattr(result.get('query_result'), 'cached_result', False) if result else False
                })
            
            return researcher_metrics
        
        # Distribute queries among researchers
        queries_per_researcher = len(workflow_queries) // sustained_config.concurrent_users
        researcher_tasks = []
        
        for researcher_id in range(sustained_config.concurrent_users):
            start_idx = researcher_id * queries_per_researcher
            end_idx = start_idx + sustained_config.queries_per_user
            researcher_queries = workflow_queries[start_idx:end_idx]
            
            researcher_task = sustained_researcher(researcher_id, researcher_queries)
            researcher_tasks.append(researcher_task)
        
        # Execute sustained workload test
        start_time = time.time()
        researcher_results = await asyncio.gather(*researcher_tasks)
        test_duration = time.time() - start_time
        
        # Analyze sustained workload performance
        all_metrics = []
        for researcher_result in researcher_results:
            all_metrics.extend(researcher_result)
        
        # Calculate aggregated metrics
        total_queries = len(all_metrics)
        successful_queries = sum(1 for m in all_metrics if m['success'])
        cached_queries = sum(1 for m in all_metrics if m.get('cached', False))
        
        response_times = [m['response_time_ms'] for m in all_metrics if m['success']]
        
        qps_sustained = total_queries / test_duration
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        cache_hit_ratio = cached_queries / total_queries if total_queries > 0 else 0
        
        # Verify sustained performance targets
        assert qps_sustained > sustained_config.target_qps * 0.9  # 90% of target
        assert success_rate > 0.98  # Very high success rate for sustained load
        assert cache_hit_ratio > sustained_config.target_cache_hit_ratio * 0.8  # 80% of target hit ratio
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            assert avg_response_time < 250  # Should be faster due to caching benefits
    
    @pytest.mark.asyncio
    async def test_mixed_workload_cache_efficiency(self, performance_system, query_generator):
        """Test cache efficiency with mixed workload patterns."""
        # Mixed workload: Different types of queries with varying cache patterns
        
        # Generate different types of query patterns
        frequent_queries = query_generator.generate_query_sequence(10, 'metabolite_focused')  # Will be repeated
        occasional_queries = query_generator.generate_query_sequence(20, 'clinical_focused')  # Less frequent
        unique_queries = query_generator.generate_query_sequence(30, 'literature_queries')   # Mostly unique
        
        # Create mixed workload pattern
        mixed_workload = []
        
        # Add frequent queries multiple times (should have high cache hit ratio)
        for _ in range(5):
            mixed_workload.extend(frequent_queries)
        
        # Add occasional queries few times
        for _ in range(2):
            mixed_workload.extend(occasional_queries[:10])
        
        # Add unique queries once
        mixed_workload.extend(unique_queries[:15])
        
        # Shuffle to create realistic access pattern
        random.shuffle(mixed_workload)
        
        # Process mixed workload
        cache_performance_samples = []
        
        for i, query in enumerate(mixed_workload[:50]):  # Limit for test performance
            start_time = time.time()
            result = await performance_system.process_query_optimized(query)
            processing_time = (time.time() - start_time) * 1000
            
            cached = getattr(result.get('query_result'), 'cached_result', False) if result else False
            
            cache_performance_samples.append({
                'query_index': i,
                'query_type': self._classify_query_type(query, frequent_queries, occasional_queries),
                'processing_time_ms': processing_time,
                'cached': cached,
                'success': result is not None
            })
        
        # Analyze cache efficiency by query type
        performance_by_type = defaultdict(list)
        for sample in cache_performance_samples:
            performance_by_type[sample['query_type']].append(sample)
        
        # Verify cache efficiency patterns
        for query_type, samples in performance_by_type.items():
            cached_count = sum(1 for s in samples if s['cached'])
            cache_hit_ratio = cached_count / len(samples) if samples else 0
            avg_time = statistics.mean([s['processing_time_ms'] for s in samples]) if samples else 0
            
            if query_type == 'frequent':
                # Frequent queries should have high cache hit ratio and low response time
                assert cache_hit_ratio > 0.7, f"Frequent queries cache hit ratio: {cache_hit_ratio}"
                assert avg_time < 150, f"Frequent queries avg time: {avg_time}ms"
            elif query_type == 'occasional':
                # Occasional queries should have moderate cache performance
                assert cache_hit_ratio > 0.3, f"Occasional queries cache hit ratio: {cache_hit_ratio}"
            elif query_type == 'unique':
                # Unique queries will have low cache hit ratio but should still be fast
                assert avg_time < 300, f"Unique queries avg time: {avg_time}ms"
    
    def _classify_query_type(self, query: str, frequent_queries: List[str], 
                           occasional_queries: List[str]) -> str:
        """Classify query type for cache efficiency analysis."""
        if query in frequent_queries:
            return 'frequent'
        elif query in occasional_queries:
            return 'occasional'
        else:
            return 'unique'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])