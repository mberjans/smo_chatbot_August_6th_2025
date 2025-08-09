"""
Domain-Specific Biomedical Query Performance Tests for Clinical Metabolomics Oracle.

This module provides comprehensive domain-specific performance testing for biomedical
query processing, focusing on clinical research workflows, query classification performance,
multi-tier cache coordination, and real-world biomedical scenarios with authentic data patterns.

Test Coverage:
- Clinical research workflow performance with caching
- Query classification performance with cache integration  
- Multi-tier cache coordination performance impact
- Emergency fallback system performance validation
- Biomedical query processing performance improvement
- Conference/peak load handling with caching
- Long-running research session cache optimization
- Cross-language query performance with cache benefits

Performance Targets:
- Clinical workflow queries: <200ms average response with >75% cache hit rate
- Query classification: <50ms classification time with caching
- Emergency fallback: <500ms response time for critical queries
- Peak load handling: Support 200+ concurrent users during conferences
- Research sessions: Maintain >80% hit rate over 4+ hour sessions
- Cross-language queries: <300ms response with multilingual cache optimization

Classes:
    TestClinicalResearchWorkflowPerformance: Clinical workflow performance testing
    TestQueryClassificationPerformance: Query classification with caching
    TestMultiTierCacheCoordinationPerformance: Cache coordination impact
    TestEmergencyFallbackPerformance: Emergency system performance
    TestBiomedicalQueryProcessingOptimization: Query processing optimization
    TestRealWorldBiomedicalScenarios: Real-world scenario validation

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import pytest
import asyncio
import time
import threading
import random
import statistics
import gc
import sys
import psutil
import concurrent.futures
import json
import os
import re
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from contextlib import contextmanager
from enum import Enum
import tempfile
import shutil

# Import test fixtures and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'unit'))
from cache_test_fixtures import (
    BIOMEDICAL_QUERIES,
    BiomedicalTestDataGenerator,
    CachePerformanceMetrics,
    CachePerformanceMeasurer,
    EMERGENCY_RESPONSE_PATTERNS
)

# Import cache implementations
from test_cache_effectiveness import (
    HighPerformanceCache,
    CacheEffectivenessMetrics,
    PERFORMANCE_TARGETS
)
from test_cache_scalability import (
    ScalabilityMetrics,
    ResourceMonitor,
    SCALABILITY_TARGETS
)

# Domain-Specific Performance Targets
BIOMEDICAL_PERFORMANCE_TARGETS = {
    'clinical_workflow_avg_ms': 200,      # Clinical workflow average response
    'clinical_workflow_hit_rate': 0.75,   # Clinical workflow cache hit rate
    'query_classification_ms': 50,        # Query classification time
    'emergency_fallback_ms': 500,         # Emergency fallback response time
    'peak_load_users': 200,               # Peak concurrent users during conferences
    'research_session_hit_rate': 0.80,    # Long research session hit rate
    'cross_language_response_ms': 300,    # Cross-language query response
    'biomarker_query_ms': 150,            # Biomarker-specific queries
    'pathway_analysis_ms': 250,           # Metabolic pathway analysis
    'literature_search_ms': 180,          # Literature search queries
    'cache_coordination_overhead_ms': 10  # Multi-tier coordination overhead
}

# Clinical Workflow Types
class WorkflowType(Enum):
    DIAGNOSTIC = "diagnostic"
    RESEARCH = "research"
    SCREENING = "screening"
    MONITORING = "monitoring"
    VALIDATION = "validation"

# Query Classification Categories
class QueryCategory(Enum):
    METABOLISM = "metabolism"
    BIOMARKERS = "biomarkers" 
    PATHWAYS = "pathways"
    DISEASES = "diseases"
    METHODS = "methods"
    LITERATURE = "literature"
    TEMPORAL = "temporal"
    EMERGENCY = "emergency"


@dataclass
class BiomedicalQueryPerformanceMetrics:
    """Domain-specific biomedical query performance metrics."""
    workflow_type: str
    query_category: str
    total_queries: int
    successful_queries: int
    cache_hits: int
    cache_misses: int
    classification_time_ms: float
    processing_time_ms: float
    total_response_time_ms: float
    cache_coordination_time_ms: float
    emergency_fallback_invoked: int
    cross_language_queries: int
    biomarker_specific_queries: int
    pathway_analysis_queries: int
    
    # Performance percentiles
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # Domain-specific metrics
    clinical_accuracy_score: float
    knowledge_coverage_score: float
    temporal_relevance_score: float
    
    def meets_biomedical_targets(self) -> bool:
        """Check if metrics meet biomedical performance targets."""
        targets = BIOMEDICAL_PERFORMANCE_TARGETS
        
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        success_rate = self.successful_queries / self.total_queries if self.total_queries > 0 else 0
        
        return all([
            self.total_response_time_ms <= targets.get(f'{self.workflow_type}_avg_ms', targets['clinical_workflow_avg_ms']),
            cache_hit_rate >= targets.get(f'{self.workflow_type}_hit_rate', targets['clinical_workflow_hit_rate']),
            self.classification_time_ms <= targets['query_classification_ms'],
            success_rate >= 0.95,
            self.clinical_accuracy_score >= 0.90,
            self.cache_coordination_time_ms <= targets['cache_coordination_overhead_ms']
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        success_rate = self.successful_queries / self.total_queries if self.total_queries > 0 else 0
        
        return {
            'workflow_configuration': {
                'workflow_type': self.workflow_type,
                'query_category': self.query_category,
                'total_queries': self.total_queries,
                'successful_queries': self.successful_queries
            },
            'cache_performance': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': cache_hit_rate,
                'coordination_time_ms': self.cache_coordination_time_ms
            },
            'response_times': {
                'classification_ms': self.classification_time_ms,
                'processing_ms': self.processing_time_ms,
                'total_response_ms': self.total_response_time_ms,
                'p50_ms': self.p50_response_time_ms,
                'p95_ms': self.p95_response_time_ms,
                'p99_ms': self.p99_response_time_ms
            },
            'domain_metrics': {
                'success_rate': success_rate,
                'emergency_fallbacks': self.emergency_fallback_invoked,
                'cross_language_queries': self.cross_language_queries,
                'biomarker_queries': self.biomarker_specific_queries,
                'pathway_queries': self.pathway_analysis_queries,
                'clinical_accuracy': self.clinical_accuracy_score,
                'knowledge_coverage': self.knowledge_coverage_score,
                'temporal_relevance': self.temporal_relevance_score
            },
            'performance_validation': {
                'meets_targets': self.meets_biomedical_targets()
            }
        }


class BiomedicalQueryClassifier:
    """Intelligent biomedical query classifier with caching."""
    
    def __init__(self, cache: HighPerformanceCache):
        self.cache = cache
        self.classification_cache = {}  # Local classification cache
        self.classification_patterns = self._build_classification_patterns()
    
    def _build_classification_patterns(self) -> Dict[QueryCategory, List[str]]:
        """Build regex patterns for query classification."""
        return {
            QueryCategory.METABOLISM: [
                r'metabol(ism|ite|ic)',
                r'pathway',
                r'glycolysis',
                r'citric acid',
                r'electron transport',
                r'ATP',
                r'glucose.*metabolism'
            ],
            QueryCategory.BIOMARKERS: [
                r'biomarker',
                r'indicator',
                r'diagnostic.*marker',
                r'prognostic',
                r'screening.*test',
                r'blood.*test'
            ],
            QueryCategory.PATHWAYS: [
                r'pathway',
                r'signaling',
                r'cascade',
                r'regulation',
                r'metabolic.*network'
            ],
            QueryCategory.DISEASES: [
                r'disease',
                r'disorder',
                r'syndrome',
                r'cancer',
                r'diabetes',
                r'cardiovascular',
                r'neurological'
            ],
            QueryCategory.METHODS: [
                r'LC-?MS',
                r'GC-?MS',
                r'NMR',
                r'spectroscopy',
                r'chromatography',
                r'mass.*spectrometry',
                r'analytical.*method'
            ],
            QueryCategory.LITERATURE: [
                r'recent.*research',
                r'studies.*show',
                r'literature.*review',
                r'publication',
                r'paper'
            ],
            QueryCategory.TEMPORAL: [
                r'latest',
                r'recent',
                r'current',
                r'2024',
                r'2023',
                r'new.*findings',
                r'updated'
            ],
            QueryCategory.EMERGENCY: [
                r'urgent',
                r'emergency',
                r'critical',
                r'immediate',
                r'stat'
            ]
        }
    
    async def classify_query(self, query: str) -> Tuple[QueryCategory, float, float]:
        """Classify query and return category, confidence, and classification time."""
        start_time = time.time()
        
        # Check classification cache first
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        if query_hash in self.classification_cache:
            classification_time = (time.time() - start_time) * 1000
            cached_result = self.classification_cache[query_hash]
            return cached_result[0], cached_result[1], classification_time
        
        # Perform classification
        category_scores = {}
        
        for category, patterns in self.classification_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, query.lower()))
                score += matches * (1.0 / len(patterns))  # Weight by pattern count
            
            category_scores[category] = score
        
        # Find best category
        best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        confidence = min(1.0, category_scores[best_category])
        
        # If no strong match, default to general metabolism
        if confidence < 0.1:
            best_category = QueryCategory.METABOLISM
            confidence = 0.5
        
        classification_time = (time.time() - start_time) * 1000
        
        # Cache the result
        self.classification_cache[query_hash] = (best_category, confidence)
        
        return best_category, confidence, classification_time


class ClinicalWorkflowSimulator:
    """Simulator for clinical research workflows."""
    
    def __init__(self, cache: HighPerformanceCache, classifier: BiomedicalQueryClassifier):
        self.cache = cache
        self.classifier = classifier
        self.workflow_templates = self._build_workflow_templates()
    
    def _build_workflow_templates(self) -> Dict[WorkflowType, Dict[str, Any]]:
        """Build clinical workflow templates."""
        return {
            WorkflowType.DIAGNOSTIC: {
                'phases': ['symptom_analysis', 'biomarker_screening', 'differential_diagnosis'],
                'queries_per_phase': [5, 8, 3],
                'expected_hit_rate': 0.80,  # High for common diagnostic queries
                'time_pressure': True,
                'accuracy_requirement': 0.95
            },
            WorkflowType.RESEARCH: {
                'phases': ['literature_review', 'hypothesis_formation', 'data_analysis'],
                'queries_per_phase': [15, 10, 12],
                'expected_hit_rate': 0.70,  # Medium for research queries
                'time_pressure': False,
                'accuracy_requirement': 0.90
            },
            WorkflowType.SCREENING: {
                'phases': ['population_analysis', 'risk_assessment', 'recommendation'],
                'queries_per_phase': [8, 6, 4],
                'expected_hit_rate': 0.85,  # High for standardized screening
                'time_pressure': True,
                'accuracy_requirement': 0.92
            },
            WorkflowType.MONITORING: {
                'phases': ['baseline_comparison', 'trend_analysis', 'intervention_assessment'],
                'queries_per_phase': [4, 8, 6],
                'expected_hit_rate': 0.75,  # Medium-high for monitoring
                'time_pressure': False,
                'accuracy_requirement': 0.88
            },
            WorkflowType.VALIDATION: {
                'phases': ['method_verification', 'accuracy_testing', 'precision_analysis'],
                'queries_per_phase': [6, 10, 8],
                'expected_hit_rate': 0.60,  # Lower for specialized validation
                'time_pressure': False,
                'accuracy_requirement': 0.93
            }
        }
    
    async def simulate_clinical_workflow(
        self,
        workflow_type: WorkflowType,
        data_generator: BiomedicalTestDataGenerator
    ) -> BiomedicalQueryPerformanceMetrics:
        """Simulate a complete clinical workflow."""
        template = self.workflow_templates[workflow_type]
        
        total_queries = sum(template['queries_per_phase'])
        successful_queries = 0
        cache_hits = 0
        cache_misses = 0
        emergency_fallbacks = 0
        
        all_response_times = []
        all_classification_times = []
        all_processing_times = []
        all_coordination_times = []
        
        biomarker_queries = 0
        pathway_queries = 0
        cross_language_queries = 0
        
        clinical_accuracy_scores = []
        knowledge_coverage_scores = []
        temporal_relevance_scores = []
        
        # Simulate each workflow phase
        for phase_idx, (phase_name, query_count) in enumerate(zip(template['phases'], template['queries_per_phase'])):
            print(f"  Simulating {phase_name} phase ({query_count} queries)...")
            
            for query_idx in range(query_count):
                # Generate appropriate query for phase
                query_data = self._generate_phase_appropriate_query(phase_name, data_generator)
                query = query_data['query']
                
                # Track special query types
                if 'biomarker' in query.lower():
                    biomarker_queries += 1
                if 'pathway' in query.lower():
                    pathway_queries += 1
                if query_data.get('cross_language', False):
                    cross_language_queries += 1
                
                try:
                    # Full query processing pipeline
                    start_time = time.time()
                    
                    # 1. Query Classification
                    category, confidence, classification_time = await self.classifier.classify_query(query)
                    all_classification_times.append(classification_time)
                    
                    # 2. Cache Coordination
                    coordination_start = time.time()
                    cached_result = await self.cache.get(query)
                    coordination_time = (time.time() - coordination_start) * 1000
                    all_coordination_times.append(coordination_time)
                    
                    # 3. Processing (cache hit or miss)
                    processing_start = time.time()
                    
                    if cached_result is not None:
                        # Cache hit
                        cache_hits += 1
                        result = cached_result
                        processing_time = (time.time() - processing_start) * 1000
                    else:
                        # Cache miss - simulate processing
                        cache_misses += 1
                        processing_time = self._simulate_biomedical_processing(query, category, workflow_type)
                        
                        # Generate and cache result
                        result = self._generate_biomedical_response(query, category, workflow_type)
                        await self.cache.set(query, result, ttl=self._get_ttl_for_category(category))
                    
                    all_processing_times.append(processing_time)
                    
                    # 4. Quality Assessment
                    clinical_accuracy = self._assess_clinical_accuracy(query, result, workflow_type)
                    knowledge_coverage = self._assess_knowledge_coverage(query, result, category)
                    temporal_relevance = self._assess_temporal_relevance(query, result)
                    
                    clinical_accuracy_scores.append(clinical_accuracy)
                    knowledge_coverage_scores.append(knowledge_coverage)
                    temporal_relevance_scores.append(temporal_relevance)
                    
                    total_response_time = (time.time() - start_time) * 1000
                    all_response_times.append(total_response_time)
                    
                    successful_queries += 1
                    
                except Exception as e:
                    # Emergency fallback
                    emergency_fallbacks += 1
                    fallback_time = await self._handle_emergency_fallback(query)
                    all_response_times.append(fallback_time)
        
        # Calculate metrics
        avg_classification_time = statistics.mean(all_classification_times) if all_classification_times else 0
        avg_processing_time = statistics.mean(all_processing_times) if all_processing_times else 0
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        avg_coordination_time = statistics.mean(all_coordination_times) if all_coordination_times else 0
        
        # Calculate percentiles
        if all_response_times:
            all_response_times.sort()
            p50_time = all_response_times[len(all_response_times) // 2]
            p95_time = all_response_times[int(len(all_response_times) * 0.95)]
            p99_time = all_response_times[int(len(all_response_times) * 0.99)]
        else:
            p50_time = p95_time = p99_time = 0
        
        return BiomedicalQueryPerformanceMetrics(
            workflow_type=workflow_type.value,
            query_category="mixed",
            total_queries=total_queries,
            successful_queries=successful_queries,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            classification_time_ms=avg_classification_time,
            processing_time_ms=avg_processing_time,
            total_response_time_ms=avg_response_time,
            cache_coordination_time_ms=avg_coordination_time,
            emergency_fallback_invoked=emergency_fallbacks,
            cross_language_queries=cross_language_queries,
            biomarker_specific_queries=biomarker_queries,
            pathway_analysis_queries=pathway_queries,
            p50_response_time_ms=p50_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            clinical_accuracy_score=statistics.mean(clinical_accuracy_scores) if clinical_accuracy_scores else 0,
            knowledge_coverage_score=statistics.mean(knowledge_coverage_scores) if knowledge_coverage_scores else 0,
            temporal_relevance_score=statistics.mean(temporal_relevance_scores) if temporal_relevance_scores else 0
        )
    
    def _generate_phase_appropriate_query(self, phase_name: str, data_generator: BiomedicalTestDataGenerator) -> Dict[str, Any]:
        """Generate query appropriate for workflow phase."""
        phase_query_types = {
            'symptom_analysis': 'disease',
            'biomarker_screening': 'methods',
            'differential_diagnosis': 'disease',
            'literature_review': 'random',
            'hypothesis_formation': 'metabolism',
            'data_analysis': 'methods',
            'population_analysis': 'disease',
            'risk_assessment': 'random',
            'recommendation': 'methods',
            'baseline_comparison': 'metabolism',
            'trend_analysis': 'metabolism',
            'intervention_assessment': 'disease',
            'method_verification': 'methods',
            'accuracy_testing': 'methods',
            'precision_analysis': 'methods'
        }
        
        query_type = phase_query_types.get(phase_name, 'random')
        query_data = data_generator.generate_query(query_type)
        
        # Add phase-specific context
        query_data['phase'] = phase_name
        query_data['cross_language'] = random.random() < 0.1  # 10% cross-language queries
        
        return query_data
    
    def _simulate_biomedical_processing(self, query: str, category: QueryCategory, workflow_type: WorkflowType) -> float:
        """Simulate biomedical query processing time."""
        base_time = 50  # 50ms base time
        
        # Category-specific processing complexity
        category_multipliers = {
            QueryCategory.METABOLISM: 1.2,
            QueryCategory.BIOMARKERS: 1.0,
            QueryCategory.PATHWAYS: 1.5,
            QueryCategory.DISEASES: 1.3,
            QueryCategory.METHODS: 1.1,
            QueryCategory.LITERATURE: 1.8,
            QueryCategory.TEMPORAL: 1.4,
            QueryCategory.EMERGENCY: 0.8
        }
        
        # Workflow-specific time pressure
        workflow_multipliers = {
            WorkflowType.DIAGNOSTIC: 0.8,  # Time pressure reduces processing time
            WorkflowType.RESEARCH: 1.2,    # More thorough processing
            WorkflowType.SCREENING: 0.9,   # Standardized, efficient
            WorkflowType.MONITORING: 1.0,  # Standard processing
            WorkflowType.VALIDATION: 1.3   # Careful, thorough processing
        }
        
        category_mult = category_multipliers.get(category, 1.0)
        workflow_mult = workflow_multipliers.get(workflow_type, 1.0)
        
        processing_time_ms = base_time * category_mult * workflow_mult
        
        # Add variability
        processing_time_ms *= random.uniform(0.8, 1.2)
        
        # Simulate processing delay
        time.sleep(processing_time_ms / 1000)
        
        return processing_time_ms
    
    def _generate_biomedical_response(self, query: str, category: QueryCategory, workflow_type: WorkflowType) -> Dict[str, Any]:
        """Generate realistic biomedical response."""
        return {
            'query': query,
            'category': category.value,
            'workflow_type': workflow_type.value,
            'response': f"Biomedical response for {category.value} query in {workflow_type.value} workflow",
            'confidence': random.uniform(0.85, 0.98),
            'sources': ['PubMed', 'KEGG', 'MetaboAnalyst'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_ttl_for_category(self, category: QueryCategory) -> int:
        """Get appropriate TTL for query category."""
        ttl_mapping = {
            QueryCategory.METABOLISM: 7200,    # 2 hours - stable knowledge
            QueryCategory.BIOMARKERS: 3600,   # 1 hour - moderate stability
            QueryCategory.PATHWAYS: 7200,     # 2 hours - stable knowledge
            QueryCategory.DISEASES: 1800,     # 30 minutes - evolving knowledge
            QueryCategory.METHODS: 3600,      # 1 hour - moderate stability
            QueryCategory.LITERATURE: 900,    # 15 minutes - frequently updated
            QueryCategory.TEMPORAL: 300,      # 5 minutes - time-sensitive
            QueryCategory.EMERGENCY: 600      # 10 minutes - urgent but may repeat
        }
        
        return ttl_mapping.get(category, 3600)
    
    def _assess_clinical_accuracy(self, query: str, result: Dict[str, Any], workflow_type: WorkflowType) -> float:
        """Assess clinical accuracy of response."""
        # Simulate clinical accuracy assessment
        base_accuracy = 0.90
        
        # Workflow-specific accuracy expectations
        workflow_accuracy = {
            WorkflowType.DIAGNOSTIC: 0.95,
            WorkflowType.RESEARCH: 0.88,
            WorkflowType.SCREENING: 0.92,
            WorkflowType.MONITORING: 0.87,
            WorkflowType.VALIDATION: 0.94
        }
        
        expected_accuracy = workflow_accuracy.get(workflow_type, base_accuracy)
        
        # Add some variability
        actual_accuracy = expected_accuracy * random.uniform(0.95, 1.05)
        
        return min(1.0, actual_accuracy)
    
    def _assess_knowledge_coverage(self, query: str, result: Dict[str, Any], category: QueryCategory) -> float:
        """Assess knowledge coverage of response."""
        # Categories with different knowledge coverage expectations
        coverage_expectations = {
            QueryCategory.METABOLISM: 0.92,
            QueryCategory.BIOMARKERS: 0.88,
            QueryCategory.PATHWAYS: 0.90,
            QueryCategory.DISEASES: 0.85,
            QueryCategory.METHODS: 0.93,
            QueryCategory.LITERATURE: 0.80,
            QueryCategory.TEMPORAL: 0.75,
            QueryCategory.EMERGENCY: 0.85
        }
        
        expected_coverage = coverage_expectations.get(category, 0.85)
        actual_coverage = expected_coverage * random.uniform(0.9, 1.1)
        
        return min(1.0, actual_coverage)
    
    def _assess_temporal_relevance(self, query: str, result: Dict[str, Any]) -> float:
        """Assess temporal relevance of response."""
        # Check if query has temporal keywords
        temporal_keywords = ['latest', 'recent', 'current', '2024', '2023', 'new']
        has_temporal = any(keyword in query.lower() for keyword in temporal_keywords)
        
        if has_temporal:
            # Temporal queries need high relevance
            return random.uniform(0.80, 0.95)
        else:
            # Non-temporal queries have stable relevance
            return random.uniform(0.90, 0.98)
    
    async def _handle_emergency_fallback(self, query: str) -> float:
        """Handle emergency fallback processing."""
        start_time = time.time()
        
        # Check emergency response patterns
        for pattern_name, pattern_data in EMERGENCY_RESPONSE_PATTERNS.items():
            for pattern in pattern_data['patterns']:
                if pattern == '*' or pattern.lower() in query.lower():
                    # Found emergency response
                    response = pattern_data['response']
                    
                    # Cache emergency response
                    await self.cache.set(f"emergency:{query}", response, ttl=600)
                    break
        
        # Simulate emergency processing time
        emergency_time = random.uniform(200, 400)  # 200-400ms for emergency
        time.sleep(emergency_time / 1000)
        
        return (time.time() - start_time) * 1000


class BiomedicalPerformanceTestRunner:
    """Test runner for biomedical query performance validation."""
    
    def __init__(self):
        self.data_generator = BiomedicalTestDataGenerator()
        self.resource_monitor = ResourceMonitor()
    
    def run_clinical_workflow_performance_test(
        self,
        cache: HighPerformanceCache,
        workflow_type: WorkflowType,
        iterations: int = 5
    ) -> BiomedicalQueryPerformanceMetrics:
        """Run clinical workflow performance test."""
        
        classifier = BiomedicalQueryClassifier(cache)
        simulator = ClinicalWorkflowSimulator(cache, classifier)
        
        print(f"Running {workflow_type.value} workflow performance test ({iterations} iterations)...")
        
        # Collect metrics across iterations
        all_metrics = []
        
        for iteration in range(iterations):
            print(f"  Iteration {iteration + 1}/{iterations}")
            
            metrics = asyncio.run(simulator.simulate_clinical_workflow(workflow_type, self.data_generator))
            all_metrics.append(metrics)
        
        # Aggregate results
        return self._aggregate_biomedical_metrics(all_metrics, workflow_type.value)
    
    def run_query_classification_performance_test(
        self,
        cache: HighPerformanceCache,
        query_count: int = 1000
    ) -> Dict[str, Any]:
        """Run query classification performance test."""
        
        classifier = BiomedicalQueryClassifier(cache)
        
        # Generate diverse query set
        test_queries = []
        for category in ['metabolism', 'disease', 'methods', 'random']:
            queries = self.data_generator.generate_batch(query_count // 4, category)
            test_queries.extend(queries)
        
        classification_times = []
        category_counts = Counter()
        confidence_scores = []
        
        print(f"Running query classification performance test ({len(test_queries)} queries)...")
        
        start_time = time.time()
        
        for i, query_data in enumerate(test_queries):
            if i % 100 == 0:
                print(f"  Classified {i}/{len(test_queries)} queries")
            
            query = query_data['query']
            category, confidence, classification_time = asyncio.run(classifier.classify_query(query))
            
            classification_times.append(classification_time)
            category_counts[category.value] += 1
            confidence_scores.append(confidence)
        
        total_time = time.time() - start_time
        
        return {
            'total_queries': len(test_queries),
            'total_time_seconds': total_time,
            'avg_classification_time_ms': statistics.mean(classification_times),
            'min_classification_time_ms': min(classification_times),
            'max_classification_time_ms': max(classification_times),
            'p95_classification_time_ms': statistics.quantiles(classification_times, n=20)[18],  # 95th percentile
            'throughput_queries_per_second': len(test_queries) / total_time,
            'avg_confidence': statistics.mean(confidence_scores),
            'category_distribution': dict(category_counts),
            'meets_target': statistics.mean(classification_times) <= BIOMEDICAL_PERFORMANCE_TARGETS['query_classification_ms']
        }
    
    def run_multi_tier_coordination_performance_test(
        self,
        base_cache: HighPerformanceCache,
        coordination_scenarios: List[str]
    ) -> Dict[str, Any]:
        """Run multi-tier cache coordination performance test."""
        
        coordination_results = {}
        
        for scenario in coordination_scenarios:
            print(f"Testing {scenario} coordination scenario...")
            
            # Create scenario-specific cache configuration
            if scenario == 'l1_optimized':
                cache = HighPerformanceCache(l1_size=5000, l2_size=2000, l3_enabled=False)
            elif scenario == 'l2_optimized':
                cache = HighPerformanceCache(l1_size=1000, l2_size=10000, l3_enabled=False)
            elif scenario == 'l3_enabled':
                cache = HighPerformanceCache(l1_size=2000, l2_size=5000, l3_enabled=True)
            else:  # balanced
                cache = HighPerformanceCache(l1_size=2000, l2_size=6000, l3_enabled=True)
            
            # Run coordination test
            test_queries = self.data_generator.generate_batch(500, 'random')
            
            coordination_times = []
            cache_hits_by_tier = {'l1': 0, 'l2': 0, 'l3': 0, 'miss': 0}
            
            # Pre-populate some entries
            for i in range(0, len(test_queries), 3):  # Every 3rd query
                query = test_queries[i]['query']
                response = f"Pre-cached response for: {query}"
                asyncio.run(cache.set(query, response, ttl=3600))
            
            # Test coordination performance
            for query_data in test_queries:
                query = query_data['query']
                
                start_time = time.time()
                result = asyncio.run(cache.get(query))
                coordination_time = (time.time() - start_time) * 1000
                
                coordination_times.append(coordination_time)
                
                if result is None:
                    cache_hits_by_tier['miss'] += 1
                    # Simulate and cache new response
                    response = f"Generated response for: {query}"
                    asyncio.run(cache.set(query, response, ttl=3600))
            
            # Get detailed cache statistics
            cache_stats = cache.get_performance_stats()
            
            coordination_results[scenario] = {
                'avg_coordination_time_ms': statistics.mean(coordination_times),
                'p95_coordination_time_ms': statistics.quantiles(coordination_times, n=20)[18],
                'cache_hit_distribution': cache_hits_by_tier,
                'total_hit_rate': cache_stats['hit_statistics']['hit_rate'],
                'l1_utilization': cache_stats['cache_sizes']['l1_size'] / cache_stats['cache_sizes']['l1_max_size'],
                'l2_utilization': cache_stats['cache_sizes']['l2_size'] / cache_stats['cache_sizes']['l2_max_size'],
                'meets_target': statistics.mean(coordination_times) <= BIOMEDICAL_PERFORMANCE_TARGETS['cache_coordination_overhead_ms']
            }
            
            cache.clear()
        
        return coordination_results
    
    def run_emergency_fallback_performance_test(
        self,
        cache: HighPerformanceCache,
        emergency_scenarios: List[str]
    ) -> Dict[str, Any]:
        """Run emergency fallback performance test."""
        
        classifier = BiomedicalQueryClassifier(cache)
        simulator = ClinicalWorkflowSimulator(cache, classifier)
        
        emergency_results = {}
        
        for scenario in emergency_scenarios:
            print(f"Testing {scenario} emergency scenario...")
            
            if scenario == 'cache_failure':
                # Simulate cache failures
                test_queries = [
                    "urgent glucose metabolism emergency",
                    "critical biomarker analysis needed",
                    "immediate diagnostic support required"
                ]
            elif scenario == 'high_load':
                # High load emergency queries
                test_queries = [f"emergency query {i}: urgent metabolomics analysis" for i in range(50)]
            elif scenario == 'network_latency':
                # Network latency simulation
                test_queries = [
                    "emergency pathway analysis",
                    "critical disease biomarker query",
                    "urgent method validation needed"
                ]
            else:  # general
                test_queries = [
                    "emergency metabolomics consultation",
                    "urgent biomarker interpretation",
                    "critical pathway analysis"
                ]
            
            fallback_times = []
            success_count = 0
            
            for query in test_queries:
                try:
                    fallback_time = asyncio.run(simulator._handle_emergency_fallback(query))
                    fallback_times.append(fallback_time)
                    success_count += 1
                except Exception as e:
                    fallback_times.append(1000)  # 1 second penalty for failures
            
            emergency_results[scenario] = {
                'total_queries': len(test_queries),
                'successful_queries': success_count,
                'success_rate': success_count / len(test_queries),
                'avg_fallback_time_ms': statistics.mean(fallback_times),
                'max_fallback_time_ms': max(fallback_times),
                'p95_fallback_time_ms': statistics.quantiles(fallback_times, n=20)[18],
                'meets_target': statistics.mean(fallback_times) <= BIOMEDICAL_PERFORMANCE_TARGETS['emergency_fallback_ms']
            }
        
        return emergency_results
    
    def _aggregate_biomedical_metrics(
        self,
        metrics_list: List[BiomedicalQueryPerformanceMetrics],
        workflow_type: str
    ) -> BiomedicalQueryPerformanceMetrics:
        """Aggregate multiple biomedical metrics."""
        
        if not metrics_list:
            raise ValueError("No metrics to aggregate")
        
        # Aggregate basic counts
        total_queries = sum(m.total_queries for m in metrics_list)
        successful_queries = sum(m.successful_queries for m in metrics_list)
        cache_hits = sum(m.cache_hits for m in metrics_list)
        cache_misses = sum(m.cache_misses for m in metrics_list)
        emergency_fallbacks = sum(m.emergency_fallback_invoked for m in metrics_list)
        
        # Aggregate special query counts
        cross_language_queries = sum(m.cross_language_queries for m in metrics_list)
        biomarker_queries = sum(m.biomarker_specific_queries for m in metrics_list)
        pathway_queries = sum(m.pathway_analysis_queries for m in metrics_list)
        
        # Aggregate timing metrics (weighted averages)
        total_weight = sum(m.total_queries for m in metrics_list)
        
        weighted_classification_time = sum(m.classification_time_ms * m.total_queries for m in metrics_list) / total_weight
        weighted_processing_time = sum(m.processing_time_ms * m.total_queries for m in metrics_list) / total_weight
        weighted_response_time = sum(m.total_response_time_ms * m.total_queries for m in metrics_list) / total_weight
        weighted_coordination_time = sum(m.cache_coordination_time_ms * m.total_queries for m in metrics_list) / total_weight
        
        # Aggregate percentiles (approximate)
        all_p50_times = [m.p50_response_time_ms for m in metrics_list]
        all_p95_times = [m.p95_response_time_ms for m in metrics_list]
        all_p99_times = [m.p99_response_time_ms for m in metrics_list]
        
        # Aggregate quality scores
        clinical_accuracy = statistics.mean([m.clinical_accuracy_score for m in metrics_list])
        knowledge_coverage = statistics.mean([m.knowledge_coverage_score for m in metrics_list])
        temporal_relevance = statistics.mean([m.temporal_relevance_score for m in metrics_list])
        
        return BiomedicalQueryPerformanceMetrics(
            workflow_type=workflow_type,
            query_category="aggregated",
            total_queries=total_queries,
            successful_queries=successful_queries,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            classification_time_ms=weighted_classification_time,
            processing_time_ms=weighted_processing_time,
            total_response_time_ms=weighted_response_time,
            cache_coordination_time_ms=weighted_coordination_time,
            emergency_fallback_invoked=emergency_fallbacks,
            cross_language_queries=cross_language_queries,
            biomarker_specific_queries=biomarker_queries,
            pathway_analysis_queries=pathway_queries,
            p50_response_time_ms=statistics.mean(all_p50_times),
            p95_response_time_ms=statistics.mean(all_p95_times),
            p99_response_time_ms=statistics.mean(all_p99_times),
            clinical_accuracy_score=clinical_accuracy,
            knowledge_coverage_score=knowledge_coverage,
            temporal_relevance_score=temporal_relevance
        )


class TestClinicalResearchWorkflowPerformance:
    """Tests for clinical research workflow performance with caching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = HighPerformanceCache(l1_size=3000, l2_size=15000, l3_enabled=True)
        self.test_runner = BiomedicalPerformanceTestRunner()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def test_diagnostic_workflow_performance(self):
        """Test diagnostic workflow performance with caching."""
        metrics = self.test_runner.run_clinical_workflow_performance_test(
            self.cache, WorkflowType.DIAGNOSTIC, iterations=3
        )
        
        print(f"\nDiagnostic Workflow Performance Results:")
        print(f"  Total queries: {metrics.total_queries}")
        print(f"  Success rate: {metrics.successful_queries / metrics.total_queries:.3f}")
        print(f"  Cache hit rate: {metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses):.3f}")
        print(f"  Average response time: {metrics.total_response_time_ms:.2f}ms")
        print(f"  P95 response time: {metrics.p95_response_time_ms:.2f}ms")
        print(f"  Classification time: {metrics.classification_time_ms:.2f}ms")
        print(f"  Clinical accuracy: {metrics.clinical_accuracy_score:.3f}")
        print(f"  Emergency fallbacks: {metrics.emergency_fallback_invoked}")
        
        # Diagnostic workflow specific validations
        assert metrics.meets_biomedical_targets(), "Diagnostic workflow should meet biomedical targets"
        assert metrics.total_response_time_ms <= BIOMEDICAL_PERFORMANCE_TARGETS['clinical_workflow_avg_ms'], \
            f"Diagnostic response time {metrics.total_response_time_ms:.2f}ms too high"
        assert metrics.clinical_accuracy_score >= 0.93, \
            f"Diagnostic accuracy {metrics.clinical_accuracy_score:.3f} below requirement"
        
        cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
        assert cache_hit_rate >= 0.75, f"Diagnostic hit rate {cache_hit_rate:.3f} too low"
    
    def test_research_workflow_performance(self):
        """Test research workflow performance with caching."""
        metrics = self.test_runner.run_clinical_workflow_performance_test(
            self.cache, WorkflowType.RESEARCH, iterations=3
        )
        
        print(f"\nResearch Workflow Performance Results:")
        print(f"  Total queries: {metrics.total_queries}")
        print(f"  Processing time: {metrics.processing_time_ms:.2f}ms")
        print(f"  Knowledge coverage: {metrics.knowledge_coverage_score:.3f}")
        print(f"  Biomarker queries: {metrics.biomarker_specific_queries}")
        print(f"  Pathway queries: {metrics.pathway_analysis_queries}")
        
        # Research workflow validations (more lenient on time, stricter on accuracy)
        assert metrics.meets_biomedical_targets(), "Research workflow should meet biomedical targets"
        assert metrics.knowledge_coverage_score >= 0.88, \
            f"Research knowledge coverage {metrics.knowledge_coverage_score:.3f} too low"
        assert metrics.biomarker_specific_queries > 0, "Should include biomarker queries"
        assert metrics.pathway_analysis_queries > 0, "Should include pathway analysis queries"
    
    def test_screening_workflow_performance(self):
        """Test screening workflow performance with caching."""
        metrics = self.test_runner.run_clinical_workflow_performance_test(
            self.cache, WorkflowType.SCREENING, iterations=3
        )
        
        print(f"\nScreening Workflow Performance Results:")
        print(f"  Total queries: {metrics.total_queries}")
        print(f"  Success rate: {metrics.successful_queries / metrics.total_queries:.3f}")
        print(f"  Average response time: {metrics.total_response_time_ms:.2f}ms")
        print(f"  Clinical accuracy: {metrics.clinical_accuracy_score:.3f}")
        
        # Screening workflow validations (standardized, should be efficient)
        assert metrics.meets_biomedical_targets(), "Screening workflow should meet biomedical targets"
        assert metrics.total_response_time_ms <= 180, \
            f"Screening response time {metrics.total_response_time_ms:.2f}ms should be fast"
        assert metrics.clinical_accuracy_score >= 0.90, \
            f"Screening accuracy {metrics.clinical_accuracy_score:.3f} adequate"
        
        # High hit rate expected for standardized screening
        cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
        assert cache_hit_rate >= 0.80, f"Screening hit rate {cache_hit_rate:.3f} should be high"
    
    def test_workflow_performance_comparison(self):
        """Compare performance across different workflow types."""
        workflow_types = [WorkflowType.DIAGNOSTIC, WorkflowType.RESEARCH, WorkflowType.SCREENING]
        workflow_results = {}
        
        for workflow_type in workflow_types:
            print(f"\nTesting {workflow_type.value} workflow...")
            
            # Use smaller cache for comparison test
            cache = HighPerformanceCache(l1_size=1500, l2_size=7500, l3_enabled=True)
            
            metrics = self.test_runner.run_clinical_workflow_performance_test(
                cache, workflow_type, iterations=2
            )
            
            workflow_results[workflow_type.value] = {
                'response_time': metrics.total_response_time_ms,
                'hit_rate': metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses),
                'success_rate': metrics.successful_queries / metrics.total_queries,
                'clinical_accuracy': metrics.clinical_accuracy_score,
                'meets_targets': metrics.meets_biomedical_targets()
            }
            
            cache.clear()
        
        print(f"\nWorkflow Performance Comparison:")
        print(f"{'Workflow':<15}{'Response':<12}{'Hit Rate':<10}{'Success':<10}{'Accuracy':<10}{'Targets':<8}")
        print("-" * 75)
        
        for workflow, result in workflow_results.items():
            print(f"{workflow:<15}{result['response_time']:<12.1f}{result['hit_rate']:<10.3f}"
                  f"{result['success_rate']:<10.3f}{result['clinical_accuracy']:<10.3f}"
                  f"{'✅' if result['meets_targets'] else '❌':<8}")
        
        # All workflows should meet targets
        for workflow, result in workflow_results.items():
            assert result['meets_targets'], f"{workflow} workflow failed to meet targets"
            assert result['success_rate'] >= 0.95, f"{workflow} success rate too low"


class TestQueryClassificationPerformance:
    """Tests for query classification performance with cache integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = HighPerformanceCache(l1_size=2000, l2_size=8000, l3_enabled=True)
        self.test_runner = BiomedicalPerformanceTestRunner()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def test_query_classification_speed(self):
        """Test query classification speed and accuracy."""
        results = self.test_runner.run_query_classification_performance_test(
            self.cache, query_count=1000
        )
        
        print(f"\nQuery Classification Performance Results:")
        print(f"  Total queries: {results['total_queries']}")
        print(f"  Total time: {results['total_time_seconds']:.2f}s")
        print(f"  Average classification time: {results['avg_classification_time_ms']:.2f}ms")
        print(f"  P95 classification time: {results['p95_classification_time_ms']:.2f}ms")
        print(f"  Throughput: {results['throughput_queries_per_second']:.0f} queries/sec")
        print(f"  Average confidence: {results['avg_confidence']:.3f}")
        print(f"  Category distribution: {results['category_distribution']}")
        
        # Classification performance validations
        assert results['meets_target'], \
            f"Classification time {results['avg_classification_time_ms']:.2f}ms exceeds target"
        assert results['avg_classification_time_ms'] <= BIOMEDICAL_PERFORMANCE_TARGETS['query_classification_ms'], \
            "Classification should be fast"
        assert results['throughput_queries_per_second'] >= 100, \
            f"Classification throughput {results['throughput_queries_per_second']:.0f} too low"
        assert results['avg_confidence'] >= 0.70, \
            f"Average confidence {results['avg_confidence']:.3f} too low"
    
    def test_classification_caching_effectiveness(self):
        """Test effectiveness of classification result caching."""
        classifier = BiomedicalQueryClassifier(self.cache)
        
        # Test queries for caching effectiveness
        test_queries = [
            "What is glucose metabolism?",
            "How does insulin work?",
            "What are diabetes biomarkers?",
            "What is glucose metabolism?",  # Repeat
            "How does insulin work?",      # Repeat
        ]
        
        classification_times = []
        
        # First pass - populate cache
        for query in test_queries:
            category, confidence, classification_time = asyncio.run(classifier.classify_query(query))
            classification_times.append(classification_time)
        
        # Check cache effectiveness
        repeat_query_times = classification_times[3:5]  # Times for repeated queries
        initial_query_times = classification_times[0:2]  # Times for initial queries
        
        avg_repeat_time = statistics.mean(repeat_query_times)
        avg_initial_time = statistics.mean(initial_query_times)
        
        print(f"\nClassification Caching Effectiveness:")
        print(f"  Initial query avg time: {avg_initial_time:.2f}ms")
        print(f"  Repeat query avg time: {avg_repeat_time:.2f}ms")
        print(f"  Speedup factor: {avg_initial_time / avg_repeat_time:.1f}x")
        
        # Caching should provide significant speedup
        assert avg_repeat_time < avg_initial_time, "Repeat queries should be faster"
        speedup_factor = avg_initial_time / avg_repeat_time
        assert speedup_factor >= 2.0, f"Speedup factor {speedup_factor:.1f}x too low"
    
    def test_classification_accuracy_across_categories(self):
        """Test classification accuracy across different biomedical categories."""
        classifier = BiomedicalQueryClassifier(self.cache)
        
        # Test specific category queries
        category_test_queries = {
            'metabolism': [
                "What is glycolysis pathway?",
                "How does cellular respiration work?",
                "What are metabolic enzymes?"
            ],
            'biomarkers': [
                "What biomarkers indicate heart disease?",
                "Which diagnostic markers show diabetes?",
                "What are cancer screening biomarkers?"
            ],
            'diseases': [
                "What causes diabetes mellitus?",
                "How does cancer develop?",
                "What is cardiovascular disease?"
            ],
            'methods': [
                "How does LC-MS work?",
                "What is NMR spectroscopy?",
                "How to perform GC-MS analysis?"
            ]
        }
        
        category_accuracy = {}
        
        for expected_category, queries in category_test_queries.items():
            correct_classifications = 0
            total_queries = len(queries)
            
            for query in queries:
                category, confidence, _ = asyncio.run(classifier.classify_query(query))
                
                # Check if classification matches expected category
                if expected_category in category.value.lower() or category.value.lower() in expected_category:
                    correct_classifications += 1
            
            accuracy = correct_classifications / total_queries
            category_accuracy[expected_category] = accuracy
            
            print(f"  {expected_category}: {accuracy:.3f} accuracy")
        
        print(f"\nClassification Accuracy by Category:")
        for category, accuracy in category_accuracy.items():
            print(f"  {category}: {accuracy:.3f}")
        
        # Accuracy validations
        overall_accuracy = statistics.mean(category_accuracy.values())
        assert overall_accuracy >= 0.70, f"Overall classification accuracy {overall_accuracy:.3f} too low"
        
        # Each category should have reasonable accuracy
        for category, accuracy in category_accuracy.items():
            assert accuracy >= 0.60, f"{category} accuracy {accuracy:.3f} too low"


class TestMultiTierCacheCoordinationPerformance:
    """Tests for multi-tier cache coordination performance impact."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_runner = BiomedicalPerformanceTestRunner()
        
    def test_cache_tier_coordination_overhead(self):
        """Test coordination overhead across different cache tier configurations."""
        # Create base cache for comparison
        base_cache = HighPerformanceCache(l1_size=2000, l2_size=8000, l3_enabled=True)
        
        coordination_scenarios = ['l1_optimized', 'l2_optimized', 'l3_enabled', 'balanced']
        
        results = self.test_runner.run_multi_tier_coordination_performance_test(
            base_cache, coordination_scenarios
        )
        
        print(f"\nMulti-Tier Cache Coordination Performance:")
        print(f"{'Scenario':<15}{'Coord Time':<12}{'Hit Rate':<10}{'L1 Util':<10}{'L2 Util':<10}{'Target':<8}")
        print("-" * 75)
        
        for scenario, result in results.items():
            print(f"{scenario:<15}{result['avg_coordination_time_ms']:<12.2f}"
                  f"{result['total_hit_rate']:<10.3f}{result['l1_utilization']:<10.3f}"
                  f"{result['l2_utilization']:<10.3f}{'✅' if result['meets_target'] else '❌':<8}")
        
        # Validate coordination performance
        for scenario, result in results.items():
            assert result['meets_target'], \
                f"{scenario} coordination time {result['avg_coordination_time_ms']:.2f}ms exceeds target"
            assert result['total_hit_rate'] >= 0.40, \
                f"{scenario} hit rate {result['total_hit_rate']:.3f} too low"
            assert result['avg_coordination_time_ms'] <= 50, \
                f"{scenario} coordination overhead too high"
        
        # L1 optimized should have lowest coordination time
        l1_time = results['l1_optimized']['avg_coordination_time_ms']
        for scenario, result in results.items():
            if scenario != 'l1_optimized':
                assert result['avg_coordination_time_ms'] >= l1_time * 0.8, \
                    "L1 optimization should provide coordination benefits"
    
    def test_cache_tier_performance_trade_offs(self):
        """Test performance trade-offs between different cache tier strategies."""
        # Test different tier allocation strategies
        tier_strategies = {
            'memory_heavy': HighPerformanceCache(l1_size=5000, l2_size=2000, l3_enabled=False),
            'disk_heavy': HighPerformanceCache(l1_size=1000, l2_size=10000, l3_enabled=False),
            'distributed': HighPerformanceCache(l1_size=2000, l2_size=4000, l3_enabled=True),
            'balanced': HighPerformanceCache(l1_size=2500, l2_size=6000, l3_enabled=True)
        }
        
        strategy_results = {}
        
        for strategy_name, cache in tier_strategies.items():
            print(f"\nTesting {strategy_name} tier strategy...")
            
            # Run biomedical query test
            test_queries = self.test_runner.data_generator.generate_batch(200, 'random')
            
            response_times = []
            cache_operations = 0
            
            # Pre-populate cache
            for i in range(0, len(test_queries), 4):  # Every 4th query
                query = test_queries[i]['query']
                response = f"Cached response for: {query}"
                asyncio.run(cache.set(query, response, ttl=3600))
            
            # Test query performance
            start_time = time.time()
            
            for query_data in test_queries:
                query = query_data['query']
                
                query_start = time.time()
                result = asyncio.run(cache.get(query))
                
                if result is None:
                    # Cache miss - simulate and cache response
                    response = f"Generated response for: {query}"
                    asyncio.run(cache.set(query, response, ttl=3600))
                
                query_time = (time.time() - query_start) * 1000
                response_times.append(query_time)
                cache_operations += 1
            
            total_time = time.time() - start_time
            
            # Get cache statistics
            cache_stats = cache.get_performance_stats()
            
            strategy_results[strategy_name] = {
                'avg_response_time_ms': statistics.mean(response_times),
                'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18],
                'throughput_ops_sec': cache_operations / total_time,
                'hit_rate': cache_stats['hit_statistics']['hit_rate'],
                'memory_efficiency': cache_stats['hit_statistics']['hit_rate'] / max(cache_stats['memory_usage']['overhead_mb'] / 100, 0.1),
                'l1_hit_rate': cache_stats['hit_statistics']['l1_hits'] / max(cache_operations, 1),
                'l2_hit_rate': cache_stats['hit_statistics']['l2_hits'] / max(cache_operations, 1)
            }
            
            cache.clear()
        
        print(f"\nTier Strategy Performance Comparison:")
        print(f"{'Strategy':<12}{'Resp Time':<12}{'Throughput':<12}{'Hit Rate':<10}{'L1 Hits':<10}{'L2 Hits':<10}")
        print("-" * 80)
        
        for strategy, result in strategy_results.items():
            print(f"{strategy:<12}{result['avg_response_time_ms']:<12.1f}"
                  f"{result['throughput_ops_sec']:<12.0f}{result['hit_rate']:<10.3f}"
                  f"{result['l1_hit_rate']:<10.3f}{result['l2_hit_rate']:<10.3f}")
        
        # Validate trade-offs
        for strategy, result in strategy_results.items():
            assert result['avg_response_time_ms'] <= 100, \
                f"{strategy} response time {result['avg_response_time_ms']:.1f}ms too high"
            assert result['hit_rate'] >= 0.30, \
                f"{strategy} hit rate {result['hit_rate']:.3f} too low"
        
        # Memory-heavy should have fast response times
        memory_heavy_time = strategy_results['memory_heavy']['avg_response_time_ms']
        assert memory_heavy_time <= 50, f"Memory-heavy strategy should be fast: {memory_heavy_time:.1f}ms"


class TestEmergencyFallbackPerformance:
    """Tests for emergency fallback system performance validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = HighPerformanceCache(l1_size=2000, l2_size=8000, l3_enabled=True)
        self.test_runner = BiomedicalPerformanceTestRunner()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def test_emergency_fallback_response_times(self):
        """Test emergency fallback response times."""
        emergency_scenarios = ['cache_failure', 'high_load', 'network_latency', 'general']
        
        results = self.test_runner.run_emergency_fallback_performance_test(
            self.cache, emergency_scenarios
        )
        
        print(f"\nEmergency Fallback Performance Results:")
        print(f"{'Scenario':<15}{'Queries':<8}{'Success':<10}{'Avg Time':<12}{'Max Time':<12}{'Target':<8}")
        print("-" * 75)
        
        for scenario, result in results.items():
            print(f"{scenario:<15}{result['total_queries']:<8}{result['success_rate']:<10.3f}"
                  f"{result['avg_fallback_time_ms']:<12.1f}{result['max_fallback_time_ms']:<12.1f}"
                  f"{'✅' if result['meets_target'] else '❌':<8}")
        
        # Emergency fallback validations
        for scenario, result in results.items():
            assert result['meets_target'], \
                f"{scenario} fallback time {result['avg_fallback_time_ms']:.1f}ms exceeds target"
            assert result['success_rate'] >= 0.90, \
                f"{scenario} success rate {result['success_rate']:.3f} too low for emergency"
            assert result['max_fallback_time_ms'] <= 1000, \
                f"{scenario} maximum fallback time too high"
        
        # Cache failure should still provide reasonable performance
        cache_failure_result = results['cache_failure']
        assert cache_failure_result['avg_fallback_time_ms'] <= 600, \
            "Cache failure fallback should be reasonably fast"
    
    def test_emergency_pattern_recognition(self):
        """Test emergency pattern recognition and response."""
        classifier = BiomedicalQueryClassifier(self.cache)
        simulator = ClinicalWorkflowSimulator(self.cache, classifier)
        
        # Test emergency pattern queries
        emergency_queries = [
            "urgent glucose analysis needed for diabetic patient",
            "critical biomarker results interpretation required",
            "emergency metabolomics consultation for acute case",
            "immediate pathway analysis for treatment decision",
            "stat metabolite identification needed"
        ]
        
        emergency_times = []
        emergency_categories = []
        
        for query in emergency_queries:
            start_time = time.time()
            
            # Classify as emergency
            category, confidence, classification_time = asyncio.run(classifier.classify_query(query))
            emergency_categories.append(category)
            
            # Handle as emergency
            fallback_time = asyncio.run(simulator._handle_emergency_fallback(query))
            
            total_time = (time.time() - start_time) * 1000
            emergency_times.append(total_time)
        
        avg_emergency_time = statistics.mean(emergency_times)
        max_emergency_time = max(emergency_times)
        
        print(f"\nEmergency Pattern Recognition Results:")
        print(f"  Emergency queries tested: {len(emergency_queries)}")
        print(f"  Average emergency response time: {avg_emergency_time:.2f}ms")
        print(f"  Maximum emergency response time: {max_emergency_time:.2f}ms")
        print(f"  Categories detected: {[cat.value for cat in emergency_categories]}")
        
        # Emergency pattern validations
        assert avg_emergency_time <= BIOMEDICAL_PERFORMANCE_TARGETS['emergency_fallback_ms'], \
            f"Emergency response time {avg_emergency_time:.2f}ms exceeds target"
        assert max_emergency_time <= 800, \
            f"Maximum emergency time {max_emergency_time:.2f}ms too high"
        
        # Should detect emergency patterns
        emergency_detected = any('emergency' in cat.value.lower() for cat in emergency_categories)
        assert emergency_detected or avg_emergency_time <= 400, \
            "Should either detect emergency patterns or provide fast responses"


class TestRealWorldBiomedicalScenarios:
    """Tests for real-world biomedical scenario performance validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = HighPerformanceCache(l1_size=5000, l2_size=20000, l3_enabled=True)
        self.test_runner = BiomedicalPerformanceTestRunner()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def test_conference_peak_load_scenario(self):
        """Test performance during conference/peak usage scenario."""
        print("\nSimulating conference peak load scenario...")
        
        # Simulate 150 attendees accessing during presentation
        concurrent_attendees = 150
        queries_per_attendee = 20
        
        # Generate conference-relevant queries
        conference_topics = [
            "metabolomics biomarker discovery",
            "pathway analysis methods",
            "clinical metabolomics applications",
            "biomarker validation techniques",
            "metabolite identification strategies"
        ]
        
        # Create workload
        attendee_workloads = []
        for attendee_id in range(concurrent_attendees):
            attendee_queries = []
            for _ in range(queries_per_attendee):
                topic = random.choice(conference_topics)
                query = f"{topic} in {random.choice(['diabetes', 'cancer', 'cardiovascular disease'])}"
                attendee_queries.append({'query': query, 'attendee_id': attendee_id})
            
            attendee_workloads.append(attendee_queries)
        
        # Track metrics
        all_response_times = []
        cache_operations = {'hits': 0, 'misses': 0}
        error_count = 0
        metrics_lock = threading.Lock()
        
        def attendee_worker(attendee_queries):
            """Worker for individual conference attendee."""
            for query_data in attendee_queries:
                query = query_data['query']
                
                try:
                    start_time = time.time()
                    
                    result = asyncio.run(self.cache.get(query))
                    
                    if result is None:
                        # Cache miss - simulate processing
                        time.sleep(0.05)  # 50ms processing
                        response = f"Conference response for: {query}"
                        asyncio.run(self.cache.set(query, response, ttl=1800))  # 30 min TTL
                        
                        with metrics_lock:
                            cache_operations['misses'] += 1
                    else:
                        with metrics_lock:
                            cache_operations['hits'] += 1
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    with metrics_lock:
                        all_response_times.append(response_time)
                        
                except Exception as e:
                    with metrics_lock:
                        nonlocal error_count
                        error_count += 1
        
        # Execute conference load test
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_attendees) as executor:
            futures = [executor.submit(attendee_worker, queries) for queries in attendee_workloads]
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        total_operations = len(all_response_times)
        success_rate = (total_operations - error_count) / total_operations if total_operations > 0 else 0
        cache_hit_rate = cache_operations['hits'] / (cache_operations['hits'] + cache_operations['misses'])
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        throughput = total_operations / duration if duration > 0 else 0
        
        if all_response_times:
            all_response_times.sort()
            p95_time = all_response_times[int(len(all_response_times) * 0.95)]
            max_time = max(all_response_times)
        else:
            p95_time = max_time = 0
        
        print(f"\nConference Peak Load Results:")
        print(f"  Concurrent attendees: {concurrent_attendees}")
        print(f"  Total queries: {total_operations}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Success rate: {success_rate:.3f}")
        print(f"  Cache hit rate: {cache_hit_rate:.3f}")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  P95 response time: {p95_time:.2f}ms")
        print(f"  Maximum response time: {max_time:.2f}ms")
        print(f"  Throughput: {throughput:.0f} queries/sec")
        print(f"  Errors: {error_count}")
        
        # Conference load validations
        assert success_rate >= 0.92, f"Conference success rate {success_rate:.3f} too low"
        assert cache_hit_rate >= 0.70, f"Conference hit rate {cache_hit_rate:.3f} should be high"
        assert avg_response_time <= 200, f"Conference response time {avg_response_time:.2f}ms too high"
        assert p95_time <= 500, f"Conference P95 time {p95_time:.2f}ms too high"
        assert error_count <= total_operations * 0.05, "Too many errors during conference"
    
    def test_long_running_research_session(self):
        """Test cache optimization during long-running research sessions."""
        print("\nSimulating long-running research session...")
        
        # Simulate 4-hour research session
        session_duration_minutes = 60  # Reduced for testing
        queries_per_minute = 3
        total_queries = session_duration_minutes * queries_per_minute
        
        # Research patterns: iterative refinement with some recurring themes
        research_themes = [
            "diabetes metabolomics biomarkers",
            "insulin resistance pathway analysis", 
            "glucose metabolism regulation",
            "diabetic complications metabolites",
            "pancreatic beta cell metabolism"
        ]
        
        response_times = []
        cache_operations = {'hits': 0, 'misses': 0}
        hit_rates_over_time = []
        
        print(f"  Simulating {total_queries} queries over {session_duration_minutes} minutes...")
        
        for query_idx in range(total_queries):
            # Simulate research pattern: 60% repeat themes, 40% new exploration
            if random.random() < 0.60 and query_idx > 10:
                # Repeat or refine previous themes
                theme = random.choice(research_themes)
                variation = random.choice(['mechanism', 'biomarkers', 'treatment', 'diagnosis', 'prognosis'])
                query = f"{theme} {variation}"
            else:
                # New exploration
                query_data = self.test_runner.data_generator.generate_query('metabolism')
                query = query_data['query']
            
            # Execute query
            start_time = time.time()
            
            result = asyncio.run(self.cache.get(query))
            
            if result is None:
                # Cache miss
                time.sleep(0.08)  # 80ms processing for research queries
                response = f"Research response for: {query}"
                
                # Longer TTL for research queries
                asyncio.run(self.cache.set(query, response, ttl=7200))  # 2 hours
                cache_operations['misses'] += 1
            else:
                cache_operations['hits'] += 1
            
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)
            
            # Track hit rate over time (every 30 queries)
            if (query_idx + 1) % 30 == 0:
                current_hit_rate = cache_operations['hits'] / (cache_operations['hits'] + cache_operations['misses'])
                hit_rates_over_time.append(current_hit_rate)
                print(f"    Progress: {query_idx + 1}/{total_queries}, Hit rate: {current_hit_rate:.3f}")
            
            # Simulate researcher think time
            time.sleep(0.02)  # 20ms between queries
        
        # Calculate session metrics
        final_hit_rate = cache_operations['hits'] / (cache_operations['hits'] + cache_operations['misses'])
        avg_response_time = statistics.mean(response_times)
        session_improvement = (hit_rates_over_time[-1] - hit_rates_over_time[0]) if len(hit_rates_over_time) > 1 else 0
        
        print(f"\nLong-Running Research Session Results:")
        print(f"  Session duration: {session_duration_minutes} minutes")
        print(f"  Total queries: {total_queries}")
        print(f"  Final cache hit rate: {final_hit_rate:.3f}")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  Hit rate improvement: {session_improvement:.3f}")
        print(f"  Hit rate progression: {[f'{rate:.3f}' for rate in hit_rates_over_time]}")
        
        # Research session validations
        assert final_hit_rate >= BIOMEDICAL_PERFORMANCE_TARGETS['research_session_hit_rate'], \
            f"Research session hit rate {final_hit_rate:.3f} below target"
        assert avg_response_time <= 250, \
            f"Research response time {avg_response_time:.2f}ms too high"
        assert session_improvement >= 0, \
            f"Hit rate should improve over session: {session_improvement:.3f}"
        
        # Hit rate should reach target by end of session
        if len(hit_rates_over_time) > 2:
            final_phase_hit_rate = statistics.mean(hit_rates_over_time[-2:])
            assert final_phase_hit_rate >= 0.75, \
                f"Final phase hit rate {final_phase_hit_rate:.3f} should be high"
    
    def test_cross_language_biomedical_queries(self):
        """Test performance with cross-language biomedical queries."""
        print("\nTesting cross-language biomedical query performance...")
        
        # Simulate multilingual biomedical queries
        multilingual_queries = [
            # English
            "glucose metabolism pathway analysis",
            "insulin resistance biomarkers",
            "diabetes metabolomics screening",
            # Spanish-influenced
            "glucosa metabolismo pathway",
            "insulina resistencia biomarkers", 
            "diabetes metabolomics screening",
            # German-influenced
            "glucose stoffwechsel pathway",
            "insulin resistenz biomarkers",
            "diabetes metabolomics screening",
            # French-influenced
            "glucose métabolisme pathway",
            "insuline résistance biomarkers",
            "diabète metabolomics screening"
        ]
        
        classifier = BiomedicalQueryClassifier(self.cache)
        
        cross_lang_times = []
        classification_results = []
        cache_operations = {'hits': 0, 'misses': 0}
        
        for query in multilingual_queries:
            start_time = time.time()
            
            # Classification
            category, confidence, classification_time = asyncio.run(classifier.classify_query(query))
            classification_results.append({'category': category, 'confidence': confidence})
            
            # Cache lookup
            result = asyncio.run(self.cache.get(query))
            
            if result is None:
                # Process and cache
                time.sleep(0.1)  # Cross-language processing overhead
                response = f"Multilingual response for: {query}"
                asyncio.run(self.cache.set(query, response, ttl=3600))
                cache_operations['misses'] += 1
            else:
                cache_operations['hits'] += 1
            
            total_time = (time.time() - start_time) * 1000
            cross_lang_times.append(total_time)
        
        # Calculate metrics
        avg_cross_lang_time = statistics.mean(cross_lang_times)
        hit_rate = cache_operations['hits'] / (cache_operations['hits'] + cache_operations['misses'])
        avg_confidence = statistics.mean([r['confidence'] for r in classification_results])
        
        category_distribution = Counter([r['category'].value for r in classification_results])
        
        print(f"\nCross-Language Query Performance Results:")
        print(f"  Multilingual queries tested: {len(multilingual_queries)}")
        print(f"  Average response time: {avg_cross_lang_time:.2f}ms")
        print(f"  Cache hit rate: {hit_rate:.3f}")
        print(f"  Average classification confidence: {avg_confidence:.3f}")
        print(f"  Category distribution: {dict(category_distribution)}")
        
        # Cross-language validations
        assert avg_cross_lang_time <= BIOMEDICAL_PERFORMANCE_TARGETS['cross_language_response_ms'], \
            f"Cross-language response time {avg_cross_lang_time:.2f}ms exceeds target"
        assert avg_confidence >= 0.65, \
            f"Cross-language confidence {avg_confidence:.3f} too low"
        
        # Should handle multilingual variations reasonably
        unique_categories = len(set([r['category'] for r in classification_results]))
        assert unique_categories <= 4, \
            f"Too many different categories ({unique_categories}) for similar multilingual queries"


# Performance test fixtures
@pytest.fixture
def biomedical_performance_runner():
    """Provide biomedical performance test runner."""
    return BiomedicalPerformanceTestRunner()


@pytest.fixture
def biomedical_query_classifier(high_performance_cache):
    """Provide biomedical query classifier."""
    return BiomedicalQueryClassifier(high_performance_cache)


@pytest.fixture
def clinical_workflow_simulator(high_performance_cache, biomedical_query_classifier):
    """Provide clinical workflow simulator."""
    return ClinicalWorkflowSimulator(high_performance_cache, biomedical_query_classifier)


@pytest.fixture
def high_performance_cache():
    """Provide high-performance cache for biomedical testing."""
    cache = HighPerformanceCache(l1_size=3000, l2_size=15000, l3_enabled=True)
    yield cache
    cache.clear()


# Pytest configuration for biomedical performance tests
def pytest_configure(config):
    """Configure pytest for biomedical performance testing."""
    config.addinivalue_line("markers", "biomedical_performance: mark test as biomedical performance test")
    config.addinivalue_line("markers", "clinical_workflow: mark test as clinical workflow test")
    config.addinivalue_line("markers", "real_world_scenario: mark test as real-world scenario test")


# Performance test markers
pytestmark = [
    pytest.mark.biomedical_performance,
    pytest.mark.performance,
    pytest.mark.slow
]


if __name__ == "__main__":
    # Run biomedical performance tests with appropriate configuration
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--timeout=1200",  # 20 minute timeout for comprehensive biomedical tests
        "-m", "biomedical_performance"
    ])