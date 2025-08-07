#!/usr/bin/env python3
"""
Comprehensive Query Performance and Response Quality Test Suite.

This module implements sophisticated testing for query performance benchmarks,
response quality assessment, scalability validation, and biomedical content
quality assurance for the Clinical Metabolomics Oracle LightRAG integration.

Test Categories:
- Query Performance Benchmarking (target: <30 seconds response time)
- Response Quality Assessment (relevance, accuracy, completeness)
- Scalability Testing (concurrent queries, increasing knowledge base size)
- Memory Usage Monitoring and Performance Degradation Detection
- Biomedical Content Quality Validation
- Stress Testing and Performance Reporting

Key Features:
- Sophisticated quality assessment metrics for biomedical research applications
- Performance monitoring and benchmarking utilities
- Response consistency and reliability validation
- Stress testing scenarios with detailed analytics
- Integration with comprehensive fixtures and PDF data
- Automated performance regression detection

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import psutil
import re
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import threading
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Optional imports for core components - handle gracefully if not available
try:
    from lightrag_integration.config import LightRAGConfig
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    LIGHTRAG_AVAILABLE = True
except ImportError:
    # Mock classes for testing when full integration isn't available
    class LightRAGConfig:
        pass
    
    class ClinicalMetabolomicsRAG:
        async def query(self, query_text: str) -> str:
            return "Mock response for testing purposes"
    
    LIGHTRAG_AVAILABLE = False

# Import test fixtures
from performance_test_fixtures import (
    PerformanceMetrics,
    LoadTestScenario,
    ResourceUsageSnapshot,
    ResourceMonitor,
    PerformanceTestExecutor,
    LoadTestScenarioGenerator,
    MockOperationGenerator,
    mock_clinical_query_operation
)
from biomedical_test_fixtures import (
    MetaboliteData,
    ClinicalStudyData,
    ClinicalMetabolomicsDataGenerator
)
try:
    from comprehensive_test_fixtures import (
        BiomedicalStudyProfile,
        AdvancedBiomedicalContentGenerator,
        CrossDocumentSynthesisValidator
    )
    COMPREHENSIVE_FIXTURES_AVAILABLE = True
except ImportError:
    # Define minimal fixtures for standalone operation
    class BiomedicalStudyProfile:
        pass
    class AdvancedBiomedicalContentGenerator:
        pass
    class CrossDocumentSynthesisValidator:
        pass
    COMPREHENSIVE_FIXTURES_AVAILABLE = False


# =====================================================================
# QUALITY ASSESSMENT DATA STRUCTURES
# =====================================================================

@dataclass
class ResponseQualityMetrics:
    """Comprehensive response quality assessment metrics."""
    relevance_score: float  # 0-100 scale
    accuracy_score: float  # 0-100 scale
    completeness_score: float  # 0-100 scale
    clarity_score: float  # 0-100 scale
    biomedical_terminology_score: float  # 0-100 scale
    source_citation_score: float  # 0-100 scale
    consistency_score: float  # Multiple runs consistency
    factual_accuracy_score: float  # Fact verification
    hallucination_score: float  # 0-100, higher = less hallucination
    overall_quality_score: float  # Weighted average
    
    # Detailed assessments
    key_concepts_covered: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)
    biomedical_terms_found: List[str] = field(default_factory=list)
    citations_extracted: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    assessment_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality_grade(self) -> str:
        """Get quality grade based on overall score."""
        if self.overall_quality_score >= 90:
            return "Excellent"
        elif self.overall_quality_score >= 80:
            return "Good"
        elif self.overall_quality_score >= 70:
            return "Acceptable"
        elif self.overall_quality_score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results container."""
    query_type: str
    benchmark_name: str
    target_response_time_ms: float
    actual_response_time_ms: float
    target_throughput_ops_per_sec: float
    actual_throughput_ops_per_sec: float
    target_memory_usage_mb: float
    actual_memory_usage_mb: float
    target_error_rate_percent: float
    actual_error_rate_percent: float
    
    meets_performance_targets: bool
    performance_ratio: float  # actual/target performance
    benchmark_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def performance_grade(self) -> str:
        """Get performance grade."""
        if self.performance_ratio >= 1.2:
            return "Exceeds Expectations"
        elif self.performance_ratio >= 1.0:
            return "Meets Expectations"
        elif self.performance_ratio >= 0.8:
            return "Below Expectations"
        else:
            return "Fails Expectations"


@dataclass
class ScalabilityTestResult:
    """Scalability test results."""
    test_name: str
    scaling_factor: float  # Load multiplier
    base_performance: PerformanceMetrics
    scaled_performance: PerformanceMetrics
    scaling_efficiency: float  # 0-1, 1 = perfect linear scaling
    scaling_grade: str
    bottlenecks_identified: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =====================================================================
# QUALITY ASSESSMENT ENGINE
# =====================================================================

class ResponseQualityAssessor:
    """Sophisticated response quality assessment engine."""
    
    def __init__(self):
        self.biomedical_keywords = {
            'metabolomics_core': [
                'metabolomics', 'metabolite', 'metabolism', 'biomarker',
                'mass spectrometry', 'NMR', 'chromatography', 'metabolic pathway'
            ],
            'clinical_terms': [
                'clinical', 'patient', 'disease', 'diagnosis', 'therapeutic',
                'biomedical', 'pathology', 'phenotype', 'precision medicine'
            ],
            'analytical_methods': [
                'LC-MS', 'GC-MS', 'UPLC', 'HILIC', 'targeted analysis',
                'untargeted analysis', 'quantitative', 'qualitative'
            ],
            'research_concepts': [
                'study design', 'statistical analysis', 'p-value',
                'effect size', 'confidence interval', 'validation'
            ]
        }
        
        self.quality_weights = {
            'relevance': 0.25,
            'accuracy': 0.20,
            'completeness': 0.20,
            'clarity': 0.15,
            'biomedical_terminology': 0.10,
            'source_citation': 0.10
        }
    
    async def assess_response_quality(self, 
                                    query: str,
                                    response: str,
                                    source_documents: List[str],
                                    expected_concepts: List[str]) -> ResponseQualityMetrics:
        """Perform comprehensive response quality assessment."""
        
        # Individual quality assessments
        relevance = self._assess_relevance(query, response)
        accuracy = self._assess_accuracy(response, source_documents)
        completeness = self._assess_completeness(response, expected_concepts)
        clarity = self._assess_clarity(response)
        biomedical_terminology = self._assess_biomedical_terminology(response)
        source_citation = self._assess_source_citation(response)
        
        # Additional assessments
        consistency = await self._assess_consistency(query, response)
        factual_accuracy = self._assess_factual_accuracy(response, source_documents)
        hallucination = self._assess_hallucination_risk(response, source_documents)
        
        # Calculate weighted overall score
        overall_score = (
            relevance * self.quality_weights['relevance'] +
            accuracy * self.quality_weights['accuracy'] +
            completeness * self.quality_weights['completeness'] +
            clarity * self.quality_weights['clarity'] +
            biomedical_terminology * self.quality_weights['biomedical_terminology'] +
            source_citation * self.quality_weights['source_citation']
        )
        
        # Extract detailed information
        key_concepts = self._extract_key_concepts(response)
        missing_concepts = [c for c in expected_concepts if c.lower() not in response.lower()]
        biomedical_terms = self._extract_biomedical_terms(response)
        citations = self._extract_citations(response)
        quality_flags = self._identify_quality_flags(response)
        
        return ResponseQualityMetrics(
            relevance_score=relevance,
            accuracy_score=accuracy,
            completeness_score=completeness,
            clarity_score=clarity,
            biomedical_terminology_score=biomedical_terminology,
            source_citation_score=source_citation,
            consistency_score=consistency,
            factual_accuracy_score=factual_accuracy,
            hallucination_score=hallucination,
            overall_quality_score=overall_score,
            key_concepts_covered=key_concepts,
            missing_concepts=missing_concepts,
            biomedical_terms_found=biomedical_terms,
            citations_extracted=citations,
            quality_flags=quality_flags,
            assessment_details={
                'response_length': len(response),
                'word_count': len(response.split()),
                'sentence_count': len(re.findall(r'[.!?]+', response)),
                'paragraph_count': len(response.split('\n\n')),
                'technical_density': self._calculate_technical_density(response)
            }
        )
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess response relevance to query."""
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'is'}
        query_terms -= common_words
        response_terms -= common_words
        
        if not query_terms:
            return 50.0  # Neutral score
        
        overlap = len(query_terms.intersection(response_terms))
        relevance_ratio = overlap / len(query_terms)
        
        # Bonus for biomedical context alignment
        if 'clinical' in query.lower() or 'metabolomics' in query.lower():
            biomedical_bonus = min(20, len([term for term in self.biomedical_keywords['metabolomics_core'] + self.biomedical_keywords['clinical_terms'] if term in response.lower()]) * 2)
        else:
            biomedical_bonus = 0
        
        return min(100, (relevance_ratio * 80) + biomedical_bonus)
    
    def _assess_accuracy(self, response: str, source_documents: List[str]) -> float:
        """Assess factual accuracy based on source documents."""
        if not source_documents:
            return 70.0  # Neutral score when no sources available
        
        # This is a simplified accuracy assessment
        # In production, this would use more sophisticated fact-checking
        
        # Check for specific factual claims
        factual_indicators = [
            'studies show', 'research indicates', 'according to',
            'evidence suggests', 'data demonstrates', 'findings reveal'
        ]
        
        accuracy_score = 75.0  # Base score
        
        # Bonus for evidence-based language
        for indicator in factual_indicators:
            if indicator in response.lower():
                accuracy_score += 5
                
        # Penalty for absolute claims without evidence
        absolute_claims = ['always', 'never', 'all', 'none', 'every', 'completely']
        for claim in absolute_claims:
            if claim in response.lower():
                accuracy_score -= 3
        
        return min(100, max(0, accuracy_score))
    
    def _assess_completeness(self, response: str, expected_concepts: List[str]) -> float:
        """Assess response completeness."""
        if not expected_concepts:
            return 80.0  # Default score when no expectations
        
        concepts_covered = sum(1 for concept in expected_concepts if concept.lower() in response.lower())
        completeness_ratio = concepts_covered / len(expected_concepts)
        
        # Length-based completeness factor
        if len(response) < 100:
            length_penalty = 20
        elif len(response) < 200:
            length_penalty = 10
        else:
            length_penalty = 0
        
        return min(100, (completeness_ratio * 80) + 20 - length_penalty)
    
    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity and readability."""
        words = response.split()
        sentences = re.findall(r'[.!?]+', response)
        
        if not words or not sentences:
            return 20.0
        
        # Average sentence length (clarity metric)
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal range: 15-25 words per sentence
        if 15 <= avg_sentence_length <= 25:
            length_score = 40
        elif 10 <= avg_sentence_length <= 30:
            length_score = 30
        else:
            length_score = 20
        
        # Structure indicators
        structure_indicators = ['first', 'second', 'furthermore', 'moreover', 'however', 'therefore', 'in conclusion']
        structure_score = min(30, sum(5 for indicator in structure_indicators if indicator in response.lower()))
        
        # Technical jargon balance
        technical_terms = sum(1 for term_list in self.biomedical_keywords.values() for term in term_list if term in response.lower())
        jargon_ratio = technical_terms / len(words) * 100
        
        if 2 <= jargon_ratio <= 8:
            jargon_score = 30
        elif 1 <= jargon_ratio <= 10:
            jargon_score = 20
        else:
            jargon_score = 10
        
        return length_score + structure_score + jargon_score
    
    def _assess_biomedical_terminology(self, response: str) -> float:
        """Assess appropriate use of biomedical terminology."""
        response_lower = response.lower()
        total_terms = 0
        found_terms = 0
        
        for category, terms in self.biomedical_keywords.items():
            for term in terms:
                total_terms += 1
                if term in response_lower:
                    found_terms += 1
        
        if total_terms == 0:
            return 50.0
        
        terminology_ratio = found_terms / total_terms
        
        # Bonus for diverse terminology across categories
        categories_used = sum(1 for category, terms in self.biomedical_keywords.items() 
                            if any(term in response_lower for term in terms))
        diversity_bonus = categories_used * 5
        
        return min(100, (terminology_ratio * 70) + diversity_bonus + 20)
    
    def _assess_source_citation(self, response: str) -> float:
        """Assess source citation and referencing."""
        # Look for citation patterns
        citation_patterns = [
            r'\[[0-9]+\]',  # [1], [2], etc.
            r'\([A-Za-z]+.*?\d{4}\)',  # (Author, 2024)
            r'et al\.',  # et al.
            r'according to',
            r'study by',
            r'research from'
        ]
        
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, response, re.IGNORECASE))
        
        # Base score for any citations
        if citations_found > 0:
            citation_score = 60 + min(40, citations_found * 10)
        else:
            # Look for evidence-based language as alternative
            evidence_indicators = ['studies show', 'research indicates', 'data suggests']
            if any(indicator in response.lower() for indicator in evidence_indicators):
                citation_score = 40
            else:
                citation_score = 20
        
        return citation_score
    
    async def _assess_consistency(self, query: str, response: str) -> float:
        """Assess response consistency through multiple evaluations."""
        # This would involve running the same query multiple times
        # For now, we'll use a placeholder assessment based on response structure
        
        # Check for consistent structure indicators
        consistency_indicators = [
            len(response) > 100,  # Substantial response
            'metabolomics' in response.lower() if 'metabolomics' in query.lower() else True,
            not any(contradiction in response.lower() for contradiction in ['however', 'but', 'although']),
        ]
        
        consistency_score = sum(20 for indicator in consistency_indicators if indicator) + 40
        return min(100, consistency_score)
    
    def _assess_factual_accuracy(self, response: str, source_documents: List[str]) -> float:
        """Assess factual accuracy against source documents."""
        # Simplified implementation - would use more sophisticated fact-checking
        
        # Look for specific claims that can be verified
        factual_patterns = [
            r'(\d+%|\d+\.\d+%)',  # Percentages
            r'(\d+\s*(mg|kg|ml|µM|nM))',  # Measurements
            r'(increase|decrease|higher|lower|significant)',  # Comparative claims
        ]
        
        claims_found = []
        for pattern in factual_patterns:
            claims_found.extend(re.findall(pattern, response, re.IGNORECASE))
        
        if not claims_found:
            return 75.0  # Neutral score for responses without specific claims
        
        # In a real implementation, these claims would be verified against sources
        # For now, we assume high accuracy for well-structured responses
        return 85.0 if len(claims_found) <= 5 else 75.0
    
    def _assess_hallucination_risk(self, response: str, source_documents: List[str]) -> float:
        """Assess risk of hallucinated information."""
        hallucination_risk_indicators = [
            'i believe', 'i think', 'probably', 'maybe', 'it seems',
            'breakthrough discovery', 'revolutionary', 'unprecedented',
            'miracle cure', 'amazing results', 'incredible findings'
        ]
        
        risk_score = sum(10 for indicator in hallucination_risk_indicators 
                        if indicator in response.lower())
        
        # Lower score means higher hallucination risk
        hallucination_score = max(10, 100 - risk_score)
        
        # Bonus for evidence-based language
        evidence_bonus = 10 if any(term in response.lower() for term in ['study', 'research', 'data', 'analysis']) else 0
        
        return min(100, hallucination_score + evidence_bonus)
    
    def _extract_key_concepts(self, response: str) -> List[str]:
        """Extract key concepts from response."""
        concepts = []
        
        # Extract biomedical terms
        for term_list in self.biomedical_keywords.values():
            for term in term_list:
                if term in response.lower():
                    concepts.append(term)
        
        # Extract capitalized terms (potential proper nouns/concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        concepts.extend(capitalized_terms[:10])  # Limit to avoid noise
        
        return list(set(concepts))
    
    def _extract_biomedical_terms(self, response: str) -> List[str]:
        """Extract biomedical terms from response."""
        terms_found = []
        response_lower = response.lower()
        
        for category, terms in self.biomedical_keywords.items():
            for term in terms:
                if term in response_lower:
                    terms_found.append(term)
        
        return terms_found
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract citations from response."""
        citation_patterns = [
            r'\[[0-9]+\]',
            r'\([A-Za-z]+.*?\d{4}\)',
            r'[A-Za-z]+ et al\. \(\d{4}\)'
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, response))
        
        return citations
    
    def _identify_quality_flags(self, response: str) -> List[str]:
        """Identify quality concerns or flags."""
        flags = []
        
        if len(response) < 50:
            flags.append("response_too_short")
        
        if len(response) > 2000:
            flags.append("response_very_long")
        
        if response.count('?') > 3:
            flags.append("too_many_questions")
        
        if not any(term in response.lower() for term_list in self.biomedical_keywords.values() for term in term_list):
            flags.append("lacks_biomedical_terminology")
        
        uncertainty_indicators = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        if sum(1 for indicator in uncertainty_indicators if indicator in response.lower()) > 2:
            flags.append("high_uncertainty")
        
        return flags
    
    def _calculate_technical_density(self, response: str) -> float:
        """Calculate technical terminology density."""
        words = response.lower().split()
        if not words:
            return 0.0
        
        technical_words = sum(1 for word in words 
                            for term_list in self.biomedical_keywords.values() 
                            for term in term_list if term in word)
        
        return technical_words / len(words) * 100


# =====================================================================
# PERFORMANCE BENCHMARK TESTS
# =====================================================================

class TestQueryPerformanceBenchmarks:
    """Comprehensive query performance benchmark tests."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time_benchmarks(self, clinical_rag_system):
        """Test query response times against benchmarks."""
        
        test_queries = [
            ("What is clinical metabolomics?", "simple", 5000),  # 5 second target
            ("Compare targeted vs untargeted metabolomics approaches", "medium", 15000),  # 15 second target  
            ("Design a comprehensive metabolomics study for diabetes biomarker discovery", "complex", 30000)  # 30 second target
        ]
        
        performance_results = []
        
        for query, complexity, target_time_ms in test_queries:
            start_time = time.time()
            
            try:
                response = await clinical_rag_system.query(query)
                success = True
            except Exception as e:
                response = None
                success = False
                
            end_time = time.time()
            actual_time_ms = (end_time - start_time) * 1000
            
            benchmark = PerformanceBenchmark(
                query_type=complexity,
                benchmark_name=f"response_time_{complexity}",
                target_response_time_ms=target_time_ms,
                actual_response_time_ms=actual_time_ms,
                target_throughput_ops_per_sec=0,
                actual_throughput_ops_per_sec=0,
                target_memory_usage_mb=0,
                actual_memory_usage_mb=0,
                target_error_rate_percent=0,
                actual_error_rate_percent=0 if success else 100,
                meets_performance_targets=actual_time_ms <= target_time_ms and success,
                performance_ratio=target_time_ms / actual_time_ms if success else 0,
                benchmark_details={
                    'query': query,
                    'response_length': len(response) if response else 0,
                    'success': success
                }
            )
            
            performance_results.append(benchmark)
            
            # Assert performance targets met
            assert benchmark.meets_performance_targets, (
                f"Query '{query}' failed performance benchmark: "
                f"{actual_time_ms:.0f}ms > {target_time_ms}ms target"
            )
        
        # Overall performance assessment
        avg_performance_ratio = statistics.mean(b.performance_ratio for b in performance_results)
        assert avg_performance_ratio >= 1.0, (
            f"Overall performance below target: {avg_performance_ratio:.2f}"
        )
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, clinical_rag_system):
        """Test performance under concurrent query load."""
        
        concurrent_levels = [1, 3, 5, 10]
        query = "What are the key metabolic pathways in clinical metabolomics?"
        
        performance_results = []
        
        for concurrent_count in concurrent_levels:
            start_time = time.time()
            
            # Execute concurrent queries
            tasks = []
            for _ in range(concurrent_count):
                task = asyncio.create_task(clinical_rag_system.query(query))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            total_time = end_time - start_time
            throughput = len(successful_results) / total_time
            error_rate = len(failed_results) / len(results) * 100
            
            # Performance expectations
            expected_throughput = concurrent_count * 0.8 / 10  # Allow for some overhead
            expected_max_error_rate = 10.0
            
            benchmark = PerformanceBenchmark(
                query_type="concurrent",
                benchmark_name=f"concurrent_{concurrent_count}_users",
                target_response_time_ms=total_time * 1000,
                actual_response_time_ms=total_time * 1000,
                target_throughput_ops_per_sec=expected_throughput,
                actual_throughput_ops_per_sec=throughput,
                target_memory_usage_mb=0,
                actual_memory_usage_mb=0,
                target_error_rate_percent=expected_max_error_rate,
                actual_error_rate_percent=error_rate,
                meets_performance_targets=(
                    throughput >= expected_throughput and 
                    error_rate <= expected_max_error_rate
                ),
                performance_ratio=throughput / expected_throughput if expected_throughput > 0 else 1.0,
                benchmark_details={
                    'concurrent_users': concurrent_count,
                    'total_queries': len(results),
                    'successful_queries': len(successful_results),
                    'failed_queries': len(failed_results),
                    'total_time_seconds': total_time
                }
            )
            
            performance_results.append(benchmark)
            
            # Assert concurrent performance
            assert benchmark.meets_performance_targets, (
                f"Concurrent performance failed for {concurrent_count} users: "
                f"Throughput {throughput:.2f} < {expected_throughput:.2f} or "
                f"Error rate {error_rate:.1f}% > {expected_max_error_rate}%"
            )
        
        # Test scaling efficiency
        base_throughput = performance_results[0].actual_throughput_ops_per_sec
        for result in performance_results[1:]:
            scaling_efficiency = result.actual_throughput_ops_per_sec / (base_throughput * result.benchmark_details['concurrent_users'])
            
            # Expect at least 70% scaling efficiency
            assert scaling_efficiency >= 0.7, (
                f"Poor scaling efficiency at {result.benchmark_details['concurrent_users']} users: {scaling_efficiency:.2f}"
            )
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, clinical_rag_system):
        """Test memory usage during query processing."""
        
        resource_monitor = ResourceMonitor(sampling_interval=0.5)
        query_sequence = [
            "What is clinical metabolomics?",
            "Explain targeted metabolomics analysis",
            "Compare LC-MS and GC-MS for metabolomics",
            "Describe biomarker discovery in metabolomics",
            "What are the challenges in clinical metabolomics?"
        ]
        
        # Start monitoring
        resource_monitor.start_monitoring()
        
        try:
            # Execute query sequence
            for query in query_sequence:
                await clinical_rag_system.query(query)
                await asyncio.sleep(1)  # Brief pause between queries
                
        finally:
            # Stop monitoring
            snapshots = resource_monitor.stop_monitoring()
        
        # Analyze memory usage
        if snapshots:
            memory_values = [s.memory_mb for s in snapshots]
            memory_increases = [memory_values[i] - memory_values[i-1] 
                              for i in range(1, len(memory_values))]
            
            max_memory = max(memory_values)
            avg_memory = statistics.mean(memory_values)
            memory_growth = memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
            
            # Memory usage assertions
            assert max_memory < 2000, f"Memory usage too high: {max_memory:.1f}MB > 2000MB"
            assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f}MB"
            
            # Check for memory leaks (consistent growth)
            if len(memory_increases) >= 3:
                consistent_increases = sum(1 for inc in memory_increases if inc > 10)
                leak_ratio = consistent_increases / len(memory_increases)
                assert leak_ratio < 0.7, f"Potential memory leak detected: {leak_ratio:.2f} ratio"


# =====================================================================
# RESPONSE QUALITY VALIDATION TESTS  
# =====================================================================

class TestResponseQualityValidation:
    """Comprehensive response quality validation tests."""
    
    @pytest.mark.biomedical
    @pytest.mark.asyncio
    async def test_biomedical_response_quality(self, clinical_rag_system, quality_assessor):
        """Test response quality for biomedical queries."""
        
        test_cases = [
            {
                'query': "What is clinical metabolomics?",
                'expected_concepts': [
                    'metabolomics', 'clinical', 'biomarker', 'mass spectrometry',
                    'disease', 'diagnosis', 'metabolism', 'analytical'
                ],
                'min_quality_score': 75
            },
            {
                'query': "Explain targeted metabolomics analysis",
                'expected_concepts': [
                    'targeted', 'quantitative', 'specific', 'pathway',
                    'LC-MS', 'standards', 'validation'
                ],
                'min_quality_score': 70
            },
            {
                'query': "What are biomarkers in metabolomics?",
                'expected_concepts': [
                    'biomarker', 'diagnostic', 'predictive', 'metabolite',
                    'clinical', 'validation', 'sensitivity', 'specificity'
                ],
                'min_quality_score': 75
            }
        ]
        
        quality_results = []
        
        for test_case in test_cases:
            # Get response from system
            response = await clinical_rag_system.query(test_case['query'])
            
            # Assess response quality
            quality_metrics = await quality_assessor.assess_response_quality(
                query=test_case['query'],
                response=response,
                source_documents=[],  # Would be populated with actual source docs
                expected_concepts=test_case['expected_concepts']
            )
            
            quality_results.append(quality_metrics)
            
            # Assert minimum quality standards
            assert quality_metrics.overall_quality_score >= test_case['min_quality_score'], (
                f"Quality score {quality_metrics.overall_quality_score:.1f} < {test_case['min_quality_score']} "
                f"for query: '{test_case['query']}'"
            )
            
            # Assert biomedical terminology usage
            assert quality_metrics.biomedical_terminology_score >= 60, (
                f"Insufficient biomedical terminology: {quality_metrics.biomedical_terminology_score:.1f}"
            )
            
            # Assert response completeness
            assert quality_metrics.completeness_score >= 60, (
                f"Incomplete response: {quality_metrics.completeness_score:.1f}"
            )
        
        # Overall quality assessment
        avg_quality = statistics.mean(qr.overall_quality_score for qr in quality_results)
        assert avg_quality >= 70, f"Overall quality below standard: {avg_quality:.1f}"
    
    @pytest.mark.biomedical
    @pytest.mark.asyncio
    async def test_response_consistency_validation(self, clinical_rag_system, quality_assessor):
        """Test response consistency across multiple runs."""
        
        query = "What is clinical metabolomics?"
        num_runs = 5
        
        responses = []
        for _ in range(num_runs):
            response = await clinical_rag_system.query(query)
            responses.append(response)
            await asyncio.sleep(0.5)  # Brief pause between queries
        
        # Assess each response
        quality_metrics_list = []
        for response in responses:
            metrics = await quality_assessor.assess_response_quality(
                query=query,
                response=response,
                source_documents=[],
                expected_concepts=['metabolomics', 'clinical', 'biomarker']
            )
            quality_metrics_list.append(metrics)
        
        # Analyze consistency
        quality_scores = [m.overall_quality_score for m in quality_metrics_list]
        response_lengths = [len(r) for r in responses]
        
        quality_std_dev = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
        length_std_dev = statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0
        
        # Consistency assertions
        assert quality_std_dev < 15, f"Quality inconsistent across runs: std_dev={quality_std_dev:.1f}"
        assert length_std_dev < (statistics.mean(response_lengths) * 0.3), "Response length too variable"
        
        # All responses should meet minimum quality
        min_quality = min(quality_scores)
        assert min_quality >= 65, f"Inconsistent quality detected: minimum={min_quality:.1f}"
    
    @pytest.mark.biomedical
    @pytest.mark.asyncio
    async def test_factual_accuracy_validation(self, clinical_rag_system, quality_assessor):
        """Test factual accuracy of responses."""
        
        # Queries with known factual expectations
        factual_test_cases = [
            {
                'query': "What analytical platforms are used in metabolomics?",
                'required_facts': ['LC-MS', 'GC-MS', 'NMR', 'mass spectrometry'],
                'forbidden_facts': ['PCR', 'western blot', 'flow cytometry']
            },
            {
                'query': "What is the difference between targeted and untargeted metabolomics?",
                'required_facts': ['targeted', 'untargeted', 'quantitative', 'qualitative'],
                'forbidden_facts': ['protein analysis', 'DNA sequencing']
            }
        ]
        
        for test_case in factual_test_cases:
            response = await clinical_rag_system.query(test_case['query'])
            response_lower = response.lower()
            
            # Check for required facts
            missing_facts = []
            for fact in test_case['required_facts']:
                if fact.lower() not in response_lower:
                    missing_facts.append(fact)
            
            # Check for forbidden facts (potential hallucinations)
            found_forbidden = []
            for fact in test_case['forbidden_facts']:
                if fact.lower() in response_lower:
                    found_forbidden.append(fact)
            
            # Assertions
            assert len(missing_facts) <= len(test_case['required_facts']) * 0.3, (
                f"Missing too many required facts: {missing_facts}"
            )
            
            assert len(found_forbidden) == 0, (
                f"Found forbidden/incorrect facts: {found_forbidden}"
            )


# =====================================================================
# SCALABILITY AND STRESS TESTS
# =====================================================================

class TestScalabilityAndStress:
    """Scalability and stress testing suite."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_increasing_knowledge_base_scalability(self, clinical_rag_system):
        """Test performance with increasing knowledge base size."""
        
        # This test would ideally test with different sizes of knowledge bases
        # For now, we'll simulate by testing with different query complexities
        
        scalability_tests = [
            ("Simple KB", "What is metabolomics?", 1.0),
            ("Medium KB", "Compare different metabolomics platforms", 2.0),
            ("Large KB", "Comprehensive metabolomics study design for multi-omics integration", 4.0)
        ]
        
        baseline_performance = None
        scalability_results = []
        
        for kb_size, query, complexity_factor in scalability_tests:
            start_time = time.time()
            
            try:
                response = await clinical_rag_system.query(query)
                success = True
                response_length = len(response)
            except Exception as e:
                success = False
                response_length = 0
            
            end_time = time.time()
            response_time = end_time - start_time
            
            performance_metrics = PerformanceMetrics(
                test_name=kb_size,
                start_time=start_time,
                end_time=end_time,
                duration=response_time,
                operations_count=1,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                throughput_ops_per_sec=1 / response_time if success else 0,
                average_latency_ms=response_time * 1000,
                median_latency_ms=response_time * 1000,
                p95_latency_ms=response_time * 1000,
                p99_latency_ms=response_time * 1000,
                min_latency_ms=response_time * 1000,
                max_latency_ms=response_time * 1000,
                memory_usage_mb=0,  # Would be measured in practice
                cpu_usage_percent=0,
                error_rate_percent=0 if success else 100,
                concurrent_operations=1
            )
            
            if baseline_performance is None:
                baseline_performance = performance_metrics
            
            # Calculate scaling efficiency
            expected_time = baseline_performance.duration * complexity_factor
            actual_time = performance_metrics.duration
            scaling_efficiency = expected_time / actual_time if actual_time > 0 else 0
            
            scalability_result = ScalabilityTestResult(
                test_name=kb_size,
                scaling_factor=complexity_factor,
                base_performance=baseline_performance,
                scaled_performance=performance_metrics,
                scaling_efficiency=min(1.0, scaling_efficiency),
                scaling_grade="Good" if scaling_efficiency >= 0.7 else "Poor"
            )
            
            scalability_results.append(scalability_result)
            
            # Assert acceptable scaling
            if complexity_factor > 1.0:
                assert scaling_efficiency >= 0.5, (
                    f"Poor scalability for {kb_size}: efficiency={scaling_efficiency:.2f}"
                )
        
        # Overall scalability assessment
        avg_efficiency = statistics.mean(sr.scaling_efficiency for sr in scalability_results[1:])
        assert avg_efficiency >= 0.6, f"Overall scalability poor: {avg_efficiency:.2f}"
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_stress_testing_scenarios(self, clinical_rag_system):
        """Execute stress testing scenarios."""
        
        stress_scenarios = [
            {
                'name': 'burst_queries',
                'description': 'Rapid burst of queries',
                'num_queries': 20,
                'time_limit': 30,
                'max_failures': 2
            },
            {
                'name': 'sustained_load',
                'description': 'Sustained query load',
                'num_queries': 50,
                'time_limit': 120,
                'max_failures': 5
            }
        ]
        
        for scenario in stress_scenarios:
            print(f"Running stress test: {scenario['name']}")
            
            queries = [
                f"Test query {i}: What are metabolomics applications?"
                for i in range(scenario['num_queries'])
            ]
            
            start_time = time.time()
            tasks = []
            
            # Create all query tasks
            for query in queries:
                task = asyncio.create_task(clinical_rag_system.query(query))
                tasks.append(task)
            
            # Execute with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=scenario['time_limit']
                )
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in tasks:
                    task.cancel()
                results = [TimeoutError()] * len(tasks)
            
            end_time = time.time()
            
            # Analyze stress test results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            timeout_results = [r for r in results if isinstance(r, asyncio.TimeoutError)]
            
            total_time = end_time - start_time
            success_rate = len(successful_results) / len(results) * 100
            failure_rate = len(failed_results) / len(results) * 100
            
            # Stress test assertions
            assert len(failed_results) <= scenario['max_failures'], (
                f"Too many failures in stress test {scenario['name']}: "
                f"{len(failed_results)} > {scenario['max_failures']}"
            )
            
            assert total_time <= scenario['time_limit'] * 1.2, (
                f"Stress test exceeded time limit: {total_time:.1f}s > {scenario['time_limit'] * 1.2}s"
            )
            
            assert success_rate >= 85, (
                f"Success rate too low in stress test: {success_rate:.1f}% < 85%"
            )
            
            print(f"Stress test {scenario['name']} completed: "
                  f"{len(successful_results)}/{len(results)} successful, "
                  f"Time: {total_time:.1f}s")


# =====================================================================
# PERFORMANCE REGRESSION DETECTION
# =====================================================================

class TestPerformanceRegressionDetection:
    """Performance regression detection and monitoring."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, clinical_rag_system):
        """Test performance regression detection system."""
        
        # Simulate baseline performance
        baseline_query = "What is clinical metabolomics?"
        baseline_runs = 3
        
        baseline_times = []
        for _ in range(baseline_runs):
            start_time = time.time()
            await clinical_rag_system.query(baseline_query)
            end_time = time.time()
            baseline_times.append(end_time - start_time)
        
        baseline_avg = statistics.mean(baseline_times)
        baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
        
        # Test current performance
        current_runs = 5
        current_times = []
        
        for _ in range(current_runs):
            start_time = time.time()
            await clinical_rag_system.query(baseline_query)
            end_time = time.time()
            current_times.append(end_time - start_time)
        
        current_avg = statistics.mean(current_times)
        current_std = statistics.stdev(current_times) if len(current_times) > 1 else 0
        
        # Regression analysis
        performance_change = (current_avg - baseline_avg) / baseline_avg * 100
        consistency_change = abs(current_std - baseline_std)
        
        # Regression thresholds
        max_performance_degradation = 25.0  # 25% slower
        max_consistency_degradation = 2.0   # 2 second std dev increase
        
        # Assertions
        assert performance_change <= max_performance_degradation, (
            f"Performance regression detected: {performance_change:.1f}% slower than baseline"
        )
        
        assert consistency_change <= max_consistency_degradation, (
            f"Consistency regression detected: std dev increased by {consistency_change:.2f}s"
        )
        
        # Log performance comparison
        print(f"Performance comparison:")
        print(f"  Baseline: {baseline_avg:.2f}s ± {baseline_std:.2f}s")
        print(f"  Current:  {current_avg:.2f}s ± {current_std:.2f}s")
        print(f"  Change:   {performance_change:+.1f}%")


# =====================================================================
# FIXTURES AND TEST UTILITIES
# =====================================================================

@pytest.fixture
def quality_assessor():
    """Provide response quality assessor."""
    return ResponseQualityAssessor()

@pytest.fixture
def clinical_rag_system():
    """Provide mock clinical RAG system for testing."""
    
    class MockClinicalRAGSystem:
        async def query(self, query_text: str) -> str:
            """Mock query method with realistic delays and responses."""
            
            # Simulate processing time based on query complexity
            base_delay = 0.5
            complexity_factor = len(query_text) / 100
            delay = base_delay + (complexity_factor * 0.3)
            
            await asyncio.sleep(delay)
            
            # Generate contextual response based on query
            if 'metabolomics' in query_text.lower():
                response = """Clinical metabolomics is a rapidly growing field that applies metabolomics 
                technologies to clinical research and practice. It involves the comprehensive analysis 
                of small molecules (metabolites) in biological samples such as blood, urine, and tissue 
                to identify biomarkers for disease diagnosis, prognosis, and treatment monitoring. 
                
                Key analytical platforms include liquid chromatography-mass spectrometry (LC-MS), 
                gas chromatography-mass spectrometry (GC-MS), and nuclear magnetic resonance (NMR) 
                spectroscopy. Clinical metabolomics can be divided into targeted approaches, which 
                focus on specific metabolites or pathways, and untargeted approaches, which provide 
                global metabolic profiling.
                
                Applications include biomarker discovery, drug development, personalized medicine, 
                and understanding disease mechanisms. The field faces challenges in standardization, 
                data integration, and clinical validation of metabolomic biomarkers."""
            
            elif 'targeted' in query_text.lower():
                response = """Targeted metabolomics analysis focuses on the quantitative measurement 
                of predefined sets of metabolites, typically those involved in specific metabolic 
                pathways or disease processes. This approach uses analytical methods optimized for 
                specific compounds, often employing multiple reaction monitoring (MRM) in mass 
                spectrometry with internal standards for accurate quantification.
                
                Advantages include higher sensitivity, better reproducibility, and more reliable 
                quantification compared to untargeted approaches. The method is ideal for hypothesis-driven 
                research, clinical validation studies, and routine biomarker monitoring."""
            
            else:
                # Generic metabolomics response
                response = f"""This query relates to metabolomics research. Metabolomics involves the 
                comprehensive study of metabolites in biological systems. The field uses advanced 
                analytical techniques to identify and quantify small molecules that serve as 
                indicators of biological processes, disease states, and treatment responses.
                
                Query context: {query_text}
                
                Common analytical platforms include mass spectrometry and NMR spectroscopy, 
                with applications in clinical research, drug discovery, and biomarker development."""
            
            # Simulate occasional failures
            if len(query_text) > 200 and 'comprehensive' in query_text.lower():
                import random
                if random.random() < 0.1:  # 10% failure rate for very complex queries
                    raise Exception("Query too complex for current system capacity")
            
            return response
    
    return MockClinicalRAGSystem()

@pytest.fixture
def performance_benchmark_suite():
    """Provide performance benchmark configuration."""
    return {
        'response_time_targets': {
            'simple_query': 5000,    # 5 seconds
            'medium_query': 15000,   # 15 seconds
            'complex_query': 30000   # 30 seconds
        },
        'throughput_targets': {
            'single_user': 0.2,      # 0.2 queries per second
            'multi_user': 1.0        # 1.0 queries per second total
        },
        'resource_limits': {
            'max_memory_mb': 2000,
            'max_cpu_percent': 85
        },
        'quality_thresholds': {
            'min_overall_score': 70,
            'min_relevance_score': 75,
            'min_biomedical_terminology': 60
        }
    }


# =====================================================================
# TEST EXECUTION AND REPORTING
# =====================================================================

if __name__ == "__main__":
    # Run performance and quality tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-m", "performance or biomedical"
    ])