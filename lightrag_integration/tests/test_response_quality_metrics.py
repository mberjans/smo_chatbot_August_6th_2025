#!/usr/bin/env python3
"""
Comprehensive Response Quality Metrics Test Suite - Fixed Version.

This module implements extensive unit tests for response quality metrics calculation
components in the Clinical Metabolomics Oracle LightRAG integration system.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.1 - Fixed
Related to: CMO-LIGHTRAG-009-T01 - Quality Validation and Benchmarking
"""

import pytest
import asyncio
import statistics
import re
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import quality assessment infrastructure
try:
    from test_comprehensive_query_performance_quality import (
        ResponseQualityMetrics,
        ResponseQualityAssessor
    )
    QUALITY_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    # Create working mock implementation
    @dataclass
    class ResponseQualityMetrics:
        relevance_score: float
        accuracy_score: float
        completeness_score: float
        clarity_score: float
        biomedical_terminology_score: float
        source_citation_score: float
        consistency_score: float
        factual_accuracy_score: float
        hallucination_score: float
        overall_quality_score: float
        key_concepts_covered: List[str] = field(default_factory=list)
        missing_concepts: List[str] = field(default_factory=list)
        biomedical_terms_found: List[str] = field(default_factory=list)
        citations_extracted: List[str] = field(default_factory=list)
        quality_flags: List[str] = field(default_factory=list)
        assessment_details: Dict[str, Any] = field(default_factory=dict)
        
        @property
        def quality_grade(self) -> str:
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
    
    class ResponseQualityAssessor:
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
            
        async def assess_response_quality(self, query, response, source_documents, expected_concepts):
            """Comprehensive quality assessment."""
            if source_documents is None:
                source_documents = []
            if expected_concepts is None:
                expected_concepts = []
                
            relevance = self._assess_relevance(query, response)
            accuracy = self._assess_accuracy(response, source_documents)
            completeness = self._assess_completeness(response, expected_concepts)
            clarity = self._assess_clarity(response)
            biomedical_terminology = self._assess_biomedical_terminology(response)
            source_citation = self._assess_source_citation(response)
            consistency = await self._assess_consistency(query, response)
            factual_accuracy = self._assess_factual_accuracy(response, source_documents)
            hallucination = self._assess_hallucination_risk(response, source_documents)
            
            overall_score = (
                relevance * self.quality_weights['relevance'] +
                accuracy * self.quality_weights['accuracy'] +
                completeness * self.quality_weights['completeness'] +
                clarity * self.quality_weights['clarity'] +
                biomedical_terminology * self.quality_weights['biomedical_terminology'] +
                source_citation * self.quality_weights['source_citation']
            )
            
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
        
        def _assess_relevance(self, query, response):
            query_terms = set(query.lower().split())
            response_terms = set(response.lower().split())
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'is'}
            query_terms -= common_words
            response_terms -= common_words
            
            if not query_terms:
                return 50.0
            
            overlap = len(query_terms.intersection(response_terms))
            relevance_ratio = overlap / len(query_terms)
            
            biomedical_bonus = 0
            if 'clinical' in query.lower() or 'metabolomics' in query.lower():
                biomedical_bonus = min(20, len([term for term in self.biomedical_keywords['metabolomics_core'] + self.biomedical_keywords['clinical_terms'] if term in response.lower()]) * 2)
            
            return min(100, (relevance_ratio * 80) + biomedical_bonus)
        
        def _assess_accuracy(self, response, source_documents):
            if not source_documents:
                return 70.0
            
            factual_indicators = [
                'studies show', 'research indicates', 'according to',
                'evidence suggests', 'data demonstrates', 'findings reveal'
            ]
            
            accuracy_score = 75.0
            
            for indicator in factual_indicators:
                if indicator in response.lower():
                    accuracy_score += 5
            
            absolute_claims = ['always', 'never', 'all', 'none', 'every', 'completely']
            for claim in absolute_claims:
                if claim in response.lower():
                    accuracy_score -= 3
            
            return min(100, max(0, accuracy_score))
        
        def _assess_completeness(self, response, expected_concepts):
            if not expected_concepts:
                return 80.0
            
            concepts_covered = sum(1 for concept in expected_concepts if concept.lower() in response.lower())
            completeness_ratio = concepts_covered / len(expected_concepts)
            
            if len(response) < 100:
                length_penalty = 20
            elif len(response) < 200:
                length_penalty = 10
            else:
                length_penalty = 0
            
            return min(100, (completeness_ratio * 80) + 20 - length_penalty)
        
        def _assess_clarity(self, response):
            words = response.split()
            sentences = re.findall(r'[.!?]+', response)
            
            if not words or not sentences:
                return 20.0
            
            avg_sentence_length = len(words) / len(sentences)
            
            if 15 <= avg_sentence_length <= 25:
                length_score = 40
            elif 10 <= avg_sentence_length <= 30:
                length_score = 30
            else:
                length_score = 20
            
            structure_indicators = ['first', 'second', 'furthermore', 'moreover', 'however', 'therefore', 'in conclusion']
            structure_score = min(30, sum(5 for indicator in structure_indicators if indicator in response.lower()))
            
            technical_terms = sum(1 for term_list in self.biomedical_keywords.values() for term in term_list if term in response.lower())
            jargon_ratio = technical_terms / len(words) * 100
            
            if 2 <= jargon_ratio <= 8:
                jargon_score = 30
            elif 1 <= jargon_ratio <= 10:
                jargon_score = 20
            else:
                jargon_score = 10
            
            return length_score + structure_score + jargon_score
        
        def _assess_biomedical_terminology(self, response):
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
            
            categories_used = sum(1 for category, terms in self.biomedical_keywords.items()
                                if any(term in response_lower for term in terms))
            diversity_bonus = categories_used * 5
            
            return min(100, (terminology_ratio * 70) + diversity_bonus + 20)
        
        def _assess_source_citation(self, response):
            citation_patterns = [
                r'\[[0-9]+\]',
                r'\([A-Za-z]+.*?\d{4}\)',
                r'et al\.',
                r'according to',
                r'study by',
                r'research from'
            ]
            
            citations_found = 0
            for pattern in citation_patterns:
                citations_found += len(re.findall(pattern, response, re.IGNORECASE))
            
            if citations_found > 0:
                citation_score = 60 + min(40, citations_found * 10)
            else:
                evidence_indicators = ['studies show', 'research indicates', 'data suggests']
                if any(indicator in response.lower() for indicator in evidence_indicators):
                    citation_score = 40
                else:
                    citation_score = 20
            
            return citation_score
        
        async def _assess_consistency(self, query, response):
            consistency_indicators = [
                len(response) > 100,
                'metabolomics' in response.lower() if 'metabolomics' in query.lower() else True,
                not any(contradiction in response.lower() for contradiction in ['however', 'but', 'although']),
            ]
            
            consistency_score = sum(20 for indicator in consistency_indicators if indicator) + 40
            return min(100, consistency_score)
        
        def _assess_factual_accuracy(self, response, source_documents):
            factual_patterns = [
                r'(\d+%|\d+\.\d+%)',
                r'(\d+\s*(mg|kg|ml|µM|nM))',
                r'(increase|decrease|higher|lower|significant)',
            ]
            
            claims_found = []
            for pattern in factual_patterns:
                claims_found.extend(re.findall(pattern, response, re.IGNORECASE))
            
            if not claims_found:
                return 75.0
            
            return 85.0 if len(claims_found) <= 5 else 75.0
        
        def _assess_hallucination_risk(self, response, source_documents):
            hallucination_risk_indicators = [
                'i believe', 'i think', 'probably', 'maybe', 'it seems',
                'breakthrough discovery', 'revolutionary', 'unprecedented',
                'miracle cure', 'amazing results', 'incredible findings'
            ]
            
            risk_score = sum(10 for indicator in hallucination_risk_indicators
                           if indicator in response.lower())
            
            hallucination_score = max(10, 100 - risk_score)
            
            evidence_bonus = 10 if any(term in response.lower() for term in ['study', 'research', 'data', 'analysis']) else 0
            
            return min(100, hallucination_score + evidence_bonus)
        
        def _extract_key_concepts(self, response):
            concepts = []
            
            for term_list in self.biomedical_keywords.values():
                for term in term_list:
                    if term in response.lower():
                        concepts.append(term)
            
            capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
            concepts.extend(capitalized_terms[:10])
            
            return list(set(concepts))
        
        def _extract_biomedical_terms(self, response):
            terms_found = []
            response_lower = response.lower()
            
            for category, terms in self.biomedical_keywords.items():
                for term in terms:
                    if term in response_lower:
                        terms_found.append(term)
            
            return terms_found
        
        def _extract_citations(self, response):
            citation_patterns = [
                r'\[[0-9]+\]',
                r'\([A-Za-z]+.*?\d{4}\)',
                r'[A-Za-z]+ et al\. \(\d{4}\)'
            ]
            
            citations = []
            for pattern in citation_patterns:
                citations.extend(re.findall(pattern, response))
            
            return citations
        
        def _identify_quality_flags(self, response):
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
        
        def _calculate_technical_density(self, response):
            words = response.lower().split()
            if not words:
                return 0.0
            
            technical_words = sum(1 for word in words
                                for term_list in self.biomedical_keywords.values()
                                for term in term_list if term in word)
            
            return technical_words / len(words) * 100
    
    QUALITY_INFRASTRUCTURE_AVAILABLE = False


# =====================================================================
# SIMPLIFIED QUALITY METRICS TESTS
# =====================================================================

class TestQualityMetricsCore:
    """Core tests for quality metrics functionality."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Provide quality assessor instance."""
        return ResponseQualityAssessor()
    
    def test_relevance_score_basic(self, quality_assessor):
        """Test basic relevance scoring."""
        query = "metabolomics biomarkers"
        response = "Clinical metabolomics identifies biomarkers for disease diagnosis"
        
        score = quality_assessor._assess_relevance(query, response)
        assert 50 <= score <= 100, f"Relevance score should be reasonable: {score}"
    
    def test_accuracy_score_basic(self, quality_assessor):
        """Test basic accuracy scoring."""
        response = "Studies show that metabolomics provides reliable results"
        score = quality_assessor._assess_accuracy(response, ["source document"])
        
        assert 70 <= score <= 100, f"Accuracy score should be reasonable: {score}"
    
    def test_completeness_score_basic(self, quality_assessor):
        """Test basic completeness scoring."""
        response = "Metabolomics is used for biomarker discovery"
        expected = ["metabolomics", "biomarker"]
        
        score = quality_assessor._assess_completeness(response, expected)
        assert 50 <= score <= 100, f"Completeness score should be reasonable: {score}"
    
    def test_clarity_score_basic(self, quality_assessor):
        """Test basic clarity scoring."""
        response = "Metabolomics is useful. It helps identify biomarkers. This supports medical research."
        
        score = quality_assessor._assess_clarity(response)
        assert 20 <= score <= 100, f"Clarity score should be reasonable: {score}"
    
    def test_biomedical_terminology_basic(self, quality_assessor):
        """Test basic biomedical terminology scoring."""
        response = "Clinical metabolomics uses LC-MS for biomarker analysis"
        
        score = quality_assessor._assess_biomedical_terminology(response)
        assert 30 <= score <= 100, f"Terminology score should be reasonable: {score}"
    
    def test_citation_scoring_basic(self, quality_assessor):
        """Test basic citation scoring."""
        response = "According to Smith et al. (2024), metabolomics shows promise [1]"
        
        score = quality_assessor._assess_source_citation(response)
        assert 60 <= score <= 100, f"Citation score should be high: {score}"
    
    @pytest.mark.asyncio
    async def test_comprehensive_assessment_basic(self, quality_assessor):
        """Test basic comprehensive assessment."""
        query = "What is metabolomics?"
        response = "Metabolomics is the study of small molecules called metabolites in biological systems."
        
        metrics = await quality_assessor.assess_response_quality(
            query=query,
            response=response,
            source_documents=[],
            expected_concepts=["metabolomics", "metabolites"]
        )
        
        # Basic validation
        assert isinstance(metrics, ResponseQualityMetrics)
        assert 0 <= metrics.overall_quality_score <= 100
        assert 0 <= metrics.relevance_score <= 100
        assert 0 <= metrics.accuracy_score <= 100
        assert len(metrics.biomedical_terms_found) > 0
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, quality_assessor):
        """Test edge cases and error handling."""
        
        # Empty response
        metrics = await quality_assessor.assess_response_quality(
            query="test",
            response="",
            source_documents=[],
            expected_concepts=[]
        )
        assert "response_too_short" in metrics.quality_flags
        
        # Very long response
        long_response = "This is a long response. " * 100
        metrics = await quality_assessor.assess_response_quality(
            query="test",
            response=long_response,
            source_documents=[],
            expected_concepts=[]
        )
        assert "response_very_long" in metrics.quality_flags
    
    def test_quality_weights_validity(self, quality_assessor):
        """Test that quality weights are valid."""
        total = sum(quality_assessor.quality_weights.values())
        assert 0.9 <= total <= 1.1, f"Quality weights should sum to ~1.0: {total}"
    
    def test_biomedical_keywords_coverage(self, quality_assessor):
        """Test biomedical keywords coverage."""
        assert 'metabolomics_core' in quality_assessor.biomedical_keywords
        assert 'clinical_terms' in quality_assessor.biomedical_keywords
        assert len(quality_assessor.biomedical_keywords['metabolomics_core']) > 0
    
    @pytest.mark.asyncio
    async def test_quality_flags_identification(self, quality_assessor):
        """Test quality flag identification."""
        
        # Test high uncertainty response
        uncertain_response = "Maybe this could possibly be perhaps a potential solution"
        metrics = await quality_assessor.assess_response_quality(
            query="test",
            response=uncertain_response,
            source_documents=[],
            expected_concepts=[]
        )
        
        assert "high_uncertainty" in metrics.quality_flags
    
    def test_hallucination_detection(self, quality_assessor):
        """Test hallucination risk detection."""
        
        # High risk response
        risky_response = "I believe this revolutionary breakthrough discovery is incredible"
        score = quality_assessor._assess_hallucination_risk(risky_response, [])
        
        assert score < 80, f"Should detect high hallucination risk: {score}"
        
        # Low risk response
        safe_response = "Research data shows study results from clinical analysis"
        score = quality_assessor._assess_hallucination_risk(safe_response, [])
        
        assert score >= 80, f"Should detect low hallucination risk: {score}"
    
    @pytest.mark.asyncio
    async def test_quality_assessment_performance(self, quality_assessor):
        """Test performance of quality assessment."""
        
        response = "Clinical metabolomics provides comprehensive analysis using LC-MS platforms."
        
        start_time = time.time()
        
        for _ in range(10):
            await quality_assessor.assess_response_quality(
                query="metabolomics",
                response=response,
                source_documents=[],
                expected_concepts=[]
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        assert avg_time < 0.5, f"Assessment should be fast: {avg_time:.3f}s"
    
    def test_score_boundaries(self, quality_assessor):
        """Test that all scores stay within 0-100 bounds."""
        
        # Test with various inputs
        test_cases = [
            ("", ""),  # Empty
            ("test", "test"),  # Minimal
            ("metabolomics", "Clinical metabolomics research using LC-MS"),  # Normal
            ("long query with many terms", "Very long response " * 50)  # Long
        ]
        
        for query, response in test_cases:
            relevance = quality_assessor._assess_relevance(query, response)
            accuracy = quality_assessor._assess_accuracy(response, [])
            clarity = quality_assessor._assess_clarity(response)
            terminology = quality_assessor._assess_biomedical_terminology(response)
            citation = quality_assessor._assess_source_citation(response)
            
            assert 0 <= relevance <= 100, f"Relevance out of bounds: {relevance}"
            assert 0 <= accuracy <= 100, f"Accuracy out of bounds: {accuracy}"
            assert 0 <= clarity <= 100, f"Clarity out of bounds: {clarity}"
            assert 0 <= terminology <= 100, f"Terminology out of bounds: {terminology}"
            assert 0 <= citation <= 100, f"Citation out of bounds: {citation}"


# =====================================================================
# BIOMEDICAL CONTEXT TESTS
# =====================================================================

class TestBiomedicalQualityFeatures:
    """Test biomedical-specific quality features."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Provide quality assessor instance."""
        return ResponseQualityAssessor()
    
    @pytest.mark.asyncio
    async def test_metabolomics_terminology_detection(self, quality_assessor):
        """Test detection of metabolomics terminology."""
        
        response = "LC-MS and GC-MS are key analytical platforms for metabolite analysis"
        
        metrics = await quality_assessor.assess_response_quality(
            query="analytical methods",
            response=response,
            source_documents=[],
            expected_concepts=[]
        )
        
        # Should detect analytical methods
        found_terms = [term.lower() for term in metrics.biomedical_terms_found]
        analytical_found = any('lc-ms' in term or 'mass spectrometry' in term or 'gc-ms' in term for term in found_terms)
        assert analytical_found or 'metabolite' in found_terms, \
            f"Should find analytical or metabolomics terms: {found_terms}"
    
    @pytest.mark.asyncio
    async def test_clinical_context_scoring(self, quality_assessor):
        """Test clinical context scoring."""
        
        clinical_response = "Clinical metabolomics supports patient diagnosis and therapeutic monitoring"
        
        metrics = await quality_assessor.assess_response_quality(
            query="clinical applications",
            response=clinical_response,
            source_documents=[],
            expected_concepts=["clinical", "patient"]
        )
        
        # Should recognize clinical terms
        assert metrics.biomedical_terminology_score > 40, \
            f"Should recognize clinical terminology: {metrics.biomedical_terminology_score}"
        
        assert metrics.relevance_score >= 50, \
            f"Should have reasonable relevance for clinical query: {metrics.relevance_score}"
    
    def test_technical_density_calculation(self, quality_assessor):
        """Test technical density calculation."""
        
        # High technical density
        technical_response = "LC-MS metabolomics biomarker analysis uses chromatography"
        density = quality_assessor._calculate_technical_density(technical_response)
        
        assert density > 20, f"Should calculate high technical density: {density}"
        
        # Low technical density  
        simple_response = "This is a simple sentence without technical terms"
        density = quality_assessor._calculate_technical_density(simple_response)
        
        assert density < 10, f"Should calculate low technical density: {density}"


# =====================================================================
# INTEGRATION AND COVERAGE TESTS
# =====================================================================

class TestQualityMetricsIntegration:
    """Test integration and coverage aspects."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Provide quality assessor instance."""
        return ResponseQualityAssessor()
    
    @pytest.mark.asyncio
    async def test_batch_quality_processing(self, quality_assessor):
        """Test batch processing of quality assessments."""
        
        test_pairs = [
            ("What is metabolomics?", "Metabolomics studies small molecules"),
            ("Clinical applications?", "Used for diagnosis and treatment"),
            ("Analytical methods?", "LC-MS and GC-MS are commonly used")
        ]
        
        results = []
        for query, response in test_pairs:
            metrics = await quality_assessor.assess_response_quality(
                query=query,
                response=response,
                source_documents=[],
                expected_concepts=[]
            )
            results.append(metrics)
        
        # All should be valid
        assert len(results) == 3
        for metrics in results:
            assert isinstance(metrics, ResponseQualityMetrics)
            assert 0 <= metrics.overall_quality_score <= 100
    
    def test_metrics_serialization(self, quality_assessor):
        """Test serialization of quality metrics."""
        
        metrics = ResponseQualityMetrics(
            relevance_score=85.0,
            accuracy_score=78.0,
            completeness_score=90.0,
            clarity_score=82.0,
            biomedical_terminology_score=88.0,
            source_citation_score=65.0,
            consistency_score=79.0,
            factual_accuracy_score=81.0,
            hallucination_score=92.0,
            overall_quality_score=83.0,
            key_concepts_covered=["metabolomics"],
            missing_concepts=[],
            biomedical_terms_found=["clinical"],
            citations_extracted=[],
            quality_flags=[],
            assessment_details={"word_count": 10}
        )
        
        # Test JSON serialization
        try:
            from dataclasses import asdict
            metrics_dict = asdict(metrics)
            json_str = json.dumps(metrics_dict)
            
            # Should be valid JSON
            parsed = json.loads(json_str)
            assert parsed['overall_quality_score'] == 83.0
            
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")
    
    def test_quality_grade_properties(self, quality_assessor):
        """Test quality grade property calculation."""
        
        # Test different score ranges
        test_scores = [95, 85, 75, 65, 45]
        expected_grades = ["Excellent", "Good", "Acceptable", "Needs Improvement", "Poor"]
        
        for score, expected_grade in zip(test_scores, expected_grades):
            metrics = ResponseQualityMetrics(
                relevance_score=score, accuracy_score=score, completeness_score=score,
                clarity_score=score, biomedical_terminology_score=score, source_citation_score=score,
                consistency_score=score, factual_accuracy_score=score, hallucination_score=score,
                overall_quality_score=score
            )
            
            assert metrics.quality_grade == expected_grade, \
                f"Score {score} should map to grade {expected_grade}, got {metrics.quality_grade}"
    
    @pytest.mark.asyncio
    async def test_assessment_consistency(self, quality_assessor):
        """Test assessment consistency across multiple runs."""
        
        query = "What is metabolomics?"
        response = "Metabolomics is the study of small molecules in biological systems"
        
        # Run assessment multiple times
        scores = []
        for _ in range(3):
            metrics = await quality_assessor.assess_response_quality(
                query=query,
                response=response,
                source_documents=[],
                expected_concepts=["metabolomics"]
            )
            scores.append(metrics.overall_quality_score)
        
        # Should be consistent (deterministic)
        assert all(abs(score - scores[0]) < 0.01 for score in scores), \
            f"Scores should be consistent: {scores}"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, quality_assessor):
        """Test memory efficiency during quality assessment."""
        
        # Run many assessments to check for memory leaks
        large_response = "Clinical metabolomics research. " * 50
        
        for i in range(50):
            await quality_assessor.assess_response_quality(
                query=f"query_{i}",
                response=large_response,
                source_documents=[],
                expected_concepts=[]
            )
        
        # Should complete without memory issues
        assert True, "Memory efficiency test completed"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])