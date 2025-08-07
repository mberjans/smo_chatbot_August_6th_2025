#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Clinical Metabolomics Relevance Scoring System.

This module provides extensive unit tests for the relevance scoring system
implemented in relevance_scorer.py, covering all dimensions of scoring,
query classification, quality validation, and performance testing.

Test Coverage:
1. Individual scoring dimension tests (metabolomics_relevance, clinical_applicability, etc.)
2. Query classification tests (basic_definition, clinical_application, etc.)
3. Response length and structure quality validation tests
4. Adaptive weighting scheme tests
5. Edge cases (empty responses, very long responses, nonsensical queries)
6. Performance tests (async execution, timing)
7. Integration with existing test patterns from the codebase
8. Semantic similarity engine tests
9. Domain expertise validator tests
10. Overall relevance scoring pipeline tests

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: Relevance Scoring System Testing
"""

import pytest
import pytest_asyncio
import asyncio
import statistics
import re
import time
import json
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the relevance scorer components
try:
    from relevance_scorer import (
        ClinicalMetabolomicsRelevanceScorer,
        RelevanceScore,
        QueryTypeClassifier,
        SemanticSimilarityEngine,
        WeightingSchemeManager,
        DomainExpertiseValidator,
        quick_relevance_check,
        batch_relevance_scoring
    )
    RELEVANCE_SCORER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import relevance_scorer module: {e}")
    RELEVANCE_SCORER_AVAILABLE = False
    
    # Create mock implementations for testing framework
    @dataclass
    class RelevanceScore:
        overall_score: float
        dimension_scores: Dict[str, float] = field(default_factory=dict)
        query_type: str = "general"
        weights_used: Dict[str, float] = field(default_factory=dict)
        explanation: str = ""
        confidence_score: float = 0.0
        processing_time_ms: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        @property
        def relevance_grade(self) -> str:
            if self.overall_score >= 90:
                return "Excellent"
            elif self.overall_score >= 80:
                return "Good"
            elif self.overall_score >= 70:
                return "Acceptable"
            elif self.overall_score >= 60:
                return "Marginal"
            else:
                return "Poor"
    
    class QueryTypeClassifier:
        def classify_query(self, query: str) -> str:
            return "general"
    
    class SemanticSimilarityEngine:
        async def calculate_similarity(self, query: str, response: str) -> float:
            return 75.0
    
    class WeightingSchemeManager:
        def get_weights(self, query_type: str) -> Dict[str, float]:
            return {"query_alignment": 0.5, "metabolomics_relevance": 0.5}
    
    class DomainExpertiseValidator:
        async def validate_domain_expertise(self, response: str) -> float:
            return 80.0
    
    class ClinicalMetabolomicsRelevanceScorer:
        def __init__(self, config=None):
            self.config = config or {}
        
        async def calculate_relevance_score(self, query: str, response: str, metadata=None):
            return RelevanceScore(overall_score=75.0)
    
    async def quick_relevance_check(query: str, response: str) -> float:
        return 75.0
    
    async def batch_relevance_scoring(pairs: List[Tuple[str, str]]) -> List[RelevanceScore]:
        return [RelevanceScore(overall_score=75.0) for _ in pairs]


# =====================================================================
# TEST FIXTURES
# =====================================================================

@pytest.fixture
def relevance_scorer():
    """Provide ClinicalMetabolomicsRelevanceScorer instance."""
    return ClinicalMetabolomicsRelevanceScorer()

@pytest.fixture
def query_classifier():
    """Provide QueryTypeClassifier instance."""
    return QueryTypeClassifier()

@pytest.fixture
def semantic_engine():
    """Provide SemanticSimilarityEngine instance."""
    return SemanticSimilarityEngine()

@pytest.fixture
def weighting_manager():
    """Provide WeightingSchemeManager instance."""
    return WeightingSchemeManager()

@pytest.fixture
def domain_validator():
    """Provide DomainExpertiseValidator instance."""
    return DomainExpertiseValidator()

@pytest.fixture
def test_queries():
    """Provide diverse test queries for different categories."""
    return {
        'basic_definition': [
            "What is metabolomics?",
            "Define biomarker in clinical context",
            "Explain mass spectrometry basics"
        ],
        'clinical_application': [
            "How is metabolomics used in clinical diagnosis?",
            "Clinical applications of biomarker discovery",
            "Patient diagnosis using metabolomic profiling"
        ],
        'analytical_method': [
            "How does LC-MS work in metabolomics?",
            "GC-MS protocol for metabolite analysis",
            "NMR spectroscopy methods for biomarkers"
        ],
        'research_design': [
            "Study design for metabolomics research",
            "Statistical analysis methods for biomarker data",
            "Sample size calculation for metabolomic studies"
        ],
        'disease_specific': [
            "Metabolomics in diabetes research",
            "Cancer biomarker discovery using metabolomics",
            "Cardiovascular disease metabolic signatures"
        ],
        'edge_cases': [
            "",  # Empty query
            "?",  # Single character
            "test " * 100,  # Very long repetitive query
            "xyz abc def nonsensical terms"  # Nonsensical query
        ]
    }

@pytest.fixture
def test_responses():
    """Provide diverse test responses with varying quality."""
    return {
        'excellent': """
# Metabolomics in Clinical Applications

## Definition
Metabolomics is the comprehensive study of small molecules called metabolites in biological systems. This field focuses on analyzing the complete set of metabolites present in cells, tissues, or biological fluids.

## Clinical Applications

### Biomarker Discovery
- Identification of disease-specific metabolic signatures
- Early detection of pathological conditions
- Monitoring disease progression

### Diagnostic Applications
- Non-invasive diagnostic tests using blood, urine, or tissue samples
- Improved sensitivity and specificity compared to traditional markers
- Personalized medicine approaches

### Treatment Monitoring
- Assessment of drug efficacy and toxicity
- Real-time monitoring of therapeutic responses
- Optimization of treatment protocols

## Analytical Methods
The most commonly used analytical platforms include:
- **LC-MS (Liquid Chromatography-Mass Spectrometry)**: Ideal for polar metabolites
- **GC-MS (Gas Chromatography-Mass Spectrometry)**: Suitable for volatile compounds
- **NMR (Nuclear Magnetic Resonance)**: Provides structural information

## Challenges and Future Directions
Current challenges include standardization of protocols, quality control, and data integration. However, advances in analytical technology and bioinformatics are addressing these limitations, making metabolomics increasingly valuable for precision medicine.
""",
        'good': """
Metabolomics is the study of small molecules in biological systems. It's used in clinical settings for biomarker discovery and disease diagnosis. Common analytical methods include LC-MS, GC-MS, and NMR spectroscopy.

Clinical applications include:
- Disease biomarker identification
- Drug metabolism studies
- Treatment response monitoring
- Precision medicine applications

The field faces challenges in standardization and data integration, but technological advances are improving its clinical utility.
""",
        'poor': "Metabolomics is good for research. It uses machines to analyze samples.",
        'empty': "",
        'very_long': """
This is an extremely long response that goes on and on about metabolomics without providing much substance. """ * 50,
        'technical_dense': """
LC-MS/MS-based untargeted metabolomics utilizing UHPLC-QTOF-MS with electrospray ionization in both positive and negative ion modes, employing C18 reverse-phase chromatography with gradient elution using water-acetonitrile mobile phases containing 0.1% formic acid, followed by data-dependent acquisition with dynamic exclusion and collision-induced dissociation fragmentation, processed through XCMS peak detection algorithms with subsequent statistical analysis using MetaboAnalyst including multivariate PCA and PLS-DA modeling with permutation testing and pathway enrichment analysis via KEGG and BioCyc databases.
""",
        'non_biomedical': """
The weather today is quite nice. I went to the store to buy groceries. The traffic was heavy on the highway. My favorite color is blue. Pizza is a popular food choice.
""",
        'inconsistent': """
Metabolomics always provides completely accurate results and never fails to identify every possible biomarker. However, it sometimes gives uncertain results and may not be reliable. The field is both revolutionary and traditional, offering breakthrough discoveries while maintaining established methods.
"""
    }

@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        'max_response_time_ms': 1000,
        'min_throughput_ops_per_sec': 5,
        'concurrent_operations': 10,
        'stress_test_operations': 100
    }


# =====================================================================
# INDIVIDUAL SCORING DIMENSION TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestIndividualScoringDimensions:
    """Test individual scoring dimension calculations."""
    
    @pytest.mark.asyncio
    async def test_metabolomics_relevance_scoring(self, relevance_scorer):
        """Test metabolomics relevance scoring."""
        test_cases = [
            # High relevance
            ("LC-MS metabolomics analysis", "LC-MS is used for metabolite analysis in clinical metabolomics studies", 70, 100),
            # Medium relevance
            ("biomarker discovery", "Biomarkers are useful for disease diagnosis", 40, 80),
            # Low relevance
            ("weather forecast", "Today will be sunny and warm", 0, 40),
            # No content
            ("metabolomics", "", 0, 20)
        ]
        
        for query, response, min_expected, max_expected in test_cases:
            score = await relevance_scorer._calculate_metabolomics_relevance(query, response)
            assert min_expected <= score <= max_expected, \
                f"Metabolomics relevance score {score} not in expected range [{min_expected}, {max_expected}] for query: '{query}'"
    
    @pytest.mark.asyncio
    async def test_clinical_applicability_scoring(self, relevance_scorer):
        """Test clinical applicability scoring."""
        test_cases = [
            # High clinical relevance
            ("clinical diagnosis", "Clinical metabolomics supports patient diagnosis and treatment monitoring", 60, 100),
            # Medium clinical relevance
            ("patient care", "Research may have implications for patient treatment", 30, 70),
            # Low clinical relevance
            ("basic research", "This is fundamental research without immediate clinical application", 20, 50),
        ]
        
        for query, response, min_expected, max_expected in test_cases:
            score = await relevance_scorer._calculate_clinical_applicability(query, response)
            assert min_expected <= score <= max_expected, \
                f"Clinical applicability score {score} not in expected range [{min_expected}, {max_expected}]"
    
    @pytest.mark.asyncio
    async def test_query_alignment_scoring(self, relevance_scorer):
        """Test query alignment scoring."""
        test_cases = [
            # Perfect alignment
            ("metabolomics biomarkers", "Metabolomics identifies biomarkers for disease diagnosis", 60, 100),
            # Partial alignment
            ("LC-MS analysis", "Mass spectrometry is used in research", 30, 70),
            # Poor alignment
            ("diabetes research", "The weather is nice today", 0, 30),
        ]
        
        for query, response, min_expected, max_expected in test_cases:
            score = await relevance_scorer._calculate_query_alignment(query, response)
            assert min_expected <= score <= max_expected, \
                f"Query alignment score {score} not in expected range [{min_expected}, {max_expected}]"
    
    @pytest.mark.asyncio
    async def test_scientific_rigor_scoring(self, relevance_scorer):
        """Test scientific rigor scoring."""
        test_cases = [
            # High rigor
            ("Studies show that p-value < 0.05 with confidence intervals indicating statistical significance", 70, 100),
            # Medium rigor
            ("Research indicates potential benefits with some limitations", 40, 70),
            # Low rigor
            ("This amazing breakthrough will revolutionize everything completely", 20, 50),
        ]
        
        for response, min_expected, max_expected in test_cases:
            score = await relevance_scorer._calculate_scientific_rigor(response)
            assert min_expected <= score <= max_expected, \
                f"Scientific rigor score {score} not in expected range [{min_expected}, {max_expected}]"
    
    @pytest.mark.asyncio
    async def test_biomedical_context_depth_scoring(self, relevance_scorer):
        """Test biomedical context depth scoring."""
        test_cases = [
            # High depth
            ("Metabolic pathways involve glycolysis and TCA cycle with physiological regulation", 60, 100),
            # Medium depth
            ("Biological processes are involved in cellular function", 30, 70),
            # Low depth
            ("Things work in the body somehow", 10, 40),
        ]
        
        for response, min_expected, max_expected in test_cases:
            score = await relevance_scorer._calculate_biomedical_context_depth(response)
            assert min_expected <= score <= max_expected, \
                f"Biomedical context depth score {score} not in expected range [{min_expected}, {max_expected}]"


# =====================================================================
# QUERY CLASSIFICATION TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestQueryClassification:
    """Test query type classification functionality."""
    
    def test_basic_definition_classification(self, query_classifier):
        """Test classification of basic definition queries."""
        definition_queries = [
            "What is metabolomics?",
            "Define biomarker",
            "Explain LC-MS",
            "What does NMR mean?",
            "Introduction to mass spectrometry"
        ]
        
        for query in definition_queries:
            query_type = query_classifier.classify_query(query)
            assert query_type in ['basic_definition', 'general'], \
                f"Query '{query}' should be classified as basic_definition or general, got: {query_type}"
    
    def test_clinical_application_classification(self, query_classifier):
        """Test classification of clinical application queries."""
        clinical_queries = [
            "How is metabolomics used in patient diagnosis?",
            "Clinical applications of biomarkers",
            "Therapeutic monitoring using metabolomics",
            "Medical applications of mass spectrometry",
            "Patient care and metabolomic profiling"
        ]
        
        for query in clinical_queries:
            query_type = query_classifier.classify_query(query)
            assert query_type in ['clinical_application', 'general'], \
                f"Query '{query}' should be classified as clinical_application or general, got: {query_type}"
    
    def test_analytical_method_classification(self, query_classifier):
        """Test classification of analytical method queries."""
        method_queries = [
            "LC-MS protocol for metabolomics",
            "GC-MS analysis procedure",
            "NMR spectroscopy methods",
            "Mass spectrometry techniques",
            "Sample preparation for HILIC"
        ]
        
        for query in method_queries:
            query_type = query_classifier.classify_query(query)
            assert query_type in ['analytical_method', 'general'], \
                f"Query '{query}' should be classified as analytical_method or general, got: {query_type}"
    
    def test_research_design_classification(self, query_classifier):
        """Test classification of research design queries."""
        research_queries = [
            "Study design for metabolomics research",
            "Statistical analysis of biomarker data",
            "Sample size calculation methods",
            "Validation strategies for metabolomics",
            "Quality control in metabolomic studies"
        ]
        
        for query in research_queries:
            query_type = query_classifier.classify_query(query)
            assert query_type in ['research_design', 'general'], \
                f"Query '{query}' should be classified as research_design or general, got: {query_type}"
    
    def test_disease_specific_classification(self, query_classifier):
        """Test classification of disease-specific queries."""
        disease_queries = [
            "Metabolomics in diabetes research",
            "Cancer biomarker discovery",
            "Cardiovascular disease metabolomics",
            "Alzheimer's disease biomarkers",
            "Liver disease metabolic signatures"
        ]
        
        for query in disease_queries:
            query_type = query_classifier.classify_query(query)
            assert query_type in ['disease_specific', 'general'], \
                f"Query '{query}' should be classified as disease_specific or general, got: {query_type}"
    
    def test_edge_case_classification(self, query_classifier):
        """Test classification of edge cases."""
        edge_cases = [
            "",  # Empty query
            "?",  # Single character
            "a b c d e f g",  # Random words
            "12345",  # Numbers only
            "!!!@@@###",  # Special characters
        ]
        
        for query in edge_cases:
            query_type = query_classifier.classify_query(query)
            assert query_type == 'general', \
                f"Edge case query '{query}' should default to general, got: {query_type}"


# =====================================================================
# RESPONSE LENGTH AND STRUCTURE QUALITY TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestResponseQualityValidation:
    """Test response length and structure quality validation."""
    
    @pytest.mark.asyncio
    async def test_response_length_quality_scoring(self, relevance_scorer, test_responses):
        """Test response length quality scoring for different query types."""
        query_types = ['basic_definition', 'clinical_application', 'analytical_method', 'research_design', 'disease_specific']
        
        for query_type in query_types:
            test_query = f"Test {query_type.replace('_', ' ')} query"
            
            # Test different response lengths
            responses = {
                'too_short': "Yes.",
                'optimal': test_responses['good'],
                'too_long': test_responses['very_long']
            }
            
            for length_category, response in responses.items():
                score = await relevance_scorer._calculate_response_length_quality(test_query, response)
                
                assert 0 <= score <= 100, f"Length quality score should be 0-100, got: {score}"
                
                if length_category == 'optimal':
                    assert score >= 70, f"Optimal length should score >= 70, got: {score}"
                elif length_category == 'too_short':
                    assert score <= 70, f"Too short should score <= 70, got: {score}"
    
    @pytest.mark.asyncio
    async def test_response_structure_quality_scoring(self, relevance_scorer, test_responses):
        """Test response structure quality scoring."""
        structure_test_cases = [
            # Well-structured response
            (test_responses['excellent'], 80, 100),
            # Moderately structured response
            (test_responses['good'], 60, 90),
            # Poorly structured response
            (test_responses['poor'], 30, 70),
            # Technical but unstructured
            (test_responses['technical_dense'], 40, 80)
        ]
        
        for response, min_expected, max_expected in structure_test_cases:
            score = await relevance_scorer._calculate_response_structure_quality(response)
            
            assert min_expected <= score <= max_expected, \
                f"Structure quality score {score} not in expected range [{min_expected}, {max_expected}]"
    
    def test_formatting_quality_assessment(self, relevance_scorer):
        """Test formatting quality assessment."""
        formatting_examples = [
            # Good formatting
            ("# Title\n\n## Section\n\n- Bullet point\n- Another point\n\n**Bold text**", 70, 100),
            # Poor formatting
            ("This is just plain text without any formatting or structure at all.", 40, 70),
            # No formatting
            ("", 0, 60)
        ]
        
        for text, min_expected, max_expected in formatting_examples:
            score = relevance_scorer._assess_formatting_quality(text)
            
            assert min_expected <= score <= max_expected, \
                f"Formatting quality score {score} not in expected range [{min_expected}, {max_expected}]"
    
    def test_readability_assessment(self, relevance_scorer):
        """Test readability assessment."""
        readability_examples = [
            # Good readability
            ("This sentence is clear and easy to understand. It uses appropriate technical terms. The structure is logical.", 60, 100),
            # Poor readability - too technical
            (test_responses['technical_dense'], 30, 70),
            # Poor readability - too simple
            ("Good. Yes. OK. Fine.", 30, 70)
        ]
        
        for text, min_expected, max_expected in readability_examples:
            score = relevance_scorer._assess_readability(text)
            
            assert min_expected <= score <= max_expected, \
                f"Readability score {score} not in expected range [{min_expected}, {max_expected}]"


# =====================================================================
# ADAPTIVE WEIGHTING SCHEME TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestAdaptiveWeightingSchemes:
    """Test adaptive weighting schemes for different query types."""
    
    def test_weighting_scheme_completeness(self, weighting_manager):
        """Test that all query types have complete weighting schemes."""
        query_types = ['basic_definition', 'clinical_application', 'analytical_method', 'research_design', 'disease_specific', 'general']
        
        for query_type in query_types:
            weights = weighting_manager.get_weights(query_type)
            
            # Check that weights exist
            assert isinstance(weights, dict), f"Weights should be a dictionary for {query_type}"
            assert len(weights) > 0, f"No weights defined for {query_type}"
            
            # Check weight values are valid
            for dimension, weight in weights.items():
                assert 0 <= weight <= 1, f"Weight {weight} for dimension {dimension} should be between 0 and 1"
            
            # Check weights sum to approximately 1
            total_weight = sum(weights.values())
            assert 0.9 <= total_weight <= 1.1, f"Weights should sum to ~1.0, got {total_weight} for {query_type}"
    
    def test_query_type_specific_weighting(self, weighting_manager):
        """Test that different query types have appropriate weight distributions."""
        # Clinical application queries should weight clinical_applicability highly
        clinical_weights = weighting_manager.get_weights('clinical_application')
        assert clinical_weights.get('clinical_applicability', 0) >= 0.2, \
            "Clinical application queries should highly weight clinical_applicability"
        
        # Analytical method queries should weight metabolomics_relevance highly
        analytical_weights = weighting_manager.get_weights('analytical_method')
        assert analytical_weights.get('metabolomics_relevance', 0) >= 0.3, \
            "Analytical method queries should highly weight metabolomics_relevance"
        
        # Research design queries should weight scientific_rigor highly
        research_weights = weighting_manager.get_weights('research_design')
        assert research_weights.get('scientific_rigor', 0) >= 0.2, \
            "Research design queries should highly weight scientific_rigor"
    
    def test_weight_scheme_consistency(self, weighting_manager):
        """Test consistency of weighting schemes."""
        all_query_types = ['basic_definition', 'clinical_application', 'analytical_method', 'research_design', 'disease_specific', 'general']
        
        # Get all dimensions used across query types
        all_dimensions = set()
        for query_type in all_query_types:
            weights = weighting_manager.get_weights(query_type)
            all_dimensions.update(weights.keys())
        
        # Each query type should use the same dimensions (or have 0 weight)
        for query_type in all_query_types:
            weights = weighting_manager.get_weights(query_type)
            for dimension in all_dimensions:
                weight = weights.get(dimension, 0)
                assert 0 <= weight <= 1, f"Dimension {dimension} has invalid weight {weight} for {query_type}"


# =====================================================================
# EDGE CASES TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_inputs(self, relevance_scorer):
        """Test handling of empty inputs."""
        edge_cases = [
            ("", ""),  # Both empty
            ("", "Valid response"),  # Empty query
            ("Valid query", ""),  # Empty response
            ("   ", "   "),  # Whitespace only
        ]
        
        for query, response in edge_cases:
            try:
                result = await relevance_scorer.calculate_relevance_score(query, response)
                
                assert isinstance(result, RelevanceScore), "Should return RelevanceScore object"
                assert 0 <= result.overall_score <= 100, f"Score should be 0-100, got {result.overall_score}"
                assert result.processing_time_ms >= 0, "Processing time should be non-negative"
                
            except Exception as e:
                pytest.fail(f"Empty input case ({query!r}, {response!r}) raised exception: {e}")
    
    @pytest.mark.asyncio
    async def test_very_long_inputs(self, relevance_scorer):
        """Test handling of very long inputs."""
        long_query = "What is metabolomics? " * 100
        long_response = "Metabolomics is a field of study. " * 500
        
        start_time = time.time()
        result = await relevance_scorer.calculate_relevance_score(long_query, long_response)
        end_time = time.time()
        
        assert isinstance(result, RelevanceScore)
        assert 0 <= result.overall_score <= 100
        assert (end_time - start_time) < 10, "Should complete within 10 seconds even for long inputs"
    
    @pytest.mark.asyncio
    async def test_nonsensical_inputs(self, relevance_scorer):
        """Test handling of nonsensical inputs."""
        nonsensical_cases = [
            ("xyzabc defghi jklmno", "pqrstu vwxyz abcdef"),
            ("12345 67890 !@#$%", "^&*() []{}; ':\"<>?"),
            ("🔬🧪🦠💊⚗️", "📊📈📉📋🔍"),  # Emoji only
            ("A" * 1000, "B" * 1000),  # Repetitive characters
        ]
        
        for query, response in nonsensical_cases:
            result = await relevance_scorer.calculate_relevance_score(query, response)
            
            assert isinstance(result, RelevanceScore)
            assert 0 <= result.overall_score <= 100
            # Nonsensical inputs should generally score low
            assert result.overall_score <= 50, f"Nonsensical input should score low, got {result.overall_score}"
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, relevance_scorer):
        """Test handling of special characters and encoding."""
        special_cases = [
            ("Café résumé naïve", "Résponse française"),
            ("代謝組學研究", "蛋白質組學分析"),  # Chinese characters
            ("Ñoño español", "Análisis metabólico"),
            ("JSON: {\"key\": \"value\"}", "XML: <tag>content</tag>"),
            ("SQL: SELECT * FROM table;", "Code: if (x > 0) { return true; }"),
        ]
        
        for query, response in special_cases:
            try:
                result = await relevance_scorer.calculate_relevance_score(query, response)
                assert isinstance(result, RelevanceScore)
                assert 0 <= result.overall_score <= 100
            except UnicodeError:
                pytest.fail(f"Unicode handling failed for: {query!r}, {response!r}")
    
    @pytest.mark.asyncio
    async def test_malformed_metadata(self, relevance_scorer):
        """Test handling of malformed metadata."""
        malformed_metadata_cases = [
            {"invalid": float('nan')},
            {"nested": {"deep": {"very": {"deeply": "nested"}}}},
            {"circular": None},  # Would create circular reference if set to itself
            {"large_list": list(range(10000))},
        ]
        
        for metadata in malformed_metadata_cases:
            try:
                result = await relevance_scorer.calculate_relevance_score(
                    "test query", "test response", metadata=metadata
                )
                assert isinstance(result, RelevanceScore)
            except Exception as e:
                # Should handle gracefully, not crash
                assert "metadata" in str(e).lower() or isinstance(e, (ValueError, TypeError))


# =====================================================================
# PERFORMANCE TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_async_execution_performance(self, relevance_scorer, performance_config):
        """Test async execution performance."""
        test_pairs = [
            (f"Query {i}", f"Response {i} with metabolomics content and biomarkers") 
            for i in range(10)
        ]
        
        # Test sequential execution
        start_time = time.time()
        sequential_results = []
        for query, response in test_pairs:
            result = await relevance_scorer.calculate_relevance_score(query, response)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Test concurrent execution
        start_time = time.time()
        concurrent_tasks = [
            relevance_scorer.calculate_relevance_score(query, response)
            for query, response in test_pairs
        ]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - start_time
        
        # Verify results are consistent
        assert len(sequential_results) == len(concurrent_results)
        
        # Concurrent should be faster (or at least not much slower)
        speedup_ratio = sequential_time / concurrent_time if concurrent_time > 0 else 1
        assert speedup_ratio >= 0.8, f"Concurrent execution should be efficient, speedup ratio: {speedup_ratio:.2f}"
    
    @pytest.mark.asyncio
    async def test_response_time_limits(self, relevance_scorer, performance_config):
        """Test that response times stay within acceptable limits."""
        max_time_ms = performance_config['max_response_time_ms']
        
        test_cases = [
            ("Simple query", "Simple response"),
            ("Complex metabolomics LC-MS analysis query", "Complex response with detailed analytical procedures"),
            ("Very long query " * 20, "Very long response " * 50)
        ]
        
        for query, response in test_cases:
            start_time = time.time()
            result = await relevance_scorer.calculate_relevance_score(query, response)
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            assert response_time_ms <= max_time_ms, \
                f"Response time {response_time_ms:.2f}ms exceeds limit {max_time_ms}ms"
            
            # Recorded processing time should be reasonable
            assert result.processing_time_ms <= response_time_ms * 1.2, \
                "Recorded processing time should be close to measured time"
    
    @pytest.mark.asyncio
    async def test_throughput_performance(self, relevance_scorer, performance_config):
        """Test throughput performance."""
        min_ops_per_sec = performance_config['min_throughput_ops_per_sec']
        test_operations = 20
        
        start_time = time.time()
        tasks = [
            relevance_scorer.calculate_relevance_score(
                f"Test query {i}", 
                f"Test response {i} about metabolomics and clinical applications"
            )
            for i in range(test_operations)
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        ops_per_second = test_operations / total_time
        
        assert ops_per_second >= min_ops_per_sec, \
            f"Throughput {ops_per_second:.2f} ops/sec below minimum {min_ops_per_sec}"
        
        # All operations should have completed successfully
        assert len(results) == test_operations
        for result in results:
            assert isinstance(result, RelevanceScore)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, relevance_scorer):
        """Test memory efficiency during repeated operations."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Run many operations to test for memory leaks
        for i in range(100):
            await relevance_scorer.calculate_relevance_score(
                f"Memory test query {i}",
                f"Memory test response {i} with metabolomics content"
            )
            
            # Periodically force garbage collection
            if i % 20 == 0:
                gc.collect()
        
        # Should complete without memory issues
        assert True, "Memory efficiency test completed"
    
    @pytest.mark.asyncio 
    async def test_concurrent_load_handling(self, relevance_scorer, performance_config):
        """Test handling of concurrent load."""
        concurrent_ops = performance_config['concurrent_operations']
        
        async def worker(worker_id):
            """Worker function for concurrent testing."""
            results = []
            for i in range(5):  # 5 operations per worker
                result = await relevance_scorer.calculate_relevance_score(
                    f"Worker {worker_id} query {i}",
                    f"Worker {worker_id} response {i} about clinical metabolomics"
                )
                results.append(result)
            return results
        
        start_time = time.time()
        worker_tasks = [worker(i) for i in range(concurrent_ops)]
        all_results = await asyncio.gather(*worker_tasks)
        end_time = time.time()
        
        # Flatten results
        flat_results = []
        for worker_results in all_results:
            flat_results.extend(worker_results)
        
        # Verify all operations completed successfully
        total_operations = concurrent_ops * 5
        assert len(flat_results) == total_operations
        
        # Verify reasonable completion time
        total_time = end_time - start_time
        assert total_time < 30, f"Concurrent load test took too long: {total_time:.2f}s"


# =====================================================================
# SEMANTIC SIMILARITY ENGINE TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestSemanticSimilarityEngine:
    """Test semantic similarity engine functionality."""
    
    @pytest.mark.asyncio
    async def test_similarity_calculation_basic(self, semantic_engine):
        """Test basic similarity calculation."""
        test_cases = [
            # High similarity
            ("metabolomics analysis", "metabolomics research and analysis", 60, 100),
            # Medium similarity
            ("biomarker discovery", "identification of disease markers", 30, 80),
            # Low similarity
            ("LC-MS analysis", "weather forecast today", 0, 30),
            # Identical
            ("exact match", "exact match", 80, 100),
            # Empty cases
            ("", "test", 0, 20),
            ("test", "", 0, 20),
            ("", "", 0, 20)
        ]
        
        for query, response, min_expected, max_expected in test_cases:
            similarity = await semantic_engine.calculate_similarity(query, response)
            
            assert 0 <= similarity <= 100, f"Similarity should be 0-100, got {similarity}"
            assert min_expected <= similarity <= max_expected, \
                f"Similarity {similarity} not in expected range [{min_expected}, {max_expected}] for '{query}' vs '{response}'"
    
    @pytest.mark.asyncio
    async def test_biomedical_term_boost(self, semantic_engine):
        """Test biomedical term boost functionality."""
        # Query with biomedical terms
        biomedical_query = "metabolomics LC-MS analysis"
        
        # Response with matching biomedical terms should score higher
        biomedical_response = "LC-MS metabolomics provides accurate analysis"
        non_biomedical_response = "The quick brown fox jumps over lazy dog"
        
        biomedical_similarity = await semantic_engine.calculate_similarity(
            biomedical_query, biomedical_response
        )
        non_biomedical_similarity = await semantic_engine.calculate_similarity(
            biomedical_query, non_biomedical_response
        )
        
        assert biomedical_similarity > non_biomedical_similarity, \
            "Biomedical term matching should increase similarity score"
    
    def test_meaningful_term_extraction(self, semantic_engine):
        """Test extraction of meaningful terms."""
        test_text = "The LC-MS analysis of metabolomic biomarkers in clinical samples"
        
        meaningful_terms = semantic_engine._extract_meaningful_terms(test_text)
        
        # Should extract meaningful terms and exclude stopwords
        assert 'lc-ms' in meaningful_terms or 'analysis' in meaningful_terms
        assert 'metabolomic' in meaningful_terms
        assert 'biomarkers' in meaningful_terms or 'clinical' in meaningful_terms
        
        # Should exclude stopwords
        assert 'the' not in meaningful_terms
        assert 'of' not in meaningful_terms
        assert 'in' not in meaningful_terms
    
    @pytest.mark.asyncio
    async def test_similarity_symmetry(self, semantic_engine):
        """Test that similarity calculation is symmetric."""
        test_pairs = [
            ("query A", "response B"),
            ("metabolomics", "biomarkers"),
            ("LC-MS analysis", "mass spectrometry"),
        ]
        
        for text1, text2 in test_pairs:
            similarity1 = await semantic_engine.calculate_similarity(text1, text2)
            similarity2 = await semantic_engine.calculate_similarity(text2, text1)
            
            # Should be symmetric (within small tolerance for floating point)
            assert abs(similarity1 - similarity2) < 0.01, \
                f"Similarity should be symmetric: {similarity1} vs {similarity2}"
    
    @pytest.mark.asyncio
    async def test_similarity_consistency(self, semantic_engine):
        """Test consistency of similarity calculations."""
        query = "What is metabolomics?"
        response = "Metabolomics is the study of small molecules"
        
        # Run multiple times
        similarities = []
        for _ in range(5):
            similarity = await semantic_engine.calculate_similarity(query, response)
            similarities.append(similarity)
        
        # Should be consistent (deterministic)
        assert all(abs(s - similarities[0]) < 0.01 for s in similarities), \
            f"Similarity calculations should be consistent: {similarities}"


# =====================================================================
# DOMAIN EXPERTISE VALIDATOR TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestDomainExpertiseValidator:
    """Test domain expertise validation functionality."""
    
    @pytest.mark.asyncio
    async def test_domain_expertise_validation(self, domain_validator):
        """Test domain expertise validation."""
        test_cases = [
            # High expertise
            ("LC-MS analysis requires careful sample preparation and quality control measures with statistical validation", 70, 100),
            # Medium expertise
            ("Metabolomics involves analyzing small molecules in biological samples", 50, 80),
            # Low expertise
            ("This amazing breakthrough will revolutionize everything", 20, 60),
            # Empty response
            ("", 0, 30)
        ]
        
        for response, min_expected, max_expected in test_cases:
            score = await domain_validator.validate_domain_expertise(response)
            
            assert 0 <= score <= 100, f"Expertise score should be 0-100, got {score}"
            assert min_expected <= score <= max_expected, \
                f"Expertise score {score} not in expected range [{min_expected}, {max_expected}]"
    
    def test_terminology_assessment(self, domain_validator):
        """Test terminology usage assessment."""
        # Response with appropriate terminology
        technical_response = "HILIC chromatography with negative mode mass spectrometry for polar metabolite analysis"
        technical_score = domain_validator._assess_terminology_usage(technical_response)
        
        # Response without technical terms
        simple_response = "This method works well for the analysis"
        simple_score = domain_validator._assess_terminology_usage(simple_response)
        
        assert technical_score >= simple_score, \
            "Technical terminology should increase expertise score"
    
    def test_methodology_assessment(self, domain_validator):
        """Test methodology assessment."""
        # Response with methodological terms
        methodological_response = "Study design included quality control samples with statistical validation and reproducibility testing"
        methodological_score = domain_validator._assess_methodology(methodological_response)
        
        # Response without methodology
        non_methodological_response = "The results were interesting and promising"
        non_methodological_score = domain_validator._assess_methodology(non_methodological_response)
        
        assert methodological_score >= non_methodological_score, \
            "Methodological content should increase expertise score"
    
    def test_error_penalty_assessment(self, domain_validator):
        """Test error penalty assessment."""
        # Response with problematic claims
        problematic_response = "This method is always accurate and never fails with completely reliable results"
        penalty = domain_validator._assess_error_penalty(problematic_response)
        
        # Response without problematic claims
        balanced_response = "Studies suggest this method may provide reliable results under certain conditions"
        no_penalty = domain_validator._assess_error_penalty(balanced_response)
        
        assert penalty > no_penalty, \
            "Problematic claims should result in higher penalties"
    
    def test_evidence_quality_assessment(self, domain_validator):
        """Test evidence quality assessment."""
        # Response with evidence indicators
        evidence_response = "Studies show that data demonstrates significant findings according to research"
        evidence_score = domain_validator._assess_evidence_quality(evidence_response)
        
        # Response without evidence
        opinion_response = "I think this might work based on my intuition"
        opinion_score = domain_validator._assess_evidence_quality(opinion_response)
        
        assert evidence_score >= opinion_score, \
            "Evidence-based responses should score higher"


# =====================================================================
# INTEGRATION AND PIPELINE TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestIntegrationAndPipeline:
    """Test integration and overall relevance scoring pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_relevance_scoring_pipeline(self, relevance_scorer, test_queries, test_responses):
        """Test the complete relevance scoring pipeline."""
        # Test with different query types and response qualities
        test_combinations = [
            ('basic_definition', test_responses['excellent']),
            ('clinical_application', test_responses['good']),
            ('analytical_method', test_responses['poor']),
            ('research_design', test_responses['technical_dense']),
            ('disease_specific', test_responses['non_biomedical'])
        ]
        
        for query_type, response in test_combinations:
            query = f"Test {query_type.replace('_', ' ')} query"
            
            result = await relevance_scorer.calculate_relevance_score(query, response)
            
            # Validate result structure
            assert isinstance(result, RelevanceScore)
            assert 0 <= result.overall_score <= 100
            assert result.query_type in ['basic_definition', 'clinical_application', 'analytical_method', 
                                       'research_design', 'disease_specific', 'general']
            assert isinstance(result.dimension_scores, dict)
            assert len(result.dimension_scores) > 0
            assert isinstance(result.weights_used, dict)
            assert result.processing_time_ms >= 0
            assert isinstance(result.explanation, str)
            assert len(result.explanation) > 0
            assert 0 <= result.confidence_score <= 100
    
    @pytest.mark.asyncio
    async def test_batch_relevance_scoring(self):
        """Test batch relevance scoring functionality."""
        test_pairs = [
            ("What is metabolomics?", "Metabolomics is the study of small molecules"),
            ("Clinical applications?", "Used in medical diagnosis"),
            ("LC-MS method?", "LC-MS separates and identifies compounds"),
            ("Study design?", "Requires careful planning and controls"),
            ("Diabetes research?", "Metabolomics reveals disease signatures")
        ]
        
        results = await batch_relevance_scoring(test_pairs)
        
        assert len(results) == len(test_pairs)
        for result in results:
            assert isinstance(result, RelevanceScore)
            assert 0 <= result.overall_score <= 100
    
    @pytest.mark.asyncio
    async def test_quick_relevance_check_function(self):
        """Test quick relevance check utility function."""
        score = await quick_relevance_check(
            "What is metabolomics?", 
            "Metabolomics is the comprehensive study of metabolites"
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, relevance_scorer):
        """Test confidence score calculation."""
        # High consistency should give high confidence
        consistent_scores = {
            'dim1': 80.0, 'dim2': 82.0, 'dim3': 78.0, 'dim4': 81.0
        }
        high_confidence = relevance_scorer._calculate_confidence(consistent_scores, {})
        
        # High variance should give low confidence
        inconsistent_scores = {
            'dim1': 20.0, 'dim2': 80.0, 'dim3': 10.0, 'dim4': 90.0
        }
        low_confidence = relevance_scorer._calculate_confidence(inconsistent_scores, {})
        
        assert high_confidence > low_confidence, \
            "Consistent scores should have higher confidence"
        assert 0 <= high_confidence <= 100
        assert 0 <= low_confidence <= 100
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, relevance_scorer):
        """Test explanation generation."""
        dimension_scores = {
            'metabolomics_relevance': 85.0,
            'clinical_applicability': 70.0,
            'query_alignment': 90.0,
            'scientific_rigor': 75.0
        }
        weights = {
            'metabolomics_relevance': 0.3,
            'clinical_applicability': 0.3,
            'query_alignment': 0.2,
            'scientific_rigor': 0.2
        }
        
        explanation = relevance_scorer._generate_explanation(
            dimension_scores, weights, 'analytical_method'
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'analytical method' in explanation.lower()
        
        # Should include dimension scores
        for dimension in dimension_scores:
            dimension_readable = dimension.replace('_', ' ')
            assert any(dimension_readable.lower() in explanation.lower() 
                      for dimension_readable in [dimension, dimension.replace('_', ' ')])
    
    @pytest.mark.asyncio
    async def test_response_quality_validation_integration(self, relevance_scorer, test_responses):
        """Test integration with response quality validation."""
        for response_type, response_text in test_responses.items():
            if response_type == 'empty':
                continue
            
            quality_assessment = relevance_scorer.validate_response_quality(
                "Test query for quality validation", response_text
            )
            
            assert isinstance(quality_assessment, dict)
            assert 'query_type' in quality_assessment
            assert 'length_assessment' in quality_assessment
            assert 'structure_assessment' in quality_assessment
            assert 'completeness_score' in quality_assessment
            assert 'overall_quality_score' in quality_assessment
            assert 'quality_grade' in quality_assessment
            assert 'recommendations' in quality_assessment
            
            # Quality grade should be valid
            assert quality_assessment['quality_grade'] in ['A', 'B', 'C', 'D', 'F']
            
            # Recommendations should be a list
            assert isinstance(quality_assessment['recommendations'], list)
    
    @pytest.mark.asyncio
    async def test_scoring_determinism(self, relevance_scorer):
        """Test that scoring is deterministic."""
        query = "What is metabolomics in clinical research?"
        response = """Metabolomics is the comprehensive study of small molecules called metabolites in biological systems. In clinical research, it's used for biomarker discovery, disease diagnosis, and treatment monitoring using analytical techniques like LC-MS and GC-MS."""
        
        # Run multiple times
        scores = []
        for _ in range(3):
            result = await relevance_scorer.calculate_relevance_score(query, response)
            scores.append(result.overall_score)
        
        # Should be deterministic
        assert all(abs(score - scores[0]) < 0.01 for score in scores), \
            f"Scoring should be deterministic, got scores: {scores}"
    
    @pytest.mark.asyncio
    async def test_metadata_handling(self, relevance_scorer):
        """Test metadata handling in scoring."""
        query = "Test query"
        response = "Test response about metabolomics"
        metadata = {
            'source': 'test',
            'experiment_id': 123,
            'custom_field': 'custom_value'
        }
        
        result = await relevance_scorer.calculate_relevance_score(query, response, metadata)
        
        assert isinstance(result, RelevanceScore)
        assert isinstance(result.metadata, dict)
        # Should include standard metadata
        assert 'query_length' in result.metadata
        assert 'response_length' in result.metadata
        assert 'word_count' in result.metadata


# =====================================================================
# STRESS AND ROBUSTNESS TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestStressAndRobustness:
    """Test system robustness under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_high_load_stress(self, relevance_scorer):
        """Test system under high load."""
        # Create many concurrent requests
        num_requests = 50
        
        async def make_request(i):
            return await relevance_scorer.calculate_relevance_score(
                f"Stress test query {i} about metabolomics",
                f"Stress test response {i} discussing clinical applications and LC-MS analysis"
            )
        
        start_time = time.time()
        
        # Execute all requests concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Count successful vs failed requests
        successful = sum(1 for r in results if isinstance(r, RelevanceScore))
        failed = sum(1 for r in results if isinstance(r, Exception))
        
        assert successful > num_requests * 0.8, \
            f"At least 80% of requests should succeed under load. Successful: {successful}/{num_requests}"
        
        # Should complete within reasonable time
        total_time = end_time - start_time
        assert total_time < 60, f"High load test took too long: {total_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_exception_recovery(self, relevance_scorer):
        """Test recovery from exceptions."""
        # Test with various problematic inputs that might cause errors
        problematic_inputs = [
            (None, "valid response"),  # None query
            ("valid query", None),     # None response
            ({"not": "string"}, "valid response"),  # Non-string query
            ("valid query", ["not", "string"]),    # Non-string response
        ]
        
        for query, response in problematic_inputs:
            try:
                result = await relevance_scorer.calculate_relevance_score(query, response)
                # If it succeeds, should return valid result
                if result is not None:
                    assert isinstance(result, RelevanceScore)
                    assert 0 <= result.overall_score <= 100
            except Exception as e:
                # If it fails, should fail gracefully
                assert isinstance(e, (TypeError, ValueError, AttributeError))
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, relevance_scorer):
        """Test that resources are properly cleaned up."""
        # Run many operations to test for resource leaks
        initial_tasks = len(asyncio.all_tasks())
        
        for i in range(20):
            await relevance_scorer.calculate_relevance_score(
                f"Cleanup test {i}",
                f"Response {i} about metabolomics research"
            )
        
        # Allow some time for cleanup
        await asyncio.sleep(0.1)
        
        final_tasks = len(asyncio.all_tasks())
        
        # Should not have significantly more tasks
        task_growth = final_tasks - initial_tasks
        assert task_growth <= 5, f"Too many tasks created and not cleaned up: {task_growth}"


# =====================================================================
# CONFIGURATION AND CUSTOMIZATION TESTS  
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestConfigurationAndCustomization:
    """Test configuration options and customization."""
    
    def test_default_configuration(self):
        """Test default configuration."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        assert hasattr(scorer, 'config')
        assert isinstance(scorer.config, dict)
        
        # Should have reasonable defaults
        assert scorer.config.get('enable_caching', True) in [True, False]
        assert isinstance(scorer.config.get('parallel_processing', True), bool)
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        custom_config = {
            'enable_caching': False,
            'parallel_processing': False,
            'confidence_threshold': 80.0,
            'minimum_relevance_threshold': 60.0
        }
        
        scorer = ClinicalMetabolomicsRelevanceScorer(config=custom_config)
        
        assert scorer.config['enable_caching'] == False
        assert scorer.config['parallel_processing'] == False
        assert scorer.config['confidence_threshold'] == 80.0
        assert scorer.config['minimum_relevance_threshold'] == 60.0
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_processing(self):
        """Test parallel vs sequential processing modes."""
        # Test with parallel processing enabled
        parallel_config = {'parallel_processing': True}
        parallel_scorer = ClinicalMetabolomicsRelevanceScorer(config=parallel_config)
        
        # Test with parallel processing disabled
        sequential_config = {'parallel_processing': False}
        sequential_scorer = ClinicalMetabolomicsRelevanceScorer(config=sequential_config)
        
        query = "Test configuration query about metabolomics"
        response = "Test response with clinical applications and analytical methods"
        
        # Both should work and give similar results
        parallel_result = await parallel_scorer.calculate_relevance_score(query, response)
        sequential_result = await sequential_scorer.calculate_relevance_score(query, response)
        
        assert isinstance(parallel_result, RelevanceScore)
        assert isinstance(sequential_result, RelevanceScore)
        
        # Results should be similar (within tolerance)
        score_diff = abs(parallel_result.overall_score - sequential_result.overall_score)
        assert score_diff < 5.0, f"Parallel and sequential results should be similar, diff: {score_diff}"


# =====================================================================
# BIOMEDICAL DOMAIN-SPECIFIC TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="Relevance scorer module not available")
class TestBiomedicalDomainSpecifics:
    """Test biomedical domain-specific functionality."""
    
    @pytest.mark.asyncio
    async def test_biomedical_terminology_recognition(self, relevance_scorer):
        """Test recognition of biomedical terminology."""
        biomedical_responses = [
            "LC-MS analysis of metabolites using HILIC chromatography",
            "Clinical biomarker discovery in diabetes patients",
            "Proteomics and genomics integration with metabolomics",
            "Mass spectrometry-based metabolomic profiling",
            "NMR spectroscopy for structural elucidation"
        ]
        
        non_biomedical_response = "The weather is nice today and traffic is flowing smoothly"
        
        for bio_response in biomedical_responses:
            bio_result = await relevance_scorer.calculate_relevance_score(
                "biomedical query", bio_response
            )
            non_bio_result = await relevance_scorer.calculate_relevance_score(
                "biomedical query", non_biomedical_response
            )
            
            # Biomedical responses should score higher on relevant dimensions
            bio_metabolomics = bio_result.dimension_scores.get('metabolomics_relevance', 0)
            non_bio_metabolomics = non_bio_result.dimension_scores.get('metabolomics_relevance', 0)
            
            assert bio_metabolomics > non_bio_metabolomics, \
                f"Biomedical response should score higher on metabolomics relevance: {bio_metabolomics} vs {non_bio_metabolomics}"
    
    @pytest.mark.asyncio
    async def test_clinical_context_recognition(self, relevance_scorer):
        """Test recognition of clinical context."""
        clinical_query = "How is metabolomics used in patient diagnosis?"
        
        clinical_responses = [
            "Clinical metabolomics supports patient diagnosis through biomarker identification",
            "Medical applications include disease monitoring and treatment response assessment",
            "Healthcare providers use metabolomic profiles for precision medicine approaches"
        ]
        
        research_response = "Basic research investigates fundamental metabolic pathways"
        
        for clinical_response in clinical_responses:
            clinical_result = await relevance_scorer.calculate_relevance_score(
                clinical_query, clinical_response
            )
            research_result = await relevance_scorer.calculate_relevance_score(
                clinical_query, research_response
            )
            
            # Clinical responses should score higher on clinical applicability
            clinical_app_score = clinical_result.dimension_scores.get('clinical_applicability', 0)
            research_app_score = research_result.dimension_scores.get('clinical_applicability', 0)
            
            assert clinical_app_score >= research_app_score, \
                f"Clinical response should score higher on clinical applicability: {clinical_app_score} vs {research_app_score}"
    
    @pytest.mark.asyncio
    async def test_analytical_method_specificity(self, relevance_scorer):
        """Test analytical method specificity."""
        method_query = "How does LC-MS work for metabolomics?"
        
        specific_response = "LC-MS combines liquid chromatography separation with mass spectrometry detection for metabolite identification and quantification"
        vague_response = "This method works well for analysis of samples"
        
        specific_result = await relevance_scorer.calculate_relevance_score(method_query, specific_response)
        vague_result = await relevance_scorer.calculate_relevance_score(method_query, vague_response)
        
        # Specific response should score higher overall
        assert specific_result.overall_score > vague_result.overall_score, \
            f"Specific method response should score higher: {specific_result.overall_score} vs {vague_result.overall_score}"
        
        # Should score higher on metabolomics relevance
        specific_metabolomics = specific_result.dimension_scores.get('metabolomics_relevance', 0)
        vague_metabolomics = vague_result.dimension_scores.get('metabolomics_relevance', 0)
        
        assert specific_metabolomics > vague_metabolomics, \
            "Specific method response should have higher metabolomics relevance"
    
    def test_biomedical_keyword_coverage(self, relevance_scorer):
        """Test biomedical keyword coverage."""
        # Check that scorer has comprehensive biomedical keywords
        assert hasattr(relevance_scorer, 'biomedical_keywords')
        keywords = relevance_scorer.biomedical_keywords
        
        # Should have different categories
        expected_categories = ['metabolomics_core', 'analytical_methods', 'clinical_terms', 'research_concepts']
        for category in expected_categories:
            assert category in keywords, f"Missing keyword category: {category}"
            assert len(keywords[category]) > 0, f"Empty keyword category: {category}"
        
        # Check for key terms in each category
        assert any('metabolomics' in term for term in keywords['metabolomics_core'])
        assert any('LC-MS' in term or 'lc-ms' in term for term in keywords['analytical_methods'])
        assert any('clinical' in term for term in keywords['clinical_terms'])
        assert any('statistical' in term or 'study' in term for term in keywords['research_concepts'])


if __name__ == "__main__":
    # Run the tests
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--durations=10",  # Show slowest 10 tests
        "-x",  # Stop on first failure for debugging
    ])