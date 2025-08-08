#!/usr/bin/env python3
"""
Comprehensive Test Fixtures for Query Classification Testing

This module provides specialized fixtures for testing the query classification system,
including biomedical query samples, mock ResearchCategorizer instances, performance
testing utilities, and integration with the existing test infrastructure.

Created specifically to support the query classification tests in
test_query_classification_biomedical_samples.py by providing clean, reusable
fixtures that integrate well with the existing test framework.

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: CMO-LIGHTRAG-012-T01 Support
"""

import pytest
import time
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from unittest.mock import Mock, MagicMock, AsyncMock
from dataclasses import dataclass, field
from pathlib import Path
import json

# Import the biomedical query samples from the standalone file
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from test_fixtures_biomedical_queries import (
        QueryTestCase,
        ResearchCategory,
        ComplexityLevel,
        get_all_test_queries,
        get_queries_by_complexity,
        get_edge_case_queries,
        get_query_statistics
    )
    BIOMEDICAL_QUERIES_AVAILABLE = True
except ImportError:
    BIOMEDICAL_QUERIES_AVAILABLE = False


# =====================================================================
# MOCK RESEARCH CATEGORIZER AND CLASSIFICATION COMPONENTS
# =====================================================================

@dataclass
class CategoryPrediction:
    """Mock CategoryPrediction for testing."""
    category: 'ResearchCategory'
    confidence: float
    evidence: List[str] = field(default_factory=list)
    subject_area: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure confidence is within valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class CategoryMetrics:
    """Mock CategoryMetrics for testing."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)


class MockResearchCategory:
    """Mock ResearchCategory enum for testing."""
    METABOLITE_IDENTIFICATION = "metabolite_identification"
    PATHWAY_ANALYSIS = "pathway_analysis"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    DRUG_DISCOVERY = "drug_discovery"
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    DATA_PREPROCESSING = "data_preprocessing"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    LITERATURE_SEARCH = "literature_search"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    DATABASE_INTEGRATION = "database_integration"
    GENERAL_QUERY = "general_query"
    
    @classmethod
    def all_categories(cls):
        """Get all research categories."""
        return [
            cls.METABOLITE_IDENTIFICATION,
            cls.PATHWAY_ANALYSIS,
            cls.BIOMARKER_DISCOVERY,
            cls.DRUG_DISCOVERY,
            cls.CLINICAL_DIAGNOSIS,
            cls.DATA_PREPROCESSING,
            cls.STATISTICAL_ANALYSIS,
            cls.LITERATURE_SEARCH,
            cls.KNOWLEDGE_EXTRACTION,
            cls.DATABASE_INTEGRATION,
            cls.GENERAL_QUERY
        ]


class MockQueryAnalyzer:
    """Mock QueryAnalyzer for testing."""
    
    def __init__(self):
        self.analysis_count = 0
        self.keyword_weights = {
            # Metabolite identification keywords
            'metabolite': 0.8, 'identification': 0.9, 'ms/ms': 0.7, 'mass spectrometry': 0.8,
            'molecular formula': 0.6, 'structure': 0.5, 'peak': 0.4, 'retention time': 0.3,
            
            # Pathway analysis keywords
            'pathway': 0.9, 'kegg': 0.8, 'reactome': 0.7, 'network': 0.6, 'flux': 0.5,
            'glycolysis': 0.7, 'metabolism': 0.6, 'enzymatic': 0.4,
            
            # Biomarker discovery keywords
            'biomarker': 0.9, 'discovery': 0.8, 'diagnostic': 0.7, 'prognostic': 0.7,
            'signature': 0.6, 'panel': 0.5, 'screening': 0.4,
            
            # Clinical diagnosis keywords
            'clinical': 0.8, 'diagnosis': 0.9, 'patient': 0.7, 'medical': 0.6,
            'hospital': 0.5, 'treatment': 0.4, 'therapy': 0.4,
            
            # Drug discovery keywords
            'drug': 0.9, 'pharmaceutical': 0.8, 'compound': 0.6, 'admet': 0.7,
            'pharmacokinetic': 0.8, 'toxicity': 0.6, 'screening': 0.5,
            
            # Statistical analysis keywords
            'statistical': 0.8, 'analysis': 0.7, 'pca': 0.6, 'pls-da': 0.6,
            'machine learning': 0.7, 'classification': 0.5, 'regression': 0.5,
            
            # Data preprocessing keywords
            'preprocessing': 0.9, 'normalization': 0.7, 'quality control': 0.8,
            'batch correction': 0.6, 'missing values': 0.5, 'imputation': 0.5,
            
            # Database integration keywords
            'database': 0.8, 'hmdb': 0.7, 'integration': 0.6, 'annotation': 0.5,
            'mapping': 0.4, 'api': 0.3,
            
            # Literature search keywords
            'literature': 0.8, 'pubmed': 0.7, 'review': 0.6, 'meta-analysis': 0.7,
            'systematic': 0.6, 'bibliography': 0.5,
            
            # Knowledge extraction keywords
            'extraction': 0.8, 'mining': 0.7, 'ontology': 0.6, 'semantic': 0.5,
            'nlp': 0.6, 'text mining': 0.7
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Mock query analysis."""
        self.analysis_count += 1
        query_lower = query.lower()
        
        # Calculate keyword scores
        keyword_scores = {}
        found_keywords = []
        
        for keyword, weight in self.keyword_weights.items():
            if keyword in query_lower:
                keyword_scores[keyword] = weight
                found_keywords.append(keyword)
        
        # Determine primary category based on keywords
        category_scores = {
            MockResearchCategory.METABOLITE_IDENTIFICATION: 0,
            MockResearchCategory.PATHWAY_ANALYSIS: 0,
            MockResearchCategory.BIOMARKER_DISCOVERY: 0,
            MockResearchCategory.DRUG_DISCOVERY: 0,
            MockResearchCategory.CLINICAL_DIAGNOSIS: 0,
            MockResearchCategory.DATA_PREPROCESSING: 0,
            MockResearchCategory.STATISTICAL_ANALYSIS: 0,
            MockResearchCategory.LITERATURE_SEARCH: 0,
            MockResearchCategory.KNOWLEDGE_EXTRACTION: 0,
            MockResearchCategory.DATABASE_INTEGRATION: 0,
            MockResearchCategory.GENERAL_QUERY: 0.1  # Base score
        }
        
        # Score each category based on keywords
        metabolite_keywords = ['metabolite', 'identification', 'ms/ms', 'molecular formula', 'structure']
        pathway_keywords = ['pathway', 'kegg', 'network', 'metabolism', 'glycolysis']
        biomarker_keywords = ['biomarker', 'discovery', 'diagnostic', 'prognostic', 'signature']
        clinical_keywords = ['clinical', 'diagnosis', 'patient', 'medical']
        drug_keywords = ['drug', 'pharmaceutical', 'compound', 'admet']
        stats_keywords = ['statistical', 'analysis', 'pca', 'pls-da', 'machine learning']
        preprocessing_keywords = ['preprocessing', 'normalization', 'quality control']
        database_keywords = ['database', 'hmdb', 'integration', 'annotation']
        literature_keywords = ['literature', 'pubmed', 'review', 'meta-analysis']
        knowledge_keywords = ['extraction', 'mining', 'ontology', 'nlp', 'text mining']
        
        for keyword, score in keyword_scores.items():
            if keyword in metabolite_keywords:
                category_scores[MockResearchCategory.METABOLITE_IDENTIFICATION] += score
            elif keyword in pathway_keywords:
                category_scores[MockResearchCategory.PATHWAY_ANALYSIS] += score
            elif keyword in biomarker_keywords:
                category_scores[MockResearchCategory.BIOMARKER_DISCOVERY] += score
            elif keyword in clinical_keywords:
                category_scores[MockResearchCategory.CLINICAL_DIAGNOSIS] += score
            elif keyword in drug_keywords:
                category_scores[MockResearchCategory.DRUG_DISCOVERY] += score
            elif keyword in stats_keywords:
                category_scores[MockResearchCategory.STATISTICAL_ANALYSIS] += score
            elif keyword in preprocessing_keywords:
                category_scores[MockResearchCategory.DATA_PREPROCESSING] += score
            elif keyword in database_keywords:
                category_scores[MockResearchCategory.DATABASE_INTEGRATION] += score
            elif keyword in literature_keywords:
                category_scores[MockResearchCategory.LITERATURE_SEARCH] += score
            elif keyword in knowledge_keywords:
                category_scores[MockResearchCategory.KNOWLEDGE_EXTRACTION] += score
        
        # Find best category and confidence
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        # Calculate confidence based on score and query length
        base_confidence = min(1.0, best_score)
        length_bonus = min(0.2, len(query) / 500)  # Bonus for longer, more detailed queries
        keyword_bonus = min(0.3, len(found_keywords) / 10)  # Bonus for more keywords
        
        confidence = min(1.0, base_confidence + length_bonus + keyword_bonus)
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = 'high'
        elif confidence >= 0.6:
            confidence_level = 'medium'
        elif confidence >= 0.4:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
        
        # Determine subject area
        subject_area = None
        if any(kw in query_lower for kw in ['clinical', 'patient', 'medical', 'diagnosis']):
            subject_area = 'clinical'
        elif any(kw in query_lower for kw in ['drug', 'pharmaceutical', 'therapeutic']):
            subject_area = 'pharmaceutical'
        elif any(kw in query_lower for kw in ['metabolomics', 'metabolite']):
            subject_area = 'metabolomics'
        
        # Check for technical terms
        technical_terms = any(kw in query_lower for kw in [
            'lc-ms', 'gc-ms', 'nmr', 'ms/ms', 'uplc', 'qtof', 'orbitrap', 
            'hplc', 'ce-ms', 'ftir', 'maldi', 'esi'
        ])
        
        return {
            'category': best_category,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'keywords_found': found_keywords,
            'keyword_scores': keyword_scores,
            'all_scores': category_scores,
            'subject_area': subject_area,
            'analysis_details': {
                'query_length': len(query),
                'word_count': len(query.split()),
                'has_technical_terms': technical_terms,
                'complexity_indicators': len(found_keywords)
            }
        }


class MockResearchCategorizer:
    """Mock ResearchCategorizer for testing."""
    
    def __init__(self):
        self.query_analyzer = MockQueryAnalyzer()
        self.categorization_count = 0
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'accuracy_rate': 0.95,
            'confidence_scores': []
        }
    
    def categorize_query(self, query: str) -> CategoryPrediction:
        """Mock query categorization."""
        start_time = time.time()
        self.categorization_count += 1
        
        # Add small delay to simulate processing
        time.sleep(0.001)  # 1ms delay
        
        # Analyze query
        analysis = self.query_analyzer.analyze_query(query)
        
        # Create evidence based on found keywords
        evidence = []
        for keyword in analysis['keywords_found'][:5]:  # Top 5 keywords as evidence
            evidence.append(f"Found keyword: {keyword}")
        
        if analysis['analysis_details']['has_technical_terms']:
            evidence.append("Technical terminology detected")
        
        if analysis['analysis_details']['complexity_indicators'] > 3:
            evidence.append("Complex query with multiple indicators")
        
        # Create prediction
        prediction = CategoryPrediction(
            category=analysis['category'],
            confidence=analysis['confidence'],
            evidence=evidence,
            subject_area=analysis['subject_area'],
            metadata={
                'confidence_level': analysis['confidence_level'],
                'all_scores': analysis['all_scores'],
                'analysis_details': analysis['analysis_details'],
                'keywords_found': analysis['keywords_found'],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        )
        
        # Update performance stats
        self.performance_stats['total_queries'] += 1
        self.performance_stats['confidence_scores'].append(prediction.confidence)
        
        if len(self.performance_stats['confidence_scores']) > 100:
            self.performance_stats['confidence_scores'].pop(0)  # Keep last 100
        
        return prediction
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        confidence_scores = self.performance_stats['confidence_scores']
        return {
            'total_queries_processed': self.performance_stats['total_queries'],
            'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_std_dev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
            'estimated_accuracy': self.performance_stats['accuracy_rate']
        }


# =====================================================================
# PERFORMANCE TESTING UTILITIES
# =====================================================================

class QueryClassificationPerformanceTester:
    """Utilities for performance testing query classification."""
    
    def __init__(self):
        self.test_results = []
        self.benchmark_data = {}
    
    def measure_response_time(self, categorizer, query: str, iterations: int = 1) -> Dict[str, float]:
        """Measure response time for query categorization."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = categorizer.categorize_query(query)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'avg_time_ms': statistics.mean(times),
            'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'iterations': iterations
        }
    
    def benchmark_query_batch(self, categorizer, queries: List[str]) -> Dict[str, Any]:
        """Benchmark processing a batch of queries."""
        start_time = time.perf_counter()
        results = []
        response_times = []
        
        for query in queries:
            query_start = time.perf_counter()
            prediction = categorizer.categorize_query(query)
            query_end = time.perf_counter()
            
            query_time = (query_end - query_start) * 1000
            response_times.append(query_time)
            results.append(prediction)
        
        total_time = time.perf_counter() - start_time
        
        return {
            'total_queries': len(queries),
            'total_time_seconds': total_time,
            'throughput_queries_per_second': len(queries) / total_time,
            'avg_response_time_ms': statistics.mean(response_times),
            'min_response_time_ms': min(response_times),
            'max_response_time_ms': max(response_times),
            'results': results
        }
    
    def generate_performance_report(self, test_name: str, results: Dict[str, Any]) -> str:
        """Generate a formatted performance report."""
        report = f"""
=== Query Classification Performance Report: {test_name} ===

Total Queries: {results.get('total_queries', 'N/A')}
Total Time: {results.get('total_time_seconds', 0):.3f} seconds
Throughput: {results.get('throughput_queries_per_second', 0):.2f} queries/second

Response Times:
  Average: {results.get('avg_response_time_ms', 0):.2f}ms
  Minimum: {results.get('min_response_time_ms', 0):.2f}ms
  Maximum: {results.get('max_response_time_ms', 0):.2f}ms

Performance Thresholds:
  < 100ms: Excellent
  < 500ms: Good  
  < 1000ms: Acceptable
  > 1000ms: Poor

Assessment: {'Excellent' if results.get('avg_response_time_ms', 1000) < 100 else
             'Good' if results.get('avg_response_time_ms', 1000) < 500 else
             'Acceptable' if results.get('avg_response_time_ms', 1000) < 1000 else 'Poor'}
        """
        return report


# =====================================================================
# BIOMEDICAL QUERY FIXTURES
# =====================================================================

class BiomedicalQueryFixtures:
    """Biomedical query fixtures for testing."""
    
    # Sample queries for each category with expected results
    SAMPLE_QUERIES = {
        'metabolite_identification': [
            {
                'query': "What is the molecular structure of glucose with exact mass 180.0634?",
                'expected_category': MockResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.7,
                'description': "Simple metabolite identification query"
            },
            {
                'query': "LC-MS/MS identification of unknown metabolite using fragmentation pattern analysis",
                'expected_category': MockResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.8,
                'description': "Technical metabolite identification query"
            }
        ],
        'pathway_analysis': [
            {
                'query': "KEGG pathway enrichment analysis for diabetes metabolomics study",
                'expected_category': MockResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.8,
                'description': "Pathway analysis with database reference"
            },
            {
                'query': "How does glycolysis pathway regulation affect cellular metabolism?",
                'expected_category': MockResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.7,
                'description': "Biological pathway mechanism query"
            }
        ],
        'biomarker_discovery': [
            {
                'query': "Discovery of diagnostic biomarkers for cardiovascular disease using metabolomics",
                'expected_category': MockResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.8,
                'description': "Biomarker discovery research query"
            },
            {
                'query': "Prognostic biomarker panel development for cancer patient stratification",
                'expected_category': MockResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.9,
                'description': "Clinical biomarker application query"
            }
        ]
    }
    
    # Edge case queries for robustness testing
    EDGE_CASE_QUERIES = [
        {
            'query': "",  # Empty query
            'expected_category': MockResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.2,
            'description': "Empty query test"
        },
        {
            'query': "metabolomics",  # Single word
            'expected_category': MockResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.4,
            'description': "Single word query"
        },
        {
            'query': "How do I cook pasta using LC-MS techniques?",  # Nonsensical
            'expected_category': MockResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.3,
            'description': "Nonsensical query with technical terms"
        }
    ]
    
    # Performance test queries of varying lengths
    PERFORMANCE_QUERIES = [
        "LC-MS metabolomics",  # Short
        "Statistical analysis of metabolomics data using PCA and PLS-DA methods",  # Medium
        """Comprehensive metabolomics study investigating biomarker discovery for 
        cardiovascular disease using LC-MS/MS analysis of plasma samples from 
        patients with statistical validation using machine learning approaches 
        and pathway enrichment analysis using KEGG and Reactome databases""",  # Long
    ]
    
    @classmethod
    def get_sample_queries_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """Get sample queries for a specific category."""
        return cls.SAMPLE_QUERIES.get(category, [])
    
    @classmethod
    def get_all_sample_queries(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Get all sample queries."""
        return cls.SAMPLE_QUERIES
    
    @classmethod
    def get_edge_cases(cls) -> List[Dict[str, Any]]:
        """Get edge case queries."""
        return cls.EDGE_CASE_QUERIES
    
    @classmethod
    def get_performance_queries(cls) -> List[str]:
        """Get performance test queries."""
        return cls.PERFORMANCE_QUERIES


# =====================================================================
# PYTEST FIXTURES
# =====================================================================

@pytest.fixture
def research_categorizer():
    """Provide a mock ResearchCategorizer instance."""
    return MockResearchCategorizer()


@pytest.fixture
def mock_query_analyzer():
    """Provide a mock QueryAnalyzer instance."""
    return MockQueryAnalyzer()


@pytest.fixture
def performance_tester():
    """Provide a performance testing utility."""
    return QueryClassificationPerformanceTester()


@pytest.fixture
def biomedical_fixtures():
    """Provide biomedical query fixtures."""
    return BiomedicalQueryFixtures()


@pytest.fixture
def sample_biomedical_queries():
    """Provide sample biomedical queries for testing."""
    return BiomedicalQueryFixtures.get_all_sample_queries()


@pytest.fixture
def edge_case_queries():
    """Provide edge case queries for robustness testing."""
    return BiomedicalQueryFixtures.get_edge_cases()


@pytest.fixture
def performance_queries():
    """Provide queries for performance testing."""
    return BiomedicalQueryFixtures.get_performance_queries()


@pytest.fixture
def research_categories():
    """Provide mock research categories."""
    return MockResearchCategory


@pytest.fixture 
def performance_requirements():
    """Provide performance requirements for validation."""
    return {
        'max_response_time_ms': 1000,  # 1 second max per query
        'min_accuracy_percent': 85,    # 85% minimum accuracy
        'min_confidence_correlation': 0.7,  # Confidence should correlate with accuracy
        'max_processing_time_batch': 10.0,  # 10 seconds for 100 queries
        'memory_limit_mb': 100,  # 100MB memory limit
        'min_throughput_qps': 10   # 10 queries per second minimum
    }


@pytest.fixture
def category_prediction_factory():
    """Factory for creating CategoryPrediction instances."""
    def create_prediction(
        category: str = MockResearchCategory.GENERAL_QUERY,
        confidence: float = 0.5,
        evidence: List[str] = None,
        subject_area: str = None,
        metadata: Dict[str, Any] = None
    ) -> CategoryPrediction:
        return CategoryPrediction(
            category=category,
            confidence=confidence,
            evidence=evidence or [],
            subject_area=subject_area,
            metadata=metadata or {}
        )
    return create_prediction


@pytest.fixture
def comprehensive_query_dataset():
    """Provide a comprehensive dataset of biomedical queries for testing."""
    if BIOMEDICAL_QUERIES_AVAILABLE:
        # Use the comprehensive dataset from the standalone file
        return get_all_test_queries()
    else:
        # Fallback to local fixtures if import fails
        return BiomedicalQueryFixtures.get_all_sample_queries()


@pytest.fixture
def query_statistics():
    """Provide statistics about the query test dataset."""
    if BIOMEDICAL_QUERIES_AVAILABLE:
        return get_query_statistics()
    else:
        # Provide basic statistics for local fixtures
        all_queries = BiomedicalQueryFixtures.get_all_sample_queries()
        total_queries = sum(len(queries) for queries in all_queries.values())
        return {
            'total_queries': total_queries,
            'categories': len(all_queries),
            'edge_cases': len(BiomedicalQueryFixtures.get_edge_cases()),
            'performance_queries': len(BiomedicalQueryFixtures.get_performance_queries())
        }


# =====================================================================
# INTEGRATION FIXTURES
# =====================================================================

@pytest.fixture
def query_classification_test_environment(
    research_categorizer,
    performance_tester,
    biomedical_fixtures,
    performance_requirements
):
    """Provide a complete test environment for query classification testing."""
    
    class QueryClassificationTestEnv:
        def __init__(self):
            self.categorizer = research_categorizer
            self.performance_tester = performance_tester
            self.fixtures = biomedical_fixtures
            self.requirements = performance_requirements
            self.test_results = []
            self.start_time = time.time()
        
        def run_category_test(self, category: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Run tests for a specific category."""
            results = {
                'category': category,
                'total_queries': len(queries),
                'correct_predictions': 0,
                'confidence_scores': [],
                'response_times': []
            }
            
            for query_data in queries:
                start_time = time.perf_counter()
                prediction = self.categorizer.categorize_query(query_data['query'])
                end_time = time.perf_counter()
                
                response_time = (end_time - start_time) * 1000
                results['response_times'].append(response_time)
                results['confidence_scores'].append(prediction.confidence)
                
                if 'expected_category' in query_data:
                    if prediction.category == query_data['expected_category']:
                        results['correct_predictions'] += 1
                
                if 'expected_confidence_min' in query_data:
                    assert prediction.confidence >= query_data['expected_confidence_min'], \
                        f"Confidence {prediction.confidence} below minimum {query_data['expected_confidence_min']}"
            
            results['accuracy'] = results['correct_predictions'] / results['total_queries']
            results['avg_confidence'] = statistics.mean(results['confidence_scores'])
            results['avg_response_time'] = statistics.mean(results['response_times'])
            
            return results
        
        def generate_comprehensive_report(self) -> Dict[str, Any]:
            """Generate a comprehensive test report."""
            return {
                'test_duration': time.time() - self.start_time,
                'categorizer_stats': self.categorizer.get_performance_stats(),
                'test_results': self.test_results,
                'requirements_met': True  # Simplified for fixture
            }
    
    return QueryClassificationTestEnv()


if __name__ == "__main__":
    """Demonstrate the fixtures functionality."""
    print("=== Query Classification Test Fixtures ===")
    
    # Test mock categorizer
    categorizer = MockResearchCategorizer()
    test_query = "LC-MS metabolite identification using exact mass and fragmentation patterns"
    
    prediction = categorizer.categorize_query(test_query)
    print(f"\nTest Query: {test_query}")
    print(f"Category: {prediction.category}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Evidence: {prediction.evidence}")
    
    # Test performance measurement
    perf_tester = QueryClassificationPerformanceTester()
    perf_results = perf_tester.measure_response_time(categorizer, test_query, iterations=10)
    print(f"\nPerformance Results:")
    print(f"Average Response Time: {perf_results['avg_time_ms']:.2f}ms")
    print(f"Min/Max: {perf_results['min_time_ms']:.2f}ms / {perf_results['max_time_ms']:.2f}ms")
    
    # Show fixtures availability
    print(f"\nBiomedical Queries Available: {BIOMEDICAL_QUERIES_AVAILABLE}")
    if BIOMEDICAL_QUERIES_AVAILABLE:
        stats = get_query_statistics()
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Categories: {len(stats['category_distribution'])}")
    
    print("\n✅ Query Classification Test Fixtures Ready!")