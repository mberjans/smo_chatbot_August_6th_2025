#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Query Classification Functionality

This test suite provides comprehensive validation of the Clinical Metabolomics Oracle
query classification system, including the BiomedicalQueryRouter with keyword-based
classification, confidence scoring mechanisms, and fallback strategies.

Test Coverage:
1. Classification Accuracy Tests (>90% accuracy target)
2. Performance Tests (<2 second classification response)  
3. Confidence Scoring Tests (multi-factor confidence calculation)
4. Integration Tests (ResearchCategorizer compatibility)
5. Real-World Scenario Tests (clinical metabolomics specific queries)
6. Production Readiness Tests (stress testing, concurrent requests)

Requirements Validation:
- >90% classification accuracy target
- Performance optimization for real-time use (<2 second classification response)
- Fallback mechanisms for uncertain classifications
- System routes queries between LightRAG and Perplexity API

Author: Claude Code (Anthropic) 
Created: August 8, 2025
Task: Comprehensive Query Classification Test Implementation
"""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Set
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from contextlib import contextmanager
import random

# Import the query router and related components
from lightrag_integration.query_router import (
    BiomedicalQueryRouter,
    RoutingDecision, 
    RoutingPrediction,
    TemporalAnalyzer,
    ConfidenceMetrics,
    FallbackStrategy
)
from lightrag_integration.research_categorizer import ResearchCategorizer, CategoryPrediction
from lightrag_integration.cost_persistence import ResearchCategory

# Import test fixtures
from .test_fixtures_query_classification import (
    MockResearchCategorizer,
    BiomedicalQueryFixtures,
    QueryClassificationPerformanceTester
)


# =====================================================================
# TEST DATA AND FIXTURES
# =====================================================================

@dataclass
class ClassificationTestCase:
    """Test case for classification accuracy testing."""
    query: str
    expected_category: ResearchCategory
    expected_routing: RoutingDecision
    expected_confidence_min: float
    description: str
    complexity: str = "medium"  # simple, medium, complex
    domain_specific: bool = True
    contains_temporal_indicators: bool = False
    contains_relationship_keywords: bool = False


@dataclass 
class PerformanceTestResult:
    """Results from performance testing."""
    test_name: str
    total_queries: int
    total_time_seconds: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    std_dev_ms: float
    throughput_qps: float
    memory_usage_mb: float
    meets_requirements: bool
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyTestResult:
    """Results from accuracy testing."""
    test_name: str
    total_queries: int
    correct_classifications: int
    accuracy_percentage: float
    confidence_scores: List[float]
    avg_confidence: float
    confidence_correlation: float
    category_breakdown: Dict[str, Dict[str, int]]
    meets_requirements: bool
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)


class ComprehensiveQueryDataset:
    """Comprehensive dataset of biomedical queries for thorough testing."""
    
    def __init__(self):
        self.test_cases = self._generate_comprehensive_test_cases()
        self.edge_cases = self._generate_edge_cases()
        self.performance_cases = self._generate_performance_cases()
        self.real_world_cases = self._generate_real_world_cases()
    
    def _generate_comprehensive_test_cases(self) -> List[ClassificationTestCase]:
        """Generate comprehensive test cases covering all categories."""
        test_cases = []
        
        # Metabolite Identification Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="What is the molecular structure of glucose with exact mass 180.0634 using LC-MS?",
                expected_category=ResearchCategory.METABOLITE_IDENTIFICATION,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Specific metabolite identification with mass and method",
                complexity="medium",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="LC-MS/MS identification of unknown metabolite peak at retention time 12.3 minutes with fragmentation pattern m/z 181, 163, 145",
                expected_category=ResearchCategory.METABOLITE_IDENTIFICATION,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.9,
                description="Technical metabolite identification with detailed parameters",
                complexity="complex",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="Identify metabolite using molecular formula C6H12O6",
                expected_category=ResearchCategory.METABOLITE_IDENTIFICATION,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.7,
                description="Simple metabolite identification by formula",
                complexity="simple",
                domain_specific=True
            )
        ])
        
        # Pathway Analysis Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="KEGG pathway enrichment analysis for diabetes metabolomics study",
                expected_category=ResearchCategory.PATHWAY_ANALYSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Database-specific pathway analysis",
                complexity="medium",
                domain_specific=True,
                contains_relationship_keywords=True
            ),
            ClassificationTestCase(
                query="How does the glycolysis pathway connect to the TCA cycle in metabolic regulation?",
                expected_category=ResearchCategory.PATHWAY_ANALYSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Pathway relationship analysis",
                complexity="complex",
                domain_specific=True,
                contains_relationship_keywords=True
            ),
            ClassificationTestCase(
                query="Metabolic network analysis of fatty acid oxidation pathway",
                expected_category=ResearchCategory.PATHWAY_ANALYSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.7,
                description="Network-based pathway analysis",
                complexity="medium",
                domain_specific=True
            )
        ])
        
        # Biomarker Discovery Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="Discovery of diagnostic biomarkers for cardiovascular disease using untargeted metabolomics",
                expected_category=ResearchCategory.BIOMARKER_DISCOVERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.8,
                description="Clinical biomarker discovery research",
                complexity="complex",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="Prognostic biomarker panel for cancer patient stratification",
                expected_category=ResearchCategory.BIOMARKER_DISCOVERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.9,
                description="Clinical application biomarker panel",
                complexity="medium",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="What are the best metabolite biomarkers for diabetes diagnosis?",
                expected_category=ResearchCategory.BIOMARKER_DISCOVERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.7,
                description="General biomarker identification query",
                complexity="simple",
                domain_specific=True
            )
        ])
        
        # Drug Discovery Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="ADMET profiling of novel pharmaceutical compounds using metabolomics",
                expected_category=ResearchCategory.DRUG_DISCOVERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.8,
                description="Drug metabolism and toxicity analysis",
                complexity="complex",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="Pharmacokinetic analysis of drug metabolites in clinical trial",
                expected_category=ResearchCategory.DRUG_DISCOVERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.8,
                description="Clinical pharmacokinetics",
                complexity="medium",
                domain_specific=True
            )
        ])
        
        # Clinical Diagnosis Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="Clinical metabolomics for patient diagnosis and treatment monitoring",
                expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Clinical application for diagnosis",
                complexity="medium",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="How can metabolomic profiles be used for precision medicine in hospital settings?",
                expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.7,
                description="Precision medicine application",
                complexity="complex",
                domain_specific=True
            )
        ])
        
        # Statistical Analysis Test Cases  
        test_cases.extend([
            ClassificationTestCase(
                query="PCA and PLS-DA analysis of metabolomics data with cross-validation",
                expected_category=ResearchCategory.STATISTICAL_ANALYSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.9,
                description="Specific statistical methods for metabolomics",
                complexity="medium",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="Machine learning classification of metabolomic profiles using random forest",
                expected_category=ResearchCategory.STATISTICAL_ANALYSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="ML-based statistical analysis",
                complexity="complex",
                domain_specific=True
            )
        ])
        
        # Literature Search Test Cases (temporal)
        test_cases.extend([
            ClassificationTestCase(
                query="Latest metabolomics research publications in 2024",
                expected_category=ResearchCategory.LITERATURE_SEARCH,
                expected_routing=RoutingDecision.PERPLEXITY,
                expected_confidence_min=0.8,
                description="Recent literature search",
                complexity="simple",
                domain_specific=True,
                contains_temporal_indicators=True
            ),
            ClassificationTestCase(
                query="What are the current trends in clinical metabolomics research?",
                expected_category=ResearchCategory.LITERATURE_SEARCH,
                expected_routing=RoutingDecision.PERPLEXITY,
                expected_confidence_min=0.7,
                description="Current research trends",
                complexity="medium",
                domain_specific=True,
                contains_temporal_indicators=True
            )
        ])
        
        # Data Preprocessing Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="Quality control and normalization of LC-MS metabolomics data",
                expected_category=ResearchCategory.DATA_PREPROCESSING,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Data preprocessing methods",
                complexity="medium",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="Batch correction and missing value imputation in metabolomics datasets",
                expected_category=ResearchCategory.DATA_PREPROCESSING,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.9,
                description="Advanced preprocessing techniques",
                complexity="complex",
                domain_specific=True
            )
        ])
        
        # Database Integration Test Cases
        test_cases.extend([
            ClassificationTestCase(
                query="HMDB database integration for metabolite annotation",
                expected_category=ResearchCategory.DATABASE_INTEGRATION,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.9,
                description="Specific database integration",
                complexity="medium",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="API integration with multiple metabolomics databases for compound identification",
                expected_category=ResearchCategory.DATABASE_INTEGRATION,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Multi-database integration",
                complexity="complex",
                domain_specific=True
            )
        ])
        
        # General Queries
        test_cases.extend([
            ClassificationTestCase(
                query="What is metabolomics?",
                expected_category=ResearchCategory.GENERAL_QUERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.5,
                description="Basic definition query",
                complexity="simple",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="Explain the principles of clinical metabolomics",
                expected_category=ResearchCategory.GENERAL_QUERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.6,
                description="General explanation query",
                complexity="medium",
                domain_specific=True
            )
        ])
        
        return test_cases
    
    def _generate_edge_cases(self) -> List[ClassificationTestCase]:
        """Generate edge cases for robustness testing."""
        return [
            ClassificationTestCase(
                query="",
                expected_category=ResearchCategory.GENERAL_QUERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.0,
                description="Empty query",
                complexity="simple",
                domain_specific=False
            ),
            ClassificationTestCase(
                query="metabolomics",
                expected_category=ResearchCategory.GENERAL_QUERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.2,
                description="Single word query",
                complexity="simple",
                domain_specific=True
            ),
            ClassificationTestCase(
                query="How do I cook pasta using LC-MS techniques for better flavor profiling?",
                expected_category=ResearchCategory.GENERAL_QUERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.1,
                description="Nonsensical query with technical terms",
                complexity="medium",
                domain_specific=False
            ),
            ClassificationTestCase(
                query="a b c d e f g h i j k l m n o p q r s t u v w x y z",
                expected_category=ResearchCategory.GENERAL_QUERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.0,
                description="Random letters",
                complexity="simple",
                domain_specific=False
            ),
            ClassificationTestCase(
                query="1234567890 !@#$%^&*() metabolite identification",
                expected_category=ResearchCategory.METABOLITE_IDENTIFICATION,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.3,
                description="Special characters with valid terms",
                complexity="medium",
                domain_specific=True
            )
        ]
    
    def _generate_performance_cases(self) -> List[str]:
        """Generate queries for performance testing."""
        performance_cases = []
        
        # Short queries
        short_queries = [
            "LC-MS metabolomics",
            "pathway analysis",
            "biomarker discovery",
            "statistical analysis",
            "clinical diagnosis"
        ]
        performance_cases.extend(short_queries)
        
        # Medium queries
        medium_queries = [
            "Statistical analysis of metabolomics data using PCA and PLS-DA methods for biomarker discovery",
            "KEGG pathway enrichment analysis for diabetes metabolomics study with clinical validation",
            "LC-MS/MS identification of unknown metabolites using fragmentation pattern analysis",
            "Quality control and normalization procedures for large-scale metabolomics datasets",
            "Machine learning classification of metabolomic profiles for disease diagnosis"
        ]
        performance_cases.extend(medium_queries)
        
        # Long queries
        long_queries = [
            """Comprehensive metabolomics study investigating biomarker discovery for cardiovascular 
            disease using LC-MS/MS analysis of plasma samples from patients with statistical validation 
            using machine learning approaches including PCA, PLS-DA, and random forest classification 
            with pathway enrichment analysis using KEGG and Reactome databases for biological 
            interpretation of results in clinical diagnostic applications""",
            
            """Multi-platform metabolomics analysis combining LC-MS, GC-MS, and NMR spectroscopy 
            for comprehensive metabolite identification and quantification in clinical samples from 
            diabetes patients with data preprocessing including quality control, normalization, 
            batch correction, and missing value imputation followed by statistical analysis using 
            univariate and multivariate methods for biomarker panel development""",
            
            """Advanced bioinformatics pipeline for metabolomics data analysis including automated 
            peak detection, metabolite identification using accurate mass and fragmentation patterns, 
            statistical analysis with multiple testing correction, pathway analysis using enrichment 
            algorithms, and integration with clinical metadata for personalized medicine applications 
            in hospital settings with regulatory compliance considerations"""
        ]
        performance_cases.extend(long_queries)
        
        return performance_cases
    
    def _generate_real_world_cases(self) -> List[ClassificationTestCase]:
        """Generate real-world clinical metabolomics scenarios."""
        return [
            # Clinical workflow scenarios
            ClassificationTestCase(
                query="I have plasma samples from 200 diabetes patients and need to identify potential biomarkers. What metabolomics approach should I use?",
                expected_category=ResearchCategory.BIOMARKER_DISCOVERY,
                expected_routing=RoutingDecision.EITHER,
                expected_confidence_min=0.7,
                description="Clinical biomarker discovery consultation",
                complexity="complex",
                domain_specific=True
            ),
            
            # Laboratory workflow scenarios
            ClassificationTestCase(
                query="My LC-MS data shows contamination peaks. How should I perform quality control and data preprocessing?",
                expected_category=ResearchCategory.DATA_PREPROCESSING,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Laboratory data quality issue",
                complexity="medium",
                domain_specific=True
            ),
            
            # Research planning scenarios
            ClassificationTestCase(
                query="What is the current state of metabolomics research in Alzheimer's disease? I need recent publications and breakthrough findings.",
                expected_category=ResearchCategory.LITERATURE_SEARCH,
                expected_routing=RoutingDecision.PERPLEXITY,
                expected_confidence_min=0.8,
                description="Research planning with literature review",
                complexity="complex",
                domain_specific=True,
                contains_temporal_indicators=True
            ),
            
            # Clinical decision support scenarios
            ClassificationTestCase(
                query="A patient shows elevated lactate and decreased amino acids in metabolomic analysis. What pathways should I investigate?",
                expected_category=ResearchCategory.PATHWAY_ANALYSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.8,
                description="Clinical decision support for pathway investigation",
                complexity="complex",
                domain_specific=True,
                contains_relationship_keywords=True
            ),
            
            # Multi-modal analysis scenarios
            ClassificationTestCase(
                query="How can I integrate metabolomics data with genomics and proteomics for comprehensive patient stratification in my clinical trial?",
                expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
                expected_routing=RoutingDecision.LIGHTRAG,
                expected_confidence_min=0.7,
                description="Multi-omics integration for clinical trial",
                complexity="complex",
                domain_specific=True
            )
        ]
    
    def get_test_cases_by_category(self, category: ResearchCategory) -> List[ClassificationTestCase]:
        """Get test cases for specific category."""
        return [case for case in self.test_cases if case.expected_category == category]
    
    def get_test_cases_by_complexity(self, complexity: str) -> List[ClassificationTestCase]:
        """Get test cases by complexity level."""
        return [case for case in self.test_cases if case.complexity == complexity]
    
    def get_temporal_test_cases(self) -> List[ClassificationTestCase]:
        """Get test cases with temporal indicators."""
        return [case for case in self.test_cases if case.contains_temporal_indicators]
    
    def get_relationship_test_cases(self) -> List[ClassificationTestCase]:
        """Get test cases with relationship keywords.""" 
        return [case for case in self.test_cases if case.contains_relationship_keywords]


# =====================================================================
# PERFORMANCE MONITORING AND MEASUREMENT UTILITIES
# =====================================================================

class PerformanceMonitor:
    """Monitor performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.process = psutil.Process()
        
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        self.start_times[operation_name] = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield self
        finally:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            duration = end_time - self.start_times[operation_name]
            
            self.metrics[operation_name] = {
                'duration_seconds': duration,
                'duration_ms': duration * 1000,
                'memory_start_mb': start_memory,
                'memory_end_mb': end_memory,
                'memory_delta_mb': end_memory - start_memory,
                'timestamp': time.time()
            }
    
    def get_metrics(self, operation_name: str) -> Dict[str, float]:
        """Get metrics for specific operation."""
        return self.metrics.get(operation_name, {})
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.start_times.clear()


class ConcurrentTester:
    """Utilities for concurrent/stress testing."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.results = []
        
    def run_concurrent_queries(self, router: BiomedicalQueryRouter, 
                             queries: List[str], 
                             concurrent_requests: int = 5) -> Dict[str, Any]:
        """Run queries concurrently to test thread safety and performance."""
        results = []
        errors = []
        start_time = time.perf_counter()
        
        def process_query(query_info):
            query_id, query = query_info
            try:
                query_start = time.perf_counter()
                prediction = router.route_query(query)
                query_end = time.perf_counter()
                
                return {
                    'query_id': query_id,
                    'query': query,
                    'prediction': prediction,
                    'response_time_ms': (query_end - query_start) * 1000,
                    'success': True
                }
            except Exception as e:
                return {
                    'query_id': query_id, 
                    'query': query,
                    'error': str(e),
                    'success': False
                }
        
        # Prepare queries with IDs
        query_list = [(i, query) for i, query in enumerate(queries)]
        
        # Run concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(process_query, query_info) for query_info in query_list]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result['success']:
                    results.append(result)
                else:
                    errors.append(result)
        
        end_time = time.perf_counter()
        
        # Calculate statistics
        total_time = end_time - start_time
        successful_queries = len(results)
        failed_queries = len(errors)
        
        response_times = [r['response_time_ms'] for r in results]
        
        return {
            'total_queries': len(queries),
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'total_time_seconds': total_time,
            'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
            'min_response_time_ms': min(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'std_dev_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'throughput_qps': len(queries) / total_time if total_time > 0 else 0,
            'error_rate': failed_queries / len(queries) if queries else 0,
            'results': results,
            'errors': errors
        }


# =====================================================================
# COMPREHENSIVE TEST FIXTURES
# =====================================================================

@pytest.fixture
def comprehensive_dataset():
    """Provide comprehensive query dataset."""
    return ComprehensiveQueryDataset()

@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utility."""
    monitor = PerformanceMonitor()
    yield monitor
    # Cleanup
    monitor.clear_metrics()

@pytest.fixture
def concurrent_tester():
    """Provide concurrent testing utility."""
    return ConcurrentTester()

@pytest.fixture  
def biomedical_router():
    """Provide BiomedicalQueryRouter instance for testing."""
    # Mock logger to avoid logging setup in tests
    mock_logger = Mock()
    router = BiomedicalQueryRouter(logger=mock_logger)
    yield router

@pytest.fixture
def accuracy_requirements():
    """Define accuracy requirements for validation."""
    return {
        'min_overall_accuracy': 0.90,  # 90% minimum accuracy
        'min_category_accuracy': 0.85,  # 85% minimum per category
        'min_confidence_correlation': 0.7,  # Confidence should correlate with accuracy
        'max_false_positive_rate': 0.05,  # 5% max false positive rate
        'min_precision': 0.85,
        'min_recall': 0.80,
        'min_f1_score': 0.82
    }

@pytest.fixture
def performance_requirements():
    """Define performance requirements for validation.""" 
    return {
        'max_response_time_ms': 2000,  # 2 second max per query (requirement)
        'max_avg_response_time_ms': 1000,  # 1 second average
        'min_throughput_qps': 50,  # 50 queries per second minimum
        'max_memory_usage_mb': 500,  # 500MB max memory usage
        'max_concurrent_response_time_ms': 3000,  # 3 seconds under load
        'min_concurrent_success_rate': 0.98,  # 98% success rate under load
        'max_error_rate': 0.02  # 2% max error rate
    }


# =====================================================================
# CLASSIFICATION ACCURACY TESTS
# =====================================================================

@pytest.mark.biomedical
class TestClassificationAccuracy:
    """Test suite for classification accuracy validation."""
    
    def test_overall_classification_accuracy(self, biomedical_router, comprehensive_dataset, accuracy_requirements):
        """Test overall classification accuracy meets >90% requirement."""
        test_cases = comprehensive_dataset.test_cases
        correct_classifications = 0
        confidence_scores = []
        detailed_results = []
        category_breakdown = {}
        
        for test_case in test_cases:
            # Skip edge cases for accuracy testing
            if not test_case.domain_specific:
                continue
                
            prediction = biomedical_router.route_query(test_case.query)
            
            # Check if classification is correct
            is_correct = prediction.research_category == test_case.expected_category
            if is_correct:
                correct_classifications += 1
            
            confidence_scores.append(prediction.confidence)
            
            # Track category-specific results
            expected_cat = test_case.expected_category.value
            if expected_cat not in category_breakdown:
                category_breakdown[expected_cat] = {'correct': 0, 'total': 0}
            
            category_breakdown[expected_cat]['total'] += 1
            if is_correct:
                category_breakdown[expected_cat]['correct'] += 1
            
            detailed_results.append({
                'query': test_case.query,
                'expected_category': test_case.expected_category.value,
                'predicted_category': prediction.research_category.value,
                'expected_routing': test_case.expected_routing.value,
                'predicted_routing': prediction.routing_decision.value,
                'confidence': prediction.confidence,
                'is_correct': is_correct,
                'description': test_case.description
            })
        
        # Calculate overall accuracy
        domain_specific_cases = [tc for tc in test_cases if tc.domain_specific]
        total_queries = len(domain_specific_cases)
        accuracy = correct_classifications / total_queries if total_queries > 0 else 0
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # Calculate confidence correlation (higher confidence should correlate with accuracy)
        confidence_accuracy_correlation = 0
        if len(detailed_results) > 1:
            accuracies = [1.0 if r['is_correct'] else 0.0 for r in detailed_results]
            confidences = [r['confidence'] for r in detailed_results]
            
            if len(set(confidences)) > 1:  # Avoid correlation calculation if all confidences are the same
                import numpy as np
                confidence_accuracy_correlation = np.corrcoef(confidences, accuracies)[0, 1]
        
        # Create test result
        result = AccuracyTestResult(
            test_name="Overall Classification Accuracy",
            total_queries=total_queries,
            correct_classifications=correct_classifications,
            accuracy_percentage=accuracy * 100,
            confidence_scores=confidence_scores,
            avg_confidence=avg_confidence,
            confidence_correlation=confidence_accuracy_correlation,
            category_breakdown=category_breakdown,
            meets_requirements=accuracy >= accuracy_requirements['min_overall_accuracy'],
            detailed_results=detailed_results
        )
        
        # Assertions for requirements
        assert accuracy >= accuracy_requirements['min_overall_accuracy'], \
            f"Overall accuracy {accuracy:.3f} below required {accuracy_requirements['min_overall_accuracy']}"
        
        assert avg_confidence >= 0.5, f"Average confidence {avg_confidence:.3f} too low"
        
        # Check confidence correlation if we have variation in confidence scores
        if len(set(confidence_scores)) > 1:
            assert confidence_accuracy_correlation >= accuracy_requirements['min_confidence_correlation'], \
                f"Confidence-accuracy correlation {confidence_accuracy_correlation:.3f} below required {accuracy_requirements['min_confidence_correlation']}"
        
        print(f"\n=== Classification Accuracy Results ===")
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Total Queries: {total_queries}")
        print(f"Correct Classifications: {correct_classifications}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Confidence-Accuracy Correlation: {confidence_accuracy_correlation:.3f}")
        
        return result
    
    def test_category_specific_accuracy(self, biomedical_router, comprehensive_dataset, accuracy_requirements):
        """Test accuracy for each research category individually."""
        category_results = {}
        
        for category in ResearchCategory:
            category_test_cases = comprehensive_dataset.get_test_cases_by_category(category)
            
            if not category_test_cases:
                continue
                
            correct = 0
            total = 0
            confidence_scores = []
            
            for test_case in category_test_cases:
                if not test_case.domain_specific:
                    continue
                    
                prediction = biomedical_router.route_query(test_case.query)
                total += 1
                
                if prediction.research_category == category:
                    correct += 1
                
                confidence_scores.append(prediction.confidence)
            
            if total > 0:
                accuracy = correct / total
                avg_confidence = statistics.mean(confidence_scores)
                
                category_results[category.value] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'avg_confidence': avg_confidence
                }
                
                # Check minimum category accuracy
                assert accuracy >= accuracy_requirements['min_category_accuracy'], \
                    f"Category {category.value} accuracy {accuracy:.3f} below minimum {accuracy_requirements['min_category_accuracy']}"
        
        print(f"\n=== Category-Specific Accuracy ===")
        for category, metrics in category_results.items():
            print(f"{category}: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%) - {metrics['correct']}/{metrics['total']}")
        
        return category_results
    
    def test_temporal_query_accuracy(self, biomedical_router, comprehensive_dataset):
        """Test accuracy for temporal queries that should route to Perplexity."""
        temporal_cases = comprehensive_dataset.get_temporal_test_cases()
        
        correct_routing = 0
        total_temporal = 0
        
        for test_case in temporal_cases:
            prediction = biomedical_router.route_query(test_case.query)
            total_temporal += 1
            
            # Temporal queries should route to Perplexity, Either, or Hybrid
            if prediction.routing_decision in [RoutingDecision.PERPLEXITY, RoutingDecision.EITHER, RoutingDecision.HYBRID]:
                correct_routing += 1
        
        temporal_accuracy = correct_routing / total_temporal if total_temporal > 0 else 0
        
        assert temporal_accuracy >= 0.8, \
            f"Temporal routing accuracy {temporal_accuracy:.3f} below 80%"
        
        print(f"\nTemporal Query Routing Accuracy: {temporal_accuracy:.3f} ({temporal_accuracy*100:.1f}%)")
        return temporal_accuracy
    
    def test_relationship_query_accuracy(self, biomedical_router, comprehensive_dataset):
        """Test accuracy for relationship queries that should route to LightRAG."""
        relationship_cases = comprehensive_dataset.get_relationship_test_cases()
        
        correct_routing = 0 
        total_relationship = 0
        
        for test_case in relationship_cases:
            prediction = biomedical_router.route_query(test_case.query)
            total_relationship += 1
            
            # Relationship queries should route to LightRAG, Either, or Hybrid
            if prediction.routing_decision in [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER, RoutingDecision.HYBRID]:
                correct_routing += 1
        
        relationship_accuracy = correct_routing / total_relationship if total_relationship > 0 else 0
        
        assert relationship_accuracy >= 0.8, \
            f"Relationship routing accuracy {relationship_accuracy:.3f} below 80%"
        
        print(f"Relationship Query Routing Accuracy: {relationship_accuracy:.3f} ({relationship_accuracy*100:.1f}%)")
        return relationship_accuracy
    
    def test_edge_case_robustness(self, biomedical_router, comprehensive_dataset):
        """Test robustness on edge cases."""
        edge_cases = comprehensive_dataset.edge_cases
        
        successful_predictions = 0
        confidence_scores = []
        
        for test_case in edge_cases:
            try:
                prediction = biomedical_router.route_query(test_case.query)
                
                # Should always return a valid prediction
                assert isinstance(prediction, RoutingPrediction)
                assert isinstance(prediction.routing_decision, RoutingDecision)
                assert 0.0 <= prediction.confidence <= 1.0
                assert isinstance(prediction.reasoning, list)
                
                successful_predictions += 1
                confidence_scores.append(prediction.confidence)
                
            except Exception as e:
                pytest.fail(f"Edge case failed: {test_case.query} - {str(e)}")
        
        success_rate = successful_predictions / len(edge_cases) if edge_cases else 0
        avg_edge_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        assert success_rate >= 1.0, "All edge cases should return valid predictions"
        
        print(f"\nEdge Case Robustness: {success_rate:.3f} ({success_rate*100:.1f}%)")
        print(f"Average Edge Case Confidence: {avg_edge_confidence:.3f}")
        
        return success_rate, avg_edge_confidence


# =====================================================================
# PERFORMANCE TESTS
# =====================================================================

@pytest.mark.performance
class TestPerformanceRequirements:
    """Test suite for performance validation (<2 second requirement)."""
    
    def test_single_query_response_time(self, biomedical_router, comprehensive_dataset, performance_requirements):
        """Test individual query response time meets <2 second requirement."""
        test_cases = comprehensive_dataset.test_cases[:50]  # Test subset for performance
        response_times = []
        slow_queries = []
        
        for test_case in test_cases:
            start_time = time.perf_counter()
            prediction = biomedical_router.route_query(test_case.query)
            end_time = time.perf_counter()
            
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            
            if response_time_ms > performance_requirements['max_response_time_ms']:
                slow_queries.append({
                    'query': test_case.query[:100] + "...",
                    'response_time_ms': response_time_ms,
                    'description': test_case.description
                })
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        # Check requirements
        queries_over_limit = len([t for t in response_times if t > performance_requirements['max_response_time_ms']])
        over_limit_percentage = queries_over_limit / len(response_times) * 100
        
        assert avg_response_time <= performance_requirements['max_avg_response_time_ms'], \
            f"Average response time {avg_response_time:.2f}ms exceeds limit {performance_requirements['max_avg_response_time_ms']}ms"
        
        assert over_limit_percentage <= 5.0, \
            f"{over_limit_percentage:.1f}% of queries exceed 2 second limit (max allowed: 5%)"
        
        result = PerformanceTestResult(
            test_name="Single Query Response Time",
            total_queries=len(test_cases),
            total_time_seconds=sum(response_times) / 1000,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            std_dev_ms=std_dev,
            throughput_qps=0,  # Not applicable for individual queries
            memory_usage_mb=0,  # Measured separately
            meets_requirements=avg_response_time <= performance_requirements['max_avg_response_time_ms'],
            detailed_metrics={'slow_queries': slow_queries, 'over_limit_count': queries_over_limit}
        )
        
        print(f"\n=== Single Query Performance ===")
        print(f"Average Response Time: {avg_response_time:.2f}ms")
        print(f"Min/Max Response Time: {min_response_time:.2f}ms / {max_response_time:.2f}ms")
        print(f"Queries Over 2s Limit: {queries_over_limit}/{len(test_cases)} ({over_limit_percentage:.1f}%)")
        
        if slow_queries:
            print(f"\nSlowest Queries:")
            for sq in slow_queries[:5]:
                print(f"  {sq['response_time_ms']:.2f}ms: {sq['description']}")
        
        return result
    
    def test_batch_processing_throughput(self, biomedical_router, comprehensive_dataset, performance_requirements):
        """Test batch processing throughput meets throughput requirements."""
        test_queries = comprehensive_dataset.performance_cases
        
        start_time = time.perf_counter()
        response_times = []
        predictions = []
        
        for query in test_queries:
            query_start = time.perf_counter()
            prediction = biomedical_router.route_query(query)
            query_end = time.perf_counter()
            
            response_times.append((query_end - query_start) * 1000)
            predictions.append(prediction)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = len(test_queries) / total_time
        
        avg_response_time = statistics.mean(response_times)
        
        assert throughput >= performance_requirements['min_throughput_qps'], \
            f"Throughput {throughput:.2f} QPS below minimum {performance_requirements['min_throughput_qps']} QPS"
        
        result = PerformanceTestResult(
            test_name="Batch Processing Throughput",
            total_queries=len(test_queries),
            total_time_seconds=total_time,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min(response_times),
            max_response_time_ms=max(response_times),
            std_dev_ms=statistics.stdev(response_times) if len(response_times) > 1 else 0,
            throughput_qps=throughput,
            memory_usage_mb=0,  # Measured separately
            meets_requirements=throughput >= performance_requirements['min_throughput_qps']
        )
        
        print(f"\n=== Batch Processing Performance ===")
        print(f"Total Queries: {len(test_queries)}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Throughput: {throughput:.2f} QPS")
        print(f"Average Response Time: {avg_response_time:.2f}ms")
        
        return result
    
    def test_concurrent_query_performance(self, biomedical_router, comprehensive_dataset, 
                                         concurrent_tester, performance_requirements):
        """Test performance under concurrent load."""
        test_queries = comprehensive_dataset.performance_cases * 5  # 5x multiplier for more load
        concurrent_requests = 10
        
        result = concurrent_tester.run_concurrent_queries(
            biomedical_router, 
            test_queries,
            concurrent_requests=concurrent_requests
        )
        
        # Check requirements
        avg_response_time = result['avg_response_time_ms']
        success_rate = result['successful_queries'] / result['total_queries']
        error_rate = result['error_rate']
        
        assert avg_response_time <= performance_requirements['max_concurrent_response_time_ms'], \
            f"Concurrent avg response time {avg_response_time:.2f}ms exceeds limit {performance_requirements['max_concurrent_response_time_ms']}ms"
        
        assert success_rate >= performance_requirements['min_concurrent_success_rate'], \
            f"Concurrent success rate {success_rate:.3f} below minimum {performance_requirements['min_concurrent_success_rate']}"
        
        assert error_rate <= performance_requirements['max_error_rate'], \
            f"Error rate {error_rate:.3f} exceeds maximum {performance_requirements['max_error_rate']}"
        
        perf_result = PerformanceTestResult(
            test_name="Concurrent Query Performance",
            total_queries=result['total_queries'],
            total_time_seconds=result['total_time_seconds'],
            avg_response_time_ms=result['avg_response_time_ms'],
            min_response_time_ms=result['min_response_time_ms'],
            max_response_time_ms=result['max_response_time_ms'],
            std_dev_ms=result['std_dev_ms'],
            throughput_qps=result['throughput_qps'],
            memory_usage_mb=0,  # Measured separately
            meets_requirements=(avg_response_time <= performance_requirements['max_concurrent_response_time_ms'] 
                              and success_rate >= performance_requirements['min_concurrent_success_rate']),
            detailed_metrics={
                'concurrent_requests': concurrent_requests,
                'success_rate': success_rate,
                'error_rate': error_rate,
                'failed_queries': result['failed_queries']
            }
        )
        
        print(f"\n=== Concurrent Performance ===")
        print(f"Concurrent Requests: {concurrent_requests}")
        print(f"Total Queries: {result['total_queries']}")
        print(f"Success Rate: {success_rate:.3f} ({success_rate*100:.1f}%)")
        print(f"Error Rate: {error_rate:.3f} ({error_rate*100:.1f}%)")
        print(f"Average Response Time: {avg_response_time:.2f}ms")
        print(f"Throughput: {result['throughput_qps']:.2f} QPS")
        
        return perf_result
    
    def test_memory_usage_performance(self, biomedical_router, comprehensive_dataset, 
                                     performance_monitor, performance_requirements):
        """Test memory usage during query processing."""
        test_queries = comprehensive_dataset.performance_cases * 3
        
        # Force garbage collection before test
        gc.collect()
        
        with performance_monitor.monitor_operation("memory_test"):
            for query in test_queries:
                prediction = biomedical_router.route_query(query)
                
                # Periodically check memory to catch leaks
                if len(test_queries) % 10 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    assert current_memory <= performance_requirements['max_memory_usage_mb'], \
                        f"Memory usage {current_memory:.2f}MB exceeds limit {performance_requirements['max_memory_usage_mb']}MB"
        
        metrics = performance_monitor.get_metrics("memory_test")
        
        peak_memory = metrics['memory_end_mb']
        memory_delta = metrics['memory_delta_mb'] 
        
        assert peak_memory <= performance_requirements['max_memory_usage_mb'], \
            f"Peak memory usage {peak_memory:.2f}MB exceeds limit {performance_requirements['max_memory_usage_mb']}MB"
        
        print(f"\n=== Memory Performance ===")
        print(f"Peak Memory Usage: {peak_memory:.2f}MB")
        print(f"Memory Delta: {memory_delta:.2f}MB")
        print(f"Total Queries Processed: {len(test_queries)}")
        
        return peak_memory, memory_delta
    
    def test_cold_start_performance(self, biomedical_router, comprehensive_dataset):
        """Test cold start performance (first query after initialization)."""
        # Create fresh router instance
        fresh_router = BiomedicalQueryRouter(logger=Mock())
        test_query = "LC-MS metabolite identification using exact mass"
        
        # Measure first query (cold start)
        start_time = time.perf_counter()
        first_prediction = fresh_router.route_query(test_query)
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Measure subsequent queries (warm)
        warm_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            prediction = fresh_router.route_query(test_query)
            warm_time = (time.perf_counter() - start_time) * 1000
            warm_times.append(warm_time)
        
        avg_warm_time = statistics.mean(warm_times)
        
        # Cold start should be reasonable (within 5x of warm time)
        cold_start_acceptable = first_time <= (avg_warm_time * 5)
        
        assert cold_start_acceptable, \
            f"Cold start time {first_time:.2f}ms too slow compared to warm time {avg_warm_time:.2f}ms"
        
        print(f"\n=== Cold Start Performance ===")
        print(f"Cold Start Time: {first_time:.2f}ms")
        print(f"Average Warm Time: {avg_warm_time:.2f}ms")
        print(f"Cold/Warm Ratio: {first_time/avg_warm_time:.2f}x")
        
        return first_time, avg_warm_time


# =====================================================================
# CONFIDENCE SCORING TESTS  
# =====================================================================

@pytest.mark.biomedical
class TestConfidenceScoring:
    """Test suite for confidence scoring mechanism validation."""
    
    def test_confidence_metrics_structure(self, biomedical_router, comprehensive_dataset):
        """Test that confidence metrics contain all required components."""
        test_query = "LC-MS metabolite identification using fragmentation patterns"
        prediction = biomedical_router.route_query(test_query)
        
        # Check ConfidenceMetrics structure
        metrics = prediction.confidence_metrics
        assert isinstance(metrics, ConfidenceMetrics)
        
        # Check all required fields
        required_fields = [
            'overall_confidence', 'research_category_confidence', 'temporal_analysis_confidence',
            'signal_strength_confidence', 'context_coherence_confidence', 'keyword_density',
            'pattern_match_strength', 'biomedical_entity_count', 'ambiguity_score', 
            'conflict_score', 'alternative_interpretations', 'calculation_time_ms'
        ]
        
        for field in required_fields:
            assert hasattr(metrics, field), f"ConfidenceMetrics missing field: {field}"
            
        # Check value ranges
        assert 0.0 <= metrics.overall_confidence <= 1.0
        assert 0.0 <= metrics.research_category_confidence <= 1.0
        assert 0.0 <= metrics.temporal_analysis_confidence <= 1.0
        assert 0.0 <= metrics.signal_strength_confidence <= 1.0
        assert 0.0 <= metrics.context_coherence_confidence <= 1.0
        assert 0.0 <= metrics.keyword_density <= 1.0
        assert 0.0 <= metrics.pattern_match_strength <= 1.0
        assert metrics.biomedical_entity_count >= 0
        assert 0.0 <= metrics.ambiguity_score <= 1.0
        assert 0.0 <= metrics.conflict_score <= 1.0
        assert metrics.calculation_time_ms >= 0
        
        print(f"\n=== Confidence Metrics Structure ===")
        print(f"Overall Confidence: {metrics.overall_confidence:.3f}")
        print(f"Component Confidences: RC={metrics.research_category_confidence:.3f}, "
              f"TA={metrics.temporal_analysis_confidence:.3f}, SS={metrics.signal_strength_confidence:.3f}, "
              f"CC={metrics.context_coherence_confidence:.3f}")
        print(f"Signal Metrics: Density={metrics.keyword_density:.3f}, "
              f"Pattern={metrics.pattern_match_strength:.3f}, Entities={metrics.biomedical_entity_count}")
        print(f"Uncertainty: Ambiguity={metrics.ambiguity_score:.3f}, Conflict={metrics.conflict_score:.3f}")
        
        return metrics
    
    def test_confidence_consistency_across_queries(self, biomedical_router, comprehensive_dataset):
        """Test that confidence scoring is consistent across similar queries."""
        # Test similar queries should have similar confidence
        similar_queries = [
            "LC-MS metabolite identification using exact mass",
            "Mass spectrometry metabolite identification with accurate mass",
            "Metabolite identification using LC-MS exact mass measurement"
        ]
        
        confidences = []
        predictions = []
        
        for query in similar_queries:
            prediction = biomedical_router.route_query(query)
            confidences.append(prediction.confidence)
            predictions.append(prediction)
        
        # Check that similar queries have similar confidence (within 0.2 range)
        confidence_range = max(confidences) - min(confidences)
        assert confidence_range <= 0.2, \
            f"Similar queries have too much confidence variation: {confidence_range:.3f}"
        
        # Check that they classify to the same category
        categories = [p.research_category for p in predictions]
        unique_categories = set(categories)
        assert len(unique_categories) == 1, \
            f"Similar queries classified to different categories: {unique_categories}"
        
        print(f"\n=== Confidence Consistency ===")
        for i, (query, conf) in enumerate(zip(similar_queries, confidences)):
            print(f"Query {i+1}: {conf:.3f} - {query[:50]}...")
        print(f"Confidence Range: {confidence_range:.3f}")
        
        return confidences
    
    def test_confidence_correlation_with_complexity(self, biomedical_router, comprehensive_dataset):
        """Test that confidence correlates appropriately with query complexity."""
        simple_queries = comprehensive_dataset.get_test_cases_by_complexity("simple")[:10]
        complex_queries = comprehensive_dataset.get_test_cases_by_complexity("complex")[:10]
        
        simple_confidences = []
        complex_confidences = []
        
        for test_case in simple_queries:
            if test_case.domain_specific:  # Only test domain-specific queries
                prediction = biomedical_router.route_query(test_case.query)
                simple_confidences.append(prediction.confidence)
        
        for test_case in complex_queries:
            if test_case.domain_specific:
                prediction = biomedical_router.route_query(test_case.query)
                complex_confidences.append(prediction.confidence)
        
        if simple_confidences and complex_confidences:
            avg_simple = statistics.mean(simple_confidences)
            avg_complex = statistics.mean(complex_confidences)
            
            # Complex queries might have higher confidence due to more specific terms
            # But both should be above reasonable thresholds
            assert avg_simple >= 0.4, f"Simple query confidence too low: {avg_simple:.3f}"
            assert avg_complex >= 0.5, f"Complex query confidence too low: {avg_complex:.3f}"
            
            print(f"\n=== Confidence vs Complexity ===")
            print(f"Simple Queries Average Confidence: {avg_simple:.3f}")
            print(f"Complex Queries Average Confidence: {avg_complex:.3f}")
            print(f"Complexity Confidence Difference: {avg_complex - avg_simple:.3f}")
            
            return avg_simple, avg_complex
    
    def test_fallback_strategy_triggers(self, biomedical_router, comprehensive_dataset):
        """Test that fallback strategies are triggered appropriately."""
        # Test with low-confidence scenarios
        low_confidence_queries = [
            "xyz abc def",  # Nonsensical
            "metabolomics maybe",  # Vague
            "",  # Empty
            "what?",  # Too simple
        ]
        
        fallback_triggers = 0
        predictions_with_fallback = []
        
        for query in low_confidence_queries:
            prediction = biomedical_router.route_query(query)
            
            if prediction.should_use_fallback():
                fallback_triggers += 1
                predictions_with_fallback.append({
                    'query': query,
                    'confidence': prediction.confidence,
                    'fallback_strategy': prediction.fallback_strategy.strategy_type if prediction.fallback_strategy else None,
                    'routing_decision': prediction.routing_decision.value
                })
        
        # At least some low-confidence queries should trigger fallback
        fallback_rate = fallback_triggers / len(low_confidence_queries)
        assert fallback_rate >= 0.5, \
            f"Fallback strategies not triggered enough: {fallback_rate:.3f}"
        
        print(f"\n=== Fallback Strategy Triggers ===")
        print(f"Queries Triggering Fallback: {fallback_triggers}/{len(low_confidence_queries)} ({fallback_rate*100:.1f}%)")
        
        for pred in predictions_with_fallback:
            print(f"  Conf: {pred['confidence']:.3f}, Strategy: {pred['fallback_strategy']}, "
                  f"Route: {pred['routing_decision']} - {pred['query'][:30]}...")
        
        return fallback_rate
    
    def test_alternative_interpretations_quality(self, biomedical_router, comprehensive_dataset):
        """Test quality of alternative routing interpretations."""
        test_queries = [
            "Latest metabolic pathway research published in 2024",  # Should have temporal vs knowledge conflict
            "Clinical biomarker discovery using statistical analysis",  # Multiple valid categories
            "How do metabolites relate to disease mechanisms?",  # Relationship-focused
        ]
        
        for query in test_queries:
            prediction = biomedical_router.route_query(query)
            alternatives = prediction.confidence_metrics.alternative_interpretations
            
            # Should have multiple alternatives
            assert len(alternatives) >= 2, \
                f"Query should have multiple interpretations: {query}"
            
            # Alternatives should be sorted by confidence (descending)
            confidences = [alt[1] for alt in alternatives]
            assert confidences == sorted(confidences, reverse=True), \
                f"Alternatives not sorted by confidence: {confidences}"
            
            # All alternatives should have valid routing decisions and reasonable confidence
            for routing_decision, confidence in alternatives:
                assert isinstance(routing_decision, RoutingDecision)
                assert 0.0 <= confidence <= 1.0
            
            print(f"\n=== Alternative Interpretations ===")
            print(f"Query: {query}")
            print(f"Primary: {prediction.routing_decision.value} ({prediction.confidence:.3f})")
            print("Alternatives:")
            for routing_decision, confidence in alternatives[:3]:  # Top 3
                print(f"  {routing_decision.value}: {confidence:.3f}")
    
    def test_confidence_calculation_performance(self, biomedical_router, comprehensive_dataset, performance_requirements):
        """Test that confidence calculation is performant."""
        test_queries = comprehensive_dataset.performance_cases[:20]
        calculation_times = []
        
        for query in test_queries:
            prediction = biomedical_router.route_query(query)
            calc_time = prediction.confidence_metrics.calculation_time_ms
            calculation_times.append(calc_time)
        
        avg_calc_time = statistics.mean(calculation_times)
        max_calc_time = max(calculation_times)
        
        # Confidence calculation should be fast (< 50ms)
        assert avg_calc_time <= 50.0, \
            f"Average confidence calculation time {avg_calc_time:.2f}ms too slow"
        
        assert max_calc_time <= 100.0, \
            f"Max confidence calculation time {max_calc_time:.2f}ms too slow"
        
        print(f"\n=== Confidence Calculation Performance ===")
        print(f"Average Calculation Time: {avg_calc_time:.2f}ms")
        print(f"Max Calculation Time: {max_calc_time:.2f}ms")
        print(f"Queries Tested: {len(test_queries)}")
        
        return avg_calc_time, max_calc_time


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

@pytest.mark.integration
class TestIntegrationWithResearchCategorizer:
    """Test integration with existing ResearchCategorizer."""
    
    def test_inheritance_compatibility(self, biomedical_router):
        """Test that BiomedicalQueryRouter properly inherits ResearchCategorizer."""
        # Should have all parent methods
        assert hasattr(biomedical_router, 'categorize_query')
        assert hasattr(biomedical_router, 'get_category_statistics') 
        assert hasattr(biomedical_router, 'update_from_feedback')
        
        # Test basic categorization functionality
        query = "What is metabolomics?"
        category_prediction = biomedical_router.categorize_query(query)
        
        assert hasattr(category_prediction, 'category')
        assert hasattr(category_prediction, 'confidence')
        assert isinstance(category_prediction.category, ResearchCategory)
        assert 0.0 <= category_prediction.confidence <= 1.0
        
        # Test routing functionality
        routing_prediction = biomedical_router.route_query(query)
        
        assert isinstance(routing_prediction, RoutingPrediction)
        assert isinstance(routing_prediction.routing_decision, RoutingDecision)
        assert routing_prediction.research_category == category_prediction.category
        
        print(f"\n=== Integration Compatibility ===")
        print(f"Categorization: {category_prediction.category.value} ({category_prediction.confidence:.3f})")
        print(f"Routing: {routing_prediction.routing_decision.value} ({routing_prediction.confidence:.3f})")
        
        return True
    
    def test_category_to_routing_consistency(self, biomedical_router, comprehensive_dataset):
        """Test consistency between category classification and routing decisions."""
        test_cases = comprehensive_dataset.test_cases[:20]
        
        inconsistencies = []
        
        for test_case in test_cases:
            category_pred = biomedical_router.categorize_query(test_case.query)
            routing_pred = biomedical_router.route_query(test_case.query)
            
            # Categories should match
            if category_pred.category != routing_pred.research_category:
                inconsistencies.append({
                    'query': test_case.query,
                    'category_result': category_pred.category.value,
                    'routing_result': routing_pred.research_category.value
                })
        
        inconsistency_rate = len(inconsistencies) / len(test_cases)
        
        assert inconsistency_rate <= 0.05, \
            f"Too many category/routing inconsistencies: {inconsistency_rate:.3f}"
        
        print(f"\n=== Category-Routing Consistency ===")
        print(f"Inconsistencies: {len(inconsistencies)}/{len(test_cases)} ({inconsistency_rate*100:.1f}%)")
        
        if inconsistencies:
            print("Sample inconsistencies:")
            for inc in inconsistencies[:3]:
                print(f"  {inc['category_result']} vs {inc['routing_result']}: {inc['query'][:50]}...")
        
        return inconsistency_rate
    
    def test_statistics_integration(self, biomedical_router, comprehensive_dataset):
        """Test that routing statistics integrate with categorization statistics."""
        # Process some queries to generate statistics
        test_queries = [tc.query for tc in comprehensive_dataset.test_cases[:10]]
        
        for query in test_queries:
            biomedical_router.route_query(query)
        
        # Get both types of statistics
        category_stats = biomedical_router.get_category_statistics()
        routing_stats = biomedical_router.get_routing_statistics()
        
        # Check that routing stats include category stats
        assert 'total_predictions' in routing_stats
        assert 'confidence_distribution' in routing_stats
        assert 'routing_thresholds' in routing_stats
        assert 'category_routing_map' in routing_stats
        
        # Check that prediction counts match
        assert category_stats['total_predictions'] == routing_stats['total_predictions']
        
        print(f"\n=== Statistics Integration ===")
        print(f"Total Predictions: {category_stats['total_predictions']}")
        print(f"Categories Tracked: {len(category_stats.get('category_distribution', {}))}")
        print(f"Routing Map Entries: {len(routing_stats.get('category_routing_map', {}))}")
        
        return category_stats, routing_stats
    
    def test_feedback_integration(self, biomedical_router):
        """Test that feedback mechanisms work with routing functionality."""
        query = "LC-MS metabolite identification"
        
        # Get initial prediction
        initial_prediction = biomedical_router.route_query(query)
        
        # Provide feedback (using parent method)
        feedback = {
            'query': query,
            'actual_category': ResearchCategory.METABOLITE_IDENTIFICATION,
            'predicted_category': initial_prediction.research_category,
            'was_correct': True,
            'user_rating': 5
        }
        
        # Should not raise exception
        try:
            biomedical_router.update_from_feedback(feedback)
            feedback_success = True
        except Exception as e:
            feedback_success = False
            print(f"Feedback integration failed: {e}")
        
        assert feedback_success, "Feedback integration should work without errors"
        
        print(f"\n=== Feedback Integration ===")
        print(f"Feedback processed successfully: {feedback_success}")
        
        return feedback_success


# =====================================================================
# REAL-WORLD SCENARIO TESTS
# =====================================================================

@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world clinical metabolomics scenarios."""
    
    def test_clinical_workflow_sequence(self, biomedical_router, comprehensive_dataset):
        """Test a realistic clinical metabolomics workflow."""
        workflow_queries = [
            ("What is clinical metabolomics and how is it used in patient care?", ResearchCategory.GENERAL_QUERY),
            ("I have plasma samples from 200 diabetes patients. What metabolomics approach should I use?", ResearchCategory.BIOMARKER_DISCOVERY),
            ("How do I prepare plasma samples for LC-MS metabolomics analysis?", ResearchCategory.DATA_PREPROCESSING),
            ("LC-MS data shows contamination peaks. How should I perform quality control?", ResearchCategory.DATA_PREPROCESSING),
            ("What statistical methods are best for metabolomics biomarker discovery?", ResearchCategory.STATISTICAL_ANALYSIS),
            ("PCA shows clear separation between groups. How do I identify the discriminating metabolites?", ResearchCategory.METABOLITE_IDENTIFICATION),
            ("I found elevated glucose and lactate levels. What metabolic pathways should I investigate?", ResearchCategory.PATHWAY_ANALYSIS),
            ("What are the latest research findings on diabetes metabolomics published in 2024?", ResearchCategory.LITERATURE_SEARCH),
            ("How can I validate these biomarkers in an independent patient cohort?", ResearchCategory.CLINICAL_DIAGNOSIS),
        ]
        
        context = {'previous_categories': [], 'workflow_stage': 'planning'}
        workflow_results = []
        
        for i, (query, expected_category) in enumerate(workflow_queries):
            prediction = biomedical_router.route_query(query, context)
            
            # Check that prediction is reasonable for workflow stage
            assert isinstance(prediction, RoutingPrediction)
            assert prediction.confidence > 0.3, f"Low confidence for workflow query: {query[:50]}..."
            
            workflow_results.append({
                'step': i + 1,
                'query': query,
                'expected_category': expected_category.value,
                'predicted_category': prediction.research_category.value,
                'routing_decision': prediction.routing_decision.value,
                'confidence': prediction.confidence,
                'correct_category': prediction.research_category == expected_category
            })
            
            # Update context for next query
            context['previous_categories'].append(prediction.research_category.value)
        
        # Calculate workflow accuracy
        correct_predictions = sum(r['correct_category'] for r in workflow_results)
        workflow_accuracy = correct_predictions / len(workflow_results)
        
        assert workflow_accuracy >= 0.8, \
            f"Clinical workflow accuracy {workflow_accuracy:.3f} below 80%"
        
        print(f"\n=== Clinical Workflow Sequence ===")
        print(f"Workflow Accuracy: {workflow_accuracy:.3f} ({workflow_accuracy*100:.1f}%)")
        print(f"Correct Predictions: {correct_predictions}/{len(workflow_results)}")
        
        for result in workflow_results:
            status = "✓" if result['correct_category'] else "✗"
            print(f"  Step {result['step']} {status}: {result['predicted_category']} "
                  f"({result['confidence']:.3f}) - {result['query'][:60]}...")
        
        return workflow_accuracy, workflow_results
    
    def test_laboratory_troubleshooting_scenarios(self, biomedical_router):
        """Test laboratory troubleshooting scenarios.""" 
        troubleshooting_scenarios = [
            {
                'query': "My LC-MS shows poor chromatography with broad peaks and low intensity. What preprocessing steps should I check?",
                'expected_routing_types': [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER],
                'description': "Analytical chemistry troubleshooting"
            },
            {
                'query': "Data shows batch effects between different analysis days. How do I correct this?",
                'expected_routing_types': [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER], 
                'description': "Batch effect correction"
            },
            {
                'query': "Quality control samples show CV > 30%. What could be causing this high variability?",
                'expected_routing_types': [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER],
                'description': "Quality control troubleshooting"
            },
            {
                'query': "Unknown peaks appearing in blank samples. How do I identify contamination sources?",
                'expected_routing_types': [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER],
                'description': "Contamination identification"
            }
        ]
        
        troubleshooting_results = []
        
        for scenario in troubleshooting_scenarios:
            prediction = biomedical_router.route_query(scenario['query'])
            
            correct_routing = prediction.routing_decision in scenario['expected_routing_types']
            
            troubleshooting_results.append({
                'description': scenario['description'],
                'query': scenario['query'],
                'predicted_routing': prediction.routing_decision.value,
                'expected_routing_types': [rt.value for rt in scenario['expected_routing_types']],
                'confidence': prediction.confidence,
                'correct_routing': correct_routing
            })
            
            # Should have reasonable confidence for technical queries
            assert prediction.confidence > 0.5, \
                f"Low confidence for technical troubleshooting: {scenario['description']}"
        
        routing_accuracy = sum(r['correct_routing'] for r in troubleshooting_results) / len(troubleshooting_results)
        
        assert routing_accuracy >= 0.8, \
            f"Troubleshooting routing accuracy {routing_accuracy:.3f} below 80%"
        
        print(f"\n=== Laboratory Troubleshooting ===")
        print(f"Routing Accuracy: {routing_accuracy:.3f} ({routing_accuracy*100:.1f}%)")
        
        for result in troubleshooting_results:
            status = "✓" if result['correct_routing'] else "✗"
            print(f"  {status} {result['description']}: {result['predicted_routing']} "
                  f"({result['confidence']:.3f})")
        
        return routing_accuracy, troubleshooting_results
    
    def test_research_planning_scenarios(self, biomedical_router):
        """Test research planning and consultation scenarios."""
        research_scenarios = [
            {
                'query': "I'm planning a metabolomics study for cardiovascular disease biomarkers. What sample size and methodology do you recommend?",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'context_type': 'study_planning'
            },
            {
                'query': "What are the current limitations and challenges in clinical metabolomics that I should address in my grant proposal?",
                'expected_category': ResearchCategory.LITERATURE_SEARCH,
                'context_type': 'grant_writing'
            },
            {
                'query': "How do I integrate metabolomics with genomics and proteomics data for systems biology analysis?",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'context_type': 'multi_omics'
            },
            {
                'query': "What regulatory requirements do I need to consider for clinical metabolomics biomarker validation?",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'context_type': 'regulatory'
            }
        ]
        
        planning_results = []
        
        for scenario in research_scenarios:
            context = {'context_type': scenario['context_type']}
            prediction = biomedical_router.route_query(scenario['query'], context)
            
            correct_category = prediction.research_category == scenario['expected_category']
            
            planning_results.append({
                'context_type': scenario['context_type'],
                'query': scenario['query'],
                'expected_category': scenario['expected_category'].value,
                'predicted_category': prediction.research_category.value,
                'routing_decision': prediction.routing_decision.value,
                'confidence': prediction.confidence,
                'correct_category': correct_category
            })
            
            # Research planning queries should have reasonable confidence
            assert prediction.confidence > 0.4, \
                f"Low confidence for research planning: {scenario['context_type']}"
        
        planning_accuracy = sum(r['correct_category'] for r in planning_results) / len(planning_results)
        
        assert planning_accuracy >= 0.75, \
            f"Research planning accuracy {planning_accuracy:.3f} below 75%"
        
        print(f"\n=== Research Planning Scenarios ===")
        print(f"Category Accuracy: {planning_accuracy:.3f} ({planning_accuracy*100:.1f}%)")
        
        for result in planning_results:
            status = "✓" if result['correct_category'] else "✗"
            print(f"  {status} {result['context_type']}: {result['predicted_category']} "
                  f"({result['confidence']:.3f})")
        
        return planning_accuracy, planning_results
    
    def test_clinical_decision_support_scenarios(self, biomedical_router):
        """Test clinical decision support scenarios."""
        clinical_scenarios = [
            {
                'query': "Patient shows elevated branched-chain amino acids and decreased glucose. What metabolic disorders should I consider?",
                'routing_preference': RoutingDecision.LIGHTRAG,  # Knowledge graph for relationships
                'urgency': 'high'
            },
            {
                'query': "Metabolomic analysis shows oxidative stress markers. What treatment options have recent clinical evidence?",
                'routing_preference': RoutingDecision.PERPLEXITY,  # Recent evidence
                'urgency': 'medium'
            },
            {
                'query': "How do metabolomic profiles change in response to diabetes medication? I need pathway-level insights.",
                'routing_preference': RoutingDecision.LIGHTRAG,  # Pathway analysis
                'urgency': 'low'
            },
            {
                'query': "Patient metabolomics indicates drug metabolism issues. What are the latest pharmacogenomics findings?",
                'routing_preference': RoutingDecision.PERPLEXITY,  # Latest findings
                'urgency': 'high'
            }
        ]
        
        clinical_results = []
        
        for scenario in clinical_scenarios:
            context = {'urgency': scenario['urgency'], 'clinical_context': True}
            prediction = biomedical_router.route_query(scenario['query'], context)
            
            # Check if routing aligns with preference (allowing flexibility)
            acceptable_routing = prediction.routing_decision in [
                scenario['routing_preference'],
                RoutingDecision.EITHER,
                RoutingDecision.HYBRID
            ]
            
            clinical_results.append({
                'urgency': scenario['urgency'],
                'query': scenario['query'],
                'preferred_routing': scenario['routing_preference'].value,
                'actual_routing': prediction.routing_decision.value,
                'confidence': prediction.confidence,
                'acceptable_routing': acceptable_routing,
                'response_time_ms': prediction.confidence_metrics.calculation_time_ms
            })
            
            # Clinical queries should have reasonable confidence and fast response
            assert prediction.confidence > 0.5, \
                f"Low confidence for clinical scenario: {scenario['urgency']} urgency"
            
            # High urgency queries should be processed quickly
            if scenario['urgency'] == 'high':
                assert prediction.confidence_metrics.calculation_time_ms < 100, \
                    f"High urgency query too slow: {prediction.confidence_metrics.calculation_time_ms:.2f}ms"
        
        routing_appropriateness = sum(r['acceptable_routing'] for r in clinical_results) / len(clinical_results)
        avg_response_time = statistics.mean([r['response_time_ms'] for r in clinical_results])
        
        assert routing_appropriateness >= 0.8, \
            f"Clinical routing appropriateness {routing_appropriateness:.3f} below 80%"
        
        print(f"\n=== Clinical Decision Support ===")
        print(f"Routing Appropriateness: {routing_appropriateness:.3f} ({routing_appropriateness*100:.1f}%)")
        print(f"Average Response Time: {avg_response_time:.2f}ms")
        
        for result in clinical_results:
            status = "✓" if result['acceptable_routing'] else "✗"
            print(f"  {status} {result['urgency'].upper()}: {result['actual_routing']} "
                  f"({result['confidence']:.3f}, {result['response_time_ms']:.1f}ms)")
        
        return routing_appropriateness, clinical_results


# =====================================================================
# PRODUCTION READINESS TESTS
# =====================================================================

@pytest.mark.performance
@pytest.mark.integration
class TestProductionReadiness:
    """Test production readiness including stress testing."""
    
    def test_stress_testing_high_load(self, biomedical_router, comprehensive_dataset, concurrent_tester):
        """Test system under high concurrent load."""
        # Generate large query set
        base_queries = comprehensive_dataset.performance_cases
        stress_queries = base_queries * 50  # 50x multiplier for stress test
        
        concurrent_levels = [5, 10, 20, 30]  # Progressive load increase
        stress_results = []
        
        for concurrent_requests in concurrent_levels:
            print(f"\nTesting with {concurrent_requests} concurrent requests...")
            
            result = concurrent_tester.run_concurrent_queries(
                biomedical_router,
                stress_queries[:concurrent_requests * 10],  # Scale queries with concurrency
                concurrent_requests=concurrent_requests
            )
            
            stress_results.append({
                'concurrent_requests': concurrent_requests,
                'total_queries': result['total_queries'],
                'success_rate': result['successful_queries'] / result['total_queries'],
                'avg_response_time_ms': result['avg_response_time_ms'],
                'throughput_qps': result['throughput_qps'],
                'error_rate': result['error_rate'],
                'max_response_time_ms': result['max_response_time_ms']
            })
            
            # Each stress level should maintain reasonable performance
            success_rate = result['successful_queries'] / result['total_queries']
            assert success_rate >= 0.95, \
                f"Success rate {success_rate:.3f} too low under {concurrent_requests} concurrent requests"
            
            assert result['error_rate'] <= 0.05, \
                f"Error rate {result['error_rate']:.3f} too high under {concurrent_requests} concurrent requests"
        
        # Check that system degrades gracefully
        response_times = [r['avg_response_time_ms'] for r in stress_results]
        throughputs = [r['throughput_qps'] for r in stress_results]
        
        print(f"\n=== Stress Testing Results ===")
        for result in stress_results:
            print(f"  {result['concurrent_requests']} concurrent: "
                  f"Success={result['success_rate']*100:.1f}%, "
                  f"AvgTime={result['avg_response_time_ms']:.1f}ms, "
                  f"Throughput={result['throughput_qps']:.1f}QPS")
        
        return stress_results
    
    def test_memory_leak_detection(self, biomedical_router, comprehensive_dataset, performance_monitor):
        """Test for memory leaks during extended operation."""
        test_queries = comprehensive_dataset.performance_cases * 20  # Extended operation
        memory_samples = []
        
        gc.collect()  # Start with clean memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process queries in batches and monitor memory
        batch_size = 50
        for i in range(0, len(test_queries), batch_size):
            batch = test_queries[i:i + batch_size]
            
            # Process batch
            for query in batch:
                prediction = biomedical_router.route_query(query)
            
            # Sample memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append({
                'queries_processed': i + len(batch),
                'memory_mb': current_memory,
                'memory_delta_mb': current_memory - initial_memory
            })
            
            # Force garbage collection periodically
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        # Analyze memory growth
        final_memory = memory_samples[-1]['memory_mb']
        memory_growth = final_memory - initial_memory
        queries_processed = memory_samples[-1]['queries_processed']
        
        # Check for reasonable memory usage (should not grow linearly with queries)
        memory_per_query = memory_growth / queries_processed if queries_processed > 0 else 0
        
        assert memory_growth <= 100.0, \
            f"Memory growth {memory_growth:.2f}MB too high"
        
        assert memory_per_query <= 0.1, \
            f"Memory per query {memory_per_query:.4f}MB suggests memory leak"
        
        print(f"\n=== Memory Leak Detection ===")
        print(f"Queries Processed: {queries_processed}")
        print(f"Initial Memory: {initial_memory:.2f}MB")
        print(f"Final Memory: {final_memory:.2f}MB")
        print(f"Memory Growth: {memory_growth:.2f}MB")
        print(f"Memory per Query: {memory_per_query:.4f}MB")
        
        return memory_growth, memory_per_query
    
    def test_circuit_breaker_functionality(self, biomedical_router):
        """Test circuit breaker functionality for failure scenarios."""
        
        # Mock failure scenarios by patching internal methods
        with patch.object(biomedical_router, '_comprehensive_query_analysis') as mock_analysis:
            # First few calls succeed
            mock_analysis.return_value = {
                'category_prediction': CategoryPrediction(
                    category=ResearchCategory.GENERAL_QUERY,
                    confidence=0.5,
                    evidence=[]
                ),
                'temporal_analysis': {'temporal_score': 0.0, 'established_score': 0.0},
                'real_time_detection': {'confidence': 0.0},
                'kg_detection': {'confidence': 0.0},
                'signal_strength': {'signal_quality_score': 0.0},
                'context_coherence': {'overall_coherence': 0.0},
                'ambiguity_analysis': {'ambiguity_score': 0.5, 'conflict_score': 0.0}
            }
            
            # Test normal operation
            query = "test query"
            prediction = biomedical_router.route_query(query)
            assert isinstance(prediction, RoutingPrediction)
            
            # Now make it fail repeatedly
            mock_analysis.side_effect = Exception("Simulated failure")
            
            failure_count = 0
            circuit_breaker_triggered = False
            
            # Trigger failures
            for i in range(5):  # Try to trigger circuit breaker
                try:
                    prediction = biomedical_router.route_query(f"failing query {i}")
                    
                    # If we get here, check if it's a circuit breaker response
                    if hasattr(prediction, 'metadata') and prediction.metadata.get('circuit_breaker_active'):
                        circuit_breaker_triggered = True
                        break
                        
                except Exception:
                    failure_count += 1
            
            # Circuit breaker should either trigger or handle failures gracefully
            assert circuit_breaker_triggered or failure_count < 5, \
                "Circuit breaker should activate or handle failures gracefully"
        
        print(f"\n=== Circuit Breaker Test ===")
        print(f"Failures before circuit breaker: {failure_count}")
        print(f"Circuit breaker triggered: {circuit_breaker_triggered}")
        
        return circuit_breaker_triggered
    
    def test_query_caching_effectiveness(self, biomedical_router, comprehensive_dataset):
        """Test query caching for performance improvement."""
        # Use repeated queries to test caching
        repeated_query = "LC-MS metabolite identification"
        cache_test_iterations = 100
        
        # First run (cache miss)
        start_time = time.perf_counter()
        first_prediction = biomedical_router.route_query(repeated_query)
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Subsequent runs (cache hits)
        cached_times = []
        for _ in range(cache_test_iterations):
            start_time = time.perf_counter()
            prediction = biomedical_router.route_query(repeated_query)
            cached_time = (time.perf_counter() - start_time) * 1000
            cached_times.append(cached_time)
        
        avg_cached_time = statistics.mean(cached_times)
        
        # Cached queries should be faster (at least 20% improvement)
        cache_improvement = (first_time - avg_cached_time) / first_time
        
        # Note: Caching might not be implemented, so this is informational
        print(f"\n=== Query Caching Test ===")
        print(f"First Query Time: {first_time:.2f}ms")
        print(f"Average Cached Time: {avg_cached_time:.2f}ms")
        print(f"Cache Improvement: {cache_improvement*100:.1f}%")
        
        if cache_improvement > 0.2:
            print("✓ Significant caching benefit detected")
        elif cache_improvement > 0.05:
            print("~ Moderate caching benefit detected") 
        else:
            print("- No significant caching benefit (caching may not be implemented)")
        
        return cache_improvement
    
    def test_system_recovery_after_failures(self, biomedical_router):
        """Test system recovery after failure scenarios."""
        recovery_scenarios = [
            "Normal query after simulated failure",
            "Complex query after recovery",
            "Temporal query after recovery", 
            "Edge case query after recovery"
        ]
        
        recovery_results = []
        
        for scenario in recovery_scenarios:
            try:
                prediction = biomedical_router.route_query(scenario)
                
                # Should return valid prediction
                assert isinstance(prediction, RoutingPrediction)
                assert prediction.confidence >= 0.0
                
                recovery_results.append({
                    'scenario': scenario,
                    'success': True,
                    'confidence': prediction.confidence,
                    'routing': prediction.routing_decision.value
                })
                
            except Exception as e:
                recovery_results.append({
                    'scenario': scenario,
                    'success': False,
                    'error': str(e)
                })
        
        success_rate = sum(r['success'] for r in recovery_results) / len(recovery_results)
        
        assert success_rate >= 1.0, \
            f"System recovery success rate {success_rate:.3f} below 100%"
        
        print(f"\n=== System Recovery Test ===")
        print(f"Recovery Success Rate: {success_rate:.3f} ({success_rate*100:.1f}%)")
        
        for result in recovery_results:
            status = "✓" if result['success'] else "✗"
            if result['success']:
                print(f"  {status} {result['scenario']}: {result['routing']} ({result['confidence']:.3f})")
            else:
                print(f"  {status} {result['scenario']}: {result.get('error', 'Unknown error')}")
        
        return success_rate


# =====================================================================
# COMPREHENSIVE TEST REPORT GENERATION
# =====================================================================

def generate_comprehensive_test_report(test_results: Dict[str, Any]) -> str:
    """Generate comprehensive test report."""
    
    report = f"""
=============================================================================
COMPREHENSIVE QUERY CLASSIFICATION TEST REPORT
=============================================================================

Test Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
System Under Test: Clinical Metabolomics Oracle - BiomedicalQueryRouter
Test Coverage: Classification Accuracy, Performance, Confidence Scoring,
              Integration, Real-World Scenarios, Production Readiness

=============================================================================
EXECUTIVE SUMMARY
=============================================================================

Overall Test Status: {'✓ PASS' if test_results.get('overall_pass', False) else '✗ FAIL'}

Key Metrics:
- Classification Accuracy: {test_results.get('accuracy', 0)*100:.1f}%
- Average Response Time: {test_results.get('avg_response_time_ms', 0):.1f}ms
- Throughput: {test_results.get('throughput_qps', 0):.1f} QPS
- Success Rate: {test_results.get('success_rate', 0)*100:.1f}%

Requirements Validation:
- >90% Classification Accuracy: {'✓ PASS' if test_results.get('accuracy', 0) >= 0.9 else '✗ FAIL'}
- <2 Second Response Time: {'✓ PASS' if test_results.get('avg_response_time_ms', 0) < 2000 else '✗ FAIL'}
- Confidence Scoring: {'✓ PASS' if test_results.get('confidence_system', False) else '✗ FAIL'}
- Fallback Mechanisms: {'✓ PASS' if test_results.get('fallback_system', False) else '✗ FAIL'}

=============================================================================
DETAILED TEST RESULTS
=============================================================================

1. CLASSIFICATION ACCURACY TESTS
   - Overall Accuracy: {test_results.get('accuracy', 0)*100:.1f}%
   - Category-Specific Accuracy: {test_results.get('category_accuracy', 'N/A')}
   - Temporal Query Accuracy: {test_results.get('temporal_accuracy', 'N/A')}
   - Edge Case Robustness: {test_results.get('edge_case_robustness', 'N/A')}

2. PERFORMANCE TESTS  
   - Single Query Response Time: {test_results.get('avg_response_time_ms', 0):.1f}ms
   - Batch Processing Throughput: {test_results.get('throughput_qps', 0):.1f} QPS
   - Concurrent Performance: {test_results.get('concurrent_performance', 'N/A')}
   - Memory Usage: {test_results.get('memory_usage_mb', 0):.1f}MB

3. CONFIDENCE SCORING TESTS
   - Confidence System Functional: {'✓' if test_results.get('confidence_system', False) else '✗'}
   - Confidence Correlation: {test_results.get('confidence_correlation', 'N/A')}
   - Fallback Triggers: {test_results.get('fallback_triggers', 'N/A')}

4. INTEGRATION TESTS
   - ResearchCategorizer Compatibility: {'✓' if test_results.get('integration_compatible', False) else '✗'}
   - Category-Routing Consistency: {test_results.get('category_routing_consistency', 'N/A')}
   - Statistics Integration: {'✓' if test_results.get('stats_integration', False) else '✗'}

5. REAL-WORLD SCENARIO TESTS
   - Clinical Workflow Accuracy: {test_results.get('clinical_workflow_accuracy', 'N/A')}
   - Laboratory Troubleshooting: {test_results.get('lab_troubleshooting_accuracy', 'N/A')}
   - Research Planning Support: {test_results.get('research_planning_accuracy', 'N/A')}

6. PRODUCTION READINESS TESTS
   - Stress Testing: {'✓ PASS' if test_results.get('stress_test_pass', False) else '✗ FAIL'}
   - Memory Leak Detection: {'✓ PASS' if test_results.get('memory_leak_check', False) else '✗ FAIL'}
   - Circuit Breaker: {'✓ PASS' if test_results.get('circuit_breaker_functional', False) else '✗ FAIL'}
   - System Recovery: {'✓ PASS' if test_results.get('system_recovery', False) else '✗ FAIL'}

=============================================================================
RECOMMENDATIONS
=============================================================================

{test_results.get('recommendations', 'No specific recommendations.')}

=============================================================================
CONCLUSION
=============================================================================

The Clinical Metabolomics Oracle query classification system has been 
comprehensively tested across all critical dimensions. {'The system meets all production requirements and is ready for deployment.' if test_results.get('overall_pass', False) else 'The system requires additional work before production deployment.'}

Test completed at {time.strftime('%Y-%m-%d %H:%M:%S')}
=============================================================================
"""
    
    return report


if __name__ == "__main__":
    """Run comprehensive tests if executed directly."""
    print("=== Comprehensive Query Classification Test Suite ===")
    print("Use pytest to run these tests:")
    print("  pytest test_comprehensive_query_classification.py -v")
    print("  pytest test_comprehensive_query_classification.py::TestClassificationAccuracy -v")
    print("  pytest test_comprehensive_query_classification.py::TestPerformanceRequirements -v") 
    print("  pytest -m biomedical test_comprehensive_query_classification.py -v")
    print("  pytest -m performance test_comprehensive_query_classification.py -v")
    print("  pytest -m integration test_comprehensive_query_classification.py -v")
    print("\nTest markers available: biomedical, performance, integration")