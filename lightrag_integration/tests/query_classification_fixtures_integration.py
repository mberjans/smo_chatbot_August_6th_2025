#!/usr/bin/env python3
"""
Query Classification Test Fixtures Integration Module

This module integrates all query classification test fixtures and provides
a unified interface for accessing comprehensive biomedical query samples,
mock components, performance testing utilities, and validation tools.

This serves as the central hub for all query classification testing needs,
ensuring proper integration between the comprehensive biomedical query
samples and the test infrastructure.

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: CMO-LIGHTRAG-012-T01 Support - Final Integration
"""

import pytest
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import logging

# Import all available fixtures and utilities
try:
    # Import comprehensive biomedical queries
    from .test_fixtures_biomedical_queries import (
        QueryTestCase,
        ResearchCategory,
        ComplexityLevel,
        get_all_test_queries,
        get_queries_by_complexity,
        get_edge_case_queries,
        get_query_statistics,
        METABOLITE_IDENTIFICATION_QUERIES,
        PATHWAY_ANALYSIS_QUERIES,
        BIOMARKER_DISCOVERY_QUERIES,
        DRUG_DISCOVERY_QUERIES,
        CLINICAL_DIAGNOSIS_QUERIES,
        DATA_PREPROCESSING_QUERIES,
        STATISTICAL_ANALYSIS_QUERIES,
        LITERATURE_SEARCH_QUERIES,
        KNOWLEDGE_EXTRACTION_QUERIES,
        DATABASE_INTEGRATION_QUERIES,
        EDGE_CASE_QUERIES
    )
    COMPREHENSIVE_QUERIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import comprehensive biomedical queries: {e}")
    COMPREHENSIVE_QUERIES_AVAILABLE = False

try:
    # Import test fixtures and utilities
    from .test_fixtures_query_classification import (
        MockResearchCategorizer,
        MockQueryAnalyzer,
        CategoryPrediction,
        CategoryMetrics,
        QueryClassificationPerformanceTester,
        BiomedicalQueryFixtures
    )
    TEST_FIXTURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import query classification fixtures: {e}")
    TEST_FIXTURES_AVAILABLE = False


# =====================================================================
# INTEGRATED QUERY CLASSIFICATION TEST SUITE
# =====================================================================

class IntegratedQueryClassificationTestSuite:
    """
    Comprehensive test suite that integrates all query classification
    testing components into a unified interface.
    """
    
    def __init__(self):
        self.comprehensive_queries_available = COMPREHENSIVE_QUERIES_AVAILABLE
        self.test_fixtures_available = TEST_FIXTURES_AVAILABLE
        
        # Initialize components if available
        if TEST_FIXTURES_AVAILABLE:
            self.categorizer = MockResearchCategorizer()
            self.performance_tester = QueryClassificationPerformanceTester()
            self.local_fixtures = BiomedicalQueryFixtures()
        else:
            self.categorizer = None
            self.performance_tester = None
            self.local_fixtures = None
    
    def get_test_queries(self, 
                        source: str = 'comprehensive',
                        category: str = None,
                        complexity: str = None,
                        count: int = None) -> List[Union[QueryTestCase, Dict[str, Any]]]:
        """
        Get test queries from available sources.
        
        Args:
            source: 'comprehensive' or 'local' - which query source to use
            category: Optional category filter
            complexity: Optional complexity filter  
            count: Optional limit on number of queries
        
        Returns:
            List of query test cases or dictionaries
        """
        queries = []
        
        if source == 'comprehensive' and COMPREHENSIVE_QUERIES_AVAILABLE:
            # Get comprehensive queries
            if category:
                all_queries = get_all_test_queries()
                if category in all_queries:
                    queries = all_queries[category]
                else:
                    logging.warning(f"Category '{category}' not found in comprehensive queries")
            elif complexity:
                try:
                    complexity_enum = ComplexityLevel(complexity.lower())
                    queries = get_queries_by_complexity(complexity_enum)
                except ValueError:
                    logging.warning(f"Invalid complexity level: {complexity}")
            else:
                # Get all queries
                all_queries = get_all_test_queries()
                for cat_queries in all_queries.values():
                    queries.extend(cat_queries)
        
        elif source == 'local' and TEST_FIXTURES_AVAILABLE:
            # Get local fixture queries
            if category:
                queries = self.local_fixtures.get_sample_queries_by_category(category)
            else:
                all_queries = self.local_fixtures.get_all_sample_queries()
                for cat_queries in all_queries.values():
                    queries.extend(cat_queries)
        
        else:
            logging.error(f"Query source '{source}' not available")
            return []
        
        # Apply count limit if specified
        if count and count < len(queries):
            queries = queries[:count]
        
        return queries
    
    def get_edge_case_queries(self, source: str = 'comprehensive') -> List[Union[QueryTestCase, Dict[str, Any]]]:
        """Get edge case queries for robustness testing."""
        if source == 'comprehensive' and COMPREHENSIVE_QUERIES_AVAILABLE:
            return get_edge_case_queries()
        elif source == 'local' and TEST_FIXTURES_AVAILABLE:
            return self.local_fixtures.get_edge_cases()
        else:
            return []
    
    def get_performance_queries(self, source: str = 'local') -> List[str]:
        """Get queries optimized for performance testing."""
        if source == 'local' and TEST_FIXTURES_AVAILABLE:
            return self.local_fixtures.get_performance_queries()
        elif source == 'comprehensive' and COMPREHENSIVE_QUERIES_AVAILABLE:
            # Extract query strings from comprehensive dataset
            all_queries = get_all_test_queries()
            performance_queries = []
            for queries in all_queries.values():
                for query_data in queries[:2]:  # Take first 2 from each category
                    if hasattr(query_data, 'query'):
                        performance_queries.append(query_data.query)
                    elif isinstance(query_data, dict) and 'query' in query_data:
                        performance_queries.append(query_data['query'])
            return performance_queries[:20]  # Limit to 20 for performance testing
        else:
            return ["metabolomics analysis", "pathway analysis", "biomarker discovery"]
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about available query datasets."""
        stats = {
            'comprehensive_available': COMPREHENSIVE_QUERIES_AVAILABLE,
            'local_fixtures_available': TEST_FIXTURES_AVAILABLE,
            'total_queries': 0,
            'categories_available': 0,
            'complexity_levels': 0,
            'edge_cases': 0
        }
        
        if COMPREHENSIVE_QUERIES_AVAILABLE:
            comp_stats = get_query_statistics()
            stats.update({
                'comprehensive_stats': comp_stats,
                'total_queries': comp_stats.get('total_queries', 0),
                'categories_available': len(comp_stats.get('category_distribution', {})),
                'complexity_levels': len(comp_stats.get('complexity_distribution', {})),
                'edge_cases': comp_stats.get('edge_cases', 0)
            })
        
        if TEST_FIXTURES_AVAILABLE:
            local_queries = self.local_fixtures.get_all_sample_queries()
            edge_cases = self.local_fixtures.get_edge_cases()
            perf_queries = self.local_fixtures.get_performance_queries()
            
            stats['local_fixtures_stats'] = {
                'sample_queries': sum(len(queries) for queries in local_queries.values()),
                'categories': len(local_queries),
                'edge_cases': len(edge_cases),
                'performance_queries': len(perf_queries)
            }
        
        return stats
    
    def run_comprehensive_test_suite(self, 
                                   max_queries_per_category: int = 10,
                                   include_performance_tests: bool = True,
                                   include_edge_cases: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive test suite across all available query sources.
        
        Args:
            max_queries_per_category: Limit queries per category for faster testing
            include_performance_tests: Whether to run performance benchmarks
            include_edge_cases: Whether to test edge cases
        
        Returns:
            Comprehensive test results
        """
        if not TEST_FIXTURES_AVAILABLE:
            return {'error': 'Test fixtures not available'}
        
        results = {
            'test_suite_version': '1.0.0',
            'comprehensive_queries_used': COMPREHENSIVE_QUERIES_AVAILABLE,
            'categories_tested': [],
            'performance_metrics': {},
            'edge_case_results': {},
            'overall_success': True
        }
        
        try:
            # Test each category with available queries
            source = 'comprehensive' if COMPREHENSIVE_QUERIES_AVAILABLE else 'local'
            
            # Get categories to test
            if COMPREHENSIVE_QUERIES_AVAILABLE:
                all_queries = get_all_test_queries()
                categories = list(all_queries.keys())
            else:
                categories = list(self.local_fixtures.get_all_sample_queries().keys())
            
            # Test each category
            for category in categories:
                if category in ['edge_cases', 'performance']:
                    continue  # Handle these separately
                
                try:
                    queries = self.get_test_queries(
                        source=source, 
                        category=category, 
                        count=max_queries_per_category
                    )
                    
                    category_results = {
                        'total_queries': len(queries),
                        'successful_classifications': 0,
                        'average_confidence': 0.0,
                        'classification_errors': []
                    }
                    
                    confidence_scores = []
                    
                    for query_data in queries:
                        try:
                            # Extract query string
                            if hasattr(query_data, 'query'):
                                query_str = query_data.query
                            elif isinstance(query_data, dict) and 'query' in query_data:
                                query_str = query_data['query']
                            else:
                                continue
                            
                            # Classify query
                            prediction = self.categorizer.categorize_query(query_str)
                            
                            if prediction and hasattr(prediction, 'confidence'):
                                category_results['successful_classifications'] += 1
                                confidence_scores.append(prediction.confidence)
                            
                        except Exception as e:
                            category_results['classification_errors'].append(str(e))
                    
                    # Calculate metrics
                    if confidence_scores:
                        category_results['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
                    
                    results['categories_tested'].append({
                        'category': category,
                        'results': category_results
                    })
                
                except Exception as e:
                    logging.error(f"Error testing category {category}: {e}")
                    results['overall_success'] = False
            
            # Performance tests
            if include_performance_tests:
                try:
                    perf_queries = self.get_performance_queries(source='local')[:10]
                    if perf_queries:
                        perf_results = self.performance_tester.benchmark_query_batch(
                            self.categorizer, 
                            perf_queries
                        )
                        results['performance_metrics'] = perf_results
                except Exception as e:
                    logging.error(f"Error in performance tests: {e}")
                    results['performance_metrics'] = {'error': str(e)}
            
            # Edge case tests
            if include_edge_cases:
                try:
                    edge_queries = self.get_edge_case_queries(source=source)[:5]
                    edge_results = {
                        'total_edge_cases': len(edge_queries),
                        'successful_handling': 0,
                        'errors': []
                    }
                    
                    for query_data in edge_queries:
                        try:
                            if hasattr(query_data, 'query'):
                                query_str = query_data.query
                            elif isinstance(query_data, dict) and 'query' in query_data:
                                query_str = query_data['query']
                            else:
                                continue
                            
                            prediction = self.categorizer.categorize_query(query_str)
                            if prediction:
                                edge_results['successful_handling'] += 1
                        except Exception as e:
                            edge_results['errors'].append(str(e))
                    
                    results['edge_case_results'] = edge_results
                
                except Exception as e:
                    logging.error(f"Error in edge case tests: {e}")
                    results['edge_case_results'] = {'error': str(e)}
        
        except Exception as e:
            logging.error(f"Error in comprehensive test suite: {e}")
            results['overall_success'] = False
            results['error'] = str(e)
        
        return results
    
    def generate_integration_report(self) -> str:
        """Generate a comprehensive integration report."""
        stats = self.get_dataset_statistics()
        
        report = """
=== Query Classification Fixtures Integration Report ===

COMPONENT AVAILABILITY:
  Comprehensive Biomedical Queries: {comprehensive}
  Test Fixtures & Utilities: {fixtures}
  Mock Categorizer: {categorizer}
  Performance Testing: {performance}

DATASET STATISTICS:
  Total Queries Available: {total_queries}
  Categories Available: {categories}
  Edge Cases: {edge_cases}
        """.format(
            comprehensive="✅ Available" if COMPREHENSIVE_QUERIES_AVAILABLE else "❌ Not Available",
            fixtures="✅ Available" if TEST_FIXTURES_AVAILABLE else "❌ Not Available", 
            categorizer="✅ Available" if self.categorizer else "❌ Not Available",
            performance="✅ Available" if self.performance_tester else "❌ Not Available",
            total_queries=stats.get('total_queries', 'Unknown'),
            categories=stats.get('categories_available', 'Unknown'),
            edge_cases=stats.get('edge_cases', 'Unknown')
        )
        
        if COMPREHENSIVE_QUERIES_AVAILABLE:
            comp_stats = stats.get('comprehensive_stats', {})
            report += f"""
COMPREHENSIVE QUERIES BREAKDOWN:
  Category Distribution: {comp_stats.get('category_distribution', {})}
  Complexity Distribution: {comp_stats.get('complexity_distribution', {})}
            """
        
        if TEST_FIXTURES_AVAILABLE:
            local_stats = stats.get('local_fixtures_stats', {})
            report += f"""
LOCAL FIXTURES BREAKDOWN:
  Sample Queries: {local_stats.get('sample_queries', 0)}
  Performance Test Queries: {local_stats.get('performance_queries', 0)}
  Edge Cases: {local_stats.get('edge_cases', 0)}
            """
        
        # Integration status
        integration_status = "FULL INTEGRATION" if (COMPREHENSIVE_QUERIES_AVAILABLE and TEST_FIXTURES_AVAILABLE) else \
                           "PARTIAL INTEGRATION" if (COMPREHENSIVE_QUERIES_AVAILABLE or TEST_FIXTURES_AVAILABLE) else \
                           "NO INTEGRATION"
        
        report += f"""
INTEGRATION STATUS: {integration_status}

USAGE RECOMMENDATIONS:
  - Use 'comprehensive' source for extensive testing with realistic biomedical queries
  - Use 'local' source for quick validation and performance testing  
  - Edge cases available from both sources for robustness testing
  - Performance benchmarking utilities ready for load testing
  - Mock categorizer provides realistic classification behavior

FIXTURE ACCESS:
  - get_test_queries(source='comprehensive', category='metabolite_identification')
  - get_edge_case_queries(source='comprehensive') 
  - get_performance_queries(source='local')
  - run_comprehensive_test_suite(max_queries_per_category=5)

        """
        
        return report


# =====================================================================
# PYTEST INTEGRATION FIXTURES
# =====================================================================

@pytest.fixture
def integrated_query_test_suite():
    """Provide integrated query classification test suite."""
    return IntegratedQueryClassificationTestSuite()


@pytest.fixture  
def comprehensive_biomedical_queries():
    """Provide comprehensive biomedical queries if available."""
    if COMPREHENSIVE_QUERIES_AVAILABLE:
        return get_all_test_queries()
    else:
        pytest.skip("Comprehensive biomedical queries not available")


@pytest.fixture
def biomedical_queries_by_complexity():
    """Provide biomedical queries organized by complexity."""
    if COMPREHENSIVE_QUERIES_AVAILABLE:
        return {
            'basic': get_queries_by_complexity(ComplexityLevel.BASIC),
            'medium': get_queries_by_complexity(ComplexityLevel.MEDIUM), 
            'complex': get_queries_by_complexity(ComplexityLevel.COMPLEX),
            'expert': get_queries_by_complexity(ComplexityLevel.EXPERT)
        }
    else:
        pytest.skip("Comprehensive biomedical queries not available")


@pytest.fixture
def query_classification_integration_report():
    """Generate and provide integration report."""
    suite = IntegratedQueryClassificationTestSuite()
    return suite.generate_integration_report()


if __name__ == "__main__":
    """Demonstrate integration functionality."""
    print("=== Query Classification Fixtures Integration Test ===\n")
    
    # Create test suite
    suite = IntegratedQueryClassificationTestSuite()
    
    # Show integration report
    print(suite.generate_integration_report())
    
    # Test basic functionality
    if suite.comprehensive_queries_available or suite.test_fixtures_available:
        print("\n=== Testing Basic Functionality ===")
        
        # Get sample queries
        queries = suite.get_test_queries(count=3)
        print(f"Retrieved {len(queries)} test queries")
        
        # Get edge cases
        edge_cases = suite.get_edge_case_queries()
        print(f"Retrieved {len(edge_cases)} edge case queries")
        
        # Get performance queries
        perf_queries = suite.get_performance_queries()
        print(f"Retrieved {len(perf_queries)} performance queries")
        
        # Show dataset statistics
        stats = suite.get_dataset_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total queries: {stats.get('total_queries', 'N/A')}")
        print(f"  Categories: {stats.get('categories_available', 'N/A')}")
        print(f"  Edge cases: {stats.get('edge_cases', 'N/A')}")
        
        print("\n✅ Integration test completed successfully!")
    else:
        print("\n❌ No fixtures available for testing")