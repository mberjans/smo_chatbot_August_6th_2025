#!/usr/bin/env python3
"""
Comprehensive validation test for CMO-LIGHTRAG-012-T01 query classification.

This script validates the query classification tests work correctly with the actual
ResearchCategorizer and provides comprehensive coverage analysis.
"""

import sys
import os
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def setup_real_categorizer():
    """Setup the real ResearchCategorizer with proper imports."""
    try:
        # Import cost_persistence first
        from cost_persistence import ResearchCategory
        
        # Temporarily fix the relative import issue
        temp_file = Path('research_categorizer_temp.py')
        with open(temp_file, 'w') as f:
            with open('../research_categorizer.py', 'r') as original:
                content = original.read()
                fixed_content = content.replace(
                    'from .cost_persistence import ResearchCategory',
                    'from cost_persistence import ResearchCategory'
                )
                f.write(fixed_content)
        
        # Import the fixed version
        import research_categorizer_temp as research_categorizer
        
        # Get classes
        ResearchCategorizer = research_categorizer.ResearchCategorizer
        CategoryPrediction = research_categorizer.CategoryPrediction
        
        # Cleanup
        temp_file.unlink()
        
        return ResearchCategorizer, CategoryPrediction, ResearchCategory
        
    except Exception as e:
        print(f"✗ Error setting up real categorizer: {e}")
        # Cleanup on error
        temp_file = Path('research_categorizer_temp.py')
        if temp_file.exists():
            temp_file.unlink()
        raise

def run_comprehensive_validation():
    """Run comprehensive validation of query classification system."""
    
    print("=== CMO-LIGHTRAG-012-T01 QUERY CLASSIFICATION VALIDATION ===\n")
    
    # Setup real categorizer
    try:
        ResearchCategorizer, CategoryPrediction, ResearchCategory = setup_real_categorizer()
        categorizer = ResearchCategorizer()
        print("✓ Real ResearchCategorizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load real ResearchCategorizer: {e}")
        return False
    
    # Load test queries from the test file
    try:
        from test_query_classification_biomedical_samples import BiomedicalQuerySamples
        samples = BiomedicalQuerySamples()
        all_queries = samples.get_all_test_queries()
        print(f"✓ Test queries loaded: {len(all_queries)} categories")
    except Exception as e:
        print(f"✗ Failed to load test queries: {e}")
        return False
    
    # Performance validation
    print("\n=== PERFORMANCE VALIDATION ===")
    
    # Test individual query response times
    test_queries = [
        "What is the molecular structure of this unknown metabolite with exact mass 180.0634 detected in LC-MS analysis?",
        "Perform KEGG pathway enrichment analysis on my list of significantly altered metabolites in Type 2 diabetes patients",
        "Identification of diagnostic biomarkers for early-stage pancreatic cancer using untargeted metabolomics profiling",
        "Multivariate statistical analysis of metabolomics data using PCA, PLS-DA and OPLS-DA for group discrimination"
    ]
    
    response_times = []
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        prediction = categorizer.categorize_query(query)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        response_times.append(response_time_ms)
        print(f"  Query {i}: {response_time_ms:.1f}ms -> {prediction.category.value} (conf: {prediction.confidence:.2f})")
    
    avg_response_time = statistics.mean(response_times)
    max_response_time = max(response_times)
    
    print(f"\nResponse Time Analysis:")
    print(f"  Average: {avg_response_time:.1f}ms")
    print(f"  Maximum: {max_response_time:.1f}ms")
    print(f"  Performance: {'✓ PASS' if avg_response_time <= 1000 else '✗ FAIL'} (< 1000ms requirement)")
    
    # Batch processing test
    print("\n=== BATCH PROCESSING VALIDATION ===")
    
    batch_queries = []
    for category, queries in all_queries.items():
        if category in ['edge_cases', 'performance']:
            continue
        # Take first 3 queries from each category for balanced testing
        for query_data in queries[:3]:
            if isinstance(query_data, dict) and 'query' in query_data:
                batch_queries.append(query_data)
    
    batch_queries = batch_queries[:30]  # Limit to 30 for reasonable test time
    
    start_time = time.time()
    batch_results = []
    for query_data in batch_queries:
        prediction = categorizer.categorize_query(query_data['query'])
        batch_results.append(prediction)
    end_time = time.time()
    
    batch_time = end_time - start_time
    throughput = len(batch_queries) / batch_time
    
    print(f"Processed {len(batch_queries)} queries in {batch_time:.2f}s")
    print(f"Throughput: {throughput:.1f} queries/sec")
    print(f"Batch Performance: {'✓ PASS' if throughput >= 5 else '✗ FAIL'} (>= 5 queries/sec requirement)")
    
    # Accuracy validation with sample queries
    print("\n=== ACCURACY VALIDATION ===")
    
    # Test key biomedical query categories
    validation_queries = [
        {
            'query': "LC-MS metabolite identification using exact mass 180.0634 and MS/MS fragmentation patterns",
            'expected': ResearchCategory.METABOLITE_IDENTIFICATION
        },
        {
            'query': "KEGG pathway enrichment analysis for significantly altered metabolites in diabetes patients",
            'expected': ResearchCategory.PATHWAY_ANALYSIS
        },
        {
            'query': "Discovery and validation of blood-based biomarkers for monitoring drug response in cancer therapy",
            'expected': ResearchCategory.BIOMARKER_DISCOVERY
        },
        {
            'query': "Clinical metabolomics approach for differential diagnosis of inflammatory bowel diseases using serum",
            'expected': ResearchCategory.CLINICAL_DIAGNOSIS
        },
        {
            'query': "Pharmacokinetic analysis of novel antidiabetic compound using LC-MS/MS metabolite profiling",
            'expected': ResearchCategory.DRUG_DISCOVERY
        },
        {
            'query': "Multivariate statistical analysis using PCA, PLS-DA and machine learning for biomarker discovery",
            'expected': ResearchCategory.STATISTICAL_ANALYSIS
        },
        {
            'query': "Metabolomics data preprocessing including peak detection, retention time alignment, and batch correction",
            'expected': ResearchCategory.DATA_PREPROCESSING
        },
        {
            'query': "Integration of HMDB, KEGG, and ChEBI databases for comprehensive metabolite annotation",
            'expected': ResearchCategory.DATABASE_INTEGRATION
        }
    ]
    
    correct_predictions = 0
    total_predictions = len(validation_queries)
    confidence_scores = []
    
    for i, test_case in enumerate(validation_queries, 1):
        prediction = categorizer.categorize_query(test_case['query'])
        is_correct = prediction.category == test_case['expected']
        confidence_scores.append(prediction.confidence)
        
        if is_correct:
            correct_predictions += 1
        
        status = '✓' if is_correct else '✗'
        print(f"  {status} Query {i}: {prediction.category.value} (expected: {test_case['expected'].value}) - conf: {prediction.confidence:.2f}")
        if not is_correct:
            print(f"    Query: {test_case['query'][:60]}...")
    
    accuracy = correct_predictions / total_predictions
    avg_confidence = statistics.mean(confidence_scores)
    
    print(f"\nAccuracy Analysis:")
    print(f"  Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Accuracy Result: {'✓ PASS' if accuracy >= 0.75 else '✗ FAIL'} (>= 75% requirement)")
    
    # Coverage validation
    print("\n=== COVERAGE VALIDATION ===")
    
    category_coverage = {}
    for category, queries in all_queries.items():
        if category in ['edge_cases', 'performance']:
            continue
        category_coverage[category] = len(queries)
    
    total_test_queries = sum(category_coverage.values())
    print(f"Test Query Coverage:")
    for category, count in category_coverage.items():
        print(f"  {category}: {count} queries")
    print(f"Total test queries: {total_test_queries}")
    
    # Edge case validation
    print("\n=== EDGE CASE VALIDATION ===")
    
    edge_cases = [
        "",  # Empty query
        "metabolomics",  # Single word
        "What is the meaning of life?",  # Non-biomedical
        "a" * 1000,  # Very long query
        "LC-MS/MS análisis metabólicos",  # Special characters
    ]
    
    edge_case_results = []
    for edge_query in edge_cases:
        try:
            prediction = categorizer.categorize_query(edge_query)
            edge_case_results.append(True)
            print(f"  ✓ Handled: '{edge_query[:30]}{'...' if len(edge_query) > 30 else ''}' -> {prediction.category.value}")
        except Exception as e:
            edge_case_results.append(False)
            print(f"  ✗ Failed: '{edge_query[:30]}{'...' if len(edge_query) > 30 else ''}' -> {e}")
    
    edge_case_success_rate = sum(edge_case_results) / len(edge_case_results)
    print(f"Edge case handling: {edge_case_success_rate:.1%} success rate")
    
    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    
    performance_ok = avg_response_time <= 1000 and throughput >= 5
    accuracy_ok = accuracy >= 0.75
    coverage_ok = total_test_queries >= 50  # Minimum coverage requirement
    edge_cases_ok = edge_case_success_rate >= 0.8
    
    all_requirements_met = performance_ok and accuracy_ok and coverage_ok and edge_cases_ok
    
    print(f"Performance Requirements: {'✓ PASS' if performance_ok else '✗ FAIL'}")
    print(f"Accuracy Requirements: {'✓ PASS' if accuracy_ok else '✗ FAIL'}")  
    print(f"Coverage Requirements: {'✓ PASS' if coverage_ok else '✗ FAIL'}")
    print(f"Edge Case Handling: {'✓ PASS' if edge_cases_ok else '✗ FAIL'}")
    
    print(f"\n{'='*60}")
    print(f"OVERALL VALIDATION RESULT: {'✓ ALL TESTS PASS' if all_requirements_met else '✗ SOME TESTS FAILED'}")
    print(f"{'='*60}")
    
    # Generate detailed report
    validation_report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'performance': {
            'avg_response_time_ms': avg_response_time,
            'max_response_time_ms': max_response_time,
            'throughput_queries_per_sec': throughput,
            'performance_requirements_met': performance_ok
        },
        'accuracy': {
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'accuracy_percentage': accuracy * 100,
            'average_confidence': avg_confidence,
            'accuracy_requirements_met': accuracy_ok
        },
        'coverage': {
            'categories_tested': len(category_coverage),
            'total_test_queries': total_test_queries,
            'category_distribution': category_coverage,
            'coverage_requirements_met': coverage_ok
        },
        'edge_cases': {
            'success_rate': edge_case_success_rate,
            'edge_case_requirements_met': edge_cases_ok
        },
        'overall_validation_passed': all_requirements_met
    }
    
    # Save validation report
    import json
    with open('query_classification_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\nDetailed validation report saved to: query_classification_validation_report.json")
    
    return all_requirements_met

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)