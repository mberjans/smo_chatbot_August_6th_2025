#!/usr/bin/env python3
"""
Detailed Accuracy Analysis Report for CMO-LIGHTRAG-012-T09
===========================================================

This script provides a comprehensive analysis of the classification accuracy
test results to understand why the >90% requirement is not being met.
"""

import sys
import time
import statistics
from pathlib import Path

# Add the current directory to the path to import test modules
sys.path.insert(0, str(Path(__file__).parent / "lightrag_integration"))

from lightrag_integration.query_router import BiomedicalQueryRouter
from lightrag_integration.tests.test_comprehensive_query_classification import ComprehensiveQueryDataset
from lightrag_integration.cost_persistence import ResearchCategory
from unittest.mock import Mock


def analyze_classification_accuracy():
    """Perform detailed accuracy analysis."""
    
    print("=" * 80)
    print("CMO-LIGHTRAG-012-T09 Classification Accuracy Analysis Report")
    print("=" * 80)
    print(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize components
    mock_logger = Mock()
    router = BiomedicalQueryRouter(logger=mock_logger)
    dataset = ComprehensiveQueryDataset()
    
    print(f"Total Test Cases: {len(dataset.test_cases)}")
    print(f"Domain-Specific Test Cases: {len([tc for tc in dataset.test_cases if tc.domain_specific])}")
    print()
    
    # Analyze accuracy by category
    print("CATEGORY-SPECIFIC ACCURACY ANALYSIS")
    print("-" * 50)
    
    category_results = {}
    overall_correct = 0
    overall_total = 0
    detailed_errors = []
    
    for category in ResearchCategory:
        category_cases = dataset.get_test_cases_by_category(category)
        domain_specific_cases = [tc for tc in category_cases if tc.domain_specific]
        
        if not domain_specific_cases:
            continue
            
        correct = 0
        total = len(domain_specific_cases)
        confidence_scores = []
        category_errors = []
        
        for test_case in domain_specific_cases:
            try:
                prediction = router.route_query(test_case.query)
                is_correct = prediction.research_category == category
                
                if is_correct:
                    correct += 1
                    overall_correct += 1
                else:
                    error_detail = {
                        'query': test_case.query[:100] + "..." if len(test_case.query) > 100 else test_case.query,
                        'expected': category.value,
                        'predicted': prediction.research_category.value,
                        'confidence': prediction.confidence,
                        'description': test_case.description
                    }
                    category_errors.append(error_detail)
                    detailed_errors.append(error_detail)
                
                confidence_scores.append(prediction.confidence)
                overall_total += 1
                
            except Exception as e:
                print(f"ERROR processing query: {test_case.query[:50]}... - {str(e)}")
                continue
        
        if total > 0:
            accuracy = correct / total
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            
            category_results[category.value] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'avg_confidence': avg_confidence,
                'errors': category_errors
            }
            
            status = "✓ PASS" if accuracy >= 0.85 else "✗ FAIL"
            print(f"{status} {category.value:25} | {accuracy:6.3f} ({accuracy*100:5.1f}%) | {correct:2d}/{total:2d} | Conf: {avg_confidence:.3f}")
    
    # Overall accuracy
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    accuracy_status = "✓ PASS" if overall_accuracy >= 0.90 else "✗ FAIL"
    
    print()
    print("OVERALL ACCURACY SUMMARY")
    print("-" * 50)
    print(f"Overall Accuracy: {accuracy_status} {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"Correct Classifications: {overall_correct}")
    print(f"Total Classifications: {overall_total}")
    print(f"Required Accuracy: 0.900 (90.0%)")
    print(f"Accuracy Gap: {0.900 - overall_accuracy:.3f} ({(0.900 - overall_accuracy)*100:.1f}%)")
    print()
    
    # Identify problem areas
    print("PROBLEM ANALYSIS")
    print("-" * 50)
    
    problem_categories = []
    for category, results in category_results.items():
        if results['accuracy'] < 0.85:
            problem_categories.append((category, results))
    
    if problem_categories:
        print("Categories Below 85% Accuracy Threshold:")
        for category, results in problem_categories:
            print(f"  • {category}: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%) - {results['total'] - results['correct']} errors")
    else:
        print("All categories meet the 85% minimum threshold individually.")
    
    # Show sample classification errors
    print()
    print("SAMPLE CLASSIFICATION ERRORS")
    print("-" * 50)
    
    if detailed_errors:
        error_sample = detailed_errors[:10]  # Show first 10 errors
        for i, error in enumerate(error_sample, 1):
            print(f"{i:2d}. Expected: {error['expected']:20} | Predicted: {error['predicted']:20}")
            print(f"    Confidence: {error['confidence']:.3f} | {error['description']}")
            print(f"    Query: {error['query']}")
            print()
    else:
        print("No classification errors found.")
    
    # Performance vs Accuracy Analysis
    print("PERFORMANCE vs ACCURACY TRADE-OFF ANALYSIS")
    print("-" * 50)
    
    # Test a few queries for timing
    sample_queries = [tc.query for tc in dataset.test_cases[:5] if tc.domain_specific]
    response_times = []
    
    for query in sample_queries:
        start_time = time.perf_counter()
        prediction = router.route_query(query)
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        response_times.append(response_time_ms)
    
    if response_times:
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        print(f"Average Response Time: {avg_response_time:.2f}ms")
        print(f"Max Response Time: {max_response_time:.2f}ms")
        print(f"Performance Requirement: <2000ms (2 seconds)")
        
        performance_status = "✓ PASS" if avg_response_time < 2000 else "✗ FAIL"
        print(f"Performance Status: {performance_status}")
    
    # Recommendations
    print()
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("-" * 50)
    
    recommendations = []
    
    if overall_accuracy < 0.90:
        gap = 0.90 - overall_accuracy
        needed_improvements = int(gap * overall_total)
        recommendations.append(f"Need to improve {needed_improvements} additional classifications to reach 90% accuracy")
    
    if problem_categories:
        for category, results in problem_categories:
            gap = 0.85 - results['accuracy']
            needed = int(gap * results['total']) + 1
            recommendations.append(f"Improve {category} accuracy by fixing {needed} more classifications")
    
    # Check confidence patterns
    low_confidence_errors = [e for e in detailed_errors if e['confidence'] < 0.6]
    if low_confidence_errors:
        recommendations.append(f"Focus on {len(low_confidence_errors)} low-confidence errors (confidence < 0.6)")
    
    high_confidence_errors = [e for e in detailed_errors if e['confidence'] > 0.8]
    if high_confidence_errors:
        recommendations.append(f"Investigate {len(high_confidence_errors)} high-confidence errors (confidence > 0.8) - these suggest systematic issues")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("No specific recommendations - system performance is acceptable.")
    
    print()
    print("CONCLUSION")
    print("-" * 50)
    
    if overall_accuracy >= 0.90:
        print("✓ SUCCESS: The system MEETS the >90% accuracy requirement for CMO-LIGHTRAG-012-T09")
        conclusion = "TASK_COMPLETED"
    else:
        print("✗ FAILURE: The system DOES NOT MEET the >90% accuracy requirement for CMO-LIGHTRAG-012-T09")
        print(f"  Current: {overall_accuracy:.1%} | Required: 90.0% | Gap: {(0.90-overall_accuracy)*100:.1f}%")
        conclusion = "NEEDS_IMPROVEMENT"
    
    print()
    print("=" * 80)
    
    return {
        'overall_accuracy': overall_accuracy,
        'meets_requirement': overall_accuracy >= 0.90,
        'category_results': category_results,
        'total_classifications': overall_total,
        'correct_classifications': overall_correct,
        'detailed_errors': detailed_errors,
        'performance_metrics': {
            'avg_response_time_ms': avg_response_time if response_times else 0,
            'max_response_time_ms': max_response_time if response_times else 0
        },
        'recommendations': recommendations,
        'conclusion': conclusion
    }


if __name__ == "__main__":
    try:
        results = analyze_classification_accuracy()
        
        # Save results to file
        import json
        
        # Create a simplified version for JSON serialization
        json_results = {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_accuracy': results['overall_accuracy'],
            'meets_90_percent_requirement': results['meets_requirement'],
            'total_classifications': results['total_classifications'],
            'correct_classifications': results['correct_classifications'],
            'accuracy_percentage': results['overall_accuracy'] * 100,
            'accuracy_gap_percentage': (0.90 - results['overall_accuracy']) * 100,
            'category_summary': {
                cat: {
                    'accuracy': data['accuracy'],
                    'accuracy_percentage': data['accuracy'] * 100,
                    'correct': data['correct'],
                    'total': data['total'],
                    'avg_confidence': data['avg_confidence']
                } 
                for cat, data in results['category_results'].items()
            },
            'performance_metrics': results['performance_metrics'],
            'error_count': len(results['detailed_errors']),
            'recommendations_count': len(results['recommendations']),
            'conclusion': results['conclusion']
        }
        
        with open('cmo_lightrag_012_t09_accuracy_analysis.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed analysis saved to: cmo_lightrag_012_t09_accuracy_analysis.json")
        
    except Exception as e:
        print(f"ERROR: Analysis failed - {str(e)}")
        import traceback
        traceback.print_exc()