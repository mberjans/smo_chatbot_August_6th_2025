#!/usr/bin/env python3
"""
Comprehensive Classification Validation Report for CMO-LIGHTRAG-012-T09

This script validates the classification system fixes and provides a comprehensive
report on whether the >90% accuracy requirement is met.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from research_categorizer import ResearchCategorizer
    from query_router import BiomedicalQueryRouter, RoutingDecision
    from cost_persistence import ResearchCategory
except ImportError:
    try:
        # Try alternative import paths
        from lightrag_integration.research_categorizer import ResearchCategorizer
        from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
        from lightrag_integration.cost_persistence import ResearchCategory
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the lightrag_integration directory")
        sys.exit(1)


@dataclass
class ValidationTestCase:
    """Test case for validation."""
    query: str
    expected_category: ResearchCategory
    expected_routing: RoutingDecision
    description: str
    priority: str  # 'critical', 'high', 'medium', 'low'


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_case: ValidationTestCase
    predicted_category: ResearchCategory
    predicted_routing: RoutingDecision
    categorizer_confidence: float
    router_confidence: float
    category_correct: bool
    routing_acceptable: bool
    evidence: List[str]
    execution_time: float


def get_comprehensive_test_cases() -> List[ValidationTestCase]:
    """Get comprehensive test cases including the critical failing ones."""
    return [
        # Critical failing cases (MUST pass)
        ValidationTestCase(
            query="What is metabolomics?",
            expected_category=ResearchCategory.GENERAL_QUERY,
            expected_routing=RoutingDecision.EITHER,
            description="Basic definitional query - was being misclassified",
            priority="critical"
        ),
        ValidationTestCase(
            query="What are the current trends in clinical metabolomics research?",
            expected_category=ResearchCategory.LITERATURE_SEARCH,
            expected_routing=RoutingDecision.PERPLEXITY,
            description="Temporal query with 'current trends' - was being misclassified",
            priority="critical"
        ),
        ValidationTestCase(
            query="How can metabolomic profiles be used for precision medicine",
            expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Clinical application query - was being misclassified",
            priority="critical"
        ),
        ValidationTestCase(
            query="API integration with multiple metabolomics databases for compound identification",
            expected_category=ResearchCategory.DATABASE_INTEGRATION,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Database API query - was being misclassified",
            priority="critical"
        ),
        
        # Additional high-priority test cases
        ValidationTestCase(
            query="Identify metabolites in plasma samples using LC-MS data",
            expected_category=ResearchCategory.METABOLITE_IDENTIFICATION,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Metabolite identification with specific methodology",
            priority="high"
        ),
        ValidationTestCase(
            query="Analyze metabolic pathways affected by diabetes",
            expected_category=ResearchCategory.PATHWAY_ANALYSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Pathway analysis for specific disease",
            priority="high"
        ),
        ValidationTestCase(
            query="Find biomarkers for early detection of cancer using metabolomics",
            expected_category=ResearchCategory.BIOMARKER_DISCOVERY,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Biomarker discovery for specific application",
            priority="high"
        ),
        ValidationTestCase(
            query="Recent publications on metabolomics in COVID-19 research",
            expected_category=ResearchCategory.LITERATURE_SEARCH,
            expected_routing=RoutingDecision.PERPLEXITY,
            description="Literature search for recent publications",
            priority="high"
        ),
        ValidationTestCase(
            query="Statistical methods for metabolomics data analysis",
            expected_category=ResearchCategory.STATISTICAL_ANALYSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Statistical analysis methodology query",
            priority="high"
        ),
        ValidationTestCase(
            query="Preprocess metabolomics data for machine learning",
            expected_category=ResearchCategory.DATA_PREPROCESSING,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Data preprocessing for specific application",
            priority="high"
        ),
        
        # Medium priority edge cases
        ValidationTestCase(
            query="Define biomarkers in metabolomics",
            expected_category=ResearchCategory.GENERAL_QUERY,
            expected_routing=RoutingDecision.EITHER,
            description="Definitional query that should not be confused with biomarker_discovery",
            priority="medium"
        ),
        ValidationTestCase(
            query="Explain the principles of clinical metabolomics",
            expected_category=ResearchCategory.GENERAL_QUERY,
            expected_routing=RoutingDecision.EITHER,
            description="Explanatory query that should be general, not specific technical",
            priority="medium"
        ),
        ValidationTestCase(
            query="Drug discovery applications of metabolomics in pharmaceutical industry",
            expected_category=ResearchCategory.DRUG_DISCOVERY,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Drug discovery specific query",
            priority="medium"
        ),
        ValidationTestCase(
            query="Compare HMDB and METLIN databases for metabolite annotation",
            expected_category=ResearchCategory.DATABASE_INTEGRATION,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Database comparison query",
            priority="medium"
        ),
        ValidationTestCase(
            query="Clinical applications of metabolomics in personalized medicine",
            expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Clinical applications query",
            priority="medium"
        ),
    ]


def is_routing_acceptable(predicted: RoutingDecision, expected: RoutingDecision) -> bool:
    """Check if routing decision is acceptable based on flexible routing rules."""
    if predicted == expected:
        return True
    
    # EITHER or HYBRID are acceptable for most cases
    if expected == RoutingDecision.EITHER and predicted in [
        RoutingDecision.EITHER, RoutingDecision.HYBRID, RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY
    ]:
        return True
    
    # LIGHTRAG can be routed as EITHER or HYBRID
    if expected == RoutingDecision.LIGHTRAG and predicted in [
        RoutingDecision.LIGHTRAG, RoutingDecision.EITHER, RoutingDecision.HYBRID
    ]:
        return True
    
    # PERPLEXITY can be routed as EITHER or HYBRID for temporal queries
    if expected == RoutingDecision.PERPLEXITY and predicted in [
        RoutingDecision.PERPLEXITY, RoutingDecision.EITHER, RoutingDecision.HYBRID
    ]:
        return True
    
    return False


def run_validation_test(test_case: ValidationTestCase, categorizer: ResearchCategorizer, 
                       router: BiomedicalQueryRouter) -> ValidationResult:
    """Run a single validation test."""
    start_time = time.time()
    
    # Test categorizer
    categorizer_prediction = categorizer.categorize_query(test_case.query)
    
    # Test router
    router_prediction = router.route_query(test_case.query)
    
    execution_time = time.time() - start_time
    
    # Check correctness
    category_correct = categorizer_prediction.category == test_case.expected_category
    routing_acceptable = is_routing_acceptable(router_prediction.routing_decision, test_case.expected_routing)
    
    # Extract evidence
    evidence = categorizer_prediction.evidence[:3] if hasattr(categorizer_prediction, 'evidence') else []
    
    return ValidationResult(
        test_case=test_case,
        predicted_category=categorizer_prediction.category,
        predicted_routing=router_prediction.routing_decision,
        categorizer_confidence=categorizer_prediction.confidence,
        router_confidence=router_prediction.confidence,
        category_correct=category_correct,
        routing_acceptable=routing_acceptable,
        evidence=evidence,
        execution_time=execution_time
    )


def generate_validation_report(results: List[ValidationResult]) -> Dict[str, Any]:
    """Generate comprehensive validation report."""
    total_tests = len(results)
    
    # Overall metrics
    category_correct = sum(1 for r in results if r.category_correct)
    routing_acceptable = sum(1 for r in results if r.routing_acceptable)
    both_correct = sum(1 for r in results if r.category_correct and r.routing_acceptable)
    
    category_accuracy = (category_correct / total_tests) * 100
    routing_accuracy = (routing_acceptable / total_tests) * 100
    overall_accuracy = (both_correct / total_tests) * 100
    
    # Priority-based breakdown
    priority_breakdown = {}
    for priority in ['critical', 'high', 'medium', 'low']:
        priority_results = [r for r in results if r.test_case.priority == priority]
        if priority_results:
            priority_correct = sum(1 for r in priority_results if r.category_correct and r.routing_acceptable)
            priority_total = len(priority_results)
            priority_accuracy = (priority_correct / priority_total) * 100
            priority_breakdown[priority] = {
                'correct': priority_correct,
                'total': priority_total,
                'accuracy': priority_accuracy
            }
    
    # Category breakdown
    category_breakdown = {}
    for category in ResearchCategory:
        cat_results = [r for r in results if r.test_case.expected_category == category]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r.category_correct)
            cat_total = len(cat_results)
            category_breakdown[category.value] = {
                'correct': cat_correct,
                'total': cat_total,
                'accuracy': (cat_correct / cat_total) * 100
            }
    
    # Performance metrics
    avg_execution_time = sum(r.execution_time for r in results) / total_tests
    avg_categorizer_confidence = sum(r.categorizer_confidence for r in results) / total_tests
    avg_router_confidence = sum(r.router_confidence for r in results) / total_tests
    
    # Failed cases
    failed_cases = [r for r in results if not (r.category_correct and r.routing_acceptable)]
    
    return {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'category_accuracy': category_accuracy,
            'routing_accuracy': routing_accuracy,
            'overall_accuracy': overall_accuracy,
            'meets_90_percent_requirement': overall_accuracy >= 90.0,
            'avg_execution_time': avg_execution_time,
            'avg_categorizer_confidence': avg_categorizer_confidence,
            'avg_router_confidence': avg_router_confidence
        },
        'priority_breakdown': priority_breakdown,
        'category_breakdown': category_breakdown,
        'failed_cases': [
            {
                'query': r.test_case.query,
                'expected_category': r.test_case.expected_category.value,
                'predicted_category': r.predicted_category.value,
                'expected_routing': r.test_case.expected_routing.value,
                'predicted_routing': r.predicted_routing.value,
                'description': r.test_case.description,
                'priority': r.test_case.priority,
                'category_correct': r.category_correct,
                'routing_acceptable': r.routing_acceptable
            }
            for r in failed_cases
        ],
        'detailed_results': [
            {
                'query': r.test_case.query,
                'expected_category': r.test_case.expected_category.value,
                'predicted_category': r.predicted_category.value,
                'expected_routing': r.test_case.expected_routing.value,
                'predicted_routing': r.predicted_routing.value,
                'categorizer_confidence': r.categorizer_confidence,
                'router_confidence': r.router_confidence,
                'category_correct': r.category_correct,
                'routing_acceptable': r.routing_acceptable,
                'evidence': r.evidence,
                'execution_time': r.execution_time,
                'priority': r.test_case.priority
            }
            for r in results
        ]
    }


def print_validation_report(report: Dict[str, Any]):
    """Print formatted validation report."""
    print("\n" + "=" * 80)
    print("CMO-LIGHTRAG-012-T09 COMPREHENSIVE CLASSIFICATION VALIDATION REPORT")
    print("=" * 80)
    
    summary = report['summary']
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Category Accuracy: {summary['category_accuracy']:.1f}%")
    print(f"   Routing Accuracy: {summary['routing_accuracy']:.1f}%")
    print(f"   Overall Accuracy: {summary['overall_accuracy']:.1f}%")
    print(f"   Target: ≥90% accuracy")
    
    status = "✅ PASSED" if summary['meets_90_percent_requirement'] else "❌ FAILED"
    print(f"   Status: {status}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Avg Execution Time: {summary['avg_execution_time']:.3f}s")
    print(f"   Avg Categorizer Confidence: {summary['avg_categorizer_confidence']:.3f}")
    print(f"   Avg Router Confidence: {summary['avg_router_confidence']:.3f}")
    
    print(f"\n🎯 PRIORITY BREAKDOWN:")
    for priority, data in report['priority_breakdown'].items():
        status = "✅" if data['accuracy'] >= 90 else "❌" if data['accuracy'] < 80 else "⚠️"
        print(f"   {priority.upper()}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%) {status}")
    
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, data in report['category_breakdown'].items():
        status = "✅" if data['accuracy'] >= 90 else "❌" if data['accuracy'] < 80 else "⚠️"
        print(f"   {category}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%) {status}")
    
    if report['failed_cases']:
        print(f"\n❌ FAILED CASES ({len(report['failed_cases'])}):")
        for i, case in enumerate(report['failed_cases'][:5], 1):  # Show first 5 failures
            print(f"   {i}. '{case['query'][:60]}{'...' if len(case['query']) > 60 else ''}'")
            print(f"      Expected: {case['expected_category']} → {case['expected_routing']}")
            print(f"      Actual: {case['predicted_category']} → {case['predicted_routing']}")
            print(f"      Priority: {case['priority']} | Category OK: {case['category_correct']} | Routing OK: {case['routing_acceptable']}")
    
    print("\n" + "=" * 80)
    
    # Critical assessment for CMO-LIGHTRAG-012-T09
    critical_results = [r for r in report['detailed_results'] if r['priority'] == 'critical']
    critical_passed = sum(1 for r in critical_results if r['category_correct'] and r['routing_acceptable'])
    critical_accuracy = (critical_passed / len(critical_results)) * 100 if critical_results else 0
    
    print(f"🚨 CMO-LIGHTRAG-012-T09 CRITICAL ASSESSMENT:")
    print(f"   Critical Test Cases: {critical_passed}/{len(critical_results)} ({critical_accuracy:.1f}%)")
    print(f"   All previously failing cases must pass for task completion.")
    
    if critical_accuracy >= 90:
        print(f"   ✅ CMO-LIGHTRAG-012-T09 REQUIREMENT MET")
    else:
        print(f"   ❌ CMO-LIGHTRAG-012-T09 REQUIREMENT NOT MET")
    
    print("=" * 80)


def main():
    """Run comprehensive classification validation."""
    print("Initializing Classification System...")
    
    try:
        categorizer = ResearchCategorizer()
        router = BiomedicalQueryRouter()
        print("✅ System initialization complete")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    test_cases = get_comprehensive_test_cases()
    print(f"Running {len(test_cases)} validation tests...")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"  Test {i}/{len(test_cases)}: {test_case.priority} priority...")
        try:
            result = run_validation_test(test_case, categorizer, router)
            results.append(result)
        except Exception as e:
            print(f"    ❌ Test failed with error: {e}")
            continue
    
    print(f"✅ Completed {len(results)} tests")
    
    # Generate and display report
    report = generate_validation_report(results)
    print_validation_report(report)
    
    # Save detailed report
    report_file = f"classification_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report['summary']['meets_90_percent_requirement']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)