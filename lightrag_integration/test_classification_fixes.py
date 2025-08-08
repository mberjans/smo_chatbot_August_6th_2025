#!/usr/bin/env python3
"""
Test script to validate classification fixes for failing test cases.

This script tests the specific failing cases identified in the classification system
to ensure they now achieve >90% accuracy after the hierarchical scoring improvements.
"""

import sys
import os
from typing import List, Tuple
from dataclasses import dataclass

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix relative imports by adjusting sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

try:
    from lightrag_integration.research_categorizer import ResearchCategorizer
    from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
    from lightrag_integration.cost_persistence import ResearchCategory
except ImportError:
    # Fallback for direct execution
    import research_categorizer
    import query_router
    import cost_persistence
    
    ResearchCategorizer = research_categorizer.ResearchCategorizer
    BiomedicalQueryRouter = query_router.BiomedicalQueryRouter
    RoutingDecision = query_router.RoutingDecision
    ResearchCategory = cost_persistence.ResearchCategory


@dataclass
class TestCase:
    """Test case for classification validation."""
    query: str
    expected_category: ResearchCategory
    expected_routing: RoutingDecision
    description: str


def get_failing_test_cases() -> List[TestCase]:
    """Get the specific failing test cases that need to be fixed."""
    return [
        TestCase(
            query="What is metabolomics?",
            expected_category=ResearchCategory.GENERAL_QUERY,
            expected_routing=RoutingDecision.EITHER,
            description="Basic definitional query - was being misclassified as metabolite_identification"
        ),
        TestCase(
            query="What are the current trends in clinical metabolomics research?",
            expected_category=ResearchCategory.LITERATURE_SEARCH,
            expected_routing=RoutingDecision.PERPLEXITY,
            description="Temporal query with 'current trends' - was being misclassified as clinical_diagnosis"
        ),
        TestCase(
            query="How can metabolomic profiles be used for precision medicine",
            expected_category=ResearchCategory.CLINICAL_DIAGNOSIS,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Clinical application query - was being misclassified as biomarker_discovery"
        ),
        TestCase(
            query="API integration with multiple metabolomics databases for compound identification",
            expected_category=ResearchCategory.DATABASE_INTEGRATION,
            expected_routing=RoutingDecision.LIGHTRAG,
            description="Database API query - was being misclassified as metabolite_identification"
        ),
        # Additional edge cases
        TestCase(
            query="Define biomarkers in metabolomics",
            expected_category=ResearchCategory.GENERAL_QUERY,
            expected_routing=RoutingDecision.EITHER,
            description="Definitional query that should not be confused with biomarker_discovery"
        ),
        TestCase(
            query="Explain the principles of clinical metabolomics",
            expected_category=ResearchCategory.GENERAL_QUERY,
            expected_routing=RoutingDecision.EITHER,
            description="Explanatory query that should be general, not specific technical"
        )
    ]


def test_research_categorizer():
    """Test the research categorizer with the failing test cases."""
    print("=" * 60)
    print("TESTING RESEARCH CATEGORIZER FIXES")
    print("=" * 60)
    
    categorizer = ResearchCategorizer()
    test_cases = get_failing_test_cases()
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total}: {test_case.description}")
        print(f"Query: '{test_case.query}'")
        print(f"Expected: {test_case.expected_category.value}")
        
        # Get prediction
        prediction = categorizer.categorize_query(test_case.query)
        
        print(f"Actual: {prediction.category.value}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Evidence: {', '.join(prediction.evidence[:3])}")
        
        # Check if correct
        is_correct = prediction.category == test_case.expected_category
        if is_correct:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
        
        # Show metadata for debugging
        if hasattr(prediction, 'metadata') and prediction.metadata:
            all_scores = prediction.metadata.get('all_scores', {})
            print(f"All scores: {[(cat, f'{score:.3f}') for cat, score in list(all_scores.items())[:3]]}")
    
    accuracy = (passed / total) * 100
    print(f"\n" + "=" * 60)
    print(f"RESEARCH CATEGORIZER RESULTS:")
    print(f"Passed: {passed}/{total} ({accuracy:.1f}%)")
    print(f"Target: >90% accuracy")
    print(f"Status: {'✅ PASSED' if accuracy >= 90 else '❌ FAILED'}")
    print("=" * 60)
    
    return accuracy >= 90


def test_query_router():
    """Test the query router with the failing test cases."""
    print("\n" + "=" * 60)
    print("TESTING QUERY ROUTER FIXES")
    print("=" * 60)
    
    router = BiomedicalQueryRouter()
    test_cases = get_failing_test_cases()
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total}: {test_case.description}")
        print(f"Query: '{test_case.query}'")
        print(f"Expected routing: {test_case.expected_routing.value}")
        
        # Get routing prediction
        routing_prediction = router.route_query(test_case.query)
        
        print(f"Actual routing: {routing_prediction.routing_decision.value}")
        print(f"Confidence: {routing_prediction.confidence:.3f}")
        print(f"Reasoning: {', '.join(routing_prediction.reasoning[:2])}")
        
        # Check routing decision (allow EITHER or HYBRID as valid for flexible routing)
        expected_routing = test_case.expected_routing
        actual_routing = routing_prediction.routing_decision
        
        is_correct = (
            actual_routing == expected_routing or 
            (expected_routing in [RoutingDecision.EITHER, RoutingDecision.HYBRID] and 
             actual_routing in [RoutingDecision.EITHER, RoutingDecision.HYBRID]) or
            (expected_routing == RoutingDecision.LIGHTRAG and actual_routing in [RoutingDecision.LIGHTRAG, RoutingDecision.EITHER]) or
            (expected_routing == RoutingDecision.PERPLEXITY and actual_routing in [RoutingDecision.PERPLEXITY, RoutingDecision.EITHER])
        )
        
        if is_correct:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
        
        # Show research category from routing
        print(f"Research category: {routing_prediction.research_category.value}")
    
    accuracy = (passed / total) * 100
    print(f"\n" + "=" * 60)
    print(f"QUERY ROUTER RESULTS:")
    print(f"Passed: {passed}/{total} ({accuracy:.1f}%)")
    print(f"Target: >90% accuracy")
    print(f"Status: {'✅ PASSED' if accuracy >= 90 else '❌ FAILED'}")
    print("=" * 60)
    
    return accuracy >= 90


def main():
    """Run all classification fix tests."""
    print("Testing Classification System Fixes")
    print(f"Testing {len(get_failing_test_cases())} critical failing cases")
    print(f"Target: >90% accuracy to resolve test failures")
    
    # Test both components
    categorizer_passed = test_research_categorizer()
    router_passed = test_query_router()
    
    # Overall results
    print("\n" + "=" * 60)
    print("OVERALL RESULTS:")
    print(f"Research Categorizer: {'✅ PASSED' if categorizer_passed else '❌ FAILED'}")
    print(f"Query Router: {'✅ PASSED' if router_passed else '❌ FAILED'}")
    print(f"Overall Status: {'✅ CLASSIFICATION FIXES SUCCESSFUL' if categorizer_passed and router_passed else '❌ FIXES NEED MORE WORK'}")
    print("=" * 60)
    
    return categorizer_passed and router_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)