#!/usr/bin/env python3
"""
Debug Classification Failures Script

This script runs a subset of classification test cases to identify specific failures
and root causes in the query classification system.
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Direct imports from current directory
from query_router import BiomedicalQueryRouter, RoutingDecision
from cost_persistence import ResearchCategory
import logging
from unittest.mock import Mock

# Suppress warnings and set up simple logging
logging.basicConfig(level=logging.WARNING)

def run_classification_debug():
    """Run classification debug with specific test cases."""
    
    print("=== Clinical Metabolomics Oracle Classification Debug ===\n")
    
    # Initialize router
    mock_logger = Mock()
    router = BiomedicalQueryRouter(logger=mock_logger)
    
    # Define focused test cases that should be failing based on the reported issues
    test_cases = [
        # General Query Test Cases (0% accuracy reported)
        {
            'query': "What is metabolomics?",
            'expected_category': ResearchCategory.GENERAL_QUERY,
            'expected_routing': RoutingDecision.EITHER,
            'description': "Basic definition query - should be GENERAL_QUERY"
        },
        {
            'query': "Explain the principles of clinical metabolomics",
            'expected_category': ResearchCategory.GENERAL_QUERY,
            'expected_routing': RoutingDecision.EITHER,
            'description': "General explanation query - should be GENERAL_QUERY"
        },
        
        # Literature Search with Temporal Detection (25% accuracy reported)
        {
            'query': "Latest metabolomics research publications in 2024",
            'expected_category': ResearchCategory.LITERATURE_SEARCH,
            'expected_routing': RoutingDecision.PERPLEXITY,
            'description': "Recent literature search with temporal indicator - should be LITERATURE_SEARCH"
        },
        {
            'query': "What are the current trends in clinical metabolomics research?",
            'expected_category': ResearchCategory.LITERATURE_SEARCH,
            'expected_routing': RoutingDecision.PERPLEXITY,
            'description': "Current trends query - should be LITERATURE_SEARCH with PERPLEXITY routing"
        },
        
        # Clinical vs. Biomarker Category Confusion (50% accuracy reported)
        {
            'query': "Discovery of diagnostic biomarkers for cardiovascular disease using untargeted metabolomics",
            'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
            'expected_routing': RoutingDecision.EITHER,
            'description': "Biomarker discovery - should be BIOMARKER_DISCOVERY"
        },
        {
            'query': "Clinical metabolomics for patient diagnosis and treatment monitoring",
            'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
            'expected_routing': RoutingDecision.LIGHTRAG,
            'description': "Clinical application - should be CLINICAL_DIAGNOSIS"
        },
        
        # Metabolite Identification (should be working well)
        {
            'query': "What is the molecular structure of glucose with exact mass 180.0634 using LC-MS?",
            'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
            'expected_routing': RoutingDecision.LIGHTRAG,
            'description': "Specific metabolite identification - should be METABOLITE_IDENTIFICATION"
        },
        
        # Pathway Analysis (should be working well)
        {
            'query': "KEGG pathway enrichment analysis for diabetes metabolomics study",
            'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
            'expected_routing': RoutingDecision.LIGHTRAG,
            'description': "Pathway analysis - should be PATHWAY_ANALYSIS"
        }
    ]
    
    print("Testing specific problematic query classifications:\n")
    
    failures = []
    successes = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected_category'].value} -> {test_case['expected_routing'].value}")
        
        try:
            # Get prediction from router
            prediction = router.route_query(test_case['query'])
            
            print(f"Predicted: {prediction.research_category.value} -> {prediction.routing_decision.value}")
            print(f"Confidence: {prediction.confidence:.3f}")
            print(f"Reasoning: {prediction.reasoning}")
            
            # Check if correct
            category_correct = prediction.research_category == test_case['expected_category']
            routing_correct = prediction.routing_decision == test_case['expected_routing'] or test_case['expected_routing'] == RoutingDecision.EITHER
            
            overall_correct = category_correct and routing_correct
            
            status = "✓ PASS" if overall_correct else "✗ FAIL"
            print(f"Result: {status}")
            
            if overall_correct:
                successes.append(test_case)
            else:
                failures.append({
                    'test_case': test_case,
                    'prediction': prediction,
                    'category_correct': category_correct,
                    'routing_correct': routing_correct
                })
            
            # Show detailed metrics for failures
            if not overall_correct:
                print(f"Category Match: {'✓' if category_correct else '✗'}")
                print(f"Routing Match: {'✓' if routing_correct else '✗'}")
                print("Confidence Metrics:")
                print(f"  Overall: {prediction.confidence_metrics.overall_confidence:.3f}")
                print(f"  Category: {prediction.confidence_metrics.research_category_confidence:.3f}")
                print(f"  Temporal: {prediction.confidence_metrics.temporal_analysis_confidence:.3f}")
                print(f"  Signal Strength: {prediction.confidence_metrics.signal_strength_confidence:.3f}")
                print(f"  Keyword Density: {prediction.confidence_metrics.keyword_density:.3f}")
                print(f"  Pattern Match: {prediction.confidence_metrics.pattern_match_strength:.3f}")
                print(f"  Biomedical Entities: {prediction.confidence_metrics.biomedical_entity_count}")
        
        except Exception as e:
            print(f"Result: ✗ ERROR - {str(e)}")
            failures.append({
                'test_case': test_case,
                'error': str(e)
            })
        
        print("-" * 80)
        print()
    
    # Summary
    total_tests = len(test_cases)
    success_count = len(successes)
    failure_count = len(failures)
    accuracy = success_count / total_tests if total_tests > 0 else 0
    
    print(f"=== SUMMARY ===")
    print(f"Total Tests: {total_tests}")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print()
    
    # Detailed failure analysis
    if failures:
        print("=== DETAILED FAILURE ANALYSIS ===")
        
        category_failures = {}
        routing_failures = {}
        
        for failure in failures:
            if 'error' in failure:
                print(f"ERROR: {failure['test_case']['description']} - {failure['error']}")
                continue
            
            test_case = failure['test_case']
            prediction = failure['prediction']
            
            # Track category failures
            expected_cat = test_case['expected_category'].value
            predicted_cat = prediction.research_category.value
            
            if expected_cat not in category_failures:
                category_failures[expected_cat] = []
            category_failures[expected_cat].append({
                'query': test_case['query'][:60] + "...",
                'predicted': predicted_cat,
                'confidence': prediction.confidence
            })
        
        print("\nCategory Misclassifications:")
        for expected_cat, failures_list in category_failures.items():
            print(f"\n{expected_cat}:")
            for fail in failures_list:
                print(f"  Expected {expected_cat} -> Got {fail['predicted']} ({fail['confidence']:.3f})")
                print(f"    Query: {fail['query']}")
    
    print("\n=== ROOT CAUSE ANALYSIS ===")
    
    # Analyze patterns in failures
    print("\n1. GENERAL QUERY CLASSIFICATION ISSUES:")
    general_tests = [tc for tc in test_cases if tc['expected_category'] == ResearchCategory.GENERAL_QUERY]
    if general_tests:
        for tc in general_tests:
            pred = router.route_query(tc['query'])
            print(f"   Query: '{tc['query']}'")
            print(f"   Classified as: {pred.research_category.value} (should be GENERAL_QUERY)")
            print(f"   Evidence found: {pred.confidence_metrics.biomedical_entity_count} biomedical entities")
            print(f"   Keyword density: {pred.confidence_metrics.keyword_density:.3f}")
            print()
    
    print("\n2. TEMPORAL DETECTION ISSUES:")
    temporal_tests = [tc for tc in test_cases if tc['expected_category'] == ResearchCategory.LITERATURE_SEARCH]
    if temporal_tests:
        for tc in temporal_tests:
            pred = router.route_query(tc['query'])
            print(f"   Query: '{tc['query']}'")
            print(f"   Temporal confidence: {pred.confidence_metrics.temporal_analysis_confidence:.3f}")
            print(f"   Classified as: {pred.research_category.value} -> {pred.routing_decision.value}")
            print(f"   Should be: LITERATURE_SEARCH -> PERPLEXITY")
            print()
    
    print("\n3. CLINICAL vs BIOMARKER CONFUSION:")
    clinical_biomarker_tests = [tc for tc in test_cases if tc['expected_category'] in [ResearchCategory.CLINICAL_DIAGNOSIS, ResearchCategory.BIOMARKER_DISCOVERY]]
    for tc in clinical_biomarker_tests:
        pred = router.route_query(tc['query'])
        print(f"   Query: '{tc['query'][:60]}...'")
        print(f"   Expected: {tc['expected_category'].value}")
        print(f"   Got: {pred.research_category.value}")
        print(f"   Confidence: {pred.confidence:.3f}")
        print()
    
    return accuracy, failures

if __name__ == "__main__":
    accuracy, failures = run_classification_debug()
    
    print("\n=== RECOMMENDED FIXES ===")
    print("1. Fix GENERAL_QUERY classification by improving scoring for basic questions")
    print("2. Enhance temporal detection for LITERATURE_SEARCH routing to PERPLEXITY")
    print("3. Improve distinction between CLINICAL_DIAGNOSIS and BIOMARKER_DISCOVERY")
    print("4. Review keyword matching logic for over-reliance on biomedical terms")
    print("5. Implement contextual analysis to reduce false positives")