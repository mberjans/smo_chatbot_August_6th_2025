#!/usr/bin/env python3
"""
Debug script to test routing accuracy and identify the core issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Test data from the test files
test_queries = {
    'lightrag': [
        "What is the relationship between glucose and insulin in diabetes?",
        "How does the glycolysis pathway interact with lipid metabolism?", 
        "What biomarkers are associated with metabolic syndrome?",
        "Mechanism of action of metformin in glucose homeostasis",
        "Metabolomic signature associated with cardiovascular disease"
    ],
    'perplexity': [
        "Latest metabolomics research 2025",
        "Current advances in LC-MS technology",
        "Recent biomarker discoveries for cancer detection", 
        "Breaking news in mass spectrometry 2024",
        "Today's advances in metabolomic analysis"
    ],
    'either': [
        "What is metabolomics?",
        "Define biomarker",
        "How does LC-MS work?",
        "Explain mass spectrometry",
        "What are metabolites?"
    ],
    'hybrid': [
        "What are the latest biomarker discoveries and how do they relate to metabolic pathways?",
        "Current LC-MS approaches for understanding insulin signaling mechanisms"
    ]
}

def test_routing_accuracy():
    """Test routing accuracy for each category"""
    router = BiomedicalQueryRouter()
    
    results = {}
    
    for expected_routing, queries in test_queries.items():
        correct = 0
        total = len(queries)
        
        print(f"\n=== Testing {expected_routing.upper()} queries ===")
        
        for query in queries:
            prediction = router.route_query(query)
            actual_routing = prediction.routing_decision.value
            
            # Check if routing matches expectation
            is_correct = False
            if expected_routing == 'lightrag' and actual_routing == 'lightrag':
                is_correct = True
            elif expected_routing == 'perplexity' and actual_routing == 'perplexity':
                is_correct = True
            elif expected_routing == 'either' and actual_routing in ['either', 'lightrag', 'perplexity']:
                is_correct = True  # EITHER is flexible
            elif expected_routing == 'hybrid' and actual_routing == 'hybrid':
                is_correct = True
                
            if is_correct:
                correct += 1
                
            print(f"Query: {query[:60]}...")
            print(f"Expected: {expected_routing} | Actual: {actual_routing} | Confidence: {prediction.confidence:.3f} | {'✓' if is_correct else '✗'}")
            print(f"Reasoning: {prediction.reasoning[:2]}")
            print()
        
        accuracy = (correct / total) * 100
        results[expected_routing] = accuracy
        print(f"{expected_routing.upper()} Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Overall accuracy
    total_correct = sum(int(accuracy/100 * len(test_queries[category])) for category, accuracy in results.items())
    total_queries = sum(len(queries) for queries in test_queries.values())
    overall_accuracy = (total_correct / total_queries) * 100
    
    print(f"\n=== OVERALL RESULTS ===")
    for category, accuracy in results.items():
        print(f"{category.upper()}: {accuracy:.1f}%")
    print(f"OVERALL: {overall_accuracy:.1f}%")
    
    return results

if __name__ == "__main__":
    test_routing_accuracy()